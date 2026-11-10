{{ config(materialized='table') }}
-- Grain: 1 row per loan_id.
--
-- DEFAULT DEFINITION (shared with the ML training label - keep identical):
--   A loan is in default ("bad 12m") when ANY installment whose due_date falls within the first
--   12 months after disbursement (due_date <= disbursed_date + 12 months) reaches 90+ days past due.
--   Delinquency is computed from due_date and paid_at against the global observation cutoff
--   (dim_snapshot.as_of_date, contract default 2026-10-31). The per-batch silver column
--   days_past_due is NEVER used for the label (it is a snapshot as of the batch that last
--   loaded the row, and installments reappear in later batches).
--     * fully paid (amount_paid >= amount_due and paid_at not null): paid_date - due_date (days late);
--     * not fully paid (unpaid or partial) and already due: as_of_date - due_date;
--     * not yet due: 0.
--   The default event is the FIRST such installment (lowest installment_number);
--   default_date = its due_date + 90 days.
-- OBSERVABILITY: the label is only final once as_of_date >= disbursed_date + 12 months + 90 days.
--   Before that the loan is 'immature': is_default_12m is NULL and the loan is excluded from every
--   default-rate denominator (immature loans are never counted as good).
--   Delinquency that starts on installments due after month 12 is not a 12m default.
with snap as (
    select as_of_date from {{ ref('dim_snapshot') }}
),
loans as (
    select
        l.*,
        cast(l.disbursed_at as date)                                   as disbursed_date,
        cast(cast(l.disbursed_at as date) + interval 12 month as date) as window_end_date,
        cast(cast(l.disbursed_at as date) + interval 12 month + interval 90 day as date) as observable_from_date
    from {{ ref('stg_loans') }} l
),
inst as (
    select
        i.* replace (cast(i.due_date as date) as due_date),
        lo.window_end_date,
        (i.paid_at is not null and i.amount_paid >= i.amount_due)       as is_fully_paid,
        case
            when i.paid_at is not null and i.amount_paid >= i.amount_due
                then greatest(0, date_diff('day', cast(i.due_date as date), cast(i.paid_at as date)))
            -- not fully paid: DPD vs the global cutoff, never vs per-batch days_past_due
            when cast(i.due_date as date) < snap.as_of_date
                then date_diff('day', cast(i.due_date as date), snap.as_of_date)
            else 0
        end                                                            as installment_dpd,
        cast(i.due_date as date) <= lo.window_end_date                 as is_due_in_window
    from {{ ref('stg_installments') }} i
    join loans lo on lo.loan_id = i.loan_id
    cross join snap
),
inst_agg as (
    select
        inst.loan_id,
        count(*)                                                       as n_installments,
        count(*) filter (where due_date <= snap.as_of_date)            as n_installments_due,
        count(*) filter (where is_fully_paid)                          as n_installments_fully_paid,
        count(*) filter (where is_fully_paid and installment_dpd > 0)  as n_installments_paid_late,
        count(*) filter (where paid_at is not null and not is_fully_paid) as n_installments_partially_paid,
        max(installment_dpd)                                           as max_dpd_observed,
        max(installment_dpd) filter (where is_due_in_window)           as max_dpd_in_window,
        min(installment_number) filter (where is_due_in_window and installment_dpd >= 90) as default_installment_number,
        min(due_date)           filter (where is_due_in_window and installment_dpd >= 90) as default_installment_due_date,
        sum(amount_due) filter (where due_date <= snap.as_of_date)     as amount_due_to_date,
        coalesce(sum(amount_paid), 0)                                  as amount_paid_total
    from inst
    cross join snap
    group by inst.loan_id
),
joined as (
    select
        lo.loan_id,
        lo.application_id,
        app.applicant_id,
        lo.disbursed_at,
        lo.disbursed_date,
        cast(date_trunc('month', lo.disbursed_at) as date)             as vintage_month,
        lo.principal,
        lo.interest_rate_monthly,
        lo.term_months,
        lo.installment_amount,
        lo.window_end_date,
        lo.observable_from_date,
        snap.as_of_date,
        date_diff('month', lo.disbursed_date, snap.as_of_date)         as months_on_book,
        snap.as_of_date >= lo.observable_from_date                     as is_observable,
        coalesce(ia.n_installments, 0)                                 as n_installments,
        coalesce(ia.n_installments_due, 0)                             as n_installments_due,
        coalesce(ia.n_installments_fully_paid, 0)                      as n_installments_fully_paid,
        coalesce(ia.n_installments_paid_late, 0)                       as n_installments_paid_late,
        coalesce(ia.n_installments_partially_paid, 0)                  as n_installments_partially_paid,
        coalesce(ia.max_dpd_observed, 0)                               as max_dpd_observed,
        coalesce(ia.max_dpd_in_window, 0)                              as max_dpd_in_window,
        ia.default_installment_number,
        ia.default_installment_due_date,
        cast(ia.default_installment_due_date + interval 90 day as date) as default_date,
        coalesce(ia.amount_due_to_date, 0)                             as amount_due_to_date,
        coalesce(ia.amount_paid_total, 0)                              as amount_paid_total,
        lo.batch_id,
        lo.source_file,
        lo.ingested_at
    from loans lo
    cross join snap
    left join inst_agg ia on ia.loan_id = lo.loan_id
    left join {{ ref('stg_applications') }} app on app.application_id = lo.application_id
)
select
    j.* exclude (batch_id, source_file, ingested_at),
    j.default_installment_number is not null                           as has_default_event_to_date,
    case
        when j.is_observable and j.default_installment_number is not null then 'default'
        when j.is_observable then 'good'
        else 'immature'
    end                                                                as outcome_status,
    case
        when not j.is_observable then null
        when j.default_installment_number is not null then 1
        else 0
    end                                                                as is_default_12m,
    -- Exposure at default: scheduled outstanding principal immediately before the defaulting
    -- installment, French/Price amortization: P(1+r)^(k-1) - A[((1+r)^(k-1) - 1)/r].
    case when j.default_installment_number is not null then
        cast(least(j.principal, greatest(0,
            case when j.interest_rate_monthly = 0
                 then j.principal - j.installment_amount * (j.default_installment_number - 1)
                 else j.principal * power(1 + j.interest_rate_monthly, j.default_installment_number - 1)
                      - j.installment_amount * (power(1 + j.interest_rate_monthly, j.default_installment_number - 1) - 1)
                        / j.interest_rate_monthly
            end)) as decimal(18,2))
    end                                                                as exposure_at_default,
    j.batch_id,
    j.source_file,
    j.ingested_at
from joined j
