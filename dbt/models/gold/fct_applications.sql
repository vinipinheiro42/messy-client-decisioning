{{ config(materialized='table') }}
-- Grain: 1 row per application_id (stg_applications is already deduplicated by the silver layer).
{% set _app_cols = adapter.get_columns_in_relation(ref('stg_applications'))
                   | map(attribute='name') | map('lower') | list %}
with app as (
    select * from {{ ref('stg_applications') }}
),
loan as (
    select application_id, loan_id, principal, installment_amount, disbursed_at
    from {{ ref('stg_loans') }}
),
applicant as (
    select applicant_id, monthly_income, birth_date from {{ ref('stg_applicants') }}
)
select
    app.application_id,
    app.applicant_id,
    app.applied_at,
    cast(date_trunc('month', app.applied_at) as date)                 as application_month,
    app.requested_amount,
    app.term_months,
    app.purpose,
    app.channel,
    app.bureau_score,
    app.existing_debt,
    -- Prefer the silver flag when present; otherwise derive from existing_debt IS NULL
    -- (upcoming silver change: missing debt is NULL + existing_debt_is_missing, not 0).
    {% if 'existing_debt_is_missing' in _app_cols %}
    coalesce(app.existing_debt_is_missing, app.existing_debt is null) as existing_debt_is_missing,
    {% else %}
    (app.existing_debt is null)                                       as existing_debt_is_missing,
    {% endif %}
    app.decision,
    app.decided_at,
    app.decision = 'approved'                                          as is_approved,
    app.decision = 'declined'                                          as is_declined,
    app.decision = 'cancelled'                                         as is_cancelled,
    loan.loan_id is not null                                           as has_loan,
    loan.loan_id,
    loan.disbursed_at,
    date_diff('minute', app.applied_at, app.decided_at) / 60.0         as hours_to_decision,
    applicant.monthly_income,
    -- Income commitment (new installment / monthly income). Uses the contract installment when a
    -- loan exists; otherwise requested_amount / term_months (interest-free proxy).
    case
        when applicant.monthly_income > 0
        then round(coalesce(loan.installment_amount, app.requested_amount / nullif(app.term_months, 0))
                   / applicant.monthly_income, 4)
    end                                                                as income_commitment_ratio,
    case
        when not (applicant.monthly_income > 0) or applicant.monthly_income is null then 'not_computable'
        when loan.installment_amount is not null then 'contract_installment'
        else 'requested_amount_div_term'
    end                                                                as income_commitment_basis,
    -- Debt-to-income on an annual basis: existing_debt is the outstanding debt STOCK (total balance),
    -- not a monthly installment. NULL existing_debt (missing) must stay NULL, never coalesce to 0.
    case
        when applicant.monthly_income > 0 and app.existing_debt is not null
        then round(app.existing_debt / (12 * applicant.monthly_income), 4)
    end                                                                as debt_to_annual_income,
    app.batch_id,
    app.source_file,
    app.ingested_at
from app
left join loan on loan.application_id = app.application_id
left join applicant on applicant.applicant_id = app.applicant_id
