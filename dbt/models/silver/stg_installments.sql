{{ config(materialized='table') }}
-- Grain: 1 row per (loan_id, installment_number), latest ingested_at.
-- days_past_due is measured as of the cutoff of the batch that last loaded
-- the surviving row, not as of the global observation date.

with conforming as (
    select *
    from {{ ref('int_installments_classified') }}
    where rule_failed is null
),

with_parent as (
    select
        c.*,
        case when l.loan_id is null then 'orphan_fk' else null end as parent_rule
    from conforming c
    left join {{ ref('stg_loans') }} l
        on l.loan_id = c.loan_id
),

deduped as (
    select *
    from with_parent
    where parent_rule is null
    qualify row_number() over (
        partition by loan_id, installment_number
        order by ingested_at desc, batch_id desc, source_row_hash
    ) = 1
),

with_cutoff as (
    select
        d.*,
        b.cutoff_date
    from deduped d
    left join {{ ref('batch_cutoffs') }} b
        on b.batch_id = d.batch_id
)

select
    installment_id,
    loan_id,
    installment_number,
    due_date,
    cast(amount_due as decimal(18, 2)) as amount_due,
    paid_at,
    -- Unpaid installments (paid_at is null) have a true zero payment, so we
    -- store 0. A raw version with paid_at filled and amount_paid missing is
    -- quarantined as inconsistent_payment, allowing an earlier clean version
    -- to survive. Never coalesce amount_paid to 0 when paid_at is present:
    -- that marks a paid installment as unpaid and inflates the default rate.
    cast(
        case
            when paid_at is null then coalesce(amount_paid, 0)
            else amount_paid
        end as decimal(18, 2)
    ) as amount_paid,
    case
        when due_date is null or cutoff_date is null then 0
        when due_date > cutoff_date then 0
        when paid_at is not null
             and cast(paid_at as date) <= cutoff_date
             and coalesce(amount_paid, 0) >= amount_due
            then 0
        else greatest(0, date_diff('day', due_date, cutoff_date))
    end as days_past_due,
    batch_id,
    source_file,
    ingested_at
from with_cutoff
