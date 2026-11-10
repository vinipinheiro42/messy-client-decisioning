{{ config(materialized='table') }}
-- Grain: 1 row per loan_id. At most one loan per application is kept
-- (latest ingested_at). Loans whose application is not in stg_applications
-- are quarantined as orphan_fk.

with conforming as (
    select *
    from {{ ref('int_loans_classified') }}
    where rule_failed is null
),

with_parent as (
    select
        c.*,
        case when p.application_id is null then 'orphan_fk' else null end as parent_rule
    from conforming c
    left join {{ ref('stg_applications') }} p
        on p.application_id = c.application_id
),

deduped as (
    select *
    from with_parent
    where parent_rule is null
    qualify row_number() over (
        partition by loan_id
        order by ingested_at desc, batch_id desc, source_row_hash
    ) = 1
),

-- Enforce 0..1 loan per application if a dirty extract ever produced two.
one_per_application as (
    select *
    from deduped
    qualify row_number() over (
        partition by application_id
        order by ingested_at desc, disbursed_at desc, loan_id
    ) = 1
)

select
    loan_id,
    application_id,
    disbursed_at,
    cast(principal as decimal(18, 2)) as principal,
    interest_rate_monthly,
    term_months,
    cast(installment_amount as decimal(18, 2)) as installment_amount,
    batch_id,
    source_file,
    ingested_at
from one_per_application
