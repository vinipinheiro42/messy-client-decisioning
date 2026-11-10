{{ config(materialized='table') }}
-- Grain: 1 row per application_id. Orphan applicant_ids (not in stg_applicants)
-- are quarantined even if the application itself parsed cleanly.

with conforming as (
    select c.*
    from {{ ref('int_applications_classified') }} c
    where c.rule_failed is null
),

with_parent as (
    select
        c.*,
        case when p.applicant_id is null then 'orphan_fk' else null end as parent_rule
    from conforming c
    left join {{ ref('stg_applicants') }} p
        on p.applicant_id = c.applicant_id
),

deduped as (
    select *
    from with_parent
    where parent_rule is null
    qualify row_number() over (
        partition by application_id
        order by ingested_at desc, batch_id desc, source_row_hash
    ) = 1
)

select
    application_id,
    applicant_id,
    applied_at,
    cast(requested_amount as decimal(18, 2)) as requested_amount,
    term_months,
    purpose,
    channel,
    bureau_score,
    -- Do not impute a missing balance as 0: that would bias credit-risk models.
    cast(existing_debt as decimal(18, 2)) as existing_debt,
    (existing_debt is null) as existing_debt_is_missing,
    decision,
    decided_at,
    batch_id,
    source_file,
    ingested_at
from deduped
