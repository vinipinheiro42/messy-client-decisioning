{{ config(materialized='table') }}
-- Grain: 1 row per applicant_id. Latest ingested_at among rows that passed
-- quality rules. A later failing resend is quarantined and does not hide an
-- earlier conforming version.

with conforming as (
    select *
    from {{ ref('int_applicants_classified') }}
    where rule_failed is null
),

deduped as (
    select *
    from conforming
    qualify row_number() over (
        partition by applicant_id
        order by ingested_at desc, batch_id desc, source_row_hash
    ) = 1
)

select
    applicant_id,
    cpf,
    full_name,
    birth_date,
    state_uf,
    cast(monthly_income as decimal(18, 2)) as monthly_income,
    employment_type,
    employment_months,
    housing_status,
    dependents,
    batch_id,
    source_file,
    ingested_at
from deduped
