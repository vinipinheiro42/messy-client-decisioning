{{ config(materialized='table') }}
-- Rows that failed a silver quality rule and cannot be safely repaired.
-- Includes parse failures and orphan foreign keys. Exact/near-duplicate
-- resends that parse cleanly are not quarantined; they are collapsed in
-- the conforming tables by latest ingested_at.

with applicants as (
    select
        'applicants' as source_table,
        source_row_hash,
        batch_id,
        rule_failed,
        raw_payload,
        source_file,
        ingested_at,
        applicant_id as row_key
    from {{ ref('int_applicants_classified') }}
    where rule_failed is not null
),

applications_parse as (
    select
        'applications' as source_table,
        source_row_hash,
        batch_id,
        rule_failed,
        raw_payload,
        source_file,
        ingested_at,
        application_id as row_key
    from {{ ref('int_applications_classified') }}
    where rule_failed is not null
),

applications_orphan as (
    select
        'applications' as source_table,
        c.source_row_hash,
        c.batch_id,
        'orphan_fk' as rule_failed,
        c.raw_payload,
        c.source_file,
        c.ingested_at,
        c.application_id as row_key
    from {{ ref('int_applications_classified') }} c
    left join {{ ref('stg_applicants') }} p
        on p.applicant_id = c.applicant_id
    where c.rule_failed is null
      and p.applicant_id is null
),

loans_parse as (
    select
        'loans' as source_table,
        source_row_hash,
        batch_id,
        rule_failed,
        raw_payload,
        source_file,
        ingested_at,
        loan_id as row_key
    from {{ ref('int_loans_classified') }}
    where rule_failed is not null
),

loans_orphan as (
    select
        'loans' as source_table,
        c.source_row_hash,
        c.batch_id,
        'orphan_fk' as rule_failed,
        c.raw_payload,
        c.source_file,
        c.ingested_at,
        c.loan_id as row_key
    from {{ ref('int_loans_classified') }} c
    left join {{ ref('stg_applications') }} p
        on p.application_id = c.application_id
    where c.rule_failed is null
      and p.application_id is null
),

installments_parse as (
    select
        'installments' as source_table,
        source_row_hash,
        batch_id,
        rule_failed,
        raw_payload,
        source_file,
        ingested_at,
        installment_id as row_key
    from {{ ref('int_installments_classified') }}
    where rule_failed is not null
),

installments_orphan as (
    select
        'installments' as source_table,
        c.source_row_hash,
        c.batch_id,
        'orphan_fk' as rule_failed,
        c.raw_payload,
        c.source_file,
        c.ingested_at,
        c.installment_id as row_key
    from {{ ref('int_installments_classified') }} c
    left join {{ ref('stg_loans') }} l
        on l.loan_id = c.loan_id
    where c.rule_failed is null
      and l.loan_id is null
)

select
    source_table,
    source_row_hash,
    batch_id,
    rule_failed,
    raw_payload,
    source_file,
    ingested_at,
    row_key
from (
    select * from applicants
    union all
    select * from applications_parse
    union all
    select * from applications_orphan
    union all
    select * from loans_parse
    union all
    select * from loans_orphan
    union all
    select * from installments_parse
    union all
    select * from installments_orphan
) q
