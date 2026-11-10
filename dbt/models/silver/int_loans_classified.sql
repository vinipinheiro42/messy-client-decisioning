{{ config(materialized='view') }}

with parsed as (
    select
        trim(loan_id) as loan_id,
        trim(application_id) as application_id,
        {{ parse_ts('disbursed_at') }} as disbursed_at,
        {{ parse_brl('principal') }} as principal,
        try_cast(trim(interest_rate_monthly) as decimal(12, 6)) as interest_rate_monthly,
        try_cast(trim(term_months) as integer) as term_months,
        {{ parse_brl('installment_amount') }} as installment_amount,
        trim(batch_id) as batch_id,
        source_file,
        {{ parse_ts('ingested_at') }} as ingested_at,
        disbursed_at as disbursed_at_raw,
        principal as principal_raw,
        installment_amount as installment_amount_raw,
        md5(concat_ws(
            '||',
            coalesce(batch_id, ''),
            coalesce(source_file, ''),
            coalesce(loan_id, ''),
            coalesce(extracted_at, ''),
            coalesce(application_id, ''),
            coalesce(disbursed_at, ''),
            coalesce(principal, '')
        )) as source_row_hash,
        to_json(struct_pack(
            loan_id := loan_id,
            application_id := application_id,
            disbursed_at := disbursed_at,
            principal := principal,
            interest_rate_monthly := interest_rate_monthly,
            term_months := term_months,
            installment_amount := installment_amount,
            extracted_at := extracted_at
        )) as raw_payload
    from {{ ref('brz_loans') }}
)

select
    *,
    case
        when loan_id is null or loan_id = '' then 'missing_id'
        when application_id is null or application_id = '' then 'missing_required'
        when disbursed_at is null and not {{ is_null_token('disbursed_at_raw') }} then 'unparseable_date'
        when disbursed_at is null then 'missing_required'
        when principal is null and not {{ is_null_token('principal_raw') }} then 'unparseable_money'
        when principal is null or principal <= 0 then 'missing_required'
        when installment_amount is null and not {{ is_null_token('installment_amount_raw') }}
            then 'unparseable_money'
        when installment_amount is null or installment_amount <= 0 then 'missing_required'
        when term_months is null or term_months <= 0 then 'missing_required'
        when interest_rate_monthly is null then 'missing_required'
        else null
    end as rule_failed
from parsed
