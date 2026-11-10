{{ config(materialized='view') }}
-- Parse and classify every bronze application row.
-- existing_debt is a total outstanding debt balance in BRL, not a monthly payment;
-- a value of 0 is valid and is not treated as a disguised null.

with mapped as (
    select
        a.*,
        pur.canonical_value as purpose_std,
        ch.canonical_value as channel_std,
        dec.canonical_value as decision_std
    from {{ ref('brz_applications') }} a
    left join {{ ref('category_mappings') }} pur
        on pur.column_name = 'purpose'
       and lower(trim(pur.raw_value)) = lower(trim(a.purpose))
    left join {{ ref('category_mappings') }} ch
        on ch.column_name = 'channel'
       and lower(trim(ch.raw_value)) = lower(trim(a.channel))
    left join {{ ref('category_mappings') }} dec
        on dec.column_name = 'decision'
       and lower(trim(dec.raw_value)) = lower(trim(a.decision))
),

parsed as (
    select
        trim(application_id) as application_id,
        trim(applicant_id) as applicant_id,
        {{ parse_ts('applied_at') }} as applied_at,
        case
            when trim(requested_amount) in ('0', '0.0', '0.00') then null
            else {{ parse_brl('requested_amount') }}
        end as requested_amount,
        try_cast(trim(term_months) as integer) as term_months,
        purpose_std as purpose,
        channel_std as channel,
        try_cast(trim(bureau_score) as integer) as bureau_score,
        {{ parse_brl('existing_debt') }} as existing_debt,
        decision_std as decision,
        {{ parse_ts('decided_at') }} as decided_at,
        trim(batch_id) as batch_id,
        source_file,
        {{ parse_ts('ingested_at') }} as ingested_at,
        applied_at as applied_at_raw,
        decided_at as decided_at_raw,
        requested_amount as requested_amount_raw,
        existing_debt as existing_debt_raw,
        purpose as purpose_raw,
        channel as channel_raw,
        decision as decision_raw,
        md5(concat_ws(
            '||',
            coalesce(batch_id, ''),
            coalesce(source_file, ''),
            coalesce(application_id, ''),
            coalesce(extracted_at, ''),
            coalesce(applicant_id, ''),
            coalesce(applied_at, ''),
            coalesce(requested_amount, ''),
            coalesce(decision, '')
        )) as source_row_hash,
        to_json(struct_pack(
            application_id := application_id,
            applicant_id := applicant_id,
            applied_at := applied_at,
            requested_amount := requested_amount,
            term_months := term_months,
            purpose := purpose,
            channel := channel,
            bureau_score := bureau_score,
            existing_debt := existing_debt,
            decision := decision,
            decided_at := decided_at,
            extracted_at := extracted_at
        )) as raw_payload
    from mapped
)

select
    *,
    case
        when application_id is null or application_id = '' then 'missing_id'
        when applicant_id is null or applicant_id = '' then 'missing_required'
        when applied_at is null and not {{ is_null_token('applied_at_raw') }} then 'unparseable_date'
        when applied_at is null then 'missing_required'
        when decided_at is null and not {{ is_null_token('decided_at_raw') }} then 'unparseable_date'
        when decided_at is null then 'missing_required'
        when requested_amount is null
             and not {{ is_null_token('requested_amount_raw') }}
             and trim(requested_amount_raw) not in ('0', '0.0', '0.00')
            then 'unparseable_money'
        when requested_amount is null then 'missing_required'
        when requested_amount >= 1000000 then 'outlier_amount'
        when term_months is null or term_months <= 0 then 'missing_required'
        when purpose is null and not {{ is_null_token('purpose_raw') }} then 'unmapped_category'
        when purpose is null then 'missing_required'
        when channel is null and not {{ is_null_token('channel_raw') }} then 'unmapped_category'
        when channel is null then 'missing_required'
        when decision is null and not {{ is_null_token('decision_raw') }} then 'unmapped_category'
        when decision is null then 'missing_required'
        when existing_debt is null and not {{ is_null_token('existing_debt_raw') }}
            then 'unparseable_money'
        else null
    end as rule_failed
from parsed
