{{ config(materialized='view') }}

with parsed as (
    select
        trim(installment_id) as installment_id,
        trim(loan_id) as loan_id,
        try_cast(trim(installment_number) as integer) as installment_number,
        {{ parse_date('due_date') }} as due_date,
        {{ parse_brl('amount_due') }} as amount_due,
        {{ parse_ts('paid_at') }} as paid_at,
        case
            when {{ is_null_token('amount_paid') }} then null
            else {{ parse_brl('amount_paid') }}
        end as amount_paid,
        trim(batch_id) as batch_id,
        source_file,
        {{ parse_ts('ingested_at') }} as ingested_at,
        due_date as due_date_raw,
        paid_at as paid_at_raw,
        amount_due as amount_due_raw,
        amount_paid as amount_paid_raw,
        md5(concat_ws(
            '||',
            coalesce(batch_id, ''),
            coalesce(source_file, ''),
            coalesce(installment_id, ''),
            coalesce(extracted_at, ''),
            coalesce(loan_id, ''),
            coalesce(installment_number, ''),
            coalesce(due_date, ''),
            coalesce(amount_due, '')
        )) as source_row_hash,
        to_json(struct_pack(
            installment_id := installment_id,
            loan_id := loan_id,
            installment_number := installment_number,
            due_date := due_date,
            amount_due := amount_due,
            paid_at := paid_at,
            amount_paid := amount_paid,
            extracted_at := extracted_at
        )) as raw_payload
    from {{ ref('brz_installments') }}
)

select
    *,
    case
        when installment_id is null or installment_id = '' then 'missing_id'
        when loan_id is null or loan_id = '' then 'missing_required'
        when installment_number is null or installment_number <= 0 then 'missing_required'
        when due_date is null and not {{ is_null_token('due_date_raw') }} then 'unparseable_date'
        when due_date is null then 'missing_required'
        when amount_due is null and not {{ is_null_token('amount_due_raw') }} then 'unparseable_money'
        when amount_due is null then 'missing_required'
        when paid_at is null
             and not {{ is_null_token('paid_at_raw') }}
             and trim(paid_at_raw) <> ''
            then 'unparseable_date'
        when amount_paid is null
             and not {{ is_null_token('amount_paid_raw') }}
            then 'unparseable_money'
        -- paid_at without a payment amount is contradictory. Quarantine this
        -- version so a previous clean version can survive; never impute 0.
        when paid_at is not null
             and (amount_paid is null or (amount_paid = 0 and amount_due > 0))
            then 'inconsistent_payment'
        else null
    end as rule_failed
from parsed
