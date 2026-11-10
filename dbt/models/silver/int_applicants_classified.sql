{{ config(materialized='view') }}
-- Parse and classify every bronze applicant row. rule_failed is null when the
-- row may enter stg_applicants. CPF check digits are intentionally invalid
-- and are not a quarantine reason; only the 11-digit format is validated.

with mapped as (
    select
        a.*,
        emp.canonical_value as employment_type_std,
        hou.canonical_value as housing_status_std,
        uf.canonical_value as state_uf_std
    from {{ ref('brz_applicants') }} a
    left join {{ ref('category_mappings') }} emp
        on emp.column_name = 'employment_type'
       and lower(trim(emp.raw_value)) = lower(trim(a.employment_type))
    left join {{ ref('category_mappings') }} hou
        on hou.column_name = 'housing_status'
       and lower(trim(hou.raw_value)) = lower(trim(a.housing_status))
    left join {{ ref('category_mappings') }} uf
        on uf.column_name = 'state_uf'
       and lower(trim(uf.raw_value)) = lower(trim(a.state_uf))
),

parsed as (
    select
        trim(applicant_id) as applicant_id,
        {{ digits_only('cpf') }} as cpf,
        trim(full_name) as full_name,
        {{ parse_date('birth_date') }} as birth_date,
        state_uf_std as state_uf,
        case
            when trim(monthly_income) in ('0', '0.0', '0.00') then null
            else {{ parse_brl('monthly_income') }}
        end as monthly_income,
        employment_type_std as employment_type,
        try_cast(trim(employment_months) as integer) as employment_months,
        housing_status_std as housing_status,
        try_cast(trim(dependents) as integer) as dependents,
        trim(batch_id) as batch_id,
        source_file,
        {{ parse_ts('ingested_at') }} as ingested_at,
        birth_date as birth_date_raw,
        monthly_income as monthly_income_raw,
        employment_type as employment_type_raw,
        housing_status as housing_status_raw,
        state_uf as state_uf_raw,
        md5(concat_ws(
            '||',
            coalesce(batch_id, ''),
            coalesce(source_file, ''),
            coalesce(applicant_id, ''),
            coalesce(extracted_at, ''),
            coalesce(cpf, ''),
            coalesce(full_name, ''),
            coalesce(birth_date, ''),
            coalesce(monthly_income, ''),
            coalesce(employment_type, ''),
            coalesce(housing_status, ''),
            coalesce(state_uf, '')
        )) as source_row_hash,
        to_json(struct_pack(
            applicant_id := applicant_id,
            cpf := cpf,
            full_name := full_name,
            birth_date := birth_date,
            state_uf := state_uf,
            monthly_income := monthly_income,
            employment_type := employment_type,
            employment_months := employment_months,
            housing_status := housing_status,
            dependents := dependents,
            extracted_at := extracted_at
        )) as raw_payload
    from mapped
),

aged as (
    select
        *,
        case
            when birth_date is null then null
            else date_diff('year', birth_date, date '2025-01-01')
                - case
                    when strftime(date '2025-01-01', '%m%d') < strftime(birth_date, '%m%d')
                    then 1 else 0
                  end
        end as age_years
    from parsed
)

select
    *,
    case
        when applicant_id is null or applicant_id = '' then 'missing_id'
        when length(cpf) <> 11 then 'invalid_cpf_format'
        when birth_date is null and not {{ is_null_token('birth_date_raw') }} then 'unparseable_date'
        when birth_date is null then 'missing_required'
        when monthly_income is null
             and not {{ is_null_token('monthly_income_raw') }}
             and trim(monthly_income_raw) not in ('0', '0.0', '0.00')
            then 'unparseable_money'
        when monthly_income is null then 'missing_required'
        when monthly_income < 0 then 'outlier_income'
        when age_years < 16 or age_years > 100 then 'outlier_age'
        when employment_type is null
             and not {{ is_null_token('employment_type_raw') }}
            then 'unmapped_category'
        when employment_type is null then 'missing_required'
        when state_uf is null
             and not {{ is_null_token('state_uf_raw') }}
            then 'unmapped_category'
        when state_uf is null then 'missing_required'
        when housing_status is null
             and not {{ is_null_token('housing_status_raw') }}
            then 'unmapped_category'
        else null
    end as rule_failed
from aged
