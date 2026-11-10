{{ config(materialized='table') }}
-- Grain: 1 row per applicant_id.
-- Age is measured at the applicant's first application (min applied_at), so the model does not
-- depend on optional applicant columns. PII-like columns (cpf, full_name) are intentionally not exposed.
{% set _app_cols = adapter.get_columns_in_relation(ref('stg_applications'))
                   | map(attribute='name') | map('lower') | list %}
with first_app as (
    select applicant_id, cast(min(applied_at) as date) as first_application_date
    from {{ ref('stg_applications') }}
    group by applicant_id
),
apps_debt as (
    -- Applicant-level rollup of missing existing_debt (application-grain flag lives on fct_applications).
    select
        applicant_id,
        count(*) as n_applications,
        count(*) filter (where existing_debt_is_missing) as n_applications_missing_existing_debt,
        bool_or(existing_debt_is_missing) as has_missing_existing_debt
    from (
        select
            applicant_id,
            {% if 'existing_debt_is_missing' in _app_cols %}
            coalesce(existing_debt_is_missing, existing_debt is null) as existing_debt_is_missing
            {% else %}
            (existing_debt is null) as existing_debt_is_missing
            {% endif %}
        from {{ ref('stg_applications') }}
    )
    group by applicant_id
),
a as (
    select
        ap.*,
        fa.first_application_date,
        date_diff('year', ap.birth_date, fa.first_application_date)
          - case when strftime(fa.first_application_date, '%m%d') < strftime(ap.birth_date, '%m%d') then 1 else 0 end
          as age_at_first_application,
        coalesce(ad.n_applications, 0) as n_applications,
        coalesce(ad.n_applications_missing_existing_debt, 0) as n_applications_missing_existing_debt,
        coalesce(ad.has_missing_existing_debt, false) as has_missing_existing_debt
    from {{ ref('stg_applicants') }} ap
    left join first_app fa on fa.applicant_id = ap.applicant_id
    left join apps_debt ad on ad.applicant_id = ap.applicant_id
)
select
    applicant_id,
    birth_date,
    age_at_first_application,
    case
        when age_at_first_application is null then 'unknown'
        when age_at_first_application < 25 then '18-24'
        when age_at_first_application < 35 then '25-34'
        when age_at_first_application < 45 then '35-44'
        when age_at_first_application < 60 then '45-59'
        else '60+'
    end as age_band,
    state_uf,
    monthly_income,
    case
        when monthly_income is null or monthly_income <= 0 then 'unknown'
        when monthly_income < 2000 then '< 2k'
        when monthly_income < 4000 then '2k-4k'
        when monthly_income < 8000 then '4k-8k'
        when monthly_income < 15000 then '8k-15k'
        else '15k+'
    end as income_band,
    employment_type,
    employment_months,
    housing_status,
    dependents,
    n_applications,
    n_applications_missing_existing_debt,
    has_missing_existing_debt,
    first_application_date,
    batch_id,
    source_file,
    ingested_at
from a
