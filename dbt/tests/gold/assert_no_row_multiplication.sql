-- Joins must not multiply (or drop) rows: gold row counts equal their driving tables.
-- fct_scores also includes the incumbent bureau_score policy (one row per application
-- with a bureau_score), so the ML portion is compared to the scoring source separately.
select * from (
    select 'fct_applications' as model,
           (select count(*) from {{ ref('fct_applications') }}) as gold_rows,
           (select count(*) from {{ ref('stg_applications') }}) as source_rows
    union all
    select 'fct_scores_ml',
           (select count(*) from {{ ref('fct_scores') }} where is_pd_model),
           (select count(*) from {{ source('ml', 'predictions_scores') }})
    union all
    select 'fct_scores_bureau',
           (select count(*) from {{ ref('fct_scores') }} where model_version = 'bureau_score'),
           (select count(*) from {{ ref('fct_applications') }} where bureau_score is not null)
    union all
    select 'dim_applicant',
           (select count(*) from {{ ref('dim_applicant') }}),
           (select count(*) from {{ ref('stg_applicants') }})
    union all
    select 'fct_loan_outcomes',
           (select count(*) from {{ ref('fct_loan_outcomes') }}),
           (select count(*) from {{ ref('stg_loans') }})
) where gold_rows <> source_rows
