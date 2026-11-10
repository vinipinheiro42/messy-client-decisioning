-- Joins must not multiply (or drop) rows: gold row counts equal their driving tables.
select * from (
    select 'fct_applications' as model,
           (select count(*) from {{ ref('fct_applications') }}) as gold_rows,
           (select count(*) from {{ ref('stg_applications') }}) as source_rows
    union all
    select 'fct_scores',
           (select count(*) from {{ ref('fct_scores') }}),
           (select count(*) from {{ source('ml', 'predictions_scores') }})
    union all
    select 'dim_applicant',
           (select count(*) from {{ ref('dim_applicant') }}),
           (select count(*) from {{ ref('stg_applicants') }})
    union all
    select 'fct_loan_outcomes',
           (select count(*) from {{ ref('fct_loan_outcomes') }}),
           (select count(*) from {{ ref('stg_loans') }})
) where gold_rows <> source_rows
