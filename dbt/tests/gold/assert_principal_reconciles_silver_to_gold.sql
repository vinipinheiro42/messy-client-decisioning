-- Sum of principal and loan count in gold must equal silver.
with s as (select count(*) n, sum(principal) p from {{ ref('stg_loans') }}),
     g as (select count(*) n, sum(principal) p from {{ ref('fct_loan_outcomes') }})
select * from s, g where s.n <> g.n or s.p <> g.p
