-- FDDS R1: a loan with an observed / matured outcome (outcome_status in default/good,
-- equivalently is_observable) must satisfy
--   disbursed_at + 12 months + 90 days <= contract cutoff 2026-10-31.
-- Structural rule against the contract cutoff; does not hardcode counts or rates.
select loan_id
from {{ ref('fct_loan_outcomes') }}
where outcome_status in ('default', 'good')
  and cast(cast(disbursed_at as date) + interval 12 month + interval 90 day as date)
      > date '2026-10-31'
union all
select loan_id
from {{ ref('fct_loan_outcomes') }}
where is_observable
  and cast(cast(disbursed_at as date) + interval 12 month + interval 90 day as date)
      > date '2026-10-31'
