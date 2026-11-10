{{ config(materialized='table') }}
-- Grain: 1 row per application_id x model_version (PK of the scoring source).
-- Joins the historical decision and the observed loan outcome. Outcomes only exist for
-- applications that were approved AND booked (reject bias) - see has_outcome / is_observable.
with s as (
    select * from {{ source('ml', 'predictions_scores') }}
)
select
    s.application_id,
    s.model_version,
    s.application_id || '|' || s.model_version                         as score_key,
    s.pd,
    s.score,
    s.risk_band,
    s.reason_codes,
    s.split,
    s.scored_at,
    a.applicant_id,
    a.application_month,
    a.requested_amount,
    a.term_months,
    a.decision,
    a.is_approved,
    a.has_loan,
    o.loan_id,
    o.principal,
    o.term_months                                                      as loan_term_months,
    o.vintage_month,
    o.outcome_status,
    coalesce(o.is_observable, false)                                   as is_observable,
    o.is_default_12m,
    o.exposure_at_default,
    o.loan_id is not null                                              as has_outcome_record
from s
left join {{ ref('fct_applications') }} a on a.application_id = s.application_id
left join {{ ref('fct_loan_outcomes') }} o on o.application_id = s.application_id
