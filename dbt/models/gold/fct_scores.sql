{{ config(materialized='table') }}
-- Grain: 1 row per application_id x model_version.
-- Includes ML versions from source ml.predictions_scores AND the incumbent policy
-- 'bureau_score' (higher bureau_score = lower risk; ranking_key = -score).
-- risk_band for ML scores is read from predictions_scores (source of truth).
-- Per-model A-D thresholds in seed policy_assumptions are documentation/validation only.
-- ranking_key is the riskiness used for AUC/KS (higher = riskier).
with s as (
    select * from {{ source('ml', 'predictions_scores') }}
),
ml as (
    select
        s.application_id,
        s.model_version,
        s.application_id || '|' || s.model_version                     as score_key,
        s.pd,
        s.score,
        s.pd                                                           as ranking_key,
        true                                                           as is_pd_model,
        s.risk_band,
        s.reason_codes,
        s.split,
        s.scored_at
    from s
),
-- One split / scored_at per application, taken from the ML scores (same split for both models).
app_split as (
    select application_id, min(split) as split, min(scored_at) as scored_at
    from s
    group by application_id
),
bureau as (
    select
        a.application_id,
        'bureau_score'                                                 as model_version,
        a.application_id || '|bureau_score'                            as score_key,
        cast(null as double)                                           as pd,
        a.bureau_score                                                 as score,
        -1.0 * a.bureau_score                                          as ranking_key,
        false                                                          as is_pd_model,
        cast(null as varchar)                                          as risk_band,
        cast(null as varchar)                                          as reason_codes,
        sp.split,
        sp.scored_at
    from {{ ref('fct_applications') }} a
    left join app_split sp on sp.application_id = a.application_id
    where a.bureau_score is not null
),
all_scores as (
    select * from ml
    union all
    select * from bureau
)
select
    s.application_id,
    s.model_version,
    s.score_key,
    s.pd,
    s.score,
    s.ranking_key,
    s.is_pd_model,
    s.risk_band,
    s.reason_codes,
    s.split,
    s.scored_at,
    a.applicant_id,
    a.application_month,
    a.requested_amount,
    a.term_months,
    a.bureau_score,
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
from all_scores s
left join {{ ref('fct_applications') }} a on a.application_id = s.application_id
left join {{ ref('fct_loan_outcomes') }} o on o.application_id = s.application_id
