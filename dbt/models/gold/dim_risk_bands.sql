{{ config(materialized='table') }}
-- Grain: 1 row per model_version x risk_band (A-D).
-- Thresholds are documented from seed policy_assumptions (per-model
-- risk_band_*_pd_upper). They are NOT used to assign fct_scores.risk_band:
-- that column is read from predictions_scores (source of truth).
-- Assignment rule for validation: A if pd < A_upper; B if A_upper <= pd < B_upper;
-- C if B_upper <= pd < C_upper; D otherwise.
with s as (
    select
        model_version,
        assumption_key,
        try_cast(value as double) as v
    from {{ ref('policy_assumptions') }}
    where coalesce(model_version, '') <> ''
),
u as (
    select
        model_version,
        max(v) filter (where assumption_key = 'risk_band_a_pd_upper') as a_hi,
        max(v) filter (where assumption_key = 'risk_band_b_pd_upper') as b_hi,
        max(v) filter (where assumption_key = 'risk_band_c_pd_upper') as c_hi
    from s
    group by model_version
)
select model_version || '|A' as band_key, model_version, 'A' as risk_band,
       0.0 as pd_low, a_hi as pd_high, 1 as band_order, 'pd < ' || a_hi as rule
from u
union all
select model_version || '|B', model_version, 'B',
       a_hi, b_hi, 2, a_hi || ' <= pd < ' || b_hi
from u
union all
select model_version || '|C', model_version, 'C',
       b_hi, c_hi, 3, b_hi || ' <= pd < ' || c_hi
from u
union all
select model_version || '|D', model_version, 'D',
       c_hi, 1.01, 4, 'pd >= ' || c_hi
from u
