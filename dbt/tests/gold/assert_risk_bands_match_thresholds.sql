-- Seed per-model A-D thresholds must reproduce predictions_scores.risk_band
-- exactly (source of truth is the score table; the seed is for validation).
-- Assignment: A if pd < A_upper; B if pd < B_upper; C if pd < C_upper; else D.
-- No hardcoded cutoffs or counts.
with edges as (
    select
        model_version,
        max(try_cast(value as double)) filter (where assumption_key = 'risk_band_a_pd_upper') as a_hi,
        max(try_cast(value as double)) filter (where assumption_key = 'risk_band_b_pd_upper') as b_hi,
        max(try_cast(value as double)) filter (where assumption_key = 'risk_band_c_pd_upper') as c_hi
    from {{ ref('policy_assumptions') }}
    where coalesce(model_version, '') <> ''
    group by model_version
),
expected as (
    select
        s.application_id,
        s.model_version,
        s.risk_band as source_band,
        case
            when e.model_version is null then null
            when s.pd < e.a_hi then 'A'
            when s.pd < e.b_hi then 'B'
            when s.pd < e.c_hi then 'C'
            else 'D'
        end as seed_band
    from {{ source('ml', 'predictions_scores') }} s
    left join edges e using (model_version)
)
select application_id || '|' || model_version as score_key
from expected
where source_band is distinct from seed_band
