{{ config(materialized='table') }}
-- Grain: 1 row per model_version x risk_band. OOT observable outcomes only.
-- risk_band comes from predictions_scores via fct_scores (source of truth).
-- pd_low / pd_high are documented seed thresholds (per-model, A-D). bureau_score has no PD band.
with pop as (
    select model_version, risk_band, pd, is_default_12m as y
    from {{ ref('fct_scores') }}
    where split = 'oot' and is_observable and risk_band is not null
)
select
    p.model_version,
    p.risk_band,
    max(b.band_order)                                                  as band_order,
    max(b.pd_low)                                                      as pd_low,
    max(b.pd_high)                                                     as pd_high,
    count(*)                                                           as n_observed,
    sum(p.y)                                                           as n_default,
    avg(p.y)                                                           as observed_default_rate,
    avg(p.pd)                                                          as avg_predicted_pd
from pop p
left join {{ ref('dim_risk_bands') }} b
    on b.model_version = p.model_version
   and b.risk_band = p.risk_band
group by p.model_version, p.risk_band
