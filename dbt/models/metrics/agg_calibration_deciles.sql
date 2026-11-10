{{ config(materialized='table') }}
-- Grain: 1 row per model_version x pd decile (1 = lowest predicted risk), OOT split only.
-- Population: OOT scored applications with an observable outcome.
-- avg_predicted_pd vs observed_default_rate (= n_default / n_observed) per decile.
with pop as (
    select model_version, pd, is_default_12m as y
    from {{ ref('fct_scores') }}
    where split = 'oot' and is_observable
),
d as (
    select *, ntile(10) over (partition by model_version order by pd) as pd_decile from pop
)
select
    model_version,
    pd_decile,
    count(*)                         as n_observed,
    sum(y)                           as n_default,
    min(pd)                          as min_pd,
    max(pd)                          as max_pd,
    avg(pd)                          as avg_predicted_pd,
    avg(y)                           as observed_default_rate,
    avg(y) - avg(pd)                 as calibration_gap
from d
group by model_version, pd_decile
