{{ config(materialized='table') }}
-- Grain: 1 row per model_version x split.
-- Population: scored applications with an observable outcome (approved, booked, matured).
-- AUC  = Mann-Whitney: (sum of ranks of bads by pd - n_bad(n_bad+1)/2) / (n_bad * n_good), ties = avg rank
-- Gini = 2*AUC - 1
-- KS   = max over pd thresholds of |cum % bads - cum % goods| (sorted by pd desc)
-- Brier = mean((pd - is_default_12m)^2)
-- The headline metric for model selection is split = 'oot'.
with pop as (
    select model_version, split, application_id, pd, is_default_12m as y
    from {{ ref('fct_scores') }}
    where is_observable
),
ranked as (
    select *, rank() over (partition by model_version, split order by pd)
              + (count(*) over (partition by model_version, split, pd) - 1) / 2.0 as avg_rank
    from pop
),
auc as (
    select
        model_version, split,
        count(*) as n_observed,
        sum(y) as n_bad,
        count(*) - sum(y) as n_good,
        avg(pd) as avg_predicted_pd,
        avg(y) as observed_default_rate,
        avg(power(pd - y, 2)) as brier_score,
        (sum(avg_rank) filter (where y = 1) - sum(y) * (sum(y) + 1) / 2.0)
            / nullif(sum(y) * (count(*) - sum(y)), 0) as auc
    from ranked
    group by model_version, split
),
ks_steps as (
    -- aggregate ties first so KS is evaluated only at distinct pd thresholds
    select model_version, split, pd, sum(y) as bads, count(*) - sum(y) as goods
    from pop group by model_version, split, pd
),
ks_cum as (
    select
        model_version, split,
        sum(bads)  over w / nullif(sum(bads)  over p, 0) as cum_bad,
        sum(goods) over w / nullif(sum(goods) over p, 0) as cum_good
    from ks_steps
    window w as (partition by model_version, split order by pd desc rows between unbounded preceding and current row),
           p as (partition by model_version, split)
),
ks as (
    select model_version, split, max(abs(cum_bad - cum_good)) as ks
    from ks_cum group by model_version, split
)
select
    auc.model_version,
    auc.split,
    auc.n_observed,
    auc.n_bad,
    auc.n_good,
    auc.avg_predicted_pd,
    auc.observed_default_rate,
    auc.brier_score,
    auc.auc,
    2 * auc.auc - 1 as gini,
    ks.ks,
    auc.n_bad >= 30 and auc.n_good >= 30 as is_sample_sufficient
from auc
left join ks using (model_version, split)
