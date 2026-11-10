-- Raising the score cutoff can never increase the approval rate.
select model_version, score_cutoff
from (
    select *, lag(approval_rate) over (partition by model_version order by score_cutoff) as prev_rate
    from {{ ref('agg_policy_simulation') }}
)
where approval_rate > prev_rate
