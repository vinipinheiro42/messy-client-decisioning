-- Every rate / probability column must lie in [0, 1].
select 'funnel' as model, application_month::varchar as k from {{ ref('agg_funnel_monthly') }}
where not (coalesce(approval_rate, 0) between 0 and 1 and coalesce(take_up_rate, 0) between 0 and 1 and end_to_end_rate between 0 and 1)
union all
select 'vintage', vintage_month::varchar from {{ ref('agg_vintage_default') }}
where not (coalesce(default_rate_12m, 0) between 0 and 1 and default_to_date_rate between 0 and 1)
union all
select 'perf', model_version || split from {{ ref('agg_model_performance') }}
where not (coalesce(auc, 0) between 0 and 1 and coalesce(ks, 0) between 0 and 1 and observed_default_rate between 0 and 1)
union all
select 'calib', model_version || pd_decile from {{ ref('agg_calibration_deciles') }}
where not (avg_predicted_pd between 0 and 1 and observed_default_rate between 0 and 1)
union all
select 'policy', model_version || score_cutoff from {{ ref('agg_policy_simulation') }}
where not (approval_rate between 0 and 1 and coalesce(observed_default_rate, 0) between 0 and 1
           and coalesce(observed_outcome_coverage, 0) between 0 and 1)
union all
select 'pov', model_version || target_approval_rate from {{ ref('agg_model_comparison_at_approval_rate') }}
where not (coalesce(observed_default_rate, 0) between 0 and 1)
union all
select 'dq', source_entity || batch_id || rule_failed from {{ ref('agg_data_quality') }}
where not (reject_rate between 0 and 1 and rule_share between 0 and 1)
union all
select 'scores', score_key from {{ ref('fct_scores') }} where not (pd between 0 and 1)
