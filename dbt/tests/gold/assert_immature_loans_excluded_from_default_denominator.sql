-- Immature loans must have a NULL label and never enter default-rate denominators.
select 'label_on_immature' as failure, loan_id::varchar as ref
from {{ ref('fct_loan_outcomes') }}
where not is_observable and (is_default_12m is not null or outcome_status <> 'immature')
union all
select 'missing_label_on_observable', loan_id::varchar
from {{ ref('fct_loan_outcomes') }}
where is_observable and is_default_12m is null
union all
select 'vintage_denominator_mismatch', cast(v.n as varchar)
from (select sum(n_observable) n from {{ ref('agg_vintage_default') }}) v,
     (select count(*) filter (where is_observable) n from {{ ref('fct_loan_outcomes') }}) f
where v.n <> f.n
union all
select 'perf_denominator_mismatch', model_version
from (
    select model_version, sum(n_observed) n from {{ ref('agg_model_performance') }} group by 1
) p
where p.n <> (select count(*) from {{ ref('fct_scores') }} s where s.is_observable and s.model_version = p.model_version)
