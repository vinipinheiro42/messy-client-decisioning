{{ config(materialized='table') }}
-- Grain: 1 row per source_entity x batch_id x rule_failed, with check_type
--   'quarantine'     — rows rejected by silver quality rules
--   'completeness'   — attribute completeness on accepted silver rows (e.g. existing_debt missing)
-- reject_rate (quarantine, per table x batch) = n_rejected_batch / (n_rejected_batch + n_accepted_batch)
-- reject_rate (completeness) = n_missing / n_rows in that batch (n_accepted_batch holds the denominator)
-- rule_share  = n_rejected (this rule) / n_rejected_batch
-- source_table is normalized by stripping brz_/stg_/raw_ prefixes so it matches silver tables.
{% set _app_cols = adapter.get_columns_in_relation(ref('stg_applications'))
                   | map(attribute='name') | map('lower') | list %}
with q as (
    select
        regexp_replace(lower(source_table), '^(stg_|brz_|raw_)', '') as source_entity,
        batch_id,
        rule_failed,
        count(*) as n_rejected
    from {{ ref('stg_quarantine') }}
    group by all
),
accepted as (
    select 'applicants'   as source_entity, batch_id, count(*) as n_accepted from {{ ref('stg_applicants') }}   group by all
    union all
    select 'applications', batch_id, count(*) from {{ ref('stg_applications') }} group by all
    union all
    select 'loans',        batch_id, count(*) from {{ ref('stg_loans') }}        group by all
    union all
    select 'installments', batch_id, count(*) from {{ ref('stg_installments') }} group by all
),
q_batch as (
    select source_entity, batch_id, sum(n_rejected) as n_rejected_batch from q group by all
),
quarantine as (
    select
        q.source_entity,
        q.batch_id,
        q.rule_failed,
        'quarantine'                                                       as check_type,
        q.n_rejected,
        qb.n_rejected_batch,
        coalesce(a.n_accepted, 0)                                          as n_accepted_batch,
        qb.n_rejected_batch * 1.0 / (qb.n_rejected_batch + coalesce(a.n_accepted, 0)) as reject_rate,
        q.n_rejected * 1.0 / qb.n_rejected_batch                           as rule_share
    from q
    join q_batch qb using (source_entity, batch_id)
    left join accepted a using (source_entity, batch_id)
),
missing_debt as (
    -- One completeness row per applications batch, even when the rate is 0, so the metric is always visible.
    -- Uses the silver flag when present; otherwise existing_debt IS NULL. Zeros are treated as genuine
    -- zero balances (not missing) until silver stops coalescing missing debt to 0.
    select
        'applications'                                                     as source_entity,
        batch_id,
        'existing_debt_missing'                                            as rule_failed,
        'completeness'                                                     as check_type,
        count(*) filter (
            where {% if 'existing_debt_is_missing' in _app_cols %}
                  coalesce(existing_debt_is_missing, existing_debt is null)
                  {% else %}
                  existing_debt is null
                  {% endif %}
        )                                                                  as n_rejected,
        count(*)                                                           as n_rows
    from {{ ref('stg_applications') }}
    group by batch_id
),
completeness as (
    select
        source_entity,
        batch_id,
        rule_failed,
        check_type,
        n_rejected,
        n_rejected                                                         as n_rejected_batch,
        n_rows                                                             as n_accepted_batch,
        n_rejected * 1.0 / nullif(n_rows, 0)                               as reject_rate,
        1.0                                                                as rule_share
    from missing_debt
)
select * from quarantine
union all
select * from completeness
