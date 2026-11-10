{{ config(materialized='table') }}
-- Grain: 1 row per model_version x target_approval_rate (0.10..0.90 step 0.05). OOT only.
-- Proof of Value: each policy approves the same share of OOT applications (top-k by score,
-- ties broken by lower pd then application_id), so they are compared at EQUAL approval rate.
-- Includes the incumbent 'bureau_score' (higher bureau = lower risk; no calibrated pd,
-- so expected-loss metrics are NULL; observed metrics are still computed).
-- observed_default_rate = observed defaults / observed outcomes within the approved set
-- (reject bias: only approved & booked & matured applications carry outcomes).
-- Economics use the same formulas as agg_policy_simulation.
with params as (
    select * from {{ ref('dim_policy_assumptions') }}
),
oot as (
    select *,
        row_number() over (partition by model_version order by score desc, pd asc nulls last, application_id) as rk,
        count(*) over (partition by model_version) as n_scored
    from {{ ref('fct_scores') }}
    where split = 'oot'
),
rates as (
    select round(r / 100.0, 2) as target_approval_rate from range(10, 91, 5) t(r)
)
select
    oot.model_version,
    rates.target_approval_rate,
    max(oot.n_scored)                                                         as n_oot_scored,
    count(*)                                                                  as n_policy_approved,
    min(oot.score)                                                            as implied_score_cutoff,
    count(*) filter (where oot.is_observable)                                 as n_observed_outcomes,
    count(*) filter (where oot.is_observable and oot.is_default_12m = 1)      as n_observed_defaults,
    count(*) filter (where oot.is_observable and oot.is_default_12m = 1) * 1.0
        / nullif(count(*) filter (where oot.is_observable), 0)                as observed_default_rate,
    avg(oot.pd)                                                               as avg_predicted_pd,
    case when count(oot.pd) = 0 then null
         else round(sum(oot.pd * oot.requested_amount) * max(p.lgd), 2) end    as expected_loss_brl,
    case when count(oot.pd) = 0 then null
         else round(sum((1 - oot.pd) * oot.requested_amount * (oot.term_months + 1) / 24.0) * max(p.net_interest_margin_annual)
          - sum(oot.pd * oot.requested_amount) * max(p.lgd)
          - count(*) * max(p.acquisition_cost_brl)
          - max(oot.n_scored) * max(p.underwriting_cost_brl), 2) end           as expected_net_result_brl,
    round(coalesce(sum(oot.principal * (oot.loan_term_months + 1) / 24.0) filter (where oot.is_observable and oot.is_default_12m = 0), 0) * max(p.net_interest_margin_annual)
          - coalesce(sum(oot.exposure_at_default) filter (where oot.is_observable and oot.is_default_12m = 1), 0) * max(p.lgd)
          - count(*) filter (where oot.is_observable) * max(p.acquisition_cost_brl), 2) as observed_net_result_brl,
    -- normalized by observed loans, because models with different coverage are not comparable in total BRL
    round((coalesce(sum(oot.principal * (oot.loan_term_months + 1) / 24.0) filter (where oot.is_observable and oot.is_default_12m = 0), 0) * max(p.net_interest_margin_annual)
          - coalesce(sum(oot.exposure_at_default) filter (where oot.is_observable and oot.is_default_12m = 1), 0) * max(p.lgd))
          / nullif(count(*) filter (where oot.is_observable), 0) - max(p.acquisition_cost_brl), 2) as observed_net_result_per_loan_brl,
    max(p.gross_interest_margin_annual)                                       as gross_interest_margin_annual,
    max(p.funding_cost_annual)                                                as funding_cost_annual,
    max(p.net_interest_margin_annual)                                         as net_interest_margin_annual,
    max(p.lgd)                                                                as lgd,
    max(p.assumptions_label)                                                  as assumptions_label
from oot
cross join rates
cross join params p
where oot.rk <= ceil(rates.target_approval_rate * oot.n_scored)
group by oot.model_version, rates.target_approval_rate
