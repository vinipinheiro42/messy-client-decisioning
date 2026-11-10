{{ config(materialized='table') }}
-- Grain: 1 row per model_version x score cutoff (grid 300..900 step 10).
-- Policy rule simulated: approve when score >= cutoff.
-- Evaluation population: OOT split ONLY (split = 'oot'), so in-sample (train/validation) scores
-- never inflate the result.
-- REJECT BIAS: observed outcomes exist only for applications that were approved by the legacy
-- policy AND booked AND are observable (12m + 90d matured). Observed default rates and observed
-- net result are computed on that subset only; observed_outcome_coverage shows how much of the
-- policy-approved population that subset represents. Expected (model-based) metrics use the
-- calibrated pd for every policy-approved application and requested_amount as exposure proxy.
-- All money assumptions come from dim_policy_assumptions (seed policy_assumptions, SYNTHETIC).
-- Margin is an AVERAGE-BALANCE approximation of an amortized loan:
--   amount * net_margin * (term_months + 1) / 24
--   = principal * (n+1)/(2n) average outstanding * n/12 years * net annual margin,
--   with net_margin = gross_interest_margin_annual - funding_cost_annual.
--   Default uses the seed gross margin, not the contract interest_rate_monthly. A future
--   version may replace the gross margin with the loan rate minus funding cost from the seed.
-- expected_loss = sum(pd * requested_amount) * LGD   (requested_amount is the EAD proxy
--                 for applications that have not booked; PD x EAD x LGD)
-- expected_net = sum((1-pd)*amount*(term+1)/24)*net_margin - expected_loss
--                - n_approved*acquisition_cost - n_scored*underwriting_cost (constant across cutoffs)
-- observed_net = sum(principal_good*(term+1)/24)*net_margin - sum(EAD_default)*LGD
--                - n_observed*acquisition_cost   (underwriting cost not included)
with params as (
    select * from {{ ref('dim_policy_assumptions') }}
),
oot as (
    select * from {{ ref('fct_scores') }} where split = 'oot'
),
grid as (
    select cast(c as integer) as score_cutoff from range(300, 901, 10) t(c)
),
totals as (
    select model_version, count(*) as n_scored from oot group by model_version
),
sim as (
    select
        oot.model_version,
        g.score_cutoff,
        count(*)                                                       as n_policy_approved,
        sum(oot.pd)                                                    as sum_pd,
        sum(oot.pd * oot.requested_amount)                             as pd_weighted_amount,
        -- lifetime margin base per application: amount * (term+1)/24
        -- (average outstanding of an amortizing loan = principal*(n+1)/(2n); times n/12 years)
        sum((1 - oot.pd) * oot.requested_amount * (oot.term_months + 1) / 24.0) as margin_base_expected,
        count(*) filter (where oot.is_observable)                      as n_observed_outcomes,
        count(*) filter (where oot.is_observable and oot.is_default_12m = 1) as n_observed_defaults,
        sum(oot.principal * (oot.loan_term_months + 1) / 24.0) filter (where oot.is_observable and oot.is_default_12m = 0) as margin_base_good_observed,
        sum(oot.exposure_at_default) filter (where oot.is_observable and oot.is_default_12m = 1) as ead_default_observed
    from grid g
    join oot on oot.score >= g.score_cutoff
    group by oot.model_version, g.score_cutoff
),
full_grid as (
    select t.model_version, g.score_cutoff, t.n_scored
    from totals t cross join grid g
)
select
    fg.model_version,
    fg.score_cutoff,
    fg.n_scored                                                        as n_oot_scored,
    coalesce(s.n_policy_approved, 0)                                   as n_policy_approved,
    coalesce(s.n_policy_approved, 0) * 1.0 / fg.n_scored               as approval_rate,
    s.sum_pd / nullif(s.n_policy_approved, 0)                          as avg_predicted_pd,
    coalesce(s.n_observed_outcomes, 0)                                 as n_observed_outcomes,
    coalesce(s.n_observed_defaults, 0)                                 as n_observed_defaults,
    s.n_observed_defaults * 1.0 / nullif(s.n_observed_outcomes, 0)     as observed_default_rate,
    s.n_observed_outcomes * 1.0 / nullif(s.n_policy_approved, 0)       as observed_outcome_coverage,
    -- expected (model-based) economics on the full policy-approved OOT population
    -- expected_loss = PD x EAD_proxy x LGD (EAD_proxy = requested_amount)
    round(coalesce(s.pd_weighted_amount, 0) * p.lgd, 2)                as expected_loss_brl,
    round(coalesce(s.margin_base_expected, 0) * p.net_interest_margin_annual
          - coalesce(s.pd_weighted_amount, 0) * p.lgd
          - coalesce(s.n_policy_approved, 0) * p.acquisition_cost_brl
          - fg.n_scored * p.underwriting_cost_brl, 2)                  as expected_net_result_brl,
    -- observed economics on the approved & booked & observable subset only (reject bias!)
    round(coalesce(s.ead_default_observed, 0) * p.lgd, 2)              as observed_loss_brl,
    round(coalesce(s.margin_base_good_observed, 0) * p.net_interest_margin_annual
          - coalesce(s.ead_default_observed, 0) * p.lgd
          - coalesce(s.n_observed_outcomes, 0) * p.acquisition_cost_brl, 2) as observed_net_result_brl,
    round((coalesce(s.margin_base_good_observed, 0) * p.net_interest_margin_annual
          - coalesce(s.ead_default_observed, 0) * p.lgd) / nullif(s.n_observed_outcomes, 0)
          - p.acquisition_cost_brl, 2)                                   as observed_net_result_per_loan_brl,
    p.gross_interest_margin_annual,
    p.funding_cost_annual,
    p.net_interest_margin_annual,
    p.lgd,
    p.acquisition_cost_brl,
    p.underwriting_cost_brl,
    p.assumptions_label
from full_grid fg
cross join params p
left join sim s on s.model_version = fg.model_version and s.score_cutoff = fg.score_cutoff
