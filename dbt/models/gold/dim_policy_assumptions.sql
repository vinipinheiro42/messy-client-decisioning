{{ config(materialized='table') }}
-- One row: SYNTHETIC policy assumptions pivoted from seed policy_assumptions
-- (columns assumption_key, value, unit, is_synthetic, description; value may be text).
-- net_interest_margin_annual = gross_interest_margin_annual - funding_cost_annual.
-- Not used here: recovery_lag_months, discount_rate_annual (no discounting in this version).
with s as (
    select assumption_key, try_cast(value as double) as v
    from {{ ref('policy_assumptions') }}
)
select
    max(v) filter (where assumption_key = 'gross_interest_margin_annual') as gross_interest_margin_annual,
    max(v) filter (where assumption_key = 'funding_cost_annual')          as funding_cost_annual,
    max(v) filter (where assumption_key = 'gross_interest_margin_annual')
        - max(v) filter (where assumption_key = 'funding_cost_annual')    as net_interest_margin_annual,
    max(v) filter (where assumption_key = 'lgd')                          as lgd,
    max(v) filter (where assumption_key = 'acquisition_cost_brl')         as acquisition_cost_brl,
    coalesce(max(v) filter (where assumption_key = 'underwriting_cost_brl'), 0) as underwriting_cost_brl,
    'synthetic assumptions (seed policy_assumptions)'                     as assumptions_label
from s
