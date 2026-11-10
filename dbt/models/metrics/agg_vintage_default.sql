{{ config(materialized='table') }}
-- Grain: 1 row per vintage_month (month of disbursement).
-- default_rate_12m = n_default / n_observable   (numerator: outcome_status = 'default';
--                    denominator: is_observable loans only - immature loans excluded)
-- default_to_date_rate = loans with a default event so far / all loans (NOT comparable across
--                    vintages; shown only to monitor immature vintages)
select
    vintage_month,
    count(*)                                                    as n_loans,
    count(*) filter (where is_observable)                       as n_observable,
    count(*) filter (where not is_observable)                   as n_immature,
    count(*) filter (where outcome_status = 'default')          as n_default,
    count(*) filter (where outcome_status = 'default') * 1.0
        / nullif(count(*) filter (where is_observable), 0)      as default_rate_12m,
    count(*) filter (where has_default_event_to_date) * 1.0 / count(*) as default_to_date_rate,
    sum(principal)                                              as principal_disbursed,
    sum(exposure_at_default) filter (where outcome_status = 'default') as ead_defaulted,
    sum(exposure_at_default) filter (where outcome_status = 'default')
        / nullif(sum(principal) filter (where is_observable), 0) as ead_default_rate_by_value,
    bool_and(is_observable)                                     as is_fully_observable
from {{ ref('fct_loan_outcomes') }}
group by vintage_month
