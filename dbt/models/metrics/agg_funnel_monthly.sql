{{ config(materialized='table') }}
-- Grain: 1 row per application_month (month of applied_at).
-- approval_rate      = n_approved / (n_approved + n_declined)   (cancelled excluded: no credit decision)
-- take_up_rate       = n_booked / n_approved                    (booked = has a loan contract)
-- end_to_end_rate    = n_booked / n_applications
select
    application_month,
    count(*)                                         as n_applications,
    count(*) filter (where is_cancelled)             as n_cancelled,
    count(*) filter (where is_declined)              as n_declined,
    count(*) filter (where is_approved)              as n_approved,
    count(*) filter (where has_loan)                 as n_booked,
    sum(requested_amount) filter (where has_loan)    as requested_amount_booked,
    count(*) filter (where is_approved) * 1.0
        / nullif(count(*) filter (where is_approved or is_declined), 0) as approval_rate,
    count(*) filter (where has_loan) * 1.0
        / nullif(count(*) filter (where is_approved), 0)               as take_up_rate,
    count(*) filter (where has_loan) * 1.0 / count(*)                  as end_to_end_rate,
    median(hours_to_decision)                        as median_hours_to_decision
from {{ ref('fct_applications') }}
group by application_month
