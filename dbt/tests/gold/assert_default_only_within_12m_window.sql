-- A 12m default must come from an installment due within 12 months of disbursement with 90+ DPD,
-- and the default date can never be after the cutoff.
select loan_id
from {{ ref('fct_loan_outcomes') }}
where has_default_event_to_date
  and (default_installment_due_date > window_end_date
       or default_date > as_of_date
       or max_dpd_in_window < 90)
union all
-- converse: no loan with 90+ DPD in window may be left without a default event
select loan_id
from {{ ref('fct_loan_outcomes') }}
where max_dpd_in_window >= 90 and not has_default_event_to_date
