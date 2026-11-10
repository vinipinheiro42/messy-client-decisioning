-- Reject bias guard: outcomes can only exist for approved + booked applications.
select score_key from {{ ref('fct_scores') }}
where has_outcome_record and not (is_approved and has_loan)
union all
select score_key from {{ ref('fct_scores') }}
where is_observable and not has_outcome_record
