select 'stg_applicants' as source_table, applicant_id as row_key, monthly_income as amount
from {{ ref('stg_applicants') }}
where monthly_income < 0
union all
select 'stg_applications', application_id, requested_amount
from {{ ref('stg_applications') }}
where requested_amount <= 0
union all
select 'stg_applications', application_id, existing_debt
from {{ ref('stg_applications') }}
where existing_debt < 0
union all
select 'stg_loans', loan_id, principal
from {{ ref('stg_loans') }}
where principal <= 0
union all
select 'stg_installments', installment_id, amount_due
from {{ ref('stg_installments') }}
where amount_due < 0
