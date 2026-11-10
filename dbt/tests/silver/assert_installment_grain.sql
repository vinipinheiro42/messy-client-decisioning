-- Fails if (loan_id, installment_number) is not unique in stg_installments.
select loan_id, installment_number, count(*) as n
from {{ ref('stg_installments') }}
group by 1, 2
having count(*) > 1
