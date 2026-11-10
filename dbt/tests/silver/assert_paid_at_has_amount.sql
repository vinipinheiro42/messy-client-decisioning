-- A filled paid_at must carry a positive amount_paid. Ground truth never has
-- paid_at with a zero payment. Imputing 0 here would treat a paid installment
-- as unpaid and inflate the 12-month default rate.
select
    installment_id,
    loan_id,
    paid_at,
    amount_paid,
    amount_due
from {{ ref('stg_installments') }}
where paid_at is not null
  and (amount_paid is null or amount_paid = 0)
