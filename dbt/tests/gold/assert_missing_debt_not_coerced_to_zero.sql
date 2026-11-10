-- Missing existing_debt must stay NULL in gold (never coalesced to 0 in DTI),
-- and existing_debt_is_missing must agree with existing_debt IS NULL.
-- Structural: no hardcoded counts. Remains valid when silver starts emitting NULLs + the flag.
select application_id
from {{ ref('fct_applications') }}
where existing_debt is null
  and (debt_to_annual_income is not null or existing_debt_is_missing is not true)
union all
select application_id
from {{ ref('fct_applications') }}
where existing_debt is not null
  and existing_debt_is_missing is distinct from false
union all
select application_id
from {{ ref('fct_applications') }}
where existing_debt is not null
  and monthly_income > 0
  and debt_to_annual_income is null
union all
select application_id
from {{ ref('fct_applications') }}
where (existing_debt is null or monthly_income is null or monthly_income <= 0)
  and debt_to_annual_income is not null
