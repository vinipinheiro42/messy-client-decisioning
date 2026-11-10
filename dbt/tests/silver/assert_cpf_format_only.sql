-- CPF must be 11 digits. Invalid check digits are expected and must not
-- appear here as a reason to drop the row (those rows should have passed).
select applicant_id, cpf
from {{ ref('stg_applicants') }}
where cpf is null
   or regexp_full_match(cpf, '^[0-9]{11}$') is not true
