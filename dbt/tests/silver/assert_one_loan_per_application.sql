select application_id, count(*) as n
from {{ ref('stg_loans') }}
group by 1
having count(*) > 1
