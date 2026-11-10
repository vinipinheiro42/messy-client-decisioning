{{ config(materialized='table') }}
-- One row: the observation (cutoff) date used for outcome maturity.
-- Priority: var('as_of_date') if provided, otherwise the contract global cutoff 2026-10-31.
-- days_past_due is a per-batch snapshot and is NOT used to infer the cutoff.
with inst_meta as (
    select count(distinct batch_id) as n_installment_batches
    from {{ ref('stg_installments') }}
)
select
    cast('{{ var("as_of_date", "2026-10-31") }}' as date) as as_of_date,
    {% if var('as_of_date', none) %}
    'var:as_of_date' as as_of_method,
    {% else %}
    'contract_cutoff' as as_of_method,
    {% endif %}
    n_installment_batches
from inst_meta
