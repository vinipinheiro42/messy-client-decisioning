{{ config(materialized='table') }}
-- Raw batch CSVs are pre-loaded by scripts/load_bronze.py, which applies
-- the correct delimiter and encoding per batch and maps renamed columns
-- (income_monthly -> monthly_income, amount_requested -> requested_amount).
-- Values are still messy; silver performs parsing and quarantine.
select * from {{ landing_parquet('applications') }}
