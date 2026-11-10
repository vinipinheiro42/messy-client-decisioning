# Interface contract — Messy Client Data → Decisioning

This document records the agreed contract between Data Engineering (bronze/silver, generator, scoring table schema) and Analytics Engineering (gold, metrics, dashboard). It applies to the **fictional** client **Credara Fintech**. Every figure, rate, and monetary value produced in this project is computed on **synthetic data**.

Bronze and silver models are implemented in milestone M2 (Data Engineering). Gold and metrics models already exist and are owned by Analytics Engineering. This file is the shared interface those layers follow.

## Platform

| Item | Value |
|------|--------|
| Adapter | dbt-duckdb |
| Profile name | `messy_client` |
| Warehouse path | `data/warehouse.duckdb` (relative to the repo root; in `dbt/profiles.yml.template` this is `../data/warehouse.duckdb`) |
| Model prefixes | bronze `brz_`, silver `stg_`, gold `fct_` / `dim_` / `agg_` |
| Money | BRL `decimal(18, 2)` |
| Dates / timestamps | `date` or `timestamp` in UTC |

## Default definition

A loan is in default if it reaches **90 or more days past due** on an installment **due on or before `disbursed_at + 12 months`**.

- Owner: gold model `fct_loan_outcomes`.
- The same definition is the training label for the credit model.
- The 90th day past due may fall after the 12-month anniversary (for example a first missed installment in month 10 reaches 90 DPD around month 13). That is still a 12-month default, which is why a loan is **observable** only when `disbursed_at + 12 months + 90 days <= observation cutoff` (2026-10-31 UTC). The extra 90 days let a miss on the month-12 installment reach 90 DPD.
- A delinquency that starts on an installment **due after** `disbursed_at + 12 months` is **not** a 12-month default, even if it later reaches 90 DPD.
- In the generator, loans that are intended to go delinquent draw a first missed installment over months `1..min(12, term)` with **extra weight on months 3–6**, so default timing has a tail into months 9–12 rather than stopping at month 8.
- Non-observable loans carry `outcome_status = 'immature'` and are excluded from realised default rates and from the training label. `outcome_status` takes exactly the values `default`, `good`, and `immature`.
- The out-of-time (OOT) **labelled** evaluation set contains only approved, observable loans (6,022 in this warehouse); its last disbursement date is recorded in the model card. The `split = 'oot'` label in `ml.predictions_scores` is broader — see the scoring-output section.
- Delinquency for the label is computed from `due_date` and `paid_at` against the global observation cutoff, not from the per-batch `days_past_due` snapshot.
- Realised default rates must be labelled as **synthetic**.

## Ground truth vs raw exports

- `data/ground_truth/` holds the clean generative output (Parquet plus a generation summary). It is the measurement baseline for cleaning, default labelling, and later model quality.
- `data/raw/batch_00N/` holds messy CSV exports, one file per entity (`applicants`, `applications`, `loans`, `installments`).
- `data/raw/_injected_issues.csv` lists every injected defect: `batch`, `file`, `row_key`, `issue_type`, `column`.

CPF-like values are 11-digit strings with **deliberately invalid check digits**. They are not valid Brazilian CPFs. See `scripts/generate_data.py` and `docs/data_issues_catalog.md`. Silver standardises CPF formatting to 11 digits and validates **format only** (exactly 11 digits after stripping punctuation). Silver must **not** quarantine a row because the check digits are invalid; every synthetic CPF is invalid by design.

### Counterfactual outcomes (ground truth only)

`data/ground_truth/counterfactual_outcomes.parquet` holds a **synthetic** potential outcome for every application that did not originate a loan: declined, cancelled, and approved-but-not-originated. Columns include `application_id`, `decision`, hypothetical `principal` / monthly interest rate / term, and `is_default_90dpd_12m_counterfactual`. The flag is drawn from the **same default model** used for originated loans. These rows exist only for later selection-bias analysis. They must **never** appear in `data/raw/` or in bronze/silver tables built from the client exports.

## Silver tables (Data Engineering, M2)

Every silver table carries lineage columns: `batch_id`, `source_file`, `ingested_at`.

Deduplication of resends uses the latest `ingested_at` (mapped from the raw export timestamp `extracted_at` when present).

### `stg_applicants`

- Grain: **1 row per applicant**.
- Primary key: `applicant_id`.
- Canonical attributes after standardisation: `cpf`, `full_name`, `birth_date`, `state_uf`, `monthly_income`, `employment_type`, `employment_months`, `housing_status`, `dependents`.
- `cpf` remains a synthetic 11-digit identifier (invalid check digits); silver standardises formatting only.
- Outliers and unparseable values are either fixed (when the rule is deterministic and safe) or sent to quarantine.

### `stg_applications`

- Grain: **1 row per application**, deduplicated by latest `ingested_at`.
- Primary key: `application_id`.
- Foreign key: `applicant_id` → `stg_applicants`.
- Canonical attributes: `applied_at`, `requested_amount`, `term_months`, `purpose`, `channel`, `bureau_score`, `existing_debt`, `existing_debt_is_missing`, `decision`, `decided_at`.
- `decision` is one of `approved`, `declined`, `cancelled`.
- `existing_debt` is the applicant's **total outstanding debt balance in BRL** at application time, not a monthly payment. It is **nullable**. Silver does **not** impute a missing or disguised-null balance as 0, because a false zero would bias credit-risk models toward "no debt". `existing_debt_is_missing` is true when `existing_debt` is null (the surviving raw version had no usable value). A genuine zero balance is stored as 0 with the flag false. Optional applicant fields (`housing_status`, `employment_months`, `dependents`) are likewise left null when missing; they are not filled with zeros.

### `stg_loans`

- Grain: **1 row per loan**.
- Primary key: `loan_id`.
- Foreign key: `application_id` → `stg_applications`.
- Cardinality: **0 or 1 loan per application** (never more than one disbursement per application).
- Canonical attributes: `disbursed_at`, `principal`, `interest_rate_monthly`, `term_months`, `installment_amount`.

### `stg_installments`

- Primary key: `installment_id`.
- Unique key: `(loan_id, installment_number)`.
- Foreign key: `loan_id` → `stg_loans`.
- Columns: `due_date`, `amount_due`, `paid_at` (nullable), `amount_paid`, `days_past_due` **as of the batch cutoff** that last loaded the row.
- `days_past_due` is 0 when the installment is not yet due or has been paid as of the cutoff; otherwise `(cutoff_date - due_date)` in days.
- `amount_paid` is 0 when the installment is unpaid (`paid_at` is null). Unpaid installments have a true zero payment, so 0 is the correct business value. A raw row that has `paid_at` filled but a missing or disguised-null `amount_paid` is an **inconsistent payment**: that version is quarantined (`inconsistent_payment`) so an earlier clean version can survive. Silver never coalesces `amount_paid` to 0 when `paid_at` is present, because a false zero would mark a paid installment as unpaid and inflate the default rate.
- All timestamps and calendar dates in bronze, silver, and gold are **UTC**. The DuckDB session time zone for this project is `UTC` (see `dbt/profiles.yml`). Casting a timestamp to `date` must use the UTC calendar day, not the host local zone (America/Sao_Paulo would shift times between 00:00 and 02:59 UTC to the previous day). Month arithmetic for the 12-month default window follows gold: `cast(disbursed_at as date) + interval 12 month` (and `+ interval 90 day` for observability), which matches pandas `DateOffset(months=12)` on month-end dates.

### `stg_quarantine`

Rows that fail a silver quality rule and cannot be safely repaired:

| Column | Meaning |
|--------|---------|
| `source_table` | Raw entity (`applicants`, `applications`, `loans`, `installments`) |
| `source_row_hash` | Hash of the raw payload (stable for a given raw row) |
| `batch_id` | Batch that provided the row |
| `rule_failed` | Rule identifier (see the issues catalogue) |
| `raw_payload` | JSON of the raw record |

## Scoring output (written by the ML pipeline, declared as a dbt source)

dbt source: `source('ml', 'predictions_scores')` (see `dbt/models/_sources.yml`).

Table: `predictions_scores`

| Column | Type / domain | Notes |
|--------|----------------|-------|
| `application_id` | text | FK to `stg_applications` |
| `model_version` | text | Immutable version string: `baseline_logreg_v1` or `gbm_v1` |
| `pd` | float, 0–1 | Calibrated probability of default (90+ DPD in 12 months) |
| `score` | int, 300–900 | Higher is better (lower PD) |
| `risk_band` | text, A–D | Model-specific band cut on calibrated PD (not on the score). Training starts from PD quintiles and merges thin or non-unique cuts; current models collapse to four bands. |
| `reason_codes` | JSON | Top-3 SHAP reason codes |
| `split` | text | `train`, `validation`, or `oot` |
| `scored_at` | timestamp UTC | When the score was produced |

Primary key: `(application_id, model_version)`.

`split = 'oot'` in `predictions_scores` covers **all applications whose `applied_at` falls in the OOT window** (13,720 rows per model in this warehouse), including declined, cancelled, and immature cases. That full slice is the right input for policy simulation. Labelled model metrics (AUC, KS, Brier, band bad rates, approval-rate-matched rates) use only the approved, observable subset (6,022 loans), i.e. originated loans with `disbursed_at + 12 months + 90 days <= 2026-10-31 UTC`.

## Gold and metrics tables (Analytics Engineering)

Built only on silver (and the scoring source). These models already exist under `dbt/models/gold/` and `dbt/models/metrics/` and are owned by Analytics Engineering. Names below are the models currently in those folders.

### Gold (`dbt/models/gold/`)

| Model | Grain / role |
|-------|----------------|
| `dim_snapshot` | One row: observation cutoff used for outcome maturity |
| `dim_policy_assumptions` | One row: synthetic policy assumptions pivoted from the AE-owned seed |
| `dim_applicant` | 1 row per applicant |
| `fct_applications` | 1 row per application |
| `fct_loan_outcomes` | 1 row per loan / contract, including `is_default_12m` and `outcome_status` (`default`, `good`, or `immature`) |
| `fct_scores` | 1 row per application × model version |
| `agg_policy_simulation` | 1 row per model version × score cutoff, with approval, default, and expected-loss metrics |

### Metrics (`dbt/models/metrics/`)

| Model | Role |
|-------|------|
| `agg_funnel_monthly` | Application funnel by month |
| `agg_vintage_default` | Vintage default rates |
| `agg_model_performance` | Model performance metrics |
| `agg_calibration_deciles` | Score calibration by decile |
| `agg_data_quality` | Data-quality rollup from silver and quarantine |
| `agg_model_comparison_at_approval_rate` | Model comparison at a fixed approval rate |

Policy economics used in simulations come from `dbt/seeds/policy_assumptions.csv`, which is **owned by Analytics Engineering** and is explicitly labelled **synthetic**. Among those assumptions, the current seed includes `lgd = 0.65` (loss given default as a fraction of exposure) and `funding_cost_annual = 0.12` (synthetic annual funding cost; net margin is the gross interest margin minus this funding cost). Do not treat these figures as market quotes or as Data Engineering inputs.

## Folder ownership

| Path | Owner |
|------|--------|
| `scripts/`, `data/`, `dbt/models/bronze/`, `dbt/models/silver/`, `ml/`, `api/` | Data Engineering |
| `dbt/models/gold/`, `dbt/models/metrics/`, `dbt/tests/gold/`, `dashboard/`, `docs/gold_metrics.md`, `dbt/seeds/policy_assumptions.csv` | Analytics Engineering |
| `review/fdds-review.md`, `docs/proof-of-value.md` | Forward Deployed Data Science review (PoV co-written with Data Engineering) |
| `docs/contract.md` | Shared; changes should be agreed |

Publication to GitHub (or any remote) requires explicit approval from the project owner and coordination through the job-search coordinator. This repository is initialised locally only.
