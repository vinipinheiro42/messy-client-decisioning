# Data issues catalogue

Injected defects in the messy Credara Fintech (fictional) CSV exports, plus the expected silver-layer handling. Rates below are **target injection rates** used by `scripts/generate_data.py`. Actual counts for a given run are in `data/ground_truth/generation_summary.json` and `data/raw/_injected_issues.csv`.

The issues log columns are: `batch`, `file`, `row_key`, `issue_type`, `column`.

Ground truth in `data/ground_truth/` does **not** contain these defects (except that CPFs are still synthetic identifiers with invalid check digits, by design).

Silver table grains, keys, types, and lineage columns are defined in [`contract.md`](contract.md). This catalogue repeats the contract briefly at the end so it can be read on its own.

## Issue types

### `exact_duplicate`

- **How injected:** A raw row is written twice in the same batch, or a previously exported row is resent unchanged in a later batch (`extracted_at` is later). Identifiers are identical.
- **Target rate:** about 1.5% of applicant, application, and loan rows are resent or duplicated; a smaller share of installments.
- **Expected silver handling:** **Fix** — collapse to one row, keeping the record with the latest `ingested_at`. Do not quarantine a clean resend.

### `near_duplicate`

- **How injected:** A resend of the same business key (`applicant_id` or `application_id`) with a small payload change (for example a typo in `full_name`, a rounding change in `monthly_income` or `requested_amount`, or a different date format).
- **Target rate:** about 0.8% of applicant and application rows.
- **Expected silver handling:** **Standardize** then **fix** — treat as the same entity, prefer the latest `ingested_at`, and record the conflict if canonical fields disagree. Quarantine only if the conflict cannot be resolved by rule.

### `malformed_cpf`

- **How injected:** The already-invalid 11-digit synthetic CPF is formatted inconsistently: `123.456.789-00`, `12345678900`, `123 456 789 00`, `123.456.789/00`, or with missing punctuation. Check digits remain invalid; no valid Brazilian CPF is generated.
- **Target rate:** about 12% of applicant rows in the raw exports (formatting only; every CPF is already a non-valid identifier).
- **Expected silver handling:** **Standardize** to 11 digits (digits only). Rows that do not contain 11 digits after stripping punctuation are **quarantined**. Do **not** quarantine because the check digits are invalid: every synthetic CPF is generated with deliberately invalid check digits.

### `mixed_date_format`

- **How injected:** Date and timestamp columns are written in a mix of `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, `YYYY/MM/DD`, `DD.MM.YYYY`, and timestamps with a trailing `Z` or a space-separated time.
- **Target rate:** about 25% of date values in raw files use a non-ISO format (ISO values are also present in the same file).
- **Expected silver handling:** **Standardize** to UTC `date` / `timestamp`. Unparseable values are **quarantined**.

### `currency_as_text`

- **How injected:** Monetary columns are written as text in mixed conventions: `R$ 1.234,56` (Brazilian), `1,234.56` (US-style thousands), `1234.56`, or `R$1234,56`.
- **Target rate:** about 40% of money values in raw files.
- **Expected silver handling:** **Standardize** to `decimal(18, 2)` BRL. Ambiguous or unparseable strings are **quarantined**.

### `disguised_null`

- **How injected:** Missing values are written as `N/A`, `-`, `""`, `NULL`, `none`, or as `0` in columns where zero is not a valid value (for example `monthly_income` or `requested_amount`).
- **Target rate:** about 3% of selected attribute cells (not identifier columns).
- **Expected silver handling:** **Standardize** to SQL `NULL`. If the column is required for the silver grain (for example a missing `applied_at` on an application), **quarantine** the row.

### `inconsistent_category`

- **How injected:** Canonical categories are replaced with case variants, accent variants, typos, or synonyms (for example `CLT` / `carteira assinada` / `celetista`; `SP` / `Sao Paulo` / `são paulo`; `app` / `aplicativo` / `mobile`).
- **Target rate:** about 20% of category values (`employment_type`, `housing_status`, `state_uf`, `purpose`, `channel`, `decision`).
- **Expected silver handling:** **Standardize** via a mapping table to the canonical values listed below. Values that do not map are **quarantined**.

Canonical values in ground truth and silver:

| Column | Canonical values |
|--------|------------------|
| `employment_type` | `clt`, `self_employed`, `public_servant`, `informal`, `unemployed` |
| `housing_status` | `owner`, `renter`, `family`, `financed` |
| `purpose` | `personal`, `debt_refinance`, `home_improvement`, `education`, `medical`, `other` |
| `channel` | `app`, `web`, `branch`, `partner` |
| `decision` | `approved`, `declined`, `cancelled` |
| `state_uf` | Official two-letter Brazilian UF codes |

### `late_arriving_event`

- **How injected:** A subset of installments whose `due_date` falls in an earlier batch window is withheld from that batch and written only in a later batch (typically `batch_003`).
- **Target rate:** about 4% of installments that were due in the `batch_001` or `batch_002` windows.
- **Expected silver handling:** **Fix** — append the late row, recompute `days_past_due` as of the new batch cutoff, and keep the latest payload for `(loan_id, installment_number)`. Do not treat late arrival by itself as a quarantine reason.

### `schema_change`

- **How injected (file-level, logged with `row_key=__FILE__`):**
  - `batch_002` uses a **semicolon** delimiter and **ISO-8859-1** encoding.
  - `batch_002` **renames** `monthly_income` → `income_monthly` (applicants) and `requested_amount` → `amount_requested` (applications).
  - `batch_002` **adds** `occupation_note` on applicants.
  - `batch_003` restores the original column names, stays comma-delimited UTF-8, and **adds** `customer_segment` on applicants plus `source_system` on applications.
- **Target rate:** applies to the listed files as a whole (not a row sample).
- **Expected silver handling:** **Standardize** in bronze: one canonical schema per entity, explicit column mapping, and explicit delimiter / encoding per batch. Do not fail the batch because of a known rename or added column.

### `outlier`

- **How injected:** Implausible values are written on a small number of otherwise well-keyed rows: negative `monthly_income`, absurd `birth_date` (age under 16 or over 100 at application time), or an extreme `requested_amount`.
- **Target rate:** about 0.4% of applicant rows (income / age) and a smaller share of applications (amount).
- **Expected silver handling:** **Quarantine** (do not silently clip in silver). Gold may later apply policy caps, but silver preserves the fact that the raw value is not trustworthy.

### `orphan_fk`

- **How injected:** Raw application rows whose `applicant_id` does not exist in any applicants file, and raw installment rows whose `loan_id` does not exist in any loans file. These rows are **not** present in ground truth.
- **Target rate:** a small absolute count (tens of rows at default scale), enough to test referential-integrity tests.
- **Expected silver handling:** **Quarantine**. Do not create implied parent records.

## Silver handling summary

| `issue_type` | Fix | Standardize | Quarantine |
|--------------|-----|-------------|------------|
| `exact_duplicate` | yes (latest `ingested_at`) | | only if the surviving row fails another rule |
| `near_duplicate` | yes (latest `ingested_at`) | yes | if a required field conflicts irreconcilably |
| `malformed_cpf` | | yes (digits-only) | if not 11 digits after cleaning |
| `mixed_date_format` | | yes (UTC) | if unparseable |
| `currency_as_text` | | yes (BRL decimal) | if unparseable or ambiguous |
| `disguised_null` | | yes (to `NULL`) | if the field is required |
| `inconsistent_category` | | yes (mapping) | if unmapped |
| `late_arriving_event` | yes (append / upsert) | | no, not by itself |
| `schema_change` | | yes (bronze mapping) | no, not by itself |
| `outlier` | | | yes |
| `orphan_fk` | | | yes |

## Silver table contract (abridged)

See [`contract.md`](contract.md) for the full text. In brief:

- `stg_applicants`: 1 row per applicant, PK `applicant_id`.
- `stg_applications`: 1 row per application (dedup by latest `ingested_at`), PK `application_id`, FK `applicant_id`. `existing_debt` is nullable; `existing_debt_is_missing` flags a missing or disguised-null balance. Missing balances are not imputed as 0.
- `stg_loans`: 1 row per loan, PK `loan_id`, FK `application_id`, 0 or 1 loan per application.
- `stg_installments`: PK `installment_id`, unique `(loan_id, installment_number)`; `due_date`, `amount_due`, `paid_at`, `amount_paid`, `days_past_due` as of batch cutoff. `amount_paid` is 0 only for unpaid installments (`paid_at` is null). A disguised null or missing `amount_paid` on a row that has `paid_at` is quarantined as `inconsistent_payment` so a previous clean version can be kept.
- `stg_quarantine`: `source_table`, `source_row_hash`, `batch_id`, `rule_failed`, `raw_payload`.
- All silver tables carry `batch_id`, `source_file`, `ingested_at`.
- Money is BRL `decimal(18, 2)`; dates and timestamps are UTC.

## Generative process (ground truth)

Documented in full in the `scripts/generate_data.py` module docstring. Short version:

1. Each applicant has a **latent risk** factor driven by income, employment type, age, housing status, dependents, and a synthetic state effect, plus noise.
2. Applications inherit that risk. Bureau score is a noisy inverse function of latent risk. Requested amount, term, purpose, channel, and existing debt are drawn conditionally.
3. A synthetic underwriting policy approves, declines, or cancels the application.
4. Most approved applications originate a loan. Installment amounts use a standard fixed-rate payment formula.
5. Payment paths are simulated so that the realised **90+ DPD within 12 months of disbursement** rate among loans with a complete observation window falls in an **8–15%** range. That rate is a **synthetic** realised default rate.

The dataset is entirely synthetic and is not based on any real lender's customers or portfolios.

## Default label and first-miss timing

These rules sit alongside the issues above and are recorded in [`contract.md`](contract.md).

- A loan is in 12-month default if any installment **due on or before `disbursed_at + 12 months`** reaches 90+ days past due. The 90th day may fall after the 12-month anniversary (for example a first miss in month 10). That is still a 12-month default.
- A delinquency that starts on an installment **due after** `disbursed_at + 12 months` is **not** a 12-month default, even if it later reaches 90 DPD.
- A loan is observable only when `disbursed_at + 12 months + 90 days <= 2026-10-31 UTC`. Non-observable loans are `immature` and are excluded from realised default rates.
- In the generator, intended delinquencies draw the first missed installment over months `1..min(12, term)` with extra weight on months 3–6, so some defaults start in months 9–12.
- Counterfactual outcomes for declined, cancelled, and approved-but-not-originated applications live only in `data/ground_truth/counterfactual_outcomes.parquet` (synthetic, for selection-bias analysis) and are never written to `data/raw/`.
