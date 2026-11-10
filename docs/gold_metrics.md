# Gold & Metrics Layer — Messy Client Data → Decisioning

> **All data is synthetic.** The client is a fictional Brazilian personal-credit fintech. Every number
> quoted in this document comes from a real `dbt build` over **synthetic fixture data** (fixed seed)
> and is labelled as such. Money figures also depend on **synthetic policy assumptions**.

Owner of this layer: Analytics Engineering. Bronze/silver, data generator, ML and API are owned by
Data Engineering; this layer consumes them only through `ref()` / `source()`.

## 1. Lineage

```
silver (DE)                         gold (AE)                       metrics (AE)                         dashboard
------------                        ---------                       ------------                         ---------
stg_installments ──► dim_snapshot ─┐
stg_loans ─────────────────────────┼► fct_loan_outcomes ─┬─► agg_vintage_default ──────────────────────► Vintages
stg_installments ──────────────────┘                     │
stg_applicants ───► dim_applicant                        │
stg_applications ─┬► fct_applications ─┬─────────────────┼─► agg_funnel_monthly ───────────────────────► Funnel
stg_loans ────────┘                    │                 │
source ml.predictions_scores ──────────┴► fct_scores ◄───┘─┬► agg_model_performance ───────────────────► Performance
                                                           ├► agg_calibration_deciles ─────────────────► Calibration
seed policy_assumptions ─► dim_policy_assumptions ────────┼► agg_policy_simulation (gold) ────────────► Policy curve
                                                           └► agg_model_comparison_at_approval_rate ───► Proof of Value
stg_quarantine + all stg_* row counts ───────────────────────► agg_data_quality ──────────────────────► Data quality
```

All models are materialized as tables (set in each model's `config()`). In the main project the
`dbt_project.yml` puts gold and metrics in schema `gold` (DuckDB schema `main_gold`); the dashboard
finds tables in whatever schema they land in. The scoring source `source('ml', 'predictions_scores')`
is declared by Data Engineering in `dbt/models/_sources.yml` (schema `ml`).

## 2. Models and grain

| Model | Layer | Grain | Notes |
|---|---|---|---|
| `dim_snapshot` | gold | 1 row | Observation cutoff `as_of_date`. Priority: `var('as_of_date')` → contract global cutoff `2026-10-31`. Per-batch `days_past_due` is not used to infer the cutoff. |
| `dim_policy_assumptions` | gold | 1 row | Synthetic assumptions pivoted from seed `policy_assumptions` (`assumption_key`, `value`), including LGD and funding cost. `net_interest_margin_annual` = gross margin − funding cost. |
| `dim_risk_bands` | gold | 1 row per `model_version ×` band A–D | Documented per-model PD edges from seed `policy_assumptions` (logistic A < 0.0407, B < 0.0734, C < 0.1525; GBM A < 0.0454, B < 0.0879, C < 0.1620; D = remainder). Not used to assign scores. |
| `dim_applicant` | gold | `applicant_id` | `age_band` (age at first application), `income_band` (monthly income, BRL), `has_missing_existing_debt` (rollup of the application-grain flag). `cpf` / `full_name` are not exposed. |
| `fct_applications` | gold | `application_id` | Decision flags, `has_loan`, `hours_to_decision`, `income_commitment_ratio`, `debt_to_annual_income` (NULL when `existing_debt` is NULL), `existing_debt_is_missing`. |
| `fct_loan_outcomes` | gold | `loan_id` | 12m default label, observability, EAD, installments paid/late/partial, amounts paid. |
| `fct_scores` | gold | `application_id × model_version` | ML scores plus the incumbent `bureau_score` policy (`ranking_key = −bureau_score`, higher bureau = lower risk). `risk_band` is read from `predictions_scores` (source of truth). |
| `agg_policy_simulation` | gold | `model_version × score_cutoff` (300–900, step 10) | OOT only. Approval, observed default, expected/observed economics. |
| `agg_funnel_monthly` | metrics | `application_month` | Applications → approved → booked. |
| `agg_vintage_default` | metrics | `vintage_month` (disbursement month) | 12m default rate on observable loans. |
| `agg_model_performance` | metrics | `model_version × split` | AUC, Gini, KS, Brier. Headline = OOT. |
| `agg_calibration_deciles` | metrics | `model_version × pd_decile` | OOT only. Mean predicted PD vs observed default. |
| `agg_model_comparison_at_approval_rate` | metrics | `model_version × target_approval_rate` (10%–90%, step 5) | OOT only. Proof of Value at equal approval rate, including `bureau_score` as the current policy. |
| `agg_risk_band_performance` | metrics | `model_version × risk_band` (A–D) | OOT observable only. Volume and observed default rate per band (`risk_band` from the score source). |
| `agg_data_quality` | metrics | `source_entity × batch_id × rule_failed` | `check_type` is `quarantine` or `completeness`. Completeness includes the missing-`existing_debt` rate (`rule_failed = existing_debt_missing`). |

## 3. Default definition (shared with the ML training label)

A loan is a **12-month default** when **any installment whose `due_date` is within 12 months of
disbursement** (`due_date <= disbursed_date + 12 months`) **reaches 90 or more days past due**.

Delinquency (`installment_dpd`) is computed from `due_date` and `paid_at` against the **global
observation cutoff** (`as_of_date`, contract default `2026-10-31`). The silver column
`days_past_due` is a per-batch snapshot (the cutoff of the batch that last loaded the row) and
is **never** used for the label.

* **fully paid** (`paid_at` not null and `amount_paid >= amount_due`): `paid_date − due_date` in days (floored at 0) — a
  payment made 90+ days late still makes the loan a default (it cured afterwards);
* **not fully paid** (unpaid or partial payment) and already due: `as_of_date − due_date`;
* **not yet due**: 0.

The default event is the **first** qualifying installment (lowest `installment_number`):
`default_installment_number`, `default_installment_due_date`, `default_date = due_date + 90 days`.
Delinquency that starts on installments due **after month 12** is **not** a 12m default.

### Observability (maturity)

The label is final only when `as_of_date >= disbursed_date + 12 months + 90 days` (`is_observable`).

| `outcome_status` | Condition | `is_default_12m` |
|---|---|---|
| `default` | observable and a default event exists | 1 |
| `good` | observable and no default event | 0 |
| `immature` | window not complete at `as_of_date` | NULL |

Immature loans are **never counted as good** and are excluded from every default-rate denominator.
`has_default_event_to_date` flags immature loans that already defaulted (monitoring only; including
them in rates would bias immature vintages upward).

### Exposure at default (EAD)

Scheduled outstanding principal immediately before the defaulting installment `k`, using French
(Price) amortization: `EAD = P(1+r)^(k−1) − A·((1+r)^(k−1) − 1)/r` (`r = 0` → `P − A(k−1)`), clipped to
`[0, principal]`. It assumes regular amortization before the delinquency; it does not add accrued
interest or fees.

## 4. Metric definitions

| Metric | Numerator | Denominator | Filters |
|---|---|---|---|
| `approval_rate` (funnel) | approved applications | approved + declined | cancelled excluded (no credit decision) |
| `take_up_rate` | applications with a loan | approved applications | — |
| `end_to_end_rate` | applications with a loan | all applications | — |
| `median_hours_to_decision` | median(`decided_at − applied_at`) in hours | — | — |
| `default_rate_12m` (vintage) | loans with `outcome_status = 'default'` | loans with `is_observable` | grouped by disbursement month |
| `default_to_date_rate` | loans with a default event so far | all loans of the vintage | monitoring only, not comparable |
| `ead_default_rate_by_value` | EAD of defaulted loans | principal of observable loans | — |
| `auc` | Mann–Whitney: `(Σ ranks of bads by pd − n_bad(n_bad+1)/2)` | `n_bad · n_good` | observable outcomes; ties get the average rank |
| `gini` | `2·AUC − 1` | — | same |
| `ks` | `max |cum% bads − cum% goods|` over distinct pd thresholds (pd desc) | — | same |
| `brier_score` | Σ `(pd − is_default_12m)²` | observable outcomes | same |
| calibration `observed_default_rate` | defaults in the decile | observable outcomes in the decile | OOT, deciles of pd (`ntile(10)`) |
| `approval_rate` (policy) | OOT applications with `score >= cutoff` | OOT applications scored by that model | `split = 'oot'` |
| `observed_default_rate` (policy / PoV) | observable defaults among policy-approved | observable outcomes among policy-approved | OOT |
| `observed_outcome_coverage` | observable outcomes among policy-approved | policy-approved | OOT — size of the reject-bias gap |
| `reject_rate` (DQ) | quarantined rows of a table × batch | quarantined + accepted silver rows | `source_table` normalized (`stg_`/`brz_`/`raw_` prefix removed) |

## 5. Policy simulation and assumptions

Policy rule: **approve when `score >= cutoff`**. Evaluated on the **OOT split only**, so train and
validation scores never inflate the result.

Assumptions live in `dbt/seeds/policy_assumptions.csv` (owned by Analytics Engineering). All of them
are **synthetic** and clearly labelled as such in the seed. The same seed also documents the
**per-model** A–D risk-band PD edges (`risk_band_*_pd_upper` plus `model_version`) for validation
only — `fct_scores.risk_band` is read from `predictions_scores`. This layer uses the economic keys:

| assumption_key | value | how it is used |
|---|---|---|
| `gross_interest_margin_annual` | 0.22 | synthetic annual GROSS margin |
| `funding_cost_annual` | 0.12 | synthetic annual funding cost; `net_interest_margin_annual` = gross − funding (0.10) |
| `lgd` | 0.65 | synthetic loss given default, as a fraction of exposure; `expected_loss = PD × EAD × LGD` |
| `acquisition_cost_brl` | 180 | per booked / policy-approved loan |
| `underwriting_cost_brl` | 25 | per evaluated OOT application (expected view only; the same at every cutoff) |

The lifetime margin is an **average-balance approximation of an amortized loan**:
`amount × net_margin × (term_months + 1) / 24`. That is the same as applying the net annual
margin to the average outstanding principal of a fully amortizing loan,
`principal × (n+1)/(2n)`, for `n/12` years. It is an approximation, not a cash-flow NPV.

This version uses the seed gross margin minus the seed funding cost, **not** the contractual
`interest_rate_monthly` from `stg_loans`. A future version can replace the gross margin with the
loan's real rate (for example `12 × interest_rate_monthly`, or an effective annual rate) minus
the same synthetic funding cost from the seed. `recovery_lag_months` and `discount_rate_annual`
are not used yet, because there is no discounting in this version.

* **Expected (model-based)**, over all policy-approved OOT applications, with `requested_amount` as the EAD proxy:
  `expected_loss = Σ pd·amount·LGD` (PD × EAD × LGD);
  `expected_net = Σ (1−pd)·amount·net_margin·(term+1)/24 − Σ pd·amount·LGD − n_approved·acq_cost − n_scored·underwriting_cost`.
  Each model grades itself with its own PD, so expected values are **not** comparable across models.
* **Observed**, only on approved + booked + observable applications:
  `observed_net = Σ principal_good·net_margin·(term+1)/24 − Σ EAD_default·LGD − n_observed·acq_cost`, and
  `observed_net_result_per_loan_brl` = the same per observed loan. Use the per-loan value when the
  models have different coverage.
* **Proof of Value** (`agg_model_comparison_at_approval_rate`): each policy approves the **same number**
  of through-the-door OOT applications (top-k by score, ties broken by lower pd, then `application_id`).
  This includes the incumbent `bureau_score` policy (higher bureau = lower risk; no calibrated PD, so
  expected-loss figures are null). Policies are compared on observed default rate and observed net per
  loan on the booked-and-matured subset (reject bias). The dashboard also shows the published
  approval-matched bad rates from `docs/model_metrics.json`, clearly labelled as the **observable OOT**
  population (a different, smaller set than the through-the-door PoV). Evaluation of performance
  metrics (AUC/KS/Brier) uses only observable OOT outcomes; the `split = 'oot'` label in
  `predictions_scores` also covers declined and immature applications, which enter the through-the-door
  policy comparison but not the performance denominator. The published paired bootstrap for GBM minus
  bureau has CIs that include 0; this layer does not claim that GBM outranks the bureau.

### Fixture run results (synthetic fixture data — illustrative only)

Run on 2026-11-10 against the dev fixture set (seed 20261110; cutoff 2026-10-31). Numbers below are refreshed after the alignment build:

* Funnel: 8,000 applications → 4,075 approved / 3,600 declined / 325 cancelled → 3,579 booked
  (approval rate 53.1%, take-up 87.8%).
* Loans: 1,759 observable, 185 defaults (12m default rate 10.5%); 1,820 immature. The last
  observable disbursement is 2025-08-02; loans disbursed from 2025-08-03 onward are immature
  under the contract cutoff 2026-10-31.
* OOT performance (679 observable outcomes, 70 bads): `gbm_v1` AUC 0.674 / Gini 0.348 / KS 0.319;
  `baseline_logreg_v1` AUC 0.646 / Gini 0.291 / KS 0.261.
* PoV at a 50% approval rate (2,806 of 5,611 OOT applications): observed default rate 7.0% for
  `gbm_v1` (35/503) vs 7.0% for `baseline_logreg_v1` (30/431); observed net per observed loan
  R$ −4.31 vs R$ 27.72 (synthetic assumptions, net annual margin 10% after a 12% funding cost).
  Coverage differs between models, so the per-loan figure is the comparable one; the drop versus
  the previous fixture run is mostly the new net-of-funding margin (0.10 instead of treating
  0.22 as already net).

These numbers only show that the pipeline works end to end. The fixture generator builds `gbm_v1`
to be less noisy than the baseline, so the gap is **by construction**. None of this is evidence
about any real portfolio.

## 6. Limitations

* **Reject bias.** Outcomes exist only for applications the legacy policy approved **and** the
  customer booked **and** that matured. Policy and PoV results on the observed subset say nothing
  about applicants the legacy policy declined. `observed_outcome_coverage` shows how small that
  subset is. Reject inference is out of scope.
* **Synthetic data and assumptions.** Data, margin, LGD and acquisition cost are all synthetic.
* **Small samples.** At high cutoffs, observed rates rest on few loans. `agg_model_performance.is_sample_sufficient`
  requires at least 30 bads and 30 goods.
* **EAD approximation.** EAD uses the scheduled amortization and ignores accrued interest, fees and missed earlier payments.
* **Income commitment** uses the contract installment when a loan exists and `requested_amount / term_months`
  (interest-free) otherwise. `existing_debt` is the outstanding debt **stock** (total balance);
  `debt_to_annual_income = existing_debt / (12 × monthly_income)` and is **NULL** (never 0) when
  `existing_debt` is NULL. `existing_debt_is_missing` uses the silver flag when that column exists,
  otherwise it is derived from `existing_debt IS NULL`. Until silver stops coalescing missing debt
  to 0, those rows look like genuine zero balances and the missing-debt rate stays 0.
* **Per-batch DPD is ignored for the label.** Silver `days_past_due` is as of the batch that last
  loaded the installment. The label always uses `due_date` / `paid_at` versus the global cutoff.
  `dim_snapshot.n_installment_batches` still reports how many installment batches reached silver.
* **Margin approximation.** Policy P&L uses the average-balance formula with the seed net margin
  (gross − funding cost), not the contractual `interest_rate_monthly`.

## 7. How to run (README)

Main project (once silver and `ml.predictions_scores` exist):

```bash
cd dbt
dbt seed --profiles-dir .
dbt build --profiles-dir .
# optional fixed cutoff: --vars '{as_of_date: "2026-09-30"}'

cd ..
pip install -r dashboard/requirements.txt
MC_DUCKDB_PATH=data/warehouse.duckdb streamlit run dashboard/app.py
```

The dashboard reads only gold/metrics tables, opens short-lived **read-only** DuckDB connections
(so `dbt build` can run while it is open), finds tables in any schema (`MC_DUCKDB_SCHEMA` forces
one), and shows a permanent "synthetic data" banner.

Tests: generic tests (unique, not_null, relationships, accepted_values) in `_gold.yml` / `_metrics.yml`,
plus singular reconciliation tests in `dbt/tests/gold/`: principal silver = gold, no row
multiplication, immature loans excluded from default denominators, default only within the 12m
window, observed/matured outcomes respect `disbursed_at + 12 months + 90 days <= 2026-10-31`
(FDDS R1), rates in [0, 1], monotonic approval rate vs cutoff, outcomes only for booked applications,
and seed per-model A–D thresholds reproducing `predictions_scores.risk_band` exactly.
None of the tests hardcode fixture counts, rates, or other sample-specific numbers — they are
structural / relational, so they remain valid when the M2 generator changes first-miss timing
and the share of immature loans, and when `existing_debt` starts arriving as NULL. A singular
test checks that missing debt is not coerced to 0 in `debt_to_annual_income`. Score-dependent
models (`fct_scores`, performance, calibration, policy, PoV) are empty until M3 and the tests
pass on zero rows without division by zero; the dashboard shows "scores pending (M3)" in those
sections. Generic tests use the pre-1.10 `accepted_values: {values: ...}` syntax, so they stay
compatible with `dbt-duckdb>=1.8` (dbt ≥ 1.10 prints a deprecation warning).

Dev fixture project (`portfolio/_ae-gold-dev/`, not part of the deliverable):

```bash
python scripts/generate_fixtures.py
DBT_PROFILES_DIR=. dbt seed && DBT_PROFILES_DIR=. dbt build
MC_DUCKDB_PATH=data/warehouse.duckdb streamlit run dashboard/app.py
python scripts/screenshot_dashboard.py   # headless smoke test + screenshots/
```
