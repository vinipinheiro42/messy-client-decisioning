# Modeling plan — Messy Client Data → Decisioning (M3)

This plan is for the **fictional** client **Credara Fintech**. Every figure, rate, and
score described here is computed on **synthetic data**. It is not a live credit
policy and is not based on any real lender.

The training set is built from the corrected silver and gold tables after the
Part A fixes (nullable `existing_debt`, no imputed zero `amount_paid` when
`paid_at` is present, DuckDB session `TimeZone = UTC`). Labels are read from
`main_gold.fct_loan_outcomes` so Data Engineering and Analytics Engineering
share a single source of truth.

## Objective

Predict the 12-month default label `is_default_12m` (also called
`is_default_90dpd_12m`) at application time, for use as a calibrated PD, a
300–900 score, an A–E risk band, and top-3 adverse-action style reason codes.

## Label

- **Definition:** a loan is in default if any installment **due on or before**
  `cast(disbursed_at as date) + interval 12 month` reaches 90 or more days past
  due. Delinquency is computed from `due_date` and `paid_at` against the global
  cutoff **2026-10-31 UTC**, never from per-batch `days_past_due`. A miss that
  starts on an installment due after the 12-month window is not a 12-month
  default.
- **Source:** `main_gold.fct_loan_outcomes.is_default_12m`, restricted to
  `is_observable = true` (`outcome_status` in `default` / `good`). Immature
  loans are excluded from training, validation, calibration, and labelled
  evaluation.
- **Observability:** `cast(disbursed_at as date) + interval 12 month + interval 90 day <= 2026-10-31`.
  In this warehouse the last observable disbursement date is **2025-08-02**
  (the next calendar day, 2025-08-03, is the first immature disbursement).

## Population and two evaluation sets

1. **Main evaluation set — approved and observable.** Originated loans whose
   observation window is complete. This is the only population used for
   fitting, early stopping, and calibration.
2. **Through-the-door evaluation (selection-bias analysis only).** Every
   scored application, attaching `data/ground_truth/counterfactual_outcomes.parquet`
   for declined, cancelled, and approved-but-not-originated rows. These
   potential outcomes are **synthetic**, were never in `data/raw`, and are
   **never used for training, tuning, or calibration**. The report compares
   metrics on (1) versus (2) so the effect of underwriting selection is visible.

## Features (application-time only, FDDS R3)

Only information that would have been known at `applied_at` is used.

**Application:** `requested_amount`, `term_months`, `purpose`, `channel`,
`bureau_score`, `existing_debt`, `existing_debt_is_missing`.

**Applicant:** `monthly_income`, `employment_type`, `employment_months`,
`housing_status`, `dependents`, `state_uf`, and `age_at_applied_at` (from
`birth_date` and `applied_at`).

**Ratios:**

- `requested_to_income = requested_amount / monthly_income`
- `debt_to_annual_income = existing_debt / (12 * monthly_income)`
- `estimated_dti`: estimated monthly commitment over monthly income. The
  installment is an annuity on `requested_amount` and `term_months` at a
  **fixed synthetic reference rate of 2.0% per month**. Outstanding debt is
  serviced at a **fixed 4% per month of the outstanding stock**. Neither
  rate is the contract `interest_rate_monthly`. If `existing_debt` is
  missing it contributes 0 to the ratio and `existing_debt_is_missing` is
  the flag.

**History, strictly before `applied_at`, with a fixed 12-month lookback:**

- `prior_application_count`: prior applications of the same applicant with
  `applied_at` in the 12 months before this application.
- `prior_decline_count`: the same window, restricted to declines.
- `prior_observable_default_count` is **not used**. A lifetime count of
  already-observable prior defaults is non-zero in ~0.08% of train vs
  ~4.5% of OOT (history accumulates; value-bin PSI is material, decile
  PSI hides it) and the logistic coefficient had the wrong sign. A
  12-month lookback cannot contain a completed 12m+90d outcome, so the
  feature would be structurally zero under the same window. It remains
  in the PSI table as a diagnostic lifetime series only.

OOT AUC before this change (lifetime history, feature included) was
0.7208 (logreg) and 0.7492 (GBM). The model card reports the AUC after
retraining.

**Forbidden (leakage):** `principal`, `interest_rate_monthly`,
`installment_amount`, `decision`, `decided_at`, `disbursed_at`, any
installment field, post-decision status, and any history at or after
`applied_at`. A leakage test in `tests/` asserts the feature list and
recomputes history windows.

### Expected direction and monotone constraints

| Feature | Expected effect on PD | LightGBM monotone |
|---------|------------------------|-------------------|
| `bureau_score` | down | −1 |
| `monthly_income` | down | −1 |
| `employment_months` | down | −1 |
| `existing_debt` | up | +1 |
| `requested_to_income` | up | +1 |
| `debt_to_annual_income` | up | +1 |
| `estimated_dti` | up | +1 |
| `dependents` | up | +1 |
| `prior_decline_count` | up | +1 |
| `age_at_applied_at` | U-shaped (no hard constraint) | none |
| `requested_amount`, `term_months`, counts, flags, categoricals | mixed or absorbed in the category | none |

Reason codes are checked against these directions: a code may only be
emitted when the contribution **increases** risk, and those contributions
must not contradict the expected sign.

## Time split (by `applied_at`)

All applications, including declined and cancelled, receive a split tag
from `applied_at`. **OOT never enters fitting, early stopping, tuning, or
calibration.**

| Split | `applied_at` window (UTC) | Role |
|-------|---------------------------|------|
| Train | `< 2024-07-01` | Fit both models |
| Validation (tune) | `2024-07-01` inclusive to `< 2024-10-01` | Early stopping / light tuning only |
| Validation (calibration) | `2024-10-01` inclusive to `< 2025-01-01` | Held-out calibration only (isotonic) |
| Out-of-time (OOT) | `>= 2025-01-01` | Frozen evaluation. Labelled OOT loans have disbursement on or before **2025-08-02**. Later applications are still scored and tagged `oot` but have no training label. |

In `ml.predictions_scores`, the schema allows only `train`, `validation`, or
`oot`. The two validation slices are both stored as `validation`. The
internal tune/calibration cut is documented here and in the model card.

Approximate labelled sizes on the current warehouse (synthetic, observable
loans): train 15,634 (default rate 10.94%), tune 2,679 (10.79%),
calibration 2,616 (10.05%), OOT 6,022 (11.09%).

## Models

1. **`baseline_logreg_v1`.** Interpretable logistic regression. Numeric
   features are standardised with train-only mean and scale. Categorical
   features are one-hot encoded with a train-only vocabulary and an
   explicit missing level. This is a standardised (not WoE) specification.
2. **`gbm_v1`.** LightGBM with the monotone constraints in the table above,
   early-stopped on the **tune** slice (not the calibration slice, not OOT).
3. **Current-policy comparator (not a fitted model):** raw `bureau_score`.
   Higher bureau score is treated as lower risk. It is **not** written to
   `ml.predictions_scores`. Its AUC, KS, and approval-rate-matched bad
   rates are reported next to the two fitted models.

**Calibration.** Isotonic regression of the raw model score onto the
label, fitted **only** on the calibration slice. If a model used the tune
slice for early stopping, that slice is not reused for calibration. OOT
is not used. The isotonic mapping itself is bounded to
`(1e-6, 1 - 1e-6)`. Immediately before the PDO score map, PD is
clipped to `(1e-12, 1 - 1e-12)` so that a PD of 0 or 1 reaches the
300–900 score floor and cap.

## Score mapping (PDO) and risk bands

Higher score is better (lower PD).

- **PDO** = 20 (the score increases by 20 points when the good:bad odds double).
- **Anchor:** score 600 corresponds to odds 50:1, i.e. PD = 1/51 ≈ 1.9608%.
- `factor = PDO / ln(2)`
- `offset = 600 - factor * ln(50)`
- `score = round(offset - factor * ln(PD / (1 - PD)))`, clipped to **300–900**.

Risk bands are cut on **calibrated PD**, not on the score, and are
**model-specific**. For each model the cut points are the quintiles of
its calibrated PD on the OOT approved-and-observable population, merged
if needed so that each band holds at least about 5% of that population.
Shared PD cuts are not used, because isotonic plateaus can leave one
model with an empty band. Exact thresholds, counts, and bad rates are
reported in the model card after each training run.

## Reason codes

Adverse-action style: only contributions that **increase** PD, top 3, as a
JSON list of `{code, text, contribution}`.

- **GBM:** SHAP values on the **pre-calibration margin** (raw LightGBM
  log-odds). SHAP is not computed on the isotonic PD. A reason is
  emitted only when SHAP > 0 *and* the raw value sits on the risky
  side of the train-set median by at least a documented minimum delta
  (for example 10 bureau points, 200 BRL of monthly income).
- **Logistic regression:** `coef × (x − reference)` on the standardised /
  dummy design matrix, with train-set reference (means of 0 after
  standardisation; dummy reference is the omitted or zero level).
  `requested_to_income` and `debt_to_annual_income` are dropped from
  the logistic model (they remain in the GBM) because they are
  nearly collinear with `monthly_income` and flipped its coefficient
  sign. Any remaining feature whose fitted sign conflicts with the
  expected direction is excluded from logistic reason codes and
  listed in the model card.

## Metrics

Reported per model and, where a label exists, per split:

- AUC, KS, Brier, top-decile bad rate, reliability (calibration) curve.
  AUC, KS, and the top-decile bad rate are computed in the **risk
  direction** (higher = more likely to default): calibrated PD for the
  fitted models, and **`-bureau_score`** for the current-policy comparator.
  Brier is not defined for the raw bureau score. Approval-rate-matched
  bad rates still rank by the 300–900 score or raw bureau (higher = better).
- Brier **before and after** isotonic calibration (validation-calibration
  slice and OOT).
- **Bootstrap 95% CIs** for AUC and KS on OOT, 1,000 resamples, seed 42,
  on the main (approved, observable) population.
- **PSI** (train vs OOT) for the calibrated score, the top continuous features (decile bins), and every `prior_*` series (value bins: 0, 1, …, 5+).
- **Default rate by vintage** (disbursement month) on the main population.
- Approval-rate-matched bad rates comparing bureau score, `baseline_logreg_v1`,
  and `gbm_v1` at 50%, 60%, the empirical historical approval rate, and 70%.
  Fitted models are ranked on the pre-calibration margin (no isotonic
  plateau ties). The bureau score uses a 20-seed random tie-break
  average. The number of ties at each cutoff is reported, as is the
  swap set (who changes decision between GBM and bureau, and the bad
  rate of each side).
- Paired bootstrap (GBM minus bureau) on the same 1,000 OOT resamples
  for ΔAUC, ΔKS, and Δbad rate at each matched approval rate.
- Selection-bias comparison of the two populations above. A higher
  through-the-door AUC is a selection-bias effect (a broader, more
  separable population), not evidence that the model itself improved.

Metrics are logged to a local MLflow file store at `ml/mlruns` (gitignored).

## Scoring output

`make train` writes `ml.predictions_scores` in `data/warehouse.duckdb`,
schema `ml`, for **every** silver application and both fitted model
versions. Columns: `application_id`, `model_version`, `pd` (calibrated),
`score`, `risk_band`, `reason_codes` (JSON list), `split`, `scored_at`.
Primary key `(application_id, model_version)`. Then gold / metrics are
rebuilt so `fct_scores` and the AE metric tables fill in. AE files are
not edited.

## Reproducibility

- Random seeds fixed at 42.
- Train-only transformers (imputation, standardisation, encoders,
  isotonic maps) are saved under `ml/artifacts/` and applied unchanged
  to validation, OOT, and unscored applications.
- The pipeline is `make train` after `make dbt`.
