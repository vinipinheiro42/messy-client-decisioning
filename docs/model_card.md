# Model card — synthetic Credara Fintech credit models

All figures in this card are computed on **synthetic** data for the
**fictional** client Credara Fintech. This is not a live policy and is
not based on any real lender or real applicants.

Generated at `2026-11-10T19:55:51+00:00` (UTC).

## Data and label

Training uses silver application and applicant attributes known at
`applied_at`, plus application-time ratios and history that is strictly
prior to `applied_at`. The label is `main_gold.fct_loan_outcomes.is_default_12m`
for **observable** loans only (90+ DPD on an installment due within 12
months of disbursement, observation complete when
`disbursed_at + 12 months + 90 days <= 2026-10-31 UTC`). The last
observable disbursement date in this warehouse is 2025-08-02.
Immature loans are excluded from fitting and from labelled metrics.

## Splits (by `applied_at`, UTC)

| Split | Window | Role |
|-------|--------|------|
| Train | `< 2024-07-01` | Fit |
| Validation (tune) | `2024-07-01` to `< 2024-10-01` | Early stopping only |
| Validation (calibration) | `2024-10-01` to `< 2025-01-01` | Isotonic calibration only |
| OOT | `>= 2025-01-01` | Frozen evaluation; never used for fit, tune, or calibration |

Both validation slices are stored as `validation` in `ml.predictions_scores`.

## Models

- `baseline_logreg_v1`: standardised logistic regression (train-only
  median imputation, scaling, and one-hot encoding).
- `gbm_v1`: LightGBM with monotone constraints on features whose
  expected direction is clear, early-stopped on the tune slice.
- Current-policy comparator: raw `bureau_score` (higher is better).
  Reported in the metrics tables only; not written to `predictions_scores`.

Isotonic calibration is fitted on the held-out calibration slice.
Brier scores below are shown before and after this step.

## Score mapping and bands

- PDO = 20; anchor score 600 at odds 50:1
  (PD ≈ 1.9608%). `score = offset - factor * ln(PD / (1-PD))`, clipped to 300–900.
- Bands are cut on calibrated PD: A < 0.03, B < 0.06, C < 0.10, D < 0.18, E otherwise.

## Feature directions

| Feature | Expected effect on PD | LightGBM monotone |
|---------|------------------------|-------------------|
| `bureau_score` | down | -1 |
| `monthly_income` | down | -1 |
| `employment_months` | down | -1 |
| `existing_debt` | up | 1 |
| `requested_to_income` | up | 1 |
| `debt_to_annual_income` | up | 1 |
| `estimated_dti` | up | 1 |
| `dependents` | up | 1 |
| `prior_decline_count` | up | 1 |
| `prior_observable_default_count` | up | 1 |
| `age_at_applied_at` | u-shaped | none |
| `requested_amount` | mixed | none |
| `term_months` | mixed | none |
| `prior_application_count` | mixed | none |
| `existing_debt_is_missing` | mixed | none |

## Metrics (main population: approved and observable)

| Model | Split | n | AUC | KS | Brier (calibrated) | Top-decile bad rate | AUC 95% CI | KS 95% CI |
|-------|-------|--:|----:|---:|---:|---:|---|---|
| `baseline_logreg_v1` | train | 15634 | 0.7478 | 0.3842 | 0.0879 | 33.50% | n/a–n/a | n/a–n/a |
| `baseline_logreg_v1` | validation_tune | 2679 | 0.7170 | 0.3413 | 0.0891 | 31.72% | n/a–n/a | n/a–n/a |
| `baseline_logreg_v1` | validation_calib | 2616 | 0.7120 | 0.3465 | 0.0834 | 29.01% | n/a–n/a | n/a–n/a |
| `baseline_logreg_v1` | oot_main | 6022 | 0.7173 | 0.3489 | 0.0904 | 33.67% | 0.6957–0.7392 | 0.3124–0.3896 |
| `gbm_v1` | train | 15634 | 0.7805 | 0.4121 | 0.0853 | 39.07% | n/a–n/a | n/a–n/a |
| `gbm_v1` | validation_tune | 2679 | 0.7279 | 0.3567 | 0.0898 | 33.21% | n/a–n/a | n/a–n/a |
| `gbm_v1` | validation_calib | 2616 | 0.7379 | 0.3765 | 0.0825 | 30.15% | n/a–n/a | n/a–n/a |
| `gbm_v1` | oot_main | 6022 | 0.7492 | 0.3903 | 0.0887 | 35.16% | 0.7299–0.7688 | 0.3535–0.4291 |

AUC, KS, and top-decile bad rate use calibrated PD (higher =
riskier). The bureau comparator uses `-bureau_score` so the
same convention applies. The 300–900 score is higher-better and
is used only for approval-rate-matched comparisons.

Bureau score on OOT (main): AUC `0.7497`,
KS `0.3702`. Bootstrap CIs use 1000
resamples, seed 42, and are reported for OOT only.

### Calibration (Brier before vs after isotonic)

| Model | Slice | Brier before | Brier after |
|-------|-------|---:|---:|
| `baseline_logreg_v1` | calibration | 0.0852 | 0.0834 |
| `baseline_logreg_v1` | oot_main | 0.0901 | 0.0904 |
| `gbm_v1` | calibration | 0.0840 | 0.0825 |
| `gbm_v1` | oot_main | 0.0887 | 0.0887 |

### Approval-rate-matched bad rates (OOT, main population)

| Approval rate | n | bureau_score | baseline_logreg_v1 | gbm_v1 |
|---:|---:|---:|---:|---:|
| 50.00% | 3011 | 4.32% | 5.18% | 4.32% |
| 60.00% | 3613 | 5.20% | 5.67% | 4.93% |
| 65.60% | 3950 | 5.80% | 5.92% | 5.27% |
| 70.00% | 4215 | 5.95% | 6.22% | 5.72% |

### Score and feature PSI (train vs OOT)

| Model | Series | PSI |
|-------|--------|----:|
| `baseline_logreg_v1` | `score` | 0.0061 |
| `baseline_logreg_v1` | `bureau_score` | 0.0005 |
| `baseline_logreg_v1` | `estimated_dti` | 0.0017 |
| `baseline_logreg_v1` | `requested_to_income` | 0.0009 |
| `baseline_logreg_v1` | `debt_to_annual_income` | 0.0013 |
| `baseline_logreg_v1` | `monthly_income` | 0.0026 |
| `baseline_logreg_v1` | `existing_debt` | 0.0032 |
| `gbm_v1` | `score` | 0.0015 |
| `gbm_v1` | `bureau_score` | 0.0005 |
| `gbm_v1` | `estimated_dti` | 0.0017 |
| `gbm_v1` | `requested_to_income` | 0.0009 |
| `gbm_v1` | `debt_to_annual_income` | 0.0013 |
| `gbm_v1` | `monthly_income` | 0.0026 |
| `gbm_v1` | `existing_debt` | 0.0032 |

### Risk-band counts and bad rates (OOT, main)

| Model | Band | PD range | n | Bad rate |
|-------|------|----------|--:|---:|
| `baseline_logreg_v1` | A | 0.00–0.03 | 102 | 4.90% |
| `baseline_logreg_v1` | B | 0.03–0.06 | 2891 | 5.12% |
| `baseline_logreg_v1` | C | 0.06–0.10 | 1245 | 8.84% |
| `baseline_logreg_v1` | D | 0.10–0.18 | 1161 | 16.88% |
| `baseline_logreg_v1` | E | 0.18–1.01 | 623 | 33.55% |
| `gbm_v1` | A | 0.00–0.03 | 88 | 2.27% |
| `gbm_v1` | B | 0.03–0.06 | 2675 | 4.07% |
| `gbm_v1` | C | 0.06–0.10 | 1379 | 8.48% |
| `gbm_v1` | D | 0.10–0.18 | 1346 | 17.98% |
| `gbm_v1` | E | 0.18–1.01 | 534 | 37.08% |

### Default rate by vintage (OOT, main, disbursement month)

| Vintage | n | Defaults | Default rate |
|---------|--:|---:|---:|
| 2025-01 | 685 | 74 | 10.80% |
| 2025-02 | 761 | 95 | 12.48% |
| 2025-03 | 892 | 86 | 9.64% |
| 2025-04 | 906 | 93 | 10.26% |
| 2025-05 | 938 | 118 | 12.58% |
| 2025-06 | 903 | 98 | 10.85% |
| 2025-07 | 884 | 99 | 11.20% |
| 2025-08 | 53 | 5 | 9.43% |

## Selection bias

The main numbers above use approved, observable loans. Through-the-door
evaluation attaches synthetic counterfactual outcomes for applications
that did not originate a loan. Counterfactuals are evaluation-only and
were never used to fit, tune, or calibrate a model.

| Model | Population | n | Default rate | AUC | KS | Brier |
|-------|------------|--:|---:|----:|---:|---:|
| `baseline_logreg_v1` | approved_observable | 6022 | 11.09% | 0.7173 | 0.3489 | 0.0904 |
| `baseline_logreg_v1` | through_the_door | 11253 | 18.16% | 0.7666 | 0.4244 | 0.1225 |
| `gbm_v1` | approved_observable | 6022 | 11.09% | 0.7492 | 0.3903 | 0.0887 |
| `gbm_v1` | through_the_door | 11253 | 18.16% | 0.7977 | 0.4516 | 0.1197 |

## Reason-code monotonicity

Reason codes are the top-3 contributions that increase predicted risk.
For the GBM they come from SHAP on the pre-calibration margin; for the
logistic regression they come from `coef × (x − reference)`.

- Inspected monotone reason assignments: 223016
- Contradictions versus documented directions: 4836
- Contradiction rate: 2.17%

## Limitations

- The data, default events, policy assumptions, and counterfactual
  outcomes are entirely synthetic.
- Labels exist only for originated, observable loans; through-the-door
  metrics rely on generated counterfactuals and should be read as a
  selection-bias diagnostic, not as a realised outcome.
- Date-only raw timestamp formats cannot restore time of day; that is a
  data limitation already measured in the quality report and does not
  affect the UTC calendar-day label.
- `existing_debt` may be missing; the model sees a flag and a train-only
  imputed value rather than a false zero.

This card does not use data from any real lender.
