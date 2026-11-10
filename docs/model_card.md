# Model card — synthetic Credara Fintech credit models

All figures in this card are computed on **synthetic** data for the
**fictional** client Credara Fintech. This is not a live policy and is
not based on any real lender or real applicants.

Generated at `2026-11-10T20:13:03+00:00` (UTC).

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
  median imputation, scaling, and one-hot encoding). The ratios
  `requested_to_income`, `debt_to_annual_income` are dropped from this model because they are
  nearly collinear with `monthly_income`, `requested_amount` and
  `existing_debt` and they flipped the sign of `monthly_income`.
  The GBM still uses them, with monotone constraints.
  Both models drop `prior_observable_default_count` and count
  prior applications and declines in a 12-month lookback.
- `gbm_v1`: LightGBM with monotone constraints on features whose
  expected direction is clear, early-stopped on the tune slice.
- Current-policy comparator: raw `bureau_score` (higher is better).
  Reported in the metrics tables only; not written to `predictions_scores`.

Isotonic calibration is fitted on the held-out calibration slice.
Brier scores below are shown before and after this step.

## Score mapping and bands

- PDO = 20; anchor score 600 at odds 50:1
  (PD ≈ 1.9608%). `score = offset - factor * ln(PD / (1-PD))`,
  with PD clipped to `(1e-12, 1-1e-12)` before the map and the
  resulting score clipped to 300–900.
- Bands are cut on calibrated PD, separately for each model,
  using that model's OOT approved-and-observable PD quintiles,
  merged if needed so each band holds at least 5% of that
  population. `baseline_logreg_v1`: A 0.0000≤PD<0.0407, B 0.0407≤PD<0.0734, C 0.0734≤PD<0.1525, D PD≥0.1525; `gbm_v1`: A 0.0000≤PD<0.0454, B 0.0454≤PD<0.0879, C 0.0879≤PD<0.1620, D PD≥0.1620.

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
| `age_at_applied_at` | u-shaped | none |
| `requested_amount` | mixed | none |
| `term_months` | mixed | none |
| `prior_application_count` | mixed | none |
| `existing_debt_is_missing` | mixed | none |

## Metrics (main population: approved and observable)

| Model | Split | n | AUC | KS | Brier (calibrated) | Top-decile bad rate | AUC 95% CI | KS 95% CI |
|-------|-------|--:|----:|---:|---:|---:|---|---|
| `baseline_logreg_v1` | train | 15634 | 0.7553 | 0.3830 | 0.0879 | 33.57% | n/a–n/a | n/a–n/a |
| `baseline_logreg_v1` | validation_tune | 2679 | 0.7348 | 0.3444 | 0.0884 | 30.97% | n/a–n/a | n/a–n/a |
| `baseline_logreg_v1` | validation_calib | 2616 | 0.7403 | 0.3844 | 0.0827 | 30.15% | n/a–n/a | n/a–n/a |
| `baseline_logreg_v1` | oot_main | 6022 | 0.7560 | 0.3891 | 0.0887 | 35.32% | 0.7364–0.7772 | 0.3555–0.4282 |
| `gbm_v1` | train | 15634 | 0.7810 | 0.4167 | 0.0852 | 38.75% | n/a–n/a | n/a–n/a |
| `gbm_v1` | validation_tune | 2679 | 0.7280 | 0.3519 | 0.0897 | 33.21% | n/a–n/a | n/a–n/a |
| `gbm_v1` | validation_calib | 2616 | 0.7348 | 0.3688 | 0.0827 | 29.77% | n/a–n/a | n/a–n/a |
| `gbm_v1` | oot_main | 6022 | 0.7517 | 0.3953 | 0.0888 | 35.16% | 0.7317–0.7718 | 0.3579–0.4322 |

AUC, KS, and top-decile bad rate use calibrated PD (higher =
riskier). The bureau comparator uses `-bureau_score` so the
same convention applies.

On the OOT approved-and-observable population, `gbm_v1` is
**essentially tied** with raw `bureau_score` on AUC
(`0.7517` vs `0.7497`;
paired ΔAUC = GBM − bureau
`0.0017`, 95% CI
`-0.0064–0.0096`, which includes 0).
Point estimates favour the GBM on KS
(`0.3953` vs `0.3702`)
and on pre-calibration-margin approval-rate-matched bad rates,
but the paired ΔKS and Δbad-rate 95% CIs also include 0 (see
tables below). That is **not a demonstrated ranking gain** over
the current-policy bureau score on this synthetic book. The
remaining value of the model is a calibrated PD (for pricing
and expected loss), reason codes and governance, the swap-set
view of who would be approved differently, and a fallback if
the bureau is unavailable or drifts.

Bureau score on OOT (main): AUC `0.7497`,
KS `0.3702`. Individual-model bootstrap CIs
use 1000 resamples, seed 42, and are reported for
OOT only. The paired Δ CIs use the same resamples for GBM and
bureau.

### Calibration (Brier before vs after isotonic)

The calibration slice default rate is 10.05%, against 11.09% on
OOT approved-and-observable loans. Isotonic calibration is fitted
only on the earlier, slightly better book, which is why it is
neutral (or very slightly worse) on OOT Brier.

| Model | Slice | Brier before | Brier after |
|-------|-------|---:|---:|
| `baseline_logreg_v1` | calibration | 0.0840 | 0.0827 |
| `baseline_logreg_v1` | oot_main | 0.0880 | 0.0887 |
| `gbm_v1` | calibration | 0.0840 | 0.0827 |
| `gbm_v1` | oot_main | 0.0887 | 0.0888 |

### Approval-rate-matched bad rates (OOT, main population)

Fitted models are ranked on the pre-calibration margin so
isotonic plateaus do not create ties. The bureau score is an
integer, so its bad rate is the average of 20
random tie-breaks (seed 42, 43, …). `ties_at_cutoff` is the
number of applications that share the marginal (k-th) score.

| Approval rate | n | bureau_score | baseline_logreg_v1 | gbm_v1 | ties bureau | ties logreg | ties gbm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.00% | 3011 | 4.34% | 4.28% | 4.35% | 33 | 1 | 1 |
| 60.00% | 3613 | 5.22% | 4.84% | 5.01% | 19 | 1 | 1 |
| 65.60% | 3950 | 5.78% | 5.34% | 5.44% | 27 | 1 | 1 |
| 70.00% | 4215 | 5.95% | 5.77% | 5.60% | 14 | 1 | 1 |

Paired bootstrap, GBM minus bureau, same 1,000 resamples:

- ΔAUC mean `0.0017`, 95% CI `-0.0064–0.0096`
- ΔKS mean `0.0181`, 95% CI `-0.0060–0.0422`

| Approval rate | Δbad rate (mean) | 95% CI |
|---:|---:|---|
| 50.00% | 0.0005 | -0.0030–0.0043 |
| 60.00% | -0.0023 | -0.0066–0.0019 |
| 65.60% | -0.0030 | -0.0068–0.0008 |
| 70.00% | -0.0038 | -0.0078–0.0000 |

Swap set (GBM vs bureau) at each approval rate. Swap-in:
approved by the GBM and declined by bureau. Swap-out: the
reverse. Bureau approvals use a single seeded tie-break
(seed 42).

| Approval rate | Swap-in n | Swap-in bad rate | Swap-out n | Swap-out bad rate |
|---:|---:|---:|---:|---:|
| 50.00% | 184 | 8.15% | 184 | 8.15% |
| 60.00% | 321 | 8.10% | 321 | 10.59% |
| 65.60% | 302 | 8.94% | 302 | 12.91% |
| 70.00% | 275 | 9.82% | 275 | 15.27% |

### Score and feature PSI (train vs OOT)

`prior_application_count` and `prior_decline_count` use a rolling
12-month lookback ending at `applied_at`, so train and OOT share
the same history depth. `prior_observable_default_count` is
**dropped**: a 12-month lookback cannot contain a completed
12m+90d outcome, and the lifetime count was non-zero in only
0.08% of train vs 4.48% of OOT (value-bin PSI hidden by deciles).
The lifetime series is still reported below as a diagnostic,
not a model input. Continuous features use decile bins; all
`prior_*` series use value bins (0, 1, …, 5+).

OOT AUC before this history change (lifetime `prior_*`,
including the dropped feature): logreg 0.7208, GBM 0.7492.
OOT AUC after: logreg `0.7560`,
GBM `0.7517`.

| Model | Series | PSI |
|-------|--------|----:|
| `baseline_logreg_v1` | `score` | 0.0009 |
| `baseline_logreg_v1` | `bureau_score` | 0.0005 |
| `baseline_logreg_v1` | `estimated_dti` | 0.0017 |
| `baseline_logreg_v1` | `requested_to_income` | 0.0009 |
| `baseline_logreg_v1` | `debt_to_annual_income` | 0.0013 |
| `baseline_logreg_v1` | `monthly_income` | 0.0026 |
| `baseline_logreg_v1` | `existing_debt` | 0.0032 |
| `baseline_logreg_v1` | `prior_application_count` (value bins) | 0.1150 |
| `baseline_logreg_v1` | `prior_decline_count` (value bins) | 0.0264 |
| `baseline_logreg_v1` | `prior_observable_default_count_lifetime` (value bins) | 0.1870 |
| `gbm_v1` | `score` | 0.0005 |
| `gbm_v1` | `bureau_score` | 0.0005 |
| `gbm_v1` | `estimated_dti` | 0.0017 |
| `gbm_v1` | `requested_to_income` | 0.0009 |
| `gbm_v1` | `debt_to_annual_income` | 0.0013 |
| `gbm_v1` | `monthly_income` | 0.0026 |
| `gbm_v1` | `existing_debt` | 0.0032 |
| `gbm_v1` | `prior_application_count` (value bins) | 0.1150 |
| `gbm_v1` | `prior_decline_count` (value bins) | 0.0264 |
| `gbm_v1` | `prior_observable_default_count_lifetime` (value bins) | 0.1870 |

### Risk-band counts and bad rates (OOT, main)

| Model | Band | PD range | n | Bad rate |
|-------|------|----------|--:|---:|
| `baseline_logreg_v1` | A | 0.0000–0.0407 | 1480 | 2.64% |
| `baseline_logreg_v1` | B | 0.0407–0.0734 | 1539 | 5.98% |
| `baseline_logreg_v1` | C | 0.0734–0.1525 | 1695 | 10.50% |
| `baseline_logreg_v1` | D | 0.1525–1.0100 | 1308 | 27.45% |
| `gbm_v1` | A | 0.0000–0.0454 | 1301 | 2.31% |
| `gbm_v1` | B | 0.0454–0.0879 | 2252 | 6.44% |
| `gbm_v1` | C | 0.0879–0.1620 | 648 | 8.80% |
| `gbm_v1` | D | 0.1620–1.0100 | 1821 | 23.94% |

### Reliability (OOT, main)

Equal-count bins of calibrated PD, with Wilson 95% CIs on the
observed default rate. Plot: `docs/img/reliability_oot.png`.

| Model | n | Mean PD | Observed | Observed 95% CI |
|-------|--:|---:|---:|---|
| `baseline_logreg_v1` | 1267 | 3.12% | 2.84% | 2.06%–3.91% |
| `baseline_logreg_v1` | 1152 | 4.04% | 3.91% | 2.93%–5.19% |
| `baseline_logreg_v1` | 598 | 5.61% | 8.19% | 6.25%–10.67% |
| `baseline_logreg_v1` | 676 | 7.34% | 7.54% | 5.78%–9.78% |
| `baseline_logreg_v1` | 573 | 11.58% | 11.69% | 9.31%–14.58% |
| `baseline_logreg_v1` | 595 | 15.06% | 14.29% | 11.70%–17.33% |
| `baseline_logreg_v1` | 658 | 19.74% | 21.43% | 18.46%–24.72% |
| `baseline_logreg_v1` | 503 | 32.59% | 38.57% | 34.42%–42.89% |
| `gbm_v1` | 1300 | 3.07% | 2.31% | 1.62%–3.28% |
| `gbm_v1` | 1397 | 4.54% | 5.51% | 4.43%–6.84% |
| `gbm_v1` | 850 | 7.40% | 8.00% | 6.36%–10.02% |
| `gbm_v1` | 521 | 8.78% | 8.83% | 6.68%–11.58% |
| `gbm_v1` | 1161 | 15.88% | 16.45% | 14.43%–18.69% |
| `gbm_v1` | 193 | 17.61% | 22.80% | 17.44%–29.21% |
| `gbm_v1` | 600 | 31.77% | 35.33% | 31.61%–39.24% |

Tail calibration: `baseline_logreg_v1` top bin observed 38.57% vs mean PD 32.59%; `gbm_v1` top bin observed 35.33% vs mean PD 31.77%. Because expected loss in a
later Proof of Value is PD × EAD × LGD, under-prediction in
the upper tail (band E) undervalues the loss avoided by
declining those loans.

### Default rate by vintage (OOT, main, disbursement month)

The 2025-08 vintage is a **partial month**: observability ends
at disbursements on 2025-08-02, so n is small and the rate is
not comparable to a full month.

| Vintage | n | Defaults | Default rate |
|---------|--:|---:|---:|
| 2025-01 | 685 | 74 | 10.80% |
| 2025-02 | 761 | 95 | 12.48% |
| 2025-03 | 892 | 86 | 9.64% |
| 2025-04 | 906 | 93 | 10.26% |
| 2025-05 | 938 | 118 | 12.58% |
| 2025-06 | 903 | 98 | 10.85% |
| 2025-07 | 884 | 99 | 11.20% |
| 2025-08 (partial) | 53 | 5 | 9.43% |

## Selection bias

The main numbers above use approved, observable loans. Through-the-door
evaluation attaches synthetic counterfactual outcomes for applications
that did not originate a loan. Counterfactuals are evaluation-only and
were never used to fit, tune, or calibrate a model.

A higher through-the-door AUC does **not** mean the model is
better. Declined applicants are a broader, more separable
population, so the lift is a selection-bias effect, not a
change in model quality.

| Model | Population | n | Default rate | AUC | KS | Brier |
|-------|------------|--:|---:|----:|---:|---:|
| `baseline_logreg_v1` | approved_observable | 6022 | 11.09% | 0.7560 | 0.3891 | 0.0887 |
| `baseline_logreg_v1` | through_the_door | 11253 | 18.16% | 0.8023 | 0.4639 | 0.1198 |
| `gbm_v1` | approved_observable | 6022 | 11.09% | 0.7517 | 0.3953 | 0.0888 |
| `gbm_v1` | through_the_door | 11253 | 18.16% | 0.7996 | 0.4556 | 0.1195 |

## Reason-code monotonicity

Reason codes are the top-3 contributions that increase predicted risk.
For the GBM they come from SHAP on the pre-calibration margin and
are emitted only when SHAP > 0 *and* the raw value sits on the
risky side of the train-set median by at least the documented
minimum delta (for example 10 bureau points, 200 BRL of monthly
income, 3 months of tenure; see `reason_min_delta` in the
metrics JSON). For the logistic regression they come from
`coef × (x − train mean)` with the same risky-side and minimum
delta rule. Features whose fitted coefficient sign conflicts
with the expected direction (`prior_decline_count`) are excluded from
logistic reason codes.

Logistic numeric coefficient signs versus expected direction:

| Feature | Fitted sign (coef) | Expected | Matches |
|---------|--------------------|----------|---------|
| `requested_amount` | up (0.294) | mixed | n/a |
| `term_months` | up (0.052) | mixed | n/a |
| `bureau_score` | down (-0.754) | down | yes |
| `existing_debt` | up (0.010) | up | yes |
| `existing_debt_is_missing` | up (0.009) | mixed | n/a |
| `monthly_income` | down (-0.643) | down | yes |
| `employment_months` | down (-0.016) | down | yes |
| `dependents` | up (0.068) | up | yes |
| `age_at_applied_at` | up (0.016) | u-shaped | n/a |
| `estimated_dti` | up (0.109) | up | yes |
| `prior_application_count` | up (0.008) | mixed | n/a |
| `prior_decline_count` | down (-0.011) | up | NO |

- Inspected monotone reason assignments: 216714
- Contradictions versus documented directions: 0
- Contradiction rate: 0.00%

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
