# Drift report — synthetic Credara Fintech

All figures are computed on **synthetic** data. PSI compares a
training-window reference to a scoring batch. Warn at 0.10, alert at 0.25.

Generated at `2026-11-10T20:18:02+00:00` (UTC).

## oot

OOT approved-and-observable versus the train labelled slice. This is the natural time shift, not an injected perturbation.

- n reference: 15634
- n current: 6022
- Overall flag: **warn**

| Feature | Binning | PSI | Flag |
|---------|---------|----:|------|
| `score` | decile | 0.0005 | ok |
| `bureau_score` | decile | 0.0005 | ok |
| `estimated_dti` | decile | 0.0017 | ok |
| `requested_to_income` | decile | 0.0009 | ok |
| `debt_to_annual_income` | decile | 0.0013 | ok |
| `monthly_income` | decile | 0.0026 | ok |
| `existing_debt` | decile | 0.0032 | ok |
| `prior_application_count` | value | 0.1150 | warn |
| `prior_decline_count` | value | 0.0264 | ok |

## bureau_score_shifted_minus_40

Synthetic shifted batch: OOT with `bureau_score` reduced by 40 points, then rescored. This typically lands in the warn band.

- n reference: 15634
- n current: 6022
- Overall flag: **warn**

| Feature | Binning | PSI | Flag |
|---------|---------|----:|------|
| `score` | decile | 0.0839 | ok |
| `bureau_score` | decile | 0.1464 | warn |
| `estimated_dti` | decile | 0.0017 | ok |
| `requested_to_income` | decile | 0.0009 | ok |
| `debt_to_annual_income` | decile | 0.0013 | ok |
| `monthly_income` | decile | 0.0026 | ok |
| `existing_debt` | decile | 0.0032 | ok |
| `prior_application_count` | value | 0.1150 | warn |
| `prior_decline_count` | value | 0.0264 | ok |

## bureau_score_shifted_minus_80

Synthetic shifted batch: OOT with `bureau_score` reduced by 80 points, then rescored, to exercise the alert path (PSI ≥ 0.25).

- n reference: 15634
- n current: 6022
- Overall flag: **alert**

| Feature | Binning | PSI | Flag |
|---------|---------|----:|------|
| `score` | decile | 0.3384 | alert |
| `bureau_score` | decile | 1.5289 | alert |
| `estimated_dti` | decile | 0.0017 | ok |
| `requested_to_income` | decile | 0.0009 | ok |
| `debt_to_annual_income` | decile | 0.0013 | ok |
| `monthly_income` | decile | 0.0026 | ok |
| `existing_debt` | decile | 0.0032 | ok |
| `prior_application_count` | value | 0.1150 | warn |
| `prior_decline_count` | value | 0.0264 | ok |

