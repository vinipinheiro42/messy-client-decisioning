# FDDS Review: Credit Model Design (Project 2, synthetic data)

Reviewer: FORWARD DEPLOYED DATA SCIENTIST. All figures come from synthetic data (Credara Fintech, fictional).

## M1: Data design review (2026-11-10)

### R1. Maturity rule diverges between contract and generator (must fix)
- `docs/contract.md`: observable only if `disbursed_at + 12m + 90d <= as_of`.
- `scripts/generate_data.py` line 805: `observation_complete = disbursed_at + 12m <= as_of`, which marks all 30,525 loans complete.
- No defaults are actually censored today, because `first_miss_installment` is drawn from 2..8, so `default_date` is at most about 8m + 90d, which is under 12m. But that is an artifact of the generator, and the dbt layer must not rely on it.
- Action: apply the contract rule in `fct_loan_outcomes`, which makes loans disbursed after 2025-07-31 immature, and set the generator flag the same way so both sources agree. Add a test that fails if any loan with `outcome_status = matured` breaks the rule.

### R2. Default timing has no tail (should fix)
- With first misses only in months 2 to 8, no default can start in months 9 to 12, so the target is easier and more front-loaded than real data.
- Action: draw first miss over 1..12 (for example, weighted toward months 3 to 6). Default stays 90+ days past due within 12 months, so a first miss in month 10 or later cures or rolls past the window and needs an explicit rule. Document it.

### R3. Leakage checks for M3
- Features may use only information available at `applied_at`. Bureau score and income must be as reported at application time, and nothing may be derived from installments or loans.
- `interest_rate_monthly` and `principal` are set after the decision. They must not be model features, but they are fine for the policy P&L.

### Agreed (from DATA ENGINEER)
- OOT uses only observable contracts, and its last disbursement date goes in the model card.
- Counterfactual outcomes for declined and cancelled applications live in `data/ground_truth` only.
- Reason codes are risk-increasing only. The logistic baseline uses coef × (x − reference). The GBM has monotonic constraints, and the SHAP margin space is documented.
- Policy P&L comes from `principal`, rate, term and installment, plus `policy_assumptions` (LGD, funding cost) through `agg_policy_simulation`.
