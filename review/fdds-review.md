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

## M3: Model review (2026-11-10, commit 028a092, synthetic data)

Overall: the design is sound. The splits are clean (OOT never used for fit, tuning or calibration), maturity is enforced, there is a champion baseline, bootstrap CIs, PSI, vintage curves, a documented PDO and two populations. The honest finding that the GBM ties `bureau_score` on AUC is the right thing to show. Items below are ordered by priority.

### M3-R1. Ties in approval-rate matching (must check before the PoV)
- At 50% approval, GBM and bureau have an *identical* bad rate (0.043175…, i.e. the same 130 of 3011 defaults). Isotonic calibration produces plateaus, so calibrated PD has many ties, and `approval_matched_bad_rates` breaks them with `np.argsort` in row order. If the OOT frame is sorted by anything correlated with risk, the tie-break is not neutral.
- Action: rank on the pre-calibration margin (isotonic is monotone, so ordering is preserved without ties), or break ties randomly and average across seeds. Report the number of ties at each cutoff.

### M3-R2. Is the GBM's gain over bureau real? (must quantify)
- AUC 0.749 vs 0.750, and the CIs overlap completely. At 60% approval the bad rate is 4.93% vs 5.20%, a gap of about 10 defaults out of 3,613.
- Action: run a paired bootstrap on the same resamples for ΔAUC, ΔKS and Δbad-rate at each matched approval, GBM minus bureau, and report the CI of each difference. If Δbad-rate's CI crosses 0, the PoV must say "no demonstrated gain in ranking".
- PoV framing: the value then comes from (a) the swap-set view (who gets approved or declined differently and their bad rate), (b) a calibrated PD that enables pricing and EL, which a raw bureau score does not, (c) reason codes and governance, and (d) resilience if the bureau is unavailable or drifts. All of this is still on synthetic data, and it is expected: the generator derives `bureau_score` from the same latent risk.

### M3-R3. Reason-code contradictions
- Logistic, 3.45%: the examples show `monthly_income` above the reference still contributing risk, which means the logistic coefficient sign is wrong. This is likely collinearity with `requested_to_income`, `estimated_dti` and `debt_to_annual_income`. Action: list the coefficient signs against the expected directions; drop or merge the redundant ratios, or exclude a feature with a wrong sign from reason codes.
- GBM, 0.71%: the examples sit right at the reference (bureau 720 vs 719), so SHAP noise and interactions are at play. Action: emit a reason code only when SHAP > 0 *and* x is on the risky side of the reference by at least a minimum delta, and document that rule.

### M3-R4. Band design
- A holds 88 loans (1.5%) and B holds 2,675 (44%), so A barely exists and B is too broad to set policy on. Action: recut the bands (for example on OOT or calibration PD quantiles, or on business PD cuts with a minimum of 5% per band), keep them monotone, and document the cut points.

### M3-R5. Calibration in the tails
- The reliability curve is under-predicted at the top (logistic: top bin observed 42.8% vs mean PD 34.5%). Action: show the reliability plot with CIs per bin in the card. Since EL in the PoV is PD × EAD × LGD, under-prediction in band E undervalues the loss of declining it, so mention that.

### M3-R6. Reading the through-the-door results
- AUC rising to 0.798 does not mean the model is better. Declined applicants are easier to separate, so the higher AUC comes from the wider population (the selection-bias effect). State this explicitly in the card.

### M3-R7. Small details
- Vintage 2025-08 has n=53 (a partial month, because of the observability cutoff). Mark it as partial or drop it from the chart.
- Leakage on `prior_observable_default_count`: checked, OK. It only counts defaults whose `observable_from_date <= applied_at`, which is conservative.

### M3-R8. `prior_observable_default_count`: sign and drift (found while reproducing on ground truth)
- Its logistic coefficient is negative, the opposite of the expected direction. It is non-zero in only 0.06% of train rows vs 5.46% of OOT rows, because history accumulates over time. PSI with value bins is about 0.24, while decile bins hide it, and the card's PSI table leaves out the `prior_*` features.
- In the ground-truth reproduction, dropping it lifted logistic OOT AUC from 0.718 to 0.760.
- Action: include the `prior_*` features in the PSI table with value-based bins. Either drop the feature or recompute it so train and OOT have comparable history depth (for example a fixed lookback window), then retrain.
- Note: the calibration slice's default rate (10.05%) is below the OOT rate (11.09%), which explains why isotonic calibration is neutral on OOT. Mention it in the card.
- Correction to M1: the last observable disbursement date is 2025-08-02, not 2025-07-31.
