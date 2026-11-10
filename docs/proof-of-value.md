# Proof of Value: Credara Fintech credit decisioning (synthetic)

**Audience:** Head of Credit / CRO, Credara Fintech (fictional client). The technical appendix is for the model-risk and data teams.
**Status:** final M3/M4 numbers, models from commit `2f71b15` (`0c4eaf4` holds the review fixes). Prepared 2026-11-10.

> **Every number in this document is synthetic.** Credara Fintech does not exist. Applicants, loans, outcomes, policy assumptions and counterfactuals are generated (`scripts/generate_data.py`, seed 42). Nothing here describes a real lender, a real portfolio or a live policy. Each figure cites its source as `file → field`.

---

## 1. Executive summary

- **Honest headline: no ranking gain over the bureau score.** On approved, observable out-of-time (OOT) loans (n = 6,022, 11.09% default), neither challenger beats raw `bureau_score` on ranking.
  - AUC is 0.756 for the logistic, 0.752 for the GBM and 0.750 for the bureau.
  - The paired bootstrap of GBM minus bureau gives ΔAUC 95% CI [−0.006, 0.010] and ΔKS 95% CI [−0.006, 0.042]. Both include 0. (`docs/model_metrics.json → metrics.*.oot_main`, `paired_bootstrap.delta_auc.ci`, `paired_bootstrap.delta_ks.ci`)
- **Bad rate at the same approval rate is not demonstrably better.** The Δbad-rate CIs at 50%, 60%, 65.6% and 70% approval all include 0; at 70% the upper bound is exactly 0.0. The point estimates favour the GBM by 0.2–0.4 pp at 60–70%, but that is not proof. (`paired_bootstrap.delta_bad_rate`)
- **Where the value is.** Credara's current policy can rank applicants but cannot price them. The model adds:
  - a calibrated PD, which enables expected loss (EL = PD × EAD × LGD) and risk-based pricing;
  - adverse-action reason codes with 0 sign contradictions;
  - drift monitoring that alerts when the bureau shifts;
  - a fallback if the bureau is unavailable.
- **The business lever is the swap set.** At 70% approval, the GBM and the bureau disagree on 275 applicants in each direction. The GBM's swap-ins default at 9.8% and its swap-outs at 15.3% (`swap_sets[approval_rate=0.7]`). This is the population to test.
- **Recommendation: no full replacement.** Run a 6-month champion/challenger pilot restricted to the swap set, with a random holdout and hard guardrails. Recalibrate the PD tail first. Consider the logistic as the challenger, because it performs the same and is easier to explain.

## 2. Client problem and decision

Credara sends messy operational exports: mixed date formats, currency as text, disguised nulls, resends, one batch with a different delimiter and encoding. Today it approves with a raw bureau score (`dbt/seeds/policy_assumptions.csv → incumbent_policy = bureau_score`; working-tree seed, AE-owned).

**The decision we support:** *Should Credara replace or augment its bureau-score policy with an in-house PD model, and if so, how should it de-risk the change?* Committee owner: Head of Credit / CRO.

## 3. What was built

| Component | What it does | Evidence |
|---|---|---|
| Data pipeline and quality | Bronze → silver → gold in dbt with quarantine. Silver reconciles to ground truth (realised default 10.87% silver vs 10.84% truth on the same rule). This caught and fixed an earlier `amount_paid` coalescing bug that had inflated default to about 29%. | `docs/quality_report.md` (realised default table), commit `438a03d` |
| Default label | 90+ DPD on an installment due within 12 months of disbursement. Observable only after 12 m + 90 d. | `docs/contract.md`, `docs/model_card.md` |
| Models | `baseline_logreg_v1` (standardised logistic) and `gbm_v1` (LightGBM with monotone constraints). Isotonic calibration on a held-out slice. Raw `bureau_score` as the champion comparator. | `docs/model_card.md → Models` |
| Score and bands | PD → 300–900 score (PDO 20, score 600 at odds 50:1). Four bands A–D per model, at least 5% of loans each, monotone. | `docs/model_card.md → Score mapping and bands` |
| Reason codes | Top 3 risk-increasing contributions only (SHAP on the GBM margin; coef × (x − mean) for the logistic), with a minimum-delta rule. 216,714 assignments checked, 0 contradictions. | `model_metrics.json → monotone_check` |
| Scoring API | FastAPI `POST /score` reuses the training feature path. Batch-parity and latency tests. | `README.md`, `tests/test_api.py` |
| Drift monitor | PSI vs the training window: warn at 0.10, alert at 0.25. | `docs/drift_report.md` |

## 4. Results (OOT, approved and observable, n = 6,022, synthetic)

**Discrimination and calibration** (`model_metrics.json → metrics.*.oot_main`):

| Policy | AUC [95% CI] | KS | Brier (calibrated) |
|---|---|---|---|
| Bureau score (champion) | 0.750 | 0.370 | n/a (no PD) |
| Logistic `baseline_logreg_v1` | 0.756 [0.736; 0.777] | 0.389 | 0.0887 |
| GBM `gbm_v1` | 0.752 [0.732; 0.772] | 0.395 | 0.0888 |

**Bad rate at the same approval rate** (`approval_matched[*]`; fitted models ranked on the pre-calibration margin, bureau ties averaged over 20 random tie-breaks):

| Approval | n approved | Bureau | Logistic | GBM | Paired Δbad GBM − bureau, 95% CI |
|---:|---:|---:|---:|---:|---|
| 50% | 3,011 | 4.34% | 4.28% | 4.35% | [−0.30; +0.43] pp |
| 60% | 3,613 | 5.22% | 4.84% | 5.01% | [−0.66; +0.19] pp |
| 65.6% (historical) | 3,950 | 5.78% | 5.34% | 5.44% | [−0.68; +0.08] pp |
| 70% | 4,215 | 5.95% | 5.77% | 5.60% | [−0.78; 0.00] pp |

**Swap set, GBM vs bureau** (`swap_sets`):

| Approval | Swap-in n / bad rate | Swap-out n / bad rate |
|---:|---|---|
| 60% | 321 / 8.10% | 321 / 10.59% |
| 65.6% | 302 / 8.94% | 302 / 12.91% |
| 70% | 275 / 9.82% | 275 / 15.27% |

At 50% the swap sets are equal (184 / 8.15% each way), so there is no difference there.

**Bands.** The bands are monotone. GBM: A 2.31% → B 6.44% → C 8.80% → D 23.94%, with n = 1,301 / 2,252 / 648 / 1,821. Logistic: A 2.64% → D 27.45% (`bands.*`).

**Tail calibration.** PD is under-predicted at the top. In the highest GBM bin, observed default is 35.3% (Wilson CI 31.6–39.2%) against a mean predicted PD of 31.8%. The logistic top bin shows 38.6% observed vs 32.6% predicted (`reliability_oot.*[last bin]`, `docs/img/reliability_oot.png`). Because EL = PD × EAD × LGD, this undervalues the loss avoided by declining the riskiest band.

## 5. Where the value is (framed honestly)

1. **A calibrated PD enables pricing and EL.** A raw bureau score produces no loss figure. The same gold comparison table gives the PD models an EL, while `expected_loss_brl` is NULL for `bureau_score`. Through-the-door at 50% approval, on 6,860 approved OOT applications with `requested_amount` as the EAD proxy, EL is R$ 3.42 m for the GBM and R$ 3.17 m for the logistic (`main_gold.agg_model_comparison_at_approval_rate → expected_loss_brl`).
   - Assumptions (seed, synthetic): LGD 0.65, funding 12% p.a., gross margin 22% p.a., so net margin 10% p.a. (`main_gold.dim_policy_assumptions`).
   - This is the basis for pricing by band, provisioning and EL-based limits.
2. **Per-contract P&L at matched approval: the bureau is slightly ahead, and the figure is indicative only.** In the current gold, observed net result per observed loan at 50% approval is R$ 686.47 for the bureau, R$ 676.45 for the GBM and R$ 672.46 for the logistic. At 60–70% the bureau is also ahead (`observed_net_result_per_loan_brl`).
   - The gold was rebuilt after retraining: `main_gold.fct_scores.pd` matches `ml.predictions_scores` exactly for both models.
   - But the gold matching runs on all 13,720 OOT applications, with outcomes only for booked, observable loans. It also ranks on the integer score: at 60–70%, 2,414 GBM applications share the cutoff score 535, so the tie-break decides membership.
   - It was built from Analytics Engineering's uncommitted working-tree SQL.
   - **Treat these P&L figures as pending** until gold ranks on the model margin and the AE changes are committed. They do not show an upside for the model, and none is claimed.
3. **Adverse-action reason codes.** The top-3 risk-increasing reasons per decision show 0 contradictions with the documented feature directions. `prior_decline_count`, whose fitted logistic sign is wrong, is excluded from logistic reasons (`monotone_check`, `logreg_excluded_from_reasons`). This supports customer notices and regulator questions.
4. **Governance and monitoring.**
   - On the natural OOT shift, score PSI is 0.0005 (ok).
   - In the synthetic stress tests, a −40-point bureau shift *warns* (bureau PSI 0.146) and a −80-point shift *alerts* (bureau 1.529, score 0.338) (`docs/drift_report.md`).
   - `prior_application_count` sits at warn (0.115) by construction and should be watched.
5. **Resilience.** The GBM uses income, debt, tenure and history alongside the bureau, and the drift monitor flags a bureau shift within one scoring batch. That gives Credara a monitored fallback if the bureau degrades or goes down. Note: this is a design property; its OOT benefit without the bureau has not been measured.
6. **The swap set is the business lever.** Whatever the averages, 275 applicants at 70% approval would change decision. Their outcomes (9.8% vs 15.3%) are what a pilot should verify.

## 6. Selection bias

- **Through-the-door AUC is higher, but not because the model is better.** It is about 0.80 (GBM 0.800, logistic 0.802; n = 11,253, 18.16% default; `selection_bias.*.through_the_door`). Declined applicants are a broader, easier-to-separate population.
- Counterfactual outcomes exist only because the data is synthetic: 26.41% default among declined, cancelled and approved-but-not-originated applications (`data/ground_truth/generation_summary.json → counterfactual_default_rate`). They were used for evaluation only, never to fit, tune or calibrate.
- In a real deployment, the only way to learn about the swap-in population is to approve some of it. That is what the pilot does.

## 7. Recommendation and next steps

**Pilot design: champion/challenger on the swap set**

| Element | Proposal |
|---|---|
| Scope | Applications where the GBM (or logistic) and the bureau disagree at the operating approval rate (~65.6%). Everyone else keeps the bureau decision. |
| Randomisation | A 10–20% random holdout of the swap set is decided by the challenger. The rest of the swap set stays with the champion as control. |
| Success metric | Bad rate of challenger swap-ins vs champion swap-outs. Lead indicator: 30+ DPD at month 3. Confirmation: the 12-month 90+ DPD label. Secondary: EL per approved loan. |
| Duration | 6 months of originations for the lead read. Final read after 12 m + 90 d. |
| Guardrails | Exposure cap per month. Stop if swap-in 30+ DPD exceeds champion swap-out by a pre-agreed margin. Weekly PSI with alert at 0.25. Band D excluded from swap-ins. |

Before the pilot:
- **Recalibrate the tail.** Refit calibration on a more recent window or add a tail adjustment, and re-check the top bin.
- **Choose the challenger.** The logistic performs the same as the GBM and is simpler to explain to customers and the regulator.
- **Fix the gold policy tables.** Rank on the model margin, and align the band thresholds: the gold seed uses fixed A–E cuts while the model card uses model-specific A–D cuts.
- **Operational readiness.** The scoring API passes its tests locally. The local p95 latency was 31 ms in the M4 run (as reported by Data Engineering; not persisted in a repo file). A rerun of `tests/test_api.py::test_score_latency_p95` on 2026-11-10 under shared load gave median 33.8 ms / p95 52.0 ms against a 2 s budget. **The Docker image has not been built or tested** (no Docker on the build machine).

## 8. Assumptions and limitations

- **Synthetic data.** All data, outcomes, policy assumptions and counterfactuals are synthetic, for a fictional client. The generator derives `bureau_score` from the same latent risk as defaults, which favours the bureau.
- **Labels.** They exist only for originated, observable loans; the last observable disbursement is 2025-08-02. The partial vintage 2025-08 (n = 53) is not comparable.
- **Calibration shift.** The calibration slice (10.05% default) is better than OOT (11.09%), so isotonic calibration is neutral on OOT Brier (GBM 0.0887 → 0.0888).
- **Simple economics.** No discounting or recovery lag, `requested_amount` as the EAD proxy for non-booked applications, LGD fixed at 0.65.
- **Gold tables.** The policy and economics tables are AE-owned and currently built from uncommitted SQL. Their tie handling differs from the model card's (section 5.2).

---

## Appendix A: definitions

- **Target:** default if any installment due on or before `disbursed_at + 12 months` reaches 90+ days past due, computed from `due_date` and `paid_at` against the cutoff of 2026-10-31 UTC.
- **Observability:** `disbursed_at::date + 12 months + 90 days ≤ 2026-10-31`. Otherwise `outcome_status = immature`, which is excluded from labels and rates.
- **Splits by `applied_at` (UTC):**
  - train < 2024-07-01 (n = 15,634);
  - tune 2024-07-01 to < 2024-10-01 (2,679; early stopping);
  - calibration 2024-10-01 to < 2025-01-01 (2,616; isotonic);
  - OOT ≥ 2025-01-01 (6,022 labelled), frozen.

  Source: `model_metrics.json → splits`.
- **Bands:** cut on calibrated PD from each model's OOT PD quintiles, merged to at least 5% per band. GBM edges are 4.54%, 8.79% and 16.20% (`band_edges`).

## Appendix B: methods

- **Paired bootstrap.** 1,000 resamples of the OOT population (seed 42). On each resample, compute GBM and bureau on the same rows and take the difference. The 95% CI is the 2.5–97.5 percentile of the differences. This is narrower and more correct than comparing two separate CIs.
- **Tie handling.** The fitted models are ranked on the pre-calibration margin, where isotonic plateaus disappear (1 tie at each cutoff). The bureau score is an integer (14–33 ties at the cutoffs), so its bad rate is averaged over 20 random tie-breaks (seeds 42…). The swap sets use a single seeded bureau tie-break (seed 42).
- **Reason codes.** Emitted only if contribution > 0 *and* the raw value is on the risky side of the train median by at least `reason_min_delta` (for example 10 bureau points).

## Appendix C: file references

`docs/model_card.md`, `docs/model_metrics.json`, `docs/drift_report.md`, `docs/img/reliability_oot.png`, `docs/quality_report.md`, `docs/contract.md`, `data/ground_truth/generation_summary.json`, `dbt/seeds/policy_assumptions.csv`, `dbt/models/gold/agg_policy_simulation.sql`, `dbt/models/metrics/agg_model_comparison_at_approval_rate.sql` (read from `data/warehouse.duckdb → main_gold.*` on 2026-11-10 around 17:25 UTC-3), `review/fdds-review.md` (M3-R1…R8), `tests/test_api.py`.
