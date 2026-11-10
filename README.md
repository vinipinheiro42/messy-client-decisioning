# Messy Client Data → Decisioning

Portfolio case study for the fictional client Credara Fintech, **synthetic data only**. Milestones M1–M3 and the M4 scoring API, Dockerfile, and drift monitor are in this tree. Gold/metrics models and the Streamlit dashboard are owned by Analytics Engineering. The English Proof of Value is not started. This repository is local-only and must not be published without the owner's approval.

## Fictional client

**Credara Fintech** is a *fictional* Brazilian personal-credit lender. The company does not exist. Every applicant, application, loan, installment, identifier, and monetary amount in this repository is **synthetic**. The dataset is not derived from, and does not contain, data, code, or names of any real employer or lender.

Credara hands over messy CSV exports from operational systems (applicants, applications, originated loans, and installments) and wants a reproducible path from those files to an explainable credit decision: cleaned tables, a documented default definition, a score, and a short Proof of Value.

## Default definition

A loan is in default if any installment **due on or before `disbursed_at + 12 months`** reaches **90 or more days past due**. The 90th day may fall after the 12-month anniversary; a delinquency that starts on an installment due *after* that window is not a 12-month default. A loan is **observable** only when `disbursed_at + 12 months + 90 days <= 2026-10-31 UTC`. Non-observable loans have `outcome_status = 'immature'` and are excluded from realised default rates and from the training label. `outcome_status` is one of `default`, `good`, or `immature`.

This definition is owned by the gold model `fct_loan_outcomes` and is the same label used to train the model. All default rates reported here are **realized on synthetic data**.

## Repository layout

```
data/raw/                  # messy batch CSVs (generated; gitignored)
data/ground_truth/         # clean synthetic truth, including counterfactual outcomes
data/bronze/               # pre-load Parquet written by scripts/load_bronze.py
dbt/models/bronze/         # Data Engineering
dbt/models/silver/         # Data Engineering
dbt/models/gold/           # Analytics Engineering
dbt/models/metrics/        # Analytics Engineering
dbt/seeds/                 # category mappings and batch cutoffs (DE); policy assumptions (AE)
ml/                        # M3
api/                       # M4
dashboard/                 # Analytics Engineering
docs/                      # contract, issue catalogue, quality report
scripts/generate_data.py   # deterministic synthetic generator
scripts/load_bronze.py     # encoding / delimiter / rename pre-load into bronze
scripts/quality_report.py  # silver vs ground truth and the issues log
scripts/monitor_drift.py   # score/feature PSI vs the training window
```

Raw batch `batch_002` uses a semicolon delimiter and ISO-8859-1 encoding, and renames some columns. `scripts/load_bronze.py` reads each batch with the correct options, maps `income_monthly` → `monthly_income` and `amount_requested` → `requested_amount`, attaches `batch_id`, `source_file`, and `ingested_at`, and writes `data/bronze/*.parquet` for the dbt bronze models. That pre-load is part of `make dbt`.

## Milestones

| Milestone | Scope | Status |
|-----------|--------|--------|
| M1 | Repo, synthetic generator, issue log, contract, tests | Delivered |
| M2 | dbt bronze/silver, quality tests, data-quality report, generator observability and counterfactuals | Delivered in this tree |
| M3 | Explainable credit model (`baseline_logreg_v1`, `gbm_v1`) and `predictions_scores` | Done locally (see `docs/modeling_plan.md` and `docs/model_card.md`) |
| M4 | Scoring API, Dockerfile, drift monitor; dashboard (AE); Proof of Value (English) | API, container file, and drift monitor done locally. Dashboard code exists (AE). PoV not started. |

## How to run (local only)

This project is intended to stay on the local machine. Do not push it to GitHub or any other remote unless the owner explicitly approves publication.

```bash
make setup        # create .venv and install dependencies
make data         # generate ground truth + messy exports (seed 42, default size)
make dbt          # bronze pre-load, dbt seed/run/test for bronze+silver; gold/metrics attempted
make quality      # write docs/quality_report.md and docs/quality_report.json
make train        # fit models, write ml.predictions_scores, refresh gold/metrics
make check        # ruff + pytest (includes API parity and latency)
make api          # uvicorn on 127.0.0.1:8000 (POST /score, GET /health, GET /model-info)
make drift        # PSI train vs OOT and a shifted batch → docs/drift_report.md
make docker-build # docker build -t messy-client-scoring:local . (requires Docker and trained artifacts)
make dashboard    # streamlit (Analytics Engineering)
```

The scoring API reuses the training feature path in `ml/features.py` and the
artifacts in `ml/artifacts/models.joblib` (rebuilt by `make train`, gitignored
if large). `POST /score` accepts application-time fields only. History fields
`prior_application_count` and `prior_decline_count` are optional and default
to 0; they should be the rolling 12-month counts when the caller has them.
The default model is `gbm_v1`; pass `model_version=baseline_logreg_v1` as a
query parameter or JSON field to use the logistic baseline.

An honest model-card finding: after the history-window fix, `gbm_v1` remains
essentially tied with raw `bureau_score` on OOT AUC; paired-bootstrap CIs for
ΔAUC, ΔKS and matched bad rates include 0. The logistic baseline's OOT AUC
rose once `prior_observable_default_count` was dropped. Details are in
[`docs/model_card.md`](docs/model_card.md).

Default generation size is approximately 20,000 applicants and 50,000 applications (`--seed` and `--scale` / size flags are available on the generator).

```bash
.venv/bin/python scripts/generate_data.py --help
```

Ground-truth CPFs are 11-digit strings with **deliberately invalid check digits**. They are not valid Brazilian CPFs. Silver validates format only (11 digits) and does not quarantine on check digits.

`data/ground_truth/counterfactual_outcomes.parquet` holds synthetic potential outcomes for declined, cancelled, and approved-but-not-originated applications, drawn from the same default model as originated loans, for later selection-bias analysis. Those rows are never written to `data/raw/`.

## Interface contract

Silver tables, scoring output, default definition, gold/metrics names, and dbt conventions are recorded in [`docs/contract.md`](docs/contract.md). Injected data-quality issues and expected silver handling are in [`docs/data_issues_catalog.md`](docs/data_issues_catalog.md). The latest silver measurement is in [`docs/quality_report.md`](docs/quality_report.md) (regenerated by `make quality`; labelled synthetic). The modelling plan and model card are in [`docs/modeling_plan.md`](docs/modeling_plan.md) and [`docs/model_card.md`](docs/model_card.md) (the card is regenerated by `make train`; labelled synthetic).

dbt uses the profile name `messy_client` and DuckDB at `data/warehouse.duckdb`. See [`dbt/profiles.yml.template`](dbt/profiles.yml.template).

## License

MIT. See [LICENSE](LICENSE).
