"""Train, calibrate, and score baseline_logreg_v1 and gbm_v1 on synthetic data."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.constants import (
    ALLOWED_FEATURES,
    ANCHOR_ODDS,
    ANCHOR_SCORE,
    BAND_EDGES,
    BOOTSTRAP_DRAWS,
    CATEGORICAL_FEATURES,
    EMPIRICAL_APPROVAL_RATES,
    EXPECTED_DIRECTION,
    LABEL_COL,
    LAST_OBSERVABLE_DISBURSEMENT,
    MODEL_GBM,
    MODEL_LOGREG,
    MONOTONE,
    NUMERIC_FEATURES,
    PDO,
    SEED,
)
from ml.features import (
    assert_history_is_causal,
    assert_no_leakage,
    build_feature_frame,
    connect_warehouse,
    feature_matrix,
)
from ml.metrics import (
    approval_matched_bad_rates,
    bootstrap_auc_ks,
    brier_score,
    classification_metrics,
    psi,
    reliability_table,
    vintage_default_rates,
)
from ml.scorecard import (
    check_reason_monotone,
    pd_to_band,
    pd_to_score,
    top_reason_codes,
)

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts"
MLRUNS = PROJECT_ROOT / "ml" / "mlruns"
DEFAULT_WAREHOUSE = PROJECT_ROOT / "data" / "warehouse.duckdb"


def _numeric_pipe() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def _categorical_pipe() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_logreg() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", _numeric_pipe(), NUMERIC_FEATURES),
            ("cat", _categorical_pipe(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("pre", pre),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    C=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _prepare_gbm_frame(
    matrix: pd.DataFrame, categories: dict[str, list] | None = None
) -> tuple[pd.DataFrame, dict[str, list]]:
    out = matrix.copy()
    fitted: dict[str, list] = {}
    for col in CATEGORICAL_FEATURES:
        if categories is None:
            levels = sorted(out[col].astype("string").fillna("missing").unique().tolist())
        else:
            levels = categories[col]
        fitted[col] = levels
        out[col] = pd.Categorical(out[col].astype("string").fillna("missing"), categories=levels)
    return out, fitted


def build_gbm() -> lgb.LGBMClassifier:
    mono = [MONOTONE.get(col, 0) for col in NUMERIC_FEATURES] + [0] * len(CATEGORICAL_FEATURES)
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=mono,
        random_state=SEED,
        verbosity=-1,
    )


def _logreg_contributions(pipeline: Pipeline, matrix: pd.DataFrame) -> pd.DataFrame:
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    lr: LogisticRegression = pipeline.named_steps["lr"]
    transformed = pre.transform(matrix)
    names = list(pre.get_feature_names_out())
    contrib = transformed * lr.coef_.reshape(1, -1)
    # Roll dummy columns back to the original feature name.
    grouped: dict[str, np.ndarray] = {col: np.zeros(len(matrix)) for col in ALLOWED_FEATURES}
    for i, name in enumerate(names):
        raw = name.split("__", 1)[-1]
        base = raw
        for col in CATEGORICAL_FEATURES:
            if raw.startswith(f"{col}_"):
                base = col
                break
        if base not in grouped:
            grouped[base] = np.zeros(len(matrix))
        grouped[base] = grouped[base] + contrib[:, i]
    return pd.DataFrame(grouped, index=matrix.index)


def _gbm_shap(model: lgb.LGBMClassifier, frame: pd.DataFrame) -> pd.DataFrame:
    booster = model.booster_
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(frame, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    return pd.DataFrame(arr, columns=list(frame.columns), index=frame.index)


def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
    iso.fit(np.asarray(raw, dtype=float), np.asarray(y, dtype=int))
    return iso


def _calibrate(iso: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(np.asarray(raw, dtype=float)), 1e-6, 1.0 - 1e-6)


def _labelled(frame: pd.DataFrame, role: str | None = None) -> pd.DataFrame:
    out = frame[frame["has_label"]].copy()
    if role is not None:
        out = out[out["fit_role"] == role]
    return out


def _metrics_block(y, score, prob) -> dict:
    return classification_metrics(y, score, y_prob=prob)


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.2%}"


def write_predictions(warehouse: Path, scores: pd.DataFrame) -> None:
    con = connect_warehouse(warehouse, read_only=False)
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS ml")
        con.execute("DROP TABLE IF EXISTS ml.predictions_scores")
        con.execute(
            """
            CREATE TABLE ml.predictions_scores (
                application_id VARCHAR,
                model_version VARCHAR,
                pd DOUBLE,
                score INTEGER,
                risk_band VARCHAR,
                reason_codes VARCHAR,
                split VARCHAR,
                scored_at TIMESTAMPTZ,
                PRIMARY KEY (application_id, model_version)
            )
            """
        )
        con.register("scores_in", scores)
        con.execute(
            """
            INSERT INTO ml.predictions_scores
            SELECT
                application_id,
                model_version,
                pd,
                score,
                risk_band,
                reason_codes,
                split,
                scored_at
            FROM scores_in
            """
        )
        n = con.execute("select count(*) from ml.predictions_scores").fetchone()[0]
        dups = con.execute(
            """
            select count(*) from (
                select application_id, model_version, count(*) c
                from ml.predictions_scores
                group by 1, 2
                having count(*) > 1
            )
            """
        ).fetchone()[0]
        if dups:
            raise RuntimeError(f"predictions_scores has {dups} duplicate keys")
        print(f"Wrote {n} rows to ml.predictions_scores")
    finally:
        con.close()


def _band_table(y: np.ndarray, pd_values: np.ndarray) -> list[dict]:
    bands = pd_to_band(pd_values)
    rows = []
    for name, low, high in BAND_EDGES:
        mask = bands == name
        if mask.sum() == 0:
            rows.append({"band": name, "pd_low": low, "pd_high": high, "n": 0, "bad_rate": None})
            continue
        rows.append(
            {
                "band": name,
                "pd_low": low,
                "pd_high": high,
                "n": int(mask.sum()),
                "bad_rate": float(np.mean(y[mask])),
            }
        )
    return rows


def render_model_card(payload: dict) -> str:
    split_rows = []
    for model, splits in payload["metrics"].items():
        if model == "bureau_score":
            continue
        for split, row in splits.items():
            if split.startswith("_"):
                continue
            ci = row.get("auc_ci") or (None, None)
            ks_ci = row.get("ks_ci") or (None, None)
            split_rows.append(
                f"| `{model}` | {split} | {row.get('n', 'n/a')} | {_fmt(row.get('auc'))} "
                f"| {_fmt(row.get('ks'))} | {_fmt(row.get('brier'))} "
                f"| {_pct(row.get('top_decile_bad_rate'))} "
                f"| {_fmt(ci[0]) if isinstance(ci, (list, tuple)) else 'n/a'}–"
                f"{_fmt(ci[1]) if isinstance(ci, (list, tuple)) else 'n/a'} "
                f"| {_fmt(ks_ci[0]) if isinstance(ks_ci, (list, tuple)) else 'n/a'}–"
                f"{_fmt(ks_ci[1]) if isinstance(ks_ci, (list, tuple)) else 'n/a'} |"
            )
    bureau = payload["metrics"].get("bureau_score", {}).get("oot_main", {})
    mono = payload.get("monotone_check", {})
    bands = payload.get("bands", {})
    band_lines = []
    for model, rows in bands.items():
        for row in rows:
            band_lines.append(
                f"| `{model}` | {row['band']} | {row['pd_low']:.2f}–{row['pd_high']:.2f} "
                f"| {row['n']} | {_pct(row['bad_rate'])} |"
            )
    dir_lines = [
        f"| `{name}` | {direction} | {MONOTONE.get(name, 'none')} |"
        for name, direction in EXPECTED_DIRECTION.items()
    ]
    psi_lines = []
    for model, values in payload.get("psi", {}).items():
        for key, value in values.items():
            psi_lines.append(f"| `{model}` | `{key}` | {_fmt(value)} |")
    match_lines = []
    for row in payload.get("approval_matched", []):
        bad = " | ".join(
            _pct(row.get(f"bad_rate__{name}"))
            for name in ("bureau_score", MODEL_LOGREG, MODEL_GBM)
        )
        rate = _pct(row["approval_rate"])
        match_lines.append(f"| {rate} | {row['n_approved']} | {bad} |")
    sel = payload.get("selection_bias", {})
    cal = payload.get("calibration", {})
    vintage = payload.get("vintage", [])
    vintage_lines = [
        f"| {row['vintage_month']} | {row['n']} | {row['defaults']} | {_pct(row['default_rate'])} |"
        for row in vintage
    ]
    return "\n".join(
        [
            "# Model card — synthetic Credara Fintech credit models",
            "",
            "All figures in this card are computed on **synthetic** data for the",
            "**fictional** client Credara Fintech. This is not a live policy and is",
            "not based on any real lender or real applicants.",
            "",
            f"Generated at `{payload['scored_at']}` (UTC).",
            "",
            "## Data and label",
            "",
            "Training uses silver application and applicant attributes known at",
            "`applied_at`, plus application-time ratios and history that is strictly",
            "prior to `applied_at`. The label is `main_gold.fct_loan_outcomes.is_default_12m`",
            "for **observable** loans only (90+ DPD on an installment due within 12",
            "months of disbursement, observation complete when",
            "`disbursed_at + 12 months + 90 days <= 2026-10-31 UTC`). The last",
            "observable disbursement date in this warehouse is "
            f"{LAST_OBSERVABLE_DISBURSEMENT.date()}.",
            "Immature loans are excluded from fitting and from labelled metrics.",
            "",
            "## Splits (by `applied_at`, UTC)",
            "",
            "| Split | Window | Role |",
            "|-------|--------|------|",
            "| Train | `< 2024-07-01` | Fit |",
            "| Validation (tune) | `2024-07-01` to `< 2024-10-01` | Early stopping only |",
            "| Validation (calibration) | `2024-10-01` to `< 2025-01-01` "
            "| Isotonic calibration only |",
            "| OOT | `>= 2025-01-01` | Frozen evaluation; never used for "
            "fit, tune, or calibration |",
            "",
            "Both validation slices are stored as `validation` in `ml.predictions_scores`.",
            "",
            "## Models",
            "",
            "- `baseline_logreg_v1`: standardised logistic regression (train-only",
            "  median imputation, scaling, and one-hot encoding).",
            "- `gbm_v1`: LightGBM with monotone constraints on features whose",
            "  expected direction is clear, early-stopped on the tune slice.",
            "- Current-policy comparator: raw `bureau_score` (higher is better).",
            "  Reported in the metrics tables only; not written to `predictions_scores`.",
            "",
            "Isotonic calibration is fitted on the held-out calibration slice.",
            "Brier scores below are shown before and after this step.",
            "",
            "## Score mapping and bands",
            "",
            f"- PDO = {PDO}; anchor score {ANCHOR_SCORE} at odds {int(ANCHOR_ODDS)}:1",
            "  (PD ≈ 1.9608%). `score = offset - factor * ln(PD / (1-PD))`, clipped to 300–900.",
            "- Bands are cut on calibrated PD: A < 0.03, B < 0.06, "
            "C < 0.10, D < 0.18, E otherwise.",
            "",
            "## Feature directions",
            "",
            "| Feature | Expected effect on PD | LightGBM monotone |",
            "|---------|------------------------|-------------------|",
            *dir_lines,
            "",
            "## Metrics (main population: approved and observable)",
            "",
            "| Model | Split | n | AUC | KS | Brier (calibrated) | "
            "Top-decile bad rate | AUC 95% CI | KS 95% CI |",
            "|-------|-------|--:|----:|---:|---:|---:|---|---|",
            *split_rows,
            "",
            "AUC, KS, and top-decile bad rate use calibrated PD (higher =",
            "riskier). The bureau comparator uses `-bureau_score` so the",
            "same convention applies. The 300–900 score is higher-better and",
            "is used only for approval-rate-matched comparisons.",
            "",
            f"Bureau score on OOT (main): AUC `{_fmt(bureau.get('auc'))}`,",
            f"KS `{_fmt(bureau.get('ks'))}`. Bootstrap CIs use {BOOTSTRAP_DRAWS}",
            "resamples, seed 42, and are reported for OOT only.",
            "",
            "### Calibration (Brier before vs after isotonic)",
            "",
            "| Model | Slice | Brier before | Brier after |",
            "|-------|-------|---:|---:|",
            *[
                f"| `{model}` | {slice_} | {_fmt(vals.get('before'))} | {_fmt(vals.get('after'))} |"
                for model, slices in cal.items()
                for slice_, vals in slices.items()
            ],
            "",
            "### Approval-rate-matched bad rates (OOT, main population)",
            "",
            "| Approval rate | n | bureau_score | baseline_logreg_v1 | gbm_v1 |",
            "|---:|---:|---:|---:|---:|",
            *match_lines,
            "",
            "### Score and feature PSI (train vs OOT)",
            "",
            "| Model | Series | PSI |",
            "|-------|--------|----:|",
            *psi_lines,
            "",
            "### Risk-band counts and bad rates (OOT, main)",
            "",
            "| Model | Band | PD range | n | Bad rate |",
            "|-------|------|----------|--:|---:|",
            *band_lines,
            "",
            "### Default rate by vintage (OOT, main, disbursement month)",
            "",
            "| Vintage | n | Defaults | Default rate |",
            "|---------|--:|---:|---:|",
            *vintage_lines,
            "",
            "## Selection bias",
            "",
            "The main numbers above use approved, observable loans. Through-the-door",
            "evaluation attaches synthetic counterfactual outcomes for applications",
            "that did not originate a loan. Counterfactuals are evaluation-only and",
            "were never used to fit, tune, or calibrate a model.",
            "",
            "| Model | Population | n | Default rate | AUC | KS | Brier |",
            "|-------|------------|--:|---:|----:|---:|---:|",
            *[
                f"| `{model}` | {pop} | {row.get('n')} | {_pct(row.get('default_rate'))} "
                f"| {_fmt(row.get('auc'))} | {_fmt(row.get('ks'))} | {_fmt(row.get('brier'))} |"
                for model, pops in sel.items()
                for pop, row in pops.items()
            ],
            "",
            "## Reason-code monotonicity",
            "",
            "Reason codes are the top-3 contributions that increase predicted risk.",
            "For the GBM they come from SHAP on the pre-calibration margin; for the",
            "logistic regression they come from `coef × (x − reference)`.",
            "",
            f"- Inspected monotone reason assignments: {mono.get('inspected')}",
            f"- Contradictions versus documented directions: {mono.get('contradictions')}",
            f"- Contradiction rate: {_pct(mono.get('contradiction_rate'))}",
            "",
            "## Limitations",
            "",
            "- The data, default events, policy assumptions, and counterfactual",
            "  outcomes are entirely synthetic.",
            "- Labels exist only for originated, observable loans; through-the-door",
            "  metrics rely on generated counterfactuals and should be read as a",
            "  selection-bias diagnostic, not as a realised outcome.",
            "- Date-only raw timestamp formats cannot restore time of day; that is a",
            "  data limitation already measured in the quality report and does not",
            "  affect the UTC calendar-day label.",
            "- `existing_debt` may be missing; the model sees a flag and a train-only",
            "  imputed value rather than a false zero.",
            "",
            "This card does not use data from any real lender.",
            "",
        ]
    )


def run(warehouse: Path | str = DEFAULT_WAREHOUSE) -> dict:
    warehouse = Path(warehouse)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MLRUNS.mkdir(parents=True, exist_ok=True)

    frame = build_feature_frame(warehouse)
    X_all = feature_matrix(frame)
    assert_no_leakage(list(X_all.columns))
    assert_history_is_causal(frame, warehouse)

    train = _labelled(frame, "train")
    tune = _labelled(frame, "tune")
    calib = _labelled(frame, "calib")
    oot_main = frame[frame["has_label"] & (frame["fit_role"] == "oot")]
    print(
        "Labelled sizes:",
        f"train={len(train)} tune={len(tune)} calib={len(calib)} oot={len(oot_main)}",
    )

    X_train = feature_matrix(train)
    y_train = train[LABEL_COL].astype(int).to_numpy()
    X_tune = feature_matrix(tune)
    y_tune = tune[LABEL_COL].astype(int).to_numpy()
    X_calib = feature_matrix(calib)
    y_calib = calib[LABEL_COL].astype(int).to_numpy()

    logreg = build_logreg()
    logreg.fit(X_train, y_train)
    gbm_train, cat_levels = _prepare_gbm_frame(X_train)
    gbm_tune, _ = _prepare_gbm_frame(X_tune, cat_levels)
    gbm_calib, _ = _prepare_gbm_frame(X_calib, cat_levels)
    # Median-impute numeric features for LightGBM with train statistics only.
    num_medians = X_train[NUMERIC_FEATURES].median(numeric_only=True)

    def _impute(mat: pd.DataFrame) -> pd.DataFrame:
        out = mat.copy()
        out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(num_medians)
        return out

    gbm = build_gbm()
    gbm.fit(
        _impute(gbm_train),
        y_train,
        eval_X=_impute(gbm_tune),
        eval_y=y_tune,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    raw_logreg = {
        "train": logreg.predict_proba(X_train)[:, 1],
        "tune": logreg.predict_proba(X_tune)[:, 1],
        "calib": logreg.predict_proba(X_calib)[:, 1],
    }
    raw_gbm = {
        "train": gbm.predict_proba(_impute(gbm_train))[:, 1],
        "tune": gbm.predict_proba(_impute(gbm_tune))[:, 1],
        "calib": gbm.predict_proba(_impute(gbm_calib))[:, 1],
    }
    iso_logreg = _fit_isotonic(raw_logreg["calib"], y_calib)
    iso_gbm = _fit_isotonic(raw_gbm["calib"], y_calib)

    # Score every application.
    X_all_gbm, _ = _prepare_gbm_frame(X_all, cat_levels)
    raw_all_logreg = logreg.predict_proba(X_all)[:, 1]
    raw_all_gbm = gbm.predict_proba(_impute(X_all_gbm))[:, 1]
    pd_all = {
        MODEL_LOGREG: _calibrate(iso_logreg, raw_all_logreg),
        MODEL_GBM: _calibrate(iso_gbm, raw_all_gbm),
    }
    raw_all = {MODEL_LOGREG: raw_all_logreg, MODEL_GBM: raw_all_gbm}

    # Reason codes on the pre-calibration margin / logit contributions.
    logreg_contrib = _logreg_contributions(logreg, X_all)
    gbm_margin_shap = _gbm_shap(gbm, _impute(X_all_gbm))
    # Train-set reference (median) for numeric reason-code wording.
    reference = {col: float(num_medians[col]) for col in NUMERIC_FEATURES if col in num_medians}

    scored_at = datetime.now(timezone.utc).replace(microsecond=0)
    reason_logreg = top_reason_codes(logreg_contrib, X_all, reference)
    reason_gbm = top_reason_codes(gbm_margin_shap, X_all, reference)
    mono_logreg = check_reason_monotone(reason_logreg, X_all, reference)
    mono_gbm = check_reason_monotone(reason_gbm, X_all, reference)
    mono_check = {
        "inspected": mono_logreg["inspected"] + mono_gbm["inspected"],
        "contradictions": mono_logreg["contradictions"] + mono_gbm["contradictions"],
        "contradiction_rate": None,
        "by_model": {MODEL_LOGREG: mono_logreg, MODEL_GBM: mono_gbm},
    }
    inspected = mono_check["inspected"]
    if inspected:
        mono_check["contradiction_rate"] = mono_check["contradictions"] / inspected

    records = []
    for model, probas, reasons in (
        (MODEL_LOGREG, pd_all[MODEL_LOGREG], reason_logreg),
        (MODEL_GBM, pd_all[MODEL_GBM], reason_gbm),
    ):
        scores = pd_to_score(probas)
        bands = pd_to_band(probas)
        chunk = pd.DataFrame(
            {
                "application_id": frame["application_id"].to_numpy(),
                "model_version": model,
                "pd": probas,
                "score": scores,
                "risk_band": bands,
                "reason_codes": reasons,
                "split": frame["split"].to_numpy(),
                "scored_at": scored_at,
            }
        )
        records.append(chunk)
    scores_out = pd.concat(records, ignore_index=True)
    write_predictions(warehouse, scores_out)

    # ---- metrics ----
    def _slice_metrics(idx, model_key: str, with_ci: bool = False) -> dict:
        y = frame.loc[idx, LABEL_COL].astype(int).to_numpy()
        prob = pd_all[model_key][idx.to_numpy()]
        # AUC/KS/top-decile use PD (higher = riskier), not the 300-900 score.
        row = _metrics_block(y, prob, prob)
        if with_ci:
            boot = bootstrap_auc_ks(y, prob, n_draws=BOOTSTRAP_DRAWS, seed=SEED)
            row["auc_ci"] = boot["auc_ci"]
            row["ks_ci"] = boot["ks_ci"]
        return row

    metrics: dict = {MODEL_LOGREG: {}, MODEL_GBM: {}, "bureau_score": {}}
    role_idx = {
        "train": frame["has_label"] & (frame["fit_role"] == "train"),
        "validation_tune": frame["has_label"] & (frame["fit_role"] == "tune"),
        "validation_calib": frame["has_label"] & (frame["fit_role"] == "calib"),
        "oot_main": frame["has_label"] & (frame["fit_role"] == "oot"),
    }
    for model in (MODEL_LOGREG, MODEL_GBM):
        for name, idx in role_idx.items():
            metrics[model][name] = _slice_metrics(idx, model, with_ci=(name == "oot_main"))

    oot_idx = role_idx["oot_main"]
    y_oot = frame.loc[oot_idx, LABEL_COL].astype(int).to_numpy()
    bureau_oot = pd.to_numeric(frame.loc[oot_idx, "bureau_score"], errors="coerce").to_numpy()
    # Raw bureau_score is higher-better; negate so AUC/KS match the PD convention.
    metrics["bureau_score"]["oot_main"] = classification_metrics(y_oot, -bureau_oot)
    metrics["bureau_score"]["oot_main"]["score_direction"] = "negated (higher bureau = lower risk)"

    empirical_rate = float((frame["decision"] == "approved").mean())
    rates = tuple(sorted(set(EMPIRICAL_APPROVAL_RATES) | {round(empirical_rate, 3)}))
    approval_matched = approval_matched_bad_rates(
        y_oot,
        {
            "bureau_score": bureau_oot,
            MODEL_LOGREG: pd_to_score(pd_all[MODEL_LOGREG][oot_idx.to_numpy()]),
            MODEL_GBM: pd_to_score(pd_all[MODEL_GBM][oot_idx.to_numpy()]),
        },
        rates,
    )

    train_idx = role_idx["train"]
    psi_report = {}
    top_features = [
        "bureau_score",
        "estimated_dti",
        "requested_to_income",
        "debt_to_annual_income",
        "monthly_income",
        "existing_debt",
    ]
    for model in (MODEL_LOGREG, MODEL_GBM):
        psi_report[model] = {
            "score": psi(
                pd_to_score(pd_all[model][train_idx.to_numpy()]).astype(float),
                pd_to_score(pd_all[model][oot_idx.to_numpy()]).astype(float),
            )
        }
        for col in top_features:
            psi_report[model][col] = psi(
                pd.to_numeric(frame.loc[train_idx, col], errors="coerce").to_numpy(),
                pd.to_numeric(frame.loc[oot_idx, col], errors="coerce").to_numpy(),
            )

    calibration = {}
    for model, raw_map, iso in (
        (MODEL_LOGREG, raw_logreg, iso_logreg),
        (MODEL_GBM, raw_gbm, iso_gbm),
    ):
        calibration[model] = {}
        for slice_name, raw, y in (
            ("calibration", raw_map["calib"], y_calib),
            (
                "oot_main",
                raw_all[model][oot_idx.to_numpy()],
                y_oot,
            ),
        ):
            calibration[model][slice_name] = {
                "before": brier_score(y, raw),
                "after": brier_score(y, _calibrate(iso, raw)),
            }

    bands = {
        model: _band_table(y_oot, pd_all[model][oot_idx.to_numpy()])
        for model in (MODEL_LOGREG, MODEL_GBM)
    }
    vintage = vintage_default_rates(frame.loc[oot_idx, "disbursed_at"], y_oot)
    vintage_records = vintage.to_dict(orient="records")

    # Selection bias: through-the-door on OOT applications with a real or CF label.
    cf_path = PROJECT_ROOT / "data" / "ground_truth" / "counterfactual_outcomes.parquet"
    ttd_metrics = {MODEL_LOGREG: {}, MODEL_GBM: {}}
    if cf_path.is_file():
        cf = pd.read_parquet(cf_path)
        merged = frame.merge(
            cf[["application_id", "is_default_90dpd_12m_counterfactual"]], how="left"
        )
        ttd_label = np.where(
            merged["has_label"],
            pd.to_numeric(merged[LABEL_COL], errors="coerce"),
            pd.to_numeric(merged["is_default_90dpd_12m_counterfactual"], errors="coerce"),
        )
        oot_all = merged["fit_role"].eq("oot") & np.isfinite(ttd_label)
        for model in (MODEL_LOGREG, MODEL_GBM):
            ttd_metrics[model]["approved_observable"] = metrics[model]["oot_main"]
            y_ttd = ttd_label[oot_all.to_numpy()]
            p_ttd = pd_all[model][oot_all.to_numpy()]
            ttd_metrics[model]["through_the_door"] = classification_metrics(
                y_ttd, p_ttd, y_prob=p_ttd
            )

    payload = {
        "client": "Credara Fintech (fictional)",
        "is_synthetic": True,
        "scored_at": scored_at.isoformat(),
        "splits": {
            "train_end": "2024-07-01",
            "tune_end": "2024-10-01",
            "calib_end": "2025-01-01",
            "last_observable_disbursement": str(LAST_OBSERVABLE_DISBURSEMENT.date()),
            "sizes": {
                "train": int(len(train)),
                "tune": int(len(tune)),
                "calib": int(len(calib)),
                "oot_main": int(len(oot_main)),
            },
        },
        "metrics": metrics,
        "calibration": calibration,
        "approval_matched": approval_matched,
        "empirical_approval_rate": empirical_rate,
        "psi": psi_report,
        "bands": bands,
        "vintage": vintage_records,
        "selection_bias": ttd_metrics,
        "monotone_check": mono_check,
        "reliability_oot": {
            model: reliability_table(y_oot, pd_all[model][oot_idx.to_numpy()]).to_dict(
                orient="records"
            )
            for model in (MODEL_LOGREG, MODEL_GBM)
        },
    }
    # Make JSON-safe.
    text = json.dumps(payload, indent=2, default=str)
    (ARTIFACTS / "model_metrics.json").write_text(text + "\n", encoding="utf-8")
    (PROJECT_ROOT / "docs" / "model_metrics.json").write_text(text + "\n", encoding="utf-8")
    card = render_model_card(payload)
    (PROJECT_ROOT / "docs" / "model_card.md").write_text(card, encoding="utf-8")

    joblib.dump(
        {
            "logreg": logreg,
            "gbm": gbm,
            "iso_logreg": iso_logreg,
            "iso_gbm": iso_gbm,
            "num_medians": num_medians,
            "cat_levels": cat_levels,
        },
        ARTIFACTS / "models.joblib",
    )

    # MLflow 3 treats the filesystem backend as opt-in; keep a local file store under ml/mlruns.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(MLRUNS.as_uri())
    mlflow.set_experiment("messy-client-decisioning")
    with mlflow.start_run(run_name=f"m3-{scored_at.strftime('%Y%m%dT%H%M%SZ')}"):
        mlflow.log_params(
            {
                "pdo": PDO,
                "anchor_score": ANCHOR_SCORE,
                "anchor_odds": ANCHOR_ODDS,
                "train_end": "2024-07-01",
                "seed": SEED,
            }
        )
        for model in (MODEL_LOGREG, MODEL_GBM):
            for split, row in metrics[model].items():
                for key in ("auc", "ks", "brier", "top_decile_bad_rate"):
                    if row.get(key) is not None:
                        mlflow.log_metric(f"{model}__{split}__{key}", float(row[key]))
        mlflow.log_artifact(str(PROJECT_ROOT / "docs" / "model_card.md"))
        mlflow.log_artifact(str(ARTIFACTS / "model_metrics.json"))

    print("Wrote docs/model_card.md and docs/model_metrics.json")
    print(
        "OOT main:",
        f"logreg AUC={metrics[MODEL_LOGREG]['oot_main']['auc']:.4f}",
        f"gbm AUC={metrics[MODEL_GBM]['oot_main']['auc']:.4f}",
        f"bureau AUC={metrics['bureau_score']['oot_main']['auc']:.4f}",
    )
    print(
        "Monotone reason-code contradictions:",
        mono_check["contradictions"],
        "of",
        mono_check["inspected"],
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
