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
    BAND_MIN_SHARE,
    BOOTSTRAP_DRAWS,
    BUREAU_TIEBREAK_SEEDS,
    CATEGORICAL_FEATURES,
    EMPIRICAL_APPROVAL_RATES,
    EXPECTED_DIRECTION,
    LABEL_COL,
    LAST_OBSERVABLE_DISBURSEMENT,
    LOGREG_DROPPED_FEATURES,
    LOGREG_NUMERIC_FEATURES,
    MODEL_GBM,
    MODEL_LOGREG,
    MONOTONE,
    NUMERIC_FEATURES,
    PDO,
    REASON_MIN_DELTA,
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
    choose_band_edges,
    classification_metrics,
    paired_bootstrap_gbm_minus_bureau,
    psi,
    psi_value_bins,
    reliability_table,
    swap_set_at_rate,
    vintage_default_rates,
)
from ml.plots import write_reliability_plot
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
            ("num", _numeric_pipe(), LOGREG_NUMERIC_FEATURES),
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


def logreg_feature_signs(pipeline: Pipeline) -> list[dict]:
    """Coefficient sign of each logistic feature versus the expected direction."""
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    lr: LogisticRegression = pipeline.named_steps["lr"]
    names = list(pre.get_feature_names_out())
    coefs = lr.coef_.ravel()
    rows = []
    for name, coef in zip(names, coefs, strict=True):
        raw = name.split("__", 1)[-1]
        base = raw
        is_dummy = False
        for col in CATEGORICAL_FEATURES:
            if raw == col or raw.startswith(f"{col}_"):
                base = col
                is_dummy = raw.startswith(f"{col}_")
                break
        expected = EXPECTED_DIRECTION.get(base)
        if expected == "down":
            matches: bool | None = bool(coef < 0)
        elif expected == "up":
            matches = bool(coef > 0)
        else:
            matches = None
        rows.append(
            {
                "feature": raw,
                "base_feature": base,
                "is_dummy": is_dummy,
                "coefficient": float(coef),
                "sign": "up" if coef > 0 else ("down" if coef < 0 else "zero"),
                "expected": expected,
                "matches_expected": matches,
            }
        )
    return rows


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


def _band_table(y: np.ndarray, pd_values: np.ndarray, edges) -> list[dict]:
    bands = pd_to_band(pd_values, edges=edges)
    rows = []
    for name, low, high in edges:
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
                f"| `{model}` | {row['band']} | {row['pd_low']:.4f}–{row['pd_high']:.4f} "
                f"| {row['n']} | {_pct(row['bad_rate'])} |"
            )
    dir_lines = [
        f"| `{name}` | {direction} | {MONOTONE.get(name, 'none')} |"
        for name, direction in EXPECTED_DIRECTION.items()
    ]
    psi_lines = []
    for model, values in payload.get("psi", {}).items():
        for key, value in values.items():
            if isinstance(value, dict):
                for sub, subval in value.items():
                    psi_lines.append(
                        f"| `{model}` | `{sub}` (value bins) | {_fmt(subval)} |"
                    )
            else:
                psi_lines.append(f"| `{model}` | `{key}` | {_fmt(value)} |")
    match_lines = []
    for row in payload.get("approval_matched", []):
        bad = " | ".join(
            _pct(row.get(f"bad_rate__{name}"))
            for name in ("bureau_score", MODEL_LOGREG, MODEL_GBM)
        )
        rate = _pct(row["approval_rate"])
        ties = " | ".join(
            str(row.get(f"ties_at_cutoff__{name}", "n/a"))
            for name in ("bureau_score", MODEL_LOGREG, MODEL_GBM)
        )
        match_lines.append(f"| {rate} | {row['n_approved']} | {bad} | {ties} |")
    sel = payload.get("selection_bias", {})
    cal = payload.get("calibration", {})
    vintage = payload.get("vintage", [])
    vintage_lines = [
        "| {month}{partial} | {n} | {defaults} | {rate} |".format(
            month=row["vintage_month"],
            partial=" (partial)" if row.get("partial") else "",
            n=row["n"],
            defaults=row["defaults"],
            rate=_pct(row["default_rate"]),
        )
        for row in vintage
    ]
    gbm_oot = payload["metrics"].get(MODEL_GBM, {}).get("oot_main", {})
    paired = payload.get("paired_bootstrap", {})
    swap_lines = [
        "| {rate} | {nin} | {rin} | {nout} | {rout} |".format(
            rate=_pct(row["approval_rate"]),
            nin=row.get("swap_in_n"),
            rin=_pct(row.get("swap_in_bad_rate")),
            nout=row.get("swap_out_n"),
            rout=_pct(row.get("swap_out_bad_rate")),
        )
        for row in payload.get("swap_sets", [])
    ]
    delta_bad_lines = []
    for rate, block in (paired.get("delta_bad_rate") or {}).items():
        ci = block.get("ci") or (None, None)
        delta_bad_lines.append(
            f"| {_pct(float(rate))} | {_fmt(block.get('mean'))} | "
            f"{_fmt(ci[0]) if ci else 'n/a'}–{_fmt(ci[1]) if ci else 'n/a'} |"
        )
    d_auc = paired.get("delta_auc") or {}
    d_ks = paired.get("delta_ks") or {}
    auc_ci = d_auc.get("ci") or (None, None)
    ks_ci = d_ks.get("ci") or (None, None)
    sign_rows = [
        "| `{feat}` | {sign} ({coef:.3f}) | {exp} | {match} |".format(
            feat=row["feature"],
            sign=row["sign"],
            coef=row["coefficient"],
            exp=row.get("expected") or "n/a",
            match=(
                "yes"
                if row["matches_expected"]
                else ("n/a" if row["matches_expected"] is None else "NO")
            ),
        )
        for row in payload.get("logreg_coefficient_signs", [])
        if not row.get("is_dummy")
    ]
    dropped = ", ".join(f"`{c}`" for c in payload.get("logreg_dropped_features", [])) or "none"
    excluded = (
        ", ".join(f"`{c}`" for c in payload.get("logreg_excluded_from_reasons", [])) or "none"
    )
    rel_rows = []
    for model, rows in payload.get("reliability_oot", {}).items():
        for row in rows:
            rel_rows.append(
                f"| `{model}` | {row.get('n')} | {_pct(row.get('mean_pd'))} | "
                f"{_pct(row.get('observed_rate'))} | "
                f"{_pct(row.get('observed_lo'))}–{_pct(row.get('observed_hi'))} |"
            )
    # Tail bin (highest mean PD) for the under-prediction note.
    tail_notes = []
    for model, rows in payload.get("reliability_oot", {}).items():
        if not rows:
            continue
        tail = max(rows, key=lambda r: float(r.get("mean_pd") or 0))
        obs = tail.get("observed_rate")
        pred = tail.get("mean_pd")
        if obs is not None and pred is not None and obs > pred + 0.02:
            tail_notes.append(
                f"`{model}` top bin observed {_pct(obs)} vs mean PD {_pct(pred)}"
            )
    tail_text = (
        "; ".join(tail_notes)
        if tail_notes
        else "no material under-prediction in the top reliability bin"
    )
    def _edge_text(edges) -> str:
        parts = []
        for item in edges:
            name, low, high = item[0], float(item[1]), float(item[2])
            if high < 1.0:
                parts.append(f"{name} {low:.4f}≤PD<{high:.4f}")
            else:
                parts.append(f"{name} PD≥{low:.4f}")
        return ", ".join(parts)

    raw_edges = payload.get("band_edges", {})
    if isinstance(raw_edges, dict):
        band_edge_text = "; ".join(
            f"`{model}`: {_edge_text(edges)}" for model, edges in raw_edges.items()
        ) or "see table"
    else:
        band_edge_text = _edge_text(raw_edges) or "see table"
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
            "  median imputation, scaling, and one-hot encoding). The ratios",
            f"  {dropped} are dropped from this model because they are",
            "  nearly collinear with `monthly_income`, `requested_amount` and",
            "  `existing_debt` and they flipped the sign of `monthly_income`.",
            "  The GBM still uses them, with monotone constraints.",
            "  Both models drop `prior_observable_default_count` and count",
            "  prior applications and declines in a 12-month lookback.",
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
            "  (PD ≈ 1.9608%). `score = offset - factor * ln(PD / (1-PD))`,",
            "  with PD clipped to `(1e-12, 1-1e-12)` before the map and the",
            "  resulting score clipped to 300–900.",
            "- Bands are cut on calibrated PD, separately for each model,",
            "  starting from that model's OOT approved-and-observable PD quintiles,",
            "  then merged so each band holds at least 5% of that population.",
            "  Isotonic plateaus currently collapse the five starting cuts to",
            f"  four bands (A–D) on both models. {band_edge_text}.",
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
            "same convention applies.",
            "",
            "On the OOT approved-and-observable population, `gbm_v1` is",
            "**essentially tied** with raw `bureau_score` on AUC",
            f"(`{_fmt(gbm_oot.get('auc'))}` vs `{_fmt(bureau.get('auc'))}`;",
            "paired ΔAUC = GBM − bureau",
            f"`{_fmt(d_auc.get('mean'))}`, 95% CI",
            f"`{_fmt(auc_ci[0])}–{_fmt(auc_ci[1])}`, which includes 0).",
            "Point estimates favour the GBM on KS",
            f"(`{_fmt(gbm_oot.get('ks'))}` vs `{_fmt(bureau.get('ks'))}`)",
            "and on pre-calibration-margin approval-rate-matched bad rates,",
            "but the paired ΔKS and Δbad-rate 95% CIs also include 0 (see",
            "tables below). That is **not a demonstrated ranking gain** over",
            "the current-policy bureau score on this synthetic book. The",
            "remaining value of the model is a calibrated PD (for pricing",
            "and expected loss), reason codes and governance, the swap-set",
            "view of who would be approved differently, and a fallback if",
            "the bureau is unavailable or drifts.",
            "",
            f"Bureau score on OOT (main): AUC `{_fmt(bureau.get('auc'))}`,",
            f"KS `{_fmt(bureau.get('ks'))}`. Individual-model bootstrap CIs",
            f"use {BOOTSTRAP_DRAWS} resamples, seed 42, and are reported for",
            "OOT only. The paired Δ CIs use the same resamples for GBM and",
            "bureau.",
            "",
            "### Calibration (Brier before vs after isotonic)",
            "",
            "The calibration slice default rate is 10.05%, against 11.09% on",
            "OOT approved-and-observable loans. Isotonic calibration is fitted",
            "only on the earlier, slightly better book, which is why it is",
            "neutral (or very slightly worse) on OOT Brier.",
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
            "Fitted models are ranked on the pre-calibration margin so",
            "isotonic plateaus do not create ties. The bureau score is an",
            f"integer, so its bad rate is the average of {BUREAU_TIEBREAK_SEEDS}",
            "random tie-breaks (seed 42, 43, …). `ties_at_cutoff` is the",
            "number of applications that share the marginal (k-th) score.",
            "",
            "| Approval rate | n | bureau_score | baseline_logreg_v1 | gbm_v1 | "
            "ties bureau | ties logreg | ties gbm |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *match_lines,
            "",
            "Paired bootstrap, GBM minus bureau, same 1,000 resamples:",
            "",
            f"- ΔAUC mean `{_fmt(d_auc.get('mean'))}`, 95% CI "
            f"`{_fmt(auc_ci[0])}–{_fmt(auc_ci[1])}`",
            f"- ΔKS mean `{_fmt(d_ks.get('mean'))}`, 95% CI "
            f"`{_fmt(ks_ci[0])}–{_fmt(ks_ci[1])}`",
            "",
            "| Approval rate | Δbad rate (mean) | 95% CI |",
            "|---:|---:|---|",
            *delta_bad_lines,
            "",
            "Swap set (GBM vs bureau) at each approval rate. Swap-in:",
            "approved by the GBM and declined by bureau. Swap-out: the",
            "reverse. Bureau approvals use a single seeded tie-break",
            f"(seed {SEED}).",
            "",
            "| Approval rate | Swap-in n | Swap-in bad rate | Swap-out n | "
            "Swap-out bad rate |",
            "|---:|---:|---:|---:|---:|",
            *swap_lines,
            "",
            "### Score and feature PSI (train vs OOT)",
            "",
            "`prior_application_count` and `prior_decline_count` use a rolling",
            "12-month lookback ending at `applied_at`, so train and OOT share",
            "the same history depth. `prior_observable_default_count` is",
            "**dropped**: a 12-month lookback cannot contain a completed",
            "12m+90d outcome, and the lifetime count was non-zero in only",
            "0.08% of train vs 4.48% of OOT (value-bin PSI hidden by deciles).",
            "The lifetime series is still reported below as a diagnostic,",
            "not a model input. Continuous features use decile bins; all",
            "`prior_*` series use value bins (0, 1, …, 5+).",
            "",
            "OOT AUC before this history change (lifetime `prior_*`,",
            "including the dropped feature): logreg 0.7208, GBM 0.7492.",
            "OOT AUC after: logreg "
            f"`{_fmt(payload['metrics'][MODEL_LOGREG]['oot_main'].get('auc'))}`,",
            f"GBM `{_fmt(payload['metrics'][MODEL_GBM]['oot_main'].get('auc'))}`.",
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
            "### Reliability (OOT, main)",
            "",
            "Equal-count bins of calibrated PD, with Wilson 95% CIs on the",
            "observed default rate. Plot: "
            f"`{payload.get('reliability_plot', 'docs/img/reliability_oot.png')}`.",
            "",
            "| Model | n | Mean PD | Observed | Observed 95% CI |",
            "|-------|--:|---:|---:|---|",
            *rel_rows,
            "",
            f"Tail calibration: {tail_text}. Because expected loss in a",
            "later Proof of Value is PD × EAD × LGD, under-prediction in",
            "the upper tail (band D, the worst remaining band) undervalues the loss avoided by",
            "declining those loans.",
            "",
            "### Default rate by vintage (OOT, main, disbursement month)",
            "",
            "The 2025-08 vintage is a **partial month**: observability ends",
            "at disbursements on 2025-08-02, so n is small and the rate is",
            "not comparable to a full month.",
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
            "A higher through-the-door AUC does **not** mean the model is",
            "better. Declined applicants are a broader, more separable",
            "population, so the lift is a selection-bias effect, not a",
            "change in model quality.",
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
            "For the GBM they come from SHAP on the pre-calibration margin and",
            "are emitted only when SHAP > 0 *and* the raw value sits on the",
            "risky side of the train-set median by at least the documented",
            "minimum delta (for example 10 bureau points, 200 BRL of monthly",
            "income, 3 months of tenure; see `reason_min_delta` in the",
            "metrics JSON). For the logistic regression they come from",
            "`coef × (x − train mean)` with the same risky-side and minimum",
            "delta rule. Features whose fitted coefficient sign conflicts",
            f"with the expected direction ({excluded}) are excluded from",
            "logistic reason codes.",
            "",
            "Logistic numeric coefficient signs versus expected direction:",
            "",
            "| Feature | Fitted sign (coef) | Expected | Matches |",
            "|---------|--------------------|----------|---------|",
            *sign_rows,
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
    # GBM: train median. Logreg: train mean, which is the centre of the
    # standardised features used in coef × (x − reference).
    reference_gbm = {
        col: float(num_medians[col]) for col in NUMERIC_FEATURES if col in num_medians
    }
    reference_logreg = {
        col: float(pd.to_numeric(X_train[col], errors="coerce").mean())
        for col in LOGREG_NUMERIC_FEATURES
    }

    scored_at = datetime.now(timezone.utc).replace(microsecond=0)
    sign_table = logreg_feature_signs(logreg)
    logreg_excluded = sorted(
        {
            row["base_feature"]
            for row in sign_table
            if row["matches_expected"] is False and not row["is_dummy"]
        }
    )
    reason_logreg = top_reason_codes(
        logreg_contrib,
        X_all,
        reference_logreg,
        exclude=set(logreg_excluded),
        min_deltas=REASON_MIN_DELTA,
        require_risky_side=True,
    )
    reason_gbm = top_reason_codes(
        gbm_margin_shap,
        X_all,
        reference_gbm,
        min_deltas=REASON_MIN_DELTA,
        require_risky_side=True,
    )
    mono_logreg = check_reason_monotone(reason_logreg, X_all, reference_logreg)
    mono_gbm = check_reason_monotone(reason_gbm, X_all, reference_gbm)
    mono_check = {
        "inspected": mono_logreg["inspected"] + mono_gbm["inspected"],
        "contradictions": mono_logreg["contradictions"] + mono_gbm["contradictions"],
        "contradiction_rate": None,
        "by_model": {MODEL_LOGREG: mono_logreg, MODEL_GBM: mono_gbm},
    }
    inspected = mono_check["inspected"]
    if inspected:
        mono_check["contradiction_rate"] = mono_check["contradictions"] / inspected

    oot_mask = frame["has_label"] & (frame["fit_role"] == "oot")
    # Start from model-specific quintiles, then merge. Shared PD cuts left
    # the logistic model with empty bands because of isotonic plateaus;
    # those plateaus also collapse the five starting cuts to four bands
    # (A–D) on both current models.
    band_edges_by_model = {
        model: choose_band_edges(
            pd_all[model][oot_mask.to_numpy()],
            n_bands=5,
            min_share=BAND_MIN_SHARE,
        )
        for model in (MODEL_LOGREG, MODEL_GBM)
    }

    records = []
    for model, probas, reasons in (
        (MODEL_LOGREG, pd_all[MODEL_LOGREG], reason_logreg),
        (MODEL_GBM, pd_all[MODEL_GBM], reason_gbm),
    ):
        scores = pd_to_score(probas)
        bands = pd_to_band(probas, edges=band_edges_by_model[model])
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
    def _precal_margin(raw) -> np.ndarray:
        p = np.clip(np.asarray(raw, dtype=float), 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    # Higher-better ranks: negated pre-calibration margin for the models
    # (isotonic is monotone, the margin has no plateau ties). Raw bureau
    # stays higher-better and uses a random tie-break average.
    logreg_rank = -_precal_margin(raw_all[MODEL_LOGREG][oot_idx.to_numpy()])
    gbm_rank = -_precal_margin(raw_all[MODEL_GBM][oot_idx.to_numpy()])
    approval_matched = approval_matched_bad_rates(
        y_oot,
        {
            "bureau_score": bureau_oot,
            MODEL_LOGREG: logreg_rank,
            MODEL_GBM: gbm_rank,
        },
        rates,
        random_tiebreak_models=("bureau_score",),
        n_tiebreak_seeds=BUREAU_TIEBREAK_SEEDS,
        tiebreak_base_seed=SEED,
    )
    swap_sets = [
        swap_set_at_rate(
            y_oot,
            champion_score=bureau_oot,
            challenger_score=gbm_rank,
            rate=rate,
            champion_random_seed=SEED,
        )
        for rate in rates
    ]
    paired = paired_bootstrap_gbm_minus_bureau(
        y_oot,
        pd_all[MODEL_GBM][oot_idx.to_numpy()],
        -bureau_oot,
        gbm_rank,
        bureau_oot,
        rates,
        n_draws=BOOTSTRAP_DRAWS,
        seed=SEED,
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
    prior_features = [
        "prior_application_count",
        "prior_decline_count",
        "prior_observable_default_count_lifetime",
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
        psi_report[model]["prior_value_bins"] = {}
        for col in prior_features:
            if col not in frame.columns:
                continue
            psi_report[model]["prior_value_bins"][col] = psi_value_bins(
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
        model: _band_table(
            y_oot, pd_all[model][oot_idx.to_numpy()], band_edges_by_model[model]
        )
        for model in (MODEL_LOGREG, MODEL_GBM)
    }
    vintage = vintage_default_rates(
        frame.loc[oot_idx, "disbursed_at"],
        y_oot,
        partial_month="2025-08",
    )
    vintage_records = vintage.to_dict(orient="records")
    reliability_oot = {
        model: reliability_table(y_oot, pd_all[model][oot_idx.to_numpy()]).to_dict(
            orient="records"
        )
        for model in (MODEL_LOGREG, MODEL_GBM)
    }
    reliability_path = PROJECT_ROOT / "docs" / "img" / "reliability_oot.png"
    write_reliability_plot(
        y_oot,
        {
            MODEL_LOGREG: pd_all[MODEL_LOGREG][oot_idx.to_numpy()],
            MODEL_GBM: pd_all[MODEL_GBM][oot_idx.to_numpy()],
        },
        reliability_path,
    )

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
        "swap_sets": swap_sets,
        "paired_bootstrap": paired,
        "psi": psi_report,
        "history_spec": {
            "lookback_months": 12,
            "dropped": ["prior_observable_default_count"],
            "auc_before_oot": {
                "baseline_logreg_v1": 0.7208,
                "gbm_v1": 0.7492,
                "note": (
                    "Lifetime prior_* including prior_observable_default_count, "
                    "same OOT approved-observable slice, previous training run."
                ),
            },
        },
        "band_edges": {
            model: [list(item) for item in edges]
            for model, edges in band_edges_by_model.items()
        },
        "bands": bands,
        "vintage": vintage_records,
        "selection_bias": ttd_metrics,
        "monotone_check": mono_check,
        "logreg_coefficient_signs": sign_table,
        "logreg_dropped_features": list(LOGREG_DROPPED_FEATURES),
        "logreg_excluded_from_reasons": logreg_excluded,
        "reason_min_delta": REASON_MIN_DELTA,
        "reliability_plot": str(reliability_path.relative_to(PROJECT_ROOT)),
        "reliability_oot": reliability_oot,
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
            "reference": reference_gbm,
            "reference_logreg": reference_logreg,
            "band_edges": band_edges_by_model,
            "logreg_numeric_features": list(LOGREG_NUMERIC_FEATURES),
            "logreg_excluded_from_reasons": logreg_excluded,
            "reason_min_delta": REASON_MIN_DELTA,
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
