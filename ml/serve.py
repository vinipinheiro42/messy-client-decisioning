"""Score applications with the trained artifacts. Shared by training checks and the API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.constants import MODEL_GBM, MODEL_LOGREG, MODEL_VERSIONS, NUMERIC_FEATURES
from ml.features import add_derived_features, feature_matrix
from ml.scorecard import pd_to_band, pd_to_score, top_reason_codes
from ml.train import (
    _calibrate,
    _gbm_shap,
    _logreg_contributions,
    _prepare_gbm_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts"
MODELS_PATH = ARTIFACTS / "models.joblib"
METRICS_PATH = PROJECT_ROOT / "docs" / "model_metrics.json"

_BUNDLE: dict | None = None


def load_artifacts(path: Path | str | None = None, *, force: bool = False) -> dict:
    """Load the joblib bundle written by `make train`."""
    global _BUNDLE
    target = Path(path) if path is not None else MODELS_PATH
    if force or _BUNDLE is None or path is not None:
        if not target.is_file():
            raise FileNotFoundError(
                f"Model artifacts not found at {target}. Run `make train` first."
            )
        bundle = joblib.load(target)
        if path is None:
            _BUNDLE = bundle
        return bundle
    return _BUNDLE


def applications_to_frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Turn request payloads into the same derived-feature frame training uses."""
    frame = pd.DataFrame(rows).copy() if not isinstance(rows, pd.DataFrame) else rows.copy()
    for col in ("prior_application_count", "prior_decline_count"):
        if col not in frame.columns:
            frame[col] = 0
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return add_derived_features(frame)


def _impute(matrix: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    out = matrix.copy()
    cols = [c for c in NUMERIC_FEATURES if c in out.columns and c in medians.index]
    out[cols] = out[cols].fillna(medians[cols])
    return out


def score_frame(
    frame: pd.DataFrame,
    model_version: str = MODEL_GBM,
    artifacts: dict | None = None,
) -> pd.DataFrame:
    """Score a feature frame. `frame` must already have derived columns."""
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"Unknown model_version {model_version!r}")
    artifacts = artifacts or load_artifacts()
    matrix = feature_matrix(frame)
    if model_version == MODEL_LOGREG:
        reference = artifacts.get("reference_logreg")
    else:
        reference = artifacts.get("reference")
    if reference is None:
        reference = artifacts.get("reference") or {}
    min_deltas = artifacts.get("reason_min_delta") or {}
    excluded = set(artifacts.get("logreg_excluded_from_reasons") or [])
    edges = artifacts.get("band_edges") or {}
    if isinstance(edges, dict):
        model_edges = edges.get(model_version, edges.get(MODEL_GBM))
    else:
        model_edges = edges

    if model_version == MODEL_LOGREG:
        raw = artifacts["logreg"].predict_proba(matrix)[:, 1]
        pd_hat = _calibrate(artifacts["iso_logreg"], raw)
        contrib = _logreg_contributions(artifacts["logreg"], matrix)
        reasons = top_reason_codes(
            contrib,
            matrix,
            reference,
            min_deltas=min_deltas,
            exclude=excluded,
            require_risky_side=True,
        )
    else:
        gbm_frame, _ = _prepare_gbm_frame(matrix, artifacts["cat_levels"])
        gbm_frame = _impute(gbm_frame, artifacts["num_medians"])
        raw = artifacts["gbm"].predict_proba(gbm_frame)[:, 1]
        pd_hat = _calibrate(artifacts["iso_gbm"], raw)
        shap = _gbm_shap(artifacts["gbm"], gbm_frame)
        reasons = top_reason_codes(
            shap,
            matrix,
            reference,
            min_deltas=min_deltas,
            exclude=set(),
            require_risky_side=True,
        )

    parsed = [json.loads(item) for item in reasons]
    # API returns {code, text} only (contribution stays in the warehouse JSON).
    public = [[{"code": r["code"], "text": r["text"]} for r in row[:3]] for row in parsed]
    return pd.DataFrame(
        {
            "pd": pd_hat,
            "score": pd_to_score(pd_hat),
            "risk_band": pd_to_band(pd_hat, edges=model_edges),
            "reason_codes": public,
            "model_version": model_version,
        }
    )


def load_metrics() -> dict:
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Missing {METRICS_PATH}")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
