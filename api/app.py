"""FastAPI scoring service for the synthetic Credara credit models."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from ml.constants import MODEL_GBM, MODEL_VERSIONS
from ml.serve import applications_to_frame, load_artifacts, load_metrics, score_frame

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ScoreRequest,
    ScoreResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Credara scoring API (synthetic)",
    description=(
        "Scores application-time attributes with `gbm_v1` (default) or "
        "`baseline_logreg_v1`. History counts default to 0; pass the 12-month "
        "lookback values when they are known. All figures are synthetic."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = True
    try:
        load_artifacts()
    except FileNotFoundError:
        loaded = False
    return HealthResponse(status="ok" if loaded else "missing_artifacts", artifacts_loaded=loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        metrics = load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    splits = metrics.get("splits") or {}
    oot = {}
    for name in (*MODEL_VERSIONS, "bureau_score"):
        row = (metrics.get("metrics") or {}).get(name, {}).get("oot_main")
        if row:
            oot[name] = {
                "n": row.get("n"),
                "auc": row.get("auc"),
                "ks": row.get("ks"),
                "brier": row.get("brier"),
            }
    return ModelInfoResponse(
        default_model=MODEL_GBM,
        available_models=list(MODEL_VERSIONS),
        training_window={
            "train_end": splits.get("train_end"),
            "tune_end": splits.get("tune_end"),
            "calib_end": splits.get("calib_end"),
            "last_observable_disbursement": splits.get("last_observable_disbursement"),
            "sizes": splits.get("sizes"),
        },
        oot_metrics=oot,
        history_features=(
            "prior_application_count and prior_decline_count are optional "
            "request fields defaulting to 0. They should be the rolling "
            "12-month counts ending at applied_at. The API does not query the "
            "warehouse for history. prior_observable_default_count is not used."
        ),
    )


@app.post("/score", response_model=ScoreResponse)
def score(
    payload: ScoreRequest,
    model_version: Literal["gbm_v1", "baseline_logreg_v1"] | None = Query(
        default=None,
        description="Override the model. Defaults to gbm_v1, or to payload.model_version if set.",
    ),
) -> ScoreResponse:
    version = model_version or payload.model_version or MODEL_GBM
    if version not in MODEL_VERSIONS:
        raise HTTPException(status_code=422, detail=f"Unknown model_version {version}")
    try:
        load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    row = payload.model_dump()
    row.pop("model_version", None)
    frame = applications_to_frame([row])
    out = score_frame(frame, model_version=version).iloc[0]
    return ScoreResponse(
        pd=float(out["pd"]),
        score=int(out["score"]),
        risk_band=str(out["risk_band"]),
        reason_codes=list(out["reason_codes"]),
        model_version=version,
    )
