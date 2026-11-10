"""FastAPI scoring tests: validation, batch parity, and local latency."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from ml.constants import MODEL_GBM, MODEL_LOGREG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = PROJECT_ROOT / "data" / "warehouse.duckdb"
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts" / "models.joblib"

VALID = {
    "applied_at": "2025-03-15T12:00:00Z",
    "birth_date": "1990-06-01T00:00:00Z",
    "requested_amount": 8000,
    "term_months": 12,
    "purpose": "personal",
    "channel": "app",
    "bureau_score": 720,
    "existing_debt": 2500.0,
    "monthly_income": 4200.0,
    "employment_type": "clt",
    "employment_months": 24,
    "housing_status": "renter",
    "dependents": 1,
    "state_uf": "SP",
}


def _as_int(value, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return default
    return int(value)


def _as_float_or_none(value, missing: bool = False) -> float | None:
    if missing or value is None or pd.isna(value):
        return None
    return float(value)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from api.app import app
    from ml.serve import load_artifacts

    load_artifacts(force=True)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_model_info(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] in {"ok", "missing_artifacts"}
    if not ARTIFACTS.is_file():
        pytest.skip("artifacts not trained")
    assert body["artifacts_loaded"] is True
    info = client.get("/model-info")
    assert info.status_code == 200
    payload = info.json()
    assert payload["default_model"] == MODEL_GBM
    assert MODEL_LOGREG in payload["available_models"]
    assert "oot_metrics" in payload
    assert "12-month" in payload["history_features"]


def test_valid_score_request(client: TestClient) -> None:
    if not ARTIFACTS.is_file():
        pytest.skip("artifacts not trained")
    res = client.post("/score", json=VALID)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["model_version"] == MODEL_GBM
    assert 0.0 < body["pd"] < 1.0
    assert 300 <= body["score"] <= 900
    assert body["risk_band"] in list("ABCDE")
    assert isinstance(body["reason_codes"], list)
    assert len(body["reason_codes"]) <= 3
    for item in body["reason_codes"]:
        assert "code" in item and "text" in item
        assert "contribution" not in item
    # Query override plus nullable existing_debt.
    res2 = client.post(
        "/score",
        params={"model_version": MODEL_LOGREG},
        json={**VALID, "existing_debt": None},
    )
    assert res2.status_code == 200, res2.text
    assert res2.json()["model_version"] == MODEL_LOGREG


def test_validation_errors(client: TestClient) -> None:
    bad_bureau = {**VALID, "bureau_score": 90}
    res = client.post("/score", json=bad_bureau)
    assert res.status_code == 422
    missing = {k: v for k, v in VALID.items() if k != "monthly_income"}
    res = client.post("/score", json=missing)
    assert res.status_code == 422
    negative_debt = {**VALID, "existing_debt": -1}
    res = client.post("/score", json=negative_debt)
    assert res.status_code == 422
    bad_enum = {**VALID, "purpose": "yacht"}
    res = client.post("/score", json=bad_enum)
    assert res.status_code == 422
    declines = {**VALID, "prior_application_count": 0, "prior_decline_count": 1}
    res = client.post("/score", json=declines)
    assert res.status_code == 422


@pytest.mark.skipif(
    not WAREHOUSE.is_file() or not ARTIFACTS.is_file(),
    reason="need warehouse and artifacts",
)
def test_api_parity_with_batch_scores(client: TestClient) -> None:
    import duckdb
    from ml.features import build_feature_frame

    frame = build_feature_frame(WAREHOUSE)
    oot = frame[frame["has_label"] & (frame["fit_role"] == "oot")].copy()
    required = [
        "applied_at",
        "birth_date",
        "requested_amount",
        "term_months",
        "purpose",
        "channel",
        "bureau_score",
        "monthly_income",
        "employment_type",
        "employment_months",
        "state_uf",
    ]
    oot = oot.dropna(subset=required)
    sample = oot.sample(n=50, random_state=42)
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    try:
        con.execute("SET TimeZone = 'UTC'")
        batch = con.execute(
            """
            select application_id, model_version, pd
            from ml.predictions_scores
            where application_id in (select unnest(?))
            """,
            [sample["application_id"].tolist()],
        ).df()
    finally:
        con.close()
    assert len(batch) == 100  # 50 apps × 2 models
    diffs = []
    for model in (MODEL_GBM, MODEL_LOGREG):
        expected = batch[batch["model_version"] == model].set_index("application_id")["pd"]
        for _, row in sample.iterrows():
            payload = {
                "applied_at": row["applied_at"].isoformat(),
                "birth_date": row["birth_date"].isoformat(),
                "requested_amount": float(row["requested_amount"]),
                "term_months": _as_int(row["term_months"]),
                "purpose": row["purpose"],
                "channel": row["channel"],
                "bureau_score": _as_int(row["bureau_score"]),
                "existing_debt": _as_float_or_none(
                    row["existing_debt"], bool(row["existing_debt_is_missing"])
                ),
                "monthly_income": float(row["monthly_income"]),
                "employment_type": row["employment_type"],
                "employment_months": _as_int(row["employment_months"]),
                "housing_status": (
                    None
                    if pd.isna(row["housing_status"])
                    else row["housing_status"]
                ),
                "dependents": _as_int(row["dependents"]),
                "state_uf": row["state_uf"],
                "prior_application_count": _as_int(row["prior_application_count"]),
                "prior_decline_count": _as_int(row["prior_decline_count"]),
            }
            res = client.post("/score", params={"model_version": model}, json=payload)
            assert res.status_code == 200, res.text
            got = res.json()["pd"]
            exp = float(expected.loc[row["application_id"]])
            diffs.append(abs(got - exp))
            assert got == pytest.approx(exp, abs=1e-9)
    assert max(diffs) < 1e-9


@pytest.mark.skipif(not ARTIFACTS.is_file(), reason="artifacts not trained")
def test_score_latency_p95(client: TestClient, capsys: pytest.CaptureFixture[str]) -> None:
    # Warm up once so the first SHAP/ explainer cost is not in the sample.
    assert client.post("/score", json=VALID).status_code == 200
    times = []
    for _ in range(200):
        start = time.perf_counter()
        res = client.post("/score", json=VALID)
        elapsed = time.perf_counter() - start
        assert res.status_code == 200
        times.append(elapsed)
    p95 = float(np.quantile(times, 0.95))
    median = float(np.median(times))
    print(f"API local latency: median={median*1000:.1f} ms, p95={p95*1000:.1f} ms (n=200)")
    # Generous bound so the test is a regression guard, not a flaky SLA.
    assert p95 < 2.0
