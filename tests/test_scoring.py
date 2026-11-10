"""Tests for the M3 score mapping, leakage guard, and score-table key."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from ml.constants import (
    ALLOWED_FEATURES,
    ANCHOR_ODDS,
    ANCHOR_SCORE,
    FORBIDDEN_FEATURES,
    PDO,
)
from ml.features import assert_no_leakage, feature_matrix
from ml.scorecard import pd_to_score, score_to_pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = PROJECT_ROOT / "data" / "warehouse.duckdb"


def test_score_mapping_anchor_and_pdo() -> None:
    pd_anchor = 1.0 / (1.0 + ANCHOR_ODDS)
    scores = pd_to_score([pd_anchor])
    assert scores[0] == ANCHOR_SCORE
    # Odds doubling (PD = 1/101) must raise the score by one PDO step.
    pd_double = 1.0 / (1.0 + ANCHOR_ODDS * 2.0)
    assert pd_to_score([pd_double])[0] == ANCHOR_SCORE + PDO
    # The inverse mapping recovers the input PD near the anchor.
    recovered = score_to_pd([ANCHOR_SCORE])[0]
    assert recovered == pytest.approx(pd_anchor, rel=1e-6)
    # Clipping.
    assert pd_to_score([0.0])[0] == 900
    assert pd_to_score([1.0])[0] == 300


def test_feature_allowlist_has_no_leakage() -> None:
    assert_no_leakage(list(ALLOWED_FEATURES))
    overlap = set(ALLOWED_FEATURES) & set(FORBIDDEN_FEATURES)
    assert not overlap
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_leakage(["bureau_score", "principal"])


def test_feature_matrix_drops_post_decision_columns() -> None:
    frame = pd.DataFrame(
        {
            "requested_amount": [1000.0],
            "term_months": [12],
            "bureau_score": [700],
            "existing_debt": [0.0],
            "existing_debt_is_missing": [0],
            "monthly_income": [4000.0],
            "employment_months": [10],
            "dependents": [0],
            "age_at_applied_at": [30.0],
            "requested_to_income": [0.25],
            "debt_to_annual_income": [0.0],
            "estimated_dti": [0.3],
            "prior_application_count": [0],
            "prior_decline_count": [0],
            "prior_observable_default_count": [0],
            "purpose": ["personal"],
            "channel": ["app"],
            "employment_type": ["clt"],
            "housing_status": ["renter"],
            "state_uf": ["SP"],
            "principal": [9999.0],
            "interest_rate_monthly": [0.03],
            "decision": ["approved"],
        }
    )
    matrix = feature_matrix(frame)
    assert "principal" not in matrix.columns
    assert "decision" not in matrix.columns
    assert_no_leakage(list(matrix.columns))


@pytest.mark.skipif(not WAREHOUSE.is_file(), reason="warehouse not built")
def test_predictions_scores_primary_key() -> None:
    import duckdb

    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    try:
        con.execute("SET TimeZone = 'UTC'")
        exists = con.execute(
            """
            select count(*) from information_schema.tables
            where table_name = 'predictions_scores'
            """
        ).fetchone()[0]
        if not exists:
            pytest.skip("predictions_scores has not been written yet")
        n = con.execute("select count(*) from ml.predictions_scores").fetchone()[0]
        if n == 0:
            pytest.skip("predictions_scores is empty")
        dups = con.execute(
            """
            select count(*) from (
                select application_id, model_version
                from ml.predictions_scores
                group by 1, 2
                having count(*) > 1
            )
            """
        ).fetchone()[0]
        assert dups == 0
        versions = set(
            con.execute("select distinct model_version from ml.predictions_scores").fetchdf()[
                "model_version"
            ]
        )
        assert versions == {"baseline_logreg_v1", "gbm_v1"}
        # Two rows per application.
        apps = con.execute(
            "select count(distinct application_id) from ml.predictions_scores"
        ).fetchone()[0]
        assert n == apps * 2
        # Scores stay in range.
        bad = con.execute(
            """
            select count(*) from ml.predictions_scores
            where score < 300 or score > 900
               or pd < 0 or pd > 1
               or split not in ('train', 'validation', 'oot')
            """
        ).fetchone()[0]
        assert bad == 0
    finally:
        con.close()
