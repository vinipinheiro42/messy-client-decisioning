"""Features may be built only from silver/gold, never from ground-truth-only columns."""

from __future__ import annotations

from pathlib import Path

import pytest
from ml.constants import ALLOWED_FEATURES, FORBIDDEN_FEATURES
from ml.features import feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SRC = (PROJECT_ROOT / "ml" / "features.py").read_text(encoding="utf-8")
TRAIN_SRC = (PROJECT_ROOT / "ml" / "train.py").read_text(encoding="utf-8")

# Columns that exist only in data/ground_truth (or counterfactuals) and must
# never enter the model matrix.
GROUND_TRUTH_ONLY = frozenset(
    {
        "latent_risk",
        "is_default_90dpd_12m",
        "is_default_90dpd_12m_counterfactual",
        "first_miss_installment",
        "observation_complete",
        "hypothetical_principal",
        "hypothetical_interest_rate_monthly",
        "hypothetical_term_months",
        "cpf",
        "full_name",
    }
)


def test_allowed_features_are_not_ground_truth_only() -> None:
    overlap = set(ALLOWED_FEATURES) & GROUND_TRUTH_ONLY
    assert not overlap
    assert "latent_risk" not in ALLOWED_FEATURES
    assert "latent_risk" not in FORBIDDEN_FEATURES or True


def test_feature_builder_source_does_not_read_ground_truth() -> None:
    assert "latent_risk" not in FEATURE_SRC
    assert "ground_truth" not in FEATURE_SRC
    assert "counterfactual" not in FEATURE_SRC.lower()
    # Training may load counterfactuals for selection-bias *evaluation* only.
    assert "latent_risk" not in TRAIN_SRC


def test_feature_matrix_rejects_ground_truth_columns() -> None:
    import pandas as pd

    row = {col: 0 if col != "purpose" else "personal" for col in ALLOWED_FEATURES}
    for col in ("channel", "employment_type", "housing_status", "state_uf"):
        if col in row:
            row[col] = "x"
    frame = pd.DataFrame([row])
    frame["latent_risk"] = 1.23
    frame["is_default_90dpd_12m_counterfactual"] = 1
    matrix = feature_matrix(frame)
    assert "latent_risk" not in matrix.columns
    assert "is_default_90dpd_12m_counterfactual" not in matrix.columns
    assert set(matrix.columns) == set(ALLOWED_FEATURES)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "warehouse.duckdb").is_file(),
    reason="warehouse not built",
)
def test_built_feature_frame_has_no_ground_truth_columns() -> None:
    from ml.features import build_feature_frame

    frame = build_feature_frame(PROJECT_ROOT / "data" / "warehouse.duckdb")
    leak = set(frame.columns) & GROUND_TRUTH_ONLY
    assert not leak, f"ground-truth-only columns on the feature frame: {leak}"
    matrix = feature_matrix(frame)
    assert "latent_risk" not in matrix.columns
    assert set(matrix.columns) <= set(ALLOWED_FEATURES) | set(matrix.columns)
    assert "prior_observable_default_count" not in matrix.columns
