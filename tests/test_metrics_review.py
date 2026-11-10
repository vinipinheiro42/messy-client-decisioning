"""Tests for M3 review helpers: ties, bands, reason-code gates, Wilson CIs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ml.metrics import (
    approval_matched_bad_rates,
    choose_band_edges,
    ties_at_cutoff,
    wilson_interval,
)
from ml.scorecard import top_reason_codes


def test_wilson_interval_bounds() -> None:
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert 0 < hi < 0.05
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_ties_at_cutoff_and_bureau_average() -> None:
    # 10 unique high scores, then 6 tied at the margin for k=12.
    score = np.array([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 10, 10, 10, 10, 10], dtype=float)
    info = ties_at_cutoff(score, k=12)
    assert info["n_at_cutoff"] == 6
    assert info["n_strictly_better"] == 10
    assert info["ties_affect_cutoff"] is True
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    # Unique ranking has no contested ties.
    unique = np.arange(16, 0, -1, dtype=float)
    rows = approval_matched_bad_rates(
        y,
        {"unique": unique, "tied": score},
        rates=(12 / 16,),
        random_tiebreak_models=("tied",),
        n_tiebreak_seeds=20,
        tiebreak_base_seed=42,
    )
    assert rows[0]["ties_at_cutoff__tied"] == 6
    assert rows[0]["ties_affect_cutoff__tied"] is True
    assert rows[0]["ties_affect_cutoff__unique"] is False
    assert 0.0 <= rows[0]["bad_rate__tied"] <= 1.0


def test_choose_band_edges_min_share() -> None:
    rng = np.random.default_rng(0)
    pd_values = rng.beta(2, 8, size=2000)
    edges = choose_band_edges(pd_values, n_bands=5, min_share=0.05)
    assert len(edges) == 5
    assert edges[0][0] == "A" and edges[-1][0] == "E"
    assert edges[0][1] == 0.0 and edges[-1][2] == 1.01
    for i in range(1, 5):
        assert edges[i][1] == edges[i - 1][2]
    shares = []
    for _name, lo, hi in edges:
        if hi >= 1.0:
            shares.append(np.mean((pd_values >= lo) & (pd_values <= 1.0)))
        else:
            shares.append(np.mean((pd_values >= lo) & (pd_values < hi)))
    assert min(shares) >= 0.05 - 1e-12


def test_gbm_reason_requires_risky_side_and_delta() -> None:
    # Bureau is a "down" feature: only values well below the reference count.
    contrib = pd.DataFrame({"bureau_score": [0.2, 0.2, 0.2, -0.1]})
    values = pd.DataFrame({"bureau_score": [700.0, 715.0, 730.0, 680.0]})
    reference = {"bureau_score": 720.0}
    reasons = top_reason_codes(
        contrib,
        values,
        reference,
        min_deltas={"bureau_score": 10.0},
        require_risky_side=True,
    )
    parsed = [json.loads(item) for item in reasons]
    # 700 is 20 below 720 → emit; 715 is only 5 below → skip; 730 is better → skip;
    # last row has SHAP < 0 → skip.
    assert [len(item) for item in parsed] == [1, 0, 0, 0]
    assert parsed[0][0]["code"] == "bureau_score"


def test_logreg_exclude_wrong_sign_feature() -> None:
    contrib = pd.DataFrame({"monthly_income": [0.5], "bureau_score": [0.1]})
    values = pd.DataFrame({"monthly_income": [2000.0], "bureau_score": [600.0]})
    reference = {"monthly_income": 4000.0, "bureau_score": 700.0}
    reasons = top_reason_codes(
        contrib, values, reference, exclude={"monthly_income"}
    )
    parsed = json.loads(reasons[0])
    assert [item["code"] for item in parsed] == ["bureau_score"]
