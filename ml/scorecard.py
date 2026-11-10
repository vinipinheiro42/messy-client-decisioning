"""PD to 300-900 score, risk bands, and adverse-action reason codes."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ml.constants import (
    ANCHOR_ODDS,
    ANCHOR_SCORE,
    BAND_EDGES,
    EXPECTED_DIRECTION,
    FEATURE_LABELS,
    PDO,
    SCORE_MAX,
    SCORE_MIN,
)

LN2 = math.log(2.0)
FACTOR = PDO / LN2
OFFSET = ANCHOR_SCORE - FACTOR * math.log(ANCHOR_ODDS)


def pd_to_score(pd_values) -> np.ndarray:
    """Map calibrated PD to a 300-900 score. Odds double every PDO points."""
    pd_arr = np.clip(np.asarray(pd_values, dtype=float), 1e-12, 1.0 - 1e-12)
    logits = np.log(pd_arr / (1.0 - pd_arr))
    raw = OFFSET - FACTOR * logits
    return np.clip(np.rint(raw), SCORE_MIN, SCORE_MAX).astype(int)


def score_to_pd(score) -> np.ndarray:
    scores = np.asarray(score, dtype=float)
    logits = (OFFSET - scores) / FACTOR
    return 1.0 / (1.0 + np.exp(-logits))


def pd_to_band(pd_values, edges=None) -> np.ndarray:
    cuts = tuple(edges) if edges is not None else BAND_EDGES
    pd_arr = np.asarray(pd_values, dtype=float)
    bands = np.full(pd_arr.shape, cuts[-1][0], dtype=object)
    for name, low, high in cuts:
        # The last band is closed on the right so PD = 1 is assigned.
        if high >= 1.0:
            mask = (pd_arr >= low) & (pd_arr <= 1.0)
        else:
            mask = (pd_arr >= low) & (pd_arr < high)
        bands[mask] = name
    return bands


def reason_text(feature: str, increases_because: str) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    return f"{label} is relatively {increases_because}, which increases predicted risk."


def _on_risky_side(
    feature: str, raw, ref, direction: str, min_delta: float
) -> bool:
    """True if x is on the documented risky side of the reference by min_delta."""
    try:
        x = float(raw)
        r = float(ref)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(x) or not np.isfinite(r):
        return False
    if direction == "up":
        return x >= r + min_delta
    if direction == "down":
        return x <= r - min_delta
    return abs(x - r) >= min_delta


def top_reason_codes(
    contributions: pd.DataFrame,
    feature_values: pd.DataFrame,
    reference: dict[str, float] | None = None,
    k: int = 3,
    *,
    min_deltas: dict[str, float] | None = None,
    exclude: set[str] | frozenset[str] | None = None,
    require_risky_side: bool = False,
) -> list[str]:
    """Return a JSON list (as text) of the top-k risk-increasing reasons per row.

    `contributions` are already on the 'higher means more default risk' scale
    (logreg coef*(x-ref) or SHAP on the pre-calibration margin).

    When `require_risky_side` is true (GBM), a feature is emitted only if its
    contribution is positive *and* the raw value sits on the risky side of the
    train-set reference by at least `min_deltas[feature]` (0 if omitted).
    Features in `exclude` are never emitted (wrong-sign logistic coefficients).
    """
    records: list[str] = []
    cols = list(contributions.columns)
    values = contributions.to_numpy(dtype=float)
    skipped = exclude or set()
    deltas = min_deltas or {}
    for i in range(len(contributions)):
        row = values[i]
        order = np.argsort(-row)
        chosen = []
        for idx in order:
            if len(chosen) >= k:
                break
            if not np.isfinite(row[idx]) or row[idx] <= 0:
                continue
            feature = cols[idx]
            if feature in skipped:
                continue
            direction = EXPECTED_DIRECTION.get(feature, "mixed")
            raw = (
                feature_values.iloc[i][feature]
                if feature in feature_values.columns
                else None
            )
            ref = None if reference is None else reference.get(feature)
            if require_risky_side:
                if ref is None or raw is None or isinstance(raw, str):
                    continue
                if not _on_risky_side(feature, raw, ref, direction, deltas.get(feature, 0.0)):
                    continue
            if pd.notna(raw) and ref is not None and not isinstance(raw, str):
                because = "high" if float(raw) >= float(ref) else "low"
            else:
                because = "unfavourable"
            if direction == "up":
                because = "high"
            elif direction == "down":
                because = "low"
            chosen.append(
                {
                    "code": feature,
                    "text": reason_text(feature, because),
                    "contribution": round(float(row[idx]), 6),
                }
            )
        records.append(json.dumps(chosen, ensure_ascii=True))
    return records


def check_reason_monotone(
    reasons: list[str],
    feature_values: pd.DataFrame,
    reference: dict[str, float],
) -> dict:
    """Count reason codes that contradict the documented expected direction."""
    contradictions = 0
    inspected = 0
    examples: list[dict] = []
    for i, payload in enumerate(reasons):
        items = json.loads(payload)
        for item in items:
            feature = item["code"]
            direction = EXPECTED_DIRECTION.get(feature)
            if direction not in {"up", "down"}:
                continue
            if feature not in feature_values.columns or feature not in reference:
                continue
            raw = feature_values.iloc[i][feature]
            ref = reference[feature]
            if pd.isna(raw) or ref is None:
                continue
            try:
                raw_f = float(raw)
                ref_f = float(ref)
            except (TypeError, ValueError):
                continue
            inspected += 1
            # Risk-increasing + 'up' feature should be at or above the reference.
            if direction == "up" and raw_f < ref_f:
                contradictions += 1
                if len(examples) < 5:
                    examples.append({"feature": feature, "value": raw_f, "reference": ref_f})
            if direction == "down" and raw_f > ref_f:
                contradictions += 1
                if len(examples) < 5:
                    examples.append({"feature": feature, "value": raw_f, "reference": ref_f})
    return {
        "inspected": inspected,
        "contradictions": contradictions,
        "contradiction_rate": (contradictions / inspected) if inspected else None,
        "examples": examples,
    }
