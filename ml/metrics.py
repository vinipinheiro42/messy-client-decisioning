"""Evaluation metrics for the synthetic credit models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    return y[mask], s[mask]


def auc_score(y_true, y_score) -> float | None:
    y, s = _as_arrays(y_true, y_score)
    if len(y) < 2 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def ks_score(y_true, y_score) -> float | None:
    y, s = _as_arrays(y_true, y_score)
    if len(y) < 2 or y.min() == y.max():
        return None
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(np.abs(tpr - fpr)))


def brier_score(y_true, y_prob) -> float | None:
    y, s = _as_arrays(y_true, y_prob)
    if len(y) == 0:
        return None
    return float(np.mean((s - y) ** 2))


def top_decile_bad_rate(y_true, y_score) -> float | None:
    y, s = _as_arrays(y_true, y_score)
    if len(y) == 0:
        return None
    n = max(int(np.ceil(len(y) * 0.10)), 1)
    order = np.argsort(-s)
    return float(y[order][:n].mean())


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y, s = _as_arrays(y_true, y_prob)
    if len(y) == 0:
        return pd.DataFrame()
    # Equal-count bins; fall back to a single bin if scores are nearly constant.
    try:
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(s, bins=n_bins)
    frame = pd.DataFrame({"y": y, "p": s, "bin": bins})
    out = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), observed_rate=("y", "mean"), mean_pd=("p", "mean"))
        .reset_index()
    )
    return out


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float | None:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return None
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-6, 1.0)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def bootstrap_auc_ks(y_true, y_score, n_draws: int = 1000, seed: int = 42) -> dict:
    y, s = _as_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    kss: list[float] = []
    n = len(y)
    if n < 2 or y.min() == y.max():
        return {"auc_ci": (None, None), "ks_ci": (None, None), "n_draws": 0}
    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.min() == yy.max():
            continue
        a = auc_score(yy, ss)
        k = ks_score(yy, ss)
        if a is not None:
            aucs.append(a)
        if k is not None:
            kss.append(k)

    def _ci(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return (None, None)
        arr = np.asarray(values)
        return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))

    return {
        "auc_ci": _ci(aucs),
        "ks_ci": _ci(kss),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "ks_mean": float(np.mean(kss)) if kss else None,
        "n_draws": len(aucs),
    }


def classification_metrics(y_true, y_score, y_prob=None) -> dict:
    """AUC, KS and top-decile treat `y_score` as risk-ranked (higher = more likely default)."""
    return {
        "n": int(np.isfinite(np.asarray(y_true)).sum()),
        "defaults": int(np.nansum(np.asarray(y_true))),
        "default_rate": float(np.nanmean(np.asarray(y_true))) if len(y_true) else None,
        "auc": auc_score(y_true, y_score),
        "ks": ks_score(y_true, y_score),
        "brier": brier_score(y_true, y_prob) if y_prob is not None else None,
        "top_decile_bad_rate": top_decile_bad_rate(y_true, y_score),
    }


def approval_matched_bad_rates(
    y_true, scores_by_model: dict[str, np.ndarray], rates: tuple[float, ...]
) -> list[dict]:
    """Bad rate among the top `rate` share by each score (higher score = better)."""
    y = np.asarray(y_true).astype(int)
    rows = []
    n = len(y)
    for rate in rates:
        k = max(int(round(n * rate)), 1)
        row = {"approval_rate": rate, "n_approved": k}
        for name, score in scores_by_model.items():
            s = np.asarray(score, dtype=float)
            order = np.argsort(-s)
            row[f"bad_rate__{name}"] = float(y[order][:k].mean())
        rows.append(row)
    return rows


def vintage_default_rates(disbursed_at, y_true) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "disbursed_at": pd.to_datetime(disbursed_at, utc=True),
            "y": np.asarray(y_true).astype(int),
        }
    )
    frame["vintage_month"] = (
        frame["disbursed_at"].dt.tz_convert("UTC").dt.to_period("M").astype(str)
    )
    return (
        frame.groupby("vintage_month", as_index=False)
        .agg(n=("y", "size"), defaults=("y", "sum"), default_rate=("y", "mean"))
        .sort_values("vintage_month")
    )
