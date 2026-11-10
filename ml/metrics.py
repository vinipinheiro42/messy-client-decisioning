"""Evaluation metrics for the synthetic credit models."""

from __future__ import annotations

import math

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


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (None, None)
    k = int(k)
    n = int(n)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


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
        .agg(
            n=("y", "size"),
            defaults=("y", "sum"),
            observed_rate=("y", "mean"),
            mean_pd=("p", "mean"),
        )
        .reset_index()
    )
    lows, highs = [], []
    for n, defaults in zip(out["n"].to_numpy(), out["defaults"].to_numpy(), strict=True):
        lo, hi = wilson_interval(int(defaults), int(n))
        lows.append(lo)
        highs.append(hi)
    out["observed_lo"] = lows
    out["observed_hi"] = highs
    return out


def psi_value_bins(
    expected: np.ndarray, actual: np.ndarray, max_inclusive: int = 5
) -> float | None:
    """PSI on integer-like values (0, 1, …, max_inclusive and above).

    Decile bins collapse a rare non-zero count into a single low quantile and
    hide train-vs-OOT accumulation of history features.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return None

    def _pack(arr: np.ndarray) -> np.ndarray:
        return np.where(
            arr >= max_inclusive, max_inclusive, np.clip(np.rint(arr), 0, max_inclusive)
        )

    cats = np.arange(max_inclusive + 1)
    e_c, a_c = _pack(expected), _pack(actual)
    e_counts = np.array([(e_c == c).sum() for c in cats], dtype=float)
    a_counts = np.array([(a_c == c).sum() for c in cats], dtype=float)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-6, 1.0)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


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


def _ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    arr = np.asarray(values, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


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


def ties_at_cutoff(score, k: int) -> dict:
    """Count observations that share the marginal (k-th) score."""
    s = np.asarray(score, dtype=float)
    k = min(max(int(k), 1), len(s))
    order = np.argsort(-s, kind="mergesort")
    cutoff = s[order[k - 1]]
    at_cutoff = int(np.sum(s == cutoff)) if np.isfinite(cutoff) else 0
    strictly_better = int(np.sum(s > cutoff)) if np.isfinite(cutoff) else 0
    slots_in_group = k - strictly_better
    contested = at_cutoff > slots_in_group and strictly_better < k
    return {
        "cutoff_score": float(cutoff) if np.isfinite(cutoff) else None,
        "n_at_cutoff": at_cutoff,
        "n_strictly_better": strictly_better,
        "slots_in_tied_group": int(max(slots_in_group, 0)),
        "ties_affect_cutoff": bool(contested),
    }


def approve_indices(score, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Indices of the top-k (higher score = better). Randomly break ties if `rng` is set."""
    s = np.asarray(score, dtype=float)
    k = min(max(int(k), 1), len(s))
    if rng is None:
        return np.argsort(-s, kind="mergesort")[:k]
    noise = rng.random(len(s))
    return np.lexsort((noise, -s))[:k]


def approval_matched_bad_rates(
    y_true,
    scores_by_model: dict[str, np.ndarray],
    rates: tuple[float, ...],
    *,
    random_tiebreak_models: tuple[str, ...] = (),
    n_tiebreak_seeds: int = 20,
    tiebreak_base_seed: int = 42,
) -> list[dict]:
    """Bad rate among the top `rate` share by each score (higher score = better).

    Models in `random_tiebreak_models` (the integer bureau score) average the
    bad rate over `n_tiebreak_seeds` random tie-breaks. Other models should be
    ranked on a pre-calibration margin so isotonic plateaus do not create ties.
    """
    y = np.asarray(y_true).astype(int)
    rows = []
    n = len(y)
    for rate in rates:
        k = max(int(round(n * rate)), 1)
        row: dict = {"approval_rate": rate, "n_approved": k}
        for name, score in scores_by_model.items():
            s = np.asarray(score, dtype=float)
            info = ties_at_cutoff(s, k)
            row[f"ties_at_cutoff__{name}"] = info["n_at_cutoff"]
            row[f"ties_affect_cutoff__{name}"] = info["ties_affect_cutoff"]
            if name in random_tiebreak_models:
                rates_i = []
                for i in range(n_tiebreak_seeds):
                    rng = np.random.default_rng(tiebreak_base_seed + i)
                    idx = approve_indices(s, k, rng)
                    rates_i.append(float(y[idx].mean()))
                row[f"bad_rate__{name}"] = float(np.mean(rates_i))
                row[f"bad_rate_tiebreak_seeds__{name}"] = n_tiebreak_seeds
            else:
                idx = approve_indices(s, k, rng=None)
                row[f"bad_rate__{name}"] = float(y[idx].mean())
        rows.append(row)
    return rows


def swap_set_at_rate(
    y_true,
    champion_score,
    challenger_score,
    rate: float,
    *,
    champion_random_seed: int | None = 42,
) -> dict:
    """Who switches approval between challenger and champion at a matched rate.

    `swap_in`: approved by the challenger, declined by the champion.
    `swap_out`: approved by the champion, declined by the challenger.
    Scores are higher-better. A seeded random tie-break is used for the
    champion when `champion_random_seed` is set (bureau integer ties).
    """
    y = np.asarray(y_true).astype(int)
    n = len(y)
    k = max(int(round(n * rate)), 1)
    rng = None if champion_random_seed is None else np.random.default_rng(champion_random_seed)
    champ = set(approve_indices(champion_score, k, rng).tolist())
    chal = set(approve_indices(challenger_score, k, rng=None).tolist())
    swap_in = np.array(sorted(chal - champ), dtype=int)
    swap_out = np.array(sorted(champ - chal), dtype=int)
    return {
        "approval_rate": rate,
        "n_approved": k,
        "swap_in_n": int(len(swap_in)),
        "swap_in_bad_rate": float(y[swap_in].mean()) if len(swap_in) else None,
        "swap_out_n": int(len(swap_out)),
        "swap_out_bad_rate": float(y[swap_out].mean()) if len(swap_out) else None,
    }


def paired_bootstrap_gbm_minus_bureau(
    y_true,
    gbm_risk,
    bureau_risk,
    gbm_rank_higher_better,
    bureau_rank_higher_better,
    rates: tuple[float, ...],
    n_draws: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of GBM minus bureau on the same resamples.

    `*_risk` are in the risk direction (higher = more likely default) for AUC/KS.
    Rank scores are higher-better for approval-rate-matched bad rates. Bureau
    ties are broken with one seeded shuffle per draw.
    """
    y = np.asarray(y_true).astype(int)
    gbm_r = np.asarray(gbm_risk, dtype=float)
    bur_r = np.asarray(bureau_risk, dtype=float)
    gbm_h = np.asarray(gbm_rank_higher_better, dtype=float)
    bur_h = np.asarray(bureau_rank_higher_better, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    d_auc: list[float] = []
    d_ks: list[float] = []
    d_bad: dict[float, list[float]] = {float(r): [] for r in rates}
    if n < 2:
        return {"n_draws": 0}
    for draw in range(n_draws):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a_g = auc_score(yy, gbm_r[idx])
        a_b = auc_score(yy, bur_r[idx])
        k_g = ks_score(yy, gbm_r[idx])
        k_b = ks_score(yy, bur_r[idx])
        if a_g is not None and a_b is not None:
            d_auc.append(a_g - a_b)
        if k_g is not None and k_b is not None:
            d_ks.append(k_g - k_b)
        tie_rng = np.random.default_rng(seed + 1 + draw)
        for rate in rates:
            kk = max(int(round(n * rate)), 1)
            g_idx = approve_indices(gbm_h[idx], kk, rng=None)
            b_idx = approve_indices(bur_h[idx], kk, tie_rng)
            d_bad[float(rate)].append(float(yy[g_idx].mean() - yy[b_idx].mean()))

    def _block(values: list[float]) -> dict:
        return {
            "mean": float(np.mean(values)) if values else None,
            "ci": list(_ci(values)),
            "n_draws": len(values),
        }

    return {
        "n_draws": len(d_auc),
        "delta_auc": _block(d_auc),
        "delta_ks": _block(d_ks),
        "delta_bad_rate": {str(rate): _block(vals) for rate, vals in d_bad.items()},
    }


def choose_band_edges(
    pd_values,
    n_bands: int = 5,
    min_share: float = 0.05,
    names: tuple[str, ...] = ("A", "B", "C", "D"),
) -> tuple[tuple[str, float, float], ...]:
    """Cut model-specific PD bands, starting from quantiles then merging.

    Training requests ``n_bands=5`` (quintiles of calibrated PD). Adjacent
    cuts are then dropped when they are non-unique — isotonic calibration
    often places many loans on the same PD plateau — or when a band would
    hold less than ``min_share`` of the population (default 5%). On the
    current synthetic models that collapse yields **four bands, A–D**,
    which is what artifacts, the API, and the model card use. The
    ``names`` default is therefore A–D. If a future run keeps more unique
    cuts than ``names`` provides, pass a longer tuple; otherwise leftover
    bands are labelled B0, B1, ….
    """
    pd_arr = np.asarray(pd_values, dtype=float)
    pd_arr = pd_arr[np.isfinite(pd_arr)]
    if len(pd_arr) == 0:
        raise ValueError("No finite PD values to cut bands on")
    qs = np.quantile(pd_arr, np.linspace(0, 1, n_bands + 1))
    interiors: list[float] = []
    for q in qs[1:-1]:
        qf = float(q)
        if not interiors or qf > interiors[-1] + 1e-15:
            interiors.append(qf)
    cuts = interiors
    # Merge a cut if the resulting band would hold less than min_share.
    while True:
        edges = [0.0, *cuts, 1.01]
        shares = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            if hi >= 1.01:
                shares.append(float(np.mean((pd_arr >= lo) & (pd_arr <= 1.0))))
            else:
                shares.append(float(np.mean((pd_arr >= lo) & (pd_arr < hi))))
        if all(s >= min_share - 1e-12 for s in shares) or len(cuts) == 0:
            break
        # Drop the cut that borders the smallest band (prefer merging the thin one).
        thin = int(np.argmin(shares))
        # Remove the interior edge next to that band (the right edge, or left if last).
        if thin >= len(cuts):
            cuts.pop(-1)
        else:
            cuts.pop(thin)
    edges = [0.0, *cuts, 1.01]
    labels = names[: len(edges) - 1]
    if len(labels) < len(edges) - 1:
        labels = tuple(f"B{i}" for i in range(len(edges) - 1))
    return tuple((labels[i], edges[i], edges[i + 1]) for i in range(len(edges) - 1))


def vintage_default_rates(
    disbursed_at, y_true, partial_month: str | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "disbursed_at": pd.to_datetime(disbursed_at, utc=True),
            "y": np.asarray(y_true).astype(int),
        }
    )
    frame["vintage_month"] = (
        frame["disbursed_at"].dt.tz_convert("UTC").dt.to_period("M").astype(str)
    )
    out = (
        frame.groupby("vintage_month", as_index=False)
        .agg(n=("y", "size"), defaults=("y", "sum"), default_rate=("y", "mean"))
        .sort_values("vintage_month")
    )
    out["partial"] = out["vintage_month"].eq(partial_month) if partial_month else False
    return out
