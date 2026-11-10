"""Plots for the synthetic model card. Not a live policy artefact."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.metrics import reliability_table


def write_reliability_plot(
    y_true,
    probs_by_model: dict[str, np.ndarray],
    path: Path,
    n_bins: int = 10,
) -> Path:
    """Reliability (calibration) plot with Wilson 95% CIs on the observed rate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_models = len(probs_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5.4 * n_models, 4.6), squeeze=False)
    for ax, (name, prob) in zip(axes[0], probs_by_model.items(), strict=True):
        table = reliability_table(y_true, prob, n_bins=n_bins)
        if table.empty:
            ax.set_title(name)
            continue
        x = table["mean_pd"].to_numpy(dtype=float)
        y = table["observed_rate"].to_numpy(dtype=float)
        lo = table["observed_lo"].to_numpy(dtype=float)
        hi = table["observed_hi"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="Observed (Wilson 95% CI)")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
        ax.set_xlim(0, max(0.6, float(np.nanmax(x)) * 1.15))
        ax.set_ylim(0, max(0.6, float(np.nanmax(hi)) * 1.15))
        ax.set_xlabel("Mean predicted PD")
        ax.set_ylabel("Observed default rate")
        ax.set_title(name)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "OOT reliability (approved, observable) — synthetic Credara data",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
