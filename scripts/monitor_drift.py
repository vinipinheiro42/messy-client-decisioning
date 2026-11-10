"""PSI drift monitor: train reference versus a scoring batch (synthetic data)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.constants import MODEL_GBM  # noqa: E402
from ml.features import build_feature_frame  # noqa: E402
from ml.metrics import psi, psi_value_bins  # noqa: E402
from ml.serve import score_frame  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = PROJECT_ROOT / "data" / "warehouse.duckdb"
DOCS = PROJECT_ROOT / "docs"

WARN = 0.10
ALERT = 0.25
CONTINUOUS = [
    "score",
    "bureau_score",
    "estimated_dti",
    "requested_to_income",
    "debt_to_annual_income",
    "monthly_income",
    "existing_debt",
]
VALUE_BIN = ["prior_application_count", "prior_decline_count"]


def _flag(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= ALERT:
        return "alert"
    if value >= WARN:
        return "warn"
    return "ok"


def _series(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce")


def compute_psi_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    rows = []
    for col in CONTINUOUS:
        if col not in reference.columns or col not in current.columns:
            continue
        value = psi(_series(reference, col).to_numpy(), _series(current, col).to_numpy())
        rows.append({"feature": col, "binning": "decile", "psi": value, "flag": _flag(value)})
    for col in VALUE_BIN:
        if col not in reference.columns or col not in current.columns:
            continue
        value = psi_value_bins(_series(reference, col).to_numpy(), _series(current, col).to_numpy())
        rows.append({"feature": col, "binning": "value", "psi": value, "flag": _flag(value)})
    worst = "ok"
    for row in rows:
        if row["flag"] == "alert":
            worst = "alert"
        elif row["flag"] == "warn" and worst != "alert":
            worst = "warn"
    return {
        "thresholds": {"warn": WARN, "alert": ALERT},
        "overall_flag": worst,
        "series": rows,
    }


def _with_scores(frame: pd.DataFrame, model: str = MODEL_GBM) -> pd.DataFrame:
    scored = score_frame(frame, model_version=model)
    out = frame.copy()
    out["pd"] = scored["pd"].to_numpy()
    out["score"] = scored["score"].to_numpy()
    return out


def render_markdown(payload: dict) -> str:
    lines = [
        "# Drift report — synthetic Credara Fintech",
        "",
        "All figures are computed on **synthetic** data. PSI compares a",
        "training-window reference to a scoring batch. Warn at 0.10, alert at 0.25.",
        "",
        f"Generated at `{payload['generated_at']}` (UTC).",
        "",
    ]
    for name, report in payload["batches"].items():
        lines += [
            f"## {name}",
            "",
            f"{report.get('description', '')}",
            "",
            f"- n reference: {report.get('n_reference')}",
            f"- n current: {report.get('n_current')}",
            f"- Overall flag: **{report.get('overall_flag')}**",
            "",
            "| Feature | Binning | PSI | Flag |",
            "|---------|---------|----:|------|",
        ]
        for row in report["series"]:
            psi_v = "n/a" if row["psi"] is None else f"{row['psi']:.4f}"
            lines.append(f"| `{row['feature']}` | {row['binning']} | {psi_v} | {row['flag']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=WAREHOUSE)
    parser.add_argument("--model", default=MODEL_GBM)
    args = parser.parse_args()

    frame = build_feature_frame(args.warehouse)
    labelled = frame[frame["has_label"]].copy()
    train = labelled[labelled["fit_role"] == "train"].copy()
    oot = labelled[labelled["fit_role"] == "oot"].copy()
    train = _with_scores(train, args.model)
    oot = _with_scores(oot, args.model)

    def _shift(points: int) -> pd.DataFrame:
        out = oot.copy()
        out["bureau_score"] = pd.to_numeric(out["bureau_score"], errors="coerce") - points
        return _with_scores(out, args.model)

    shifted_40 = _shift(40)
    shifted_80 = _shift(80)

    oot_report = compute_psi_report(train, oot)
    oot_report["description"] = (
        "OOT approved-and-observable versus the train labelled slice. "
        "This is the natural time shift, not an injected perturbation."
    )
    oot_report["n_reference"] = int(len(train))
    oot_report["n_current"] = int(len(oot))

    shift40 = compute_psi_report(train, shifted_40)
    shift40["description"] = (
        "Synthetic shifted batch: OOT with `bureau_score` reduced by 40 "
        "points, then rescored. This typically lands in the warn band."
    )
    shift40["n_reference"] = int(len(train))
    shift40["n_current"] = int(len(shifted_40))

    shift80 = compute_psi_report(train, shifted_80)
    shift80["description"] = (
        "Synthetic shifted batch: OOT with `bureau_score` reduced by 80 "
        "points, then rescored, to exercise the alert path (PSI ≥ 0.25)."
    )
    shift80["n_reference"] = int(len(train))
    shift80["n_current"] = int(len(shifted_80))

    payload = {
        "client": "Credara Fintech (fictional)",
        "is_synthetic": True,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_version": args.model,
        "batches": {
            "oot": oot_report,
            "bureau_score_shifted_minus_40": shift40,
            "bureau_score_shifted_minus_80": shift80,
        },
    }
    md = render_markdown(payload)
    (DOCS / "drift_report.md").write_text(md, encoding="utf-8")
    (DOCS / "drift_report.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Wrote {DOCS / 'drift_report.md'}")
    print("OOT overall:", oot_report["overall_flag"])
    for name, report in payload["batches"].items():
        if name == "oot":
            continue
        bureau = next(r for r in report["series"] if r["feature"] == "bureau_score")
        score = next(r for r in report["series"] if r["feature"] == "score")
        print(
            f"{name}: overall={report['overall_flag']} "
            f"bureau PSI={bureau['psi']:.4f} ({bureau['flag']}) "
            f"score PSI={score['psi']:.4f} ({score['flag']})"
        )


if __name__ == "__main__":
    main()
