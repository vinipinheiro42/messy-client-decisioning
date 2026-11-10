"""Pre-load messy Credara Fintech CSVs into canonical bronze Parquet files.

DuckDB's read_csv path is awkward for a mix of UTF-8 comma files and a
semicolon / ISO-8859-1 batch, plus mid-stream column renames. This script
reads each raw batch with the correct delimiter and encoding, maps renamed
columns to the canonical names, attaches lineage, and writes one Parquet
file per entity under data/bronze/. dbt bronze models read those files.

It also ensures an empty ml.predictions_scores table exists in the DuckDB
warehouse so gold models that depend on the declared source can compile
before the M3 scoring job has run. Existing scores are left untouched.

All data are synthetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
WAREHOUSE = DEFAULT_DATA_DIR / "warehouse.duckdb"

BATCHES = (
    {"batch_id": "batch_001", "delimiter": ",", "encoding": "utf-8"},
    {"batch_id": "batch_002", "delimiter": ";", "encoding": "iso-8859-1"},
    {"batch_id": "batch_003", "delimiter": ",", "encoding": "utf-8"},
)

RENAMES = {
    "applicants": {"income_monthly": "monthly_income"},
    "applications": {"amount_requested": "requested_amount"},
}

CANONICAL = {
    "applicants": [
        "applicant_id",
        "cpf",
        "full_name",
        "birth_date",
        "state_uf",
        "monthly_income",
        "employment_type",
        "employment_months",
        "housing_status",
        "dependents",
        "occupation_note",
        "customer_segment",
        "extracted_at",
        "batch_id",
        "source_file",
        "ingested_at",
    ],
    "applications": [
        "application_id",
        "applicant_id",
        "applied_at",
        "requested_amount",
        "term_months",
        "purpose",
        "channel",
        "bureau_score",
        "existing_debt",
        "decision",
        "decided_at",
        "source_system",
        "extracted_at",
        "batch_id",
        "source_file",
        "ingested_at",
    ],
    "loans": [
        "loan_id",
        "application_id",
        "disbursed_at",
        "principal",
        "interest_rate_monthly",
        "term_months",
        "installment_amount",
        "extracted_at",
        "batch_id",
        "source_file",
        "ingested_at",
    ],
    "installments": [
        "installment_id",
        "loan_id",
        "installment_number",
        "due_date",
        "amount_due",
        "paid_at",
        "amount_paid",
        "extracted_at",
        "batch_id",
        "source_file",
        "ingested_at",
    ],
}

ENTITIES = ("applicants", "applications", "loans", "installments")


def _read_batch_file(path: Path, delimiter: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def load_entity(data_dir: Path, entity: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch in BATCHES:
        source_file = f"{entity}.csv"
        path = data_dir / "raw" / batch["batch_id"] / source_file
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw export: {path}")
        frame = _read_batch_file(path, batch["delimiter"], batch["encoding"])
        frame = frame.rename(columns=RENAMES.get(entity, {}))
        frame["batch_id"] = batch["batch_id"]
        frame["source_file"] = source_file
        if "extracted_at" not in frame.columns:
            frame["extracted_at"] = ""
        frame["ingested_at"] = frame["extracted_at"]
        for column in CANONICAL[entity]:
            if column not in frame.columns:
                frame[column] = ""
        frames.append(frame[CANONICAL[entity]].copy())
    out = pd.concat(frames, ignore_index=True)
    return out


def ensure_predictions_scores(warehouse: Path) -> None:
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse))
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute("CREATE SCHEMA IF NOT EXISTS ml")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ml.predictions_scores (
                application_id VARCHAR,
                model_version VARCHAR,
                pd DOUBLE,
                score INTEGER,
                risk_band VARCHAR,
                reason_codes VARCHAR,
                split VARCHAR,
                scored_at TIMESTAMPTZ
            )
            """
        )
    finally:
        con.close()


def load_bronze(
    *,
    data_dir: Path | str | None = None,
    warehouse: Path | str | None = None,
) -> dict[str, int]:
    data_path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    warehouse_path = Path(warehouse) if warehouse is not None else data_path / "warehouse.duckdb"
    bronze_dir = data_path / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for entity in ENTITIES:
        frame = load_entity(data_path, entity)
        frame.to_parquet(bronze_dir / f"{entity}.parquet", index=False)
        counts[entity] = int(len(frame))
    ensure_predictions_scores(warehouse_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-load messy raw CSVs into bronze Parquet files."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--warehouse", type=Path, default=None)
    args = parser.parse_args()
    counts = load_bronze(data_dir=args.data_dir, warehouse=args.warehouse)
    print("Bronze pre-load complete (synthetic Credara Fintech exports)")
    for entity, n in counts.items():
        print(f"  {entity}: {n}")


if __name__ == "__main__":
    main()
