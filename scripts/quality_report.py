"""Measure silver cleaning against synthetic ground truth and the issues log.

Produces docs/quality_report.md and docs/quality_report.json.
All figures are computed on synthetic Credara Fintech data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AS_OF = pd.Timestamp("2026-10-31", tz="UTC")

ISSUE_TYPES = (
    "exact_duplicate",
    "near_duplicate",
    "malformed_cpf",
    "mixed_date_format",
    "currency_as_text",
    "disguised_null",
    "inconsistent_category",
    "late_arriving_event",
    "schema_change",
    "outlier",
    "orphan_fk",
)

ENTITY_OF_FILE = {
    "applicants.csv": "applicants",
    "applications.csv": "applications",
    "loans.csv": "loans",
    "installments.csv": "installments",
}

PK = {
    "applicants": "applicant_id",
    "applications": "application_id",
    "loans": "loan_id",
    "installments": "installment_id",
}

DATE_COLS = {
    "applicants": ["birth_date"],
    "applications": ["applied_at", "decided_at"],
    "loans": ["disbursed_at"],
    "installments": ["due_date", "paid_at"],
}

# Silver types these as date (not timestamp). Compare the calendar day only.
DATE_ONLY_COLS = frozenset({"birth_date", "due_date"})

MONEY_COLS = {
    "applicants": ["monthly_income"],
    "applications": ["requested_amount", "existing_debt"],
    "loans": ["principal", "installment_amount"],
    "installments": ["amount_due", "amount_paid"],
}

OUTLIER_RULES = {"outlier_income", "outlier_age", "outlier_amount"}
PARSE_DATE_RULES = {"unparseable_date"}
PARSE_MONEY_RULES = {"unparseable_money"}

EMPTY_BUCKET = {
    "n": 0,
    "recovered": 0,
    "correctly_nullified": 0,
    "day_correct": 0,
    "time_lost": 0,
    "quarantined": 0,
    "detected": 0,
    "missed": 0,
}


def _read_gt(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "ground_truth" / f"{name}.parquet")


def _silver_schema(con: duckdb.DuckDBPyConnection) -> str:
    rows = con.execute(
        """
        select table_schema
        from information_schema.tables
        where table_name = 'stg_applicants'
        order by case when table_schema = 'silver' then 0 else 1 end, table_schema
        """
    ).fetchall()
    if not rows:
        raise SystemExit("Could not find stg_applicants in the warehouse. Run `make dbt` first.")
    return str(rows[0][0])


def _load_warehouse(warehouse: Path) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        con.execute("SET TimeZone = 'UTC'")
        schema = _silver_schema(con)
        tables = {}
        for name in (
            "stg_applicants",
            "stg_applications",
            "stg_loans",
            "stg_installments",
            "stg_quarantine",
        ):
            tables[name] = con.execute(f'select * from "{schema}"."{name}"').df()
        return tables
    finally:
        con.close()


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _is_na(value) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (ValueError, TypeError):
        return False
    if isinstance(result, bool):
        return result
    return bool(result)


def _to_utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _dates_equal(left, right, column: str | None = None) -> bool:
    if _is_na(left) and _is_na(right):
        return True
    if _is_na(left) or _is_na(right):
        return False
    a = _to_utc_ts(left)
    b = _to_utc_ts(right)
    # due_date and birth_date are date columns in the contract / silver models.
    if column in DATE_ONLY_COLS:
        return a.date() == b.date()
    return a.floor("s") == b.floor("s")


def _days_equal(left, right) -> bool:
    if _is_na(left) and _is_na(right):
        return True
    if _is_na(left) or _is_na(right):
        return False
    return _to_utc_ts(left).date() == _to_utc_ts(right).date()


def _money_equal(left, right) -> bool:
    if _is_na(left) and _is_na(right):
        return True
    if _is_na(left) or _is_na(right):
        return False
    return abs(float(left) - float(right)) < 0.005


def _index_silver(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for entity, pk in PK.items():
        frame = tables[f"stg_{entity}"].copy()
        out[entity] = frame.set_index(pk, drop=False)
    return out


def _quarantine_by_key(quarantine: pd.DataFrame) -> dict[tuple, list[str]]:
    mapping: dict[tuple, list[str]] = {}
    for row in quarantine.itertuples(index=False):
        key = (str(row.source_table), str(row.row_key))
        mapping.setdefault(key, []).append(str(row.rule_failed))
    return mapping


def _schema_change_ok(data_dir: Path) -> bool:
    path = data_dir / "bronze" / "applicants.parquet"
    if not path.is_file():
        return False
    applicants = pd.read_parquet(path)
    needed = {"monthly_income", "batch_id", "source_file", "ingested_at"}
    if not needed.issubset(applicants.columns):
        return False
    if set(applicants["batch_id"].unique()) < {"batch_001", "batch_002", "batch_003"}:
        return False
    applications = pd.read_parquet(data_dir / "bronze" / "applications.parquet")
    return "requested_amount" in applications.columns


def _empty_bucket() -> dict:
    return dict(EMPTY_BUCKET)


def _field_recovered(sil_row, gt_row, entity: str, column: str) -> bool:
    if column not in sil_row.index or column not in gt_row.index:
        return False
    if column in MONEY_COLS.get(entity, []):
        return _money_equal(sil_row[column], gt_row[column])
    if column in DATE_COLS.get(entity, []):
        return _dates_equal(sil_row[column], gt_row[column], column)
    left, right = sil_row[column], gt_row[column]
    if _is_na(left) and _is_na(right):
        return True
    if _is_na(left) or _is_na(right):
        return False
    if isinstance(left, str) or isinstance(right, str):
        return str(left).strip() == str(right).strip()
    return left == right


def evaluate_issues(
    issues: pd.DataFrame,
    silver: dict[str, pd.DataFrame],
    quarantine_index: dict[tuple, list[str]],
    ground: dict[str, pd.DataFrame],
    data_dir: Path,
) -> dict[str, dict]:
    schema_ok = _schema_change_ok(data_dir)
    summary = {name: _empty_bucket() for name in ISSUE_TYPES}

    gt_by_id = {entity: frame.set_index(PK[entity], drop=False) for entity, frame in ground.items()}

    for row in issues.itertuples(index=False):
        issue_type = str(row.issue_type)
        if issue_type not in summary:
            summary[issue_type] = _empty_bucket()
        bucket = summary[issue_type]
        bucket["n"] += 1
        entity = ENTITY_OF_FILE.get(str(row.file))
        row_key = str(row.row_key)
        column = str(row.column)
        recovered = False
        correctly_nullified = False
        day_correct = False
        time_lost = False
        quarantined = False

        if issue_type == "schema_change":
            recovered = schema_ok
        elif entity is None:
            recovered = False
        else:
            q_rules = quarantine_index.get((entity, row_key), [])
            quarantined = len(q_rules) > 0
            in_silver = row_key in silver[entity].index if entity in silver else False
            in_gt = row_key in gt_by_id[entity].index if entity in gt_by_id else False
            sil_row = None
            gt_row = None
            if in_silver:
                sil_row = silver[entity].loc[row_key]
                if isinstance(sil_row, pd.DataFrame):
                    sil_row = sil_row.iloc[0]
            if in_gt:
                gt_row = gt_by_id[entity].loc[row_key]
                if isinstance(gt_row, pd.DataFrame):
                    gt_row = gt_row.iloc[0]

            if issue_type == "orphan_fk":
                quarantined = "orphan_fk" in q_rules or quarantined
                recovered = False
                if in_silver:
                    quarantined = False
            elif issue_type == "outlier":
                quarantined = bool(OUTLIER_RULES.intersection(q_rules)) or any(
                    rule.startswith("outlier") for rule in q_rules
                )
                if in_silver and in_gt and not quarantined and sil_row is not None:
                    if column == "monthly_income" and float(sil_row["monthly_income"]) < 0:
                        quarantined = False
                    elif column == "birth_date":
                        age = (
                            pd.Timestamp("2025-01-01", tz="UTC")
                            - _as_utc(pd.Series([sil_row["birth_date"]])).iloc[0]
                        ).days / 365.25
                        if age < 16 or age > 100:
                            quarantined = False
                        else:
                            recovered = True
                    elif (
                        column == "requested_amount"
                        and float(sil_row["requested_amount"]) >= 1_000_000
                    ):
                        quarantined = False
                    else:
                        recovered = True
            elif issue_type == "late_arriving_event":
                recovered = in_silver
            elif issue_type in {"exact_duplicate", "near_duplicate"}:
                if in_silver and in_gt and sil_row is not None and gt_row is not None:
                    if column in sil_row.index:
                        recovered = _field_recovered(sil_row, gt_row, entity, column)
                    else:
                        recovered = True
                elif in_silver and not in_gt:
                    recovered = not quarantined
            elif issue_type == "malformed_cpf":
                if in_silver and in_gt and sil_row is not None and gt_row is not None:
                    recovered = str(sil_row["cpf"]) == str(gt_row["cpf"])
                quarantined = "invalid_cpf_format" in q_rules
            elif issue_type == "disguised_null":
                if in_silver and in_gt and sil_row is not None and gt_row is not None:
                    if column in sil_row.index:
                        if _field_recovered(sil_row, gt_row, entity, column):
                            recovered = True
                        elif _is_na(sil_row[column]):
                            correctly_nullified = True
                quarantined = "missing_required" in q_rules or quarantined
            elif issue_type == "mixed_date_format":
                if in_silver and in_gt and sil_row is not None and gt_row is not None:
                    if column in sil_row.index and column in gt_row.index:
                        day_correct = _days_equal(sil_row[column], gt_row[column])
                        recovered = _dates_equal(sil_row[column], gt_row[column], column)
                        time_lost = day_correct and not recovered
                quarantined = bool(PARSE_DATE_RULES.intersection(q_rules)) or quarantined
            else:
                if in_silver and in_gt and sil_row is not None and gt_row is not None:
                    if column in silver[entity].columns:
                        recovered = _field_recovered(sil_row, gt_row, entity, column)
                if issue_type == "currency_as_text":
                    quarantined = bool(PARSE_MONEY_RULES.intersection(q_rules)) or quarantined
                elif issue_type == "inconsistent_category":
                    quarantined = "unmapped_category" in q_rules or quarantined

        detected = recovered or quarantined or correctly_nullified or day_correct
        if recovered:
            bucket["recovered"] += 1
        if correctly_nullified:
            bucket["correctly_nullified"] += 1
        if day_correct:
            bucket["day_correct"] += 1
        if time_lost:
            bucket["time_lost"] += 1
        if quarantined:
            bucket["quarantined"] += 1
        if detected:
            bucket["detected"] += 1
        else:
            bucket["missed"] += 1

    for bucket in summary.values():
        n = bucket["n"]
        bucket["recall"] = (bucket["detected"] / n) if n else None
        bucket["recovery_rate"] = (bucket["recovered"] / n) if n else None
    return summary


def reconcile(silver: dict[str, pd.DataFrame], ground: dict[str, pd.DataFrame]) -> dict:
    report = {}
    for entity, pk in PK.items():
        stg = silver[entity]
        gt = ground[entity]
        stg_keys = set(stg[pk].astype(str))
        gt_keys = set(gt[pk].astype(str))
        matched = stg_keys & gt_keys
        rec = {
            "ground_truth_rows": int(len(gt)),
            "silver_rows": int(len(stg)),
            "matched_keys": int(len(matched)),
            "gt_missing_from_silver": int(len(gt_keys - stg_keys)),
            "silver_not_in_gt": int(len(stg_keys - gt_keys)),
            "key_coverage": (len(matched) / len(gt_keys)) if gt_keys else None,
        }
        if matched:
            merged = stg.merge(gt, on=pk, suffixes=("_s", "_t"), how="inner")
            for col in DATE_COLS[entity]:
                if col not in stg.columns or col not in gt.columns:
                    continue
                left = pd.to_datetime(merged[f"{col}_s"], utc=True, errors="coerce")
                right = pd.to_datetime(merged[f"{col}_t"], utc=True, errors="coerce")
                both_na = left.isna() & right.isna()
                day_ok = both_na | (left.dt.date == right.dt.date)
                if col in DATE_ONLY_COLS:
                    rec[f"date_exact_match_rate__{col}"] = float(day_ok.mean())
                else:
                    exact = both_na | (left.dt.floor("s") == right.dt.floor("s"))
                    rec[f"date_exact_match_rate__{col}"] = float(exact.mean())
                    rec[f"date_day_match_rate__{col}"] = float(day_ok.mean())
                    rec[f"date_time_lost_rate__{col}"] = float((day_ok & ~exact).mean())
            for col in MONEY_COLS[entity]:
                if col not in stg.columns or col not in gt.columns:
                    continue
                left = pd.to_numeric(merged[f"{col}_s"], errors="coerce")
                right = pd.to_numeric(merged[f"{col}_t"], errors="coerce")
                both_na = left.isna() & right.isna()
                rec[f"money_exact_match_rate__{col}"] = float(
                    (both_na | ((left - right).abs() < 0.005)).mean()
                )
        report[entity] = rec
    return report


def installment_coverage_breakdown(
    silver: dict[str, pd.DataFrame],
    ground: dict[str, pd.DataFrame],
    data_dir: Path,
) -> dict:
    """Explain why silver installment coverage is below 100% versus ground truth."""
    gt = ground["installments"].copy()
    gt_loans = ground["loans"][["loan_id", "application_id"]].copy()
    gt = gt.merge(gt_loans, on="loan_id", how="left")
    sil_ids = set(silver["installments"]["installment_id"].astype(str))
    sil_loans = set(silver["loans"]["loan_id"].astype(str))
    missing = gt[~gt["installment_id"].astype(str).isin(sil_ids)].copy()

    bronze_path = data_dir / "bronze" / "installments.parquet"
    if bronze_path.is_file():
        bronze_ids = set(
            pd.read_parquet(bronze_path, columns=["installment_id"])["installment_id"].astype(str)
        )
    else:
        bronze_ids = set()

    in_bronze = missing["installment_id"].astype(str).isin(bronze_ids)
    loan_in_silver = missing["loan_id"].astype(str).isin(sil_loans)
    due = pd.to_datetime(missing["due_date"], utc=True, errors="coerce")
    due_after_cutoff = due > AS_OF

    future_loan_present = int((~in_bronze & loan_in_silver).sum())
    future_and_cascade = int((~in_bronze & ~loan_in_silver).sum())
    cascade_in_bronze = int((in_bronze & ~loan_in_silver).sum())
    other = int((in_bronze & loan_in_silver).sum())

    return {
        "gt_rows": int(len(gt)),
        "silver_rows": int(len(silver["installments"])),
        "missing": int(len(missing)),
        "coverage": (
            (len(sil_ids & set(gt["installment_id"].astype(str))) / len(gt)) if len(gt) else None
        ),
        "not_in_raw_extract": int((~in_bronze).sum()),
        "due_after_last_cutoff": int(due_after_cutoff.sum()),
        "future_schedule_parent_present": future_loan_present,
        "future_schedule_and_parent_quarantined": future_and_cascade,
        "cascade_orphan_in_raw": cascade_in_bronze,
        "other_unexpected": other,
        "parent_loan_quarantined_total": int((~loan_in_silver).sum()),
        "as_of": str(AS_OF),
    }


def _label_sql(loan_from: str, inst_from: str, as_of: str = "date '2026-10-31'") -> str:
    """AE fct_loan_outcomes rule: UTC dates, + interval 12 month, + 90 day."""
    return f"""
    with snap as (select {as_of} as as_of_date),
    loans as (
        select
            l.*,
            cast(l.disbursed_at as date) as disbursed_date,
            cast(cast(l.disbursed_at as date) + interval 12 month as date) as window_end_date,
            cast(
                cast(l.disbursed_at as date) + interval 12 month + interval 90 day
                as date
            ) as observable_from_date
        from {loan_from} l
    ),
    inst as (
        select
            i.loan_id,
            (i.paid_at is not null and i.amount_paid >= i.amount_due) as is_fully_paid,
            case
                when i.paid_at is not null and i.amount_paid >= i.amount_due
                    then greatest(
                        0,
                        date_diff('day', cast(i.due_date as date), cast(i.paid_at as date))
                    )
                when cast(i.due_date as date) < snap.as_of_date
                    then date_diff('day', cast(i.due_date as date), snap.as_of_date)
                else 0
            end as installment_dpd,
            cast(i.due_date as date) <= lo.window_end_date as is_due_in_window
        from {inst_from} i
        join loans lo on lo.loan_id = i.loan_id
        cross join snap
    ),
    agg as (
        select
            lo.loan_id,
            lo.observable_from_date,
            max(inst.installment_dpd) filter (where inst.is_due_in_window) as max_dpd_in_window
        from loans lo
        left join inst on inst.loan_id = lo.loan_id
        group by 1, 2
    )
    select
        count(*) as n_loans,
        sum((observable_from_date <= snap.as_of_date)::int) as observable,
        sum((observable_from_date > snap.as_of_date)::int) as immature,
        sum(
            (observable_from_date <= snap.as_of_date and coalesce(max_dpd_in_window, 0) >= 90)::int
        ) as defaults,
        defaults * 1.0 / nullif(observable, 0) as default_rate
    from agg
    cross join snap
    """


def outcome_comparison(
    warehouse: Path,
    silver_schema: str,
    data_dir: Path,
) -> dict:
    """Realised default rate from silver vs ground truth, same UTC contract rule."""
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        con.execute("SET TimeZone = 'UTC'")
        silver = (
            con.execute(
                _label_sql(
                    f'"{silver_schema}"."stg_loans"',
                    f'"{silver_schema}"."stg_installments"',
                )
            )
            .fetchdf()
            .iloc[0]
            .to_dict()
        )
        gt = (
            con.execute(
                _label_sql(
                    "read_parquet('"
                    + str((data_dir / "ground_truth" / "loans.parquet").as_posix())
                    + "')",
                    "read_parquet('"
                    + str((data_dir / "ground_truth" / "installments.parquet").as_posix())
                    + "')",
                )
            )
            .fetchdf()
            .iloc[0]
            .to_dict()
        )
    finally:
        con.close()

    def _clean(row: dict) -> dict:
        out = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                out[key] = None
            elif key == "default_rate":
                out[key] = float(value)
            else:
                out[key] = int(value)
        return out

    return {"silver": _clean(silver), "ground_truth": _clean(gt)}


def paid_in_full_reconciliation(
    silver: dict[str, pd.DataFrame],
    ground: dict[str, pd.DataFrame],
) -> dict:
    sil = silver["installments"]
    gt = ground["installments"]
    merged = sil.merge(gt, on="installment_id", suffixes=("_s", "_t"), how="inner")
    sil_paid = merged["paid_at_s"].notna()
    sil_full = sil_paid & (
        pd.to_numeric(merged["amount_paid_s"], errors="coerce")
        >= pd.to_numeric(merged["amount_due_s"], errors="coerce")
    )
    gt_paid = merged["paid_at_t"].notna()
    gt_full = gt_paid & (
        pd.to_numeric(merged["amount_paid_t"], errors="coerce")
        >= pd.to_numeric(merged["amount_due_t"], errors="coerce")
    )
    contradiction = sil_paid & (
        merged["amount_paid_s"].isna()
        | (pd.to_numeric(merged["amount_paid_s"], errors="coerce") == 0)
    )
    return {
        "matched_keys": int(len(merged)),
        "silver_paid_in_full": int(sil_full.sum()),
        "gt_paid_in_full": int(gt_full.sum()),
        "paid_in_full_rate_silver": float(sil_full.mean()) if len(merged) else None,
        "paid_in_full_rate_gt": float(gt_full.mean()) if len(merged) else None,
        "gt_full_but_silver_not": int((gt_full & ~sil_full).sum()),
        "silver_full_but_gt_not": int((sil_full & ~gt_full).sum()),
        "paid_at_with_zero_or_null_amount": int(contradiction.sum()),
    }


def _pct2(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _outcome_row(label: str, row: dict) -> str:
    return (
        f"| {label} | {row.get('n_loans', 'n/a')} | "
        f"{row.get('observable', 'n/a')} | {row.get('immature', 'n/a')} | "
        f"{row.get('defaults', 'n/a')} | {_pct2(row.get('default_rate'))} |"
    )


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def render_markdown(payload: dict) -> str:
    lines = [
        "# Silver quality report",
        "",
        "All figures in this report are computed on **synthetic** Credara data",
        "(fictional client: Credara Fintech).",
        "Cleaning is measured against `data/ground_truth/` and `data/raw/_injected_issues.csv`.",
        "",
        f"Generated at warehouse `{payload['warehouse']}`.",
        "",
        "## Per-issue-type results",
        "",
        "A row is **detected** when silver handled the issue correctly, even if the",
        "original ground-truth value could not be restored. That includes quarantining",
        "the raw version, standardising a field so it matches ground truth",
        "(**recovered**), turning a disguised null into a real `NULL`",
        "(**correctly nullified**), or parsing a mixed date onto the correct calendar",
        "day (**day-correct**). **Recall** is detected / injected.",
        "**Recovery rate** is recovered / injected: the share of injections where the",
        "ground-truth value itself could be restored (for example because an earlier",
        "clean version of the row survived).",
        "",
        (
            "| issue_type | injected | recovered | quarantined | detected "
            "| missed | recall | recovery_rate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ISSUE_TYPES:
        row = payload["issue_types"][name]
        lines.append(
            f"| `{name}` | {row['n']} | {row['recovered']} | {row['quarantined']} | "
            f"{row['detected']} | {row['missed']} | {_pct(row['recall'])} | "
            f"{_pct(row['recovery_rate'])} |"
        )

    mixed = payload["issue_types"]["mixed_date_format"]
    lines += [
        "",
        "### `mixed_date_format` — day-level vs time-of-day",
        "",
        "Many injected timestamp formats are date-only strings (for example",
        "`dd/mm/yyyy`) and literally do not carry a time of day. Silver can parse the",
        "calendar day and still cannot restore the original time. Date columns",
        "(`birth_date`, `due_date`) are compared on the calendar day only.",
        "",
        f"- Injected: {mixed['n']}",
        f"- Day-correct (calendar day matches ground truth): {mixed['day_correct']}",
        "- Time-of-day lost (day-correct but the timestamp time does not match): "
        f"{mixed['time_lost']}",
        "- Fully recovered (exact timestamp, or exact date for date columns): "
        f"{mixed['recovered']}",
        f"- Quarantined (unparseable): {mixed['quarantined']}",
        f"- Missed (day wrong, or the entity is absent and not quarantined): {mixed['missed']}",
        f"- Detection recall: {_pct(mixed['recall'])}",
        f"- Recovery rate: {_pct(mixed['recovery_rate'])}",
        "",
        "### `disguised_null` — detection vs recoverability",
        "",
    ]
    nulls = payload["issue_types"]["disguised_null"]
    lines += [
        "A disguised null that silver correctly stores as a real `NULL` is **detected**",
        "but not **recovered**: the original value was overwritten in the raw export and",
        "cannot be invented back. Recovery happens only when a clean earlier or later",
        "version of the same key is the one that survives dedup. Required-field",
        "disguised nulls are quarantined (also detected).",
        "",
        f"- Injected: {nulls['n']}",
        f"- Recovered (ground-truth value restored): {nulls['recovered']}",
        f"- Correctly nullified (silver is `NULL`): {nulls['correctly_nullified']}",
        f"- Quarantined: {nulls['quarantined']}",
        f"- Missed: {nulls['missed']}",
        f"- Detection recall: {_pct(nulls['recall'])}",
        f"- Recovery rate: {_pct(nulls['recovery_rate'])}",
        "",
        "## Row-level reconciliation (silver vs ground truth)",
        "",
    ]
    for entity, rec in payload["reconciliation"].items():
        lines.append(f"### {entity}")
        lines.append("")
        lines.append(f"- Ground truth rows: {rec['ground_truth_rows']}")
        lines.append(f"- Silver rows: {rec['silver_rows']}")
        lines.append(f"- Matched keys: {rec['matched_keys']}")
        lines.append(f"- Ground-truth keys missing from silver: {rec['gt_missing_from_silver']}")
        lines.append(f"- Silver keys not in ground truth: {rec['silver_not_in_gt']}")
        cov = rec.get("key_coverage")
        lines.append(f"- Key coverage: {'n/a' if cov is None else f'{cov:.2%}'}")
        for key, value in rec.items():
            if key.startswith("date_exact_match_rate__"):
                col = key.replace("date_exact_match_rate__", "")
                lines.append(f"- date exact-match {col}: {value:.2%}")
            elif key.startswith("date_day_match_rate__"):
                col = key.replace("date_day_match_rate__", "")
                lines.append(f"- date day-level match {col}: {value:.2%}")
            elif key.startswith("date_time_lost_rate__"):
                col = key.replace("date_time_lost_rate__", "")
                lines.append(f"- date time-of-day lost {col}: {value:.2%}")
            elif key.startswith("money_exact_match_rate__"):
                col = key.replace("money_exact_match_rate__", "")
                lines.append(f"- money exact-match {col}: {value:.2%}")
        lines.append("")

    cov = payload.get("installment_coverage") or {}
    lines += [
        "## Why installment coverage is not 100%",
        "",
        "Ground truth holds the **full contractual schedule**, including installments",
        "that are not yet due as of the last client extract. Silver can only contain",
        "rows that appeared in a raw batch or that were quarantined from one.",
        "The last batch cutoff / observation date is "
        f"{cov.get('as_of', '2026-10-31 UTC')}. The generator only exports",
        "installments with `due_date <=` that cutoff, which is realistic for a",
        "snapshot extract.",
        "",
        f"- Ground-truth installments: {cov.get('gt_rows', 'n/a')}",
        f"- Silver installments: {cov.get('silver_rows', 'n/a')}",
        f"- Missing from silver: {cov.get('missing', 'n/a')}",
        f"- Coverage: {_pct(cov.get('coverage'))}",
        "- Not present in any raw extract (due after the last cutoff): "
        f"{cov.get('not_in_raw_extract', 'n/a')}",
        "- parent loan is in silver (future schedule only): "
        f"{cov.get('future_schedule_parent_present', 'n/a')}",
        "- parent loan also missing (future schedule and cascade): "
        f"{cov.get('future_schedule_and_parent_quarantined', 'n/a')}",
        "- Present in raw but quarantined as `orphan_fk` because the parent "
        f"loan was rejected: {cov.get('cascade_orphan_in_raw', 'n/a')}",
        "- Parent loan quarantined, all missing installments: "
        f"{cov.get('parent_loan_quarantined_total', 'n/a')}",
        "- Present in raw with a parent loan in silver, but not kept in silver "
        "(quarantined as inconsistent_payment or another row rule): "
        f"{cov.get('other_unexpected', 'n/a')}",
        "",
        "In plain words: most of the gap versus the full ground-truth schedule is",
        "**future installments the client has not yet exported** (due after",
        "2026-10-31). A further share is **cascade from quarantined parents**: if an",
        "applicant, application, or loan fails a silver rule, child installments are",
        "quarantined as orphans. A smaller share is **inconsistent_payment**: a raw",
        "row with `paid_at` filled and `amount_paid` missing is quarantined so a",
        "previous clean version can survive; if none exists, the installment is absent.",
        "Valid, already-exported, consistent installments are not dropped silently.",
        "",
        "## Realised default rate (silver vs ground truth)",
        "",
        "The label uses the same rule as Analytics Engineering `fct_loan_outcomes`:",
        "a loan is in default if any installment due on or before",
        "`cast(disbursed_at as date) + interval 12 month` reaches 90+ DPD, computed",
        "from `due_date` and `paid_at` against 2026-10-31 UTC. A loan is observable",
        "when that window plus 90 days is on or before the cutoff. All calendar dates",
        "are UTC (`SET TimeZone = 'UTC'`). Host local time (America/Sao_Paulo) is not",
        "used, because it would shift 00:00–02:59 UTC timestamps to the previous day",
        "and can move loans across the observability boundary.",
        "",
    ]
    sil_o = (payload.get("outcomes") or {}).get("silver") or {}
    gt_o = (payload.get("outcomes") or {}).get("ground_truth") or {}
    lines += [
        "| source | loans | observable | immature | defaults | default rate |",
        "|---|---:|---:|---:|---:|---:|",
        _outcome_row("silver (contract rule)", sil_o),
        _outcome_row("ground truth (same rule)", gt_o),
        "",
                "The loan counts differ because some loans never reach silver (quarantine or",
        "parent cascade). On the same rule and UTC dates, the rates should be close;",
        "a large gap is a cleaning bug, not a definition mismatch.",
        "",
        "## Paid-in-full reconciliation (matched installments)",
        "",
        "An installment is paid in full when `paid_at` is not null and",
        "`amount_paid >= amount_due`. Silver must not store `amount_paid = 0` (or null)",
        "alongside a filled `paid_at`.",
        "",
    ]
    pf = payload.get("paid_in_full") or {}
    lines += [
        f"- Matched keys: {pf.get('matched_keys', 'n/a')}",
        f"- Paid in full in silver: {pf.get('silver_paid_in_full', 'n/a')} "
        f"({_pct(pf.get('paid_in_full_rate_silver'))})",
        f"- Paid in full in ground truth: {pf.get('gt_paid_in_full', 'n/a')} "
        f"({_pct(pf.get('paid_in_full_rate_gt'))})",
        f"- Ground-truth paid-in-full but not in silver: {pf.get('gt_full_but_silver_not', 'n/a')}",
        f"- Silver paid-in-full but not in ground truth: {pf.get('silver_full_but_gt_not', 'n/a')}",
        f"- `paid_at` present with zero or null `amount_paid`: "
        f"{pf.get('paid_at_with_zero_or_null_amount', 'n/a')}",
        "",
        "## Notable misses and caveats",
        "",
        payload.get("notes", "See JSON for details."),
        "",
        "This report does not use data from any real lender.",
        "",
    ]
    return "\n".join(lines)


def build_notes(issue_summary: dict, reconciliation: dict, tables: dict) -> str:
    bits = []
    for name, row in issue_summary.items():
        if row["n"] and row["missed"]:
            bits.append(
                f"- `{name}`: {row['missed']} of {row['n']} injected rows were not "
                f"detected ({row['missed'] / row['n']:.1%} miss rate). "
                f"Recovered {row['recovered']} ({_pct(row['recovery_rate'])})."
            )
    for entity, rec in reconciliation.items():
        if rec["gt_missing_from_silver"]:
            bits.append(
                f"- {entity}: {rec['gt_missing_from_silver']} ground-truth keys are absent "
                "from silver. For applicants, applications and loans the usual cause is "
                "an outlier or required-field failure on every raw version, or a cascade "
                "from a quarantined parent. For installments, see the coverage section: "
                "most absences are future dues not present in the raw extracts."
            )
        if rec["silver_not_in_gt"]:
            bits.append(
                f"- {entity}: {rec['silver_not_in_gt']} silver keys are not in ground truth "
                "(unexpected; investigate orphans leaking into silver)."
            )
    n_q = len(tables["stg_quarantine"])
    bits.append(
        f"- `stg_quarantine` contains {n_q} rejected raw rows (all versions that failed a rule)."
    )
    bits.append(
        "- Near-duplicates were injected on a later batch. Silver keeps the latest "
        "conforming `ingested_at`. Trailing-space name/channel mutations are trimmed "
        "and should match ground truth; a later numeric drift (for example amount + 1) "
        "is kept and is counted as not recovered if it disagrees with ground truth."
    )
    bits.append(
        "- `existing_debt` is left `NULL` when the surviving raw value is missing or a "
        "disguised null; `existing_debt_is_missing` is true on those rows. A missing "
        "balance is never imputed as 0. `amount_paid` is 0 only when the installment "
        "is unpaid (`paid_at` is null). A paid_at with a missing amount is quarantined "
        "as inconsistent_payment so an earlier clean version can survive."
    )
    bits.append(
        "- A small number of field-level misses may be keys that were quarantined "
        "under a different rule (for example an outlier) and are therefore not "
        "attributed to the injected issue type."
    )
    bits.append(
        "- Late-arriving installments are expected to appear in silver once the later "
        "batch is loaded; they are not quarantined solely for arriving late."
    )
    return "\n".join(bits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the silver quality report.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--warehouse", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=PROJECT_ROOT / "docs" / "quality_report.md")
    parser.add_argument(
        "--json-out", type=Path, default=PROJECT_ROOT / "docs" / "quality_report.json"
    )
    args = parser.parse_args()
    warehouse = args.warehouse or args.data_dir / "warehouse.duckdb"
    if not warehouse.is_file():
        raise SystemExit(f"Warehouse not found: {warehouse}. Run `make dbt` first.")

    issues = pd.read_csv(args.data_dir / "raw" / "_injected_issues.csv")
    ground = {
        name: _read_gt(args.data_dir, name)
        for name in ("applicants", "applications", "loans", "installments")
    }
    tables = _load_warehouse(warehouse)
    silver = _index_silver(tables)
    q_index = _quarantine_by_key(tables["stg_quarantine"])
    issue_summary = evaluate_issues(issues, silver, q_index, ground, args.data_dir)
    silver_by_entity = {k: tables[f"stg_{k}"] for k in PK}
    reconciliation = reconcile(silver_by_entity, ground)
    coverage = installment_coverage_breakdown(silver_by_entity, ground, args.data_dir)
    con_tmp = duckdb.connect(str(warehouse), read_only=True)
    try:
        con_tmp.execute("SET TimeZone = 'UTC'")
        silver_schema = _silver_schema(con_tmp)
    finally:
        con_tmp.close()
    outcomes = outcome_comparison(warehouse, silver_schema, args.data_dir)
    paid_full = paid_in_full_reconciliation(silver_by_entity, ground)
    notes = build_notes(issue_summary, reconciliation, tables)
    payload = {
        "client": "Credara Fintech (fictional)",
        "is_synthetic": True,
        "warehouse": str(warehouse),
        "issue_types": issue_summary,
        "reconciliation": reconciliation,
        "installment_coverage": coverage,
        "outcomes": outcomes,
        "paid_in_full": paid_full,
        "quarantine_rows": int(len(tables["stg_quarantine"])),
        "notes": notes,
    }
    args.json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.md_out}")
    print(f"Wrote {args.json_out}")
    for name in ISSUE_TYPES:
        row = issue_summary[name]
        print(
            f"  {name}: n={row['n']} recovered={row['recovered']} "
            f"nullified={row['correctly_nullified']} day_correct={row['day_correct']} "
            f"time_lost={row['time_lost']} quarantined={row['quarantined']} "
            f"missed={row['missed']} recall={_pct(row['recall'])} "
            f"recovery={_pct(row['recovery_rate'])}"
        )
    print(
        "  installment coverage: "
        f"missing={coverage['missing']} future={coverage['not_in_raw_extract']} "
        f"cascade_parent={coverage['parent_loan_quarantined_total']} "
        f"other={coverage['other_unexpected']}"
    )
    sil_o = outcomes["silver"]
    gt_o = outcomes["ground_truth"]
    print(
        "  default rate silver: "
        f"obs={sil_o['observable']} immature={sil_o['immature']} "
        f"defaults={sil_o['defaults']} rate={_pct(sil_o['default_rate'])}"
    )
    print(
        "  default rate GT:     "
        f"obs={gt_o['observable']} immature={gt_o['immature']} "
        f"defaults={gt_o['defaults']} rate={_pct(gt_o['default_rate'])}"
    )
    print(
        "  paid-in-full: "
        f"silver={paid_full['silver_paid_in_full']} "
        f"gt={paid_full['gt_paid_in_full']} "
        f"gt_full_silver_not={paid_full['gt_full_but_silver_not']} "
        f"paid_at_zero_amt={paid_full['paid_at_with_zero_or_null_amount']}"
    )


if __name__ == "__main__":
    main()
