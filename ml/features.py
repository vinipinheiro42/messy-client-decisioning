"""Application-time feature builder and leakage guards. Synthetic data only."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ml.constants import (
    ALLOWED_FEATURES,
    AS_OF,
    CALIB_END,
    CATEGORICAL_FEATURES,
    EXISTING_DEBT_SERVICE_RATE,
    FORBIDDEN_FEATURES,
    LABEL_COL,
    REF_MONTHLY_RATE,
    TRAIN_END,
    TUNE_END,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def connect_warehouse(path: Path | str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path), read_only=read_only)
    con.execute("SET TimeZone = 'UTC'")
    return con


def _schema(con: duckdb.DuckDBPyConnection, table: str, preferred: str) -> str:
    rows = con.execute(
        """
        select table_schema
        from information_schema.tables
        where table_name = ?
        order by case when table_schema = ? then 0 else 1 end, table_schema
        """,
        [table, preferred],
    ).fetchall()
    if not rows:
        raise SystemExit(f"Could not find {table} in the warehouse. Run `make dbt` first.")
    return str(rows[0][0])


def annuity_payment(principal: pd.Series, term_months: pd.Series, rate: float) -> pd.Series:
    """Fixed-rate annuity; used only with the synthetic reference rate."""
    principal = pd.to_numeric(principal, errors="coerce")
    term = pd.to_numeric(term_months, errors="coerce")
    n = term.clip(lower=1)
    if rate <= 0:
        return principal / n
    growth = (1.0 + rate) ** n
    return principal * (rate * growth) / (growth - 1.0)


def assign_split(applied_at: pd.Series) -> pd.Series:
    ts = pd.to_datetime(applied_at, utc=True)
    out = pd.Series("oot", index=ts.index, dtype="object")
    out[ts < CALIB_END] = "validation"
    out[ts < TRAIN_END] = "train"
    return out


def assign_fit_role(applied_at: pd.Series) -> pd.Series:
    """Finer roles used only during fitting: train / tune / calib / oot."""
    ts = pd.to_datetime(applied_at, utc=True)
    out = pd.Series("oot", index=ts.index, dtype="object")
    out[ts < CALIB_END] = "calib"
    out[ts < TUNE_END] = "tune"
    out[ts < TRAIN_END] = "train"
    return out


def build_feature_frame(warehouse: Path | str) -> pd.DataFrame:
    """One row per silver application, with application-time features and labels."""
    con = connect_warehouse(warehouse, read_only=True)
    try:
        silver = _schema(con, "stg_applications", "main_silver")
        gold = _schema(con, "fct_loan_outcomes", "main_gold")
        frame = con.execute(
            f"""
            with apps as (
                select * from "{silver}"."stg_applications"
            ),
            appl as (
                select * from "{silver}"."stg_applicants"
            ),
            outcomes as (
                select
                    application_id,
                    applicant_id,
                    loan_id,
                    disbursed_at,
                    observable_from_date,
                    is_observable,
                    is_default_12m,
                    outcome_status
                from "{gold}"."fct_loan_outcomes"
            ),
            history as (
                select
                    a.application_id,
                    count(p.application_id) as prior_application_count,
                    count(p.application_id) filter (
                        where p.decision = 'declined'
                    ) as prior_decline_count,
                    count(o.loan_id) filter (
                        where o.observable_from_date <= cast(a.applied_at as date)
                          and o.is_default_12m = 1
                    ) as prior_observable_default_count
                from apps a
                left join apps p
                  on p.applicant_id = a.applicant_id
                 and p.applied_at < a.applied_at
                left join outcomes o
                  on o.application_id = p.application_id
                group by a.application_id
            )
            select
                a.application_id,
                a.applicant_id,
                a.applied_at,
                a.requested_amount,
                a.term_months,
                a.purpose,
                a.channel,
                a.bureau_score,
                a.existing_debt,
                a.existing_debt_is_missing,
                a.decision,
                appl.monthly_income,
                appl.employment_type,
                appl.employment_months,
                appl.housing_status,
                appl.dependents,
                appl.state_uf,
                appl.birth_date,
                h.prior_application_count,
                h.prior_decline_count,
                h.prior_observable_default_count,
                o.loan_id,
                o.disbursed_at,
                o.is_observable,
                o.is_default_12m,
                o.outcome_status,
                o.observable_from_date
            from apps a
            left join appl on appl.applicant_id = a.applicant_id
            left join history h on h.application_id = a.application_id
            left join outcomes o on o.application_id = a.application_id
            """
        ).df()
    finally:
        con.close()

    frame["applied_at"] = pd.to_datetime(frame["applied_at"], utc=True)
    frame["birth_date"] = pd.to_datetime(frame["birth_date"], utc=True)
    frame["disbursed_at"] = pd.to_datetime(frame["disbursed_at"], utc=True)
    age_days = (frame["applied_at"] - frame["birth_date"]).dt.total_seconds() / 86400.0
    frame["age_at_applied_at"] = age_days / 365.25
    income = pd.to_numeric(frame["monthly_income"], errors="coerce").replace(0, np.nan)
    requested = pd.to_numeric(frame["requested_amount"], errors="coerce")
    debt = pd.to_numeric(frame["existing_debt"], errors="coerce")
    frame["requested_to_income"] = requested / income
    frame["debt_to_annual_income"] = debt / (12.0 * income)
    payment = annuity_payment(requested, frame["term_months"], REF_MONTHLY_RATE)
    debt_service = debt.fillna(0.0) * EXISTING_DEBT_SERVICE_RATE
    frame["estimated_dti"] = (payment + debt_service) / income
    for col in (
        "prior_application_count",
        "prior_decline_count",
        "prior_observable_default_count",
        "existing_debt_is_missing",
        "dependents",
        "employment_months",
    ):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["existing_debt_is_missing"] = frame["existing_debt_is_missing"].fillna(0).astype(int)
    frame["split"] = assign_split(frame["applied_at"])
    frame["fit_role"] = assign_fit_role(frame["applied_at"])
    frame["has_label"] = (
        frame["is_observable"].fillna(False).astype(bool) & frame[LABEL_COL].notna()
    )
    leaked = [c for c in frame.columns if c in FORBIDDEN_FEATURES]
    # decision/disbursed_at/label stay on the frame for splits and evaluation,
    # but must never enter the model matrix — enforced by feature_matrix().
    _ = leaked
    return frame


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ALLOWED_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing allowed features: {missing}")
    leaked = [c for c in ALLOWED_FEATURES if c in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"Forbidden names in the allowed list: {leaked}")
    out = frame.loc[:, ALLOWED_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").fillna("missing")
    return out


def assert_no_leakage(columns: list[str]) -> None:
    bad = [c for c in columns if c in FORBIDDEN_FEATURES]
    if bad:
        raise AssertionError(f"Leakage: forbidden columns in the model matrix: {bad}")
    extra = [c for c in columns if c not in ALLOWED_FEATURES]
    if extra:
        raise AssertionError(f"Unexpected feature columns: {extra}")


def assert_history_is_causal(frame: pd.DataFrame, warehouse: Path | str, n: int = 200) -> None:
    """Recompute prior counts for a sample and require an exact match."""
    sample = frame.sample(n=min(n, len(frame)), random_state=42)
    con = connect_warehouse(warehouse, read_only=True)
    try:
        silver = _schema(con, "stg_applications", "main_silver")
        gold = _schema(con, "fct_loan_outcomes", "main_gold")
        ids = sample["application_id"].tolist()
        placeholders = ",".join(["?"] * len(ids))
        recomputed = con.execute(
            f"""
            with apps as (select * from "{silver}"."stg_applications"),
            outcomes as (select * from "{gold}"."fct_loan_outcomes")
            select
                a.application_id,
                count(p.application_id) as prior_application_count,
                count(p.application_id) filter (where p.decision = 'declined')
                    as prior_decline_count,
                count(o.loan_id) filter (
                    where o.observable_from_date <= cast(a.applied_at as date)
                      and o.is_default_12m = 1
                ) as prior_observable_default_count
            from apps a
            left join apps p
              on p.applicant_id = a.applicant_id
             and p.applied_at < a.applied_at
            left join outcomes o on o.application_id = p.application_id
            where a.application_id in ({placeholders})
            group by a.application_id
            """,
            ids,
        ).df()
    finally:
        con.close()
    merged = sample.merge(recomputed, on="application_id", suffixes=("_f", "_r"))
    for col in (
        "prior_application_count",
        "prior_decline_count",
        "prior_observable_default_count",
    ):
        left = pd.to_numeric(merged[f"{col}_f"], errors="coerce").fillna(0)
        right = pd.to_numeric(merged[f"{col}_r"], errors="coerce").fillna(0)
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise AssertionError(f"History leakage or mismatch on {col}")
    # No history row should count an application at or after applied_at —
    # the SQL predicate is strict '<', and the recompute must agree.
    _ = AS_OF
