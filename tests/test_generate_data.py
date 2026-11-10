"""Tests for the synthetic Credara Fintech data generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from generate_data import (
    AS_OF,
    ISSUE_TYPES,
    generate,
    is_observation_complete,
    is_valid_cpf,
    observable_from_date,
    window_end_date,
)

MEDIUM_APPLICANTS = 2000
MEDIUM_APPLICATIONS = 5000
SMALL_APPLICANTS = 250
SMALL_APPLICATIONS = 600


@pytest.fixture(scope="module")
def medium(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, Path]:
    path = tmp_path_factory.mktemp("medium")
    result = generate(
        seed=42,
        n_applicants=MEDIUM_APPLICANTS,
        n_applications=MEDIUM_APPLICATIONS,
        output_dir=path,
    )
    return result, path


def _gt(path: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(path / "ground_truth" / f"{name}.parquet")


def test_determinism(tmp_path: Path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first = generate(
        seed=42,
        n_applicants=SMALL_APPLICANTS,
        n_applications=SMALL_APPLICATIONS,
        output_dir=first_dir,
    )
    second = generate(
        seed=42,
        n_applicants=SMALL_APPLICANTS,
        n_applications=SMALL_APPLICATIONS,
        output_dir=second_dir,
    )
    assert first.n_applicants == second.n_applicants
    assert first.n_applications == second.n_applications
    assert first.n_loans == second.n_loans
    assert first.n_installments == second.n_installments
    assert first.n_defaults == second.n_defaults
    assert first.default_rate == second.default_rate
    assert first.issue_counts == second.issue_counts
    assert first.batch_counts == second.batch_counts

    for name in (
        "applicants",
        "applications",
        "loans",
        "installments",
        "counterfactual_outcomes",
    ):
        left = _gt(first_dir, name)
        right = _gt(second_dir, name)
        pd.testing.assert_frame_equal(left, right)

    first_issues = (first_dir / "raw" / "_injected_issues.csv").read_bytes()
    second_issues = (second_dir / "raw" / "_injected_issues.csv").read_bytes()
    assert first_issues == second_issues

    for batch in ("batch_001", "batch_002", "batch_003"):
        for entity in ("applicants", "applications", "loans", "installments"):
            rel = Path("raw") / batch / f"{entity}.csv"
            assert (first_dir / rel).read_bytes() == (second_dir / rel).read_bytes()


def test_row_counts_match_request_and_summary(medium: tuple[object, Path]) -> None:
    result, path = medium
    applicants = _gt(path, "applicants")
    applications = _gt(path, "applications")
    loans = _gt(path, "loans")
    installments = _gt(path, "installments")

    assert len(applicants) == MEDIUM_APPLICANTS == result.n_applicants
    assert len(applications) == MEDIUM_APPLICATIONS == result.n_applications
    assert len(loans) == result.n_loans > 0
    assert len(installments) == result.n_installments > 0

    term_by_loan = loans.set_index("loan_id")["term_months"]
    counts = installments.groupby("loan_id").size()
    assert counts.index.difference(term_by_loan.index).empty
    assert (counts == term_by_loan.loc[counts.index]).all()
    assert installments.groupby(["loan_id", "installment_number"]).size().max() == 1


def test_ground_truth_referential_integrity(medium: tuple[object, Path]) -> None:
    _result, path = medium
    applicants = _gt(path, "applicants")
    applications = _gt(path, "applications")
    loans = _gt(path, "loans")
    installments = _gt(path, "installments")

    assert applicants["applicant_id"].is_unique
    assert applications["application_id"].is_unique
    assert loans["loan_id"].is_unique
    assert installments["installment_id"].is_unique
    assert applicants["cpf"].is_unique

    assert applications["applicant_id"].isin(applicants["applicant_id"]).all()
    assert loans["application_id"].isin(applications["application_id"]).all()
    assert installments["loan_id"].isin(loans["loan_id"]).all()
    assert loans.groupby("application_id").size().max() == 1

    approved = set(applications.loc[applications["decision"] == "approved", "application_id"])
    assert set(loans["application_id"]).issubset(approved)

    assert not applicants["cpf"].map(is_valid_cpf).any()
    assert applicants["cpf"].str.fullmatch(r"\d{11}").all()
    assert (applicants["monthly_income"] >= 0).all()

    age_years = (
        pd.Timestamp("2025-01-01", tz="UTC") - pd.to_datetime(applicants["birth_date"], utc=True)
    ).dt.days / 365.25
    assert age_years.min() >= 21
    assert age_years.max() <= 70


def test_issues_log_nonempty_for_every_type(medium: tuple[object, Path]) -> None:
    _result, path = medium
    log = pd.read_csv(path / "raw" / "_injected_issues.csv")
    assert list(log.columns) == ["batch", "file", "row_key", "issue_type", "column"]
    present = set(log["issue_type"].unique())
    missing = set(ISSUE_TYPES) - present
    assert not missing, f"Missing issue types: {sorted(missing)}"
    counts = log["issue_type"].value_counts()
    for issue_type in ISSUE_TYPES:
        assert counts[issue_type] >= 1


def test_default_rate_in_plausible_range(medium: tuple[object, Path]) -> None:
    result, path = medium
    loans = _gt(path, "loans")
    complete = loans.loc[loans["observation_complete"]]
    assert len(complete) == result.n_loans_observation_complete
    assert len(complete) > 0
    realized = float(complete["is_default_90dpd_12m"].mean())
    assert realized == pytest.approx(result.default_rate)
    assert 0.08 <= realized <= 0.15


def test_raw_batches_exist_and_schema_change_is_visible(medium: tuple[object, Path]) -> None:
    _result, path = medium
    raw = path / "raw"
    for batch in ("batch_001", "batch_002", "batch_003"):
        for entity in ("applicants", "applications", "loans", "installments"):
            assert (raw / batch / f"{entity}.csv").is_file()

    header_001 = (raw / "batch_001" / "applicants.csv").read_text(encoding="utf-8").splitlines()[0]
    header_002 = (
        (raw / "batch_002" / "applicants.csv").read_text(encoding="iso-8859-1").splitlines()[0]
    )
    header_003 = (raw / "batch_003" / "applicants.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "monthly_income" in header_001.split(",")
    assert ";" in header_002
    assert "income_monthly" in header_002
    assert "occupation_note" in header_002
    assert "monthly_income" in header_003.split(",")
    assert "customer_segment" in header_003.split(",")


def _recompute_default_flag(loans: pd.DataFrame, installments: pd.DataFrame) -> pd.Series:
    """Independent implementation of the 12-month default rule for testing."""
    as_of = pd.Timestamp(AS_OF).tz_convert("UTC").normalize()
    flags = pd.Series(False, index=loans.index)
    inst = installments.merge(loans[["loan_id", "disbursed_at"]], on="loan_id", how="left")
    for idx, loan in loans.iterrows():
        window_end = window_end_date(loan["disbursed_at"])
        rows = inst.loc[inst["loan_id"] == loan["loan_id"]]
        defaulted = False
        for _, inst_row in rows.iterrows():
            due = pd.Timestamp(inst_row["due_date"])
            if due.tzinfo is None:
                due = due.tz_localize("UTC")
            due = due.tz_convert("UTC").normalize()
            if due > window_end:
                continue
            paid = pd.notna(inst_row["paid_at"]) and float(inst_row["amount_paid"]) >= float(
                inst_row["amount_due"]
            )
            if paid:
                paid_at = pd.Timestamp(inst_row["paid_at"])
                if paid_at.tzinfo is None:
                    paid_at = paid_at.tz_localize("UTC")
                dpd = int((paid_at.tz_convert("UTC").normalize() - due).days)
            elif due <= as_of:
                dpd = int((as_of - due).days)
            else:
                dpd = 0
            if dpd >= 90:
                defaulted = True
                break
        flags.loc[idx] = defaulted
    return flags


def test_observation_complete_uses_12m_plus_90d(medium: tuple[object, Path]) -> None:
    result, path = medium
    loans = _gt(path, "loans")
    as_of = pd.Timestamp(AS_OF).tz_convert("UTC").normalize()
    computed = loans["disbursed_at"].map(is_observation_complete)
    assert computed.astype(bool).to_numpy().tolist() == loans["observation_complete"].tolist()
    observable = loans.loc[loans["observation_complete"]]
    immature = loans.loc[~loans["observation_complete"]]
    assert len(immature) == result.n_loans_immature
    assert len(observable) == result.n_loans_observation_complete
    assert (observable["disbursed_at"].map(observable_from_date) <= as_of).all()
    if len(immature):
        assert (immature["disbursed_at"].map(observable_from_date) > as_of).all()
    # No observable loan may violate the contract rule.
    for _, row in observable.iterrows():
        assert is_observation_complete(row["disbursed_at"])
        assert observable_from_date(row["disbursed_at"]) <= as_of


def test_default_label_matches_installment_rule(medium: tuple[object, Path]) -> None:
    _result, path = medium
    loans = _gt(path, "loans")
    installments = _gt(path, "installments")
    observable = loans.loc[loans["observation_complete"]].copy()
    recomputed = _recompute_default_flag(observable, installments)
    assert recomputed.to_numpy().tolist() == observable["is_default_90dpd_12m"].tolist()


def test_some_defaults_start_in_months_9_to_12(medium: tuple[object, Path]) -> None:
    _result, path = medium
    loans = _gt(path, "loans")
    defaults = loans.loc[loans["observation_complete"] & loans["is_default_90dpd_12m"]]
    assert len(defaults) > 0
    assert (defaults["first_miss_installment"].between(1, 12)).all()
    assert (defaults["first_miss_installment"].between(9, 12)).any(), (
        "expected at least one observable default whose first miss is in months 9-12"
    )


def test_counterfactual_outcomes_cover_non_originated_only(medium: tuple[object, Path]) -> None:
    result, path = medium
    applications = _gt(path, "applications")
    loans = _gt(path, "loans")
    counterfactuals = _gt(path, "counterfactual_outcomes")
    expected_ids = set(applications["application_id"]) - set(loans["application_id"])
    assert set(counterfactuals["application_id"]) == expected_ids
    assert len(counterfactuals) == result.n_counterfactuals == len(expected_ids)
    assert counterfactuals["application_id"].is_unique
    assert set(counterfactuals["decision"]).issubset({"approved", "declined", "cancelled"})
    required = {
        "application_id",
        "decision",
        "hypothetical_principal",
        "hypothetical_interest_rate_monthly",
        "hypothetical_term_months",
        "is_default_90dpd_12m_counterfactual",
    }
    assert required.issubset(counterfactuals.columns)
    # Counterfactuals must not leak into the messy client exports.
    for batch in ("batch_001", "batch_002", "batch_003"):
        for entity in ("applicants", "applications", "loans", "installments"):
            text = (path / "raw" / batch / f"{entity}.csv").read_text(encoding="latin-1")
            assert "is_default_90dpd_12m_counterfactual" not in text
            assert "hypothetical_principal" not in text
    if len(counterfactuals):
        rate = float(counterfactuals["is_default_90dpd_12m_counterfactual"].mean())
        assert rate == pytest.approx(result.counterfactual_default_rate)
        assert 0.0 <= rate <= 1.0
