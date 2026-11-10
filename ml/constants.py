"""Shared constants for the M3 models. All figures are synthetic."""

from __future__ import annotations

from pandas import Timestamp

AS_OF = Timestamp("2026-10-31", tz="UTC")
LAST_OBSERVABLE_DISBURSEMENT = Timestamp("2025-08-02", tz="UTC")

# Time split on applied_at (UTC). OOT is never used for fit, tune, or calibration.
TRAIN_END = Timestamp("2024-07-01", tz="UTC")
TUNE_END = Timestamp("2024-10-01", tz="UTC")  # start of the calibration slice
CALIB_END = Timestamp("2025-01-01", tz="UTC")  # start of OOT

MODEL_LOGREG = "baseline_logreg_v1"
MODEL_GBM = "gbm_v1"
MODEL_VERSIONS = (MODEL_LOGREG, MODEL_GBM)

# Points-to-double-odds score: higher is better (lower PD).
PDO = 20
ANCHOR_SCORE = 600
ANCHOR_ODDS = 50.0  # good:bad at the anchor; PD = 1 / (1 + 50) ≈ 0.019608
SCORE_MIN = 300
SCORE_MAX = 900

# Default risk-band edges (overwritten at train time from GBM PD quintiles
# so that each band holds at least about 5% of OOT approved loans).
BAND_EDGES = (
    ("A", 0.00, 0.03),
    ("B", 0.03, 0.06),
    ("C", 0.06, 0.10),
    ("D", 0.10, 0.18),
    ("E", 0.18, 1.01),
)
BAND_MIN_SHARE = 0.05

# Logistic regression drops these two ratios: they are near-linear functions
# of monthly_income, requested_amount and existing_debt, and they flipped the
# sign of monthly_income. The GBM keeps them (with monotone constraints).
LOGREG_DROPPED_FEATURES = (
    "requested_to_income",
    "debt_to_annual_income",
)

# Minimum |x - reference| required before a GBM reason code is emitted,
# in addition to SHAP > 0 and sitting on the risky side of the reference.
REASON_MIN_DELTA = {
    "bureau_score": 10.0,  # points
    "monthly_income": 200.0,  # BRL / month
    "employment_months": 3.0,
    "existing_debt": 500.0,  # BRL
    "requested_to_income": 0.5,  # months of income
    "debt_to_annual_income": 0.02,
    "estimated_dti": 0.02,
    "dependents": 1.0,
    "prior_decline_count": 1.0,
    "prior_application_count": 1.0,
    "age_at_applied_at": 5.0,  # years
    "requested_amount": 500.0,  # BRL
    "term_months": 6.0,
    "existing_debt_is_missing": 1.0,
}

BUREAU_TIEBREAK_SEEDS = 20

# Rolling history window for prior_application_count and prior_decline_count.
# prior_observable_default_count is dropped: a 12-month lookback cannot
# contain a completed 12m+90d outcome, and the lifetime count drifts
# sharply between train and OOT.
HISTORY_LOOKBACK_MONTHS = 12
DROPPED_HISTORY_FEATURES = ("prior_observable_default_count",)

# Synthetic application-time DTI assumptions — not the contract rate.
REF_MONTHLY_RATE = 0.02
EXISTING_DEBT_SERVICE_RATE = 0.04

SEED = 42
BOOTSTRAP_DRAWS = 1000

# Features whose expected effect on PD is clear, with LightGBM monotone signs.
# -1 = higher value must not increase PD; +1 = higher value must not decrease PD.
MONOTONE = {
    "bureau_score": -1,
    "monthly_income": -1,
    "employment_months": -1,
    "existing_debt": 1,
    "requested_to_income": 1,
    "debt_to_annual_income": 1,
    "estimated_dti": 1,
    "dependents": 1,
    "prior_decline_count": 1,
}

EXPECTED_DIRECTION = {
    **{name: ("down" if sign < 0 else "up") for name, sign in MONOTONE.items()},
    "age_at_applied_at": "u-shaped",
    "requested_amount": "mixed",
    "term_months": "mixed",
    "prior_application_count": "mixed",
    "existing_debt_is_missing": "mixed",
}

NUMERIC_FEATURES = [
    "requested_amount",
    "term_months",
    "bureau_score",
    "existing_debt",
    "existing_debt_is_missing",
    "monthly_income",
    "employment_months",
    "dependents",
    "age_at_applied_at",
    "requested_to_income",
    "debt_to_annual_income",
    "estimated_dti",
    "prior_application_count",
    "prior_decline_count",
]

LOGREG_NUMERIC_FEATURES = [
    c for c in NUMERIC_FEATURES if c not in LOGREG_DROPPED_FEATURES
]

CATEGORICAL_FEATURES = [
    "purpose",
    "channel",
    "employment_type",
    "housing_status",
    "state_uf",
]

ALLOWED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN_FEATURES = frozenset(
    {
        "principal",
        "interest_rate_monthly",
        "installment_amount",
        "decision",
        "decided_at",
        "disbursed_at",
        "paid_at",
        "amount_paid",
        "days_past_due",
        "is_default_12m",
        "is_default_90dpd_12m",
        "outcome_status",
        "default_date",
        "loan_id",
    }
)

LABEL_COL = "is_default_12m"
FEATURE_LABELS = {
    "bureau_score": "Bureau score",
    "monthly_income": "Monthly income",
    "employment_months": "Employment tenure",
    "existing_debt": "Existing debt balance",
    "existing_debt_is_missing": "Missing existing-debt flag",
    "requested_to_income": "Requested amount to monthly income",
    "debt_to_annual_income": "Existing debt to annual income",
    "estimated_dti": "Estimated debt-to-income",
    "dependents": "Number of dependents",
    "prior_decline_count": "Prior declines",
    "prior_observable_default_count": "Prior observable defaults",
    "prior_application_count": "Prior applications",
    "age_at_applied_at": "Age at application",
    "requested_amount": "Requested amount",
    "term_months": "Requested term",
    "purpose": "Loan purpose",
    "channel": "Application channel",
    "employment_type": "Employment type",
    "housing_status": "Housing status",
    "state_uf": "State of residence",
}

EMPIRICAL_APPROVAL_RATES = (0.50, 0.60, 0.656, 0.70)
