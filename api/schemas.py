"""Pydantic v2 request and response models. Application-time inputs only."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Purpose(str, Enum):
    personal = "personal"
    debt_refinance = "debt_refinance"
    home_improvement = "home_improvement"
    education = "education"
    medical = "medical"
    other = "other"


class Channel(str, Enum):
    app = "app"
    web = "web"
    branch = "branch"
    partner = "partner"


class EmploymentType(str, Enum):
    clt = "clt"
    self_employed = "self_employed"
    informal = "informal"
    public_servant = "public_servant"
    unemployed = "unemployed"


class HousingStatus(str, Enum):
    renter = "renter"
    owner = "owner"
    financed = "financed"
    family = "family"


BRAZILIAN_UF = Literal[
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
]


class ScoreRequest(BaseModel):
    """Application-time attributes known at `applied_at`.

    History features (`prior_application_count`, `prior_decline_count`) are
    optional and default to 0. The API does not look up the warehouse; the
    caller must supply the 12-month lookback counts when they are known.
    `prior_observable_default_count` is not an input (it was dropped from
    the models).
    """

    applied_at: datetime
    birth_date: datetime
    requested_amount: float = Field(..., gt=0, le=500_000)
    term_months: int = Field(..., ge=1, le=60)
    purpose: Purpose
    channel: Channel
    bureau_score: int = Field(..., ge=300, le=900)
    existing_debt: float | None = Field(default=None, ge=0, le=5_000_000)
    monthly_income: float = Field(..., gt=0, le=1_000_000)
    employment_type: EmploymentType
    employment_months: int = Field(..., ge=0, le=600)
    housing_status: HousingStatus | None = None
    dependents: int = Field(..., ge=0, le=20)
    state_uf: BRAZILIAN_UF
    prior_application_count: int = Field(default=0, ge=0, le=100)
    prior_decline_count: int = Field(default=0, ge=0, le=100)
    model_version: Literal["gbm_v1", "baseline_logreg_v1"] | None = None

    @field_validator("term_months")
    @classmethod
    def term_is_supported(cls, value: int) -> int:
        # The generator uses 6, 9, 12, 18, 24, 36; other terms in 1–60 are
        # accepted so the API can score slightly different request shapes.
        return value

    @model_validator(mode="after")
    def decline_cannot_exceed_applications(self) -> "ScoreRequest":
        if self.prior_decline_count > self.prior_application_count:
            raise ValueError("prior_decline_count cannot exceed prior_application_count")
        return self


class ReasonCode(BaseModel):
    code: str
    text: str


class ScoreResponse(BaseModel):
    pd: float
    score: int
    risk_band: str
    reason_codes: list[ReasonCode]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool


class ModelInfoResponse(BaseModel):
    default_model: str
    available_models: list[str]
    training_window: dict
    oot_metrics: dict
    history_features: str
    is_synthetic: bool = True
    client: str = "Credara Fintech (fictional)"
