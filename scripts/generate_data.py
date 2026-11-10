"""Synthetic data generator for the Credara Fintech (fictional) case study.

Credara Fintech is a fictional Brazilian personal-credit lender and does not
exist. Every applicant, contract, identifier and amount produced here is
synthetic. The generator does not use data, code, or names from any real
employer or lender.

Generative process
------------------
1. Draw applicants with a *latent risk* factor. Latent risk increases when
   income is lower, employment is more informal or unemployed, the applicant
   is very young or older, housing is not ownership, dependents are high
   relative to income, and a small synthetic state effect plus noise.
2. Draw applications. Bureau score is a noisy inverse function of latent
   risk. Requested amount, term, purpose, channel and existing debt are
   drawn conditionally on income and risk. A synthetic underwriting rule
   approves, declines or cancels the application.
3. Originate a loan for most approved applications. The monthly installment
   is the standard fixed-rate payment (PMT) on principal, monthly rate and
   term. Disbursement dates run through 2025-10-31, so some loans are still
   *immature* as of the global cutoff (2026-10-31 UTC).
4. Simulate installment payment paths. A loan that is intended to go
   delinquent draws a first missed installment over months 1..min(12, term)
   with extra weight on months 3-6, so the timing has a tail into months
   9-12. The 12-month default label is then *computed from installments*:
   any installment due on or before disbursed_at + 12 months that reaches
   90+ DPD is a default (the 90th day may fall after the 12-month
   anniversary, which is why a loan is observable only when
   disbursed_at + 12 months + 90 days <= 2026-10-31 UTC). A delinquency
   that starts on an installment due after that 12-month window is not a
   12-month default. The realised default rate among observable loans is
   calibrated to an 8-15% range and is synthetic.
5. Declined, cancelled, and approved-but-not-originated applications get a
   *counterfactual* outcome in data/ground_truth/ only, using the same
   default model, for later selection-bias analysis. These rows never
   appear in the raw client exports.

CPF-like identifiers
--------------------
Values that look like Brazilian CPFs are 11-digit strings whose two check
digits are computed and then deliberately shifted so they are *not* valid
CPFs. They must never be treated as real personal identifiers.

Outputs
-------
- data/ground_truth/*.parquet and generation_summary.json
- data/raw/batch_00N/{applicants,applications,loans,installments}.csv
- data/raw/_injected_issues.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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

DEFAULT_SEED = 42
DEFAULT_N_APPLICANTS = 20_000
DEFAULT_N_APPLICATIONS = 50_000
TARGET_DEFAULT_RATE = 0.11

AS_OF = pd.Timestamp("2026-10-31", tz="UTC")
APP_START = pd.Timestamp("2023-01-01", tz="UTC")
APP_END = pd.Timestamp("2025-10-15", tz="UTC")
LATEST_DISBURSEMENT = pd.Timestamp("2025-10-31", tz="UTC")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

BATCHES: list[dict] = [
    {
        "batch_id": "batch_001",
        "cutoff": pd.Timestamp("2024-06-30", tz="UTC"),
        "extracted_at": pd.Timestamp("2024-07-02T15:00:00", tz="UTC"),
        "delimiter": ",",
        "encoding": "utf-8",
    },
    {
        "batch_id": "batch_002",
        "cutoff": pd.Timestamp("2025-03-31", tz="UTC"),
        "extracted_at": pd.Timestamp("2025-04-02T15:00:00", tz="UTC"),
        "delimiter": ";",
        "encoding": "iso-8859-1",
    },
    {
        "batch_id": "batch_003",
        "cutoff": pd.Timestamp("2026-10-31", tz="UTC"),
        "extracted_at": pd.Timestamp("2026-11-02T15:00:00", tz="UTC"),
        "delimiter": ",",
        "encoding": "utf-8",
    },
]
BATCH_BY_ID = {b["batch_id"]: b for b in BATCHES}

EMPLOYMENT_TYPES = ("clt", "self_employed", "public_servant", "informal", "unemployed")
HOUSING_STATUSES = ("owner", "renter", "family", "financed")
PURPOSES = ("personal", "debt_refinance", "home_improvement", "education", "medical", "other")
CHANNELS = ("app", "web", "branch", "partner")
DECISIONS = ("approved", "declined", "cancelled")
TERM_CHOICES = np.array([6, 9, 12, 18, 24, 36])

STATE_UFS = (
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
)
# Synthetic sampling weights loosely inspired by population share, not official stats.
STATE_WEIGHTS = {
    "SP": 0.22,
    "MG": 0.10,
    "RJ": 0.08,
    "BA": 0.07,
    "PR": 0.06,
    "RS": 0.06,
    "PE": 0.05,
    "CE": 0.04,
    "PA": 0.04,
    "SC": 0.04,
    "GO": 0.03,
    "MA": 0.03,
    "ES": 0.02,
    "PB": 0.02,
    "AM": 0.02,
    "MT": 0.02,
    "MS": 0.015,
    "RN": 0.015,
    "AL": 0.015,
    "PI": 0.015,
    "DF": 0.015,
    "SE": 0.01,
    "RO": 0.01,
    "TO": 0.008,
    "AC": 0.005,
    "AP": 0.004,
    "RR": 0.003,
}
# Small synthetic risk offsets used only to give the later model a regional signal.
STATE_RISK = {
    "SP": -0.05,
    "DF": -0.08,
    "SC": -0.06,
    "RS": -0.04,
    "PR": -0.03,
    "RJ": 0.04,
    "BA": 0.06,
    "PE": 0.05,
    "CE": 0.05,
    "MA": 0.08,
    "PA": 0.07,
    "AL": 0.07,
    "PI": 0.06,
    "AM": 0.05,
}

EMP_RISK = {
    "clt": -0.15,
    "self_employed": 0.20,
    "public_servant": -0.35,
    "informal": 0.45,
    "unemployed": 0.80,
}
EMP_INCOME_MULT = {
    "clt": 1.00,
    "self_employed": 0.95,
    "public_servant": 1.15,
    "informal": 0.70,
    "unemployed": 0.45,
}
HOUSING_RISK = {"owner": -0.15, "renter": 0.05, "family": 0.10, "financed": 0.08}

FIRST_NAMES = (
    "Ana",
    "Aline",
    "Amanda",
    "Andre",
    "André",
    "Beatriz",
    "Bruno",
    "Caio",
    "Camila",
    "Carla",
    "Carolina",
    "Claudia",
    "Cláudia",
    "Daniel",
    "Diego",
    "Eduarda",
    "Eduardo",
    "Eliane",
    "Fabiana",
    "Felipe",
    "Fernanda",
    "Fatima",
    "Fátima",
    "Gabriel",
    "Gabriela",
    "Guilherme",
    "Helena",
    "Henrique",
    "Igor",
    "Isabel",
    "Jessica",
    "Joao",
    "João",
    "Jose",
    "José",
    "Julia",
    "Júlia",
    "Larissa",
    "Leonardo",
    "Leticia",
    "Letícia",
    "Lucas",
    "Lucia",
    "Lúcia",
    "Luciana",
    "Luis",
    "Luís",
    "Manuela",
    "Marcelo",
    "Marcio",
    "Márcio",
    "Marcos",
    "Maria",
    "Mariana",
    "Marina",
    "Mateus",
    "Natalia",
    "Natália",
    "Patricia",
    "Patrícia",
    "Paulo",
    "Pedro",
    "Rafael",
    "Renata",
    "Ricardo",
    "Roberto",
    "Rodrigo",
    "Sandra",
    "Sergio",
    "Sérgio",
    "Sofia",
    "Thiago",
    "Vanessa",
    "Victor",
    "Vitor",
    "Viviane",
)
LAST_NAMES = (
    "Almeida",
    "Alves",
    "Araujo",
    "Araújo",
    "Barbosa",
    "Batista",
    "Borges",
    "Campos",
    "Cardoso",
    "Carvalho",
    "Castro",
    "Costa",
    "Dias",
    "Duarte",
    "Fernandes",
    "Ferreira",
    "Garcia",
    "Gomes",
    "Lima",
    "Lopes",
    "Machado",
    "Martins",
    "Mendes",
    "Moreira",
    "Nascimento",
    "Nunes",
    "Oliveira",
    "Pereira",
    "Pinto",
    "Ramos",
    "Reis",
    "Ribeiro",
    "Rocha",
    "Rodrigues",
    "Santos",
    "Silva",
    "Souza",
    "Teixeira",
    "Vieira",
)

CATEGORY_VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    "employment_type": {
        "clt": ("CLT", "Clt", "clt", "carteira assinada", "celetista"),
        "self_employed": ("autonomo", "autônomo", "Autônomo", "MEI", "self-employed"),
        "public_servant": ("servidor", "servidor público", "Servidor Publico", "public servant"),
        "informal": ("Informal", "informal", "sem carteira", "bico"),
        "unemployed": ("desempregado", "Desempregado", "unemployed", "sem vinculo"),
    },
    "housing_status": {
        "owner": ("owner", "Owner", "proprio", "próprio", "casa propria"),
        "renter": ("renter", "alugado", "Alugado", "aluguel", "rent"),
        "family": ("family", "familia", "família", "com os pais", "parentes"),
        "financed": ("financed", "financiado", "Financiado", "financiamento"),
    },
    "purpose": {
        "personal": ("personal", "pessoal", "Pessoal", "livre"),
        "debt_refinance": (
            "debt_refinance",
            "refinanciamento",
            "troca de divida",
            "debt refinance",
        ),
        "home_improvement": ("home_improvement", "reforma", "Reforma", "casa"),
        "education": ("education", "educacao", "educação", "curso"),
        "medical": ("medical", "saude", "saúde", "hospital"),
        "other": ("other", "outros", "Outros", "diversos"),
    },
    "channel": {
        "app": ("app", "App", "aplicativo", "mobile"),
        "web": ("web", "Web", "site", "internet"),
        "branch": ("branch", "agencia", "agência", "loja"),
        "partner": ("partner", "parceiro", "Parceiro", "correspondente"),
    },
    "decision": {
        "approved": ("approved", "Approved", "aprovado", "Aprovado"),
        "declined": ("declined", "Declined", "reprovado", "negado"),
        "cancelled": ("cancelled", "canceled", "cancelado", "Cancelado"),
    },
}

UF_NAME = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapa",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceara",
    "DF": "Distrito Federal",
    "ES": "Espirito Santo",
    "GO": "Goias",
    "MA": "Maranhao",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Para",
    "PB": "Paraiba",
    "PR": "Parana",
    "PE": "Pernambuco",
    "PI": "Piaui",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondonia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "Sao Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}

DISGUISED_NULLS = ("N/A", "-", "", "NULL", "none")


@dataclass
class GenerationResult:
    seed: int
    n_applicants_requested: int
    n_applications_requested: int
    n_applicants: int
    n_applications: int
    n_loans: int
    n_installments: int
    n_loans_observation_complete: int
    n_loans_immature: int
    n_defaults: int
    default_rate: float
    n_counterfactuals: int
    counterfactual_default_rate: float
    issue_counts: dict[str, int]
    batch_counts: dict[str, dict[str, int]]
    output_dir: str


def _cpf_check_digits(body9: str) -> str:
    def one(digits: str, start: int) -> str:
        total = sum(int(d) * weight for d, weight in zip(digits, range(start, 1, -1)))
        rem = total % 11
        return "0" if rem < 2 else str(11 - rem)

    first = one(body9, 10)
    second = one(body9 + first, 11)
    return first + second


def is_valid_cpf(cpf11: str) -> bool:
    digits = "".join(ch for ch in str(cpf11) if ch.isdigit())
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    return digits[-2:] == _cpf_check_digits(digits[:9])


def generate_fake_cpf(rng: np.random.Generator) -> str:
    """Return an 11-digit string that is guaranteed not to be a valid CPF."""
    while True:
        body = "".join(str(int(rng.integers(0, 10))) for _ in range(9))
        if len(set(body)) == 1:
            continue
        genuine = _cpf_check_digits(body)
        shifted = "".join(str((int(ch) + 1) % 10) for ch in genuine)
        cpf = body + shifted
        if not is_valid_cpf(cpf):
            return cpf


def generate_unique_cpfs(rng: np.random.Generator, n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        cpf = generate_fake_cpf(rng)
        if cpf not in seen:
            seen.add(cpf)
            out.append(cpf)
    return out


def payment_amount_vec(
    principal: np.ndarray, monthly_rate: np.ndarray, term_months: np.ndarray
) -> np.ndarray:
    principal = np.asarray(principal, dtype=float)
    monthly_rate = np.asarray(monthly_rate, dtype=float)
    term_months = np.asarray(term_months, dtype=float)
    out = np.empty(principal.shape, dtype=float)
    zero = monthly_rate <= 0
    out[zero] = principal[zero] / term_months[zero]
    rate = monthly_rate[~zero]
    term = term_months[~zero]
    principal_nz = principal[~zero]
    factor = (1.0 + rate) ** term
    out[~zero] = principal_nz * rate * factor / (factor - 1.0)
    return np.round(out, 2)


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=int(months))


def as_utc_day(ts) -> pd.Timestamp:
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.normalize()


def window_end_date(disbursed_at) -> pd.Timestamp:
    """Last due date included in the 12-month default window."""
    return as_utc_day(disbursed_at) + pd.DateOffset(months=12)


def observable_from_date(disbursed_at) -> pd.Timestamp:
    """Earliest as-of date at which the 12-month default label is final."""
    return window_end_date(disbursed_at) + pd.Timedelta(days=90)


def is_observation_complete(disbursed_at, as_of: pd.Timestamp = AS_OF) -> bool:
    return observable_from_date(disbursed_at) <= as_utc_day(as_of)


# First missed installment: months 1..12, extra weight on 3..6 so the
# default timing has a tail into months 9-12 rather than stopping at 8.
FIRST_MISS_MONTHS = np.arange(1, 13)
FIRST_MISS_WEIGHTS = np.array(
    [1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float
)


def sample_first_miss(rng: np.random.Generator, terms: np.ndarray) -> np.ndarray:
    """Draw first missed installment in 1..min(12, term) with extra weight on 3-6."""
    terms = np.asarray(terms, dtype=int)
    out = np.zeros(len(terms), dtype=int)
    for term in np.unique(terms):
        cap = max(1, min(12, int(term)))
        months = FIRST_MISS_MONTHS[:cap]
        weights = FIRST_MISS_WEIGHTS[:cap]
        weights = weights / weights.sum()
        mask = terms == term
        out[mask] = rng.choice(months, size=int(mask.sum()), p=weights)
    return out


def default_model_logit(
    latent: np.ndarray,
    requested: np.ndarray,
    income: np.ndarray,
    existing: np.ndarray,
    bureau: np.ndarray,
) -> np.ndarray:
    """Shared credit-default logit used for originated loans and counterfactuals."""
    income = np.clip(np.asarray(income, dtype=float), 1.0, None)
    requested = np.asarray(requested, dtype=float)
    existing = np.asarray(existing, dtype=float)
    latent = np.asarray(latent, dtype=float)
    bureau = np.asarray(bureau, dtype=float)
    return (
        -2.2
        + 0.90 * latent
        + 0.30 * ((requested / income) - 3.0) / 2.0
        + 0.20 * (existing / (income * 12.0))
        - 0.35 * ((bureau - 650.0) / 100.0)
    )


def hypothetical_rate(latent: np.ndarray, bureau: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return np.round(
        np.clip(0.025 + 0.010 * latent - 0.00002 * (bureau - 600) + noise, 0.015, 0.075),
        4,
    )


def installment_reaches_90dpd(
    due_date, paid_at, amount_paid, amount_due, as_of: pd.Timestamp = AS_OF
) -> bool:
    due = as_utc_day(due_date)
    cutoff = as_utc_day(as_of)
    paid_flag = pd.notna(paid_at) and float(amount_paid) >= float(amount_due)
    if paid_flag:
        return int((as_utc_day(paid_at) - due).days) >= 90
    if due <= cutoff:
        return int((cutoff - due).days) >= 90
    return False


def label_defaults_from_installments(
    loans: pd.DataFrame, installments: pd.DataFrame, as_of: pd.Timestamp = AS_OF
) -> pd.DataFrame:
    """Set observation_complete, is_default_90dpd_12m and default_date from installments.

    A loan is in 12-month default if any installment *due on or before*
    disbursed_at + 12 months reaches 90+ DPD (the 90th day may fall after
    the 12-month anniversary). Delinquency on an installment due after
    that window is not a 12-month default.
    """
    out = loans.copy()
    disbursed = pd.to_datetime(out["disbursed_at"], utc=True)
    out["observation_complete"] = disbursed.map(
        lambda ts: is_observation_complete(ts, as_of)
    ).astype(bool)

    inst = installments.merge(out[["loan_id", "disbursed_at"]], on="loan_id", how="left")
    inst["due_day"] = pd.to_datetime(inst["due_date"], utc=True).map(as_utc_day)
    inst["window_end"] = pd.to_datetime(inst["disbursed_at"], utc=True).map(window_end_date)
    in_window = inst["due_day"] <= inst["window_end"]
    paid = inst["paid_at"].notna() & (
        inst["amount_paid"].astype(float) >= inst["amount_due"].astype(float)
    )
    paid_day = pd.to_datetime(inst["paid_at"], utc=True)
    cutoff = as_utc_day(as_of)
    dpd = np.zeros(len(inst), dtype=int)
    paid_idx = paid.to_numpy()
    if paid_idx.any():
        dpd[paid_idx] = np.maximum(
            0, (paid_day[paid].map(as_utc_day) - inst.loc[paid, "due_day"]).dt.days.to_numpy()
        )
    unpaid = ~paid
    due_by_cutoff = unpaid & (inst["due_day"] <= cutoff)
    if due_by_cutoff.any():
        dpd[due_by_cutoff.to_numpy()] = (
            cutoff - inst.loc[due_by_cutoff, "due_day"]
        ).dt.days.to_numpy()
    inst["reached_90"] = in_window & (dpd >= 90)
    events = inst.loc[inst["reached_90"]].copy()
    if events.empty:
        out["is_default_90dpd_12m"] = False
        out["default_date"] = pd.NaT
        return out

    events = events.sort_values(["loan_id", "installment_number"])
    first = events.groupby("loan_id", as_index=True).first()
    default_date = first["due_day"] + pd.Timedelta(days=90)
    out["is_default_90dpd_12m"] = out["loan_id"].isin(first.index)
    out["default_date"] = out["loan_id"].map(default_date)
    return out


def _shift_logit_to_mean(logit: np.ndarray, target: float) -> float:
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        mean = (1.0 / (1.0 + np.exp(-(logit + mid)))).mean()
        if mean > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _choose_indices(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    k = min(max(int(k), 0), int(n))
    if k == 0:
        return np.array([], dtype=int)
    return np.sort(rng.choice(n, size=k, replace=False))


def _n_issues(n_rows: int, rate: float, minimum: int = 3) -> int:
    if n_rows <= 0:
        return 0
    return min(n_rows, max(minimum, int(round(n_rows * rate))))


def _log_issue(
    issues: list[dict],
    batch: str,
    file_name: str,
    row_key: str,
    issue_type: str,
    column: str,
) -> None:
    issues.append(
        {
            "batch": batch,
            "file": file_name,
            "row_key": str(row_key),
            "issue_type": issue_type,
            "column": column,
        }
    )


def _state_variants(uf: str) -> tuple[str, ...]:
    name = UF_NAME[uf]
    return (uf, uf.lower(), name, name.lower(), name.upper())


def _format_cpf(cpf11: str, kind: str) -> str:
    if kind == "digits":
        return cpf11
    if kind == "punct":
        return f"{cpf11[:3]}.{cpf11[3:6]}.{cpf11[6:9]}-{cpf11[9:]}"
    if kind == "spaces":
        return f"{cpf11[:3]} {cpf11[3:6]} {cpf11[6:9]} {cpf11[9:]}"
    if kind == "slash":
        return f"{cpf11[:3]}.{cpf11[3:6]}.{cpf11[6:9]}/{cpf11[9:]}"
    return f"{cpf11[:3]}.{cpf11[3:6]}{cpf11[6:9]}-{cpf11[9:]}"


def _format_date(value: pd.Timestamp, kind: str) -> str:
    if pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if kind == "iso":
        return ts.strftime("%Y-%m-%d")
    if kind == "dmy_slash":
        return ts.strftime("%d/%m/%Y")
    if kind == "dmy_dash":
        return ts.strftime("%d-%m-%Y")
    if kind == "ymd_slash":
        return ts.strftime("%Y/%m/%d")
    if kind == "dmy_dot":
        return ts.strftime("%d.%m.%Y")
    if kind == "iso_z":
        return (
            ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            if ts.tzinfo
            else ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _format_money(value: float, kind: str) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    sign = "-" if number < 0 else ""
    absolute = abs(number)
    if kind == "numeric":
        return f"{number:.2f}"
    if kind == "us":
        return f"{number:,.2f}"
    grouped = f"{absolute:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if kind == "br_no_space":
        return f"{sign}R${grouped}"
    return f"{sign}R$ {grouped}"


def generate_applicants(rng: np.random.Generator, n: int) -> pd.DataFrame:
    states = list(STATE_UFS)
    state_p = np.array([STATE_WEIGHTS[s] for s in states], dtype=float)
    state_p = state_p / state_p.sum()
    state_uf = rng.choice(states, size=n, p=state_p)

    emp_p = np.array([0.46, 0.18, 0.10, 0.18, 0.08], dtype=float)
    employment_type = rng.choice(np.array(EMPLOYMENT_TYPES), size=n, p=emp_p)
    housing_status = rng.choice(
        np.array(HOUSING_STATUSES), size=n, p=np.array([0.28, 0.38, 0.24, 0.10])
    )
    dependents = rng.choice(
        np.array([0, 1, 2, 3, 4, 5]), size=n, p=np.array([0.34, 0.28, 0.22, 0.11, 0.04, 0.01])
    )

    age_years = np.clip(rng.beta(2.2, 3.0, size=n) * 47.0 + 21.0, 21.0, 68.0)
    census = pd.Timestamp("2025-01-01", tz="UTC")
    birth_dates = census - pd.to_timedelta((age_years * 365.25).astype(int), unit="D")

    log_income = rng.normal(8.16, 0.55, size=n)
    log_income = log_income + 0.15 * ((age_years - 21.0) / 30.0 - 0.5)
    for emp, mult in EMP_INCOME_MULT.items():
        log_income[employment_type == emp] += np.log(mult)
    monthly_income = np.clip(np.round(np.exp(log_income), 2), 800.0, 80_000.0)

    max_months = np.clip((age_years - 16.0) * 12.0, 0.0, 480.0)
    tenure_frac = rng.beta(1.2, 3.0, size=n)
    employment_months = np.floor(tenure_frac * np.minimum(max_months, 240.0) * 0.7).astype(int)
    unemployed = employment_type == "unemployed"
    n_unemp = int(unemployed.sum())
    if n_unemp:
        employment_months[unemployed] = rng.integers(0, 18, size=n_unemp)

    log_inc_z = (np.log(monthly_income) - np.log(3500.0)) / 0.55
    emp_risk = np.array([EMP_RISK[v] for v in employment_type], dtype=float)
    housing_risk = np.array([HOUSING_RISK[v] for v in housing_status], dtype=float)
    state_risk = np.array([STATE_RISK.get(v, 0.0) for v in state_uf], dtype=float)
    age_risk = 0.25 * (age_years < 26).astype(float) + 0.15 * (age_years > 60).astype(float)
    dep_pressure = dependents / np.clip(monthly_income / 2500.0, 0.5, 10.0)
    noise = rng.normal(0.0, 0.50, size=n)
    latent = (
        0.10
        - 0.55 * log_inc_z
        + emp_risk
        + age_risk
        + 0.20 * dep_pressure
        + housing_risk
        + state_risk
        + noise
    )
    latent = (latent - latent.mean()) / (latent.std() + 1e-9)

    first = rng.choice(np.array(FIRST_NAMES), size=n)
    last_a = rng.choice(np.array(LAST_NAMES), size=n)
    last_b = rng.choice(np.array(LAST_NAMES), size=n)
    use_two = rng.random(n) < 0.75
    full_name = np.where(use_two, first + " " + last_a + " " + last_b, first + " " + last_a)

    created_at = APP_START - pd.to_timedelta(rng.integers(10, 400, size=n), unit="D")

    return pd.DataFrame(
        {
            "applicant_id": [f"APPNT-{i:08d}" for i in range(1, n + 1)],
            "cpf": generate_unique_cpfs(rng, n),
            "full_name": full_name,
            "birth_date": birth_dates,
            "state_uf": state_uf,
            "monthly_income": monthly_income,
            "employment_type": employment_type,
            "employment_months": employment_months,
            "housing_status": housing_status,
            "dependents": dependents,
            "latent_risk": np.round(latent, 6),
            "created_at": created_at,
        }
    )


def generate_applications(
    rng: np.random.Generator, applicants: pd.DataFrame, n_applications: int
) -> pd.DataFrame:
    n_people = len(applicants)
    base = np.arange(min(n_applications, n_people), dtype=int)
    rest_size = n_applications - len(base)
    rest = rng.integers(0, n_people, size=rest_size) if rest_size else np.array([], dtype=int)
    idx = np.concatenate([base, rest])
    rng.shuffle(idx)

    applicant_ids = applicants["applicant_id"].to_numpy()
    latent = applicants["latent_risk"].to_numpy()
    income = applicants["monthly_income"].to_numpy()

    span_days = int((APP_END - APP_START).days)
    applied_at = APP_START + pd.to_timedelta(
        rng.integers(0, span_days + 1, size=n_applications), unit="D"
    )
    applied_at = applied_at + pd.to_timedelta(
        rng.integers(0, 24 * 60, size=n_applications), unit="m"
    )

    amount_mult = rng.lognormal(mean=np.log(3.0), sigma=0.55, size=n_applications)
    requested = np.clip(np.round(income[idx] * amount_mult / 50.0) * 50.0, 800.0, 50_000.0)
    term_p = np.array([0.08, 0.10, 0.28, 0.24, 0.20, 0.10])
    term_months = rng.choice(TERM_CHOICES, size=n_applications, p=term_p)
    purpose = rng.choice(
        np.array(PURPOSES),
        size=n_applications,
        p=np.array([0.38, 0.16, 0.14, 0.12, 0.10, 0.10]),
    )
    channel = rng.choice(
        np.array(CHANNELS),
        size=n_applications,
        p=np.array([0.48, 0.28, 0.14, 0.10]),
    )

    bureau = np.clip(
        np.round(680.0 - 110.0 * latent[idx] + rng.normal(0.0, 55.0, size=n_applications)), 300, 900
    )
    bureau = bureau.astype(int)
    has_debt = rng.random(n_applications) < 0.72
    existing = np.zeros(n_applications, dtype=float)
    if has_debt.any():
        existing[has_debt] = np.round(
            income[idx][has_debt] * rng.lognormal(mean=1.2, sigma=0.7, size=int(has_debt.sum())),
            2,
        )
        existing = np.clip(existing, 0.0, 200_000.0)

    est_pmt = payment_amount_vec(requested, np.full(n_applications, 0.035), term_months)
    dti = (existing * 0.04 + est_pmt) / np.clip(income[idx], 1.0, None)
    scorecard = (
        0.85
        - 0.004 * (700 - bureau)
        - 1.5 * np.clip(dti - 0.35, 0.0, 2.0)
        - 0.40 * (income[idx] < 1500).astype(float)
        - 0.30 * (requested / np.clip(income[idx], 1.0, None) > 10).astype(float)
    )
    hard_decline = (
        (bureau < 430) | (income[idx] < 1100) | (dti > 0.85) | (requested > 18.0 * income[idx])
    )
    p_approve = 1.0 / (1.0 + np.exp(-scorecard * 3.0))
    decision = np.where(
        ~hard_decline & (rng.random(n_applications) <= p_approve), "approved", "declined"
    )
    decision = decision.astype(object)
    cancel = (decision == "approved") & (rng.random(n_applications) < 0.07)
    decision[cancel] = "cancelled"

    decided_at = applied_at + pd.to_timedelta(
        rng.integers(30, 5 * 24 * 60, size=n_applications), unit="m"
    )

    frame = pd.DataFrame(
        {
            "application_id": [f"APP-{i:08d}" for i in range(1, n_applications + 1)],
            "applicant_id": applicant_ids[idx],
            "applied_at": applied_at,
            "requested_amount": requested,
            "term_months": term_months,
            "purpose": purpose,
            "channel": channel,
            "bureau_score": bureau,
            "existing_debt": existing,
            "decision": decision,
            "decided_at": decided_at,
            "latent_risk": latent[idx],
            "monthly_income": income[idx],
        }
    )
    frame = frame.sort_values(["applicant_id", "applied_at", "application_id"]).reset_index(
        drop=True
    )
    return frame


def generate_loans(
    rng: np.random.Generator, applications: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Originate loans and draw a planned first miss. Labels are finalized later."""
    approved = applications.loc[applications["decision"] == "approved"].copy()
    if approved.empty:
        raise RuntimeError("No approved applications; cannot originate loans.")
    keep = rng.random(len(approved)) < 0.93
    originated = approved.loc[keep].reset_index(drop=True)
    n = len(originated)
    if n == 0:
        raise RuntimeError("Underwriting originated zero loans; adjust the policy or sample size.")

    lag_days = rng.integers(1, 12, size=n)
    disbursed_at = originated["decided_at"] + pd.to_timedelta(lag_days, unit="D")
    disbursed_at = pd.to_datetime(disbursed_at, utc=True)
    disbursed_at = disbursed_at.clip(upper=LATEST_DISBURSEMENT)

    haircut = rng.uniform(0.97, 1.00, size=n)
    principal = np.round(originated["requested_amount"].to_numpy() * haircut, 2)
    latent = originated["latent_risk"].to_numpy()
    bureau = originated["bureau_score"].to_numpy()
    rate = hypothetical_rate(latent, bureau, rng.normal(0.0, 0.003, size=n))
    term = originated["term_months"].to_numpy().astype(int)
    installment = payment_amount_vec(principal, rate, term)

    income = originated["monthly_income"].to_numpy()
    requested = originated["requested_amount"].to_numpy()
    existing = originated["existing_debt"].to_numpy()
    logit = default_model_logit(latent, requested, income, existing, bureau)
    # Calibrate to the target rate among loans that will be observable.
    complete_mask = np.array([is_observation_complete(ts) for ts in disbursed_at], dtype=bool)
    if complete_mask.any():
        shift = _shift_logit_to_mean(logit[complete_mask], TARGET_DEFAULT_RATE)
    else:
        shift = _shift_logit_to_mean(logit, TARGET_DEFAULT_RATE)
    prob = 1.0 / (1.0 + np.exp(-(logit + shift)))
    intends_delinquency = rng.random(n) < prob
    first_miss = np.zeros(n, dtype=int)
    if intends_delinquency.any():
        first_miss[intends_delinquency] = sample_first_miss(rng, term[intends_delinquency])

    loans = pd.DataFrame(
        {
            "loan_id": [f"LN-{i:08d}" for i in range(1, n + 1)],
            "application_id": originated["application_id"].to_numpy(),
            "disbursed_at": disbursed_at,
            "principal": principal,
            "interest_rate_monthly": rate,
            "term_months": term,
            "installment_amount": installment,
            "is_default_90dpd_12m": intends_delinquency,
            "default_date": pd.Series(pd.NaT, index=np.arange(n), dtype="datetime64[ns, UTC]"),
            "observation_complete": complete_mask,
            "first_miss_installment": first_miss,
        }
    )
    return loans, float(shift)


def generate_installments(rng: np.random.Generator, loans: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    inst_id = 1
    as_of_day = AS_OF.normalize()
    for row in loans.itertuples(index=False):
        term = int(row.term_months)
        first_miss = (
            int(row.first_miss_installment) if int(row.first_miss_installment) > 0 else 10_000
        )
        is_default = first_miss < 10_000
        amount = float(row.installment_amount)
        disbursed = pd.Timestamp(row.disbursed_at)
        for number in range(1, term + 1):
            due = add_months(disbursed, number)
            due_day = pd.Timestamp(due).normalize()
            if is_default and number >= first_miss:
                paid_at = pd.NaT
                amount_paid = 0.0
                dpd = int((as_of_day - due_day).days) if due_day <= as_of_day else 0
            else:
                draw = float(rng.random())
                if draw < 0.72:
                    delay = int(rng.integers(0, 4))
                elif draw < 0.93:
                    delay = int(rng.integers(4, 18))
                else:
                    delay = int(rng.integers(18, 55))
                paid = pd.Timestamp(due) + pd.Timedelta(days=delay)
                paid_day = paid.normalize()
                if due_day > as_of_day:
                    paid_at = pd.NaT
                    amount_paid = 0.0
                    dpd = 0
                elif paid_day > as_of_day:
                    paid_at = pd.NaT
                    amount_paid = 0.0
                    dpd = int((as_of_day - due_day).days)
                else:
                    paid_at = paid
                    amount_paid = amount
                    dpd = 0
            records.append(
                {
                    "installment_id": f"INST-{inst_id:08d}",
                    "loan_id": row.loan_id,
                    "installment_number": number,
                    "due_date": due,
                    "amount_due": amount,
                    "paid_at": paid_at,
                    "amount_paid": amount_paid,
                    "days_past_due": int(max(dpd, 0)),
                }
            )
            inst_id += 1
    return pd.DataFrame.from_records(records)


def generate_counterfactual_outcomes(
    rng: np.random.Generator,
    applications: pd.DataFrame,
    loans: pd.DataFrame,
    intercept_shift: float,
) -> pd.DataFrame:
    """Synthetic outcomes for applications that never originated a loan.

    Written only to data/ground_truth (never to the messy raw exports). Used
    later to study selection bias. The default flag uses the same latent-risk
    model and intercept as originated loans.
    """
    originated = set(loans["application_id"].tolist())
    pending = applications.loc[~applications["application_id"].isin(originated)].copy()
    n = len(pending)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "application_id",
                "decision",
                "hypothetical_principal",
                "hypothetical_interest_rate_monthly",
                "hypothetical_term_months",
                "is_default_90dpd_12m_counterfactual",
            ]
        )
    haircut = rng.uniform(0.97, 1.00, size=n)
    principal = np.round(pending["requested_amount"].to_numpy() * haircut, 2)
    latent = pending["latent_risk"].to_numpy()
    bureau = pending["bureau_score"].to_numpy()
    rate = hypothetical_rate(latent, bureau, rng.normal(0.0, 0.003, size=n))
    term = pending["term_months"].to_numpy().astype(int)
    logit = default_model_logit(
        latent,
        pending["requested_amount"].to_numpy(),
        pending["monthly_income"].to_numpy(),
        pending["existing_debt"].to_numpy(),
        bureau,
    )
    prob = 1.0 / (1.0 + np.exp(-(logit + intercept_shift)))
    is_default = rng.random(n) < prob
    return pd.DataFrame(
        {
            "application_id": pending["application_id"].to_numpy(),
            "decision": pending["decision"].to_numpy(),
            "hypothetical_principal": principal,
            "hypothetical_interest_rate_monthly": rate,
            "hypothetical_term_months": term,
            "is_default_90dpd_12m_counterfactual": is_default,
        }
    )


def _write_ground_truth(
    data_dir: Path,
    applicants: pd.DataFrame,
    applications: pd.DataFrame,
    loans: pd.DataFrame,
    installments: pd.DataFrame,
    counterfactuals: pd.DataFrame | None = None,
) -> None:
    gt = data_dir / "ground_truth"
    gt.mkdir(parents=True, exist_ok=True)
    applicants.to_parquet(gt / "applicants.parquet", index=False)
    applications.drop(columns=["latent_risk", "monthly_income"]).to_parquet(
        gt / "applications.parquet", index=False
    )
    loans.to_parquet(gt / "loans.parquet", index=False)
    installments.to_parquet(gt / "installments.parquet", index=False)
    if counterfactuals is not None:
        counterfactuals.to_parquet(gt / "counterfactual_outcomes.parquet", index=False)


def _assign_batch(timestamps: pd.Series) -> np.ndarray:
    values = pd.to_datetime(timestamps, utc=True)
    out = np.empty(len(values), dtype=object)
    previous = pd.Timestamp("1677-01-01", tz="UTC")
    for batch in BATCHES:
        mask = (values > previous) & (values <= batch["cutoff"])
        out[mask.to_numpy()] = batch["batch_id"]
        previous = batch["cutoff"]
    missing = pd.isna(pd.Series(out, dtype=object))
    if missing.any():
        # Timestamps after the last cutoff (should not happen for events we export).
        out[missing.to_numpy()] = ""
    return out


def _apply_category_noise(
    rng: np.random.Generator,
    series: pd.Series,
    field: str,
    n_take: int,
) -> tuple[pd.Series, np.ndarray]:
    out = series.astype(object).copy()
    idx = _choose_indices(rng, len(out), n_take)
    for i in idx:
        canonical = str(out.iloc[i])
        if field == "state_uf":
            options = _state_variants(canonical)
        else:
            options = CATEGORY_VARIANTS[field][canonical]
        # Prefer a non-canonical spelling when one exists.
        alts = [opt for opt in options if opt != canonical]
        out.iloc[i] = alts[int(rng.integers(0, len(alts)))] if alts else options[0]
    return out, idx


def _apply_date_noise(
    rng: np.random.Generator, series: pd.Series, n_take: int, timestamp: bool
) -> tuple[pd.Series, np.ndarray]:
    out = series.copy().astype(object)
    # First write ISO for every row so the file is mixed, not uniformly messy.
    default_kind = "iso_space" if timestamp else "iso"
    for i, value in enumerate(series):
        out.iloc[i] = _format_date(value, default_kind)
    kinds = (
        ("dmy_slash", "dmy_dash", "ymd_slash", "dmy_dot", "iso_z")
        if timestamp
        else (
            "dmy_slash",
            "dmy_dash",
            "ymd_slash",
            "dmy_dot",
        )
    )
    idx = _choose_indices(rng, len(out), n_take)
    for i in idx:
        kind = kinds[int(rng.integers(0, len(kinds)))]
        out.iloc[i] = _format_date(series.iloc[i], kind)
    return out, idx


def _apply_money_noise(
    rng: np.random.Generator, series: pd.Series, n_take: int
) -> tuple[pd.Series, np.ndarray]:
    out = series.copy().astype(object)
    for i, value in enumerate(series):
        out.iloc[i] = _format_money(value, "numeric")
    kinds = ("br", "us", "br_no_space")
    idx = _choose_indices(rng, len(out), n_take)
    for i in idx:
        kind = kinds[int(rng.integers(0, len(kinds)))]
        out.iloc[i] = _format_money(float(series.iloc[i]), kind)
    return out, idx


def _mess_entity(
    rng: np.random.Generator,
    df: pd.DataFrame,
    *,
    entity: str,
    pk: str,
    batch_id: str,
    file_name: str,
    issues: list[dict],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    # Pandas 3 will not upcast int/float columns when disguised nulls or text
    # formats are written; raw exports are text, so work in object dtype.
    out = df.copy().astype(object)
    n = len(out)

    if entity == "applicants":
        n_cpf = _n_issues(n, 0.12, minimum=3)
        cpf_idx = _choose_indices(rng, n, n_cpf)
        kinds = ("punct", "spaces", "slash", "mixed")
        # Everyone starts as punctuation-free digits; selected rows get messy formats.
        for i in cpf_idx:
            kind = kinds[int(rng.integers(0, len(kinds)))]
            out.at[out.index[i], "cpf"] = _format_cpf(str(df.iloc[i]["cpf"]), kind)
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "malformed_cpf", "cpf")

        out["birth_date"], date_idx = _apply_date_noise(
            rng, df["birth_date"], _n_issues(n, 0.25, minimum=3), timestamp=False
        )
        for i in date_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "mixed_date_format", "birth_date"
            )

        out["monthly_income"], money_idx = _apply_money_noise(
            rng, df["monthly_income"], _n_issues(n, 0.40, minimum=3)
        )
        for i in money_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "currency_as_text", "monthly_income"
            )

        for field in ("employment_type", "housing_status", "state_uf"):
            out[field], cat_idx = _apply_category_noise(
                rng, df[field], field, _n_issues(n, 0.20, minimum=3)
            )
            for i in cat_idx:
                _log_issue(
                    issues, batch_id, file_name, df.iloc[i][pk], "inconsistent_category", field
                )

        # Disguised nulls on optional-looking attributes.
        n_null = _n_issues(n, 0.03, minimum=2)
        null_idx = _choose_indices(rng, n, n_null)
        null_cols = ("employment_months", "housing_status", "dependents", "monthly_income")
        for i in null_idx:
            col = null_cols[int(rng.integers(0, len(null_cols)))]
            token = DISGUISED_NULLS[int(rng.integers(0, len(DISGUISED_NULLS)))]
            if col == "monthly_income" and token == "":
                token = "0"
            out.at[out.index[i], col] = token
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "disguised_null", col)

        n_outliers = _n_issues(n, 0.004, minimum=2)
        out_idx = _choose_indices(rng, n, n_outliers)
        for j, i in enumerate(out_idx):
            key = df.iloc[i][pk]
            if j % 2 == 0:
                out.at[out.index[i], "monthly_income"] = -abs(float(df.iloc[i]["monthly_income"]))
                _log_issue(issues, batch_id, file_name, key, "outlier", "monthly_income")
            else:
                # Absurd age: either ~120 years old or a child.
                if j % 4 == 1:
                    out.at[out.index[i], "birth_date"] = _format_date(
                        pd.Timestamp("1902-04-01", tz="UTC"), "iso"
                    )
                else:
                    out.at[out.index[i], "birth_date"] = _format_date(
                        pd.Timestamp("2019-08-15", tz="UTC"), "iso"
                    )
                _log_issue(issues, batch_id, file_name, key, "outlier", "birth_date")

    elif entity == "applications":
        out["applied_at"], a_idx = _apply_date_noise(
            rng, df["applied_at"], _n_issues(n, 0.25, minimum=3), timestamp=True
        )
        for i in a_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "mixed_date_format", "applied_at"
            )
        out["decided_at"], d_idx = _apply_date_noise(
            rng, df["decided_at"], _n_issues(n, 0.25, minimum=3), timestamp=True
        )
        for i in d_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "mixed_date_format", "decided_at"
            )

        out["requested_amount"], m_idx = _apply_money_noise(
            rng, df["requested_amount"], _n_issues(n, 0.40, minimum=3)
        )
        for i in m_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "currency_as_text", "requested_amount"
            )
        out["existing_debt"], e_idx = _apply_money_noise(
            rng, df["existing_debt"], _n_issues(n, 0.40, minimum=3)
        )
        for i in e_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "currency_as_text", "existing_debt"
            )

        for field in ("purpose", "channel", "decision"):
            out[field], c_idx = _apply_category_noise(
                rng, df[field], field, _n_issues(n, 0.20, minimum=3)
            )
            for i in c_idx:
                _log_issue(
                    issues, batch_id, file_name, df.iloc[i][pk], "inconsistent_category", field
                )

        n_null = _n_issues(n, 0.03, minimum=2)
        null_idx = _choose_indices(rng, n, n_null)
        null_cols = ("purpose", "channel", "existing_debt", "requested_amount")
        for i in null_idx:
            col = null_cols[int(rng.integers(0, len(null_cols)))]
            token = DISGUISED_NULLS[int(rng.integers(0, len(DISGUISED_NULLS)))]
            if col in {"existing_debt", "requested_amount"} and token == "":
                token = "0"
            out.at[out.index[i], col] = token
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "disguised_null", col)

        n_outliers = _n_issues(n, 0.002, minimum=1)
        for i in _choose_indices(rng, n, n_outliers):
            out.at[out.index[i], "requested_amount"] = 9_999_999.00
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "outlier", "requested_amount")

    elif entity == "loans":
        out["disbursed_at"], d_idx = _apply_date_noise(
            rng, df["disbursed_at"], _n_issues(n, 0.25, minimum=3), timestamp=True
        )
        for i in d_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "mixed_date_format", "disbursed_at"
            )
        out["principal"], p_idx = _apply_money_noise(
            rng, df["principal"], _n_issues(n, 0.40, minimum=3)
        )
        for i in p_idx:
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "currency_as_text", "principal")
        out["installment_amount"], i_idx = _apply_money_noise(
            rng, df["installment_amount"], _n_issues(n, 0.40, minimum=3)
        )
        for i in i_idx:
            _log_issue(
                issues,
                batch_id,
                file_name,
                df.iloc[i][pk],
                "currency_as_text",
                "installment_amount",
            )

    elif entity == "installments":
        out["due_date"], due_idx = _apply_date_noise(
            rng, df["due_date"], _n_issues(n, 0.20, minimum=3), timestamp=False
        )
        for i in due_idx:
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "mixed_date_format", "due_date")
        # paid_at may be null; only format non-null values, keep blanks as empty.
        paid_series = df["paid_at"]
        n_paid = int(paid_series.notna().sum())
        out["paid_at"] = [_format_date(v, "iso_space") if pd.notna(v) else "" for v in paid_series]
        paid_targets = np.flatnonzero(paid_series.notna().to_numpy())
        n_take_paid = min(_n_issues(n_paid, 0.20, minimum=3), len(paid_targets))
        if n_take_paid:
            chosen = np.sort(rng.choice(paid_targets, size=n_take_paid, replace=False))
            kinds = ("dmy_slash", "dmy_dash", "ymd_slash", "iso_z")
            for i in chosen:
                kind = kinds[int(rng.integers(0, len(kinds)))]
                out.at[out.index[i], "paid_at"] = _format_date(paid_series.iloc[i], kind)
                _log_issue(
                    issues, batch_id, file_name, df.iloc[i][pk], "mixed_date_format", "paid_at"
                )

        out["amount_due"], ad_idx = _apply_money_noise(
            rng, df["amount_due"], _n_issues(n, 0.30, minimum=3)
        )
        for i in ad_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "currency_as_text", "amount_due"
            )
        out["amount_paid"], ap_idx = _apply_money_noise(
            rng, df["amount_paid"], _n_issues(n, 0.30, minimum=3)
        )
        for i in ap_idx:
            _log_issue(
                issues, batch_id, file_name, df.iloc[i][pk], "currency_as_text", "amount_paid"
            )

        n_null = _n_issues(n, 0.02, minimum=2)
        for i in _choose_indices(rng, n, n_null):
            out.at[out.index[i], "amount_paid"] = "N/A"
            _log_issue(issues, batch_id, file_name, df.iloc[i][pk], "disguised_null", "amount_paid")

    out["extracted_at"] = BATCH_BY_ID[batch_id]["extracted_at"].strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.reset_index(drop=True)


def _duplicate_rows(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    if len(indices) == 0 or df.empty:
        return df.iloc[0:0].copy()
    return df.iloc[list(indices)].copy()


def _schema_and_write(path: Path, df: pd.DataFrame, batch: dict, entity: str) -> None:
    written = df.copy()
    if batch["batch_id"] == "batch_002":
        if entity == "applicants" and "monthly_income" in written.columns:
            written = written.rename(columns={"monthly_income": "income_monthly"})
            written["occupation_note"] = ""
        if entity == "applications" and "requested_amount" in written.columns:
            written = written.rename(columns={"requested_amount": "amount_requested"})
    if batch["batch_id"] == "batch_003":
        if entity == "applicants":
            written["customer_segment"] = "retail"
        if entity == "applications":
            written["source_system"] = "LOS-CORE"
    path.parent.mkdir(parents=True, exist_ok=True)
    written.to_csv(
        path,
        sep=batch["delimiter"],
        index=False,
        encoding=batch["encoding"],
        lineterminator="\n",
    )


def _log_schema_changes(issues: list[dict]) -> None:
    # batch_002: delimiter, encoding, renamed columns, added column
    for entity, renamed, added in (
        ("applicants", "monthly_income", "occupation_note"),
        ("applications", "requested_amount", None),
        ("loans", None, None),
        ("installments", None, None),
    ):
        file_name = f"{entity}.csv"
        _log_issue(issues, "batch_002", file_name, "__FILE__", "schema_change", "delimiter")
        _log_issue(issues, "batch_002", file_name, "__FILE__", "schema_change", "encoding")
        if renamed:
            _log_issue(issues, "batch_002", file_name, "__FILE__", "schema_change", renamed)
        if added:
            _log_issue(issues, "batch_002", file_name, "__FILE__", "schema_change", added)
    _log_issue(
        issues, "batch_003", "applicants.csv", "__FILE__", "schema_change", "customer_segment"
    )
    _log_issue(
        issues, "batch_003", "applications.csv", "__FILE__", "schema_change", "source_system"
    )


def build_raw_exports(
    rng: np.random.Generator,
    applicants: pd.DataFrame,
    applications: pd.DataFrame,
    loans: pd.DataFrame,
    installments: pd.DataFrame,
    data_dir: Path,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    issues: list[dict] = []
    first_app = applications.groupby("applicant_id", as_index=True)["applied_at"].min()
    applicant_event = applicants["applicant_id"].map(first_app)
    applicant_event = applicant_event.fillna(applicants["created_at"])
    applicants = applicants.copy()
    applications = applications.copy()
    loans = loans.copy()
    installments = installments.copy()
    applicants["_batch"] = _assign_batch(applicant_event)
    applications["_batch"] = _assign_batch(applications["applied_at"])
    loans["_batch"] = _assign_batch(loans["disbursed_at"])
    # Only export installments already due as of the global cutoff.
    installments = installments.loc[
        pd.to_datetime(installments["due_date"], utc=True) <= AS_OF
    ].copy()
    installments["_batch"] = _assign_batch(installments["due_date"])

    # Late-arriving installments: withhold a slice of early-period rows until batch_003.
    early_mask = installments["_batch"].isin(["batch_001", "batch_002"])
    early_idx = np.flatnonzero(early_mask.to_numpy())
    n_late = _n_issues(len(early_idx), 0.04, minimum=5)
    late_idx = (
        early_idx[_choose_indices(rng, len(early_idx), n_late)]
        if len(early_idx)
        else np.array([], dtype=int)
    )
    late_ids = (
        set(installments.iloc[late_idx]["installment_id"].tolist()) if len(late_idx) else set()
    )
    for inst_id in sorted(late_ids):
        _log_issue(
            issues, "batch_003", "installments.csv", inst_id, "late_arriving_event", "due_date"
        )

    entity_specs = (
        ("applicants", applicants, "applicant_id"),
        ("applications", applications, "application_id"),
        ("loans", loans, "loan_id"),
        ("installments", installments, "installment_id"),
    )

    # Clean home-batch slices (late installments skipped until batch_003).
    slices: dict[str, dict[str, pd.DataFrame]] = {b["batch_id"]: {} for b in BATCHES}
    for entity, frame, pk in entity_specs:
        for batch in BATCHES:
            bid = batch["batch_id"]
            if entity == "installments":
                if bid == "batch_003":
                    mask = (frame["_batch"] == bid) | frame[pk].isin(late_ids)
                else:
                    mask = (frame["_batch"] == bid) & ~frame[pk].isin(late_ids)
            else:
                mask = frame["_batch"] == bid
            part = frame.loc[mask].drop(columns=["_batch"]).reset_index(drop=True)
            # Drop ground-truth-only columns from raw exports.
            drop_cols = [
                c
                for c in part.columns
                if c
                in {
                    "latent_risk",
                    "created_at",
                    "is_default_90dpd_12m",
                    "default_date",
                    "observation_complete",
                    "first_miss_installment",
                    "days_past_due",
                    "monthly_income",
                }
                and not (entity == "applicants" and c == "monthly_income")
            ]
            if entity == "applications":
                drop_cols = [c for c in drop_cols]  # monthly_income / latent already handled
            part = part.drop(columns=drop_cols, errors="ignore")
            slices[bid][entity] = part

    # Mess each home slice first, then attach resends/duplicates/orphans.
    messy: dict[str, dict[str, pd.DataFrame]] = {b["batch_id"]: {} for b in BATCHES}
    for batch in BATCHES:
        bid = batch["batch_id"]
        for entity, _frame, pk in entity_specs:
            messy[bid][entity] = _mess_entity(
                rng,
                slices[bid][entity],
                entity=entity,
                pk=pk,
                batch_id=bid,
                file_name=f"{entity}.csv",
                issues=issues,
            )

    # Intra-batch exact duplicates and cross-batch exact / near-duplicate resends.
    for entity, _frame, pk in entity_specs:
        for batch, nxt in zip(BATCHES, BATCHES[1:] + [None]):
            bid = batch["batch_id"]
            current = messy[bid][entity]
            if current.empty:
                continue
            n = len(current)
            # Intra-batch exact duplicate (resend inside the same extract).
            rate = 0.008 if entity == "installments" else 0.012
            dup_idx = _choose_indices(rng, n, _n_issues(n, rate, minimum=2))
            if len(dup_idx):
                extra = _duplicate_rows(current, dup_idx)
                messy[bid][entity] = pd.concat([current, extra], ignore_index=True)
                for i in dup_idx:
                    _log_issue(
                        issues, bid, f"{entity}.csv", current.iloc[i][pk], "exact_duplicate", pk
                    )
            if nxt is None or entity not in {"applicants", "applications", "loans"}:
                continue
            # Refresh current after concat for resend sampling from the original n rows.
            source = current
            next_id = nxt["batch_id"]
            exact_idx = _choose_indices(rng, len(source), _n_issues(len(source), 0.010, minimum=2))
            if len(exact_idx):
                resend = _duplicate_rows(source, exact_idx)
                resend["extracted_at"] = nxt["extracted_at"].strftime("%Y-%m-%dT%H:%M:%SZ")
                messy[next_id][entity] = pd.concat(
                    [messy[next_id][entity], resend], ignore_index=True
                )
                for i in exact_idx:
                    _log_issue(
                        issues, next_id, f"{entity}.csv", source.iloc[i][pk], "exact_duplicate", pk
                    )
            if entity in {"applicants", "applications"}:
                near_idx = _choose_indices(
                    rng, len(source), _n_issues(len(source), 0.008, minimum=2)
                )
                near_rows = []
                for i in near_idx:
                    row = source.iloc[i].copy()
                    if entity == "applicants":
                        row["full_name"] = f"{row['full_name']} "
                        col = "full_name"
                    else:
                        # Tiny amount drift on a field that may already be text.
                        raw_amount = row["requested_amount"]
                        col = "requested_amount"
                        try:
                            as_num = float(
                                str(raw_amount)
                                .replace("R$", "")
                                .replace(" ", "")
                                .replace(".", "", 0)
                            )
                        except ValueError:
                            as_num = None
                        if as_num is not None and isinstance(raw_amount, (int, float)):
                            row["requested_amount"] = float(raw_amount) + 1.0
                        else:
                            row["channel"] = str(row["channel"]) + " "
                            col = "channel"
                    row["extracted_at"] = nxt["extracted_at"].strftime("%Y-%m-%dT%H:%M:%SZ")
                    near_rows.append(row)
                    _log_issue(
                        issues, next_id, f"{entity}.csv", source.iloc[i][pk], "near_duplicate", col
                    )
                if near_rows:
                    messy[next_id][entity] = pd.concat(
                        [messy[next_id][entity], pd.DataFrame(near_rows)], ignore_index=True
                    )

    # Orphan foreign keys: applications without a parent applicant, installments without a loan.
    def _add_orphans(bid: str, n_app: int, n_inst: int, start: int) -> int:
        if n_app:
            rows = []
            for k in range(n_app):
                app_id = f"APP-ORPHAN-{start + k:04d}"
                parent = f"APPNT-MISSING-{start + k:04d}"
                rows.append(
                    {
                        "application_id": app_id,
                        "applicant_id": parent,
                        "applied_at": "15/03/2024",
                        "requested_amount": "R$ 1.000,00",
                        "term_months": 12,
                        "purpose": "personal",
                        "channel": "app",
                        "bureau_score": 600,
                        "existing_debt": "0.00",
                        "decision": "declined",
                        "decided_at": "16/03/2024",
                        "extracted_at": BATCH_BY_ID[bid]["extracted_at"].strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    }
                )
                _log_issue(issues, bid, "applications.csv", app_id, "orphan_fk", "applicant_id")
            messy[bid]["applications"] = pd.concat(
                [messy[bid]["applications"], pd.DataFrame(rows)], ignore_index=True
            )
        if n_inst:
            rows = []
            for k in range(n_inst):
                inst_id = f"INST-ORPHAN-{start + k:04d}"
                loan_id = f"LN-MISSING-{start + k:04d}"
                rows.append(
                    {
                        "installment_id": inst_id,
                        "loan_id": loan_id,
                        "installment_number": 1,
                        "due_date": "15/04/2024",
                        "amount_due": "R$ 250,00",
                        "paid_at": "",
                        "amount_paid": "0.00",
                        "extracted_at": BATCH_BY_ID[bid]["extracted_at"].strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    }
                )
                _log_issue(issues, bid, "installments.csv", inst_id, "orphan_fk", "loan_id")
            messy[bid]["installments"] = pd.concat(
                [messy[bid]["installments"], pd.DataFrame(rows)], ignore_index=True
            )
        return start + max(n_app, n_inst)

    cursor = 1
    cursor = _add_orphans("batch_001", 4, 4, cursor)
    cursor = _add_orphans("batch_002", 3, 3, cursor)
    _add_orphans("batch_003", 3, 3, cursor)

    _log_schema_changes(issues)

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    batch_counts: dict[str, dict[str, int]] = {}
    for batch in BATCHES:
        bid = batch["batch_id"]
        batch_counts[bid] = {}
        for entity, _frame, _pk in entity_specs:
            path = raw_dir / bid / f"{entity}.csv"
            table = messy[bid][entity]
            _schema_and_write(path, table, batch, entity)
            batch_counts[bid][entity] = int(len(table))

    issues_df = pd.DataFrame(issues, columns=["batch", "file", "row_key", "issue_type", "column"])
    issues_df = issues_df.sort_values(
        ["batch", "file", "issue_type", "column", "row_key"], kind="mergesort"
    ).reset_index(drop=True)
    issues_df.to_csv(raw_dir / "_injected_issues.csv", index=False, lineterminator="\n")
    return issues, batch_counts


def _reset_generated(data_dir: Path) -> None:
    raw = data_dir / "raw"
    gt = data_dir / "ground_truth"
    raw.mkdir(parents=True, exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)
    for path in raw.glob("batch_*"):
        shutil.rmtree(path)
    issues = raw / "_injected_issues.csv"
    if issues.exists():
        issues.unlink()
    for path in gt.iterdir():
        if path.name != ".gitkeep" and path.is_file():
            path.unlink()


def generate(
    *,
    seed: int = DEFAULT_SEED,
    n_applicants: int = DEFAULT_N_APPLICANTS,
    n_applications: int = DEFAULT_N_APPLICATIONS,
    output_dir: Path | str | None = None,
) -> GenerationResult:
    if n_applicants < 20 or n_applications < 20:
        raise ValueError("n_applicants and n_applications must be at least 20.")
    data_dir = Path(output_dir) if output_dir is not None else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated(data_dir)

    rng = np.random.default_rng(seed)
    applicants = generate_applicants(rng, n_applicants)
    applications = generate_applications(rng, applicants, n_applications)
    first_app = applications.groupby("applicant_id")["applied_at"].min()
    mapped = pd.to_datetime(applicants["applicant_id"].map(first_app), utc=True)
    adjust = mapped.notna()
    lags = pd.to_timedelta(rng.integers(5, 180, size=int(adjust.sum())), unit="D")
    created = mapped.loc[adjust].reset_index(drop=True) - lags
    applicants.loc[adjust, "created_at"] = created.to_numpy()

    loans, intercept_shift = generate_loans(rng, applications)
    installments = generate_installments(rng, loans)
    loans = label_defaults_from_installments(loans, installments)
    counterfactuals = generate_counterfactual_outcomes(rng, applications, loans, intercept_shift)
    _write_ground_truth(data_dir, applicants, applications, loans, installments, counterfactuals)

    issues, batch_counts = build_raw_exports(
        rng, applicants, applications, loans, installments, data_dir
    )
    issue_counts = {name: 0 for name in ISSUE_TYPES}
    for row in issues:
        issue_counts[row["issue_type"]] = issue_counts.get(row["issue_type"], 0) + 1

    complete = loans.loc[loans["observation_complete"]]
    n_complete = int(len(complete))
    n_immature = int((~loans["observation_complete"]).sum())
    n_defaults = int(complete["is_default_90dpd_12m"].sum())
    default_rate = float(n_defaults / n_complete) if n_complete else float("nan")
    n_cf = int(len(counterfactuals))
    cf_rate = (
        float(counterfactuals["is_default_90dpd_12m_counterfactual"].mean())
        if n_cf
        else float("nan")
    )

    result = GenerationResult(
        seed=seed,
        n_applicants_requested=n_applicants,
        n_applications_requested=n_applications,
        n_applicants=int(len(applicants)),
        n_applications=int(len(applications)),
        n_loans=int(len(loans)),
        n_installments=int(len(installments)),
        n_loans_observation_complete=n_complete,
        n_loans_immature=n_immature,
        n_defaults=n_defaults,
        default_rate=default_rate,
        n_counterfactuals=n_cf,
        counterfactual_default_rate=cf_rate,
        issue_counts=issue_counts,
        batch_counts=batch_counts,
        output_dir=str(data_dir),
    )
    summary = asdict(result)
    summary["client"] = "Credara Fintech (fictional)"
    summary["default_definition"] = (
        "90+ DPD on an installment due within 12 months of disbursement; "
        "observable only if disbursed_at + 12 months + 90 days <= 2026-10-31 UTC"
    )
    summary["default_rate_is_synthetic"] = True
    summary["counterfactual_outcomes_are_synthetic"] = True
    summary["as_of"] = str(AS_OF)
    summary["cpf_policy"] = "11-digit identifiers with deliberately invalid check digits"
    summary["first_miss_policy"] = (
        "Intended delinquencies draw first_miss over 1..min(12, term) "
        "with extra weight on months 3-6"
    )
    (data_dir / "ground_truth" / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Credara Fintech ground truth and messy CSV exports."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (default: 42)")
    parser.add_argument(
        "--n-applicants",
        type=int,
        default=DEFAULT_N_APPLICANTS,
        help="Applicant count before --scale (default: 20000)",
    )
    parser.add_argument(
        "--n-applications",
        type=int,
        default=DEFAULT_N_APPLICATIONS,
        help="Application count before --scale (default: 50000)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier applied to applicant and application counts (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Data directory (default: <repo>/data)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> GenerationResult:
    args = _parse_args(argv)
    n_applicants = max(20, int(round(args.n_applicants * args.scale)))
    n_applications = max(20, int(round(args.n_applications * args.scale)))
    result = generate(
        seed=args.seed,
        n_applicants=n_applicants,
        n_applications=n_applications,
        output_dir=args.output_dir,
    )
    print("Credara Fintech (fictional) — synthetic generation complete")
    print(f"  seed:            {result.seed}")
    print(f"  applicants:      {result.n_applicants}")
    print(f"  applications:    {result.n_applications}")
    print(f"  loans:           {result.n_loans}")
    print(f"  installments:    {result.n_installments}")
    print(f"  complete loans:  {result.n_loans_observation_complete}")
    print(f"  immature loans:  {result.n_loans_immature}")
    print(f"  defaults:        {result.n_defaults}")
    print(f"  default rate:    {result.default_rate:.4f} (synthetic, observable 90+ DPD in 12m)")
    print(f"  counterfactuals: {result.n_counterfactuals}")
    print(
        f"  cf default rate: {result.counterfactual_default_rate:.4f} "
        "(synthetic, declined/cancelled/approved-not-originated)"
    )
    print("  issue counts:")
    for name in ISSUE_TYPES:
        print(f"    {name}: {result.issue_counts.get(name, 0)}")
    print("  rows per batch:")
    for batch_id, counts in result.batch_counts.items():
        print(f"    {batch_id}: {counts}")
    print(f"  output:          {result.output_dir}")
    return result


if __name__ == "__main__":
    main()
