"""Messy Client Data -> Decisioning: gold/metrics dashboard (Streamlit).

Reads ONLY gold/metrics tables from DuckDB (read-only).
DB path: env MC_DUCKDB_PATH (default data/warehouse.duckdb, relative to the working dir).
Optional env MC_DUCKDB_SCHEMA to force a schema; otherwise each table is located in whatever
schema contains it (handles custom dbt schemas such as main_gold / main_metrics).

All data is SYNTHETIC (fictional Brazilian personal-credit fintech).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.environ.get("MC_DUCKDB_PATH", "data/warehouse.duckdb")
FORCED_SCHEMA = os.environ.get("MC_DUCKDB_SCHEMA")
GOLD_TABLES = [
    "dim_snapshot", "agg_data_quality", "agg_funnel_monthly", "agg_vintage_default",
    "agg_model_performance", "agg_calibration_deciles", "agg_policy_simulation",
    "agg_model_comparison_at_approval_rate", "agg_risk_band_performance",
    "fct_loan_outcomes",
]
SYNTHETIC_BANNER = (
    "⚠️ **SYNTHETIC DATA** — every number on this page comes from synthetic data for a "
    "fictional Brazilian personal-credit fintech. Money results use SYNTHETIC policy assumptions "
    "(margin, LGD, funding cost, acquisition cost). Not real customers, not a real lender."
)
SCORES_PENDING = (
    "Scores pending (M3). `fct_scores` and the model-performance / policy / Proof of Value "
    "tables are empty until `predictions_scores` is written with `baseline_logreg_v1` and `gbm_v1`."
)

st.set_page_config(page_title="Messy Client → Decisioning (synthetic)", layout="wide")


def query(sql: str) -> pd.DataFrame:
    # short-lived read-only connection so the dashboard does not hold the DuckDB file lock
    # while `dbt build` runs (DuckDB allows only one writer process)
    with duckdb.connect(DB_PATH, read_only=True) as con:
        return con.execute(sql).df()


@st.cache_data
def locate_tables() -> dict[str, str]:
    found = query(
        "select table_schema, table_name from information_schema.tables where table_name in ({})".format(
            ",".join(f"'{t}'" for t in GOLD_TABLES))
    ).itertuples(index=False, name=None)
    out: dict[str, str] = {}
    for schema, name in found:
        if FORCED_SCHEMA and schema != FORCED_SCHEMA:
            continue
        out.setdefault(name, f'"{schema}"."{name}"')
    return out


@st.cache_data
def load(table: str) -> pd.DataFrame:
    tables = locate_tables()
    if table not in tables:
        raise RuntimeError(f"Table {table} not found in {DB_PATH}. Run `dbt build` first.")
    return query(f"select * from {tables[table]}")


def is_empty(df: pd.DataFrame) -> bool:
    return df is None or df.empty


def pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


def brl(x) -> str:
    return "—" if pd.isna(x) else ("R$ " + f"{x:,.0f}".replace(",", "."))


def safe_div(n, d):
    if d is None or pd.isna(d) or d == 0:
        return float("nan")
    return n / d



def find_model_metrics_path() -> Path | None:
    env = os.environ.get("MC_MODEL_METRICS_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    repo_from_db = Path(DB_PATH).resolve().parent.parent / "docs" / "model_metrics.json"
    repo_from_app = Path(__file__).resolve().parent.parent / "docs" / "model_metrics.json"
    candidates.extend([repo_from_db, repo_from_app, Path("docs/model_metrics.json")])
    for c in candidates:
        if c.is_file():
            return c
    return None


@st.cache_data
def load_published_metrics() -> dict | None:
    path = find_model_metrics_path()
    if path is None:
        return None
    with path.open() as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


st.title("Messy Client Data → Decisioning")
st.warning(SYNTHETIC_BANNER)
with st.sidebar:
    st.error("SYNTHETIC DATA ONLY")
    st.caption(f"DuckDB: `{DB_PATH}`")
    snap = load("dim_snapshot")
    if not is_empty(snap):
        st.caption(f"Observation cutoff (as_of_date): **{snap['as_of_date'].iloc[0]}** "
                   f"({snap['as_of_method'].iloc[0]})")
    st.markdown("Sections: Data quality · Funnel · Vintages · Model performance · Policy · Proof of Value")

# ---------------------------------------------------------------- data quality
st.header("1. Data quality (quarantine)")
dq = load("agg_data_quality")
if is_empty(dq):
    st.info("No data-quality rows yet.")
else:
    has_type = "check_type" in dq.columns
    dq_q = dq[dq["check_type"] == "quarantine"] if has_type else dq
    dq_c = dq[dq["check_type"] == "completeness"] if has_type else dq.iloc[0:0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rejected rows", f"{int(dq_q['n_rejected'].sum()) if not is_empty(dq_q) else 0:,}")
    if not is_empty(dq_q):
        batch = dq_q.drop_duplicates(["source_entity", "batch_id"])
        tot_rej, tot_acc = batch["n_rejected_batch"].sum(), batch["n_accepted_batch"].sum()
        c2.metric("Overall reject rate", pct(safe_div(tot_rej, tot_rej + tot_acc)))
        c3.metric("Distinct failed rules", int(dq_q["rule_failed"].nunique()))
    else:
        c2.metric("Overall reject rate", "—")
        c3.metric("Distinct failed rules", "0")
    miss = dq_c[dq_c["rule_failed"] == "existing_debt_missing"] if not is_empty(dq_c) else dq_c
    if not is_empty(miss):
        n_miss = int(miss["n_rejected"].sum())
        n_den = int(miss["n_accepted_batch"].sum())
        c4.metric("Missing existing_debt rate", pct(safe_div(n_miss, n_den)),
                  help=f"{n_miss:,} of {n_den:,} accepted applications (completeness, not quarantine)")
    else:
        c4.metric("Missing existing_debt rate", "—",
                  help="Completeness metric; stays — until silver emits NULL existing_debt (today missing debt is still coalesced to 0).")
    if not is_empty(dq_q):
        by_rule = dq_q.groupby(["source_entity", "rule_failed"], as_index=False)["n_rejected"].sum()
        st.plotly_chart(px.bar(by_rule, x="rule_failed", y="n_rejected", color="source_entity",
                               title="Rejected rows by rule and entity (synthetic)"), use_container_width=True)

# ---------------------------------------------------------------------- funnel
st.header("2. Funnel: applications → approved → booked")
fn = load("agg_funnel_monthly")
if is_empty(fn):
    st.info("No funnel rows yet.")
else:
    fn = fn.sort_values("application_month")
    tot = fn[["n_applications", "n_cancelled", "n_declined", "n_approved", "n_booked"]].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{int(tot.n_applications):,}")
    c2.metric("Approval rate (approved / decided)", pct(safe_div(tot.n_approved, tot.n_approved + tot.n_declined)))
    c3.metric("Take-up (booked / approved)", pct(safe_div(tot.n_booked, tot.n_approved)))
    c4.metric("End-to-end (booked / applications)", pct(safe_div(tot.n_booked, tot.n_applications)))
    colA, colB = st.columns([1, 2])
    colA.plotly_chart(go.Figure(go.Funnel(y=["Applications", "Approved", "Booked"],
                                          x=[tot.n_applications, tot.n_approved, tot.n_booked])),
                      use_container_width=True)
    colB.plotly_chart(px.line(fn, x="application_month", y=["approval_rate", "take_up_rate", "end_to_end_rate"],
                              title="Monthly funnel rates (synthetic)"), use_container_width=True)

# -------------------------------------------------------------------- vintages
st.header("3. Vintages: 12-month default rate")
vt = load("agg_vintage_default")
if is_empty(vt):
    st.info("No vintage rows yet.")
else:
    vt = vt.sort_values("vintage_month")
    st.caption("default_rate_12m = defaults / observable loans. Immature vintages (12m + 90d window not "
               "complete at cutoff) have no rate and are never counted as good.")
    fig = go.Figure()
    fig.add_bar(x=vt["vintage_month"], y=vt["n_observable"], name="observable loans", opacity=0.35, yaxis="y2")
    fig.add_bar(x=vt["vintage_month"], y=vt["n_immature"], name="immature loans", opacity=0.2, yaxis="y2")
    fig.add_scatter(x=vt["vintage_month"], y=vt["default_rate_12m"], name="default_rate_12m", mode="lines+markers")
    fig.update_layout(barmode="stack", yaxis=dict(title="12m default rate", tickformat=".0%"),
                      yaxis2=dict(title="loans", overlaying="y", side="right"), title="Default rate by vintage (synthetic)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------- model performance
st.header("4. Model performance & calibration by version")
perf = load("agg_model_performance")
cal = load("agg_calibration_deciles")
if is_empty(perf) and is_empty(cal):
    st.info(SCORES_PENDING)
else:
    st.caption("Population: scored applications with an observable outcome (approved, booked, matured). "
               "Headline split = OOT. `bureau_score` is the incumbent policy (higher score = lower risk); "
               "it has no calibrated PD, so Brier is blank. On the published OOT set, logistic leads "
               "slightly on AUC; paired bootstrap CIs for GBM minus bureau (AUC, KS, and equal-approval "
               "bad-rate deltas) include 0, so GBM has no demonstrated ranking gain over the bureau.")
    if is_empty(perf):
        st.info(SCORES_PENDING)
    else:
        show = perf.sort_values(["split", "model_version"])[
            ["model_version", "split", "n_observed", "n_bad", "auc", "gini", "ks", "brier_score",
             "avg_predicted_pd", "observed_default_rate", "is_sample_sufficient"]]
        st.dataframe(show.style.format({"auc": "{:.3f}", "gini": "{:.3f}", "ks": "{:.3f}", "brier_score": "{:.4f}",
                                        "avg_predicted_pd": "{:.1%}", "observed_default_rate": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
        published = load_published_metrics()
        if published and "metrics" in published:
            rows = []
            for mv, block in published["metrics"].items():
                oot = block.get("oot_main") or {}
                auc_ci = oot.get("auc_ci") or [None, None]
                ks_ci = oot.get("ks_ci") or [None, None]
                rows.append({
                    "model_version": mv,
                    "n": oot.get("n"),
                    "defaults": oot.get("defaults"),
                    "auc": oot.get("auc"),
                    "auc_ci_low": None if not auc_ci else auc_ci[0],
                    "auc_ci_high": None if not auc_ci else auc_ci[1],
                    "ks": oot.get("ks"),
                    "ks_ci_low": None if not ks_ci else ks_ci[0],
                    "ks_ci_high": None if not ks_ci else ks_ci[1],
                    "brier": oot.get("brier"),
                })
            pub = pd.DataFrame(rows)
            st.caption("Published OOT metrics from `docs/model_metrics.json` (same observable OOT population; "
                       "CIs shown when the JSON includes them).")
            st.dataframe(pub.style.format({
                "auc": "{:.3f}", "auc_ci_low": "{:.3f}", "auc_ci_high": "{:.3f}",
                "ks": "{:.3f}", "ks_ci_low": "{:.3f}", "ks_ci_high": "{:.3f}",
                "brier": "{:.4f}",
            }, na_rep="—"), use_container_width=True, hide_index=True)
            pb = published.get("paired_bootstrap") or {}
            if pb.get("delta_auc"):
                d = pb["delta_auc"]
                ci = d.get("ci") or [None, None]
                includes0 = ci[0] is not None and ci[1] is not None and ci[0] <= 0 <= ci[1]
                st.caption(
                    f"Paired bootstrap Δ AUC (GBM − bureau): mean {d.get('mean'):.4f}, "
                    f"CI [{ci[0]:.4f}, {ci[1]:.4f}] over {pb.get('n_draws', '—')} draws"
                    + (", which includes 0 (no demonstrated ranking gain)." if includes0 else ".")
                )
    if is_empty(cal):
        st.info(SCORES_PENDING)
    else:
        fig = px.line(cal, x="avg_predicted_pd", y="observed_default_rate", color="model_version", markers=True,
                      hover_data=["pd_decile", "n_observed"], title="Calibration by PD decile, OOT (synthetic)")
        mx = float(max(cal["avg_predicted_pd"].max(), cal["observed_default_rate"].max()))
        if pd.notna(mx) and mx > 0:
            fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="perfect calibration", line=dict(dash="dash"))
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- risk bands
st.header("4b. Risk bands A–D (OOT, observable)")
bands = load("agg_risk_band_performance") if "agg_risk_band_performance" in locate_tables() else pd.DataFrame()
if is_empty(bands):
    st.info("No risk-band rows yet (need scores with a calibrated PD).")
else:
    st.caption("Four model-specific bands (A–D). `risk_band` is taken from `predictions_scores` "
               "(source of truth). Seed `policy_assumptions` documents the PD edges used to validate "
               "that assignment (logistic A < 0.0407, B < 0.0734, C < 0.1525; "
               "GBM A < 0.0454, B < 0.0879, C < 0.1620; D is the remainder in each model). "
               "The incumbent bureau policy has no PD band.")
    fig = px.bar(bands.sort_values(["model_version", "band_order"]),
                 x="risk_band", y="n_observed", color="model_version", barmode="group",
                 title="OOT observable volume by risk band (synthetic)")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(bands.sort_values(["model_version", "band_order"]),
                 x="risk_band", y="observed_default_rate", color="model_version", barmode="group",
                 title="Observed default rate by risk band, OOT (synthetic)")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bands.sort_values(["model_version", "band_order"]).style.format({
        "observed_default_rate": "{:.1%}", "avg_predicted_pd": "{:.2%}", "pd_low": "{:.2%}", "pd_high": "{:.2%}",
    }), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------- policy
st.header("5. Policy curve: approval × default × net result (OOT)")
pol = load("agg_policy_simulation")
if is_empty(pol):
    st.info(SCORES_PENDING)
else:
    net_m = float(pol["net_interest_margin_annual"].iloc[0]) if "net_interest_margin_annual" in pol.columns else float(pol["gross_interest_margin_annual"].iloc[0])
    fund_m = float(pol["funding_cost_annual"].iloc[0]) if "funding_cost_annual" in pol.columns else None
    st.caption("Approve if score ≥ cutoff. Observed metrics exist only for applications the legacy policy "
               "approved AND booked AND matured (reject bias) — see coverage. Expected loss = PD × EAD "
               "proxy × LGD (EAD proxy = requested_amount). Margin is an average-balance approximation "
               "of an amortized loan, net of synthetic funding cost. Assumptions (SYNTHETIC): net annual "
               "margin {:.1%} (= gross {:.1%}"
               "{}), LGD {:.0%}, acquisition R$ {:.0f} per loan, underwriting R$ {:.0f} per application. "
               "The incumbent `bureau_score` policy is compared in Proof of Value (it has no cutoff grid here).".format(
                   net_m, pol["gross_interest_margin_annual"].iloc[0],
                   (f", funding cost {fund_m:.1%}" if fund_m is not None else ""),
                   pol["lgd"].iloc[0],
                   pol["acquisition_cost_brl"].iloc[0], pol["underwriting_cost_brl"].iloc[0]))
    models = sorted(pol["model_version"].unique())
    cA, cB = st.columns([1, 3])
    default_m = "gbm_v1" if "gbm_v1" in models else models[0]
    model = cA.selectbox("Model version", models, index=models.index(default_m))
    cuts = sorted(pol["score_cutoff"].unique())
    cutoff = cA.select_slider("Score cutoff", options=cuts, value=cuts[len(cuts) // 2])
    pm = pol[pol["model_version"] == model].sort_values("score_cutoff")
    row = pm[pm["score_cutoff"] == cutoff].iloc[0]
    cA.metric("Approval rate", pct(row.approval_rate))
    cA.metric("Observed default rate", pct(row.observed_default_rate), help=f"n observed = {int(row.n_observed_outcomes)}")
    cA.metric("Observed outcome coverage", pct(row.observed_outcome_coverage))
    cA.metric("Expected net result", brl(row.expected_net_result_brl))
    cA.metric("Observed net result (covered subset)", brl(row.observed_net_result_brl),
              help=f"per observed loan: {brl(row.observed_net_result_per_loan_brl)}")
    fig = go.Figure()
    fig.add_scatter(x=pm["approval_rate"], y=pm["observed_default_rate"], name="observed default rate", mode="lines+markers")
    fig.add_scatter(x=pm["approval_rate"], y=pm["avg_predicted_pd"], name="avg predicted PD", mode="lines", line=dict(dash="dot"))
    fig.add_scatter(x=pm["approval_rate"], y=pm["expected_net_result_brl"], name="expected net result (BRL)", yaxis="y2", mode="lines")
    fig.add_scatter(x=pm["approval_rate"], y=pm["observed_net_result_brl"], name="observed net result (BRL, covered subset)", yaxis="y2", mode="lines")
    fig.add_vline(x=row.approval_rate, line_dash="dash", annotation_text=f"cutoff {cutoff}")
    fig.update_layout(xaxis=dict(title="approval rate", tickformat=".0%"), yaxis=dict(title="rate", tickformat=".0%"),
                      yaxis2=dict(title="BRL", overlaying="y", side="right"), title=f"{model} policy curve (synthetic)",
                      legend=dict(orientation="h", y=-0.25))
    cB.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- proof of value
st.header("6. Proof of Value: challenger vs incumbent at the same approval rate (OOT)")
pov = load("agg_model_comparison_at_approval_rate")
if is_empty(pov):
    st.info(SCORES_PENDING)
else:
    pov = pov.sort_values(["model_version", "target_approval_rate"])
    mv = sorted(pov["model_version"].unique())
    c1, c2, c3 = st.columns(3)
    default_ch = "baseline_logreg_v1" if "baseline_logreg_v1" in mv else ( "gbm_v1" if "gbm_v1" in mv else mv[-1])
    challenger = c1.selectbox("Challenger", mv, index=mv.index(default_ch))
    others = [m for m in mv if m != challenger] or mv
    default_base = "bureau_score" if "bureau_score" in others else (
        "baseline_logreg_v1" if "baseline_logreg_v1" in others else others[0])
    baseline = c2.selectbox("Baseline (incumbent)", others, index=others.index(default_base))
    rates = sorted(pov["target_approval_rate"].unique())
    target = c3.select_slider("Target approval rate", options=rates, value=0.5 if 0.5 in rates else rates[len(rates) // 2],
                              format_func=lambda r: f"{r:.0%}")
    sel = pov[pov["target_approval_rate"] == target].set_index("model_version")
    if challenger != baseline and {challenger, baseline} <= set(sel.index):
        g, b = sel.loc[challenger], sel.loc[baseline]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Observed default rate — {challenger}", pct(g.observed_default_rate),
                  delta=None if pd.isna(g.observed_default_rate) or pd.isna(b.observed_default_rate)
                  else f"{(g.observed_default_rate - b.observed_default_rate) * 100:+.1f} pp vs {baseline}", delta_color="inverse")
        c2.metric(f"Observed net result per observed loan — {challenger}", brl(g.observed_net_result_per_loan_brl),
                  delta=brl(g.observed_net_result_per_loan_brl - b.observed_net_result_per_loan_brl) + f" vs {baseline}")
        c3.metric(f"Observed outcomes ({challenger} / {baseline})", f"{int(g.n_observed_outcomes)} / {int(b.n_observed_outcomes)}")
    else:
        st.info("Select two different model versions.")
    st.caption("Through-the-door PoV: each policy approves the top-k of **all** OOT-scored applications "
               "by score (same k; for `bureau_score`, higher bureau = lower risk). Observed default and "
               "net result are computed on the booked-and-matured subset of those approvals (reject bias). "
               "Expected net result uses each model's own PD, so it is not defined for the bureau policy. "
               "This is a different population from the published approval-matched rates below, which use "
               "only observable OOT loans. GBM has no demonstrated ranking gain over the bureau on the "
               "published paired bootstrap (delta CIs include 0); the dashboard does not claim otherwise.")
    pov_sel = pov[pov["model_version"].isin([challenger, baseline])]
    fig = px.line(pov_sel, x="target_approval_rate", y="observed_default_rate", color="model_version", markers=True,
                  title="Observed default rate at equal approval rate (synthetic)")
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig, use_container_width=True)
    c2.plotly_chart(px.line(pov_sel, x="target_approval_rate", y="observed_net_result_per_loan_brl", color="model_version", markers=True,
                            title="Observed net result per observed loan (BRL, synthetic assumptions)")
                    .update_layout(xaxis_tickformat=".0%"), use_container_width=True)
    st.dataframe(pov[(pov["target_approval_rate"] == target) & pov["model_version"].isin([challenger, baseline])],
                 use_container_width=True, hide_index=True)

    published = load_published_metrics()
    matched = (published or {}).get("approval_matched") or []
    if matched:
        st.subheader("Published approval-matched bad rates (observable OOT only)")
        n_oot = None
        metrics = (published or {}).get("metrics") or {}
        for block in metrics.values():
            if isinstance(block, dict) and (block.get("oot_main") or {}).get("n"):
                n_oot = block["oot_main"]["n"]
                break
        st.caption(
            "These figures come from `docs/model_metrics.json` and are **not** the through-the-door "
            f"PoV above. Population: observable OOT loans only"
            + (f" (n = {int(n_oot):,})" if n_oot else "")
            + ". Each policy approves the top X% of that observable set. "
            "Individual bad-rate CIs are shown only when the JSON includes them; "
            "paired bootstrap intervals below are for Δ (GBM − bureau)."
        )
        rows = []
        pb_bad = ((published or {}).get("paired_bootstrap") or {}).get("delta_bad_rate") or {}
        for row in matched:
            rate = row.get("approval_rate")
            key = str(rate)
            # JSON keys may be "0.5" or "0.50"
            delta = pb_bad.get(key) or pb_bad.get(f"{float(rate):.1f}") or pb_bad.get(f"{float(rate):g}") or {}
            ci = delta.get("ci") or [None, None]
            rows.append({
                "approval_rate": rate,
                "n_approved": row.get("n_approved"),
                "bad_rate_bureau_score": row.get("bad_rate__bureau_score"),
                "bad_rate_baseline_logreg_v1": row.get("bad_rate__baseline_logreg_v1"),
                "bad_rate_gbm_v1": row.get("bad_rate__gbm_v1"),
                "delta_gbm_minus_bureau_mean": delta.get("mean"),
                "delta_ci_low": None if not ci else ci[0],
                "delta_ci_high": None if not ci else ci[1],
                "bureau_ties_at_cutoff": row.get("ties_at_cutoff__bureau_score"),
            })
        pub_df = pd.DataFrame(rows)
        st.dataframe(pub_df.style.format({
            "approval_rate": "{:.1%}",
            "bad_rate_bureau_score": "{:.2%}",
            "bad_rate_baseline_logreg_v1": "{:.2%}",
            "bad_rate_gbm_v1": "{:.2%}",
            "delta_gbm_minus_bureau_mean": "{:.4%}",
            "delta_ci_low": "{:.4%}",
            "delta_ci_high": "{:.4%}",
        }, na_rep="—"), use_container_width=True, hide_index=True)

st.divider()
st.warning(SYNTHETIC_BANNER)
