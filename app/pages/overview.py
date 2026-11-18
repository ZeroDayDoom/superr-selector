"""SupeRR Selector — main app entry (Streamlit multipage root).

Run from the project root (PowerShell):
    streamlit run app/main.py
Pages live in app/pages/ and read only saved analytical outputs via
src/superr/dashboard/data.py (single source of truth).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from superr.dashboard import data as D

D.page_config("Overview", "🏏")

st.title("SupeRR Selector — Pressure Analytics War Room")
st.caption("Rajasthan Royals SupeRR Selector Season 2 · IPL & T20I 2022–2026 · "
           "every number on this dashboard comes from the saved pipeline outputs")

status = D.phase_status()
missing = [k for k, v in status.items() if not v]
if missing:
    st.warning("Pipeline artefacts missing for: "
               + ", ".join(missing) + ". See the note at the bottom.")
else:
    st.success("All pipeline artefacts present — data current.")

# ---- KPI row ---------------------------------------------------------------
balls = D.balls_clean()
inn = D.innings_summary()
mm = D.match_summary()
metrics = D.player_metrics()
pressure = D.pressure_scores()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches analysed", f"{len(mm):,}" if mm is not None else "—")
c2.metric("Deliveries analysed", f"{len(balls):,}" if balls is not None else "—")
c3.metric("Players evaluated",
          f"{metrics['player_id'].nunique():,}" if metrics is not None else "—")
c4.metric("Pressure deliveries (top-20%)",
          f"{int(pressure['hp_top20'].sum()):,}" if pressure is not None else "—")

# ---- model + validation summary -------------------------------------------
st.subheader("Models")
vs = D.validation_summary()
mc = D.model_comparison()
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Selected models**")
    st.info("Pressure: hybrid Model C (WP leverage + situational rules) · "
            "Win probability: calibrated logistic (chase AUC ≈ 0.92) · "
            "Expected outcomes: LightGBM buckets — see Methodology page.")
with col2:
    st.markdown("**Validation snapshot**")
    if vs:
        sh_b = vs.get("split_half_batters", {})
        st.write(f"- Split-half Spearman (batters): {sh_b.get('spearman_mean', '—')}")
        tb = vs.get("temporal_batters", {})
        if tb:
            first = list(tb.values())[0]
            st.write(f"- Temporal persistence: Spearman "
                     f"{first.get('spearman')} (rank 2022-23 → 2024-26)")
    else:
        st.write("Run `scripts/run_validation.py` for the full battery.")

# ---- top of the rankings ---------------------------------------------------
if metrics is not None and status["ratings"]:
    b = D.rankings("batters")
    w = D.rankings("bowlers")
    st.subheader("Head of the rankings (moderate eligibility)")
    c1, c2 = st.columns(2)
    with c1:
        if b is not None and len(b):
            st.markdown("**Top 5 batters**")
            st.dataframe(b.head(5)[["rank", "player_name", "rating_primary",
                                    "hp_balls", "level_lcb",
                                    "uplift_adjusted"]], hide_index=True)
        else:
            D.missing_phase_message()
    with c2:
        if w is not None and len(w):
            st.markdown("**Top 5 bowlers**")
            st.dataframe(w.head(5)[["rank", "player_name", "rating_primary",
                                    "hp_balls", "level_lcb",
                                    "uplift_adjusted"]], hide_index=True)
        else:
            D.missing_phase_message()

# ---- pressure distribution -------------------------------------------------
if pressure is not None:
    import plotly.express as px
    st.subheader("Pressure distribution")
    fig = px.histogram(pressure, x="PRESSURE_MODEL_C", nbins=60,
                       color_discrete_sequence=["#0b1c3d"])
    fig.add_vline(x=pressure["PRESSURE_MODEL_C"].quantile(0.8),
                  line_color="#c8102e",
                  annotation_text="HP threshold (top 20%)")
    fig.update_layout(height=320, xaxis_title="Pressure score (Model C)",
                      yaxis_title="Deliveries", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # batting vs bowling HP vs normal value
    st.subheader("Pressure vs normal performance")
    tab_b, tab_w = st.tabs(["Batters", "Bowlers"])
    with tab_b:
        m = metrics[metrics["role"] == "batter"] if metrics is not None else None
        if m is not None:
            fig = px.histogram(m, x="hp_value_mean", nbins=50,
                               color_discrete_sequence=["#c8102e"],
                               title="HP value-over-expectation mean (batters)")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with tab_w:
        m = metrics[metrics["role"] == "bowler"] if metrics is not None else None
        if m is not None:
            fig = px.histogram(m, x="hp_value_mean", nbins=50,
                               color_discrete_sequence=["#0b1c3d"],
                               title="HP value-over-expectation mean (bowlers)")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
else:
    D.missing_phase_message()

# ---- ranking uncertainty ---------------------------------------------------
boot = D.top10_probability()
if boot is not None:
    st.subheader("Ranking uncertainty (1000-draw match bootstrap)")
    fig = px.bar(boot.head(15), x="p_top10",
                 y="player_name", orientation="h",
                 color_discrete_sequence=["#e8b84b"])
    fig.update_layout(height=420, yaxis=dict(autorange="reversed"),
                      xaxis_title="P(Top 10)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Bootstrap uncertainty appears after Phase 8 "
            "(`python scripts/run_validation.py`).")

st.divider()
st.caption("Pages: Batter rankings · Bowler rankings · Player profile · "
           "Player comparison · Pressure explorer · Match case studies · "
           "Methodology · Robustness — use the sidebar.")
if missing:
    st.error("Missing artefacts: " + ", ".join(missing))
