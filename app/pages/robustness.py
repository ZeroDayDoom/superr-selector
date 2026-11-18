"""Robustness page: alternative rankings, uncertainty, model comparison."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st

from superr.dashboard import data as D

D.page_config("Robustness", "🛡️")

st.title("Robustness — does the ranking survive alternative specifications?")
vs = D.validation_summary()
if vs is None:
    D.missing_phase_message()
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Rank stability", "Bootstrap uncertainty", "IPL vs T20I",
     "Scenario rankings", "Model comparison"])

with tab1:
    st.subheader("Rank stability under alternative specifications")
    rs = D.rank_stability()
    if rs is not None:
        st.dataframe(rs.style.format(precision=3), hide_index=True,
                     use_container_width=True)
        st.caption("Spearman correlation of each variant's ranking with the "
                   "primary ranking; top-10 overlap in players.")
    for key, label in [("threshold_stability", "HP threshold (top 10–30%)"),
                       ("pressure_model_stability", "Pressure model (A/B/C)"),
                       ("eligibility_stability_batters", "Eligibility tiers"),
                       ("shrinkage_stability_batters", "Shrinkage strength"),
                       ("weight_stability_batters", "Composite weights"),
                       ("ablation_batters", "Ablations")]:
        rows = vs.get(key, [])
        if rows:
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

with tab2:
    st.subheader("Bootstrap uncertainty (1000 match-level draws)")
    boot = D.bootstrap()
    top10p = D.top10_probability()
    if top10p is not None:
        st.dataframe(top10p.style.format(precision=3, na_rep="—"),
                     use_container_width=True, height=420)
        st.download_button("Download top-10 probabilities",
                           top10p.to_csv(index=False).encode("utf-8"),
                           "top10_probability.csv", "text/csv")
    elif boot is not None:
        st.dataframe(boot, use_container_width=True)
    else:
        st.info("Run `scripts/run_validation.py`.")
    if boot is not None:
        st.markdown("**Most robust** (narrowest P05–P95 rank interval)")
        b = boot.dropna(subset=["rank_p95"]).nsmallest(10, "rank_p95")
        st.dataframe(b[["player_name", "rank_p05", "rank_p50", "rank_p95",
                        "p_top10"]], hide_index=True)
        st.markdown("**Most uncertain** (widest intervals)")
        u = boot.dropna(subset=["rank_p95"]).nlargest(10, "rank_p95")
        st.dataframe(u[["player_name", "rank_p05", "rank_p50", "rank_p95",
                        "p_top10"]], hide_index=True)

with tab3:
    st.subheader("IPL vs T20I")
    st.json(vs.get("competition_overlap_batters", {}))
    st.json(vs.get("competition_balance", {}))
    st.info(vs.get("associate_team_check", {}).get("note", ""))

with tab4:
    st.subheader("Scenario-specific rankings")
    files = D.scenario_rankings_dir()
    if files:
        choice = st.selectbox("Scenario", [f.stem for f in files])
        f = next(p for p in files if p.stem == choice)
        st.dataframe(pd.read_csv(f), use_container_width=True, height=420)
    else:
        st.info("Run `scripts/run_validation.py` to generate scenario tables.")

with tab5:
    st.subheader("Expected-performance model comparison")
    mc = D.model_comparison()
    if mc is not None:
        st.dataframe(mc.style.format(precision=4, na_rep="—"),
                     use_container_width=True, height=420)
        st.caption("Selection rule: lowest validation log loss per task; "
                   "deep models retained only if they win — see the "
                   "model training report.")
    else:
        st.info("Run `scripts/run_models.py`.")
