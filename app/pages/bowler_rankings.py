"""Bowler rankings page."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from superr.dashboard import data as D

D.page_config("Bowler Rankings", "🎯")

st.title("Bowler Rankings — Pressure Reliability")
st.caption("Same composite as batters (see Methodology). Bowler value = "
           "runs prevented + dynamic-λ·(wickets above expected). Run-outs and "
           "non-bowler dismissals are never credited. Eligibility (moderate): "
           "≥60 HP balls AND ≥8 HP matches.")

metrics = D.player_metrics()
if metrics is None:
    D.missing_phase_message()
    st.stop()

w = metrics[metrics["role"] == "bowler"].copy()
boot = D.bootstrap()
elig = D.eligibility()
if boot is not None:
    w = w.merge(boot[["player_id", "rank_p05", "rank_p50", "rank_p95",
                      "p_top10", "p_top20"]], on="player_id", how="left")
if elig is not None:
    w = w.merge(elig[["player_id", "eligible_strict", "eligible_moderate",
                      "eligible_inclusive"]], on="player_id", how="left")

st.subheader("Top 10 bowlers (moderate eligibility)")
elig_ids = set(elig[(elig["role"] == "bowler") & elig["eligible_moderate"]]["player_id"]) \
    if elig is not None else set()
head = w[w["player_id"].isin(elig_ids)].nsmallest(10, "rank")
cols = ["rank", "player_name", "player_id", "rating_primary", "hp_balls",
        "hp_economy", "hp_wicket_rate", "hp_dot_pct", "hp_value_mean",
        "uplift_adjusted", "positive_match_pct", "cvar5", "p_top10",
        "rank_p05", "rank_p95", "bowler_role"]
st.dataframe(head[[c for c in cols if c in head.columns]]
             .style.format(precision=3, na_rep="—"),
             use_container_width=True, hide_index=True)
st.download_button("Download Top 10 bowlers (CSV)",
                   head.to_csv(index=False).encode("utf-8"),
                   "top10_bowlers.csv", "text/csv")

st.divider()
st.subheader("Full ranking — search, filter, sort")
c1, c2, c3 = st.columns([2, 1, 1])
query = c1.text_input("Search player", key="bq")
role_f = c2.multiselect("Bowling role",
                        sorted(w["bowler_role"].dropna().unique()))
min_balls = c3.number_input("Min HP balls", 0, 2000, 60, step=10, key="bmb")
only_eligible = c1.checkbox("Only moderate-eligible", value=True, key="be")

f = w.copy()
if query:
    f = f[f["player_name"].str.contains(query, case=False, na=False)]
if role_f:
    f = f[f["bowler_role"].isin(role_f)]
f = f[f["hp_balls"] >= min_balls]
if only_eligible and elig is not None:
    f = f[f["eligible_moderate"] == True]  # noqa: E712
f = f.sort_values("rating_primary", ascending=False).reset_index(drop=True)
if "rank" in f.columns:
    f = f.drop(columns=["rank"])
f.insert(0, "rank", np.arange(1, len(f) + 1))

show_cols = [c for c in [
    "rank", "player_name", "player_id", "bowler_role", "rating_primary",
    "level_lcb", "hp_value_mean", "hp_balls", "hp_matches", "hp_economy",
    "hp_wicket_rate", "hp_dot_pct", "hp_boundary_pct", "uplift_raw",
    "uplift_adjusted", "normal_mean", "positive_match_pct", "match_std",
    "cvar5", "p_top10", "rank_p05", "rank_p50", "rank_p95", "n_seasons",
] if c in f.columns]
st.dataframe(f[show_cols].style.format(precision=3, na_rep="—"),
            use_container_width=True, height=520)
st.download_button("Download full bowler ranking (CSV)",
                   f.to_csv(index=False).encode("utf-8"),
                   "rankings_bowlers_full.csv", "text/csv")
st.caption(f"{len(f)} bowlers shown · full eligibility audit in "
           "outputs/eligibility_audit.csv.")
