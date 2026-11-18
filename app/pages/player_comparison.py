"""Player comparison page: 2-4 players, radar + tables."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from superr.dashboard import data as D

matplotlib.use("Agg")
D.page_config("Player Comparison", "⚖️")

metrics = D.player_metrics()
if metrics is None:
    D.missing_phase_message()
    st.stop()

st.title("Player Comparison")
role_sel = st.radio("Role", ["batter", "bowler"], horizontal=True)
m = metrics[metrics["role"] == role_sel]
names = sorted(m["player_name"].dropna().unique())
picked = st.multiselect("Pick 2–4 players", names, default=names[:2])

if len(picked) < 2:
    st.info("Select at least two players.")
    st.stop()

sub = m[m["player_name"].isin(picked)].set_index("player_name")

AXES = [
    ("Level under pressure (LCB)", "level_lcb"),
    ("Uplift (context-adj.)", "uplift_adjusted"),
    ("Consistency (pos. match %)", "positive_match_pct"),
    ("Reliability (HP balls)", "hp_balls"),
    ("Downside control (CVaR5)", "cvar5"),
    ("P(top 10)", "p_top10"),
]

# normalise each axis to 0-100 by the POOL percentile (clearly stated)
norm = {}
for label, col in AXES:
    if col not in m.columns:
        continue
    s = m[col].astype(float)
    norm[label] = s.rank(pct=True)  # 0-1 percentile within role pool

boot = D.bootstrap()
if boot is not None:
    pct_top10 = (boot.drop_duplicates("player_id")
                 .set_index("player_id")["p_top10"])
    norm["P(top 10)"] = m["player_id"].map(pct_top10).rank(pct=True)

fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
angles = np.linspace(0, 2 * np.pi, len(norm), endpoint=False).tolist()
angles += angles[:1]
colors = ["#0b1c3d", "#c8102e", "#e8b84b", "#3f7d3f"]
for i, pname in enumerate(sub.index):
    vals = [float(norm[label].reindex([pname]).iloc[0]) * 100
            for label in norm]
    vals += vals[:1]
    ax.plot(angles, vals, color=colors[i % 4], lw=2,
            label=f"{pname} (pool percentile)")
    ax.fill(angles, vals, color=colors[i % 4], alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([l.split(" (")[0] for l in norm.keys()], size=8)
ax.set_ylim(0, 100)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25th", "50th", "75th", "100th pct"], size=7)
ax.set_title("Percentile profile within the role pool\n(50 = median player; "
             "radial scale is percentile, NOT raw value)", size=9, pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
st.pyplot(fig, use_container_width=True)

st.caption("The radar uses within-pool PERCENTILES (0–100) so different raw "
           "scales are comparable; exact raw values are in the table below.")

st.subheader("Raw values")
tbl = sub[[c for c in ["rating_primary", "level_lcb", "hp_value_mean",
                       "normal_mean", "uplift_raw", "uplift_adjusted",
                       "hp_balls", "hp_matches", "positive_match_pct",
                       "match_std", "cvar5", "negative_impact_rate",
                       "n_seasons", "n_opponents", "scenario_diversity",
                       "batter_role" if role_sel == "batter" else "bowler_role"]
           if c in sub.columns]].T
st.dataframe(tbl.style.format(precision=3, na_rep="—"), height=420)

st.subheader("Season performance (HP value mean)")
evid = D.player_match_evidence()
mm = D.match_summary()
if evid is not None and mm is not None:
    idc = "striker_id" if role_sel == "batter" else "bowler_id"
    pid_map = m.set_index("player_name")["player_id"]
    rows = []
    for pname in sub.index:
        pid = pid_map.get(pname)
        mine = evid[evid[idc] == pid].copy()
        mine["season"] = pd.to_datetime(mine["match_date"]).dt.year
        vcol = "bat_value" if role_sel == "batter" else "bowl_value"
        s = mine.groupby("season")[vcol].mean()
        for season, v in s.items():
            rows.append({"player": pname, "season": season, "hp_value": v})
    if rows:
        sdf = pd.DataFrame(rows)
        fig2 = plt.figure(figsize=(7, 3))
        for pname in sub.index:
            d = sdf[sdf["player"] == pname]
            plt.plot(d["season"], d["hp_value"], marker="o",
                     label=pname)
        plt.legend(fontsize=7)
        plt.ylabel("HP value mean")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.info("No match evidence available.")
else:
    D.missing_phase_message()
