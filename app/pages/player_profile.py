"""Player profile page: comprehensive per-player evidence."""

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
D.page_config("Player Profile", "🔎")

metrics = D.player_metrics()
if metrics is None:
    D.missing_phase_message()
    st.stop()

st.title("Player Profile")
names = metrics["player_name"].dropna().unique()
query = st.selectbox("Player", sorted(names))
role_sel = st.radio("Role view", ["batter", "bowler"], horizontal=True)

p = metrics[(metrics["player_name"] == query)
            & (metrics["role"] == role_sel)]
if len(p) == 0:
    st.warning("No rating for this player in this role (may be ineligible).")
    st.stop()
r = p.iloc[0]
pid = r["player_id"]

boot = D.bootstrap()
br = boot[boot["player_id"] == pid] if boot is not None else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Composite rating", f"{r['rating_primary']:.3f}")
c2.metric("HP balls", f"{r['hp_balls']:.0f}")
c3.metric("HP matches", f"{r['hp_matches']:.0f}")
c4.metric("Level LCB", f"{r['level_lcb']:.3f}")
c5.metric("Positive matches", f"{r['positive_match_pct']:.0%}")

st.divider()
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Performance", "Situations", "Consistency & risk", "Trends", "Match evidence"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("HP value mean", f"{r['hp_value_mean']:.3f}")
    c2.metric("Normal baseline", f"{r['normal_mean']:.3f}")
    c3.metric("Uplift (adj.)", f"{r['uplift_adjusted']:.3f}")
    if role_sel == "batter":
        c1.metric("HP strike rate", f"{r.get('hp_strike_rate', np.nan):.1f}")
        c2.metric("HP average", f"{r.get('hp_average', np.nan):.1f}")
        c3.metric("Runs above expected (HP)",
                  f"{r.get('hp_runs_above_expected', np.nan):.1f}")
    else:
        c1.metric("HP economy", f"{r.get('hp_economy', np.nan):.2f}")
        c2.metric("HP wicket rate", f"{r.get('hp_wicket_rate', np.nan):.3f}")
        c3.metric("HP dot %", f"{r.get('hp_dot_pct', np.nan):.1f}")
    if br is not None and len(br):
        row = br.iloc[0]
        st.write(f"**Bootstrap (1000 draws):** rank P05 {row['rank_p05']:.0f} · "
                 f"P50 {row['rank_p50']:.0f} · P95 {row['rank_p95']:.0f} · "
                 f"P(top10) {row['p_top10']:.0%} · "
                 f"value CI [{row['boot_ci_lo']:.3f}, {row['boot_ci_hi']:.3f}]")

with tab2:
    scen_dir = D.dossiers_dir() if False else None
    from superr.ratings.rankings import scenario_rankings as _  # noqa: F401
    evid = D.player_match_evidence()
    if evid is not None:
        idc, nnc, vcol = (("striker_id", "striker_name", "bat_value")
                          if role_sel == "batter" else
                          ("bowler_id", "bowler_name", "bowl_value"))
        mine = evid[evid[idc] == pid]
        scen_cols = [c for c in mine.columns if c.startswith("scenario_")
                     and c not in ("scenario_count", "scenario_diversity_index")]
        if len(scen_cols) and len(mine):
            s = {c.replace("scenario_", ""): float(mine[mine[c].astype(bool)][vcol].mean())
                 for c in scen_cols
                 if mine[c].astype(bool).sum() >= 20}
            if s:
                srt = sorted(s.items(), key=lambda kv: -kv[1])
                c1, c2 = st.columns(2)
                c1.markdown("**Strengths (best scenarios)**")
                for k, v in srt[:4]:
                    c1.write(f"- {k}: {v:+.3f}")
                c2.markdown("**Weaknesses (worst scenarios)**")
                for k, v in reversed(srt[-4:]):
                    c2.write(f"- {k}: {v:+.3f}")
                fig, ax = plt.subplots(figsize=(7, 0.4 * len(s) + 1.5))
                keys = [k for k, _ in srt]
                ax.barh(keys, [s[k] for k in keys],
                        color=["#c8102e" if s[k] < 0 else "#0b1c3d" for k in keys])
                ax.axvline(0, color="grey", lw=0.8)
                ax.set_title("Value over expectation by scenario")
                st.pyplot(fig, use_container_width=True)
        # phase + competition split
        c1, c2 = st.columns(2)
        ph = mine.groupby("innings_phase")[vcol].mean()
        c1.bar_chart(ph)
        c1.caption("By innings phase")
        cs = mine.groupby("competition_class")[vcol].mean()
        c2.bar_chart(cs)
        c2.caption("By competition")

with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Match sd", f"{r.get('match_std', np.nan):.3f}")
    c2.metric("CVaR5 (worst 5% matches)",
              f"{r.get('cvar5', np.nan):.3f}")
    c3.metric("Negative-match rate",
              f"{r.get('negative_impact_rate', np.nan):.0%}")
    st.caption("Worst / best pressure matches are listed under Match evidence.")

with tab4:
    evid = D.player_match_evidence()
    if evid is not None:
        idc, vcol = (("striker_id", "bat_value") if role_sel == "batter"
                     else ("bowler_id", "bowl_value"))
        mine = evid[evid[idc] == pid]
        seas = mine.merge(
            D.balls_clean()[["match_id"]].drop_duplicates(), on="match_id"
        ) if False else mine
        # season from match_summary
        mm = D.match_summary()
        if mm is not None:
            mine2 = mine.copy()
            mine2["season"] = pd.to_datetime(mine2["match_date"]).dt.year
            st.bar_chart(mine2.groupby("season")[vcol].mean())
            st.caption("HP value mean by season")
    doss = [p for p in D.dossiers_dir() if f"_{pid}.md" in p.name]
    if doss:
        st.info(doss[0].read_text(encoding="utf-8"))

with tab5:
    evid = D.player_match_evidence()
    mm = D.match_summary()
    if evid is not None and mm is not None:
        idc, nnc, vcol = (("striker_id", "striker_name", "bat_value")
                          if role_sel == "batter" else
                          ("bowler_id", "bowler_name", "bowl_value"))
        mine = evid[evid[idc] == pid]
        mv = (mine.groupby(["match_id"])
              .agg(value=(vcol, "mean"), balls=(vcol, "size"),
                   runs=("batter_runs", "sum"),
                   actual_value_sum=(vcol, "sum"),
                   xruns=("expected_runs_off_bat", "sum"),
                   pressure=("PRESSURE_MODEL_C", "mean"),
                   wpa=("realized_wpa", "sum"))
              .reset_index()
              .merge(mm[["match_id", "match_date", "competition_class",
                         "home_team", "away_team", "venue_country"]],
                     on="match_id", how="left"))
        mv["above_expectation"] = mv["actual_value_sum"]
        mv = mv.sort_values("value", ascending=False)
        st.dataframe(
            mv[["match_id", "match_date", "competition_class",
                "home_team", "away_team", "venue_country", "balls", "runs",
                "xruns", "above_expectation", "pressure", "wpa", "value"]]
            .style.format(precision=3, na_rep="—"),
            use_container_width=True, height=420)
        st.download_button("Download match evidence (CSV)",
                           mv.to_csv(index=False).encode("utf-8"),
                           f"match_evidence_{pid}.csv", "text/csv")
    else:
        D.missing_phase_message()

# role comparison
st.divider()
st.subheader("Role comparison")
role_col = "batter_role" if role_sel == "batter" else "bowler_role"
peers = metrics[(metrics["role"] == role_sel)
                & (metrics[role_col] == r[role_col])]
st.write(f"{r['player_name']} ranks "
         f"#{int((peers['rating_primary'] > r['rating_primary']).sum()) + 1} "
         f"of {len(peers)} {r[role_col]}s in the composite rating.")
st.caption("Opponent context: opponent team ids faced are part of the "
           "expectation models; per-opponent splits live in "
           "all_player_metrics.csv (n_opponents) and the dossier files.")
