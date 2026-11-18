"""Match case studies: gallery of real high-pressure situations."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st

from superr.dashboard import data as D

D.page_config("Match Case Studies", "📋")

st.title("Match Case Studies — real high-pressure situations")
st.caption("Selected by model output (highest-pressure deliveries), showing "
           "both positive and negative outcomes. All values from saved "
           "pipeline outputs.")

evid = D.player_match_evidence()
if evid is None:
    D.missing_phase_message()
    st.stop()

mm = D.match_summary()
if mm is not None:
    evid = evid.merge(mm[["match_id", "match_date", "home_team",
                          "away_team", "venue_country"]], on="match_id",
                      how="left")

st.subheader("Filters")
c1, c2, c3 = st.columns(3)
kind = c1.radio("Case type", ["Most extreme", "Biggest positive WPA",
                              "Biggest negative WPA"], horizontal=False)
min_pressure = c2.slider("Min pressure score", 0.0, 1.0, 0.5, 0.05)
n_cases = c3.slider("Number of cases", 5, 20, 8)

cases = evid[evid["PRESSURE_MODEL_C"] >= min_pressure].copy()
if kind == "Most extreme":
    cases = cases.nlargest(n_cases, "PRESSURE_MODEL_C")
elif kind == "Biggest positive WPA":
    cases = cases.nlargest(n_cases, "realized_wpa")
else:
    cases = cases.nsmallest(n_cases, "realized_wpa")

if len(cases) == 0:
    st.warning("No deliveries match the filters.")
    st.stop()

for _, r in cases.iterrows():
    with st.expander(f"Match {int(r['match_id'])} · {r.get('match_date','')} · "
                     f"{r.get('home_team','')} v {r.get('away_team','')} · "
                     f"innings {int(r['innings'])}, over {int(r['overid'])}"
                     f" — pressure {r['PRESSURE_MODEL_C']:.3f}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score before", f"{r['score_before']:.0f}/{int(r['wickets_before'])}")
        c2.metric("Balls left", int(r["legal_balls_remaining"]))
        c3.metric("Pressure", f"{r['PRESSURE_MODEL_C']:.3f}")
        c4.metric("Realized WPA", f"{r['realized_wpa']:+.3f}")
        line1 = (f"- **Situation:** {r['score_before']:.0f}/{int(r['wickets_before'])}"
                 f" after {(r['legal_balls_before'] // 6) + 1:.0f}-over mark; "
                 f"{int(r['legal_balls_remaining'])} balls left")
        if pd.notna(r.get("target")) and r["innings"] == 2:
            rrr = (r["target"] - r["score_before"]) / max(r["legal_balls_remaining"] / 6, 1/6)
            line1 += f"; target {r['target']:.0f}, RRR {rrr:.2f}"
        st.markdown(line1)
        st.markdown(
            f"- **Expected (model):** xRuns {r['expected_runs_off_bat']:.2f}, "
            f"xWicket {r['p_wicket']:.1%}, xConceded {r['expected_bowler_conceded_runs']:.2f}")
        st.markdown(
            f"- **Actual:** {r['batter_runs']:.0f} off the bat"
            f"{', WICKET' if r['is_team_wicket'] else ''}"
            f"{' (bowler wicket)' if r['is_bowler_wicket'] else ' (non-bowler)' if r['is_team_wicket'] else ''};"
            f" bowler conceded {r['bowler_runs_conceded']:.0f}")
        batter_name = r.get("striker_name", "—")
        bowler_name = r.get("bowler_name", "—")
        st.markdown(f"- **Player contribution:** batter {batter_name} "
                    f"(runs above expectation "
                    f"{r['batter_runs'] - r['expected_runs_off_bat']:+.2f}); "
                    f"bowler {bowler_name} (runs prevented "
                    f"{r['expected_bowler_conceded_runs'] - r['bowler_runs_conceded']:+.2f})")
        st.markdown(f"- **Outcome:** {r.get('result','')}")

st.caption("Expected values come from the LightGBM bucket models "
           "(Phase 6); WPA from the calibrated win-probability model "
           "(Phase 5). Positive and negative examples are both shown.")
