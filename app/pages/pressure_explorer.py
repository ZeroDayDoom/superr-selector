"""Pressure explorer: interactive match-state simulator.

Uses the SAVED win-probability model and the documented component formulas —
no player ratings are recomputed here, and no player recommendations are
generated (the state → pressure mapping is model machinery, not opinion).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from superr.dashboard import data as D

D.page_config("Pressure Explorer", "🧭")

st.title("Pressure Explorer — match-state simulator")
st.caption("Computes the pre-delivery pressure components for a state you "
           "control, using the saved win-probability model and the exact "
           "documented formulas. No player data is used or recommended.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
wp_path = PROJECT_ROOT / "outputs" / "models" / "wp2_full.joblib"
wp1_path = PROJECT_ROOT / "outputs" / "models" / "wp1_full.joblib"

with st.sidebar:
    st.header("Match state")
    innings = st.radio("Innings", [1, 2], horizontal=True)
    score = st.number_input("Score", 0, 300, 96)
    wickets = st.number_input("Wickets lost", 0, 10, 3)
    balls_remaining = st.number_input("Balls remaining", 0, 130, 24)
    target = st.number_input("Target (innings 2)", 0, 400, 174)
    competition = st.selectbox("Competition", ["IPL", "senior T20I"])
    knockout = st.checkbox("Knockout match")
    recent_wickets = st.slider("Wickets in last 12 balls", 0, 3, 1)
    dot_streak = st.slider("Consecutive dot balls", 0, 24, 6)
    boundary_drought = st.slider("Balls since last boundary", 0, 36, 10)
    revised = st.checkbox("DLS-revised chase")

st.subheader("Derived state")
runs_required = max(0, target - score) if innings == 2 else np.nan
overs_remaining = max(balls_remaining, 1) / 6
rrr = runs_required / overs_remaining if innings == 2 else np.nan
crr = score / max(1e-9, (130 - balls_remaining) / 6)
gap = rrr - crr if innings == 2 else np.nan
resources = max(0.0, balls_remaining / 120) * ((10 - wickets) / 10)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Runs required", f"{runs_required:.0f}" if innings == 2 else "—")
c2.metric("Required RR", f"{rrr:.2f}" if innings == 2 else "—")
c3.metric("Current RR", f"{crr:.2f}")
c4.metric("RRR − CRR", f"{gap:+.2f}" if innings == 2 else "—")
c5.metric("Resources left", f"{resources:.0%}")

# ---- win probability (saved model) -----------------------------------------
def build_state() -> pd.DataFrame:
    sched = 120.0
    return pd.DataFrame({
        "runs_required_before": [runs_required if innings == 2 else 0.0],
        "legal_balls_remaining": [float(balls_remaining)],
        "wickets_before": [float(wickets)],
        "wickets_in_hand": [float(10 - wickets)],
        "required_run_rate": [rrr if innings == 2 else 0.0],
        "current_run_rate": [crr],
        "rrr_crr_gap": [gap if innings == 2 else 0.0],
        "target": [float(target) if innings == 2 else 0.0],
        "score_before": [float(score)],
        "over_number_before": [(130 - balls_remaining) // 6 + 1],
        "ball_in_over_before": [0],
        "venue_season_par": [170.0],
        "is_knockout": [1 if knockout else 0],
        "is_dls_revised": [1 if revised else 0],
        "is_ipl": [1 if competition == "IPL" else 0],
        "resources_remaining": [resources],
    })


wp = None
try:
    if innings == 2 and wp_path.exists():
        wp = joblib.load(wp_path)
    elif innings == 1 and wp1_path.exists():
        wp = joblib.load(wp1_path)
except Exception as e:  # noqa: BLE001
    st.warning(f"Could not load WP model: {e}")

wp_before = None
if wp is not None:
    feats = wp["features"]
    X = build_state()
    for f in feats:
        if f not in X.columns:
            X[f] = 0.0
    wp_before = float(wp["model"].predict_proba(X[feats])[:, 1][0])
    if wp.get("calibrator") is not None:
        wp_before = float(wp["calibrator"].predict([wp_before])[0])
    if innings == 2:
        if runs_required <= 0:
            wp_before = 1.0
        elif wickets >= 10:
            wp_before = 0.0

# ---- components (documented formulas) ---------------------------------------
over_idx = (130 - balls_remaining) // 6 + 1
difficulty = (
    (0.45 * np.clip(rrr / 12, 0, 1) + 0.35 * np.clip(gap / 6, 0, 1) + 0.20
     * np.clip(runs_required / max(6 * overs_remaining, 1), 0, 1))
    if innings == 2 else
    0.6 * np.clip(0 / 30, 0, 1) + 0.4 * (np.clip((8.5 - crr) / 4, 0, 1)
                                         if (130 - balls_remaining) >= 12 else 0)
)
leverage = float(np.clip(wp_before * (1 - wp_before) * 4 / 0.12, 0, 1)) \
    if wp_before is not None else np.nan
closeness = 1 - abs(2 * wp_before - 1) if wp_before is not None else np.nan
resource_stress = 1 - resources
collapse = 0.6 * min(1.0, recent_wickets / 3.0) + 0.4 * min(1.0, recent_wickets / 2.0)
momentum = 0.5 * np.clip(dot_streak / 12, 0, 1) + 0.5 * np.clip(boundary_drought / 18, 0, 1)
phase = 0.3 if over_idx <= 6 else (0.4 if over_idx <= 15 else 1.0)
importance = 1.0 if knockout else 0.2

# Model A (rules) and B (WP) then C (hybrid) — same formulas AND the same
# calibrated weights as the engine (read from the pressure-phase artefact,
# falling back to the calibrated 2025-season values)
_W = (D.reports_json("pressure_validation.json") or {}).get(
    "calibrated_weights", {})
WB = _W.get("model_b", {"match_closeness": 0.4, "leverage_norm": 0.2,
                        "difficulty": 0.4})
WC = _W.get("model_c", {"B": 0.7, "A": 0.2, "resource_stress": 0.1})
A = (0.30 * difficulty + 0.15 * resource_stress + 0.20 * collapse
     + 0.15 * momentum + 0.10 * phase + 0.10 * importance)
B = (WB["match_closeness"] * closeness + WB["leverage_norm"] * leverage
     + WB["difficulty"] * difficulty)
C = (WC["B"] * B + WC["A"] * A + WC["resource_stress"] * resource_stress)
C = float(np.clip(C, 0, 1))

# category from the saved distribution
ps = D.pressure_scores()
if ps is not None:
    q50, q75, q90 = (ps["PRESSURE_MODEL_C"].quantile(q) for q in (0.5, 0.75, 0.9))
    category = ("LOW" if C <= q50 else "MODERATE" if C <= q75
                else "HIGH" if C <= q90 else "EXTREME")
else:
    category = "—"

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Win probability", f"{wp_before:.1%}" if wp_before is not None else "—")
m2.metric("Pressure (Model C)", f"{C:.3f}")
m3.metric("Category", category)
m4.metric("Pressure / 100", f"{C * 100:.1f}")

st.subheader("Components (0–1 each)")
comp_df = pd.DataFrame({
    "component": ["difficulty", "match_closeness", "leverage", "resource_stress",
                  "collapse", "momentum", "phase", "match_importance"],
    "value": [difficulty, closeness, leverage, resource_stress, collapse,
              momentum, phase, importance],
}).set_index("component")
st.bar_chart(comp_df["value"])

# scenario classification (same rules as the engine)
scen = []
if over_idx <= 6 and (wickets >= 2 or recent_wickets >= 1):
    scen.append("EARLY_COLLAPSE_RESCUE")
if 7 <= over_idx <= 15 and wickets >= 4:
    scen.append("MIDDLE_ORDER_RESCUE")
if recent_wickets >= 2:
    scen.append("RAPID_COLLAPSE")
if innings == 2 and rrr >= 10:
    scen.append("HIGH_RRR_CHASE / HIGH_RRR_DEFENCE")
if innings == 2 and (10 - wickets) <= 3:
    scen.append("LOW_WICKET_RESOURCE_CHASE / LOW_WICKET_ATTACK")
if innings == 2 and gap > 2:
    scen.append("ACCELERATION_REQUIRED")
if over_idx >= 16:
    scen.append("DEATH_OVERS_PRESSURE / DEATH_DEFENCE")
if boundary_drought >= 12:
    scen.append("BOUNDARY_DROUGHT")
if dot_streak >= 8 and boundary_drought >= 8:
    scen.append("BOUNDARY_SUPPRESSION")
if balls_remaining <= 6:
    scen.append("FINAL_OVER_PRESSURE / FINAL_OVER_DEFENCE")
if wp_before is not None and wp_before >= 0.4:
    scen.append("HIGH_LEVERAGE")
if wp_before is not None and closeness >= 0.75 and (innings == 2 or over_idx >= 15):
    scen.append("MATCH_ON_LINE")
st.subheader("Active scenarios")
st.write(", ".join(scen) if scen else "None")

st.subheader("Why this score?")
st.markdown(f"""
- **Match closeness {closeness:.2f}**: WP = {wp_before:.1%}; closeness = 1 − |2·WP − 1|.
- **Leverage {leverage:.2f}**: expected |ΔWP| from the outcome distribution at
  this state, normalised by a 0.12 physical maximum.
- **Difficulty {difficulty:.2f}**: chase steepness (RRR, RRR−CRR, boundary
  burden) for innings 2; score-vs-par and rate-vs-par for innings 1.
- **Resource stress {resource_stress:.2f}** = 1 − (balls fraction × wickets fraction).
- **Collapse {collapse:.2f}** and **momentum {momentum:.2f}** come from your
  recent-wickets, dot-streak and boundary-drought inputs.
- **Phase {phase:.2f}** (death overs weight highest) and **importance
  {importance:.2f}** (knockout) are single components among eight — no phase
  automatically forces a high pressure score.
- **Model C = 0.70·B + 0.20·A + 0.10·resource stress**, weights calibrated on
  the 2025 validation season (see Methodology).
""")
