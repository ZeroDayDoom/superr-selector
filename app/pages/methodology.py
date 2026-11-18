"""Methodology page: renders the saved methodology documents."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from superr.dashboard import data as D

D.page_config("Methodology", "📚")

st.title("Methodology")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Pressure definition", "Ranking", "Data audit", "Full design doc",
     "Limitations"])

with tab1:
    t = D.report_text("pressure_model_report.md")
    if t:
        st.markdown(t)
    else:
        st.info("Run `scripts/run_pressure.py` to generate the pressure report.")

with tab2:
    t = D.report_text("player_rating_methodology.md")
    if t:
        st.markdown(t)
    else:
        st.info("Run `scripts/run_ratings.py` to generate the rating report.")

with tab3:
    t = D.report_text("data_audit_decisions.md")
    if t:
        st.markdown(t)
    else:
        st.info("Run `python scripts/run_audit.py` to generate the audit.")

with tab4:
    t = D.report_text("methodology_design.md")
    if t:
        # long doc: collapsible sections
        st.markdown(t)

with tab5:
    st.markdown("""
### Pre-registered limitations
- Innings-1 win probability is a total-projection proxy (AUC ≈ 0.66), not a
  true mid-innings match-win probability.
- No ball-tracking or fielding data: expected runs cannot fully separate
  shot quality from delivery quality.
- Clutch persistence in sport is modest; small HP samples remain uncertain
  even after shrinkage — every player carries a bootstrap CI.
- Run-out attribution: a striker is charged only for their own dismissal;
  a sensitivity variant with team wickets exists.
- The two-pass opponent-quality adjustment is an approximation of a full
  hierarchical plus-minus model.
- Pressure is necessarily normative; the multiverse and validation battery
  (Robustness page) quantify — but cannot eliminate — that judgement.
""")

st.divider()
st.subheader("Data & metric glossary")
st.markdown("""
| term | meaning |
|---|---|
| HP balls | deliveries in the top-20% of the pressure index within innings × phase strata |
| Value over expectation | runs/runs-prevented above the contextual model + dynamic-λ wicket term |
| Level LCB | EB-shrunk HP value mean − 0.85 × bootstrap SE |
| Uplift (adjusted) | residual of HP value on normal value across the pool |
| Dynamic λ | run-value of a wicket in a state, from total-projection curves |
| WPA | win probability added by the delivery's actual outcome |
""")
