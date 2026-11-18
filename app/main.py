"""SupeRR Selector — app entry point.

Streamlit multipage root. The Overview page lives in app/pages/overview.py
alongside the other pages; Streamlit lists this file as the home page.

Run from the project root (PowerShell):
    streamlit run app/main.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from superr.dashboard import data as D

D.page_config("Home", "🏏")

st.title("SupeRR Selector — Pressure Analytics War Room")
st.caption("Rajasthan Royals SupeRR Selector Season 2 · IPL & T20I 2022–2026")

status = D.phase_status()
missing = [k for k, v in status.items() if not v]
if missing:
    st.warning("Missing pipeline artefacts: " + ", ".join(missing) +
               ". Build them with the commands below.")
else:
    st.success("All pipeline artefacts present — dashboard reads live outputs.")

st.subheader("Start here")
st.markdown("""
- **Overview** — coverage, model status, pressure distribution, top-10 reliability
- **Batter / Bowler rankings** — full sortable tables with eligibility and uncertainty
- **Player profile** — per-player evidence dossier with match-by-match breakdown
- **Player comparison** — radar + raw values across 2–4 players
- **Pressure explorer** — interactive match-state simulator
- **Match case studies** — real extreme-pressure deliveries, positive and negative
- **Methodology** — pressure definition, rating formula, audit, limitations
- **Robustness** — bootstrap uncertainty, stability under alternative specifications
""")

st.subheader("Rebuild the pipeline (PowerShell, project root)")
st.code("""
python scripts/run_audit.py        # Phase 1: data audit
python scripts/run_prepare.py      # Phase 3: cleaning + match states
python scripts/run_features.py     # Phase 4: features + leakage audit
python scripts/run_pressure.py     # Phase 5: WP + pressure engine
python scripts/run_models.py       # Phase 6: expected performance models
python scripts/run_ratings.py      # Phase 7: ratings + rankings + dossiers
python scripts/run_validation.py   # Phase 8: validation battery
streamlit run app/main.py          # Phase 9: this dashboard
""", language="powershell")
