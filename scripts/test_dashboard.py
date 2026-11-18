"""Dashboard smoke test: execute every page via Streamlit AppTest."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streamlit.testing.v1 import AppTest

PAGES = [
    "app/main.py",
    "app/pages/overview.py",
    "app/pages/batter_rankings.py",
    "app/pages/bowler_rankings.py",
    "app/pages/player_profile.py",
    "app/pages/player_comparison.py",
    "app/pages/pressure_explorer.py",
    "app/pages/match_case_studies.py",
    "app/pages/methodology.py",
    "app/pages/robustness.py",
]

failed = []
for page in PAGES:
    at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=120)
    at.run()
    if at.exception:
        failed.append((page, str(at.exception)))
        print(f"FAIL {page}: {at.exception}")
    else:
        print(f"OK   {page}")

if failed:
    raise SystemExit(1)
print("ALL PAGES EXECUTE CLEANLY")
