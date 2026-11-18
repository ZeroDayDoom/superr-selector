"""Single source of truth for the dashboard: reads SAVED analytical outputs.

No rating or model is ever recomputed inside the UI. Every loader returns
None (plus a flag) when an upstream artefact is missing so pages can render
a clear "run phase X first" message instead of crashing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
REPORTS = PROJECT_ROOT / "reports"

CSS = """
<style>
:root { --navy:#0b1c3d; --pink:#c8102e; --gold:#e8b84b; }
h1 { color: var(--navy); letter-spacing: .5px; }
h2, h3 { color: #12325e; }
div[data-testid="stMetric"] {
    background: #f5f7fb; border-radius: 10px; padding: 12px 16px;
    border-left: 4px solid var(--pink);
}
div[data-testid="stMetricLabel"] p { font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #eef1f7; border-radius: 8px 8px 0 0; padding: 8px 14px;
}
.stTabs [aria-selected="true"] { background-color: var(--navy); color: white; }
.dataframe th { background-color: var(--navy); color: white; }
</style>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_config(title: str, icon: str = "🏏") -> None:
    st.set_page_config(page_title=f"SupeRR Selector — {title}",
                       page_icon=icon, layout="wide")
    apply_style()


def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read {path.name}: {e}")
        return None


@lru_cache(maxsize=1)
def player_metrics() -> Optional[pd.DataFrame]:
    for p in (OUTPUTS / "all_player_metrics.csv",
              OUTPUTS / "all_player_metrics.parquet"):
        if p.exists():
            return pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    return None


@lru_cache(maxsize=1)
def rankings(role: str) -> Optional[pd.DataFrame]:
    p = OUTPUTS / f"rankings_{role}.csv"
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def eligibility() -> Optional[pd.DataFrame]:
    p = OUTPUTS / "eligibility_audit.csv"
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def bootstrap() -> Optional[pd.DataFrame]:
    p = OUTPUTS / "bootstrap_rankings.csv"
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def top10_probability() -> Optional[pd.DataFrame]:
    p = OUTPUTS / "top10_probability.csv"
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def rank_stability() -> Optional[pd.DataFrame]:
    p = OUTPUTS / "rank_stability.csv"
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def pressure_scores() -> Optional[pd.DataFrame]:
    return _read_parquet(DATA / "pressure_scores.parquet")


@lru_cache(maxsize=1)
def expected_performance() -> Optional[pd.DataFrame]:
    return _read_parquet(DATA / "expected_performance.parquet")


@lru_cache(maxsize=1)
def balls_clean() -> Optional[pd.DataFrame]:
    return _read_parquet(DATA / "balls_clean.parquet")


@lru_cache(maxsize=1)
def innings_summary() -> Optional[pd.DataFrame]:
    return _read_parquet(DATA / "innings_summary.parquet")


@lru_cache(maxsize=1)
def match_summary() -> Optional[pd.DataFrame]:
    return _read_parquet(DATA / "match_summary.parquet")


@lru_cache(maxsize=1)
def scenario_rankings_dir() -> list[Path]:
    d = OUTPUTS / "scenario_rankings"
    return sorted(d.glob("*.csv")) if d.exists() else []


@lru_cache(maxsize=1)
def validation_summary() -> Optional[dict]:
    p = REPORTS / "validation_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def reports_json(name: str) -> Optional[dict]:
    """Any JSON artefact from reports/ (e.g. pressure_validation.json)."""
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def model_comparison() -> Optional[pd.DataFrame]:
    p = REPORTS / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def dossiers_dir() -> list[Path]:
    d = OUTPUTS / "player_dossiers"
    return sorted(d.glob("*.md")) if d.exists() else []


@lru_cache(maxsize=1)
def report_text(name: str) -> Optional[str]:
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def phase_status() -> dict:
    """Which pipeline phases have produced their artefacts."""
    return {
        "data_prepared": (DATA / "balls_clean.parquet").exists(),
        "pressure_scored": (DATA / "pressure_scores.parquet").exists(),
        "expected_performance": (DATA / "expected_performance.parquet").exists(),
        "ratings": (OUTPUTS / "all_player_metrics.csv").exists(),
        "validation": (REPORTS / "validation_summary.json").exists(),
    }


def missing_phase_message() -> None:
    st.warning(
        "This page needs pipeline outputs that are not built yet. "
        "Run from the project root (PowerShell):\n\n"
        "`python scripts/run_prepare.py` → `python scripts/run_features.py` → "
        "`python scripts/run_pressure.py` → `python scripts/run_models.py` → "
        "`python scripts/run_ratings.py` → `python scripts/run_validation.py`"
    )


# ---- formatting helpers ----------------------------------------------------

def fmt(x, nd: int = 3, pct: bool = False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.1%}" if pct else f"{x:.{nd}f}"


@lru_cache(maxsize=1)
def player_match_evidence() -> Optional[pd.DataFrame]:
    """Match-level evidence per player: joins saved delivery outputs.

    Columns: player role-keyed rows (striker for batting, bowler for bowling)
    with per-delivery value contributions aggregated per (player, match).
    """
    b = balls_clean()
    e = expected_performance()
    p = pressure_scores()
    if b is None or e is None or p is None:
        return None
    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
    df = b.merge(e.drop(columns=["striker_id", "non_striker_id", "bowler_id",
                                 "competition_class", "innings_phase"],
                         errors="ignore"), on=key, how="left",
                 validate="one_to_one")
    df = df.merge(p[key + ["PRESSURE_MODEL_C", "hp_top20", "realized_wpa"]],
                  on=key, how="left", validate="one_to_one")
    # match metadata (date/season/competition) from the match summary
    mm = match_summary()
    if mm is not None:
        df = df.merge(mm[["match_id", "match_date", "competition_class"]],
                      on="match_id", how="left", validate="many_to_one",
                      suffixes=("", "_mm"))
    if "competition_class" not in df.columns:
        df["competition_class"] = df.get("competition_class_mm")
    if "match_date" not in df.columns:
        df["match_date"] = df.get("match_date_mm")
    df["season"] = pd.to_datetime(df["match_date"]).dt.year
    if "innings_phase" not in df.columns:
        over_idx = (df["legal_balls_before"] // 6 + 1).astype(int)
        df["innings_phase"] = np.select([over_idx <= 6, over_idx <= 15],
                                        ["powerplay", "middle"], default="death")
    # mean dynamic wicket value from the ratings-phase summary (falls back
    # to 1.2 only when Phase 7 has not produced its summary yet)
    rs = reports_json("ratings_summary.json") or {}
    lam_mean = float(rs.get("mean_lambda_runs", 1.2))
    if "lambda_state" not in df.columns:
        df["lambda_state"] = lam_mean  # display-level approximation in UI only
    df["bat_value"] = (df["batter_runs"] - df["expected_runs_off_bat"]) \
        - df["lambda_state"] * (
        df["dismissed_is_striker"].astype("boolean").fillna(False).astype(float)
        - df["p_wicket"])
    df["bowl_value"] = (df["expected_bowler_conceded_runs"]
                        - df["bowler_runs_conceded"])
    return df
