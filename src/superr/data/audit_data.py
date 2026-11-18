"""Reproducible data audit for the SupeRR Selector Season 2 data pack.

Every check is data-driven: nothing about the semantics of the files is
assumed; candidate interpretations are verified against the data itself
(e.g. delivery run totals are validated against parsed match results).

Outputs (written to ``<project>/reports``):
    data_inventory.md
    schema_dictionary.csv
    data_quality_report.md
    data_quality_checks.json
    competition_audit.csv
    match_exclusion_audit.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from superr.config import load_config

# Raw files, by role. Kept here (not in config) because the audit is the place
# where the existence and identity of the files is established in the first
# place. Downstream modules import RAW_FILES from this module.
RAW_FILES = {
    "ball_by_ball": "ball_by_ball.csv",
    "matches": "matches.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "playing_eleven": "playing_eleven.csv",
    "mapping": "competition_team_player_mapping.csv",
}

# Known non-tabular artefacts we expect (or look for) around the data pack.
NON_DATA_GLOBS = ["*.zip", "*.pdf", "*.docx", "*.ipynb", "*.py", "*.xlsx"]

BOWLER_RUN_EXTRAS_CANDIDATES = {"wide", "no_ball", "Overstep"}
ILLEGAL_DELIVERY_CANDIDATES = BOWLER_RUN_EXTRAS_CANDIDATES | {"penalty", "other", "Penalty"}

RESULT_PATTERNS = {
    "runs_win": re.compile(r"won by\s+(\d+)\s+runs?", re.IGNORECASE),
    "wickets_win": re.compile(r"won by\s+(\d+)\s+wickets?", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(search_dirs: List[Path]) -> pd.DataFrame:
    """Find CSVs and non-data artefacts (zip/pdf/docx/notebooks/scripts)."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in NON_DATA_GLOBS + ["*.csv"]:
            for f in d.glob(pat):
                key = (str(f.resolve()), f.stat().st_size)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "file": f.name,
                        "path": str(f),
                        "dir": str(d),
                        "type": f.suffix.lower().lstrip("."),
                        "size_bytes": f.stat().st_size,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dir", "type", "file"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load every raw CSV. Low-memory off so dtypes stabilise on full read."""
    out: Dict[str, pd.DataFrame] = {}
    for role, fname in RAW_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Required raw file missing: {path}")
        out[role] = pd.read_csv(path, low_memory=False)
    return out


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def _grain_guess(df: pd.DataFrame) -> str:
    if {"match_id", "innings", "overid", "over_ball_num"}.issubset(df.columns):
        return "one row per delivery (match x innings x over x ball-in-over)"
    if {"match_id", "comp_id"}.issubset(df.columns):
        return "one row per match"
    if {"comp_id", "team_id", "player_id"}.issubset(df.columns):
        return "one row per competition x team x player"
    if {"comp_id", "team_id"}.issubset(df.columns) and "player_id" not in df.columns:
        return "one row per competition x team"
    if {"match_id", "team_id", "player_id"}.issubset(df.columns):
        return "one row per match x team x player (squad / playing XI)"
    if {"comp_id", "match_format"}.issubset(df.columns):
        return "one row per competition"
    return "unknown"


def _pk_candidates(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    id_like = [c for c in cols if c.endswith("_id") or c in ("match_number",)]
    combos = []
    for c in id_like:
        if df[c].is_unique:
            combos.append(c)
    if "match_id" in cols:
        for extra in ("innings", "overid", "over_ball_num", "team_id", "player_id"):
            if extra in cols:
                subset = ["match_id", extra]
                if not df.duplicated(subset=subset).any():
                    combos.append("+".join(subset))
        if {"innings", "overid", "over_ball_num"}.issubset(cols) and not df.duplicated(
            subset=["match_id", "innings", "overid", "over_ball_num"]
        ).any():
            combos.append("match_id+innings+overid+over_ball_num")
        if {"team_id", "player_id"}.issubset(cols) and not df.duplicated(
            subset=["match_id", "team_id", "player_id"]
        ).any():
            combos.append("match_id+team_id+player_id")
    if {"comp_id", "team_id", "player_id"}.issubset(cols) and not df.duplicated(
        subset=["comp_id", "team_id", "player_id"]
    ).any():
        combos.append("comp_id+team_id+player_id")
    return combos


def _fk_candidates(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in df.columns
        if c.endswith("_id") and c != "match_number" and not df[c].is_unique
    ]


def build_inventory(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for role, df in data.items():
        rows.append(
            {
                "file": RAW_FILES[role],
                "role": role,
                "rows": len(df),
                "columns": df.shape[1],
                "grain": _grain_guess(df),
                "primary_key_candidates": "; ".join(_pk_candidates(df)),
                "foreign_key_candidates": "; ".join(_fk_candidates(df)),
                "duplicate_full_rows": int(df.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema dictionary
# ---------------------------------------------------------------------------

# Documented/verified meanings. Entries marked "verified-by-check" are confirmed
# by the quality checks in this module rather than assumed.
COLUMN_MEANING: Dict[str, str] = {
    "match_id": "Match identifier; join key to matches/playing_eleven",
    "comp_id": "Competition identifier; join key to competitions/teams/mapping",
    "innings": "Innings number (1/2 normal, 3/4 = super over of tied match)",
    "overid": "Over number within the innings (1-20 normal, 1 in super overs)",
    "over_ball_num": "Delivery counter within the over INCLUDING extras",
    "innings_ball_num": "Delivery counter within the innings INCLUDING extras",
    "batting_team_id": "Team id batting on this delivery",
    "bowling_team_id": "Team id bowling on this delivery",
    "striker_id": "Player id of the batter facing (join via player ids)",
    "striker_name": "Display name of striker (NOT a key)",
    "non_striker_id": "Player id of the non-striker batter",
    "non_striker_name": "Display name of non-striker (NOT a key)",
    "bowler_id": "Player id of the bowler",
    "bowler_name": "Display name of bowler (NOT a key)",
    "batting_type": "Batter handedness (RHB/LHB)",
    "bowling_type": "Bowling style free text (e.g. Left-arm fast-medium)",
    "batting_order": "Batter position in the innings order",
    "runs": "Runs scored off the bat by the striker (0-6); excludes extras",
    "is_four": "Boundary four off the bat (boolean as text true/false)",
    "is_six": "Six off the bat (boolean as text true/false)",
    "is_wicket": "A wicket fell on this delivery (boolean as text true/false)",
    "dismissal_type_id": "Numeric dismissal code (sparse)",
    "dismissal_type": "Dismissal mode text; null when no wicket",
    "dismissed_player_id": "Player id of dismissed batter (null when no wicket)",
    "extras_type": "Extra type: wide/Overstep(=no-ball)/byes/legbyes/other/Penalty",
    "extras1_runs": "Value of the extra itself (1 for wide/no-ball; N for byes/legbyes)",
    "extras_2_runs": "Additional runs on the same delivery (e.g. byes off a no-ball)",
    "pace_or_spin": "Bowler class: Pace or Spin",
    "bowl_speed": "Delivery speed (units undocumented; 0 = not recorded)",
    "stroke": "Shot played (free text; null for deliveries without a shot)",
    "delivery_style": "Ball movement (e.g. Inswing, Offcutter)",
    "over_or_around": "Wicket position of the bowling action: Over / Around",
    "on_or_off": "Line side: On-side / Off-side",
    "delivery_ist_time": "Delivery timestamp (IST, mixed precision)",
    "match_number": "Match number within its competition",
    "match_date": "Match start date",
    "stage": "Stage label (League/knockouts)",
    "venue_id": "Numeric venue identifier (no venue name supplied)",
    "venue_country": "Venue country",
    "home_team_id": "Home team id",
    "home_team": "Home team name (NOT a key)",
    "away_team_id": "Away team id",
    "away_team": "Away team name (NOT a key)",
    "home_team_captain_id": "Captain player id (home); 0/blank when unknown",
    "away_team_captain_id": "Captain player id (away); 0/blank when unknown",
    "toss_win_team_id": "Team id winning the toss",
    "toss_opted": "Toss decision (Batting/Fielding)",
    "man_of_the_match": "Player id of MOTM; 0 when not recorded",
    "first_inn_revised_overs": "Revised overs for innings 1 (DLS); null if not revised",
    "second_inn_revised_overs": "Revised overs for innings 2 (DLS); null if not revised",
    "revised_target": "Revised target for innings 2 (DLS); null if not revised",
    "knock_out": "1 if knockout match",
    "win_team_id": "Winning team id; null/0 for no-result, abandoned, tied",
    "result": "Free-text result string",
    "team_id": "Team id",
    "team_name": "Team name within that competition (NOT globally unique)",
    "player_id": "Player id (primary identity key for all player analysis)",
    "player_name": "Display player name (NOT a key)",
    "batting_order_PE": "Preferred batting position recorded in playing XI",
    "is_played": "True if the player took part in the match",
    "is_captain": "True if match captain (null when is_played is False)",
    "is_wk": "True if wicketkeeper (null when is_played is False)",
    "is_impact": "Impact-player flag (IPL rule)",
    "is_impact_off": "Impact-player substitution flag",
}


def build_schema_dictionary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for role, df in data.items():
        n = len(df)
        for col in df.columns:
            s = df[col]
            nunique = s.nunique(dropna=True)
            non_null = int(s.notna().sum())
            sample_vals = s.dropna().astype(str).unique()[:4]
            meaning = COLUMN_MEANING.get(col, "")
            if not meaning:
                # try role-specific hints
                meaning = COLUMN_MEANING.get(f"{col}_PE", "")
            suit = "yes"
            if non_null == 0:
                suit = "no (entirely empty)"
            elif nunique <= 1 and n > 1:
                suit = "no (constant)"
            rows.append(
                {
                    "file": RAW_FILES[role],
                    "column": col,
                    "dtype": str(s.dtype),
                    "non_null": non_null,
                    "null_count": n - non_null,
                    "null_pct": round(100.0 * (n - non_null) / max(n, 1), 3),
                    "unique": nunique,
                    "sample_values": " | ".join(sample_vals),
                    "meaning": meaning,
                    "suitable_for_modeling": suit,
                }
            )
    return pd.DataFrame(rows)


def parse_result_string(result: str) -> Dict[str, Any]:
    """Parse the free-text result into structured fields (verified in checks)."""
    if not isinstance(result, str):
        return {"result_type": "missing", "margin": np.nan, "margin_unit": None,
                "dls": False}
    r = result.strip()
    dls = bool(re.search(r"D/?LS?|Duckworth", r, re.IGNORECASE))
    if r.lower().startswith("no result"):
        return {"result_type": "no_result", "margin": np.nan, "margin_unit": None,
                "dls": dls}
    if r.lower().startswith("match abandoned"):
        return {"result_type": "abandoned", "margin": np.nan, "margin_unit": None,
                "dls": dls}
    if r.lower() == "match tied" or r.lower() == "tied":
        return {"result_type": "tied", "margin": np.nan, "margin_unit": None,
                "dls": dls}
    m = RESULT_PATTERNS["runs_win"].search(r)
    if m:
        return {"result_type": "runs", "margin": float(m.group(1)),
                "margin_unit": "runs", "dls": dls}
    m = RESULT_PATTERNS["wickets_win"].search(r)
    if m:
        return {"result_type": "wickets", "margin": float(m.group(1)),
                "margin_unit": "wickets", "dls": dls}
    return {"result_type": "unparsed", "margin": np.nan, "margin_unit": None,
            "dls": dls}
