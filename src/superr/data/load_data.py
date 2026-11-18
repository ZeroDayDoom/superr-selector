"""Raw data loading and master delivery-table construction.

All joins are many-to-one and verified: row counts are checked before and
after every join so that no join can silently multiply rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from superr.data.audit_data import RAW_FILES

BOOL_COLS = {
    "is_four": "ball_by_ball",
    "is_six": "ball_by_ball",
    "is_wicket": "ball_by_ball",
    "is_played": "playing_eleven",
    "is_captain": "playing_eleven",
    "is_wk": "playing_eleven",
    "is_impact": "playing_eleven",
    "is_impact_off": "playing_eleven",
    "knock_out": "matches",
}

MATCH_JOIN_COLS = [
    "match_id", "comp_id", "match_number", "match_date", "stage", "venue_id",
    "venue_country", "home_team_id", "home_team", "away_team_id", "away_team",
    "toss_win_team_id", "toss_opted", "first_inn_revised_overs",
    "second_inn_revised_overs", "revised_target", "knock_out", "win_team_id",
    "result",
]


@dataclass
class JoinReport:
    """Row-count verification for every join performed."""
    steps: List[Tuple[str, int, int, str]]

    def add(self, step: str, before: int, after: int, kind: str) -> None:
        self.steps.append((step, before, after, kind))

    @property
    def ok(self) -> bool:
        return all(b == a for _, b, a, _ in self.steps)

    def __str__(self) -> str:
        lines = [f"{step}: {before:,} -> {after:,} ({kind})"
                 for step, before, after, kind in self.steps]
        return "\n".join(lines)


def load_tables(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load raw CSVs with dtypes normalised."""
    out: Dict[str, pd.DataFrame] = {}
    for role, fname in RAW_FILES.items():
        df = pd.read_csv(raw_dir / fname, low_memory=False)
        # boolean columns arrive as text true/false
        for col, src in BOOL_COLS.items():
            if col in df.columns and (src == role or src == "matches"):
                df[col] = (
                    df[col].astype("string").str.lower().eq("true").astype("boolean")
                )
        # integer-ish id columns: keep as nullable Int64 for safe joins
        for col in df.columns:
            if col.endswith("_id") and df[col].dtype == "float64":
                df[col] = df[col].astype("Int64")
            elif col.endswith("_id") and df[col].dtype == "int64":
                df[col] = df[col].astype("Int64")
        out[role] = df
    return out


def build_master_delivery_table(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
) -> Tuple[pd.DataFrame, JoinReport]:
    """Join deliveries -> matches -> competitions (both many-to-one)."""
    report = JoinReport(steps=[])
    n0 = len(balls)

    mcols = [c for c in MATCH_JOIN_COLS if c in matches.columns]
    m = matches[mcols].copy()
    if m["match_id"].duplicated().any():
        raise ValueError("matches.csv has duplicated match_id; refusing to join")

    master = balls.merge(m, on="match_id", how="left", validate="many_to_one")
    report.add("join_matches", n0, len(master), "many_to_one")

    c = competitions[["comp_id", "comp_name", "match_format", "season"]].copy()
    if c["comp_id"].duplicated().any():
        raise ValueError("competitions.csv has duplicated comp_id; refusing to join")
    master = master.merge(c, on="comp_id", how="left", validate="many_to_one")
    report.add("join_competitions", n0, len(master), "many_to_one")

    if not report.ok:
        raise RuntimeError(f"Join multiplied rows:\n{report}")
    if master["comp_name"].isna().any():
        n_missing = int(master["comp_name"].isna().sum())
        raise ValueError(f"{n_missing} deliveries have no competition record")

    # chronological + in-match ordering for every downstream cumsum
    master = master.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="mergesort"
    ).reset_index(drop=True)
    return master, report
