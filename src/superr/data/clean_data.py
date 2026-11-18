"""Delivery-level reconstruction: legal balls, runs, wickets.

Semantics follow the verified audit (reports/data_audit_decisions.md):
- total_runs = runs (off bat) + extras1_runs + extras_2_runs
- legal ball = extras_type not in {wide, no_ball, penalty, other}
- bowler-conceded runs = off-bat runs + wides + no-balls (+ runs on them);
  byes / leg-byes / penalty / other are NOT charged to the bowler
- bowler wickets = Bowled / Caught / LBW / Stumped / Hit Wicket;
  run-outs, retired-out, obstructing are team wickets but not bowler wickets
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# canonical extras labels (raw label 'Overstep' = no-ball)
ILLEGAL_EXTRAS = {"wide", "no_ball", "penalty", "other"}
BOWLER_CHARGED_EXTRAS = {"wide", "no_ball"}

BOWLER_WICKET_MODES = {"Bowled", "Caught", "LBW", "Stump", "Hit Wicket"}

DISMISSAL_TYPE_MAP = {
    "Obstracting the field": "Obstructing the field",  # raw spelling variant
}

# dismissal types that are wickets for the batting side
TEAM_WICKET_TYPES = {
    "Bowled", "Caught", "LBW", "RunOut", "Stump", "Hit Wicket", "Retire Out",
    "Obstructing the field",
}


def clean_extras(master: pd.DataFrame) -> pd.DataFrame:
    """Normalise extras_type labels to a canonical lowercase vocabulary."""
    df = master.copy()
    df["extras_type_raw"] = df["extras_type"]
    df["extras_type"] = (
        df["extras_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"overstep": "no_ball"})
    )
    # canonical labels after normalisation:
    # nan, wide, no_ball, byes, legbyes, other, penalty
    # 2 deliveries in the pack carry extras values with no type label; they
    # are explicitly labelled 'unknown' (kept legal, extras uncharged to the
    # bowler) rather than silently discarded or guessed. See cleaning report.
    unknown = df["extras_type"].isna() & (
        (df["extras1_runs"].fillna(0) + df["extras_2_runs"].fillna(0)) > 0
    )
    df.loc[unknown, "extras_type"] = "unknown"
    return df


def add_run_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    """batter_runs / extras_runs / total_runs / bowler_runs_conceded."""
    df = df.copy()
    runs = df["runs"].fillna(0).astype("float64")
    ex1 = df["extras1_runs"].fillna(0).astype("float64")
    ex2 = df["extras_2_runs"].fillna(0).astype("float64")

    df["batter_runs"] = runs
    df["extras_runs"] = ex1 + ex2
    df["total_runs"] = runs + ex1 + ex2

    is_bowler_charged = (
        df["extras_type"].isin(BOWLER_CHARGED_EXTRAS).fillna(False)
        | df["extras_type_raw"].isin(BOWLER_CHARGED_EXTRAS).fillna(False)
    )
    # bowler concedes: off-bat runs always; extras only if wide/no-ball
    # (including any additional runs accumulated on those deliveries)
    df["bowler_runs_conceded"] = runs + np.where(
        is_bowler_charged, ex1 + ex2, 0.0
    )
    return df


def add_legal_ball(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_legal_ball"] = ~df["extras_type"].isin(ILLEGAL_EXTRAS).fillna(False)
    # byes/leg-byes are legal deliveries but produce no off-bat runs
    df["is_dot_ball"] = (
        df["is_legal_ball"] & df["total_runs"].eq(0)
    )
    df["is_boundary"] = df[["is_four", "is_six"]].astype("boolean").fillna(False).any(axis=1)
    return df


def add_wicket_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_wicket = df["is_wicket"].astype("boolean").fillna(False)
    dt = df["dismissal_type"].astype("string").str.strip().replace(DISMISSAL_TYPE_MAP)
    df["wicket_type"] = dt

    df["is_team_wicket"] = (is_wicket & dt.isin(TEAM_WICKET_TYPES)).fillna(False)
    df["is_bowler_wicket"] = (is_wicket & dt.isin(BOWLER_WICKET_MODES)).fillna(False)

    # batter dismissal: the dismissed player is one of the two at the crease
    disp = df["dismissed_player_id"]
    df["is_batter_dismissal"] = (
        is_wicket & disp.notna() & (
            disp.eq(df["striker_id"]) | disp.eq(df["non_striker_id"])
        )
    ).fillna(False)
    df["dismissed_is_striker"] = (
        is_wicket & disp.notna() & disp.eq(df["striker_id"])
    ).fillna(False)
    df["is_retired_out"] = dt.eq("Retire Out").fillna(False)
    df["is_run_out"] = dt.eq("RunOut").fillna(False)
    # retired hurt does not appear in this data pack; guarded anyway
    df["is_retired_hurt"] = (
        dt.str.lower().eq("retired hurt") | dt.str.lower().eq("retired not out")
    ).fillna(False)

    # consistency guard: a wicket flag without a known type is invalid
    bad = is_wicket & dt.isna()
    if bad.any():
        n = int(bad.sum())
        raise ValueError(f"{n} deliveries flagged is_wicket with no dismissal_type")
    unknown_types = sorted(set(dt.dropna()) - TEAM_WICKET_TYPES)
    if unknown_types:
        raise ValueError(f"unmapped dismissal types present: {unknown_types}")
    return df


def add_special_handling(df: pd.DataFrame, match_scope: pd.DataFrame) -> pd.DataFrame:
    """Flags for super overs, DLS, ties; drop super-over innings from scope."""
    df = df.copy()
    df["is_super_over_innings"] = df["innings"].ge(3)
    df["is_dls_revised"] = (
        df["first_inn_revised_overs"].notna()
        | df["second_inn_revised_overs"].notna()
    )
    df["has_revised_target"] = df["revised_target"].notna()
    df["match_result_type"] = df["match_id"].map(
        match_scope.set_index("match_id")["result_type"]
    )
    df["match_in_scope"] = df["match_id"].map(
        match_scope.set_index("match_id")["scope_include"]
    )
    return df


def apply_scope(
    df: pd.DataFrame, keep_super_over_innings: bool = False
) -> pd.DataFrame:
    """Filter to the analysis scope; report what was dropped (never silent)."""
    n0 = len(df)
    out = df[df["match_in_scope"].astype("boolean").fillna(False)]
    n1 = len(out)
    if not keep_super_over_innings:
        out = out[~out["is_super_over_innings"]]
    n2 = len(out)
    out = out.reset_index(drop=True)
    out.attrs["rows_dropped_out_of_scope"] = n0 - n1
    out.attrs["rows_dropped_super_over"] = n1 - n2
    return out
