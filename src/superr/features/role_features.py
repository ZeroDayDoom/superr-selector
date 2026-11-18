"""Role, handedness, and match-context categorical features.

Roles are data-driven (from actual batting positions and overs bowled,
Phase 3). No player-reputation or name-based information is used.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

MODULE = "role_features"

FEATURE_DOCS: Dict[str, dict] = {
    "batter_role": {
        "definition": "Batter role: opener/top/middle/finisher/lower",
        "formula": "position bands {1,2},{3,4},{5,6,7},{8-11}; middle reclassified as finisher when >=50% of the batter's innings show >=25% death-over balls",
        "source": "player_state.add_batter_roles (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Finisher refinement uses the batter's OWN current-innings death exposure at the time of the delivery and career rates computed from the full dataset — used for roles only (categorical context), not outcome prediction",
        "missing_handling": "'unknown' when batting_order missing", "relevance": "Role-fair comparisons"},
    "bowler_role": {
        "definition": "Bowler phase role: powerplay/middle/death or hybrid-*",
        "formula": "share of legal balls by phase within innings; main phase >=70% => role, else hybrid-main",
        "source": "player_state.add_bowler_roles (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Same-innings composition (pre-delivery part dominates; used categorically)",
        "missing_handling": "n/a", "relevance": "Bowler phase specialisation"},
    "batting_type": {
        "definition": "Batter handedness", "formula": "raw batting_type (RHB/LHB)",
        "source": "ball_by_ball.csv", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "'unknown' when blank", "relevance": "Matchup physics"},
    "pace_or_spin": {
        "definition": "Bowler class", "formula": "raw pace_or_spin (Pace/Spin)",
        "source": "ball_by_ball.csv", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "'unknown' when blank", "relevance": "Matchup physics"},
    "bowling_style_group": {
        "definition": "Grouped bowling style", "formula": "fast (fast/fast-medium/medium), medium-fast, spin (break/orthodox/wrist/googly), other",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "'unknown'", "relevance": "Finer matchup axis than pace/spin"},
    "batter_vs_pace_spin": {
        "definition": "Matchup cell: batter hand x bowler class", "formula": "batting_type x pace_or_spin (e.g. RHB_vs_Pace)",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "'unknown_vs_unknown'", "relevance": "League-rate matchup baselines"},
    "over_or_around_wicket": {
        "definition": "Bowling action position", "formula": "raw over_or_around",
        "source": "ball_by_ball.csv", "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "'unknown'", "relevance": "Angle information"},
    "delivery_line": {
        "definition": "Line side of the delivery", "formula": "raw on_or_off",
        "source": "ball_by_ball.csv", "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "'unknown'", "relevance": "Line context"},
    "venue_id": {
        "definition": "Venue identifier", "formula": "matches.venue_id",
        "source": "matches.csv", "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "n/a", "relevance": "Venue par / environment"},
    "is_impact_player_rule": {
        "definition": "Match played under the IPL impact-player rule (2023+ IPL)", "formula": "competition_class=='IPL' AND season>=2023",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "False", "relevance": "Rule changes alter batting depth and scoring"},
}


def add_role_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bt = df["batting_type"].astype("string").str.upper().str.strip().fillna("unknown")
    ps = df["pace_or_spin"].astype("string").str.title().str.strip().fillna("unknown")
    df["batting_type"] = bt
    df["pace_or_spin"] = ps

    style = df["bowling_type"].astype("string").str.lower().str.strip()
    df["bowling_style_group"] = np.select(
        [
            style.str.contains("fast", na=False) & ~style.str.contains("medium", na=False),
            style.str.contains("medium", na=False),
            style.str.contains("spin|break|orthodox|googly|chinaman", na=False),
        ],
        ["fast", "medium", "spin"],
        default="other",
    )
    df["batter_vs_pace_spin"] = df["batting_type"] + "_vs_" + df["pace_or_spin"]
    df["over_or_around_wicket"] = (
        df["over_or_around"].astype("string").str.strip().fillna("unknown")
    )
    df["delivery_line"] = df["on_or_off"].astype("string").str.strip().fillna("unknown")

    is_ipl = df["competition_class"].eq("IPL")
    season_num = pd.to_numeric(df["season"], errors="coerce")
    df["is_impact_player_rule"] = (is_ipl & season_num.ge(2023)).fillna(False)
    return df
