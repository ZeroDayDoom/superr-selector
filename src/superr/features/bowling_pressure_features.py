"""Bowling pressure features (per delivery, bowler perspective).

Includes defensive-situation mirrors of the chase state, bowling-first par
pressure, wicket-drought metrics, bowler momentum (recomputed over the
bowler's own deliveries with legal-ball denominators), and breakthrough
context. All strictly pre-delivery.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

MODULE = "bowling_pressure_features"

FEATURE_DOCS: Dict[str, dict] = {
    # --- defensive situation (chase mirror) ---
    "runs_required_before": {
        "definition": "Runs the bowling side must still defend", "formula": "target - score_before",
        "source": "match_state.add_chase_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "NaN in innings 1",
        "relevance": "Defence difficulty in chases"},
    "required_run_rate": {
        "definition": "Required run rate the bowler must suppress", "formula": "runs_required/(balls_remaining/6)",
        "source": "match_state.add_chase_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "NaN in innings 1",
        "relevance": "High RRR defence is peak bowling pressure"},
    "rrr_crr_gap": {
        "definition": "Required minus current run rate", "formula": "required_run_rate - current_run_rate",
        "source": "match_state.add_chase_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "NaN in innings 1",
        "relevance": "Rising gap = bowler in command; falling gap = bowler under pump"},
    "wickets_remaining": {
        "definition": "Wickets the batting side has left", "formula": "10 - wickets_before",
        "source": "match_state.add_pre_delivery_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "n/a",
        "relevance": "Fewer wickets = attacking opportunity"},
    "target_progress": {
        "definition": "Fraction of target scored by batting side", "formula": "score_before/target",
        "source": "match_state.add_chase_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "NaN in innings 1",
        "relevance": "How far the defence has been penetrated"},
    "batting_resources_remaining": {
        "definition": "Batting-side resource proxy (balls x wickets)", "formula": "(balls_left/scheduled)*(wickets_left/10)",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "n/a", "relevance": "Mirror of resources_remaining for the fielding side"},
    # --- bowling-first pressure ---
    "score_vs_par": {
        "definition": "Batting side's runs above/below state-expected", "formula": "score_before - expected_score_at_state",
        "source": "match_state.add_par_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "LOO curves", "missing_handling": "Fallback hierarchy",
        "relevance": "Bowler suppressing scoring below par = runs saved"},
    "expected_final_score": {
        "definition": "Projected batting total from state", "formula": "expected_score_at_state + expected_runs_remaining",
        "source": "match_state.add_par_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "LOO curves", "missing_handling": "Fallback hierarchy",
        "relevance": "What the bowler is defending / containing"},
    "expected_remaining_runs": {
        "definition": "Expected runs the batting side will still score", "formula": "expected_runs_remaining (LOO curve)",
        "source": "match_state.add_par_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "LOO curves", "missing_handling": "Fallback hierarchy",
        "relevance": "Damage control baseline for the bowler"},
    "batting_rate_vs_par": {
        "definition": "Batting run rate minus par run rate", "formula": "current_run_rate - par_run_rate",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "Venue par leave-one-match-out", "missing_handling": "NaN when par unavailable",
        "relevance": "Whether the bowler has contained below the environment"},
    "wickets_in_hand": {
        "definition": "Batting-side wickets remaining", "formula": "10 - wickets_before",
        "source": "match_state.add_pre_delivery_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "n/a",
        "relevance": "Attack opportunity when low"},
    "overs_remaining": {
        "definition": "Overs left for the bowling side", "formula": "legal_balls_remaining/6",
        "source": "match_state.add_pre_delivery_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "n/a", "relevance": "Time to take wickets"},
    # --- wicket drought ---
    "bowling_team_wicket_drought": {
        "definition": "Deliveries since the bowling team last took a wicket", "formula": "position - last team-wicket position",
        "source": "player_state.add_bowler_state", "available_before_delivery": True,
        "leakage_risk": "Pre-delivery only", "missing_handling": "NaN before first wicket",
        "relevance": "Extended droughts raise defensive pressure"},
    "wicket_drought_severity": {
        "definition": "Drought relative to expectation, 0-1", "formula": "clip(drought / (2/league_wicket_rate), 0, 1) with league_wicket_rate a FIXED pre-computed constant",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "None — normalizer is a fixed constant (10392/178000), never frame-derived",
        "missing_handling": "0 before first wicket", "relevance": "Scales drought by how unusual it is"},
    "striker_setness": {
        "definition": "Striker balls faced (setness)", "formula": "batter_balls_faced_before",
        "source": "player_state.add_striker_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "0 at entry",
        "relevance": "Set batters are the drought's cause"},
    "striker_runs": {
        "definition": "Striker runs so far", "formula": "batter_runs_before",
        "source": "player_state.add_striker_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "0 at entry",
        "relevance": "Threat level of the set batter"},
    "wicket_ends_drought": {
        "definition": "Would a wicket here end a long drought (context flag)", "formula": "wicket_drought_severity >= 0.5",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "False", "relevance": "Marks drought-ending breakthrough opportunities"},
    # --- bowling momentum (bowler's own deliveries) ---
    "bowler_recent_economy": {
        "definition": "Bowler's economy over previous 12 own deliveries", "formula": "shifted rolling runs / (shifted rolling legal/6)",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted; excludes current ball",
        "missing_handling": "NaN in first 12 own deliveries", "relevance": "Recent form of the bowler"},
    "bowler_recent_dot_rate": {
        "definition": "Dot share over previous 12 own deliveries", "formula": "shifted dots / shifted legal",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Containment form"},
    "bowler_recent_wicket_rate": {
        "definition": "Wickets per legal ball over previous 24 own deliveries", "formula": "shifted wkts / shifted legal",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Striking form"},
    "bowler_last6_runs": {
        "definition": "Runs conceded in previous 6 own deliveries", "formula": "shifted rolling sum",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Shifted", "missing_handling": "NaN early", "relevance": "Short-form pressure"},
    "bowler_last12_runs": {
        "definition": "Runs conceded in previous 12 own deliveries", "formula": "shifted rolling sum",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Shifted", "missing_handling": "NaN early", "relevance": "Recent economy"},
    "bowler_last18_runs": {
        "definition": "Runs conceded in previous 18 own deliveries", "formula": "shifted rolling sum",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Recent economy"},
    "bowler_last12_dots": {
        "definition": "Dots in previous 12 own deliveries", "formula": "shifted rolling count",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Shifted", "missing_handling": "NaN early", "relevance": "Containment"},
    "bowler_last12_wickets": {
        "definition": "Wickets in previous 12 own deliveries", "formula": "shifted rolling count",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Shifted", "missing_handling": "NaN early", "relevance": "Striking"},
    "bowler_last12_boundaries": {
        "definition": "Boundaries conceded in previous 12 own deliveries", "formula": "shifted rolling count",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Shifted", "missing_handling": "NaN early", "relevance": "Leakage of boundaries"},
    "bowler_last18_dots": {
        "definition": "Dots in previous 18 own deliveries", "formula": "shifted rolling count",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Containment"},
    "bowler_last18_wickets": {
        "definition": "Wickets in previous 18 own deliveries", "formula": "shifted rolling count",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Striking"},
    "bowler_boundary_streak_before": {
        "definition": "Consecutive boundaries conceded immediately before", "formula": "streak length",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "Pre-delivery streak", "missing_handling": "0 when none",
        "relevance": "Bowler under boundary siege"},
    "bowler_overs_completed": {
        "definition": "Completed overs by this bowler in the innings", "formula": "floor(bowler_balls_before/6)",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "0 at start", "relevance": "Spell management"},
    "bowler_spell_overs_before": {
        "definition": "Overs in the current spell before this over", "formula": "consecutive same-bowler overs",
        "source": "player_state.add_bowler_state (re-exposed)", "available_before_delivery": True,
        "leakage_risk": "None", "missing_handling": "0 at spell start", "relevance": "Fatigue proxy"},
    "bowler_last6_legal": {
        "definition": "Legal balls in previous 6 own deliveries", "formula": "shifted rolling legal count",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Denominator for recent economy"},
    "bowler_last6_legalruns": {
        "definition": "Runs conceded in previous 6 own legal-denominated deliveries", "formula": "shifted rolling runs",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Recent economy numerator"},
    "bowler_last12_legal": {
        "definition": "Legal balls in previous 12 own deliveries", "formula": "shifted rolling legal count",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Denominator for recent economy/dot rate"},
    "bowler_last12_legalruns": {
        "definition": "Runs conceded in previous 12 own deliveries", "formula": "shifted rolling runs",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Recent economy numerator"},
    "bowler_last18_legal": {
        "definition": "Legal balls in previous 18 own deliveries", "formula": "shifted rolling legal count",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Denominator"},
    "bowler_last18_legalruns": {
        "definition": "Runs conceded in previous 18 own deliveries", "formula": "shifted rolling runs",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Recent economy numerator"},
    "bowler_last24_legal": {
        "definition": "Legal balls in previous 24 own deliveries", "formula": "shifted rolling legal count",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Denominator for recent wicket rate"},
    "bowler_last24_legalruns": {
        "definition": "Runs conceded in previous 24 own deliveries", "formula": "shifted rolling runs",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "Shifted",
        "missing_handling": "NaN early", "relevance": "Recent economy numerator"},
}

# League wicket rate per legal ball: FIXED, pre-computed normalizer (never
# derived from the frame being scored — a frame-derived rate would make
# wicket_drought_severity depend on the current delivery's own outcome,
# which the outcome-perturbation leakage audit correctly flags).
LEAGUE_WICKET_RATE_PER_BALL = 10392.0 / 178000.0  # ~0.058


def add_bowling_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sched = df["scheduled_legal_balls"].replace(0, np.nan)
    df["batting_resources_remaining"] = (
        (df["legal_balls_remaining"] / sched).clip(0, 1)
        * (df["wickets_in_hand"] / 10.0)
    ).clip(0, 1)
    df["batting_rate_vs_par"] = df["current_run_rate"] - df["par_run_rate"]
    df["expected_remaining_runs"] = df["expected_runs_remaining"]
    df["striker_setness"] = df["batter_balls_faced_before"]
    df["striker_runs"] = df["batter_runs_before"]

    league_wpb = 1.0 / LEAGUE_WICKET_RATE_PER_BALL  # fixed normalizer
    df["wicket_drought_severity"] = (
        df["bowling_team_wicket_drought"].fillna(0) / (league_wpb * 2)
    ).clip(0, 1)
    df["wicket_ends_drought"] = df["wicket_drought_severity"].ge(0.5)

    # bowler momentum: rolling over the bowler's OWN deliveries with legal
    # denominators, shifted by one so the current ball is excluded.
    base = pd.DataFrame(index=df.index)
    base["match_id"] = df["match_id"]
    base["innings"] = df["innings"]
    base["bowler_id"] = df["bowler_id"]
    base["runs"] = df["bowler_runs_conceded"]
    base["legal"] = df["is_legal_ball"].astype("int8")
    base["dots"] = df["is_dot_ball"].astype("int8")
    base["wkts"] = df["is_bowler_wicket"].astype("int8")
    bg = base.groupby(["match_id", "innings", "bowler_id"], sort=False)
    for w in (6, 12, 18, 24):
        r = bg.rolling(w, min_periods=1).sum().reset_index(level=[0, 1, 2], drop=True)
        rs = r.groupby([df["match_id"], df["innings"], df["bowler_id"]],
                       sort=False).shift(1)
        df[f"bowler_last{w}_legal"] = rs["legal"]
        df[f"bowler_last{w}_legalruns"] = rs["runs"]
        if w == 6:
            df["bowler_last6_runs"] = rs["runs"]
        elif w == 12:
            df["bowler_last12_runs"] = rs["runs"]
        elif w == 18:
            df["bowler_last18_runs"] = rs["runs"]
    df["bowler_last12_dots"] = (
        base.groupby(["match_id", "innings", "bowler_id"], sort=False)["dots"]
        .rolling(12, min_periods=1).sum().reset_index(level=[0, 1, 2], drop=True)
        .groupby([df["match_id"], df["innings"], df["bowler_id"]], sort=False).shift(1)
    )
    df["bowler_last18_dots"] = (
        base.groupby(["match_id", "innings", "bowler_id"], sort=False)["dots"]
        .rolling(18, min_periods=1).sum().reset_index(level=[0, 1, 2], drop=True)
        .groupby([df["match_id"], df["innings"], df["bowler_id"]], sort=False).shift(1)
    )
    df["bowler_last12_wickets"] = (
        base.groupby(["match_id", "innings", "bowler_id"], sort=False)["wkts"]
        .rolling(12, min_periods=1).sum().reset_index(level=[0, 1, 2], drop=True)
        .groupby([df["match_id"], df["innings"], df["bowler_id"]], sort=False).shift(1)
    )

    legal12 = df["bowler_last12_legal"]
    legal24 = df["bowler_last24_legal"]
    df["bowler_recent_economy"] = np.where(
        legal12 > 0, df["bowler_last12_legalruns"] / (legal12 / 6.0), np.nan
    )
    df["bowler_recent_dot_rate"] = np.where(
        legal12 > 0, df["bowler_last12_dots"] / legal12, np.nan
    )
    wkts24 = base.groupby(["match_id", "innings", "bowler_id"], sort=False)["wkts"] \
        .rolling(24, min_periods=1).sum().reset_index(level=[0, 1, 2], drop=True) \
        .groupby([df["match_id"], df["innings"], df["bowler_id"]], sort=False).shift(1)
    df["bowler_recent_wicket_rate"] = np.where(
        legal24 > 0, wkts24 / legal24, np.nan
    )
    return df
