"""Matchup features: league-rate baselines, never raw player-vs-player stats.

Player-specific matchup statistics are NOT used without shrinkage and are
NOT used from the future: the matchup baselines here are LEAGUE-level rates
by (batter hand x bowler class / style group), computed leave-one-season-out
so a season's own deliveries never inform its baseline. Player-vs-bowler
career matchup features are deliberately deferred to the model phase where
they can be computed walk-forward.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

MODULE = "matchup_features"

# shrinkage strength (pseudo-count) for league matchup rates
MATCHUP_K = 200.0
# neutral, data-free priors used ONLY when no other season exists to form a
# leave-one-season-out prior (guarantees perturbation invariance)
NEUTRAL_RPB = 1.4
NEUTRAL_WICKET_RATE = 0.055

FEATURE_DOCS: Dict[str, dict] = {
    "matchup_lboro_runs_per_ball": {
        "definition": "League runs-per-legal-ball for this hand x bowler-class cell",
        "formula": "EB-shrunk mean: (S_cell - S_own_season + k*global) / (N_cell - N_own_season + k), leave-one-season-out",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "Leave-one-season-out: the delivery's own season never contributes to its baseline; no player-specific stat involved",
        "missing_handling": "Falls back to hand-level, then global mean",
        "relevance": "Pre-delivery difficulty of the handedness matchup"},
    "matchup_lboro_wicket_rate": {
        "definition": "League wicket probability for this hand x bowler-class cell",
        "formula": "as above on bowler-wicket frequency per legal ball",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "Leave-one-season-out", "missing_handling": "Fallback hierarchy",
        "relevance": "Dismissal likelihood of the matchup"},
    "matchup_style_runs_per_ball": {
        "definition": "League runs-per-legal-ball for hand x bowling-style group",
        "formula": "as matchup_lboro_runs_per_ball on style groups",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "Leave-one-season-out", "missing_handling": "Fallback to class-level",
        "relevance": "Finer matchup axis (fast vs medium vs spin variants)"},
    "matchup_style_wicket_rate": {
        "definition": "League wicket probability for hand x bowling-style group",
        "formula": "as matchup_lboro_wicket_rate on style groups",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "Leave-one-season-out", "missing_handling": "Fallback to class-level",
        "relevance": "Finer matchup dismissal axis"},
    "matchup_difficulty_index": {
        "definition": "Cell difficulty relative to league average, 0-centred",
        "formula": "matchup_lboro_runs_per_ball - global runs per legal ball",
        "source": MODULE, "available_before_delivery": True,
        "leakage_risk": "Leave-one-season-out", "missing_handling": "0 when unavailable",
        "relevance": "Single-axis matchup difficulty"},
    "bowling_style_group_bat_hand": {
        "definition": "Combined hand x style-group cell key",
        "formula": "batting_type + '_vs_' + bowling_style_group",
        "source": MODULE, "available_before_delivery": True, "leakage_risk": "None",
        "missing_handling": "'unknown_vs_unknown'", "relevance": "Grouping key for style-level matchup rates"},
}


def _loo_group_rates(
    df: pd.DataFrame, group_col: str, season_col: str = "season"
) -> pd.DataFrame:
    """Leave-one-season-out group means for runs/balls and wickets/balls.

    The EB prior is ALSO leave-one-season-out: the global rate used as the
    prior for season s excludes season s entirely, so no delivery's own
    outcome can move its own baseline.
    """
    legal = df["is_legal_ball"]
    g = (
        df[legal]
        .groupby([group_col, season_col], observed=True)
        .agg(runs=("batter_runs", "sum"), legal=("is_legal_ball", "size"),
             wkts=("is_bowler_wicket", "sum"))
        .reset_index()
    )
    tot = g.groupby(group_col, observed=True)[["runs", "legal", "wkts"]].sum()
    seas = g.groupby(season_col, observed=True)[["runs", "legal", "wkts"]].sum()

    # global leave-one-season-out priors: for season s, exclude season s
    glob = seas.copy()
    grand = {
        "runs": float(df.loc[legal, "batter_runs"].sum()),
        "legal": float(legal.sum()),
        "wkts": float(df["is_bowler_wicket"].sum()),
    }
    glob["loo_runs"] = grand["runs"] - glob["runs"]
    glob["loo_legal"] = grand["legal"] - glob["legal"]
    glob["loo_wkts"] = grand["wkts"] - glob["wkts"]
    glob["prior_rpb"] = (glob["loo_runs"] / glob["loo_legal"].replace(0, np.nan))
    glob["prior_wkt"] = (glob["loo_wkts"] / glob["loo_legal"].replace(0, np.nan))
    # no other season exists -> neutral data-free prior (perturbation-invariant)
    glob["prior_rpb"] = glob["prior_rpb"].fillna(NEUTRAL_RPB)
    glob["prior_wkt"] = glob["prior_wkt"].fillna(NEUTRAL_WICKET_RATE)

    out = g.merge(tot.add_prefix("g_").reset_index(), on=group_col, how="left")
    out = out.merge(seas.add_prefix("s_").reset_index(), on=season_col, how="left")
    for c in ("runs", "legal", "wkts"):
        out[f"loo_{c}"] = out[f"g_{c}"] - out[f"s_{c}"]
    out = out.merge(glob[["prior_rpb", "prior_wkt"]].reset_index(),
                    on=season_col, how="left")
    out["loo_runs_per_ball"] = (out["loo_runs"] + MATCHUP_K * out["prior_rpb"]) / (
        out["loo_legal"] + MATCHUP_K
    )
    out["loo_wicket_rate"] = (out["loo_wkts"] + MATCHUP_K * out["prior_wkt"]) / (
        out["loo_legal"] + MATCHUP_K
    )
    return out[[group_col, season_col, "loo_runs_per_ball", "loo_wicket_rate"]]


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # per-season leave-one-season-out global fallbacks
    legal = df["is_legal_ball"]
    seas = (
        df[legal].groupby("season", observed=True)
        .agg(runs=("batter_runs", "sum"), lg=("is_legal_ball", "size"),
             wk=("is_bowler_wicket", "sum"))
    )
    grand_r = float(df.loc[legal, "batter_runs"].sum())
    grand_l = float(legal.sum())
    grand_w = float(df["is_bowler_wicket"].sum())
    seas["prior_rpb"] = ((grand_r - seas["runs"]) / (grand_l - seas["lg"]).replace(0, np.nan))
    seas["prior_wkt"] = ((grand_w - seas["wk"]) / (grand_l - seas["lg"]).replace(0, np.nan))
    seas["prior_rpb"] = seas["prior_rpb"].fillna(NEUTRAL_RPB)
    seas["prior_wkt"] = seas["prior_wkt"].fillna(NEUTRAL_WICKET_RATE)
    prior_map = seas[["prior_rpb", "prior_wkt"]]
    df = df.merge(prior_map, left_on="season", right_index=True,
                  how="left", validate="many_to_one")

    for group_col, rpb_col, wkt_col, fb_col in [
        ("batter_vs_pace_spin", "matchup_lboro_runs_per_ball",
         "matchup_lboro_wicket_rate", None),
        ("bowling_style_group_bat_hand", "matchup_style_runs_per_ball",
         "matchup_style_wicket_rate", "matchup_lboro_runs_per_ball"),
    ]:
        if group_col not in df.columns:
            df[group_col] = df["batting_type"] + "_vs_" + df["bowling_style_group"]
        rates = _loo_group_rates(df, group_col)
        key = df[[group_col, "season"]].merge(
            rates, on=[group_col, "season"], how="left"
        )
        vals_rpb = pd.Series(key["loo_runs_per_ball"].values, index=df.index)
        vals_wkt = pd.Series(key["loo_wicket_rate"].values, index=df.index)
        if fb_col is not None:
            vals_rpb = vals_rpb.fillna(df[fb_col])
            vals_wkt = vals_wkt.fillna(df["matchup_lboro_wicket_rate"])
        df[rpb_col] = vals_rpb.fillna(df["prior_rpb"])
        df[wkt_col] = vals_wkt.fillna(df["prior_wkt"])
        df["matchup_difficulty_index"] = (
            df["matchup_lboro_runs_per_ball"] - df["prior_rpb"]
        )
    df = df.drop(columns=["prior_rpb", "prior_wkt"])
    return df
