"""Player baselines: pressure vs normal performance aggregates and uplift.

Normal performance is NOT a raw average: it is computed on normal-pressure
deliveries and context-adjusted by comparing each player's HP mean against
the pool relationship between normal and HP performance (the residual
regression of Methodology §4.1). Four uplift versions are produced:
  - absolute:  HP mean value (level)
  - raw:       HP mean - Normal mean
  - adjusted:  residual of HP mean on Normal mean across the pool
  - normalised: raw uplift / (|Normal mean| + scale), signed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HP_COL = "hp_top20"


def _agg_block(d: pd.DataFrame, value_col: str) -> pd.DataFrame:
    hp = d[d[HP_COL].astype(bool)]
    normal = d[~d[HP_COL].astype(bool)]
    g_hp = hp.groupby("player_id")[value_col]
    g_no = normal.groupby("player_id")[value_col]
    out = pd.DataFrame({
        "hp_mean": g_hp.mean(),
        "hp_balls": g_hp.size(),
        "hp_sum": g_hp.sum(),
        "normal_mean": g_no.mean(),
        "normal_balls": g_no.size(),
    })
    return out


def pressure_metrics(d: pd.DataFrame, role: str) -> pd.DataFrame:
    """All pressure performance metrics per player for one role."""
    hp = d[d[HP_COL].astype(bool)]
    value_col = "bat_value" if role == "batter" else "bowl_value"

    g = hp.groupby("player_id")
    out = pd.DataFrame({
        "hp_balls": g.size(),
        "hp_value_sum": g[value_col].sum(),
        "hp_value_mean": g[value_col].mean(),
        "hp_runs": g["batter_runs"].sum(),
        "hp_wpa_sum": g["realized_wpa" if role == "batter" else "bowl_wpa"].sum(),
        "hp_runs_above_expected": g["bat_runs_above_expected"].sum()
        if role == "batter" else np.nan,
        "hp_wickets_above_expected": g["bowl_wickets_above_expected"].sum()
        if role == "bowler" else np.nan,
    })
    # strike rate / average (batters) on HP balls
    if role == "batter":
        out["hp_strike_rate"] = 100.0 * out["hp_runs"] / out["hp_balls"].clip(lower=1)
        dismissals = hp[hp["dismissed_is_striker"].astype(bool)] \
            .groupby("player_id").size()
        out["hp_dismissals"] = dismissals.reindex(out.index).fillna(0)
        out["hp_average"] = out["hp_runs"] / out["hp_dismissals"].clip(lower=1)
    else:
        legal = hp.groupby("player_id")["is_legal_ball"].sum()
        out["hp_legal_balls"] = legal
        conceded = hp.groupby("player_id")["bowler_runs_conceded"].sum()
        out["hp_economy"] = 6.0 * conceded / legal.clip(lower=1)
        out["hp_wicket_rate"] = (hp.groupby("player_id")["is_bowler_wicket"].sum()
                                 / legal.clip(lower=1))
        out["hp_dot_pct"] = 100.0 * hp.groupby("player_id")["is_dot_ball"].mean()
        out["hp_boundary_pct"] = 100.0 * hp.groupby("player_id")["is_boundary"].mean()

    # extreme pressure subset (top 10% overall)
    extreme = hp[hp["hp_extreme"].astype(bool)] if "hp_extreme" in hp.columns else hp
    out["extreme_balls"] = extreme.groupby("player_id").size().reindex(out.index).fillna(0)
    out["extreme_value_mean"] = extreme.groupby("player_id")[value_col] \
        .mean().reindex(out.index)

    # opportunity / scenario diversity
    out["hp_matches"] = hp.groupby("player_id")["match_id"].nunique()
    out["hp_innings"] = (
        hp.drop_duplicates(["match_id", "innings", "player_id"])
        .groupby("player_id").size()
    )
    out["n_seasons"] = hp.groupby("player_id")["season"].nunique()
    out["n_opponents"] = hp.groupby("player_id")["opponent_team_id"].nunique()

    # normal baseline
    normal = d[~d[HP_COL].astype(bool)]
    gn = normal.groupby("player_id")[value_col]
    out["normal_mean"] = gn.mean()
    out["normal_balls"] = gn.size()

    # uplift versions
    out["uplift_raw"] = out["hp_value_mean"] - out["normal_mean"]
    return out.reset_index()


def context_adjusted_uplift(agg: pd.DataFrame) -> pd.Series:
    """Residual of HP mean on Normal mean across the pool (OLS)."""
    d = agg.dropna(subset=["hp_value_mean", "normal_mean"])
    if len(d) < 20 or d["normal_mean"].std() == 0:
        return pd.Series(np.nan, index=agg.index)
    x = d["normal_mean"].values
    y = d["hp_value_mean"].values
    b, a = np.polyfit(x, y, 1)
    pred = a + b * agg["normal_mean"]
    return (agg["hp_value_mean"] - pred).values


def normalise_uplift(agg: pd.DataFrame, scale: float = 0.05) -> pd.Series:
    """Uplift relative to the player's own baseline magnitude."""
    return agg["uplift_raw"] / (agg["normal_mean"].abs() + scale)
