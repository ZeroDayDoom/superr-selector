"""Role and competition validation: systematic-advantage checks."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def role_advantage(agg: pd.DataFrame, role_col: str) -> Dict:
    """Is any role systematically advantaged in the composite?"""
    g = agg.groupby(role_col)["rating_primary"].agg(["mean", "std", "size"])
    g = g[g["size"] >= 5].sort_values("mean", ascending=False)
    spread = float(g["mean"].max() - g["mean"].min()) if len(g) > 1 else 0.0
    # top-10 composition by role
    top10 = agg.nlargest(10, "rating_primary")[role_col].value_counts().to_dict()
    return {
        "role_means": g["mean"].round(3).to_dict(),
        "role_sizes": g["size"].to_dict(),
        "max_mean_gap": round(spread, 3),
        "top10_composition": top10,
    }


def role_normalized_vs_raw(agg: pd.DataFrame, role_col: str) -> Dict:
    """Rank with role-mean adjustment (within-role z) vs raw composite."""
    d = agg.copy()
    role_mean = d.groupby(role_col)["rating_primary"].transform("mean")
    d["rating_role_norm"] = d["rating_primary"] - role_mean
    raw_rank = d["rating_primary"].rank(ascending=False)
    norm_rank = d["rating_role_norm"].rank(ascending=False)
    rho = spearmanr(raw_rank, norm_rank).statistic
    moved = d.assign(raw=raw_rank, norm=norm_rank)
    moved["delta"] = moved["raw"] - moved["norm"]
    biggest_gain = moved.nlargest(5, "delta")[
        ["player_name", role_col, "raw", "norm", "delta"]]
    return {
        "spearman_raw_vs_rolenorm": round(float(rho), 3),
        "biggest_gainers_from_normalization": biggest_gain.to_dict("records"),
    }


def competition_balance(agg: pd.DataFrame, comp_share_col: str = None,
                        hp_ipl: str = "hpval_IPL",
                        hp_t20i: str = "hpval_senior T20I") -> Dict:
    """Do IPL and T20I performers get systematically different ratings?"""
    out = {}
    have_ipl = agg[agg.get(hp_ipl).notna()] if hp_ipl in agg.columns else agg.iloc[0:0]
    have_t20i = agg[agg.get(hp_t20i).notna()] if hp_t20i in agg.columns else agg.iloc[0:0]
    out["mean_rating_players_with_ipl_hp"] = round(
        float(have_ipl["rating_primary"].mean()), 3) if len(have_ipl) else None
    out["mean_rating_players_with_t20i_hp"] = round(
        float(have_t20i["rating_primary"].mean()), 3) if len(have_t20i) else None
    if hp_ipl in agg.columns and hp_t20i in agg.columns:
        both = agg[agg[hp_ipl].notna() & agg[hp_t20i].notna()]
        if len(both) >= 10:
            out["n_players_in_both"] = int(len(both))
            out["spearman_ipl_vs_t20i_value"] = round(float(
                spearmanr(both[hp_ipl], both[hp_t20i]).statistic), 3)
    return out
