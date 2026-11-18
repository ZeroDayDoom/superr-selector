"""Temporal validation: train-era rankings vs later-era performance.

Folds (chronological, no random splits):
  rank on 2022-23 -> evaluate on 2024-26
  rank on 2022-24 -> evaluate on 2025-26
  rank on 2022-25 -> evaluate on 2026
Persistence = Spearman correlation of era-1 HP value with era-2 HP value
for players qualified in BOTH eras, plus top-15 overlap.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def era_value(hp: pd.DataFrame, value_col: str,
              max_year: int, min_year: int = 2022,
              min_balls: int = 40) -> pd.Series:
    d = hp[(hp["season_year"] >= min_year) & (hp["season_year"] <= max_year)]
    g = d.groupby("player_id").agg(mean=(value_col, "mean"),
                                   balls=(value_col, "size"))
    return g[g["balls"] >= min_balls]["mean"]


def temporal_folds(hp: pd.DataFrame, value_col: str,
                   folds: List = None, min_balls: int = 40) -> Dict[str, Dict]:
    if folds is None:
        folds = [((2022, 2023), (2024, 2026)),
                 ((2022, 2024), (2025, 2026)),
                 ((2022, 2025), (2026, 2026))]
    results = {}
    for (tr_lo, tr_hi), (ev_lo, ev_hi) in folds:
        v1 = era_value(hp, value_col, tr_hi, tr_lo, min_balls=min_balls)
        v2 = era_value(hp, value_col, ev_hi, ev_lo, min_balls=min_balls)
        common = v1.index.intersection(v2.index)
        if len(common) < 10:
            continue
        rho = spearmanr(v1[common], v2[common]).statistic
        tau = kendalltau(v1[common], v2[common]).statistic
        top1 = set(v1.sort_values(ascending=False).head(15).index)
        top2 = set(v2.sort_values(ascending=False).head(15).index)
        results[f"rank_{tr_lo}-{tr_hi}_eval_{ev_lo}-{ev_hi}"] = {
            "n_players_both_eras": int(len(common)),
            "spearman": round(float(rho), 3),
            "kendall": round(float(tau), 3),
            "top15_overlap": len(top1 & top2),
            "top15_jaccard": round(len(top1 & top2) / len(top1 | top2), 3),
        }
    return results


def season_trend_table(hp: pd.DataFrame, value_col: str,
                       min_balls: int = 30) -> pd.DataFrame:
    """Per-player per-season HP value for trend classification."""
    t = (hp.groupby(["player_id", "season_year"])[value_col]
         .agg(["mean", "size"]).reset_index())
    t = t[t["size"] >= min_balls]
    t = t.rename(columns={"mean": "hp_value", "size": "hp_balls"})
    return t


def classify_trend(trend: pd.DataFrame, player_id: int) -> Dict:
    d = trend[trend["player_id"] == player_id].sort_values("season_year")
    if len(d) < 2:
        return {"trend": "insufficient_seasons", "n_seasons": int(len(d))}
    vals = d["hp_value"].values
    slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
    span = float(vals.max() - vals.min())
    if span > 0.35 and (vals.argmax() == len(vals) - 1 or vals.argmin() == len(vals) - 1):
        label = "single_season_dominance"
    elif slope > 0.05:
        label = "improving"
    elif slope < -0.05:
        label = "declining"
    else:
        label = "consistent"
    return {"trend": label, "slope": round(slope, 3),
            "n_seasons": int(len(d)), "first": round(float(vals[0]), 3),
            "last": round(float(vals[-1]), 3)}
