"""Scenario robustness: scenario-restricted rankings and specificity index."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def scenario_specificity(deliveries: pd.DataFrame, scenario_cols: List[str],
                         value_col: str, player_col: str = "player_id",
                         min_balls: int = 30) -> pd.DataFrame:
    """Per player: value per scenario + specificity (dispersion across
    scenarios). High specificity = ranking driven by a narrow situation."""
    rows = []
    for pid, grp in deliveries.groupby(player_col):
        base = {sc.replace("scenario_", ""): np.nan for sc in scenario_cols}
        n = 0
        for sc in scenario_cols:
            sub = grp[grp[sc].astype(bool)]
            if len(sub) >= min_balls:
                base[sc.replace("scenario_", "")] = float(sub[value_col].mean())
                n += len(sub)
        vals = [v for v in base.values() if np.isfinite(v)]
        if len(vals) >= 3:
            spread = float(np.max(vals) - np.min(vals))
            best = max(base, key=lambda k: (base[k] if np.isfinite(base[k]) else -9e9))
            worst = min(base, key=lambda k: (base[k] if np.isfinite(base[k]) else 9e9))
        else:
            spread, best, worst = np.nan, None, None
        rows.append({"player_id": pid, "n_scenario_balls": n,
                     "scenario_spread": spread,
                     "best_scenario": best, "worst_scenario": worst, **base})
    return pd.DataFrame(rows)


def scenario_restricted_top10(deliveries: pd.DataFrame, scenario_col: str,
                              value_col: str, player_name_col: str,
                              min_balls: int = 40) -> pd.DataFrame:
    sub = deliveries[deliveries[scenario_col].astype(bool)]
    g = sub.groupby(player_col).agg(
        player_name=(player_name_col, "first"),
        balls=(value_col, "size"),
        value_mean=(value_col, "mean"))
    g = g[g["balls"] >= min_balls].sort_values("value_mean", ascending=False)
    g.insert(0, "rank", np.arange(1, len(g) + 1))
    return g.reset_index()
