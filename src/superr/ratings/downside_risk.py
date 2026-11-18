"""Downside risk metrics over HIGH-PRESSURE MATCH means.

All metrics are computed on a player's HP-match value distribution, so a
player with few matches is NOT penalised beyond the uncertainty that the
bootstrap/shrinkage layers already capture (the risk metrics are reported
and enter the composite with a small weight, sensitivity-tested).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def downside_metrics(match_values: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = match_values.groupby("player_id")[value_col]
    out = pd.DataFrame({
        "risk_mean": g.mean(),
        "risk_n_matches": g.size(),
    })

    def _downside_dev(x: pd.Series) -> float:
        neg = x[x < x.mean()]
        if len(neg) == 0:
            return 0.0
        return float(np.sqrt(np.mean((x.mean() - neg) ** 2)))

    def _cvar5(x: pd.Series) -> float:
        if len(x) == 0:
            return np.nan
        q = x.quantile(0.05)
        tail = x[x <= q]
        return float(tail.mean()) if len(tail) else float(q)

    out["downside_deviation"] = g.apply(_downside_dev)
    out["worst10pct_match_avg"] = g.apply(_cvar5)
    out["cvar5"] = out["worst10pct_match_avg"]
    out["catastrophic_rate"] = match_values.assign(
        cat=(match_values[value_col] < -2.0).astype(float)
    ).groupby("player_id")["cat"].mean()
    out["negative_impact_rate"] = match_values.assign(
        neg=(match_values[value_col] < 0).astype(float)
    ).groupby("player_id")["neg"].mean()
    return out.reset_index()
