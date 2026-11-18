"""Composite rating: primary formula + pre-registered sensitivity variants.

Primary composite (Methodology §4.1, WUSA):
    Rating = 0.45*z(Level_LCB) + 0.20*z(Consistency) + 0.20*z(Uplift)
             + 0.15*z(Volume) - 0.10*z(|Downside|)

where Level_LCB = EB-shrunk HP mean value - 0.85*bootstrap SE, Consistency =
positive-HP-match share, Uplift = context-adjusted residual, Volume =
log HP balls x scenario diversity, Downside = |CVaR5 of HP-match means|.

Sensitivity variants re-weight the same z-scored components (no component is
chosen or dropped based on player outcomes); rank stability across variants
is reported. Nothing here references player names.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

PRIMARY_WEIGHTS = {
    "level_lcb": 0.45, "consistency": 0.20, "uplift": 0.20,
    "volume": 0.15, "downside": -0.10,
}
# canonical source column for each composite component in the player
# aggregation table (see reports/player_rating_methodology.md):
#   consistency -> positive-HP-match share
#   uplift      -> context-adjusted residual of HP mean on Normal mean
#   downside    -> absolute CVaR5 of HP-match means
COMPONENT_SOURCES = {
    "level_lcb": "level_lcb",
    "consistency": "positive_match_pct",
    "uplift": "uplift_adjusted",
    "volume": "volume",
    "downside": "downside_abs",
}
VARIANTS: Dict[str, Dict[str, float]] = {
    "level_heavy": {"level_lcb": 0.60, "consistency": 0.15, "uplift": 0.15,
                    "volume": 0.10, "downside": -0.05},
    "balanced": {"level_lcb": 0.35, "consistency": 0.25, "uplift": 0.25,
                 "volume": 0.15, "downside": -0.10},
    "uplift_heavy": {"level_lcb": 0.30, "consistency": 0.20, "uplift": 0.35,
                     "volume": 0.10, "downside": -0.05},
    "risk_averse": {"level_lcb": 0.40, "consistency": 0.25, "uplift": 0.15,
                    "volume": 0.10, "downside": -0.25},
}


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.std() == 0 or not np.isfinite(s.std()):
        return s * 0.0
    return (s - s.mean()) / s.std()


def composite(df: pd.DataFrame, weights: Dict[str, float],
              prefix: str = "rating") -> pd.Series:
    """Weighted z-composite.

    A player with an UNDEFINED component (e.g. no normal-pressure
    deliveries makes uplift undefined) contributes the pool mean for that
    component (z = 0): neutral by construction, never a penalty — and the
    total stays finite so every downstream rank comparison is well-defined.
    """
    total = 0.0
    for comp, w in weights.items():
        col = COMPONENT_SOURCES.get(comp, comp)
        if col not in df.columns and comp in df.columns:
            col = comp  # caller supplied the component under its own name
        z = zscore(df[col])
        if not np.isfinite(z).all():
            z = z.fillna(0.0)
        total = total + w * z
    return total.rename(prefix)


def add_composites(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()
    df["downside_abs"] = df["cvar5"].abs().fillna(df["cvar5"].abs().median())
    df["rating_primary"] = composite(df, PRIMARY_WEIGHTS, "rating_primary")
    for name, w in VARIANTS.items():
        df[f"rating_{name}"] = composite(df, w, f"rating_{name}")
    return df
