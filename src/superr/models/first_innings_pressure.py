"""First-innings pressure components (0-1 each, strictly pre-delivery).

Built from score-vs-par state, resource depletion, collapse, momentum,
death-over acceleration requirement and match leverage.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

MODULE = "first_innings_pressure"

COMPONENT_FORMULAS: Dict[str, str] = {
    "difficulty": "0.6*clip(-score_vs_par/30, 0, 1) + 0.4*clip((par_run_rate - current_run_rate)/4, 0, 1)",
    "resource_stress": "1 - resources_remaining (balls x wickets proxy)",
    "collapse_pressure": "0.6*min(1, wk12/3) + 0.4*min(1, wk6/2)",
    "momentum_stress": "0.5*clip(consecutive_dots/12) + 0.5*clip(balls_since_boundary/18)",
    "phase_pressure": "powerplay 0.3 / middle 0.4 / death 1.0 (fixed mapping; one of eight components)",
    "match_importance": "knockout 1.0 / league 0.2",
    "acceleration_deficit": "clip((par_run_rate - last6_run_rate)/6, 0, 1) when over >= 10, else 0",
}


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(0, 1)


def add_first_innings_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    over_idx = (df["legal_balls_before"] // 6).astype("int64") + 1
    wk6 = df["last6_wickets"].fillna(0)
    wk12 = df["last12_wickets"].fillna(0)

    # CRR is 0 by definition at ball 0; the par-gap term is only informative
    # once the innings has actually run (>= 12 legal balls)
    crr_term = _clip01((df["par_run_rate"] - df["current_run_rate"]) / 4.0)
    crr_term = crr_term.where(df["legal_balls_before"].ge(12), 0.0)
    diff = 0.6 * _clip01(-df["score_vs_par"] / 30.0) + 0.4 * crr_term
    res = 1.0 - df["resources_remaining"]
    collapse = 0.6 * np.minimum(1.0, wk12 / 3.0) + 0.4 * np.minimum(1.0, wk6 / 2.0)
    momentum = (
        0.5 * _clip01(df["consecutive_dots_before"] / 12.0)
        + 0.5 * _clip01(df["balls_since_last_boundary"] / 18.0)
    )
    phase = np.select([over_idx <= 6, over_idx <= 15], [0.3, 0.4], default=1.0)
    importance = np.where(df["is_knockout"].astype(bool), 1.0, 0.2)
    accel = np.where(
        over_idx >= 10,
        _clip01((df["par_run_rate"] - df["last6_run_rate"]) / 6.0),
        0.0,
    )

    df["fi_difficulty"] = diff
    df["fi_resource_stress"] = res
    df["fi_collapse_pressure"] = collapse
    df["fi_momentum_stress"] = momentum
    df["fi_phase_pressure"] = phase
    df["fi_match_importance"] = importance
    df["fi_acceleration_deficit"] = accel
    return df
