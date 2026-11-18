"""Second-innings (chase) pressure components (0-1 each, pre-delivery)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

MODULE = "chase_pressure"

COMPONENT_FORMULAS: Dict[str, str] = {
    "difficulty": "0.45*clip(RRR/12,0,1) + 0.35*clip((RRR-CRR)/6,0,1) + 0.20*clip(runs_required/(6*balls_remaining),0,1)",
    "resource_stress": "1 - resources_remaining",
    "collapse_pressure": "0.6*min(1, wk12/3) + 0.4*min(1, wk6/2)",
    "momentum_stress": "0.5*clip(consecutive_dots/12) + 0.5*clip(balls_since_boundary/18)",
    "phase_pressure": "powerplay 0.3 / middle 0.4 / death 1.0",
    "match_importance": "knockout 1.0 / league 0.2",
    "boundary_burden": "clip(required_boundaries_estimate / max(1, balls_remaining/6), 0, 1)",
}


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(0, 1)


def add_chase_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    over_idx = (df["legal_balls_before"] // 6).astype("int64") + 1
    wk6 = df["last6_wickets"].fillna(0)
    wk12 = df["last12_wickets"].fillna(0)

    rrr = df["required_run_rate"].fillna(0.0)
    gap = df["rrr_crr_gap"].fillna(0.0)
    balls = df["legal_balls_remaining"].replace(0, np.nan)
    per_over_need = df["runs_required_before"] / (balls * 6.0).clip(lower=1)

    diff = (
        0.45 * _clip01(rrr / 12.0)
        + 0.35 * _clip01(gap / 6.0)
        + 0.20 * _clip01(per_over_need)
    )
    res = 1.0 - df["resources_remaining"]
    collapse = 0.6 * np.minimum(1.0, wk12 / 3.0) + 0.4 * np.minimum(1.0, wk6 / 2.0)
    momentum = (
        0.5 * _clip01(df["consecutive_dots_before"] / 12.0)
        + 0.5 * _clip01(df["balls_since_last_boundary"] / 18.0)
    )
    phase = np.select([over_idx <= 6, over_idx <= 15], [0.3, 0.4], default=1.0)
    importance = np.where(df["is_knockout"].astype(bool), 1.0, 0.2)
    burden = _clip01(df["required_boundaries_estimate"] / (df["overs_remaining"] + 0.5))

    df["ch_difficulty"] = diff
    df["ch_resource_stress"] = res
    df["ch_collapse_pressure"] = collapse
    df["ch_momentum_stress"] = momentum
    df["ch_phase_pressure"] = phase
    df["ch_match_importance"] = importance
    df["ch_boundary_burden"] = burden
    return df
