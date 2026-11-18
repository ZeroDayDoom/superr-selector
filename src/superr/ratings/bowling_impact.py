"""Bowling impact: per-delivery value over expectation for the bowler.

Bowler Value (BoV):
    BoV = (xConceded - conceded)                    [runs prevented]
          + lambda(state) * (bowler_wicket - p_bowler_wicket)   [wicket value]

Only audit-attributed bowler wickets count (run-outs / retired-out /
obstructing are in `other_wicket` and never credited). Dots and boundary
prevention are tracked as evidence components; realised WPA enters with the
sign flipped (bowling-side value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_bowling_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bowl_runs_prevented"] = (
        df["expected_bowler_conceded_runs"] - df["bowler_runs_conceded"]
    )
    is_bw = df["is_bowler_wicket"].astype("boolean").fillna(False).astype(float)
    df["bowl_wickets_above_expected"] = is_bw - df["expected_bowler_wicket"]
    df["bowl_wicket_value"] = df["lambda_state"] * df["bowl_wickets_above_expected"]
    df["bowl_value"] = df["bowl_runs_prevented"] + df["bowl_wicket_value"]

    df["bowl_dot_impact"] = (
        df["is_dot_ball"].astype(float) - df["expected_dot_prob"]
    )
    df["bowl_boundary_prevention"] = (
        df["expected_boundary_prob"] - df["is_boundary"].astype(float)
    )
    df["bowl_wpa"] = -df["realized_wpa"]  # bowling-side perspective
    df["bowl_expected_state_value"] = -df["leverage_expected_dir_swing"]
    return df
