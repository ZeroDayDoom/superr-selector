"""Batting impact: per-delivery value over expectation for the striker.

Batter Value (BV) on a delivery:
    BV = (runs_off_bat - xRuns_off_bat)
         - lambda(state) * (striker_dismissed - p_wicket)

lambda(state) is the DYNAMIC run-value of losing a wicket, estimated from
the leave-one-innings-out total-projection curves:
    lambda(balls_left_bin, w) = E[remaining | w] - E[remaining | w+1]
so a wicket costs more when resources (balls x wickets) are plentiful and
less when the innings is nearly over. Byes/leg-byes never enter (off-bat
runs only). Wicket attribution uses `dismissed_is_striker`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- dynamic wicket value ---------------------------------------------------

def build_lambda_table(balls: pd.DataFrame) -> pd.DataFrame:
    """Dynamic wicket value per (innings, balls_left_over_bin, wickets_before).

    Estimated from the LOO expected-runs-remaining curves in balls_clean:
    the value of a wicket is the expected remaining runs lost by moving from
    w to w+1 wickets down, within the same balls-left bin.
    """
    df = balls[balls["innings"].isin([1, 2])].copy()
    df["_bl_bin"] = (df["legal_balls_remaining"] // 6).clip(lower=0)
    curve = (
        df.groupby(["innings", "_bl_bin", "wickets_before"], as_index=False)[
            "expected_runs_remaining"]
        .mean()
        .rename(columns={"expected_runs_remaining": "mean_remaining"})
    )
    rows = []
    for (inn, bl), grp in curve.groupby(["innings", "_bl_bin"]):
        g = grp.set_index("wickets_before")["mean_remaining"]
        for w in range(0, 10):
            if w in g.index and (w + 1) in g.index:
                rows.append({"innings": inn, "bl_bin": bl, "wickets": w,
                             "lambda_runs": max(0.0, float(g[w] - g[w + 1]))})
    lam = pd.DataFrame(rows)
    if lam.empty:
        lam = pd.DataFrame({"innings": [1, 2], "bl_bin": [0, 0],
                            "wickets": [0, 0], "lambda_runs": [0.0, 0.0]})
    # fill missing bins: interpolate per innings on wickets, then global mean
    global_mean = float(lam["lambda_runs"].mean())
    lam = (
        lam.set_index(["innings", "bl_bin", "wickets"])["lambda_runs"]
    )
    return lam, global_mean


def attach_lambda(df: pd.DataFrame, lam: pd.Series,
                  global_mean: float) -> pd.Series:
    key = pd.MultiIndex.from_arrays([
        df["innings"],
        (df["legal_balls_remaining"] // 6).clip(lower=0),
        df["wickets_before"],
    ])
    return lam.reindex(key).values if hasattr(lam, "reindex") else None


def add_batting_value(df: pd.DataFrame, lam_series: pd.Series,
                      global_mean: float) -> pd.DataFrame:
    """Add per-delivery batting value columns (pre-delivery expectations +
    post-delivery outcomes; the value itself is a post-delivery metric)."""
    df = df.copy()
    lam = lam_series.reindex(pd.MultiIndex.from_arrays([
        df["innings"].values,
        (df["legal_balls_remaining"] // 6).clip(lower=0).values,
        df["wickets_before"].values,
    ])).values
    lam = np.where(np.isnan(lam), global_mean, lam)
    df["lambda_state"] = lam

    striker_dismissed = df["dismissed_is_striker"].astype("boolean").fillna(False)
    df["bat_runs_above_expected"] = df["batter_runs"] - df["expected_runs_off_bat"]
    df["bat_wicket_loss"] = df["lambda_state"] * (
        striker_dismissed.astype(float) - df["p_wicket"]
    )
    df["bat_value"] = df["bat_runs_above_expected"] - df["bat_wicket_loss"]

    # component contributions (evidence columns)
    df["bat_boundary_contribution"] = df["is_boundary"].astype(float) - df["p_boundary"]
    df["bat_dot_contribution"] = (
        (~df["is_dot_ball"].astype(bool)).astype(float) - (1.0 - df["p_dot"])
    )
    # match-state value: realised WPA (batting side) and its expectation proxy
    df["bat_wpa"] = df["realized_wpa"]
    df["bat_expected_state_value"] = df["leverage_expected_dir_swing"]
    return df
