"""Match-level consistency metrics (never delivery-level).

Per player, computed over their high-pressure MATCH means of value:
mean, median, share of positive/negative matches, variance, sd, CV, best /
worst / 10th / 90th percentile match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def consistency_metrics(match_values: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """match_values: one row per (player_id, match_id) with mean value."""
    g = match_values.groupby("player_id")[value_col]
    out = pd.DataFrame({
        "match_mean": g.mean(),
        "match_median": g.median(),
        "match_std": g.std(),
        "match_min": g.min(),
        "match_max": g.max(),
        "n_matches": g.size(),
    })
    out["match_var"] = out["match_std"] ** 2
    out["positive_match_pct"] = match_values.assign(
        pos=(match_values[value_col] > 0).astype(float)
    ).groupby("player_id")["pos"].mean()
    out["negative_match_pct"] = 1.0 - out["positive_match_pct"]
    out["match_p10"] = match_values.groupby("player_id")[value_col].quantile(0.10)
    out["match_p90"] = match_values.groupby("player_id")[value_col].quantile(0.90)
    mean_abs = out["match_mean"].abs().clip(lower=1e-6)
    out["match_cv"] = out["match_std"] / mean_abs
    return out.reset_index()


def split_half_stability(match_values: pd.DataFrame, value_col: str,
                         n_splits: int = 100, seed: int = 20260131,
                         min_matches: int = 6) -> float:
    """Correlation of player means across random match halves (signal share)."""
    rng = np.random.default_rng(seed)
    matches = match_values["match_id"].unique()
    counts = match_values.groupby("player_id")["match_id"].nunique()
    players = counts[counts >= min_matches].index
    mv = match_values[match_values["player_id"].isin(players)]
    if len(players) < 20:
        return float("nan")
    corrs = []
    for _ in range(n_splits):
        half = set(rng.choice(matches, size=len(matches) // 2, replace=False))
        m1 = mv[mv["match_id"].isin(half)].groupby("player_id")[value_col].mean()
        m2 = mv[~mv["match_id"].isin(half)].groupby("player_id")[value_col].mean()
        common = m1.index.intersection(m2.index)
        if len(common) < 15:
            continue
        c = np.corrcoef(m1[common], m2[common])[0, 1]
        if np.isfinite(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else float("nan")
