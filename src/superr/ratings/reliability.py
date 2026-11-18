"""Reliability: bootstrap confidence intervals (match-level clusters)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_player_mean(
    deliveries: pd.DataFrame, value_col: str, n_boot: int = 200,
    seed: int = 20260131,
) -> pd.DataFrame:
    """Per-player cluster bootstrap over that player's MATCHES.

    Each draw resamples the player's own matches with replacement (preserving
    within-match correlation) and recomputes the mean value. Returns mean,
    SE, and 2.5/97.5 percentiles per player.
    """
    rng = np.random.default_rng(seed)
    out = []
    for pid, grp in deliveries.groupby("player_id", sort=False):
        matches = grp["match_id"].unique()
        v = grp[value_col].values
        mids = grp["match_id"].values
        if len(matches) < 2:
            out.append({"player_id": pid, "boot_mean": float(np.mean(v)),
                        "boot_se": np.nan, "boot_lo": np.nan, "boot_hi": np.nan})
            continue
        sums = pd.Series(v).groupby(mids).agg(["sum", "size"])
        stats = np.empty(n_boot)
        for b in range(n_boot):
            picks = rng.choice(len(sums), size=len(sums), replace=True)
            stats[b] = sums["sum"].values[picks].sum() / sums["size"].values[picks].sum()
        out.append({
            "player_id": pid,
            "boot_mean": float(np.mean(stats)),
            "boot_se": float(np.std(stats)),
            "boot_lo": float(np.percentile(stats, 2.5)),
            "boot_hi": float(np.percentile(stats, 97.5)),
        })
    return pd.DataFrame(out).set_index("player_id")


def lcb(mean: float, se: float, z: float = 0.85) -> float:
    """Lower confidence bound: mean - z*SE (z=0.85 ~= one-sided 80%)."""
    if not np.isfinite(se):
        return mean
    return mean - z * se
