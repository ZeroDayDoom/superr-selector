"""Empirical-Bayes shrinkage with data-estimated strength.

Method of moments on the match-level means: the between-player variance of
raw means is signal + noise; the noise variance is estimated from the
within-player variance of match means divided by matches. The shrinkage
constant k = noise_var / between_var, applied as:
    shrunk = (n * mean + k * pool_mean) / (n + k)
Multiple strengths (k x {0.5, 1, 2}) are computed so the sensitivity of the
final ranking to shrinkage can be reported.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def estimate_k(match_values: pd.DataFrame, value_col: str,
               min_matches: int = 3) -> Dict[str, float]:
    """Method-of-moments EB strength from match-level means."""
    g = match_values.groupby("player_id")[value_col]
    n = g.size()
    means = g.mean()
    within_var = g.var()
    keep = n >= min_matches
    if keep.sum() < 20:
        return {"k": 50.0, "pool_mean": float(match_values[value_col].mean()),
                "between_var": np.nan, "noise_var": np.nan}
    # noise_var: average of var/matched weight  (1/(n-1) weighting approximation)
    noise_var = float(np.average(within_var[keep] / n[keep].clip(lower=1),
                                 weights=n[keep]))
    between_var = float(means[keep].var() - noise_var)
    between_var = max(between_var, 1e-6)
    k = noise_var / between_var
    return {"k": float(k), "pool_mean": float(match_values[value_col].mean()),
            "between_var": between_var, "noise_var": noise_var}


def shrink(raw_mean: pd.Series, n: pd.Series, k: float,
           pool_mean: float) -> pd.Series:
    return (n * raw_mean + k * pool_mean) / (n + k)


def shrinkage_table(raw_mean: pd.Series, n: pd.Series, k_hat: float,
                    pool_mean: float) -> pd.DataFrame:
    """Shrunk values at k_hat and sensitivity multipliers."""
    out = pd.DataFrame({"raw_mean": raw_mean, "n": n})
    for mult, label in [(0.5, "shrunk_k_half"), (1.0, "shrunk_k"),
                        (2.0, "shrunk_k_double")]:
        out[label] = shrink(raw_mean, n, k_hat * mult, pool_mean)
    return out
