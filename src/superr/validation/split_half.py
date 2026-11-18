"""Split-half reliability of the ranking (repeated random match halves)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def split_half_ranking_reliability(
    hp: pd.DataFrame, value_col: str, eligible_ids: set,
    n_splits: int = 100, seed: int = 20260131,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    matches = hp["match_id"].unique()
    d = hp[hp["player_id"].isin(eligible_ids)]
    counts = d.groupby("player_id")["match_id"].nunique()
    players = counts[counts >= 6].index
    d = d[d["player_id"].isin(players)]

    rhos, taus, r_corrs, overlaps10, overlaps20 = [], [], [], [], []
    for _ in range(n_splits):
        half = set(rng.choice(matches, size=len(matches) // 2, replace=False))
        m1 = d[d["match_id"].isin(half)].groupby("player_id")[value_col].mean()
        m2 = d[~d["match_id"].isin(half)].groupby("player_id")[value_col].mean()
        common = m1.index.intersection(m2.index)
        if len(common) < 20:
            continue
        rhos.append(spearmanr(m1[common], m2[common]).statistic)
        taus.append(kendalltau(m1[common], m2[common]).statistic)
        r_corrs.append(np.corrcoef(m1[common], m2[common])[0, 1])
        t1 = set(m1.sort_values(ascending=False).head(10).index)
        t2 = set(m2.sort_values(ascending=False).head(10).index)
        overlaps10.append(len(t1 & t2))
        t1b = set(m1.sort_values(ascending=False).head(20).index)
        t2b = set(m2.sort_values(ascending=False).head(20).index)
        overlaps20.append(len(t1b & t2b))
    if not rhos:
        return {}
    return {
        "n_splits": n_splits,
        "spearman_mean": round(float(np.mean(rhos)), 3),
        "kendall_mean": round(float(np.mean(taus)), 3),
        "score_corr_mean": round(float(np.mean(r_corrs)), 3),
        "top10_overlap_mean": round(float(np.mean(overlaps10)), 2),
        "top20_overlap_mean": round(float(np.mean(overlaps20)), 2),
    }
