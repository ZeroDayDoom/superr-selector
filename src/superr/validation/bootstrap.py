"""Match-level cluster bootstrap for ranking uncertainty (>=1000 draws).

Resamples MATCHES with replacement per draw, recomputes each player's HP
mean value from their resampled matches, re-ranks the eligible pool, and
records each player's rank distribution and top-k probabilities.

Implementation is vectorised: per-draw work is O(matches + player-match
pairs) with numpy bincounts, so 1,000+ draws finish in seconds. A draw in
which a player's matches are never picked contributes no rank observation
for that player (matching the "ranked only if sampled" semantics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ranks(
    hp_deliveries: pd.DataFrame,
    value_col: str,
    eligible_ids: set,
    n_boot: int = 1000,
    seed: int = 20260131,
    top_k: tuple = (10, 20),
) -> pd.DataFrame:
    """Full-ranking bootstrap.

    hp_deliveries: one row per HP delivery with player_id / match_id /
    value_col. Each draw resamples matches GLOBALLY (whole ranking resample,
    so rank distributions are comparable across players).
    """
    rng = np.random.default_rng(seed)
    d = hp_deliveries[hp_deliveries["player_id"].isin(eligible_ids)]

    # precompute per-player per-match sums/sizes (one row per player-match)
    pm = (d.groupby(["player_id", "match_id"])[value_col]
           .agg(sum="sum", size="size").reset_index())
    players, pid_codes = np.unique(pm["player_id"].values, return_inverse=True)
    match_ids, mid_codes = np.unique(pm["match_id"].values, return_inverse=True)
    n_matches = len(match_ids)

    sums = pm["sum"].values.astype(float)
    sizes = pm["size"].values.astype(float)

    rank_counts = {p: np.zeros(max(top_k), dtype=int) for p in players}
    ranks_all = {p: [] for p in players}
    means_all = {p: np.full(n_boot, np.nan) for p in players}

    n_players = len(players)
    for b in range(n_boot):
        picks = rng.integers(0, n_matches, n_matches)
        counts_per_match = np.bincount(picks, minlength=n_matches)
        row_counts = counts_per_match[mid_codes]  # draws of each player-match
        p_tot = np.bincount(pid_codes, weights=sums * row_counts,
                            minlength=n_players)
        p_cnt = np.bincount(pid_codes, weights=row_counts,
                            minlength=n_players)
        sampled = p_cnt > 0
        if not sampled.any():
            continue
        mean = np.zeros(n_players)
        mean[sampled] = p_tot[sampled] / p_cnt[sampled]
        # rank only the sampled players (descending mean); scatter the rank
        # of each player directly from its position in the sorted order
        idx = np.flatnonzero(sampled)
        order = idx[np.argsort(-mean[idx], kind="stable")]
        ranks = np.empty(n_players, dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        for pid_code in order:
            p = players[pid_code]
            r = int(ranks[pid_code])
            ranks_all[p].append(r)
            means_all[p][b] = mean[pid_code]
            for k in top_k:
                if r <= k:
                    rank_counts[p][k - 1] += 1

    rows = []
    for p in players:
        r = np.array(ranks_all[p])
        m = means_all[p]
        m_finite = m[np.isfinite(m)]
        rows.append({
            "player_id": p,
            "boot_mean_score": float(np.mean(m_finite)) if len(m_finite) else np.nan,
            "boot_median_score": float(np.median(m_finite)) if len(m_finite) else np.nan,
            "boot_ci_lo": float(np.percentile(m_finite, 2.5)) if len(m_finite) else np.nan,
            "boot_ci_hi": float(np.percentile(m_finite, 97.5)) if len(m_finite) else np.nan,
            "rank_p05": float(np.percentile(r, 5)) if len(r) else np.nan,
            "rank_p50": float(np.percentile(r, 50)) if len(r) else np.nan,
            "rank_p95": float(np.percentile(r, 95)) if len(r) else np.nan,
            "p_top10": rank_counts[p][9] / n_boot,
            "p_top20": rank_counts[p][19] / n_boot,
        })
    return pd.DataFrame(rows)
