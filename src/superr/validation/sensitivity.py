"""Sensitivity engine: re-rank from stored per-player components under
alternative specifications (thresholds, pressure models, competition scope,
eligibility tiers, shrinkage strengths, composite weights, ablations).

All variants recombine PRE-COMPUTED, pre-delivery-derived components; no
variant is selected post hoc to favour any player, and every variant's
full ranking is retained for the stability report.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from superr.ratings import composite_rating as cr
from superr.ratings import reliability as rel


def rerank(df: pd.DataFrame, score_builder, eligibility_mask: pd.Series,
           name: str) -> pd.DataFrame:
    """Build score, filter, rank; return (player_id, rank) table."""
    d = df[eligibility_mask].copy()
    d[name] = score_builder(d)
    d = d.sort_values(name, ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    return d[["player_id", "player_name", name, "rank"]]


def threshold_sensitivity(base_agg: pd.DataFrame, variants: Dict[str, pd.DataFrame],
                          eligible: set) -> Dict[str, pd.DataFrame]:
    """variants: name -> agg table computed with that HP threshold."""
    out = {}
    base_scores = base_agg.set_index("player_id")["rating_primary"]
    for name, agg in variants.items():
        m = agg["player_id"].isin(eligible)
        r = rerank(agg, lambda d: d["level_lcb"], m, f"score_{name}")
        common = r[r["player_id"].isin(base_agg["player_id"])]
        rho = spearmanr(
            base_agg.set_index("player_id").reindex(common["player_id"])["rating_primary"],
            common.set_index("player_id")["score_" + name],
        ).statistic
        r.attrs["spearman_vs_primary"] = float(rho) if np.isfinite(rho) else np.nan
        out[name] = r
    return out


def weight_sensitivity(agg: pd.DataFrame, eligible_mask: pd.Series,
                       variants: Dict[str, Dict[str, float]]) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, weights in variants.items():
        m = eligible_mask
        d = agg[m].copy()
        d[f"rating_{name}"] = cr.composite(d, weights, f"rating_{name}")
        r = d.sort_values(f"rating_{name}", ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        out[name] = r[["player_id", "player_name", f"rating_{name}", "rank"]]
    return out


def ablation_sensitivity(agg: pd.DataFrame, eligible_mask: pd.Series,
                         primary: Dict[str, float]) -> Dict[str, pd.DataFrame]:
    """Zero out one component at a time and re-rank."""
    out = {}
    for comp in ["level_lcb", "consistency", "uplift", "volume", "downside"]:
        weights = dict(primary)
        weights[comp] = 0.0
        d = agg[eligible_mask].copy()
        d[f"rating_no_{comp}"] = cr.composite(d, weights, f"rating_no_{comp}")
        r = d.sort_values(f"rating_no_{comp}", ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        out[f"ablate_{comp}"] = r[["player_id", "player_name",
                                   f"rating_no_{comp}", "rank"]]
    return out


def shrinkage_sensitivity(agg: pd.DataFrame, eligible_mask: pd.Series,
                          shrunk_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Rank with different shrinkage strengths feeding Level_LCB."""
    out = {}
    boot_se = agg.set_index("player_id")["boot_se"]
    for col in shrunk_cols:
        d = agg[eligible_mask].copy()
        d["_lcb"] = [rel.lcb(m, s) for m, s in zip(d[col], d["boot_se"])]
        d[f"rating_{col}"] = cr.composite(
            d.assign(level_lcb=d["_lcb"]), cr.PRIMARY_WEIGHTS, f"rating_{col}")
        r = d.sort_values(f"rating_{col}", ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        out[col] = r[["player_id", "player_name", f"rating_{col}", "rank"]]
    return out


def rank_stability_table(variant_tables: Dict[str, pd.DataFrame],
                         primary: pd.DataFrame) -> pd.DataFrame:
    """Spearman + top-10 overlap of every variant vs the primary ranking."""
    base = primary.set_index("player_id")["rank"]
    rows = []
    for name, t in variant_tables.items():
        v = t.set_index("player_id")["rank"]
        common = base.index.intersection(v.index)
        if len(common) < 10:
            continue
        rho = spearmanr(base[common], v[common]).statistic
        top10_base = set(base[common].sort_values().head(10).index)
        top10_var = set(v[common].sort_values().head(10).index)
        rows.append({"variant": name, "n_common": int(len(common)),
                     "spearman_vs_primary": round(float(rho), 3),
                     "top10_overlap": len(top10_base & top10_var)})
    return pd.DataFrame(rows)


def most_and_least_robust(bootstrap_table: pd.DataFrame, top_n: int = 10):
    b = bootstrap_table.dropna(subset=["rank_p05", "rank_p95"])
    robust = b.nsmallest(top_n, "rank_p95")[["player_id", "rank_p05",
                                             "rank_p50", "rank_p95", "p_top10"]]
    uncertain = b.nlargest(top_n, "rank_p95")[["player_id", "rank_p05",
                                               "rank_p50", "rank_p95", "p_top10"]]
    return robust, uncertain
