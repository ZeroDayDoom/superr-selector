"""Ranking stability aggregation + validation charts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stability_summary(tables: Dict[str, pd.DataFrame],
                      primary: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    base = primary.set_index("player_id")["rank"]
    rows = []
    for name, t in tables.items():
        if "rank" not in t.columns:
            continue
        v = t.set_index("player_id")["rank"]
        common = base.index.intersection(v.index)
        if len(common) < 10:
            continue
        rho = spearmanr(base[common], v[common]).statistic
        top10_base = set(base[common].sort_values().head(10).index)
        top10_var = set(v[common].sort_values().head(10).index)
        rows.append({"variant": name, "n": int(len(common)),
                     "spearman": round(float(rho), 3),
                     "top10_overlap": int(len(top10_base & top10_var))})
    return pd.DataFrame(rows)


def make_charts(bootstrap: pd.DataFrame, names: pd.Series,
                threshold_ranks: Dict[str, pd.DataFrame],
                trend: pd.DataFrame, comp_ranks: Dict[str, pd.DataFrame],
                role_adv: Dict, ablation: pd.DataFrame,
                figures: Path) -> None:
    plt.rcParams.update({"figure.dpi": 120, "font.size": 8})

    # 1. rank distribution (P05-P95) for top-20 by P50
    b = bootstrap.dropna(subset=["rank_p50"]).nsmallest(20, "rank_p50")
    if len(b):
        fig, ax = plt.subplots(figsize=(6, 5))
        y = np.arange(len(b))
        names_map = names.to_dict()
        labels = [str(names_map.get(p, p))[:16] for p in b["player_id"]]
        ax.hlines(y, b["rank_p05"], b["rank_p95"], color="#2b6099", lw=3)
        ax.plot(b["rank_p50"], y, "o", color="#c8102e")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Rank (P05-P95, 1000-draw bootstrap)")
        ax.set_title("Rank uncertainty — 20 most reliably ranked players")
        fig.tight_layout()
        fig.savefig(figures / "rank_distribution.png")
        plt.close(fig)

        # 2. top-10 probability
        fig, ax = plt.subplots(figsize=(6, 4))
        b2 = b.head(15)
        ax.barh([str(names_map.get(p, p))[:16] for p in b2["player_id"]],
                b2["p_top10"], color="#c8102e")
        ax.invert_yaxis()
        ax.set_xlabel("P(Top 10) across 1000 bootstrap draws")
        ax.set_title("Top-10 probability")
        fig.tight_layout()
        fig.savefig(figures / "top10_probability.png")
        plt.close(fig)

    # 3. threshold sensitivity: rank of a fixed reference set across thresholds
    if threshold_ranks:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, t in threshold_ranks.items():
            sub = t.head(10)
            ax.plot([name] * len(sub), sub["rank"], "o", alpha=0.6, label=name)
        ax.set_ylabel("rank within variant")
        ax.set_title("Top-10 membership across HP thresholds")
        fig.tight_layout()
        fig.savefig(figures / "threshold_sensitivity.png")
        plt.close(fig)

    # 4. season stability: distribution of per-season HP value trends
    if len(trend):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for pid, grp in list(trend.groupby("player_id"))[:40]:
            if len(grp) >= 2:
                ax.plot(grp["season_year"], grp["hp_value"], alpha=0.25,
                        color="#2b6099")
        ax.set_xlabel("Season"); ax.set_ylabel("HP value mean")
        ax.set_title("Season stability (sample of players)")
        fig.tight_layout()
        fig.savefig(figures / "season_stability.png")
        plt.close(fig)

    # 5. IPL vs T20I scatter
    if "ipl_only" in comp_ranks and "t20i_only" in comp_ranks:
        i = comp_ranks["ipl_only"].set_index("player_id")["hp_value_mean"]
        t = comp_ranks["t20i_only"].set_index("player_id")["hp_value_mean"]
        common = i.index.intersection(t.index)
        if len(common) >= 10:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(i[common], t[common], s=12, alpha=0.6, color="#2b6099")
            lims = [min(i[common].min(), t[common].min()),
                    max(i[common].max(), t[common].max())]
            ax.plot(lims, lims, "--", color="grey")
            ax.set_xlabel("IPL-only HP value"); ax.set_ylabel("T20I-only HP value")
            ax.set_title(f"IPL vs T20I (n={len(common)} players qualified in both)")
            fig.tight_layout()
            fig.savefig(figures / "ipl_vs_t20i.png")
            plt.close(fig)

    # 6. role sensitivity
    if role_adv.get("role_means"):
        fig, ax = plt.subplots(figsize=(5.5, 3))
        rm = role_adv["role_means"]
        ax.bar(list(rm.keys()), list(rm.values()), color="#2b6099")
        ax.set_ylabel("mean composite rating")
        ax.set_title("Mean rating by role (systematic-advantage check)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(figures / "role_sensitivity.png")
        plt.close(fig)

    # 7. score confidence intervals for top players
    # 8. ablation impact
    if len(ablation) and "spearman" in ablation.columns:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(ablation["variant"], ablation["spearman"], color="#2b6099")
        lo = min(0.8, float(ablation["spearman"].min()) - 0.05)
        ax.set_ylim(lo, 1.0)
        ax.set_ylabel("Spearman vs primary")
        ax.set_title("Ablation: ranking impact of removing each component")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(figures / "model_ablation.png")
        plt.close(fig)
