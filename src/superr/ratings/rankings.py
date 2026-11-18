"""Ranking assembly: main / role / scenario / competition / facet rankings,
duplicate checks, and top-10 player dossiers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def rank_table(agg: pd.DataFrame, score_col: str, ascending: bool = False,
               extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = extra_cols or [c for c in agg.columns if c not in ("player_name",)]
    t = agg[["player_id", "player_name"] + [c for c in cols
                                            if c in agg.columns]].copy()
    t = t.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


def check_no_duplicates(table: pd.DataFrame, id_col: str = "player_id") -> bool:
    return not table[id_col].duplicated().any()


def scenario_rankings(deliveries: pd.DataFrame, scenario_cols: List[str],
                      value_col: str, min_balls: int = 40) -> Dict[str, pd.DataFrame]:
    """Per-scenario player value rankings (min HP balls within scenario)."""
    out = {}
    for sc in scenario_cols:
        sub = deliveries[deliveries[sc].astype(bool)]
        if len(sub) == 0:
            continue
        g = sub.groupby("player_id").agg(
            player_name=("player_name", "first"),
            balls=(value_col, "size"),
            value_mean=(value_col, "mean"),
        )
        g = g[g["balls"] >= min_balls].sort_values("value_mean", ascending=False)
        g.insert(0, "rank", np.arange(1, len(g) + 1))
        out[sc] = g.reset_index()
    return out


def dossier_markdown(row: pd.Series, rank: int, role: str,
                     match_values: pd.DataFrame, season_trend: pd.Series,
                     comp_split: pd.Series, scenario_strengths: Dict,
                     scenario_weaknesses: Dict) -> str:
    lines = []
    a = lines.append
    a(f"# Player Dossier — {row['player_name']} (Rank {rank}, {role})")
    a("")
    a(f"- **Final score:** {row['rating_primary']:.3f} "
      f"(z-composite; pool mean = 0)")
    a(f"- **Reliability:** {row['hp_balls']:.0f} HP balls, "
      f"{row['hp_matches']:.0f} HP matches, {row['n_seasons']:.0f} seasons; "
      f"bootstrap 90% CI of HP value [{row['boot_lo']:.3f}, {row['boot_hi']:.3f}]")
    a(f"- **Level (EB-shrunk LCB):** {row['level_lcb']:.3f} "
      f"(raw HP mean {row['shrunk_k']:.3f})")
    a(f"- **Pressure uplift:** raw {row['uplift_raw']:.3f}, "
      f"context-adjusted {row['uplift_adjusted']:.3f} "
      f"(normal baseline {row['normal_mean']:.3f})")
    a(f"- **Consistency:** positive-match share {row['positive_match_pct']:.1%}, "
      f"match sd {row['match_std']:.3f}")
    a(f"- **Downside:** CVaR5 {row['cvar5']:.3f}, "
      f"negative-match rate {row['negative_impact_rate']:.1%}")
    a("")
    a("## Pressure sample")
    a(f"- HP strike rate/economy metrics: see all_player_metrics.csv row; "
      f"HP value mean {row['hp_value_mean']:.3f} over {row['hp_balls']:.0f} balls")
    a(f"- Extreme-pressure value mean: "
      f"{row.get('extreme_value_mean', float('nan')):.3f} "
      f"over {row.get('extreme_balls', 0):.0f} balls")
    a("")
    a("## Scenario strengths (top) / weaknesses (bottom)")
    a("")
    a("- Strengths: " + ", ".join(f"{k} ({v:+.3f})" for k, v in
                                  scenario_strengths.items()))
    a("- Weaknesses: " + ", ".join(f"{k} ({v:+.3f})" for k, v in
                                   scenario_weaknesses.items()))
    a("")
    a("## Competition split (HP value mean)")
    a("")
    for k, v in comp_split.items():
        a(f"- {k}: {v:+.3f}")
    a("")
    a("## Season trend (HP value mean by season)")
    a("")
    for k, v in season_trend.items():
        a(f"- {k}: {v:+.3f}")
    a("")
    if len(match_values):
        best = match_values.loc[match_values["hp_match_value"].idxmax()]
        worst = match_values.loc[match_values["hp_match_value"].idxmin()]
        a(f"## Best / worst pressure matches")
        a(f"- Best: match {int(best['match_id'])} ({best['hp_match_value']:+.3f})")
        a(f"- Worst: match {int(worst['match_id'])} ({worst['hp_match_value']:+.3f})")
        a("")
    a("## Why this player ranks here")
    a("")
    a(f"- The composite is dominated by the player's level under pressure "
      f"(shrunk lower-confidence-bound), weighted 0.45, plus consistency "
      f"(0.20), context-adjusted uplift (0.20), volume (0.15) and downside "
      f"(−0.10). All components are z-scores across the eligible pool.")
    a("")
    a("## What could make this ranking wrong")
    a("")
    a("- Small HP samples: the bootstrap CI above quantifies this; a player "
      "near the eligibility floor could move several places within the CI.")
    a("- The pressure definition (Model C hybrid, WP-based) could "
      "misclassify states; the multiverse analysis in the methodology report "
      "covers threshold and weight sensitivity.")
    a("- Matchup/opponent strength is adjusted through the expectation "
      "models; a bowler who faced weak opposition line-ups could be "
      "overrated if the two-pass quality terms under-correct.")
    a("- Clutch persistence in sport is modest; some of the observed uplift "
      "may be noise despite shrinkage.")
    return "\n".join(lines)
