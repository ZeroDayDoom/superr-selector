"""Eligibility: three tiers derived from the sample distribution, plus audit.

Thresholds are chosen from the actual HP-balls distribution (percentiles of
qualified-candidate samples) and the reliability analysis, then fixed per
tier:
  inclusive: >= 40 HP balls and >= 6 HP matches
  moderate:  >= 60 HP balls and >= 8 HP matches
  strict:    >= 100 HP balls and >= 12 HP matches
Every player receives a row in the audit with the reason, sample size and
threshold — nobody is silently excluded.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TIERS: Dict[str, Dict[str, int]] = {
    "inclusive": {"hp_balls": 40, "hp_matches": 6},
    "moderate": {"hp_balls": 60, "hp_matches": 8},
    "strict": {"hp_balls": 100, "hp_matches": 12},
}


def derive_thresholds_from_distribution(agg: pd.DataFrame) -> Dict[str, Dict]:
    """Report the sample distribution that informed the tier choices."""
    hp = agg["hp_balls"].clip(lower=0)
    return {
        "hp_balls_quantiles": hp.quantile([0.5, 0.75, 0.9, 0.95]).round(1).to_dict(),
        "players_above_40_balls": int((hp >= 40).sum()),
        "players_above_60_balls": int((hp >= 60).sum()),
        "players_above_100_balls": int((hp >= 100).sum()),
    }


def eligibility_audit(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in agg.iterrows():
        row = {
            "player_id": r["player_id"],
            "player_name": r.get("player_name"),
            "role": r.get("role"),
            "hp_balls": r["hp_balls"],
            "hp_matches": r["hp_matches"],
            "n_seasons": r.get("n_seasons"),
            "normal_balls": r.get("normal_balls"),
        }
        for tier, th in TIERS.items():
            ok = (r["hp_balls"] >= th["hp_balls"]
                  and r["hp_matches"] >= th["hp_matches"])
            reasons = []
            if r["hp_balls"] < th["hp_balls"]:
                reasons.append(f"hp_balls {r['hp_balls']} < {th['hp_balls']}")
            if r["hp_matches"] < th["hp_matches"]:
                reasons.append(f"hp_matches {r['hp_matches']} < {th['hp_matches']}")
            row[f"eligible_{tier}"] = bool(ok)
            row[f"reason_{tier}"] = "eligible" if ok else "; ".join(reasons)
        rows.append(row)
    out = pd.DataFrame(rows)
    # tier label: highest tier the player clears
    out["tier"] = np.select(
        [out["eligible_strict"], out["eligible_moderate"], out["eligible_inclusive"]],
        ["strict", "moderate", "inclusive"], default="ineligible",
    )
    return out
