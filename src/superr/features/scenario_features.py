"""Multi-label pressure scenarios.

A delivery can carry several scenario labels simultaneously (e.g. death
overs + high RRR + new batter). Labels are booleans; intensities are
continuous [0,1] aggregates per pressure family. Two labels
(HIGH_LEVERAGE, MATCH_ON_LINE) receive WP-based proxies here and are
replaced by the calibrated win-probability model in Phase 5.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

MODULE = "scenario_features"

SCENARIO_LABELS = [
    "EARLY_COLLAPSE_RESCUE",
    "MIDDLE_ORDER_RESCUE",
    "RAPID_COLLAPSE",
    "ANCHOR_UNDER_PAR",
    "ACCELERATION_REQUIRED",
    "DEATH_OVERS_PRESSURE",
    "HIGH_RRR_CHASE",
    "LOW_WICKET_RESOURCE_CHASE",
    "PARTNERSHIP_REBUILD",
    "FINISHER_PRESSURE",
    "HIGH_LEVERAGE",
    "BOUNDARY_DROUGHT",
    "NEW_BATTER_AFTER_WICKET",
    "FINAL_OVER_PRESSURE",
    "MATCH_ON_LINE",
    # bowling-side labels
    "POWERPLAY_PRESSURE",
    "MIDDLE_OVERS_CONTROL",
    "DEATH_DEFENCE",
    "HIGH_RRR_DEFENCE",
    "LOW_WICKET_ATTACK",
    "WICKET_DROUGHT_BREAKTHROUGH",
    "SET_BATTER_DISMISSAL",
    "NEW_BATTER_PRESSURE",
    "BOUNDARY_SUPPRESSION",
    "FINAL_OVER_DEFENCE",
    "HIGH_LEVERAGE_DELIVERY",
]

_SCENARIO_FORMULAS = {
    "EARLY_COLLAPSE_RESCUE": "over<=6 AND (wickets_before>=2 OR last6_wickets>=1)",
    "MIDDLE_ORDER_RESCUE": "7<=over<=15 AND wickets_before>=4",
    "RAPID_COLLAPSE": "last6_wickets>=2 OR last12_wickets>=3",
    "ANCHOR_UNDER_PAR": "score_vs_par<=-15 AND batter_is_set",
    "ACCELERATION_REQUIRED": "rrr_crr_gap>2 (chase) OR (innings1 AND over>=10 AND current_run_rate<0.85*par_run_rate)",
    "DEATH_OVERS_PRESSURE": "over>=16",
    "HIGH_RRR_CHASE": "chase AND required_run_rate>=10",
    "LOW_WICKET_RESOURCE_CHASE": "chase AND wickets_in_hand<=3",
    "PARTNERSHIP_REBUILD": "partnership_balls_before<=6 AND wickets_before>=2",
    "FINISHER_PRESSURE": "over>=16 AND batter_role in {finisher, middle}",
    "HIGH_LEVERAGE": "match_balance_proxy>=0.6 (WP-based in Phase 5)",
    "BOUNDARY_DROUGHT": "balls_since_last_boundary>=12",
    "NEW_BATTER_AFTER_WICKET": "batter_is_new AND entering_after_wicket",
    "FINAL_OVER_PRESSURE": "legal_balls_remaining<=6",
    "MATCH_ON_LINE": "match_balance_proxy>=0.75 AND (chase OR over>=15) (WP-based in Phase 5)",
    "POWERPLAY_PRESSURE": "over<=6",
    "MIDDLE_OVERS_CONTROL": "7<=over<=15",
    "DEATH_DEFENCE": "over>=16",
    "HIGH_RRR_DEFENCE": "chase AND required_run_rate>=10",
    "LOW_WICKET_ATTACK": "wickets_in_hand<=3",
    "WICKET_DROUGHT_BREAKTHROUGH": "wicket_drought_severity>=0.5",
    "SET_BATTER_DISMISSAL": "batter_is_set AND batter_runs_before>=25",
    "NEW_BATTER_PRESSURE": "batter_is_new",
    "BOUNDARY_SUPPRESSION": "bowler_boundary_streak_before==0 AND balls_since_last_boundary>=8",
    "FINAL_OVER_DEFENCE": "legal_balls_remaining<=6",
    "HIGH_LEVERAGE_DELIVERY": "match_balance_proxy>=0.6",
}

FEATURE_DOCS: Dict[str, dict] = {}
for _lbl in SCENARIO_LABELS:
    FEATURE_DOCS[f"scenario_{_lbl}"] = {
        "definition": f"Scenario label {_lbl} (multi-label boolean)",
        "formula": _SCENARIO_FORMULAS.get(_lbl, "see scenario_features module"),
        "source": MODULE,
        "available_before_delivery": True,
        "leakage_risk": "All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model",
        "missing_handling": "False when inputs unavailable",
        "relevance": "Explicit pressure regimes for ratings, dossiers and scenario diversity",
    }
FEATURE_DOCS["scenario_count"] = {
    "definition": "Number of scenario labels active on the delivery",
    "formula": "sum of all scenario labels",
    "source": MODULE, "available_before_delivery": True,
    "leakage_risk": "None beyond the labels", "missing_handling": "0",
    "relevance": "Scenario stacking intensity",
}
FEATURE_DOCS["scenario_diversity_index"] = {
    "definition": "Per-match entropy of active scenario labels (delivery-level)",
    "formula": "-sum p_i log p_i over scenario labels within the match so far (pre-delivery)",
    "source": MODULE, "available_before_delivery": True,
    "leakage_risk": "Computed over PRIOR deliveries only",
    "missing_handling": "0 at match start",
    "relevance": "Whether the player has faced varied pressure situations",
}


def add_scenario_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    over_idx = (df["legal_balls_before"] // 6).astype("int64") + 1
    is_chase = df["innings"].eq(2)
    wk6 = df["last6_wickets"].fillna(0)
    wk12 = df["last12_wickets"].fillna(0)
    bsb = df["balls_since_last_boundary"].fillna(0)

    lbl: Dict[str, pd.Series] = {
        "EARLY_COLLAPSE_RESCUE": (over_idx <= 6)
        & ((df["wickets_before"] >= 2) | (wk6 >= 1)),
        "MIDDLE_ORDER_RESCUE": over_idx.between(7, 15) & (df["wickets_before"] >= 4),
        "RAPID_COLLAPSE": (wk6 >= 2) | (wk12 >= 3),
        "ANCHOR_UNDER_PAR": df["score_vs_par"].le(-15) & df["batter_is_set"].astype(bool),
        "ACCELERATION_REQUIRED": (
            (is_chase & df["rrr_crr_gap"].gt(2))
            | (~is_chase & (over_idx >= 10)
               & df["current_run_rate"].lt(0.85 * df["par_run_rate"]))
        ),
        "DEATH_OVERS_PRESSURE": over_idx >= 16,
        "HIGH_RRR_CHASE": is_chase & df["required_run_rate"].ge(10),
        "LOW_WICKET_RESOURCE_CHASE": is_chase & df["wickets_in_hand"].le(3),
        "PARTNERSHIP_REBUILD": df["partnership_balls_before"].le(6)
        & df["wickets_before"].ge(2),
        "FINISHER_PRESSURE": (over_idx >= 16)
        & df["batter_role"].isin(["finisher", "middle"]),
        "HIGH_LEVERAGE": df["match_balance_proxy"].ge(0.6),
        "BOUNDARY_DROUGHT": bsb >= 12,
        "NEW_BATTER_AFTER_WICKET": df["batter_is_new"].astype(bool)
        & df["entering_after_wicket"].astype(bool),
        "FINAL_OVER_PRESSURE": df["legal_balls_remaining"] <= 6,
        "MATCH_ON_LINE": df["match_balance_proxy"].ge(0.75)
        & (is_chase | (over_idx >= 15)),
        "POWERPLAY_PRESSURE": over_idx <= 6,
        "MIDDLE_OVERS_CONTROL": over_idx.between(7, 15),
        "DEATH_DEFENCE": over_idx >= 16,
        "HIGH_RRR_DEFENCE": is_chase & df["required_run_rate"].ge(10),
        "LOW_WICKET_ATTACK": df["wickets_in_hand"].le(3),
        "WICKET_DROUGHT_BREAKTHROUGH": df["wicket_drought_severity"].ge(0.5),
        "SET_BATTER_DISMISSAL": df["batter_is_set"].astype(bool)
        & df["batter_runs_before"].ge(25),
        "NEW_BATTER_PRESSURE": df["batter_is_new"].astype(bool),
        "BOUNDARY_SUPPRESSION": df["bowler_boundary_streak_before"].eq(0) & (bsb >= 8),
        "FINAL_OVER_DEFENCE": df["legal_balls_remaining"] <= 6,
        "HIGH_LEVERAGE_DELIVERY": df["match_balance_proxy"].ge(0.6),
    }
    for name, val in lbl.items():
        df[f"scenario_{name}"] = val.fillna(False).astype(bool)

    label_cols = [f"scenario_{n}" for n in SCENARIO_LABELS]
    df["scenario_count"] = df[label_cols].sum(axis=1)

    # scenario diversity experienced SO FAR in the match (pre-delivery):
    # entropy of label frequencies over PRIOR deliveries of the innings
    g = df.groupby(["match_id", "innings"], sort=False)
    counts = g[label_cols].cumsum() - df[label_cols]
    total = counts.sum(axis=1)
    p = counts.div(total.replace(0, np.nan), axis=0)
    ent = -(p * np.log(p.replace(0, np.nan))).sum(axis=1)
    df["scenario_diversity_index"] = ent.fillna(0.0)
    return df


# Phase labels fire on every delivery (context, not events); excluded from
# the provisional pressure flag used for opportunity accounting.
PHASE_CONTEXT_LABELS = {"POWERPLAY_PRESSURE", "MIDDLE_OVERS_CONTROL",
                        "DEATH_DEFENCE"}


def provisional_pressure_flag(df: pd.DataFrame) -> pd.Series:
    """Provisional HP flag from scenarios (Phase-4 placeholder).

    A delivery is provisionally 'high pressure' when at least FIVE non-phase
    scenario labels are active (a quantile-style rule: ~top 31% of
    deliveries; the label union alone covers most cricket and carries no
    information). Phase 5 replaces this with the calibrated
    win-probability pressure index; opportunity metrics computed here are
    therefore provisional and will be recomputed.
    """
    event_labels = [
        f"scenario_{l}" for l in SCENARIO_LABELS if l not in PHASE_CONTEXT_LABELS
    ]
    return df[event_labels].sum(axis=1) >= 5


def build_opportunity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player opportunity accounting (batters and bowlers, one table).

    Uses the provisional scenario-based HP flag; recomputed in Phase 5 with
    the calibrated pressure index.
    """
    df = df.copy()
    df["_hp"] = provisional_pressure_flag(df)
    scen_cols = [f"scenario_{l}" for l in SCENARIO_LABELS]
    hp = df[df["_hp"]]

    out = []
    for role_key, id_col, opp_col, name_col in [
        ("batter", "striker_id", "bowling_team_id", "striker_name"),
        ("bowler", "bowler_id", "batting_team_id", "bowler_name"),
    ]:
        grp = df.groupby(id_col, sort=False)
        hp_flag = df["_hp"].astype(bool)
        agg = pd.DataFrame({
            "total_deliveries": grp.size(),
            "hp_deliveries": grp["_hp"].sum().astype("int64"),
            "normal_deliveries": grp.apply(
                lambda x: int((~x["_hp"].astype(bool)).sum()), include_groups=False
            ).astype("int64"),
            "n_seasons": grp["season"].nunique(),
            "n_opponents": grp[opp_col].nunique(),
            "n_matches": grp["match_id"].nunique(),
        })
        hp_grp = hp.groupby(id_col, sort=False)
        agg["hp_innings"] = (
            hp.drop_duplicates(["match_id", "innings", id_col])
            .groupby(id_col, sort=False).size().reindex(agg.index).fillna(0).astype("int64")
        )
        agg["hp_matches"] = hp_grp["match_id"].nunique().reindex(agg.index).fillna(0).astype("int64")
        agg["n_scenarios_experienced"] = (
            hp_grp[scen_cols].max().sum(axis=1).reindex(agg.index).fillna(0).astype("int64")
        )
        agg["hp_opportunity_rate"] = agg["hp_deliveries"] / agg["total_deliveries"]
        agg["role"] = role_key
        agg["player_name"] = grp[name_col].first()
        out.append(agg.reset_index().rename(columns={id_col: "player_id"}))

    return pd.concat(out, ignore_index=True)[[
        "player_id", "player_name", "role", "total_deliveries",
        "hp_deliveries", "normal_deliveries", "hp_innings", "hp_matches",
        "n_matches", "n_seasons", "n_opponents", "n_scenarios_experienced",
        "hp_opportunity_rate",
    ]]
