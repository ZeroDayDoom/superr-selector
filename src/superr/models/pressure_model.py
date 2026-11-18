"""Composite pressure models A / B / C, categories, and thresholds.

Components (0-1, per innings type) are assembled into three scores:

  MODEL A (transparent rules): non-WP components only.
  MODEL B (WP-based):          closeness + leverage + situational difficulty.
  MODEL C (hybrid):            B + A + resource stress.

Weights are calibrated on the validation season (2025) against a
pre-registered composite criterion (see calibrate_weights):
  (1) outcome-variance ratio HP vs rest (outcome spread)
  (2) realized-|WPA| ratio HP vs rest (stakes)
  (3) split-half stability of player HP means (rating reliability)
Model selection uses the same criterion; categories are distribution-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MODEL_A_COMPONENTS = ["difficulty", "resource_stress", "collapse_pressure",
                      "momentum_stress", "phase_pressure", "match_importance"]
MODEL_A_WEIGHTS_START = {"difficulty": 0.30, "resource_stress": 0.15,
                         "collapse_pressure": 0.20, "momentum_stress": 0.15,
                         "phase_pressure": 0.10, "match_importance": 0.10}

MODEL_B_COMPONENTS = ["match_closeness", "leverage_norm", "difficulty"]
MODEL_B_WEIGHTS_START = {"match_closeness": 0.40, "leverage_norm": 0.35,
                         "difficulty": 0.25}

MODEL_C_WEIGHTS_START = {"B": 0.55, "A": 0.30, "resource_stress": 0.15}


def assemble_components(df: pd.DataFrame) -> pd.DataFrame:
    """Build the unified component set for every delivery."""
    df = df.copy()
    is_chase = df["innings"].eq(2)
    for comp in MODEL_A_COMPONENTS:
        df[f"pc_{comp}"] = np.where(
            is_chase, df[f"ch_{comp}"], df[f"fi_{comp}"]
        )
    if "acceleration_deficit" in df.columns:
        df["pc_acceleration_deficit"] = np.where(
            is_chase, np.nan, df["fi_acceleration_deficit"]
        )
    if "boundary_burden" in df.columns:
        df["pc_boundary_burden"] = np.where(is_chase, df["ch_boundary_burden"], np.nan)

    # WP-derived components (added by the runner after WP/leverage scoring)
    if "wp_before" in df.columns:
        df["pc_match_closeness"] = 1.0 - (2.0 * df["wp_before"] - 1.0).abs()
    if "leverage_expected_abs_swing" in df.columns:
        # fixed-constant normalization (no future information): a swing of
        # 0.12+ in WP is near the physical maximum for a single delivery
        li = df["leverage_expected_abs_swing"]
        df["pc_leverage_norm"] = (li / 0.12).clip(0, 1)
    return df


def _weighted(components: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    total_w = sum(weights.values())
    out = None
    for name, w in weights.items():
        term = components[f"pc_{name}"].fillna(0.0) * (w / total_w)
        out = term if out is None else out + term
    return out.clip(0, 1)


def build_scores(df: pd.DataFrame,
                 weights_a: Optional[Dict[str, float]] = None,
                 weights_b: Optional[Dict[str, float]] = None,
                 weights_c: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    df = df.copy()
    wa = weights_a or MODEL_A_WEIGHTS_START
    wb = weights_b or MODEL_B_WEIGHTS_START
    wc = weights_c or MODEL_C_WEIGHTS_START
    df["PRESSURE_MODEL_A"] = _weighted(df, wa)
    df["PRESSURE_MODEL_B"] = _weighted(df, wb)
    df["PRESSURE_MODEL_C"] = (
        wc["B"] * df["PRESSURE_MODEL_B"] + wc["A"] * df["PRESSURE_MODEL_A"]
        + wc["resource_stress"] * df["pc_resource_stress"].fillna(0.0)
    ).clip(0, 1)
    df["PRESSURE_DISPLAY_100"] = (df["PRESSURE_MODEL_C"] * 100).round(1)
    return df


# ---------------------------------------------------------------------------
# data-driven weight calibration on the validation season
# ---------------------------------------------------------------------------

def _validation_metrics(df: pd.DataFrame, score_col: str,
                        hp_quantile: float = 0.80) -> Dict[str, float]:
    """Pre-registered criterion on a scored frame."""
    d = df[df[score_col].notna()]
    thr = d[score_col].quantile(hp_quantile)
    hp = d[d[score_col] >= thr]
    rest = d[d[score_col] < thr]

    # (1) outcome variance ratio (runs per delivery)
    var_hp = hp["total_runs"].var()
    var_rest = rest["total_runs"].var()
    var_ratio = float(var_hp / var_rest) if var_rest > 0 else 0.0

    # (2) realized |WPA| ratio
    if "realized_wpa" in d.columns:
        wpa_hp = hp["realized_wpa"].abs().mean()
        wpa_rest = rest["realized_wpa"].abs().mean()
        wpa_ratio = float(wpa_hp / wpa_rest) if wpa_rest and wpa_rest > 0 else 0.0
    else:
        wpa_ratio = 0.0

    # (3) split-half stability of player HP means (batters)
    stability = float(split_half_stability(d, score_col, hp_quantile))
    return {"variance_ratio": var_ratio, "wpa_swing_ratio": wpa_ratio,
            "split_half_stability": stability}


def split_half_stability(df: pd.DataFrame, score_col: str,
                         hp_quantile: float = 0.80,
                         n_splits: int = 30, seed: int = 20260131) -> float:
    """Mean correlation of player HP means across random match halves."""
    rng = np.random.default_rng(seed)
    matches = df["match_id"].unique()
    d = df[df[score_col].notna()]
    hp_thr = d[score_col].quantile(hp_quantile)
    players = d.groupby("striker_id")["match_id"].nunique()
    players = players[players >= 4].index
    d = d[d["striker_id"].isin(players)]
    if len(d) < 2000 or len(players) < 20:
        return np.nan
    corrs = []
    for _ in range(n_splits):
        half = set(rng.choice(matches, size=len(matches) // 2, replace=False))
        h1 = d[d["match_id"].isin(half)]
        h2 = d[~d["match_id"].isin(half)]
        m1 = h1[h1[score_col] >= h1[score_col].quantile(hp_quantile)] \
            .groupby("striker_id")["total_runs"].mean()
        m2 = h2[h2[score_col] >= h2[score_col].quantile(hp_quantile)] \
            .groupby("striker_id")["total_runs"].mean()
        common = m1.index.intersection(m2.index)
        if len(common) < 15:
            continue
        c = np.corrcoef(m1[common], m2[common])[0, 1]
        if np.isfinite(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan


def calibrate_weights(
    valid_df: pd.DataFrame,
) -> Dict[str, Dict[str, object]]:
    """Grid-search B and C weights on the validation season.

    A's weights stay at literature-informed starting values (they are the
    transparent-rule definition; their sensitivity is covered by the C-grid
    mixing weight). B's three weights vary on a 0.05 grid summing to 1;
    C's mix varies over B in {0.45..0.70}.
    """
    grid_b: List[Dict[str, float]] = []
    for w_close in np.arange(0.25, 0.56, 0.05):
        for w_lev in np.arange(0.20, 0.51, 0.05):
            w_diff = round(1.0 - w_close - w_lev, 2)
            if 0.10 <= w_diff <= 0.45:
                grid_b.append({"match_closeness": round(w_close, 2),
                               "leverage_norm": round(w_lev, 2),
                               "difficulty": w_diff})

    results_b = []
    for wb in grid_b:
        scored = build_scores(valid_df, weights_b=wb)
        m = _validation_metrics(scored, "PRESSURE_MODEL_B")
        results_b.append({"weights": wb, **m})
    # normalise criterion components to [0,1]-ish and average
    def score_of(m: Dict[str, float]) -> float:
        return (
            min(m["variance_ratio"], 2.0) / 2.0
            + min(m["wpa_swing_ratio"], 2.0) / 2.0
            + (m["split_half_stability"] if np.isfinite(m["split_half_stability"]) else 0.0)
        )
    best_b = max(results_b, key=score_of)

    results_c = []
    for w_B in np.arange(0.45, 0.71, 0.05):
        wc = {"B": round(w_B, 2), "A": round(1 - w_B - 0.10, 2),
              "resource_stress": 0.10}
        scored = build_scores(valid_df, weights_b=best_b["weights"], weights_c=wc)
        m = _validation_metrics(scored, "PRESSURE_MODEL_C")
        results_c.append({"weights": wc, **m})
    best_c = max(results_c, key=score_of)

    scored_a = build_scores(valid_df)
    metrics_a = _validation_metrics(scored_a, "PRESSURE_MODEL_A")

    return {
        "model_a": {"weights": MODEL_A_WEIGHTS_START, "metrics": metrics_a},
        "model_b": {"weights": best_b["weights"], "metrics": {k: v for k, v in best_b.items() if k != "weights"}},
        "model_c": {"weights": best_c["weights"], "metrics": {k: v for k, v in best_c.items() if k != "weights"}},
        "grid_b": results_b,
        "grid_c": results_c,
    }


# ---------------------------------------------------------------------------
# categories and percentile labels
# ---------------------------------------------------------------------------

def add_categories(df: pd.DataFrame, score_col: str = "PRESSURE_MODEL_C") -> pd.DataFrame:
    """Distribution-based categories + stratified percentile labels."""
    df = df.copy()
    s = df[score_col]
    q50, q75, q90 = s.quantile(0.50), s.quantile(0.75), s.quantile(0.90)
    df["pressure_category"] = np.select(
        [s <= q50, s <= q75, s <= q90],
        ["LOW", "MODERATE", "HIGH"],
        default="EXTREME",
    )
    # stratified high-pressure labels (within innings x phase strata)
    stratum = df["innings"].astype(str) + "_" + df["innings_phase"].astype(str)
    for q in (0.90, 0.85, 0.80, 0.75, 0.70):
        pct = int(round((1 - q) * 100))
        thr = df.groupby(stratum)[score_col].transform(lambda x: x.quantile(q))
        df[f"hp_top{pct}"] = (df[score_col] >= thr).astype(int)
    # overall labels for comparison
    for q in (0.90, 0.85, 0.80, 0.75, 0.70):
        pct = int(round((1 - q) * 100))
        df[f"hp_overall_top{pct}"] = (df[score_col] >= df[score_col].quantile(q)).astype(int)
    return df
