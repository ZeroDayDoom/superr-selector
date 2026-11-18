"""Delivery leverage engine.

Leverage = expected absolute win-probability swing of a delivery, computed
from an outcome-probability model and the win-probability models:

  leverage(state) = sum_o P(outcome=o | state) * |WP(after state, o) - WP(state)|

The current delivery's outcome is NEVER used: the outcome model provides
probabilities, and the "after" states are hypothetical transitions.
Realized WPA is computed separately (post-delivery evidence).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

OUTCOME_BUCKETS = ["dot", "single", "two_three", "four", "six", "wicket", "other"]

# runs added and legal-ball/wicket effects per bucket (hypothetical transitions)
BUCKET_EFFECT = {
    "dot":       {"runs": 0, "legal": 1, "wicket": 0},
    "single":    {"runs": 1, "legal": 1, "wicket": 0},
    "two_three": {"runs": 2, "legal": 1, "wicket": 0},
    "four":      {"runs": 4, "legal": 1, "wicket": 0},
    "six":       {"runs": 6, "legal": 1, "wicket": 0},
    "wicket":    {"runs": 0, "legal": 1, "wicket": 1},
    "other":     {"runs": 1, "legal": 0, "wicket": 0},  # wide/no-ball type extra
}

LEVERAGE_FEATURES = [
    "legal_balls_remaining", "wickets_in_hand", "over_number_before",
    "ball_in_over_before", "score_before", "expected_final_score",
    "score_vs_par", "required_run_rate", "current_run_rate", "rrr_crr_gap",
    "target_progress", "venue_season_par", "is_ipl", "is_knockout",
    "batter_balls_faced_before", "batter_is_set", "resources_remaining",
]

OUTCOME_FEATURES = [
    "over_number_before", "ball_in_over_before", "legal_balls_remaining",
    "wickets_in_hand", "score_before", "score_vs_par", "expected_final_score",
    "required_run_rate", "current_run_rate", "rrr_crr_gap", "venue_season_par",
    "is_ipl", "is_knockout", "batter_balls_faced_before", "batter_is_set",
    "batter_dismissal_risk", "pace_or_spin", "batter_vs_pace_spin",
    "resources_remaining", "last6_wickets", "last12_runs", "consecutive_dots_before",
]


def _outcome_label(row: pd.Series) -> str:
    if bool(row["is_team_wicket"]):
        return "wicket"
    if bool(row["is_four"]):
        return "four"
    if bool(row["is_six"]):
        return "six"
    ex = row["extras_type"]
    if isinstance(ex, str) and ex in ("wide", "no_ball", "penalty", "other", "unknown"):
        return "other"
    r = float(row["total_runs"]) - float(row["extras_runs"])
    if r >= 2:
        return "two_three"
    if r == 1:
        return "single"
    return "dot"


def train_outcome_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """Multiclass outcome-probability model from pre-delivery features."""
    d = df.copy()
    if "is_ipl" not in d.columns:
        d["is_ipl"] = d["competition_class"].eq("IPL").astype(int)
    d["is_knockout"] = d["knock_out"].astype("boolean").fillna(False).astype(int)
    d["batter_is_set"] = d["batter_is_set"].astype(bool).astype(int)
    d["pace_or_spin"] = d["pace_or_spin"].astype("string").fillna("unknown").astype("category")
    d["batter_vs_pace_spin"] = d["batter_vs_pace_spin"].astype("string").fillna("unknown").astype("category")
    d["batter_dismissal_risk"] = d["striker_dismissal_risk_career"].fillna(0.0)
    d["_bucket"] = d.apply(_outcome_label, axis=1)

    d["_season_year"] = pd.to_numeric(
        d["season"].astype("string").str.extract(r"(\d{4})")[0], errors="coerce"
    )
    train = d[d["_season_year"] <= 2024]
    X = train[OUTCOME_FEATURES]
    y = train["_bucket"]

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.08, num_leaves=31,
        min_child_samples=300, reg_lambda=1.0, random_state=20260131, verbose=-1,
    )
    model.fit(X, y, categorical_feature=["pace_or_spin", "batter_vs_pace_spin"])
    model.bucket_order_ = list(model.classes_)  # type: ignore[attr-defined]
    return model


def outcome_probabilities(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in ("pace_or_spin", "batter_vs_pace_spin"):
        if str(X[c].dtype) == "string":
            X[c] = X[c].astype("category")
    probs = model.predict_proba(X[OUTCOME_FEATURES])
    return pd.DataFrame(probs, columns=model.bucket_order_, index=X.index)  # type: ignore[attr-defined]


def _successor_features(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Hypothetical post-outcome features for the WP models."""
    eff = BUCKET_EFFECT[bucket]
    s = df.copy()
    s["score_before"] = s["score_before"] + eff["runs"]
    s["legal_balls_before"] = s["legal_balls_before"] + eff["legal"]
    s["wickets_before"] = s["wickets_before"] + eff["wicket"]
    s["wickets_in_hand"] = (10 - s["wickets_before"]).clip(lower=0)
    sched = s["scheduled_legal_balls"].replace(0, np.nan)
    s["resources_remaining"] = (
        (s["legal_balls_remaining"] - eff["legal"]).clip(lower=0) / sched
    ).clip(0, 1) * (s["wickets_in_hand"] / 10.0)
    # chase quantities
    s["legal_balls_remaining"] = (s["legal_balls_remaining"] - eff["legal"]).clip(lower=0)
    s["runs_required_before"] = (
        s["target"].astype("float64") - s["score_before"]
    ).clip(lower=0)
    denom = s["legal_balls_remaining"].replace(0, np.nan) / 6.0
    s["required_run_rate"] = s["runs_required_before"] / denom
    s["current_run_rate"] = np.where(
        s["legal_balls_before"] > 0,
        s["score_before"] / (s["legal_balls_before"] / 6.0),
        0.0,
    )
    s["rrr_crr_gap"] = s["required_run_rate"] - s["current_run_rate"]
    s["target_progress"] = s["score_before"] / s["target"].astype("float64")
    # first-innings quantities
    s["expected_final_score"] = s["expected_final_score"] + eff["runs"]
    s["score_vs_par"] = s["score_vs_par"] + eff["runs"]
    return s


def wp_after(
    df: pd.DataFrame,
    bucket: str,
    wp2,
    wp1,
) -> np.ndarray:
    """Win probability for the batting side after a hypothetical outcome."""
    s = _successor_features(df, bucket)
    p2 = np.full(len(s), np.nan)
    p1 = np.full(len(s), np.nan)
    chase_mask = s["innings"].eq(2).values
    inn1_mask = ~chase_mask
    if wp2 is not None and chase_mask.any():
        p2[chase_mask] = wp2.predict(s[chase_mask])
    if wp1 is not None and inn1_mask.any():
        p1[inn1_mask] = wp1.predict(s[inn1_mask])

    # terminal states
    p2 = np.where(s["runs_required_before"] <= 0, 1.0, p2)
    p2 = np.where(s["wickets_in_hand"] <= 0, 0.0, p2)
    # innings over with runs still required -> batting side has lost
    p2 = np.where(
        (s["legal_balls_remaining"] <= 0) & (s["runs_required_before"] > 0), 0.0, p2
    )
    is_chase = s["innings"].eq(2).values
    return np.where(is_chase, p2, p1)


def compute_leverage(
    df: pd.DataFrame,
    outcome_model: lgb.LGBMClassifier,
    wp2,
    wp1,
) -> pd.DataFrame:
    """Expected absolute WP swing (leverage) and directional expectation."""
    d = df.copy()
    if "is_ipl" not in d.columns:
        d["is_ipl"] = d["competition_class"].eq("IPL").astype(int)
    d["is_knockout"] = d["knock_out"].astype("boolean").fillna(False).astype(int)
    d["batter_is_set"] = d["batter_is_set"].astype(bool).astype(int)
    d["batter_dismissal_risk"] = d["striker_dismissal_risk_career"].fillna(0.0)
    d["pace_or_spin"] = d["pace_or_spin"].astype("string").fillna("unknown").astype("category")
    d["batter_vs_pace_spin"] = d["batter_vs_pace_spin"].astype("string").fillna("unknown").astype("category")

    if wp2 is not None:
        wp_before_chase = np.full(len(d), np.nan)
        chase_mask = d["innings"].eq(2).values
        if chase_mask.any():
            wp_before_chase[chase_mask] = wp2.predict(d[chase_mask])
    else:
        wp_before_chase = np.full(len(d), np.nan)
    if wp1 is not None:
        wp_before_inn1 = np.full(len(d), np.nan)
        inn1_mask = d["innings"].eq(1).values
        if inn1_mask.any():
            wp_before_inn1[inn1_mask] = wp1.predict(d[inn1_mask])
    else:
        wp_before_inn1 = np.full(len(d), np.nan)
    is_chase = d["innings"].eq(2).values
    wp_before = np.where(is_chase, wp_before_chase, wp_before_inn1)

    # terminal chase states
    rr = d["runs_required_before"]
    wi = d["wickets_in_hand"]
    term_win = rr.notna() & rr.le(0)
    term_lose = wi.notna() & wi.le(0)
    wp_before = np.where(term_win, 1.0, wp_before)
    wp_before = np.where(term_lose, 0.0, wp_before)

    probs = outcome_probabilities(outcome_model, d)
    exp_abs = np.zeros(len(d))
    exp_dir = np.zeros(len(d))
    for bucket in OUTCOME_BUCKETS:
        p_o = probs[bucket].values
        wp_after_o = wp_after(d, bucket, wp2, wp1)
        delta = wp_after_o - wp_before
        exp_abs += p_o * np.abs(delta)
        exp_dir += p_o * delta
    out = pd.DataFrame({
        "wp_before": wp_before,
        "leverage_expected_abs_swing": exp_abs,
        "leverage_expected_dir_swing": exp_dir,
    }, index=df.index)
    return out


def realized_wpa(
    df: pd.DataFrame,
    wp2,
    wp1,
) -> pd.Series:
    """Realized WPA of each delivery's actual outcome (post-delivery evidence).

    WPA = WP(after actual outcome) - WP(before). Uses the same transition
    machinery with the realised bucket; used only for validation/reporting.
    """
    d = df.copy()
    buckets = d.apply(_outcome_label, axis=1)
    if "is_ipl" not in d.columns:
        d["is_ipl"] = d["competition_class"].eq("IPL").astype(int)
    d["is_knockout"] = d["knock_out"].astype("boolean").fillna(False).astype(int)
    d["batter_is_set"] = d["batter_is_set"].astype(bool).astype(int)
    d["batter_dismissal_risk"] = d["striker_dismissal_risk_career"].fillna(0.0)
    d["pace_or_spin"] = d["pace_or_spin"].astype("string").fillna("unknown").astype("category")
    d["batter_vs_pace_spin"] = d["batter_vs_pace_spin"].astype("string").fillna("unknown").astype("category")

    wp_before_chase = np.full(len(d), np.nan)
    wp_before_inn1 = np.full(len(d), np.nan)
    chase_mask = d["innings"].eq(2).values
    inn1_mask = d["innings"].eq(1).values
    if wp2 is not None and chase_mask.any():
        wp_before_chase[chase_mask] = wp2.predict(d[chase_mask])
    if wp1 is not None and inn1_mask.any():
        wp_before_inn1[inn1_mask] = wp1.predict(d[inn1_mask])
    wp_before = np.where(chase_mask, wp_before_chase, wp_before_inn1)

    wp_afters = {b: wp_after(d, b, wp2, wp1) for b in OUTCOME_BUCKETS}
    wpa = np.array([
        wp_afters[b][i] - wp_before[i] for i, b in enumerate(buckets)
    ])
    return pd.Series(wpa, index=df.index, name="realized_wpa")
