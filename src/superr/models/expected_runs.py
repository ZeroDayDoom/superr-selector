"""Expected batting performance: contextual outcome models.

The league baseline predicts what an AVERAGE player would do in a context —
player identity is never a feature. Outcomes use mutually exclusive buckets:
{dot, single, two_three, four, six, wicket, other}; every derived expected
metric (xRuns, P(dot), P(boundary), ...) is computed from the same bucket
distribution, so the probabilities are consistent by construction.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from superr.models.leverage_model import _outcome_label

BUCKETS = ["dot", "single", "two_three", "four", "six", "wicket", "other"]
BUCKET_RUNS = {"dot": 0.0, "single": 1.0, "two_three": 2.5, "four": 4.0,
               "six": 6.0, "wicket": 0.0, "other": 0.0}

NUMERIC_FEATURES = [
    "over_number_before", "ball_in_over_before", "legal_balls_remaining_norm",
    "wickets_in_hand", "score_norm", "current_run_rate", "required_run_rate",
    "rrr_crr_gap", "pressure_score", "leverage", "venue_season_par_norm",
    "resources_remaining", "batter_balls_faced_before", "batter_runs_before",
    "batter_is_new", "batter_is_set", "is_knockout", "last12_runs",
    "last12_wickets", "last12_boundaries", "consecutive_dots_before",
    "balls_since_last_boundary", "season_year", "is_chase",
]
CATEGORICAL_FEATURES = [
    "batting_type", "pace_or_spin", "bowling_style_group",
    "competition_class", "innings_phase",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def season_to_year(season: pd.Series) -> pd.Series:
    s = season.astype("string").str.extract(r"(\d{4})")[0].fillna("")
    return pd.to_numeric(s, errors="coerce").astype("Float64")


def build_feature_frame(
    balls: pd.DataFrame, pressure: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Model-ready frame: one row per delivery, pre-delivery features only."""
    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
    df = balls.merge(
        pressure[key + ["PRESSURE_MODEL_C", "leverage_expected_abs_swing",
                        "wp_before"]],
        on=key, how="left", validate="one_to_one",
    )
    df = df.merge(
        labels[["comp_id", "competition_class"]], on="comp_id",
        how="left", validate="many_to_one",
    )
    df["season_year"] = season_to_year(df["season"])
    sched = df["scheduled_legal_balls"].replace(0, np.nan)
    df["legal_balls_remaining_norm"] = df["legal_balls_remaining"] / sched
    df["score_norm"] = df["score_before"] / sched.clip(lower=1)
    df["venue_season_par_norm"] = df["venue_season_par"] / 180.0
    df["resources_remaining"] = (
        df["legal_balls_remaining_norm"].clip(0, 1) * (df["wickets_in_hand"] / 10.0)
    )
    df["pressure_score"] = df["PRESSURE_MODEL_C"]
    df["leverage"] = df["leverage_expected_abs_swing"]
    df["batter_is_new"] = df["batter_is_new"].astype(bool).astype(int)
    df["batter_is_set"] = df["batter_is_set"].astype(bool).astype(int)
    df["is_knockout"] = df["knock_out"].astype("boolean").fillna(False).astype(int)
    df["is_chase"] = df["innings"].eq(2).astype(int)
    df["required_run_rate"] = df["required_run_rate"].fillna(0.0)
    df["rrr_crr_gap"] = df["rrr_crr_gap"].fillna(0.0)
    # momentum windows are NaN at innings start; "since last boundary" is NaN
    # before the first boundary. Fill with pre-delivery-safe values: 0 for
    # windows, elapsed legal balls for the boundary drought.
    for c in ("last12_runs", "last12_wickets", "last12_boundaries"):
        df[c] = df[c].fillna(0.0)
    df["balls_since_last_boundary"] = df["balls_since_last_boundary"].fillna(
        df["legal_balls_before"])
    df["consecutive_dots_before"] = df["consecutive_dots_before"].fillna(0)
    df["batting_type"] = df["batting_type"].astype("string").fillna("unknown")
    df["pace_or_spin"] = df["pace_or_spin"].astype("string").fillna("unknown")
    df["bowling_style_group"] = df.get(
        "bowling_style_group",
        pd.Series("unknown", index=df.index),
    ).astype("string")
    over_idx = (df["legal_balls_before"] // 6).astype("int64") + 1
    df["innings_phase"] = np.select([over_idx <= 6, over_idx <= 15],
                                    ["powerplay", "middle"], default="death")
    df["_bucket"] = df.apply(_outcome_label, axis=1)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    return df


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season_year"] <= 2024]
    valid = df[df["season_year"] == 2025]
    test = df[df["season_year"] == 2026]
    assert len(train) and len(valid) and len(test)
    return train, valid, test


def bucket_target_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Empirical bucket distribution overall (league-average baseline)."""
    return train["_bucket"].value_counts(normalize=True)


def expected_from_buckets(probs: pd.DataFrame) -> pd.DataFrame:
    """Derived expected metrics; mutually consistent by construction."""
    out = pd.DataFrame(index=probs.index)
    for b in BUCKETS:
        out[f"p_{b}"] = probs[b]
    out["expected_runs_off_bat"] = sum(
        probs[b] * r for b, r in BUCKET_RUNS.items()
    )
    out["p_dot"] = probs["dot"]
    out["p_single"] = probs["single"]
    out["p_multiple"] = probs["two_three"]
    out["p_boundary"] = probs["four"] + probs["six"]
    out["p_four"] = probs["four"]
    out["p_six"] = probs["six"]
    out["p_wicket"] = probs["wicket"]
    return out


def train_logistic_outcomes(train: pd.DataFrame) -> Tuple[PipelineLike, object]:
    """Statistical baseline: multinomial logistic on standardised features."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=1500, C=1.0,
                                               random_state=20260131))])
    pipe.fit(train[FEATURES], train["_bucket"])
    return pipe


PipelineLike = object


def train_lgbm_outcomes(train: pd.DataFrame) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.07, num_leaves=31,
        min_child_samples=300, reg_lambda=1.0, random_state=20260131, verbose=-1,
    )
    model.fit(train[FEATURES], train["_bucket"],
              categorical_feature=CATEGORICAL_FEATURES)
    return model


def bucket_log_loss(y_true: pd.Series, probs: pd.DataFrame) -> float:
    return float(log_loss(y_true, probs[probs.columns], labels=list(probs.columns)))


def mae_expected_runs(y_bucket: pd.Series, exp_runs: pd.Series) -> float:
    """MAE of expected runs vs realised off-bat runs per delivery."""
    actual = y_bucket.map(BUCKET_RUNS).fillna(0.0)
    return float((actual - exp_runs).abs().mean())
