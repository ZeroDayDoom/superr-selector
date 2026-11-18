"""Bowling expected performance: contextual outcome + conceded-runs models.

Bucket set splits wickets by attribution so bowler-wicket probability is a
first-class output: {dot, single, two_three, four, six, bowler_wicket,
other_wicket (run-out/retired-out/obstructing), other_extras}.
Bowler-conceded runs follow the audit definition (off-bat + wides + no-balls;
byes/leg-byes excluded) and the expected-conceded mapping is estimated
empirically from training data per bucket.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

from superr.models.expected_runs import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    chronological_split,
)

BOWL_BUCKETS = ["dot", "single", "two_three", "four", "six",
                "bowler_wicket", "other_wicket", "other_extras"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def bowling_bucket(row: pd.Series) -> str:
    if bool(row["is_team_wicket"]):
        return "bowler_wicket" if bool(row["is_bowler_wicket"]) else "other_wicket"
    if bool(row["is_four"]):
        return "four"
    if bool(row["is_six"]):
        return "six"
    ex = row["extras_type"]
    if isinstance(ex, str) and ex in ("wide", "no_ball", "penalty", "other", "unknown"):
        return "other_extras"
    r = float(row["total_runs"]) - float(row["extras_runs"])
    if r >= 2:
        return "two_three"
    if r == 1:
        return "single"
    return "dot"


def empirical_conceded_map(train: pd.DataFrame) -> Dict[str, float]:
    """Mean bowler-conceded runs per bucket, estimated on training data."""
    b = train.apply(bowling_bucket, axis=1)
    tmp = pd.DataFrame({"b": b, "rc": train["bowler_runs_conceded"]})
    m = tmp.groupby("b")["rc"].mean().to_dict()
    return {k: float(v) for k, v in m.items()}


def expected_conceded_from_buckets(probs: pd.DataFrame,
                                   conceded_map: Dict[str, float]) -> pd.Series:
    total = 0.0
    for bucket in BOWL_BUCKETS:
        total = total + probs[bucket] * conceded_map.get(bucket, 0.0)
    return total


def train_lgbm_bowling(train: pd.DataFrame) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.07, num_leaves=31,
        min_child_samples=300, reg_lambda=1.0, random_state=20260131, verbose=-1,
    )
    model.fit(train[FEATURES], train["_bowl_bucket"],
              categorical_feature=CATEGORICAL_FEATURES)
    return model


def bowling_logistic(train: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=1500, C=1.0,
                                               random_state=20260131))])
    pipe.fit(train[FEATURES], train["_bowl_bucket"])
    return pipe
