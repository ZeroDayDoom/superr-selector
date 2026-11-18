"""Expected wicket models (batting-side and bowler-attributed).

Bowler wickets exclude run-outs / retired-out / obstructing (audit rules).
Two targets:
  - team wicket probability (context for batting expectations)
  - bowler wicket probability (context for bowling expectations)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from superr.models.expected_runs import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    chronological_split,
)

WICKET_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def league_wicket_rate(train: pd.DataFrame, col: str) -> float:
    return float(train[col].mean())


def train_logistic_wicket(train: pd.DataFrame, target: str):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=1500, C=1.0,
                                               random_state=20260131))])
    pipe.fit(train[WICKET_FEATURES], train[target])
    return pipe


def train_lgbm_wicket(train: pd.DataFrame, target: str) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.07, num_leaves=31,
        min_child_samples=400, reg_lambda=1.0, random_state=20260131, verbose=-1,
    )
    model.fit(train[WICKET_FEATURES], train[target],
              categorical_feature=CATEGORICAL_FEATURES)
    return model


def eval_wicket(y: pd.Series, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(np.mean(p)),
        "mean_actual": float(np.mean(y)),
    }


def train_wicket_models(df: pd.DataFrame, target: str) -> Dict[str, Dict]:
    """Baseline / logistic / LGBM comparison for a wicket target.

    target: 'is_team_wicket' or 'is_bowler_wicket' (audit-attributed).
    """
    train, valid, test = chronological_split(df)
    ytr, yv, yte = train[target].astype(int), valid[target].astype(int), test[target].astype(int)
    models: Dict[str, Dict] = {}

    league_rate = league_wicket_rate(train, target)
    models["league_average"] = {
        "predict": lambda X: np.full(len(X), league_rate),
        "metrics": {
            "validation": eval_wicket(yv, np.full(len(yv), league_rate)),
            "test": eval_wicket(yte, np.full(len(yte), league_rate)),
        },
        "artifact": {"league_rate": league_rate},
    }

    logm = train_logistic_wicket(train, target)
    models["logistic"] = {
        "predict": lambda X: logm.predict_proba(X[WICKET_FEATURES])[:, 1],
        "metrics": {
            "validation": eval_wicket(yv, logm.predict_proba(valid[WICKET_FEATURES])[:, 1]),
            "test": eval_wicket(yte, logm.predict_proba(test[WICKET_FEATURES])[:, 1]),
        },
        "artifact": logm,
    }

    lgbm = train_lgbm_wicket(train, target)
    models["lightgbm"] = {
        "predict": lambda X: lgbm.predict_proba(X[WICKET_FEATURES])[:, 1],
        "metrics": {
            "validation": eval_wicket(yv, lgbm.predict_proba(valid[WICKET_FEATURES])[:, 1]),
            "test": eval_wicket(yte, lgbm.predict_proba(test[WICKET_FEATURES])[:, 1]),
        },
        "artifact": lgbm,
    }
    return models


def select_best(models: Dict[str, Dict], metric: str = "brier") -> Tuple[str, Dict]:
    """Select on validation Brier (lower is better), report test metrics."""

    def key(item: Tuple[str, Dict]) -> float:
        return item[1]["metrics"]["validation"][metric]

    name, best = min(models.items(), key=key)
    return name, best
