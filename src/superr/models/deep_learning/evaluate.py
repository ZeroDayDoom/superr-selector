"""Evaluation utilities for the deep models (same metrics as tabular ones)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from superr.models.deep_learning.trainer import predict_heads


def bucket_metrics(y_true: pd.Series, probs: pd.DataFrame,
                   bucket_runs: Dict[str, float]) -> Dict[str, float]:
    """Log loss + MAE of the expected-runs derivation."""
    eps = 1e-9
    ll = float(-np.mean(np.log(np.clip(
        probs.reindex(columns=list(y_true.unique())).to_numpy()[
            np.arange(len(y_true)), [list(y_true.unique()).index(v) for v in y_true]
        ], eps, 1))))
    actual = y_true.map(bucket_runs).fillna(0.0).values
    exp_runs = sum(probs[b] * r for b, r in bucket_runs.items()).values
    return {"log_loss": ll, "mae_expected_runs": float(np.abs(actual - exp_runs).mean())}


def residuals_by_strata(df: pd.DataFrame, actual_col: str, pred_col: str,
                        strata_cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in strata_cols:
        g = df.groupby(col, observed=True).apply(
            lambda x: float((x[actual_col] - x[pred_col]).abs().mean()),
            include_groups=False,
        )
        out[col] = g.round(4).to_dict()
    return out
