"""Calibration utilities: reliability tables, ECE, stratified calibration."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def reliability_table(
    y: pd.Series, p: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Quantile-binned reliability table with counts."""
    prob_true, prob_pred = calibration_curve(
        y, p, n_bins=n_bins, strategy="quantile"
    )
    t = pd.DataFrame({"bin_pred": prob_pred, "bin_obs": prob_true})
    t["bin_n"] = np.linspace(len(y) / n_bins, len(y) / n_bins, len(t)).astype(int)
    t["gap"] = t["bin_obs"] - t["bin_pred"]
    return t


def expected_calibration_error(y: pd.Series, p: np.ndarray, n_bins: int = 10) -> float:
    bins = pd.qcut(pd.Series(p), q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": bins})
    g = df.groupby("bin", observed=True).agg(n=("y", "size"),
                                             ybar=("y", "mean"),
                                             pbar=("p", "mean"))
    return float((g["n"] / len(df) * (g["ybar"] - g["pbar"]).abs()).sum())


def calibration_by_strata(
    y: pd.Series, p: np.ndarray, strata: pd.Series, n_bins: int = 5
) -> Dict[str, pd.DataFrame]:
    """Reliability tables computed within each stratum separately."""
    df = pd.DataFrame({"y": np.asarray(y), "p": p, "stratum": np.asarray(strata)})
    out: Dict[str, pd.DataFrame] = {}
    for s, grp in df.groupby("stratum", observed=True):
        if grp["y"].nunique() < 2 or len(grp) < 500:
            continue
        out[str(s)] = reliability_table(grp["y"], grp["p"].values, n_bins=n_bins)
    return out


def brier_decomposition(y: pd.Series, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Murphy decomposition: reliability, resolution, uncertainty."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": p})
    bins = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby(bins, observed=True).agg(n=("y", "size"), ybar=("y", "mean"),
                                            pbar=("p", "mean"))
    ybar = df["y"].mean()
    rel = float((g["n"] / len(df) * (g["ybar"] - g["pbar"]) ** 2).sum())
    res = float((g["n"] / len(df) * (g["ybar"] - ybar) ** 2).sum())
    unc = float(ybar * (1 - ybar))
    return {"reliability": rel, "resolution": res, "uncertainty": unc}
