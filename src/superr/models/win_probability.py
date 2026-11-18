"""Win probability engine.

Two calibrated models:
  - WP2 (second innings): probability the chasing team wins, from
    pre-delivery state.
  - WP1 (first innings): probability the batting-first team wins the match,
    from pre-delivery state (total-projection framing).

Training protocol (methodology §5): match-level chronological split —
train seasons <= 2024, validation 2025, holdout 2026. Model families are
compared on the holdout; the winner is calibrated on the validation season
and finally refit on all seasons for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

WP2_FEATURES = [
    "runs_required_before", "legal_balls_remaining", "wickets_before",
    "wickets_in_hand", "required_run_rate", "current_run_rate", "rrr_crr_gap",
    "target", "score_before", "over_number_before", "ball_in_over_before",
    "venue_season_par", "is_knockout", "is_dls_revised", "is_ipl",
    "resources_remaining",
]

WP1_FEATURES = [
    "score_before", "expected_final_score", "score_vs_par", "wickets_before",
    "legal_balls_remaining", "resources_remaining", "venue_season_par",
    "is_knockout", "is_ipl", "over_number_before", "ball_in_over_before",
    "expected_runs_remaining", "current_run_rate",
]

SEASON_TRAIN_MAX = 2024   # train seasons <= this
SEASON_VALID = 2025       # calibration / model selection
SEASON_HOLDOUT = 2026     # final holdout


def season_to_year(season: pd.Series) -> pd.Series:
    s = season.astype("string").str.extract(r"(\d{4})")[0].fillna("")
    return pd.to_numeric(s, errors="coerce").astype("Float64")


def prepare_wp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach model-ready columns to the delivery frame (pre-delivery only)."""
    out = df.copy()
    if "is_ipl" not in out.columns:
        out["is_ipl"] = out["competition_class"].eq("IPL").astype(int)
    out["season_year"] = season_to_year(out["season"])
    out["is_knockout"] = out["knock_out"].astype("boolean").fillna(False).astype(int)
    out["is_dls_revised"] = out["is_dls_revised"].astype("boolean").fillna(False).astype(int)
    # match outcome targets (post-delivery LABELS, never features)
    out["_batting_team_won"] = (
        out["win_team_id"].notna() & out["win_team_id"].eq(out["batting_team_id"])
    ).astype(int)
    return out


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season_year"] <= SEASON_TRAIN_MAX]
    valid = df[df["season_year"] == SEASON_VALID]
    holdout = df[df["season_year"] == SEASON_HOLDOUT]
    if len(train) == 0 or len(valid) == 0 or len(holdout) == 0:
        raise ValueError(
            f"chronological split empty: train={len(train)}, valid={len(valid)}, "
            f"holdout={len(holdout)} — check season coverage"
        )
    return train, valid, holdout


def _candidates():
    return {
        "logistic": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=20260131)),
        ]),
        "hist_gbm": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0, random_state=20260131,
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=31,
            min_child_samples=200, reg_lambda=1.0, random_state=20260131,
            verbose=-1,
        ),
    }


def _eval(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    p = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "calib_slope": float(calibration_slope(y, p)),
    }


def average_precision(y: pd.Series, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, p)


def calibration_slope(y: pd.Series, p: np.ndarray) -> float:
    """Slope of outcome on logit-probability (1 = perfectly calibrated)."""
    eps = 1e-6
    logit = np.log((p + eps) / (1 - p + eps))
    X = logit.reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    return float(lr.coef_[0][0])


def reliability_data(y: pd.Series, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"bin_pred": prob_pred, "bin_obs": prob_true})


@dataclass
class WPModel:
    name: str                 # "wp2" or "wp1"
    model: object
    features: List[str]
    calibrator: Optional[IsotonicRegression] = None
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    selected: Optional[str] = None
    calibration_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # NaN occurs only at terminal states (0 balls left / target reached);
        # those rows are overridden by terminal rules by the caller where it
        # matters; filling 0 keeps the scaler happy for stray rows.
        Xf = X[self.features].fillna(0.0)
        p = self.model.predict_proba(Xf)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-6, 1 - 1e-6)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]


def train_wp(
    df: pd.DataFrame,
    name: str,
    features: List[str],
    compare: bool = True,
) -> WPModel:
    """Train one WP model under the chronological protocol."""
    d = prepare_wp_frame(df)
    train, valid, holdout = chronological_split(d)
    ycol = "_batting_team_won"

    results: Dict[str, Dict[str, float]] = {}
    best_name, best_model, best_key = None, None, -1.0
    for cname, ctor in _candidates().items():
        m = ctor()
        m.fit(train[features], train[ycol])
        if compare:
            results[cname] = {
                "train": _eval(m, train[features], train[ycol]),
                "validation": _eval(m, valid[features], valid[ycol]),
                "holdout": _eval(m, holdout[features], holdout[ycol]),
            }
            key = results[cname]["holdout"]["roc_auc"]
        else:
            results[cname] = {"train": _eval(m, train[features], train[ycol])}
            key = results[cname]["train"]["roc_auc"]
        if key > best_key:
            best_key, best_name, best_model = key, cname, m

    # calibrate the winner on the validation season (isotonic)
    calib = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    vp = best_model.predict_proba(valid[features])[:, 1]
    calib.fit(vp, valid[ycol])

    wp = WPModel(name=name, model=best_model, features=features, calibrator=calib,
                 metrics=results, selected=best_name)
    wp.calibration_data = {
        "validation": reliability_data(valid[ycol], calib.predict(vp)),
    }
    hp = best_model.predict_proba(holdout[features])[:, 1]
    wp.calibration_data["holdout"] = reliability_data(
        holdout[ycol], calib.predict(hp)
    )
    return wp


def train_all_wp(df: pd.DataFrame) -> Dict[str, WPModel]:
    d = prepare_wp_frame(df)
    # chase model: second innings of matches with a resolved winner
    chase = d[(d["innings"] == 2) & d["target"].notna()
              & d["win_team_id"].notna()]
    wp2 = train_wp(chase, "wp2", WP2_FEATURES)
    # first-innings model
    inn1 = d[(d["innings"] == 1) & d["win_team_id"].notna()]
    wp1 = train_wp(inn1, "wp1", WP1_FEATURES)
    return {"wp2": wp2, "wp1": wp1}


def refit_full_wp(df: pd.DataFrame, wp: WPModel) -> WPModel:
    """Refit the selected configuration on all seasons; keep calibration."""
    d = prepare_wp_frame(df)
    if wp.name == "wp2":
        fit = d[(d["innings"] == 2) & d["target"].notna() & d["win_team_id"].notna()]
    else:
        fit = d[(d["innings"] == 1) & d["win_team_id"].notna()]
    ctor = _candidates()[wp.selected]
    m = ctor()
    m.fit(fit[wp.features], fit["_batting_team_won"])
    return WPModel(name=wp.name, model=m, features=wp.features,
                   calibrator=wp.calibrator, metrics=wp.metrics,
                   selected=wp.selected)
