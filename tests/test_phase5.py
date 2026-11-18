"""Phase-5 tests: WP transitions, leverage invariance, pressure assembly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.models.leverage_model import (  # noqa: E402
    BUCKET_EFFECT,
    OUTCOME_BUCKETS,
    _successor_features,
    _outcome_label,
)
from superr.models.pressure_model import (  # noqa: E402
    add_categories,
    assemble_components,
    build_scores,
)
from superr.models.pressure_calibration import (  # noqa: E402
    expected_calibration_error,
    reliability_table,
)
from superr.models.win_probability import season_to_year  # noqa: E402


@pytest.fixture
def state_frame() -> pd.DataFrame:
    n = 10
    return pd.DataFrame({
        "match_id": 1, "innings": 2, "overid": [20] * n,
        "over_ball_num": list(range(1, n + 1)),
        "innings_ball_num": list(range(111, 111 + n)),
        "striker_id": 1, "non_striker_id": 2, "bowler_id": 3,
        "score_before": [150.0] * n, "legal_balls_before": list(range(110, 120)),
        "legal_balls_remaining": [10 - i for i in range(n)],
        "wickets_before": [4] * n, "wickets_in_hand": [6] * n,
        "scheduled_legal_balls": 120,
        "target": 165.0, "runs_required_before": [15.0] * n,
        "required_run_rate": [9.0] * n, "current_run_rate": [8.0] * n,
        "rrr_crr_gap": [1.0] * n, "target_progress": [150.0 / 165.0] * n,
        "expected_final_score": [158.0] * n, "score_vs_par": [-8.0] * n,
        "venue_season_par": [170.0] * n, "expected_runs_remaining": [8.0] * n,
        "is_knockout": False, "is_dls_revised": False,
    })


def test_season_to_year():
    s = pd.Series(["2022", "2022/23", "2024", None])
    out = season_to_year(s).tolist()
    assert out[0] == 2022 and out[1] == 2022 and out[2] == 2024
    assert pd.isna(out[3])


def test_outcome_label_buckets():
    rows = [
        {"is_team_wicket": False, "is_four": False, "is_six": False,
         "extras_type": None, "total_runs": 0, "extras_runs": 0},
        {"is_team_wicket": False, "is_four": True, "is_six": False,
         "extras_type": None, "total_runs": 4, "extras_runs": 0},
        {"is_team_wicket": True, "is_four": False, "is_six": False,
         "extras_type": None, "total_runs": 1, "extras_runs": 0},
        {"is_team_wicket": False, "is_four": False, "is_six": False,
         "extras_type": "wide", "total_runs": 1, "extras_runs": 1},
        {"is_team_wicket": False, "is_four": False, "is_six": False,
         "extras_type": None, "total_runs": 3, "extras_runs": 0},
    ]
    labels = [_outcome_label(pd.Series(r)) for r in rows]
    assert labels == ["dot", "four", "wicket", "other", "two_three"]


def test_successor_transitions(state_frame):
    for bucket, eff in BUCKET_EFFECT.items():
        s = _successor_features(state_frame, bucket)
        assert s["score_before"].iloc[0] == 150 + eff["runs"]
        assert s["legal_balls_remaining"].iloc[0] == max(0, 10 - eff["legal"])
        assert s["wickets_before"].iloc[0] == 4 + eff["wicket"]
    # wicket transition reduces wickets in hand by exactly one
    s = _successor_features(state_frame, "wicket")
    assert s["wickets_in_hand"].iloc[0] == 5
    # dot on the last ball leaves zero balls remaining
    s = _successor_features(state_frame, "dot")
    assert s["legal_balls_remaining"].iloc[-1] == 0


def test_pressure_scores_in_unit_interval():
    n = 200
    rng = np.random.default_rng(0)
    comp_names = ["difficulty", "resource_stress", "collapse_pressure",
                  "momentum_stress", "phase_pressure", "match_importance"]
    df = pd.DataFrame({
        "match_id": 1, "innings": 2,
        "legal_balls_before": np.arange(n),
        "legal_balls_remaining": np.linspace(120, 1, n),
        "scheduled_legal_balls": 120.0,
        "wickets_in_hand": rng.integers(1, 11, n).astype(float),
        "resources_remaining": rng.random(n),
        "wp_before": rng.random(n),
        "leverage_expected_abs_swing": rng.random(n) * 0.08,
        "total_runs": rng.integers(0, 7, n).astype(float),
        "striker_id": 1, "innings_phase": "death",
    })
    for comp in comp_names:
        df[f"ch_{comp}"] = rng.random(n)
        df[f"fi_{comp}"] = rng.random(n)
    df = assemble_components(df)
    df = build_scores(df)
    for col in ("PRESSURE_MODEL_A", "PRESSURE_MODEL_B", "PRESSURE_MODEL_C"):
        assert df[col].between(0, 1).all()
    assert df["PRESSURE_DISPLAY_100"].between(0, 100).all()
    df = add_categories(df)
    cats = set(df["pressure_category"].unique())
    assert cats <= {"LOW", "MODERATE", "HIGH", "EXTREME"}
    assert "EXTREME" in cats and "LOW" in cats
    # stratified labels respect quantiles roughly
    assert df["hp_top20"].mean() == pytest.approx(0.20, abs=0.15)
    assert df["hp_top10"].mean() <= df["hp_top30"].mean()


def test_calibration_utilities():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 5000))
    p = np.clip(y * 0.8 + rng.random(5000) * 0.3, 0.01, 0.99)
    t = reliability_table(y, p)
    assert len(t) > 0 and set(t.columns) >= {"bin_pred", "bin_obs", "gap"}
    ece = expected_calibration_error(y, p)
    assert 0.0 <= ece < 0.5
    # perfectly calibrated predictions -> near-zero ECE
    p_perfect = pd.Series(y).rolling(200, min_periods=1).mean().values
    assert expected_calibration_error(y, p_perfect) < expected_calibration_error(
        y, pd.Series(p)
    )
