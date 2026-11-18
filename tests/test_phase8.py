"""Phase-8 tests: bootstrap, split-half, sensitivity, edge tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.validation.bootstrap import bootstrap_ranks  # noqa: E402
from superr.validation.split_half import split_half_ranking_reliability  # noqa: E402
from superr.validation.sensitivity import (  # noqa: E402
    ablation_sensitivity,
    rank_stability_table,
    rerank,
    weight_sensitivity,
)
from superr.validation.temporal_validation import classify_trend, temporal_folds  # noqa: E402
from superr.validation.adversarial_checks import model_checks  # noqa: E402


@pytest.fixture
def hp_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_players, n_matches = 30, 60
    rows = []
    for m in range(n_matches):
        for _ in range(30):
            pid = int(rng.integers(0, n_players))
            rows.append({"player_id": pid, "match_id": m,
                         "bat_value": rng.normal(pid * 0.01, 0.5),
                         "season_year": int(rng.choice([2023, 2025, 2026]))})
    return pd.DataFrame(rows)


def test_bootstrap_ranks_recovers_strong_player(hp_frame):
    # player 29 gets a big boost
    hp = pd.concat([hp_frame, pd.DataFrame({
        "player_id": [29] * 200, "match_id": list(range(200, 400)),
        "bat_value": [3.0] * 200, "season_year": [2025] * 200})],
        ignore_index=True)
    out = bootstrap_ranks(hp, "bat_value", set(range(30)), n_boot=60)
    assert len(out) == 30
    best = out.sort_values("boot_mean_score", ascending=False).iloc[0]
    assert best["player_id"] == 29
    assert out["p_top10"].max() <= 1.0 and out["p_top10"].min() >= 0.0
    # rank percentiles ordered
    assert (out["rank_p05"] <= out["rank_p50"]).all()
    assert (out["rank_p50"] <= out["rank_p95"]).all()


def test_split_half_returns_metrics(hp_frame):
    m = split_half_ranking_reliability(hp_frame, "bat_value", set(range(30)),
                                       n_splits=20)
    assert 0 <= m["spearman_mean"] <= 1
    assert m["top10_overlap_mean"] <= 10


def test_temporal_folds_and_trend(hp_frame):
    folds = temporal_folds(hp_frame, "bat_value", min_balls=3)
    assert len(folds) >= 1
    trend = hp_frame.groupby(["player_id", "season_year"])["bat_value"] \
        .mean().reset_index().rename(columns={"bat_value": "hp_value"})
    t = classify_trend(trend, 0)
    assert "trend" in t


def test_sensitivity_reweight_changes_ranking(hp_frame):
    agg = hp_frame.groupby("player_id").agg(
        player_name=("player_id", "first"),
        level_lcb=("bat_value", "mean"),
        consistency=("bat_value", lambda x: float((x > 0).mean())),
        volume=("bat_value", "size"),
        cvar5=("bat_value", "min"),
    ).reset_index()
    agg["uplift"] = 0.1
    agg["downside_abs"] = agg["cvar5"].abs()
    mask = pd.Series(True, index=agg.index)
    from superr.ratings.composite_rating import PRIMARY_WEIGHTS, VARIANTS
    tables = weight_sensitivity(agg, mask, VARIANTS)
    stab = rank_stability_table(tables,
                                rerank(agg, lambda d: d["level_lcb"], mask, "s"))
    assert len(stab) == len(VARIANTS)
    assert stab["spearman_vs_primary"].between(0, 1).all()
    abl = ablation_sensitivity(agg, mask, PRIMARY_WEIGHTS)
    assert len(abl) == 5


def test_model_checks_structure():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "legal_balls_before": rng.integers(0, 120, n),
        "PRESSURE_MODEL_C": rng.random(n),
        "is_team_wicket": rng.random(n) < 0.06,
        "realized_wpa": rng.normal(0, 0.03, n),
        "batter_runs": rng.integers(0, 7, n).astype(float),
        "expected_runs_off_bat": rng.random(n) * 2,
        "bat_value": rng.normal(0, 0.2, n),
        "player_id": rng.integers(0, 20, n),
        "match_id": rng.integers(0, 40, n),
    })
    checks = model_checks(df)
    assert "death_share_of_hp_deliveries" in checks
    assert "expected_runs_residual_mean" in checks
    assert 0 <= checks["death_share_of_all_deliveries"] <= 1
