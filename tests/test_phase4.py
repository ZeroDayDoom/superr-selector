"""Phase-4 tests: scenario multi-label logic, registry completeness, LOO matchups."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.features.scenario_features import (  # noqa: E402
    SCENARIO_LABELS,
    add_scenario_features,
)
from superr.features.feature_registry import build_registry  # noqa: E402
from superr.features.matchup_features import (  # noqa: E402
    NEUTRAL_RPB,
    _loo_group_rates,
)


@pytest.fixture
def scen_frame() -> pd.DataFrame:
    """Minimal frame covering scenario-relevant columns."""
    n = 20
    df = pd.DataFrame({
        "match_id": 1, "innings": 2, "overid": [(i // 6) + 1 for i in range(n)],
        "over_ball_num": [(i % 6) + 1 for i in range(n)],
        "innings_ball_num": list(range(1, n + 1)),
        "striker_id": 1, "non_striker_id": 2, "bowler_id": 9,
        "legal_balls_before": list(range(n)),
        "legal_balls_remaining": [120 - i for i in range(n)],
        "scheduled_legal_balls": 120,
        "wickets_before": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        "wickets_in_hand": [10 - w for w in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2],
        "target": 200.0, "runs_required_before": [200 - i for i in range(n)],
        "required_run_rate": [12.0] * n,
        "current_run_rate": [6.0] * n,
        "rrr_crr_gap": [6.0] * n,
        "target_progress": [(200 - i) / 200 for i in range(n)],
        "score_vs_par": [-20.0] * n,
        "expected_final_score": [180.0] * n,
        "venue_season_par": [180.0] * n,
        "par_run_rate": [9.0] * n,
        "last6_wickets": [1] * n, "last12_wickets": [3] * n,
        "balls_since_last_boundary": [15] * n,
        "bowler_boundary_streak_before": [0] * n,
        "batter_is_new": True, "entering_after_wicket": True,
        "partnership_balls_before": [2] * n,
        "batter_runs_before": [30.0] * n, "batter_is_set": True,
        "batter_role": "finisher", "wicket_drought_severity": [0.6] * n,
        "match_balance_proxy": [0.8] * n,
    })
    return df


def test_scenarios_are_multilabel(scen_frame):
    out = add_scenario_features(scen_frame)
    for lbl in SCENARIO_LABELS:
        assert f"scenario_{lbl}" in out.columns
    # several labels must co-occur on the same delivery
    label_cols = [f"scenario_{l}" for l in SCENARIO_LABELS]
    counts = out[label_cols].sum(axis=1)
    assert counts.max() >= 3  # stacking happens
    assert out["scenario_count"].equals(counts)
    # stacked example: high RRR + new batter + drought all true at once
    assert out["scenario_HIGH_RRR_CHASE"].all()
    assert out["scenario_NEW_BATTER_PRESSURE"].all()
    assert out["scenario_BOUNDARY_DROUGHT"].all()
    assert out["scenario_RAPID_COLLAPSE"].any()
    assert not out["scenario_DEATH_OVERS_PRESSURE"].any()  # only 4 overs spanned


def test_registry_documents_all_features():
    reg = build_registry()
    assert len(reg.specs) > 100
    for name, spec in reg.specs.items():
        assert spec.definition and spec.formula and spec.source
        assert spec.leakage_risk and spec.missing_handling and spec.relevance


def test_matchup_loo_excludes_own_season():
    # one season, one cell: LOO must fall back entirely to the EB prior and
    # the prior must exclude the season's own runs.
    n = 100
    df = pd.DataFrame({
        "match_id": range(n), "innings": 1, "overid": 1, "over_ball_num": 1,
        "innings_ball_num": range(n),
        "striker_id": 1, "non_striker_id": 2, "bowler_id": 3,
        "batting_team_id": 1, "bowling_team_id": 2,
        "season": 2024, "is_legal_ball": True, "is_bowler_wicket": False,
        "batter_runs": [2.0] * n,
    })
    df["batting_type"] = "RHB"
    df["pace_or_spin"] = "Pace"
    df["batter_vs_pace_spin"] = "RHB_vs_Pace"
    rates = _loo_group_rates(df, "batter_vs_pace_spin")
    row = rates.iloc[0]
    # only one season: no other season exists, so the baseline must fall back
    # to the neutral data-free prior — NOT the season's own rate
    assert row["loo_runs_per_ball"] == pytest.approx(NEUTRAL_RPB, abs=1e-12)
    # perturbing the season's runs must NOT change the LOO baseline
    df2 = df.copy()
    df2.loc[0, "batter_runs"] = 6.0
    rates2 = _loo_group_rates(df2, "batter_vs_pace_spin")
    assert rates2.iloc[0]["loo_runs_per_ball"] == pytest.approx(
        row["loo_runs_per_ball"], abs=1e-12)
