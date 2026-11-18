"""Phase-6 tests: bucket consistency, transitions, deep model plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.models.expected_runs import (  # noqa: E402
    BUCKETS,
    BUCKET_RUNS,
    expected_from_buckets,
    season_to_year,
)
from superr.models.bowling_outcome_model import (  # noqa: E402
    bowling_bucket,
    empirical_conceded_map,
    expected_conceded_from_buckets,
)
from superr.models.deep_learning.sequence_model import build_sequences  # noqa: E402


def test_expected_metrics_consistent():
    rng = np.random.default_rng(3)
    probs = pd.DataFrame(
        rng.dirichlet(np.ones(len(BUCKETS)), size=500), columns=BUCKETS)
    exp = expected_from_buckets(probs)
    # bucket probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)
    # derived probabilities are exactly the bucket components
    assert np.allclose(exp["p_boundary"], probs["four"] + probs["six"])
    assert np.allclose(exp["p_wicket"], probs["wicket"])
    # expected runs within the feasible range
    assert exp["expected_runs_off_bat"].between(0, 6).all()
    # wicket bucket contributes zero runs
    row = pd.DataFrame([{b: 0.0 for b in BUCKETS}])
    row["wicket"] = 1.0
    assert expected_from_buckets(row)["expected_runs_off_bat"].iloc[0] == 0.0


def test_bowling_bucket_attribution():
    def row(**kw):
        base = {"is_team_wicket": False, "is_bowler_wicket": False,
                "is_four": False, "is_six": False, "extras_type": None,
                "total_runs": 0, "extras_runs": 0}
        base.update(kw)
        return pd.Series(base)

    assert bowling_bucket(row(is_team_wicket=True, is_bowler_wicket=True)) == "bowler_wicket"
    assert bowling_bucket(row(is_team_wicket=True, is_bowler_wicket=False)) == "other_wicket"
    assert bowling_bucket(row(is_four=True, total_runs=4)) == "four"
    assert bowling_bucket(row(extras_type="wide", total_runs=1, extras_runs=1)) == "other_extras"
    assert bowling_bucket(row(total_runs=2)) == "two_three"
    assert bowling_bucket(row(total_runs=0)) == "dot"


def test_empirical_conceded_map_excludes_byes():
    df = pd.DataFrame({
        "is_team_wicket": [True, False, False, False],
        "is_bowler_wicket": [True, False, False, False],
        "is_four": [False, True, False, False],
        "is_six": [False, False, False, False],
        "extras_type": [None, None, "byes", None],
        "total_runs": [0.0, 4.0, 1.0, 1.0],
        "extras_runs": [0.0, 0.0, 1.0, 0.0],
        "bowler_runs_conceded": [0.0, 4.0, 0.0, 1.0],  # byes NOT charged
    })
    m = empirical_conceded_map(df)
    assert m["four"] == 4.0
    assert m["single"] == 1.0  # byes excluded from the bowler's account
    assert m["bowler_wicket"] == 0.0

    probs = pd.DataFrame([{b: 0.0 for b in
                           ["dot", "single", "two_three", "four", "six",
                            "bowler_wicket", "other_wicket", "other_extras"]}])
    probs["single"] = 1.0
    assert expected_conceded_from_buckets(probs, m).iloc[0] == pytest.approx(1.0)


def test_sequences_exclude_current_ball():
    n = 20
    df = pd.DataFrame({
        "match_id": 1, "innings": 1,
        "total_runs": [float(i % 7) for i in range(n)],
        "is_team_wicket": [i == 10 for i in range(n)],
        "is_legal_ball": True,
        "extras_type": [None] * n,
        "legal_balls_before": list(range(n)),
        "PRESSURE_MODEL_C": [0.5] * n,
    })
    seq = build_sequences(df, seq_length=6)
    # step k of the sequence for delivery i must equal delivery i-1-k's runs
    assert seq[5][0][0] * 6 == df["total_runs"].iloc[4]   # prev-1 of ball 5
    assert seq[5][1][0] * 6 == df["total_runs"].iloc[3]   # prev-2 of ball 5
    # the first deliveries are zero-padded (no prior deliveries)
    assert seq[0].sum() == 0.0
    # delivery 2 has exactly two prior deliveries: step 2 (prev-3) is padded
    assert seq[2][2].sum() == 0.0
    assert seq[2][1].sum() > 0.0
    # current ball never appears as prev-1
    assert seq[10][0][1] == 0.0     # wicket at ball 10 must not be in own seq


def test_sequences_handle_nullable_dtypes():
    """Regression: production data stores is_team_wicket as nullable
    `boolean`; the lag must survive shift + fill without a masked-array
    TypeError (this crashed Phase 6 on the full dataset)."""
    n = 20
    df = pd.DataFrame({
        "match_id": 1, "innings": 1,
        "total_runs": pd.array([i % 7 for i in range(n)], dtype="Int64"),
        "is_team_wicket": pd.array([i == 10 for i in range(n)], dtype="boolean"),
        "is_legal_ball": pd.array([True] * n, dtype="boolean"),
        "extras_type": [None] * n,
        "legal_balls_before": list(range(n)),
        "PRESSURE_MODEL_C": pd.array([0.5] * n, dtype="Float64"),
    })
    seq = build_sequences(df, seq_length=6)
    assert np.isfinite(seq).all()
    assert seq[11][0][1] == 1.0     # wicket at ball 10 IS prev-1 of ball 11
    assert seq[10][0][1] == 0.0     # ...and never ball 10's own sequence
    assert seq[0].sum() == 0.0      # zero-padded at innings start


def test_season_to_year_phase6():
    s = pd.Series(["2022", "2025/26", None])
    out = season_to_year(s)
    assert out.iloc[0] == 2022 and pd.isna(out.iloc[2])
