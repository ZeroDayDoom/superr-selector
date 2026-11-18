"""Phase-3 tests: synthetic-fixture invariants + smoke tests on outputs.

Run:  pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.data.clean_data import (  # noqa: E402
    add_legal_ball,
    add_run_reconstruction,
    add_wicket_reconstruction,
    clean_extras,
)
from superr.data.match_state import (  # noqa: E402
    add_chase_state,
    add_momentum_features,
    add_pre_delivery_state,
    build_innings_summary,
)
from superr.data.player_state import add_bowler_state, add_striker_state  # noqa: E402
from superr.data.validate_data import run_validation  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixture: one match, extras, wickets, DLS-style revised target
# ---------------------------------------------------------------------------

@pytest.fixture
def mini() -> pd.DataFrame:
    rows = [
        # match 1, innings 1: 6 legal balls + 1 wide, 1 wicket
        dict(match_id=1, innings=1, overid=1, over_ball_num=1, innings_ball_num=1,
             striker_id=101, non_striker_id=102, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=1, extras_type=None, extras1_runs=0,
             extras_2_runs=0, is_four="false", is_six="false", is_wicket="false",
             dismissal_type=None, dismissed_player_id=None),
        dict(match_id=1, innings=1, overid=1, over_ball_num=2, innings_ball_num=2,
             striker_id=101, non_striker_id=102, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=4, extras_type=None, extras1_runs=0,
             extras_2_runs=0, is_four="true", is_six="false", is_wicket="false",
             dismissal_type=None, dismissed_player_id=None),
        dict(match_id=1, innings=1, overid=1, over_ball_num=3, innings_ball_num=3,
             striker_id=101, non_striker_id=102, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=0, extras_type="wide", extras1_runs=1,
             extras_2_runs=0, is_four="false", is_six="false", is_wicket="false",
             dismissal_type=None, dismissed_player_id=None),
        dict(match_id=1, innings=1, overid=1, over_ball_num=4, innings_ball_num=4,
             striker_id=101, non_striker_id=102, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=6, extras_type=None, extras1_runs=0,
             extras_2_runs=0, is_four="false", is_six="true", is_wicket="false",
             dismissal_type=None, dismissed_player_id=None),
        # run-out: team wicket, NOT bowler wicket
        dict(match_id=1, innings=1, overid=1, over_ball_num=5, innings_ball_num=5,
             striker_id=101, non_striker_id=102, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=1, extras_type=None, extras1_runs=0,
             extras_2_runs=0, is_four="false", is_six="false", is_wicket="true",
             dismissal_type="RunOut", dismissed_player_id=102),
        dict(match_id=1, innings=1, overid=1, over_ball_num=6, innings_ball_num=6,
             striker_id=101, non_striker_id=103, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=0, extras_type=None, extras1_runs=0,
             extras_2_runs=0, is_four="false", is_six="false", is_wicket="false",
             dismissal_type=None, dismissed_player_id=None),
        # bowled: team + bowler wicket
        dict(match_id=1, innings=1, overid=2, over_ball_num=1, innings_ball_num=7,
             striker_id=103, non_striker_id=101, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=0, extras_type=None, extras1_runs=0,
             extras_2_runs=0, is_four="false", is_six="false", is_wicket="true",
             dismissal_type="Bowled", dismissed_player_id=103),
        dict(match_id=1, innings=1, overid=2, over_ball_num=2, innings_ball_num=8,
             striker_id=101, non_striker_id=104, bowler_id=201, batting_team_id=11,
             bowling_team_id=12, runs=2, extras_type="legbyes", extras1_runs=2,
             extras_2_runs=0, is_four="false", is_six="false", is_wicket="false",
             dismissal_type=None, dismissed_player_id=None),
    ]
    df = pd.DataFrame(rows)
    df["delivery_ist_time"] = "2024-01-01 00:00:00 UTC"
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # mimic load_tables' boolean normalisation
    for c in ["is_four", "is_six", "is_wicket"]:
        df[c] = df[c].astype("string").str.lower().eq("true").astype("boolean")
    df = clean_extras(df)
    df = add_run_reconstruction(df)
    df = add_legal_ball(df)
    df = add_wicket_reconstruction(df)
    # special-handling columns (normally added by add_special_handling)
    for c in ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]:
        df[c] = pd.NA
    df["is_super_over_innings"] = False
    df["is_dls_revised"] = False
    df["has_revised_target"] = False
    df["match_result_type"] = "decisive"
    df["match_in_scope"] = True
    if "innings_ball_num" in df.columns:
        df = df.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    return df


def test_run_reconstruction(mini):
    df = _prep(mini)
    # total = off-bat + extras
    assert (df["total_runs"] == df["batter_runs"] + df["extras_runs"]).all()
    # wide row: total 1, batter 0, bowler charged 1
    wide = df[df["extras_type"] == "wide"].iloc[0]
    assert wide["total_runs"] == 1 and wide["batter_runs"] == 0
    assert wide["bowler_runs_conceded"] == 1
    # legbye row: total 4 (2 off-bat + 2 legbyes), bowler charged 2 (off-bat)
    lb = df[df["extras_type"] == "legbyes"].iloc[0]
    assert lb["total_runs"] == 4 and lb["bowler_runs_conceded"] == 2
    # legal balls: wide is illegal, legbye is legal
    assert int(df["is_legal_ball"].sum()) == 7


def test_wicket_reconstruction(mini):
    df = _prep(mini)
    team_wk = df[df["is_team_wicket"]]
    assert len(team_wk) == 2
    bowler_wk = df[df["is_bowler_wicket"]]
    assert len(bowler_wk) == 1
    assert bowler_wk.iloc[0]["wicket_type"] == "Bowled"
    runout = df[df["is_run_out"]].iloc[0]
    assert runout["dismissed_player_id"] == 102  # non-striker run out
    assert not runout["is_bowler_wicket"]


def test_innings_summary(mini):
    df = _prep(mini)
    innings = build_innings_summary(df)
    assert len(innings) == 1
    row = innings.iloc[0]
    assert row["innings_total"] == 1 + 4 + 1 + 6 + 1 + 0 + 0 + 4  # 17
    assert row["total_wickets"] == 2
    assert row["legal_balls"] == 7
    assert not row["is_all_out"]


def test_pre_delivery_state_exclusive(mini):
    df = _prep(mini)
    df = add_pre_delivery_state(df)
    # first delivery: zeroed state
    assert df.iloc[0]["score_before"] == 0
    assert df.iloc[0]["wickets_before"] == 0
    # after the wide (row 2), score includes the wide's run but no legal ball
    assert df.iloc[2]["score_before"] == 5  # 1 + 4
    assert df.iloc[2]["legal_balls_before"] == 2
    # run-out row: wickets_before counts PRIOR wickets only
    assert df.iloc[4]["wickets_before"] == 0
    assert df.iloc[5]["wickets_before"] == 1
    # last row: exclusive cumsum excludes its own 2 legbyes
    assert df.iloc[-1]["score_before"] == 13


def test_chase_state_and_validation(mini):
    # Build BOTH innings as raw rows, then run the full pipeline once.
    rows2 = []
    for k, r in enumerate([4, 0, 6, 1]):
        rows2.append(dict(
            match_id=1, innings=2, overid=1, over_ball_num=k + 1,
            innings_ball_num=k + 1,
            striker_id=105, non_striker_id=106, bowler_id=201,
            batting_team_id=12, bowling_team_id=11, runs=r, extras_type=None,
            extras1_runs=0, extras_2_runs=0, is_four="true" if r == 4 else "false",
            is_six="true" if r == 6 else "false", is_wicket="false",
            dismissal_type=None, dismissed_player_id=None,
        ))
    raw = pd.concat([mini, pd.DataFrame(rows2)], ignore_index=True)
    raw["delivery_ist_time"] = "2024-01-01 00:00:00 UTC"
    df_all = _prep(raw)

    innings = build_innings_summary(df_all)
    df_all = add_pre_delivery_state(df_all)
    df_all = add_chase_state(df_all, innings)
    chase = df_all[df_all["innings"] == 2]
    assert (chase["target"] == innings.iloc[0]["innings_total"] + 1).all()
    # momentum: shifted windows never include the current ball
    df_all = add_momentum_features(df_all, [6])
    first_b = df_all.iloc[0]
    assert pd.isna(first_b["last6_runs"])  # no previous deliveries
    # player state (validation battery inspects these columns)
    feats = {"new_batter_balls": 6, "set_batter_balls": 12,
             "role_bands": {"opener": [1, 2], "top": [3, 4], "middle": [5, 6, 7],
                            "lower": [8, 9, 10, 11]},
             "finisher_death_share": 0.25}
    df_all = add_striker_state(df_all, feats)
    df_all = add_bowler_state(df_all, [6])
    # validation battery on the synthetic data must not FAIL
    scope = pd.DataFrame({
        "match_id": [1], "revised_target": [pd.NA],
        "result_type": ["decisive"], "scope_include": [True],
    })
    results = run_validation(df_all, innings, scope)
    fails = [r for r in results if r["status"] == "fail"]
    assert not fails, fails


# ---------------------------------------------------------------------------
# Smoke tests on the real processed outputs (skipped if not yet built)
# ---------------------------------------------------------------------------

PROCESSED = PROJECT_ROOT / "data" / "processed"


@pytest.mark.skipif(
    not (PROCESSED / "balls_clean.parquet").exists(), reason="outputs not built"
)
class TestProcessedOutputs:
    def test_loads_and_keys(self):
        b = pd.read_parquet(PROCESSED / "balls_clean.parquet")
        assert not b.duplicated(subset=["match_id", "innings", "overid",
                                        "over_ball_num"]).any()
        assert b["score_before"].ge(0).all()

    def test_state_invariants(self):
        b = pd.read_parquet(PROCESSED / "balls_clean.parquet")
        g = b.groupby(["match_id", "innings"], sort=False)
        cum = g["total_runs"].cumsum() - b["total_runs"]
        assert (b["score_before"] - cum).abs().max() < 1e-9
        assert b["legal_balls_remaining"].ge(0).all()
        assert b.loc[b["innings"] == 2, "target"].notna().all()

    def test_no_super_overs(self):
        b = pd.read_parquet(PROCESSED / "balls_clean.parquet")
        assert b["innings"].isin([1, 2]).all()

    def test_bowler_exclusions(self):
        b = pd.read_parquet(PROCESSED / "balls_clean.parquet")
        m = b["extras_type"].isin(["byes", "legbyes", "penalty", "other", "unknown"])
        assert (
            b.loc[m, "bowler_runs_conceded"] == b.loc[m, "batter_runs"].fillna(0)
        ).all()
