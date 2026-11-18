"""Batting pressure features (per delivery, striker perspective).

Many base quantities (chase state, par state, momentum windows, striker
state) are built in Phase 3 (match_state / player_state) and are re-exposed
here unchanged. This module adds batting-specific derivations and collapse
metrics. Every feature is documented in FEATURE_DOCS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODULE = "batting_pressure_features"

# passthrough columns from Phase 3 (documented here for the registry)
PASSTHROUGH_BATTING = {
    "runs_required_before": ("Chase: runs still needed", "target - score_before (clipped at 0)",
                             "match_state.add_chase_state", True,
                             "None — target known at innings break", "NaN in innings 1",
                             "Core chase difficulty quantity"),
    "required_run_rate": ("Chase: required run rate", "runs_required / (balls_remaining/6)",
                          "match_state.add_chase_state", True, "None", "NaN in innings 1",
                          "Primary chase steepness measure"),
    "current_run_rate": ("Batting run rate so far", "score_before / (legal_balls_before/6)",
                         "match_state.add_pre_delivery_state", True, "None", "0 when no balls faced",
                         "Context for required-rate gap"),
    "rrr_crr_gap": ("Chase: required minus current run rate", "required_run_rate - current_run_rate",
                    "match_state.add_chase_state", True, "None", "NaN in innings 1",
                    "Positive gap = escalating chase pressure"),
    "required_boundaries_estimate": ("Sixes needed to close the chase", "ceil(runs_required/6)",
                                     "match_state.add_chase_state", True, "None", "NaN in innings 1",
                                     "Upper bound on boundary burden"),
    "wickets_in_hand": ("Wickets remaining", "10 - wickets_before",
                        "match_state.add_pre_delivery_state", True, "None", "n/a",
                        "Resource availability"),
    "legal_balls_remaining": ("Legal balls remaining", "scheduled_legal_balls - legal_balls_before",
                              "match_state.add_pre_delivery_state", True, "None (DLS-aware)",
                              "n/a", "Time pressure"),
    "target_progress": ("Fraction of target scored", "score_before / target",
                        "match_state.add_chase_state", True, "None", "NaN in innings 1",
                        "Chase position on the win-probability curve"),
    "score_vs_par": ("Runs above/below state-expected score", "score_before - expected_score_at_state (LOO curve)",
                     "match_state.add_par_state", True, "LOO curve excludes own innings; venue par leave-one-match-out",
                     "Fallback hierarchy documented in match_state",
                     "Batting-first difficulty vs historical state baselines"),
    "expected_final_score": ("Projected final total from state", "expected_score_at_state + expected_runs_remaining",
                             "match_state.add_par_state", True, "LOO curves; no own-innings information",
                             "Fallback hierarchy", "Batting-first difficulty / target-set strength"),
    "score_deficit_vs_par": ("Alias of score_vs_par", "score_before - expected_score_at_state",
                             "match_state.add_par_state", True, "As score_vs_par", "As score_vs_par",
                             "Batting-first difficulty"),
    "expected_runs_remaining": ("Expected runs still to be scored", "LOO mean of (final - score) at (balls-left, wickets)",
                                "match_state.add_par_state", True, "LOO curves", "Fallback hierarchy",
                                "Bowling-first pressure and chase context"),
    "par_run_rate": ("Venue-season expected run rate", "venue_season_par / scheduled overs",
                     "match_state.add_par_state", True, "Venue par shrinkage k=8, leave-one-match-out",
                     "Filled with global par when venue unseen", "Baseline scoring environment"),
    "last6_wickets": ("Team wickets in previous 6 deliveries", "shifted rolling count",
                      "match_state.add_momentum_features", True, "Shifted by one row; excludes current ball",
                      "NaN in first 6 deliveries of innings", "Collapse detection"),
    "last12_wickets": ("Team wickets in previous 12 deliveries", "shifted rolling count",
                       "match_state.add_momentum_features", True, "Shifted; excludes current ball",
                       "NaN in first 12 deliveries", "Collapse detection"),
    "last18_wickets": ("Team wickets in previous 18 deliveries", "shifted rolling count",
                       "match_state.add_momentum_features", True, "Shifted; excludes current ball",
                       "NaN in first 18 deliveries", "Collapse detection"),
    "last24_wickets": ("Team wickets in previous 24 deliveries", "shifted rolling count",
                       "match_state.add_momentum_features", True, "Shifted; excludes current ball",
                       "NaN in first 24 deliveries", "Collapse detection"),
    "consecutive_dots_before": ("Dot streak immediately before", "streak of consecutive dot deliveries",
                                "match_state.add_momentum_features", True, "Pre-delivery streak only",
                                "0 at innings start", "Scoreboard pressure"),
    "balls_since_last_boundary": ("Deliveries since team boundary", "position - last boundary position",
                                  "match_state.add_momentum_features", True, "Pre-delivery only",
                                  "NaN before first boundary", "Boundary drought pressure"),
    "boundary_streak_before": ("Consecutive boundaries immediately before", "streak length",
                               "match_state.add_momentum_features", True, "Pre-delivery only",
                               "0 when not on a streak", "Momentum in batter's favour"),
    "last6_run_rate": ("Recent scoring rate (6 deliveries)", "last6_runs / (last6_legal/6)",
                       "match_state.add_momentum_features", True, "Shifted; excludes current ball",
                       "NaN in first 6 deliveries", "Recent momentum"),
    "last12_run_rate": ("Recent scoring rate (12 deliveries)", "last12_runs / (last12_legal/6)",
                        "match_state.add_momentum_features", True, "Shifted", "NaN early in innings",
                        "Recent momentum"),
    "last12_dot_balls": ("Dots in previous 12 deliveries", "shifted rolling count",
                         "match_state.add_momentum_features", True, "Shifted", "NaN early",
                         "Recent dot pressure"),
    "last12_boundaries": ("Boundaries in previous 12 deliveries", "shifted rolling count",
                          "match_state.add_momentum_features", True, "Shifted", "NaN early",
                          "Recent boundary flow"),
    "batter_balls_faced_before": ("Striker's legal balls faced this innings", "per-striker exclusive cumsum",
                                  "player_state.add_striker_state", True, "Per-player grouping; exclusive of current ball",
                                  "0 at entry", "Setness / responsibility"),
    "batter_runs_before": ("Striker's runs this innings", "per-striker exclusive cumsum",
                           "player_state.add_striker_state", True, "Per-player grouping", "0 at entry",
                           "Setness / responsibility"),
    "batter_is_new": ("Striker has faced < 6 balls", "balls_faced_before < 6 (config)",
                      "player_state.add_striker_state", True, "None", "n/a",
                      "New-batter vulnerability"),
    "batter_is_set": ("Striker has faced >= 12 balls", "balls_faced_before >= 12 (config)",
                      "player_state.add_striker_state", True, "None", "n/a",
                      "Set batters are prized wickets"),
    "batter_entry_wickets": ("Wickets lost when batter came in", "wickets_before at entry (ffill/bfill)",
                             "player_state.add_striker_state", True, "None", "n/a",
                             "Entry difficulty"),
    "batter_entry_ball": ("Legal ball at which batter entered", "legal_balls_before at entry",
                          "player_state.add_striker_state", True, "None", "n/a",
                          "Entry difficulty"),
    "batter_role": ("Batter role from actual positions + death exposure", "role bands + finisher refinement",
                    "player_state.add_batter_roles", True, "None", "'unknown' when order missing",
                    "Role-conditioned baselines"),
    "striker_dismissal_risk_career": ("Career-to-date dismissal rate", "dismissals/legal balls over PRIOR matches (>=30 balls)",
                                      "player_state.add_career_dismissal_risk", True,
                                      "Shifted by one match; prior seasons only in walk-forward",
                                      "NaN until 30 career legal balls", "Prior vulnerability"),
    "partnership_runs_before": ("Runs in current partnership before this ball", "exclusive cumsum within partnership block",
                                "player_state.add_striker_state", True, "Block = pre-delivery wicket count",
                                "0 after wicket", "Partnership stability"),
    "partnership_balls_before": ("Legal balls in current partnership", "exclusive cumsum within block",
                                 "player_state.add_striker_state", True, "As partnership_runs_before",
                                 "0 after wicket", "Partnership stability / rebuild detection"),
}

NEW_FEATURE_DOCS: Dict[str, dict] = {
    "balls_per_wicket_remaining": (
        "Legal balls available per wicket in hand",
        "legal_balls_remaining / max(1, wickets_in_hand)",
        MODULE, True, "None", "n/a",
        "Low values = double jeopardy (must score fast with few wickets)",
    ),
    "resources_remaining": (
        "Simple resource proxy: balls left x wickets left",
        "(legal_balls_remaining/scheduled) * (wickets_in_hand/10), clipped [0,1]",
        MODULE, True,
        "Proxy — official DLS resource tables are not supplied in the pack",
        "n/a", "Unified scarcity measure across chases and first innings",
    ),
    "wickets_lost_vs_expected": (
        "Wickets lost relative to state-expected wickets",
        "wickets_before - LOO mean wickets at (balls_elapsed_over, innings type)",
        MODULE, True, "LOO curve excludes the current innings", "Fallback to global wicket curve",
        "Positive = batting side behind the typical wicket-loss schedule",
    ),
    "consecutive_wickets": (
        "Wickets in the current collapse cluster",
        "wickets in last 18 legal balls when the most recent wicket is within 12 legal balls, else 1 if wickets_in_last_12>0",
        MODULE, True, "Derived from shifted windows only", "0 when no recent wicket",
        "Clustered wickets define a collapse, not isolated losses",
    ),
    "collapse_severity": (
        "Continuous collapse intensity, 0-1",
        "0.5*min(1, wk12/3) + 0.3*1[wickets_in_hand<=4] + 0.2*1[over>=7]",
        MODULE, True, "Derived from pre-delivery state", "0 when no recent wickets",
        "Single collapse axis for scenario labels and ratings",
    ),
    "entering_after_wicket": (
        "Batter's first delivery comes directly after a wicket",
        "partnership_balls_before == 0",
        MODULE, True, "None", "False otherwise",
        "Incoming-batter vulnerability window"),
    "entry_score": (
        "Team score when batter entered", "score_before at entry (ffill/bfill within batter-innings)",
        MODULE, True, "None", "n/a",
        "Context of the situation inherited by the batter",
    ),
    "entry_required_rate": (
        "Required run rate when batter entered (chases)", "required_run_rate at entry",
        MODULE, True, "None", "NaN in innings 1",
        "Chase difficulty inherited at entry",
    ),
    "entry_balls_remaining": (
        "Balls remaining when batter entered", "legal_balls_remaining at entry",
        MODULE, True, "None", "n/a",
        "Time pressure inherited at entry",
    ),
}

FEATURE_DOCS: Dict[str, dict] = {
    **{k: dict(zip(("definition", "formula", "source", "available_before_delivery",
                    "leakage_risk", "missing_handling", "relevance"), v))
       for k, v in PASSTHROUGH_BATTING.items()},
    **{k: dict(zip(("definition", "formula", "source", "available_before_delivery",
                    "leakage_risk", "missing_handling", "relevance"), v))
       for k, v in NEW_FEATURE_DOCS.items()},
}


def _loo_wicket_curve(df: pd.DataFrame) -> pd.Series:
    """LOO mean wickets_before at (balls_elapsed_over, innings type)."""
    df = df.assign(_elapsed=(df["legal_balls_before"] // 6).astype("int64"))
    out = pd.Series(np.nan, index=df.index, name="wickets_lost_vs_expected_curve")
    for inn in (1, 2):
        sub = df[df["innings"].eq(inn)]
        if not len(sub):
            continue
        base = sub.groupby("_elapsed")["wickets_before"].agg(["sum", "size"])
        own = (
            sub.groupby(["match_id", "innings", "_elapsed"])["wickets_before"]
            .agg(["sum", "size"]).reset_index()
        )
        own = own.merge(base, on="_elapsed", how="left")
        own["loo"] = (own["sum_y"] - own["sum_x"]) / (own["size_y"] - own["size_x"])
        key = sub[["match_id", "innings", "_elapsed"]].merge(
            own[["match_id", "innings", "_elapsed", "loo"]],
            on=["match_id", "innings", "_elapsed"], how="left",
        )
        s = pd.Series(key["loo"].values, index=sub.index)
        # fallback: global wicket curve at elapsed-over level
        glob = (
            sub.groupby(["match_id", "innings", "_elapsed"]).size().rename("n").reset_index()
        )
        out.loc[sub.index] = s
    # fill remaining NaN with overall mean wickets by elapsed bin
    overall = df.groupby("_elapsed")["wickets_before"].mean()
    out = out.fillna(df["_elapsed"].map(overall))
    return out


def add_batting_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sched = df["scheduled_legal_balls"].replace(0, np.nan)
    df["balls_per_wicket_remaining"] = df["legal_balls_remaining"] / (
        df["wickets_in_hand"].clip(lower=1)
    )
    df["resources_remaining"] = (
        (df["legal_balls_remaining"] / sched).clip(0, 1)
        * (df["wickets_in_hand"] / 10.0)
    ).clip(0, 1)

    df["wickets_lost_vs_expected"] = (
        df["wickets_before"] - _loo_wicket_curve(df)
    )

    # collapse cluster / severity from pre-delivery windows
    wk12 = df["last12_wickets"].fillna(0)
    wk6 = df["last6_wickets"].fillna(0)
    recent_wicket = (wk12 > 0) | (wk6 > 0)
    over_idx = (df["legal_balls_before"] // 6).astype("int64") + 1
    cluster = np.where(
        wk12 >= 3, 3.0, np.where(wk12 == 2, 2.0, np.where(wk6 >= 1, 1.0, 0.0))
    )
    df["consecutive_wickets"] = np.where(recent_wicket, cluster, 0.0)
    df["collapse_severity"] = (
        0.5 * np.minimum(1.0, wk12 / 3.0)
        + 0.3 * df["wickets_in_hand"].le(4).astype(float)
        + 0.2 * (over_idx >= 7).astype(float)
    ).clip(0, 1)
    df["entering_after_wicket"] = df["partnership_balls_before"].eq(0)

    # entry context per batter-innings
    sk = ["match_id", "innings", "striker_id"]
    g2 = df.groupby(sk, sort=False)
    first_idx = g2.cumcount().eq(0)
    for src, dst in [("score_before", "entry_score"),
                     ("required_run_rate", "entry_required_rate"),
                     ("legal_balls_remaining", "entry_balls_remaining")]:
        df[dst] = (
            df[src].where(first_idx).groupby(
                [df["match_id"], df["innings"], df["striker_id"]], sort=False
            ).ffill().bfill()
        )
    return df
