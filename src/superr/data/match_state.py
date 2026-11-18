"""Match-state reconstruction.

Everything here is strictly pre-delivery: cumulative quantities are computed
exclusively (cumsum minus the current row), so no delivery outcome can leak
into its own state. Innings boundaries: innings 1 length = 120 legal balls
(or DLS-revised); innings 2 length = 120 (or DLS-revised); the chase target
is the revised target when supplied, otherwise innings-1 total + 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FULL_INNINGS_LEGAL_BALLS = 120


def _exclusive_cumsum(
    g: pd.core.groupby.GroupBy, col: str, parent: pd.DataFrame
) -> pd.Series:
    """Cumulative sum of col within groups, excluding the current row."""
    return g[col].cumsum() - parent[col]


def _scheduled_legal_balls(innings: pd.Series, revised: pd.Series) -> pd.Series:
    """Scheduled innings length in legal balls, honouring DLS revisions.

    The audit found revised values are recorded in BALLS (e.g. 102); values
    <= 20 are treated as OVERS defensively and converted.
    """
    rev = revised.astype("Float64")
    as_balls = rev.where(rev > 20, rev * 6)
    base = pd.Series(FULL_INNINGS_LEGAL_BALLS, index=innings.index, dtype="Float64")
    return base.fillna(as_balls).where(as_balls.isna(), as_balls).astype("int64")


def build_innings_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, innings): totals, wickets, lengths, status."""
    g = df.groupby(["match_id", "innings"], sort=False)
    agg = g.agg(
        batting_team_id=("batting_team_id", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        innings_total=("total_runs", "sum"),
        total_wickets=("is_team_wicket", "sum"),
        legal_balls=("is_legal_ball", "sum"),
        deliveries=("total_runs", "size"),
        extras_total=("extras_runs", "sum"),
        byes=("extras_type", lambda s: float((s == "byes").sum())),
        legbyes=("extras_type", lambda s: float((s == "legbyes").sum())),
        wides=("extras_type", lambda s: float((s == "wide").sum())),
        noballs=("extras_type", lambda s: float((s == "no_ball").sum())),
        first_ts=("delivery_ist_time", "first"),
        last_ts=("delivery_ist_time", "last"),
        is_super_over=("is_super_over_innings", "first"),
        is_dls_revised=("is_dls_revised", "first"),
        revised_target=("revised_target", "first"),
    ).reset_index()
    agg["overs_played"] = agg["legal_balls"] / 6.0
    lb = pd.to_numeric(agg["legal_balls"], errors="coerce").fillna(0)
    tot = pd.to_numeric(agg["innings_total"], errors="coerce").fillna(0)
    denom = (lb / 6.0).replace(0, np.nan)
    agg["innings_run_rate"] = np.where(lb > 0, tot / denom, 0.0)
    agg["is_all_out"] = agg["total_wickets"].ge(10)
    # scheduled length: DLS revisions recorded in balls (values <= 20 = overs)
    m_cols = ["match_id", "first_inn_revised_overs", "second_inn_revised_overs"]
    mm = df[m_cols].drop_duplicates("match_id")
    agg = agg.merge(mm, on="match_id", how="left", validate="many_to_one")
    rev_col = np.where(
        agg["innings"].eq(2),
        agg["second_inn_revised_overs"],
        agg["first_inn_revised_overs"],
    )
    agg["scheduled_legal_balls"] = _scheduled_legal_balls(
        agg["innings"], pd.Series(rev_col, index=agg.index)
    )
    agg["innings_length_balls"] = agg["legal_balls"]
    agg["innings_end_ts"] = agg["last_ts"]
    agg["innings_start_ts"] = agg["first_ts"]
    agg["bowler_runs_conceded_total"] = (
        df.groupby(["match_id", "innings"], sort=False)["bowler_runs_conceded"]
        .sum()
        .values
    )
    drop = [c for c in ["first_inn_revised_overs", "second_inn_revised_overs",
                        "first_ts", "last_ts"] if c in agg.columns]
    agg = agg.drop(columns=drop)
    return agg


def add_pre_delivery_state(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly pre-delivery innings state for every delivery."""
    df = df.copy()
    g = df.groupby(["match_id", "innings"], sort=False)

    df["score_before"] = _exclusive_cumsum(g, "total_runs", df)
    df["wickets_before"] = _exclusive_cumsum(g, "is_team_wicket", df).astype("int64")
    df["legal_balls_before"] = _exclusive_cumsum(g, "is_legal_ball", df).astype("int64")

    # scheduled length in legal balls for this delivery's innings
    rev1 = df.groupby("match_id", sort=False)["first_inn_revised_overs"].first()
    rev2 = df.groupby("match_id", sort=False)["second_inn_revised_overs"].first()
    per_inn = (
        df[["match_id", "innings"]]
        .drop_duplicates()
        .merge(rev1.rename("rev1"), on="match_id", validate="many_to_one")
        .merge(rev2.rename("rev2"), on="match_id", validate="many_to_one")
    )
    per_inn["rev"] = np.where(
        per_inn["innings"].eq(2), per_inn["rev2"], per_inn["rev1"]
    )
    per_inn["sched_legal"] = _scheduled_legal_balls(
        per_inn["innings"], per_inn["rev"]
    )
    df = df.merge(
        per_inn[["match_id", "innings", "sched_legal"]],
        on=["match_id", "innings"],
        how="left",
        validate="many_to_one",
    )
    sched = df["sched_legal"].astype("int64")
    df["scheduled_legal_balls"] = sched

    df["legal_balls_remaining"] = (sched - df["legal_balls_before"]).clip(lower=0)
    df["overs_remaining"] = df["legal_balls_remaining"] / 6.0
    df["balls_elapsed"] = df["legal_balls_before"]

    over_index = (df["legal_balls_before"] // 6).astype("int64")
    ball_in_over = (df["legal_balls_before"] % 6).astype("int64")
    df["over_number_before"] = over_index + 1  # 1-based "current over"
    df["ball_in_over_before"] = ball_in_over

    df["current_run_rate"] = np.where(
        df["legal_balls_before"] > 0,
        df["score_before"] / (df["legal_balls_before"] / 6.0),
        0.0,
    )
    df["innings_run_rate"] = df["current_run_rate"]
    df["wickets_in_hand"] = (10 - df["wickets_before"]).clip(lower=0)
    return df


def add_chase_state(df: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """Second-innings chase state (target-aware, DLS-correct)."""
    df = df.copy()

    inn1 = innings[innings["innings"].eq(1)][["match_id", "innings_total"]]
    inn2_rev = innings[innings["innings"].eq(2)][["match_id", "revised_target"]]
    tgt = inn1.merge(inn2_rev, on="match_id", how="inner", validate="one_to_one")
    tgt["target"] = np.where(
        tgt["revised_target"].notna(), tgt["revised_target"],
        tgt["innings_total"] + 1,
    )
    tgt["target_is_revised"] = tgt["revised_target"].notna()
    tgt = tgt[["match_id", "target", "target_is_revised"]]

    df = df.merge(tgt, on="match_id", how="left", validate="many_to_one")
    df["is_chase"] = df["innings"].eq(2)
    df["target"] = df["target"].astype("Float64")
    df["runs_required_before"] = np.where(
        df["is_chase"], (df["target"].astype("float64") - df["score_before"]).clip(lower=0),
        np.nan,
    )
    denom = df["overs_remaining"].replace(0, np.nan)
    df["required_run_rate"] = np.where(
        df["is_chase"], df["runs_required_before"] / denom, np.nan
    )
    df["rrr_crr_gap"] = df["required_run_rate"] - df["current_run_rate"]
    df["target_progress"] = np.where(
        df["is_chase"], df["score_before"] / df["target"].astype("float64"), np.nan
    )
    df["runs_required_per_remaining_over"] = df["required_run_rate"]
    # maximum-boundary estimate: sixes needed to close the target
    df["required_boundaries_estimate"] = np.where(
        df["is_chase"], np.ceil(df["runs_required_before"] / 6.0), np.nan
    )
    df["balls_remaining"] = df["legal_balls_remaining"]
    df["wickets_remaining"] = df["wickets_in_hand"]
    return df


# ---------------------------------------------------------------------------
# First-innings par / expected-scoring state (Step 9)
# ---------------------------------------------------------------------------

def _loo_curve(
    df: pd.DataFrame,
    value_col: str,
    bin_cols: list[str],
    state_col: str,
) -> pd.Series:
    """Leave-one-innings-out historical mean of `value_col` per state bin.

    For delivery i in innings h at bin b, the curve value is the mean of
    `value_col` over all deliveries at bin b from innings h' != h. Exact
    leave-one-out (no leakage from the current innings), vectorised.
    """
    base = (
        df.groupby(bin_cols, sort=False)[value_col]
        .agg(["sum", "size"])
        .rename(columns={"sum": "sum_all", "size": "n_all"})
        .reset_index()
    )
    own = (
        df.groupby(["match_id", "innings"] + bin_cols, sort=False)[value_col]
        .agg(["sum", "size"])
        .rename(columns={"sum": "sum_own", "size": "n_own"})
        .reset_index()
    )
    own = own.merge(base, on=bin_cols, how="left", validate="many_to_one")
    den = own["n_all"] - own["n_own"]
    own[state_col] = (own["sum_all"] - own["sum_own"]) / den.replace(0, np.nan)

    key = df[["match_id", "innings"] + bin_cols]
    out = key.merge(
        own[["match_id", "innings"] + bin_cols + [state_col]],
        on=["match_id", "innings"] + bin_cols,
        how="left",
        validate="many_to_one",
    )
    return pd.Series(out[state_col].values, index=df.index, name=state_col)


def add_par_state(
    df: pd.DataFrame,
    venue_par_k: float = 8.0,
) -> pd.DataFrame:
    """Historical expected-scoring context for every delivery.

    - expected_score_at_state: LOO mean of score_before at (over, wickets)
    - expected_runs_remaining: LOO mean of (final total - score) at
      (balls-left-over, wickets); separate curves for innings 1 and 2
    - expected_final_score = expected_score_at_state + expected_runs_remaining
    - venue-season par with shrinkage toward season par (LOO for own match)
    - par_score_before = venue-adjusted expected final score
    - score_vs_par / score_deficit_vs_par = score_before - expected_score_at_state
    """
    df = df.copy()
    final_total = (
        df.groupby(["match_id", "innings"], sort=False)["total_runs"].transform("sum")
    )
    df["_final_total"] = final_total
    df["_runs_remaining_after"] = df["_final_total"] - df["score_before"]
    df["_balls_left_over_bin"] = (df["legal_balls_remaining"] // 6).astype("int64")
    df["_balls_elapsed_over_bin"] = (df["legal_balls_before"] // 6).astype("int64")

    # fallback tables (wicket-level and global) per innings type
    curves: dict[str, dict[int, pd.Series]] = {}

    for inn_type in (1, 2):
        mask = df["innings"].eq(inn_type)
        sub = df[mask]
        if len(sub) == 0:
            continue
        for value_col, state_name, bins in [
            ("score_before", "expected_score_at_state",
             ["_balls_elapsed_over_bin", "wickets_before"]),
            ("_runs_remaining_after", "expected_runs_remaining",
             ["_balls_left_over_bin", "wickets_before"]),
        ]:
            curve = _loo_curve(sub, value_col, bins, state_name)
            # fallback: wickets-only curve
            wk_curve = _loo_curve(sub, value_col, ["wickets_before"], state_name)
            curve = curve.fillna(wk_curve)
            # fallback: global mean (LOO at innings level)
            g = sub.groupby(["match_id", "innings"], sort=False)[value_col].agg(["sum", "size"])
            glob = (g["sum"].sum() - g["sum"]) / (g["size"].sum() - g["size"])
            glob_map = glob.reindex(
                pd.MultiIndex.from_arrays(
                    [sub["match_id"], sub["innings"]], names=["match_id", "innings"]
                )
            )
            curve = curve.fillna(pd.Series(glob_map.values, index=sub.index))
            curves.setdefault(state_name, {})[inn_type] = curve

    for name, by_inn in curves.items():
        combined = pd.concat(by_inn.values()).sort_index()
        df[name] = combined.reindex(df.index)
    df["expected_final_score"] = (
        df["expected_score_at_state"] + df["expected_runs_remaining"]
    )
    df["score_vs_par"] = df["score_before"] - df["expected_score_at_state"]
    df["score_deficit_vs_par"] = df["score_vs_par"]

    # venue-season par (innings-1 totals), leave-one-match-out, shrinkage
    inn1 = df[df["innings"].eq(1)]
    v = (
        inn1.groupby(["venue_id", "season"], sort=False)
        .agg(v_sum=("_final_total", "sum"), v_n=("_final_total", "size"))
        .reset_index()
    )
    own_inn1 = (
        inn1.groupby(["match_id", "venue_id", "season"], sort=False)["_final_total"]
        .sum()
        .rename("own_total")
        .reset_index()
    )
    seas = (
        inn1.groupby("season", sort=False)
        .agg(s_sum=("_final_total", "sum"), s_n=("_final_total", "size"))
        .reset_index()
    )
    own_match = (
        inn1.groupby("match_id", sort=False)["_final_total"].sum().rename("own_match_total")
    )
    vv = inn1.merge(v, on=["venue_id", "season"], how="left", validate="many_to_one")
    vv = vv.merge(seas, on="season", how="left", validate="many_to_one")
    vv = vv.merge(own_match, on="match_id", how="left", validate="many_to_one")
    vv["venue_par"] = (
        (vv["v_sum"] - vv["own_match_total"] + venue_par_k * ((vv["s_sum"] - vv["own_match_total"]) / (vv["s_n"] - 1)))
        / (vv["v_n"] - 1 + venue_par_k)
    )
    par_series = pd.Series(vv["venue_par"].values, index=inn1.index)
    df["venue_season_par"] = par_series.reindex(df.index)
    glob_par = float(inn1["_final_total"].mean())
    df["venue_season_par"] = df["venue_season_par"].fillna(glob_par)

    sched_overs = (df["scheduled_legal_balls"] / 6.0).replace(0, np.nan)
    df["par_run_rate"] = df["venue_season_par"] / sched_overs
    df["par_score_before"] = (
        df["expected_final_score"]
        + (df["venue_season_par"] - glob_par) * 0.5
    )
    df["par_score_before"] = df["par_score_before"].fillna(df["expected_final_score"])

    df = df.drop(columns=["_final_total", "_runs_remaining_after",
                          "_balls_left_over_bin", "_balls_elapsed_over_bin"])
    return df


# ---------------------------------------------------------------------------
# Recent momentum features (Step 10) â€” strictly pre-delivery rolling windows
# ---------------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling recent-form features over the last W deliveries of the innings.

    Windows operate on delivery rows (including extras rows) in chronological
    order; every window is computed on the *previous* W rows only (shifted by
    one), so no current-delivery outcome is ever included.
    """
    df = df.copy()
    g = df.groupby(["match_id", "innings"], sort=False)

    base = pd.DataFrame(index=df.index)
    base["match_id"] = df["match_id"]
    base["innings"] = df["innings"]
    base["runs"] = df["total_runs"]
    base["wkts"] = df["is_team_wicket"].astype("int8")
    base["legal"] = df["is_legal_ball"].astype("int8")
    base["dots"] = df["is_dot_ball"].astype("int8")
    base["bounds"] = df["is_boundary"].astype("int8")
    base["bowler_charged"] = (
        df["extras_type"].isin(["wide", "no_ball"]).fillna(False).astype("int8")
    )

    for w in windows:
        r = base.groupby(["match_id", "innings"], sort=False).rolling(
            w, min_periods=1
        )
        agg = r.sum().reset_index(level=[0, 1], drop=True)
        # shift by one row inside each group: previous W deliveries only
        agg_shift = agg.groupby(
            [df["match_id"], df["innings"]], sort=False
        ).shift(1)
        prefix = f"last{w}_"
        df[prefix + "runs"] = agg_shift["runs"]
        df[prefix + "wickets"] = agg_shift["wkts"]
        df[prefix + "legal_balls"] = agg_shift["legal"]
        df[prefix + "dot_balls"] = agg_shift["dots"]
        df[prefix + "boundaries"] = agg_shift["bounds"]
        df[prefix + "bowler_charged_extras"] = agg_shift["bowler_charged"]
        df[prefix + "run_rate"] = np.where(
            (agg_shift["legal"].fillna(0) > 0),
            agg_shift["runs"] / (agg_shift["legal"] / 6.0),
            np.nan,
        )
        df[prefix + "wickets_per_ball"] = np.where(
            agg_shift["legal"].fillna(0) > 0,
            agg_shift["wkts"] / agg_shift["legal"].replace(0, np.nan),
            np.nan,
        )

    # consecutive dot balls immediately before this delivery (streak)
    dot = base["dots"]
    block = (~dot.astype(bool)).groupby(
        [df["match_id"], df["innings"]], sort=False
    ).cumsum()
    streak = dot.groupby(
        [df["match_id"], df["innings"], block], sort=False
    ).cumsum()
    df["consecutive_dots_before"] = (streak - dot).astype("int64")

    # balls since last boundary / team wicket (position-difference method)
    grp_keys = [df["match_id"], df["innings"]]
    pos = df.groupby(grp_keys, sort=False).cumcount().astype("int64")

    def _since(flag: pd.Series, name: str) -> pd.Series:
        last = (
            pd.Series(np.where(flag.astype(bool), pos, np.nan), index=df.index)
            .groupby(grp_keys, sort=False)
            .ffill()
        )
        return (pos - last).astype("float64")

    df["balls_since_last_boundary"] = _since(base["bounds"], "bnd")
    df["balls_since_last_team_wicket"] = _since(base["wkts"], "wkt")

    # boundary streak: consecutive boundary deliveries immediately before
    bblock = (~base["bounds"].astype(bool)).groupby(
        grp_keys, sort=False
    ).cumsum()
    bstreak = base["bounds"].groupby(
        [df["match_id"], df["innings"], bblock], sort=False
    ).cumsum()
    df["boundary_streak_before"] = (bstreak - base["bounds"]).astype("int64")
    return df
