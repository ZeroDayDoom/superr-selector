"""Player-level pre-delivery state (striker and bowler perspectives).

All quantities are computed with exclusive cumsums / shifted windows so the
current delivery's outcome never enters its own state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _excl_cumsum(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """Cumulative sum of col within groups defined by keys, excluding current row."""
    return df.groupby(keys, sort=False)[col].cumsum() - df[col]


def add_striker_state(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Striker-centric pre-delivery state."""
    df = df.copy()
    striker_keys = ["match_id", "innings", "striker_id"]

    df["_legal_faced"] = (
        df["is_legal_ball"].astype("int8").where(df["striker_id"].notna(), 0)
    )

    df["batter_balls_faced_before"] = _excl_cumsum(
        df, striker_keys, "_legal_faced"
    ).astype("int64")
    df["batter_runs_before"] = _excl_cumsum(
        df, striker_keys, "batter_runs"
    ).astype("float64")
    df["batter_boundaries_before"] = _excl_cumsum(
        df, striker_keys, "is_boundary"
    ).astype("int64")

    # entry situation: state at the batter's first delivery of the innings
    g2 = df.groupby(striker_keys, sort=False)
    first_idx = g2.cumcount().eq(0)
    skey = [df["match_id"], df["innings"], df["striker_id"]]
    entry_wkts = df["wickets_before"].where(first_idx).groupby(
        skey, sort=False
    ).ffill().bfill()
    df["batter_entry_wickets"] = entry_wkts
    df["batter_entry_ball"] = (
        df["legal_balls_before"].where(first_idx).groupby(
            skey, sort=False
        ).ffill().bfill()
    )

    new_balls = int(cfg["new_batter_balls"])
    set_balls = int(cfg["set_batter_balls"])
    df["batter_is_new"] = df["batter_balls_faced_before"].lt(new_balls)
    df["batter_is_set"] = df["batter_balls_faced_before"].ge(set_balls)

    # partnership: state since the last team wicket. The block id is the
    # PRE-delivery wicket count, so the wicket delivery itself belongs to the
    # outgoing partnership and the next delivery starts a fresh one.
    df["_partnership_block"] = df["wickets_before"]
    p_keys = ["match_id", "innings", "_partnership_block"]
    df["partnership_runs_before"] = _excl_cumsum(
        df, p_keys, "total_runs"
    ).astype("float64")
    df["partnership_balls_before"] = _excl_cumsum(
        df, p_keys, "is_legal_ball"
    ).astype("int64")

    # striker strike-rotation correctness signal: striker changed after a
    # legal ball with an odd number of runs, or after a wicket (informational
    # only; validated in validate_data).
    df = df.drop(columns=["_legal_faced", "_partnership_block"])
    return df


def add_bowler_state(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Bowler-centric pre-delivery state within the innings."""
    df = df.copy()
    b_keys = ["match_id", "innings", "bowler_id"]

    df["_legal_bowled"] = df["is_legal_ball"].astype("int8")
    df["bowler_balls_before"] = _excl_cumsum(
        df, b_keys, "_legal_bowled"
    ).astype("int64")
    df["bowler_overs_completed"] = (df["bowler_balls_before"] // 6).astype("int64")
    df["bowler_runs_conceded_before"] = _excl_cumsum(
        df, b_keys, "bowler_runs_conceded"
    ).astype("float64")
    df["bowler_wickets_before"] = _excl_cumsum(
        df, b_keys, "is_bowler_wicket"
    ).astype("int64")
    df["bowler_dots_before"] = _excl_cumsum(
        df, b_keys, "is_dot_ball"
    ).astype("int64")

    # spell: consecutive overs by the same bowler immediately before this over
    over_bowler = (
        df.groupby(["match_id", "innings", "overid"], sort=False)["bowler_id"]
        .first()
        .rename("over_bowler")
        .reset_index()
    )
    over_bowler["prev_bowler"] = over_bowler.groupby(
        ["match_id", "innings"]
    )["over_bowler"].shift(1)
    over_bowler["prev_same"] = over_bowler["over_bowler"].eq(
        over_bowler["prev_bowler"]
    )
    # spell length at each over = consecutive same-bowler overs ending here
    grp = over_bowler.groupby(["match_id", "innings"], sort=False)
    block = (~over_bowler["prev_same"].fillna(False)).groupby(
        [over_bowler["match_id"], over_bowler["innings"]]
    ).cumsum()
    over_bowler["spell_at_over"] = (
        over_bowler.groupby(
            [over_bowler["match_id"], over_bowler["innings"], block], sort=False
        ).cumcount()
        + 1
    )
    df = df.merge(
        over_bowler[["match_id", "innings", "overid", "spell_at_over"]],
        on=["match_id", "innings", "overid"],
        how="left",
        validate="many_to_one",
    )
    df["bowler_spell_overs_before"] = (df["spell_at_over"] - 1).astype("int64")
    df = df.drop(columns=["spell_at_over"])

    # recent form of this bowler within the innings (shifted rolling windows)
    b = pd.DataFrame(index=df.index)
    b["match_id"] = df["match_id"]
    b["innings"] = df["innings"]
    b["bowler_id"] = df["bowler_id"]
    b["runs"] = df["bowler_runs_conceded"]
    b["dots"] = df["is_dot_ball"].astype("int8")
    b["wkts"] = df["is_bowler_wicket"].astype("int8")
    b["bounds"] = df["is_boundary"].astype("int8")
    for w in windows:
        r = b.groupby(["match_id", "innings", "bowler_id"], sort=False).rolling(
            w, min_periods=1
        )
        agg = r.sum().reset_index(level=[0, 1, 2], drop=True)
        agg_shift = agg.groupby(
            [df["match_id"], df["innings"], df["bowler_id"]], sort=False
        ).shift(1)
        p = f"bowler_last{w}_"
        df[p + "runs"] = agg_shift["runs"]
        df[p + "dots"] = agg_shift["dots"]
        df[p + "wickets"] = agg_shift["wkts"]
        df[p + "boundaries"] = agg_shift["bounds"]

    # bowler's team wicket drought (balls since this bowling team took a wicket)
    pos = df.groupby(["match_id", "innings"], sort=False).cumcount().astype("int64")
    flag = df["is_team_wicket"].astype(bool)
    last_wkt = (
        pd.Series(np.where(flag, pos, np.nan), index=df.index)
        .groupby([df["match_id"], df["innings"]], sort=False)
        .ffill()
    )
    df["bowling_team_wicket_drought"] = (pos - last_wkt).astype("float64")

    # boundary streak conceded by this bowler (immediately before)
    bb = (~df["is_boundary"].astype(bool)).groupby(
        [df["match_id"], df["innings"], df["bowler_id"]]
    ).cumsum()
    bstreak = df["is_boundary"].astype("int8").groupby(
        [df["match_id"], df["innings"], df["bowler_id"], bb]
    ).cumsum()
    df["bowler_boundary_streak_before"] = (
        bstreak - df["is_boundary"].astype("int8")
    ).astype("int64")

    # matchup: legal balls this striker has already faced from this bowler
    df["matchup_balls_before"] = _excl_cumsum(
        df, ["match_id", "innings", "bowler_id", "striker_id"], "_legal_bowled"
    ).astype("int64")
    df = df.drop(columns=["_legal_bowled"])
    return df


def add_batter_roles(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Data-driven batter roles from actual batting positions + death exposure."""
    df = df.copy()
    bands = cfg["role_bands"]

    def band(order: float) -> str:
        if pd.isna(order):
            return "unknown"
        o = int(order)
        for role, members in bands.items():
            if o in members:
                return role
        return "unknown"

    df["batter_role"] = df["batting_order"].map(band)

    # finisher refinement: middle-order with meaningful death-overs exposure
    death = (
        df.assign(_death=df["over_number_before"].ge(16) & df["is_legal_ball"])
        .groupby(["match_id", "innings", "striker_id"])
        .agg(death_balls=("_death", "sum"), total_balls=("is_legal_ball", "sum"))
        .reset_index()
    )
    death["death_share"] = death["death_balls"] / death["total_balls"].replace(0, np.nan)
    death["is_finisher_innings"] = death["death_share"].ge(
        float(cfg["finisher_death_share"])
    )
    finisher_rate = (
        death.groupby("striker_id")["is_finisher_innings"].mean()
        .rename("finisher_rate")
    )
    df = df.merge(finisher_rate, left_on="striker_id", right_index=True,
                  how="left", validate="many_to_one")
    df["batter_role"] = np.where(
        (df["batter_role"] == "middle") & df["finisher_rate"].fillna(0).ge(0.5),
        "finisher",
        df["batter_role"],
    )
    df = df.drop(columns=["finisher_rate"])
    return df


def add_bowler_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Data-driven bowler phase roles from overs actually bowled."""
    df = df.copy()
    phase = np.select(
        [df["over_number_before"].le(6), df["over_number_before"].le(15)],
        ["powerplay", "middle"],
        default="death",
    )
    df["bowling_phase"] = phase

    per_phase = (
        df.assign(_legal=df["is_legal_ball"].astype("int8"))
        .groupby(["match_id", "innings", "bowler_id", "bowling_phase"])["_legal"]
        .sum()
        .unstack("bowling_phase")
        .fillna(0)
        .reset_index()
    )
    for col in ("powerplay", "middle", "death"):
        if col not in per_phase.columns:
            per_phase[col] = 0
    tot = per_phase[["powerplay", "middle", "death"]].sum(axis=1).replace(0, np.nan)

    def role(pp: float, mid: float, dth: float, t: float) -> str:
        shares = {"powerplay": pp / t, "middle": mid / t, "death": dth / t}
        main = max(shares, key=shares.get)
        others = [v for k, v in shares.items() if k != main]
        if max(others) >= 0.30:
            return f"hybrid-{main}"
        return main

    per_phase["bowler_role"] = [
        role(row.powerplay, row.middle, row.death, t)
        for row, t in zip(per_phase.itertuples(index=False), tot)
    ]
    role_map = per_phase.set_index(["match_id", "innings", "bowler_id"])["bowler_role"]
    key = pd.MultiIndex.from_arrays(
        [df["match_id"], df["innings"], df["bowler_id"]]
    )
    df["bowler_role"] = role_map.reindex(key).values
    return df


def add_career_dismissal_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Career-to-date dismissal risk per batter, prior matches only.

    Expanding rate of dismissals per legal ball faced, shifted by one match so
    the current match can never inform its own risk. Ordered by match_date.
    """
    df = df.copy()
    striker_rows = (
        df[df["is_legal_ball"]]
        .groupby(["match_id", "striker_id"])
        .agg(
            balls=("is_legal_ball", "size"),
            match_date=("match_date", "first"),
        )
        .reset_index()
    )
    dism = (
        df[df["is_batter_dismissal"] & df["dismissed_is_striker"]]
        .groupby(["match_id", "striker_id"])
        .size()
        .rename("dismissals")
        .reset_index()
    )
    pm = striker_rows.merge(dism, on=["match_id", "striker_id"], how="left",
                            validate="one_to_one")
    pm["dismissals"] = pm["dismissals"].fillna(0).astype("float64")
    pm = pm.sort_values(["striker_id", "match_date", "match_id"], kind="mergesort")
    pg = pm.groupby("striker_id", sort=False)
    pm["cum_balls_prior"] = pg["balls"].cumsum() - pm["balls"]
    pm["cum_dismissals_prior"] = pg["dismissals"].cumsum() - pm["dismissals"]
    pm["dismissal_risk_career"] = np.where(
        pm["cum_balls_prior"] >= 30,
        pm["cum_dismissals_prior"] / pm["cum_balls_prior"].replace(0, np.nan),
        np.nan,
    )
    risk = pm.set_index(["match_id", "striker_id"])["dismissal_risk_career"]
    key = pd.MultiIndex.from_arrays([df["match_id"], df["striker_id"]])
    df["striker_dismissal_risk_career"] = risk.reindex(key).values
    return df
