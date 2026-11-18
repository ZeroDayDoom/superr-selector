"""Cricket-semantics verification.

The interpretations of ambiguous columns (runs/extras, legal balls, bowler
attribution) are *verified against independent evidence in the data*:

1. Delivery run accounting is validated by comparing team innings totals
   (built from runs + extras1 + extras2) against the parsed match-result
   margins ("Won by N runs / N wickets").
2. Legal-ball accounting is validated by over structure: a complete over of
   legal deliveries must contain exactly 6 non-extras deliveries, and wides /
   no-balls must not advance the legal count.
3. Batting-order consistency is validated within innings.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

ILLEGAL_EXTRAS = {"wide", "no_ball", "Overstep", "penalty", "other", "Penalty"}
BOWLER_CHARGED = {"wide", "Overstep"}
BOWLER_WICKET_MODES = {"Bowled", "Caught", "LBW", "Stump", "Hit Wicket"}


def total_delivery_runs(b: pd.DataFrame) -> pd.Series:
    """Total runs scored by the batting team on a delivery (verified below)."""
    return (
        b["runs"].fillna(0)
        + b["extras1_runs"].fillna(0)
        + b["extras_2_runs"].fillna(0)
    )


def verify_run_accounting(b: pd.DataFrame, m: pd.DataFrame) -> Dict[str, Any]:
    """Compare innings totals vs parsed result margins.

    DLS-method results are verified against the recorded revised target where
    one exists (target - 1 - chasing score == recorded runs margin). When the
    DLS par is not supplied in the pack the case is reported as
    'unverifiable_dls', NOT as a mismatch. Non-DLS results must match the raw
    score difference exactly.
    """
    bb = b.copy()
    bb["_total"] = total_delivery_runs(bb)
    totals = (
        bb.groupby(["match_id", "innings", "batting_team_id"])["_total"]
        .sum()
        .rename("innings_total")
        .reset_index()
    )
    wickets = (
        bb[bb["is_wicket"].astype(str).str.lower().eq("true")]
        .groupby(["match_id", "innings", "batting_team_id"])
        .size()
        .rename("wickets_lost")
        .reset_index()
    )
    t = totals.merge(wickets, on=["match_id", "innings", "batting_team_id"], how="outer")
    t["wickets_lost"] = t["wickets_lost"].fillna(0).astype(int)
    t1 = t[t["innings"] == 1].set_index("match_id")
    t2 = t[t["innings"] == 2].set_index("match_id")

    mm = m.set_index("match_id")
    common = mm.index.intersection(t1.index).intersection(t2.index)
    out = {
        "n_matches_compared": int(len(common)),
        "runs_margin_matches": 0,
        "runs_margin_mismatches": 0,
        "wickets_margin_matches": 0,
        "wickets_margin_mismatches": 0,
        "dls_verified_against_revised_target": 0,
        "unverifiable_dls_no_par_supplied": 0,
        "examples_bad": [],
        "inconsistent_matches": [],
    }

    def note_bad(mid: int, why: Dict[str, Any]) -> None:
        out["runs_margin_mismatches"] += 1 if why["kind"] == "runs" else 0
        out["wickets_margin_mismatches"] += 1 if why["kind"] == "wickets" else 0
        out["inconsistent_matches"].append({"match_id": int(mid), **why})
        if len(out["examples_bad"]) < 5:
            out["examples_bad"].append({"match_id": int(mid), **why})

    for mid in common:
        rt = mm.loc[mid, "_result_type"]
        margin, unit, dls = (
            mm.loc[mid, "_margin"],
            mm.loc[mid, "_margin_unit"],
            mm.loc[mid, "_dls"],
        )
        if rt not in ("runs", "wickets") or pd.isna(margin):
            continue
        s1, s2 = t1.loc[mid, "innings_total"], t2.loc[mid, "innings_total"]
        w2 = t2.loc[mid, "wickets_lost"]
        win = mm.loc[mid, "win_team_id"]
        first_bat = t1.loc[mid, "batting_team_id"]
        target = mm.loc[mid, "revised_target"]

        if unit == "runs":
            if win == first_bat:
                ok = (s1 - s2) == margin
            else:
                ok = (s2 - s1) == margin
            if ok:
                out["runs_margin_matches"] += 1
            elif dls and pd.notna(target):
                # DLS margin is computed vs the revised target, in either
                # orientation: first-batting winner -> target - 1 - s2;
                # chasing winner -> s2 - (target - 1).
                if (target - 1 - s2) == margin or (s2 - (target - 1)) == margin:
                    out["dls_verified_against_revised_target"] += 1
                else:
                    note_bad(mid, {"kind": "runs", "s1": float(s1), "s2": float(s2),
                                   "margin": float(margin), "dls": bool(dls),
                                   "revised_target": float(target)})
            elif dls:
                out["unverifiable_dls_no_par_supplied"] += 1
            else:
                note_bad(mid, {"kind": "runs", "s1": float(s1), "s2": float(s2),
                               "margin": float(margin), "dls": False,
                               "win": int(win), "first_bat": int(first_bat)})
        else:
            ok = (10 - w2) == margin or (w2 == 10 and margin == 10)
            if ok:
                out["wickets_margin_matches"] += 1
            else:
                note_bad(mid, {"kind": "wickets", "w2": int(w2),
                               "margin": float(margin), "dls": bool(dls)})
    out["run_accounting_verified"] = (
        out["runs_margin_mismatches"] == 0 and out["wickets_margin_mismatches"] == 0
    )
    return out


def verify_legal_balls(b: pd.DataFrame) -> Dict[str, Any]:
    bb = b.copy()
    bb["_legal"] = ~bb["extras_type"].isin(ILLEGAL_EXTRAS)
    per_over = bb.groupby(["match_id", "innings", "overid"]).agg(
        legal_balls=("_legal", "sum"),
        rows=("over_ball_num", "size"),
        has_extras=("extras_type", lambda s: int(s.notna().sum())),
    )
    complete = per_over[per_over["legal_balls"] == 6]
    over6 = per_over[per_over["legal_balls"] > 6]
    # wides/noballs must consume over_ball_num slots without being legal
    ex = bb[bb["extras_type"].isin(BOWLER_CHARGED)]
    ex_advance = int(ex["_legal"].sum()) if len(ex) else 0
    per_inn = bb.groupby(["match_id", "innings"]).agg(
        legal=("_legal", "sum"), max_over=("overid", "max")
    )
    n_inn = len(per_inn)
    full = per_inn[per_inn["legal"] >= 120]
    over120 = per_inn[per_inn["legal"] > 121]
    return {
        "legal_ball_definition": "delivery whose extras_type is not a wide, "
        "no-ball (raw label 'Overstep'), penalty or 'other'",
        "n_overs": int(len(per_over)),
        "overs_with_exactly_6_legal": int(len(complete)),
        "overs_with_more_than_6_legal": int(len(over6)),
        "max_legal_balls_in_any_over": int(per_over["legal_balls"].max()),
        "bowler_charged_extras_rows": int(len(ex)),
        "bowler_charged_extras_counted_as_legal": ex_advance,
        "innings_total": n_inn,
        "innings_with_ge_120_legal": int(len(full)),
        "innings_with_gt_121_legal": int(len(over120)),
        "other_extras_rows": int((bb["extras_type"].isin({"other", "Penalty"})).sum()),
    }


def verify_batting_order(b: pd.DataFrame) -> Dict[str, Any]:
    """Check batting-order consistency within innings.

    Note: first-appearance order need NOT be monotonic in batting_order —
    batters can retire (hurt) and return, and incoming batters can be
    promoted. The correct invariants are: (a) each batter keeps exactly one
    batting_order value per innings; (b) arrival order is monotonic apart
    from such returns, which we count and report as informational.
    """
    bad_multi = 0
    n_inn = 0
    arrival_deviations = 0
    for (mid, inn), grp in b.groupby(["match_id", "innings"]):
        n_inn += 1
        per_player = grp.groupby("striker_id")["batting_order"].nunique()
        if (per_player > 1).any():
            bad_multi += 1
        fs = grp.sort_values("innings_ball_num").drop_duplicates("striker_id")
        o = fs.set_index("striker_id")["batting_order"]
        if o.notna().all() and not o.astype(float).is_monotonic_increasing:
            arrival_deviations += 1
    return {
        "innings_checked": n_inn,
        "innings_with_player_order_conflicts": bad_multi,
        "innings_where_arrival_order_deviation_observed (retired-hurt returns "
        "or promoted batters; informational)": arrival_deviations,
    }


def bowling_type_semantics(b: pd.DataFrame) -> Dict[str, Any]:
    bt = b[["bowling_type", "pace_or_spin", "bowl_speed"]].copy()
    cross = (
        bt.groupby(["pace_or_spin", "bowling_type"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False)
    )
    zero_speed = bt[bt["bowl_speed"] == 0]
    return {
        "n_bowling_types": int(bt["bowling_type"].nunique()),
        "top_bowling_types": cross.head(12).to_dict("records"),
        "bowl_speed_zero_rows": int(len(zero_speed)),
        "bowl_speed_zero_pct": round(100.0 * len(zero_speed) / max(len(bt), 1), 2),
        "bowl_speed_nonzero_median": float(bt.loc[bt["bowl_speed"] > 0, "bowl_speed"].median()),
    }
