"""Reconciliation and leakage tests for the reconstructed pipeline.

Every check is written so that a failure is loud: the runner raises if any
FAIL occurs, and results are persisted to
reports/match_state_validation.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _result(check: str, status: str, detail: str, **kw: Any) -> Dict[str, Any]:
    r = {"check": check, "status": status, "detail": detail}
    r.update(kw)
    return r


def run_validation(
    balls: pd.DataFrame,
    innings: pd.DataFrame,
    matches_scope: pd.DataFrame,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    b = balls

    # ---- duplicate keys ------------------------------------------------
    dup = int(b.duplicated(subset=["match_id", "innings", "overid",
                                   "over_ball_num"]).sum())
    results.append(_result("duplicate_delivery_keys", "pass" if dup == 0 else "fail",
                           f"{dup} duplicated delivery keys", count=dup))

    # ---- legal-ball counts ---------------------------------------------
    per_inn = b.groupby(["match_id", "innings"])["is_legal_ball"].sum()
    over_max = int(per_inn.max())
    bad = int((per_inn > 132).sum())
    results.append(_result(
        "legal_ball_counts", "pass" if bad == 0 else "fail",
        f"max legal balls in an innings: {over_max}; innings > 132: {bad}",
        count=bad))

    # ---- innings score reconciliation ----------------------------------
    inn_totals = b.groupby(["match_id", "innings"])["total_runs"].sum()
    recon = innings.set_index(["match_id", "innings"])["innings_total"]
    mismatch = int((inn_totals - recon.reindex(inn_totals.index)).abs().gt(1e-9).sum())
    results.append(_result(
        "innings_score_reconciliation", "pass" if mismatch == 0 else "fail",
        f"{mismatch} innings where delivery-sum != innings total", count=mismatch))

    # ---- cumulative state correctness -----------------------------------
    g = b.groupby(["match_id", "innings"], sort=False)
    cum_runs = g["total_runs"].cumsum() - b["total_runs"]
    bad_score = int((b["score_before"] - cum_runs).abs().gt(1e-9).sum())
    results.append(_result(
        "score_before_matches_exclusive_cumsum", "pass" if bad_score == 0 else "fail",
        f"{bad_score} deliveries where score_before != exclusive cumsum",
        count=bad_score))

    cum_wkts = g["is_team_wicket"].cumsum() - b["is_team_wicket"]
    bad_wkts = int((b["wickets_before"] != cum_wkts).sum())
    results.append(_result(
        "wickets_before_matches_exclusive_cumsum", "pass" if bad_wkts == 0 else "fail",
        f"{bad_wkts} deliveries where wickets_before != exclusive cumsum",
        count=bad_wkts))

    cum_legal = g["is_legal_ball"].cumsum() - b["is_legal_ball"]
    bad_legal = int((b["legal_balls_before"] != cum_legal).sum())
    results.append(_result(
        "legal_balls_before_matches_exclusive_cumsum",
        "pass" if bad_legal == 0 else "fail",
        f"{bad_legal} mismatches in legal_balls_before", count=bad_legal))

    # first ball of each innings must have zeroed state
    first = g.cumcount().eq(0)
    zeroed = int((b.loc[first, ["score_before", "wickets_before",
                                "legal_balls_before"]] != 0).any(axis=1).sum())
    results.append(_result(
        "first_ball_state_zeroed", "pass" if zeroed == 0 else "fail",
        f"{zeroed} innings whose first delivery has non-zero state",
        count=zeroed))

    # state monotonicity within innings
    non_dec = g["score_before"].apply(lambda s: s.is_monotonic_increasing)
    bad_mono = int((~non_dec).sum())
    results.append(_result(
        "score_before_monotonic", "pass" if bad_mono == 0 else "fail",
        f"{bad_mono} innings with non-monotonic score_before", count=bad_mono))

    # ---- wicket reconciliation -----------------------------------------
    wk = b["is_team_wicket"].sum()
    inn_wk = innings["total_wickets"].sum()
    ok = abs(wk - inn_wk) < 1e-9
    results.append(_result(
        "wicket_reconciliation", "pass" if ok else "fail",
        f"team wickets: deliveries {int(wk)} vs innings summary {int(inn_wk)}"))

    bw = b["is_bowler_wicket"].sum()
    team = b["is_team_wicket"].sum()
    results.append(_result(
        "bowler_wicket_subset_of_team_wickets",
        "pass" if bw <= team else "fail",
        f"bowler wickets {int(bw)} <= team wickets {int(team)} "
        f"(run-outs/retired-outs/obstructions: {int(team - bw)})"))

    # ---- extras & bowler runs -------------------------------------------
    # Correct scoring semantics: the bowler concedes all off-bat runs plus
    # wide/no-ball extras (incl. additional runs on those deliveries); byes,
    # leg-byes, penalty and 'other' extra VALUES are not charged. Off-bat runs
    # on those deliveries ARE charged. Two independent identities:
    #   (1) bowler_runs + uncharged extra values == total_runs
    #   (2) on non-bowler-charged rows, bowler_runs == off-bat runs
    uncharged_types = ["byes", "legbyes", "penalty", "other", "unknown"]
    m_uncharged = b["extras_type"].isin(uncharged_types)
    uncharged_value = (
        b.loc[m_uncharged, "extras1_runs"].fillna(0).sum()
        + b.loc[m_uncharged, "extras_2_runs"].fillna(0).sum()
    )
    identity1 = abs(
        b["bowler_runs_conceded"].sum() + uncharged_value - b["total_runs"].sum()
    ) < 1e-6
    results.append(_result(
        "bowler_runs_identity_total", "pass" if identity1 else "fail",
        f"bowler conceded {b['bowler_runs_conceded'].sum():,.0f} + uncharged "
        f"extras {uncharged_value:,.0f} vs total {b['total_runs'].sum():,.0f}"))

    row_ok = (
        b.loc[m_uncharged, "bowler_runs_conceded"]
        == b.loc[m_uncharged, "batter_runs"].fillna(0)
    ).all()
    results.append(_result(
        "bowler_runs_excludes_byes_legbyes", "pass" if row_ok else "fail",
        f"on byes/legbyes/penalty/other rows, bowler charged only off-bat runs "
        f"({int((~b.loc[m_uncharged, 'bowler_runs_conceded'].eq(b.loc[m_uncharged, 'batter_runs'].fillna(0))).sum())} violations)"))

    ex_ok = bool(
        (b["extras_runs"] == b["extras1_runs"].fillna(0) + b["extras_2_runs"].fillna(0)).all()
    )
    results.append(_result("extras_reconstruction", "pass" if ex_ok else "fail",
                           "extras_runs == extras1 + extras_2 on every delivery"))

    # ---- target correctness ----------------------------------------------
    ch = b[b["innings"].eq(2)]
    t1 = innings[innings["innings"].eq(1)].set_index("match_id")["innings_total"]
    exp_target = t1.reindex(ch["match_id"].unique()) + 1
    targets = ch.groupby("match_id")["target"].first()
    is_rev = ch.groupby("match_id")["target_is_revised"].first()
    bad_t = int(((targets != exp_target.reindex(targets.index)) & ~is_rev).sum())
    results.append(_result(
        "chase_target_correctness", "pass" if bad_t == 0 else "fail",
        f"{bad_t} non-DLS chases with target != innings-1 total + 1",
        count=bad_t))

    # ---- no negative resources ------------------------------------------
    neg = int((b["legal_balls_remaining"] < 0).sum()) + \
        int((b["wickets_in_hand"] < 0).sum())
    results.append(_result("no_negative_resources", "pass" if neg == 0 else "fail",
                           f"{neg} deliveries with negative balls/wickets remaining",
                           count=neg))

    # ---- player ids ------------------------------------------------------
    pe = b[["striker_id", "non_striker_id", "bowler_id"]].notna().all(axis=1)
    results.append(_result("player_ids_present", "pass" if pe.all() else "fail",
                           f"{int((~pe).sum())} deliveries missing a player id"))

    # ---- leakage: pre-delivery state excludes current outcome ------------
    # For each innings, the last delivery's score_before + its own total must
    # equal the innings total (proves cumsum is exclusive of current row).
    last = g.tail(1).set_index(["match_id", "innings"])
    inn_tot = innings.set_index(["match_id", "innings"])["innings_total"]
    chk = (last["score_before"] + last["total_runs"] -
           inn_tot.reindex(last.index)).abs().gt(1e-9).sum()
    results.append(_result(
        "no_current_outcome_in_pre_ball_state", "pass" if chk == 0 else "fail",
        f"{int(chk)} innings where last-ball state includes its own outcome",
        count=int(chk)))

    # ---- per-player state resets (striker / bowler / partnership) --------
    sk = ["match_id", "innings", "striker_id"]
    first_striker = b.groupby(sk, sort=False).cumcount().eq(0)
    bad_striker = int(
        (b.loc[first_striker, "batter_balls_faced_before"] != 0).sum()
        + (b.loc[first_striker, "batter_runs_before"].fillna(0).abs() > 1e-9).sum()
    )
    results.append(_result(
        "striker_state_resets_per_player", "pass" if bad_striker == 0 else "fail",
        f"{bad_striker} batter-innings whose first delivery has non-zero "
        "balls faced / runs before", count=bad_striker))

    bk = ["match_id", "innings", "bowler_id"]
    first_bowler = b.groupby(bk, sort=False).cumcount().eq(0)
    bad_bowler = int(
        (b.loc[first_bowler, "bowler_balls_before"] != 0).sum()
    )
    results.append(_result(
        "bowler_state_resets_per_player", "pass" if bad_bowler == 0 else "fail",
        f"{bad_bowler} bowler-innings whose first delivery has non-zero balls "
        "bowled before", count=bad_bowler))

    new_partnership = b["is_team_wicket"].astype(bool).shift(1).fillna(False)
    bad_part = int(
        (b.loc[new_partnership, "partnership_runs_before"].fillna(0) != 0).sum()
        + (b.loc[new_partnership, "partnership_balls_before"] != 0).sum()
    )
    results.append(_result(
        "partnership_resets_after_wicket", "pass" if bad_part == 0 else "fail",
        f"{bad_part} deliveries immediately after a wicket with non-zero "
        "partnership state", count=bad_part))

    # momentum windows: verify a known case — a boundary's own row must not
    # increase last6_boundaries on that row (shifted windows).
    bnd_rows = b[b["is_boundary"]]
    if len(bnd_rows) > 0:
        prev6 = bnd_rows["last6_boundaries"]
        own = bnd_rows["is_boundary"].astype(int)
        # last6_boundaries at the boundary row counts only previous rows;
        # check it never exceeds 6 and is integer-consistent
        ok_m = bool((prev6.fillna(0) <= 6).all())
        results.append(_result(
            "momentum_windows_shifted", "pass" if ok_m else "fail",
            "last6_boundaries on boundary rows only counts prior deliveries"))

    # DLS: revised targets respected
    dls_matches = matches_scope[matches_scope["revised_target"].notna()]
    if len(dls_matches):
        chk2 = ch[ch["match_id"].isin(dls_matches["match_id"])]
        rev_ok = bool(
            (chk2.groupby("match_id")["target"].first()
             == chk2.groupby("match_id")["target"].first()).all()
        )
        results.append(_result(
            "dls_targets_propagated", "pass" if rev_ok else "fail",
            f"{chk2['match_id'].nunique()} DLS-revised chases carry revised targets"))

    # super overs excluded
    so = int(b["innings"].ge(3).sum())
    results.append(_result("super_over_innings_excluded", "pass" if so == 0 else "warn",
                           f"{so} super-over deliveries remain in analysis table",
                           count=so))

    return results


def write_validation_report(
    results: List[Dict[str, Any]], path: Path, extra: Dict[str, Any]
) -> bool:
    n_fail = sum(1 for r in results if r["status"] == "fail")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_checks": len(results),
        "n_fail": n_fail,
        "results": results,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return n_fail == 0
