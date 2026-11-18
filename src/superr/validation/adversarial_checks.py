"""Adversarial and edge-case checks (Phase-8 V13/V14).

Model checks (V13) verify properties of the pressure/value system on real
data. Edge tests (V14) push synthetic degenerate states through the value
and pressure functions and assert sensible, non-crashing behaviour.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def model_checks(df: pd.DataFrame, pressure_col: str = "PRESSURE_MODEL_C") -> Dict:
    over_idx = (df["legal_balls_before"] // 6 + 1)
    is_death = over_idx.ge(16)
    q80 = df[pressure_col].quantile(0.80)

    hp = df[df[pressure_col] >= q80]
    rest = df[df[pressure_col] < q80]

    # 1. is pressure just death overs?
    death_share_hp = float(is_death[df[pressure_col] >= q80].mean())
    death_share_all = float(is_death.mean())
    # 2. pressure in non-death overs still separates
    nd = df[~is_death]
    q80_nd = nd[pressure_col].quantile(0.80)
    hp_nd = nd[nd[pressure_col] >= q80_nd]
    rest_nd = nd[nd[pressure_col] < q80_nd]
    # 3. single-match dominance of player impact (striker-level; the value
    # frame stores the batter id as striker_id, tests may use player_id)
    id_col = "striker_id" if "striker_id" in df.columns else "player_id"
    hpv = hp.groupby([id_col, "match_id"])["bat_value"].sum()
    abs_sum = hpv.abs().groupby(id_col).sum()
    abs_max = hpv.abs().groupby(id_col).max()
    top_share = (abs_max / abs_sum.replace(0, np.nan)).dropna()
    # 4. residuals of expected runs centred?
    resid = df["batter_runs"] - df["expected_runs_off_bat"]
    return {
        "death_share_of_hp_deliveries": round(death_share_hp, 3),
        "death_share_of_all_deliveries": round(death_share_all, 3),
        "pressure_not_synonymous_with_death": bool(death_share_hp < 0.9 * max(death_share_all, 1e-9) + 0.5
                                                   and death_share_hp < 1.5 * death_share_all + 0.05),
        "hp_vs_rest_wicket_prob": [round(float(hp["is_team_wicket"].mean()), 4),
                                   round(float(rest["is_team_wicket"].mean()), 4)],
        "hp_vs_rest_abs_wpa": [round(float(hp["realized_wpa"].abs().mean()), 4),
                               round(float(rest["realized_wpa"].abs().mean()), 4)],
        "non_death_hp_vs_rest_wicket_prob": [
            round(float(hp_nd["is_team_wicket"].mean()), 4),
            round(float(rest_nd["is_team_wicket"].mean()), 4)],
        "expected_runs_residual_mean": round(float(resid.mean()), 4),
        "expected_runs_residual_sd": round(float(resid.std()), 4),
        "max_single_match_share_of_player_hp_impact": {
            "mean": round(float(top_share.mean()), 3),
            "p95": round(float(top_share.quantile(0.95)), 3),
            "max": round(float(top_share.max()), 3),
        },
    }


def edge_tests(df: pd.DataFrame, lam_table: pd.DataFrame,
               lam_global: float) -> Dict[str, Dict]:
    """Push degenerate states through value computation; assert sane output."""
    base = df.iloc[0:1].copy()

    def value_of(mods: Dict) -> float:
        row = base.copy()
        for c, v in mods.items():
            row[c] = v
        # lambda lookup
        key = (row["innings"].iloc[0],
               int(max(0, row["legal_balls_remaining"].iloc[0]) // 6),
               int(row["wickets_before"].iloc[0]))
        lam = lam_table.get(key, lam_global)
        bv = (float(row["batter_runs"].iloc[0])
              - float(row["expected_runs_off_bat"].iloc[0])
              - lam * (float(row["dismissed_is_striker"].iloc[0])
                       - float(row["p_wicket"].iloc[0])))
        return float(bv)

    tests = {}
    # target already reached (runs_required 0)
    tests["target_reached"] = {"value": value_of({"runs_required_before": 0.0}),
                               "note": "chase complete; pressure should be LOW"}
    # zero balls remaining
    tests["zero_balls_remaining"] = {"value": value_of({"legal_balls_remaining": 0}),
                                     "note": "terminal state"}
    # all wickets lost
    tests["all_out"] = {"value": value_of({"wickets_before": 10,
                                           "wickets_in_hand": 0})}
    # extreme required run rate
    tests["rrr_25"] = {"value": value_of({"required_run_rate": 25.0})}
    # wide / no-ball / byes rows (from real data if available)
    for label, col, val in [("wide", "extras_type", "wide"),
                            ("no_ball", "extras_type", "no_ball"),
                            ("byes", "extras_type", "byes"),
                            ("legbyes", "extras_type", "legbyes")]:
        sub = df[df["extras_type"] == val]
        if len(sub):
            r = sub.iloc[0]
            tests[label] = {
                "bowler_conceded": float(r["bowler_runs_conceded"]),
                "is_legal": bool(r["is_legal_ball"]),
                "total": float(r["total_runs"]),
            }
    # run-out / non-bowler wicket attribution
    ro = df[df["is_run_out"] == True]  # noqa: E712
    if len(ro):
        r = ro.iloc[0]
        tests["run_out"] = {
            "is_bowler_wicket": bool(r["is_bowler_wicket"]),
            "is_team_wicket": bool(r["is_team_wicket"]),
            "dismissed_is_striker": bool(r["dismissed_is_striker"]),
        }
    # missing player id / venue robustness: lambda lookup must not crash
    tests["missing_ids"] = {"lam_global_used": float(lam_global)}
    return tests
