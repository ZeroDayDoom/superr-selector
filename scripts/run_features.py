"""Phase-4 pipeline: feature engineering + registry + leakage audit.

Single command:
    python scripts/run_features.py

Outputs:
    data/processed/features_batting.parquet
    data/processed/features_bowling.parquet
    data/processed/features_pressure.parquet
    reports/feature_dictionary.md
    reports/feature_importance_hypotheses.md
    reports/leakage_audit.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superr.config import load_config  # noqa: E402
from superr.features.batting_pressure_features import (  # noqa: E402
    add_batting_pressure_features,
)
from superr.features.bowling_pressure_features import (  # noqa: E402
    add_bowling_pressure_features,
)
from superr.features.feature_registry import build_registry  # noqa: E402
from superr.features.match_pressure_features import (  # noqa: E402
    add_match_pressure_features,
)
from superr.features.matchup_features import add_matchup_features  # noqa: E402
from superr.features.role_features import add_role_features  # noqa: E402
from superr.features.scenario_features import (  # noqa: E402
    add_scenario_features,
    build_opportunity_features,
)

KEY = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
OUTCOME_COLS = ["total_runs", "batter_runs", "is_team_wicket", "is_bowler_wicket",
                "extras_runs", "is_four", "is_six"]

# Outcome-dependent columns are the only columns allowed to change when a
# delivery's own outcome is perturbed.
ALLOWED_OUTCOME_DEPENDENT = set(OUTCOME_COLS) | {
    "total_runs", "batter_runs", "extras_runs", "bowler_runs_conceded",
    "wicket_type", "is_team_wicket", "is_bowler_wicket", "is_batter_dismissal",
    "dismissed_is_striker", "is_run_out", "is_retired_out", "is_retired_hurt",
    "is_dot_ball", "is_boundary",
}


def build_features(balls: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = balls.copy()
    labels = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "competition_labels.csv"
    )
    df = df.merge(
        labels[["comp_id", "competition_class"]], on="comp_id",
        how="left", validate="many_to_one",
    )
    df = add_match_pressure_features(df)
    df = add_batting_pressure_features(df)
    df = add_bowling_pressure_features(df)
    df = add_role_features(df)
    df = add_matchup_features(df)
    df = add_scenario_features(df)
    return df, list(df.columns)


def leakage_audit(balls: pd.DataFrame, registry, sample_matches: int = 50,
                  perturbations_per_match: int = 3,
                  seed: int = 20260131) -> dict:
    """Dynamic leakage audit.

    For a sample of matches, perturb the outcome columns of SEVERAL
    deliveries (only those rows) and recompute every registered feature. A
    pre-delivery feature MUST NOT change at a perturbed row: it depends only
    on prior deliveries. Only registry-documented FEATURE columns are
    checked — raw outcome columns (inputs, not features) are excluded by
    construction.
    """
    rng = np.random.default_rng(seed)
    matches = balls["match_id"].drop_duplicates()
    picked = rng.choice(matches.values, size=min(sample_matches, len(matches)),
                        replace=False)
    sub = balls[balls["match_id"].isin(picked)].reset_index(drop=True)

    base, _ = build_features(sub)
    check_cols = [c for c in registry.specs if c in base.columns]
    violations: list[dict] = []
    n_perturbed = 0

    for mid in picked:
        mrows = base.index[base["match_id"] == mid]
        # mid-innings deliveries (not the first/last few balls); perturb a
        # spread of positions so early-innings and late-innings features are
        # both exercised
        candidates = list(mrows[3:-3])
        if not candidates:
            continue
        k = min(perturbations_per_match, len(candidates))
        chosen = list(rng.choice(candidates, size=k, replace=False))
        for i in chosen:
            i = int(i)
            pert = sub.copy()
            # change the outcome at row i only (keep all other rows identical)
            new_runs = (int(pert.at[i, "runs"]) + 1) % 7
            pert.at[i, "runs"] = new_runs
            pert.at[i, "is_four"] = bool(new_runs == 4)
            pert.at[i, "is_six"] = bool(new_runs == 6)
            pert.at[i, "is_wicket"] = False
            pert.at[i, "is_team_wicket"] = False
            pert.at[i, "is_bowler_wicket"] = False
            pert.at[i, "total_runs"] = new_runs
            pert.at[i, "batter_runs"] = new_runs
            pert.at[i, "is_dot_ball"] = bool(new_runs == 0)
            pert.at[i, "is_boundary"] = bool(new_runs in (4, 6))
            pert.at[i, "is_legal_ball"] = True
            pert.at[i, "dismissal_type"] = None
            pert.at[i, "dismissed_player_id"] = np.nan
            pert.at[i, "extras_type"] = None
            pert.at[i, "extras1_runs"] = 0
            pert.at[i, "extras_2_runs"] = 0
            pert.at[i, "bowler_runs_conceded"] = new_runs
            n_perturbed += 1

            alt, _ = build_features(pert)
            for col in check_cols:
                a, b = base.at[i, col], alt.at[i, col]
                if pd.isna(a) and pd.isna(b):
                    continue
                try:
                    changed = not (a == b)
                except Exception:
                    changed = str(a) != str(b)
                if changed:
                    violations.append({"match_id": int(mid), "row": i, "feature": col,
                                       "before": str(a), "after": str(b)})
    return {
        "n_matches_sampled": int(len(picked)),
        "perturbations_per_match": int(perturbations_per_match),
        "n_deliveries_perturbed": n_perturbed,
        "n_feature_columns_checked": len(check_cols),
        "n_violations": len(violations),
        "violations": violations[:50],
        "verdict": "NO LEAKAGE DETECTED" if not violations else "LEAKAGE DETECTED",
    }


def write_hypotheses_md(path: Path) -> None:
    text = """# Feature Importance Hypotheses (pre-registered before modelling)

Written before any model is trained (Phase 5). These are testable priors,
not results — each will be checked against the trained models and reported.

## Expected dominant predictors of pressure (per the methodology)

| Rank | Feature family | Hypothesis | Test in Phase 5 |
|---|---|---|---|
| 1 | `match_balance_proxy` / WP balance | Balance dominates pressure perception; deliveries near 50/50 carry the most variance | Permutation importance in WP/leverage models |
| 2 | `required_run_rate`, `rrr_crr_gap` | Chase steepness is the single strongest situational stressor in innings 2 | GBM gain shares; partial dependence |
| 3 | `legal_balls_remaining`, `innings_phase` | Death-over proximity amplifies every other stressor | Interaction analysis |
| 4 | `wickets_in_hand`, `collapse_severity` | Resource scarcity and recent collapse raise wicket probability | Calibration by stratum |
| 5 | `bowler_recent_economy`, `bowler_recent_wicket_rate` | Bowler form is modestly predictive but heavily shrunk | Shrinkage sensitivity |
| 6 | `batter_is_new` vs `batter_is_set` | New batters dismiss ~1.5-2x the rate of set batters in identical states | Rate ratio with CI |
| 7 | `score_vs_par` | Batting-first pressure tracks score vs par more than raw run rate | Model comparison |
| 8 | `matchup_difficulty_index` | Hand x class matchups move expected runs by ~5-10% | Effect size vs bootstrap CI |
| 9 | `is_knockout` | Knockouts raise pressure independent of state (crowd/stakes not in state features) | Fixed-effect contrast |
| 10 | `bowling_team_wicket_drought` | Drought length mildly raises bowling pressure but is largely captured by setness | Partial correlation |

## Hypotheses expected NOT to matter much (pre-registered)

- `bowl_speed` — excluded (38.7% unrecorded).
- `delivery_line`, `over_or_around_wicket` — marginal once phase and class are known.
- Home/away — expected small; T20I neutral venues blur the signal.

## Validation gates these feed

- Pressure index (Phase 5) uses balance/severity components validated in the
  methodology document; scenario labels feed scenario-diversity metrics in
  the ratings and dossiers.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    cfg = load_config()
    processed = cfg.paths.processed_dir
    reports = cfg.paths.reports_dir
    balls = pd.read_parquet(processed / "balls_clean.parquet")
    print(f"[1/6] Loaded balls_clean: {len(balls):,} x {balls.shape[1]}")

    df, cols = build_features(balls)
    print(f"[2/6] Features built: {len(cols)} columns total")

    reg = build_registry()
    # key/outcome columns are excluded from documentation requirements
    ignore = set(KEY) | set(balls.columns) | ALLOWED_OUTCOME_DEPENDENT | {
        "match_id", "innings", "comp_id", "season"}
    undocumented = reg.validate(cols, ignore=ignore)
    if undocumented:
        print(f"      WARNING: {len(undocumented)} undocumented feature columns "
              f"(first 15): {undocumented[:15]}")

    # split into three feature tables
    key_df = df[KEY + ["striker_id", "non_striker_id", "bowler_id"]]
    batting_cols = [c for c in df.columns if c.startswith("batter_")] + [
        "runs_required_before", "required_run_rate", "current_run_rate",
        "rrr_crr_gap", "required_boundaries_estimate", "wickets_in_hand",
        "legal_balls_remaining", "target_progress", "score_vs_par",
        "expected_final_score", "score_deficit_vs_par", "expected_runs_remaining",
        "par_run_rate", "resources_remaining", "balls_per_wicket_remaining",
        "wickets_lost_vs_expected", "last6_wickets", "last12_wickets",
        "last18_wickets", "last24_wickets", "last6_run_rate", "last12_run_rate",
        "last12_dot_balls", "last12_boundaries", "consecutive_dots_before",
        "balls_since_last_boundary", "boundary_streak_before",
        "consecutive_wickets", "collapse_severity", "entering_after_wicket",
        "entry_score", "entry_required_rate", "entry_balls_remaining",
        "partnership_runs_before", "partnership_balls_before",
    ]
    bowling_cols = [c for c in df.columns if c.startswith("bowler_")] + [
        "bowling_team_wicket_drought", "wicket_drought_severity",
        "wicket_ends_drought", "batting_resources_remaining",
        "batting_rate_vs_par", "expected_remaining_runs", "score_vs_par",
        "expected_final_score", "required_run_rate", "rrr_crr_gap",
        "runs_required_before", "target_progress", "wickets_remaining",
        "wickets_in_hand", "overs_remaining", "striker_setness", "striker_runs",
        "batting_type", "pace_or_spin", "matchup_lboro_runs_per_ball",
        "matchup_lboro_wicket_rate", "matchup_difficulty_index",
    ]
    pressure_cols = ([c for c in df.columns if c.startswith("scenario_")]
                     + ["match_balance_proxy", "severity_proxy", "innings_phase",
                        "is_knockout", "over_number_before", "ball_in_over_before"])

    batting = df[KEY + ["striker_id"] + batting_cols].copy()
    bowling = df[KEY + ["bowler_id"] + bowling_cols].copy()
    pressure = df[KEY + pressure_cols].copy()

    for table, name in [(batting, "features_batting"),
                        (bowling, "features_bowling"),
                        (pressure, "features_pressure")]:
        # de-duplicate column lists while preserving order
        table = table.loc[:, ~pd.Index(table.columns).duplicated()]
        table.to_parquet(processed / f"{name}.parquet", index=False)
        print(f"[3/6] {name}: {table.shape}")

    reg.write_dictionary_md(reports / "feature_dictionary.md")
    write_hypotheses_md(reports / "feature_importance_hypotheses.md")
    print(f"[4/6] Registry: {len(reg.specs)} features documented -> feature_dictionary.md")

    opp = build_opportunity_features(df)
    opp.to_parquet(processed / "opportunity_features.parquet", index=False)
    print(f"      opportunity_features: {opp.shape} "
          f"(provisional HP flag; recomputed in Phase 5)")

    print("[5/6] Leakage audit (perturbing current-delivery outcomes) ...")
    audit = leakage_audit(balls, reg)
    audit["undocumented_columns"] = undocumented
    (reports / "leakage_audit.json").write_text(
        json.dumps({**audit,
                    "generated_utc": datetime.now(timezone.utc).isoformat()},
                   indent=2, default=str), encoding="utf-8")
    print(f"      {audit['n_deliveries_perturbed']} deliveries perturbed, "
          f"{audit['n_feature_columns_checked']} features checked, "
          f"{audit['n_violations']} violations -> {audit['verdict']}")

    print("[6/6] DONE")
    return 0 if audit["verdict"] == "NO LEAKAGE DETECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
