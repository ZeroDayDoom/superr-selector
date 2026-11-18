"""Phase-3 pipeline: master delivery table -> match states -> validation.

Single command to regenerate all processed data:
    python scripts/run_prepare.py

Outputs:
    data/processed/balls_clean.parquet
    data/processed/innings_summary.parquet
    data/processed/match_summary.parquet
    data/processed/competition_labels.csv
    reports/data_cleaning_report.md
    reports/match_state_validation.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from superr.config import load_config  # noqa: E402
from superr.data.load_data import build_master_delivery_table, load_tables  # noqa: E402
from superr.data.competition_filter import (  # noqa: E402
    build_scope_filter,
    classify_competitions,
    competition_scope_summary,
)
from superr.data.clean_data import (  # noqa: E402
    add_legal_ball,
    add_run_reconstruction,
    add_special_handling,
    add_wicket_reconstruction,
    apply_scope,
    clean_extras,
)
from superr.data.match_state import (  # noqa: E402
    add_chase_state,
    add_momentum_features,
    add_par_state,
    add_pre_delivery_state,
    build_innings_summary,
)
from superr.data.player_state import (  # noqa: E402
    add_batter_roles,
    add_bowler_roles,
    add_bowler_state,
    add_career_dismissal_risk,
    add_striker_state,
)
from superr.data.validate_data import run_validation, write_validation_report  # noqa: E402


def main() -> int:
    cfg = load_config()
    feats = cfg.raw.get("features", {})
    windows = list(feats.get("momentum_windows", [6, 12, 18, 24]))
    processed = cfg.paths.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    reports = cfg.paths.reports_dir

    print("[1/9] Loading raw tables ...")
    tables = load_tables(cfg.paths.raw_data_dir)
    balls_raw, matches, competitions = (
        tables["ball_by_ball"], tables["matches"], tables["competitions"]
    )
    print(f"      deliveries={len(balls_raw):,} matches={len(matches):,} "
          f"competitions={len(competitions):,}")

    print("[2/9] Building master delivery table (verified joins) ...")
    master, join_report = build_master_delivery_table(balls_raw, matches, competitions)
    print(join_report)

    print("[3/9] Competition labels + scope filter ...")
    comp_labels = classify_competitions(competitions)
    comp_labels.to_csv(processed / "competition_labels.csv", index=False)
    match_scope = build_scope_filter(
        comp_labels, matches, scope="approved",
        include_a_team=bool(cfg.cleaning["exclude"]["a_team_competitions"]) is False,
    )
    scope_summary = competition_scope_summary(match_scope)
    print("      ", scope_summary)

    print("[4/9] Reconstruction: extras, runs, wickets, special handling ...")
    df = clean_extras(master)
    df = add_run_reconstruction(df)
    df = add_legal_ball(df)
    df = add_wicket_reconstruction(df)
    df = add_special_handling(df, match_scope)
    n_before = len(df)
    df = apply_scope(df, keep_super_over_innings=False)
    print(f"      {n_before:,} -> {len(df):,} deliveries in analysis scope "
          f"(dropped {n_before - len(df):,}: out-of-scope matches + super overs)")

    print("[5/9] Innings summary + pre-delivery state ...")
    innings = build_innings_summary(df)
    df = add_pre_delivery_state(df)
    df = add_chase_state(df, innings)

    print("[6/9] Par / expected-scoring state (leave-one-innings-out) ...")
    df = add_par_state(df, venue_par_k=float(feats.get("venue_par_k", 8.0)))

    print("[7/9] Momentum + player state ...")
    df = add_momentum_features(df, windows)
    df = add_striker_state(df, feats)
    df = add_bowler_state(df, windows)
    df = add_batter_roles(df, feats)
    df = add_bowler_roles(df)
    df = add_career_dismissal_risk(df)

    print("[8/9] Validation battery ...")
    results = run_validation(df, innings, match_scope)
    extra = {
        "rows_in_scope": int(len(df)),
        "join_report": [list(s) for s in join_report.steps],
        "scope_summary": scope_summary,
        "dropped_out_of_scope": n_before - len(df),
    }
    ok = write_validation_report(results, reports / "match_state_validation.json", extra)
    n_fail = sum(1 for r in results if r["status"] == "fail")
    for r in results:
        print(f"      [{r['status']}] {r['check']}: {r['detail']}")
    if not ok:
        print(f"VALIDATION FAILED with {n_fail} FAIL checks — not writing outputs.")
        return 1

    print("[9/9] Saving processed artefacts ...")
    df.to_parquet(processed / "balls_clean.parquet", index=False)
    innings.to_parquet(processed / "innings_summary.parquet", index=False)

    match_summary = match_scope.merge(
        innings.groupby("match_id").agg(
            n_innings=("innings", "size"),
            max_legal_balls=("legal_balls", "max"),
        ).reset_index(),
        on="match_id", how="left", validate="one_to_one",
    )
    match_summary.to_parquet(processed / "match_summary.parquet", index=False)

    write_cleaning_report(reports / "data_cleaning_report.md", df, innings,
                          match_scope, join_report, scope_summary, results)
    print("DONE — validation passed, artefacts written to", processed)
    return 0


def write_cleaning_report(path, df, innings, match_scope, join_report,
                          scope_summary, results) -> None:
    n_fail = sum(1 for r in results if r["status"] == "fail")
    lines = [
        "# Data Cleaning & Match-State Report",
        "",
        "## Joins (row-count verified)",
        "```", str(join_report), "```",
        "",
        "## Scope",
        f"- matches in scope: **{scope_summary['matches_in_scope']}** of "
        f"{scope_summary['matches_total']}",
        f"- by competition class: {scope_summary['by_class']}",
        f"- deliveries in analysis table: **{len(df):,}**",
        f"- innings summarised: {len(innings):,} "
        f"(incl. {int(innings['is_super_over'].sum())} super-over innings, "
        "excluded from the delivery table)",
        "",
        "## Reconstructed columns",
        "- runs: `batter_runs`, `extras_runs`, `total_runs`, "
        "`bowler_runs_conceded` (byes/leg-byes excluded from bowler)",
        "- wickets: `is_team_wicket`, `is_bowler_wicket`, `wicket_type`, "
        "`is_run_out`, `is_retired_out`, `dismissed_is_striker`",
        "- state: `score_before`, `wickets_before`, `legal_balls_before`, "
        "`legal_balls_remaining`, chase state (target/RRR), par state "
        "(leave-one-innings-out curves), momentum windows "
        "(6/12/18/24), striker & bowler state, roles",
        "",
        "## Validation results",
        f"- checks: {len(results)}, failures: **{n_fail}**",
    ]
    for r in results:
        lines.append(f"- [{r['status']}] `{r['check']}` — {r['detail']}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
