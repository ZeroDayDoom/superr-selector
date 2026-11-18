"""Phase-8 pipeline: full validation & robustness battery.

Single command:
    python scripts/run_validation.py

Requires Phase-7 outputs (outputs/all_player_metrics.csv).

Outputs:
    reports/validation_report.md, robustness_report.md, validation_summary.json
    outputs/scenario_rankings.csv, bootstrap_rankings.csv, rank_stability.csv,
    top10_probability.csv
    outputs/figures/* (8 charts)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superr.config import load_config  # noqa: E402
from superr.ratings.batting_impact import build_lambda_table  # noqa: E402
from superr.ratings.bowling_impact import add_bowling_value  # noqa: E402
from superr.ratings.batting_impact import add_batting_value  # noqa: E402
from superr.ratings.composite_rating import (  # noqa: E402
    PRIMARY_WEIGHTS,
    VARIANTS as WEIGHT_VARIANTS,
    composite as _composite,
)
from superr.validation import (  # noqa: E402
    adversarial_checks as adv,
    competition_validation as comp,
    role_validation as rv,
    scenario_analysis as sa,
    sensitivity as sens,
    split_half as sh,
    temporal_validation as tv,
)
from superr.validation.bootstrap import bootstrap_ranks  # noqa: E402
from superr.validation.ranking_stability import (  # noqa: E402
    make_charts,
    stability_summary,
)

N_BOOT = load_config().ratings["n_bootstrap"]  # 1000 — single source of truth


def load_value_frame(processed: Path) -> pd.DataFrame:
    balls = pd.read_parquet(processed / "balls_clean.parquet")
    expected = pd.read_parquet(processed / "expected_performance.parquet")
    pressure = pd.read_parquet(processed / "pressure_scores.parquet")
    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
    df = balls.merge(expected.drop(
        columns=["striker_id", "non_striker_id", "bowler_id",
                 "competition_class", "innings_phase"], errors="ignore"),
        on=key, how="left", validate="one_to_one")
    press_cols = key + [
        "PRESSURE_MODEL_A", "PRESSURE_MODEL_B", "PRESSURE_MODEL_C",
        "hp_top10", "hp_top15", "hp_top20", "hp_top25", "hp_top30",
        "leverage_expected_dir_swing", "realized_wpa"]
    df = df.merge(pressure[press_cols], on=key, how="left",
                  validate="one_to_one")
    # scenario labels (Phase-4 artefact) for the scenario-robustness checks
    scen = pd.read_parquet(processed / "features_pressure.parquet")
    scen_cols = key + [c for c in scen.columns if c.startswith("scenario_")]
    df = df.merge(scen[scen_cols], on=key, how="left", validate="one_to_one")
    # competition class (IPL / senior T20I) from the Phase-1 label table
    labels = pd.read_csv(processed / "competition_labels.csv")
    df = df.merge(labels[["comp_id", "competition_class"]], on="comp_id",
                  how="left", validate="many_to_one")
    df["hp_top20"] = df["hp_top20"].astype("boolean").fillna(False)
    lam_series, lam_mean = build_lambda_table(balls)
    df["lambda_state"] = (
        lam_series.reindex(pd.MultiIndex.from_arrays([
            df["innings"], (df["legal_balls_remaining"] // 6).clip(lower=0),
            df["wickets_before"]]))
        .fillna(lam_mean).values
    )
    df = add_batting_value(df, lam_series, lam_mean)
    df = add_bowling_value(df)
    df["season_year"] = pd.to_numeric(
        df["season"].astype("string").str.extract(r"(\d{4})")[0], errors="coerce")
    df["opponent_team_id"] = np.where(df["innings"].eq(2),
                                      df["bowling_team_id"], df["batting_team_id"])
    return df, lam_series, lam_mean


def main() -> int:
    cfg = load_config()
    processed = cfg.paths.processed_dir
    outputs = cfg.paths.outputs_dir
    reports = cfg.paths.reports_dir
    figures = cfg.paths.figures_dir

    metrics = pd.read_csv(outputs / "all_player_metrics.csv")
    bat_all = metrics[metrics["role"] == "batter"].copy()
    bowl_all = metrics[metrics["role"] == "bowler"].copy()
    audit = pd.read_csv(outputs / "eligibility_audit.csv")

    print("[1/9] Building value frame ...")
    df, lam_table, lam_mean = load_value_frame(processed)
    summary: dict = {"generated_utc": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()}

    eligible_bat = set(audit[(audit["role"] == "batter")
                             & audit["eligible_moderate"]]["player_id"])
    eligible_bowl = set(audit[(audit["role"] == "bowler")
                              & audit["eligible_moderate"]]["player_id"])

    hp_bat = df[(df["hp_top20"].astype(bool)) & df["striker_id"].isin(eligible_bat)] \
        .rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
    hp_bowl = df[(df["hp_top20"].astype(bool)) & df["bowler_id"].isin(eligible_bowl)] \
        .rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})

    # ---------------------------------------------------------- V1 temporal
    print("[2/9] V1 temporal persistence ...")
    summary["temporal_batters"] = tv.temporal_folds(hp_bat, "bat_value")
    summary["temporal_bowlers"] = tv.temporal_folds(hp_bowl, "bowl_value")
    print("      batters:", summary["temporal_batters"])

    # ------------------------------------------------------- V2 bootstrap
    print(f"[3/9] V2 match-level bootstrap ({N_BOOT} draws) ...")
    boot_bat = bootstrap_ranks(hp_bat, "bat_value", eligible_bat, n_boot=N_BOOT)
    boot_bowl = bootstrap_ranks(hp_bowl, "bowl_value", eligible_bowl, n_boot=N_BOOT)
    names_bat = bat_all.set_index("player_id")["player_name"]
    names_bowl = bowl_all.set_index("player_id")["player_name"]
    boot_bat["player_name"] = boot_bat["player_id"].map(names_bat)
    boot_bowl["player_name"] = boot_bowl["player_id"].map(names_bowl)
    boot_all = pd.concat([boot_bat.assign(role="batter"),
                          boot_bowl.assign(role="bowler")], ignore_index=True)
    boot_all.to_csv(outputs / "bootstrap_rankings.csv", index=False)
    top10p = boot_all[boot_all["p_top10"] > 0].sort_values("p_top10",
                                                           ascending=False)
    top10p.to_csv(outputs / "top10_probability.csv", index=False)
    print(f"      players bootstrapped: {len(boot_all)}")

    # ------------------------------------------------------ V3 split-half
    print("[4/9] V3 split-half reliability ...")
    summary["split_half_batters"] = sh.split_half_ranking_reliability(
        hp_bat, "bat_value", eligible_bat)
    summary["split_half_bowlers"] = sh.split_half_ranking_reliability(
        hp_bowl, "bowl_value", eligible_bowl)

    # --------------------------------------------------- V4 season trends
    print("[5/9] V4 season stability ...")
    trend_bat = tv.season_trend_table(hp_bat, "bat_value")
    trend_bowl = tv.season_trend_table(hp_bowl, "bowl_value")
    trends = {}
    for pid in list(eligible_bat)[:200]:
        trends[f"batter_{pid}"] = tv.classify_trend(trend_bat, pid)
    trend_counts = pd.Series([t["trend"] for t in trends.values()]).value_counts()
    summary["season_trend_distribution_top200_batters"] = trend_counts.to_dict()

    # ------------------------------------------------ V5 threshold + model
    print("[6/9] V5/V6 pressure-threshold + competition sensitivity ...")
    # threshold variants: rebuild agg tables at different HP quantiles is
    # expensive; approximate via the stored hp labels (they are stratified
    # top-10..30 flags): level = HP mean under each label
    threshold_tables = {}
    primary_bat = bat_all
    tables = {}
    for lbl in ("hp_top10", "hp_top15", "hp_top20", "hp_top25", "hp_top30"):
        sub = df[df[lbl].astype("boolean").fillna(False)
                 & df["striker_id"].isin(eligible_bat)]
        g = sub.groupby("striker_id").agg(
            player_name=("striker_name", "first"),
            hp_value_mean=("bat_value", "mean"),
            hp_balls=("bat_value", "size"))
        g = g[g["hp_balls"] >= 30]
        g = g.sort_values("hp_value_mean", ascending=False)
        g.insert(0, "rank", np.arange(1, len(g) + 1))
        g = g.reset_index().rename(columns={"striker_id": "player_id"})
        tables[f"threshold_{lbl}"] = g
    # primary ranks for stability comparisons: rating rank
    primary_rank = bat_all[["player_id", "player_name", "rating_primary"]].copy()
    primary_rank["rank"] = primary_rank["rating_primary"].rank(ascending=False)
    stability = stability_summary(tables, primary_rank)
    stability.to_csv(outputs / "rank_stability.csv", index=False)
    threshold_ranks = {k: v for k, v in tables.items()}
    summary["threshold_stability"] = stability.to_dict("records")

    # pressure-model sensitivity (A/B/C stored in pressure_scores)
    model_tables = {}
    for mdl in ("PRESSURE_MODEL_A", "PRESSURE_MODEL_B", "PRESSURE_MODEL_C"):
        sub = df[df[mdl] >= df[mdl].quantile(0.80)]
        sub = sub[sub["striker_id"].isin(eligible_bat)]
        g = sub.groupby("striker_id").agg(
            hp_value_mean=("bat_value", "mean"), hp_balls=("bat_value", "size"))
        g = g[g["hp_balls"] >= 30].sort_values("hp_value_mean", ascending=False)
        g.insert(0, "rank", np.arange(1, len(g) + 1))
        g = g.reset_index().rename(columns={"striker_id": "player_id"})
        model_tables[mdl] = g
    summary["pressure_model_stability"] = stability_summary(
        model_tables, primary_rank).to_dict("records")

    # ------------------------------------------------------- V6 competition
    names_bat_all = bat_all.set_index("player_id")["player_name"]
    comp_ranks_bat = comp.competition_scope_rankings(hp_bat, "bat_value",
                                                     names_bat_all)
    comp_ranks_bowl = comp.competition_scope_rankings(hp_bowl, "bowl_value",
                                                      names_bowl)
    summary["competition_overlap_batters"] = comp.scope_overlap(
        comp_ranks_bat["ipl_only"], comp_ranks_bat["t20i_only"],
        comp_ranks_bat["combined"])
    summary["competition_balance"] = rv.competition_balance(bat_all)
    summary["associate_team_check"] = comp.associate_team_check(df)

    # ------------------------------------------------------------ V7 roles
    summary["role_advantage_batters"] = rv.role_advantage(bat_all, "batter_role")
    summary["role_advantage_bowlers"] = rv.role_advantage(bowl_all, "bowler_role")
    summary["role_normalization_batters"] = rv.role_normalized_vs_raw(
        bat_all, "batter_role")

    # -------------------------------------------------- V8/V9/V10 tiers
    tier_tables = {}
    for tier in ("strict", "moderate", "inclusive"):
        ids = set(audit[(audit["role"] == "batter")
                        & audit[f"eligible_{tier}"]]["player_id"])
        sub = bat_all[bat_all["player_id"].isin(ids)].copy()
        sub["rank"] = sub["rating_primary"].rank(ascending=False)
        tier_tables[f"eligibility_{tier}"] = sub[["player_id", "player_name",
                                                  "rating_primary", "rank"]]
    summary["eligibility_stability_batters"] = stability_summary(
        tier_tables, primary_rank).to_dict("records")

    shrunk_tables = {}
    for col in ("shrunk_k_half", "shrunk_k", "shrunk_k_double", "raw_mean"):
        if col in bat_all.columns:
            d = bat_all.copy()
            # recompute the composite with this level variant (level LCB uses
            # the same 0.85*SE penalty as the primary rating)
            lvl = d[col] - 0.85 * d["boot_se"]
            d2 = d.assign(level_lcb=lvl)
            d2[f"rating_{col}"] = _composite(d2, PRIMARY_WEIGHTS, f"rating_{col}")
            r = d2.sort_values(f"rating_{col}", ascending=False).reset_index(drop=True)
            r["rank"] = np.arange(1, len(r) + 1)
            shrunk_tables[f"shrinkage_{col}"] = r[["player_id", "player_name",
                                                   f"rating_{col}", "rank"]]
    summary["shrinkage_stability_batters"] = stability_summary(
        shrunk_tables, primary_rank).to_dict("records")

    weight_tables = {}
    for name, w in WEIGHT_VARIANTS.items():
        d = bat_all.copy()
        d[f"rating_{name}"] = _composite(d, w, f"rating_{name}")
        r = d.sort_values(f"rating_{name}", ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        weight_tables[f"weights_{name}"] = r[["player_id", "player_name",
                                              f"rating_{name}", "rank"]]
    summary["weight_stability_batters"] = stability_summary(
        weight_tables, primary_rank).to_dict("records")

    # ------------------------------------------------------ V12 ablation
    ablations = {}
    for comp_name in ("level_lcb", "consistency", "uplift", "volume", "downside"):
        w = dict(PRIMARY_WEIGHTS)
        w[comp_name] = 0.0
        d = bat_all.copy()
        d[f"rating_no_{comp_name}"] = _composite(d, w, f"rating_no_{comp_name}")
        r = d.sort_values(f"rating_no_{comp_name}", ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        ablations[f"ablate_{comp_name}"] = r[["player_id", "player_name",
                                              f"rating_no_{comp_name}", "rank"]]
    abl_stab = stability_summary(ablations, primary_rank)
    summary["ablation_batters"] = abl_stab.to_dict("records")

    # --------------------------------------------------- V11 scenario
    print("[7/9] V11 scenario robustness ...")
    scen_bat = sa.scenario_specificity(
        df.rename(columns={"striker_id": "player_id",
                           "striker_name": "player_name"}),
        [c for c in df.columns if c.startswith("scenario_")
         and c not in ("scenario_count", "scenario_diversity_index")],
        "bat_value", min_balls=30)
    scen_bat = scen_bat.merge(
        bat_all[["player_id", "player_name", "rating_primary"]], on="player_id",
        how="left")
    scen_bat.sort_values("scenario_spread", ascending=False).to_csv(
        outputs / "scenario_rankings.csv", index=False)
    summary["scenario_specificity_batters"] = {
        "n_players_with_3plus_scenarios": int(
            scen_bat["scenario_spread"].notna().sum()),
        "median_spread": round(float(scen_bat["scenario_spread"].median()), 3)
        if len(scen_bat) else None,
    }

    # ---------------------------------------------------- V13 model checks
    print("[8/9] V13/V14 model checks + edge tests ...")
    summary["model_checks"] = adv.model_checks(df)
    summary["edge_tests"] = adv.edge_tests(df, lam_table, lam_mean)

    # ------------------------------------------------ most/least robust
    robust_b, uncertain_b = sens.most_and_least_robust(boot_bat)
    robust_l, uncertain_l = sens.most_and_least_robust(boot_bowl)
    summary["most_robust_batters"] = robust_b.to_dict("records")
    summary["most_uncertain_batters"] = uncertain_b.to_dict("records")
    summary["most_robust_bowlers"] = robust_l.to_dict("records")
    summary["most_uncertain_bowlers"] = uncertain_l.to_dict("records")

    # ----------------------------------------------------------- charts
    print("[9/9] Charts + reports ...")
    make_charts(boot_all, names_bat, threshold_ranks, trend_bat,
                comp_ranks_bat, summary["role_advantage_batters"],
                abl_stab, figures)

    (reports / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_reports(reports, summary, boot_all)

    print("DONE — validation complete")
    return 0


def write_reports(reports: Path, summary: dict, boot_all: pd.DataFrame) -> None:
    # validation_report.md
    lines = ["# Validation Report", "",
             f"_Generated {summary['generated_utc']}_", "",
             "## V1 Temporal persistence (era-to-era HP value)", ""]
    for role in ("batters", "bowlers"):
        lines.append(f"### {role}")
        for fold, m in summary.get(f"temporal_{role}", {}).items():
            lines.append(f"- `{fold}`: Spearman {m['spearman']}, "
                         f"top-15 overlap {m['top15_overlap']}/15 "
                         f"(n={m['n_players_both_eras']})")
        lines.append("")
    lines += ["## V3 Split-half reliability", "",
              f"- Batters: {summary.get('split_half_batters')}",
              f"- Bowlers: {summary.get('split_half_bowlers')}", "",
              "## V4 Season trends (top-200 batters by sample)",
              f"- {summary.get('season_trend_distribution_top200_batters')}", "",
              "## V5/V6/V8/V9/V10 rank stability vs primary (batters)", "",
              "| variant | Spearman | top-10 overlap |", "|---|---|---|"]
    for r in summary.get("threshold_stability", []) + \
            summary.get("pressure_model_stability", []) + \
            summary.get("eligibility_stability_batters", []) + \
            summary.get("shrinkage_stability_batters", []) + \
            summary.get("weight_stability_batters", []):
        lines.append(f"| {r['variant']} | {r['spearman']} | {r['top10_overlap']} |")
    lines += ["", "## V12 Ablation (batters)", "",
              "| removed component | Spearman vs primary |", "|---|---|"]
    for r in summary.get("ablation_batters", []):
        lines.append(f"| {r['variant']} | {r['spearman']} |")
    lines += ["", "## V7 Role checks", "",
              f"- Batters: {summary.get('role_advantage_batters')}",
              f"- Role normalisation vs raw: "
              f"{summary.get('role_normalization_batters', {}).get('spearman_raw_vs_rolenorm')}",
              "", "## V13 Model checks", ""]
    for k, v in summary.get("model_checks", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## V14 Edge tests", ""]
    for k, v in summary.get("edge_tests", {}).items():
        lines.append(f"- `{k}`: {v}")
    (reports / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    # robustness_report.md
    lines = ["# Robustness Report", "",
             "## Most robust players (tightest bootstrap rank intervals)", ""]
    for role in ("batters", "bowlers"):
        lines.append(f"### {role}")
        for r in summary.get(f"most_robust_{role}", []):
            nm = boot_all[boot_all["player_id"] == r["player_id"]]["player_name"]
            nm = nm.iloc[0] if len(nm) else r["player_id"]
            lines.append(f"- {nm}: rank P05-P95 = "
                         f"[{r['rank_p05']:.0f}, {r['rank_p95']:.0f}], "
                         f"P(top10)={r['p_top10']:.2f}")
        lines.append("")
    lines += ["## Most uncertain players (widest intervals)", ""]
    for role in ("batters", "bowlers"):
        lines.append(f"### {role}")
        for r in summary.get(f"most_uncertain_{role}", []):
            nm = boot_all[boot_all["player_id"] == r["player_id"]]["player_name"]
            nm = nm.iloc[0] if len(nm) else r["player_id"]
            lines.append(f"- {nm}: rank P05-P95 = "
                         f"[{r['rank_p05']:.0f}, {r['rank_p95']:.0f}], "
                         f"P(top10)={r['p_top10']:.2f}")
        lines.append("")
    lines += ["## Interpretation", "",
              "- A ranking that survives threshold changes, competition scope,",
              "  eligibility tiers, shrinkage strengths, weight variants and",
              "  component ablations with high Spearman stability is robust;",
              "  the numbers above quantify exactly that.",
              "- The final recommendation for the submission is the set of",
              "  players with BOTH a high primary rating AND a high",
              f"  P(top-10) across {N_BOOT} bootstrap draws — see",
              "  `outputs/top10_probability.csv`."]
    (reports / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
