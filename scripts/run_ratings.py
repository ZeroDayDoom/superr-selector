"""Phase-7 pipeline: player aggregation, reliability, rankings, dossiers.

Single command:
    python scripts/run_ratings.py

Requires Phase-6 outputs (expected_performance.parquet).

Outputs:
    outputs/rankings_batters.csv, rankings_bowlers.csv
    outputs/all_player_metrics.csv, eligibility_audit.csv
    outputs/player_dossiers/*.md
    outputs/role_rankings/*.csv, scenario_rankings/*.csv
    reports/player_rating_methodology.md
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
from superr.ratings import batting_impact as bi  # noqa: E402
from superr.ratings import bowling_impact as bw  # noqa: E402
from superr.ratings import bayesian_shrinkage as bs  # noqa: E402
from superr.ratings import composite_rating as cr  # noqa: E402
from superr.ratings import consistency as cons  # noqa: E402
from superr.ratings import downside_risk as dr  # noqa: E402
from superr.ratings import eligibility as el  # noqa: E402
from superr.ratings import player_baselines as pb  # noqa: E402
from superr.ratings import public_context as pubcon  # noqa: E402
from superr.ratings import reliability as rel  # noqa: E402
from superr.ratings import rankings as rk  # noqa: E402

N_BOOT = load_config().ratings["n_bootstrap"]  # 1000 — single source of truth
SCENARIO_COLS = [
    "scenario_HIGH_RRR_CHASE", "scenario_DEATH_OVERS_PRESSURE",
    "scenario_EARLY_COLLAPSE_RESCUE", "scenario_MIDDLE_ORDER_RESCUE",
    "scenario_RAPID_COLLAPSE", "scenario_HIGH_LEVERAGE",
    "scenario_MATCH_ON_LINE", "scenario_FINAL_OVER_PRESSURE",
    "scenario_PARTNERSHIP_REBUILD", "scenario_BOUNDARY_DROUGHT",
    "scenario_NEW_BATTER_AFTER_WICKET", "scenario_LOW_WICKET_RESOURCE_CHASE",
    "scenario_DEATH_DEFENCE", "scenario_WICKET_DROUGHT_BREAKTHROUGH",
    "scenario_SET_BATTER_DISMISSAL", "scenario_NEW_BATTER_PRESSURE",
]


def build_value_frame(processed: Path) -> pd.DataFrame:
    balls = pd.read_parquet(processed / "balls_clean.parquet")
    expected = pd.read_parquet(processed / "expected_performance.parquet")
    pressure = pd.read_parquet(processed / "pressure_scores.parquet")
    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
    df = balls.merge(expected.drop(
        columns=["striker_id", "non_striker_id", "bowler_id",
                 "competition_class", "innings_phase"], errors="ignore"),
        on=key, how="left", validate="one_to_one")
    df = df.merge(pressure[key + ["PRESSURE_MODEL_C", "hp_top20", "hp_top10",
                                  "leverage_expected_dir_swing",
                                  "realized_wpa", "leverage_expected_abs_swing"]],
                  on=key, how="left", validate="one_to_one")
    # scenario labels (Phase-4 artefact) for diversity / scenario rankings
    scen = pd.read_parquet(processed / "features_pressure.parquet")
    scen_cols = key + [c for c in scen.columns if c.startswith("scenario_")]
    df = df.merge(scen[scen_cols], on=key, how="left", validate="one_to_one")
    # competition class (IPL / senior T20I) from the Phase-1 label table
    labels = pd.read_csv(processed / "competition_labels.csv")
    df = df.merge(labels[["comp_id", "competition_class"]], on="comp_id",
                  how="left", validate="many_to_one")
    df["hp_extreme"] = df["hp_top10"].astype("boolean").fillna(False)
    df["hp_top20"] = df["hp_top20"].astype("boolean").fillna(False)
    # opponent context per role
    df["opponent_team_id"] = np.where(df["innings"].eq(2),
                                      df["bowling_team_id"], df["batting_team_id"])
    return df


def build_role_table(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Player-level aggregation for one role (batter or bowler)."""
    value_col = "bat_value" if role == "batter" else "bowl_value"
    d = df.rename(columns={"striker_id" if role == "batter" else "bowler_id":
                           "player_id",
                           "striker_name" if role == "batter" else "bowler_name":
                           "player_name"})
    # only rows where the player actually participated (batter faced / bowler bowled)
    d = d[d["player_id"].notna()]

    agg = pb.pressure_metrics(d, role)
    agg["player_name"] = agg["player_id"].map(
        d.drop_duplicates("player_id").set_index("player_id")["player_name"])
    role_col = "batter_role" if role == "batter" else "bowler_role"
    agg[role_col] = agg["player_id"].map(
        d.drop_duplicates("player_id").set_index("player_id")[role_col])
    agg["uplift_adjusted"] = pb.context_adjusted_uplift(agg.set_index("player_id"))
    agg["uplift_normalised"] = pb.normalise_uplift(agg.set_index("player_id"))

    # HP match means for consistency / downside / reliability
    hp = d[d["hp_top20"].astype(bool)]
    mv = hp.groupby(["player_id", "match_id"])[value_col].mean().rename(
        "hp_match_value").reset_index()
    cons_m = cons.consistency_metrics(mv, "hp_match_value")
    agg = agg.merge(cons_m, on="player_id", how="left")
    down = dr.downside_metrics(mv, "hp_match_value")
    agg = agg.merge(down, on="player_id", how="left")

    # split-half stability of HP value across the pool
    agg["split_half_stability_pool"] = cons.split_half_stability(
        mv, "hp_match_value", n_splits=60)

    # EB shrinkage of the HP mean (k from match-level means)
    eb = bs.estimate_k(mv, "hp_match_value")
    shrunk = bs.shrinkage_table(agg.set_index("player_id")["hp_value_mean"],
                                agg.set_index("player_id")["hp_balls"],
                                eb["k"], eb["pool_mean"])
    agg = agg.merge(shrunk.reset_index(), on="player_id", how="left")
    agg["eb_k"] = eb["k"]
    agg["eb_pool_mean"] = eb["pool_mean"]

    # bootstrap CI per player on HP value
    boot = rel.bootstrap_player_mean(hp.rename(columns={}), value_col,
                                     n_boot=N_BOOT)
    boot.index.name = "player_id"
    agg = agg.merge(boot.reset_index(), on="player_id", how="left")
    agg["level_lcb"] = [rel.lcb(m, s) for m, s in
                        zip(agg["shrunk_k"], agg["boot_se"])]

    # volume: log HP balls + scenario diversity
    scen_div = (hp.groupby("player_id")[SCENARIO_COLS].max().sum(axis=1)
                .rename("scenario_diversity"))
    agg = agg.merge(scen_div.reset_index(), on="player_id", how="left")
    role_value_col = "batter_role" if role == "batter" else "bowler_role"
    role_share = hp.groupby("player_id")[role_value_col].nunique()
    agg["role_diversity"] = agg["player_id"].map(role_share).fillna(0)
    agg["volume"] = np.log1p(agg["hp_balls"]) * agg["scenario_diversity"].clip(lower=1)
    agg["role"] = role

    # competition split and season trend (evidence for dossiers)
    comp_split = hp.groupby(["player_id", "competition_class"])[
        value_col].mean().unstack("competition_class")
    season_trend = hp.groupby(["player_id", "season"])[value_col].mean() \
        .unstack("season")
    agg = agg.merge(comp_split.add_prefix("hpval_").reset_index(),
                    on="player_id", how="left")

    return agg, mv, season_trend


def main() -> int:
    cfg = load_config()
    processed = cfg.paths.processed_dir
    outputs = cfg.paths.outputs_dir
    reports = cfg.paths.reports_dir
    dossiers = outputs / "player_dossiers"
    role_dir = outputs / "role_rankings"
    scen_dir = outputs / "scenario_rankings"
    for p in (dossiers, role_dir, scen_dir):
        p.mkdir(parents=True, exist_ok=True)

    print("[1/8] Building per-delivery value frame ...")
    df = build_value_frame(processed)
    lam_series, lam_mean = bi.build_lambda_table(df)
    df["lambda_state"] = (
        lam_series.reindex(pd.MultiIndex.from_arrays([
            df["innings"], (df["legal_balls_remaining"] // 6).clip(lower=0),
            df["wickets_before"]]))
        .fillna(lam_mean).values
    )
    df = bi.add_batting_value(df, lam_series, lam_mean)
    df = bw.add_bowling_value(df)
    print(f"      lambda mean={lam_mean:.3f} runs; value frame {df.shape}")

    print("[2/8] Batter aggregation ...")
    bat_agg, bat_mv, bat_season = build_role_table(df, "batter")
    print("[3/8] Bowler aggregation ...")
    bowl_agg, bowl_mv, bowl_season = build_role_table(df, "bowler")

    print("[4/8] Eligibility audit ...")
    dist = el.derive_thresholds_from_distribution(bat_agg)
    print("      ", dist)
    audit = pd.concat([
        el.eligibility_audit(bat_agg), el.eligibility_audit(bowl_agg)],
        ignore_index=True)
    audit.to_csv(outputs / "eligibility_audit.csv", index=False)

    print("[5/8] Composite ratings ...")
    bat_agg = cr.add_composites(bat_agg)
    bowl_agg = cr.add_composites(bowl_agg)

    # rank stability across sensitivity variants (eligible players only;
    # NaN ratings excluded so Spearman is defined)
    for name, agg in (("batters", bat_agg), ("bowlers", bowl_agg)):
        ok = agg["rating_primary"].notna() & agg["hp_balls"].ge(60)
        sub = agg[ok]
        base = sub["rating_primary"].rank()
        for v in cr.VARIANTS:
            alt = sub[f"rating_{v}"].rank()
            from scipy.stats import spearmanr
            rho = spearmanr(base, alt).statistic
            print(f"      {name} rank stability vs {v}: rho={rho:.3f}")
            agg.loc[:, f"rank_rho_{v}"] = rho

    # -------------------------------------------------------- rankings
    print("[6/8] Rankings ...")
    metric_cols = [c for c in bat_agg.columns if c not in ("player_name",)]
    bat_all = bat_agg.sort_values("rating_primary", ascending=False).reset_index(drop=True)
    bowl_all = bowl_agg.sort_values("rating_primary", ascending=False).reset_index(drop=True)
    bat_all.insert(0, "rank", np.arange(1, len(bat_all) + 1))
    bowl_all.insert(0, "rank", np.arange(1, len(bowl_all) + 1))

    assert rk.check_no_duplicates(bat_all), "duplicate batter in ranking"
    assert rk.check_no_duplicates(bowl_all), "duplicate bowler in ranking"

    # eligibility filter for the headline lists (moderate tier)
    bat_elig = audit[(audit["role"] == "batter") & (audit["eligible_moderate"])]
    bowl_elig = audit[(audit["role"] == "bowler") & (audit["eligible_moderate"])]
    top10_batters = bat_all[bat_all["player_id"].isin(bat_elig["player_id"])].head(10).copy()
    top10_bowlers = bowl_all[bowl_all["player_id"].isin(bowl_elig["player_id"])].head(10).copy()
    # renumber 1..10 for the submission (the global pool rank is kept in
    # rank_in_pool for context)
    for t in (top10_batters, top10_bowlers):
        t.rename(columns={"rank": "rank_in_pool"}, inplace=True)
        t.insert(0, "rank", np.arange(1, len(t) + 1))
    # submission ranks are 1..10; the global pool rank is retained separately
    for t in (top10_batters, top10_bowlers):
        t.rename(columns={"rank": "pool_rank"}, inplace=True)
        t.insert(0, "rank", np.arange(1, len(t) + 1))

    top10_batters.to_csv(outputs / "rankings_batters.csv", index=False)
    top10_bowlers.to_csv(outputs / "rankings_bowlers.csv", index=False)

    # public context (strictly separate from the ranking; the pressure
    # rating is never modified by public information — see module docstring)
    claims_path = cfg.paths.raw_data_dir / "public_claims.csv"
    claims = pubcon.load_claims(claims_path) if claims_path.exists() else None
    if claims is not None:
        problems = pubcon.validate_claims(claims)
        if problems:
            print(f"      WARNING: {len(problems)} public-claim sheet "
                  f"problems (missing sources/flags)")
    pub_table = pd.concat(
        [pubcon.build_auction_table(top10_batters, "batter", claims)
         .assign(role="batter"),
         pubcon.build_auction_table(top10_bowlers, "bowler", claims)
         .assign(role="bowler")],
        ignore_index=True)
    pub_table.to_csv(outputs / "player_public_context.csv", index=False)

    # all metrics + facet rankings
    all_metrics = pd.concat([bat_all.assign(role="batter"),
                             bowl_all.assign(role="bowler")], ignore_index=True)
    all_metrics.to_csv(outputs / "all_player_metrics.csv", index=False)

    # role-specific rankings
    role_col = {"batter": "batter_role", "bowler": "bowler_role"}
    for role_name, agg, role_c in (("batters", bat_all, role_col["batter"]),
                                   ("bowlers", bowl_all, role_col["bowler"])):
        for role_val, grp in agg.groupby(role_c):
            if len(grp) < 5:
                continue
            grp.sort_values("rating_primary", ascending=False).head(20).to_csv(
                role_dir / f"{role_name}_{role_val}.csv", index=False)

    # scenario rankings
    for role_name, agg in (("batters", bat_all), ("bowlers", bowl_all)):
        vcol = "bat_value" if role_name == "batters" else "bowl_value"
        scen = rk.scenario_rankings(df.rename(columns={
            "striker_id" if role_name == "batters" else "bowler_id": "player_id",
            "striker_name" if role_name == "batters" else "bowler_name": "player_name"}),
            SCENARIO_COLS, vcol, min_balls=40)
        for sc, t in scen.items():
            t.to_csv(scen_dir / f"{role_name}_{sc}.csv", index=False)

    # competition-scope rankings
    for scope, mask in (("ipl", df["competition_class"].eq("IPL")),
                        ("t20i", df["competition_class"].eq("senior T20I"))):
        sub = df[mask]
        for role_name, idc, nnc, vcol in (
                ("batters", "striker_id", "striker_name", "bat_value"),
                ("bowlers", "bowler_id", "bowler_name", "bowl_value")):
            g = (sub[sub["hp_top20"].astype(bool)]
                 .groupby(idc)[vcol].agg(["mean", "size"])
                 .rename(columns={"mean": "hp_value_mean", "size": "hp_balls"}))
            names = sub.groupby(idc)[nnc].first()
            g = g.join(names).query("hp_balls >= 40")
            g = g.sort_values("hp_value_mean", ascending=False)
            g.insert(0, "rank", np.arange(1, len(g) + 1))
            g.to_csv(role_dir / f"{role_name}_{scope}_only.csv")

    # facet rankings (reliability / consistency / uplift / downside)
    for role_name, agg in (("batters", bat_all), ("bowlers", bowl_all)):
        for facet, col, asc in (("reliability", "hp_balls", False),
                                ("consistency", "positive_match_pct", False),
                                ("uplift", "uplift_adjusted", False),
                                ("downside", "cvar5", True)):
            agg.sort_values(col, ascending=asc).head(20).to_csv(
                role_dir / f"{role_name}_{facet}_ranking.csv", index=False)

    # ------------------------------------------------------- dossiers
    print("[7/8] Dossiers ...")
    for rank, (_, row) in enumerate(top10_batters.iterrows(), start=1):
        mv = (bat_mv[bat_mv["player_id"] == row["player_id"]]
              [["match_id", "hp_match_value"]])
        scen_means = {
            sc.replace("scenario_", ""): float(
                df[(df["striker_id"] == row["player_id"])
                   & df[sc].fillna(False).astype(bool)]["bat_value"].mean())
            for sc in SCENARIO_COLS
            if ((df["striker_id"] == row["player_id"])
                & df[sc].fillna(False).astype(bool)).any()
        }.items()
        scen_sorted = dict(sorted(scen_means, key=lambda kv: -kv[1]))
        text = rk.dossier_markdown(
            row, rank, "batter", mv,
            bat_season.loc[row["player_id"]] if row["player_id"] in bat_season.index else {},
            (bat_agg.set_index("player_id").loc[row["player_id"], ["hpval_IPL", "hpval_senior T20I"]]
             if row["player_id"] in bat_agg["player_id"].values else {}),
            dict(list(scen_sorted.items())[:3]), dict(list(scen_sorted.items())[-3:]),
        )
        (dossiers / f"batter_rank{rank:02d}_{row['player_id']}.md").write_text(
            text, encoding="utf-8")

    for rank, (_, row) in enumerate(top10_bowlers.iterrows(), start=1):
        mv = (bowl_mv[bowl_mv["player_id"] == row["player_id"]]
              [["match_id", "hp_match_value"]])
        scen_means = {
            sc.replace("scenario_", ""): float(
                df[(df["bowler_id"] == row["player_id"])
                   & df[sc].fillna(False).astype(bool)]["bowl_value"].mean())
            for sc in SCENARIO_COLS
            if ((df["bowler_id"] == row["player_id"])
                & df[sc].fillna(False).astype(bool)).any()
        }.items()
        scen_sorted = dict(sorted(scen_means, key=lambda kv: -kv[1]))
        text = rk.dossier_markdown(
            row, rank, "bowler", mv,
            bowl_season.loc[row["player_id"]] if row["player_id"] in bowl_season.index else {},
            (bowl_agg.set_index("player_id").loc[row["player_id"], ["hpval_IPL", "hpval_senior T20I"]]
             if row["player_id"] in bowl_agg["player_id"].values else {}),
            dict(list(scen_sorted.items())[:3]), dict(list(scen_sorted.items())[-3:]),
        )
        (dossiers / f"bowler_rank{rank:02d}_{row['player_id']}.md").write_text(
            text, encoding="utf-8")

    # --------------------------------------------------------- report
    print("[8/8] Methodology report ...")
    write_methodology_report(reports, bat_agg, bowl_agg, audit, dist, lam_mean)

    # summary artefact consumed by build_final_report.py (single source of
    # truth for cross-phase numbers like the mean dynamic wicket value)
    (reports / "ratings_summary.json").write_text(json.dumps({
        "mean_lambda_runs": round(float(lam_mean), 4),
        "n_bootstrap": int(N_BOOT),
        "eb_k_batters": round(float(bat_agg["eb_k"].iloc[0]), 3),
        "eb_k_bowlers": round(float(bowl_agg["eb_k"].iloc[0]), 3),
        "n_batters": int(len(bat_agg)),
        "n_bowlers": int(len(bowl_agg)),
        "split_half_batters": round(
            float(bat_agg["split_half_stability_pool"].iloc[0]), 3),
        "split_half_bowlers": round(
            float(bowl_agg["split_half_stability_pool"].iloc[0]), 3),
    }, indent=2), encoding="utf-8")

    print("Top 10 batters (moderate eligibility):")
    print(top10_batters[["rank", "player_name", "rating_primary", "hp_balls",
                         "level_lcb", "uplift_adjusted"]].to_string(index=False))
    print()
    print("Top 10 bowlers (moderate eligibility):")
    print(top10_bowlers[["rank", "player_name", "rating_primary", "hp_balls",
                         "level_lcb", "uplift_adjusted"]].to_string(index=False))
    print("DONE — ratings complete")
    return 0


def write_methodology_report(reports: Path, bat_agg, bowl_agg, audit,
                             dist, lam_mean) -> None:
    bat_k = float(bat_agg["eb_k"].iloc[0])
    bowl_k = float(bowl_agg["eb_k"].iloc[0])
    text = f"""# Player Rating Methodology

_Generated by `scripts/run_ratings.py`; every number below is reproducible._

## Value metrics

**Batter value per delivery**
`BV = (runs_off_bat - xRuns) - lambda(state) * (striker_dismissed - p_wicket)`

**Bowler value per delivery**
`BoV = (xConceded - conceded) + lambda(state) * (bowler_wicket - p_bowler_wicket)`

`lambda(state)` is the dynamic wicket value: the expected remaining runs lost
when moving from w to w+1 wickets down within the same balls-left bin,
estimated from the leave-one-innings-out total-projection curves (mean
lambda = {lam_mean:.2f} runs). Run-outs, retired-out and obstructing
dismissals never credit the bowler; a striker is only charged for their own
dismissal.

## Aggregation

- **Level** = HP (top-20% within innings x phase) mean value, EB-shrunk:
  `shrunk = (n*mean + k*pool)/(n+k)` with k estimated by the
  method-of-moments from match-level means (batters k={bat_k:.1f},
  bowlers k={bowl_k:.1f}). Sensitivity: k/2 and 2k computed and stored.
- **Level_LCB** = shrunk level - 0.85 x bootstrap SE (match-cluster
  bootstrap, {N_BOOT} draws per player).
- **Uplift** = HP mean - Normal mean (raw), plus the context-adjusted
  residual of HP mean on Normal mean across the pool (the ranking uses the
  adjusted residual so a poor normal baseline is not rewarded), plus a
  normalised variant.
- **Consistency** = positive-HP-match share + match-level sd (match level,
  never delivery level). Pool split-half stability of HP match means:
  batters {bat_agg['split_half_stability_pool'].iloc[0]:.3f},
  bowlers {bowl_agg['split_half_stability_pool'].iloc[0]:.3f}.
- **Downside** = CVaR5 of HP-match means (mean of the worst 5% matches),
  downside deviation, catastrophic (<-2) rate.
- **Reliability** = HP balls/matches/innings, seasons, opponents, scenario
  diversity, role diversity, bootstrap CI.

## Composite

`Rating = 0.45 z(Level_LCB) + 0.20 z(Consistency) + 0.20 z(Uplift_adjusted)
+ 0.15 z(Volume) - 0.10 z(|CVaR5|)`

Sensitivity variants (level-heavy, balanced, uplift-heavy, risk-averse)
re-weight the same components; Spearman rank stability of each variant
against the primary ranking is printed by the runner and stored in
`all_player_metrics.csv`.

## Eligibility (from the sample distribution)

HP-balls distribution informed the tiers: {dist}

- inclusive: >= 40 HP balls and >= 6 HP matches
- moderate: >= 60 HP balls and >= 8 HP matches (headline lists)
- strict: >= 100 HP balls and >= 12 HP matches

Every player appears in `eligibility_audit.csv` with reasons; nobody is
silently excluded.

## Safeguards

- No player names or reputation anywhere in the scoring path.
- No player duplicated within any list (asserted).
- player_id is the identity key; names are display-only (verified mapping).
- Rankings are produced programmatically; nothing is hand-adjusted.
"""
    (reports / "player_rating_methodology.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
