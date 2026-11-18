"""Phase-5 pipeline: win probability, leverage, pressure models, validation.

Single command:
    python scripts/run_pressure.py

Outputs:
    data/processed/pressure_scores.parquet
    data/processed/win_probability_states.parquet
    reports/pressure_model_report.md
    reports/win_probability_validation.md
    reports/pressure_validation.json
    outputs/models/wp2.joblib, wp1.joblib, outcome_model.joblib
    outputs/figures/*.png
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superr.config import load_config  # noqa: E402
from superr.models import pressure_calibration as pcal  # noqa: E402
from superr.models.chase_pressure import add_chase_components  # noqa: E402
from superr.models.first_innings_pressure import (  # noqa: E402
    add_first_innings_components,
)
from superr.models.leverage_model import (  # noqa: E402
    compute_leverage,
    realized_wpa,
    train_outcome_model,
)
from superr.models.pressure_model import (  # noqa: E402
    add_categories,
    assemble_components,
    build_scores,
    calibrate_weights,
)
from superr.models.win_probability import (  # noqa: E402
    prepare_wp_frame,
    refit_full_wp,
    season_to_year,
    train_all_wp,
)

SEED = 20260131


def main() -> int:
    cfg = load_config()
    processed = cfg.paths.processed_dir
    reports = cfg.paths.reports_dir
    figs = cfg.paths.figures_dir
    models_dir = cfg.paths.models_dir

    feats_bat = pd.read_parquet(processed / "features_batting.parquet")
    feats_bowl = pd.read_parquet(processed / "features_bowling.parquet")
    feats_press = pd.read_parquet(processed / "features_pressure.parquet")
    balls = pd.read_parquet(processed / "balls_clean.parquet")

    # assemble the modelling frame: key + balls + needed feature columns
    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
    labels = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "competition_labels.csv"
    )
    df = balls.merge(
        labels[["comp_id", "competition_class"]], on="comp_id",
        how="left", validate="many_to_one",
    )

    def merge_new(frame: pd.DataFrame, right: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
        cols = [c for c in right.columns
                if (c in key or c not in df.columns) and c not in drop_cols]
        return frame.merge(right[cols], on=key, how="left", validate="one_to_one")

    df = merge_new(df, feats_bat, ["striker_id", "non_striker_id", "bowler_id"])
    df = merge_new(df, feats_bowl, ["striker_id", "non_striker_id", "bowler_id"])
    df = merge_new(df, feats_press, ["striker_id", "non_striker_id", "bowler_id"])
    print(f"[1/8] Modelling frame: {df.shape}")

    # ------------------------------------------------------------------ WP
    print("[2/8] Training win-probability models (chronological protocol) ...")
    wp_all = train_all_wp(df)
    wp2, wp1 = wp_all["wp2"], wp_all["wp1"]
    print(f"      wp2 selected: {wp2.selected}; wp1 selected: {wp1.selected}")
    for name, w in (("wp2", wp2), ("wp1", wp1)):
        h = w.metrics[w.selected]["holdout"]
        print(f"      {name} holdout: AUC={h['roc_auc']:.3f} Brier={h['brier']:.3f} "
              f"logloss={h['log_loss']:.3f} slope={h['calib_slope']:.2f}")

    joblib.dump({"model": wp2.model, "features": wp2.features,
                 "calibrator": wp2.calibrator, "selected": wp2.selected},
                models_dir / "wp2.joblib")
    joblib.dump({"model": wp1.model, "features": wp1.features,
                 "calibrator": wp1.calibrator, "selected": wp1.selected},
                models_dir / "wp1.joblib")

    # refit on all seasons for final scoring
    wp2_full = refit_full_wp(df, wp2)
    wp1_full = refit_full_wp(df, wp1)
    joblib.dump({"model": wp2_full.model, "features": wp2_full.features,
                 "calibrator": wp2_full.calibrator, "selected": wp2_full.selected},
                models_dir / "wp2_full.joblib")
    joblib.dump({"model": wp1_full.model, "features": wp1_full.features,
                 "calibrator": wp1_full.calibrator, "selected": wp1_full.selected},
                models_dir / "wp1_full.joblib")

    # ------------------------------------------------------- outcome model
    print("[3/8] Training outcome-probability model (leverage engine) ...")
    om = train_outcome_model(df)
    joblib.dump(om, models_dir / "outcome_model.joblib")

    # ----------------------------------------------------------- leverage
    print("[4/8] Computing per-delivery leverage and realized WPA ...")
    lev = compute_leverage(df, om, wp2_full, wp1_full)
    r_wpa = realized_wpa(df, wp2_full, wp1_full)
    df["wp_before"] = lev["wp_before"]
    df["leverage_expected_abs_swing"] = lev["leverage_expected_abs_swing"]
    df["leverage_expected_dir_swing"] = lev["leverage_expected_dir_swing"]
    df["realized_wpa"] = r_wpa

    # -------------------------------------------------------- components
    print("[5/8] Assembling pressure components ...")
    df = prepare_wp_frame(df)
    df = add_first_innings_components(df)
    df = add_chase_components(df)
    df = assemble_components(df)

    # ------------------------------------------------- weight calibration
    print("[6/8] Calibrating model weights on the validation season (2025) ...")
    d = df.copy()
    d["_season_year"] = season_to_year(d["season"])
    valid = d[d["_season_year"] == 2025]
    calibration = calibrate_weights(valid)
    print("      A:", round(calibration["model_a"]["metrics"]["wpa_swing_ratio"], 2),
          "| B weights:", calibration["model_b"]["weights"],
          "| C weights:", calibration["model_c"]["weights"])

    # ------------------------------------------------ final scores + labels
    print("[7/8] Scoring all deliveries ...")
    df = build_scores(df,
                      weights_b=calibration["model_b"]["weights"],
                      weights_c=calibration["model_c"]["weights"])
    df["innings_phase"] = np.select(
        [(df["legal_balls_before"] // 6 + 1) <= 6,
         (df["legal_balls_before"] // 6 + 1) <= 15],
        ["powerplay", "middle"], default="death",
    )
    df = add_categories(df)

    score_cols = ["wp_before", "leverage_expected_abs_swing",
                  "leverage_expected_dir_swing", "realized_wpa"] + \
                 [c for c in df.columns if c.startswith("pc_")] + \
                 ["PRESSURE_MODEL_A", "PRESSURE_MODEL_B", "PRESSURE_MODEL_C",
                  "PRESSURE_DISPLAY_100", "pressure_category",
                  "hp_top10", "hp_top15", "hp_top20", "hp_top25", "hp_top30",
                  "hp_overall_top10", "hp_overall_top15", "hp_overall_top20",
                  "hp_overall_top25", "hp_overall_top30"]
    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num",
           "striker_id", "non_striker_id", "bowler_id"]
    scores = df[key + score_cols]
    scores.to_parquet(processed / "pressure_scores.parquet", index=False)

    wp_states_cols = key + ["season", "competition_class", "innings_phase"] + [
        c for c in ["runs_required_before", "legal_balls_remaining",
                    "wickets_before", "required_run_rate", "target",
                    "score_before", "is_dls_revised"] if c in df.columns
    ] + ["wp_before", "leverage_expected_abs_swing", "realized_wpa",
         "_batting_team_won"]
    wp_states = df[wp_states_cols]
    wp_states.to_parquet(processed / "win_probability_states.parquet", index=False)

    # -------------------------------------------------------- validation
    print("[8/8] Pressure validation + reports ...")
    validation = run_validation(df, wp2, wp1, calibration, figs)
    # persist the calibrated weights so downstream consumers (dashboard
    # simulator, reports) use the exact engine weights, not stale copies
    validation["calibrated_weights"] = {
        "model_b": {k: float(v) for k, v
                    in calibration["model_b"]["weights"].items()},
        "model_c": {k: float(v) for k, v
                    in calibration["model_c"]["weights"].items()},
    }
    (reports / "pressure_validation.json").write_text(
        json.dumps(validation, indent=2, default=str), encoding="utf-8")
    write_reports(reports, df, wp2, wp1, calibration, validation, om)

    print("DONE — pressure engine complete")
    return 0


def run_validation(df, wp2, wp1, calibration, figs) -> dict:
    """High-pressure vs ordinary comparison + worked examples."""
    out: dict = {}
    d = df.copy()
    d["_season_year"] = season_to_year(d["season"])
    hp = d["hp_top20"].astype(bool)
    rest = ~hp

    def rate(x: pd.DataFrame) -> dict:
        legal = x["is_legal_ball"]
        runs = x["batter_runs"]
        return {
            "strike_rate": float(runs[legal].sum() / max(1, legal.sum()) * 100),
            "dot_pct": float((x["is_dot_ball"]).mean() * 100),
            "boundary_pct": float((x["is_boundary"]).mean() * 100),
            "wicket_prob": float((x["is_team_wicket"]).mean() * 100),
            "bowling_economy": float(x["bowler_runs_conceded"].sum()
                                     / max(1, legal.sum()) * 6),
            "mean_abs_realized_wpa": float(x["realized_wpa"].abs().mean()),
            "mean_leverage": float(x["leverage_expected_abs_swing"].mean()),
            "n": int(len(x)),
        }

    out["hp_vs_rest"] = {"high_pressure": rate(d[hp]), "ordinary": rate(d[rest])}
    out["ratios"] = {
        "variance_ratio_runs": float(d.loc[hp, "total_runs"].var()
                                     / d.loc[rest, "total_runs"].var()),
        "wpa_swing_ratio": float(d.loc[hp, "realized_wpa"].abs().mean()
                                 / d.loc[rest, "realized_wpa"].abs().mean()),
    }

    # distributions
    over_idx = (d["legal_balls_before"] // 6 + 1)
    out["pressure_by_over"] = d.groupby(over_idx)["PRESSURE_MODEL_C"].mean().round(4).to_dict()
    out["pressure_by_wickets"] = d.groupby("wickets_before")["PRESSURE_MODEL_C"].mean().round(4).to_dict()
    rrr_bin = pd.cut(d["required_run_rate"], [0, 6, 8, 10, 12, 20],
                     labels=["<6", "6-8", "8-10", "10-12", "12+"])
    out["pressure_by_rrr"] = d.groupby(rrr_bin, observed=True)["PRESSURE_MODEL_C"].mean().round(4).to_dict()
    out["pressure_by_innings"] = d.groupby("innings")["PRESSURE_MODEL_C"].mean().round(4).to_dict()
    out["pressure_by_phase"] = d.groupby("innings_phase")["PRESSURE_MODEL_C"].mean().round(4).to_dict()
    out["hp_share_by_competition"] = d.groupby("competition_class")["hp_top20"].mean().round(4).to_dict()

    # calibration of WP by pressure quintile (does WP stay calibrated everywhere?)
    d2 = d[d["innings"] == 2].dropna(subset=["wp_before", "_batting_team_won"])
    dec = pd.qcut(d2["PRESSURE_MODEL_C"], 5, duplicates="drop")
    cal_by_dec = {}
    for name, grp in d2.groupby(dec, observed=True):
        cal_by_dec[str(name)] = {
            "ece": round(pcal.expected_calibration_error(
                grp["_batting_team_won"], grp["wp_before"]), 4),
            "n": int(len(grp)),
        }
    out["wp_calibration_by_pressure_decile"] = cal_by_dec

    # worked examples: 10 interesting real states
    out["worked_examples"] = worked_examples(d)

    # figures
    make_figures(d, wp2, wp1, figs)
    return out


def worked_examples(d: pd.DataFrame) -> list:
    """10 real match states with component-level explanations."""
    examples = []
    d = d.copy()
    over_idx = (d["legal_balls_before"] // 6 + 1)

    picks = {
        "extreme_death_chase": d[(d["innings"] == 2) & (over_idx == 20)
                                 & (d["required_run_rate"] > 12) & (d["wickets_in_hand"] >= 2)],
        "easy_chase_end": d[(d["innings"] == 2) & (d["runs_required_before"] <= 10)
                            & (d["legal_balls_remaining"] >= 18)],
        "powerplay_start": d[(over_idx == 1) & (d["legal_balls_before"] == 0)],
        "collapse_middle": d[(d["last12_wickets"] >= 3) & (over_idx.between(8, 14))],
        "anchor_under_par": d[(d["score_vs_par"] <= -25) & (over_idx.between(10, 15))
                              & (d["innings"] == 1)],
        "high_rrr_mid": d[(d["innings"] == 2) & d["required_run_rate"].between(11, 14)
                          & (over_idx.between(10, 15)) & (d["wickets_in_hand"] >= 4)],
        "defending_low_total": d[(d["innings"] == 1) & (d["score_vs_par"] <= -30)
                                 & (over_idx >= 16)],
        "knockout_death": d[(d["is_knockout"] == 1) & (over_idx >= 18)],
        "boundary_drought": d[d["balls_since_last_boundary"] >= 24],
        "new_batter_after_wicket": d[(d["entering_after_wicket"] == True)
                                     & (d["batter_is_new"] == True) & (d["wickets_before"] >= 4)],
    }
    component_cols = ["pc_difficulty", "pc_resource_stress", "pc_collapse_pressure",
                      "pc_momentum_stress", "pc_phase_pressure", "pc_match_importance",
                      "pc_match_closeness", "pc_leverage_norm"]

    for name, sel in picks.items():
        if len(sel) == 0:
            # relax: nearest match by score
            sel = d.nlargest(1, "PRESSURE_MODEL_C")
        # representative (median-pressure) member of the scenario, not the
        # single most extreme delivery, so examples describe the scenario
        med = sel["PRESSURE_MODEL_C"].median()
        row = sel.iloc[(sel["PRESSURE_MODEL_C"] - med).abs().argmin()]
        comps = {c.replace("pc_", ""): round(float(row[c]), 3)
                 for c in component_cols if pd.notna(row.get(c, np.nan))}
        examples.append({
            "example": name,
            "match_id": int(row["match_id"]),
            "innings": int(row["innings"]),
            "over": int(row["overid"]),
            "score_before": float(row["score_before"]),
            "wickets_before": int(row["wickets_before"]),
            "balls_remaining": int(row["legal_balls_remaining"]),
            "target": (None if pd.isna(row.get("target")) else float(row["target"])),
            "required_run_rate": (None if pd.isna(row.get("required_run_rate"))
                                  else round(float(row["required_run_rate"]), 2)),
            "wp_before": round(float(row["wp_before"]), 3),
            "leverage": round(float(row["leverage_expected_abs_swing"]), 4),
            "components": comps,
            "pressure_a": round(float(row["PRESSURE_MODEL_A"]), 3),
            "pressure_b": round(float(row["PRESSURE_MODEL_B"]), 3),
            "pressure_c": round(float(row["PRESSURE_MODEL_C"]), 3),
            "category": str(row["pressure_category"]),
        })
    return examples


def make_figures(d: pd.DataFrame, wp2, wp1, figs: Path) -> None:
    plt.rcParams.update({"figure.dpi": 120, "font.size": 8})

    # 1. pressure by over
    fig, ax = plt.subplots(figsize=(7, 3))
    over_idx = (d["legal_balls_before"] // 6 + 1)
    for inn, grp in d.groupby("innings"):
        if inn > 2:
            continue
        ax.plot(grp.groupby(over_idx.loc[grp.index])["PRESSURE_MODEL_C"].mean(),
                label=f"innings {inn}")
    ax.set_xlabel("Over"); ax.set_ylabel("Mean pressure (Model C)")
    ax.set_title("Pressure profile by over"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "pressure_by_over.png"); plt.close(fig)

    # 2. WP calibration (holdout-style: all chase deliveries, full model)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    d2 = d[(d["innings"] == 2) & d["_batting_team_won"].notna()]
    t = pcal.reliability_table(d2["_batting_team_won"], d2["wp_before"])
    axes[0].plot(t["bin_pred"], t["bin_obs"], "o-")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_xlabel("Predicted WP"); axes[0].set_ylabel("Observed win rate")
    axes[0].set_title("WP2 reliability (all seasons)")
    dec = pd.qcut(d2["PRESSURE_MODEL_C"], 5, duplicates="drop")
    for i, (name, grp) in enumerate(d2.groupby(dec, observed=True)):
        tt = pcal.reliability_table(grp["_batting_team_won"], grp["wp_before"], n_bins=5)
        axes[1].plot(tt["bin_pred"], tt["bin_obs"], "o-", alpha=0.7, label=str(name))
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_xlabel("Predicted WP"); axes[1].set_title("WP2 by pressure quintile")
    axes[1].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(figs / "wp_calibration.png"); plt.close(fig)

    # 3. component distributions
    comp_cols = [c for c in d.columns if c.startswith("pc_")]
    fig, axes = plt.subplots(1, len(comp_cols), figsize=(2.2 * len(comp_cols), 2.4),
                             sharey=True)
    for ax, c in zip(axes, comp_cols):
        ax.hist(d[c].dropna(), bins=30, color="#2b6099")
        ax.set_title(c.replace("pc_", ""), fontsize=7)
    fig.suptitle("Pressure component distributions")
    fig.tight_layout(); fig.savefig(figs / "pressure_components.png"); plt.close(fig)

    # 4. category distribution
    fig, ax = plt.subplots(figsize=(5, 2.6))
    order = ["LOW", "MODERATE", "HIGH", "EXTREME"]
    counts = d["pressure_category"].value_counts().reindex(order).fillna(0)
    ax.bar(counts.index, counts.values, color="#c8102e")
    ax.set_title("Pressure categories (distribution-based)")
    fig.tight_layout(); fig.savefig(figs / "pressure_categories.png"); plt.close(fig)


def write_reports(reports: Path, df, wp2, wp1, calibration, validation, om) -> None:
    # win_probability_validation.md
    lines = ["# Win Probability Validation", ""]
    for name, w in (("WP2 (chase)", wp2), ("WP1 (first innings)", wp1)):
        lines += [f"## {name}", "",
                  f"- Selected model family: **{w.selected}**",
                  "- Chronological protocol: train <=2024, calibrate 2025, holdout 2026", "",
                  "| family | split | ROC-AUC | PR-AUC | log loss | Brier | calib slope |",
                  "|---|---|---|---|---|---|---|"]
        for fam, splits in w.metrics.items():
            for split, m in splits.items():
                lines.append(f"| {fam} | {split} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} "
                             f"| {m['log_loss']:.3f} | {m['brier']:.3f} | {m['calib_slope']:.2f} |")
        lines.append("")
    v = validation
    lines += ["## WP calibration by pressure quintile (WP2, all seasons)", "",
              "| pressure quintile | ECE | n |", "|---|---|---|"]
    for k, v2 in (validation.get("wp_calibration_by_pressure_decile") or {}).items():
        lines.append(f"| {k} | {v2['ece']} | {v2['n']} |")
    lines.append("")
    (reports / "win_probability_validation.md").write_text("\n".join(lines), encoding="utf-8")

    # pressure_model_report.md
    hp = validation["hp_vs_rest"]["high_pressure"]
    rest = validation["hp_vs_rest"]["ordinary"]
    ratios = validation["ratios"]
    lines = ["# Pressure Model Report", "",
             f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_", "",
             "## Selected model", "",
             "- **PRESSURE MODEL C (hybrid) is the primary pressure score.**",
             f"- Model B weights (data-calibrated on the 2025 validation season): "
             f"`{calibration['model_b']['weights']}`",
             f"- Model C mix: `{calibration['model_c']['weights']}`",
             "- Model A is retained as the transparent rules-based anchor.",
             "", "## Weight calibration criterion (pre-registered)", "",
             "Weights maximise the mean of (a) outcome-variance ratio HP vs rest,",
             "(b) realized-|WPA| ratio HP vs rest, and (c) split-half stability of",
             "player HP means — computed on the 2025 validation season only.", "",
             "## High-pressure (top 20% within innings x phase) vs ordinary deliveries", "",
             "| metric | high pressure | ordinary |", "|---|---|---|"]
    for k in hp:
        lines.append(f"| {k} | {hp[k]:.3f} | {rest[k]:.3f} |")
    lines += ["", f"- Variance ratio (runs): **{ratios['variance_ratio_runs']:.2f}x**",
              f"- Realized |WPA| ratio: **{ratios['wpa_swing_ratio']:.2f}x**", "",
             "## Pressure distributions", ""]
    for key_ in ("pressure_by_over", "pressure_by_wickets", "pressure_by_rrr",
                 "pressure_by_innings", "pressure_by_phase", "hp_share_by_competition"):
        lines.append(f"- `{key_}`: {validation.get(key_)}")
    lines += ["", "## Worked examples (10 real match states)", ""]
    for ex in validation["worked_examples"]:
        lines += [f"### {ex['example']} — match {ex['match_id']}, "
                  f"innings {ex['innings']}, over {ex['over']}", "",
                  f"- State: {ex['score_before']:.0f}/{ex['wickets_before']}, "
                  f"{ex['balls_remaining']} balls left"
                  + (f", target {ex['target']:.0f}, RRR {ex['required_run_rate']}"
                     if ex["target"] else ""),
                  f"- WP(before) = {ex['wp_before']}, leverage = {ex['leverage']}",
                  f"- Components: {ex['components']}",
                  f"- Pressure A={ex['pressure_a']}, B={ex['pressure_b']}, "
                  f"**C={ex['pressure_c']}** -> category **{ex['category']}**",
                  f"- Why: the weighted component mix above — dominant components "
                  f"are {[k for k, v in sorted(ex['components'].items(), key=lambda kv: -kv[1])[:3]]}",
                  ""]
    (reports / "pressure_model_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
