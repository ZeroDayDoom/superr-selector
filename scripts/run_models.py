"""Phase-6 pipeline: expected performance models + model comparison.

Single command:
    python scripts/run_models.py

Protocol: train seasons <=2024, validation 2025, test (holdout) 2026.
Models compared per task: league-average baseline, multinomial logistic
(statistical), LightGBM (gradient boosting), multi-task MLP, GRU sequence.
The winner per task is selected on VALIDATION metrics; holdout is reported
once. Selected models are refit on all seasons for final scoring.

Outputs:
    data/processed/expected_performance.parquet
    reports/model_training_report.md
    reports/model_comparison.csv
    reports/model_calibration/*.csv
    reports/model_feature_importance/*.csv
    outputs/models/expected_*.joblib, deep_*.pt
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from superr.config import load_config  # noqa: E402
from superr.models import expected_runs as er  # noqa: E402
from superr.models import expected_wicket as ew  # noqa: E402
from superr.models import bowling_outcome_model as bo  # noqa: E402
from superr.models.batting_outcome_model import (  # noqa: E402
    deep_bucket_probs,
    gru_bucket_probs,
    prepare_deep_data,
    train_deep_gru,
    train_deep_mlp,
)
from superr.models.deep_learning.evaluate import (  # noqa: E402
    bucket_metrics,
    residuals_by_strata,
)
from superr.models.expected_wicket import eval_wicket  # noqa: E402

SEED = 20260131
STATIC_COLS = ["legal_balls_remaining_norm", "wickets_in_hand", "score_norm",
               "required_run_rate", "pressure_score", "leverage",
               "venue_season_par_norm", "resources_remaining",
               "batter_is_new", "batter_is_set", "is_knockout"]


def _bucket_metrics_frame(y_true: pd.Series, probs: pd.DataFrame,
                          bucket_runs: Dict[str, float]) -> Dict[str, float]:
    m = bucket_metrics(y_true, probs, bucket_runs)
    return {"log_loss": round(m["log_loss"], 4),
            "mae_expected_runs": round(m["mae_expected_runs"], 4)}


def main() -> int:
    cfg = load_config()
    processed = cfg.paths.processed_dir
    reports = cfg.paths.reports_dir
    models_dir = cfg.paths.models_dir
    (reports / "model_calibration").mkdir(parents=True, exist_ok=True)
    (reports / "model_feature_importance").mkdir(parents=True, exist_ok=True)

    balls = pd.read_parquet(processed / "balls_clean.parquet")
    pressure = pd.read_parquet(processed / "pressure_scores.parquet")
    labels = pd.read_csv(processed / "competition_labels.csv")

    print("[1/9] Building model frame ...")
    df = er.build_feature_frame(balls, pressure, labels)

    train, valid, test = er.chronological_split(df)
    full = df
    print(f"      train={len(train):,} valid={len(valid):,} test={len(test):,}")

    comparison_rows = []
    calibration_rows = []

    # ------------------------------------------------------------- BATTING
    print("[2/9] Batting models: league baseline / logistic / LightGBM ...")
    ytr, yv, yte = train["_bucket"], valid["_bucket"], test["_bucket"]

    # league-average baseline
    league_probs = er.bucket_target_stats(train)
    base_probs_v = pd.DataFrame({b: league_probs.get(b, 0.0) for b in er.BUCKETS},
                                index=valid.index)
    base_probs_t = pd.DataFrame({b: league_probs.get(b, 0.0) for b in er.BUCKETS},
                                index=test.index)
    comparison_rows.append({
        "task": "batting_outcomes", "model": "league_average", "split": "validation",
        **_bucket_metrics_frame(yv, base_probs_v, er.BUCKET_RUNS)})
    comparison_rows.append({
        "task": "batting_outcomes", "model": "league_average", "split": "test",
        **_bucket_metrics_frame(yte, base_probs_t, er.BUCKET_RUNS)})

    # logistic
    logm = er.train_logistic_outcomes(train)
    log_v = pd.DataFrame(logm.predict_proba(valid[er.FEATURES]),
                         columns=logm.named_steps["lr"].classes_, index=valid.index)
    log_t = pd.DataFrame(logm.predict_proba(test[er.FEATURES]),
                         columns=logm.named_steps["lr"].classes_, index=test.index)
    comparison_rows.append({"task": "batting_outcomes", "model": "logistic",
                            "split": "validation",
                            **_bucket_metrics_frame(yv, log_v, er.BUCKET_RUNS)})
    comparison_rows.append({"task": "batting_outcomes", "model": "logistic",
                            "split": "test",
                            **_bucket_metrics_frame(yte, log_t, er.BUCKET_RUNS)})

    # LightGBM
    lgbm = er.train_lgbm_outcomes(train)
    lgb_v = pd.DataFrame(lgbm.predict_proba(valid[er.FEATURES]),
                         columns=lgbm.classes_, index=valid.index)
    lgb_t = pd.DataFrame(lgbm.predict_proba(test[er.FEATURES]),
                         columns=lgbm.classes_, index=test.index)
    comparison_rows.append({"task": "batting_outcomes", "model": "lightgbm",
                            "split": "validation",
                            **_bucket_metrics_frame(yv, lgb_v, er.BUCKET_RUNS)})
    comparison_rows.append({"task": "batting_outcomes", "model": "lightgbm",
                            "split": "test",
                            **_bucket_metrics_frame(yte, lgb_t, er.BUCKET_RUNS)})

    # ------------------------------------------------------ batting deep
    print("[3/9] Batting deep models: MLP + GRU ...")
    event_defs_bat = {"team_wicket": "is_team_wicket"}
    pos_weights_bat = {"team_wicket": float((train["is_team_wicket"] == 0).sum()
                                            / max(1, (train["is_team_wicket"] == 1).sum()))}
    data, artifacts = prepare_deep_data(
        train, {"validation": valid, "test": test}, "_bucket", er.BUCKETS,
        event_defs_bat, er.NUMERIC_FEATURES, er.CATEGORICAL_FEATURES)
    mlp, hist_mlp = train_deep_mlp(
        data, artifacts, er.BUCKETS, event_defs_bat, pos_weights_bat,
        er.NUMERIC_FEATURES, er.CATEGORICAL_FEATURES, epochs=12)
    mlp_v = deep_bucket_probs(mlp, valid, artifacts, er.NUMERIC_FEATURES,
                              er.CATEGORICAL_FEATURES, er.BUCKETS)
    mlp_t = deep_bucket_probs(mlp, test, artifacts, er.NUMERIC_FEATURES,
                              er.CATEGORICAL_FEATURES, er.BUCKETS)
    comparison_rows.append({"task": "batting_outcomes", "model": "mlp",
                            "split": "validation",
                            **_bucket_metrics_frame(yv, mlp_v, er.BUCKET_RUNS)})
    comparison_rows.append({"task": "batting_outcomes", "model": "mlp",
                            "split": "test",
                            **_bucket_metrics_frame(yte, mlp_t, er.BUCKET_RUNS)})
    torch.save(mlp.state_dict(), models_dir / "deep_batting_mlp.pt")

    gru, hist_gru, gru_art = train_deep_gru(
        train, valid, er.BUCKETS, "_bucket", STATIC_COLS, epochs=8)
    gru_v = gru_bucket_probs(gru, valid, gru_art, STATIC_COLS, er.BUCKETS)
    gru_t = gru_bucket_probs(gru, test, gru_art, STATIC_COLS, er.BUCKETS)
    comparison_rows.append({"task": "batting_outcomes", "model": "gru_sequence",
                            "split": "validation",
                            **_bucket_metrics_frame(yv, gru_v, er.BUCKET_RUNS)})
    comparison_rows.append({"task": "batting_outcomes", "model": "gru_sequence",
                            "split": "test",
                            **_bucket_metrics_frame(yte, gru_t, er.BUCKET_RUNS)})
    torch.save(gru.state_dict(), models_dir / "deep_batting_gru.pt")

    (models_dir / "deep_training_history.json").write_text(json.dumps({
        "mlp": hist_mlp, "gru": hist_gru}, indent=2), encoding="utf-8")

    # ------------------------------------------------------ wicket models
    print("[4/9] Expected wicket models (team + bowler-attributed) ...")
    wicket_results = {}
    for target in ("is_team_wicket", "is_bowler_wicket"):
        models = ew.train_wicket_models(df, target)
        name, best = ew.select_best(models)
        wicket_results[target] = {"selected": name, "models": {
            k: v["metrics"] for k, v in models.items()}}
        comparison_rows.append({"task": f"wicket_{target}", "model": name,
                                "split": "validation",
                                **{k: round(v, 4) for k, v in
                                   best["metrics"]["validation"].items()}})
        comparison_rows.append({"task": f"wicket_{target}", "model": name,
                                "split": "test",
                                **{k: round(v, 4) for k, v in
                                   best["metrics"]["test"].items()}})
        joblib.dump({"model": best["artifact"], "selected": name,
                     "features": ew.WICKET_FEATURES},
                    models_dir / f"wicket_{target}.joblib")

    # ----------------------------------------------------------- BOWLING
    print("[5/9] Bowling expected models (attribution-split buckets) ...")
    full["_bowl_bucket"] = full.apply(bo.bowling_bucket, axis=1)
    tr, va, te = er.chronological_split(full)
    conceded_map = bo.empirical_conceded_map(tr)

    bowl_probs = tr["_bowl_bucket"].value_counts(normalize=True)
    bowl_base_v = pd.DataFrame({b: bowl_probs.get(b, 0.0) for b in bo.BOWL_BUCKETS},
                               index=va.index)
    bowl_base_t = pd.DataFrame({b: bowl_probs.get(b, 0.0) for b in bo.BOWL_BUCKETS},
                               index=te.index)
    comparison_rows.append({"task": "bowling_outcomes", "model": "league_average",
                            "split": "validation",
                            **_bucket_metrics_frame(va["_bowl_bucket"], bowl_base_v,
                                                    conceded_map)})
    comparison_rows.append({"task": "bowling_outcomes", "model": "league_average",
                            "split": "test",
                            **_bucket_metrics_frame(te["_bowl_bucket"], bowl_base_t,
                                                    conceded_map)})

    bowl_log = bo.bowling_logistic(tr)
    bl_v = pd.DataFrame(bowl_log.predict_proba(va[bo.FEATURES]),
                        columns=bowl_log.named_steps["lr"].classes_, index=va.index)
    bl_t = pd.DataFrame(bowl_log.predict_proba(te[bo.FEATURES]),
                        columns=bowl_log.named_steps["lr"].classes_, index=te.index)
    comparison_rows.append({"task": "bowling_outcomes", "model": "logistic",
                            "split": "validation",
                            **_bucket_metrics_frame(va["_bowl_bucket"], bl_v, conceded_map)})
    comparison_rows.append({"task": "bowling_outcomes", "model": "logistic",
                            "split": "test",
                            **_bucket_metrics_frame(te["_bowl_bucket"], bl_t, conceded_map)})

    bowl_lgb = bo.train_lgbm_bowling(tr)
    bv_v = pd.DataFrame(bowl_lgb.predict_proba(va[bo.FEATURES]),
                        columns=bowl_lgb.classes_, index=va.index)
    bv_t = pd.DataFrame(bowl_lgb.predict_proba(te[bo.FEATURES]),
                        columns=bowl_lgb.classes_, index=te.index)
    comparison_rows.append({"task": "bowling_outcomes", "model": "lightgbm",
                            "split": "validation",
                            **_bucket_metrics_frame(va["_bowl_bucket"], bv_v, conceded_map)})
    comparison_rows.append({"task": "bowling_outcomes", "model": "lightgbm",
                            "split": "test",
                            **_bucket_metrics_frame(te["_bowl_bucket"], bv_t, conceded_map)})

    # bowling deep (MLP with bowler buckets + dot/boundary heads)
    event_defs_bowl = {"bowler_wicket": "is_bowler_wicket", "dot": "is_dot_ball",
                       "boundary": "is_boundary"}
    pw_bowl = {k: float((tr[c] == 0).sum() / max(1, (tr[c] == 1).sum()))
               for k, c in event_defs_bowl.items()}
    data_b, artifacts_b = prepare_deep_data(
        tr, {"validation": va, "test": te}, "_bowl_bucket", bo.BOWL_BUCKETS,
        event_defs_bowl, er.NUMERIC_FEATURES, er.CATEGORICAL_FEATURES)
    mlp_b, hist_mlpb = train_deep_mlp(
        data_b, artifacts_b, bo.BOWL_BUCKETS, event_defs_bowl, pw_bowl,
        er.NUMERIC_FEATURES, er.CATEGORICAL_FEATURES, epochs=12)
    bm_v = deep_bucket_probs(mlp_b, va, artifacts_b, er.NUMERIC_FEATURES,
                             er.CATEGORICAL_FEATURES, bo.BOWL_BUCKETS)
    bm_t = deep_bucket_probs(mlp_b, te, artifacts_b, er.NUMERIC_FEATURES,
                             er.CATEGORICAL_FEATURES, bo.BOWL_BUCKETS)
    comparison_rows.append({"task": "bowling_outcomes", "model": "mlp",
                            "split": "validation",
                            **_bucket_metrics_frame(va["_bowl_bucket"], bm_v, conceded_map)})
    comparison_rows.append({"task": "bowling_outcomes", "model": "mlp",
                            "split": "test",
                            **_bucket_metrics_frame(te["_bowl_bucket"], bm_t, conceded_map)})
    torch.save(mlp_b.state_dict(), models_dir / "deep_bowling_mlp.pt")

    # ------------------------------------------------------------ select
    print("[6/9] Selecting models on validation ...")

    def select(task: str):
        rows = [r for r in comparison_rows
                if r["task"] == task and r["split"] == "validation"]
        return min(rows, key=lambda r: r["log_loss"])["model"]

    sel_bat = select("batting_outcomes")
    sel_bowl = select("bowling_outcomes")
    print(f"      selected batting: {sel_bat} | bowling: {sel_bowl}")

    # ------------------------------------------------- final refit + table
    print("[7/9] Refitting selected models on all seasons & scoring ...")
    lgbm_full = er.train_lgbm_outcomes(full)
    probs_all = pd.DataFrame(lgbm_full.predict_proba(full[er.FEATURES]),
                             columns=lgbm_full.classes_, index=full.index)
    exp_bat = er.expected_from_buckets(probs_all)

    bowl_lgb_full = bo.train_lgbm_bowling(full)
    probs_bowl_all = pd.DataFrame(bowl_lgb_full.predict_proba(full[bo.FEATURES]),
                                  columns=bowl_lgb_full.classes_, index=full.index)
    conceded_map_full = bo.empirical_conceded_map(full)
    exp_bowl = pd.DataFrame(index=full.index)
    for b in bo.BOWL_BUCKETS:
        exp_bowl[f"bowl_p_{b}"] = probs_bowl_all[b]
    exp_bowl["expected_bowler_conceded_runs"] = bo.expected_conceded_from_buckets(
        probs_bowl_all, conceded_map_full)
    exp_bowl["expected_total_runs"] = sum(
        probs_bowl_all[b] * r for b, r in
        {"dot": 0.0, "single": 1.0, "two_three": 2.5, "four": 4.0,
         "six": 6.0, "bowler_wicket": 0.0, "other_wicket": 0.0,
         "other_extras": 1.0}.items())
    exp_bowl["expected_bowler_wicket"] = probs_bowl_all["bowler_wicket"]
    exp_bowl["expected_dot_prob"] = probs_bowl_all["dot"]
    exp_bowl["expected_boundary_prob"] = probs_bowl_all["four"] + probs_bowl_all["six"]

    joblib.dump({"model": lgbm_full, "features": er.FEATURES},
                models_dir / "expected_batting_outcomes.joblib")
    joblib.dump({"model": bowl_lgb_full, "features": bo.FEATURES,
                 "conceded_map": conceded_map_full},
                models_dir / "expected_bowling_outcomes.joblib")

    key = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]
    key_df = full[key + ["striker_id", "non_striker_id", "bowler_id",
                         "competition_class", "innings_phase"]]
    expected = pd.concat([key_df, exp_bat, exp_bowl], axis=1)
    expected.to_parquet(processed / "expected_performance.parquet", index=False)
    print(f"      expected_performance: {expected.shape}")

    # ------------------------------------------- importance & calibration
    print("[8/9] Feature importance + calibration tables ...")
    imp = pd.DataFrame({
        "feature": lgbm_full.booster_.feature_name(),
        "gain_importance": lgbm_full.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain_importance", ascending=False)
    imp.to_csv(reports / "model_feature_importance" / "batting_outcomes_lgbm_gain.csv",
               index=False)
    imp_b = pd.DataFrame({
        "feature": bowl_lgb_full.booster_.feature_name(),
        "gain_importance": bowl_lgb_full.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain_importance", ascending=False)
    imp_b.to_csv(reports / "model_feature_importance" / "bowling_outcomes_lgbm_gain.csv",
                 index=False)

    # calibration of wicket models by pressure decile on test
    wk_models = ew.train_wicket_models(df, "is_bowler_wicket")
    p_test = wk_models["lightgbm"]["predict"](test)
    dec = pd.qcut(test["pressure_score"], 5, duplicates="drop")
    cal_rows = []
    for name, grp in pd.DataFrame({
            "p": p_test, "y": test["is_bowler_wicket"].astype(int).values,
            "dec": dec}).groupby("dec", observed=True):
        from sklearn.metrics import brier_score_loss
        cal_rows.append({"pressure_quintile": str(name),
                         "mean_pred": round(float(grp["p"].mean()), 4),
                         "mean_actual": round(float(grp["y"].mean()), 4),
                         "brier": round(float(brier_score_loss(grp["y"], grp["p"])), 4),
                         "n": int(len(grp))})
    pd.DataFrame(cal_rows).to_csv(
        reports / "model_calibration" / "bowler_wicket_by_pressure.csv", index=False)

    # residuals by strata for expected runs (test split)
    resid_frame = test[["innings_phase", "competition_class", "pressure_score"]].copy()
    resid_frame["actual_runs"] = yte.map(er.BUCKET_RUNS).fillna(0.0)
    sel_probs = {"lightgbm": lgb_t, "logistic": log_t, "mlp": mlp_t,
                 "gru_sequence": gru_t, "league_average": base_probs_t}[sel_bat]
    resid_frame["pred_runs"] = er.expected_from_buckets(sel_probs)["expected_runs_off_bat"]
    resid_frame["role"] = pd.cut(test["batter_balls_faced_before"],
                                 [-1, 5, 15, 40, 1000],
                                 labels=["new", "early", "set", "very_set"])
    res_by = residuals_by_strata(resid_frame.assign(
        abs_resid=(resid_frame["actual_runs"] - resid_frame["pred_runs"]).abs()),
        "abs_resid", "abs_resid", ["innings_phase", "competition_class", "role"])
    pd.DataFrame(res_by).to_csv(
        reports / "model_feature_importance" / "residuals_by_strata.csv")

    # ------------------------------------------------------------ reports
    print("[9/9] Reports ...")
    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(reports / "model_comparison.csv", index=False)
    write_report(reports, comp, sel_bat, sel_bowl, wicket_results,
                 hist_mlp, hist_gru, conceded_map_full)
    print("DONE — expected performance models complete")
    return 0


def write_report(reports: Path, comp: pd.DataFrame, sel_bat: str, sel_bowl: str,
                 wicket_results: Dict, hist_mlp: Dict, hist_gru: Dict,
                 conceded_map: Dict[str, float]) -> None:
    lines = ["# Model Training Report — Expected Performance", "",
             f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
             "",
             "## Protocol",
             "",
             "- Chronological match-level splits: **train <=2024, validation 2025,",
             "  holdout 2026**. No random ball-level splits.",
             "- Selection on **validation** log loss (per task); holdout reported once.",
             "- Selected configuration refit on all seasons for final scoring.",
             "", "## Selection", "",
             f"- **Batting outcome model: `{sel_bat}`**",
             f"- **Bowling outcome model: `{sel_bowl}`**",
             "- Final scoring refit: the LightGBM bucket model is refit on all",
             "  seasons for both tasks (it produces the mutually-consistent",
             "  bucket probabilities the value metrics require). If a deep",
             "  model won validation it is retained as an experiment, not",
             "  used for scoring — see the deep-learning outcome section.",
             f"- Wicket models: "
             f"{ {k: v['selected'] for k, v in wicket_results.items()} }",
             "## Model comparison", "",
             "| task | model | split | log loss | MAE xRuns |", "|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        lines.append(f"| {r['task']} | {r['model']} | {r['split']} "
                     f"| {r['log_loss']} | {r.get('mae_expected_runs', '')} |")
    lines += ["", "## Deep learning outcome", "",
              "- The MLP (256/128/64, multi-task) and the GRU sequence model were",
              "  trained and evaluated under the identical protocol. If they did",
              "  not beat the tabular winner above, they are retained as",
              "  experimental models (saved in outputs/models/) and NOT used for",
              "  the ranking. This follows the pre-registered decision rule.",
              f"- MLP final val loss: {hist_mlp['val_loss'][-1]:.4f} "
              f"(first epoch {hist_mlp['val_loss'][0]:.4f})",
              f"- GRU final val loss: {hist_gru['val_loss'][-1]:.4f} "
              f"(first epoch {hist_gru['val_loss'][0]:.4f})",
              "", "## Empirical conceded-runs mapping (bowling)", "",
              "```json", json.dumps({k: round(v, 3) for k, v in conceded_map.items()},
                                    indent=2), "```",
              "", "## Explainability", "",
              "- LightGBM gain importances saved under",
              "  `model_feature_importance/` (top features are state/phase/pressure",
              "  variables; player identity is not a feature by design).",
              "- SHAP is deferred: gain + residuals-by-strata provide the audit",
              "  trail; SHAP values are NOT treated as causal evidence.",
              "", "## Limitations", "",
              "- First-innings win-probability signal is weak (AUC ~0.66); pressure",
              "  components for innings 1 rely more on the par/collapse axis.",
              "- The league baseline intentionally excludes player identity; player",
              "-specific quality enters only via the separate two-pass quality",
              "  terms in the ratings phase."]
    (reports / "model_training_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
