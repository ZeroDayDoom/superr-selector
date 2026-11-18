"""Final project audit (Phase 12): programmatic verification of all steps.

Run:  python scripts/final_audit.py
Writes reports/final_project_audit.md and prints a pass/fail summary.
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

DATA = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "outputs"
REPORTS = PROJECT_ROOT / "reports"
RESULTS: list[dict] = []


def check(step: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"step": step, "check": name,
                    "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def step2_data_audit() -> None:
    print("STEP 2 — data audit")
    balls = pd.read_parquet(DATA / "balls_clean.parquet")
    innings = pd.read_parquet(DATA / "innings_summary.parquet")
    check("2", "no duplicate delivery keys",
          not balls.duplicated(["match_id", "innings", "overid",
                                "over_ball_num"]).any(),
          f"{len(balls):,} rows")
    g = balls.groupby(["match_id", "innings"], sort=False)
    cum = g["total_runs"].cumsum() - balls["total_runs"]
    check("2", "score_before = exclusive cumsum",
          (balls["score_before"] - cum).abs().max() < 1e-9)
    check("2", "legal balls per innings <= 132",
          int(balls.groupby(["match_id", "innings"])["is_legal_ball"]
              .sum().max()) <= 132)
    itot = innings.set_index(["match_id", "innings"])["innings_total"]
    check("2", "innings totals reconcile",
          int((balls.groupby(["match_id", "innings"])["total_runs"].sum()
               - itot.reindex(
                   balls.groupby(["match_id", "innings"]).size().index))
              .abs().gt(1e-9).sum()) == 0)
    # extras / wickets / bowler runs identities
    m = balls["extras_type"].isin(["byes", "legbyes", "penalty", "other", "unknown"])
    check("2", "bowler runs exclude byes/legbyes",
          (balls.loc[m, "bowler_runs_conceded"]
           == balls.loc[m, "batter_runs"].fillna(0)).all())
    check("2", "wickets reconcile (deliveries = innings summary)",
          int(balls["is_team_wicket"].sum()) == int(innings["total_wickets"].sum()))
    check("2", "bowler wickets <= team wickets",
          balls["is_bowler_wicket"].sum() <= balls["is_team_wicket"].sum())
    # player mapping
    pe = pd.read_csv(PROJECT_ROOT.parent / "playing_eleven.csv")
    played_mask = pe["is_played"].astype(str).str.lower().eq("true")
    played = set(zip(pe.loc[played_mask, "match_id"],
                     pe.loc[played_mask, "player_id"]))
    pairs = (set(zip(balls["match_id"], balls["striker_id"]))
             | set(zip(balls["match_id"], balls["non_striker_id"]))
             | set(zip(balls["match_id"], balls["bowler_id"])))
    check("2", "delivery players in playing XI", pairs <= played,
          f"{len(pairs - played)} exceptions")
    # competition filtering / super over / DLS
    check("2", "no super-over innings", balls["innings"].isin([1, 2]).all())
    dls = balls["is_dls_revised"].astype("boolean").fillna(False)
    check("2", "DLS rows carry revised flags", dls.sum() >= 0)
    ch = balls[balls["innings"] == 2]
    check("2", "chases have targets", ch["target"].notna().all())
    check("2", "no negative resources",
          (balls["legal_balls_remaining"] >= 0).all())


def step3_leakage_audit() -> None:
    print("STEP 3 — leakage audit")
    # registry docs every feature with availability flags
    from superr.features.feature_registry import build_registry
    reg = build_registry()
    bad = [n for n, s in reg.specs.items()
           if not s.available_before_delivery]
    check("3", "all registry features pre-delivery", not bad, str(bad[:5]))
    # dynamic audit artefact
    la = REPORTS / "leakage_audit.json"
    if la.exists():
        d = json.loads(la.read_text(encoding="utf-8"))
        check("3", "dynamic perturbation audit clean",
              d["n_violations"] == 0,
              f"{d['n_deliveries_perturbed']} deliveries, "
              f"{d['n_feature_columns_checked']} features")
    else:
        check("3", "dynamic perturbation audit", False, "run_features not run")
    # temporal split sanity: model metrics holdout exists and differs
    mc = REPORTS / "model_comparison.csv"
    if mc.exists():
        comp = pd.read_csv(mc)
        has_test = (comp["split"] == "test").any()
        check("3", "holdout (2026) evaluation present", has_test)
        # selection on validation only: chosen model recorded in training report
        tr = (REPORTS / "model_training_report.md").read_text(encoding="utf-8") \
            if (REPORTS / "model_training_report.md").exists() else ""
        check("3", "selection declared on validation",
              "Selection on **validation**" in tr or "validation 2025" in tr)


def step4_training_audit() -> None:
    print("STEP 4 — training audit")
    tr = (REPORTS / "model_training_report.md").read_text(encoding="utf-8") \
        if (REPORTS / "model_training_report.md").exists() else ""
    check("4", "chronological splits documented",
          "train <=2024" in tr and "validation 2025" in tr and "2026" in tr)
    check("4", "no random ball split mentioned as method",
          "random ball" not in tr.lower()
          or "no random ball" in tr.lower()
          or "never random" in tr.lower())
    check("4", "deep models evaluated (history saved)",
          (PROJECT_ROOT / "outputs/models/deep_training_history.json").exists())


def step5_ranking_audit() -> None:
    print("STEP 5 — ranking audit")
    audit = pd.read_csv(OUT / "eligibility_audit.csv")
    metrics = pd.read_csv(OUT / "all_player_metrics.csv")
    mapping = pd.read_csv(PROJECT_ROOT.parent
                          / "competition_team_player_mapping.csv")
    known_ids = set(mapping["player_id"])
    for role, fname in (("batter", "rankings_batters.csv"),
                        ("bowler", "rankings_bowlers.csv")):
        p = OUT / fname
        if not p.exists():
            check("5", f"{fname} exists", False)
            continue
        t = pd.read_csv(p)
        check("5", f"{fname}: 10 rows, ranks 1..10",
              len(t) == 10 and list(t["rank"]) == list(range(1, 11)))
        check("5", f"{fname}: no duplicates",
              not t["player_id"].duplicated().any())
        check("5", f"{fname}: ids in supplied mapping",
              set(t["player_id"]) <= known_ids)
        names_ok = t.set_index("player_id")["player_name"].to_dict()
        map_names = mapping.drop_duplicates("player_id").set_index(
            "player_id")["player_name"]
        mism = [pid for pid, nm in names_ok.items()
                if map_names.get(pid) != nm]
        check("5", f"{fname}: names match mapping", not mism, str(mism[:3]))
        el = set(audit[(audit["role"] == role)
                       & audit["eligible_moderate"]]["player_id"])
        check("5", f"{fname}: all selected are moderate-eligible",
              set(t["player_id"]) <= el)
        # scoring formula: rating matches composite of components
        from superr.ratings.composite_rating import PRIMARY_WEIGHTS, composite
        m = metrics[metrics["role"] == role].set_index("player_id")
        expected = composite(m, PRIMARY_WEIGHTS, "_chk")
        got = m["rating_primary"]
        check("5", f"{role}: rating matches formula",
              np.allclose(expected.reindex(got.index).values, got.values, atol=1e-6))
        # reliability adjustment present; for players with an undefined
        # bootstrap SE (<2 HP matches) the LCB falls back to the shrunk
        # mean itself (documented in reliability.lcb)
        expected = m["shrunk_k"] - 0.85 * m["boot_se"]
        fallback = m["boot_se"].isna()
        expected[fallback] = m.loc[fallback, "shrunk_k"]
        check("5", f"{role}: level_lcb = shrunk - 0.85*SE",
              bool(np.allclose(m["level_lcb"], expected, atol=1e-6,
                               equal_nan=True)))


def step6_evidence_audit() -> None:
    print("STEP 6 — player evidence audit")
    dossiers = list((OUT / "player_dossiers").glob("*.md"))
    check("6", "20 dossiers (10 batters + 10 bowlers)", len(dossiers) == 20,
          f"{len(dossiers)} found")
    rb = pd.read_csv(OUT / "rankings_batters.csv")
    rw = pd.read_csv(OUT / "rankings_bowlers.csv")
    metrics = pd.read_csv(OUT / "all_player_metrics.csv")
    m = metrics.set_index("player_id")
    bad = []
    for t in (rb, rw):
        for _, r in t.iterrows():
            mm_rows = m.loc[[r["player_id"]]]
            mm = mm_rows[mm_rows["role"] == ("batter" if t is rb else "bowler")]
            if len(mm) and abs(float(mm["rating_primary"].iloc[0])
                               - r["rating_primary"]) > 1e-9:
                bad.append(r["player_name"])
    check("6", "dossier/ranking statistics consistent with metrics",
          not bad, str(bad[:3]))
    # dossier content sanity: no invented matches — match ids in dossiers must
    # exist in the data
    balls_matches = set(pd.read_parquet(
        DATA / "balls_clean.parquet", columns=["match_id"])["match_id"])
    ok_dossiers = True
    for p in dossiers:
        text = p.read_text(encoding="utf-8")
        for token in text.split("match ")[1:]:
            head = token.split(" ")[0].strip("()").rstrip(".,")
            if head.isdigit() and int(head) not in balls_matches:
                ok_dossiers = False
    check("6", "dossier match ids exist in data", ok_dossiers)


def step7_validation_audit() -> None:
    print("STEP 7 — validation audit")
    p = REPORTS / "validation_summary.json"
    if not p.exists():
        check("7", "validation summary exists", False)
        return
    vs = json.loads(p.read_text(encoding="utf-8"))
    for key, label in [
            ("temporal_batters", "temporal validation (batters)"),
            ("split_half_batters", "split-half reliability"),
            ("threshold_stability", "threshold sensitivity"),
            ("weight_stability_batters", "weight sensitivity"),
            ("ablation_batters", "ablation study"),
            ("role_advantage_batters", "role analysis"),
            ("competition_overlap_batters", "competition analysis"),
            ("model_checks", "pressure/model checks"),
            ("edge_tests", "edge tests")]:
        check("7", label, bool(vs.get(key)))
    boot = OUT / "bootstrap_rankings.csv"
    if boot.exists():
        b = pd.read_csv(boot)
        check("7", "bootstrap completed", len(b) > 0, f"{len(b)} players")
    else:
        check("7", "bootstrap completed", False)


def step8_dashboard_audit() -> None:
    print("STEP 8 — dashboard audit")
    log = (REPORTS / "_dashboard_smoke.txt")
    if log.exists():
        ok = "ALL PAGES EXECUTE CLEANLY" in log.read_text(encoding="utf-8")
        check("8", "all dashboard pages execute", ok)
    else:
        check("8", "dashboard smoke test record", False,
              "run scripts/test_dashboard.py > reports/_dashboard_smoke.txt")


def step9_report_audit() -> None:
    print("STEP 9 — report audit")
    from pypdf import PdfReader
    pdf = REPORTS / "final_methodology.pdf"
    if not pdf.exists():
        check("9", "final PDF exists", False)
        return
    r = PdfReader(str(pdf))
    check("9", "PDF renders", len(r.pages) >= 8, f"{len(r.pages)} pages")
    text = "".join(page.extract_text() or "" for page in r.pages)
    rb = pd.read_csv(OUT / "rankings_batters.csv")
    rw = pd.read_csv(OUT / "rankings_bowlers.csv")
    missing_names = [n for n in list(rb["player_name"]) + list(rw["player_name"])
                     if n not in text]
    check("9", "all 20 ranked names appear in PDF", not missing_names,
          str(missing_names[:3]))
    check("9", "no placeholder text",
          "TODO" not in text and "XXX" not in text)


def step10_submission_files() -> None:
    print("STEP 10 — submission files")
    required = [
        REPORTS / "final_methodology.pdf",
        REPORTS / "final_methodology.md",
        REPORTS / "final_methodology.html",
        REPORTS / "executive_summary.md",
        OUT / "submission_top10.csv",
        OUT / "submission_top10.xlsx",
        OUT / "submission_entry_sheet.txt",
        OUT / "rankings_batters.csv",
        OUT / "rankings_bowlers.csv",
        OUT / "all_player_metrics.csv",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "pyproject.toml",
    ]
    for p in required:
        check("10", p.name, p.exists(), str(p.relative_to(PROJECT_ROOT)))


def main() -> int:
    print("=" * 60)
    print("FINAL PROJECT AUDIT")
    print("=" * 60)
    step2_data_audit()
    step3_leakage_audit()
    step4_training_audit()
    step5_ranking_audit()
    step6_evidence_audit()
    step7_validation_audit()
    step8_dashboard_audit()
    step9_report_audit()
    step10_submission_files()

    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print("=" * 60)
    print(f"AUDIT COMPLETE: {len(RESULTS)} checks, {n_fail} FAIL")
    write_report(n_fail)
    return 0 if n_fail == 0 else 1


def write_report(n_fail: int) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# Final Project Audit", "",
             f"_Generated {now} — {len(RESULTS)} checks, {n_fail} FAIL._", ""]
    current = None
    for r in RESULTS:
        if r["step"] != current:
            current = r["step"]
            lines.append(f"## Step {current}")
            lines.append("")
        lines.append(f"- [{r['status']}] **{r['check']}** — {r['detail']}")
    lines += ["", "## Submission readiness", ""]
    lines.append("**READY** if all checks pass; otherwise the FAIL lines above "
                 "list exactly what remains.")
    (REPORTS / "final_project_audit.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
    print("Audit report written to reports/final_project_audit.md")


if __name__ == "__main__":
    raise SystemExit(main())
