"""Run the full Phase-1 data audit.

Usage (PowerShell, from the superr_selector project root):
    python scripts/run_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.config import load_config  # noqa: E402
from superr.data.audit_data import (  # noqa: E402
    build_inventory,
    build_schema_dictionary,
    discover_files,
    load_raw,
)
from superr.data.audit_checks import run_quality_checks  # noqa: E402
from superr.data.audit_semantics import (  # noqa: E402
    bowling_type_semantics,
    verify_batting_order,
    verify_legal_balls,
    verify_run_accounting,
)
from superr.data.audit_scope import (  # noqa: E402
    build_competition_audit,
    build_match_exclusion_audit,
    summarize_match_audit,
)
from superr.data.audit_report import (  # noqa: E402
    write_data_inventory_md,
    write_json,
    write_quality_report_md,
)


def main() -> int:
    cfg = load_config()
    reports = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    # Step 1: discovery (project dir + parent data pack dir)
    search_dirs = [PROJECT_ROOT, cfg.paths.raw_data_dir]
    discovery = discover_files(search_dirs)
    print(f"[1/8] File discovery: {len(discovery)} files found")
    zip_files = discovery[discovery["type"] == "zip"]
    if len(zip_files):
        print(f"      ZIP archives present: {list(zip_files['file'])}")

    # Step 2-3: load + inventory + schema
    data = load_raw(cfg.paths.raw_data_dir)
    inventory = build_inventory(data)
    print(f"[2/8] Loaded {len(data)} raw files; inventory built")

    schema = build_schema_dictionary(data)
    schema.to_csv(reports / "schema_dictionary.csv", index=False)
    print(f"[3/8] Schema dictionary: {len(schema)} columns documented")

    # Step 4-5: quality checks
    checks = run_quality_checks(data)
    print(f"[4/8] Quality checks: {len(checks)} executed")

    # Step 6: semantics verification
    b = data["ball_by_ball"]
    m = data["matches"]
    m = m.copy()
    semantics = {
        "run_accounting": verify_run_accounting(b, m),
        "legal_balls": verify_legal_balls(b),
        "batting_order": verify_batting_order(b),
        "bowling_types": bowling_type_semantics(b),
    }
    print("[5/8] Cricket semantics verified against match results")

    # Step 7: competition & match scope
    comp_audit = build_competition_audit(
        data["competitions"], m, b, data["mapping"]
    )
    comp_audit.to_csv(reports / "competition_audit.csv", index=False)
    match_audit = build_match_exclusion_audit(m, b, comp_audit)
    match_audit.to_csv(reports / "match_exclusion_audit.csv", index=False)
    scope_summary = summarize_match_audit(match_audit)
    print(f"[6/8] Competition audit: {len(comp_audit)} competitions; "
          f"match audit: {len(match_audit)} matches")

    # Step 9: artefacts
    write_data_inventory_md(
        reports / "data_inventory.md", discovery, inventory, semantics, scope_summary
    )
    write_quality_report_md(reports / "data_quality_report.md", checks, semantics)
    payload = {
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "checks": checks,
        "semantics": semantics,
        "scope_summary": scope_summary,
        "inventory": inventory.to_dict(orient="records"),
    }
    write_json(reports / "data_quality_checks.json", payload)
    print("[7/8] Reports written to", reports)

    # Fail gate. Documented, diagnosed single-row inconsistencies in the
    # free-text result string are non-blocking: rankings are computed from
    # ball-by-ball data, never from result strings. Each exception is listed
    # with its diagnosis in data_audit_decisions.md.
    documented_non_blocking = {20190}  # result string says 6 wickets; ball data: 5
    fails = [c for c in checks if c["status"] == "fail"]
    ra = semantics["run_accounting"]
    ra_bad = {x["match_id"] for x in ra["inconsistent_matches"]}
    undocumented = ra_bad - documented_non_blocking
    print(f"[8/8] Gate: {len(fails)} FAIL checks; "
          f"run-accounting anomalies: {sorted(ra_bad)} "
          f"(documented non-blocking: {sorted(ra_bad & documented_non_blocking)}, "
          f"undocumented: {sorted(undocumented)})")
    if fails:
        for f in fails:
            print("  FAIL:", f["check"], "-", f["detail"])
    verdict = "SAFE TO PROCEED" if not fails and not undocumented else "ISSUES NEED RESOLUTION"
    print("VERDICT:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
