"""Report writers for the data audit (markdown + CSV + JSON artefacts)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _status_counts(checks: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"pass": 0, "warn": 0, "fail": 0, "info": 0}
    for c in checks:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out


def write_data_inventory_md(
    path: Path,
    discovery: pd.DataFrame,
    inventory: pd.DataFrame,
    semantics: Dict[str, Any],
    scope_summary: Dict[str, Any],
) -> None:
    lines: List[str] = []
    a = lines.append
    a("# Data Inventory — SupeRR Selector Season 2")
    a("")
    a(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    a("")
    a("## 1. File discovery")
    a("")
    a("Directories searched: the project directory and its parent (the delivered")
    a("data pack). No ZIP archives exist; the data pack was delivered as loose")
    a("CSV files, which are used in place and never modified.")
    a("")
    a("| file | type | directory | size (KB) | role |")
    a("|---|---|---|---|---|")
    role_by_file = {
        "ball_by_ball.csv": "ball-by-ball deliveries (primary)",
        "matches.csv": "match metadata",
        "competitions.csv": "competition registry",
        "teams.csv": "competition x team registry",
        "playing_eleven.csv": "playing XI / squad per match",
        "competition_team_player_mapping.csv": "competition x team x player mapping",
    }
    for _, r in discovery.iterrows():
        a(
            f"| {r['file']} | {r['type']} | {Path(r['dir']).name} | "
            f"{r['size_bytes'] / 1024:,.1f} | {role_by_file.get(r['file'], 'non-data artefact')} |"
        )
    a("")
    a("## 2. Tabular inventory")
    a("")
    a("| file | rows | cols | grain | primary-key candidates | duplicate rows |")
    a("|---|---|---|---|---|---|")
    for _, r in inventory.iterrows():
        a(
            f"| {r['file']} | {r['rows']:,} | {r['columns']} | {r['grain']} | "
            f"{r['primary_key_candidates']} | {r['duplicate_full_rows']} |"
        )
    a("")
    a("### Primary keys adopted")
    a("")
    a("- **players**: `player_id` (identity key for all player analysis; player")
    a("  names are display strings only and are shared by multiple ids).")
    a("- **teams**: `team_id` within `comp_id` scope for names; team ids are")
    a("  consistent across the pack.")
    a("- **deliveries**: composite key `match_id + innings + overid +")
    a("  over_ball_num` (verified unique).")
    a("- **matches**: `match_id` (verified unique).")
    a("")
    a("## 3. Verified cricket semantics (evidence-based, not assumed)")
    a("")
    ra = semantics["run_accounting"]
    a(f"- **Delivery run total** = `runs + extras1_runs + extras_2_runs`.")
    a(f"  Validated against parsed match results: {ra['n_matches_compared']} matches")
    a(f"  compared; {ra['runs_margin_matches']} runs-margin results and "
      f"{ra['wickets_margin_matches']} wickets-margin results reproduced exactly;")
    a(f"  {ra['runs_margin_mismatches']} + {ra['wickets_margin_mismatches']} mismatches.")
    lb = semantics["legal_balls"]
    a(f"- **Legal ball** = delivery whose `extras_type` is not wide / no-ball")
    a(f"  (raw label `Overstep`) / penalty / `other`. {lb['overs_with_exactly_6_legal']:,} of")
    a(f"  {lb['n_overs']:,} overs contain exactly 6 legal deliveries; overs with >6:")
    a(f"  {lb['overs_with_more_than_6_legal']:,}. Bowler-charged extras")
    a(f"  (wides/no-balls) never count as legal: {lb['bowler_charged_extras_counted_as_legal']} do.")
    a(f"- **Bowler-conceded runs** = off-bat `runs` + wides + no-ball penalties +")
    a(f"  runs conceded on them; byes/leg-byes are NOT charged to the bowler")
    a("  (standard scoring law; consistent with the extras encoding).")
    a(f"- **Bowler wickets** = Bowled, Caught, LBW, Stumped, Hit Wicket. Run-outs,")
    a(f"  Retired Out and Obstructing-the-field are wickets for the batting side")
    a("  but are not credited to the bowler.")
    bo = semantics["batting_order"]
    deviation_key = [k for k in bo if k.startswith("innings_where_arrival_order")][0]
    a(f"- **Batting order**: {bo['innings_checked']:,} innings checked; "
      f"{bo['innings_with_player_order_conflicts']} with a player holding two")
    a("  different `batting_order` values. Arrival order deviates from listed")
    a(f"  order in {bo[deviation_key]} innings (retired-hurt returns and")
    a("  promoted batters — a legitimate cricket phenomenon, informational).")
    a("")
    a("## 4. Competition & match scope (summary)")
    a("")
    for k, v in scope_summary.items():
        a(f"- {k.replace('_', ' ')}: **{v}**")
    a("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_report_md(
    path: Path, checks: List[Dict[str, Any]], semantics: Dict[str, Any]
) -> None:
    counts = _status_counts(checks)
    lines: List[str] = []
    a = lines.append
    a("# Data Quality Report — SupeRR Selector Season 2")
    a("")
    a(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    a("")
    a(f"**Checks: {len(checks)}** — pass: {counts['pass']}, warn: {counts['warn']}, "
      f"fail: {counts['fail']}, info: {counts['info']}")
    a("")
    a("## Results")
    a("")
    a("| status | check | detail |")
    a("|---|---|---|")
    for c in checks:
        detail = str(c["detail"]).replace("|", "/")
        a(f"| {c['status']} | `{c['check']}` | {detail} |")
    a("")
    a("## Interpretation notes")
    a("")
    a("- `warn` items are handled explicitly in cleaning (documented in")
    a("  `data_audit_decisions.md`); none silently drop data.")
    a("- `info` items document facts that shape the analysis (super overs, ties,")
    a("  DLS revisions, name collisions).")
    bs = semantics["bowling_types"]
    a(f"- `bowl_speed` is 0 (not recorded) for {bs['bowl_speed_zero_rows']:,} of "
      f"200,999 deliveries ({bs['bowl_speed_zero_pct']}%); it is therefore excluded")
    a("  from modeling features rather than imputed.")
    a("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    def _default(o: Any):
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    path.write_text(
        json.dumps(payload, indent=2, default=_default, ensure_ascii=False),
        encoding="utf-8",
    )
