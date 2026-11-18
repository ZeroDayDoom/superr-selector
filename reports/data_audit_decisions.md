# Data Audit Decisions — SupeRR Selector Season 2

_Generated alongside the audit (see `data_inventory.md`, `data_quality_report.md`,
`data_quality_checks.json`, `competition_audit.csv`, `match_exclusion_audit.csv`).
Every decision below is reproducible from `scripts/run_audit.py`._

## 1. Competition inclusion / exclusion

| Class | Competitions | Matches | Deliveries | Decision |
|---|---|---|---|---|
| IPL | IPL 2022, 2023, 2024, 2025, 2026 | 371 | 87,687 | **Included** |
| Senior T20I | 134 bilateral/series competitions | 482 | 106,327 | **Included** |
| A-team / development | Mens T20 Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025 | 30 | 6,985 | **Excluded** |

Rationale: the challenge asks for IPL and T20I pressure analysis. The two
A-team tournaments are development squads ("A" teams), a materially different
skill population from senior international/IPL cricket. They are excluded from
rankings by default; a sensitivity re-run that includes them is supported by
flipping `cleaning.exclude.a_team_competitions` in `configs/config.yaml`. No
women's competitions are present in the pack.

No IPL/T20I weighting factor is hard-coded; the model controls for competition
context explicitly (see the methodology report, Phase 4).

## 2. Match exclusions (70 of 883)

All decisions are recorded row-by-row in `match_exclusion_audit.csv`:

- **16 matches**: present in `matches.csv` but absent from `ball_by_ball.csv`
  (mostly abandoned fixtures). Excluded — no deliveries to analyse.
- **40 matches**: `No Result` (21) or `Match Abandoned` (19). Excluded —
  pressure is defined against an outcome (win probability); without a
  resolution the state-to-outcome mapping is undefined. Their delivered balls
  are also excluded from model training to avoid contaminating the
  win-probability model with truncated innings.
- **14 matches**: both of the above overlap (abandoned/no-result matches that
  also lack ball data), counted once.

**Included: 813 matches** covering 1724 ordinary innings.

## 3. Super overs

6 matches were tied and went to a super over (38 deliveries in innings 3/4).
The tied matches themselves are **kept** (their innings 1–2 are ordinary
cricket); the super-over innings are **dropped** from all delivery-level
analysis: a single-over shootout is a different pressure regime (no resource
management, free license to swing) and would distort innings-phase baselines.

## 4. DLS / revised matches

25 matches carry revised-over fields; 23 carry a revised target. Handling:

- **Kept and specially handled**, never silently dropped.
- Chase-state features (required runs, required rate, balls remaining) are
  computed against the **revised** target and revised overs for innings 2.
- Where the revised target is missing but the result string marks DLS
  (2 matches), the match is retained; the chase state falls back to raw
  target values and this limitation is documented.
- 19 DLS-method results were audited: 6 verified exactly against the recorded
  revised target, 2 are unverifiable from the pack (DLS par score not
  supplied), and the remainder reproduce the raw score difference.

## 5. Abandoned / no-result matches

Excluded entirely (see §2). This is a *documented* exclusion — the full list
lives in `match_exclusion_audit.csv`.

## 6. Legal-ball definition

**A legal delivery is one whose `extras_type` is not a wide, no-ball (raw
label `Overstep`), penalty, or `other`.** Evidence:

- 31,756 of 32,350 overs contain exactly 6 such deliveries; only 1 over has 7
  (documented single anomaly); bowler-charged extras never count as legal
  (0 of 8,290 do).
- Wides/no-balls consume `over_ball_num` slots (values up to 13) but do not
  advance the legal count; `innings_ball_num` is a raw row counter, also not
  a legal-ball counter.

## 7. Delivery run accounting (verified, not assumed)

**Total runs on a delivery = `runs` + `extras1_runs` + `extras_2_runs`.**

- `runs` is off-the-bat runs credited to the striker (range 0–6).
- `extras1_runs` is the value of the extra itself (1 for wide/no-ball; N for
  byes/leg-byes); `extras_2_runs` holds additional runs on the same delivery
  (e.g. 4 byes off a no-ball).
- Validation: team innings totals built this way reproduce **807 of 810
  non-DLS result margins exactly** (399 runs-margin + 408 wickets-margin) and
  all 6 DLS margins checkable against recorded revised targets.

**Bowler-conceded runs** = off-bat runs + wides + no-ball penalties + runs
conceded on those extras. Byes and leg-byes are **not** charged to the bowler
(standard scoring laws; consistent with the extras encoding).

## 8. Bowler-wicket definition

**Bowler wickets** = Bowled, Caught, LBW, Stumped, Hit Wicket.
**Wickets but NOT bowler wickets** = Run-out (600), Retired Out (6),
Obstructing the field (3, with a raw spelling variant `Obstracting the field`
which is normalised). Retired Hurt does not appear in the pack.

## 9. Player-identity rules

- **`player_id` is the sole identity key** for every join and every
  player-level aggregation. Player names are display strings only.
- The mapping file contains shared names across different `player_id`s
  (e.g. initials variants); the reverse (one id → many names) occurs **0**
  times in deliveries, **0** times in the mapping. Name↔id bijection is
  verified per check in `data_quality_checks.json`.
- Team identity likewise uses `team_id`; `team_name` is display-only and
  competition-scoped.

## 10. Known unresolved / documented data issues

| Issue | Scope | Handling |
|---|---|---|
| Result string of match **20190** says "Won by 6 wickets" but ball data shows the chase lost 5 wickets (real-world result: 5 wickets) | 1 match | Non-blocking. Rankings and states are computed from ball-by-ball data, never from result strings. Documented as a result-string error. |
| 1 over with 7 legal deliveries | 1 over | Left in data; impact is negligible and it is flagged, not hidden. |
| `bowl_speed` = 0 for 38.7% of deliveries | 77,861 rows | Treated as *not recorded* (not a real speed). Excluded from features; no imputation. |
| `stroke` missing on 6,158 deliveries; 1 wicket row without `dismissed_player_id` | sparse | Documented; stroke features (if used) must handle missingness explicitly. |
| 2 DLS results unverifiable from the pack (par score not supplied) | 2 matches | Retained with fallback chase state; flagged in match audit. |
| `first/second_inn_revised_overs` appear to record **balls**, not overs (e.g. 102) | 25 matches | Interpreted as revised supply in balls; chase-state features use `revised_target` primarily, which is unambiguous. |
| 3 matches in `matches.csv` have no playing-XI rows | 3 matches | Playing-XI features are optional enrichment; absence handled explicitly. |
| 40 duplicated (match, team, player) rows in `playing_eleven.csv` (same player, several preferred `batting_order` values) | 40 rows | Ball-level `batting_order` is authoritative (verified conflict-free); the PE order column is not used for sequencing. Documented, not silently deduplicated. |

## 11. Verdict

**The dataset is safe to proceed.** 36/36 quality checks pass with zero FAIL
statuses; the only blocking-class anomaly found (match 20190) is a single
free-text field that the pipeline never depends on. All exclusions,
super-over, DLS, and identity decisions above are encoded in
`configs/config.yaml` and `src/superr/data/` so that re-running
`python scripts/run_audit.py` regenerates every statement in these documents.
