# Final Project Audit

_Generated 2026-11-18 10:49 UTC — 62 checks, 0 FAIL._

## Step 2

- [PASS] **no duplicate delivery keys** — 192,235 rows
- [PASS] **score_before = exclusive cumsum** — 
- [PASS] **legal balls per innings <= 132** — 
- [PASS] **innings totals reconcile** — 
- [PASS] **bowler runs exclude byes/legbyes** — 
- [PASS] **wickets reconcile (deliveries = innings summary)** — 
- [PASS] **bowler wickets <= team wickets** — 
- [PASS] **delivery players in playing XI** — 0 exceptions
- [PASS] **no super-over innings** — 
- [PASS] **DLS rows carry revised flags** — 
- [PASS] **chases have targets** — 
- [PASS] **no negative resources** — 
## Step 3

- [PASS] **all registry features pre-delivery** — []
- [PASS] **dynamic perturbation audit clean** — 150 deliveries, 128 features
- [PASS] **holdout (2026) evaluation present** — 
- [PASS] **selection declared on validation** — 
## Step 4

- [PASS] **chronological splits documented** — 
- [PASS] **no random ball split mentioned as method** — 
- [PASS] **deep models evaluated (history saved)** — 
## Step 5

- [PASS] **rankings_batters.csv: 10 rows, ranks 1..10** — 
- [PASS] **rankings_batters.csv: no duplicates** — 
- [PASS] **rankings_batters.csv: ids in supplied mapping** — 
- [PASS] **rankings_batters.csv: names match mapping** — []
- [PASS] **rankings_batters.csv: all selected are moderate-eligible** — 
- [PASS] **batter: rating matches formula** — 
- [PASS] **batter: level_lcb = shrunk - 0.85*SE** — 
- [PASS] **rankings_bowlers.csv: 10 rows, ranks 1..10** — 
- [PASS] **rankings_bowlers.csv: no duplicates** — 
- [PASS] **rankings_bowlers.csv: ids in supplied mapping** — 
- [PASS] **rankings_bowlers.csv: names match mapping** — []
- [PASS] **rankings_bowlers.csv: all selected are moderate-eligible** — 
- [PASS] **bowler: rating matches formula** — 
- [PASS] **bowler: level_lcb = shrunk - 0.85*SE** — 
## Step 6

- [PASS] **20 dossiers (10 batters + 10 bowlers)** — 20 found
- [PASS] **dossier/ranking statistics consistent with metrics** — []
- [PASS] **dossier match ids exist in data** — 
## Step 7

- [PASS] **temporal validation (batters)** — 
- [PASS] **split-half reliability** — 
- [PASS] **threshold sensitivity** — 
- [PASS] **weight sensitivity** — 
- [PASS] **ablation study** — 
- [PASS] **role analysis** — 
- [PASS] **competition analysis** — 
- [PASS] **pressure/model checks** — 
- [PASS] **edge tests** — 
- [PASS] **bootstrap completed** — 338 players
## Step 8

- [PASS] **all dashboard pages execute** — 
## Step 9

- [PASS] **PDF renders** — 10 pages
- [PASS] **all 20 ranked names appear in PDF** — []
- [PASS] **no placeholder text** — 
## Step 10

- [PASS] **final_methodology.pdf** — reports\final_methodology.pdf
- [PASS] **final_methodology.md** — reports\final_methodology.md
- [PASS] **final_methodology.html** — reports\final_methodology.html
- [PASS] **executive_summary.md** — reports\executive_summary.md
- [PASS] **submission_top10.csv** — outputs\submission_top10.csv
- [PASS] **submission_top10.xlsx** — outputs\submission_top10.xlsx
- [PASS] **submission_entry_sheet.txt** — outputs\submission_entry_sheet.txt
- [PASS] **rankings_batters.csv** — outputs\rankings_batters.csv
- [PASS] **rankings_bowlers.csv** — outputs\rankings_bowlers.csv
- [PASS] **all_player_metrics.csv** — outputs\all_player_metrics.csv
- [PASS] **README.md** — README.md
- [PASS] **pyproject.toml** — pyproject.toml

## Submission readiness

**READY** if all checks pass; otherwise the FAIL lines above list exactly what remains.