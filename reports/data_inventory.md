# Data Inventory — SupeRR Selector Season 2

_Generated: 2026-11-18T07:24:44+00:00_

## 1. File discovery

Directories searched: the project directory and its parent (the delivered
data pack). No ZIP archives exist; the data pack was delivered as loose
CSV files, which are used in place and never modified.

| file | type | directory | size (KB) | role |
|---|---|---|---|---|
| ball_by_ball.csv | csv | Data Pack | 40,109.2 | ball-by-ball deliveries (primary) |
| competition_team_player_mapping.csv | csv | Data Pack | 452.1 | competition x team x player mapping |
| competitions.csv | csv | Data Pack | 7.4 | competition registry |
| matches.csv | csv | Data Pack | 121.0 | match metadata |
| playing_eleven.csv | csv | Data Pack | 943.6 | playing XI / squad per match |
| teams.csv | csv | Data Pack | 6.8 | competition x team registry |
| Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf | pdf | Data Pack | 248.1 | non-data artefact |

## 2. Tabular inventory

| file | rows | cols | grain | primary-key candidates | duplicate rows |
|---|---|---|---|---|---|
| ball_by_ball.csv | 200,999 | 33 | one row per delivery (match x innings x over x ball-in-over) | match_id+innings+overid+over_ball_num | 0 |
| matches.csv | 883 | 22 | one row per match | match_id | 0 |
| competitions.csv | 141 | 4 | one row per competition | comp_id | 0 |
| teams.csv | 353 | 3 | one row per competition x team |  | 0 |
| playing_eleven.csv | 28,381 | 9 | one row per match x team x player (squad / playing XI) |  | 0 |
| competition_team_player_mapping.csv | 6,413 | 6 | one row per competition x team x player | comp_id+team_id+player_id | 0 |

### Primary keys adopted

- **players**: `player_id` (identity key for all player analysis; player
  names are display strings only and are shared by multiple ids).
- **teams**: `team_id` within `comp_id` scope for names; team ids are
  consistent across the pack.
- **deliveries**: composite key `match_id + innings + overid +
  over_ball_num` (verified unique).
- **matches**: `match_id` (verified unique).

## 3. Verified cricket semantics (evidence-based, not assumed)

- **Delivery run total** = `runs + extras1_runs + extras_2_runs`.
  Validated against parsed match results: 845 matches
  compared; 399 runs-margin results and 408 wickets-margin results reproduced exactly;
  0 + 1 mismatches.
- **Legal ball** = delivery whose `extras_type` is not wide / no-ball
  (raw label `Overstep`) / penalty / `other`. 31,756 of
  32,350 overs contain exactly 6 legal deliveries; overs with >6:
  1. Bowler-charged extras
  (wides/no-balls) never count as legal: 0 do.
- **Bowler-conceded runs** = off-bat `runs` + wides + no-ball penalties +
  runs conceded on them; byes/leg-byes are NOT charged to the bowler
  (standard scoring law; consistent with the extras encoding).
- **Bowler wickets** = Bowled, Caught, LBW, Stumped, Hit Wicket. Run-outs,
  Retired Out and Obstructing-the-field are wickets for the batting side
  but are not credited to the bowler.
- **Batting order**: 1,724 innings checked; 0 with a player holding two
  different `batting_order` values. Arrival order deviates from listed
  order in 360 innings (retired-hurt returns and
  promoted batters — a legitimate cricket phenomenon, informational).

## 4. Competition & match scope (summary)

- total matches: **883**
- included: **813**
- excluded: **70**
- excluded no ball data: **16**
- excluded no result: **40**
- excluded competition scope: **30**
- tied matches kept: **1**
- dls revised kept: **24**
- super over matches flagged: **6**
