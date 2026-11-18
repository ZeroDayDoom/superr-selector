# Data Cleaning & Match-State Report

## Joins (row-count verified)
```
join_matches: 200,999 -> 200,999 (many_to_one)
join_competitions: 200,999 -> 200,999 (many_to_one)
```

## Scope
- matches in scope: **813** of 883
- by competition class: {'IPL': 362, 'senior T20I': 451}
- deliveries in analysis table: **192,235**
- innings summarised: 1,626 (incl. 0 super-over innings, excluded from the delivery table)

## Reconstructed columns
- runs: `batter_runs`, `extras_runs`, `total_runs`, `bowler_runs_conceded` (byes/leg-byes excluded from bowler)
- wickets: `is_team_wicket`, `is_bowler_wicket`, `wicket_type`, `is_run_out`, `is_retired_out`, `dismissed_is_striker`
- state: `score_before`, `wickets_before`, `legal_balls_before`, `legal_balls_remaining`, chase state (target/RRR), par state (leave-one-innings-out curves), momentum windows (6/12/18/24), striker & bowler state, roles

## Validation results
- checks: 23, failures: **0**
- [pass] `duplicate_delivery_keys` — 0 duplicated delivery keys
- [pass] `legal_ball_counts` — max legal balls in an innings: 120; innings > 132: 0
- [pass] `innings_score_reconciliation` — 0 innings where delivery-sum != innings total
- [pass] `score_before_matches_exclusive_cumsum` — 0 deliveries where score_before != exclusive cumsum
- [pass] `wickets_before_matches_exclusive_cumsum` — 0 deliveries where wickets_before != exclusive cumsum
- [pass] `legal_balls_before_matches_exclusive_cumsum` — 0 mismatches in legal_balls_before
- [pass] `first_ball_state_zeroed` — 0 innings whose first delivery has non-zero state
- [pass] `score_before_monotonic` — 0 innings with non-monotonic score_before
- [pass] `wicket_reconciliation` — team wickets: deliveries 10392 vs innings summary 10392
- [pass] `bowler_wicket_subset_of_team_wickets` — bowler wickets 9817 <= team wickets 10392 (run-outs/retired-outs/obstructions: 575)
- [pass] `bowler_runs_identity_total` — bowler conceded 268,191 + uncharged extras 4,721 vs total 272,912
- [pass] `bowler_runs_excludes_byes_legbyes` — on byes/legbyes/penalty/other rows, bowler charged only off-bat runs (0 violations)
- [pass] `extras_reconstruction` — extras_runs == extras1 + extras_2 on every delivery
- [pass] `chase_target_correctness` — 0 non-DLS chases with target != innings-1 total + 1
- [pass] `no_negative_resources` — 0 deliveries with negative balls/wickets remaining
- [pass] `player_ids_present` — 0 deliveries missing a player id
- [pass] `no_current_outcome_in_pre_ball_state` — 0 innings where last-ball state includes its own outcome
- [pass] `striker_state_resets_per_player` — 0 batter-innings whose first delivery has non-zero balls faced / runs before
- [pass] `bowler_state_resets_per_player` — 0 bowler-innings whose first delivery has non-zero balls bowled before
- [pass] `partnership_resets_after_wicket` — 0 deliveries immediately after a wicket with non-zero partnership state
- [pass] `momentum_windows_shifted` — last6_boundaries on boundary rows only counts prior deliveries
- [pass] `dls_targets_propagated` — 22 DLS-revised chases carry revised targets
- [pass] `super_over_innings_excluded` — 0 super-over deliveries remain in analysis table