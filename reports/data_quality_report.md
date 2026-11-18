# Data Quality Report — SupeRR Selector Season 2

_Generated: 2026-11-18T07:24:44+00:00_

**Checks: 37** — pass: 25, warn: 6, fail: 0, info: 6

## Results

| status | check | detail |
|---|---|---|
| pass | `duplicate_full_rows_ball_by_ball` | 0 fully duplicated delivery rows |
| pass | `duplicate_delivery_keys` | 0 duplicated (match_id, innings, overid, over_ball_num) keys |
| pass | `duplicate_match_ids` | 0 duplicated match ids in matches.csv |
| pass | `ball_match_ids_missing_from_matches` | 0 ball match_ids not found in matches.csv |
| warn | `matches_without_ball_data` | 16 matches in matches.csv have no deliveries (typically abandoned/no-result matches) |
| warn | `matches_without_playing_xi` | 3 matches have no playing-XI rows |
| pass | `null_player_ids_in_deliveries` | 0 deliveries with a null striker/non-striker/bowler id |
| pass | `delivery_players_present_in_mapping` | 0 (comp_id, player_id) delivery pairs absent from the competition-team-player mapping |
| pass | `invalid_innings_numbers` | 0 deliveries with innings outside 1-4 (3/4 = super over) |
| pass | `invalid_over_numbers` | 0 innings start before over 1 |
| pass | `invalid_delivery_numbers` | all over_ball_num within 1-13 (max reflects extended overs of extras) |
| pass | `ball_sequence_monotonic` | 0 innings with non-monotonic innings_ball_num after sorting |
| pass | `impossible_runs_off_bat` | off-bat 'runs' always within 0-6 (extras recorded separately) |
| pass | `negative_extras` | 0 deliveries with negative extras |
| pass | `wicket_without_dismissal_type` | 0 is_wicket rows with null dismissal_type |
| pass | `dismissal_type_without_wicket_flag` | 0 dismissal_type rows where is_wicket is false |
| warn | `wicket_without_dismissed_player` | 1 wickets with null dismissed_player_id |
| pass | `runout_dismissed_player_in_pair` | 0 run-outs where dismissed_player_id is neither striker nor non-striker (n run-outs = 600) |
| pass | `dismissed_player_matches_batting_pair` | 0 wickets where dismissed_player_id is not one of the two batters at the crease |
| pass | `wickets_per_innings_le_10` | 0 innings with more than 10 wickets |
| warn | `result_strings_unparsed` | 25 result strings could not be parsed by known patterns |
| pass | `decisive_result_without_win_team` | 0 decisive results with null win_team_id |
| info | `tied_matches` | 1 tied matches (super-over deliveries appear as innings 3/4) |
| info | `no_result_or_abandoned_matches` | 40 matches with no result (19 abandoned, 21 no-result) |
| info | `dls_revised_matches` | 34 matches with revised overs; 23 with a revised target |
| pass | `name_to_multiple_ids_striker` | 0 striker_id values map to multiple names |
| pass | `name_to_multiple_ids_non_striker` | 0 non_striker_id values map to multiple names |
| pass | `name_to_multiple_ids_bowler` | 0 bowler_id values map to multiple names |
| pass | `mapping_id_to_multiple_names` | 0 player_id values map to multiple names in the mapping file |
| info | `shared_player_names_in_mapping` | 818 player names are shared by more than one player_id (names are NOT unique identifiers; player_id is the identity key) |
| pass | `team_id_to_multiple_names` | 0 team_id values with multiple names in teams.csv |
| info | `playing_xi_is_played_false_rows` | 8667 playing-XI rows are squad members who did not play (their captain/wk flags are null) |
| info | `playing_xi_duplicate_player_rows` | 40 (match, team, player) rows are duplicated with different batting_order values (preferred positions varying across the tournament). Ball-level batting_order is authoritative; the PE batting_order column is not used for ordering. |
| pass | `playing_xi_captain_flag_null_for_players` | 0 played rows with null is_captain |
| warn | `delivery_players_in_playing_xi` | 6 (match, player) delivery pairs with no is_played=True row |
| pass | `delivery_timestamps_parseable` | 0 unparseable delivery timestamps; range 2022-01-23 00:00:00+00:00 to 2026-11-01 16:50:46+00:00 |
| warn | `delivery_date_matches_match_date` | 10430 deliveries whose date differs from match_date (timezones: IST timestamps vs UTC dates; tolerance of +-1 day is expected) |

## Interpretation notes

- `warn` items are handled explicitly in cleaning (documented in
  `data_audit_decisions.md`); none silently drop data.
- `info` items document facts that shape the analysis (super overs, ties,
  DLS revisions, name collisions).
- `bowl_speed` is 0 (not recorded) for 77,861 of 200,999 deliveries (38.74%); it is therefore excluded
  from modeling features rather than imputed.
