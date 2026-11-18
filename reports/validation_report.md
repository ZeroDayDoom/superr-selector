# Validation Report

_Generated 2026-11-18T10:43:27.032365+00:00_

## V1 Temporal persistence (era-to-era HP value)

### batters
- `rank_2022-2023_eval_2024-2026`: Spearman 0.038, top-15 overlap 1/15 (n=84)
- `rank_2022-2024_eval_2025-2026`: Spearman 0.032, top-15 overlap 1/15 (n=74)
- `rank_2022-2025_eval_2026-2026`: Spearman 0.212, top-15 overlap 1/15 (n=34)

### bowlers
- `rank_2022-2023_eval_2024-2026`: Spearman -0.042, top-15 overlap 2/15 (n=81)
- `rank_2022-2024_eval_2025-2026`: Spearman -0.041, top-15 overlap 1/15 (n=75)
- `rank_2022-2025_eval_2026-2026`: Spearman -0.062, top-15 overlap 1/15 (n=23)

## V3 Split-half reliability

- Batters: {'n_splits': 100, 'spearman_mean': 0.093, 'kendall_mean': 0.064, 'score_corr_mean': 0.054, 'top10_overlap_mean': 0.82, 'top20_overlap_mean': 2.42}
- Bowlers: {'n_splits': 100, 'spearman_mean': 0.114, 'kendall_mean': 0.078, 'score_corr_mean': 0.135, 'top10_overlap_mean': 1.27, 'top20_overlap_mean': 4.39}

## V4 Season trends (top-200 batters by sample)
- {'insufficient_seasons': 56, 'improving': 31, 'single_season_dominance': 30, 'declining': 26, 'consistent': 26}

## V5/V6/V8/V9/V10 rank stability vs primary (batters)

| variant | Spearman | top-10 overlap |
|---|---|---|
| threshold_hp_top10 | 0.579 | 3 |
| threshold_hp_top15 | 0.714 | 4 |
| threshold_hp_top20 | 0.847 | 5 |
| threshold_hp_top25 | 0.797 | 5 |
| threshold_hp_top30 | 0.727 | 4 |
| PRESSURE_MODEL_A | 0.294 | 1 |
| PRESSURE_MODEL_B | 0.6 | 2 |
| PRESSURE_MODEL_C | 0.55 | 2 |
| eligibility_strict | 1.0 | 10 |
| eligibility_moderate | 1.0 | 10 |
| eligibility_inclusive | 1.0 | 10 |
| shrinkage_shrunk_k_half | 0.942 | 1 |
| shrinkage_shrunk_k | 0.941 | 1 |
| shrinkage_shrunk_k_double | 0.939 | 1 |
| shrinkage_raw_mean | 0.943 | 1 |
| weights_level_heavy | 0.996 | 10 |
| weights_balanced | 0.998 | 10 |
| weights_uplift_heavy | 0.994 | 9 |
| weights_risk_averse | 0.962 | 10 |

## V12 Ablation (batters)

| removed component | Spearman vs primary |
|---|---|
| ablate_level_lcb | 0.98 |
| ablate_consistency | 0.932 |
| ablate_uplift | 0.997 |
| ablate_volume | 0.957 |
| ablate_downside | 0.979 |

## V7 Role checks

- Batters: {'role_means': {'finisher': 0.181, 'middle': 0.15, 'opener': 0.015, 'top': -0.085, 'lower': -0.129}, 'role_sizes': {'finisher': 93, 'middle': 96, 'opener': 110, 'top': 141, 'lower': 161}, 'max_mean_gap': 0.309, 'top10_composition': {'lower': 7, 'finisher': 2, 'opener': 1}}
- Role normalisation vs raw: 0.966

## V13 Model checks

- death_share_of_hp_deliveries: 0.402
- death_share_of_all_deliveries: 0.222
- pressure_not_synonymous_with_death: False
- hp_vs_rest_wicket_prob: [0.071, 0.0498]
- hp_vs_rest_abs_wpa: [0.0311, 0.0139]
- non_death_hp_vs_rest_wicket_prob: [0.0538, 0.0444]
- expected_runs_residual_mean: -0.0259
- expected_runs_residual_sd: 0.6255
- max_single_match_share_of_player_hp_impact: {'mean': 0.579, 'p95': 1.0, 'max': 1.0}

## V14 Edge tests

- `target_reached`: {'value': -0.0893237743547659, 'note': 'chase complete; pressure should be LOW'}
- `zero_balls_remaining`: {'value': 2.9476536955297896, 'note': 'terminal state'}
- `all_out`: {'value': -0.0893237743547659}
- `rrr_25`: {'value': -0.0893237743547659}
- `wide`: {'bowler_conceded': 1.0, 'is_legal': False, 'total': 1.0}
- `no_ball`: {'bowler_conceded': 1.0, 'is_legal': False, 'total': 1.0}
- `byes`: {'bowler_conceded': 0.0, 'is_legal': True, 'total': 2.0}
- `legbyes`: {'bowler_conceded': 0.0, 'is_legal': True, 'total': 4.0}
- `run_out`: {'is_bowler_wicket': False, 'is_team_wicket': True, 'dismissed_is_striker': True}
- `missing_ids`: {'lam_global_used': 9.827805989449828}