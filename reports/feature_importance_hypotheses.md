# Feature Importance Hypotheses (pre-registered before modelling)

Written before any model is trained (Phase 5). These are testable priors,
not results — each will be checked against the trained models and reported.

## Expected dominant predictors of pressure (per the methodology)

| Rank | Feature family | Hypothesis | Test in Phase 5 |
|---|---|---|---|
| 1 | `match_balance_proxy` / WP balance | Balance dominates pressure perception; deliveries near 50/50 carry the most variance | Permutation importance in WP/leverage models |
| 2 | `required_run_rate`, `rrr_crr_gap` | Chase steepness is the single strongest situational stressor in innings 2 | GBM gain shares; partial dependence |
| 3 | `legal_balls_remaining`, `innings_phase` | Death-over proximity amplifies every other stressor | Interaction analysis |
| 4 | `wickets_in_hand`, `collapse_severity` | Resource scarcity and recent collapse raise wicket probability | Calibration by stratum |
| 5 | `bowler_recent_economy`, `bowler_recent_wicket_rate` | Bowler form is modestly predictive but heavily shrunk | Shrinkage sensitivity |
| 6 | `batter_is_new` vs `batter_is_set` | New batters dismiss ~1.5-2x the rate of set batters in identical states | Rate ratio with CI |
| 7 | `score_vs_par` | Batting-first pressure tracks score vs par more than raw run rate | Model comparison |
| 8 | `matchup_difficulty_index` | Hand x class matchups move expected runs by ~5-10% | Effect size vs bootstrap CI |
| 9 | `is_knockout` | Knockouts raise pressure independent of state (crowd/stakes not in state features) | Fixed-effect contrast |
| 10 | `bowling_team_wicket_drought` | Drought length mildly raises bowling pressure but is largely captured by setness | Partial correlation |

## Hypotheses expected NOT to matter much (pre-registered)

- `bowl_speed` — excluded (38.7% unrecorded).
- `delivery_line`, `over_or_around_wicket` — marginal once phase and class are known.
- Home/away — expected small; T20I neutral venues blur the signal.

## Validation gates these feed

- Pressure index (Phase 5) uses balance/severity components validated in the
  methodology document; scenario labels feed scenario-diversity metrics in
  the ratings and dossiers.
