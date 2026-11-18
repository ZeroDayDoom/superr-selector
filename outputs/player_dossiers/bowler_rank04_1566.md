# Player Dossier — Mohammed Shami (Rank 4, bowler)

- **Final score:** 0.904 (z-composite; pool mean = 0)
- **Reliability:** 234 HP balls, 41 HP matches, 4 seasons; bootstrap 90% CI of HP value [0.098, 0.489]
- **Level (EB-shrunk LCB):** 0.204 (raw HP mean 0.291)
- **Pressure uplift:** raw 0.278, context-adjusted 0.298 (normal baseline 0.014)
- **Consistency:** positive-match share 58.5%, match sd 1.101
- **Downside:** CVaR5 -1.043, negative-match rate 41.5%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.292 over 234 balls
- Extreme-pressure value mean: 0.537 over 120 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: LOW_WICKET_RESOURCE_CHASE (+1.070), NEW_BATTER_AFTER_WICKET (+0.425), MATCH_ON_LINE (+0.419)
- Weaknesses: DEATH_DEFENCE (-0.050), BOUNDARY_DROUGHT (-0.104), WICKET_DROUGHT_BREAKTHROUGH (-0.199)

## Competition split (HP value mean)

- hpval_IPL: +0.302
- hpval_senior T20I: -1.844

## Season trend (HP value mean by season)

- 2022: +0.237
- 2022/23: +nan
- 2023: +0.286
- 2023/24: +nan
- 2024: +nan
- 2024/25: +nan
- 2025: +0.100
- 2026: +0.474

## Best / worst pressure matches
- Best: match 50389 (+4.860)
- Worst: match 40038 (-1.844)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.