# Player Dossier — Mohammad Rizwan (Rank 1, batter)

- **Final score:** 0.774 (z-composite; pool mean = 0)
- **Reliability:** 467 HP balls, 28 HP matches, 4 seasons; bootstrap 90% CI of HP value [0.073, 0.258]
- **Level (EB-shrunk LCB):** 0.132 (raw HP mean 0.170)
- **Pressure uplift:** raw -0.009, context-adjusted 0.387 (normal baseline 0.180)
- **Consistency:** positive-match share 78.6%, match sd 1.128
- **Downside:** CVaR5 -3.548, negative-match rate 21.4%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.171 over 467 balls
- Extreme-pressure value mean: 0.140 over 297 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: EARLY_COLLAPSE_RESCUE (+0.290), BOUNDARY_DROUGHT (+0.251), MIDDLE_ORDER_RESCUE (+0.233)
- Weaknesses: FINAL_OVER_PRESSURE (-0.118), NEW_BATTER_AFTER_WICKET (-0.152), LOW_WICKET_RESOURCE_CHASE (-0.275)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.171

## Season trend (HP value mean by season)

- 2022: +0.226
- 2022/23: -0.002
- 2023: +0.472
- 2023/24: +nan
- 2024: +0.162
- 2024/25: +nan
- 2025: +nan
- 2026: +nan

## Best / worst pressure matches
- Best: match 28964 (+1.593)
- Worst: match 17740 (-3.687)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.