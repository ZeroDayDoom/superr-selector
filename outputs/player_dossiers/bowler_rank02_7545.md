# Player Dossier — Kuldeep Sen (Rank 2, bowler)

- **Final score:** 1.070 (z-composite; pool mean = 0)
- **Reliability:** 73 HP balls, 9 HP matches, 3 seasons; bootstrap 90% CI of HP value [0.076, 0.629]
- **Level (EB-shrunk LCB):** 0.093 (raw HP mean 0.219)
- **Pressure uplift:** raw 0.209, context-adjusted 0.226 (normal baseline 0.012)
- **Consistency:** positive-match share 100.0%, match sd 1.885
- **Downside:** CVaR5 0.031, negative-match rate 0.0%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.221 over 73 balls
- Extreme-pressure value mean: 0.018 over 44 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: LOW_WICKET_RESOURCE_CHASE (+0.474), SET_BATTER_DISMISSAL (+0.261), FINAL_OVER_PRESSURE (+0.212)
- Weaknesses: RAPID_COLLAPSE (-0.112), MIDDLE_ORDER_RESCUE (-0.138), EARLY_COLLAPSE_RESCUE (-0.279)

## Competition split (HP value mean)

- hpval_IPL: +0.221
- hpval_senior T20I: +nan

## Season trend (HP value mean by season)

- 2022: +0.143
- 2022/23: +nan
- 2023: +0.291
- 2023/24: +nan
- 2024: +0.363
- 2024/25: +nan
- 2025: +nan
- 2026: +nan

## Best / worst pressure matches
- Best: match 31245 (+5.794)
- Worst: match 14146 (+0.031)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.