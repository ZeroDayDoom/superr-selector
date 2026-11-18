# Player Dossier — Nitish Reddy (Rank 2, batter)

- **Final score:** 0.756 (z-composite; pool mean = 0)
- **Reliability:** 116 HP balls, 11 HP matches, 3 seasons; bootstrap 90% CI of HP value [-0.012, 0.221]
- **Level (EB-shrunk LCB):** 0.056 (raw HP mean 0.107)
- **Pressure uplift:** raw 0.070, context-adjusted 0.324 (normal baseline 0.039)
- **Consistency:** positive-match share 90.9%, match sd 0.216
- **Downside:** CVaR5 -0.347, negative-match rate 9.1%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.110 over 116 balls
- Extreme-pressure value mean: 0.092 over 61 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: NEW_BATTER_AFTER_WICKET (+0.395), MATCH_ON_LINE (+0.319), RAPID_COLLAPSE (+0.251)
- Weaknesses: SET_BATTER_DISMISSAL (-0.106), FINAL_OVER_PRESSURE (-0.135), BOUNDARY_DROUGHT (-0.149)

## Competition split (HP value mean)

- hpval_IPL: +0.088
- hpval_senior T20I: +0.342

## Season trend (HP value mean by season)

- 2022: +nan
- 2022/23: +nan
- 2023: +nan
- 2023/24: +nan
- 2024: +0.169
- 2024/25: +nan
- 2025: -0.062
- 2026: +0.177

## Best / worst pressure matches
- Best: match 40734 (+0.383)
- Worst: match 40713 (-0.347)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.