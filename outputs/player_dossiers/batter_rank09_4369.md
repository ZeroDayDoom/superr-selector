# Player Dossier — Shai Hope (Rank 9, batter)

- **Final score:** 0.672 (z-composite; pool mean = 0)
- **Reliability:** 179 HP balls, 27 HP matches, 5 seasons; bootstrap 90% CI of HP value [0.061, 0.356]
- **Level (EB-shrunk LCB):** 0.153 (raw HP mean 0.216)
- **Pressure uplift:** raw 0.151, context-adjusted 0.433 (normal baseline 0.067)
- **Consistency:** positive-match share 74.1%, match sd 1.137
- **Downside:** CVaR5 -2.812, negative-match rate 25.9%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.218 over 179 balls
- Extreme-pressure value mean: 0.218 over 84 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: NEW_BATTER_AFTER_WICKET (+0.503), FINAL_OVER_PRESSURE (+0.331), WICKET_DROUGHT_BREAKTHROUGH (+0.301)
- Weaknesses: MATCH_ON_LINE (-0.064), BOUNDARY_DROUGHT (-0.114), MIDDLE_ORDER_RESCUE (-0.128)

## Competition split (HP value mean)

- hpval_IPL: +0.496
- hpval_senior T20I: +0.213

## Season trend (HP value mean by season)

- 2022: +0.017
- 2022/23: +nan
- 2023: +0.428
- 2023/24: +nan
- 2024: +0.335
- 2024/25: +nan
- 2025: +0.170
- 2026: +0.248

## Best / worst pressure matches
- Best: match 28065 (+2.094)
- Worst: match 50505 (-4.513)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.