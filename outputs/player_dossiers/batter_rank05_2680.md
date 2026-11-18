# Player Dossier — Nurul Hasan (Rank 5, batter)

- **Final score:** 0.716 (z-composite; pool mean = 0)
- **Reliability:** 93 HP balls, 14 HP matches, 4 seasons; bootstrap 90% CI of HP value [-0.040, 0.365]
- **Level (EB-shrunk LCB):** 0.100 (raw HP mean 0.188)
- **Pressure uplift:** raw 0.232, context-adjusted 0.404 (normal baseline -0.041)
- **Consistency:** positive-match share 85.7%, match sd 0.737
- **Downside:** CVaR5 -1.915, negative-match rate 14.3%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.191 over 93 balls
- Extreme-pressure value mean: 0.075 over 41 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: LOW_WICKET_RESOURCE_CHASE (+0.647), EARLY_COLLAPSE_RESCUE (+0.397), WICKET_DROUGHT_BREAKTHROUGH (+0.291)
- Weaknesses: MIDDLE_ORDER_RESCUE (-0.033), HIGH_LEVERAGE (-0.138), MATCH_ON_LINE (-0.163)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.191

## Season trend (HP value mean by season)

- 2022: +0.357
- 2022/23: +0.440
- 2023: +nan
- 2023/24: +nan
- 2024: +nan
- 2024/25: +nan
- 2025: +0.115
- 2026: -0.503

## Best / worst pressure matches
- Best: match 17739 (+0.950)
- Worst: match 51384 (-1.915)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.