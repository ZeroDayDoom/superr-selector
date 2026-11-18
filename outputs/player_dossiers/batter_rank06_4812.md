# Player Dossier — Harry Tector (Rank 6, batter)

- **Final score:** 0.679 (z-composite; pool mean = 0)
- **Reliability:** 204 HP balls, 24 HP matches, 6 seasons; bootstrap 90% CI of HP value [0.031, 0.315]
- **Level (EB-shrunk LCB):** 0.103 (raw HP mean 0.164)
- **Pressure uplift:** raw 0.222, context-adjusted 0.379 (normal baseline -0.056)
- **Consistency:** positive-match share 66.7%, match sd 0.512
- **Downside:** CVaR5 -0.421, negative-match rate 33.3%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.166 over 204 balls
- Extreme-pressure value mean: 0.266 over 61 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: NEW_BATTER_AFTER_WICKET (+0.239), WICKET_DROUGHT_BREAKTHROUGH (+0.170), RAPID_COLLAPSE (+0.158)
- Weaknesses: HIGH_RRR_CHASE (-0.117), PARTNERSHIP_REBUILD (-0.133), MATCH_ON_LINE (-0.136)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.166

## Season trend (HP value mean by season)

- 2022: +0.099
- 2022/23: +nan
- 2023: -0.270
- 2023/24: +0.376
- 2024: +0.167
- 2024/25: +nan
- 2025: +0.241
- 2026: +0.151

## Best / worst pressure matches
- Best: match 30730 (+2.208)
- Worst: match 19692 (-0.460)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.