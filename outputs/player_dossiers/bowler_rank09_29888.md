# Player Dossier — Harshit Rana (Rank 9, bowler)

- **Final score:** 0.684 (z-composite; pool mean = 0)
- **Reliability:** 189 HP balls, 25 HP matches, 5 seasons; bootstrap 90% CI of HP value [-0.004, 0.718]
- **Level (EB-shrunk LCB):** 0.125 (raw HP mean 0.286)
- **Pressure uplift:** raw 0.249, context-adjusted 0.295 (normal baseline 0.038)
- **Consistency:** positive-match share 52.0%, match sd 1.119
- **Downside:** CVaR5 -1.428, negative-match rate 48.0%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.287 over 189 balls
- Extreme-pressure value mean: 0.336 over 75 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: LOW_WICKET_RESOURCE_CHASE (+0.213), EARLY_COLLAPSE_RESCUE (+0.123), HIGH_LEVERAGE (+0.102)
- Weaknesses: NEW_BATTER_AFTER_WICKET (-0.193), BOUNDARY_DROUGHT (-0.307), RAPID_COLLAPSE (-0.806)

## Competition split (HP value mean)

- hpval_IPL: +0.356
- hpval_senior T20I: +0.181

## Season trend (HP value mean by season)

- 2022: -1.102
- 2022/23: +nan
- 2023: +0.192
- 2023/24: +nan
- 2024: +0.059
- 2024/25: +nan
- 2025: +0.486
- 2026: -0.326

## Best / worst pressure matches
- Best: match 40750 (+4.199)
- Worst: match 48592 (-1.565)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.