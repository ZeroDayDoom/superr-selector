# Player Dossier — Fareed Malik (Rank 1, bowler)

- **Final score:** 1.270 (z-composite; pool mean = 0)
- **Reliability:** 79 HP balls, 14 HP matches, 4 seasons; bootstrap 90% CI of HP value [0.050, 1.014]
- **Level (EB-shrunk LCB):** 0.329 (raw HP mean 0.534)
- **Pressure uplift:** raw 0.564, context-adjusted 0.542 (normal baseline -0.024)
- **Consistency:** positive-match share 78.6%, match sd 0.747
- **Downside:** CVaR5 -0.606, negative-match rate 21.4%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.540 over 79 balls
- Extreme-pressure value mean: 0.556 over 29 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: LOW_WICKET_RESOURCE_CHASE (+0.612), RAPID_COLLAPSE (+0.402), HIGH_LEVERAGE (+0.294)
- Weaknesses: MATCH_ON_LINE (-0.098), MIDDLE_ORDER_RESCUE (-0.252), BOUNDARY_DROUGHT (-0.254)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.540

## Season trend (HP value mean by season)

- 2022: +0.353
- 2022/23: +nan
- 2023: +0.470
- 2023/24: +nan
- 2024: +0.795
- 2024/25: +nan
- 2025: +0.500
- 2026: +nan

## Best / worst pressure matches
- Best: match 45851 (+1.995)
- Worst: match 45856 (-0.606)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.