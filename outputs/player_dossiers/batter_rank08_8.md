# Player Dossier — Mahendra Singh Dhoni (Rank 8, batter)

- **Final score:** 0.673 (z-composite; pool mean = 0)
- **Reliability:** 85 HP balls, 17 HP matches, 4 seasons; bootstrap 90% CI of HP value [-0.001, 0.265]
- **Level (EB-shrunk LCB):** 0.091 (raw HP mean 0.149)
- **Pressure uplift:** raw 0.080, context-adjusted 0.368 (normal baseline 0.072)
- **Consistency:** positive-match share 82.4%, match sd 0.319
- **Downside:** CVaR5 -0.429, negative-match rate 17.6%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.153 over 85 balls
- Extreme-pressure value mean: -0.000 over 28 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: MATCH_ON_LINE (+0.470), RAPID_COLLAPSE (+0.366), MIDDLE_ORDER_RESCUE (+0.315)
- Weaknesses: DEATH_DEFENCE (+0.021), FINAL_OVER_PRESSURE (+0.014), SET_BATTER_DISMISSAL (+0.011)

## Competition split (HP value mean)

- hpval_IPL: +0.153
- hpval_senior T20I: +nan

## Season trend (HP value mean by season)

- 2022: +0.220
- 2022/23: +nan
- 2023: +0.235
- 2023/24: +nan
- 2024: +0.413
- 2024/25: +nan
- 2025: +0.009
- 2026: +nan

## Best / worst pressure matches
- Best: match 31267 (+0.768)
- Worst: match 14131 (-0.429)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.