# Player Dossier — Digvesh Rathi (Rank 8, bowler)

- **Final score:** 0.685 (z-composite; pool mean = 0)
- **Reliability:** 79 HP balls, 14 HP matches, 2 seasons; bootstrap 90% CI of HP value [-0.029, 0.421]
- **Level (EB-shrunk LCB):** 0.091 (raw HP mean 0.187)
- **Pressure uplift:** raw 0.323, context-adjusted 0.180 (normal baseline -0.134)
- **Consistency:** positive-match share 64.3%, match sd 0.501
- **Downside:** CVaR5 -0.735, negative-match rate 35.7%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.189 over 79 balls
- Extreme-pressure value mean: 0.336 over 34 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: DEATH_OVERS_PRESSURE (+0.086), DEATH_DEFENCE (+0.086), SET_BATTER_DISMISSAL (+0.050)
- Weaknesses: EARLY_COLLAPSE_RESCUE (-0.376), MIDDLE_ORDER_RESCUE (-0.441), RAPID_COLLAPSE (-0.482)

## Competition split (HP value mean)

- hpval_IPL: +0.189
- hpval_senior T20I: +nan

## Season trend (HP value mean by season)

- 2022: +nan
- 2022/23: +nan
- 2023: +nan
- 2023/24: +nan
- 2024: +nan
- 2024/25: +nan
- 2025: +0.142
- 2026: +0.273

## Best / worst pressure matches
- Best: match 40719 (+1.010)
- Worst: match 40738 (-0.735)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.