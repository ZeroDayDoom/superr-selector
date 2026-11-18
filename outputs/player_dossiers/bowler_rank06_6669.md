# Player Dossier — Mohsin Khan (Rank 6, bowler)

- **Final score:** 0.744 (z-composite; pool mean = 0)
- **Reliability:** 161 HP balls, 23 HP matches, 4 seasons; bootstrap 90% CI of HP value [-0.003, 0.417]
- **Level (EB-shrunk LCB):** 0.109 (raw HP mean 0.201)
- **Pressure uplift:** raw 0.164, context-adjusted 0.209 (normal baseline 0.037)
- **Consistency:** positive-match share 56.5%, match sd 0.648
- **Downside:** CVaR5 -0.762, negative-match rate 43.5%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.201 over 161 balls
- Extreme-pressure value mean: 0.202 over 78 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: MIDDLE_ORDER_RESCUE (+0.498), NEW_BATTER_AFTER_WICKET (+0.362), BOUNDARY_DROUGHT (+0.268)
- Weaknesses: FINAL_OVER_PRESSURE (-0.261), EARLY_COLLAPSE_RESCUE (-0.318), RAPID_COLLAPSE (-0.895)

## Competition split (HP value mean)

- hpval_IPL: +0.201
- hpval_senior T20I: +nan

## Season trend (HP value mean by season)

- 2022: +0.179
- 2022/23: +nan
- 2023: +0.384
- 2023/24: +nan
- 2024: +0.192
- 2024/25: +nan
- 2025: +nan
- 2026: +0.187

## Best / worst pressure matches
- Best: match 50359 (+2.298)
- Worst: match 50350 (-0.996)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.