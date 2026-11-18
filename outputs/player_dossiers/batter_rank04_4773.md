# Player Dossier — Jaker Ali Anik (Rank 4, batter)

- **Final score:** 0.723 (z-composite; pool mean = 0)
- **Reliability:** 142 HP balls, 15 HP matches, 2 seasons; bootstrap 90% CI of HP value [-0.049, 0.296]
- **Level (EB-shrunk LCB):** 0.058 (raw HP mean 0.134)
- **Pressure uplift:** raw 0.061, context-adjusted 0.351 (normal baseline 0.074)
- **Consistency:** positive-match share 80.0%, match sd 0.411
- **Downside:** CVaR5 -0.864, negative-match rate 20.0%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.136 over 142 balls
- Extreme-pressure value mean: -0.061 over 73 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: RAPID_COLLAPSE (+0.678), NEW_BATTER_AFTER_WICKET (+0.384), WICKET_DROUGHT_BREAKTHROUGH (+0.321)
- Weaknesses: BOUNDARY_DROUGHT (-0.122), HIGH_RRR_CHASE (-0.163), LOW_WICKET_RESOURCE_CHASE (-1.466)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.136

## Season trend (HP value mean by season)

- 2022: +nan
- 2022/23: +nan
- 2023: +nan
- 2023/24: +nan
- 2024: +0.225
- 2024/25: +nan
- 2025: +0.036
- 2026: +nan

## Best / worst pressure matches
- Best: match 46754 (+0.725)
- Worst: match 45511 (-0.864)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.