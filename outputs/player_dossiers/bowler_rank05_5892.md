# Player Dossier — Blair Tickner (Rank 5, bowler)

- **Final score:** 0.845 (z-composite; pool mean = 0)
- **Reliability:** 75 HP balls, 9 HP matches, 3 seasons; bootstrap 90% CI of HP value [0.016, 0.686]
- **Level (EB-shrunk LCB):** 0.119 (raw HP mean 0.268)
- **Pressure uplift:** raw 0.279, context-adjusted 0.274 (normal baseline -0.009)
- **Consistency:** positive-match share 66.7%, match sd 0.710
- **Downside:** CVaR5 -0.072, negative-match rate 33.3%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.271 over 75 balls
- Extreme-pressure value mean: 0.116 over 43 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: FINAL_OVER_PRESSURE (+0.690), NEW_BATTER_AFTER_WICKET (+0.395), MATCH_ON_LINE (+0.313)
- Weaknesses: BOUNDARY_DROUGHT (-0.174), WICKET_DROUGHT_BREAKTHROUGH (-0.226), RAPID_COLLAPSE (-0.273)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.271

## Season trend (HP value mean by season)

- 2022: +0.348
- 2022/23: -0.003
- 2023: +0.761
- 2023/24: +nan
- 2024: +nan
- 2024/25: +nan
- 2025: +nan
- 2026: +nan

## Best / worst pressure matches
- Best: match 21407 (+2.117)
- Worst: match 17743 (-0.072)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.