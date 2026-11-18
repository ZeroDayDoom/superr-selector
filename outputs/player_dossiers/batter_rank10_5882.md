# Player Dossier — Afif Hossain (Rank 10, batter)

- **Final score:** 0.669 (z-composite; pool mean = 0)
- **Reliability:** 113 HP balls, 10 HP matches, 3 seasons; bootstrap 90% CI of HP value [0.021, 0.358]
- **Level (EB-shrunk LCB):** 0.114 (raw HP mean 0.186)
- **Pressure uplift:** raw 0.169, context-adjusted 0.403 (normal baseline 0.019)
- **Consistency:** positive-match share 70.0%, match sd 0.445
- **Downside:** CVaR5 -0.115, negative-match rate 30.0%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.188 over 113 balls
- Extreme-pressure value mean: 0.131 over 55 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: LOW_WICKET_RESOURCE_CHASE (+0.672), NEW_BATTER_AFTER_WICKET (+0.418), RAPID_COLLAPSE (+0.318)
- Weaknesses: MATCH_ON_LINE (+0.028), SET_BATTER_DISMISSAL (+0.024), EARLY_COLLAPSE_RESCUE (-0.036)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.188

## Season trend (HP value mean by season)

- 2022: +0.309
- 2022/23: +0.019
- 2023: +nan
- 2023/24: +nan
- 2024: +0.305
- 2024/25: +nan
- 2025: +nan
- 2026: +nan

## Best / worst pressure matches
- Best: match 16655 (+1.363)
- Worst: match 17742 (-0.115)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.