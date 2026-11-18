# Player Dossier — Shamar Joseph (Rank 10, bowler)

- **Final score:** 0.665 (z-composite; pool mean = 0)
- **Reliability:** 123 HP balls, 14 HP matches, 3 seasons; bootstrap 90% CI of HP value [-0.058, 0.957]
- **Level (EB-shrunk LCB):** 0.043 (raw HP mean 0.251)
- **Pressure uplift:** raw 0.221, context-adjusted 0.260 (normal baseline 0.032)
- **Consistency:** positive-match share 57.1%, match sd 2.763
- **Downside:** CVaR5 -0.420, negative-match rate 42.9%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.253 over 123 balls
- Extreme-pressure value mean: 0.029 over 49 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: EARLY_COLLAPSE_RESCUE (+0.391), NEW_BATTER_PRESSURE (+0.151), PARTNERSHIP_REBUILD (+0.101)
- Weaknesses: HIGH_RRR_CHASE (-0.304), RAPID_COLLAPSE (-0.422), LOW_WICKET_RESOURCE_CHASE (-0.740)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.253

## Season trend (HP value mean by season)

- 2022: +nan
- 2022/23: +nan
- 2023: +nan
- 2023/24: +nan
- 2024: +0.288
- 2024/25: +nan
- 2025: -0.257
- 2026: +0.919

## Best / worst pressure matches
- Best: match 50504 (+8.635)
- Worst: match 32618 (-0.420)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.