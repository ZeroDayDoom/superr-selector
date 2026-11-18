# Player Dossier — Shakib Al Hasan (Rank 3, bowler)

- **Final score:** 0.945 (z-composite; pool mean = 0)
- **Reliability:** 73 HP balls, 11 HP matches, 4 seasons; bootstrap 90% CI of HP value [0.030, 1.417]
- **Level (EB-shrunk LCB):** 0.164 (raw HP mean 0.451)
- **Pressure uplift:** raw 0.330, context-adjusted 0.473 (normal baseline 0.127)
- **Consistency:** positive-match share 63.6%, match sd 3.796
- **Downside:** CVaR5 -0.341, negative-match rate 36.4%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.456 over 73 balls
- Extreme-pressure value mean: 0.305 over 37 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: FINAL_OVER_PRESSURE (+1.793), EARLY_COLLAPSE_RESCUE (+0.932), HIGH_RRR_CHASE (+0.712)
- Weaknesses: MATCH_ON_LINE (-0.182), MIDDLE_ORDER_RESCUE (-0.226), RAPID_COLLAPSE (-0.473)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.456

## Season trend (HP value mean by season)

- 2022: +0.096
- 2022/23: +0.101
- 2023: +0.553
- 2023/24: +nan
- 2024: +1.203
- 2024/25: +nan
- 2025: +nan
- 2026: +nan

## Best / worst pressure matches
- Best: match 23674 (+11.203)
- Worst: match 23673 (-0.341)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.