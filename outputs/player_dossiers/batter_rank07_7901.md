# Player Dossier — Parvez Emon (Rank 7, batter)

- **Final score:** 0.677 (z-composite; pool mean = 0)
- **Reliability:** 81 HP balls, 16 HP matches, 4 seasons; bootstrap 90% CI of HP value [0.036, 0.469]
- **Level (EB-shrunk LCB):** 0.157 (raw HP mean 0.250)
- **Pressure uplift:** raw 0.407, context-adjusted 0.467 (normal baseline -0.152)
- **Consistency:** positive-match share 81.2%, match sd 0.643
- **Downside:** CVaR5 -1.386, negative-match rate 18.8%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.255 over 81 balls
- Extreme-pressure value mean: 0.282 over 30 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: RAPID_COLLAPSE (+0.162), WICKET_DROUGHT_BREAKTHROUGH (+0.124), MATCH_ON_LINE (+0.066)
- Weaknesses: HIGH_RRR_CHASE (-0.435), NEW_BATTER_AFTER_WICKET (-1.093), FINAL_OVER_PRESSURE (-2.395)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.255

## Season trend (HP value mean by season)

- 2022: +0.227
- 2022/23: +nan
- 2023: +nan
- 2023/24: +nan
- 2024: +0.983
- 2024/25: +nan
- 2025: +0.133
- 2026: +0.286

## Best / worst pressure matches
- Best: match 42744 (+1.517)
- Worst: match 51384 (-1.386)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.