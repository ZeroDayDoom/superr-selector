# Player Dossier — Fakhar Zaman (Rank 3, batter)

- **Final score:** 0.729 (z-composite; pool mean = 0)
- **Reliability:** 380 HP balls, 30 HP matches, 5 seasons; bootstrap 90% CI of HP value [0.016, 0.215]
- **Level (EB-shrunk LCB):** 0.071 (raw HP mean 0.114)
- **Pressure uplift:** raw 0.119, context-adjusted 0.329 (normal baseline -0.005)
- **Consistency:** positive-match share 73.3%, match sd 0.510
- **Downside:** CVaR5 -1.126, negative-match rate 26.7%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.115 over 380 balls
- Extreme-pressure value mean: 0.016 over 213 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: EARLY_COLLAPSE_RESCUE (+0.296), WICKET_DROUGHT_BREAKTHROUGH (+0.236), MIDDLE_ORDER_RESCUE (+0.194)
- Weaknesses: HIGH_RRR_CHASE (-0.070), DEATH_OVERS_PRESSURE (-0.086), DEATH_DEFENCE (-0.086)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.115

## Season trend (HP value mean by season)

- 2022: +0.052
- 2022/23: +nan
- 2023: +0.450
- 2023/24: +nan
- 2024: +0.033
- 2024/25: +nan
- 2025: +0.138
- 2026: +0.233

## Best / worst pressure matches
- Best: match 47979 (+1.073)
- Worst: match 16654 (-1.139)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.