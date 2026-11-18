# Player Dossier — Mohammad Wasim (Rank 7, bowler)

- **Final score:** 0.688 (z-composite; pool mean = 0)
- **Reliability:** 104 HP balls, 11 HP matches, 6 seasons; bootstrap 90% CI of HP value [-0.135, 0.541]
- **Level (EB-shrunk LCB):** 0.011 (raw HP mean 0.157)
- **Pressure uplift:** raw 0.144, context-adjusted 0.164 (normal baseline 0.015)
- **Consistency:** positive-match share 72.7%, match sd 1.580
- **Downside:** CVaR5 -0.528, negative-match rate 27.3%

## Pressure sample
- HP strike rate/economy metrics: see all_player_metrics.csv row; HP value mean 0.158 over 104 balls
- Extreme-pressure value mean: 0.214 over 57 balls

## Scenario strengths (top) / weaknesses (bottom)

- Strengths: MIDDLE_ORDER_RESCUE (+1.009), EARLY_COLLAPSE_RESCUE (+0.304), SET_BATTER_DISMISSAL (+0.144)
- Weaknesses: MATCH_ON_LINE (-0.278), HIGH_RRR_CHASE (-0.297), LOW_WICKET_RESOURCE_CHASE (-0.437)

## Competition split (HP value mean)

- hpval_IPL: +nan
- hpval_senior T20I: +0.158

## Season trend (HP value mean by season)

- 2022: +0.331
- 2022/23: -0.034
- 2023: +0.715
- 2023/24: +nan
- 2024: +0.148
- 2024/25: +nan
- 2025: -0.090
- 2026: +5.252

## Best / worst pressure matches
- Best: match 48512 (+5.252)
- Worst: match 17738 (-0.528)

## Why this player ranks here

- The composite is dominated by the player's level under pressure (shrunk lower-confidence-bound), weighted 0.45, plus consistency (0.20), context-adjusted uplift (0.20), volume (0.15) and downside (−0.10). All components are z-scores across the eligible pool.

## What could make this ranking wrong

- Small HP samples: the bootstrap CI above quantifies this; a player near the eligibility floor could move several places within the CI.
- The pressure definition (Model C hybrid, WP-based) could misclassify states; the multiverse analysis in the methodology report covers threshold and weight sensitivity.
- Matchup/opponent strength is adjusted through the expectation models; a bowler who faced weak opposition line-ups could be overrated if the two-pass quality terms under-correct.
- Clutch persistence in sport is modest; some of the observed uplift may be noise despite shrinkage.