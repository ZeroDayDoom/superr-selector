# Executive Summary — SupeRR Selector Season 2 (Round 1)

## The question

Which T20 batters and bowlers deliver most reliably under pressure? The
answer must come from the supplied ball-by-ball data (IPL + senior T20I,
2022–2026), ranked 1–10 for each role, with a transparent methodology.

## The answer

**Top 10 batters**

1. **Mohammad Rizwan** (rating 0.774, 467 HP balls, P(top-10) 0.85)
2. **Nitish Reddy** (rating 0.756, 116 HP balls, P(top-10) 0.17)
3. **Fakhar Zaman** (rating 0.729, 380 HP balls, P(top-10) 0.31)
4. **Jaker Ali Anik** (rating 0.723, 142 HP balls, P(top-10) 0.27)
5. **Nurul Hasan** (rating 0.716, 93 HP balls, P(top-10) 0.25)
6. **Harry Tector** (rating 0.679, 204 HP balls, P(top-10) 0.28)
7. **Parvez Emon** (rating 0.677, 81 HP balls, P(top-10) 0.17)
8. **Mahendra Singh Dhoni** (rating 0.673, 85 HP balls, P(top-10) 0.01)
9. **Shai Hope** (rating 0.672, 179 HP balls, P(top-10) 0.29)
10. **Afif Hossain** (rating 0.669, 113 HP balls, P(top-10) 0.57)

**Top 10 bowlers**

1. **Fareed Malik** (rating 1.270, 79 HP balls, P(top-10) 0.68)
2. **Kuldeep Sen** (rating 1.070, 73 HP balls, P(top-10) 0.24)
3. **Shakib Al Hasan** (rating 0.945, 73 HP balls, P(top-10) 0.02)
4. **Mohammed Shami** (rating 0.904, 234 HP balls, P(top-10) 0.19)
5. **Blair Tickner** (rating 0.845, 75 HP balls, P(top-10) 0.46)
6. **Mohsin Khan** (rating 0.744, 161 HP balls, P(top-10) 0.16)
7. **Mohammad Wasim** (rating 0.688, 104 HP balls, P(top-10) 0.32)
8. **Digvesh Rathi** (rating 0.685, 79 HP balls, P(top-10) 0.04)
9. **Harshit Rana** (rating 0.684, 189 HP balls, P(top-10) 0.46)
10. **Shamar Joseph** (rating 0.665, 123 HP balls, P(top-10) 0.48)

## The method in five steps

1. **Define pressure from the data.** A calibrated win-probability model
   (holdout AUC 0.915 on chases) converts every pre-delivery state into
   a closeness × leverage signal; combined with situational stress
   (required-rate burden, resources, collapse, momentum, phase, match
   importance) into a continuous pressure index. Weights are calibrated on
   the 2025 season against a pre-registered criterion; high-pressure (HP)
   deliveries = top 20% within innings × phase. HP states show
   2.31× the realised
   win-probability swing of ordinary deliveries.
2. **Measure performance against context.** LightGBM expectation models
   (trained ≤2024, validated 2025, held out 2026 — never random ball splits)
   predict the outcome distribution of an average player in the identical
   situation; player value = actual minus expected, with wickets valued
   dynamically (λ from total-projection curves, mean λ ≈
   9.83 runs).
3. **Rate reliability, not hot streaks.** Empirical-Bayes-shrunk HP level
   (lower confidence bound), context-adjusted pressure uplift, match-level
   consistency, volume/diversity and downside risk combine into a
   z-composite with pre-registered sensitivity variants.
4. **Quantify uncertainty.** 1000-draw match-level
   bootstrap gives every rank an interval and every player a P(top-10).
5. **Stress-test everything.** Thresholds, pressure-model variants
   (A/B/C), competition scopes, eligibility tiers, shrinkage strengths,
   weight variants and component ablations — the full multiverse is
   reported; rankings must survive it.

## Evidence the pressure definition works

| metric | high pressure | ordinary |
|---|---|---|
| wicket probability | 6.3% | 5.2% |
| realised \|WPA\| | 0.032 | 0.014 |
| bowling economy | 8.75 | 8.74 |

Split-half reliability of HP value (batters): Spearman
0.093. Temporal persistence: rank_2022-2023_eval_2024-2026: ρ=0.038; rank_2022-2024_eval_2025-2026: ρ=0.032; rank_2022-2025_eval_2026-2026: ρ=0.212.

## Scope

883 matches, 192,235 deliveries, 1,116
players evaluated, three eligibility tiers with a per-player audit trail.
Known limitations (no tracking/pitch/weather data; innings-1 WP is a
projection proxy; modest clutch persistence) are quantified in the full
methodology report.

## Reproduce

```powershell
python scripts/run_audit.py; python scripts/run_prepare.py
python scripts/run_features.py; python scripts/run_pressure.py
python scripts/run_models.py; python scripts/run_ratings.py
python scripts/run_validation.py; python scripts/build_final_report.py
python scripts/final_audit.py
```
