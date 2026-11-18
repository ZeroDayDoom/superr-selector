# SupeRR Selector Season 2 — Pressure Analytics Pipeline

Identify the most reliable T20 batters and bowlers under pressure using
ball-by-ball IPL and T20I data (2022–2026), with a reproducible,
leakage-audited machine-learning pipeline.

## Quick start (Windows PowerShell)

```powershell
# from the superr_selector directory
python -m pip install -e .
python scripts/run_audit.py        # Phase 1: data audit (reports/)
python scripts/run_prepare.py      # Phase 3: cleaning + match states
python scripts/run_features.py     # Phase 4: features + leakage audit
python scripts/run_pressure.py     # Phase 5: WP + pressure engine
python scripts/run_models.py       # Phase 6: expected performance models
python scripts/run_ratings.py      # Phase 7: ratings + rankings + dossiers
python scripts/run_validation.py   # Phase 8: validation battery
python scripts/build_final_report.py   # Phase 10: report + submission files
python scripts/test_dashboard.py > reports\_dashboard_smoke.txt
python scripts/final_audit.py      # Phase 12: programmatic verification
streamlit run app/main.py          # Phase 9: dashboard (http://localhost:8501)
```

Requires Python 3.11+ with the dependencies in `pyproject.toml`
(`pip install -e .` handles everything, including PyTorch).

## Project layout

| path | purpose |
|---|---|
| `configs/config.yaml` | all analytical constants (thresholds, weights, tiers, bootstrap draws) |
| `src/superr/data/` | audit, cleaning, match states, leakage-safe features |
| `src/superr/features/` | feature engineering + registry (128 documented features) |
| `src/superr/models/` | WP, leverage, pressure, expected-performance, deep models |
| `src/superr/ratings/` | value metrics, shrinkage, reliability, rankings |
| `src/superr/validation/` | bootstrap, split-half, sensitivity, adversarial checks |
| `src/superr/dashboard/` | dashboard data layer (single source of truth) |
| `scripts/` | one command per phase + `test_dashboard.py` |
| `tests/` | pytest suites (37 tests) |
| `data/processed/` | parquet artefacts (balls_clean, pressure_scores, expected_performance, features_*) |
| `outputs/` | rankings, metrics, dossiers, models, figures, exports |
| `reports/` | audit, methodology, validation, robustness reports |

## Headline outputs

- `outputs/submission_top10.csv` / `.xlsx` / `submission_entry_sheet.txt` — the two ordered top-10 lists
- `outputs/rankings_batters.csv` / `rankings_bowlers.csv` — top-10 with full rating components
- `outputs/all_player_metrics.csv` — every metric for every player
- `outputs/top10_probability.csv` — P(top-10) over 1,000 match bootstraps
- `outputs/player_dossiers/` — evidence dossiers for every top-10 player
- `reports/final_methodology.pdf` (+ `.md`, `.html`), `reports/executive_summary.md`
- `reports/final_project_audit.md` — programmatic pass/fail verification of every claim

## Method in one paragraph

Pressure is a continuous per-delivery index built from a calibrated
win-probability model (leverage + closeness) and situational stress
components; the hybrid (Model C) is the primary definition, with three
models and five threshold quantiles compared in a pre-registered validation
battery. Player value is performance above the contextual expectation
models (LightGBM outcome buckets) with dynamically-valued wickets
(λ from total-projection curves). Ratings combine EB-shrunk level (lower
confidence bound), context-adjusted pressure uplift, match consistency,
volume/diversity and downside risk, ranked with 1,000-draw match-cluster
bootstrap uncertainty. No player names or reputation enter any scoring path;
every threshold is sensitivity-tested and the full multiverse is reported.

## Data requirements

The six supplied CSVs must sit in the parent directory of `superr_selector/`:
`ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`,
`playing_eleven.csv`, `competition_team_player_mapping.csv`. The raw files
are read in place and never modified.

## Methodology summary

1. **Pressure** (continuous, pre-delivery): hybrid of win-probability
   closeness + leverage and situational stress; weights calibrated on the
   2025 validation season; HP = top-20% within innings × phase strata.
2. **Expected performance**: LightGBM outcome-bucket models (mutually
   exclusive, consistent by construction) for batting and bowling
   (attribution-split for bowlers), plus logistic and league baselines,
   multi-task MLP and GRU sequence experiments; chronological
   train ≤2024 / validate 2025 / holdout 2026.
3. **Player value**: runs above expectation + dynamic-λ wicket value
   (batting); runs prevented + wickets above expectation (bowling).
4. **Ratings**: EB-shrunk HP level (LCB), context-adjusted uplift,
   match consistency, volume/diversity, downside risk — z-composite with
   sensitivity variants.
5. **Uncertainty**: 1,000-draw match-cluster bootstrap → rank intervals and
   P(top-10); match-level split-half reliability; temporal persistence folds.

## Limitations

- No ball-tracking, fielding, pitch, weather, dew or boundary-dimension data.
- First-innings win probability is a total-projection proxy (AUC ≈ 0.66).
- Clutch persistence in sport is modest; uplift estimates are shrunk and
  interval-quantified, not certainties.
- Opponent adjustment is a two-pass approximation, not a full hierarchical
  plus-minus model.
- `bowl_speed` unrecorded on 38.7% of deliveries (excluded, not imputed).

## Reproducibility

- All constants live in `configs/config.yaml`; seeds fixed (`20260131`).
- Every phase writes a report; audits re-verify claims
  (`scripts/run_audit.py`, `scripts/final_audit.py`).
- Leakage audit: feature registry + outcome-perturbation test
  (50 sampled matches × 3 perturbed deliveries each →
  `reports/leakage_audit.json`).
- The full chain from raw CSVs to the final report is deterministic given
  the fixed seed and package versions in `pyproject.toml`.

## Final submission artefacts

- `reports/final_methodology.pdf` (+ `.md`, `.html`)
- `outputs/submission_top10.csv` / `.xlsx` / `submission_entry_sheet.txt`
- `outputs/rankings_batters.csv`, `rankings_bowlers.csv`
- `outputs/all_player_metrics.csv`, `outputs/eligibility_audit.csv`
- `outputs/player_dossiers/`, `outputs/player_public_context.csv`
- `reports/executive_summary.md`, `reports/final_project_audit.md`
