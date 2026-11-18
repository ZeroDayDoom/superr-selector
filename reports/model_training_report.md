# Model Training Report — Expected Performance

_Generated 2026-11-18T10:19:03+00:00_

## Protocol

- Chronological match-level splits: **train <=2024, validation 2025,
  holdout 2026**. No random ball-level splits.
- Selection on **validation** log loss (per task); holdout reported once.
- Selected configuration refit on all seasons for final scoring.

## Selection

- **Batting outcome model: `logistic`**
- **Bowling outcome model: `mlp`**
- Wicket models: {'is_team_wicket': 'lightgbm', 'is_bowler_wicket': 'lightgbm'}

## Comparison

| task | model | split | log loss | MAE xRuns |
|---|---|---|---|---|
| batting_outcomes | league_average | validation | 1.63 | 1.3493 |
| batting_outcomes | league_average | test | 1.6541 | 1.3929 |
| batting_outcomes | logistic | validation | 1.0277 | 0.5668 |
| batting_outcomes | logistic | test | 1.0299 | 0.5786 |
| batting_outcomes | lightgbm | validation | 1.0282 | 0.555 |
| batting_outcomes | lightgbm | test | 1.0316 | 0.5683 |
| batting_outcomes | mlp | validation | 1.0357 | 0.5809 |
| batting_outcomes | mlp | test | 1.0353 | 0.5852 |
| batting_outcomes | gru_sequence | validation | 1.5921 | 1.3465 |
| batting_outcomes | gru_sequence | test | 1.6166 | 1.4015 |
| wicket_is_team_wicket | lightgbm | validation | 0.1996 | nan |
| wicket_is_team_wicket | lightgbm | test | 0.1935 | nan |
| wicket_is_bowler_wicket | lightgbm | validation | 0.1925 | nan |
| wicket_is_bowler_wicket | lightgbm | test | 0.187 | nan |
| bowling_outcomes | league_average | validation | 1.6417 | 1.2816 |
| bowling_outcomes | league_average | test | 1.6644 | 1.3216 |
| bowling_outcomes | logistic | validation | 1.039 | 0.5173 |
| bowling_outcomes | logistic | test | 1.0405 | 0.5275 |
| bowling_outcomes | lightgbm | validation | 1.0403 | 0.5062 |
| bowling_outcomes | lightgbm | test | 1.0441 | 0.5197 |
| bowling_outcomes | mlp | validation | 1.0369 | 0.5187 |
| bowling_outcomes | mlp | test | 1.0365 | 0.5243 |

## Deep learning outcome

- The MLP (256/128/64, multi-task) and the GRU sequence model were
  trained and evaluated under the identical protocol. If they did
  not beat the tabular winner above, they are retained as
  experimental models (saved in outputs/models/) and NOT used for
  the ranking. This follows the pre-registered decision rule.
- MLP final val loss: 2.1772 (first epoch 2.7747)
- GRU final val loss: 1.5921 (first epoch 1.6216)

## Empirical conceded-runs mapping (bowling)

```json
{
  "bowler_wicket": 0.003,
  "dot": 0.0,
  "four": 4.004,
  "other_extras": 1.196,
  "other_wicket": 0.471,
  "single": 1.0,
  "six": 6.004,
  "two_three": 2.059
}
```

## Explainability

- LightGBM gain importances saved under
  `model_feature_importance/` (top features are state/phase/pressure
  variables; player identity is not a feature by design).
- SHAP is deferred: gain + residuals-by-strata provide the audit
  trail; SHAP values are NOT treated as causal evidence.

## Limitations

- First-innings win-probability signal is weak (AUC ~0.66); pressure
  components for innings 1 rely more on the par/collapse axis.
- The league baseline intentionally excludes player identity; player
-specific quality enters only via the separate two-pass quality
  terms in the ratings phase.