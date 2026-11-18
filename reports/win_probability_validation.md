# Win Probability Validation

## WP2 (chase)

- Selected model family: **logistic**
- Chronological protocol: train <=2024, calibrate 2025, holdout 2026

| family | split | ROC-AUC | PR-AUC | log loss | Brier | calib slope |
|---|---|---|---|---|---|---|
| logistic | train | 0.891 | 0.874 | 0.411 | 0.133 | 1.00 |
| logistic | validation | 0.910 | 0.909 | 0.376 | 0.123 | 1.09 |
| logistic | holdout | 0.915 | 0.923 | 0.389 | 0.127 | 0.99 |
| hist_gbm | train | 0.984 | 0.982 | 0.191 | 0.056 | 2.06 |
| hist_gbm | validation | 0.897 | 0.894 | 0.465 | 0.139 | 0.54 |
| hist_gbm | holdout | 0.836 | 0.853 | 0.758 | 0.205 | 0.34 |
| lightgbm | train | 0.990 | 0.989 | 0.165 | 0.047 | 2.30 |
| lightgbm | validation | 0.891 | 0.888 | 0.501 | 0.147 | 0.48 |
| lightgbm | holdout | 0.845 | 0.860 | 0.751 | 0.205 | 0.34 |

## WP1 (first innings)

- Selected model family: **logistic**
- Chronological protocol: train <=2024, calibrate 2025, holdout 2026

| family | split | ROC-AUC | PR-AUC | log loss | Brier | calib slope |
|---|---|---|---|---|---|---|
| logistic | train | 0.705 | 0.721 | 0.621 | 0.217 | 1.00 |
| logistic | validation | 0.715 | 0.690 | 0.627 | 0.219 | 0.98 |
| logistic | holdout | 0.657 | 0.602 | 0.712 | 0.253 | 0.58 |
| hist_gbm | train | 0.989 | 0.990 | 0.214 | 0.057 | 2.85 |
| hist_gbm | validation | 0.639 | 0.629 | 0.907 | 0.289 | 0.22 |
| hist_gbm | holdout | 0.576 | 0.541 | 0.999 | 0.326 | 0.13 |
| lightgbm | train | 0.994 | 0.994 | 0.178 | 0.045 | 2.95 |
| lightgbm | validation | 0.594 | 0.573 | 1.044 | 0.316 | 0.13 |
| lightgbm | holdout | 0.591 | 0.541 | 1.064 | 0.325 | 0.13 |

## WP calibration by pressure quintile (WP2, all seasons)

| pressure quintile | ECE | n |
|---|---|---|
| (0.0426, 0.269] | 0.0196 | 18381 |
| (0.269, 0.391] | 0.0296 | 18380 |
| (0.391, 0.46] | 0.0145 | 18380 |
| (0.46, 0.514] | 0.0194 | 18380 |
| (0.514, 0.878] | 0.0365 | 18381 |
