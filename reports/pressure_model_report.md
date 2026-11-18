# Pressure Model Report

_Generated 2026-11-18T08:45:20+00:00_

## Selected model

- **PRESSURE MODEL C (hybrid) is the primary pressure score.**
- Model B weights (data-calibrated on the 2025 validation season): `{'match_closeness': np.float64(0.4), 'leverage_norm': np.float64(0.2), 'difficulty': np.float64(0.4)}`
- Model C mix: `{'B': np.float64(0.7), 'A': np.float64(0.2), 'resource_stress': 0.1}`
- Model A is retained as the transparent rules-based anchor.

## Weight calibration criterion (pre-registered)

Weights maximise the mean of (a) outcome-variance ratio HP vs rest,
(b) realized-|WPA| ratio HP vs rest, and (c) split-half stability of
player HP means — computed on the 2025 validation season only.

## High-pressure (top 20% within innings x phase) vs ordinary deliveries

| metric | high pressure | ordinary |
|---|---|---|
| strike_rate | 140.319 | 139.888 |
| dot_pct | 33.356 | 33.837 |
| boundary_pct | 17.855 | 18.141 |
| wicket_prob | 6.297 | 5.183 |
| bowling_economy | 8.753 | 8.743 |
| mean_abs_realized_wpa | 0.032 | 0.014 |
| mean_leverage | 0.031 | 0.014 |
| n | 38449.000 | 153786.000 |

- Variance ratio (runs): **0.99x**
- Realized |WPA| ratio: **2.31x**

## Pressure distributions

- `pressure_by_over`: {1: 0.3477, 2: 0.3384, 3: 0.3556, 4: 0.3536, 5: 0.3551, 6: 0.3581, 7: 0.3607, 8: 0.3683, 9: 0.3745, 10: 0.3815, 11: 0.3895, 12: 0.3948, 13: 0.4006, 14: 0.4061, 15: 0.4136, 16: 0.43, 17: 0.4368, 18: 0.4389, 19: 0.4466, 20: 0.4577}
- `pressure_by_wickets`: {0: 0.3354, 1: 0.3581, 2: 0.3807, 3: 0.4039, 4: 0.4224, 5: 0.4292, 6: 0.4435, 7: 0.4401, 8: 0.4408, 9: 0.4369}
- `pressure_by_rrr`: {'<6': 0.1889, '6-8': 0.3071, '8-10': 0.4467, '10-12': 0.477, '12+': 0.4754}
- `pressure_by_innings`: {1: 0.3713, 2: 0.406}
- `pressure_by_phase`: {'death': 0.4411, 'middle': 0.3875, 'powerplay': 0.3514}
- `hp_share_by_competition`: {'IPL': 0.1868, 'senior T20I': 0.211}

## Worked examples (10 real match states)

### extreme_death_chase — match 20188, innings 2, over 20

- State: 186/7, 3 balls left, target 198, RRR 24.0
- WP(before) = 0.001, leverage = 0.0144
- Components: {'difficulty': 0.933, 'resource_stress': 0.993, 'collapse_pressure': 0.4, 'momentum_stress': 0.167, 'phase_pressure': 1.0, 'match_importance': 0.2, 'match_closeness': 0.002, 'leverage_norm': 0.12}
- Pressure A=0.654, B=0.398, **C=0.509** -> category **HIGH**
- Why: the weighted component mix above — dominant components are ['phase_pressure', 'resource_stress', 'difficulty']

### easy_chase_end — match 39810, innings 2, over 16

- State: 199/1, 26 balls left, target 205, RRR 1.38
- WP(before) = 1.0, leverage = 0.0
- Components: {'difficulty': 0.06, 'resource_stress': 0.805, 'collapse_pressure': 0.0, 'momentum_stress': 0.0, 'phase_pressure': 1.0, 'match_importance': 0.2, 'match_closeness': 0.0, 'leverage_norm': 0.0}
- Pressure A=0.259, B=0.024, **C=0.149** -> category **LOW**
- Why: the weighted component mix above — dominant components are ['phase_pressure', 'resource_stress', 'match_importance']

### powerplay_start — match 29924, innings 1, over 1

- State: 0/0, 120 balls left, target 161, RRR None
- WP(before) = 0.499, leverage = 0.0173
- Components: {'difficulty': 0.056, 'resource_stress': 0.0, 'collapse_pressure': 0.0, 'phase_pressure': 0.3, 'match_importance': 0.2, 'match_closeness': 0.997, 'leverage_norm': 0.144}
- Pressure A=0.067, B=0.45, **C=0.328** -> category **LOW**
- Why: the weighted component mix above — dominant components are ['match_closeness', 'phase_pressure', 'match_importance']

### collapse_middle — match 40753, innings 2, over 9

- State: 70/5, 71 balls left, target 279, RRR 17.66
- WP(before) = 0.0, leverage = 0.0
- Components: {'difficulty': 0.898, 'resource_stress': 0.704, 'collapse_pressure': 1.0, 'momentum_stress': 0.111, 'phase_pressure': 0.4, 'match_importance': 0.2, 'match_closeness': 0.0, 'leverage_norm': 0.0}
- Pressure A=0.652, B=0.359, **C=0.452** -> category **MODERATE**
- Why: the weighted component mix above — dominant components are ['collapse_pressure', 'difficulty', 'resource_stress']

### anchor_under_par — match 40734, innings 1, over 15

- State: 81/5, 36 balls left, target 144, RRR None
- WP(before) = 0.153, leverage = 0.0057
- Components: {'difficulty': 0.865, 'resource_stress': 0.85, 'collapse_pressure': 0.0, 'momentum_stress': 0.0, 'phase_pressure': 0.4, 'match_importance': 0.2, 'match_closeness': 0.306, 'leverage_norm': 0.047}
- Pressure A=0.447, B=0.478, **C=0.509** -> category **HIGH**
- Why: the weighted component mix above — dominant components are ['difficulty', 'resource_stress', 'phase_pressure']

### high_rrr_mid — match 20225, innings 2, over 10

- State: 77/2, 66 balls left, target 215, RRR 12.55
- WP(before) = 0.159, leverage = 0.0427
- Components: {'difficulty': 0.752, 'resource_stress': 0.56, 'collapse_pressure': 0.0, 'momentum_stress': 0.139, 'phase_pressure': 0.4, 'match_importance': 0.2, 'match_closeness': 0.317, 'leverage_norm': 0.356}
- Pressure A=0.391, B=0.499, **C=0.483** -> category **HIGH**
- Why: the weighted component mix above — dominant components are ['difficulty', 'resource_stress', 'phase_pressure']

### defending_low_total — match 31279, innings 1, over 17

- State: 108/3, 22 balls left, target 142, RRR None
- WP(before) = 0.179, leverage = 0.0337
- Components: {'difficulty': 0.713, 'resource_stress': 0.872, 'collapse_pressure': 0.0, 'momentum_stress': 0.0, 'phase_pressure': 1.0, 'match_importance': 0.2, 'match_closeness': 0.359, 'leverage_norm': 0.281}
- Pressure A=0.465, B=0.485, **C=0.52** -> category **HIGH**
- Why: the weighted component mix above — dominant components are ['phase_pressure', 'resource_stress', 'difficulty']

### knockout_death — match 14145, innings 2, over 20

- State: 171/5, 3 balls left, target 178, RRR 14.0
- WP(before) = 0.489, leverage = 0.361
- Components: {'difficulty': 0.833, 'resource_stress': 0.988, 'collapse_pressure': 0.6, 'momentum_stress': 0.139, 'phase_pressure': 1.0, 'match_importance': 0.2, 'match_closeness': 0.978, 'leverage_norm': 1.0}
- Pressure A=0.659, B=0.924, **C=0.878** -> category **EXTREME**
- Why: the weighted component mix above — dominant components are ['phase_pressure', 'leverage_norm', 'resource_stress']

### boundary_drought — match 16655, innings 1, over 10

- State: 42/4, 65 balls left, target 128, RRR None
- WP(before) = 0.158, leverage = 0.0047
- Components: {'difficulty': 0.687, 'resource_stress': 0.675, 'collapse_pressure': 0.0, 'momentum_stress': 0.5, 'phase_pressure': 0.4, 'match_importance': 0.2, 'match_closeness': 0.316, 'leverage_norm': 0.039}
- Pressure A=0.442, B=0.409, **C=0.442** -> category **MODERATE**
- Why: the weighted component mix above — dominant components are ['difficulty', 'resource_stress', 'momentum_stress']

### new_batter_after_wicket — match 45852, innings 2, over 11

- State: 76/5, 58 balls left, target 208, RRR 13.66
- WP(before) = 0.001, leverage = 0.0003
- Components: {'difficulty': 0.876, 'resource_stress': 0.758, 'collapse_pressure': 0.6, 'momentum_stress': 0.222, 'phase_pressure': 0.4, 'match_importance': 0.2, 'match_closeness': 0.002, 'leverage_norm': 0.002}
- Pressure A=0.59, B=0.352, **C=0.44** -> category **MODERATE**
- Why: the weighted component mix above — dominant components are ['difficulty', 'resource_stress', 'collapse_pressure']
