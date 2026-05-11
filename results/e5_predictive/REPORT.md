# Experiment E5 — Predictive Monitor Validation Report

## Predictor AUC-ROC at k=5 (95% Bootstrap CI)

| Predictor | AUC | 95% CI | AP | Brier |
|-----------|-----|--------|----|-------|
| drift_only | 0.609 | [0.596, 0.622] | 0.127 | 0.241 |
| oscillation_only | 0.589 | [0.576, 0.602] | 0.116 | 0.243 |
| fidelity_only | 0.495 | [0.479, 0.510] | 0.098 | 0.250 |
| combined | 0.752 | [0.741, 0.762] | 0.211 | 0.210 |
| mlp | 0.516 | [0.501, 0.530] | 0.099 | 0.103 |

## Lead-Time vs. AUC Trade-off (combined model)

| k | AUC | 95% CI |
|---|-----|--------|
| 3 | 0.744 | [0.730, 0.757] |
| 5 | 0.752 | [0.741, 0.762] |
| 10 | 0.771 | [0.763, 0.780] |

## Hypothesis Tests (Holm-Bonferroni corrected)

### H5.1: Single-signal predictors better than chance (k=5)
  - drift_only: AUC=0.609, p=0.0, reject H0=True
  - oscillation_only: AUC=0.589, p=0.0, reject H0=True
  - fidelity_only: AUC=0.495, p=0.6883, reject H0=False

### H5.2: Combined vs. best single-signal (k=5)
  - Combined AUC: 0.752
  - Best single-signal AUC: 0.609
  - p-value (bootstrap): 0.946
  - Reject H0: False

### H5.3: AUC decreases as k increases (monotone check)
  - k=3: AUC=0.744
  - k=5: AUC=0.752
  - k=10: AUC=0.771

## Statistical Notes
- All AUC CIs are bootstrap 95% (10,000 resamples).
- Splits are by task_id; no turn-level leakage.
- Multiple comparisons corrected with Holm-Bonferroni.
- Calibration (Brier score + reliability diagrams) reported separately.
- 'Better than chance' test: bootstrap p-value for H0: AUC=0.5.

## Figures
- `e5_roc_curves.pdf`: ROC curves at k=5
- `e5_lead_time_tradeoff.pdf`: AUC vs. k
- `e5_calibration.pdf`: Reliability diagrams at k=5
- `e5_feature_importance.pdf`: Combined model coefficients