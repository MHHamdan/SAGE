# Experiment E4 — Closed-Loop Adaptive Stability Controller Report

## Configuration
- Tasks: 50 × 3 seeds = 150 total evaluations per condition
- Total turns per episode: 50
- Completion threshold: drift < 0.35
- Max interventions per task: 10

## Results Table

| Condition | Completion Rate | 95% CI | Mean Cost | CNSR | Mean Interventions |
|-----------|----------------|--------|-----------|------|-------------------|
| NoControl | 2.0% | [0.0%, 4.7%] | $1.000 | 0.020 | 0.0 |
| FixedSchedule | 36.7% | [29.3%, 44.7%] | $1.050 | 0.349 | 5.0 |
| ThresholdController | 89.3% | [84.0%, 94.0%] | $1.416 | 0.631 | 9.7 |
| PredictiveController | 83.3% | [77.3%, 88.7%] | $1.302 | 0.640 | 5.6 |

## Hypothesis Tests (Holm-Bonferroni corrected)

### H4.1: PredictiveController vs. NoControl (completion rate)
- NoControl: 2.0% (95% CI [0.0%, 4.7%])
- PredictiveController: 83.3% (95% CI [77.3%, 88.7%])
- Δ = 81.3%
- Cohen's h = 2.017
- McNemar p-value = 0.0
- Reject H0 (Holm-corrected): True

### H4.2: PredictiveController vs. FixedSchedule (CNSR)
- FixedSchedule CNSR: 0.349
- PredictiveController CNSR: 0.640

### H4.3: ThresholdController vs. NoControl (completion rate)
- ThresholdController: 89.3%
- NoControl: 2.0%

## Cost Overhead of PredictiveController vs. NoControl
- NoControl mean cost: $1.000
- PredictiveController mean cost: $1.302
- Overhead: 30.2% (target: <25%)

## Statistical Notes
- All CIs are bootstrap 95% (10,000 resamples).
- McNemar's test used for paired binary outcomes (same tasks across seeds).
- Multiple comparisons corrected with Holm-Bonferroni (3 hypothesis tests).
- Non-significant results are reported without hedging.

## Figures
- `e4_completion_rate.pdf`: Completion rates with bootstrap CIs
- `e4_cost_vs_completion.pdf`: Pareto plot
- `e4_intervention_timing.pdf`: Intervention histograms by controller
- `e4_drift_trajectories.pdf`: Mean drift ± SD over time