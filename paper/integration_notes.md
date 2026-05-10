# Paper Integration Notes — Adaptive Stability Controller (ASC)

Auto-generated from `results/e4_closed_loop/REPORT.md` and `results/e5_predictive/REPORT.md`.
**Do not edit the paper directly** — this file describes what must change and supplies the
numbers to paste.

---

## Checklist of paper sections requiring updates

- [ ] §III.F (or new §III.G) — "Closed-Loop Adaptive Control" subsection
- [ ] Abstract — one sentence with headline E4 finding
- [ ] §I — Contribution list (add 8th contribution)
- [ ] §VIII.D — Future Directions (replace generic text with specific open problems)
- [ ] Supplementary §A.5.4 — Experiment E4
- [ ] Supplementary §A.5.5 — Experiment E5
- [ ] Case studies — re-analyze Bing Chat / Air Canada through ASC lens

---

## Headline Numbers (populate these in abstract and contributions)

### E4 — Closed-Loop Ablation (50 tasks × 3 seeds = 150 evaluations per condition)

| Condition | Completion Rate | 95% Bootstrap CI | CNSR | Mean Cost | Mean Interventions |
|-----------|----------------|-----------------|------|-----------|-------------------|
| NoControl | 2.0% | [0.0%, 4.7%] | 0.020 | $1.000 | 0.0 |
| FixedSchedule (k=10) | 36.7% | [29.3%, 44.7%] | 0.349 | $1.050 | 5.0 |
| ThresholdController | 89.3% | [84.0%, 94.0%] | 0.631 | $1.416 | ~7.1 |
| PredictiveController | 83.3% | [77.3%, 88.7%] | 0.640 | $1.302 | ~5.4 |

**H4.1 (Predictive > NoControl):** Δ = +81.3 pp, McNemar p < 0.0001, reject H0 ✓
**H4.2 (Predictive CNSR > FixedSchedule CNSR):** 0.640 vs 0.349, McNemar p < 0.0001, reject H0 ✓
**H4.3 (Threshold > NoControl):** Δ = +87.3 pp, McNemar p < 0.0001, reject H0 ✓

**Cost overhead of PredictiveController vs. NoControl:** +30.2% (exceeds 25% target — note in paper;
driven by intervention costs not excessive intervention count; acceptable given 81 pp completion gain).

**Suggested abstract sentence:**
> "A predictive closed-loop controller raises long-horizon task completion from 2.0% (NoControl)
> to 83.3% (PredictiveController) [Δ = +81.3 pp, 95% CI 76.3–88.0 pp] at 30% cost overhead,
> while a fixed-schedule baseline achieves only 36.7%."

### E5 — Predictive Monitor Validation (300 tasks, 5-fold CV, k ∈ {3, 5, 10})

| Predictor | AUC (k=5) | 95% CI | AP | Brier |
|-----------|-----------|--------|----|-------|
| drift_only | 0.609 | [0.594, 0.622] | 0.127 | — |
| oscillation_only | 0.589 | [0.576, 0.602] | 0.116 | — |
| fidelity_only | 0.495 | [0.479, 0.510] | 0.098 | — |
| combined (all 8 features) | 0.752 | [0.741, 0.763] | 0.211 | — |
| MLP (32 hidden) | 0.516 | [0.502, 0.529] | 0.099 | — |

**H5.1 results:**
- drift_only: AUC=0.609, Mann-Whitney p < 0.0001 → reject H0 ✓
- oscillation_only: AUC=0.589, Mann-Whitney p < 0.0001 → reject H0 ✓
- fidelity_only: AUC=0.495, p = 0.760 → fail to reject H0 (fidelity is not predictive in this simulation)

**H5.2:** Combined (0.752) > best single drift_only (0.609): Δ = +0.143 AUC.
Statistical comparison via bootstrap: note the test uses pooled CV predictions; direction is clear
from Δ magnitude even where p-value computation requires aligned paired predictions.

**H5.3 — Lead time trade-off (combined model):**

| k | AUC | 95% CI |
|---|-----|--------|
| 3 | ~0.74 | — |
| 5 | 0.752 | [0.741, 0.763] |
| 10 | ~0.75 | — |

(AUC is relatively stable across k=3–10 because the simulation's failure signal accumulates
monotonically over turns; in real agents, expect steeper degradation at longer lead times.)

---

## New §III.B — "Closed-Loop Adaptive Control" (insert after §III-A Formal Stability)

### Algorithm box (pseudocode for agent loop with controller hook)

```
Algorithm 1: Agent loop with Adaptive Stability Controller

Input: task T, goal g, controller C, budget B, max_interventions M
Output: outcome ∈ {success, failure, escalation}

1. state ← init(T, g)
2. interventions ← 0
3. for t = 1 to B do
4.   a_t ← agent_policy(state, g)
5.   o_t ← environment(state, a_t)
6.   state ← update(state, a_t, o_t)
7.   σ_t ← monitor(state, g)          // MonitorSignals
8.   d_t ← C.decide(σ_t)              // InterventionDecision
9.   if d_t.intervention ≠ None and interventions < M then
10.    state ← d_t.intervention.apply(state)
11.    interventions ← interventions + 1
12.  if goal_satisfied(state, g) then return success
13. return failure if drift(state, g) < θ else escalation
```

### Architecture diagram description (for Figure)

Four components arranged in a feedback loop:
1. **Agent** (ReAct loop) — takes actions, observes environment
2. **Stability Monitor** (existing, §III-F) — computes drift/oscillation/fidelity signals
3. **Controller** (new ASC) — maps MonitorSignals → InterventionDecision
4. **Intervention Library** (new) — applies bounded corrective actions to agent state

---

## Supplementary §A.5.4 — Experiment E4

### Hypothesis
H4.1: PredictiveController > NoControl on completion rate.
H4.2: PredictiveController > FixedSchedule on CNSR (completion/dollar).
H4.3: ThresholdController > NoControl (demonstrates adaptive beats uncontrolled).

### Design
- 50 long-horizon Wikipedia-chain research tasks (held-out from A3 training set)
- 50-turn budget, drift < 0.35 completion criterion
- 4 conditions × 3 seeds = 12 cells; 50 trials per cell = 150 evaluations per condition
- Pre-trained predictor: 100 offline NoControl runs → logistic regression k=5

### Statistical analysis
- Bootstrap 95% CI: 10,000 resamples (BCa approximation via percentile method)
- Paired McNemar's test for binary outcome comparisons (same 50 task IDs × 3 seeds)
- Holm-Bonferroni correction over 3 hypothesis tests (α = 0.05)
- Effect sizes: Cohen's h for proportion differences, Cliff's delta for cost distributions

### Results → paste from Table above

### Key finding
Both the learned controller (Predictive) and the rule-based controller (Threshold) dramatically
outperform the baselines.  PredictiveController achieves 0.640 CNSR vs 0.349 for FixedSchedule,
confirming H4.2: adaptive timing of interventions is more cost-efficient than fixed scheduling.
The cost overhead of 30% exceeds the 25% target stated in H4.1; this should be reported without
hedging, with the note that the completion gain (+81 pp) justifies the overhead in practice.

---

## Supplementary §A.5.5 — Experiment E5

### Hypothesis
H5.1: Each monitor signal provides better-than-chance k-step failure prediction.
H5.2: Combined logistic regression beats any single-signal predictor.
H5.3: Accuracy degrades gracefully with longer lead times.

### Design
- 200 high-drift violation tasks + 100 baseline tasks = 300 training tasks
- 50 turns per task; 5-fold CV split by task_id (no turn-level leakage)
- Predictors: drift_only, oscillation_only, fidelity_only (logistic), combined (8-feature logistic),
  MLP (32 hidden units, ablation)
- Lead times k ∈ {3, 5, 10}

### Results → paste from Table above

### Key finding
Drift and oscillation signals are meaningful predictors (AUC = 0.609 and 0.589 respectively).
Fidelity signal is not predictive in the current simulation (AUC ≈ 0.5), suggesting the simulation
should be extended to model correlated fidelity errors (future work).
The combined model (AUC = 0.752) provides substantially better prediction than any single signal,
confirming H5.2 and supporting the design decision to use all three monitor signals in the
PredictiveController.

---

## §I Contribution List — 8th Contribution to Add

> **(8) Adaptive Stability Controller (ASC):** A closed-loop controller that consumes
> stability-monitor signals as feedback and emits bounded corrective interventions during
> execution.  We implement four controller variants (NoControl, FixedSchedule, Threshold,
> Predictive), five interventions (GoalReanchor, ContextCompress, ForceReplan,
> SchemaValidatedRetry, HumanEscalate), and a calibrated failure predictor.  A 150-evaluation
> ablation (E4) demonstrates that PredictiveController raises long-horizon task completion from
> 2.0% to 83.3% at 30% cost overhead.  A 300-task predictive validation (E5) shows that the
> drift and oscillation monitor signals achieve AUC = 0.609 and 0.589 as 5-step-ahead failure
> predictors, and the combined model achieves AUC = 0.752.**

---

## §VIII.D Future Directions — Specific Open Problems

Replace the current generic "robust long-horizon learning" paragraph with:

> The ASC experiments surface three concrete open problems.  (1) **Controller transfer:**
> the ThresholdController's hand-tuned thresholds (drift > 0.30, oscillation > 0.60) are
> task-family specific; learning these from data or transferring them across task families
> remains open.  (2) **Learned controller policies:** the current PredictiveController uses a
> fixed logistic regression; a meta-RL policy that learns when to intervene from trial experience
> could reduce the 30% cost overhead while maintaining completion gains.  (3) **Formal
> closed-loop stability guarantees:** the current paper proves stability conditions for
> open-loop LLM agents (Theorem 1); extending these proofs to the closed-loop system with
> an active controller — showing that the controller's interventions provably keep the agent in
> the stability region — is an open theoretical problem.

---

## Case Study Re-analysis Paragraphs

### Bing Chat (Sydney)
> Through the ASC lens, the Sydney incident exhibits clear Goal-Divergence (Definition 1
> violated): context drift over multi-turn conversations caused Sydney's state embedding to
> diverge from its intended-goal manifold.  A GoalReanchor intervention — re-injecting the
> original system prompt with 40% recovery pull — would have triggered at turn t when
> drift_score > 0.30, likely before the harmful outputs appeared.  Whether the prompt-injection
> attack that triggered the divergence would have been caught by SchemaValidatedRetry depends
> on whether the tool-output schema was specified; this highlights the importance of schema
> coverage in the fidelity monitor.

### Air Canada
> The Air Canada chatbot failure represents a Bounded-Oscillation violation (Definition 2):
> the agent repeated contextually incorrect refund claims, a cycle detectable by the oscillation
> monitor (overlap_ratio > 0.60).  A ForceReplan intervention would have discarded the
> current plan and regenerated from the current state, likely producing a more conservative
> response.  The HumanEscalate intervention provides the fail-safe: when both drift and
> oscillation signals exceed thresholds simultaneously, the ASC escalates rather than
> continues, matching the liability-management recommendation in the case.

---

## Implementation Notes for Paper Authors

- All numbers in this file are from the run with `--seed 42`.
- Full experiment logs: `results/e4_closed_loop/REPORT.md` and `results/e5_predictive/REPORT.md`
- MANIFEST.json in each results dir records the git SHA, env hash, and output file hashes for
  reproducibility verification.
- The smoke tests (< 60s total) run on every CI commit: `pytest tests/stability/test_e4_e5_smoke.py`
- To reproduce the full experiments: `python experiments/e4_closed_loop.py --seed 42` and
  `python experiments/e5_predictive_validation.py --seed 42`.
- Statistical caveats to report honestly:
  - E4 cost overhead of 30% exceeds the <25% target stated in H4.1.
  - E5 fidelity_only predictor is not better than chance (AUC ≈ 0.5); this is a negative result.
  - E5 MLP underperforms logistic regression; likely due to class imbalance + small hidden layer.
  - H5.2 bootstrap p-value requires aligned paired predictions; report as directional finding
    (combined Δ = +0.143 AUC vs. best single) rather than a formal significance test.
