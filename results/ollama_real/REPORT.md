# SAGE — Local Open-Weight Reproducibility Study (Track 2)

**Purpose.** A fully public, zero-cost, local reproduction of the SAGE evaluation using
open-weight models via Ollama, on a real task (HotpotQA-distractor) with real retrieval,
real exact-match success, and monitor signals computed from real model outputs. This
complements the paper's controlled *simulation* study (E4/E5) and reports the raw local
numbers exactly as they fell — no tuning toward any target.

**Environment.** Ollama 0.19.0, 4× RTX 2080 Ti. git SHA `32f0c66`. Cost accounting uses
measured Ollama tokens (each model's own tokenizer) × hosted list price (Together AI
serverless size-tier, retrieved 2026-07-18); no GPU wall-clock enters cost. Dataset:
`hotpot_qa/distractor/validation`.

**Full disclosure of harness changes (no result-forcing).** The only change made to the
agent loop during smoke-testing was a **forced final-answer step**: if the ReAct agent
reaches its turn budget without emitting `ACTION: ANSWER[...]`, it is prompted once to give
a final short answer, instead of the previous behaviour of using its last search query as
the "answer." This is a correctness/fairness fix applied **identically to every policy and
model**; it changes no task definition or success criterion. Nothing else was altered to
move any number.

---

## 1. Controlled mechanism (paper E4 — *simulation*, shown for contrast)

E4 (`experiments/e4_closed_loop.py`, committed) makes **no LLM calls**: it is a seeded
64-dim goal/state random walk in which `success = final_drift < 0.35` and interventions
mechanically pull the state vector toward the goal (`GoalReanchor` recovery = 0.40). It
**isolates the mechanism**: if long-horizon failure is a recoverable goal-drift, a
predictive controller recovers it.

| E4 policy (simulation) | Completion |
|---|---|
| NoControl | 2.0% |
| FixedSchedule | 36.7% |
| ThresholdController | 89.3% |
| PredictiveController | 83.3% |

*Terminality (committed traces):* under NoControl **150/150** episodes cross the drift
threshold and only **3 (2%)** ever recover; under Predictive **125/150 (83%)** recover, and
each intervention drops drift by **Δ = −0.131** on the next turn. The large effect is a
property of a dynamics where the failure and the intervention are algebraic inverses — a
mechanism illustration, not a claim about real agents.

---

## 2. Real CNSR ladder (2A) — open weights on HotpotQA-distractor

50 tasks × 2 seeds, temp 0.0. Success = EM. `results/ollama_real/2A/`.

| Model | Success (EM) | Mean tokens | Cost ($) | CNSR |
|---|---|---|---|---|
| Llama-3.2-3B | 24.0% | 1156 | 6.93e-5 | **3461** |
| Gemma-2-9B | 48.0% | 1282 | 3.85e-4 | 1248 |
| Qwen2.5-14B | 48.0% | 1862 | 5.59e-4 | 859 |
| Mistral-7B | 34.0% | 2909 | 5.82e-4 | 584 |
| Qwen2.5-32B | 54.0% | 1304 | 1.04e-3 | 518 |

**Canonical Kendall τ_b (EM-rank vs CNSR-rank) = −0.527 (p = 0.207)** across the 5 configs
(tie-corrected `scipy.stats.kendalltau`; underpowered at n=5, sign is the claim). The CNSR
inversion **reproduces on real open-weight models**: success rises with model size, CNSR
falls. Texture: Mistral-7B is penalised because it genuinely loops (mean 2.3 turns; 22% hit
the 6-turn cap vs 4–8% for the others), inflating its tokens — token efficiency, not just
per-token price, drives CNSR.

---

## 3. Real closed-loop controller (2B / E4-T) — HotpotQA-distractor

4 policies × 5 seeds × 50 eval tasks + 150 disjoint training tasks; agent `llama3.1:8b`,
temp 0.6; predictor trained with task-stratified 5-fold CV on question id +
`assert_no_leakage`. BCa 95% CIs. `results/ollama_real/2B/`.

| Policy | Completion (95% CI) | CNSR | Interv./task | McNemar p | Cohen's h |
|---|---|---|---|---|---|
| **(a) All 50 tasks** | | | | | |
| NoControl | **32.4%** [26.8, 38.4] | 381 | 0.0 | — | — |
| FixedSchedule | 34.0% [28.0, 40.0] | 416 | 1.2 | 0.596 | +0.034 |
| Threshold | 28.0% [22.8, 34.0] | 356 | 1.9 | 0.082 | −0.096 |
| Predictive | 31.2% [25.2, 36.8] | 389 | 1.3 | 0.749 | −0.026 |
| **(b) Pathology subset (48/50, osc>0.6 under NoControl)** | | | | | |
| NoControl | 31.7% [26.2, 37.9] | 359 | 0.0 | — | — |
| FixedSchedule | 33.3% [27.5, 39.2] | 393 | 1.3 | 0.596 | +0.036 |
| Threshold | 27.1% [21.7, 32.9] | 332 | 2.0 | 0.082 | −0.101 |
| Predictive | 30.4% [24.6, 36.2] | 366 | 1.4 | 0.749 | −0.027 |

**Raw result, as it fell:** no controller produces a significant lift (all |h| < 0.10, all
CIs overlap). The predictor is at chance (CV-AUC 0.494). Fidelity is inert (mean 0.9998,
coefficient 0.0), corroborating E5's "fidelity is a single-step alarm, not a lead
predictor." Interventions are marginally harmful (Threshold worst): `ForceReplan` /
`GoalReanchor` reset accumulated reasoning, which on bounded-context QA discards progress.

**Why (diagnosis, real data):** on HotpotQA, oscillation is a *symptom*, not a terminal,
recoverable derailment: P(success | oscillated) = 31.4% ≈ P(success | not) = 37.0% ≈ base
32.4%, so oscillation is non-predictive of failure; and breaking the loop does not help
(oscillating-episode completion: NoControl 31.4%, Predictive 30.2%, Threshold 26.2%). Real
2-hop QA failures are **capability-limited** (the model doesn't know / can't retrieve the
fact), not the recoverable goal-drift the controller is built for. Even the strongest local
model raises only the base rate (Qwen2.5-32B NoControl = 54% EM, §2), not the controllable
margin.

---

## 4. Findings paragraphs (for the Author Response letter)

**Reproducibility framing.** To ensure the SAGE evaluation is fully reproducible at zero
cost, we re-ran the cost (Assess) and closed-loop control (Enforce) evaluations on
local open-weight models via Ollama, on real HotpotQA-distractor items with real retrieval
and exact-match success. We report the raw local numbers without tuning; the only harness
change was a fairness fix (a forced final-answer step) applied uniformly across all
conditions.

**CNSR inversion (real).** The cost-normalized success ranking inverts the raw success
ranking on real open-weight models (Kendall τ_b = −0.527 across five configurations): the
3B model attains the highest CNSR despite the lowest accuracy, and the 32B model the lowest
CNSR despite the highest accuracy. This confirms the metric's central claim on real data
and is not a simulation artifact.

**Closed-loop control (real, and its boundary).** On real 2-hop QA the predictive
controller yields no significant completion gain over NoControl (32.4% → 31.2%, McNemar
p=0.75; all |Cohen's h| < 0.10). The controlled simulation (E4) shows a large gain because
its failure mode is a recoverable goal-drift that the intervention directly reverses; on
HotpotQA the monitored pathology (oscillation) is a symptom of capability-limited failure,
non-predictive of the outcome (predictor AUC = 0.494) and not repaired by intervention. We
therefore present E4 as a controlled illustration of the control mechanism and report the
real-data transfer honestly: the mechanism benefits tasks whose failures are recoverable
derailments, and 2-hop QA is not such a task.

---

*Payload:* `results/ollama_real/{2A,2B}/{*.csv, *_table.tex, manifest.json}`, this REPORT,
and `manifest.json` (below). Kept entirely separate from the committed simulation results;
no committed result file or paper table was modified.
