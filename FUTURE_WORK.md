# Future Work

Items explicitly deferred from the ASC implementation scope.

## Out of scope for this PR

- Training the agent itself (frozen pre-trained LLMs only)
- New agent architectures beyond the existing ReAct
- Dashboard / UI for controller monitoring
- Optimizing inference latency below the existing baseline
- Multi-agent settings (single-agent only for E4/E5)
- Live API runs in CI (cached simulation only)
- Writing the paper text directly

## Open problems surfaced by ASC experiments

1. **Controller transfer across task families.** The ThresholdController's hand-tuned
   thresholds (drift > 0.30, oscillation > 0.60) were set for the Wikipedia-chain drift
   simulation. Validating them on coding or web-navigation tasks requires re-calibration.

2. **Learned controller policies.** The PredictiveController fires GoalReanchor or ForceReplan
   based on a simple dominance rule (drift vs. oscillation). A meta-RL or bandit policy
   that learns which intervention to apply — and when — from episode experience could
   reduce the 30% cost overhead while maintaining completion gains.

3. **Formal closed-loop stability guarantees.** Theorem 1 (paper §III-F) provides sufficient
   conditions for open-loop LLM agent stability. Extending these proofs to the closed-loop
   system — showing that ASC's interventions provably keep the agent in the stability region —
   is an open theoretical problem.

4. **Fidelity signal as a predictive feature.** In the current simulation, schema-validation
   errors are generated independently of drift, making the fidelity signal non-predictive
   (AUC ≈ 0.5 in E5). A more realistic simulation where tool errors co-occur with goal
   misalignment would be needed to validate the fidelity monitor as a predictive signal.

5. **Multi-agent ASC.** The current ASC is single-agent only. In hierarchical or collaborative
   multi-agent systems, intervention decisions interact across agents — one agent's ForceReplan
   may destabilize a dependent agent's context.

6. **Online predictor updates.** The current FailurePredictor is trained offline and frozen.
   An online variant that updates its weights as new trace data arrives within a deployment
   session could adapt to distribution shift in long deployments.
