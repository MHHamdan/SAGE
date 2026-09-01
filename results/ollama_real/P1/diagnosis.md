# P1 — Long-horizon terminal-derailment diagnosis

Regime: qwen2.5:32b, temp 0.6, 5 seeds, max_turns=8, **no forced answer** (derailment is terminal). This restores the E4 failure mode that 2B's forced-answer masked.

## Does the monitored pathology precede *unrecoverable* failure?
- NoControl base completion: **61.7%**; derail rate 0.0%.
- Of NoControl failures (23): **0% are derailments** (ran out of budget without answering = terminal) vs **100% wrong-answers** (answered but incorrect = capability-limited).
- Pathology episodes (oscillation > 0.6): 8. Of these, **0% are terminal** (derail) and **100% self-recover** (still answer).

## Does the controller rescue the derailed episodes?
- On the (seed,task) episodes that derailed under NoControl, **Threshold** completion = 0.0%, still-derailed 0.0% (n=0).
- On the (seed,task) episodes that derailed under NoControl, **Predictive** completion = 0.0%, still-derailed 0.0% (n=0).

## Verdict
NoControl 61.7% vs Predictive 60.0% (McNemar p=1.000, Cohen's h=-0.03). The controller helps iff derailment is both common and rescuable; the numbers above show whether that holds. Reported straight, no tuning.