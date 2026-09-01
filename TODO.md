# TODO

- [ ] **Govern pillar**: factor the failure taxonomy (10 classes,
      currently `sage.evaluation.failure_taxonomy.FailurePathology` +
      `MITIGATION_STRATEGIES` + `FailureDetector` in
      `src/sage/evaluation/failure_taxonomy.py`, ~925 lines) and the
      STRIDE threat catalog (11 vectors, currently
      `sage.security.threat_validator.STRIDECategory` /
      `ThreatDefinition` / `STRIDEReport` in
      `src/sage/security/threat_validator.py`) into a single
      `sage/governance/` package with typed enums and per-class
      metadata. Today these live in `evaluation/` and `security/` for
      historical reasons; the SAGE pillar story is cleaner with them
      consolidated. See paper §VIII and §XI for the design spec.

- [ ] **Known test failures** (pre-existing at v1.2.0; 42 failing, 8 errors
      out of 701 collected). None affects a paper claim — the `stability`,
      `monitoring`, and `security` suites are green in full, as are the CNSR
      estimator and goal-drift sentinel checks. CI runs the paper-critical
      suites as blocking and the full suite as advisory so this stays visible.
      The failures cluster into four groups:
  - [ ] `tests/test_equation_consistency.py::TestTaskCost` (2) — written
        against a `CostTracker` API (`add_tokens`, `add_tool_call(cost=)`,
        `add_human_intervention`, `calculate_total_cost`, `token_cost`) that
        `sage.core.cost.CostTracker` does not expose; the shipped API is
        `record_tokens` / `record_tool_call` / `get_summary`. Broken tests,
        not a broken cost model — rewrite them against the real API.
  - [ ] `tests/evaluation/` (11 failed, 8 errors) — chiefly
        `test_long_horizon.py`, whose fixtures construct an evaluator
        signature that has since changed.
  - [ ] `tests/protocols/` (10) — MCP/A2A client tests that assume a live
        server; they need fixtures or `pytest.mark.integration`.
  - [ ] `tests/integration/` (4) and `tests/tools/` — end-to-end flows and
        policy-enforcement assertions that drifted from the current
        `verification/` and `tools/` interfaces.
