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
