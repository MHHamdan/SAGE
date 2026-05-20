# Changelog

All notable changes to this project are documented here.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/),
and this project follows [Semantic Versioning](https://semver.org/).

## [1.2.0] — 2026-05-20 — SAGE rebrand

### Changed
- **Framework rebrand**: the project is now named **SAGE** — *Stabilize, Assess,
  Govern, Enforce*. The Python import name is `sage` and the PyPI distribution
  is `sage-framework`. The GitHub repository moved to
  [`MHHamdan/SAGE`](https://github.com/MHHamdan/SAGE).
- **Package directory**: `src/agentic_toolkit/` renamed to `src/sage/`. All
  internal imports updated.
- **Tests**: internal test suite switched from `from agentic_toolkit.X` to
  `from sage.X`. Test outcomes are identical to the prior `main`
  (651 passing, 42 pre-existing failures, 8 pre-existing errors — all
  unrelated to the rename).
- **README**: rewritten around the four-pillar framing with explicit pillar
  → code-path table; resolved stale merge conflict markers and the duplicate
  "## HEAD" intro left over from a previous merge.
- **Packaging**: `[project.name]` `agentic-ai-toolkit` → `sage-framework`,
  `[project.version]` `1.1.0` → `1.2.0` (also fixed prior drift between
  `pyproject.toml` at 1.1.0 and the now-removed `src/agentic_toolkit/__init__.py`
  at 0.1.0), all four `[project.urls]` repointed at `MHHamdan/SAGE`.
- **Experiment bootstrap**: removed a broken
  `sys.path.insert(0, ROOT / "agentic_ai_toolkit" / "src")` line from
  `experiments/exp_obs_fidelity.py`, `exp_context_noise.py`, and
  `cnsr_multitask.py` — that path has never existed in this repo.

### Added
- **Backwards-compatibility shim** at `src/agentic_toolkit/__init__.py` that
  aliases every `agentic_toolkit.*` submodule to the corresponding `sage.*`
  module in `sys.modules` via `pkgutil.walk_packages`. Class identity and
  `isinstance` checks are preserved across both paths. Importing
  `agentic_toolkit` emits a `DeprecationWarning`. The shim will be removed
  in a future release.
- This `CHANGELOG.md`.
- `TODO.md` capturing the planned `sage/governance/` consolidation (see G
  pillar mapping in README).

### Not changed
- No algorithm, threshold, default argument, or numerical claim has been
  modified. Empirical results for experiments A1, A2, A3, CNSR multi-task,
  E4 closed-loop, and E5 predictive validation reproduce identically.
- LICENSE, cached LLM responses under `results/cache/`, experiment input
  data, and notebook output cells were not touched.

## [1.1.0] — prior to SAGE rebrand *(reconstructed from git history)*

### Added
- Closed-loop **Adaptive Stability Controller (ASC)** with four policies
  (NoControl, FixedSchedule, Threshold, Predictive) and five bounded
  interventions (GoalReanchor, ContextCompress, ForceReplan,
  SchemaValidatedRetry, HumanEscalate) — commit
  [`7c2afe2`](../../commit/7c2afe2).
- Experiments **E4** (closed-loop ablation) and **E5** (predictive monitor
  validation), with full reproducibility artifacts under
  `results/e4_closed_loop/` and `results/e5_predictive/`.
- Research-experiment suite (CNSR multi-task, Proposition 1 violations
  A1/A2/A3, LLM-as-Judge bias) and the lightweight `eval/metrics.py` shim
  — commit [`27fb2c6`](../../commit/27fb2c6).
- System architecture diagrams in the README.

[1.2.0]: https://github.com/MHHamdan/SAGE/releases/tag/v1.2.0
[1.1.0]: https://github.com/MHHamdan/SAGE/releases/tag/v1.1.0
