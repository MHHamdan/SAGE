# Changelog

All notable changes to this project are documented here.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased] — public research-artifact release

Prepares the repository as the official open-source artifact accompanying the
IEEE Transactions on Artificial Intelligence article. **No algorithm, threshold,
default argument, experiment, or published number changed.** Every committed
result artifact under `results/` is byte-identical to the previous commit.

### Added
- **`docs/assets/`** — all fifteen figures the paper includes (five in the main text,
  ten in the supplement), rendered to PNG at 3x from the publication vector sources.
- **`paper/public/figures/`** — the same fifteen figures as vector PDF, the version of
  record.
- **`CITATION.cff`** — machine-readable citation metadata; GitHub renders it under
  *Cite this repository*.
- **`scripts/env_report.py`** — environment/provenance report (text and JSON) covering
  interpreter, platform, git revision, determinism-relevant environment variables, and
  the library versions that affect numerical output. Credential *presence* is reported;
  credential values never are. The `make env-report` and `make env-report-json` targets
  referenced this script but it did not exist.
- **CI `experiments` job** — runs every experiment entry point end to end at reduced
  task counts and asserts its artifacts are written.
- **`src/sage/core/backends.py`** — explicit provider backend layer (`litellm` /
  `ollama` / `simulator`) returning uniform completions with *measured* token counts.
  Credentials are read only from the environment or `.env`; a real backend requested
  without credentials raises `CredentialError` rather than silently degrading to the
  simulator, so a run can never be mistaken for a real one.
- **`experiments/revision/`** — the supplementary-material studies: real open-weight
  and metered-API CNSR ladders, leave-one-out pillar ablation, the `MCP-I02`
  prompt-injection attack and sanitizer, ASC latency/memory overhead, CNSR
  cost-parameter sensitivity, E4 threshold surface, and E5 monitor error analysis;
  with their committed artifacts under `results/`.

### Fixed
- **`results/cache/` documented as empty.** Supplementary F.5 describes an offline
  replay mode backed by cached LLM responses there; the cache was not retained, so the
  README states this plainly and points offline reproduction at `--backend simulator`
  instead of repeating a claim the artifact does not support.
- **Simulator determinism in `experiments/cnsr_multitask.py`** — the per-run RNG seed
  was derived from `hash(task_id) ^ hash(model)`, but Python salts string hashing, so
  the simulated success rates, token counts, and the resulting Kendall's tau changed
  from process to process. Seeds are now derived from an MD5 of the identity string
  and are stable across processes. `experiments/revision/cnsr_tau_canonical.py
  --salted` reproduces the old behavior for comparison.

### Changed
- **`README.md`** — rewritten from scratch against the paper source, with
  **Supplementary Material F (Reference Implementation Details)** as the authoritative
  spec for what this repository claims. Covers the evaluation gap, the closed-loop
  model, per-pillar sections carrying the paper's own scope limits, key contributions
  with the three negative findings stated as prominently as the positive ones, results
  for every experiment with its data source labelled (simulation vs real models), the
  consolidated experiment table from Supp. H.1, a paper-to-code map spanning the main
  text and all supplements, and a reproducibility section built around the checks the
  paper says the artifacts support. Every code snippet was executed against this
  repository's API, and every reported number is machine-checked against the committed
  CSVs and manifests.
- **Reproducibility framing corrected.** The paper scopes its reproducibility claim to
  120 pillar-aligned tests (32 in `tests/monitoring/`, 88 in `tests/stability/`) and
  explicitly places other modules out of scope. The README now leads with that scope
  and with `tests/stability/test_e4_e5_smoke.py` — the 18-test, ~9-second suite that
  validates the headline E4/E5 invariants offline — rather than with a raw full-suite
  pass/fail count.
- **`requirements.txt`** — was still the stale "Agentic AI Toolkit" list, missing
  `scikit-learn`, `scipy`, `pandas`, `litellm`, and `sentence-transformers` (all of
  which the experiments import) while carrying unused Sphinx pins. Now mirrors the
  `[experiments]` extra of `pyproject.toml` with matching version floors.
- **`Dockerfile`** — default `CMD` pointed at `./run_all.sh`, which does not exist in
  this repository; it now runs the test suite. Also pins `PYTHONHASHSEED=0` for
  determinism, installs the `[experiments]` extra, and copies dependency metadata
  before source so the dependency layer caches.
- **CI `test` job** — the paper-critical suites (`stability`, `monitoring`, `security`,
  equation consistency) now run as a blocking step, with the full suite as an advisory
  step. The pre-existing failures in peripheral subsystems are surfaced rather than
  masked by a single red check.

### Removed
- **`src/agentic_toolkit/`** — the `agentic_toolkit` → `sage` deprecation shim added in
  1.2.0. The rename has landed; `import sage` is the only supported entry point.
- **CI `integration` job** — invoked `paper_assets/scripts/generate_tables.py` and
  `generate_figures.py`, paths that have never existed in this repository. Replaced by
  the `experiments` job above, which exercises the real `experiments/*.py` scripts.

### Security
- **`.private/`** (gitignored) now holds all local-only development material:
  manuscript drafts and the submission bundle, reviewer correspondence, agent
  prompts and instruction files, and internal audit notes.
- **`paper/`** split into `paper/public/` (released) and `paper/private/` (local only).
- **`.gitignore`** extended with explicit rules for manuscript sources, rebuttals,
  submission bundles, and agent prompts and transcripts. Generated LaTeX *table
  fragments* under `results/` remain tracked as reproducibility artifacts, and
  `results/api_real/` is published: it holds only per-task token counts, cost, and
  exact-match scores -- no credentials or account identifiers -- and Supplementary
  H.2's API table is not reproducible without it.
- Two revision scripts carried peer-review references in their docstrings
  (`e4_threshold_surface.py`, `ollama_2b.py`); both now cite the supplementary
  material instead.
- Audited every tracked file for API keys, tokens, email addresses, and absolute local
  paths: none found, and no private path has ever been committed to this repository's
  history.

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
