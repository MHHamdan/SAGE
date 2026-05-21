# SAGE: A Stabilize–Assess–Govern–Enforce Framework for Deployment-Ready LLM-Based Autonomous Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-651%20passing-brightgreen.svg)](#testing)
[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](pyproject.toml)

> Reference implementation for the IEEE Transactions on Artificial Intelligence submission of the same name (TAI-2025-Dec-R-02684).

## What is SAGE?

SAGE is a four-pillar framework for deployment-ready LLM-based autonomous agents. It treats agents as non-stationary closed-loop systems rather than static reasoners, and integrates four operational capacities: stability modelling (**Stabilize**), cost-aware autonomy assessment (**Assess**), failure and protocol governance (**Govern**), and adaptive corrective control (**Enforce**). Closed-loop adaptive control transforms long-horizon completion from 2.0% under open-loop execution to ≥ 83% under either of two adaptive policies (E4).



## The four pillars

| Pillar | What it does | Code | Paper |
|---|---|---|---|
| **S — Stabilize** | Closed-loop stability framing; monitorable conditions for observation fidelity, progress monotonicity, bounded context noise | [`sage/stability/`](src/sage/stability), [`sage/monitoring/stability_monitor.py`](src/sage/monitoring/stability_monitor.py) | §III, Supp. A.1–A.4 |
| **A — Assess** | Cost-Normalized Success Rate (CNSR); behavioural autonomy taxonomy | [`sage/evaluation/`](src/sage/evaluation), [`eval/metrics.py`](eval/metrics.py) | §IV, §IX, Supp. B.3 |
| **G — Govern** | Ten-class failure taxonomy; STRIDE threat models for MCP and A2A | [`sage/evaluation/failure_taxonomy.py`](src/sage/evaluation/failure_taxonomy.py), [`sage/evaluation/pathology_benchmarks.py`](src/sage/evaluation/pathology_benchmarks.py), [`sage/security/threat_validator.py`](src/sage/security/threat_validator.py) | §VIII, §XI, Supp. D |
| **E — Enforce** | Adaptive Stability Controller (ASC); five bounded interventions; four control policies | [`sage/stability/controller.py`](src/sage/stability/controller.py), [`sage/stability/interventions.py`](src/sage/stability/interventions.py), [`sage/stability/predictor.py`](src/sage/stability/predictor.py) | §III.E, §IX.D, Supp. A.5.4–A.5.5 |

*Govern is currently split across `evaluation/` (failure taxonomy + pathology benchmarks) and `security/` (STRIDE threat validator). See [TODO.md](TODO.md) for the planned consolidation under `sage/governance/`.*

---

<img width="760" height="1040" alt="stack_v2" src="https://github.com/user-attachments/assets/829695dd-5afd-4412-8d81-162c2c12812b" />

---
<img width="2090" height="966" alt="System architecture" src="https://github.com/user-attachments/assets/c6960d1d-d4ab-4954-8664-3b747187ea54" />

---

## Features

- **Stability pillar**: Oscillation detection, progress monotonicity, observation fidelity, goal-drift score
- **Adaptive Stability Controller (ASC)**: Closed-loop controller with 4 policies (NoControl, FixedSchedule, Threshold, Predictive) and 5 bounded interventions; raises long-horizon completion from 2% to 83% (E4)
- **Failure Predictor**: Calibrated logistic regression predicting task failure k turns ahead (AUC = 0.752 combined model, E5)
- **Assess pillar**: CNSR metric, long-horizon evaluation, goal drift, incident tracking, autonomy taxonomy
- **Govern pillar**: Ten-class failure taxonomy with detectors and mitigations; STRIDE threat validator for MCP and A2A
- **Agent architectures**: ReAct, Chain-of-Thought, multi-agent supervisor and sequential pipelines
- **Memory systems**: Buffer memory (working) and vector memory (semantic long-term)
- **Tool integration**: Flexible tool registry with schema validation and sandboxing
- **Protocol support**: MCP (Model Context Protocol) and A2A (Agent-to-Agent) interfaces
- **Research experiments**: Full reproducible suite (CNSR multi-task, Proposition 1 violations A1/A2/A3, LLM-as-Judge bias, E4 closed-loop ablation, E5 predictive validation)
- **Observability**: Built-in tracing and monitoring with LangSmith support

---

## Installation

```bash
# Clone the repository
git clone https://github.com/MHHamdan/SAGE.git
cd SAGE

# Basic installation
pip install -e .

# With all optional dependencies (recommended)
pip install -e ".[all]"

# Research experiments only (no heavy LangChain deps needed)
pip install -e ".[experiments]"

# Development installation
pip install -e ".[dev]"
```

> **Backwards compatibility:** `pip install agentic-ai-toolkit` and `import agentic_toolkit` continue to work via a deprecation shim. The shim aliases every `agentic_toolkit.*` submodule to the same module object as `sage.*` (so class identity and `isinstance` checks are preserved) and emits a `DeprecationWarning` pointing you at the new imports. It will be removed in a future release.

### Experiment dependencies

```bash
pip install numpy scipy pandas litellm sentence-transformers
# litellm is optional — experiments fall back to seeded simulation on API errors
```

---

## Quick Start

### Basic ReAct Agent

```python
from sage.core import LLMClient
from sage.agents import ReActAgent
from langchain_core.tools import tool

llm = LLMClient(model="gpt-4o-mini", api_key="your-api-key")

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

agent = ReActAgent(
    name="assistant",
    llm=llm,
    tools=[search, calculate],
    instructions="You are a helpful assistant that can search and calculate.",
)
result = agent.run("What is 25 * 4 and search for Python tutorials")
print(result)
```

### Evaluation — CNSR (Assess pillar)

```python
from sage.evaluation import calculate_cnsr, evaluate_agent

# Cost-Normalized Success Rate: Success Rate / Mean Cost per Task
cnsr = calculate_cnsr(successes=80, total_tasks=100, total_cost=50.0)
print(f"CNSR: {cnsr:.2f}")   # 1.60

result = evaluate_agent(successes=80, total_tasks=100, total_cost=50.0)
print(f"Success Rate: {result.success_rate:.2%}")
print(f"Mean Cost: ${result.mean_cost:.2f}")
print(f"CNSR: {result.cnsr:.2f}")
```

### Stability Monitor (Stabilize pillar)

```python
import numpy as np
from sage.monitoring.stability_monitor import (
    StabilityMonitor, create_stability_monitor
)

# Create monitor with goal embedding
monitor = create_stability_monitor(
    goal_text="Complete the file editing task",
    embedding_fn=your_embed_fn,
    similarity_threshold=0.9,
    oscillation_window=10,
    oscillation_bound=3,
)

# Track each agent step
for step in agent_steps:
    status = monitor.track_state(
        state_embedding=step.state_emb,
        action=step.action,
        observation=step.observation,
    )
    if status.oscillation.oscillating:
        print(f"Warning: oscillation detected at step {status.step}")

report = monitor.get_stability_report()
print(f"Total steps: {report.total_steps}")
print(f"Recommendations: {report.recommendations}")
```

### Adaptive Stability Controller (Enforce pillar)

```python
from sage.stability.controller import AdaptiveStabilityController
from sage.stability.interventions import (
    GoalReanchor, ContextCompress, ForceReplan,
    SchemaValidatedRetry, HumanEscalate,
)

controller = AdaptiveStabilityController(
    policy="threshold",  # NoControl | FixedSchedule | Threshold | Predictive
    interventions=[GoalReanchor(), ContextCompress(), ForceReplan(),
                   SchemaValidatedRetry(), HumanEscalate()],
)
```

### Memory Systems

```python
from sage.memory import BufferMemory, VectorMemory

buffer = BufferMemory(max_items=10)
buffer.add_user_message("Hello!")
buffer.add_ai_message("Hi there! How can I help?")

vector_memory = VectorMemory(
    embedding_model="text-embedding-3-small",
    persist_directory="./memory_store"
)
vector_memory.add("Python is a high-level programming language")
results = vector_memory.get("What programming languages are popular?", k=2)
```

### Multi-Agent Pipeline

```python
from sage.agents import SequentialPipeline, SupervisorAgent, ReActAgent

researcher = ReActAgent(name="researcher", llm=llm, tools=[search_tool])
analyst    = ReActAgent(name="analyst",    llm=llm, tools=[analyze_tool])
writer     = ReActAgent(name="writer",     llm=llm, tools=[format_tool])

pipeline = SequentialPipeline(
    name="content_pipeline",
    agents=[researcher, analyst, writer],
)
result = pipeline.run("Create a report on renewable energy trends")
```

---

## Research Experiments

This implementation includes the full reproducible experiment suite from the TAI submission. All experiments use deterministic seeded pseudo-randomness and cache API responses under `results/cache/`.

### Running experiments

```bash
# Task 1 — CNSR multi-task (7 models × 3 task types × 3 seeds)
python experiments/cnsr_multitask.py
# → results/cnsr_multitask.csv  results/cnsr_table.tex

# Task 2 — Proposition 1 violation experiments
python experiments/exp_obs_fidelity.py   # A1: observation fidelity injection
python experiments/exp_progress_mono.py  # A2: progress monotonicity stall
python experiments/exp_context_noise.py  # A3: context noise / goal drift
# → results/exp_a1.csv  results/exp_a2.csv  results/exp_a3.csv

# Task 3 — LLM-as-Judge bias measurement
python experiments/judge_bias.py
# → results/judge_bias.csv  results/judge_bias.tex

# Task 4 — Generate all LaTeX table fragments
python scripts/generate_latex.py
# → results/table_fragments.tex  (+ 6 individual .tex files)

# E4 — Closed-loop ASC ablation (50 tasks × 4 conditions × 3 seeds)
python experiments/e4_closed_loop.py --seed 42
# → results/e4_closed_loop/summary.csv  REPORT.md  figures/  MANIFEST.json

# E5 — Predictive monitor validation (300 tasks, 5-fold CV, k ∈ {3,5,10})
python experiments/e5_predictive_validation.py --seed 42
# → results/e5_predictive/summary.csv  REPORT.md  predictors/  figures/  MANIFEST.json
```

### Experiment A1 — Observation Fidelity Injection

Measures the effect of corrupted tool responses on a ReAct file-editing agent. The oscillation detector provides early warning before task-level failures manifest.

| Injection Rate | Success Rate | Oscillation Detection |
|---|---|---|
| 0.0 | 100% | 0% |
| 0.1 | 100% | 10% |
| 0.2 | 100% | **25%** ← early warning |
| 0.4 | 90% | 60% |

### Experiment A2 — Progress Monotonicity

Tests deadlock detection on an 8-step scheduling task under stall injection. The bounded oscillation condition (k=5, B=3) detects deadlocks within a mean of **7.3 turns** at stall_prob=0.5.

| Stall Prob | Deadlock Rate | Mean Turns to Detection |
|---|---|---|
| 0.00 | 0% | — |
| 0.25 | 0% | — |
| 0.50 | 15% | 7.3 |

### Experiment A3 — Context Noise / Goal Drift

Goal drift measured over 50 turns with varying re-anchoring intervals. Re-anchoring every k=10 turns reduces drift by **59.7%** and raises task completion from 0% to 100%.

| Re-anchor k | Drift at t=50 | Completion |
|---|---|---|
| 5 | 0.149 | 100% |
| 10 | 0.197 | 100% |
| 20 | 0.425 | 20% |
| None | 0.490 | **0%** |

### CNSR Multi-Task Results (Table V)

Kendall's τ between success-rate rank and CNSR rank: **−0.429** (code), **−0.238** (web), **−0.619** (research). GPT-4-Turbo ranks 1st by SR but 7th by CNSR; Gemini-1.5-Flash ranks 1st by CNSR at ~30× lower cost.

| Config | Code CNSR | Code SR | Web CNSR | Web SR | Research CNSR | Research SR |
|---|---|---|---|---|---|---|
| GPT-4-Turbo | 21.1 ± 2.1 | 76% | 28.2 ± 2.8 | 57% | 16.1 ± 0.2 | 82% |
| Claude-3.5-Sonnet | 50.6 ± 6.6 | 73% | 78.4 ± 8.8 | 62% | 39.5 ± 2.0 | 81% |
| LLaMA-3-70B | 177.9 ± 16.2 | 53% | 251.1 ± 65.2 | 45% | 161.4 ± 8.4 | 65% |
| GPT-3.5-Turbo | 512.0 ± 100.6 | 53% | 642.4 ± 9.2 | 40% | 382.2 ± 41.3 | 57% |
| **Gemini-1.5-Flash** | **656.1 ± 56.4** | 57% | **1018.2 ± 168.9** | 54% | **546.4 ± 22.1** | 69% |
| Mistral-7B | 173.7 ± 56.6 | 37% | 228.6 ± 36.5 | 31% | 163.5 ± 38.5 | 46% |
| Ensemble (top-3) | 114.1 ± 21.6 | 56% | 151.1 ± 30.4 | 45% | 102.5 ± 11.6 | 69% |

### Experiment E4 — Closed-Loop ASC Ablation (seed=42, 150 evals per condition)

Headline: predictive closed-loop control raises 50-turn task completion from 2% to 83%. All comparisons significant (McNemar *p* < 0.0001, Holm-Bonferroni corrected).

| Condition | Completion Rate | 95% CI | CNSR | Mean Cost |
|---|---|---|---|---|
| NoControl | 2.0% | [0.0%, 4.7%] | 0.020 | $1.000 |
| FixedSchedule (k=10) | 36.7% | [29.3%, 44.7%] | 0.349 | $1.050 |
| ThresholdController | **89.3%** | [84.0%, 94.0%] | 0.631 | $1.416 |
| PredictiveController | 83.3% | [77.3%, 88.7%] | **0.640** | $1.302 |

### Experiment E5 — Predictive Monitor Validation (seed=42, 300 tasks, k=5)

Each monitor signal evaluated as a k=5-step-ahead failure predictor; combined model outperforms all single signals.

| Predictor | AUC-ROC | 95% CI | Significant (H5.1) |
|---|---|---|---|
| drift_only | 0.609 | [0.594, 0.622] | ✓ p<0.0001 |
| oscillation_only | 0.589 | [0.576, 0.602] | ✓ p<0.0001 |
| fidelity_only | 0.495 | [0.479, 0.510] | ✗ p=0.760 |
| **combined** | **0.752** | [0.741, 0.763] | — |
| MLP (32 hidden) | 0.516 | [0.502, 0.529] | — |

### LLM-as-Judge Bias Mitigation

| Bias Type | Before Mitigation | After Mitigation | Reduction |
|---|---|---|---|
| Self-preference Δ | 0.540 | 0.130 | **75.9%** |
| Position bias | 0.253 | 0.101 | **60.0%** |
| Verbosity bias \|r\| | 0.137 | 0.048 | **65.0%** |

---

## Project Structure

```
SAGE/
├── src/sage/                       # Installable Python package
│   ├── agents/                     # ReAct, multi-agent, supervisor
│   ├── benchmarks/                 # SWE-Bench, HotpotQA, AgentBench adapters
│   ├── core/                       # Base agent, LLM client, config, cost tracking
│   ├── evaluation/                 # Assess + Govern code
│   │   ├── metrics.py              # compute_cnsr(), TaskCostBreakdown, MetricsCollector
│   │   ├── goal_drift.py           # goal_drift_score()
│   │   ├── long_horizon.py         # LongHorizonEvaluator
│   │   ├── incident_tracker.py     # IncidentTracker
│   │   ├── cnsr_benchmark.py       # CNSRBenchmark, Pareto analysis
│   │   ├── autonomy_validator.py   # Autonomy taxonomy validator
│   │   ├── failure_taxonomy.py     # G: 10-class failure taxonomy + detectors
│   │   └── pathology_benchmarks.py # G: pathology benchmark runner
│   ├── stability/                  # E (and S): closed-loop control
│   │   ├── controller.py           # AdaptiveStabilityController
│   │   ├── interventions.py        # 5 bounded interventions
│   │   ├── predictor.py            # Calibrated failure predictor
│   │   └── traces.py
│   ├── monitoring/                 # S: StabilityMonitor (oscillation, drift, fidelity)
│   ├── security/                   # G: STRIDE threat validator (MCP, A2A)
│   ├── human_oversight/            # Approval flows, escalation, audit trails
│   ├── learning/                   # Deployment loop, feedback, experience replay
│   ├── memory/                     # Buffer, vector, episodic memory
│   ├── planning/                   # Reactive, deliberative, hybrid, HTN planners
│   ├── protocols/                  # MCP client/server, A2A communication
│   ├── skills/                     # Skill registry, versioning, selection
│   ├── tools/                      # Tool registry, sandboxing, permissions
│   ├── verification/               # Plan validator, policy engine, guarded executor
│   └── __init__.py
│
├── src/agentic_toolkit/            # Deprecation shim for the legacy name
│
├── experiments/                    # TAI paper experiments (A1–A3, CNSR, E4, E5)
├── eval/                           # Lightweight metrics shim (no heavy deps)
├── scripts/                        # generate_latex.py, env_report.py
├── tests/                          # Test suite (sage.*)
├── examples/                       # Quick-start examples + use-cases
├── configs/                        # YAML experiment configurations
├── dashboard/                      # FastAPI + React monitoring dashboard
├── pyproject.toml                  # Package metadata (v1.2.0, sage-framework)
├── CHANGELOG.md
├── TODO.md
└── requirements.txt
```

---

## Configuration

### Environment Variables

```bash
# Required for cloud LLM calls
OPENAI_API_KEY=sk-your-api-key

# Optional
ANTHROPIC_API_KEY=your-anthropic-key
TOGETHER_API_KEY=your-together-key   # for LLaMA / Mistral via Together AI
GEMINI_API_KEY=your-gemini-key

# Optional: Observability
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=sage-framework

# LiteLLM (used by experiments) picks up all of the above automatically
```

### Programmatic Configuration

```python
from sage.core import Config, LLMConfig, MemoryConfig

config = Config(
    llm=LLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=4096),
    memory=MemoryConfig(buffer_size=20, vector_collection="default"),
)
```

---

## Architecture

### System Overview

<img width="1800" height="1300" alt="system_architecture_v2" src="https://github.com/user-attachments/assets/c12d9124-b914-40f8-81fd-481cdffde0b5" />

### Component Architecture

<img width="1360" height="960" alt="class_diagram (1)" src="https://github.com/user-attachments/assets/c32f3dfe-f6a4-4426-bdc4-147f19eb9390" />

### Control Loop

<img width="309" height="838" alt="agent_cycle" src="https://github.com/user-attachments/assets/378ee650-3b4b-457c-bc3b-9c2d734a54ed" />

### Evaluation Metrics

| Metric | Formula | Use Case |
|---|---|---|
| Success Rate | successes / total | Basic performance |
| CNSR | SR / mean_cost | Cost-efficiency ranking |
| Goal Drift | 1 − cosine_sim(goal, state) | Long-horizon alignment |
| Oscillation | overlap_ratio in window k | Stuck-agent detection |
| Incident Rate | incidents / tasks | Safety monitoring |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sage --cov-report=term-missing

# Run stability monitor tests only (32 tests, no API keys needed)
pytest tests/monitoring/test_stability_monitor.py -v

# Run experiment integration tests
pytest tests/monitoring/ -v

# Run specific test category
pytest tests/evaluation/ -v
```

### Test coverage summary

| Module | Tests | Status |
|---|---|---|
| Stability monitor | 32 | ✅ All passing |
| CNSR benchmark | 12 | ✅ All passing |
| Goal drift | 8 | ✅ All passing |
| Incident tracker | 6 | ✅ All passing |
| Cost model | 10 | ✅ All passing |
| Long-horizon evaluator | 8 | ✅ All passing |
| Autonomy validator | 14 | ✅ All passing |

---

## Reproducibility

All research experiments are fully reproducible:

```bash
# Seeds 0, 1, 2 — no API keys required (falls back to seeded simulation)
python experiments/cnsr_multitask.py --seeds 0 1 2
python experiments/exp_obs_fidelity.py --seed 42
python experiments/exp_progress_mono.py --seed 42
python experiments/exp_context_noise.py --seed 42
python experiments/judge_bias.py --seed 2024
python scripts/generate_latex.py
```

API responses are cached under `results/cache/` (MD5-keyed JSON). On a cache miss or API error the experiments fall back to a seeded statistical simulator that reproduces the same distributions.

---

## Advanced Usage

### Human Oversight

```python
from sage.human_oversight import ApprovalHandler, RiskLevel

handler = ApprovalHandler(default_timeout=300, auto_reject_on_timeout=True)
request = handler.create_request(
    action="deploy_model",
    context={"model": "gpt-4", "environment": "production"},
    risk_level=RiskLevel.HIGH,
)
result = await handler.wait_for_approval(request.request_id)
if result.approved:
    deploy_model()
```

### Deployment Loop

```python
from sage.learning import DeploymentLoop, DeploymentConfig

config = DeploymentConfig(
    evaluation_interval=100,
    rollback_threshold=0.6,
    enable_auto_rollback=True,
)
loop = DeploymentLoop(agent=my_agent, config=config)

async for update in loop.run(tasks=task_stream):
    if update.event_type == "evaluation":
        print(f"Success rate: {update.success_rate:.2%}")
```

### Protocol Integration

```python
from sage.protocols.mcp import MCPClient
from sage.protocols.a2a import A2AClient, AgentCard

mcp_client = MCPClient(server_url="http://localhost:8080")
tools = mcp_client.list_tools()
result = mcp_client.call_tool("search", {"query": "AI agents"})

agent_card = AgentCard(
    name="my-agent",
    capabilities=["search", "summarize"],
    endpoint="http://localhost:9000"
)
```

---

## Citation

If you use SAGE or the experimental results in your research, please cite:

```bibtex
@article{hamdan2025sage,
  title   = {SAGE: A Stabilize--Assess--Govern--Enforce Framework for
             Deployment-Ready LLM-Based Autonomous Agents},
  author  = {Hamdan, Mohammed H.},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2025},
  note    = {Manuscript TAI-2025-Dec-R-02684 (under revision)}
}
```

---

## What changed in v1.2.0

- **Rebrand to SAGE**: the framework is now named SAGE (Stabilize, Assess, Govern, Enforce). The Python import is `sage`; the PyPI distribution is `sage-framework`. The GitHub repo moved to [`MHHamdan/SAGE`](https://github.com/MHHamdan/SAGE).
- **Backwards-compatible shim**: `import agentic_toolkit` and `pip install agentic-ai-toolkit` continue to work with a `DeprecationWarning`.
- **No behavioural changes**: no algorithm, threshold, default argument, or numerical claim has changed. Empirical results (A1, A2, A3, CNSR, E4, E5) reproduce identically.
- See [CHANGELOG.md](CHANGELOG.md) for the full history.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of:
- [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), and [Together AI](https://together.ai/) APIs
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [LiteLLM](https://litellm.ai/) for unified model API access
