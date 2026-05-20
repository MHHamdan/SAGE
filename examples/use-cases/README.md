# Use Case Examples

Working code examples demonstrating the Agentic AI Toolkit for common evaluation scenarios.

**All examples use LOCAL Ollama models for REAL LLM inference with actual token tracking.**

## Minimum Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 8GB | 16GB |
| Disk Space | 4GB (for models) | 10GB |
| GPU | Optional | NVIDIA 8GB+ VRAM |
| Ollama | v0.1.0+ | Latest |

**Note:** Examples can run on CPU-only systems, but will be slower. GPU acceleration significantly improves inference speed.

## Prerequisites

### 1. Install Ollama
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

### 2. Pull a Model
```bash
# Recommended: Small, fast model
ollama pull gemma2:2b

# Alternative options
ollama pull phi3:latest      # Fast, good reasoning
ollama pull mistral:latest   # Balanced
ollama pull llama3.1:8b      # Higher quality
```

### 3. Start Ollama Server
```bash
ollama serve
```

## Examples

| Example | Description | Components Used |
|---------|-------------|-----------------|
| `enterprise_agent_evaluation.py` | Customer service bot assessment | CNSR, Failure Taxonomy |
| `cost_optimization_analysis.py` | Multi-model cost comparison | 4-Component Cost Model |
| `coding_assistant_evaluation.py` | Code assistant quality check | Stability Analysis, Hallucination Detection |
| `research_comparison.py` | Architecture comparison for papers | Statistical Analysis, Autonomy Framework |
| `multi_agent_evaluation.py` | Supervisor-worker system analysis | Distributed Costs, Communication Patterns |
| `safety_compliance_check.py` | Pre-deployment safety verification | Compliance Reports, Risk Assessment |

## Quick Start

```bash
# Make sure Ollama is running
ollama serve &

# Run any example
python examples/use-cases/enterprise_agent_evaluation.py
```

## What Each Example Demonstrates

### Enterprise Agent Evaluation
- **Use case**: Evaluating customer service chatbots before deployment
- **Real LLM calls**: Tests agent responses to customer queries
- **Metrics**: CNSR, success rate, failure pathology screening
- **Output**: Deployment recommendation (APPROVED/CONDITIONAL)

### Cost Optimization Analysis
- **Use case**: Comparing models to find optimal cost/performance
- **Real LLM calls**: Runs same tasks on different model sizes
- **Metrics**: Token usage, latency, CNSR per model
- **Output**: Best model recommendation, cost forecast

### Coding Assistant Evaluation
- **Use case**: Assessing code completion tools
- **Real LLM calls**: Code generation, explanation, debugging tasks
- **Metrics**: Hallucination rate, session stability, CNSR
- **Output**: Quality assessment with specific improvements

### Research Comparison
- **Use case**: Academic benchmarking of agent architectures
- **Real LLM calls**: Standardized benchmark tasks
- **Metrics**: Statistical analysis (mean, std, effect size)
- **Output**: Publication-ready results table

### Multi-Agent Evaluation
- **Use case**: Evaluating coordinator-worker systems
- **Real LLM calls**: Task delegation and execution
- **Metrics**: Per-agent costs, communication overhead
- **Output**: System efficiency analysis

### Safety Compliance Check
- **Use case**: Pre-deployment safety verification
- **Real LLM calls**: Safety boundary testing
- **Metrics**: Pass/fail on safety requirements
- **Output**: Compliance report with risk assessment

## Token Tracking

All examples track **real token usage** from Ollama:
- `prompt_tokens`: Tokens in the input
- `completion_tokens`: Tokens generated
- `total_tokens`: Sum of both
- Estimated costs based on model-specific rates

## GPU Memory Notes

If you encounter GPU memory errors:
1. Close other GPU-intensive applications
2. Use a smaller model (gemma2:2b is recommended)
3. Run examples one at a time

## Helper Module

The `ollama_helper.py` module provides:
- `OllamaClient`: Wrapper for Ollama API with token tracking
- `check_ollama_ready()`: Verify Ollama is working
- `select_best_model()`: Auto-select available model
- `estimate_cost()`: Calculate inference costs

## Supported Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| gemma2:2b | 1.6GB | Fast | Good | Quick demos |
| phi3:latest | 2.2GB | Fast | Good | Reasoning tasks |
| llama3.2:3b | 2.0GB | Fast | Good | General use |
| mistral:latest | 4.4GB | Medium | Better | Production |
| llama3.1:8b | 4.9GB | Slower | High | Quality-critical |

## Output Example

```
============================================================
ENTERPRISE AI AGENT EVALUATION
Customer Service Agent - Pre-Deployment Assessment
============================================================

Checking Ollama readiness with gemma2:2b...

Agent Configuration:
  Model: gemma2:2b
  Escalation Rate: 10%

Running 10 tasks with real LLM inference...
  Completed 5/10 tasks...
  Completed 10/10 tasks...

========================================
Tasks evaluated: 10
Total tokens used: 3,245
Success Rate: 80.0%
Mean Cost per Task: $0.0324
CNSR: 24.69

DEPLOYMENT RECOMMENDATION
Recommendation: APPROVED
Details: Agent meets safety and cost requirements.

============================================================
EVALUATION COMPLETE
============================================================
Total LLM calls: 13
Total tokens: 3,892
Estimated cost: $0.0389
```

## Customization

### Using Different Models

Each example auto-selects the best available model. To force a specific model:

```python
# In any example, modify the model selection:
model = "mistral:latest"  # Instead of select_best_model()
```

### Custom Cost Rates

Edit `ollama_helper.py` to adjust token rates:

```python
TOKEN_RATES = {
    "gemma2:2b": 0.00001,      # Very cheap
    "phi3:latest": 0.00001,
    "mistral:latest": 0.00003,
    "llama3.1:8b": 0.00005,
    "custom:model": 0.00002,   # Add your model
}
```

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/version

# Check available models
curl http://localhost:11434/api/tags
```

If Ollama won't start:
```bash
# Check if another process is using the port
lsof -i :11434

# Kill any stuck Ollama processes
pkill ollama

# Restart
ollama serve
```

### GPU Out of Memory
```
Error: cudaMalloc failed: out of memory
```

**Solutions:**
1. Free GPU memory by closing other applications
2. Use a smaller model (`gemma2:2b` recommended)
3. Run examples one at a time
4. Check GPU usage: `nvidia-smi`

```bash
# Clear GPU memory
nvidia-smi --gpu-reset

# Or use CPU-only mode (slower but works)
CUDA_VISIBLE_DEVICES="" python examples/use-cases/enterprise_agent_evaluation.py
```

### Model Not Found
```bash
# List installed models
ollama list

# Pull missing model
ollama pull gemma2:2b

# Verify model works
ollama run gemma2:2b "Hello"
```

### Slow First Run
The first run of each example downloads model weights (~1-5GB). This is normal and only happens once. Subsequent runs use cached weights.

### Connection Refused
```
Error: Connection refused (http://localhost:11434)
```

**Solutions:**
1. Ensure Ollama is running: `ollama serve`
2. Check firewall settings
3. Verify the port is correct (default: 11434)

### Python Import Errors
```bash
# Ensure you're in the correct directory
cd /path/to/SAGE

# Install dependencies
pip install requests

# Run from project root
python examples/use-cases/enterprise_agent_evaluation.py
```

### Timeout Errors
If inference is timing out:
1. Model may still be loading (first run)
2. System may be under heavy load
3. Try increasing timeout in `ollama_helper.py`

## Documentation

See the corresponding documentation for detailed guides:
- [Enterprise Agents](../../docs/use-cases/enterprise-agents.md)
- [Coding Assistants](../../docs/use-cases/coding-assistants.md)
- [Research Evaluation](../../docs/use-cases/research-evaluation.md)
- [Multi-Agent Systems](../../docs/use-cases/multi-agent-systems.md)
- [Safety & Compliance](../../docs/use-cases/safety-compliance.md)
- [Cost Optimization](../../docs/use-cases/cost-optimization.md)
