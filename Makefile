# Makefile for SAGE (Stabilize, Assess, Govern, Enforce)
# Paper-grade reproducibility targets

.PHONY: all install test lint format clean results paper-assets env-report \
        run-examples run-benchmarks validate help

# Default target
all: help

# =============================================================================
# Setup
# =============================================================================

install:
	@echo "Installing sage-framework..."
	pip install -e ".[dev]"

install-dev:
	@echo "Installing development dependencies..."
	pip install -e ".[dev,test]"

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running all tests..."
	PYTHONPATH=src pytest tests/ -v --tb=short

test-unit:
	@echo "Running unit tests..."
	PYTHONPATH=src pytest tests/ -v --tb=short -m "not integration"

test-integration:
	@echo "Running integration tests..."
	PYTHONPATH=src pytest tests/ -v --tb=short -m "integration"

test-equations:
	@echo "Testing equation consistency..."
	PYTHONPATH=src pytest tests/test_equation_consistency.py -v

test-security:
	@echo "Testing security enforcement..."
	PYTHONPATH=src pytest tests/tools/test_security_enforcement.py -v

test-protocols:
	@echo "Testing protocol implementation..."
	PYTHONPATH=src pytest tests/protocols/ -v

test-coverage:
	@echo "Running tests with coverage..."
	PYTHONPATH=src pytest tests/ --cov=src/sage --cov-report=html --cov-report=term

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "Running linters..."
	ruff check src/ tests/
	mypy src/sage --ignore-missing-imports

format:
	@echo "Formatting code..."
	ruff format src/ tests/
	ruff check --fix src/ tests/

# =============================================================================
# Results Generation (Paper Assets)
# =============================================================================

results: paper-assets

paper-assets: tables figures
	@echo "All paper assets generated."
	@echo "See paper_assets/ for outputs."

tables:
	@echo "Generating tables..."
	PYTHONPATH=src python paper_assets/scripts/generate_tables.py
	@echo "Tables generated in paper_assets/tables/"

figures:
	@echo "Generating figures..."
	PYTHONPATH=src python paper_assets/scripts/generate_figures.py
	@echo "Figures generated in paper_assets/figures/"

# =============================================================================
# Reproducibility
# =============================================================================

env-report:
	@echo "Generating environment report..."
	python scripts/env_report.py --format text
	@echo ""
	@echo "For JSON output: python scripts/env_report.py --format json"

env-report-json:
	@echo "Generating JSON environment report..."
	python scripts/env_report.py --format json --output env_report.json
	@echo "Report saved to env_report.json"

validate:
	@echo "Validating results reproducibility..."
	@echo "Running determinism check..."
	PYTHONPATH=src python -c "\
from sage.evaluation import ReasoningBenchmark; \
from sage.core.seeding import set_global_seed; \
set_global_seed(42); \
b1 = ReasoningBenchmark(); \
tasks1 = [t.task_id for t in b1.get_tasks()]; \
set_global_seed(42); \
b2 = ReasoningBenchmark(); \
tasks2 = [t.task_id for t in b2.get_tasks()]; \
assert tasks1 == tasks2, 'Determinism check failed!'; \
print('Determinism check passed!')"

# =============================================================================
# Examples
# =============================================================================

run-examples:
	@echo "Running all examples..."
	@for f in examples/*.py; do \
		echo "Running $$f..."; \
		PYTHONPATH=src python "$$f" || exit 1; \
	done
	@echo "All examples completed."

example-basic:
	@echo "Running basic agent example..."
	PYTHONPATH=src python examples/01_basic_agent.py

example-planning:
	@echo "Running planning example..."
	PYTHONPATH=src python examples/02_planning.py

example-skills:
	@echo "Running skills example..."
	PYTHONPATH=src python examples/03_skills.py

example-evaluation:
	@echo "Running evaluation example..."
	PYTHONPATH=src python examples/04_evaluation.py

example-security:
	@echo "Running security policy demo..."
	PYTHONPATH=src python examples/05_security_policy_demo.py

example-protocols:
	@echo "Running protocol demo..."
	PYTHONPATH=src python examples/06_protocols_demo.py

# =============================================================================
# Benchmarks
# =============================================================================

run-benchmarks:
	@echo "Running full benchmark suite..."
	@echo "This may take ~30 minutes..."
	PYTHONPATH=src SEED=42 python -c "\
from sage.evaluation import EvaluationHarness, EvaluationConfig, create_benchmark_suite; \
config = EvaluationConfig(seed=42, save_results=True); \
harness = EvaluationHarness(config); \
for b in create_benchmark_suite(): harness.register_benchmark(b); \
harness.register_agent(lambda t: {'answer': 'test'}); \
result = harness.run(); \
print(f'CNSR: {result.aggregate_metrics.get(\"cnsr\", 0):.2f}')"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/*.egg-info build dist
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

clean-results:
	@echo "Cleaning generated results..."
	rm -f paper_assets/tables/*.tex
	rm -f paper_assets/figures/*.pdf paper_assets/figures/*.png
	@echo "Results cleaned."

clean-all: clean clean-results
	@echo "Full cleanup complete."

# =============================================================================
# Docker (optional)
# =============================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t sage-framework .

docker-run:
	@echo "Running in Docker..."
	docker run -it --rm sage-framework

# =============================================================================
# CI Targets
# =============================================================================

ci: lint test-coverage
	@echo "CI checks complete."

ci-fast: lint test-unit
	@echo "Fast CI checks complete."

# =============================================================================
# Help
# =============================================================================

help:
	@echo "SAGE Framework - Makefile Targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install the package"
	@echo "  make install-dev      Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-equations   Test equation consistency"
	@echo "  make test-security    Test security enforcement"
	@echo "  make test-protocols   Test protocol implementation"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (ruff, mypy)"
	@echo "  make format           Format code with ruff"
	@echo ""
	@echo "Paper Assets:"
	@echo "  make paper-assets     Generate all tables and figures"
	@echo "  make tables           Generate LaTeX tables"
	@echo "  make figures          Generate PDF/PNG figures"
	@echo "  make results          Alias for paper-assets"
	@echo ""
	@echo "Reproducibility:"
	@echo "  make env-report       Generate environment report"
	@echo "  make validate         Validate determinism"
	@echo ""
	@echo "Examples:"
	@echo "  make run-examples     Run all examples"
	@echo "  make example-basic    Run basic agent example"
	@echo "  make example-security Run security demo"
	@echo "  make example-protocols Run protocol demo"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make run-benchmarks   Run full benchmark suite"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Clean build artifacts"
	@echo "  make clean-results    Clean generated results"
	@echo "  make clean-all        Full cleanup"
	@echo ""
	@echo "CI:"
	@echo "  make ci               Run full CI checks"
	@echo "  make ci-fast          Run fast CI checks"
