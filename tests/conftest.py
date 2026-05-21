"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a test response.",
        "tokens": {"input": 10, "output": 20},
        "model": "test-model",
    }


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return {
        "task_id": "test_001",
        "description": "Test task",
        "input": "Test input",
        "expected_output": "Test output",
    }


@pytest.fixture
def sample_plan():
    """Sample plan for testing."""
    return {
        "plan_id": "plan_001",
        "goal": "Complete test task",
        "steps": [
            {"step_id": "step_1", "action": "analyze", "description": "Analyze input"},
            {"step_id": "step_2", "action": "process", "description": "Process data"},
            {"step_id": "step_3", "action": "output", "description": "Generate output"},
        ],
    }


@pytest.fixture
def sample_skill():
    """Sample skill for testing."""
    return {
        "name": "test_skill",
        "description": "A test skill",
        "version": "1.0.0",
        "trust_score": 0.8,
    }
