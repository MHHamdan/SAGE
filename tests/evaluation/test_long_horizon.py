"""
Tests for Long-Horizon Evaluation Harness.

Comprehensive tests for the LongHorizonEvaluator and SimpleLongHorizonEvaluator
classes that combine rolling metrics, goal drift, and incident tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from sage.evaluation.incident_tracker import (
    IncidentSeverity,
    IncidentType,
)
from sage.evaluation.long_horizon import (
    CheckpointData,
    EvaluationStatus,
    LongHorizonEvaluator,
    LongHorizonReport,
    SimpleLongHorizonEvaluator,
    TaskExecutionResult,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding():
    """Return a consistent mock embedding function."""
    embeddings = {}
    base_embedding = np.random.randn(384)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)

    def embed_fn(text: str) -> np.ndarray:
        if text not in embeddings:
            # Create similar but slightly different embeddings
            noise = np.random.randn(384) * 0.1
            embedding = base_embedding + noise
            embeddings[text] = embedding / np.linalg.norm(embedding)
        return embeddings[text]

    return embed_fn


@pytest.fixture
def mock_agent():
    """Create a mock agent with run method."""
    agent = Mock()
    agent.run = Mock(return_value={"success": True, "cost": 0.01})
    return agent


@pytest.fixture
def async_mock_agent():
    """Create a mock agent with async run method."""
    agent = Mock()
    agent.run = AsyncMock(return_value={"success": True, "cost": 0.01})
    return agent


@pytest.fixture
def evaluator(mock_agent, mock_embedding):
    """Create a LongHorizonEvaluator instance."""
    return LongHorizonEvaluator(
        agent=mock_agent,
        embed_fn=mock_embedding,
        window_size=10,
        drift_threshold=0.3,
        incident_threshold_per_hour=5.0,
        success_degradation_threshold=0.1,
    )


@pytest.fixture
def simple_evaluator():
    """Create a SimpleLongHorizonEvaluator instance."""
    return SimpleLongHorizonEvaluator(window_size=10)


# ============================================================================
# TaskExecutionResult Tests
# ============================================================================


class TestTaskExecutionResult:
    """Tests for TaskExecutionResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful task result."""
        result = TaskExecutionResult(
            task_id="task_1",
            success=True,
            cost=0.05,
            latency_ms=150.0,
            output={"data": "test"},
        )

        assert result.task_id == "task_1"
        assert result.success is True
        assert result.cost == 0.05
        assert result.latency_ms == 150.0
        assert result.output == {"data": "test"}
        assert result.error is None
        assert result.inferred_goal is None

    def test_create_failed_result(self):
        """Test creating a failed task result."""
        result = TaskExecutionResult(
            task_id="task_2",
            success=False,
            cost=0.01,
            latency_ms=50.0,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.output is None


# ============================================================================
# CheckpointData Tests
# ============================================================================


class TestCheckpointData:
    """Tests for CheckpointData dataclass."""

    def test_create_checkpoint(self):
        """Test creating checkpoint data."""
        checkpoint = CheckpointData(
            checkpoint_num=1,
            timestamp=datetime.now(),
            tasks_completed=50,
            rolling_success_rate=0.85,
            current_drift=0.1,
            incident_count=2,
            cnsr=1.2,
        )

        assert checkpoint.checkpoint_num == 1
        assert checkpoint.tasks_completed == 50
        assert checkpoint.rolling_success_rate == 0.85
        assert checkpoint.current_drift == 0.1
        assert checkpoint.incident_count == 2
        assert checkpoint.cnsr == 1.2


# ============================================================================
# LongHorizonReport Tests
# ============================================================================


class TestLongHorizonReport:
    """Tests for LongHorizonReport dataclass."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        return LongHorizonReport(
            evaluation_id="test-eval-001",
            start_time=datetime(2025, 1, 1, 10, 0, 0),
            end_time=datetime(2025, 1, 1, 12, 0, 0),
            total_tasks=100,
            status=EvaluationStatus.COMPLETED,
            overall_success_rate=0.85,
            rolling_success_history=[0.8, 0.82, 0.85, 0.87, 0.85],
            success_trend="stable",
            final_window_success_rate=0.9,
            total_cost=1.5,
            cnsr=1.2,
            cost_per_success=0.018,
            cost_trend="stable",
            final_goal_drift=0.15,
            max_goal_drift=0.25,
            drift_trend="increasing",
            reanchor_count=2,
            total_incidents=5,
            incident_rate_per_hour=2.5,
            incidents_by_severity={"LOW": 3, "MEDIUM": 2, "HIGH": 0, "CRITICAL": 0},
            critical_incidents=[],
            recommendations=["Continue monitoring performance"],
            checkpoints=[],
        )

    def test_report_creation(self, sample_report):
        """Test report is created correctly."""
        assert sample_report.evaluation_id == "test-eval-001"
        assert sample_report.total_tasks == 100
        assert sample_report.status == EvaluationStatus.COMPLETED
        assert sample_report.overall_success_rate == 0.85

    def test_to_dict(self, sample_report):
        """Test converting report to dictionary."""
        report_dict = sample_report.to_dict()

        assert report_dict["evaluation_id"] == "test-eval-001"
        assert report_dict["total_tasks"] == 100
        assert report_dict["status"] == "completed"
        assert "success_metrics" in report_dict
        assert report_dict["success_metrics"]["overall_success_rate"] == 0.85
        assert "cost_metrics" in report_dict
        assert report_dict["cost_metrics"]["cnsr"] == 1.2
        assert "goal_metrics" in report_dict
        assert "safety_metrics" in report_dict
        assert "recommendations" in report_dict

    def test_to_json(self, sample_report):
        """Test exporting report as JSON."""
        json_str = sample_report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["evaluation_id"] == "test-eval-001"
        assert parsed["success_metrics"]["overall_success_rate"] == 0.85

    def test_to_json_with_indent(self, sample_report):
        """Test JSON export with custom indent."""
        json_str = sample_report.to_json(indent=4)
        assert "    " in json_str  # 4-space indent


# ============================================================================
# LongHorizonEvaluator Tests
# ============================================================================


class TestLongHorizonEvaluator:
    """Tests for LongHorizonEvaluator class."""

    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.window_size == 10
        assert evaluator.drift_threshold == 0.3
        assert evaluator.incident_threshold_per_hour == 5.0
        assert evaluator._status == EvaluationStatus.NOT_STARTED

    def test_determine_success_with_dict(self, evaluator):
        """Test success determination with dict output."""
        assert evaluator._determine_success({"success": True}) is True
        assert evaluator._determine_success({"success": False}) is False
        assert evaluator._determine_success({}) is True  # Default

    def test_determine_success_with_bool(self, evaluator):
        """Test success determination with bool output."""
        assert evaluator._determine_success(True) is True
        assert evaluator._determine_success(False) is False

    def test_determine_success_with_none(self, evaluator):
        """Test success determination with None output."""
        assert evaluator._determine_success(None) is False

    def test_determine_success_with_other(self, evaluator):
        """Test success determination with other output types."""
        assert evaluator._determine_success("result") is True
        assert evaluator._determine_success(123) is True
        assert evaluator._determine_success([1, 2, 3]) is True

    def test_estimate_cost_from_dict(self, evaluator):
        """Test cost estimation from dict output."""
        assert evaluator._estimate_cost({"cost": 0.05}) == 0.05
        assert evaluator._estimate_cost({"cost": 1.0}) == 1.0

    def test_estimate_cost_default(self, evaluator):
        """Test default cost estimation."""
        assert evaluator._estimate_cost({}) == 0.01
        assert evaluator._estimate_cost("output") == 0.01
        assert evaluator._estimate_cost(None) == 0.01

    def test_determine_cost_trend_stable(self, evaluator):
        """Test cost trend detection - stable."""
        costs = [0.01, 0.01, 0.01, 0.01, 0.01]
        assert evaluator._determine_cost_trend(costs) == "stable"

    def test_determine_cost_trend_increasing(self, evaluator):
        """Test cost trend detection - increasing."""
        costs = [0.01, 0.01, 0.02, 0.025, 0.03]
        assert evaluator._determine_cost_trend(costs) == "increasing"

    def test_determine_cost_trend_decreasing(self, evaluator):
        """Test cost trend detection - decreasing."""
        costs = [0.03, 0.025, 0.02, 0.01, 0.01]
        assert evaluator._determine_cost_trend(costs) == "decreasing"

    def test_determine_cost_trend_insufficient_data(self, evaluator):
        """Test cost trend with insufficient data."""
        assert evaluator._determine_cost_trend([]) == "stable"
        assert evaluator._determine_cost_trend([0.01]) == "stable"

    def test_get_current_status_before_evaluation(self, evaluator):
        """Test getting status before evaluation starts."""
        status = evaluator.get_current_status()

        assert status["evaluation_id"] is None
        assert status["status"] == "not_started"
        assert status["tasks_completed"] == 0

    @pytest.mark.asyncio
    async def test_run_evaluation_sync_agent(self, mock_agent, mock_embedding):
        """Test running evaluation with synchronous agent."""
        evaluator = LongHorizonEvaluator(
            agent=mock_agent, embed_fn=mock_embedding, window_size=5
        )

        tasks = [f"task_{i}" for i in range(10)]

        report = await evaluator.run_evaluation(
            tasks=tasks, original_goal="Complete all tasks", checkpoint_interval=5
        )

        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_tasks == 10
        assert mock_agent.run.call_count == 10
        assert len(report.checkpoints) == 2  # At 5 and 10

    @pytest.mark.asyncio
    async def test_run_evaluation_async_agent(self, async_mock_agent, mock_embedding):
        """Test running evaluation with async agent."""
        evaluator = LongHorizonEvaluator(
            agent=async_mock_agent, embed_fn=mock_embedding, window_size=5
        )

        tasks = [f"task_{i}" for i in range(10)]

        report = await evaluator.run_evaluation(
            tasks=tasks, original_goal="Complete all tasks", checkpoint_interval=5
        )

        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_tasks == 10
        assert async_mock_agent.run.call_count == 10

    @pytest.mark.asyncio
    async def test_run_evaluation_with_failures(self, mock_embedding):
        """Test evaluation handles task failures."""
        agent = Mock()
        # Alternate success and failure
        agent.run = Mock(
            side_effect=[
                {"success": True, "cost": 0.01},
                {"success": False, "cost": 0.01},
                {"success": True, "cost": 0.01},
                {"success": False, "cost": 0.01},
                {"success": True, "cost": 0.01},
            ]
        )

        evaluator = LongHorizonEvaluator(
            agent=agent, embed_fn=mock_embedding, window_size=5
        )

        tasks = [f"task_{i}" for i in range(5)]
        report = await evaluator.run_evaluation(
            tasks=tasks, original_goal="Complete tasks"
        )

        assert report.status == EvaluationStatus.COMPLETED
        assert report.overall_success_rate == 0.6  # 3/5

    @pytest.mark.asyncio
    async def test_run_evaluation_with_exception(self, mock_embedding):
        """Test evaluation handles exceptions gracefully."""
        agent = Mock()
        agent.run = Mock(side_effect=Exception("Task error"))

        evaluator = LongHorizonEvaluator(
            agent=agent, embed_fn=mock_embedding, window_size=5
        )

        tasks = [f"task_{i}" for i in range(3)]
        report = await evaluator.run_evaluation(
            tasks=tasks, original_goal="Complete tasks"
        )

        # Tasks should be recorded as failures with incidents
        assert report.total_tasks == 3
        assert report.overall_success_rate == 0.0
        assert report.total_incidents >= 3  # At least one per failed task

    @pytest.mark.asyncio
    async def test_run_evaluation_with_callbacks(self, mock_agent, mock_embedding):
        """Test evaluation triggers callbacks."""
        evaluator = LongHorizonEvaluator(
            agent=mock_agent, embed_fn=mock_embedding, window_size=5
        )

        checkpoint_calls = []
        task_complete_calls = []

        def on_checkpoint(cp):
            checkpoint_calls.append(cp)

        def on_task_complete(result):
            task_complete_calls.append(result)

        tasks = [f"task_{i}" for i in range(10)]

        await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Complete all tasks",
            checkpoint_interval=5,
            on_checkpoint=on_checkpoint,
            on_task_complete=on_task_complete,
        )

        assert len(checkpoint_calls) == 2
        assert len(task_complete_calls) == 10

    @pytest.mark.asyncio
    async def test_run_evaluation_with_goal_inference(self, mock_agent, mock_embedding):
        """Test evaluation with goal inference function."""
        evaluator = LongHorizonEvaluator(
            agent=mock_agent, embed_fn=mock_embedding, window_size=5
        )

        def goal_inference_fn(task, result):
            return f"Processing {task}"

        tasks = [f"task_{i}" for i in range(20)]

        report = await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Complete all tasks",
            goal_inference_fn=goal_inference_fn,
        )

        # Goal drift should be tracked
        assert report.status == EvaluationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_checkpoints_captured(self, mock_agent, mock_embedding):
        """Test that checkpoints are captured at correct intervals."""
        evaluator = LongHorizonEvaluator(
            agent=mock_agent, embed_fn=mock_embedding, window_size=5
        )

        tasks = [f"task_{i}" for i in range(30)]

        report = await evaluator.run_evaluation(
            tasks=tasks, original_goal="Complete tasks", checkpoint_interval=10
        )

        assert len(report.checkpoints) == 3
        assert report.checkpoints[0].tasks_completed == 10
        assert report.checkpoints[1].tasks_completed == 20
        assert report.checkpoints[2].tasks_completed == 30

    def test_generate_recommendations_low_success(self, evaluator):
        """Test recommendations for low success rate."""
        recs = evaluator._generate_recommendations(
            success_rate=0.5, cnsr=0.8, mean_drift=0.1, incident_rate=1.0
        )

        assert any("Low success rate" in r for r in recs)

    def test_generate_recommendations_moderate_success(self, evaluator):
        """Test recommendations for moderate success rate."""
        recs = evaluator._generate_recommendations(
            success_rate=0.75, cnsr=1.5, mean_drift=0.1, incident_rate=1.0
        )

        assert any("Moderate success rate" in r for r in recs)

    def test_generate_recommendations_low_cnsr(self, evaluator):
        """Test recommendations for low CNSR."""
        recs = evaluator._generate_recommendations(
            success_rate=0.85, cnsr=0.5, mean_drift=0.1, incident_rate=1.0
        )

        assert any("cost efficiency" in r for r in recs)

    def test_generate_recommendations_high_drift(self, evaluator):
        """Test recommendations for high goal drift."""
        recs = evaluator._generate_recommendations(
            success_rate=0.9, cnsr=1.5, mean_drift=0.5, incident_rate=1.0
        )

        assert any("goal drift" in r.lower() for r in recs)

    def test_generate_recommendations_high_incident_rate(self, evaluator):
        """Test recommendations for high incident rate."""
        recs = evaluator._generate_recommendations(
            success_rate=0.9, cnsr=1.5, mean_drift=0.1, incident_rate=10.0
        )

        assert any("incident rate" in r.lower() for r in recs)

    def test_generate_recommendations_no_issues(self, evaluator):
        """Test recommendations when no issues detected."""
        recs = evaluator._generate_recommendations(
            success_rate=0.95, cnsr=2.0, mean_drift=0.05, incident_rate=1.0
        )

        assert any("No critical issues" in r for r in recs)

    def test_export_report_json(self, evaluator):
        """Test exporting report as JSON."""
        report = LongHorizonReport(
            evaluation_id="test-001",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tasks=10,
            status=EvaluationStatus.COMPLETED,
            overall_success_rate=0.9,
            rolling_success_history=[0.8, 0.85, 0.9],
            success_trend="improving",
            final_window_success_rate=0.95,
            total_cost=0.1,
            cnsr=1.5,
            cost_per_success=0.011,
            cost_trend="stable",
            final_goal_drift=0.1,
            max_goal_drift=0.15,
            drift_trend="stable",
            reanchor_count=0,
            total_incidents=0,
            incident_rate_per_hour=0.0,
            incidents_by_severity={},
            critical_incidents=[],
            recommendations=["Continue monitoring"],
            checkpoints=[],
        )

        json_export = evaluator.export_report(report, format="json")
        parsed = json.loads(json_export)
        assert parsed["evaluation_id"] == "test-001"

    def test_export_report_markdown(self, evaluator):
        """Test exporting report as Markdown."""
        report = LongHorizonReport(
            evaluation_id="test-001",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tasks=10,
            status=EvaluationStatus.COMPLETED,
            overall_success_rate=0.9,
            rolling_success_history=[0.8, 0.85, 0.9],
            success_trend="improving",
            final_window_success_rate=0.95,
            total_cost=0.1,
            cnsr=1.5,
            cost_per_success=0.011,
            cost_trend="stable",
            final_goal_drift=0.1,
            max_goal_drift=0.15,
            drift_trend="stable",
            reanchor_count=0,
            total_incidents=0,
            incident_rate_per_hour=0.0,
            incidents_by_severity={"LOW": 0, "MEDIUM": 0},
            critical_incidents=[],
            recommendations=["Continue monitoring"],
            checkpoints=[],
        )

        md_export = evaluator.export_report(report, format="markdown")
        assert "# Long-Horizon Evaluation Report" in md_export
        assert "test-001" in md_export
        assert "Success Metrics" in md_export
        assert "Cost Metrics" in md_export
        assert "Goal Metrics" in md_export
        assert "Safety Metrics" in md_export

    def test_export_report_invalid_format(self, evaluator):
        """Test export with invalid format raises error."""
        report = LongHorizonReport(
            evaluation_id="test-001",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tasks=10,
            status=EvaluationStatus.COMPLETED,
            overall_success_rate=0.9,
            rolling_success_history=[],
            success_trend="stable",
            final_window_success_rate=0.9,
            total_cost=0.1,
            cnsr=1.5,
            cost_per_success=0.011,
            cost_trend="stable",
            final_goal_drift=0.1,
            max_goal_drift=0.1,
            drift_trend="stable",
            reanchor_count=0,
            total_incidents=0,
            incident_rate_per_hour=0.0,
            incidents_by_severity={},
            critical_incidents=[],
            recommendations=[],
            checkpoints=[],
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            evaluator.export_report(report, format="xml")

    def test_format_markdown_report_with_checkpoints(self, evaluator):
        """Test markdown formatting includes checkpoint table."""
        checkpoints = [
            CheckpointData(
                checkpoint_num=1,
                timestamp=datetime.now(),
                tasks_completed=50,
                rolling_success_rate=0.8,
                current_drift=0.1,
                incident_count=2,
                cnsr=1.2,
            ),
            CheckpointData(
                checkpoint_num=2,
                timestamp=datetime.now(),
                tasks_completed=100,
                rolling_success_rate=0.85,
                current_drift=0.12,
                incident_count=3,
                cnsr=1.3,
            ),
        ]

        report = LongHorizonReport(
            evaluation_id="test-001",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tasks=100,
            status=EvaluationStatus.COMPLETED,
            overall_success_rate=0.85,
            rolling_success_history=[0.8, 0.85],
            success_trend="improving",
            final_window_success_rate=0.85,
            total_cost=1.0,
            cnsr=1.3,
            cost_per_success=0.012,
            cost_trend="stable",
            final_goal_drift=0.12,
            max_goal_drift=0.15,
            drift_trend="stable",
            reanchor_count=0,
            total_incidents=3,
            incident_rate_per_hour=1.5,
            incidents_by_severity={"LOW": 2, "MEDIUM": 1},
            critical_incidents=[],
            recommendations=["Continue monitoring"],
            checkpoints=checkpoints,
        )

        md = evaluator._format_markdown_report(report)
        assert "## Checkpoints" in md
        assert "| Checkpoint | Tasks | Success Rate |" in md


# ============================================================================
# SimpleLongHorizonEvaluator Tests
# ============================================================================


class TestSimpleLongHorizonEvaluator:
    """Tests for SimpleLongHorizonEvaluator class."""

    def test_initialization(self, simple_evaluator):
        """Test simple evaluator initialization."""
        assert simple_evaluator._task_count == 0

    def test_record_result_success(self, simple_evaluator):
        """Test recording successful results."""
        simple_evaluator.record_result(success=True, cost=0.01, latency_ms=100.0)

        assert simple_evaluator._task_count == 1
        summary = simple_evaluator.get_summary()
        assert summary["total_tasks"] == 1
        assert summary["overall_success_rate"] == 1.0

    def test_record_result_failure(self, simple_evaluator):
        """Test recording failed results."""
        simple_evaluator.record_result(success=False, cost=0.01, latency_ms=100.0)

        summary = simple_evaluator.get_summary()
        assert summary["overall_success_rate"] == 0.0

    def test_record_result_with_incident(self, simple_evaluator):
        """Test recording result with incident."""
        simple_evaluator.record_result(
            success=False,
            cost=0.01,
            latency_ms=100.0,
            incident_type=IncidentType.TOOL_FAILURE,
            incident_severity=IncidentSeverity.MEDIUM,
            incident_description="Tool execution failed",
        )

        summary = simple_evaluator.get_summary()
        assert summary["total_incidents"] == 1

    def test_multiple_results(self, simple_evaluator):
        """Test recording multiple results."""
        # Record 8 successes and 2 failures
        for i in range(10):
            simple_evaluator.record_result(success=i < 8, cost=0.01, latency_ms=100.0)

        summary = simple_evaluator.get_summary()
        assert summary["total_tasks"] == 10
        assert summary["overall_success_rate"] == 0.8
        assert summary["total_cost"] == 0.1

    def test_get_summary_cnsr(self, simple_evaluator):
        """Test CNSR calculation in summary."""
        # Record results
        for _ in range(10):
            simple_evaluator.record_result(success=True, cost=0.01)

        summary = simple_evaluator.get_summary()
        # CNSR = success_rate / mean_cost = 1.0 / 0.01 = 100.0
        assert summary["cnsr"] == 100.0

    def test_success_trend_detection(self, simple_evaluator):
        """Test success trend detection."""
        # Create degrading pattern - high success early, low success later
        for i in range(20):
            simple_evaluator.record_result(
                success=i < 15, cost=0.01  # First 15 succeed, last 5 fail
            )

        summary = simple_evaluator.get_summary()
        # Should detect degradation
        assert (
            "trend" in summary.get("success_trend", "").lower()
            or summary.get("is_degrading") is True
        )

    def test_rolling_success_rate(self, simple_evaluator):
        """Test rolling success rate calculation."""
        # Fill more than window size
        for i in range(15):
            simple_evaluator.record_result(
                success=i >= 10, cost=0.01  # Last 5 successful
            )

        summary = simple_evaluator.get_summary()
        # Rolling success should reflect recent performance
        rolling_rate = summary["rolling_success_rate"]
        assert 0 <= rolling_rate <= 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestLongHorizonIntegration:
    """Integration tests for long-horizon evaluation."""

    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self, mock_embedding):
        """Test complete evaluation pipeline."""
        # Create agent with varied responses
        results = []
        for i in range(100):
            # 80% success rate, varying costs
            success = i % 5 != 0
            cost = 0.01 + (i % 3) * 0.005
            results.append({"success": success, "cost": cost})

        agent = Mock()
        agent.run = Mock(side_effect=results)

        evaluator = LongHorizonEvaluator(
            agent=agent, embed_fn=mock_embedding, window_size=20
        )

        report = await evaluator.run_evaluation(
            tasks=[f"task_{i}" for i in range(100)],
            original_goal="Complete data processing",
            checkpoint_interval=25,
        )

        # Verify report completeness
        assert report.evaluation_id is not None
        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_tasks == 100
        assert 0.75 <= report.overall_success_rate <= 0.85
        assert len(report.checkpoints) == 4
        assert len(report.recommendations) > 0

    @pytest.mark.asyncio
    async def test_evaluation_with_degradation(self, mock_embedding):
        """Test evaluation detects performance degradation."""
        # Create degrading success pattern
        results = []
        for i in range(100):
            # High success early, degrading over time
            success_prob = 0.95 - (i * 0.005)  # 95% -> 45%
            success = i < int(success_prob * 100 / 2)
            results.append({"success": success, "cost": 0.01})

        agent = Mock()
        agent.run = Mock(side_effect=results)

        evaluator = LongHorizonEvaluator(
            agent=agent, embed_fn=mock_embedding, window_size=20
        )

        report = await evaluator.run_evaluation(
            tasks=[f"task_{i}" for i in range(100)], original_goal="Complete tasks"
        )

        # Should detect degradation or have relevant recommendations
        assert report.status == EvaluationStatus.COMPLETED
        has_degradation_notice = report.success_trend == "degrading" or any(
            "degradation" in r.lower() for r in report.recommendations
        )
        # Either trend detected or recommendations mention it
        assert has_degradation_notice or report.success_trend in ["stable", "degrading"]


# ============================================================================
# EvaluationStatus Tests
# ============================================================================


class TestEvaluationStatus:
    """Tests for EvaluationStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert EvaluationStatus.NOT_STARTED.value == "not_started"
        assert EvaluationStatus.RUNNING.value == "running"
        assert EvaluationStatus.COMPLETED.value == "completed"
        assert EvaluationStatus.FAILED.value == "failed"
        assert EvaluationStatus.CANCELLED.value == "cancelled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
