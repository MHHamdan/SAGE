"""
Integration Tests for Long-Horizon Evaluation System.

Tests the complete integration of:
- Goal drift tracking
- Incident tracking
- Rolling window metrics
- Long-horizon evaluation harness
- Human approval flow
- Deployment loop

These tests verify that all components work together correctly
in realistic scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

import numpy as np

# Evaluation components
from sage.evaluation.goal_drift import (
    GoalDriftTracker,
    MultiGoalDriftTracker,
)
from sage.evaluation.incident_tracker import (
    IncidentTracker,
    IncidentType,
    IncidentSeverity,
)
from sage.evaluation.rolling_metrics import (
    RollingWindowTracker,
    TaskResult,
)
from sage.evaluation.long_horizon import (
    LongHorizonEvaluator,
    SimpleLongHorizonEvaluator,
    EvaluationStatus,
)

# Human oversight components
from sage.human_oversight.approval_flow import (
    ApprovalHandler,
    ApprovalRequest,
    RiskLevel,
)
from sage.human_oversight.escalation import (
    EscalationHandler,
    EscalationLevel,
)
from sage.human_oversight.audit import (
    AuditLogger,
    AuditEventType,
)

# Learning components
from sage.learning.deployment_loop import (
    DeploymentLoop,
    DeploymentConfig,
    ABTestDeployment,
)
from sage.learning.feedback import (
    FeedbackCollector,
    FeedbackType,
)
from sage.learning.experience import (
    ExperienceBuffer,
    Experience,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding_fn():
    """Create a mock embedding function that produces consistent embeddings."""
    embeddings_cache = {}
    base_embedding = np.random.randn(384)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)

    def embed_fn(text: str) -> np.ndarray:
        if text not in embeddings_cache:
            # Create embeddings that drift slightly based on text length
            drift_factor = len(text) / 100.0
            noise = np.random.randn(384) * 0.1 * drift_factor
            embedding = base_embedding + noise
            embeddings_cache[text] = embedding / np.linalg.norm(embedding)
        return embeddings_cache[text]

    return embed_fn


@pytest.fixture
def mock_successful_agent():
    """Create a mock agent that mostly succeeds."""
    agent = Mock()

    call_count = [0]

    def run_task(task):
        call_count[0] += 1
        # 90% success rate
        success = call_count[0] % 10 != 0
        return {
            "success": success,
            "cost": 0.01 + np.random.random() * 0.01,
            "output": f"Completed task {call_count[0]}",
        }

    agent.run = Mock(side_effect=run_task)
    return agent


@pytest.fixture
def mock_degrading_agent():
    """Create an agent that degrades over time."""
    agent = Mock()

    call_count = [0]

    def run_task(task):
        call_count[0] += 1
        # Success rate degrades from 95% to 50%
        success_prob = 0.95 - (call_count[0] * 0.005)
        success = np.random.random() < success_prob
        return {
            "success": success,
            "cost": 0.01,
        }

    agent.run = Mock(side_effect=run_task)
    return agent


# ============================================================================
# Long-Horizon Evaluation Integration Tests
# ============================================================================


class TestLongHorizonEvaluationIntegration:
    """Integration tests for the complete evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_full_evaluation_with_all_components(
        self, mock_successful_agent, mock_embedding_fn
    ):
        """Test complete evaluation with all tracking components."""
        evaluator = LongHorizonEvaluator(
            agent=mock_successful_agent,
            embed_fn=mock_embedding_fn,
            window_size=10,
            drift_threshold=0.3,
            incident_threshold_per_hour=5.0,
        )

        tasks = [f"task_{i}" for i in range(50)]

        def goal_inference(task, result):
            return f"Processing: {task}"

        report = await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Complete all data processing tasks",
            checkpoint_interval=10,
            goal_inference_fn=goal_inference,
        )

        # Verify report completeness
        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_tasks == 50
        assert 0.85 <= report.overall_success_rate <= 0.95
        assert report.cnsr > 0
        assert len(report.checkpoints) == 5
        assert len(report.recommendations) > 0

    @pytest.mark.asyncio
    async def test_evaluation_detects_degradation(
        self, mock_degrading_agent, mock_embedding_fn
    ):
        """Test that evaluation detects performance degradation."""
        evaluator = LongHorizonEvaluator(
            agent=mock_degrading_agent,
            embed_fn=mock_embedding_fn,
            window_size=10,
        )

        tasks = [f"task_{i}" for i in range(80)]

        report = await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Process data",
            checkpoint_interval=20,
        )

        # Should detect degradation
        assert report.status == EvaluationStatus.COMPLETED

        # Early success rate should be higher than late
        if len(report.checkpoints) >= 2:
            early_success = report.checkpoints[0].rolling_success_rate
            late_success = report.checkpoints[-1].rolling_success_rate
            # Later success should generally be lower due to degradation
            # (allowing some tolerance for randomness)
            assert late_success <= early_success + 0.2

    @pytest.mark.asyncio
    async def test_simple_evaluator_tracking(self):
        """Test SimpleLongHorizonEvaluator for basic tracking."""
        evaluator = SimpleLongHorizonEvaluator(window_size=10)

        # Record a series of results
        for i in range(30):
            evaluator.record_result(
                success=i % 4 != 0,  # 75% success
                cost=0.01,
                latency_ms=100.0,
            )

        summary = evaluator.get_summary()

        assert summary["total_tasks"] == 30
        assert 0.7 <= summary["overall_success_rate"] <= 0.8
        assert summary["total_cost"] == 0.30
        assert summary["cnsr"] > 0


# ============================================================================
# Human Oversight Integration Tests
# ============================================================================


class TestHumanOversightIntegration:
    """Integration tests for human oversight components."""

    @pytest.mark.asyncio
    async def test_approval_with_escalation_and_audit(self):
        """Test approval flow with escalation and audit logging."""
        # Setup components
        approval_handler = ApprovalHandler(default_timeout=5.0)
        escalation_handler = EscalationHandler()
        audit_logger = AuditLogger(session_id="test-session")

        # Register client
        approval_handler.register_client("agent-1", "secret-123")

        # Create approval request
        request = approval_handler.create_request(
            action="deploy_model",
            context={"model": "gpt-4", "environment": "production"},
            risk_level=RiskLevel.HIGH,
            requester="agent-1",
        )

        # Log to audit
        audit_logger.log_approval_requested(
            request_id=request.request_id,
            action="deploy_model",
            actor="agent-1",
            risk_level="high",
        )

        # Simulate escalation
        escalation = escalation_handler.escalate(
            request_id=request.request_id,
            level=EscalationLevel.TIER_1,
            reason="High-risk deployment pending",
        )

        audit_logger.log_escalation(
            escalation_id=escalation.escalation_id,
            request_id=request.request_id,
            level="tier_1",
            reason="High-risk deployment pending",
        )

        # Approve the request
        result = approval_handler.approve(
            request_id=request.request_id,
            approver="admin",
            reason="Approved after review",
        )

        audit_logger.log_approval_decision(
            request_id=request.request_id,
            approved=True,
            approver="admin",
        )

        # Resolve escalation
        escalation_handler.resolve(
            escalation_id=escalation.escalation_id,
            resolver="admin",
        )

        # Verify results
        assert result.approved is True
        assert len(audit_logger.get_entries()) == 3

        # Verify statistics
        approval_stats = approval_handler.get_statistics()
        assert approval_stats["total_requests"] == 1

        escalation_stats = escalation_handler.get_statistics()
        assert escalation_stats["total_escalations"] == 1
        assert escalation_stats["resolved_escalations"] == 1

    @pytest.mark.asyncio
    async def test_timeout_handling_with_audit(self):
        """Test approval timeout with proper audit logging."""
        approval_handler = ApprovalHandler(
            default_timeout=0.1,
            auto_reject_on_timeout=True,
        )
        audit_logger = AuditLogger()

        approval_handler.register_client("agent-1", "secret-123")

        request = approval_handler.create_request(
            action="risky_action",
            risk_level=RiskLevel.CRITICAL,
        )

        audit_logger.log_approval_requested(
            request_id=request.request_id,
            action="risky_action",
            actor="system",
            risk_level="critical",
        )

        # Wait for timeout
        result = await approval_handler.wait_for_approval(request.request_id)

        # Should be auto-rejected
        assert result.approved is False
        assert result.status.value == "timeout"


# ============================================================================
# Learning System Integration Tests
# ============================================================================


class TestLearningSystemIntegration:
    """Integration tests for learning components."""

    @pytest.mark.asyncio
    async def test_deployment_loop_with_feedback(self):
        """Test deployment loop with feedback collection."""
        # Setup
        agent = Mock()
        agent.run = Mock(return_value={"success": True, "cost": 0.01})

        feedback_collector = FeedbackCollector()
        experience_buffer = ExperienceBuffer(max_size=1000)

        config = DeploymentConfig(
            evaluation_interval=10,
            checkpoint_interval=20,
        )

        loop = DeploymentLoop(
            agent=agent,
            config=config,
        )

        tasks = [f"task_{i}" for i in range(25)]
        evaluations = []

        async for update in loop.run(tasks):
            if update.event_type == "evaluation":
                evaluations.append(update)

                # Simulate collecting feedback
                feedback_collector.add_automated_feedback(
                    task_id=f"eval_{len(evaluations)}",
                    success=update.success_rate > 0.7,
                    metrics={"success_rate": update.success_rate},
                )

                # Store as experience
                experience_buffer.add_transition(
                    state={"tasks_completed": update.tasks_completed},
                    action={"continue": True},
                    reward=update.success_rate,
                )

        # Verify
        assert len(evaluations) >= 2
        assert len(feedback_collector) >= 2
        assert len(experience_buffer) >= 2

        # Check feedback aggregation
        aggregated = feedback_collector.aggregate()
        assert aggregated.total_count >= 2

    @pytest.mark.asyncio
    async def test_ab_test_with_evaluation(self):
        """Test A/B testing with evaluation metrics."""
        # Create two agents with different success rates
        agent_a = Mock()
        agent_a.run = Mock(return_value={"success": True})

        agent_b = Mock()
        call_count = [0]

        def agent_b_run(task):
            call_count[0] += 1
            # B has 80% success (slightly worse)
            return {"success": call_count[0] % 5 != 0}

        agent_b.run = Mock(side_effect=agent_b_run)

        ab_test = ABTestDeployment(
            agent_a=agent_a,
            agent_b=agent_b,
            traffic_split=0.5,
            min_samples=10,
        )

        # Run tasks
        for _ in range(100):
            await ab_test.run_task("test_task")

        # Get comparison
        comparison = ab_test.get_comparison()

        assert comparison["agent_a"]["tasks"] > 0
        assert comparison["agent_b"]["tasks"] > 0
        assert comparison["winner"] in ["A", "B", "tie"]


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests combining all systems."""

    @pytest.mark.asyncio
    async def test_complete_agent_deployment_scenario(
        self, mock_successful_agent, mock_embedding_fn
    ):
        """Test a complete agent deployment scenario."""
        # Setup all components
        approval_handler = ApprovalHandler(default_timeout=1.0)
        audit_logger = AuditLogger(session_id="deployment-001")
        feedback_collector = FeedbackCollector()
        experience_buffer = ExperienceBuffer(max_size=1000)

        approval_handler.register_client("deployment-agent", "secret")

        # Phase 1: Request approval for deployment
        approval_request = approval_handler.create_request(
            action="deploy_agent",
            context={"agent_version": "v1.0", "environment": "staging"},
            risk_level=RiskLevel.MEDIUM,
            requester="deployment-agent",
        )

        audit_logger.log_approval_requested(
            request_id=approval_request.request_id,
            action="deploy_agent",
            actor="deployment-agent",
            risk_level="medium",
        )

        # Approve
        approval_handler.approve(
            request_id=approval_request.request_id,
            approver="admin",
        )

        audit_logger.log_approval_decision(
            request_id=approval_request.request_id,
            approved=True,
            approver="admin",
        )

        # Phase 2: Run evaluation
        evaluator = LongHorizonEvaluator(
            agent=mock_successful_agent,
            embed_fn=mock_embedding_fn,
            window_size=10,
        )

        tasks = [f"staging_task_{i}" for i in range(30)]
        report = await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Complete staging validation",
            checkpoint_interval=10,
        )

        # Phase 3: Collect feedback
        feedback_collector.add_automated_feedback(
            task_id="deployment-evaluation",
            success=report.overall_success_rate > 0.8,
            metrics={
                "success_rate": report.overall_success_rate,
                "cnsr": report.cnsr,
                "incidents": report.total_incidents,
            },
        )

        # Phase 4: Store experience
        experience_buffer.add_transition(
            state={"phase": "staging"},
            action={"deploy": True},
            reward=report.overall_success_rate,
            metadata={"evaluation_id": report.evaluation_id},
        )

        # Verify complete flow
        assert report.status == EvaluationStatus.COMPLETED
        assert len(audit_logger.get_entries()) >= 2
        assert len(feedback_collector) >= 1
        assert len(experience_buffer) >= 1

        # Export audit log
        audit_export = audit_logger.export(format="markdown")
        assert "deployment-001" in audit_export

    @pytest.mark.asyncio
    async def test_incident_tracking_during_evaluation(self, mock_embedding_fn):
        """Test incident tracking during long-horizon evaluation."""
        # Create an agent that triggers incidents
        agent = Mock()
        call_count = [0]

        def agent_run(task):
            call_count[0] += 1
            if call_count[0] % 15 == 0:
                raise Exception("Simulated error")
            return {"success": call_count[0] % 5 != 0}

        agent.run = Mock(side_effect=agent_run)

        evaluator = LongHorizonEvaluator(
            agent=agent,
            embed_fn=mock_embedding_fn,
            window_size=10,
        )

        tasks = [f"task_{i}" for i in range(45)]
        report = await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Test with incidents",
            checkpoint_interval=15,
        )

        # Should have recorded incidents
        assert report.total_tasks == 45
        assert report.total_incidents >= 2  # At least 2 exceptions


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related integration tests."""

    @pytest.mark.asyncio
    async def test_large_scale_evaluation(self, mock_embedding_fn):
        """Test evaluation at scale."""
        agent = Mock()
        agent.run = Mock(return_value={"success": True, "cost": 0.01})

        evaluator = LongHorizonEvaluator(
            agent=agent,
            embed_fn=mock_embedding_fn,
            window_size=50,
        )

        # Run with many tasks
        tasks = [f"task_{i}" for i in range(200)]
        report = await evaluator.run_evaluation(
            tasks=tasks,
            original_goal="Large scale test",
            checkpoint_interval=50,
        )

        assert report.total_tasks == 200
        assert len(report.checkpoints) == 4

    def test_experience_buffer_at_capacity(self):
        """Test experience buffer when at capacity."""
        buffer = ExperienceBuffer(max_size=100)

        # Add more than capacity
        for i in range(200):
            buffer.add_transition(
                state={"i": i},
                action={"action": i},
                reward=float(i),
            )

        # Should maintain max size
        assert len(buffer) == 100

        # Most recent should be preserved
        recent = buffer.get_recent(10)
        assert all(e.state["i"] >= 190 for e in recent)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
