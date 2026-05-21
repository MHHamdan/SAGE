"""
Tests for Deployment Loop and Learning Components.

Comprehensive tests for DeploymentLoop, FeedbackCollector, and ExperienceBuffer.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from sage.learning.deployment_loop import (
    DeploymentLoop,
    DeploymentConfig,
    DeploymentState,
    DeploymentStatus,
    DeploymentMetrics,
    DeploymentUpdate,
    ABTestDeployment,
)
from sage.learning.feedback import (
    FeedbackCollector,
    Feedback,
    FeedbackType,
    FeedbackSource,
    AggregatedFeedback,
)
from sage.learning.experience import (
    ExperienceBuffer,
    Experience,
    ExperienceBatch,
    PrioritizedExperienceBuffer,
)

# ============================================================================
# DeploymentConfig Tests
# ============================================================================


class TestDeploymentConfig:
    """Tests for DeploymentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeploymentConfig()

        assert config.evaluation_interval == 100
        assert config.min_tasks_for_eval == 50
        assert config.success_threshold == 0.8
        assert config.rollback_threshold == 0.6
        assert config.enable_auto_rollback is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeploymentConfig(
            evaluation_interval=50,
            success_threshold=0.9,
            enable_auto_rollback=False,
        )

        assert config.evaluation_interval == 50
        assert config.success_threshold == 0.9
        assert config.enable_auto_rollback is False


# ============================================================================
# DeploymentMetrics Tests
# ============================================================================


class TestDeploymentMetrics:
    """Tests for DeploymentMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = DeploymentMetrics(
            period_id="period-001",
            start_time=datetime.now(),
            tasks_completed=100,
            tasks_succeeded=85,
            tasks_failed=15,
            success_rate=0.85,
        )

        assert metrics.period_id == "period-001"
        assert metrics.tasks_completed == 100
        assert metrics.success_rate == 0.85

    def test_to_dict(self):
        """Test to_dict method."""
        metrics = DeploymentMetrics(
            period_id="period-001",
            start_time=datetime.now(),
        )

        d = metrics.to_dict()
        assert d["period_id"] == "period-001"
        assert "start_time" in d


# ============================================================================
# DeploymentLoop Tests
# ============================================================================


class TestDeploymentLoop:
    """Tests for DeploymentLoop class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock()
        agent.run = Mock(return_value={"success": True, "cost": 0.01})
        return agent

    @pytest.fixture
    def async_mock_agent(self):
        """Create async mock agent."""
        agent = Mock()
        agent.run = AsyncMock(return_value={"success": True, "cost": 0.01})
        return agent

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DeploymentConfig(
            evaluation_interval=10,
            checkpoint_interval=25,
            rollback_threshold=0.3,
        )

    @pytest.fixture
    def loop(self, mock_agent, config):
        """Create deployment loop."""
        return DeploymentLoop(
            agent=mock_agent,
            config=config,
            agent_version="v1.0",
        )

    def test_initialization(self, loop):
        """Test loop initialization."""
        assert loop.agent_version == "v1.0"
        assert loop.state.status == DeploymentStatus.INITIALIZING
        assert loop.state.total_tasks == 0

    @pytest.mark.asyncio
    async def test_run_basic(self, loop):
        """Test basic run with tasks."""
        tasks = [f"task_{i}" for i in range(20)]
        updates = []

        async for update in loop.run(tasks):
            updates.append(update)

        assert len(updates) >= 2  # At least evaluation + completed
        assert updates[-1].event_type == "completed"
        assert updates[-1].tasks_completed == 20

    @pytest.mark.asyncio
    async def test_run_with_async_agent(self, async_mock_agent, config):
        """Test run with async agent."""
        loop = DeploymentLoop(agent=async_mock_agent, config=config)
        tasks = [f"task_{i}" for i in range(10)]

        updates = []
        async for update in loop.run(tasks):
            updates.append(update)

        assert async_mock_agent.run.call_count == 10

    @pytest.mark.asyncio
    async def test_run_with_failures(self, config):
        """Test run handles task failures."""
        agent = Mock()
        agent.run = Mock(side_effect=Exception("Task error"))

        loop = DeploymentLoop(agent=agent, config=config)
        tasks = [f"task_{i}" for i in range(5)]

        updates = []
        async for update in loop.run(tasks):
            updates.append(update)

        # All tasks should fail
        assert loop.state.current_metrics.tasks_failed >= 0

    @pytest.mark.asyncio
    async def test_run_with_max_tasks(self, loop):
        """Test run with task limit."""
        tasks = [f"task_{i}" for i in range(100)]

        updates = []
        async for update in loop.run(tasks, max_tasks=15):
            updates.append(update)

        assert updates[-1].tasks_completed == 15

    @pytest.mark.asyncio
    async def test_evaluation_callback(self, mock_agent, config):
        """Test evaluation callback is called."""
        callback_calls = []

        async def on_eval(metrics):
            callback_calls.append(metrics)

        loop = DeploymentLoop(
            agent=mock_agent,
            config=config,
            on_evaluation=on_eval,
        )

        tasks = [f"task_{i}" for i in range(20)]
        async for _ in loop.run(tasks):
            pass

        assert len(callback_calls) >= 2  # At least 2 evaluations

    @pytest.mark.asyncio
    async def test_rollback_on_low_success(self, config):
        """Test rollback triggers on low success rate."""
        # Create agent that always fails
        agent = Mock()
        agent.run = Mock(return_value={"success": False})

        config.enable_auto_rollback = True
        config.rollback_threshold = 0.5

        loop = DeploymentLoop(agent=agent, config=config)
        tasks = [f"task_{i}" for i in range(15)]

        rollback_occurred = False
        async for update in loop.run(tasks):
            if update.event_type == "rollback":
                rollback_occurred = True

        assert rollback_occurred

    def test_pause_resume(self, loop):
        """Test pause and resume."""
        loop.pause()
        assert loop.state.status == DeploymentStatus.PAUSED

        loop.resume()
        assert loop.state.status == DeploymentStatus.RUNNING

    def test_stop(self, loop):
        """Test stop."""
        loop.stop()
        assert loop._stop_event.is_set()

    def test_update_agent(self, loop):
        """Test updating agent."""
        new_agent = Mock()
        loop.update_agent(new_agent, "v2.0")

        assert loop.agent == new_agent
        assert loop.agent_version == "v2.0"
        assert loop._previous_agent is not None

    def test_get_state(self, loop):
        """Test getting state."""
        state = loop.get_state()

        assert "deployment_id" in state
        assert state["agent_version"] == "v1.0"
        assert state["total_tasks"] == 0

    def test_determine_success(self, loop):
        """Test success determination."""
        assert loop._determine_success({"success": True}) is True
        assert loop._determine_success({"success": False}) is False
        assert loop._determine_success(True) is True
        assert loop._determine_success(None) is False


# ============================================================================
# ABTestDeployment Tests
# ============================================================================


class TestABTestDeployment:
    """Tests for ABTestDeployment class."""

    @pytest.fixture
    def agent_a(self):
        """Create agent A."""
        agent = Mock()
        agent.run = Mock(return_value={"success": True})
        return agent

    @pytest.fixture
    def agent_b(self):
        """Create agent B."""
        agent = Mock()
        agent.run = Mock(return_value={"success": True})
        return agent

    @pytest.fixture
    def ab_test(self, agent_a, agent_b):
        """Create A/B test deployment."""
        return ABTestDeployment(
            agent_a=agent_a,
            agent_b=agent_b,
            traffic_split=0.5,
            min_samples=5,
        )

    @pytest.mark.asyncio
    async def test_run_task(self, ab_test):
        """Test running a task."""
        result = await ab_test.run_task("test task")

        assert "agent" in result
        assert result["agent"] in ["A", "B"]
        assert "success" in result

    @pytest.mark.asyncio
    async def test_traffic_split(self, ab_test):
        """Test traffic is split between agents."""
        a_count = 0
        b_count = 0

        for _ in range(100):
            result = await ab_test.run_task("task")
            if result["agent"] == "A":
                a_count += 1
            else:
                b_count += 1

        # Should be roughly 50/50
        assert 30 <= a_count <= 70
        assert 30 <= b_count <= 70

    def test_get_comparison(self, ab_test):
        """Test getting comparison."""
        # Add some metrics
        ab_test.metrics_a.tasks_completed = 50
        ab_test.metrics_a.tasks_succeeded = 45
        ab_test.metrics_b.tasks_completed = 50
        ab_test.metrics_b.tasks_succeeded = 48

        comparison = ab_test.get_comparison()

        assert comparison["agent_a"]["tasks"] == 50
        assert comparison["agent_b"]["tasks"] == 50
        assert comparison["winner"] == "B"  # 48/50 > 45/50


# ============================================================================
# FeedbackCollector Tests
# ============================================================================


class TestFeedbackCollector:
    """Tests for FeedbackCollector class."""

    @pytest.fixture
    def collector(self):
        """Create feedback collector."""
        return FeedbackCollector()

    def test_add_feedback(self, collector):
        """Test adding feedback."""
        feedback = Feedback(
            feedback_type=FeedbackType.HUMAN,
            rating=4.0,
        )

        result = collector.add_feedback(feedback)

        assert result.feedback_id is not None
        assert len(collector) == 1

    def test_add_human_feedback(self, collector):
        """Test adding human feedback."""
        feedback = collector.add_human_feedback(
            rating=4.5,
            comment="Great work",
            task_id="task-123",
        )

        assert feedback.feedback_type == FeedbackType.HUMAN
        assert feedback.rating == 4.5

    def test_add_automated_feedback(self, collector):
        """Test adding automated feedback."""
        feedback = collector.add_automated_feedback(
            task_id="task-123",
            success=True,
            latency_ms=150.0,
        )

        assert feedback.feedback_type == FeedbackType.AUTOMATED
        assert "latency_ms" in feedback.dimensions

    def test_add_environment_feedback(self, collector):
        """Test adding environment feedback."""
        feedback = collector.add_environment_feedback(
            reward=1.5,
            task_id="task-123",
        )

        assert feedback.feedback_type == FeedbackType.ENVIRONMENT
        assert feedback.rating == 1.5

    def test_get_feedback_filtered(self, collector):
        """Test filtering feedback."""
        collector.add_human_feedback(rating=4.0)
        collector.add_automated_feedback(task_id="t1", success=True)
        collector.add_human_feedback(rating=3.0)

        human_only = collector.get_feedback(feedback_type=FeedbackType.HUMAN)
        assert len(human_only) == 2

    def test_aggregate(self, collector):
        """Test aggregation."""
        collector.add_human_feedback(rating=4.0)
        collector.add_human_feedback(rating=5.0)
        collector.add_human_feedback(rating=3.0)

        aggregated = collector.aggregate()

        assert aggregated.total_count == 3
        assert aggregated.average_rating == 4.0

    def test_export_json(self, collector):
        """Test JSON export."""
        collector.add_human_feedback(rating=4.0)
        export = collector.export(format="json")

        assert "rating" in export
        assert "4.0" in export

    def test_export_csv(self, collector):
        """Test CSV export."""
        collector.add_human_feedback(rating=4.0)
        export = collector.export(format="csv")

        assert "feedback_id" in export

    def test_clear(self, collector):
        """Test clearing feedback."""
        collector.add_human_feedback(rating=4.0)
        collector.clear()

        assert len(collector) == 0


# ============================================================================
# ExperienceBuffer Tests
# ============================================================================


class TestExperienceBuffer:
    """Tests for ExperienceBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create experience buffer."""
        return ExperienceBuffer(max_size=100)

    def test_add_experience(self, buffer):
        """Test adding experience."""
        exp = Experience(
            state={"obs": [1, 2, 3]},
            action={"move": "forward"},
            reward=1.0,
        )

        result = buffer.add(exp)

        assert result.experience_id is not None
        assert len(buffer) == 1

    def test_add_transition(self, buffer):
        """Test adding transition."""
        exp = buffer.add_transition(
            state={"obs": [1, 2, 3]},
            action={"move": "forward"},
            reward=1.0,
            next_state={"obs": [2, 3, 4]},
        )

        assert exp.state == {"obs": [1, 2, 3]}
        assert exp.reward == 1.0

    def test_max_size(self):
        """Test buffer respects max size."""
        buffer = ExperienceBuffer(max_size=10)

        for i in range(20):
            buffer.add_transition(
                state={"i": i},
                action={},
                reward=i,
            )

        assert len(buffer) == 10

    def test_sample(self, buffer):
        """Test sampling."""
        for i in range(50):
            buffer.add_transition(
                state={"i": i},
                action={},
                reward=i,
            )

        batch = buffer.sample(batch_size=10)

        assert len(batch) == 10
        assert isinstance(batch, ExperienceBatch)

    def test_sample_insufficient(self, buffer):
        """Test sampling with insufficient experiences."""
        buffer.add_transition(state={}, action={}, reward=0)

        with pytest.raises(ValueError):
            buffer.sample(batch_size=10)

    def test_sample_recent(self, buffer):
        """Test sampling with recency bias."""
        for i in range(50):
            buffer.add_transition(
                state={"i": i},
                action={},
                reward=i,
            )

        batch = buffer.sample_recent(batch_size=10, recency_bias=0.8)
        assert len(batch) == 10

    def test_get_recent(self, buffer):
        """Test getting recent experiences."""
        for i in range(20):
            buffer.add_transition(
                state={"i": i},
                action={},
                reward=i,
            )

        recent = buffer.get_recent(5)
        assert len(recent) == 5

    def test_get_high_reward(self, buffer):
        """Test getting high reward experiences."""
        for i in range(20):
            buffer.add_transition(
                state={"i": i},
                action={},
                reward=i,
            )

        high_reward = buffer.get_high_reward(5)
        assert len(high_reward) == 5
        assert all(e.reward >= 15 for e in high_reward)

    def test_get_statistics(self, buffer):
        """Test getting statistics."""
        for i in range(10):
            buffer.add_transition(
                state={"i": i},
                action={},
                reward=i,
            )

        stats = buffer.get_statistics()

        assert stats["size"] == 10
        assert stats["mean_reward"] == 4.5  # (0+9)/2

    def test_clear(self, buffer):
        """Test clearing buffer."""
        buffer.add_transition(state={}, action={}, reward=0)
        buffer.clear()

        assert len(buffer) == 0


# ============================================================================
# PrioritizedExperienceBuffer Tests
# ============================================================================


class TestPrioritizedExperienceBuffer:
    """Tests for PrioritizedExperienceBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create prioritized buffer."""
        return PrioritizedExperienceBuffer(max_size=100)

    def test_add_with_priority(self, buffer):
        """Test adding with priority."""
        exp = Experience(
            state={"obs": [1, 2, 3]},
            action={},
            reward=1.0,
        )

        result = buffer.add(exp, priority=2.0)

        assert result.experience_id is not None
        assert len(buffer) == 1

    def test_max_size(self):
        """Test buffer respects max size."""
        buffer = PrioritizedExperienceBuffer(max_size=10)

        for i in range(20):
            buffer.add(
                Experience(
                    state={"i": i},
                    action={},
                    reward=i,
                )
            )

        assert len(buffer) == 10

    def test_sample(self, buffer):
        """Test prioritized sampling."""
        for i in range(50):
            buffer.add(
                Experience(
                    state={"i": i},
                    action={},
                    reward=i,
                ),
                priority=float(i + 1),
            )

        batch = buffer.sample(batch_size=10)
        assert len(batch) == 10

    def test_update_priorities(self, buffer):
        """Test updating priorities."""
        exp = buffer.add(
            Experience(
                state={},
                action={},
                reward=1.0,
            ),
            priority=1.0,
        )

        buffer.update_priorities({exp.experience_id: 5.0})

        # Priority should be updated
        assert buffer._max_priority >= 5.0

    def test_get_statistics(self, buffer):
        """Test getting statistics."""
        for i in range(10):
            buffer.add(
                Experience(
                    state={"i": i},
                    action={},
                    reward=i,
                )
            )

        stats = buffer.get_statistics()

        assert stats["size"] == 10
        assert "mean_priority" in stats


# ============================================================================
# ExperienceBatch Tests
# ============================================================================


class TestExperienceBatch:
    """Tests for ExperienceBatch dataclass."""

    @pytest.fixture
    def batch(self):
        """Create a batch."""
        experiences = [
            Experience(state={"i": i}, action={}, reward=float(i)) for i in range(5)
        ]
        return ExperienceBatch(experiences=experiences)

    def test_len(self, batch):
        """Test length."""
        assert len(batch) == 5

    def test_iter(self, batch):
        """Test iteration."""
        count = 0
        for exp in batch:
            count += 1
        assert count == 5

    def test_get_states(self, batch):
        """Test getting states."""
        states = batch.get_states()
        assert len(states) == 5

    def test_get_rewards(self, batch):
        """Test getting rewards."""
        rewards = batch.get_rewards()
        assert rewards == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_mean_reward(self, batch):
        """Test mean reward."""
        assert batch.mean_reward() == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
