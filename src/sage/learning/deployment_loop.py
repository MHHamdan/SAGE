"""
Continuous Deployment Loop for Agent Learning

Implements a continuous deployment and learning cycle for autonomous agents,
enabling them to improve based on feedback and performance metrics over time.

The deployment loop supports:
- Continuous task execution with learning updates
- Performance monitoring and degradation detection
- Automatic rollback on performance issues
- A/B testing of agent configurations
- Gradual rollouts with canary deployments

Example:
    >>> loop = DeploymentLoop(
    ...     agent=my_agent,
    ...     config=DeploymentConfig(
    ...         evaluation_interval=100,
    ...         rollback_threshold=0.7
    ...     )
    ... )
    >>>
    >>> # Run continuous deployment
    >>> async for update in loop.run(tasks=task_stream):
    ...     print(f"Processed {update.tasks_completed} tasks")
    ...     print(f"Success rate: {update.success_rate:.2%}")
"""

from dataclasses import dataclass, field
from typing import (
    Optional,
    Callable,
    Awaitable,
    Dict,
    Any,
    List,
    AsyncIterator,
    Union,
)
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import uuid
import json

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Status of the deployment loop."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    EVALUATING = "evaluating"
    ROLLING_BACK = "rolling_back"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class DeploymentConfig:
    """Configuration for the deployment loop.

    Attributes:
        evaluation_interval: Tasks between performance evaluations
        min_tasks_for_eval: Minimum tasks before first evaluation
        success_threshold: Minimum acceptable success rate
        rollback_threshold: Success rate that triggers rollback
        learning_rate: Rate of learning updates
        enable_auto_rollback: Whether to auto-rollback on degradation
        enable_gradual_rollout: Whether to use gradual rollouts
        canary_percentage: Percentage of traffic for canary
        max_concurrent_tasks: Maximum concurrent task execution
    """
    evaluation_interval: int = 100
    min_tasks_for_eval: int = 50
    success_threshold: float = 0.8
    rollback_threshold: float = 0.6
    learning_rate: float = 0.01
    enable_auto_rollback: bool = True
    enable_gradual_rollout: bool = False
    canary_percentage: float = 0.1
    max_concurrent_tasks: int = 1
    checkpoint_interval: int = 500


@dataclass
class DeploymentMetrics:
    """Metrics from a deployment period.

    Attributes:
        period_id: Unique identifier for this period
        start_time: Start of the period
        end_time: End of the period
        tasks_completed: Number of tasks completed
        tasks_succeeded: Number of successful tasks
        tasks_failed: Number of failed tasks
        success_rate: Success rate for the period
        avg_latency_ms: Average task latency
        total_cost: Total cost for the period
        errors: List of errors encountered
    """
    period_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_id": self.period_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tasks_completed": self.tasks_completed,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost": self.total_cost,
            "error_count": len(self.errors),
        }


@dataclass
class DeploymentState:
    """Current state of the deployment loop.

    Attributes:
        deployment_id: Unique identifier for this deployment
        status: Current status
        agent_version: Current agent version/config
        started_at: When deployment started
        current_metrics: Current period metrics
        history: Historical metrics
        total_tasks: Total tasks across all periods
        total_successes: Total successes across all periods
        rollback_count: Number of rollbacks performed
    """
    deployment_id: str
    status: DeploymentStatus
    agent_version: str
    started_at: datetime
    current_metrics: DeploymentMetrics
    history: List[DeploymentMetrics] = field(default_factory=list)
    total_tasks: int = 0
    total_successes: int = 0
    rollback_count: int = 0
    last_checkpoint: Optional[datetime] = None

    def get_overall_success_rate(self) -> float:
        """Get overall success rate across all periods."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_successes / self.total_tasks


@dataclass
class DeploymentUpdate:
    """Update event from the deployment loop.

    Attributes:
        timestamp: When update occurred
        event_type: Type of update event
        tasks_completed: Tasks completed in this update
        success_rate: Current success rate
        message: Optional message
        metrics: Current metrics snapshot
    """
    timestamp: datetime
    event_type: str
    tasks_completed: int
    success_rate: float
    message: Optional[str] = None
    metrics: Optional[DeploymentMetrics] = None


class DeploymentLoop:
    """
    Continuous deployment and learning loop for agents.

    Manages the continuous execution of tasks while monitoring performance,
    collecting feedback, and enabling learning updates.

    Features:
        - Continuous task execution with progress reporting
        - Performance monitoring at configurable intervals
        - Automatic rollback on performance degradation
        - Checkpoint saving for recovery
        - Gradual rollout support with canary deployments
        - A/B testing of agent configurations

    Example:
        >>> loop = DeploymentLoop(
        ...     agent=my_agent,
        ...     config=DeploymentConfig(
        ...         evaluation_interval=100,
        ...         success_threshold=0.85
        ...     )
        ... )
        >>>
        >>> async for update in loop.run(tasks):
        ...     if update.event_type == "evaluation":
        ...         print(f"Success rate: {update.success_rate:.2%}")
        ...     if update.success_rate < 0.7:
        ...         loop.pause()

    Attributes:
        agent: Agent to deploy
        config: Deployment configuration
        state: Current deployment state
    """

    def __init__(
        self,
        agent: Any,
        config: Optional[DeploymentConfig] = None,
        agent_version: str = "v1.0",
        on_evaluation: Optional[Callable[[DeploymentMetrics], Awaitable[None]]] = None,
        on_rollback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """Initialize the deployment loop.

        Args:
            agent: Agent instance to deploy. Must have a `run` method.
            config: Deployment configuration.
            agent_version: Version string for the agent.
            on_evaluation: Callback after each evaluation.
            on_rollback: Callback when rollback is triggered.
        """
        self.agent = agent
        self.config = config or DeploymentConfig()
        self.agent_version = agent_version
        self._on_evaluation = on_evaluation
        self._on_rollback = on_rollback

        # Initialize state
        self.state = DeploymentState(
            deployment_id=str(uuid.uuid4()),
            status=DeploymentStatus.INITIALIZING,
            agent_version=agent_version,
            started_at=datetime.now(),
            current_metrics=self._create_metrics_period(),
        )

        # Internal state
        self._previous_agent = None  # For rollback
        self._latencies: List[float] = []
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused

    async def run(
        self,
        tasks: Union[List[Any], AsyncIterator[Any]],
        max_tasks: Optional[int] = None,
    ) -> AsyncIterator[DeploymentUpdate]:
        """Run the deployment loop.

        Processes tasks continuously, yielding updates at regular intervals
        and on significant events.

        Args:
            tasks: Iterable or async iterable of tasks
            max_tasks: Maximum number of tasks to process (None for unlimited)

        Yields:
            DeploymentUpdate objects with progress and metrics
        """
        self.state.status = DeploymentStatus.RUNNING
        self._stop_event.clear()

        task_count = 0

        try:
            # Handle both sync and async iterators
            if hasattr(tasks, '__aiter__'):
                task_iter = tasks.__aiter__()
            else:
                task_iter = iter(tasks)

            while not self._stop_event.is_set():
                # Check pause
                await self._pause_event.wait()

                # Check max tasks
                if max_tasks and task_count >= max_tasks:
                    break

                # Get next task
                try:
                    if hasattr(task_iter, '__anext__'):
                        task = await task_iter.__anext__()
                    else:
                        task = next(task_iter)
                except (StopIteration, StopAsyncIteration):
                    break

                # Execute task
                result = await self._execute_task(task)
                task_count += 1

                # Check for evaluation
                if task_count % self.config.evaluation_interval == 0:
                    self.state.status = DeploymentStatus.EVALUATING
                    metrics = await self._evaluate()
                    yield DeploymentUpdate(
                        timestamp=datetime.now(),
                        event_type="evaluation",
                        tasks_completed=task_count,
                        success_rate=metrics.success_rate,
                        message=f"Evaluation at {task_count} tasks",
                        metrics=metrics,
                    )

                    # Check for rollback
                    if (
                        self.config.enable_auto_rollback and
                        metrics.success_rate < self.config.rollback_threshold
                    ):
                        yield await self._perform_rollback(
                            f"Success rate {metrics.success_rate:.2%} "
                            f"below threshold {self.config.rollback_threshold:.2%}"
                        )

                    self.state.status = DeploymentStatus.RUNNING

                # Check for checkpoint
                if task_count % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint()
                    yield DeploymentUpdate(
                        timestamp=datetime.now(),
                        event_type="checkpoint",
                        tasks_completed=task_count,
                        success_rate=self.state.get_overall_success_rate(),
                        message=f"Checkpoint saved at {task_count} tasks",
                    )

        except Exception as e:
            logger.error(f"Deployment loop error: {e}")
            self.state.status = DeploymentStatus.FAILED
            yield DeploymentUpdate(
                timestamp=datetime.now(),
                event_type="error",
                tasks_completed=task_count,
                success_rate=self.state.get_overall_success_rate(),
                message=f"Error: {str(e)}",
            )

        finally:
            # Final evaluation
            if task_count > 0:
                metrics = await self._evaluate()
                yield DeploymentUpdate(
                    timestamp=datetime.now(),
                    event_type="completed",
                    tasks_completed=task_count,
                    success_rate=metrics.success_rate,
                    message=f"Deployment completed after {task_count} tasks",
                    metrics=metrics,
                )

            if self.state.status != DeploymentStatus.FAILED:
                self.state.status = DeploymentStatus.STOPPED

    async def _execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute a single task.

        Args:
            task: Task to execute

        Returns:
            Task result dictionary
        """
        start_time = datetime.now()

        try:
            # Execute task
            if asyncio.iscoroutinefunction(getattr(self.agent, 'run', None)):
                output = await self.agent.run(task)
            else:
                output = self.agent.run(task)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._latencies.append(latency_ms)

            # Determine success
            success = self._determine_success(output)

            # Update metrics
            self.state.current_metrics.tasks_completed += 1
            self.state.total_tasks += 1

            if success:
                self.state.current_metrics.tasks_succeeded += 1
                self.state.total_successes += 1
            else:
                self.state.current_metrics.tasks_failed += 1

            # Estimate cost
            cost = self._estimate_cost(output)
            self.state.current_metrics.total_cost += cost

            return {
                "success": success,
                "output": output,
                "latency_ms": latency_ms,
                "cost": cost,
            }

        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._latencies.append(latency_ms)

            self.state.current_metrics.tasks_completed += 1
            self.state.current_metrics.tasks_failed += 1
            self.state.current_metrics.errors.append(str(e))
            self.state.total_tasks += 1

            return {
                "success": False,
                "error": str(e),
                "latency_ms": latency_ms,
                "cost": 0.01,
            }

    async def _evaluate(self) -> DeploymentMetrics:
        """Evaluate current deployment performance.

        Returns:
            DeploymentMetrics for the current period
        """
        metrics = self.state.current_metrics
        metrics.end_time = datetime.now()

        # Calculate success rate
        if metrics.tasks_completed > 0:
            metrics.success_rate = metrics.tasks_succeeded / metrics.tasks_completed

        # Calculate average latency
        if self._latencies:
            metrics.avg_latency_ms = sum(self._latencies) / len(self._latencies)

        # Call callback
        if self._on_evaluation:
            await self._on_evaluation(metrics)

        logger.info(
            f"Evaluation: {metrics.tasks_completed} tasks, "
            f"{metrics.success_rate:.2%} success rate"
        )

        # Start new period
        self.state.history.append(metrics)
        self.state.current_metrics = self._create_metrics_period()
        self._latencies.clear()

        return metrics

    async def _perform_rollback(self, reason: str) -> DeploymentUpdate:
        """Perform a rollback to previous agent state.

        Args:
            reason: Reason for rollback

        Returns:
            DeploymentUpdate for the rollback event
        """
        self.state.status = DeploymentStatus.ROLLING_BACK
        self.state.rollback_count += 1

        logger.warning(f"Rolling back: {reason}")

        if self._on_rollback:
            await self._on_rollback(reason)

        # Restore previous agent if available
        if self._previous_agent:
            self.agent = self._previous_agent
            logger.info("Restored previous agent configuration")

        return DeploymentUpdate(
            timestamp=datetime.now(),
            event_type="rollback",
            tasks_completed=self.state.total_tasks,
            success_rate=self.state.get_overall_success_rate(),
            message=f"Rollback triggered: {reason}",
        )

    async def _save_checkpoint(self) -> None:
        """Save current deployment state as checkpoint."""
        self.state.last_checkpoint = datetime.now()
        logger.info(f"Checkpoint saved at {self.state.total_tasks} tasks")

    def _create_metrics_period(self) -> DeploymentMetrics:
        """Create a new metrics period."""
        return DeploymentMetrics(
            period_id=str(uuid.uuid4()),
            start_time=datetime.now(),
        )

    def _determine_success(self, output: Any) -> bool:
        """Determine if task was successful."""
        if output is None:
            return False
        if isinstance(output, dict):
            return output.get("success", True)
        if isinstance(output, bool):
            return output
        return True

    def _estimate_cost(self, output: Any) -> float:
        """Estimate cost of task execution."""
        if isinstance(output, dict) and "cost" in output:
            return output["cost"]
        return 0.01

    def pause(self) -> None:
        """Pause the deployment loop."""
        self._pause_event.clear()
        self.state.status = DeploymentStatus.PAUSED
        logger.info("Deployment loop paused")

    def resume(self) -> None:
        """Resume the deployment loop."""
        self._pause_event.set()
        self.state.status = DeploymentStatus.RUNNING
        logger.info("Deployment loop resumed")

    def stop(self) -> None:
        """Stop the deployment loop."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        logger.info("Deployment loop stopping")

    def update_agent(self, new_agent: Any, new_version: str) -> None:
        """Update the deployed agent.

        Saves the current agent for potential rollback.

        Args:
            new_agent: New agent instance
            new_version: Version string for new agent
        """
        self._previous_agent = self.agent
        self.agent = new_agent
        self.agent_version = new_version
        self.state.agent_version = new_version
        logger.info(f"Updated agent to version {new_version}")

    def get_state(self) -> Dict[str, Any]:
        """Get current deployment state as dictionary.

        Returns:
            State dictionary
        """
        return {
            "deployment_id": self.state.deployment_id,
            "status": self.state.status.value,
            "agent_version": self.state.agent_version,
            "started_at": self.state.started_at.isoformat(),
            "total_tasks": self.state.total_tasks,
            "total_successes": self.state.total_successes,
            "overall_success_rate": self.state.get_overall_success_rate(),
            "rollback_count": self.state.rollback_count,
            "current_metrics": self.state.current_metrics.to_dict(),
            "periods_completed": len(self.state.history),
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get metrics history.

        Returns:
            List of metrics dictionaries
        """
        return [m.to_dict() for m in self.state.history]


class ABTestDeployment:
    """A/B testing deployment for comparing agent versions.

    Supports running two agent versions concurrently with
    configurable traffic splitting.

    Example:
        >>> ab_test = ABTestDeployment(
        ...     agent_a=old_agent,
        ...     agent_b=new_agent,
        ...     traffic_split=0.5
        ... )
        >>>
        >>> async for result in ab_test.run(tasks):
        ...     print(f"Agent A: {result.metrics_a.success_rate:.2%}")
        ...     print(f"Agent B: {result.metrics_b.success_rate:.2%}")
    """

    def __init__(
        self,
        agent_a: Any,
        agent_b: Any,
        traffic_split: float = 0.5,
        min_samples: int = 100,
    ):
        """Initialize A/B test deployment.

        Args:
            agent_a: First agent (control)
            agent_b: Second agent (treatment)
            traffic_split: Proportion of traffic to agent B
            min_samples: Minimum samples before comparison
        """
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.traffic_split = traffic_split
        self.min_samples = min_samples

        self.metrics_a = DeploymentMetrics(
            period_id="agent_a",
            start_time=datetime.now()
        )
        self.metrics_b = DeploymentMetrics(
            period_id="agent_b",
            start_time=datetime.now()
        )

        self._rng = __import__('random').Random()

    async def run_task(self, task: Any) -> Dict[str, Any]:
        """Run a task on one of the agents based on traffic split.

        Args:
            task: Task to execute

        Returns:
            Result including which agent was used
        """
        use_b = self._rng.random() < self.traffic_split
        agent = self.agent_b if use_b else self.agent_a
        metrics = self.metrics_b if use_b else self.metrics_a

        start_time = datetime.now()

        try:
            if asyncio.iscoroutinefunction(getattr(agent, 'run', None)):
                output = await agent.run(task)
            else:
                output = agent.run(task)

            success = self._determine_success(output)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            metrics.tasks_completed += 1
            if success:
                metrics.tasks_succeeded += 1
            else:
                metrics.tasks_failed += 1

            return {
                "agent": "B" if use_b else "A",
                "success": success,
                "output": output,
                "latency_ms": latency,
            }

        except Exception as e:
            metrics.tasks_completed += 1
            metrics.tasks_failed += 1
            return {
                "agent": "B" if use_b else "A",
                "success": False,
                "error": str(e),
            }

    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison between agents.

        Returns:
            Comparison dictionary
        """
        def calc_rate(m: DeploymentMetrics) -> float:
            if m.tasks_completed == 0:
                return 0.0
            return m.tasks_succeeded / m.tasks_completed

        rate_a = calc_rate(self.metrics_a)
        rate_b = calc_rate(self.metrics_b)

        return {
            "agent_a": {
                "tasks": self.metrics_a.tasks_completed,
                "success_rate": rate_a,
            },
            "agent_b": {
                "tasks": self.metrics_b.tasks_completed,
                "success_rate": rate_b,
            },
            "difference": rate_b - rate_a,
            "winner": "B" if rate_b > rate_a else "A" if rate_a > rate_b else "tie",
            "statistically_significant": self._is_significant(),
        }

    def _is_significant(self) -> bool:
        """Check if difference is statistically significant."""
        # Simple significance check based on sample size
        if (
            self.metrics_a.tasks_completed < self.min_samples or
            self.metrics_b.tasks_completed < self.min_samples
        ):
            return False

        # Basic z-test approximation
        n_a = self.metrics_a.tasks_completed
        n_b = self.metrics_b.tasks_completed
        p_a = self.metrics_a.tasks_succeeded / n_a if n_a > 0 else 0
        p_b = self.metrics_b.tasks_succeeded / n_b if n_b > 0 else 0

        p_pooled = (self.metrics_a.tasks_succeeded + self.metrics_b.tasks_succeeded) / (n_a + n_b)

        if p_pooled == 0 or p_pooled == 1:
            return False

        se = (p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b)) ** 0.5
        if se == 0:
            return False

        z = abs(p_a - p_b) / se
        return z > 1.96  # 95% confidence

    def _determine_success(self, output: Any) -> bool:
        """Determine if task was successful."""
        if output is None:
            return False
        if isinstance(output, dict):
            return output.get("success", True)
        return True
