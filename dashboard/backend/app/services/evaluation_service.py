"""Evaluation service for managing agent evaluations with real Ollama integration."""
import asyncio
import uuid
import time
import httpx
from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.evaluation import (
    EvaluationConfig,
    EvaluationStatus,
    EvaluationStatusEnum,
    EvaluationResult,
    TaskResult,
)
from app.config import settings

# Try to import toolkit metrics
try:
    from sage.evaluation.metrics import TaskCostBreakdown, compute_cnsr
    TOOLKIT_AVAILABLE = True
except ImportError:
    TOOLKIT_AVAILABLE = False

    class TaskCostBreakdown:
        def __init__(self, inference_cost=0.0, tool_cost=0.0, latency_cost=0.0, human_cost=0.0):
            self.inference_cost = inference_cost
            self.tool_cost = tool_cost
            self.latency_cost = latency_cost
            self.human_cost = human_cost

        @property
        def total_cost(self):
            return self.inference_cost + self.tool_cost + self.latency_cost + self.human_cost

    def compute_cnsr(success_rate: float, mean_cost: float) -> float:
        if mean_cost <= 0:
            return 0.0
        return success_rate / mean_cost


# Sample benchmark tasks for evaluation
BENCHMARK_TASKS = [
    {"prompt": "What is 2 + 2?", "expected_contains": ["4"]},
    {"prompt": "What is the capital of France?", "expected_contains": ["Paris"]},
    {"prompt": "List 3 primary colors.", "expected_contains": ["red", "blue", "yellow"]},
    {"prompt": "What is the chemical symbol for water?", "expected_contains": ["H2O"]},
    {"prompt": "Who wrote Romeo and Juliet?", "expected_contains": ["Shakespeare"]},
    {"prompt": "What is the largest planet in our solar system?", "expected_contains": ["Jupiter"]},
    {"prompt": "What is the square root of 144?", "expected_contains": ["12"]},
    {"prompt": "In what year did World War II end?", "expected_contains": ["1945"]},
    {"prompt": "What is the boiling point of water in Celsius?", "expected_contains": ["100"]},
    {"prompt": "Name the three states of matter.", "expected_contains": ["solid", "liquid", "gas"]},
    {"prompt": "What is Python in programming?", "expected_contains": ["language", "programming"]},
    {"prompt": "What does CPU stand for?", "expected_contains": ["Central", "Processing", "Unit"]},
    {"prompt": "What is machine learning?", "expected_contains": ["learn", "data", "algorithm"]},
    {"prompt": "Explain what an API is in one sentence.", "expected_contains": ["interface", "application"]},
    {"prompt": "What is the purpose of a database?", "expected_contains": ["store", "data"]},
]


class EvaluationService:
    """Service for managing evaluations with real Ollama integration."""

    def __init__(self):
        self._evaluations: Dict[str, EvaluationStatus] = {}
        self._results: Dict[str, EvaluationResult] = {}
        self._running: Dict[str, bool] = {}
        self._ollama_base = f"http://{settings.ollama_host}:{settings.ollama_port}"

    async def _call_ollama(self, model: str, prompt: str, timeout: float = 30.0) -> Dict:
        """Call Ollama API and return response with metrics."""
        start_time = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self._ollama_base}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": 100}
                    }
                )
                response.raise_for_status()
                data = response.json()

                elapsed = time.time() - start_time

                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "eval_count": data.get("eval_count", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "total_duration": data.get("total_duration", 0) / 1e9,  # Convert to seconds
                    "elapsed": elapsed,
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "response": "",
                    "eval_count": 0,
                    "prompt_eval_count": 0,
                    "total_duration": 0,
                    "elapsed": time.time() - start_time,
                    "error": str(e)
                }

    def _evaluate_response(self, response: str, expected_contains: List[str]) -> bool:
        """Check if response contains expected keywords."""
        response_lower = response.lower()
        return any(keyword.lower() in response_lower for keyword in expected_contains)

    def _calculate_cost(self, prompt_tokens: int, output_tokens: int, duration_s: float) -> TaskCostBreakdown:
        """Calculate task cost using the 4-component model."""
        # Cost rates from settings
        inference_cost = (
            prompt_tokens * settings.input_token_rate +
            output_tokens * settings.output_token_rate
        )
        tool_cost = 0.0  # No tool calls for basic evaluation
        latency_cost = duration_s * settings.latency_rate
        human_cost = 0.0  # No human intervention needed

        return TaskCostBreakdown(
            inference_cost=inference_cost,
            tool_cost=tool_cost,
            latency_cost=latency_cost,
            human_cost=human_cost
        )

    async def list_evaluations(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[EvaluationStatus]:
        """List all evaluations."""
        evals = list(self._evaluations.values())
        if status:
            evals = [e for e in evals if e.status.value == status]
        return sorted(evals, key=lambda x: x.started_at, reverse=True)[:limit]

    async def create_evaluation(self, config: EvaluationConfig) -> EvaluationStatus:
        """Create a new evaluation."""
        eval_id = str(uuid.uuid4())[:8]

        status = EvaluationStatus(
            evaluation_id=eval_id,
            name=config.name,
            status=EvaluationStatusEnum.PENDING,
            progress=0.0,
            tasks_completed=0,
            tasks_total=config.num_tasks,
            current_success_rate=0.0,
            current_cnsr=0.0,
            current_cost=0.0,
            started_at=datetime.now(),
            model=config.model,
        )

        self._evaluations[eval_id] = status
        self._running[eval_id] = False

        return status

    async def get_evaluation(self, evaluation_id: str) -> Optional[EvaluationStatus]:
        """Get evaluation status."""
        return self._evaluations.get(evaluation_id)

    async def get_results(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Get evaluation results."""
        return self._results.get(evaluation_id)

    async def run_evaluation(self, evaluation_id: str):
        """Run an evaluation with real Ollama calls."""
        status = self._evaluations.get(evaluation_id)
        if not status:
            return

        self._running[evaluation_id] = True
        status.status = EvaluationStatusEnum.RUNNING

        task_results = []
        total_cost_breakdown = TaskCostBreakdown()
        successes = 0
        total_tokens = 0

        # Get model from status or use default
        model = getattr(status, 'model', None) or "gemma2:2b"

        # Use benchmark tasks, cycling if needed
        num_tasks = min(status.tasks_total, 100)  # Cap at 100 tasks

        for i in range(num_tasks):
            if not self._running.get(evaluation_id, False):
                status.status = EvaluationStatusEnum.CANCELLED
                break

            # Get task (cycle through benchmark tasks)
            task = BENCHMARK_TASKS[i % len(BENCHMARK_TASKS)]

            # Call Ollama
            result = await self._call_ollama(model, task["prompt"])

            # Evaluate response
            task_success = False
            if result["success"]:
                task_success = self._evaluate_response(
                    result["response"],
                    task["expected_contains"]
                )

            # Calculate cost
            prompt_tokens = result["prompt_eval_count"]
            output_tokens = result["eval_count"]
            cost = self._calculate_cost(
                prompt_tokens,
                output_tokens,
                result["elapsed"]
            )

            # Create task result
            task_result = TaskResult(
                task_id=f"task_{i+1}",
                success=task_success,
                cost=cost.total_cost,
                duration_seconds=result["elapsed"],
                steps_taken=1,
                tokens_used=prompt_tokens + output_tokens,
                error=result["error"],
            )
            task_results.append(task_result)

            # Accumulate metrics
            total_cost_breakdown = TaskCostBreakdown(
                inference_cost=total_cost_breakdown.inference_cost + cost.inference_cost,
                tool_cost=total_cost_breakdown.tool_cost + cost.tool_cost,
                latency_cost=total_cost_breakdown.latency_cost + cost.latency_cost,
                human_cost=total_cost_breakdown.human_cost + cost.human_cost,
            )
            total_tokens += prompt_tokens + output_tokens
            if task_success:
                successes += 1

            # Update status
            status.tasks_completed = i + 1
            status.progress = (i + 1) / num_tasks
            status.current_success_rate = successes / (i + 1)
            status.current_cost = total_cost_breakdown.total_cost
            mean_cost = total_cost_breakdown.total_cost / (i + 1)
            status.current_cnsr = compute_cnsr(status.current_success_rate, mean_cost)

        # Finalize
        if self._running.get(evaluation_id, False):
            status.status = EvaluationStatusEnum.COMPLETED
            status.tasks_total = num_tasks  # Update to actual tasks run

            success_rate = successes / num_tasks if num_tasks > 0 else 0
            mean_cost = total_cost_breakdown.total_cost / num_tasks if num_tasks > 0 else 0
            cnsr = compute_cnsr(success_rate, mean_cost)

            # Determine recommendation
            if success_rate >= 0.8 and cnsr >= 20:
                recommendation = "APPROVED"
                reason = "High success rate and excellent cost efficiency"
            elif success_rate >= 0.6 and cnsr >= 10:
                recommendation = "CONDITIONAL"
                reason = "Acceptable performance, consider optimization"
            else:
                recommendation = "REJECTED"
                reason = "Performance or cost efficiency below threshold"

            result = EvaluationResult(
                evaluation_id=evaluation_id,
                name=status.name,
                status=EvaluationStatusEnum.COMPLETED,
                config=EvaluationConfig(
                    name=status.name,
                    num_tasks=num_tasks,
                    model=model,
                ),
                total_tasks=num_tasks,
                successful_tasks=successes,
                success_rate=round(success_rate, 4),
                cnsr=round(cnsr, 2),
                total_cost=round(total_cost_breakdown.total_cost, 6),
                mean_cost=round(mean_cost, 6),
                total_tokens=total_tokens,
                cost_breakdown={
                    "inference": round(total_cost_breakdown.inference_cost, 6),
                    "tools": round(total_cost_breakdown.tool_cost, 6),
                    "latency": round(total_cost_breakdown.latency_cost, 6),
                    "human": round(total_cost_breakdown.human_cost, 6),
                },
                task_results=task_results,
                started_at=status.started_at,
                completed_at=datetime.now(),
                duration_seconds=(datetime.now() - status.started_at).total_seconds(),
                recommendation=recommendation,
                recommendation_reason=reason,
            )

            self._results[evaluation_id] = result

        self._running[evaluation_id] = False

    async def cancel_evaluation(self, evaluation_id: str) -> bool:
        """Cancel a running evaluation."""
        if evaluation_id in self._running:
            self._running[evaluation_id] = False
            if evaluation_id in self._evaluations:
                self._evaluations[evaluation_id].status = EvaluationStatusEnum.CANCELLED
            return True
        return False

    async def start_evaluation_background(self, evaluation_id: str):
        """Start evaluation in background."""
        asyncio.create_task(self.run_evaluation(evaluation_id))

    async def run_quick_demo(self, model: str = "gemma2:2b") -> Optional[EvaluationResult]:
        """Run a quick 5-task demo evaluation."""
        config = EvaluationConfig(
            name=f"Quick Demo ({model})",
            num_tasks=5,
            model=model,
        )
        status = await self.create_evaluation(config)
        await self.run_evaluation(status.evaluation_id)
        return self._results.get(status.evaluation_id)


# Singleton instance
evaluation_service = EvaluationService()
