"""
HotpotQA Adapter

Adapter for HotpotQA multi-hop reasoning benchmark.

HotpotQA is a question answering dataset that requires reasoning over
multiple Wikipedia passages to answer questions.

Reference: https://hotpotqa.github.io/

This adapter supports:
- Loading questions from HotpotQA
- Evaluating answers with F1 and EM metrics
- Tracking reasoning chain quality

Example:
    >>> adapter = HotpotQAAdapter()
    >>> tasks = adapter.load_tasks(n=200, difficulty="hard")
    >>> for task in tasks:
    ...     answer, reasoning = agent.answer(task.question, task.context)
    ...     result = adapter.evaluate(task, answer)
    ...     print(f"F1: {result.f1_score:.2f}")
"""

from __future__ import annotations

import logging
import json
import re
import string
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from collections import Counter

import numpy as np

from .base_adapter import BenchmarkAdapter, BenchmarkTask, BenchmarkResult

logger = logging.getLogger(__name__)


class HotpotQAType(Enum):
    """HotpotQA question types."""
    BRIDGE = "bridge"
    COMPARISON = "comparison"


class HotpotQADifficulty(Enum):
    """HotpotQA difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class HotpotQATask(BenchmarkTask):
    """HotpotQA task definition.

    Attributes:
        question: The question to answer
        context: Supporting passages
        supporting_facts: Sentences needed for answer
        answer: Ground truth answer
        question_type: Bridge or comparison
        difficulty: Question difficulty
    """
    question: str = ""
    context: List[Tuple[str, List[str]]] = field(default_factory=list)  # [(title, [sentences])]
    supporting_facts: List[Tuple[str, int]] = field(default_factory=list)  # [(title, sent_idx)]
    answer: str = ""
    question_type: HotpotQAType = HotpotQAType.BRIDGE
    difficulty: HotpotQADifficulty = HotpotQADifficulty.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "question": self.question,
            "context": self.context,
            "supporting_facts": self.supporting_facts,
            "answer": self.answer,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value
        })
        return base

    def get_context_text(self) -> str:
        """Get formatted context as a single string."""
        parts = []
        for title, sentences in self.context:
            parts.append(f"## {title}")
            parts.extend(sentences)
        return "\n".join(parts)

    def get_supporting_sentences(self) -> List[str]:
        """Get the supporting fact sentences."""
        sentences = []
        for title, sent_idx in self.supporting_facts:
            for ctx_title, ctx_sents in self.context:
                if ctx_title == title and sent_idx < len(ctx_sents):
                    sentences.append(ctx_sents[sent_idx])
                    break
        return sentences


@dataclass
class HotpotQAResult(BenchmarkResult):
    """HotpotQA evaluation result.

    Attributes:
        f1_score: F1 score for answer tokens
        exact_match: Whether answer exactly matches
        supporting_precision: Precision of retrieved supporting facts
        supporting_recall: Recall of supporting facts
        supporting_f1: F1 for supporting facts
        reasoning_steps: Number of reasoning steps identified
    """
    f1_score: float = 0.0
    exact_match: bool = False
    supporting_precision: float = 0.0
    supporting_recall: float = 0.0
    supporting_f1: float = 0.0
    reasoning_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "f1_score": self.f1_score,
            "exact_match": self.exact_match,
            "supporting_precision": self.supporting_precision,
            "supporting_recall": self.supporting_recall,
            "supporting_f1": self.supporting_f1,
            "reasoning_steps": self.reasoning_steps
        })
        return base


# Sample HotpotQA questions for demonstration
SAMPLE_HOTPOTQA_TASKS = [
    {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "context": [
            ("Scott Derrickson", [
                "Scott Derrickson (born July 16, 1966) is an American director.",
                "He is best known for his horror films."
            ]),
            ("Ed Wood", [
                "Edward Davis Wood Jr. was an American filmmaker.",
                "He is known for his low-budget B movies."
            ])
        ],
        "supporting_facts": [("Scott Derrickson", 0), ("Ed Wood", 0)],
        "difficulty": "medium"
    },
    {
        "id": "5a8b5a1c5542995d1e6f13a8",
        "question": "What is the name of the director of 'Doctor Strange' who also directed 'Sinister'?",
        "answer": "Scott Derrickson",
        "type": "bridge",
        "context": [
            ("Doctor Strange", [
                "Doctor Strange is a 2016 American superhero film.",
                "It was directed by Scott Derrickson."
            ]),
            ("Sinister", [
                "Sinister is a 2012 American horror film.",
                "The film was directed by Scott Derrickson."
            ]),
            ("Scott Derrickson", [
                "Scott Derrickson is an American director.",
                "He directed Doctor Strange and Sinister."
            ])
        ],
        "supporting_facts": [("Doctor Strange", 1), ("Sinister", 1)],
        "difficulty": "medium"
    },
    {
        "id": "5a8b5c3a5542995d1e6f13c2",
        "question": "Which film has more Academy Award nominations, The Shawshank Redemption or Forrest Gump?",
        "answer": "Forrest Gump",
        "type": "comparison",
        "context": [
            ("The Shawshank Redemption", [
                "The Shawshank Redemption is a 1994 drama film.",
                "It received seven Academy Award nominations."
            ]),
            ("Forrest Gump", [
                "Forrest Gump is a 1994 American comedy-drama.",
                "The film received thirteen Academy Award nominations.",
                "It won six awards including Best Picture."
            ])
        ],
        "supporting_facts": [("The Shawshank Redemption", 1), ("Forrest Gump", 1)],
        "difficulty": "easy"
    },
    {
        "id": "5a8b5d7f5542995d1e6f13e5",
        "question": "Who founded the company that produced 'Toy Story'?",
        "answer": "Edwin Catmull and Alvy Ray Smith",
        "type": "bridge",
        "context": [
            ("Toy Story", [
                "Toy Story is a 1995 computer-animated film.",
                "It was produced by Pixar Animation Studios."
            ]),
            ("Pixar", [
                "Pixar Animation Studios is an American animation studio.",
                "It was founded by Edwin Catmull and Alvy Ray Smith in 1986."
            ])
        ],
        "supporting_facts": [("Toy Story", 1), ("Pixar", 1)],
        "difficulty": "hard"
    },
    {
        "id": "5a8b5e9a5542995d1e6f1401",
        "question": "Is the Eiffel Tower taller than the Statue of Liberty?",
        "answer": "yes",
        "type": "comparison",
        "context": [
            ("Eiffel Tower", [
                "The Eiffel Tower is a wrought-iron lattice tower.",
                "It stands 330 meters tall including antennas."
            ]),
            ("Statue of Liberty", [
                "The Statue of Liberty is a copper statue.",
                "The statue is 93 meters tall from ground to torch tip."
            ])
        ],
        "supporting_facts": [("Eiffel Tower", 1), ("Statue of Liberty", 1)],
        "difficulty": "easy"
    }
]


class HotpotQAAdapter(BenchmarkAdapter[HotpotQATask, HotpotQAResult]):
    """Adapter for HotpotQA multi-hop reasoning benchmark.

    This adapter provides access to HotpotQA questions requiring
    multi-hop reasoning over multiple passages.

    Example:
        >>> adapter = HotpotQAAdapter()
        >>>
        >>> # Load questions
        >>> tasks = adapter.load_tasks(n=200)
        >>>
        >>> # Evaluate agent
        >>> for task in tasks:
        ...     answer = agent.answer(task.question, task.get_context_text())
        ...     result = adapter.evaluate(task, answer)
        ...     print(f"F1: {result.f1_score:.3f}")
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        seed: int = 42,
        use_sample_data: bool = True
    ):
        """Initialize HotpotQA adapter.

        Args:
            data_path: Path to HotpotQA data
            seed: Random seed
            use_sample_data: Use built-in sample data if real data unavailable
        """
        super().__init__(data_path, seed)
        self.use_sample_data = use_sample_data
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "HotpotQA"

    @property
    def description(self) -> str:
        return "Multi-hop reasoning benchmark requiring evidence from multiple passages"

    def load_tasks(
        self,
        n: Optional[int] = None,
        difficulty: Optional[str] = None,
        question_type: Optional[str] = None,
        **kwargs
    ) -> List[HotpotQATask]:
        """Load HotpotQA tasks.

        Args:
            n: Number of tasks to load
            difficulty: Filter by difficulty ("easy", "medium", "hard")
            question_type: Filter by type ("bridge", "comparison")

        Returns:
            List of HotpotQA tasks
        """
        # Try to load from file
        if self.data_path and Path(self.data_path).exists():
            tasks = self._load_from_file()
        elif self.use_sample_data:
            tasks = self._generate_sample_tasks(n or 200)
        else:
            raise FileNotFoundError(
                f"HotpotQA data not found at {self.data_path}. "
                "Set use_sample_data=True to use built-in samples."
            )

        # Filter by difficulty
        if difficulty is not None:
            diff = HotpotQADifficulty(difficulty)
            tasks = [t for t in tasks if t.difficulty == diff]

        # Filter by question type
        if question_type is not None:
            qtype = HotpotQAType(question_type)
            tasks = [t for t in tasks if t.question_type == qtype]

        # Limit number of tasks
        if n is not None and len(tasks) > n:
            indices = self._rng.choice(len(tasks), size=n, replace=False)
            tasks = [tasks[i] for i in sorted(indices)]

        self._tasks = tasks
        self._loaded = True

        logger.info(f"Loaded {len(tasks)} HotpotQA tasks")
        return tasks

    def _load_from_file(self) -> List[HotpotQATask]:
        """Load tasks from file."""
        path = Path(self.data_path) / "hotpotqa.json"
        if not path.exists():
            logger.warning(f"File not found: {path}, using sample data")
            return self._generate_sample_tasks(200)

        with open(path) as f:
            data = json.load(f)

        tasks = []
        for item in data:
            task = HotpotQATask(
                task_id=item["_id"],
                query=item["question"],
                question=item["question"],
                context=[(c[0], c[1]) for c in item.get("context", [])],
                supporting_facts=[(s[0], s[1]) for s in item.get("supporting_facts", [])],
                answer=item.get("answer", ""),
                question_type=HotpotQAType(item.get("type", "bridge")),
                difficulty=HotpotQADifficulty(item.get("level", "medium"))
            )
            tasks.append(task)

        return tasks

    def _generate_sample_tasks(self, n: int) -> List[HotpotQATask]:
        """Generate sample tasks for demonstration."""
        tasks = []

        for item in SAMPLE_HOTPOTQA_TASKS[:n]:
            task = HotpotQATask(
                task_id=item["id"],
                query=item["question"],
                question=item["question"],
                context=item["context"],
                supporting_facts=item["supporting_facts"],
                answer=item["answer"],
                question_type=HotpotQAType(item["type"]),
                difficulty=HotpotQADifficulty(item["difficulty"]),
                metadata={"source": "sample"}
            )
            tasks.append(task)

        # Pad with generated tasks if needed
        while len(tasks) < n:
            idx = len(tasks)
            qtype = HotpotQAType.BRIDGE if idx % 2 == 0 else HotpotQAType.COMPARISON
            difficulty = HotpotQADifficulty(["easy", "medium", "hard"][idx % 3])

            task = HotpotQATask(
                task_id=f"sample_{idx:04d}",
                query=f"Sample question {idx}",
                question=f"Sample question {idx}",
                context=[("Article", [f"Sample context {idx}"])],
                answer=f"sample_answer_{idx}",
                question_type=qtype,
                difficulty=difficulty,
                metadata={"source": "generated"}
            )
            tasks.append(task)

        return tasks[:n]

    def evaluate(
        self,
        task: HotpotQATask,
        answer: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None
    ) -> HotpotQAResult:
        """Evaluate agent answer.

        Args:
            task: The task
            answer: Agent's answer
            supporting_facts: Optional list of supporting facts identified

        Returns:
            HotpotQAResult with evaluation metrics
        """
        # Compute answer metrics
        f1, precision, recall = self._compute_f1(task.answer, answer)
        exact_match = self._compute_em(task.answer, answer)

        # Compute supporting facts metrics if provided
        if supporting_facts:
            sp_precision, sp_recall, sp_f1 = self._compute_supporting_facts_metrics(
                task.supporting_facts, supporting_facts
            )
        else:
            sp_precision, sp_recall, sp_f1 = 0.0, 0.0, 0.0

        # Count reasoning steps (heuristic from answer length)
        reasoning_steps = self._estimate_reasoning_steps(answer)

        # Overall score
        score = f1
        success = exact_match or f1 >= 0.5

        return HotpotQAResult(
            task_id=task.task_id,
            success=success,
            output=answer,
            score=score,
            f1_score=f1,
            exact_match=exact_match,
            supporting_precision=sp_precision,
            supporting_recall=sp_recall,
            supporting_f1=sp_f1,
            reasoning_steps=reasoning_steps
        )

    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = ''.join(c for c in text if c not in string.punctuation)

        # Remove articles
        articles = {'a', 'an', 'the'}
        words = text.split()
        words = [w for w in words if w not in articles]

        # Remove extra whitespace
        text = ' '.join(words)

        return text.strip()

    def _get_tokens(self, text: str) -> List[str]:
        """Get tokens from normalized text."""
        return self._normalize_answer(text).split()

    def _compute_f1(
        self,
        gold: str,
        pred: str
    ) -> Tuple[float, float, float]:
        """Compute F1 score between gold and predicted answers.

        Returns:
            Tuple of (f1, precision, recall)
        """
        gold_tokens = self._get_tokens(gold)
        pred_tokens = self._get_tokens(pred)

        if not gold_tokens or not pred_tokens:
            return (1.0, 1.0, 1.0) if gold_tokens == pred_tokens else (0.0, 0.0, 0.0)

        gold_counter = Counter(gold_tokens)
        pred_counter = Counter(pred_tokens)

        common = gold_counter & pred_counter
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1, precision, recall

    def _compute_em(self, gold: str, pred: str) -> bool:
        """Compute exact match."""
        return self._normalize_answer(gold) == self._normalize_answer(pred)

    def _compute_supporting_facts_metrics(
        self,
        gold_facts: List[Tuple[str, int]],
        pred_facts: List[Tuple[str, int]]
    ) -> Tuple[float, float, float]:
        """Compute supporting facts precision, recall, F1.

        Returns:
            Tuple of (precision, recall, f1)
        """
        gold_set = set(gold_facts)
        pred_set = set(pred_facts)

        if not pred_set:
            return 0.0, 0.0, 0.0

        common = gold_set & pred_set

        precision = len(common) / len(pred_set)
        recall = len(common) / len(gold_set) if gold_set else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        return precision, recall, f1

    def _estimate_reasoning_steps(self, answer: str) -> int:
        """Estimate number of reasoning steps from answer.

        Heuristic based on answer structure.
        """
        # Count reasoning indicators
        indicators = [
            r'\bfirst\b', r'\bthen\b', r'\bnext\b', r'\bfinally\b',
            r'\bbecause\b', r'\btherefore\b', r'\bso\b', r'\bthus\b',
            r'\bstep \d+', r'\d\)'
        ]

        count = 0
        answer_lower = answer.lower()
        for pattern in indicators:
            count += len(re.findall(pattern, answer_lower))

        # At least 1 step if answer exists
        return max(1, count)

    def get_statistics(self, results: List[HotpotQAResult]) -> Dict[str, Any]:
        """Get detailed statistics for HotpotQA results."""
        base_stats = super().get_statistics(results)

        if not results:
            return base_stats

        # Add HotpotQA specific stats
        em_count = sum(1 for r in results if r.exact_match)
        mean_f1 = sum(r.f1_score for r in results) / len(results)
        mean_sp_f1 = sum(r.supporting_f1 for r in results) / len(results)

        base_stats.update({
            "exact_match_rate": em_count / len(results),
            "mean_f1": mean_f1,
            "mean_supporting_f1": mean_sp_f1,
            "mean_reasoning_steps": sum(r.reasoning_steps for r in results) / len(results)
        })

        return base_stats


def download_hotpotqa_data(
    output_dir: str = "data/hotpotqa",
    split: str = "dev"
) -> str:
    """Download HotpotQA data.

    Args:
        output_dir: Directory to save data
        split: Data split ("train", "dev", "test")

    Returns:
        Path to downloaded data
    """
    instructions = """
    HotpotQA data can be obtained from:
    https://hotpotqa.github.io/

    Using HuggingFace datasets:
    pip install datasets
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor")

    Or download directly:
    - Training set: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    - Dev set: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    """
    logger.info(instructions)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir
