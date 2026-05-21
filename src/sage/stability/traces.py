"""Trace logging schema for the Adaptive Stability Controller (ASC).

Pydantic v2 models for per-turn trace events.  All experiments write JSONL
traces using TraceWriter; analyses read these files — no in-memory coupling.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Pydantic models ────────────────────────────────────────────────────────────

class MonitorSignalsModel(BaseModel):
    drift_score: float = Field(ge=0.0, le=1.0)
    oscillation_score: float = Field(ge=0.0, le=1.0)
    fidelity_score: float = Field(ge=0.0, le=1.0)
    convergence_progress: float = Field(ge=0.0, le=1.0)
    turn: int = Field(ge=1)
    cost_so_far: float = Field(ge=0.0)


class InterventionDecisionModel(BaseModel):
    intervention_name: Optional[str]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class TraceEvent(BaseModel):
    task_id: str
    seed: int
    turn: int
    timestamp_iso: str
    agent_action: Optional[str]
    monitor_signals: MonitorSignalsModel
    controller_decision: Optional[InterventionDecisionModel]
    intervention_applied: Optional[str]
    cost_this_turn: float
    cumulative_cost: float
    task_outcome: Optional[Literal["in_progress", "success", "failure"]]


class ManifestModel(BaseModel):
    git_sha: str
    seed: int
    env_hash: str
    config: dict
    output_files: dict  # filename -> sha256


# ── I/O helpers ────────────────────────────────────────────────────────────────

class TraceWriter:
    """Append-mode JSONL writer for TraceEvent objects."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, event: TraceEvent) -> None:
        self._fh.write(event.model_dump_json() + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def read_traces(path: Path) -> list[TraceEvent]:
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(TraceEvent.model_validate_json(line))
    return events


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    directory: Path,
    seed: int,
    config: dict,
    output_files: list[Path],
    git_sha: str = "unknown",
    env_hash: str = "unknown",
) -> Path:
    manifest = ManifestModel(
        git_sha=git_sha,
        seed=seed,
        env_hash=env_hash,
        config=config,
        output_files={p.name: sha256_file(p) for p in output_files if p.exists()},
    )
    dest = directory / "MANIFEST.json"
    dest.write_text(manifest.model_dump_json(indent=2))
    return dest


def _get_git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_env_hash() -> str:
    import numpy as np
    import sklearn
    import scipy
    s = f"numpy={np.__version__},sklearn={sklearn.__version__},scipy={scipy.__version__}"
    return hashlib.sha256(s.encode()).hexdigest()[:12]
