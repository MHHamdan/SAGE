"""
SAGE — Stabilize, Assess, Govern, Enforce
=========================================

Reference implementation of the SAGE four-pillar framework for
deployment-ready LLM-based autonomous agents.

Pillars:
- Stabilize: closed-loop stability monitoring (sage.stability, sage.monitoring)
- Assess:    cost-normalised success rate, autonomy taxonomy (sage.evaluation)
- Govern:    failure taxonomy + STRIDE threat models
             (sage.evaluation.failure_taxonomy, sage.security)
- Enforce:   Adaptive Stability Controller and bounded interventions
             (sage.stability.controller, .interventions, .predictor)
"""

from sage.core.config import Config
from sage.core.base_agent import BaseAgent
from sage.core.llm_client import LLMClient

__version__ = "1.2.0"

__all__ = [
    "Config",
    "BaseAgent",
    "LLMClient",
    "__version__",
]
