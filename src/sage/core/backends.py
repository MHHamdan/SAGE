"""Provider backend layer — explicit backend selection, credentials from .env.

Design goals (reproducibility packaging):
  * Credentials come ONLY from environment / .env (python-dotenv). No keys in code.
  * The real-API path is first-class. If credentials or litellm are missing, we
    raise a clear CredentialError — we NEVER silently fall back to the simulator.
  * The parametric simulator runs ONLY when explicitly requested
    (backend="simulator"), never as an invisible default.

Backends
--------
  litellm  / openai / anthropic / google : real hosted APIs via litellm.
  ollama                                  : local open-weights via localhost:11434.
  simulator                               : parametric draws (opt-in, offline, $0).

Every backend returns a uniform Completion carrying MEASURED token counts
(prompt/completion) so downstream cost accounting is real.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# Load .env once at import (best-effort). python-dotenv is a runtime dependency
# for the real-API path; its absence must not crash the simulator path.
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
    _DOTENV = True
except ImportError:  # pragma: no cover
    _DOTENV = False


class CredentialError(RuntimeError):
    """Raised when a real backend is requested but credentials/deps are missing."""


# provider -> accepted env var names (first non-empty wins)
PROVIDER_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
}


def provider_for_model(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt", "o1", "o3", "openai/")):
        return "openai"
    if m.startswith(("claude", "anthropic/")):
        return "anthropic"
    if m.startswith(("gemini", "google/", "vertex")):
        return "google"
    # litellm-style "provider/model" prefixes
    if "/" in m:
        head = m.split("/", 1)[0]
        if head in ("together_ai", "mistral", "cohere", "groq"):
            return head
    return "openai"


def get_credential(provider: str) -> str:
    for var in PROVIDER_ENV.get(provider, []):
        val = os.environ.get(var)
        if val:
            return val
    expected = PROVIDER_ENV.get(provider, ["<PROVIDER>_API_KEY"])
    hint = (
        ""
        if _DOTENV
        else " (python-dotenv is not installed; `pip install python-dotenv`)"
    )
    raise CredentialError(
        f"no provider credentials found for '{provider}'; set one of "
        f"{expected} in your environment or .env — see .env.example{hint}"
    )


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    backend: str


class Backend(ABC):
    name: str

    @abstractmethod
    def complete(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> Completion: ...


class LiteLLMBackend(Backend):
    """Real hosted APIs via litellm. Validates credentials up-front."""

    name = "litellm"

    def __init__(self) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise CredentialError(
                "litellm is not installed; `pip install litellm python-dotenv` and "
                "populate .env, or use backend='simulator'."
            ) from e
        self._litellm = litellm

    def complete(
        self, model, prompt, *, temperature=0.0, seed=0, max_tokens=1024, system=None
    ) -> Completion:
        provider = provider_for_model(model)
        get_credential(provider)  # raises CredentialError if missing
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self._litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0)),
            completion_tokens=int(getattr(usage, "completion_tokens", 0)),
            model=model,
            backend=self.name,
        )


class OllamaBackend(Backend):
    """Local open-weights via Ollama. Returns MEASURED token counts
    (prompt_eval_count / eval_count) from each model's own tokenizer."""

    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")

    def complete(
        self, model, prompt, *, temperature=0.0, seed=0, max_tokens=1024, system=None
    ) -> Completion:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:  # pragma: no cover
            raise CredentialError(
                f"cannot reach Ollama at {self.host}: {e}. Is `ollama serve` running?"
            ) from e
        return Completion(
            text=d.get("response", ""),
            prompt_tokens=int(d.get("prompt_eval_count", 0)),
            completion_tokens=int(d.get("eval_count", 0)),
            model=model,
            backend=self.name,
        )


class SimulatorBackend(Backend):
    """Parametric simulator — OPT-IN ONLY. Never an implicit default.

    This backend does not call any model; it exists so the committed synthetic
    studies remain reproducible via an explicit `--backend=simulator`.
    """

    name = "simulator"

    def complete(
        self, model, prompt, *, temperature=0.0, seed=0, max_tokens=1024, system=None
    ) -> Completion:
        raise NotImplementedError(
            "SimulatorBackend.complete() is a placeholder; the parametric CNSR/E4/E5 "
            "simulators generate their own draws and do not call complete(). Select "
            "backend='simulator' at the experiment level."
        )


def resolve_backend(name: str) -> Backend:
    """Return a Backend for an explicit name. Real backends raise CredentialError
    (never silently degrade). `simulator` must be requested explicitly."""
    key = (name or "").lower()
    if key == "simulator":
        return SimulatorBackend()
    if key == "ollama":
        return OllamaBackend()
    if key in ("litellm", "real", "openai", "anthropic", "google"):
        return LiteLLMBackend()
    raise ValueError(
        f"unknown backend '{name}'. Choose one of: litellm | ollama | simulator."
    )
