#!/usr/bin/env python3
"""Environment provenance report for SAGE experiment runs.

Emits the information needed to reproduce (or audit) a result: interpreter,
platform, git revision, seeding-relevant environment variables, and the
installed versions of the libraries that affect numerical output.

The same payload is embedded in each experiment's ``MANIFEST.json``; this
script exposes it standalone so an environment can be captured before or
after a run, or attached to a bug report.

Usage:
    python scripts/env_report.py                          # human-readable
    python scripts/env_report.py --format json            # machine-readable
    python scripts/env_report.py --format json -o env.json

Deliberately reports no credential values -- only whether a provider
variable is set, never its contents.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Libraries whose version can change numerical results or figure output.
TRACKED_PACKAGES: List[str] = [
    "sage-framework",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "sentence-transformers",
    "litellm",
    "openai",
    "pydantic",
    "httpx",
    "tiktoken",
    "langchain",
    "langgraph",
]

# Environment variables that affect determinism or backend selection. Values
# are reported for these because none of them is a secret.
TRACKED_ENV_VARS: List[str] = [
    "PYTHONHASHSEED",
    "SEED",
    "SAGE_BACKEND",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "OLLAMA_HOST",
    "LOG_LEVEL",
]

# Credential variables: presence is reported, values never are.
CREDENTIAL_ENV_VARS: List[str] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "LANGSMITH_API_KEY",
]


def _git(*args: str) -> Optional[str]:
    """Run a git command in the repo root, returning None if it is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_info() -> Dict[str, Any]:
    """Collect the git revision the working tree currently sits on."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "short_commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> Dict[str, Optional[str]]:
    """Report installed versions of the libraries that affect results."""
    versions: Dict[str, Optional[str]] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment() -> Dict[str, Any]:
    """Report determinism-relevant env vars, and credential presence only."""
    import os

    return {
        "variables": {k: os.environ.get(k) for k in TRACKED_ENV_VARS},
        "credentials_present": {
            k: bool(os.environ.get(k)) for k in CREDENTIAL_ENV_VARS
        },
    }


def build_report() -> Dict[str, Any]:
    """Assemble the full provenance report."""
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "git": git_info(),
        "packages": package_versions(),
        "environment": environment(),
    }


def render_text(report: Dict[str, Any]) -> str:
    """Render the report as an aligned, human-readable block."""
    lines: List[str] = ["SAGE — environment report", "=" * 52, ""]

    py, plat, git = report["python"], report["platform"], report["git"]
    lines += [
        "Interpreter",
        f"  {py['implementation']} {py['version']}",
        f"  {py['executable']}",
        "",
        "Platform",
        f"  {plat['system']} {plat['release']} ({plat['machine']})",
        "",
        "Git",
    ]
    if git["commit"]:
        dirty = " (uncommitted changes)" if git["dirty"] else ""
        lines.append(f"  {git['short_commit']} on {git['branch']}{dirty}")
    else:
        lines.append("  not a git checkout")

    lines += ["", "Packages"]
    for name, version in report["packages"].items():
        lines.append(f"  {name:<24} {version or '— not installed'}")

    lines += ["", "Environment"]
    for name, value in report["environment"]["variables"].items():
        lines.append(f"  {name:<24} {value if value else '— unset'}")

    lines += ["", "Credentials (presence only, values never reported)"]
    for name, present in report["environment"]["credentials_present"].items():
        lines.append(f"  {name:<24} {'set' if present else '— unset'}")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Parse arguments, build the report, and write it out."""
    parser = argparse.ArgumentParser(
        description="Report the environment SAGE experiments run in."
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write to this file instead of stdout.",
    )
    args = parser.parse_args()

    report = build_report()
    rendered = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_text(report)
    )

    if args.output:
        args.output.write_text(rendered)
        print(f"Environment report written to {args.output}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
