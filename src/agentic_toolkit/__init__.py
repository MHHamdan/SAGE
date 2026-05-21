"""Backwards-compatibility shim for the legacy ``agentic_toolkit`` import path.

The package was renamed to :mod:`sage` as part of the SAGE rebrand
(Stabilize, Assess, Govern, Enforce). Importing ``agentic_toolkit`` continues
to work for downstream notebooks and scripts but emits a ``DeprecationWarning``.

Update your imports::

    from sage.evaluation import calculate_cnsr
    from sage.monitoring.stability_monitor import StabilityMonitor

The shim aliases ``agentic_toolkit`` and all of its submodules to the same
module objects under :mod:`sage` in ``sys.modules`` so that ``isinstance`` and
identity checks remain consistent across the two import paths. It will be
removed in a future release.
"""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "agentic_toolkit has been renamed to sage. "
    "Update your imports: `from sage import ...`. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


def _install_aliases() -> None:
    import pkgutil

    import sage

    sys.modules[__name__] = sage

    for module_info in pkgutil.walk_packages(sage.__path__, prefix="sage."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:  # pragma: no cover - best-effort alias
            continue
        alias = "agentic_toolkit." + module_info.name[len("sage.") :]
        sys.modules[alias] = module


_install_aliases()
