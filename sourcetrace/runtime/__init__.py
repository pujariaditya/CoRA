"""Environment guards: exit with an actionable message, never a traceback.

Every ``scripts/`` entry point and ``python -m sourcetrace.*`` module calls the
guard matching what it is about to need, at the top of ``main()`` and before any
download or disk scan.

* :mod:`sourcetrace.runtime.interpreter` -- is this Python usable at all
  (:func:`require_torch`)? The wrong-interpreter case, including a stub ``torch``
  that imports cleanly and then cannot compute.
* :mod:`sourcetrace.runtime.deps` -- are the front-end packages importable
  (:func:`require_packages`, :func:`require_frontend`)?
* :mod:`sourcetrace.runtime.cache` -- can the model and feature caches serve this
  run (:func:`require_model_cache`, :func:`require_feature_cache`)?

Each module pairs a *pure* predicate with a thin wrapper that raises, so the
diagnosis is testable against a fake without uninstalling anything.
"""

from __future__ import annotations

from .cache import (
    cache_help,
    model_cache_problem,
    require_feature_cache,
    require_model_cache,
)
from .deps import (
    FRONTEND_DEPENDENCIES,
    dependency_help,
    missing_dependencies,
    require_frontend,
    require_packages,
)
from .interpreter import (
    ENV_NAME,
    interpreter_help,
    require_torch,
    torch_problem,
)

__all__ = [
    "ENV_NAME", "FRONTEND_DEPENDENCIES",
    "torch_problem", "interpreter_help", "require_torch",
    "missing_dependencies", "dependency_help", "require_packages", "require_frontend",
    "model_cache_problem", "cache_help", "require_model_cache", "require_feature_cache",
]
