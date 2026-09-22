"""Report every missing third-party package at once, not one traceback at a time.

:func:`missing_dependencies` is pure -- it probes import specs and returns names --
so the diagnosis is testable without uninstalling anything, and the ``require_*``
wrappers are the thin entry points that exit with :func:`dependency_help`.

:func:`require_frontend` is the one an extraction entry point wants: it checks the
interpreter, the packages and the model cache in the order a failure would actually
be hit.
"""

from __future__ import annotations

import sys

from .cache import require_model_cache
from .interpreter import require_torch

#: Third-party packages the frozen front-end needs beyond torch. Reported together, so a
#: user with three missing packages installs once rather than discovering them one
#: traceback at a time.
FRONTEND_DEPENDENCIES = ("numpy", "soundfile", "transformers", "huggingface_hub", "s3prl")


def missing_dependencies(names: tuple[str, ...]) -> list[str]:
    """Which of ``names`` cannot be imported. Pure apart from the imports themselves."""
    import importlib.util

    absent = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                absent.append(name)
        except (ImportError, ValueError):
            absent.append(name)
    return absent


def dependency_help(absent: list[str]) -> str:
    """An actionable message for packages the front-end needs and cannot import."""
    return (
        "missing dependencies: " + ", ".join(absent) + ".\n\n"
        "--- sourcetrace: incomplete environment --------------------------------\n"
        f"Running under: {sys.executable}\n\n"
        "Install the pinned set (the versions matter: transformers is held at\n"
        ">=4.38,<4.40 because EnCodec-24kHz's encode() return unpacking changed\n"
        "after that line):\n\n"
        "    pip install -r requirements.txt\n"
        "    # or, for the exact versions every number in results/ was measured with:\n"
        "    pip install -r requirements-lock.txt\n\n"
        "Then export PYTHONNOUSERSITE=1, so a newer huggingface_hub in your user\n"
        "site-packages cannot shadow the pinned one.\n"
        "-----------------------------------------------------------------------"
    )


def require_packages(*names: str) -> None:
    """Exit if any of ``names`` cannot be imported, naming all of them at once.

    For entry points that need *some* third-party package but not PyTorch --
    ``fetch_mlaad.py`` wants ``huggingface_hub`` and nothing else, and telling
    its user to install torch would be actively misleading.
    """
    absent = missing_dependencies(tuple(names))
    if absent:
        raise SystemExit(dependency_help(absent))


def require_frontend(extras: tuple[str, ...] = FRONTEND_DEPENDENCIES) -> None:
    """:func:`require_torch`, then report every other missing front-end dependency.

    Torch is checked first because a stub torch is a different problem with a different
    fix, and reporting "install numpy, soundfile, transformers" to somebody running the
    wrong interpreter would send them off in the wrong direction entirely.
    """
    require_torch()
    absent = missing_dependencies(extras)
    if absent:
        raise SystemExit(dependency_help(absent))
    require_model_cache()
