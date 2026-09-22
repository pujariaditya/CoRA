"""Fail readably when the interpreter cannot actually run this package.

The failure this module exists for is the *wrong interpreter*, which is the first
thing a new user hits and the least self-explanatory. Every entry point needs a
working PyTorch; run one under a system ``python3`` that has no torch, or a
placeholder ``torch`` package with the name but not the library, and the result is
an ``ImportError`` or an ``AttributeError`` from deep inside a feature extractor.
Neither says "you are using the wrong Python", which is the only fact the user needs.

It is not hypothetical on the machine this was developed on: the system ``python3``
has carried a stub ``torch`` that imports cleanly and then has no working tensor
operations -- so the failure surfaces as a nonsense result rather than as an
import error, which is the hardest kind to recognise.

The split here mirrors :mod:`sourcetrace.features._deps`: :func:`torch_problem` is
pure and takes the module object, so the diagnosis can be tested against a fake stub
without uninstalling anything, and :func:`require_torch` is the thin entry point that
probes the real import and exits.
"""

from __future__ import annotations

import sys

#: Attributes any usable torch exposes. A stub that shadows the real package on
#: ``sys.path`` typically satisfies the import and then fails on one of these.
_REQUIRED_ATTRS = ("__version__", "nn", "zeros", "float32")

ENV_NAME = "sourcetrace"


def torch_problem(torch_module: object | None) -> str | None:
    """Describe why ``torch_module`` is unusable, or return ``None`` if it is fine.

    ``None`` as the argument means the import itself failed. Anything else is probed
    for the attributes the feature extractor needs and then asked to do one trivial
    tensor operation -- a stub can declare ``zeros`` and still not compute.
    """
    if torch_module is None:
        return "PyTorch is not installed in this interpreter"

    missing = [a for a in _REQUIRED_ATTRS if not hasattr(torch_module, a)]
    if missing:
        return (
            "the imported 'torch' is missing "
            + ", ".join(missing)
            + " -- it looks like a stub or placeholder package, not PyTorch"
        )

    try:
        value = float(torch_module.zeros(2, dtype=torch_module.float32).sum())
    except Exception as exc:  # noqa: BLE001 - any failure here means torch is unusable
        return f"PyTorch imported but cannot compute: {type(exc).__name__}: {exc}"
    if value != 0.0:
        return f"PyTorch imported but torch.zeros(2).sum() returned {value!r}, expected 0.0"
    return None


def interpreter_help(problem: str, executable: str) -> str:
    """An actionable message naming the interpreter that failed and how to fix it."""
    return (
        f"{problem}.\n\n"
        "--- sourcetrace: wrong interpreter ------------------------------------\n"
        f"Running under: {executable}\n\n"
        f"This package needs the '{ENV_NAME}' environment (Python 3.10 + the pinned\n"
        "PyTorch / transformers in requirements-lock.txt). Create and use it with:\n\n"
        f"    conda create -n {ENV_NAME} python=3.10 && conda activate {ENV_NAME}\n"
        "    pip install -r requirements.txt\n\n"
        "or call that environment's interpreter directly, e.g.\n\n"
        f"    ~/anaconda3/envs/{ENV_NAME}/bin/python -m sourcetrace.evaluate --task mlaad_v5\n\n"
        "Export PYTHONNOUSERSITE=1 as well, so a newer huggingface_hub in your user\n"
        "site-packages cannot shadow the pinned one.\n"
        "-----------------------------------------------------------------------"
    )


def require_torch() -> None:
    """Exit with guidance if this interpreter has no usable PyTorch.

    Call at the top of ``main()`` in any entry point that will import torch, so the
    diagnosis arrives before a dataset scan or a model download has been attempted.
    Raises :class:`SystemExit`; does nothing at all when torch is healthy.
    """
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    problem = torch_problem(torch)
    if problem is not None:
        raise SystemExit(interpreter_help(problem, sys.executable))
