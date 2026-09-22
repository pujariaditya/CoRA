"""Protocol split builders for the two source-tracing benchmarks.

* :mod:`sourcetrace.datasets.mlaad` -- MLAAD v5 family-level open-set split.
* :mod:`sourcetrace.datasets.stopa` -- STOPA EET / TEE / Trials verification tables.

Both are pure-stdlib + numpy: they enumerate audio deterministically and return
plain tables (lists / numpy arrays). No torch, no subprocess, no GPU. The bundled
split-definition assets ship at the top level, under ``assets/`` (see
:data:`sourcetrace.config.ASSETS_DIR`), so the protocols are self-contained
and reproducible.
"""
from __future__ import annotations

# Re-exported so the split builders can say ``from . import ASSETS_DIR``; the
# definition lives with the other paths, in sourcetrace.config.
from ..config import ASSETS_DIR

__all__ = ["ASSETS_DIR"]
