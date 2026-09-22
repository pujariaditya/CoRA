"""The exact protocol metrics, as the two benchmarks define them.

* :mod:`sourcetrace.metrics.protocol` -- FPR95 / EER formulas (ported verbatim).

Lightweight and eager: numpy and scikit-learn only, no torch, no model. The
evaluation protocols that call these live in :mod:`sourcetrace.tasks`.

.. warning::

   :func:`ood_eer` and :func:`stopa_eer` implement two DIFFERENT, deliberately
   non-interchangeable EER conventions -- asymmetric ``fpr[eer_index]`` for MLAAD
   and symmetric ``(fpr+fnr)/2`` for STOPA. See
   :mod:`sourcetrace.metrics.protocol`. Do not unify them.
"""
from __future__ import annotations

from .protocol import fpr95_panda, ood_eer, stopa_eer

__all__ = ["fpr95_panda", "ood_eer", "stopa_eer"]
