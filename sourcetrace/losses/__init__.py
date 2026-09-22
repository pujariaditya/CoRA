"""Training objectives used by the source-tracing head.

Only the three objectives actually invoked by ``Method.fit`` are ported here:
scaled-cosine cross-entropy (no margin — see below), supervised-contrastive, and HamOS
virtual-OOD.  See :mod:`sourcetrace.losses.objectives` for the faithful ports and their
citations to the champion ``method.py`` line ranges.

Two things to keep straight when editing:

* :func:`cosine_cross_entropy` has **no additive margin** (``m = 0``); it is plain
  temperature-scaled cosine softmax CE, not CosFace/ArcFace.
* :func:`hamos_loss` draws from the global torch RNG on every call, so its call order
  in ``fit`` is load-bearing and must not change.
"""
from __future__ import annotations

from .objectives import (
    cosine_ce_logits,
    cosine_cross_entropy,
    hamos_loss,
    supcon_loss,
)

__all__ = [
    "cosine_ce_logits",
    "cosine_cross_entropy",
    "supcon_loss",
    "hamos_loss",
]
