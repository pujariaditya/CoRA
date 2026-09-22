"""Trainable model components for the source-tracing method.

Reconstructed from the published method description and the ``method.py``
interface contract — this is NOT recovered champion code; see ``head.py`` for the
RECONSTRUCTION NOTICE and for the two properties that are easy to misread: the
construction order that fixes the RNG draws, and the auxiliary head that is trained
but never read at inference.
"""
from __future__ import annotations

from .head import FactorizedGatedHead, build_head_pair

__all__ = ["FactorizedGatedHead", "build_head_pair"]
