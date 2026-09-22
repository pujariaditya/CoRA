"""Open-set scoring: calibrated conformal margin + relative-Mahalanobis fusion.

Faithful numpy ports of the champion scoring stack (Original: method.py + inlined
``score_primitives.shrinkage_cov``).  See the individual modules for line-range
citations.
"""
from __future__ import annotations

from .conformal import ConformalCalibration, conformal_pvalue, stratified_holdout
from .mahalanobis import RelativeMahalanobis, shrinkage_cov
from .openset import ZNormStats, fuse_openset_score

__all__ = [
    "ConformalCalibration",
    "conformal_pvalue",
    "stratified_holdout",
    "RelativeMahalanobis",
    "shrinkage_cov",
    "ZNormStats",
    "fuse_openset_score",
]
