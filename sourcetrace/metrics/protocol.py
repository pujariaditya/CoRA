"""Exact FPR95 and EER formulas for the two source-tracing protocols.

Ported verbatim (math unchanged) from the original research grader:

* :func:`fpr95_panda` <- ``mlaad_klein.compute_fpr95_panda`` (L821-835): the
  FPR@95 in the convention of Neamtu et al. 2026 (the paper's 3.36 metric).
* :func:`ood_eer` <- ``mlaad_klein.compute_eer_klein`` (L371-387): the MLAAD v5
  OOD-EER (``fpr[argmin|fpr-(1-tpr)|]``, the piotrkawa/Klein repo convention).
* :func:`stopa_eer` <- ``stopa_verif.compute_eer_stopa`` (L202-223): the STOPA
  standard *symmetric* EER (``(fpr+fnr)/2`` at ``argmin|fpr-fnr|``).

.. warning::

   **The two EER definitions differ on purpose.** :func:`ood_eer` returns the
   asymmetric ``fpr[eer_index]``; :func:`stopa_eer` returns the symmetric
   ``(fpr[eer_index] + fnr[eer_index]) / 2``. Each mirrors ITS OWN published
   repo (MLAAD reports ``fpr[eer_index]``; STOPA reports the symmetric mean).
   Do NOT "unify" them -- a SOTA comparison must be on the byte-identical
   metric, and collapsing the two conventions silently changes every published
   number in one of the two tables. The only code they legitimately share is
   input coercion and the degenerate-input guard (:func:`_roc`); the index rule
   and the returned quantity are kept written out in full in each function so a
   reviewer can diff them against upstream line by line.

Numerical conventions that are load-bearing (do not "improve" these)
--------------------------------------------------------------------
* :func:`fpr95_panda` thresholds by *index-based order statistic* with ``int()``
  truncation on a sorted array -- NOT ``np.quantile``. Quantile interpolation
  would return a different threshold and hence a different FPR95.
* Both EERs pick the operating point with ``np.nanargmin``, which resolves ties
  to the FIRST minimising ROC vertex. Tie-breaking and ROC construction
  (:func:`sklearn.metrics.roc_curve`, no ``drop_intermediate`` override) must
  not be changed.
* Callers may deliberately hand in ``float32`` scores (STOPA does -- see
  :mod:`sourcetrace.tasks.stopa`) because ties at ``float32`` change which vertex
  ``nanargmin`` selects. The widening cast to ``float64`` performed here is
  exact and preserves those ties; it does not undo the caller's ``float32``.

Score/label conventions
-----------------------
* MLAAD OOD path: labels ``1 = OOD, 0 = ID``; the OOD score is *higher = more OOD*
  (the original negates the method's higher-is-more-ID ``score_openset`` output --
  see :mod:`sourcetrace.tasks.mlaad_v5`).
* STOPA path: labels ``1 = TARGET, 0 = NON-TARGET``; cosine score *higher = more
  target* (positively correlated with the label).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_curve

__all__ = ["fpr95_panda", "ood_eer", "stopa_eer"]

_NAN = float("nan")


def _roc(
    labels: ArrayLike,
    scores: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Coerce inputs, reject degenerate label vectors, and build the ROC curve.

    Shared by :func:`ood_eer` and :func:`stopa_eer`. This is the *only* logic the
    two EER conventions have in common -- it deliberately stops short of picking
    an operating point, which is where they diverge.

    A label vector that is empty or single-class carries no ROC: ``roc_curve``
    would emit all-``nan`` rates and ``nanargmin`` would then raise. Returning
    ``None`` lets both callers surface a clean ``nan`` instead.

    Args:
        labels: binary label vector (the positive class is ``1``).
        scores: decision scores, positively correlated with label ``1``. Cast to
            ``float64``; this widening is exact, so any ``float32`` ties the
            caller created on purpose are preserved.

    Returns:
        ``(fpr, tpr, thresholds)`` from :func:`sklearn.metrics.roc_curve`, or
        ``None`` if the label vector is empty or single-class.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or labels.min() == labels.max():
        return None
    return roc_curve(labels, scores)


def fpr95_panda(
    cal_ood_scores: ArrayLike,
    id_eval_scores: ArrayLike,
    target_recall: float = 0.95,
) -> float:
    """FPR@95 in the convention of Neamtu et al. 2026 -- the MLAAD v5 headline metric (SOTA 3.36).

    Faithful port of ``mlaad_klein.compute_fpr95_panda`` (L821-835).

    Which scores go where (this metric touches only two of the four pools)
    ---------------------------------------------------------------------
    The threshold is *calibrated* on the **DEV-OOD** scores and *tested* on the
    **EVAL-ID** scores. Dev-ID scores and eval-OOD scores never enter this
    computation at all. (Contrast :func:`ood_eer`, which consumes the full eval
    split -- ID and OOD rows together.) Calibrating on dev and testing on eval is
    what keeps the number leak-safe.

    Definition
    ----------
    With OOD-ness scores sorted ascending (higher score = more OOD), the
    threshold is the order statistic at index ``int(len(cal) * (1 - recall))``,
    i.e. the score below which ``(1 - recall)`` of the OOD-calibration samples
    fall, so ``target_recall`` (0.95) of them are still detected as OOD. FPR@95
    is then the fraction of in-distribution *eval* samples scoring **strictly
    above** that threshold -- ID samples wrongly flagged OOD.

    The index is computed by ``int()`` truncation on the sorted array and clamped
    to ``[0, len(cal) - 1]``. This is an order statistic, **not** an interpolated
    quantile: substituting ``np.quantile`` changes the threshold and therefore
    the reported metric. See the module warning.

    Args:
        cal_ood_scores: OOD-ness scores (higher = more OOD) of the dev/calibration
            OOD samples. These set the threshold.
        id_eval_scores: OOD-ness scores of the in-distribution eval samples.
            These are what the threshold is applied to.
        target_recall: OOD recall the threshold is calibrated to (default 0.95).

    Returns:
        FPR@95 as a fraction in ``[0, 1]`` (multiply by 100 for percent), or
        ``nan`` if either input is empty.
    """
    cal = np.sort(np.asarray(cal_ood_scores, dtype=np.float64))
    ide = np.asarray(id_eval_scores, dtype=np.float64)
    if len(cal) == 0 or len(ide) == 0:
        return _NAN
    idx = min(max(int(len(cal) * (1.0 - target_recall)), 0), len(cal) - 1)
    thr = cal[idx]
    return float(np.mean(ide > thr))


def ood_eer(labels: ArrayLike, scores: ArrayLike) -> tuple[float, float]:
    """MLAAD v5 OOD-EER -- **asymmetric** ``fpr[eer_index]`` (Klein convention).

    Faithful port of ``mlaad_klein.compute_eer_klein`` (L371-387), which mirrors
    the official piotrkawa/Klein repo ``scripts/ood_detector.py::compute_eer``
    byte-for-byte::

        fpr, tpr, thresholds = roc_curve(labels, scores)
        eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
        return fpr[eer_index], thresholds[eer_index]

    This returns ``fpr[eer_index]`` alone -- **not** the symmetric
    ``(fpr + fnr) / 2`` that :func:`stopa_eer` returns. That asymmetry is the
    upstream repo's exact definition and must not be changed; see the module
    warning.

    Unlike :func:`fpr95_panda` (dev-OOD calibration + eval-ID test only), this is
    computed over the **full eval split**: every eval row, ID and OOD alike,
    contributes a ``(label, score)`` pair.

    Returns ``(nan, nan)`` for an empty or single-class label vector, matching
    :func:`stopa_eer`. On any valid two-class input the ROC path below is reached
    unchanged, so no non-degenerate result is affected.

    Args:
        labels: ``1 = OOD, 0 = ID``.
        scores: OOD score (higher = more OOD).

    Returns:
        ``(eer_fraction, threshold)``; multiply the fraction by 100 for percent.
    """
    roc = _roc(labels, scores)
    if roc is None:
        return _NAN, _NAN
    fpr, tpr, thresholds = roc
    eer_index = int(np.nanargmin(np.abs(fpr - (1.0 - tpr))))
    # CONVENTION (MLAAD/Klein): return the FPR at the EER vertex, not the mean.
    return float(fpr[eer_index]), float(thresholds[eer_index])


def stopa_eer(labels: ArrayLike, scores: ArrayLike) -> tuple[float, float]:
    """STOPA verification EER -- **symmetric** ``(fpr + fnr) / 2`` (Chhibber).

    Faithful port of ``stopa_verif.compute_eer_stopa`` (L202-223), mirroring the
    repo ``compute_eer`` ROC step then returning the standard symmetric EER::

        fpr, tpr, thr = roc_curve(labels, scores)    # labels: TARGET=1, NON=0
        fnr = 1.0 - tpr
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
        return eer, thr[eer_index]

    This returns the *mean* of FPR and FNR at the EER vertex -- **not** the bare
    ``fpr[eer_index]`` that :func:`ood_eer` returns. See the module warning.

    Higher cosine score = more target (positively correlated with label 1).
    Callers pass ``float32`` scores on purpose (:mod:`sourcetrace.tasks.stopa`);
    do not upcast earlier in the pipeline.

    Returns ``(nan, nan)`` for an empty or single-class label vector.

    Args:
        labels: ``1 = TARGET, 0 = NON-TARGET``.
        scores: cosine similarity (higher = more target).

    Returns:
        ``(eer_fraction, threshold)``; multiply the fraction by 100 for percent.
    """
    roc = _roc(labels, scores)
    if roc is None:
        return _NAN, _NAN
    fpr, tpr, thr = roc
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    # CONVENTION (STOPA): return the symmetric mean at the EER vertex.
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    return float(eer), float(thr[eer_index])
