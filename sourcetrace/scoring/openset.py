"""Z-normalized open-set fusion of the conformal margin and the RMD density score.

Faithful port of the fusion tail of ``Method.score_openset`` and the z-norm stat
caching in ``Method._fit_density`` (Original: method.py lines 684-690, 818-826).

The two continuous scores live on different scales, so each is z-normalized with
TRAIN-ID constants (inductive — no eval-batch statistics) before blending::

    z_cos = (m_cos - cos_mean) / cos_std
    z_rmd = (m_rmd - rmd_mean) / rmd_std
    score = cos_weight * z_cos + dens_weight * z_rmd     (Original: 823-825)

The champion (v5) values are ``cos_weight = dens_weight = 1.0``, so the blend
evaluates to the plain ``z_cos + z_rmd`` of the original; ``cos_weight`` exists only
as an ablation knob and does not move the published numbers.

When ``dens_weight <= 0`` (or no density model / no z-norm stats were fit) the RAW,
un-z-normed conformal margin is returned unchanged (Original: 821, 826).  That early
return is on a different scale from the fused branch — see the "Known asymmetry"
note on :func:`fuse_openset_score`.

Ablation knobs
--------------
:mod:`sourcetrace.config` reads five ``ST_ABL_*`` environment variables; unset, every
one of them holds its frozen champion value and this module is byte-identical to the
published run.  Two of them change the arithmetic in :func:`fuse_openset_score`
directly:

``ST_ABL_COS_WEIGHT`` -> ``SCORING.cos_weight`` (default 1.0)
    Weight on the z-normed conformal margin.  ``0`` gives the relative-Mahalanobis-only
    arm, ``dens_weight * z_rmd``, still z-normed (the ``dens_weight > 0`` branch is
    taken, since only ``dens_weight`` gates the early return).
``ST_ABL_DENS_WEIGHT`` -> ``SCORING.dens_weight`` (default 1.0)
    Weight on the z-normed RMD density term.  ``0`` (or negative) takes the early
    return and yields the conformal-margin-only arm as the RAW ``m_cos`` — the RMD
    model is not even evaluated, and ``cos_weight`` is NOT applied.

The remaining three act upstream, on feature construction and the embedding head, and
reach this function only through the ``e_normed`` / ``anchors_normed`` it is handed —
no branch here inspects them:

``ST_ABL_CODEC_WEIGHT`` -> ``MODEL.codec_weight`` (default 0.5; ``0`` drops the
codec-residual subspace), ``ST_ABL_VOC_WEIGHT`` -> ``MODEL.voc_weight`` (default 0.1;
``0`` drops the signature/vocoder subspace), and ``ST_ABL_NO_GATING=1`` -> zeroes all
three FiLM gate betas (naive concatenation instead of learned gating).  Each shifts
the embedding geometry, hence ``m_cos``, ``m_rmd``, and the TRAIN z-norm constants
alike; the fusion formula itself is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SCORING
from .conformal import ConformalCalibration
from .mahalanobis import RelativeMahalanobis

__all__ = ["ZNormStats", "fuse_openset_score"]


@dataclass
class ZNormStats:
    """TRAIN z-normalization constants for the margin and RMD terms.

    Faithful port of ``_cos_mean``/``_cos_std``/``_rmd_mean``/``_rmd_std`` (Original:
    method.py lines 277-278, 689-690).  Standard deviations carry the ``+ 1e-9``
    floor from the original.
    """

    cos_mean: float
    cos_std: float
    rmd_mean: float
    rmd_std: float

    @staticmethod
    def fit(
        z_all: np.ndarray,
        anchors_normed: np.ndarray,
        calibration: ConformalCalibration,
        density: RelativeMahalanobis,
    ) -> ZNormStats:
        """Compute the z-norm stats on the TRAIN score-branch embeddings.

        Faithful port of the stat block in ``_fit_density`` (Original: method.py lines
        684-690).

        Caveat — the margin used here is built from the RAW ``calibration.cal_q``
        offsets, deliberately NOT from the ``conf_lambda``-blended
        :meth:`ConformalCalibration.offsets`, exactly as the original ``m_cos``
        computation.  The scored quantity in :func:`fuse_openset_score` DOES go through
        ``offsets()``.  At the champion ``conf_lambda = 0`` the two coincide, so this is
        a no-op; if anyone sets ``conf_lambda > 0``, ``cos_mean``/``cos_std`` would be
        estimated on a different statistic than the one they normalize, biasing ``z_cos``
        and hence the fused score.  Fixing that would change the v5 numbers, so it is
        left as-is and flagged here.

        Parameters
        ----------
        z_all:
            ``[N, base_emb_dim]`` L2-normed FULL-train score-branch embeddings.
        anchors_normed:
            ``[C, base_emb_dim]`` L2-normed anchors ``Wn`` (float64).
        calibration:
            the fitted :class:`ConformalCalibration` (provides ``cal_q``).
        density:
            the fitted :class:`RelativeMahalanobis` model.
        """
        z = np.asarray(z_all, dtype=np.float64)
        wn = np.asarray(anchors_normed, dtype=np.float64)
        s_all = -(z @ wn.T)                                          # [N, C] nonconformity
        m_cos = (calibration.cal_q[None, :] - s_all).max(axis=1)     # margin on train
        m_rmd = density.score(z)                                     # RMD on train
        return ZNormStats(
            cos_mean=float(m_cos.mean()),
            cos_std=float(m_cos.std() + 1e-9),
            rmd_mean=float(m_rmd.mean()),
            rmd_std=float(m_rmd.std() + 1e-9),
        )


def fuse_openset_score(
    e_normed: np.ndarray,
    anchors_normed: np.ndarray,
    calibration: ConformalCalibration,
    density: RelativeMahalanobis | None,
    znorm: ZNormStats | None,
) -> np.ndarray:
    """Continuous open-set score (higher = more ID).

    Faithful port of the fusion tail of ``score_openset`` (Original: method.py lines
    817-826), with ``cos_weight`` added as an ablation knob (champion value 1.0, so
    the champion path is unchanged)::

        m_cos = calibration.margin(E, Wn)          # conf_lambda-blended offsets
        if dens_weight > 0 and density is not None and znorm is not None:
            z_cos = (m_cos - cos_mean) / cos_std
            z_rmd = (rmd_score(E) - rmd_mean) / rmd_std
            return cos_weight * z_cos + dens_weight * z_rmd
        return m_cos                               # RAW margin, not z-normed

    ``SCORING.cos_weight`` / ``SCORING.dens_weight`` come from the ``ST_ABL_COS_WEIGHT``
    / ``ST_ABL_DENS_WEIGHT`` environment overrides; see the module docstring for all
    five ``ST_ABL_*`` knobs and which of them reach this function.

    Known asymmetry (deliberate — do not "fix" without re-running the ablations)
    --------------------------------------------------------------------------
    The two single-term ablation arms are NOT on the same scale:

    * ``dens_weight <= 0`` (margin-only arm) takes the early return and yields the RAW
      ``m_cos``, in cosine-margin units, with neither the z-norm nor ``cos_weight``
      applied.
    * ``cos_weight = 0`` (RMD-only arm) still enters the fused branch and yields the
      z-normed ``dens_weight * z_rmd``, in TRAIN-z units.

    Only ``dens_weight`` gates the branch, which is what produces the asymmetry.  It is
    harmless for every metric this repo reports — FPR95, EER and AUROC are rank-based
    and therefore invariant to the strictly monotone rescaling
    ``m_cos -> (m_cos - cos_mean) / cos_std`` (``cos_std > 0`` by the ``+ 1e-9`` floor).
    It IS wrong for anything comparing absolute score values or a fixed threshold
    across arms, so do not read one arm's threshold against the other's.

    The behavior is kept because ``results/ablation/*.json`` and paper Table 2 were
    produced with exactly this arithmetic; z-norming the early return would change
    those numbers (though not their ranking).

    Parameters
    ----------
    e_normed:
        ``[N, base_emb_dim]`` L2-normed score-branch embeddings (float64).
    anchors_normed:
        ``[C, base_emb_dim]`` L2-normed anchors ``Wn`` (float64).
    """
    m_cos = calibration.margin(e_normed, anchors_normed)
    if SCORING.dens_weight > 0 and density is not None and znorm is not None:
        m_rmd = density.score(e_normed)
        z_cos = (m_cos - znorm.cos_mean) / znorm.cos_std
        z_rmd = (m_rmd - znorm.rmd_mean) / znorm.rmd_std
        return SCORING.cos_weight * z_cos + SCORING.dens_weight * z_rmd
    return m_cos
