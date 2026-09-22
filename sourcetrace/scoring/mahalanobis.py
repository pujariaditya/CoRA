"""Relative-Mahalanobis (RMD) density model with within-class covariance inflation.

Faithful numpy port of ``Method._fit_density`` / ``Method._rmd_score`` (Original:
method.py lines 647-690, 749-764) and the inlined ``shrinkage_cov`` primitive
(Original: score_primitives.py lines 27-39).

The model fits, on the L2-normed TRAIN score-branch embeddings:

* a background (label-agnostic) Gaussian ``(bg_mu, bg_prec)`` over all train speech,
* per-class Gaussians ``(cls_mu[c], cls_prec[c])`` with Ledoit-Wolf-style shrinkage
  covariance, each covariance INFLATED along the top-``k`` principal directions of
  that class's OWN within-class scatter (``lang_infl_k = 3`` for v5) by
  ``lang_infl_alpha = 16``, which makes the metric tolerant of whatever dominates
  variation inside a generator.

.. note::
   The ``lang_`` prefix on those two constants is a misnomer inherited from the
   reference implementation, and is kept only because ``sourcetrace/config.py`` is
   frozen against the measured results. The intent there was to absorb *language*
   variation, but nothing here is language-aware: ``fit`` receives no language labels,
   and the directions come from an SVD of ``zc - cls_mu[cls]`` alone. Whether the
   leading within-class directions are in fact language directions is untested. Read
   every ``lang_infl_*`` name as "within-class inflation".

The score is the Relative-Mahalanobis distance (Ren 2021)::

    RMD(x) = -min_c [ MD_c(x) - MD_bg(x) ]                (Original: 749-764)

Higher = more ID (closer to some known class manifold relative to the background).
Everything runs in float64, matching the original.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SCORING

__all__ = ["shrinkage_cov", "RelativeMahalanobis"]


def shrinkage_cov(x: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage covariance toward the scaled identity.

    Faithful port of ``score_primitives.shrinkage_cov`` (Original: score_primitives.py
    lines 27-39).  With ``gamma=None`` the shrinkage is data-driven,
    ``gamma = clip(d / (d + max(n, 1)), 0.05, 0.9)`` (shrink more when few samples
    relative to dimension).
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    xc = x - x.mean(0, keepdims=True)
    s = (xc.T @ xc) / max(n - 1, 1)
    mu = np.trace(s) / d
    target = mu * np.eye(d)
    if gamma is None:
        gamma = float(np.clip(d / (d + max(n, 1)),
                              SCORING.shrink_gamma_min, SCORING.shrink_gamma_max))
    return (1 - gamma) * s + gamma * target


@dataclass
class RelativeMahalanobis:
    """Per-class + background Mahalanobis density, scored as Relative-Mahalanobis.

    Attributes mirror the champion state (Original: ``_bg_mu``, ``_bg_prec``,
    ``_cls_mu``, ``_cls_prec``).
    """

    bg_mu: np.ndarray
    bg_prec: np.ndarray
    cls_mu: np.ndarray
    cls_prec: list[np.ndarray]
    num_classes: int

    @staticmethod
    def fit(
        z_all: np.ndarray,
        gen_all: np.ndarray,
        num_classes: int,
        *,
        cov_gamma: float | None = None,
    ) -> RelativeMahalanobis:
        """Fit per-class + background Gaussians on L2-normed train embeddings.

        Faithful port of the density block in ``Method._fit_density`` (Original:
        method.py lines 656-683).

        Parameters
        ----------
        z_all:
            ``[N, base_emb_dim]`` L2-normed FULL-train score-branch embeddings.
        gen_all:
            integer generator labels for ``z_all``.
        num_classes:
            number of known classes ``C`` (selects the within-class inflation ``k``).
        cov_gamma:
            shrinkage gamma passed to :func:`shrinkage_cov` (Original ``COV_GAMMA``
            = None -> data-driven).
        """
        z = np.asarray(z_all, dtype=np.float64)
        c = num_classes

        # class-count-adaptive within-class inflation K (Original: 656).  v5 uses
        # lang_infl_k=3; the large-os (v9) value is inlined for completeness.
        infl_k = _large_infl_k() if c >= SCORING.large_os_threshold else SCORING.lang_infl_k

        # background Gaussian over ALL train speech           (Original: 660-661)
        bg_mu = z.mean(0)
        bg_prec = np.linalg.pinv(shrinkage_cov(z, gamma=cov_gamma))

        cls_mu = np.zeros((c, z.shape[1]), dtype=np.float64)
        cls_prec: list[np.ndarray] = []
        for cls in range(c):
            zc = z[gen_all == cls]
            if len(zc) >= 2:
                cls_mu[cls] = zc.mean(0)
                sig = shrinkage_cov(zc, gamma=cov_gamma)
                # within-class covariance inflation            (Original: 673-679)
                if (
                    SCORING.lang_infl_k
                    and SCORING.lang_infl_k > 0
                    and len(zc) > infl_k + 1
                ):
                    try:
                        _, _, vt = np.linalg.svd(zc - cls_mu[cls], full_matrices=False)
                        vk = vt[:infl_k]              # (k, D) top within-class directions
                        sig = sig + SCORING.lang_infl_alpha * (vk.T @ vk)
                    except Exception:
                        # Intentional silent fallback: on SVD failure this class keeps its
                        # UNINFLATED shrinkage covariance rather than aborting the whole fit.
                        # The class Gaussian stays well-defined (`sig` is untouched — the
                        # inflated value is only bound on the successful path), so scoring
                        # degrades gracefully for that one class instead of raising.
                        #
                        # The catch is deliberately NOT narrowed to ``np.linalg.LinAlgError``.
                        # That is the documented/expected failure ("SVD did not converge",
                        # incl. non-finite input reaching LAPACK), but it is not provably the
                        # only one: the LAPACK workspace allocation can raise ``MemoryError``
                        # at this dimensionality, and numpy's own dispatch can raise
                        # ``ValueError``/``TypeError`` on a degenerate array. Narrowing would
                        # turn those from "class left uninflated" into a hard failure — a
                        # behavior change on a path that produced the published numbers.
                        # No logging here: `score`/`fit` run inside the eval loop and must
                        # not write to stdout.
                        pass
                cls_prec.append(np.linalg.pinv(sig))
            else:
                cls_mu[cls] = bg_mu
                cls_prec.append(bg_prec)

        return RelativeMahalanobis(
            bg_mu=bg_mu, bg_prec=bg_prec, cls_mu=cls_mu, cls_prec=cls_prec, num_classes=c
        )

    def score(self, e: np.ndarray) -> np.ndarray:
        """Relative-Mahalanobis score ``-min_c[MD_c(x) - MD_bg(x)]`` (higher = more ID).

        Faithful port of ``Method._rmd_score`` (Original: method.py lines 749-764).
        """
        e = np.asarray(e, dtype=np.float64)
        dbg = e - self.bg_mu
        md_bg = np.einsum("ij,jk,ik->i", dbg, self.bg_prec, dbg)
        rel: np.ndarray | None = None
        for c in range(self.num_classes):
            dc = e - self.cls_mu[c]
            mdc = np.einsum("ij,jk,ik->i", dc, self.cls_prec[c], dc) - md_bg
            rel = mdc if rel is None else np.minimum(rel, mdc)
        return -np.asarray(rel, dtype=np.float64)


def _large_infl_k() -> int:
    """Large-open-set (v9) within-class inflation K (Original const LANG_INFL_K_LARGE=8).

    Retained for completeness; v5 (the target) uses ``SCORING.lang_infl_k = 3``.
    """
    return 8
