"""Per-class conformal-margin calibration and continuous score.

Faithful numpy port of the calibration + margin logic in the champion
``Method.fit`` / ``Method.score_openset`` (Original: method.py lines 549-633,
791-826), the small-open-set (v5, ``C < large_os_threshold = 80``) branch.

The nonconformity of an embedding ``E`` w.r.t. class ``c`` is::

    s_c(E) = -cos(E, w_c) = -(E @ Wn[c])                 (Original: 812)

Each class gets a calibrated offset ``q_c`` = margin-style ``mean + k*std`` of its
own-class train nonconformity, with empirical-Bayes shrinkage (strength ``N0``)
toward the global bound ``g_mean + k*g_std`` (Original: 567-603)::

    loc_c = mean(nc_c) + k * std(nc_c)
    w_c   = N0 / (n_c + N0)
    q_c   = (1 - w_c) * loc_c + w_c * g_bound

The continuous score is the calibrated margin ``max_c (q_c - s_c)`` (Original: 817).

Which embeddings feed which quantity (deliberate asymmetry)
-----------------------------------------------------------
:func:`stratified_holdout` carves a per-class ``cal_holdout = 0.15`` split-conformal
holdout that the head never trains on.  The two calibration products then use
*different* embedding sets, and this asymmetry is intentional champion parity:

* ``cal_q`` (the operative margin offsets) is fit on the **FULL** train embeddings
  ``z_all`` — holdout rows included — because the per-class ``mean + k*std`` needs
  every sample it can get to be stable (Original: 559, 580-587).  The champion also
  computed the fit-split nonconformity but did not use it for the v5 offsets.
* ``cal_conf_q`` (the split-conformal ``1 - alpha`` quantile) and ``cal_nc`` (the
  sorted own-class nonconformity backing the rank p-value) use **only** the held-out
  ``z_cal`` — these are the quantities whose validity depends on the holdout being
  unseen (Original: 558, 627-633).

Why the conformal quantile does not currently move the score
------------------------------------------------------------
:meth:`ConformalCalibration.offsets` blends the two offset families with
``conf_lambda``.  ``ScoringConfig.conf_lambda`` defaults to **0.0**, so
``cal_conf_q`` is blended *out* entirely and the operative offset is exactly the
shrunk ``mean + 3.5*std`` margin ``cal_q`` (v5 byte-identical).  ``cal_conf_q`` is
still fit, saved and loaded so a non-zero ``conf_lambda`` is a pure config flip.

Class-count-adaptive operating point
------------------------------------
``C >= large_os_threshold`` (80) selects the large-open-set (v9) constants
``k = 0.3`` / ``N0 = 150.0``; below that the small-open-set
:class:`sourcetrace.config.ScoringConfig` constants ``conf_std_k = 3.5`` /
``cal_shrink_n0 = 20.0`` apply.  **MLAAD v5 has C = 65**, so the shipped numbers
always come from the small-open-set branch; the large branch is retained only so a
larger-taxonomy run reproduces the champion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SCORING

__all__ = ["ConformalCalibration", "stratified_holdout", "conformal_pvalue"]

#: Large-open-set (v9) constants, used only when ``C >= SCORING.large_os_threshold``.
#: Not part of :class:`ScoringConfig` (whose values target v5), so the champion
#: values (``CONF_STD_K_LARGE`` / ``CAL_SHRINK_N0_LARGE``) are inlined here.
_LARGE_OS_STD_K = 0.3
_LARGE_OS_SHRINK_N0 = 150.0


def stratified_holdout(
    gen_labels: np.ndarray,
    num_classes: int,
    holdout: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class stratified split-conformal holdout.

    Faithful port of the calibration-mask construction in ``Method.fit`` (Original:
    method.py lines 468-481).  For each class with >= 8 samples, hold out
    ``max(2, round(n_c * holdout))`` indices (drawn via a per-class permutation of a
    ``np.random.default_rng(seed)`` stream) as calibration; the head trains on the
    complement.  Classes with < 8 samples contribute no calibration rows *and* draw
    no randomness, so the per-class draw order fixes the split — the loop runs over
    ``range(num_classes)`` in order and consumes one ``permutation`` per eligible
    class.

    When ``holdout <= 0`` both index sets are the full range (no holdout, and the
    RNG is never constructed).

    Returns ``(fit_idx, cal_idx)``.
    """
    n_all = len(gen_labels)
    if holdout <= 0:
        all_idx = np.arange(n_all)
        return all_idx, all_idx
    rng = np.random.default_rng(seed)
    cal_mask = np.zeros(n_all, dtype=bool)
    for cls in range(num_classes):
        cls_idx = np.where(gen_labels == cls)[0]
        if len(cls_idx) < 8:
            continue
        n_hold = max(2, int(round(len(cls_idx) * holdout)))
        cal_mask[rng.permutation(cls_idx)[:n_hold]] = True
    return np.where(~cal_mask)[0], np.where(cal_mask)[0]


def _shrunk_offset(own_nc: np.ndarray, k: float, n0: float, g_bound: float) -> float:
    """Empirical-Bayes shrunk per-class margin offset ``q_c`` (Original: 587-603).

    ``own_nc`` is one class's own-class nonconformity.  The location estimate is the
    margin ``mean + k*std`` (a lone sample degenerates to that sample), shrunk toward
    the global bound with weight ``N0 / (n_c + N0)``.  With no samples at all the
    class falls back to the global bound (or ``0.0`` when shrinkage is disabled).
    """
    n_c = len(own_nc)
    if n_c == 0:
        return g_bound if n0 > 0 else 0.0
    loc = float(own_nc.mean() + k * float(own_nc.std())) if n_c > 1 else float(own_nc[0])
    if n0 <= 0:
        return loc
    w = n0 / (n_c + n0)
    return (1.0 - w) * loc + w * g_bound


@dataclass
class ConformalCalibration:
    """Calibrated per-class conformal margin (small-open-set / v5 branch).

    Field names are load-bearing: ``Method.save`` / ``Method.load`` read and rebuild
    these attributes directly from the checkpoint dict.

    Attributes
    ----------
    cal_q:
        ``[C]`` float64 shrunk margin offsets ``q_c`` (Original: ``_cal_q``).  These
        are the offsets that actually drive the score at ``conf_lambda = 0``.
    cal_conf_q:
        ``[C]`` float64 split-conformal ``(1 - alpha)`` quantile offsets (Original:
        ``_cal_conf_q``), computed on the holdout.  Blended out by default.
    cal_nc:
        length-``C`` list of sorted held-out own-class nonconformities (Original:
        ``_cal_nc``), used by the rank p-value.  A class with no holdout rows gets
        the one-element placeholder ``[0.0]`` rather than an empty array.
    coverage:
        measured finite-sample coverage (percent) of the rank p-value per ``alpha``
        (Original: ``_coverage``), stored for reporting only — nothing reads it back.
    """

    cal_q: np.ndarray
    cal_conf_q: np.ndarray
    cal_nc: list[np.ndarray]
    coverage: dict[float, float]

    @staticmethod
    def fit(
        *,
        z_all: np.ndarray,
        z_cal: np.ndarray,
        gen_all: np.ndarray,
        gen_cal: np.ndarray,
        anchors_normed: np.ndarray,
        num_classes: int,
    ) -> ConformalCalibration:
        """Fit the per-class calibrated offsets.

        Faithful port of the calibration block in ``Method.fit`` (Original: method.py
        lines 556-633).  See the module docstring for why ``z_all`` (not the fit
        split) drives ``cal_q`` while only ``z_cal`` drives ``cal_conf_q`` / ``cal_nc``.

        Parameters
        ----------
        z_all:
            ``[N, base_emb_dim]`` L2-normed FULL-train score-branch embeddings — the
            per-class margin mean/std and the global bound are estimated here, on the
            full train rather than the fit split, for stability (Original: 559,
            580-587).
        z_cal:
            ``[N_cal, base_emb_dim]`` L2-normed held-out score-branch embeddings
            (the split-conformal calibration set — Original: 558).
        gen_all / gen_cal:
            integer generator labels aligned with ``z_all`` / ``z_cal``.
        anchors_normed:
            ``[C, base_emb_dim]`` L2-normed classifier anchors ``Wn`` (float64).
        num_classes:
            number of known classes ``C``; selects the small/large-open-set constants.
        """
        n_cls = num_classes
        wn = np.asarray(anchors_normed, dtype=np.float64)

        # nonconformity = -cos(E, own-class anchor)      (Original: 562-563)
        nc_all = -(np.asarray(z_all, dtype=np.float64) * wn[gen_all]).sum(axis=1)
        nc_cal = -(np.asarray(z_cal, dtype=np.float64) * wn[gen_cal]).sum(axis=1)

        # class-count-adaptive operating point           (Original: 573-578)
        large_os = n_cls >= SCORING.large_os_threshold
        k = _LARGE_OS_STD_K if large_os else SCORING.conf_std_k
        n0 = _LARGE_OS_SHRINK_N0 if large_os else SCORING.cal_shrink_n0
        g_mean = float(nc_all.mean()) if len(nc_all) else 0.0
        g_std = float(nc_all.std()) if len(nc_all) else 0.0
        g_bound = g_mean + k * g_std

        cal_q = np.zeros(n_cls, dtype=np.float64)
        cal_nc: list[np.ndarray] = []
        for cls in range(n_cls):
            cal_q[cls] = _shrunk_offset(nc_all[gen_all == cls], k, n0, g_bound)
            held_out = nc_cal[gen_cal == cls]
            cal_nc.append(np.sort(held_out) if len(held_out) else np.array([0.0]))

        # measured finite-sample coverage of the rank p-value   (Original: 608-620)
        # Each holdout sample is scored against its own class's calibration set with
        # itself excluded from the numerator count, so p = (1 + (ge - 1)) / n; the
        # sample is "covered" iff that p-value would not have been rejected at alpha.
        coverage: dict[float, float] = {}
        for alpha in (0.01, 0.05, 0.10):
            covered: list[float] = []
            for cls in range(n_cls):
                cal_c = cal_nc[cls]
                if len(cal_c) < 4:
                    continue
                for s in cal_c:
                    ge = len(cal_c) - np.searchsorted(cal_c, s, side="left")
                    p = (1.0 + ge - 1) / len(cal_c)    # leave-one-out (exclude self)
                    covered.append(1.0 if p > alpha else 0.0)
            coverage[alpha] = float(np.mean(covered) * 100) if covered else 0.0

        # split-conformal (1 - alpha) quantile offset            (Original: 627-633)
        # Classes with a degenerate holdout (the [0.0] placeholder or a single row)
        # fall back to the margin offset so the blend stays well-defined.
        cal_conf_q = np.zeros(n_cls, dtype=np.float64)
        for cls in range(n_cls):
            cal_c = cal_nc[cls]
            if len(cal_c) >= 2:
                cal_conf_q[cls] = float(np.quantile(cal_c, 1.0 - SCORING.conf_alpha))
            else:
                cal_conf_q[cls] = float(cal_q[cls])

        return ConformalCalibration(cal_q=cal_q, cal_conf_q=cal_conf_q,
                                    cal_nc=cal_nc, coverage=coverage)

    # -- scoring ------------------------------------------------------------- #
    def offsets(self) -> np.ndarray:
        """Blend the legacy margin offset with the conformal quantile offset.

        ``q_off = (1 - conf_lambda) * q_c + conf_lambda * q_c^conf``  (Original: 816).
        ``conf_lambda`` defaults to 0.0, so this returns ``cal_q`` unchanged (v5
        byte-identical) — the conformal quantile is blended out.

        Note the train-side z-norm statistics in :mod:`sourcetrace.scoring.openset`
        deliberately use raw ``cal_q`` instead of this blend, matching the champion.
        """
        lam = SCORING.conf_lambda
        return (1.0 - lam) * self.cal_q + lam * self.cal_conf_q

    def margin(self, e_normed: np.ndarray, anchors_normed: np.ndarray) -> np.ndarray:
        """Continuous calibrated conformal margin ``max_c (q_c - s_c)``, shape ``[N]``.

        Faithful port of the margin computation in ``score_openset`` (Original: 812,
        816-817).  ``e_normed`` ``[N, base_emb_dim]`` and ``anchors_normed``
        ``[C, base_emb_dim]`` are L2-normed float64 arrays.  Higher = more ID.
        """
        e = np.asarray(e_normed, dtype=np.float64)
        wn = np.asarray(anchors_normed, dtype=np.float64)
        s = -(e @ wn.T)                                     # [N, C] nonconformity
        q_off = self.offsets()
        return (q_off[None, :] - s).max(axis=1)


def conformal_pvalue(
    e_normed: np.ndarray,
    anchors_normed: np.ndarray,
    cal_nc: list[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """Split-conformal per-class rank p-value, returning ``p_max = max_c p_c(x)``.

    Faithful port of ``Method._conformal_pvalue`` (Original: method.py lines 828-852)::

        s_c(x) = -cos(x, w_c)
        p_c(x) = (1 + #{cal_j >= s_c(x)}) / (1 + n_c)
        p_max  = max_c p_c(x)

    This backs the distribution-free abstention rule ``Method.abstain`` (abstain iff
    ``p_max < alpha``, i.e. no class is plausible at level ``alpha``).  That rule is a
    **secondary, currently-unused** decision path: it is exported and checkpoint-round-
    trip tested, but nothing in the active scoring path (``Method.score_openset`` ->
    :mod:`sourcetrace.scoring.openset`) calls it, and the reported numbers come from
    the continuous fused score alone.

    Parameters
    ----------
    e_normed:
        ``[N, base_emb_dim]`` L2-normed float64 embeddings.
    anchors_normed:
        ``[C, base_emb_dim]`` L2-normed float64 anchors ``Wn``.
    cal_nc:
        ``cal_nc[c]`` is the **sorted** held-out own-class nonconformity for class
        ``c`` (``ConformalCalibration.cal_nc``); ``searchsorted`` relies on that order.
        The empty-class guard below is defensive only — ``fit`` substitutes a
        one-element ``[0.0]`` placeholder rather than an empty array.
    num_classes:
        number of known classes ``C``.

    Returns ``[N]`` float64 ``p_max``.
    """
    e = np.asarray(e_normed, dtype=np.float64)
    wn = np.asarray(anchors_normed, dtype=np.float64)
    s = -(e @ wn.T)                                          # [N, C]
    n = e.shape[0]
    p_max = np.zeros(n, dtype=np.float64)
    for c in range(num_classes):
        ca = cal_nc[c]
        n_c = len(ca)
        if n_c == 0:
            continue
        ge = n_c - np.searchsorted(ca, s[:, c], side="left")
        p_c = (1.0 + ge) / (1.0 + n_c)
        p_max = np.maximum(p_max, p_c)
    return p_max
