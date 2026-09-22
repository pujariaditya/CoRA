"""Public source-tracing method: factorized gated head + anchor margin + RMD, on one embedding.

One embedding for every task
----------------------------
``embed()`` returns the fused 288-d embedding ``z = L2([z_ssl(192) | 0.1 z_spec(64) |
0.5 z_codec(32)])`` of the paper (Sec. 2.2). Everything downstream reads that one
vector: the open-set score (anchor margin + relative Mahalanobis, Sec. 2.4), the
known-model label (nearest trained anchor, ``attribute()``), and the STOPA cross-corpus
protocol (cosine to enrolment fingerprints, :mod:`sourcetrace.tasks.stopa`), which
applies the MLAAD-trained model without any STOPA training. An earlier design carried a
second, PCA-whitened SSL channel next to ``z`` for STOPA and labelled known models on
that wider vector. It is gone, and so is its checkpoint format: :meth:`Method.load`
reads only files written by this code (``format_version`` 2, ``METHOD_TAG`` above).

What is and is not a port
-------------------------
This file is a line-for-line port of the champion ``Method`` class (commit
``fa876f7d``): its numeric operations, constants and order of operations are preserved
from the original ``method.py``, with only structure, naming, typing and docs
improved. :mod:`sourcetrace.losses` and :mod:`sourcetrace.scoring` are likewise ports,
with line-range citations.

**The head is not.** ``sourcetrace/models/`` was absent from the public checkout, so
:mod:`sourcetrace.models.head` was reconstructed from the published method
description and this file's interface contract. The numbers this code produces are
the ones in ``results/`` (MLAAD v5 FPR95 = 0.50%).

The auxiliary head
------------------
``head_xfer`` is the auxiliary head of the paper (Sec. 2.3: "an auxiliary head with
scale 14 is used only during training"). It is a second :class:`FactorizedGatedHead`
without the codec subspace, trained by its own cosine cross-entropy / SupCon / anchor
terms, sharing no parameter with the score head. Nothing at inference reads it. It is
kept because the fit is defined with it: dropping it would advance the global RNG
differently and change the virtual-outlier draws, and with them every reported number.

Public API::

    Method(num_known_classes, num_languages)
    fit(features[N, FEAT_DIM], gen_labels[N], lang_labels=None, seed=0)
    embed(X[*, FEAT_DIM]) -> np.ndarray[*, EMB_DIM=288]        # L2-normed z
    score_openset(X) -> np.ndarray[N]                          # higher = more known
    attribute(X) -> np.ndarray[N]                              # nearest-anchor class
    save(path) -> str                                          # torch.save all state
    Method.load(path) -> Method                               # exact round-trip

Numeric-faithfulness contract preserved from the original
--------------------------------------------------------
* 0.15 per-class stratified split-conformal calibration split (seed-driven).
* Codec-residual TRAIN normalization (mean/std over train, applied at embed).
* Adam with ``lr = 3e-4`` (``TRAIN.lr``) and **no weight decay**: the original optimizer
  call was ``torch.optim.Adam(params, lr=LR)``, relying on the library default.
  ``TRAIN.weight_decay`` now states that 0.0 explicitly and :meth:`fit` passes it, which
  is bit-identical and keeps the value the paper reports in one place.
* 300 epochs, batch 512, shuffled per epoch via ``np.random.default_rng(seed)``.
* Deterministic seeding: ``torch.manual_seed`` + ``cuda.manual_seed_all`` +
  ``np.random.seed`` + ``use_deterministic_algorithms(True, warn_only=True)``.
* Anchor init: ``randn(C, D) * 0.01`` with a CPU generator seeded 0 (score) / 1
  (auxiliary), then moved to device.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import FEATURES, MODEL, SCORING, TRAIN

if TYPE_CHECKING:  # annotation-only imports: never executed at runtime, so a bare
    # ``import sourcetrace.method`` still pulls in no torch until fit()/load().
    import torch

    from .models import FactorizedGatedHead
    from .scoring import ConformalCalibration, RelativeMahalanobis, ZNormStats

FEAT_DIM: int = FEATURES.feat_dim        # 2133  (raw frontend width)
BASE_EMB_DIM: int = MODEL.base_emb_dim   # 288   (the fused embedding z)
EMB_DIM: int = MODEL.emb_dim             # 288   (== BASE_EMB_DIM: one embedding)
XFER_DIM: int = MODEL.fact_am_dim + MODEL.fact_voc_dim  # 256 (auxiliary head width, training only)
METHOD_TAG: str = "codecsig_dualgate_hamos_bifocal_rmd_z288_v10"
CHECKPOINT_FORMAT_VERSION: int = 2

assert EMB_DIM == BASE_EMB_DIM, "one embedding: MODEL.emb_dim must equal MODEL.base_emb_dim"

_CODEC_OFF: int = FEATURES.ssl_dim + FEATURES.sig_dim   # 2100


def _l2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a ``1e-12`` denominator floor.

    Faithful port of the module-level ``_l2`` (Original: method.py lines 250-251).
    """
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class Method:
    """Open-set source-tracing method: one fused embedding, anchors, margin + RMD score.

    Parameters
    ----------
    num_known_classes:
        number of known generator classes ``C`` (inferred from labels if ``None``).
    num_languages:
        retained for API parity with the champion (unused by the ported core).
    """

    def __init__(self, num_known_classes: int | None = None,
                 num_languages: int | None = None) -> None:
        self.num_known_classes = num_known_classes
        self.num_languages = num_languages

        # trainable heads (built in fit)  (Original: method.py lines 258-265)
        self.head: FactorizedGatedHead | None = None       # the embedding head (codec=True)
        # The auxiliary head of the paper (Sec. 2.3): trained by its own loss terms and
        # checkpointed, never read at inference. Shares no parameters with self.head.
        self.head_xfer: FactorizedGatedHead | None = None  # auxiliary head (codec=False)

        self._device: torch.device | None = None
        self._C: int | None = None
        self._Wn: np.ndarray | None = None          # [C, 288] L2-normed anchors (float64)

        # scoring state (Original: method.py lines 269-290)
        self._calibration: ConformalCalibration | None = None
        self._density: RelativeMahalanobis | None = None
        self._znorm: ZNormStats | None = None

    @property
    def anchors(self) -> np.ndarray:
        """``[C, 288]`` L2-normed class anchors ``W`` (float64), fitted by :meth:`fit`."""
        if self._Wn is None:
            raise RuntimeError("anchors requested before fit()/load()")
        return self._Wn

    # -- deterministic seeding ------------------------------------------------ #
    @staticmethod
    def _seed_all(seed: int) -> None:
        """Deterministic seeding (Original: method.py lines 355-367).

        What this buys you: training is deterministic *per seed* given the
        CuBLAS/cudnn settings below, so re-fitting at the same fit seed on the same
        machine reproduces the same numbers. The default fit seed 0 on split seed 42
        is what produced every value in ``results/`` — MLAAD v5 FPR95 = 0.50%
        (``results/ablation/full.json``).


        The load-bearing env lever, ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, must be set
        BEFORE the CUDA context is created (i.e. before ``import torch`` / first CUDA
        use), so it cannot be forced from here; we set it defensively for the case
        where no CUDA context exists yet, and always set the runtime-settable cudnn
        flags. For reproducible fits, launch the process with that variable already
        exported.
        """
        import os

        import torch

        # Defensive: only takes effect if the CUDA context is not yet created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # runtime-settable determinism flags (safe to set after torch init).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    # -- head construction ---------------------------------------------------- #
    def build_head(self) -> FactorizedGatedHead:
        """Build the embedding head and the auxiliary head on the resolved device.

        Faithful port of ``Method.build_head`` (Original: method.py lines 283-330),
        delegating the architecture to :class:`FactorizedGatedHead`.  Returns the
        embedding head (contract parity).

        The device is resolved through :func:`sourcetrace.config.device`, the same
        single policy :meth:`load` uses, so ``ST_CPU=1`` forces CPU on both paths.
        (This previously read ``torch.cuda.is_available()`` directly and silently
        ignored ``ST_CPU``, so on a CUDA box a Method could end up holding a
        GPU-built head alongside a CPU-resolved device after a reload.)
        """
        import torch

        from .config import device as resolve_device
        from .models import build_head_pair

        # Resolve through config.device() -- the single device policy, shared with
        # load() -- so ST_CPU=1 forces CPU here too.  With ST_CPU unset it returns
        # exactly "cuda" if torch.cuda.is_available() else "cpu", i.e. the default
        # GPU path is bit-for-bit unchanged.
        self._device = torch.device(resolve_device())
        # Build both branches in the champion's submodule order (Original: method.py
        # lines 306-329). That fixes the ORDER of RNG draws, which is what makes this
        # codebase reproducible seed-to-seed. It does NOT make the weights identical to
        # the champion's: models/head.py was reconstructed, not ported, so its per-module
        # init differs and cannot be recovered. See this module's docstring.
        self.head, self.head_xfer = build_head_pair(self._device)
        return self.head

    # -- fit ------------------------------------------------------------------ #
    def fit(
        self,
        features: np.ndarray,
        gen_labels: np.ndarray,
        lang_labels: np.ndarray | None = None,
        seed: int = 0,
    ) -> Method:
        """Train the heads and fit the scoring stack.

        Faithful port of ``Method.fit`` (Original: method.py lines 443-645).  The
        order-of-operations — seeding, codec-norm, stratified split, anchor init,
        optimizer, 300-epoch loop, then calibration / density / z-norm fits — is
        preserved exactly. (The original ended by fitting a PCA whitener for a second
        STOPA channel; that channel no longer exists, and its fit drew no random
        numbers, so removing it leaves every other quantity untouched.)
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from .losses import cosine_cross_entropy, hamos_loss, supcon_loss
        from .scoring import (
            ConformalCalibration,
            RelativeMahalanobis,
            ZNormStats,
            stratified_holdout,
        )

        self._seed_all(seed)
        self.build_head()

        gen_np = np.asarray(gen_labels, dtype=np.int64)
        c = int(self.num_known_classes or (int(gen_np.max()) + 1))
        self._C = c
        x_all_np = np.asarray(features, dtype=np.float32)

        # codec-residual TRAIN normalization (Original: method.py lines 458-464).
        if FEATURES.codec_dim > 0:
            cr_train = x_all_np[:, _CODEC_OFF:_CODEC_OFF + FEATURES.codec_dim].astype(np.float64)
            codec_mu = torch.tensor(cr_train.mean(0), device=self._device, dtype=torch.float32)
            codec_sd = torch.tensor(cr_train.std(0) + 1e-6,
                                    device=self._device, dtype=torch.float32)
        else:
            codec_mu = None
            codec_sd = None
        self.head.set_codec_norm(codec_mu, codec_sd)
        # the auxiliary head is codec-OFF; installing None keeps parity (it never reads codec).
        self.head_xfer.set_codec_norm(None, None)

        # split-conformal stratified holdout (Original: method.py lines 468-481).
        fit_idx, cal_idx = stratified_holdout(gen_np, c, SCORING.cal_holdout, seed)
        x_fit = torch.tensor(x_all_np[fit_idx], device=self._device)
        n = x_fit.shape[0]
        gen_fit = gen_np[fit_idx]
        ytr = torch.tensor(gen_fit, device=self._device)

        # anchor init: randn * TRAIN.anchor_init_std (0.01) with CPU generators seeded
        # 0 (embedding head) / 1 (auxiliary head),
        # then moved to device (Original: method.py lines 489-492).
        gtor = torch.Generator(device="cpu")
        gtor.manual_seed(0)
        w = nn.Parameter(
            (torch.randn(c, BASE_EMB_DIM, generator=gtor) * TRAIN.anchor_init_std).to(self._device)
        )
        gtor_x = torch.Generator(device="cpu")
        gtor_x.manual_seed(1)
        wx = nn.Parameter(
            (torch.randn(c, XFER_DIM, generator=gtor_x) * TRAIN.anchor_init_std).to(self._device)
        )

        # Parameter set, built in the champion's EXACT order (Original: method.py
        # lines 493-506).  Two faithfulness points that materially change the
        # numerics and are preserved here:
        #   1. ORDER: head(AM), W, head_voc, gate_v, head_xfer(AM), Wx,
        #      head_voc_xfer, gate_v_xfer, gate_tmp, gate_tmp_xfer.
        #   2. The residual path: the original champion never added the embedding head's
        #      ``proj_codec`` / ``gate_codec`` to the optimizer (a fixed random projection).
        #      Since 2026-09-18 they are appended AFTER the champion list when
        #      ``MODEL.train_codec_path`` is True (the default): appending keeps the
        #      champion's parameter order intact, and training them is worth 0.28 FPR95
        #      points and 0.9 OOD-EER points on MLAAD v5 (results/ablation/fixed_proj.json
        #      is the old behaviour, ``train_codec_path: false``).
        s = self.head        # the embedding head
        xh = self.head_xfer  # the auxiliary head (training only)
        params = list(s.proj_am.parameters()) + [w]
        if s.proj_voc is not None:
            params += list(s.proj_voc.parameters())
        if s.gate_voc is not None:
            params += list(s.gate_voc.parameters())
        params += list(xh.proj_am.parameters()) + [wx]
        if xh.proj_voc is not None:
            params += list(xh.proj_voc.parameters())
        if xh.gate_voc is not None:
            params += list(xh.gate_voc.parameters())
        if s.gate_tmp is not None:
            params += list(s.gate_tmp.parameters())
        if xh.gate_tmp is not None:
            params += list(xh.gate_tmp.parameters())
        if MODEL.train_codec_path and s.proj_codec is not None:
            params += list(s.proj_codec.parameters()) + list(s.gate_codec.parameters())

        # Adam, lr=3e-4, NO weight decay (Original: method.py line 507 —
        # ``torch.optim.Adam(params, lr=LR)``, i.e. the library default). TRAIN.weight_decay
        # is 0.0, so passing it explicitly is bit-identical and makes the paper's stated
        # value traceable to config.py rather than to an omitted argument.
        opt = torch.optim.Adam(params, lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)

        # 300-epoch loop, batch 512, shuffled per epoch (Original: 509-537).
        rng = np.random.default_rng(seed)
        bs = min(TRAIN.batch_size, n)
        for _ep in range(TRAIN.epochs):
            order = rng.permutation(n)
            for s0 in range(0, n, bs):
                idx = order[s0:s0 + bs]
                if len(idx) < 2:
                    continue
                bi = torch.as_tensor(idx, device=self._device)
                opt.zero_grad()

                # the embedding head
                emb = self.head(x_fit[bi])
                wn = F.normalize(w, p=2, dim=1)
                loss = cosine_cross_entropy(emb, wn, ytr[bi], TRAIN.logit_scale)
                if TRAIN.supcon_weight > 0:
                    loss = loss + TRAIN.supcon_weight * supcon_loss(emb, ytr[bi], TRAIN.supcon_tau)
                if TRAIN.hamos_weight > 0:
                    loss = loss + TRAIN.hamos_weight * hamos_loss(wn)

                # auxiliary head (separate scale/anchors; Original: 527-535)
                emb_x = self.head_xfer(x_fit[bi])
                wxn = F.normalize(wx, p=2, dim=1)
                loss_x = cosine_cross_entropy(emb_x, wxn, ytr[bi], TRAIN.xfer_logit_scale)
                if TRAIN.supcon_weight > 0:
                    loss_x = loss_x + TRAIN.supcon_weight * supcon_loss(
                        emb_x, ytr[bi], TRAIN.supcon_tau)
                if TRAIN.hamos_weight > 0:
                    loss_x = loss_x + TRAIN.hamos_weight * hamos_loss(wxn)
                loss = loss + TRAIN.xfer_loss_weight * loss_x

                loss.backward()
                opt.step()

        # eval mode for the trained heads (Original: method.py lines 539-548).
        self.head.eval()
        self.head_xfer.eval()

        def _score_emb_np(x_np: np.ndarray) -> np.ndarray:
            """L2-normed float64 embeddings z, chunked (Original: 549-554)."""
            xt = torch.tensor(x_np, device=self._device)
            out: list[np.ndarray] = []
            for s0 in range(0, xt.shape[0], 4096):
                out.append(self.head(xt[s0:s0 + 4096]).cpu().numpy())
            return _l2(np.concatenate(out, 0).astype(np.float64))

        with torch.no_grad():
            self._Wn = F.normalize(w.detach(), p=2, dim=1).cpu().numpy().astype(np.float64)
            z_fit = _score_emb_np(x_all_np[fit_idx])   # unused by v5 path but kept for parity
            # held-out -> rank p-value / conformal quantile
            z_cal = _score_emb_np(x_all_np[cal_idx])
            # full train -> stable margin offset + density
            z_all = _score_emb_np(x_all_np)
        # The champion computed nc_fit but v5 offsets use nc_all; keep parity,
        # drop the array.
        del z_fit

        gen_cal = gen_np[cal_idx]

        # calibration: per-class conformal margin + shrinkage (Original: 556-633).
        self._calibration = ConformalCalibration.fit(
            z_all=z_all,
            z_cal=z_cal,
            gen_all=gen_np,
            gen_cal=gen_cal,
            anchors_normed=self._Wn,
            num_classes=c,
        )

        # density: relative-Mahalanobis + within-class inflation, whose config names
        # say `lang_` but use no language labels (Original: 640, 647-683).
        self._density = RelativeMahalanobis.fit(z_all, gen_np, c)

        # z-norm stats for the margin/RMD fusion (Original: 684-690).
        self._znorm = ZNormStats.fit(z_all, self._Wn, self._calibration, self._density)

        return self

    # -- embed ---------------------------------------------------------------- #
    def embed(self, x: np.ndarray) -> np.ndarray:
        """``X[*, FEAT_DIM]`` -> ``[*, EMB_DIM=288]``: the fused embedding ``z``.

        Port of the head half of ``Method.embed`` (Original: method.py lines 766-779):
        the frozen 2133-d feature row goes through :class:`FactorizedGatedHead`, which
        returns ``L2([z_ssl | 0.1 z_spec | 0.5 z_codec])``. The result is re-normalised
        in float64 so that every consumer (the open-set score, the nearest-anchor label,
        the STOPA cosine protocol) sees exactly the same unit vectors.

        The original then concatenated a second, PCA-whitened SSL channel for STOPA;
        that channel is gone, so ``embed`` is the whole embedding and ``head_xfer`` is
        never called (see the module docstring).
        """
        import torch

        if self.head is None:
            raise RuntimeError("embed() called before fit()/build_head()")

        # Chunked: the STOPA trials matrix is 629,800 x 2133 floats (5 GB), which does not fit
        # a shared GPU in one tensor. Rows are independent, so chunking changes nothing.
        x32 = np.asarray(x, dtype=np.float32)
        if x32.ndim == 1:
            x32 = x32[None, :]
        with torch.no_grad():
            parts = []
            for s0 in range(0, len(x32), 8192):
                t = torch.tensor(x32[s0:s0 + 8192], device=self._device)
                parts.append(self.head(t).detach().cpu().numpy())
        return _l2(np.concatenate(parts, 0).astype(np.float64))      # [N, 288]

    # -- shared input coercion for the scorers -------------------------------- #
    def _embedding_rows(self, x: np.ndarray) -> np.ndarray:
        """Coerce an input into L2-normed embedding rows.

        :meth:`score_openset`, :meth:`attribute` and :meth:`abstain` accept the same
        two widths and reduce them identically before scoring (Original: method.py
        lines 791-826 and 828-879, where the dispatch was duplicated verbatim):

        * ``FEAT_DIM`` raw frontend features -> run through :meth:`embed`;
        * ``EMB_DIM`` embeddings (the :meth:`embed` output) -> used as-is.

        Returns:
            ``[N, EMB_DIM]`` float64 rows, unit-norm.

        Raises:
            ValueError: if the width matches neither accepted dim.
        """
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[-1] == FEAT_DIM:
            return self.embed(a)
        if a.shape[-1] == EMB_DIM:
            return _l2(a)
        raise ValueError(
            f"input width {a.shape[-1]} matches neither FEAT_DIM={FEAT_DIM} nor "
            f"EMB_DIM={EMB_DIM}"
        )

    # -- score_openset -------------------------------------------------------- #
    def score_openset(self, eval_x: np.ndarray) -> np.ndarray:
        """Continuous open-set score (higher = more known).

        Faithful port of ``Method.score_openset`` (Original: method.py lines 791-826).
        ``eval_x`` may be the :meth:`embed` output ``[N, EMB_DIM]`` or raw features
        ``[N, FEAT_DIM]``. The z-normalised anchor margin and relative-Mahalanobis
        terms are summed with equal weight (paper Sec. 2.4); the paper's OOD score is
        the negative of this value.
        """
        from .scoring import fuse_openset_score

        if self._Wn is None:
            raise RuntimeError("score_openset called before fit()")

        return fuse_openset_score(
            e_normed=self._embedding_rows(eval_x),
            anchors_normed=self._Wn,
            calibration=self._calibration,
            density=self._density,
            znorm=self._znorm,
        )

    # -- known-model label ---------------------------------------------------- #
    def attribute(self, eval_x: np.ndarray) -> np.ndarray:
        """Known-model label: the class of the nearest trained anchor.

        The paper's attribution rule (Sec. 2.4): an utterance accepted as known is
        attributed to ``argmin_c delta_c(z) = argmax_c cos(z, w_c)``. Same inputs as
        :meth:`score_openset`; returns ``[N]`` int64 class indices in ``[0, C)``.
        """
        if self._Wn is None:
            raise RuntimeError("attribute called before fit()")
        return np.argmax(self._embedding_rows(eval_x) @ self._Wn.T, axis=1).astype(np.int64)

    # -- distribution-free abstention (optional decision rule) ---------------- #
    def abstain(self, eval_x: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Split-conformal abstention: ``True`` = reject as unknown.

        Faithful port of ``Method.abstain`` / ``_conformal_pvalue`` (Original:
        method.py lines 828-879).  Abstain iff ``max_c p_c(x) < alpha`` (no class
        plausible at level ``alpha``).  ``alpha`` defaults to ``conf_alpha``.
        """
        from .scoring import conformal_pvalue

        if self._Wn is None or self._calibration is None:
            raise RuntimeError("abstain called before fit()")
        if alpha is None:
            alpha = SCORING.conf_alpha
        e = self._embedding_rows(eval_x)
        p_max = conformal_pvalue(e, self._Wn, self._calibration.cal_nc, self._C)
        return p_max < alpha

    # -- checkpoint save / load ---------------------------------------------- #
    def save(self, path: str) -> str:
        """Serialize ALL fitted state to a single ``.pt`` checkpoint.

        Writes a ``torch.save`` dict holding the two head ``state_dict``s (embedding
        head and auxiliary head), the codec-residual train-normalization buffers
        (which are non-persistent and so are captured explicitly), and every numpy
        array of the fitted scoring stack — class anchors ``Wn``, the conformal
        calibration (``cal_q`` / ``cal_conf_q`` / ``cal_nc`` / ``coverage``), the
        relative-Mahalanobis density (per-class + background means and precisions)
        and the z-norm constants. A frozen snapshot of :mod:`sourcetrace.config`
        FEATURES / MODEL / SCORING / TRAIN is stored for provenance and load-time
        validation.

        The serialization is deterministic (no RNG), and a checkpoint round-trip
        (``save`` -> :meth:`load`) reproduces :meth:`embed` and
        :meth:`score_openset` bit-identically.

        Args:
            path: destination ``.pt`` path. Parent directories are created.

        Returns:
            The path written (as a string).
        """
        import dataclasses as _dc
        from pathlib import Path as _Path

        import torch

        from .config import FEATURES as _FEATURES
        from .config import MODEL as _MODEL
        from .config import SCORING as _SCORING
        from .config import TRAIN as _TRAIN

        if self.head is None or self._Wn is None:
            raise RuntimeError("save() called before fit()")

        def _np(a) -> np.ndarray | None:
            """``None``-preserving :func:`np.asarray` for optional fitted arrays."""
            return None if a is None else np.asarray(a)

        # codec-residual buffers are non-persistent (not in state_dict); capture them.
        codec_mu = None if self.head.codec_mu is None else self.head.codec_mu.detach().cpu().numpy()
        codec_sd = None if self.head.codec_sd is None else self.head.codec_sd.detach().cpu().numpy()

        cal = self._calibration
        dens = self._density
        zn = self._znorm

        state: dict = {
            "format": "sourcetrace-method-checkpoint",
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "method_tag": METHOD_TAG,
            # -- constructor / dims ------------------------------------------- #
            "num_known_classes": self.num_known_classes,
            "num_languages": self.num_languages,
            "C": self._C,
            # -- trainable heads (state_dicts, CPU) --------------------------- #
            "head_state": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "head_xfer_state": {k: v.detach().cpu()
                                for k, v in self.head_xfer.state_dict().items()},
            "codec_mu": codec_mu,
            "codec_sd": codec_sd,
            # -- class anchors ------------------------------------------------ #
            "Wn": _np(self._Wn),
            # -- conformal calibration ---------------------------------------- #
            "cal": None if cal is None else {
                "cal_q": _np(cal.cal_q),
                "cal_conf_q": _np(cal.cal_conf_q),
                "cal_nc": [_np(a) for a in cal.cal_nc],
                "coverage": dict(cal.coverage),
            },
            # -- relative-Mahalanobis density --------------------------------- #
            "density": None if dens is None else {
                "bg_mu": _np(dens.bg_mu),
                "bg_prec": _np(dens.bg_prec),
                "cls_mu": _np(dens.cls_mu),
                "cls_prec": [_np(a) for a in dens.cls_prec],
                "num_classes": dens.num_classes,
            },
            # -- z-norm fusion constants -------------------------------------- #
            "znorm": None if zn is None else {
                "cos_mean": zn.cos_mean, "cos_std": zn.cos_std,
                "rmd_mean": zn.rmd_mean, "rmd_std": zn.rmd_std,
            },
            # -- config snapshot (provenance + load validation) --------------- #
            "config": {
                "FEATURES": _dc.asdict(_FEATURES),
                "MODEL": _dc.asdict(_MODEL),
                "SCORING": _dc.asdict(_SCORING),
                "TRAIN": _dc.asdict(_TRAIN),
            },
        }

        out = _Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(out))
        return str(out)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> Method:
        """Load a :class:`Method` from a ``.pt`` checkpoint written by :meth:`save`.

        Rebuilds the two heads on the resolved device (the champion submodule order
        via :func:`sourcetrace.models.build_head_pair`), loads their ``state_dict``s,
        reinstalls the codec-residual normalization buffers, and reconstructs the
        numpy scoring stack (calibration / density / z-norm). A loaded method
        reproduces :meth:`embed` / :meth:`score_openset` bit-identically. Only files
        written by this code load: the format version and the method tag are checked,
        so a checkpoint of the earlier two-channel design is refused rather than
        half-read.

        Args:
            path: source ``.pt`` path.
            device: torch device string (default: ``config.device()``).

        Returns:
            A fitted :class:`Method` ready for :meth:`embed` / :meth:`score_openset`.
        """
        from pathlib import Path

        import torch

        from .config import FEATURES as _FEATURES
        from .config import device as _device
        from .models import build_head_pair
        from .scoring import ConformalCalibration, RelativeMahalanobis, ZNormStats

        # No checkpoint is committed here (checkpoints/ is gitignored), so a missing
        # path is the FIRST thing a new user hits. torch.load's own error for a missing
        # file does not say that, or say what to run instead.
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(
                f"No checkpoint at {str(src)!r}. Fit one:\n"
                f"    python -m sourcetrace.train --task mlaad_v5 --checkpoint {str(src)}\n"
                f"or download the published head:\n"
                f"    python scripts/download_weights.py --task mlaad_v5\n"
                f"Expected numbers are in results/ablation/full.json."
            )
        if src.is_dir():
            raise IsADirectoryError(
                f"{str(src)!r} is a directory; pass the .pt file itself."
            )

        state = torch.load(str(src), map_location="cpu", weights_only=False)
        if state.get("format") != "sourcetrace-method-checkpoint":
            raise ValueError(f"{path!r} is not a sourcetrace method checkpoint")
        if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"{path!r}: checkpoint format_version {state.get('format_version')!r}, this "
                f"code writes and reads {CHECKPOINT_FORMAT_VERSION}. Refit it "
                f"(python -m sourcetrace.train --checkpoint {str(src)}) or download the "
                f"released file (python scripts/download_weights.py).")
        if state.get("method_tag") != METHOD_TAG:
            raise ValueError(f"{path!r}: method tag {state.get('method_tag')!r} is not the one "
                             f"this code was measured with ({METHOD_TAG!r})")

        # validate the feature layout matches (a stale-cache / config-drift guard).
        ckpt_feat_dim = state.get("config", {}).get("FEATURES", {}).get("feat_dim")
        if ckpt_feat_dim is not None and ckpt_feat_dim != _FEATURES.feat_dim:
            raise ValueError(
                f"checkpoint feat_dim {ckpt_feat_dim} != current config feat_dim "
                f"{_FEATURES.feat_dim}; the loaded config differs from the trained one")

        m = cls(num_known_classes=state["num_known_classes"],
                num_languages=state["num_languages"])
        m._C = state["C"]

        # -- rebuild + load the trainable heads --------------------------------- #
        dev = torch.device(device or _device())
        m._device = dev
        m.head, m.head_xfer = build_head_pair(dev)
        m.head.load_state_dict(state["head_state"])
        m.head_xfer.load_state_dict(state["head_xfer_state"])
        # reinstall the non-persistent codec-norm buffers on the embedding head.
        cmu = state.get("codec_mu")
        csd = state.get("codec_sd")
        m.head.set_codec_norm(
            None if cmu is None else torch.as_tensor(cmu, device=dev, dtype=torch.float32),
            None if csd is None else torch.as_tensor(csd, device=dev, dtype=torch.float32),
        )
        m.head_xfer.set_codec_norm(None, None)
        m.head.eval()
        m.head_xfer.eval()

        # -- anchors ------------------------------------------------------------ #
        m._Wn = None if state["Wn"] is None else np.asarray(state["Wn"], dtype=np.float64)

        # -- conformal calibration --------------------------------------------- #
        cal = state.get("cal")
        if cal is not None:
            m._calibration = ConformalCalibration(
                cal_q=np.asarray(cal["cal_q"], dtype=np.float64),
                cal_conf_q=np.asarray(cal["cal_conf_q"], dtype=np.float64),
                cal_nc=[np.asarray(a, dtype=np.float64) for a in cal["cal_nc"]],
                coverage={float(k): float(v) for k, v in cal["coverage"].items()},
            )

        # -- relative-Mahalanobis density -------------------------------------- #
        dens = state.get("density")
        if dens is not None:
            m._density = RelativeMahalanobis(
                bg_mu=np.asarray(dens["bg_mu"], dtype=np.float64),
                bg_prec=np.asarray(dens["bg_prec"], dtype=np.float64),
                cls_mu=np.asarray(dens["cls_mu"], dtype=np.float64),
                cls_prec=[np.asarray(a, dtype=np.float64) for a in dens["cls_prec"]],
                num_classes=int(dens["num_classes"]),
            )

        # -- z-norm fusion constants ------------------------------------------- #
        zn = state.get("znorm")
        if zn is not None:
            m._znorm = ZNormStats(
                cos_mean=float(zn["cos_mean"]), cos_std=float(zn["cos_std"]),
                rmd_mean=float(zn["rmd_mean"]), rmd_std=float(zn["rmd_std"]),
            )

        return m
