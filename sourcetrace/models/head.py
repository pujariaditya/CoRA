"""Factorized gated embedding head: the one embedding z of the paper.

RECONSTRUCTION NOTICE
---------------------
This module was reconstructed from the published method description and the interface
contract asserted throughout ``sourcetrace/method.py``, because the original
``sourcetrace/models/`` (commit ``fa876f7d``) was not included in the public checkout.
The architecture, dimensions, gates, subspace weights and forward math follow METHOD.md;
every design choice not pinned by it is marked ``# ASSUMED``.

RNG / INITIALIZATION ORDER IS LOAD-BEARING
------------------------------------------
Given a fixed seed, every ``nn.Module`` constructed here draws from the global torch RNG
in source order. Reordering, adding, or removing any submodule construction, parameter,
or ``_init_gate`` call shifts every subsequent draw and silently changes all downstream
results. Treat the construction sequence in :class:`FactorizedGatedHead.__init__` and in
:func:`build_head_pair` as frozen; edit the docs here, never the order.

FIXED RANDOM CODEC SUBSPACE
---------------------------
``proj_codec`` and ``gate_codec`` (embedding head only) are **trained with the rest of the
head** since 2026-09-18 (``MODEL.train_codec_path``, default True; ``Method.fit`` appends
them after the original parameter list). The earlier design left them at their seeded
initialisation, a fixed random projection of the train-normalized codec residual; that
is ``train_codec_path: false`` (``configs/ablation_fixed_proj.yaml``) and costs 0.28 FPR95
points on MLAAD v5 (0.7771 against 0.5029).

Interface required by ``method.py``:
    build_head_pair(device) -> (embedding_head, auxiliary_head)
    FactorizedGatedHead(nn.Module):
        .proj_am / .proj_voc / .gate_voc / .gate_tmp   (proj_voc/gates may be None)
        .proj_codec / .gate_codec                      (embedding head only; else None)
        .codec_mu / .codec_sd                          (non-persistent buffers)
        .set_codec_norm(mu, sd); forward(x[N, 2133]) -> [N, 288] (embedding) / [N, 256] (auxiliary)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FEATURES, MODEL

# feature-layout offsets (METHOD.md §1): [ SSL 0:2048 | sig 2048:2100 | codec 2100:2133 ]
_SSL = FEATURES.ssl_dim                       # 2048
_SIG = FEATURES.sig_dim                       # 52
_CODEC = FEATURES.codec_dim                   # 33
_SIG_OFF = _SSL                               # 2048
_CODEC_OFF = _SSL + _SIG                      # 2100
_HALF = _SSL // 2                             # 1024 (mean half | std half)


def _init_gate(lin: nn.Linear) -> None:
    """Near-identity gate init: ``N(0, MODEL.gate_init_std**2)`` weight, zero bias.

    Small-random rather than exactly zero, so the gate units are not symmetric: an
    all-zero weight makes every unit of ``lin`` see the same gradient and stay tied to
    its neighbours. The *near-identity* property comes from ``tanh(0) = 0``, which
    leaves ``z * (1 + beta * tanh(g)) == z`` at initialisation, so training begins from
    a clean concatenation and the modulation is learned on top of it.

    Note the reason to avoid an exact zero is symmetry, not vanishing gradient:
    ``tanh'(0) = 1`` is the *maximal* slope, so gradient flows fine either way.
    See METHOD.md §2.
    """
    nn.init.normal_(lin.weight, mean=0.0, std=MODEL.gate_init_std)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


class FactorizedGatedHead(nn.Module):
    """Project the three frozen channels into gated subspaces and concatenate.

    Embedding head (``codec=True``) -> ``L2([z_am(192) | 0.1 z_voc(64) | 0.5 z_codec(32)])``
    = 288-d, the paper's ``z`` (Eq. 5: ``z_ssl``, ``z_spec``, ``z_codec``).
    Auxiliary head (``codec=False``) -> ``L2([z_am(192) | 0.1 z_voc(64)])`` = 256-d, trained
    only (paper Sec. 2.3) and never read at inference.

    The submodules below are constructed in a fixed order that determines the seeded
    RNG draw sequence; see the module-level "RNG / INITIALIZATION ORDER IS LOAD-BEARING"
    notice. Do not reorder them.
    """

    def __init__(self, codec: bool) -> None:
        super().__init__()
        self.codec = codec

        # AM subspace from the (tmp-gated) SSL vector.
        self.proj_am = nn.Linear(_SSL, MODEL.fact_am_dim)                 # 2048 -> 192
        # temporal-std FiLM: mean-half modulates std-half, bottleneck 1024->64->1024.
        self.gate_tmp = nn.Sequential(                                    # ASSUMED: ReLU inner
            nn.Linear(_HALF, MODEL.tmp_gate_hidden),
            nn.ReLU(),
            nn.Linear(MODEL.tmp_gate_hidden, _HALF),
        )
        _init_gate(self.gate_tmp[0])
        _init_gate(self.gate_tmp[2])

        # vocoder subspace from the 52-d signature, AM->voc FiLM gate.
        self.proj_voc = nn.Linear(_SIG, MODEL.fact_voc_dim)              # 52 -> 64
        self.gate_voc = nn.Linear(MODEL.fact_am_dim, MODEL.fact_voc_dim)  # 192 -> 64
        _init_gate(self.gate_voc)

        # Codec subspace (embedding head only). Trained with the rest of the head since
        # 2026-09-18 (MODEL.train_codec_path); see the module docstring.
        if codec:
            self.proj_codec = nn.Linear(_CODEC, MODEL.fact_codec_dim)    # 33 -> 32
            self.gate_codec = nn.Linear(MODEL.fact_am_dim, MODEL.fact_codec_dim)  # 192 -> 32
            _init_gate(self.gate_codec)
        else:
            self.proj_codec = None
            self.gate_codec = None

        # codec-residual TRAIN normalization buffers (non-persistent; set in fit()).
        self.register_buffer("codec_mu", None, persistent=False)
        self.register_buffer("codec_sd", None, persistent=False)

    # -- codec-residual train-normalization ---------------------------------- #
    def set_codec_norm(self, mu: torch.Tensor | None, sd: torch.Tensor | None) -> None:
        self.codec_mu = mu
        self.codec_sd = sd

    # -- forward -------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ssl = x[:, :_SSL]
        mean, std = ssl[:, :_HALF], ssl[:, _HALF:]
        # temporal-std FiLM (beta = tmp_gate_beta), bounded by tanh (METHOD.md §2).
        std = std * (1.0 + MODEL.tmp_gate_beta * torch.tanh(self.gate_tmp(mean)))
        ssl = torch.cat([mean, std], dim=1)

        z_am = F.normalize(self.proj_am(ssl), p=2, dim=1)                 # [N, 192]
        parts = [z_am]

        # vocoder subspace, AM->voc FiLM (beta = voc_gate_beta).
        sig = x[:, _SIG_OFF:_SIG_OFF + _SIG]
        z_voc = F.normalize(self.proj_voc(sig), p=2, dim=1)              # [N, 64]
        z_voc = z_voc * (1.0 + MODEL.voc_gate_beta * torch.tanh(self.gate_voc(z_am)))
        parts.append(MODEL.voc_weight * z_voc)

        # codec subspace (embedding head), AM->codec FiLM (beta = codec_gate_beta).
        if self.codec and self.proj_codec is not None:
            cr = x[:, _CODEC_OFF:_CODEC_OFF + _CODEC]
            if self.codec_mu is not None and self.codec_sd is not None:
                cr = (cr - self.codec_mu) / self.codec_sd                 # TRAIN-normalized
            z_codec = F.normalize(self.proj_codec(cr), p=2, dim=1)        # [N, 32]
            z_codec = z_codec * (1.0 + MODEL.codec_gate_beta * torch.tanh(self.gate_codec(z_am)))
            parts.append(MODEL.codec_weight * z_codec)

        return F.normalize(torch.cat(parts, dim=1), p=2, dim=1)


def build_head_pair(device: torch.device | str):
    """Build the embedding head (codec=True) and the auxiliary head (codec=False).

    The embedding head is constructed first, then the auxiliary head. This order is ASSUMED
    (the champion's exact submodule/RNG order is not recoverable, so bit-identical
    seeded init is not guaranteed — see the module RECONSTRUCTION NOTICE), but it is
    now fixed: both heads draw from the same global torch RNG, so swapping the two
    lines below re-rolls every weight in both heads and changes all results. Do not
    reorder them.

    The two heads are **independent objects sharing no parameters**. The auxiliary head
    is trained by its own loss terms and then never read at inference (``Method.embed``
    calls only the embedding head), so no gradient from its loss reaches the embedding
    head and its trained weights influence no reported number.

    IT IS STILL CONSTRUCTED, AND DELETING IT CHANGES THE RESULTS. Not the embedding head,
    which is built on the line above with its initial weights already drawn, and not the
    per-class anchors, which ``Method.fit`` draws from dedicated CPU generators seeded 0
    and 1, independent of the global stream. What it moves is everything the global RNG
    serves *after* this point --
    in particular the HamOS virtual-outlier draw
    (``losses/objectives.py``, which documents its dependence on the global stream), once
    per batch per epoch. Different outliers, different fit, different reported numbers.
    Since ``results/`` was measured with this call present, removing it is not a
    simplification; it is a silent invalidation of every number the paper reports.
    """
    embedding = FactorizedGatedHead(codec=True).to(device)
    # DEAD AT INFERENCE, load-bearing for the RNG draw: see the note above.
    auxiliary = FactorizedGatedHead(codec=False).to(device)
    return embedding, auxiliary
