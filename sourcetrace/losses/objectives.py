"""Training objectives actually used by the champion ``Method.fit``.

The champion ``loss.py`` shipped a *menu* of metric-learning heads (ArcFace,
sub-center ArcFace, RPL, Proxy-Anchor, GRL/HSIC language-adversarial). **None of
them is called in ``Method.fit``.**  The fit loop trains a bare classifier weight
matrix ``W`` with three terms only (Original: method.py lines 519-535):

#. **Cosine cross-entropy** (a cosine classifier at scale 12): logits
   ``= LOGIT_SCALE * (emb @ Wn.T)`` with L2-normed anchors ``Wn``, fed to
   ``F.cross_entropy``.  No additive angular margin is applied (the margin heads in
   the menu are unused), so this is plain scaled-cosine softmax CE, not CosFace.
#. **Supervised contrastive** (``SUPCON_W = 0.05``, ``SUPCON_TAU = 0.10``).
#. **HamOS virtual-OOD** (``HAMOS_W = 0.03``, ``HAMOS_NOOD = 256``, ``HAMOS_TAU = 0.07``).

This module ports exactly those three, preserving every operation and constant.
Constants are imported from :mod:`sourcetrace.config` (never redefined).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import TRAIN

__all__ = ["cosine_ce_logits", "cosine_cross_entropy", "supcon_loss", "hamos_loss"]


def cosine_ce_logits(emb: torch.Tensor, anchors_normed: torch.Tensor) -> torch.Tensor:
    """UNscaled cosine logits (Original: method.py lines 520-521, 528-529).

    Returns exactly ``emb @ Wn.T`` — a matrix of cosine similarities, since ``emb`` and
    ``anchors_normed`` (``Wn``) are both L2-normed. **The logit scale is not applied
    here.** :func:`cosine_cross_entropy` multiplies by it: ``TRAIN.logit_scale`` (12) for
    the embedding head, ``TRAIN.xfer_logit_scale`` (14) for the auxiliary head.

    Naming this "scaled-cosine logits" would describe behaviour the function does not
    have, which is worse than no docstring: a caller who fed the result straight to
    ``cross_entropy`` would train at temperature 1 and never see an error.
    """
    return emb @ anchors_normed.t()


def cosine_cross_entropy(
    emb: torch.Tensor,
    anchors_normed: torch.Tensor,
    labels: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Temperature-scaled cosine softmax cross-entropy. **There is no margin.**

    Faithful port of ``F.cross_entropy(scale * (emb @ Wn.t()), y)`` (Original:
    method.py lines 521-522 for the embedding head, 529-530 for the auxiliary head).

    This is often described as "CosFace-style" because it uses a cosine classifier with
    L2-normed anchors, but the CosFace additive margin is identically ``m = 0`` here:
    nothing is subtracted from the target logit, and no margin parameter exists anywhere
    in this file or in ``config.py``. It is plain scaled-cosine softmax CE — do not read
    a margin into it, and do not "restore" one.
    """
    logits = scale * cosine_ce_logits(emb, anchors_normed)
    return F.cross_entropy(logits, labels)


def supcon_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    tau: float = TRAIN.supcon_tau,
) -> torch.Tensor:
    """Batch supervised-contrastive loss on already-L2-normalized embeddings.

    Faithful port of ``Method._supcon_t`` (Original: method.py lines 413-432).
    Weighted by ``TRAIN.supcon_weight`` (0.05) at the call site.  Returns a scalar
    ``0`` tensor when the batch has no valid positive pairs (n < 3 or no same-class
    off-diagonal pairs), matching the original guards exactly.
    """
    n = emb.shape[0]
    if n < 3:
        return emb.new_tensor(0.0)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(n, device=emb.device, dtype=torch.bool)
    pos = same & ~eye
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return emb.new_tensor(0.0)
    sim = (emb @ emb.t()) / float(tau)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * (~eye).to(emb.dtype)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_pos = (log_prob * pos.to(emb.dtype)).sum(dim=1)[valid] / pos_count[valid].to(emb.dtype)
    return -mean_log_pos.mean()


def hamos_loss(anchors_normed: torch.Tensor) -> torch.Tensor:
    """HamOS-style virtual-OOD loss (Original: method.py lines 434-441).

    Sample ``HAMOS_NOOD`` (256) points uniformly on the unit sphere and penalise any
    that lie closer than ``HAMOS_TAU`` (0.07) cosine to the nearest known anchor,
    carving open space around the known generators::

        z_ood   = normalize(randn(n_ood, D))
        max_cos = (z_ood @ Wn.T).max(dim=1)
        loss    = clamp(max_cos - tau, min=0).mean()

    Weighted by ``TRAIN.hamos_weight`` (0.03) at the call site.

    .. warning::
       The ``randn`` below consumes the (already-seeded) global torch RNG on the
       anchors' device, exactly as the original does. Every later random draw in
       training depends on it, so do NOT reorder this call relative to the rest of
       ``fit``, do NOT make it conditional (e.g. skipping it when the weight is 0),
       and do NOT hoist the sample out of the loop. Any of those re-rolls the whole
       run and silently changes the reported numbers.
    """
    z_ood = F.normalize(
        torch.randn(TRAIN.hamos_n_ood, anchors_normed.shape[1], device=anchors_normed.device),
        p=2,
        dim=1,
    )
    max_cos = (z_ood @ anchors_normed.t()).max(dim=1).values
    return torch.clamp(max_cos - TRAIN.hamos_tau, min=0.0).mean()
