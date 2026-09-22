"""Frozen WavLM-Large front-end: waveform -> 2048-d cross-layer pooled vector.

Faithful port of the SSL half of the champion feature extractor. Every numerical
operation and constant is preserved exactly; only structure, naming, typing and
docs are improved.

Model-load mechanism
--------------------
This frontend loads WavLM-Large directly through **s3prl**
(``s3prl.hub.wavlm_large()``) — the exact model the champion used. We KEEP s3prl
rather than ``transformers.WavLMModel`` because the two are NOT numerically
identical (differing input normalization, feature-projection handling and
hidden-state indexing), and this is a faithful port whose results must
reproduce. s3prl returns a dict ``{"hidden_states": [...]}`` of 25 tensors
``(B, T, 1024)``; we consume those raw hidden states and pool them ourselves.

Source-line map (champion ``extractor.py`` / ``model.py``)
--------------------------------------------------------
* preprocessing / tile-pad ...... ``extractor.py`` ``_tile_pad`` (L115-127)
* per-layer temporal mean|std ... ``model.py`` ``_layerstats`` (L291-302)
* stack of 25 hidden states ...... ``model.py`` ``_layerstats`` (L293-294)
* layer subset [4,5,6,7] ......... ``extractor.py`` ``LAYERS`` (L56) + ``_pool_layers`` (L162-172)
* cross-layer mean ............... ``extractor.py`` ``_pool_layers`` POOLING="mean" (L165-166)

POOL_STATS note
---------------
``model.py`` line 84 defines ``POOL_STATS = ("mean", "std")`` and
``_layerstats`` concatenates BOTH along the feature axis -> per-layer width is
``2 * 1024 = 2048``. The champion ``extractor.py`` docstring claiming
``("mean",)`` (1024) is STALE and does not match the executed code. This port
follows the code: SSL half = 2048-d, matching ``config.FeatureConfig.ssl_dim``.
(Verified against ``config.FeatureConfig``: ``pool_stats == ("mean", "std")``
and ``ssl_dim == 2048``.)

Layout of the 2048-d output
---------------------------
The output is ``[ mean-half (1024) | std-half (1024) ]``, where each half has
ALREADY been averaged across layers 1-4. **No per-layer identity survives**:
``pool_layers`` collapses the layer axis, so the 2048-d vector holds no separate
L4/L5/L6/L7 blocks. Any downstream split of the 1024-d mean-half into four
256-wide blocks is a slicing of the *feature* axis (contiguous quarters of the
WavLM hidden dimension), NOT a per-layer decomposition -- see the note on
:meth:`WavLMFrontend.pool_layers`.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ..config import FEATURES
from ..config import device as resolve_device

# Which per-layer temporal statistics to compute; mirrors the executed
# ``model.py`` POOL_STATS = ("mean", "std"). Only MEMBERSHIP is read here --
# the concatenation order is fixed by the if-chain in ``_layerstats`` (mean,
# then std, then max), so reordering this tuple does not reorder the output,
# but adding/removing an entry changes the output width and the downstream
# meaning of each half of the 2048-d vector.
_POOL_STATS: tuple[str, ...] = FEATURES.pool_stats

# WavLM-Large hidden width per transformer layer.
_HIDDEN_DIM = 1024


def tile_pad(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Mono-mix, nearest-index resample to 16 kHz, tile-pad/crop to ``clip_seconds``.

    Faithful port of ``extractor.py`` ``_tile_pad`` (L115-127). The resample is a
    deliberately cheap nearest-index gather (NOT a polyphase/sinc resampler) --
    reproducing it exactly is required for identical features.

    Args:
        waveform: 1-D mono or 2-D ``(T, C)`` multi-channel float array.
        sr: Sample rate of ``waveform``.

    Returns:
        ``float32`` array of length ``clip_seconds * sample_rate`` (64000 @ 16 kHz).
        An empty or all-silent input yields an all-zero clip.
    """
    target_sr = FEATURES.sample_rate
    n_out = int(FEATURES.clip_seconds * target_sr)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim > 1:  # mono-mix multi-channel by averaging across channels
        audio = audio.mean(axis=1)
    if sr != target_sr:  # nearest-index resample (matches champion exactly)
        idx = (np.arange(int(len(audio) * target_sr / sr)) * sr / target_sr).astype(np.int64)
        audio = audio[np.clip(idx, 0, len(audio) - 1)] if len(audio) else audio
    if audio.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    # Tile-repeat short clips up to n_out samples then crop; long clips crop to n_out.
    audio = np.tile(audio, n_out // len(audio) + 1)[:n_out] if len(audio) < n_out else audio[:n_out]
    return audio.astype(np.float32, copy=False)


class WavLMFrontend:
    """Frozen WavLM-Large front-end producing a 2048-d cross-layer pooled vector.

    Loads WavLM-Large once (lazily, via s3prl), freezes it, and pools the
    selected transformer layers with per-layer temporal mean|std followed by a
    cross-layer mean. The model is never trained here (frozen, ``torch.no_grad``).

    The 2048-d result is ``[mean-half (1024) | std-half (1024)]`` with the layer
    axis already collapsed -- it carries no per-layer structure. Use
    :meth:`layerstats` if per-layer (B, L, 2D) statistics are needed.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        layers: Sequence[int] = FEATURES.wavlm_layers,
    ) -> None:
        """Build the frozen front-end.

        Args:
            device: Torch device string/object; defaults to ``config.device()``.
            layers: Hidden-state indices to pool (default ``(1, 2, 3, 4)``).

                A three-arm sweep at matched width and capacity
                (``results/layer_band_summary.json``, reproduce with
                ``analysis/pooling_variants.py`` + ``analysis/fit_variant.py``) gives
                FPR95 0.50 at layers 1-4, 0.91 at 4-7 and 1.58 at 8-11 (trained residual
                projection) -- monotone in depth, and the development split agrees
                (dev OOD-EER 1.95 / 2.11 / 3.16). 1-4 is the band every file in
                ``results/`` and both checkpoints use. Sec. 3.4 of the paper reports
                the sweep.
        """
        self.device = torch.device(device or resolve_device())
        self._requested_layers: tuple[int, ...] = tuple(int(x) for x in layers)
        self._model: torch.nn.Module | None = None  # lazy: built on first use
        # Requested layers clamped to what the loaded model actually exposes;
        # populated by _ensure_model().
        self._layers: list[int] = []

    # -- lazy model construction ------------------------------------------- #
    def _ensure_model(self) -> None:
        """Load, freeze and eval WavLM-Large on first use (single GPU load)."""
        if self._model is not None:
            return
        # Load WavLM-Large directly via s3prl. This is the exact model the
        # champion used (it wrapped this same ``hub.wavlm_large()``); we read its
        # raw ``hidden_states`` and pool them ourselves, so features are identical.
        from s3prl import hub

        wavlm = hub.wavlm_large().to(self.device)
        for param in wavlm.parameters():
            param.requires_grad_(False)
        wavlm.eval()
        self._model = wavlm

        # Discover the real hidden-state count via a 1-sample dummy forward, then
        # clamp the requested layers (mirrors extractor.py L148-158 intent).
        with torch.no_grad():
            probe = self._model(torch.zeros(1, FEATURES.sample_rate, device=self.device))
            n_layers = len(probe["hidden_states"])
        self._layers = [i for i in self._requested_layers if 0 <= i < n_layers]
        if not self._layers:
            raise ValueError(
                f"No valid WavLM layers: requested {self._requested_layers} but the "
                f"frontend exposes only {n_layers} layers (0..{n_layers - 1})."
            )

    @property
    def out_dim(self) -> int:
        """Output width = ``len(pool_stats) * 1024`` (2048 for mean|std)."""
        return len(_POOL_STATS) * _HIDDEN_DIM

    # -- pooling ------------------------------------------------------------ #
    @staticmethod
    def _layerstats(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Per-layer temporal mean|std over all hidden states.

        Faithful port of ``model.py`` ``_layerstats`` (L291-302): stack the 25
        hidden states ``(B, L, T, D)``, then concatenate temporal ``mean`` and
        (unbiased=False) ``std`` along the feature axis -> ``(B, L, 2D)``.

        The concatenation order below (mean, std, max) is fixed by this if-chain,
        independent of the order of ``_POOL_STATS``.
        """
        hidden_states = model(x)["hidden_states"]  # list of L tensors (B, T, D)
        hs = torch.stack(hidden_states, dim=1)  # (B, L, T, D)
        parts: list[torch.Tensor] = []
        if "mean" in _POOL_STATS:
            parts.append(hs.mean(dim=2))
        if "std" in _POOL_STATS:
            parts.append(hs.std(dim=2, unbiased=False))
        if "max" in _POOL_STATS:
            parts.append(hs.amax(dim=2))
        return torch.cat(parts, dim=-1)  # (B, L, len(POOL_STATS)*D)

    def layerstats(self, x: torch.Tensor) -> torch.Tensor:
        """Public wrapper: ``(B, T)`` waveform -> ``(B, L, 2D)`` per-layer stats.

        A ``(B, T, C)`` input is reduced to its FIRST channel (not mono-mixed);
        :func:`tile_pad` is the path that averages channels.
        """
        self._ensure_model()
        x = x.to(self.device)
        if x.ndim == 3:
            x = x[:, :, 0]
        return self._layerstats(self._model, x)

    def pool_layers(self, ls: torch.Tensor) -> torch.Tensor:
        """Cross-layer mean over the selected layers.

        Faithful port of ``extractor.py`` ``_pool_layers`` (L162-172) for the
        settled ``POOLING = "mean"``: select ``ls[:, layers, :]`` then mean over
        the layer axis -> ``(B, 2D)``.

        This averaging DESTROYS per-layer identity. The returned ``(B, 2048)``
        vector is ``[mean-half (1024) | std-half (1024)]``, each half being a
        single cross-layer average over the band -- there are no per-layer
        sub-blocks anywhere in it. Downstream code that splits the 1024-d
        mean-half into four 256-wide blocks is slicing contiguous quarters of the
        WavLM feature axis, not separating layers.

        Args:
            ls: ``(B, L, 2D)`` per-layer statistics from :meth:`layerstats`.

        Returns:
            ``(B, 2D)`` cross-layer mean over ``self._layers``.
        """
        return ls[:, self._layers, :].mean(1)

    # -- top-level embedding ------------------------------------------------ #
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` waveform tensor on device -> ``(B, 2048)`` pooled feature."""
        self._ensure_model()
        with torch.no_grad():
            return self.pool_layers(self.layerstats(x))
