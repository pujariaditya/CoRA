"""Compose the three feature channels into the 2133-d source-tracing vector.

Faithful port of ``extractor.py`` ``SSLExtractor.embed`` (L279-295), which
concatenates:

    [ WavLM cross-layer pool (2048) | signature (52) | codec residual (33) ]
    = 2133-d  (== config.FeatureConfig.feat_dim)

On top of that port it adds the public :class:`Extractor`, which offers three
entry points -- :meth:`~Extractor.embed` (arrays in memory),
:meth:`~Extractor.embed_cached` (same, backed by an on-disk ``.npy`` cache keyed
by an explicit key or a content hash) and :meth:`~Extractor.embed_files` (audio
paths on disk) -- plus the :func:`extract_features` one-shot helper. No paths are
hardcoded: the cache root comes from ``config.PATHS.feature_cache``
(``ST_FEATURE_CACHE``).

Source-line map (champion ``extractor.py``)
------------------------------------------
* tile-pad each waveform ................. L281  -> wavlm.tile_pad
* batched no_grad forward (256) .......... L283-285
* WavLM pooled feature ................... L286-287
* + signature channel .................... L288-290
* + codec-residual channel (RAW) ......... L291-293
* concat over batches -> (N, 2133) ....... L294-295
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch

from ..config import FEATURES, PATHS
from ..config import device as resolve_device
from .codec_residual import CodecResidual
from .signature import compute_signature
from .wavlm import WavLMFrontend, tile_pad

# Batch size for the SSL / signature / codec forward passes (= champion L284).
_BATCH = 256


class Extractor:
    """Public feature extractor: waveforms -> 2133-d vectors, with a disk cache.

    Composes :class:`WavLMFrontend` (2048), :func:`compute_signature` (52) and
    :class:`CodecResidual` (33). All three sub-models load lazily on first use
    and stay frozen (``torch.no_grad``). Deterministic: identical input arrays
    yield identical features.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        """Args: ``device`` -- torch device; defaults to ``config.device()``."""
        self.device = torch.device(device or resolve_device())
        self._wavlm = WavLMFrontend(self.device)
        self._codec = CodecResidual(self.device)

    @property
    def feat_dim(self) -> int:
        """Total feature width = 2048 + 52 + 33 = 2133."""
        return self._wavlm.out_dim + FEATURES.sig_dim + FEATURES.codec_dim

    # -- core embedding ----------------------------------------------------- #
    def embed(self, waveforms: Sequence[np.ndarray], sr: int) -> np.ndarray:
        """Embed raw waveforms -> ``(N, 2133)`` ``float32``.

        Faithful port of ``SSLExtractor.embed``: tile-pad, batch by 256, and
        concatenate the WavLM, signature and (RAW) codec-residual channels.

        Args:
            waveforms: Iterable of 1-D mono (or 2-D multi-channel) float arrays.
            sr: Sample rate shared by all input arrays.

        Returns:
            ``(N, 2133)`` feature matrix (SSL 2048 | signature 52 | codec 33).
        """
        # tile_pad is idempotent, so callers that already normalized to 16 kHz
        # (e.g. embed_files) pay only a cheap no-op crop here.
        padded = np.stack([tile_pad(w, sr) for w in waveforms])  # (N, 64000)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(padded), _BATCH):
                x = torch.from_numpy(padded[i : i + _BATCH]).to(self.device)
                feat = self._wavlm.embed(x)  # (B, 2048)
                sig = compute_signature(x).to(feat.dtype)  # (B, 52)
                feat = torch.cat([feat, sig], dim=1)
                cr = self._codec.compute(x).to(feat.dtype)  # (B, 33) RAW
                feat = torch.cat([feat, cr], dim=1)  # (B, 2133)
                chunks.append(feat.cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    # -- on-disk cache ------------------------------------------------------ #
    @staticmethod
    def _content_hash(waveforms: Sequence[np.ndarray], sr: int) -> str:
        """Stable content hash over the (tile-padded) input, keyed by feat layout.

        Includes the feature-config fingerprint so a config change invalidates
        the cache. Hashes the tile-padded 16 kHz arrays (post-preprocessing) so
        equivalent inputs at any source rate map to one key.

        FROZEN -- do not edit the fingerprint string, the hashed bytes, the
        digest, or the 16-char truncation. These define the on-disk cache key of
        an already-extracted feature cache; any change silently invalidates it
        and forces a full GPU re-extraction.
        """
        h = hashlib.sha256()
        h.update(
            f"v1|{FEATURES.feat_dim}|{FEATURES.wavlm_layers}|{FEATURES.pool_stats}|"
            f"{FEATURES.codec_bandwidths}|{sr}".encode()
        )
        for w in waveforms:
            a = tile_pad(w, sr)
            h.update(a.tobytes())
        return h.hexdigest()[:16]

    def embed_cached(
        self,
        waveforms: Sequence[np.ndarray],
        sr: int,
        *,
        key: str | None = None,
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        """Embed with an on-disk ``.npy`` cache keyed by ``key`` or a content hash.

        Args:
            waveforms: Input arrays.
            sr: Shared sample rate.
            key: Explicit cache key (e.g. ``"train/en"``). Slashes are allowed and
                map to sub-directories. If ``None``, a content hash is used.
            cache_dir: Override cache root (default ``config.PATHS.feature_cache``).

        Returns:
            ``(N, 2133)`` feature matrix, loaded from cache when present.
        """
        root = Path(cache_dir) if cache_dir is not None else PATHS.feature_cache
        name = key if key is not None else self._content_hash(waveforms, sr)
        path = (root / f"{name}.npy").resolve()
        if path.exists():
            return np.load(path)
        feats = self.embed(waveforms, sr)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, feats)
        return feats

    # -- file -> feature batch runner --------------------------------------- #
    def embed_files(
        self,
        paths: Iterable[str | Path],
        *,
        key: str | None = None,
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        """Load audio files and embed them -> ``(N, 2133)``.

        Every file is read at its native rate and normalized by :func:`tile_pad`
        (mono-mix, nearest-index resample to 16 kHz, tile-pad/crop to 4 s) before
        the batch goes to :meth:`embed_cached`, so files of differing rates and
        channel counts can be mixed freely. Requires ``soundfile``.

        Args:
            paths: Audio file paths.
            key: Optional cache key (see :meth:`embed_cached`).
            cache_dir: Optional cache-root override.

        Returns:
            ``(N, 2133)`` feature matrix in the order of ``paths``.
        """
        import soundfile as sf

        files = [str(p) for p in paths]
        assert files, "no input files provided"
        # One open per file: sf.read returns (samples, samplerate) read from the
        # same libsndfile header that sf.info(...).samplerate reports, so using
        # it directly leaves tile_pad's inputs bit-identical.
        waveforms: list[np.ndarray] = []
        for path in files:
            samples, file_sr = sf.read(path, dtype="float32", always_2d=False)
            waveforms.append(tile_pad(samples, file_sr))
        # Already at FEATURES.sample_rate, so tile_pad inside embed_cached /
        # embed is a no-op (idempotent) -- pass the post-resample rate, not the
        # per-file one.
        return self.embed_cached(
            waveforms, FEATURES.sample_rate, key=key, cache_dir=cache_dir
        )


def extract_features(
    waveforms: Sequence[np.ndarray],
    sr: int,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Convenience one-shot: build an :class:`Extractor` and embed.

    Args:
        waveforms: Input arrays.
        sr: Shared sample rate.
        device: Optional device override.

    Returns:
        ``(N, 2133)`` feature matrix.
    """
    return Extractor(device=device).embed(waveforms, sr)
