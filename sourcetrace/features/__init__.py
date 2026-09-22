"""Feature extraction for open-set audio-deepfake source tracing.

Faithful port of the champion (commit ``fa876f7d``) feature pipeline: a frozen
WavLM-Large front-end plus two classical channels measured directly from the
waveform, concatenated into a 2133-d vector. Whether the SSL already carries what
those channels measure is untested here; see :mod:`.signature`.

    audio (16 kHz, 4 s) -> [ WavLM pool (2048) | signature (52) | codec (33) ]

Channels
--------
* :mod:`.wavlm`          -- frozen WavLM-Large (s3prl), layers 1-4, per-layer
  temporal mean|std, cross-layer mean -> 2048-d.
* :mod:`.signature`      -- 52-d phase group-delay + modulation + codec-grid,
  per-clip standardized.
* :mod:`.codec_residual` -- 33-d EnCodec-24kHz reconstruction residual, RAW
  (train-normalized later in the method head).

Public API
----------
* :class:`Extractor`       -- lazy, cached, frozen extractor with ``embed`` /
  ``embed_cached`` / ``embed_files``.
* :func:`extract_features` -- one-shot ``waveforms -> (N, 2133)`` helper.

Import cost
-----------
Importing this subpackage imports ``torch`` (the model weights themselves still
load lazily, on first ``embed``). The parent package keeps ``import sourcetrace``
torch-free by resolving :class:`Extractor` / :func:`extract_features` through a
PEP-562 ``__getattr__``, so nothing here may be imported from
``sourcetrace/__init__.py`` at module level.
"""
from __future__ import annotations

from .codec_residual import CodecResidual
from .extract import Extractor, extract_features
from .signature import compute_signature
from .wavlm import WavLMFrontend, tile_pad

__all__ = [
    "Extractor",
    "extract_features",
    "WavLMFrontend",
    "tile_pad",
    "compute_signature",
    "CodecResidual",
]
