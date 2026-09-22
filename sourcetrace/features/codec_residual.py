"""Neural-codec reconstruction-residual channel: waveform -> 33-d RAW vector.

Faithful port of ``extractor.py`` ``_codec_residual`` (L238-277). Each clip is
re-encoded through EnCodec-24kHz at three bandwidths; the per-bandwidth
reconstruction RESIDUAL is turned into features.

WHAT THIS CHANNEL IS KNOWN TO DO, AND WHAT IT IS NOT. Measured: removing it raises
open-set FPR95 from 0.50% to 1.28% and OOD-EER from 2.45% to 3.45%, lowers known-model
accuracy from 99.86% to 99.68% (``results/ablation/``) and raises the STOPA held-out
EER from 6.80% to 8.08%, so the model reads it. NOT established: what it encodes.
The residual is a property of the clip
as it reaches us, not of the generator alone, so it may equally track the encoder or
transmission channel a family's clips passed through, or silence and duration
statistics; Sec. 4 of the paper names the control that would separate those accounts.
Read this as a measurement taken against the codec's own analysis grid, outside the
SSL -- not as a generator fingerprint.

IMPORTANT -- emits RAW features. Unlike the signature channel, this channel is
NOT per-clip standardized here. Train-set mean/std normalization happens later in
the method head (``method.py.fit``), so cross-clip variation survives. Do not add
normalization here.

Layout (33 = 3 * (1 + 8) + 3 + 3)
---------------------------------
Per bandwidth (x3): 1 log residual-energy ratio (dB) + 8 log1p band energies = 9,
so 27 dims total. The 8 bands are UNIFORM LINEAR-FREQUENCY splits of the residual
STFT bins (``torch.linspace`` over bin indices) -- they are NOT mel bands.
Cross-codec attribution appends 3 softmax(-energy) weights + 3 pairwise energy
diffs = 6. Total 27 + 6 = 33.

Source-line map (champion ``extractor.py``)
------------------------------------------
* lazy EnCodec-24kHz load ............... L245-247
* 16k -> 24k nearest-index resample ..... L248-253
* encode/decode per bandwidth ........... L256-259
* residual + log energy ratio (dB) ...... L260-263
* residual STFT (nfft=1024, hop=256) .... L264-266
* 8 linear-freq band means, log1p ....... L267-270
* softmax(-energy) attribution + pairwise diffs .. L271-276
"""
from __future__ import annotations

import torch

from ..config import ENCODEC_MODEL_ID, FEATURES

# Frozen codec constants (= champion extractor.py module constants).
_SR = FEATURES.sample_rate               # 16000
_CODEC_SR = FEATURES.codec_sr            # 24000
_BWS: tuple[float, ...] = FEATURES.codec_bandwidths  # (1.5, 6.0, 12.0) kbps
_BANDS = FEATURES.codec_bands            # 8 residual band energies per bandwidth

# Residual-STFT geometry (frozen; hard-coded in the champion extractor).
_RES_NFFT = 1024
_RES_HOP = 256

_EPS = 1e-12  # energy-ratio guard (frozen)


def _resample_nearest(x16: torch.Tensor) -> torch.Tensor:
    """Upsample 16 kHz -> 24 kHz by nearest-index gather.

    This is deliberately NOT a real resampler (no filtering / interpolation): it
    is a plain index gather that reproduces the champion extractor bit-for-bit.
    Do not replace it with ``torchaudio.resample``.

    Args:
        x16: ``(B, T)`` waveform at 16 kHz.

    Returns:
        ``(B, T24)`` gathered waveform, same dtype/device as ``x16``.
    """
    ratio = _CODEC_SR / float(_SR)
    n24 = int(round(x16.shape[-1] * ratio))
    idx = (torch.arange(n24, device=x16.device, dtype=torch.float32) / ratio).long()
    idx = torch.clamp(idx, 0, x16.shape[-1] - 1)
    return x16.index_select(-1, idx)


class CodecResidual:
    """EnCodec-24kHz reconstruction-residual feature extractor (33-d, RAW)."""

    def __init__(self, device: str | torch.device) -> None:
        """Args: ``device`` -- torch device the EnCodec model runs on."""
        self.device = torch.device(device)
        self._encodec = None  # lazy: loaded on first call (keeps import cheap)

    def _ensure_model(self) -> None:
        """Load and eval the EnCodec-24kHz model on first use."""
        if self._encodec is None:
            # Routed through _deps so a user-site huggingface_hub shadow produces
            # actionable guidance instead of transformers' own advice to upgrade,
            # which would break the >=4.38,<4.40 EnCodec pin.
            from ._deps import load_encodec_cls

            EncodecModel = load_encodec_cls()

            self._encodec = (
                EncodecModel.from_pretrained(ENCODEC_MODEL_ID).to(self.device).eval()
            )

    def _reconstruct(self, x24c: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Encode + decode one batch at ``bandwidth`` kbps.

        Args:
            x24c: ``(B, 1, T24)`` channel-first 24 kHz waveform.
            bandwidth: EnCodec target bandwidth in kbps.

        Returns:
            ``(B, T_dec)`` decoded waveform (channel dim squeezed).
        """
        codes, scales = self._encodec.encode(
            x24c, bandwidth=float(bandwidth), return_dict=False
        )
        out = self._encodec.decode(codes, scales)
        decoded = out.audio_values if hasattr(out, "audio_values") else out[0]
        return decoded.squeeze(1)

    def compute(self, x16: torch.Tensor) -> torch.Tensor:
        """Reconstruction-residual features for a batch of 16 kHz clips.

        Faithful port of ``extractor.py`` ``_codec_residual``.

        Args:
            x16: ``(B, T)`` waveform at 16 kHz on any device.

        Returns:
            ``(B, 33)`` RAW features (no normalization), same dtype as ``x16``.
        """
        self._ensure_model()

        x24 = _resample_nearest(x16)  # (B, T24)
        x24c = x24.unsqueeze(1)  # (B, 1, T24)

        per_bw: list[torch.Tensor] = []       # [er, band energies] per bandwidth
        energy_ratios: list[torch.Tensor] = []  # (B,) dB ratio per bandwidth
        with torch.no_grad():
            for bw in _BWS:
                dec = self._reconstruct(x24c, bw)
                t = min(x24.shape[-1], dec.shape[-1])
                residual = x24[..., :t] - dec[..., :t]

                # log residual-energy ratio, in dB.
                er = 10.0 * torch.log10(
                    (residual.pow(2).sum(-1) + _EPS)
                    / (x24[..., :t].pow(2).sum(-1) + _EPS)
                )

                # residual STFT -> 8 linear-frequency band-mean powers -> log1p.
                win = torch.hann_window(
                    _RES_NFFT, device=residual.device, dtype=residual.dtype
                )
                spec = torch.stft(
                    residual, n_fft=_RES_NFFT, hop_length=_RES_HOP, window=win,
                    return_complex=True, center=True,
                )
                power = spec.abs() ** 2  # (B, F, T)
                n_bins = power.shape[1]
                # Uniform split of the FFT bin axis == linear frequency bands.
                edges = torch.linspace(
                    0, n_bins, _BANDS + 1, device=residual.device
                ).long()
                band_energy = torch.stack(
                    [
                        power[:, edges[b] : edges[b + 1]].mean(-1).mean(-1)
                        for b in range(_BANDS)
                    ],
                    dim=1,
                )  # (B, 8)

                per_bw.append(er.unsqueeze(1))
                per_bw.append(torch.log1p(band_energy))
                energy_ratios.append(er)

            feats = torch.cat(per_bw, dim=1)  # (B, 27)
            ers = torch.stack(energy_ratios, dim=1)  # (B, n_bw)
            # cross-codec attribution: softmax over NEGATED residual energies
            # (lower residual = better-matched codec).
            sm = torch.softmax(-ers, dim=1)
            pairs = [
                (ers[:, i] - ers[:, j]).unsqueeze(1)
                for i in range(len(_BWS))
                for j in range(i + 1, len(_BWS))
            ]
            cross = torch.cat([sm] + pairs, dim=1) if pairs else sm  # (B, 6)
            feats = torch.cat([feats, cross], dim=1)  # (B, 33)
        return feats.to(x16.dtype)
