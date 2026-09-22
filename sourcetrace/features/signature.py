"""Classical phase / modulation signature channel: waveform -> 52-d vector.

Faithful port of ``extractor.py`` ``_signature`` (L174-236). It measures phase and
envelope structure directly rather than through the SSL -- phase group-delay kinks,
modulation-envelope ripple, and neural-codec frame-grid periodicity. All STFT parameters,
band edges, epsilons and the final per-clip standardization are preserved EXACTLY; the
numbers must reproduce.

What is measured is the leave-one-out cost of removing the channel, 0.14 FPR95 points
(``results/ablation/no_signature.json`` against ``full.json``). Whether what it measures
overlaps what the SSL retains is untested here.

Layout (52 = 24 + 16 + 12)
--------------------------
* 24  group-delay band means            (``SIG_BANDS``)
* 16  low-freq modulation-spectrum bins  (``SIG_MOD``)
* 12  neural-codec fingerprint           (``SIG_CODEC``): 8 codec-rate band means
      + peakiness + peak location + HF-bandwidth ratio + spectral flatness

Every band split in this file (group-delay bands and codec-rate sub-bands) is a
UNIFORM LINEAR split of the underlying bin axis via ``torch.linspace`` -- these
are linear-frequency bands, NOT mel bands.

Source-line map (champion ``extractor.py``)
------------------------------------------
* STFT (nfft=512, hop=160, hann) ...... L181-183
* group delay = -d(phase)/d(freq) ...... L185-188
* 24 band means over freq, mean/time ... L189-195
* log-energy modulation spectrum ....... L196-200
* codec-rate band (25-95 Hz), 8 means .. L202-216
* token-grid peakiness + location ...... L217-220
* HF (>6 kHz) energy ratio .............. L221-226
* spectral flatness ..................... L227-230
* concat + per-clip standardize ........ L231-236
"""
from __future__ import annotations

import torch

from ..config import FEATURES

# STFT + band constants (frozen; = champion extractor.py module constants).
_NFFT = FEATURES.sig_nfft          # 512
_HOP = FEATURES.sig_hop            # 160
_BANDS = FEATURES.sig_bands        # 24 group-delay bands
_MOD = FEATURES.sig_mod            # 16 low-freq modulation bins
_CODEC = FEATURES.sig_codec        # 12 codec-fingerprint dims (8 bands + 4 scalars)
_SR = FEATURES.sample_rate         # 16000

# Codec-fingerprint geometry (frozen; hard-coded in the champion extractor).
_CODEC_SUBBANDS = 8                # linear sub-bands inside the codec-rate band
_CODEC_RATE_LO_HZ = 25.0           # lowest neural-codec token rate probed
_CODEC_RATE_HI_HZ = 95.0           # highest neural-codec token rate probed
_HF_CUTOFF_HZ = 6000.0             # band-limiting probe for vocoder/codec HF roll-off


def compute_signature(x: torch.Tensor) -> torch.Tensor:
    """Phase-group-delay + modulation + codec-grid signature.

    Faithful, vectorized port of ``extractor.py`` ``_signature``. One GPU STFT
    per batch (no per-frame Python loop).

    Args:
        x: ``(B, T)`` waveform on any device (16 kHz, tile-padded to 4 s).

    Returns:
        ``(B, 52)`` per-clip-standardized signature, same dtype as ``x``.
    """
    win = torch.hann_window(_NFFT, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x, n_fft=_NFFT, hop_length=_HOP, window=win, return_complex=True
    )  # (B, F, T)
    phase = torch.angle(spec)  # (B, F, T)

    # -- group delay ~ -d(phase)/d(freq), wrapped to (-pi, pi] --------------- #
    dphi = phase[:, 1:, :] - phase[:, :-1, :]
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    gd = -dphi  # (B, F-1, T)

    # Time-average, then average over SIG_BANDS uniform (linear-frequency) bands.
    n_gd_bins = gd.shape[1]
    edges = torch.linspace(0, n_gd_bins, _BANDS + 1, device=x.device).long()
    gd_time_mean = gd.mean(dim=2)  # (B, F-1)
    gd_band = torch.stack(
        [gd_time_mean[:, edges[b] : edges[b + 1]].mean(dim=1) for b in range(_BANDS)],
        dim=1,
    )  # (B, SIG_BANDS)

    # -- modulation spectrum of the log-energy envelope --------------------- #
    env = torch.log((spec.abs() ** 2).sum(dim=1) + 1e-8)  # (B, T)
    env = env - env.mean(dim=1, keepdim=True)
    mod_full = torch.fft.rfft(env, dim=1).abs()  # (B, n_mod_bins)
    mod = mod_full[:, :_MOD]  # (B, SIG_MOD) low-freq rhythm

    # -- neural-codec fingerprint (12-d = 8 sub-band means + 4 scalars) ------ #
    n_frames = env.shape[1]  # STFT frames feeding the modulation FFT
    fps = float(_SR) / float(_HOP)  # frames/sec (~100)
    n_mod_bins = mod_full.shape[1]
    # Modulation bin k <-> k * fps / n_frames Hz; codec token rates ~25-95 Hz.
    k_lo = max(1, int(round(_CODEC_RATE_LO_HZ * n_frames / fps)))
    k_hi = min(n_mod_bins, int(round(_CODEC_RATE_HI_HZ * n_frames / fps)))
    # Guarantee at least one bin per sub-band when the band comes out too narrow.
    if k_hi <= k_lo + _CODEC_SUBBANDS:
        k_hi = min(n_mod_bins, k_lo + _CODEC_SUBBANDS + 1)
    codec_band = mod_full[:, k_lo:k_hi]  # (B, W) codec-rate band
    band_w = codec_band.shape[1]
    sub_edges = torch.linspace(0, band_w, _CODEC_SUBBANDS + 1, device=x.device).long()
    sub_means = torch.stack(
        [
            codec_band[:, sub_edges[i] : sub_edges[i + 1]].mean(dim=1)
            for i in range(_CODEC_SUBBANDS)
        ],
        dim=1,
    )  # (B, 8)

    # Token-grid peak: strength (max/mean) + normalized location within the band.
    cmax, cidx = codec_band.max(dim=1)
    cpeak = (cmax / (codec_band.mean(dim=1) + 1e-6)).unsqueeze(1)  # (B, 1)
    cloc = (cidx.float() / float(max(band_w - 1, 1))).unsqueeze(1)  # (B, 1)

    # HF bandwidth cutoff: energy ratio above ~6 kHz (codec/vocoder band-limiting).
    power = spec.abs() ** 2  # (B, F, T)
    n_bins = power.shape[1]
    hf_bin = min(n_bins - 1, int(round(_HF_CUTOFF_HZ / (_SR / 2.0) * (n_bins - 1))))
    hf_ratio = (
        power[:, hf_bin:, :].sum(dim=(1, 2)) / (power.sum(dim=(1, 2)) + 1e-8)
    ).unsqueeze(1)  # (B, 1)

    # Quantization flatness: spectral flatness (geo/arith mean) of mean power spec.
    mean_power = power.mean(dim=2) + 1e-8  # (B, F)
    flat = (
        torch.exp(torch.log(mean_power).mean(dim=1)) / (mean_power.mean(dim=1) + 1e-8)
    ).unsqueeze(1)  # (B, 1)

    codec = torch.cat(
        [sub_means, cpeak, cloc, hf_ratio, flat], dim=1
    )  # (B, SIG_CODEC = 8 + 4 = 12)

    # -- concat + per-clip standardize -------------------------------------- #
    sig = torch.cat([gd_band, mod, codec], dim=1)  # (B, SIG_DIM=52)
    # Per-clip standardize so the scale matches the SSL half.
    return (sig - sig.mean(dim=1, keepdim=True)) / (
        sig.std(dim=1, keepdim=True) + 1e-6
    )
