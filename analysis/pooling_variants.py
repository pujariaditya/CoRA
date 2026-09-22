#!/usr/bin/env python
"""Extract WavLM pooling/layer-band variants for the two front-end read-out controls.

Both controls have been run; the paper reports them in Sec. 3.4 and this script is how
you reproduce the feature blocks they were fit on. What each answers:

  1. Does the SSL layer band matter, i.e. is the shallow 4-7 band a measured choice or an
     inherited one? -> `results/layer_band_summary.json`. It is the largest effect in the
     paper after the codec channel. With the trained residual projection (the front end since
     2026-09-18): FPR95 0.5029 (layers 1-4) -> 0.9143 (4-7) -> 1.5771 (8-11).
  2. Does temporal ORDER carry generator identity? The earlier control
     (`analysis/pooling_control.py`) zeroes the std-half, which removes a pooling
     STATISTIC but leaves the representation order- and shift-invariant, so it cannot
     answer this. `seq2` below can. -> `results/sequence_control_summary.json`: the
     order-sensitive arm is WORSE (1.0743 -> 1.3257), so the answer is no.

Both need the WavLM forward pass re-run, but only that: the signature (52) and codec (33)
channels are pooling-independent and are spliced from the existing cache. One forward pass
yields every hidden state, so all variants below cost a single pass over the corpus.

Variants, all exactly 2048-d so head capacity is identical across arms:

  band_4_7_meanstd  [mean|std] over layers 4-7      <- the pre-2026-09-18 front-end
  band_1_4_meanstd  [mean|std] over layers 1-4      <- the shipped front-end (since 2026-09-18)
  band_8_11_meanstd [mean|std] over layers 8-11     <- deeper band
  seq2_4_7          [mean(first half)|mean(2nd half)] over layers 4-7

`seq2` is the point of the exercise: same width, same capacity, but ORDER-SENSITIVE -
swapping the halves changes it, whereas mean|std is invariant to any permutation of
frames. That is the sequence-preserving readout the moment-pooled arms cannot provide.
It is order-sensitive but coarse: it rules out the specific claim that order-invariance
is what loses generator identity, not finer sequence structure.

Writes one `<variant>__<split>.npy` per variant per split to --out, each a full 2133-d
feature matrix (re-extracted 2048-d SSL block, signature and codec channels spliced from
the shipped cache). Feed them to `analysis/fit_variant.py`. This pass reproduces the
shipped front-end to cosine 0.9997, not bit-exactly, so arms are comparable with each
other and NOT with `results/ablation/full.json`.

Usage:
    python analysis/pooling_variants.py --out "$ST_FEATURE_CACHE/variants"

The variant blocks must NOT land in the shipped feature cache: they are a
different front-end, and an arm read back as if it were the shipped one would
silently compare the wrong thing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# config is pure stdlib and safe before the guard; everything needing torch is deferred
# into main() and guarded AFTER parse_args, so --help works in exactly the broken
# environment the guard exists to explain.
from sourcetrace.config import FEATURES, PATHS, PROTOCOL  # noqa: E402
from sourcetrace.runtime import require_feature_cache, require_frontend  # noqa: E402

VARIANTS = {
    "band_4_7_meanstd":  {"layers": (4, 5, 6, 7),      "mode": "meanstd"},
    "band_1_4_meanstd":  {"layers": (1, 2, 3, 4),      "mode": "meanstd"},
    "band_8_11_meanstd": {"layers": (8, 9, 10, 11),    "mode": "meanstd"},
    "seq2_4_7":          {"layers": (4, 5, 6, 7),      "mode": "seq2"},
}


def summarize(hs, layers, mode):
    """hs: (B, L, T, D) -> (B, 2048). Layer-averaged, matching the shipped pooling order."""
    import torch
    idx = [i for i in layers if i < hs.shape[1]]
    sel = hs[:, idx]                                     # (B, k, T, D)
    if mode == "meanstd":
        parts = [sel.mean(dim=2), sel.std(dim=2, unbiased=False)]
    elif mode == "seq2":
        T = sel.shape[2]
        h = T // 2
        parts = [sel[:, :, :h].mean(dim=2), sel[:, :, h:].mean(dim=2)]
    else:
        raise ValueError(mode)
    return torch.cat(parts, dim=-1).mean(dim=1)          # (B, 2048)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=PROTOCOL.panda_seed)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N clips per split")
    ap.add_argument("--cache-dir", default=None,
                    help="Shipped MLAAD v5 feature cache to splice sig+codec from "
                         "(default: $ST_FEATURE_CACHE/mlaad_v5, else "
                         "$ST_DATA_ROOT/features/mlaad_v5).")
    ap.add_argument("--mlaad-root", default=None,
                    help="MLAAD v5 audio root (default: $ST_MLAAD_ROOT, else "
                         "$ST_DATA_ROOT/MLAAD).")
    args = ap.parse_args()

    require_frontend()
    import soundfile as sf
    import torch

    from sourcetrace.datasets.mlaad import build_split
    from sourcetrace.features.wavlm import WavLMFrontend, tile_pad
    from sourcetrace.tasks._features import stack_features

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Resolve through the same helpers every other entry point uses. Indexing
    # os.environ directly raises a bare KeyError in exactly the DOCUMENTED default case
    # -- Paths falls back to <ST_DATA_ROOT>/features and <ST_DATA_ROOT>/MLAAD -- so a
    # fresh clone following the README got a traceback instead of the diagnosis.
    cache = require_feature_cache("mlaad_v5", args.cache_dir)
    split = build_split(seed=args.split_seed)
    root = Path(args.mlaad_root) if args.mlaad_root else PATHS.mlaad_root
    if not root.is_dir():
        raise SystemExit(
            f"MLAAD audio root not found: {root}\n"
            "Set ST_MLAAD_ROOT (or pass --mlaad-root), or fetch the corpus with\n"
            "    python scripts/fetch_mlaad.py"
        )
    front = WavLMFrontend()
    front._ensure_model()
    sr = FEATURES.sample_rate

    for name, table in (("train", split.train), ("dev", split.dev), ("eval", split.eval)):
        cached = np.asarray(stack_features(table, cache))          # (N, 2133)
        paths = list(table.path)
        n = len(paths) if not args.limit else min(args.limit, len(paths))
        acc = {v: np.zeros((n, FEATURES.ssl_dim), dtype=np.float32) for v in VARIANTS}

        for start in range(0, n, args.batch):
            chunk = paths[start : start + args.batch]
            waves = []
            for p in chunk:
                fp = Path(p)
                if not fp.is_absolute():
                    fp = root / p
                w, fsr = sf.read(str(fp), dtype="float32", always_2d=False)
                if w.ndim > 1:
                    w = w.mean(axis=1)
                if fsr != sr:
                    # Nearest-neighbour resample, kept exactly as the shipped
                    # extractor does it -- the variant arms must differ only in
                    # the pooling operator and the layer band.
                    idx = np.round(np.arange(0, len(w), fsr / sr)).astype(int)
                    w = w[idx.clip(0, len(w) - 1)]
                waves.append(tile_pad(w, sr))
            x = torch.from_numpy(np.stack(waves)).to(front.device)
            with torch.no_grad():
                hsl = front._model(x)["hidden_states"]
                hs = torch.stack(hsl, dim=1)
                for v, spec in VARIANTS.items():
                    acc[v][start : start + len(chunk)] = (
                        summarize(hs, spec["layers"], spec["mode"]).float().cpu().numpy())
            print(f"  {name}: {min(start+args.batch, n)}/{n}", end="\r", flush=True)

        for v in VARIANTS:
            X = cached[:n].copy()
            X[:, : FEATURES.ssl_dim] = acc[v]              # splice: keep sig+codec as measured
            np.save(out / f"{v}__{name}.npy", X)
        print(f"  {name}: wrote {len(VARIANTS)} variants, {n} rows            ")

    print("done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
