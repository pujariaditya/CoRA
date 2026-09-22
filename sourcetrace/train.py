#!/usr/bin/env python3
"""Train (fit) the source-tracing :class:`Method` on MLAAD v5 and checkpoint it.

Loads the cached per-group features (produced by ``sourcetrace/extract.py``),
fits a :class:`sourcetrace.method.Method` on the *train* split (ID-only rows,
labels = architecture family) via :func:`sourcetrace.tasks.mlaad_v5.fit_mlaad_v5`,
then serializes ALL fitted state to a ``.pt`` checkpoint with :meth:`Method.save`.

That one checkpoint serves both protocols: ``python -m sourcetrace.evaluate`` scores
it on MLAAD v5 and, without any STOPA training, on STOPA (paper Sec. 3.5).

Only the train split is touched (leak-safe); dev/eval never enter ``fit``. Fitting
is deterministic per seed. The **protocol split seed** (``--split-seed``, default
``PROTOCOL.panda_seed=42``, eval n=9,620) and the **head-fit RNG seed**
(``--fit-seed``/``--seed``, default ``0``) are separate. The defaults (split seed 42,
fit seed 0) are the ones every file in ``results/`` was measured with, and reproduce
MLAAD v5 FPR95 = 0.50% (``results/ablation/full.json``).

The feature cache dir must contain the ``<group>.npy`` files for the train split
(extract them first). It defaults to ``<ST_FEATURE_CACHE>/mlaad_v5``, matching
``python -m sourcetrace.extract --dataset mlaad_v5``.

Example
-------
    python -m sourcetrace.train --checkpoint checkpoints/mlaad_v5.pt   # fit seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sourcetrace.config import PROTOCOL
from sourcetrace.runtime import require_feature_cache, require_torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sourcetrace.train",
        description="Fit the source-tracing Method on the MLAAD v5 train split and save it.")
    # Read at import time by sourcetrace.config, not here -- see the note in that
    # module. Declared so it appears in --help and so a stray value is rejected.
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="hyperparameter overlay, e.g. configs/ablation_no_codec.yaml "
                             "(default: the frozen champion values, == configs/base.yaml)")
    parser.add_argument("--task", default="mlaad_v5", choices=["mlaad_v5"],
                        help="the training protocol; MLAAD v5 is the only one "
                             "(nothing is trained on STOPA)")
    parser.add_argument("--checkpoint", required=True,
                        help="output .pt checkpoint path")
    parser.add_argument("--cache-dir", default=None,
                        help="feature-cache dir (default: <ST_FEATURE_CACHE>/mlaad_v5)")
    # NOTE: the split seed and the fit seed are DELIBERATELY separate. The
    # champion (and the shipped checkpoint) reproduces at the protocol split seed
    # 42 (eval n=9,620) with head-fit RNG seed 0. Fusing them (e.g. a single
    # --seed 0) would build a different, non-protocol split (eval n=6,349) and
    # NOT reproduce the headline number. `--seed` sets the FIT seed only, aliased
    # to --fit-seed, so the historical `--seed 0` recipe still does the right thing.
    parser.add_argument("--fit-seed", "--seed", dest="fit_seed", type=int, default=0,
                        help="head-fit RNG seed (default: 0, which with split seed "
                             "42 reproduces the measured MLAAD v5 FPR95 of 0.50%%)")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="OOD-family permutation / protocol-split seed "
                             "(default: PROTOCOL.panda_seed=42, eval n=9,620)")
    args = parser.parse_args(argv)

    # Diagnose the wrong interpreter before any download or dataset scan: a system
    # python with no torch (or a stub one) otherwise fails much later, from inside a
    # feature extractor, with an error that never mentions the interpreter.
    require_torch()

    split_seed = PROTOCOL.panda_seed if args.split_seed is None else args.split_seed
    fit_seed = args.fit_seed

    cache_dir = require_feature_cache(args.task, args.cache_dir)

    from sourcetrace.datasets.mlaad import build_split
    from sourcetrace.method import EMB_DIM, Method
    from sourcetrace.tasks.mlaad_v5 import fit_mlaad_v5

    print(f"[train] task={args.task} split_seed={split_seed} fit_seed={fit_seed}")
    print(f"[train] features: {cache_dir}")

    split = build_split(seed=split_seed)
    print(f"[train] MLAAD v5 split: K={split.n_known_classes} known families, "
          f"train n={len(split.train)}")
    print("[train] fitting (300 epochs) ...", flush=True)
    method = Method()
    # split built at split_seed; method.fit uses fit_seed.
    fit_mlaad_v5(method, cache_dir, split=split, seed=fit_seed)

    out = method.save(args.checkpoint)
    size_mb = Path(out).stat().st_size / (1024 * 1024)
    print(f"[train] fitted C={method._C} classes; embedding dim={EMB_DIM}")
    print(f"[train] K={split.n_known_classes} known families, train rows={len(split.train)}")
    print(f"[train] checkpoint saved -> {out}  ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
