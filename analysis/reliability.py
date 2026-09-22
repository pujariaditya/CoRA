#!/usr/bin/env python
"""Dump the calibration evidence the open-set score already produces but discards.

``ConformalCalibration.fit`` measures the finite-sample coverage of its rank p-value
at alpha in {0.01, 0.05, 0.10} and stores it purely "for reporting only -- nothing
reads it back" (see ``sourcetrace/scoring/conformal.py``). That measurement is the
paper's strongest under-reported result: none of the comparable open-set source-tracing
systems report a coverage guarantee at all, only discrimination metrics.

This script fits the MLAAD v5 model under the same protocol as the ablation runs
(split seed 42, fit seed 0) and writes ``results/reliability.json`` containing:

  * measured vs nominal coverage per alpha,
  * binned ID / OOD open-set score histograms and the FPR95 operating threshold,
  * the headline metrics, so the file can be checked against the ablation results.

Usage (roughly 35-40 min on one GPU, and it needs the MLAAD v5 feature cache built by
``sourcetrace/extract.py``):

    source .st_env.sh
    python analysis/reliability.py

Writes
------
``results/reliability.json`` unless ``--out`` overrides it. That path is the file the
paper's Section 3.5 quotes, so re-running overwrites measured ground truth; pass
``--out`` to write elsewhere unless you mean to replace it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.config import PROTOCOL  # noqa: E402  (pure dataclasses; no torch)


def _parser() -> argparse.ArgumentParser:
    """Build the CLI. Defined before the heavy imports so ``--help`` stays cheap."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-dir", default=None,
                    help="MLAAD v5 feature cache (default: $ST_FEATURE_CACHE/mlaad_v5)")
    ap.add_argument("--split-seed", type=int, default=PROTOCOL.panda_seed,
                    help="family-holdout split seed (default: 42, the published split)")
    ap.add_argument("--fit-seed", type=int, default=0,
                    help="head/anchor initialisation seed (default: 0, as published)")
    ap.add_argument("--out", default="results/reliability.json",
                    help="output JSON (default: results/reliability.json, which "
                         "OVERWRITES the file Section 3.5 of the paper quotes)")
    ap.add_argument("--bins", type=int, default=60,
                    help="histogram bins for the ID/OOD score distributions (default: 60)")
    return ap


# Answering "what does this script do?" should not require torch, CUDA, or a ten-second
# import. Handle the help request before the imports below rather than after them.
if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    _parser().print_help()
    raise SystemExit(0)

# Determinism: must be set before any CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.runtime import require_feature_cache, require_frontend  # noqa: E402

# The imports below build the frozen front-end, so the environment is diagnosed
# here rather than at first use: under a wrong interpreter or a half-installed env
# they fail as an ImportError from inside the extractor, which names neither. These
# two scripts are run rarely enough that hitting that cold is exactly when a clear
# message matters most.
require_frontend()
from sourcetrace.datasets.mlaad import build_split  # noqa: E402
from sourcetrace.method import Method  # noqa: E402
from sourcetrace.metrics.protocol import fpr95_panda, ood_eer  # noqa: E402
from sourcetrace.tasks._features import stack_features  # noqa: E402

NOMINAL_ALPHAS = (0.01, 0.05, 0.10)


def main() -> int:
    args = _parser().parse_args()

    cache = require_feature_cache("mlaad_v5", args.cache_dir)
    if not cache.is_dir():
        print(f"feature cache not found: {cache}", file=sys.stderr)
        return 1

    split = build_split(seed=args.split_seed)
    n_known = split.n_known_classes
    print(f"split seed {args.split_seed}: {n_known} known classes, eval n={len(split.eval.path)}")

    train_X = stack_features(split.train, cache)
    dev_X = stack_features(split.dev, cache)
    eval_X = stack_features(split.eval, cache)

    method = Method(num_known_classes=n_known)
    print(f"fitting (seed {args.fit_seed}) -- this is the slow part ...")
    method.fit(train_X, split.train.class_id, split.train.lang_id, seed=args.fit_seed)

    cov = method._calibration.coverage
    print("measured coverage:", cov)

    dev_emb = np.asarray(method.embed(dev_X))
    eval_emb = np.asarray(method.embed(eval_X))

    dev_ood = -np.asarray(method.score_openset(dev_emb), dtype=np.float64)
    eval_ood = -np.asarray(method.score_openset(eval_emb), dtype=np.float64)
    dev_is_ood = split.dev.class_id >= n_known
    eval_is_ood = split.eval.class_id >= n_known
    eval_is_id = ~eval_is_ood

    fpr95 = fpr95_panda(dev_ood[dev_is_ood], eval_ood[eval_is_id]) * 100.0
    eer, _ = ood_eer(eval_is_ood.astype(np.int64), eval_ood)

    # The FPR95 operating point: the order statistic the metric actually uses.
    cal = np.sort(dev_ood[dev_is_ood])
    thr = float(cal[int(np.clip(int(len(cal) * 0.05), 0, len(cal) - 1))])

    lo, hi = float(eval_ood.min()), float(eval_ood.max())
    edges = np.linspace(lo, hi, args.bins + 1)
    id_hist, _ = np.histogram(eval_ood[eval_is_id], bins=edges)
    ood_hist, _ = np.histogram(eval_ood[eval_is_ood], bins=edges)

    payload = {
        "split_seed": args.split_seed,
        "fit_seed": args.fit_seed,
        "protocol": "MLAAD v5 family-level open-set",
        "coverage": {
            "_note": "Measured finite-sample coverage (%) of the split-conformal rank "
                     "p-value on held-out calibration data, versus the nominal 100*(1-alpha).",
            "nominal_pct": {str(a): round(100.0 * (1.0 - a), 2) for a in NOMINAL_ALPHAS},
            "measured_pct": {str(a): round(float(cov[a]), 4) for a in sorted(cov)},
        },
        "open_set_scores": {
            "_note": "Eval-split open-set scores, OOD-positive (higher = more OOD). "
                     "Binned so the figure can be drawn without shipping per-sample data.",
            "bin_edges": [round(float(e), 6) for e in edges],
            "id_counts": [int(v) for v in id_hist],
            "ood_counts": [int(v) for v in ood_hist],
            "fpr95_threshold": round(thr, 6),
            "n_eval_id": int(eval_is_id.sum()),
            "n_eval_ood": int(eval_is_ood.sum()),
        },
        "results": {
            "mlaad_v5": {
                "fpr95": float(fpr95),
                "ood_eer": float(eer * 100.0),
                "n_known_classes": int(n_known),
                "n_eval": int(len(split.eval.path)),
            }
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    print(f"sanity: fpr95={fpr95:.4f} (ablation full.json reports 0.5029), "
          f"ood_eer={eer * 100:.4f} (2.4457)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
