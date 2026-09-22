#!/usr/bin/env python
"""The pooling control: make temporal pooling the independent variable.

RESULT (already run; see ``results/pooling_control_summary.json``): the codec channel's
benefit does not grow when pooling is made lossier -- it shrinks, 1.5543 -> 1.2571 FPR95
points, an interaction of -0.2971. The channel's value is unchanged and real; what the
2x2 establishes is that its evidence is not a substitute for what the pooling summary
discards.

Every ablation varies the ADDED channel while holding pooling fixed, so it can show the
residual is *used* but not how it relates to what pooling discarded. This script varies
pooling itself.

The frozen front-end emits ``[mean-half (1024) | std-half (1024) | sig (52) | codec (33)]``.
The std-half is the temporal-variability statistic: zeroing it marginalises strictly more
temporal structure while leaving the layer band, head architecture, capacity, data, and
every hyperparameter untouched. That is the cleanest available "vary the pooling operator"
control on cached features -- no re-extraction, no config change.

2x2 design (arms A/B already exist in results/ablation/):

    A  mean+std, +codec   full.json      fpr95 1.1429   (2026-08 front end: layers 4-7, fixed projection)
    B  mean+std, -codec   no_codec.json  fpr95 2.6971
    C  mean-only, -codec  <-- measured here
    D  mean-only, +codec  <-- measured here

The design is falsifiable: if the residual substituted for what pooling discards, its
benefit would GROW as pooling gets lossier, i.e.

    (C - D)  >  (B - A) = 1.5543 FPR95 points.

The measured arms were C fpr95 2.3086 and D fpr95 1.0514, so (C - D) = 1.2571 < 1.5543
and the benefit shrinks instead.


What this control does NOT establish: zeroing the std-half removes a pooling *statistic*,
not the temporal axis. That gap has since been closed by a different experiment --
`analysis/pooling_variants.py` re-extracts WavLM and fits an order-SENSITIVE readout
([mean(1st half)|mean(2nd half)]) at matched width and capacity. It is WORSE, fpr95
1.0743 -> 1.3257 (`results/sequence_control_summary.json`), so temporal order is not the
missing evidence either. Claim no more than the two controls jointly support.

Usage (~40 min per arm on one GPU, and it needs the MLAAD v5 feature cache built by
``sourcetrace/extract.py``):

    source .st_env.sh
    python analysis/pooling_control.py --arm mean_only_nocodec
    python analysis/pooling_control.py --arm mean_only_codec

Writes
------
One JSON per arm to ``results/pooling_control_<arm>.json`` unless ``--out`` overrides
it -- e.g. ``results/pooling_control_mean_only_codec.json`` -- each holding the arm's
fpr95, ood_eer, class count and eval size alongside the seeds it was run at. The two
arms committed here plus the two ablation runs they are compared against are summarised
in ``results/pooling_control_summary.json``, which is what the paper's Table 2 quotes.
``results/`` is the measured ground truth for this repository: re-running an arm
overwrites a file the paper cites, so write somewhere else with ``--out`` unless you
mean to replace it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.config import PROTOCOL  # noqa: E402  (pure dataclasses; no torch)

ARMS = {
    "mean_only_nocodec": {"zero_std": True, "codec_weight": "0"},
    "mean_only_codec": {"zero_std": True, "codec_weight": None},
    "full_check": {"zero_std": False, "codec_weight": None},   # harness sanity arm
}


def _parser() -> argparse.ArgumentParser:
    """Build the CLI. Defined before the heavy imports so ``--help`` stays cheap."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=sorted(ARMS),
                    help="which cell of the 2x2 to run; see the table above")
    ap.add_argument("--cache-dir", default=None,
                    help="MLAAD v5 feature cache (default: $ST_FEATURE_CACHE/mlaad_v5)")
    ap.add_argument("--split-seed", type=int, default=PROTOCOL.panda_seed,
                    help="family-holdout split seed (default: 42, the published split)")
    ap.add_argument("--fit-seed", type=int, default=0,
                    help="head/anchor initialisation seed (default: 0, as published)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: results/pooling_control_<arm>.json, "
                         "which OVERWRITES a file the paper cites)")
    return ap


# Answering "what does this script do?" should not require torch, CUDA, or a ten-second
# import. Handle the help request before the imports below rather than after them.
if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    _parser().print_help()
    raise SystemExit(0)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ST_ABL_* knobs are read when sourcetrace.config is IMPORTED, so they must be set before
# any sourcetrace import. Setting them inside main() silently no-ops -- which produced two
# identical arms on the first run of this script. Scan argv here, ahead of the imports.
if "--arm" in sys.argv:
    _arm = sys.argv[sys.argv.index("--arm") + 1]
    if _arm == "mean_only_nocodec":
        os.environ["ST_ABL_CODEC_WEIGHT"] = "0"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.config import FEATURES  # noqa: E402
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

# Slice offsets are derived from config, never hardcoded, so a front-end change
# cannot silently point this control at the wrong columns.
MEAN_HALF = FEATURES.ssl_dim // 2                      # 1024
STD_SLICE = slice(MEAN_HALF, FEATURES.ssl_dim)         # 1024:2048

def main() -> int:
    args = _parser().parse_args()

    arm = ARMS[args.arm]
    # Prove the knob actually took effect rather than trusting that it did.
    from sourcetrace.config import MODEL
    want = 0.0 if arm["codec_weight"] == "0" else 0.5
    if abs(MODEL.codec_weight - want) > 1e-9:
        print(f"FATAL: MODEL.codec_weight={MODEL.codec_weight}, expected {want}. "
              "ST_ABL_CODEC_WEIGHT was not applied before config import.", file=sys.stderr)
        return 2
    print(f"verified MODEL.codec_weight={MODEL.codec_weight}")

    cache = require_feature_cache("mlaad_v5", args.cache_dir)
    split = build_split(seed=args.split_seed)
    n_known = split.n_known_classes

    def prep(table):
        X = np.array(stack_features(table, cache), copy=True)
        if arm["zero_std"]:
            X[:, STD_SLICE] = 0.0
        return X

    train_X, dev_X, eval_X = prep(split.train), prep(split.dev), prep(split.eval)
    print(f"arm={args.arm}  zero_std={arm['zero_std']}  codec_weight={arm['codec_weight']}")
    print(f"train {train_X.shape}  std-half nonzero: {np.count_nonzero(train_X[:, STD_SLICE])}")

    method = Method(num_known_classes=n_known)
    method.fit(train_X, split.train.class_id, split.train.lang_id, seed=args.fit_seed)

    dev_s = -np.asarray(method.score_openset(np.asarray(method.embed(dev_X))),
                        dtype=np.float64)
    eval_s = -np.asarray(method.score_openset(np.asarray(method.embed(eval_X))),
                         dtype=np.float64)

    dev_is_ood = split.dev.class_id >= n_known
    eval_is_ood = split.eval.class_id >= n_known
    fpr95 = fpr95_panda(dev_s[dev_is_ood], eval_s[~eval_is_ood]) * 100.0
    eer, _ = ood_eer(eval_is_ood.astype(np.int64), eval_s)

    payload = {
        "arm": args.arm,
        "split_seed": args.split_seed,
        "fit_seed": args.fit_seed,
        "pooling": "mean-only (std-half zeroed)" if arm["zero_std"] else "mean+std",
        "codec_channel": "off" if arm["codec_weight"] == "0" else "on",
        "_note": "Temporal pooling as the independent variable; layer band, head, "
                 "capacity, data and all hyperparameters identical to the published arms.",
        "results": {"mlaad_v5": {
            "fpr95": float(fpr95),
            "ood_eer": float(eer * 100.0),
            "n_known_classes": int(n_known),
            "n_eval": int(len(split.eval.path)),
        }},
    }
    out = Path(args.out or f"results/pooling_control_{args.arm}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}\n  fpr95={fpr95:.4f}  ood_eer={eer * 100:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
