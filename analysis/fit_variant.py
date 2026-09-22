#!/usr/bin/env python
"""Fit + evaluate one pooling/layer-band variant under the frozen protocol.

Consumes the .npy blocks written by analysis/pooling_variants.py. Every arm shares the
same head, capacity, split, seeds and hyperparameters; only the 2048-d SSL block differs,
so any change in FPR95 is attributable to the pooling operator or layer band.

The matched baseline is `band_4_7_meanstd`, re-extracted through the same pipeline as the
other arms. It is NOT compared against results/ablation/full.json: this re-extraction
reproduces the shipped front-end to cosine 0.9997, not bit-exactly, so cross-pipeline
comparison would confound extraction drift with the effect under test.

Writes one JSON file per arm (default `results/variant_<variant>.json`, override with
`--out`)::

    {"variant": ..., "split_seed": 42, "fit_seed": 0, "_note": ...,
     "results": {"mlaad_v5": {"fpr95": <percent>, "ood_eer": <percent>,
                              "n_known_classes": <int>, "n_eval": <int>}}}

and prints the two metrics to stdout. Note `results/` in this repository is read-only
ground truth for the paper; point `--out` somewhere else when experimenting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# sourcetrace.config is pure stdlib, so it is safe to import before the guard and lets
# --help print the real protocol seed. Everything that needs torch is deferred into
# main() and guarded AFTER parse_args, per the convention
# tests/test_runtime_guard.py::test_help_still_works_on_every_python_entry_point pins:
# a guard that eats --help tells the user to install a package with a command that no
# longer runs.
from sourcetrace.config import PROTOCOL  # noqa: E402
from sourcetrace.runtime import require_frontend  # noqa: E402


def _default_variant_dir() -> str:
    """Where pooling_variants.py writes, derived from the same env vars as everything else.

    Hard-coding an absolute path here would contradict the README's promise that paths are
    environment-driven, and would send a stranger's run to a directory that exists only on
    the machine this was developed on.
    """
    explicit = os.environ.get("ST_VARIANT_CACHE")
    if explicit:
        return explicit
    cache = os.environ.get("ST_FEATURE_CACHE")
    return str(Path(cache) / "variants") if cache else "variants"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variant", required=True,
        help="Variant name as written by analysis/pooling_variants.py, e.g. "
             "band_4_7_meanstd (the matched baseline), band_1_4_meanstd, "
             "band_8_11_meanstd, seq2_4_7. Reads <dir>/<variant>__{train,dev,eval}.npy.")
    ap.add_argument(
        "--dir", default=_default_variant_dir(),
        help="Directory holding the .npy feature blocks from pooling_variants.py. "
             "Defaults to $ST_VARIANT_CACHE, else $ST_FEATURE_CACHE/variants "
             "(currently: %(default)s).")
    ap.add_argument(
        "--split-seed", type=int, default=PROTOCOL.panda_seed,
        help="Family-holdout split seed; must match the extraction run "
             "(default: %(default)s, the frozen protocol seed).")
    ap.add_argument(
        "--fit-seed", type=int, default=0,
        help="Head/anchor initialisation seed (default: %(default)s).")
    ap.add_argument(
        "--out", default=None,
        help="Where to write the result JSON (default: results/variant_<variant>.json). "
             "The file records variant, split_seed, fit_seed and "
             "results.mlaad_v5.{fpr95, ood_eer, n_known_classes, n_eval}, with fpr95 and "
             "ood_eer as percentages.")
    a = ap.parse_args()

    require_frontend()
    from sourcetrace.datasets.mlaad import build_split
    from sourcetrace.method import Method
    from sourcetrace.metrics.protocol import fpr95_panda, ood_eer

    d = Path(a.dir)
    split = build_split(seed=a.split_seed)
    n_known = split.n_known_classes
    X = {s: np.load(d / f"{a.variant}__{s}.npy") for s in ("train", "dev", "eval")}
    print(f"{a.variant}: train {X['train'].shape}")

    m = Method(num_known_classes=n_known)
    m.fit(X["train"], split.train.class_id, split.train.lang_id, seed=a.fit_seed)
    dev_s = -np.asarray(m.score_openset(np.asarray(m.embed(X["dev"]))), dtype=np.float64)
    ev_s = -np.asarray(m.score_openset(np.asarray(m.embed(X["eval"]))), dtype=np.float64)
    dv = split.dev.class_id >= n_known
    ev = split.eval.class_id >= n_known
    fpr95 = fpr95_panda(dev_s[dv], ev_s[~ev]) * 100.0
    eer, _ = ood_eer(ev.astype(np.int64), ev_s)

    out = Path(a.out or f"results/variant_{a.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": a.variant, "split_seed": a.split_seed, "fit_seed": a.fit_seed,
        "_note": "Re-extracted WavLM block; sig+codec spliced from the shipped cache. "
                 "Compare these arms only against each other, never against results/ablation/.",
        "results": {"mlaad_v5": {"fpr95": float(fpr95), "ood_eer": float(eer * 100.0),
                                  "n_known_classes": int(n_known),
                                  "n_eval": int(len(split.eval.path))}}}, indent=1))
    print(f"wrote {out}\n  fpr95={fpr95:.4f}  ood_eer={eer*100:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
