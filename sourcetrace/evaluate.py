#!/usr/bin/env python3
"""Evaluate the MLAAD-trained :class:`Method` on the MLAAD v5 and/or STOPA protocols.

Loads a fitted :class:`sourcetrace.method.Method` from a ``.pt`` checkpoint (or
fits one on MLAAD v5 on the fly when no checkpoint is given), runs the evaluation
entry points on the cached features, and prints a results table comparing the
headline numbers to the published reference figures:

* MLAAD v5 -> FPR@95 (vs 3.36, Neamtu et al., same corpus and FPR95
  definition -- a like-for-like comparison). OOD-EER and known-model accuracy go
  to ``--json``.
* STOPA    -> held-out synthesis-model EER (vs 16.43, Chhibber et al.), printed as
  **n/c**: the same model is applied to STOPA without any STOPA training, but 16.43
  was measured on a different data split, so the table shows the reference for
  context and refuses to compute a margin (paper Sec. 3.5). The known-model and
  vocoder EERs are printed below the table and written to ``--json``.

One checkpoint serves both tasks: there is no STOPA checkpoint.

* ``--task mlaad_v5`` / ``--task stopa`` / ``--task both``.
* ``--checkpoint PATH`` loads the pre-fit method (:meth:`Method.load`); omit it to
  fit on MLAAD v5 first (the MLAAD feature cache is then required for every task).
* ``--json OUT`` also writes the results as JSON.

The feature cache dirs must hold the ``<group>.npy`` files for the task's splits
(extract them first). They default to ``<ST_FEATURE_CACHE>/<task>``.

Examples
--------
    python -m sourcetrace.evaluate --task both --checkpoint checkpoints/mlaad_v5.pt --json out.json
    python -m sourcetrace.evaluate --task stopa --checkpoint checkpoints/mlaad_v5.pt
    python -m sourcetrace.evaluate --task mlaad_v5     # fit-then-eval (no checkpoint)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sourcetrace.config import (
    CONFIG_NAME,
    PROTOCOL,
    SOTA_MLAAD_V5_FPR95,
    SOTA_STOPA_UNKNOWN_EER,
)
from sourcetrace.runtime import require_feature_cache, require_torch


def _load(checkpoint: str):
    from sourcetrace.method import Method

    print(f"[eval] loading checkpoint {checkpoint}")
    try:
        return Method.load(checkpoint)
    except (FileNotFoundError, IsADirectoryError) as exc:
        # Method.load already explains what is wrong and how to fix it; an entry
        # point should deliver that as a message, not a traceback. Same shape as the
        # missing-feature-cache diagnostic in runtime.require_feature_cache.
        raise SystemExit(
            f"{exc}\n\n"
            "--- sourcetrace: no checkpoint to evaluate -----------------------------\n"
            "Omit --checkpoint to fit on MLAAD v5 and evaluate in one run instead.\n"
            "-----------------------------------------------------------------------"
        ) from None


def _fit_on_mlaad(cache_dir: Path, split_seed: int, fit_seed: int):
    from sourcetrace.datasets.mlaad import build_split
    from sourcetrace.method import Method
    from sourcetrace.tasks.mlaad_v5 import fit_mlaad_v5

    print(f"[eval] no checkpoint: fitting on MLAAD v5 (split seed {split_seed}, "
          f"fit seed {fit_seed}) ...", flush=True)
    return fit_mlaad_v5(Method(), cache_dir, split=build_split(seed=split_seed), seed=fit_seed)


def _run_mlaad(method, cache_dir: Path, split_seed: int) -> dict:
    from sourcetrace.datasets.mlaad import build_split
    from sourcetrace.tasks.mlaad_v5 import eval_mlaad_v5

    # The protocol split uses split_seed (42, eval n=9,620).
    res = eval_mlaad_v5(method, cache_dir, split=build_split(seed=split_seed), seed=split_seed)
    return {
        "fpr95": res.fpr95, "ood_eer": res.ood_eer, "id_acc": res.id_acc,
        "n_known_classes": res.n_known_classes,
        "n_eval": res.n_eval, "n_eval_id": res.n_eval_id, "n_eval_ood": res.n_eval_ood,
    }


def _run_stopa(method, cache_dir: Path) -> dict:
    from sourcetrace.datasets.stopa import build_tables
    from sourcetrace.tasks.stopa import eval_stopa

    res = eval_stopa(method, cache_dir, tables=build_tables())
    return {
        "unknown_atk_eer": res.unknown_atk_eer, "six_eers": res.six_eers,
        "n_tee": res.n_tee, "n_trials": res.n_trials,
    }


def _print_table(rows: list[tuple[str, str, float, float, bool]]) -> None:
    """rows: (task, metric, value, reference, comparable). Percentages.

    ``comparable`` is what stops this table lying. A WIN/DELTA verdict only means
    something when our number and the reference were produced under the same protocol:

    * **MLAAD v5** -- Neamtu et al.'s 3.36 is the same v5 splits and the same FPR95
      definition, so the delta is a margin and WIN is a verdict.
    * **STOPA** -- Chhibber et al.'s 16.43 was measured on a different data split. Both
      systems are applied without STOPA training, but a difference across splits
      confounds method with data, so the paper (Sec. 3.5) draws no conclusion from it
      and neither does this table -- the terminal output is what people screenshot.

    Non-comparable rows therefore print the reference for context and ``n/c`` in place of
    a verdict, with the delta suppressed.
    """
    print()
    print(f"{'TASK':<10} {'METRIC':<22} {'OURS':>8} {'REF':>8} {'DELTA':>8}  RESULT")
    print("-" * 68)
    any_nc = False
    for task, metric, val, ref, comparable in rows:
        if comparable:
            verdict = "WIN " if val <= ref else "----"  # lower is better
            delta = f"{val - ref:>+8.2f}"
        else:
            any_nc, verdict, delta = True, "n/c ", f"{'--':>8}"
        print(f"{task:<10} {metric:<22} {val:>8.2f} {ref:>8.2f} {delta}  {verdict}")
    print("-" * 68)
    print("(lower is better; WIN means at or below the published reference)")
    if any_nc:
        print("(n/c = not comparable: the reference uses a different data split; no margin")
        print(" claimed, see Sec. 3.5 of the paper)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sourcetrace.evaluate",
        description="Evaluate the MLAAD-trained Method on MLAAD v5 / STOPA against the "
                    "published references.")
    # Read at import time by sourcetrace.config, not here -- see the note in that
    # module. Declared so it appears in --help and so a stray value is rejected.
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="hyperparameter overlay, e.g. configs/ablation_no_codec.yaml "
                             "(default: the frozen champion values, == configs/base.yaml)")
    parser.add_argument("--task", required=True, choices=["mlaad_v5", "stopa", "both"])
    parser.add_argument("--checkpoint", default=None,
                        help="the MLAAD-trained checkpoint (serves both tasks); omit it to "
                             "fit on MLAAD v5 first")
    parser.add_argument("--cache-dir-mlaad", default=None,
                        help="MLAAD feature cache (default: <ST_FEATURE_CACHE>/mlaad_v5)")
    parser.add_argument("--cache-dir-stopa", default=None,
                        help="STOPA feature cache (default: <ST_FEATURE_CACHE>/stopa)")
    # Two separate seeds (see sourcetrace/train.py): the protocol split seed (42) and
    # the on-the-fly fit seed (0). With a loaded checkpoint only the split matters.
    parser.add_argument("--split-seed", type=int, default=None,
                        help="protocol split seed (default: PROTOCOL.panda_seed=42, "
                             "eval n=9,620)")
    parser.add_argument("--fit-seed", "--seed", dest="fit_seed", type=int, default=0,
                        help="fit seed for the no-checkpoint fit-then-eval path "
                             "(default: 0 -> the measured MLAAD v5 FPR95 of 0.50%%)")
    parser.add_argument("--json", default=None, help="write results JSON to this path")
    args = parser.parse_args(argv)

    # Diagnose the wrong interpreter before any download or dataset scan: a system
    # python with no torch (or a stub one) otherwise fails much later, from inside a
    # feature extractor, with an error that never mentions the interpreter.
    require_torch()

    split_seed = PROTOCOL.panda_seed if args.split_seed is None else args.split_seed
    fit_seed = args.fit_seed
    results: dict = {}
    table: list[tuple[str, str, float, float, bool]] = []

    # The MLAAD cache is needed for the MLAAD task and for the no-checkpoint fit.
    mlaad_cache = None
    if args.task in ("mlaad_v5", "both") or not args.checkpoint:
        mlaad_cache = require_feature_cache("mlaad_v5", args.cache_dir_mlaad)
        print(f"[eval] MLAAD v5 features: {mlaad_cache}")

    method = (_load(args.checkpoint) if args.checkpoint
              else _fit_on_mlaad(mlaad_cache, split_seed, fit_seed))

    if args.task in ("mlaad_v5", "both"):
        r = _run_mlaad(method, mlaad_cache, split_seed)
        results["mlaad_v5"] = r
        table.append(("mlaad_v5", "FPR@95", r["fpr95"], SOTA_MLAAD_V5_FPR95, True))
        print(f"[eval] MLAAD v5: FPR95 {r['fpr95']:.4f}  OOD-EER {r['ood_eer']:.4f}  "
              f"known-model accuracy {r['id_acc']:.4f}")

    if args.task in ("stopa", "both"):
        cdir = require_feature_cache("stopa", args.cache_dir_stopa)
        print(f"[eval] STOPA features: {cdir}")
        r = _run_stopa(method, cdir)
        results["stopa"] = r
        # not comparable: 16.43 was measured on a different data split (see _print_table)
        table.append(("stopa", "held-out model EER", r["unknown_atk_eer"],
                      SOTA_STOPA_UNKNOWN_EER, False))
        e = r["six_eers"]
        print(f"[eval] STOPA (no STOPA training): held-out models {e['ATK']['unknown']:.4f}  "
              f"known models {e['ATK']['known']:.4f}  vocoders held-out {e['VM']['unknown']:.4f}  "
              f"known {e['VM']['known']:.4f}")

    _print_table(table)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            # `config` records which configs/*.yaml produced these numbers, or
            # null for the frozen defaults.
            json.dump({"split_seed": split_seed, "fit_seed": fit_seed,
                       "config": CONFIG_NAME, "checkpoint": args.checkpoint,
                       "results": results,
                       "sota": {"mlaad_v5_fpr95": SOTA_MLAAD_V5_FPR95,
                                "stopa_unknown_atk_eer": SOTA_STOPA_UNKNOWN_EER}},
                      fh, indent=2)
        print(f"[eval] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
