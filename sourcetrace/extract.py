#!/usr/bin/env python3
"""Extract frozen-SSL features for a dataset and write the per-group feature cache.

For the chosen dataset this script:

1. builds the protocol split/tables (:func:`sourcetrace.datasets.mlaad.build_split` or
   :func:`sourcetrace.datasets.stopa.build_tables`),
2. groups every wav by its per-row cache key (``table.group`` -- a ``(split,
   system)`` for MLAAD or ``(split, attack)`` for STOPA), and
3. runs the :class:`sourcetrace.features.Extractor` over each group's wavs,
   writing one ``<group>.npy`` (shape ``(n_group, 2133)``) into the feature-cache
   directory.

The resulting directory is exactly the ``FeatureSource`` the eval protocols expect
(``sourcetrace.tasks._features.stack_features`` loads ``<group>.npy`` per row).

Row order within each ``.npy`` matches the table's per-group enumeration order (the
split builders enumerate group-by-group deterministically and each group's wavs in
table order), so features stay row-aligned with the labels at eval time.

Resumable: a group whose ``.npy`` already exists (with the right row count) is
skipped. Idempotent across re-runs.

The cache directory defaults to ``<ST_FEATURE_CACHE>/<dataset>``. Point the eval /
train scripts at the SAME directory (they default to the same location).

Examples
--------
    python -m sourcetrace.extract --dataset stopa
    python -m sourcetrace.extract --dataset mlaad_v5 --cache-dir /data/feat/mlaad
    ST_CPU=1 python -m sourcetrace.extract --dataset stopa --limit-groups 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sourcetrace.config import FEATURES, PATHS, STOPA_ZENODO_DOI
from sourcetrace.runtime import require_frontend


def _default_cache_dir(dataset: str, override: str | None) -> Path:
    return Path(override) if override else (PATHS.feature_cache / dataset)


def _collect_groups(dataset: str, seed: int) -> dict[str, list[str]]:
    """Map each cache-group key -> ordered list of wav paths (table order)."""
    groups: dict[str, list[str]] = {}

    def _add(table) -> None:
        # strict: group and path are parallel columns of one table, so a length
        # mismatch is a corrupt table, not something to silently truncate away.
        for g, p in zip(table.group, table.path, strict=True):
            groups.setdefault(g, []).append(p)

    if dataset == "mlaad_v5":
        from sourcetrace.datasets.mlaad import build_split

        split = build_split(seed=seed)
        _add(split.train)
        _add(split.dev)
        _add(split.eval)
    elif dataset == "stopa":
        from sourcetrace.datasets.stopa import build_tables

        # No STOPA utterance is used for training, so the EET partition is never read:
        # only the enrolment (TEE) and probe (Trials) groups are extracted.
        tables = build_tables()
        _add(tables.tee)
        _add(tables.trials)
    else:  # pragma: no cover - argparse guards this
        raise ValueError(dataset)
    return groups


def _missing_audio_sample(
    groups: dict[str, list[str]], keys: list[str], probe: int = 12
) -> list[str]:
    """Return up to ``probe`` manifest paths that are not on disk.

    Checked before the frontend models are loaded, because the alternative is what
    this script used to do: spend a minute building WavLM and EnCodec, then surface
    ``Error opening '...': System error`` from ``soundfile`` on the first clip. That
    message names neither the cause (the dataset was never downloaded) nor the fix.

    Only a sample is probed -- ``os.stat`` on 38k files is slow enough to be its own
    annoyance, and a dataset that is missing is missing at the first clip of every
    group. Probing across *different* groups rather than the first N of one group is
    deliberate: a partially-synced corpus typically loses whole systems.
    """
    missing: list[str] = []
    for key in keys[:probe]:
        wavs = groups.get(key) or []
        if wavs and not Path(wavs[0]).is_file():
            missing.append(wavs[0])
    return missing


def _report_missing_audio(dataset: str, missing: list[str], n_wavs: int) -> None:
    """Explain a missing corpus in terms of the two things the user can change."""
    root_var, root, fetch, layout = {
        "mlaad_v5": ("ST_MLAAD_ROOT", PATHS.mlaad_root,
                     "python scripts/fetch_mlaad.py",
                     "./fake/<language>/<system>/<clip>.wav"),
        "stopa": ("ST_STOPA_ROOT", PATHS.stopa_root,
                  f"download STOPA from Zenodo (DOI {STOPA_ZENODO_DOI}) and unpack it there",
                  "./<attack>/<clip>.wav"),
    }[dataset]
    if missing:
        found = (f"  {len(missing)} of the first files probed are missing, e.g.\n"
                 f"    {missing[0]}\n\n"
                 f"  The split manifests reference {n_wavs} clips relative to\n")
    else:
        found = ("  The split builder found no audio at all, so there is nothing to\n"
                 "  extract and every later step would fail further from the cause.\n\n"
                 "  It looks for audio under\n")
    print(
        f"\n[extract] ERROR: the {dataset} audio is not where this script expects it.\n\n"
        f"{found}"
        f"    {root_var} = {root}\n"
        f"\n  Either the corpus has not been downloaded yet:\n"
        f"    {fetch}\n"
        f"  or it lives elsewhere, in which case point {root_var} at it:\n"
        f"    export {root_var}=/path/to/{dataset}\n"
        f"\n  The expected layout under that root is\n"
        f"    {layout}\n",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sourcetrace.extract",
        description="Extract frozen-SSL features into the per-group feature cache.")
    # Read at import time by sourcetrace.config, not here -- see the note in that
    # module. Declared so it appears in --help and so a stray value is rejected.
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="hyperparameter overlay, e.g. configs/ablation_no_codec.yaml "
                             "(default: the frozen champion values, == configs/base.yaml)")
    parser.add_argument("--dataset", required=True, choices=["mlaad_v5", "stopa"],
                        help="which protocol's wavs to extract")
    parser.add_argument("--cache-dir", default=None,
                        help="feature-cache dir (default: <ST_FEATURE_CACHE>/<dataset>)")
    parser.add_argument("--seed", type=int, default=None,
                        help="split seed (default: PROTOCOL.panda_seed=42)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: config.device())")
    parser.add_argument("--limit-groups", type=int, default=None,
                        help="only process the first N groups (debug/smoke)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract even if a group's .npy already exists")
    args = parser.parse_args(argv)

    # Diagnose the environment before any download or dataset scan: a system python
    # with no torch (or a stub one), or a half-installed env, otherwise fails much
    # later from inside the extractor with an error that names neither.
    require_frontend()

    from sourcetrace.config import PROTOCOL

    seed = PROTOCOL.panda_seed if args.seed is None else args.seed
    cache_dir = _default_cache_dir(args.dataset, args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] dataset={args.dataset} seed={seed}")
    print(f"[extract] cache dir: {cache_dir}")
    print("[extract] building split/tables ...", flush=True)
    groups = _collect_groups(args.dataset, seed)
    keys = sorted(groups)
    if args.limit_groups is not None:
        keys = keys[: args.limit_groups]
    n_wavs = sum(len(groups[k]) for k in keys)
    print(f"[extract] {len(keys)} groups, {n_wavs} wavs total.")

    # Two distinct ways an absent corpus shows up, and both used to pass silently.
    # MLAAD ships file manifests, so the split still enumerates 38k paths that are
    # not on disk. STOPA's tables are built by scanning the audio root, so an absent
    # corpus yields ZERO groups and the loop below then "succeeds" having written
    # nothing -- the worst outcome, because the next script fails somewhere further
    # away from the cause.
    if not keys:
        _report_missing_audio(args.dataset, [], 0)
        return 1
    missing = _missing_audio_sample(groups, keys)
    if missing:
        _report_missing_audio(args.dataset, missing, n_wavs)
        return 1

    # Build the extractor lazily (loads WavLM + EnCodec on first embed).
    from sourcetrace.features import Extractor

    extractor: Extractor | None = None

    done = skipped = 0
    for i, g in enumerate(keys, 1):
        out = cache_dir / f"{g}.npy"
        wavs = groups[g]
        if out.exists() and not args.force:
            try:
                arr = np.load(out, mmap_mode="r")
                if arr.shape[0] == len(wavs) and arr.shape[1] == FEATURES.feat_dim:
                    skipped += 1
                    print(f"[extract] ({i}/{len(keys)}) skip {g} "
                          f"(cached {arr.shape[0]}x{arr.shape[1]})")
                    continue
            except Exception:  # noqa: BLE001 -- corrupt cache, re-extract
                pass
        if extractor is None:
            print("[extract] loading frontend models ...", flush=True)
            extractor = Extractor(device=args.device)
        print(f"[extract] ({i}/{len(keys)}) {g}: {len(wavs)} wavs ...", flush=True)
        try:
            # Write directly to <cache_dir>/<group>.npy via the extractor's own
            # keyed on-disk cache (no redundant content-hash copy). --force paths
            # here have no stale file because embed_cached only reads an existing
            # .npy; we removed it below when forcing.
            if args.force and out.exists():
                out.unlink()
            feats = extractor.embed_files(wavs, key=g, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[extract]   FAILED {g}: {exc}", file=sys.stderr)
            return 1
        # ensure float32 on disk (embed_files already returns float32).
        if feats.dtype != np.float32:
            np.save(out, feats.astype(np.float32))
        done += 1

    print(f"[extract] done. extracted={done} skipped={skipped} -> {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
