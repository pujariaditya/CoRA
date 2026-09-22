#!/usr/bin/env python3
"""Download the MLAAD dataset from Hugging Face into ``config.PATHS.mlaad_root``.

MLAAD (``mueller91/MLAAD``) is a large (~242 GB) multi-language audio-deepfake
corpus. This script pulls the HF dataset snapshot into ``ST_MLAAD_ROOT`` via
``huggingface_hub.snapshot_download`` (resumable; already-present files are
skipped). The MLAAD v5 split is defined by the bundled manifests under
``assets/protocols/mlaad/`` and resolves audio paths relative to
this root, so the on-disk layout must match those manifest paths
(``./fake/<lang>/<system>/<file>.wav``).

Because the download is huge, ``--dry-run`` prints the plan (repo, destination,
approximate size, throttling note) without transferring anything.

Rate limits: the HF free tier throttles many-file dataset pulls (~1000 req / 5
min); ``snapshot_download`` retries with backoff, and re-running resumes.

Examples
--------
    python scripts/fetch_mlaad.py --dry-run
    ST_MLAAD_ROOT=/data/MLAAD python scripts/fetch_mlaad.py
    python scripts/fetch_mlaad.py --allow-pattern "fake/en/**"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a script from anywhere (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.config import MLAAD_HF_REPO, PATHS  # noqa: E402
from sourcetrace.runtime import require_packages  # noqa: E402

_APPROX_SIZE = "~242 GB (full corpus)"


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} PiB"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download MLAAD (mueller91/MLAAD) into ST_MLAAD_ROOT.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without downloading")
    parser.add_argument("--allow-pattern", action="append", default=None,
                        metavar="GLOB",
                        help="restrict to matching files (repeatable), e.g. 'fake/en/**'")
    parser.add_argument("--revision", default=None,
                        help="dataset git revision/tag (default: latest)")
    args = parser.parse_args(argv)

    # After parse_args, not at module scope: the real import is deferred to the
    # download call below, so guarding at import time would break --help in exactly
    # the environment the guard exists to explain. This one needs the hub client and
    # not PyTorch -- telling its user to install torch would be actively misleading.
    require_packages("huggingface_hub")

    dest = PATHS.mlaad_root
    print(f"[download_mlaad] repo:        {MLAAD_HF_REPO} (dataset)")
    print(f"[download_mlaad] destination: {dest}")
    print(f"[download_mlaad] size:        {_APPROX_SIZE}")
    print(f"[download_mlaad] patterns:    {args.allow_pattern or 'ALL'}")
    print("[download_mlaad] note: HF free tier throttles many-file pulls "
          "(~1000 req/5min); re-run to resume.")

    if args.dry_run:
        print("[download_mlaad] --dry-run: nothing downloaded.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            repo_id=MLAAD_HF_REPO,
            repo_type="dataset",
            local_dir=str(dest),
            revision=args.revision,
            allow_patterns=args.allow_pattern,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[download_mlaad] FAILED: {exc}", file=sys.stderr)
        print("[download_mlaad] re-run this command to resume the transfer.",
              file=sys.stderr)
        return 1

    size = _dir_size_bytes(Path(local))
    print(f"[download_mlaad] done -> {local}  ({_human(size)})")
    print("[download_mlaad] verify the layout matches the bundled manifests: "
          "./fake/<lang>/<system>/<file>.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
