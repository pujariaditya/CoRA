#!/usr/bin/env python3
"""Download the pretrained frontends (WavLM-Large + EnCodec-24kHz) into the cache.

Populates ``config.PATHS.model_cache`` (``ST_MODEL_CACHE``; ``HF_HOME`` is pointed
there by :mod:`sourcetrace.config` on import) with the two frozen models the
feature extractor needs:

* **WavLM-Large** (``microsoft/wavlm-large``) -- the SSL frontend, fetched here as
  reference weights. Extraction does NOT load this snapshot: ``features/wavlm.py``
  goes through ``s3prl.hub.wavlm_large()``, which pulls its own converted
  checkpoint into s3prl's download directory. Pre-fetching here therefore does not
  make the first extraction run offline.
* **EnCodec-24kHz** (``facebook/encodec_24khz``) -- the neural codec used for the
  residual channel, loaded via ``transformers.EncodecModel``.

Idempotent: ``huggingface_hub.snapshot_download`` skips files already present.
After download it verifies the snapshots exist and prints their on-disk sizes.

Examples
--------
    python scripts/download_models.py
    ST_MODEL_CACHE=/data/models python scripts/download_models.py
    python scripts/download_models.py --wavlm-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a script from anywhere (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.config import ENCODEC_MODEL_ID, PATHS, WAVLM_MODEL_ID  # noqa: E402
from sourcetrace.runtime import require_frontend  # noqa: E402


def _dir_size_bytes(path: Path) -> int:
    """Total size (bytes) of all files under ``path`` (follows symlinks)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _human(n: int) -> str:
    """Human-readable byte count."""
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} PiB"


def download_model(repo_id: str) -> Path:
    """Snapshot-download one HF repo into the model cache; return its local dir."""
    from huggingface_hub import snapshot_download

    print(f"[download_models] fetching {repo_id} ...", flush=True)
    local = snapshot_download(repo_id=repo_id)  # honors HF_HOME (set by config)
    return Path(local)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download WavLM-Large + EnCodec-24kHz into the model cache.")
    parser.add_argument("--wavlm-only", action="store_true",
                        help="download only WavLM-Large")
    parser.add_argument("--encodec-only", action="store_true",
                        help="download only EnCodec-24kHz")
    args = parser.parse_args(argv)

    # Diagnose the environment before any download or dataset scan: a system python
    # with no torch (or a stub one), or a half-installed env, otherwise fails much
    # later from inside the extractor with an error that names neither.
    require_frontend()

    print(f"[download_models] model cache: {PATHS.model_cache}")
    PATHS.model_cache.mkdir(parents=True, exist_ok=True)

    targets: list[str] = []
    if not args.encodec_only:
        targets.append(WAVLM_MODEL_ID)
    if not args.wavlm_only:
        targets.append(ENCODEC_MODEL_ID)

    ok = True
    for repo in targets:
        try:
            local = download_model(repo)
            size = _dir_size_bytes(local)
            print(f"[download_models]   OK {repo} -> {local}  ({_human(size)})")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[download_models]   FAILED {repo}: {exc}", file=sys.stderr)

    print("[download_models] done." if ok else "[download_models] finished with errors.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
