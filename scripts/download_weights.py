#!/usr/bin/env python3
"""Fetch the released checkpoint and verify it by sha256.

    python scripts/download_weights.py

One checkpoint is published, ``mlaad_v5.pt``, and it serves both protocols: it is
scored on MLAAD v5 and, without any STOPA training, on STOPA. It is not a standalone
model: a ``torch.save`` dict holding the small trained head plus the fitted scoring
stack (class anchors, relative-Mahalanobis density, z-norm constants, conformal
calibration). The frozen front-ends are not here -- they are fetched by ``setup.sh``
-- and evaluation still reads the extracted feature cache, so downloading saves the
fit and nothing else.

The hash is compiled in rather than fetched alongside the file. A digest served from
the same place as the file it describes attests to nothing, and a wrong checkpoint
here produces a plausible number that is wrong, not an error anyone would notice.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Importable without an install, so a fresh clone can fetch weights before
# `pip install -e .` has run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.runtime import require_packages

REPO = os.environ.get("ST_HF_REPO", "RootAccess4Life/ood-source-tracing")

#: filename, sha256, what it is. Verified by downloading the file back off the Hub,
#: hashing it, loading it with Method.load and re-running the evaluation.
CHECKPOINT = (
    "mlaad_v5.pt",
    "adfbb3533b2114c37839d938cdf0eb70dc4a29c645550372743375c2430d11ce",
    "MLAAD v5, 65 known synthesis models, WavLM layers 1-4, trained residual projection, "
    "one 288-d embedding. Reproduces results/ablation/full.json exactly: FPR95 0.50, "
    "OOD-EER 2.45, known-model accuracy 99.86; applied to STOPA without retraining: "
    "6.80 held-out model EER (results/stopa_measured.json).",
)

#: Retired files, named so a stale copy is recognised rather than puzzled over. None
#: of them loads: Method.load reads only the current checkpoint format.
_KNOWN_BAD = {
    "96c6472d": "the earlier two-channel mlaad_v5.pt (an 800-d embedding with a PCA-whitened "
                "STOPA branch, checkpoint format 1); this code does not load it",
    "9229e0f3": "the retired stopa.pt (a head refitted on STOPA); nothing is trained on "
                "STOPA any more",
    "1bcddc3a": "a pre-refactor mlaad_v5.pt withdrawn on 2026-08-19",
}


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(dest_root):
    """Download the checkpoint and refuse it unless the digest matches."""
    from huggingface_hub import hf_hub_download

    name, expected, note = CHECKPOINT
    print(f"==> {name}")
    print(f"|   {note}")

    cached = hf_hub_download(REPO, name)
    got = sha256_of(cached)
    if got != expected:
        hint = _KNOWN_BAD.get(got[:8])
        raise SystemExit(
            f"sha256 mismatch for {name}\n"
            f"  expected {expected}\n"
            f"  got      {got}\n"
            + (f"  that digest is {hint}.\n" if hint else "")
            + "Refusing to use a checkpoint that is not the released one."
        )
    print(f"|   sha256 ok: {got}")

    os.makedirs(dest_root, exist_ok=True)
    dest = os.path.join(dest_root, name)
    # Copy rather than symlink: checkpoints/ is what --checkpoint points at, and a
    # dangling link into a pruned HF cache is a worse failure than a duplicated file.
    if os.path.realpath(cached) != os.path.realpath(dest):
        import shutil
        shutil.copy(cached, dest)
    print(f"|   {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(prog="python scripts/download_weights.py",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default="checkpoints",
                    help="destination directory (default: checkpoints)")
    ap.add_argument("--verify-only", action="store_true",
                    help="hash the local file against the compiled-in digest and "
                         "download nothing")
    args = ap.parse_args()

    name, expected, _ = CHECKPOINT
    if args.verify_only:
        path = os.path.join(args.dest, name)
        if not os.path.isfile(path):
            print(f"| {name}: absent")
            return 1
        got = sha256_of(path)
        ok = got == expected
        print(f"| {name}: {'ok' if ok else 'MISMATCH'}  {got}")
        if not ok and got[:8] in _KNOWN_BAD:
            print(f"|   that digest is {_KNOWN_BAD[got[:8]]}")
        return 0 if ok else 1

    # Before any network call, and after --help, so a broken interpreter gets one
    # readable line instead of a traceback from inside the hub client.
    require_packages("huggingface_hub")
    fetch(args.dest)

    print()
    print("Next:")
    print("  python -m sourcetrace.evaluate --task both \\")
    print("      --checkpoint checkpoints/mlaad_v5.pt --json runs/my_run.json")
    print("  Expect FPR95 0.50 on MLAAD v5 and 6.80 held-out model EER on STOPA.")
    print("  Evaluation reads the feature caches, which the download does not provide;")
    print("  see the README for building them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
