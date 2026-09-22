#!/usr/bin/env python3
"""Prove the install works, in seconds, without downloading anything.

The real pipeline needs MLAAD v5 (242 GB) and a ~35-40 minute fit before it prints a
number, so a newcomer cannot tell a broken install from a slow one until the end. This
script exercises the same plumbing -- fit, embed, open-set score, save, reload -- on a
tiny synthetic feature matrix, so the answer arrives immediately.

What it checks:

* the package imports and the frozen feature layout is self-consistent,
* ``Method.fit`` converges on separable toy classes,
* ``score_openset`` ranks a held-out cluster below the known ones and ``attribute``
  names the right class,
* a checkpoint round-trips and reproduces its scores after reload.

What it does NOT check: anything about real audio, the WavLM front-end, the codec
residual extractor, or any published number. A passing smoke test means the code and its
dependencies are wired up correctly -- nothing about MLAAD v5 accuracy. Use
``scripts/verify_results.sh`` for that.

Exit status is 0 on success and 1 on the first failed check, so it is usable in CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same as the sibling entry points: make ``sourcetrace`` importable when this file is run
# directly from a clone, before any package import below. Cheap, and it does not pull in
# torch, so ``--help`` still works on an interpreter without it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.runtime import require_torch  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--classes", type=int, default=8,
                   help="synthetic known classes to fit (default: 8)")
    p.add_argument("--per-class", type=int, default=72,
                   help="synthetic clips per class (default: 72); classes x per-class "
                        "must be at least 512")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the synthetic features (default: 0)")
    p.add_argument("--keep", metavar="PATH", default=None,
                   help="write the smoke checkpoint here instead of a temp dir")
    return p


def main() -> int:
    args = _parser().parse_args()

    # Heavy imports live below parse_args so ``--help`` works on an interpreter with no
    # torch -- the invariant tests/test_runtime_guard.py pins for every entry point here.
    # This script fits a head but never touches the WavLM front-end or the codec
    # extractor, so torch alone is the honest requirement, not require_frontend().
    require_torch()

    import tempfile

    import numpy as np

    from sourcetrace import Method
    from sourcetrace.method import EMB_DIM, FEAT_DIM

    # The relative-Mahalanobis density fits a 288 x 288 covariance per class and for the
    # pooled background; a few hundred rows keep those well conditioned. Say so here
    # rather than letting the user discover it part-way through a fit.
    n_rows = args.classes * args.per_class
    if n_rows < 512:
        print(f"--classes x --per-class = {n_rows}; use at least 512 synthetic clips so the "
              f"288-d density fit is well conditioned. Try --classes {args.classes} "
              f"--per-class {-(-512 // args.classes)}.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    print(f"sourcetrace smoke test -- {args.classes} synthetic classes, "
          f"{args.per_class} clips each, {FEAT_DIM}-d features")

    centres = rng.normal(0, 1, (args.classes, FEAT_DIM)).astype("float32")
    X = np.concatenate([
        centres[c] + 0.15 * rng.normal(0, 1, (args.per_class, FEAT_DIM)).astype("float32")
        for c in range(args.classes)
    ])
    y = np.repeat(np.arange(args.classes), args.per_class).astype("int64")

    m = Method(num_known_classes=args.classes, num_languages=1)
    m.fit(X, y, np.zeros_like(y), seed=args.seed)
    check("fit completed", True)

    emb = m.embed(X)
    check("embeddings are finite and 288-d",
          bool(np.all(np.isfinite(emb))) and emb.shape[1] == EMB_DIM, f"shape {emb.shape}")

    known = (centres[0] + 0.15 * rng.normal(0, 1, (20, FEAT_DIM))).astype("float32")
    unseen = (centres.mean(0) + 6.0 * rng.normal(0, 1, (20, FEAT_DIM))).astype("float32")
    s_known, s_unseen = m.score_openset(known), m.score_openset(unseen)
    check("known clips score above unseen ones",
          float(np.mean(s_known)) > float(np.mean(s_unseen)),
          f"{float(np.mean(s_known)):.3f} vs {float(np.mean(s_unseen)):.3f}")
    check("known clips are attributed to their class",
          bool(np.all(m.attribute(known) == 0)))

    with tempfile.TemporaryDirectory() as td:
        dest = Path(args.keep) if args.keep else Path(td) / "smoke.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(dest))
        reloaded = Method.load(str(dest))
        check("checkpoint round-trips",
              bool(np.allclose(s_known, reloaded.score_openset(known),
                               atol=1e-5, rtol=0)),
              f"{dest}" if args.keep else "")

    print("\nSMOKE PASS -- install and plumbing are working." if ok else
          "\nSMOKE FAIL -- see the failed check above.")
    print("This says nothing about MLAAD v5 accuracy; see scripts/verify_results.sh for that.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
