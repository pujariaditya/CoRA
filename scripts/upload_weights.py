#!/usr/bin/env python3
"""Publish the fitted head to Hugging Face, refresh the model card, retire old files.

    HF_TOKEN=... python scripts/upload_weights.py --mlaad_v5 checkpoints/mlaad_v5.pt
    HF_TOKEN=... python scripts/upload_weights.py --card-only
    HF_TOKEN=... python scripts/upload_weights.py --delete-retired

Maintainer tool; end users want ``download_weights.py``.

THE TOKEN IS READ FROM THE ENVIRONMENT ONLY. It is never written to a config, a
committed file, or the upload itself. ``huggingface-cli login`` would persist it
to ``~/.cache/huggingface/token``, which is world-readable by default -- pass it
per-invocation instead.

The file is hashed before upload and the digest printed. That is the value that
belongs in ``download_weights.py``; the point of pinning it there is that a reader
verifies against a constant in the source, never against a digest served from the
same place as the file.

The card lives here, as a constant, rather than being edited on the Hub, so that a
change to it shows up in a diff.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.runtime import require_packages

REPO = os.environ.get("ST_HF_REPO", "RootAccess4Life/ood-source-tracing")

#: Files of the earlier design that --delete-retired removes from the Hub: the STOPA
#: refit (nothing is trained on STOPA any more).
RETIRED = ("stopa.pt",)

CARD = """---
license: mit
library_name: pytorch
pipeline_tag: audio-classification
tags:
  - audio
  - audio-classification
  - deepfake-detection
  - source-tracing
  - open-set-recognition
  - wavlm
  - encodec
datasets:
  - mueller91/MLAAD
metrics:
  - accuracy
---

# sourcetrace — the fitted head of CoRA (open-set audio deepfake attribution)

The fitted head of [`sourcetrace`](https://github.com/pujariaditya/CoRA),
which names **which synthesis model produced a synthetic speech clip** — or reports
that the model is not one it has seen. It is the artifact behind the paper *CoRA:
Robust Open-Set Audio Deepfake Attribution with Neural Codec Residuals*.

**This is not a standalone model.** `mlaad_v5.pt` is a `torch.save` dict with
`format: "sourcetrace-method-checkpoint"`, loaded by `sourcetrace.method.Method.load`.
It holds the small trained head plus the fitted scoring stack (class anchors,
relative-Mahalanobis density, z-norm constants, conformal calibration). The
front-ends — `microsoft/wavlm-large` and `facebook/encodec_24khz` — are frozen, are
**not** included here, and are fetched separately by `./setup.sh`.

## Model

| | |
|---|---|
| Input | 2133-d feature vector, **not** audio |
| SSL front end | `microsoft/wavlm-large`, **frozen**, layers 1–4, mean\\|std pooled → 2048-d |
| Spectral signature | 24 group-delay bands + 16 modulation bins + 12 codec-grid dims → 52-d |
| Codec residual | `facebook/encodec_24khz` reconstruction residual at 1.5 / 6 / 12 kbps → 33-d |
| Head | factorized gated: SSL 192 + spectral 64 + codec 32, tanh FiLM gates |
| Embedding | **one** 288-d vector `z = L2([z_ssl ‖ 0.1 z_spec ‖ 0.5 z_codec])` |
| Open-set score | anchor margin + relative Mahalanobis on `z`; label = nearest anchor |
| Training | MLAAD v5 only: Adam, lr 3e-4, no weight decay, 300 epochs, batch 512 |

The same `z` serves every task: the open-set score, the known-model label, and the
STOPA cross-corpus protocol (cosine to enrolment fingerprints), where the model is
applied **without any STOPA training**.

The codec residual is the contribution: re-encode each clip through EnCodec-24 kHz
and keep what it got *wrong*. Those 33 numbers come off the waveform the speech
model never sees. Removing the channel moves FPR95 from 0.50 % to 1.28 %, OOD-EER
from 2.45 % to 3.45 % and known-model accuracy from 99.86 % to 99.68 % on MLAAD v5,
and the STOPA held-out model EER from 6.80 % to 8.08 %.

The head in the public code is a reconstruction from the published method
description, not a recovered original; its construction order is load-bearing for
RNG reproducibility.

## Input

Not audio. A **2133-d** feature vector per clip, laid out as
`[ SSL 0:2048 | signature 2048:2100 | codec residual 2100:2133 ]`, produced by
`python -m sourcetrace.extract`. There is no way to run these weights without the
repository and an extracted feature cache.

## Files

| file | what it is |
|---|---|
| `mlaad_v5.pt` | the fitted head, 65 known synthesis models, split seed 42 / fit seed 0 |

It is sha256-verified on download against the digest compiled into
`scripts/download_weights.py`, not served from here. Earlier files of this
repository (an 800-d two-channel `mlaad_v5.pt` and a STOPA refit `stopa.pt`) are
retired: the current code does not load them.

## Use

```bash
git clone https://github.com/pujariaditya/CoRA && cd CoRA
pip install -e . && ./setup.sh
python scripts/download_weights.py      # sha256-pinned
python -m sourcetrace.evaluate --task both --checkpoint checkpoints/mlaad_v5.pt
```

Downloading saves the fit and nothing else: evaluation still reads the feature
caches, so MLAAD v5 and STOPA must be downloaded and extracted first. See the
repository README for that step.

## Results

**MLAAD v5**, family-level open-set protocol: 65 known synthesis models, 9,620
evaluation utterances (4,375 known-model, 5,245 held-out-model), split seed 42, fit
seed 0.

| Metric | Value | Published |
|---|---|---|
| FPR95 ↓ | **0.50 %** | 3.36 % (Neamtu et al.) |
| OOD-EER ↓ | 2.45 % | — |
| Known-model accuracy ↑ | 99.86 % | — |

**STOPA**, released split and cosine-scoring protocol, the MLAAD-trained model
applied without retraining (33,200 enrolment and 629,800 probe utterances):

| Metric | Value |
|---|---|
| Held-out synthesis models, EER ↓ | **6.80 %** |
| Known synthesis models, EER ↓ | 9.24 % |
| Vocoder attribution, held-out / known, EER ↓ | 7.81 % / 3.86 % |

STOPA's own trained baselines report 35.34 % (ASVspoof-trained AASIST), 47.75 %
(STOPA-trained AASIST) and 49.55 % (ResNet-34) for held-out models; the 16.43 %
zero-shot EER of Chhibber et al. uses a different data split, so no margin is claimed
over it.

**Reproducible, not just reported.** Refitting from the public code at these seeds
reproduces `results/ablation/full.json` exactly, and this file is that fit.

**Single-seed.** One split seed, one fit seed. These are point estimates; the paper
reports the spread over five training seeds.

## Limitations

- Trained on MLAAD v5 only. Attribution across other corpora, languages, codecs or
  recording conditions is tested only on STOPA.
- For research on open-set attribution. **Not validated for forensic, legal or
  moderation use**, and the abstention rule is calibrated on this protocol — its
  coverage guarantee does not transfer off it.
- The harm from an attribution model is a confident wrong name, not a refusal. On an
  unseen synthesis model the calibrated answer is *unknown*, and that answer is the
  point of the system; do not deploy it anywhere the abstention is discarded, and do
  not present an attribution as evidence about a person.

## Licence

MIT, matching the code repository. MLAAD and STOPA carry their own terms; no audio is
redistributed here.

## Citation

See [`CITATION.cff`](https://github.com/pujariaditya/CoRA/blob/master/CITATION.cff)
in the code repository, which is the single source for how to cite this.

Please also cite the benchmarks (MLAAD, STOPA) and the baselines this is compared
against (Neamtu et al.; Chhibber et al., Odyssey 2026).
"""


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(prog="python scripts/upload_weights.py",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--mlaad_v5", default=None, help="path to the fitted head to publish")
    ap.add_argument("--card-only", action="store_true",
                    help="refresh the model card and upload no checkpoint")
    ap.add_argument("--delete-retired", action="store_true",
                    help=f"delete the retired files {RETIRED} from the Hub if present")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--dry-run", action="store_true", help="hash and report, upload nothing")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit(
            "set HF_TOKEN in the environment (do not use `huggingface-cli login`, "
            "which writes a world-readable token file)"
        )

    if not (args.mlaad_v5 or args.card_only or args.delete_retired):
        raise SystemExit("nothing to do: pass --mlaad_v5 PATH, --card-only or --delete-retired")
    if args.mlaad_v5 and args.card_only:
        raise SystemExit("--card-only uploads no checkpoint; drop the path or drop --card-only")

    plan = []
    if args.mlaad_v5:
        if not os.path.isfile(args.mlaad_v5):
            raise SystemExit(f"not a file: {args.mlaad_v5}")
        digest = sha256_of(args.mlaad_v5)
        print(f"| mlaad_v5.pt  {os.path.getsize(args.mlaad_v5) / 1e6:.1f} MB  sha256 {digest}")
        print("| paste this digest into CHECKPOINT in scripts/download_weights.py")
        plan.append((args.mlaad_v5, "mlaad_v5.pt"))

    if args.dry_run:
        if args.card_only:
            print(CARD)
        print("| dry run: nothing uploaded")
        return 0

    require_packages("huggingface_hub")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="model", private=args.private, exist_ok=True)

    for local, remote in plan:
        print(f"| uploading {remote}")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=REPO, repo_type="model")

    if args.delete_retired:
        present = set(api.list_repo_files(REPO, repo_type="model"))
        for name in RETIRED:
            if name in present:
                print(f"| deleting retired {name}")
                api.delete_file(path_in_repo=name, repo_id=REPO, repo_type="model")
            else:
                print(f"| {name}: not on the Hub, nothing to delete")

    # The card describes every artifact, so refresh it whenever anything changes.
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print(f"| done: https://huggingface.co/{REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
