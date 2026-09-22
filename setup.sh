#!/usr/bin/env bash
# Fetch everything sourcetrace needs but does not redistribute.
#
# The method itself is in this repository, and so are the protocol definitions --
# the MLAAD manifests, the merge map, the STOPA attack grid all ship in assets/.
# What this fetches is the two frozen front-end models the feature extractor runs
# audio through, and neither is redistributed here.
#
# It installs no Python packages. Do that first, with `pip install -e .`, under
# the sourcetrace environment -- every entry point checks the interpreter and
# will tell you if it is the wrong one.
# --help: the header comment above IS the documentation, so print it rather than
# maintaining a second copy that can drift out of date. Anything other than -h/--help
# is rejected: this script takes no positional arguments, and silently ignoring one
# would look like it had been honoured -- here that means `./setup.sh --help`
# starting a multi-gigabyte download instead of printing anything.
usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") : ;;
  *) echo "$(basename "$0"): unexpected argument '$1'" >&2
     echo "This script takes no arguments; run '$(basename "$0") --help'." >&2
     exit 2 ;;
esac

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

# ~/.local site-packages can shadow the pinned huggingface_hub and break the
# pinned transformers. See sourcetrace/features/_deps.py.
export PYTHONNOUSERSITE=1

echo "==> frozen front ends (WavLM-Large via s3prl, EnCodec-24kHz)"
"$PY" scripts/download_models.py

echo
echo "Setup complete. Next:"
echo "  python scripts/smoke.py              # 20 s, no download: check the install"
echo "  python scripts/download_weights.py   # the fitted head, sha256-pinned"
echo "  ./scripts/verify_results.sh          # reproduce the reported table"
