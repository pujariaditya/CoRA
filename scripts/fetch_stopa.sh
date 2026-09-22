#!/usr/bin/env bash
# Download + extract the STOPA corpus from Zenodo into ST_STOPA_ROOT.
#
# STOPA (Zenodo record 15606628, DOI 10.5281/zenodo.15606628) ships as a 4-part
# 7z archive (STOPA.7z.001 .. .004) plus a README. This script:
#   1. downloads all parts (resumable via `wget -c`) into $ST_STOPA_ROOT,
#   2. verifies the parts are present,
#   3. lists the archive contents (sanity check), and
#   4. extracts them (7z auto-joins the multi-part set from part .001).
#
# The audio must land so the STOPA table builder finds
#   $ST_STOPA_ROOT/{EET,TEE,Trials}/wav/LA_*.wav
# (see sourcetrace/datasets/stopa.py). Adjust ST_STOPA_ROOT to control where it goes.
#
# Checksums: after download, cross-check each part against the sizes/hashes on the
# Zenodo record page (https://zenodo.org/records/15606628); Zenodo lists an MD5
# per file. To verify locally:  md5sum STOPA.7z.00*  and compare.
#
# Env:
#   ST_STOPA_ROOT   destination dir (default: ./data/STOPA)
#   STOPA_NO_EXTRACT=1  download + verify only, skip extraction
# --help: the header comment above IS the documentation, so print it rather than
# maintaining a second copy that can drift out of date. Anything other than -h/--help
# is rejected: this script takes no positional arguments and silently ignoring one
# would look like it had been honoured.
usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") : ;;
  *) echo "$(basename "$0"): unexpected argument '$1'" >&2
     echo "This script takes no arguments; run '$(basename "$0") --help'." >&2
     exit 2 ;;
esac

set -uo pipefail

DEST="${ST_STOPA_ROOT:-${ST_DATA_ROOT:-./data}/STOPA}"
BASE="https://zenodo.org/records/15606628/files"
PARTS=(STOPA.7z.001 STOPA.7z.002 STOPA.7z.003 STOPA.7z.004)

mkdir -p "$DEST"
cd "$DEST" || { echo "cannot cd to $DEST" >&2; exit 1; }
echo "[stopa] destination: $DEST"

# --- download ------------------------------------------------------------- #
for f in README.txt "${PARTS[@]}"; do
  echo "[stopa] $(date +%H:%M:%S) downloading $f"
  if wget -c -q --tries=5 --retry-connrefused --waitretry=10 \
        "$BASE/$f?download=1" -O "$f"; then
    echo "[stopa]   done $f ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "[stopa]   WARNING: download of $f did not complete (re-run to resume)" >&2
  fi
done

# --- verify parts present ------------------------------------------------- #
missing=0
for f in "${PARTS[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[stopa]   MISSING/empty: $f" >&2
    missing=1
  fi
done
echo "=== parts ==="; ls -lh "$DEST"/STOPA.7z.* 2>/dev/null
if [[ "$missing" -ne 0 ]]; then
  echo "[stopa] some parts are missing; re-run to resume before extracting." >&2
  exit 1
fi

# --- verify 7z available -------------------------------------------------- #
if ! command -v 7z >/dev/null 2>&1; then
  echo "[stopa] '7z' not found. Install p7zip (e.g. apt-get install p7zip-full) " \
       "then run: 7z x STOPA.7z.001" >&2
  exit 1
fi

echo "=== archive listing (verify size/structure) ==="
7z l STOPA.7z.001 2>&1 | tail -30

# --- extract -------------------------------------------------------------- #
if [[ "${STOPA_NO_EXTRACT:-0}" == "1" ]]; then
  echo "[stopa] STOPA_NO_EXTRACT=1 set; skipping extraction."
else
  echo "[stopa] extracting (7z joins the .001..004 set automatically) ..."
  if 7z x -y STOPA.7z.001 >/dev/null; then
    echo "[stopa] extraction complete."
  else
    echo "[stopa] extraction FAILED (check disk space / archive integrity)." >&2
    exit 1
  fi
  echo "=== extracted top-level ==="; ls -1 "$DEST" | head -20
fi

echo "=== STOPA DOWNLOAD+EXTRACT DONE ($DEST) ==="
