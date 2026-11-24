#!/usr/bin/env bash
# Regenerate the per-track MESA 26.4.1 reference histories of the astero
# search cases and the mesa-astero-tracks-v1 bundle (#430). See README.md.
#
# Usage:
#   OUT=<scratch dir> STORED_THINNED=<mesa_histories>/thinned/astero \
#       run_tracks.sh
#
# Runs each case with scripts/mesa_reference/run_mesa_case_26_4_1.sh and the
# history tap linked in, thins every track into $OUT/thinned/<case>/, and
# packages $OUT/mesa-astero-tracks-v1.tar.zst plus the per-file manifest.
# Cases run one after another. The MESA tree is only read, never modified.

set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${OUT:?set OUT to a scratch directory outside the repository}"
: "${STORED_THINNED:?set STORED_THINNED to mesa-histories-v1 thinned/astero}"
CASES=(fast_newuoa fast_simplex fast_scan_grid fast_from_file example_astero)

# shellcheck disable=SC1091
source "$REPO/scripts/mesa_reference/env_mesa.sh"
mkdir -p "$OUT/build" "$OUT/runs/astero"
gcc -O2 -Wall -c "$HERE/history_tap_wrap.c" -o "$OUT/build/history_tap_wrap.o"

for c in "${CASES[@]}"; do
    OUT="$OUT/runs/astero" HISTORY_TAP_OBJ="$OUT/build/history_tap_wrap.o" \
        bash "$REPO/scripts/mesa_reference/run_mesa_case_26_4_1.sh" astero "$c"
    python3 "$HERE/thin_track_histories.py" "$OUT/runs/astero/$c" \
        "$OUT/thinned/$c" --last-reference "$STORED_THINNED/$c.data"
done

python3 "$HERE/build_bundle.py" "$OUT/thinned" "$STORED_THINNED" \
    "$OUT/mesa-astero-tracks-v1.tar.zst" \
    "$REPO/src/flexmesa/data/bundle_hashes/mesa_astero_tracks.sha256.tsv"
