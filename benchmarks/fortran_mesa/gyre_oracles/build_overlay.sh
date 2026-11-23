#!/usr/bin/env bash
# Build an overlay copy of MESA 26.4.1's bundled GYRE 8.1 for the FlexMESA
# GYRE oracles (#170, package G0). The pristine MESA tree is only read.
#
# Usage:
#   GYRE_ORACLE_WORK=<scratch dir> build_overlay.sh <taps|coverage>
#
# Environment:
#   MESA_DIR        built MESA 26.4.1 tree (default: $HOME/mesa-26.04.1)
#   MESASDK_ROOT    MESA SDK x86_64-linux-26.6.1
#   GYRE_ORACLE_WORK  scratch directory; <variant>/gyre must not exist yet
#
# Variants:
#   taps      same compiler flags as the MESA build of GYRE (-O2
#             -ffp-contract=off -fopenmp -ffpe-trap=..., CRMATH) plus the
#             read-only taps of gyre_tap_m.fypp
#   coverage  -O0 --coverage instead of -O2, no taps (line coverage only)
#
# GYRE is configured as in gyre/make/makefile_full of MESA 26.4.1:
# FRONTENDS=no TOOLS=no IFACES=yes FORUM=no CRMATH=yes SHARED=no.
set -euo pipefail

VARIANT="${1:?usage: build_overlay.sh <taps|coverage>}"
case "$VARIANT" in taps|coverage) ;; *) echo "unknown variant $VARIANT" >&2; exit 2;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESA_DIR="${MESA_DIR:-$HOME/mesa-26.04.1}"
WORK="${GYRE_ORACLE_WORK:?set GYRE_ORACLE_WORK to a scratch directory}"
DEST="$WORK/$VARIANT/gyre"

if [ -e "$DEST" ]; then
    echo "refusing: $DEST exists; remove it first" >&2
    exit 2
fi

set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u

mkdir -p "$WORK/$VARIANT"
cp -a "$MESA_DIR/gyre/gyre" "$DEST"
touch "$DEST/.flexmesa_gyre_overlay"

# Drop build products copied from the pristine tree (keep generated .f90,
# whose timestamps are older than any .fypp edited below).
find "$DEST/build" -maxdepth 1 -type f \
    \( -name '*.o' -o -name '*.mod' -o -name '*.smod' -o -name '*.a' \
       -o -name '*.so' -o -name 'test_gyre_mesa' -o -name '*.gcda' -o -name '*.gcno' \) -delete
# The copied dependency files hold paths relative to the pristine tree
# (e.g. ../../../include/forum.inc); drop them so make regenerates them.
find "$DEST/build/.depend" -maxdepth 1 -type f \
    \( -name '*.anc' -o -name '*.dpp' -o -name '*.dep' \) -delete 2>/dev/null || true
find "$DEST" -maxdepth 1 -type d \( -name lib -o -name include -o -name bin \) \
    -exec find {} -type f -delete \;

python3 "$HERE/apply_taps.py" "$DEST" "--$VARIANT"

make -C "$DEST" install \
    FRONTENDS=no TOOLS=no IFACES=yes FORUM=no CRMATH=yes SHARED=no \
    FORUM_LIB_DIR="$MESA_DIR/lib" FORUM_INC_DIR="$MESA_DIR/include" \
    -j "${GYRE_BUILD_JOBS:-4}"

echo "built $VARIANT overlay: $DEST/build (libgyre.a, libgyre_mesa.a, *.mod)"
