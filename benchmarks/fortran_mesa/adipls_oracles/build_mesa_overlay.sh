#!/usr/bin/env bash
# Build an overlay of MESA 26.4.1 whose astero library carries the read-only
# ADIPLS glue taps (#171, package A0). The pristine MESA tree is only read.
#
# Usage:
#   ADIPLS_ORACLE_WORK=<scratch dir> build_mesa_overlay.sh
#
# Environment:
#   MESA_DIR            built MESA 26.4.1 tree (default: $HOME/mesa-26.04.1)
#   MESASDK_ROOT        MESA SDK x86_64-linux-26.6.1
#   ADIPLS_ORACLE_WORK  scratch directory; mesa_overlay/ must not exist yet
#
# Layout: $ADIPLS_ORACLE_WORK/mesa_overlay has real copies of astero/, lib/ and
# include/ and symbolic links to every other top-level entry of MESA_DIR.
# astero/private/adipls_support{,_procs}.f90 are patched by
# apply_astero_taps.py, rebuilt with MESA's own astero makefile, and installed
# into the overlay's lib/ and include/. libadipls.a is the unmodified MESA
# build. The script also compiles the pulse_tap objects of
# scripts/mesa_reference (the modes.tsv capture used by the mesa_histories
# bundle) into $ADIPLS_ORACLE_WORK/gt/build for run_mesa_captures.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
MESA_DIR="${MESA_DIR:-$HOME/mesa-26.04.1}"
WORK="${ADIPLS_ORACLE_WORK:?set ADIPLS_ORACLE_WORK to a scratch directory}"
OV="$WORK/mesa_overlay"

if [ -e "$OV" ]; then
    echo "refusing: $OV exists; remove it first" >&2
    exit 2
fi

set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u

mkdir -p "$OV"
for entry in "$MESA_DIR"/* "$MESA_DIR"/.[!.]*; do
    [ -e "$entry" ] || continue
    name="$(basename "$entry")"
    case "$name" in
        astero|lib|include) cp -a "$entry" "$OV/$name" ;;
        *) ln -s "$entry" "$OV/$name" ;;
    esac
done
touch "$OV/astero/.flexmesa_adipls_overlay"

python3 "$HERE/apply_astero_taps.py" "$OV/astero"

(cd "$OV/astero/make" && make MESA_DIR="$OV" && make MESA_DIR="$OV" install)

# The pristine archive must not contain the tap symbols.
if [ "$(nm "$MESA_DIR/lib/libastero.a" 2>/dev/null | grep -c a0tap_ || true)" != 0 ]; then
    echo "pristine libastero.a contains a0tap_ symbols; stop" >&2
    exit 1
fi
if [ "$(nm "$OV/lib/libastero.a" | grep -c a0tap_call || true)" = 0 ]; then
    echo "overlay libastero.a lacks taps" >&2
    exit 1
fi

# pulse_tap objects (same sources as the mesa_histories reference runs)
GT="$WORK/gt/build"
mkdir -p "$GT"
SRC="$REPO/scripts/mesa_reference"
(cd "$GT" &&
    gfortran -O2 -ffree-line-length-none -I"$OV/include" -c "$SRC/pulse_tap_gyre.f90" -o pulse_tap_gyre.o &&
    gfortran -O2 -ffree-line-length-none -I"$OV/include" -c "$SRC/pulse_tap_adipls.f90" -o pulse_tap_adipls.o &&
    gcc -O2 -DPTAP_WITH_ADIPLS -c "$SRC/pulse_tap_wrap.c" -o pulse_tap_wrap_all.o)

echo "built overlay $OV (tapped libastero.a) and pulse_tap objects in $GT"
