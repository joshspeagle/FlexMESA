#!/usr/bin/env bash
# Run one upstream MESA 26.4.1 test_suite case with a real MESA build.
#
# Adapted directly from FlexMESA notes/audit/tools/run_mesa_case.sh for MESA
# release 26.4.1 (which uses make/makefile via `make -C make` rather than a
# top-level Makefile). Zero inlists or upstream files are modified. For cases
# that invoke GYRE or ADIPLS, links the external `pulse_tap` object via
# `LOAD_EXTRAS1` so every returned mode (frequency, inertia, growth rate) and
# the exact model it was computed from are recorded under `pulse_tap/`.
#
# Usage:
#   OUT=<work dir> run_mesa_case_26_4_1.sh <star|binary|astero> <case> [full]

set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${MESA_GROUND_TRUTH_ROOT:-$HOME/mesa-ground-truth-26.4.1}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_mesa.sh"

: "${MESA_DIR:?set MESA_DIR to a built MESA checkout}"
: "${MESASDK_ROOT:?set MESASDK_ROOT to the MESA SDK}"
: "${OUT:?set OUT to a scratch directory for case copies}"

mod=$1
case_name=$2
mode=${3:-skip}

mkdir -p "$OUT"
dest="$OUT/$case_name"
if [ -e "$dest" ]; then
    echo "refusing to overwrite existing $dest" >&2
    exit 1
fi

cp -r "$MESA_DIR/$mod/test_suite/$case_name" "$dest"
cd "$dest"

LOAD_EXTRAS1_VAL=""
if [ "$mod" = "astero" ]; then
    LOAD_EXTRAS1_VAL="$ROOT_DIR/build/pulse_tap_gyre.o $ROOT_DIR/build/pulse_tap_adipls.o $ROOT_DIR/build/pulse_tap_wrap_all.o -Wl,--wrap=__gyre_mesa_m_MOD_set_model -Wl,--wrap=__gyre_mesa_m_MOD_get_modes -Wl,--wrap=adipls_ -Wl,--wrap=spcout_adi_"
elif [[ "$case_name" == gyre_in_mesa_* ]] || [ "$case_name" = "rsp_gyre" ]; then
    LOAD_EXTRAS1_VAL="$ROOT_DIR/build/pulse_tap_gyre.o $ROOT_DIR/build/pulse_tap_wrap_gyre.o -Wl,--wrap=__gyre_mesa_m_MOD_set_model -Wl,--wrap=__gyre_mesa_m_MOD_get_modes"
fi

if [ -f Makefile ] || [ -f makefile ]; then
    if [ -n "$LOAD_EXTRAS1_VAL" ]; then
        make MESA_DIR="$MESA_DIR" WITH_PGSTAR=no LOAD_EXTRAS1="$LOAD_EXTRAS1_VAL" > mk.log 2>&1
    else
        make MESA_DIR="$MESA_DIR" WITH_PGSTAR=no > mk.log 2>&1
    fi
else
    if [ -n "$LOAD_EXTRAS1_VAL" ]; then
        make -C make MESA_DIR="$MESA_DIR" LOAD_EXTRAS1="$LOAD_EXTRAS1_VAL" > mk.log 2>&1
    else
        make -C make MESA_DIR="$MESA_DIR" > mk.log 2>&1
    fi
fi

if [ "$mode" = "full" ]; then
    unset MESA_SKIP_OPTIONAL || true
else
    export MESA_SKIP_OPTIONAL=t
fi

start=$(date +%s)
./rn > out.txt 2>&1 || echo "rn exit $?" >> out.txt
echo "wall_s=$(( $(date +%s) - start ))" >> out.txt
tail -n 25 out.txt
