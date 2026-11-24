#!/usr/bin/env bash
# Run MESA 26.4.1's astero_gyre test case, unmodified, with a link-time tap
# that records the star state get_gyre_data reads at every GYRE set_model
# call (#170, G3b). The MESA tree is only read; the case is copied to WORK.
#
# Usage: WORK=<scratch dir> bash run_capture.sh
# Needs: MESA_DIR (built MESA 26.4.1), MESASDK_ROOT (x86_64-linux-26.6.1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${WORK:?set WORK to a scratch directory}"
# shellcheck disable=SC1091
source "$REPO/scripts/mesa_reference/env_mesa.sh"
mkdir -p "$WORK/build"
gfortran -O2 -ffree-form -ffree-line-length-none -I"$MESA_DIR/include" \
    -J "$WORK/build" -c "$HERE/pulse_state_tap.f90" -o "$WORK/build/pulse_state_tap.o"
gcc -O2 -c "$HERE/pulse_state_wrap.c" -o "$WORK/build/pulse_state_wrap.o"
dest="$WORK/astero_gyre"
if [ -e "$dest" ]; then
    echo "refusing to overwrite $dest" >&2
    exit 1
fi
cp -r "$MESA_DIR/astero/test_suite/astero_gyre" "$dest"
cd "$dest"
make -C make MESA_DIR="$MESA_DIR" \
    LOAD_EXTRAS1="$WORK/build/pulse_state_tap.o $WORK/build/pulse_state_wrap.o -Wl,--wrap=__gyre_mesa_m_MOD_set_model" \
    > mk.log 2>&1
export MESA_SKIP_OPTIONAL=t
./rn > out.txt 2>&1 || echo "rn exit $?" >> out.txt
tail -n 5 out.txt
ls pulse_state | wc -l
