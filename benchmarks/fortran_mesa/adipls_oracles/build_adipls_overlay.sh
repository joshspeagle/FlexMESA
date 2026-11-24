#!/usr/bin/env bash
# Build ADIPLS variants for the FlexMESA ADIPLS oracles (#171, package A0).
# The pristine MESA tree is only read.
#
# Usage:
#   ADIPLS_ORACLE_WORK=<scratch dir> build_adipls_overlay.sh <taps|coverage|libm>
#
# Environment:
#   MESA_DIR            built MESA 26.4.1 tree (default: $HOME/mesa-26.04.1)
#   MESASDK_ROOT        MESA SDK x86_64-linux-26.6.1
#   ADIPLS_ORACLE_WORK  scratch directory; <variant>/ must not exist yet
#
# Variants:
#   taps      copy of adipls/ with the read-only taps of apply_adipls_taps.py,
#             compiled exactly as MESA compiles ADIPLS (gfortran -std=legacy -w,
#             no -O flag), plus adipls_tap.o
#   coverage  unmodified copy compiled with the same flags plus --coverage
#   libm      no ADIPLS rebuild: crlibm 1.0beta4 from the vendored sources of
#             crates/flexmesa-crlibm-sys (compiled with that crate's flags) and
#             adipls_libm_wrap.o, for linking with the pristine libadipls.a
#
# Outputs: <variant>/adipls/make/libadipls.a (taps, coverage) and the helper
# objects in <variant>/; see README.md for the driver link lines.
set -euo pipefail

VARIANT="${1:?usage: build_adipls_overlay.sh <taps|coverage|libm>}"
case "$VARIANT" in taps|coverage|libm) ;; *) echo "unknown variant $VARIANT" >&2; exit 2;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
MESA_DIR="${MESA_DIR:-$HOME/mesa-26.04.1}"
WORK="${ADIPLS_ORACLE_WORK:?set ADIPLS_ORACLE_WORK to a scratch directory}"
DEST="$WORK/$VARIANT"

if [ -e "$DEST" ]; then
    echo "refusing: $DEST exists; remove it first" >&2
    exit 2
fi

set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u
mkdir -p "$DEST"

if [ "$VARIANT" = libm ]; then
    CR="$REPO/crates/flexmesa-crlibm-sys"
    mkdir -p "$DEST/crlibm"
    # Same sources, configuration headers and flags as crates/flexmesa-crlibm-sys/build.rs
    srcs="crlibm_private.c triple-double.c exp-td.c exp-td-standalone.c expm1-standalone.c
          expm1.c log.c log1p.c rem_pio2_accurate.c trigo_fast.c trigo_accurate.c trigpi.c
          asincos.c atan_fast.c atan_accurate.c csh_fast.c pow.c
          scs_lib/scs_private.c scs_lib/double2scs.c scs_lib/division_scs.c
          scs_lib/addition_scs.c scs_lib/multiplication_scs.c scs_lib/scs2double.c
          scs_lib/zero_scs.c"
    for s in $srcs; do
        gcc -O2 -DHAVE_CONFIG_H=1 -I"$CR/config/x86_64" -I"$CR/config/x86_64/scs_inc" \
            -I"$CR/csrc" -I"$CR/csrc/scs_lib" -std=c99 -ffp-contract=off -fno-fast-math \
            -fwrapv -fno-strict-aliasing -msse2 -mfpmath=sse -w \
            -c "$CR/csrc/$s" -o "$DEST/crlibm/$(echo "$s" | tr / _ | sed 's/\.c$/.o/')"
    done
    ar rcs "$DEST/libflexmesa_crlibm.a" "$DEST"/crlibm/*.o
    gcc -O2 -I"$CR/csrc" -c "$HERE/adipls_libm_wrap.c" -o "$DEST/adipls_libm_wrap.o"
    echo "built $DEST/libflexmesa_crlibm.a and $DEST/adipls_libm_wrap.o"
    exit 0
fi

cp -a "$MESA_DIR/adipls" "$DEST/adipls"
touch "$DEST/adipls/.flexmesa_adipls_overlay"
find "$DEST/adipls/make" -maxdepth 1 -type f \( -name '*.o' -o -name '*.a' -o -name '*.gcda' \
    -o -name '*.gcno' -o -name 'ofiles_blstio.f' \) -delete

FCV=gfortran
if [ "$VARIANT" = taps ]; then
    python3 "$HERE/apply_adipls_taps.py" "$DEST/adipls"
    gcc -O2 -c "$HERE/adipls_tap.c" -o "$DEST/adipls_tap.o"
else
    FCV="gfortran --coverage"
fi

# MESA's own makefile and rules (adipls/make/makefile_adipls); only the
# library target, so nothing is installed into MESA_DIR.
make -C "$DEST/adipls/make" -f makefile_adipls MESA_DIR="$MESA_DIR" FC="$FCV" \
    libadipls.a -j "${ADIPLS_BUILD_JOBS:-4}"

echo "built $VARIANT: $DEST/adipls/make/libadipls.a"
