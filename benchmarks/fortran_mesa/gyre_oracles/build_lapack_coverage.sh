#!/usr/bin/env bash
# Build a line-coverage copy of reference LAPACK/BLAS 3.12.0 (double real and
# double complex parts only) to measure which routines and branches GYRE
# reaches (#170, package G0).
#
# MESA 26.4.1 links the MESA SDK x86_64-linux-26.6.1 `crmath` math slot,
# whose liblapack/libblas are netlib 3.12.0 compiled with gfortran 15.2.0
# against CRMATH (see README.md). That build ships no debug or coverage
# information, so branch reach is measured on this source build instead.
# The coverage archive is linked ahead of the SDK libraries; it is used
# only to record reach, never to produce oracle values.
#
# Usage:
#   GYRE_ORACLE_WORK=<scratch dir> LAPACK_TARBALL=<path to v3.12.0.tar.gz> \
#       build_lapack_coverage.sh
#
# The tarball is https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
set -euo pipefail

WORK="${GYRE_ORACLE_WORK:?set GYRE_ORACLE_WORK}"
TARBALL="${LAPACK_TARBALL:?set LAPACK_TARBALL}"
EXPECTED_SHA256="eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b"
DEST="$WORK/lapack-cov"

if [ -e "$DEST" ]; then
    echo "refusing: $DEST exists; remove it first" >&2
    exit 2
fi

echo "$EXPECTED_SHA256  $TARBALL" | sha256sum -c -

set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u

mkdir -p "$DEST/obj" "$DEST/mod"
tar -xzf "$TARBALL" -C "$DEST"
SRC="$DEST/lapack-3.12.0"

FFLAGS=(-O0 --coverage -frecursive -fPIC -J "$DEST/mod" -I "$DEST/mod")

compile() {
    local f="$1"
    local base
    base="$(basename "${f%.*}")"
    # Prefix object names with the source directory (BLAS/SRC, SRC, INSTALL):
    # BLAS/SRC and SRC both contain xerbla.f.
    local tag
    tag="$(basename "$(dirname "$(dirname "$f")")")_$(basename "$(dirname "$f")")"
    gfortran "${FFLAGS[@]}" -c "$f" -o "$DEST/obj/${tag}_${base}.o"
}

# Modules first
compile "$SRC/SRC/la_constants.f90"
compile "$SRC/SRC/la_xisnan.F90"

mapfile -t FILES < <(
    {
        find "$SRC/BLAS/SRC" -maxdepth 1 \( -name 'd*.f' -o -name 'z*.f' -o -name 'i*.f' \
            -o -name 'lsame.f' -o -name 'xerbla*.f' -o -name 'dnrm2.f90' \
            -o -name 'dznrm2.f90' -o -name 'drotg.f90' -o -name 'zrotg.f90' \)
        find "$SRC/SRC" -maxdepth 1 \( -name 'd*.f' -o -name 'z*.f' -o -name 'i*.f' \
            -o -name 'la*.f' -o -name 'd*.F' -o -name 'z*.F' -o -name 'i*.F' \
            -o -name 'dlartg.f90' -o -name 'dlassq.f90' -o -name 'zlartg.f90' \
            -o -name 'zlassq.f90' \)
        echo "$SRC/INSTALL/dlamch.f"
        echo "$SRC/INSTALL/droundup_lwork.f"
    } | sort
)

export -f compile
export DEST
export FFLAGS_STR="${FFLAGS[*]}"
printf '%s\n' "${FILES[@]}" | xargs -P "${GYRE_BUILD_JOBS:-4}" -I{} bash -c '
    f="{}"; base="$(basename "${f%.*}")"
    tag="$(basename "$(dirname "$(dirname "$f")")")_$(basename "$(dirname "$f")")"
    gfortran $FFLAGS_STR -c "$f" -o "$DEST/obj/${tag}_${base}.o"'

ar cr "$DEST/liblapack_cov.a" "$DEST"/obj/*.o
echo "built $DEST/liblapack_cov.a from ${#FILES[@]} sources"
