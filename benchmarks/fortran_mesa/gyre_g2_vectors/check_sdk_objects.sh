#!/usr/bin/env bash
# Establish how the MESA SDK x86_64-linux-26.6.1 builds the LAPACK/BLAS
# routines the GYRE G2 port reproduces (#170, package G2).
#
# For each routine, the SDK crmath-slot object (liblapack.a / libblas.a) is
# disassembled and compared, instruction by instruction with addresses and
# relocation offsets masked, with Reference LAPACK 3.12.0's source compiled
# by the SDK's gfortran at several flag sets. The result table and the
# dbdsqr TOLMUL disassembly excerpt are written to sdk_object_check.txt.
#
# Usage:
#   MESASDK_ROOT=<sdk> LAPACK_TARBALL=<v3.12.0.tar.gz> WORK=<scratch> check_sdk_objects.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:?set WORK to a scratch directory}"
TARBALL="${LAPACK_TARBALL:?set LAPACK_TARBALL}"
echo "eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b  $TARBALL" | sha256sum -c - >/dev/null

set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u

mkdir -p "$WORK/src" "$WORK/obj" "$WORK/mod"
tar -xzf "$TARBALL" -C "$WORK/src"
SRC="$WORK/src/lapack-3.12.0"
SLOT="$MESASDK_ROOT/math-slots/crmath/lib"
cd "$WORK/obj"
gfortran -O2 -fPIC -J "$WORK/mod" -c "$SRC/SRC/la_constants.f90" -o la_constants.o
gfortran -O2 -fPIC -J "$WORK/mod" -c "$SRC/SRC/la_xisnan.F90" -o la_xisnan.o

norm() {
    objdump -d --no-show-raw-insn "$1" | sed -n '/<.*>:/,$p' | awk '{$1=""; print}' \
        | sed 's/<[^>]*>//g; s/0x[0-9a-f]*//g; s/[0-9a-f]\{4,\}//g'
}

ROUTINES="dasum daxpy dcopy dgemm dgemv dger dnrm2 drot dscal dswap dtrsm dtrsv idamax lsame
dbdsqr dgebd2 dgebrd dgecon dgeequ dgeqrf dgerfs dgesvd dgesvx dgetrf dgetrf2 dgetrs disnan
dlacn2 dlacpy dlaisnan dlamch dlange dlantr dlapy2 dlaqge dlarf dlarfg dlartg dlas2 dlasr
dlasv2 dlaswp dlatrs dorg2r dorgbr dorgl2 dorglq dorgqr iladlc iladlr ilaenv
zgemm zgemv zgerc ztrsm ztrsv zscal zswap zcopy zlaswp zlacpy zlarf zgetrf zgetrs zung2r zungqr
zgebd2 zgetrf2 zlahqr zladiv dladiv iparmq"

OUT="$HERE/sdk_object_check.txt"
{
    echo "# SDK crmath-slot LAPACK/BLAS objects vs Reference LAPACK 3.12.0 source (#170, G2)"
    echo "# mesasdk_version: x86_64-linux-26.6.1; compiler: $(gfortran --version | head -1)"
    echo "# comparison: objdump -d, addresses and relocation offsets masked"
    echo "# routine	library	matching_flags (empty: none of -O0/-O2/-O3, each +-fPIC)"
} > "$OUT"
for r in $ROUTINES; do
    lib=""
    for l in lapack blas; do
        if ar t "$SLOT/lib$l.a" | grep -qx "$r.o"; then lib=$l; ar x "$SLOT/lib$l.a" "$r.o"; mv "$r.o" "sdk_$r.o"; break; fi
    done
    src=""
    for c in "$SRC/SRC/$r.f" "$SRC/SRC/$r.f90" "$SRC/BLAS/SRC/$r.f" "$SRC/BLAS/SRC/$r.f90" "$SRC/INSTALL/$r.f"; do
        [ -f "$c" ] && { src=$c; break; }
    done
    matches=""
    for fl in "-O0 -fPIC" "-O2 -fPIC" "-O3 -fPIC" "-O0" "-O2" "-O3"; do
        # shellcheck disable=SC2086
        gfortran $fl -I "$WORK/mod" -c "$src" -o "cand_$r.o" 2>/dev/null || continue
        if diff -q <(norm "cand_$r.o") <(norm "sdk_$r.o") >/dev/null; then
            matches="$matches${matches:+, }$fl"
        fi
    done
    printf '%s\t%s\t%s\n' "$r" "$lib" "$matches" >> "$OUT"
done
{
    echo
    echo "# dbdsqr: TOLMUL = MAX(TEN, MIN(HNDRD, EPS**MEIGTH)) in the SDK object"
    echo "# (0x180(%rsp) holds DLAMCH('Epsilon'); pxor zeroes the multiplier of log(EPS))"
    objdump -dr "sdk_dbdsqr.o" | grep -B6 -A14 "crmath_MOD_log_r_dp" | sed 's/^/#   /'
} >> "$OUT"
{
    echo
    echo "# SDK objects that reference CRMATH (undefined __crmath_MOD_* symbols)"
    for l in lapack blas; do
        nm -A -u "$SLOT/lib$l.a" 2>/dev/null | awk -v l="$l" '$NF ~ /__crmath_MOD/ {
            split($1, p, ":"); o = p[2]; sub(/\.o$/, "", o); s = $NF
            sub(/^__crmath_MOD_/, "", s); sub(/_$/, "", s); print l "\t" o "\t" s }' | sort -u
    done | awk -F'\t' '{k = $1 "\t" $2; a[k] = (k in a) ? a[k] " " $3 : $3} END {for (k in a) print "#   " k "\t" a[k]}' | sort
} >> "$OUT"
echo "wrote $OUT"
