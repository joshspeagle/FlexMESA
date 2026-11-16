#!/usr/bin/env bash
# Run one upstream MESA test_suite case with a real MESA build.
#
# Usage:
#   MESA_DIR=<mesa checkout, built> MESASDK_ROOT=<sdk> OUT=<work dir> \
#     run_mesa_case.sh <star|binary|astero> <case> [full]
#
# By default MESA_SKIP_OPTIONAL is set, so optional early stages are replaced
# by the saved models shipped with the case (the same branch FlexMESA's runner
# takes with skip_optional). Pass "full" as the third argument to run them.
set -eu
: "${MESA_DIR:?set MESA_DIR to a built MESA checkout}"
: "${MESASDK_ROOT:?set MESASDK_ROOT to the MESA SDK}"
: "${OUT:?set OUT to a scratch directory for case copies}"
set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u
export MESA_DIR MESASDK_ROOT OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mod=$1
case=$2
mode=${3:-skip}
mkdir -p "$OUT"
dest="$OUT/$case"
if [ -e "$dest" ]; then
    echo "refusing to overwrite existing $dest" >&2
    exit 1
fi
cp -r "$MESA_DIR/$mod/test_suite/$case" "$dest"
cd "$dest"
make WITH_PGSTAR=no > mk.log 2>&1
if [ "$mode" = full ]; then
    unset MESA_SKIP_OPTIONAL || true
else
    export MESA_SKIP_OPTIONAL=t
fi
start=$(date +%s)
./rn > out.txt 2>&1 || echo "rn exit $?" >> out.txt
echo "wall_s=$(( $(date +%s) - start ))" >> out.txt
tail -n 25 out.txt
