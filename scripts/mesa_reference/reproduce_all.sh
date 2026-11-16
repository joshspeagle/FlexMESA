#!/usr/bin/env bash
# End-to-end reproducible pipeline to build MESA 26.4.1 outside the repository,
# capture exact IEEE-754 hex values from all module tests, run all 106
# star/binary/astero test_suite cases (including GYRE/ADIPLS mode + model
# capture), record provenance, generate summary tables, and package
# mesa-histories-v1.tar.zst and mesa-module-hex-v1.tar.zst.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_ROOT="${MESA_GROUND_TRUTH_ROOT:-$HOME/mesa-ground-truth-26.4.1}"
export MESA_GROUND_TRUTH_ROOT="$WORK_ROOT"
export FLEXMESA_DIR="${FLEXMESA_DIR:-$REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$WORK_ROOT/build"

# 1. Build MESA 26.4.1 with MESA SDK 26.6.1 and verify all 26 module self-checks
bash "$SCRIPT_DIR/build_mesa_26_4_1.sh"

# 2. Compile external GYRE/ADIPLS linker wrapper objects (kept outside $MESA_DIR and repo)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_mesa.sh"
gfortran -O2 -ffree-form -ffree-line-length-none -I"$MESA_DIR/include" \
    -c "$SCRIPT_DIR/pulse_tap_gyre.f90" -o "$WORK_ROOT/build/pulse_tap_gyre.o"
gfortran -O2 -ffree-form -ffree-line-length-none -I"$MESA_DIR/include" \
    -c "$SCRIPT_DIR/pulse_tap_adipls.f90" -o "$WORK_ROOT/build/pulse_tap_adipls.o"
gcc -O2 -c "$SCRIPT_DIR/pulse_tap_wrap.c" -o "$WORK_ROOT/build/pulse_tap_wrap_gyre.o"
gcc -O2 -DPTAP_WITH_ADIPLS -c "$SCRIPT_DIR/pulse_tap_wrap.c" -o "$WORK_ROOT/build/pulse_tap_wrap_all.o"

# 3. Run all module test programs with external libgfortran_hex_tap.so LD_PRELOAD wrapper
"$PYTHON_BIN" "$SCRIPT_DIR/run_module_hex_tests.py"

# 4. Record full build, compiler-flag, hardware, and runtime provenance
"$PYTHON_BIN" "$SCRIPT_DIR/write_provenance.py"

# 5. Create 3-level parent symlinks in external $WORK_ROOT required by upstream
#    starspots hardcoded MESA_DIR=../../.., and run all 106 test_suite cases
for d in data include lib star utils; do
    ln -sfn "$MESA_DIR/$d" "$WORK_ROOT/$d"
done
"$PYTHON_BIN" "$SCRIPT_DIR/run_all_cases.py" \
    --workers "${WORKERS:-3}" \
    --timeout-s "${TIMEOUT_S:-14400}" \
    --max-rss-gb "${MAX_RSS_GB:-20}"

# 6. Generate summary tables (MD, TSV, JSON) and package release archives
"$PYTHON_BIN" "$SCRIPT_DIR/generate_summary_and_archive.py"
"$PYTHON_BIN" "$REPO_ROOT/scripts/data/build_mesa_histories.py" \
    --ground-truth-dir "$WORK_ROOT" \
    --repo-root "$REPO_ROOT" \
    --out-dir "$WORK_ROOT/bundles_out"
