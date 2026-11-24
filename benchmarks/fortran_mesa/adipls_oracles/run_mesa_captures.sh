#!/usr/bin/env bash
# Run one ADIPLS-calling MESA 26.4.1 astero case against the tapped overlay
# from build_mesa_overlay.sh (#171, package A0).
#
# Usage:
#   ADIPLS_ORACLE_WORK=<scratch dir> run_mesa_captures.sh <case>
#
# The case runs exactly as in the mesa_histories reference campaign
# (scripts/mesa_reference/run_mesa_case_26_4_1.sh: unmodified inlists,
# MESA_SKIP_OPTIONAL=t, OMP_NUM_THREADS=4, pulse_tap linked), with MESA_DIR
# pointing at the overlay. Besides the usual pulse_tap/ capture it writes the
# glue taps of astero_glue_tap.inc into <case dir>/a0tap/. Comparing the run's
# pulse_tap/modes.tsv with the bundle's shows that the taps changed nothing.
set -euo pipefail

CASE="${1:?usage: run_mesa_captures.sh <astero case>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${ADIPLS_ORACLE_WORK:?set ADIPLS_ORACLE_WORK to a scratch directory}"

export MESA_DIR="$WORK/mesa_overlay"
export MESA_GROUND_TRUTH_ROOT="$WORK/gt"
export OUT="$WORK/mesa_runs"
export ADIPLS_A0_TAP_DIR=a0tap
export OMP_NUM_THREADS=4

[ -d "$MESA_DIR/astero" ] || { echo "run build_mesa_overlay.sh first" >&2; exit 2; }
bash "$REPO/scripts/mesa_reference/run_mesa_case_26_4_1.sh" astero "$CASE"
