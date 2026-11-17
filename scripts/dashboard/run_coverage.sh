#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Shared Rust (`cargo llvm-cov`) and Python (`pytest --cov`) line-coverage
# runner invoked by `.github/workflows/ci.yml` and `.github/workflows/dashboard.yml`
# (#174) so PR gates and `main` dashboard refreshes measure identical suites.
#
# Multi-case end-to-end trajectory sweeps run uninstrumented in the dedicated
# `rust`, `e2e`, and `history-harness` CI jobs; skipping those multi-hundred-step
# campaign loops under LLVM/coverage instrumentation keeps coverage wall time
# bounded while measuring every unit, component-oracle, and short-step suite.

set -euo pipefail

MODE="${1:?Usage: run_coverage.sh <rust|python> <output-json>}"
OUT_JSON="${2:?Usage: run_coverage.sh <rust|python> <output-json>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$(dirname "$OUT_JSON")"

case "$MODE" in
  rust)
    cargo llvm-cov \
      --manifest-path "$REPO_ROOT/Cargo.toml" \
      --workspace \
      --exclude flexmesa-py \
      --release \
      --json \
      --summary-only \
      --output-path "$OUT_JSON" \
      -- \
      --skip test_astero_adipls_and_gyre_uncapped_pass \
      --skip test_all_eight_astero_cases_bounded_eight_steps_pass \
      --skip test_e2e_1_tier1_physics_helpers_and_suite \
      --skip test_e2e_2_tier2_physics_helpers_and_suite \
      --skip test_e2e_4_tier4_pulsation_astero_and_dev_cases \
      --skip test_e2e_5_tier5_star_hooks_and_90_active_star_inventory \
      --skip test_uncapped_ \
      --skip test_rsp_hyd_multicycle_period_save_load_and_photo_restart_parity
    ;;
  python)
    PYTEST_BIN="${PYTEST_BIN:-pytest}"
    "$PYTEST_BIN" -c "$REPO_ROOT/pyproject.toml" \
      "$REPO_ROOT/src" \
      "$REPO_ROOT/tests" \
      --ignore="$REPO_ROOT/tests/test_docs_snippets.py" \
      --ignore="$REPO_ROOT/tests/test_mesa_paper_figures.py" \
      --ignore="$REPO_ROOT/tests/test_history_harness.py" \
      --ignore-glob="$REPO_ROOT/tests/test_e2e_*.py" \
      --cov="$REPO_ROOT/src/flexmesa" \
      --cov-report="json:$OUT_JSON" \
      -q
    ;;
  *)
    echo "Unknown coverage mode: $MODE (expected 'rust' or 'python')" >&2
    exit 2
    ;;
esac
