#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Shared Rust (`cargo llvm-cov`) and Python (`pytest --cov`) line-coverage
# runner invoked by `.github/workflows/dashboard.yml` (#174).
#
# Multi-case end-to-end and multi-model evolutionary sweeps run uninstrumented
# on every PR in the `rust`, `e2e`, and `history-harness` CI jobs; skipping
# those multi-hundred-step loops under LLVM/coverage instrumentation keeps
# coverage wall time around 8-12 minutes while measuring every unit,
# component-oracle, and short-step suite across all workspace crates.

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
      --skip test_surface_effects_case_recovers_synthetic_targets_and_bison_solar_fits \
      --skip test_coupled_double_bh_35_steps_preserves_star_controls \
      --skip test_recursive_binary_inlists_and_all_8_cases_capped_4_steps \
      --skip test_uncapped_ \
      --skip test_rsp_hyd_multicycle_period_save_load_and_photo_restart_parity \
      --skip test_rsp_hyd_step1_matches_all_7_mesa_26_4_1_reference_models \
      --skip test_rsp_lina_and_steffen_edges_match_mesa_26_4_1_5m_reference \
      --skip test_rsp_build_and_lina_match_all_7_mesa_26_4_1_reference_models \
      --skip test_rsp_check_2nd_crossing_on_7m_mesa_test_suite \
      --skip test_default_am_d_mix_zero_omega_transport_and_gold2_chem_increment_convergence \
      --skip test_relax_composition_with_henyey_low_metallicity_equilibration \
      --skip test_point_mass_rlof_binary_is_bit_identical_across_thread_counts \
      --skip test_two_star_coupled_binary_is_bit_identical_across_thread_counts \
      --skip test_binary_checkpoint_restart_in_child_process_is_bit_exact
    ;;
  python)
    PYTEST_BIN="${PYTEST_BIN:-pytest}"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTEST_BIN" -c "$REPO_ROOT/pyproject.toml" \
      "$REPO_ROOT/src" \
      "$REPO_ROOT/tests" \
      --ignore="$REPO_ROOT/tests/test_docs_snippets.py" \
      --ignore="$REPO_ROOT/tests/test_mesa_paper_figures.py" \
      --ignore="$REPO_ROOT/tests/test_history_harness.py" \
      --ignore-glob="$REPO_ROOT/tests/test_e2e_*.py" \
      -k "not test_dashboard_freshness_gate_matches_committed_outputs" \
      --cov="$REPO_ROOT/src/flexmesa" \
      --cov-report="json:$OUT_JSON" \
      -q
    ;;
  *)
    echo "Unknown coverage mode: $MODE (expected 'rust' or 'python')" >&2
    exit 2
    ;;
esac
