#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Shared Rust (`cargo llvm-cov`) and Python (`pytest --cov`) line-coverage
# runner invoked by `.github/workflows/dashboard.yml` (#174).
#
# Rust coverage runs as shards (`rust-shard`), one per `Safe Rust Tests` CI
# shard, each writing an LCOV report that `merge_lcov.py` combines (#345).

set -euo pipefail

MODE="${1:?Usage: run_coverage.sh rust-shard <package|rest> <index> <total> <out.lcov> | python <output-json>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Multi-hundred-step evolutions are left out under instrumentation; they run
# uninstrumented in the CI test shards.
RUST_SKIPS=(
  test_just_call_my_extras_
  test_all_pinned_raw_eos_slices_fit_without_non_finite_entries
  test_all_eight_astero_cases_bounded_eight_steps_pass
  test_surface_effects_case_recovers_synthetic_targets_and_bison_solar_fits
  test_coupled_double_bh_35_steps_preserves_star_controls
  test_recursive_binary_inlists_and_all_8_cases_capped_4_steps
  test_uncapped_
  test_rsp_hyd_multicycle_period_save_load_and_photo_restart_parity
  test_rsp_hyd_step1_matches_all_7_mesa_26_4_1_reference_models
  test_rsp_lina_and_steffen_edges_match_mesa_26_4_1_5m_reference
  test_rsp_build_and_lina_match_all_7_mesa_26_4_1_reference_models
  test_rsp_check_2nd_crossing_on_7m_mesa_test_suite
  test_default_am_d_mix_zero_omega_transport_and_gold2_chem_increment_convergence
  test_relax_composition_with_henyey_low_metallicity_equilibration
  test_point_mass_rlof_binary_is_bit_identical_across_thread_counts
  test_two_star_coupled_binary_is_bit_identical_across_thread_counts
  test_binary_checkpoint_restart_in_child_process_is_bit_exact
)

case "$MODE" in
  rust-shard)
    # One shard of the Rust coverage run (#345). The shards are those of the
    # `Safe Rust Tests` CI jobs: `scripts/ci/cargo_test_shard.py` for the
    # split packages and every other package together as `rest`. Each shard
    # writes an LCOV report; `scripts/dashboard/merge_lcov.py` merges them.
    PACKAGE="${2:?package or rest}"
    INDEX="${3:?shard index}"
    TOTAL="${4:?shard total}"
    OUT_LCOV="${5:?output LCOV path}"
    mkdir -p "$(dirname "$OUT_LCOV")"
    cd "$REPO_ROOT"
    # Instrumented builds through plain `cargo test`, so the shard script
    # partitions exactly as in CI. The `ci` profile (no LTO, 16 codegen
    # units) replaces `--release` (fat LTO, one codegen unit).
    eval "$(cargo llvm-cov show-env --sh)"
    cargo llvm-cov clean --workspace
    if [ "$PACKAGE" = "rest" ]; then
      SKIP_ARGS=()
      for name in "${RUST_SKIPS[@]}"; do SKIP_ARGS+=(--skip "$name"); done
      cargo test --workspace --exclude flexmesa-py --exclude flexmesa-star \
        --exclude flexmesa-astero --exclude flexmesa-binary --profile ci \
        -- "${SKIP_ARGS[@]}"
    else
      SHARD_SKIPS=()
      for name in "${RUST_SKIPS[@]}"; do SHARD_SKIPS+=(--skip "$name"); done
      python3 scripts/ci/cargo_test_shard.py --package "$PACKAGE" \
        --index "$INDEX" --total "$TOTAL" --profile ci "${SHARD_SKIPS[@]}"
    fi
    cargo llvm-cov report --profile ci --lcov --output-path "$OUT_LCOV"
    ;;
  python)
    OUT_JSON="${2:?Usage: run_coverage.sh python <output-json>}"
    mkdir -p "$(dirname "$OUT_JSON")"
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
    echo "Unknown coverage mode: $MODE (expected 'rust-shard' or 'python')" >&2
    exit 2
    ;;
esac
