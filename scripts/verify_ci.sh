#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Clean-room local pre-PR verification gate mirroring `.github/workflows/ci.yml`
# and `.github/workflows/docs.yml`.
#
# Executes from an isolated temporary working directory so relative repository
# caches cannot mask missing data or test dependencies, and runs all CI gates
# (`rust`, `docs` + static analysis, `python-jax`, and `e2e`) with
# `FLEXMESA_DATA_DIR`, `MESA_DIR`, `FLEXMESA_REQUIRE_ZERO_SKIPS=1`, and
# `JAX_PLATFORMS=cpu` configured identically to GitHub Actions.

set -euo pipefail

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
PYTEST_BIN="${PYTEST_BIN:-$(dirname "$PYTHON_BIN")/pytest}"
MKDOCS_BIN="${MKDOCS_BIN:-$(dirname "$PYTHON_BIN")/mkdocs}"
MATURIN_BIN="${MATURIN_BIN:-$(dirname "$PYTHON_BIN")/maturin}"
RUFF_BIN="${RUFF_BIN:-$(command -v ruff)}"
MYPY_BIN="${MYPY_BIN:-$(dirname "$PYTHON_BIN")/mypy}"

if [[ ! -x "$PYTEST_BIN" ]]; then
  PYTEST_BIN="$(command -v pytest)"
fi
if [[ ! -x "$MKDOCS_BIN" ]]; then
  MKDOCS_BIN="$(command -v mkdocs)"
fi
if [[ ! -x "$MATURIN_BIN" ]]; then
  MATURIN_BIN="$(command -v maturin)"
fi
if [[ ! -x "$MYPY_BIN" ]]; then
  MYPY_BIN="$(command -v mypy)"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export FLEXMESA_REQUIRE_ZERO_SKIPS=1

DATA_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/flexmesa"
export FLEXMESA_DATA_DIR="${FLEXMESA_DATA_DIR:-$DATA_CACHE_ROOT/data}"
export MESA_DIR="${MESA_DIR:-$DATA_CACHE_ROOT/mesa_test_inputs}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/flexmesa-jax}"

echo "[verify_ci] 1/5 Verifying data-v1 physics & MESA test bundles..."
"$PYTHON_BIN" "$REPO_ROOT/src/flexmesa/data/fetch.py" fetch --include-test-inputs

CLEAN_CWD="$(mktemp -d)"
SITE_OUT=""
cleanup() {
  "$PYTHON_BIN" -c '
import shutil, sys
for p in sys.argv[1:]:
    if p:
        shutil.rmtree(p, ignore_errors=True)
' "$CLEAN_CWD" "$SITE_OUT" 2>/dev/null || true
}
trap cleanup EXIT
cd "$CLEAN_CWD"

echo "[verify_ci] 2/5 Running Cargo Clippy, rustdoc, Safe Rust tests & maturin build..."
cargo clippy --manifest-path "$REPO_ROOT/Cargo.toml" --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path "$REPO_ROOT/Cargo.toml" --workspace --no-deps --exclude flexmesa-py
cargo test --manifest-path "$REPO_ROOT/Cargo.toml" --workspace --exclude flexmesa-py --release
cargo test --manifest-path "$REPO_ROOT/Cargo.toml" --workspace --exclude flexmesa-py --doc
PYO3_PYTHON="$PYTHON_BIN" "$MATURIN_BIN" develop --release --manifest-path "$REPO_ROOT/crates/flexmesa-py/Cargo.toml"

echo "[verify_ci] 3/5 Running Ruff, Mypy (--strict), figures, MkDocs (--strict) & snippet checks..."
"$RUFF_BIN" format --check --line-length 80 "$REPO_ROOT/src" "$REPO_ROOT/tests"
"$RUFF_BIN" check "$REPO_ROOT/src" "$REPO_ROOT/tests"
"$MYPY_BIN" --python-executable "$PYTHON_BIN" --config-file "$REPO_ROOT/pyproject.toml" --strict "$REPO_ROOT/src" "$REPO_ROOT/tests"
"$PYTHON_BIN" "$REPO_ROOT/scripts/build_docs_figures.py"
SITE_OUT="$(mktemp -d)"
"$MKDOCS_BIN" build --config-file "$REPO_ROOT/mkdocs.yml" --strict --site-dir "$SITE_OUT"
"$PYTHON_BIN" -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$SITE_OUT"
"$PYTEST_BIN" -c "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/tests/test_docs_snippets.py" \
  "$REPO_ROOT/tests/test_mesa_paper_figures.py" -v

echo "[verify_ci] 4/5 Running Python, PyO3 & JAX parity suite (zero skips)..."
"$PYTEST_BIN" -c "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/src" \
  "$REPO_ROOT/tests" \
  --ignore="$REPO_ROOT/tests/test_docs_snippets.py" \
  --ignore="$REPO_ROOT/tests/test_mesa_paper_figures.py" \
  --ignore-glob="$REPO_ROOT/tests/test_e2e_*.py" -v

echo "[verify_ci] 5/5 Running upstream MESA Tier 1-5 & dev_cases E2E suites (zero skips)..."
"$PYTEST_BIN" -c "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT"/tests/test_e2e_*.py -v

echo "[verify_ci] ALL CI GATES PASSED."
