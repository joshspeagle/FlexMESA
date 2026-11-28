# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the P1.5 Python & JAX inventory (`build_python_inventory`)."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "dashboard" / "build_python_inventory.py"
)

_ALLOWED_KINDS = frozenset(
    {
        "pure-python mirror",
        "PyO3 wrapper",
        "JAX kernel",
        "JAX engine twin (frozen)",
    }
)
_ALLOWED_STATES = frozenset(
    {
        "mirror-current",
        "mirror-stale",
        "python-only",
    }
)


def _load_script_module() -> Any:
    """Load ``scripts/dashboard/build_python_inventory.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "build_python_inventory", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_path_line(
    repo_root: pathlib.Path,
    path_line: str,
    line_counts: dict[str, int],
) -> None:
    """Assert ``<relpath>:<line>`` resolves to a valid line in the repo."""
    rel, line_str = path_line.rsplit(":", 1)
    line_no = int(line_str)
    if rel not in line_counts:
        target = repo_root / rel
        assert target.is_file(), f"missing file referenced by {path_line}"
        line_counts[rel] = len(target.read_text(encoding="utf-8").splitlines())
    assert 1 <= line_no <= line_counts[rel], (
        f"line out of range in {path_line} (max {line_counts[rel]})"
    )


def test_committed_inventory_summary_and_invariants() -> None:
    """Committed JSON and Markdown inventory tables satisfy all invariants."""
    mod = _load_script_module()
    json_path = _REPO_ROOT / mod.JSON_RELPATH
    md_path = _REPO_ROOT / mod.MD_RELPATH
    assert json_path.is_file()
    assert md_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    summary: list[dict[str, Any]] = data["summary_by_subpackage"]
    totals: dict[str, Any] = data["totals"]
    py_items: list[dict[str, Any]] = data["python_items"]
    rust_only: list[dict[str, Any]] = data["rust_only_items"]
    xfails: list[dict[str, Any]] = data["xfail_295_tests"]
    oldest_stale: list[dict[str, Any]] = data["oldest_mirror_stale"]

    assert "git_head" not in data["header"]

    sum_cols = (
        "python_total",
        "pure_python_mirror",
        "pyo3_wrapper",
        "jax_kernel",
        "jax_engine_twin_frozen",
        "mirror_current",
        "mirror_stale",
        "python_only",
        "rust_only",
        "xfail_295",
    )
    for row in summary:
        kind_sum = (
            row["pure_python_mirror"]
            + row["pyo3_wrapper"]
            + row["jax_kernel"]
            + row["jax_engine_twin_frozen"]
        )
        state_sum = (
            row["mirror_current"] + row["mirror_stale"] + row["python_only"]
        )
        assert kind_sum == row["python_total"] == state_sum

    assert totals["subpackage"] == "TOTAL"
    for col in sum_cols:
        assert totals[col] == sum(int(r[col]) for r in summary)

    assert len(py_items) == totals["python_total"]
    assert len(rust_only) == totals["rust_only"]
    assert len(xfails) == totals["xfail_295"]
    assert totals["python_total"] > 0
    assert totals["xfail_295"] >= 19

    line_counts: dict[str, int] = {}
    stale_rows: list[dict[str, Any]] = []
    for item in py_items:
        assert item["kind"] in _ALLOWED_KINDS
        assert item["state"] in _ALLOWED_STATES
        _check_path_line(_REPO_ROOT, item["python_path_line"], line_counts)
        if item["state"] == "python-only":
            assert item["rust_path_line"] == ""
            assert item["rust_path"] == ""
        else:
            assert item["rust_path_line"] != ""
            _check_path_line(_REPO_ROOT, item["rust_path_line"], line_counts)

        if item["kind"] == "PyO3 wrapper":
            assert item["state"] == "mirror-current"
        if item["state"] == "mirror-stale":
            assert item["kind"] != "PyO3 wrapper"
            assert item["rs_last_ts"] > item["py_last_ts"]
            assert item["rs_last_sha"] != item["py_last_sha"]
            stale_rows.append(item)

    assert len(stale_rows) == totals["mirror_stale"]
    expected_oldest = sorted(
        stale_rows,
        key=lambda r: (r["py_last_ts"], r["rs_last_ts"], r["python_path_line"]),
    )[:20]
    assert oldest_stale == expected_oldest

    for ro in rust_only:
        assert ro["state"] == "rust-only"
        _check_path_line(_REPO_ROOT, ro["rust_path_line"], line_counts)

    for xf in xfails:
        _check_path_line(_REPO_ROOT, xf["test_ref"], line_counts)
        assert "#295" in xf["reason"]

    md_text = md_path.read_text(encoding="utf-8")
    expected_total_row = (
        f"| **TOTAL** | **{totals['python_total']}** | "
        f"**{totals['pure_python_mirror']}** | **{totals['pyo3_wrapper']}** | "
        f"**{totals['jax_kernel']}** | "
        f"**{totals['jax_engine_twin_frozen']}** | "
        f"**{totals['mirror_current']}** | **{totals['mirror_stale']}** | "
        f"**{totals['python_only']}** | **{totals['rust_only']}** | "
        f"**{totals['xfail_295']}** |"
    )
    assert expected_total_row in md_text


def _git(
    repo: pathlib.Path,
    *args: str,
    commit_date: str | None = None,
) -> None:
    """Run a deterministic ``git`` command inside ``repo``."""
    env = os.environ.copy()
    if commit_date is not None:
        env["GIT_AUTHOR_DATE"] = commit_date
        env["GIT_COMMITTER_DATE"] = commit_date
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )


def test_stale_detection_and_check_in_synthetic_git_repo(
    tmp_path: pathlib.Path,
) -> None:
    """Staleness tracking and ``--check`` work on a synthetic 2-commit repo."""
    mod = _load_script_module()

    def write(rel: str, text: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    write(
        "src/flexmesa/kap/__init__.py",
        '__all__ = ["fast_wrap", "jax_eval_jax", "py_only_fn", "pure_mirror"]\n'
        "from flexmesa.kap.kernels import (\n"
        "    fast_wrap,\n"
        "    jax_eval_jax,\n"
        "    pure_mirror,\n"
        "    py_only_fn,\n"
        ")\n",
    )
    write(
        "src/flexmesa/kap/kernels.py",
        "from flexmesa import _rust\n\n"
        "def pure_mirror(x: float) -> float:\n"
        "    return x + 1.0\n\n"
        "def jax_eval_jax(x: float) -> float:\n"
        "    return x + 2.0\n\n"
        "def fast_wrap(x: float) -> float:\n"
        "    return float(_rust.kap_fast(x))\n\n"
        "def py_only_fn(x: float) -> float:\n"
        "    return x\n",
    )
    write(
        "crates/flexmesa-kap/src/lib.rs",
        "pub fn pure_mirror(x: f64) -> f64 {\n"
        "    x + 1.0\n"
        "}\n\n"
        "pub fn jax_eval(x: f64) -> f64 {\n"
        "    x + 2.0\n"
        "}\n\n"
        "pub fn fast_impl(x: f64) -> f64 {\n"
        "    x * 2.0\n"
        "}\n\n"
        "pub fn unexposed_rust_fn(x: f64) -> f64 {\n"
        "    x - 1.0\n"
        "}\n",
    )
    write(
        "crates/flexmesa-py/src/kap_py.rs",
        "use flexmesa_kap::fast_impl;\n\n"
        "#[pyfunction]\n"
        "fn kap_fast(x: f64) -> f64 {\n"
        "    fast_impl(x)\n"
        "}\n",
    )
    write(
        "tests/test_kap_sync.py",
        "import pytest\n\n"
        '@pytest.mark.xfail(reason="#295: synthetic mismatch")\n'
        "def test_kap_jax_parity() -> None:\n"
        "    _ = jax_eval_jax(1.0)\n",
    )

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "add", ".")
    _git(
        tmp_path,
        "commit",
        "-m",
        "initial",
        commit_date="2026-11-01T00:00:00Z",
    )

    # Modify pure_mirror and fast_impl in Rust at a later timestamp.
    write(
        "crates/flexmesa-kap/src/lib.rs",
        "pub fn pure_mirror(x: f64) -> f64 {\n"
        "    x + 1.5\n"
        "}\n\n"
        "pub fn jax_eval(x: f64) -> f64 {\n"
        "    x + 2.0\n"
        "}\n\n"
        "pub fn fast_impl(x: f64) -> f64 {\n"
        "    x * 3.0\n"
        "}\n\n"
        "pub fn unexposed_rust_fn(x: f64) -> f64 {\n"
        "    x - 1.0\n"
        "}\n",
    )
    _git(tmp_path, "add", "crates/flexmesa-kap/src/lib.rs")
    _git(
        tmp_path,
        "commit",
        "-m",
        "update rust kernels",
        commit_date="2026-11-10T00:00:00Z",
    )

    assert mod.main(["--repo-root", str(tmp_path)]) == 0
    assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 0

    inv = json.loads((tmp_path / mod.JSON_RELPATH).read_text(encoding="utf-8"))
    by_name = {r["python_name"]: r for r in inv["python_items"]}
    assert by_name["pure_mirror"]["kind"] == "pure-python mirror"
    assert by_name["pure_mirror"]["state"] == "mirror-stale"
    assert by_name["fast_wrap"]["kind"] == "PyO3 wrapper"
    assert by_name["fast_wrap"]["state"] == "mirror-current"
    assert by_name["jax_eval_jax"]["kind"] == "JAX kernel"
    assert by_name["jax_eval_jax"]["state"] == "mirror-current"
    assert by_name["jax_eval_jax"]["xfail_295"] == [
        "tests/test_kap_sync.py:4::test_kap_jax_parity"
    ]
    assert by_name["py_only_fn"]["state"] == "python-only"
    assert inv["totals"]["rust_only"] == 1
    assert inv["rust_only_items"][0]["rust_name"] == "unexposed_rust_fn"
    assert inv["totals"]["xfail_295"] == 1


# The byte-for-byte regeneration (`build_python_inventory.py --check`) is not a
# CI test: git timestamps and line numbers move with every engine PR, so it runs
# on demand (nothing in CI calls it). The structural invariants above stay
# in CI.
