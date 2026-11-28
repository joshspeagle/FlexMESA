# SPDX-License-Identifier: GPL-3.0-or-later
"""Automated tests for ``scripts/docs/check_api_coverage.py`` (P1.4)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKER_PATH = REPO_ROOT / "scripts" / "docs" / "check_api_coverage.py"


def _load_checker() -> ModuleType:
    """Import ``check_api_coverage.py`` without path side effects."""
    spec = importlib.util.spec_from_file_location(
        "check_api_coverage", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_api_coverage"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_synthetic_repo(root: pathlib.Path, *, documented: bool) -> None:
    """Create a synthetic repo exercising Python and PyO3 API doc checks."""
    upkeep = root / ".claude" / "rules" / "docs-upkeep.md"
    upkeep.parent.mkdir(parents=True, exist_ok=True)
    upkeep.write_text(
        "| `flexmesa-eos` | `eos` | c | t | `api/eos.md` | b |\n",
        encoding="utf-8",
    )

    py_pkg = root / "src" / "flexmesa" / "eos"
    py_pkg.mkdir(parents=True, exist_ok=True)
    (py_pkg / "__init__.py").write_text(
        "from flexmesa.eos import core\n"
        "eval_eos = core.eval_eos\n"
        '__all__ = ["eval_eos", "extra_py_sym"]\n',
        encoding="utf-8",
    )
    (py_pkg / "core.py").write_text(
        "def eval_eos() -> float:\n    return 1.0\n",
        encoding="utf-8",
    )

    (root / "src" / "flexmesa" / "_rust.pyi").write_text(
        "def eos_rust_fn(x: float) -> float: ...\n",
        encoding="utf-8",
    )

    py_src = root / "crates" / "flexmesa-py" / "src"
    py_src.mkdir(parents=True, exist_ok=True)
    (py_src / "lib.rs").write_text("pub mod eos_py;\n", encoding="utf-8")
    (py_src / "eos_py.rs").write_text(
        "fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {\n"
        "    m.add_function(wrap_pyfunction!(eos_rust_fn, m)?)?;\n"
        "    Ok(())\n"
        "}\n",
        encoding="utf-8",
    )

    api_md = root / "docs" / "api" / "eos.md"
    api_md.parent.mkdir(parents=True, exist_ok=True)
    if documented:
        api_md.write_text(
            "::: flexmesa.eos.core\n\n"
            "- **`extra_py_sym`**: Extra Python symbol.\n"
            "- **`eos_rust_fn`**: Rust bridge function.\n",
            encoding="utf-8",
        )
    else:
        api_md.write_text("::: flexmesa.eos.core\n", encoding="utf-8")


def test_docs_api_coverage_check_passes_on_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Repository API documentation satisfies ``--check`` cleanly."""
    mod = _load_checker()
    code = mod.main(["--repo-root", str(REPO_ROOT), "--check"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "API coverage gate passed" in captured.out


def test_docs_api_coverage_report_lists_all_api_pages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--report`` succeeds and prints per-page coverage totals."""
    mod = _load_checker()
    code = mod.main(["--repo-root", str(REPO_ROOT), "--report"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    for page in (
        "docs/api/astero.md",
        "docs/api/binary.md",
        "docs/api/eos.md",
        "docs/api/index.md",
        "docs/api/io-plotting-bench.md",
        "docs/api/nuclear.md",
        "docs/api/numerics.md",
        "docs/api/opacity-atm-colors.md",
        "docs/api/star.md",
    ):
        assert page in captured.out


def test_docs_api_coverage_catches_missing_and_stale_entries(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checker fails on undocumented symbols and on stale allowlist entries."""
    mod = _load_checker()
    _write_synthetic_repo(tmp_path, documented=False)

    # Seed empty allowlist first -> missing extra_py_sym and eos_rust_fn fail.
    allow_path = tmp_path / "docs" / "api_coverage_allowlist.toml"
    allow_path.write_text(
        "[counts]\npython = 0\nrust_pyi = 0\ntotal = 0\n", encoding="utf-8"
    )
    assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 1
    err_out = capsys.readouterr().err
    assert "extra_py_sym" in err_out
    assert "eos_rust_fn" in err_out

    # --write-allowlist seeds missing items so --check passes.
    assert mod.main(["--repo-root", str(tmp_path), "--write-allowlist"]) == 0
    assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 0
    capsys.readouterr()

    # Documenting all symbols without updating allowlist fails as stale.
    _write_synthetic_repo(tmp_path, documented=True)
    assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 1
    stale_err = capsys.readouterr().err
    assert "STALE_ALLOWLIST" in stale_err

    # Re-writing the allowlist shrinks it to 0 and restores exit 0.
    assert mod.main(["--repo-root", str(tmp_path), "--write-allowlist"]) == 0
    assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 0


def test_docs_api_coverage_ignores_prose_mentions(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A whole-word mention in prose or a snippet does not document a symbol."""
    mod = _load_checker()
    _write_synthetic_repo(tmp_path, documented=False)
    api_md = tmp_path / "docs" / "api" / "eos.md"
    api_md.write_text(
        "::: flexmesa.eos.core\n\n"
        "Prose names extra_py_sym and eos_rust_fn but has no bullet.\n\n"
        "```python\nextra_py_sym()\neos_rust_fn(1.0)\n```\n",
        encoding="utf-8",
    )
    allow_path = tmp_path / "docs" / "api_coverage_allowlist.toml"
    allow_path.write_text(
        "[counts]\npython = 0\nrust_pyi = 0\ntotal = 0\n", encoding="utf-8"
    )
    assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 1
    err_out = capsys.readouterr().err
    assert "UNDOCUMENTED" in err_out
    assert "extra_py_sym" in err_out
    assert "eos_rust_fn" in err_out
