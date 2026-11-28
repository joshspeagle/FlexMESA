# SPDX-License-Identifier: GPL-3.0-or-later
"""W4 test-suite classification and per-routine MESA oracle mapping (`#563`).

Classifies every Rust integration test and Python test file into one of three
mutually exclusive categories per `DECISIONS #67` and `DECISIONS #72`:

- ``oracle``: compares FlexMESA outputs against MESA ``26.4.1`` reference data
  (hex dumps, ``test_output`` fixtures, Fortran oracle TSVs, MESA ``26.4.1``
  histories, or unmodified ``*.defaults`` tables).
- ``invariant``: checks algebraic properties, conservation laws, AD
  consistency, error handling, input validation, or Rust-vs-JAX parity.
- ``regression``: checks against self-generated FlexMESA regression snapshots
  or determinism without an external MESA oracle.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
import tomllib
from dataclasses import dataclass
from typing import Any, Literal, cast

from flexmesa.testing import (
    _oracle_provenance,
    _routine_call_gate,
    port_manifest,
)

DEFAULT_TAXONOMY_REL: str = "benchmarks/dashboard/test_taxonomy.toml"

TestKind = Literal["oracle", "invariant", "regression"]
TestLanguage = Literal["rust", "python"]

VALID_KINDS: tuple[TestKind, ...] = ("oracle", "invariant", "regression")
VALID_LANGUAGES: tuple[TestLanguage, ...] = ("rust", "python")
VALID_MODULES: tuple[str, ...] = (*port_manifest.MODULE_ORDER, "tooling")

RESTRICTED_ORACLE_TOKENS: tuple[str, ...] = (
    "golden",
    "parity",
    "fortran",
    "reference",
)

PARSER_UNIT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "crates/flexmesa-testing/tests/golden_tests.rs",
        "tests/test_golden.py",
    }
)

_ISSUE_RE = re.compile(r"#\d+")
_RUST_COMMENT_RE = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_RUST_TEST_ATTR_RE = re.compile(
    r"#\[\s*(?:core::prelude::\w+::|tokio::)?test(?:\s*\([^\)]*\))?\s*\]"
)
_RUST_TEST_KIND_RE = re.compile(
    r"^\s*//!?\s*@test-kind:\s*(oracle|invariant|regression)\s*$",
    flags=re.MULTILINE,
)
_HEX64_RE = re.compile(r"0[xX][0-9a-fA-F][0-9a-fA-F_]{11,}")
_FN_TEST_DEF_RE = re.compile(r"\b(fn|def)\s+test_[A-Za-z0-9_]+\b")
_RUST_MOD_COMMON_RE = re.compile(r"\bmod\s+common\s*;")
_MESA_PKGS = (
    "adipls|astero|atm|auto_diff|binary|chem|colors|const|eos|gyre|"
    "interp_1d|interp_2d|ionization|kap|math|mtx|net|neu|num|rates|"
    "star|star_data|turb|utils"
)
_REPO_ORACLE_PATH_RE = re.compile(
    r"(?:benchmarks/(?:fortran_mesa|history_harness/reference_histories|"
    r"module_hex)/[\w./{},\*-]+"
    r"|tests/fixtures/golden/[\w./{},\*-]+"
    r"|crates/flexmesa-[a-z0-9_-]+/tests/data/[\w./{},\*-]+)"
)
_MESA_TEST_INPUTS_ANCHOR_RE = re.compile(
    r"\bmesa_test_inputs\b.*\b(?:defaults|history\.data|inlist|profile|"
    r"test_output)\b|\b(?:defaults|history\.data|inlist|profile|test_output)"
    r"\b.*\bmesa_test_inputs\b"
)
_BUNDLE_ORACLE_ID_RE = re.compile(
    r"\b(?:mesa-module-hex-v1(?:\.tar\.zst)?|mesa-replay-v1(?:\.tar\.zst)?|"
    r"FLEXMESA_ADIPLS_CAPTURES|FLEXMESA_PULSE_TAP_DIR|mesa-astero-tracks)\b"
)
_UPSTREAM_MESA_ANCHOR_RE = re.compile(
    rf"\b(?:{_MESA_PKGS})/"
    r"(?:test/test_output[\w.-]*|test/src/[\w./{},\*-]+|"
    r"test/LOGS/[\w./{},\*-]+|test_suite/[\w./{},\*-]+|"
    r"private/[\w./-]+|public/[\w./-]+|data/[\w./-]+)"
)
_GOLDEN_VERSION_RE = re.compile(
    r"(?:version_number|mesa_version_number|flexmesa_version)",
    flags=re.IGNORECASE,
)
_MESA_ORIGIN_RE = re.compile(
    r"(?:\bmesa\b|\bmesa_[A-Za-z0-9_]+|\br?26\.4\.1\b|\bgfortran\b|\bcrmath\b)",
    flags=re.IGNORECASE,
)
_ORACLE_CODE_SIGNALS_RE = re.compile(
    r"("
    r"benchmarks/fortran_mesa|"
    r"fortran_mesa|"
    r"fixtures/golden|"
    r"module_hex|"
    r"reference_histories|"
    r"mesa-astero-tracks|"
    r"FLEXMESA_ADIPLS_CAPTURES|"
    r"FLEXMESA_PULSE_TAP_DIR|"
    r"resolve_mesa_test_inputs_dir|"
    r"require_mesa_test_inputs_dir|"
    r"evaluate_defaults_parity|"
    r"load_upstream_reference|"
    r"load_golden_outputs|"
    r"parse_golden_file|"
    r"data/pulse_fortran_format|"
    r"data/[\w./-]+\.replay|"
    r"fmt_oracle_expected\.txt|"
    r"tests/data/mesa_26_4_1_|"
    r"mesa_26_4_1_gyre_|"
    r"history\.data|"
    r"test_output|"
    r"zams_model0|"
    r"lina_26_4_1|"
    r"\bGroup1_solar\b|"
    r"(?:^|[^A-Za-z0-9])_?GOLDEN_[A-Z0-9_]+|"
    r"(?:^|[^A-Za-z0-9])_?golden(?:_|_output|_rows|_cases|_pts|_roots|"
    r"_values|_reference|_references|_test_output|_bools|_nums|_cno|"
    r"_table|_sin_root|_value|_baseline|_parity|_direct|_logs|_powers|"
    r"_inverse|_sections|_test_|_mods|_rsp|_oracle)[A-Za-z0-9_]*|"
    r"match(?:es)?_(?:upstream_|fortran_|mesa_)?golden"
    r")"
)
_PYTEST_MARK_ATTR = "pytestmark"


@dataclass(frozen=True)
class TestFileRecord:
    """Single classified test file entry in `test_taxonomy.toml`."""

    path: str
    language: TestLanguage
    module: str
    kind: TestKind
    provenance: str = ""
    routines: tuple[str, ...] = ()
    mislabel_reason: str = ""
    oracles: tuple[_oracle_provenance.OracleEntry, ...] = ()

    @property
    def oracle_routines(self) -> tuple[str, ...]:
        """Routine ids claimed by this record's ``[[tests.oracles]]``."""
        seen: dict[str, None] = {}
        for entry in self.oracles:
            for rid in entry.routines:
                seen.setdefault(rid, None)
        return tuple(seen)


@dataclass(frozen=True)
class TestTaxonomy:
    """Parsed `test_taxonomy.toml` catalog."""

    schema_version: int
    entries: tuple[TestFileRecord, ...]

    def by_path(self) -> dict[str, TestFileRecord]:
        """Return a mapping from repository-relative path to record."""
        return {entry.path: entry for entry in self.entries}


def _scan_rust_code(text: str, *, strip_strings: bool) -> str:
    """Strip Rust comments (and optionally string literals) safely."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            nl = text.find("\n", i + 2)
            i = n if nl == -1 else nl
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif text[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            out.append(" ")
            i = j
            continue
        # Skip Rust char literals like `'"'` so they do not start strings.
        if ch == "'" and i + 2 < n:
            if text[i + 1] == "\\":
                end_q = text.find("'", i + 2)
                if (
                    end_q != -1
                    and end_q - i <= 10
                    and "\n" not in text[i:end_q]
                ):
                    out.append(" " if strip_strings else text[i : end_q + 1])
                    i = end_q + 1
                    continue
            elif text[i + 1] != "'" and text[i + 2] == "'":
                out.append(" " if strip_strings else text[i : i + 3])
                i += 3
                continue
        if ch in ("r", "b", "c") and (
            i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        ):
            k = i + 1
            if ch in ("b", "c") and k < n and text[k] == "r":
                k += 1
            if (ch == "r" or k > i + 1) and k < n:
                hashes = 0
                while k < n and text[k] == "#":
                    hashes += 1
                    k += 1
                if k < n and text[k] == '"':
                    closing = '"' + ("#" * hashes)
                    end = text.find(closing, k + 1)
                    j = n if end == -1 else end + len(closing)
                    out.append(" " if strip_strings else text[i:j])
                    i = j
                    continue
        if ch == '"' or (
            ch in ("b", "c")
            and i + 1 < n
            and text[i + 1] == '"'
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
        ):
            q = i if ch == '"' else i + 1
            j = q + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(" " if strip_strings else text[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_rust_comments(text: str) -> str:
    """Strip Rust `//` and nested `/* ... */` comments outside literals."""
    return _scan_rust_code(text, strip_strings=False)


def _ast_div_path_parts(expr: ast.expr) -> list[str] | None:
    """Return joined constant path parts for `base / 'a' / 'b'` chains."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        if isinstance(expr.right, ast.Constant) and isinstance(
            expr.right.value, str
        ):
            left_parts = _ast_div_path_parts(expr.left)
            if left_parts is not None:
                return [*left_parts, expr.right.value]
            return [expr.right.value]
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    return None


def _strip_python_comments_and_docstrings(
    text: str,
    *,
    strip_strings: bool = False,
) -> str:
    """Strip Python `#` comments, docstrings, and optional string literals."""
    tree = ast.parse(text)
    joined_paths: list[str] = []
    if not strip_strings:
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                parts = _ast_div_path_parts(node)
                if parts and len(parts) >= 2:
                    joined_paths.append("/".join(parts))
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body[0].value.value = ""
        if (
            strip_strings
            and isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        ):
            node.value = ""
    lines: list[str] = []
    for raw in ast.unparse(tree).splitlines():
        stripped = raw.lstrip()
        if not stripped.startswith("#"):
            lines.append(raw)
    if joined_paths:
        lines.extend(repr(p) for p in joined_paths)
    return "\n".join(lines)


def discover_test_files(repo_root: pathlib.Path) -> list[str]:
    """Discover all Rust integration and Python test files deterministically.

    Args:
        repo_root: Repository root directory (follows top-level symlinks).

    Returns:
        Sorted list of POSIX paths relative to ``repo_root``.
    """
    discovered: list[str] = []
    crates_dir = repo_root / "crates"
    if crates_dir.exists():
        resolved_crates = crates_dir.resolve()
        for crate_dir in sorted(resolved_crates.iterdir()):
            if not crate_dir.is_dir():
                continue
            tests_dir = (crate_dir / "tests").resolve()
            if not tests_dir.is_dir():
                continue
            for rs_file in sorted(tests_dir.rglob("*.rs")):
                if not rs_file.is_file():
                    continue
                rel = rs_file.relative_to(tests_dir)
                parts = rel.parts
                if "common" in parts or rs_file.name == "mod.rs":
                    continue
                discovered.append(
                    f"crates/{crate_dir.name}/tests/{rel.as_posix()}"
                )

    tests_root = repo_root / "tests"
    if tests_root.exists():
        resolved_tests = tests_root.resolve()
        for py_file in sorted(resolved_tests.rglob("*.py")):
            if not py_file.is_file():
                continue
            name = py_file.name
            if not (name.startswith("test_") or name.endswith("_test.py")):
                continue
            rel = py_file.relative_to(resolved_tests)
            discovered.append(f"tests/{rel.as_posix()}")

    return sorted(discovered)


def count_tests_in_file(path: pathlib.Path) -> int:
    """Count declared test functions in a Rust or Python test file.

    Args:
        path: Absolute path to a ``.rs`` or ``.py`` test file.

    Returns:
        Number of test functions found (``#[test]`` or ``proptest!`` `fn`s in
        Rust; top-level and class ``test_*`` functions in Python).
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rs":
        code = _strip_rust_comments(text)
        attr_count = len(_RUST_TEST_ATTR_RE.findall(code))
        proptest_fns = 0
        for match in re.finditer(r"proptest!\s*\{", code):
            depth = 1
            idx = match.end()
            while idx < len(code) and depth > 0:
                if code[idx] == "{":
                    depth += 1
                elif code[idx] == "}":
                    depth -= 1
                idx += 1
            block = code[match.end() : idx]
            proptest_fns += len(re.findall(r"\bfn\s+\w+\s*\(", block))
        return attr_count + proptest_fns

    if path.suffix == ".py":
        tree = ast.parse(text, filename=str(path))
        count = 0
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                count += 1
            elif isinstance(node, ast.ClassDef) and node.name.startswith(
                "Test"
            ):
                for item in node.body:
                    if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and item.name.startswith("test_"):
                        count += 1
        return count

    raise ValueError(f"Unsupported test file extension: {path}")


def _extract_pytest_mark_name(expr: ast.expr) -> str | None:
    """Return marker name if ``expr`` looks like ``pytest.mark.<name>(...)``."""
    target = expr.func if isinstance(expr, ast.Call) else expr
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
    ):
        return target.attr
    return None


def detect_inline_test_kind(
    path: pathlib.Path,
) -> tuple[TestKind | None, str | None]:
    """Detect optional file-level taxonomy marker inside a Rust or Python file.

    Args:
        path: Path to a test file.

    Returns:
        Pair ``(kind, error)``. ``kind`` is ``None`` if no file-level marker is
        present or on conflict; ``error`` describes conflicting markers if any.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rs":
        found = sorted(set(_RUST_TEST_KIND_RE.findall(text)))
        if not found:
            return (None, None)
        if len(found) > 1:
            return (None, f"conflicting @test-kind tags: {found}")
        return (cast(TestKind, found[0]), None)

    if path.suffix == ".py":
        tree = ast.parse(text, filename=str(path))
        kinds: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == _PYTEST_MARK_ATTR
                for t in node.targets
            ):
                continue
            items = (
                list(node.value.elts)
                if isinstance(node.value, (ast.List, ast.Tuple))
                else [node.value]
            )
            for item in items:
                mark = _extract_pytest_mark_name(item)
                if mark in VALID_KINDS:
                    kinds.add(mark)
        if not kinds:
            return (None, None)
        if len(kinds) > 1:
            return (
                None,
                "conflicting module pytestmark taxonomy kinds: "
                f"{sorted(kinds)}",
            )
        return (cast(TestKind, next(iter(kinds))), None)

    return (None, None)


def _read_non_comment_code(
    path: pathlib.Path,
    *,
    strip_strings: bool = False,
    strip_test_fn_names: bool = False,
    include_rust_common: bool = False,
) -> str:
    """Return code with comments and optional strings/test names stripped."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rs":
        code = _scan_rust_code(text, strip_strings=strip_strings)
        if include_rust_common and _RUST_MOD_COMMON_RE.search(code):
            for cand in (
                path.parent / "common" / "mod.rs",
                path.parent / "common.rs",
            ):
                if cand.is_file():
                    code += "\n" + _scan_rust_code(
                        cand.read_text(encoding="utf-8"),
                        strip_strings=strip_strings,
                    )
    elif path.suffix == ".py":
        code = _strip_python_comments_and_docstrings(
            text, strip_strings=strip_strings
        )
    else:
        return ""
    if strip_test_fn_names:
        code = _FN_TEST_DEF_RE.sub(lambda m: f"{m.group(1)} _test_fn", code)
    return code


def file_has_oracle_evidence(path: pathlib.Path) -> bool:
    """Return ``True`` when code cites MESA oracle artifacts or hex tables."""
    code = _read_non_comment_code(
        path,
        strip_strings=False,
        strip_test_fn_names=False,
        include_rust_common=True,
    )
    if not code:
        return False
    if _ORACLE_CODE_SIGNALS_RE.search(code):
        return True
    return bool(_HEX64_RE.search(code))


def _expand_brace_paths(pattern: str) -> list[str]:
    """Expand a single `{a,b,c}` brace group in a repository path pattern."""
    start = pattern.find("{")
    if start == -1:
        return [pattern]
    end = pattern.find("}", start)
    if end == -1:
        return [pattern]
    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    options = [
        opt.strip()
        for opt in pattern[start + 1 : end].split(",")
        if opt.strip()
    ]
    if not options:
        return [pattern]
    return [f"{prefix}{opt}{suffix}" for opt in options]


def _verify_golden_fixture_origin(
    golden_path: pathlib.Path,
    rel_label: str,
) -> str | None:
    """Check header metadata on a `tests/fixtures/golden/` file for origin."""
    if not golden_path.is_file():
        return None
    snippet = "\n".join(
        golden_path.read_text(encoding="utf-8", errors="replace").splitlines()[
            :120
        ]
    )
    low = snippet.lower()
    has_mesa_origin = bool(_MESA_ORIGIN_RE.search(snippet))
    if ("self-generated" in low or "regression snapshot" in low) and (
        not has_mesa_origin
    ):
        return (
            f"cited golden fixture {rel_label!r} looks self-generated "
            "rather than MESA-origin"
        )
    has_version_hdr = bool(_GOLDEN_VERSION_RE.search(snippet))
    has_json_meta = "compiler_executable" in snippet or (
        "total_packages_in_install" in snippet
    )
    if has_version_hdr or has_json_meta:
        # `rsp_*_cycle0.mod` is a starting model at pre-release a108d91.
        is_rsp_cycle0 = golden_path.name.endswith("_cycle0.mod") and (
            "'a108d91'" in snippet
        )
        if not is_rsp_cycle0 and not has_mesa_origin:
            return (
                f"cited golden fixture {rel_label!r} declares version/metadata "
                "without MESA 26.4.1 origin"
            )
    elif snippet.lstrip().startswith("#") and "flexmesa" in low:
        if not has_mesa_origin:
            return (
                f"cited golden fixture {rel_label!r} header mentions FlexMESA "
                "without MESA 26.4.1 origin"
            )
    return None


def _verify_replay_fixture_origin(
    replay_path: pathlib.Path,
    rel_label: str,
) -> str | None:
    """Check header metadata on a cited `.replay` fixture for MESA origin."""
    if not replay_path.is_file():
        return None
    snippet = "\n".join(
        replay_path.read_text(encoding="utf-8", errors="replace").splitlines()[
            :40
        ]
    )
    low = snippet.lower()
    has_mesa_origin = bool(_MESA_ORIGIN_RE.search(snippet))
    if ("self-generated" in low or "regression snapshot" in low) and (
        not has_mesa_origin
    ):
        return (
            f"cited replay fixture {rel_label!r} looks self-generated "
            "rather than MESA-origin"
        )
    if not re.search(
        r"^\s*#\s*mesa_version:\s*r?26\.4\.1\b",
        snippet,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        return (
            f"cited replay fixture {rel_label!r} declares replay data "
            "without MESA 26.4.1 origin"
        )
    return None


def _path_tokens_for_code_check(rel: str) -> tuple[str, ...]:
    """Return concrete path or filename markers expected in test code."""
    clean = rel.rstrip("/")
    pure = pathlib.PurePosixPath(clean)
    name = pure.name
    tokens: list[str] = [clean]
    if "*" not in clean and "?" not in clean:
        tokens.append(name)
        if len(pure.parts) >= 2:
            tokens.append(f"{pure.parent.name}/{name}")
        if pure.suffix in {".tsv", ".txt", ".data", ".mod"}:
            tokens.append(pure.stem)
        if clean.startswith("crates/") and "/tests/data/" in clean:
            after_tests = clean.split("/tests/", 1)[1]
            tokens.append(after_tests)
            tokens.append(pure.name)
    else:
        parent = clean
        for idx, part in enumerate(pure.parts):
            if "*" in part or "?" in part:
                parent = "/".join(pure.parts[:idx])
                stem_prefix = part.split("*", 1)[0].split("?", 1)[0]
                stem_suffix = part.rsplit("*", 1)[-1].rsplit("?", 1)[-1]
                if stem_prefix and len(stem_prefix) >= 4:
                    tokens.append(stem_prefix)
                if stem_suffix and len(stem_suffix) >= 5:
                    tokens.append(stem_suffix)
                break
        if parent:
            tokens.append(parent)
            if len(pathlib.PurePosixPath(parent).parts) >= 2:
                p_pure = pathlib.PurePosixPath(parent)
                tokens.append(f"{p_pure.parent.name}/{p_pure.name}")
    if clean.startswith("tests/fixtures/golden/"):
        sub = clean[len("tests/fixtures/golden/") :]
        pkg = sub.split("/", 1)[0]
        if pkg:
            tokens.extend(
                (
                    f"golden/{pkg}",
                    f"golden_{pkg}",
                    "load_golden_outputs",
                    "parse_golden_file",
                )
            )
        if name == "test_output.txt":
            tokens.extend(("test_output.txt", "test_output", "GOLDEN"))
    return tuple(dict.fromkeys(t for t in tokens if t))


def validate_oracle_provenance(
    repo_root: pathlib.Path,
    rec: TestFileRecord,
    full_path: pathlib.Path,
) -> list[str]:
    """Validate MESA-origin provenance artifact, existence, and code linkage."""
    prov = rec.provenance.strip()
    errors: list[str] = []
    if not prov:
        errors.append(
            f"{rec.path}: kind='oracle' requires non-empty provenance"
        )
        if not file_has_oracle_evidence(full_path):
            errors.append(
                f"{rec.path}: kind='oracle' lacks code-level MESA oracle "
                "evidence"
            )
        return errors

    has_anchor = bool(
        _REPO_ORACLE_PATH_RE.search(prov)
        or _BUNDLE_ORACLE_ID_RE.search(prov)
        or _MESA_TEST_INPUTS_ANCHOR_RE.search(prov)
        or _UPSTREAM_MESA_ANCHOR_RE.search(prov)
    )
    if not has_anchor:
        errors.append(
            f"{rec.path}: kind='oracle' provenance must cite a concrete "
            "MESA 26.4.1 artifact path, bundle ID, or upstream <pkg>/... anchor"
        )
    code = _read_non_comment_code(
        full_path,
        strip_strings=False,
        strip_test_fn_names=True,
        include_rust_common=True,
    )

    for raw_match in _REPO_ORACLE_PATH_RE.findall(prov):
        cleaned = raw_match.rstrip(".,;:")
        expanded = _expand_brace_paths(cleaned)
        matched_any_in_code = False
        for rel in expanded:
            is_dir_ref = rel.endswith("/")
            rel_no_slash = rel.rstrip("/")
            golden_files_to_check: list[tuple[pathlib.Path, str]] = []
            replay_files_to_check: list[tuple[pathlib.Path, str]] = []
            if "*" in rel_no_slash or "?" in rel_no_slash:
                parts = pathlib.PurePosixPath(rel_no_slash).parts
                split_idx = next(
                    i
                    for i, part in enumerate(parts)
                    if "*" in part or "?" in part
                )
                base_rel = "/".join(parts[:split_idx])
                sub_glob = "/".join(parts[split_idx:])
                base_dir = repo_root / base_rel
                if not base_dir.is_dir():
                    errors.append(
                        f"{rec.path}: cited provenance directory "
                        f"{base_rel!r} does not exist on disk"
                    )
                else:
                    matches = sorted(base_dir.glob(sub_glob))
                    if not matches:
                        errors.append(
                            f"{rec.path}: cited provenance pattern {rel!r} "
                            "matches no files on disk"
                        )
                    else:
                        for m in matches:
                            if rel_no_slash.startswith(
                                "tests/fixtures/golden/"
                            ):
                                golden_files_to_check.append((m, rel))
                            if m.suffix == ".replay":
                                replay_files_to_check.append((m, rel))
            else:
                target = repo_root / rel_no_slash
                if not target.exists():
                    errors.append(
                        f"{rec.path}: cited provenance path {rel!r} "
                        "does not exist on disk"
                    )
                else:
                    if rel_no_slash.startswith("tests/fixtures/golden/"):
                        if target.is_file():
                            golden_files_to_check.append((target, rel))
                        elif target.is_dir() and is_dir_ref:
                            for f in sorted(target.iterdir()):
                                if f.is_file():
                                    golden_files_to_check.append((f, rel))
                    if target.is_file() and target.suffix == ".replay":
                        replay_files_to_check.append((target, rel))
                    elif target.is_dir() and is_dir_ref:
                        for f in sorted(target.iterdir()):
                            if f.is_file() and f.suffix == ".replay":
                                replay_files_to_check.append((f, rel))

            for g_file, g_label in golden_files_to_check:
                g_err = _verify_golden_fixture_origin(g_file, g_label)
                if g_err:
                    errors.append(f"{rec.path}: {g_err}")
            for r_file, r_label in replay_files_to_check:
                r_err = _verify_replay_fixture_origin(r_file, r_label)
                if r_err:
                    errors.append(f"{rec.path}: {r_err}")

            if any(tok in code for tok in _path_tokens_for_code_check(rel)):
                matched_any_in_code = True

        if not matched_any_in_code:
            errors.append(
                f"{rec.path}: non-comment test code does not reference cited "
                f"provenance artifact {cleaned!r}"
            )

    bundle_match = _BUNDLE_ORACLE_ID_RE.search(prov)
    if bundle_match is not None:
        bid = bundle_match.group(0)
        if bid.startswith("mesa-module-hex-v1") and (
            "module_hex" not in code and not _HEX64_RE.search(code)
        ):
            errors.append(
                f"{rec.path}: cites {bid!r} in provenance without referencing "
                "`module_hex` or 64-bit IEEE-754 hex captures in test code"
            )
        elif bid in ("FLEXMESA_ADIPLS_CAPTURES", "FLEXMESA_PULSE_TAP_DIR"):
            if bid not in code:
                errors.append(
                    f"{rec.path}: cites {bid!r} in provenance without "
                    f"referencing `{bid}` in test code"
                )
        elif bid.startswith("mesa-replay-v1") and "replay" not in code:
            errors.append(
                f"{rec.path}: cites {bid!r} in provenance without referencing "
                "`replay` in test code"
            )
        elif bid == "mesa-astero-tracks" and "mesa-astero-tracks" not in code:
            errors.append(
                f"{rec.path}: cites {bid!r} in provenance without referencing "
                "`mesa-astero-tracks` in test code"
            )
    elif _MESA_TEST_INPUTS_ANCHOR_RE.search(prov):
        if not any(
            tok in code
            for tok in (
                "mesa_test_inputs",
                "resolve_mesa_test_inputs_dir",
                "require_mesa_test_inputs_dir",
                "evaluate_defaults_parity",
            )
        ):
            errors.append(
                f"{rec.path}: cites `mesa_test_inputs` in provenance without "
                "loading `mesa_test_inputs` in test code"
            )

    if not file_has_oracle_evidence(full_path):
        errors.append(
            f"{rec.path}: kind='oracle' lacks code-level MESA oracle evidence"
        )
    return errors


def load_test_taxonomy(path: pathlib.Path) -> TestTaxonomy:
    """Parse ``test_taxonomy.toml`` into typed immutable records."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = raw.get("metadata", {})
    schema_version = int(meta.get("schema_version", 1))
    records: list[TestFileRecord] = []
    for item in raw.get("tests", ()):
        rel_path = str(item["path"]).replace("\\", "/")
        language = cast(TestLanguage, str(item["language"]))
        module = str(item["module"])
        kind = cast(TestKind, str(item["kind"]))
        provenance = str(item.get("provenance", ""))
        routines = tuple(str(r) for r in item.get("routines", ()))
        mislabel_reason = str(item.get("mislabel_reason", ""))
        records.append(
            TestFileRecord(
                path=rel_path,
                language=language,
                module=module,
                kind=kind,
                provenance=provenance,
                routines=routines,
                mislabel_reason=mislabel_reason,
                oracles=_oracle_provenance.parse_oracles(item),
            )
        )
    return TestTaxonomy(schema_version=schema_version, entries=tuple(records))


def _toml_quote(value: str) -> str:
    """Serialize a plain string as a basic TOML quoted literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def dump_test_taxonomy(taxonomy: TestTaxonomy) -> str:
    """Render a `TestTaxonomy` in deterministic sorted TOML form."""
    lines: list[str] = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# W4 test-suite classification & per-routine MESA 26.4.1 oracle map.",
        "# Schema enforced by `flexmesa.testing.test_taxonomy` (#563); oracle",
        "# claims are [[tests.oracles]] entries checked per routine by",
        "# `flexmesa.testing._oracle_provenance` (DECISIONS #80).",
        "",
        "[metadata]",
        f"schema_version = {taxonomy.schema_version}",
        "",
    ]
    for entry in sorted(taxonomy.entries, key=lambda r: r.path):
        lines.extend(
            [
                "[[tests]]",
                f"path = {_toml_quote(entry.path)}",
                f"language = {_toml_quote(entry.language)}",
                f"module = {_toml_quote(entry.module)}",
                f"kind = {_toml_quote(entry.kind)}",
            ]
        )
        if entry.provenance:
            lines.append(f"provenance = {_toml_quote(entry.provenance)}")
        if entry.routines:
            if len(entry.routines) == 1:
                lines.append(f"routines = [{_toml_quote(entry.routines[0])}]")
            else:
                lines.append("routines = [")
                for routine in sorted(entry.routines):
                    lines.append(f"  {_toml_quote(routine)},")
                lines.append("]")
        if entry.mislabel_reason:
            lines.append(
                f"mislabel_reason = {_toml_quote(entry.mislabel_reason)}"
            )
        for oracle in entry.oracles:
            lines.append("")
            lines.extend(_dump_oracle(oracle))
        lines.append("")
    return "\n".join(lines)


def _toml_list(key: str, values: tuple[str, ...]) -> list[str]:
    """Render ``key = [...]`` one value per line (inline when single)."""
    if len(values) == 1:
        return [f"{key} = [{_toml_quote(values[0])}]"]
    out = [f"{key} = ["]
    out.extend(f"  {_toml_quote(v)}," for v in values)
    out.append("]")
    return out


def _dump_oracle(oracle: _oracle_provenance.OracleEntry) -> list[str]:
    """Render one ``[[tests.oracles]]`` table."""
    lines = [
        "[[tests.oracles]]",
        f"reference = {_toml_quote(oracle.reference)}",
    ]
    if oracle.recipe:
        lines.append(f"recipe = {_toml_quote(oracle.recipe)}")
    lines.append(f"precision = {_toml_quote(oracle.precision)}")
    if oracle.tolerance:
        lines.append(f"tolerance = {_toml_quote(oracle.tolerance)}")
    if oracle.consts:
        lines.extend(_toml_list("consts", oracle.consts))
    lines.extend(_toml_list("routines", tuple(sorted(oracle.routines))))
    return lines


def _collect_manifest_routine_meta(
    manifest: port_manifest.PortManifest,
) -> dict[str, tuple[str, str, str]]:
    """Return ``{"<upstream>::<routine>": (module, upstream, rust_target)}``."""
    routine_meta: dict[str, tuple[str, str, str]] = {}
    for entry in manifest.entries:
        for upstream_routine, rust_symbol in entry.routine_map:
            rid = f"{entry.upstream_path}::{upstream_routine}"
            routine_meta[rid] = (entry.module, upstream_routine, rust_symbol)
    return routine_meta


def _scan_rust_stripped(text: str) -> str:
    """Strip Rust comments and string literals for symbol inspection."""
    return _scan_rust_code(text, strip_strings=True)


def _verify_routine_claims(
    rec: TestFileRecord,
    full_path: pathlib.Path,
    repo_root: pathlib.Path,
    routine_meta: dict[str, tuple[str, str, str]],
    source_cache: dict[str, dict[str, _routine_call_gate.RustSymbolInfo]],
) -> list[str]:
    """Verify each `routine_map` claim against real call sites and imports."""
    errs: list[str] = []
    rust_ev: _routine_call_gate.RustCallEvidence | None = None
    py_ev: _routine_call_gate.PythonCallEvidence | None = None
    if full_path.is_file():
        if rec.language == "rust":
            code = _read_non_comment_code(
                full_path,
                strip_strings=True,
                strip_test_fn_names=True,
                include_rust_common=False,
            )
            rust_ev = _routine_call_gate.extract_rust_evidence(code)
        elif rec.language == "python":
            py_ev = _routine_call_gate.extract_python_evidence(full_path)
            if not py_ev.imports_rust:
                errs.append(
                    f"{rec.path}: Python test claiming Rust routines must "
                    "import `flexmesa._rust`"
                )

    for routine_id in rec.oracle_routines:
        meta = routine_meta.get(routine_id)
        if meta is None:
            errs.append(
                f"{rec.path}: routine {routine_id!r} not in "
                "port_manifest.toml routine_map"
            )
            continue
        allowed = _routine_call_gate.parse_target_symbols(meta[2])
        if rec.language == "rust" and rust_ev is not None:
            missing_crate, matched = (
                _routine_call_gate.check_rust_routine_claim(
                    rust_ev,
                    repo_root,
                    meta[0],
                    meta[2],
                    _scan_rust_stripped,
                    source_cache,
                )
            )
            if missing_crate is not None:
                errs.append(
                    f"{rec.path}: listed routine {routine_id!r} "
                    f"targets crate {missing_crate!r}, which is "
                    "not referenced in test code"
                )
            if not matched:
                errs.append(
                    f"{rec.path}: listed routine {routine_id!r} "
                    f"(expected call to one of {sorted(allowed)}) not "
                    "called in non-comment test code"
                )
        elif rec.language == "python" and py_ev is not None:
            if not (py_ev.called_syms & allowed):
                errs.append(
                    f"{rec.path}: listed routine {routine_id!r} "
                    f"(expected call to one of {sorted(allowed)}) not "
                    "called in non-comment test code"
                )
    return errs


def _has_restricted_oracle_token(rel_path: str) -> bool:
    """Return ``True`` if basename contains a restricted oracle word."""
    name_lower = pathlib.PurePosixPath(rel_path).name.lower()
    return any(tok in name_lower for tok in RESTRICTED_ORACLE_TOKENS)


def verify_test_taxonomy(
    repo_root: pathlib.Path,
    taxonomy_path: pathlib.Path | None = None,
    manifest_path: pathlib.Path | None = None,
) -> list[str]:
    """Verify bijection, provenance, mislabel tracking, and routine symbols.

    Args:
        repo_root: Repository root path.
        taxonomy_path: Override path to ``test_taxonomy.toml``.
        manifest_path: Override path to ``port_manifest.toml``.

    Returns:
        List of human-readable verification errors (empty when clean).
    """
    tax_p = taxonomy_path or (repo_root / DEFAULT_TAXONOMY_REL)
    if not tax_p.is_file():
        return [f"Missing test taxonomy manifest: {tax_p}"]

    taxonomy = load_test_taxonomy(tax_p)
    errors: list[str] = []

    if taxonomy.schema_version != 1:
        errors.append(
            f"{tax_p}: unsupported schema_version={taxonomy.schema_version}"
        )

    man_p = manifest_path or (
        repo_root / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    routine_meta: dict[str, tuple[str, str, str]] = {}
    if man_p.is_file():
        manifest = port_manifest.load_port_manifest(man_p)
        routine_meta = _collect_manifest_routine_meta(manifest)

    discovered = set(discover_test_files(repo_root))
    seen: dict[str, int] = {}
    source_cache: dict[str, dict[str, _routine_call_gate.RustSymbolInfo]] = {}

    for rec in taxonomy.entries:
        seen[rec.path] = seen.get(rec.path, 0) + 1
        if rec.language not in VALID_LANGUAGES:
            errors.append(f"{rec.path}: invalid language {rec.language!r}")
        expected_lang: TestLanguage | None = None
        if rec.path.startswith("crates/") and rec.path.endswith(".rs"):
            expected_lang = "rust"
        elif rec.path.startswith("tests/") and rec.path.endswith(".py"):
            expected_lang = "python"
        if expected_lang is not None and rec.language != expected_lang:
            errors.append(
                f"{rec.path}: language {rec.language!r} mismatches path"
            )
        if rec.module not in VALID_MODULES:
            errors.append(f"{rec.path}: unknown module {rec.module!r}")
        if rec.kind not in VALID_KINDS:
            errors.append(f"{rec.path}: invalid kind {rec.kind!r}")

        full_path = repo_root / rec.path
        if rec.path in discovered and full_path.is_file():
            inline_kind, inline_err = detect_inline_test_kind(full_path)
            if inline_err:
                errors.append(f"{rec.path}: {inline_err}")
            elif inline_kind is not None and inline_kind != rec.kind:
                errors.append(
                    f"{rec.path}: inline marker {inline_kind!r} disagrees "
                    f"with taxonomy kind {rec.kind!r}"
                )

            if rec.kind == "oracle":
                errors.extend(
                    validate_oracle_provenance(repo_root, rec, full_path)
                )
                for oracle in rec.oracles:
                    errors.extend(
                        _oracle_provenance.structural_errors(rec.path, oracle)
                    )
            elif rec.oracles:
                errors.append(
                    f"{rec.path}: kind={rec.kind!r} cannot claim "
                    "oracle routines"
                )
        if rec.routines:
            errors.append(
                f"{rec.path}: the top-level `routines` list is retired; "
                "claim routines through [[tests.oracles]] entries that "
                "name their MESA reference (DECISIONS #80)"
            )
        if rec.kind == "oracle" or not _has_restricted_oracle_token(rec.path):
            if rec.mislabel_reason.strip():
                errors.append(
                    f"{rec.path}: unexpected mislabel_reason on non-mislabeled "
                    f"entry ({rec.kind!r})"
                )
        elif rec.path not in PARSER_UNIT_ALLOWLIST:
            if not rec.mislabel_reason.strip() or not _ISSUE_RE.search(
                rec.mislabel_reason
            ):
                errors.append(
                    f"{rec.path}: non-oracle filename uses restricted oracle "
                    "vocabulary and requires mislabel_reason citing #<issue>"
                )

        if routine_meta and rec.oracle_routines:
            errors.extend(
                _verify_routine_claims(
                    rec, full_path, repo_root, routine_meta, source_cache
                )
            )

    for dup_path, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"{dup_path}: duplicate entry ({count}x)")

    manifest_paths = set(seen)
    for missing in sorted(discovered - manifest_paths):
        errors.append(
            f"{missing}: missing from {DEFAULT_TAXONOMY_REL} "
            "-- add a [[tests]] entry (oracle/invariant/regression) "
            "per DECISIONS #72"
        )
    for stale in sorted(manifest_paths - discovered):
        errors.append(
            f"{stale}: present in {DEFAULT_TAXONOMY_REL} but missing on disk "
            "-- remove or update its [[tests]] entry per DECISIONS #72"
        )

    return errors


def summarize_test_taxonomy(
    repo_root: pathlib.Path,
    taxonomy: TestTaxonomy,
    manifest: port_manifest.PortManifest | None = None,
) -> dict[str, Any]:
    """Compute file-level classification and listed routine oracle counts."""
    routine_meta: dict[str, tuple[str, str, str]] = {}
    if manifest is not None:
        routine_meta = _collect_manifest_routine_meta(manifest)

    by_kind_files: dict[str, int] = {k: 0 for k in VALID_KINDS}
    by_lang_files: dict[str, int] = {lang: 0 for lang in VALID_LANGUAGES}
    by_lang_tests: dict[str, int] = {lang: 0 for lang in VALID_LANGUAGES}
    mod_stats: dict[str, dict[str, Any]] = {
        mod: {
            "module": mod,
            "oracle_files": 0,
            "invariant_files": 0,
            "regression_files": 0,
            "total_files": 0,
            "total_tests": 0,
            "mislabeled_files": 0,
            "mapped_routines": 0,
            "listed_oracle_routines": 0,
            "listed_oracle_routines_pct": 0.0,
        }
        for mod in VALID_MODULES
    }
    listed_oracle_routines: set[str] = set()
    mislabeled: list[dict[str, str]] = []
    total_tests = 0

    for _rid, meta in routine_meta.items():
        mod = meta[0]
        if mod in mod_stats:
            mod_stats[mod]["mapped_routines"] += 1

    for rec in taxonomy.entries:
        full_path = repo_root / rec.path
        n_tests = count_tests_in_file(full_path) if full_path.is_file() else 0
        total_tests += n_tests
        if rec.kind in by_kind_files:
            by_kind_files[rec.kind] += 1
        if rec.language in by_lang_files:
            by_lang_files[rec.language] += 1
            by_lang_tests[rec.language] += n_tests
        if rec.module in mod_stats:
            ms = mod_stats[rec.module]
            ms["total_files"] += 1
            ms["total_tests"] += n_tests
            ms[f"{rec.kind}_files"] += 1
            if rec.mislabel_reason.strip():
                ms["mislabeled_files"] += 1
        if rec.mislabel_reason.strip():
            mislabeled.append(
                {
                    "path": rec.path,
                    "kind": rec.kind,
                    "module": rec.module,
                    "mislabel_reason": rec.mislabel_reason,
                }
            )
        if rec.kind == "oracle":
            for rid in rec.oracle_routines:
                if not routine_meta or rid in routine_meta:
                    listed_oracle_routines.add(rid)

    for rid in listed_oracle_routines:
        cov_meta = routine_meta.get(rid)
        cov_mod = cov_meta[0] if cov_meta else None
        if cov_mod and cov_mod in mod_stats:
            mod_stats[cov_mod]["listed_oracle_routines"] += 1

    by_module: list[dict[str, Any]] = []
    for mod in VALID_MODULES:
        ms = mod_stats[mod]
        mapped = int(ms["mapped_routines"])
        listed = int(ms["listed_oracle_routines"])
        ms["listed_oracle_routines_pct"] = (
            round(100.0 * listed / mapped, 1) if mapped > 0 else 0.0
        )
        if int(ms["total_files"]) > 0 or mapped > 0:
            by_module.append(ms)

    mapped_total = len(routine_meta)
    listed_total = len(listed_oracle_routines)
    listed_pct = (
        round(100.0 * listed_total / mapped_total, 1)
        if mapped_total > 0
        else 0.0
    )

    return {
        "total_files": len(taxonomy.entries),
        "total_tests": total_tests,
        "by_kind": {k: {"files": by_kind_files[k]} for k in VALID_KINDS},
        "by_language": {
            lang: {"files": by_lang_files[lang], "tests": by_lang_tests[lang]}
            for lang in VALID_LANGUAGES
        },
        "mislabeled_count": len(mislabeled),
        "mislabeled_files": sorted(mislabeled, key=lambda x: x["path"]),
        "mapped_routines_total": mapped_total,
        "listed_oracle_routines_total": listed_total,
        "listed_oracle_routines_pct": listed_pct,
        "listed_oracle_routine_ids": sorted(listed_oracle_routines),
        "by_module": by_module,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify W4 test-suite taxonomy and report file-level counts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail non-zero if taxonomy verification finds any error.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print test taxonomy file counts and listed routine summary.",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional JSON output path for the summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for W4 test-taxonomy verification and reporting."""
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    tax_path = repo_root / DEFAULT_TAXONOMY_REL
    man_path = repo_root / "benchmarks" / "dashboard" / "port_manifest.toml"

    errors: list[str] = []
    if args.check or not args.summary:
        errors = verify_test_taxonomy(repo_root, tax_path, man_path)
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            if args.check:
                return 1

    if args.summary or args.json_out is not None:
        taxonomy = load_test_taxonomy(tax_path)
        manifest = (
            port_manifest.load_port_manifest(man_path)
            if man_path.is_file()
            else None
        )
        summary = summarize_test_taxonomy(repo_root, taxonomy, manifest)
        if args.json_out is not None:
            import json

            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.summary:
            bk = summary["by_kind"]
            print(
                f"Test taxonomy: {bk['oracle']['files']} oracle files, "
                f"{bk['invariant']['files']} invariant files, "
                f"{bk['regression']['files']} regression files "
                f"(out of {summary['total_files']} test files totaling "
                f"{summary['total_tests']:,} test functions); "
                f"mislabeled={summary['mislabeled_count']}; "
                f"routines with a listed oracle test="
                f"{summary['listed_oracle_routines_total']}/"
                f"{summary['mapped_routines_total']} "
                f"({summary['listed_oracle_routines_pct']:.1f}%)"
            )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
