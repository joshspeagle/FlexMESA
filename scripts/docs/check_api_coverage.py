# SPDX-License-Identifier: GPL-3.0-or-later
"""Check public Python and PyO3 API documentation coverage (P1.4).

Reads the crate-to-page and subpackage-to-page mapping table in
``.claude/rules/docs-upkeep.md``, collects:

1. every name listed in each mapped Python subpackage's ``__all__`` (or public
   module-level definitions when ``__all__`` is omitted, as in
   ``flexmesa.constants`` and ``flexmesa._dispatch``), and
2. every top-level function and class stub in ``src/flexmesa/_rust.pyi``,
   grouped by the ``crates/flexmesa-py/src/*.rs`` PyO3 module that exports it.

For each collected symbol, checks that its mapped ``docs/api/*.md`` page either
includes a ``::: flexmesa...`` directive that renders that symbol or has an API
bullet for it (a list item that starts ``- **`name`**``), or, for a submodule
name in a package ``__all__``, has a ``::: <package>.<name>`` directive. A
whole-word mention in prose, a table or a code snippet does not count. Symbols undocumented at the P1.4 baseline are
tracked in ``docs/api_coverage_allowlist.toml``; ``--check`` fails on any
undocumented symbol outside the allowlist and on any stale allowlist entry.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import pathlib
import re
import sys
import tomllib

DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UPKEEP_RELPATH = pathlib.Path(".claude") / "rules" / "docs-upkeep.md"
ALLOWLIST_RELPATH = pathlib.Path("docs") / "api_coverage_allowlist.toml"
PYI_RELPATH = pathlib.Path("src") / "flexmesa" / "_rust.pyi"
PY_SRC_RELPATH = pathlib.Path("crates") / "flexmesa-py" / "src"

BASELINE_REASON = "undocumented as of 2026-11-28 (P1.4)"

# Static PyO3 module -> owning Rust crate mapping for crates/flexmesa-py/src/.
PYO3_MODULE_TO_CRATE: dict[str, str] = {
    "astero_adipls_py": "flexmesa-adipls",
    "astero_case_hooks_py": "flexmesa-astero",
    "astero_gyre_py": "flexmesa-gyre",
    "astero_py": "flexmesa-astero",
    "atm_py": "flexmesa-atm",
    "binary_py": "flexmesa-binary",
    "chem_py": "flexmesa-chem",
    "colors_py": "flexmesa-colors",
    "eos_helm_py": "flexmesa-eos",
    "eos_pc_skye_py": "flexmesa-eos",
    "eos_py": "flexmesa-eos",
    "eos_solver_py": "flexmesa-eos",
    "extras_py": "flexmesa-star",
    "gyre_mesa_py": "flexmesa-gyre",
    "interp_1d": "flexmesa-interp1d",
    "interp_2d": "flexmesa-interp2d",
    "ionization_py": "flexmesa-ionization",
    "kap_py": "flexmesa-kap",
    "math_hex_probe": "flexmesa-math",
    "math_solvers": "flexmesa-math",
    "math_vec_audit": "flexmesa-math",
    "net_py": "flexmesa-net",
    "neu_py": "flexmesa-neu",
    "rates_py": "flexmesa-rates",
    "star_e2e_py": "flexmesa-star",
    "star_helpers_py": "flexmesa-star",
    "star_lib_py": "flexmesa-star",
    "star_py": "flexmesa-star",
    "star_wave5_2_py": "flexmesa-star",
    "star_wave5_py": "flexmesa-star",
    "tier_a_data": "flexmesa-data",
    "tier_a_parse": "flexmesa-schema",
    "trace_py": "flexmesa-testing",
    "turb_py": "flexmesa-turb",
    "wave5_bench_py": "flexmesa-bench",
}

# Explicit per-symbol overrides for multi-crate PyO3 bridge files.
PYO3_SYMBOL_CRATE_OVERRIDES: dict[tuple[str, str], str] = {
    ("tier_a_parse", "is_bad"): "flexmesa-utils",
    ("tier_a_parse", "is_bad_array"): "flexmesa-utils",
    ("tier_a_parse", "replace_bad_nums"): "flexmesa-utils",
    ("tier_a_parse", "remove_underbars"): "flexmesa-utils",
    ("tier_a_parse", "split_line"): "flexmesa-utils",
    ("tier_a_parse", "tokenize"): "flexmesa-utils",
    ("tier_a_parse", "is_fortran_float_token"): "flexmesa-utils",
    ("tier_a_parse", "parse_fortran_float"): "flexmesa-utils",
    ("tier_a_parse", "parse_golden_text"): "flexmesa-testing",
    ("tier_a_data", "const_probe_values"): "flexmesa-const",
    ("extras_py", "binary_extras_snapshot"): "flexmesa-binary",
    ("math_solvers", "get_numerics_mode"): "flexmesa-py",
    ("math_solvers", "set_numerics_mode"): "flexmesa-py",
}

DIRECTIVE_RE = re.compile(
    r"^:::\s+(flexmesa[a-zA-Z0-9_.]*)(?:\n((?:[ \t]+.*\n)*))?",
    flags=re.MULTILINE,
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)


@dataclasses.dataclass(frozen=True, order=True)
class ApiSymbol:
    """One public Python or Rust PyO3 symbol mapped to an API docs page.

    Attributes:
        page: Repo-relative docs page path (e.g. ``docs/api/star.md``).
        source: Origin category (``python`` or ``rust_pyi``).
        group: Python subpackage name or PyO3 module stem.
        name: Public symbol identifier.
    """

    page: str
    source: str
    group: str
    name: str

    def key(self) -> tuple[str, str, str, str]:
        """Return the canonical tuple identity for allowlist matching."""
        return (self.page, self.source, self.group, self.name)


@dataclasses.dataclass(frozen=True)
class CoverageSnapshot:
    """Collected API surface, page coverage, and structural diagnostics.

    Attributes:
        symbols: Sorted tuple of all mapped public symbols.
        missing: Sorted tuple of symbols missing from their mapped docs page.
        pages: Sorted tuple of all mapped ``docs/api/*.md`` page paths.
        structural_errors: Unallowlistable structural mapping/file errors.
    """

    symbols: tuple[ApiSymbol, ...]
    missing: tuple[ApiSymbol, ...]
    pages: tuple[str, ...]
    structural_errors: tuple[str, ...]


def parse_upkeep_table(
    repo_root: pathlib.Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse crate-to-page and package-to-page maps from ``docs-upkeep.md``.

    Args:
        repo_root: Root directory of the FlexMESA repository.

    Returns:
        Tuple ``(crate_to_page, pkg_to_page)`` mapping crate names and dotted
        Python package names to repo-relative ``docs/api/*.md`` paths.
    """
    upkeep_path = repo_root / UPKEEP_RELPATH
    text = upkeep_path.read_text(encoding="utf-8")
    crate_to_page: dict[str, str] = {}
    pkg_to_page: dict[str, str] = {}

    for line in text.splitlines():
        if not line.startswith("| `flexmesa-"):
            continue
        cols = [col.strip() for col in line.strip("|").split("|")]
        if len(cols) < 5:
            continue
        raw_page = cols[4].strip("`").lstrip("/")
        if not raw_page.startswith("docs/"):
            raw_page = f"docs/{raw_page}"
        for crate in re.findall(r"`(flexmesa-[a-z0-9-]+)`", cols[0]):
            crate_to_page[crate] = raw_page
        if "(none yet" not in cols[1]:
            for raw_pkg in re.findall(r"`([A-Za-z0-9_.]+)`", cols[1]):
                pkg = (
                    raw_pkg
                    if raw_pkg.startswith("flexmesa")
                    else f"flexmesa.{raw_pkg}"
                )
                pkg_to_page[pkg] = raw_page

    return crate_to_page, pkg_to_page


def _extract_string_list(node: ast.expr | None) -> list[str] | None:
    """Return literal strings when an AST node is a list or tuple of strs."""
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)) and all(
        isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        for elt in node.elts
    ):
        return [
            str(elt.value) for elt in node.elts if isinstance(elt, ast.Constant)
        ]
    return None


def _extract_public_defs(
    statements: list[ast.stmt], *, skip_reexport_attrs: bool
) -> list[str]:
    """Return top-level public function, class, and constant names."""
    names: list[str] = []
    for stmt in statements:
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if not stmt.name.startswith("_"):
                names.append(stmt.name)
        elif isinstance(stmt, ast.Assign):
            if skip_reexport_attrs and isinstance(stmt.value, ast.Attribute):
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and not target.id.startswith("_")
                    and target.id != "__all__"
                ):
                    names.append(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(
            stmt.target, ast.Name
        ):
            if skip_reexport_attrs and isinstance(stmt.value, ast.Attribute):
                continue
            if (
                not stmt.target.id.startswith("_")
                and stmt.target.id != "__all__"
            ):
                names.append(stmt.target.id)
    return names


def _extract_dunder_all(tree: ast.Module) -> list[str] | None:
    """Return explicit ``__all__`` exports from a module AST if declared."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return _extract_string_list(stmt.value)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "__all__"
        ):
            return _extract_string_list(stmt.value)
        elif isinstance(stmt, ast.If) and stmt.orelse:
            # Matches src/flexmesa/__init__.py where full __all__ sits in else.
            for sub in stmt.orelse:
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "__all__"
                        ):
                            return _extract_string_list(sub.value)
    return None


def _resolve_module_file(
    repo_root: pathlib.Path, mod_path: str
) -> pathlib.Path | None:
    """Resolve a dotted ``flexmesa...`` module path under ``src/``."""
    parts = mod_path.split(".")
    py_candidate = (
        repo_root / "src" / pathlib.Path(*parts[:-1], f"{parts[-1]}.py")
    )
    if py_candidate.is_file():
        return py_candidate
    init_candidate = repo_root / "src" / pathlib.Path(*parts, "__init__.py")
    if init_candidate.is_file():
        return init_candidate
    return None


def collect_python_symbols(
    repo_root: pathlib.Path, pkg_to_page: dict[str, str]
) -> tuple[list[ApiSymbol], list[str]]:
    """Collect public symbols for every Python subpackage in ``docs-upkeep.md``.

    Args:
        repo_root: Root directory of the FlexMESA repository.
        pkg_to_page: Package-to-API-page mapping from ``docs-upkeep.md``.

    Returns:
        Tuple ``(symbols, errors)`` for all mapped Python subpackages.
    """
    symbols: list[ApiSymbol] = []
    errors: list[str] = []

    src_pkg_root = repo_root / "src" / "flexmesa"
    if src_pkg_root.is_dir():
        for init_path in sorted(src_pkg_root.glob("*/__init__.py")):
            subpkg = init_path.parent.name
            if subpkg.startswith("_"):
                continue
            dotted = f"flexmesa.{subpkg}"
            if dotted not in pkg_to_page:
                errors.append(
                    f"Python subpackage {dotted} missing from {UPKEEP_RELPATH}"
                )

    for pkg, page in sorted(pkg_to_page.items()):
        mod_file = _resolve_module_file(repo_root, pkg)
        if mod_file is None:
            errors.append(
                f"Python module {pkg} mapped to {page} does not exist"
            )
            continue
        tree = ast.parse(mod_file.read_text(encoding="utf-8"))
        dunder_all = _extract_dunder_all(tree)
        names = (
            dunder_all
            if dunder_all is not None
            else _extract_public_defs(tree.body, skip_reexport_attrs=False)
        )
        for name in sorted(set(names)):
            symbols.append(
                ApiSymbol(page=page, source="python", group=pkg, name=name)
            )

    return symbols, errors


def _scan_pyo3_exports(
    repo_root: pathlib.Path,
) -> tuple[dict[str, list[str]], list[str]]:
    """Map exported Python names to their declaring ``crates/flexmesa-py/src/``
    module stems.
    """
    py_src_dir = repo_root / PY_SRC_RELPATH
    lib_rs = py_src_dir / "lib.rs"
    if not lib_rs.is_file():
        return {}, [f"Missing {lib_rs.relative_to(repo_root)}"]

    lib_text = lib_rs.read_text(encoding="utf-8")
    mod_stems = re.findall(
        r"^\s*pub\s+mod\s+([A-Za-z0-9_]+)\s*;", lib_text, flags=re.MULTILINE
    )
    sym_to_mods: dict[str, list[str]] = {}
    errors: list[str] = []

    for stem in sorted(set(mod_stems)):
        rs_path = py_src_dir / f"{stem}.rs"
        if not rs_path.is_file():
            errors.append(f"Declared PyO3 module {stem}.rs not found")
            continue
        text = rs_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        exported: set[str] = set()

        for match in re.finditer(
            r"wrap_pyfunction!\(\s*(?:[A-Za-z0-9_]+::)?([A-Za-z0-9_]+)", text
        ):
            exported.add(match.group(1))
        for match in re.finditer(
            r"add_class::<(?:[A-Za-z0-9_]+::)?([A-Za-z0-9_]+)>", text
        ):
            exported.add(match.group(1))
        for idx, line in enumerate(lines):
            if "#[pyclass" in line or "#[pyfunction" in line:
                block = "\n".join(lines[idx : idx + 12])
                name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
                if name_match:
                    exported.add(name_match.group(1))
            for match in re.finditer(r'm\.add\(\s*"([^"]+)"', line):
                exported.add(match.group(1))

        for sym in sorted(exported):
            sym_to_mods.setdefault(sym, []).append(stem)

    return sym_to_mods, errors


def collect_rust_pyi_symbols(
    repo_root: pathlib.Path, crate_to_page: dict[str, str]
) -> tuple[list[ApiSymbol], list[str]]:
    """Collect ``_rust.pyi`` functions and classes grouped by PyO3 module.

    Args:
        repo_root: Root directory of the FlexMESA repository.
        crate_to_page: Crate-to-API-page mapping from ``docs-upkeep.md``.

    Returns:
        Tuple ``(symbols, errors)`` for all top-level ``_rust.pyi`` definitions.
    """
    pyi_path = repo_root / PYI_RELPATH
    if not pyi_path.is_file():
        return [], [f"Missing {PYI_RELPATH}"]

    sym_to_mods, errors = _scan_pyo3_exports(repo_root)
    tree = ast.parse(pyi_path.read_text(encoding="utf-8"))
    symbols: list[ApiSymbol] = []

    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        mods = sym_to_mods.get(node.name, [])
        if len(mods) != 1:
            errors.append(
                f"_rust.pyi symbol {node.name!r} matched PyO3 modules {mods}"
            )
            continue
        mod = mods[0]
        crate = PYO3_SYMBOL_CRATE_OVERRIDES.get(
            (mod, node.name), PYO3_MODULE_TO_CRATE.get(mod)
        )
        if crate is None:
            errors.append(
                f"PyO3 module {mod!r} (for {node.name!r}) missing from "
                "PYO3_MODULE_TO_CRATE"
            )
            continue
        page = crate_to_page.get(crate)
        if page is None:
            errors.append(
                f"Crate {crate!r} (for {mod}::{node.name}) missing from "
                f"{UPKEEP_RELPATH}"
            )
            continue
        symbols.append(
            ApiSymbol(page=page, source="rust_pyi", group=mod, name=node.name)
        )

    return symbols, errors


def _rendered_names_for_directive(
    repo_root: pathlib.Path, mod_path: str, opt_block: str | None
) -> tuple[set[str], str | None]:
    """Return symbol names rendered by one ``::: <module>`` directive."""
    if opt_block and "members:" in opt_block:
        explicit = re.findall(
            r"^\s*-\s+([A-Za-z0-9_]+)\s*$", opt_block, flags=re.MULTILINE
        )
        if explicit:
            return set(explicit), None

    mod_file = _resolve_module_file(repo_root, mod_path)
    if mod_file is None:
        return set(), f"Directive ::: {mod_path} does not resolve under src/"

    tree = ast.parse(mod_file.read_text(encoding="utf-8"))
    is_init = mod_file.name == "__init__.py"
    names = set(_extract_public_defs(tree.body, skip_reexport_attrs=is_init))
    return names, None


def inspect_repo(repo_root: pathlib.Path) -> CoverageSnapshot:
    """Evaluate Python and Rust PyO3 API doc coverage for ``repo_root``.

    Args:
        repo_root: Repository root containing ``.claude/rules/docs-upkeep.md``,
            ``src/``, ``crates/``, and ``docs/api/``.

    Returns:
        :class:`CoverageSnapshot` with sorted symbols, missing symbols, page
        list, and any structural mapping or file-resolution errors.
    """
    crate_to_page, pkg_to_page = parse_upkeep_table(repo_root)
    py_syms, py_errs = collect_python_symbols(repo_root, pkg_to_page)
    rust_syms, rust_errs = collect_rust_pyi_symbols(repo_root, crate_to_page)
    all_syms = sorted(
        {sym.key(): sym for sym in (*py_syms, *rust_syms)}.values()
    )
    pages = sorted(
        {
            *crate_to_page.values(),
            *pkg_to_page.values(),
            *(sym.page for sym in all_syms),
        }
    )

    errors: list[str] = [*py_errs, *rust_errs]
    rendered_by_page: dict[str, set[str]] = {}
    modules_by_page: dict[str, set[str]] = {}
    text_by_page: dict[str, str] = {}

    for page in pages:
        page_path = repo_root / page
        if not page_path.is_file():
            errors.append(f"Mapped API docs page missing: {page}")
            rendered_by_page[page] = set()
            modules_by_page[page] = set()
            text_by_page[page] = ""
            continue
        raw_text = page_path.read_text(encoding="utf-8")
        cleaned = HTML_COMMENT_RE.sub("", raw_text)
        text_by_page[page] = cleaned
        rendered: set[str] = set()
        for match in DIRECTIVE_RE.finditer(cleaned):
            names, dir_err = _rendered_names_for_directive(
                repo_root, match.group(1), match.group(2)
            )
            if dir_err is not None:
                errors.append(f"{page}: {dir_err}")
            rendered |= names
        rendered_by_page[page] = rendered
        modules_by_page[page] = {
            m.group(1) for m in DIRECTIVE_RE.finditer(cleaned)
        }

    missing: list[ApiSymbol] = []
    for sym in all_syms:
        if sym.name in rendered_by_page.get(sym.page, ()):
            continue
        if (
            sym.source == "python"
            and f"{sym.group}.{sym.name}" in modules_by_page[sym.page]
        ):
            continue  # submodule name documented by its own directive
        if re.search(
            rf"^[ \t]*[-*][ \t]+\*\*`{re.escape(sym.name)}`\*\*",
            text_by_page[sym.page],
            flags=re.MULTILINE,
        ):
            continue
        missing.append(sym)

    return CoverageSnapshot(
        symbols=tuple(all_syms),
        missing=tuple(sorted(missing)),
        pages=tuple(pages),
        structural_errors=tuple(errors),
    )


def load_allowlist(
    allowlist_path: pathlib.Path,
) -> tuple[dict[tuple[str, str, str, str], str], dict[str, int], list[str]]:
    """Load ``docs/api_coverage_allowlist.toml``.

    Args:
        allowlist_path: Path to the TOML allowlist file.

    Returns:
        Tuple ``(entries, counts, errors)`` where ``entries`` maps
        ``(page, source, group, name)`` to its reason string.
    """
    if not allowlist_path.is_file():
        return {}, {}, [f"Missing allowlist file: {allowlist_path}"]
    data = tomllib.loads(allowlist_path.read_text(encoding="utf-8"))
    counts = {str(k): int(v) for k, v in data.get("counts", {}).items()}
    entries: dict[tuple[str, str, str, str], str] = {}
    errors: list[str] = []
    for item in data.get("allow", []):
        key = (
            str(item["page"]),
            str(item["source"]),
            str(item["group"]),
            str(item["name"]),
        )
        reason = str(item.get("reason", "")).strip()
        if key in entries:
            errors.append(f"Duplicate allowlist entry: {key}")
        if not reason:
            errors.append(f"Allowlist entry missing reason: {key}")
        entries[key] = reason
    return entries, counts, errors


def write_allowlist(
    allowlist_path: pathlib.Path,
    missing: tuple[ApiSymbol, ...],
    existing_reasons: dict[tuple[str, str, str, str], str] | None = None,
) -> None:
    """Write ``docs/api_coverage_allowlist.toml`` deterministically.

    Args:
        allowlist_path: Target TOML file path.
        missing: Missing symbols to record.
        existing_reasons: Optional mapping preserving existing entry reasons.
    """
    reasons = existing_reasons or {}
    sorted_missing = sorted(missing)
    py_count = sum(1 for sym in sorted_missing if sym.source == "python")
    rust_count = sum(1 for sym in sorted_missing if sym.source == "rust_pyi")

    lines: list[str] = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# GENERATED VIA python scripts/docs/check_api_coverage.py "
        "--write-allowlist (P1.4).",
        "# API symbols undocumented as of 2026-11-28 (P1.4). The gate fails on",
        "# any missing symbol outside this file and on any stale entry here,",
        "# so this file only shrinks.",
        "",
        "[counts]",
        f"python = {py_count}",
        f"rust_pyi = {rust_count}",
        f"total = {len(sorted_missing)}",
    ]
    for sym in sorted_missing:
        reason = reasons.get(sym.key(), BASELINE_REASON).replace('"', '\\"')
        lines.extend(
            [
                "",
                "[[allow]]",
                f'page = "{sym.page}"',
                f'source = "{sym.source}"',
                f'group = "{sym.group}"',
                f'name = "{sym.name}"',
                f'reason = "{reason}"',
            ]
        )
    lines.append("")
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text("\n".join(lines), encoding="utf-8")


def format_report(
    snapshot: CoverageSnapshot,
    allow_keys: set[tuple[str, str, str, str]],
) -> str:
    """Render a concise per-page and per-PyO3-module coverage report."""
    missing_keys = {sym.key() for sym in snapshot.missing}
    lines: list[str] = [
        "API docs coverage report:",
        f"{'Page':28s} | {'Python':12s} | {'Rust _rust':12s} | Allowlisted",
        f"{'-' * 28}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 11}",
    ]
    for page in snapshot.pages:
        py_all = [
            s
            for s in snapshot.symbols
            if s.page == page and s.source == "python"
        ]
        rs_all = [
            s
            for s in snapshot.symbols
            if s.page == page and s.source == "rust_pyi"
        ]
        py_miss = sum(1 for s in py_all if s.key() in missing_keys)
        rs_miss = sum(1 for s in rs_all if s.key() in missing_keys)
        alw = sum(1 for s in (*py_all, *rs_all) if s.key() in allow_keys)
        py_str = f"{len(py_all) - py_miss}/{len(py_all)}"
        rs_str = f"{len(rs_all) - rs_miss}/{len(rs_all)}"
        lines.append(f"{page:28s} | {py_str:12s} | {rs_str:12s} | {alw:11d}")

    py_tot = sum(1 for s in snapshot.symbols if s.source == "python")
    rs_tot = sum(1 for s in snapshot.symbols if s.source == "rust_pyi")
    py_m_tot = sum(1 for s in snapshot.missing if s.source == "python")
    rs_m_tot = sum(1 for s in snapshot.missing if s.source == "rust_pyi")
    lines.append(
        f"{'TOTAL':28s} | {f'{py_tot - py_m_tot}/{py_tot}':12s} | "
        f"{f'{rs_tot - rs_m_tot}/{rs_tot}':12s} | {len(allow_keys):11d}"
    )
    if snapshot.missing:
        lines.append("")
        lines.append("Missing symbols by PyO3/package group:")
        by_group: dict[str, int] = {}
        for sym in snapshot.missing:
            by_group[f"{sym.page}::{sym.group}"] = (
                by_group.get(f"{sym.page}::{sym.group}", 0) + 1
            )
        for key, cnt in sorted(by_group.items()):
            lines.append(f"  {key}: {cnt}")
    return "\n".join(lines)


def run_check(
    snapshot: CoverageSnapshot, allowlist_path: pathlib.Path
) -> tuple[bool, list[str]]:
    """Run the API coverage gate against ``allowlist_path``.

    Returns:
        Tuple ``(passed, messages)``.
    """
    messages: list[str] = list(snapshot.structural_errors)
    entries, counts, load_errs = load_allowlist(allowlist_path)
    messages.extend(load_errs)

    missing_keys = {sym.key(): sym for sym in snapshot.missing}
    allow_keys = set(entries)

    expected_py = sum(1 for k in allow_keys if k[1] == "python")
    expected_rs = sum(1 for k in allow_keys if k[1] == "rust_pyi")
    if not load_errs and (
        counts.get("python") != expected_py
        or counts.get("rust_pyi") != expected_rs
        or counts.get("total") != len(allow_keys)
    ):
        messages.append(
            f"Allowlist [counts] mismatch: got {counts}, "
            f"expected python={expected_py}, rust_pyi={expected_rs}, "
            f"total={len(allow_keys)}"
        )

    new_missing = sorted(set(missing_keys) - allow_keys)
    stale = sorted(allow_keys - set(missing_keys))

    for key in new_missing:
        messages.append(
            f"UNDOCUMENTED: {key[0]} | {key[1]} | {key[2]}::{key[3]}"
        )
    for key in stale:
        messages.append(
            f"STALE_ALLOWLIST: {key[0]} | {key[1]} | {key[2]}::{key[3]}"
        )

    return (len(messages) == 0), messages


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for API doc coverage checks and allowlist seeding.

    Args:
        argv: Optional CLI argument list overriding ``sys.argv[1:]``.

    Returns:
        Exit status code (``0`` on clean pass/report/write, ``1`` on failure).
    """
    parser = argparse.ArgumentParser(
        description="Verify docs/api coverage of Python __all__ and _rust.pyi."
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the FlexMESA repository root.",
    )
    parser.add_argument(
        "--allowlist",
        type=pathlib.Path,
        default=None,
        help="Optional override path for docs/api_coverage_allowlist.toml.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check API docs coverage against the allowlist (default).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print per-page Python and Rust PyO3 coverage counts.",
    )
    parser.add_argument(
        "--write-allowlist",
        action="store_true",
        help="Write the missing symbols allowlist file.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    allowlist_path = (
        args.allowlist.resolve()
        if args.allowlist is not None
        else repo_root / ALLOWLIST_RELPATH
    )
    snapshot = inspect_repo(repo_root)

    if args.write_allowlist:
        if snapshot.structural_errors:
            for err in snapshot.structural_errors:
                sys.stderr.write(f"ERROR: {err}\n")
            return 1
        existing: dict[tuple[str, str, str, str], str] = {}
        if allowlist_path.is_file():
            existing, _, _ = load_allowlist(allowlist_path)
        write_allowlist(allowlist_path, snapshot.missing, existing)
        print(
            f"Wrote {allowlist_path} ({len(snapshot.missing)} allowlisted "
            f"of {len(snapshot.symbols)} symbols)"
        )
        return 0

    if args.report:
        entries, _, _ = (
            load_allowlist(allowlist_path)
            if allowlist_path.is_file()
            else ({}, {}, [])
        )
        print(format_report(snapshot, set(entries)))
        if snapshot.structural_errors:
            for err in snapshot.structural_errors:
                sys.stderr.write(f"ERROR: {err}\n")
            return 1
        return 0

    passed, messages = run_check(snapshot, allowlist_path)
    if not passed:
        for msg in messages:
            sys.stderr.write(f"{msg}\n")
        return 1

    entries, _, _ = load_allowlist(allowlist_path)
    print(
        f"API coverage gate passed ({len(snapshot.symbols)} symbols checked, "
        f"{len(entries)} allowlisted, 0 undocumented outside allowlist, "
        "0 stale)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
