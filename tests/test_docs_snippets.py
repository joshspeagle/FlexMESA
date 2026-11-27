# SPDX-License-Identifier: GPL-3.0-or-later
"""Automated verification of public documentation, links, API directives, and
executable Python code blocks (`tests/test_docs_snippets.py`).
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys
import textwrap

import matplotlib
import matplotlib.pyplot as plt

from flexmesa.data import fetch

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

EXPECTED_API_PAGES: tuple[str, ...] = (
    "index.md",
    "star.md",
    "binary.md",
    "astero.md",
    "eos.md",
    "opacity-atm-colors.md",
    "nuclear.md",
    "numerics.md",
    "io-plotting-bench.md",
)

INTERNAL_REEXPORTED_HELPERS: frozenset[str] = frozenset(
    {
        "flexmesa.bench.bicub_fast",
        "flexmesa.bench.tdc_fast",
        "flexmesa.bench.tridiag_fast",
        "flexmesa.binary.jax_kernels",
    }
)


def _iter_user_facing_doc_files() -> list[pathlib.Path]:
    """Return user-facing markdown files across the 6 navigation sections.

    Returns:
        Ordered list of user-facing markdown documentation paths.
    """
    files: list[pathlib.Path] = [
        DOCS_DIR / "index.md",
        DOCS_DIR / "design" / "index.md",
    ]
    for subdir in (
        "getting-started",
        "concepts",
        "tutorials",
        "guides",
        "api",
        "benchmarks",
        "reference",
    ):
        files.extend(sorted((DOCS_DIR / subdir).glob("*.md")))
    return files


def test_mkdocs_nav_files_and_internal_links_resolve() -> None:
    """Every nav entry in mkdocs.yml and relative link in docs/ must resolve."""
    mkdocs_path = REPO_ROOT / "mkdocs.yml"
    mkdocs_text = mkdocs_path.read_text(encoding="utf-8")
    assert "\nnav:\n" in mkdocs_text
    nav_section = mkdocs_text.split("\nnav:\n", 1)[1]
    nav_entries = re.findall(
        r":\s+([a-zA-Z0-9_./-]+\.md)\s*$", nav_section, flags=re.MULTILINE
    )
    assert len(nav_entries) >= 50

    for rel_md in nav_entries:
        target_md = DOCS_DIR / rel_md
        assert target_md.is_file(), f"Missing mkdocs.yml nav file: {rel_md}"

    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    fence_pattern = re.compile(r"```.*?```", flags=re.DOTALL)

    for md_path in sorted(DOCS_DIR.rglob("*.md")):
        raw_text = md_path.read_text(encoding="utf-8")
        prose_text = fence_pattern.sub("", raw_text)
        for match in link_pattern.findall(prose_text):
            target = match.strip().split()[0].split("#")[0]
            if not target or target.startswith(
                ("http://", "https://", "mailto:", "/")
            ):
                continue
            resolved = (md_path.parent / target).resolve()
            assert resolved.exists(), (
                f"Broken relative link in {md_path.relative_to(REPO_ROOT)}: "
                f"{target!r}"
            )


def test_api_reference_covers_all_public_submodules() -> None:
    """All 8 domain API pages and ::: directives must import and cover src/."""
    api_dir = DOCS_DIR / "api"
    for page_name in EXPECTED_API_PAGES:
        assert (api_dir / page_name).is_file(), f"Missing API page: {page_name}"

    directive_re = re.compile(
        r"^:::\s+(flexmesa[a-zA-Z0-9_.]*)", flags=re.MULTILINE
    )
    documented_modules: set[str] = set()
    for api_md in sorted(api_dir.glob("*.md")):
        text = api_md.read_text(encoding="utf-8")
        directives = directive_re.findall(text)
        assert directives, f"No ::: directives found in {api_md.name}"
        for mod_path in directives:
            mod = importlib.import_module(mod_path)
            assert mod is not None
            documented_modules.add(mod_path)

    src_pkg_root = REPO_ROOT / "src" / "flexmesa"
    for init_path in sorted(src_pkg_root.glob("*/__init__.py")):
        pkg_name = f"flexmesa.{init_path.parent.name}"
        assert any(
            mod == pkg_name or mod.startswith(f"{pkg_name}.")
            for mod in documented_modules
        ), f"Public subpackage {pkg_name} has no ::: directive in docs/api/*.md"

    for py_path in sorted(src_pkg_root.rglob("*.py")):
        if py_path.name == "__init__.py" or py_path.name.startswith("_"):
            continue
        rel_parts = py_path.relative_to(REPO_ROOT / "src").with_suffix("").parts
        mod_name = ".".join(rel_parts)
        if mod_name in INTERNAL_REEXPORTED_HELPERS:
            continue
        assert mod_name in documented_modules, (
            f"Public submodule {mod_name} is missing a '::: {mod_name}' "
            f"directive in docs/api/*.md"
        )


def test_public_docs_prose_and_path_hygiene() -> None:
    """Docs, notes, and rules contain zero bad paths, GFM alerts, or em-dash."""
    forbidden_paths: tuple[str, ...] = (
        "/" + "home/",
        "/" + "mnt/d/",
        "/" + "tmp/",
        "." + "venvs/",
        "file" + "://",
        "speagle" + "-desktop",
    )
    claude_md = REPO_ROOT / "CLAUDE.md"
    agents_claude_md = REPO_ROOT / ".agents" / "rules" / "CLAUDE.md"
    claude_upkeep = REPO_ROOT / ".claude" / "rules" / "docs-upkeep.md"
    agents_upkeep = REPO_ROOT / ".agents" / "rules" / "docs-upkeep.md"
    claude_gov = (
        REPO_ROOT / ".claude" / "rules" / "verification-and-governance.md"
    )
    agents_gov = (
        REPO_ROOT / ".agents" / "rules" / "verification-and-governance.md"
    )
    rule_files: tuple[pathlib.Path, ...] = (
        claude_md,
        agents_claude_md,
        claude_upkeep,
        agents_upkeep,
        claude_gov,
        agents_gov,
    )
    for canonical, link in (
        (claude_md, agents_claude_md),
        (claude_upkeep, agents_upkeep),
        (claude_gov, agents_gov),
    ):
        assert link.is_symlink() and link.resolve() == canonical.resolve(), (
            f"{link.relative_to(REPO_ROOT)} must be a symlink to "
            f"{canonical.relative_to(REPO_ROOT)}; recreate the link with "
            "ln -s instead of copying the file"
        )
        assert canonical.read_text(encoding="utf-8") == link.read_text(
            encoding="utf-8"
        )

    root_prose_files: tuple[pathlib.Path, ...] = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "notes" / "DECISIONS.md",
    )
    doc_files = sorted(DOCS_DIR.rglob("*.md"))
    gfm_alert_re = re.compile(
        r"^\s*>\s*\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]",
        flags=re.MULTILINE,
    )
    em_dash = "\u2014"
    all_checked_files = [*doc_files, *rule_files, *root_prose_files]
    for path in all_checked_files:
        assert path.is_file(), f"Expected file missing: {path}"
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        for token in forbidden_paths:
            assert token not in text, (
                f"Forbidden machine-specific token {token!r} found in {rel}"
            )
        assert em_dash not in text, f"Prose em-dash found in {rel}"
    for doc_path in doc_files:
        doc_rel = doc_path.relative_to(REPO_ROOT)
        doc_text = doc_path.read_text(encoding="utf-8")
        assert not gfm_alert_re.search(doc_text), (
            f"Raw GFM alert callout found in {doc_rel}; use MkDocs `!!!` syntax"
        )


def test_executable_python_snippets_in_docs() -> None:
    """Every fenced python snippet in user-facing docs must execute cleanly."""
    code_re = re.compile(
        r"^[ \t]*```python\n(.*?)^[ \t]*```", flags=re.MULTILINE | re.DOTALL
    )
    executed_count = 0

    for md_path in _iter_user_facing_doc_files():
        text = md_path.read_text(encoding="utf-8")
        blocks = code_re.findall(text)
        if not blocks:
            continue
        rel = md_path.relative_to(REPO_ROOT)
        namespace: dict[str, object] = {"__name__": "__main__"}
        for idx, raw_code in enumerate(blocks, start=1):
            code = textwrap.dedent(raw_code)
            compiled = compile(code, f"{rel}#block{idx}", "exec")
            exec(compiled, namespace)
            plt.close("all")
            executed_count += 1

    assert executed_count >= 60


def test_upstream_mesa_26_4_1_citations_and_defaults_diff_resolve() -> None:
    """Every upstream file:line citation and defaults diff must match 26.4.1."""
    import importlib.util

    mesa_root = fetch.resolve_mesa_test_inputs_dir()
    assert (mesa_root / "star" / "private").is_dir()

    by_basename: dict[str, list[pathlib.Path]] = {}
    for up_path in sorted(mesa_root.rglob("*")):
        if (
            up_path.is_file()
            and up_path.suffix in {".f90", ".f", ".inc", ".dek", ".defaults"}
            and "test_suite" not in up_path.parts
        ):
            by_basename.setdefault(up_path.name, []).append(up_path)

    line_count_cache: dict[pathlib.Path, int] = {}

    def _num_lines(target: pathlib.Path) -> int:
        if target not in line_count_cache:
            line_count_cache[target] = len(
                target.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            )
        return line_count_cache[target]

    colon_re = re.compile(
        r"\b([A-Za-z0-9_./-]+\.(?:f90|f|inc|dek|defaults)"
        r"|[A-Za-z0-9_]+/test/test_output):"
        r"(\d+(?:[-\u2013]\d+)?(?:\s*,\s*\d+(?:[-\u2013]\d+)?)*)"
    )
    prose_re = re.compile(
        r"`?([A-Za-z0-9_./-]+\.(?:f90|f|inc|dek|defaults))`?\s+lines?\s+"
        r"(\d+)(?:(?:\s*(?:[-\u2013]|to)\s*)(\d+))?"
    )
    range_item_re = re.compile(r"^(\d+)(?:[-\u2013](\d+))?$")

    frozen_jax_dirs = {"star", "binary", "astero"}
    non_frozen_py = [
        p
        for p in sorted((REPO_ROOT / "src" / "flexmesa").rglob("*.py"))
        if p.relative_to(REPO_ROOT / "src" / "flexmesa").parts[0]
        not in frozen_jax_dirs
    ]
    scan_paths: list[pathlib.Path] = [
        REPO_ROOT / "CLAUDE.md",
        REPO_ROOT / "notes" / "DECISIONS.md",
        REPO_ROOT / "notes" / "SCOPE.md",
        REPO_ROOT / "notes" / "audit" / "defaults_diff_26_4_1.md",
        *sorted((REPO_ROOT / ".claude" / "rules").glob("*.md")),
        *sorted(DOCS_DIR.rglob("*.md")),
        *sorted((REPO_ROOT / "crates").rglob("*.rs")),
        *non_frozen_py,
        *sorted((REPO_ROOT / "tests").rglob("*.py")),
    ]

    # Bundled MESA SDK 26.6.1 CRMATH 1.2 Fortran source line counts (#165).
    sdk_crmath_lines: dict[str, int] = {
        "crmath.f90": 861,
        "crmath_r_dp.f90": 424,
        "crmath_c_dp.f90": 254,
    }

    # Reference LAPACK/BLAS 3.12.0 sources cited by flexmesa-gyre (#170):
    # basename -> line count, from the committed manifest.
    lapack_manifest = (
        REPO_ROOT
        / "benchmarks"
        / "fortran_mesa"
        / "gyre_g2_vectors"
        / "lapack_3_12_0_sources.tsv"
    )
    lapack_lines: dict[str, int] = {}
    for row in lapack_manifest.read_text(encoding="utf-8").splitlines():
        if not row or row.startswith("#"):
            continue
        base, n_lines, sha256 = row.split("\t")
        assert re.fullmatch(r"[0-9a-f]{64}", sha256), row
        assert base not in by_basename, (
            f"LAPACK manifest basename {base!r} collides with a MESA file"
        )
        lapack_lines[base] = int(n_lines)
    assert lapack_lines
    # Bare basenames, or the netlib source directories they live in.
    lapack_dirs = {"", "SRC", "BLAS/SRC", "INSTALL"}

    def _resolve_target_lines(
        raw_fp: str, rel_src: pathlib.Path, lno: int, cite_tok: str
    ) -> tuple[str, int]:
        if raw_fp in sdk_crmath_lines:
            return f"mesasdk/crmath-1.2/{raw_fp}", sdk_crmath_lines[raw_fp]
        lapack_dir, _, lapack_base = raw_fp.rpartition("/")
        if lapack_base in lapack_lines and lapack_dir in lapack_dirs:
            return f"lapack-3.12.0/{raw_fp}", lapack_lines[lapack_base]
        if "/" in raw_fp:
            cand = mesa_root / raw_fp
            assert cand.is_file(), (
                f"{rel_src}:{lno}: upstream MESA 26.4.1 file not found "
                f"for citation {cite_tok!r}"
            )
            return str(cand.relative_to(mesa_root)), _num_lines(cand)
        cands = by_basename.get(raw_fp, [])
        assert cands, (
            f"{rel_src}:{lno}: upstream MESA 26.4.1 file not found "
            f"for citation {cite_tok!r}"
        )
        assert len(cands) == 1, (
            f"{rel_src}:{lno}: ambiguous bare basename {raw_fp!r} in "
            f"{cite_tok!r}; matches "
            f"{[str(p.relative_to(mesa_root)) for p in cands]}"
        )
        return str(cands[0].relative_to(mesa_root)), _num_lines(cands[0])

    checked = 0
    for src_path in scan_paths:
        rel_src = src_path.relative_to(REPO_ROOT)
        for lno, line in enumerate(
            src_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in colon_re.finditer(line):
                raw_fp = match.group(1)
                target_label, n_max = _resolve_target_lines(
                    raw_fp, rel_src, lno, match.group(0)
                )
                for part in match.group(2).split(","):
                    rm = range_item_re.match(part.strip())
                    assert rm is not None
                    lo = int(rm.group(1))
                    hi = int(rm.group(2)) if rm.group(2) else lo
                    assert 1 <= lo <= hi <= n_max, (
                        f"{rel_src}:{lno}: citation {match.group(0)!r} out of "
                        f"bounds (1..={n_max}) in {target_label}"
                    )
                    checked += 1
            for match in prose_re.finditer(line):
                raw_fp = match.group(1)
                target_label, n_max = _resolve_target_lines(
                    raw_fp, rel_src, lno, match.group(0)
                )
                lo = int(match.group(2))
                hi = int(match.group(3)) if match.group(3) else lo
                assert 1 <= lo <= hi <= n_max, (
                    f"{rel_src}:{lno}: citation {match.group(0)!r} out of "
                    f"bounds (1..={n_max}) in {target_label}"
                )
                checked += 1

    assert checked >= 1100

    # Verify generated defaults_diff_26_4_1.md is up to date in-process.
    spec = importlib.util.spec_from_file_location(
        "diff_mesa_defaults",
        REPO_ROOT / "scripts" / "audit" / "diff_mesa_defaults.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diff_mesa_defaults"] = mod
    spec.loader.exec_module(mod)
    expected_md = mod.generate_markdown(mesa_root)
    actual_md = (
        REPO_ROOT / "notes" / "audit" / "defaults_diff_26_4_1.md"
    ).read_text(encoding="utf-8")
    assert actual_md == expected_md
