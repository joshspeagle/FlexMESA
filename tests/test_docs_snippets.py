# SPDX-License-Identifier: LGPL-3.0-or-later
"""Automated verification of public documentation, links, API directives, and
executable Python code blocks (`tests/test_docs_snippets.py`).
"""

from __future__ import annotations

import importlib
import pathlib
import re
import textwrap

import matplotlib
import matplotlib.pyplot as plt

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
    """Docs and rule files must contain zero machine paths or em-dashes."""
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
    rule_files: tuple[pathlib.Path, ...] = (
        claude_md,
        agents_claude_md,
        claude_upkeep,
        agents_upkeep,
    )
    assert claude_md.read_text(encoding="utf-8") == agents_claude_md.read_text(
        encoding="utf-8"
    )
    assert claude_upkeep.read_text(encoding="utf-8") == agents_upkeep.read_text(
        encoding="utf-8"
    )

    user_facing_set = frozenset((*_iter_user_facing_doc_files(), *rule_files))
    em_dash = "\u2014"
    all_checked_files = [*sorted(DOCS_DIR.rglob("*.md")), *rule_files]
    for path in all_checked_files:
        assert path.is_file(), f"Expected file missing: {path}"
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        for token in forbidden_paths:
            assert token not in text, (
                f"Forbidden machine-specific token {token!r} found in {rel}"
            )
        if path in user_facing_set:
            assert em_dash not in text, f"Prose em-dash found in {rel}"


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
