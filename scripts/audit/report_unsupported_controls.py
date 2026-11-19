#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit unsupported and headless plotting controls across MESA 26.4.1 cases.

Scans all 106 manifest cases in ``$MESA_DIR`` / ``mesa_test_inputs`` across
``&star_job``, ``&eos``, ``&kap``, ``&controls``, ``&binary_job``,
``&binary_controls``, ``&astero_search_controls``, and headless ``&pgstar`` /
``&pgbinary`` / ``&astero_pgstar_controls`` blocks, generating:

* ``notes/audit/unsupported_controls_26_4_1.md``
* ``benchmarks/history_harness/unsupported_controls_26_4_1.json``
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import pathlib
import re
import shlex
import sys
import tomllib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_CONST_ARRAY_RE = re.compile(
    r"pub const (?P<name>[A-Z0-9_]+):\s*&\[&str\]\s*=\s*&\[(?P<body>.*?)\];",
    re.DOTALL,
)
_STRING_LIT_RE = re.compile(r'"([^"]+)"')
_DO_ONE_RE = re.compile(
    r"^\s*do_one\s+(?P<inlist>\S+)(?:\s+\S+)?(?:\s+\S+)?",
)
_SECTION_START_RE = re.compile(r"^&([A-Za-z_][A-Za-z0-9_]*)\b(.*)$")
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?\s*=\s*(.+)$")
_HEADLESS_SECTIONS = frozenset({"pgstar", "pgbinary", "astero_pgstar_controls"})
_INCLUDE_PREFIX = {
    "astero_pgstar_controls": "astero_pgstar",
    "astero_search_controls": "astero_search",
    "simplex_search_controls": "simplex_search",
}
_KNOWN_UPSTREAM_ONLY_SECTIONS = frozenset(
    {"simplex_search_controls", "colors", "multi_stars_job", "stella"}
)


@dataclasses.dataclass(frozen=True)
class InventorySets:
    """Rust supported, unused, headless, and upstream key registries."""

    upstream: dict[str, frozenset[str]]
    supported: dict[str, frozenset[str]]
    unused: dict[str, frozenset[str]]
    headless_gui: dict[str, frozenset[str]]
    terminal_output: dict[str, frozenset[str]]


@dataclasses.dataclass(frozen=True)
class ParsedAssignment:
    """Single ``name[(subscript)] = value`` assignment inside a namelist."""

    name: str
    subscript: str | None
    value: bool | str


@dataclasses.dataclass(frozen=True)
class ParsedBlock:
    """Single ``&section ... /`` Fortran namelist block."""

    name: str
    assignments: tuple[ParsedAssignment, ...]


def _resolve_mesa_root(explicit: str | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    env_dir = os.environ.get("MESA_DIR")
    if env_dir:
        candidates.append(pathlib.Path(env_dir))
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(pathlib.Path(xdg) / "flexmesa" / "mesa_test_inputs")
    candidates.append(
        pathlib.Path.home() / ".cache" / "flexmesa" / "mesa_test_inputs"
    )
    for cand in candidates:
        if (cand / "star" / "test_suite").is_dir():
            return cand
    raise FileNotFoundError(
        "could not locate unpacked MESA test_inputs; pass --mesa-dir or set "
        "MESA_DIR"
    )


def _strip_fortran_comment(line: str) -> str:
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            if ch == quote:
                if i + 1 < len(line) and line[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        else:
            if ch in ("'", '"'):
                quote = ch
            elif ch == "!":
                return line[:i]
        i += 1
    return line


def _split_top_level_statements(text: str) -> list[str]:
    stmts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    paren = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    buf.append(text[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == "(":
            paren += 1
            buf.append(ch)
        elif ch == ")":
            paren = max(0, paren - 1)
            buf.append(ch)
        elif ch in (";", ",", "\n") and paren == 0:
            part = "".join(buf).strip()
            if part:
                stmts.append(part)
            buf.clear()
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def _parse_literal(raw: str) -> bool | str:
    s = raw.strip()
    low = s.lower()
    if low == ".true.":
        return True
    if low == ".false.":
        return False
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].replace(s[0] * 2, s[0])
    return s


def _parse_namelist_file(path: pathlib.Path) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    in_block: str | None = None
    body_lines: list[str] = []

    def finish_block() -> None:
        nonlocal in_block
        if in_block is None:
            return
        assigns: list[ParsedAssignment] = []
        body = "\n".join(body_lines)
        for stmt in _split_top_level_statements(body):
            m = _KEY_RE.match(stmt)
            if m is None:
                continue
            name = m.group(1).strip()
            sub = m.group(2).strip() if m.group(2) is not None else None
            val = _parse_literal(m.group(3))
            assigns.append(
                ParsedAssignment(name=name, subscript=sub, value=val)
            )
        blocks.append(ParsedBlock(name=in_block, assignments=tuple(assigns)))
        in_block = None
        body_lines.clear()

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = _strip_fortran_comment(raw).strip()
        if not clean:
            continue
        if in_block is None:
            m = _SECTION_START_RE.match(clean)
            if m is None:
                continue
            sec = m.group(1)
            rest = m.group(2).strip()
            in_block = sec
            if rest.endswith("/") or rest.endswith("&end"):
                trimmed = (
                    rest[:-1] if rest.endswith("/") else rest[:-4]
                ).strip()
                if trimmed:
                    body_lines.append(trimmed)
                finish_block()
            elif rest:
                body_lines.append(rest)
            continue

        # Check for end of namelist outside quotes
        end_pos: int | None = None
        quote: str | None = None
        i = 0
        while i < len(clean):
            ch = clean[i]
            if quote is not None:
                if ch == quote:
                    if i + 1 < len(clean) and clean[i + 1] == quote:
                        i += 2
                        continue
                    quote = None
            else:
                if ch in ("'", '"'):
                    quote = ch
                elif ch == "/":
                    end_pos = i
                    break
            i += 1
        if end_pos is not None:
            before = clean[:end_pos].strip()
            if before:
                body_lines.append(before)
            finish_block()
        elif clean.lower() == "&end":
            finish_block()
        else:
            body_lines.append(clean)

    finish_block()
    return blocks


def _extract_rust_const_arrays(path: pathlib.Path) -> dict[str, frozenset[str]]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, frozenset[str]] = {}
    for match in _CONST_ARRAY_RE.finditer(text):
        body = re.sub(r"//.*", "", match.group("body"))
        keys = frozenset(_STRING_LIT_RE.findall(body))
        out[match.group("name")] = keys
    return out


def _load_upstream_keys(path: pathlib.Path) -> dict[str, frozenset[str]]:
    sections: dict[str, set[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().lower()
            sections.setdefault(current, set())
        elif current is not None:
            sections[current].add(line.lower())
    return {sec: frozenset(keys) for sec, keys in sections.items()}


def load_inventories() -> InventorySets:
    """Load upstream MESA 26.4.1 keys and Rust supported/unused sets."""
    schema_txt = (
        REPO_ROOT
        / "crates"
        / "flexmesa-schema"
        / "data"
        / "mesa_26_4_1_namelist_keys.txt"
    )
    star_inv = _extract_rust_const_arrays(
        REPO_ROOT / "crates" / "flexmesa-star" / "src" / "inlist_inventory.rs"
    )
    bin_inv = _extract_rust_const_arrays(
        REPO_ROOT / "crates" / "flexmesa-binary" / "src" / "inlist_inventory.rs"
    )
    ast_inv = _extract_rust_const_arrays(
        REPO_ROOT / "crates" / "flexmesa-astero" / "src" / "inlist_parser.rs"
    )

    supported = {
        "star_job": star_inv["STAR_JOB_SUPPORTED"],
        "eos": star_inv["EOS_SUPPORTED"],
        "kap": star_inv["KAP_SUPPORTED"],
        "controls": star_inv["CONTROLS_SUPPORTED"],
        "binary_job": bin_inv["BINARY_JOB_SUPPORTED"],
        "binary_controls": bin_inv["BINARY_CONTROLS_SUPPORTED"],
        "astero_search_controls": ast_inv["ASTERO_SEARCH_CONTROLS_SUPPORTED"],
    }
    unused = {
        "star_job": star_inv["STAR_JOB_UNUSED"],
        "eos": frozenset(),
        "kap": frozenset(),
        "controls": star_inv["CONTROLS_UNUSED"],
        "binary_job": frozenset(),
        "binary_controls": frozenset(),
        "astero_search_controls": ast_inv["ASTERO_SEARCH_CONTROLS_UNUSED"],
    }
    headless_gui = {
        "star_job": star_inv["STAR_JOB_HEADLESS_GUI"],
        "binary_job": bin_inv["BINARY_JOB_HEADLESS_GUI"],
    }
    return InventorySets(
        upstream=_load_upstream_keys(schema_txt),
        supported=supported,
        unused=unused,
        headless_gui=headless_gui,
        terminal_output={
            "star_job": star_inv["STAR_JOB_TERMINAL_OUTPUT"],
            "controls": star_inv["CONTROLS_TERMINAL_OUTPUT"],
            "binary_job": bin_inv["BINARY_JOB_TERMINAL_OUTPUT"],
            "binary_controls": bin_inv["BINARY_CONTROLS_TERMINAL_OUTPUT"],
        },
    )


def _is_include_key(section: str, key: str) -> bool:
    prefix = _INCLUDE_PREFIX.get(section, section)
    if key in (
        f"read_extra_{prefix}_inlist",
        f"extra_{prefix}_inlist_name",
    ):
        return True
    return any(
        key
        in (
            f"read_extra_{prefix}_inlist{i}",
            f"extra_{prefix}_inlist{i}_name",
        )
        for i in range(1, 6)
    )


def _parse_include_slot(
    section: str,
    key: str,
    subscript: str | None,
) -> tuple[str, int] | None:
    prefix = _INCLUDE_PREFIX.get(section, section)
    if (
        key == f"read_extra_{prefix}_inlist"
        and subscript is not None
        and subscript.isdigit()
        and 1 <= int(subscript) <= 5
    ):
        return ("flag", int(subscript))
    if (
        key == f"extra_{prefix}_inlist_name"
        and subscript is not None
        and subscript.isdigit()
        and 1 <= int(subscript) <= 5
    ):
        return ("name", int(subscript))
    for i in range(1, 6):
        if key == f"read_extra_{prefix}_inlist{i}" and subscript is None:
            return ("flag", i)
        if key == f"extra_{prefix}_inlist{i}_name" and subscript is None:
            return ("name", i)
    return None


def _parse_rn_do_one_inlists(
    case_dir: pathlib.Path,
    *,
    skip_optional: bool = True,
) -> list[str]:
    rn_path = case_dir / "rn"
    if not rn_path.is_file():
        return ["inlist"] if (case_dir / "inlist").is_file() else []

    inlists: list[str] = []
    branch_stack: list[tuple[bool, bool]] = []
    rn_text = rn_path.read_text(encoding="utf-8", errors="replace")
    for raw in rn_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("if "):
            if "MESA_SKIP_OPTIONAL" in line:
                negated = "!" in line or "-z" in line
                active = (not skip_optional) if negated else skip_optional
                branch_stack.append((True, active))
            else:
                branch_stack.append((False, True))
            continue
        if line == "else" or line.startswith("else "):
            if branch_stack:
                is_opt, active = branch_stack[-1]
                if is_opt:
                    branch_stack[-1] = (True, not active)
            continue
        if line == "fi" or line.startswith("fi "):
            if branch_stack:
                branch_stack.pop()
            continue
        if any(is_opt and not act for is_opt, act in branch_stack):
            continue
        if line.startswith("check_okay"):
            continue
        if line == "do_one" or line.startswith("do_one "):
            try:
                toks = shlex.split(line)
            except ValueError:
                match = _DO_ONE_RE.match(line)
                toks = ["do_one", match.group("inlist")] if match else []
            entry = toks[1] if len(toks) >= 2 else "inlist"
            if entry == "inlist" and not (case_dir / "inlist").is_file():
                # Bare `do_one` defaults to `inlist`; multi-star cases expand
                # `inlist_multi_stars_job` into one stage per `inlist_names(i)`
                # entry (`crates/flexmesa-star/src/rn_parser.rs`).
                multi = _multi_stars_inlist_names(case_dir)
                if multi:
                    for name in multi:
                        if name not in inlists:
                            inlists.append(name)
                    continue
            if entry not in inlists:
                inlists.append(entry)
    if not inlists and (case_dir / "inlist").is_file():
        inlists.append("inlist")
    return inlists


def _multi_stars_inlist_names(case_dir: pathlib.Path) -> list[str]:
    """Return ``inlist_names(i)`` from ``inlist_multi_stars_job`` in order."""
    path = case_dir / "inlist_multi_stars_job"
    if not path.is_file():
        return []
    names: dict[int, str] = {}
    for blk in _parse_namelist_file(path):
        if blk.name.lower() != "multi_stars_job":
            continue
        for item in blk.assignments:
            if item.name.lower() == "inlist_names" and item.subscript:
                if isinstance(item.value, str):
                    names[int(item.subscript)] = item.value.strip()
    return [names[k] for k in sorted(names)]


def _resolve_section_assignments(
    case_dir: pathlib.Path,
    entry_file: str,
) -> tuple[list[tuple[str, str, str]], dict[str, bool]]:
    """Resolve assignments plus the final ``pgstar_flag``/``pgbinary_flag``."""
    visited: set[tuple[str, pathlib.Path]] = set()
    assigns: list[tuple[str, str, str]] = []
    flags_seen: dict[str, bool] = {}

    def walk(rel_file: str, only_section: str | None) -> None:
        fpath = (case_dir / rel_file).resolve()
        sec_key = "" if only_section is None else only_section
        if (sec_key, fpath) in visited or not fpath.is_file():
            return
        visited.add((sec_key, fpath))
        blocks = _parse_namelist_file(fpath)
        pending: dict[str, tuple[dict[int, bool], dict[int, str]]] = {}
        for blk in blocks:
            sec = blk.name.lower()
            if only_section is not None and sec != only_section:
                continue
            flags, names = pending.setdefault(sec, ({}, {}))
            for item in blk.assignments:
                key = item.name.lower()
                slot = _parse_include_slot(sec, key, item.subscript)
                if slot is not None:
                    kind, idx = slot
                    if kind == "flag" and isinstance(item.value, bool):
                        flags[idx] = item.value
                    elif kind == "name" and isinstance(item.value, str):
                        names[idx] = item.value.strip()
                if key in ("pgstar_flag", "pgbinary_flag") and isinstance(
                    item.value, bool
                ):
                    flags_seen[key] = item.value
                assigns.append((sec, key, rel_file))
        for sec, (flags, names) in pending.items():
            for idx in range(1, 6):
                if flags.get(idx, False) and idx in names and names[idx]:
                    walk(names[idx], sec)

    walk(entry_file, None)
    return assigns, flags_seen


def _collect_case_files(
    suite: str,
    case_dir: pathlib.Path,
) -> list[str]:
    if suite == "star":
        return _parse_rn_do_one_inlists(case_dir, skip_optional=True)

    if suite == "binary":
        files: list[str] = []
        if (case_dir / "inlist").is_file():
            files.append("inlist")
        evolve_both = False
        inlist_names: dict[int, str] = {}
        seen: set[pathlib.Path] = set()

        def scan_bjob(rel: str) -> None:
            nonlocal evolve_both
            p = (case_dir / rel).resolve()
            if p in seen or not p.is_file():
                return
            seen.add(p)
            flags: dict[int, bool] = {}
            extras: dict[int, str] = {}
            for blk in _parse_namelist_file(p):
                if blk.name.lower() != "binary_job":
                    continue
                for a in blk.assignments:
                    k = a.name.lower()
                    if k == "evolve_both_stars" and isinstance(a.value, bool):
                        evolve_both = a.value
                    elif (
                        k == "inlist_names"
                        and a.subscript is not None
                        and a.subscript.isdigit()
                        and isinstance(a.value, str)
                    ):
                        inlist_names[int(a.subscript)] = a.value.strip()
                    else:
                        slot = _parse_include_slot("binary_job", k, a.subscript)
                        if slot is not None:
                            kind, idx = slot
                            if kind == "flag" and isinstance(a.value, bool):
                                flags[idx] = a.value
                            elif kind == "name" and isinstance(a.value, str):
                                extras[idx] = a.value.strip()
            for idx in range(1, 6):
                if flags.get(idx, False) and idx in extras and extras[idx]:
                    scan_bjob(extras[idx])

        scan_bjob("inlist")
        s1 = inlist_names.get(1, "inlist1")
        if s1 and (case_dir / s1).is_file() and s1 not in files:
            files.append(s1)
        if evolve_both:
            s2 = inlist_names.get(2, "inlist2")
            if s2 and (case_dir / s2).is_file() and s2 not in files:
                files.append(s2)
        return files

    files = _parse_rn_do_one_inlists(case_dir, skip_optional=True)
    for extra in ("inlist", "inlist_astero_search_controls"):
        if (case_dir / extra).is_file() and extra not in files:
            files.append(extra)
    return files


def _load_manifest_cases() -> list[tuple[str, str]]:
    manifest_path = (
        REPO_ROOT / "benchmarks" / "history_harness" / "case_manifest.toml"
    )
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    return [(str(item["suite"]), str(item["case"])) for item in data["cases"]]


def _load_upstream_ref() -> dict[str, str]:
    bundles_path = REPO_ROOT / "src" / "flexmesa" / "data" / "bundles.toml"
    data = tomllib.loads(bundles_path.read_text(encoding="utf-8"))
    ref = data["upstream_reference"]
    return {
        "name": str(ref["name"]),
        "git_tag": str(ref["git_tag"]),
        "git_commit": str(ref["git_commit"])[:7],
        "zenodo_record": str(ref["zenodo_record"]),
    }


def scan_cases(
    mesa_root: pathlib.Path,
    inventories: InventorySets,
) -> dict[str, Any]:
    """Scan all 106 manifest cases and classify every control key."""
    cases = _load_manifest_cases()
    case_reports: list[dict[str, Any]] = []
    unsupported_freq: collections.Counter[tuple[str, str, str]] = (
        collections.Counter()
    )
    unsupported_cases: dict[tuple[str, str, str], list[str]] = (
        collections.defaultdict(list)
    )
    plotting_freq: collections.Counter[tuple[str, str]] = collections.Counter()
    terminal_freq: collections.Counter[tuple[str, str]] = collections.Counter()
    plotting_unknown: dict[tuple[str, str], list[str]] = (
        collections.defaultdict(list)
    )
    plotting_cases_count = 0
    all_unknown: set[tuple[str, str]] = set()

    for suite, case_name in cases:
        case_dir = mesa_root / suite / "test_suite" / case_name
        entry_files = _collect_case_files(suite, case_dir)

        unsupported_seen: set[tuple[str, str, str]] = set()
        unknown_seen: set[tuple[str, str]] = set()
        plotting_seen: set[tuple[str, str]] = set()
        terminal_seen: set[tuple[str, str]] = set()

        for ef in entry_files:
            assigns, flags = _resolve_section_assignments(case_dir, ef)
            # Upstream reads the plotting namelists only when the matching
            # flag is true (`star/job/run_star_support.f90:718`,
            # `binary/private/pgbinary_full.f90:132`); DECISIONS #25.
            active = {
                "pgstar": flags.get("pgstar_flag", False),
                "astero_pgstar_controls": flags.get("pgstar_flag", False),
                "pgbinary": flags.get("pgbinary_flag", False),
            }
            for sec, key, _rel_file in assigns:
                if _is_include_key(sec, key):
                    continue
                if sec in _HEADLESS_SECTIONS:
                    up_keys = inventories.upstream.get(sec, frozenset())
                    if key in up_keys:
                        plotting_seen.add((sec, key))
                    elif active[sec]:
                        unknown_seen.add((sec, key))
                        plotting_unknown[(sec, key)].append(
                            f"{suite}/{case_name}"
                        )
                    continue
                if key in inventories.headless_gui.get(sec, frozenset()):
                    plotting_seen.add((sec, key))
                    continue
                if sec in _KNOWN_UPSTREAM_ONLY_SECTIONS:
                    unsupported_seen.add((sec, key, "unimplemented"))
                    continue
                if sec not in inventories.supported:
                    unknown_seen.add((sec, key))
                    continue

                if key in inventories.supported[sec]:
                    continue
                # Keys whose only upstream effect is terminal text are
                # accepted and recorded (DECISIONS #26).
                if key in inventories.terminal_output.get(sec, frozenset()):
                    terminal_seen.add((sec, key))
                    continue
                if key in inventories.unused.get(sec, frozenset()):
                    unsupported_seen.add((sec, key, "unused"))
                elif key in inventories.upstream.get(sec, frozenset()):
                    unsupported_seen.add((sec, key, "unimplemented"))
                else:
                    unknown_seen.add((sec, key))

        all_unknown |= unknown_seen
        if plotting_seen:
            plotting_cases_count += 1
        for sec, key in plotting_seen:
            plotting_freq[(sec, key)] += 1
        for sec, key in terminal_seen:
            terminal_freq[(sec, key)] += 1

        for sec, key, kind in unsupported_seen:
            unsupported_freq[(sec, key, kind)] += 1
            unsupported_cases[(sec, key, kind)].append(f"{suite}/{case_name}")

        strict_ok = not unsupported_seen and not unknown_seen
        case_reports.append(
            {
                "suite": suite,
                "case": case_name,
                "strict_ok": strict_ok,
                "entry_files": entry_files,
                "unsupported_controls": [
                    f"{s}::{k}::{cat}" for s, k, cat in sorted(unsupported_seen)
                ],
                "unknown_controls": [
                    f"{s}::{k}" for s, k in sorted(unknown_seen)
                ],
                "headless_plotting_keys_count": len(plotting_seen),
                "terminal_output_keys": [
                    f"{s}::{k}" for s, k in sorted(terminal_seen)
                ],
            }
        )

    strict_pass = sum(1 for r in case_reports if r["strict_ok"])
    top_unsupported = [
        {
            "section": sec,
            "key": key,
            "category": cat,
            "case_count": count,
            "cases": sorted(unsupported_cases[(sec, key, cat)]),
        }
        for (sec, key, cat), count in sorted(
            unsupported_freq.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
        )
    ]
    top_plotting = [
        {
            "section": sec,
            "key": key,
            "case_count": count,
        }
        for (sec, key), count in sorted(
            plotting_freq.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
        )
    ]

    ref = _load_upstream_ref()
    section_coverage = []
    for sec in (
        "star_job",
        "eos",
        "kap",
        "controls",
        "binary_job",
        "binary_controls",
        "astero_search_controls",
    ):
        sup_n = len(inventories.supported[sec])
        unused_n = len(inventories.unused[sec])
        gui_n = len(inventories.headless_gui.get(sec, frozenset()))
        term_n = len(inventories.terminal_output.get(sec, frozenset()))
        up_n = len(inventories.upstream.get(sec, frozenset()))
        section_coverage.append(
            {
                "section": sec,
                "upstream_keys": up_n,
                "supported_keys": sup_n,
                "unused_keys": unused_n,
                "headless_gui_keys": gui_n,
                "terminal_output_keys": term_n,
                "unimplemented_upstream_keys": max(
                    0, up_n - sup_n - unused_n - gui_n - term_n
                ),
            }
        )

    return {
        "schema_version": "mesa_26_4_1",
        "upstream_reference": ref,
        "summary": {
            "total_cases": len(case_reports),
            "strict_pass_cases": strict_pass,
            "strict_fail_cases": len(case_reports) - strict_pass,
            "distinct_unsupported_controls": len(top_unsupported),
            "distinct_unknown_controls": len(all_unknown),
            "cases_with_headless_plotting": plotting_cases_count,
            "distinct_headless_plotting_keys": len(top_plotting),
        },
        "section_coverage": section_coverage,
        "unsupported_controls": top_unsupported,
        "terminal_output_controls": {
            "decision": "DECISIONS #26",
            "keys": [
                {"section": sec, "key": key, "case_count": count}
                for (sec, key), count in sorted(
                    terminal_freq.items(),
                    key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
                )
            ],
        },
        "headless_plotting_issue_173": {
            "issue": "#173",
            "decision": "DECISIONS #25",
            "cases_with_plotting_controls": plotting_cases_count,
            "distinct_valid_plotting_keys": len(top_plotting),
            "unknown_plotting_keys": [
                {"section": s, "key": k, "cases": sorted(cs)}
                for (s, k), cs in sorted(plotting_unknown.items())
            ],
            "plotting_keys": top_plotting,
        },
        "cases": case_reports,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render ``notes/audit/unsupported_controls_26_4_1.md``."""
    ref = payload["upstream_reference"]
    sm = payload["summary"]
    lines: list[str] = [
        (
            "<!-- GENERATED BY "
            "`python scripts/audit/report_unsupported_controls.py` (#147). "
            "DO NOT EDIT BY HAND. -->"
        ),
        "# Unsupported and Headless Controls Report (MESA `26.4.1`)",
        "",
        (
            f"Generated mechanically against **{ref['name']}** "
            f"(git tag `{ref['git_tag']}`, commit `{ref['git_commit']}`, "
            f"Zenodo `{ref['zenodo_record']}`) for Issue `#147` and `#173` "
            "(DECISIONS `#25`)."
        ),
        "",
        "## Summary",
        "",
        f"- **Total manifest cases scanned**: `{sm['total_cases']}`",
        (
            "- **Cases passing strict `#147` validation**: "
            f"`{sm['strict_pass_cases']}`"
        ),
        (
            "- **Cases rejected by strict `#147` validation**: "
            f"`{sm['strict_fail_cases']}`"
        ),
        (
            "- **Distinct unsupported physics/solver controls encountered**: "
            f"`{sm['distinct_unsupported_controls']}`"
        ),
        (
            "- **Distinct unknown/misspelled controls encountered**: "
            f"`{sm['distinct_unknown_controls']}`"
        ),
        (
            "- **Cases using headless plotting controls "
            "(`#173` / `DECISIONS #25`)**: "
            f"`{sm['cases_with_headless_plotting']}` "
            f"(`{sm['distinct_headless_plotting_keys']}` distinct keys)"
        ),
        "",
        "## Namelist Inventory Coverage",
        "",
        (
            "| Section | Upstream `26.4.1` | Supported | Unused (`#147`) "
            "| Headless GUI (`#173`) | Terminal output (`DECISIONS #26`) "
            "| Unimplemented |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cov in payload["section_coverage"]:
        lines.append(
            f"| `&{cov['section']}` | {cov['upstream_keys']} "
            f"| {cov['supported_keys']} | {cov['unused_keys']} "
            f"| {cov['headless_gui_keys']} "
            f"| {cov['terminal_output_keys']} "
            f"| {cov['unimplemented_upstream_keys']} |"
        )

    lines.extend(
        [
            "",
            "## Top Unsupported Physics & Solver Controls (`#147`)",
            "",
            "| Control | Category | Cases | Sample Cases |",
            "| :--- | :--- | ---: | :--- |",
        ]
    )
    for item in payload["unsupported_controls"][:60]:
        sample = ", ".join(f"`{c}`" for c in item["cases"][:4])
        if len(item["cases"]) > 4:
            sample += f", ... (+{len(item['cases']) - 4})"
        lines.append(
            f"| `&{item['section']}::{item['key']}` "
            f"| `{item['category']}` | {item['case_count']} | {sample} |"
        )

    lines.extend(
        [
            "",
            "## Terminal-Output Controls (`DECISIONS #26`)",
            "",
            (
                "These keys only gate or format text that MESA writes to the "
                "terminal. Strict validation accepts and records them. "
                "`FlexMESA` does not print MESA's terminal log, so they have "
                "no effect on a run."
            ),
            "",
            "| Control | Cases |",
            "| :--- | ---: |",
        ]
    )
    for item in payload["terminal_output_controls"]["keys"]:
        lines.append(
            f"| `&{item['section']}::{item['key']}` | {item['case_count']} |"
        )

    lines.extend(
        [
            "",
            "## Headless Plotting Controls Inventory (Issue `#173`)",
            "",
            (
                "Per `DECISIONS #25` (matching upstream `USE_PGSTAR = NO`), "
                "keys in `&pgstar`, `&pgbinary`, and `&astero_pgstar_controls` "
                "(plus `pgstar_flag` / `pgbinary_flag`) are validated against "
                "upstream `*.defaults` schemas so typos fail loudly, while "
                "valid plotting keys execute as headless no-ops and are "
                "tracked separately here for `#173`."
            ),
            "",
        ]
    )
    plot_sec = payload["headless_plotting_issue_173"]
    unknown_plot = plot_sec["unknown_plotting_keys"]
    lines.append(f"- **Unknown plotting keys found**: `{len(unknown_plot)}`")
    if unknown_plot:
        for u in unknown_plot:
            lines.append(
                f"  - `&{u['section']}::{u['key']}` in "
                + ", ".join(f"`{c}`" for c in u["cases"])
            )
    lines.extend(
        [
            "",
            "| Headless Plotting Control | Cases Using |",
            "| :--- | ---: |",
        ]
    )
    for pitem in plot_sec["plotting_keys"][:50]:
        lines.append(
            f"| `&{pitem['section']}::{pitem['key']}` | {pitem['case_count']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Strict Validation Summary (`106` Cases)",
            "",
            (
                "| Suite | Case | Strict Status | Unsupported | Unknown "
                "| Plot Keys |"
            ),
            "| :--- | :--- | :--- | ---: | ---: | ---: |",
        ]
    )
    for rep in payload["cases"]:
        status = "pass" if rep["strict_ok"] else "fail_unsupported"
        lines.append(
            f"| `{rep['suite']}` | `{rep['case']}` | `{status}` "
            f"| {len(rep['unsupported_controls'])} "
            f"| {len(rep['unknown_controls'])} "
            f"| {rep['headless_plotting_keys_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-dir",
        default=None,
        help="Optional override for unpacked mesa-test-inputs-v2 root.",
    )
    parser.add_argument(
        "--md-out",
        default=str(
            REPO_ROOT / "notes" / "audit" / "unsupported_controls_26_4_1.md"
        ),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--json-out",
        default=str(
            REPO_ROOT
            / "benchmarks"
            / "history_harness"
            / "unsupported_controls_26_4_1.json"
        ),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that committed reports match freshly generated content.",
    )
    ns = parser.parse_args(argv)
    mesa_root = _resolve_mesa_root(ns.mesa_dir)
    inventories = load_inventories()
    payload = scan_cases(mesa_root, inventories)
    md_text = render_markdown(payload)
    json_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"

    md_path = pathlib.Path(ns.md_out)
    json_path = pathlib.Path(ns.json_out)
    if ns.check:
        ok_md = (
            md_path.is_file() and md_path.read_text(encoding="utf-8") == md_text
        )
        ok_js = (
            json_path.is_file()
            and json_path.read_text(encoding="utf-8") == json_text
        )
        if not (ok_md and ok_js):
            print(
                f"out of date: {md_path} / {json_path}",
                file=sys.stderr,
            )
            return 1
        return 0

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    print(
        f"Wrote {md_path.relative_to(REPO_ROOT)} and "
        f"{json_path.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
