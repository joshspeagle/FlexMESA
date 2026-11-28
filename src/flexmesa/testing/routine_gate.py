# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate on ``routine_map`` entries against the routine inventory (`#611`).

Two checks run over every ``routine_map`` entry of ``port_manifest.toml``:

* ``routine_key``: the key names exactly one routine of that upstream file in
  ``benchmarks/dashboard/routine_inventory.json`` (Fortran names are matched
  without regard to case). A key is either a bare name or ``parent::name``
  for a contained procedure. A bare name that matches more than one routine
  of the file (for example three contained ``check`` procedures) is a
  violation, and the qualified form must be used;
* ``fn_item``: every symbol of the mapped target ``<path>::<sym>[+<sym>...]``
  exists as a function item (``fn`` in Rust, ``def`` in Python) in that file.
  A struct, constant, or enum with the right name does not satisfy it. A
  symbol is a bare name or a type-qualified method ``Type::name``. A
  qualified symbol requires an ``impl`` block for ``Type`` in the file (an
  inherent impl, a generic impl, or ``impl Trait for Type``) that holds
  ``fn name``. A bare Rust name is a violation when the file defines more than
  one ``fn`` of that name (free functions and methods together), because it
  cannot say which one is meant; use ``Type::name``.

Two structural checks run without an allow-list:

* every name in a file's ``routine_map_partial`` is a key of that file's
  ``routine_map``;
* every routine declared not applicable in ``port_status_na.toml`` (read by
  ``routine_na``, when that file exists) names exactly one routine of the
  inventory and is not also a ``routine_map`` key naming the same routine.

Violations present when the gate was introduced are listed in
``benchmarks/dashboard/routine_map_allowlist.toml`` with per-check counts. The
gate fails on any violation outside the allow-list and on any allow-list entry
that no longer violates, so the allow-list can only shrink.

Run ``python -m flexmesa.testing.routine_gate`` to check,
``--report`` to print counts, and ``--write-allowlist`` to regenerate the
allow-list after a deliberate change.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys
import tomllib
from typing import Any

from flexmesa.testing import (
    _port_parsers,
    port_manifest,
    routine_inventory,
)

ALLOWLIST_RELPATH = "benchmarks/dashboard/routine_map_allowlist.toml"
CHECKS = ("routine_key", "fn_item")


@dataclasses.dataclass(frozen=True, order=True)
class Violation:
    """One ``routine_map`` entry that fails a gate check.

    Attributes:
        check: ``routine_key`` or ``fn_item``.
        upstream_path: Manifest ``upstream_path`` of the entry.
        routine: The ``routine_map`` key.
        target: The mapped ``<path>::<sym>[+<sym>...]`` value.
        detail: What is missing (the file, or the offending symbol).
    """

    check: str
    upstream_path: str
    routine: str
    target: str
    detail: str

    def ident(self) -> tuple[str, str, str, str]:
        """Return the identity used to match allow-list entries."""
        return (self.check, self.upstream_path, self.routine, self.detail)


def match_routines(key: str, routines: list[list[Any]]) -> list[list[Any]]:
    """Return the inventory routines that a ``routine_map`` key names.

    Args:
        key: ``name`` or ``parent::name`` (case-insensitive). The parent is the
            enclosing routine recorded in the inventory.
        routines: Inventory rows ``[name, kind, parent, first, last]``.

    Returns:
        Matching rows. A usable key matches exactly one; none means the key
        is unknown and several mean it is ambiguous.
    """
    parent, sep, name = key.rpartition("::")
    name = name.lower()
    if not sep:
        return [r for r in routines if r[0].lower() == name]
    parent = parent.lower()
    return [
        r for r in routines if r[0].lower() == name and r[2].lower() == parent
    ]


def structural_errors(
    manifest: port_manifest.PortManifest,
    inventory: dict[str, Any],
    na_entries: list[Any],
) -> list[str]:
    """Return errors for partial markers and N/A declarations.

    Args:
        manifest: Loaded port manifest.
        inventory: Loaded routine inventory.
        na_entries: ``routine_na.NotApplicable`` declarations.

    Returns:
        Error messages, empty when every marker and declaration is valid.
    """
    errors: list[str] = []
    files = inventory.get("files", {})
    for entry in manifest.entries:
        keys = {k for k, _ in entry.routine_map}
        errors.extend(
            f"{entry.upstream_path}: routine_map_partial[{n!r}] is not a "
            "routine_map key"
            for n in entry.routine_map_partial
            if n not in keys
        )
    by_path = {e.upstream_path: e for e in manifest.entries}
    for na in na_entries:
        rec = files.get(na.upstream_path)
        if rec is None:
            errors.append(f"{na.upstream_path}: N/A file not in inventory")
            continue
        rows = na_rows(na, rec["routines"])
        if len(rows) != 1:
            errors.append(
                f"{na.upstream_path}: N/A routine {na.key()!r} matches "
                f"{len(rows)} inventory routines"
            )
            continue
        mapped_entry = by_path.get(na.upstream_path)
        for key, _ in mapped_entry.routine_map if mapped_entry else ():
            if any(r is rows[0] for r in match_routines(key, rec["routines"])):
                errors.append(
                    f"{na.upstream_path}: N/A routine {na.key()!r} is also "
                    f"in routine_map as {key!r}"
                )
    return errors


def na_rows(na: Any, routines: list[list[Any]]) -> list[Any]:
    """Return inventory rows named by an N/A entry (name, parent, span)."""
    rows = match_routines(na.key(), routines)
    if na.first is not None:
        rows = [r for r in rows if r[3] == na.first]
    if na.last is not None:
        rows = [r for r in rows if r[4] == na.last]
    return rows


@dataclasses.dataclass(frozen=True)
class _FileFns:
    """Function items of one target file.

    Attributes:
        counts: Occurrences of each function name (Rust: free functions and
            methods together; Python: every ``def``).
        methods: Rust impl self type to its method names (empty for Python).
        is_rust: Whether the file is Rust (the ambiguity rule applies).
    """

    counts: dict[str, int]
    methods: dict[str, set[str]]
    is_rust: bool


def _fn_names(
    repo_root: pathlib.Path, rel: str, cache: dict[str, _FileFns]
) -> _FileFns | None:
    """Return function items defined in ``rel`` (``None`` if absent)."""
    if rel in cache:
        return cache[rel]
    path = repo_root / rel
    if not path.is_file() or path.suffix not in {".rs", ".py"}:
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rs":
        text = _port_parsers.strip_rust_comments_and_strings(text)
        fns = _FileFns(
            _port_parsers.rust_fn_counts(text),
            _port_parsers.rust_impl_methods(text),
            True,
        )
    else:
        pat = r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        counts: dict[str, int] = {}
        for name in re.findall(pat, text, flags=re.MULTILINE):
            counts[name] = counts.get(name, 0) + 1
        fns = _FileFns(counts, {}, False)
    cache[rel] = fns
    return fns


def _symbol_problem(fns: _FileFns, owner: str | None, sym: str) -> str | None:
    """Return the ``detail`` for a failing symbol, or ``None`` if it passes."""
    if owner is not None:
        if sym in fns.methods.get(owner, set()):
            return None
        return f"{owner}::{sym}"
    n = fns.counts.get(sym, 0)
    if n == 0:
        return sym
    if fns.is_rust and n > 1:
        return f"{sym} (ambiguous, {n} fn items; use Type::{sym})"
    return None


def collect_violations(
    manifest: port_manifest.PortManifest,
    inventory: dict[str, Any],
    repo_root: pathlib.Path,
) -> list[Violation]:
    """Return every ``routine_map`` entry failing a gate check, sorted."""
    out: list[Violation] = []
    cache: dict[str, _FileFns] = {}
    files = inventory.get("files", {})
    for entry in manifest.entries:
        if not entry.routine_map:
            continue
        rec = files.get(entry.upstream_path)
        for key, target in entry.routine_map:
            up = entry.upstream_path
            if rec is None:
                out.append(Violation("routine_key", up, key, target, "<file>"))
            else:
                n = len(match_routines(key, rec["routines"]))
                if n == 0:
                    out.append(Violation("routine_key", up, key, target, key))
                elif n > 1:
                    out.append(
                        Violation(
                            "routine_key",
                            up,
                            key,
                            target,
                            f"{key} (ambiguous, {n} routines; "
                            "use parent::name)",
                        )
                    )
            rel, _, syms = target.partition("::")
            fns = _fn_names(repo_root, rel, cache)
            for owner, sym in _port_parsers.split_target_symbols(syms):
                if fns is None:
                    out.append(Violation("fn_item", up, key, target, rel))
                    break
                problem = _symbol_problem(fns, owner, sym)
                if problem is not None:
                    out.append(Violation("fn_item", up, key, target, problem))
    return sorted(out)


def load_allowlist(
    path: pathlib.Path,
) -> tuple[dict[str, int], list[Violation]]:
    """Load ``(declared counts, allowed violations)`` from the allow-list."""
    if not path.is_file():
        return {}, []
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    counts = {k: int(v) for k, v in raw.get("counts", {}).items()}
    allowed = [
        Violation(
            str(a["check"]),
            str(a["upstream_path"]),
            str(a["routine"]),
            str(a["target"]),
            str(a["detail"]),
        )
        for a in raw.get("allow", [])
    ]
    return counts, allowed


def _toml_str(text: str) -> str:
    """Quote ``text`` as a TOML basic string."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_allowlist(violations: list[Violation]) -> str:
    """Render the allow-list for ``violations`` deterministically."""
    lines = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# GENERATED VIA python -m flexmesa.testing.routine_gate "
        "--write-allowlist (#611).",
        "# routine_map entries that failed a gate check when the gate was",
        "# introduced. The gate allows these and nothing else, and fails",
        "# when an entry here stops violating, so this file only shrinks.",
        "",
        "[counts]",
    ]
    for check in CHECKS:
        n = sum(1 for v in violations if v.check == check)
        lines.append(f"{check} = {n}")
    lines.append("")
    for v in violations:
        lines += [
            "[[allow]]",
            f"check = {_toml_str(v.check)}",
            f"upstream_path = {_toml_str(v.upstream_path)}",
            f"routine = {_toml_str(v.routine)}",
            f"target = {_toml_str(v.target)}",
            f"detail = {_toml_str(v.detail)}",
            "",
        ]
    return "\n".join(lines)


def _load_na_entries(repo_root: pathlib.Path) -> list[Any]:
    """Load N/A declarations (imported late: ``routine_na`` imports this)."""
    from flexmesa.testing import routine_na

    return routine_na.load_na(repo_root / routine_na.NA_RELPATH)


def verify_routine_map(
    repo_root: pathlib.Path,
    manifest: port_manifest.PortManifest | None = None,
    mesa_dir: pathlib.Path | None = None,
) -> list[str]:
    """Run the inventory consistency and ``routine_map`` gate checks.

    Args:
        repo_root: Repository checkout root.
        manifest: Loaded manifest (loaded from the repository when omitted).
        mesa_dir: Optional MESA 26.4.1 source tree. When it holds the
            manifest's Fortran files (a partial tree is ignored), the
            inventory is also regenerated and compared; otherwise only
            internal consistency is checked.

    Returns:
        Error messages (empty when the gate passes).
    """
    dash = repo_root / "benchmarks" / "dashboard"
    manifest = manifest or port_manifest.load_port_manifest(
        dash / "port_manifest.toml"
    )
    inv_path = repo_root / routine_inventory.INVENTORY_RELPATH
    if not inv_path.is_file():
        return [f"{routine_inventory.INVENTORY_RELPATH}: missing"]
    inventory = routine_inventory.load_inventory(inv_path)
    errors = routine_inventory.validate_inventory(inventory, manifest)
    if mesa_dir is not None and all(
        (mesa_dir / e.upstream_path).is_file()
        for e in manifest.entries
        if routine_inventory.is_covered(e.upstream_path)
    ):
        try:
            fresh = routine_inventory.render_inventory(
                routine_inventory.build_inventory(manifest, mesa_dir)
            )
        except (ValueError, OSError) as exc:
            errors.append(f"routine inventory regeneration failed: {exc}")
        else:
            if fresh != inv_path.read_text(encoding="utf-8"):
                errors.append("routine inventory differs from regeneration")
    errors.extend(
        structural_errors(manifest, inventory, _load_na_entries(repo_root))
    )
    found = collect_violations(manifest, inventory, repo_root)
    counts, allowed = load_allowlist(repo_root / ALLOWLIST_RELPATH)
    allowed_ids = {a.ident() for a in allowed}
    found_ids = {v.ident() for v in found}
    for v in found:
        if v.ident() not in allowed_ids:
            errors.append(
                f"{v.upstream_path}: routine_map[{v.routine!r}] fails "
                f"{v.check} ({v.detail!r}); fix it or justify it in "
                "the allow-list"
            )
    for a in allowed:
        if a.ident() not in found_ids:
            errors.append(
                f"{a.upstream_path}: routine_map[{a.routine!r}] no longer "
                f"fails {a.check}; remove it from the allow-list"
            )
    for check in CHECKS:
        n = sum(1 for a in allowed if a.check == check)
        if counts.get(check) != n:
            errors.append(
                f"allow-list counts[{check}]={counts.get(check)} != {n} entries"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, default=None)
    ap.add_argument("--mesa-dir", type=pathlib.Path, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--write-allowlist", action="store_true")
    ns = ap.parse_args(argv)
    root = ns.repo_root or pathlib.Path(__file__).resolve().parents[3]
    manifest = port_manifest.load_port_manifest(
        root / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    inventory = routine_inventory.load_inventory(
        root / routine_inventory.INVENTORY_RELPATH
    )
    found = collect_violations(manifest, inventory, root)
    if ns.report or ns.write_allowlist:
        total = sum(len(e.routine_map) for e in manifest.entries)
        print(f"routine_map entries: {total}")
        for check in CHECKS:
            n = sum(1 for v in found if v.check == check)
            print(f"{check}: {n} violation(s)")
        for v in found:
            print(f"  {v.check} {v.upstream_path} {v.routine} -> {v.detail}")
    if ns.write_allowlist:
        (root / ALLOWLIST_RELPATH).write_text(
            render_allowlist(found), encoding="utf-8"
        )
        return 0
    errors = verify_routine_map(root, manifest, ns.mesa_dir)
    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    if not errors:
        print("routine_map gate: zero new violations")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
