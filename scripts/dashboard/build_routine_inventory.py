#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate or check ``benchmarks/dashboard/routine_inventory.json`` (#611).

Parses every Fortran file listed in ``benchmarks/dashboard/port_manifest.toml``
under ``--mesa-dir`` (a MESA 26.4.1 source tree), verifies each file's SHA-256
against the manifest, and writes the routine inventory. With ``--check`` the
inventory is regenerated and compared when ``--mesa-dir`` is given, and is
otherwise validated for internal consistency against the manifest only.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flexmesa.testing import port_manifest, routine_inventory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the generator or its ``--check`` mode."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesa-dir", type=pathlib.Path, default=None)
    ap.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args(argv)
    root: pathlib.Path = ns.repo_root
    out = root / routine_inventory.INVENTORY_RELPATH
    manifest = port_manifest.load_port_manifest(
        root / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    if ns.check:
        if not out.is_file():
            print(f"ERROR: missing {out.name}", file=sys.stderr)
            return 1
        errors = routine_inventory.validate_inventory(
            routine_inventory.load_inventory(out), manifest
        )
        if ns.mesa_dir is not None:
            try:
                fresh = routine_inventory.render_inventory(
                    routine_inventory.build_inventory(manifest, ns.mesa_dir)
                )
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if fresh != out.read_text(encoding="utf-8"):
                    errors.append(
                        "committed inventory differs from regeneration"
                    )
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        if not errors:
            mode = "regenerated" if ns.mesa_dir else "internal"
            print(f"routine inventory consistent ({mode} check)")
        return 1 if errors else 0
    if ns.mesa_dir is None:
        ap.error("--mesa-dir is required unless --check is given")
    try:
        payload = routine_inventory.build_inventory(manifest, ns.mesa_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out.write_text(
        routine_inventory.render_inventory(payload), encoding="utf-8"
    )
    head = payload["header"]
    print(
        f"Wrote {out.name}: {head['files']} files, {head['routines']} routines"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
