#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate or check the routine-level port table (#611).

Writes ``notes/audit/routine_table.md`` and ``notes/audit/routine_table.json``
from ``benchmarks/dashboard/routine_inventory.json`` and the ``routine_map``
entries of ``benchmarks/dashboard/port_manifest.toml``. With ``--check`` the
files are regenerated in memory and compared with the committed copies. No
MESA source is needed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flexmesa.testing import routine_table  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Write or check the derived routine table files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args(argv)
    root: pathlib.Path = ns.repo_root
    expected = routine_table.render_outputs(root)
    bad = 0
    for path, text in expected.items():
        rel = path.relative_to(root).as_posix()
        if ns.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(
                    f"ERROR: {rel} differs from regeneration", file=sys.stderr
                )
                bad += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"Wrote {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
