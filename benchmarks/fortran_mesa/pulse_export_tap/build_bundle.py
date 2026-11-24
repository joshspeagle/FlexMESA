#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Package tapped MESA pulse-export calls as a data bundle (#491).

Reads `call_NNNN.{dump,flags,out,osc}` from a `pulse_dump/` directory
written by this recipe's tap during `astero/fast_from_file`, and writes
`mesa-pulse-export-v1.tar.zst` with members
`mesa_pulse_export/astero/fast_from_file/call_NNNN.*` and
`mesa_pulse_export/PROVENANCE.json`, plus the per-file SHA-256 manifest read
by `flexmesa data verify`. Tar members carry a fixed mtime and owner so the
archive is reproducible. The MESA version, compiler, build and SDK are read
from the run's own `LOGS/history.data` header.

Usage:
    python build_bundle.py <case_work_dir> <archive_out> <manifest_out> \
        [--calls N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tarfile
from typing import Any

BUNDLE_DIR = "mesa_pulse_export"
CASE = "astero/fast_from_file"
SUFFIXES = ("dump", "flags", "out", "osc")


def _load_builder() -> Any:
    """Import `scripts/data/build_mesa_histories.py` for its tar helpers."""
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "scripts"
        / "data"
        / "build_mesa_histories.py"
    )
    spec = importlib.util.spec_from_file_location("build_mesa_histories", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history_header(case_dir: pathlib.Path) -> dict[str, str]:
    """MESA header fields of the run's `LOGS/history.data` (unquoted)."""
    lines = (case_dir / "LOGS" / "history.data").read_text().splitlines()
    names, values = lines[1].split(), lines[2].split()
    return {n: v.strip('"') for n, v in zip(names, values, strict=False)}


def main() -> int:
    """Build the archive, manifest and provenance.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case_dir", type=pathlib.Path)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--calls", type=int, default=4)
    args = parser.parse_args()
    builder = _load_builder()
    tap = args.case_dir / "pulse_dump"
    members: list[tuple[str, bytes]] = []
    calls: list[dict[str, Any]] = []
    for n in range(1, args.calls + 1):
        files = {sfx: tap / f"call_{n:04d}.{sfx}" for sfx in SUFFIXES}
        if not all(p.is_file() for p in files.values()):
            print(f"call {n}: missing tap files", file=sys.stderr)
            return 1
        fmt = files["dump"].read_text().splitlines()[0].split()[1]
        calls.append({"call": n, "format": fmt})
        for sfx, path in files.items():
            members.append(
                (f"{BUNDLE_DIR}/{CASE}/call_{n:04d}.{sfx}", path.read_bytes())
            )
    hdr = _history_header(args.case_dir)
    provenance = {
        "bundle": "mesa-pulse-export-v1",
        "issue": "#491",
        "case": CASE,
        "mesa_version_number": hdr.get("version_number", ""),
        "compiler": hdr.get("compiler", ""),
        "build": hdr.get("build", ""),
        "mesa_sdk_version": hdr.get("MESA_SDK_version", ""),
        "math_backend": hdr.get("math_backend", ""),
        "recipe": "benchmarks/fortran_mesa/pulse_export_tap/",
        "method": (
            "-Wl,--wrap=__pulse_MOD_export_pulse_data linker tap: dumps the "
            "star_info fields the GYRE, FGONG and OSC producers read (IEEE-754 "
            "hex), calls MESA's export_pulse_data, keeps the file it wrote "
            "(.out), and calls it again with data_format 'OSC' (.osc). No "
            "MESA source is modified."
        ),
        "calls": calls,
    }
    prov = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    members.insert(0, (f"{BUNDLE_DIR}/PROVENANCE.json", prov))
    rows = ["# sha256\tsize_bytes\trelpath"]
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with builder._open_zstd_writer(args.archive) as zout:
        with tarfile.open(fileobj=zout, mode="w|") as tf:
            for relpath, payload in members:
                sha, size = builder._add_member(tf, relpath, payload)
                rows.append(f"{sha}\t{size}\t{relpath}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"archive_sha256 = {builder._sha256_file(args.archive)}")
    print(f"archive_size_bytes = {args.archive.stat().st_size}")
    print(f"unpacked_size_bytes = {sum(len(p) for _, p in members)}")
    print(f"members = {len(members)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
