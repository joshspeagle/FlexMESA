#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Package the thinned per-track astero histories as a data bundle (#430).

Reads ``<thinned_dir>/<case>/track_NNNN.data`` (written by
``thin_track_histories.py``) and writes ``mesa-astero-tracks-v1.tar.zst``
with members ``mesa_astero_tracks/PROVENANCE.json`` and
``mesa_astero_tracks/astero/<case>/track_NNNN.data``, plus the per-file
SHA-256 manifest read by ``flexmesa data verify``. The tar members carry a
fixed mtime and owner so the archive is reproducible.

``PROVENANCE.json`` records the MESA version, compiler, build and SDK read
from the tracks' own MESA headers, the recipe, and for every case whether
the last track is byte-identical below its header to the stored thinned
reference of the whole run (``thinned/astero/<case>.data`` in
``mesa-histories-v1``), whose SHA-256 it also records.

Usage:
    python build_bundle.py <thinned_dir> <stored_thinned_astero_dir> \
        <archive_out> <manifest_out>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys
import tarfile
from typing import Any

BUNDLE_DIR = "mesa_astero_tracks"
CASES = (
    "example_astero",
    "fast_from_file",
    "fast_newuoa",
    "fast_scan_grid",
    "fast_simplex",
)
_TRACK_RE = re.compile(r"^track_(\d{4})\.data$")
_HEADER_LINES = 3


def _load_builder() -> Any:
    """Import ``scripts/data/build_mesa_histories.py`` for its tar helpers."""
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


def _mesa_header(payload: bytes) -> dict[str, str]:
    """Return the MESA history header fields of ``payload`` (unquoted)."""
    lines = payload.decode("utf-8").splitlines()
    names, values = lines[1].split(), lines[2].split()
    return {n: v.strip('"') for n, v in zip(names, values, strict=True)}


def main() -> int:
    """Build the archive, manifest and provenance.

    Returns:
        Process exit code (``0`` on success, ``1`` on a failed check).
    """
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    thinned = pathlib.Path(sys.argv[1])
    stored = pathlib.Path(sys.argv[2])
    archive = pathlib.Path(sys.argv[3])
    manifest = pathlib.Path(sys.argv[4])
    builder = _load_builder()

    members: list[tuple[str, bytes]] = []
    cases: dict[str, Any] = {}
    header: dict[str, str] = {}
    ok = True
    for case in CASES:
        tracks = sorted(
            (int(m.group(1)), p)
            for p in (thinned / case).iterdir()
            if (m := _TRACK_RE.match(p.name)) is not None
        )
        numbers = [n for n, _ in tracks]
        if numbers != list(range(1, len(numbers) + 1)):
            print(f"{case}: tracks not numbered 1..N: {numbers}")
            return 1
        last = b""
        for number, path in tracks:
            last = path.read_bytes()
            members.append(
                (f"{BUNDLE_DIR}/astero/{case}/track_{number:04d}.data", last)
            )
            header = header or _mesa_header(last)
        ref_path = stored / f"{case}.data"
        ref = ref_path.read_bytes()
        same = (
            ref.split(b"\n")[_HEADER_LINES:]
            == last.split(b"\n")[_HEADER_LINES:]
        )
        ok = ok and same
        cases[case] = {
            "tracks": len(tracks),
            "last_track_matches_stored_reference_below_header": same,
            "stored_reference": f"mesa-histories-v1:thinned/astero/{case}.data",
            "stored_reference_sha256": hashlib.sha256(ref).hexdigest(),
        }
    if not ok:
        print("a last track differs from its stored reference", file=sys.stderr)
        return 1

    provenance = {
        "bundle": "mesa-astero-tracks-v1",
        "issue": "#430",
        "mesa_version_number": header.get("version_number", ""),
        "compiler": header.get("compiler", ""),
        "build": header.get("build", ""),
        "mesa_sdk_version": header.get("MESA_SDK_version", ""),
        "math_backend": header.get("math_backend", ""),
        "recipe": "benchmarks/fortran_mesa/astero_track_tap/",
        "method": (
            "-Wl,--wrap=__run_star_support_MOD_run1_star linker tap copies "
            "LOGS/history.data after every run1_star call "
            "(astero_run_support.f90:224); no MESA source is modified. Each "
            "track is thinned with build_mesa_histories."
            "thin_history_preserving_mesa_header (6-line MESA header kept)."
        ),
        "track_numbering": (
            "1-based order of run1_star calls, accepted or not by the search"
        ),
        "cases": cases,
    }
    prov_bytes = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    members.insert(0, (f"{BUNDLE_DIR}/PROVENANCE.json", prov_bytes))

    rows: list[str] = ["# sha256\tsize_bytes\trelpath"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with builder._open_zstd_writer(archive) as zout:
        with tarfile.open(fileobj=zout, mode="w|") as tf:
            for relpath, payload in members:
                sha, size = builder._add_member(tf, relpath, payload)
                rows.append(f"{sha}\t{size}\t{relpath}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    total = sum(len(p) for _, p in members)
    print(f"archive_sha256 = {builder._sha256_file(archive)}")
    print(f"archive_size_bytes = {archive.stat().st_size}")
    print(f"unpacked_size_bytes = {total}")
    print(f"members = {len(members)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
