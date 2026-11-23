#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build the ``mesa-stepdumps-v1.tar.zst`` bundle and its SHA-256 manifest.

Packs the MESA 26.4.1 step dumps written by
``benchmarks/fortran_mesa/stepdumps/run_stepdumps.py`` (one
``<suite>/<case>.stepdump`` per case) into a deterministic tar archive
(sorted members, fixed mtime, root ownership) compressed with zstd, with a
``PROVENANCE.json`` at its root, and writes the per-file manifest
``src/flexmesa/data/bundle_hashes/mesa_stepdumps.sha256.tsv`` (#372).

Usage::

    python scripts/data/build_mesa_stepdumps.py --dumps <dir> --out <archive>

Prints the archive SHA-256 and the sizes for ``bundles.toml``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import tarfile
from typing import BinaryIO

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = "mesa_stepdumps"
CANONICAL_MTIME = 0
FORBIDDEN = (b"/home/", b"/tmp/", b"/mnt/", b"/Users/")
MANIFEST = REPO / "src/flexmesa/data/bundle_hashes/mesa_stepdumps.sha256.tsv"


def _zstd_writer(path: pathlib.Path) -> BinaryIO:
    from compression import zstd  # Python >= 3.14

    return zstd.open(path, "wb", level=19)  # type: ignore[return-value]


def _add(tf: tarfile.TarFile, rel: str, payload: bytes) -> tuple[str, int]:
    for needle in FORBIDDEN:
        if needle in payload:
            raise ValueError(f"machine path {needle!r} in {rel}")
    info = tarfile.TarInfo(name=rel)
    info.size = len(payload)
    info.mtime = CANONICAL_MTIME
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    tf.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _provenance(dumps: list[pathlib.Path]) -> bytes:
    head = {}
    for line in dumps[0].read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        key, _, val = line[1:].partition(":")
        head[key.strip()] = val.strip()
    doc = {
        "bundle": "mesa-stepdumps-v1",
        "mesa_version": head.get("mesa_version"),
        "compiler": head.get("compiler"),
        "mesa_sdk_version": head.get("mesa_sdk_version"),
        "omp_num_threads": head.get("omp_num_threads"),
        "steps_per_star": head.get("steps_per_star"),
        "window": "first models of the first ./rn stage (MESA_SKIP_OPTIONAL=t)",
        "recipe": "benchmarks/fortran_mesa/stepdumps/run_stepdumps.py "
        "with stepdump_patch.py (write statements only)",
        "cases": [f"{d.parent.name}/{d.stem}" for d in dumps],
        "not_dumped": "rsp_* cases (RSP has its own solver); "
        "star/1.4M_ms_op_mono and star/radiative_levitation skip themselves",
    }
    return (json.dumps(doc, indent=2) + "\n").encode("utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    dumps = sorted(args.dumps.glob("*/*.stepdump"))
    if not dumps:
        raise SystemExit(f"no *.stepdump files under {args.dumps}")
    rows: list[tuple[str, int, str]] = []
    with _zstd_writer(args.out) as zfh:
        with tarfile.open(
            fileobj=zfh, mode="w|", format=tarfile.PAX_FORMAT
        ) as tf:
            rel = f"{ROOT}/PROVENANCE.json"
            rows.append((*_add(tf, rel, _provenance(dumps)), rel))
            for d in dumps:
                rel = f"{ROOT}/{d.parent.name}/{d.name}"
                rows.append((*_add(tf, rel, d.read_bytes()), rel))
    MANIFEST.write_text(
        "# sha256\tsize_bytes\trelpath\n"
        + "\n".join(f"{s}\t{n}\t{r}" for s, n, r in rows)
        + "\n",
        encoding="utf-8",
    )
    arc = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"archive:             {args.out.name}")
    print(f"archive_sha256:      {arc}")
    print(f"archive_size_bytes:  {args.out.stat().st_size}")
    print(f"unpacked_size_bytes: {sum(n for _, n, _ in rows)}")
    print(f"members:             {len(rows)}")


if __name__ == "__main__":
    main()
