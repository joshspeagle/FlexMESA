#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a ``mesa-replay-<version>.tar.zst`` bundle and its manifest.

Packs the MESA 26.4.1 record/replay captures written by
``benchmarks/fortran_mesa/replay/run_capture.py`` (one ``<suite>/<case>.replay``
and the ``<suite>/<case>.stepdump`` of the same run per case) into a
deterministic tar archive (sorted members, fixed mtime, root ownership)
compressed with zstd, with a ``PROVENANCE.json`` at its root that lists the
records per family and case, and writes the per-file manifest
(``sha256``, ``size_bytes``, ``relpath``) in the layout of
``src/flexmesa/data/bundle_hashes/`` (#552).

Usage::

    python scripts/data/build_mesa_replay.py --captures <dir> \\
        --out <archive> --manifest <tsv> [--version v1] [--level 19] \\
        [--threads 4]

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

ROOT = "mesa_replay"
CANONICAL_MTIME = 0
FORBIDDEN = (b"/home/", b"/tmp/", b"/mnt/", b"/Users/")
HEADER_KEYS = ("mesa_version", "compiler", "mesa_sdk_version")


def _zstd_writer(path: pathlib.Path, level: int, threads: int) -> BinaryIO:
    from compression import zstd  # Python >= 3.14

    options = {
        zstd.CompressionParameter.compression_level: level,
        zstd.CompressionParameter.nb_workers: threads,
    }
    return zstd.open(path, "wb", options=options)  # type: ignore[no-any-return]


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


def _scan(path: pathlib.Path) -> tuple[dict[str, str], dict[str, int]]:
    """Returns the ``# key: value`` header and the record count per family
    of one capture."""
    head: dict[str, str] = {}
    counts: dict[str, int] = {}
    with path.open("rb") as fh:
        lines = list(fh)
    for raw in lines:
        if raw.startswith(b"#"):
            key, _, val = raw[1:].decode("utf-8").partition(":")
            head.setdefault(key.strip(), val.strip())
            continue
        tag = raw.split(b" ", 1)[0].strip().decode("ascii")
        if tag in ("", "RUN", "SCHEMA", "CONFIG"):
            continue
        counts[tag] = counts.get(tag, 0) + 1
    for key in HEADER_KEYS:
        if not head.get(key):
            raise ValueError(f"capture header lacks `{key}`")
    return head, counts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--level", type=int, default=19)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    captures = sorted(args.captures.glob("*/*.replay"))
    if not captures:
        raise SystemExit(f"no *.replay files under {args.captures}")
    rows: list[tuple[str, int, str]] = []
    cases: dict[str, dict[str, int]] = {}
    heads: list[dict[str, str]] = []
    members: list[tuple[str, pathlib.Path]] = []
    for cap in captures:
        ident = f"{cap.parent.name}/{cap.stem}"
        head, counts = _scan(cap)
        heads.append(head)
        cases[ident] = counts
        members.append((f"{ROOT}/{ident}.replay", cap))
        dump = cap.with_suffix(".stepdump")
        if dump.exists():
            members.append((f"{ROOT}/{ident}.stepdump", dump))
    for key in HEADER_KEYS:
        values = {h[key] for h in heads}
        if len(values) != 1:
            raise SystemExit(f"captures disagree on {key}: {sorted(values)}")
    first = heads[0]
    doc = {
        "bundle": f"mesa-replay-{args.version}",
        "mesa_version": first["mesa_version"],
        "compiler": first["compiler"],
        "mesa_sdk_version": first["mesa_sdk_version"],
        "omp_num_threads": first.get("omp_num_threads"),
        "models_per_star": first.get("models_per_star"),
        "window": "first models of the first ./rn stage (MESA_SKIP_OPTIONAL=t)",
        "recipe": "benchmarks/fortran_mesa/replay/run_capture.py with "
        "replay_patch.py and stepdump_patch.py (write statements only); "
        "replay with `flexmesa-replay <case>.replay`",
        "families": {
            "TDC": "set_TDC calls in Get_results (turb_support.f90)",
            "MLT": "set_MLT calls in Get_results (turb_support.f90)",
            "GETR": "Get_results arguments and results (do1_mlt_eval)",
            "EOS": "get_eos call of do_eos_for_cell (micro.f90:357-361) "
            "with the run's CONFIG EOS line",
        },
        "records": cases,
    }
    with _zstd_writer(args.out, args.level, args.threads) as zfh:
        with tarfile.open(
            fileobj=zfh, mode="w|", format=tarfile.PAX_FORMAT
        ) as tf:
            rel = f"{ROOT}/PROVENANCE.json"
            prov = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
            rows.append((*_add(tf, rel, prov), rel))
            for rel, path in members:
                rows.append((*_add(tf, rel, path.read_bytes()), rel))
    args.manifest.write_text(
        "# sha256\tsize_bytes\trelpath\n"
        + "\n".join(f"{s}\t{n}\t{r}" for s, n, r in rows)
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256()
    with args.out.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    print(f"archive:             {args.out.name}")
    print(f"archive_sha256:      {digest.hexdigest()}")
    print(f"archive_size_bytes:  {args.out.stat().st_size}")
    print(f"unpacked_size_bytes: {sum(n for _, n, _ in rows)}")
    print(f"members:             {len(rows)}")


if __name__ == "__main__":
    main()
