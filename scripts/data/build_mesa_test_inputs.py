#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build ``mesa-test-inputs-v2.tar.zst`` and its per-file SHA-256 manifest.

Extracts upstream MESA release ``26.4.1`` test inputs, module ``test_output``
goldens, ``*.defaults`` files, Fortran source files (for offline ``file:line``
citation verification in CI), and installed ``data/`` reference tables into a
deterministic ``.tar.zst`` bundle (`#141`, `N1-10`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import pathlib
import posixpath
import stat
import tarfile
import zipfile
from typing import BinaryIO

EXPECTED_ZIP_MD5 = "994fd28f38da92efc80e38ca11d8317b"
CANONICAL_MTIME = 1776902400  # 2026-04-23T00:00:00Z (MESA 26.4.1 release date)

INSTALLED_DATA_SUBDIRS: tuple[str, ...] = (
    "atm_data",
    "chem_data",
    "net_data",
    "roche_data",
    "star_data",
)

SOURCE_SUFFIXES: tuple[str, ...] = (
    ".f",
    ".f90",
    ".inc",
    ".dek",
    ".defaults",
    ".list",
)


def _open_zstd_writer(path: pathlib.Path, level: int = 10) -> BinaryIO:
    """Open a streaming binary zstd compressor for ``path``."""
    try:
        from compression import zstd as std_zstd

        return std_zstd.open(path, "wb", level=level)  # type: ignore[return-value]
    except ImportError:
        import zstandard

        fh = path.open("wb")
        cctx = zstandard.ZstdCompressor(level=level, threads=4)
        return cctx.stream_writer(fh, closefd=True)  # type: ignore[return-value]


def _md5_file(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _should_keep_zip_relpath(rel: str) -> bool:
    parts = rel.split("/")
    if not parts or any(p in {".git", "__pycache__"} for p in parts):
        return False
    if rel == "data/version_number":
        return True
    if parts[0] == "data":
        return False
    if len(parts) >= 2:
        two = f"{parts[0]}/{parts[1]}"
        if parts[1] in {"test", "test_suite", "defaults"}:
            return True
        if two in {"astero/work", "star/rsp2_utils"} or (
            parts[0] == "star" and parts[1].startswith("dev_cases_")
        ):
            return True
    return rel.endswith(SOURCE_SUFFIXES)


def _add_bytes_member(
    tf: tarfile.TarFile,
    relpath: str,
    payload: bytes,
    mode: int = 0o644,
) -> tuple[str, int]:
    sha = hashlib.sha256(payload).hexdigest()
    info = tarfile.TarInfo(name=relpath)
    info.size = len(payload)
    info.mtime = CANONICAL_MTIME
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    tf.addfile(info, io.BytesIO(payload))
    return sha, len(payload)


def build_bundle(
    mesa_zip: pathlib.Path,
    mesa_dir: pathlib.Path,
    out_archive: pathlib.Path,
    manifest_out: pathlib.Path,
    skip_md5: bool = False,
) -> None:
    """Build ``mesa-test-inputs-v2.tar.zst`` and write its SHA-256 manifest."""
    if not skip_md5:
        actual_md5 = _md5_file(mesa_zip)
        if actual_md5 != EXPECTED_ZIP_MD5:
            msg = (
                f"{mesa_zip.name} MD5 mismatch: expected {EXPECTED_ZIP_MD5}, "
                f"got {actual_md5}"
            )
            raise ValueError(msg)

    staged: dict[str, tuple[bytes, int]] = {}
    with zipfile.ZipFile(mesa_zip, "r") as zf:
        top_prefix = zf.namelist()[0].split("/")[0] + "/"
        by_name = {zi.filename: zi for zi in zf.infolist()}
        for zi in zf.infolist():
            if zi.is_dir() or not zi.filename.startswith(top_prefix):
                continue
            rel = zi.filename[len(top_prefix) :]
            if not _should_keep_zip_relpath(rel):
                continue
            raw_mode = (zi.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(raw_mode):
                link_rel = zf.read(zi).decode("utf-8").strip()
                target_name = posixpath.normpath(
                    posixpath.join(posixpath.dirname(zi.filename), link_rel)
                )
                target_zi = by_name[target_name]
                data = zf.read(target_zi)
                unix_mode = (target_zi.external_attr >> 16) & 0o777
            else:
                data = zf.read(zi)
                unix_mode = raw_mode & 0o777
            mode = 0o755 if (unix_mode & 0o111) else 0o644
            staged[rel] = (data, mode)

    data_root = mesa_dir / "data"
    for sub in INSTALLED_DATA_SUBDIRS:
        sub_dir = data_root / sub
        if not sub_dir.is_dir():
            raise FileNotFoundError(f"Missing installed data dir: {sub_dir}")
        for fpath in sorted(sub_dir.rglob("*")):
            if not fpath.is_file():
                continue
            rel_in_data = fpath.relative_to(data_root).as_posix()
            if "/cache/" in f"/{rel_in_data}" or rel_in_data.endswith(".bin"):
                continue
            rel = f"data/{rel_in_data}"
            mode = 0o755 if os.access(fpath, os.X_OK) else 0o644
            staged[rel] = (fpath.read_bytes(), mode)

    version_txt = staged.get("data/version_number", (b"", 0))[0].decode(
        "utf-8", errors="replace"
    )
    if version_txt.strip() != "r26.4.1":
        raise ValueError(f"Expected r26.4.1, got {version_txt.strip()!r}")

    out_archive.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    unpacked_bytes = 0
    with _open_zstd_writer(out_archive, level=10) as zout:
        with tarfile.open(fileobj=zout, mode="w|") as tf:
            for rel in sorted(staged):
                payload, mode = staged[rel]
                sha, sz = _add_bytes_member(tf, rel, payload, mode=mode)
                rows.append(f"{sha}\t{sz}\t{rel}")
                unpacked_bytes += sz

            sums_payload = (
                "# sha256\tsize_bytes\trelpath\n" + "\n".join(rows) + "\n"
            ).encode("utf-8")
            sha_s, sz_s = _add_bytes_member(tf, "SHA256SUMS.tsv", sums_payload)
            unpacked_bytes += sz_s

    manifest_out.write_text(
        "# sha256\tsize_bytes\trelpath\n"
        + "\n".join(rows)
        + f"\n{sha_s}\t{sz_s}\tSHA256SUMS.tsv\n",
        encoding="utf-8",
    )

    arc_sha = hashlib.sha256(out_archive.read_bytes()).hexdigest()
    arc_sz = out_archive.stat().st_size
    print(f"Built {out_archive.name}:")
    print(f"  files:               {len(rows) + 1}")
    print(f"  archive_size_bytes:  {arc_sz}")
    print(f"  archive_sha256:      {arc_sha}")
    print(f"  unpacked_size_bytes: {unpacked_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-zip",
        type=pathlib.Path,
        required=True,
        help="Path to mesa-26.04.1.zip (Zenodo record 19722306).",
    )
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        required=True,
        help="Path to installed MESA 26.4.1 tree (for unpacked data/ tables).",
    )
    parser.add_argument(
        "--out-archive",
        type=pathlib.Path,
        required=True,
        help="Output path for mesa-test-inputs-v2.tar.zst.",
    )
    parser.add_argument(
        "--manifest-out",
        type=pathlib.Path,
        required=True,
        help="Output path for mesa_test_inputs.sha256.tsv.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification of --mesa-zip.",
    )
    args = parser.parse_args()
    build_bundle(
        args.mesa_zip.resolve(),
        args.mesa_dir.resolve(),
        args.out_archive.resolve(),
        args.manifest_out.resolve(),
        skip_md5=bool(args.skip_md5),
    )


if __name__ == "__main__":
    main()
