#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build FlexMESA physics data bundles and embedded crate assets from MESA.

Reconstructs the `data-v1` release assets and embedded `nets.bin` payload from
an extracted MESA `26.4.1` tree (`$MESA_DIR`, Zenodo record `19722306`):

1. `crates/flexmesa-net/data/nets.bin` (`FNET0001` map of `.net` and `add_*`)
2. `helm_table_full.bin.zst` (`FMHLM001` 9-jet `HELM` grid)
3. `kap_tables_full_v2.bin.zst` (`FMKAP001` full opacity table bundle)
4. `eos_opal_scvh_raw.tar.zst` (`FMDTRW01` `OPAL/SCVH` 4D EOS slices)
5. `eos_free_eos_raw.tar.zst` (`FMDTRW01` `FreeEOS` 4D EOS slices)
6. `eos_cms_raw.tar.zst` (`FMCMSR01` `CMS` H/He 4D EOS slices)

Every `.tar.zst` bundle embeds a `SHA256SUMS.tsv` manifest of unpacked member
hashes, and an external `<bundle>.sha256.tsv` manifest is written alongside the
archive (`N1-10`, `#141`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

import numpy as np

DT_RAW_MAGIC = b"FMDTRW01"
CMS_RAW_MAGIC = b"FMCMSR01"
NET_BUNDLE_MAGIC = b"FNET0001"


def sha256_file(path: pathlib.Path) -> str:
    """Return lowercase hex SHA-256 of `path`."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_nets_bin(mesa_dir: pathlib.Path, out_path: pathlib.Path) -> str:
    """Build `FNET0001` `nets.bin` from `$MESA_DIR/data/net_data/nets`."""
    nets_dir = mesa_dir / "data" / "net_data" / "nets"
    if not nets_dir.is_dir():
        raise FileNotFoundError(f"Missing MESA nets directory: {nets_dir}")
    entries: list[tuple[str, bytes]] = []
    for p in sorted(nets_dir.iterdir(), key=lambda item: item.name):
        if p.is_file() and p.name != "README":
            entries.append((p.name, p.read_bytes()))
    buf = bytearray(NET_BUNDLE_MAGIC)
    buf.extend(struct.pack("<I", len(entries)))
    for name, raw in entries:
        nb = name.encode("utf-8")
        buf.extend(struct.pack("<I", len(nb)))
        buf.extend(nb)
        buf.extend(struct.pack("<I", len(raw)))
        buf.extend(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(buf))
    return sha256_file(out_path)


def _parse_header_tokens(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return `(header_names, header_vals)` from a MESA EOS ASCII table."""
    non_empty = [ln.strip() for ln in lines[:12] if ln.strip()]
    return non_empty[0].split(), non_empty[1].split()


def convert_dt_ascii_to_raw(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Convert one `mesa-EOS5_*` or `mesa-FreeEOS_*` `.data` file to `.raw`.

    Writes the 84-byte `FMDTRW01` header (`magic`, `version`, `X`, `Z`,
    `num_logTs`, `logT_min`, `logT_max`, `del_logT`, `num_logQs`, `logQ_min`,
    `logQ_max`, `del_logQ`) followed by `num_logQs * num_logTs` cells of 16
    little-endian `float64` values in `(iq, it)` block order.
    """
    lines = src.read_text(encoding="utf-8").splitlines()
    names, vals = _parse_header_tokens(lines)
    hdr = dict(zip(names, vals, strict=True))
    version = int(hdr["version"])
    x_val = float(hdr["X"])
    z_val = float(hdr["Z"])
    n_lt = int(hdr["num_logTs"])
    lt_min = float(hdr["logT_min"])
    lt_max = float(hdr["logT_max"])
    d_lt = float(hdr["del_logT"])
    n_lq = int(hdr["num_logQs"])
    lq_min = float(hdr["logQ_min"])
    lq_max = float(hdr["logQ_max"])
    d_lq = float(hdr["del_logQ"])

    numeric_rows: list[list[float]] = []
    for ln in lines[4:]:
        s = ln.strip()
        if not s or s.startswith("logQ") or s.startswith("logT"):
            continue
        parts = s.split()
        if len(parts) == 17:
            # Data row: logT + 16 EOS columns
            numeric_rows.append(
                [float(tok.replace("D", "E")) for tok in parts[1:]]
            )
    arr = np.asarray(numeric_rows, dtype="<f8")
    if arr.shape != (n_lq * n_lt, 16):
        raise ValueError(
            f"{src.name}: expected ({n_lq * n_lt}, 16), got {arr.shape}"
        )
    header_bytes = struct.pack(
        "<8siddIdddIddd",
        DT_RAW_MAGIC,
        version,
        x_val,
        z_val,
        n_lt,
        lt_min,
        lt_max,
        d_lt,
        n_lq,
        lq_min,
        lq_max,
        d_lq,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fh:
        fh.write(header_bytes)
        fh.write(arr.tobytes(order="C"))


def convert_cms_ascii_to_raw(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Convert one `mesa-CMS_*.data` file to 76-byte header `FMCMSR01` `.raw`."""
    lines = src.read_text(encoding="utf-8").splitlines()
    names, vals = _parse_header_tokens(lines)
    hdr = dict(zip(names, vals, strict=True))
    version = int(hdr["version"])
    x_val = float(hdr["X"])
    n_lt = int(hdr["num_logTs"])
    lt_min = float(hdr["logT_min"])
    lt_max = float(hdr["logT_max"])
    d_lt = float(hdr["del_logT"])
    n_lr = int(hdr["num_logRhos"])
    lr_min = float(hdr["logRho_min"])
    lr_max = float(hdr["logRho_max"])
    d_lr = float(hdr["del_logRho"])

    numeric_rows: list[list[float]] = []
    for ln in lines[4:]:
        s = ln.strip()
        if not s or s.startswith("logT") or s.startswith("logRho"):
            continue
        parts = s.split()
        if len(parts) == 18:
            # Data row: logRho, logQ, + 16 EOS columns
            numeric_rows.append(
                [float(tok.replace("D", "E")) for tok in parts[2:]]
            )
    arr = np.asarray(numeric_rows, dtype="<f8")
    if arr.shape != (n_lt * n_lr, 16):
        raise ValueError(
            f"{src.name}: expected ({n_lt * n_lr}, 16), got {arr.shape}"
        )
    header_bytes = struct.pack(
        "<8sidIdddIddd",
        CMS_RAW_MAGIC,
        version,
        x_val,
        n_lt,
        lt_min,
        lt_max,
        d_lt,
        n_lr,
        lr_min,
        lr_max,
        d_lr,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fh:
        fh.write(header_bytes)
        fh.write(arr.tobytes(order="C"))


def compress_zst(src: pathlib.Path, dst: pathlib.Path, level: int = 19) -> str:
    """Compress `src` to `dst` with Zstandard and return SHA-256 of `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from compression import zstd as std_zstd  # type: ignore[import-not-found]

        with (
            src.open("rb") as fin,
            std_zstd.open(dst, "wb", level=level) as fout,
        ):
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        return sha256_file(dst)
    except ImportError:
        pass
    try:
        import zstandard  # type: ignore[import-not-found]

        cctx = zstandard.ZstdCompressor(level=level, threads=-1)
        with src.open("rb") as fin, dst.open("wb") as fout:
            cctx.copy_stream(fin, fout)
        return sha256_file(dst)
    except ImportError:
        pass
    subprocess.run(
        ["zstd", f"-{level}", "-T0", "-f", "-q", str(src), "-o", str(dst)],
        check=True,
    )
    return sha256_file(dst)


def decompress_zst(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Decompress `src` `.zst` archive to `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from compression import zstd as std_zstd  # type: ignore[import-not-found]

        with std_zstd.open(src, "rb") as fin, dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        return
    except ImportError:
        pass
    import zstandard  # type: ignore[import-not-found]

    dctx = zstandard.ZstdDecompressor()
    with src.open("rb") as fin, dst.open("wb") as fout:
        dctx.copy_stream(fin, fout)


def pack_deterministic_tar_zst(
    staging_root: pathlib.Path,
    rel_files: list[str],
    out_archive: pathlib.Path,
    manifest_out: pathlib.Path,
    level: int = 19,
) -> str:
    """Write a deterministic `.tar.zst` containing `rel_files` + `SHA256SUMS.tsv`."""
    rel_sorted = sorted(rel_files)
    manifest_lines = ["# sha256\tsize_bytes\trelpath"]
    for rel in rel_sorted:
        fp = staging_root / rel
        manifest_lines.append(f"{sha256_file(fp)}\t{fp.stat().st_size}\t{rel}")
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_bytes(manifest_bytes)

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
        tar_path = pathlib.Path(tmp_tar.name)
    try:
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
            for rel in rel_sorted:
                fp = staging_root / rel
                ti = tarfile.TarInfo(name=rel)
                ti.size = fp.stat().st_size
                ti.mtime = 0
                ti.uid = 0
                ti.gid = 0
                ti.uname = ""
                ti.gname = ""
                ti.mode = 0o644
                with fp.open("rb") as fh:
                    tf.addfile(ti, fh)
            ti_m = tarfile.TarInfo(name="SHA256SUMS.tsv")
            ti_m.size = len(manifest_bytes)
            ti_m.mtime = 0
            ti_m.uid = 0
            ti_m.gid = 0
            ti_m.uname = ""
            ti_m.gname = ""
            ti_m.mode = 0o644
            tf.addfile(ti_m, io.BytesIO(manifest_bytes))
        return compress_zst(tar_path, out_archive, level=level)
    finally:
        tar_path.unlink(missing_ok=True)


def build_kap_full_bundle(
    mesa_dir: pathlib.Path,
    out_dir: pathlib.Path,
    level: int = 19,
) -> tuple[str, str, int]:
    """Build `kap_tables_full_v2.bin.zst` via `flexmesa._rust` opacity loader."""
    with tempfile.TemporaryDirectory(prefix="flexmesa_kap_build_") as tmp:
        tmp_data = pathlib.Path(tmp)
        env = dict(os.environ)
        env["FLEXMESA_DATA_DIR"] = str(tmp_data)
        env["MESA_DIR"] = str(mesa_dir)
        env["JAX_PLATFORMS"] = "cpu"
        cmd = [
            sys.executable,
            "-c",
            "from flexmesa import kap; "
            "kap.kap_get_fixed_metal(0.01, 0.7, -4.0, 3.2, "
            "prefix='lowT_Freedman11', backend='rust')",
        ]
        subprocess.run(cmd, env=env, check=True)
        raw_bin = tmp_data / "kap_full" / "kap_tables_full.bin"
        if not raw_bin.is_file():
            raise RuntimeError("Failed to build kap_tables_full.bin")
        unpacked_sha = sha256_file(raw_bin)
        size_bytes = raw_bin.stat().st_size
        out_zst = out_dir / "kap_tables_full_v2.bin.zst"
        archive_sha = compress_zst(raw_bin, out_zst, level=level)
        manifest = out_dir / "kap_full.sha256.tsv"
        manifest.write_text(
            "# sha256\tsize_bytes\trelpath\n"
            f"{unpacked_sha}\t{size_bytes}\tkap_full/kap_tables_full.bin\n",
            encoding="utf-8",
        )
        return archive_sha, unpacked_sha, size_bytes


def main() -> int:
    """CLI entry point for building physics bundles from `$MESA_DIR`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("MESA_DIR", "")),
        help="Path to extracted MESA 26.4.1 root directory.",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
        help="Path to FlexMESA repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to write release `.zst` bundles and SHA-256 manifests.",
    )
    parser.add_argument(
        "--nets-only",
        action="store_true",
        help="Only rebuild crates/flexmesa-net/data/nets.bin.",
    )
    parser.add_argument(
        "--kap-only",
        action="store_true",
        help="Rebuild nets.bin and kap_tables_full_v2.bin.zst.",
    )
    parser.add_argument(
        "--install-local-kap",
        type=pathlib.Path,
        default=None,
        help="Optional local $FLEXMESA_DATA_DIR to update kap_tables_full.bin.",
    )
    args = parser.parse_args()

    if not args.mesa_dir or not (args.mesa_dir / "data").is_dir():
        raise SystemExit(
            "Provide a valid MESA 26.4.1 tree via --mesa-dir or $MESA_DIR."
        )

    nets_out = args.repo_root / "crates" / "flexmesa-net" / "data" / "nets.bin"
    nets_sha = build_nets_bin(args.mesa_dir, nets_out)
    print(f"nets.bin: sha256={nets_sha} bytes={nets_out.stat().st_size}")
    if args.nets_only:
        return 0

    if args.out_dir is None:
        raise SystemExit("--out-dir is required when building release bundles.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kap_arch_sha, kap_unp_sha, kap_size = build_kap_full_bundle(
        args.mesa_dir, args.out_dir
    )
    print(
        f"kap_tables_full_v2.bin.zst: archive_sha256={kap_arch_sha} "
        f"unpacked_sha256={kap_unp_sha} unpacked_bytes={kap_size}"
    )
    if args.install_local_kap is not None:
        dst_kap = args.install_local_kap / "kap_full" / "kap_tables_full.bin"
        decompress_zst(args.out_dir / "kap_tables_full_v2.bin.zst", dst_kap)
        print(f"Installed local {dst_kap} ({dst_kap.stat().st_size} bytes)")

    if args.kap_only:
        return 0

    with tempfile.TemporaryDirectory(prefix="flexmesa_eos_raw_") as tmp:
        stage = pathlib.Path(tmp)
        dt_rels: list[str] = []
        for p in sorted(
            (args.mesa_dir / "data" / "eosDT_data").glob("mesa-EOS5_*.data")
        ):
            rel = f"eos_4d_raw/{p.stem}.raw"
            convert_dt_ascii_to_raw(p, stage / rel)
            dt_rels.append(rel)
        dt_sha = pack_deterministic_tar_zst(
            stage,
            dt_rels,
            args.out_dir / "eos_opal_scvh_raw.tar.zst",
            args.out_dir / "eos_opal_scvh.sha256.tsv",
        )
        print(f"eos_opal_scvh_raw.tar.zst: sha256={dt_sha}")

        fe_rels: list[str] = []
        for p in sorted(
            (args.mesa_dir / "data" / "eosFreeEOS_data").glob(
                "mesa-FreeEOS_*.data"
            )
        ):
            rel = f"eos_4d_raw/{p.stem}.raw"
            convert_dt_ascii_to_raw(p, stage / rel)
            fe_rels.append(rel)
        fe_sha = pack_deterministic_tar_zst(
            stage,
            fe_rels,
            args.out_dir / "eos_free_eos_raw.tar.zst",
            args.out_dir / "eos_free_eos.sha256.tsv",
        )
        print(f"eos_free_eos_raw.tar.zst: sha256={fe_sha}")

        cms_rels: list[str] = []
        for p in sorted(
            (args.mesa_dir / "data" / "eosCMS_data").glob("mesa-CMS_*.data")
        ):
            rel = f"eos_4d_raw/{p.stem}.raw"
            convert_cms_ascii_to_raw(p, stage / rel)
            cms_rels.append(rel)
        cms_sha = pack_deterministic_tar_zst(
            stage,
            cms_rels,
            args.out_dir / "eos_cms_raw.tar.zst",
            args.out_dir / "eos_cms.sha256.tsv",
        )
        print(f"eos_cms_raw.tar.zst: sha256={cms_sha}")
        shutil.rmtree(stage, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
