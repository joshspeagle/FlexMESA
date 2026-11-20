#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build FlexMESA physics data bundles and embedded crate assets from MESA.

Reconstructs the `data-v1` release assets and embedded `nets.bin` payload from
an extracted MESA `26.4.1` tree (`$MESA_DIR`, Zenodo record `19722306`):

1. `crates/flexmesa-net/data/nets.bin` (`FNET0001` map of `.net` and `add_*`)
2. `helm_table_full.bin.zst` (`FMHLM001` 9-jet `HELM` grid)
3. `kap_tables_full_v2.bin.zst` (`FMKAP001` full opacity table bundle)
4. `eos_opal_scvh_raw_v1.tar.zst` (`FMDTRW01` `OPAL/SCVH` 4D EOS slices)
5. `eos_free_eos_raw_v1.tar.zst` (`FMDTRW01` `FreeEOS` 4D EOS slices)
6. `eos_cms_raw_v1.tar.zst` (`FMCMSR01` `CMS` H/He 4D EOS slices)

Every `.tar.zst` bundle writes a deterministic `<bundle>.sha256.tsv` manifest of
unpacked member hashes alongside the archive (`N1-10`, `#141`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import struct
import subprocess
import tarfile
import tempfile
import tomllib

import numpy as np

DT_RAW_MAGIC = b"FMDTRW01"
CMS_RAW_MAGIC = b"FMCMSR01"
HELM_FULL_MAGIC = b"FMHELM01"
NET_BUNDLE_MAGIC = b"FNET0001"
HELM_IMAX = 2701
HELM_JMAX = 1001
OPAL_SCVH_STEMS: tuple[str, ...] = (
    "mesa-eosDT_00z00x",
    "mesa-eosDT_00z20x",
    "mesa-eosDT_00z40x",
    "mesa-eosDT_00z60x",
    "mesa-eosDT_00z80x",
    "mesa-eosDT_00z100x",
    "mesa-eosDT_02z00x",
    "mesa-eosDT_02z20x",
    "mesa-eosDT_02z40x",
    "mesa-eosDT_02z60x",
    "mesa-eosDT_02z80x",
    "mesa-eosDT_04z00x",
    "mesa-eosDT_04z20x",
    "mesa-eosDT_04z40x",
    "mesa-eosDT_04z60x",
    "mesa-eosDT_04z80x",
)


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


def _extract_header_and_body_lines(
    src: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Return `(11_header_value_tokens, body_lines)` from a MESA EOS `.data`.

    Upstream MESA `.data` headers format multi-word labels with spaces
    (`num logTs`, `logT min`, `eos/private/eosdt_load_tables.f90:275-282`), so
    header metadata is parsed positionally from the second non-empty line.
    """
    lines = src.read_text(encoding="utf-8").splitlines()
    non_empty_indices = [idx for idx, ln in enumerate(lines) if ln.strip()]
    if len(non_empty_indices) < 2:
        raise ValueError(f"{src.name}: missing 2-line MESA EOS header")
    vals = lines[non_empty_indices[1]].replace("D", "E").split()
    if len(vals) < 11:
        raise ValueError(
            f"{src.name}: expected >= 11 header values, got {len(vals)}"
        )
    body = lines[non_empty_indices[1] + 1 :]
    return vals, body


def convert_dt_ascii_to_raw(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Convert one `mesa-eosDT_*` or `mesa-FreeEOS_*` `.data` file to `.raw`.

    Writes the 84-byte `FMDTRW01` header (`magic`, `version`, `X`, `Z`,
    `num_logTs`, `logT_min`, `logT_max`, `del_logT`, `num_logQs`, `logQ_min`,
    `logQ_max`, `del_logQ`) followed by `num_logQs * num_logTs` cells of the 16
    EOS columns (`parts[1:17]`, ignoring trailing FreeEOS diagnostic columns) in
    little-endian `float64` `(iq, it)` block order.
    """
    vals, body = _extract_header_and_body_lines(src)
    version = int(float(vals[0]))
    x_val = float(vals[1])
    z_val = float(vals[2])
    n_lt = int(float(vals[3]))
    lt_min = float(vals[4])
    lt_max = float(vals[5])
    d_lt = float(vals[6])
    n_lq = int(float(vals[7]))
    lq_min = float(vals[8])
    lq_max = float(vals[9])
    d_lq = float(vals[10])

    target_rows = n_lq * n_lt
    numeric_rows: list[list[float]] = []
    for ln in body:
        s = ln.strip()
        if not s or s.startswith("logQ") or s.startswith("logT"):
            continue
        parts = s.split()
        if len(parts) >= 17:
            # Data row: logT + 16 EOS columns (+5 trailing cols in FreeEOS).
            # Stop at n_lq * n_lt matching upstream `do iQ=1,ep% num_logQs` in
            # `eos/private/eosdt_load_tables.f90:328-344` (`mesa-FreeEOS_04z50x`
            # carries unreferenced rows past `logQ_max = 3.23`).
            numeric_rows.append(
                [float(tok.replace("D", "E")) for tok in parts[1:17]]
            )
            if len(numeric_rows) == target_rows:
                break
    arr = np.asarray(numeric_rows, dtype="<f8")
    if arr.shape != (target_rows, 16):
        raise ValueError(
            f"{src.name}: expected ({target_rows}, 16), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{src.name}: contains non-finite values")
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
    """Convert one `mesa-CMS_*.data` file to 76-byte `FMCMSR01` `.raw`."""
    vals, body = _extract_header_and_body_lines(src)
    version = int(float(vals[0]))
    x_val = float(vals[1])
    n_lt = int(float(vals[3]))
    lt_min = float(vals[4])
    lt_max = float(vals[5])
    d_lt = float(vals[6])
    n_lr = int(float(vals[7]))
    lr_min = float(vals[8])
    lr_max = float(vals[9])
    d_lr = float(vals[10])

    numeric_rows: list[list[float]] = []
    for ln in body:
        s = ln.strip()
        if not s or s.startswith("logT") or s.startswith("logRho"):
            continue
        parts = s.split()
        if len(parts) >= 18:
            # Data row: logT, logRho, + 16 EOS columns (+ dsp, dse in CMS)
            numeric_rows.append(
                [float(tok.replace("D", "E")) for tok in parts[2:18]]
            )
    arr = np.asarray(numeric_rows, dtype="<f8")
    if arr.shape != (n_lt * n_lr, 16):
        raise ValueError(
            f"{src.name}: expected ({n_lt * n_lr}, 16), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{src.name}: contains non-finite values")
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
        from compression import zstd as std_zstd

        with (
            src.open("rb") as fin,
            std_zstd.open(dst, "wb", level=level) as fout,
        ):
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        return sha256_file(dst)
    except ImportError:
        pass
    try:
        import zstandard

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
        from compression import zstd as std_zstd

        with std_zstd.open(src, "rb") as fin, dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        return
    except ImportError:
        pass
    import zstandard

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
    """Write a deterministic `.tar.zst` and external `.sha256.tsv` manifest."""
    if not rel_files:
        raise ValueError(f"{out_archive.name}: refusing to pack 0 member files")
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
        return compress_zst(tar_path, out_archive, level=level)
    finally:
        tar_path.unlink(missing_ok=True)


def convert_helm_ascii_to_full(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    imax: int = HELM_IMAX,
    jmax: int = HELM_JMAX,
) -> None:
    """Convert `helm_table.dat` into an `FMHELM01` `imax x jmax x 21` `.bin`."""
    n_total = imax * jmax
    sec_sizes = (9, 4, 4, 4)
    chunks: list[np.ndarray] = []
    with src.open("r", encoding="utf-8") as fh:
        lines = (ln.strip() for ln in fh)
        non_empty = (ln for ln in lines if ln)
        for sec_idx, ncols in enumerate(sec_sizes):
            vals = np.empty((n_total, ncols), dtype="<f8")
            for idx in range(n_total):
                row = next(non_empty, None)
                if row is None:
                    raise ValueError(
                        f"{src.name}: unexpected EOF in section "
                        f"{sec_idx} row {idx}"
                    )
                toks = row.replace("D", "E").split()
                if len(toks) != ncols:
                    raise ValueError(
                        f"{src.name}: expected {ncols} cols at section "
                        f"{sec_idx} row {idx}, got {len(toks)}"
                    )
                vals[idx] = [float(t) for t in toks]
            if not np.all(np.isfinite(vals)):
                raise ValueError(
                    f"{src.name}: non-finite values in HELM section {sec_idx}"
                )
            chunks.append(vals)
        if next(non_empty, None) is not None:
            raise ValueError(
                f"{src.name}: unexpected trailing data after section 3"
            )
    merged = np.concatenate(chunks, axis=1).astype("<f8", copy=False)
    header = struct.pack("<8sII", HELM_FULL_MAGIC, imax, jmax)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fout:
        fout.write(header)
        fout.write(merged.tobytes(order="C"))


def build_helm_full_bundle(
    mesa_dir: pathlib.Path,
    out_dir: pathlib.Path,
    level: int = 19,
) -> tuple[str, str, int]:
    """Build `helm_table_full.bin.zst` and SHA manifest from `$MESA_DIR`."""
    src_dat = mesa_dir / "data" / "eosDT_data" / "helm_table.dat"
    if not src_dat.is_file():
        raise FileNotFoundError(f"Missing MESA HELM table: {src_dat}")
    with tempfile.TemporaryDirectory(prefix="flexmesa_helm_build_") as tmp:
        raw_bin = pathlib.Path(tmp) / "helm_table_full.bin"
        convert_helm_ascii_to_full(src_dat, raw_bin)
        unpacked_sha = sha256_file(raw_bin)
        size_bytes = raw_bin.stat().st_size
        out_zst = out_dir / "helm_table_full.bin.zst"
        archive_sha = compress_zst(raw_bin, out_zst, level=level)
        manifest = out_dir / "helm_full.sha256.tsv"
        manifest.write_text(
            "# sha256\tsize_bytes\trelpath\n"
            f"{unpacked_sha}\t{size_bytes}\thelm_table_full.bin\n",
            encoding="utf-8",
        )
        return archive_sha, unpacked_sha, size_bytes


def convert_aesopus_hdf5_to_section(
    h5_path: pathlib.Path, out_path: pathlib.Path
) -> None:
    """Convert native float32 `AESOPUS_AGSS09.h5` into a binary section.

    Asserts that upstream `kap` datasets are natively `float32` (`N1-13`) and
    finite so zero implicit quantization occurs between MESA's HDF5 table and
    FlexMESA's `FMKAP001` archive.
    """
    import h5py

    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing AESOPUS HDF5 table: {h5_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f, out_path.open("wb") as out:
        for k in ["Zsun", "fCO_ref", "fC_ref", "fN_ref"]:
            val = float(f[k][()])
            if not np.isfinite(val):
                raise ValueError(f"{h5_path.name}: non-finite scalar {k}")
            out.write(struct.pack("<d", val))
        log_rs = np.asarray(f["logRs"][:], dtype="<f8")
        log_ts = np.asarray(f["logTs"][:], dtype="<f8")
        zs = np.asarray(f["Zs"][:], dtype="<f8")
        for arr_name, arr in (
            ("logRs", log_rs),
            ("logTs", log_ts),
            ("Zs", zs),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{h5_path.name}: non-finite {arr_name}")
        out.write(struct.pack("<III", len(log_rs), len(log_ts), len(zs)))
        out.write(log_rs.tobytes())
        out.write(log_ts.tobytes())
        out.write(zs.tobytes())
        for z in zs:
            g = f[f"{z:.6f}"]
            xs = np.asarray(g["Xs"][:], dtype="<f8")
            fcos = np.asarray(g["fCOs"][:], dtype="<f8")
            fcs = np.asarray(g["fCs"][:], dtype="<f8")
            fns = np.asarray(g["fNs"][:], dtype="<f8")
            out.write(
                struct.pack("<IIII", len(xs), len(fcos), len(fcs), len(fns))
            )
            out.write(xs.tobytes())
            out.write(fcos.tobytes())
            out.write(fcs.tobytes())
            out.write(fns.tobytes())
            out.write(b"\x01")
            kap_ds = g["kap"]
            if kap_ds.dtype != np.dtype("float32"):
                raise ValueError(
                    f"{h5_path.name}: expected float32 kap dtype, "
                    f"got {kap_ds.dtype}"
                )
            kap = np.asarray(kap_ds[:], dtype="<f4")
            if not np.all(np.isfinite(kap)):
                raise ValueError(
                    f"{h5_path.name}: non-finite kap values at Z={z:.6f}"
                )
            slices = np.moveaxis(kap, (0, 1), (-2, -1)).reshape(
                -1, len(log_ts) * len(log_rs)
            )
            for s in slices:
                out.write(b"\x01")
                out.write(s.tobytes())


def build_kap_full_bundle(
    mesa_dir: pathlib.Path,
    out_dir: pathlib.Path,
    level: int = 19,
) -> tuple[str, str, int]:
    """Build `kap_tables_full_v2.bin.zst` offline into `out_dir`."""
    from flexmesa import _rust

    kap_dir = mesa_dir / "data" / "kap_data"
    h5_path = kap_dir / "AESOPUS_AGSS09.h5"
    with tempfile.TemporaryDirectory(prefix="flexmesa_kap_build_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        aes_section = tmp_dir / "aesopus_section.bin"
        raw_bin = tmp_dir / "kap_tables_full.bin"
        convert_aesopus_hdf5_to_section(h5_path, aes_section)
        _rust.kap_build_full_tables_bin(
            str(kap_dir), str(aes_section), str(raw_bin)
        )
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


def _load_sha_manifest(tsv_path: pathlib.Path) -> dict[str, tuple[str, int]]:
    """Read a `# sha256/size/relpath` TSV into `{relpath: (sha, size)}`."""
    entries: dict[str, tuple[str, int]] = {}
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        sha, sz_s, rel = s.split("\t")
        entries[rel] = (sha.lower(), int(sz_s))
    return entries


def verify_unpacked_physics_bundles(
    mesa_dir: pathlib.Path,
    repo_root: pathlib.Path,
    *,
    stage_dir: pathlib.Path | None = None,
) -> dict[str, dict[str, object]]:
    """Verify converter outputs from `mesa_dir` against pinned SHA-256 records.

    Rebuilds `nets.bin`, `helm_table_full.bin`, `kap_tables_full.bin`, and all
    116 unpacked EOS `.raw` slices (`eos_opal_scvh`, `eos_cms`, `eos_free_eos`)
    in a temporary staging directory and checks every byte size and `SHA-256`
    digest against `src/flexmesa/data/bundles.toml`,
    `src/flexmesa/data/bundle_hashes/*.sha256.tsv`, and
    `crates/flexmesa-net/data/nets.bin`.

    Raises:
        ValueError: If any converted artifact differs from its pinned digest or
            size.
    """
    from flexmesa import _rust

    data_pkg = repo_root / "src" / "flexmesa" / "data"
    raw_toml = tomllib.loads(
        (data_pkg / "bundles.toml").read_text(encoding="utf-8")
    )
    by_name = {str(b["name"]): b for b in raw_toml["bundles"]}
    bh_dir = data_pkg / "bundle_hashes"

    own_tmp = stage_dir is None
    work_dir = (
        pathlib.Path(tempfile.mkdtemp(prefix="flexmesa_physics_verify_"))
        if stage_dir is None
        else stage_dir
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    mismatches: list[str] = []
    summary: dict[str, dict[str, object]] = {}
    try:
        # 1. Embedded nets.bin
        nets_out = work_dir / "nets.bin"
        nets_sha = build_nets_bin(mesa_dir, nets_out)
        nets_sz = nets_out.stat().st_size
        committed_nets = (
            repo_root / "crates" / "flexmesa-net" / "data" / "nets.bin"
        )
        pinned_nets_sha = sha256_file(committed_nets)
        pinned_nets_sz = committed_nets.stat().st_size
        if (nets_sha, nets_sz) != (pinned_nets_sha, pinned_nets_sz):
            mismatches.append(
                f"nets.bin: built ({nets_sha}, {nets_sz}) != "
                f"committed ({pinned_nets_sha}, {pinned_nets_sz})"
            )
        summary["nets"] = {
            "sha256": nets_sha,
            "size_bytes": nets_sz,
            "matched": (nets_sha, nets_sz) == (pinned_nets_sha, pinned_nets_sz),
        }

        # 2. helm_full (FMHELM01)
        helm_out = work_dir / "helm_table_full.bin"
        convert_helm_ascii_to_full(
            mesa_dir / "data" / "eosDT_data" / "helm_table.dat", helm_out
        )
        helm_sha = sha256_file(helm_out)
        helm_sz = helm_out.stat().st_size
        helm_pin = (
            str(by_name["helm_full"]["unpacked_sha256"]).lower(),
            int(by_name["helm_full"]["unpacked_size_bytes"]),
        )
        helm_tsv = _load_sha_manifest(bh_dir / "helm_full.sha256.tsv").get(
            "helm_table_full.bin"
        )
        if (helm_sha, helm_sz) != helm_pin or (helm_sha, helm_sz) != helm_tsv:
            mismatches.append(
                f"helm_full: built ({helm_sha}, {helm_sz}) != "
                f"pinned {helm_pin} / tsv {helm_tsv}"
            )
        summary["helm_full"] = {
            "sha256": helm_sha,
            "size_bytes": helm_sz,
            "matched": (helm_sha, helm_sz) == helm_pin
            and (helm_sha, helm_sz) == helm_tsv,
        }

        # 3. kap_full (FMKAP001)
        kap_dir = mesa_dir / "data" / "kap_data"
        aes_out = work_dir / "aesopus_section.bin"
        kap_out = work_dir / "kap_tables_full.bin"
        convert_aesopus_hdf5_to_section(kap_dir / "AESOPUS_AGSS09.h5", aes_out)
        _rust.kap_build_full_tables_bin(
            str(kap_dir), str(aes_out), str(kap_out)
        )
        kap_sha = sha256_file(kap_out)
        kap_sz = kap_out.stat().st_size
        kap_pin = (
            str(by_name["kap_full"]["unpacked_sha256"]).lower(),
            int(by_name["kap_full"]["unpacked_size_bytes"]),
        )
        kap_tsv = _load_sha_manifest(bh_dir / "kap_full.sha256.tsv").get(
            "kap_full/kap_tables_full.bin"
        )
        if (kap_sha, kap_sz) != kap_pin or (kap_sha, kap_sz) != kap_tsv:
            mismatches.append(
                f"kap_full: built ({kap_sha}, {kap_sz}) != "
                f"pinned {kap_pin} / tsv {kap_tsv}"
            )
        summary["kap_full"] = {
            "sha256": kap_sha,
            "size_bytes": kap_sz,
            "matched": (kap_sha, kap_sz) == kap_pin
            and (kap_sha, kap_sz) == kap_tsv,
        }

        # 4-6. Unpacked 4D EOS slices (OPAL/SCVH, CMS, FreeEOS)
        eos_specs = (
            (
                "eos_opal_scvh",
                [
                    mesa_dir / "data" / "eosDT_data" / f"{stem}.data"
                    for stem in sorted(OPAL_SCVH_STEMS)
                ],
                convert_dt_ascii_to_raw,
            ),
            (
                "eos_cms",
                sorted(
                    (mesa_dir / "data" / "eosCMS_data").glob("mesa-CMS_*.data")
                ),
                convert_cms_ascii_to_raw,
            ),
            (
                "eos_free_eos",
                sorted(
                    (mesa_dir / "data" / "eosFreeEOS_data").glob(
                        "mesa-FreeEOS_*.data"
                    )
                ),
                convert_dt_ascii_to_raw,
            ),
        )
        for bundle_name, src_files, conv_fn in eos_specs:
            tsv = _load_sha_manifest(bh_dir / f"{bundle_name}.sha256.tsv")
            bad_count = 0
            total_sz = 0
            seen_rels: set[str] = set()
            for src in src_files:
                rel = f"eos_4d_raw/{src.stem}.raw"
                seen_rels.add(rel)
                dst = work_dir / rel
                conv_fn(src, dst)
                got = (sha256_file(dst), dst.stat().st_size)
                total_sz += got[1]
                if got != tsv.get(rel):
                    bad_count += 1
                    mismatches.append(
                        f"{bundle_name} {rel}: built {got} != {tsv.get(rel)}"
                    )
            if seen_rels != set(tsv.keys()):
                bad_count += 1
                mismatches.append(
                    f"{bundle_name}: file set differs "
                    f"({len(seen_rels)} built vs {len(tsv)} pinned)"
                )
            summary[bundle_name] = {
                "files": len(src_files),
                "total_size_bytes": total_sz,
                "mismatches": bad_count,
                "matched": bad_count == 0,
            }
    finally:
        if own_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)

    if mismatches:
        raise ValueError(
            "Unpacked physics converter verification failed:\n"
            + "\n".join(mismatches[:20])
        )
    return summary


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
        "--helm-only",
        action="store_true",
        help="Rebuild nets.bin and helm_table_full.bin.zst.",
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
    parser.add_argument(
        "--verify-unpacked",
        action="store_true",
        help=(
            "Rebuild unpacked physics artifacts in a temp dir and verify "
            "every SHA-256 and byte size against bundles.toml and "
            "bundle_hashes/*.sha256.tsv."
        ),
    )
    args = parser.parse_args()

    if not args.mesa_dir or not (args.mesa_dir / "data").is_dir():
        raise SystemExit(
            "Provide a valid MESA 26.4.1 tree via --mesa-dir or $MESA_DIR."
        )

    if args.verify_unpacked:
        results = verify_unpacked_physics_bundles(args.mesa_dir, args.repo_root)
        for name, info in results.items():
            print(f"verified {name}: {info}")
        return 0

    nets_out = args.repo_root / "crates" / "flexmesa-net" / "data" / "nets.bin"
    nets_sha = build_nets_bin(args.mesa_dir, nets_out)
    print(f"nets.bin: sha256={nets_sha} bytes={nets_out.stat().st_size}")
    if args.nets_only:
        return 0

    if args.out_dir is None:
        raise SystemExit("--out-dir is required when building release bundles.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.helm_only:
        h_arch_sha, h_unp_sha, h_size = build_helm_full_bundle(
            args.mesa_dir, args.out_dir
        )
        print(
            f"helm_table_full.bin.zst: archive_sha256={h_arch_sha} "
            f"unpacked_sha256={h_unp_sha} unpacked_bytes={h_size}"
        )
        return 0

    if not args.kap_only:
        h_arch_sha, h_unp_sha, h_size = build_helm_full_bundle(
            args.mesa_dir, args.out_dir
        )
        print(
            f"helm_table_full.bin.zst: archive_sha256={h_arch_sha} "
            f"unpacked_sha256={h_unp_sha} unpacked_bytes={h_size}"
        )

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
        dt_dir = args.mesa_dir / "data" / "eosDT_data"
        dt_rels: list[str] = []
        for stem in sorted(OPAL_SCVH_STEMS):
            p = dt_dir / f"{stem}.data"
            if not p.is_file():
                raise FileNotFoundError(f"Missing OPAL/SCVH table: {p}")
            rel = f"eos_4d_raw/{stem}.raw"
            convert_dt_ascii_to_raw(p, stage / rel)
            dt_rels.append(rel)
        dt_sha = pack_deterministic_tar_zst(
            stage,
            dt_rels,
            args.out_dir / "eos_opal_scvh_raw_v1.tar.zst",
            args.out_dir / "eos_opal_scvh.sha256.tsv",
        )
        print(f"eos_opal_scvh_raw_v1.tar.zst: sha256={dt_sha}")

        fe_files = sorted(
            (args.mesa_dir / "data" / "eosFreeEOS_data").glob(
                "mesa-FreeEOS_*.data"
            )
        )
        if len(fe_files) != 89:
            raise ValueError(
                f"Expected 89 FreeEOS .data files, found {len(fe_files)}"
            )
        fe_rels: list[str] = []
        for p in fe_files:
            rel = f"eos_4d_raw/{p.stem}.raw"
            convert_dt_ascii_to_raw(p, stage / rel)
            fe_rels.append(rel)
        fe_sha = pack_deterministic_tar_zst(
            stage,
            fe_rels,
            args.out_dir / "eos_free_eos_raw_v1.tar.zst",
            args.out_dir / "eos_free_eos.sha256.tsv",
        )
        print(f"eos_free_eos_raw_v1.tar.zst: sha256={fe_sha}")

        cms_files = sorted(
            (args.mesa_dir / "data" / "eosCMS_data").glob("mesa-CMS_*.data")
        )
        if len(cms_files) != 11:
            raise ValueError(
                f"Expected 11 CMS .data files, found {len(cms_files)}"
            )
        cms_rels: list[str] = []
        for p in cms_files:
            rel = f"eos_4d_raw/{p.stem}.raw"
            convert_cms_ascii_to_raw(p, stage / rel)
            cms_rels.append(rel)
        cms_sha = pack_deterministic_tar_zst(
            stage,
            cms_rels,
            args.out_dir / "eos_cms_raw_v1.tar.zst",
            args.out_dir / "eos_cms.sha256.tsv",
        )
        print(f"eos_cms_raw_v1.tar.zst: sha256={cms_sha}")
        shutil.rmtree(stage, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
