#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build MESA 26.4.1 history and module-hex bundles and SHA-256 manifests.

Produces:
1. 6-line-header thinned MESA ``26.4.1`` histories for all finished cases (each
   preserving MESA's ``version_number "r26.4.1"``, ``compiler``, ``build``,
   ``MESA_SDK_version``, ``math_backend``, and ``date`` header), committing
   files ``<= --repo-max-kb`` into
   ``benchmarks/history_harness/reference_histories/<suite>/<case>.data``.
2. ``mesa-histories-v1.tar.zst`` containing all 102 thinned histories, all full
   histories, ``.mod`` outputs, ``34,393`` GYRE/ADIPLS oscillation modes and
   saved pulsation models, flagged incomplete/skipped case records, provenance,
   and ``SHA256SUMS.tsv``.
3. ``mesa-module-hex-v1.tar.zst`` containing all 25 module ``test/`` exact
   IEEE-754 hexadecimal / raw ``u64`` reference outputs, compiler FP flags, and
   ``SHA256SUMS.tsv``.
4. Per-file SHA-256 manifests under ``src/flexmesa/data/bundle_hashes/`` for
   every ``data-v1`` bundle (`#141`, `N1-10`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import tarfile
from typing import BinaryIO

CANONICAL_MTIME = 1776902400  # 2026-04-23T00:00:00Z

STAR_KEEP_COLUMNS: tuple[str, ...] = (
    "model_number",
    "num_zones",
    "star_age",
    "log_dt",
    "star_mass",
    "log_LH",
    "log_LHe",
    "log_L",
    "log_Teff",
    "log_R",
    "log_g",
    "log_cntr_T",
    "log_cntr_Rho",
    "log_center_T",
    "log_center_Rho",
    "center_h1",
    "center_he4",
    "center_c12",
    "he_core_mass",
    "co_core_mass",
    "mass_conv_core",
    "num_retries",
    "num_iters",
    "rsp_phase",
    "rsp_num_periods",
    "rsp_DeltaMag",
    "rsp_DeltaR",
    "rsp_period_in_days",
    "v_surf_km_s",
)

BINARY_KEEP_COLUMNS: tuple[str, ...] = (
    "model_number",
    "age",
    "period_days",
    "binary_separation",
    "v_orb_1",
    "v_orb_2",
    "eccentricity",
    "rl_1",
    "rl_2",
    "rl_relative_overflow_1",
    "rl_relative_overflow_2",
    "star_1_mass",
    "star_2_mass",
    "lg_mtransfer_rate",
    "lg_mstar_dot_1",
    "lg_mstar_dot_2",
    "lg_system_mdot_1",
    "lg_system_mdot_2",
    "xfer_fraction",
    "J_orb",
    "J_spin_1",
    "J_spin_2",
    "J_total",
    "jdot",
    "jdot_mb",
    "jdot_gr",
    "jdot_ml",
    "jdot_ls",
    "num_tries",
)

FORBIDDEN_SUBSTRINGS: tuple[bytes, ...] = (
    b"/home/",
    b"/mnt/",
    b"/tmp/",
    b"microsoft-standard",
    b"file://",
)

_PATH_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/" + r"home/[^/\s\"']+/mesa-26\.04\.1"), "$MESA_DIR"),
    (re.compile(r"/" + r"home/[^/\s\"']+/mesasdk-26\.6\.1"), "$MESASDK_ROOT"),
    (
        re.compile(r"/" + r"home/[^/\s\"']+/mesa-ground-truth-26\.4\.1"),
        "$MESA_GROUND_TRUTH",
    ),
    (re.compile(r"/" + r"home/[^/\s\"']+"), "$HOME"),
    (re.compile(r"/" + r"mnt/[^\s\"']+"), "$WORKSPACE"),
    (re.compile(r"/" + r"tmp/[^\s\"']+"), "$TMPDIR/work"),
    (
        re.compile(r"Linux [^\s)\"']*microsoft-standard-[A-Za-z0-9]+"),
        "Linux",
    ),
)


def _open_zstd_writer(path: pathlib.Path, level: int = 10) -> BinaryIO:
    try:
        from compression import zstd as std_zstd

        return std_zstd.open(path, "wb", level=level)  # type: ignore[return-value]
    except ImportError:
        import zstandard

        fh = path.open("wb")
        cctx = zstandard.ZstdCompressor(level=level, threads=4)
        return cctx.stream_writer(fh, closefd=True)  # type: ignore[return-value]


def sanitize_text_bytes(raw: bytes) -> bytes:
    """Scrub host-specific paths and kernel strings from UTF-8 text payloads."""
    txt = raw.decode("utf-8", errors="replace")
    for pat, repl in _PATH_SUBS:
        txt = pat.sub(repl, txt)
    out = txt.encode("utf-8")
    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in out:
            raise ValueError(f"Sanitization failed for {needle!r}")
    return out


def thin_history_preserving_mesa_header(
    src_path: pathlib.Path,
    suite: str,
) -> bytes:
    """Thin ``src_path`` while keeping MESA's exact 6-line header."""
    lines = src_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{src_path}: fewer than 7 lines ({len(lines)})")
    if '"r26.4.1"' not in lines[2]:
        raise ValueError(f"{src_path}: missing r26.4.1 in header: {lines[2]}")

    col_names = lines[5].split()
    keep_set = (
        frozenset(BINARY_KEEP_COLUMNS)
        if suite == "binary"
        else frozenset(STAR_KEEP_COLUMNS)
    )
    kept_indices = [i for i, name in enumerate(col_names) if name in keep_set]
    if not kept_indices:
        raise ValueError(f"{src_path}: zero kept columns from {col_names}")

    kept_names = [col_names[i] for i in kept_indices]
    out_lines: list[str] = [
        lines[0],
        lines[1],
        lines[2],
        lines[3],
        " ".join(f"{idx:24d}" for idx in range(1, len(kept_names) + 1)),
        " ".join(f"{name:>24s}" for name in kept_names),
    ]
    for raw in lines[6:]:
        cols = raw.split()
        if not cols:
            continue
        out_lines.append(" ".join(f"{cols[i]:>24s}" for i in kept_indices))
    return ("\n".join(out_lines) + "\n").encode("utf-8")


def _add_member(
    tf: tarfile.TarFile,
    relpath: str,
    payload: bytes,
    is_binary: bool = False,
) -> tuple[str, int]:
    if not is_binary:
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in payload:
                raise ValueError(
                    f"Forbidden substring {needle!r} in member {relpath}"
                )
    sha = hashlib.sha256(payload).hexdigest()
    info = tarfile.TarInfo(name=relpath)
    info.size = len(payload)
    info.mtime = CANONICAL_MTIME
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    tf.addfile(info, io.BytesIO(payload))
    return sha, len(payload)


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_histories_and_repo_refs(
    gt_dir: pathlib.Path,
    staging_dir: pathlib.Path,
    repo_root: pathlib.Path,
    repo_max_kb: int = 150,
) -> None:
    """Build thinned 6-line histories and ``mesa-histories-v1.tar.zst``."""
    summary_data = json.loads(
        (gt_dir / "summary_table.json").read_text(encoding="utf-8")
    )
    cases = summary_data["cases"]

    repo_ref_root = (
        repo_root / "benchmarks" / "history_harness" / "reference_histories"
    )
    repo_max_bytes = repo_max_kb * 1024
    committed_count = 0
    release_only_count = 0

    thinned_map: dict[str, bytes] = {}
    for row in cases:
        suite = str(row["suite"])
        case = str(row["case"])
        hfiles = list(row.get("history_files", []))
        if row["status"] != "ok" or not hfiles:
            continue
        final_rel = "binary_history.data" if suite == "binary" else hfiles[-1]
        src_hist = gt_dir / "runs" / suite / case / final_rel
        thinned_bytes = thin_history_preserving_mesa_header(src_hist, suite)
        rel_key = f"thinned/{suite}/{case}.data"
        thinned_map[rel_key] = thinned_bytes

        dst_repo = repo_ref_root / suite / f"{case}.data"
        if len(thinned_bytes) <= repo_max_bytes:
            dst_repo.parent.mkdir(parents=True, exist_ok=True)
            dst_repo.write_bytes(thinned_bytes)
            committed_count += 1
        else:
            if dst_repo.exists():
                dst_repo.unlink()
            release_only_count += 1

    out_arc = staging_dir / "mesa-histories-v1.tar.zst"
    manifest_out = (
        repo_root
        / "src"
        / "flexmesa"
        / "data"
        / "bundle_hashes"
        / "mesa_histories.sha256.tsv"
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    unpacked_bytes = 0
    with _open_zstd_writer(out_arc, level=10) as zout:
        with tarfile.open(fileobj=zout, mode="w|") as tf:
            for top_name in (
                "PROVENANCE.json",
                "PROVENANCE.md",
                "SUMMARY_TABLE.md",
                "summary_table.json",
                "summary_table.tsv",
            ):
                payload = sanitize_text_bytes((gt_dir / top_name).read_bytes())
                sha, sz = _add_member(tf, top_name, payload)
                rows.append(f"{sha}\t{sz}\t{top_name}")
                unpacked_bytes += sz

            for rel_key in sorted(thinned_map):
                payload = thinned_map[rel_key]
                sha, sz = _add_member(tf, rel_key, payload)
                rows.append(f"{sha}\t{sz}\t{rel_key}")
                unpacked_bytes += sz

            for row in sorted(cases, key=lambda r: (r["suite"], r["case"])):
                suite = str(row["suite"])
                case = str(row["case"])
                cdir = gt_dir / "runs" / suite / case
                if not cdir.is_dir():
                    continue
                keep_rels: set[str] = set()
                for meta_name in ("case_result.json", "out.txt", "mk.log"):
                    if (cdir / meta_name).is_file():
                        keep_rels.add(meta_name)
                for hrel in row.get("history_files", []):
                    if (cdir / hrel).is_file():
                        keep_rels.add(str(hrel))
                for mrel in row.get("produced_mods", []):
                    if (cdir / mrel).is_file():
                        keep_rels.add(str(mrel))
                if (cdir / "7B_2nd_crossing.data").is_file():
                    keep_rels.add("7B_2nd_crossing.data")
                for sub_dir_name in ("pulse_tap", "outputs"):
                    sdir = cdir / sub_dir_name
                    if sdir.is_dir():
                        for fp in sorted(sdir.rglob("*")):
                            if fp.is_file():
                                keep_rels.add(fp.relative_to(cdir).as_posix())

                for rel in sorted(keep_rels):
                    fp = cdir / rel
                    raw = fp.read_bytes()
                    is_bin = rel.endswith(".bin")
                    payload = raw if is_bin else sanitize_text_bytes(raw)
                    arc_rel = f"runs/{suite}/{case}/{rel}"
                    sha, sz = _add_member(
                        tf, arc_rel, payload, is_binary=is_bin
                    )
                    rows.append(f"{sha}\t{sz}\t{arc_rel}")
                    unpacked_bytes += sz

            sums_payload = (
                "# sha256\tsize_bytes\trelpath\n" + "\n".join(rows) + "\n"
            ).encode("utf-8")
            sha_s, sz_s = _add_member(tf, "SHA256SUMS.tsv", sums_payload)
            unpacked_bytes += sz_s

    manifest_out.write_text(
        "# sha256\tsize_bytes\trelpath\n"
        + "\n".join(rows)
        + f"\n{sha_s}\t{sz_s}\tSHA256SUMS.tsv\n",
        encoding="utf-8",
    )
    arc_sha = _sha256_file(out_arc)
    arc_sz = out_arc.stat().st_size
    print(f"Built {out_arc.name}:")
    print(
        f"  thinned total:       {len(thinned_map)} "
        f"(committed={committed_count}, release_only={release_only_count})"
    )
    print(f"  archive files:       {len(rows) + 1}")
    print(f"  archive_size_bytes:  {arc_sz}")
    print(f"  archive_sha256:      {arc_sha}")
    print(f"  unpacked_size_bytes: {unpacked_bytes}")


def build_module_hex_bundle(
    gt_dir: pathlib.Path,
    staging_dir: pathlib.Path,
    repo_root: pathlib.Path,
) -> None:
    """Build ``mesa-module-hex-v1.tar.zst`` and its SHA-256 manifest."""
    mod_root = gt_dir / "module_tests"
    out_arc = staging_dir / "mesa-module-hex-v1.tar.zst"
    manifest_out = (
        repo_root
        / "src"
        / "flexmesa"
        / "data"
        / "bundle_hashes"
        / "mesa_module_hex.sha256.tsv"
    )
    rows: list[str] = []
    unpacked_bytes = 0

    with _open_zstd_writer(out_arc, level=10) as zout:
        with tarfile.open(fileobj=zout, mode="w|") as tf:
            for fp in sorted(mod_root.rglob("*")):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(mod_root).as_posix()
                payload = sanitize_text_bytes(fp.read_bytes())
                sha, sz = _add_member(tf, rel, payload)
                rows.append(f"{sha}\t{sz}\t{rel}")
                unpacked_bytes += sz

            sums_payload = (
                "# sha256\tsize_bytes\trelpath\n" + "\n".join(rows) + "\n"
            ).encode("utf-8")
            sha_s, sz_s = _add_member(tf, "SHA256SUMS.tsv", sums_payload)
            unpacked_bytes += sz_s

    manifest_out.write_text(
        "# sha256\tsize_bytes\trelpath\n"
        + "\n".join(rows)
        + f"\n{sha_s}\t{sz_s}\tSHA256SUMS.tsv\n",
        encoding="utf-8",
    )
    arc_sha = _sha256_file(out_arc)
    arc_sz = out_arc.stat().st_size
    print(f"Built {out_arc.name}:")
    print(f"  archive files:       {len(rows) + 1}")
    print(f"  archive_size_bytes:  {arc_sz}")
    print(f"  archive_sha256:      {arc_sha}")
    print(f"  unpacked_size_bytes: {unpacked_bytes}")


def write_existing_physics_bundle_manifests(
    data_dir: pathlib.Path,
    repo_root: pathlib.Path,
) -> None:
    """Write per-file SHA-256 manifests for unpacked physics data bundles."""
    hashes_dir = repo_root / "src" / "flexmesa" / "data" / "bundle_hashes"
    hashes_dir.mkdir(parents=True, exist_ok=True)

    single_bundles = (
        ("helm_full", "helm_table_full.bin"),
        ("kap_full", "kap_full/kap_tables_full.bin"),
    )
    for bname, rel in single_bundles:
        fp = data_dir / rel
        sha = _sha256_file(fp)
        sz = fp.stat().st_size
        (hashes_dir / f"{bname}.sha256.tsv").write_text(
            f"# sha256\tsize_bytes\trelpath\n{sha}\t{sz}\t{rel}\n",
            encoding="utf-8",
        )

    eos_dir = data_dir / "eos_4d_raw"
    prefixes = (
        ("eos_opal_scvh", "mesa-eosDT_"),
        ("eos_cms", "mesa-CMS_"),
        ("eos_free_eos", "mesa-FreeEOS_"),
    )
    for bname, pref in prefixes:
        matched = sorted(eos_dir.glob(f"{pref}*.raw"))
        if not matched:
            raise FileNotFoundError(f"No {pref}*.raw files in {eos_dir}")
        b_rows = []
        for fp in matched:
            rel = f"eos_4d_raw/{fp.name}"
            b_rows.append(f"{_sha256_file(fp)}\t{fp.stat().st_size}\t{rel}")
        (hashes_dir / f"{bname}.sha256.tsv").write_text(
            "# sha256\tsize_bytes\trelpath\n" + "\n".join(b_rows) + "\n",
            encoding="utf-8",
        )
    print("Wrote physics per-file SHA-256 manifests in", hashes_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=pathlib.Path, required=True)
    parser.add_argument("--staging-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo-max-kb", type=int, default=150)
    args = parser.parse_args()

    gt = args.ground_truth_dir.resolve()
    stg = args.staging_dir.resolve()
    repo = args.repo_root.resolve()
    ddir = args.data_dir.resolve()

    build_histories_and_repo_refs(gt, stg, repo, repo_max_kb=args.repo_max_kb)
    build_module_hex_bundle(gt, stg, repo)
    write_existing_physics_bundle_manifests(ddir, repo)


if __name__ == "__main__":
    main()
