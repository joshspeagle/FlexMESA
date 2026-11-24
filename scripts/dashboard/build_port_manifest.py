#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Seed or refresh ``benchmarks/dashboard/port_manifest.toml`` from MESA 26.4.1.

Counts exact upstream line counts and SHA-256 digests directly from
``mesa-26.04.1.zip`` (including bundled ``gyre-8.1.tar.gz``, ``ADIPLS``,
``STELLA``, and ``pgstar``/``pgbinary``), seeds statuses and file mappings from
``notes/audit/coverage-map.md`` and repository crate inspection, maps public
Fortran routines to verified ``FlexMESA`` symbols, downgrades any ``PORTED``
candidate missing a public-routine counterpart to ``partial``, and ensures zero
entries start in a ``verified`` tier (`#174`).
"""

from __future__ import annotations

import argparse
import collections.abc
import hashlib
import io
import pathlib
import re
import stat
import sys
import tarfile
import zipfile

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(REPO_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _seed_mappings  # noqa: E402

from flexmesa.testing import _port_parsers, port_manifest  # noqa: E402


def _md5_file(path: pathlib.Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_mesa_zip_sources(
    mesa_zip: pathlib.Path,
) -> dict[str, tuple[str, int, str, str]]:
    """Return ``{relpath: (module, lines, sha256, decoded_text)}`` from zip."""
    out: dict[str, tuple[str, int, str, str]] = {}
    with zipfile.ZipFile(mesa_zip, "r") as zf:
        top = zf.namelist()[0].split("/")[0] + "/"
        for zi in zf.infolist():
            if zi.is_dir() or not zi.filename.startswith(top):
                continue
            raw_mode = (zi.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(raw_mode):
                continue
            rel = zi.filename[len(top) :]
            mod = _port_parsers.classify_upstream_relpath(rel)
            if mod is None:
                continue
            data = zf.read(zi)
            out[rel] = (
                mod,
                len(data.splitlines()),
                hashlib.sha256(data).hexdigest(),
                data.decode("utf-8", errors="replace"),
            )

        gyre_tar = zf.read(f"{top}gyre/gyre-8.1.tar.gz")
        with tarfile.open(fileobj=io.BytesIO(gyre_tar), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                mpath = pathlib.PurePosixPath(member.name)
                sub_rel = "/".join(mpath.parts[1:])
                rel = f"gyre/gyre/{sub_rel}"
                mod = _port_parsers.classify_upstream_relpath(rel)
                if mod is None:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data = fobj.read()
                out[rel] = (
                    mod,
                    len(data.splitlines()),
                    hashlib.sha256(data).hexdigest(),
                    data.decode("utf-8", errors="replace"),
                )
    return out


def _resolve_coverage_map_files(upstream_path: str, raw_fm: str) -> list[str]:
    """Resolve shorthand file tokens from ``notes/audit/coverage-map.md``."""
    low = raw_fm.strip()
    if low in {"none", "n/a", ""} or "internal BVP solver" in raw_fm:
        return []
    if low.startswith("none "):
        return []
    top = upstream_path.split("/")[0]
    default_crate = {
        "star": "crates/flexmesa-star/src",
        "star_data": "crates/flexmesa-star/src",
        "binary": "crates/flexmesa-binary/src",
        "astero": "crates/flexmesa-astero/src",
    }.get(top, "crates/flexmesa-star/src")

    resolved: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_/*.-]+\.rs", raw_fm):
        if token.startswith("crates/"):
            pat = token
        elif "/" in token:
            pat = f"crates/{token}"
        else:
            pat = f"{default_crate}/{token}"
        for match in sorted(REPO_ROOT.glob(pat)):
            if match.is_file():
                resolved.add(match.relative_to(REPO_ROOT).as_posix())
    if "flexmesa-schema" in raw_fm:
        resolved.add("crates/flexmesa-schema/src/namelist.rs")
    return sorted(resolved)


def _parse_coverage_map() -> dict[
    str, tuple[port_manifest.PortStatus, list[str], str]
]:
    """Parse per-file classifications from ``notes/audit/coverage-map.md``."""
    cm_path = REPO_ROOT / "notes" / "audit" / "coverage-map.md"
    status_lookup = {
        "ABSENT": port_manifest.PortStatus.ABSENT,
        "STUB": port_manifest.PortStatus.STUB,
        "PARTIAL": port_manifest.PortStatus.PARTIAL,
        "PORTED": port_manifest.PortStatus.PORTED_UNVERIFIED,
    }
    parsed: dict[str, tuple[port_manifest.PortStatus, list[str], str]] = {}
    for line in cm_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 5:
            continue
        up = cols[0].strip("`")
        st_raw = cols[2].strip()
        if "/" not in up or st_raw not in status_lookup:
            continue
        st = status_lookup[st_raw]
        fm_files = _resolve_coverage_map_files(up, cols[3])
        note = f"seeded from coverage-map.md ({st_raw.lower()}): {cols[4]}"
        if st != port_manifest.PortStatus.ABSENT and not fm_files:
            st = port_manifest.PortStatus.ABSENT
        if st == port_manifest.PortStatus.ABSENT:
            fm_files = []
        parsed[up] = (st, fm_files, note)
    return parsed


def _make_archive_include_resolver(
    upstream_path: str,
    sources: dict[str, tuple[str, int, str, str]],
) -> collections.abc.Callable[[str], tuple[str, str] | None]:
    """Resolve ``include`` directives against ``sources`` from ``mesa_zip``."""
    up_posix = pathlib.PurePosixPath(upstream_path)
    parent = up_posix.parent.as_posix()
    mod_root = up_posix.parts[0]

    def _resolve(inc_name: str) -> tuple[str, str] | None:
        candidates = [
            f"{parent}/{inc_name}",
            f"{mod_root}/private/{inc_name}",
            f"{mod_root}/public/{inc_name}",
        ]
        for mod in _port_parsers.CORE_UPSTREAM_MODULES:
            candidates.append(f"{mod}/public/{inc_name}")
            candidates.append(f"{mod}/private/{inc_name}")
        for cand in candidates:
            if cand in sources:
                return (
                    sources[cand][3],
                    pathlib.PurePosixPath(cand).suffix,
                )
        return None

    return _resolve


def _build_routine_map(
    upstream_path: str,
    source_text: str,
    flexmesa_files: list[str],
    sym_cache: dict[str, set[str]],
    sources: dict[str, tuple[str, int, str, str]],
) -> tuple[dict[str, str], list[str]]:
    """Match public Fortran procedures to symbols in ``flexmesa_files``."""
    suffix = pathlib.PurePosixPath(upstream_path).suffix
    pubs = _port_parsers.extract_fortran_public_routines(
        source_text,
        suffix,
        include_resolver=_make_archive_include_resolver(upstream_path, sources),
    )
    hints = _seed_mappings.ROUTINE_COUNTERPART_HINTS.get(upstream_path, {})

    for rel_fm in flexmesa_files:
        if rel_fm not in sym_cache:
            sym_cache[rel_fm] = _port_parsers.extract_flexmesa_symbols(
                REPO_ROOT / rel_fm
            )

    rmap: dict[str, str] = {}
    missing: list[str] = []
    for proc in pubs:
        if proc in hints:
            target = hints[proc]
            rel_fm, syms_raw = target.split("::", 1)
            if rel_fm in flexmesa_files and all(
                s in sym_cache[rel_fm] for s in syms_raw.split("+")
            ):
                rmap[proc] = target
                continue
        found: str | None = None
        proc_low = proc.lower()
        for rel_fm in flexmesa_files:
            by_low = {s.lower(): s for s in sym_cache[rel_fm]}
            if proc_low in by_low:
                found = f"{rel_fm}::{by_low[proc_low]}"
                break
        if found is not None:
            rmap[proc] = found
        else:
            missing.append(proc)
    return rmap, missing


def build_manifest(mesa_zip: pathlib.Path, out_path: pathlib.Path) -> None:
    """Build and validate ``port_manifest.toml`` from ``mesa_zip``."""
    actual_md5 = _md5_file(mesa_zip)
    if actual_md5 != port_manifest.EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"{mesa_zip.name} MD5 {actual_md5} != "
            f"{port_manifest.EXPECTED_ARCHIVE_MD5}"
        )

    sources = _load_mesa_zip_sources(mesa_zip)
    cm_rows = _parse_coverage_map()
    sym_cache: dict[str, set[str]] = {}
    entries: list[port_manifest.PortManifestEntry] = []

    for relpath in sorted(sources):
        mod, nlines, sha, text = sources[relpath]
        if relpath in _seed_mappings.POST_AUDIT_PARTIALS:
            fm_files, note = _seed_mappings.POST_AUDIT_PARTIALS[relpath]
            status = port_manifest.PortStatus.PARTIAL
        elif relpath in cm_rows:
            status, fm_files, note = cm_rows[relpath]
        else:
            status, fm_files, note = (
                _seed_mappings.classify_non_coverage_map_file(relpath, mod)
            )

        routine_pairs: tuple[tuple[str, str], ...] = ()
        if status == port_manifest.PortStatus.PORTED_UNVERIFIED:
            hinted_files = {
                t.split("::", 1)[0]
                for t in _seed_mappings.ROUTINE_COUNTERPART_HINTS.get(
                    relpath, {}
                ).values()
            }
            merged_fm = sorted(set(fm_files) | hinted_files)
            rmap, missing = _build_routine_map(
                relpath, text, merged_fm, sym_cache, sources
            )
            if missing:
                status = port_manifest.PortStatus.PARTIAL
                note = (
                    f"{note}; downgraded to partial (unmapped public "
                    f"routines: {', '.join(missing[:5])})"
                )
            else:
                fm_files = merged_fm
                routine_pairs = tuple(sorted(rmap.items()))

        entries.append(
            port_manifest.PortManifestEntry(
                upstream_path=relpath,
                module=mod,
                upstream_lines=nlines,
                upstream_sha256=sha,
                status=status,
                flexmesa_files=tuple(fm_files),
                routine_map=routine_pairs,
                notes=note,
            )
        )

    manifest = port_manifest.PortManifest(
        schema_version=1,
        upstream_version=port_manifest.EXPECTED_UPSTREAM_VERSION,
        upstream_commit=port_manifest.EXPECTED_UPSTREAM_COMMIT,
        archive_md5=port_manifest.EXPECTED_ARCHIVE_MD5,
        entries=tuple(entries),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        port_manifest.dump_port_manifest(manifest), encoding="utf-8"
    )
    print(f"Wrote {len(entries)} entries to {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    """Parse CLI arguments and generate ``port_manifest.toml``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-zip",
        type=pathlib.Path,
        required=True,
        help="Path to mesa-26.04.1.zip (MD5 994fd28f38da92efc80e38ca11d8317b).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml",
        help="Output path for port_manifest.toml.",
    )
    args = parser.parse_args()
    build_manifest(args.mesa_zip.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
