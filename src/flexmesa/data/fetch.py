# SPDX-License-Identifier: LGPL-3.0-or-later
"""Download, SHA-256-verify, and unpack ``data-v1`` physics and test bundles.

Can run either through the installed ``flexmesa data`` CLI or directly via
``python3 src/flexmesa/data/fetch.py`` in clean Rust-only CI jobs prior to
building ``flexmesa._rust``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from collections.abc import Iterator
from typing import IO


@dataclasses.dataclass(frozen=True, slots=True)
class ReleaseBundle:
    """Metadata and checksum record for one ``data-v1`` release archive."""

    name: str
    archive: str
    kind: str
    target_relpath: str
    archive_sha256: str
    archive_size_bytes: int
    unpacked_sha256: str
    unpacked_size_bytes: int
    test_only: bool
    description: str
    sentinel_relpaths: tuple[str, ...] = ()


def resolve_data_dir(
    explicit: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve the canonical FlexMESA physics table directory.

    Precedence:
    1. ``explicit`` argument if provided.
    2. ``$FLEXMESA_DATA_DIR`` if set and non-empty.
    3. ``$XDG_CACHE_HOME/flexmesa/data`` if ``XDG_CACHE_HOME`` is set.
    4. ``~/.cache/flexmesa/data``.

    Args:
        explicit: Optional directory override.

    Returns:
        Resolved ``pathlib.Path`` to the physics data directory.
    """
    if explicit is not None:
        return pathlib.Path(explicit).expanduser().resolve()
    if env_dir := os.environ.get("FLEXMESA_DATA_DIR", "").strip():
        return pathlib.Path(env_dir).expanduser().resolve()
    if xdg := os.environ.get("XDG_CACHE_HOME", "").strip():
        return (pathlib.Path(xdg).expanduser() / "flexmesa" / "data").resolve()
    return (pathlib.Path.home() / ".cache" / "flexmesa" / "data").resolve()


def resolve_mesa_test_inputs_dir(
    explicit: str | pathlib.Path | None = None,
    data_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve the extraction root for ``mesa-test-inputs-v1.tar.zst``.

    Args:
        explicit: Optional explicit target directory.
        data_dir: Resolved physics data directory (defaults to sibling
            ``mesa_test_inputs`` next to ``data_dir``).

    Returns:
        Resolved ``pathlib.Path`` for upstream MESA test inputs.
    """
    if explicit is not None:
        return pathlib.Path(explicit).expanduser().resolve()
    if mesa_dir := os.environ.get("MESA_DIR", "").strip():
        return pathlib.Path(mesa_dir).expanduser().resolve()
    base = data_dir if data_dir is not None else resolve_data_dir()
    return (base.parent / "mesa_test_inputs").resolve()


def load_bundles_manifest(
    manifest_path: pathlib.Path | None = None,
) -> tuple[str, str, tuple[ReleaseBundle, ...]]:
    """Parse ``bundles.toml`` into ``(repo, tag, bundles)``.

    Args:
        manifest_path: Optional override path to ``bundles.toml``.

    Returns:
        Tuple ``(repo, tag, bundles)``.
    """
    path = (
        manifest_path
        if manifest_path is not None
        else pathlib.Path(__file__).with_name("bundles.toml")
    )
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    rel = raw["release"]
    repo = str(rel["repo"])
    tag = str(rel["tag"])
    items: list[ReleaseBundle] = []
    for entry in raw["bundles"]:
        sentinels = tuple(str(p) for p in entry.get("sentinel_relpaths", ()))
        items.append(
            ReleaseBundle(
                name=str(entry["name"]),
                archive=str(entry["archive"]),
                kind=str(entry["kind"]),
                target_relpath=str(entry["target_relpath"]),
                archive_sha256=str(entry["archive_sha256"]).lower(),
                archive_size_bytes=int(entry["archive_size_bytes"]),
                unpacked_sha256=str(entry.get("unpacked_sha256", "")).lower(),
                unpacked_size_bytes=int(entry["unpacked_size_bytes"]),
                test_only=bool(entry.get("test_only", False)),
                description=str(entry.get("description", "")),
                sentinel_relpaths=sentinels,
            )
        )
    return repo, tag, tuple(items)


def _sha256_file(path: pathlib.Path) -> str:
    """Compute the lowercase 64-character SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _open_zstd_reader(path: pathlib.Path) -> Iterator[IO[bytes]]:
    """Open a streaming binary reader for a ``.zst`` archive."""
    try:
        from compression import zstd as std_zstd

        with std_zstd.open(path, "rb") as reader:
            yield reader
        return
    except ImportError:
        pass

    try:
        import zstandard

        with path.open("rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                yield reader
        return
    except ImportError:
        pass

    zstd_bin = shutil.which("zstd")
    if zstd_bin is None:
        msg = (
            "Neither stdlib 'compression.zstd' (Python >=3.14), 'zstandard', "
            "nor the 'zstd' CLI is available to decompress .zst bundles."
        )
        raise RuntimeError(msg)
    proc = subprocess.Popen(
        [zstd_bin, "-dc", str(path)],
        stdout=subprocess.PIPE,
    )
    completed = False
    try:
        assert proc.stdout is not None
        yield proc.stdout
        proc.stdout.close()
        completed = True
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd decompression exited with status {rc}")
    finally:
        if not completed:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.wait()


def _stamp_path(root: pathlib.Path, bundle: ReleaseBundle) -> pathlib.Path:
    short = bundle.archive_sha256[:16]
    return root / ".stamps" / f"{bundle.name}.{short}.ready"


def is_bundle_ready(root: pathlib.Path, bundle: ReleaseBundle) -> bool:
    """Return ``True`` when ``bundle`` is already unpacked and stamped."""
    if not _stamp_path(root, bundle).is_file():
        return False
    if not all((root / rel).exists() for rel in bundle.sentinel_relpaths):
        return False
    target = root / bundle.target_relpath
    if bundle.kind == "single_zst":
        return (
            target.is_file()
            and target.stat().st_size == bundle.unpacked_size_bytes
        )
    return target.is_dir()


def verify_bundles(
    *,
    dest: str | pathlib.Path | None = None,
    include_test_inputs: bool = False,
    mesa_test_dest: str | pathlib.Path | None = None,
    quiet: bool = False,
) -> bool:
    """Verify unpacked ``data-v1`` bundles, sizes, sentinels, and SHA-256."""
    data_root = resolve_data_dir(dest)
    test_root = resolve_mesa_test_inputs_dir(mesa_test_dest, data_dir=data_root)
    _, _, bundles = load_bundles_manifest()
    all_ok = True
    for b in bundles:
        if b.test_only and not include_test_inputs:
            continue
        root = test_root if b.test_only else data_root
        ok = is_bundle_ready(root, b)
        if ok and b.kind == "single_zst" and b.unpacked_sha256:
            ok = _sha256_file(root / b.target_relpath) == b.unpacked_sha256
        if not quiet:
            status = "VERIFIED" if ok else "FAILED"
            print(f"[flexmesa-data] {b.name:18s} {status} ({root})", flush=True)
        all_ok = all_ok and ok
    return all_ok


def _download_archive(url: str, dst: pathlib.Path, expected_size: int) -> None:
    """Stream ``url`` to ``dst`` with optional ``GITHUB_TOKEN`` header."""
    headers = {"User-Agent": "FlexMESA-Data-Fetch/1.0"}
    token = (
        os.environ.get("GITHUB_TOKEN", "").strip()
        or os.environ.get("GH_TOKEN", "").strip()
    )
    if token and "github.com/" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".part.{os.getpid()}")
    try:
        with (
            urllib.request.urlopen(req, timeout=120) as resp,
            tmp.open("wb") as out,
        ):
            while chunk := resp.read(4 * 1024 * 1024):
                out.write(chunk)
        actual_size = tmp.stat().st_size
        if expected_size > 0 and actual_size != expected_size:
            raise ValueError(
                f"Downloaded size mismatch for {dst.name}: "
                f"expected {expected_size} B, got {actual_size} B"
            )
        tmp.replace(dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _install_archive(
    archive_path: pathlib.Path,
    dest_root: pathlib.Path,
    bundle: ReleaseBundle,
) -> None:
    """Verify ``archive_path`` SHA-256 and unpack it into ``dest_root``."""
    actual_arc_sha = _sha256_file(archive_path)
    if actual_arc_sha != bundle.archive_sha256:
        raise ValueError(
            f"Archive SHA-256 mismatch for {bundle.archive}: "
            f"expected {bundle.archive_sha256}, got {actual_arc_sha}"
        )

    dest_root.mkdir(parents=True, exist_ok=True)
    if bundle.kind == "single_zst":
        out_path = dest_root / bundle.target_relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out_path.with_suffix(f".tmp.{os.getpid()}")
        h_raw = hashlib.sha256()
        raw_sz = 0
        try:
            with (
                _open_zstd_reader(archive_path) as zin,
                tmp_out.open("wb") as fout,
            ):
                while chunk := zin.read(8 * 1024 * 1024):
                    h_raw.update(chunk)
                    raw_sz += len(chunk)
                    fout.write(chunk)
            if raw_sz != bundle.unpacked_size_bytes:
                raise ValueError(
                    f"Unpacked size mismatch for {out_path.name}: "
                    f"expected {bundle.unpacked_size_bytes}, got {raw_sz}"
                )
            if (
                bundle.unpacked_sha256
                and h_raw.hexdigest() != bundle.unpacked_sha256
            ):
                raise ValueError(
                    f"Unpacked SHA-256 mismatch for {out_path.name}: "
                    f"expected {bundle.unpacked_sha256}, "
                    f"got {h_raw.hexdigest()}"
                )
            tmp_out.replace(out_path)
        except BaseException:
            if tmp_out.exists():
                tmp_out.unlink()
            raise
    elif bundle.kind == "tar_zst":
        extracted_bytes = 0
        with _open_zstd_reader(archive_path) as zin:
            with tarfile.open(fileobj=zin, mode="r|") as tf:
                for member in tf:
                    if member.isfile():
                        extracted_bytes += member.size
                    tf.extract(member, path=dest_root, filter="data")
        if (
            bundle.unpacked_size_bytes > 0
            and extracted_bytes != bundle.unpacked_size_bytes
        ):
            raise ValueError(
                f"Unpacked byte sum mismatch for {bundle.archive}: "
                f"expected {bundle.unpacked_size_bytes}, got {extracted_bytes}"
            )
    else:
        raise ValueError(f"Unsupported bundle kind: {bundle.kind!r}")

    stamp = _stamp_path(dest_root, bundle)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"{bundle.archive_sha256}\n", encoding="utf-8")


def fetch_bundles(
    *,
    dest: str | pathlib.Path | None = None,
    include_test_inputs: bool = False,
    mesa_test_dest: str | pathlib.Path | None = None,
    source_dir: str | pathlib.Path | None = None,
    base_url: str | None = None,
    quiet: bool = False,
) -> pathlib.Path:
    """Ensure all required ``data-v1`` bundles are verified and unpacked.

    Args:
        dest: Physics table destination directory (defaults to
            ``resolve_data_dir()``).
        include_test_inputs: Also fetch ``mesa-test-inputs-v1.tar.zst``.
        mesa_test_dest: Destination directory for ``mesa_test_inputs``.
        source_dir: Optional local directory containing pre-downloaded ``.zst``
            archives (for offline/staging installation).
        base_url: Optional override for the GitHub Release download base URL.
        quiet: Suppress progress lines when ``True``.

    Returns:
        Resolved physics data directory ``pathlib.Path``.
    """
    data_root = resolve_data_dir(dest)
    test_root = resolve_mesa_test_inputs_dir(mesa_test_dest, data_dir=data_root)
    repo, tag, bundles = load_bundles_manifest()
    release_base = (
        base_url.rstrip("/")
        if base_url
        else os.environ.get(
            "FLEXMESA_DATA_URL",
            f"https://github.com/{repo}/releases/download/{tag}",
        ).rstrip("/")
    )
    local_src = (
        pathlib.Path(source_dir).expanduser().resolve()
        if source_dir is not None
        else None
    )

    for bundle in bundles:
        if bundle.test_only and not include_test_inputs:
            continue
        target_root = test_root if bundle.test_only else data_root
        if is_bundle_ready(target_root, bundle):
            if not quiet:
                print(
                    f"[flexmesa-data] {bundle.name}: ready ({target_root})",
                    flush=True,
                )
            continue

        if local_src is not None and (local_src / bundle.archive).is_file():
            arc_path = local_src / bundle.archive
            remove_after = False
            if not quiet:
                print(
                    f"[flexmesa-data] {bundle.name}: unpacking from "
                    f"local archive {arc_path.name} ...",
                    flush=True,
                )
        else:
            url = f"{release_base}/{bundle.archive}"
            dl_dir = target_root / ".downloads"
            arc_path = dl_dir / bundle.archive
            remove_after = True
            mb = bundle.archive_size_bytes / 1.0e6
            if not quiet:
                print(
                    f"[flexmesa-data] {bundle.name}: downloading "
                    f"{bundle.archive} ({mb:.1f} MB) ...",
                    flush=True,
                )
            _download_archive(url, arc_path, bundle.archive_size_bytes)

        _install_archive(arc_path, target_root, bundle)
        if remove_after and arc_path.exists():
            arc_path.unlink()
        if not quiet:
            print(
                f"[flexmesa-data] {bundle.name}: verified & unpacked",
                flush=True,
            )

    return data_root


def _add_dir_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dest",
        "--data-dir",
        dest="dest",
        type=pathlib.Path,
        default=None,
        help=(
            "Physics data directory "
            "($FLEXMESA_DATA_DIR or ~/.cache/flexmesa/data)."
        ),
    )
    p.add_argument(
        "--mesa-test-dest",
        type=pathlib.Path,
        default=None,
        help="Upstream MESA test inputs directory.",
    )


def build_parser(
    prog: str = "flexmesa data",
) -> argparse.ArgumentParser:
    """Construct the ``argparse.ArgumentParser`` for ``flexmesa data``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Manage SHA-256-verified FlexMESA physics & test bundles.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    p_fetch = sub.add_parser(
        "fetch",
        help="Download, SHA-256-verify, and unpack data-v1 bundles.",
    )
    _add_dir_args(p_fetch)
    p_fetch.add_argument(
        "--include-test-inputs",
        action="store_true",
        help="Also fetch and unpack mesa-test-inputs-v1.tar.zst.",
    )
    p_fetch.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=None,
        help="Optional local directory containing .zst release archives.",
    )
    p_fetch.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )

    p_status = sub.add_parser(
        "status",
        help="Report installation status of all data-v1 bundles.",
    )
    _add_dir_args(p_status)

    p_verify = sub.add_parser(
        "verify",
        help="Verify stamps, sentinel files, byte sizes, and unpacked SHA-256.",
    )
    _add_dir_args(p_verify)
    p_verify.add_argument(
        "--include-test-inputs",
        action="store_true",
        help="Also verify mesa-test-inputs-v1.",
    )
    p_verify.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-bundle verification lines.",
    )

    p_path = sub.add_parser(
        "path",
        help="Print the resolved physics data (or --test-inputs) directory.",
    )
    _add_dir_args(p_path)
    p_path.add_argument(
        "--test-inputs",
        action="store_true",
        help="Print the resolved MESA test inputs directory instead.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m flexmesa.data``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand == "fetch":
        fetch_bundles(
            dest=args.dest,
            include_test_inputs=bool(args.include_test_inputs),
            mesa_test_dest=args.mesa_test_dest,
            source_dir=args.source_dir,
            quiet=bool(args.quiet),
        )
        return 0
    if args.subcommand == "status":
        data_root = resolve_data_dir(args.dest)
        test_root = resolve_mesa_test_inputs_dir(
            args.mesa_test_dest, data_dir=data_root
        )
        _, tag, bundles = load_bundles_manifest()
        print(f"Release tag:     {tag}")
        print(f"Data directory:  {data_root}")
        print(f"Test inputs dir: {test_root}")
        for b in bundles:
            root = test_root if b.test_only else data_root
            state = "READY" if is_bundle_ready(root, b) else "MISSING"
            kind_tag = " [test-only]" if b.test_only else ""
            print(f"  - {b.name:18s} {state:8s} ({b.archive}){kind_tag}")
        return 0
    if args.subcommand == "verify":
        ok = verify_bundles(
            dest=args.dest,
            include_test_inputs=bool(args.include_test_inputs),
            mesa_test_dest=args.mesa_test_dest,
            quiet=bool(args.quiet),
        )
        return 0 if ok else 1
    if args.subcommand == "path":
        data_root = resolve_data_dir(args.dest)
        if args.test_inputs:
            print(
                resolve_mesa_test_inputs_dir(
                    args.mesa_test_dest, data_dir=data_root
                )
            )
        else:
            print(data_root)
        return 0
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
