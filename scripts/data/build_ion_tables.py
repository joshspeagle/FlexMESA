#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the embedded `FMION001` ionization table from MESA `26.4.1` data.

Rebuilds `crates/flexmesa-ionization/data/ion_tables_full.bin` and its
provenance manifest `ion_tables_full.toml` from the 22 upstream ASCII tables
`ion_<Z>z<X>x<suffix>.data` read by `ionization/private/ion_tables_load.f90`.
The tables are taken from an extracted MESA `26.4.1` tree
(`$MESA_DIR/data/ionization_data`) or from the release tarball
`$MESA_DIR/ionization/test/ionization_data.tar.xz` (byte-identical copies).

Every upstream table value is printed with exactly three decimals (`%.3f`),
so the converter stores each value as the exact signed integer number of
thousandths. Rust decodes `f64::from(n) / 1000.0`, a correctly rounded
division of two exactly representable numbers, which equals the correctly
rounded decimal parse of the same `%.3f` token for every `|n| < 2**53`.

`FMION001` layout (little-endian):

- Header: `b"FMION001"`, then `u32` `num_tables`, `num_vals`, `num_logQs`,
  `num_logTs`.
- Body: for each table in `(Z, X)` order (`ion_Zs` outer, `ion_Xs` inner,
  `X + Z <= 1`), each of the 28 output channels, and each `logT` column
  `jT`, one flag byte and the `i32` value at `iQ = 0`. Flag `0` means the
  column is constant in `logQ`; flag `1` is followed by `num_logQs - 1`
  `i16` successive differences. (Decoders also accept flag `2`, a single
  `i16` constant difference, which this builder never writes.)

The `Z = 1` table follows the `ionization_Z1_suffix` control: MESA `26.4.1`
`star/defaults/star_job.defaults:2813` sets `''` (`ion_100z00x.data`), which
`star/private/init.f90` passes to `ionization_init`. The suffixes `_CO_0` and
`_CO_1` select the other two upstream `Z = 1` tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import lzma
import math
import os
import pathlib
import struct
import sys
import tarfile
import tempfile

import numpy as np
import numpy.typing as npt

FMION_MAGIC = b"FMION001"
NUM_ION_VALS = 28
ION_ZS: tuple[float, ...] = (0.00, 0.02, 0.04, 0.20, 1.00)
ION_XS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NUM_ION_XS_FOR_Z: tuple[int, ...] = (6, 5, 5, 5, 1)
Z1_SUFFIXES: tuple[str, ...] = ("", "_CO_0", "_CO_1")
VALUE_SCALE = 1000
# ionization/public/ionization_def.f90:91 (`min_version = 49`).
MIN_VERSION = 49
# ionization/private/ion_tables_load.f90:236 declares
# `real(dp), parameter :: tiny = 1e-6`; the default-real literal `1e-6` is
# rounded to single precision before widening to double.
TINY = float(np.float32(1.0e-6))
MESA_VERSION = "26.4.1"
TARBALL_RELPATH = "ionization/test/ionization_data.tar.xz"
DATA_RELPATH = "data/ionization_data"

_I16_MIN, _I16_MAX = -(2**15), 2**15 - 1
_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1


@dataclasses.dataclass(frozen=True)
class IonHeader:
    """Second header row of an upstream `ion_*.data` table.

    Attributes:
        version: Table format version (`ion_version`).
        x: Hydrogen mass fraction `X` of the table.
        z: Metal mass fraction `Z` of the table.
        num_log_ts: Number of `log10(T / K)` nodes.
        log_t_min: Minimum `log10(T / K)`.
        log_t_max: Maximum `log10(T / K)`.
        del_log_t: `log10(T / K)` spacing.
        num_log_qs: Number of `logQ = log10(rho) - 2 log10(T) + 12` nodes.
        log_q_min: Minimum `logQ`.
        log_q_max: Maximum `logQ`.
        del_log_q: `logQ` spacing.
    """

    version: int
    x: float
    z: float
    num_log_ts: int
    log_t_min: float
    log_t_max: float
    del_log_t: float
    num_log_qs: int
    log_q_min: float
    log_q_max: float
    del_log_q: float


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def table_filename(
    z: float, x: float, z1_suffix: str, prefix: str = "ion"
) -> str:
    """Return the upstream table name (`Get_ion_Table_Filenames`).

    Mirrors `setstr` in `ionization/private/ion_tables_load.f90:206-216`.

    Args:
        z: Metal mass fraction from `ion_Zs`.
        x: Hydrogen mass fraction from `ion_Xs`.
        z1_suffix: Suffix appended only to the `Z = 1` table.
        prefix: Table file prefix (`ionization_file_prefix`).

    Returns:
        File name such as `ion_02z70x.data`.
    """

    def setstr(v: float) -> str:
        if v > 0.99999:
            return "100"
        return f"{math.floor(100.0 * v + 0.5):02d}"

    z_str, x_str = setstr(z), setstr(x)
    suffix = z1_suffix if z_str == "100" else ""
    return f"{prefix}_{z_str}z{x_str}x{suffix}.data"


def table_compositions() -> list[tuple[float, float]]:
    """Return the 22 `(Z, X)` pairs in `Load_ion_Table` order."""
    return [
        (z, x)
        for z, n_xs in zip(ION_ZS, NUM_ION_XS_FOR_Z, strict=True)
        for x in ION_XS[:n_xs]
    ]


def parse_header(name: str, text: str) -> IonHeader:
    """Parse the 11-field second header row of an `ion_*.data` table.

    Args:
        name: File name used in error messages.
        text: Full table text.

    Returns:
        Parsed header.

    Raises:
        ValueError: If the header row is missing, short, or non-finite.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError(f"{name}: missing header row")
    toks = lines[1].split()
    if len(toks) < 11:
        raise ValueError(f"{name}: incomplete header row ({len(toks)} fields)")
    floats = [float(toks[i]) for i in (1, 2, 4, 5, 6, 8, 9, 10)]
    if not all(math.isfinite(v) for v in floats):
        raise ValueError(f"{name}: non-finite header value in {toks}")
    hdr = IonHeader(
        version=int(toks[0]),
        x=floats[0],
        z=floats[1],
        num_log_ts=int(toks[3]),
        log_t_min=floats[2],
        log_t_max=floats[3],
        del_log_t=floats[4],
        num_log_qs=int(toks[7]),
        log_q_min=floats[5],
        log_q_max=floats[6],
        del_log_q=floats[7],
    )
    if (
        hdr.num_log_ts < 2
        or hdr.num_log_qs < 2
        or hdr.del_log_t <= 0.0
        or hdr.del_log_q <= 0.0
    ):
        raise ValueError(
            f"{name}: invalid grid ({hdr.num_log_qs} x {hdr.num_log_ts}, "
            f"del_logQ={hdr.del_log_q}, del_logT={hdr.del_log_t})"
        )
    return hdr


def validate_headers(
    headers: list[tuple[str, IonHeader, float, float]],
) -> IonHeader:
    """Apply the upstream header checks to every table.

    Mirrors `ion_read_sizes` (`ionization/private/ion_tables_load.f90:54-69`,
    reference header from the first table and `version >= min_version`) and
    `Load1_ion_Table` (`ion_tables_load.f90:259-290`, every table against the
    reference grid and its expected `(X, Z)` within `tiny`).

    Args:
        headers: `(name, header, expected_z, expected_x)` in load order.

    Returns:
        Reference header (first table).

    Raises:
        ValueError: Naming the file and field of the first violation.
    """
    ref_name, ref, _, _ = headers[0]
    if ref.version < MIN_VERSION:
        raise ValueError(
            f"{ref_name}: header field version = {ref.version} is below "
            f"min_version = {MIN_VERSION}"
        )
    for name, hdr, z_exp, x_exp in headers:
        if ref.version > hdr.version:
            raise ValueError(
                f"{name}: header field version = {hdr.version} is older "
                f"than reference version = {ref.version} in {ref_name}"
            )
        for field in ("num_log_qs", "num_log_ts"):
            got, want = getattr(hdr, field), getattr(ref, field)
            if got != want:
                raise ValueError(
                    f"{name}: header field {field} = {got} differs from "
                    f"{want} in {ref_name}"
                )
        for field, want_f in (("x", x_exp), ("z", z_exp)):
            got_f = getattr(hdr, field)
            if abs(want_f - got_f) > TINY:
                raise ValueError(
                    f"{name}: header field {field.upper()} = {got_f} differs "
                    f"from expected table mass fraction {want_f}"
                )
        for field in (
            "log_t_min",
            "log_t_max",
            "del_log_t",
            "log_q_min",
            "log_q_max",
            "del_log_q",
        ):
            got_f, want_f = getattr(hdr, field), getattr(ref, field)
            if abs(want_f - got_f) > TINY:
                raise ValueError(
                    f"{name}: header field {field} = {got_f} differs from "
                    f"{want_f} in {ref_name}"
                )
    return ref


def _thousandths(token: str, where: str) -> int:
    """Return the exact integer thousandths of a `%.3f` table token."""
    if len(token) < 5 or token[-4] != "." or not token[-3:].isdigit():
        if not math.isfinite(float(token)):
            raise ValueError(f"{where}: non-finite value {token!r}")
        raise ValueError(f"{where}: value {token!r} is not printed as %.3f")
    return int(token[:-4] + token[-3:])


def parse_table_body(
    name: str, text: str, hdr: IonHeader
) -> npt.NDArray[np.int64]:
    """Parse table values as exact integer thousandths.

    Follows the line layout read by `Load1_ion_Table`
    (`ionization/private/ion_tables_load.f90:247-335`): three header lines,
    then for each `logQ` block four lines, `num_logTs` value rows, and two
    separator lines between blocks.

    Args:
        name: File name used in error messages.
        text: Full table text.
        hdr: Validated header of this table.

    Returns:
        Array of shape `(28, num_logQs, num_logTs)` of thousandths.

    Raises:
        ValueError: On truncation, short rows, or non-`%.3f`/non-finite
            values, naming the file and cell.
    """
    lines = text.splitlines()
    nq, nt = hdr.num_log_qs, hdr.num_log_ts
    out = np.empty((NUM_ION_VALS, nq, nt), dtype=np.int64)
    pos = 3
    for iq in range(nq):
        pos += 4
        for jt in range(nt):
            if pos >= len(lines):
                raise ValueError(f"{name}: unexpected EOF at iQ={iq}, jT={jt}")
            toks = lines[pos].split()
            pos += 1
            if len(toks) < NUM_ION_VALS + 1:
                raise ValueError(f"{name}: short row at iQ={iq}, jT={jt}")
            for v in range(NUM_ION_VALS):
                out[v, iq, jt] = _thousandths(
                    toks[v + 1], f"{name} var={v} iQ={iq} jT={jt}"
                )
        if iq + 1 < nq:
            pos += 2
    return out


def encode_fmion001(tables: list[npt.NDArray[np.int64]]) -> bytes:
    """Encode `(28, nq, nt)` thousandth tables into `FMION001` bytes.

    Args:
        tables: Tables in `Load_ion_Table` order, all of equal shape.

    Returns:
        Complete `FMION001` payload.

    Raises:
        ValueError: If shapes differ or a value or difference overflows its
            `i32` or `i16` field.
    """
    n_vals, nq, nt = tables[0].shape
    buf = bytearray(FMION_MAGIC)
    buf += struct.pack("<4I", len(tables), n_vals, nq, nt)
    for t, tbl in enumerate(tables):
        if tbl.shape != (n_vals, nq, nt):
            raise ValueError(f"table {t}: shape {tbl.shape} differs")
        if tbl.min() < _I32_MIN or tbl.max() > _I32_MAX:
            raise ValueError(f"table {t}: value overflows i32")
        diffs = np.diff(tbl, axis=1)
        if diffs.min() < _I16_MIN or diffs.max() > _I16_MAX:
            raise ValueError(f"table {t}: logQ difference overflows i16")
        for v in range(n_vals):
            for jt in range(nt):
                d = diffs[v, :, jt]
                if not d.any():
                    buf += struct.pack("<Bi", 0, int(tbl[v, 0, jt]))
                else:
                    buf += struct.pack("<Bi", 1, int(tbl[v, 0, jt]))
                    buf += d.astype("<i2").tobytes()
    return bytes(buf)


def decode_fmion001(
    data: bytes,
) -> list[npt.NDArray[np.int64]]:
    """Decode `FMION001` bytes into `(28, nq, nt)` thousandth tables.

    Args:
        data: Complete `FMION001` payload.

    Returns:
        Tables in stored order.

    Raises:
        ValueError: On bad magic, truncation, unknown flags, or trailing
            bytes.
    """
    if len(data) < 24 or data[:8] != FMION_MAGIC:
        raise ValueError("invalid FMION001 header")
    n_tab, n_vals, nq, nt = struct.unpack_from("<4I", data, 8)
    if n_tab == 0 or n_vals == 0 or nq == 0 or nt == 0:
        raise ValueError("invalid FMION001 dimensions")
    pos = 24
    tables: list[npt.NDArray[np.int64]] = []
    try:
        for _ in range(n_tab):
            arr = np.empty((n_vals, nq, nt), dtype=np.int64)
            for v in range(n_vals):
                for jt in range(nt):
                    flag, cur = struct.unpack_from("<Bi", data, pos)
                    pos += 5
                    if flag == 0:
                        arr[v, :, jt] = cur
                        continue
                    if flag == 2:
                        (d0,) = struct.unpack_from("<h", data, pos)
                        pos += 2
                        d = np.full(nq - 1, d0, dtype=np.int64)
                    elif flag == 1:
                        need = 2 * (nq - 1)
                        if pos + need > len(data):
                            raise ValueError(
                                f"truncated FMION001 payload at byte {pos}"
                            )
                        d = np.frombuffer(
                            data, dtype="<i2", count=nq - 1, offset=pos
                        ).astype(np.int64)
                        pos += need
                    else:
                        raise ValueError(f"unknown FMION001 flag {flag}")
                    arr[v, 0, jt] = cur
                    arr[v, 1:, jt] = cur + np.cumsum(d)
            tables.append(arr)
    except struct.error as exc:
        raise ValueError(f"truncated FMION001 payload: {exc}") from exc
    if pos != len(data):
        raise ValueError(
            f"FMION001 has {len(data) - pos} trailing bytes after the body"
        )
    return tables


def read_sources(
    source: pathlib.Path, z1_suffix: str, prefix: str = "ion"
) -> dict[str, bytes]:
    """Read the 22 table files from a directory or `.tar.xz` tarball.

    Args:
        source: `ionization_data` directory or `ionization_data.tar.xz`.
        z1_suffix: `Z = 1` suffix (`""`, `"_CO_0"`, or `"_CO_1"`).
        prefix: Table file prefix.

    Returns:
        Mapping from table file name to raw bytes, in load order.

    Raises:
        FileNotFoundError: If any table is missing.
    """
    names = [
        table_filename(z, x, z1_suffix, prefix) for z, x in table_compositions()
    ]
    out: dict[str, bytes] = {}
    if source.is_dir():
        for n in names:
            p = source / n
            if not p.is_file():
                raise FileNotFoundError(f"missing ionization table {p}")
            out[n] = p.read_bytes()
        return out
    with lzma.open(source) as xz, tarfile.open(fileobj=xz, mode="r|") as tf:
        wanted = set(names)
        found: dict[str, bytes] = {}
        for member in tf:
            base = pathlib.PurePosixPath(member.name).name
            if member.isfile() and base in wanted:
                fh = tf.extractfile(member)
                assert fh is not None
                found[base] = fh.read()
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(f"{source} lacks ionization tables {missing}")
    return {n: found[n] for n in names}


@dataclasses.dataclass(frozen=True)
class BuildResult:
    """Output of [`build`].

    Attributes:
        payload: `FMION001` bytes.
        manifest: Provenance manifest text (`ion_tables_full.toml`).
        tables: Decoded thousandth tables, shape `(28, nq, nt)` each.
    """

    payload: bytes
    manifest: str
    tables: list[npt.NDArray[np.int64]]


def build(
    sources: dict[str, bytes], z1_suffix: str, prefix: str = "ion"
) -> BuildResult:
    """Validate, parse, and encode the 22 upstream tables.

    Args:
        sources: Table name to raw bytes in load order (from
            [`read_sources`]).
        z1_suffix: `Z = 1` suffix used to select the sources.
        prefix: Table file prefix.

    Returns:
        Encoded payload, manifest text, and parsed tables.
    """
    texts = {n: b.decode("ascii") for n, b in sources.items()}
    comps = table_compositions()
    headers = [
        (n, parse_header(n, texts[n]), z, x)
        for n, (z, x) in zip(texts, comps, strict=True)
    ]
    ref = validate_headers(headers)
    tables = [parse_table_body(n, texts[n], h) for n, h, _, _ in headers]
    payload = encode_fmion001(tables)
    manifest = render_manifest(payload, ref, sources, z1_suffix, prefix)
    return BuildResult(payload=payload, manifest=manifest, tables=tables)


def _fmt_f64(v: float) -> str:
    """Format `v` as the shortest round-tripping TOML float."""
    s = repr(float(v))
    return s if ("." in s or "e" in s) else f"{s}.0"


def render_manifest(
    payload: bytes,
    ref: IonHeader,
    sources: dict[str, bytes],
    z1_suffix: str,
    prefix: str,
) -> str:
    """Render the deterministic `ion_tables_full.toml` provenance manifest.

    Top-level `key = value` lines are read by the Rust loader
    (`IonTableGrid::try_from_embedded_full`); keep them flat and ahead of the
    `[source_sha256]` table.
    """
    n_tab, n_vals, nq, nt = struct.unpack_from("<4I", payload, 8)
    lines = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# Provenance of the embedded FMION001 ionization table.",
        "# Generated by scripts/data/build_ion_tables.py (make "
        "data-ionization);",
        "# do not edit by hand.",
        'format = "FMION001"',
        f'sha256 = "{sha256_bytes(payload)}"',
        f"size_bytes = {len(payload)}",
        f'mesa_version = "{MESA_VERSION}"',
        f'source_dir = "{DATA_RELPATH}"',
        f'source_tarball = "{TARBALL_RELPATH}"',
        f'file_prefix = "{prefix}"',
        f'z1_suffix = "{z1_suffix}"',
        f"ion_version = {ref.version}",
        f"num_tables = {n_tab}",
        f"num_vals = {n_vals}",
        f"num_log_qs = {nq}",
        f"log_q_min = {_fmt_f64(ref.log_q_min)}",
        f"log_q_max = {_fmt_f64(ref.log_q_max)}",
        f"del_log_q = {_fmt_f64(ref.del_log_q)}",
        f"num_log_ts = {nt}",
        f"log_t_min = {_fmt_f64(ref.log_t_min)}",
        f"log_t_max = {_fmt_f64(ref.log_t_max)}",
        f"del_log_t = {_fmt_f64(ref.del_log_t)}",
        f"value_scale = {VALUE_SCALE}",
        "",
        "[source_sha256]",
    ]
    lines += [f'"{n}" = "{sha256_bytes(b)}"' for n, b in sources.items()]
    return "\n".join(lines) + "\n"


def _default_source(mesa_dir: pathlib.Path) -> pathlib.Path:
    data_dir = mesa_dir / DATA_RELPATH
    if data_dir.is_dir():
        return data_dir
    return mesa_dir / TARBALL_RELPATH


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build or check the embedded `FMION001` table."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    crate_data = repo_root / "crates" / "flexmesa-ionization" / "data"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("MESA_DIR", "")),
        help="Extracted MESA 26.4.1 root (uses data/ionization_data, "
        "falling back to ionization/test/ionization_data.tar.xz).",
    )
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=None,
        help="Explicit ionization_data directory or .tar.xz tarball.",
    )
    parser.add_argument(
        "--z1-suffix",
        choices=Z1_SUFFIXES,
        default="",
        help="Z = 1 table suffix (MESA star_job default: '').",
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=crate_data / "ion_tables_full.bin"
    )
    parser.add_argument(
        "--manifest-out",
        type=pathlib.Path,
        default=crate_data / "ion_tables_full.toml",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against --out/--manifest-out without writing.",
    )
    args = parser.parse_args(argv)

    source = args.source
    if source is None:
        if not str(args.mesa_dir) or str(args.mesa_dir) == ".":
            raise SystemExit("Provide --mesa-dir, $MESA_DIR, or --source.")
        source = _default_source(args.mesa_dir)
    result = build(read_sources(source, args.z1_suffix), args.z1_suffix)
    decoded = decode_fmion001(result.payload)
    if not all(
        np.array_equal(a, b)
        for a, b in zip(decoded, result.tables, strict=True)
    ):
        raise SystemExit("FMION001 round trip failed")
    sha = sha256_bytes(result.payload)
    print(f"ion_tables_full.bin: sha256={sha} bytes={len(result.payload)}")
    if args.check:
        ok = (
            args.out.is_file()
            and args.out.read_bytes() == result.payload
            and args.manifest_out.is_file()
            and args.manifest_out.read_text(encoding="utf-8") == result.manifest
        )
        print("check:", "identical" if ok else "MISMATCH")
        return 0 if ok else 1
    for path, data in (
        (args.out, result.payload),
        (args.manifest_out, result.manifest.encode("utf-8")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
