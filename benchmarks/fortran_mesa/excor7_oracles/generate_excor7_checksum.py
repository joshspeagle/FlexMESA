#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate SHA-256 and IEEE-754 hex oracle from MESA EXCOR7.data."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import struct

EXCOR7_N_RS = 351
EXCOR7_N_GAME = 601
EXCOR7_NVARS = 7
EXCOR7_VAR_NAMES = ("FXC", "UXC", "PXC", "CVXC", "SXC", "PDTXC", "PDRXC")

SAMPLE_NODES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 300),
    (0, 600),
    (100, 150),
    (175, 300),
    (200, 350),
    (250, 100),
    (300, 500),
    (349, 599),
    (350, 0),
    (350, 300),
    (350, 600),
)


def _parse_fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def parse_excor7_data(
    excor7_path: pathlib.Path,
) -> tuple[str, str, dict[tuple[int, int], tuple[float, float, list[float]]]]:
    """Parse shipped MESA `EXCOR7.data` and hash its IEEE-754 stream."""
    raw_bytes = excor7_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    lines = raw_bytes.decode("ascii").splitlines()
    idx = 0

    # Header line 1, parameter line 2, blank line 3 (`pc_support.f90:636-668`).
    idx += 1
    params = [_parse_fortran_float(tok) for tok in lines[idx].split()]
    idx += 1
    n_rs = int(params[0])
    n_game = int(params[4])
    if n_rs != EXCOR7_N_RS or n_game != EXCOR7_N_GAME:
        raise ValueError(f"unexpected EXCOR7 grid shape: {n_rs}x{n_game}")
    idx += 1  # skip blank line before first block

    sample_set = set(SAMPLE_NODES)
    sampled: dict[tuple[int, int], tuple[float, float, list[float]]] = {}
    stream_hasher = hashlib.sha256()

    for i_rs in range(n_rs):
        idx += 1  # "logRS" label
        log_rs = _parse_fortran_float(lines[idx].strip())
        idx += 1
        idx += 2  # blank line + column names line

        for j_game in range(n_game):
            parts = [_parse_fortran_float(tok) for tok in lines[idx].split()]
            idx += 1
            if len(parts) < 1 + EXCOR7_NVARS:
                raise ValueError(
                    f"malformed EXCOR7 row at ({i_rs}, {j_game}): {len(parts)}"
                )
            log_game = parts[0]
            vals = parts[1 : 1 + EXCOR7_NVARS]
            stream_hasher.update(struct.pack("<7d", *vals))
            if (i_rs, j_game) in sample_set:
                sampled[(i_rs, j_game)] = (log_rs, log_game, vals)

        if i_rs + 1 < n_rs:
            idx += 2  # two blank separator lines between logRS blocks

    return file_sha256, stream_hasher.hexdigest(), sampled


def _read_mesa_version(mesa_dir: pathlib.Path) -> str:
    ver_file = mesa_dir / "data" / "version_number"
    if not ver_file.is_file():
        return "r26.4.1"
    raw = ver_file.read_text(encoding="utf-8").strip().split()[0]
    return raw if raw.startswith("r") else f"r{raw}"


def main() -> int:
    """Parse MESA `EXCOR7.data` and write the checksum + node TSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=None,
        help="Path to MESA installation (defaults to $MESA_DIR).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("excor7_26_4_1_checksum.tsv"),
        help="Output TSV path.",
    )
    args = parser.parse_args()
    mesa_dir = args.mesa_dir or (
        pathlib.Path(os.environ["MESA_DIR"])
        if "MESA_DIR" in os.environ
        else None
    )
    if mesa_dir is None:
        raise SystemExit("pass --mesa-dir or set $MESA_DIR")

    excor7_path = mesa_dir / "data" / "eosPC_support_data" / "EXCOR7.data"
    file_sha, stream_sha, sampled = parse_excor7_data(excor7_path)
    mesa_ver = _read_mesa_version(mesa_dir)

    lines = [
        "# FlexMESA MESA 26.4.1 EXCOR7.data checksum and sample nodes (#161)",
        f"# mesa_version: {mesa_ver}",
        "# compiler: gfortran 15.2.0",
        "# mesa_sdk_version: x86_64-linux-26.6.1",
        "# source: data/eosPC_support_data/EXCOR7.data",
        f"# excor7_data_sha256: {file_sha}",
        f"# ieee754_le_bytes_sha256: {stream_sha}",
        f"# value_count: {EXCOR7_N_RS * EXCOR7_N_GAME * EXCOR7_NVARS}",
        "i_rs\tj_game\tlogRS\tlogGAME\tvar\tvalue\tvalue_hex",
    ]
    for i_rs, j_game in SAMPLE_NODES:
        log_rs, log_game, vals = sampled[(i_rs, j_game)]
        for name, val in zip(EXCOR7_VAR_NAMES, vals, strict=True):
            hex_u64 = struct.pack(">d", val).hex().upper()
            lines.append(
                f"{i_rs}\t{j_game}\t{log_rs:.6f}\t{log_game:.6f}\t"
                f"{name}\t{val:.16e}\t{hex_u64}"
            )
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: ieee754_le_bytes_sha256={stream_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
