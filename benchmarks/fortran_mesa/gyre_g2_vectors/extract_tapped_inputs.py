#!/usr/bin/env python3
"""Extract the real LAPACK inputs GYRE passed in the G0 oracle calls (#170).

Reads the `*.taps.tsv` files of `benchmarks/fortran_mesa/gyre_oracles/data/`
(package G0) and writes one line per real `XGETRF`, `XGESVX` or `DGESVD`
input, in the format `gen_gyre_g2_vectors` reads:

    kind id m n nrhs bits...

where each `bits` value is the IEEE-754 bit pattern as a signed decimal
64-bit integer (list-directed `INTEGER(INT64)` input). Complex (`reim`)
records are skipped: they belong to the complex packages.
"""

from __future__ import annotations

import argparse
import pathlib
import re

SHAPE = re.compile(r"\[(\d+)x(\d+),colmajor\]$")


def signed(h: str) -> int:
    v = int(h, 16)
    return v - (1 << 64) if v >= 1 << 63 else v


def record(kind: str, ident: int, dims: str, vals: list[str]) -> str:
    return f"{kind} {ident} {dims} " + " ".join(str(signed(v)) for v in vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("taps_dir", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    args = ap.parse_args()
    lines: list[str] = []
    ident = 0
    for path in sorted(args.taps_dir.glob("*.taps.tsv")):
        pending_a: tuple[int, int, list[str]] | None = None
        for raw in path.read_text().splitlines():
            if raw.startswith("#"):
                continue
            f = raw.split("\t")
            if f[0] != "LAPACK_IN":
                continue
            name, vals = f[1], f[3].split()
            m = SHAPE.search(name)
            if m is None:  # complex record (reim) or vector
                continue
            rows, cols = int(m.group(1)), int(m.group(2))
            if name.startswith(("XGETRF_FACTOR:A", "XGETRF_DET:A")):
                ident += 1
                lines.append(record("GETRF", ident, f"{rows} {cols} 0", vals))
            elif name.startswith("DGESVD:A"):
                ident += 1
                lines.append(record("GESVD", ident, f"{rows} {cols} 0", vals))
            elif name.startswith("XGESVX:A"):
                pending_a = (rows, cols, vals)
            elif name.startswith("XGESVX:B") and pending_a is not None:
                ident += 1
                n = pending_a[0]
                allv = pending_a[2] + vals
                lines.append(record("GESVX", ident, f"{n} {n} {cols}", allv))
                pending_a = None
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} tapped inputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
