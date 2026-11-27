#!/usr/bin/env python3
"""Extract the complex LAPACK inputs GYRE passed in the G0 oracle calls (#170).

Reads the `*.taps.tsv` files of `benchmarks/fortran_mesa/gyre_oracles/data/`
(package G0) and writes two files.

`inputs` holds one line per input, in the format `gen_gyre_g4_vectors` reads:

    kind id m n bits...

where each `bits` value is the IEEE-754 bit pattern as a signed decimal
64-bit integer and a complex matrix is listed column-major as (re, im)
pairs. The kinds are

- `GESVD`: the `ZGESVD:A` input (`sing_decompose_c_`, `n = 2 n_e`);
- `FACTOR`: the four `n_e x n_e` blocks `E_l(k), E_r(k), E_l(k+1),
  E_r(k+1)` of the first two complex blocks tapped in each `BUILD`, which
  the generator feeds through GYRE's cyclic-reduction step
  (`factor_c_`, `block_sysmtx_m.fypp:203-310`: `ZGETRF(2n_e, n_e)`,
  `ZLASWP`, `ZTRSM`, `ZGEMM`);
- `GEMV`: the first complex block pair of each `BUILD`, for GYRE's
  back-substitution `ZGEMV` pair (`backsub_c_`, `block_sysmtx_m.fypp:564-625`).

The G0 taps hold no complex `XGETRF_FACTOR`/`XGETRF_DET` records (the tap
budget of those names went to real calls), so the complex LU inputs are
built from the tapped blocks and from the `ZGESVD:A` boundary matrices,
which are the matrices `det_c_` factors with `ZGETRF(2n_e, 2n_e)`.

`outputs` holds GYRE's own `ZGESVD` outputs for the replay check:

    id info bits(S) bits(U) bits(VH)
"""

from __future__ import annotations

import argparse
import pathlib
import re

SHAPE = re.compile(r"\[(\d+)x(\d+),colmajor,reim\]$")


def signed(h: str) -> int:
    v = int(h, 16)
    return v - (1 << 64) if v >= 1 << 63 else v


def bits(vals: list[str]) -> str:
    return " ".join(str(signed(v)) for v in vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("taps_dir", type=pathlib.Path)
    ap.add_argument("inputs", type=pathlib.Path)
    ap.add_argument("outputs", type=pathlib.Path)
    args = ap.parse_args()
    lines: list[str] = []
    outs: list[str] = []
    ident = 0
    for path in sorted(args.taps_dir.glob("*.taps.tsv")):
        blocks: list[list[str]] = []  # E_l, E_r, E_l, E_r of this BUILD
        ne = 0
        pending_out: dict[str, str] = {}
        svd_id = 0
        for raw in path.read_text().splitlines():
            if raw.startswith("#"):
                continue
            f = raw.split("\t")
            kind, name = f[0], f[1]
            if kind == "BUILD" and name == "omega":
                blocks = []
                continue
            if kind == "BLOCK" and (":E_l[" in name or ":E_r[" in name):
                m = SHAPE.search(name)
                if m is None:  # real block
                    continue
                ne = int(m.group(1))
                if len(blocks) < 4:
                    blocks.append(f[3].split())
                if len(blocks) == 4:
                    ident += 1
                    allv = [v for b in blocks for v in b]
                    lines.append(f"FACTOR {ident} {ne} {ne} {bits(allv)}")
                    ident += 1
                    pair = blocks[0] + blocks[1]
                    lines.append(f"GEMV {ident} {ne} {ne} {bits(pair)}")
                    blocks.append([])  # one record pair per BUILD
                continue
            if kind == "LAPACK_IN" and name.startswith("ZGESVD:A"):
                m = SHAPE.search(name)
                assert m is not None, name
                rows, cols = int(m.group(1)), int(m.group(2))
                ident += 1
                svd_id = ident
                vals = f[3].split()
                lines.append(f"GESVD {ident} {rows} {cols} {bits(vals)}")
                pending_out = {}
                continue
            if kind == "LAPACK_OUT" and name.startswith("ZGESVD:"):
                key = name.split(":")[1].split("[")[0]
                pending_out[key] = f[3]
                if len(pending_out) == 4:
                    outs.append(
                        f"{svd_id} {pending_out['info']} {pending_out['S']} "
                        f"{pending_out['U']} {pending_out['VH']}"
                    )
                    pending_out = {}
    args.inputs.write_text("\n".join(lines) + "\n")
    args.outputs.write_text("\n".join(outs) + "\n")
    print(
        f"wrote {len(lines)} tapped inputs to {args.inputs} and "
        f"{len(outs)} tapped ZGESVD outputs to {args.outputs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
