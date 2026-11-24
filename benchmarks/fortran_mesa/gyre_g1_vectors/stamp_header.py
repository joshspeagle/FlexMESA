#!/usr/bin/env python3
"""Run the G1 vector generator and prepend a provenance header (#170, G1)."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile

HEADER = [
    "# FlexMESA GYRE G1 numerics-substrate reference vectors (#170, package G1)",
    "# mesa_version: r26.4.1",
    "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
    "# mesa_zenodo_record: 19722306",
    "# gyre_version: 8.1 (bundled with MESA 26.4.1); ForUM 1.0.4 (MESA forum package)",
    "# mesasdk_version: x86_64-linux-26.6.1",
    "# compiler: GCC version 15.2.0 (gfortran)",
    "# math_backend: CRMATH 1.2 (USE_CRMATH = YES)",
    "# generator: benchmarks/fortran_mesa/gyre_g1_vectors/gen_gyre_g1_vectors.f90",
    "# library_code: pristine MESA GYRE build (ext_m, math_m, integ_m), ForUM order_m, CRMATH",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flags", required=True, help="compiler flags used for the generator")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("cmd", nargs="+", help="generator command (after --)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        raw = pathlib.Path(tmp) / "vectors.tsv"
        subprocess.run([*args.cmd, str(raw)], check=True, env={"OMP_NUM_THREADS": "1"})
        body = raw.read_text()
    head = [
        *HEADER,
        f"# fflags: {args.flags}",
        "# columns: section id real_inputs int_inputs real_outputs int_outputs",
        "# reals: IEEE-754 binary64 bit patterns in hex (complex as re im); ints decimal; '-' empty",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(head) + "\n" + body)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
