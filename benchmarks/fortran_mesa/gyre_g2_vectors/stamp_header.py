#!/usr/bin/env python3
"""Run the G2 vector generator and prepend a provenance header (#170, G2)."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import tempfile

HEADER = [
    "# FlexMESA GYRE G2 real LAPACK/BLAS reference vectors (#170, package G2)",
    "# mesa_version: r26.4.1",
    "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
    "# mesa_zenodo_record: 19722306",
    "# mesasdk_version: x86_64-linux-26.6.1",
    "# compiler: GCC version 15.2.0 (gfortran)",
    "# libraries: SDK crmath math slot liblapack.a / libblas.a"
    " (Reference LAPACK 3.12.0,",
    "#   gfortran -O2 -fPIC; dlamch -O0; dbdsqr and iparmq CRMATH-patched)",
    "# generator: benchmarks/fortran_mesa/gyre_g2_vectors/"
    "gen_gyre_g2_vectors.f90",
    "# tapped_inputs: real LAPACK inputs of the G0 oracle calls",
    "#   (benchmarks/fortran_mesa/gyre_oracles/data/*.taps.tsv)",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taps-dir", type=pathlib.Path, required=True)
    ap.add_argument("--inputs", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("cmd", nargs="+")
    args = ap.parse_args()
    h = hashlib.sha256()
    for p in sorted(args.taps_dir.glob("*.taps.tsv")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        raw = pathlib.Path(tmp) / "v.tsv"
        subprocess.run([*args.cmd, str(args.inputs), str(raw)], check=True)
        body = raw.read_text()
    head = [
        *HEADER,
        f"# tapped_inputs_sha256 (taps files by name): {h.hexdigest()}",
        "# columns: section id real_inputs int_inputs real_outputs int_outputs",
        "# reals: IEEE-754 binary64 bit patterns in hex; ints decimal;"
        " '-' empty",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(head) + "\n" + body)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
