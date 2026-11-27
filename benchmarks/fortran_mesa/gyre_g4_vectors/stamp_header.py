#!/usr/bin/env python3
"""Run the G4 vector generator, check the replay, add a header (#170, G4).

The replay check compares each `gesvd_tap` record's `S`, `U` and `VT`
outputs and `INFO` with GYRE's own tapped `ZGESVD` outputs
(`LAPACK_OUT ZGESVD:*` in the G0 taps) and fails unless every bit matches.

Inputs and outputs that the committed G0 taps already hold are not written
again (`compact`): the `gesvd_tap` records are dropped once the replay check
passes, because the test compares against the tapped `ZGESVD` inputs and
outputs directly; the real inputs of `getrf_det_tap` and `factor_tap`
(tapped boundary matrices and blocks) are written as `-`, and those of
`gemv_tap` keep only the synthetic `x1`, `x2`, `y0` after the tapped
blocks, for the test to rebuild from the taps by record id as
`extract_tapped_inputs.py` numbers them.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import tempfile

HEADER = [
    "# FlexMESA GYRE G4 complex LAPACK/BLAS reference vectors"
    " (#170, package G4)",
    "# mesa_version: r26.4.1",
    "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
    "# mesa_zenodo_record: 19722306",
    "# mesasdk_version: x86_64-linux-26.6.1",
    "# compiler: GCC version 15.2.0 (gfortran)",
    "# libraries: SDK crmath math slot liblapack.a / libblas.a"
    " (Reference LAPACK 3.12.0,",
    "#   gfortran -O2 -fPIC; zgetrf2, zlange, zdscal, zladiv, zlarfg and"
    " zbdsqr CRMATH-patched)",
    "# generator: benchmarks/fortran_mesa/gyre_g4_vectors/"
    "gen_gyre_g4_vectors.f90",
    "# tapped_inputs: complex LAPACK inputs and blocks of the G0 oracle calls",
    "#   (benchmarks/fortran_mesa/gyre_oracles/data/*.taps.tsv)",
]


def replay_check(body: str, tapped: pathlib.Path) -> tuple[int, int]:
    """Return (matching, total) gesvd_tap records against GYRE's outputs."""
    recs: dict[int, tuple[list[str], list[str], list[str]]] = {}
    for line in body.splitlines():
        c = line.split("\t")
        if c[0] != "gesvd_tap":
            continue
        n = int(c[3].split()[0])
        out = c[4].split()
        # a (2n^2), s (n), u (2n^2), vt (2n^2), work(1) (2)
        q = 2 * n * n
        s = out[q : q + n]
        u = out[q + n : 2 * q + n]
        vt = out[2 * q + n : 3 * q + n]
        recs[int(c[1])] = (s + u + vt, c[5].split(), [])
    total = 0
    good = 0
    for raw in tapped.read_text().splitlines():
        if not raw.strip():
            continue
        f = raw.split()
        ident, info, vals = int(f[0]), f[1], f[2:]
        total += 1
        got = recs.get(ident)
        if got is None:
            continue
        if [v.upper() for v in vals] == got[0] and got[1] == [info]:
            good += 1
        else:
            bad = next(
                k
                for k, (a, b) in enumerate(zip(vals, got[0], strict=False))
                if a != b
            )
            print(f"replay mismatch: gesvd_tap {ident} at output {bad}")
    return good, total


FROM_TAPS = ("getrf_det_tap", "factor_tap")


def compact(body: str) -> str:
    """Drop data the committed G0 taps already hold (see module docstring)."""
    out: list[str] = []
    for line in body.splitlines():
        c = line.split("\t")
        if c[0] == "gesvd_tap":
            continue
        if c[0] in FROM_TAPS:
            c[2] = "-"
        elif c[0] == "gemv_tap":
            # keep the synthetic x1, x2, y0 after the tapped E_l, E_r
            ne = int(c[3])
            c[2] = " ".join(c[2].split()[4 * ne * ne :])
        out.append("\t".join(c))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taps-dir", type=pathlib.Path, required=True)
    ap.add_argument("--inputs", type=pathlib.Path, required=True)
    ap.add_argument("--tapped-outputs", type=pathlib.Path, required=True)
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
    good, total = replay_check(body, args.tapped_outputs)
    print(f"replay: {good}/{total} ZGESVD records reproduce GYRE's outputs")
    if good != total or total == 0:
        return 1
    head = [
        *HEADER,
        f"# tapped_inputs_sha256 (taps files by name): {h.hexdigest()}",
        f"# replay: {good}/{total} tapped ZGESVD calls reproduce GYRE's"
        " own S, U, VH and INFO bit for bit",
        "# columns: section id real_inputs int_inputs real_outputs int_outputs",
        "# reals: IEEE-754 binary64 bit patterns in hex, complex values as"
        " (re, im); ints decimal; '-' empty",
        "# from the taps: gesvd_tap records (inputs and GYRE's outputs),"
        " the real inputs of getrf_det_tap and factor_tap ('-') and the"
        " tapped E_l, E_r that precede the gemv_tap inputs are read from"
        " the G0 taps by record id",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(head) + "\n" + compact(body))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
