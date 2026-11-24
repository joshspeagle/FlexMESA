#!/usr/bin/env python3
"""Check the GCC expansions the GYRE G1 port reproduces (#170).

Compiles `probe_expansions.f90` with gfortran (MESA SDK x86_64-linux-26.6.1,
GCC 15.2.0) and GYRE's flags, dumps the optimized GIMPLE and the front-end
tree, checks each expansion the Rust port transcribes, and writes the
relevant excerpts to `gimple_evidence.txt`. Exits non-zero on any mismatch.

Run from this directory after sourcing the SDK (`mesasdk_init.sh`).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FLAGS = (
    "-std=f2018 -finit-real=snan -fmax-errors=25 -fPIC "
    "-ffpe-trap=invalid,overflow,zero -O2 -fno-backtrace -fopenmp -ffp-contract=off"
).split()

# Expected statement sequences (SSA names stripped to '_'), per function
EXPECT = {
    "cmul": ["_ = _ * _;", "_ = _ * _;", "_ = _ * _;", "_ = _ * _;",
             "c$real_ = _ - _;", "c$imag_ = _ + _;"],
    "cdiv": ["_ = ABS_EXPR <_>;", "_ = ABS_EXPR <_>;", "if (_ < _)",
             "_ = _ / _;", "_ = _ * _;", "_ = _ + _;", "_ = _ * _;", "_ = _ + _;",
             "_ = _ * _;", "_ = _ - _;", "_ = _ / _;", "_ = _ / _;",
             "_ = _ / _;", "_ = _ * _;", "_ = _ + _;", "_ = _ * _;", "_ = _ + _;",
             "_ = _ * _;", "_ = _ - _;", "_ = _ / _;", "_ = _ / _;"],
    "crmul": ["_ = _ * _;", "_ = _ * 0.0;", "_ = _ * 0.0;", "_ = _ * _;",
              "c$real_ = _ - _;", "c$imag_ = _ + _;"],
    "cdivr": ["_ = 0.0 / _;", "_ = _ * 0.0;", "_ = _ + _;", "_ = _ * _;", "_ = _ + _;",
              "_ = _ * _;", "_ = _ - _;", "_ = _ / _;", "_ = _ / _;"],
    "rp5": ["powmult_ = _ * _;", "powmult_ = _ * powmult_;", "powmult_ = powmult_ * powmult_;"],
    "rp6": ["powmult_ = _ * _;", "powmult_ = _ * powmult_;", "powmult_ = powmult_ * powmult_;"],
    "ipown": ["y_ = __builtin_powi (_, _); [tail call]"],
    "cp2": ["_ = _ * _;", "_ = _ * _;", "_ = _ * _;", "_ = _ - _;", "_ = _ * 2.0e+0;"],
}


def norm(line: str) -> str:
    s = line.strip()
    s = re.sub(r"\b(_|c\$real_|c\$imag_|powmult_|y_)\d+(\(D\))?", r"\1", s)
    s = re.sub(r"\*\w+_\d+\(D\)", "_", s)
    # * and + commute exactly in IEEE-754: compare them operand-order free
    m = re.match(r"^(\S+) = (\S+) ([*+]) (\S+);$", s)
    if m:
        a, b = sorted([m.group(2), m.group(4)])
        s = f"{m.group(1)} = {a} {m.group(3)} {b};"
    return s


def function_body(dump: str, name: str) -> list[str]:
    m = re.search(rf"^;; Function {name} \(.*?\n(.*?)^}}", dump, re.S | re.M)
    if not m:
        raise SystemExit(f"function {name} not found in dump")
    return m.group(1).splitlines()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        src = HERE / "probe_expansions.f90"
        subprocess.run(
            ["gfortran", *FLAGS, "-fdump-tree-optimized", "-fdump-tree-original",
             "-c", str(src), "-o", f"{tmp}/probe.o", "-J", tmp],
            check=True, cwd=tmp,
        )
        opt = next(pathlib.Path(tmp).glob("*.optimized")).read_text()
        orig = next(pathlib.Path(tmp).glob("*.original")).read_text()
        version = subprocess.run(["gfortran", "--version"], capture_output=True,
                                 text=True, check=True).stdout.splitlines()[0]

    out = [
        "# GCC expansions reproduced by crates/flexmesa-gyre (#170, package G1)",
        f"# compiler: {version}",
        f"# flags: {' '.join(FLAGS)}",
        "# source: benchmarks/fortran_mesa/gyre_g1_vectors/probe_expansions.f90",
        "",
    ]
    bad = 0
    for name, want in EXPECT.items():
        body = function_body(opt, name)
        stmts = [norm(l) for l in body
                 if re.search(r" = .*[-+*/]|ABS_EXPR|if \(|__builtin_powi", l)
                 and "REALPART" not in l and "IMAGPART" not in l
                 and not re.search(r" = \*\w", l)]
        ok = stmts == [norm(w) for w in want]
        bad += not ok
        out.append(f"== {name}: {'OK' if ok else 'MISMATCH'}")
        out += ["   " + l.strip() for l in body
                if re.search(r" = |if \(|goto|return", l)
                and not re.match(r"\s*(real|complex)\(kind", l)]
        if not ok:
            out.append(f"   expected: {want}")
            out.append(f"   found:    {stmts}")
        out.append("")

    # Inlined MATMUL: zero fill, then C(i,j) = C(i,j) + A(i,l)*B(l,j) inside
    # j (outer), l, i (inner) loops, guarded by m*k*n <= 2.7e4 (REAL(4))
    mm = re.search(r"^void mm .*?^}", orig, re.S | re.M).group(0)
    checks = {
        "size guard <= 2.7e+4": "<= 2.7e+4" in mm,
        "zero fill": "= 0.0;" in mm,
        "accumulate C + A*B": re.search(r"c\.0 \+ .*\) = \*\(\(real\(kind=8\) \*\) c\.0 .*\+ \*\(\(real\(kind=8\) \*\) a\.0", mm) is not None,
        "outer j, middle l, inner i": re.search(r"__var_3_do.*__var_4_do.*__var_5_do", mm, re.S) is not None,
        "library fallback": "_gfortran_matmul_r8" in mm,
    }
    mv = re.search(r"^void mv .*?^}", orig, re.S | re.M).group(0)
    checks["matvec: no size guard"] = "2.7e+4" not in mv and "_gfortran_matmul" not in mv
    out.append("== MATMUL (front-end tree, -fdump-tree-original)")
    for k, v in checks.items():
        out.append(f"   {k}: {'OK' if v else 'MISMATCH'}")
        bad += not v
    acc = [l.strip() for l in mm.splitlines() if "a.0" in l and "c.0" in l and "=" in l]
    out += ["   " + l for l in acc[:1]]

    (HERE / "gimple_evidence.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[-10:]))
    print(f"{'all expansions as expected' if bad == 0 else f'{bad} mismatches'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
