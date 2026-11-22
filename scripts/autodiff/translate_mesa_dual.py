#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Translate a generated MESA ``auto_diff`` module into Rust functions.

MESA generates each ``auto_diff/private/auto_diff_real_*_module.f90`` with one
straight-line block of assignments per operation: temporaries ``q0, q1, ...``
followed by one assignment per component of the result. This script reads such
a module from a MESA checkout and writes one Rust function per routine, keeping
every expression, its parentheses and the order of the assignments, so that
the Rust type rounds exactly as the Fortran one does (`#349`).

Comparison, assignment and ``integer``-operand routines are not translated: the
Rust types compare on ``val`` in their trait implementations and convert
integers to ``f64`` before calling the real-operand routine.

Usage::

    python scripts/autodiff/translate_mesa_dual.py \
        --mesa-dir <MESA 26.4.1 checkout> \
        --module auto_diff_real_2var_order3 \
        --rust-type AutoDiffReal2VarOrder3 \
        --out crates/flexmesa-autodiff/src/order3_gen.rs
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SKIP_PREFIXES = (
    "assign_",
    "equal_",
    "neq_",
    "greater_",
    "less_",
    "leq_",
    "geq_",
    "differentiate_",
)

# Fortran name -> Rust name. Names absent here are emitted unchanged.
FUNCTION_MAP = {
    "Abs": "f_abs",
    "Max": "f_max",
    "Min": "f_min",
    "Heaviside": "heaviside",
    "sgn": "sgn",
    "sqrt": "f_sqrt",
}
CONSTANT_MAP = {"pi": "PI", "ln10": "LN10"}

FLOAT_RE = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eEdD][+-]?\d+)?|\d+[eEdD][+-]?\d+)(?:_dp)?")
INT_RE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def read_module(mesa_dir: Path, module: str, tag: str) -> str:
    """Return the module source at ``tag`` from a git checkout of MESA."""
    rel = f"auto_diff/private/{module}_module.f90"
    out = subprocess.run(
        ["git", "-C", str(mesa_dir), "show", f"{tag}:{rel}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def join_statements(body: str) -> list[str]:
    """Join Fortran continuation lines and drop declarations and comments."""
    stmts: list[str] = []
    buf = ""
    for raw in body.split("\n"):
        line = raw.split("!", 1)[0].rstrip()
        if not line.strip():
            continue
        text = line.strip()
        if text.startswith("&"):
            text = text[1:].strip()
        cont = text.endswith("&")
        if cont:
            text = text[:-1].rstrip()
        buf = f"{buf} {text}".strip()
        if cont:
            continue
        stmts.append(buf)
        buf = ""
    return [
        s
        for s in stmts
        if not re.match(r"^(type\(|real\(dp\)|integer|logical|include)", s)
    ]


def translate_expr(expr: str) -> str:
    """Translate one Fortran expression to Rust, keeping its structure."""
    out = expr.replace("%", ".")

    def float_sub(m: re.Match[str]) -> str:
        lit = m.group(1).replace("d", "e").replace("D", "e")
        if "." not in lit and "e" in lit:
            mant, exp = lit.split("e", 1)
            lit = f"{mant}.0e{exp}"
        if lit.endswith("."):
            lit += "0"
        return lit

    out = FLOAT_RE.sub(float_sub, out)
    out = INT_RE.sub(lambda m: f"{m.group(1)}.0", out)
    for fort, rust in FUNCTION_MAP.items():
        out = re.sub(rf"\b{fort}\s*\(", f"{rust}(", out)
    for fort, rust in CONSTANT_MAP.items():
        out = re.sub(rf"(?<![\w.]){fort}(?![\w(])", rust, out)
    return out


def translate_routine(name: str, args: list[str], result: str, body: str,
                     decls: dict[str, str], rust_type: str) -> str:
    params = []
    for a in args:
        kind = decls.get(a)
        if kind is None:
            raise ValueError(f"{name}: no declaration for argument {a}")
        params.append(f"{a}: {rust_type if kind == 'dual' else 'f64'}")
    lines = [
        f"/// `{name}` in the upstream module.",
        "#[inline]",
        "#[must_use]",
        f"pub(crate) fn {name}({', '.join(params)}) -> {rust_type} {{",
        f"    let mut {result} = {rust_type}::default();",
    ]
    for stmt in join_statements(body):
        m = re.match(r"^([\w%]+)\s*=\s*(.*)$", stmt)
        if not m:
            raise ValueError(f"{name}: cannot translate statement: {stmt}")
        lhs, rhs = m.group(1), m.group(2)
        rhs_rust = translate_expr(rhs)
        if "%" in lhs:
            lines.append(f"    {lhs.replace('%', '.')} = {rhs_rust};")
        else:
            lines.append(f"    let {lhs} = {rhs_rust};")
    lines.append(f"    {result}")
    lines.append("}")
    return "\n".join(lines)


def translate_module(
    src: str, module: str, rust_type: str, tag: str
) -> tuple[str, int]:
    # Join continued lines first, so that a routine header split over several
    # lines matches like any other.
    src = re.sub(r"&[ \t]*\n[ \t]*&?", " ", src)
    routines = re.findall(
        r"\n   function (\w+)\((.*?)\)\s*result\((\w+)\)(.*?)\n   end function \1",
        src,
        re.S,
    )
    parts = [
        "// SPDX-License-Identifier: LGPL-3.0-or-later",
        f"//! Generated from MESA `{tag}` `auto_diff/private/{module}_module.f90`",
        "//! by `scripts/autodiff/translate_mesa_dual.py`. Do not edit by hand:",
        "//! rerun the script against a MESA checkout instead (`#349`).",
        "//!",
        "//! Each function keeps the upstream routine's name, its temporaries,",
        "//! their order and every parenthesis, so the result rounds as MESA's.",
        "",
        "#![allow(",
        "    clippy::many_single_char_names,",
        "    clippy::too_many_lines,",
        "    clippy::too_many_arguments,",
        "    clippy::neg_multiply,",
        "    clippy::field_reassign_with_default,",
        "    clippy::unreadable_literal,",
        "    clippy::suboptimal_flops,",
        "    clippy::similar_names,",
        "    clippy::manual_midpoint,",
        "    clippy::precedence,",
        "    clippy::unnecessary_cast,",
        "    unused_parens,",
        "    dead_code",
        ")]",
        "",
        "use flexmesa_const::{LN10, PI};",
        "use flexmesa_math::{",
        "    acos, acosh, asin, asinh, atan, atanh, cos, cosh, exp, expm1, log,",
        "    log1p, pow, pow2, pow3, pow4, pow5, pow6, pow7, pow8, powm1, safe_log,",
        "    sin, sinh, tan, tanh,",
        "};",
        "",
        f"use crate::support::{{f_abs, f_max, f_min, f_sqrt, heaviside, sgn}};",
        f"use crate::{rust_type};",
    ]
    n = 0
    for name, arglist, result, body in routines:
        if name.startswith(SKIP_PREFIXES) or name.endswith("_int") or "_int_" in name:
            continue
        args = [a.strip() for a in arglist.split(",") if a.strip()]
        decls: dict[str, str] = {}
        for d in re.findall(r"^\s*(type\(\w+\)|real\(dp\)), intent\(in\) :: (.*)$", body, re.M):
            kind = "dual" if d[0].startswith("type") else "real"
            for v in d[1].split(","):
                decls[v.strip()] = kind
        parts.append("")
        parts.append(translate_routine(name, args, result, body, decls, rust_type))
        n += 1
    parts.append("")
    return "\n".join(parts), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mesa-dir", type=Path, required=True)
    ap.add_argument("--tag", default="26.4.1")
    ap.add_argument("--module", required=True)
    ap.add_argument("--rust-type", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = read_module(a.mesa_dir, a.module, a.tag)
    text, n = translate_module(src, a.module, a.rust_type, a.tag)
    a.out.write_text(text, encoding="utf-8")
    print(f"wrote {n} routines to {a.out}")


if __name__ == "__main__":
    main()
