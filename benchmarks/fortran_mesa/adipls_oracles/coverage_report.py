#!/usr/bin/env python3
"""Turn gcov output of `run_oracles.py coverage` into the ADIPLS reach tables.

Usage:
  coverage_report.py --cov <coverage dir> --adipls-src <adipls/adipack.c> --out <results dir>

<coverage dir> holds adipls_coverage.json of the union run and one
subdirectory per case (from --per-case). The ADIPLS library sources are the
LIB_SRCS of adipls/make/makefile_adipls (MESA 26.4.1); `source_lines` counts
every line of those files, `executable_lines` the lines gcov instruments.
Writes adipls_reach_files.tsv, adipls_reach_procedures.tsv and
reach_summary.md (#171, package A0).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct

SHORT = {
    "astero__astero_adipls": "astero_adipls",
    "astero__surface_effects": "surface_effects",
    "astero__example_astero": "example_astero",
    "astero__fast_scan_grid": "fast_scan_grid",
    "astero__fast_simplex": "fast_simplex",
}


def lib_sources(adipls_src: pathlib.Path, makefile: pathlib.Path) -> dict[str, pathlib.Path]:
    """LIB_SRCS of makefile_adipls, resolved through its SRC_PATH, plus the
    two generated files combined into ofiles_blstio.f."""
    text = makefile.read_text()
    m = re.search(r"LIB_SRCS = \\\n(.*?)\n\n", text, re.S)
    assert m, "LIB_SRCS not found"
    names = m.group(1).replace("\\", " ").split()
    out: dict[str, pathlib.Path] = {}
    for n in names:
        for d in ("adipls", "gensr", "adiajobs/sr"):
            p = adipls_src / d / n
            if p.exists():
                out[f"{d}/{n}"] = p
                break
        else:
            raise SystemExit(f"source {n} not found")
    # makefile_adipls compiles ofiles.c.f and blstio.f as their concatenation
    # ofiles_blstio.f (a linker workaround for the BLOCK DATA in blstio.f)
    out["ofiles_blstio.f"] = adipls_src / "gensr" / "ofiles.c.f"
    return out


def source_text(fkey: str, path: pathlib.Path) -> str:
    if fkey == "ofiles_blstio.f":
        return path.read_text(errors="replace") + (path.parent / "blstio.f").read_text(
            errors="replace"
        )
    return path.read_text(errors="replace")


LOGICAL_OPS = re.compile(r"\.(eq|ne|lt|le|gt|ge|and|or|not|eqv|neqv|true|false)\.", re.I)
REAL_LIT = re.compile(
    r"(?<![\w])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w])"
)


def sp_literals(line: str) -> list[str]:
    """Default-kind (binary32) real literals in one fixed-form statement line."""
    if not line or line[0] in "cC*!":
        return []
    body = line[6:72]
    body = re.sub(r"'[^']*'", "''", body)
    body = body.split("!")[0]
    body = LOGICAL_OPS.sub(" ", body)
    return REAL_LIT.findall(body)


SP_CONV = re.compile(r"(?<![\w])(float|sngl|real)\s*\(", re.I)


def sp_conversions(line: str) -> list[str]:
    """Conversions to default-kind real (float(i), sngl(x), real(x) without kind)."""
    if not line or line[0] in "cC*!":
        return []
    body = re.sub(r"'[^']*'", "''", line[6:72]).split("!")[0]
    if re.match(r"\s*(double precision|real|implicit)\b", body, re.I):
        return []
    found = []
    for m in SP_CONV.finditer(body):
        # real(x, kind) is not default kind; take the argument text up to the matching paren
        depth, j = 1, m.end()
        while j < len(body) and depth:
            depth += {"(": 1, ")": -1}.get(body[j], 0)
            j += 1
        arg = body[m.end() : j - 1]
        if m.group(1).lower() == "real" and "," in arg:
            continue
        found.append(f"{m.group(1)}({arg})")
    return found


def f32_exact(lit: str) -> bool:
    v = float(lit)
    return struct.unpack("<f", struct.pack("<f", v))[0] == v


def key_of(name: str) -> str:
    # gcov paths look like ../adipack.c/adipls/x.f or ofiles_blstio.f
    name = re.sub(r"^(\.\./)+", "", name).replace("adipack.c/", "")
    return name


def load(path: pathlib.Path) -> dict:
    raw = json.loads(path.read_text())
    return {key_of(k): v for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", type=pathlib.Path, required=True)
    ap.add_argument("--adipls-src", type=pathlib.Path, required=True, help="MESA adipls/adipack.c")
    ap.add_argument("--makefile", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    mk = args.makefile or args.adipls_src.parent / "make" / "makefile_adipls"
    srcs = lib_sources(args.adipls_src, mk)
    union = load(args.cov / "adipls_coverage.json")
    # ofiles.c.f and blstio.f are compiled as the concatenation ofiles_blstio.f
    per_case = {}
    for d in sorted(args.cov.iterdir()):
        if d.is_dir() and (d / "adipls_coverage.json").exists():
            per_case[SHORT.get(d.name, d.name)] = load(d / "adipls_coverage.json")
    case_names = list(per_case)

    def reached_cases(fkey: str, fn: str | None = None) -> str:
        cs = []
        for c, cov in per_case.items():
            ent = cov.get(fkey)
            if not ent:
                continue
            if fn is None:
                if any(v > 0 for v in ent["lines"].values()):
                    cs.append(c)
            elif ent["functions"].get(fn, {}).get("count", 0) > 0:
                cs.append(c)
        return ",".join(cs) or "-"

    file_rows = [
        "file\tsource_lines\texecutable_lines\texecuted_lines\tprocedures\tprocedures_reached"
        "\treached_procedure_span_lines\tcases"
    ]
    proc_rows = [
        "file\tprocedure\tstart_line\tend_line\tcalls\texecutable_lines\texecuted_lines"
        "\tblocks_executed\tblocks\tcases"
    ]
    tot = dict(src=0, exe=0, hit=0, procs=0, preach=0, span=0, files_hit=0, src_hit=0)
    for fkey, path in sorted(srcs.items()):
        nsrc = len(source_text(fkey, path).splitlines())
        tot["src"] += nsrc
        ckey = fkey if fkey in union else None
        ent = union.get(ckey) if ckey else None
        if ent is None:
            file_rows.append(f"{fkey}\t{nsrc}\t-\t0\t-\t0\t0\t-")
            continue
        lines = ent["lines"]
        exe = len(lines)
        hit = sum(1 for v in lines.values() if v > 0)
        fns = ent["functions"]
        reached = {k: v for k, v in fns.items() if v["count"] > 0}
        span = sum(v["end"] - v["start"] + 1 for v in reached.values())
        tot["exe"] += exe
        tot["hit"] += hit
        tot["procs"] += len(fns)
        tot["preach"] += len(reached)
        tot["span"] += span
        if hit:
            tot["files_hit"] += 1
            tot["src_hit"] += nsrc
        file_rows.append(
            f"{fkey}\t{nsrc}\t{exe}\t{hit}\t{len(fns)}\t{len(reached)}\t{span}\t{reached_cases(ckey)}"
        )
        for name, v in sorted(fns.items(), key=lambda kv: kv[1]["start"]):
            if v["count"] == 0:
                continue
            in_fn = [c for ln_, c in lines.items() if v["start"] <= int(ln_) <= v["end"]]
            proc_rows.append(
                f"{fkey}\t{name}\t{v['start']}\t{v['end']}\t{v['count']}\t{len(in_fn)}"
                f"\t{sum(c > 0 for c in in_fn)}\t{v['blocks_executed']}"
                f"\t{v['blocks']}\t{reached_cases(ckey, name)}"
            )
    # binary32 literals on executed lines (the port must use (lit_f32 as f64))
    lit_rows = ["file\tline\tcount\tliteral_or_conversion\texact_in_binary32\tsource"]
    n_inexact = 0
    n_conv = 0
    for fkey, path in sorted(srcs.items()):
        ckey = fkey if fkey in union else None
        if ckey is None:
            continue
        src_lines = source_text(fkey, path).splitlines()
        for ln, cnt in sorted(union[ckey]["lines"].items(), key=lambda kv: int(kv[0])):
            if cnt <= 0:
                continue
            text = src_lines[int(ln) - 1]
            for lit in sp_literals(text):
                ex = f32_exact(lit)
                n_inexact += not ex
                lit_rows.append(
                    f"{fkey}\t{ln}\t{cnt}\t{lit}\t{'yes' if ex else 'no'}\t{text.strip()}"
                )
            for conv in sp_conversions(text):
                n_conv += 1
                lit_rows.append(f"{fkey}\t{ln}\t{cnt}\t{conv}\tconversion\t{text.strip()}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adipls_sp_literals_reached.tsv").write_text("\n".join(lit_rows) + "\n")
    (args.out / "adipls_reach_files.tsv").write_text("\n".join(file_rows) + "\n")
    (args.out / "adipls_reach_procedures.tsv").write_text("\n".join(proc_rows) + "\n")

    sel = json.loads((args.cov / "coverage_selection.json").read_text())
    md = [
        "# ADIPLS reach measured on the MESA 26.4.1 ADIPLS calls",
        "",
        "Generated by `coverage_report.py` from gcov output of `run_oracles.py coverage`.",
        "ADIPLS as bundled with MESA r26.4.1 (commit c8c202662922a22b3efefa824473ad615c623afd),",
        "rebuilt with MESA's flags (`gfortran -std=legacy -w`, no `-O` flag) plus `--coverage`;",
        "MESA SDK x86_64-linux-26.6.1, GCC 15.2.0. Inputs: the glue-tap captures of the MESA",
        "runs (`run_mesa_captures.sh`). Coverage records reach only; oracle values come from",
        "builds with MESA's flags.",
        "",
        "Union run:",
        "",
    ]
    md += [f"- `{k}`: {len(v)} calls" for k, v in sel.items()]
    md += [
        "",
        f"- Library source files (`LIB_SRCS` of `makefile_adipls`): {len(srcs)}, "
        f"{tot['src']} lines",
        f"- Files with at least one executed line: {tot['files_hit']} "
        f"({tot['src_hit']} source lines)",
        f"- Executable lines executed: {tot['hit']} of {tot['exe']}",
        f"- Procedures reached: {tot['preach']} of {tot['procs']}",
        f"- Source span of reached procedures: {tot['span']} lines",
        f"- Default-kind (binary32) real literals on executed lines: {len(lit_rows) - 1}, "
        f"of which {n_inexact} are not exact in binary32; conversions to binary32 "
        f"(`float`, `sngl`, `real` without kind) on executed lines: {n_conv} "
        "(`adipls_sp_literals_reached.tsv`)",
        "",
        "## Per-case files reached",
        "",
        "| file | " + " | ".join(case_names) + " |",
        "|---|" + "---|" * len(case_names),
    ]
    for row in file_rows[1:]:
        f = row.split("\t")
        if f[3] == "0":
            continue
        marks = []
        for c in case_names:
            marks.append("x" if c in f[7].split(",") else "")
        md.append(f"| `{f[0]}` | " + " | ".join(marks) + " |")
    (args.out / "reach_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[10:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
