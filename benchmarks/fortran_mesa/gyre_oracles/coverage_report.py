#!/usr/bin/env python3
"""Turn the gcov summaries of `run_oracles.py coverage` into reach tables.

Reads `<cov>/gyre_coverage.json` and `<cov>/lapack_coverage.json` (union of
all calls) plus `<cov>/per_case/<suite>__<case>/` (one pass per case with
counters reset) and writes, into `--out`:

  gyre_reach_files.tsv        per GYRE source file: source lines, executable
                              lines, executed lines, procedures reached,
                              source span of the reached procedures, cases
  gyre_reach_procedures.tsv   per reached GYRE procedure (template instances
                              separately): line span, calls, blocks, cases
  lapack_reach_routines.tsv   per reached LAPACK/BLAS routine: calls, lines
                              executed of executable lines, cases, and the
                              executable lines never executed (the branches
                              GYRE does not take)
  reach_summary.md            totals and the per-case file matrix

Compiler-generated procedures (`__copy_*`, `__final_*`) are excluded from
procedure counts; their lines are type definitions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

GEN_RE = re.compile(r"___(copy|final)_")
MOD_RE = re.compile(r"^__(\w+?)_MOD_(\w+)$")


def proc_name(raw: str) -> str:
    m = MOD_RE.match(raw)
    return f"{m.group(1)}::{m.group(2)}" if m else raw


def ranges(nums: list[int]) -> str:
    if not nums:
        return "-"
    nums = sorted(nums)
    out = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ",".join(out)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def reached_files(cov: dict) -> set[str]:
    return {f for f, d in cov.items() if any(c > 0 for c in d["lines"].values())}


def reached_procs(cov: dict) -> set[tuple[str, str]]:
    out = set()
    for f, d in cov.items():
        for name, v in d["functions"].items():
            if v["count"] > 0 and not GEN_RE.search(name):
                out.add((f, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cov", type=pathlib.Path, required=True)
    ap.add_argument(
        "--gyre-src",
        type=pathlib.Path,
        required=True,
        help="$MESA_DIR/gyre/gyre/src (for source line counts)",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gyre = load(args.cov / "gyre_coverage.json")
    lap = load(args.cov / "lapack_coverage.json")
    per_case: dict[str, tuple[dict, dict]] = {}
    pc_dir = args.cov / "per_case"
    if pc_dir.is_dir():
        for d in sorted(pc_dir.iterdir()):
            if d.is_dir():
                key = d.name.replace("__", "/")
                per_case[key] = (load(d / "gyre_coverage.json"), load(d / "lapack_coverage.json"))
    short = {k: k.split("/")[1].replace("gyre_in_mesa_", "gim_") for k in per_case}

    case_files = {k: reached_files(g) for k, (g, _) in per_case.items()}
    case_procs = {k: reached_procs(g) for k, (g, _) in per_case.items()}
    case_lap = {k: reached_procs(lp) for k, (_, lp) in per_case.items()}

    # GYRE files
    rows = [
        "file\tsource_lines\texecutable_lines\texecuted_lines\tprocedures\t"
        "procedures_reached\treached_procedure_span_lines\tcases"
    ]
    totals = dict(src=0, exe=0, hit=0, procs=0, reached=0, span=0)
    reached_total_src = 0
    proc_rows = ["file\tprocedure\tstart_line\tend_line\tcalls\tblocks_executed\tblocks\tcases"]
    for f in sorted(gyre):
        d = gyre[f]
        src_path = args.gyre_src / f
        src_lines = len(src_path.read_text().splitlines()) if src_path.exists() else 0
        exe = len(d["lines"])
        hit = sum(1 for c in d["lines"].values() if c > 0)
        procs = {n: v for n, v in d["functions"].items() if not GEN_RE.search(n)}
        reached = {n: v for n, v in procs.items() if v["count"] > 0}
        span_lines: set[int] = set()
        for v in reached.values():
            span_lines.update(range(v["start"], v["end"] + 1))
        cases = [short[k] for k in per_case if f in case_files[k]]
        rows.append(
            f"{f}\t{src_lines}\t{exe}\t{hit}\t{len(procs)}\t{len(reached)}\t"
            f"{len(span_lines)}\t{','.join(cases) or '-'}"
        )
        totals["src"] += src_lines
        totals["exe"] += exe
        totals["hit"] += hit
        totals["procs"] += len(procs)
        totals["reached"] += len(reached)
        totals["span"] += len(span_lines)
        if hit:
            reached_total_src += src_lines
        for n, v in sorted(reached.items(), key=lambda kv: kv[1]["start"]):
            pc = [short[k] for k in per_case if (f, n) in case_procs[k]]
            proc_rows.append(
                f"{f}\t{proc_name(n)}\t{v['start']}\t{v['end']}\t{v['count']}\t"
                f"{v['blocks_executed']}\t{v['blocks']}\t{','.join(pc) or '-'}"
            )
    (args.out / "gyre_reach_files.tsv").write_text("\n".join(rows) + "\n")
    (args.out / "gyre_reach_procedures.tsv").write_text("\n".join(proc_rows) + "\n")

    # LAPACK / BLAS routines
    lrows = ["source\troutine\tcalls\texecuted_lines\texecutable_lines\tcases\tunexecuted_lines"]
    n_routines = 0
    lap_by_case: dict[str, set[str]] = {k: set() for k in per_case}
    for f in sorted(lap):
        d = lap[f]
        for n, v in sorted(d["functions"].items()):
            if v["count"] <= 0:
                continue
            n_routines += 1
            lines = {
                int(ln): c for ln, c in d["lines"].items() if v["start"] <= int(ln) <= v["end"]
            }
            hit = [ln for ln, c in lines.items() if c > 0]
            miss = [ln for ln, c in lines.items() if c == 0]
            cases = [short[k] for k in per_case if (f, n) in case_lap[k]]
            for k in per_case:
                if (f, n) in case_lap[k]:
                    lap_by_case[k].add(n.rstrip("_"))
            lrows.append(
                f"{f}\t{n.rstrip('_')}\t{v['count']}\t{len(hit)}\t{len(lines)}\t"
                f"{','.join(cases) or '-'}\t{ranges(miss)}"
            )
    (args.out / "lapack_reach_routines.tsv").write_text("\n".join(lrows) + "\n")

    # Summary
    md = [
        "# GYRE reach measured on the MESA 26.4.1 GYRE calls",
        "",
        "Generated by `coverage_report.py` from gcov output of `run_oracles.py coverage`.",
        "GYRE 8.1 as bundled with MESA r26.4.1 (commit c8c202662922a22b3efefa824473ad615c623afd),",
        "rebuilt with `-O0 --coverage` (otherwise MESA's GYRE configuration); reference",
        "LAPACK/BLAS 3.12.0 sources rebuilt with `-O0 --coverage`; MESA SDK",
        "x86_64-linux-26.6.1, GCC 15.2.0. Inputs: the pulse_tap captures of the",
        "`mesa_histories` bundle. Coverage records reach only; oracle values come",
        "from builds with MESA's flags.",
        "",
    ]
    sel = load(args.cov / "coverage_selection.json")
    md.append("Union run (all captured calls unless noted in coverage_selection.json):")
    md.append("")
    for k, v in sel.items():
        md.append(f"- `{k}`: {len(v)} calls")
    md += [
        "",
        f"- GYRE files with at least one executed line: "
        f"{sum(1 for f in gyre if any(c > 0 for c in gyre[f]['lines'].values()))} "
        f"of {len(gyre)} linked files",
        f"- Source lines of reached files: {reached_total_src}",
        f"- Executable lines executed: {totals['hit']} of {totals['exe']} in linked files",
        f"- Procedures reached: {totals['reached']} of {totals['procs']} in linked files "
        "(template instances counted separately)",
        f"- Source span of reached procedures: {totals['span']} lines",
        f"- LAPACK/BLAS routines reached: {n_routines}",
        "",
        "## Per-case GYRE files reached",
        "",
        "| file | " + " | ".join(short[k] for k in per_case) + " |",
        "|---|" + "---|" * len(per_case),
    ]
    for f in sorted(gyre):
        if not any(c > 0 for c in gyre[f]["lines"].values()):
            continue
        md.append(
            f"| `{f}` | " + " | ".join("x" if f in case_files[k] else "" for k in per_case) + " |"
        )
    md += ["", "## Per-case LAPACK/BLAS routines reached", ""]
    for k in per_case:
        md.append(f"- `{short[k]}` ({len(lap_by_case[k])}): " + ", ".join(sorted(lap_by_case[k])))
    (args.out / "reach_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
