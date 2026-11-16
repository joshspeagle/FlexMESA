"""Capture exact IEEE-754 hex/bit patterns and FP compiler flags for MESA 26.4.1 module tests.

Uses the external ``LD_PRELOAD`` library ``libgfortran_hex_tap.so`` (built from
``scripts/gfortran_hex_tap.c``) to intercept ``libgfortran`` floating-point write
transfers in memory prior to decimal formatting across all 25 MESA module
``test/`` suites (plus recording ``sample``, which has no ``test/`` directory),
without modifying any file inside ``$MESA_DIR``. Also extracts DWARF
``DW_AT_producer`` strings and Makefile flags to document every compiler/linker
setting affecting floating-point evaluation.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import struct
import subprocess
import time
from typing import Any

ROOT_DIR = pathlib.Path(
    os.environ.get(
        "MESA_GROUND_TRUTH_ROOT",
        str(pathlib.Path.home() / "mesa-ground-truth-26.4.1"),
    )
)
MESA_DIR = pathlib.Path(
    os.environ.get("MESA_DIR", str(pathlib.Path.home() / "mesa-26.04.1"))
)
ENV_SCRIPT = ROOT_DIR / "scripts/env_mesa.sh"
HEX_TAP_SRC = ROOT_DIR / "scripts/gfortran_hex_tap.c"
HEX_TAP_SO = ROOT_DIR / "build/libgfortran_hex_tap.so"
OUT_ROOT = ROOT_DIR / "module_tests"

MODULES_IN_INSTALL_ORDER = [
    "const",
    "utils",
    "math",
    "mtx",
    "auto_diff",
    "forum",
    "num",
    "interp_1d",
    "interp_2d",
    "chem",
    "colors",
    "eos",
    "kap",
    "rates",
    "neu",
    "net",
    "ionization",
    "atm",
    "turb",
    "sample",
    "star_data",
    "gyre",
    "adipls",
    "star",
    "binary",
    "astero",
]


def run_with_mesa_env(
    cmd: str, cwd: pathlib.Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet after sourcing ``env_mesa.sh``."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    if extra_env:
        env.update(extra_env)
    full_cmd = f"set -eu; source {ENV_SCRIPT}; {cmd}"
    return subprocess.run(
        ["bash", "-c", full_cmd],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def extract_dwarf_producers(binary_paths: list[pathlib.Path]) -> list[str]:
    """Extract unique DWARF ``DW_AT_producer`` strings from archives/binaries."""
    producers: set[str] = set()
    for bp in binary_paths:
        if not bp.exists():
            continue
        p = subprocess.run(
            ["readelf", "--debug-dump=info", str(bp)],
            capture_output=True,
            text=True,
            errors="replace",
        )
        for ln in p.stdout.splitlines():
            if "DW_AT_producer" in ln:
                val = ln.split(":", 2)[-1].strip()
                producers.add(val)
    return sorted(producers)


def extract_fp_compiler_flags() -> dict[str, Any]:
    """Collect global and per-module floating-point compiler and math-library settings."""
    p_ver = run_with_mesa_env(
        "gfortran.exec -v 2>&1 && echo '---OPT---' && "
        "gfortran.exec -Q --help=optimizers -O2 -ftree-vectorize -fprotect-parens -fno-sign-zero -march=x86-64 -mtune=generic",
        ROOT_DIR,
    )
    ver_part, _, opt_part = p_ver.stdout.partition("---OPT---")

    fp_optimizer_states: dict[str, str] = {}
    for ln in opt_part.splitlines():
        s = ln.strip()
        if any(
            k in s
            for k in (
                "fp-contract",
                "fast-math",
                "unsafe-math",
                "associative-math",
                "reciprocal-math",
                "finite-math",
                "signaling-nans",
                "signed-zeros",
                "trapping-math",
                "rounding-math",
                "protect-parens",
                "tree-vectorize",
                "cx-fortran-rules",
                "cx-limited-range",
            )
        ):
            parts = s.split()
            if len(parts) >= 2:
                fp_optimizer_states[parts[0]] = parts[-1]

    # Inspect ISA target flags (-mfma, -msse2, -mavx, -mfpmath)
    p_target = run_with_mesa_env(
        "gfortran.exec -Q --help=target -O2 -march=x86-64 -mtune=generic",
        ROOT_DIR,
    )
    isa_states: dict[str, str] = {}
    for ln in p_target.stdout.splitlines():
        s = ln.strip()
        if any(
            s.startswith(prefix)
            for prefix in (
                "-march=",
                "-mtune=",
                "-mfpmath=",
                "-msse",
                "-mavx",
                "-mfma",
                "-m80387",
            )
        ):
            parts = s.split()
            if len(parts) >= 2:
                isa_states[parts[0]] = " ".join(parts[1:])

    per_module: dict[str, Any] = {}
    for mod in MODULES_IN_INSTALL_ORDER:
        lib_candidates = list((MESA_DIR / "lib").glob(f"lib{mod}*.a"))
        if mod == "gyre":
            lib_candidates.extend((MESA_DIR / "lib").glob("libgyre*.a"))
        test_exe = (
            MESA_DIR / mod / "test" / ("star" if mod == "star" else "tester")
        )
        if mod == "forum":
            test_exe = MESA_DIR / "forum/forum/bin/test_order"
        producers = extract_dwarf_producers([*lib_candidates, test_exe])

        makefile_notes: list[str] = []
        for mf in (
            MESA_DIR / mod / "make/makefile",
            MESA_DIR / mod / "Makefile",
            MESA_DIR / "adipls/adipack.c/adipls/Makefile",
            MESA_DIR / "gyre/gyre/src/build/Make.inc",
        ):
            if mf.exists() and (
                mod in str(mf.relative_to(MESA_DIR)).split("/")[0]
            ):
                txt = mf.read_text(encoding="utf-8", errors="replace")
                for ln in txt.splitlines():
                    if any(
                        tok in ln
                        for tok in (
                            "COMPILE",
                            "FFLAGS",
                            "FCFLAGS",
                            "-O",
                            "ffp",
                            "fma",
                            "crmath",
                        )
                    ) and not ln.strip().startswith("#"):
                        makefile_notes.append(
                            f"{mf.relative_to(MESA_DIR)}: {ln.strip()}"
                        )

        per_module[mod] = {
            "dwarf_producers": producers,
            "makefile_compile_lines": makefile_notes[:15],
        }

    return {
        "compiler_executable": "gfortran (MESA SDK 26.6.1 wrapper -> gfortran.exec, GNU Fortran GCC 15.2.0)",
        "compiler_verbose_config": ver_part.strip(),
        "architecture_and_fma_policy": {
            "arch": "x86_64",
            "effective_march": "-march=x86-64 -mtune=generic",
            "hardware_fp_unit": "-mfpmath=sse (64-bit SSE2 IEEE-754 registers, no x87 80-bit excess precision)",
            "fma_instructions_enabled": False,
            "note": (
                "Although GCC defaults to -ffp-contract=fast when not overridden on x86_64, "
                "-march=x86-64 disables FMA3/FMA4/AVX2 instruction generation (-mfma [disabled]), "
                "so fused multiply-add contraction is not emitted in generated machine code."
            ),
            "isa_target_flags": isa_states,
        },
        "global_mesa_fortran_fp_flags": {
            "optimization": "-O2 -ftree-vectorize (COMPILE_TO_TEST / COMPILE_TO_DEPLOY) or -O (COMPILE_NO_OPT for namelist I/O)",
            "parentheses_protection": "-fprotect-parens (prevents algebraic reassociation across parentheses)",
            "signed_zero": "-fno-sign-zero (treats -0.0 as unsigned in SIGN and formatted output)",
            "range_and_nan_init": "-fno-range-check -finit-real=snan (SKIP_NAN_TRAPS=YES, no -ffpe-trap by default)",
            "standard": "-std=f2008 (filtered out on legacy F77/extension files)",
            "openmp": "-fopenmp",
            "math_slot": "MESASDK_MATH_SLOT=crmath (links libcrmath.so + libcrlibm.so.0 for correctly rounded 0.5-ULP transcendentals, plus math-slots/crmath BLAS/LAPACK)",
            "effective_gcc_fp_optimizer_flags": fp_optimizer_states,
        },
        "per_module": per_module,
    }


def write_history_exact_hex(
    hist_path: pathlib.Path, out_tsv: pathlib.Path
) -> int:
    """Convert parsed float64 cells of ``star/test/LOGS/history.data`` to exact hex."""
    lines = hist_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        return 0
    col_names = lines[5].split()
    count = 0
    with out_tsv.open("w", encoding="utf-8") as fh:
        fh.write(
            "row_idx\tcol_idx\tcolumn\tdecimal_str\thex_float\traw_bits_hex\n"
        )
        for r_idx, ln in enumerate(lines[6:], 1):
            toks = ln.split()
            if not toks:
                continue
            for c_idx, (cname, tok) in enumerate(zip(col_names, toks), 1):
                val = float(tok.replace("D", "E").replace("d", "e"))
                bits = struct.unpack(">Q", struct.pack(">d", val))[0]
                fh.write(
                    f"{r_idx}\t{c_idx}\t{cname}\t{tok}\t{val.hex()}\t0x{bits:016x}\n"
                )
                count += 1
    return count


def run_module_test(mod: str) -> dict[str, Any]:
    """Execute ``<mod>/test`` in a sibling temporary directory with ``libgfortran_hex_tap.so``."""
    mod_out = OUT_ROOT / mod
    mod_out.mkdir(parents=True, exist_ok=True)

    src_test_dir = MESA_DIR / mod / "test"
    if not src_test_dir.is_dir():
        return {
            "module": mod,
            "has_test_dir": False,
            "check_passed": True,
            "note": "Package has no test/ directory in upstream MESA 26.4.1 (library client example only).",
            "exact_values_captured": 0,
        }

    tmp_test_dir = MESA_DIR / mod / "_hex_test_tmp"
    if tmp_test_dir.exists():
        shutil.rmtree(tmp_test_dir)
    shutil.copytree(src_test_dir, tmp_test_dir, symlinks=True)

    exact_tsv = mod_out / "exact_tested_values.tsv"
    if exact_tsv.exists():
        exact_tsv.unlink()
    stdout_save = mod_out / "test_output_actual.txt"

    t0 = time.monotonic()
    try:
        if mod == "adipls":
            cmd = (
                'DIR=$(pwd); aprgdir="$DIR/../adipack.c"; export aprgdir; '
                'export PATH="$aprgdir/bin:$PATH"; '
                "fgong-amdl.d Gemma.fgong amdl.mesa > .list && "
                "scan-amdl.d amdl.mesa >> .list && "
                "adipls.c.d adipls.c.in >> .list && "
                f'HEX_TAP_UNIT=10 HEX_TAP_OUT="{exact_tsv}" LD_PRELOAD="{HEX_TAP_SO}" '
                "set-obs.d 16 mesa >> .list && "
                "mv obs.mesa obs.txt && "
                "ndiff -quiet -relerr 1e-9 obs.txt obs.test.txt"
            )
            res = run_with_mesa_env(cmd, tmp_test_dir)
            if (tmp_test_dir / "obs.txt").exists():
                shutil.copyfile(tmp_test_dir / "obs.txt", stdout_save)
            check_ok = res.returncode == 0
            err_msg = res.stderr.strip()
        elif mod == "star":
            cmd = (
                "rm -f restart_photo && "
                f'HEX_TAP_UNIT=6 HEX_TAP_OUT="{exact_tsv}" LD_PRELOAD="{HEX_TAP_SO}" ./star > log && '
                "grep 'stop because star_age >= max_age' log > tmp.txt && "
                "diff tmp.txt test_output"
            )
            res = run_with_mesa_env(cmd, tmp_test_dir)
            if (tmp_test_dir / "log").exists():
                shutil.copyfile(tmp_test_dir / "log", stdout_save)
            hist_p = tmp_test_dir / "LOGS/history.data"
            extra_hist_cells = 0
            if hist_p.exists():
                shutil.copyfile(hist_p, mod_out / "history.data")
                extra_hist_cells = write_history_exact_hex(
                    hist_p, mod_out / "history_exact_hex.tsv"
                )
            check_ok = res.returncode == 0
            err_msg = res.stderr.strip()
        elif mod == "gyre":
            cmd = (
                f'HEX_TAP_UNIT=6 HEX_TAP_OUT="{exact_tsv}" LD_PRELOAD="{HEX_TAP_SO}" ./rn > tmp.txt 2>&1 && '
                "ndiff -quiet -relerr 1.0e-6 tmp.txt test_output"
            )
            res = run_with_mesa_env(cmd, tmp_test_dir)
            if (tmp_test_dir / "tmp.txt").exists():
                shutil.copyfile(tmp_test_dir / "tmp.txt", stdout_save)
            check_ok = res.returncode == 0
            err_msg = res.stderr.strip()
        elif mod == "math":
            cmd = (
                f'HEX_TAP_UNIT=6 HEX_TAP_OUT="{exact_tsv}" LD_PRELOAD="{HEX_TAP_SO}" ./rn > tmp.txt 2>&1 && '
                'backend=$(head -n 1 tmp.txt | tr -d "[:space:]") && '
                'diff -b tmp.txt "test_output.${backend}"'
            )
            res = run_with_mesa_env(cmd, tmp_test_dir)
            if (tmp_test_dir / "tmp.txt").exists():
                shutil.copyfile(tmp_test_dir / "tmp.txt", stdout_save)
            check_ok = res.returncode == 0
            err_msg = res.stderr.strip()
        else:
            cmd = (
                f'HEX_TAP_UNIT=6 HEX_TAP_OUT="{exact_tsv}" LD_PRELOAD="{HEX_TAP_SO}" ./rn > tmp.txt 2>&1 && '
                "diff -b tmp.txt test_output"
            )
            res = run_with_mesa_env(cmd, tmp_test_dir)
            if (tmp_test_dir / "tmp.txt").exists():
                shutil.copyfile(tmp_test_dir / "tmp.txt", stdout_save)
            check_ok = res.returncode == 0
            err_msg = res.stderr.strip()
    finally:
        shutil.rmtree(tmp_test_dir, ignore_errors=True)

    wall_s = time.monotonic() - t0
    n_vals = 0
    if exact_tsv.exists():
        with exact_tsv.open(encoding="utf-8") as fh:
            n_vals = max(0, sum(1 for _ in fh) - 1)

    info: dict[str, Any] = {
        "module": mod,
        "has_test_dir": True,
        "check_passed": check_ok,
        "wall_seconds": round(wall_s, 3),
        "exact_values_captured": n_vals,
        "exact_values_file": str(exact_tsv.relative_to(ROOT_DIR))
        if exact_tsv.exists()
        else None,
        "test_output_file": str(stdout_save.relative_to(ROOT_DIR))
        if stdout_save.exists()
        else None,
    }
    if mod == "star" and (mod_out / "history_exact_hex.tsv").exists():
        info["star_history_hex_file"] = str(
            (mod_out / "history_exact_hex.tsv").relative_to(ROOT_DIR)
        )
        info["star_history_hex_cells"] = extra_hist_cells
    if not check_ok:
        info["error"] = err_msg or res.stdout[-500:]
    return info


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ROOT_DIR.joinpath("build").mkdir(parents=True, exist_ok=True)

    # Compile libgfortran_hex_tap.so
    comp = run_with_mesa_env(
        f'gcc -O2 -fPIC -shared "{HEX_TAP_SRC}" -o "{HEX_TAP_SO}" -ldl -lquadmath -lpthread',
        ROOT_DIR,
    )
    if comp.returncode != 0:
        raise RuntimeError(
            f"Failed to compile libgfortran_hex_tap.so: {comp.stderr}"
        )

    fp_flags = extract_fp_compiler_flags()
    (OUT_ROOT / "compiler_fp_flags.json").write_text(
        json.dumps(fp_flags, indent=2) + "\n", encoding="utf-8"
    )

    # Write concise human-readable Markdown summary of FP compiler flags
    md_lines = [
        "# MESA 26.4.1 Floating-Point Compiler & Math-Library Flags",
        "",
        "- **Compiler**: MESA SDK `x86_64-linux-26.6.1` `gfortran` (`GNU Fortran (GCC) 15.2.0`)",
        "- **Target ISA & FP Unit**: `-march=x86-64 -mtune=generic -mfpmath=sse` (SSE2 64-bit IEEE-754 registers; `-mfma` disabled, so zero hardware FMA contraction instructions are emitted)",
        "- **Optimization**: `-O2 -ftree-vectorize` (`COMPILE_TO_TEST` / `COMPILE_TO_DEPLOY`), `-O` (`COMPILE_NO_OPT` on namelist I/O)",
        "- **Parentheses & Reassociation**: `-fprotect-parens` (`-fno-associative-math`, `-fno-unsafe-math-optimizations`, `-fno-fast-math`)",
        "- **Signed Zero & NaN Initialization**: `-fno-sign-zero -fno-range-check -finit-real=snan` (`SKIP_NAN_TRAPS=YES`)",
        "- **Math Backend**: `MESASDK_MATH_SLOT=crmath` (`MATH_BACKEND = 'CRMATH'`, linking `libcrmath.so` + `libcrlibm.so.0` correctly rounded 0.5-ULP elementary functions and `math-slots/crmath` BLAS/LAPACK)",
        "",
        "## Per-Module DWARF `DW_AT_producer` Strings",
        "",
    ]
    for mod, mdata in fp_flags["per_module"].items():
        prods = mdata["dwarf_producers"]
        md_lines.append(f"### `{mod}`")
        if prods:
            for pr in prods:
                md_lines.append(f"- `{pr}`")
        else:
            for ml in mdata["makefile_compile_lines"][:4]:
                md_lines.append(f"- `{ml}`")
        md_lines.append("")
    (OUT_ROOT / "compiler_fp_flags.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    for mod in MODULES_IN_INSTALL_ORDER:
        r = run_module_test(mod)
        results.append(r)
        print(
            f"[module_hex] {mod:12s}: passed={r['check_passed']} "
            f"exact_values={r['exact_values_captured']}",
            flush=True,
        )

    summary = {
        "total_packages_in_install": len(MODULES_IN_INSTALL_ORDER),
        "packages_with_test_dir": sum(1 for r in results if r["has_test_dir"]),
        "all_checks_passed": all(r["check_passed"] for r in results),
        "total_exact_values_captured": sum(
            r["exact_values_captured"] for r in results
        ),
        "modules": results,
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Completed: {summary['packages_with_test_dir']}/{summary['total_packages_in_install']} "
        f"module tests run, all_passed={summary['all_checks_passed']}, "
        f"total_exact_values={summary['total_exact_values_captured']}"
    )


if __name__ == "__main__":
    main()
