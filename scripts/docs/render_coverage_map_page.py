# SPDX-License-Identifier: GPL-3.0-or-later
"""Render or verify the public MkDocs port coverage map (`#142`, `#174`).

Builds `docs/benchmarks/coverage-map.md` deterministically from
`benchmarks/dashboard/port_manifest.toml`, `notes/audit/defaults_diff_26_4_1.md`
tables, `benchmarks/fortran_mesa/math_hex_vectors/mesa_26_4_1_math_vectors.txt`,
`benchmarks/history_harness/{case_manifest,known_gaps}.toml`, and
`notes/audit/coverage-map.md` (without volatile per-merge JSON snapshots).
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Any

from flexmesa.testing import (
    _dashboard_collect,
    _dashboard_render,
    port_manifest,
)

try:
    import _common
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _common

GENERATED_COMMENT = _common.generated_comment(
    "scripts/docs/render_coverage_map_page.py", "#142, #174"
)
DEFAULT_OUTPUT = pathlib.Path("docs") / "benchmarks" / "coverage-map.md"
_PRE_AUDIT_DEFAULT_KEYS = ("PORTED", "PARTIAL", "STUB", "ABSENT")


def _count_math_vectors(repo_root: pathlib.Path) -> int:
    vec_path = (
        repo_root
        / "benchmarks"
        / "fortran_mesa"
        / "math_hex_vectors"
        / "mesa_26_4_1_math_vectors.txt"
    )
    if not vec_path.is_file():
        return 0
    return sum(
        1
        for line in vec_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "["))
    )


def _parse_pre_audit_coverage(repo_root: pathlib.Path) -> dict[str, int]:
    audit_path = repo_root / "notes" / "audit" / "coverage-map.md"
    counts = dict.fromkeys(_PRE_AUDIT_DEFAULT_KEYS, 0)
    counts["denom_non_plot"] = 0
    if not audit_path.is_file():
        return counts

    in_totals = False
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 8. Totals"):
            in_totals = True
            continue
        if in_totals and line.startswith("## "):
            break
        if not in_totals or not line.startswith("|"):
            continue
        if "share excluding plotting" in line:
            match = re.search(r"excluding plotting\s*\((\d+)\)", line)
            if match:
                counts["denom_non_plot"] = int(match.group(1))
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) == 4 and cols[0] in _PRE_AUDIT_DEFAULT_KEYS:
            if cols[1].isdigit():
                counts[cols[0]] = int(cols[1])
            if cols[0] == "ABSENT":
                raw_non_plot = cols[3].split("=")[0].strip()
                if raw_non_plot.isdigit():
                    counts["ABSENT"] = int(raw_non_plot)

    if counts["denom_non_plot"] == 0:
        counts["denom_non_plot"] = sum(
            counts[k] for k in _PRE_AUDIT_DEFAULT_KEYS
        )
    return counts


def _pct_rounded(part: int, whole: int) -> int:
    return round(100.0 * part / whole) if whole else 0


def render_coverage_map_markdown(repo_root: pathlib.Path) -> str:
    """Render deterministic Markdown for `docs/benchmarks/coverage-map.md`.

    Args:
        repo_root: Root directory of the `FlexMESA` repository checkout.

    Returns:
        Deterministic UTF-8 Markdown document ending with a newline.
    """
    manifest = port_manifest.load_port_manifest(
        repo_root / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    mod_cov = _dashboard_collect.collect_module_coverage(manifest, repo_root)
    modules: list[dict[str, Any]] = mod_cov["rows"]
    totals = _dashboard_render._sum_rows(modules)
    core = _dashboard_render._sum_rows(
        [r for r in modules if r["module"] in port_manifest.CORE_ENGINE_MODULES]
    )

    hh = _dashboard_collect.collect_harness_summary(repo_root)
    hh_tot = _dashboard_render._summarize_harness(hh["by_suite"])
    numerics = _dashboard_collect.collect_numerics_and_module_verification(
        repo_root
    )
    math_vector_count = _count_math_vectors(repo_root)
    other_suites_total = sum(
        1 for item in numerics["modules"] if item["module"] != "math"
    )
    other_suites_verified = sum(
        1
        for item in numerics["modules"]
        if item["module"] != "math"
        and str(item["exact_strict_compared"]) not in {"0", "not yet measured"}
    )

    controls = _dashboard_collect.collect_controls_summary(repo_root, None)
    star_ctrl = next(c for c in controls if c["namelist"] == "star &controls")
    job_ctrl = next(c for c in controls if c["namelist"] == "star &star_job")
    star_pct_unimp = (
        100.0 * star_ctrl["unimplemented"] / star_ctrl["upstream_scalars"]
    )
    job_pct_unimp = (
        100.0 * job_ctrl["unimplemented"] / job_ctrl["upstream_scalars"]
    )

    pre = _parse_pre_audit_coverage(repo_root)
    pre_den = pre["denom_non_plot"]
    bundled_loc = sum(
        int(r["upstream_lines"])
        for r in modules
        if r["module"] in port_manifest.BUNDLED_MODULES
    )
    plot_counts = mod_cov["plotting_panels"]

    c_loc = f"{core['upstream_lines']:,}"
    c_unver = (
        f"{core['ported_unverified']:,} "
        f"({100.0 * core['ported_unverified'] / core['upstream_lines']:.1f}%)"
        if core["upstream_lines"]
        else f"{core['ported_unverified']:,} (0.0%)"
    )
    c_ver = f"{core['verified_total']:,} ({core['pct_verified']:.1f}%)"
    t_loc = f"{totals['upstream_lines']:,}"
    t_unver_pct = (
        100.0 * totals["ported_unverified"] / totals["upstream_lines"]
        if totals["upstream_lines"]
        else 0.0
    )
    t_unver = f"{totals['ported_unverified']:,} ({t_unver_pct:.1f}%)"
    t_ver = f"{totals['verified_total']:,} ({totals['pct_verified']:.1f}%)"
    s_unimp = (
        f"{star_ctrl['unimplemented']:,} / {star_ctrl['upstream_scalars']:,}"
    )
    j_unimp = (
        f"{job_ctrl['unimplemented']:,} / {job_ctrl['upstream_scalars']:,}"
    )

    lines: list[str] = [
        GENERATED_COMMENT,
        "# Upstream Port Coverage Map (MESA `26.4.1`)",
        "",
        "Routine- and line-level port tracking in "
        "`benchmarks/dashboard/port_manifest.toml` measures `FlexMESA` against "
        f"**MESA release `{manifest.upstream_version}`** (commit "
        f"`{manifest.upstream_commit[:7]}`, archive MD5 "
        f"`{manifest.archive_md5}`), reconciled below with the pre-rebaseline "
        "structural audit in [`notes/audit/coverage-map.md`]"
        f"({_common.GH_BLOB}/notes/audit/coverage-map.md). Live per-commit "
        "Rust/Python line counts and test coverage appear in the "
        "[Live Progress & Verification Dashboard](progress-dashboard.md).",
        "",
        "Progress is tracked by **upstream MESA `26.4.1` lines by manifest "
        "status** and by routine-level evidence (`notes/audit/"
        "routine_table.md`), never by written Rust or Python lines "
        "(`DECISIONS #24`, `#174`). A file's status is raised above "
        "`ported_unverified` only with committed MESA `26.4.1` vector or "
        "history evidence in a pull request.",
        "",
        "## Headline Coverage Reconciliation",
        "",
        "| Denominator Scope | Upstream Lines | Ported (Unverified) | "
        "Compared Against MESA `26.4.1` | Notes |",
        "| :--- | ---: | ---: | ---: | :--- |",
        (
            f"| Core engines (`star` + `star_data` + `binary` + `astero`) | "
            f"{c_loc} | {c_unver} | {c_ver} | "
            f"Excludes stub (`{core['stub']:,}`) and partial "
            f"(`{core['partial']:,}`) lines |"
        ),
        (
            f"| All `{len(modules)}` MESA `26.4.1` modules (incl. `gyre`, "
            "`adipls`, `stella`, `plotting`) | "
            f"{t_loc} | {t_unver} | {t_ver} | "
            "Full `v0.1.0` target denominator (`notes/SCOPE.md`) |"
        ),
        (
            "| Pre-rebaseline non-plotting `star`+`binary`+`astero` audit "
            f"(`0b2e6bb`) | {pre_den:,} | {pre['PORTED']:,} `PORTED` "
            f"({_pct_rounded(pre['PORTED'], pre_den)}%) / "
            f"{pre['PARTIAL']:,} `PARTIAL` "
            f"({_pct_rounded(pre['PARTIAL'], pre_den)}%) | "
            "unverified pre-audit | "
            f"`{_pct_rounded(pre['STUB'], pre_den)}%` `STUB` "
            f"({pre['STUB']:,} LOC), "
            f"`{_pct_rounded(pre['ABSENT'], pre_den)}%` `ABSENT` "
            f"({pre['ABSENT']:,} LOC) |"
        ),
        (
            "| `math` strict-mode scalar vectors (`#165`) | "
            f"{math_vector_count:,} vectors | "
            f"{numerics['strict_kernels_bit_identical']} | "
            "None against MESA: "
            f"{numerics['strict_cross_machine_identical']} | "
            f"{other_suites_verified} / {other_suites_total} remaining module "
            "suites at `verified` status in `port_manifest.toml` |"
        ),
        (
            f"| End-to-end `test_suite` cases | {hh_tot['total']} cases | "
            f"{hh_tot['total']} in manifest | "
            f"**{hh_tot['passing']} / {hh_tot['total']}** passing | "
            f"{hh_tot['declared_gaps']} declared gaps, "
            f"{hh_tot['no_reference_yet']} `no_reference_yet` |"
        ),
        "",
        f"## Per-Module Port Status (`{len(modules)}` MESA `26.4.1` Modules)",
        "",
        "| Module | Upstream LOC | Absent | Stub | Partial | "
        "Ported (Unverified) | LOC at `verified` Status | % Ported | "
        "% at `verified` Status |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in modules:
        lines.append(
            f"| `{row['module']}` | {row['upstream_lines']:,} | "
            f"{row['absent']:,} | {row['stub']:,} | {row['partial']:,} | "
            f"{row['ported_unverified']:,} | {row['verified_total']:,} | "
            f"{row['pct_ported']:.1f}% | {row['pct_verified']:.1f}% |"
        )

    lines.extend(
        [
            (
                f"| **Core Engine Subtotal** | **{c_loc}** | "
                f"**{core['absent']:,}** | **{core['stub']:,}** | "
                f"**{core['partial']:,}** | "
                f"**{core['ported_unverified']:,}** | "
                f"**{core['verified_total']:,}** | "
                f"**{core['pct_ported']:.1f}%** | "
                f"**{core['pct_verified']:.1f}%** |"
            ),
            (
                f"| **All {len(modules)} Modules Total** | **{t_loc}** | "
                f"**{totals['absent']:,}** | **{totals['stub']:,}** | "
                f"**{totals['partial']:,}** | "
                f"**{totals['ported_unverified']:,}** | "
                f"**{totals['verified_total']:,}** | "
                f"**{totals['pct_ported']:.1f}%** | "
                f"**{totals['pct_verified']:.1f}%** |"
            ),
            "",
            "## Namelist Controls & Defaults Alignment",
            "",
            "Compared against MESA `26.4.1` `.defaults` schemas, "
            f"**{s_unimp}** `&controls` keys (`{star_pct_unimp:.1f}%`) "
            f"and **{j_unimp}** `&star_job` keys (`{job_pct_unimp:.1f}%`) "
            "are currently unimplemented (details in "
            "[`notes/audit/defaults_diff_26_4_1.md`]"
            f"({_common.GH_BLOB}/notes/audit/defaults_diff_26_4_1.md); "
            "loud rejection of unsupported keys is tracked in `#147` and "
            "default alignment in `#149`):",
            "",
            "| Namelist | Upstream Scalars | Matched in FlexMESA | "
            "Defaults Agree | Value Mismatch | Disabled (`none`/limit) | "
            "Non-Scalar Rust | Unimplemented |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for ctrl in controls:
        lines.append(
            f"| `{ctrl['namelist']}` | {ctrl['upstream_scalars']:,} | "
            f"{ctrl['matched_in_flexmesa']:,} | {ctrl['defaults_agree']:,} | "
            f"{ctrl['value_mismatch']:,} | {ctrl['disabled_limit_none']:,} | "
            f"{ctrl.get('skipped_unparseable_rust', 0):,} | "
            f"{ctrl['unimplemented']:,} |"
        )

    lines.extend(
        [
            "",
            "## Key Structural & Physics Gaps Behind "
            f"`{hh_tot['passing']} / {hh_tot['total']}` Acceptance Passes",
            "",
            "- **Model Construction & Relaxation (`#154`, `#150`)**: "
            "Multi-step `relax.f90` and `create_initial_model.f90` operations "
            "(`#154`) and upstream binary `.photo` compatibility (`#150`) "
            "remain unported; current `.photo` byte round-trips validate "
            "internal continuation consistency only.",
            "- **Silent Inlist Drops & Default Mismatches (`#147`, `#149`)**: "
            "Existing parsers silently ignore unimplemented namelist keys or "
            "map unknown option strings (`mlt_option`, `atm_option`, "
            "`screening_mode`) onto defaults instead of failing loudly.",
            "- **In-Library Per-Case Extras Removed (`#148`)**: In-library "
            "`Tier1..Tier5` test-case extras and fabricated ZAMS/gravity/"
            "opacity overrides have been deleted from library crates; upstream "
            "`run_star_extras.f90` and `run_binary_extras.f90` callbacks now "
            "live strictly in `tests/case_extras/`.",
            "- **Solver Control, Remeshing, Mixing & Transport "
            "(`#152`-`#162`)**: Upstream Newton retry scheduling, "
            "`mesh_adjust.f90` zoning, convective premixing, predictive "
            "mixing, Burgers diffusion, monochromatic radiative levitation "
            "(`op_radacc`), shellular rotation variables, and coupled binary "
            "stepping remain partial or unverified.",
            "- **Bundled Codes & Plotting (`#170`-`#173`)**: Upstream `GYRE`, "
            f"`ADIPLS`, and `STELLA` (`{bundled_loc:,}` combined upstream LOC) "
            "are the bundled codes: `GYRE` and `ADIPLS` are partly ported to "
            "Safe Rust (`flexmesa-gyre`, `flexmesa-adipls`; per-routine "
            "status in `notes/audit/routine_table.md`), `STELLA` follows "
            "`v0.1.0` (`#172`), and `flexmesa.plotting` provides "
            f"`{plot_counts['flexmesa_plot_builders']}` Matplotlib builders "
            f"against `{plot_counts['upstream_panel_modules']}` upstream "
            "`PGSTAR`/`PGBINARY` panel modules.",
            "- **Derived JAX Engine Twins (`#166`)**: Python/JAX kernels under "
            "`src/flexmesa/{star,binary,astero}` are frozen from new engine "
            "feature additions until canonical Safe Rust `strict` mode "
            "clears the MESA `26.4.1` trajectory harness.",
            "",
            "## Reading `port_manifest.toml` in Python",
            "",
            "```python",
            "import collections",
            "from flexmesa.testing import case_manifest, port_manifest",
            "",
            "m = port_manifest.load_port_manifest(",
            '    case_manifest.repo_root() / "benchmarks" / "dashboard" '
            '/ "port_manifest.toml"',
            ")",
            'assert m.upstream_version == "r26.4.1"',
            "counts = collections.Counter(e.status.value for e in m.entries)",
            'print("Port manifest statuses:", dict(sorted(counts.items())))',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `docs/benchmarks/coverage-map.md`.

    Args:
        argv: Optional argument list overriding `sys.argv[1:]`.

    Returns:
        Exit status code (`0` when rendered or matching under `--check`).
    """
    return _common.run_render_cli(
        argv=argv,
        description="Render or check docs/benchmarks/coverage-map.md.",
        default_rel_output=DEFAULT_OUTPUT,
        render_fn=render_coverage_map_markdown,
        regen_cmd="python scripts/docs/render_coverage_map_page.py",
        check_ok_msg=(
            "OK: {out_path} matches port_manifest.toml, harness inputs, and "
            "audit defaults."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
