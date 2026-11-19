# SPDX-License-Identifier: LGPL-3.0-or-later
"""Render or verify the upstream ``PGSTAR``/``PGBINARY`` coverage page (`#173`).

Generates ``docs/benchmarks/pgstar-coverage.md`` deterministically from MESA
``26.4.1`` ``pgstar.defaults``, ``pgbinary.defaults``, and
``astero_pgstar.defaults`` paired with ``flexmesa.plotting.pgstar_coverage``.
"""

from __future__ import annotations

import pathlib
import sys

from flexmesa.data import fetch
from flexmesa.plotting import pgstar_coverage

try:
    import _common
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _common

GENERATED_COMMENT = _common.generated_comment(
    "scripts/docs/render_pgstar_coverage.py", "#173"
)
DEFAULT_OUTPUT = pathlib.Path("docs") / "benchmarks" / "pgstar-coverage.md"


def render_pgstar_coverage_markdown(repo_root: pathlib.Path) -> str:
    """Render deterministic Markdown for ``docs/benchmarks/pgstar-coverage.md``.

    Args:
        repo_root: Repository root path accepted for CLI renderer compatibility.

    Returns:
        Deterministic UTF-8 Markdown document ending with a newline.
    """
    del repo_root
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    report = pgstar_coverage.build_pgstar_coverage_report(mesa_dir)

    eq_fams = part_fams = abs_fams = eq_wins = part_wins = 0
    for p in report.panels:
        if p.status == "equivalent":
            eq_fams += 1
            eq_wins += p.window_count
        elif p.status == "partial":
            part_fams += 1
            part_wins += p.window_count
        else:
            abs_fams += 1

    lines: list[str] = [
        GENERATED_COMMENT,
        "# Upstream PGSTAR & PGBINARY Coverage Report (MESA `26.4.1`)",
        "",
        "Generated mechanically from MESA `26.4.1` "
        "`star/defaults/pgstar.defaults`, `binary/defaults/pgbinary.defaults`, "
        "and `astero/defaults/astero_pgstar.defaults` paired with "
        "`flexmesa.plotting.pgstar_coverage` (`#173`). All plots read strictly "
        "from saved `history.data`, `binary_history.data`, `profileN.data`, "
        "and MESA `26.4.1` `plot_info` files (`flexmesa plot compare` / "
        "`profile` / `kipp` / `binary` / `track` / `grid`).",
        "",
        "The status of each family is assigned by hand in "
        "`flexmesa.plotting.pgstar_coverage` from a reading of the upstream "
        "window source. The generator checks only that every upstream window "
        "family has a row and that each named builder exists. No figure has "
        "been compared with a PGSTAR or PGBINARY rendering, so `equivalent` "
        "means the builder is intended to draw the same quantities, and "
        "`partial` rows say what is known to be missing.",
        "",
        "## Headline Summary",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        (
            "| Panel families catalogued (`star` / `binary` / `astero`) | "
            f"`{len(report.panels)}` |"
        ),
        (
            f"| Upstream `*_win_flag` window instances | "
            f"`{report.total_windows}` |"
        ),
        (
            "| Families whose builder is classed `equivalent` | "
            f"`{eq_fams}` (`{eq_wins}` windows) |"
        ),
        (
            "| Families with partial builder coverage | "
            f"`{part_fams}` (`{part_wins}` windows) |"
        ),
        (
            "| Families absent (pending engine/plotting increments) | "
            f"`{abs_fams}` |"
        ),
        (
            "| Parsed `.defaults` keys (`pgstar` / `pgbinary` / "
            "`astero_pgstar`) | "
            f"`{report.source_counts['pgstar.defaults']:,}` / "
            f"`{report.source_counts['pgbinary.defaults']:,}` / "
            f"`{report.source_counts['astero_pgstar.defaults']:,}` |"
        ),
        "",
        "## Panel Family Coverage Matrix",
        "",
        (
            "| Namelist | Panel Family | Windows | Status | "
            "`flexmesa.plotting` Equivalent | Issue | Notes |"
        ),
        "| :--- | :--- | ---: | :--- | :--- | :--- | :--- |",
    ]
    for p in report.panels:
        builders = (
            ", ".join(f"`{b.strip()}`" for b in p.flexmesa_builders.split(","))
            if p.flexmesa_builders != "-"
            else "-"
        )
        lines.append(
            f"| `{p.namelist}` | `{p.family}` | {p.window_count} | "
            f"`{p.status}` | {builders} | {p.issue} | {p.notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``docs/benchmarks/pgstar-coverage.md``.

    Args:
        argv: Optional CLI argument list overriding ``sys.argv[1:]``.

    Returns:
        Exit status code (`0` when rendered or matching under ``--check``).
    """
    return _common.run_render_cli(
        argv=argv,
        description="Render or check docs/benchmarks/pgstar-coverage.md.",
        default_rel_output=DEFAULT_OUTPUT,
        render_fn=render_pgstar_coverage_markdown,
        regen_cmd="python scripts/docs/render_pgstar_coverage.py",
        check_ok_msg=(
            "OK: {out_path} matches MESA 26.4.1 pgstar defaults and catalog."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
