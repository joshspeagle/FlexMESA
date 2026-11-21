# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact, visual README view of the progress dashboard (`#174`).

The README shows badges, one headline, a status chart, a module map and two
short tables. Everything here is computed from the same data as the detailed
page (`docs/benchmarks/progress-dashboard.md`), which keeps the full tables.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from flexmesa.testing import case_manifest

REPO_URL = "https://github.com/joshspeagle/FlexMESA"
_DOCS_PAGE = f"{REPO_URL}/blob/main/docs/benchmarks/progress-dashboard.md"
_CASE_TABLE = f"{REPO_URL}/blob/main/notes/audit/case_table.md"

# Module groups, in display order. Every module of the port manifest must
# appear exactly once; `render_readme_dashboard` raises otherwise.
MODULE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Foundations",
        (
            "const",
            "utils",
            "math",
            "auto_diff",
            "mtx",
            "interp_1d",
            "interp_2d",
            "num",
        ),
    ),
    (
        "Microphysics",
        (
            "chem",
            "eos",
            "kap",
            "rates",
            "net",
            "neu",
            "atm",
            "turb",
            "ionization",
            "colors",
        ),
    ),
    ("Stellar evolution", ("star_data", "star", "binary", "astero")),
    ("Bundled codes and plotting", ("gyre", "adipls", "stella", "plotting")),
)

# Arrows point from a module to one that builds on it.
_EDGES: tuple[tuple[str, str], ...] = (
    ("const", "math"),
    ("math", "auto_diff"),
    ("utils", "mtx"),
    ("mtx", "num"),
    ("interp_1d", "interp_2d"),
    ("chem", "eos"),
    ("eos", "kap"),
    ("chem", "rates"),
    ("rates", "net"),
    ("eos", "net"),
    ("star_data", "star"),
    ("star", "binary"),
    ("star", "astero"),
    ("gyre", "astero"),
    ("adipls", "astero"),
)

_BAR_WIDTH = 20


def _badge(label: str, message: str, color: str, link: str) -> str:
    def esc(text: str) -> str:
        return quote(text.replace("-", "--").replace("_", "__"), safe="")

    url = f"https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color}"
    return f"[![{label}: {message}]({url})]({link})"


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _shares(row: dict[str, Any]) -> tuple[float, float, float]:
    """Return (evidence, ported, in progress) as percentages of upstream.

    `evidence` is lines with MESA `26.4.1` reference vectors or case-level
    evidence; `ported` is all ported lines including those; `in progress`
    adds lines whose file is partially ported or a stub.
    """
    up = int(row["upstream_lines"])
    evidence = int(row["verified_total"])
    ported = int(row["ported_total"])
    started = ported + int(row["partial"]) + int(row["stub"])
    return _pct(evidence, up), _pct(ported, up), _pct(started, up)


def _bar(row: dict[str, Any]) -> str:
    """Twenty-cell bar: full block ported, shaded in progress, light absent."""
    _, ported, started = _shares(row)
    n_ported = round(ported / 100.0 * _BAR_WIDTH)
    n_started = max(round(started / 100.0 * _BAR_WIDTH), n_ported)
    return (
        "█" * n_ported
        + "▒" * (n_started - n_ported)
        + "░" * (_BAR_WIDTH - n_started)
    )


def _css_class(row: dict[str, Any]) -> str:
    _, ported, started = _shares(row)
    if ported >= 50.0:
        return "ported"
    if started >= 50.0:
        return "partial"
    if started > 0.0:
        return "started"
    return "absent"


def _sum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "upstream_lines",
        "absent",
        "stub",
        "partial",
        "ported_unverified",
        "ported_total",
        "verified_total",
    )
    return {k: sum(int(r[k]) for r in rows) for k in keys}


def _table_row(label: str, row: dict[str, Any]) -> str:
    _, ported, started = _shares(row)
    return (
        f"| {label} | `{_bar(row)}` | {ported:.0f}% | {started:.0f}% | "
        f"{int(row['upstream_lines']):,} |"
    )


def _mermaid_pie(total: dict[str, Any]) -> list[str]:
    def k(key: str) -> str:
        return f"{int(total[key]) / 1000.0:.1f}"

    return [
        "```mermaid",
        "%%{init: {'themeVariables': {'pie1': '#1a7f37', 'pie2': '#7fc97f', "
        "'pie3': '#f2cc60', 'pie4': '#d0d7de', "
        "'pieSectionTextColor': '#000000', 'pieStrokeColor': '#57606a'}}}%%",
        "pie showData",
        "    title Engine upstream lines by status (thousands)",
        f'    "Ported, with MESA reference evidence" : {k("verified_total")}',
        f'    "Ported, not yet checked against MESA" : '
        f"{k('ported_unverified')}",
        f'    "In progress (file partly ported)" : '
        f"{(int(total['partial']) + int(total['stub'])) / 1000.0:.1f}",
        f'    "Not started" : {k("absent")}',
        "```",
    ]


def _mermaid_map(by_module: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["```mermaid", "flowchart LR"]
    for title, modules in MODULE_GROUPS:
        gid = title.replace(" ", "_")
        lines.append(f'    subgraph {gid}["{title}"]')
        for mod in modules:
            row = by_module[mod]
            _, ported, started = _shares(row)
            lines.append(
                f'        {mod}["{mod}<br/>{ported:.0f}% ported<br/>'
                f'{started:.0f}% started"]:::{_css_class(row)}'
            )
        lines.append("    end")
    lines.extend(f"    {a} --> {b}" for a, b in _EDGES)
    lines.extend(
        [
            "    Foundations --> Microphysics",
            "    Microphysics --> Stellar_evolution",
            "    classDef ported fill:#1a7f37,color:#ffffff,stroke:#57606a",
            "    classDef partial fill:#f2cc60,color:#000000,stroke:#57606a",
            "    classDef started fill:#fff8c5,color:#000000,stroke:#57606a",
            "    classDef absent fill:#d0d7de,color:#000000,stroke:#57606a",
            "```",
        ]
    )
    return lines


def render_readme_dashboard(
    data: dict[str, Any], head: dict[str, Any]
) -> list[str]:
    """Render the README dashboard body (without the block markers).

    Args:
        data: Dashboard data from `dashboard.build_dashboard_data`.
        head: Headline metrics from `_dashboard_render.summarize_headline`.

    Returns:
        Markdown lines.

    Raises:
        ValueError: If `MODULE_GROUPS` and the port manifest list different
            modules.
    """
    rows: list[dict[str, Any]] = data["module_coverage"]["rows"]
    by_module = {str(r["module"]): r for r in rows}
    grouped = [m for _, mods in MODULE_GROUPS for m in mods]
    if sorted(grouped) != sorted(by_module):
        raise ValueError(
            "README module groups do not match the port manifest: "
            f"{sorted(set(grouped) ^ set(by_module))}"
        )

    eng = head["engine_tot"]
    hh = head["hh_tot"]
    cov = data["coverage"]
    rs_pct = float(cov["rust"]["overall"]["percent"])
    py_pct = float(cov["python"]["overall"]["percent"])
    engine_rows = [
        by_module[m] for m in ("star_data", "star", "binary", "astero")
    ]
    engine_sum = _sum(engine_rows)
    engine_ported = int(eng["ported_total"])
    engine_up = int(eng["upstream_lines"])
    n_prs = len(data["github"]["open_pull_requests"])

    case_color = "1a7f37" if hh["passing"] == hh["total"] else "d4a72c"
    undeclared_color = "1a7f37" if hh["undeclared_failures"] == 0 else "cf222e"
    badges = " ".join(
        [
            _badge(
                "engine ported",
                f"{eng['pct_ported']:.1f}%",
                "2da44e",
                _DOCS_PAGE,
            ),
            _badge(
                "cases passing",
                f"{hh['passing']} / {hh['total']}",
                case_color,
                _CASE_TABLE,
            ),
            _badge(
                "undeclared failures",
                str(hh["undeclared_failures"]),
                undeclared_color,
                _CASE_TABLE,
            ),
            _badge("Rust coverage", f"{rs_pct:.1f}%", "0969da", _DOCS_PAGE),
            _badge("Python coverage", f"{py_pct:.1f}%", "0969da", _DOCS_PAGE),
            _badge(
                "packages closed",
                f"{head['closed_140']} / {head['total_140']}",
                "8250df",
                f"{REPO_URL}/issues/140",
            ),
        ]
    )

    lines: list[str] = [
        "Generated by `flexmesa dashboard render`; do not edit by hand. "
        f"Full tables: [progress dashboard]({_DOCS_PAGE}) and "
        f"[per-case table]({_CASE_TABLE}).",
        "",
        badges,
        "",
        f"**{eng['pct_ported']:.1f}% of the stellar-evolution engine is "
        f"ported** ({engine_ported / 1000.0:.1f}k of "
        f"{engine_up / 1000.0:.0f}k upstream MESA `26.4.1` lines in `star`, "
        "`star_data`, `binary` and `astero`). Upstream test-suite cases "
        f"passing: **{hh['passing']} / {hh['total']}**; "
        f"{hh['reached_termination']} reach upstream's stopping condition "
        f"and {hh['within_numeric_tol']} are within tolerance of the MESA "
        "history.",
        "",
        *_mermaid_pie(engine_sum),
        "",
        "### Modules",
        "",
        *_mermaid_map(by_module),
        "",
        "Colour: green = at least half ported; yellow = at least half "
        "started; pale = started; grey = not started. Arrows point from a "
        "module to what builds on it. Percentages are shares of upstream "
        "lines from the port manifest.",
        "",
        "| Module | Progress | Ported | Started | Upstream lines |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for title, modules in MODULE_GROUPS:
        group_rows = [by_module[m] for m in modules]
        lines.append(_table_row(f"**{title}**", _sum(group_rows)))
        lines.extend(_table_row(f"`{m}`", by_module[m]) for m in modules)
    lines.extend(
        [
            "",
            "Bar: `█` ported, `▒` started (file partly ported), "
            "`░` not started.",
            "",
            "### Upstream test-suite cases",
            "",
            "| Suite | Cases | Run to upstream's stop | Within tolerance | "
            "Passing | Undeclared failures |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for suite in case_manifest.VALID_SUITES:
        s = data["harness"]["by_suite"][suite]
        lines.append(
            f"| `{suite}` | {s['total']} | {s['reached_termination']} | "
            f"{s['within_numeric_tol']} | {s['passing']} | "
            f"{s['undeclared_failures']} |"
        )
    lines.extend(
        [
            f"| **Total** | **{hh['total']}** | "
            f"**{hh['reached_termination']}** | "
            f"**{hh['within_numeric_tol']}** | **{hh['passing']}** | "
            f"**{hh['undeclared_failures']}** |",
            "",
            "A case passes only when upstream's unmodified inlists run to "
            "upstream's stopping condition and the history matches MESA "
            "`26.4.1` within the declared tolerances.",
            "",
            "### Work in flight",
            "",
            f"- Packages under [#140]({REPO_URL}/issues/140): "
            f"**{head['closed_140']} closed**, {head['open_140']} open.",
            f"- Open pull requests: {n_prs}.",
            f"- Deferred upstream issues ([#169]({REPO_URL}/issues/169)): "
            f"{head['deferred_169_count']} open.",
            f"- Test line coverage: Rust {rs_pct:.1f}%, Python {py_pct:.1f}%.",
        ]
    )
    return lines
