# SPDX-License-Identifier: LGPL-3.0-or-later
"""Level-2 IEEE-754 module hex verification gate (``mesa_module_hex``).

Compares FlexMESA double-precision computations in ``strict`` arithmetic mode
against upstream MESA 26.4.1 ``raw_bits_hex`` captures across module test
suites while accounting bijectively for all 26 upstream MESA packages in
``summary.json`` against ``benchmarks/module_hex/known_module_gaps.toml``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import tomllib
from collections.abc import Sequence
from typing import Any, Literal

from flexmesa import _rust
from flexmesa.data import fetch
from flexmesa.testing import (
    _module_hex_probes,
    case_manifest,
    known_gaps_gate,
)

ModuleGapStatus = Literal[
    "partial_0_ulp",
    "nonzero_ulp",
    "unported_hex_harness",
    "zero_values_in_mesa_reference",
]

DEFAULT_GAPS_TOML_REL = pathlib.Path(
    "benchmarks/module_hex/known_module_gaps.toml"
)
DEFAULT_SUMMARY_JSON_REL = pathlib.Path(
    "benchmarks/module_hex/module_hex_summary.json"
)

_VALID_MODULE_STATUSES: frozenset[str] = frozenset(
    {
        "partial_0_ulp",
        "nonzero_ulp",
        "unported_hex_harness",
        "zero_values_in_mesa_reference",
    }
)

float64_to_raw_hex = _module_hex_probes.float64_to_raw_hex
raw_hex_to_float64 = _module_hex_probes.raw_hex_to_float64
ulp_distance = _module_hex_probes.ulp_distance
_load_tsv_rows = _module_hex_probes.load_tsv_rows


@dataclasses.dataclass(frozen=True)
class DeclaredModuleGap:
    """Single declared module hex gap from ``known_module_gaps.toml``."""

    module: str
    status: ModuleGapStatus
    issue: str | None
    max_declared_ulp: int
    min_zero_ulp_count: int
    expect_nonzero_ulp: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class ModuleProbeResult:
    """Measured IEEE-754 ULP audit outcome for one MESA package."""

    module: str
    status: str
    upstream_exact_values: int
    expected_outputs: int
    checked_values: int
    zero_ulp_values: int
    max_ulp: int
    issue: str | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize module probe result to JSON-safe dict."""
        return {
            "module": self.module,
            "status": self.status,
            "upstream_exact_values": self.upstream_exact_values,
            "expected_outputs": self.expected_outputs,
            "checked_values": self.checked_values,
            "zero_ulp_values": self.zero_ulp_values,
            "max_ulp": self.max_ulp,
            "issue": self.issue,
            "notes": self.notes,
        }


@dataclasses.dataclass(frozen=True)
class ModuleHexAuditResult:
    """Complete Level-2 module-hex gate outcome across all 26 MESA packages."""

    total_packages_in_install: int
    packages_with_test_dir: int
    total_exact_values_captured: int
    module_results: tuple[ModuleProbeResult, ...]
    undeclared_module_gaps: tuple[str, ...]
    undeclared_hex_mismatches: tuple[str, ...]
    stale_module_gaps: tuple[str, ...]
    zero_ulp_regressions: tuple[str, ...]
    closed_or_unknown_issues: tuple[str, ...]
    duplicate_declarations: tuple[str, ...]
    unknown_declared_modules: tuple[str, ...]
    inconsistent_module_declarations: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether all module-hex checks and declarations pass."""
        return not any(
            (
                self.undeclared_module_gaps,
                self.undeclared_hex_mismatches,
                self.stale_module_gaps,
                self.zero_ulp_regressions,
                self.closed_or_unknown_issues,
                self.duplicate_declarations,
                self.unknown_declared_modules,
                self.inconsistent_module_declarations,
            )
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Build deterministic JSON matching ``module_hex_summary.json``."""
        total_checked = sum(r.checked_values for r in self.module_results)
        total_zero = sum(r.zero_ulp_values for r in self.module_results)
        covered_mods = sum(
            1 for r in self.module_results if r.checked_values > 0
        )
        passing_0ulp_mods = sum(
            1 for r in self.module_results if r.status == "pass_0_ulp"
        )
        return {
            "ok": self.ok,
            "total_packages_in_install": self.total_packages_in_install,
            "packages_with_test_dir": self.packages_with_test_dir,
            "total_exact_values_captured": self.total_exact_values_captured,
            "modules_with_executable_probes": covered_mods,
            "modules_passing_0_ulp": passing_0ulp_mods,
            "total_values_checked": total_checked,
            "total_values_0_ulp": total_zero,
            "modules": [r.to_dict() for r in self.module_results],
        }


def resolve_hex_root(
    explicit_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Locate ``mesa_module_hex`` directory or fail cleanly when missing."""
    root = (
        explicit_dir.resolve()
        if explicit_dir is not None
        else fetch.resolve_mesa_module_hex_dir()
    )
    summary = root / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(
            f"Missing mesa_module_hex summary at {summary}. Run "
            "'python -m flexmesa.data.fetch fetch --include-test-inputs' "
            "or set FLEXMESA_MODULE_HEX_DIR."
        )
    return root


def load_declared_module_gaps(
    toml_path: pathlib.Path,
) -> tuple[dict[str, DeclaredModuleGap], list[str]]:
    """Parse ``known_module_gaps.toml`` into ``(by_module, duplicates)``."""
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"Declared module-hex gaps file not found: {toml_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    entries = raw.get("gaps", [])
    if not isinstance(entries, list):
        raise ValueError(f"'gaps' must be a list of tables in {toml_path}")

    by_mod: dict[str, DeclaredModuleGap] = {}
    duplicates: list[str] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"gaps[{idx}] must be a table in {toml_path}")
        mod = str(item.get("module", "")).strip()
        status = str(item.get("status", "")).strip()
        if not mod or status not in _VALID_MODULE_STATUSES:
            raise ValueError(
                f"gaps[{idx}] invalid module={mod!r} or status={status!r}"
            )
        issue_raw = item.get("issue")
        issue = str(issue_raw).strip() if issue_raw else None
        if status != "zero_values_in_mesa_reference" and not issue:
            raise ValueError(f"gaps[{idx}] ({mod}) requires an open 'issue'")
        gap = DeclaredModuleGap(
            module=mod,
            status=status,  # type: ignore[arg-type]
            issue=issue,
            max_declared_ulp=int(item.get("max_declared_ulp", 0)),
            min_zero_ulp_count=int(item.get("min_zero_ulp_count", 0)),
            expect_nonzero_ulp=bool(item.get("expect_nonzero_ulp", False)),
            reason=str(item.get("reason", "")).strip(),
        )
        if mod in by_mod:
            duplicates.append(mod)
        else:
            by_mod[mod] = gap
    return by_mod, duplicates


def _check_module_issue_states(
    repo_root: pathlib.Path,
    declared_gaps: dict[str, DeclaredModuleGap],
    *,
    fetch_github_api: bool,
) -> list[str]:
    """Verify every cited module-gap issue exists and remains open."""
    issue_nums = [
        int(g.issue[1:]) for g in declared_gaps.values() if g.issue is not None
    ]
    issue_states = known_gaps_gate.load_issue_states(
        repo_root,
        extra_issue_numbers=issue_nums,
        fetch_github_api=fetch_github_api,
    )
    violations: list[str] = []
    for mod, gap in sorted(declared_gaps.items()):
        if gap.issue is None:
            continue
        st = issue_states.get(int(gap.issue[1:]))
        if st is None:
            violations.append(f"{mod} ({gap.issue}: unknown issue)")
        elif st != "open":
            violations.append(f"{mod} ({gap.issue}: {st})")
    return violations


def _validate_declaration_consistency(
    mod: str,
    up_count: int,
    has_probe: bool,
    gap_opt: DeclaredModuleGap | None,
) -> list[str]:
    """Validate consistency between declared status and probe presence."""
    if gap_opt is None:
        return []
    errors: list[str] = []
    if gap_opt.status == "unported_hex_harness" and has_probe:
        errors.append(
            f"{mod}: status='unported_hex_harness' cannot coexist with an "
            "executable probe"
        )
    if gap_opt.status == "zero_values_in_mesa_reference" and (
        up_count > 0 or has_probe
    ):
        errors.append(
            f"{mod}: status='zero_values_in_mesa_reference' invalid with "
            f"exact_values_captured={up_count}"
        )
    if gap_opt.status == "partial_0_ulp" and (
        gap_opt.max_declared_ulp > 0 or gap_opt.expect_nonzero_ulp
    ):
        errors.append(
            f"{mod}: status='partial_0_ulp' requires max_declared_ulp=0 and "
            "expect_nonzero_ulp=false"
        )
    if gap_opt.status == "nonzero_ulp" and (
        gap_opt.max_declared_ulp <= 0 or not gap_opt.expect_nonzero_ulp
    ):
        errors.append(
            f"{mod}: status='nonzero_ulp' requires max_declared_ulp>0 and "
            "expect_nonzero_ulp=true"
        )
    return errors


def _evaluate_partial_or_nonzero_probe(
    mod: str,
    expected_outputs: int,
    checked: int,
    zeros: int,
    max_u: int,
    gap_opt: DeclaredModuleGap | None,
) -> tuple[str, list[str], list[str], list[str]]:
    """Classify partial or nonzero-ULP probe outcomes against declarations."""
    undeclared_mismatches: list[str] = []
    stale_gaps: list[str] = []
    zero_regressions: list[str] = []
    if gap_opt is None:
        undeclared_mismatches.append(
            f"{mod}: checked={checked}/{expected_outputs}, zero_ulp={zeros}, "
            f"max_ulp={max_u} (undeclared in known_module_gaps.toml)"
        )
        return ("undeclared_mismatch", undeclared_mismatches, stale_gaps, [])
    if (
        gap_opt.expect_nonzero_ulp
        and max_u == 0
        and gap_opt.status == "nonzero_ulp"
    ):
        stale_gaps.append(
            f"{mod} probe now has 0 ULP across {checked} values "
            f"(declared under {gap_opt.issue})"
        )
    if max_u > gap_opt.max_declared_ulp:
        undeclared_mismatches.append(
            f"{mod}: max_ulp={max_u} exceeds max_declared_ulp="
            f"{gap_opt.max_declared_ulp}"
        )
    if zeros < gap_opt.min_zero_ulp_count:
        zero_regressions.append(
            f"{mod}: zero_ulp_values={zeros} fell below "
            f"min_zero_ulp_count={gap_opt.min_zero_ulp_count}"
        )
    return (gap_opt.status, undeclared_mismatches, stale_gaps, zero_regressions)


def _evaluate_single_module(
    h_root: pathlib.Path,
    item: dict[str, Any],
    declared_gaps: dict[str, DeclaredModuleGap],
) -> tuple[
    ModuleProbeResult, list[str], list[str], list[str], list[str], list[str]
]:
    """Run one package's hex probe and validate its gap declarations."""
    mod = str(item["module"])
    up_count = int(item.get("exact_values_captured", 0))
    gap_opt = declared_gaps.get(mod)
    probe_fn = _module_hex_probes.MODULE_PROBES.get(mod)
    inconsistent = _validate_declaration_consistency(
        mod, up_count, probe_fn is not None, gap_opt
    )
    if probe_fn is None:
        kind_lbl = (
            "0-value package undeclared"
            if up_count == 0
            else f"{up_count} upstream values, no probe"
        )
        undeclared_gaps = [f"{mod} ({kind_lbl})"] if gap_opt is None else []
        st_str: str = gap_opt.status if gap_opt else "undeclared_gap"
        note = gap_opt.reason if gap_opt else ""
        res = ModuleProbeResult(
            module=mod,
            status=st_str,
            upstream_exact_values=up_count,
            expected_outputs=up_count,
            checked_values=0,
            zero_ulp_values=0,
            max_ulp=0,
            issue=gap_opt.issue if gap_opt else None,
            notes=note,
        )
        return (res, undeclared_gaps, [], [], [], inconsistent)

    expected_outputs, checked, zeros, max_u, note = probe_fn(h_root)
    if max_u == 0 and checked == expected_outputs:
        stale_gaps = (
            [
                f"{mod} achieved {zeros}/{checked} 0-ULP agreement "
                f"while declared under {gap_opt.issue} [stale_gap]"
            ]
            if gap_opt is not None
            else []
        )
        res = ModuleProbeResult(
            module=mod,
            status="pass_0_ulp",
            upstream_exact_values=up_count,
            expected_outputs=expected_outputs,
            checked_values=checked,
            zero_ulp_values=zeros,
            max_ulp=0,
            issue=gap_opt.issue if gap_opt else None,
            notes=note,
        )
        return (res, [], [], stale_gaps, [], inconsistent)

    st_str, um, stale_gaps, zero_reg = _evaluate_partial_or_nonzero_probe(
        mod, expected_outputs, checked, zeros, max_u, gap_opt
    )
    res = ModuleProbeResult(
        module=mod,
        status=st_str,
        upstream_exact_values=up_count,
        expected_outputs=expected_outputs,
        checked_values=checked,
        zero_ulp_values=zeros,
        max_ulp=max_u,
        issue=gap_opt.issue if gap_opt else None,
        notes=note,
    )
    return (res, [], um, stale_gaps, zero_reg, inconsistent)


def verify_module_hex_gate(
    repo_root: pathlib.Path,
    *,
    hex_dir: pathlib.Path | None = None,
    gaps_toml_path: pathlib.Path | None = None,
    fetch_github_api: bool = False,
) -> ModuleHexAuditResult:
    """Execute Level-2 IEEE-754 bit-exact checks across ``mesa_module_hex``."""
    h_root = resolve_hex_root(hex_dir)
    summary_raw = json.loads(
        (h_root / "summary.json").read_text(encoding="utf-8")
    )
    g_path = gaps_toml_path or (repo_root / DEFAULT_GAPS_TOML_REL)
    declared_gaps, duplicates = load_declared_module_gaps(g_path)
    closed_or_unknown = _check_module_issue_states(
        repo_root, declared_gaps, fetch_github_api=fetch_github_api
    )
    summary_mods = [str(m["module"]) for m in summary_raw.get("modules", [])]
    unknown_declared = sorted(set(declared_gaps) - set(summary_mods))

    prev_mode = _rust.get_numerics_mode()
    results: list[ModuleProbeResult] = []
    undeclared_gaps: list[str] = []
    undeclared_mismatches: list[str] = []
    stale_gaps: list[str] = []
    zero_regressions: list[str] = []
    inconsistent: list[str] = []
    try:
        _rust.set_numerics_mode("strict")
        for item in summary_raw.get("modules", []):
            res, ug, um, sg, zr, inc = _evaluate_single_module(
                h_root, item, declared_gaps
            )
            results.append(res)
            undeclared_gaps.extend(ug)
            undeclared_mismatches.extend(um)
            stale_gaps.extend(sg)
            zero_regressions.extend(zr)
            inconsistent.extend(inc)
    finally:
        _rust.set_numerics_mode(prev_mode)

    return ModuleHexAuditResult(
        total_packages_in_install=int(
            summary_raw.get("total_packages_in_install", len(results))
        ),
        packages_with_test_dir=int(
            summary_raw.get("packages_with_test_dir", 0)
        ),
        total_exact_values_captured=int(
            summary_raw.get("total_exact_values_captured", 0)
        ),
        module_results=tuple(results),
        undeclared_module_gaps=tuple(sorted(undeclared_gaps)),
        undeclared_hex_mismatches=tuple(sorted(undeclared_mismatches)),
        stale_module_gaps=tuple(sorted(stale_gaps)),
        zero_ulp_regressions=tuple(sorted(zero_regressions)),
        closed_or_unknown_issues=tuple(sorted(closed_or_unknown)),
        duplicate_declarations=tuple(sorted(duplicates)),
        unknown_declared_modules=tuple(unknown_declared),
        inconsistent_module_declarations=tuple(sorted(inconsistent)),
    )


def render_markdown_summary(result: ModuleHexAuditResult) -> str:
    """Format a Markdown table and violations for the Level-2 hex gate."""
    s = result.to_summary_dict()
    verdict = "PASS" if result.ok else "FAIL"
    lines = [
        f"## Level-2 Module Hex Gate: {verdict}",
        "",
        f"- **Packages in summary**: `{result.total_packages_in_install}`",
        f"- **Modules with executable probes**: "
        f"`{s['modules_with_executable_probes']}` "
        f"(`{s['modules_passing_0_ulp']}` complete `0 ULP`)",
        f"- **Checked vs `0 ULP` values**: `{s['total_values_0_ulp']}` / "
        f"`{s['total_values_checked']}` "
        f"(of `{result.total_exact_values_captured}` captured)",
        "",
        "| Module | Status | Upstream | Outputs | Checked | 0 ULP | Max ULP "
        "| Issue |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in result.module_results:
        iss = r.issue or "-"
        lines.append(
            f"| `{r.module}` | `{r.status}` | {r.upstream_exact_values} | "
            f"{r.expected_outputs} | {r.checked_values} | "
            f"{r.zero_ulp_values} | {r.max_ulp} | {iss} |"
        )
    for title, items in (
        ("Undeclared module gaps", result.undeclared_module_gaps),
        ("Undeclared hex mismatches", result.undeclared_hex_mismatches),
        ("Stale module gaps", result.stale_module_gaps),
        ("0-ULP count regressions", result.zero_ulp_regressions),
        ("Closed or unknown issues", result.closed_or_unknown_issues),
        ("Duplicate declarations", result.duplicate_declarations),
        ("Unknown declared modules", result.unknown_declared_modules),
        (
            "Inconsistent module declarations",
            result.inconsistent_module_declarations,
        ),
    ):
        if items:
            lines.extend(["", f"### {title}", ""])
            lines.extend(f"- `{item}`" for item in items)
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flexmesa.testing.module_hex_gate",
        description="Verify Level-2 0-ULP IEEE-754 agreement across modules.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--hex-dir", default=None)
    parser.add_argument("--gaps-toml", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--check-summary", action="store_true")
    parser.add_argument("--live-github", action="store_true")
    parser.add_argument("--markdown-out", default=None)
    return parser


def _write_or_check_summary(
    summary_path: pathlib.Path,
    rendered_json: str,
    *,
    write_summary: bool,
    check_summary: bool,
) -> bool:
    if write_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered_json, encoding="utf-8")
    if not check_summary:
        return True
    if not summary_path.is_file():
        print(f"Missing summary file: {summary_path}", file=sys.stderr)
        return False
    if summary_path.read_text(encoding="utf-8") != rendered_json:
        print(
            f"{summary_path} is stale; regenerate with --write-summary",
            file=sys.stderr,
        )
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning ``0`` on clean gate pass, ``1`` on failure."""
    ns = _build_arg_parser().parse_args(argv)
    root = (
        pathlib.Path(ns.repo_root).resolve()
        if ns.repo_root
        else case_manifest.repo_root()
    )
    result = verify_module_hex_gate(
        root,
        hex_dir=pathlib.Path(ns.hex_dir).resolve() if ns.hex_dir else None,
        gaps_toml_path=(
            pathlib.Path(ns.gaps_toml).resolve() if ns.gaps_toml else None
        ),
        fetch_github_api=bool(ns.live_github),
    )
    md = render_markdown_summary(result)
    print(md, end="")
    summary_path = (
        pathlib.Path(ns.summary_json).resolve()
        if ns.summary_json
        else (root / DEFAULT_SUMMARY_JSON_REL)
    )
    summary_ok = _write_or_check_summary(
        summary_path,
        json.dumps(result.to_summary_dict(), indent=2) + "\n",
        write_summary=bool(ns.write_summary),
        check_summary=bool(ns.check_summary),
    )
    if ns.markdown_out:
        out_md = pathlib.Path(ns.markdown_out)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    return 0 if (result.ok and summary_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
