# SPDX-License-Identifier: GPL-3.0-or-later
"""Bijective known-gaps verification gate for tests and skips.

Enforces bidirectional correspondence between code-level suppression markers
(Rust ``#[ignore]`` and Python ``@pytest.mark.{xfail,skip,skipif}``) and the
``[[test_gaps]]`` section of ``benchmarks/history_harness/known_gaps.toml``.
Every declared gap must cite an open GitHub issue, and ``--ignored`` campaign
runs fail if a test passes unexpectedly (``stale_known_gap``) or fails
unexpectedly.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import pathlib
import re
import sys
import tomllib
import warnings
from collections.abc import Sequence
from typing import Any, Literal

from flexmesa.testing import _dashboard_github, case_manifest

GapKind = Literal["rust_ignore", "python_xfail", "python_skip"]
GapRole = Literal["physics_gap", "campaign", "benchmark"]

DEFAULT_REPO_SLUG = "joshspeagle/FlexMESA"
DEFAULT_TEST_GAPS_REL = pathlib.Path(
    "benchmarks/history_harness/known_gaps.toml"
)
DEFAULT_HISTORY_GAPS_REL = pathlib.Path(
    "benchmarks/history_harness/known_gaps.toml"
)
DEFAULT_GITHUB_SNAPSHOT_REL = pathlib.Path(
    "benchmarks/dashboard/github_snapshot.json"
)

_VALID_KINDS: frozenset[str] = frozenset(
    {"rust_ignore", "python_xfail", "python_skip"}
)
_VALID_ROLES: frozenset[str] = frozenset(
    {"physics_gap", "campaign", "benchmark"}
)
_OPT_OUT_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})

_RUST_COMMENT_RE = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_RUST_IGNORE_RE = re.compile(
    r"#\s*\[\s*(?:cfg_attr\s*\([^]]*,\s*)?ignore"
    r'(?:\s*=\s*"([^"]*)")?\s*\)?\s*\]'
)
_RUST_FN_RE = re.compile(
    r"(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_ISSUE_RE = re.compile(r"#([1-9][0-9]*)\b")
_CARGO_OUTCOME_RE = re.compile(
    r"^test\s+([A-Za-z0-9_:<>]+)(?:\s+-\s+should panic)?\s+\.\.\.\s+"
    r"(ok|FAILED|ignored)\s*$"
)


@dataclasses.dataclass(frozen=True)
class DeclaredTestGap:
    """Single declared test-suppression record from ``known_gaps.toml``.

    Attributes:
        kind: Marker type (``rust_ignore``, ``python_xfail``, or
            ``python_skip``).
        file: Repository-relative POSIX path to the test source file.
        test_name: Function or class name carrying the marker.
        issue: Primary owning open issue tag (``#<number>``).
        crate: Cargo package name for Rust tests, or ``None`` for Python.
        role: Functional category (``physics_gap``, ``campaign``, or
            ``benchmark``).
        expect_fail_when_run: Whether passing under ``cargo test -- --ignored``
            must fail the gate as a stale gap; if ``False``, failing under
            ``--ignored`` triggers an unexpected failure violation.
        reason: Human-readable explanation anchored to open work.
    """

    kind: GapKind
    file: str
    test_name: str
    issue: str
    crate: str | None
    role: GapRole
    expect_fail_when_run: bool
    reason: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Unique ``(kind, file, test_name)`` identifier."""
        return (self.kind, self.file, self.test_name)


@dataclasses.dataclass(frozen=True)
class DiscoveredMarker:
    """Test suppression marker extracted from Rust or Python source.

    Attributes:
        kind: Marker type (``rust_ignore``, ``python_xfail``, or
            ``python_skip``).
        file: Repository-relative POSIX path to the source file.
        line: 1-based source line number of the attribute or decorator.
        test_name: Annotated Rust function or Python function/class name.
        issues: Issue tags (``#<number>``) parsed from the marker metadata.
        reason: Extracted attribute or decorator text.
    """

    kind: GapKind
    file: str
    line: int
    test_name: str
    issues: tuple[str, ...]
    reason: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Unique ``(kind, file, test_name)`` identifier."""
        return (self.kind, self.file, self.test_name)


@dataclasses.dataclass(frozen=True)
class KnownGapsAuditResult:
    """Outcome summary of the bijective known-gaps verification gate."""

    declared_count: int
    discovered_count: int
    undeclared_markers: tuple[str, ...]
    stale_declarations: tuple[str, ...]
    duplicate_declarations: tuple[str, ...]
    unlinked_markers: tuple[str, ...]
    issue_mismatches: tuple[str, ...]
    closed_issue_violations: tuple[str, ...]
    unknown_issue_violations: tuple[str, ...]
    unexpected_passes: tuple[str, ...]
    unexpected_failures: tuple[str, ...]
    undeclared_ignored_runs: tuple[str, ...]
    missing_ignored_runs: tuple[str, ...]
    history_gap_issue_violations: tuple[str, ...]
    imperative_python_skips: tuple[str, ...]
    fail_on_imperative_skips: bool = False

    @property
    def only_unexpected_passes(self) -> bool:
        """Whether the gate's only finding is ignored tests that now pass.

        The nightly campaign reports this improvement with its own exit
        status so that it asks for the ledger to be updated instead of
        raising the regression alert (DECISIONS #81).
        """
        if self.ok or not self.unexpected_passes:
            return False
        return dataclasses.replace(self, unexpected_passes=()).ok

    @property
    def ok(self) -> bool:
        """Whether all bijective, issue, and execution checks pass."""
        if self.fail_on_imperative_skips and self.imperative_python_skips:
            return False
        return not any(
            (
                self.undeclared_markers,
                self.stale_declarations,
                self.duplicate_declarations,
                self.unlinked_markers,
                self.issue_mismatches,
                self.closed_issue_violations,
                self.unknown_issue_violations,
                self.unexpected_passes,
                self.unexpected_failures,
                self.undeclared_ignored_runs,
                self.missing_ignored_runs,
                self.history_gap_issue_violations,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit findings into a JSON-compatible dictionary."""
        return {
            "ok": self.ok,
            "declared_count": self.declared_count,
            "discovered_count": self.discovered_count,
            "undeclared_markers": list(self.undeclared_markers),
            "stale_declarations": list(self.stale_declarations),
            "duplicate_declarations": list(self.duplicate_declarations),
            "unlinked_markers": list(self.unlinked_markers),
            "issue_mismatches": list(self.issue_mismatches),
            "closed_issue_violations": list(self.closed_issue_violations),
            "unknown_issue_violations": list(self.unknown_issue_violations),
            "unexpected_passes": list(self.unexpected_passes),
            "unexpected_failures": list(self.unexpected_failures),
            "undeclared_ignored_runs": list(self.undeclared_ignored_runs),
            "missing_ignored_runs": list(self.missing_ignored_runs),
            "history_gap_issue_violations": list(
                self.history_gap_issue_violations
            ),
            "imperative_python_skips": list(self.imperative_python_skips),
            "fail_on_imperative_skips": self.fail_on_imperative_skips,
        }


def require_zero_skips_enabled() -> bool:
    """Return whether strict non-vacuous zero-skip enforcement is active.

    Defaults to ``True`` unless ``FLEXMESA_REQUIRE_ZERO_SKIPS`` is explicitly
    set to ``0``, ``false``, ``no``, or ``off`` for local partial runs.
    """
    raw = os.environ.get("FLEXMESA_REQUIRE_ZERO_SKIPS", "").strip().lower()
    return raw not in _OPT_OUT_VALUES


def require_upstream_file(
    path: pathlib.Path, *, label: str = "upstream test input"
) -> bool:
    """Verify an upstream MESA test file exists under zero-skip enforcement.

    Args:
        path: Filesystem path expected from ``MESA_DIR`` or bundled inputs.
        label: Descriptive label included in the assertion message.

    Returns:
        ``True`` if ``path`` exists on disk, or ``False`` only when
        ``FLEXMESA_REQUIRE_ZERO_SKIPS`` is explicitly opted out and ``MESA_DIR``
        is unset.

    Raises:
        AssertionError: If ``path`` is missing while ``MESA_DIR`` is set or
            zero-skip enforcement is enabled.
    """
    if path.is_file():
        return True
    has_mesa_dir = bool(os.environ.get("MESA_DIR", "").strip())
    if has_mesa_dir or require_zero_skips_enabled():
        raise AssertionError(
            f"Missing required {label}: {path}. Set MESA_DIR or run "
            "'python -m flexmesa.data.fetch fetch --include-test-inputs' "
            "(or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 to opt out locally)."
        )
    _warn_not_run(f"{label} {path}")
    return False


def _warn_not_run(what: str) -> None:
    """Say on stderr that a test body is not being run for lack of inputs.

    Only reachable with ``FLEXMESA_REQUIRE_ZERO_SKIPS`` opted out. The
    calling test then returns without checking anything, so it is reported
    as passed; this warning is what makes that visible.
    """
    warnings.warn(
        f"NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {what}; "
        "the calling test checks nothing and is not evidence of anything",
        RuntimeWarning,
        stacklevel=3,
    )


def require_mesa_test_inputs_dir() -> pathlib.Path | None:
    """Resolve ``MESA_DIR`` or cached ``mesa_test_inputs`` without skipping.

    Returns:
        Resolved directory containing ``star/test_suite``, or ``None`` only
        when ``FLEXMESA_REQUIRE_ZERO_SKIPS`` is explicitly opted out and the
        directory is absent on disk.

    Raises:
        AssertionError: If ``star/test_suite`` is missing while ``MESA_DIR``
            is set or zero-skip enforcement is enabled.
    """
    from flexmesa.data import fetch

    env_dir = os.environ.get("MESA_DIR", "").strip()
    root = (
        pathlib.Path(env_dir)
        if env_dir
        else fetch.resolve_mesa_test_inputs_dir()
    )
    if (root / "star" / "test_suite").is_dir():
        return root
    if env_dir or require_zero_skips_enabled():
        raise AssertionError(
            f"Missing required upstream MESA test inputs under {root}. "
            "Set MESA_DIR or run "
            "'python -m flexmesa.data.fetch fetch --include-test-inputs' "
            "(or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 to opt out locally)."
        )
    _warn_not_run(f"upstream MESA test inputs under {root}")
    return None


def _skip_rust_attribute(text: str, start: int) -> int:
    """Advance past a ``#[...]`` attribute starting at ``start``."""
    pos = start + 2
    depth = 1
    in_str = False
    escaped = False
    n = len(text)
    while pos < n and depth > 0:
        ch = text[pos]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        pos += 1
    return pos


def _find_following_rust_fn(cleaned: str, start: int) -> str | None:
    """Find the Rust ``fn <name>`` declaration following an attribute."""
    pos = start
    n = len(cleaned)
    while pos < n:
        while pos < n and cleaned[pos].isspace():
            pos += 1
        if cleaned.startswith("#[", pos):
            pos = _skip_rust_attribute(cleaned, pos)
            continue
        match = _RUST_FN_RE.match(cleaned, pos)
        if match is not None:
            return match.group(1)
        return None
    return None


def extract_rust_test_ignores(
    raw_text: str, rel_path: str
) -> list[DiscoveredMarker]:
    """Extract ``#[ignore]`` markers and their test function names from Rust."""
    if "ignore" not in raw_text:
        return []
    cleaned = _RUST_COMMENT_RE.sub(
        lambda m: "\n" * m.group(0).count("\n"), raw_text
    )
    markers: list[DiscoveredMarker] = []
    for match in _RUST_IGNORE_RE.finditer(cleaned):
        line = cleaned.count("\n", 0, match.start()) + 1
        reason = match.group(1) or ""
        test_name = _find_following_rust_fn(cleaned, match.end()) or "<unknown>"
        issues = tuple(
            f"#{num}"
            for num in sorted({int(tok) for tok in _ISSUE_RE.findall(reason)})
        )
        markers.append(
            DiscoveredMarker(
                kind="rust_ignore",
                file=rel_path,
                line=line,
                test_name=test_name,
                issues=issues,
                reason=reason,
            )
        )
    return markers


def _scan_python_file(
    source: str, rel_path: str
) -> tuple[list[DiscoveredMarker], list[str]]:
    """Extract suppression decorators and imperative skips in one AST pass."""
    if "xfail" not in source and "skip" not in source:
        return [], []
    tree = ast.parse(source, filename=rel_path)
    markers: list[DiscoveredMarker] = []
    imp_skips: list[str] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            for dec in node.decorator_list:
                seg = ast.get_source_segment(source, dec) or ""
                mark_match = re.search(r"\bmark\.(xfail|skip|skipif)\b", seg)
                if mark_match is None:
                    continue
                kind: GapKind = (
                    "python_xfail"
                    if mark_match.group(1) == "xfail"
                    else "python_skip"
                )
                issues = tuple(
                    f"#{num}"
                    for num in sorted(
                        {int(tok) for tok in _ISSUE_RE.findall(seg)}
                    )
                )
                markers.append(
                    DiscoveredMarker(
                        kind=kind,
                        file=rel_path,
                        line=int(dec.lineno),
                        test_name=node.name,
                        issues=issues,
                        reason=" ".join(seg.split()),
                    )
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"skip", "xfail", "importorskip"}
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"
            ):
                imp_skips.append(f"{rel_path}:{node.lineno}")
    return markers, sorted(imp_skips)


def extract_python_test_markers(
    source: str, rel_path: str
) -> list[DiscoveredMarker]:
    """Extract ``@pytest.mark.{xfail,skip,skipif}`` decorators via AST."""
    return _scan_python_file(source, rel_path)[0]


def scan_imperative_python_skips(root: pathlib.Path) -> list[str]:
    """Return ``file:line`` for runtime skip, xfail and importorskip calls."""
    return _discover_repo_test_state(root)[1]


def _discover_repo_test_state(
    root: pathlib.Path,
) -> tuple[list[DiscoveredMarker], list[str]]:
    """Scan Rust crates once and Python sources in a single AST pass."""
    markers: list[DiscoveredMarker] = []
    imp_skips: list[str] = []
    crates_dir = root / "crates"
    if crates_dir.is_dir():
        for rs_path in sorted(crates_dir.rglob("*.rs")):
            rel = rs_path.relative_to(root).as_posix()
            markers.extend(
                extract_rust_test_ignores(
                    rs_path.read_text(encoding="utf-8"), rel
                )
            )
    for sub in ("tests", "src/flexmesa"):
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        for py_path in sorted(sub_dir.rglob("*.py")):
            rel = py_path.relative_to(root).as_posix()
            py_markers, py_skips = _scan_python_file(
                py_path.read_text(encoding="utf-8"), rel
            )
            markers.extend(py_markers)
            imp_skips.extend(py_skips)
    return markers, sorted(imp_skips)


def discover_repo_test_markers(root: pathlib.Path) -> list[DiscoveredMarker]:
    """Scan all Rust crates and Python sources for test suppression markers."""
    return _discover_repo_test_state(root)[0]


def _parse_declared_gap_item(
    item: Any, idx: int, toml_path: pathlib.Path
) -> DeclaredTestGap:
    """Validate and construct one :class:`DeclaredTestGap` TOML row."""
    if not isinstance(item, dict):
        raise ValueError(f"gaps[{idx}] must be a table in {toml_path}")
    kind = str(item.get("kind", "")).strip()
    if kind not in _VALID_KINDS:
        raise ValueError(f"gaps[{idx}].kind invalid: {kind!r}")
    file_rel = str(item.get("file", "")).strip().replace("\\", "/")
    test_name = str(item.get("test_name", "")).strip()
    issue = str(item.get("issue", "")).strip()
    if not file_rel or not test_name:
        raise ValueError(f"gaps[{idx}] requires non-empty file/test_name")
    if not re.fullmatch(r"#[1-9][0-9]*", issue):
        raise ValueError(
            f"gaps[{idx}].issue must match '#<number>', got {issue!r}"
        )
    crate_val = item.get("crate")
    crate = str(crate_val).strip() if crate_val is not None else None
    if kind == "rust_ignore" and not crate:
        raise ValueError(f"gaps[{idx}] rust_ignore requires 'crate'")
    role = str(item.get("role", "physics_gap")).strip()
    if role not in _VALID_ROLES:
        raise ValueError(f"gaps[{idx}].role invalid: {role!r}")
    reason = str(item.get("reason", "")).strip()
    if not reason:
        raise ValueError(f"gaps[{idx}].reason must be non-empty")
    return DeclaredTestGap(
        kind=kind,  # type: ignore[arg-type]
        file=file_rel,
        test_name=test_name,
        issue=issue,
        crate=crate,
        role=role,  # type: ignore[arg-type]
        expect_fail_when_run=bool(item.get("expect_fail_when_run", True)),
        reason=reason,
    )


def load_declared_test_gaps(
    toml_path: pathlib.Path,
) -> tuple[list[DeclaredTestGap], list[str]]:
    """Parse ``[[test_gaps]]`` from ``known_gaps.toml`` and validate fields.

    Args:
        toml_path: Path to the TOML declaration file.

    Returns:
        Tuple ``(gaps, duplicate_keys)``.

    Raises:
        FileNotFoundError: If ``toml_path`` does not exist.
        ValueError: If any record violates required schema constraints.
    """
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"Declared test-gaps file not found: {toml_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    if "test_gaps" in raw:
        entries = raw["test_gaps"]
        if not isinstance(entries, list):
            raise ValueError(
                f"'test_gaps' must be an array of tables in {toml_path}"
            )
    else:
        raw_gaps = raw.get("gaps", [])
        if not isinstance(raw_gaps, list):
            raise ValueError(
                f"'gaps' must be an array of tables in {toml_path}"
            )
        entries = [
            item
            for item in raw_gaps
            if isinstance(item, dict) and "kind" in item
        ]

    seen: set[tuple[str, str, str]] = set()
    duplicates: list[str] = []
    gaps: list[DeclaredTestGap] = []
    for idx, item in enumerate(entries):
        gap = _parse_declared_gap_item(item, idx, toml_path)
        if gap.key in seen:
            duplicates.append(f"{gap.kind}:{gap.file}::{gap.test_name}")
        else:
            seen.add(gap.key)
            gaps.append(gap)
    return gaps, duplicates


def load_issue_states(
    root: pathlib.Path,
    *,
    extra_issue_numbers: Sequence[int] = (),
    fetch_github_api: bool = False,
    repo_slug: str = DEFAULT_REPO_SLUG,
) -> dict[int, str]:
    """Load issue open/closed states from snapshot and optionally GitHub API."""
    states: dict[int, str] = {}
    snap_path = root / DEFAULT_GITHUB_SNAPSHOT_REL
    if snap_path.is_file():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        for key in ("tracker_issue", "backlog_issue"):
            if isinstance(snap.get(key), int):
                states[int(snap[key])] = "open"
        for phase in snap.get("phases_140", []):
            for item in phase.get("issues", []):
                states[int(item["number"])] = str(item["state"]).lower()
        for item in snap.get("deferred_169", []):
            states[int(item["number"])] = str(item["state"]).lower()

    if fetch_github_api:
        del extra_issue_numbers
        page = 1
        while True:
            batch = _dashboard_github._gh_api_json(
                f"repos/{repo_slug}/issues?state=all&per_page=100&page={page}"
            )
            if not isinstance(batch, list) or not batch:
                break
            for item in batch:
                if "pull_request" not in item:
                    states[int(item["number"])] = str(
                        item.get("state", "unknown")
                    ).lower()
            if len(batch) < 100:
                break
            page += 1
    return states


def parse_cargo_test_outcomes(log_text: str) -> dict[str, str]:
    """Parse ``test <name> ... ok|FAILED|ignored`` lines from Cargo logs.

    Returns:
        Mapping from short test function name (stripping module path prefix) to
        its reported outcome (``ok``, ``FAILED``, or ``ignored``).
    """
    outcomes: dict[str, str] = {}
    for raw_line in log_text.splitlines():
        match = _CARGO_OUTCOME_RE.match(raw_line.strip())
        if match is not None:
            short_name = match.group(1).rsplit("::", maxsplit=1)[-1]
            outcomes[short_name] = match.group(2)
    return outcomes


def _collect_marker_mismatches(
    declared: Sequence[DeclaredTestGap],
    discovered: Sequence[DiscoveredMarker],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compute undeclared, stale, unlinked, and issue-mismatch diagnostics."""
    declared_by_key = {gap.key: gap for gap in declared}
    discovered_by_key = {m.key: m for m in discovered}
    undeclared = sorted(
        f"{m.kind}:{m.file}:{m.line}:{m.test_name}"
        for m in discovered
        if m.key not in declared_by_key
    )
    stale = sorted(
        f"{g.kind}:{g.file}::{g.test_name} ({g.issue})"
        for g in declared
        if g.key not in discovered_by_key
    )
    unlinked = sorted(
        f"{m.kind}:{m.file}:{m.line}:{m.test_name}"
        for m in discovered
        if not m.issues
    )
    issue_mismatches = sorted(
        f"{gap.file}::{gap.test_name}: declared {gap.issue} not in "
        f"source marker issues {marker.issues}"
        for key, gap in declared_by_key.items()
        if (marker := discovered_by_key.get(key)) is not None
        and gap.issue not in marker.issues
    )
    return undeclared, stale, unlinked, issue_mismatches


def _check_issue_states(
    root: pathlib.Path,
    declared: Sequence[DeclaredTestGap],
    history_gaps_path: pathlib.Path,
    *,
    fetch_github_api: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Verify every cited test-gap and history-gap issue is open."""
    history_gaps: dict[tuple[str, str], case_manifest.KnownGapSpec] = {}
    referenced_nums: list[int] = [int(g.issue[1:]) for g in declared]
    if history_gaps_path.is_file():
        history_gaps = case_manifest.load_known_gaps(history_gaps_path)
        referenced_nums.extend(
            int(hg.issue[1:]) for hg in history_gaps.values()
        )

    issue_states = load_issue_states(
        root,
        extra_issue_numbers=referenced_nums,
        fetch_github_api=fetch_github_api,
    )
    closed: list[str] = []
    unknown: list[str] = []
    for gap in declared:
        state = issue_states.get(int(gap.issue[1:]))
        loc = f"{gap.file}::{gap.test_name} ({gap.issue})"
        if state is None:
            unknown.append(loc)
        elif state != "open":
            closed.append(f"{loc} is {state}")

    history_violations: list[str] = []
    for (suite, case), hg in sorted(history_gaps.items()):
        state = issue_states.get(int(hg.issue[1:]))
        if state is None:
            history_violations.append(
                f"{suite}/{case} ({hg.issue}): issue missing from snapshot"
            )
        elif state != "open":
            history_violations.append(
                f"{suite}/{case} ({hg.issue}): issue is {state}"
            )
    return sorted(closed), sorted(unknown), sorted(history_violations)


def _evaluate_cargo_ignored_logs(
    declared: Sequence[DeclaredTestGap],
    cargo_ignored_logs: Sequence[pathlib.Path],
    *,
    require_all_executed: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Validate ``cargo test -- --ignored`` logs against declared gaps."""
    if not cargo_ignored_logs and not require_all_executed:
        return [], [], [], []
    merged_outcomes: dict[str, str] = {}
    for log_path in cargo_ignored_logs:
        merged_outcomes.update(
            parse_cargo_test_outcomes(log_path.read_text(encoding="utf-8"))
        )
    rust_by_name = {g.test_name: g for g in declared if g.kind == "rust_ignore"}
    unexpected_passes: list[str] = []
    unexpected_failures: list[str] = []
    undeclared_runs: list[str] = []
    for test_name, status in sorted(merged_outcomes.items()):
        if status == "ignored":
            continue
        gap_opt = rust_by_name.get(test_name)
        if gap_opt is None:
            undeclared_runs.append(f"{test_name} ({status})")
        elif gap_opt.expect_fail_when_run and status == "ok":
            unexpected_passes.append(
                f"{gap_opt.file}::{test_name} ({gap_opt.issue}) "
                "passed under --ignored [stale_known_gap]"
            )
        elif not gap_opt.expect_fail_when_run and status != "ok":
            unexpected_failures.append(
                f"{gap_opt.file}::{test_name} ({gap_opt.issue}) "
                f"returned {status} under --ignored [unexpected_failure]"
            )

    missing_runs: list[str] = []
    if require_all_executed:
        for test_name, gap in sorted(rust_by_name.items()):
            exec_status = merged_outcomes.get(test_name)
            if exec_status is None or exec_status == "ignored":
                missing_runs.append(
                    f"{gap.file}::{test_name} ({gap.issue}) not executed"
                )
    return (
        sorted(unexpected_passes),
        sorted(unexpected_failures),
        sorted(undeclared_runs),
        sorted(missing_runs),
    )


def verify_known_gaps_gate(
    root: pathlib.Path,
    *,
    test_gaps_path: pathlib.Path | None = None,
    history_gaps_path: pathlib.Path | None = None,
    cargo_ignored_logs: Sequence[pathlib.Path] = (),
    require_all_rust_ignores_executed: bool = False,
    fail_on_imperative_skips: bool = False,
    fetch_github_api: bool = False,
) -> KnownGapsAuditResult:
    """Run bijective, issue-status, and ignored-outcome verification gates.

    Args:
        root: Repository root directory.
        test_gaps_path: Override TOML path for ``[[test_gaps]]``.
        history_gaps_path: Override path to ``known_gaps.toml``.
        cargo_ignored_logs: Optional Cargo ``--ignored`` output files to enforce
            ``unexpected_passes``, ``unexpected_failures``, and undeclared runs.
        require_all_rust_ignores_executed: Require every ``rust_ignore``
            entry to appear with a non-ignored outcome in the Cargo logs.
        fail_on_imperative_skips: Include runtime ``pytest.skip()`` sites in
            the pass/fail evaluation of :attr:`KnownGapsAuditResult.ok`.
        fetch_github_api: Whether to query GitHub's REST API for live open/
            closed verification of every cited issue.

    Returns:
        Structured :class:`KnownGapsAuditResult`.
    """
    t_path = test_gaps_path or (root / DEFAULT_TEST_GAPS_REL)
    h_path = history_gaps_path or (root / DEFAULT_HISTORY_GAPS_REL)
    declared, duplicates = load_declared_test_gaps(t_path)
    discovered, imp_skips = _discover_repo_test_state(root)
    undeclared, stale, unlinked, issue_mismatches = _collect_marker_mismatches(
        declared, discovered
    )
    closed, unknown, history_violations = _check_issue_states(
        root, declared, h_path, fetch_github_api=fetch_github_api
    )
    unexp_pass, unexp_fail, und_runs, miss_runs = _evaluate_cargo_ignored_logs(
        declared,
        cargo_ignored_logs,
        require_all_executed=require_all_rust_ignores_executed,
    )
    return KnownGapsAuditResult(
        declared_count=len(declared),
        discovered_count=len(discovered),
        undeclared_markers=tuple(undeclared),
        stale_declarations=tuple(stale),
        duplicate_declarations=tuple(sorted(duplicates)),
        unlinked_markers=tuple(unlinked),
        issue_mismatches=tuple(issue_mismatches),
        closed_issue_violations=tuple(closed),
        unknown_issue_violations=tuple(unknown),
        unexpected_passes=tuple(unexp_pass),
        unexpected_failures=tuple(unexp_fail),
        undeclared_ignored_runs=tuple(und_runs),
        missing_ignored_runs=tuple(miss_runs),
        history_gap_issue_violations=tuple(history_violations),
        imperative_python_skips=tuple(imp_skips),
        fail_on_imperative_skips=fail_on_imperative_skips,
    )


def render_markdown_summary(result: KnownGapsAuditResult) -> str:
    """Format a Markdown summary of the known-gaps bijective audit."""
    status = "PASS" if result.ok else "FAIL"
    closed_total = len(result.closed_issue_violations) + len(
        result.unknown_issue_violations
    )
    lines = [
        f"## Known-Gaps Gate: {status}",
        "",
        f"- **Declared test gaps**: `{result.declared_count}`",
        f"- **Discovered source markers**: `{result.discovered_count}`",
        f"- **Undeclared markers**: `{len(result.undeclared_markers)}`",
        f"- **Stale declarations**: `{len(result.stale_declarations)}`",
        f"- **Closed/unknown issue citations**: `{closed_total}`",
        f"- **Unexpected `--ignored` passes**: "
        f"`{len(result.unexpected_passes)}`",
        f"- **Unexpected `--ignored` failures**: "
        f"`{len(result.unexpected_failures)}`",
        f"- **Missing `--ignored` runs**: `{len(result.missing_ignored_runs)}`",
        f"- **History-gap issue violations**: "
        f"`{len(result.history_gap_issue_violations)}`",
    ]
    if result.imperative_python_skips:
        lines.append(
            "- **Imperative `pytest.skip` call sites**: "
            f"`{len(result.imperative_python_skips)}`"
        )
    sections = [
        ("Undeclared source markers", result.undeclared_markers),
        ("Stale manifest declarations", result.stale_declarations),
        ("Duplicate manifest declarations", result.duplicate_declarations),
        ("Unlinked source markers", result.unlinked_markers),
        ("Declared vs source issue mismatches", result.issue_mismatches),
        ("Closed issue violations", result.closed_issue_violations),
        ("Unknown issue violations", result.unknown_issue_violations),
        ("Unexpected passes (stale_known_gap)", result.unexpected_passes),
        ("Unexpected failures", result.unexpected_failures),
        ("Undeclared --ignored executions", result.undeclared_ignored_runs),
        ("Missing --ignored executions", result.missing_ignored_runs),
        ("History gap issue violations", result.history_gap_issue_violations),
    ]
    if result.fail_on_imperative_skips:
        sections.append(
            ("Imperative pytest.skip() calls", result.imperative_python_skips)
        )
    for title, items in sections:
        if items:
            lines.extend(["", f"### {title}", ""])
            lines.extend(f"- `{item}`" for item in items)
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flexmesa.testing.known_gaps_gate",
        description="Verify bijective test-gap declarations and open issues.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (defaults to workspace containing this module).",
    )
    parser.add_argument(
        "--test-gaps",
        default=None,
        help="Override path to TOML containing [[test_gaps]].",
    )
    parser.add_argument(
        "--history-gaps",
        default=None,
        help="Override path to known_gaps.toml.",
    )
    parser.add_argument(
        "--cargo-ignored-log",
        action="append",
        default=[],
        help="Path to 'cargo test -- --ignored' output log (repeatable).",
    )
    parser.add_argument(
        "--require-all-rust-ignores-executed",
        action="store_true",
        help="Require every declared rust_ignore test to execute in logs.",
    )
    parser.add_argument(
        "--live-github",
        action="store_true",
        help="Verify issue open/closed state against live GitHub REST API.",
    )
    parser.add_argument(
        "--fail-on-imperative-skips",
        action="store_true",
        help="Also exit non-zero if any imperative pytest.skip() remains.",
    )
    parser.add_argument(
        "--stale-gap-status",
        type=int,
        default=1,
        help=(
            "Exit status when the only finding is --ignored tests that now "
            "pass (default 1; the nightly campaign uses 3, DECISIONS #81)."
        ),
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning ``0`` on clean gate pass, ``1`` on failure."""
    ns = _build_arg_parser().parse_args(argv)
    root = (
        pathlib.Path(ns.repo_root).resolve()
        if ns.repo_root
        else case_manifest.repo_root()
    )
    result = verify_known_gaps_gate(
        root,
        test_gaps_path=(
            pathlib.Path(ns.test_gaps).resolve() if ns.test_gaps else None
        ),
        history_gaps_path=(
            pathlib.Path(ns.history_gaps).resolve() if ns.history_gaps else None
        ),
        cargo_ignored_logs=[
            pathlib.Path(p).resolve() for p in ns.cargo_ignored_log
        ],
        require_all_rust_ignores_executed=bool(
            ns.require_all_rust_ignores_executed
        ),
        fail_on_imperative_skips=bool(ns.fail_on_imperative_skips),
        fetch_github_api=bool(ns.live_github),
    )
    md = render_markdown_summary(result)
    print(md, end="")
    if ns.markdown_out:
        out_md = pathlib.Path(ns.markdown_out)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    if ns.json_out:
        out_json = pathlib.Path(ns.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    if result.ok:
        return 0
    return int(ns.stale_gap_status) if result.only_unexpected_passes else 1


if __name__ == "__main__":
    sys.exit(main())
