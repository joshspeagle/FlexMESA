# SPDX-License-Identifier: LGPL-3.0-or-later
"""TOML case manifest and declared known-gaps loader for history comparisons.

Defines the schema for ``benchmarks/history_harness/case_manifest.toml`` and
``benchmarks/history_harness/known_gaps.toml``, covering the 106 upstream test
cases of MESA release ``26.4.1`` (``90 star``, ``8 binary``, ``8 astero``).
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import tomllib
from typing import Any, Literal

SuiteName = Literal["star", "binary", "astero"]
ReferenceStatus = Literal[
    "no_reference_yet",
    "interim_0b2e6bb",
    "mesa_26_4_1",
]

VALID_SUITES: tuple[str, ...] = ("star", "binary", "astero")
VALID_REFERENCE_STATUSES: tuple[str, ...] = (
    "no_reference_yet",
    "interim_0b2e6bb",
    "mesa_26_4_1",
)
_ISSUE_RE = re.compile(r"^#[1-9][0-9]*$")


def repo_root() -> pathlib.Path:
    """Return the repository root directory relative to this module.

    Returns:
        Resolved repository root ``pathlib.Path`` when running from a source
        checkout, or the current working directory otherwise.
    """
    candidate = pathlib.Path(__file__).resolve().parents[3]
    if (candidate / "benchmarks" / "history_harness").is_dir():
        return candidate
    return pathlib.Path.cwd()


def default_manifest_path() -> pathlib.Path:
    """Return the default ``case_manifest.toml`` path.

    Returns:
        Path to ``benchmarks/history_harness/case_manifest.toml``.
    """
    return repo_root() / "benchmarks" / "history_harness" / "case_manifest.toml"


def default_known_gaps_path() -> pathlib.Path:
    """Return the default ``known_gaps.toml`` path.

    Returns:
        Path to ``benchmarks/history_harness/known_gaps.toml``.
    """
    return repo_root() / "benchmarks" / "history_harness" / "known_gaps.toml"


@dataclasses.dataclass(frozen=True)
class DepartureSpec:
    """Documented intentional or pending departure attached to a case.

    Attributes:
        issue: Linked GitHub issue reference formatted as ``#<number>``.
        rationale: Brief explanation of the departure and upstream mechanism.
    """

    issue: str
    rationale: str


@dataclasses.dataclass(frozen=True)
class KnownGapSpec:
    """Declared known gap for a failing or incomplete upstream case.

    Attributes:
        suite: Upstream test suite (``"star"``, ``"binary"``, or ``"astero"``).
        case: Upstream test-case directory name.
        issue: Linked open GitHub issue formatted as ``#<number>``.
        reason: Concise statement of missing physics or divergence mechanism.
    """

    suite: SuiteName
    case: str
    issue: str
    reason: str


@dataclasses.dataclass(frozen=True)
class CaseManifestEntry:
    """Manifest specification for a single upstream MESA ``26.4.1`` case.

    Attributes:
        suite: Upstream test suite (``"star"``, ``"binary"``, or ``"astero"``).
        case: Upstream test-case directory name.
        reference_status: Availability and provenance of stored reference data
            (``"no_reference_yet"``, ``"interim_0b2e6bb"``, or
            ``"mesa_26_4_1"``).
        reference_path: Repository-relative path to the stored reference file
            (empty string when ``reference_status == "no_reference_yet"``).
        required_termination: Expected upstream termination code string (may
            contain ``|``-separated accepted alternatives when upstream sets
            ``required_termination_code_string(1..N)``).
        independent_var: Monotonic coordinate column name for track comparison
            (for example ``"star_age"`` in yr, ``"age"`` in yr, or
            ``"model_number"``).
        final_tolerances: Mapping from column name (or ``"rel_<xcol>"``) to
            maximum allowed absolute (or relative) difference at termination.
        track_tolerances: Mapping from column name to maximum allowed absolute
            deviation across the interpolated common track overlap.
        pr_ci: Whether the case belongs to the fast per-pull-request CI subset.
        uses_pulsation_scaffolding: True when the case currently relies on the
            in-house pulsation solver rather than ported GYRE or ADIPLS; such
            cases cannot count as passing under ``notes/SCOPE.md``.
        scaffolding_issue: Linked ``#<number>`` issue when
            ``uses_pulsation_scaffolding`` is True.
        uses_library_extras: True if a case dispatches per-case test hooks
            inside library crates rather than test-side ``extras_factory``
            callbacks (deprecated gate retained for manifest schema stability;
            ``False`` for all cases after ``#148``).
        library_extras_issue: Optional ``#<number>`` issue when
            ``uses_library_extras`` is True.
        extras_factory: Optional ``"module:callable"`` path returning test-side
            extras handlers for the case.
        departures: Tuple of documented ``DepartureSpec`` items.
    """

    suite: SuiteName
    case: str
    reference_status: ReferenceStatus
    reference_path: str
    required_termination: str
    independent_var: str
    final_tolerances: dict[str, float]
    track_tolerances: dict[str, float]
    pr_ci: bool = False
    uses_pulsation_scaffolding: bool = False
    scaffolding_issue: str = ""
    uses_library_extras: bool = False
    library_extras_issue: str = ""
    extras_factory: str = ""
    departures: tuple[DepartureSpec, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        """Return the ``(suite, case)`` lookup key."""
        return (self.suite, self.case)


@dataclasses.dataclass(frozen=True)
class HarnessManifest:
    """Validated collection of upstream MESA ``26.4.1`` case specifications.

    Attributes:
        mesa_release: Pinned upstream MESA release tag (``"26.4.1"``).
        zenodo_record: Zenodo archive identifier for the pinned release.
        cases: Tuple of ``CaseManifestEntry`` records in canonical order.
        extras_factory: Optional default ``"module:callable"`` spec resolving
            test-side extras handlers.
    """

    mesa_release: str
    zenodo_record: str
    cases: tuple[CaseManifestEntry, ...]
    extras_factory: str = ""

    def by_key(self) -> dict[tuple[str, str], CaseManifestEntry]:
        """Return a dictionary mapping ``(suite, case)`` to its entry.

        Returns:
            Dictionary keyed by ``(suite, case)``.
        """
        return {entry.key: entry for entry in self.cases}


def _validate_issue_ref(issue: str, context: str) -> str:
    cleaned = issue.strip()
    if not _ISSUE_RE.match(cleaned):
        msg = f"{context}: invalid issue reference {issue!r} (expected '#N')"
        raise ValueError(msg)
    return cleaned


def _parse_tol_map(
    raw: dict[str, Any] | None,
    base: dict[str, float],
    context: str,
) -> dict[str, float]:
    merged = dict(base)
    if raw is None:
        return merged
    for col, val in raw.items():
        fval = float(val)
        if fval <= 0.0:
            msg = f"{context}: tolerance for {col!r} must be positive ({fval})"
            raise ValueError(msg)
        merged[str(col)] = fval
    return merged


def load_case_manifest(
    path: pathlib.Path | str | None = None,
) -> HarnessManifest:
    """Load and validate ``case_manifest.toml``.

    Args:
        path: Optional path to ``case_manifest.toml``. Defaults to
            ``default_manifest_path()``.

    Returns:
        Validated ``HarnessManifest`` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any suite, status, tolerance, issue link, or duplicate
            case entry is invalid.
    """
    manifest_path = (
        default_manifest_path() if path is None else pathlib.Path(path)
    )
    with manifest_path.open("rb") as fh:
        data = tomllib.load(fh)

    meta = data.get("metadata", {})
    mesa_release = str(meta.get("mesa_release", ""))
    zenodo_record = str(meta.get("zenodo_record", ""))
    if not mesa_release:
        raise ValueError(f"{manifest_path}: missing metadata.mesa_release")

    defaults_raw = data.get("defaults", {})
    global_extras_factory = str(defaults_raw.get("extras_factory", "")).strip()
    suite_defaults: dict[
        str,
        tuple[str, dict[str, float], dict[str, float], bool, str, str],
    ] = {}
    for s_name in VALID_SUITES:
        s_block = defaults_raw.get(s_name, {})
        xcol = str(s_block.get("independent_var", "star_age"))
        ftol = _parse_tol_map(
            s_block.get("final_tolerances"),
            {},
            f"defaults.{s_name}.final_tolerances",
        )
        ttol = _parse_tol_map(
            s_block.get("track_tolerances"),
            {},
            f"defaults.{s_name}.track_tolerances",
        )
        def_lib_ext = bool(s_block.get("uses_library_extras", False))
        def_lib_iss = str(s_block.get("library_extras_issue", "")).strip()
        if def_lib_ext:
            def_lib_iss = _validate_issue_ref(
                def_lib_iss, f"defaults.{s_name}.library_extras_issue"
            )
        def_ext_factory = str(
            s_block.get("extras_factory", global_extras_factory)
        ).strip()
        suite_defaults[s_name] = (
            xcol,
            ftol,
            ttol,
            def_lib_ext,
            def_lib_iss,
            def_ext_factory,
        )

    raw_cases = data.get("cases", [])
    if not raw_cases:
        raise ValueError(f"{manifest_path}: 'cases' list cannot be empty")

    entries: list[CaseManifestEntry] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_cases:
        suite = str(raw.get("suite", "")).strip()
        case = str(raw.get("case", "")).strip()
        ctx = f"{suite}/{case}"
        if suite not in VALID_SUITES:
            raise ValueError(f"{ctx}: unknown suite {suite!r}")
        if not case:
            raise ValueError(f"{manifest_path}: empty case name in {raw!r}")
        if (suite, case) in seen:
            raise ValueError(f"{ctx}: duplicate manifest entry")
        seen.add((suite, case))

        ref_status = str(raw.get("reference_status", "")).strip()
        if ref_status not in VALID_REFERENCE_STATUSES:
            raise ValueError(f"{ctx}: invalid reference_status {ref_status!r}")
        ref_path = str(raw.get("reference_path", "")).strip()
        if ref_status == "no_reference_yet" and ref_path:
            raise ValueError(
                f"{ctx}: reference_path must be empty when status is "
                "no_reference_yet"
            )
        if ref_status != "no_reference_yet" and not ref_path:
            raise ValueError(
                f"{ctx}: reference_path required for status {ref_status!r}"
            )

        req_term = str(raw.get("required_termination", "")).strip()
        if not req_term:
            raise ValueError(f"{ctx}: required_termination cannot be empty")

        (
            def_x,
            def_ftol,
            def_ttol,
            def_lib_ext,
            def_lib_iss,
            def_ext_factory,
        ) = suite_defaults[suite]
        inherit_defaults = bool(raw.get("inherit_default_tolerances", True))
        base_ftol = def_ftol if inherit_defaults else {}
        base_ttol = def_ttol if inherit_defaults else {}
        xcol = str(raw.get("independent_var", def_x)).strip()
        ftol = _parse_tol_map(
            raw.get("final_tolerances"), base_ftol, f"{ctx}.final_tolerances"
        )
        ttol = _parse_tol_map(
            raw.get("track_tolerances"), base_ttol, f"{ctx}.track_tolerances"
        )

        uses_scaff = bool(raw.get("uses_pulsation_scaffolding", False))
        scaff_issue = str(raw.get("scaffolding_issue", "")).strip()
        if uses_scaff:
            scaff_issue = _validate_issue_ref(
                scaff_issue, f"{ctx}.scaffolding_issue"
            )

        uses_lib_ext = bool(raw.get("uses_library_extras", def_lib_ext))
        lib_ext_issue = str(
            raw.get("library_extras_issue", def_lib_iss)
        ).strip()
        if uses_lib_ext:
            lib_ext_issue = _validate_issue_ref(
                lib_ext_issue, f"{ctx}.library_extras_issue"
            )

        deps: list[DepartureSpec] = []
        for dep in raw.get("departures", []):
            d_issue = _validate_issue_ref(
                str(dep.get("issue", "")), f"{ctx}.departures"
            )
            d_rat = str(dep.get("rationale", "")).strip()
            if not d_rat:
                raise ValueError(f"{ctx}: departure {d_issue} lacks rationale")
            deps.append(DepartureSpec(issue=d_issue, rationale=d_rat))

        extras_factory = str(raw.get("extras_factory", def_ext_factory)).strip()

        entries.append(
            CaseManifestEntry(
                suite=suite,  # type: ignore[arg-type]
                case=case,
                reference_status=ref_status,  # type: ignore[arg-type]
                reference_path=ref_path,
                required_termination=req_term,
                independent_var=xcol,
                final_tolerances=ftol,
                track_tolerances=ttol,
                pr_ci=bool(raw.get("pr_ci", False)),
                uses_pulsation_scaffolding=uses_scaff,
                scaffolding_issue=scaff_issue,
                uses_library_extras=uses_lib_ext,
                library_extras_issue=lib_ext_issue,
                extras_factory=extras_factory,
                departures=tuple(deps),
            )
        )

    return HarnessManifest(
        mesa_release=mesa_release,
        zenodo_record=zenodo_record,
        cases=tuple(entries),
        extras_factory=global_extras_factory,
    )


def load_known_gaps(
    path: pathlib.Path | str | None = None,
) -> dict[tuple[str, str], KnownGapSpec]:
    """Load and validate ``known_gaps.toml``.

    Args:
        path: Optional path to ``known_gaps.toml``. Defaults to
            ``default_known_gaps_path()``.

    Returns:
        Dictionary mapping ``(suite, case)`` to its ``KnownGapSpec``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any gap lacks a valid ``#<number>`` issue link or reason.
    """
    gaps_path = (
        default_known_gaps_path() if path is None else pathlib.Path(path)
    )
    with gaps_path.open("rb") as fh:
        data = tomllib.load(fh)

    out: dict[tuple[str, str], KnownGapSpec] = {}
    for raw in data.get("gaps", []):
        suite = str(raw.get("suite", "")).strip()
        case = str(raw.get("case", "")).strip()
        ctx = f"known_gaps:{suite}/{case}"
        if suite not in VALID_SUITES or not case:
            raise ValueError(f"{ctx}: invalid suite or case")
        if (suite, case) in out:
            raise ValueError(f"{ctx}: duplicate known gap entry")
        issue = _validate_issue_ref(str(raw.get("issue", "")), ctx)
        reason = str(raw.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"{ctx}: reason cannot be empty")
        out[(suite, case)] = KnownGapSpec(
            suite=suite,  # type: ignore[arg-type]
            case=case,
            issue=issue,
            reason=reason,
        )
    return out


def _matches_single_termination(req: str, obs: str) -> bool:
    if not req or not obs or obs == "max_steps_cap":
        return False
    if obs == req:
        return True
    if obs.startswith(f"{req}(") and obs.endswith(")"):
        return True
    if "_term=" in obs:
        tail = obs.rsplit("_term=", 1)[1]
        return _matches_single_termination(req, tail)
    if req in ("xtra1", "t_xtra1"):
        return obs in ("xtra1", "t_xtra1")
    return False


def termination_matches(required: str, observed: str) -> bool:
    """Check whether ``observed`` termination satisfies ``required``.

    Supports ``|``-delimited multi-code specifications (corresponding to MESA
    ``required_termination_code_string(1..N)``), species-parameterized codes
    such as ``xa_central_lower_limit(he4)``, and ``astero`` summary suffixes
    ending in ``_term=<code>`` while keeping ``extras_finish_step`` and
    ``extras_check_model`` distinct.

    Args:
        required: Expected upstream termination code(s) from the manifest.
        observed: Observed termination code returned by the tracer.

    Returns:
        True if ``observed`` matches at least one alternative in ``required``
        and is not ``"max_steps_cap"``.
    """
    obs = observed.strip()
    if not obs or obs == "max_steps_cap":
        return False
    for alt in required.split("|"):
        if _matches_single_termination(alt.strip(), obs):
            return True
    return False
