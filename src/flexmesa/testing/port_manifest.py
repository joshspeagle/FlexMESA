# SPDX-License-Identifier: GPL-3.0-or-later
"""Port manifest schema, loader, serializer, and CI validator (`#174`).

Every upstream source file in MESA release ``26.4.1`` (plus bundled ``GYRE``,
``ADIPLS``, ``STELLA``, and ``pgstar``/``pgbinary`` plotting) has one entry
recording its exact upstream line count, ``SHA-256`` digest, implementing
``FlexMESA`` files, status tier, public-routine mapping, and verification
evidence links.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import tomllib

from flexmesa.data import fetch

_UPSTREAM_REF = fetch.load_upstream_reference()
EXPECTED_UPSTREAM_VERSION: str = _UPSTREAM_REF.version_number
EXPECTED_UPSTREAM_COMMIT: str = _UPSTREAM_REF.git_commit
EXPECTED_ARCHIVE_MD5: str = _UPSTREAM_REF.archive_md5

MODULE_ORDER: tuple[str, ...] = (
    "star",
    "star_data",
    "binary",
    "astero",
    "eos",
    "kap",
    "net",
    "rates",
    "neu",
    "atm",
    "turb",
    "chem",
    "ionization",
    "colors",
    "num",
    "mtx",
    "interp_1d",
    "interp_2d",
    "auto_diff",
    "math",
    "const",
    "utils",
    "gyre",
    "adipls",
    "stella",
    "plotting",
)

CORE_ENGINE_MODULES: frozenset[str] = frozenset(
    {"star", "star_data", "binary", "astero"}
)
BUNDLED_MODULES: frozenset[str] = frozenset({"gyre", "adipls", "stella"})


class PortStatus(enum.StrEnum):
    """Ordered port and verification status tiers for an upstream file."""

    ABSENT = "absent"
    STUB = "stub"
    PARTIAL = "partial"
    PORTED_UNVERIFIED = "ported_unverified"
    KERNEL_REFERENCE_VECTORS = "kernel_reference_vectors"
    CASE_LEVEL_EVIDENCE = "case_level_evidence"

    @property
    def rank(self) -> int:
        """Integer promotion rank (0 = ``absent`` .. 5 = ``case``)."""
        order = (
            PortStatus.ABSENT,
            PortStatus.STUB,
            PortStatus.PARTIAL,
            PortStatus.PORTED_UNVERIFIED,
            PortStatus.KERNEL_REFERENCE_VECTORS,
            PortStatus.CASE_LEVEL_EVIDENCE,
        )
        return order.index(self)

    @property
    def is_ported(self) -> bool:
        """Whether status is at least ``ported_unverified``."""
        return self.rank >= PortStatus.PORTED_UNVERIFIED.rank

    @property
    def is_verified(self) -> bool:
        """Whether status is backed by committed MESA 26.4.1 evidence."""
        return self.rank >= PortStatus.KERNEL_REFERENCE_VECTORS.rank

    @property
    def label(self) -> str:
        """Human-readable dashboard label for this status."""
        labels = {
            PortStatus.ABSENT: "absent",
            PortStatus.STUB: "stub",
            PortStatus.PARTIAL: "partial",
            PortStatus.PORTED_UNVERIFIED: "ported (unverified)",
            PortStatus.KERNEL_REFERENCE_VECTORS: (
                "kernel matches MESA reference vectors"
            ),
            PortStatus.CASE_LEVEL_EVIDENCE: "case-level evidence",
        }
        return labels[self]


@dataclasses.dataclass(frozen=True)
class PortManifestEntry:
    """Single upstream MESA 26.4.1 source-file port record.

    Attributes:
        upstream_path: Relative POSIX path inside MESA 26.4.1.
        module: Canonical dashboard module name in ``MODULE_ORDER``.
        upstream_lines: Exact line count in MESA 26.4.1.
        upstream_sha256: Lowercase hexadecimal SHA-256 of the upstream file.
        status: Port and verification status tier.
        flexmesa_files: Implementing repository-relative source paths.
        routine_map: Mapping from upstream public Fortran routine name to
            ``<flexmesa_file>::<symbol>[+<symbol>...]``.
        evidence_artifacts: Committed generated reference/comparison artifacts.
        evidence_tests: Test files exercising ``evidence_artifacts``.
        harness_cases: Passing ``<suite>/<case>`` IDs for case-level evidence.
        notes: Provenance note (e.g. ``coverage-map.md`` or audit finding ID).
    """

    upstream_path: str
    module: str
    upstream_lines: int
    upstream_sha256: str
    status: PortStatus
    flexmesa_files: tuple[str, ...] = ()
    routine_map: tuple[tuple[str, str], ...] = ()
    evidence_artifacts: tuple[str, ...] = ()
    evidence_tests: tuple[str, ...] = ()
    harness_cases: tuple[str, ...] = ()
    notes: str = ""

    def routine_dict(self) -> dict[str, str]:
        """Return ``routine_map`` as a dictionary."""
        return dict(self.routine_map)


@dataclasses.dataclass(frozen=True)
class PortManifest:
    """Complete upstream MESA 26.4.1 port manifest.

    Attributes:
        schema_version: Manifest schema version integer.
        upstream_version: Pinned MESA release string (``r26.4.1``).
        upstream_commit: Pinned MESA git commit hash.
        archive_md5: MD5 checksum of ``mesa-26.04.1.zip``.
        entries: Tuple of per-file ``PortManifestEntry`` records.
    """

    schema_version: int
    upstream_version: str
    upstream_commit: str
    archive_md5: str
    entries: tuple[PortManifestEntry, ...]

    def by_upstream_path(self) -> dict[str, PortManifestEntry]:
        """Index entries by ``upstream_path``."""
        return {entry.upstream_path: entry for entry in self.entries}


def parse_port_manifest_text(
    text: str, source_name: str = "<memory>"
) -> PortManifest:
    """Parse ``port_manifest.toml`` TOML text into a ``PortManifest``.

    Args:
        text: UTF-8 TOML text of a port manifest.
        source_name: Label included in validation error messages.

    Returns:
        Parsed ``PortManifest`` instance.

    Raises:
        ValueError: If required schema fields or status values are invalid.
    """
    raw = tomllib.loads(text)
    meta = raw.get("metadata", {})
    files_raw = raw.get("files", [])
    if not isinstance(files_raw, list) or not files_raw:
        raise ValueError(f"{source_name}: expected non-empty [[files]] array")

    entries: list[PortManifestEntry] = []
    for item in files_raw:
        rmap_raw = item.get("routine_map", {})
        if not isinstance(rmap_raw, dict):
            raise ValueError(
                f"{item.get('upstream_path')}: routine_map must be a table"
            )
        rmap = tuple(sorted((str(k), str(v)) for k, v in rmap_raw.items()))
        entries.append(
            PortManifestEntry(
                upstream_path=str(item["upstream_path"]),
                module=str(item["module"]),
                upstream_lines=int(item["upstream_lines"]),
                upstream_sha256=str(item["upstream_sha256"]),
                status=PortStatus(str(item["status"])),
                flexmesa_files=tuple(
                    str(x) for x in item.get("flexmesa_files", [])
                ),
                routine_map=rmap,
                evidence_artifacts=tuple(
                    str(x) for x in item.get("evidence_artifacts", [])
                ),
                evidence_tests=tuple(
                    str(x) for x in item.get("evidence_tests", [])
                ),
                harness_cases=tuple(
                    str(x) for x in item.get("harness_cases", [])
                ),
                notes=str(item.get("notes", "")),
            )
        )

    return PortManifest(
        schema_version=int(meta.get("schema_version", 1)),
        upstream_version=str(meta.get("upstream_version", "")),
        upstream_commit=str(meta.get("upstream_commit", "")),
        archive_md5=str(meta.get("archive_md5", "")),
        entries=tuple(entries),
    )


def load_port_manifest(path: pathlib.Path) -> PortManifest:
    """Load and parse ``port_manifest.toml`` from ``path``.

    Args:
        path: Path to a TOML port manifest file.

    Returns:
        Parsed ``PortManifest`` instance.
    """
    return parse_port_manifest_text(path.read_text(encoding="utf-8"), str(path))


def _toml_str(val: str) -> str:
    return json.dumps(val, ensure_ascii=False)


def _toml_str_array(items: tuple[str, ...]) -> str:
    joined = ", ".join(_toml_str(x) for x in items)
    return f"[{joined}]"


def dump_port_manifest(manifest: PortManifest) -> str:
    """Serialize ``manifest`` into deterministic canonical TOML text.

    Args:
        manifest: ``PortManifest`` instance to format.

    Returns:
        Canonical UTF-8 TOML string.
    """
    lines: list[str] = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# GENERATED VIA scripts/dashboard/build_port_manifest.py (#174).",
        "# Line counts and SHA-256 digests are computed from MESA 26.4.1.",
        "",
        "[metadata]",
        f"schema_version = {manifest.schema_version}",
        f"upstream_version = {_toml_str(manifest.upstream_version)}",
        f"upstream_commit = {_toml_str(manifest.upstream_commit)}",
        f"archive_md5 = {_toml_str(manifest.archive_md5)}",
        "",
    ]
    for entry in manifest.entries:
        lines.append("[[files]]")
        lines.append(f"upstream_path = {_toml_str(entry.upstream_path)}")
        lines.append(f"module = {_toml_str(entry.module)}")
        lines.append(f"upstream_lines = {entry.upstream_lines}")
        lines.append(f"upstream_sha256 = {_toml_str(entry.upstream_sha256)}")
        lines.append(f"status = {_toml_str(entry.status.value)}")
        if entry.flexmesa_files:
            lines.append(
                f"flexmesa_files = {_toml_str_array(entry.flexmesa_files)}"
            )
        if entry.evidence_artifacts:
            lines.append(
                "evidence_artifacts = "
                f"{_toml_str_array(entry.evidence_artifacts)}"
            )
        if entry.evidence_tests:
            lines.append(
                f"evidence_tests = {_toml_str_array(entry.evidence_tests)}"
            )
        if entry.harness_cases:
            lines.append(
                f"harness_cases = {_toml_str_array(entry.harness_cases)}"
            )
        if entry.notes:
            lines.append(f"notes = {_toml_str(entry.notes)}")
        if entry.routine_map:
            lines.append("[files.routine_map]")
            for k, v in entry.routine_map:
                lines.append(f"{_toml_str(k)} = {_toml_str(v)}")
        lines.append("")
    return "\n".join(lines)


def verify_port_manifest(
    manifest: PortManifest,
    repo_root: pathlib.Path,
    mesa_dir: pathlib.Path | None = None,
    baseline_harness_json: pathlib.Path | None = None,
    base_manifest: PortManifest | None = None,
    changed_paths: frozenset[str] | None = None,
) -> list[str]:
    """Validate ``manifest`` schema, line counts, routines, evidence, and diffs.

    Args:
        manifest: Loaded ``PortManifest`` under check.
        repo_root: Repository checkout root directory.
        mesa_dir: Optional MESA 26.4.1 source/test-inputs directory for live
            line-count, SHA-256, and Fortran public-routine verification.
        baseline_harness_json: Optional path to ``baseline_run_26_4_1.json``
            used to verify ``case_level_evidence`` passing cases.
        base_manifest: Optional base-branch ``PortManifest`` for PR promotion
            and upstream-digest immutability checks.
        changed_paths: Repository-relative paths changed in ``base...HEAD``.

    Returns:
        List of human-readable validation error strings (empty when valid).
    """
    from flexmesa.testing import _port_manifest_verify

    return _port_manifest_verify.verify_port_manifest_impl(
        manifest=manifest,
        repo_root=repo_root,
        mesa_dir=mesa_dir,
        baseline_harness_json=baseline_harness_json,
        base_manifest=base_manifest,
        changed_paths=changed_paths,
    )
