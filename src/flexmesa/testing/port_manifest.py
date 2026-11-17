# SPDX-License-Identifier: LGPL-3.0-or-later
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
import hashlib
import json
import pathlib
import tomllib
from typing import Any

from flexmesa.testing import _port_parsers

EXPECTED_UPSTREAM_VERSION = "r26.4.1"
EXPECTED_UPSTREAM_COMMIT = "c8c202662922a22b3efefa824473ad615c623afd"
EXPECTED_ARCHIVE_MD5 = "994fd28f38da92efc80e38ca11d8317b"

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
        """Return integer promotion rank (0 = ``absent`` .. 5 = ``case``)."""
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
        """Return ``True`` when at least ``ported_unverified``."""
        return self.rank >= PortStatus.PORTED_UNVERIFIED.rank

    @property
    def is_verified(self) -> bool:
        """Return ``True`` when backed by committed MESA 26.4.1 evidence."""
        return self.rank >= PortStatus.KERNEL_REFERENCE_VECTORS.rank

    @property
    def label(self) -> str:
        """Return the human-readable dashboard label for this status."""
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


def load_port_manifest(path: pathlib.Path) -> PortManifest:
    """Load and parse ``port_manifest.toml`` from ``path``.

    Args:
        path: Path to a TOML port manifest file.

    Returns:
        Parsed ``PortManifest`` instance.

    Raises:
        ValueError: If required schema fields or status values are invalid.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = raw.get("metadata", {})
    files_raw = raw.get("files", [])
    if not isinstance(files_raw, list) or not files_raw:
        raise ValueError(f"{path}: expected non-empty [[files]] array")

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


def _toml_str(val: str) -> str:
    return json.dumps(val, ensure_ascii=False)


def _toml_str_array(items: tuple[str, ...]) -> str:
    return "[" + ", ".join(_toml_str(x) for x in items) + "]"


def dump_port_manifest(manifest: PortManifest) -> str:
    """Serialize ``manifest`` into deterministic canonical TOML text.

    Args:
        manifest: ``PortManifest`` instance to format.

    Returns:
        Canonical UTF-8 TOML string.
    """
    lines: list[str] = [
        "# SPDX-License-Identifier: LGPL-3.0-or-later",
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


def _artifact_has_provenance(path: pathlib.Path) -> bool:
    """Confirm a committed evidence artifact carries MESA 26.4.1 provenance."""
    head = path.read_bytes()[:8192].decode("utf-8", errors="replace")
    markers = ("r26.4.1", "26.4.1", "mesa_26_4_1", "GENERATED BY")
    return any(tok in head for tok in markers)


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
            checks.
        changed_paths: Repository-relative paths changed in ``base...HEAD``.

    Returns:
        List of human-readable validation error strings (empty when valid).
    """
    errors: list[str] = []
    if manifest.upstream_version != EXPECTED_UPSTREAM_VERSION:
        errors.append(
            f"upstream_version={manifest.upstream_version!r} != "
            f"{EXPECTED_UPSTREAM_VERSION!r}"
        )
    if manifest.upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        errors.append(
            f"upstream_commit={manifest.upstream_commit!r} != "
            f"{EXPECTED_UPSTREAM_COMMIT!r}"
        )
    if manifest.archive_md5 != EXPECTED_ARCHIVE_MD5:
        errors.append(
            f"archive_md5={manifest.archive_md5!r} != {EXPECTED_ARCHIVE_MD5!r}"
        )

    valid_modules = frozenset(MODULE_ORDER)
    seen_paths: set[str] = set()
    sym_cache: dict[str, set[str]] = {}

    passing_cases: set[str] = set()
    if baseline_harness_json is not None and baseline_harness_json.is_file():
        payload: dict[str, Any] = json.loads(
            baseline_harness_json.read_text(encoding="utf-8")
        )
        for rec in payload.get("cases", payload.get("results", [])):
            if (
                rec.get("verdict") == "pass"
                and rec.get("reference_status") == "mesa_26_4_1"
            ):
                passing_cases.add(f"{rec.get('suite')}/{rec.get('case')}")

    paths_list = [e.upstream_path for e in manifest.entries]
    if paths_list != sorted(paths_list):
        errors.append("[[files]] entries must be sorted by upstream_path")

    for entry in manifest.entries:
        up = entry.upstream_path
        if up in seen_paths:
            errors.append(f"{up}: duplicate upstream_path")
        seen_paths.add(up)

        if entry.module not in valid_modules:
            errors.append(f"{up}: unknown module {entry.module!r}")
        expected_mod = _port_parsers.classify_upstream_relpath(up)
        if expected_mod is not None and entry.module != expected_mod:
            errors.append(
                f"{up}: module={entry.module!r} != classified {expected_mod!r}"
            )
        if entry.upstream_lines < 0:
            errors.append(f"{up}: negative upstream_lines")

        if entry.status == PortStatus.ABSENT:
            if entry.flexmesa_files:
                errors.append(
                    f"{up}: absent entry must not list flexmesa_files"
                )
        else:
            if not entry.flexmesa_files:
                errors.append(
                    f"{up}: status={entry.status.value} requires flexmesa_files"
                )
            for rel_fm in entry.flexmesa_files:
                if not rel_fm.startswith(("crates/", "src/flexmesa/")):
                    errors.append(
                        f"{up}: flexmesa_file {rel_fm!r} must be in "
                        "crates/ or src/flexmesa/"
                    )
                fm_path = repo_root / rel_fm
                if not fm_path.is_file():
                    errors.append(f"{up}: missing flexmesa_file {rel_fm!r}")

        if mesa_dir is not None:
            up_file = mesa_dir / up
            if up_file.is_file():
                raw_bytes = up_file.read_bytes()
                actual_lines = len(raw_bytes.splitlines())
                actual_sha = hashlib.sha256(raw_bytes).hexdigest()
                if actual_lines != entry.upstream_lines:
                    errors.append(
                        f"{up}: upstream_lines={entry.upstream_lines} != "
                        f"actual {actual_lines}"
                    )
                if actual_sha != entry.upstream_sha256:
                    errors.append(
                        f"{up}: upstream_sha256 mismatch vs {mesa_dir.name}"
                    )
                if entry.status.is_ported:
                    pubs = _port_parsers.extract_fortran_public_routines(
                        raw_bytes.decode("utf-8", errors="replace"),
                        suffix=up_file.suffix,
                    )
                    rmap = entry.routine_dict()
                    for proc in pubs:
                        if proc not in rmap:
                            errors.append(
                                f"{up}: status={entry.status.value} missing "
                                f"public routine {proc!r} in routine_map"
                            )

        if entry.status.is_ported:
            fm_set = frozenset(entry.flexmesa_files)
            for proc, target in entry.routine_map:
                if "::" not in target:
                    errors.append(
                        f"{up}: routine_map[{proc!r}]={target!r} must use "
                        "'<flexmesa_file>::<symbol>'"
                    )
                    continue
                rel_fm, syms_raw = target.split("::", 1)
                if rel_fm not in fm_set:
                    errors.append(
                        f"{up}: routine_map[{proc!r}] file {rel_fm!r} not in "
                        "flexmesa_files"
                    )
                    continue
                fm_path = repo_root / rel_fm
                if fm_path.is_file():
                    if rel_fm not in sym_cache:
                        sym_cache[rel_fm] = (
                            _port_parsers.extract_flexmesa_symbols(fm_path)
                        )
                    avail = sym_cache[rel_fm]
                    for sym in syms_raw.split("+"):
                        clean_sym = sym.strip()
                        if not clean_sym or clean_sym not in avail:
                            errors.append(
                                f"{up}: symbol {clean_sym!r} not found in "
                                f"{rel_fm}"
                            )

        if not entry.status.is_verified:
            if entry.evidence_artifacts or entry.harness_cases:
                errors.append(
                    f"{up}: unverified status={entry.status.value} must not "
                    "populate evidence_artifacts or harness_cases"
                )
        else:
            if not entry.evidence_artifacts:
                errors.append(
                    f"{up}: verified status={entry.status.value} requires "
                    "non-empty evidence_artifacts"
                )
            for art_rel in entry.evidence_artifacts:
                if not (
                    art_rel.startswith(("benchmarks/", "tests/fixtures/"))
                    or art_rel == "notes/audit/case_table.md"
                ):
                    errors.append(
                        f"{up}: evidence_artifact {art_rel!r} must reside in "
                        "benchmarks/, tests/fixtures/, or "
                        "notes/audit/case_table.md"
                    )
                art_path = repo_root / art_rel
                if not art_path.is_file():
                    errors.append(
                        f"{up}: missing evidence_artifact {art_rel!r}"
                    )
                elif not _artifact_has_provenance(art_path):
                    errors.append(
                        f"{up}: evidence_artifact {art_rel!r} lacks MESA "
                        "26.4.1 / GENERATED provenance header"
                    )
            for test_rel in entry.evidence_tests:
                if not (repo_root / test_rel).is_file():
                    errors.append(f"{up}: missing evidence_test {test_rel!r}")

            if entry.status == PortStatus.CASE_LEVEL_EVIDENCE:
                if not entry.harness_cases:
                    errors.append(
                        f"{up}: case_level_evidence requires harness_cases"
                    )
                for cid in entry.harness_cases:
                    if cid not in passing_cases:
                        errors.append(
                            f"{up}: harness_case {cid!r} is not passing "
                            "against mesa_26_4_1 in baseline_run_26_4_1.json"
                        )

    if mesa_dir is not None and mesa_dir.is_dir():
        for disk_path in sorted(mesa_dir.rglob("*")):
            if not disk_path.is_file():
                continue
            rel = disk_path.relative_to(mesa_dir).as_posix()
            if _port_parsers.classify_upstream_relpath(rel) is not None:
                if rel not in seen_paths:
                    errors.append(
                        f"{rel}: upstream MESA 26.4.1 file missing from "
                        "port_manifest.toml"
                    )

    if base_manifest is not None and changed_paths is not None:
        base_by_path = base_manifest.by_upstream_path()
        for entry in manifest.entries:
            prev = base_by_path.get(entry.upstream_path)
            if prev is None:
                continue
            if entry.status.rank <= prev.status.rank:
                continue
            if entry.status.is_verified:
                evidence_gate = (
                    set(entry.evidence_artifacts)
                    | set(entry.evidence_tests)
                    | {"benchmarks/history_harness/baseline_run_26_4_1.json"}
                )
                if not (changed_paths & evidence_gate):
                    errors.append(
                        f"{entry.upstream_path}: status raised from "
                        f"{prev.status.value} to {entry.status.value} without "
                        "adding or updating evidence artifacts/tests in PR"
                    )
            else:
                impl_gate = set(entry.flexmesa_files)
                if not (changed_paths & impl_gate):
                    errors.append(
                        f"{entry.upstream_path}: status raised from "
                        f"{prev.status.value} to {entry.status.value} without "
                        "modifying implementing flexmesa_files in PR"
                    )

    return errors
