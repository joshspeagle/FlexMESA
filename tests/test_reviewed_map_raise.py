# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the reviewed routine-map raise exception (`DECISIONS #79`).

A pull request may raise a ``port_manifest.toml`` entry from ``absent`` to
``partial`` without changing its ``flexmesa_files`` when every mapped routine
passes ``routine_gate`` and an independent review record changed in the pull
request lists the file. Each test builds a one-entry manifest and a small
repository root (routine inventory, one Rust file, review texts) in
``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

from flexmesa.testing import _port_manifest_verify, port_manifest

UPSTREAM = "star/private/demo.f90"
RUST = "crates/flexmesa-demo/src/lib.rs"
RECORD = "notes/audit/routine_map_reviews/demo.toml"
MANIFEST = "benchmarks/dashboard/port_manifest.toml"
GOOD_MAP = (
    ("alpha", f"{RUST}::alpha"),
    ("beta", f"{RUST}::Demo::beta"),
)
GOOD_RECORD = (
    "schema_version = 1\n"
    'review = "notes/audit/routine_map_reviews/demo.md"\n'
    "seed = 7\n"
    "sample_full = 2\n"
    "sample_partial = 0\n"
    "over_claims = 0\n"
    "under_claims = 0\n"
    "wrong_target = 0\n"
    "not_ported = 0\n"
    f'files = ["{UPSTREAM}"]\n'
)


def _repo(tmp_path: pathlib.Path, record: str | None) -> pathlib.Path:
    """Write the inventory, Rust file, review text and optional record."""
    inv = tmp_path / "benchmarks" / "dashboard" / "routine_inventory.json"
    inv.parent.mkdir(parents=True)
    inv.write_text(
        json.dumps(
            {
                "files": {
                    UPSTREAM: {
                        "routines": [
                            ["alpha", "subroutine", "", 1, 10],
                            ["beta", "function", "", 11, 20],
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rust = tmp_path / RUST
    rust.parent.mkdir(parents=True)
    rust.write_text(
        "pub fn alpha() {}\n"
        "pub struct Demo;\n"
        "impl Demo {\n"
        "    pub fn beta(&self) {}\n"
        "}\n",
        encoding="utf-8",
    )
    reviews = tmp_path / "notes" / "audit" / "routine_map_reviews"
    reviews.mkdir(parents=True)
    (reviews / "demo.md").write_text("review text\n", encoding="utf-8")
    if record is not None:
        (tmp_path / RECORD).write_text(record, encoding="utf-8")
    return tmp_path


def _manifest(
    status: port_manifest.PortStatus,
    routine_map: tuple[tuple[str, str], ...] = (),
) -> port_manifest.PortManifest:
    """Return a one-entry manifest for ``UPSTREAM`` at ``status``."""
    entry = port_manifest.PortManifestEntry(
        upstream_path=UPSTREAM,
        module="star",
        upstream_lines=20,
        upstream_sha256="0" * 64,
        status=status,
        flexmesa_files=(
            () if status == port_manifest.PortStatus.ABSENT else (RUST,)
        ),
        routine_map=routine_map,
    )
    return port_manifest.PortManifest(
        schema_version=1,
        upstream_version="r26.4.1",
        upstream_commit="c8c2026",
        archive_md5="0" * 32,
        entries=(entry,),
    )


def _raise_errors(
    repo: pathlib.Path,
    changed: set[str],
    status: port_manifest.PortStatus = port_manifest.PortStatus.PARTIAL,
    routine_map: tuple[tuple[str, str], ...] = GOOD_MAP,
) -> list[str]:
    """Run the raise check for ``absent`` -> ``status`` on the fixture."""
    base = _manifest(port_manifest.PortStatus.ABSENT)
    new = _manifest(status, routine_map)
    errors: list[str] = []
    _port_manifest_verify._verify_base_diff_and_immutability(
        new,
        base,
        {UPSTREAM},
        frozenset(changed),
        repo,
        errors,
    )
    return errors


def test_partial_raise_without_record_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """No review record and no code change: the raise needs code."""
    errs = _raise_errors(_repo(tmp_path, None), {MANIFEST})
    assert len(errs) == 1
    assert "not listed in a review record" in errs[0]


def test_partial_raise_with_record_listing_file_is_allowed(
    tmp_path: pathlib.Path,
) -> None:
    """A valid changed record listing the file evidences the raise."""
    errs = _raise_errors(_repo(tmp_path, GOOD_RECORD), {MANIFEST, RECORD})
    assert errs == []


def test_partial_raise_with_record_not_listing_file_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """A record that lists other files does not cover this one."""
    record = GOOD_RECORD.replace(UPSTREAM, "star/private/other.f90")
    errs = _raise_errors(_repo(tmp_path, record), {MANIFEST, RECORD})
    assert len(errs) == 1
    assert "not listed in a review record" in errs[0]


def test_malformed_record_is_reported_and_does_not_cover(
    tmp_path: pathlib.Path,
) -> None:
    """A record missing its integer keys is an error and covers nothing."""
    record = 'seed = "seven"\nfiles = ["' + UPSTREAM + '"]\n'
    errs = _raise_errors(_repo(tmp_path, record), {MANIFEST, RECORD})
    assert any("review record lacks integer keys" in e for e in errs)
    assert any("not listed in a review record" in e for e in errs)


def test_record_not_changed_in_pull_request_does_not_cover(
    tmp_path: pathlib.Path,
) -> None:
    """A committed record counts only when the pull request changes it."""
    errs = _raise_errors(_repo(tmp_path, GOOD_RECORD), {MANIFEST})
    assert len(errs) == 1
    assert "not listed in a review record" in errs[0]


def test_partial_raise_with_code_change_needs_no_record(
    tmp_path: pathlib.Path,
) -> None:
    """Changing an implementing file satisfies the original rule."""
    errs = _raise_errors(_repo(tmp_path, None), {MANIFEST, RUST})
    assert errs == []


def test_partial_raise_with_failing_routine_gate_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """A mapped target missing from the Rust file voids the exception."""
    broken = (GOOD_MAP[0], ("beta", f"{RUST}::gamma"))
    errs = _raise_errors(
        _repo(tmp_path, GOOD_RECORD),
        {MANIFEST, RECORD},
        routine_map=broken,
    )
    assert len(errs) == 1
    assert "routine_map fails routine_gate" in errs[0]


def test_partial_raise_without_routine_map_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """A record cannot evidence a file with nothing mapped."""
    errs = _raise_errors(
        _repo(tmp_path, GOOD_RECORD), {MANIFEST, RECORD}, routine_map=()
    )
    assert len(errs) == 1
    assert "no routine_map" in errs[0]


def test_raise_above_partial_with_record_still_needs_code(
    tmp_path: pathlib.Path,
) -> None:
    """The record never evidences a raise to ``ported_unverified``."""
    errs = _raise_errors(
        _repo(tmp_path, GOOD_RECORD),
        {MANIFEST, RECORD},
        status=port_manifest.PortStatus.PORTED_UNVERIFIED,
    )
    assert len(errs) == 1
    assert "absent to ported_unverified without modifying" in errs[0]


def test_lowering_a_reviewed_partial_back_to_absent_is_allowed(
    tmp_path: pathlib.Path,
) -> None:
    """Only raises are gated: lowering ``partial`` needs no evidence."""
    base = _manifest(port_manifest.PortStatus.PARTIAL, GOOD_MAP)
    new = dataclasses.replace(
        base,
        entries=(
            dataclasses.replace(
                base.entries[0],
                status=port_manifest.PortStatus.ABSENT,
                flexmesa_files=(),
                routine_map=(),
            ),
        ),
    )
    errors: list[str] = []
    _port_manifest_verify._verify_base_diff_and_immutability(
        new,
        base,
        {UPSTREAM},
        frozenset({MANIFEST}),
        _repo(tmp_path, None),
        errors,
    )
    assert errors == []
