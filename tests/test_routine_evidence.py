# SPDX-License-Identifier: GPL-3.0-or-later
"""Oracle evidence, status ceiling, and backlog attribution (`#611`)."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from flexmesa.testing import (
    port_manifest,
    routine_backlog,
    routine_inventory,
    routine_table,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
S = port_manifest.PortStatus


@pytest.mark.parametrize(
    ("routines", "mapped", "oracle", "expected"),
    [
        (0, 0, 0, None),
        (4, 0, 0, None),
        (4, 1, 1, S.PARTIAL),
        (4, 3, 3, S.PARTIAL),
        (4, 4, 0, S.PORTED_UNVERIFIED),
        (4, 4, 3, S.PORTED_UNVERIFIED),
        (4, 4, 4, S.KERNEL_REFERENCE_VECTORS),
    ],
)
def test_status_ceiling_follows_the_documented_rule(
    routines: int, mapped: int, oracle: int, expected: S | None
) -> None:
    """None mapped gives no claim; all mapped and all oracle gives vectors."""
    assert routine_table.status_ceiling(routines, mapped, oracle) == expected


def _mini(
    tmp_path: pathlib.Path, status: S, keys: tuple[str, ...]
) -> tuple[port_manifest.PortManifest, dict[str, Any]]:
    (tmp_path / "crates").mkdir(exist_ok=True)
    (tmp_path / "crates" / "a.rs").write_text(
        "pub fn f_one() {}\npub fn f_two() {}\n", encoding="utf-8"
    )
    entry = port_manifest.PortManifestEntry(
        upstream_path="star/private/x.f90",
        module="star",
        upstream_lines=30,
        upstream_sha256="0" * 64,
        status=status,
        flexmesa_files=("crates/a.rs",),
        routine_map=tuple(
            (k, f"crates/a.rs::f_{n}")
            for k, n in zip(keys, ("one", "two"), strict=False)
        ),
    )
    manifest = port_manifest.PortManifest(
        schema_version=1,
        upstream_version="r26.4.1",
        upstream_commit="c",
        archive_md5="m",
        entries=(entry,),
    )
    inv = {
        "files": {
            "star/private/x.f90": {
                "lines": 30,
                "outside": 10,
                "routines": [
                    ["One", "subroutine", "", 1, 10],
                    ["Two", "subroutine", "", 11, 20],
                ],
            }
        }
    }
    return manifest, inv


def _row(table: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = table["files"][0]
    return row


def test_oracle_level_needs_a_listed_routine_id_and_a_mapped_target(
    tmp_path: pathlib.Path,
) -> None:
    """Only listed ids of mapped routines count; lines use the span."""
    manifest, inv = _mini(tmp_path, S.PARTIAL, ("One", "Two"))
    ids = {"star/private/x.f90::One", "star/private/x.f90::Unmapped"}
    t = routine_table.build_table(
        manifest, inv, tmp_path, oracle_ids=ids, gaps=[]
    )
    row = _row(t)
    assert (row["routines_mapped"], row["routines_oracle"]) == (2, 1)
    assert row["lines_in_oracle_routines"] == 10
    assert row["ceiling"] == "ported_unverified"
    assert t["engine"]["routines_oracle"] == 1


def test_oracle_level_ignores_unmapped_routines(
    tmp_path: pathlib.Path,
) -> None:
    """A listed id without a map entry gives no oracle evidence."""
    manifest, inv = _mini(tmp_path, S.PARTIAL, ("One",))
    ids = {"star/private/x.f90::Two"}
    t = routine_table.build_table(
        manifest, inv, tmp_path, oracle_ids=ids, gaps=[]
    )
    assert _row(t)["routines_oracle"] == 0
    assert _row(t)["ceiling"] == "partial"


def test_ceiling_report_lists_exceeding_and_raise_candidate_files(
    tmp_path: pathlib.Path,
) -> None:
    """Status above the ceiling and fully mapped partial files are listed."""
    manifest, inv = _mini(tmp_path, S.KERNEL_REFERENCE_VECTORS, ("One", "Two"))
    t = routine_table.build_table(
        manifest, inv, tmp_path, oracle_ids=set(), gaps=[]
    )
    rep = t["ceiling"]
    assert [f["ceiling"] for f in rep["files_exceeding_ceiling"]] == [
        "ported_unverified"
    ]
    assert rep["raise_candidates"] == []
    manifest, inv = _mini(tmp_path, S.PARTIAL, ("One", "Two"))
    t = routine_table.build_table(
        manifest, inv, tmp_path, oracle_ids=set(), gaps=[]
    )
    assert t["ceiling"]["files_exceeding_ceiling"] == []
    assert len(t["ceiling"]["raise_candidates"]) == 1


def test_ceiling_report_makes_no_claim_for_files_without_mapped_routines(
    tmp_path: pathlib.Path,
) -> None:
    """A ported file with no mapped routine is counted, not flagged."""
    manifest, inv = _mini(tmp_path, S.PORTED_UNVERIFIED, ())
    t = routine_table.build_table(
        manifest, inv, tmp_path, oracle_ids=set(), gaps=[]
    )
    rep = t["ceiling"]
    assert rep["files_exceeding_ceiling"] == []
    assert rep["ported_files_without_ceiling_claim"] == 1


def _backlog() -> list[dict[str, Any]]:
    def row(path: str, name: str) -> dict[str, Any]:
        return {"upstream_path": path, "routine": name}

    return [
        row("star/private/a.f90", "build_thing_model"),
        row("star/private/a.f90", "get"),
        row("star/private/b.f90", "shared_routine"),
        row("star/private/c.f90", "shared_routine"),
        row("star/private/d_module.f90", "other_routine"),
        row("star/private/d_module.f90", "second_routine"),
    ]


def test_attribution_by_routine_file_and_ambiguity() -> None:
    """Routine names match alone; shared names need a file; short never."""
    rows = _backlog()
    gaps = [
        ("star/c1", "Diagnostic run lacks build_thing_model (#1)."),
        ("star/c2", "Uses get and shared_routine without a file."),
        ("star/c3", "shared_routine in star/private/c.f90 differs."),
        ("star/c4", "Needs d_module.f90 to be ported."),
    ]
    routine_backlog.attribute(rows, gaps)
    got = {(r["upstream_path"], r["routine"]): r for r in rows}
    assert got["star/private/a.f90", "build_thing_model"]["cases"] == [
        "star/c1"
    ]
    assert got["star/private/a.f90", "get"]["cases"] == []
    assert got["star/private/b.f90", "shared_routine"]["cases"] == []
    shared_c = got["star/private/c.f90", "shared_routine"]
    assert (shared_c["cases"], shared_c["basis"]) == (["star/c3"], "routine")
    for name in ("other_routine", "second_routine"):
        r = got["star/private/d_module.f90", name]
        assert (r["cases"], r["basis"]) == (["star/c4"], "file")


def test_committed_table_oracle_counts_do_not_exceed_mapped_counts() -> None:
    """Oracle evidence is a subset of mapped evidence everywhere."""
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    inv = routine_inventory.load_inventory(
        REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    )
    t = routine_table.build_table(manifest, inv, REPO_ROOT)
    for f in t["files"]:
        assert f["routines_oracle"] <= f["routines_mapped"], f
        assert f["lines_in_oracle_routines"] <= f["lines_in_mapped_routines"]
    for mod, m in t["modules"].items():
        assert m["routines_oracle"] <= m["routines_mapped"], mod
    ids = routine_table.oracle_routine_ids(REPO_ROOT)
    listed = {i for i in ids if i.split("::")[0] in inv["files"]}
    assert t["engine"]["routines_oracle"] <= len(listed)
    core = port_manifest.CORE_ENGINE_MODULES
    for b in t["backlog"]:
        entry = next(
            e for e in manifest.entries if e.upstream_path == b["upstream_path"]
        )
        assert entry.module in core
        assert entry.status.value in routine_backlog.BACKLOG_STATUSES
