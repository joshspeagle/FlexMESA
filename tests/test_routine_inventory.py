# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the routine inventory parser, inventory, and gate (`#611`)."""

from __future__ import annotations

import dataclasses
import pathlib
import textwrap
from typing import Any

import pytest

from flexmesa.testing import (
    _fortran_routines,
    port_manifest,
    routine_gate,
    routine_inventory,
    routine_table,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _parse(src: str, suffix: str = ".f90") -> _fortran_routines.ParseResult:
    return _fortran_routines.parse_fortran(
        textwrap.dedent(src).splitlines(), suffix
    )


def _names(res: _fortran_routines.ParseResult) -> list[tuple[str, str, str]]:
    return [(r.name, r.kind, r.parent) for r in res.routines]


def test_free_form_end_subroutine_and_function_forms_are_found() -> None:
    """Named ends, glued ends, and a contained procedure are all recorded."""
    res = _parse(
        """\
        module m
        contains
        subroutine a(x)
           real :: x
        end subroutine a
        function b(x) result(y)
           real :: x, y
        endfunction
        end module m
        """
    )
    assert _names(res) == [("a", "subroutine", ""), ("b", "function", "")]
    assert [(r.first, r.last) for r in res.routines] == [(3, 5), (6, 8)]
    assert res.warnings == ()


def test_type_and_attribute_prefixed_functions_are_functions() -> None:
    """Type specs and pure/elemental/recursive prefixes do not hide a name."""
    res = _parse(
        """\
        real(dp) pure function f(x)
        end function f
        elemental integer function g(i)
        end function g
        recursive function h(i) result(r)
        end function h
        character(len=len(s)) function k(s)
        end function k
        type(foo_t) function q(a)
        end function q
        double precision function d(a)
        end function d
        """
    )
    assert [r.name for r in res.routines] == ["f", "g", "h", "k", "q", "d"]
    assert all(r.kind == "function" for r in res.routines)


def test_bare_end_closes_routine_and_program_units() -> None:
    """A bare ``end`` ends the routine, and an unnamed main program counts."""
    res = _parse(
        """\
        subroutine s
        end
        x = 1
        end
        """
    )
    assert _names(res) == [("s", "subroutine", ""), ("", "program", "")]
    assert [(r.first, r.last) for r in res.routines] == [(1, 2), (3, 4)]


def test_names_on_continuation_lines_and_comments_are_handled() -> None:
    """The name may follow ``&``; comment lines inside a statement skip."""
    res = _parse(
        """\
        subroutine &
           ! a comment between continuation lines
           & long_name(a, &
           b)
        end subroutine long_name
        """
    )
    assert _names(res) == [("long_name", "subroutine", "")]
    assert (res.routines[0].first, res.routines[0].last) == (1, 5)


def test_interface_block_bodies_are_not_routines() -> None:
    """Interface bodies and ``module procedure`` name lists are skipped."""
    res = _parse(
        """\
        module m
        interface
           subroutine ext(a)
              real :: a
           end subroutine ext
           real function ef(a)
              real :: a
           end function ef
        end interface
        interface gen
           module procedure p1, p2
           module procedure p3
        end interface gen
        abstract interface
           subroutine cb(i)
              integer :: i
           end subroutine cb
        end interface
        contains
        subroutine real_one
        end subroutine real_one
        end module m
        """
    )
    assert _names(res) == [("real_one", "subroutine", "")]
    assert res.warnings == ()


def test_submodule_module_procedure_is_a_routine() -> None:
    """``module procedure`` outside an interface defines a routine."""
    res = _parse(
        """\
        submodule (parent) child
        contains
        module procedure impl
        end procedure impl
        end submodule child
        """
    )
    assert _names(res) == [("impl", "procedure", "")]


def test_contained_procedures_record_their_parent() -> None:
    """Internal procedures list the enclosing routine as parent."""
    res = _parse(
        """\
        subroutine outer
        call inner
        contains
        subroutine inner
        end subroutine inner
        real function helper(x)
        end function helper
        end subroutine outer
        """
    )
    assert _names(res) == [
        ("outer", "subroutine", ""),
        ("inner", "subroutine", "outer"),
        ("helper", "function", "outer"),
    ]


def test_derived_type_and_select_type_are_not_routines_or_scopes() -> None:
    """``type`` definitions close with ``end type``; select-type blocks nest."""
    res = _parse(
        """\
        module m
        type :: t
           integer :: i
        contains
           procedure :: p
        end type t
        type, extends(t) :: u
        end type u
        contains
        subroutine s(x)
           class(t) :: x
           select type (x)
           type is (u)
              continue
           end select
        end subroutine s
        end module m
        """
    )
    assert _names(res) == [("s", "subroutine", "")]
    assert res.warnings == ()


def test_fixed_form_with_labels_continuations_and_blanks_omitted() -> None:
    """Fixed-form text, including ratfor-style blank-free statements."""
    src = [
        "c a comment line",
        "      subroutine foo(a,",
        "     &   b)",
        "   10 continue",
        "      end",
        "      IntegerfunctionBAR(Needle)",
        "      BAR=1",
        "      end",
    ]
    res = _fortran_routines.parse_fortran(src, ".f")
    assert _names(res) == [("foo", "subroutine", ""), ("BAR", "function", "")]
    assert [(r.first, r.last) for r in res.routines] == [(2, 5), (6, 8)]


def test_fixed_form_tab_format_is_read() -> None:
    """A tab-led statement is a statement and a tab-digit line continues it."""
    src = ["\tfunction length(t)", "\tl=len(t)", "\tend"]
    res = _fortran_routines.parse_fortran(src, ".f")
    assert _names(res) == [("length", "function", "")]


def test_preprocessor_alternatives_open_one_routine() -> None:
    """``#:if``/``#:else`` branches with one shared ``end`` give one routine."""
    src = [
        "#:if flag",
        "subroutine w_${k}$(a, b)",
        "#:else",
        "subroutine w_${k}$(a)",
        "#:endif",
        "end subroutine w_${k}$",
    ]
    res = _fortran_routines.parse_fortran(src, ".fypp")
    assert [(r.name, r.first, r.last) for r in res.routines] == [
        ("w_${k}$", 2, 6)
    ]
    assert res.warnings == ()


def test_unterminated_routine_is_reported() -> None:
    """A routine with no end statement yields a warning, not a record."""
    res = _parse("subroutine s\n")
    assert res.routines == ()
    assert any("unterminated" in w for w in res.warnings)


def test_lines_outside_routines_count_the_complement_of_top_level_spans() -> (
    None
):
    """Outside lines exclude the union of spans; nested spans count once."""
    res = _parse(
        """\
        module m
        integer :: x
        contains
        subroutine a
        contains
        subroutine b
        end subroutine b
        end subroutine a
        end module m
        """
    )
    assert res.total_lines == 9
    assert res.outside_lines == 9 - 5


def test_committed_inventory_is_consistent_with_the_manifest() -> None:
    """The committed inventory matches the manifest without MESA source."""
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    inv = routine_inventory.load_inventory(
        REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    )
    assert routine_inventory.validate_inventory(inv, manifest) == []
    assert inv["header"]["mesa_version"] == "r26.4.1"
    assert inv["header"]["upstream_commit"] == manifest.upstream_commit


def test_inventory_validation_flags_a_tampered_span() -> None:
    """A span outside the file, or a wrong outside count, is reported."""
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    inv = routine_inventory.load_inventory(
        REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    )
    path = next(p for p, r in inv["files"].items() if r["routines"])
    inv["files"][path]["routines"][0][4] = inv["files"][path]["lines"] + 50
    errs = routine_inventory.validate_inventory(inv, manifest)
    assert any("span" in e and path in e for e in errs)
    inv = routine_inventory.load_inventory(
        REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    )
    inv["files"][path]["outside"] += 1
    errs = routine_inventory.validate_inventory(inv, manifest)
    assert any("outside-routine" in e and path in e for e in errs)


def test_inventory_rendering_is_deterministic_and_round_trips() -> None:
    """Rendering the loaded inventory reproduces the committed bytes."""
    path = REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    inv = routine_inventory.load_inventory(path)
    assert routine_inventory.render_inventory(inv) == path.read_text(
        encoding="utf-8"
    )


def _mini_repo(
    tmp_path: pathlib.Path, target: str, key: str
) -> tuple[port_manifest.PortManifest, dict[str, Any]]:
    (tmp_path / "crates").mkdir(exist_ok=True)
    (tmp_path / "crates" / "a.rs").write_text(
        "pub struct Thing;\n// fn commented_out() {}\n"
        "pub fn real_fn() {}\nimpl Thing { pub fn method(&self) {} }\n",
        encoding="utf-8",
    )
    entry = port_manifest.PortManifestEntry(
        upstream_path="star/private/x.f90",
        module="star",
        upstream_lines=10,
        upstream_sha256="0" * 64,
        status=port_manifest.PortStatus.PARTIAL,
        flexmesa_files=("crates/a.rs",),
        routine_map=((key, target),),
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
                "lines": 10,
                "outside": 0,
                "routines": [["Real_Routine", "subroutine", "", 1, 10]],
            }
        }
    }
    return manifest, inv


@pytest.mark.parametrize(
    ("target", "key", "expected"),
    [
        ("crates/a.rs::real_fn", "real_routine", []),
        ("crates/a.rs::real_fn+method", "Real_Routine", []),
        ("crates/a.rs::Thing", "Real_Routine", [("fn_item", "Thing")]),
        (
            "crates/a.rs::commented_out",
            "Real_Routine",
            [("fn_item", "commented_out")],
        ),
        (
            "crates/missing.rs::real_fn",
            "Real_Routine",
            [("fn_item", "crates/missing.rs")],
        ),
        ("crates/a.rs::real_fn", "not_there", [("routine_key", "not_there")]),
    ],
)
def test_gate_checks_routine_key_and_function_item(
    tmp_path: pathlib.Path,
    target: str,
    key: str,
    expected: list[tuple[str, str]],
) -> None:
    """Keys must be inventory routines and targets must be ``fn`` items."""
    manifest, inv = _mini_repo(tmp_path, target, key)
    got = routine_gate.collect_violations(manifest, inv, tmp_path)
    assert [(v.check, v.detail) for v in got] == expected


_TYPED_RS = """\
pub struct AdTrans;
pub struct AdBvp<T>(T);
pub trait Build { fn build(&self); }
impl AdTrans { pub fn new() -> Self { AdTrans } fn helper() { fn inner() {} } }
impl<T: Clone> AdBvp<T> {
    pub fn new(t: T) -> Self { AdBvp(t) }
    pub fn solve(&self) where T: Copy {}
}
impl Build for AdTrans { fn build(&self) {} }
impl<T> Build for AdBvp<T> where T: Clone { fn build(&self) {} }
pub fn unique_fn() {}
// fn commented_out() {}
"""


def _typed_violations(
    tmp_path: pathlib.Path, target: str
) -> list[tuple[str, str]]:
    manifest, inv = _mini_repo(tmp_path, target, "Real_Routine")
    (tmp_path / "crates" / "a.rs").write_text(_TYPED_RS, encoding="utf-8")
    got = routine_gate.collect_violations(manifest, inv, tmp_path)
    return [(v.check, v.detail) for v in got]


@pytest.mark.parametrize(
    "target",
    [
        "crates/a.rs::AdTrans::new",
        "crates/a.rs::AdBvp::new+AdBvp::solve",
        "crates/a.rs::AdTrans::build+AdBvp::build",
        "crates/a.rs::unique_fn+AdBvp::new",
        "crates/a.rs::solve",
    ],
)
def test_gate_accepts_qualified_inherent_generic_and_trait_methods(
    tmp_path: pathlib.Path, target: str
) -> None:
    """Qualified targets resolve through inherent, generic, and trait impls."""
    assert _typed_violations(tmp_path, target) == []


@pytest.mark.parametrize(
    ("target", "detail"),
    [
        ("crates/a.rs::AdBvp::helper", "AdBvp::helper"),
        ("crates/a.rs::AdTrans::solve", "AdTrans::solve"),
        ("crates/a.rs::Missing::new", "Missing::new"),
        ("crates/a.rs::AdTrans::inner", "AdTrans::inner"),
        ("crates/a.rs::AdTrans::commented_out", "AdTrans::commented_out"),
    ],
)
def test_gate_rejects_qualified_target_whose_type_lacks_the_method(
    tmp_path: pathlib.Path, target: str, detail: str
) -> None:
    """A wrong type, a nested fn, or a free fn does not satisfy ``Type::fn``."""
    assert _typed_violations(tmp_path, target) == [("fn_item", detail)]


@pytest.mark.parametrize(
    ("target", "detail"),
    [
        (
            "crates/a.rs::new",
            "new (ambiguous, 2 fn items; use Type::new)",
        ),
        (
            "crates/a.rs::build",
            "build (ambiguous, 3 fn items; use Type::build)",
        ),
    ],
)
def test_gate_rejects_bare_name_defined_more_than_once(
    tmp_path: pathlib.Path, target: str, detail: str
) -> None:
    """A bare name shared by several ``fn`` items is ambiguous."""
    assert _typed_violations(tmp_path, target) == [("fn_item", detail)]


def test_split_target_symbols_and_call_gate_leaf_symbols_agree() -> None:
    """The call-site gate reads ``Type::fn`` as a method, not a module path."""
    from flexmesa.testing import _port_parsers, _routine_call_gate

    syms = "AdBvp::new+AdMatch::new+free_fn"
    assert _port_parsers.split_target_symbols(syms) == [
        ("AdBvp", "new"),
        ("AdMatch", "new"),
        (None, "free_fn"),
    ]
    assert _routine_call_gate.parse_target_symbols(
        f"crates/flexmesa-gyre/src/ad.rs::{syms}"
    ) == {"new", "free_fn"}


def _write_mini_repo(
    tmp_path: pathlib.Path, target: str, allow: list[routine_gate.Violation]
) -> None:
    manifest, inv = _mini_repo(tmp_path, target, "Real_Routine")
    manifest = dataclasses.replace(
        manifest,
        upstream_commit=port_manifest.EXPECTED_UPSTREAM_COMMIT,
        entries=(
            dataclasses.replace(
                manifest.entries[0],
                upstream_sha256="a" * 64,
                upstream_lines=10,
            ),
        ),
    )
    dash = tmp_path / "benchmarks" / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    (dash / "port_manifest.toml").write_text(
        port_manifest.dump_port_manifest(manifest), encoding="utf-8"
    )
    inv["header"] = {
        "mesa_version": "r26.4.1",
        "upstream_commit": port_manifest.EXPECTED_UPSTREAM_COMMIT,
        "schema_version": routine_inventory.SCHEMA_VERSION,
        "files": 1,
        "routines": 1,
    }
    inv["files"]["star/private/x.f90"]["sha256"] = "a" * 64
    (dash / "routine_inventory.json").write_text(
        routine_inventory.render_inventory(inv), encoding="utf-8"
    )
    (dash / "routine_map_allowlist.toml").write_text(
        routine_gate.render_allowlist(allow), encoding="utf-8"
    )


def test_gate_fails_on_new_violations_and_stale_allowlist_entries(
    tmp_path: pathlib.Path,
) -> None:
    """Only listed violations pass; a fixed one must leave the allow-list."""
    manifest, inv = _mini_repo(tmp_path, "crates/a.rs::Thing", "Real_Routine")
    (violation,) = routine_gate.collect_violations(manifest, inv, tmp_path)

    _write_mini_repo(tmp_path, "crates/a.rs::Thing", [violation])
    assert routine_gate.verify_routine_map(tmp_path) == []

    _write_mini_repo(tmp_path, "crates/a.rs::Thing", [])
    errs = routine_gate.verify_routine_map(tmp_path)
    assert len(errs) == 1
    assert "fails fn_item" in errs[0]

    _write_mini_repo(tmp_path, "crates/a.rs::real_fn", [violation])
    errs = routine_gate.verify_routine_map(tmp_path)
    assert any("no longer fails fn_item" in e for e in errs)


def test_committed_routine_map_has_zero_new_violations() -> None:
    """The committed manifest, inventory, and allow-list agree."""
    assert routine_gate.verify_routine_map(REPO_ROOT) == []


def test_allowlist_counts_match_entries() -> None:
    """Declared per-check counts equal the listed entries."""
    counts, allowed = routine_gate.load_allowlist(
        REPO_ROOT / routine_gate.ALLOWLIST_RELPATH
    )
    for check in routine_gate.CHECKS:
        assert counts[check] == sum(1 for a in allowed if a.check == check)


def test_routine_table_files_match_regeneration() -> None:
    """The committed derived table equals what the script produces."""
    for path, text in routine_table.render_outputs(REPO_ROOT).items():
        assert path.read_text(encoding="utf-8") == text, path.name


def test_routine_table_engine_whole_file_figure_matches_the_manifest() -> None:
    """The whole-file fraction in the table uses the headline definition."""
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    core = [
        e
        for e in manifest.entries
        if e.module in port_manifest.CORE_ENGINE_MODULES
    ]
    ported = sum(e.upstream_lines for e in core if e.status.is_ported)
    total = sum(e.upstream_lines for e in core)
    inv = routine_inventory.load_inventory(
        REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    )
    table = routine_table.build_table(manifest, inv, REPO_ROOT)
    assert table["engine"]["upstream_lines"] == total
    assert table["engine"]["lines_in_whole_file_ported"] == ported
    assert table["engine"]["lines_full"] <= total


def test_own_lines_of_all_routines_plus_outside_lines_equal_file_lines() -> (
    None
):
    """Over the committed inventory, no line is credited twice or dropped."""
    inv = routine_inventory.load_inventory(
        REPO_ROOT / routine_inventory.INVENTORY_RELPATH
    )
    for path, rec in inv["files"].items():
        own = routine_table.own_lines(rec["routines"])
        assert min(own, default=1) >= 0, path
        assert sum(own) + rec["outside"] == rec["lines"], path
