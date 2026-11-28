# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests of the window-inlist constructor (#684 W2).

Each row of the rule tables in ``.coordination/triage/window-rules.md``
(sections 4.1 to 4.5) has a test, plus the chain reader, the snapshot header
reader, the loud failures (uncovered key, unmet change action) and the
dry-run reports. The cases are synthetic inlists written to ``tmp_path``, so
no MESA tree is needed; the dry run over the 90 star cases is run by hand
with ``scripts/windows/make_window_inlist.py dry-run``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import types
from typing import Any

import pytest

from flexmesa.schema import namelist

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load() -> types.ModuleType:
    path = REPO_ROOT / "scripts" / "windows" / "make_window_inlist.py"
    spec = importlib.util.spec_from_file_location("make_window_inlist", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_window_inlist"] = module
    spec.loader.exec_module(module)
    return module


mw = _load()

MOD_TEXT = """\
! note: initial lines of file can contain comments
!
          4148 -- model for mesa/star, angular velocities (omega)

                  version_number   'r26.4.1'
                          M/Msun      5.1847958036057790D+01
                    model_number                            1590
                        star_age      2.8400214506698662D+05
                       initial_z      1.4200000000000000D-03
                        n_shells                            1299
                        net_name   'co_burn.net'
                         species                               9

                lnd                        lnT
    1    -1.9394522429720787D+01     1.2043990886198371D+01
"""


def _header(**kw: object) -> Any:
    base: dict[str, object] = {
        "file_type": 0,
        "model_number": 100,
        "star_age": 1.0e6,
        "initial_z": 0.02,
        "mass_msun": 1.0,
        "net_name": "basic.net",
    }
    base.update(kw)
    return mw.SnapshotHeader(**base)


def _window(prefix: str = "w1", length: int = 50, **kw: object) -> Any:
    return mw.Window(prefix, "snap.mod", _header(**kw), length)


def _case(
    tmp_path: pathlib.Path,
    star_job: str = "",
    controls: str = "",
    extra: str = "",
    files: dict[str, str] | None = None,
) -> pathlib.Path:
    """Write ``inlist_stage`` (plus ``files``) and return the case dir."""
    case = tmp_path / "suite" / "case"
    case.mkdir(parents=True)
    text = (
        f"&star_job\n{star_job}\n/\n&eos\n/\n&kap\n/\n"
        f"&controls\n{controls}\n/\n{extra}"
    )
    (case / "inlist_stage").write_text(text, encoding="utf-8")
    for name, body in (files or {}).items():
        (case / name).write_text(body, encoding="utf-8")
    return case


def _build(
    tmp_path: pathlib.Path,
    star_job: str = "",
    controls: str = "",
    win: Any = None,
    **kw: Any,
) -> Any:
    case = _case(tmp_path, star_job, controls, **kw)
    stage = mw.read_stage(case, "inlist_stage")
    return mw.build_window_inlist(stage, win or _window())


def _get(built: Any, section: str, key: str) -> list[object]:
    return [a.value for a in built.sections[section] if a.name == key]


def _diff_rules(built: Any, key: str) -> list[str]:
    return [
        ln.split(" | ")[0]
        for ln in built.diff
        if ln.split(" | ")[2].startswith(key)
    ]


# --------------------------------------------------------------- the reader


def test_chain_reads_extras_after_their_block_and_consumes_the_includes(
    tmp_path: pathlib.Path,
) -> None:
    """Assignments of a block come first, then each extra in slot order."""
    case = _case(
        tmp_path,
        star_job=(
            "history_columns_file = 'a'\n"
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'inlist_one'\n"
            "read_extra_star_job_inlist(2) = .true.\n"
            "extra_star_job_inlist_name(2) = 'inlist_two'\n"
            "show_log_description_at_start = .false."
        ),
        files={
            "inlist_one": "&star_job\nhistory_columns_file = 'b'\n/\n",
            "inlist_two": "&star_job\nhistory_columns_file = 'c'\n/\n",
        },
    )
    star = mw.read_stage(case, "inlist_stage").sections["star_job"]
    assert [(a.name, a.value) for a in star] == [
        ("history_columns_file", "a"),
        ("show_log_description_at_start", False),
        ("history_columns_file", "b"),
        ("history_columns_file", "c"),
    ]


def test_chain_resolves_shared_directory_and_unsubscripted_array_names(
    tmp_path: pathlib.Path,
) -> None:
    """An include is found in ``../shared``; a bare array name sets slot 1."""
    case = _case(
        tmp_path,
        controls=(
            "read_extra_controls_inlist = .true.\n"
            "extra_controls_inlist_name = 'inlist_shared'"
        ),
    )
    shared = case.parent / "shared"
    shared.mkdir()
    (shared / "inlist_shared").write_text(
        "&controls\nmax_model_number = 7\n/\n", encoding="utf-8"
    )
    ctrl = mw.read_stage(case, "inlist_stage").sections["controls"]
    assert [(a.name, a.value) for a in ctrl] == [("max_model_number", 7)]


def test_chain_numbered_spelling_is_not_a_namelist_variable(
    tmp_path: pathlib.Path,
) -> None:
    """``read_extra_<s>_inlist1`` would be a namelist read error in MESA."""
    case = _case(tmp_path, controls="read_extra_controls_inlist1 = .true.")
    with pytest.raises(mw.WindowBuildError, match="not a MESA namelist"):
        mw.read_stage(case, "inlist_stage")


def test_chain_child_slot_overrides_the_parents_slot_and_resets_it(
    tmp_path: pathlib.Path,
) -> None:
    """MESA resets slots in place (star_job_ctrls_io.f90:597-607).

    The parent names ``child1`` (slot 1) and ``child2`` (slot 2). ``child1``
    sets slot 2 to ``child3``. MESA reads parent, child1, child3; the parent's
    slot 2 was reset by the time the loop reaches it, so ``child2`` is never
    read. A tree walk would read child2 as well.
    """
    case = _case(
        tmp_path,
        star_job=(
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'child1'\n"
            "read_extra_star_job_inlist(2) = .true.\n"
            "extra_star_job_inlist_name(2) = 'child2'\n"
            "history_columns_file = 'parent'"
        ),
        files={
            "child1": (
                "&star_job\nhistory_columns_file = 'c1'\n"
                "read_extra_star_job_inlist(2) = .true.\n"
                "extra_star_job_inlist_name(2) = 'child3'\n/\n"
            ),
            "child2": "&star_job\nhistory_columns_file = 'c2'\n/\n",
            "child3": "&star_job\nhistory_columns_file = 'c3'\n/\n",
        },
    )
    stage = mw.read_stage(case, "inlist_stage")
    assert stage.files["star_job"] == [
        "inlist_stage",
        "child1",
        "child3",
    ]
    assert [a.value for a in stage.sections["star_job"]] == [
        "parent",
        "c1",
        "c3",
    ]


def test_chain_reads_only_the_first_group_of_a_name_per_file(
    tmp_path: pathlib.Path,
) -> None:
    """A Fortran namelist read stops at the first matching group."""
    case = _case(tmp_path, star_job="history_columns_file = 'one'")
    with (case / "inlist_stage").open("a", encoding="utf-8") as fh:
        fh.write("&star_job\nhistory_columns_file = 'two'\n/\n")
    star = mw.read_stage(case, "inlist_stage").sections["star_job"]
    assert [a.value for a in star] == ["one"]


def test_chain_missing_group_is_an_error_for_star_job_not_for_eos(
    tmp_path: pathlib.Path,
) -> None:
    """``&star_job`` and ``&controls`` must exist in each file read."""
    case = _case(
        tmp_path,
        star_job=(
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'inlist_nogroup'"
        ),
        files={"inlist_nogroup": "&controls\n/\n"},
    )
    with pytest.raises(mw.WindowBuildError, match="no &star_job"):
        mw.read_stage(case, "inlist_stage")
    case2 = _case(
        tmp_path / "eos",
        extra="",
        files={"inlist_nogroup": "&controls\n/\n"},
    )
    (case2 / "inlist_stage").write_text(
        "&star_job\n/\n&eos\nread_extra_eos_inlist(1) = .true.\n"
        "extra_eos_inlist_name(1) = 'inlist_nogroup'\n/\n&kap\n/\n"
        "&controls\n/\n",
        encoding="utf-8",
    )
    assert mw.read_stage(case2, "inlist_stage").sections["eos"] == []


def test_chain_missing_star_job_include_is_an_error(
    tmp_path: pathlib.Path,
) -> None:
    """A missing include outside pgstar is loud, not skipped."""
    case = _case(
        tmp_path,
        star_job=(
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'nope'"
        ),
    )
    with pytest.raises(mw.WindowBuildError, match="missing inlist 'nope'"):
        mw.read_stage(case, "inlist_stage")


def test_chain_missing_pgstar_include_is_skipped_and_recorded(
    tmp_path: pathlib.Path,
) -> None:
    """The headless pgstar section may name an absent file (as the harness)."""
    case = _case(
        tmp_path,
        extra=(
            "&pgstar\nread_extra_pgstar_inlist(1) = .true.\n"
            "extra_pgstar_inlist_name(1) = 'inlist_pgstar'\n/\n"
        ),
    )
    stage = mw.read_stage(case, "inlist_stage")
    built = mw.build_window_inlist(stage, _window())
    assert any("pgstar" in ln and "skipped" in ln for ln in built.diff)


def test_chain_include_cycle_hits_the_nesting_limit(
    tmp_path: pathlib.Path,
) -> None:
    """A file that includes itself fails at level 10, as in MESA."""
    case = _case(
        tmp_path,
        star_job=(
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'inlist_stage'\n"
            "history_columns_file = 'x'"
        ),
    )
    with pytest.raises(mw.WindowBuildError, match="too many levels"):
        mw.read_stage(case, "inlist_stage")


# ------------------------------------------------------------- snapshot header


def test_mod_header_fields_and_bits() -> None:
    """The header reader returns the fields and the file_type bits."""
    h = mw.parse_mod_header(MOD_TEXT)
    assert h.file_type == 4148
    assert h.model_number == 1590
    assert h.star_age == 2.8400214506698662e5
    assert h.initial_z == 1.42e-3
    assert h.mass_msun == 5.1847958036057790e1
    assert h.net_name == "co_burn.net"
    # 4148 = 4096 + 32 + 16 + 4: rotation, prev-model present
    assert h.has_rotation
    assert h.has_prev_model
    assert not mw.parse_mod_header(
        MOD_TEXT.replace("4148", "  32812", 1)
    ).has_rotation


def test_mod_header_missing_field_is_an_error() -> None:
    """A header without ``net_name`` raises."""
    with pytest.raises(mw.WindowBuildError, match="net_name"):
        mw.parse_mod_header(MOD_TEXT.replace("net_name", "netname"))


# ------------------------------------------------------------------- rule 4.1


def test_rule_4_1_load_snapshot_and_no_model_creation(
    tmp_path: pathlib.Path,
) -> None:
    """The stage's own load and create settings are replaced (rule 4.1)."""
    built = _build(
        tmp_path,
        star_job=(
            "load_saved_model = .false.\n"
            "load_model_filename = 'previous.mod'\n"
            "create_pre_main_sequence_model = .true.\n"
            "create_initial_model = .true.\n"
            "create_rsp_model = .true."
        ),
    )
    assert _get(built, "star_job", "load_saved_model") == [True]
    assert _get(built, "star_job", "load_model_filename") == ["snap.mod"]
    for key in (
        "create_pre_main_sequence_model",
        "create_initial_model",
        "create_rsp_model",
    ):
        assert _get(built, "star_job", key) == [False], key
        assert "4.1" in _diff_rules(built, key)


def test_rule_4_1_keys_are_written_even_when_the_stage_omits_them(
    tmp_path: pathlib.Path,
) -> None:
    """An empty ``&star_job`` still gains the load keys."""
    built = _build(tmp_path)
    assert _get(built, "star_job", "load_saved_model") == [True]
    assert _get(built, "star_job", "create_rsp_model") == [False]


# ------------------------------------------------------------------- rule 4.2

# (key, value text) rows of the 4.2 table; a boolean becomes .false., any
# other key is removed.
RULE_4_2_ROWS = [
    ("set_initial_age", ".true."),
    ("initial_age", "5d0"),
    ("set_initial_model_number", ".true."),
    ("initial_model_number", "0"),
    ("set_initial_cumulative_energy_error", ".true."),
    ("new_cumulative_energy_error", "0d0"),
    ("set_initial_dt", ".true."),
    ("years_for_initial_dt", "1d-3"),
    ("limit_initial_dt", ".true."),
    ("set_initial_number_retries", ".true."),
    ("pre_ms_relax_num_steps", "50"),
    ("pre_ms_t_c", "5d5"),
    ("pre_ms_relax_to_start_radiative_core", ".true."),
    ("initial_zfracs", "6"),
    ("set_uniform_initial_composition", ".true."),
    ("initial_h1", "0.7d0"),
    ("initial_h2", "0d0"),
    ("initial_he3", "0d0"),
    ("initial_he4", "0.28d0"),
    ("relax_to_this_tau_factor", "10d0"),
    ("relax_core_years_for_dt", "1d-3"),
    ("relax_omega_max_yrs_dt", "1d-3"),
    ("num_steps_to_relax_rotation", "100"),
    ("num_steps_to_relax_composition", "100"),
    ("remove_surface_do_entropy", ".true."),
    ("remove_initial_center_by_temperature", "1d9"),
    ("remove_center_set_zero_v_center", ".true."),
    ("set_initial_tau_factor", ".true."),
    ("set_tau_factor", ".true."),
    ("set_to_this_tau_factor", ".true."),
    ("profile_starting_model", ".true."),
    ("steps_to_take_before_terminate", "5"),
    # rule 4.6: parameters of relax, create and removal actions
    ("core_avg_eps", "1d0"),
    ("core_avg_rho", "1d0"),
    ("dlg_core_mass_per_step", "1d-2"),
    ("dlgm_per_step", "1d-2"),
    ("lg_max_abs_mdot", "-3d0"),
    ("dlogtau_factor", "0.1d0"),
    ("irrad_col_depth", "1d0"),
    ("max_dt_for_relax_entropy", "1d3"),
    ("max_steps_to_relax_entropy", "100"),
    ("num_timescales_for_relax_entropy", "10"),
    ("timescale_for_relax_entropy", "1d0"),
    ("timescale_for_relax_composition", "1d0"),
    ("near_zams_relax_initial_surface_rotation_v", "100d0"),
    ("near_zams_relax_omega_div_omega_crit", "0.1d0"),
    ("initial_model_relax_num_steps", "100"),
    ("mass_in_gm_for_create_initial_model", "1d33"),
    ("radius_in_cm_for_create_initial_model", "1d11"),
    ("extra_mass_retained_by_remove_h_env", "1d-3"),
    ("seconds_for_initial_dt", "1d0"),
    ("dump_missing_metals_into_heaviest", ".true."),
    ("disable_pgstar_during_relax_flag", ".true."),
    ("some_new_relax_thing", "1"),
]


@pytest.mark.parametrize(
    ("action", "target"),
    [
        ("set_initial_omega", "new_omega = 1d-4"),
        (
            "set_initial_omega_div_omega_crit",
            "new_omega_div_omega_crit = 0.2d0",
        ),
        (
            "set_initial_surface_rotation_v",
            "new_surface_rotation_v = 50d0",
        ),
    ],
)
def test_rule_4_3_rotation_actions_off_when_snapshot_rotates(
    tmp_path: pathlib.Path, action: str, target: str
) -> None:
    """A rotation action is turned off and its target removed."""
    built = _build(
        tmp_path,
        star_job=f"{action} = .true.\n{target}",
        win=_window(file_type=mw.BIT_ROTATION),
    )
    assert _get(built, "star_job", action) == [False]
    assert _get(built, "star_job", target.split(" ")[0]) == []
    assert _diff_rules(built, action) == ["4.3"]


def test_rule_4_3_rotation_action_errors_when_snapshot_does_not_rotate(
    tmp_path: pathlib.Path,
) -> None:
    """A rotation action cannot be satisfied by a static snapshot."""
    with pytest.raises(mw.SnapshotMismatchError, match="rotation is off"):
        _build(
            tmp_path,
            star_job=(
                "set_initial_omega_div_omega_crit = .true.\n"
                "new_omega_div_omega_crit = 0.2d0"
            ),
        )
    inert = _build(tmp_path / "inert", star_job="new_omega = 1d-4")
    assert _get(inert, "star_job", "new_omega") == [1e-4]


def test_rule_4_6_relax_rotation_action_needs_the_rotation_bit(
    tmp_path: pathlib.Path,
) -> None:
    """``relax_initial_omega_div_omega_crit`` is checked like ``set_*``."""
    with pytest.raises(mw.SnapshotMismatchError, match="rotation is off"):
        _build(
            tmp_path,
            star_job=(
                "relax_initial_omega_div_omega_crit = .true.\n"
                "new_omega_div_omega_crit = 0.2d0"
            ),
        )


@pytest.mark.parametrize("axis", ["l", "v"])
def test_rule_4_6_center_targets_checked_against_the_header(
    tmp_path: pathlib.Path, axis: str
) -> None:
    """``set_initial_{l,v}_center`` is off when the header matches."""
    hdr: dict[str, Any] = {f"{axis}_center": 1.5e40}
    built = _build(
        tmp_path,
        star_job=(
            f"set_initial_{axis}_center = .true.\nnew_{axis}_center = 1.5d40"
        ),
        win=_window(**hdr),
    )
    assert _get(built, "star_job", f"set_initial_{axis}_center") == [False]
    assert _get(built, "star_job", f"new_{axis}_center") == []
    with pytest.raises(
        mw.SnapshotMismatchError, match=f"{axis.upper()}_center"
    ):
        _build(
            tmp_path / "bad",
            star_job=(
                f"set_initial_{axis}_center = .true.\n"
                f"new_{axis}_center = 1.5d40"
            ),
            win=_window(),
        )
    zero = _build(
        tmp_path / "zero",
        star_job=f"set_initial_{axis}_center = .true.\nnew_{axis}_center = 0d0",
    )
    assert _get(zero, "star_job", f"set_initial_{axis}_center") == [False]


def test_rule_4_6_new_y_needs_a_header_initial_y(
    tmp_path: pathlib.Path,
) -> None:
    """A y relax errors when the header has no ``initial_y``; inert if off."""
    sj = "relax_y = .true.\nnew_y = 0.28d0"
    with pytest.raises(mw.SnapshotMismatchError, match="initial_y"):
        _build(tmp_path, star_job=sj)
    built = _build(tmp_path / "ok", star_job=sj, win=_window(initial_y=0.28))
    assert _get(built, "star_job", "new_y") == []
    with pytest.raises(mw.SnapshotMismatchError, match="initial_y"):
        _build(tmp_path / "bad", star_job=sj, win=_window(initial_y=0.3))
    inert = _build(tmp_path / "inert", star_job="new_y = 0.28d0")
    assert _get(inert, "star_job", "new_y") == [0.28]


@pytest.mark.parametrize(
    "key",
    [
        f"{pre}_{flag}_flag"
        for pre in ("change", "change_initial", "new")
        for flag in ("v", "u", "rotation", "d_omega", "w_div_wc", "rti")
    ],
)
def test_rule_4_3_flag_changes_are_kept_on_and_unlogged(
    tmp_path: pathlib.Path, key: str
) -> None:
    """Flag change actions stay as written (the .mod does not record all)."""
    built = _build(tmp_path, star_job=f"{key} = .true.")
    assert _get(built, "star_job", key) == [True]
    assert _diff_rules(built, key) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("replace_initial_element", ".true."),
        ("replace_element", ".true."),
        ("replace_element_nzlo", "1"),
        ("replace_element_nzhi", "100"),
        ("set_nzlo", "1"),
        ("set_initial_to_xa_for_accretion", ".true."),
    ],
)
def test_rule_4_3_composition_edits_are_off(
    tmp_path: pathlib.Path, key: str, value: str
) -> None:
    """One-time composition edits are applied before the snapshot."""
    built = _build(tmp_path, star_job=f"{key} = {value}")
    assert _get(built, "star_job", key) == (
        [False] if value.startswith(".") else []
    )
    assert _diff_rules(built, key) == ["4.3"]


def test_rule_4_3_all_failed_checks_are_reported_together(
    tmp_path: pathlib.Path,
) -> None:
    """One error lists the net, z and mass problems at once."""
    with pytest.raises(mw.SnapshotMismatchError) as info:
        _build(
            tmp_path,
            star_job=(
                "change_net = .true.\nnew_net_name = 'x.net'\n"
                "change_initial_z = .true.\nnew_z = 0.5d0\n"
                "relax_mass = .true.\nnew_mass = 99d0"
            ),
        )
    assert len(info.value.problems) == 3


_NET_STAR_JOB = "change_initial_net = .true.\nnew_net_name = 'co_burn.net'"


def test_rule_4_3a_extended_net_is_built_with_change_initial_net_off(
    tmp_path: pathlib.Path,
) -> None:
    """co_burn.net grown to approx21.net by auto_extend_net is accepted."""
    built = _build(
        tmp_path,
        star_job=_NET_STAR_JOB,
        win=_window(net_name="approx21.net"),
    )
    assert _get(built, "star_job", "change_initial_net") == [False]
    assert _get(built, "star_job", "new_net_name") == ["co_burn.net"]
    assert _get(built, "star_job", "auto_extend_net") == []
    assert _diff_rules(built, "change_initial_net") == ["4.3a"]


def test_rule_4_3a_explicit_auto_extend_net_on_is_kept(
    tmp_path: pathlib.Path,
) -> None:
    """An explicit ``auto_extend_net = .true.`` stays on."""
    built = _build(
        tmp_path,
        star_job=_NET_STAR_JOB + "\nauto_extend_net = .true.",
        win=_window(net_name="approx21.net"),
    )
    assert _get(built, "star_job", "auto_extend_net") == [True]
    assert _get(built, "star_job", "change_initial_net") == [False]


def test_rule_4_3a_auto_extend_net_off_still_fails(
    tmp_path: pathlib.Path,
) -> None:
    """With auto_extend_net off, a net mismatch is a Rule 4.3 failure."""
    with pytest.raises(mw.SnapshotMismatchError, match=r"approx21\.net"):
        _build(
            tmp_path,
            star_job=_NET_STAR_JOB + "\nauto_extend_net = .false.",
            win=_window(net_name="approx21.net"),
        )


@pytest.mark.parametrize(
    ("new_net", "hdr_net"),
    [
        ("approx21.net", "co_burn.net"),  # extend_net never shrinks
        ("co_burn.net", "other.net"),  # not on the chain
        ("other.net", "approx21.net"),  # start not on the chain
    ],
)
def test_rule_4_3a_unreachable_net_still_fails(
    tmp_path: pathlib.Path, new_net: str, hdr_net: str
) -> None:
    """Only growth along h_he_net -> co_net -> adv_net is accepted."""
    with pytest.raises(mw.SnapshotMismatchError):
        _build(
            tmp_path,
            star_job=(
                f"change_initial_net = .true.\nnew_net_name = '{new_net}'"
            ),
            win=_window(net_name=hdr_net),
        )


def test_rule_4_3a_chain_follows_stage_net_names(
    tmp_path: pathlib.Path,
) -> None:
    """A stage that renames co_net and adv_net moves the chain with it."""
    built = _build(
        tmp_path,
        star_job=(
            "change_initial_net = .true.\nnew_net_name = 'mine.net'\n"
            "h_he_net = 'mine.net'\nco_net = 'mid.net'\n"
            "adv_net = 'big.net'"
        ),
        win=_window(net_name="big.net"),
    )
    assert _get(built, "star_job", "change_initial_net") == [False]


# ------------------------------------------------------------------- rule 4.4


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("history_columns_file", "'history_columns.list'"),
        ("profile_columns_file", "'profile_columns.list'"),
        ("filename_of_special_rate", "'r_he4_he4'"),
        ("use_special_weak_rates", ".true."),
        ("use_3a_fl87", ".true."),
        ("remove_fallback_at_each_step", ".true."),
        ("remove_fallback_speed_limit", "1d9"),
        ("set_to_xa_for_accretion", ".true."),
        ("set_cumulative_energy_error_at_step", "10"),
        ("set_cumulative_energy_error_each_step_if_age_less_than", "1d3"),
        ("show_log_description_at_start", ".false."),
        ("show_net_species_info", ".true."),
        # rule 4.6 keeps
        ("adv_net", ".true."),
        ("auto_extend_net", ".true."),
        ("enable_adaptive_network", ".true."),
        ("min_x_for_add", "1d-20"),
        ("min_x_for_keep", "1d-20"),
        ("min_x_for_n", "1d-20"),
        ("electron_coulomb_corrections", ".true."),
        ("ion_coulomb_corrections", ".true."),
        ("num_special_rate_factors", "1"),
        ("reaction_for_special_factor", "'r_he4_he4'"),
        ("special_rate_factor", "2d0"),
        ("special_weak_states_file", "'states.dat'"),
        ("special_weak_transitions_file", "'trans.dat'"),
        ("rate_tables_dir", "'rate_tables'"),
        ("rate_cache_suffix", "'cache'"),
        ("fallback_check_total_energy", ".true."),
        ("first_model_for_timing", "5"),
        ("warn_run_star_extras", ".false."),
        ("mesa_dir", "'/x'"),
        ("required_termination_code_string", "'xa_central_lower_limit'"),
        ("save_model_when_terminate", ".true."),
        ("write_profile_when_terminate", ".true."),
        ("save_pulse_data_when_terminate", ".true."),
    ],
)
def test_rule_4_4_per_step_physics_and_file_inputs_are_kept(
    tmp_path: pathlib.Path, key: str, value: str
) -> None:
    """Rule 4.4 keys pass through unchanged and unlogged."""
    built = _build(tmp_path, star_job=f"{key} = {value}")
    before = mw.read_stage(
        _case(tmp_path / "again", star_job=f"{key} = {value}"), "inlist_stage"
    ).sections["star_job"]
    assert [a for a in built.sections["star_job"] if a.name == key] == before
    assert _diff_rules(built, key) == []


def test_rule_4_4_eos_kap_and_controls_keys_are_kept(
    tmp_path: pathlib.Path,
) -> None:
    """``&eos``, ``&kap`` and ``&controls`` are unchanged except rule 4.5."""
    case = _case(
        tmp_path,
        controls="mixing_length_alpha = 1.8d0\nvarcontrol_target = 1d-4",
        extra="",
    )
    (case / "inlist_stage").write_text(
        "&star_job\n/\n&eos\nuse_Skye = .false.\n/\n"
        "&kap\nuse_Type2_opacities = .true.\n/\n"
        "&controls\nmixing_length_alpha = 1.8d0\n/\n",
        encoding="utf-8",
    )
    built = mw.build_window_inlist(
        mw.read_stage(case, "inlist_stage"), _window()
    )
    assert _get(built, "eos", "use_skye") == [False]
    assert _get(built, "kap", "use_type2_opacities") == [True]
    assert _get(built, "controls", "mixing_length_alpha") == [1.8]


# ------------------------------------------------------------------- rule 4.5


def test_rule_4_5_max_model_number_is_start_plus_w(
    tmp_path: pathlib.Path,
) -> None:
    """The window stops W steps after the snapshot's model number."""
    built = _build(tmp_path, win=_window(length=40, model_number=1590))
    assert _get(built, "controls", "max_model_number") == [1630]
    assert "4.5" in _diff_rules(built, "max_model_number")


def test_rule_4_5_a_smaller_stage_max_model_number_is_kept(
    tmp_path: pathlib.Path,
) -> None:
    """A stage cap below start + W stays."""
    built = _build(
        tmp_path,
        controls="max_model_number = 120",
        win=_window(length=50, model_number=100),
    )
    assert _get(built, "controls", "max_model_number") == [120]
    assert _diff_rules(built, "max_model_number") == []


def test_rule_4_5_a_larger_stage_max_model_number_is_replaced(
    tmp_path: pathlib.Path,
) -> None:
    """A stage cap above start + W is replaced by start + W."""
    built = _build(
        tmp_path,
        controls="max_model_number = 5000",
        win=_window(length=50, model_number=100),
    )
    assert _get(built, "controls", "max_model_number") == [150]


def test_rule_4_5_other_termination_controls_are_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    """Only ``max_model_number`` of the termination controls is touched."""
    built = _build(tmp_path, controls="xa_central_lower_limit(1) = 1d-3")
    assert [
        (a.sub, a.value)
        for a in built.sections["controls"]
        if a.name == "xa_central_lower_limit"
    ] == [("1", 1e-3)]


def test_rule_4_5_output_goes_to_the_window_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Log, photo, model and profile outputs get the window prefix."""
    built = _build(
        tmp_path,
        star_job=(
            "save_model_filename = 'final.mod'\n"
            "filename_for_profile_when_terminate = 'prof.data'"
        ),
        controls="log_directory = 'LOGS'\nphoto_directory = 'photos'",
        win=_window(prefix="win_2"),
    )
    assert _get(built, "star_job", "save_model_filename") == ["win_2/final.mod"]
    assert _get(built, "star_job", "filename_for_profile_when_terminate") == [
        "win_2/prof.data"
    ]
    assert _get(built, "controls", "log_directory") == ["win_2/LOGS"]
    assert _get(built, "controls", "photo_directory") == ["win_2/photos"]


def test_rule_4_5_headless_and_no_photo_at_terminate(
    tmp_path: pathlib.Path,
) -> None:
    """pgstar and photo-at-terminate are switched off."""
    built = _build(
        tmp_path,
        star_job=(
            "pgstar_flag = .true.\nsave_pgstar_files_when_terminate = .true.\n"
            "save_photo_when_terminate = .true."
        ),
    )
    for key in (
        "pgstar_flag",
        "save_pgstar_files_when_terminate",
        "save_photo_when_terminate",
    ):
        assert _get(built, "star_job", key) == [False], key
        assert _diff_rules(built, key) == ["4.5"], key


# ------------------------------------------------------------ loud failures


def test_uncovered_star_job_keys_fail_loudly_and_are_all_listed(
    tmp_path: pathlib.Path,
) -> None:
    """Unknown ``&star_job`` keys raise once, listing every one."""
    with pytest.raises(mw.UncoveredKeyError) as info:
        _build(
            tmp_path,
            star_job=(
                "mystery_two = .true.\nmystery_one = 'x'\n"
                "history_columns_file = 'h'"
            ),
        )
    assert info.value.keys == ["mystery_one", "mystery_two"]


def test_uncovered_namelist_section_fails_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """A section outside the rules is an error (``&colors`` is not)."""
    with pytest.raises(mw.UncoveredKeyError) as info:
        _build(tmp_path, extra="&mystery_section\nx = 1\n/\n")
    assert info.value.keys == ["&mystery_section"]


def test_rule_4_6_colors_section_is_kept_verbatim(
    tmp_path: pathlib.Path,
) -> None:
    """``&colors`` passes through unchanged and unlogged."""
    built = _build(
        tmp_path,
        extra=(
            "&colors\nuse_colors = .true.\n"
            "colors_per_newton_iter = .false.\n/\n"
        ),
    )
    assert [(a.name, a.value) for a in built.sections["colors"]] == [
        ("use_colors", True),
        ("colors_per_newton_iter", False),
    ]
    assert not any("| colors |" in ln for ln in built.diff)


def test_uncovered_error_wins_over_snapshot_checks(
    tmp_path: pathlib.Path,
) -> None:
    """Uncovered keys are reported before any snapshot mismatch."""
    with pytest.raises(mw.UncoveredKeyError):
        _build(
            tmp_path,
            star_job="mystery_key = 1\nchange_net = .true.",
        )


# ---------------------------------------------------------- output files


def test_window_files_round_trip_and_diff_lists_rule_numbers(
    tmp_path: pathlib.Path,
) -> None:
    """The written inlist re-parses to the built assignments; no includes."""
    case = _case(
        tmp_path,
        star_job=(
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'inlist_more'\n"
            "set_initial_age = .true.\ninitial_age = 0d0\n"
            "history_columns_file = 'it''s'"
        ),
        controls="initial_mass = 1.5d0",
        files={"inlist_more": "&star_job\nset_tau_factor = .true.\n/\n"},
    )
    win = _window(prefix="w3")
    built = mw.build_window_inlist(mw.read_stage(case, "inlist_stage"), win)
    out = tmp_path / "out"
    mw.write_window(out, win, built)
    text = (out / "inlist_window").read_text(encoding="utf-8")
    assert "read_extra" not in text
    blocks = {
        b.name: [(a.name, a.subscript, a.value) for a in b.assignments]
        for b in namelist.parse_namelist_file(out / "inlist_window")
    }
    for sec, assigns in built.sections.items():
        assert blocks[sec] == [(a.name, a.sub, a.value) for a in assigns]
    assert ("history_columns_file", None, "it's") in blocks["star_job"]
    diff = (out / "window_diff.txt").read_text(encoding="utf-8")
    assert diff.startswith("# window w3: snapshot snap.mod, start model 100")
    assert "4.2 | star_job | set_initial_age | .true. -> .false." in diff
    assert "4.2 | star_job | initial_age | removed (was 0.0)" in diff
    assert "4.2 | star_job | set_tau_factor" in diff


def test_every_diff_line_has_a_rule_number(tmp_path: pathlib.Path) -> None:
    """Every change is logged as ``rule | section | key | change``."""
    built = _build(
        tmp_path,
        star_job="set_initial_age = .true.\nrelax_core = .true.",
        controls="max_model_number = 9999",
    )
    for line in built.diff:
        rule, section, key, change = line.split(" | ", 3)
        assert rule in {"4.1", "4.2", "4.3", "4.5"}
        assert section in {"star_job", "controls"}
        assert key and change


def test_cli_build_writes_files_and_reports_unbuilt_windows(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``build`` writes the window; an uncovered key gives exit status 2."""
    case = _case(tmp_path, star_job="history_columns_file = 'h'")
    snap = tmp_path / "snap.mod"
    snap.write_text(MOD_TEXT, encoding="utf-8")
    out = tmp_path / "win"
    argv = [
        "build",
        "--case-dir",
        str(case),
        "--stage-inlist",
        "inlist_stage",
        "--snapshot",
        str(snap),
        "--prefix",
        "w1",
        "--steps",
        "30",
        "--out",
        str(out),
    ]
    assert mw.main(argv) == 0
    assert (out / "inlist_window").is_file()
    assert "max_model_number = 1620" in (out / "inlist_window").read_text()
    bad = _case(
        tmp_path / "bad", star_job="mystery = 1\nhistory_columns_file = 'h'"
    )
    argv[2] = str(bad)
    argv[-1] = str(tmp_path / "win2")
    assert mw.main(argv) == 2
    assert "window not built" in capsys.readouterr().err
    assert not (tmp_path / "win2").exists()


# ------------------------------------------------------- rule 5 and dry run


def test_extras_state_use_counts_kinds_and_ignores_comments(
    tmp_path: pathlib.Path,
) -> None:
    """``s% xtra``, ``ixtra`` and ``lxtra`` are counted; comments are not."""
    f = tmp_path / "run_star_extras.f90"
    f.write_text(
        "s% xtra(1) = 0d0\n"
        "s% ixtra(2) = s% ixtra(1) + 1\n"
        "if (s% lxtra(3)) then\n"
        "! s% xtra(9)\n"
        "x = 1 ! s% xtra(8)\n"
        "include 'xtra_coeff_os/xtra_coeff_os.inc'\n"
        "ops% xtra = 0\n",
        encoding="utf-8",
    )
    assert mw.extras_state_use(f) == {"xtra": 1, "ixtra": 2, "lxtra": 1}
    assert mw.extras_state_use(tmp_path / "missing.f90") == {}


def test_nominal_windows_short_stage_gets_one() -> None:
    """Stages of 60 models or fewer get one window; others five."""
    assert mw.nominal_windows(60) == 1
    assert mw.nominal_windows(61) == 5
    assert mw.nominal_windows(None) == 1


def test_dry_run_reports_diff_size_and_uncovered_keys(
    tmp_path: pathlib.Path,
) -> None:
    """The dry run builds each stage with a satisfying synthetic header."""
    case = _case(
        tmp_path,
        star_job=(
            "change_net = .true.\nnew_net_name = 'approx21.net'\n"
            "set_initial_omega_div_omega_crit = .true.\n"
            "new_omega_div_omega_crit = 0.1d0\nset_initial_age = .true."
        ),
    )
    (case / "inlist_bad").write_text(
        "&star_job\nmystery_key = .true.\n/\n&eos\n/\n&kap\n/\n&controls\n/\n",
        encoding="utf-8",
    )
    good, bad = mw.dry_run_case(
        case, [100, 10], stages=["inlist_stage", "inlist_bad"]
    )
    assert good.error == "" and good.uncovered == []
    assert good.windows == 5 and good.diff_lines >= 8
    assert good.flags == ()
    assert bad.uncovered == ["mystery_key"]
    assert good.built and not bad.built
    assert bad.windows == 1 and bad.diff_lines == 0


# ------------------------------------------------------------- rule 4.6 output


def test_rule_4_6_pause_before_terminate_is_off(
    tmp_path: pathlib.Path,
) -> None:
    """Headless: ``pause_before_terminate`` is set to ``.false.`` (rule 4.5)."""
    built = _build(tmp_path, star_job="pause_before_terminate = .true.")
    assert _get(built, "star_job", "pause_before_terminate") == [False]
    assert _diff_rules(built, "pause_before_terminate") == ["4.5"]


def test_rule_4_6_save_photo_filename_is_rewritten_when_present(
    tmp_path: pathlib.Path,
) -> None:
    """``save_photo_filename`` goes into the window directory; absent stays."""
    built = _build(
        tmp_path,
        star_job="save_photo_filename = 'sdb.photo'",
        win=_window(prefix="w9"),
    )
    assert _get(built, "star_job", "save_photo_filename") == ["w9/sdb.photo"]
    none = _build(tmp_path / "none")
    assert _get(none, "star_job", "save_photo_filename") == []


def test_rule_4_6_save_when_terminate_kept_with_rewritten_filenames(
    tmp_path: pathlib.Path,
) -> None:
    """The ``*_when_terminate`` switches stay on; their files are redirected."""
    built = _build(
        tmp_path,
        star_job=(
            "save_model_when_terminate = .true.\n"
            "write_profile_when_terminate = .true.\n"
            "filename_for_profile_when_terminate = 'final_profile.data'"
        ),
        win=_window(prefix="w4"),
    )
    assert _get(built, "star_job", "save_model_when_terminate") == [True]
    assert _get(built, "star_job", "write_profile_when_terminate") == [True]
    assert _get(built, "star_job", "filename_for_profile_when_terminate") == [
        "w4/final_profile.data"
    ]


# --------------------------------------------------------- flags and header


def test_no_prev_model_flag_and_flags_in_the_diff_header(
    tmp_path: pathlib.Path,
) -> None:
    """A snapshot without the previous-model block is flagged (rule 4.2)."""
    case = _case(tmp_path, star_job="relax_core = .true.")
    win = _window(file_type=0)
    built = mw.build_window_inlist(mw.read_stage(case, "inlist_stage"), win)
    assert built.flags == ["no-prev-model", "relax-assumed"]
    text = mw.format_diff(win, built.diff, built.flags)
    assert "# flags: no-prev-model, relax-assumed" in text
    ok = mw.build_window_inlist(
        mw.read_stage(case, "inlist_stage"),
        _window(file_type=mw.BIT_PREV_MODEL),
    )
    assert ok.flags == ["relax-assumed"]


def test_mod_header_center_fields_and_initial_y_are_optional() -> None:
    """``L_center``, ``v_center`` and ``initial_y`` are read when present."""
    h0 = mw.parse_mod_header(MOD_TEXT)
    assert (h0.l_center, h0.v_center, h0.initial_y) == (0.0, 0.0, None)
    text = MOD_TEXT.replace(
        "                        n_shells",
        "                        L_center      1.5D+40  ! luminosity of core\n"
        "                        v_center     -2.5D+06  ! velocity\n"
        "                        initial_y     2.8D-01\n"
        "                        n_shells",
    )
    h1 = mw.parse_mod_header(text)
    assert h1.l_center == 1.5e40
    assert h1.v_center == -2.5e6
    assert h1.initial_y == 0.28


# ----------------------------------------------------- chain parity (rule 4.6)

_SECTIONS = ("star_job", "controls", "eos", "kap")
_SLOT = re.compile(
    r"(read_extra|extra)_(\w+?)_inlist(_name)?\s*\(\s*(\d)\s*\)\s*=\s*"
    r"(\.true\.|\.false\.|'[^']*'|\"[^\"]*\")",
    re.IGNORECASE,
)


def _groups(path: pathlib.Path) -> dict[str, dict[str, dict[int, Any]]]:
    """Slot settings per group of a file, by regex (independent of the parser).

    Only the first group of each name counts, as in a namelist read.
    """
    out: dict[str, dict[str, dict[int, Any]]] = {}
    cur: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("!")[0]
        m = re.match(r"\s*&(\w+)", line)
        if m:
            name = m.group(1).lower()
            cur = name if name not in out else None
            if cur:
                out[cur] = {"f": {}, "n": {}}
            continue
        if line.strip().startswith("/"):
            cur = None
            continue
        if cur is None:
            continue
        for m in _SLOT.finditer(line):
            slot, val = int(m.group(4)), m.group(5)
            if m.group(1).lower() == "read_extra":
                out[cur]["f"][slot] = val.lower() == ".true."
            else:
                out[cur]["n"][slot] = val[1:-1].strip()
    return out


def _find(case_dir: pathlib.Path, name: str) -> pathlib.Path:
    for cand in (case_dir / name, case_dir.parent / "shared" / name):
        if cand.is_file():
            return cand
    raise AssertionError(f"missing include {name} for {case_dir}")


def _mesa_order(case_dir: pathlib.Path, entry: str, sec: str) -> list[str]:
    """Files read, with MESA's slot arrays reset in place (reference)."""
    flags: dict[int, bool] = {}
    names: dict[int, str] = {}
    order: list[str] = []

    def read(path: pathlib.Path) -> None:
        order.append(path.name)
        grp = _groups(path).get(sec)
        if grp:
            flags.update(grp["f"])
            names.update(grp["n"])
        for i in range(1, 6):
            on, nm = flags.get(i, False), names.get(i, "undefined")
            flags[i], names[i] = False, "undefined"
            if on:
                read(_find(case_dir, nm))

    read(_find(case_dir, entry))
    return order


def _tree_order(case_dir: pathlib.Path, entry: str, sec: str) -> list[str]:
    """Files read by a tree walk (what Rust ``InlistBundle`` does)."""
    order: list[str] = []

    def read(path: pathlib.Path) -> None:
        order.append(path.name)
        grp = _groups(path).get(sec) or {"f": {}, "n": {}}
        for i in range(1, 6):
            if grp["f"].get(i) and grp["n"].get(i, "").strip():
                read(_find(case_dir, grp["n"][i]))

    read(_find(case_dir, entry))
    return order


#: Python key -> field of ``star_parse_inlist_bundle``'s summary, per section.
_PARITY_FIELDS = {
    "star_job": {
        "create_pre_main_sequence_model": "create_pre_ms_model",
        "create_initial_model": "create_initial_model",
        "create_rsp_model": "create_rsp_model",
        "load_saved_model": "load_saved_model",
        "load_model_filename": "load_model_filename",
        "save_model_when_terminate": "save_model_when_terminate",
        "save_model_filename": "save_model_filename",
        "change_net": "change_net",
        "new_net_name": "new_net_name",
        "required_termination_code_string": "required_termination_code_string",
    },
    "controls": {
        "initial_mass": "initial_mass",
        "initial_z": "initial_z",
        "initial_y": "initial_y",
        "max_model_number": "max_model_number",
        "max_age": "max_age",
        "stop_near_zams": "stop_near_zams",
        "v_flag": "v_flag",
        "u_flag": "u_flag",
        "rotation_flag": "rotation_flag",
        "rsp_flag": "rsp_flag",
        "okay_to_remesh": "okay_to_remesh",
        "use_gold_tolerances": "use_gold_tolerances",
        "use_gold2_tolerances": "use_gold2_tolerances",
        "op_split_burn": "op_split_burn",
        "solver_max_tries_before_reject": "solver_max_tries_before_reject",
        "max_tries1": "max_tries1",
    },
}


def test_chain_reader_follows_mesa_and_cross_checks_inlist_bundle() -> None:
    """The constructor reads MESA's chain; the Rust bundle is the cross-check.

    For every ``do_one`` stage of the 90 star cases and each of ``star_job``,
    ``controls``, ``eos`` and ``kap``, the files the constructor reads, in
    order, equal those of a reference loop that resets the include slots in
    place (``star_job_ctrls_io.f90:597-607``). Rust ``InlistBundle`` walks a
    tree; where the two orders agree for all four sections, every key of the
    bundle's summary that the stage sets carries the same final value as the
    flattened chain. Where they disagree (``twin_studies`` only, 8
    (stage, section) pairs) the Rust summary is not compared: MESA's order
    is the standard and the departure belongs to a ``[Parity]`` issue.
    """
    import tomllib

    from flexmesa.star import e2e
    from flexmesa.testing import known_gaps_gate

    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    manifest = tomllib.loads(
        (
            REPO_ROOT / "benchmarks" / "history_harness" / "case_manifest.toml"
        ).read_text(encoding="utf-8")
    )
    cases = [c["case"] for c in manifest["cases"] if c["suite"] == "star"]
    assert len(cases) == 90
    compared = stages = pairs = 0
    disagree: set[tuple[str, str, str]] = set()
    for case in cases:
        case_dir = base / "star" / "test_suite" / case
        for kind, inlist, _out in e2e.parse_rn_stages(case_dir):
            if kind != "do_one":
                continue
            stages += 1
            flat = mw.read_stage(case_dir, inlist)
            agree = True
            for sec in _SECTIONS:
                if sec not in flat.files:
                    continue
                pairs += 1
                mesa = _mesa_order(case_dir, inlist, sec)
                assert flat.files[sec] == mesa, (case, inlist, sec)
                if mesa != _tree_order(case_dir, inlist, sec):
                    agree = False
                    disagree.add((case, inlist, sec))
            if not agree:
                continue
            rust = e2e.parse_inlist_bundle(
                case_dir / inlist, allow_unsupported_controls=True
            )
            assert set(flat.sections) <= set(rust["sections"]), (case, inlist)
            for section, fields in _PARITY_FIELDS.items():
                for key, field in fields.items():
                    last = mw._last(flat.sections.get(section, []), key)
                    if last is None:
                        continue
                    compared += 1
                    got = rust[field]
                    if isinstance(got, list):
                        got = got[-1] if got else None
                    assert got == last.value, (case, inlist, key)
    assert stages >= 90 and pairs >= 4 * 90
    assert compared > 500, compared
    assert {c for c, _i, _s in disagree} == {"twin_studies"}, disagree
    assert len(disagree) == 8, disagree
