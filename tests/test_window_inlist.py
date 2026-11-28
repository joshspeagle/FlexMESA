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


def test_chain_resolves_shared_directory_and_legacy_numbered_keys(
    tmp_path: pathlib.Path,
) -> None:
    """An include is found in ``../shared``; legacy numbered keys work."""
    case = _case(
        tmp_path,
        controls=(
            "read_extra_controls_inlist1 = .true.\n"
            "extra_controls_inlist1_name = 'inlist_shared'"
        ),
    )
    shared = case.parent / "shared"
    shared.mkdir()
    (shared / "inlist_shared").write_text(
        "&controls\nmax_model_number = 7\n/\n", encoding="utf-8"
    )
    ctrl = mw.read_stage(case, "inlist_stage").sections["controls"]
    assert [(a.name, a.value) for a in ctrl] == [("max_model_number", 7)]


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


def test_chain_include_cycle_does_not_recurse_forever(
    tmp_path: pathlib.Path,
) -> None:
    """A file that includes itself is read once."""
    case = _case(
        tmp_path,
        star_job=(
            "read_extra_star_job_inlist(1) = .true.\n"
            "extra_star_job_inlist_name(1) = 'inlist_stage'\n"
            "history_columns_file = 'x'"
        ),
    )
    star = mw.read_stage(case, "inlist_stage").sections["star_job"]
    assert [a.name for a in star] == ["history_columns_file"]


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
    ("relax_mass", ".true."),
    ("relax_initial_mass", ".true."),
    ("relax_initial_z", ".true."),
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
]


@pytest.mark.parametrize(("key", "value"), RULE_4_2_ROWS)
def test_rule_4_2_one_time_actions_are_off(
    tmp_path: pathlib.Path, key: str, value: str
) -> None:
    """Each row of the 4.2 table is turned off and logged with rule 4.2."""
    built = _build(tmp_path, star_job=f"{key} = {value}")
    got = _get(built, "star_job", key)
    if value.startswith("."):
        assert got == [False]
    else:
        assert got == []
    assert _diff_rules(built, key) == ["4.2"]


def test_rule_4_2_a_switch_already_off_is_not_logged(
    tmp_path: pathlib.Path,
) -> None:
    """``relax_mass = .false.`` stays and produces no diff line."""
    built = _build(tmp_path, star_job="relax_mass = .false.")
    assert _get(built, "star_job", "relax_mass") == [False]
    assert _diff_rules(built, "relax_mass") == []


# ------------------------------------------------------------------- rule 4.3


def test_rule_4_3_net_change_off_when_header_net_matches(
    tmp_path: pathlib.Path,
) -> None:
    """``change_net`` and ``change_initial_net`` go off; the name is kept."""
    built = _build(
        tmp_path,
        star_job=(
            "change_net = .true.\nchange_initial_net = .true.\n"
            "new_net_name = 'co_burn.net'"
        ),
        win=_window(net_name="co_burn.net"),
    )
    assert _get(built, "star_job", "change_net") == [False]
    assert _get(built, "star_job", "change_initial_net") == [False]
    assert _get(built, "star_job", "new_net_name") == ["co_burn.net"]
    assert _diff_rules(built, "change_net") == ["4.3"]


def test_rule_4_3_net_mismatch_is_an_error(tmp_path: pathlib.Path) -> None:
    """A snapshot on another net is reported, not built."""
    with pytest.raises(mw.SnapshotMismatchError, match=r"co_burn\.net"):
        _build(
            tmp_path,
            star_job="change_net = .true.\nnew_net_name = 'co_burn.net'",
            win=_window(net_name="basic.net"),
        )


def test_rule_4_3_net_change_without_target_name_is_an_error(
    tmp_path: pathlib.Path,
) -> None:
    """``change_net`` with no ``new_net_name`` cannot be checked."""
    with pytest.raises(mw.SnapshotMismatchError, match="new_net_name"):
        _build(tmp_path, star_job="change_net = .true.")


def test_rule_4_3_z_change_off_when_header_z_matches(
    tmp_path: pathlib.Path,
) -> None:
    """``change_initial_z`` goes off when ``initial_z`` equals ``new_z``."""
    built = _build(
        tmp_path,
        star_job="change_initial_z = .true.\nnew_z = 0.014d0",
        win=_window(initial_z=0.014),
    )
    assert _get(built, "star_job", "change_initial_z") == [False]
    assert _diff_rules(built, "change_initial_z") == ["4.3"]


def test_rule_4_3_z_mismatch_is_an_error(tmp_path: pathlib.Path) -> None:
    """A header ``initial_z`` that differs from ``new_z`` is reported."""
    with pytest.raises(mw.SnapshotMismatchError, match="initial_z"):
        _build(
            tmp_path,
            star_job="change_initial_z = .true.\nnew_z = 0.014d0",
            win=_window(initial_z=0.02),
        )


def test_rule_4_3_mass_targets_off_when_header_mass_matches(
    tmp_path: pathlib.Path,
) -> None:
    """``new_mass`` and ``change_mass_years_for_dt`` are removed."""
    built = _build(
        tmp_path,
        star_job="new_mass = 15d0\nchange_mass_years_for_dt = 1d-3",
        win=_window(mass_msun=15.0 * (1 + 1e-14)),
    )
    assert _get(built, "star_job", "new_mass") == []
    assert _get(built, "star_job", "change_mass_years_for_dt") == []
    assert _diff_rules(built, "new_mass") == ["4.3"]


def test_rule_4_3_mass_mismatch_is_an_error(tmp_path: pathlib.Path) -> None:
    """A stage that relaxes mass over the run is not built from a snapshot."""
    with pytest.raises(mw.SnapshotMismatchError, match="M/Msun"):
        _build(
            tmp_path, star_job="new_mass = 15d0", win=_window(mass_msun=20.0)
        )


def test_rule_4_3_new_core_mass_cannot_be_checked_and_errors(
    tmp_path: pathlib.Path,
) -> None:
    """Rule 4.3 names only M/Msun; ``new_core_mass`` is listed for claude-a."""
    with pytest.raises(mw.SnapshotMismatchError, match="new_core_mass"):
        _build(tmp_path, star_job="new_core_mass = 1d0")


ROTATION_ROWS = [
    ("new_omega_div_omega_crit", "0.2d0"),
    ("set_initial_omega", ".true."),
    ("set_initial_omega_div_omega_crit", ".true."),
    ("new_omega", "1d-4"),
    ("set_initial_surface_rotation_v", ".true."),
    ("new_surface_rotation_v", "50d0"),
]


@pytest.mark.parametrize(("key", "value"), ROTATION_ROWS)
def test_rule_4_3_rotation_targets_off_when_snapshot_rotates(
    tmp_path: pathlib.Path, key: str, value: str
) -> None:
    """Rotation actions go off when the file_type has the rotation bit."""
    built = _build(
        tmp_path,
        star_job=f"{key} = {value}",
        win=_window(file_type=mw.BIT_ROTATION),
    )
    got = _get(built, "star_job", key)
    assert got == ([False] if value.startswith(".") else [])
    assert _diff_rules(built, key) == ["4.3"]


def test_rule_4_3_rotation_targets_error_when_snapshot_does_not_rotate(
    tmp_path: pathlib.Path,
) -> None:
    """The stage's rotation target cannot be satisfied by a static snapshot."""
    with pytest.raises(mw.SnapshotMismatchError, match="rotation is off"):
        _build(tmp_path, star_job="new_omega_div_omega_crit = 0.2d0")


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
                "change_initial_z = .true.\nnew_z = 0.5d0\nnew_mass = 99d0"
            ),
        )
    assert len(info.value.problems) == 3


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
                "save_model_when_terminate = .true.\n"
                "required_termination_code_string = 'x'\n"
                "history_columns_file = 'h'"
            ),
        )
    assert info.value.keys == [
        "required_termination_code_string",
        "save_model_when_terminate",
    ]


def test_uncovered_namelist_section_fails_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """A section outside the rules (``&colors``) is an error."""
    with pytest.raises(mw.UncoveredKeyError) as info:
        _build(tmp_path, extra="&colors\nuse_colors = .true.\n/\n")
    assert info.value.keys == ["&colors"]


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
        star_job="set_initial_age = .true.\nrelax_mass = .true.",
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
            "new_omega_div_omega_crit = 0.1d0\nset_initial_age = .true."
        ),
    )
    (case / "inlist_bad").write_text(
        "&star_job\nsave_model_when_terminate = .true.\n/\n"
        "&eos\n/\n&kap\n/\n&controls\n/\n",
        encoding="utf-8",
    )
    good, bad = mw.dry_run_case(
        case, [100, 10], stages=["inlist_stage", "inlist_bad"]
    )
    assert good.error == "" and good.uncovered == []
    assert good.windows == 5 and good.diff_lines >= 8
    assert bad.uncovered == ["save_model_when_terminate"]
    assert bad.windows == 1 and bad.diff_lines == 0
