# SPDX-License-Identifier: GPL-3.0-or-later
"""Seed status and routine-counterpart tables for ``build_port_manifest.py``."""

from __future__ import annotations

from flexmesa.testing import port_manifest

ROUTINE_COUNTERPART_HINTS: dict[str, dict[str, str]] = {
    "star/private/eps_grav.f90": {
        "eval_eps_grav_and_partials": (
            "crates/flexmesa-star/src/eps_grav.rs::eval_eps_grav"
        ),
        "zero_eps_grav_and_partials": (
            "crates/flexmesa-star/src/eps_grav.rs::eval_eps_grav"
        ),
    },
    "star/private/hydro_chem_eqns.f90": {
        "do_chem_eqns": (
            "crates/flexmesa-star/src/hydro_eqns.rs::assemble_henyey_system"
        ),
        "do1_chem_eqns": (
            "crates/flexmesa-star/src/hydro_eqns.rs::do1_chem_eqns"
        ),
    },
    "star/private/hydro_momentum.f90": {
        "do_surf_momentum_eqn": (
            "crates/flexmesa-star/src/hydro_momentum.rs::"
            "eval_surf_momentum_bc_pressure_ad"
        ),
    },
    "star/private/init_model.f90": {
        "get_zams_model": ("crates/flexmesa-star/src/zams.rs::load_zams_model"),
    },
    "star/private/ionization_potentials.f90": {
        "set_ionization_potentials": (
            "crates/flexmesa-ionization/src/potentials_data.rs::"
            "IONIZATION_POTENTIALS"
        ),
    },
    "star/private/neu.f90": {
        "do_neu_for_cell": "crates/flexmesa-star/src/micro.rs::do_neu",
        "do_clear_neu_for_cell": "crates/flexmesa-star/src/micro.rs::do_neu",
    },
    "star/private/pulse_utils.f90": {
        "eval_face_X": (
            "crates/flexmesa-star/src/pulse_utils.rs::zone_iso_fractions"
        ),
        "eval_face_rho": ("crates/flexmesa-star/src/pulse_utils.rs::eval_face"),
        "eval_center": (
            "crates/flexmesa-star/src/pulse_utils.rs::eval_center_parabolic"
        ),
        "eval_center_X": (
            "crates/flexmesa-star/src/pulse_utils.rs::zone_iso_fractions"
        ),
    },
    "star/private/rsp.f90": {
        "rsp_setup_part1": (
            "crates/flexmesa-star/src/rsp_driver.rs::rsp_setup_model"
        ),
        "rsp_setup_part2": (
            "crates/flexmesa-star/src/rsp_driver.rs::rsp_setup_model"
        ),
        "build_rsp_model": (
            "crates/flexmesa-star/src/rsp_build.rs::do_rsp_build_model"
        ),
        "do1_rsp_build": (
            "crates/flexmesa-star/src/rsp_build.rs::build_rsp_envelope_model"
        ),
    },
    "star/private/rsp_build.f90": {
        "do_rsp_build": (
            "crates/flexmesa-star/src/rsp_build.rs::do_rsp_build_model"
        ),
    },
    "star/private/rsp_def.f90": {
        "init_def": "crates/flexmesa-star/src/rsp_def.rs::RspDerivedConsts",
        "init_allocate": "crates/flexmesa-star/src/rsp_def.rs::RspState",
        "init_free": "crates/flexmesa-star/src/rsp_def.rs::RspState",
        "rsp_phase_time0": "crates/flexmesa-star/src/rsp_def.rs::RspState",
        "rsp_WORK": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_WORKQ": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_WORKT": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_WORKC": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_photo_out": (
            "crates/flexmesa-star/src/rsp_state_io.rs::pack_scalars"
        ),
        "rsp_photo_in": (
            "crates/flexmesa-star/src/rsp_state_io.rs::unpack_scalars"
        ),
        "finish_rsp_photo_in": (
            "crates/flexmesa-star/src/rsp_state_io.rs::unpack_scalars"
        ),
        "set_build_vars": (
            "crates/flexmesa-star/src/rsp_grid.rs::finish_build_rsp_model"
        ),
        "set_star_vars": (
            "crates/flexmesa-star/src/rsp_grid.rs::set_star_vars_rsp"
        ),
        "check_for_T_or_P_inversions": (
            "crates/flexmesa-star/src/rsp_grid.rs::finish_after_build_model"
        ),
        "check_R": (
            "crates/flexmesa-star/src/rsp_grid.rs::finish_after_build_model"
        ),
        "rsp_dump_for_debug": ("crates/flexmesa-star/src/rsp_def.rs::RspState"),
    },
    "star/private/rsp_eval_eos_and_kap.f90": {
        "restart_rsp_eos_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::RspChemCache"
        ),
        "init_for_rsp_eos_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::RspChemCache"
        ),
        "eval1_mesa_eosDEgas_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_inv.rs::"
            "eval1_mesa_eos_degas_and_kap"
        ),
        "eval1_mesa_eosDE_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_inv.rs::eval1_mesa_eos_de_and_kap"
        ),
        "eval1_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::eval1_kap_with_eos"
        ),
        "eval1_gamma_PT_getRho": (
            "crates/flexmesa-star/src/rsp_eos_inv.rs::eval1_gamma_pt_get_rho"
        ),
        "update_eos_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::eval_mesa_eos_and_kap"
        ),
    },
    "star/private/rsp_lina.f90": {
        "mesa_eos_kap": ("crates/flexmesa-star/src/rsp_grid.rs::mesa_eos_kap"),
        "SORT": "crates/flexmesa-star/src/rsp_lina.rs::do_lina",
    },
    "star/private/rsp_relax_env.f90": {
        "EOP": "crates/flexmesa-star/src/rsp_grid.rs::eop",
    },
    "star/private/rsp_step.f90": {
        "calc_equations": "crates/flexmesa-star/src/rsp_eqns.rs::eval_eqns",
        "calc_Fr": "crates/flexmesa-star/src/rsp_eqns.rs::fr_eqn",
        "calc_cell_equations": (
            "crates/flexmesa-star/src/rsp_eqns_cell.rs::"
            "total_energy_eqn+turbulent_energy_eqn+t_form_of_erad_eqn"
        ),
    },
    "star/private/star_bcyclic.f90": {
        "bcyclic_factor": (
            "crates/flexmesa-mtx/src/bcyclic.rs::solve_block_tridiag_bcyclic"
        ),
        "bcyclic_solve": (
            "crates/flexmesa-mtx/src/bcyclic.rs::solve_block_tridiag_bcyclic"
        ),
        "clear_storage": (
            "crates/flexmesa-mtx/src/bcyclic.rs::solve_block_tridiag_bcyclic"
        ),
    },
    "star/private/write_model.f90": {
        "do_write_model": (
            "crates/flexmesa-star/src/model_io.rs::write_mod_string"
        ),
    },
    "star/job/run_star.f90": {
        "do_run_star": (
            "crates/flexmesa-star/src/inlist_runner.rs::run_test_suite_case"
        ),
    },
    "binary/job/run_binary.f90": {
        "do_run_binary": (
            "crates/flexmesa-binary/src/e2e.rs::run_binary_test_suite_case"
        ),
    },
    "binary/private/binary_disk.f90": {
        "kap": "crates/flexmesa-binary/src/disk.rs::kap_kramers",
        "T_the": "crates/flexmesa-binary/src/disk.rs::t_the_nonzero",
        "fL2_the": "crates/flexmesa-binary/src/disk.rs::fl2_from_the_and_t",
    },
    "binary/private/binary_edot.f90": {
        "get_edot": (
            "crates/flexmesa-binary/src/system_jdot.rs::eval_coupled_edot"
        ),
        "edot_tidal": (
            "crates/flexmesa-binary/src/system_jdot.rs::eval_coupled_edot"
        ),
        "edot_tidal_Hut": (
            "crates/flexmesa-binary/src/jdot_tides.rs::hut_f2+hut_f3"
        ),
        "edot_enhancement_Isotropic": (
            "crates/flexmesa-binary/src/mdot.rs::"
            "eval_edot_enhancement_isotropic"
        ),
    },
    "binary/private/binary_jdot.f90": {
        "get_jdot": (
            "crates/flexmesa-binary/src/system_jdot.rs::eval_gated_jdot"
        ),
        "default_jdot_gr": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_gr"
        ),
        "default_jdot_ml": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_ml"
        ),
        "default_jdot_ls": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_ls"
        ),
        "default_jdot_missing_wind": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_missing_wind"
        ),
        "default_jdot_mb": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_mb"
        ),
    },
    "binary/private/binary_mdot.f90": {
        "check_implicit_rlo": (
            "crates/flexmesa-binary/src/mdot_implicit.rs::ImplicitMdotSolver"
        ),
        "pick_mdot_for_implicit_rlo": (
            "crates/flexmesa-binary/src/mdot_implicit.rs::record_and_propose"
        ),
        "adjust_mdots": (
            "crates/flexmesa-binary/src/system_rlof.rs::prepare_coupled_mdots"
        ),
        "rlo_mdot": (
            "crates/flexmesa-binary/src/system_rlof.rs::eval_scheme_explicit_mdot"
        ),
        "get_info_for_ritter": (
            "crates/flexmesa-binary/src/mdot.rs::eval_ritter_mdot_thin"
        ),
        "calculate_kolb_mdot_thick": (
            "crates/flexmesa-binary/src/mdot.rs::eval_kolb_mdot_thick"
        ),
        "get_info_for_kolb": (
            "crates/flexmesa-binary/src/mdot.rs::eval_kolb_mdot_thick"
        ),
        "get_info_for_arras": (
            "crates/flexmesa-binary/src/mdot.rs::eval_arras_mdot_thin"
        ),
        "get_info_for_ritter_eccentric": (
            "crates/flexmesa-binary/src/mdot.rs::eval_ritter_mdot_eccentric"
        ),
        "get_info_for_kolb_eccentric": (
            "crates/flexmesa-binary/src/mdot.rs::eval_kolb_mdot_eccentric"
        ),
        "set_accretion_composition": (
            "crates/flexmesa-binary/src/system_rlof.rs::surface_h1_and_he4"
        ),
    },
    "binary/private/binary_wind.f90": {
        "eval_wind_xfer_fractions": (
            "crates/flexmesa-binary/src/edd_wind.rs::"
            "eval_bondi_hoyle_wind_fraction"
        ),
        "Bondi_Hoyle_wind_transfer": (
            "crates/flexmesa-binary/src/edd_wind.rs::"
            "eval_bondi_hoyle_wind_fraction"
        ),
        "Tout_enhance_wind": (
            "crates/flexmesa-binary/src/edd_wind.rs::eval_tout_wind_enhancement"
        ),
    },
    "const/public/const_def.f90": {
        "do_const_init": "crates/flexmesa-const/src/lib.rs::PI",
    },
    "const/public/const_lib.f90": {
        "const_init": "crates/flexmesa-const/src/lib.rs::PI",
    },
    "turb/private/mlt.f90": {
        "calc_MLT": "crates/flexmesa-turb/src/mlt.rs::set_mlt",
    },
    "turb/private/semiconvection.f90": {
        "calc_semiconvection": (
            "crates/flexmesa-turb/src/semiconvection.rs::set_semiconvection"
        ),
    },
    "turb/private/tdc.f90": {
        "get_TDC_solution": (
            "crates/flexmesa-turb/src/tdc.rs::get_tdc_solution"
        ),
    },
    "turb/private/thermohaline.f90": {
        "get_D_thermohaline": (
            "crates/flexmesa-turb/src/thermohaline.rs::get_d_thermohaline"
        ),
    },
    "turb/public/turb.f90": {
        "set_thermohaline": (
            "crates/flexmesa-turb/src/thermohaline.rs::get_d_thermohaline"
        ),
        "set_MLT": "crates/flexmesa-turb/src/mlt.rs::set_mlt",
        "set_TDC": "crates/flexmesa-turb/src/tdc.rs::get_tdc_solution",
        "set_semiconvection": (
            "crates/flexmesa-turb/src/semiconvection.rs::set_semiconvection"
        ),
    },
    "neu/private/mod_neu.f90": {
        "neutrinos": "crates/flexmesa-neu/src/lib.rs::neu_get",
        "set_inputs": "crates/flexmesa-neu/src/inputs.rs::NeuInputs",
    },
    "neu/public/neu_lib.f90": {
        "neu_get": "crates/flexmesa-neu/src/lib.rs::neu_get",
    },
    "interp_2d/private/bipm_db.f90": {
        "do_mkbipm_db": (
            "crates/flexmesa-interp2d/src/bipm.rs::interp_mkbipm_db"
        ),
        "do_evbipm_db": (
            "crates/flexmesa-interp2d/src/bipm.rs::interp_evbipm_db"
        ),
    },
}

POST_AUDIT_PARTIALS: dict[str, tuple[list[str], str]] = {
    "star/private/history.f90": (
        ["crates/flexmesa-star/src/mesa_history.rs"],
        (
            "updated for #163: selectable history_columns.list writer "
            "(partial; microphysics columns under #161)"
        ),
    ),
    "star/private/history_specs.f90": (
        ["crates/flexmesa-star/src/mesa_history.rs"],
        (
            "updated for #163: recursive history_columns.list parser "
            "(partial; #161)"
        ),
    ),
    "star/private/profile.f90": (
        ["crates/flexmesa-star/src/mesa_profile.rs"],
        (
            "updated for #163: profileN.data and profiles.index priority "
            "writer (partial; #161)"
        ),
    ),
    "star/private/profile_getval.f90": (
        ["crates/flexmesa-star/src/mesa_profile.rs"],
        "updated for #163: profile column evaluator (partial; #161)",
    ),
    "binary/private/binary_history.f90": (
        ["crates/flexmesa-binary/src/binary_history.rs"],
        (
            "updated for #163: binary_history.data writer "
            "(partial; CE columns under #160)"
        ),
    ),
    "binary/private/binary_history_specs.f90": (
        ["crates/flexmesa-binary/src/binary_history.rs"],
        (
            "updated for #163: binary_history_columns.list parser "
            "(partial; #160)"
        ),
    ),
}

_EXACT_PORTED_MICRO: dict[str, list[str]] = {
    "const/public/const_def.f90": ["crates/flexmesa-const/src/lib.rs"],
    "const/public/const_lib.f90": ["crates/flexmesa-const/src/lib.rs"],
    "turb/private/mlt.f90": ["crates/flexmesa-turb/src/mlt.rs"],
    "turb/private/semiconvection.f90": [
        "crates/flexmesa-turb/src/semiconvection.rs"
    ],
    "turb/private/tdc.f90": ["crates/flexmesa-turb/src/tdc.rs"],
    "turb/private/thermohaline.f90": [
        "crates/flexmesa-turb/src/thermohaline.rs"
    ],
    "turb/public/turb.f90": [
        "crates/flexmesa-turb/src/lib.rs",
        "crates/flexmesa-turb/src/mlt.rs",
        "crates/flexmesa-turb/src/tdc.rs",
        "crates/flexmesa-turb/src/semiconvection.rs",
        "crates/flexmesa-turb/src/thermohaline.rs",
    ],
    "neu/private/mod_neu.f90": [
        "crates/flexmesa-neu/src/lib.rs",
        "crates/flexmesa-neu/src/inputs.rs",
        "crates/flexmesa-neu/src/pair_plas.rs",
        "crates/flexmesa-neu/src/phot.rs",
        "crates/flexmesa-neu/src/brem.rs",
        "crates/flexmesa-neu/src/reco.rs",
    ],
    "neu/public/neu_def.f90": ["crates/flexmesa-neu/src/lib.rs"],
    "neu/public/neu_lib.f90": ["crates/flexmesa-neu/src/lib.rs"],
    "interp_2d/private/bipm_db.f90": ["crates/flexmesa-interp2d/src/bipm.rs"],
    "auto_diff/private/auto_diff_real_1var_order1_module.f90": [
        "crates/flexmesa-autodiff/src/order1.rs"
    ],
    "auto_diff/private/auto_diff_real_2var_order1_module.f90": [
        "crates/flexmesa-autodiff/src/order1.rs"
    ],
    "auto_diff/private/auto_diff_real_4var_order1_module.f90": [
        "crates/flexmesa-autodiff/src/order1.rs"
    ],
    "auto_diff/private/auto_diff_real_15var_order1_module.f90": [
        "crates/flexmesa-autodiff/src/order1.rs"
    ],
    "auto_diff/private/auto_diff_real_star_order1_module.f90": [
        "crates/flexmesa-autodiff/src/order1.rs"
    ],
    "auto_diff/private/auto_diff_real_2var_order2_module.f90": [
        "crates/flexmesa-autodiff/src/order2.rs",
        "crates/flexmesa-autodiff/src/order2_ops.rs",
    ],
    "auto_diff/private/auto_diff_real_2var_order3_module.f90": [
        "crates/flexmesa-autodiff/src/order3.rs",
        "crates/flexmesa-autodiff/src/order3_ops.rs",
    ],
}

_ABSENT_MICRO_PREFIXES: tuple[str, ...] = (
    "eos/private/create_EXCOR7_table.f90",
    "eos/private/create_FSCRliq8_table.f90",
    "kap/private/op_",
    "rates/private/reaclib_print.f90",
    "ionization/private/ion_table_plot.f90",
    "colors/private/colors_ctrls_io.f90",
    "colors/private/colors_history.f90",
    "num/private/estrad.dek",
    "num/private/slvrad.dek",
    "num/private/slvseu.dek",
    "num/public/num_radau5_debug_routine.dek",
    "mtx/private/pre_conditioners.f90",
    "mtx/private/DGBSVX_wrapper.f90",
    "auto_diff/private/auto_diff_real_tdc_module.f90",
    "math/private/math_io.f90",
    "math/public/math_lib_crmath.f90",
    "utils/private/utils_dict.f90",
    "utils/private/utils_openmp",
    "utils/private/utils_system.f90",
)

_CRATE_DEFAULT_FILES: dict[str, list[str]] = {
    "eos": ["crates/flexmesa-eos/src/lib.rs"],
    "kap": ["crates/flexmesa-kap/src/lib.rs"],
    "net": ["crates/flexmesa-net/src/lib.rs"],
    "rates": ["crates/flexmesa-rates/src/lib.rs"],
    "neu": [
        "crates/flexmesa-neu/src/pair_plas.rs",
        "crates/flexmesa-neu/src/phot.rs",
        "crates/flexmesa-neu/src/brem.rs",
        "crates/flexmesa-neu/src/reco.rs",
    ],
    "atm": ["crates/flexmesa-atm/src/lib.rs"],
    "turb": ["crates/flexmesa-turb/src/tdc_support.rs"],
    "chem": ["crates/flexmesa-chem/src/lib.rs"],
    "ionization": ["crates/flexmesa-ionization/src/lib.rs"],
    "colors": ["crates/flexmesa-colors/src/lib.rs"],
    "num": ["crates/flexmesa-num/src/lib.rs"],
    "mtx": ["crates/flexmesa-mtx/src/lib.rs"],
    "interp_1d": ["crates/flexmesa-interp1d/src/lib.rs"],
    "interp_2d": ["crates/flexmesa-interp2d/src/lib.rs"],
    "auto_diff": ["crates/flexmesa-autodiff/src/lib.rs"],
    "math": ["crates/flexmesa-math/src/lib.rs"],
    "const": ["crates/flexmesa-const/src/lib.rs"],
    "utils": ["crates/flexmesa-utils/src/lib.rs"],
}


def classify_non_coverage_map_file(
    relpath: str, module: str
) -> tuple[port_manifest.PortStatus, list[str], str]:
    """Classify files outside the 199 ``coverage-map.md`` engine `.f90` rows."""
    p = port_manifest.PortStatus
    if module in {"gyre", "adipls", "stella", "plotting"}:
        pkg = {
            "gyre": "#170",
            "adipls": "#171",
            "stella": "#172",
            "plotting": "#173",
        }[module]
        return p.ABSENT, [], f"not ported ({pkg})"

    if relpath.startswith("star_data/public/") and relpath.endswith(".inc"):
        return (
            p.PARTIAL,
            [
                "crates/flexmesa-star/src/model.rs",
                "crates/flexmesa-star/src/hydro_state.rs",
            ],
            "seeded from coverage-map.md aggregate: StarModel subset",
        )
    if relpath.startswith("star_data/private/star_controls"):
        return (
            p.PARTIAL,
            ["crates/flexmesa-star/src/controls.rs"],
            "seeded from coverage-map.md aggregate: &controls subset",
        )
    if relpath.startswith("star_data/private/star_job_controls"):
        return (
            p.PARTIAL,
            ["crates/flexmesa-star/src/job_controls.rs"],
            "seeded from coverage-map.md aggregate: &star_job subset",
        )
    if relpath in {
        "binary/public/binary_controls.inc",
        "binary/private/binary_job_controls.inc",
    }:
        return (
            p.PARTIAL,
            ["crates/flexmesa-binary/src/inlist.rs"],
            "seeded from coverage-map.md aggregate: binary controls subset",
        )
    if relpath == "binary/public/binary_data.inc":
        return (
            p.PARTIAL,
            ["crates/flexmesa-binary/src/system.rs"],
            "seeded from coverage-map.md aggregate: BinarySystem subset",
        )
    if relpath.startswith(
        (
            "star/other/",
            "binary/other/",
            "star/job/",
            "binary/job/",
            "star/work/",
            "binary/work/",
            "astero/work/",
            "astero/private/adipls_support",
            "astero/public/",
        )
    ):
        return p.ABSENT, [], "seeded from coverage-map.md aggregate: absent"

    parts = relpath.split("/")
    if len(parts) >= 2 and (
        parts[1] in {"preprocessor", "plotter", "other"}
        or parts[1].endswith("_builder")
    ):
        return p.ABSENT, [], "offline table builder, plotter, or hook stub"
    if relpath.endswith(("_sg.f90", "_sg.f", "_qp.f90", "_sp.f90", ".c", ".h")):
        return (
            p.ABSENT,
            [],
            "single/quad-precision or C helper not used in f64 Rust",
        )
    if relpath in _EXACT_PORTED_MICRO:
        return (
            p.PORTED_UNVERIFIED,
            _EXACT_PORTED_MICRO[relpath],
            "seeded from coverage-map.md section 7 (unverified)",
        )
    if any(relpath.startswith(pref) for pref in _ABSENT_MICRO_PREFIXES):
        return p.ABSENT, [], "seeded from coverage-map.md section 7: absent"
    return (
        p.PARTIAL,
        _CRATE_DEFAULT_FILES[module],
        "seeded from coverage-map.md section 7: partial / uncompared",
    )
