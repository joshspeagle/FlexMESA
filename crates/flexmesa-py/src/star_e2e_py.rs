// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-star` E2E-0 `inlist` bundle parsing, `./rn`
//! stage resolution, photo restart parity verification, and multi-stage
//! `test_suite` case execution (`inlist` and `inlist_runner`).

use std::path::Path;

use flexmesa_star::{
    eval_conductive_flame_state, eval_duffell_rti_shock_state,
    eval_fe_core_infall_state, eval_hot_cool_blended_wind_mdot,
    eval_magnetic_braking_torque, eval_model_colors,
    eval_pair_instability_gamma1_and_energy,
    eval_planetary_irradiation_heating, eval_roche_gravity_darkening,
    eval_solar_calibration_objective, eval_spruit_tayler_dynamo_diagnostics,
    eval_starspot_effective_luminosity, eval_t_tau_gradr_factor,
    eval_tov_effective_cgrav, eval_tzo_core_accretion_step,
    eval_wd_phase_separation, export_stella_ascii,
    nelder_mead_solar_calibration_step, parse_rn_stages, run_case_list,
    run_test_suite_case, verify_photo_restart_parity, AtmTTauRelation,
    CaseRunReport, InlistBundle, RnStage, StarError, TIER1_CASES, TIER2_CASES,
    TIER3_CASES,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::star_helpers_py::to_py_err;
use crate::star_py::PyStarModel;

/// Parses a MESA `inlist` file (recursively resolving all `read_extra_*_inlist`
/// includes) and returns a summary dictionary of parsed controls.
#[pyfunction]
pub fn star_parse_inlist_bundle<'py>(
    py: Python<'py>,
    inlist_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let bundle =
        InlistBundle::from_file(Path::new(inlist_path)).map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("entry_inlist", &bundle.entry_inlist)?;
    dict.set_item("create_pre_ms_model", bundle.job.create_pre_ms_model)?;
    dict.set_item("create_initial_model", bundle.job.create_initial_model)?;
    dict.set_item("create_rsp_model", bundle.job.create_rsp_model)?;
    dict.set_item("load_saved_model", bundle.job.load_saved_model)?;
    dict.set_item("saved_model_name", &bundle.job.saved_model_name)?;
    dict.set_item(
        "save_model_when_terminate",
        bundle.job.save_model_when_terminate,
    )?;
    dict.set_item("save_model_filename", &bundle.job.save_model_filename)?;
    dict.set_item("change_net", bundle.job.change_net)?;
    dict.set_item("new_net_name", &bundle.job.new_net_name)?;
    dict.set_item(
        "required_termination_code_string",
        &bundle.job.required_termination_code_string,
    )?;
    dict.set_item("initial_mass", bundle.controls.initial_mass)?;
    dict.set_item("initial_z", bundle.controls.initial_z)?;
    dict.set_item("initial_y", bundle.controls.initial_y)?;
    dict.set_item("max_model_number", bundle.controls.max_model_number)?;
    dict.set_item("max_age", bundle.controls.max_age)?;
    dict.set_item("stop_near_zams", bundle.controls.stop_near_zams)?;
    dict.set_item("v_flag", bundle.controls.v_flag)?;
    dict.set_item("u_flag", bundle.controls.u_flag)?;
    dict.set_item("rotation_flag", bundle.controls.rotation_flag)?;
    dict.set_item("rsp_flag", bundle.controls.rsp_flag)?;
    dict.set_item("rsp2_flag", bundle.controls.rsp2_flag)?;
    dict.set_item("use_other_cgrav", bundle.controls.use_other_cgrav)?;
    let sections: Vec<String> = bundle.sections.keys().cloned().collect();
    dict.set_item("sections", sections)?;
    Ok(dict)
}

/// Parses an upstream MESA `./rn` script in `case_dir` into a list of
/// `(stage_kind, arg1, arg2)` tuples.
#[pyfunction]
#[pyo3(signature = (case_dir, skip_optional=true))]
pub fn star_parse_rn_stages(
    case_dir: &str,
    skip_optional: bool,
) -> PyResult<Vec<(String, String, String)>> {
    let stages = parse_rn_stages(Path::new(case_dir), skip_optional)
        .map_err(to_py_err)?;
    let mut out = Vec::with_capacity(stages.len());
    for st in stages {
        match st {
            RnStage::DoOne {
                inlist, output_mod, ..
            } => {
                out.push((
                    String::from("do_one"),
                    inlist,
                    output_mod.unwrap_or_default(),
                ));
            }
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                out.push((String::from("cp"), src_mod, dst_mod));
            }
        }
    }
    Ok(out)
}

/// Verifies bit-identical `.photo` restart parity on `model` over `n_steps`.
#[pyfunction]
#[pyo3(signature = (model, n_steps=1))]
pub fn star_verify_photo_restart(
    model: &PyStarModel,
    n_steps: usize,
) -> PyResult<(u64, u64)> {
    verify_photo_restart_parity(&model.inner, n_steps).map_err(to_py_err)
}

pub(crate) fn case_report_to_pydict<'py>(
    py: Python<'py>,
    rep: &CaseRunReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("case_name", &rep.case_name)?;
    out.set_item("tier", rep.tier)?;
    out.set_item("passed", rep.passed)?;
    out.set_item("termination_code", &rep.termination_code)?;
    out.set_item("total_steps", rep.total_steps)?;
    out.set_item("total_newton_iters", rep.total_newton_iterations)?;
    out.set_item("total_newton_iterations", rep.total_newton_iterations)?;
    out.set_item("total_retries", rep.total_retries)?;
    out.set_item("total_runtime_sec", rep.total_runtime_sec)?;
    out.set_item("wall_time_ms", rep.wall_time_ms)?;
    out.set_item("final_teff", rep.final_teff)?;
    out.set_item("final_log_l", rep.final_log_l)?;
    out.set_item("final_log_r", rep.final_log_r)?;
    out.set_item("final_log_tc", rep.final_log_tc)?;
    out.set_item("final_log_rhoc", rep.final_log_rhoc)?;
    out.set_item("final_center_h1", rep.final_center_h1)?;
    out.set_item("final_center_he4", rep.final_center_he4)?;
    out.set_item("photo_restart_passed", rep.photo_restart_verified)?;
    out.set_item("photo_restart_verified", rep.photo_restart_verified)?;
    out.set_item("assertions_passed", rep.assertions_passed)?;

    let stages_list = PyList::empty(py);
    for st in &rep.stages {
        let sdict = PyDict::new(py);
        sdict.set_item("inlist_name", &st.inlist_name)?;
        sdict.set_item("steps_taken", st.steps_taken)?;
        sdict
            .set_item("total_newton_iterations", st.total_newton_iterations)?;
        sdict.set_item("total_retries", st.total_retries)?;
        sdict.set_item("termination_code", &st.termination_code)?;
        sdict.set_item("runtime_sec", st.runtime_sec)?;
        sdict.set_item("star_age", st.final_history.star_age)?;
        sdict.set_item("star_mass", st.final_history.star_mass)?;
        sdict.set_item("log_l", st.final_history.log_l)?;
        sdict.set_item("log_teff", st.final_history.log_teff)?;
        sdict.set_item("log_r", st.final_history.log_r)?;
        sdict.set_item("log_center_t", st.final_history.log_center_t)?;
        sdict.set_item("log_center_rho", st.final_history.log_center_rho)?;
        sdict.set_item("center_h1", st.final_history.center_h1)?;
        sdict.set_item("center_he4", st.final_history.center_he4)?;
        sdict.set_item("num_zones", st.final_history.num_zones)?;
        stages_list.append(sdict)?;
    }
    out.set_item("stages", stages_list)?;
    Ok(out)
}

/// Runs a multi-stage upstream MESA `test_suite` case directory and returns a
/// structured telemetry dictionary.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    skip_optional=true,
    max_steps_per_stage=None,
    verify_restart=false
))]
pub fn star_run_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let rep = py
        .detach(|| {
            run_test_suite_case(
                Path::new(case_dir),
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    case_report_to_pydict(py, &rep)
}

fn run_suite_to_pylist<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    cases: &'static [&'static str],
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    let reports = py
        .detach(|| {
            run_case_list(
                Path::new(mesa_dir),
                cases,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    let list = PyList::empty(py);
    for rep in &reports {
        list.append(case_report_to_pydict(py, rep)?)?;
    }
    Ok(list)
}

/// Returns the canonical 33-case Tier 1 `star/test_suite` inventory.
#[pyfunction]
pub fn star_tier1_cases() -> Vec<String> {
    TIER1_CASES.iter().map(|&s| String::from(s)).collect()
}

/// Runs all 33 Tier 1 `star/test_suite` cases under `mesa_dir/star/test_suite`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, max_steps_per_stage=Some(15), verify_restart=true))]
pub fn star_run_tier1_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    run_suite_to_pylist(
        py,
        mesa_dir,
        &TIER1_CASES,
        max_steps_per_stage,
        verify_restart,
    )
}

/// Evaluates the YREC starspot effective unspotted luminosity and blocking factor.
#[pyfunction]
pub fn star_eval_starspot_luminosity(
    l_surf: f64,
    f_spot: f64,
    x_spot: f64,
) -> (f64, f64) {
    eval_starspot_effective_luminosity(l_surf, f_spot, x_spot)
}

/// Evaluates the Hopf function $q(\tau)$ and $1 + \tau\,dq/d\tau$ factor.
#[pyfunction]
pub fn star_eval_t_tau_gradr_factor(
    tau: f64,
    scheme: &str,
) -> PyResult<(f64, f64)> {
    let rel = match scheme.to_ascii_lowercase().as_str() {
        "eddington" => AtmTTauRelation::Eddington,
        "krishna_swamy" | "krishnaswamy" => AtmTTauRelation::KrishnaSwamy,
        "solar_hopf_grey" | "solar_hopf" | "solarhopf" => {
            AtmTTauRelation::SolarHopf
        }
        "trampedach_solar" | "trampedach" => AtmTTauRelation::TrampedachSolar,
        other => {
            return Err(to_py_err(StarError::InvalidInput(format!(
                "unknown T(tau) relation '{other}'"
            ))));
        }
    };
    Ok(eval_t_tau_gradr_factor(tau, rel))
}

/// Evaluates the 10-column synthetic Johnson-Cousins photometry vector on `model`.
#[pyfunction]
pub fn star_eval_model_colors(model: &PyStarModel) -> PyResult<[f64; 10]> {
    eval_model_colors(&model.inner).map_err(to_py_err)
}

/// Evaluates the solar calibration $\chi^2$ objective.
#[pyfunction]
pub fn star_eval_solar_calibration_objective(
    log_l: f64,
    log_r: f64,
    zx_surf: f64,
    target_zx: f64,
) -> f64 {
    eval_solar_calibration_objective(log_l, log_r, zx_surf, target_zx)
}

/// Performs one Nelder-Mead simplex step on a 4-vertex solar calibration simplex.
#[pyfunction]
pub fn star_nelder_mead_solar_step(
    simplex: Vec<([f64; 3], f64)>,
) -> PyResult<[f64; 3]> {
    if simplex.len() != 4 {
        return Err(to_py_err(StarError::InvalidInput(String::from(
            "simplex must have exactly 4 vertices",
        ))));
    }
    let mut arr = [simplex[0], simplex[1], simplex[2], simplex[3]];
    Ok(nelder_mead_solar_calibration_step(&mut arr))
}

/// Returns the canonical 17-case Tier 2 `star/test_suite` inventory.
#[pyfunction]
pub fn star_tier2_cases() -> Vec<String> {
    TIER2_CASES.iter().map(|&s| String::from(s)).collect()
}

/// Runs all 17 Tier 2 `star/test_suite` cases under `mesa_dir/star/test_suite`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, max_steps_per_stage=Some(15), verify_restart=true))]
pub fn star_run_tier2_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    run_suite_to_pylist(
        py,
        mesa_dir,
        &TIER2_CASES,
        max_steps_per_stage,
        verify_restart,
    )
}

/// Evaluates planetary irradiation heating profile $\epsilon_{\mathrm{irr}}(k)$
/// and planetary radius in Jupiter radii $R / R_{\mathrm{Jup}}$.
#[pyfunction]
pub fn star_eval_planetary_irradiation_heating(
    model: &PyStarModel,
    f_irr: f64,
    col_depth_irr: f64,
) -> (Vec<f64>, f64) {
    eval_planetary_irradiation_heating(&model.inner, f_irr, col_depth_irr)
}

/// Evaluates white dwarf Coulomb crystallization and 2/3-component phase separation.
#[pyfunction]
#[pyo3(signature = (model, option="CO"))]
pub fn star_eval_wd_phase_separation(
    model: &mut PyStarModel,
    option: &str,
) -> (f64, f64) {
    eval_wd_phase_separation(&mut model.inner, option)
}

/// Evaluates the conductive carbon/oxygen flame speed $v_{\mathrm{flame}}$
/// ($\mathrm{cm\,s^{-1}}$) and flame mass coordinate $q_{\mathrm{flame}}$.
#[pyfunction]
pub fn star_eval_conductive_flame_state(model: &PyStarModel) -> (f64, f64) {
    eval_conductive_flame_state(&model.inner)
}

/// Returns the canonical 22-case Tier 3 `star/test_suite` inventory.
#[pyfunction]
pub fn star_tier3_cases() -> Vec<String> {
    TIER3_CASES.iter().map(|&s| String::from(s)).collect()
}

/// Runs all 22 Tier 3 `star/test_suite` cases under `mesa_dir/star/test_suite`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, max_steps_per_stage=Some(15), verify_restart=true))]
pub fn star_run_tier3_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    run_suite_to_pylist(
        py,
        mesa_dir,
        &TIER3_CASES,
        max_steps_per_stage,
        verify_restart,
    )
}

/// Evaluates the temperature-blended hot Vink + cool Reimers/Blöcker wind
/// mass-loss rate $\dot{M}$ ($M_\odot\,\mathrm{yr}^{-1}$) (`hot_cool_wind`).
#[pyfunction]
pub fn star_eval_hot_cool_blended_wind_mdot(model: &PyStarModel) -> f64 {
    eval_hot_cool_blended_wind_mdot(&model.inner)
}

/// Evaluates Roche oblateness $R_{\mathrm{eq}}/R_{\mathrm{pol}}$, $(f_P, f_T)$,
/// and Espinosa Lara & Rieutord (2011) equatorial-to-polar $T_{\mathrm{eff}}$
/// ratio (`high_rot_darkening`).
#[pyfunction]
pub fn star_eval_roche_gravity_darkening(
    omega_div_omega_crit: f64,
) -> (f64, f64, f64, f64) {
    eval_roche_gravity_darkening(omega_div_omega_crit)
}

/// Evaluates the 13 Spruit-Tayler magnetic dynamo and angular momentum
/// diagnostics (`15M_dynamo`, `20M_z2m2_high_rotation`).
#[pyfunction]
pub fn star_eval_spruit_tayler_dynamo_diagnostics(
    model: &PyStarModel,
) -> [f64; 13] {
    eval_spruit_tayler_dynamo_diagnostics(&model.inner)
}

/// Evaluates the ud-Doula, Owocki & Townsend (2009) magnetospheric wind
/// braking torque profile $\dot{j}_{\mathrm{extra}}(k)$ and braked surface
/// rotation velocity (`magnetic_braking`).
#[pyfunction]
#[pyo3(signature = (model, b_dipolar_gauss=5000.0))]
pub fn star_eval_magnetic_braking_torque(
    model: &PyStarModel,
    b_dipolar_gauss: f64,
) -> (Vec<f64>, f64) {
    eval_magnetic_braking_torque(&model.inner, b_dipolar_gauss)
}

/// Evaluates iron-group core mass $M_{\mathrm{Fe,core}}$ ($M_\odot$) and peak
/// core infall speed $v_{\mathrm{infall}}$ ($\mathrm{cm\,s^{-1}}$).
#[pyfunction]
pub fn star_eval_fe_core_infall_state(model: &PyStarModel) -> (f64, f64) {
    eval_fe_core_infall_state(&model.inner)
}

/// Evaluates supernova shock mass $M_{\mathrm{shock}}$ ($M_\odot$), peak
/// Duffell (2016) Rayleigh-Taylor diffusivity $D_{\mathrm{RTI}}$, and mixed
/// $^{1}\mathrm{H}$ mass fraction (`ccsn_IIp`).
#[pyfunction]
pub fn star_eval_duffell_rti_shock_state(
    model: &PyStarModel,
) -> (f64, f64, f64) {
    eval_duffell_rti_shock_state(&model.inner)
}

/// Exports `STELLA` radiation-hydrodynamics `.hyd` and `.abn` ASCII tables
/// (`ccsn_IIp`, Issue #4).
#[pyfunction]
pub fn star_export_stella_ascii(model: &PyStarModel) -> (String, String) {
    export_stella_ascii(&model.inner)
}

/// Evaluates pressure-weighted $\langle\Gamma_1 - 4/3\rangle_P$ and total
/// energy $E_{\mathrm{tot}}$ ($\mathrm{erg}$) (`pisn`, `ppisn`).
#[pyfunction]
pub fn star_eval_pair_instability_gamma1_and_energy(
    model: &PyStarModel,
) -> (f64, f64) {
    eval_pair_instability_gamma1_and_energy(&model.inner)
}

/// Evaluates the post-Newtonian TOV effective gravity profile $G_{\mathrm{eff}}(k)$
/// (`ns_h`, `ns_he`, `ns_c`, `tzo`).
#[pyfunction]
pub fn star_eval_tov_effective_cgrav(model: &PyStarModel) -> Vec<f64> {
    eval_tov_effective_cgrav(&model.inner)
}

/// Evaluates Thorne-Zytkow Eddington-limited neutron core accretion (`tzo`).
#[pyfunction]
#[pyo3(signature = (model, eta_ledd=1.0, eta_medd=1.0, target_mass_msun=1.4))]
pub fn star_eval_tzo_core_accretion_step(
    model: &mut PyStarModel,
    eta_ledd: f64,
    eta_medd: f64,
    target_mass_msun: f64,
) -> (f64, f64) {
    eval_tzo_core_accretion_step(
        &mut model.inner,
        eta_ledd,
        eta_medd,
        target_mass_msun,
    )
}

/// Registers `E2E-0`, `E2E-1`, `E2E-2`, and `E2E-3` runner functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_parse_inlist_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(star_parse_rn_stages, m)?)?;
    m.add_function(wrap_pyfunction!(star_verify_photo_restart, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_test_suite_case, m)?)?;
    m.add_function(wrap_pyfunction!(star_tier1_cases, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_tier1_suite, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_starspot_luminosity, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_t_tau_gradr_factor, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_model_colors, m)?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_solar_calibration_objective,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(star_nelder_mead_solar_step, m)?)?;
    m.add_function(wrap_pyfunction!(star_tier2_cases, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_tier2_suite, m)?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_planetary_irradiation_heating,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(star_eval_wd_phase_separation, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_conductive_flame_state, m)?)?;
    m.add_function(wrap_pyfunction!(star_tier3_cases, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_tier3_suite, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_hot_cool_blended_wind_mdot, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_roche_gravity_darkening, m)?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_spruit_tayler_dynamo_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(star_eval_magnetic_braking_torque, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_fe_core_infall_state, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_duffell_rti_shock_state, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_stella_ascii, m)?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_pair_instability_gamma1_and_energy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(star_eval_tov_effective_cgrav, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_tzo_core_accretion_step, m)?)?;
    Ok(())
}
