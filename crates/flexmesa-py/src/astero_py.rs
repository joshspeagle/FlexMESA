// SPDX-License-Identifier: GPL-3.0-or-later
//! `PyO3` bindings for Wave 5.4 `astero` scaling relations, surface corrections,
//! frequency ratios, $\chi^2$ evaluation, and optimization drivers.

use flexmesa_astero::{
    eval_acoustic_profile, eval_astero_chi2, eval_combined_corr,
    eval_cubic_corr, eval_delta_nu_asymptotic, eval_delta_nu_radius_scaling,
    eval_delta_nu_scaling, eval_delta_pg_bildsten2012,
    eval_delta_pg_traditional, eval_frequency_ratios_r01_r10,
    eval_frequency_ratios_r02, eval_glitch_acoustic_depths, eval_kjeldsen_corr,
    eval_nu_ac, eval_nu_max, eval_power_law_corr, eval_radial_delta_nu_ols,
    eval_sonoi_corr, match_modes_by_degree, scan_cartesian_grid,
    try_bobyqa_minimize, try_simplex_minimize_with_options, AsteroChi2Weights,
};
use flexmesa_star::StarError;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn as_slice<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray1<'_, T>,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn call_py_objective_res(
    py: Python<'_>,
    objective: &Bound<'_, PyAny>,
    x: &[f64],
    err_slot: &mut Option<PyErr>,
) -> Result<f64, StarError> {
    if err_slot.is_some() {
        return Err(StarError::InvalidInput("callback failed".to_owned()));
    }
    let arg = PyArray1::from_slice(py, x);
    match objective.call1((arg,)).and_then(|res| res.extract::<f64>()) {
        Ok(val) => Ok(val),
        Err(err) => {
            *err_slot = Some(err);
            Err(StarError::InvalidInput("callback failed".to_owned()))
        }
    }
}

fn call_py_objective(
    py: Python<'_>,
    objective: &Bound<'_, PyAny>,
    x: &[f64],
    err_slot: &mut Option<PyErr>,
) -> f64 {
    call_py_objective_res(py, objective, x, err_slot).unwrap_or(1.0e99)
}

fn validate_bounds_inputs(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    context: &str,
) -> PyResult<()> {
    if x_first.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{context} requires at least one parameter"
        )));
    }
    if x_first.len() != x_lower.len() || x_first.len() != x_upper.len() {
        return Err(PyValueError::new_err(format!(
            "{context}: x_first ({}), x_lower ({}), and x_upper ({}) must have equal length",
            x_first.len(),
            x_lower.len(),
            x_upper.len()
        )));
    }
    for i in 0..x_first.len() {
        if !x_first[i].is_finite()
            || !x_lower[i].is_finite()
            || !x_upper[i].is_finite()
            || x_lower[i] > x_upper[i]
        {
            return Err(PyValueError::new_err(format!(
                "{context}: invalid or inverted bounds at index {i}"
            )));
        }
    }
    Ok(())
}

fn optimizer_result_dict<'py>(
    py: Python<'py>,
    x_best: Vec<f64>,
    f_best: f64,
    num_evals: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("x_best", PyArray1::from_vec(py, x_best))?;
    d.set_item("f_best", f_best)?;
    d.set_item("num_evals", num_evals)?;
    Ok(d)
}

/// Evaluates the acoustic radius profile $\tau(r_k)$, acoustic depth profile
/// $\tau_{\mathrm{depth}}(r_k)$, and photospheric acoustic radius $\tau_0$ \[`s`\].
#[pyfunction]
pub fn astero_eval_acoustic_profile<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    csound: PyReadonlyArray1<'py, f64>,
    r_center: f64,
    r_phot: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r_s = as_slice(&r)?;
    let cs_s = as_slice(&csound)?;
    if r_s.len() != cs_s.len() {
        return Err(PyValueError::new_err(format!(
            "r ({}) and csound ({}) must have equal length",
            r_s.len(),
            cs_s.len()
        )));
    }
    let (tau_r, tau_depth, tau_0) =
        eval_acoustic_profile(r_s, cs_s, r_center, r_phot);
    let d = PyDict::new(py);
    d.set_item("acoustic_radius", PyArray1::from_vec(py, tau_r))?;
    d.set_item("acoustic_depth", PyArray1::from_vec(py, tau_depth))?;
    d.set_item("photosphere_acoustic_r", tau_0)?;
    Ok(d)
}

/// Evaluates asymptotic $\Delta\nu = 10^6 / (2 \tau_0)$ \[`\muHz`\].
#[pyfunction]
#[must_use]
pub fn astero_eval_delta_nu_asymptotic(photosphere_acoustic_r: f64) -> f64 {
    eval_delta_nu_asymptotic(photosphere_acoustic_r)
}

/// Evaluates solar-scaled $\Delta\nu(M, T_{\mathrm{eff}}, L)$ \[`\muHz`\].
#[pyfunction]
#[must_use]
pub fn astero_eval_delta_nu_scaling(
    m_g: f64,
    teff_k: f64,
    l_phot_erg_s: f64,
) -> f64 {
    eval_delta_nu_scaling(m_g, teff_k, l_phot_erg_s)
}

/// Evaluates solar-scaled $\Delta\nu(M, R_{\mathrm{phot}})$ \[`\muHz`\].
#[pyfunction]
#[must_use]
pub fn astero_eval_delta_nu_radius_scaling(m_g: f64, r_phot_cm: f64) -> f64 {
    eval_delta_nu_radius_scaling(m_g, r_phot_cm)
}

/// Evaluates frequency of maximum oscillation power $\nu_{\max}$ \[`\muHz`\].
#[pyfunction]
#[must_use]
pub fn astero_eval_nu_max(m_g: f64, r_phot_cm: f64, teff_k: f64) -> f64 {
    eval_nu_max(m_g, r_phot_cm, teff_k)
}

/// Evaluates surface acoustic cutoff frequency $\nu_{\mathrm{ac}}$ \[`\muHz`\].
#[pyfunction]
#[must_use]
pub fn astero_eval_nu_ac(
    g_surf_cgs: f64,
    gamma1_surf: f64,
    rho_surf: f64,
    peos_surf: f64,
) -> f64 {
    eval_nu_ac(g_surf_cgs, gamma1_surf, rho_surf, peos_surf)
}

/// Evaluates asymptotic dipole ($l=1$) g-mode spacing $\Delta\Pi_1$ \[`s`\].
#[pyfunction]
pub fn astero_eval_delta_pg(
    r: PyReadonlyArray1<'_, f64>,
    r_mid: PyReadonlyArray1<'_, f64>,
    brunt_n2: PyReadonlyArray1<'_, f64>,
    csound_face: PyReadonlyArray1<'_, f64>,
    mode_freq_uhz: f64,
    bildsten2012: bool,
) -> PyResult<f64> {
    let r_s = as_slice(&r)?;
    let rm_s = as_slice(&r_mid)?;
    let n2_s = as_slice(&brunt_n2)?;
    let cs_s = as_slice(&csound_face)?;
    if rm_s.len() != r_s.len()
        || n2_s.len() != r_s.len()
        || (bildsten2012 && cs_s.len() != r_s.len())
    {
        return Err(PyValueError::new_err(
            "r, r_mid, brunt_n2 (and csound_face for Bildsten2012) must have equal length",
        ));
    }
    if bildsten2012 {
        Ok(eval_delta_pg_bildsten2012(
            r_s,
            rm_s,
            n2_s,
            cs_s,
            mode_freq_uhz,
        ))
    } else {
        Ok(eval_delta_pg_traditional(r_s, rm_s, n2_s))
    }
}

/// Evaluates glitch locations $(R_{\mathrm{BCZ}}/R_\odot, \tau_{\mathrm{BCZ}}, \tau_{\mathrm{HeII}})$.
#[pyfunction]
pub fn astero_eval_glitch_acoustic_depths(
    r: PyReadonlyArray1<'_, f64>,
    csound: PyReadonlyArray1<'_, f64>,
    gradr: PyReadonlyArray1<'_, f64>,
    grada: PyReadonlyArray1<'_, f64>,
    gamma1: PyReadonlyArray1<'_, f64>,
    t: PyReadonlyArray1<'_, f64>,
    r_center: f64,
) -> PyResult<(f64, f64, f64)> {
    let r_s = as_slice(&r)?;
    let cs_s = as_slice(&csound)?;
    let gr_s = as_slice(&gradr)?;
    let ga_s = as_slice(&grada)?;
    let g1_s = as_slice(&gamma1)?;
    let t_s = as_slice(&t)?;
    let nz = r_s.len();
    if cs_s.len() != nz
        || gr_s.len() != nz
        || ga_s.len() != nz
        || g1_s.len() != nz
        || t_s.len() != nz
    {
        return Err(PyValueError::new_err(
            "r, csound, gradr, grada, gamma1, and t must have equal length",
        ));
    }
    Ok(eval_glitch_acoustic_depths(
        r_s, cs_s, gr_s, ga_s, g1_s, t_s, r_center,
    ))
}

/// Evaluates OLS radial large frequency separation $\Delta\nu_{\mathrm{model}}$ \[`\muHz`\].
#[pyfunction]
pub fn astero_eval_radial_delta_nu_ols(
    n_orders: PyReadonlyArray1<'_, f64>,
    model_freqs: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let n_s = as_slice(&n_orders)?;
    let f_s = as_slice(&model_freqs)?;
    if n_s.len() != f_s.len() {
        return Err(PyValueError::new_err(format!(
            "n_orders ({}) and model_freqs ({}) must have equal length",
            n_s.len(),
            f_s.len()
        )));
    }
    Ok(eval_radial_delta_nu_ols(n_s, f_s))
}

/// Matches observed mode frequencies to candidate model modes (`find_closest`).
#[pyfunction]
pub fn astero_match_modes_by_degree<'py>(
    py: Python<'py>,
    obs_freqs: PyReadonlyArray1<'py, f64>,
    cand_freqs: PyReadonlyArray1<'py, f64>,
    cand_inertias: PyReadonlyArray1<'py, f64>,
    cand_orders: PyReadonlyArray1<'py, i32>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let obs_s = as_slice(&obs_freqs)?;
    let cf_s = as_slice(&cand_freqs)?;
    let ci_s = as_slice(&cand_inertias)?;
    let co_s = as_slice(&cand_orders)?;
    if cf_s.len() != ci_s.len() || cf_s.len() != co_s.len() {
        return Err(PyValueError::new_err(
            "cand_freqs, cand_inertias, and cand_orders must have equal length",
        ));
    }
    let (mf, mi, mo) = match_modes_by_degree(obs_s, cf_s, ci_s, co_s);
    Ok((
        PyArray1::from_vec(py, mf),
        PyArray1::from_vec(py, mi),
        PyArray1::from_vec(py, mo),
    ))
}

/// Ensures observed and model mode arrays share equal lengths.
fn ensure_surface_mode_lengths(
    obs_f: &[f64],
    obs_s: &[f64],
    mod_f: &[f64],
    mod_i: &[f64],
) -> PyResult<()> {
    if obs_f.len() != obs_s.len()
        || obs_f.len() != mod_f.len()
        || obs_f.len() != mod_i.len()
    {
        return Err(PyValueError::new_err(
            "obs_freqs, obs_sigmas, model_freqs, and model_inertias must have equal length",
        ));
    }
    Ok(())
}

/// Ensures radial frequency and inertia arrays share equal lengths.
fn ensure_radial_inertia_lengths(
    scheme: &str,
    l0_mf: &[f64],
    l0_mi: &[f64],
) -> PyResult<()> {
    if l0_mf.len() != l0_mi.len() {
        return Err(PyValueError::new_err(format!(
            "mismatched mode or radial inertia array lengths for {scheme} surface correction"
        )));
    }
    Ok(())
}

/// Evaluates a surface-effect frequency correction (`kjeldsen`, `cubic`,
/// `combined`, `power_law`, or `sonoi`).
#[pyfunction]
pub fn astero_eval_surface_correction<'py>(
    py: Python<'py>,
    scheme: &str,
    obs_freqs: PyReadonlyArray1<'py, f64>,
    obs_sigmas: PyReadonlyArray1<'py, f64>,
    model_freqs: PyReadonlyArray1<'py, f64>,
    model_inertias: PyReadonlyArray1<'py, f64>,
    l0_obs_freqs: PyReadonlyArray1<'py, f64>,
    l0_model_freqs: PyReadonlyArray1<'py, f64>,
    l0_inertias: PyReadonlyArray1<'py, f64>,
    l0_orders: PyReadonlyArray1<'py, f64>,
    delta_nu_obs: f64,
    nu_max: f64,
    b_power: f64,
    correction_factor: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let obs_f = as_slice(&obs_freqs)?;
    let obs_s = as_slice(&obs_sigmas)?;
    let mod_f = as_slice(&model_freqs)?;
    let mod_i = as_slice(&model_inertias)?;
    let l0_of = as_slice(&l0_obs_freqs)?;
    let l0_mf = as_slice(&l0_model_freqs)?;
    let l0_mi = as_slice(&l0_inertias)?;
    let l0_or = as_slice(&l0_orders)?;

    let (corr, p0, p1) = match scheme {
        "kjeldsen" => {
            if mod_f.len() != mod_i.len()
                || l0_of.len() != l0_mf.len()
                || (!l0_mi.is_empty() && l0_mi.len() != l0_mf.len())
                || (!l0_or.is_empty() && l0_or.len() != l0_of.len())
            {
                return Err(PyValueError::new_err(
                    "mismatched mode array lengths for kjeldsen surface correction",
                ));
            }
            eval_kjeldsen_corr(
                l0_of,
                l0_mf,
                l0_mi,
                l0_or,
                delta_nu_obs,
                mod_f,
                mod_i,
                nu_max,
                b_power,
                correction_factor,
            )
        }
        "cubic" | "bg14_1" => {
            ensure_surface_mode_lengths(obs_f, obs_s, mod_f, mod_i)?;
            let (c, a3) =
                eval_cubic_corr(obs_f, obs_s, mod_f, mod_i, correction_factor);
            (c, a3, 0.0)
        }
        "combined" | "bg14_2" => {
            ensure_surface_mode_lengths(obs_f, obs_s, mod_f, mod_i)?;
            eval_combined_corr(obs_f, obs_s, mod_f, mod_i, correction_factor)
        }
        "power_law" => {
            ensure_surface_mode_lengths(obs_f, obs_s, mod_f, mod_i)?;
            ensure_radial_inertia_lengths(scheme, l0_mf, l0_mi)?;
            eval_power_law_corr(
                obs_f,
                obs_s,
                mod_f,
                mod_i,
                l0_mf,
                l0_mi,
                nu_max,
                correction_factor,
            )
        }
        "sonoi" => {
            ensure_surface_mode_lengths(obs_f, obs_s, mod_f, mod_i)?;
            ensure_radial_inertia_lengths(scheme, l0_mf, l0_mi)?;
            eval_sonoi_corr(
                obs_f,
                obs_s,
                mod_f,
                mod_i,
                l0_mf,
                l0_mi,
                nu_max,
                correction_factor,
            )
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown surface correction scheme: {other}"
            )));
        }
    };
    let d = PyDict::new(py);
    d.set_item("nu_corrected", PyArray1::from_vec(py, corr))?;
    d.set_item("p0", p0)?;
    d.set_item("p1", p1)?;
    Ok(d)
}

/// Evaluates 5-point frequency separation ratios $r_{01}, r_{10}$.
#[pyfunction]
pub fn astero_eval_ratios_r01_r10<'py>(
    py: Python<'py>,
    l0_freqs: PyReadonlyArray1<'py, f64>,
    l0_sigs: PyReadonlyArray1<'py, f64>,
    l1_freqs: PyReadonlyArray1<'py, f64>,
    l1_sigs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let l0_f = as_slice(&l0_freqs)?;
    let l0_s = as_slice(&l0_sigs)?;
    let l1_f = as_slice(&l1_freqs)?;
    let l1_s = as_slice(&l1_sigs)?;
    if l0_f.len() != l0_s.len() || l1_f.len() != l1_s.len() {
        return Err(PyValueError::new_err(
            "frequency and sigma arrays must have equal length for l=0 and l=1",
        ));
    }
    let (r01, sig_r01, r10, sig_r10) =
        eval_frequency_ratios_r01_r10(l0_f, l0_s, l1_f, l1_s);
    let d = PyDict::new(py);
    d.set_item("r01", PyArray1::from_vec(py, r01))?;
    d.set_item("sig_r01", PyArray1::from_vec(py, sig_r01))?;
    d.set_item("r10", PyArray1::from_vec(py, r10))?;
    d.set_item("sig_r10", PyArray1::from_vec(py, sig_r10))?;
    Ok(d)
}

/// Evaluates quadrupole-radial separation ratios $r_{02}(n)$ and uncertainties.
#[pyfunction]
pub fn astero_eval_ratios_r02<'py>(
    py: Python<'py>,
    l0_freqs: PyReadonlyArray1<'py, f64>,
    l0_sigs: PyReadonlyArray1<'py, f64>,
    l1_freqs: PyReadonlyArray1<'py, f64>,
    l1_sigs: PyReadonlyArray1<'py, f64>,
    l2_freqs: PyReadonlyArray1<'py, f64>,
    l2_sigs: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let l0_f = as_slice(&l0_freqs)?;
    let l0_s = as_slice(&l0_sigs)?;
    let l1_f = as_slice(&l1_freqs)?;
    let l1_s = as_slice(&l1_sigs)?;
    let l2_f = as_slice(&l2_freqs)?;
    let l2_s = as_slice(&l2_sigs)?;
    if l0_f.len() != l0_s.len()
        || l1_f.len() != l1_s.len()
        || l2_f.len() != l2_s.len()
    {
        return Err(PyValueError::new_err(
            "frequency and sigma arrays must have equal length for l=0, l=1, and l=2",
        ));
    }
    let (r02, sig_r02) =
        eval_frequency_ratios_r02(l0_f, l0_s, l1_f, l1_s, l2_f, l2_s);
    Ok((PyArray1::from_vec(py, r02), PyArray1::from_vec(py, sig_r02)))
}

/// Evaluates composite asteroseismic + spectroscopic $\chi^2$ (`get_chi2`).
#[pyfunction]
pub fn astero_eval_chi2<'py>(
    py: Python<'py>,
    obs_freqs: PyReadonlyArray1<'py, f64>,
    obs_sigmas: PyReadonlyArray1<'py, f64>,
    corr_model_freqs: PyReadonlyArray1<'py, f64>,
    r010_obs: PyReadonlyArray1<'py, f64>,
    r010_sigmas: PyReadonlyArray1<'py, f64>,
    r010_model: PyReadonlyArray1<'py, f64>,
    r02_obs: PyReadonlyArray1<'py, f64>,
    r02_sigmas: PyReadonlyArray1<'py, f64>,
    r02_model: PyReadonlyArray1<'py, f64>,
    delta_nu_obs: f64,
    delta_nu_sigma: f64,
    delta_nu_model: f64,
    nu_max_obs: f64,
    nu_max_sigma: f64,
    nu_max_model: f64,
    spectro_obs: PyReadonlyArray1<'py, f64>,
    spectro_sigmas: PyReadonlyArray1<'py, f64>,
    spectro_model: PyReadonlyArray1<'py, f64>,
    chi2_seismo_fraction: f64,
    chi2_seismo_delta_nu_fraction: f64,
    chi2_seismo_nu_max_fraction: f64,
    chi2_seismo_r_010_fraction: f64,
    chi2_seismo_r_02_fraction: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let weights = AsteroChi2Weights {
        chi2_seismo_fraction,
        chi2_seismo_delta_nu_fraction,
        chi2_seismo_nu_max_fraction,
        chi2_seismo_r_010_fraction,
        chi2_seismo_r_02_fraction,
        ..AsteroChi2Weights::default()
    };
    let obs_f = as_slice(&obs_freqs)?;
    let obs_s = as_slice(&obs_sigmas)?;
    let corr_f = as_slice(&corr_model_freqs)?;
    let r010_o = as_slice(&r010_obs)?;
    let r010_s = as_slice(&r010_sigmas)?;
    let r010_m = as_slice(&r010_model)?;
    let r02_o = as_slice(&r02_obs)?;
    let r02_s = as_slice(&r02_sigmas)?;
    let r02_m = as_slice(&r02_model)?;
    let spec_o = as_slice(&spectro_obs)?;
    let spec_s = as_slice(&spectro_sigmas)?;
    let spec_m = as_slice(&spectro_model)?;
    if obs_f.len() != obs_s.len()
        || obs_f.len() != corr_f.len()
        || r010_o.len() != r010_s.len()
        || r010_o.len() != r010_m.len()
        || r02_o.len() != r02_s.len()
        || r02_o.len() != r02_m.len()
        || spec_o.len() != spec_s.len()
        || spec_o.len() != spec_m.len()
    {
        return Err(PyValueError::new_err(
            "observed, sigma, and model arrays must have equal length in each chi2 term",
        ));
    }
    let (chi2_total, chi2_seismo, chi2_spectro, chi2_freq, chi2_r010, chi2_r02) =
        eval_astero_chi2(
            &weights,
            obs_f,
            obs_s,
            corr_f,
            r010_o,
            r010_s,
            r010_m,
            r02_o,
            r02_s,
            r02_m,
            delta_nu_obs,
            delta_nu_sigma,
            delta_nu_model,
            nu_max_obs,
            nu_max_sigma,
            nu_max_model,
            spec_o,
            spec_s,
            spec_m,
        );
    let d = PyDict::new(py);
    d.set_item("chi2_total", chi2_total)?;
    d.set_item("chi2_seismo", chi2_seismo)?;
    d.set_item("chi2_spectro", chi2_spectro)?;
    d.set_item("chi2_frequencies", chi2_freq)?;
    d.set_item("chi2_r_010", chi2_r010)?;
    d.set_item("chi2_r_02", chi2_r02)?;
    Ok(d)
}

/// Minimizes `objective(x_vec) -> float` via Nelder-Mead (`mod_simplex.f90`).
#[pyfunction]
pub fn astero_nelder_mead_minimize<'py>(
    py: Python<'py>,
    objective: &Bound<'py, PyAny>,
    x_first: Vec<f64>,
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
    max_evals: usize,
    x_atol: f64,
    x_rtol: f64,
    chi2_tol: f64,
    centroid_weight_power: f64,
) -> PyResult<Bound<'py, PyDict>> {
    validate_bounds_inputs(
        &x_first,
        &x_lower,
        &x_upper,
        "astero_nelder_mead_minimize",
    )?;
    let mut cb_err: Option<PyErr> = None;
    let res = try_simplex_minimize_with_options(
        &x_first,
        &x_lower,
        &x_upper,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        true,
        false,
        107_461_921,
        |x| call_py_objective_res(py, objective, x, &mut cb_err),
    );
    if let Some(err) = cb_err {
        return Err(err);
    }
    let (x_best, f_best, evals) =
        res.map_err(|e| PyValueError::new_err(e.to_string()))?;
    optimizer_result_dict(py, x_best, f_best, evals)
}

/// Minimizes `objective(x_vec) -> float` via Powell's quadratic trust-region.
#[pyfunction]
pub fn astero_powell_quadratic_minimize<'py>(
    py: Python<'py>,
    objective: &Bound<'py, PyAny>,
    x_first: Vec<f64>,
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
    max_evals: usize,
    rhobeg: f64,
    rhoend: f64,
) -> PyResult<Bound<'py, PyDict>> {
    validate_bounds_inputs(
        &x_first,
        &x_lower,
        &x_upper,
        "astero_powell_quadratic_minimize",
    )?;
    if !rhobeg.is_finite()
        || rhobeg <= 0.0
        || !rhoend.is_finite()
        || rhoend <= 0.0
    {
        return Err(PyValueError::new_err(
            "rhobeg and rhoend must be finite and strictly positive",
        ));
    }
    let mut cb_err: Option<PyErr> = None;
    let res = try_bobyqa_minimize(
        &x_first,
        &x_lower,
        &x_upper,
        max_evals,
        rhobeg,
        rhoend,
        |x| call_py_objective_res(py, objective, x, &mut cb_err),
    );
    if let Some(err) = cb_err {
        return Err(err);
    }
    let (x_best, f_best, evals) =
        res.map_err(|e| PyValueError::new_err(e.to_string()))?;
    optimizer_result_dict(py, x_best, f_best, evals)
}

/// Scans a Cartesian parameter grid using `objective(x_vec) -> float`.
#[pyfunction]
pub fn astero_scan_cartesian_grid<'py>(
    py: Python<'py>,
    objective: &Bound<'py, PyAny>,
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
    x_delta: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if x_lower.is_empty() {
        return Err(PyValueError::new_err(
            "astero_scan_cartesian_grid requires at least one parameter",
        ));
    }
    if x_lower.len() != x_upper.len() || x_lower.len() != x_delta.len() {
        return Err(PyValueError::new_err(
            "x_lower, x_upper, and x_delta must have equal length",
        ));
    }
    for i in 0..x_lower.len() {
        if !x_lower[i].is_finite()
            || !x_upper[i].is_finite()
            || !x_delta[i].is_finite()
            || x_lower[i] > x_upper[i]
            || (x_upper[i] > x_lower[i] && x_delta[i] <= 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "invalid scan grid bounds or step at index {i}"
            )));
        }
    }
    let mut cb_err: Option<PyErr> = None;
    let (x_best, f_best, _, all_vals) =
        scan_cartesian_grid(&x_lower, &x_upper, &x_delta, |x| {
            call_py_objective(py, objective, x, &mut cb_err)
        });
    if let Some(err) = cb_err {
        return Err(err);
    }
    let d = optimizer_result_dict(py, x_best, f_best, all_vals.len())?;
    d.set_item("all_values", PyArray1::from_vec(py, all_vals))?;
    Ok(d)
}

/// Registers Wave 5.4 `astero` bindings on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(astero_eval_acoustic_profile, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_delta_nu_asymptotic, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_delta_nu_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_delta_nu_radius_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_nu_max, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_nu_ac, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_delta_pg, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_glitch_acoustic_depths, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_radial_delta_nu_ols, m)?)?;
    m.add_function(wrap_pyfunction!(astero_match_modes_by_degree, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_surface_correction, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_ratios_r01_r10, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_ratios_r02, m)?)?;
    m.add_function(wrap_pyfunction!(astero_eval_chi2, m)?)?;
    m.add_function(wrap_pyfunction!(astero_nelder_mead_minimize, m)?)?;
    m.add_function(wrap_pyfunction!(astero_powell_quadratic_minimize, m)?)?;
    m.add_function(wrap_pyfunction!(astero_scan_cartesian_grid, m)?)?;
    Ok(())
}
