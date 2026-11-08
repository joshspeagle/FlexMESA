// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for Wave 5.4 `astero` scaling relations, surface corrections,
//! frequency ratios, $\chi^2$ evaluation, and optimization drivers.

use flexmesa_astero::{
    AsteroChi2Weights, eval_acoustic_profile, eval_astero_chi2,
    eval_combined_corr, eval_cubic_corr, eval_delta_nu_asymptotic,
    eval_delta_nu_radius_scaling, eval_delta_nu_scaling,
    eval_delta_pg_bildsten2012, eval_delta_pg_traditional,
    eval_frequency_ratios_r01_r10, eval_frequency_ratios_r02,
    eval_glitch_acoustic_depths, eval_kjeldsen_corr, eval_nu_ac, eval_nu_max,
    eval_power_law_corr, eval_radial_delta_nu_ols, eval_sonoi_corr,
    match_modes_by_degree, nelder_mead_minimize, powell_quadratic_minimize,
    scan_cartesian_grid,
};
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

fn call_py_objective(
    py: Python<'_>,
    objective: &Bound<'_, PyAny>,
    x: &[f64],
    err_slot: &mut Option<PyErr>,
) -> f64 {
    if err_slot.is_some() {
        return 1.0e99;
    }
    let arg = PyArray1::from_slice(py, x);
    match objective.call1((arg,)).and_then(|res| res.extract::<f64>()) {
        Ok(val) => val,
        Err(err) => {
            *err_slot = Some(err);
            1.0e99
        }
    }
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
/// $\tau_{\mathrm{depth}}(r_k)$, and photospheric acoustic radius $\tau_0$ [`s`].
#[pyfunction]
pub fn astero_eval_acoustic_profile<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    csound: PyReadonlyArray1<'py, f64>,
    r_center: f64,
    r_phot: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (tau_r, tau_depth, tau_0) = eval_acoustic_profile(
        as_slice(&r)?,
        as_slice(&csound)?,
        r_center,
        r_phot,
    );
    let d = PyDict::new(py);
    d.set_item("acoustic_radius", PyArray1::from_vec(py, tau_r))?;
    d.set_item("acoustic_depth", PyArray1::from_vec(py, tau_depth))?;
    d.set_item("photosphere_acoustic_r", tau_0)?;
    Ok(d)
}

/// Evaluates asymptotic $\Delta\nu = 10^6 / (2 \tau_0)$ [`\muHz`].
#[pyfunction]
#[must_use]
pub fn astero_eval_delta_nu_asymptotic(photosphere_acoustic_r: f64) -> f64 {
    eval_delta_nu_asymptotic(photosphere_acoustic_r)
}

/// Evaluates solar-scaled $\Delta\nu(M, T_{\mathrm{eff}}, L)$ [`\muHz`].
#[pyfunction]
#[must_use]
pub fn astero_eval_delta_nu_scaling(
    m_g: f64,
    teff_k: f64,
    l_phot_erg_s: f64,
) -> f64 {
    eval_delta_nu_scaling(m_g, teff_k, l_phot_erg_s)
}

/// Evaluates solar-scaled $\Delta\nu(M, R_{\mathrm{phot}})$ [`\muHz`].
#[pyfunction]
#[must_use]
pub fn astero_eval_delta_nu_radius_scaling(m_g: f64, r_phot_cm: f64) -> f64 {
    eval_delta_nu_radius_scaling(m_g, r_phot_cm)
}

/// Evaluates frequency of maximum oscillation power $\nu_{\max}$ [`\muHz`].
#[pyfunction]
#[must_use]
pub fn astero_eval_nu_max(m_g: f64, r_phot_cm: f64, teff_k: f64) -> f64 {
    eval_nu_max(m_g, r_phot_cm, teff_k)
}

/// Evaluates surface acoustic cutoff frequency $\nu_{\mathrm{ac}}$ [`\muHz`].
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

/// Evaluates asymptotic dipole ($l=1$) g-mode spacing $\Delta\Pi_1$ [`s`].
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
    Ok(eval_glitch_acoustic_depths(
        as_slice(&r)?,
        as_slice(&csound)?,
        as_slice(&gradr)?,
        as_slice(&grada)?,
        as_slice(&gamma1)?,
        as_slice(&t)?,
        r_center,
    ))
}

/// Evaluates OLS radial large frequency separation $\Delta\nu_{\mathrm{model}}$ [`\muHz`].
#[pyfunction]
pub fn astero_eval_radial_delta_nu_ols(
    n_orders: PyReadonlyArray1<'_, f64>,
    model_freqs: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    Ok(eval_radial_delta_nu_ols(
        as_slice(&n_orders)?,
        as_slice(&model_freqs)?,
    ))
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
    let (mf, mi, mo) = match_modes_by_degree(
        as_slice(&obs_freqs)?,
        as_slice(&cand_freqs)?,
        as_slice(&cand_inertias)?,
        as_slice(&cand_orders)?,
    );
    Ok((
        PyArray1::from_vec(py, mf),
        PyArray1::from_vec(py, mi),
        PyArray1::from_vec(py, mo),
    ))
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
        "kjeldsen" => eval_kjeldsen_corr(
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
        ),
        "cubic" | "bg14_1" => {
            let (c, a3) =
                eval_cubic_corr(obs_f, obs_s, mod_f, mod_i, correction_factor);
            (c, a3, 0.0)
        }
        "combined" | "bg14_2" => {
            eval_combined_corr(obs_f, obs_s, mod_f, mod_i, correction_factor)
        }
        "power_law" => eval_power_law_corr(
            obs_f,
            obs_s,
            mod_f,
            mod_i,
            l0_mf,
            l0_mi,
            nu_max,
            correction_factor,
        ),
        "sonoi" => eval_sonoi_corr(
            obs_f,
            obs_s,
            mod_f,
            mod_i,
            l0_mf,
            l0_mi,
            nu_max,
            correction_factor,
        ),
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
    let (r01, sig_r01, r10, sig_r10) = eval_frequency_ratios_r01_r10(
        as_slice(&l0_freqs)?,
        as_slice(&l0_sigs)?,
        as_slice(&l1_freqs)?,
        as_slice(&l1_sigs)?,
    );
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
    let (r02, sig_r02) = eval_frequency_ratios_r02(
        as_slice(&l0_freqs)?,
        as_slice(&l0_sigs)?,
        as_slice(&l1_freqs)?,
        as_slice(&l1_sigs)?,
        as_slice(&l2_freqs)?,
        as_slice(&l2_sigs)?,
    );
    Ok((
        PyArray1::from_vec(py, r02),
        PyArray1::from_vec(py, sig_r02),
    ))
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
    let (chi2_total, chi2_seismo, chi2_spectro, chi2_freq, chi2_r010, chi2_r02) =
        eval_astero_chi2(
            &weights,
            as_slice(&obs_freqs)?,
            as_slice(&obs_sigmas)?,
            as_slice(&corr_model_freqs)?,
            as_slice(&r010_obs)?,
            as_slice(&r010_sigmas)?,
            as_slice(&r010_model)?,
            as_slice(&r02_obs)?,
            as_slice(&r02_sigmas)?,
            as_slice(&r02_model)?,
            delta_nu_obs,
            delta_nu_sigma,
            delta_nu_model,
            nu_max_obs,
            nu_max_sigma,
            nu_max_model,
            as_slice(&spectro_obs)?,
            as_slice(&spectro_sigmas)?,
            as_slice(&spectro_model)?,
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
    let mut cb_err: Option<PyErr> = None;
    let (x_best, f_best, evals) = nelder_mead_minimize(
        &x_first,
        &x_lower,
        &x_upper,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        |x| call_py_objective(py, objective, x, &mut cb_err),
    );
    if let Some(err) = cb_err {
        return Err(err);
    }
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
    let mut cb_err: Option<PyErr> = None;
    let (x_best, f_best, evals) = powell_quadratic_minimize(
        &x_first,
        &x_lower,
        &x_upper,
        max_evals,
        rhobeg,
        rhoend,
        |x| call_py_objective(py, objective, x, &mut cb_err),
    );
    if let Some(err) = cb_err {
        return Err(err);
    }
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
