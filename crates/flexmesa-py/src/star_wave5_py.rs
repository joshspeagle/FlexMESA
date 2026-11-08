// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for Wave 5 `flexmesa-star` extensions: multi-species Burgers
//! element diffusion, shellular rotation distortion and Spruit-Tayler magnetic
//! dynamo, convective premixing, predictive mixing, and white-dwarf
//! crystallization phase separation.

use flexmesa_star::{
    add_predictive_mixing, do_conv_premix, do_element_diffusion,
    do_phase_separation, eval_burgers_velocities, eval_caplan_ocp_dstar,
    eval_castellani_penetration_vel, eval_omega_crit,
    eval_phase_separation_shift, eval_radiative_levitation_accel,
    eval_rotation_fp_ft, eval_rotational_mixing_zone,
    eval_wd_sedimentation_heating, set_rotation_mixing_info, DiffusionScheme,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::star_py::PyStarModel;

fn parse_diffusion_scheme(scheme: &str) -> PyResult<DiffusionScheme> {
    match scheme.to_ascii_lowercase().as_str() {
        "iben_macdonald" | "iben" => Ok(DiffusionScheme::IbenMacDonald),
        "paquette" => Ok(DiffusionScheme::Paquette),
        "stanton_murillo" | "stanton" => Ok(DiffusionScheme::StantonMurillo),
        "caplan_ocp" | "caplan" | "ocp" => Ok(DiffusionScheme::CaplanOcp),
        other => Err(PyValueError::new_err(format!(
            "unknown diffusion scheme '{other}'; expected 'iben_macdonald', \
             'paquette', 'stanton_murillo', or 'caplan_ocp'"
        ))),
    }
}

/// Solves the single-zone multi-species Burgers resistance equations and
/// returns `(v_gt, d_self, ee_force)`.
#[pyfunction]
#[pyo3(signature = (
    a, z, x_mass, rho, t, dlnt_dr, g_grav, g_rad=None,
    scheme="stanton_murillo"
))]
pub fn star_eval_burgers_velocities<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    x_mass: PyReadonlyArray1<'py, f64>,
    rho: f64,
    t: f64,
    dlnt_dr: f64,
    g_grav: f64,
    g_rad: Option<PyReadonlyArray1<'py, f64>>,
    scheme: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let a_s = a.as_slice()?;
    let z_s = z.as_slice()?;
    let x_s = x_mass.as_slice()?;
    if a_s.len() != z_s.len() || a_s.len() != x_s.len() {
        return Err(PyValueError::new_err(
            "a, z, and x_mass must have identical lengths",
        ));
    }
    let grad_slice = match g_rad.as_ref() {
        Some(gr) => {
            let s = gr.as_slice()?;
            if s.len() != a_s.len() {
                return Err(PyValueError::new_err(
                    "g_rad must have the same length as a",
                ));
            }
            Some(s)
        }
        None => None,
    };
    let sch = parse_diffusion_scheme(scheme)?;
    let res = eval_burgers_velocities(
        a_s, z_s, x_s, rho, t, dlnt_dr, g_grav, grad_slice, sch,
    );
    Ok((
        PyArray1::from_vec(py, res.v_gt),
        PyArray1::from_vec(py, res.d_self),
        res.ee_force,
    ))
}

/// Evaluates the Caplan, Bauer & Freeman (2022) strong-coupling OCP
/// dimensionless self-diffusion coefficient $D^*(\Gamma, \kappa)$.
#[pyfunction]
pub fn star_eval_caplan_ocp_dstar(gamma_i: f64, kappa: f64) -> f64 {
    eval_caplan_ocp_dstar(gamma_i, kappa)
}

/// Evaluates the radiative levitation acceleration $g_{\mathrm{rad},i}$ in
/// $\mathrm{cm\,s^{-2}}$.
#[pyfunction]
pub fn star_eval_radiative_levitation_accel(
    kappa_species: f64,
    lum_face: f64,
    r_face: f64,
) -> f64 {
    eval_radiative_levitation_accel(kappa_species, lum_face, r_face)
}

/// Evaluates the white-dwarf gravitational sedimentation heating rate
/// $\epsilon_{\mathrm{WD,sed}}$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$.
#[pyfunction]
pub fn star_eval_wd_sedimentation_heating<'py>(
    a: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    x_mass: PyReadonlyArray1<'py, f64>,
    v_gt: PyReadonlyArray1<'py, f64>,
    abar: f64,
    zbar: f64,
    g_grav: f64,
) -> PyResult<f64> {
    let a_s = a.as_slice()?;
    let z_s = z.as_slice()?;
    let x_s = x_mass.as_slice()?;
    let v_s = v_gt.as_slice()?;
    if a_s.len() != z_s.len()
        || a_s.len() != x_s.len()
        || a_s.len() != v_s.len()
    {
        return Err(PyValueError::new_err(
            "a, z, x_mass, and v_gt must have identical lengths",
        ));
    }
    Ok(eval_wd_sedimentation_heating(
        a_s, z_s, x_s, v_s, abar, zbar, g_grav,
    ))
}

/// Runs multi-species element diffusion across all zones of `model` over `dt`
/// seconds and returns `(v_gt_2d, eps_wd_sed_1d)`.
#[pyfunction]
#[pyo3(signature = (model, dt, scheme="stanton_murillo"))]
pub fn star_do_element_diffusion<'py>(
    py: Python<'py>,
    model: &mut PyStarModel,
    dt: f64,
    scheme: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let sch = parse_diffusion_scheme(scheme)?;
    let nz = model.inner.nz;
    let nc = model.inner.species;
    let (v_gt, eps_wd) = do_element_diffusion(&mut model.inner, dt, sch);
    let v_arr = PyArray1::from_vec(py, v_gt).reshape([nz, nc])?;
    Ok((v_arr, PyArray1::from_vec(py, eps_wd)))
}

/// Evaluates the Fabry et al. (2022) / Paxton et al. (2019) Roche rotational
/// distortion polynomials $(f_P, f_T, i_{\mathrm{rot}}/r^2)$.
#[pyfunction]
#[pyo3(signature = (w_div_w_crit, fp_min=0.75, ft_min=0.95))]
pub fn star_eval_rotation_fp_ft(
    w_div_w_crit: f64,
    fp_min: f64,
    ft_min: f64,
) -> (f64, f64, f64) {
    eval_rotation_fp_ft(w_div_w_crit, fp_min, ft_min)
}

/// Evaluates the local critical Roche angular velocity $\Omega_{\mathrm{crit}}$
/// in $\mathrm{rad\,s^{-1}}$.
#[pyfunction]
#[pyo3(signature = (cgrav, m_grav, r_polar, lum, opacity, max_gamma_edd=0.99))]
pub fn star_eval_omega_crit(
    cgrav: f64,
    m_grav: f64,
    r_polar: f64,
    lum: f64,
    opacity: f64,
    max_gamma_edd: f64,
) -> f64 {
    eval_omega_crit(cgrav, m_grav, r_polar, lum, opacity, max_gamma_edd)
}

/// Evaluates all five rotational instabilities and the Spruit-Tayler dynamo at
/// a single spherical shell face, returning
/// `(d_dsi, d_sh, d_ssi, d_es, d_gsf, d_st, nu_st, b_phi, b_r)`.
#[pyfunction]
#[pyo3(signature = (
    r, hp, grav, rho, t, cp, opacity, chi_t, chi_rho, grada, gradt,
    brunt_n2_t, brunt_n2_mu, omega, dln_omega_dln_r, abar, zbar, d_visc=0.0
))]
#[allow(clippy::too_many_arguments)]
pub fn star_eval_rotational_mixing_zone(
    r: f64,
    hp: f64,
    grav: f64,
    rho: f64,
    t: f64,
    cp: f64,
    opacity: f64,
    chi_t: f64,
    chi_rho: f64,
    grada: f64,
    gradt: f64,
    brunt_n2_t: f64,
    brunt_n2_mu: f64,
    omega: f64,
    dln_omega_dln_r: f64,
    abar: f64,
    zbar: f64,
    d_visc: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let z = eval_rotational_mixing_zone(
        r,
        hp,
        grav,
        rho,
        t,
        cp,
        opacity,
        chi_t,
        chi_rho,
        grada,
        gradt,
        brunt_n2_t,
        brunt_n2_mu,
        omega,
        dln_omega_dln_r,
        abar,
        zbar,
        d_visc,
    );
    (
        z.d_dsi, z.d_sh, z.d_ssi, z.d_es, z.d_gsf, z.d_st, z.nu_st, z.b_phi,
        z.b_r,
    )
}

/// Updates rotational distortion $(f_P, f_T)$ and rotational mixing on `model`
/// and returns a `(nz, 9)` array of `(d_dsi, d_sh, d_ssi, d_es, d_gsf, d_st,
/// nu_st, b_phi, b_r)`.
#[pyfunction]
pub fn star_set_rotation_mixing_info<'py>(
    py: Python<'py>,
    model: &mut PyStarModel,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let zones = set_rotation_mixing_info(&mut model.inner);
    let nz = zones.len();
    let mut flat = Vec::with_capacity(nz * 9);
    for z in zones {
        flat.extend_from_slice(&[
            z.d_dsi, z.d_sh, z.d_ssi, z.d_es, z.d_gsf, z.d_st, z.nu_st,
            z.b_phi, z.b_r,
        ]);
    }
    PyArray1::from_vec(py, flat).reshape([nz, 9])
}

/// Evaluates the Blouin et al. (2020, 2021) solid-liquid distillation shift
/// $\Delta x_{\mathrm{heavy}}$.
#[pyfunction]
#[pyo3(signature = (x_heavy, is_one_mixture=false))]
pub fn star_eval_phase_separation_shift(
    x_heavy: f64,
    is_one_mixture: bool,
) -> f64 {
    eval_phase_separation_shift(x_heavy, is_one_mixture)
}

/// Evaluates the Castellani et al. (1971) convective boundary penetration
/// velocity $v_p$ in $\mathrm{cm\,s^{-1}}$.
#[pyfunction]
pub fn star_eval_castellani_penetration_vel(
    conv_vel: f64,
    r: f64,
    hp: f64,
    gradr: f64,
    grada: f64,
    abar_in: f64,
    abar_out: f64,
) -> f64 {
    eval_castellani_penetration_vel(
        conv_vel, r, hp, gradr, grada, abar_in, abar_out,
    )
}

/// Runs iterative convective premixing on `model` and returns the number of
/// cells absorbed across convective boundaries.
#[pyfunction]
#[pyo3(signature = (model, max_iters=10, avoid_fuel_increase=true))]
pub fn star_do_conv_premix(
    model: &mut PyStarModel,
    max_iters: usize,
    avoid_fuel_increase: bool,
) -> usize {
    do_conv_premix(&mut model.inner, max_iters, avoid_fuel_increase)
}

/// Applies predictive mixing at convective boundaries on `model`.
#[pyfunction]
#[pyo3(signature = (
    model, ingested_mass_limit_coeff=0.1, superad_thresh=0.005
))]
pub fn star_add_predictive_mixing(
    model: &mut PyStarModel,
    ingested_mass_limit_coeff: f64,
    superad_thresh: f64,
) -> usize {
    add_predictive_mixing(
        &mut model.inner,
        ingested_mass_limit_coeff,
        superad_thresh,
    )
}

/// Executes white-dwarf crystallization phase separation on `model` and returns
/// the gravothermal heating rate array $\epsilon_{\mathrm{phase\_sep},k}$.
#[pyfunction]
#[pyo3(signature = (
    model, dt, gamma_crystallization=175.0, is_one_mixture=false
))]
pub fn star_do_phase_separation<'py>(
    py: Python<'py>,
    model: &mut PyStarModel,
    dt: f64,
    gamma_crystallization: f64,
    is_one_mixture: bool,
) -> Bound<'py, PyArray1<f64>> {
    let eps = do_phase_separation(
        &mut model.inner,
        dt,
        gamma_crystallization,
        is_one_mixture,
    );
    PyArray1::from_vec(py, eps)
}

/// Registers Wave 5.1 `flexmesa-star` bindings on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_eval_burgers_velocities, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_caplan_ocp_dstar, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_radiative_levitation_accel, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_wd_sedimentation_heating, m)?)?;
    m.add_function(wrap_pyfunction!(star_do_element_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_rotation_fp_ft, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_omega_crit, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_rotational_mixing_zone, m)?)?;
    m.add_function(wrap_pyfunction!(star_set_rotation_mixing_info, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_phase_separation_shift, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_castellani_penetration_vel, m)?)?;
    m.add_function(wrap_pyfunction!(star_do_conv_premix, m)?)?;
    m.add_function(wrap_pyfunction!(star_add_predictive_mixing, m)?)?;
    m.add_function(wrap_pyfunction!(star_do_phase_separation, m)?)?;
    Ok(())
}
