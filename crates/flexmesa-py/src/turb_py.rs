// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B (`turb`) `PyO3` bindings for Mixing-Length Theory (`MLT`) and
//! Time-Dependent Convection (`TDC`).
//!
//! Exposes scalar `f64` (`set_mlt_f64`) and parallel multi-zone `DualOrder1<6>`
//! batch (`set_mlt_batch`, `set_tdc_batch`) evaluators with Implicit Function
//! Theorem root sensitivities and persistent 4-thread `rayon` execution.
//!
//! # Physical Conventions
//! - Temperature gradients ($\nabla_{\mathrm{rad}}, \nabla_{\mathrm{ad}},
//!   \nabla_L, \nabla_T$) and superadiabaticity $Y_{\mathrm{face}}$ are
//!   dimensionless logarithmic derivatives $d\ln T / d\ln P$.
//! - Convective speeds $v_{\mathrm{conv}}$ are in $\mathrm{cm\,s^{-1}}$ and
//!   diffusion coefficients $D$ are in $\mathrm{cm^2\,s^{-1}}$.

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{KERG, LSUN, MP, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltOption, MltParams, TdcInputs, TdcParams,
};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

/// Solar surface gravitational acceleration $G M_{\odot} / R_{\odot}^2$ in
/// $\mathrm{cm\,s^{-2}}$.
const GRAV_SOLAR: f64 = MSUN * STANDARD_CGRAV / (RSUN * RSUN);

/// Ideal monatomic specific gas constant $k_{\mathrm{B}} / m_p$ in
/// $\mathrm{erg\,g^{-1}\,K^{-1}}$.
const RGAS_IDEAL: f64 = KERG / MP;

/// Ideal monatomic specific heat at constant pressure $C_P = \frac{5}{2}
/// k_{\mathrm{B}} / m_p$ in $\mathrm{erg\,g^{-1}\,K^{-1}}$.
const CP_IDEAL: f64 = 2.5 * RGAS_IDEAL;

/// Evaluates `set_mlt` in `f64` at a single zone, returning
/// `([gamma, grad_t, y_face, conv_vel, d], mixing_type_code)`.
///
/// Matches upstream MESA `turb_lib::set_MLT` across all 6 geometric MLT
/// prescriptions (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`, `Mihalas`).
///
/// # Arguments
/// - `option`: geometric MLT prescription name (`"Cox"`, `"TDC"`, `"Henyey"`,
///   `"ML1"`, `"ML2"`, `"Mihalas"`).
/// - `mixing_length_alpha`: dimensionless mixing-length parameter $\alpha$.
/// - `henyey_y_param`, `henyey_nu_param`: Henyey (1965) $(y, \nu)$ parameters.
/// - `chi_t`, `chi_rho`, `cp`, `grav`, `lambda_mlt`, `rho`, `p`, `t`,
///   `opacity`, `gradr`, `grada`, `gradl`: local CGS structure and EOS state.
#[pyfunction]
fn set_mlt_f64(
    option: &str,
    mixing_length_alpha: f64,
    henyey_y_param: f64,
    henyey_nu_param: f64,
    chi_t: f64,
    chi_rho: f64,
    cp: f64,
    grav: f64,
    lambda_mlt: f64,
    rho: f64,
    p: f64,
    t: f64,
    opacity: f64,
    gradr: f64,
    grada: f64,
    gradl: f64,
) -> PyResult<([f64; 5], i32)> {
    let opt: MltOption =
        option.parse().map_err(|e: flexmesa_turb::TurbError| {
            PyValueError::new_err(e.to_string())
        })?;
    let params = MltParams {
        option: opt,
        mixing_length_alpha,
        henyey_y: henyey_y_param,
        henyey_nu: henyey_nu_param,
        max_conv_vel: 1.0e99,
    };
    let inp = MltInputs {
        chi_t,
        chi_rho,
        cp,
        grav,
        lambda: lambda_mlt,
        rho,
        p,
        t,
        opacity,
        gradr,
        grada,
        gradl,
    };
    let out = set_mlt(&params, &inp)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        [out.gamma, out.grad_t, out.y_face, out.conv_vel, out.d],
        out.mixing_type.code(),
    ))
}

/// Evaluates Cox (1968) MLT with `DualOrder1<6>` sensitivities at `s_frac`.
///
/// Solves the depressed cubic $a_0 \Gamma^3 + \Gamma^2 + \Gamma - a_0 B^3 = 0$
/// and returns `[grad_t, d, conv_vel, d_grad_t_d_gradr]`.
#[inline]
fn eval_mlt_zone(s_frac: f64) -> [f64; 4] {
    let t_val = 1.0e5 * (1.0 + 0.5 * s_frac);
    let rho_val = 1.0e-5 * (1.0 + 0.5 * s_frac);
    let p_val = rho_val * t_val * RGAS_IDEAL;
    let hp_val = p_val / (rho_val * GRAV_SOLAR);
    let gradr_val = 0.45 + 0.25 * s_frac;
    let params = MltParams::cox(2.0);
    let inp: MltInputs<DualOrder1<6>> = MltInputs {
        chi_t: DualOrder1::constant(1.0),
        chi_rho: DualOrder1::constant(1.0),
        cp: DualOrder1::constant(CP_IDEAL),
        grav: DualOrder1::constant(GRAV_SOLAR),
        lambda: DualOrder1::independent(2.0 * hp_val, 3),
        rho: DualOrder1::independent(rho_val, 5),
        p: DualOrder1::constant(p_val),
        t: DualOrder1::independent(t_val, 4),
        opacity: DualOrder1::constant(1.0),
        gradr: DualOrder1::independent(gradr_val, 0),
        grada: DualOrder1::independent(0.40, 1),
        gradl: DualOrder1::independent(0.40, 2),
    };
    set_mlt(&params, &inp).map_or([f64::NAN; 4], |out| {
        [
            out.grad_t.val,
            out.d.val,
            out.conv_vel.val,
            out.grad_t.d1[0],
        ]
    })
}

/// Evaluates Kuhfuss (1986) TDC with `DualOrder1<6>` sensitivities at `s_frac`.
///
/// Solves the bracketed Newton-bisection root over $Z = \ln|Y|$ and propagates
/// analytical derivatives via the Implicit Function Theorem, returning
/// `[grad_t, d, conv_vel, d_grad_t_d_gradr]`.
#[inline]
fn eval_tdc_zone(s_frac: f64) -> [f64; 4] {
    let t_val = 1.0e5 * (1.0 + 0.2 * s_frac);
    let rho_val = 1.0e-5 * (1.0 + 0.2 * s_frac);
    let p_val = rho_val * t_val * RGAS_IDEAL;
    let hp_val = p_val / (rho_val * GRAV_SOLAR);
    let gradr_val = 0.42 + 0.08 * s_frac;
    let params = TdcParams {
        conv_vel_start: 1.0e5,
        mixing_length_alpha: 2.0,
        tdc_alpha_d: 1.0,
        tdc_alpha_r: 0.0,
        tdc_alpha_pt: 0.0,
        tdc_alpha_c: 1.0,
        tdc_alpha_s: 1.0,
        dt: 1.0e4,
        cgrav: STANDARD_CGRAV,
        m: MSUN,
        scale: 70.0 * LSUN * 1.0e-3,
        max_conv_vel: 1.0e99,
        include_mlt_corr_to_tdc: true,
        use_tdc_enthalpy_flux_limiter: false,
        y_face_guess: 0.0,
    };
    let inp: TdcInputs<DualOrder1<6>> = TdcInputs {
        chi_t: DualOrder1::constant(1.0),
        chi_rho: DualOrder1::constant(1.0),
        gradr: DualOrder1::independent(gradr_val, 0),
        r: DualOrder1::constant(RSUN),
        p: DualOrder1::constant(p_val),
        t: DualOrder1::independent(t_val, 4),
        rho: DualOrder1::independent(rho_val, 5),
        dv: DualOrder1::constant(0.0),
        cp: DualOrder1::constant(CP_IDEAL),
        opacity: DualOrder1::constant(1.0),
        scale_height: DualOrder1::independent(hp_val, 3),
        gradl: DualOrder1::independent(0.40, 2),
        grada: DualOrder1::independent(0.40, 1),
        eq_div_w: DualOrder1::constant(0.0),
        grav: DualOrder1::constant(GRAV_SOLAR),
        energy: DualOrder1::constant(0.0),
    };
    set_tdc(&params, &inp).map_or([f64::NAN; 4], |res| {
        [
            res.output.grad_t.val,
            res.output.d.val,
            res.output.conv_vel.val,
            res.output.grad_t.d1[0],
        ]
    })
}

/// Evaluates Cox (1968) MLT with `DualOrder1<6>` sensitivities across `s_frac`.
///
/// Dispatches across the persistent 4-thread `rayon` worker pool when
/// `parallel=True` and `nz >= 64`.
#[pyfunction]
#[pyo3(signature = (s_frac, parallel=false))]
fn set_mlt_batch<'py>(
    py: Python<'py>,
    s_frac: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sf = s_frac
        .as_slice()
        .map_err(|_| PyValueError::new_err("s_frac must be C-contiguous"))?
        .to_vec();
    eval_batch_2d(py, sf.len(), parallel, 64, move |k| eval_mlt_zone(sf[k]))
}

/// Evaluates Kuhfuss (1986) TDC with `DualOrder1<6>` sensitivities across
/// `s_frac`.
///
/// Dispatches across the persistent 4-thread `rayon` worker pool when
/// `parallel=True` and `nz >= 32`.
#[pyfunction]
#[pyo3(signature = (s_frac, parallel=false))]
fn set_tdc_batch<'py>(
    py: Python<'py>,
    s_frac: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sf = s_frac
        .as_slice()
        .map_err(|_| PyValueError::new_err("s_frac must be C-contiguous"))?
        .to_vec();
    eval_batch_2d(py, sf.len(), parallel, 32, move |k| eval_tdc_zone(sf[k]))
}

/// Registers Tier B `turb` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_mlt_f64, m)?)?;
    m.add_function(wrap_pyfunction!(set_mlt_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_tdc_batch, m)?)?;
    Ok(())
}
