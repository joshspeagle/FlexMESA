// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B `PyO3` bindings for `flexmesa-kap` (`kap` radiative, conductive, and
//! Compton opacities).

use flexmesa_kap::{
    aesopus_get, compton_opacity, do_electron_conduction, get1_kap_co_results,
    get1_kap_fixed_metal_results, get_kap_results, kap_get, kapcn_get,
    KapCoOption, KapConfig, KapLowTOption, KapOption, NUM_KAP_FRACS,
};
use numpy::{
    PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_kap_option(opt: &str) -> PyResult<KapOption> {
    KapOption::from_prefix(opt).ok_or_else(|| {
        PyValueError::new_err(format!("unknown kap_option: {opt}"))
    })
}

fn parse_kap_co_option(opt: &str) -> PyResult<KapCoOption> {
    KapCoOption::from_prefix(opt).ok_or_else(|| {
        PyValueError::new_err(format!("unknown kap_co_option: {opt}"))
    })
}

fn parse_kap_low_t_option(opt: &str) -> PyResult<KapLowTOption> {
    KapLowTOption::from_prefix(opt).ok_or_else(|| {
        PyValueError::new_err(format!("unknown kap_low_t_option: {opt}"))
    })
}

#[allow(clippy::fn_params_excessive_bools)]
fn build_kap_config(
    zbase: f64,
    kap_option: &str,
    kap_co_option: &str,
    kap_low_t_option: &str,
    kap_blend_log_t_upper_bdy: f64,
    kap_blend_log_t_lower_bdy: f64,
    cubic_interpolation_in_x: bool,
    cubic_interpolation_in_z: bool,
    include_electron_conduction: bool,
    use_blouin_conductive_opacities: bool,
    use_zbase_for_type1: bool,
    use_type2_opacities: bool,
    kap_type2_full_off_x: f64,
    kap_type2_full_on_x: f64,
    kap_type2_full_off_dz: f64,
    kap_type2_full_on_dz: f64,
) -> PyResult<KapConfig> {
    Ok(KapConfig {
        zbase,
        kap_option: parse_kap_option(kap_option)?,
        kap_co_option: parse_kap_co_option(kap_co_option)?,
        kap_low_t_option: parse_kap_low_t_option(kap_low_t_option)?,
        kap_blend_log_t_upper_bdy,
        kap_blend_log_t_lower_bdy,
        cubic_interpolation_in_x,
        cubic_interpolation_in_z,
        include_electron_conduction,
        use_blouin_conductive_opacities,
        use_zbase_for_type1,
        use_type2_opacities,
        kap_type2_full_off_x,
        kap_type2_full_on_x,
        kap_type2_full_off_dz,
        kap_type2_full_on_dz,
        ..KapConfig::default()
    })
}

/// Evaluate `get_kap_results` (`(kap, dlnkap_dlnRho, dlnkap_dlnT, kap_fracs[4])`).
#[pyfunction]
#[pyo3(signature = (
    z, x, zbase, xc, xn, xo, xne, log_rho, log_t,
    lnfree_e, d_lnfree_e_dln_rho, d_lnfree_e_dln_t,
    eta, d_eta_dln_rho, d_eta_dln_t, zbar=1.0,
    kap_option="gs98", kap_co_option="gs98_co", kap_low_t_option="lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy=3.88, kap_blend_log_t_lower_bdy=3.80,
    cubic_interpolation_in_x=false, cubic_interpolation_in_z=false,
    include_electron_conduction=true, use_blouin_conductive_opacities=true,
    use_zbase_for_type1=false, use_type2_opacities=false,
    kap_type2_full_off_x=1.0e-3, kap_type2_full_on_x=1.0e-6,
    kap_type2_full_off_dz=0.001, kap_type2_full_on_dz=0.01
))]
#[allow(clippy::fn_params_excessive_bools)]
pub fn kap_get_results_rust(
    z: f64,
    x: f64,
    zbase: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    xne: f64,
    log_rho: f64,
    log_t: f64,
    lnfree_e: f64,
    d_lnfree_e_dln_rho: f64,
    d_lnfree_e_dln_t: f64,
    eta: f64,
    d_eta_dln_rho: f64,
    d_eta_dln_t: f64,
    zbar: f64,
    kap_option: &str,
    kap_co_option: &str,
    kap_low_t_option: &str,
    kap_blend_log_t_upper_bdy: f64,
    kap_blend_log_t_lower_bdy: f64,
    cubic_interpolation_in_x: bool,
    cubic_interpolation_in_z: bool,
    include_electron_conduction: bool,
    use_blouin_conductive_opacities: bool,
    use_zbase_for_type1: bool,
    use_type2_opacities: bool,
    kap_type2_full_off_x: f64,
    kap_type2_full_on_x: f64,
    kap_type2_full_off_dz: f64,
    kap_type2_full_on_dz: f64,
) -> PyResult<(f64, f64, f64, [f64; NUM_KAP_FRACS])> {
    let cfg = build_kap_config(
        zbase,
        kap_option,
        kap_co_option,
        kap_low_t_option,
        kap_blend_log_t_upper_bdy,
        kap_blend_log_t_lower_bdy,
        cubic_interpolation_in_x,
        cubic_interpolation_in_z,
        include_electron_conduction,
        use_blouin_conductive_opacities,
        use_zbase_for_type1,
        use_type2_opacities,
        kap_type2_full_off_x,
        kap_type2_full_on_x,
        kap_type2_full_off_dz,
        kap_type2_full_on_dz,
    )?;
    let out = get_kap_results(
        &cfg,
        zbar,
        x,
        z,
        xc,
        xn,
        xo,
        xne,
        log_rho,
        log_t,
        lnfree_e,
        d_lnfree_e_dln_rho,
        d_lnfree_e_dln_t,
        eta,
        d_eta_dln_rho,
        d_eta_dln_t,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.kap, out.dlnkap_dln_rho, out.dlnkap_dln_t, out.kap_fracs))
}

/// Evaluate `kap_get` from 1-based isotope IDs `chem_id` and mass fractions `xa`.
#[pyfunction]
#[pyo3(signature = (
    chem_id, xa, log_rho, log_t,
    lnfree_e, d_lnfree_e_dln_rho, d_lnfree_e_dln_t,
    eta, d_eta_dln_rho, d_eta_dln_t, zbase=0.02,
    kap_option="gs98", kap_co_option="gs98_co", kap_low_t_option="lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy=3.88, kap_blend_log_t_lower_bdy=3.80,
    cubic_interpolation_in_x=false, cubic_interpolation_in_z=false,
    include_electron_conduction=true, use_blouin_conductive_opacities=true,
    use_zbase_for_type1=false, use_type2_opacities=false,
    kap_type2_full_off_x=1.0e-3, kap_type2_full_on_x=1.0e-6,
    kap_type2_full_off_dz=0.001, kap_type2_full_on_dz=0.01
))]
#[allow(clippy::fn_params_excessive_bools)]
pub fn kap_get_rust(
    chem_id: PyReadonlyArray1<'_, usize>,
    xa: PyReadonlyArray1<'_, f64>,
    log_rho: f64,
    log_t: f64,
    lnfree_e: f64,
    d_lnfree_e_dln_rho: f64,
    d_lnfree_e_dln_t: f64,
    eta: f64,
    d_eta_dln_rho: f64,
    d_eta_dln_t: f64,
    zbase: f64,
    kap_option: &str,
    kap_co_option: &str,
    kap_low_t_option: &str,
    kap_blend_log_t_upper_bdy: f64,
    kap_blend_log_t_lower_bdy: f64,
    cubic_interpolation_in_x: bool,
    cubic_interpolation_in_z: bool,
    include_electron_conduction: bool,
    use_blouin_conductive_opacities: bool,
    use_zbase_for_type1: bool,
    use_type2_opacities: bool,
    kap_type2_full_off_x: f64,
    kap_type2_full_on_x: f64,
    kap_type2_full_off_dz: f64,
    kap_type2_full_on_dz: f64,
) -> PyResult<(f64, f64, f64, [f64; NUM_KAP_FRACS])> {
    let cfg = build_kap_config(
        zbase,
        kap_option,
        kap_co_option,
        kap_low_t_option,
        kap_blend_log_t_upper_bdy,
        kap_blend_log_t_lower_bdy,
        cubic_interpolation_in_x,
        cubic_interpolation_in_z,
        include_electron_conduction,
        use_blouin_conductive_opacities,
        use_zbase_for_type1,
        use_type2_opacities,
        kap_type2_full_off_x,
        kap_type2_full_on_x,
        kap_type2_full_off_dz,
        kap_type2_full_on_dz,
    )?;
    let cid = chem_id.as_slice()?;
    let x_slice = xa.as_slice()?;
    let out = kap_get(
        &cfg,
        cid,
        x_slice,
        log_rho,
        log_t,
        lnfree_e,
        d_lnfree_e_dln_rho,
        d_lnfree_e_dln_t,
        eta,
        d_eta_dln_rho,
        d_eta_dln_t,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.kap, out.dlnkap_dln_rho, out.dlnkap_dln_t, out.kap_fracs))
}

/// Evaluate electron thermal conduction opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
#[pyfunction]
#[pyo3(signature = (zbar, log_rho, log_t, use_blouin=true))]
pub fn kap_get_elect_cond_opacity_rust(
    zbar: f64,
    log_rho: f64,
    log_t: f64,
    use_blouin: bool,
) -> PyResult<(f64, f64, f64)> {
    let cfg = KapConfig {
        use_blouin_conductive_opacities: use_blouin,
        ..KapConfig::default()
    };
    do_electron_conduction(&cfg, zbar, log_rho, log_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate Poutanen (2017) Compton opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
#[pyfunction]
pub fn kap_get_compton_opacity_rust(
    rho: f64,
    t: f64,
    lnfree_e: f64,
    d_lnfree_e_dln_rho: f64,
    d_lnfree_e_dln_t: f64,
    eta: f64,
    d_eta_dln_rho: f64,
    d_eta_dln_t: f64,
) -> (f64, f64, f64) {
    compton_opacity(
        rho,
        t,
        lnfree_e,
        d_lnfree_e_dln_rho,
        d_lnfree_e_dln_t,
        eta,
        d_eta_dln_rho,
        d_eta_dln_t,
    )
}

/// Evaluate Type 1 fixed-metal opacity `(log10_kap, dlnkap_dlnRho, dlnkap_dlnT)`.
#[pyfunction]
#[pyo3(signature = (z, x, log_rho, log_t, prefix="gs98", cubic_x=false, cubic_z=false))]
pub fn kap_get_fixed_metal_rust(
    z: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
    prefix: &str,
    cubic_x: bool,
    cubic_z: bool,
) -> PyResult<(f64, f64, f64)> {
    let cfg = KapConfig {
        cubic_interpolation_in_x: cubic_x,
        cubic_interpolation_in_z: cubic_z,
        ..KapConfig::default()
    };
    get1_kap_fixed_metal_results(&cfg, prefix, z, x, log_rho, log_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate Type 2 C/O-enhanced opacity `(log10_kap, dlnkap_dlnRho, dlnkap_dlnT)`.
#[pyfunction]
#[pyo3(signature = (zbase, x, dxc, dxo, log_rho, log_t, kap_co_option="gn93_co", cubic_x=false, cubic_z=false))]
pub fn kap_get_co_rust(
    zbase: f64,
    x: f64,
    dxc: f64,
    dxo: f64,
    log_rho: f64,
    log_t: f64,
    kap_co_option: &str,
    cubic_x: bool,
    cubic_z: bool,
) -> PyResult<(f64, f64, f64)> {
    let cfg = KapConfig {
        kap_co_option: parse_kap_co_option(kap_co_option)?,
        cubic_interpolation_in_x: cubic_x,
        cubic_interpolation_in_z: cubic_z,
        ..KapConfig::default()
    };
    get1_kap_co_results(&cfg, zbase, x, dxc, dxo, log_rho, log_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate `AESOPUS_AGSS09` low-T opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
#[pyfunction]
pub fn kap_get_aesopus_rust(
    zbase: f64,
    x: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    log_rho: f64,
    log_t: f64,
) -> PyResult<(f64, f64, f64)> {
    aesopus_get(zbase, x, xc, xn, xo, log_rho, log_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate `kapCN` Lederer & Aringer (2009) low-T opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
#[pyfunction]
pub fn kap_get_kapcn_rust(
    zbase: f64,
    x: f64,
    fc: f64,
    fn_val: f64,
    log_rho: f64,
    log_t: f64,
) -> PyResult<(f64, f64, f64)> {
    kapcn_get(zbase, x, fc, fn_val, log_rho, log_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate `kap_get` across `N_zones` zones in parallel, returning `(N_zones, 7)` array
/// `[kap, dlnkap_dlnRho, dlnkap_dlnT, frac_lowT, frac_highT, frac_Type2, frac_Compton]`.
#[pyfunction]
#[pyo3(signature = (
    chem_id, xa_zones, log_rho_zones, log_t_zones,
    lnfree_e_zones, d_lnfree_e_dln_rho_zones, d_lnfree_e_dln_t_zones,
    eta_zones, d_eta_dln_rho_zones, d_eta_dln_t_zones,
    zbase=0.02, kap_option="gs98", kap_co_option="gs98_co", kap_low_t_option="lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy=3.88, kap_blend_log_t_lower_bdy=3.80,
    cubic_interpolation_in_x=false, cubic_interpolation_in_z=false,
    include_electron_conduction=true, use_blouin_conductive_opacities=true,
    use_zbase_for_type1=false, use_type2_opacities=false,
    kap_type2_full_off_x=1.0e-3, kap_type2_full_on_x=1.0e-6,
    kap_type2_full_off_dz=0.001, kap_type2_full_on_dz=0.01,
    parallel=true
))]
#[allow(clippy::fn_params_excessive_bools)]
pub fn kap_get_zones_rust<'py>(
    py: Python<'py>,
    chem_id: PyReadonlyArray1<'py, usize>,
    xa_zones: PyReadonlyArray2<'py, f64>,
    log_rho_zones: PyReadonlyArray1<'py, f64>,
    log_t_zones: PyReadonlyArray1<'py, f64>,
    lnfree_e_zones: PyReadonlyArray1<'py, f64>,
    d_lnfree_e_dln_rho_zones: PyReadonlyArray1<'py, f64>,
    d_lnfree_e_dln_t_zones: PyReadonlyArray1<'py, f64>,
    eta_zones: PyReadonlyArray1<'py, f64>,
    d_eta_dln_rho_zones: PyReadonlyArray1<'py, f64>,
    d_eta_dln_t_zones: PyReadonlyArray1<'py, f64>,
    zbase: f64,
    kap_option: &str,
    kap_co_option: &str,
    kap_low_t_option: &str,
    kap_blend_log_t_upper_bdy: f64,
    kap_blend_log_t_lower_bdy: f64,
    cubic_interpolation_in_x: bool,
    cubic_interpolation_in_z: bool,
    include_electron_conduction: bool,
    use_blouin_conductive_opacities: bool,
    use_zbase_for_type1: bool,
    use_type2_opacities: bool,
    kap_type2_full_off_x: f64,
    kap_type2_full_on_x: f64,
    kap_type2_full_off_dz: f64,
    kap_type2_full_on_dz: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = build_kap_config(
        zbase,
        kap_option,
        kap_co_option,
        kap_low_t_option,
        kap_blend_log_t_upper_bdy,
        kap_blend_log_t_lower_bdy,
        cubic_interpolation_in_x,
        cubic_interpolation_in_z,
        include_electron_conduction,
        use_blouin_conductive_opacities,
        use_zbase_for_type1,
        use_type2_opacities,
        kap_type2_full_off_x,
        kap_type2_full_on_x,
        kap_type2_full_off_dz,
        kap_type2_full_on_dz,
    )?;
    let cid = chem_id.as_slice()?.to_vec();
    let n_spec = cid.len();
    let xa_shape = xa_zones.shape();
    let lr = log_rho_zones.as_slice()?.to_vec();
    let lt = log_t_zones.as_slice()?.to_vec();
    let lnf = lnfree_e_zones.as_slice()?.to_vec();
    let dlnf_dr = d_lnfree_e_dln_rho_zones.as_slice()?.to_vec();
    let dlnf_dt = d_lnfree_e_dln_t_zones.as_slice()?.to_vec();
    let et = eta_zones.as_slice()?.to_vec();
    let det_dr = d_eta_dln_rho_zones.as_slice()?.to_vec();
    let det_dt = d_eta_dln_t_zones.as_slice()?.to_vec();
    let nz = lr.len();
    if xa_shape != [nz, n_spec]
        || lt.len() != nz
        || lnf.len() != nz
        || dlnf_dr.len() != nz
        || dlnf_dt.len() != nz
        || et.len() != nz
        || det_dr.len() != nz
        || det_dt.len() != nz
    {
        return Err(PyValueError::new_err(
            "kap_get_zones_rust: mismatched zone or species dimensions",
        ));
    }
    let xa_flat = xa_zones.as_slice()?.to_vec();

    let out_arr =
        crate::eval_batch_2d::<7, _>(py, nz, parallel, 8, move |k| {
            let xa_k = &xa_flat[k * n_spec..(k + 1) * n_spec];
            match kap_get(
                &cfg, &cid, xa_k, lr[k], lt[k], lnf[k], dlnf_dr[k], dlnf_dt[k],
                et[k], det_dr[k], det_dt[k],
            ) {
                Ok(out) => [
                    out.kap,
                    out.dlnkap_dln_rho,
                    out.dlnkap_dln_t,
                    out.kap_fracs[0],
                    out.kap_fracs[1],
                    out.kap_fracs[2],
                    out.kap_fracs[3],
                ],
                Err(_) => [f64::NAN; 7],
            }
        })?;
    if out_arr.readonly().as_slice()?.iter().any(|v| v.is_nan()) {
        return Err(PyValueError::new_err(
            "kap_get failed in kap_get_zones_rust",
        ));
    }
    Ok(out_arr)
}

/// Register `kap` bindings on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kap_get_results_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_elect_cond_opacity_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_compton_opacity_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_fixed_metal_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_co_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_aesopus_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_kapcn_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kap_get_zones_rust, m)?)?;
    Ok(())
}
