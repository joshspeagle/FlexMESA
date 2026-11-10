// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for Wave 5.2 (`riemann`, `rsp2`, and `pulse` in
//! `flexmesa-star`).

use flexmesa_star::{
    build_pulse_points, eval_center_parabolic, eval_hllc_face,
    eval_hllc_surface_face, eval_riemann_momentum_residual,
    eval_riemann_profile, eval_rsp2_face, eval_rsp2_zone, export_cafein,
    export_fgong, export_gr1d, export_gsm, export_gyre, export_osc,
    export_pulse_by_format, export_saio, export_stella_pair, parse_stella_abn,
    parse_stella_hyd, run_stella_from_hyd_abn, solve_adipls_fgong_modes,
    solve_gyre_mesa_modes, solve_stationary_rsp2_w, FeII5169EtaTable,
    Rsp2Config, StellaExportOptions,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::star_py::PyStarModel;

/// Evaluates the well-balanced Lagrangian HLLC Riemann state at an interior
/// cell interface (`hydro_riemann.f90`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn star_eval_hllc_face(
    p_l: f64,
    rho_l: f64,
    cs_l: f64,
    u_l: f64,
    dm_l: f64,
    p_r: f64,
    rho_r: f64,
    cs_r: f64,
    u_r: f64,
    dm_r: f64,
    r_face: f64,
    m_grav_face: f64,
    cgrav_face: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    eval_hllc_face(
        p_l,
        rho_l,
        cs_l,
        u_l,
        dm_l,
        p_r,
        rho_r,
        cs_r,
        u_r,
        dm_r,
        r_face,
        m_grav_face,
        cgrav_face,
    )
    .to_tuple()
}

/// Evaluates the surface ($k = 0$) HLLC Riemann state (`hydro_riemann.f90`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn star_eval_hllc_surface_face(
    p_surf_cell: f64,
    rho_surf: f64,
    cs_surf: f64,
    u_surf: f64,
    dm_surf: f64,
    r_surf: f64,
    m_surf: f64,
    cgrav_surf: f64,
    p_bc: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    eval_hllc_surface_face(
        p_surf_cell,
        rho_surf,
        cs_surf,
        u_surf,
        dm_surf,
        r_surf,
        m_surf,
        cgrav_surf,
        p_bc,
    )
    .to_tuple()
}

/// Evaluates the spherical Lagrangian HLLC momentum residual for cell $k$
/// (`hydro_momentum.f90`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn star_eval_riemann_momentum_residual(
    u_k: f64,
    u_old_k: f64,
    dt: f64,
    dm_k: f64,
    r_outer: f64,
    r_inner: f64,
    p_face_outer: f64,
    p_face_inner: f64,
    p_cell_k: f64,
    g_cell_k: f64,
) -> f64 {
    eval_riemann_momentum_residual(
        u_k,
        u_old_k,
        dt,
        dm_k,
        r_outer,
        r_inner,
        p_face_outer,
        p_face_inner,
        p_cell_k,
        g_cell_k,
    )
}

/// Evaluates the full-profile HLLC Riemann face states `(nz, 7)` and cell
/// momentum residuals `(nz,)` across all zones of `model`.
#[pyfunction]
#[pyo3(signature = (model, u_cell, u_old_cell, p_surf_bc = 0.0, parallel = true))]
pub fn star_eval_riemann_profile<'py>(
    py: Python<'py>,
    model: &PyStarModel,
    u_cell: PyReadonlyArray1<'py, f64>,
    u_old_cell: PyReadonlyArray1<'py, f64>,
    p_surf_bc: f64,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let u_s = u_cell.as_slice()?;
    let u_old_s = u_old_cell.as_slice()?;
    let (faces, residuals) = py.detach(|| {
        eval_riemann_profile(&model.inner, u_s, u_old_s, p_surf_bc, parallel)
    });
    let rows: Vec<Vec<f64>> = faces
        .into_iter()
        .map(|f| {
            let (a, b, c, d, e, g, h) = f.to_tuple();
            vec![a, b, c, d, e, g, h]
        })
        .collect();
    let arr_faces = PyArray2::from_vec2(py, &rows)?;
    Ok((arr_faces, residuals.into_pyarray(py)))
}

/// Evaluates a single cell's `RSP2` turbulent pressure, eddy viscosity,
/// viscous heating, buoyant source, dissipation sink, radiative cooling sink,
/// and net turbulent energy rate (`hydro_rsp2.f90`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    w, t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t,
    r_outer, r_inner, u_outer, u_inner, dm, dlnd_dt = 0.0,
    alfa = 1.5, alfap = 1.0, alfas = 1.0, alfac = 1.0, alfad = 1.0,
    alfam = 0.25, alfat = 0.01, gammar = 1.0, min_w2_for_turbulence = 1.0e-4
))]
pub fn star_eval_rsp2_zone(
    w: f64,
    t: f64,
    rho: f64,
    p: f64,
    cp: f64,
    grad_ad: f64,
    delta: f64,
    kappa: f64,
    hp_cell: f64,
    grad_t: f64,
    r_outer: f64,
    r_inner: f64,
    u_outer: f64,
    u_inner: f64,
    dm: f64,
    dlnd_dt: f64,
    alfa: f64,
    alfap: f64,
    alfas: f64,
    alfac: f64,
    alfad: f64,
    alfam: f64,
    alfat: f64,
    gammar: f64,
    min_w2_for_turbulence: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let cfg = Rsp2Config {
        alfa,
        alfap,
        alfas,
        alfac,
        alfad,
        alfam,
        alfat,
        gammar,
        min_w2_for_turbulence,
    };
    let z = eval_rsp2_zone(
        w, t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t, r_outer,
        r_inner, u_outer, u_inner, dm, dlnd_dt, &cfg,
    );
    (
        z.p_turb, z.chi_turb, z.eq_visc, z.source_s, z.sink_d, z.sink_dr,
        z.de_t_dt,
    )
}

/// Evaluates face-centered `RSP2` scale height, superadiabaticity, viscous
/// stress, and radiative/convective/turbulent luminosities (`hydro_rsp2.f90`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    r_face, m_grav_face, cgrav_face, p_face, rho_face, t_face, cp_face,
    grad_ad_face, kappa_face, w_inner, w_outer, t4_inner, t4_outer,
    lnp_inner, lnp_outer, chi_face, du_over_r_face, dm_bar,
    alfa = 1.5, alfac = 1.0, alfat = 0.01
))]
pub fn star_eval_rsp2_face(
    r_face: f64,
    m_grav_face: f64,
    cgrav_face: f64,
    p_face: f64,
    rho_face: f64,
    t_face: f64,
    cp_face: f64,
    grad_ad_face: f64,
    kappa_face: f64,
    w_inner: f64,
    w_outer: f64,
    t4_inner: f64,
    t4_outer: f64,
    lnp_inner: f64,
    lnp_outer: f64,
    chi_face: f64,
    du_over_r_face: f64,
    dm_bar: f64,
    alfa: f64,
    alfac: f64,
    alfat: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let cfg = Rsp2Config {
        alfa,
        alfac,
        alfat,
        ..Rsp2Config::default()
    };
    let f = eval_rsp2_face(
        r_face,
        m_grav_face,
        cgrav_face,
        p_face,
        rho_face,
        t_face,
        cp_face,
        grad_ad_face,
        kappa_face,
        w_inner,
        w_outer,
        t4_inner,
        t4_outer,
        lnp_inner,
        lnp_outer,
        chi_face,
        du_over_r_face,
        dm_bar,
        &cfg,
    );
    (f.hp_face, f.y_face, f.pi_face, f.l_r, f.l_c, f.l_t)
}

/// Solves the stationary local `RSP2` turbulent balance for $w$
/// (`rsp2_solve.f90`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t,
    alfa = 1.5, alfas = 1.0, alfad = 1.0, gammar = 1.0
))]
pub fn star_solve_stationary_rsp2_w(
    t: f64,
    rho: f64,
    p: f64,
    cp: f64,
    grad_ad: f64,
    delta: f64,
    kappa: f64,
    hp_cell: f64,
    grad_t: f64,
    alfa: f64,
    alfas: f64,
    alfad: f64,
    gammar: f64,
) -> f64 {
    let cfg = Rsp2Config {
        alfa,
        alfas,
        alfad,
        gammar,
        ..Rsp2Config::default()
    };
    solve_stationary_rsp2_w(
        t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t, &cfg,
    )
}

/// Parabolically extrapolates $v(r)$ to $r = 0$ (`pulse_utils.f90`).
#[pyfunction]
pub fn star_eval_center_parabolic(
    r_inner: f64,
    r_outer: f64,
    v_inner: f64,
    v_outer: f64,
) -> f64 {
    eval_center_parabolic(r_inner, r_outer, v_inner, v_outer)
}

/// Builds the center-to-surface pulsation structure array `(n_pts, 26)`
/// containing physical variables and Dziembowski variables $(V, V_g, A_*, U,
/// c_1)$.
#[pyfunction]
#[pyo3(signature = (
    model,
    add_center_point = true,
    add_double_points = false,
    grad_mu_threshold = 1.0e99
))]
pub fn star_build_pulse_points<'py>(
    py: Python<'py>,
    model: &PyStarModel,
    add_center_point: bool,
    add_double_points: bool,
    grad_mu_threshold: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pts = py.detach(|| {
        build_pulse_points(
            &model.inner,
            add_center_point,
            add_double_points,
            grad_mu_threshold,
        )
    });
    let rows: Vec<Vec<f64>> = pts
        .into_iter()
        .map(|p| {
            vec![
                p.r, p.m, p.lum, p.p, p.t, p.rho, p.grad_t, p.n2, p.gamma1,
                p.grad_ad, p.delta, p.kap, p.kap_t, p.kap_rho, p.eps, p.eps_t,
                p.eps_rho, p.omega, p.x_h1, p.x_he4, p.z_metal, p.v_dziem,
                p.v_g, p.a_star, p.u_dziem, p.c_1,
            ]
        })
        .collect();
    Ok(PyArray2::from_vec2(py, &rows)?)
}

/// Exports `model` in `GYRE` v120 ASCII format (`pulse_gyre.f90`).
#[pyfunction]
#[pyo3(signature = (model, add_center_point = true))]
pub fn star_export_gyre(
    py: Python<'_>,
    model: &PyStarModel,
    add_center_point: bool,
) -> String {
    py.detach(|| export_gyre(&model.inner, add_center_point))
}

/// Exports `model` in `FGONG` v1300 ASCII format (`pulse_fgong.f90`).
#[pyfunction]
#[pyo3(signature = (model, add_center_point = true))]
pub fn star_export_fgong(
    py: Python<'_>,
    model: &PyStarModel,
    add_center_point: bool,
) -> String {
    py.detach(|| export_fgong(&model.inner, add_center_point))
}

/// Exports `model` in `OSC` ASCII format (`pulse_osc.f90`).
#[pyfunction]
#[pyo3(signature = (model, add_center_point = true))]
pub fn star_export_osc(
    py: Python<'_>,
    model: &PyStarModel,
    add_center_point: bool,
) -> String {
    py.detach(|| export_osc(&model.inner, add_center_point))
}

/// Exports `model` in `GYRE` `GSM` v110 binary/HDF-compatible format (`pulse_gsm.f90`).
#[pyfunction]
#[pyo3(signature = (model, add_center_point = true))]
pub fn star_export_gsm(
    py: Python<'_>,
    model: &PyStarModel,
    add_center_point: bool,
) -> Vec<u8> {
    py.detach(|| export_gsm(&model.inner, add_center_point).into_bytes())
}

/// Exports `model` in `SAIO` ASCII format (`pulse_saio.f90`).
#[pyfunction]
#[pyo3(signature = (model, add_center_point = true))]
pub fn star_export_saio(
    py: Python<'_>,
    model: &PyStarModel,
    add_center_point: bool,
) -> String {
    py.detach(|| export_saio(&model.inner, add_center_point, true))
}

/// Exports `model` in `CAFEIN` ASCII format (`pulse_cafein.f90`).
#[pyfunction]
#[pyo3(signature = (model, add_center_point = true))]
pub fn star_export_cafein(
    py: Python<'_>,
    model: &PyStarModel,
    add_center_point: bool,
) -> String {
    py.detach(|| export_cafein(&model.inner, add_center_point, true, false))
}

/// Exports `model` in `GR1D` ASCII format (`pulse_gr1d.f90`).
#[pyfunction]
#[pyo3(signature = (model, _add_center_point = true))]
pub fn star_export_gr1d(
    py: Python<'_>,
    model: &PyStarModel,
    _add_center_point: bool,
) -> String {
    py.detach(|| export_gr1d(&model.inner))
}

/// Exports `model` in any pulsation format (`gyre`, `fgong`, `osc`, `gsm`, `saio`,
/// `cafein`, `gr1d`).
#[pyfunction]
#[pyo3(signature = (model, format, add_center_point = true))]
pub fn star_export_pulse_by_format(
    py: Python<'_>,
    model: &PyStarModel,
    format: &str,
    add_center_point: bool,
) -> String {
    py.detach(|| {
        export_pulse_by_format(&model.inner, format, add_center_point)
            .unwrap_or_default()
    })
}

/// Solves the 4th-order BVP on a `GYRE` MESA-format model string.
#[pyfunction]
#[pyo3(signature = (mesa_str, l, omega_min, omega_max, n_scan = 200, _nonadiabatic = true))]
#[allow(clippy::type_complexity)]
pub fn star_solve_gyre_mesa_modes(
    py: Python<'_>,
    mesa_str: &str,
    l: usize,
    omega_min: f64,
    omega_max: f64,
    n_scan: usize,
    _nonadiabatic: bool,
) -> Vec<(usize, i32, i32, i32, f64, f64, f64, f64, f64)> {
    py.detach(|| {
        solve_gyre_mesa_modes(mesa_str, l, omega_min, omega_max, n_scan)
            .into_iter()
            .map(|m| {
                (
                    m.l,
                    m.n_p,
                    m.n_g,
                    m.n_pg,
                    m.omega,
                    m.omega_int,
                    m.freq_uhz,
                    m.e_norm,
                    m.omega_imag,
                )
            })
            .collect()
    })
}

/// Solves the 4th-order `ADIPLS` BVP on an `FGONG` model string.
#[pyfunction]
#[pyo3(signature = (fgong_str, l, freq_min_uhz, freq_max_uhz, n_scan = 500))]
#[allow(clippy::type_complexity)]
pub fn star_solve_adipls_fgong_modes(
    py: Python<'_>,
    fgong_str: &str,
    l: usize,
    freq_min_uhz: f64,
    freq_max_uhz: f64,
    n_scan: usize,
) -> Vec<(usize, i32, i32, i32, f64, f64, f64, f64)> {
    py.detach(|| {
        solve_adipls_fgong_modes(
            fgong_str,
            l,
            freq_min_uhz,
            freq_max_uhz,
            n_scan,
        )
        .into_iter()
        .map(|m| {
            (
                m.l,
                m.n_p,
                m.n_g,
                m.n_pg,
                m.freq_uhz,
                m.freq_var_uhz,
                m.e_adipls,
                m.e_norm,
            )
        })
        .collect()
    })
}

/// Exports `model` as a `STELLA` `(hyd_str, abn_str)` pair (`stella_io.f90`).
#[pyfunction]
pub fn star_export_stella_pair(
    py: Python<'_>,
    model: &PyStarModel,
) -> (String, String) {
    py.detach(|| {
        export_stella_pair(&model.inner, &StellaExportOptions::default())
    })
}

/// Runs the `STELLA` radiation-hydrodynamics light-curve and Sobolev `Fe II` 5169
/// line-velocity evaluation on `.hyd`, `.abn`, and `FeII_5169_eta.dat` strings.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn star_run_stella_from_hyd_abn(
    py: Python<'_>,
    hyd_str: &str,
    abn_str: &str,
    eta_str: &str,
    times_days: Vec<f64>,
) -> (Vec<(f64, f64)>, Vec<(f64, f64, f64, f64)>) {
    py.detach(|| {
        let hyd = parse_stella_hyd(hyd_str);
        let abn = parse_stella_abn(abn_str);
        let eta = FeII5169EtaTable::parse(eta_str);
        let (lbol, vel) =
            run_stella_from_hyd_abn(&hyd, &abn, &eta, &times_days);
        let v_rows = vel
            .into_iter()
            .map(|r| {
                (
                    r.t_post_max_days,
                    r.v_tau_lo_km_s,
                    r.v_tau_med_km_s,
                    r.v_tau_hi_km_s,
                )
            })
            .collect();
        (lbol, v_rows)
    })
}

/// Registers Wave 5.2 functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_eval_hllc_face, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_hllc_surface_face, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_riemann_momentum_residual, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_riemann_profile, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_rsp2_zone, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_rsp2_face, m)?)?;
    m.add_function(wrap_pyfunction!(star_solve_stationary_rsp2_w, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_center_parabolic, m)?)?;
    m.add_function(wrap_pyfunction!(star_build_pulse_points, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_gyre, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_fgong, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_osc, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_gsm, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_saio, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_cafein, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_gr1d, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_pulse_by_format, m)?)?;
    m.add_function(wrap_pyfunction!(star_solve_gyre_mesa_modes, m)?)?;
    m.add_function(wrap_pyfunction!(star_solve_adipls_fgong_modes, m)?)?;
    m.add_function(wrap_pyfunction!(star_export_stella_pair, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_stella_from_hyd_abn, m)?)?;
    Ok(())
}
