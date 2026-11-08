// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dimensionless Dziembowski pulsation variables, parabolic center
//! extrapolation, composition-jump double-point insertion, and `GYRE` (v120),
//! `FGONG` (v1300), and `OSC` ASCII pulsation exporters
//! (`star/private/pulse_gyre.f90`, `pulse_fgong.f90`, `pulse_osc.f90`, and
//! `pulse_utils.f90`, resolving Issue #4).

use std::fmt::Write as _;

use flexmesa_const::{FOUR_THIRDS_PI, PI4, SECYER, STANDARD_CGRAV};

use crate::model::StarModel;

/// Pulsation profile point containing physical and dimensionless Dziembowski
/// variables $(V, V_g, A_*, U, c_1)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulsePoint {
    /// Radial coordinate $r$ ($\mathrm{cm}$).
    pub r: f64,
    /// Enclosed mass $m$ ($\mathrm{g}$).
    pub m: f64,
    /// Luminosity $L(r)$ ($\mathrm{erg\,s^{-1}}$).
    pub lum: f64,
    /// Total pressure $P$ ($\mathrm{dyn\,cm^{-2}}$).
    pub p: f64,
    /// Temperature $T$ ($\mathrm{K}$).
    pub t: f64,
    /// Density $\rho$ ($\mathrm{g\,cm^{-3}}$).
    pub rho: f64,
    /// Temperature gradient $\nabla = \mathrm{d}\ln T/\mathrm{d}\ln P$.
    pub grad_t: f64,
    /// Squared Brunt-Väisälä frequency $N^2$ ($\mathrm{s^{-2}}$).
    pub n2: f64,
    /// First adiabatic exponent $\Gamma_1$.
    pub gamma1: f64,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    pub grad_ad: f64,
    /// Volumetric expansion coefficient
    /// $\delta = -(\partial\ln\rho/\partial\ln T)_P = \chi_T / \chi_\rho$.
    pub delta: f64,
    /// Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$).
    pub kap: f64,
    /// Opacity temperature partial
    /// $\kappa (\partial\ln\kappa/\partial\ln T)_\rho$.
    pub kap_t: f64,
    /// Opacity density partial $\kappa (\partial\ln\kappa/\partial\ln\rho)_T$.
    pub kap_rho: f64,
    /// Net nuclear + neutrino energy generation $\epsilon_{\mathrm{net}}$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub eps: f64,
    /// Energy generation temperature partial
    /// $\epsilon (\partial\ln\epsilon/\partial\ln T)_\rho$.
    pub eps_t: f64,
    /// Energy generation density partial
    /// $\epsilon (\partial\ln\epsilon/\partial\ln\rho)_T$.
    pub eps_rho: f64,
    /// Rotational angular velocity $\Omega$ ($\mathrm{rad\,s^{-1}}$).
    pub omega: f64,
    /// Hydrogen mass fraction $X$ (`h1`).
    pub x_h1: f64,
    /// Helium-4 mass fraction $Y$ (`he4`).
    pub x_he4: f64,
    /// Heavy-element mass fraction $Z = 1 - X - Y$.
    pub z_metal: f64,
    /// Dziembowski homology variable $V = G m \rho / (P r)$.
    pub v_dziem: f64,
    /// Dziembowski acoustic variable $V_g = V / \Gamma_1$.
    pub v_g: f64,
    /// Dziembowski buoyancy variable $A_* = r N^2 / g$.
    pub a_star: f64,
    /// Dziembowski mass concentration variable $U = 4\pi r^3 \rho / m$.
    pub u_dziem: f64,
    /// Dziembowski scaled volume variable $c_1 = (r/R)^3 / (m/M)$.
    pub c_1: f64,
}

/// Parabolically extrapolates $v(r) = v_c + a r^2$ to $r = 0$ from two inner
/// radii (`eval_center` in `star/private/pulse_utils.f90`):
/// $$v_c = \frac{v_{\mathrm{in}} r_{\mathrm{out}}^2 -
///   v_{\mathrm{out}} r_{\mathrm{in}}^2}{r_{\mathrm{out}}^2 - r_{\mathrm{in}}^2}.$$
#[must_use]
pub fn eval_center_parabolic(
    r_inner: f64,
    r_outer: f64,
    v_inner: f64,
    v_outer: f64,
) -> f64 {
    let r2_in = r_inner * r_inner;
    let r2_out = r_outer * r_outer;
    let denom = r2_out - r2_in;
    if denom.abs() > 1.0e-30 {
        (v_inner * r2_out - v_outer * r2_in) / denom
    } else {
        v_inner
    }
}

fn zone_abundances(model: &StarModel, k: usize) -> (f64, f64, f64) {
    let spec = model.species;
    let x_h1 = model
        .xh_mass
        .get(k)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or_else(|| if spec > 0 { model.xa[k * spec] } else { 0.7 });
    let x_he4 = model
        .yhe_mass
        .get(k)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or_else(|| {
            if spec > 2 {
                model.xa[k * spec + 2]
            } else if spec > 1 {
                model.xa[k * spec + 1]
            } else {
                0.28
            }
        });
    let z_metal = (1.0 - x_h1 - x_he4).clamp(0.0, 1.0);
    (x_h1, x_he4, z_metal)
}

fn make_pulse_point(
    model: &StarModel,
    k: usize,
    r_tot: f64,
    m_tot: f64,
    thermo_k: usize,
) -> PulsePoint {
    let r = model.r[k].max(1.0e-10);
    let m = model.m[k].max(1.0e-30);
    let lum = model.lum[k];
    let p = model.peos[k].max(1.0e-30);
    let t = model.t[k].max(1.0);
    let rho = model.rho[thermo_k].max(1.0e-99);
    let grad_t = model.gradt[k];
    let n2 = model.brunt_n2[k];
    let gamma1 = model.gamma1[thermo_k].max(1.0);
    let grad_ad = model.grada[thermo_k];
    let delta = (model.chi_t[thermo_k] / model.chi_rho[thermo_k].max(1.0e-30))
        .max(1.0e-10);
    let kap = model.opacity[thermo_k];
    let kap_t = model.d_opacity_dlnt[thermo_k];
    let kap_rho = model.d_opacity_dlnd[thermo_k];
    let eps = model.eps_nuc[thermo_k] - model.non_nuc_neu[thermo_k];
    let eps_t =
        model.d_epsnuc_dlnt[thermo_k] - model.d_non_nuc_neu_dlnt[thermo_k];
    let eps_rho =
        model.d_epsnuc_dlnd[thermo_k] - model.d_non_nuc_neu_dlnd[thermo_k];
    let omega = model.omega[k];
    let (x_h1, x_he4, z_metal) = zone_abundances(model, thermo_k);

    let g = (model.cgrav[k] * model.m_grav[k] / (r * r)).max(1.0e-30);
    let v_dziem = g * r * rho / p;
    let v_g = v_dziem / gamma1;
    let a_star = r * n2 / g;
    let u_dziem = PI4 * r.powi(3) * rho / m;
    let c_1 = (r / r_tot).powi(3) / (m / m_tot);

    PulsePoint {
        r,
        m,
        lum,
        p,
        t,
        rho,
        grad_t,
        n2,
        gamma1,
        grad_ad,
        delta,
        kap,
        kap_t,
        kap_rho,
        eps,
        eps_t,
        eps_rho,
        omega,
        x_h1,
        x_he4,
        z_metal,
        v_dziem,
        v_g,
        a_star,
        u_dziem,
        c_1,
    }
}

/// Builds the center-to-surface sequence of [`PulsePoint`] records for `model`,
/// with optional $r = 0$ parabolic center extrapolation and composition-jump
/// double-point splitting (`add_double_points` when $|\nabla_\mu| \ge
/// \text{grad\_mu\_threshold}$).
#[must_use]
pub fn build_pulse_points(
    model: &StarModel,
    add_center_point: bool,
    add_double_points: bool,
    grad_mu_threshold: f64,
) -> Vec<PulsePoint> {
    let nz = model.nz;
    if nz == 0 {
        return Vec::new();
    }
    let m_tot = model.mstar.max(1.0e-30);
    let r_tot = model.r[0].max(1.0e-30);
    let rho_mean = m_tot / (FOUR_THIRDS_PI * r_tot.powi(3));

    let mut pts = Vec::with_capacity(nz + 4);
    if add_center_point && model.r_center <= 0.0 && nz >= 2 {
        let k_in = nz - 1;
        let k_out = nz - 2;
        let r_in = model.r[k_in];
        let r_out = model.r[k_out];
        let mut p_c = make_pulse_point(model, k_in, r_tot, m_tot, k_in);
        p_c.r = 0.0;
        p_c.m = 0.0;
        p_c.lum = 0.0;
        p_c.p = eval_center_parabolic(
            r_in,
            r_out,
            model.peos[k_in],
            model.peos[k_out],
        )
        .max(1.0e-30);
        p_c.t =
            eval_center_parabolic(r_in, r_out, model.t[k_in], model.t[k_out])
                .max(1.0);
        p_c.rho = eval_center_parabolic(
            r_in,
            r_out,
            model.rho[k_in],
            model.rho[k_out],
        )
        .max(1.0e-99);
        p_c.n2 = 0.0;
        p_c.v_dziem = 0.0;
        p_c.v_g = 0.0;
        p_c.a_star = 0.0;
        p_c.u_dziem = 3.0;
        p_c.c_1 = (rho_mean / p_c.rho).max(1.0e-30);
        pts.push(p_c);
    }

    for k in (0..nz).rev() {
        pts.push(make_pulse_point(model, k, r_tot, m_tot, k));
        if add_double_points
            && k > 0
            && model.brunt_b[k].abs() >= grad_mu_threshold
        {
            pts.push(make_pulse_point(model, k, r_tot, m_tot, k - 1));
        }
    }

    pts
}

/// Exports `model` in `GYRE` version `120` ASCII stellar model format
/// (`star/private/pulse_gyre.f90`, center-to-surface ordering).
#[must_use]
pub fn export_gyre(model: &StarModel, add_center_point: bool) -> String {
    let pts = build_pulse_points(model, add_center_point, false, 1.0e99);
    let m_tot = model.mstar;
    let r_tot = if model.nz > 0 { model.r[0] } else { 0.0 };
    let l_tot = if model.nz > 0 { model.lum[0] } else { 0.0 };

    let mut out = String::with_capacity(128 + pts.len() * 340);
    let _ = writeln!(
        out,
        "{:6} {:23.16E} {:23.16E} {:23.16E} {:6}",
        pts.len(),
        m_tot,
        r_tot,
        l_tot,
        120
    );
    for (idx, pt) in pts.iter().enumerate() {
        let _ = writeln!(
            out,
            "{:6} {:23.16E} {:23.16E} {:23.16E} {:23.16E} {:23.16E} \
             {:23.16E} {:23.16E} {:23.16E} {:23.16E} {:23.16E} {:23.16E} \
             {:23.16E} {:23.16E} {:23.16E} {:23.16E} {:23.16E} {:23.16E} \
             {:23.16E}",
            idx + 1,
            pt.r,
            pt.m,
            pt.lum,
            pt.p,
            pt.t,
            pt.rho,
            pt.grad_t,
            pt.n2,
            pt.gamma1,
            pt.grad_ad,
            pt.delta,
            pt.kap,
            pt.kap_t,
            pt.kap_rho,
            pt.eps,
            pt.eps_t,
            pt.eps_rho,
            pt.omega
        );
    }
    out
}

fn write_5e16(out: &mut String, vals: &[f64]) {
    for chunk in vals.chunks(5) {
        for &v in chunk {
            let _ = write!(out, "{v:16.9E}");
        }
        out.push('\n');
    }
}

fn write_5e19(out: &mut String, vals: &[f64]) {
    for chunk in vals.chunks(5) {
        for &v in chunk {
            let _ = write!(out, "{v:19.12E}");
        }
        out.push('\n');
    }
}

/// Exports `model` in `FGONG` version `1300` ASCII pulsation format
/// (`star/private/pulse_fgong.f90`, surface-to-center ordering, 15 global
/// constants, 40 point variables in `5E16.9` rows).
#[must_use]
pub fn export_fgong(model: &StarModel, add_center_point: bool) -> String {
    let pts = build_pulse_points(model, add_center_point, false, 1.0e99);
    let m_tot = model.mstar.max(1.0e-30);
    let r_tot = if model.nz > 0 { model.r[0] } else { 1.0 };
    let l_tot = if model.nz > 0 { model.lum[0] } else { 0.0 };
    let z_init = model.controls.initial_z;
    let x_init = 1.0 - model.controls.initial_y - z_init;
    let p_c = pts.first().map_or(0.0, |p| p.p);
    let t_c = pts.first().map_or(0.0, |p| p.t);

    let mut out = String::with_capacity(512 + pts.len() * 680);
    out.push_str("FGONG model exported by FlexMESA\n");
    out.push_str("1D Lagrangian stellar evolution profile\n");
    out.push_str("Variables: 15 global constants, 40 point columns (v1300)\n");
    out.push_str("Surface-to-center ordering\n");
    let _ = writeln!(out, "{:10}{:10}{:10}{:10}", pts.len(), 15, 40, 1300);

    let glob = [
        m_tot,
        r_tot,
        l_tot,
        z_init,
        x_init,
        model.controls.mixing_length_alpha,
        0.0,
        0.0,
        0.0,
        p_c,
        t_c,
        0.0,
        model.star_age * SECYER,
        t_c,
        model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV),
    ];
    write_5e16(&mut out, &glob);

    for pt in pts.iter().rev() {
        let ln_q = (pt.m / m_tot).max(1.0e-30).ln();
        let mut var = [0.0_f64; 40];
        var[0] = pt.r;
        var[1] = ln_q;
        var[2] = pt.t;
        var[3] = pt.p;
        var[4] = pt.rho;
        var[5] = pt.x_h1;
        var[6] = pt.lum;
        var[7] = pt.kap;
        var[8] = pt.eps;
        var[9] = pt.gamma1;
        var[10] = pt.grad_ad;
        var[11] = pt.delta;
        var[14] = pt.a_star;
        var[16] = pt.z_metal;
        var[17] = pt.r - r_tot;
        var[20] = pt.x_he4;
        write_5e16(&mut out, &var);
    }
    out
}

/// Exports `model` in `OSC` (Aarhus / CESAM2k) ASCII pulsation format
/// (`star/private/pulse_osc.f90`, surface-to-center ordering, 15 global
/// constants and 25 point columns across `5E19.12` rows).
#[must_use]
pub fn export_osc(model: &StarModel, add_center_point: bool) -> String {
    let pts = build_pulse_points(model, add_center_point, false, 1.0e99);
    let m_tot = model.mstar.max(1.0e-30);
    let r_tot = if model.nz > 0 { model.r[0] } else { 1.0 };
    let l_tot = if model.nz > 0 { model.lum[0] } else { 0.0 };
    let z_init = model.controls.initial_z;
    let x_init = 1.0 - model.controls.initial_y - z_init;
    let p_c = pts.first().map_or(0.0, |p| p.p);
    let t_c = pts.first().map_or(0.0, |p| p.t);

    let mut out = String::with_capacity(512 + pts.len() * 520);
    out.push_str("OSC stellar model exported by FlexMESA\n");
    let _ =
        writeln!(out, "{:10}{:10}{:10}{:10}{:10}", pts.len(), 15, 22, 3, 2000);
    let glob = [
        m_tot,
        r_tot,
        l_tot,
        z_init,
        x_init,
        model.controls.mixing_length_alpha,
        0.5,
        1.0 / 12.0,
        0.0,
        p_c,
        t_c,
        model.star_age * SECYER * 1.0e-6,
        0.0,
        0.0,
        model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV),
    ];
    write_5e19(&mut out, &glob);

    for pt in pts.iter().rev() {
        let ln_q = (pt.m / m_tot).max(1.0e-30).ln();
        let var = [
            pt.r, ln_q, pt.t, pt.p, pt.rho, pt.grad_t, pt.lum, pt.kap, pt.eps,
            pt.gamma1, pt.grad_ad, pt.delta, 0.0, 0.0, pt.a_star, pt.omega,
            pt.kap_t, pt.kap_rho, pt.eps_t, pt.eps_rho, 1.0, pt.grad_t,
            pt.x_h1, pt.x_he4, pt.z_metal,
        ];
        write_5e19(&mut out, &var);
    }
    out
}
