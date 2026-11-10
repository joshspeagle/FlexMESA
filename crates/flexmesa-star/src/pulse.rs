// SPDX-License-Identifier: LGPL-3.0-or-later
//! `GYRE` (`v101`/`v110`/`v120`), `FGONG` (`v300`/`v1300`, 40 columns), and
//! `OSC` (36 columns) ASCII pulsation exporters (`star/private/pulse_gyre.f90`,
//! `pulse_fgong.f90`, and `pulse_osc.f90`, resolving Issue #4).

use std::fmt::Write as _;

use flexmesa_const::{FOUR_THIRDS_PI, PI4, SECYER, STANDARD_CGRAV};

use crate::model::StarModel;
use crate::pulse_utils::make_pulse_point_on_segment;
pub use crate::pulse_utils::{
    build_pulse_points, build_pulse_points_with_options, eval_center_d2,
    eval_center_parabolic, eval_center_rho, eval_face, eval_face_a_ast,
    set_segment_indices, PulseExportOptions, PulsePoint, PULSE_ISO_NAMES,
};

/// Builds an optically thin $T(\tau)$ atmosphere above `model`'s surface (`build_atm`).
#[must_use]
pub fn build_pulse_atmosphere_points(
    model: &StarModel,
    r_tot: f64,
    m_tot: f64,
    n_atm: usize,
) -> Vec<PulsePoint> {
    if model.nz == 0 || n_atm < 2 {
        return Vec::new();
    }
    let surf = make_pulse_point_on_segment(
        model,
        0,
        0,
        model.nz - 1,
        r_tot,
        m_tot,
        true,
    );
    let tau_base = model.tau_base.max(2.0 / 3.0);
    let tau_top = (1.0e-4 * tau_base).min(1.0e-4);
    let g_surf = (model.cgrav[0] * m_tot / (r_tot * r_tot)).max(1.0e-30);
    let teff = if model.teff > 100.0 {
        model.teff
    } else {
        surf.t
    };
    let hp_surf = (surf.p / (g_surf * surf.rho)).max(1.0);
    let mut atm = Vec::with_capacity(n_atm - 1);
    for i in (1..n_atm).rev() {
        let frac = (i as f64) / (n_atm as f64);
        let tau = (tau_top.ln() + frac * (tau_base.ln() - tau_top.ln())).exp();
        let t_tau = teff * (0.75 * (tau + 2.0 / 3.0)).powf(0.25);
        let p_ratio = (tau / tau_base).max(1.0e-8);
        let (p_tau, rho_tau) = (
            surf.p * p_ratio,
            surf.rho * p_ratio * (surf.t / t_tau.max(1.0)),
        );
        let r = (r_tot - hp_surf * p_ratio.ln()).max(r_tot);
        let g = (model.cgrav[0] * m_tot / (r * r)).max(1.0e-30);
        let grad_t = 0.1875 * tau / (tau + 2.0 / 3.0);
        let n2 =
            g * g * (rho_tau / p_tau) * surf.delta * (surf.grad_ad - grad_t);
        let mut pt = surf;
        pt.r = r;
        pt.p = p_tau;
        pt.pgas = p_tau * (surf.pgas / surf.p.max(1.0e-30));
        pt.t = t_tau;
        pt.rho = rho_tau;
        pt.grad_t = grad_t;
        pt.grad_r = grad_t;
        pt.tau = tau;
        pt.eps_nuc = 0.0;
        pt.eps = 0.0;
        pt.eps_grav = 0.0;
        pt.entropy = 0.0;
        pt.r_x_h1 = 0.0;
        pt.n2 = n2;
        pt.v_dziem = g * r * rho_tau / p_tau;
        pt.v_g = pt.v_dziem / pt.gamma1;
        pt.a_star = r * n2 / g;
        pt.u_dziem = PI4 * r.powi(3) * rho_tau / m_tot;
        pt.c_1 = (r / r_tot).powi(3);
        atm.push(pt);
    }
    atm
}

/// Exports `model` in `GYRE` ASCII stellar model format with schema `schema`
/// (`101`, `110`, or `120`; `star/private/pulse_gyre.f90`, center-to-surface
/// ordering).
#[must_use]
pub fn export_gyre_with_schema(
    model: &StarModel,
    add_center_point: bool,
    schema: usize,
) -> String {
    let pts = build_pulse_points(model, add_center_point, false, 1.0e99);
    let m_tot = model.m_grav.first().copied().unwrap_or(model.mstar);
    let r_tot = if model.nz > 0 { model.r[0] } else { 0.0 };
    let l_tot = if model.nz > 0 { model.lum[0] } else { 0.0 };
    let schema_ver = if schema == 101 || schema == 110 {
        schema
    } else {
        120
    };

    let mut out = String::with_capacity(128 + pts.len() * 480);
    let _ = writeln!(
        out,
        "{:6} {:26.16E} {:26.16E} {:26.16E} {:6}",
        pts.len(),
        m_tot,
        r_tot,
        l_tot,
        schema_ver
    );
    for (idx, pt) in pts.iter().enumerate() {
        if schema_ver == 120 {
            let _ = writeln!(
                out,
                "{:6} {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} \
                 {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} \
                 {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} \
                 {:26.16E} {:26.16E}",
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
                pt.eps_nuc,
                pt.eps_t,
                pt.eps_rho,
                pt.eps_grav,
                pt.omega
            );
        } else {
            let _ = writeln!(
                out,
                "{:6} {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} \
                 {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} \
                 {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} {:26.16E} \
                 {:26.16E}",
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
                pt.eps_nuc,
                pt.eps_t,
                pt.eps_rho,
                pt.omega
            );
        }
    }
    out
}

/// Exports `model` in `GYRE` version `120` ASCII stellar model format
/// (`star/private/pulse_gyre.f90`, center-to-surface ordering, 19 columns).
#[must_use]
pub fn export_gyre(model: &StarModel, add_center_point: bool) -> String {
    export_gyre_with_schema(model, add_center_point, 120)
}

pub(crate) fn write_5e16(out: &mut String, vals: &[f64]) {
    for chunk in vals.chunks(5) {
        for &v in chunk {
            let _ = write!(out, "{v:16.9E}");
        }
        out.push('\n');
    }
}

pub(crate) fn write_5e26(out: &mut String, vals: &[f64]) {
    for chunk in vals.chunks(5) {
        for &v in chunk {
            let _ = write!(out, " {v:26.18E}");
        }
        out.push('\n');
    }
}

pub(crate) fn write_5e19(out: &mut String, vals: &[f64]) {
    for chunk in vals.chunks(5) {
        for &v in chunk {
            let _ = write!(out, "{v:19.12E}");
        }
        out.push('\n');
    }
}

fn central_second_deriv_globals(
    model: &StarModel,
    pts: &[PulsePoint],
    r_tot: f64,
) -> (f64, f64) {
    let nz = model.nz;
    let p_c = pts.first().map_or(1.0, |p| p.p.max(1.0e-30));
    let rho_c = pts.first().map_or(1.0, |p| p.rho.max(1.0e-99));
    let g_c = model.cgrav.last().copied().unwrap_or(STANDARD_CGRAV);
    let d2p_dr2_c = -FOUR_THIRDS_PI * g_c * rho_c * rho_c;
    let d2rho_dr2_c = if nz >= 2 {
        let r_in = model.rmid.get(nz - 1).copied().unwrap_or(model.r[nz - 1]);
        let r_out = model.rmid.get(nz - 2).copied().unwrap_or(model.r[nz - 2]);
        eval_center_d2(r_in, r_out, model.rho[nz - 1], model.rho[nz - 2])
    } else {
        0.0
    };
    (
        r_tot * r_tot * d2p_dr2_c / p_c,
        r_tot * r_tot * d2rho_dr2_c / rho_c,
    )
}

/// Exports `model` in `FGONG` ASCII pulsation format with version `ivers`
/// (`300` or `1300`; `star/private/pulse_fgong.f90`, surface-to-center
/// ordering, 15 global constants, 40 point variables).
#[must_use]
pub fn export_fgong_with_version(
    model: &StarModel,
    add_center_point: bool,
    ivers: usize,
) -> String {
    let pts = build_pulse_points(model, add_center_point, false, 1.0e99);
    let m_tot = model
        .m_grav
        .first()
        .copied()
        .unwrap_or(model.mstar)
        .max(1.0e-30);
    let r_tot = if model.nz > 0 { model.r[0] } else { 1.0 };
    let l_tot = if model.nz > 0 { model.lum[0] } else { 0.0 };
    let z_init = model.controls.initial_z;
    let x_init = 1.0 - model.controls.initial_y - z_init;
    let (d2p_dimless, d2rho_dimless) =
        central_second_deriv_globals(model, &pts, r_tot);
    let teff = if model.teff > 0.0 {
        model.teff
    } else {
        pts.last().map_or(5777.0, |p| p.t)
    };
    let fgong_ver = if ivers == 300 { 300 } else { 1300 };

    let mut out = String::with_capacity(512 + pts.len() * 900);
    out.push_str("FGONG file\n");
    out.push_str("Created by FlexMESA\n");
    out.push('\n');
    out.push('\n');
    let _ = writeln!(out, "{:10}{:10}{:10}{:10}", pts.len(), 15, 40, fgong_ver);

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
        0.0,
        d2p_dimless,
        d2rho_dimless,
        model.star_age,
        teff,
        STANDARD_CGRAV,
    ];
    if fgong_ver == 300 {
        write_5e16(&mut out, &glob);
    } else {
        write_5e26(&mut out, &glob);
    }

    let ln_tiny = f64::MIN_POSITIVE.ln();
    for pt in pts.iter().rev() {
        let ln_q = if pt.m <= 0.0 {
            ln_tiny
        } else {
            (pt.m / m_tot).max(f64::MIN_POSITIVE).ln()
        };
        let mut var = [0.0_f64; 40];
        var[0] = pt.r;
        var[1] = ln_q;
        var[2] = pt.t;
        var[3] = pt.p;
        var[4] = pt.rho;
        var[5] = pt.x_h1;
        var[6] = pt.lum;
        var[7] = pt.kap;
        var[8] = pt.eps_nuc + pt.eps_grav;
        var[9] = pt.gamma1;
        var[10] = pt.grad_ad;
        var[11] = pt.delta;
        var[12] = pt.cp;
        var[13] = pt.rec_mu_e;
        var[14] = pt.a_star;
        var[15] = pt.r_x_h1;
        var[16] = pt.z_metal;
        var[17] = r_tot - pt.r;
        var[18] = pt.eps_grav;
        var[19] = 0.0;
        // Isotopes mapped to PULSE_ISO_NAMES:
        // [0]=h1, [1]=h2, [2]=he3, [3]=he4, [4]=li7, [5]=be7, [6]=be9,
        // [7]=c12, [8]=c13, [9]=n14, [10]=n15, [11]=o16, [12]=o17, [13]=o18, [14]=ne20, [15]=si28
        var[20] = pt.x_iso[2]; // X_He3
        var[21] = pt.x_iso[7]; // X_C12
        var[22] = pt.x_iso[8]; // X_C13
        var[23] = pt.x_iso[9]; // X_N14
        var[24] = pt.x_iso[11]; // X_O16
        var[25] = pt.dln_gamma1_dlnrho;
        var[26] = pt.dln_gamma1_dlnp;
        var[27] = pt.dln_gamma1_dy;
        var[28] = pt.x_iso[1]; // X_H2
        var[29] = pt.x_he4; // X_He4
        var[30] = pt.x_iso[4]; // X_Li7
        var[31] = pt.x_iso[5]; // X_Be7
        var[32] = pt.x_iso[10]; // X_N15
        var[33] = pt.x_iso[12]; // X_O17
        var[34] = pt.x_iso[13]; // X_O18
        var[35] = pt.x_iso[14]; // X_Ne20
        if fgong_ver == 300 {
            write_5e16(&mut out, &var);
        } else {
            write_5e26(&mut out, &var);
        }
    }
    out
}

/// Exports `model` in `FGONG` version `1300` ASCII pulsation format
/// (`star/private/pulse_fgong.f90`, surface-to-center ordering, 15 global
/// constants, 40 point variables).
#[must_use]
pub fn export_fgong(model: &StarModel, add_center_point: bool) -> String {
    export_fgong_with_version(model, add_center_point, 1300)
}

/// Exports `model` in `OSC` (CoRoT / CESAM2k) 36-column ASCII pulsation format
/// (`star/private/pulse_osc.f90`, surface-to-center ordering, 15 global
/// constants, 22 structure columns + 14 abundance columns across `5E19.12` rows).
#[must_use]
pub fn export_osc(model: &StarModel, add_center_point: bool) -> String {
    let pts = build_pulse_points(model, add_center_point, false, 1.0e99);
    let m_tot = model
        .m_grav
        .first()
        .copied()
        .unwrap_or(model.mstar)
        .max(1.0e-30);
    let r_tot = if model.nz > 0 { model.r[0] } else { 1.0 };
    let l_tot = if model.nz > 0 { model.lum[0] } else { 0.0 };
    let z_init = model.controls.initial_z;
    let x_init = 1.0 - model.controls.initial_y - z_init;
    let (d2p_dimless, d2rho_dimless) =
        central_second_deriv_globals(model, &pts, r_tot);
    let omega_surf = model.omega.first().copied().unwrap_or(0.0);

    let mut out = String::with_capacity(512 + pts.len() * 760);
    out.push_str("OSC file\n");
    out.push_str("Created by FlexMESA\n");
    out.push('\n');
    out.push('\n');
    out.push_str("14 H1 H2 He3 He4 Li7 Be7 C12 C13 N14 N15 O16 O17 Be9 Si28\n");
    let _ = writeln!(
        out,
        "{:10}{:10}{:10}{:10}{:10}",
        pts.len(),
        15,
        22,
        14,
        2000
    );

    let glob = [
        m_tot,
        r_tot,
        l_tot,
        z_init,
        x_init,
        model.controls.mixing_length_alpha,
        pts.last().map_or(x_init, |p| p.x_h1),
        pts.last().map_or(model.controls.initial_y, |p| p.x_he4),
        d2p_dimless,
        d2rho_dimless,
        model.star_age * SECYER * 1.0e-6,
        omega_surf,
        0.0,
        0.0,
        STANDARD_CGRAV,
    ];
    write_5e19(&mut out, &glob);

    let ln_tiny = f64::MIN_POSITIVE.ln();
    for pt in pts.iter().rev() {
        let ln_q = if pt.m <= 0.0 {
            ln_tiny
        } else {
            (pt.m / m_tot).max(f64::MIN_POSITIVE).ln()
        };
        let kap_safe = pt.kap.max(1.0e-30);
        let p_tot_gas = pt.p / pt.pgas.max(1.0e-30);
        let var = [
            pt.r,
            ln_q,
            pt.t,
            pt.p,
            pt.rho,
            pt.grad_t,
            pt.lum,
            pt.kap,
            pt.eps_nuc + pt.eps_grav,
            pt.gamma1,
            pt.grad_ad,
            pt.delta,
            pt.cp,
            pt.rec_mu_e,
            pt.a_star,
            pt.omega,
            pt.kap_t / kap_safe,
            pt.kap_rho / kap_safe,
            pt.eps_t,
            pt.eps_rho,
            p_tot_gas,
            pt.grad_r,
            // 14 isotopes: H1, H2, He3, He4, Li7, Be7, C12, C13, N14, N15, O16, O17, Be9, Si28
            pt.x_iso[0],
            pt.x_iso[1],
            pt.x_iso[2],
            pt.x_iso[3],
            pt.x_iso[4],
            pt.x_iso[5],
            pt.x_iso[7],
            pt.x_iso[8],
            pt.x_iso[9],
            pt.x_iso[10],
            pt.x_iso[11],
            pt.x_iso[12],
            pt.x_iso[6],
            pt.x_iso[15],
        ];
        write_5e19(&mut out, &var);
    }
    out
}
