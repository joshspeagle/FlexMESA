// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared pulsation grid construction, cell-to-face interpolation, composition
//! double-point splitting, $T(\tau)$ atmosphere attachment, and central
//! $r \to 0$ parabolic extrapolation (`star/private/pulse_utils.f90` and
//! `star_utils.f90`, resolving Issue #4).

#![allow(missing_docs)]

use flexmesa_const::{FOUR_THIRDS_PI, PI4};
use flexmesa_eos::{I_GAMMA1, NUM_EOS_BASIC_RESULTS};

use crate::model::StarModel;

/// Canonical 16 light-element species exported across `FGONG` and `OSC`.
pub const PULSE_ISO_NAMES: [&str; 16] = [
    "h1", "h2", "he3", "he4", "li7", "be7", "be9", "c12", "c13", "n14", "n15",
    "o16", "o17", "o18", "ne20", "si28",
];

/// Configuration options for constructing a [`PulsePoint`] sequence (`pulse_utils.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseExportOptions {
    pub add_center_point: bool,
    pub keep_surface_point: bool,
    pub add_atmosphere: bool,
    pub add_double_points: bool,
    pub grad_mu_threshold: f64,
    pub max_double_points: usize,
    pub interpolate_rho: bool,
}

impl Default for PulseExportOptions {
    fn default() -> Self {
        Self {
            add_center_point: true,
            keep_surface_point: true,
            add_atmosphere: false,
            add_double_points: false,
            grad_mu_threshold: 10.0,
            max_double_points: 0,
            interpolate_rho: true,
        }
    }
}

/// Pulsation profile point containing physical, thermodynamic, nuclear, and
/// dimensionless Dziembowski variables $(V, V_g, A_*, U, c_1)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulsePoint {
    pub r: f64,
    pub m: f64,
    pub lum: f64,
    pub p: f64,
    pub pgas: f64,
    pub t: f64,
    pub rho: f64,
    pub grad_t: f64,
    pub grad_r: f64,
    pub n2: f64,
    pub gamma1: f64,
    pub grad_ad: f64,
    pub delta: f64,
    pub cv: f64,
    pub cp: f64,
    pub chi_rho: f64,
    pub chi_t: f64,
    pub rec_mu_e: f64,
    pub entropy: f64,
    pub kap: f64,
    pub kap_t: f64,
    pub kap_rho: f64,
    pub eps_nuc: f64,
    pub eps: f64,
    pub eps_t: f64,
    pub eps_rho: f64,
    pub eps_grav: f64,
    pub omega: f64,
    pub tau: f64,
    pub scale_height: f64,
    pub x_h1: f64,
    pub x_he4: f64,
    pub z_metal: f64,
    pub r_x_h1: f64,
    pub dln_gamma1_dlnrho: f64,
    pub dln_gamma1_dlnp: f64,
    pub dln_gamma1_dy: f64,
    pub x_iso: [f64; 16],
    pub v_dziem: f64,
    pub v_g: f64,
    pub a_star: f64,
    pub u_dziem: f64,
    pub c_1: f64,
}

/// Parabolically extrapolates $v(r) = v_c + a r^2$ to $r = 0$ (`eval_center` in `pulse_utils.f90`).
#[must_use]
pub fn eval_center_parabolic(
    r_in: f64,
    r_out: f64,
    v_in: f64,
    v_out: f64,
) -> f64 {
    let (r2_in, r2_out) = (r_in * r_in, r_out * r_out);
    let d = r2_out - r2_in;
    if d.abs() > 1.0e-30 {
        (v_in * r2_out - v_out * r2_in) / d
    } else {
        v_in
    }
}

/// Evaluates $\mathrm{d}^2 v / \mathrm{d}r^2|_{r=0}$ from the parabolic fit (`eval_center_d2`).
#[must_use]
pub fn eval_center_d2(r_in: f64, r_out: f64, v_in: f64, v_out: f64) -> f64 {
    let d = r_out * r_out - r_in * r_in;
    if d.abs() > 1.0e-30 {
        2.0 * (v_out - v_in) / d
    } else {
        0.0
    }
}

/// Evaluates the mass-conserving central density $\rho_c$ (`eval_center_rho` in `pulse_utils.f90`).
#[must_use]
pub fn eval_center_rho(model: &StarModel, k_b: usize) -> f64 {
    let r1 = model
        .rmid
        .get(k_b)
        .copied()
        .unwrap_or(0.5 * model.r[k_b])
        .max(1.0e-10);
    let rho1 = model.rho[k_b].max(1.0e-99);
    let m1 = (model.m[k_b] - 0.5 * model.dm[k_b]).max(0.0);
    let fit =
        0.375 * ((5.0 * m1) / (std::f64::consts::PI * r1.powi(3)) - 4.0 * rho1);
    if fit > 0.1 * rho1 && fit < 10.0 * rho1 {
        fit
    } else if k_b >= 1 {
        let (r_in, r_out) = (model.r[k_b], model.r[k_b - 1]);
        eval_center_parabolic(r_in, r_out, model.rho[k_b], model.rho[k_b - 1])
            .max(1.0e-99)
    } else {
        rho1
    }
}

/// Partitions `0..nz` into continuous composition segments `(k_a, k_b)` split at
/// faces where $|\nabla_\mu(k)| > \text{grad\_mu\_threshold}$ (`set_segment_indices`).
#[must_use]
pub fn set_segment_indices(
    model: &StarModel,
    add_double_points: bool,
    grad_mu_threshold: f64,
    max_double_points: usize,
    include_last_face: bool,
) -> Vec<(usize, usize)> {
    let nz = model.nz;
    if nz == 0 {
        return Vec::new();
    }
    if !add_double_points || nz < 2 {
        return vec![(0, nz - 1)];
    }
    let k_end = if include_last_face { nz } else { nz - 1 };
    let mut cands: Vec<(usize, f64)> = Vec::new();
    for k in 1..k_end {
        let dlnp =
            (model.peos[k].max(1.0e-30) / model.peos[k - 1].max(1.0e-30)).ln();
        let dlnmu =
            (model.mu[k].max(1.0e-10) / model.mu[k - 1].max(1.0e-10)).ln();
        let gmu = if dlnp.abs() > 1.0e-14 {
            dlnmu / dlnp
        } else {
            model.brunt_b[k]
        };
        if gmu.abs() > grad_mu_threshold {
            cands.push((k, gmu.abs()));
        }
    }
    if max_double_points > 0 && cands.len() > max_double_points {
        cands.sort_by(|a, b| b.1.total_cmp(&a.1));
        cands.truncate(max_double_points);
        cands.sort_by_key(|c| c.0);
    }
    let mut segs = Vec::with_capacity(cands.len() + 1);
    let mut k_a = 0;
    for (k_s, _) in cands {
        if k_s > k_a {
            segs.push((k_a, k_s - 1));
            k_a = k_s;
        }
    }
    segs.push((k_a, nz - 1));
    segs
}

/// Evaluates cell-centered slice `v` at face `k` using cells `k_a..=k_b` (`eval_face`).
#[must_use]
pub fn eval_face(
    dq: &[f64],
    v: &[f64],
    k: usize,
    k_a: usize,
    k_b: usize,
    v_lo: Option<f64>,
    v_hi: Option<f64>,
) -> f64 {
    let val = if k_a >= k_b {
        v[k_a]
    } else if k <= k_a {
        let (d0, d1) = (dq[k_a].max(1.0e-30), dq[k_a + 1].max(1.0e-30));
        v[k_a] - d0 * (v[k_a + 1] - v[k_a]) / (d0 + d1)
    } else if k == k_a + 1 {
        let (d0, d1) = (dq[k_a].max(1.0e-30), dq[k_a + 1].max(1.0e-30));
        (v[k_a] * d1 + v[k_a + 1] * d0) / (d0 + d1)
    } else if k == k_b {
        let (dm, db) = (dq[k_b - 1].max(1.0e-30), dq[k_b].max(1.0e-30));
        (v[k_b] * dm + v[k_b - 1] * db) / (db + dm)
    } else if k > k_b {
        let (dm, db) = (dq[k_b - 1].max(1.0e-30), dq[k_b].max(1.0e-30));
        v[k_b] + db * (v[k_b] - v[k_b - 1]) / (db + dm)
    } else {
        let h1 = 0.5 * (dq[k - 2] + dq[k - 1]).max(1.0e-30);
        let h2 = 0.5 * (dq[k - 1] + dq[k]).max(1.0e-30);
        let h3 = 0.5 * (dq[k] + dq[k + 1]).max(1.0e-30);
        let (s1, s2, s3) = (
            (v[k - 1] - v[k - 2]) / h1,
            (v[k] - v[k - 1]) / h2,
            (v[k + 1] - v[k]) / h3,
        );
        let (p1, p2) = (
            (s1 * h2 + s2 * h1) / (h1 + h2),
            (s2 * h3 + s3 * h2) / (h2 + h3),
        );
        let dy1 = if s1 * s2 <= 0.0 {
            0.0
        } else {
            s1.signum() * (2.0 * s1.abs()).min(2.0 * s2.abs()).min(p1.abs())
        };
        let dy2 = if s2 * s3 <= 0.0 {
            0.0
        } else {
            s2.signum() * (2.0 * s2.abs()).min(2.0 * s3.abs()).min(p2.abs())
        };
        let dx = 0.5 * dq[k - 1];
        let a = (dy1 + dy2 - 2.0 * s2) / (h2 * h2);
        let b = (3.0 * s2 - 2.0 * dy1 - dy2) / h2;
        v[k - 1] + dx * (dy1 + dx * (b + dx * a))
    };
    let val = v_lo.map_or(val, |lo| val.max(lo));
    v_hi.map_or(val, |hi| val.min(hi))
}

/// Evaluates $A^* = r N^2 / g$ at face `k` within segment `(k_a, k_b)` (`eval_face_A_ast`).
#[must_use]
pub fn eval_face_a_ast(
    model: &StarModel,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> f64 {
    let a_at = |idx: usize| {
        model.brunt_n2[idx] * model.r[idx] / model.grav[idx].max(1.0e-30)
    };
    if k_a >= k_b || (k > k_a && k <= k_b) {
        a_at(k.min(model.nz - 1))
    } else if k <= k_a {
        if k_a + 2 < model.nz {
            let (a1, a2) = (a_at(k_a + 1), a_at(k_a + 2));
            a1 - (model.dq[k_a] / model.dq[k_a + 1].max(1.0e-30)) * (a2 - a1)
        } else {
            a_at(k_a)
        }
    } else if k_b >= 1 {
        let g_b = model.grav[k_b].max(1.0e-30);
        let a1 = model.brunt_n2[k_b - 1] * model.r[k_b - 1] / g_b;
        let a2 = model.brunt_n2[k_b] * model.r[k_b] / g_b;
        a2 + (model.dq[k_b] / model.dq[k_b - 1].max(1.0e-30)) * (a2 - a1)
    } else {
        a_at(k_b)
    }
}

fn zone_iso_fractions(
    model: &StarModel,
    k: usize,
) -> (f64, f64, f64, [f64; 16]) {
    let spec = model.species;
    let mut x_iso = [0.0_f64; 16];
    for (idx, &name) in PULSE_ISO_NAMES.iter().enumerate() {
        if let Some(j) = model.iso_names.iter().position(|s| s == name) {
            x_iso[idx] = model.xa[k * spec + j].clamp(0.0, 1.0);
        }
    }
    let fallback_h = if x_iso[0] > 0.0 {
        x_iso[0]
    } else {
        model.xa.get(k * spec).copied().unwrap_or(0.7)
    };
    let fallback_he = if x_iso[3] > 0.0 {
        x_iso[3]
    } else {
        model.xa.get(k * spec + 2).copied().unwrap_or(0.28)
    };
    let x_h1 = model
        .xh_mass
        .get(k)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or(fallback_h);
    let x_he4 = model
        .yhe_mass
        .get(k)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or(fallback_he);
    if x_iso[0] <= 0.0 {
        x_iso[0] = x_h1;
    }
    if x_iso[3] <= 0.0 {
        x_iso[3] = x_he4;
    }
    (
        x_h1,
        x_he4,
        (1.0 - x_h1 - x_he4 - x_iso[2]).clamp(0.0, 1.0),
        x_iso,
    )
}

fn eval_gamma1_partials(model: &StarModel, k: usize) -> (f64, f64, f64) {
    let (g1, cr, ct) = (
        model.gamma1[k].max(1.0),
        model.chi_rho[k].max(1.0e-30),
        model.chi_t[k].max(1.0e-30),
    );
    let dg1_dlnd = model
        .eos_d_dlnd
        .get(k)
        .filter(|r| r.len() == NUM_EOS_BASIC_RESULTS)
        .map_or(0.0, |r| r[I_GAMMA1]);
    let dg1_dlnt = model
        .eos_d_dlnt
        .get(k)
        .filter(|r| r.len() == NUM_EOS_BASIC_RESULTS)
        .map_or(0.0, |r| r[I_GAMMA1]);
    let dln_g1_dlnrho = (dg1_dlnd - dg1_dlnt * (cr / ct)) / g1;
    let dln_g1_dlnp = dg1_dlnt / (g1 * ct);
    (dln_g1_dlnrho, dln_g1_dlnp, -0.625 * dln_g1_dlnp)
}

pub(crate) fn make_pulse_point_on_segment(
    model: &StarModel,
    face_k: usize,
    k_a: usize,
    k_b: usize,
    r_tot: f64,
    m_tot: f64,
    interp_rho: bool,
) -> PulsePoint {
    let nz = model.nz;
    let (gk, tk) = (face_k.min(nz - 1), face_k.clamp(k_a, k_b));
    let r = if face_k < nz {
        model.r[face_k].max(1.0e-10)
    } else {
        model.r_center.max(0.5 * model.r[nz - 1]).max(1.0e-10)
    };
    let m = if face_k < nz {
        model.m_grav[face_k].max(model.m[face_k]).max(1.0e-30)
    } else {
        model.m_center.max(0.5 * model.m[nz - 1]).max(1.0e-30)
    };
    let lum = if face_k < nz {
        model.lum[face_k]
    } else {
        model.l_center
    };
    let f_all = |v: &[f64], lo: Option<f64>| {
        eval_face(&model.dq, v, face_k, 0, nz - 1, lo, None)
    };
    let f_seg = |v: &[f64], lo: Option<f64>| {
        eval_face(&model.dq, v, face_k, k_a, k_b, lo, None)
    };
    let (p, pgas, t) = (
        f_all(&model.peos, Some(1.0e-30)),
        f_all(&model.pgas, Some(1.0e-30)),
        f_all(&model.t, Some(1.0)),
    );
    let rho = if interp_rho {
        f_seg(&model.rho, Some(1.0e-99))
    } else {
        model.rho[tk].max(1.0e-99)
    };
    let (gamma1, grad_ad, chi_t, chi_rho) = (
        f_seg(&model.gamma1, Some(1.0)),
        f_seg(&model.grada, None),
        f_seg(&model.chi_t, Some(1.0e-10)),
        f_seg(&model.chi_rho, Some(1.0e-30)),
    );
    let delta = (chi_t / chi_rho).max(1.0e-10);
    let (cv, cp, rec_mu_e, entropy) = (
        f_seg(&model.cv, Some(1.0e-30)),
        f_seg(&model.cp, Some(1.0e-30)),
        f_seg(&model.lnfree_e, None).exp(),
        f_seg(&model.entropy, Some(0.0)),
    );
    let (kap, kap_t, kap_rho) = (
        f_seg(&model.opacity, Some(1.0e-30)),
        f_seg(&model.d_opacity_dlnt, None),
        f_seg(&model.d_opacity_dlnd, None),
    );
    let eps_nuc = f_seg(&model.eps_nuc, None);
    let eps = eps_nuc - f_seg(&model.non_nuc_neu, None);
    let eps_t = f_seg(&model.d_epsnuc_dlnt, None)
        - f_seg(&model.d_non_nuc_neu_dlnt, None);
    let eps_rho = f_seg(&model.d_epsnuc_dlnd, None)
        - f_seg(&model.d_non_nuc_neu_dlnd, None);
    let eps_grav = model.eps_grav_ad.get(tk).map_or(0.0, |a| a.val);
    let (x_h1, x_he4, z_metal, x_iso) = zone_iso_fractions(model, tk);
    let r_x_h1 = model
        .iso_names
        .iter()
        .position(|s| s == "h1")
        .and_then(|j| model.dxdt_nuc.get(tk * model.species + j).copied())
        .unwrap_or(0.0);
    let (dln_gamma1_dlnrho, dln_gamma1_dlnp, dln_gamma1_dy) =
        eval_gamma1_partials(model, tk);
    let g = (model.cgrav[gk] * m / (r * r)).max(1.0e-30);
    let a_star = eval_face_a_ast(model, gk, k_a, k_b);
    let (n2, v_dziem) = (a_star * g / r, g * r * rho / p);

    PulsePoint {
        r,
        m,
        lum,
        p,
        pgas,
        t,
        rho,
        grad_t: model.gradt[gk],
        grad_r: model.gradr[gk],
        n2,
        gamma1,
        grad_ad,
        delta,
        cv,
        cp,
        chi_rho,
        chi_t,
        rec_mu_e,
        entropy,
        kap,
        kap_t,
        kap_rho,
        eps_nuc,
        eps,
        eps_t,
        eps_rho,
        eps_grav,
        omega: model.omega[gk],
        tau: model.tau[gk],
        scale_height: model.scale_height[gk],
        x_h1,
        x_he4,
        z_metal,
        r_x_h1,
        dln_gamma1_dlnrho,
        dln_gamma1_dlnp,
        dln_gamma1_dy,
        x_iso,
        v_dziem,
        v_g: v_dziem / gamma1,
        a_star,
        u_dziem: PI4 * r.powi(3) * rho / m,
        c_1: (r / r_tot).powi(3) / (m / m_tot),
    }
}

pub use crate::pulse::build_pulse_atmosphere_points;

/// Builds the center-to-surface sequence of [`PulsePoint`] records for `model`.
#[must_use]
pub fn build_pulse_points_with_options(
    model: &StarModel,
    opts: &PulseExportOptions,
) -> Vec<PulsePoint> {
    let nz = model.nz;
    if nz == 0 {
        return Vec::new();
    }
    let m_tot = model
        .m_grav
        .first()
        .copied()
        .unwrap_or(model.mstar)
        .max(1.0e-30);
    let r_tot = model.r[0].max(1.0e-30);
    let rho_mean = m_tot / (FOUR_THIRDS_PI * r_tot.powi(3));
    let segs = set_segment_indices(
        model,
        opts.add_double_points,
        opts.grad_mu_threshold,
        opts.max_double_points,
        opts.add_center_point,
    );
    let mut pts = Vec::with_capacity(nz + segs.len() + 16);
    if opts.add_center_point && model.r_center <= 0.0 && nz >= 2 {
        let &(ka, kb) = segs.last().unwrap_or(&(0, nz - 1));
        let (k_in, k_out) = (nz - 1, nz - 2);
        let r_in = model.rmid.get(k_in).copied().unwrap_or(model.r[k_in]);
        let r_out = model.rmid.get(k_out).copied().unwrap_or(model.r[k_out]);
        let mut p_c = make_pulse_point_on_segment(
            model,
            k_in,
            ka,
            kb,
            r_tot,
            m_tot,
            opts.interpolate_rho,
        );
        p_c.r = 0.0;
        p_c.m = 0.0;
        p_c.lum = 0.0;
        p_c.rho = if opts.interpolate_rho {
            eval_center_rho(model, k_in)
        } else {
            eval_center_parabolic(
                r_in,
                r_out,
                model.rho[k_in],
                model.rho[k_out],
            )
            .max(1.0e-99)
        };
        let d2p = -FOUR_THIRDS_PI * model.cgrav[k_in] * p_c.rho * p_c.rho;
        p_c.p = (model.peos[k_in] - 0.5 * d2p * r_in * r_in).max(1.0e-30);
        p_c.t =
            eval_center_parabolic(r_in, r_out, model.t[k_in], model.t[k_out])
                .max(1.0);
        p_c.n2 = 0.0;
        p_c.v_dziem = 0.0;
        p_c.v_g = 0.0;
        p_c.a_star = 0.0;
        p_c.u_dziem = 3.0;
        p_c.c_1 = (rho_mean / p_c.rho).max(1.0e-30);
        pts.push(p_c);
    }
    for &(k_a, k_b) in segs.iter().rev() {
        for k in (k_a..=k_b).rev() {
            if k == 0 && !opts.keep_surface_point {
                continue;
            }
            pts.push(make_pulse_point_on_segment(
                model,
                k,
                k_a,
                k_b,
                r_tot,
                m_tot,
                opts.interpolate_rho,
            ));
        }
        if k_a > 0 {
            let &(ka_o, kb_o) = segs
                .iter()
                .find(|&&(_, kb)| kb + 1 == k_a)
                .unwrap_or(&(0, k_a - 1));
            pts.push(make_pulse_point_on_segment(
                model,
                k_a,
                ka_o,
                kb_o,
                r_tot,
                m_tot,
                opts.interpolate_rho,
            ));
        }
    }
    if opts.add_atmosphere {
        pts.extend(build_pulse_atmosphere_points(model, r_tot, m_tot, 8));
    }
    pts
}

/// Builds the center-to-surface sequence of [`PulsePoint`] records for `model`.
#[must_use]
pub fn build_pulse_points(
    model: &StarModel,
    add_center_point: bool,
    add_double_points: bool,
    grad_mu_threshold: f64,
) -> Vec<PulsePoint> {
    let opts = PulseExportOptions {
        add_center_point,
        add_double_points,
        grad_mu_threshold,
        ..PulseExportOptions::default()
    };
    build_pulse_points_with_options(model, &opts)
}
