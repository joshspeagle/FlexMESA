// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shellular rotation distortion polynomials (`hydro_rotation.f90`), local
//! rotational mixing instabilities (`rotation_mix_info.f90`), Spruit-Tayler
//! (2002) dynamo (`magnetic_diffusion.f90`), and post-remesh reconstruction.

use flexmesa_const::{AMU, CLIGHT, CRAD, KERG, PI, QE};

use crate::diffusion::eval_effective_kt;
use crate::model::StarModel;

/// Critical Richardson number $\mathrm{Ri}_{\mathrm{crit}} = 1/4$ (`rotation_mix_info.f90`).
pub const RI_CRIT_DSI: f64 = 0.25;

/// Rotational instability diffusivities and Spruit-Tayler outputs at a single face.
#[derive(Debug, Clone, Copy, Default)]
pub struct RotationalMixZone {
    /// Dynamical shear instability diffusivity $D_{\mathrm{DSI}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub d_dsi: f64,
    /// Solberg-Hoiland instability diffusivity $D_{\mathrm{SH}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub d_sh: f64,
    /// Secular shear instability diffusivity $D_{\mathrm{SSI}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub d_ssi: f64,
    /// Eddington-Sweet meridional circulation diffusivity $D_{\mathrm{ES}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub d_es: f64,
    /// Goldreich-Schubert-Fricke instability diffusivity $D_{\mathrm{GSF}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub d_gsf: f64,
    /// Spruit-Tayler magnetic compositional diffusivity $D_{\mathrm{ST}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub d_st: f64,
    /// Spruit-Tayler magnetic angular momentum viscosity $\nu_{\mathrm{ST}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub nu_st: f64,
    /// Spruit-Tayler azimuthal magnetic field $B_\phi$ in Gauss (`b_phi`).
    pub b_phi: f64,
    /// Spruit-Tayler radial magnetic field $B_r$ in Gauss (`b_r`).
    pub b_r: f64,
}

impl RotationalMixZone {
    /// Returns weighted compositional diffusivity sum across active rotational modes.
    #[must_use]
    #[inline]
    pub fn total_d_rot(
        &self, f_dsi: f64, f_sh: f64, f_ssi: f64, f_es: f64, f_gsf: f64, f_st: f64,
    ) -> f64 {
        f_dsi * self.d_dsi + f_sh * self.d_sh + f_ssi * self.d_ssi
            + f_es * self.d_es + f_gsf * self.d_gsf + f_st * self.d_st
    }
}

/// Evaluates Roche distortion factors $(f_P, f_T, i_{\mathrm{rot}}/r^2)$ (`hydro_rotation.f90`).
#[must_use]
pub fn eval_rotation_fp_ft(w_div_w_crit: f64, fp_min: f64, ft_min: f64) -> (f64, f64, f64) {
    let w = w_div_w_crit.abs().min(0.9999);
    let (w2, w4) = (w * w, w.powi(4));
    let w6 = w2 * w4;
    let a_w = (1.0 + 0.3293 * w4 - 0.4926 * w6 - 0.5560 * (1.0 - w.powf(5.626)).max(1.0e-12).ln()).max(1.0);
    let fp = ((1.0 - (2.0 / 3.0) * w2 - 0.2133 * w4 - 0.1068 * w6) / a_w).clamp(fp_min, 1.0);
    let ft = ((1.0 - 0.07955 * w4 - 0.2322 * w6) / a_w).clamp(ft_min, 1.0);
    let i_rot_over_r2 = (2.0 / 3.0) * (1.0 + 0.1325 * w2 + 0.0480 * w4) / (1.0 + 0.0450 * w2);
    (fp, ft, i_rot_over_r2)
}

/// Evaluates local critical Roche angular velocity $\Omega_{\mathrm{crit}}$ ($\mathrm{rad\,s^{-1}}$).
#[must_use]
pub fn eval_omega_crit(
    cgrav: f64, m_grav: f64, r_polar: f64, lum: f64, opacity: f64, max_gamma_edd: f64,
) -> f64 {
    let (m_eff, r_eff) = (m_grav.max(1.0e20), r_polar.max(1.0e5));
    let l_edd = (4.0 * PI * CLIGHT * cgrav * m_eff) / opacity.max(1.0e-10);
    let gamma_edd = (lum.max(0.0) / l_edd.max(1.0)).clamp(0.0, max_gamma_edd);
    ((1.0 - gamma_edd) * cgrav * m_eff / r_eff.powi(3)).max(0.0).sqrt()
}

/// Evaluates thermal diffusivity $K = 4 a c T^3 / (3 \kappa \rho^2 C_P)$ ($\mathrm{cm^2\,s^{-1}}$).
#[must_use]
#[inline]
pub fn eval_thermal_diffusivity(t: f64, rho: f64, opacity: f64, cp: f64) -> f64 {
    (4.0 * CRAD * CLIGHT * t.max(10.0).powi(3))
        / (3.0 * opacity.max(1.0e-10) * rho.max(1.0e-20).powi(2) * cp.max(1.0))
}

/// Evaluates Spitzer + Nandkumar & Pethick (1984) magnetic diffusivity ($\mathrm{cm^2\,s^{-1}}$).
#[must_use]
pub fn eval_magnetic_diffusivity(t: f64, rho: f64, zbar: f64, abar: f64) -> f64 {
    let (kt, e2, z, a) = (KERG * t.max(100.0), QE * QE, zbar.max(1.0), abar.max(1.0));
    let n_e = (rho.max(1.0e-15) / (a * AMU) * z).max(1.0e-10);
    let kt_eff = eval_effective_kt(n_e, kt);
    let lam_d = (kt_eff / (4.0 * PI * e2 * n_e * (1.0 + z))).sqrt();
    let ln_lam = (3.0 * kt_eff * lam_d / (z * e2)).max(1.5).ln().max(0.5);
    (7.0e11 * z * ln_lam * (kt_eff / KERG).max(100.0).powf(-1.5)).clamp(1.0e-6, 1.0e16)
}

pub(crate) fn eval_spruit_tayler_dynamo(
    r: f64, rho: f64, t: f64, cp: f64, opacity: f64, brunt_n2_t: f64, brunt_n2_mu: f64,
    omega: f64, q_shear_raw: f64, abar: f64, zbar: f64, cap_d: f64,
) -> (f64, f64, f64, f64) {
    let om = omega.abs();
    if om <= 1.0e-20 || r <= 1.0e5 || rho <= 1.0e-20 || (brunt_n2_t + brunt_n2_mu) <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let q = q_shear_raw.abs().clamp(1.0e-30, 1.0e30);
    let k_th = eval_thermal_diffusivity(t, rho, opacity, cp).max(1.0e-30);
    let eta_m = eval_magnetic_diffusivity(t, rho, zbar, abar);
    let s4pd = (4.0 * PI * rho).sqrt();

    let (mut s0, mut eta0, mut q0, mut bphi0, mut br0) = (0.0, 0.0, 0.0, 0.0, 0.0);
    if brunt_n2_mu > 0.0 {
        let n_mu = brunt_n2_mu.sqrt();
        let wn = om / n_mu;
        q0 = (n_mu / om).powf(1.75) * (eta_m / (r * r * n_mu)).max(1.0e-99).powf(0.25);
        s0 = rho * om * om * r * r * q.powi(3) * wn.powi(4);
        eta0 = r * r * om * q.powi(4) * wn.powi(6);
        bphi0 = (q * wn * om) * s4pd * r;
        br0 = bphi0 * q * wn * wn;
    }

    let (mut s1, mut eta1, mut q1, mut bphi1, mut br1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    if brunt_n2_t > 0.0 {
        let nt = brunt_n2_t.sqrt();
        let wn = om / nt;
        let kr2n = (k_th / (r * r * nt)).max(1.0e-99);
        q1 = if wn > 1.0e-42 {
            (nt / om).powf(1.75)
                * (eta_m / (r * r * nt)).max(1.0e-99).powf(0.25)
                * (eta_m / k_th).powf(0.75)
        } else {
            0.0
        };
        let s1a = rho * (om * r).powi(2) * q * (wn * kr2n).sqrt();
        let s1b = rho * (om * r).powi(2) * q.powi(3) * wn.powi(4);
        s1 = s1a.max(s1b);
        let eta1a = r * r * om * q * (wn * kr2n).powf(0.75);
        let eta1b = r * r * om * q.powi(4) * wn.powi(6);
        eta1 = eta1a.max(eta1b);
        bphi1 = q.sqrt() * om * (wn * kr2n).powf(0.125) * s4pd * r;
        br1 = bphi1 * (wn * kr2n).powf(0.25);
    }

    let (sm, etam, qm, b_phi, b_r) = if brunt_n2_t > 0.0 && brunt_n2_mu > 0.0 && (s0 + s1) > 0.0 {
        let et = if eta0 + eta1 > 0.0 { eta0 * eta1 / (eta0 + eta1) } else { 0.0 };
        let w0 = s1 / (s0 + s1);
        (s0 * w0, et, q0 + q1, w0 * bphi0 + (1.0 - w0) * bphi1, w0 * br0 + (1.0 - w0) * br1)
    } else if brunt_n2_t <= 0.0 && brunt_n2_mu > 0.0 {
        (s0, eta0, q0, bphi0, br0)
    } else if brunt_n2_t > 0.0 {
        (s1, eta1, q1, bphi1, br1)
    } else {
        return (0.0, 0.0, 0.0, 0.0);
    };

    let f_mag = (1.0 - (qm / q).min(1.0)).max(0.0);
    if f_mag <= 0.0 || !sm.is_finite() || !etam.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let nu_st = ((f_mag * sm) / (om * q * rho)).clamp(0.0, cap_d);
    let d_st = (f_mag * etam).min(nu_st).clamp(0.0, cap_d);
    let f_b = f_mag.sqrt();
    (d_st, nu_st, (f_b * b_phi).clamp(0.0, 1.0e12), (f_b * b_r).clamp(0.0, 1.0e12))
}

/// Evaluates hydrodynamic rotational instabilities and Spruit-Tayler dynamo at one face.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn eval_rotational_mixing_zone(
    r: f64, hp: f64, grav: f64, rho: f64, t: f64, cp: f64, opacity: f64,
    chi_t: f64, chi_rho: f64, grada: f64, gradt: f64, brunt_n2_t: f64,
    brunt_n2_mu: f64, omega: f64, dln_omega_dln_r: f64, abar: f64, zbar: f64, d_visc: f64,
) -> RotationalMixZone {
    let _ = (chi_t, chi_rho, d_visc);
    let om = omega.abs();
    if om < 1.0e-20 || r < 1.0e5 || hp < 1.0e3 {
        return RotationalMixZone::default();
    }
    let q_shear = dln_omega_dln_r.abs();
    let k_therm = eval_thermal_diffusivity(t, rho, opacity, cp);
    let n2_tot = brunt_n2_t + brunt_n2_mu;
    let tau_dyn = (r / grav.max(1.0)).sqrt().max(1.0e-6);
    let shear_rate = om * q_shear;
    let d_dsi = if shear_rate > 0.0 && n2_tot >= 0.0 {
        let ri = n2_tot / (shear_rate * shear_rate).max(1.0e-30);
        if ri < RI_CRIT_DSI { (hp * hp / tau_dyn) * (1.0 - ri / RI_CRIT_DSI).powi(2) } else { 0.0 }
    } else {
        0.0
    };
    let sh_crit = n2_tot + 4.0 * om * om * (1.0 + 0.5 * dln_omega_dln_r);
    let d_sh = if sh_crit < 0.0 { hp * hp * (-sh_crit).sqrt().min(1.0 / tau_dyn) } else { 0.0 };
    let n2_eff_ssi = (brunt_n2_t / 6.0).max(0.0) + brunt_n2_mu.max(0.0);
    let d_ssi = if shear_rate > 0.0 && n2_eff_ssi > 0.0 {
        let ri_sec = n2_eff_ssi / (shear_rate * shear_rate);
        if ri_sec < RI_CRIT_DSI { k_therm * (RI_CRIT_DSI / ri_sec) } else { 0.0 }
    } else {
        0.0
    };
    let d_es = (k_therm / (grada - gradt).abs().max(1.0e-4) * (om * om * r / grav.max(1.0)).min(0.9)).max(0.0);
    let d_gsf = if q_shear > 1.0e-6 && brunt_n2_mu <= 1.0e-6 * brunt_n2_t.abs() {
        k_therm * (om * om * r / grav.max(1.0)) * q_shear * (r / hp).min(10.0)
    } else {
        0.0
    };
    let (d_st, nu_st, b_phi, b_r) = eval_spruit_tayler_dynamo(
        r, rho, t, cp, opacity, brunt_n2_t, brunt_n2_mu, omega, q_shear, abar, zbar, 1.0e18,
    );
    RotationalMixZone {
        d_dsi: d_dsi.clamp(0.0, 1.0e18), d_sh: d_sh.clamp(0.0, 1.0e18),
        d_ssi: d_ssi.clamp(0.0, 1.0e18), d_es: d_es.clamp(0.0, 1.0e18),
        d_gsf: d_gsf.clamp(0.0, 1.0e18), d_st, nu_st, b_phi, b_r,
    }
}

/// Updates shellular distortion (`fp_rot`, `ft_rot`, `i_rot`) across faces (`hydro_rotation.f90`).
pub fn update_rotation_distortion_factors(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.fp_rot.len() != nz || model.aux.d_fp_rot_dlnr.len() != nz {
        model.aux.resize(nz, model.species);
    }
    if !model.controls.rotation_flag {
        model.aux.fp_rot.fill(1.0);
        model.aux.ft_rot.fill(1.0);
        model.aux.d_fp_rot_dlnr.fill(0.0);
        model.aux.d_ft_rot_dlnr.fill(0.0);
        for k in 0..nz {
            let ir = (2.0 / 3.0) * model.r[k].max(1.0).powi(2);
            model.aux.i_rot[k] = ir;
            model.aux.d_i_rot_dlnr[k] = 2.0 * ir;
        }
        return;
    }
    let max_w = model.controls.mix_wind.w_div_wcrit_max.clamp(0.0, 0.99);
    let tidal = model.aux.tidal_deformation.clone();
    for k in 0..nz {
        let (r_k, m_k, om_k) = (model.r[k].max(1.0e5), model.m_grav[k].max(1.0e20), model.omega[k]);
        let (fp_s, ft_s, i_rot_s) = if om_k.abs() > 0.0 {
            let om_c = eval_omega_crit(model.cgrav[k], m_k, r_k, 0.0, 0.0, 0.99);
            let w = if om_c > 0.0 { (om_k.abs() / om_c).min(max_w) } else { 0.0 };
            let (fp, ft, i_r2) = eval_rotation_fp_ft(w, 0.75, 0.95);
            (fp, ft, i_r2 * r_k * r_k)
        } else {
            let fp0 = if model.aux.fp_rot[k] > 0.0 { model.aux.fp_rot[k] } else { 1.0 };
            let ft0 = if model.aux.ft_rot[k] > 0.0 { model.aux.ft_rot[k] } else { 1.0 };
            let ir0 = if model.aux.i_rot[k] > 0.0 { model.aux.i_rot[k] } else { (2.0 / 3.0) * r_k * r_k };
            (fp0, ft0, ir0)
        };
        if let Some(ref hook) = tidal {
            let td = hook.0.eval_shell(k, r_k, om_k);
            let (f_sw, f_1m) = (td.f_switch.clamp(0.0, 1.0), 1.0 - td.f_switch.clamp(0.0, 1.0));
            model.aux.fp_rot[k] = (f_sw * td.fp + f_1m * fp_s).clamp(0.05, 2.0);
            model.aux.d_fp_rot_dlnr[k] = f_sw * td.d_fp_dlnr;
            model.aux.ft_rot[k] = (f_sw * td.ft + f_1m * ft_s).clamp(0.05, 2.0);
            model.aux.d_ft_rot_dlnr[k] = f_sw * td.d_ft_dlnr;
            model.aux.i_rot[k] = (f_sw * td.i_rot + f_1m * i_rot_s).max(1.0e-30);
            model.aux.d_i_rot_dlnr[k] = f_sw * td.d_irot_dlnr + f_1m * (2.0 * i_rot_s);
        } else {
            model.aux.fp_rot[k] = fp_s;
            model.aux.d_fp_rot_dlnr[k] = 0.0;
            model.aux.ft_rot[k] = ft_s;
            model.aux.d_ft_rot_dlnr[k] = 0.0;
            model.aux.i_rot[k] = i_rot_s;
            model.aux.d_i_rot_dlnr[k] = 2.0 * i_rot_s;
        }
        model.j_rot[k] = model.aux.i_rot[k] * om_k;
    }
}

/// Recomputes $(f_P, f_T, i_{\mathrm{rot}})$ and $\Omega_k = j_{\mathrm{rot},k} / i_{\mathrm{rot},k}$.
pub fn conserve_shell_j_rot_and_distortion(model: &mut StarModel) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 || model.j_rot.len() != nz
        || (model.j_rot.iter().all(|&j| j == 0.0) && model.omega.iter().any(|&w| w != 0.0))
    {
        update_rotation_distortion_factors(model);
        return;
    }
    let j_target = model.j_rot.clone();
    for _ in 0..25 {
        let mut max_rel = 0.0_f64;
        for k in 0..nz {
            let ir = model.aux.i_rot.get(k).copied().filter(|&v| v > 0.0)
                .unwrap_or_else(|| (2.0 / 3.0) * model.r[k].max(1.0).powi(2));
            let om_new = j_target[k] / ir;
            let scale = om_new.abs().max(model.omega[k].abs());
            if scale > 0.0 {
                max_rel = max_rel.max((om_new - model.omega[k]).abs() / scale);
            }
            model.omega[k] = om_new;
        }
        update_rotation_distortion_factors(model);
        if max_rel < 1.0e-15 {
            break;
        }
    }
    for k in 0..nz {
        model.omega[k] = j_target[k] / model.aux.i_rot[k].max(1.0e-30);
        model.j_rot[k] = j_target[k];
        let (r_k, m_k) = (model.r[k].max(1.0e5), model.m_grav[k].max(1.0e20));
        model.grav[k] = model.aux.fp_rot[k] * model.cgrav[k] * m_k / (r_k * r_k);
    }
}

/// Augments `model.d_mix` with Heger et al. (2000) and Spruit (2002) diffusivities.
pub fn augment_d_mix_with_rotation(model: &mut StarModel) {
    crate::rotation_mix::augment_d_mix_with_rotation(model);
}

/// Updates post-Henyey $\Omega_k = j_{\mathrm{rot},k} / i_{\mathrm{rot},k}$ and $\Omega$ transport.
pub fn sync_post_henyey_rotation(model: &mut StarModel, dt: f64) {
    crate::rotation_mix::sync_post_henyey_rotation(model, dt);
}

/// Evaluates shellular distortion and returns rotational mixing diagnostics.
pub fn set_rotation_mixing_info(model: &mut StarModel) -> Vec<RotationalMixZone> {
    let nz = model.nz;
    let mut zones = vec![RotationalMixZone::default(); nz];
    if !model.controls.rotation_flag || nz < 2 {
        return zones;
    }
    conserve_shell_j_rot_and_distortion(model);
    for k in 0..nz {
        let (r_k, m_k) = (model.r[k].max(1.0e5), model.m_grav[k].max(1.0e20));
        model.grav[k] = model.aux.fp_rot[k] * model.cgrav[k] * m_k / (r_k * r_k);
    }
    for k in 1..(nz - 1) {
        let r_k = model.r[k].max(1.0e5);
        let dr_ln = (model.ln_r[k - 1] - model.ln_r[k + 1]).abs().max(1.0e-6);
        let dln_om = (model.omega[k - 1].abs().max(1.0e-20).ln()
            - model.omega[k + 1].abs().max(1.0e-20).ln()) / dr_ln;
        let stored_n2_mu = model.aux.brunt_n2_composition_term.get(k).copied().unwrap_or(0.0);
        let n2_mu = if stored_n2_mu.abs() > 0.0 {
            stored_n2_mu
        } else {
            (model.grav[k] / model.scale_height[k].max(1.0e5))
                * (model.chi_t[k] / model.chi_rho[k].max(1.0e-6)).abs()
                * model.brunt_b[k]
        };
        let mut z_mix = eval_rotational_mixing_zone(
            r_k, model.scale_height[k], model.grav[k], model.rho[k], model.t[k],
            model.cp[k], model.opacity[k], model.chi_t[k], model.chi_rho[k],
            model.grada[k], model.gradt[k], model.brunt_n2[k] - n2_mu, n2_mu,
            model.omega[k], dln_om, model.abar[k], model.zbar[k], model.mlt_d[k],
        );
        z_mix.nu_st = 0.0;
        zones[k] = z_mix;
    }
    zones
}
