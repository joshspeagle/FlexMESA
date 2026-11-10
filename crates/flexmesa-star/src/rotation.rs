// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shellular rotation distortion polynomials (`hydro_rotation.f90`), five
//! hydrodynamic rotational mixing instabilities (`rotation_mix_info.f90`), and
//! the Spruit-Tayler (2002) magnetic dynamo (`magnetic_diffusion.f90`).

use flexmesa_const::{AMU, CLIGHT, CRAD, KERG, PI, QE};

use crate::diffusion::eval_effective_kt;
use crate::hydro_vars::update_sig_from_d_mix;
use crate::model::StarModel;

/// Critical Richardson number $\mathrm{Ri}_{\mathrm{crit}} = 1/4$ for the
/// dynamical shear instability (`rotation_mix_info.f90`).
pub const RI_CRIT_DSI: f64 = 0.25;

/// Rotational instability diffusion coefficients and Spruit-Tayler magnetic
/// dynamo outputs at a single zone interface (`rotation_mix_info.f90`,
/// `magnetic_diffusion.f90`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RotationalMixZone {
    /// Dynamical shear instability diffusivity $D_{\mathrm{DSI}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_dsi: f64,
    /// Solberg-Høiland instability diffusivity $D_{\mathrm{SH}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_sh: f64,
    /// Secular shear instability diffusivity $D_{\mathrm{SSI}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_ssi: f64,
    /// Eddington-Sweet meridional circulation diffusivity $D_{\mathrm{ES}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_es: f64,
    /// Goldreich-Schubert-Fricke instability diffusivity $D_{\mathrm{GSF}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_gsf: f64,
    /// Spruit-Tayler magnetic compositional diffusivity $D_{\mathrm{ST}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_st: f64,
    /// Spruit-Tayler magnetic angular momentum viscosity $\nu_{\mathrm{ST}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub nu_st: f64,
    /// Spruit-Tayler azimuthal magnetic field $B_\phi$ in Gauss (`b_phi`).
    pub b_phi: f64,
    /// Spruit-Tayler radial magnetic field $B_r$ in Gauss (`b_r`).
    pub b_r: f64,
}

impl RotationalMixZone {
    /// Returns the total compositional diffusion coefficient from all active
    /// rotational instabilities and the Spruit-Tayler dynamo.
    #[must_use]
    #[inline]
    pub fn total_d_rot(
        &self,
        f_dsi: f64,
        f_sh: f64,
        f_ssi: f64,
        f_es: f64,
        f_gsf: f64,
        f_st: f64,
    ) -> f64 {
        f_dsi * self.d_dsi
            + f_sh * self.d_sh
            + f_ssi * self.d_ssi
            + f_es * self.d_es
            + f_gsf * self.d_gsf
            + f_st * self.d_st
    }
}

/// Evaluates the Fabry, Marchant & Sana (2022) / Paxton et al. (2019 MESA V)
/// Roche rotational distortion factors $(f_P, f_T, i_{\mathrm{rot}}/r^2)$ as a
/// function of dimensionless rotation rate
/// $w = |\Omega / \Omega_{\mathrm{crit}}|$
/// (`star/private/hydro_rotation.f90:eval_fp_ft`).
///
/// The polynomials match numerical Roche integrations up to $w \to 1$:
/// $$A(w) = 1 + 0.3293 w^4 - 0.4926 w^6 - 0.5560 \ln(1 - w^{5.626}),$$
/// $$f_P(w) = \frac{1 - \tfrac{2}{3} w^2 - 0.2133 w^4 - 0.1068 w^6}{A(w)},
///   \quad
///   f_T(w) = \frac{1 - 0.07955 w^4 - 0.2322 w^6}{A(w)}.$$
#[must_use]
pub fn eval_rotation_fp_ft(
    w_div_w_crit: f64,
    fp_min: f64,
    ft_min: f64,
) -> (f64, f64, f64) {
    let w = w_div_w_crit.abs().min(0.9999);
    let w2 = w * w;
    let w4 = w2 * w2;
    let w6 = w4 * w2;
    let one_m_w56 = (1.0 - w.powf(5.626)).max(1.0e-12);
    let a_w =
        (1.0 + 0.3293 * w4 - 0.4926 * w6 - 0.5560 * one_m_w56.ln()).max(1.0);

    let fp_raw = (1.0 - (2.0 / 3.0) * w2 - 0.2133 * w4 - 0.1068 * w6) / a_w;
    let ft_raw = (1.0 - 0.07955 * w4 - 0.2322 * w6) / a_w;
    let fp = fp_raw.clamp(fp_min, 1.0);
    let ft = ft_raw.clamp(ft_min, 1.0);

    let i_rot_over_r2 =
        (2.0 / 3.0) * (1.0 + 0.1325 * w2 + 0.0480 * w4) / (1.0 + 0.0450 * w2);
    (fp, ft, i_rot_over_r2)
}

/// Evaluates the local critical Roche angular velocity
/// $\Omega_{\mathrm{crit}} =
/// \sqrt{(1 - \Gamma_{\mathrm{Edd}}) G m_{\mathrm{grav}} / r_{\mathrm{eq}}^3}$
/// in $\mathrm{rad\,s^{-1}}$ (`hydro_rotation.f90:w_div_w_crit_roche`).
#[must_use]
pub fn eval_omega_crit(
    cgrav: f64,
    m_grav: f64,
    r_polar: f64,
    lum: f64,
    opacity: f64,
    max_gamma_edd: f64,
) -> f64 {
    let m_eff = m_grav.max(1.0e20);
    let r_eff = r_polar.max(1.0e5);
    let l_edd = (4.0 * PI * CLIGHT * cgrav * m_eff) / opacity.max(1.0e-10);
    let gamma_edd = (lum.max(0.0) / l_edd.max(1.0)).clamp(0.0, max_gamma_edd);
    ((1.0 - gamma_edd) * cgrav * m_eff / r_eff.powi(3))
        .max(0.0)
        .sqrt()
}

/// Evaluates the thermal diffusivity
/// $K = \frac{4 a c T^3}{3 \kappa \rho^2 C_P}$ in $\mathrm{cm^2\,s^{-1}}$
/// (`rotation_mix_info.f90`).
#[must_use]
#[inline]
pub fn eval_thermal_diffusivity(
    t: f64,
    rho: f64,
    opacity: f64,
    cp: f64,
) -> f64 {
    let denom =
        3.0 * opacity.max(1.0e-10) * rho.max(1.0e-20).powi(2) * cp.max(1.0);
    (4.0 * CRAD * CLIGHT * t.max(10.0).powi(3)) / denom
}

/// Evaluates the Spitzer + Nandkumar & Pethick (1984) electrical conductivity
/// $\sigma_e$ and magnetic diffusivity $\eta_m = c^2 / (4\pi \sigma_e)$ in
/// $\mathrm{cm^2\,s^{-1}}$ (`magnetic_diffusion.f90`).
#[must_use]
pub fn eval_magnetic_diffusivity(
    t: f64,
    rho: f64,
    zbar: f64,
    abar: f64,
) -> f64 {
    let kt = KERG * t.max(100.0);
    let e2 = QE * QE;
    let z = zbar.max(1.0);
    let a = abar.max(1.0);
    let n_i = rho.max(1.0e-15) / (a * AMU);
    let n_e = (n_i * z).max(1.0e-10);

    let kt_eff = eval_effective_kt(n_e, kt);
    let lam_d = (kt_eff / (4.0 * PI * e2 * n_e * (1.0 + z))).sqrt();
    let x_lam = (3.0 * kt_eff * lam_d / (z * e2)).max(1.5);
    let ln_lam = x_lam.ln().max(0.5);

    let t_eff = (kt_eff / KERG).max(100.0);
    let eta_m = 7.0e11 * z * ln_lam * t_eff.powf(-1.5);
    eta_m.clamp(1.0e-6, 1.0e16)
}

/// Evaluates all five hydrodynamic rotational instabilities (`DSI`, `SH`,
/// `SSI`, `ES`, `GSF`) and the Spruit-Tayler (2002) magnetic dynamo (`ST`) at a
/// single spherical shell interface (`rotation_mix_info.f90`,
/// `magnetic_diffusion.f90`).
#[must_use]
pub fn eval_rotational_mixing_zone(
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

    // 1. Dynamical Shear Instability (DSI; Heger et al. 2000 Eq. 31)
    let shear_rate = om * q_shear;
    let d_dsi = if shear_rate > 0.0 && n2_tot >= 0.0 {
        let ri = n2_tot / (shear_rate * shear_rate).max(1.0e-30);
        if ri < RI_CRIT_DSI {
            let f_ri = (1.0 - ri / RI_CRIT_DSI).powi(2);
            (hp * hp / tau_dyn) * f_ri
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 2. Solberg-Hoiland Instability (SH; epicyclic + buoyancy < 0)
    let kappa_epi2 = 4.0 * om * om * (1.0 + 0.5 * dln_omega_dln_r);
    let sh_crit = n2_tot + kappa_epi2;
    let d_sh = if sh_crit < 0.0 {
        let growth = (-sh_crit).sqrt();
        hp * hp * growth.min(1.0 / tau_dyn)
    } else {
        0.0
    };

    // 3. Secular Shear Instability (SSI; Endal & Sofia 1978 / Zahn 1992)
    let peclet_crit = 6.0;
    let n2_eff_ssi = (brunt_n2_t / peclet_crit).max(0.0) + brunt_n2_mu.max(0.0);
    let d_ssi = if shear_rate > 0.0 && n2_eff_ssi > 0.0 {
        let ri_sec = n2_eff_ssi / (shear_rate * shear_rate);
        if ri_sec < RI_CRIT_DSI {
            k_therm * (shear_rate * shear_rate / n2_eff_ssi) * RI_CRIT_DSI
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 4. Eddington-Sweet Meridional Circulation (ES; Kippenhahn 1974)
    let grad_sub = (grada - gradt).abs().max(1.0e-4);
    let f_cen = (om * om * r / grav.max(1.0)).min(0.9);
    let v_es = k_therm / (hp * grad_sub) * f_cen;
    let d_es = (hp * v_es).max(0.0);

    // 5. Goldreich-Schubert-Fricke Instability (GSF; Goldreich & Schubert 1967)
    let d_gsf = if q_shear > 1.0e-6 && brunt_n2_mu <= 1.0e-6 * brunt_n2_t.abs()
    {
        k_therm * (om * om * r / grav.max(1.0)) * q_shear * (r / hp).min(10.0)
    } else {
        0.0
    };

    // 6. Spruit-Tayler (2002) Magnetic Dynamo (ST; Spruit 2002 Eqs. 19-33)
    let eta_m = eval_magnetic_diffusivity(t, rho, zbar, abar);
    let n_t2 = brunt_n2_t.max(0.0);
    let n_mu2 = brunt_n2_mu.max(0.0);
    let n_eff2 = n_mu2 + (eta_m / k_therm.max(eta_m)) * n_t2;
    let mut d_st = 0.0;
    let mut nu_st = 0.0;
    let mut b_phi = 0.0;
    let mut b_r = 0.0;

    if q_shear > 1.0e-8 && n_eff2 > 1.0e-20 && om > 1.0e-15 {
        let n_eff = n_eff2.sqrt();
        let q_min =
            (n_eff / om).powf(7.0 / 4.0) * (eta_m / (r * r * n_eff)).powf(0.25);
        if q_shear > q_min {
            let f_damp = (1.0 - (q_min / q_shear).sqrt()).clamp(0.0, 1.0);
            if n_mu2 >= (eta_m / k_therm.max(eta_m)) * n_t2 && n_mu2 > 0.0 {
                let n_mu = n_mu2.sqrt();
                b_phi = (4.0 * PI * rho).sqrt()
                    * r
                    * om
                    * q_shear.sqrt()
                    * (om / n_mu);
                b_r = b_phi * (om / n_mu) * q_shear.sqrt();
                nu_st = (r * r * om / q_shear) * (om / n_mu).powi(4) * f_damp;
                d_st = (r * r * om)
                    * (om / n_mu).powi(4)
                    * (eta_m / (r * r * om)).max(1.0e-12).powf(0.25)
                    * f_damp;
            } else if n_t2 > 0.0 {
                let n_t = n_t2.sqrt();
                let eps_k = (eta_m / (r * r * n_t)).max(1.0e-20);
                b_phi = (4.0 * PI * rho).sqrt()
                    * r
                    * om
                    * (om / n_t).powf(0.5)
                    * q_shear.powf(0.5)
                    * (k_therm / (r * r * n_t)).max(1.0e-20).powf(0.125);
                b_r = b_phi * (om / n_t).powf(0.5) * eps_k.powf(0.125);
                nu_st = r
                    * r
                    * om
                    * q_shear.powf(0.5)
                    * (om / n_t).powf(0.5)
                    * (k_therm / (r * r * n_t)).max(1.0e-20).powf(0.25)
                    * f_damp;
                d_st = nu_st
                    * (eta_m / k_therm.max(eta_m)).sqrt().clamp(1.0e-4, 1.0);
            }
        }
    }

    RotationalMixZone {
        d_dsi: d_dsi.clamp(0.0, 1.0e18),
        d_sh: d_sh.clamp(0.0, 1.0e18),
        d_ssi: d_ssi.clamp(0.0, 1.0e18),
        d_es: d_es.clamp(0.0, 1.0e18),
        d_gsf: d_gsf.clamp(0.0, 1.0e18),
        d_st: d_st.clamp(0.0, 1.0e18),
        nu_st: nu_st.clamp(0.0, 1.0e18),
        b_phi: b_phi.clamp(0.0, 1.0e12),
        b_r: b_r.clamp(0.0, 1.0e12),
    }
}

/// Updates shellular rotational distortion factors (`fp_rot`, `ft_rot`) and
/// specific moment of inertia `i_rot` on `model.aux` across all faces
/// (`eval_fp_ft` in `star/private/hydro_rotation.f90:753-811`).
pub fn update_rotation_distortion_factors(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.fp_rot.len() != nz {
        model.aux.resize(nz, model.species);
    }
    if !model.controls.rotation_flag {
        model.aux.fp_rot.fill(1.0);
        model.aux.ft_rot.fill(1.0);
        for k in 0..nz {
            let r_k = model.r[k].max(1.0);
            model.aux.i_rot[k] = (2.0 / 3.0) * r_k * r_k;
        }
        return;
    }
    for k in 0..nz {
        let r_k = model.r[k].max(1.0e5);
        let m_k = model.m_grav[k].max(1.0e20);
        let om_k = model.omega[k];
        if om_k.abs() > 0.0 {
            let om_crit = eval_omega_crit(
                model.cgrav[k],
                m_k,
                r_k,
                model.lum[k],
                model.opacity[k],
                0.99,
            );
            let w = if om_crit > 0.0 {
                (om_k.abs() / om_crit).min(0.99)
            } else {
                0.0
            };
            let (fp, ft, i_rot_over_r2) = eval_rotation_fp_ft(w, 0.75, 0.95);
            model.aux.fp_rot[k] = fp;
            model.aux.ft_rot[k] = ft;
            model.aux.i_rot[k] = i_rot_over_r2 * r_k * r_k;
        } else {
            if model.aux.fp_rot[k] <= 0.0 {
                model.aux.fp_rot[k] = 1.0;
            }
            if model.aux.ft_rot[k] <= 0.0 {
                model.aux.ft_rot[k] = 1.0;
            }
            if model.aux.i_rot[k] <= 0.0 {
                model.aux.i_rot[k] = (2.0 / 3.0) * r_k * r_k;
            }
        }
        model.j_rot[k] = model.aux.i_rot[k] * om_k;
    }
}

/// Evaluates shellular rotational distortion $(f_P, f_T)$, specific angular
/// momentum $j_{\mathrm{rot},k}$, and rotational mixing instabilities across
/// all faces of `model`, updating `model.grav`, `model.gradr`, `model.j_rot`,
/// `model.d_mix`, and `model.sig` when `model.controls.rotation_flag` is
/// enabled.
pub fn set_rotation_mixing_info(
    model: &mut StarModel,
) -> Vec<RotationalMixZone> {
    let nz = model.nz;
    let mut zones = vec![RotationalMixZone::default(); nz];
    if !model.controls.rotation_flag || nz < 2 {
        return zones;
    }
    update_rotation_distortion_factors(model);

    for k in 0..nz {
        let r_k = model.r[k].max(1.0e5);
        let m_k = model.m_grav[k].max(1.0e20);
        let g_sph = model.cgrav[k] * m_k / (r_k * r_k);
        let fp = model.aux.fp_rot[k];
        model.grav[k] = fp * g_sph;

        if k > 0 && k + 1 < nz {
            let dr_ln =
                (model.ln_r[k - 1] - model.ln_r[k + 1]).abs().max(1.0e-6);
            let om_up = model.omega[k - 1].abs().max(1.0e-20);
            let om_dn = model.omega[k + 1].abs().max(1.0e-20);
            let dln_om_dln_r = (om_up.ln() - om_dn.ln()) / dr_ln;
            let n2_tot = model.brunt_n2[k];
            let stored_n2_mu = model
                .aux
                .brunt_n2_composition_term
                .get(k)
                .copied()
                .unwrap_or(0.0);
            let n2_mu = if stored_n2_mu.abs() > 0.0 {
                stored_n2_mu
            } else {
                let g_hp = model.grav[k] / model.scale_height[k].max(1.0e5);
                let delta =
                    (model.chi_t[k] / model.chi_rho[k].max(1.0e-6)).abs();
                g_hp * delta * model.brunt_b[k]
            };
            let n2_t = n2_tot - n2_mu;

            let z_mix = eval_rotational_mixing_zone(
                r_k,
                model.scale_height[k],
                model.grav[k],
                model.rho[k],
                model.t[k],
                model.cp[k],
                model.opacity[k],
                model.chi_t[k],
                model.chi_rho[k],
                model.grada[k],
                model.gradt[k],
                n2_t,
                n2_mu,
                model.omega[k],
                dln_om_dln_r,
                model.abar[k],
                model.zbar[k],
                model.mlt_d[k],
            );
            let d_rot = z_mix.total_d_rot(
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0,
            );
            model.d_mix[k] += d_rot;
            zones[k] = z_mix;
        }
    }
    update_sig_from_d_mix(model);
    zones
}
