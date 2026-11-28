// SPDX-License-Identifier: GPL-3.0-or-later
//! Shellular rotation distortion polynomials (`hydro_rotation.f90`), local
//! rotational mixing instabilities (`rotation_mix_info.f90`), Spruit-Tayler
//! (2002) dynamo (`magnetic_diffusion.f90`), and post-remesh reconstruction.

use flexmesa_const::{AMU, CLIGHT, CRAD, KERG, PI, QE};

use crate::controls::StarError;
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

/// Evaluates Roche distortion factors $(f_P, f_T, i_{\mathrm{rot}}/r^2)$ (`hydro_rotation.f90`).
#[must_use]
pub fn eval_rotation_fp_ft(
    w_div_w_crit: f64,
    fp_min: f64,
    ft_min: f64,
) -> (f64, f64, f64) {
    let w = w_div_w_crit.abs().min(0.9999);
    let (w2, w4) = (w * w, flexmesa_math::pow4(w));
    let w6 = w2 * w4;
    let a_w = (1.0 + 0.3293 * w4
        - 0.4926 * w6
        - 0.5560
            * flexmesa_math::log(
                (1.0 - flexmesa_math::pow(w, 5.626)).max(1.0e-12),
            ))
    .max(1.0);
    let fp = ((1.0 - (2.0 / 3.0) * w2 - 0.2133 * w4 - 0.1068 * w6) / a_w)
        .clamp(fp_min, 1.0);
    let ft = ((1.0 - 0.07955 * w4 - 0.2322 * w6) / a_w).clamp(ft_min, 1.0);
    let i_rot_over_r2 =
        (2.0 / 3.0) * (1.0 + 0.1325 * w2 + 0.0480 * w4) / (1.0 + 0.0450 * w2);
    (fp, ft, i_rot_over_r2)
}

/// Evaluates local critical Roche angular velocity $\Omega_{\mathrm{crit}}$ ($\mathrm{rad\,s^{-1}}$).
#[must_use]
pub fn eval_omega_crit(
    cgrav: f64,
    m_grav: f64,
    r_polar: f64,
    lum: f64,
    opacity: f64,
    max_gamma_edd: f64,
) -> f64 {
    let (m_eff, r_eff) = (m_grav.max(1.0e20), r_polar.max(1.0e5));
    let l_edd = (4.0 * PI * CLIGHT * cgrav * m_eff) / opacity.max(1.0e-10);
    let gamma_edd = (lum.max(0.0) / l_edd.max(1.0)).clamp(0.0, max_gamma_edd);
    ((1.0 - gamma_edd) * cgrav * m_eff / flexmesa_math::pow3(r_eff))
        .max(0.0)
        .sqrt()
}

/// Evaluates thermal diffusivity $K = 4 a c T^3 / (3 \kappa \rho^2 C_P)$ ($\mathrm{cm^2\,s^{-1}}$).
#[must_use]
#[inline]
pub fn eval_thermal_diffusivity(
    t: f64,
    rho: f64,
    opacity: f64,
    cp: f64,
) -> f64 {
    (4.0 * CRAD * CLIGHT * flexmesa_math::pow3(t.max(10.0)))
        / (3.0
            * opacity.max(1.0e-10)
            * flexmesa_math::pow2(rho.max(1.0e-20))
            * cp.max(1.0))
}

/// Evaluates Spitzer + Nandkumar & Pethick (1984) magnetic diffusivity ($\mathrm{cm^2\,s^{-1}}$).
#[must_use]
pub fn eval_magnetic_diffusivity(
    t: f64,
    rho: f64,
    zbar: f64,
    abar: f64,
) -> f64 {
    let (kt, e2, z, a) =
        (KERG * t.max(100.0), QE * QE, zbar.max(1.0), abar.max(1.0));
    let n_e = (rho.max(1.0e-15) / (a * AMU) * z).max(1.0e-10);
    let kt_eff = eval_effective_kt(n_e, kt);
    let lam_d = (kt_eff / (4.0 * PI * e2 * n_e * (1.0 + z))).sqrt();
    let ln_lam =
        flexmesa_math::log((3.0 * kt_eff * lam_d / (z * e2)).max(1.5)).max(0.5);
    (7.0e11 * z * ln_lam * flexmesa_math::pow((kt_eff / KERG).max(100.0), -1.5))
        .clamp(1.0e-6, 1.0e16)
}

pub(crate) fn eval_spruit_tayler_dynamo(
    r: f64,
    rho: f64,
    t: f64,
    cp: f64,
    opacity: f64,
    brunt_n2_t: f64,
    brunt_n2_mu: f64,
    omega: f64,
    q_shear_raw: f64,
    abar: f64,
    zbar: f64,
    cap_d: f64,
) -> (f64, f64, f64, f64) {
    let om = omega.abs();
    if om <= 1.0e-20
        || r <= 1.0e5
        || rho <= 1.0e-20
        || (brunt_n2_t + brunt_n2_mu) <= 0.0
    {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let q = q_shear_raw.abs().clamp(1.0e-30, 1.0e30);
    let k_th = eval_thermal_diffusivity(t, rho, opacity, cp).max(1.0e-30);
    let eta_m = eval_magnetic_diffusivity(t, rho, zbar, abar);
    let s4pd = (4.0 * PI * rho).sqrt();

    let (mut s0, mut eta0, mut q0, mut bphi0, mut br0) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    if brunt_n2_mu > 0.0 {
        let n_mu = brunt_n2_mu.sqrt();
        let wn = om / n_mu;
        q0 = flexmesa_math::pow(n_mu / om, 1.75)
            * flexmesa_math::pow((eta_m / (r * r * n_mu)).max(1.0e-99), 0.25);
        s0 = rho
            * om
            * om
            * r
            * r
            * flexmesa_math::pow3(q)
            * flexmesa_math::pow4(wn);
        eta0 = r * r * om * flexmesa_math::pow4(q) * flexmesa_math::pow6(wn);
        bphi0 = (q * wn * om) * s4pd * r;
        br0 = bphi0 * q * wn * wn;
    }

    let (mut s1, mut eta1, mut q1, mut bphi1, mut br1) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    if brunt_n2_t > 0.0 {
        let nt = brunt_n2_t.sqrt();
        let wn = om / nt;
        let kr2n = (k_th / (r * r * nt)).max(1.0e-99);
        q1 = if wn > 1.0e-42 {
            flexmesa_math::pow(nt / om, 1.75)
                * flexmesa_math::pow((eta_m / (r * r * nt)).max(1.0e-99), 0.25)
                * flexmesa_math::pow(eta_m / k_th, 0.75)
        } else {
            0.0
        };
        let s1a = rho * flexmesa_math::pow2(om * r) * q * (wn * kr2n).sqrt();
        let s1b = rho
            * flexmesa_math::pow2(om * r)
            * flexmesa_math::pow3(q)
            * flexmesa_math::pow4(wn);
        s1 = s1a.max(s1b);
        let eta1a = r * r * om * q * flexmesa_math::pow(wn * kr2n, 0.75);
        let eta1b =
            r * r * om * flexmesa_math::pow4(q) * flexmesa_math::pow6(wn);
        eta1 = eta1a.max(eta1b);
        bphi1 = q.sqrt() * om * flexmesa_math::pow(wn * kr2n, 0.125) * s4pd * r;
        br1 = bphi1 * flexmesa_math::pow(wn * kr2n, 0.25);
    }

    let (sm, etam, qm, b_phi, b_r) =
        if brunt_n2_t > 0.0 && brunt_n2_mu > 0.0 && (s0 + s1) > 0.0 {
            let et = if eta0 + eta1 > 0.0 {
                eta0 * eta1 / (eta0 + eta1)
            } else {
                0.0
            };
            let w0 = s1 / (s0 + s1);
            (
                s0 * w0,
                et,
                q0 + q1,
                w0 * bphi0 + (1.0 - w0) * bphi1,
                w0 * br0 + (1.0 - w0) * br1,
            )
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
    (
        d_st,
        nu_st,
        (f_b * b_phi).clamp(0.0, 1.0e12),
        (f_b * b_r).clamp(0.0, 1.0e12),
    )
}

/// Evaluates hydrodynamic rotational instabilities and Spruit-Tayler dynamo at one face.
#[must_use]
#[allow(clippy::too_many_arguments)]
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
    let shear_rate = om * q_shear;
    let d_dsi = if shear_rate > 0.0 && n2_tot >= 0.0 {
        let ri = n2_tot / (shear_rate * shear_rate).max(1.0e-30);
        if ri < RI_CRIT_DSI {
            (hp * hp / tau_dyn) * flexmesa_math::pow2(1.0 - ri / RI_CRIT_DSI)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let sh_crit = n2_tot + 4.0 * om * om * (1.0 + 0.5 * dln_omega_dln_r);
    let d_sh = if sh_crit < 0.0 {
        hp * hp * (-sh_crit).sqrt().min(1.0 / tau_dyn)
    } else {
        0.0
    };
    let n2_eff_ssi = (brunt_n2_t / 6.0).max(0.0) + brunt_n2_mu.max(0.0);
    let d_ssi = if shear_rate > 0.0 && n2_eff_ssi > 0.0 {
        let ri_sec = n2_eff_ssi / (shear_rate * shear_rate);
        if ri_sec < RI_CRIT_DSI {
            k_therm * (RI_CRIT_DSI / ri_sec)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let d_es = (k_therm / (grada - gradt).abs().max(1.0e-4)
        * (om * om * r / grav.max(1.0)).min(0.9))
    .max(0.0);
    let d_gsf = if q_shear > 1.0e-6 && brunt_n2_mu <= 1.0e-6 * brunt_n2_t.abs()
    {
        k_therm * (om * om * r / grav.max(1.0)) * q_shear * (r / hp).min(10.0)
    } else {
        0.0
    };
    let (d_st, nu_st, b_phi, b_r) = eval_spruit_tayler_dynamo(
        r,
        rho,
        t,
        cp,
        opacity,
        brunt_n2_t,
        brunt_n2_mu,
        omega,
        q_shear,
        abar,
        zbar,
        1.0e18,
    );
    RotationalMixZone {
        d_dsi: d_dsi.clamp(0.0, 1.0e18),
        d_sh: d_sh.clamp(0.0, 1.0e18),
        d_ssi: d_ssi.clamp(0.0, 1.0e18),
        d_es: d_es.clamp(0.0, 1.0e18),
        d_gsf: d_gsf.clamp(0.0, 1.0e18),
        d_st,
        nu_st,
        b_phi,
        b_r,
    }
}

/// Bootstraps `j_rot` from `(ln_r, omega)` if `j_rot == 0 && omega != 0`
/// (`use_xh_to_update_i_rot_and_j_rot`) and recomputes
/// `(w_div_wc, i_rot, omega, fp_rot, ft_rot)` preserving shell `j_rot`
/// (`set_rotation_info` in `star/private/hydro_rotation.f90:382-397, 537-586`).
pub fn conserve_shell_j_rot_and_distortion(model: &mut StarModel) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 {
        return;
    }
    if model.j_rot.len() != nz {
        model.j_rot.resize(nz, 0.0);
    }
    if model.j_rot.iter().all(|&j| j == 0.0)
        && model.omega.iter().any(|&w| w != 0.0)
    {
        crate::hydro_rotation::use_xh_to_update_i_rot_and_j_rot(model);
    }
    crate::hydro_rotation::set_rotation_info(model, false);
}

/// Augments `model.d_mix` with Heger et al. (2000) and Spruit (2002) diffusivities.
///
/// # Errors
/// Propagates `update_rotation_mixing_info` errors.
pub fn augment_d_mix_with_rotation(
    model: &mut StarModel,
) -> Result<(), StarError> {
    crate::rotation_mix::augment_d_mix_with_rotation(model)
}

/// Updates post-Henyey $\Omega_k = j_{\mathrm{rot},k} / i_{\mathrm{rot},k}$ and $\Omega$ transport.
///
/// # Errors
/// Propagates `update_rotation_mixing_info` and `solve_omega_mix` failures,
/// including a requested retry or termination.
pub fn sync_post_henyey_rotation(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    crate::rotation_mix::sync_post_henyey_rotation(model, dt)
}

#[cfg(test)]
mod tests {
    use super::conserve_shell_j_rot_and_distortion;
    use crate::controls::{StarControls, StarJobControls};
    use crate::hydro_rotation::use_xh_to_update_i_rot_and_j_rot;
    use crate::zams::load_zams_model;

    #[test]
    fn test_conserve_shell_j_rot_bootstraps_from_exp_ln_r_not_r() {
        let ctrl = StarControls {
            initial_mass: 1.0,
            rotation_flag: true,
            zbase: 0.02,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("1 Msun ZAMS");
        model.omega.fill(2.0e-4);
        model.j_rot.fill(0.0);
        for k in 0..model.nz {
            model.r[k] = 1.05 * flexmesa_math::exp(model.ln_r[k]);
        }
        let mut expected = model.clone();
        use_xh_to_update_i_rot_and_j_rot(&mut expected);

        conserve_shell_j_rot_and_distortion(&mut model);
        assert!(model.nz > 0);
        for k in 0..model.nz {
            assert_eq!(
                model.j_rot[k].to_bits(),
                expected.j_rot[k].to_bits(),
                "j_rot[{k}] must bootstrap from exp(ln_r), not model.r"
            );
        }
        assert!(model.aux.fp_rot[0] < 1.0 && model.aux.ft_rot[0] < 1.0);
    }

    #[test]
    fn test_conserve_shell_j_rot_is_noop_when_rotation_flag_false() {
        let ctrl = StarControls {
            initial_mass: 1.0,
            rotation_flag: false,
            zbase: 0.02,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("1 Msun ZAMS");
        model.omega.fill(5.0e-5);
        model.j_rot.fill(0.0);
        model.aux.fp_rot.fill(1.0);
        model.aux.ft_rot.fill(1.0);
        model.aux.i_rot.fill(0.0);
        let fp_before = model.aux.fp_rot.clone();
        let ft_before = model.aux.ft_rot.clone();
        let i_before = model.aux.i_rot.clone();
        let j_before = model.j_rot.clone();

        conserve_shell_j_rot_and_distortion(&mut model);
        assert_eq!(model.aux.fp_rot, fp_before);
        assert_eq!(model.aux.ft_rot, ft_before);
        assert_eq!(model.aux.i_rot, i_before);
        assert_eq!(model.j_rot, j_before);
    }
}
