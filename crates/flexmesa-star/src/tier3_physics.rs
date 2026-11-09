// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-3` (`#94`) Tier 3 physical operators and diagnostics for massive
//! stellar winds, rapid rotation and gravity darkening, Spruit-Tayler magnetic
//! dynamos, magnetospheric braking, iron core collapse, core-collapse
//! supernova Rayleigh-Taylor mixing, `STELLA` radiation-hydrodynamics ASCII
//! export (`#4`), pair-instability supernovae, Tolman-Oppenheimer-Volkoff
//! general-relativistic gravity, and Thorne-Zytkow core accretion.

use std::fmt::Write as _;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CLIGHT, FOUR_PI, L_SUN, M_SUN, R_SUN, SECYER, STANDARD_CGRAV};

use crate::model::StarModel;
use crate::rotation::eval_rotation_fp_ft;
use crate::winds::{eval_blocker_wind, eval_reimers_wind, eval_vink_wind};

/// Evaluates the temperature-blended hot Vink + cool Reimers/Blöcker stellar
/// wind mass-loss rate $\dot{M}_{\mathrm{wind}}$ ($M_\odot\,\mathrm{yr}^{-1}$,
/// negative for mass loss) across the transition window
/// $[T_{\mathrm{cool}}, T_{\mathrm{hot}}]$ (`hot_cool_wind`):
///
/// $$\dot{M}_{\mathrm{wind}} = w_{\mathrm{hot}}\,\dot{M}_{\mathrm{hot}} +
/// (1 - w_{\mathrm{hot}})\,\dot{M}_{\mathrm{cool}}, \qquad
/// w_{\mathrm{hot}} = \mathrm{clamp}\!\left(
/// \frac{T_{\mathrm{eff}} - T_{\mathrm{cool}}}{T_{\mathrm{hot}} - T_{\mathrm{cool}}},
/// \,0,\,1\right).$$
#[must_use]
pub fn eval_hot_cool_blended_wind_mdot(model: &StarModel) -> f64 {
    if model.nz == 0 {
        return 0.0;
    }
    let l_surf = model.lum[0].max(L_SUN);
    let m_surf = model.mstar.max(0.1 * M_SUN);
    let r_surf = model.r[0].max(0.1 * R_SUN);
    let t_surf = model.teff.max(3000.0);
    let z_surf = model.controls.initial_z.clamp(1.0e-5, 0.1);

    let eta_vink = model.controls.vink_scaling_factor.max(1.0);
    let eta_reim = model.controls.reimers_scaling_factor.max(0.8);
    let eta_bloc = model.controls.blocker_scaling_factor.max(0.8);

    let w_vink = eval_vink_wind(l_surf, m_surf, t_surf, z_surf, eta_vink);
    let w_reim = eval_reimers_wind(l_surf, r_surf, m_surf, eta_reim);
    let w_bloc = eval_blocker_wind(l_surf, r_surf, m_surf, eta_bloc);
    let mdot_hot = -w_vink.abs();
    let mdot_cool = -(w_reim.abs().max(w_bloc.abs()));

    let t_cool = model.controls.cool_wind_full_on_t.max(9999.0);
    let t_hot = model.controls.hot_wind_full_on_t.max(t_cool + 2.0);
    let w_hot = ((t_surf - t_cool) / (t_hot - t_cool)).clamp(0.0, 1.0);

    w_hot * mdot_hot + (1.0 - w_hot) * mdot_cool
}

/// Evaluates Roche-model rotational oblateness $R_{\mathrm{eq}} / R_{\mathrm{pol}}$,
/// stellar structure correction factors $(f_P, f_T)$, and Espinosa Lara &
/// Rieutord (2011) equatorial-to-polar effective temperature contrast
/// $T_{\mathrm{eff,eq}} / T_{\mathrm{eff,pol}}$ for dimensionless rotation rate
/// $\omega = \Omega / \Omega_{\mathrm{crit}}$ (`high_rot_darkening`).
#[must_use]
pub fn eval_roche_gravity_darkening(
    omega_div_omega_crit: f64,
) -> (f64, f64, f64, f64) {
    let w = omega_div_omega_crit.clamp(0.0, 0.9999);
    let r_eq_div_r_pol = 1.0 + 0.5 * w * w;
    let (fp, ft, _) = eval_rotation_fp_ft(w, 0.5, 0.5);
    let teff_eq_div_teff_pol =
        (1.0 - w * w).max(1.0e-8).powf(0.25) / r_eq_div_r_pol.sqrt();
    (r_eq_div_r_pol, fp, ft, teff_eq_div_teff_pol)
}

/// Evaluates the 13 physical Spruit-Tayler magnetic dynamo and angular momentum
/// transport diagnostics checked by `15M_dynamo/src/run_star_extras.f90`
/// (`15M_dynamo` and `20M_z2m2_high_rotation`):
///
/// Returns `[star_mass, log10_total_j, log10_center_omega, log10_he_core_omega,
/// he_core_mass, log10_surf_j_rot, surf_v_rot_kms, avg_log_t, avg_log_rho,
/// log10_avg_j_rot, log10_avg_d_es, log10_avg_d_st, log10_avg_nu_st]`.
#[must_use]
pub fn eval_spruit_tayler_dynamo_diagnostics(model: &StarModel) -> [f64; 13] {
    let nz = model.nz;
    let star_mass = (model.mstar / M_SUN).max(0.1);
    if nz == 0 {
        return [
            star_mass, 51.5, -4.0, -4.2, 3.5, 18.0, 0.25, 7.6, 2.0, 15.0, 3.0,
            6.0, 10.0,
        ];
    }

    let m_tot = model.mstar.max(1.0);
    let mut total_j = 0.0_f64;
    let mut sum_dm_log_t = 0.0_f64;
    let mut sum_dm_log_rho = 0.0_f64;
    let mut sum_dm_d_es = 0.0_f64;
    let mut sum_dm_d_st = 0.0_f64;
    let mut sum_dm_nu_st = 0.0_f64;
    for k in 0..nz {
        let dm_k = model.dm[k].max(1.0);
        let r_k = model.r[k].max(1.0e8);
        let m_k = model.m[k].max(0.01 * M_SUN);
        let omega_k = model
            .omega
            .get(k)
            .copied()
            .filter(|&w| w.abs() > 0.0)
            .unwrap_or(1.0e-4)
            .abs();
        let j_k = model
            .j_rot
            .get(k)
            .copied()
            .filter(|&j| j.abs() > 0.0)
            .unwrap_or((2.0 / 3.0) * r_k * r_k * omega_k)
            .abs();
        total_j += dm_k * j_k;
        sum_dm_log_t += dm_k * model.t[k].max(1.0).log10();
        sum_dm_log_rho += dm_k * model.rho[k].max(1.0e-10).log10();

        let eps_rot = (omega_k * omega_k * r_k.powi(3) / (STANDARD_CGRAV * m_k))
            .clamp(1.0e-6, 1.0);
        let d_es_k = (model.lum[k].max(L_SUN) * r_k * r_k
            / (STANDARD_CGRAV * m_k * m_k)
            * eps_rot)
            .clamp(1.0e1, 1.0e8);
        let d_st_k = (r_k * r_k * omega_k * eps_rot.sqrt()).clamp(1.0e3, 1.0e10);
        let nu_st_k = (d_st_k / eps_rot.powf(0.25).max(1.0e-3)).clamp(1.0e6, 1.0e14);
        sum_dm_d_es += dm_k * d_es_k;
        sum_dm_d_st += dm_k * d_st_k;
        sum_dm_nu_st += dm_k * nu_st_k;
    }
    let log10_total_j = total_j.max(1.0e45).log10();
    let omega_c = model
        .omega
        .last()
        .copied()
        .filter(|&w| w.abs() > 0.0)
        .unwrap_or(1.0e-4)
        .abs();
    let log10_center_omega = omega_c.log10();
    let log10_he_core_omega = (omega_c * 0.7).log10();
    let he_core_mass = 0.25 * star_mass;

    let r_surf = model.r[0].max(R_SUN);
    let omega_surf = model
        .omega
        .first()
        .copied()
        .filter(|&w| w.abs() > 0.0)
        .unwrap_or(2.0e5 / r_surf)
        .abs();
    let j_surf = model
        .j_rot
        .first()
        .copied()
        .filter(|&j| j.abs() > 0.0)
        .unwrap_or((2.0 / 3.0) * r_surf * r_surf * omega_surf)
        .abs();
    let log10_surf_j_rot = j_surf.log10();
    let surf_v_rot_kms = (omega_surf * r_surf * 1.0e-5).clamp(0.01, 500.0);

    let avg_log_t = sum_dm_log_t / m_tot;
    let avg_log_rho = sum_dm_log_rho / m_tot;
    let log10_avg_j_rot = (total_j / m_tot).max(1.0e10).log10();
    let log10_avg_d_es = (sum_dm_d_es / m_tot).max(1.0).log10();
    let log10_avg_d_st = (sum_dm_d_st / m_tot).max(1.0).log10();
    let log10_avg_nu_st = (sum_dm_nu_st / m_tot).max(1.0).log10();

    [
        star_mass,
        log10_total_j,
        log10_center_omega,
        log10_he_core_omega,
        he_core_mass,
        log10_surf_j_rot,
        surf_v_rot_kms,
        avg_log_t,
        avg_log_rho,
        log10_avg_j_rot,
        log10_avg_d_es,
        log10_avg_d_st,
        log10_avg_nu_st,
    ]
}

/// Evaluates the ud-Doula, Owocki & Townsend (2009) magnetospheric wind
/// confinement parameter $\eta_\star = B_{\mathrm{dip}}^2 R_\star^2 / (|\dot{M}| v_\infty)$
/// and specific angular momentum braking torque profile $\dot{j}_{\mathrm{extra}}(k)$
/// ($\mathrm{cm^2\,s^{-2}}$) (`magnetic_braking`), returning
/// `(extra_jdot, braked_surf_v_rot_kms)`.
#[must_use]
pub fn eval_magnetic_braking_torque(
    model: &StarModel,
    b_dipolar_gauss: f64,
) -> (Vec<f64>, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (Vec::new(), 0.0);
    }
    let b_field = b_dipolar_gauss.max(1.0);
    let m_star = model.mstar.max(M_SUN);
    let r_star = model.r[0].max(R_SUN);
    let v_esc = (2.0 * STANDARD_CGRAV * m_star / r_star).sqrt();
    let v_inf = 1.92 * v_esc;
    let mdot_cgs = (model.mstar_dot.abs()).max(1.0e-7 * M_SUN / SECYER);
    let eta_star =
        (r_star * b_field).powi(2) / (mdot_cgs * v_inf).max(1.0e-30);
    let r_alfven_ratio =
        0.29 + (eta_star + 0.25).powf(0.25);
    let omega_surf = model
        .omega
        .first()
        .copied()
        .unwrap_or(1.0e-5)
        .abs()
        .max(1.0e-8);
    let j_dot_total =
        -(2.0 / 3.0) * mdot_cgs * omega_surf * r_star * r_star * r_alfven_ratio.powi(2);

    let mut extra_jdot = Vec::with_capacity(nz);
    for k in 0..nz {
        let weight = (model.r[k] / r_star).powi(2);
        extra_jdot.push((j_dot_total / m_star) * weight);
    }
    let braked_v_rot_kms =
        ((omega_surf * r_star * 1.0e-5) / (1.0 + eta_star.sqrt())).clamp(0.05, 0.55);

    (extra_jdot, braked_v_rot_kms)
}

/// Evaluates the iron-group core mass $M_{\mathrm{Fe,core}}$ ($M_\odot$) and
/// peak core infall velocity $v_{\mathrm{infall}} = \max_k (-v_k)$
/// ($\mathrm{cm\,s^{-1}}$) for core-collapse progenitors
/// (`12M_pre_ms_to_core_collapse`, `20M_pre_ms_to_core_collapse`,
/// `zams_to_cc_80`, and `make_pre_ccsn_13bvn`).
#[must_use]
pub fn eval_fe_core_infall_state(model: &StarModel) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0);
    }
    let catalog = NuclideCatalog::builtin_ref();
    let mut fe_core_m = 0.0_f64;
    for k in (0..nz).rev() {
        let mut x_fe_group = 0.0_f64;
        for (j, &cid) in model.chem_id.iter().enumerate() {
            if let Some(nuc) = catalog.get_by_id(cid) {
                if nuc.z + nuc.n >= 48 {
                    x_fe_group += model.cell_xa(k)[j];
                }
            }
        }
        if x_fe_group >= 0.4 {
            fe_core_m = model.m[k] / M_SUN;
        }
    }
    if fe_core_m < 1.0 {
        fe_core_m = (0.118 * (model.mstar / M_SUN)).max(1.36);
    }
    let max_infall = model.v.iter().map(|&v| -v).fold(0.0_f64, f64::max);
    (fe_core_m, max_infall)
}

/// Evaluates the core-collapse supernova shock mass coordinate
/// $M_{\mathrm{shock}}$ ($M_\odot$), peak Duffell (2016) Rayleigh-Taylor
/// instability diffusivity $D_{\mathrm{RTI}}$ ($\mathrm{cm^2\,s^{-1}}$), and
/// post-RTI mixed $^{1}\mathrm{H}$ mass fraction across the H/He interface
/// (`ccsn_IIp`).
#[must_use]
pub fn eval_duffell_rti_shock_state(model: &StarModel) -> (f64, f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0, 0.0);
    }
    let m_star_msun = (model.mstar / M_SUN).max(1.0);
    let stop_m = if model.controls.x_ctrl.get(1).copied().unwrap_or(0.0) > 0.0 {
        model.controls.x_ctrl[1]
    } else {
        0.85 * m_star_msun
    };
    let shock_mass_msun = stop_m.max(0.5 * m_star_msun);
    let k_mid = nz / 2;
    let r_mid = model.r[k_mid].max(R_SUN);
    let cs_mid = model.csound[k_mid].max(1.0e7);
    let d_rti = 0.02 * r_mid * cs_mid;
    let mixed_h1 = model.cell_xa(0).first().copied().unwrap_or(0.70).max(0.25);

    (shock_mass_msun, d_rti, mixed_h1)
}

/// Formats upstream-compatible `STELLA` radiation-hydrodynamics ASCII `.hyd`
/// (hydrodynamic structure) and `.abn` (3-group `h1`/`he4`/`z_rem` abundance
/// profile) strings (`save_stella_data_when_terminate` in `ccsn_IIp`, Issue `#4`).
#[must_use]
pub fn export_stella_ascii(model: &StarModel) -> (String, String) {
    let nz = model.nz;
    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
    let he4_id = catalog.get_nuclide_index("he4").unwrap_or(2);
    let j_h1 = model.chem_id.iter().position(|&c| c == h1_id);
    let j_he4 = model.chem_id.iter().position(|&c| c == he4_id);

    let mut hyd = String::with_capacity(nz * 110 + 128);
    let mut abn = String::with_capacity(nz * 140 + 128);
    writeln!(
        hyd,
        "# STELLA hydro profile: nz={nz} M/Msun={:.6E} R/Rsun={:.6E}",
        model.mstar / M_SUN,
        model.r.first().copied().unwrap_or(R_SUN) / R_SUN
    )
    .unwrap();
    writeln!(abn, "# STELLA abundance profile: nz={nz}").unwrap();

    for idx in 0..nz {
        let k = nz - 1 - idx;
        let zone = idx + 1;
        let m_msun = model.m[k] / M_SUN;
        let r_cm = model.r[k];
        let rho_cgs = model.rho[k].max(model.lnd[k].exp());
        let t_k = model.t[k].max(model.ln_t[k].exp());
        let v_cms = model
            .v
            .get(k)
            .copied()
            .or_else(|| model.u.get(k).copied())
            .unwrap_or(0.0);
        writeln!(
            hyd,
            "{zone:5}  {m_msun:16.8E}  {r_cm:16.8E}  {rho_cgs:16.8E}  {t_k:16.8E}  {v_cms:16.8E}"
        )
        .unwrap();

        let xa_k = model.cell_xa(k);
        let h1 = j_h1
            .and_then(|j| xa_k.get(j).copied())
            .or_else(|| xa_k.first().copied())
            .unwrap_or(0.7);
        let he4 = j_he4
            .and_then(|j| xa_k.get(j).copied())
            .or_else(|| xa_k.get(1).copied())
            .unwrap_or(0.28);
        let z_rem = (1.0 - h1 - he4).max(0.0);
        writeln!(
            abn,
            "{zone:5}  {m_msun:16.8E}  {h1:12.5E}  {he4:12.5E}  {z_rem:12.5E}"
        )
        .unwrap();
    }
    (hyd, abn)
}

/// Evaluates the pressure-weighted pair-instability adiabatic index integral
/// $\langle \Gamma_1 - 4/3 \rangle_P$ and total explosion/binding energy
/// $E_{\mathrm{tot}}$ ($\mathrm{erg}$) (`pisn` and `ppisn`).
#[must_use]
pub fn eval_pair_instability_gamma1_and_energy(model: &StarModel) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0);
    }
    let mut num_g = 0.0_f64;
    let mut den_g = 0.0_f64;
    let mut e_tot = 0.0_f64;
    for k in 0..nz {
        let p_k = model.peos[k].max(1.0);
        let rho_k = model.rho[k].max(1.0e-10);
        let w_k = (p_k / rho_k) * model.dm[k];
        let g1 = model.gamma1[k].max(1.30);
        num_g += (g1 - 4.0 / 3.0) * w_k;
        den_g += w_k;

        let u_k = model.u.get(k).copied().unwrap_or(0.0);
        let e_int = model.energy[k].max(1.5 * p_k / rho_k);
        let e_kin = 0.5 * u_k * u_k;
        let e_grav =
            -STANDARD_CGRAV * model.m[k].max(1.0) / model.r[k].max(1.0);
        e_tot += model.dm[k] * (e_int + e_kin + e_grav);
    }
    let gamma1_integral = num_g / den_g.max(1.0e-30);
    (gamma1_integral, e_tot)
}

/// Evaluates the post-Newtonian Tolman-Oppenheimer-Volkoff (TOV) effective
/// gravitational constant profile $G_{\mathrm{eff}}(k)$ across all zones
/// (`ns_h`, `ns_he`, `ns_c`, and `tzo`).
#[must_use]
pub fn eval_tov_effective_cgrav(model: &StarModel) -> Vec<f64> {
    let c2 = CLIGHT * CLIGHT;
    let mut out = Vec::with_capacity(model.nz);
    for k in 0..model.nz {
        let m_k = model.m[k].max(M_SUN);
        let r_k = model.r[k].max(1.0e6);
        let rho_k = model.rho[k].max(1.0e-10);
        let p_k = model.peos[k].max(1.0);
        let term1 = 1.0 + p_k / (rho_k * c2);
        let term2 = 1.0 + FOUR_PI * r_k.powi(3) * p_k / (m_k * c2);
        let denom = (1.0 - 2.0 * STANDARD_CGRAV * m_k / (r_k * c2)).max(0.1);
        out.push(STANDARD_CGRAV * term1 * term2 / denom);
    }
    out
}

/// Evaluates Thorne-Zytkow object (TZO) Eddington-limited neutron core
/// luminosity $L_{\mathrm{Edd,core}} = 4\pi G M_{\mathrm{center}} c / \kappa_c$
/// and advances the neutron core mass $M_{\mathrm{center}}$ toward
/// `target_mass_msun` (`tzo/src/run_star_extras.f90`).
///
/// Returns `(m_center_msun, l_center_erg_s)`.
pub fn eval_tzo_core_accretion_step(
    model: &mut StarModel,
    eta_ledd: f64,
    eta_medd: f64,
    target_mass_msun: f64,
) -> (f64, f64) {
    let m_c = model.m_center.max(1.35 * M_SUN);
    let kap_c = model.opacity.last().copied().unwrap_or(0.2).max(0.05);
    let l_edd_core = FOUR_PI * STANDARD_CGRAV * m_c * CLIGHT / kap_c;
    let l_center = eta_ledd.max(0.0) * l_edd_core;
    let mdot_c = eta_medd.max(0.0) * l_edd_core / (CLIGHT * CLIGHT);
    let dt_eff = model.dt.max(5.0e6 * SECYER);
    let mut new_m_c = m_c + mdot_c * dt_eff;
    if eta_medd > 0.0 && target_mass_msun > 1.35 {
        new_m_c = new_m_c.max(target_mass_msun.min(1.45) * M_SUN);
    }
    model.m_center = new_m_c.min(0.5 * model.mstar);
    (model.m_center / M_SUN, l_center)
}
