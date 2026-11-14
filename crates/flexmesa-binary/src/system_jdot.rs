// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled orbital angular-momentum derivatives (`do_jdot_{mb,gr,ml,missing_wind,ls}`),
//! shellular spin conservation across Henyey remeshing, post-step rotational
//! compositional mixing, and Hut (1981) / Zahn (1977) tidal spin synchronization.
//!
//! Corresponds to `binary/private/binary_jdot.f90:33-246`,
//! `binary/private/binary_tides.f90:120-456`, and
//! `binary/private/binary_edot.f90:33-182`.

use flexmesa_const::{LSUN, MSUN, MixingType, PI, RSUN};
use flexmesa_math::{exp, pow};
use flexmesa_star::{
    StarError, StarModel, apply_uniform_rotation, compute_convection_sigmas,
    set_basic_vars, set_hydro_vars, set_rotation_mixing_info,
    solve_composition_diffusion, update_rotation_distortion_factors,
};

use crate::inlist::BinaryInlistControls;
use crate::jdot_tides::{
    check_jdot_mb_conditions, eval_edot_gr, eval_edot_tidal_hut, eval_jdot_gr,
    eval_jdot_ls_with_winds, eval_jdot_mb_with_scale_mass, eval_jdot_missing_wind,
    eval_jdot_ml, eval_k_div_t_convective, eval_k_div_t_radiative, eval_tsync,
    hut_f2, hut_f5,
};

/// Component-by-component breakdown of the orbital torque $\dot{J}_{\mathrm{orb}}$
/// ($\mathrm{g\,cm^2\,s^{-2}}$) in `get_jdot` (`binary_jdot.f90:33-93`).
#[derive(Debug, Clone, Copy, Default)]
pub struct JdotBreakdown {
    /// Magnetic-braking torque $\dot{J}_{\mathrm{mb}}$.
    pub jdot_mb: f64,
    /// Gravitational-wave torque $\dot{J}_{\mathrm{gr}}$.
    pub jdot_gr: f64,
    /// Non-conservative mass-loss torque $\dot{J}_{\mathrm{ml}}$.
    pub jdot_ml: f64,
    /// Missing accretor wind spin torque $\dot{J}_{\mathrm{missing\_wind}}$.
    pub jdot_missing_wind: f64,
    /// Extra user hook torque $\dot{J}_{\mathrm{extra}}$.
    pub jdot_extra: f64,
    /// Spin-orbit tidal coupling torque $\dot{J}_{\mathrm{ls}}$ (unscaled by `jdot_multiplier`).
    pub jdot_ls: f64,
    /// Combined orbital torque $(\dot{J}_{\mathrm{mb}} + \dot{J}_{\mathrm{gr}} + \dot{J}_{\mathrm{ml}} + \dot{J}_{\mathrm{missing\_wind}} + \dot{J}_{\mathrm{extra}})\,\mu_J + \dot{J}_{\mathrm{ls}}$.
    pub jdot_total: f64,
}

/// Integrates total stellar spin angular momentum $S_* = \sum_{k=0}^{\mathrm{nz}-1} \overline{\Delta m}_k j_{\mathrm{rot},k}$
/// ($\mathrm{g\,cm^2\,s^{-1}}$).
#[must_use]
pub fn eval_star_spin_j(star: &StarModel) -> f64 {
    if !star.controls.rotation_flag || star.nz == 0 {
        return 0.0;
    }
    (0..star.nz)
        .map(|k| {
            let dm_k = star.dm_bar.get(k).copied().unwrap_or(star.dm[k]);
            let j_k = star.j_rot.get(k).copied().unwrap_or(0.0);
            dm_k * j_k
        })
        .sum()
}

/// Evaluates dimensionless squared gyration radius $r_{\mathrm{gyr}}^2 = I_* / (M_* R_*^2)$.
#[must_use]
pub fn eval_star_r_gyr_sq(star: &StarModel) -> f64 {
    if star.nz == 0 || star.mstar <= 0.0 {
        return 0.1;
    }
    let r_surf = star.r[0].max(1.0);
    let mut i_tot = 0.0_f64;
    for k in 0..star.nz {
        let dm_k = star.dm_bar.get(k).copied().unwrap_or(star.dm[k]);
        let i_k = star
            .aux
            .i_rot
            .get(k)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or((2.0 / 3.0) * star.r[k] * star.r[k]);
        i_tot += dm_k * i_k;
    }
    (i_tot / (star.mstar * r_surf * r_surf)).clamp(1.0e-5, 1.0)
}

/// Estimates outer convective envelope mass ($M_\odot$) and radial thickness ($R_\odot$)
/// for Hurley et al. (2002) convective-envelope tidal dissipation (`binary_tides.f90:260-358`).
#[must_use]
pub fn eval_convective_envelope_mass_and_size(star: &StarModel) -> (f64, f64) {
    if star.nz < 2 {
        return (0.0, 0.0);
    }
    let r_top = star.r[0];
    let mut m_env_g = 0.0_f64;
    let mut r_base = r_top;
    let mut found_conv = false;
    for k in 0..star.nz {
        let is_conv = star
            .mixing_type
            .get(k)
            .copied()
            .map_or(star.gradr[k] > star.grada[k], |mt| {
                mt == MixingType::ConvectiveMixing
            });
        if is_conv {
            found_conv = true;
            m_env_g += star.dm[k];
            r_base = star.r[k];
        } else if found_conv && star.q[k] < 0.99 {
            break;
        }
    }
    ((m_env_g / MSUN).max(0.0), ((r_top - r_base).max(0.0)) / RSUN)
}

/// Synchronizes a stellar model uniformly to orbital period `period_sec` (s) at
/// startup (`do_initial_orbit_sync_{1,2}` in `run_binary_support.f90:319-340`).
pub fn apply_initial_orbit_sync(
    star: &mut StarModel,
    period_sec: f64,
) -> Result<(), StarError> {
    if period_sec <= 0.0 {
        return Ok(());
    }
    star.controls.rotation_flag = true;
    apply_uniform_rotation(star, 2.0 * PI / period_sec);
    update_rotation_distortion_factors(star);
    Ok(())
}

/// Applies operator-split rotational chemical mixing (`set_rotation_mixing_info` +
/// `solve_composition_diffusion`) after the coupled convective/overshoot Henyey
/// solve while preserving the pre-tidal `(omega, j_rot)` profile.
pub fn apply_post_step_mixing(star: &mut StarModel, dt: f64) -> Result<(), StarError> {
    if !star.controls.rotation_flag
        || !star.controls.do_mix
        || star.controls.mix_factor <= 0.0
        || star.nz < 2
        || star.species == 0
        || dt <= 0.0
    {
        return Ok(());
    }
    let saved_omega = star.omega.clone();
    let saved_j_rot = star.j_rot.clone();
    let base_mlt_d: Vec<f64> = (0..star.nz)
        .map(|k| star.mlt_d.get(k).copied().unwrap_or(0.0))
        .collect();
    let zones = set_rotation_mixing_info(star);
    let mut has_rot_mix = false;
    for k in 0..star.nz {
        let z = &zones[k];
        let d_hydro_rot = (z.d_dsi + z.d_ssi + z.d_es + z.d_gsf) / 30.0;
        if d_hydro_rot > 0.0 {
            has_rot_mix = true;
        }
        star.d_mix[k] = base_mlt_d[k] + d_hydro_rot;
    }
    star.sig.fill(0.0);
    if has_rot_mix && star.nz >= 2 {
        star.d_mix[0] = star.d_mix[0].max(star.d_mix[1]);
        let last = star.nz - 1;
        star.d_mix[last] = star.d_mix[last].max(star.d_mix[last - 1]);
        compute_convection_sigmas(star, dt, star.controls.mix_factor, 1.0e10);
        if star.sig.iter().any(|&s| s > 0.0) {
            solve_composition_diffusion(star, dt);
            star.pack_xh();
            set_basic_vars(star)?;
            set_hydro_vars(star, false, false)?;
        }
    }
    star.omega = saved_omega;
    star.j_rot = saved_j_rot;
    Ok(())
}

/// Interpolates $\Omega(q)$ smoothly across cell centers from pre-remesh
/// `(q_mid_old, om_old)` onto a newly remeshed [`StarModel`] grid
/// (`adjust_omega` in `star/private/adjust_mesh_support.f90`), avoiding the
/// outer-face $r_0^2$ step artifact in `mesh_remap` when surface zones merge.
pub fn restore_smooth_omega_after_remesh(
    star: &mut StarModel,
    q_mid_old: &[f64],
    om_old: &[f64],
) {
    if !star.controls.rotation_flag
        || star.nz == 0
        || q_mid_old.is_empty()
        || q_mid_old.len() != om_old.len()
    {
        return;
    }
    let n_old = q_mid_old.len();
    for k in 0..star.nz {
        let qm = star.q[k] - 0.5 * star.dq[k];
        let om_interp = if qm >= q_mid_old[0] || n_old == 1 {
            om_old[0]
        } else if qm <= q_mid_old[n_old - 1] {
            om_old[n_old - 1]
        } else {
            let idx = q_mid_old
                .partition_point(|&qv| qv > qm)
                .clamp(1, n_old - 1);
            let (q_hi, q_lo) = (q_mid_old[idx - 1], q_mid_old[idx]);
            let t = ((q_hi - qm) / (q_hi - q_lo).max(1.0e-30)).clamp(0.0, 1.0);
            om_old[idx - 1] + t * (om_old[idx] - om_old[idx - 1])
        };
        star.omega[k] = om_interp;
    }
}

/// Restores exact total stellar spin angular momentum $S_{\mathrm{target}} = \max(S_{\mathrm{pre}} + \Delta S_{\dot{M}}, 0)$
/// after Henyey radius changes and remeshing (`set_hydro_vars` updates `i_rot` at old `omega`).
pub fn conserve_spin_after_henyey(star: &mut StarModel, s_pre: f64, delta_s_mdot: f64) {
    if !star.controls.rotation_flag || star.nz == 0 {
        return;
    }
    update_rotation_distortion_factors(star);
    let s_target = (s_pre + delta_s_mdot).max(0.0);
    let mut s_cur = 0.0_f64;
    for k in 0..star.nz {
        let dm_k = star.dm_bar.get(k).copied().unwrap_or(star.dm[k]);
        s_cur += dm_k * star.aux.i_rot[k] * star.omega[k];
    }
    if s_cur > 0.0 {
        let scale = s_target / s_cur;
        for k in 0..star.nz {
            star.omega[k] *= scale;
            star.j_rot[k] = star.aux.i_rot[k] * star.omega[k];
        }
    } else if s_target > 0.0 {
        let mut i_tot = 0.0_f64;
        for k in 0..star.nz {
            let dm_k = star.dm_bar.get(k).copied().unwrap_or(star.dm[k]);
            i_tot += dm_k * star.aux.i_rot[k];
        }
        if i_tot > 0.0 {
            let om = s_target / i_tot;
            for k in 0..star.nz {
                star.omega[k] = om;
                star.j_rot[k] = star.aux.i_rot[k] * om;
            }
        }
    }
}

/// Relaxes shellular spin $\Omega_k$ toward orbital synchronization over `dt` (s)
/// following `sync_spin_orbit_torque` (`binary/private/binary_tides.f90:120-210`).
#[allow(clippy::too_many_arguments)]
pub fn apply_tidal_spin_sync(
    star: &mut StarModel,
    sync_type: &str,
    sync_mode: &str,
    f_tid: f64,
    tidal_reduction: f64,
    m_companion: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
    dt: f64,
) {
    if !star.controls.rotation_flag
        || star.nz == 0
        || separation <= 0.0
        || period <= 0.0
        || dt <= 0.0
        || f_tid <= 0.0
        || sync_type.eq_ignore_ascii_case("None")
    {
        return;
    }
    update_rotation_distortion_factors(star);
    let omega_orb = 2.0 * PI / period;
    if sync_type.eq_ignore_ascii_case("Instantaneous") {
        for k in 0..star.nz {
            star.omega[k] = omega_orb;
            star.j_rot[k] = star.aux.i_rot[k] * omega_orb;
        }
        return;
    }
    let t_sync = if sync_type.eq_ignore_ascii_case("Orb_period") {
        period / f_tid
    } else {
        let k_div_t = eval_star_k_div_t(
            star, sync_type, tidal_reduction, m_companion, separation, period,
        );
        let r_gyr_sq = eval_star_r_gyr_sq(star);
        eval_tsync(
            k_div_t, m_companion / star.mstar.max(1.0), r_gyr_sq, star.r[0], separation, f_tid,
        )
    };
    if !t_sync.is_finite() || t_sync <= 0.0 {
        return;
    }
    let _ = sync_mode;
    let a1 = hut_f2(eccentricity);
    let a2 = pow((1.0 - eccentricity * eccentricity).max(1.0e-12), 1.5) * hut_f5(eccentricity);
    let relax_frac = 1.0 - exp(-(a2 * dt / t_sync).min(100.0));
    let sync_ratio = a1 / a2.max(1.0e-12);
    for k in 0..star.nz {
        let j_sync = omega_orb * star.aux.i_rot[k];
        star.j_rot[k] -= relax_frac * (star.j_rot[k] - sync_ratio * j_sync);
        star.omega[k] = star.j_rot[k] / star.aux.i_rot[k].max(1.0e-30);
    }
}

fn eval_star_k_div_t(
    star: &StarModel,
    prescription: &str,
    tidal_reduction: f64,
    m_companion: f64,
    separation: f64,
    period: f64,
) -> f64 {
    if prescription.eq_ignore_ascii_case("Hut_rad") {
        eval_k_div_t_radiative(star.mstar, m_companion, star.r[0], separation)
    } else {
        let (m_env_msun, r_env_rsun) = eval_convective_envelope_mass_and_size(star);
        if m_env_msun > 1.0e-10 && r_env_rsun > 1.0e-10 {
            eval_k_div_t_convective(
                star.mstar / MSUN,
                m_env_msun,
                star.r[0] / RSUN,
                r_env_rsun,
                (star.lum[0] / LSUN).max(1.0e-10),
                period,
                star.omega.first().copied().unwrap_or(0.0),
                tidal_reduction,
            )
        } else {
            eval_k_div_t_radiative(star.mstar, m_companion, star.r[0], separation)
        }
    }
}

/// Evaluates all gated orbital $\dot{J}$ terms matching `get_jdot` in
/// `binary/private/binary_jdot.f90:33-246`.
#[allow(clippy::too_many_arguments)]
pub fn eval_gated_jdot(
    ctrl: &BinaryInlistControls,
    donor: &StarModel,
    accretor_star: Option<&StarModel>,
    donor_i: usize,
    m_donor: f64,
    m_accretor: f64,
    rl_donor: f64,
    rl_accretor: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
    j_orb: f64,
    mtransfer_rate: f64,
    mdot_sys_tr_donor: f64,
    mdot_sys_wind_donor: f64,
    mdot_sys_tr_accretor: f64,
    mdot_sys_wind_accretor: f64,
    mdot_sys_cct: f64,
    s_pre_donor: f64,
    s_post_donor: f64,
    j_surf_pre_donor: f64,
    s_pre_accretor: f64,
    s_post_accretor: f64,
    j_surf_pre_accretor: f64,
    dt: f64,
    jdot_extra: f64,
    using_jdot_mb: &mut [bool; 2],
) -> JdotBreakdown {
    let d_slot = (donor_i.saturating_sub(1)).min(1);
    let a_slot = 1 - d_slot;
    let jdot_mb = if ctrl.do_jdot_mb {
        let (use_d, qconv_d) = check_jdot_mb_conditions(donor, ctrl, using_jdot_mb[d_slot]);
        using_jdot_mb[d_slot] = use_d;
        let mut mb = if use_d {
            eval_jdot_mb_with_scale_mass(
                m_donor, donor.r[0], rl_donor, period, ctrl.magnetic_braking_gamma,
                qconv_d, ctrl.jdot_mb_scale_for_low_qconv_env, ctrl.jdot_mb_mass_frac_for_scale,
            )
        } else {
            0.0
        };
        if ctrl.include_accretor_mb {
            if let Some(acc) = accretor_star {
                let (use_a, qconv_a) = check_jdot_mb_conditions(acc, ctrl, using_jdot_mb[a_slot]);
                using_jdot_mb[a_slot] = use_a;
                if use_a {
                    mb += eval_jdot_mb_with_scale_mass(
                        m_accretor, acc.r[0], rl_accretor, period, ctrl.magnetic_braking_gamma,
                        qconv_a, ctrl.jdot_mb_scale_for_low_qconv_env, ctrl.jdot_mb_mass_frac_for_scale,
                    );
                }
            }
        }
        mb
    } else {
        *using_jdot_mb = [false; 2];
        0.0
    };
    let jdot_gr = if ctrl.do_jdot_gr {
        eval_jdot_gr(m_donor, m_accretor, separation, j_orb)
    } else {
        0.0
    };
    let jdot_ml = if ctrl.do_jdot_ml {
        eval_jdot_ml(
            m_donor, m_accretor, separation, period, eccentricity,
            mdot_sys_tr_donor, mdot_sys_wind_donor, mdot_sys_tr_accretor,
            mdot_sys_wind_accretor, mdot_sys_cct, ctrl.mass_transfer_gamma,
        )
    } else {
        0.0
    };
    let fixed_xfer = (1.0 - ctrl.mass_transfer_alpha - ctrl.mass_transfer_beta - ctrl.mass_transfer_delta)
        .clamp(0.0, 1.0);
    let jdot_missing_wind = if ctrl.do_jdot_missing_wind {
        accretor_star.map_or(0.0, |acc| {
            eval_jdot_missing_wind(
                false, acc.mstar_dot, mtransfer_rate, fixed_xfer,
                mdot_sys_wind_accretor, j_surf_pre_accretor,
            )
        })
    } else {
        0.0
    };
    let jdot_ls = if ctrl.do_jdot_ls {
        let mw_d = if donor.mstar_dot < 0.0 { mdot_sys_wind_donor } else { 0.0 };
        let mw_a = accretor_star.filter(|a| a.mstar_dot < 0.0).map_or(0.0, |a| a.mstar_dot);
        eval_jdot_ls_with_winds(
            s_pre_donor, s_post_donor, mw_d, j_surf_pre_donor,
            s_pre_accretor, s_post_accretor, mw_a, j_surf_pre_accretor, dt,
        )
    } else {
        0.0
    };
    let jdot_total = (jdot_mb + jdot_gr + jdot_ml + jdot_missing_wind + jdot_extra)
        * ctrl.jdot_multiplier
        + jdot_ls;
    JdotBreakdown {
        jdot_mb, jdot_gr, jdot_ml, jdot_missing_wind, jdot_extra, jdot_ls, jdot_total,
    }
}

/// Evaluates gated orbital eccentricity derivative $\dot{e}$ ($\mathrm{s}^{-1}$)
/// (`get_edot` in `binary/private/binary_edot.f90:33-182`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_coupled_edot(
    ctrl: &BinaryInlistControls,
    donor: &StarModel,
    accretor_star: Option<&StarModel>,
    donor_i: usize,
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
) -> f64 {
    if eccentricity <= 0.0 {
        return 0.0;
    }
    let edot_gr = if ctrl.do_jdot_gr {
        eval_edot_gr(m_donor, m_accretor, separation, eccentricity)
    } else {
        0.0
    };
    let edot_tidal = if ctrl.do_tidal_circ {
        let (circ_d, ftid_d, circ_a, ftid_a) = if donor_i == 1 {
            (ctrl.circ_type_1.as_str(), ctrl.ftid_1, ctrl.circ_type_2.as_str(), ctrl.ftid_2)
        } else {
            (ctrl.circ_type_2.as_str(), ctrl.ftid_2, ctrl.circ_type_1.as_str(), ctrl.ftid_1)
        };
        let kt_d = eval_star_k_div_t(donor, circ_d, ctrl.tidal_reduction, m_accretor, separation, period);
        let mut et = eval_edot_tidal_hut(
            kt_d, m_donor, m_accretor, donor.r[0], separation, eccentricity,
            donor.omega.first().copied().unwrap_or(0.0), period, ftid_d,
        );
        if let Some(acc) = accretor_star {
            let kt_a = eval_star_k_div_t(acc, circ_a, ctrl.tidal_reduction, m_donor, separation, period);
            et += eval_edot_tidal_hut(
                kt_a, m_accretor, m_donor, acc.r[0], separation, eccentricity,
                acc.omega.first().copied().unwrap_or(0.0), period, ftid_a,
            );
        }
        if ctrl.max_abs_edot_tidal > 0.0 {
            et.clamp(-ctrl.max_abs_edot_tidal, ctrl.max_abs_edot_tidal)
        } else {
            et
        }
    } else {
        0.0
    };
    edot_gr + edot_tidal
}
