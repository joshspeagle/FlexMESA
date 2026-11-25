// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled orbital angular-momentum derivatives (`do_jdot_{mb,gr,ml,missing_wind,ls}`),
//! shellular spin conservation across Henyey remeshing, post-step rotational
//! compositional mixing, and Hut (1981) / Zahn (1977) tidal spin synchronization.
//!
//! Corresponds to `binary/private/binary_jdot.f90:33-246`,
//! `binary/private/binary_tides.f90:120-456`, and
//! `binary/private/binary_edot.f90:33-182`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use flexmesa_const::{MixingType, MSUN, PI, RSUN};
use flexmesa_math::{exp, pow};
use flexmesa_star::{
    apply_uniform_rotation, eval_photosphere_info, eval_surf_avg_rotation_info,
    update_rotation_distortion_factors, BinaryOtherTorqueHook, StarError,
    StarModel,
};

use crate::inlist::BinaryInlistControls;
use crate::jdot_tides::{
    check_jdot_mb_conditions, eval_edot_tidal_hut, eval_jdot_gr,
    eval_jdot_ls_with_am_removed, eval_jdot_mb_with_scale_mass,
    eval_jdot_missing_wind, eval_jdot_ml, eval_k_div_t_convective,
    eval_k_div_t_radiative, eval_tsync, hut_f2, hut_f5,
};
use crate::mdot::eval_edot_enhancement_isotropic;
use crate::orbit::eval_rlobe;

/// Returns photospheric radius `b% r(i) = Rsun * s% photosphere_r` in cm
/// (`binary/private/binary_evolve.f90:92,97,286,291`).
#[inline]
#[must_use]
pub(crate) fn star_photosphere_radius_cm(star: &StarModel) -> f64 {
    RSUN * eval_photosphere_info(star).r_rsun
}

/// Component-by-component breakdown of the orbital torque $\dot{J}_{\mathrm{orb}}$
/// ($\mathrm{g\,cm^2\,s^{-2}}$) in `get_jdot` (`binary_jdot.f90:33-93`).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
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

/// Evaluates dimensionless squared gyration radius $r_{\mathrm{gyr}}^2 = I_* / (M_* R_{\mathrm{phot}}^2)$
/// (`binary/private/binary_tides.f90:355-359`) using explicit binary mass `m_star` (g)
/// and photospheric radius `r_phot_cm` (cm).
#[must_use]
pub(crate) fn eval_star_r_gyr_sq_with_mr(
    star: &StarModel,
    m_star: f64,
    r_phot_cm: f64,
) -> f64 {
    if star.nz == 0 || m_star <= 0.0 || r_phot_cm <= 0.0 {
        return 0.1;
    }
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
    i_tot / (m_star * r_phot_cm * r_phot_cm)
}

/// Evaluates dimensionless squared gyration radius $r_{\mathrm{gyr}}^2 = I_* / (M_* R_{\mathrm{phot}}^2)$
/// (`binary/private/binary_tides.f90:355-359`).
#[must_use]
pub fn eval_star_r_gyr_sq(star: &StarModel) -> f64 {
    eval_star_r_gyr_sq_with_mr(
        star,
        star.mstar,
        star_photosphere_radius_cm(star).max(1.0),
    )
}

/// Estimates outer convective envelope mass ($M_\odot$) and radial thickness ($R_\odot$)
/// for Hurley et al. (2002) convective-envelope tidal dissipation (`binary_tides.f90:418-427`)
/// using explicit `r_phot_cm` (`b% r(i)`).
#[must_use]
pub(crate) fn eval_convective_envelope_mass_and_size_with_rphot(
    star: &StarModel,
    r_phot_cm: f64,
) -> (f64, f64) {
    if star.nz == 0 {
        return (0.0, 0.0);
    }
    let m_top = star.m.first().copied().unwrap_or(star.mstar);
    let r_top = star.r[0];
    let rho_top = star.rho[0];
    let mut m_env = m_top / MSUN;
    let mut r_env = r_top / RSUN;
    for k in 0..star.nz {
        let is_conv = star
            .mixing_type
            .get(k)
            .copied()
            .map_or(star.gradr[k] > star.grada[k], |mt| {
                mt == MixingType::ConvectiveMixing
            });
        if !is_conv && star.rho[k] > 1.0e5 * rho_top {
            r_env = (r_phot_cm - star.r[k]) / RSUN;
            m_env = (m_top - star.m[k]) / MSUN;
            break;
        }
    }
    (m_env, r_env)
}

/// Estimates outer convective envelope mass ($M_\odot$) and radial thickness ($R_\odot$)
/// for Hurley et al. (2002) convective-envelope tidal dissipation (`binary_tides.f90:418-427`).
#[must_use]
pub fn eval_convective_envelope_mass_and_size(star: &StarModel) -> (f64, f64) {
    eval_convective_envelope_mass_and_size_with_rphot(
        star,
        star_photosphere_radius_cm(star),
    )
}

/// Synchronizes a stellar model uniformly to orbital period `period_sec` (s) at
/// startup (`do_initial_orbit_sync_{1,2}` in `run_binary_support.f90:319-340`).
///
/// When rotation is already enabled on the star, relaxes $\Omega$ smoothly via
/// [`flexmesa_star::relax_uniform_omega`] (`star_relax_uniform_omega` with
/// upstream defaults `num_steps_to_relax_rotation = 100`,
/// `relax_omega_max_yrs_dt = 1e9`), matching MESA `26.4.1`.
pub fn apply_initial_orbit_sync(
    star: &mut StarModel,
    period_sec: f64,
) -> Result<(), StarError> {
    if period_sec <= 0.0 {
        return Ok(());
    }
    let target_omega = 2.0 * PI / period_sec;
    if star.controls.rotation_flag {
        flexmesa_star::relax_uniform_omega(star, target_omega, 100, 1.0e9)?;
    } else {
        star.controls.rotation_flag = true;
        apply_uniform_rotation(star, target_omega);
        update_rotation_distortion_factors(star);
    }
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
            let idx =
                q_mid_old.partition_point(|&qv| qv > qm).clamp(1, n_old - 1);
            let (q_hi, q_lo) = (q_mid_old[idx - 1], q_mid_old[idx]);
            let t = ((q_hi - qm) / (q_hi - q_lo).max(1.0e-30)).clamp(0.0, 1.0);
            om_old[idx - 1] + t * (om_old[idx] - om_old[idx - 1])
        };
        star.omega[k] = om_interp;
    }
}

/// Restores exact total stellar spin angular momentum $S_{\mathrm{target}} = \max(S_{\mathrm{pre}} + \Delta S_{\dot{M}}, 0)$
/// after Henyey radius changes and remeshing (`set_hydro_vars` updates `i_rot` at old `omega`).
pub fn conserve_spin_after_henyey(
    star: &mut StarModel,
    s_pre: f64,
    delta_s_mdot: f64,
) {
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

pub(crate) fn validate_sync_type(s: &str) -> Result<(), StarError> {
    if matches!(
        s.trim().to_ascii_lowercase().as_str(),
        "instantaneous" | "orb_period" | "hut_conv" | "hut_rad" | "none"
    ) {
        Ok(())
    } else {
        Err(StarError::InvalidInput(format!(
            "unrecognized tidal sync_type '{s}'"
        )))
    }
}

pub(crate) fn validate_sync_mode(s: &str) -> Result<(), StarError> {
    if matches!(
        s.trim().to_ascii_lowercase().as_str(),
        "uniform" | "tdyn_div_tkh"
    ) {
        Ok(())
    } else {
        Err(StarError::InvalidInput(format!(
            "unrecognized tidal sync_mode '{s}'"
        )))
    }
}

pub(crate) fn validate_circ_type(s: &str) -> Result<(), StarError> {
    if matches!(
        s.trim().to_ascii_lowercase().as_str(),
        "hut_conv" | "hut_rad" | "none"
    ) {
        Ok(())
    } else {
        Err(StarError::InvalidInput(format!(
            "unrecognized tidal circ_type '{s}'"
        )))
    }
}

/// Evaluates exact stellar spin angular momentum removed during mass loss
/// (`s% angular_momentum_removed` in `star/private/adjust_mass.f90:527-552, 1508-1569`),
/// matching [`flexmesa_star::adjust_mass::do_adjust_mass_step`].
#[must_use]
pub(crate) fn eval_star_am_removed_post_step(star: &StarModel) -> f64 {
    if !star.controls.rotation_flag || star.mstar_dot >= 0.0 || star.nz == 0 {
        return 0.0;
    }
    let mass_lost = (star.mstar_old - star.mstar).max(0.0);
    if mass_lost <= 0.0
        || star.dq_old.len() != star.nz
        || star.aux.j_rot_old.len() != star.nz
    {
        return 0.0;
    }
    let xm_old = (star.mstar_old - star.m_center).max(0.0);
    let nz = star.nz;
    let mut dm00 = 0.0_f64;
    let mut dm_sum = 0.0_f64;
    let mut j_removed = 0.0_f64;
    for k in 0..nz {
        let dmm1 = dm00;
        dm00 = xm_old * star.dq_old[k];
        let mut dm_k = if k + 1 == nz {
            0.5 * dmm1 + dm00
        } else {
            0.5 * (dmm1 + dm00)
        };
        if dm_sum + dm_k >= mass_lost {
            dm_k = (mass_lost - dm_sum).max(0.0);
            j_removed += dm_k * star.aux.j_rot_old[k];
            break;
        }
        dm_sum += dm_k;
        j_removed += dm_k * star.aux.j_rot_old[k];
    }
    let mw = &star.controls.mix_wind;
    if mw.do_adjust_j_lost {
        let f_adj = mw.adjust_j_fraction.clamp(0.0, 1.0);
        f_adj * mass_lost * star.aux.j_rot_old[0] + (1.0 - f_adj) * j_removed
    } else {
        j_removed
    }
}

/// Relaxes shellular spin $\Omega_k$ toward orbital synchronization over `dt` (s)
/// following `sync_spin_orbit_torque` (`binary/private/binary_tides.f90:93-271`).
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
) -> Result<(), StarError> {
    let _ = apply_tidal_spin_sync_step(
        star,
        false,
        sync_type,
        sync_mode,
        f_tid,
        tidal_reduction,
        m_companion,
        separation,
        period,
        eccentricity,
        dt,
    )?;
    Ok(())
}

/// Coupling parameters passed to `sync_spin_to_orbit`
/// (`binary/private/binary_tides.f90:93-271`).
struct SyncSpinOrbitParams<'a> {
    sync_type: &'a str,
    sync_mode: &'a str,
    f_tid: f64,
    tidal_reduction: f64,
    m_star: f64,
    r_phot_cm: f64,
    l_phot_lsun: f64,
    omega_avg_surf: f64,
    m_companion: f64,
    rl_cm: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
    dt_next: f64,
}

/// Evaluates `sync_spin_to_orbit` (`binary/private/binary_tides.f90:93-254`),
/// returning `Ok(None)` on upstream early returns or `Ok(Some((t_sync, delta_j)))`.
fn eval_sync_spin_to_orbit_delta_j(
    star: &StarModel,
    p: &SyncSpinOrbitParams<'_>,
) -> Result<Option<(f64, Vec<f64>)>, StarError> {
    validate_sync_type(p.sync_type)?;
    validate_sync_mode(p.sync_mode)?;
    let nz = star.nz;
    let qratio = if p.m_star > 0.0 {
        p.m_companion / p.m_star
    } else {
        0.0
    };
    if p.separation <= 0.0
        || qratio <= 0.0
        || p.rl_cm <= 0.0
        || p.dt_next <= 0.0
        || p.f_tid <= 0.0
        || p.period <= 0.0
        || p.r_phot_cm <= 0.0
        || !star.controls.rotation_flag
        || nz == 0
        || p.sync_type.eq_ignore_ascii_case("None")
    {
        return Ok(None);
    }

    let omega_orb = 2.0 * PI / p.period;
    let j_sync: Vec<f64> = (0..nz)
        .map(|k| omega_orb * star.aux.i_rot.get(k).copied().unwrap_or(0.0))
        .collect();

    if p.sync_type.eq_ignore_ascii_case("Instantaneous") {
        let delta_j = (0..nz).map(|k| star.j_rot[k] - j_sync[k]).collect();
        return Ok(Some((0.0, delta_j)));
    }

    let t_sync = if p.sync_type.eq_ignore_ascii_case("Orb_period") {
        p.period / p.f_tid
    } else {
        let k_div_t = eval_star_k_div_t_with_state(
            star,
            p.m_star,
            p.r_phot_cm,
            p.l_phot_lsun,
            p.omega_avg_surf,
            p.sync_type,
            p.tidal_reduction,
            p.m_companion,
            p.separation,
            p.period,
        )?;
        if k_div_t <= 0.0 {
            return Ok(Some((0.0, vec![0.0; nz])));
        }
        let r_gyr_sq = eval_star_r_gyr_sq_with_mr(star, p.m_star, p.r_phot_cm);
        if r_gyr_sq <= 0.0 {
            return Ok(Some((0.0, vec![0.0; nz])));
        }
        eval_tsync(
            k_div_t,
            qratio,
            r_gyr_sq,
            p.r_phot_cm,
            p.separation,
            p.f_tid,
        )
    };

    if !t_sync.is_finite() || t_sync <= 0.0 {
        return Ok(Some((0.0, vec![0.0; nz])));
    }

    let delta_j = if p.sync_mode.eq_ignore_ascii_case("tdyn_div_tkh") {
        eval_tdyn_div_tkh_delta_j(star, &j_sync, t_sync, p.dt_next)
    } else {
        let a1 = hut_f2(p.eccentricity);
        let a2 = pow(1.0 - p.eccentricity * p.eccentricity, 1.5)
            * hut_f5(p.eccentricity);
        let factor = 1.0 - exp(-a2 * p.dt_next / t_sync);
        (0..nz)
            .map(|k| factor * (star.j_rot[k] - a1 / a2 * j_sync[k]))
            .collect()
    };
    Ok(Some((t_sync, delta_j)))
}

/// Evaluates `sync_spin_to_orbit` (`binary/private/binary_tides.f90:93-271`)
/// directly on `star`, returning `t_sync` (s) (`0.0` when inactive or `"Instantaneous"`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tidal_spin_sync_step(
    star: &mut StarModel,
    doing_first_model_of_run: bool,
    sync_type: &str,
    sync_mode: &str,
    f_tid: f64,
    tidal_reduction: f64,
    m_companion: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
    dt: f64,
) -> Result<f64, StarError> {
    validate_sync_type(sync_type)?;
    validate_sync_mode(sync_mode)?;
    if !star.controls.rotation_flag
        || star.nz == 0
        || separation <= 0.0
        || period <= 0.0
        || dt <= 0.0
        || f_tid <= 0.0
        || sync_type.eq_ignore_ascii_case("None")
    {
        return Ok(0.0);
    }
    update_rotation_distortion_factors(star);
    let phot = eval_photosphere_info(star);
    let r_phot_cm = RSUN * phot.r_rsun;
    let omega_avg_surf = eval_surf_avg_rotation_info(star).omega;
    let params = SyncSpinOrbitParams {
        sync_type,
        sync_mode,
        f_tid,
        tidal_reduction,
        m_star: star.mstar,
        r_phot_cm,
        l_phot_lsun: phot.l_lsun,
        omega_avg_surf,
        m_companion,
        rl_cm: r_phot_cm,
        separation,
        period,
        eccentricity,
        dt_next: dt,
    };
    let Some((t_sync, delta_j)) =
        eval_sync_spin_to_orbit_delta_j(star, &params)?
    else {
        return Ok(0.0);
    };
    if !doing_first_model_of_run {
        for k in 0..star.nz {
            star.j_rot[k] -= delta_j[k];
            star.omega[k] = star.j_rot[k] / star.aux.i_rot[k].max(1.0e-30);
        }
    }
    Ok(t_sync)
}

/// [`BinaryOtherTorqueHook`] implementing upstream `sync_spin_orbit_torque`
/// (`binary/private/binary_tides.f90:35-271`) inside `do_solve_omega_mix`
/// (`star/private/solve_omega_mix.f90:90-97`).
#[derive(Debug, Clone)]
pub(crate) struct BinarySpinOrbitTorqueHook {
    pub do_tidal_sync: bool,
    pub doing_first_model_of_run: bool,
    pub is_star_1: bool,
    pub model_twins_flag: bool,
    pub separation: f64,
    pub period: f64,
    pub eccentricity: f64,
    pub rlr: f64,
    pub m_star: f64,
    pub m_companion: f64,
    pub r_phot: f64,
    /// Prior end-of-step/relax surface-averaged $\Omega$ (`s% omega_avg_surf`),
    /// matching `binary_tides.f90:430-431` where `s% omega_avg_surf` updates at
    /// `relax`/`report`/`select_mdot_action` rather than inside `do_solve_omega_mix`.
    /// An implicit-wind redo inside a single star step does not refresh this
    /// binary-side snapshot (`#160`; unused by `Hut_rad`).
    pub omega_avg_surf: f64,
    pub f_tid: f64,
    pub sync_type: String,
    pub sync_mode: String,
    pub tidal_reduction: f64,
    pub t_sync_cells: Arc<[AtomicU64; 2]>,
}

impl BinaryOtherTorqueHook for BinarySpinOrbitTorqueHook {
    fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
        if !self.do_tidal_sync {
            return Ok(());
        }
        let dt_next = model.dt;
        let l_phot_lsun = eval_photosphere_info(model).l_lsun;
        let params = SyncSpinOrbitParams {
            sync_type: self.sync_type.as_str(),
            sync_mode: self.sync_mode.as_str(),
            f_tid: self.f_tid,
            tidal_reduction: self.tidal_reduction,
            m_star: self.m_star,
            r_phot_cm: self.r_phot,
            l_phot_lsun,
            omega_avg_surf: self.omega_avg_surf,
            m_companion: self.m_companion,
            rl_cm: self.rlr,
            separation: self.separation,
            period: self.period,
            eccentricity: self.eccentricity,
            dt_next,
        };
        let Some((t_sync, delta_j)) =
            eval_sync_spin_to_orbit_delta_j(model, &params)?
        else {
            return Ok(());
        };
        let slot = usize::from(!self.is_star_1);
        self.t_sync_cells[slot].store(t_sync.to_bits(), Ordering::Relaxed);
        if self.is_star_1 && self.model_twins_flag {
            self.t_sync_cells[1].store(t_sync.to_bits(), Ordering::Relaxed);
        }
        if !self.doing_first_model_of_run {
            for (ej, &dj) in
                model.aux.extra_jdot.iter_mut().take(model.nz).zip(&delta_j)
            {
                *ej -= dj / dt_next;
            }
        }
        Ok(())
    }
}

fn interp_val_to_face(vals: &[f64], dq: &[f64], k: usize) -> f64 {
    if k == 0 || k >= vals.len() {
        return vals.first().copied().unwrap_or(0.0);
    }
    let dq_m1 = dq.get(k - 1).copied().unwrap_or(1.0);
    let dq_00 = dq.get(k).copied().unwrap_or(1.0);
    let sum = (dq_m1 + dq_00).max(1.0e-30);
    (dq_00 * vals[k - 1] + dq_m1 * vals[k]) / sum
}

fn eval_tdyn_div_tkh_delta_j(
    star: &StarModel,
    j_sync: &[f64],
    t_sync: f64,
    dt_next: f64,
) -> Vec<f64> {
    let nz = star.nz;
    let mut tdyn_div_tkh = vec![0.0_f64; nz];
    let mut delta_total_j = 0.0_f64;
    for k in 0..nz {
        let rho_f = interp_val_to_face(&star.rho, &star.dq, k);
        let cv_f = interp_val_to_face(&star.cv, &star.dq, k);
        let t_f = interp_val_to_face(&star.t, &star.dq, k);
        let cs_f = interp_val_to_face(&star.csound, &star.dq, k).max(1.0e-30);
        let lum_k = if star.lum[k] == 0.0 {
            1.0e-30
        } else {
            star.lum[k]
        };
        let tkh = 4.0 * PI * star.r[k] * star.r[k] * rho_f * cv_f * t_f / lum_k;
        let tdyn = 1.0 / cs_f;
        tdyn_div_tkh[k] = if tkh == 0.0 { 0.0 } else { tdyn / tkh };
        let dm_k = star.dm_bar.get(k).copied().unwrap_or(star.dm[k]);
        delta_total_j += (star.j_rot[k] - j_sync[k]).abs() * dm_k;
    }
    delta_total_j *= 1.0 - exp(-dt_next / t_sync);
    let mut layers_in_sync = vec![false; nz];
    let mut delta_j = vec![0.0_f64; nz];
    let mut num_sync_layers = 0_usize;
    loop {
        let (mut sum_j_sync, mut sum_j_non_sync) = (0.0_f64, 0.0_f64);
        for k in 0..nz {
            let dj = star.j_rot[k] - j_sync[k];
            delta_j[k] = dj;
            let dm_k = star.dm_bar.get(k).copied().unwrap_or(star.dm[k]);
            if layers_in_sync[k] {
                sum_j_sync += dm_k * dj.abs();
            } else {
                sum_j_non_sync +=
                    dm_k * dj.abs() * tdyn_div_tkh[k] * tdyn_div_tkh[k];
            }
        }
        if sum_j_non_sync <= 0.0 || delta_total_j <= sum_j_sync {
            for k in 0..nz {
                if !layers_in_sync[k] {
                    delta_j[k] = 0.0;
                }
            }
            break;
        }
        let ff = (delta_total_j - sum_j_sync) / sum_j_non_sync;
        for k in 0..nz {
            if layers_in_sync[k] {
                continue;
            }
            let dj_full = star.j_rot[k] - j_sync[k];
            let scaled = dj_full * tdyn_div_tkh[k] * tdyn_div_tkh[k] * ff;
            if (dj_full >= 0.0 && scaled >= dj_full)
                || (dj_full < 0.0 && scaled <= dj_full)
            {
                delta_j[k] = dj_full;
                layers_in_sync[k] = true;
            } else {
                delta_j[k] = scaled;
            }
        }
        let new_sync_count = layers_in_sync.iter().filter(|&&b| b).count();
        if new_sync_count == num_sync_layers {
            break;
        }
        num_sync_layers = new_sync_count;
    }
    delta_j
}

#[allow(clippy::too_many_arguments)]
fn eval_star_k_div_t_with_state(
    star: &StarModel,
    m_star: f64,
    r_phot_cm: f64,
    l_phot_lsun: f64,
    omega_surf: f64,
    prescription: &str,
    tidal_reduction: f64,
    m_companion: f64,
    separation: f64,
    period: f64,
) -> Result<f64, StarError> {
    if prescription.eq_ignore_ascii_case("Hut_rad") {
        Ok(eval_k_div_t_radiative(
            m_star,
            m_companion,
            r_phot_cm,
            separation,
        ))
    } else if prescription.eq_ignore_ascii_case("Hut_conv") {
        let (m_env_msun, r_env_rsun) =
            eval_convective_envelope_mass_and_size_with_rphot(star, r_phot_cm);
        if m_env_msun > 0.0 && r_env_rsun > 0.0 {
            Ok(eval_k_div_t_convective(
                m_star / MSUN,
                m_env_msun,
                r_phot_cm / RSUN,
                r_env_rsun,
                l_phot_lsun.max(1.0e-30),
                period,
                omega_surf,
                tidal_reduction,
            ))
        } else {
            Ok(0.0)
        }
    } else if prescription.eq_ignore_ascii_case("None") {
        Ok(0.0)
    } else {
        Err(StarError::InvalidInput(format!(
            "unrecognized tidal prescription '{prescription}'"
        )))
    }
}

fn eval_star_k_div_t(
    star: &StarModel,
    m_star: f64,
    prescription: &str,
    tidal_reduction: f64,
    m_companion: f64,
    separation: f64,
    period: f64,
) -> Result<f64, StarError> {
    let phot = eval_photosphere_info(star);
    let r_phot_cm = RSUN * phot.r_rsun;
    let omega_surf = eval_surf_avg_rotation_info(star).omega;
    eval_star_k_div_t_with_state(
        star,
        m_star,
        r_phot_cm,
        phot.l_lsun,
        omega_surf,
        prescription,
        tidal_reduction,
        m_companion,
        separation,
        period,
    )
}

/// Evaluates all gated orbital $\dot{J}$ terms matching `get_jdot` in
/// `binary/private/binary_jdot.f90:33-328`.
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
    let old_mb = *using_jdot_mb;
    let mut cf = ctrl.max_change_factor;
    let am_rem_d = if donor.mstar_dot < 0.0 {
        (-donor.mstar_dot * dt).max(0.0) * j_surf_pre_donor
    } else {
        0.0
    };
    let am_rem_a = accretor_star.map_or(0.0, |acc| {
        if acc.mstar_dot < 0.0 {
            (-acc.mstar_dot * dt).max(0.0) * j_surf_pre_accretor
        } else {
            0.0
        }
    });
    eval_gated_jdot_step(
        ctrl,
        false,
        donor,
        accretor_star,
        donor_i,
        m_donor,
        m_accretor,
        rl_donor,
        rl_accretor,
        separation,
        period,
        eccentricity,
        j_orb,
        mtransfer_rate,
        mdot_sys_tr_donor,
        mdot_sys_wind_donor,
        mdot_sys_tr_accretor,
        mdot_sys_wind_accretor,
        mdot_sys_cct,
        s_pre_donor,
        s_post_donor,
        am_rem_d,
        s_pre_accretor,
        s_post_accretor,
        am_rem_a,
        dt,
        jdot_extra,
        old_mb,
        using_jdot_mb,
        &mut cf,
    )
}

/// Evaluates all gated orbital $\dot{J}$ terms in `get_jdot`
/// (`binary/private/binary_jdot.f90:33-328`) during a coupled binary step,
/// honoring `doing_first_model_of_run` (`jdot_ls == 0`), post-step `j_rot[0]`
/// in `default_jdot_missing_wind`, and resetting `change_factor` to
/// `max_change_factor` when magnetic braking switches off (`binary_jdot.f90:289,319`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_gated_jdot_step(
    ctrl: &BinaryInlistControls,
    doing_first_model_of_run: bool,
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
    am_removed_donor: f64,
    s_pre_accretor: f64,
    s_post_accretor: f64,
    am_removed_accretor: f64,
    dt: f64,
    jdot_extra: f64,
    using_jdot_mb_old: [bool; 2],
    using_jdot_mb: &mut [bool; 2],
    change_factor: &mut f64,
) -> JdotBreakdown {
    let d_slot = (donor_i.saturating_sub(1)).min(1);
    let a_slot = 1 - d_slot;
    let point_mass_i = if accretor_star.is_some() {
        0
    } else if donor_i == 1 {
        2
    } else {
        1
    };
    let jdot_gr = if ctrl.do_jdot_gr {
        eval_jdot_gr(m_donor, m_accretor, separation, j_orb)
    } else {
        0.0
    };
    let jdot_ml = if ctrl.do_jdot_ml {
        eval_jdot_ml(
            m_donor,
            m_accretor,
            separation,
            period,
            eccentricity,
            mdot_sys_tr_donor,
            mdot_sys_wind_donor,
            mdot_sys_tr_accretor,
            mdot_sys_wind_accretor,
            mdot_sys_cct,
            ctrl.mass_transfer_gamma,
        )
    } else {
        0.0
    };
    let jdot_ls = if ctrl.do_jdot_ls {
        eval_jdot_ls_with_am_removed(
            doing_first_model_of_run,
            point_mass_i,
            false,
            s_pre_donor,
            s_post_donor,
            donor.mstar_dot,
            am_removed_donor,
            mdot_sys_wind_donor,
            s_pre_accretor,
            s_post_accretor,
            accretor_star.map_or(0.0, |acc| acc.mstar_dot),
            am_removed_accretor,
            dt,
        )
    } else {
        0.0
    };
    let fixed_xfer = 1.0
        - ctrl.mass_transfer_alpha
        - ctrl.mass_transfer_beta
        - ctrl.mass_transfer_delta;
    let jdot_missing_wind = if ctrl.do_jdot_missing_wind {
        accretor_star.map_or(0.0, |acc| {
            let j_surf_post_acc = acc.j_rot.first().copied().unwrap_or(0.0);
            eval_jdot_missing_wind(
                false,
                acc.mstar_dot,
                mtransfer_rate,
                fixed_xfer,
                mdot_sys_wind_accretor,
                j_surf_post_acc,
            )
        })
    } else {
        0.0
    };
    let jdot_mb = if ctrl.do_jdot_mb {
        let (use_d, qconv_d) =
            check_jdot_mb_conditions(donor, ctrl, using_jdot_mb_old[d_slot]);
        let mut mb = if use_d {
            using_jdot_mb[d_slot] = true;
            eval_jdot_mb_with_scale_mass(
                m_donor,
                star_photosphere_radius_cm(donor),
                rl_donor,
                period,
                ctrl.magnetic_braking_gamma,
                qconv_d,
                ctrl.jdot_mb_scale_for_low_qconv_env,
                ctrl.jdot_mb_mass_frac_for_scale,
            )
        } else {
            if using_jdot_mb_old[d_slot] {
                *change_factor = ctrl.max_change_factor;
            }
            using_jdot_mb[d_slot] = false;
            0.0
        };
        if ctrl.include_accretor_mb {
            if let Some(acc) = accretor_star {
                let (use_a, qconv_a) = check_jdot_mb_conditions(
                    acc,
                    ctrl,
                    using_jdot_mb_old[a_slot],
                );
                if use_a {
                    using_jdot_mb[a_slot] = true;
                    mb += eval_jdot_mb_with_scale_mass(
                        m_accretor,
                        star_photosphere_radius_cm(acc),
                        rl_accretor,
                        period,
                        ctrl.magnetic_braking_gamma,
                        qconv_a,
                        ctrl.jdot_mb_scale_for_low_qconv_env,
                        ctrl.jdot_mb_mass_frac_for_scale,
                    );
                } else {
                    if using_jdot_mb_old[a_slot] {
                        *change_factor = ctrl.max_change_factor;
                    }
                    using_jdot_mb[a_slot] = false;
                }
            }
        }
        mb
    } else {
        0.0
    };
    let jdot_total =
        (jdot_mb + jdot_gr + jdot_ml + jdot_missing_wind + jdot_extra)
            * ctrl.jdot_multiplier
            + jdot_ls;
    JdotBreakdown {
        jdot_mb,
        jdot_gr,
        jdot_ml,
        jdot_missing_wind,
        jdot_extra,
        jdot_ls,
        jdot_total,
    }
}

/// Eccentricity-rate terms applied by the orbital update ($\mathrm{s}^{-1}$).
///
/// These are the values the engine used, stored so that `binary_history.data`
/// reports them (`b% edot`, `b% edot_tidal`, `b% edot_enhance` in
/// `binary/private/binary_history.f90`) and never a recomputed quantity.
/// Upstream `get_edot` (`binary/private/binary_edot.f90:93`) sums only
/// `edot_tidal + edot_enhance + extra_edot` (`#175`, `DECISIONS #36`).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize,
)]
pub struct EdotBreakdown {
    /// Tidal circularisation term (`edot_tidal`).
    pub edot_tidal: f64,
    /// Eccentricity-enhancement term (`edot_enhance`).
    pub edot_enhance: f64,
    /// User-defined extra eccentricity rate (`extra_edot`).
    pub extra_edot: f64,
    /// Sum applied to the eccentricity (`edot = edot_tidal + edot_enhance + extra_edot`).
    pub edot_total: f64,
}

/// Evaluates gated orbital eccentricity derivative $\dot{e}$ ($\mathrm{s}^{-1}$)
/// (`get_edot` in `binary/private/binary_edot.f90:33-182`).
#[allow(clippy::too_many_arguments)]
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
) -> Result<f64, StarError> {
    Ok(eval_coupled_edot_breakdown(
        ctrl,
        donor,
        accretor_star,
        donor_i,
        m_donor,
        m_accretor,
        separation,
        period,
        eccentricity,
    )?
    .edot_total)
}

/// Evaluates the terms of the gated orbital eccentricity derivative
/// (`get_edot` in `binary/private/binary_edot.f90:33-182`).
#[allow(clippy::too_many_arguments)]
pub fn eval_coupled_edot_breakdown(
    ctrl: &BinaryInlistControls,
    donor: &StarModel,
    accretor_star: Option<&StarModel>,
    donor_i: usize,
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
) -> Result<EdotBreakdown, StarError> {
    validate_circ_type(&ctrl.circ_type_1)?;
    validate_circ_type(&ctrl.circ_type_2)?;
    if eccentricity <= 0.0 {
        return Ok(EdotBreakdown::default());
    }
    // `#175` (`binary/private/binary_edot.f90:33-99`): upstream `get_edot`
    // omits Peters (1964) gravitational-wave eccentricity decay and sums
    // `edot_tidal + edot_enhance + extra_edot` only (`DECISIONS #36`).
    let edot_tidal = if ctrl.do_tidal_circ {
        let (circ_d, circ_a) = if donor_i == 1 {
            (ctrl.circ_type_1.as_str(), ctrl.circ_type_2.as_str())
        } else {
            (ctrl.circ_type_2.as_str(), ctrl.circ_type_1.as_str())
        };
        let r_d_phot = star_photosphere_radius_cm(donor);
        let omega_d = eval_surf_avg_rotation_info(donor).omega;
        let kt_d = eval_star_k_div_t(
            donor,
            m_donor,
            circ_d,
            ctrl.tidal_reduction,
            m_accretor,
            separation,
            period,
        )?;
        // `#186` (`binary/private/binary_edot.f90:171-172`): `edot_tidal_Hut`
        // multiplies by `b% Ftid_1` unconditionally on both component 1 and
        // component 2 (ignoring `Ftid_2`), reproduced verbatim for MESA `26.4.1`.
        let mut et = eval_edot_tidal_hut(
            kt_d,
            m_donor,
            m_accretor,
            r_d_phot,
            separation,
            eccentricity,
            omega_d,
            period,
            ctrl.ftid_1,
        );
        if let Some(acc) = accretor_star {
            let r_a_phot = star_photosphere_radius_cm(acc);
            let omega_a = eval_surf_avg_rotation_info(acc).omega;
            let kt_a = eval_star_k_div_t(
                acc,
                m_accretor,
                circ_a,
                ctrl.tidal_reduction,
                m_donor,
                separation,
                period,
            )?;
            et += eval_edot_tidal_hut(
                kt_a,
                m_accretor,
                m_donor,
                r_a_phot,
                separation,
                eccentricity,
                omega_a,
                period,
                ctrl.ftid_1,
            );
        }
        if et < -ctrl.max_abs_edot_tidal {
            -ctrl.max_abs_edot_tidal
        } else {
            et
        }
    } else {
        0.0
    };
    let edot_enhance = if ctrl.use_eccentricity_enhancement && donor.nz > 0 {
        let rl_d = eval_rlobe(m_donor, m_accretor, separation);
        let r_d_phot = star_photosphere_radius_cm(donor);
        let is_kolb = ctrl.mdot_scheme.eq_ignore_ascii_case("Kolb");
        // `binary/private/binary_edot.f90:216` & `binary/private/binary_mdot.f90:843-847`:
        // pass raw surface `rho(1)`, `Peos(1)`, `T(1)`, and `mu(1)` without clamps.
        let mut enh = eval_edot_enhancement_isotropic(
            m_donor,
            m_accretor,
            r_d_phot,
            rl_d,
            eccentricity,
            donor.rho[0],
            donor.peos[0],
            donor.t[0],
            donor.mu[0],
            &donor.r,
            &donor.peos,
            &donor.t,
            &donor.mu,
            &donor.gamma1,
            is_kolb,
            ctrl.anomaly_steps,
        );
        if enh > ctrl.max_abs_edot_enhance {
            enh = ctrl.max_abs_edot_enhance;
        }
        enh
    } else {
        0.0
    };
    let extra_edot = 0.0;
    Ok(EdotBreakdown {
        edot_tidal,
        edot_enhance,
        extra_edot,
        edot_total: edot_tidal + edot_enhance + extra_edot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jdot_tides::{
        eval_k_div_t_convective, eval_k_div_t_radiative, eval_tsync, hut_f2,
        hut_f5,
    };
    use flexmesa_const::SECDAY;
    use flexmesa_star::{load_zams_model, StarControls, StarJobControls};

    fn rotating_zams_star(mass_msun: f64, omega: f64) -> StarModel {
        let mut star = load_zams_model(
            StarControls {
                initial_mass: mass_msun,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("ZAMS model");
        star.controls.rotation_flag = true;
        apply_uniform_rotation(&mut star, omega);
        update_rotation_distortion_factors(&mut star);
        star.dt = 10.0 * SECDAY;
        star.aux.extra_jdot = vec![0.0; star.nz];
        star
    }

    #[test]
    fn test_sync_spin_orbit_torque_first_model_records_tsync_with_zero_torque()
    {
        let period = 1.2 * SECDAY;
        let omega_orb = 2.0 * PI / period;
        let mut star = rotating_zams_star(5.0, 1.25 * omega_orb);
        let r_phot = star_photosphere_radius_cm(&star);
        let m_comp = 4.0 * MSUN;
        let osep = 12.0 * RSUN;
        let scratch = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let hook = BinarySpinOrbitTorqueHook {
            do_tidal_sync: true,
            doing_first_model_of_run: true,
            is_star_1: true,
            model_twins_flag: false,
            separation: osep,
            period,
            eccentricity: 0.0,
            rlr: 0.4 * osep,
            m_star: star.mstar,
            m_companion: m_comp,
            r_phot,
            omega_avg_surf: eval_surf_avg_rotation_info(&star).omega,
            f_tid: 1.0,
            sync_type: String::from("Hut_rad"),
            sync_mode: String::from("Uniform"),
            tidal_reduction: 2.0,
            t_sync_cells: scratch.clone(),
        };

        hook.apply(&mut star).expect("first-model hook succeeds");
        let ts_recorded = f64::from_bits(scratch[0].load(Ordering::Relaxed));
        let k_div_t = eval_k_div_t_radiative(star.mstar, m_comp, r_phot, osep);
        let r_gyr_sq = eval_star_r_gyr_sq_with_mr(&star, star.mstar, r_phot);
        let ts_expected = eval_tsync(
            k_div_t,
            m_comp / star.mstar,
            r_gyr_sq,
            r_phot,
            osep,
            1.0,
        );
        assert!(
            (ts_recorded - ts_expected).abs() / ts_expected < 1.0e-14,
            "expected t_sync={ts_expected:.16e}, got {ts_recorded:.16e}"
        );
        assert!(star.aux.extra_jdot.iter().all(|&ej| ej == 0.0));
    }

    #[test]
    fn test_sync_spin_orbit_torque_uses_model_dt_and_start_baselines() {
        let period = 1.0 * SECDAY;
        let omega_orb = 2.0 * PI / period;
        let mut star = rotating_zams_star(1.0, 0.8 * omega_orb);
        let m_start = star.mstar;
        let r_start = star_photosphere_radius_cm(&star);
        let om_surf_start = eval_surf_avg_rotation_info(&star).omega;
        let m_comp = 0.8 * MSUN;
        let osep = 5.1 * RSUN;
        let ecc = 0.2;
        let scratch = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let hook = BinarySpinOrbitTorqueHook {
            do_tidal_sync: true,
            doing_first_model_of_run: false,
            is_star_1: false,
            model_twins_flag: false,
            separation: osep,
            period,
            eccentricity: ecc,
            rlr: 0.4 * osep,
            m_star: m_start,
            m_companion: m_comp,
            r_phot: r_start,
            omega_avg_surf: om_surf_start,
            f_tid: 1.5,
            sync_type: String::from("Hut_conv"),
            sync_mode: String::from("Uniform"),
            tidal_reduction: 2.0,
            t_sync_cells: scratch.clone(),
        };

        // Mutate star.mstar and star.omega to simulate mid-step adjust_mass / Henyey
        // changes; upstream binary_tides reads start-of-step b% m, b% r, and
        // prior report omega_avg_surf rather than mutating star-side baselines.
        star.mstar *= 0.92;
        for om in &mut star.omega {
            *om *= 1.3;
        }
        let dt_next = 2.0e6;
        star.dt = dt_next;

        hook.apply(&mut star).expect("convective hook succeeds");
        let ts_recorded = f64::from_bits(scratch[1].load(Ordering::Relaxed));
        let (m_env, r_env) =
            eval_convective_envelope_mass_and_size_with_rphot(&star, r_start);
        let l_phot = eval_photosphere_info(&star).l_lsun;
        let k_div_t = eval_k_div_t_convective(
            m_start / MSUN,
            m_env,
            r_start / RSUN,
            r_env,
            l_phot,
            period,
            om_surf_start,
            2.0,
        );
        let r_gyr_sq = eval_star_r_gyr_sq_with_mr(&star, m_start, r_start);
        let ts_expected =
            eval_tsync(k_div_t, m_comp / m_start, r_gyr_sq, r_start, osep, 1.5);
        assert!(
            (ts_recorded - ts_expected).abs() / ts_expected < 1.0e-14,
            "expected t_sync={ts_expected:.16e}, got {ts_recorded:.16e}"
        );

        let a1 = hut_f2(ecc);
        let a2 = pow(1.0 - ecc * ecc, 1.5) * hut_f5(ecc);
        let factor = 1.0 - exp(-a2 * dt_next / ts_expected);
        for k in [0, star.nz / 2, star.nz - 1] {
            let j_sync = omega_orb * star.aux.i_rot[k];
            let dj = factor * (star.j_rot[k] - a1 / a2 * j_sync);
            let expected_ej = -dj / dt_next;
            let actual_ej = star.aux.extra_jdot[k];
            assert!(
                (actual_ej - expected_ej).abs()
                    <= 1.0e-14 * expected_ej.abs().max(1.0),
                "zone {k}: actual={actual_ej:.16e}, expected={expected_ej:.16e}"
            );
        }
    }
}
