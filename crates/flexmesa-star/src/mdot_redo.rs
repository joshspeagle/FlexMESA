// SPDX-License-Identifier: LGPL-3.0-or-later
//! Critical-rotation implicit wind-revision redo loop (`setup_for_implicit_mdot_loop`
//! and `select_mdot_action` in `star/private/evolve.f90:747-1057`, paired with
//! `set_surf_avg_rotation_info` in `star/private/hydro_rotation.f90:593-744`).

use flexmesa_const::{
    CLIGHT, CRAD, FOUR_PI, M_SUN, PI, SECYER, STANDARD_CGRAV,
};

use crate::model::StarModel;
use crate::rotation::conserve_shell_j_rot_and_distortion;

/// Action selected by [`MdotRedoState::select_action`] after a converged Henyey
/// structure-burn-mix trial (`select_mdot_action` in `evolve.f90:788-1057`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectMdotAction {
    /// Accept the current `model.mstar_dot` and proceed to `timestep_controller`.
    Accept,
    /// Restore `model` to start-of-step values, set `model.mstar_dot = new_mdot`,
    /// and re-run `do_adjust_mass_step` + `do_struct_burn_mix` at the same `dt`.
    RedoWithMdot(f64),
    /// Cannot bring $\Omega / \Omega_{\mathrm{crit}}$ below limit at current `dt`;
    /// trigger an outer timestep-reduction retry.
    ForceRetry,
}

/// Evaluates optical-depth mass-averaged $\langle \Omega / \Omega_{\mathrm{crit}} \rangle_{\mathrm{surf}}$
/// across $[\tau_{\min}, \tau_{\max}]$ (`set_surf_avg_rotation_info` in
/// `star/private/hydro_rotation.f90:593-744`).
pub fn eval_surf_avg_w_div_w_crit(model: &mut StarModel) -> f64 {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz < 2 {
        return 0.0;
    }
    conserve_shell_j_rot_and_distortion(model);

    let tau_min = model.controls.mix_wind.surf_avg_tau_min.max(0.0);
    let tau_max = model.controls.mix_wind.surf_avg_tau.max(tau_min + 1.0e-6);
    let tau0 = (model.controls.tau_factor * model.tau_base).max(0.0);

    let mut tau = tau0;
    let mut dm_sum = 0.0_f64;
    let mut lrad_div_ledd_sum = 0.0_f64;

    let eval_face_lrad_div_ledd = |m: &StarModel, k: usize| -> f64 {
        let j = if k == 0 { 1 } else { k };
        let cgrav_j = m.cgrav[j];
        if cgrav_j <= 0.0 {
            return 0.0;
        }
        let del_m = m.dm_bar.get(j).copied().unwrap_or(m.dm[j]).max(1.0e-30);
        let del_t4 =
            flexmesa_math::pow4(m.t[j - 1]) - flexmesa_math::pow4(m.t[j]);
        let area = FOUR_PI * m.r[j] * m.r[j];
        let denom = (FOUR_PI * cgrav_j * m.m_grav[j].max(1.0)).max(1.0);
        (-(area * area * CRAD * (del_t4 / del_m) / 3.0) / denom).max(0.0)
    };

    for k in 0..(nz - 1) {
        let kap = model.opacity[k].max(1.0e-6);
        let rmid = model
            .rmid
            .get(k)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or_else(|| 0.5 * (model.r[k] + model.r[k + 1]))
            .max(1.0);
        let mut dm = model.dm[k].max(0.0);
        let dtau = (dm * kap / (FOUR_PI * rmid * rmid)).max(1.0e-30);
        if tau + dtau <= tau_min {
            tau += dtau;
            continue;
        }
        if tau < tau_max {
            if tau < tau_min {
                dm *= (tau + dtau - tau_min) / dtau;
            } else if tau + dtau > tau_max {
                dm *= (tau_max - tau) / dtau;
            }
        }
        dm_sum += dm;
        lrad_div_ledd_sum += dm * eval_face_lrad_div_ledd(model, k);
        tau += dtau;
        if tau >= tau_max {
            break;
        }
    }

    let lrad_div_ledd_avg = if dm_sum > 0.0 {
        lrad_div_ledd_sum / dm_sum
    } else {
        eval_face_lrad_div_ledd(model, 0)
    };
    let gamma_factor = 1.0 - lrad_div_ledd_avg.clamp(0.0, 0.9999);

    let has_tidal_def = model.aux.tidal_deformation.is_some();
    let max_w = model.controls.mix_wind.w_div_wcrit_max.clamp(0.0, 0.99);
    let eval_r_eq = |m: &StarModel, idx: usize| -> f64 {
        let r_i = m.r[idx].max(1.0);
        if has_tidal_def {
            return r_i;
        }
        let om_c = (m.cgrav[idx].max(0.0) * m.m_grav[idx].max(1.0)
            / flexmesa_math::pow3(r_i))
        .sqrt()
        .max(1.0e-30);
        let w = (m.omega[idx].abs() / om_c).min(max_w);
        let w2 = w * w;
        let w4 = w2 * w2;
        r_i * (1.0 + w2 / 6.0 - 0.005124 * w4 + 0.06562 * w4 * w2).max(1.0)
    };

    tau = tau0;
    dm_sum = 0.0;
    let mut w_div_wc_sum = 0.0_f64;
    for k in 0..(nz - 1) {
        let kap = model.opacity[k].max(1.0e-6);
        let rmid =
            (0.5 * (eval_r_eq(model, k) + eval_r_eq(model, k + 1))).max(1.0);
        let mut dm = model.dm[k].max(0.0);
        let dtau = (dm * kap / (FOUR_PI * rmid * rmid)).max(1.0e-30);
        if tau + dtau <= tau_min {
            tau += dtau;
            continue;
        }
        if tau < tau_max {
            if tau < tau_min {
                dm *= (tau + dtau - tau_min) / dtau;
            } else if tau + dtau > tau_max {
                dm *= (tau_max - tau) / dtau;
            }
        }
        let cgrav = 0.5 * (model.cgrav[k] + model.cgrav[k + 1]);
        let mmid = (0.5 * (model.m_grav[k] + model.m_grav[k + 1])).max(1.0);
        let om = 0.5 * (model.omega[k] + model.omega[k + 1]);
        let om_crit = (gamma_factor * cgrav * mmid / (rmid * rmid * rmid))
            .sqrt()
            .max(1.0e-30);
        dm_sum += dm;
        w_div_wc_sum += dm * (om / om_crit).abs();
        tau += dtau;
        if tau >= tau_max {
            break;
        }
    }

    if dm_sum > 0.0 {
        w_div_wc_sum / dm_sum
    } else {
        let r0 = eval_r_eq(model, 0);
        let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
        let om_crit = (gamma_factor * cgrav * model.m_grav[0].max(1.0)
            / (r0 * r0 * r0))
            .sqrt()
            .max(1.0e-30);
        (model.omega[0] / om_crit).abs()
    }
}

/// Per-attempt state machine for `max_mdot_redo_cnt` rotational mass-loss
/// iteration (`evolve.f90:747-1057`).
#[derive(Debug, Clone)]
pub struct MdotRedoState {
    ignored_first_step: bool,
    mstar_dot_tracked: f64,
    w_div_w_crit: f64,
    surf_limit: f64,
    mdot_redo_cnt: usize,
    max_mdot_redo_cnt: usize,
    max_wind_mdot: f64,
    have_too_large_wind_mdot: bool,
    have_too_small_wind_mdot: bool,
    too_large_wind_mdot: f64,
    too_small_wind_mdot: f64,
    explicit_mdot: f64,
    abs_mstar_delta: f64,
    force_accept: bool,
}

impl MdotRedoState {
    /// Initializes the redo state before the first `do_adjust_mass_step` try of a
    /// timestep (`setup_for_implicit_mdot_loop` in `evolve.f90:747-785`).
    pub fn setup(model: &mut StarModel) -> Self {
        let surf_limit = model
            .controls
            .surf_omega_div_omega_crit_limit
            .unwrap_or(0.99);
        let max_cnt = model.controls.mix_wind.max_mdot_redo_cnt;
        let explicit_mdot = model.mstar_dot;
        let mdot_old = model.aux.step_hist.mstar_dot_old;
        let was_in_limit = model.aux.step_hist.was_in_implicit_wind_limit;
        let rev_fac = model.controls.mix_wind.mdot_revise_factor.max(1.001);

        let mut ignored_first_step = false;
        let mut mdot_redo_cnt = 0_usize;
        if max_cnt > 0
            && was_in_limit
            && mdot_old.abs() > 0.0
            && ((model.mstar_dot - mdot_old) / mdot_old).abs() + 1.0 > rev_fac
        {
            model.mstar_dot = mdot_old;
            mdot_redo_cnt = 1;
            ignored_first_step = true;
        }

        Self {
            ignored_first_step,
            mstar_dot_tracked: 0.0,
            w_div_w_crit: -1.0,
            surf_limit,
            mdot_redo_cnt,
            max_mdot_redo_cnt: max_cnt,
            max_wind_mdot: 10.0 * M_SUN / SECYER,
            have_too_large_wind_mdot: false,
            have_too_small_wind_mdot: false,
            too_large_wind_mdot: 0.0,
            too_small_wind_mdot: 0.0,
            explicit_mdot,
            abs_mstar_delta: 0.0,
            force_accept: false,
        }
    }

    /// Returns the most recently evaluated surface-averaged $\Omega / \Omega_{\mathrm{crit}}$.
    #[must_use]
    pub fn last_w_div_w_crit(&self) -> f64 {
        self.w_div_w_crit
    }

    /// Marks the next converged fallback trial for immediate acceptance.
    pub fn force_accept_next(&mut self) {
        self.force_accept = true;
    }

    /// Records a Henyey failure on a boosted wind trial (`failed_mdot < 0`) as an
    /// upper wind bound and returns a bisected `mstar_dot` trial when possible.
    pub fn note_solver_failure(
        &mut self,
        failed_mdot: f64,
        tol: f64,
    ) -> Option<f64> {
        let w_fail = -failed_mdot;
        if self.ignored_first_step
            && !self.have_too_small_wind_mdot
            && w_fail > -self.explicit_mdot
        {
            self.ignored_first_step = false;
            self.mdot_redo_cnt = 0;
            self.have_too_large_wind_mdot = true;
            self.too_large_wind_mdot = w_fail;
            return Some(self.explicit_mdot);
        }
        if !self.have_too_small_wind_mdot || w_fail <= self.too_small_wind_mdot
        {
            return None;
        }
        self.too_large_wind_mdot = if self.have_too_large_wind_mdot {
            self.too_large_wind_mdot.min(w_fail)
        } else {
            w_fail
        };
        self.have_too_large_wind_mdot = true;
        self.mdot_redo_cnt += 1;
        if self.mdot_redo_cnt >= self.max_mdot_redo_cnt {
            return None;
        }
        if (self.too_large_wind_mdot - self.too_small_wind_mdot)
            > tol * self.too_large_wind_mdot.abs().max(1.0e-30)
        {
            Some(-0.5 * (self.too_large_wind_mdot + self.too_small_wind_mdot))
        } else {
            None
        }
    }

    /// Evaluates `select_mdot_action` (`evolve.f90:788-1057`) on a converged
    /// post-Henyey trial at `dt` (s).
    pub fn select_action(
        &mut self,
        model: &mut StarModel,
        dt: f64,
    ) -> SelectMdotAction {
        if model.mstar_dot == 0.0
            || self.max_mdot_redo_cnt == 0
            || !model.controls.rotation_flag
        {
            return SelectMdotAction::Accept;
        }
        if self.force_accept {
            self.force_accept = false;
            model.aux.step_hist.was_in_implicit_wind_limit = true;
            return SelectMdotAction::Accept;
        }

        let mstar_dot_prev = self.mstar_dot_tracked;
        self.mstar_dot_tracked = model.mstar_dot;
        let mstar_dot = model.mstar_dot;
        let wind_mdot = -mstar_dot;

        if self.mdot_redo_cnt == 0
            || (self.mdot_redo_cnt == 1 && self.ignored_first_step)
        {
            let l1 = model.lum[0].max(1.0);
            let teff = if model.teff > 0.0 {
                model.teff
            } else {
                flexmesa_math::pow(
                    l1 / (PI
                        * flexmesa_math::pow2(model.r[0].max(1.0))
                        * CRAD
                        * CLIGHT),
                    0.25,
                )
            };
            let r_phot = (l1
                / (PI * CRAD * CLIGHT * flexmesa_math::pow4(teff)))
            .sqrt()
            .max(1.0);
            let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
            let kh_timescale = (0.75 * cgrav * model.mstar * model.mstar
                / (r_phot * l1))
                .max(1.0);
            let dmskhf =
                model.controls.mix_wind.rotational_mdot_kh_fac.max(0.0);
            let dmsfac =
                model.controls.mix_wind.rotational_mdot_boost_fac.max(1.0);
            self.max_wind_mdot = dmskhf * model.mstar.max(1.0) / kh_timescale;
            if wind_mdot > 0.0 {
                self.max_wind_mdot = self.max_wind_mdot.min(wind_mdot * dmsfac);
            }
        }

        let w_prev = self.w_div_w_crit;
        self.w_div_w_crit = eval_surf_avg_w_div_w_crit(model);
        let w = self.w_div_w_crit;
        // Used as given, with no floor
        // (`star/private/evolve.f90:943,1004`).
        let tol = model.controls.mix_wind.surf_omega_div_omega_crit_tol;

        if wind_mdot >= self.max_wind_mdot {
            if (self.mdot_redo_cnt > 0 || w > self.surf_limit)
                && dt >= 0.1 * SECYER
            {
                return SelectMdotAction::ForceRetry;
            }
            model.aux.step_hist.was_in_implicit_wind_limit =
                w > self.surf_limit;
            return SelectMdotAction::Accept;
        }

        if w <= self.surf_limit && self.mdot_redo_cnt == 0 {
            model.aux.step_hist.was_in_implicit_wind_limit = false;
            return SelectMdotAction::Accept;
        }
        if w <= self.surf_limit
            && (model.mstar_dot - self.explicit_mdot).abs() == 0.0
        {
            model.aux.step_hist.was_in_implicit_wind_limit = false;
            return SelectMdotAction::Accept;
        }

        model.aux.step_hist.was_in_implicit_wind_limit = true;
        if dt / SECYER < model.controls.mix_wind.min_years_dt_for_redo_mdot {
            return SelectMdotAction::Accept;
        }
        if w <= self.surf_limit && (self.surf_limit - w) <= tol {
            return SelectMdotAction::Accept;
        }

        self.mdot_redo_cnt += 1;
        let rev_fac = model.controls.mix_wind.mdot_revise_factor.max(1.001);

        if self.mdot_redo_cnt == 1 {
            self.have_too_small_wind_mdot = true;
            self.too_small_wind_mdot = wind_mdot;
            let mut next = if mstar_dot < 0.0 {
                mstar_dot * rev_fac
            } else {
                mstar_dot / rev_fac
            };
            if -next > self.max_wind_mdot {
                next = -self.max_wind_mdot;
            }
            self.abs_mstar_delta = next.abs();
            return SelectMdotAction::RedoWithMdot(next);
        } else if self.mdot_redo_cnt == 2 && self.ignored_first_step {
            self.abs_mstar_delta = model.aux.step_hist.mstar_dot_old.abs();
        }

        if w <= self.surf_limit
            && self.have_too_small_wind_mdot
            && wind_mdot.abs() > 0.0
            && ((wind_mdot - self.too_small_wind_mdot) / wind_mdot).abs() < tol
        {
            return SelectMdotAction::Accept;
        }

        if self.mdot_redo_cnt >= self.max_mdot_redo_cnt
            || (w > self.surf_limit
                && w_prev >= self.surf_limit
                && -mstar_dot >= self.max_wind_mdot)
        {
            if dt < 0.1 * SECYER {
                return SelectMdotAction::Accept;
            }
            return SelectMdotAction::ForceRetry;
        }

        if w >= self.surf_limit {
            if !self.have_too_small_wind_mdot
                || wind_mdot > self.too_small_wind_mdot
            {
                self.too_small_wind_mdot = wind_mdot;
                self.have_too_small_wind_mdot = true;
            }
        } else if !self.have_too_large_wind_mdot
            || wind_mdot < self.too_large_wind_mdot
        {
            self.too_large_wind_mdot = wind_mdot;
            self.have_too_large_wind_mdot = true;
        }

        let next =
            if self.have_too_large_wind_mdot && self.have_too_small_wind_mdot {
                if self.too_large_wind_mdot.abs() > 0.0
                    && ((self.too_large_wind_mdot - self.too_small_wind_mdot)
                        / self.too_large_wind_mdot)
                        .abs()
                        < tol
                {
                    -self.too_large_wind_mdot
                } else {
                    -0.5 * (self.too_large_wind_mdot + self.too_small_wind_mdot)
                }
            } else {
                let boost = model.controls.mix_wind.implicit_mdot_boost;
                let mut nxt = if self.have_too_small_wind_mdot {
                    if self.mdot_redo_cnt % 2 == 1 {
                        mstar_dot - boost * self.abs_mstar_delta
                    } else if mstar_dot < 0.0 {
                        mstar_dot * rev_fac
                    } else {
                        mstar_dot / rev_fac
                    }
                } else if self.mdot_redo_cnt % 2 == 1 {
                    mstar_dot + boost * self.abs_mstar_delta
                } else if mstar_dot < 0.0 {
                    mstar_dot / rev_fac
                } else {
                    mstar_dot * rev_fac
                };
                if (mstar_dot_prev - self.explicit_mdot).abs() > 0.0 {
                    nxt = nxt.min(self.explicit_mdot);
                }
                if -nxt > self.max_wind_mdot {
                    nxt = -self.max_wind_mdot;
                }
                nxt
            };
        SelectMdotAction::RedoWithMdot(next)
    }
}
