// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled `BinarySystem::step` / `BinarySystem::try_step` explicit and implicit
//! RLOF redo loop, single-trial Henyey + RK2 orbital integrator, and binary
//! termination checks (`binary/private/binary_evolve.f90:257-620`).

use flexmesa_const::{MSUN, SECYER};
use flexmesa_star::{
    StarError, StarModel, check_termination_limits, compute_core_masses_msun,
    eval_total_energy_integrals, remesh_star_before_step, set_hydro_vars,
    sync_post_henyey_rotation,
};

use crate::extras::{BinaryExtras, BinaryExtrasAction, DefaultBinaryExtras};
use crate::mdot_implicit::{
    ImplicitMdotSolver, ImplicitStepAction, update_change_factor_after_step,
};
use crate::orbit::{
    eval_l2_overflow_limit, eval_rlobe, period_from_separation, separation_from_j_orb,
};
use crate::system::{BinaryCompanion, BinarySystem, compute_rl_and_gaps};
use crate::system_jdot::{
    JdotBreakdown, apply_tidal_spin_sync, conserve_spin_after_henyey,
    eval_coupled_edot, eval_gated_jdot, eval_star_spin_j,
};
use crate::system_rlof::{
    blend_explicit_mdot, eval_scheme_explicit_mdot, evolve_single_star_without_term_rewind,
    prepare_coupled_mdots_with_extras, star_exceeds_delta_mdot_hard_limit,
};
use crate::timestep::{
    B_TLIM_COMP, BinaryTimestepControls, binary_pick_next_timestep, violates_binary_hard_limits,
};

#[allow(clippy::large_enum_variant)]
enum SingleTrialOutcome {
    Completed { dt_used: f64, ts_ctrl: BinaryTimestepControls },
    CutDt(f64),
}

impl BinarySystem {
    /// Advances the coupled binary system by one timestep `dt` (s) using [`DefaultBinaryExtras`].
    pub fn step(&mut self, dt: f64) -> (f64, i32) {
        self.step_with_extras(dt, &mut DefaultBinaryExtras)
    }

    /// Advances the coupled binary system by one timestep `dt` (s) with custom [`BinaryExtras`].
    pub fn step_with_extras(&mut self, dt: f64, extras: &mut dyn BinaryExtras) -> (f64, i32) {
        self.try_step_with_extras(dt, extras)
            .unwrap_or((0.5 * dt.max(1.0), B_TLIM_COMP))
    }

    /// Advances the coupled binary system by one timestep `dt` (s) using [`DefaultBinaryExtras`],
    /// returning `(next_dt_sec, limit_reason)` or [`StarError`] on unrecoverable solver failure.
    pub fn try_step(&mut self, dt: f64) -> Result<(f64, i32), StarError> {
        self.try_step_with_extras(dt, &mut DefaultBinaryExtras)
    }

    /// Advances the coupled binary system by one timestep `dt` (s) with custom [`BinaryExtras`],
    /// returning `(next_dt_sec, limit_reason)` or [`StarError`] on unrecoverable solver failure.
    pub fn try_step_with_extras(
        &mut self,
        dt: f64,
        extras: &mut dyn BinaryExtras,
    ) -> Result<(f64, i32), StarError> {
        self.controls.mass_transfer_alpha = self.mass_transfer_alpha;
        self.controls.mass_transfer_beta = self.mass_transfer_beta;
        self.controls.mass_transfer_delta = self.mass_transfer_delta;
        self.controls.mass_transfer_gamma = self.mass_transfer_gamma;
        if !self.use_implicit_rlof && self.controls.max_tries_to_achieve > 0 {
            self.controls.max_tries_to_achieve = 0;
        }
        self.last_termination_reason = None;
        self.maybe_switch_donor_at_step_start();

        let pre_step_snap = self.capture_snapshot();
        let mut retry_holds = [(None, 0_usize); 2];
        let s2_min = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => s2.controls.min_timestep_limit,
        };
        let dt_floor = self.donor.controls.min_timestep_limit.max(s2_min).clamp(1.0e-12, 1.0e-2);
        let check_cut_dt = |dt_cut: f64, cur_dt: f64| -> Result<f64, StarError> {
            let next_dt = dt_cut.max(dt_floor);
            if next_dt >= 0.999_999 * cur_dt {
                return Err(StarError::ConvergenceFailure(format!("CutDt hit dt_floor ({dt_cut:.3e} s)")));
            }
            Ok(next_dt)
        };
        let mut dt_try = dt.min(self.max_timestep_sec).max(dt_floor);
        let implicit_mode = !self.ignore_rlof_flag && self.controls.max_tries_to_achieve > 0;
        let max_tries = self.controls.max_tries_to_achieve.max(1) as usize;

        let tracing = flexmesa_star::step_trace::enabled(self.model_number + 1);
        for outer_retry in 0..16_usize {
            if outer_retry > 0 {
                self.record_retry_holds(&mut retry_holds);
                self.restore_snapshot(&pre_step_snap)?;
                self.apply_retry_holds(&retry_holds);
            }
            if tracing {
                eprintln!(
                    "[trace] binary model={} outer_retry={outer_retry} dt_try={dt_try:.4e} \
                     implicit={implicit_mode} donor_i={} mtransfer={:.3e} gaps=[{:+.4e},{:+.4e}]",
                    self.model_number + 1,
                    self.donor_i,
                    self.mtransfer_rate,
                    self.rl_relative_gap[0],
                    self.rl_relative_gap[1],
                );
            }
            if !implicit_mode {
                let pre_mdot = if self.ignore_rlof_flag { 0.0 } else { self.mtransfer_rate };
                match self.run_single_trial(pre_mdot, dt_try, outer_retry == 0, extras) {
                    Ok(SingleTrialOutcome::Completed { dt_used, ts_ctrl }) => {
                        if outer_retry < 12 && dt_used > 1.0 && self.exceeds_hard_limits(&ts_ctrl) {
                            dt_try = 0.5 * dt_used;
                            continue;
                        }
                        self.mtransfer_rate = if self.ignore_rlof_flag {
                            0.0
                        } else {
                            let m_exp = eval_scheme_explicit_mdot(
                                &self.controls.mdot_scheme, &self.donor, self.accretor_mass_g(),
                                self.rl[0], self.separation, self.eccentricity,
                            );
                            blend_explicit_mdot(
                                pre_mdot, m_exp, self.controls.cur_mdot_frac,
                                self.controls.max_explicit_abs_mdot,
                            )
                        };
                        return Ok(self.finish_accepted_step(dt_used, &ts_ctrl, extras));
                    }
                    Ok(SingleTrialOutcome::CutDt(dt_cut)) => {
                        dt_try = check_cut_dt(dt_cut, dt_try)?;
                        continue;
                    }
                    Err(_) if outer_retry < 10 && dt_try > dt_floor => {
                        let cut = if outer_retry == 0 { 0.5 } else { 0.25 };
                        dt_try = (cut * dt_try).max(dt_floor);
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }

            let mut remeshed_outer0 = false;
            if outer_retry == 0 {
                let _ = self.refresh_tidal_deformation_hooks(false);
                if self.donor.controls.mesh_delta_coeff > 0.0 {
                    let (ns, nm, _) = remesh_star_before_step(&mut self.donor, dt_try)?;
                    remeshed_outer0 |= ns + nm > 0;
                }
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    if s2.controls.mesh_delta_coeff > 0.0 {
                        let (ns, nm, _) = remesh_star_before_step(s2, dt_try)?;
                        remeshed_outer0 |= ns + nm > 0;
                    }
                }
                let _ = self.refresh_tidal_deformation_hooks(false);
            }
            let mut trial_snap = self.capture_snapshot();
            let mut solver = ImplicitMdotSolver::from_controls(&self.controls, self.change_factor);
            let mut trial_mdot = if self.mtransfer_rate < 0.0 { self.mtransfer_rate.max(-solver.max_abs_mdot_cgs) } else { 0.0 };
            let mut accepted: Option<(f64, BinaryTimestepControls)> = None;

            for try_idx in 1..=max_tries {
                if try_idx > 1 {
                    self.restore_snapshot(&trial_snap)?;
                }
                let (dt_used, ts_ctrl) = match self.run_single_trial(trial_mdot, dt_try, false, extras) {
                    Ok(SingleTrialOutcome::Completed { dt_used, ts_ctrl }) => (dt_used, ts_ctrl),
                    Ok(SingleTrialOutcome::CutDt(dt_cut)) => {
                        dt_try = check_cut_dt(dt_cut, dt_try)?;
                        break;
                    }
                    Err(e) => {
                        if outer_retry >= 10 || dt_try <= dt_floor {
                            return Err(e);
                        }
                        dt_try = ((if remeshed_outer0 { 0.5 } else { 0.25 }) * dt_try).max(dt_floor);
                        break;
                    }
                };
                if try_idx == 1 && dt_used < 0.999_999 * dt_try {
                    dt_try = dt_used;
                }

                let (m_d, m_a) = (self.donor.mstar, self.accretor_mass_g());
                let m_exp = eval_scheme_explicit_mdot(
                    &self.controls.mdot_scheme, &self.donor, m_a,
                    self.rl[0], self.separation, self.eccentricity,
                );
                let act = solver.evaluate_trial(
                    &self.controls.mdot_scheme, try_idx, max_tries, trial_mdot, m_exp,
                    self.rl_relative_gap[0], self.rl_relative_gap[1], m_a / m_d.max(1.0),
                );
                match act {
                    ImplicitStepAction::Converged(m_conv) => {
                        let m_acc = m_conv.min(0.0);
                        self.mtransfer_rate = m_acc;
                        self.change_factor = update_change_factor_after_step(
                            self.change_factor, pre_step_snap.mtransfer_rate, m_acc, try_idx, &self.controls,
                        );
                        accepted = Some((dt_used, ts_ctrl));
                        break;
                    }
                    ImplicitStepAction::Detached => {
                        if trial_mdot == 0.0 {
                            self.mtransfer_rate = 0.0;
                            accepted = Some((dt_used, ts_ctrl));
                        } else {
                            self.restore_snapshot(&trial_snap)?;
                            if let Ok(SingleTrialOutcome::Completed { dt_used: dt0, ts_ctrl: ts0 }) =
                                self.run_single_trial(0.0, dt_try, false, extras)
                            {
                                self.mtransfer_rate = 0.0;
                                accepted = Some((dt0, ts0));
                            } else {
                                dt_try = (0.5 * dt_try).max(dt_floor);
                            }
                        }
                        break;
                    }
                    ImplicitStepAction::TryNextMdot(next_m) => {
                        if next_m > 0.0 && self.point_mass_i == 0 {
                            self.restore_snapshot(&trial_snap)?;
                            let _ = self.swap_donor_and_accretor();
                            trial_snap = self.capture_snapshot();
                            solver.flip_for_contact_donor_switch();
                            trial_mdot = -next_m;
                        } else {
                            trial_mdot = next_m.min(0.0);
                        }
                    }
                    ImplicitStepAction::ExceededMaxTries => {
                        dt_try = (0.5 * dt_try).max(dt_floor);
                        break;
                    }
                }
            }

            if let Some((dt_used, ts_ctrl)) = accepted {
                let dmdot_hard = outer_retry < 6
                    && (star_exceeds_delta_mdot_hard_limit(&self.donor)
                        || matches!(&self.companion, BinaryCompanion::Star(s2) if star_exceeds_delta_mdot_hard_limit(s2)));
                if outer_retry < 12 && dt_used > 1.0 && (self.exceeds_hard_limits(&ts_ctrl) || dmdot_hard) {
                    dt_try = (0.5 * dt_used).max(dt_floor);
                    continue;
                }
                return Ok(self.finish_accepted_step(dt_used, &ts_ctrl, extras));
            }
        }
        Err(StarError::ConvergenceFailure(String::from(
            "BinarySystem::try_step exhausted outer timestep retries",
        )))
    }

    fn exceeds_hard_limits(&self, ts_ctrl: &BinaryTimestepControls) -> bool {
        if self.model_number == 0 {
            return false;
        }
        violates_binary_hard_limits(
            ts_ctrl, self.controls.fm_hard, self.controls.fr_hard,
            self.controls.fj_hard, self.controls.fa_hard,
            self.controls.fe_hard, self.controls.fdm_hard,
        )
    }

    fn run_single_trial(
        &mut self,
        trial_mdot: f64,
        dt_try: f64,
        allow_remesh: bool,
        extras: &mut dyn BinaryExtras,
    ) -> Result<SingleTrialOutcome, StarError> {
        let _ = self.refresh_tidal_deformation_hooks(false);
        let (m_d_old, m_a_old) = (self.donor.mstar, self.accretor_mass_g());
        let (j_old, a_old, e_old, gap_old) = (
            self.angular_momentum_j, self.separation, self.eccentricity, self.rl_relative_gap,
        );
        let env_d_old = (m_d_old / MSUN - compute_core_masses_msun(&self.donor).0).max(0.0);

        let mut s_pre_d = eval_star_spin_j(&self.donor);
        let j_surf_pre_d = self.donor.j_rot.first().copied().unwrap_or(0.0);
        let (mut s_pre_a, j_surf_pre_a) = match &self.companion {
            BinaryCompanion::PointMass { .. } => (0.0, 0.0),
            BinaryCompanion::Star(s2) => (eval_star_spin_j(s2), s2.j_rot.first().copied().unwrap_or(0.0)),
        };

        let pm_state = match &self.companion {
            BinaryCompanion::PointMass { mass_g, m_bh_zero_eq_g, .. } => Some((*mass_g, *m_bh_zero_eq_g)),
            BinaryCompanion::Star(_) => None,
        };
        let acc_opt = match &mut self.companion {
            BinaryCompanion::PointMass { .. } => None,
            BinaryCompanion::Star(s2) => Some(&mut **s2),
        };
        let mut acct = prepare_coupled_mdots_with_extras(
            &self.controls, &mut self.donor, acc_opt, pm_state, self.donor_i,
            self.rl, self.separation, self.eccentricity, trial_mdot, self.ignore_rlof_flag, extras,
        )?;

        let dt_used =
            evolve_single_star_without_term_rewind(&mut self.donor, dt_try, allow_remesh)?.dt;
        let extra_w_d = (self.donor.mstar_dot - acct.component_mdot[0]).min(0.0);
        if extra_w_d < 0.0 {
            acct.mdot_wind_intrinsic[0] += extra_w_d;
            let dw_tr = extra_w_d * acct.wind_xfer_fraction[0];
            acct.mdot_wind_transfer[0] += dw_tr;
            acct.mdot_system_wind[0] += extra_w_d - dw_tr;
            acct.component_mdot[0] = self.donor.mstar_dot;
        }
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            let rep_a = evolve_single_star_without_term_rewind(s2, dt_used, allow_remesh)?;
            if rep_a.dt < 0.999_999 * dt_used {
                return Ok(SingleTrialOutcome::CutDt(rep_a.dt));
            }
            let extra_w_a = (s2.mstar_dot - acct.component_mdot[1]).min(0.0);
            if extra_w_a < 0.0 {
                acct.mdot_wind_intrinsic[1] += extra_w_a;
                let dw_tr = extra_w_a * acct.wind_xfer_fraction[1];
                acct.mdot_wind_transfer[1] += dw_tr;
                acct.mdot_system_wind[1] += extra_w_a - dw_tr;
                acct.component_mdot[1] = s2.mstar_dot;
            }
        }

        let ds_mdot_d = self.donor.mstar_dot.min(0.0) * j_surf_pre_d * dt_used;
        if self.donor.mstar_dot >= 0.0 {
            conserve_spin_after_henyey(&mut self.donor, s_pre_d, ds_mdot_d);
        } else {
            s_pre_d = eval_star_spin_j(&self.donor) - ds_mdot_d;
        }

        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            let ds_wind_a = acct.mdot_system_wind[1].min(0.0) * j_surf_pre_a * dt_used;
            let ds_acc_j_a = if self.controls.do_j_accretion {
                acct.retained_accretion_rate.max(0.0) * acct.accreted_material_j * dt_used
            } else {
                0.0
            };
            if s2.mstar_dot >= 0.0 {
                conserve_spin_after_henyey(s2, s_pre_a, ds_wind_a + ds_acc_j_a);
            } else {
                s_pre_a = eval_star_spin_j(s2) - (ds_wind_a + ds_acc_j_a);
            }
        }

        if self.controls.do_tidal_sync {
            let (st_d, sm_d, ft_d, st_a, sm_a, ft_a) = if self.donor_i == 1 {
                (
                    self.controls.sync_type_1.as_str(), self.controls.sync_mode_1.as_str(), self.controls.ftid_1,
                    self.controls.sync_type_2.as_str(), self.controls.sync_mode_2.as_str(), self.controls.ftid_2,
                )
            } else {
                (
                    self.controls.sync_type_2.as_str(), self.controls.sync_mode_2.as_str(), self.controls.ftid_2,
                    self.controls.sync_type_1.as_str(), self.controls.sync_mode_1.as_str(), self.controls.ftid_1,
                )
            };
            apply_tidal_spin_sync(
                &mut self.donor, st_d, sm_d, ft_d, self.controls.tidal_reduction,
                m_a_old, a_old, self.period, e_old, dt_used,
            );
            if let BinaryCompanion::Star(ref mut s2) = self.companion {
                apply_tidal_spin_sync(
                    s2, st_a, sm_a, ft_a, self.controls.tidal_reduction,
                    m_d_old, a_old, self.period, e_old, dt_used,
                );
            }
        }

        if let BinaryCompanion::PointMass { ref mut mass_g, .. } = self.companion {
            *mass_g = (m_a_old + acct.retained_accretion_rate * dt_used).max(1.0e20);
        }

        let (m_d_new, m_a_new) = (self.donor.mstar, self.accretor_mass_g());
        let s_post_d = eval_star_spin_j(&self.donor);
        let s_post_a = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => eval_star_spin_j(s2),
        };

        let n_sub = 8_usize;
        let h = dt_used / (n_sub as f64);
        let (mut j_cur, mut e_cur, mut dj_non_ls_accum) = (j_old, e_old, 0.0_f64);
        let mut last_jb = JdotBreakdown::default();
        let acc_ref = match &self.companion {
            BinaryCompanion::PointMass { .. } => None,
            BinaryCompanion::Star(s2) => Some(&**s2),
        };
        let mtr_eff = if self.ignore_rlof_flag { 0.0 } else { trial_mdot.min(0.0) };

        for sub in 0..n_sub {
            let f0 = (sub as f64) / (n_sub as f64);
            let f_mid = ((sub as f64) + 0.5) / (n_sub as f64);
            let (md0, ma0) = (m_d_old + (m_d_new - m_d_old) * f0, m_a_old + (m_a_new - m_a_old) * f0);
            let a0_s = separation_from_j_orb(md0, ma0, j_cur, e_cur);
            let p0_s = period_from_separation(md0, ma0, a0_s);
            let mut mb_tmp = self.using_jdot_mb;
            let jb0 = eval_gated_jdot(
                &self.controls, &self.donor, acc_ref, self.donor_i, md0, ma0,
                eval_rlobe(md0, ma0, a0_s), eval_rlobe(ma0, md0, a0_s), a0_s, p0_s, e_cur, j_cur,
                mtr_eff, acct.mdot_system_transfer[0], acct.mdot_system_wind[0],
                acct.mdot_system_transfer[1], acct.mdot_system_wind[1], acct.mdot_system_cct,
                s_pre_d, s_post_d, j_surf_pre_d, s_pre_a, s_post_a, j_surf_pre_a, dt_used, 0.0, &mut mb_tmp,
            );
            let ed0 = eval_coupled_edot(&self.controls, &self.donor, acc_ref, self.donor_i, md0, ma0, a0_s, p0_s, e_cur);

            let (md_m, ma_m) = (m_d_old + (m_d_new - m_d_old) * f_mid, m_a_old + (m_a_new - m_a_old) * f_mid);
            let j_m = j_cur + 0.5 * jb0.jdot_total * h;
            if j_m <= 0.0 {
                let f_j = self.controls.dt_reduction_factor_for_j.clamp(0.05, 0.5);
                return Ok(SingleTrialOutcome::CutDt(dt_used * f_j));
            }
            let e_m = (e_cur + 0.5 * ed0 * h).clamp(self.controls.min_eccentricity, self.controls.max_eccentricity);
            let a_m = separation_from_j_orb(md_m, ma_m, j_m, e_m);
            let p_m = period_from_separation(md_m, ma_m, a_m);
            let mut mb_mid = self.using_jdot_mb;
            let jb_m = eval_gated_jdot(
                &self.controls, &self.donor, acc_ref, self.donor_i, md_m, ma_m,
                eval_rlobe(md_m, ma_m, a_m), eval_rlobe(ma_m, md_m, a_m), a_m, p_m, e_m, j_m,
                mtr_eff, acct.mdot_system_transfer[0], acct.mdot_system_wind[0],
                acct.mdot_system_transfer[1], acct.mdot_system_wind[1], acct.mdot_system_cct,
                s_pre_d, s_post_d, j_surf_pre_d, s_pre_a, s_post_a, j_surf_pre_a, dt_used, 0.0, &mut mb_mid,
            );
            let ed_m = eval_coupled_edot(&self.controls, &self.donor, acc_ref, self.donor_i, md_m, ma_m, a_m, p_m, e_m);
            if sub + 1 == n_sub {
                self.using_jdot_mb = mb_mid;
            }
            j_cur += jb_m.jdot_total * h;
            if j_cur <= 0.0 {
                let f_j = self.controls.dt_reduction_factor_for_j.clamp(0.05, 0.5);
                return Ok(SingleTrialOutcome::CutDt(dt_used * f_j));
            }
            e_cur = (e_cur + ed_m * h).clamp(self.controls.min_eccentricity, self.controls.max_eccentricity);
            dj_non_ls_accum += (jb_m.jdot_total - jb_m.jdot_ls) * h;
            last_jb = jb_m;
        }

        self.angular_momentum_j = j_cur;
        self.eccentricity = e_cur;
        self.jdot = (j_cur - j_old) / dt_used.max(1.0e-99);
        self.separation = separation_from_j_orb(m_d_new, m_a_new, j_cur, e_cur);
        self.period = period_from_separation(m_d_new, m_a_new, self.separation);
        let (rl_new, gap_new) = compute_rl_and_gaps(
            m_d_new, m_a_new, self.donor.r[0], self.accretor_radius_cm(), self.separation, self.eccentricity,
        );
        self.rl = rl_new;
        self.rl_relative_gap = gap_new;
        self.step_mtransfer_rate = mtr_eff;
        self.last_mdot = acct;
        self.last_jdot = last_jb;
        let _ = self.refresh_tidal_deformation_hooks(false);
        finalize_star_post_binary_torques(&mut self.donor)?;
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            finalize_star_post_binary_torques(s2)?;
        }
        if self.controls.do_jdot_ls && !self.controls.use_tidal_deformation {
            let s_fin_a = match &self.companion {
                BinaryCompanion::PointMass { .. } => 0.0,
                BinaryCompanion::Star(s2) => eval_star_spin_j(s2),
            };
            self.angular_momentum_j -= (eval_star_spin_j(&self.donor) - s_post_d) + (s_fin_a - s_post_a);
        }

        let wind_spin_d = if self.donor.mstar_dot < 0.0 { acct.mdot_system_wind[0] * j_surf_pre_d * dt_used } else { 0.0 };
        let wind_spin_a = match &self.companion {
            BinaryCompanion::Star(_) => acct.mdot_system_wind[1].min(0.0) * j_surf_pre_a * dt_used,
            BinaryCompanion::PointMass { .. } => 0.0,
        };
        let ext_spin_delta = if self.controls.do_jdot_ls {
            wind_spin_d + wind_spin_a
        } else {
            (s_post_d - s_pre_d) + (s_post_a - s_pre_a)
        };
        self.cumulative_j_ext += dj_non_ls_accum + ext_spin_delta;

        let env_d_new = (m_d_new / MSUN - compute_core_masses_msun(&self.donor).0).max(0.0);
        let c = &self.controls;
        let ts_ctrl = BinaryTimestepControls {
            donor_time_step_yr: dt_used / SECYER,
            max_timestep_sec: self.max_timestep_sec,
            time_delta_coeff: c.time_delta_coeff,
            dt_softening_factor: c.dt_softening_factor,
            rl_relative_gap: self.rl_relative_gap,
            rl_relative_gap_old: gap_old,
            angular_momentum_j: self.angular_momentum_j,
            angular_momentum_j_old: j_old,
            separation: self.separation,
            separation_old: a_old,
            eccentricity: self.eccentricity,
            eccentricity_old: e_old,
            env_donor: env_d_new,
            env_donor_old: env_d_old,
            m: [m_d_new, m_a_new],
            m_old: [m_d_old, m_a_old],
            is_point_mass_companion: self.point_mass_i != 0,
            fm: c.fm, fm_limit: c.fm_limit, fr: c.fr, fr_limit: c.fr_limit,
            fr_dt_limit: c.fr_dt_limit, fj: c.fj, fa: c.fa, fe: c.fe,
            fe_limit: c.fe_limit, fdm: c.fdm,
        };
        Ok(SingleTrialOutcome::Completed { dt_used, ts_ctrl })
    }

    fn finish_accepted_step(
        &mut self,
        dt_used: f64,
        ts_ctrl: &BinaryTimestepControls,
        extras: &mut dyn BinaryExtras,
    ) -> (f64, i32) {
        self.binary_age_yr += dt_used / SECYER;
        self.model_number += 1;
        self.update_stage_varcontrol();

        let (bin_max_dt, reason) = if self.model_number == 1 {
            (self.max_timestep_sec, B_TLIM_COMP)
        } else {
            binary_pick_next_timestep(ts_ctrl)
        };
        self.max_timestep_sec = bin_max_dt;
        let star_next = match &self.companion {
            BinaryCompanion::PointMass { .. } => self.donor.dt_next,
            BinaryCompanion::Star(s2) => self.donor.dt_next.min(s2.dt_next),
        };
        let next_dt = star_next.min(bin_max_dt);

        if self.model_number == 1
            && self.controls.terminate_if_initial_overflow
            && !self.ignore_rlof_flag
            && (self.rl_relative_gap[0] >= 0.0
                || (self.point_mass_i == 0 && self.rl_relative_gap[1] >= 0.0))
        {
            self.last_termination_reason = Some(String::from("terminate_if_initial_overflow"));
        } else if self.point_mass_i == 0
            && !self.ignore_rlof_flag
            && self.rl_relative_gap[1] >= 0.0
            && self.rl_relative_gap[1] >= self.controls.accretor_overflow_terminate
        {
            self.last_termination_reason = Some(String::from("accretor_overflow_terminate"));
        } else if self.point_mass_i == 0
            && !self.ignore_rlof_flag
            && self.controls.terminate_if_l2_overflow
        {
            let m_all = self.star_masses_g();
            let gaps = self.rl_relative_gap_by_star();
            let (l2_lim, gap_sec) = if m_all[0] >= m_all[1] {
                (eval_l2_overflow_limit(m_all[0], m_all[1]), gaps[1])
            } else {
                (eval_l2_overflow_limit(m_all[1], m_all[0]), gaps[0])
            };
            if gap_sec > l2_lim {
                self.last_termination_reason = Some(String::from("terminate_if_L2_overflow"));
            }
        }
        if self.last_termination_reason.is_none() {
            self.last_termination_reason =
                check_termination_limits(&self.donor).or_else(|| match &self.companion {
                    BinaryCompanion::PointMass { .. } => None,
                    BinaryCompanion::Star(s2) => check_termination_limits(s2),
                });
        }
        if let BinaryExtrasAction::Terminate(reason_str) = extras.check_model(self) {
            self.last_termination_reason = Some(reason_str);
        } else if let BinaryExtrasAction::Terminate(reason_str) = extras.finish_step(self) {
            self.last_termination_reason = Some(reason_str);
        }
        (next_dt, reason)
    }

    fn record_retry_holds(&self, holds: &mut [(Option<usize>, usize); 2]) {
        let upd = |slot: &mut (Option<usize>, usize), s: &StarModel| {
            let sh = &s.aux.step_hist;
            slot.0 = slot.0.max(sh.model_number_for_last_retry);
            slot.1 = slot.1.max(sh.timestep_hold);
        };
        let d_idx = (self.donor_i - 1).min(1);
        upd(&mut holds[d_idx], &self.donor);
        if let BinaryCompanion::Star(ref s2) = self.companion {
            upd(&mut holds[1 - d_idx], s2);
        }
    }

    fn apply_retry_holds(&mut self, holds: &[(Option<usize>, usize); 2]) {
        let app = |s: &mut StarModel, slot: (Option<usize>, usize)| {
            let sh = &mut s.aux.step_hist;
            sh.model_number_for_last_retry = sh.model_number_for_last_retry.max(slot.0);
            sh.timestep_hold = sh.timestep_hold.max(slot.1);
        };
        let d_idx = (self.donor_i - 1).min(1);
        app(&mut self.donor, holds[d_idx]);
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            app(s2, holds[1 - d_idx]);
        }
    }
}

fn finalize_star_post_binary_torques(star: &mut StarModel) -> Result<(), StarError> {
    star.mstar_start = star.mstar;
    if !star.controls.rsp_flag {
        star.sync_mesh_and_pack()?;
        sync_post_henyey_rotation(star, 0.0);
        set_hydro_vars(star, false, false)?;
        star.aux.total_energy = eval_total_energy_integrals(star).5;
    }
    Ok(())
}
