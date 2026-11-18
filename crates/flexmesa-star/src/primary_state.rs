// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compact rollback buffer ([`PrimaryStateBuffer`]) storing evolved stellar
//! state without cloning controls, derived microphysics tables, net Jacobians,
//! AD partial vectors, or retry counters.

use std::mem::size_of_val;

use flexmesa_chem::NUM_CATEGORIES;
use flexmesa_const::MixingType;
use flexmesa_eos::EosConfig;
use flexmesa_kap::KapConfig;

use crate::controls::{StarError, VarLayout};
use crate::hydro_state::{StepHistoryScalars, TidalDeformationHandle};
use crate::model::StarModel;
use crate::rsp_def::RspState;
use crate::star_ad::{StarAdExt, StarAdOrder1};

macro_rules! copy_scalars {
    ($dst:expr, $src:expr, [ $($field:ident),+ $(,)? ]) => {
        $( $dst.$field = $src.$field; )+
    };
}

macro_rules! clone_vecs {
    ($dst:expr, $src:expr, [ $($field:ident),+ $(,)? ]) => {
        $( $dst.$field.clone_from(&$src.$field); )+
    };
}

macro_rules! sum_vec_bytes {
    ($obj:expr, [ $($field:ident),+ $(,)? ]) => {{
        let mut total = 0_usize;
        $( total += size_of_val($obj.$field.as_slice()); )+
        total
    }};
}

fn model_scalar_copy_to_buf(dst: &mut PrimaryStateBuffer, src: &StarModel) {
    copy_scalars!(dst, src, [
        nz, species, eos_config, model_number, star_age, star_age_old, dt,
        dt_next, dt_old, dt_limit_ratio_old, mstar, mstar_start, mstar_old,
        xmstar, mstar_dot, m_center, r_center, l_center, v_center, tau_base,
        teff
    ]);
}

fn model_scalar_copy_from_buf(dst: &mut StarModel, src: &PrimaryStateBuffer) {
    copy_scalars!(dst, src, [
        eos_config, model_number, star_age, star_age_old, dt, dt_next, dt_old,
        dt_limit_ratio_old, mstar, mstar_start, mstar_old, xmstar, mstar_dot,
        m_center, r_center, l_center, v_center, tau_base, teff
    ]);
}

fn rsp_heap_bytes(rsp: &RspState) -> usize {
    sum_vec_bytes!(rsp, [
        et_rsp, erad_rsp, fr_rsp, lc_rsp, lt_rsp, ptrb_rsp, pvsc_rsp,
        egas_rsp, hp_face, vol_rsp, f_edd_rsp, y_face_rsp, pii_rsp, chi_rsp,
        coupl_rsp, eq_rsp, qq_rsp, source_rsp, damp_rsp, dampr_rsp, vol_start,
        w_start, fr_start, erad_start, egas_start, pgas_start, prad_start,
        opacity_start, qq_start, lc_start, lt_start, t_start_rsp, r_start_rsp,
        v_start_rsp, csound_start_rsp, pvsc_start, ptrb_start, chi_start,
        coupl_start, ppp0, ppq0, ppt0, ppc0, vv0, work, workq, workt, workc
    ])
}

/// Computes live heap bytes of a [`StarModel`]'s primary, grid, mixing, aux,
/// and `RSP` fields (excluding `aux.extra_grav`, counted separately in
/// [`StarModel::heap_bytes`]) with zero heap allocations.
pub(crate) fn primary_heap_bytes(model: &StarModel) -> usize {
    let model_vecs = sum_vec_bytes!(model, [
        lnd, ln_t, ln_r, lum, v, u, omega, j_rot, xa, xh, lnd_start,
        ln_t_start, ln_r_start, lum_start, v_start, u_start, ln_pgas_start,
        ln_peos_start, ln_e_start, ln_s_start, energy_start, peos_start,
        non_nuc_neu_start, xa_start, xa_sub_xa_start, xh_start, lnd_old,
        ln_t_old, ln_r_old, lum_old, v_old, u_old, xa_old, dq_old, dq, q, xq,
        dm, dm_bar, m, m_grav, cgrav, grav, r, rmid, energy, eps_nuc,
        dxdt_nuc, eps_mdot, gradt, gradr, gradl, scale_height, mlt_vc, mlt_d,
        mlt_cdc, mlt_gamma, d_mix, conv_vel, sig, brunt_b, brunt_n2, chem_id,
        mixing_type
    ]);
    let aux = &model.aux;
    let aux_vecs = sum_vec_bytes!(aux, [
        cp_start, grada_start, chi_t_start, chi_rho_start, burn_avg_epsnuc,
        extra_heat, d_extra_heat_dlnd, d_extra_heat_dlnt,
        eps_wd_sedimentation, eps_diffusion, eps_pre_mix, eps_phase_separation,
        irradiation_heat, etrb, etrb_start, et_rsp_old, hp_face_old,
        csound_start, mlt_vc_old, eta_rti, etamid_rti, sig_rti, sigmid_rti,
        dpdr_drhodr_info, source_plus_alpha_rti, source_minus_alpha_rti,
        alpha_rti, alpha_rti_start, alpha_rti_old, u_face_start, p_face_start,
        w_div_wc, fp_rot, d_fp_rot_dlnr, ft_rot, d_ft_rot_dlnr, i_rot,
        d_i_rot_dlnr, brunt_n2_composition_term, smoothed_brunt_b,
        mass_correction, omega_old, j_rot_old, eps_mdot_old,
        eps_nuc_categories, fixed_gradr_sticky, mlt_prev_conv, mlt_flip_count
    ]);
    let name_bytes: usize = model.iso_names.iter().map(String::len).sum();
    model_vecs + aux_vecs + name_bytes + rsp_heap_bytes(&aux.rsp)
}

/// Compact snapshot of a [`StarModel`]'s evolved primary, start, old, grid,
/// mixing, rotation, RTI, and `RSP` state for remesh smoothing and step/trial
/// rollback without cloning [`crate::controls::StarControls`],
/// [`crate::job_controls::StarJobControls`], `net_iso`,
/// derived EOS/kap/neu/net-Jacobian buffers, `StarAdOrder1` vectors, micro
/// caches, or cumulative diagnostic counters.
#[derive(Debug, Clone)]
pub struct PrimaryStateBuffer {
    /// Number of active zones (`nz`).
    pub nz: usize,
    /// Number of active isotopes (`species`).
    pub species: usize,
    /// 1-based `chem_id` list.
    pub chem_id: Vec<usize>,
    /// Isotope names.
    pub iso_names: Vec<String>,
    /// Variable/equation layout.
    pub layout: VarLayout,
    /// EOS configuration.
    pub eos_config: EosConfig,
    /// Opacity configuration.
    pub kap_config: KapConfig,

    pub(crate) model_number: usize,
    pub(crate) star_age: f64,
    pub(crate) star_age_old: f64,
    pub(crate) dt: f64,
    pub(crate) dt_next: f64,
    pub(crate) dt_old: f64,
    pub(crate) dt_limit_ratio_old: f64,
    pub(crate) mstar: f64,
    pub(crate) mstar_start: f64,
    pub(crate) mstar_old: f64,
    pub(crate) xmstar: f64,
    pub(crate) mstar_dot: f64,
    pub(crate) m_center: f64,
    pub(crate) r_center: f64,
    pub(crate) l_center: f64,
    pub(crate) v_center: f64,
    pub(crate) tau_base: f64,
    pub(crate) teff: f64,

    pub(crate) lnd: Vec<f64>,
    pub(crate) ln_t: Vec<f64>,
    pub(crate) ln_r: Vec<f64>,
    pub(crate) lum: Vec<f64>,
    pub(crate) v: Vec<f64>,
    pub(crate) u: Vec<f64>,
    pub(crate) omega: Vec<f64>,
    pub(crate) j_rot: Vec<f64>,
    pub(crate) xa: Vec<f64>,
    pub(crate) xh: Vec<f64>,
    pub(crate) lnd_start: Vec<f64>,
    pub(crate) ln_t_start: Vec<f64>,
    pub(crate) ln_r_start: Vec<f64>,
    pub(crate) lum_start: Vec<f64>,
    pub(crate) v_start: Vec<f64>,
    pub(crate) u_start: Vec<f64>,
    pub(crate) ln_pgas_start: Vec<f64>,
    pub(crate) ln_peos_start: Vec<f64>,
    pub(crate) ln_e_start: Vec<f64>,
    pub(crate) ln_s_start: Vec<f64>,
    pub(crate) energy_start: Vec<f64>,
    pub(crate) peos_start: Vec<f64>,
    pub(crate) non_nuc_neu_start: Vec<f64>,
    pub(crate) xa_start: Vec<f64>,
    pub(crate) xa_sub_xa_start: Vec<f64>,
    pub(crate) xh_start: Vec<f64>,
    pub(crate) lnd_old: Vec<f64>,
    pub(crate) ln_t_old: Vec<f64>,
    pub(crate) ln_r_old: Vec<f64>,
    pub(crate) lum_old: Vec<f64>,
    pub(crate) v_old: Vec<f64>,
    pub(crate) u_old: Vec<f64>,
    pub(crate) xa_old: Vec<f64>,
    pub(crate) dq_old: Vec<f64>,
    pub(crate) dq: Vec<f64>,
    pub(crate) q: Vec<f64>,
    pub(crate) xq: Vec<f64>,
    pub(crate) dm: Vec<f64>,
    pub(crate) dm_bar: Vec<f64>,
    pub(crate) m: Vec<f64>,
    pub(crate) m_grav: Vec<f64>,
    pub(crate) cgrav: Vec<f64>,
    pub(crate) grav: Vec<f64>,
    pub(crate) r: Vec<f64>,
    pub(crate) rmid: Vec<f64>,
    pub(crate) energy: Vec<f64>,
    pub(crate) eps_nuc: Vec<f64>,
    pub(crate) dxdt_nuc: Vec<f64>,
    pub(crate) eps_mdot: Vec<f64>,
    pub(crate) gradt: Vec<f64>,
    pub(crate) gradr: Vec<f64>,
    pub(crate) gradl: Vec<f64>,
    pub(crate) scale_height: Vec<f64>,
    pub(crate) mlt_vc: Vec<f64>,
    pub(crate) mlt_d: Vec<f64>,
    pub(crate) mlt_cdc: Vec<f64>,
    pub(crate) mlt_gamma: Vec<f64>,
    pub(crate) d_mix: Vec<f64>,
    pub(crate) conv_vel: Vec<f64>,
    pub(crate) sig: Vec<f64>,
    pub(crate) mixing_type: Vec<MixingType>,
    pub(crate) brunt_b: Vec<f64>,
    pub(crate) brunt_n2: Vec<f64>,

    pub(crate) doing_relax: bool,
    pub(crate) doing_struct_burn_mix: bool,
    pub(crate) total_energy: f64,
    pub(crate) total_internal_energy_start: f64,
    pub(crate) cumulative_energy_error: f64,
    pub(crate) non_epsnuc_energy_change_from_split_burn: f64,
    pub(crate) cp_start: Vec<f64>,
    pub(crate) grada_start: Vec<f64>,
    pub(crate) chi_t_start: Vec<f64>,
    pub(crate) chi_rho_start: Vec<f64>,
    pub(crate) burn_avg_epsnuc: Vec<f64>,
    pub(crate) eps_nuc_categories: Vec<[f64; NUM_CATEGORIES]>,
    pub(crate) extra_heat: Vec<f64>,
    pub(crate) d_extra_heat_dlnd: Vec<f64>,
    pub(crate) d_extra_heat_dlnt: Vec<f64>,
    pub(crate) eps_wd_sedimentation: Vec<f64>,
    pub(crate) eps_diffusion: Vec<f64>,
    pub(crate) eps_pre_mix: Vec<f64>,
    pub(crate) eps_phase_separation: Vec<f64>,
    pub(crate) irradiation_heat: Vec<f64>,
    pub(crate) etrb: Vec<f64>,
    pub(crate) etrb_start: Vec<f64>,
    pub(crate) et_rsp_old: Vec<f64>,
    pub(crate) hp_face_old: Vec<f64>,
    pub(crate) csound_start: Vec<f64>,
    pub(crate) mlt_vc_old: Vec<f64>,
    pub(crate) eta_rti: Vec<f64>,
    pub(crate) etamid_rti: Vec<f64>,
    pub(crate) sig_rti: Vec<f64>,
    pub(crate) sigmid_rti: Vec<f64>,
    pub(crate) dpdr_drhodr_info: Vec<f64>,
    pub(crate) source_plus_alpha_rti: Vec<f64>,
    pub(crate) source_minus_alpha_rti: Vec<f64>,
    pub(crate) alpha_rti: Vec<f64>,
    pub(crate) alpha_rti_start: Vec<f64>,
    pub(crate) alpha_rti_old: Vec<f64>,
    pub(crate) u_face_start: Vec<f64>,
    pub(crate) p_face_start: Vec<f64>,
    pub(crate) w_div_wc: Vec<f64>,
    pub(crate) fp_rot: Vec<f64>,
    pub(crate) d_fp_rot_dlnr: Vec<f64>,
    pub(crate) ft_rot: Vec<f64>,
    pub(crate) d_ft_rot_dlnr: Vec<f64>,
    pub(crate) i_rot: Vec<f64>,
    pub(crate) d_i_rot_dlnr: Vec<f64>,
    pub(crate) brunt_n2_composition_term: Vec<f64>,
    pub(crate) smoothed_brunt_b: Vec<f64>,
    pub(crate) smoothed_brunt_b_hash: u64,
    pub(crate) mass_correction: Vec<f64>,
    pub(crate) omega_old: Vec<f64>,
    pub(crate) j_rot_old: Vec<f64>,
    pub(crate) eps_mdot_old: Vec<f64>,
    pub(crate) solver_mlt_eval_count: usize,
    pub(crate) solver_had_mlt_flip: bool,
    pub(crate) fixed_gradr_sticky: Vec<bool>,
    pub(crate) mlt_prev_conv: Vec<u8>,
    pub(crate) mlt_flip_count: Vec<u8>,
    pub(crate) step_hist: StepHistoryScalars,
    pub(crate) rsp: RspState,
    pub(crate) tidal_deformation: Option<TidalDeformationHandle>,
    pub(crate) extra_grav_if_nonzero: Option<Vec<StarAdOrder1>>,
}

macro_rules! make_empty_buf {
    (vecs: [$($v:ident),+ $(,)?], zeros: [$($z:ident),+ $(,)?], falses: [$($b:ident),+ $(,)?]) => {
        Self {
            nz: 0,
            species: 0,
            layout: VarLayout::new(0, false, false),
            eos_config: EosConfig::default(),
            kap_config: KapConfig::default(),
            dt_limit_ratio_old: 1.0,
            step_hist: StepHistoryScalars::default(),
            rsp: RspState::new(0),
            tidal_deformation: None,
            extra_grav_if_nonzero: None,
            model_number: 0,
            smoothed_brunt_b_hash: 0,
            solver_mlt_eval_count: 0,
            $( $v: Vec::new(), )+
            $( $z: 0.0, )+
            $( $b: false, )+
        }
    };
}

impl PrimaryStateBuffer {
    fn empty() -> Self {
        make_empty_buf!(
            vecs: [
                chem_id, iso_names, lnd, ln_t, ln_r, lum, v, u, omega, j_rot,
                xa, xh, lnd_start, ln_t_start, ln_r_start, lum_start, v_start,
                u_start, ln_pgas_start, ln_peos_start, ln_e_start, ln_s_start,
                energy_start, peos_start, non_nuc_neu_start, xa_start,
                xa_sub_xa_start, xh_start, lnd_old, ln_t_old, ln_r_old, lum_old,
                v_old, u_old, xa_old, dq_old, dq, q, xq, dm, dm_bar, m, m_grav,
                cgrav, grav, r, rmid, energy, eps_nuc, dxdt_nuc, eps_mdot, gradt,
                gradr, gradl, scale_height, mlt_vc, mlt_d, mlt_cdc, mlt_gamma,
                d_mix, conv_vel, sig, mixing_type, brunt_b, brunt_n2, cp_start,
                grada_start, chi_t_start, chi_rho_start, burn_avg_epsnuc,
                eps_nuc_categories, extra_heat, d_extra_heat_dlnd,
                d_extra_heat_dlnt, eps_wd_sedimentation, eps_diffusion,
                eps_pre_mix, eps_phase_separation, irradiation_heat, etrb,
                etrb_start, et_rsp_old, hp_face_old, csound_start, mlt_vc_old,
                eta_rti, etamid_rti, sig_rti, sigmid_rti, dpdr_drhodr_info,
                source_plus_alpha_rti, source_minus_alpha_rti, alpha_rti,
                alpha_rti_start, alpha_rti_old, u_face_start, p_face_start,
                w_div_wc, fp_rot, d_fp_rot_dlnr, ft_rot, d_ft_rot_dlnr, i_rot,
                d_i_rot_dlnr, brunt_n2_composition_term, smoothed_brunt_b,
                mass_correction, omega_old, j_rot_old, eps_mdot_old,
                fixed_gradr_sticky, mlt_prev_conv, mlt_flip_count
            ],
            zeros: [
                star_age, star_age_old, dt, dt_next, dt_old, mstar, mstar_start,
                mstar_old, xmstar, mstar_dot, m_center, r_center, l_center,
                v_center, tau_base, teff, total_energy,
                total_internal_energy_start, cumulative_energy_error,
                non_epsnuc_energy_change_from_split_burn
            ],
            falses: [doing_relax, doing_struct_burn_mix, solver_had_mlt_flip]
        )
    }

    /// Captures a new [`PrimaryStateBuffer`] from `model`.
    #[must_use]
    pub fn from_model(model: &StarModel) -> Self {
        let mut buf = Self::empty();
        buf.capture_from(model);
        buf
    }

    /// Overwrites this buffer in place from `model`, reusing existing heap
    /// capacities across retries and implicit trials.
    pub fn capture_from(&mut self, model: &StarModel) {
        model_scalar_copy_to_buf(self, model);
        clone_vecs!(self, model, [
            chem_id, iso_names, layout, kap_config, lnd, ln_t, ln_r, lum, v, u,
            omega, j_rot, xa, xh, lnd_start, ln_t_start, ln_r_start, lum_start,
            v_start, u_start, ln_pgas_start, ln_peos_start, ln_e_start,
            ln_s_start, energy_start, peos_start, non_nuc_neu_start, xa_start,
            xa_sub_xa_start, xh_start, lnd_old, ln_t_old, ln_r_old, lum_old,
            v_old, u_old, xa_old, dq_old, dq, q, xq, dm, dm_bar, m, m_grav,
            cgrav, grav, r, rmid, energy, eps_nuc, dxdt_nuc, eps_mdot, gradt,
            gradr, gradl, scale_height, mlt_vc, mlt_d, mlt_cdc, mlt_gamma,
            d_mix, conv_vel, sig, mixing_type, brunt_b, brunt_n2
        ]);
        let a = &model.aux;
        copy_scalars!(self, a, [
            doing_relax, doing_struct_burn_mix, total_energy,
            total_internal_energy_start, cumulative_energy_error,
            non_epsnuc_energy_change_from_split_burn, smoothed_brunt_b_hash,
            solver_mlt_eval_count, solver_had_mlt_flip
        ]);
        clone_vecs!(self, a, [
            cp_start, grada_start, chi_t_start, chi_rho_start, burn_avg_epsnuc,
            eps_nuc_categories, extra_heat, d_extra_heat_dlnd,
            d_extra_heat_dlnt, eps_wd_sedimentation, eps_diffusion, eps_pre_mix,
            eps_phase_separation, irradiation_heat, etrb, etrb_start,
            et_rsp_old, hp_face_old, csound_start, mlt_vc_old, eta_rti,
            etamid_rti, sig_rti, sigmid_rti, dpdr_drhodr_info,
            source_plus_alpha_rti, source_minus_alpha_rti, alpha_rti,
            alpha_rti_start, alpha_rti_old, u_face_start, p_face_start,
            w_div_wc, fp_rot, d_fp_rot_dlnr, ft_rot, d_ft_rot_dlnr, i_rot,
            d_i_rot_dlnr, brunt_n2_composition_term, smoothed_brunt_b,
            mass_correction, omega_old, j_rot_old, eps_mdot_old,
            fixed_gradr_sticky, mlt_prev_conv, mlt_flip_count, step_hist, rsp,
            tidal_deformation
        ]);
        let has_nonzero = a
            .extra_grav
            .iter()
            .any(|g| g.val != 0.0 || g.d1.iter().any(|&x| x != 0.0));
        if has_nonzero {
            match &mut self.extra_grav_if_nonzero {
                Some(dst) => dst.clone_from(&a.extra_grav),
                None => {
                    self.extra_grav_if_nonzero = Some(a.extra_grav.clone());
                }
            }
        } else {
            self.extra_grav_if_nonzero = None;
        }
    }

    /// Restores buffered state into `model` in place while preserving `controls`,
    /// `job`, `aux.micro_cache`, and all 6 cumulative solver/retry/redo/cache
    /// counters, then repopulates derived microphysics.
    ///
    /// # Errors
    /// Returns [`StarError`] if microphysics recomputation fails.
    pub fn restore_into(&self, model: &mut StarModel) -> Result<(), StarError> {
        let chem_changed = model.chem_id != self.chem_id;
        let resize_needed = model.nz != self.nz
            || model.species != self.species
            || model.layout != self.layout;
        model.species = self.species;
        clone_vecs!(model, self, [chem_id, iso_names, layout]);
        if chem_changed {
            model.net_iso.fill(None);
            for (j, &cid) in model.chem_id.iter().enumerate() {
                if cid < model.net_iso.len() {
                    model.net_iso[cid] = Some(j);
                }
            }
        }
        if resize_needed {
            model.resize_zones(self.nz);
        } else {
            model.nz = self.nz;
        }
        model_scalar_copy_from_buf(model, self);
        clone_vecs!(model, self, [
            kap_config, lnd, ln_t, ln_r, lum, v, u, omega, j_rot, xa, xh,
            lnd_start, ln_t_start, ln_r_start, lum_start, v_start, u_start,
            ln_pgas_start, ln_peos_start, ln_e_start, ln_s_start, energy_start,
            peos_start, non_nuc_neu_start, xa_start, xa_sub_xa_start, xh_start,
            lnd_old, ln_t_old, ln_r_old, lum_old, v_old, u_old, xa_old, dq_old,
            dq, q, xq, dm, dm_bar, m, m_grav, cgrav, grav, r, rmid, energy, sig
        ]);

        {
            let a = &mut model.aux;
            copy_scalars!(a, self, [
                doing_relax, total_energy, total_internal_energy_start,
                cumulative_energy_error, non_epsnuc_energy_change_from_split_burn
            ]);
            a.doing_struct_burn_mix = false;
            clone_vecs!(a, self, [
                cp_start, grada_start, chi_t_start, chi_rho_start,
                burn_avg_epsnuc, extra_heat, d_extra_heat_dlnd,
                d_extra_heat_dlnt, eps_wd_sedimentation, eps_diffusion,
                eps_pre_mix, eps_phase_separation, irradiation_heat, etrb,
                etrb_start, et_rsp_old, hp_face_old, csound_start, mlt_vc_old,
                eta_rti, etamid_rti, sig_rti, sigmid_rti, dpdr_drhodr_info,
                source_plus_alpha_rti, source_minus_alpha_rti, alpha_rti,
                alpha_rti_start, alpha_rti_old, u_face_start, p_face_start,
                fp_rot, ft_rot, i_rot, brunt_n2_composition_term,
                mass_correction, omega_old, j_rot_old, eps_mdot_old, step_hist,
                rsp, tidal_deformation
            ]);
            a.solver_mlt_eval_count = 0;
            a.solver_had_mlt_flip = false;
            a.fixed_gradr_sticky.fill(false);
            a.mlt_prev_conv.fill(0);
            a.mlt_flip_count.fill(0);
            if let Some(ref eg) = self.extra_grav_if_nonzero {
                a.extra_grav.clone_from(eg);
            } else {
                a.extra_grav.fill(StarAdOrder1::ZERO);
            }
        }

        if self.nz > 0 {
            crate::hydro_vars::set_hydro_vars(model, false, false)?;
        }

        // Restore exact post-step values (e.g., overshoot/predictive mixing,
        // operator-split burn averages, rotation distortion + `grav`, and
        // solver latches) overwritten by `set_hydro_vars` without pre-physics.
        clone_vecs!(model, self, [
            omega, j_rot, grav, eps_nuc, dxdt_nuc, eps_mdot, gradt, gradr,
            gradl, scale_height, mlt_vc, mlt_d, mlt_cdc, mlt_gamma, d_mix,
            conv_vel, sig, mixing_type, brunt_b, brunt_n2
        ]);
        let a = &mut model.aux;
        copy_scalars!(a, self, [
            doing_struct_burn_mix, smoothed_brunt_b_hash,
            solver_mlt_eval_count, solver_had_mlt_flip
        ]);
        clone_vecs!(a, self, [
            eps_nuc_categories, w_div_wc, fp_rot, d_fp_rot_dlnr, ft_rot,
            d_ft_rot_dlnr, i_rot, d_i_rot_dlnr, smoothed_brunt_b,
            fixed_gradr_sticky, mlt_prev_conv, mlt_flip_count
        ]);
        Ok(())
    }

    /// Returns the approximate live heap byte size of this rollback buffer.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let own_vecs = sum_vec_bytes!(self, [
            chem_id, lnd, ln_t, ln_r, lum, v, u, omega, j_rot, xa, xh,
            lnd_start, ln_t_start, ln_r_start, lum_start, v_start, u_start,
            ln_pgas_start, ln_peos_start, ln_e_start, ln_s_start, energy_start,
            peos_start, non_nuc_neu_start, xa_start, xa_sub_xa_start, xh_start,
            lnd_old, ln_t_old, ln_r_old, lum_old, v_old, u_old, xa_old, dq_old,
            dq, q, xq, dm, dm_bar, m, m_grav, cgrav, grav, r, rmid, energy,
            eps_nuc, dxdt_nuc, eps_mdot, gradt, gradr, gradl, scale_height,
            mlt_vc, mlt_d, mlt_cdc, mlt_gamma, d_mix, conv_vel, sig,
            mixing_type, brunt_b, brunt_n2, cp_start, grada_start, chi_t_start,
            chi_rho_start, burn_avg_epsnuc, eps_nuc_categories, extra_heat,
            d_extra_heat_dlnd, d_extra_heat_dlnt, eps_wd_sedimentation,
            eps_diffusion, eps_pre_mix, eps_phase_separation, irradiation_heat,
            etrb, etrb_start, et_rsp_old, hp_face_old, csound_start,
            mlt_vc_old, eta_rti, etamid_rti, sig_rti, sigmid_rti,
            dpdr_drhodr_info, source_plus_alpha_rti, source_minus_alpha_rti,
            alpha_rti, alpha_rti_start, alpha_rti_old, u_face_start,
            p_face_start, w_div_wc, fp_rot, d_fp_rot_dlnr, ft_rot,
            d_ft_rot_dlnr, i_rot, d_i_rot_dlnr, brunt_n2_composition_term,
            smoothed_brunt_b, mass_correction, omega_old, j_rot_old,
            eps_mdot_old, fixed_gradr_sticky, mlt_prev_conv, mlt_flip_count
        ]);
        let name_bytes: usize = self.iso_names.iter().map(String::len).sum();
        let extra_grav = self
            .extra_grav_if_nonzero
            .as_ref()
            .map_or(0, |v| size_of_val(v.as_slice()));
        own_vecs + name_bytes + rsp_heap_bytes(&self.rsp) + extra_grav
    }
}
