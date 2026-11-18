// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compact rollback buffer ([`PrimaryStateBuffer`]) storing evolved stellar
//! state without cloning controls, derived microphysics tables, net Jacobians,
//! AD partial vectors, or retry counters.

use std::mem::size_of_val;

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::MixingType;
use flexmesa_eos::EosConfig;
use flexmesa_kap::KapConfig;

use crate::controls::{StarError, VarLayout};
use crate::hydro_state::{StepHistoryScalars, TidalDeformationHandle};
use crate::model::StarModel;
use crate::rsp_def::RspState;
use crate::star_ad::{StarAdExt, StarAdOrder1};

#[inline]
fn f64_vec_heap(v: &[f64]) -> usize {
    size_of_val(v)
}

/// Compact snapshot of a [`StarModel`]'s evolved primary, start, old, grid,
/// mixing, rotation, RTI, and `RSP` state for remesh smoothing and step/trial
/// rollback without cloning [`StarControls`], [`StarJobControls`], `net_iso`,
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

    // --- Scalar Evolutionary State ---
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

    // --- Primary State Vectors ---
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

    // --- Start / Old State Vectors ---
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

    // --- Mesh Geometry & Post-Remesh Energy ---
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

    // --- Post-Step Burning, Mixing, Brunt & Rotation State ---
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

    // --- Aux State ---
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

impl PrimaryStateBuffer {
    /// Captures a new [`PrimaryStateBuffer`] from `model`.
    #[must_use]
    pub fn from_model(model: &StarModel) -> Self {
        let mut buf = Self {
            nz: model.nz,
            species: model.species,
            chem_id: model.chem_id.clone(),
            iso_names: model.iso_names.clone(),
            layout: model.layout.clone(),
            eos_config: model.eos_config,
            kap_config: model.kap_config.clone(),
            model_number: model.model_number,
            star_age: model.star_age,
            star_age_old: model.star_age_old,
            dt: model.dt,
            dt_next: model.dt_next,
            dt_old: model.dt_old,
            dt_limit_ratio_old: model.dt_limit_ratio_old,
            mstar: model.mstar,
            mstar_start: model.mstar_start,
            mstar_old: model.mstar_old,
            xmstar: model.xmstar,
            mstar_dot: model.mstar_dot,
            m_center: model.m_center,
            r_center: model.r_center,
            l_center: model.l_center,
            v_center: model.v_center,
            tau_base: model.tau_base,
            teff: model.teff,
            lnd: model.lnd.clone(),
            ln_t: model.ln_t.clone(),
            ln_r: model.ln_r.clone(),
            lum: model.lum.clone(),
            v: model.v.clone(),
            u: model.u.clone(),
            omega: model.omega.clone(),
            j_rot: model.j_rot.clone(),
            xa: model.xa.clone(),
            xh: model.xh.clone(),
            lnd_start: model.lnd_start.clone(),
            ln_t_start: model.ln_t_start.clone(),
            ln_r_start: model.ln_r_start.clone(),
            lum_start: model.lum_start.clone(),
            v_start: model.v_start.clone(),
            u_start: model.u_start.clone(),
            ln_pgas_start: model.ln_pgas_start.clone(),
            ln_peos_start: model.ln_peos_start.clone(),
            ln_e_start: model.ln_e_start.clone(),
            ln_s_start: model.ln_s_start.clone(),
            energy_start: model.energy_start.clone(),
            peos_start: model.peos_start.clone(),
            non_nuc_neu_start: model.non_nuc_neu_start.clone(),
            xa_start: model.xa_start.clone(),
            xa_sub_xa_start: model.xa_sub_xa_start.clone(),
            xh_start: model.xh_start.clone(),
            lnd_old: model.lnd_old.clone(),
            ln_t_old: model.ln_t_old.clone(),
            ln_r_old: model.ln_r_old.clone(),
            lum_old: model.lum_old.clone(),
            v_old: model.v_old.clone(),
            u_old: model.u_old.clone(),
            xa_old: model.xa_old.clone(),
            dq_old: model.dq_old.clone(),
            dq: model.dq.clone(),
            q: model.q.clone(),
            xq: model.xq.clone(),
            dm: model.dm.clone(),
            dm_bar: model.dm_bar.clone(),
            m: model.m.clone(),
            m_grav: model.m_grav.clone(),
            cgrav: model.cgrav.clone(),
            grav: model.grav.clone(),
            r: model.r.clone(),
            rmid: model.rmid.clone(),
            energy: model.energy.clone(),
            eps_nuc: model.eps_nuc.clone(),
            dxdt_nuc: model.dxdt_nuc.clone(),
            eps_mdot: model.eps_mdot.clone(),
            gradt: model.gradt.clone(),
            gradr: model.gradr.clone(),
            gradl: model.gradl.clone(),
            scale_height: model.scale_height.clone(),
            mlt_vc: model.mlt_vc.clone(),
            mlt_d: model.mlt_d.clone(),
            mlt_cdc: model.mlt_cdc.clone(),
            mlt_gamma: model.mlt_gamma.clone(),
            d_mix: model.d_mix.clone(),
            conv_vel: model.conv_vel.clone(),
            sig: model.sig.clone(),
            mixing_type: model.mixing_type.clone(),
            brunt_b: model.brunt_b.clone(),
            brunt_n2: model.brunt_n2.clone(),
            doing_relax: model.aux.doing_relax,
            doing_struct_burn_mix: model.aux.doing_struct_burn_mix,
            total_energy: model.aux.total_energy,
            total_internal_energy_start: model.aux.total_internal_energy_start,
            cumulative_energy_error: model.aux.cumulative_energy_error,
            non_epsnuc_energy_change_from_split_burn: model
                .aux
                .non_epsnuc_energy_change_from_split_burn,
            cp_start: model.aux.cp_start.clone(),
            grada_start: model.aux.grada_start.clone(),
            chi_t_start: model.aux.chi_t_start.clone(),
            chi_rho_start: model.aux.chi_rho_start.clone(),
            burn_avg_epsnuc: model.aux.burn_avg_epsnuc.clone(),
            eps_nuc_categories: model.aux.eps_nuc_categories.clone(),
            extra_heat: model.aux.extra_heat.clone(),
            d_extra_heat_dlnd: model.aux.d_extra_heat_dlnd.clone(),
            d_extra_heat_dlnt: model.aux.d_extra_heat_dlnt.clone(),
            eps_wd_sedimentation: model.aux.eps_wd_sedimentation.clone(),
            eps_diffusion: model.aux.eps_diffusion.clone(),
            eps_pre_mix: model.aux.eps_pre_mix.clone(),
            eps_phase_separation: model.aux.eps_phase_separation.clone(),
            irradiation_heat: model.aux.irradiation_heat.clone(),
            etrb: model.aux.etrb.clone(),
            etrb_start: model.aux.etrb_start.clone(),
            et_rsp_old: model.aux.et_rsp_old.clone(),
            hp_face_old: model.aux.hp_face_old.clone(),
            csound_start: model.aux.csound_start.clone(),
            mlt_vc_old: model.aux.mlt_vc_old.clone(),
            eta_rti: model.aux.eta_rti.clone(),
            etamid_rti: model.aux.etamid_rti.clone(),
            sig_rti: model.aux.sig_rti.clone(),
            sigmid_rti: model.aux.sigmid_rti.clone(),
            dpdr_drhodr_info: model.aux.dpdr_drhodr_info.clone(),
            source_plus_alpha_rti: model.aux.source_plus_alpha_rti.clone(),
            source_minus_alpha_rti: model.aux.source_minus_alpha_rti.clone(),
            alpha_rti: model.aux.alpha_rti.clone(),
            alpha_rti_start: model.aux.alpha_rti_start.clone(),
            alpha_rti_old: model.aux.alpha_rti_old.clone(),
            u_face_start: model.aux.u_face_start.clone(),
            p_face_start: model.aux.p_face_start.clone(),
            w_div_wc: model.aux.w_div_wc.clone(),
            fp_rot: model.aux.fp_rot.clone(),
            d_fp_rot_dlnr: model.aux.d_fp_rot_dlnr.clone(),
            ft_rot: model.aux.ft_rot.clone(),
            d_ft_rot_dlnr: model.aux.d_ft_rot_dlnr.clone(),
            i_rot: model.aux.i_rot.clone(),
            d_i_rot_dlnr: model.aux.d_i_rot_dlnr.clone(),
            brunt_n2_composition_term: model
                .aux
                .brunt_n2_composition_term
                .clone(),
            smoothed_brunt_b: model.aux.smoothed_brunt_b.clone(),
            smoothed_brunt_b_hash: model.aux.smoothed_brunt_b_hash,
            mass_correction: model.aux.mass_correction.clone(),
            omega_old: model.aux.omega_old.clone(),
            j_rot_old: model.aux.j_rot_old.clone(),
            eps_mdot_old: model.aux.eps_mdot_old.clone(),
            solver_mlt_eval_count: model.aux.solver_mlt_eval_count,
            solver_had_mlt_flip: model.aux.solver_had_mlt_flip,
            fixed_gradr_sticky: model.aux.fixed_gradr_sticky.clone(),
            mlt_prev_conv: model.aux.mlt_prev_conv.clone(),
            mlt_flip_count: model.aux.mlt_flip_count.clone(),
            step_hist: model.aux.step_hist.clone(),
            rsp: model.aux.rsp.clone(),
            tidal_deformation: model.aux.tidal_deformation.clone(),
            extra_grav_if_nonzero: None,
        };
        buf.capture_extra_grav(model);
        buf
    }

    fn capture_extra_grav(&mut self, model: &StarModel) {
        let has_nonzero = model
            .aux
            .extra_grav
            .iter()
            .any(|g| g.val != 0.0 || g.d1.iter().any(|&x| x != 0.0));
        if has_nonzero {
            match &mut self.extra_grav_if_nonzero {
                Some(dst) => dst.clone_from(&model.aux.extra_grav),
                None => {
                    self.extra_grav_if_nonzero =
                        Some(model.aux.extra_grav.clone());
                }
            }
        } else {
            self.extra_grav_if_nonzero = None;
        }
    }

    /// Overwrites this buffer in place from `model`, reusing existing heap
    /// capacities across retries and implicit trials.
    pub fn capture_from(&mut self, model: &StarModel) {
        self.nz = model.nz;
        self.species = model.species;
        self.chem_id.clone_from(&model.chem_id);
        self.iso_names.clone_from(&model.iso_names);
        self.layout.clone_from(&model.layout);
        self.eos_config = model.eos_config;
        self.kap_config.clone_from(&model.kap_config);
        self.model_number = model.model_number;
        self.star_age = model.star_age;
        self.star_age_old = model.star_age_old;
        self.dt = model.dt;
        self.dt_next = model.dt_next;
        self.dt_old = model.dt_old;
        self.dt_limit_ratio_old = model.dt_limit_ratio_old;
        self.mstar = model.mstar;
        self.mstar_start = model.mstar_start;
        self.mstar_old = model.mstar_old;
        self.xmstar = model.xmstar;
        self.mstar_dot = model.mstar_dot;
        self.m_center = model.m_center;
        self.r_center = model.r_center;
        self.l_center = model.l_center;
        self.v_center = model.v_center;
        self.tau_base = model.tau_base;
        self.teff = model.teff;

        self.lnd.clone_from(&model.lnd);
        self.ln_t.clone_from(&model.ln_t);
        self.ln_r.clone_from(&model.ln_r);
        self.lum.clone_from(&model.lum);
        self.v.clone_from(&model.v);
        self.u.clone_from(&model.u);
        self.omega.clone_from(&model.omega);
        self.j_rot.clone_from(&model.j_rot);
        self.xa.clone_from(&model.xa);
        self.xh.clone_from(&model.xh);
        self.lnd_start.clone_from(&model.lnd_start);
        self.ln_t_start.clone_from(&model.ln_t_start);
        self.ln_r_start.clone_from(&model.ln_r_start);
        self.lum_start.clone_from(&model.lum_start);
        self.v_start.clone_from(&model.v_start);
        self.u_start.clone_from(&model.u_start);
        self.ln_pgas_start.clone_from(&model.ln_pgas_start);
        self.ln_peos_start.clone_from(&model.ln_peos_start);
        self.ln_e_start.clone_from(&model.ln_e_start);
        self.ln_s_start.clone_from(&model.ln_s_start);
        self.energy_start.clone_from(&model.energy_start);
        self.peos_start.clone_from(&model.peos_start);
        self.non_nuc_neu_start.clone_from(&model.non_nuc_neu_start);
        self.xa_start.clone_from(&model.xa_start);
        self.xa_sub_xa_start.clone_from(&model.xa_sub_xa_start);
        self.xh_start.clone_from(&model.xh_start);
        self.lnd_old.clone_from(&model.lnd_old);
        self.ln_t_old.clone_from(&model.ln_t_old);
        self.ln_r_old.clone_from(&model.ln_r_old);
        self.lum_old.clone_from(&model.lum_old);
        self.v_old.clone_from(&model.v_old);
        self.u_old.clone_from(&model.u_old);
        self.xa_old.clone_from(&model.xa_old);
        self.dq_old.clone_from(&model.dq_old);
        self.dq.clone_from(&model.dq);
        self.q.clone_from(&model.q);
        self.xq.clone_from(&model.xq);
        self.dm.clone_from(&model.dm);
        self.dm_bar.clone_from(&model.dm_bar);
        self.m.clone_from(&model.m);
        self.m_grav.clone_from(&model.m_grav);
        self.cgrav.clone_from(&model.cgrav);
        self.grav.clone_from(&model.grav);
        self.r.clone_from(&model.r);
        self.rmid.clone_from(&model.rmid);
        self.energy.clone_from(&model.energy);
        self.eps_nuc.clone_from(&model.eps_nuc);
        self.dxdt_nuc.clone_from(&model.dxdt_nuc);
        self.eps_mdot.clone_from(&model.eps_mdot);
        self.gradt.clone_from(&model.gradt);
        self.gradr.clone_from(&model.gradr);
        self.gradl.clone_from(&model.gradl);
        self.scale_height.clone_from(&model.scale_height);
        self.mlt_vc.clone_from(&model.mlt_vc);
        self.mlt_d.clone_from(&model.mlt_d);
        self.mlt_cdc.clone_from(&model.mlt_cdc);
        self.mlt_gamma.clone_from(&model.mlt_gamma);
        self.d_mix.clone_from(&model.d_mix);
        self.conv_vel.clone_from(&model.conv_vel);
        self.sig.clone_from(&model.sig);
        self.mixing_type.clone_from(&model.mixing_type);
        self.brunt_b.clone_from(&model.brunt_b);
        self.brunt_n2.clone_from(&model.brunt_n2);

        let a = &model.aux;
        self.doing_relax = a.doing_relax;
        self.doing_struct_burn_mix = a.doing_struct_burn_mix;
        self.total_energy = a.total_energy;
        self.total_internal_energy_start = a.total_internal_energy_start;
        self.cumulative_energy_error = a.cumulative_energy_error;
        self.non_epsnuc_energy_change_from_split_burn =
            a.non_epsnuc_energy_change_from_split_burn;
        self.cp_start.clone_from(&a.cp_start);
        self.grada_start.clone_from(&a.grada_start);
        self.chi_t_start.clone_from(&a.chi_t_start);
        self.chi_rho_start.clone_from(&a.chi_rho_start);
        self.burn_avg_epsnuc.clone_from(&a.burn_avg_epsnuc);
        self.eps_nuc_categories.clone_from(&a.eps_nuc_categories);
        self.extra_heat.clone_from(&a.extra_heat);
        self.d_extra_heat_dlnd.clone_from(&a.d_extra_heat_dlnd);
        self.d_extra_heat_dlnt.clone_from(&a.d_extra_heat_dlnt);
        self.eps_wd_sedimentation
            .clone_from(&a.eps_wd_sedimentation);
        self.eps_diffusion.clone_from(&a.eps_diffusion);
        self.eps_pre_mix.clone_from(&a.eps_pre_mix);
        self.eps_phase_separation
            .clone_from(&a.eps_phase_separation);
        self.irradiation_heat.clone_from(&a.irradiation_heat);
        self.etrb.clone_from(&a.etrb);
        self.etrb_start.clone_from(&a.etrb_start);
        self.et_rsp_old.clone_from(&a.et_rsp_old);
        self.hp_face_old.clone_from(&a.hp_face_old);
        self.csound_start.clone_from(&a.csound_start);
        self.mlt_vc_old.clone_from(&a.mlt_vc_old);
        self.eta_rti.clone_from(&a.eta_rti);
        self.etamid_rti.clone_from(&a.etamid_rti);
        self.sig_rti.clone_from(&a.sig_rti);
        self.sigmid_rti.clone_from(&a.sigmid_rti);
        self.dpdr_drhodr_info.clone_from(&a.dpdr_drhodr_info);
        self.source_plus_alpha_rti
            .clone_from(&a.source_plus_alpha_rti);
        self.source_minus_alpha_rti
            .clone_from(&a.source_minus_alpha_rti);
        self.alpha_rti.clone_from(&a.alpha_rti);
        self.alpha_rti_start.clone_from(&a.alpha_rti_start);
        self.alpha_rti_old.clone_from(&a.alpha_rti_old);
        self.u_face_start.clone_from(&a.u_face_start);
        self.p_face_start.clone_from(&a.p_face_start);
        self.w_div_wc.clone_from(&a.w_div_wc);
        self.fp_rot.clone_from(&a.fp_rot);
        self.d_fp_rot_dlnr.clone_from(&a.d_fp_rot_dlnr);
        self.ft_rot.clone_from(&a.ft_rot);
        self.d_ft_rot_dlnr.clone_from(&a.d_ft_rot_dlnr);
        self.i_rot.clone_from(&a.i_rot);
        self.d_i_rot_dlnr.clone_from(&a.d_i_rot_dlnr);
        self.brunt_n2_composition_term
            .clone_from(&a.brunt_n2_composition_term);
        self.smoothed_brunt_b.clone_from(&a.smoothed_brunt_b);
        self.smoothed_brunt_b_hash = a.smoothed_brunt_b_hash;
        self.mass_correction.clone_from(&a.mass_correction);
        self.omega_old.clone_from(&a.omega_old);
        self.j_rot_old.clone_from(&a.j_rot_old);
        self.eps_mdot_old.clone_from(&a.eps_mdot_old);
        self.solver_mlt_eval_count = a.solver_mlt_eval_count;
        self.solver_had_mlt_flip = a.solver_had_mlt_flip;
        self.fixed_gradr_sticky.clone_from(&a.fixed_gradr_sticky);
        self.mlt_prev_conv.clone_from(&a.mlt_prev_conv);
        self.mlt_flip_count.clone_from(&a.mlt_flip_count);
        self.step_hist.clone_from(&a.step_hist);
        self.rsp.clone_from(&a.rsp);
        self.tidal_deformation.clone_from(&a.tidal_deformation);
        self.capture_extra_grav(model);
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
        model.chem_id.clone_from(&self.chem_id);
        model.iso_names.clone_from(&self.iso_names);
        model.layout.clone_from(&self.layout);
        if chem_changed {
            let catalog = NuclideCatalog::builtin_ref();
            model.net_iso.fill(None);
            for (j, &cid) in model.chem_id.iter().enumerate() {
                if cid < model.net_iso.len() {
                    model.net_iso[cid] = Some(j);
                }
                if j < model.iso_names.len() {
                    model.iso_names[j] = catalog.get_by_id(cid).map_or_else(
                        || format!("iso_{cid}"),
                        |n| n.name.clone(),
                    );
                }
            }
        }
        if resize_needed {
            model.resize_zones(self.nz);
        } else {
            model.nz = self.nz;
        }
        model.eos_config = self.eos_config;
        model.kap_config.clone_from(&self.kap_config);
        model.model_number = self.model_number;
        model.star_age = self.star_age;
        model.star_age_old = self.star_age_old;
        model.dt = self.dt;
        model.dt_next = self.dt_next;
        model.dt_old = self.dt_old;
        model.dt_limit_ratio_old = self.dt_limit_ratio_old;
        model.mstar = self.mstar;
        model.mstar_start = self.mstar_start;
        model.mstar_old = self.mstar_old;
        model.xmstar = self.xmstar;
        model.mstar_dot = self.mstar_dot;
        model.m_center = self.m_center;
        model.r_center = self.r_center;
        model.l_center = self.l_center;
        model.v_center = self.v_center;
        model.tau_base = self.tau_base;
        model.teff = self.teff;

        model.lnd.clone_from(&self.lnd);
        model.ln_t.clone_from(&self.ln_t);
        model.ln_r.clone_from(&self.ln_r);
        model.lum.clone_from(&self.lum);
        model.v.clone_from(&self.v);
        model.u.clone_from(&self.u);
        model.omega.clone_from(&self.omega);
        model.j_rot.clone_from(&self.j_rot);
        model.xa.clone_from(&self.xa);
        model.xh.clone_from(&self.xh);
        model.lnd_start.clone_from(&self.lnd_start);
        model.ln_t_start.clone_from(&self.ln_t_start);
        model.ln_r_start.clone_from(&self.ln_r_start);
        model.lum_start.clone_from(&self.lum_start);
        model.v_start.clone_from(&self.v_start);
        model.u_start.clone_from(&self.u_start);
        model.ln_pgas_start.clone_from(&self.ln_pgas_start);
        model.ln_peos_start.clone_from(&self.ln_peos_start);
        model.ln_e_start.clone_from(&self.ln_e_start);
        model.ln_s_start.clone_from(&self.ln_s_start);
        model.energy_start.clone_from(&self.energy_start);
        model.peos_start.clone_from(&self.peos_start);
        model.non_nuc_neu_start.clone_from(&self.non_nuc_neu_start);
        model.xa_start.clone_from(&self.xa_start);
        model.xa_sub_xa_start.clone_from(&self.xa_sub_xa_start);
        model.xh_start.clone_from(&self.xh_start);
        model.lnd_old.clone_from(&self.lnd_old);
        model.ln_t_old.clone_from(&self.ln_t_old);
        model.ln_r_old.clone_from(&self.ln_r_old);
        model.lum_old.clone_from(&self.lum_old);
        model.v_old.clone_from(&self.v_old);
        model.u_old.clone_from(&self.u_old);
        model.xa_old.clone_from(&self.xa_old);
        model.dq_old.clone_from(&self.dq_old);
        model.dq.clone_from(&self.dq);
        model.q.clone_from(&self.q);
        model.xq.clone_from(&self.xq);
        model.dm.clone_from(&self.dm);
        model.dm_bar.clone_from(&self.dm_bar);
        model.m.clone_from(&self.m);
        model.m_grav.clone_from(&self.m_grav);
        model.cgrav.clone_from(&self.cgrav);
        model.grav.clone_from(&self.grav);
        model.r.clone_from(&self.r);
        model.rmid.clone_from(&self.rmid);
        model.energy.clone_from(&self.energy);

        let a = &mut model.aux;
        a.doing_relax = self.doing_relax;
        a.doing_struct_burn_mix = false;
        a.total_energy = self.total_energy;
        a.total_internal_energy_start = self.total_internal_energy_start;
        a.cumulative_energy_error = self.cumulative_energy_error;
        a.non_epsnuc_energy_change_from_split_burn =
            self.non_epsnuc_energy_change_from_split_burn;
        a.cp_start.clone_from(&self.cp_start);
        a.grada_start.clone_from(&self.grada_start);
        a.chi_t_start.clone_from(&self.chi_t_start);
        a.chi_rho_start.clone_from(&self.chi_rho_start);
        a.burn_avg_epsnuc.clone_from(&self.burn_avg_epsnuc);
        a.eps_nuc_categories.clone_from(&self.eps_nuc_categories);
        a.extra_heat.clone_from(&self.extra_heat);
        a.d_extra_heat_dlnd.clone_from(&self.d_extra_heat_dlnd);
        a.d_extra_heat_dlnt.clone_from(&self.d_extra_heat_dlnt);
        a.eps_wd_sedimentation
            .clone_from(&self.eps_wd_sedimentation);
        a.eps_diffusion.clone_from(&self.eps_diffusion);
        a.eps_pre_mix.clone_from(&self.eps_pre_mix);
        a.eps_phase_separation
            .clone_from(&self.eps_phase_separation);
        a.irradiation_heat.clone_from(&self.irradiation_heat);
        a.etrb.clone_from(&self.etrb);
        a.etrb_start.clone_from(&self.etrb_start);
        a.et_rsp_old.clone_from(&self.et_rsp_old);
        a.hp_face_old.clone_from(&self.hp_face_old);
        a.csound_start.clone_from(&self.csound_start);
        a.mlt_vc_old.clone_from(&self.mlt_vc_old);
        a.eta_rti.clone_from(&self.eta_rti);
        a.etamid_rti.clone_from(&self.etamid_rti);
        a.sig_rti.clone_from(&self.sig_rti);
        a.sigmid_rti.clone_from(&self.sigmid_rti);
        a.dpdr_drhodr_info.clone_from(&self.dpdr_drhodr_info);
        a.source_plus_alpha_rti
            .clone_from(&self.source_plus_alpha_rti);
        a.source_minus_alpha_rti
            .clone_from(&self.source_minus_alpha_rti);
        a.alpha_rti.clone_from(&self.alpha_rti);
        a.alpha_rti_start.clone_from(&self.alpha_rti_start);
        a.alpha_rti_old.clone_from(&self.alpha_rti_old);
        a.u_face_start.clone_from(&self.u_face_start);
        a.p_face_start.clone_from(&self.p_face_start);
        a.w_div_wc.clone_from(&self.w_div_wc);
        a.fp_rot.clone_from(&self.fp_rot);
        a.d_fp_rot_dlnr.clone_from(&self.d_fp_rot_dlnr);
        a.ft_rot.clone_from(&self.ft_rot);
        a.d_ft_rot_dlnr.clone_from(&self.d_ft_rot_dlnr);
        a.i_rot.clone_from(&self.i_rot);
        a.d_i_rot_dlnr.clone_from(&self.d_i_rot_dlnr);
        a.brunt_n2_composition_term
            .clone_from(&self.brunt_n2_composition_term);
        a.smoothed_brunt_b.clone_from(&self.smoothed_brunt_b);
        a.smoothed_brunt_b_hash = self.smoothed_brunt_b_hash;
        a.mass_correction.clone_from(&self.mass_correction);
        a.omega_old.clone_from(&self.omega_old);
        a.j_rot_old.clone_from(&self.j_rot_old);
        a.eps_mdot_old.clone_from(&self.eps_mdot_old);
        a.step_hist.clone_from(&self.step_hist);
        a.rsp.clone_from(&self.rsp);
        a.tidal_deformation.clone_from(&self.tidal_deformation);
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

        if self.nz > 0 {
            crate::hydro_vars::set_hydro_vars(model, false, false)?;
        }

        // Restore exact post-step values (e.g., overshoot/predictive mixing,
        // operator-split burn averages, rotation distortion, and solver latches)
        // that `set_hydro_vars` may have recomputed without pre-physics passes.
        model.teff = self.teff;
        model.omega.clone_from(&self.omega);
        model.j_rot.clone_from(&self.j_rot);
        model.eps_nuc.clone_from(&self.eps_nuc);
        model.dxdt_nuc.clone_from(&self.dxdt_nuc);
        model.eps_mdot.clone_from(&self.eps_mdot);
        model.gradt.clone_from(&self.gradt);
        model.gradr.clone_from(&self.gradr);
        model.gradl.clone_from(&self.gradl);
        model.scale_height.clone_from(&self.scale_height);
        model.mlt_vc.clone_from(&self.mlt_vc);
        model.mlt_d.clone_from(&self.mlt_d);
        model.mlt_cdc.clone_from(&self.mlt_cdc);
        model.mlt_gamma.clone_from(&self.mlt_gamma);
        model.d_mix.clone_from(&self.d_mix);
        model.conv_vel.clone_from(&self.conv_vel);
        model.sig.clone_from(&self.sig);
        model.mixing_type.clone_from(&self.mixing_type);
        model.brunt_b.clone_from(&self.brunt_b);
        model.brunt_n2.clone_from(&self.brunt_n2);
        model.aux.doing_struct_burn_mix = self.doing_struct_burn_mix;
        model
            .aux
            .eps_nuc_categories
            .clone_from(&self.eps_nuc_categories);
        model.aux.w_div_wc.clone_from(&self.w_div_wc);
        model.aux.fp_rot.clone_from(&self.fp_rot);
        model.aux.d_fp_rot_dlnr.clone_from(&self.d_fp_rot_dlnr);
        model.aux.ft_rot.clone_from(&self.ft_rot);
        model.aux.d_ft_rot_dlnr.clone_from(&self.d_ft_rot_dlnr);
        model.aux.i_rot.clone_from(&self.i_rot);
        model.aux.d_i_rot_dlnr.clone_from(&self.d_i_rot_dlnr);
        model
            .aux
            .smoothed_brunt_b
            .clone_from(&self.smoothed_brunt_b);
        model.aux.smoothed_brunt_b_hash = self.smoothed_brunt_b_hash;
        model.aux.solver_mlt_eval_count = self.solver_mlt_eval_count;
        model.aux.solver_had_mlt_flip = self.solver_had_mlt_flip;
        model
            .aux
            .fixed_gradr_sticky
            .clone_from(&self.fixed_gradr_sticky);
        model.aux.mlt_prev_conv.clone_from(&self.mlt_prev_conv);
        model.aux.mlt_flip_count.clone_from(&self.mlt_flip_count);
        Ok(())
    }

    /// Returns the approximate live heap byte size of this rollback buffer.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let rsp = &self.rsp;
        let f64_bytes: usize = [
            &self.lnd,
            &self.ln_t,
            &self.ln_r,
            &self.lum,
            &self.v,
            &self.u,
            &self.omega,
            &self.j_rot,
            &self.xa,
            &self.xh,
            &self.lnd_start,
            &self.ln_t_start,
            &self.ln_r_start,
            &self.lum_start,
            &self.v_start,
            &self.u_start,
            &self.ln_pgas_start,
            &self.ln_peos_start,
            &self.ln_e_start,
            &self.ln_s_start,
            &self.energy_start,
            &self.peos_start,
            &self.non_nuc_neu_start,
            &self.xa_start,
            &self.xa_sub_xa_start,
            &self.xh_start,
            &self.lnd_old,
            &self.ln_t_old,
            &self.ln_r_old,
            &self.lum_old,
            &self.v_old,
            &self.u_old,
            &self.xa_old,
            &self.dq_old,
            &self.dq,
            &self.q,
            &self.xq,
            &self.dm,
            &self.dm_bar,
            &self.m,
            &self.m_grav,
            &self.cgrav,
            &self.grav,
            &self.r,
            &self.rmid,
            &self.energy,
            &self.eps_nuc,
            &self.dxdt_nuc,
            &self.eps_mdot,
            &self.gradt,
            &self.gradr,
            &self.gradl,
            &self.scale_height,
            &self.mlt_vc,
            &self.mlt_d,
            &self.mlt_cdc,
            &self.mlt_gamma,
            &self.d_mix,
            &self.conv_vel,
            &self.sig,
            &self.brunt_b,
            &self.brunt_n2,
            &self.cp_start,
            &self.grada_start,
            &self.chi_t_start,
            &self.chi_rho_start,
            &self.burn_avg_epsnuc,
            &self.extra_heat,
            &self.d_extra_heat_dlnd,
            &self.d_extra_heat_dlnt,
            &self.eps_wd_sedimentation,
            &self.eps_diffusion,
            &self.eps_pre_mix,
            &self.eps_phase_separation,
            &self.irradiation_heat,
            &self.etrb,
            &self.etrb_start,
            &self.et_rsp_old,
            &self.hp_face_old,
            &self.csound_start,
            &self.mlt_vc_old,
            &self.eta_rti,
            &self.etamid_rti,
            &self.sig_rti,
            &self.sigmid_rti,
            &self.dpdr_drhodr_info,
            &self.source_plus_alpha_rti,
            &self.source_minus_alpha_rti,
            &self.alpha_rti,
            &self.alpha_rti_start,
            &self.alpha_rti_old,
            &self.u_face_start,
            &self.p_face_start,
            &self.w_div_wc,
            &self.fp_rot,
            &self.d_fp_rot_dlnr,
            &self.ft_rot,
            &self.d_ft_rot_dlnr,
            &self.i_rot,
            &self.d_i_rot_dlnr,
            &self.brunt_n2_composition_term,
            &self.smoothed_brunt_b,
            &self.mass_correction,
            &self.omega_old,
            &self.j_rot_old,
            &self.eps_mdot_old,
            &rsp.et_rsp,
            &rsp.erad_rsp,
            &rsp.fr_rsp,
            &rsp.lc_rsp,
            &rsp.lt_rsp,
            &rsp.ptrb_rsp,
            &rsp.pvsc_rsp,
            &rsp.egas_rsp,
            &rsp.hp_face,
            &rsp.vol_rsp,
            &rsp.f_edd_rsp,
            &rsp.y_face_rsp,
            &rsp.pii_rsp,
            &rsp.chi_rsp,
            &rsp.coupl_rsp,
            &rsp.eq_rsp,
            &rsp.qq_rsp,
            &rsp.source_rsp,
            &rsp.damp_rsp,
            &rsp.dampr_rsp,
            &rsp.vol_start,
            &rsp.w_start,
            &rsp.fr_start,
            &rsp.erad_start,
            &rsp.egas_start,
            &rsp.pgas_start,
            &rsp.prad_start,
            &rsp.opacity_start,
            &rsp.qq_start,
            &rsp.lc_start,
            &rsp.lt_start,
            &rsp.t_start_rsp,
            &rsp.r_start_rsp,
            &rsp.v_start_rsp,
            &rsp.csound_start_rsp,
            &rsp.pvsc_start,
            &rsp.ptrb_start,
            &rsp.chi_start,
            &rsp.coupl_start,
            &rsp.ppp0,
            &rsp.ppq0,
            &rsp.ppt0,
            &rsp.ppc0,
            &rsp.vv0,
            &rsp.work,
            &rsp.workq,
            &rsp.workt,
            &rsp.workc,
        ]
        .into_iter()
        .map(|v| f64_vec_heap(v))
        .sum();
        f64_bytes
            + size_of_val(self.chem_id.as_slice())
            + self.iso_names.iter().map(String::len).sum::<usize>()
            + size_of_val(self.mixing_type.as_slice())
            + size_of_val(self.eps_nuc_categories.as_slice())
            + size_of_val(self.fixed_gradr_sticky.as_slice())
            + self.mlt_prev_conv.len()
            + self.mlt_flip_count.len()
            + self
                .extra_grav_if_nonzero
                .as_ref()
                .map_or(0, |v| size_of_val(v.as_slice()))
    }
}
