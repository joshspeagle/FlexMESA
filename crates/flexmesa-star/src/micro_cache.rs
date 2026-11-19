// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bit-exact 2-slot LRU microphysics cache for [`crate::hydro_vars::set_hydro_vars`].
//!
//! Avoids redundant `set_basic_vars`, `do_eos`, `do_kap`, `do_neu`, and `do_net`
//! evaluations across unchanged states (pre-remesh -> start of
//! `do_struct_burn_mix` -> initial residual evaluation -> retry/redo
//! transitions) while invalidating on any 1-ULP change in thermodynamic,
//! geometric, abundance, or control inputs.

use flexmesa_chem::NUM_CATEGORIES;
use flexmesa_eos::{EosConfig, NUM_EOS_BASIC_RESULTS};
use flexmesa_kap::KapConfig;
use flexmesa_rates::ScreeningMode;

use crate::controls::StarError;
use crate::micro::{
    do_eos, do_kap, do_net, do_neu, resolve_model_net, set_basic_vars,
};
use crate::model::StarModel;
use crate::star_ad::StarAdOrder1;

#[inline]
fn f64_slices_bit_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[derive(Debug, Clone)]
struct CachedMicroKey {
    nz: usize,
    species: usize,
    chem_id: Vec<usize>,
    net_name: String,
    rsp_flag: bool,
    rsp2_flag: bool,
    screening_mode: ScreeningMode,
    eos_config: EosConfig,
    kap_config: KapConfig,
    include_composition_in_eps_grav: bool,
    op_split_burn: bool,
    doing_split: bool,
    mstar_bits: u64,
    m_center_bits: u64,
    r_center_bits: u64,
    tau_base_bits: u64,
    tau_factor_bits: u64,
    opacity_factor_bits: u64,
    min_t_for_neu_bits: u64,
    non_nuc_neu_factor_bits: u64,
    max_logt_for_net_bits: u64,
    max_abar_for_burning_bits: u64,
    eps_nuc_factor_bits: u64,
    dxdt_nuc_factor_bits: u64,
    op_split_burn_min_t_bits: u64,
    fe56ec_fake_factor_bits: u64,
    min_t_for_fe56ec_fake_factor_bits: u64,
    lnd: Vec<f64>,
    ln_t: Vec<f64>,
    ln_r: Vec<f64>,
    dq: Vec<f64>,
    q: Vec<f64>,
    cgrav: Vec<f64>,
    xa: Vec<f64>,
    ln_t_start_split: Vec<f64>,
}

impl Default for CachedMicroKey {
    fn default() -> Self {
        Self {
            nz: 0,
            species: 0,
            chem_id: Vec::new(),
            net_name: String::new(),
            rsp_flag: false,
            rsp2_flag: false,
            screening_mode: ScreeningMode::Chugunov,
            eos_config: EosConfig::default(),
            kap_config: KapConfig::default(),
            include_composition_in_eps_grav: false,
            op_split_burn: false,
            doing_split: false,
            mstar_bits: 0,
            m_center_bits: 0,
            r_center_bits: 0,
            tau_base_bits: 0,
            tau_factor_bits: 0,
            opacity_factor_bits: 0,
            min_t_for_neu_bits: 0,
            non_nuc_neu_factor_bits: 0,
            max_logt_for_net_bits: 0,
            max_abar_for_burning_bits: 0,
            eps_nuc_factor_bits: 0,
            dxdt_nuc_factor_bits: 0,
            op_split_burn_min_t_bits: 0,
            fe56ec_fake_factor_bits: 0,
            min_t_for_fe56ec_fake_factor_bits: 0,
            lnd: Vec::new(),
            ln_t: Vec::new(),
            ln_r: Vec::new(),
            dq: Vec::new(),
            q: Vec::new(),
            cgrav: Vec::new(),
            xa: Vec::new(),
            ln_t_start_split: Vec::new(),
        }
    }
}

impl CachedMicroKey {
    fn matches(&self, model: &StarModel, doing_split: bool) -> bool {
        let c = &model.controls;
        if self.nz != model.nz
            || self.species != model.species
            || self.rsp_flag != c.rsp_flag
            || self.rsp2_flag != c.rsp2_flag
            || self.screening_mode != c.screening_mode
            || self.include_composition_in_eps_grav
                != c.include_composition_in_eps_grav
            || self.op_split_burn != c.op_split_burn
            || self.doing_split != doing_split
            || self.mstar_bits != model.mstar.to_bits()
            || self.m_center_bits != model.m_center.to_bits()
            || self.r_center_bits != model.r_center.to_bits()
            || self.tau_base_bits != model.tau_base.to_bits()
            || self.tau_factor_bits != c.tau_factor.to_bits()
            || self.opacity_factor_bits != c.opacity_factor.to_bits()
            || self.min_t_for_neu_bits != c.min_t_for_neu.to_bits()
            || self.non_nuc_neu_factor_bits != c.non_nuc_neu_factor.to_bits()
            || self.max_logt_for_net_bits != c.max_logt_for_net.to_bits()
            || self.max_abar_for_burning_bits
                != c.max_abar_for_burning.to_bits()
            || self.eps_nuc_factor_bits != c.eps_nuc_factor.to_bits()
            || self.dxdt_nuc_factor_bits != c.dxdt_nuc_factor.to_bits()
            || self.op_split_burn_min_t_bits != c.op_split_burn_min_t.to_bits()
            || self.fe56ec_fake_factor_bits != c.fe56ec_fake_factor.to_bits()
            || self.min_t_for_fe56ec_fake_factor_bits
                != c.min_t_for_fe56ec_fake_factor.to_bits()
            || self.chem_id != model.chem_id
            || self.net_name != model.job.net_name
            || self.eos_config != model.eos_config
            || self.kap_config != model.kap_config
        {
            return false;
        }
        if !f64_slices_bit_eq(&self.lnd, &model.lnd)
            || !f64_slices_bit_eq(&self.ln_t, &model.ln_t)
            || !f64_slices_bit_eq(&self.ln_r, &model.ln_r)
            || !f64_slices_bit_eq(&self.dq, &model.dq)
            || !f64_slices_bit_eq(&self.q, &model.q)
            || !f64_slices_bit_eq(&self.cgrav, &model.cgrav)
            || !f64_slices_bit_eq(&self.xa, &model.xa)
        {
            return false;
        }
        if doing_split
            && !f64_slices_bit_eq(&self.ln_t_start_split, &model.ln_t_start)
        {
            return false;
        }
        true
    }

    fn capture_from(&mut self, model: &StarModel, doing_split: bool) {
        let c = &model.controls;
        self.nz = model.nz;
        self.species = model.species;
        self.chem_id.clone_from(&model.chem_id);
        self.net_name.clone_from(&model.job.net_name);
        self.rsp_flag = c.rsp_flag;
        self.rsp2_flag = c.rsp2_flag;
        self.screening_mode = c.screening_mode;
        self.eos_config = model.eos_config;
        self.kap_config = model.kap_config.clone();
        self.include_composition_in_eps_grav =
            c.include_composition_in_eps_grav;
        self.op_split_burn = c.op_split_burn;
        self.doing_split = doing_split;
        self.mstar_bits = model.mstar.to_bits();
        self.m_center_bits = model.m_center.to_bits();
        self.r_center_bits = model.r_center.to_bits();
        self.tau_base_bits = model.tau_base.to_bits();
        self.tau_factor_bits = c.tau_factor.to_bits();
        self.opacity_factor_bits = c.opacity_factor.to_bits();
        self.min_t_for_neu_bits = c.min_t_for_neu.to_bits();
        self.non_nuc_neu_factor_bits = c.non_nuc_neu_factor.to_bits();
        self.max_logt_for_net_bits = c.max_logt_for_net.to_bits();
        self.max_abar_for_burning_bits = c.max_abar_for_burning.to_bits();
        self.eps_nuc_factor_bits = c.eps_nuc_factor.to_bits();
        self.dxdt_nuc_factor_bits = c.dxdt_nuc_factor.to_bits();
        self.op_split_burn_min_t_bits = c.op_split_burn_min_t.to_bits();
        self.fe56ec_fake_factor_bits = c.fe56ec_fake_factor.to_bits();
        self.min_t_for_fe56ec_fake_factor_bits =
            c.min_t_for_fe56ec_fake_factor.to_bits();
        self.lnd.clone_from(&model.lnd);
        self.ln_t.clone_from(&model.ln_t);
        self.ln_r.clone_from(&model.ln_r);
        self.dq.clone_from(&model.dq);
        self.q.clone_from(&model.q);
        self.cgrav.clone_from(&model.cgrav);
        self.xa.clone_from(&model.xa);
        if doing_split {
            self.ln_t_start_split.clone_from(&model.ln_t_start);
        } else {
            self.ln_t_start_split.clear();
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CachedMicroOutputs {
    rho: Vec<f64>,
    t: Vec<f64>,
    abar: Vec<f64>,
    zbar: Vec<f64>,
    z2bar: Vec<f64>,
    z53bar: Vec<f64>,
    ye: Vec<f64>,
    xh_mass: Vec<f64>,
    yhe_mass: Vec<f64>,
    z_mass: Vec<f64>,

    pgas: Vec<f64>,
    ln_pgas: Vec<f64>,
    prad: Vec<f64>,
    peos: Vec<f64>,
    ln_peos: Vec<f64>,
    energy: Vec<f64>,
    ln_e: Vec<f64>,
    entropy: Vec<f64>,
    ln_s: Vec<f64>,
    grada: Vec<f64>,
    cv: Vec<f64>,
    cp: Vec<f64>,
    chi_rho: Vec<f64>,
    chi_t: Vec<f64>,
    gamma1: Vec<f64>,
    gamma3: Vec<f64>,
    eta: Vec<f64>,
    gam: Vec<f64>,
    mu: Vec<f64>,
    lnfree_e: Vec<f64>,
    csound: Vec<f64>,
    eos_frac_pc: Vec<f64>,
    latent_ddlnt: Vec<f64>,
    latent_ddlnrho: Vec<f64>,
    dln_e_dxa_for_partials: Vec<f64>,
    dln_peos_dxa_for_partials: Vec<f64>,
    ln_peos_ad: Vec<StarAdOrder1>,
    ln_e_ad: Vec<StarAdOrder1>,
    ln_s_ad: Vec<StarAdOrder1>,
    eos_res: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    eos_d_dlnd: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    eos_d_dlnt: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,

    opacity: Vec<f64>,
    d_opacity_dlnd: Vec<f64>,
    d_opacity_dlnt: Vec<f64>,
    tau: Vec<f64>,

    non_nuc_neu: Vec<f64>,
    d_non_nuc_neu_dlnd: Vec<f64>,
    d_non_nuc_neu_dlnt: Vec<f64>,

    has_net: bool,
    has_net_jac: bool,
    eps_nuc: Vec<f64>,
    d_epsnuc_dlnd: Vec<f64>,
    d_epsnuc_dlnt: Vec<f64>,
    eps_nuc_neu_total: Vec<f64>,
    eps_nuc_categories: Vec<[f64; NUM_CATEGORIES]>,
    d_epsnuc_dx: Vec<f64>,
    dxdt_nuc: Vec<f64>,
    d_dxdt_nuc_dlnd: Vec<f64>,
    d_dxdt_nuc_dlnt: Vec<f64>,
    d_dxdt_nuc_dx: Vec<f64>,
}

/// Cap per-slot cached network Jacobian storage at ~64 MB (`8_000_000` `f64`
/// elements, e.g., skipping `d_dxdt_nuc_dx` caching on 200+ isotope networks
/// such as `mesa_201.net` so two cache slots do not add ~430 MB RSS).
const MAX_CACHED_NET_JAC_ELEMS: usize = 8_000_000;

macro_rules! copy_pre_net_fields {
    ($dst:expr, $src:expr, [ $($field:ident),+ $(,)? ]) => {
        $( $dst.$field.clone_from(&$src.$field); )+
    };
}

impl CachedMicroOutputs {
    fn capture_pre_net(&mut self, model: &StarModel) {
        copy_pre_net_fields!(self, model, [
            rho, t, abar, zbar, z2bar, z53bar, ye, xh_mass, yhe_mass, z_mass,
            pgas, ln_pgas, prad, peos, ln_peos, energy, ln_e, entropy, ln_s,
            grada, cv, cp, chi_rho, chi_t, gamma1, gamma3, eta, gam, mu,
            lnfree_e, csound, ln_peos_ad, ln_e_ad, ln_s_ad, eos_res, eos_d_dlnd,
            eos_d_dlnt, opacity, d_opacity_dlnd, d_opacity_dlnt, tau,
            non_nuc_neu, d_non_nuc_neu_dlnd, d_non_nuc_neu_dlnt
        ]);
        copy_pre_net_fields!(self, model.aux, [
            eos_frac_pc, latent_ddlnt, latent_ddlnrho, dln_e_dxa_for_partials,
            dln_peos_dxa_for_partials
        ]);
    }

    fn capture_net(&mut self, model: &StarModel, has_net_jac: bool) {
        let store_jac = has_net_jac
            && model.d_dxdt_nuc_dx.len() <= MAX_CACHED_NET_JAC_ELEMS;
        self.has_net = true;
        self.has_net_jac = store_jac;
        copy_pre_net_fields!(self, model, [
            eps_nuc, d_epsnuc_dlnd, d_epsnuc_dlnt, eps_nuc_neu_total, dxdt_nuc
        ]);
        self.eps_nuc_categories
            .clone_from(&model.aux.eps_nuc_categories);
        if store_jac {
            copy_pre_net_fields!(self, model, [
                d_epsnuc_dx, d_dxdt_nuc_dlnd, d_dxdt_nuc_dlnt, d_dxdt_nuc_dx
            ]);
        } else {
            self.d_epsnuc_dx.clear();
            self.d_epsnuc_dx.shrink_to_fit();
            self.d_dxdt_nuc_dlnd.clear();
            self.d_dxdt_nuc_dlnd.shrink_to_fit();
            self.d_dxdt_nuc_dlnt.clear();
            self.d_dxdt_nuc_dlnt.shrink_to_fit();
            self.d_dxdt_nuc_dx.clear();
            self.d_dxdt_nuc_dx.shrink_to_fit();
        }
    }

    fn restore_pre_net(&self, model: &mut StarModel) {
        model.set_m_and_dm();
        model.set_m_grav_and_grav();
        copy_pre_net_fields!(model, self, [
            rho, t, abar, zbar, z2bar, z53bar, ye, xh_mass, yhe_mass, z_mass,
            pgas, ln_pgas, prad, peos, ln_peos, energy, ln_e, entropy, ln_s,
            grada, cv, cp, chi_rho, chi_t, gamma1, gamma3, eta, gam, mu,
            lnfree_e, csound, ln_peos_ad, ln_e_ad, ln_s_ad, eos_res, eos_d_dlnd,
            eos_d_dlnt, opacity, d_opacity_dlnd, d_opacity_dlnt, tau,
            non_nuc_neu, d_non_nuc_neu_dlnd, d_non_nuc_neu_dlnt
        ]);
        copy_pre_net_fields!(model.aux, self, [
            eos_frac_pc, latent_ddlnt, latent_ddlnrho, dln_e_dxa_for_partials,
            dln_peos_dxa_for_partials
        ]);
    }

    fn restore_net(
        &self,
        model: &mut StarModel,
        doing_split: bool,
        skip_net_jacobian: bool,
    ) {
        let nz = model.nz;
        let sp = model.species;
        if nz == 0 || sp == 0 {
            return;
        }
        if model.aux.eps_nuc_categories.len() != nz {
            model
                .aux
                .eps_nuc_categories
                .resize(nz, [0.0; NUM_CATEGORIES]);
        }
        if !doing_split {
            model.eps_nuc.clone_from(&self.eps_nuc);
            model.d_epsnuc_dlnd.clone_from(&self.d_epsnuc_dlnd);
            model.d_epsnuc_dlnt.clone_from(&self.d_epsnuc_dlnt);
            model.eps_nuc_neu_total.clone_from(&self.eps_nuc_neu_total);
            model
                .aux
                .eps_nuc_categories
                .clone_from(&self.eps_nuc_categories);
            model.dxdt_nuc.clone_from(&self.dxdt_nuc);
            if skip_net_jacobian {
                model.d_epsnuc_dx.fill(0.0);
                model.d_dxdt_nuc_dlnd.fill(0.0);
                model.d_dxdt_nuc_dlnt.fill(0.0);
                model.d_dxdt_nuc_dx.fill(0.0);
            } else {
                model.d_epsnuc_dx.clone_from(&self.d_epsnuc_dx);
                model.d_dxdt_nuc_dlnd.clone_from(&self.d_dxdt_nuc_dlnd);
                model.d_dxdt_nuc_dlnt.clone_from(&self.d_dxdt_nuc_dlnt);
                model.d_dxdt_nuc_dx.clone_from(&self.d_dxdt_nuc_dx);
            }
            return;
        }

        // In operator-split mode inside `do_struct_burn_mix`, `do_net` leaves
        // zones with `T_start >= op_split_burn_min_T` untouched so
        // `burn_avg_epsnuc` and burnt-zone categories remain intact.
        let split_min_t = model.controls.op_split_burn_min_t;
        for k in 0..nz {
            let t_start_k = model
                .ln_t_start
                .get(k)
                .copied()
                .map_or(model.t[k], flexmesa_math::exp);
            if t_start_k >= split_min_t {
                continue;
            }
            model.eps_nuc[k] = self.eps_nuc[k];
            model.d_epsnuc_dlnd[k] = self.d_epsnuc_dlnd[k];
            model.d_epsnuc_dlnt[k] = self.d_epsnuc_dlnt[k];
            model.eps_nuc_neu_total[k] = self.eps_nuc_neu_total[k];
            model.aux.eps_nuc_categories[k] = self.eps_nuc_categories[k];
            let s0 = k * sp;
            let s1 = s0 + sp;
            model.dxdt_nuc[s0..s1].copy_from_slice(&self.dxdt_nuc[s0..s1]);
            let j0 = k * sp * sp;
            let j1 = j0 + sp * sp;
            if skip_net_jacobian {
                model.d_epsnuc_dx[s0..s1].fill(0.0);
                model.d_dxdt_nuc_dlnd[s0..s1].fill(0.0);
                model.d_dxdt_nuc_dlnt[s0..s1].fill(0.0);
                model.d_dxdt_nuc_dx[j0..j1].fill(0.0);
            } else {
                model.d_epsnuc_dx[s0..s1]
                    .copy_from_slice(&self.d_epsnuc_dx[s0..s1]);
                model.d_dxdt_nuc_dlnd[s0..s1]
                    .copy_from_slice(&self.d_dxdt_nuc_dlnd[s0..s1]);
                model.d_dxdt_nuc_dlnt[s0..s1]
                    .copy_from_slice(&self.d_dxdt_nuc_dlnt[s0..s1]);
                model.d_dxdt_nuc_dx[j0..j1]
                    .copy_from_slice(&self.d_dxdt_nuc_dx[j0..j1]);
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CachedMicroEntry {
    key: CachedMicroKey,
    out: CachedMicroOutputs,
}

/// Per-model 2-slot LRU cache for zone microphysics (`set_basic_vars`,
/// `do_eos`, `do_kap`, `do_neu`, and `do_net`).
///
/// Cloning a `MicrophysicsCache` always yields an empty cache so checkpoint or
/// Python wrapper clones never duplicate cached arrays.
#[derive(Debug, Default)]
pub struct MicrophysicsCache {
    slots: [Option<Box<CachedMicroEntry>>; 2],
    mru: usize,
}

impl Clone for MicrophysicsCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[inline]
fn eval_raw_microphysics(
    model: &mut StarModel,
    skip_net: bool,
    skip_net_jacobian: bool,
) -> Result<(), StarError> {
    set_basic_vars(model)?;
    do_eos(model)?;
    do_kap(model)?;
    do_neu(model)?;
    if !skip_net {
        let net_info = resolve_model_net(model)?;
        do_net(model, &net_info, skip_net_jacobian)?;
    }
    Ok(())
}

impl MicrophysicsCache {
    /// Clears both cached slots.
    pub fn clear(&mut self) {
        self.slots = [None, None];
        self.mru = 0;
    }

    fn eval_or_restore_inner(
        &mut self,
        model: &mut StarModel,
        doing_split: bool,
        skip_net: bool,
        skip_net_jacobian: bool,
    ) -> Result<(), StarError> {
        let matched_slot = self.slots.iter().position(|slot| {
            slot.as_ref()
                .is_some_and(|e| e.key.matches(model, doing_split))
        });

        if let Some(idx) = matched_slot {
            let entry = self.slots[idx].as_mut().expect("matched slot exists");
            let net_ready = skip_net
                || (entry.out.has_net
                    && (skip_net_jacobian || entry.out.has_net_jac));
            if net_ready {
                entry.out.restore_pre_net(model);
                if !skip_net {
                    entry.out.restore_net(
                        model,
                        doing_split,
                        skip_net_jacobian,
                    );
                }
                self.mru = idx;
                model.micro_cache_hits += 1;
                return Ok(());
            }

            // Pre-net outputs match; populate only `do_net` (or its Jacobian).
            entry.out.restore_pre_net(model);
            model.micro_cache_misses += 1;
            let net_info = resolve_model_net(model)?;
            do_net(model, &net_info, skip_net_jacobian)?;
            entry.out.capture_net(model, !skip_net_jacobian);
            self.mru = idx;
            return Ok(());
        }

        model.micro_cache_misses += 1;
        eval_raw_microphysics(model, skip_net, skip_net_jacobian)?;

        let target_slot = if self.slots[0].is_none() {
            0
        } else if self.slots[1].is_none() {
            1
        } else {
            1 - self.mru.min(1)
        };
        let entry = self.slots[target_slot]
            .get_or_insert_with(|| Box::new(CachedMicroEntry::default()));
        entry.key.capture_from(model, doing_split);
        entry.out.capture_pre_net(model);
        if skip_net {
            entry.out.has_net = false;
            entry.out.has_net_jac = false;
        } else {
            entry.out.capture_net(model, !skip_net_jacobian);
        }
        self.mru = target_slot;
        Ok(())
    }

    /// Evaluates or restores `set_basic_vars`, `do_eos`, `do_kap`, `do_neu`, and
    /// optional `do_net` while updating `micro_cache_hits` / `micro_cache_misses`.
    pub(crate) fn eval_or_restore(
        model: &mut StarModel,
        skip_net: bool,
        skip_net_jacobian: bool,
    ) -> Result<(), StarError> {
        if model.controls.use_other_kap || model.controls.use_other_neu {
            model.micro_cache_misses += 1;
            return eval_raw_microphysics(model, skip_net, skip_net_jacobian);
        }

        let doing_split =
            model.aux.doing_struct_burn_mix && model.controls.op_split_burn;
        let mut cache = std::mem::take(&mut model.aux.micro_cache);
        let res = cache.eval_or_restore_inner(
            model,
            doing_split,
            skip_net,
            skip_net_jacobian,
        );
        model.aux.micro_cache = cache;
        res
    }
}
