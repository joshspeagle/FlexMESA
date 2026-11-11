// SPDX-License-Identifier: LGPL-3.0-or-later
//! Homogeneous envelope abundance construction and $(T, V)$ EOS/opacity
//! partials for `RSP` (`star/private/rsp_eval_eos_and_kap.f90:1-693` and
//! `star/private/adjust_xyz.f90:991-1220`).

use flexmesa_chem::{
    chem_xsol, get_composition_info, lodders03_element_atom_percent,
    solar_element_zfrac, NuclideCatalog, SolarScale,
};
use flexmesa_const::CRAD;
use flexmesa_eos::{
    eos_dt_get_from_weights_with_tables, CmsTableSet, EosDtTableSet,
    EosEvalOutput, HelmTable, I_CHI_RHO, I_CHI_T, I_CP, I_ETA, I_GAMMA1,
    I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS,
};
use flexmesa_kap::get_kap_results;

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;

pub use crate::rsp_eos_inv::{
    eval1_gamma_pt_get_rho, eval1_mesa_eos_de_and_kap,
    eval1_mesa_eos_degas_and_kap, eval1_mesa_rho_given_pt,
    eval1_mesa_t_given_dp, get_surf_p_t_kap, set_rho_for_new_pgas,
    set_t_for_new_egas, set_t_for_new_energy, set_t_for_new_pgas,
};

/// Cached homogeneous `RSP` abundance state and species weights
/// (`rsp_def.f90:36-46` and `rsp_eval_eos_and_kap.f90:54-139`).
#[derive(Debug, Clone)]
pub struct RspChemCache {
    /// 1-based `chem_id` indices (`len = species`).
    pub chem_id: Vec<usize>,
    /// Normalized species mass fractions $X_j$ (`len = species`).
    pub xa: Vec<f64>,
    /// Hydrogen, helium, and metal mass fractions $(X, Y, Z)$.
    pub x: f64,
    /// Helium mass fraction $Y$.
    pub y: f64,
    /// Metal mass fraction $Z$.
    pub z: f64,
    /// Composition moments $(\bar{A}, \bar{Z}, \overline{Z^{5/3}})$.
    pub abar: f64,
    /// Mean atomic charge $\bar{Z}$.
    pub zbar: f64,
    /// Mean $Z^{5/3}$.
    pub z53bar: f64,
    /// Total C, N, O, and Ne mass fractions for Type 2 opacities.
    pub xc: f64,
    /// Total nitrogen mass fraction.
    pub xn: f64,
    /// Total oxygen mass fraction.
    pub xo: f64,
    /// Total neon mass fraction.
    pub xne: f64,
    z_int: Vec<usize>,
    azion: Vec<f64>,
    acmi: Vec<f64>,
    a_baryon: Vec<f64>,
}

/// Thermodynamic and opacity outputs in $(T, V = 1/\rho)$ coordinates returned
/// by [`eval_mesa_eos_and_kap`] (`rsp_eval_eos_and_kap.f90:160-335`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RspEosKapOut {
    /// Gas pressure $P_{\mathrm{gas}}$ ($\mathrm{dyn\,cm^{-2}}$) and partials wrt $(V, T)$.
    pub pgas: f64,
    /// $\partial P_{\mathrm{gas}} / \partial V$.
    pub d_pg_dv: f64,
    /// $\partial P_{\mathrm{gas}} / \partial T$.
    pub d_pg_dt: f64,
    /// Radiation pressure $P_{\mathrm{rad}} = \frac{1}{3}a T^4$ and $\partial P_{\mathrm{rad}}/\partial T$.
    pub prad: f64,
    /// $\partial P_{\mathrm{rad}} / \partial T$.
    pub d_pr_dt: f64,
    /// Specific gas energy $e_{\mathrm{gas}} = e - e_{\mathrm{rad}}$ ($\mathrm{erg\,g^{-1}}$) and partials.
    pub egas: f64,
    /// $\partial e_{\mathrm{gas}} / \partial V$.
    pub d_egas_dv: f64,
    /// $\partial e_{\mathrm{gas}} / \partial T$.
    pub d_egas_dt: f64,
    /// Specific radiation energy $e_{\mathrm{rad}} = 3 P_{\mathrm{rad}} V$ and partials.
    pub erad: f64,
    /// $\partial e_{\mathrm{rad}} / \partial V$.
    pub d_erad_dv: f64,
    /// $\partial e_{\mathrm{rad}} / \partial T$.
    pub d_erad_dt: f64,
    /// Adiabatic sound speed $c_s$ ($\mathrm{cm\,s^{-1}}$).
    pub csnd: f64,
    /// Specific heat $C_P$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$) and $(V, T)$ partials.
    pub cp: f64,
    /// $\partial C_P / \partial V$.
    pub cpv: f64,
    /// $\partial C_P / \partial T$.
    pub cpt: f64,
    /// Thermal expansion coefficient $Q = \chi_T / (\rho T \chi_\rho)$ ($\mathrm{K^{-1}}$) and partials.
    pub q: f64,
    /// $\partial Q / \partial V$.
    pub qv: f64,
    /// $\partial Q / \partial T$.
    pub qt: f64,
    /// Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$) and $(V, T)$ partials.
    pub op: f64,
    /// $\partial \kappa / \partial V$.
    pub opv: f64,
    /// $\partial \kappa / \partial T$.
    pub opt: f64,
}

impl RspChemCache {
    /// Builds standard-metals `xa` from `(RSP_X, RSP_Z, initial_he3, initial_zfracs)`
    /// (`init_for_rsp_eos_and_kap` in `rsp_eval_eos_and_kap.f90:54-139` and
    /// `adjust_z_fractions` in `adjust_xyz.f90:1153-1220`).
    pub fn from_controls(
        chem_id: &[usize],
        job: &StarJobControls,
        controls: &StarControls,
    ) -> Result<Self, StarError> {
        let xa = compute_standard_metals_xa(
            chem_id,
            controls.rsp_x,
            controls.rsp_z,
            job.initial_he3,
            job.initial_zfracs.max(1) as usize,
            controls.rsp.initial_dump_missing_heaviest,
        )?;
        Self::from_xa(chem_id, xa)
    }

    /// Builds [`RspChemCache`] from zone `0` of an existing [`StarModel`]
    /// (`restart_rsp_eos_and_kap` when loading a saved `.mod` or `.photo`).
    pub fn from_model_zone0(model: &StarModel) -> Result<Self, StarError> {
        if model.nz == 0 || model.species == 0 {
            return Err(StarError::InvalidInput(String::from(
                "empty StarModel in RspChemCache::from_model_zone0",
            )));
        }
        Self::from_xa(&model.chem_id, model.cell_xa(0).to_vec())
    }

    /// Builds [`RspChemCache`] from explicit `chem_id` and `xa`.
    pub fn from_xa(chem_id: &[usize], xa: Vec<f64>) -> Result<Self, StarError> {
        let catalog = NuclideCatalog::builtin_ref();
        let comp =
            get_composition_info(catalog, chem_id, &xa, true).map_err(|e| {
                StarError::MicrophysicsError(format!("RSP composition: {e}"))
            })?;
        let n = chem_id.len();
        let mut z_int = Vec::with_capacity(n);
        let mut azion = Vec::with_capacity(n);
        let mut acmi = Vec::with_capacity(n);
        let mut a_baryon = Vec::with_capacity(n);
        let (mut xc, mut xn, mut xo, mut xne) = (0.0, 0.0, 0.0, 0.0);
        for (&cid, &x_i) in chem_id.iter().zip(xa.iter()) {
            let nuc = catalog.get_by_id(cid).ok_or_else(|| {
                StarError::InvalidInput(format!("unknown chem_id {cid}"))
            })?;
            z_int.push(nuc.z);
            azion.push(nuc.z as f64);
            acmi.push(nuc.w);
            a_baryon.push(nuc.z_plus_n as f64);
            match nuc.z {
                6 => xc += x_i,
                7 => xn += x_i,
                8 => xo += x_i,
                10 => xne += x_i,
                _ => {}
            }
        }
        Ok(Self {
            chem_id: chem_id.to_vec(),
            xa,
            x: comp.xh,
            y: comp.xhe,
            z: comp.xz,
            abar: comp.abar,
            zbar: comp.zbar,
            z53bar: comp.z53bar,
            xc,
            xn,
            xo,
            xne,
            z_int,
            azion,
            acmi,
            a_baryon,
        })
    }

    pub(crate) fn eval_eos_dt(
        &self,
        model: &StarModel,
        rho: f64,
        log_rho: f64,
        t: f64,
        log_t: f64,
    ) -> Result<EosEvalOutput, StarError> {
        eos_dt_get_from_weights_with_tables(
            HelmTable::builtin(),
            EosDtTableSet::builtin(),
            CmsTableSet::builtin(),
            self.x,
            self.z,
            rho,
            log_rho,
            t,
            log_t,
            self.abar,
            self.zbar,
            &self.xa,
            &self.z_int,
            &self.azion,
            &self.acmi,
            &self.a_baryon,
            &model.eos_config,
        )
        .map_err(|e| StarError::MicrophysicsError(format!("RSP eosDT: {e}")))
    }
}

/// Computes standard solar-scaled abundances `xa` for `chem_id`
/// (`get_xa_for_standard_metals` and `adjust_z_fractions` in `adjust_xyz.f90:991-1220`).
pub fn compute_standard_metals_xa(
    chem_id: &[usize],
    rsp_x: f64,
    rsp_z: f64,
    initial_he3: f64,
    initial_zfracs: usize,
    dump_missing_into_heaviest: bool,
) -> Result<Vec<f64>, StarError> {
    let species = chem_id.len();
    if species == 0 {
        return Err(StarError::InvalidInput(String::from("empty chem_id")));
    }
    let catalog = NuclideCatalog::builtin_ref();
    let initial_x = rsp_x.clamp(0.0, 1.0);
    let initial_z = rsp_z.clamp(0.0, 1.0);
    let initial_y = (1.0 - (initial_x + initial_z)).max(0.0);
    let (he3, he4) = if initial_he3 < 0.0 {
        let x3 = chem_xsol("he3");
        let x4 = chem_xsol("he4");
        let denom = x3 + x4;
        (initial_y * x3 / denom, initial_y * x4 / denom)
    } else if initial_he3 <= initial_y {
        (initial_he3, initial_y - initial_he3)
    } else {
        return Err(StarError::InvalidInput(format!(
            "initial_he3 ({initial_he3}) > initial_y ({initial_y})"
        )));
    };

    let mut xa = vec![0.0_f64; species];
    let mut isos = Vec::with_capacity(species);
    for (j, &cid) in chem_id.iter().enumerate() {
        let iso = catalog.get_by_id(cid).ok_or_else(|| {
            StarError::InvalidInput(format!("unknown chem_id {cid}"))
        })?;
        xa[j] = match iso.name.as_str() {
            "h1" => initial_x,
            "he3" => he3,
            "he4" => he4,
            _ => 0.0,
        };
        isos.push(iso);
    }

    let scale = SolarScale::from_id(initial_zfracs).unwrap_or(SolarScale::Gs98);
    let zfrac = solar_element_zfrac(scale, catalog);
    let jskip = dump_missing_into_heaviest.then(|| {
        let (mut wmax, mut best) = (-1.0_f64, 0_usize);
        for (j, iso) in isos.iter().enumerate() {
            if iso.w > wmax {
                wmax = iso.w;
                best = j;
            }
        }
        best
    });

    let mut new_xa = vec![0.0_f64; species];
    for (j, iso) in isos.iter().enumerate() {
        if Some(j) == jskip || iso.z <= 2 || iso.z > zfrac.len() {
            continue;
        }
        let frac = 0.01 * lodders03_element_atom_percent(&iso.name) * iso.w;
        if frac > 0.0 {
            let frac_sum: f64 = isos
                .iter()
                .filter(|other| other.z == iso.z)
                .map(|other| {
                    0.01 * lodders03_element_atom_percent(&other.name) * other.w
                })
                .sum();
            if frac_sum > 0.0 {
                new_xa[j] = initial_z * zfrac[iso.z - 1] * (frac / frac_sum);
            }
        }
    }

    let sum_metals: f64 = new_xa.iter().sum();
    if let Some(js) = jskip {
        new_xa[js] = (initial_z - sum_metals).max(0.0);
        for (j, iso) in isos.iter().enumerate() {
            if j == js || iso.z > 2 {
                xa[j] = new_xa[j];
            }
        }
    } else if sum_metals > 0.0 {
        for (j, iso) in isos.iter().enumerate() {
            if iso.z > 2 {
                xa[j] = new_xa[j] * initial_z / sum_metals;
            }
        }
    }

    let total: f64 = xa.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Err(StarError::InvalidInput(format!("bad sum(xa) = {total}")));
    }
    for v in &mut xa {
        *v /= total;
    }
    Ok(xa)
}

pub(crate) fn effective_opacity_factor(
    controls: &StarControls,
    log_t: f64,
) -> f64 {
    let mut f = controls.opacity_factor;
    let r = &controls.rsp;
    if f != 1.0 && r.min_logt_for_opacity_factor_off > 0.0 {
        if log_t >= r.max_logt_for_opacity_factor_off
            || log_t <= r.min_logt_for_opacity_factor_off
        {
            f = 1.0;
        } else if log_t > r.max_logt_for_opacity_factor_on {
            f = 1.0
                + (f - 1.0) * (log_t - r.max_logt_for_opacity_factor_off)
                    / (r.max_logt_for_opacity_factor_on
                        - r.max_logt_for_opacity_factor_off);
        } else if log_t < r.min_logt_for_opacity_factor_on {
            f = 1.0
                + (f - 1.0) * (log_t - r.min_logt_for_opacity_factor_off)
                    / (r.min_logt_for_opacity_factor_on
                        - r.min_logt_for_opacity_factor_off);
        }
    }
    f
}

fn store_eos_cell(
    model: &mut StarModel,
    chem: &RspChemCache,
    k: usize,
    rho: f64,
    t: f64,
    eos: &EosEvalOutput,
) {
    model.rho[k] = rho;
    model.lnd[k] = rho.ln();
    model.t[k] = t;
    model.ln_t[k] = t.ln();
    model.abar[k] = chem.abar;
    model.zbar[k] = chem.zbar;
    model.z53bar[k] = chem.z53bar;
    let ln_pgas = eos.res[I_LN_PGAS];
    let pgas = ln_pgas.exp();
    let prad = CRAD * t.powi(4) / 3.0;
    let peos = pgas + prad;
    model.ln_pgas[k] = ln_pgas;
    model.pgas[k] = pgas;
    model.prad[k] = prad;
    model.peos[k] = peos;
    model.ln_peos[k] = peos.ln();
    model.ln_e[k] = eos.res[I_LN_E];
    model.energy[k] = eos.res[I_LN_E].exp();
    model.cp[k] = eos.res[I_CP];
    model.chi_rho[k] = eos.res[I_CHI_RHO];
    model.chi_t[k] = eos.res[I_CHI_T];
    model.gamma1[k] = eos.res[I_GAMMA1];
    model.grada[k] = eos.res[I_GRAD_AD];
    model.lnfree_e[k] = eos.res[I_LN_FREE_E];
    model.eta[k] = eos.res[I_ETA];
    model.csound[k] = (eos.res[I_GAMMA1] * peos / rho).max(0.0).sqrt();
    model.eos_res[k] = eos.res;
    model.eos_d_dlnd[k] = eos.d_dlnd;
    model.eos_d_dlnt[k] = eos.d_dln_t;
}

fn eval1_kap_with_eos(
    model: &mut StarModel,
    chem: &RspChemCache,
    cell_k: Option<usize>,
    vol: f64,
    log_rho: f64,
    dlnd_dv: f64,
    t: f64,
    log_t: f64,
    eos: &EosEvalOutput,
) -> Result<(f64, f64, f64), StarError> {
    if model.controls.rsp.kap_density_factor > 0.0 {
        let f = model.controls.rsp.kap_density_factor;
        return Ok((f * vol, f, 0.0));
    }
    let kap_out = get_kap_results(
        &model.kap_config,
        chem.zbar,
        chem.x,
        chem.z,
        chem.xc,
        chem.xn,
        chem.xo,
        chem.xne,
        log_rho,
        log_t,
        eos.res[I_LN_FREE_E],
        eos.d_dlnd[I_LN_FREE_E],
        eos.d_dln_t[I_LN_FREE_E],
        eos.res[I_ETA],
        eos.d_dlnd[I_ETA],
        eos.d_dln_t[I_ETA],
    )
    .map_err(|e| StarError::MicrophysicsError(format!("RSP eval1_kap: {e}")))?;
    let factor = effective_opacity_factor(&model.controls, log_t);
    if let Some(k) = cell_k.filter(|&idx| idx < model.nz) {
        model.opacity[k] = kap_out.kap;
    }
    Ok((
        kap_out.kap * factor,
        kap_out.dlnkap_dln_rho * kap_out.kap * dlnd_dv,
        kap_out.dlnkap_dln_t * kap_out.kap / t,
    ))
}

/// Evaluates `eosDT` and Rosseland opacity in $(T, V = 1/\rho)$ coordinates
/// (`eval1_mesa_eos_and_kap` in `rsp_eval_eos_and_kap.f90:181-335`).
pub fn eval1_mesa_eos_and_kap(
    model: &mut StarModel,
    chem: &RspChemCache,
    cell_k: Option<usize>,
    skip_kap: bool,
    t: f64,
    vol: f64,
) -> Result<RspEosKapOut, StarError> {
    if !(t > 0.0 && t.is_finite() && vol > 0.0 && vol.is_finite()) {
        return Err(StarError::MicrophysicsError(format!(
            "non-positive RSP (T, V) = ({t}, {vol})"
        )));
    }
    let rho = 1.0 / vol;
    let log_rho = rho.log10();
    let log_t = t.log10();
    let dlnd_dv = -rho;
    let eos = chem.eval_eos_dt(model, rho, log_rho, t, log_t)?;
    if let Some(k) = cell_k.filter(|&idx| idx < model.nz) {
        store_eos_cell(model, chem, k, rho, t, &eos);
    }
    let (op, opv, opt) = if skip_kap {
        (0.0, 0.0, 0.0)
    } else {
        eval1_kap_with_eos(
            model, chem, cell_k, vol, log_rho, dlnd_dv, t, log_t, &eos,
        )?
    };

    let prad = CRAD * t.powi(4) / 3.0;
    let d_pr_dt = 4.0 * prad / t;
    let erad = 3.0 * prad / rho;
    let d_erad_dt = 3.0 * d_pr_dt / rho;
    let d_erad_dv = 3.0 * prad;

    let e_tot = eos.res[I_LN_E].exp();
    let de_dv = e_tot * eos.d_dlnd[I_LN_E] * dlnd_dv;
    let de_dt = e_tot * eos.d_dln_t[I_LN_E] / t;
    let egas = e_tot - erad;
    if !egas.is_finite() || egas <= 0.0 {
        return Err(StarError::MicrophysicsError(format!(
            "non-positive RSP egas={egas} at T={t}, V={vol}"
        )));
    }
    let pgas = eos.res[I_LN_PGAS].exp();
    let csnd = (eos.res[I_GAMMA1] * (prad + pgas) / rho).max(0.0).sqrt();
    let chi_t = eos.res[I_CHI_T];
    let chi_rho = eos.res[I_CHI_RHO];
    let q = chi_t / (rho * t * chi_rho);
    let dq_dlnd = q
        * (eos.d_dlnd[I_CHI_T] / chi_t - eos.d_dlnd[I_CHI_RHO] / chi_rho - 1.0);
    let dq_dlnt = q
        * (eos.d_dln_t[I_CHI_T] / chi_t
            - eos.d_dln_t[I_CHI_RHO] / chi_rho
            - 1.0);

    Ok(RspEosKapOut {
        pgas,
        d_pg_dv: pgas * eos.d_dlnd[I_LN_PGAS] * dlnd_dv,
        d_pg_dt: pgas * eos.d_dln_t[I_LN_PGAS] / t,
        prad,
        d_pr_dt,
        egas,
        d_egas_dv: de_dv - d_erad_dv,
        d_egas_dt: de_dt - d_erad_dt,
        erad,
        d_erad_dv,
        d_erad_dt,
        csnd,
        cp: eos.res[I_CP],
        cpv: eos.d_dlnd[I_CP] * dlnd_dv,
        cpt: eos.d_dln_t[I_CP] / t,
        q,
        qv: dq_dlnd * dlnd_dv,
        qt: dq_dlnt / t,
        op,
        opv,
        opt,
    })
}

/// Evaluates `eosDT` and Rosseland opacity (`skip_kap = false`).
pub fn eval_mesa_eos_and_kap(
    model: &mut StarModel,
    chem: &RspChemCache,
    cell_k: Option<usize>,
    t: f64,
    vol: f64,
) -> Result<RspEosKapOut, StarError> {
    eval1_mesa_eos_and_kap(model, chem, cell_k, false, t, vol)
}
