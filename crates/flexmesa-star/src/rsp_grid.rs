// SPDX-License-Identifier: LGPL-3.0-or-later
//! Smolec 1-based (`1..=nzn`) center-to-surface envelope grid, Marsaglia-Zaman
//! `RANMAR` generator (`star_utils.f90:3487-3535`), combined gas+radiation EOS
//! and `EOP` volume root-finder (`rsp_lina.f90:1170-1204`, `rsp_relax_env.f90:36-124`),
//! and `StarModel` synchronization helpers (`rsp_def.f90:492-517, 719-868`, `rsp.f90:154-201`).

use flexmesa_const::{MixingType, CRAD, PI};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::{RspDerivedConsts, F_EDD_ISOTROPIC};
use crate::rsp_eos_kap::{
    eval1_gamma_pt_get_rho, eval1_mesa_rho_given_pt, eval_mesa_eos_and_kap,
    RspChemCache,
};

/// Marsaglia & Zaman (1987) `RANMAR` uniform generator matching `init_random`
/// and `rand(s)` in `star/private/star_utils.f90:3487-3535` (`ijkl = 54217137`).
#[derive(Debug, Clone)]
pub struct MesaRanmar {
    u: [f64; 98],
    c: f64,
    cd: f64,
    cm: f64,
    i97: usize,
    j97: usize,
}

impl Default for MesaRanmar {
    fn default() -> Self {
        Self::new()
    }
}

impl MesaRanmar {
    /// Initializes `RANMAR` with upstream MESA's fixed seed `ijkl = 54217137`.
    #[must_use]
    pub fn new() -> Self {
        let ijkl: i64 = 54_217_137;
        let ij = ijkl / 30_082;
        let kl = ijkl - 30_082 * ij;
        let mut i = ((ij / 177) % 177) + 2;
        let mut j = (ij % 177) + 2;
        let mut k = ((kl / 169) % 178) + 1;
        let mut l = kl % 169;
        let mut u = [0.0_f64; 98];
        for slot in u.iter_mut().take(98).skip(1) {
            let mut x = 0.0_f64;
            let mut t = 0.5_f64;
            for _ in 1..=24 {
                let m = (((i * j) % 179) * k) % 179;
                i = j;
                j = k;
                k = m;
                l = (53 * l + 1) % 169;
                if ((l * m) % 64) >= 32 {
                    x += t;
                }
                t *= 0.5;
            }
            *slot = x;
        }
        Self {
            u,
            c: 362_436.0 / 16_777_216.0,
            cd: 7_654_321.0 / 16_777_216.0,
            cm: 16_777_213.0 / 16_777_216.0,
            i97: 97,
            j97: 33,
        }
    }

    /// Draws the next uniform deviate in $[0, 1)$ (`rand(s)`).
    pub fn next_f64(&mut self) -> f64 {
        let mut uni = self.u[self.i97] - self.u[self.j97];
        if uni < 0.0 {
            uni += 1.0;
        }
        self.u[self.i97] = uni;
        self.i97 = if self.i97 == 1 { 97 } else { self.i97 - 1 };
        self.j97 = if self.j97 == 1 { 97 } else { self.j97 - 1 };
        self.c -= self.cd;
        if self.c < 0.0 {
            self.c += self.cm;
        }
        uni -= self.c;
        if uni < 0.0 {
            uni += 1.0;
        }
        uni
    }
}

/// 1-based center-to-surface (`1..=nzn`) Smolec envelope arrays (`rsp_build.f90:70-76`).
#[derive(Debug, Clone)]
pub struct RspEnvGrid {
    /// Active zone count `NZN` (may decrease if `T_inner` is reached early).
    pub nzn: usize,
    /// Enclosed mass $M_i$ at outer face of zone $i$ ($\mathrm{g}$).
    pub m: Vec<f64>,
    /// Zone mass $\Delta m_i$ ($\mathrm{g}$).
    pub dm: Vec<f64>,
    /// Face control-volume mass $\overline{\Delta m}_i$ ($\mathrm{g}$).
    pub dm_bar: Vec<f64>,
    /// Outer face radius $R_i$ ($\mathrm{cm}$).
    pub r: Vec<f64>,
    /// Zone specific volume $V_i = 1/\rho_i$ ($\mathrm{cm^3\,g^{-1}}$).
    pub vol: Vec<f64>,
    /// Zone temperature $T_i$ ($\mathrm{K}$).
    pub t: Vec<f64>,
    /// Zone specific turbulent energy $e_{t,i} = w_i^2$ ($\mathrm{erg\,g^{-1}}$).
    pub et: Vec<f64>,
    /// Zone total specific internal energy $e_i$ ($\mathrm{erg\,g^{-1}}$).
    pub e: Vec<f64>,
    /// Zone total thermodynamic pressure $P_i = P_{\mathrm{gas}} + P_{\mathrm{rad}}$ ($\mathrm{dyn\,cm^{-2}}$).
    pub p: Vec<f64>,
    /// Face radiative luminosity $L_{r,i}$ ($\mathrm{erg\,s^{-1}}$).
    pub lr: Vec<f64>,
    /// Face convective luminosity $L_{c,i}$ ($\mathrm{erg\,s^{-1}}$).
    pub lc: Vec<f64>,
    /// Face pressure scale height $H_{p,i}$ ($\mathrm{cm}$).
    pub hp_face: Vec<f64>,
    /// Face superadiabatic gradient $Y_i$.
    pub y_face: Vec<f64>,
    /// Zone Rosseland opacity $\kappa_i$ ($\mathrm{cm^2\,g^{-1}}$).
    pub k: Vec<f64>,
    /// Zone specific heat $C_{P,i}$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    pub cps: Vec<f64>,
    /// Zone thermal expansion coefficient $Q_i$ ($\mathrm{K^{-1}}$).
    pub qqs: Vec<f64>,
}

impl RspEnvGrid {
    /// Allocates 1-based (`0..=nzn + 1`) vectors for `nzn` zones.
    #[must_use]
    pub fn new(nzn: usize) -> Self {
        let len = nzn + 2;
        Self {
            nzn,
            m: vec![0.0; len],
            dm: vec![0.0; len],
            dm_bar: vec![0.0; len],
            r: vec![0.0; len],
            vol: vec![0.0; len],
            t: vec![0.0; len],
            et: vec![0.0; len],
            e: vec![0.0; len],
            p: vec![0.0; len],
            lr: vec![0.0; len],
            lc: vec![0.0; len],
            hp_face: vec![0.0; len],
            y_face: vec![0.0; len],
            k: vec![0.0; len],
            cps: vec![0.0; len],
            qqs: vec![0.0; len],
        }
    }
}

/// Combined gas + radiation EOS and opacity outputs matching `mesa_eos_kap`
/// (`star/private/rsp_lina.f90:1170-1204`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RspTotalEosKap {
    /// Total pressure $P = P_{\mathrm{gas}} + P_{\mathrm{rad}}$.
    pub p: f64,
    /// $\partial P / \partial V = \partial P_{\mathrm{gas}} / \partial V$.
    pub dpv: f64,
    /// $\partial P / \partial T$.
    pub dpt: f64,
    /// Total specific energy $e = e_{\mathrm{gas}} + e_{\mathrm{rad}}$.
    pub e: f64,
    /// $\partial e / \partial V$.
    pub dev: f64,
    /// $\partial e / \partial T$.
    pub det: f64,
    /// Specific heat $C_P$.
    pub cp: f64,
    /// $\partial C_P / \partial V$.
    pub cpv: f64,
    /// $\partial C_P / \partial T$.
    pub cpt: f64,
    /// Thermal expansion coefficient $Q$.
    pub qq: f64,
    /// $\partial Q / \partial V$.
    pub qqv: f64,
    /// $\partial Q / \partial T$.
    pub qqt: f64,
    /// Rosseland opacity $\kappa$.
    pub op: f64,
    /// $\partial \kappa / \partial V$.
    pub opv: f64,
    /// $\partial \kappa / \partial T$.
    pub opt: f64,
}

/// Evaluates combined gas + radiation EOS and opacity (`mesa_eos_kap` in `rsp_lina.f90:1170-1204`).
pub fn mesa_eos_kap(
    model: &mut StarModel,
    chem: &RspChemCache,
    k_smolec: usize,
    nzn: usize,
    t: f64,
    vol: f64,
) -> Result<RspTotalEosKap, StarError> {
    let cell_k = (1..=nzn).contains(&k_smolec).then(|| nzn - k_smolec);
    let o = eval_mesa_eos_and_kap(model, chem, cell_k, t, vol)?;
    Ok(RspTotalEosKap {
        p: o.pgas + o.prad,
        dpv: o.d_pg_dv,
        dpt: o.d_pg_dt + o.d_pr_dt,
        e: o.egas + o.erad,
        dev: o.d_egas_dv + o.d_erad_dv,
        det: o.d_egas_dt + o.d_erad_dt,
        cp: o.cp,
        cpv: o.cpv,
        cpt: o.cpt,
        qq: o.q,
        qqv: o.qv,
        qqt: o.qt,
        op: o.op,
        opv: o.opv,
        opt: o.opt,
    })
}

/// Solves $P(T, V) = P_{\mathrm{target}}$ for specific volume $V = 1/\rho$ (`EOP` in `rsp_relax_env.f90:36-124`).
pub fn eop(
    model: &mut StarModel,
    chem: &RspChemCache,
    nzn: usize,
    t: f64,
    p: f64,
) -> Result<(f64, RspTotalEosKap), StarError> {
    if !(p > 0.0 && p.is_finite() && t > 0.0 && t.is_finite()) {
        return Err(StarError::MicrophysicsError(format!(
            "EOP non-positive (T, P) = ({t}, {p})"
        )));
    }
    let prad = CRAD * t.powi(4) / 3.0;
    if p <= prad {
        return Err(StarError::MicrophysicsError(format!(
            "EOP P ({p}) <= Prad ({prad}) at T={t}"
        )));
    }
    let rho_guess = eval1_gamma_pt_get_rho(chem, p, t);
    let rho = eval1_mesa_rho_given_pt(model, chem, p, t, rho_guess)?;
    let mut vol = 1.0 / rho;
    for _ in 1..=25 {
        let out = mesa_eos_kap(model, chem, 0, nzn, t, vol)?;
        let f2 = out.p / p - 1.0;
        vol -= f2 * p / out.dpv;
        if !vol.is_finite() || vol <= 0.0 {
            return Err(StarError::ConvergenceFailure(format!(
                "EOP bad V={vol} at T={t}, P={p}"
            )));
        }
        if f2.abs() < 5.0e-13 {
            return Ok((vol, out));
        }
    }
    Err(StarError::ConvergenceFailure(format!(
        "EOP failed to converge in 25 iterations at T={t}, P={p}"
    )))
}

/// Synchronizes `StarModel` and rounds `(M, DM, DM_BAR, R, Vol, T, Et)` through
/// `lnR`, `lnT`, `lnd`, and `dq` prior to `do_LINA` (`cleanup_for_LINA` in `rsp_def.f90:719-782`).
pub fn cleanup_for_lina(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    grid: &mut RspEnvGrid,
) {
    let nzn = grid.nzn;
    if model.nz != nzn {
        model.resize_zones(nzn);
        model.aux.rsp.resize(nzn);
    }
    for i in 1..=nzn {
        let k = nzn - i;
        model.m[k] = grid.m[i];
        model.dq[k] = grid.dm[i] / model.xmstar;
        model.r[k] = grid.r[i];
        model.t[k] = grid.t[i];
        model.aux.rsp.et_rsp[k] = grid.et[i].max(0.0).sqrt().powi(2);
        model.peos[k] = grid.p[i];
        model.prad[k] = CRAD * model.t[k].powi(4) / 3.0;
        model.pgas[k] = model.peos[k] - model.prad[k];
    }
    model.dq[nzn - 1] = (model.m[nzn - 1] - model.m_center) / model.xmstar;
    let _ = model.normalize_dqs();
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.aux.rsp.fr_rsp.fill(0.0);
    model.aux.rsp.lc_rsp.fill(0.0);
    model.aux.rsp.lt_rsp.fill(0.0);
    copy_results(model, dc);

    for i in 1..=nzn {
        let k = nzn - i;
        grid.m[i] = model.m[k];
        grid.dm[i] = model.dm[k];
        grid.dm_bar[i] = model.dm_bar[k];
        grid.r[i] = model.r[k];
        grid.vol[i] = 1.0 / model.rho[k];
        grid.t[i] = model.t[k];
        grid.et[i] = model.aux.rsp.et_rsp[k];
    }
}

/// Applies exact `lnR`/`lnT`/`lnd` round-trip rounding and updates `Vol`, `L`,
/// `mixing_type`, `R_center`, and `M_center` (`copy_results` in `rsp_def.f90:785-868`).
pub fn copy_results(model: &mut StarModel, dc: &RspDerivedConsts) {
    let nzn = model.nz;
    for i in 1..=nzn {
        let k = nzn - i;
        let w = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
        model.aux.rsp.et_rsp[k] = w * w;
        model.ln_r[k] = model.r[k].ln();
        model.r[k] = model.ln_r[k].exp();
        model.ln_t[k] = model.t[k].ln();
        model.t[k] = model.ln_t[k].exp();
        model.peos[k] = model.pgas[k] + model.prad[k];
    }
    for k in 0..nzn {
        let r_in3 = if k + 1 < nzn {
            model.r[k + 1].powi(3)
        } else {
            model.r_center.powi(3)
        };
        model.rmid[k] = (0.5 * (model.r[k].powi(3) + r_in3)).cbrt();
    }
    for i in 1..=nzn {
        let k = nzn - i;
        let vol = if i == 1 {
            dc.p43 / model.dm[k] * (model.r[k].powi(3) - model.r_center.powi(3))
        } else {
            dc.p43 / model.dm[k] * (model.r[k].powi(3) - model.r[k + 1].powi(3))
        };
        model.lnd[k] = (1.0 / vol).ln();
        model.rho[k] = model.lnd[k].exp();
        let vol_rev = 1.0 / model.rho[k];
        model.lum[k] = 4.0 * PI * model.r[k].powi(2) * model.aux.rsp.fr_rsp[k]
            + model.aux.rsp.lc_rsp[k]
            + model.aux.rsp.lt_rsp[k];
        model.mixing_type[k] = if model.aux.rsp.et_rsp[k].sqrt() > 1.0e4 {
            MixingType::ConvectiveMixing
        } else {
            MixingType::NoMixing
        };
        if i == 1 {
            model.r_center =
                (model.r[k].powi(3) - vol_rev * model.dm[k] / dc.p43).cbrt();
        }
    }
    model.m_center = model.mstar - model.xmstar;
}

/// Copies post-`LINA` static fields onto `StarModel` and runs the exact
/// `finish_build_rsp_model` + `set_star_vars` + `finish_after_build_model` +
/// `copy_results` initialization sequence (`rsp_def.f90:447-586, 785-868` and
/// `rsp.f90:154-201, 294-317`).
pub fn finish_build_rsp_model(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    grid: &RspEnvGrid,
    l_surf: f64,
) {
    let nzn = grid.nzn;
    for i in 1..=nzn {
        let k = nzn - i;
        model.m[k] = grid.m[i];
        model.dm[k] = grid.dm[i];
        model.dm_bar[k] = grid.dm_bar[i];
        model.r[k] = grid.r[i];
        model.ln_r[k] = grid.r[i].ln();
        model.t[k] = grid.t[i];
        model.ln_t[k] = grid.t[i].ln();
        let rho = 1.0 / grid.vol[i];
        model.lnd[k] = rho.ln();
        model.rho[k] = model.lnd[k].exp();
        let w = grid.et[i].max(0.0).sqrt();
        model.aux.rsp.et_rsp[k] = w * w;
        model.aux.rsp.fr_rsp[k] = grid.lr[i] / (4.0 * PI * model.r[k].powi(2));
        model.aux.rsp.lc_rsp[k] = 0.0;
        model.aux.rsp.lt_rsp[k] = 0.0;
        model.aux.rsp.erad_rsp[k] = CRAD * model.t[k].powi(4) * grid.vol[i];
        model.prad[k] =
            F_EDD_ISOTROPIC * model.aux.rsp.erad_rsp[k] / grid.vol[i];
        model.peos[k] = grid.p[i];
        model.pgas[k] = model.peos[k] - model.prad[k];
        model.lum[k] = l_surf;
        model.lum_start[k] = 0.0;
        model.dq[k] = model.dm[k] / model.xmstar;
        model.cell_xa_mut(k).copy_from_slice(&chem.xa);
    }
    model.l_center = l_surf;
    model.dq[nzn - 1] = (model.m[nzn - 1] - model.m_center) / model.xmstar;
    let _ = model.normalize_dqs();
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.set_m_grav_and_grav();

    // `set_star_vars` (`rsp_def.f90:520-586`) + `finish_after_build_model`
    // (`rsp_def.f90:447-456`) + `copy_results` (`rsp_def.f90:785-868`).
    let sum_dm: f64 = model.dm[..nzn].iter().sum();
    for k in 0..nzn {
        model.dq[k] = model.dm[k] / sum_dm;
    }
    model.xmstar = sum_dm;
    let _ = model.normalize_dqs();
    model.set_qs();
    model.set_xqs();
    model.m[0] = model.mstar;
    for k in 1..nzn {
        model.dm[k - 1] = model.dq[k - 1] * model.xmstar;
        model.m[k] = model.m[k - 1] - model.dm[k - 1];
    }
    model.dm[nzn - 1] = model.m[nzn - 1] - model.m_center;
    model.set_dm_bar();
    model.set_m_grav_and_grav();
    for k in 0..nzn {
        let vol_s = if k + 1 == nzn {
            dc.p43 / model.dm[k] * (model.r[k].powi(3) - model.r_center.powi(3))
        } else {
            dc.p43 / model.dm[k] * (model.r[k].powi(3) - model.r[k + 1].powi(3))
        };
        model.lnd[k] = (1.0 / vol_s).ln();
        model.rho[k] = model.lnd[k].exp();
        model.ln_t[k] = model.t[k].ln();
        model.t[k] = model.ln_t[k].exp();
        model.ln_r[k] = model.r[k].ln();
        model.r[k] = model.ln_r[k].exp();
        let vol_hyd = if k + 1 == nzn {
            dc.p43 / model.dm[k] * (model.r[k].powi(3) - model.r_center.powi(3))
        } else {
            dc.p43 / model.dm[k] * (model.r[k].powi(3) - model.r[k + 1].powi(3))
        };
        model.aux.rsp.erad_rsp[k] = CRAD * model.t[k].powi(4) * vol_hyd;
        model.prad[k] = F_EDD_ISOTROPIC * model.aux.rsp.erad_rsp[k] / vol_hyd;
    }
    copy_results(model, dc);
}
