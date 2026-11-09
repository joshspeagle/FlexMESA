// SPDX-License-Identifier: LGPL-3.0-or-later
//! Staggered Lagrangian grid geometry, mass-weighted face interpolation, and
//! state vector packing/snapshot operations on [`StarModel`]
//! (`star/private/star_utils.f90` and `star/private/hydro_vars.f90`).

use flexmesa_eos::{I_LN_E, I_LN_S, NUM_EOS_BASIC_RESULTS};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

impl StarModel {
    /// Returns the index `midq` of the zone with the largest relative mass
    /// fraction `dq[midq]`, used as the meeting point for bidirectional
    /// surface-inward and center-outward summation (`star_utils.f90:1577`).
    #[must_use]
    pub fn midq_index(&self) -> usize {
        self.dq
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(idx, _)| idx)
    }

    /// Normalizes `dq` so $\sum_{k=0}^{nz-1} \Delta q_k = 1$ using
    /// bidirectional summation meeting at `midq = argmax(dq)` (`normalize_dqs`
    /// in `star/private/star_utils.f90:1577`).
    pub fn normalize_dqs(&mut self) -> Result<(), StarError> {
        if self.nz == 0 {
            return Ok(());
        }
        let sum: f64 = self.dq.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "invalid sum(dq) = {sum}"
            )));
        }
        for dq_k in &mut self.dq {
            *dq_k /= sum;
        }
        let midq = self.midq_index();
        let sum_outer: f64 = self.dq[..midq].iter().sum();
        let sum_inner: f64 = self.dq[midq + 1..].iter().sum();
        let dq_mid = 1.0 - (sum_outer + sum_inner);
        if dq_mid > 0.0 {
            self.dq[midq] = dq_mid;
        }
        Ok(())
    }

    /// Computes enclosed mass fractions `q` from `dq` by summing inward from
    /// the surface to `midq` and outward from the center to `midq + 1`
    /// (`set_qs` in `star/private/star_utils.f90:1592`).
    pub fn set_qs(&mut self) {
        let nz = self.nz;
        if nz == 0 {
            return;
        }
        let midq = self.midq_index();
        self.q[0] = 1.0;
        for k in 1..=midq {
            self.q[k] = (self.q[k - 1] - self.dq[k - 1]).max(0.0);
        }
        if nz >= 2 && midq + 1 < nz {
            self.q[nz - 1] = self.dq[nz - 1];
            for k in ((midq + 1)..(nz - 1)).rev() {
                self.q[k] = (self.q[k + 1] + self.dq[k]).min(1.0);
            }
        }
    }

    /// Computes exterior mass fractions `xq` (`1 - q`) from `dq` (`set_xqs` in
    /// `star/private/star_utils.f90:1557`).
    pub fn set_xqs(&mut self) {
        if self.nz == 0 {
            return;
        }
        self.xq[0] = 0.0;
        for k in 1..self.nz {
            self.xq[k] = self.xq[k - 1] + self.dq[k - 1];
        }
    }

    /// Updates `xmstar`, `dm`, `m`, `dm_bar`, and `m_grav` (`set_m_and_dm` and
    /// `set_dm_bar` in `star/private/star_utils.f90:679-709`).
    pub fn set_m_and_dm(&mut self) {
        self.xmstar = self.mstar - self.m_center;
        for k in 0..self.nz {
            self.dm[k] = self.dq[k] * self.xmstar;
            self.m[k] = self.m_center + self.q[k] * self.xmstar;
            self.m_grav[k] = self.m[k];
        }
        self.set_dm_bar();
    }

    /// Computes face control-volume masses `dm_bar` (`set_dm_bar` in
    /// `star/private/star_utils.f90:698`).
    pub fn set_dm_bar(&mut self) {
        let nz = self.nz;
        if nz == 0 {
            return;
        }
        self.dm_bar[0] = 0.5 * self.dm[0];
        for k in 1..nz.saturating_sub(1) {
            self.dm_bar[k] = 0.5 * (self.dm[k - 1] + self.dm[k]);
        }
        if nz >= 2 {
            self.dm_bar[nz - 1] = 0.5 * self.dm[nz - 2] + self.dm[nz - 1];
        }
    }

    /// Updates face radii `r`, cell volumetric mid-radii `rmid`, and face
    /// gravitational acceleration `grav` (`set_m_grav_and_grav` in
    /// `hydro_vars.f90`).
    pub fn set_m_grav_and_grav(&mut self) {
        let nz = self.nz;
        for k in 0..nz {
            self.r[k] = self.ln_r[k].exp();
            self.m_grav[k] = self.m[k];
            self.grav[k] =
                self.cgrav[k] * self.m_grav[k] / (self.r[k] * self.r[k]);
        }
        for k in 0..nz {
            let r_outer3 = self.r[k].powi(3);
            let r_inner3 = if k + 1 < nz {
                self.r[k + 1].powi(3)
            } else {
                self.r_center.powi(3)
            };
            self.rmid[k] = (0.5 * (r_outer3 + r_inner3)).cbrt();
        }
    }

    /// Reconstructs outer face radii `ln_r` outward from `r_center` using cell
    /// densities `lnd` and cell masses `dm` so the spherical continuity
    /// equation $\Delta(r^3) = 3\,\Delta m / (4\pi\rho)$ holds identically.
    pub fn reconstruct_ln_r_from_lnd_and_dm(&mut self) {
        let mut r3 = self.r_center.powi(3);
        for k in (0..self.nz).rev() {
            let rho_k = self.lnd[k].exp().max(1.0e-30);
            r3 += 3.0 * self.dm[k] / (flexmesa_const::FOUR_PI * rho_k);
            self.ln_r[k] = r3.max(1.0).cbrt().ln();
        }
    }

    /// Synchronizes `q`, `xq`, `dm`, `dm_bar`, `m`, `m_grav`, `r`, `rmid`,
    /// `grav`, and `xh` after `dq` or `ln_r` changes.
    pub fn sync_mesh_and_pack(&mut self) -> Result<(), StarError> {
        self.normalize_dqs()?;
        self.set_qs();
        self.set_xqs();
        self.set_m_and_dm();
        self.set_m_grav_and_grav();
        self.pack_xh();
        Ok(())
    }

    /// Synchronizes mesh geometry, evaluates initial EOS thermodynamic state,
    /// and snapshots `*_start` and `*_old` arrays for freshly initialized ZAMS
    /// or pre-MS models.
    pub fn finish_init_model(&mut self) -> Result<(), StarError> {
        self.sync_mesh_and_pack()?;
        crate::micro::set_basic_vars(self)?;
        crate::micro::do_eos(self)?;
        self.save_start_values();
        self.save_old_values();
        Ok(())
    }

    /// Returns face interpolation weights $(\alpha_k, \beta_k)$ at face `k`
    /// (`get_face_weights` in `star/private/star_utils.f90:1947`):
    /// $f_{\mathrm{face},k} = \alpha_k f_k + \beta_k f_{k-1}$.
    #[must_use]
    #[inline]
    pub fn get_face_weights(&self, k: usize) -> (f64, f64) {
        if k == 0 {
            (1.0, 0.0)
        } else {
            let dq_m1 = self.dq[k - 1];
            let dq_00 = self.dq[k];
            let denom = dq_m1 + dq_00;
            if denom > 0.0 {
                let alfa = dq_m1 / denom;
                (alfa, 1.0 - alfa)
            } else {
                (0.5, 0.5)
            }
        }
    }

    /// Interpolates a cell-centered array `v` to face `k` (`interp_val_to_pt`).
    #[must_use]
    #[inline]
    pub fn interp_val_to_pt(&self, v: &[f64], k: usize) -> f64 {
        if k == 0 {
            v[0]
        } else {
            let (alfa, beta) = self.get_face_weights(k);
            alfa * v[k] + beta * v[k - 1]
        }
    }

    /// Interpolates a cell-centered quantity
    /// $(v, \partial v/\partial\ln\rho, \partial v/\partial\ln T)$ to face `k`
    /// as a [`StarAdOrder1`] in cell $k$'s stencil.
    #[must_use]
    #[inline]
    pub fn interp_cell_ad_to_face(
        &self,
        val: &[f64],
        d_dlnd: &[f64],
        d_dlnt: &[f64],
        k: usize,
    ) -> StarAdOrder1 {
        let v_00 =
            StarAdOrder1::from_dlnd_dlnt_00(val[k], d_dlnd[k], d_dlnt[k]);
        if k == 0 {
            v_00
        } else {
            let v_m1 = StarAdOrder1::from_dlnd_dlnt_m1(
                val[k - 1],
                d_dlnd[k - 1],
                d_dlnt[k - 1],
            );
            let (alfa, beta) = self.get_face_weights(k);
            v_00 * alfa + v_m1 * beta
        }
    }

    /// Interpolates an EOS basic result slot `eos_slot` (`I_GRAD_AD`, `I_CP`,
    /// `I_CHI_RHO`, `I_CHI_T`, etc.) to face `k` as a [`StarAdOrder1`] in cell
    /// $k$'s stencil.
    #[must_use]
    #[inline]
    pub fn interp_eos_ad_to_face(
        &self,
        val: &[f64],
        eos_slot: usize,
        k: usize,
    ) -> StarAdOrder1 {
        let v_00 = StarAdOrder1::from_dlnd_dlnt_00(
            val[k],
            self.eos_d_dlnd[k][eos_slot],
            self.eos_d_dlnt[k][eos_slot],
        );
        if k == 0 {
            v_00
        } else {
            let km1 = k - 1;
            let v_m1 = StarAdOrder1::from_dlnd_dlnt_m1(
                val[km1],
                self.eos_d_dlnd[km1][eos_slot],
                self.eos_d_dlnt[km1][eos_slot],
            );
            let (alfa, beta) = self.get_face_weights(k);
            v_00 * alfa + v_m1 * beta
        }
    }

    /// Packs `(lnd, ln_t, ln_r, lum, [v | u])` into `xh` (`pack_xh`).
    pub fn pack_xh(&mut self) {
        let nvar = self.layout.nvar_hydro;
        self.xh.resize(self.nz * nvar, 0.0);
        for k in 0..self.nz {
            let base = k * nvar;
            self.xh[base + self.layout.i_lnd] = self.lnd[k];
            self.xh[base + self.layout.i_lnt] = self.ln_t[k];
            self.xh[base + self.layout.i_lnr] = self.ln_r[k];
            self.xh[base + self.layout.i_lum] = self.lum[k];
            if let Some(iv) = self.layout.i_v {
                self.xh[base + iv] = self.v[k];
            }
            if let Some(iu) = self.layout.i_u {
                self.xh[base + iu] = self.u[k];
            }
        }
    }

    /// Unpacks `xh` into `(lnd, ln_t, ln_r, lum, [v | u])` (`unpack_xh`).
    pub fn unpack_xh(&mut self) {
        let nvar = self.layout.nvar_hydro;
        for k in 0..self.nz {
            let base = k * nvar;
            self.lnd[k] = self.xh[base + self.layout.i_lnd];
            self.ln_t[k] = self.xh[base + self.layout.i_lnt];
            self.ln_r[k] = self.xh[base + self.layout.i_lnr];
            self.lum[k] = self.xh[base + self.layout.i_lum];
            if let Some(iv) = self.layout.i_v {
                self.v[k] = self.xh[base + iv];
            }
            if let Some(iu) = self.layout.i_u {
                self.u[k] = self.xh[base + iu];
            }
        }
    }

    /// Snapshots current state into `*_start` arrays at the beginning of a step.
    pub fn save_start_values(&mut self) {
        self.pack_xh();
        self.mstar_start = self.mstar;
        self.lnd_start.clone_from(&self.lnd);
        self.ln_t_start.clone_from(&self.ln_t);
        self.ln_r_start.clone_from(&self.ln_r);
        self.lum_start.clone_from(&self.lum);
        self.v_start.clone_from(&self.v);
        self.u_start.clone_from(&self.u);
        self.ln_pgas_start.clone_from(&self.ln_pgas);
        self.ln_peos_start.clone_from(&self.ln_peos);
        self.ln_e_start.clone_from(&self.ln_e);
        self.ln_s_start.clone_from(&self.ln_s);
        self.energy_start.clone_from(&self.energy);
        self.peos_start.clone_from(&self.peos);
        self.xa_start.clone_from(&self.xa);
        self.xh_start.clone_from(&self.xh);
        if self.sig.iter().all(|&s| s == 0.0) {
            crate::hydro_vars::update_sig_from_d_mix(self);
        }
    }

    /// Snapshots current state into `*_old` arrays after a successful step.
    pub fn save_old_values(&mut self) {
        self.star_age_old = self.star_age;
        self.dt_old = self.dt;
        self.mstar_old = self.mstar;
        self.lnd_old.clone_from(&self.lnd);
        self.ln_t_old.clone_from(&self.ln_t);
        self.ln_r_old.clone_from(&self.ln_r);
        self.lum_old.clone_from(&self.lum);
        self.xa_old.clone_from(&self.xa);
        self.dq_old.clone_from(&self.dq);
    }

    /// Updates `ln_peos_ad`, `ln_e_ad`, and `ln_s_ad` from EOS values and partials.
    pub fn refresh_cell_eos_ad(&mut self) {
        for k in 0..self.nz {
            let pgas = self.pgas[k];
            let prad = self.prad[k];
            let peos = self.peos[k].max(1.0e-30);
            let beta = pgas / peos;
            let alfa = prad / peos;
            let d_lnpeos_dlnd =
                beta * self.eos_d_dlnd[k][flexmesa_eos::I_LN_PGAS];
            let d_lnpeos_dlnt =
                beta * self.eos_d_dlnt[k][flexmesa_eos::I_LN_PGAS] + 4.0 * alfa;
            self.ln_peos_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
                self.ln_peos[k],
                d_lnpeos_dlnd,
                d_lnpeos_dlnt,
            );
            self.ln_e_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
                self.ln_e[k],
                self.eos_d_dlnd[k][I_LN_E],
                self.eos_d_dlnt[k][I_LN_E],
            );
            self.ln_s_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
                self.ln_s[k],
                self.eos_d_dlnd[k][I_LN_S],
                self.eos_d_dlnt[k][I_LN_S],
            );
        }
    }

    /// Resizes all zone-indexed vectors on [`StarModel`] to `nz` zones.
    pub fn resize_zones(&mut self, nz: usize) {
        let (sp, nv) = (self.species, self.layout.nvar_hydro);
        self.nz = nz;
        for v in [
            &mut self.lnd,
            &mut self.ln_t,
            &mut self.ln_r,
            &mut self.lum,
            &mut self.v,
            &mut self.u,
            &mut self.omega,
            &mut self.j_rot,
            &mut self.lnd_start,
            &mut self.ln_t_start,
            &mut self.ln_r_start,
            &mut self.lum_start,
            &mut self.v_start,
            &mut self.u_start,
            &mut self.ln_pgas_start,
            &mut self.ln_peos_start,
            &mut self.ln_e_start,
            &mut self.ln_s_start,
            &mut self.energy_start,
            &mut self.peos_start,
            &mut self.lnd_old,
            &mut self.ln_t_old,
            &mut self.ln_r_old,
            &mut self.lum_old,
            &mut self.dq_old,
            &mut self.dq,
            &mut self.q,
            &mut self.xq,
            &mut self.dm,
            &mut self.dm_bar,
            &mut self.m,
            &mut self.m_grav,
            &mut self.grav,
            &mut self.r,
            &mut self.rmid,
            &mut self.abar,
            &mut self.zbar,
            &mut self.z2bar,
            &mut self.z53bar,
            &mut self.ye,
            &mut self.xh_mass,
            &mut self.yhe_mass,
            &mut self.z_mass,
            &mut self.rho,
            &mut self.t,
            &mut self.pgas,
            &mut self.ln_pgas,
            &mut self.prad,
            &mut self.peos,
            &mut self.ln_peos,
            &mut self.energy,
            &mut self.ln_e,
            &mut self.entropy,
            &mut self.ln_s,
            &mut self.grada,
            &mut self.cv,
            &mut self.cp,
            &mut self.chi_rho,
            &mut self.chi_t,
            &mut self.gamma1,
            &mut self.gamma3,
            &mut self.eta,
            &mut self.gam,
            &mut self.mu,
            &mut self.lnfree_e,
            &mut self.csound,
            &mut self.opacity,
            &mut self.d_opacity_dlnd,
            &mut self.d_opacity_dlnt,
            &mut self.tau,
            &mut self.non_nuc_neu,
            &mut self.d_non_nuc_neu_dlnd,
            &mut self.d_non_nuc_neu_dlnt,
            &mut self.eps_nuc,
            &mut self.d_epsnuc_dlnd,
            &mut self.d_epsnuc_dlnt,
            &mut self.eps_nuc_neu_total,
            &mut self.eps_mdot,
            &mut self.gradt,
            &mut self.gradr,
            &mut self.gradl,
            &mut self.scale_height,
            &mut self.mlt_vc,
            &mut self.mlt_d,
            &mut self.mlt_cdc,
            &mut self.mlt_gamma,
            &mut self.d_mix,
            &mut self.conv_vel,
            &mut self.sig,
            &mut self.brunt_b,
            &mut self.brunt_n2,
        ] {
            v.resize(nz, 0.0);
        }
        let cg = self.cgrav.first().copied().unwrap_or(6.674_30e-8);
        self.cgrav.resize(nz, cg);
        for v in [
            &mut self.xa,
            &mut self.xa_start,
            &mut self.xa_old,
            &mut self.d_epsnuc_dx,
            &mut self.dxdt_nuc,
            &mut self.d_dxdt_nuc_dlnd,
            &mut self.d_dxdt_nuc_dlnt,
        ] {
            v.resize(nz * sp, 0.0);
        }
        self.xh.resize(nz * nv, 0.0);
        self.xh_start.resize(nz * nv, 0.0);
        self.d_dxdt_nuc_dx.resize(nz * sp * sp, 0.0);
        self.eos_res.resize(nz, [0.0; NUM_EOS_BASIC_RESULTS]);
        self.eos_d_dlnd.resize(nz, [0.0; NUM_EOS_BASIC_RESULTS]);
        self.eos_d_dlnt.resize(nz, [0.0; NUM_EOS_BASIC_RESULTS]);
        self.mixing_type
            .resize(nz, flexmesa_const::MixingType::NoMixing);
        for ad in [
            &mut self.ln_peos_ad,
            &mut self.ln_e_ad,
            &mut self.ln_s_ad,
            &mut self.eps_grav_ad,
            &mut self.gradt_ad,
            &mut self.gradr_ad,
            &mut self.gradl_ad,
            &mut self.scale_height_ad,
            &mut self.mlt_vc_ad,
            &mut self.mlt_d_ad,
            &mut self.mlt_cdc_ad,
        ] {
            ad.resize(nz, StarAdOrder1::ZERO);
        }
    }
}
