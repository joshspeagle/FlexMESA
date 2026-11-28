// SPDX-License-Identifier: GPL-3.0-or-later
//! Staggered Lagrangian grid geometry, mass-weighted face interpolation, and
//! state vector packing/snapshot operations on [`StarModel`]
//! (`star/private/star_utils.f90` and `star/private/hydro_vars.f90`).

use flexmesa_eos::{I_DE_DRHO, I_LN_E, I_LN_S, NUM_EOS_BASIC_RESULTS};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Returns the index `midq` of the largest entry of `dq`: the first maximum,
/// as Fortran `maxloc(dq(1:nz), dim=1)` returns it
/// (`star/private/star_utils.f90:687, 739`).
#[must_use]
pub(crate) fn midq_index_of(dq: &[f64]) -> usize {
    let mut midq = 0;
    for k in 1..dq.len() {
        if dq[k] > dq[midq] {
            midq = k;
        }
    }
    midq
}

/// Rescales `dq` (`len = nz`) so that it sums to one, summing in from both
/// boundaries to meet at the largest cell (`normalize_dqs`,
/// `star/private/star_utils.f90:669-709`).
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if any `dq` is not positive, as
/// upstream fails with `ierr = -1`.
pub(crate) fn normalize_dqs_slice(dq: &mut [f64]) -> Result<(), StarError> {
    let nz = dq.len();
    if nz == 0 {
        return Ok(());
    }
    let bad = |k: usize, dq: f64| {
        StarError::InvalidInput(format!("normalize_dqs: bad dq({k}) = {dq:e}"))
    };
    let mut kmin = 0;
    for k in 1..nz {
        if dq[k] < dq[kmin] {
            kmin = k;
        }
    }
    if dq[kmin] <= 0.0 {
        return Err(bad(kmin + 1, dq[kmin]));
    }
    let midq = midq_index_of(dq);
    // Surface inward, k = 1..midq (star_utils.f90:688-696).
    let mut dqsum1 = 0.0_f64;
    for k in 0..=midq {
        dqsum1 += dq[k];
        if dq[k] <= 0.0 {
            return Err(bad(k + 1, dq[k]));
        }
    }
    // Center outward, k = nz..midq+1 (star_utils.f90:697-704).
    let mut dqsum2 = 0.0_f64;
    for k in (midq + 1..nz).rev() {
        dqsum2 += dq[k];
        if dq[k] <= 0.0 {
            return Err(bad(k + 1, dq[k]));
        }
    }
    for v in dq.iter_mut() {
        *v /= dqsum1 + dqsum2;
    }
    Ok(())
}

/// Sets the outer-face mass fractions `q` from `dq` (both `len = nz`;
/// `set_qs`, `star/private/star_utils.f90:712-752`): cumulative from the
/// surface, `q(1) = 1`, `q(k) = q(k-1) - dq(k-1)`; only if some `q` leaves
/// `[0, 1]` are they redone as surface-inward and center-outward sums
/// meeting at the largest cell. `dq` is not renormalized
/// (`do_normalize_dqs_as_part_of_set_qs = .false.`, `#381`).
pub(crate) fn set_qs_slice(q: &mut [f64], dq: &[f64]) {
    let nz = dq.len();
    if nz == 0 {
        return;
    }
    q[0] = 1.0;
    let mut okay = true;
    for k in 1..nz {
        q[k] = q[k - 1] - dq[k - 1];
        if q[k] < 0.0 || q[k] > 1.0 {
            okay = false;
            break;
        }
    }
    if okay {
        return;
    }
    let midq = midq_index_of(dq);
    let mut dqsum1 = 0.0_f64;
    for k in 0..=midq {
        q[k] = 1.0 - dqsum1;
        dqsum1 += dq[k];
    }
    let mut dqsum2 = 0.0_f64;
    for k in (midq + 1..nz).rev() {
        dqsum2 += dq[k];
        q[k] = dqsum2;
    }
}

impl StarModel {
    /// Returns the index `midq` of the zone with the largest `dq`: the first
    /// maximum, as Fortran `maxloc(dq(1:nz), dim=1)` returns it
    /// (`star/private/star_utils.f90:687, 739`).
    #[must_use]
    pub fn midq_index(&self) -> usize {
        midq_index_of(&self.dq[..self.nz])
    }

    /// Rescales `dq` so that it sums to one, summing in from both boundaries
    /// to meet at the largest cell (`normalize_dqs`,
    /// `star/private/star_utils.f90:669-709`).
    ///
    /// Upstream calls this only for new or reshaped models (pre-MS
    /// construction, shell removal, RSP, the post-remesh `dq_old`); a model
    /// read from a file is not renormalized (`set_qs` with
    /// `do_normalize_dqs_as_part_of_set_qs = .false.`, `#381`).
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidInput`] if any `dq` is not positive, as
    /// upstream fails with `ierr = -1`.
    pub fn normalize_dqs(&mut self) -> Result<(), StarError> {
        let nz = self.nz;
        normalize_dqs_slice(&mut self.dq[..nz])
    }

    /// Sets the outer-face mass fractions `q` from `dq` (`set_qs`,
    /// `star/private/star_utils.f90:712-752`): cumulative from the surface,
    /// `q(1) = 1`, `q(k) = q(k-1) - dq(k-1)`; only if some `q` leaves
    /// `[0, 1]` are they redone as surface-inward and center-outward sums
    /// meeting at the largest cell. `dq` is not renormalized
    /// (`do_normalize_dqs_as_part_of_set_qs = .false.`, `#381`).
    pub fn set_qs(&mut self) {
        let nz = self.nz;
        set_qs_slice(&mut self.q[..nz], &self.dq[..nz]);
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

    /// Updates `dm`, `m`, `dm_bar`, and `m_grav` from `dq`, `q`, `M_center`
    /// and `xmstar` (`set_m_and_dm` and `set_dm_bar` in
    /// `star/private/star_utils.f90:635-666`). `xmstar` is an input: upstream
    /// never recomputes it as `mstar - M_center`, which can round differently.
    pub fn set_m_and_dm(&mut self) {
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
            self.dm_bar[nz - 1] =
                if self.controls.rsp_flag || self.controls.rsp2_flag {
                    0.5 * (self.dm[nz - 2] + self.dm[nz - 1])
                } else {
                    0.5 * self.dm[nz - 2] + self.dm[nz - 1]
                };
        }
    }

    /// Updates face radii `r`, gravitational masses `m_grav` (applying
    /// baryonic `mass_correction` when `use_mass_corrections` is enabled,
    /// `star/private/star_utils.f90:387-410`), cell volumetric mid-radii
    /// `rmid`, and face gravitational acceleration `grav`.
    pub fn set_m_grav_and_grav(&mut self) {
        let nz = self.nz;
        if nz == 0 {
            return;
        }
        if self.controls.use_mass_corrections {
            let mc_nz =
                self.aux.mass_correction.get(nz - 1).copied().unwrap_or(1.0);
            self.m_grav[nz - 1] = self.m_center + self.dm[nz - 1] * mc_nz;
            for k in (0..nz - 1).rev() {
                let mc =
                    self.aux.mass_correction.get(k).copied().unwrap_or(1.0);
                self.m_grav[k] = self.m_grav[k + 1] + self.dm[k] * mc;
            }
        } else {
            self.m_grav[..nz].copy_from_slice(&self.m[..nz]);
        }
        for k in 0..nz {
            self.r[k] = flexmesa_math::exp(self.ln_r[k]);
            self.grav[k] =
                self.cgrav[k] * self.m_grav[k] / (self.r[k] * self.r[k]);
        }
        for k in 0..nz {
            let r_inner = if k + 1 < nz {
                self.r[k + 1]
            } else {
                self.r_center
            };
            self.rmid[k] = if self.controls.rsp_flag {
                0.5 * (self.r[k] + r_inner)
            } else {
                flexmesa_math::pow(
                    0.5 * (flexmesa_math::pow3(self.r[k])
                        + flexmesa_math::pow3(r_inner)),
                    1.0 / 3.0,
                )
            };
        }
    }

    /// Reconstructs outer face radii `ln_r` outward from `r_center` using cell
    /// densities `lnd` and cell masses `dm` so the spherical continuity
    /// equation $\Delta(r^3) = 3\,\Delta m / (4\pi\rho)$ holds identically.
    pub fn reconstruct_ln_r_from_lnd_and_dm(&mut self) {
        let mut r3 = flexmesa_math::pow3(self.r_center);
        for k in (0..self.nz).rev() {
            let rho_k = flexmesa_math::exp(self.lnd[k]).max(1.0e-30);
            r3 += 3.0 * self.dm[k] / (flexmesa_const::FOUR_PI * rho_k);
            self.ln_r[k] =
                flexmesa_math::log(flexmesa_math::pow(r3.max(1.0), 1.0 / 3.0));
        }
    }

    /// Synchronizes `q`, `xq`, `dm`, `dm_bar`, `m`, `m_grav`, `r`, `rmid`,
    /// `grav`, and `xh` after `dq` or `ln_r` changes. `dq` is used as it is;
    /// callers that build a new mesh call [`Self::normalize_dqs`] first, as
    /// upstream does (`#381`).
    ///
    /// # Errors
    ///
    /// Kept fallible for its callers; currently it does not fail.
    pub fn sync_mesh_and_pack(&mut self) -> Result<(), StarError> {
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
        crate::micro::do_neu(self)?;
        self.save_start_values()?;
        self.save_old_values();
        // A built or loaded model is one generation
        // (`star/private/init.f90:1006, 1009`,
        // `star/private/read_model.f90:438, 463`).
        self.generations = 1;
        Ok(())
    }

    /// Returns face interpolation weights $(\alpha_k, \beta_k)$ at face `k`
    /// (`get_face_weights` in `star/private/star_utils.f90:4063-4071`):
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

    /// Face value `alfa*v(k) + beta*v(k-1)` with the weights of
    /// [`Self::get_face_weights`], `v(1)` at the surface face (the pattern
    /// of `s% csound_face(k)`, `star/private/micro.f90:91-101`).
    ///
    /// This is not MESA's `interp_val_to_pt` (`star_utils.f90:779-811`),
    /// which interpolates through four cells; see
    /// `pulse_gyre_data::interp_val_to_pt_slices` (#491).
    #[must_use]
    #[inline]
    pub fn face_value_alfa_beta(&self, v: &[f64], k: usize) -> f64 {
        if k == 0 {
            v[0]
        } else {
            let (alfa, beta) = self.get_face_weights(k);
            alfa * v[k] + beta * v[k - 1]
        }
    }

    /// `get_face_values` at face `k` (`star/private/star_utils.f90:2046-2060`):
    /// `v_face(1) = v_mid(1)` and
    /// `v_face(k) = (v_mid(k)*dq(k-1) + v_mid(k-1)*dq(k))/(dq(k-1) + dq(k))`.
    #[must_use]
    #[inline]
    pub fn get_face_value(&self, v: &[f64], k: usize) -> f64 {
        if k == 0 {
            v[0]
        } else {
            let dq_sum = self.dq[k - 1] + self.dq[k];
            (v[k] * self.dq[k - 1] + v[k - 1] * self.dq[k]) / dq_sum
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

    /// Packs `(lnd, ln_t, ln_r, lum, [v | u], [w, hp])` into `xh` (`pack_xh`).
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
            if let Some(iw) = self.layout.i_w {
                self.xh[base + iw] = self.aux.rsp.et_rsp[k].max(0.0).sqrt();
            }
            if let Some(ihp) = self.layout.i_hp {
                self.xh[base + ihp] = self.aux.rsp.hp_face[k].max(1.0);
            }
            if let Some(ia) = self.layout.i_alpha_rti {
                self.xh[base + ia] = self.aux.alpha_rti[k].max(0.0);
            }
        }
    }

    /// Unpacks `xh` into `(lnd, ln_t, ln_r, lum, [v | u], [w, hp], [alpha_rti])` (`unpack_xh`).
    pub fn unpack_xh(&mut self) {
        let nvar = self.layout.nvar_hydro;
        let w_fix = self.controls.rsp2.w_fix_if_neg;
        if self.layout.i_v.is_some() && self.aux.dxh_v.len() != self.nz {
            self.aux.dxh_v.resize(self.nz, 0.0);
        }
        if self.layout.i_u.is_some() && self.aux.dxh_u.len() != self.nz {
            self.aux.dxh_u.resize(self.nz, 0.0);
        }
        for k in 0..self.nz {
            let base = k * nvar;
            self.lnd[k] = self.xh[base + self.layout.i_lnd];
            self.ln_t[k] = self.xh[base + self.layout.i_lnt];
            self.ln_r[k] = self.xh[base + self.layout.i_lnr];
            self.lum[k] = self.xh[base + self.layout.i_lum];
            if let Some(iv) = self.layout.i_v {
                self.v[k] = self.xh[base + iv];
                self.aux.dxh_v[k] = 0.0;
            }
            if let Some(iu) = self.layout.i_u {
                self.u[k] = self.xh[base + iu];
                // `s% dxh_u(k) = 0` (`star/private/hydro_vars.f90:339`).
                self.aux.dxh_u[k] = 0.0;
            }
            if let Some(iw) = self.layout.i_w {
                let mut w = self.xh[base + iw];
                if w < 0.0 {
                    w = if w_fix < 0.0 { 0.0 } else { w_fix };
                    self.xh[base + iw] = w;
                }
                let et = w * w;
                self.aux.rsp.et_rsp[k] = et;
                self.aux.etrb[k] = et;
            }
            if let Some(ihp) = self.layout.i_hp {
                let raw_hp = self.xh[base + ihp];
                let hp = if raw_hp < 1.0 { 1.0 } else { raw_hp };
                self.xh[base + ihp] = hp;
                self.aux.rsp.hp_face[k] = hp;
            }
            if let Some(ia) = self.layout.i_alpha_rti {
                let raw_a = self.xh[base + ia];
                let a = if raw_a < 0.0 { 0.0 } else { raw_a };
                self.xh[base + ia] = a;
                self.aux.alpha_rti[k] = a;
            }
        }
    }

    /// Snapshots current state into `*_start` arrays at the beginning of a step.
    ///
    /// Propagates errors from Riemann face velocity reconstruction (`u_flag`)
    /// matching `save_start_values(s, ierr)` in
    /// `star/private/struct_burn_mix.f90:268-356`.
    pub fn save_start_values(&mut self) -> Result<(), StarError> {
        if self.controls.rsp2_flag {
            let alfap = crate::rsp2::X_ALFAP * self.controls.rsp2.alfap;
            for k in 0..self.nz {
                let et = self.aux.rsp.et_rsp[k].max(0.0);
                self.aux.etrb[k] = et;
                self.aux.rsp.w_start[k] = et.sqrt();
                self.aux.rsp.ptrb_start[k] = alfap * self.rho[k] * et;
            }
            self.aux.rsp.lt_start.clone_from(&self.aux.rsp.lt_rsp);
            self.aux.rsp.csound_start_rsp.clone_from(&self.csound);
        }
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
        // `save_start_values` leaves `total_internal_energy_old` alone
        // (`star/private/struct_burn_mix.f90:268-356`): it is set once per
        // step on the new mesh, before phase separation and element diffusion
        // change the energies (`evolve.f90:420-431, 622-639`).
        self.peos_start.clone_from(&self.peos);
        self.non_nuc_neu_start.clone_from(&self.non_nuc_neu);
        self.xa_start.clone_from(&self.xa);
        self.xa_sub_xa_start.clear();
        self.xh_start.clone_from(&self.xh);
        self.aux.opacity_start_surf =
            self.opacity.first().copied().unwrap_or(0.0);
        let mut de_drho = vec![0.0; self.nz];
        for (dst, row) in de_drho.iter_mut().zip(self.eos_res.iter()) {
            *dst = row[I_DE_DRHO];
        }
        self.aux.save_start(
            &self.cp,
            &self.cv,
            &de_drho,
            &self.grada,
            &self.chi_t,
            &self.chi_rho,
            &self.csound,
            &self.mlt_vc,
        );
        // `s% Pvsc_start(k) = -1d99` (`star/private/struct_burn_mix.f90:313`).
        self.aux.pvsc_start.clear();
        self.aux
            .pvsc_start
            .resize(self.nz, crate::hydro_state::PVSC_START_UNSET);
        if self.sig.iter().all(|&s| s == 0.0) {
            crate::hydro_vars::update_sig_from_d_mix(self)?;
        }
        if self.controls.rti_flag {
            crate::hydro_rti::set_rti_mixing_info(self);
        }
        if self.controls.u_flag {
            self.aux.p_face_start.resize(self.nz, -1.0);
            self.aux.p_face_start.fill(-1.0);
            self.aux.u_face_start.resize(self.nz, 0.0);
            crate::riemann_ad::do_uface_and_pface(self)?;
        }
        if self.controls.rti_flag {
            crate::hydro_rti::set_dpdr_drhodr_info(self);
        }
        Ok(())
    }

    /// Snapshots current state into `*_old` arrays after a successful step.
    ///
    /// Also stores `model_number_old = model_number`, as `new_generation`
    /// does (`star/private/evolve_support.f90:80`).
    pub fn save_old_values(&mut self) {
        self.model_number_old = self.model_number;
        self.star_age_old = self.star_age;
        self.dt_old = self.dt;
        self.mstar_old = self.mstar;
        self.lnd_old.clone_from(&self.lnd);
        self.ln_t_old.clone_from(&self.ln_t);
        self.ln_r_old.clone_from(&self.ln_r);
        self.lum_old.clone_from(&self.lum);
        self.v_old.clone_from(&self.v);
        self.u_old.clone_from(&self.u);
        self.xa_old.clone_from(&self.xa);
        self.dq_old.clone_from(&self.dq);
        self.aux.et_rsp_old.clone_from(&self.aux.rsp.et_rsp);
        self.aux.hp_face_old.clone_from(&self.aux.rsp.hp_face);
        self.aux.alpha_rti_old.clone_from(&self.aux.alpha_rti);
        self.aux.mlt_vc_old.clone_from(&self.mlt_vc);
        crate::core_boundaries::snapshot_step_history_old(self);
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
            &mut self.non_nuc_neu_start,
            &mut self.lnd_old,
            &mut self.ln_t_old,
            &mut self.ln_r_old,
            &mut self.lum_old,
            &mut self.v_old,
            &mut self.u_old,
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
        self.xa_sub_xa_start.clear();
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
        self.aux.resize(nz, sp);
    }
}
