// SPDX-License-Identifier: LGPL-3.0-or-later
//! Lossless zone-vector accessors, resize helpers, and `.photo` scalar
//! serialization for [`RspState`] (`star/private/rsp_def.f90:240-620`).

use crate::rsp_def::{RspState, F_EDD_ISOTROPIC};

impl RspState {
    /// Returns immutable references to all `48` zone-indexed `RSP` vectors.
    #[must_use]
    pub fn zone_vecs(&self) -> [&[f64]; 48] {
        [
            &self.et_rsp,
            &self.erad_rsp,
            &self.fr_rsp,
            &self.lc_rsp,
            &self.lt_rsp,
            &self.ptrb_rsp,
            &self.pvsc_rsp,
            &self.egas_rsp,
            &self.hp_face,
            &self.vol_rsp,
            &self.f_edd_rsp,
            &self.y_face_rsp,
            &self.pii_rsp,
            &self.chi_rsp,
            &self.coupl_rsp,
            &self.eq_rsp,
            &self.qq_rsp,
            &self.source_rsp,
            &self.damp_rsp,
            &self.dampr_rsp,
            &self.vol_start,
            &self.w_start,
            &self.fr_start,
            &self.erad_start,
            &self.egas_start,
            &self.pgas_start,
            &self.prad_start,
            &self.opacity_start,
            &self.qq_start,
            &self.lc_start,
            &self.lt_start,
            &self.t_start_rsp,
            &self.r_start_rsp,
            &self.v_start_rsp,
            &self.csound_start_rsp,
            &self.pvsc_start,
            &self.ptrb_start,
            &self.chi_start,
            &self.coupl_start,
            &self.ppp0,
            &self.ppq0,
            &self.ppt0,
            &self.ppc0,
            &self.vv0,
            &self.work,
            &self.workq,
            &self.workt,
            &self.workc,
        ]
    }

    /// Returns mutable references to all `48` zone-indexed `RSP` vectors.
    pub fn zone_vecs_mut(&mut self) -> [&mut Vec<f64>; 48] {
        [
            &mut self.et_rsp,
            &mut self.erad_rsp,
            &mut self.fr_rsp,
            &mut self.lc_rsp,
            &mut self.lt_rsp,
            &mut self.ptrb_rsp,
            &mut self.pvsc_rsp,
            &mut self.egas_rsp,
            &mut self.hp_face,
            &mut self.vol_rsp,
            &mut self.f_edd_rsp,
            &mut self.y_face_rsp,
            &mut self.pii_rsp,
            &mut self.chi_rsp,
            &mut self.coupl_rsp,
            &mut self.eq_rsp,
            &mut self.qq_rsp,
            &mut self.source_rsp,
            &mut self.damp_rsp,
            &mut self.dampr_rsp,
            &mut self.vol_start,
            &mut self.w_start,
            &mut self.fr_start,
            &mut self.erad_start,
            &mut self.egas_start,
            &mut self.pgas_start,
            &mut self.prad_start,
            &mut self.opacity_start,
            &mut self.qq_start,
            &mut self.lc_start,
            &mut self.lt_start,
            &mut self.t_start_rsp,
            &mut self.r_start_rsp,
            &mut self.v_start_rsp,
            &mut self.csound_start_rsp,
            &mut self.pvsc_start,
            &mut self.ptrb_start,
            &mut self.chi_start,
            &mut self.coupl_start,
            &mut self.ppp0,
            &mut self.ppq0,
            &mut self.ppt0,
            &mut self.ppc0,
            &mut self.vv0,
            &mut self.work,
            &mut self.workq,
            &mut self.workt,
            &mut self.workc,
        ]
    }

    /// Resizes all zone-indexed `RSP` vectors to `nz`.
    pub fn resize(&mut self, nz: usize) {
        for v in self.zone_vecs_mut() {
            v.resize(nz, 0.0);
        }
        self.f_edd_rsp.fill(F_EDD_ISOTROPIC);
    }

    /// Packs scalar `RSP` cycle state and `RANMAR` state into `[f64; 155]` for `.photo` I/O.
    #[must_use]
    pub fn pack_scalars(&self) -> [f64; 155] {
        let mut out = [0.0_f64; 155];
        let head = [
            self.lina_periods[0],
            self.lina_periods[1],
            self.lina_periods[2],
            self.lina_growth_rates[0],
            self.lina_growth_rates[1],
            self.lina_growth_rates[2],
            self.num_periods as f64,
            self.period,
            self.period_lin,
            self.rsp_dt,
            self.delta_r,
            self.delta_mag,
            self.grpdv,
            self.grekm,
            self.grekm_avg_abs,
            self.psurf_from_atm,
            self.etot,
            self.e0,
            self.ede_start,
            self.econ,
            self.egrv,
            self.ethe,
            self.ekin,
            self.total_energy_old,
            self.te_start,
            self.tt1,
            self.un,
            self.ull,
            self.ekdel,
            self.vmax,
            self.rmax,
            self.rmin,
            self.lmax,
            self.lmin,
            self.ekmax,
            self.ekmin,
            self.ekmaxl,
            self.pdvwork,
            self.min_dr_div_cs,
            self.min_rad_diff_time,
            self.g_edd,
            self.rsp_tau_factor,
            self.time_sec,
            self.time_old_sec,
            f64::from(self.first),
            f64::from(self.inside),
            self.id_step as f64,
            self.min_k_for_turbulent_flux as f64,
            f64::from(u8::from(self.have_set_velocities)),
            f64::from(u8::from(self.just_set_velocities)),
            self.iwork as f64,
            0.0,
        ];
        out[..52].copy_from_slice(&head);
        out[52..155].copy_from_slice(&self.rng.pack());
        out
    }

    /// Unpacks scalar `RSP` cycle state and `RANMAR` state from a `.photo` slice.
    pub fn unpack_scalars(&mut self, s: &[f64]) {
        if s.len() < 155 {
            return;
        }
        self.lina_periods = [s[0], s[1], s[2]];
        self.lina_growth_rates = [s[3], s[4], s[5]];
        self.num_periods = s[6].max(0.0) as usize;
        (self.period, self.period_lin, self.rsp_dt) = (s[7], s[8], s[9]);
        (self.delta_r, self.delta_mag, self.grpdv) = (s[10], s[11], s[12]);
        (self.grekm, self.grekm_avg_abs, self.psurf_from_atm) =
            (s[13], s[14], s[15]);
        (self.etot, self.e0, self.ede_start, self.econ) =
            (s[16], s[17], s[18], s[19]);
        (self.egrv, self.ethe, self.ekin, self.total_energy_old) =
            (s[20], s[21], s[22], s[23]);
        (self.te_start, self.tt1, self.un, self.ull) =
            (s[24], s[25], s[26], s[27]);
        (self.ekdel, self.vmax) = (s[28], s[29]);
        (self.rmax, self.rmin, self.lmax, self.lmin) =
            (s[30], s[31], s[32], s[33]);
        (self.ekmax, self.ekmin, self.ekmaxl, self.pdvwork) =
            (s[34], s[35], s[36], s[37]);
        (self.min_dr_div_cs, self.min_rad_diff_time) = (s[38], s[39]);
        (self.g_edd, self.rsp_tau_factor) = (s[40], s[41]);
        (self.time_sec, self.time_old_sec) = (s[42], s[43]);
        self.first = s[44] as i32;
        self.inside = s[45] as i32;
        self.id_step = s[46].max(0.0) as usize;
        self.min_k_for_turbulent_flux = s[47].max(0.0) as usize;
        self.have_set_velocities = s[48] != 0.0;
        self.just_set_velocities = s[49] != 0.0;
        self.iwork = s[50].max(0.0) as usize;
        self.rng.unpack(&s[52..155]);
    }
}
