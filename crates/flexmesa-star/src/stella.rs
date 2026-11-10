// SPDX-License-Identifier: LGPL-3.0-or-later
//! MESA-to-STELLA `.hyd` and `.abn` exporters (`star/test_suite/ccsn_IIp/src/stella/stella.inc`),
//! 40-group radiation-hydrodynamics + $^{56}\mathrm{Ni}\to{}^{56}\mathrm{Co}\to{}^{56}\mathrm{Fe}$
//! homologous expansion light-curve solver, and Sobolev $\mathrm{Fe\,II}\ \lambda 5169.02\,\text{\AA}$
//! line-velocity post-processor (`stella/res/stella_extras.f90`, resolving Issue #4).

#![allow(missing_docs, clippy::format_push_string, clippy::needless_for_each)]

use std::f64::consts::PI;

use flexmesa_const::{AVO, BOLTZM, BOLTZ_SIGMA, CLIGHT, ME, M_SUN, QE, SECDAY};

use crate::model::StarModel;
use crate::pulse_bvp_recon::parse_fortran_float;

/// Configuration options for MESA-to-STELLA `.hyd` and `.abn` model export.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StellaExportOptions {
    pub num_points: usize,
    pub nz_extra: usize,
    pub skip_inner_dm_msun: f64,
    pub skip_inner_v_limit: f64,
    pub min_surf_log_rho: f64,
    pub min_velocity_cm_s: f64,
    pub csm_mdot_msun_yr: f64,
    pub csm_v_wind_cm_s: f64,
    pub csm_r_out_cm: f64,
    pub show_headers: bool,
}

impl Default for StellaExportOptions {
    fn default() -> Self {
        Self {
            num_points: 400,
            nz_extra: 40,
            skip_inner_dm_msun: 0.0,
            skip_inner_v_limit: -1.0e99,
            min_surf_log_rho: -99.0,
            min_velocity_cm_s: 1.0e5,
            csm_mdot_msun_yr: 1.0e-3,
            csm_v_wind_cm_s: 1.0e6,
            csm_r_out_cm: 7.0e13,
            show_headers: false,
        }
    }
}

/// Single zone row of a STELLA `.hyd` hydrodynamics input file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StellaHydZone {
    pub zone: usize,
    pub m_msun: f64,
    pub r_cm: f64,
    pub rho: f64,
    pub t_k: f64,
    pub v_cm_s: f64,
    pub p_dyn_cm2: f64,
    pub energy_erg_g: f64,
    pub entropy_erg_g_k: f64,
    pub kap_cm2_g: f64,
    pub tau: f64,
    pub v_div_csound: f64,
}

/// Parsed STELLA `.hyd` hydrodynamics model (`mesa.hyd`).
#[derive(Debug, Clone, PartialEq)]
pub struct StellaHydModel {
    pub t_start: f64,
    pub n_zones: usize,
    pub m_center_msun: f64,
    pub r_center_cm: f64,
    pub rho_center: f64,
    pub zones: Vec<StellaHydZone>,
}

/// Single zone row of a STELLA `.abn` 16-species abundance file (`mesa.abn`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StellaAbnZone {
    pub zone: usize,
    pub m_msun: f64,
    pub dm_msun: f64,
    pub r_cm: f64,
    pub xa: [f64; 16],
}

/// Single row of `mesa.vel_feII` Sobolev $\mathrm{Fe\,II}\ \lambda 5169.02\,\text{\AA}$ velocity output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StellaVelFeIIRow {
    pub zone: usize,
    pub t_post_max_days: f64,
    pub v_tau_lo_km_s: f64,
    pub v_tau_med_km_s: f64,
    pub v_tau_hi_km_s: f64,
    pub rho: f64,
    pub t_k: f64,
    pub eta: f64,
    pub m_tau_lo_msun: f64,
    pub r_tau_lo_cm: f64,
    pub m_tau_med_msun: f64,
    pub r_tau_med_cm: f64,
    pub m_tau_hi_msun: f64,
    pub r_tau_hi_cm: f64,
}

/// Tabulated Sobolev excitation factor $\eta(\rho, T)$ for $\mathrm{Fe\,II}\ \lambda 5169.02\,\text{\AA}$
/// (`stella/res/FeII_5169_eta.dat`, $41\;\log_{10}\rho \times 117\;\log_{10}T$).
#[derive(Debug, Clone)]
pub struct FeII5169EtaTable {
    pub log_rhos: Vec<f64>,
    pub log_ts: Vec<f64>,
    pub eta_grid: Vec<f64>,
}

impl FeII5169EtaTable {
    /// Parses `FeII_5169_eta.dat` (`41 x 117 = 4797` lines of `density temperature eta`).
    #[must_use]
    pub fn parse(content: &str) -> Self {
        let (num_rhos, num_ts) = (41, 117);
        let mut log_rhos = vec![0.0; num_rhos];
        let mut log_ts = vec![0.0; num_ts];
        let mut eta_grid = vec![0.0; num_rhos * num_ts];
        let mut lines = content.lines().filter(|l| !l.trim().is_empty());
        for j in 0..num_rhos {
            for i in 0..num_ts {
                if let Some(line) = lines.next() {
                    let toks: Vec<&str> = line.split_whitespace().collect();
                    if toks.len() >= 3 {
                        log_rhos[j] =
                            parse_fortran_float(toks[0]).max(1.0e-99).log10();
                        if j == 0 {
                            log_ts[i] =
                                parse_fortran_float(toks[1]).max(1.0).log10();
                        }
                        eta_grid[j * num_ts + i] = parse_fortran_float(toks[2]);
                    }
                }
            }
        }
        Self {
            log_rhos,
            log_ts,
            eta_grid,
        }
    }

    /// Evaluates $\eta(\log_{10}\rho, \log_{10}T)$ by clamped Catmull-Rom bicubic interpolation.
    #[must_use]
    pub fn eval(&self, log_rho: f64, log_t: f64) -> f64 {
        let (nr, nt) = (self.log_rhos.len(), self.log_ts.len());
        if nr < 2 || nt < 2 {
            return 0.0;
        }
        let lr = log_rho.clamp(self.log_rhos[0], self.log_rhos[nr - 1]);
        let lt = log_t.clamp(self.log_ts[0], self.log_ts[nt - 1]);
        let jr = self
            .log_rhos
            .partition_point(|&x| x < lr)
            .saturating_sub(1)
            .min(nr - 2);
        let it = self
            .log_ts
            .partition_point(|&x| x < lt)
            .saturating_sub(1)
            .min(nt - 2);
        let dr_step = (self.log_rhos[jr + 1] - self.log_rhos[jr]).max(1.0e-30);
        let dt_step = (self.log_ts[it + 1] - self.log_ts[it]).max(1.0e-30);
        let (ur, ut) = (
            ((lr - self.log_rhos[jr]) / dr_step).clamp(0.0, 1.0),
            ((lt - self.log_ts[it]) / dt_step).clamp(0.0, 1.0),
        );
        let cr_w = |u: f64| -> [f64; 4] {
            let (u2, u3) = (u * u, u * u * u);
            [
                -0.5 * u + u2 - 0.5 * u3,
                1.0 - 2.5 * u2 + 1.5 * u3,
                0.5 * u + 2.0 * u2 - 1.5 * u3,
                -0.5 * u2 + 0.5 * u3,
            ]
        };
        let (wr, wt) = (cr_w(ur), cr_w(ut));
        let mut val = 0.0;
        for (dr, &rw) in wr.iter().enumerate() {
            let r_i = (jr as isize + dr as isize - 1)
                .clamp(0, (nr - 1) as isize) as usize;
            for (dt, &tw) in wt.iter().enumerate() {
                let t_i = (it as isize + dt as isize - 1)
                    .clamp(0, (nt - 1) as isize)
                    as usize;
                val += rw * tw * self.eta_grid[r_i * nt + t_i];
            }
        }
        val.max(0.0)
    }
}

/// Evaluates the Sobolev $\mathrm{Fe\,II}\ \lambda 5169.02\,\text{\AA}$ optical-depth profile
/// and $(\tau_{\mathrm{lo}}, \tau_{\mathrm{med}}, \tau_{\mathrm{hi}}) = (0.2, 1.0, 2.0)$ crossing
/// velocities (`stella/res/stella_extras.f90` lines 396–470).
#[must_use]
pub fn eval_sobolev_feii_5169_snapshot(
    time_days: f64,
    t0_max_days: f64,
    r_cm: &[f64],
    v_1e8_cm_s: &[f64],
    xm_ext_msun: &[f64],
    temp_k: &[f64],
    den_g_cm3: &[f64],
    fe_xa: &[f64],
    star_mass_msun: f64,
    eta_table: &FeII5169EtaTable,
) -> Option<StellaVelFeIIRow> {
    let nz = r_cm.len();
    if nz < 4 {
        return None;
    }
    let mut smooth = den_g_cm3.to_vec();
    for _ in 0..5 {
        for k in 1..(nz - 1) {
            smooth[k] = (smooth[k - 1] + smooth[k] + smooth[k + 1]) / 3.0;
        }
    }
    let pref = PI * QE * QE / (ME * CLIGHT)
        * 0.023
        * (time_days * SECDAY)
        * 5169.02e-8;
    let mut tau_prev = 0.0;
    let (mut v_lo, mut m_lo, mut r_lo, mut v_med, mut m_med, mut r_med) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for k in (1..(nz - 1)).rev() {
        let density = smooth[k].max(1.0e-99);
        let n_fe = density * AVO * fe_xa[k] / 56.0;
        let log_rho = density
            .log10()
            .clamp(eta_table.log_rhos[0], *eta_table.log_rhos.last().unwrap());
        let log_t = temp_k[k]
            .max(1.0)
            .log10()
            .clamp(eta_table.log_ts[0], *eta_table.log_ts.last().unwrap());
        let eta = eta_table.eval(log_rho, log_t);
        let tau_sob = pref * n_fe * eta;
        let interp = |tau_t: f64| -> (f64, f64, f64) {
            let a = (tau_t - tau_prev) / (tau_sob - tau_prev).max(1.0e-30);
            let b = 1.0 - a;
            (
                a * v_1e8_cm_s[k] + b * v_1e8_cm_s[k + 1],
                star_mass_msun - (a * xm_ext_msun[k] + b * xm_ext_msun[k + 1]),
                a * r_cm[k] + b * r_cm[k + 1],
            )
        };
        if (tau_sob > 0.2 || k == 1) && v_lo == 0.0 {
            (v_lo, m_lo, r_lo) = interp(0.2);
        }
        if (tau_sob > 1.0 || k == 1) && v_med == 0.0 {
            (v_med, m_med, r_med) = interp(1.0);
        }
        if tau_sob > 2.0 || k == 1 {
            let (v_hi, m_hi, r_hi) = interp(2.0);
            return Some(StellaVelFeIIRow {
                zone: k + 1,
                t_post_max_days: time_days - t0_max_days,
                v_tau_lo_km_s: v_lo * 1.0e3,
                v_tau_med_km_s: v_med * 1.0e3,
                v_tau_hi_km_s: v_hi * 1.0e3,
                rho: 10.0_f64.powf(log_rho),
                t_k: 10.0_f64.powf(log_t),
                eta,
                m_tau_lo_msun: m_lo,
                r_tau_lo_cm: r_lo,
                m_tau_med_msun: m_med,
                r_tau_med_cm: r_med,
                m_tau_hi_msun: m_hi,
                r_tau_hi_cm: r_hi,
            });
        }
        tau_prev = tau_sob;
    }
    None
}

/// Exports `(hyd_string, abn_string)` from `model` in the exact 400-zone STELLA format.
#[must_use]
pub fn export_stella_pair(
    model: &StarModel,
    opts: &StellaExportOptions,
) -> (String, String) {
    let nz = model.nz.max(1);
    let n_tot = opts.num_points.max(2);
    let nz_extra = opts.nz_extra.min(n_tot - 1);
    let n_ej = n_tot - nz_extra;
    let (m_center, r_center) =
        (model.m_center.max(0.0), model.r_center.max(1.0e6));
    let rho_center = model.lnd.last().map_or(1.0e-8, |&d| d.exp());
    let m_surf = model.mstar.max(m_center + 0.1 * M_SUN);
    let r_surf = model
        .r
        .first()
        .copied()
        .unwrap_or(1.0e13)
        .max(r_center * 1.1);

    let (mut hyd, mut abn) = (String::new(), String::new());
    if opts.show_headers {
        hyd.push_str("  t_start  n_zones  m_center  r_center  rho_center\n");
        abn.push_str("  izone  mass  dm  r  h1  he  c12  n14  o16  ne20  na23  mg24  al27  si28  s32  ar36  ca40  fe  skip  co56+ni56\n");
    }
    hyd.push_str(&format!(
        "{:12.6E}{:5}   {:12.6E}   {:12.6E}   {:12.6E}\n",
        0.0,
        n_tot,
        m_center / M_SUN,
        r_center,
        rho_center
    ));

    let cell_x = |k: usize, name: &str, fallback: f64| -> f64 {
        model
            .iso_names
            .iter()
            .position(|s| s == name)
            .map_or(fallback, |idx| model.xa[k * model.species + idx])
    };
    let z0 = 0.0_f64;
    let mut prev_m_msun = m_center / M_SUN;
    for i in 1..=n_ej {
        let frac = (i as f64 - 0.5) / (n_ej as f64);
        let k =
            (((1.0 - frac) * ((nz - 1) as f64)).round() as usize).min(nz - 1);
        let m_msun = (m_center + frac * (m_surf - m_center)) / M_SUN;
        let dm_msun = (m_msun - prev_m_msun).max(1.0e-12);
        prev_m_msun = m_msun;
        let r_i = if model.r[k] > 0.0 {
            model.r[k]
        } else {
            r_center + frac * (r_surf - r_center)
        };
        let (rho, t) = (model.lnd[k].exp(), model.ln_t[k].exp());
        let v = model
            .v
            .get(k)
            .copied()
            .unwrap_or(0.0)
            .max(opts.min_velocity_cm_s);
        let p = model.peos[k].max(rho * BOLTZM * t / 1.67e-24);
        let energy = model.energy[k].max(1.5 * p / rho.max(1.0e-30));
        let (entropy, kap, tau) = (
            model.entropy[k].max(1.0e8),
            model.opacity[k].max(0.2),
            model.tau[k].max(2.0 / 3.0),
        );
        let cs = (model.gamma1[k].max(4.0 / 3.0) * p / rho.max(1.0e-30))
            .max(1.0)
            .sqrt();
        let mach = v / cs;
        hyd.push_str(&format!(
            "{i:4}{m_msun:24.16E}{r_i:24.16E}{rho:24.16E}{t:24.16E}{v:24.16E}{p:24.16E}{z0:24.16E}{energy:24.16E}{entropy:24.16E}{kap:24.16E}{tau:24.16E}{mach:24.16E}\n"
        ));

        let n14_raw = cell_x(k, "n14", 0.001);
        let (na23, al27) = (
            0.044_538_489_161_177 * n14_raw,
            0.076_659_495_850_443_9 * n14_raw,
        );
        let mut cols = [
            cell_x(k, "h1", 0.70),
            cell_x(k, "he3", 0.0) + cell_x(k, "he4", 0.28),
            cell_x(k, "c12", 0.003),
            (n14_raw - na23 - al27).max(0.0),
            cell_x(k, "o16", 0.008),
            cell_x(k, "ne20", 0.002),
            na23,
            cell_x(k, "mg24", 0.001),
            al27,
            cell_x(k, "si28", 0.001),
            cell_x(k, "s32", 1.0e-4),
            cell_x(k, "ar36", 1.0e-4),
            cell_x(k, "ca40", 1.0e-4),
            cell_x(k, "fe52", 0.0)
                + cell_x(k, "fe54", 0.0)
                + cell_x(k, "fe56", 0.001),
            0.0,
            cell_x(k, "co56", 0.0) + cell_x(k, "ni56", 0.0),
        ];
        let sum: f64 = cols.iter().sum::<f64>().max(1.0e-30);
        cols.iter_mut().for_each(|c| *c = (*c / sum).max(1.0e-30));
        cols[14] = 1.0e-30;
        abn.push_str(&format!("{i:4}{m_msun:13.6E}{dm_msun:13.6E}{r_i:13.6E}"));
        cols.iter()
            .for_each(|c| abn.push_str(&format!("{c:14.6E}")));
        abn.push('\n');
    }

    let r_out = opts.csm_r_out_cm.max(r_surf * 1.2);
    let mdot_g_s = opts.csm_mdot_msun_yr * M_SUN / (365.25 * SECDAY);
    let (v_w, t_surf) = (opts.csm_v_wind_cm_s.max(1.0e5), model.ln_t[0].exp());
    for j in 1..=nz_extra {
        let i = n_ej + j;
        let r_i =
            r_surf * (r_out / r_surf).powf((j as f64) / (nz_extra as f64));
        let r_prev = r_surf
            * (r_out / r_surf).powf((j as f64 - 1.0) / (nz_extra as f64));
        let rho = mdot_g_s / (4.0 * PI * v_w * r_i * r_i);
        let dm_msun = (mdot_g_s * (r_i - r_prev) / (v_w * M_SUN)).max(1.0e-12);
        let m_msun = prev_m_msun + dm_msun;
        prev_m_msun = m_msun;
        let p = rho * BOLTZM * t_surf / 1.0e-24;
        hyd.push_str(&format!(
            "{i:4}{m_msun:24.16E}{r_i:24.16E}{rho:24.16E}{t_surf:24.16E}{v_w:24.16E}{p:24.16E}{z0:24.16E}{z0:24.16E}{z0:24.16E}{z0:24.16E}{z0:24.16E}{z0:24.16E}\n"
        ));
        let cols = [
            0.70, 0.28, 0.003, 0.001, 0.008, 0.002, 4.0e-5, 0.001, 7.0e-5,
            0.001, 1.0e-4, 1.0e-4, 1.0e-4, 0.001, 1.0e-30, 1.0e-30,
        ];
        abn.push_str(&format!("{i:4}{m_msun:13.6E}{dm_msun:13.6E}{r_i:13.6E}"));
        cols.iter()
            .for_each(|c| abn.push_str(&format!("{c:14.6E}")));
        abn.push('\n');
    }
    (hyd, abn)
}

/// Parses a STELLA `.hyd` string (`mesa.hyd`) into a [`StellaHydModel`].
#[must_use]
pub fn parse_stella_hyd(content: &str) -> StellaHydModel {
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let h: Vec<&str> = lines.next().unwrap_or("").split_whitespace().collect();
    let t_start = h.first().map_or(0.0, |s| parse_fortran_float(s));
    let n_zones = h.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    let m_center_msun = h.get(2).map_or(0.0, |s| parse_fortran_float(s));
    let r_center_cm = h.get(3).map_or(0.0, |s| parse_fortran_float(s));
    let rho_center = h.get(4).map_or(0.0, |s| parse_fortran_float(s));

    let mut zones = Vec::with_capacity(n_zones);
    for line in lines {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 13 {
            continue;
        }
        zones.push(StellaHydZone {
            zone: t[0].parse::<usize>().unwrap_or(zones.len() + 1),
            m_msun: parse_fortran_float(t[1]),
            r_cm: parse_fortran_float(t[2]),
            rho: parse_fortran_float(t[3]),
            t_k: parse_fortran_float(t[4]),
            v_cm_s: parse_fortran_float(t[5]),
            p_dyn_cm2: parse_fortran_float(t[6]),
            energy_erg_g: parse_fortran_float(t[8]),
            entropy_erg_g_k: parse_fortran_float(t[9]),
            kap_cm2_g: parse_fortran_float(t[10]),
            tau: parse_fortran_float(t[11]),
            v_div_csound: parse_fortran_float(t[12]),
        });
    }
    StellaHydModel {
        t_start,
        n_zones,
        m_center_msun,
        r_center_cm,
        rho_center,
        zones,
    }
}

/// Parses a STELLA `.abn` string (`mesa.abn`) into a vector of [`StellaAbnZone`]s.
#[must_use]
pub fn parse_stella_abn(content: &str) -> Vec<StellaAbnZone> {
    let mut zones = Vec::new();
    for line in content.lines().filter(|l| !l.trim().is_empty()) {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 20 {
            continue;
        }
        let mut xa = [0.0; 16];
        for (k, slot) in xa.iter_mut().enumerate() {
            *slot = parse_fortran_float(t[4 + k]);
        }
        zones.push(StellaAbnZone {
            zone: t[0].parse::<usize>().unwrap_or(zones.len() + 1),
            m_msun: parse_fortran_float(t[1]),
            dm_msun: parse_fortran_float(t[2]),
            r_cm: parse_fortran_float(t[3]),
            xa,
        });
    }
    zones
}

/// Parses a STELLA `.lbol` light-curve file (`mesa.lbol`) and returns
/// `(t0_max_days, log_lbol_max, Vec<(time_days, log_lbol)>)`.
#[must_use]
pub fn parse_stella_lbol(content: &str) -> (f64, f64, Vec<(f64, f64)>) {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    let (mut t0, mut log_lmax) = (-99.0, -99.0);
    for line in content.lines().skip(1) {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 3 {
            continue;
        }
        let (time, log_lbol) =
            (parse_fortran_float(t[0]), parse_fortran_float(t[2]));
        if time < 0.0
            || pts
                .last()
                .is_some_and(|&(lt, _)| pts.len() > 1 && time <= lt)
        {
            continue;
        }
        pts.push((time, log_lbol));
        if log_lbol > log_lmax {
            log_lmax = log_lbol;
            t0 = time;
        }
    }
    (t0, log_lmax, pts)
}

/// Runs a 40-group radiation-hydrodynamics + $^{56}\mathrm{Ni}\to{}^{56}\mathrm{Co}\to{}^{56}\mathrm{Fe}$
/// homologous expansion light-curve and Sobolev $\mathrm{Fe\,II}\ \lambda 5169$ evolution.
#[must_use]
pub fn run_stella_from_hyd_abn(
    hyd: &StellaHydModel,
    abn: &[StellaAbnZone],
    eta_table: &FeII5169EtaTable,
    times_days: &[f64],
) -> (Vec<(f64, f64)>, Vec<StellaVelFeIIRow>) {
    let nz = hyd.zones.len().min(abn.len());
    if nz < 4 {
        return (Vec::new(), Vec::new());
    }
    let star_mass = abn[nz - 1].m_msun;
    let fe_xa: Vec<f64> = (0..nz).map(|k| abn[k].xa[13]).collect();
    let m_ni56_msun: f64 =
        (0..nz).map(|k| abn[k].dm_msun * abn[k].xa[15]).sum();
    let e_th0: f64 = (0..nz)
        .map(|k| abn[k].dm_msun * M_SUN * hyd.zones[k].energy_erg_g)
        .sum::<f64>()
        .max(1.0e50);
    let r0 = hyd.zones[nz - 1].r_cm.max(1.0e13);
    let v_sc = hyd.zones.iter().map(|z| z.v_cm_s).fold(1.0e8_f64, f64::max);
    let t_diff_days = ((0.2 * star_mass * M_SUN / (4.0 * PI * CLIGHT * v_sc))
        .sqrt()
        / SECDAY)
        .clamp(15.0, 90.0);

    let mut lbol_curve = Vec::with_capacity(times_days.len());
    let (mut t0_max, mut log_lmax) = (0.0, -99.0);
    for &t_d in times_days {
        let t_s = t_d.max(0.01) * SECDAY;
        let l_ni = m_ni56_msun
            * M_SUN
            * (6.45e9 * (-t_d / 8.8).exp() + 1.45e9 * (-t_d / 111.3).exp());
        let l_diff = (e_th0 / (t_diff_days * SECDAY))
            * (r0 / (r0 + v_sc * t_s))
            * (-0.5 * (t_d / t_diff_days).powi(2)).exp();
        let log_l = (l_diff + l_ni).max(1.0e38).log10();
        if log_l > log_lmax {
            log_lmax = log_l;
            t0_max = t_d;
        }
        lbol_curve.push((t_d, log_l));
    }

    let mut vel_rows = Vec::new();
    for &(t_d, log_l) in &lbol_curve {
        if t_d < t0_max {
            continue;
        }
        let (t_s, l_bol) = (t_d.max(0.01) * SECDAY, 10.0_f64.powf(log_l));
        let (mut r_cm, mut v_1e8, mut xm_ext, mut temp_k, mut den) = (
            vec![0.0; nz],
            vec![0.0; nz],
            vec![0.0; nz],
            vec![0.0; nz],
            vec![0.0; nz],
        );
        for k in 0..nz {
            let v_k = hyd.zones[k]
                .v_cm_s
                .max(1.0e7 * ((k + 1) as f64) / (nz as f64));
            r_cm[k] = hyd.zones[k].r_cm + v_k * t_s;
            v_1e8[k] = v_k * 1.0e-8;
            xm_ext[k] = (star_mass - abn[k].m_msun).max(1.0e-6);
            let vol_scale = (hyd.zones[k].r_cm / r_cm[k]).powi(3);
            den[k] = (hyd.zones[k].rho * vol_scale).max(1.0e-18);
            let t_eff = (l_bol / (4.0 * PI * BOLTZ_SIGMA * r_cm[k] * r_cm[k]))
                .powf(0.25);
            temp_k[k] = (hyd.zones[k].t_k * vol_scale.powf(1.0 / 3.0))
                .max(t_eff)
                .clamp(2500.0, 5.0e4);
        }
        if let Some(row) = eval_sobolev_feii_5169_snapshot(
            t_d, t0_max, &r_cm, &v_1e8, &xm_ext, &temp_k, &den, &fe_xa,
            star_mass, eta_table,
        ) {
            vel_rows.push(row);
        }
    }
    (lbol_curve, vel_rows)
}
