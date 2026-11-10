// SPDX-License-Identifier: LGPL-3.0-or-later
//! Additional pulsation exporters: `GSM` (`pulse_gsm.f90`), `SAIO`
//! (`pulse_saio.f90`, 20 columns), `CAFEIN` (`pulse_cafein.f90`, 35 columns),
//! and `GR1D` (`pulse_gr1d.f90`, 7 columns), plus unified format dispatch
//! (`star/private/pulse.f90`, resolving Issue #4).

#![allow(missing_docs)]

use std::fmt::Write as _;

use flexmesa_const::{
    CLIGHT, CRAD, FOUR_THIRDS_PI, L_SUN, M_SUN, PI, PI4, R_SUN, STANDARD_CGRAV,
};

use crate::model::StarModel;
use crate::pulse::{
    build_pulse_points_with_options, export_fgong, export_gyre, export_osc,
    write_5e16, PulseExportOptions,
};

/// Structured representation of a `GSM` (GYRE Stellar Model, `pulse_gsm.f90`)
/// dataset (`version = 110` or `120`, center-to-surface ordering).
#[derive(Debug, Clone, PartialEq)]
pub struct GsmData {
    /// Number of radial points `n`.
    pub n: usize,
    /// Total stellar mass $M_*$ ($\mathrm{g}$).
    pub m_star: f64,
    /// Photospheric radius $R_*$ ($\mathrm{cm}$).
    pub r_star: f64,
    /// Surface luminosity $L_*$ ($\mathrm{erg\,s^{-1}}$).
    pub l_star: f64,
    /// GYRE schema version (`110` or `120`).
    pub version: usize,
    /// Named 1D column datasets (`18` for v110, `19` for v120), each of length `n`.
    pub datasets: Vec<(&'static str, Vec<f64>)>,
}

/// Builds the [`GsmData`] column structure for `model` (`pulse_gsm.f90`).
#[must_use]
pub fn export_gsm_data(
    model: &StarModel,
    add_center_point: bool,
    schema: usize,
) -> GsmData {
    let opts = PulseExportOptions {
        add_center_point,
        ..PulseExportOptions::default()
    };
    let pts = build_pulse_points_with_options(model, &opts);
    let n = pts.len();
    let m_star = model.m_grav.first().copied().unwrap_or(model.mstar);
    let r_star = if model.nz > 0 { model.r[0] } else { 0.0 };
    let l_star = if model.nz > 0 { model.lum[0] } else { 0.0 };
    let version = if schema == 110 { 110 } else { 120 };

    let mut datasets: Vec<(&'static str, Vec<f64>)> = vec![
        ("r", pts.iter().map(|p| p.r).collect()),
        ("M_r", pts.iter().map(|p| p.m).collect()),
        ("L_r", pts.iter().map(|p| p.lum).collect()),
        ("P", pts.iter().map(|p| p.p).collect()),
        ("T", pts.iter().map(|p| p.t).collect()),
        ("rho", pts.iter().map(|p| p.rho).collect()),
        ("nabla", pts.iter().map(|p| p.grad_t).collect()),
        ("N2", pts.iter().map(|p| p.n2).collect()),
        ("Gamma_1", pts.iter().map(|p| p.gamma1).collect()),
        ("nabla_ad", pts.iter().map(|p| p.grad_ad).collect()),
        ("delta", pts.iter().map(|p| p.delta).collect()),
        ("kap", pts.iter().map(|p| p.kap).collect()),
        ("kap_kap_T", pts.iter().map(|p| p.kap_t).collect()),
        ("kap_kap_rho", pts.iter().map(|p| p.kap_rho).collect()),
        ("eps", pts.iter().map(|p| p.eps_nuc).collect()),
        ("eps_eps_T", pts.iter().map(|p| p.eps_t).collect()),
        ("eps_eps_rho", pts.iter().map(|p| p.eps_rho).collect()),
    ];
    if version == 120 {
        datasets.push(("eps_grav", pts.iter().map(|p| p.eps_grav).collect()));
    }
    datasets.push(("Omega_rot", pts.iter().map(|p| p.omega).collect()));

    GsmData {
        n,
        m_star,
        r_star,
        l_star,
        version,
        datasets,
    }
}

/// Serializes `model` in `GSM` (GYRE Stellar Model, `pulse_gsm.f90`) format.
#[must_use]
pub fn export_gsm(model: &StarModel, add_center_point: bool) -> String {
    let gsm = export_gsm_data(model, add_center_point, 120);
    let mut out = String::with_capacity(256 + gsm.n * 480);
    let _ = writeln!(
        out,
        "# GSM v{} n={} M_star={:.16E} R_star={:.16E} L_star={:.16E}",
        gsm.version, gsm.n, gsm.m_star, gsm.r_star, gsm.l_star
    );
    let _ = write!(out, "# {:>4}", "idx");
    for (name, _) in &gsm.datasets {
        let _ = write!(out, " {name:>26}");
    }
    out.push('\n');
    for i in 0..gsm.n {
        let _ = write!(out, "{:6}", i + 1);
        for (_, col) in &gsm.datasets {
            let _ = write!(out, " {:26.16E}", col[i]);
        }
        out.push('\n');
    }
    out
}

/// Exports `model` in `SAIO` (Tohoku non-adiabatic pulsation) 20-column ASCII
/// format (`star/private/pulse_saio.f90`, center-to-surface ordering, 4 global
/// constants, 20 columns across `5(1PE16.9)` rows).
///
/// # Upstream MESA Bug Fixed
/// In upstream `pulse_saio.f90` (`store_saio_data_env`), `kap` is never
/// initialized before being used to compute `kap_rho`, `kap_T`, and `Lrad`.
/// Here `pt.kap` is properly evaluated from cell-to-face opacity interpolation.
#[must_use]
pub fn export_saio(
    model: &StarModel,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> String {
    let opts = PulseExportOptions {
        add_center_point: false,
        keep_surface_point,
        add_atmosphere,
        ..PulseExportOptions::default()
    };
    let pts = build_pulse_points_with_options(model, &opts);
    let m_outer = model
        .m_grav
        .first()
        .copied()
        .unwrap_or(model.mstar)
        .max(1.0e-30);
    let r_outer = if model.nz > 0 {
        model.r[0].max(1.0e-30)
    } else {
        R_SUN
    };
    let l_outer = if model.nz > 0 {
        model.lum[0].max(1.0e-30)
    } else {
        L_SUN
    };

    let mut out = String::with_capacity(256 + pts.len() * 340);
    let _ = writeln!(
        out,
        "{:6}{:16.9E}{:16.9E}{:16.9E}{:16.9E}",
        pts.len(),
        m_outer,
        (l_outer / L_SUN).log10(),
        (r_outer / R_SUN).log10(),
        model.star_age
    );
    let c_rad_factor = 16.0 * PI * CLIGHT * CRAD / 3.0;
    let g_cgs = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
    for pt in &pts {
        let kap_safe = pt.kap.max(1.0e-30);
        let l_rad = if pt.tau >= 2.0 / 3.0 {
            c_rad_factor * g_cgs * pt.m * pt.t.powi(4) * pt.grad_t
                / (pt.p.max(1.0e-30) * kap_safe)
        } else {
            pt.lum
        };
        let var = [
            pt.r,
            pt.m,
            l_rad,
            pt.t,
            pt.rho,
            pt.p,
            pt.eps_nuc + pt.eps_grav,
            pt.kap,
            pt.cv,
            pt.chi_rho,
            pt.chi_t,
            pt.eps_rho,
            pt.eps_t,
            pt.kap_rho / kap_safe,
            pt.kap_t / kap_safe,
            pt.grad_t,
            pt.grad_ad,
            pt.x_h1,
            pt.x_he4,
            pt.lum,
        ];
        write_5e16(&mut out, &var);
    }
    out
}

const CAFEIN_GLOBAL_HEADERS: [&str; 13] = [
    "..................M(Msun)",
    "..................R(Rsun)",
    "..................L(Lsun)",
    "......................P_c",
    "......................T_c",
    "....................rho_c",
    ".....................Teff",
    "..................s_units",
    "..................g_units",
    ".................cm_units",
    "................erg_units",
    "..................K_units",
    "...................unused",
];

const CAFEIN_COL_HEADERS: [&str; 35] = [
    "........................m",
    "...................logrho",
    ".....................logP",
    ".....................logT",
    "........................r",
    "......................V_g",
    ".......................As",
    "........................U",
    "......................c_1",
    ".......................N2",
    "...................L2_ll1",
    "........................g",
    "........................l",
    "........................V",
    ".................nabla_ad",
    "....................nabla",
    "......................C_P",
    "......................v_t",
    "....................kap_S",
    "...................eps_ad",
    "....................eps_S",
    "..................c_2_fit",
    "......................c_3",
    "......................c_4",
    "...............dlnLr_dlnr",
    "..................P_scale",
    "...........surf_r_rad_beg",
    "...........surf_r_rad_end",
    "..................Gamma_1",
    "..................entropy",
    "......................kap",
    "..................chi_rho",
    "....................chi_T",
    ".................U_U_surf",
    ".......................c2",
];

/// Exports `model` in `CAFEIN` 35-column dimensionless ASCII format
/// (`star/private/pulse_cafein.f90`, center-to-surface ordering).
#[must_use]
pub fn export_cafein(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> String {
    let opts = PulseExportOptions {
        add_center_point,
        keep_surface_point,
        add_atmosphere,
        add_double_points: false,
        ..PulseExportOptions::default()
    };
    let pts = build_pulse_points_with_options(model, &opts);
    let m_star = model
        .m_grav
        .first()
        .copied()
        .unwrap_or(model.mstar)
        .max(1.0e-30);
    let r_star = if model.nz > 0 {
        model.r[0].max(1.0e-30)
    } else {
        R_SUN
    };
    let l_star = if model.nz > 0 { model.lum[0] } else { L_SUN };
    let g1 = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
    let r_gas = 8.314_462_618e7_f64;

    let s_units = (r_star.powi(3) / (g1 * m_star)).sqrt();
    let g_units = m_star;
    let cm_units = r_star;
    let erg_units = g1 * m_star * m_star / r_star;
    let k_units = erg_units / (m_star * r_gas);
    let rho_units = g_units / cm_units.powi(3);
    let p_units = erg_units / cm_units.powi(3);
    let cp_units = erg_units / (k_units * g_units);

    let p_c = pts.first().map_or(0.0, |p| p.p);
    let t_c = pts.first().map_or(0.0, |p| p.t);
    let rho_c = pts.first().map_or(1.0, |p| p.rho.max(1.0e-99));
    let teff = if model.teff > 0.0 {
        model.teff
    } else {
        pts.last().map_or(5777.0, |p| p.t)
    };

    let surf_r_rad_end = pts
        .iter()
        .find(|p| p.n2 < 0.0 && p.r > 0.0)
        .map_or(1.0, |p| p.r / r_star);
    let u_surf = pts.last().map_or(1.0e-30, |p| p.u_dziem.max(1.0e-30));

    let mut out = String::with_capacity(1024 + pts.len() * 900);
    for h in CAFEIN_GLOBAL_HEADERS {
        let _ = write!(out, " {h:>24}");
    }
    out.push('\n');
    let glob = [
        m_star / M_SUN,
        r_star / R_SUN,
        l_star,
        p_c,
        t_c,
        rho_c,
        teff,
        s_units,
        g_units,
        cm_units,
        erg_units,
        k_units,
        0.0,
    ];
    for v in glob {
        let _ = write!(out, " {v:24.14E}");
    }
    out.push_str("\n\n");
    for h in CAFEIN_COL_HEADERS {
        let _ = write!(out, " {h:>24}");
    }
    out.push('\n');

    let c_rad_factor = 16.0 * PI * CLIGHT * CRAD / 3.0;
    for pt in &pts {
        let is_center = pt.r <= 0.0;
        let kap_safe = pt.kap.max(1.0e-30);
        let kap_t_ln = pt.kap_t / kap_safe;
        let kap_rho_ln = pt.kap_rho / kap_safe;
        let kap_s = kap_t_ln - pt.delta * kap_rho_ln;
        let kap_ad = pt.grad_ad * kap_t_ln + kap_rho_ln / pt.gamma1;
        let (eps_t_ln, eps_rho_ln) = if pt.eps_nuc.abs() > 1.0e-99 {
            (pt.eps_t / pt.eps_nuc, pt.eps_rho / pt.eps_nuc)
        } else {
            (0.0, 0.0)
        };
        let eps_ad = pt.grad_ad * eps_t_ln + eps_rho_ln / pt.gamma1;
        let eps_s = eps_t_ln - pt.delta * eps_rho_ln;
        let g_local = if is_center {
            0.0
        } else {
            g1 * pt.m / (pt.r * pt.r)
        };
        let l2_ll1 = if is_center {
            1.0e99
        } else {
            (pt.gamma1 * pt.p / (pt.rho * pt.r * pt.r)) * s_units * s_units
        };
        let l_rad = if is_center {
            1.0
        } else {
            (c_rad_factor * g1 * pt.m * pt.t.powi(4) * pt.grad_t
                / (pt.p * kap_safe))
                .max(1.0e-30)
        };
        let c_3 = if is_center {
            9.0 * pt.eps_nuc * kap_safe * pt.p
                / (16.0
                    * PI
                    * CRAD
                    * CLIGHT
                    * pt.t.powi(4)
                    * pt.grad_t.max(1.0e-10)
                    * g1)
        } else {
            PI4 * pt.r.powi(3) * pt.rho * pt.eps_nuc / l_rad
        };
        let c_4 = if is_center {
            (9.0 * pt.t * pt.cp * kap_safe * pt.p
                / (16.0
                    * PI
                    * CRAD
                    * CLIGHT
                    * pt.t.powi(4)
                    * pt.grad_t.max(1.0e-10)
                    * g1))
                / s_units
        } else {
            (PI4 * pt.r.powi(3) * pt.rho * pt.t * pt.cp / l_rad) / s_units
        };
        let c_2 = (kap_ad - 4.0 * pt.grad_ad) * pt.v_dziem * pt.grad_t
            + pt.grad_ad * pt.v_dziem;
        let c_1_val = if is_center {
            m_star / (FOUR_THIRDS_PI * r_star.powi(3) * rho_c)
        } else {
            pt.c_1
        };

        let row = [
            pt.m / g_units,
            (pt.rho / rho_units).max(1.0e-99).log10(),
            (pt.p / p_units).max(1.0e-99).log10(),
            (pt.t / k_units).max(1.0e-99).log10(),
            pt.r / cm_units,
            pt.v_g,
            pt.a_star,
            pt.u_dziem,
            c_1_val,
            pt.n2 * s_units * s_units,
            l2_ll1,
            g_local / (cm_units / (s_units * s_units)),
            pt.lum / (erg_units / s_units),
            pt.v_dziem,
            pt.grad_ad,
            pt.grad_t,
            pt.cp / cp_units,
            pt.delta,
            kap_s,
            eps_ad,
            eps_s,
            c_2,
            c_3,
            c_4,
            3.0,
            pt.scale_height / cm_units,
            1.0,
            surf_r_rad_end,
            pt.gamma1,
            pt.entropy / cp_units,
            pt.kap / (cm_units * cm_units / g_units),
            pt.chi_rho,
            pt.chi_t,
            pt.u_dziem / u_surf,
            c_2,
        ];
        for v in row {
            let _ = write!(out, " {v:24.14E}");
        }
        out.push('\n');
    }
    out
}

/// Exports `model` in `GR1D` 7-column cell-centered core-collapse format
/// (`star/private/pulse_gr1d.f90`, center-to-surface ordering: `m, r, T, rho,
/// v, ye, omega`).
///
/// # Upstream MESA Bugs Fixed
/// Upstream `pulse_gr1d.f90` and `pulse.f90` contain three bugs:
/// 1. `j = j + 1` is missing in `get_gr1d_data` (`pulse_gr1d.f90:68-78`),
///    overwriting row 1 for every zone.
/// 2. `case ('gr1d')` in `pulse.f90:174` dispatches to `write_saio_data`.
/// 3. `write_gr1d_data` uses header format label `100` (`I6`) instead of row
///    format label `110` (`I6, 1P, 99E20.10`).
///
/// This implementation increments the row counter and writes all 7 columns
/// using the intended `(I6, 1P, 7E20.10)` format.
#[must_use]
pub fn export_gr1d(model: &StarModel) -> String {
    let nz = model.nz;
    let mut out = String::with_capacity(64 + nz * 160);
    let _ = writeln!(out, "{nz:6}");
    for (idx, k) in (0..nz).rev().enumerate() {
        let m_mid = (model.m[k] - 0.5 * model.dm[k]).max(0.0);
        let r_in = if k + 1 < nz {
            model.r[k + 1]
        } else {
            model.r_center
        };
        let r_mid = 0.5 * (model.r[k] + r_in);
        let v_in = if k + 1 < nz {
            model.v[k + 1]
        } else {
            model.v_center
        };
        let v_mid = 0.5 * (model.v[k] + v_in);
        let omega_mid = if k + 1 < nz {
            0.5 * (model.omega[k] + model.omega[k + 1])
        } else {
            model.omega[k]
        };
        let ye = model.ye.get(k).copied().unwrap_or(0.5);
        let _ = writeln!(
            out,
            "{:6}{:20.10E}{:20.10E}{:20.10E}{:20.10E}{:20.10E}{:20.10E}{:20.10E}",
            idx + 1,
            m_mid,
            r_mid,
            model.t[k],
            model.rho[k],
            v_mid,
            ye,
            omega_mid
        );
    }
    out
}

/// Dispatches pulsation data export by format name (`"gyre"`, `"fgong"`,
/// `"osc"`, `"gsm"`, `"saio"`, `"cafein"`, `"gr1d"`, case-insensitive;
/// `star/private/pulse.f90`).
#[must_use]
pub fn export_pulse_by_format(
    model: &StarModel,
    format: &str,
    add_center_point: bool,
) -> Option<String> {
    match format.trim().to_ascii_lowercase().as_str() {
        "gyre" => Some(export_gyre(model, add_center_point)),
        "fgong" => Some(export_fgong(model, add_center_point)),
        "osc" => Some(export_osc(model, add_center_point)),
        "gsm" => Some(export_gsm(model, add_center_point)),
        "saio" => Some(export_saio(model, true, false)),
        "cafein" => Some(export_cafein(model, add_center_point, true, false)),
        "gr1d" => Some(export_gr1d(model)),
        _ => None,
    }
}
