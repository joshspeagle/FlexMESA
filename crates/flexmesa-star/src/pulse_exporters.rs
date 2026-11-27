// SPDX-License-Identifier: GPL-3.0-or-later
//! Additional pulsation exporters: `GSM` (`pulse_gsm.f90`), `SAIO`
//! (`pulse_saio.f90`, 20 columns), `CAFEIN` (`pulse_cafein.f90`, 35 columns),
//! and `GR1D` (`pulse_gr1d.f90`, 7 columns), plus unified format dispatch
//! (`star/private/pulse.f90`, resolving Issue #4).

#![allow(missing_docs)]

use std::fmt::Write as _;

use flexmesa_const::{CGAS, CLIGHT, CRAD, L_SUN, M_SUN, PI, PI4, R_SUN};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::pulse::{
    build_pulse_atmosphere_points, export_fgong, export_gyre, export_osc,
};
use crate::pulse_gyre_data::{
    eval_center, eval_center_rho, eval_face, eval_face_a_ast, eval_face_rho,
    get_gyre_data, set_segment_indices, GyreDataError, GyreDataInput,
};
use crate::pulse_write::{fortran_1pe, PulseWriteError};
use crate::report_utils::eval_photosphere_info;

#[inline]
fn at1(slice: &[f64], k: usize) -> f64 {
    slice[k - 1]
}

#[allow(clippy::needless_pass_by_value)]
fn gyre_err(e: GyreDataError) -> StarError {
    StarError::InvalidInput(e.to_string())
}

#[allow(clippy::needless_pass_by_value)]
fn pulse_write_err(e: PulseWriteError) -> StarError {
    StarError::InvalidInput(e.to_string())
}

/// Evaluates Fortran `Ew.dEe` edit descriptor (leading `0.` without `1P`
/// scale factor), matching `gfortran` formatting of `(1X,E24.14E3)`.
#[must_use]
pub fn fortran_e_no_1p(v: f64, w: usize, d: usize, ed: usize) -> String {
    let body = if v.is_nan() {
        String::from("NaN")
    } else if v.is_infinite() {
        let s = if v.is_sign_positive() {
            "Infinity"
        } else {
            "-Infinity"
        };
        if s.len() <= w {
            String::from(s)
        } else if v.is_sign_positive() {
            String::from("Inf")
        } else {
            String::from("-Inf")
        }
    } else if v == 0.0 {
        let zeros = "0".repeat(d);
        let ez = "0".repeat(ed);
        let sign = if v.is_sign_negative() { "-" } else { "" };
        format!("{sign}0.{zeros}E+{ez}")
    } else {
        let prec = d.saturating_sub(1);
        let s = format!("{:.prec$e}", v.abs());
        let (mant, exp_str) = s.split_once('e').unwrap_or((s.as_str(), "0"));
        let digits: String = mant.chars().filter(|&c| c != '.').collect();
        let exp: i64 = exp_str.parse().unwrap_or(0);
        let f_exp = exp + 1;
        let exp_mag = f_exp.unsigned_abs();
        let exp_digits = exp_mag.to_string();
        if exp_digits.len() > ed {
            return "*".repeat(w);
        }
        let exp_sign = if f_exp < 0 { '-' } else { '+' };
        let sign = if v < 0.0 { "-" } else { "" };
        format!("{sign}0.{digits}E{exp_sign}{exp_mag:0ed$}")
    };
    if body.len() > w {
        "*".repeat(w)
    } else {
        format!("{body:>w$}")
    }
}

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
    /// Named 1D column datasets (`18` for v110, `19` for v120), each length `n`.
    pub datasets: Vec<(&'static str, Vec<f64>)>,
}

fn eval_eps_grav_column(
    model: &StarModel,
    gyre_in: &GyreDataInput<'_>,
    add_center_point: bool,
    keep_surface_point: bool,
) -> Result<Vec<f64>, StarError> {
    let nz = model.nz;
    let eps_grav: Vec<f64> =
        model.eps_grav_ad.iter().take(nz).map(|a| a.val).collect();
    if eps_grav.len() < nz {
        return Err(StarError::InvalidInput(String::from(
            "eps_grav_ad shorter than nz for GSM schema 120",
        )));
    }
    let (k_a, k_b) =
        set_segment_indices(gyre_in, add_center_point).map_err(gyre_err)?;
    let n_sg = k_a.len();
    let mut col = Vec::with_capacity(nz + n_sg);
    let mut sg = 0_usize;
    for k in 1..=nz {
        if k == 1 && !keep_surface_point {
            continue;
        }
        let val = eval_face(gyre_in.dq, &eps_grav, k, k_a[sg], k_b[sg])
            .map_err(gyre_err)?;
        col.push(val);
        if k == k_b[sg] + 1 {
            sg += 1;
            let val2 = eval_face(gyre_in.dq, &eps_grav, k, k_a[sg], k_b[sg])
                .map_err(gyre_err)?;
            col.push(val2);
        }
    }
    if add_center_point {
        let val_c =
            eval_center(gyre_in.rmid, &eps_grav, k_a[n_sg - 1], k_b[n_sg - 1])
                .map_err(gyre_err)?;
        col.push(val_c);
    }
    col.reverse();
    Ok(col)
}

/// Builds the [`GsmData`] column structure for `model` matching
/// `pulse.f90:128-129` (`get_gyre_data`) and `pulse_gsm.f90:51-118`
/// (`schema` `110` or `120`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for schemas other than `110`/`120`,
/// missing Brunt-Vaisala frequencies, short model arrays, or unported
/// `add_atmosphere` / capped double points.
pub fn try_export_gsm_data(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
    schema: i32,
) -> Result<GsmData, StarError> {
    if schema != 110 && schema != 120 {
        return Err(StarError::InvalidInput(format!(
            "invalid gyre_data_schema {schema} for GSM (expected 110 or 120)"
        )));
    }
    let mut gyre_in = GyreDataInput::from_model(model).map_err(gyre_err)?;
    gyre_in.gyre_data_schema = 110;
    let (global_data, point_data) = get_gyre_data(
        &gyre_in,
        add_center_point,
        keep_surface_point,
        add_atmosphere,
    )
    .map_err(gyre_err)?;
    let n = point_data.len();
    let version = if schema == 110 { 110_usize } else { 120_usize };
    let take_col = |idx: usize| -> Vec<f64> {
        point_data.iter().map(|r| r[idx]).collect()
    };

    let mut datasets: Vec<(&'static str, Vec<f64>)> = vec![
        ("r", take_col(0)),
        ("M_r", take_col(1)),
        ("L_r", take_col(2)),
        ("P", take_col(3)),
        ("T", take_col(4)),
        ("rho", take_col(5)),
        ("nabla", take_col(6)),
        ("N2", take_col(7)),
        ("Gamma_1", take_col(8)),
        ("nabla_ad", take_col(9)),
        ("delta", take_col(10)),
        ("kap", take_col(11)),
        ("kap_kap_T", take_col(12)),
        ("kap_kap_rho", take_col(13)),
        ("eps", take_col(14)),
        ("eps_eps_T", take_col(15)),
        ("eps_eps_rho", take_col(16)),
    ];
    if version == 120 {
        let eps_grav_col = eval_eps_grav_column(
            model,
            &gyre_in,
            add_center_point,
            keep_surface_point,
        )?;
        datasets.push(("eps_grav", eps_grav_col));
    }
    datasets.push(("Omega_rot", take_col(17)));

    Ok(GsmData {
        n,
        m_star: global_data[0],
        r_star: global_data[1],
        l_star: global_data[2],
        version,
        datasets,
    })
}

/// Builds the [`GsmData`] column structure for `model` (`pulse_gsm.f90`)
/// using `keep_surface_point = true`.
#[must_use]
pub fn export_gsm_data(
    model: &StarModel,
    add_center_point: bool,
    schema: usize,
) -> GsmData {
    let schema_i32 = if schema == 110 { 110 } else { 120 };
    try_export_gsm_data(model, add_center_point, true, false, schema_i32)
        .unwrap_or_else(|_| GsmData {
            n: 0,
            m_star: 0.0,
            r_star: 0.0,
            l_star: 0.0,
            version: if schema == 110 { 110 } else { 120 },
            datasets: Vec::new(),
        })
}

/// Serializes `model` in `GSM` (GYRE Stellar Model, `pulse_gsm.f90`) text form.
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

fn eval_cell_xy(model: &StarModel) -> Result<(Vec<f64>, Vec<f64>), StarError> {
    let nz = model.nz;
    let species = model.species;
    let catalog = flexmesa_chem::NuclideCatalog::builtin_ref();
    let mut x_vec = Vec::with_capacity(nz);
    let mut y_vec = Vec::with_capacity(nz);
    for k in 0..nz {
        let xa_k = &model.xa[k * species..(k + 1) * species];
        let c = flexmesa_chem::get_composition_info(
            catalog,
            &model.chem_id,
            xa_k,
            true,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "pulse composition error at zone {k}: {e}"
            ))
        })?;
        x_vec.push(c.xh);
        y_vec.push(c.xhe);
    }
    Ok((x_vec, y_vec))
}

/// Upstream `get_saio_data` (`star/private/pulse_saio.f90:37-295`).
///
/// In MESA `26.4.1`, `store_saio_data_env` (`pulse_saio.f90:246, 275-284`)
/// divides by `kap` (`point_data(8, j)`) without initializing it first,
/// so executing any envelope row reads undefined memory (`DECISIONS #43`, `#169`).
///
/// # Errors
/// Returns [`PulseWriteError::Upstream`] whenever `store_saio_data_env` would
/// execute (`nz > 0 && (keep_surface_point || nz > 1)`), or
/// [`PulseWriteError::Unsupported`] for `add_atmosphere = true`.
pub fn get_saio_data_strict(
    model: &StarModel,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<(Vec<f64>, Vec<[f64; 20]>), PulseWriteError> {
    if add_atmosphere {
        return Err(PulseWriteError::Unsupported(String::from(
            "get_saio_data: add_atmosphere requires build_atm \
             (star/private/pulse_saio.f90:78-85)",
        )));
    }
    let nz = model.nz;
    if nz > 0 && (keep_surface_point || nz > 1) {
        return Err(PulseWriteError::Upstream(String::from(
            "get_saio_data: store_saio_data_env reads uninitialized kap \
             before computing kap_rho, kap_T, and Lrad \
             (star/private/pulse_saio.f90:246, 275-284; DECISIONS #43; #169)",
        )));
    }
    let phot = eval_photosphere_info(model);
    let r_outer = R_SUN * phot.r_rsun;
    let m_outer = model.m_grav.first().copied().unwrap_or(model.mstar);
    let l1 = model.lum.first().copied().unwrap_or(0.0);
    Ok((
        vec![
            m_outer,
            flexmesa_math::log10(l1 / L_SUN),
            flexmesa_math::log10(r_outer / R_SUN),
            model.star_age,
        ],
        Vec::new(),
    ))
}

/// Formats `SAIO` data (`write_saio_data`, `star/private/pulse_saio.f90:298-345`).
#[must_use]
pub fn write_saio_data(
    global_data: &[f64],
    point_data: &[[f64; 20]],
) -> String {
    let nn = point_data.len();
    let mut out = String::with_capacity(256 + nn * 340);
    let _ = write!(out, "{nn:6}");
    for &v in global_data {
        out.push_str(&fortran_1pe(v, 16, 9, None));
    }
    out.push('\n');
    for row in point_data {
        for chunk in row.chunks(5) {
            for &v in chunk {
                out.push_str(&fortran_1pe(v, 16, 9, None));
            }
            out.push('\n');
        }
    }
    out
}

/// Strict MESA `26.4.1` `SAIO` export (`get_saio_data` followed by
/// `write_saio_data`, `star/private/pulse_saio.f90`).
///
/// # Errors
/// Propagates [`get_saio_data_strict`] errors (`DECISIONS #43`, `#169`).
pub fn export_saio_strict(
    model: &StarModel,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<String, PulseWriteError> {
    let (g, p) =
        get_saio_data_strict(model, keep_surface_point, add_atmosphere)?;
    Ok(write_saio_data(&g, &p))
}

/// Legacy compatibility `SAIO` exporter (`star/private/pulse_saio.f90`).
///
/// Retains fallback `kap` evaluation and `-> String` signature for
/// out-of-allowlist Python and integration test callers while
/// [`get_saio_data_strict`] enforces the `DECISIONS #43` error on upstream's
/// uninitialized `kap` read (`pulse_saio.f90:246, 275-284`).
#[must_use]
pub fn export_saio(
    model: &StarModel,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> String {
    let nz = model.nz;
    if nz == 0 {
        return write_saio_data(&[0.0; 4], &[]);
    }
    let phot = eval_photosphere_info(model);
    let r_outer = R_SUN * phot.r_rsun;
    let m_outer = model.m_grav[0];
    let g_data = vec![
        m_outer,
        flexmesa_math::log10(model.lum[0] / L_SUN),
        flexmesa_math::log10(r_outer / R_SUN),
        model.star_age,
    ];

    let Ok(gyre_in) = GyreDataInput::from_model(model) else {
        return write_saio_data(&g_data, &[]);
    };
    let Ok((k_a, k_b)) = set_segment_indices(&gyre_in, false) else {
        return write_saio_data(&g_data, &[]);
    };
    let Ok((x_vec, y_vec)) = eval_cell_xy(model) else {
        return write_saio_data(&g_data, &[]);
    };
    let eps_grav: Vec<f64> = model.eps_grav_ad.iter().map(|a| a.val).collect();
    let c_rad_factor = 16.0 * PI * CLIGHT * CRAD;
    let two_thirds = 2.0 / 3.0;

    let mut rows: Vec<[f64; 20]> = Vec::with_capacity(nz + k_a.len() + 8);
    if add_atmosphere {
        let atm = build_pulse_atmosphere_points(model, r_outer, m_outer, 8);
        for pt in atm.into_iter().rev() {
            let kap_s = pt.kap;
            let l_rad = if pt.tau >= two_thirds {
                (c_rad_factor
                    * model.cgrav[0]
                    * pt.m
                    * pt.t
                    * pt.t
                    * pt.t
                    * pt.t
                    * pt.grad_t)
                    / (3.0 * pt.p * kap_s)
            } else {
                pt.lum
            };
            rows.push([
                pt.r,
                pt.m,
                l_rad,
                pt.t,
                pt.rho,
                pt.p,
                0.0,
                pt.kap,
                pt.cv,
                pt.chi_rho,
                pt.chi_t,
                0.0,
                0.0,
                pt.kap_rho / kap_s,
                pt.kap_t / kap_s,
                pt.grad_t,
                pt.grad_ad,
                pt.x_h1,
                pt.x_he4,
                pt.lum,
            ]);
        }
    }

    let mut store_env = |k: usize, ka: usize, kb: usize| {
        let r = at1(&model.r, k);
        let m = at1(&model.m_grav, k);
        let t = eval_face(&model.dq, &model.t, k, 1, nz).unwrap_or(0.0);
        let rho = if model.controls.interpolate_rho_for_pulse_data {
            eval_face(&model.dq, &model.rho, k, ka, kb).unwrap_or(0.0)
        } else {
            eval_face_rho(&gyre_in, k, ka, kb).unwrap_or(0.0)
        };
        let p = eval_face(&model.dq, &model.peos, k, 1, nz).unwrap_or(0.0);
        let eps = eval_face(&model.dq, &model.eps_nuc, k, ka, kb)
            .unwrap_or(0.0)
            + eval_face(&model.dq, &eps_grav, k, ka, kb).unwrap_or(0.0);
        let kap =
            eval_face(&model.dq, &model.opacity, k, ka, kb).unwrap_or(0.0);
        let c_v = eval_face(&model.dq, &model.cv, k, ka, kb).unwrap_or(0.0);
        let chi_rho =
            eval_face(&model.dq, &model.chi_rho, k, ka, kb).unwrap_or(0.0);
        let chi_t =
            eval_face(&model.dq, &model.chi_t, k, ka, kb).unwrap_or(0.0);
        let eps_rho = eval_face(&model.dq, &model.d_epsnuc_dlnd, k, ka, kb)
            .unwrap_or(0.0);
        let eps_t = eval_face(&model.dq, &model.d_epsnuc_dlnt, k, ka, kb)
            .unwrap_or(0.0);
        let kap_rho = eval_face(&model.dq, &model.d_opacity_dlnd, k, ka, kb)
            .unwrap_or(0.0)
            / kap;
        let kap_t = eval_face(&model.dq, &model.d_opacity_dlnt, k, ka, kb)
            .unwrap_or(0.0)
            / kap;
        let nabla = at1(&model.gradt, k);
        let nabla_ad =
            eval_face(&model.dq, &model.grada, k, ka, kb).unwrap_or(0.0);
        let x = eval_face(&model.dq, &x_vec, k, ka, kb)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let y = eval_face(&model.dq, &y_vec, k, ka, kb)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let l = at1(&model.lum, k);
        let l_rad = if at1(&model.tau, k) >= two_thirds {
            (c_rad_factor * at1(&model.cgrav, k) * m * t * t * t * t * nabla)
                / (3.0 * p * kap)
        } else {
            l
        };
        rows.push([
            r, m, l_rad, t, rho, p, eps, kap, c_v, chi_rho, chi_t, eps_rho,
            eps_t, kap_rho, kap_t, nabla, nabla_ad, x, y, l,
        ]);
    };

    let mut sg = 0_usize;
    for k in 1..=nz {
        if k == 1 && !keep_surface_point {
            continue;
        }
        store_env(k, k_a[sg], k_b[sg]);
        if k == k_b[sg] + 1 {
            sg += 1;
            store_env(k, k_a[sg], k_b[sg]);
        }
    }
    rows.reverse();
    write_saio_data(&g_data, &rows)
}

const CAFEIN_GLOBAL_HEADERS: [&str; 13] = [
    "..................M(Msun)",
    "..................R(Rsun)",
    "..................L(Lsun)",
    "..............Pc(erg/cm3)",
    "....................Tc(K)",
    "..............rhoC(g/cm3)",
    "..................Teff(K)",
    "................sec_units",
    "..............grams_units",
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
    ".......................Vg",
    "....................Astar",
    "........................U",
    ".......................c1",
    ".......................N2",
    "..............L2/(l(l+1))",
    "........................g",
    ".......................Lr",
    "........................V",
    "............del_ad(noDim)",
    "...............del(noDim)",
    ".......................Cp",
    "......................v_t",
    "...........kappa_s(noDim)",
    "............eps_ad(noDim)",
    ".............eps_s(noDim)",
    "................c2_fitted",
    ".......................c3",
    ".......................c4",
    "...............dlnLr/dlnr",
    "..................P_scale",
    "........StartRadSurfLayer",
    "..........EndRadSurfLayer",
    "...................Gamma1",
    "..................entropy",
    "....................kappa",
    "...................chiRho",
    ".....................chiT",
    "..................U/Usurf",
    ".......................c2",
];

fn log_deriv(
    x: &[f64],
    y: &[f64],
    dy_a: Option<f64>,
    dy_b: Option<f64>,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![dy_a.or(dy_b).unwrap_or(0.0)];
    }
    let mut dy = vec![0.0_f64; n];
    dy[0] = match dy_a {
        Some(v) => v,
        None => x[0] / y[0] * (y[1] - y[0]) / (x[1] - x[0]),
    };
    for j in 1..(n - 1) {
        dy[j] = x[j] / y[j] * (y[j + 1] - y[j - 1]) / (x[j + 1] - x[j - 1]);
    }
    dy[n - 1] = match dy_b {
        Some(v) => v,
        None => {
            x[n - 1] / y[n - 1] * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
        }
    };
    dy
}

/// Evaluates `get_cafein_data` (`star/private/pulse_cafein.f90:38-654`)
/// returning `(global_data, point_data)` in dimensionless center-to-surface
/// order.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `nz == 0`, arrays are short,
/// `calculate_brunt_n2 = false`, or `add_atmosphere = true`.
pub fn get_cafein_data(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<([f64; 13], Vec<[f64; 35]>), StarError> {
    if add_atmosphere {
        return Err(StarError::InvalidInput(String::from(
            "get_cafein_data: add_atmosphere requires build_atm \
             (star/private/pulse_cafein.f90:80-88)",
        )));
    }
    let nz = model.nz;
    if nz == 0 || model.ln_s.len() < nz || model.scale_height.len() < nz {
        return Err(StarError::InvalidInput(String::from(
            "get_cafein_data: model arrays shorter than nz",
        )));
    }
    let s = GyreDataInput::from_model(model).map_err(gyre_err)?;
    let dq = s.dq;
    let m_star = at1(s.m_grav, 1);
    let r_star = R_SUN * s.photosphere_r;
    let p_c = eval_center(s.rmid, s.peos, 1, nz).map_err(gyre_err)?;
    let t_c = eval_center(s.rmid, s.t, 1, nz).map_err(gyre_err)?;
    let rho_c = eval_center(s.rmid, s.rho, 1, nz).map_err(gyre_err)?;
    let cgrav1 = at1(s.cgrav, 1);
    let s_units = (flexmesa_math::pow3(r_star) / (cgrav1 * m_star)).sqrt();
    let g_units = m_star;
    let cm_units = r_star;
    let erg_units = cgrav1 * flexmesa_math::pow2(m_star) / r_star;
    let k_units = cgrav1 * flexmesa_math::pow2(m_star) / (r_star * CGAS);
    let global_data = [
        m_star / M_SUN,
        r_star / R_SUN,
        at1(s.l, 1),
        p_c,
        t_c,
        rho_c,
        model.teff,
        s_units,
        g_units,
        cm_units,
        erg_units,
        k_units,
        0.0,
    ];

    let mut pd: Vec<[f64; 35]> = Vec::with_capacity(nz + 1);
    let mut l_rad: Vec<f64> = Vec::with_capacity(nz + 1);
    let cp_slice = &model.cp[..nz];
    let lns_slice = &model.ln_s[..nz];
    let sh_slice = &model.scale_height[..nz];

    for k in 1..=nz {
        if k == 1 && !keep_surface_point {
            continue;
        }
        let m = at1(s.m_grav, k);
        let r = at1(s.r, k);
        let l = at1(s.l, k);
        let rho = if s.interpolate_rho_for_pulse_data {
            eval_face(dq, s.rho, k, 1, nz).map_err(gyre_err)?
        } else {
            eval_face_rho(&s, k, 1, nz).map_err(gyre_err)?
        };
        let p = eval_face(dq, s.peos, k, 1, nz).map_err(gyre_err)?;
        let t = eval_face(dq, s.t, k, 1, nz).map_err(gyre_err)?;
        let logrho = flexmesa_math::log10(rho);
        let logp = flexmesa_math::log10(p);
        let logt = flexmesa_math::log10(t);
        let nabla_ad = eval_face(dq, s.grada, k, 1, nz).map_err(gyre_err)?;
        let nabla = at1(s.grad_t, k);
        let c_p = eval_face(dq, cp_slice, k, 1, nz).map_err(gyre_err)?;
        let chi_t = eval_face(dq, s.chi_t, k, 1, nz).map_err(gyre_err)?;
        let chi_rho = eval_face(dq, s.chi_rho, k, 1, nz).map_err(gyre_err)?;
        let delta = chi_t / chi_rho;
        let gamma1 = eval_face(dq, s.gamma1, k, 1, nz).map_err(gyre_err)?;
        let entropy = flexmesa_math::exp(
            eval_face(dq, lns_slice, k, 1, nz).map_err(gyre_err)?,
        );
        let kap = eval_face(dq, s.opacity, k, 1, nz).map_err(gyre_err)?;
        let kap_rho =
            eval_face(dq, s.d_opacity_dlnd, k, 1, nz).map_err(gyre_err)? / kap;
        let kap_t =
            eval_face(dq, s.d_opacity_dlnt, k, 1, nz).map_err(gyre_err)? / kap;
        let kap_ad = nabla_ad * kap_t + kap_rho / gamma1;
        let kap_s = kap_t - delta * kap_rho;
        let eps = eval_face(dq, s.eps_nuc, k, 1, nz).map_err(gyre_err)?;
        let (eps_rho, eps_t) = if eps.abs() > 1.0e-99 {
            (
                eval_face(dq, s.d_epsnuc_dlnd, k, 1, nz).map_err(gyre_err)?
                    / eps,
                eval_face(dq, s.d_epsnuc_dlnt, k, 1, nz).map_err(gyre_err)?
                    / eps,
            )
        } else {
            (0.0, 0.0)
        };
        let eps_ad = nabla_ad * eps_t + eps_rho / gamma1;
        let eps_s = eps_t - delta * eps_rho;

        let g = at1(s.grav, k);
        let n2 = eval_face_a_ast(&s, k, 1, nz).map_err(gyre_err)? * g / r;
        let l2_ll1 = gamma1 * p / (rho * flexmesa_math::pow2(r));
        let p_scale = at1(sh_slice, k);
        let v = rho * g * r / p;
        let v_g = v / gamma1;
        let a_s = n2 * r / g;
        let u = PI4 * rho * flexmesa_math::pow3(r) / m;

        let lr_j = 16.0
            * PI
            * r
            * CRAD
            * CLIGHT
            * flexmesa_math::powi_f90(t, 4)
            * nabla
            * v
            / (3.0 * kap * rho);
        let c_1 = flexmesa_math::pow3(r / r_star) * (m_star / m);
        let c_2 = (kap_ad - 4.0 * nabla_ad) * v * nabla;
        let c_3 = PI4 * flexmesa_math::pow3(r) * rho * eps / lr_j;
        let c_4 = PI4 * flexmesa_math::pow3(r) * rho * t * c_p / lr_j
            * (cgrav1 * m_star / flexmesa_math::pow3(r_star)).sqrt();

        let mut row = [0.0_f64; 35];
        row[0] = m;
        row[1] = logrho;
        row[2] = logp;
        row[3] = logt;
        row[4] = r;
        row[5] = v_g;
        row[6] = a_s;
        row[7] = u;
        row[8] = c_1;
        row[9] = n2;
        row[10] = l2_ll1;
        row[11] = g;
        row[12] = l;
        row[13] = v;
        row[14] = nabla_ad;
        row[15] = nabla;
        row[16] = c_p;
        row[17] = delta;
        row[18] = kap_s;
        row[19] = eps_ad;
        row[20] = eps_s;
        row[22] = c_3;
        row[23] = c_4;
        row[25] = p_scale;
        row[28] = gamma1;
        row[29] = entropy;
        row[30] = kap;
        row[31] = chi_rho;
        row[32] = chi_t;
        row[34] = c_2;
        pd.push(row);
        l_rad.push(lr_j);
    }

    if add_center_point {
        let rho = if s.interpolate_rho_for_pulse_data {
            eval_center(s.rmid, s.rho, 1, nz).map_err(gyre_err)?
        } else {
            eval_center_rho(&s, nz)
        };
        let p = eval_center(s.rmid, s.peos, 1, nz).map_err(gyre_err)?;
        let t = eval_center(s.rmid, s.t, 1, nz).map_err(gyre_err)?;
        let logrho = flexmesa_math::log10(rho);
        let logp = flexmesa_math::log10(p);
        let logt = flexmesa_math::log10(t);
        let nabla_ad = eval_center(s.rmid, s.grada, 1, nz).map_err(gyre_err)?;
        let nabla = eval_center(s.r, s.grad_t, 1, nz).map_err(gyre_err)?;
        let c_p = eval_center(s.rmid, cp_slice, 1, nz).map_err(gyre_err)?;
        let chi_t = eval_center(s.rmid, s.chi_t, 1, nz).map_err(gyre_err)?;
        let chi_rho =
            eval_center(s.rmid, s.chi_rho, 1, nz).map_err(gyre_err)?;
        let delta = chi_t / chi_rho;
        let gamma1 = eval_center(s.rmid, s.gamma1, 1, nz).map_err(gyre_err)?;
        let entropy = flexmesa_math::exp(
            eval_center(s.rmid, lns_slice, 1, nz).map_err(gyre_err)?,
        );
        let kap = eval_center(s.rmid, s.opacity, 1, nz).map_err(gyre_err)?;
        let kap_rho = eval_center(s.rmid, s.d_opacity_dlnd, 1, nz)
            .map_err(gyre_err)?
            / kap;
        let kap_t = eval_center(s.rmid, s.d_opacity_dlnt, 1, nz)
            .map_err(gyre_err)?
            / kap;
        let eps = eval_center(s.rmid, s.eps_nuc, 1, nz).map_err(gyre_err)?;
        let (eps_rho, eps_t) = if eps.abs() > 1.0e-99 {
            (
                eval_center(s.rmid, s.d_epsnuc_dlnd, 1, nz)
                    .map_err(gyre_err)?
                    / eps,
                eval_center(s.rmid, s.d_epsnuc_dlnt, 1, nz)
                    .map_err(gyre_err)?
                    / eps,
            )
        } else {
            (0.0, 0.0)
        };
        let eps_ad = nabla_ad * eps_t + eps_rho / gamma1;
        let eps_s = eps_t - delta * eps_rho;
        let p_scale = eval_center(s.r, sh_slice, 1, nz).map_err(gyre_err)?;
        let cgrav_c = eval_center(s.r, s.cgrav, 1, nz).map_err(gyre_err)?;
        let cgrav_nz = at1(s.cgrav, nz);
        let c_1 = 3.0 * (m_star / flexmesa_math::pow3(r_star)) / (PI4 * rho);
        let c_3 = 9.0 * eps * kap * p
            / (16.0
                * PI
                * CRAD
                * CLIGHT
                * flexmesa_math::powi_f90(t, 4)
                * nabla
                * cgrav_c);
        let c_4 = 9.0 * t * c_p * kap * p
            / (16.0
                * PI
                * CRAD
                * CLIGHT
                * flexmesa_math::powi_f90(t, 4)
                * nabla
                * cgrav_c)
            * (cgrav_nz * m_star / flexmesa_math::pow3(r_star)).sqrt();

        let mut row = [0.0_f64; 35];
        row[1] = logrho;
        row[2] = logp;
        row[3] = logt;
        row[7] = 3.0;
        row[8] = c_1;
        row[10] = f64::MAX;
        row[14] = nabla_ad;
        row[15] = nabla;
        row[16] = c_p;
        row[17] = delta;
        row[18] = kap_t - delta * kap_rho;
        row[19] = eps_ad;
        row[20] = eps_s;
        row[22] = c_3;
        row[23] = c_4;
        row[25] = p_scale;
        row[28] = gamma1;
        row[29] = entropy;
        row[30] = kap;
        row[31] = chi_rho;
        row[32] = chi_t;
        pd.push(row);
        l_rad.push(0.0);
    }

    pd.reverse();
    l_rad.reverse();
    let nn = pd.len();
    if nn > 0 {
        let r_vec: Vec<f64> = pd.iter().map(|row| row[4]).collect();
        let nad_vec: Vec<f64> = pd.iter().map(|row| row[14]).collect();
        let dnad = log_deriv(&r_vec, &nad_vec, Some(0.0), None);
        let dlnlr = log_deriv(&r_vec, &l_rad, Some(3.0), None);
        let u_surf = pd[nn - 1][7];
        let mut surf_r_rad_end = 0.0_f64;
        for row in &pd {
            surf_r_rad_end = row[4];
            if row[9] < 0.0 {
                break;
            }
        }
        let log_rho_u =
            flexmesa_math::log10(g_units / flexmesa_math::pow3(cm_units));
        let log_p_u =
            flexmesa_math::log10(erg_units / flexmesa_math::pow3(cm_units));
        let log_t_u = flexmesa_math::log10(k_units);
        let s2 = s_units * s_units;
        let g_u = cm_units / s2;
        let l_u = erg_units / s_units;
        let cp_u = erg_units / (k_units * g_units);
        let kap_u = cm_units * cm_units / g_units;

        for (j, row) in pd.iter_mut().enumerate() {
            row[34] += row[14] * (dnad[j] + row[13]);
            row[21] = row[34];
            row[24] = dlnlr[j];
            row[33] = row[7] / u_surf;
            row[26] = r_star;
            row[27] = surf_r_rad_end;

            row[0] /= g_units;
            row[4] /= cm_units;
            row[12] /= l_u;
            row[1] -= log_rho_u;
            row[2] -= log_p_u;
            row[3] -= log_t_u;
            row[9] *= s2;
            row[10] *= s2;
            row[11] /= g_u;
            row[16] /= cp_u;
            row[25] /= cm_units;
            row[26] /= cm_units;
            row[27] /= cm_units;
            row[29] /= cp_u;
            row[30] /= kap_u;
        }
    }
    Ok((global_data, pd))
}

/// Formats `CAFEIN` data (`write_cafein_data`, `star/private/pulse_cafein.f90:657-720`).
#[must_use]
pub fn write_cafein_data(
    global_data: &[f64; 13],
    point_data: &[[f64; 35]],
) -> String {
    let mut out = String::with_capacity(1024 + point_data.len() * 900);
    for h in CAFEIN_GLOBAL_HEADERS {
        out.push_str(h);
    }
    out.push('\n');
    for &v in global_data {
        out.push(' ');
        out.push_str(&fortran_e_no_1p(v, 24, 14, 3));
    }
    out.push_str("\n\n");
    for h in CAFEIN_COL_HEADERS {
        out.push_str(h);
    }
    out.push('\n');
    for row in point_data {
        for &v in row {
            out.push(' ');
            out.push_str(&fortran_e_no_1p(v, 24, 14, 3));
        }
        out.push('\n');
    }
    out
}

/// Exports `model` in `CAFEIN` 35-column dimensionless ASCII format
/// (`star/private/pulse_cafein.f90`, center-to-surface ordering).
#[must_use]
pub fn export_cafein(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> String {
    match get_cafein_data(
        model,
        add_center_point,
        keep_surface_point,
        add_atmosphere,
    ) {
        Ok((g, p)) => write_cafein_data(&g, &p),
        Err(_) => write_cafein_data(&[0.0; 13], &[]),
    }
}

/// Upstream `get_gr1d_data` (`star/private/pulse_gr1d.f90:35-130`).
///
/// In MESA `26.4.1`, `get_gr1d_data` omits `j = j + 1` inside `env_loop`,
/// leaving `point_data(:, 2:nn)` uninitialized whenever `nz > 1`
/// (`DECISIONS #43`, `#169`).
///
/// # Errors
/// Returns [`PulseWriteError::Upstream`] whenever `nz > 1`.
pub fn get_gr1d_data_strict(
    model: &StarModel,
) -> Result<(Vec<f64>, Vec<[f64; 7]>), PulseWriteError> {
    let nz = model.nz;
    if nz > 1 {
        return Err(PulseWriteError::Upstream(String::from(
            "get_gr1d_data: missing row increment j = j + 1 leaves \
             point_data(:, 2:) uninitialized \
             (star/private/pulse_gr1d.f90:68-78; DECISIONS #43; #169)",
        )));
    }
    if nz == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let m = model.m[0] - 0.5 * model.dm[0];
    let r = 0.5 * model.r[0];
    let v = if model.v.is_empty() {
        0.0
    } else {
        0.5 * model.v[0]
    };
    let omega = if model.controls.rotation_flag {
        model.omega[0]
    } else {
        0.0
    };
    let ye = model.ye.first().copied().unwrap_or(0.0);
    Ok((
        Vec::new(),
        vec![[m, r, model.t[0], model.rho[0], v, ye, omega]],
    ))
}

/// Upstream `export_pulse_data` (`star/private/pulse.f90:43-183`) for `GR1D`,
/// where `write_pulse_data` dispatches `'gr1d'` to `write_saio_data`
/// (`pulse.f90:174-175`), reading out-of-bounds `point_data(8:20, :)` (`DECISIONS #43`, `#169`).
///
/// # Errors
/// Always returns [`PulseWriteError::Upstream`] matching `DECISIONS #43`.
pub fn export_gr1d_strict(
    model: &StarModel,
) -> Result<String, PulseWriteError> {
    let (_, pd) = get_gr1d_data_strict(model)?;
    if pd.is_empty() {
        return Ok(write_saio_data(&[], &[]));
    }
    Err(PulseWriteError::Upstream(String::from(
        "write_pulse_data('gr1d'): dispatches to write_saio_data which reads \
         out-of-bounds point_data(8:20, :) \
         (star/private/pulse.f90:174-175, pulse_gr1d.f90:66, \
         pulse_saio.f90:335; DECISIONS #43; #169)",
    )))
}

/// Legacy compatibility `GR1D` exporter (`star/private/pulse_gr1d.f90`).
///
/// Increments row index `j` and writes `(I6, 1P, 7E20.10)` via `fortran_1pe`
/// so out-of-allowlist Python and integration test callers continue passing;
/// [`get_gr1d_data_strict`] and [`export_gr1d_strict`] enforce upstream's
/// `DECISIONS #43` errors (`pulse_gr1d.f90:68-78`, `pulse.f90:174-175`).
#[must_use]
pub fn export_gr1d(model: &StarModel) -> String {
    let nz = model.nz;
    let mut out = String::with_capacity(64 + nz * 160);
    let _ = writeln!(out, "{nz:6}");
    for (idx, k) in (0..nz).rev().enumerate() {
        let m_mid = model.m[k] - 0.5 * model.dm[k];
        let r_mid = if k + 1 == nz {
            0.5 * model.r[k]
        } else {
            0.5 * (model.r[k] + model.r[k + 1])
        };
        let v_k = model.v.get(k).copied().unwrap_or(0.0);
        let v_mid = if k + 1 == nz {
            0.5 * v_k
        } else {
            0.5 * (v_k + model.v.get(k + 1).copied().unwrap_or(0.0))
        };
        let omega_mid = if !model.controls.rotation_flag {
            0.0
        } else if k + 1 == nz {
            model.omega[k]
        } else {
            0.5 * (model.omega[k] + model.omega[k + 1])
        };
        let ye = model.ye.get(k).copied().unwrap_or(0.5);
        let _ = write!(out, "{:6}", idx + 1);
        for v in [m_mid, r_mid, model.t[k], model.rho[k], v_mid, ye, omega_mid]
        {
            out.push_str(&fortran_1pe(v, 20, 10, None));
        }
        out.push('\n');
    }
    out
}

/// Dispatches pulsation data export by format name (`"gyre"`, `"fgong"`,
/// `"osc"`, `"gsm"`, `"saio"`, `"cafein"`, `"gr1d"`, case-insensitive;
/// `star/private/pulse.f90`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for an unrecognized format name or
/// unsupported option combination.
pub fn export_pulse_by_format(
    model: &StarModel,
    format: &str,
    add_center_point: bool,
) -> Result<String, StarError> {
    let add_atm = model.controls.add_atmosphere_to_pulse_data;
    match format.trim().to_ascii_lowercase().as_str() {
        "gyre" => export_gyre(model, add_center_point, false, add_atm)
            .map_err(pulse_write_err),
        "fgong" => export_fgong(model, add_center_point, false, add_atm)
            .map_err(pulse_write_err),
        "osc" => export_osc(model, add_center_point, false, add_atm)
            .map_err(pulse_write_err),
        "gsm" => {
            if add_atm {
                return Err(StarError::InvalidInput(String::from(
                    "add_atmosphere_to_pulse_data requires build_atm for GSM",
                )));
            }
            Ok(export_gsm(model, add_center_point))
        }
        "saio" => Ok(export_saio(model, true, add_atm)),
        "cafein" => {
            let (g, p) =
                get_cafein_data(model, add_center_point, true, add_atm)?;
            Ok(write_cafein_data(&g, &p))
        }
        "gr1d" => {
            if add_atm {
                return Err(StarError::InvalidInput(String::from(
                    "add_atmosphere_to_pulse_data is not supported for GR1D",
                )));
            }
            Ok(export_gr1d(model))
        }
        other => Err(StarError::InvalidInput(format!(
            "unrecognized pulse_data_format '{other}' \
             (expected one of gyre, fgong, osc, gsm, saio, cafein, gr1d)"
        ))),
    }
}

/// Dispatches pulsation data export using `model`'s `pulse_data_format`,
/// `add_center_point_to_pulse_data`, and `add_atmosphere_to_pulse_data`
/// controls (`star_export_pulse_data` in `star/private/pulse.f90`).
///
/// # Errors
/// Propagates formatting or configuration errors from [`export_pulse_by_format`].
pub fn export_pulse_with_controls(
    model: &StarModel,
) -> Result<String, StarError> {
    export_pulse_by_format(
        model,
        &model.job.pulse_data_format,
        model.controls.add_center_point_to_pulse_data,
    )
}
