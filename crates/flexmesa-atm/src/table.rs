// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tabulated stellar and white dwarf atmosphere evaluation (`table_atm.f90` and `atm_table.f90`).
//!
//! Unpacks `crates/flexmesa-atm/data/atm_tables.bin` (`FATM0001`), fits the 2D
//! not-a-knot bicubic splines in $(\log g, T_{\mathrm{eff}})$ and 1D Steffen
//! spline for $\log_{10} E_2(x)$, and implements `get_table_values`,
//! `eval_table`, `get_table_alfa_beta`, and `get_table_base`.

use crate::types::{
    AtmError, AtmEvalOutput, ATM_TABLE_DB_WD_TAU_25, ATM_TABLE_PHOTOSPHERE,
    ATM_TABLE_TAU_1, ATM_TABLE_TAU_10, ATM_TABLE_TAU_100, ATM_TABLE_TAU_1M1,
    ATM_TABLE_WD_TAU_25,
};
use crate::utils::radiation_pressure;
use flexmesa_const::LN10;
use flexmesa_interp1d::{interp_pm, interp_value};
use flexmesa_interp2d::{interp_evbicub_db, interp_mkbicub_db, BicubicBc};
use flexmesa_math::safe_log10;
use flexmesa_utils::is_bad;
use std::sync::LazyLock;

/// Grevesse & Noels (1993) solar metallicity (`GN93_zsol = 0.0179` in `chem_def.f90`).
pub const GN93_ZSOL: f64 = 0.0179;

const ATM_TABLES_BIN: &[u8] = include_bytes!("../data/atm_tables.bin");

/// Single atmosphere table set (`Atm_Info` in `atm_def.f90`).
#[derive(Debug, Clone)]
pub struct AtmTableInfo {
    /// Table identifier (`ATM_TABLE_*`).
    pub id: i32,
    /// Number of metallicity slices $n_Z$.
    pub nz: usize,
    /// Number of effective temperature grid points $n_T$.
    pub nt: usize,
    /// Number of surface gravity grid points $n_g$.
    pub ng: usize,
    /// Whether this table includes tabulated $T(\log g, T_{\mathrm{eff}})$ (`false` for `ATM_TABLE_PHOTOSPHERE`).
    pub has_temp: bool,
    /// Uniform grid flag along $\log g$.
    pub iling: bool,
    /// Uniform grid flag along $T_{\mathrm{eff}}$.
    pub ilint: bool,
    /// Metallicity grid $\log_{10}(Z / Z_{\odot,\mathrm{GN93}})$ of length `nz`.
    pub log_z: Vec<f64>,
    /// $[\alpha/\mathrm{Fe}]$ enhancement of length `nz`.
    pub alpha_fe: Vec<f64>,
    /// Effective temperature grid $T_{\mathrm{eff}}$ ($\mathrm{K}$) of length `nt`.
    pub teff_array: Vec<f64>,
    /// Surface gravity grid $\log_{10}(g / \mathrm{cm\,s^{-2}})$ of length `ng`.
    pub logg_array: Vec<f64>,
    /// Maximum valid $T_{\mathrm{eff}}$ ($\mathrm{K}$) at each $\log g$ index of length `ng`.
    pub teff_bound: Vec<f64>,
    /// Precomputed 2D bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]` for $P_{\mathrm{gas}}$ per $Z$ slice.
    pub pgas_splines: Vec<Vec<[f64; 4]>>,
    /// Precomputed 2D bicubic spline coefficients for $T$ per $Z$ slice (empty when `has_temp == false`).
    pub temp_splines: Vec<Vec<[f64; 4]>>,
}

/// Embedded atmosphere tables and $E_2(x)$ Steffen spline coefficients.
pub struct AtmTables {
    /// 571 $x$ nodes for $E_2(x)$ (`[0.0, 10.0]`).
    pub e2_x: Vec<f64>,
    /// Fitted 1D Steffen cubic spline coefficients for $\log_{10} E_2(x)$.
    pub e2_f: Vec<[f64; 4]>,
    /// Seven atmosphere table sets (`ATM_TABLE_PHOTOSPHERE`, `TAU_100`, `TAU_10`, `TAU_1`, `TAU_1M1`, `WD_TAU_25`, `DB_WD_TAU_25`).
    pub sets: Vec<AtmTableInfo>,
}

static BUILTIN_ATM_TABLES: LazyLock<AtmTables> =
    LazyLock::new(|| parse_atm_tables(ATM_TABLES_BIN));

fn read_u32(bytes: &[u8], pos: &mut usize) -> usize {
    let v =
        u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    v
}

fn read_f64(bytes: &[u8], pos: &mut usize) -> f64 {
    let v = f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn read_f64_vec(bytes: &[u8], pos: &mut usize, n: usize) -> Vec<f64> {
    (0..n).map(|_| read_f64(bytes, pos)).collect()
}

fn parse_atm_tables(bytes: &[u8]) -> AtmTables {
    assert_eq!(&bytes[..8], b"FATM0001", "invalid atm_tables.bin magic");
    let mut pos = 8;
    let n_e2 = read_u32(bytes, &mut pos);
    let mut e2_x = Vec::with_capacity(n_e2);
    let mut e2_y = Vec::with_capacity(n_e2);
    for _ in 0..n_e2 {
        e2_x.push(read_f64(bytes, &mut pos));
        e2_y.push(read_f64(bytes, &mut pos));
    }
    let e2_f = interp_pm(&e2_x, &e2_y).expect("failed to fit E2 interp_pm");

    let n_sets = read_u32(bytes, &mut pos);
    let mut sets = Vec::with_capacity(n_sets);
    for _ in 0..n_sets {
        let id = read_u32(bytes, &mut pos) as i32;
        let nz = read_u32(bytes, &mut pos);
        let nt = read_u32(bytes, &mut pos);
        let ng = read_u32(bytes, &mut pos);
        let has_temp = bytes[pos] != 0;
        pos += 1;

        let log_z = read_f64_vec(bytes, &mut pos, nz);
        let alpha_fe = read_f64_vec(bytes, &mut pos, nz);
        let teff_array = read_f64_vec(bytes, &mut pos, nt);
        let logg_array = read_f64_vec(bytes, &mut pos, ng);
        let teff_bound = read_f64_vec(bytes, &mut pos, ng);

        let mut pgas_splines = Vec::with_capacity(nz);
        let mut iling = false;
        let mut ilint = false;

        for _ in 0..nz {
            let mut z_grid = vec![0.0_f64; ng * nt];
            for it in 0..nt {
                for ig in 0..ng {
                    z_grid[ig * nt + it] = read_f64(bytes, &mut pos);
                }
            }
            let (coeffs, ig_lin, it_lin) = interp_mkbicub_db(
                &logg_array,
                &teff_array,
                &z_grid,
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
            )
            .expect("failed to fit Pgas bicubic spline");
            iling = ig_lin;
            ilint = it_lin;
            pgas_splines.push(coeffs);
        }

        let mut temp_splines = Vec::new();
        if has_temp {
            temp_splines.reserve(nz);
            for _ in 0..nz {
                let mut z_grid = vec![0.0_f64; ng * nt];
                for it in 0..nt {
                    for ig in 0..ng {
                        z_grid[ig * nt + it] = read_f64(bytes, &mut pos);
                    }
                }
                let (coeffs, _, _) = interp_mkbicub_db(
                    &logg_array,
                    &teff_array,
                    &z_grid,
                    BicubicBc::NotAKnot,
                    &[],
                    BicubicBc::NotAKnot,
                    &[],
                    BicubicBc::NotAKnot,
                    &[],
                    BicubicBc::NotAKnot,
                    &[],
                )
                .expect("failed to fit T bicubic spline");
                temp_splines.push(coeffs);
            }
        }

        sets.push(AtmTableInfo {
            id,
            nz,
            nt,
            ng,
            has_temp,
            iling,
            ilint,
            log_z,
            alpha_fe,
            teff_array,
            logg_array,
            teff_bound,
            pgas_splines,
            temp_splines,
        });
    }

    AtmTables { e2_x, e2_f, sets }
}

impl AtmTables {
    /// Returns a reference to the lazily initialized built-in atmosphere tables.
    #[must_use]
    pub fn builtin() -> &'static Self {
        &BUILTIN_ATM_TABLES
    }

    /// Finds the [`AtmTableInfo`] for `id`.
    ///
    /// # Errors
    /// Returns [`AtmError::UnknownId`] if `id` is not one of the 7 `ATM_TABLE_*` IDs.
    pub fn get_info(&self, id: i32) -> Result<&AtmTableInfo, AtmError> {
        self.sets
            .iter()
            .find(|s| s.id == id)
            .ok_or(AtmError::UnknownId {
                context: "AtmTables::get_info",
                id,
            })
    }
}

/// Interpolates `(Pgas, dPgas_dTeff, dPgas_dlogg, T, dT_dTeff, dT_dlogg)` in
/// $(Z, \log g, T_{\mathrm{eff}})$ (`get_table_values` in `table_atm.f90`).
///
/// # Errors
/// Returns [`AtmError`] if `id` is unknown or spline interpolation fails.
pub fn get_table_values(
    id: i32,
    new_z: f64,
    new_logg_in: f64,
    new_teff_in: f64,
) -> Result<(f64, f64, f64, f64, f64, f64), AtmError> {
    let ai = AtmTables::builtin().get_info(id)?;
    let nz = ai.nz;
    let nt = ai.nt;
    let ng = ai.ng;

    let mut new_log_z = safe_log10(new_z / GN93_ZSOL);
    if new_log_z < ai.log_z[0] {
        new_log_z = ai.log_z[0];
    } else if new_log_z > ai.log_z[nz - 1] {
        new_log_z = ai.log_z[nz - 1];
    }

    let (new_teff, clip_teff) = if new_teff_in < ai.teff_array[0] {
        (ai.teff_array[0], true)
    } else if new_teff_in > ai.teff_array[nt - 1] {
        (ai.teff_array[nt - 1], true)
    } else {
        (new_teff_in, false)
    };

    let (new_logg, clip_logg) = if new_logg_in < ai.logg_array[0] {
        (ai.logg_array[0], true)
    } else if new_logg_in > ai.logg_array[ng - 1] {
        (ai.logg_array[ng - 1], true)
    } else {
        (new_logg_in, false)
    };

    let z_index = if new_log_z <= ai.log_z[0] {
        1
    } else if new_log_z >= ai.log_z[nz - 1] {
        nz
    } else {
        let mut idx = nz;
        for j in 1..nz {
            if ai.log_z[j - 1] <= new_log_z {
                idx = j;
            }
        }
        idx
    };

    let mut zlo = z_index.saturating_sub(1).max(1);
    let mut zhi = (z_index + 2).min(nz);
    if zhi - zlo != 3 {
        zlo = zlo.saturating_sub(1).max(1);
    }
    if zhi - zlo != 3 {
        zhi = (zhi + 1).min(nz);
    }
    let num_zs = zhi - zlo + 1;

    let do_interp =
        |splines: &[Vec<[f64; 4]>]| -> Result<(f64, f64, f64), AtmError> {
            let mut vals = Vec::with_capacity(num_zs);
            let mut d_dlogg = Vec::with_capacity(num_zs);
            let mut d_dteff = Vec::with_capacity(num_zs);

            for iz in (zlo - 1)..zhi {
                let ev = interp_evbicub_db(
                    new_logg,
                    new_teff,
                    &ai.logg_array,
                    &ai.teff_array,
                    ai.iling,
                    ai.ilint,
                    &splines[iz],
                )
                .map_err(|e| AtmError::CallbackError {
                    context: "get_table_values::interp_evbicub_db",
                    detail: e.to_string(),
                })?;
                vals.push(ev.f);
                d_dlogg.push(ev.df_dx);
                d_dteff.push(ev.df_dy);
            }

            if num_zs == 1 {
                Ok((vals[0], d_dlogg[0], d_dteff[0]))
            } else {
                let z_sub = &ai.log_z[(zlo - 1)..zhi];
                let interp_z = |ys: &[f64]| -> Result<f64, AtmError> {
                    let f = interp_pm(z_sub, ys).map_err(|e| {
                        AtmError::CallbackError {
                            context: "get_table_values::interp_pm",
                            detail: e.to_string(),
                        }
                    })?;
                    interp_value(z_sub, &f, new_log_z).map_err(|e| {
                        AtmError::CallbackError {
                            context: "get_table_values::interp_value",
                            detail: e.to_string(),
                        }
                    })
                };
                Ok((interp_z(&vals)?, interp_z(&d_dlogg)?, interp_z(&d_dteff)?))
            }
        };

    let (new_pgas, mut dpgas_dlogg, mut dpgas_dteff) =
        do_interp(&ai.pgas_splines)?;
    if clip_logg {
        dpgas_dlogg = 0.0;
    }
    if clip_teff {
        dpgas_dteff = 0.0;
    }

    if id == ATM_TABLE_PHOTOSPHERE {
        let new_t = new_teff;
        let dt_dteff = if clip_teff { 0.0 } else { 1.0 };
        let dt_dlogg = 0.0;
        return Ok((
            new_pgas,
            dpgas_dteff,
            dpgas_dlogg,
            new_t,
            dt_dteff,
            dt_dlogg,
        ));
    }

    let (new_t, mut dt_dlogg, mut dt_dteff) = do_interp(&ai.temp_splines)?;
    if clip_logg {
        dt_dlogg = 0.0;
    }
    if clip_teff {
        dt_dteff = 0.0;
    }

    if is_bad(new_pgas) || is_bad(new_t) {
        return Err(AtmError::CallbackError {
            context: "get_table_values",
            detail: "NaN or infinite Pgas or T from table interpolation"
                .to_string(),
        });
    }

    Ok((
        new_pgas,
        dpgas_dteff,
        dpgas_dlogg,
        new_t,
        dt_dteff,
        dt_dlogg,
    ))
}

/// Evaluates surface $(\ln T, \ln P)$ and partial derivatives from a tabulated
/// model atmosphere (`eval_table` in `atm_table.f90`).
///
/// # Errors
/// Returns [`AtmError`] if $L, R, M \le 0$ or table interpolation fails.
#[allow(clippy::too_many_arguments)]
pub fn eval_table(
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    id: i32,
    z: f64,
    skip_partials: bool,
    teff: f64,
) -> Result<AtmEvalOutput, AtmError> {
    if l <= 0.0 || r <= 0.0 || m <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "eval_table",
            detail: format!("require L, R, M > 0, got L={l}, R={r}, M={m}"),
        });
    }
    let g = cgrav * m / (r * r);
    let logg = g.log10();

    let (pgas, dpgas_dteff, dpgas_dlogg, t, dt_dteff, dt_dlogg) =
        get_table_values(id, z, logg, teff)?;

    let ln_t = t.ln();
    let prad = radiation_pressure(t);
    let p = pgas + prad;
    let ln_p = p.ln();

    if is_bad(ln_p) || is_bad(ln_t) {
        return Err(AtmError::CallbackError {
            context: "eval_table",
            detail: "non-finite lnP or lnT".to_string(),
        });
    }

    if skip_partials {
        return Ok(AtmEvalOutput {
            ln_t,
            dln_t_dl: 0.0,
            dln_t_dln_r: 0.0,
            dln_t_dln_m: 0.0,
            dln_t_dln_kap: 0.0,
            ln_p,
            dln_p_dl: 0.0,
            dln_p_dln_r: 0.0,
            dln_p_dln_m: 0.0,
            dln_p_dln_kap: 0.0,
            kap: 0.0,
        });
    }

    let dln_teff_dln_r = -0.5;
    let dln_teff_dl = 0.25 / l;
    let dteff_dln_r = teff * dln_teff_dln_r;

    let dlogg_dln_r = -2.0 / LN10;
    let dlogg_dln_m = 1.0 / LN10;

    let dpgas_dln_r = dpgas_dlogg * dlogg_dln_r + dpgas_dteff * dteff_dln_r;
    let dpgas_dln_m = dpgas_dlogg * dlogg_dln_m;
    let dpgas_dln_teff = dpgas_dteff * teff;

    let dprad_dln_teff = 4.0 * prad * dt_dteff * teff / t;

    let dln_p_dl = (dpgas_dln_teff + dprad_dln_teff) * dln_teff_dl / p;
    let dln_p_dln_r = (dpgas_dln_r + dprad_dln_teff * dln_teff_dln_r) / p;
    let dln_p_dln_m = dpgas_dln_m / p;

    let dt_dln_teff = dt_dteff * teff;
    let dt_dln_r = dt_dlogg * dlogg_dln_r + dt_dln_teff * dln_teff_dln_r;
    let dt_dln_m = dt_dlogg * dlogg_dln_m;

    let dln_t_dl = dt_dln_teff * dln_teff_dl / t;
    let dln_t_dln_r = dt_dln_r / t;
    let dln_t_dln_m = dt_dln_m / t;

    Ok(AtmEvalOutput {
        ln_t,
        dln_t_dl,
        dln_t_dln_r,
        dln_t_dln_m,
        dln_t_dln_kap: 0.0,
        ln_p,
        dln_p_dl,
        dln_p_dln_r,
        dln_p_dln_m,
        dln_p_dln_kap: 0.0,
        kap: 0.0,
    })
}

/// Evaluates blending weights $(\alpha, \beta)$ between grey ($1, 0$) and table ($0, 1$)
/// atmospheres near table boundaries (`get_table_alfa_beta` in `atm_table.f90`).
///
/// # Errors
/// Returns [`AtmError`] if $L, R, M \le 0$ or `id` is unknown.
pub fn get_table_alfa_beta(
    l: f64,
    teff: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    id: i32,
) -> Result<(f64, f64), AtmError> {
    if l <= 0.0 || r <= 0.0 || m <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "get_table_alfa_beta",
            detail: "require L, R, M > 0".to_string(),
        });
    }
    let ai = AtmTables::builtin().get_info(id)?;
    let g = cgrav * m / (r * r);
    let log_teff = teff.log10();
    let logg = g.log10();

    let ng = ai.ng;
    let logg_max = ai.logg_array[ng - 1];
    let logg_min = ai.logg_array[0];

    let logg_index = if logg <= logg_min {
        1
    } else if logg >= logg_max {
        ng
    } else {
        let mut idx = ng;
        for j in 1..ng {
            if ai.logg_array[j - 1] <= logg {
                idx = j;
            }
        }
        idx
    };

    let teff_max = ai.teff_bound[logg_index - 1];
    let teff_min = ai.teff_array[0];
    let log_teff_max = teff_max.log10();
    let log_teff_min = teff_min.log10();

    let logg1 = logg_max;
    let logg2 = logg_max - 0.01;
    let logg3 = logg_min + 0.5;
    let logg4 = logg_min;
    let log_teff1 = log_teff_max;
    let log_teff2 = log_teff_max - 0.01;
    let log_teff3 = log_teff_min;

    let alfa = if logg < logg4
        || logg > logg1
        || log_teff > log_teff1
        || log_teff < log_teff3
    {
        1.0
    } else if log_teff > log_teff2 {
        let c_dy = (log_teff - log_teff2) / (log_teff1 - log_teff2);
        if logg > logg2 {
            let c_dx = (logg - logg2) / (logg1 - logg2);
            (c_dx * c_dx + c_dy * c_dy).sqrt().min(1.0)
        } else if logg > logg3 {
            c_dy
        } else {
            let c_dx = (logg - logg3) / (logg4 - logg3);
            (c_dx * c_dx + c_dy * c_dy).sqrt().min(1.0)
        }
    } else if logg > logg2 {
        (logg - logg2) / (logg1 - logg2)
    } else if logg > logg3 {
        0.0
    } else {
        (logg - logg3) / (logg4 - logg3)
    };

    Ok((alfa, 1.0 - alfa))
}

/// Returns the base optical depth $\tau_{\mathrm{base}}$ of atmosphere table `id` (`get_table_base` in `atm_table.f90`).
///
/// # Errors
/// Returns [`AtmError::UnknownId`] if `id` is not one of the 7 `ATM_TABLE_*` IDs.
pub fn get_table_base(id: i32) -> Result<f64, AtmError> {
    match id {
        ATM_TABLE_PHOTOSPHERE => Ok(2.0 / 3.0),
        ATM_TABLE_TAU_100 => Ok(100.0),
        ATM_TABLE_TAU_10 => Ok(10.0),
        ATM_TABLE_TAU_1 => Ok(1.0),
        ATM_TABLE_TAU_1M1 => Ok(0.1),
        ATM_TABLE_WD_TAU_25 => Ok(25.1188),
        ATM_TABLE_DB_WD_TAU_25 => Ok(25.0),
        _ => Err(AtmError::UnknownId {
            context: "get_table_base",
            id,
        }),
    }
}
