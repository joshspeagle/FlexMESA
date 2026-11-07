// SPDX-License-Identifier: LGPL-3.0-or-later
//! Type 2 C/O-enhanced opacity evaluation (`kap_eval_co.f90`).
//!
//! Mirrors `Get1_kap_CO_Results`, `Get_Kap_for_CO_Z_cubic`, `Get_Kap_for_CO_Z_linear`,
//! `Get_Kap_for_CO_X`, and `Get_Kap_for_dXCO` in `kap/private/kap_eval_co.f90`.

use crate::co_find::{
    find_co_tables, get_co_kap_for_log_rho_log_t, get_kap_at_dxc,
    get_kap_at_dxo,
};
use crate::loader::{get_dx_lookup, kap_tables, KapCoXTable, KapCoZTable};
use crate::types::{KapConfig, KapError};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_interp1d::interpolate_vector_pm_autodiff;

/// Evaluate Type 2 C/O-enhanced opacity at `(Zbase, X, dXC, dXO, logRho, logT)`
/// (`Get1_kap_CO_Results` in `kap_eval_co.f90`).
///
/// Returns `(log10_kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if the table set is missing or coordinates lie outside bounds.
pub fn get1_kap_co_results(
    cfg: &KapConfig,
    zbase: f64,
    x: f64,
    dxc: f64,
    dxo: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let tables = kap_tables();
    let prefix = cfg.kap_co_option.prefix();
    let set = tables
        .co_sets
        .iter()
        .find(|s| s.prefix == prefix)
        .unwrap_or(&tables.co_sets[0]);
    let z_tables = &set.z_tables;
    let num_zs = z_tables.len();
    if num_zs == 0 {
        return Err(KapError::MissingTable("empty z_tables in CO set"));
    }

    if num_zs == 1 || zbase >= z_tables[num_zs - 1].zbase {
        return get_kap_for_co_x(
            cfg,
            &z_tables[num_zs - 1],
            dxc,
            dxo,
            x,
            log_rho,
            log_t,
        );
    }
    if zbase <= z_tables[0].zbase {
        return get_kap_for_co_x(
            cfg,
            &z_tables[0],
            dxc,
            dxo,
            x,
            log_rho,
            log_t,
        );
    }

    let mut iz = 0_usize;
    for idx in 0..(num_zs - 1) {
        if zbase < z_tables[idx + 1].zbase {
            iz = idx;
            break;
        }
    }
    let z0 = z_tables[iz].zbase;
    let z1 = z_tables[iz + 1].zbase;
    if zbase <= z0 {
        return get_kap_for_co_x(
            cfg,
            &z_tables[iz],
            dxc,
            dxo,
            x,
            log_rho,
            log_t,
        );
    }
    if zbase >= z1 {
        return get_kap_for_co_x(
            cfg,
            &z_tables[iz + 1],
            dxc,
            dxo,
            x,
            log_rho,
            log_t,
        );
    }

    if num_zs >= 4 && cfg.cubic_interpolation_in_z {
        let i1 = if iz + 2 >= num_zs {
            num_zs - 3
        } else if iz == 0 {
            1
        } else {
            iz
        };
        let indices = [i1 - 1, i1, i1 + 1, i1 + 2];
        let all_present = indices
            .iter()
            .all(|&idx| z_tables[idx].x_tables.iter().any(|xt| xt.present));
        if all_present {
            let mut z_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
            let mut v_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
            for (k, &izz) in indices.iter().enumerate() {
                z_old[k] = AutoDiffReal2VarOrder1::from(z_tables[izz].zbase);
                let (lk, dlk_dr, dlk_dt) = get_kap_for_co_x(
                    cfg,
                    &z_tables[izz],
                    dxc,
                    dxo,
                    x,
                    log_rho,
                    log_t,
                )?;
                v_old[k] = AutoDiffReal2VarOrder1 {
                    val: lk,
                    d1: [dlk_dt, dlk_dr],
                };
            }
            let z_new = [AutoDiffReal2VarOrder1::from(zbase)];
            let v_new = interpolate_vector_pm_autodiff(&z_old, &v_old, &z_new)
                .map_err(|_| {
                    KapError::NumericalFailure(
                        "interp_pm_autodiff failed in CO Z",
                    )
                })?;
            return Ok((v_new[0].val, v_new[0].d1[1], v_new[0].d1[0]));
        }
    }

    let iz0 = nearest_present_co_z(z_tables, iz);
    let iz1 = nearest_present_co_z(z_tables, iz + 1);
    let (lk0, dlk0_dr, dlk0_dt) =
        get_kap_for_co_x(cfg, &z_tables[iz0], dxc, dxo, x, log_rho, log_t)?;
    if iz0 == iz1 {
        return Ok((lk0, dlk0_dr, dlk0_dt));
    }
    let (lk1, dlk1_dr, dlk1_dt) =
        get_kap_for_co_x(cfg, &z_tables[iz1], dxc, dxo, x, log_rho, log_t)?;
    let z0_act = z_tables[iz0].zbase;
    let z1_act = z_tables[iz1].zbase;
    let beta = (zbase - z1_act) / (z0_act - z1_act);
    let alfa = 1.0 - beta;
    Ok((
        beta * lk0 + alfa * lk1,
        beta * dlk0_dr + alfa * dlk1_dr,
        beta * dlk0_dt + alfa * dlk1_dt,
    ))
}

fn nearest_present_co_z(z_tables: &[KapCoZTable], target_iz: usize) -> usize {
    if z_tables[target_iz].x_tables.iter().any(|xt| xt.present) {
        return target_iz;
    }
    let mut best = target_iz;
    let mut best_dist = f64::INFINITY;
    for (idx, zt) in z_tables.iter().enumerate() {
        if zt.x_tables.iter().any(|xt| xt.present) {
            let d = (zt.zbase - z_tables[target_iz].zbase).abs();
            if d < best_dist {
                best_dist = d;
                best = idx;
            }
        }
    }
    best
}

fn nearest_present_co_x(x_tables: &[KapCoXTable], target_ix: usize) -> usize {
    if x_tables[target_ix].present {
        return target_ix;
    }
    let mut best = target_ix;
    let mut best_dist = f64::INFINITY;
    for (idx, xt) in x_tables.iter().enumerate() {
        if xt.present {
            let d = (xt.x - x_tables[target_ix].x).abs();
            if d < best_dist {
                best_dist = d;
                best = idx;
            }
        }
    }
    best
}

fn get_kap_for_co_x(
    cfg: &KapConfig,
    zt: &KapCoZTable,
    dxc: f64,
    dxo: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    if !(0.0..=1.0).contains(&x) {
        return Err(KapError::InvalidComposition("X outside [0, 1] in CO"));
    }
    let x_tables = &zt.x_tables;
    let num_xs = x_tables.len();
    if num_xs == 1 || x >= x_tables[num_xs - 1].x {
        let idx = nearest_present_co_x(x_tables, num_xs - 1);
        return get_kap_for_dxco(&x_tables[idx], dxc, dxo, log_rho, log_t);
    }
    if x <= x_tables[0].x {
        let idx = nearest_present_co_x(x_tables, 0);
        return get_kap_for_dxco(&x_tables[idx], dxc, dxo, log_rho, log_t);
    }

    let mut ix = num_xs - 1;
    for i in 0..(num_xs - 1) {
        if x < x_tables[i + 1].x {
            ix = i;
            break;
        }
    }
    if ix == num_xs - 1 {
        let idx = nearest_present_co_x(x_tables, num_xs - 1);
        return get_kap_for_dxco(&x_tables[idx], dxc, dxo, log_rho, log_t);
    }

    let x0 = x_tables[ix].x;
    let x1 = x_tables[ix + 1].x;
    if x0 >= x {
        let idx = nearest_present_co_x(x_tables, ix);
        return get_kap_for_dxco(&x_tables[idx], dxc, dxo, log_rho, log_t);
    }
    if x1 <= x {
        let idx = nearest_present_co_x(x_tables, ix + 1);
        return get_kap_for_dxco(&x_tables[idx], dxc, dxo, log_rho, log_t);
    }

    if num_xs >= 4 && cfg.cubic_interpolation_in_x {
        let i1 = if ix + 2 >= num_xs {
            num_xs - 3
        } else if ix == 0 {
            1
        } else {
            ix
        };
        let indices = [i1 - 1, i1, i1 + 1, i1 + 2];
        if indices.iter().all(|&idx| x_tables[idx].present) {
            let mut x_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
            let mut v_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
            for (k, &ixx) in indices.iter().enumerate() {
                x_old[k] = AutoDiffReal2VarOrder1::from(x_tables[ixx].x);
                let (lk, dlk_dr, dlk_dt) =
                    get_kap_for_dxco(&x_tables[ixx], dxc, dxo, log_rho, log_t)?;
                v_old[k] = AutoDiffReal2VarOrder1 {
                    val: lk,
                    d1: [dlk_dt, dlk_dr],
                };
            }
            let x_new = [AutoDiffReal2VarOrder1::from(x)];
            let v_new = interpolate_vector_pm_autodiff(&x_old, &v_old, &x_new)
                .map_err(|_| {
                    KapError::NumericalFailure(
                        "interp_pm_autodiff failed in CO X",
                    )
                })?;
            return Ok((v_new[0].val, v_new[0].d1[1], v_new[0].d1[0]));
        }
    }

    let ix0 = nearest_present_co_x(x_tables, ix);
    let ix1 = nearest_present_co_x(x_tables, ix + 1);
    let (lk0, dlk0_dr, dlk0_dt) =
        get_kap_for_dxco(&x_tables[ix0], dxc, dxo, log_rho, log_t)?;
    if ix0 == ix1 {
        return Ok((lk0, dlk0_dr, dlk0_dt));
    }
    let (lk1, dlk1_dr, dlk1_dt) =
        get_kap_for_dxco(&x_tables[ix1], dxc, dxo, log_rho, log_t)?;
    let x0_act = x_tables[ix0].x;
    let x1_act = x_tables[ix1].x;
    let beta = (x - x1_act) / (x0_act - x1_act);
    let alfa = 1.0 - beta;
    Ok((
        beta * lk0 + alfa * lk1,
        beta * dlk0_dr + alfa * dlk1_dr,
        beta * dlk0_dt + alfa * dlk1_dt,
    ))
}

fn get_kap_for_dxco(
    xt: &KapCoXTable,
    dxc_in: f64,
    dxo_in: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let mut dxc = dxc_in.max(0.0);
    let mut dxo = dxo_in.max(0.0);
    let dxco_max = 1.0 - (xt.x + xt.z);
    if dxc + dxo > dxco_max {
        let fac = dxco_max / (dxc + dxo);
        dxc *= fac;
        dxo *= fac;
    }

    let dxc_lookup = get_dx_lookup(dxc, xt.z);
    let dxo_lookup = get_dx_lookup(dxo, xt.z);

    let (i1, i2, i3, i4) =
        find_co_tables(xt, dxco_max, dxc, dxo, dxc_lookup, dxo_lookup)?;

    let matches_exact = |idx_1based: i32| -> bool {
        if idx_1based < 1 {
            return false;
        }
        let sub = &xt.co_tables[(idx_1based as usize) - 1];
        (dxc_lookup - sub.dxc_lookup).abs() == 0.0
            && (dxo_lookup - sub.dxo_lookup).abs() == 0.0
    };

    if i1 > 0 && i2 <= 0 && i3 <= 0 && i4 <= 0 {
        return get_co_kap_for_log_rho_log_t(xt, i1, log_rho, log_t);
    }
    if i1 <= 0 || i2 <= 0 || i3 <= 0 {
        return Err(KapError::NumericalFailure(
            "invalid Find_CO_Tables result",
        ));
    }
    if matches_exact(i2) {
        return get_co_kap_for_log_rho_log_t(xt, i2, log_rho, log_t);
    }
    if matches_exact(i3) {
        return get_co_kap_for_log_rho_log_t(xt, i3, log_rho, log_t);
    }
    if i4 > 0 && matches_exact(i4) {
        return get_co_kap_for_log_rho_log_t(xt, i4, log_rho, log_t);
    }

    let (lk1, dlk1_dr, dlk1_dt) =
        get_co_kap_for_log_rho_log_t(xt, i1, log_rho, log_t)?;
    let (dxc1_l, dxo1_l) = {
        let s = &xt.co_tables[(i1 as usize) - 1];
        (s.dxc_lookup, s.dxo_lookup)
    };

    let (lk2, dlk2_dr, dlk2_dt) =
        get_co_kap_for_log_rho_log_t(xt, i2, log_rho, log_t)?;
    let (dxc2_l, dxo2_l) = {
        let s = &xt.co_tables[(i2 as usize) - 1];
        (s.dxc_lookup, s.dxo_lookup)
    };

    let (lk3, dlk3_dr, dlk3_dt) =
        get_co_kap_for_log_rho_log_t(xt, i3, log_rho, log_t)?;
    let (dxc3_l, dxo3_l) = {
        let s = &xt.co_tables[(i3 as usize) - 1];
        (s.dxc_lookup, s.dxo_lookup)
    };

    let (lk4, dlk4_dr, dlk4_dt, dxc4_l, dxo4_l) = if i4 > 0 {
        let (v, dr, dt) = get_co_kap_for_log_rho_log_t(xt, i4, log_rho, log_t)?;
        let s = &xt.co_tables[(i4 as usize) - 1];
        (v, dr, dt, s.dxc_lookup, s.dxo_lookup)
    } else {
        (lk3, dlk3_dr, dlk3_dt, dxc3_l, dxo3_l)
    };

    let (lk_2_4, dlk_2_4_dr, dlk_2_4_dt, lk_1_3, dlk_1_3_dr, dlk_1_3_dt, alfa) =
        if dxc >= dxo {
            let (v24, dr24, dt24, dxc24_l) = get_kap_at_dxo(
                dxo_lookup, dxc2_l, dxo2_l, lk2, dlk2_dr, dlk2_dt, dxc4_l,
                dxo4_l, lk4, dlk4_dr, dlk4_dt,
            );
            let (v13, dr13, dt13, dxc13_l) = get_kap_at_dxo(
                dxo_lookup, dxc1_l, dxo1_l, lk1, dlk1_dr, dlk1_dt, dxc3_l,
                dxo3_l, lk3, dlk3_dr, dlk3_dt,
            );
            let a = if (dxc13_l - dxc24_l).abs() == 0.0 {
                0.0
            } else {
                (dxc_lookup - dxc24_l) / (dxc13_l - dxc24_l)
            };
            (v24, dr24, dt24, v13, dr13, dt13, a)
        } else {
            let (v24, dr24, dt24, dxo24_l) = get_kap_at_dxc(
                dxc_lookup, dxc2_l, dxo2_l, lk2, dlk2_dr, dlk2_dt, dxc4_l,
                dxo4_l, lk4, dlk4_dr, dlk4_dt,
            );
            let (v13, dr13, dt13, dxo13_l) = get_kap_at_dxc(
                dxc_lookup, dxc1_l, dxo1_l, lk1, dlk1_dr, dlk1_dt, dxc3_l,
                dxo3_l, lk3, dlk3_dr, dlk3_dt,
            );
            let a = if (dxo13_l - dxo24_l).abs() == 0.0 {
                0.0
            } else {
                (dxo_lookup - dxo24_l) / (dxo13_l - dxo24_l)
            };
            (v24, dr24, dt24, v13, dr13, dt13, a)
        };

    let beta = 1.0 - alfa;
    Ok((
        alfa * lk_1_3 + beta * lk_2_4,
        alfa * dlk_1_3_dr + beta * dlk_2_4_dr,
        alfa * dlk_1_3_dt + beta * dlk_2_4_dt,
    ))
}
