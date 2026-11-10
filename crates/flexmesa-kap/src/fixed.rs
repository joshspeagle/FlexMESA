//! Fixed-metal Type 1 and Ferguson/Freedman low-$T$ opacity evaluation (`kap_eval_fixed.f90`).
//!
//! Mirrors `Get1_kap_fixed_metal_results`, `Get_Kap_for_X_Z_cubic`, `Get_Kap_for_X_Z_linear`,
//! `Get_Kap_for_X`, `Get_Kap_for_X_cubic`, `Get_Kap_for_X_linear`, and
//! `Get_Kap_for_logRho_logT` in `kap/private/kap_eval_fixed.f90`.

use crate::loader::{kap_tables, KapXTable, KapZTable};
use crate::spline::{do_kap_interpolations, locate_log};
use crate::types::{KapConfig, KapError, KapLowTOption};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_interp1d::interpolate_vector_pm_autodiff;

/// Evaluate a fixed-metal or Ferguson/Freedman low-$T$ table set at `(Z, X, logRho, logT)`
/// (`Get1_kap_fixed_metal_results` in `kap_eval_fixed.f90`).
///
/// Returns `(log10_kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if the table set is missing or coordinates lie outside bounds.
pub fn get1_kap_fixed_metal_results(
    cfg: &KapConfig,
    prefix: &str,
    z: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let tables = kap_tables();
    let canonical =
        KapLowTOption::from_prefix(prefix).map_or(prefix, |o| o.prefix());
    let set = tables
        .fixed_sets
        .iter()
        .find(|s| s.prefix == prefix || s.prefix == canonical)
        .ok_or(KapError::MissingTable("unknown fixed-metal table prefix"))?;
    let z_tables = &set.z_tables;
    let num_zs = z_tables.len();
    if num_zs == 0 {
        return Err(KapError::MissingTable("empty z_tables in fixed set"));
    }

    if num_zs == 1 || z >= z_tables[num_zs - 1].z {
        return get_kap_for_x(cfg, &z_tables[num_zs - 1], x, log_rho, log_t);
    }
    if z <= z_tables[0].z {
        return get_kap_for_x(cfg, &z_tables[0], x, log_rho, log_t);
    }

    let mut iz = 0_usize;
    for idx in 0..(num_zs - 1) {
        if z < z_tables[idx + 1].z {
            iz = idx;
            break;
        }
    }
    let z0 = z_tables[iz].z;
    let z1 = z_tables[iz + 1].z;

    if z <= z0 {
        return get_kap_for_x(cfg, &z_tables[iz], x, log_rho, log_t);
    }
    if z >= z1 {
        return get_kap_for_x(cfg, &z_tables[iz + 1], x, log_rho, log_t);
    }

    if num_zs >= 4 && cfg.cubic_interpolation_in_z {
        get_kap_for_x_z_cubic(cfg, z_tables, iz, z, x, log_rho, log_t)
    } else {
        get_kap_for_x_z_linear(cfg, z_tables, iz, z, z0, z1, x, log_rho, log_t)
    }
}

fn get_kap_for_x_z_cubic(
    cfg: &KapConfig,
    z_tables: &[KapZTable],
    iz: usize,
    z: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let num_zs = z_tables.len();
    let i1 = if iz + 2 >= num_zs {
        num_zs - 3
    } else if iz == 0 {
        1
    } else {
        iz
    };

    let indices = [i1 - 1, i1, i1 + 1, i1 + 2];
    let mut z_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
    let mut v_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
    for (k, &izz) in indices.iter().enumerate() {
        z_old[k] = AutoDiffReal2VarOrder1::from(z_tables[izz].z);
        let (lk, dlk_dr, dlk_dt) =
            get_kap_for_x(cfg, &z_tables[izz], x, log_rho, log_t)?;
        v_old[k] = AutoDiffReal2VarOrder1 {
            val: lk,
            d1: [dlk_dt, dlk_dr],
        };
    }
    let z_new = [AutoDiffReal2VarOrder1::from(z)];
    let v_new = interpolate_vector_pm_autodiff(&z_old, &v_old, &z_new)
        .map_err(|_| {
            KapError::NumericalFailure("interp_pm_autodiff failed in Z")
        })?;
    Ok((v_new[0].val, v_new[0].d1[1], v_new[0].d1[0]))
}

#[allow(clippy::too_many_arguments)]
fn get_kap_for_x_z_linear(
    cfg: &KapConfig,
    z_tables: &[KapZTable],
    iz: usize,
    z: f64,
    z0: f64,
    z1: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let (lk0, dlk0_dr, dlk0_dt) =
        get_kap_for_x(cfg, &z_tables[iz], x, log_rho, log_t)?;
    let (lk1, dlk1_dr, dlk1_dt) =
        get_kap_for_x(cfg, &z_tables[iz + 1], x, log_rho, log_t)?;
    let beta = (z - z1) / (z0 - z1);
    let alfa = 1.0 - beta;
    Ok((
        beta * lk0 + alfa * lk1,
        beta * dlk0_dr + alfa * dlk1_dr,
        beta * dlk0_dt + alfa * dlk1_dt,
    ))
}

fn get_kap_for_x(
    cfg: &KapConfig,
    zt: &KapZTable,
    x: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    if !(0.0..=1.0).contains(&x) {
        return Err(KapError::InvalidComposition("X outside [0, 1]"));
    }
    let x_tables = &zt.x_tables;
    let num_xs = x_tables.len();
    if num_xs == 1 || x >= x_tables[num_xs - 1].x {
        return get_kap_for_log_rho_log_t(
            &x_tables[num_xs - 1],
            log_rho,
            log_t,
        );
    }
    if x <= x_tables[0].x {
        return get_kap_for_log_rho_log_t(&x_tables[0], log_rho, log_t);
    }

    let mut ix = num_xs - 1;
    for i in 0..(num_xs - 1) {
        if x < x_tables[i + 1].x {
            ix = i;
            break;
        }
    }
    if ix == num_xs - 1 {
        return get_kap_for_log_rho_log_t(
            &x_tables[num_xs - 1],
            log_rho,
            log_t,
        );
    }

    let x0 = x_tables[ix].x;
    let x1 = x_tables[ix + 1].x;
    if x0 >= x {
        return get_kap_for_log_rho_log_t(&x_tables[ix], log_rho, log_t);
    }
    if x1 <= x {
        return get_kap_for_log_rho_log_t(&x_tables[ix + 1], log_rho, log_t);
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
        let mut x_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
        let mut v_old = [AutoDiffReal2VarOrder1::from(0.0); 4];
        for (k, &ixx) in indices.iter().enumerate() {
            x_old[k] = AutoDiffReal2VarOrder1::from(x_tables[ixx].x);
            let (lk, dlk_dr, dlk_dt) =
                get_kap_for_log_rho_log_t(&x_tables[ixx], log_rho, log_t)?;
            v_old[k] = AutoDiffReal2VarOrder1 {
                val: lk,
                d1: [dlk_dt, dlk_dr],
            };
        }
        let x_new = [AutoDiffReal2VarOrder1::from(x)];
        let v_new = interpolate_vector_pm_autodiff(&x_old, &v_old, &x_new)
            .map_err(|_| {
                KapError::NumericalFailure("interp_pm_autodiff failed in X")
            })?;
        return Ok((v_new[0].val, v_new[0].d1[1], v_new[0].d1[0]));
    }

    let (lk0, dlk0_dr, dlk0_dt) =
        get_kap_for_log_rho_log_t(&x_tables[ix], log_rho, log_t)?;
    let (lk1, dlk1_dr, dlk1_dt) =
        get_kap_for_log_rho_log_t(&x_tables[ix + 1], log_rho, log_t)?;
    let beta = (x - x1) / (x0 - x1);
    let alfa = 1.0 - beta;
    Ok((
        beta * lk0 + alfa * lk1,
        beta * dlk0_dr + alfa * dlk1_dr,
        beta * dlk0_dt + alfa * dlk1_dt,
    ))
}

fn get_kap_for_log_rho_log_t(
    xt: &KapXTable,
    log_rho: f64,
    log_t_in: f64,
) -> Result<(f64, f64, f64), KapError> {
    let log_r_in = log_rho - 3.0 * log_t_in + 18.0;
    if log_t_in > xt.log_t_max {
        return Err(KapError::OutOfBounds(
            "logT > logT_max in fixed kap table",
        ));
    }
    let (mut log_t, clipped_log_t) = if log_t_in < xt.log_t_min {
        (xt.log_t_min, true)
    } else {
        (log_t_in, false)
    };
    let (mut log_r, clipped_log_r) = if log_r_in < xt.log_r_min {
        (xt.log_r_min, true)
    } else if log_r_in > xt.log_r_max {
        (xt.log_r_max, true)
    } else {
        (log_r_in, false)
    };

    let (kap1, ili_log_rs, ili_log_ts) = xt.spline_data()?;
    let (i_r, log_r_loc, log_r0, log_r1) = locate_log(
        xt.log_r_min,
        xt.log_r_max,
        *ili_log_rs,
        &xt.log_rs,
        log_r,
        true,
    )?;
    log_r = log_r_loc;
    let (j_temp, log_t_loc, log_t0, log_t1) = locate_log(
        xt.log_t_min,
        xt.log_t_max,
        *ili_log_ts,
        &xt.log_ts,
        log_t,
        true,
    )?;
    log_t = log_t_loc;

    let (log_k, mut df_dx, mut df_dy) = do_kap_interpolations(
        kap1,
        xt.log_rs.len(),
        xt.log_ts.len(),
        i_r,
        j_temp,
        log_r0,
        log_r,
        log_r1,
        log_t0,
        log_t,
        log_t1,
    );
    if clipped_log_r {
        df_dx = 0.0;
    }
    if clipped_log_t {
        df_dy = 0.0;
    }
    let dlog_k_dlog_rho = df_dx;
    let dlog_k_dlog_t = df_dy - 3.0 * df_dx;
    Ok((log_k, dlog_k_dlog_rho, dlog_k_dlog_t))
}
