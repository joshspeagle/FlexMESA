// SPDX-License-Identifier: LGPL-3.0-or-later
//! Convex-hull $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ sub-table lookup and
//! $(\log_{10} R, \log_{10} T)$ evaluation (`Find_CO_Tables` and
//! `Get_CO_Kap_for_logRho_logT` in `kap/private/kap_eval_co.f90`).

use crate::loader::{KapCoXTable, KAP_CO_DXS};
use crate::spline::{do_kap_interpolations, locate_log};
use crate::types::KapError;

fn lerp_co_pair(
    target_l: f64,
    pri_a_l: f64,
    sec_a_l: f64,
    lk_a: f64,
    dlk_a_dr: f64,
    dlk_a_dt: f64,
    pri_b_l: f64,
    sec_b_l: f64,
    lk_b: f64,
    dlk_b_dr: f64,
    dlk_b_dt: f64,
) -> (f64, f64, f64, f64) {
    let alfa = if (pri_a_l - pri_b_l).abs() == 0.0 {
        0.0
    } else {
        (target_l - pri_b_l) / (pri_a_l - pri_b_l)
    };
    let sec_ab_l = sec_b_l + (sec_a_l - sec_b_l) * alfa;
    let beta = 1.0 - alfa;
    (
        alfa * lk_a + beta * lk_b,
        alfa * dlk_a_dr + beta * dlk_b_dr,
        alfa * dlk_a_dt + beta * dlk_b_dt,
        sec_ab_l,
    )
}

/// Linearly interpolate between two $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ tables
/// to target $\delta X_{\mathrm{O}}$ lookup coordinate (`Get_Kap_at_dXO` in `kap_eval_co.f90`).
pub(crate) fn get_kap_at_dxo(
    dxo_l: f64,
    dxc_a_l: f64,
    dxo_a_l: f64,
    lk_a: f64,
    dlk_a_dr: f64,
    dlk_a_dt: f64,
    dxc_b_l: f64,
    dxo_b_l: f64,
    lk_b: f64,
    dlk_b_dr: f64,
    dlk_b_dt: f64,
) -> (f64, f64, f64, f64) {
    lerp_co_pair(
        dxo_l, dxo_a_l, dxc_a_l, lk_a, dlk_a_dr, dlk_a_dt, dxo_b_l, dxc_b_l,
        lk_b, dlk_b_dr, dlk_b_dt,
    )
}

/// Linearly interpolate between two $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ tables
/// to target $\delta X_{\mathrm{C}}$ lookup coordinate (`Get_Kap_at_dXC` in `kap_eval_co.f90`).
pub(crate) fn get_kap_at_dxc(
    dxc_l: f64,
    dxc_a_l: f64,
    dxo_a_l: f64,
    lk_a: f64,
    dlk_a_dr: f64,
    dlk_a_dt: f64,
    dxc_b_l: f64,
    dxo_b_l: f64,
    lk_b: f64,
    dlk_b_dr: f64,
    dlk_b_dt: f64,
) -> (f64, f64, f64, f64) {
    lerp_co_pair(
        dxc_l, dxc_a_l, dxo_a_l, lk_a, dlk_a_dr, dlk_a_dt, dxc_b_l, dxo_b_l,
        lk_b, dlk_b_dr, dlk_b_dt,
    )
}

/// Locate up to 4 bounding $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ sub-tables
/// (`Find_CO_Tables` in `kap/private/kap_eval_co.f90`).
pub(crate) fn find_co_tables(
    xt: &KapCoXTable,
    dxco_max: f64,
    dxc: f64,
    dxo: f64,
    dxc_lookup: f64,
    dxo_lookup: f64,
) -> Result<(i32, i32, i32, i32), KapError> {
    const TINY: f64 = 1.0e-7;

    let mut idxc = 7_usize;
    for (idx, &val) in KAP_CO_DXS.iter().enumerate().skip(1) {
        if val >= dxc {
            idxc = idx;
            break;
        }
    }
    let mut idxo = 7_usize;
    for (idx, &val) in KAP_CO_DXS.iter().enumerate().skip(1) {
        if val >= dxo {
            idxo = idx;
            break;
        }
    }

    let matches_table = |i_1based: i32| -> bool {
        if i_1based < 1 {
            return false;
        }
        let sub = &xt.co_tables[(i_1based as usize) - 1];
        (dxc_lookup - sub.dxc_lookup).abs() < TINY
            && (dxo_lookup - sub.dxo_lookup).abs() < TINY
    };
    let on_midline = |i_1based: i32| -> bool {
        let sub = &xt.co_tables[(i_1based as usize) - 1];
        (sub.dxc - sub.dxo).abs() < TINY
    };
    let on_diagonal = |i_1based: i32| -> bool {
        let sub = &xt.co_tables[(i_1based as usize) - 1];
        ((sub.dxc + sub.dxo) - dxco_max).abs() < TINY
    };

    let mut i1 = xt.co_table_numbers[idxc - 1][idxo - 1];
    if matches_table(i1) {
        return Ok((i1, -1, -1, -1));
    }

    let (mut i2, mut i3) = if dxc >= dxo {
        (
            xt.co_table_numbers[idxc][idxo - 1],
            xt.co_table_numbers[idxc - 1][idxo],
        )
    } else {
        (
            xt.co_table_numbers[idxc - 1][idxo],
            xt.co_table_numbers[idxc][idxo - 1],
        )
    };
    let mut i4 = xt.co_table_numbers[idxc][idxo];

    if i4 > 0 {
        if i1 <= 0 || i2 <= 0 || i3 <= 0 {
            return Err(KapError::NumericalFailure(
                "logical failure in Find_CO_Tables",
            ));
        }
        if matches_table(i2) {
            return Ok((i2, -1, -1, -1));
        }
        if matches_table(i3) {
            return Ok((i3, -1, -1, -1));
        }
        if matches_table(i4) {
            return Ok((i4, -1, -1, -1));
        }
        return Ok((i1, i2, i3, i4));
    }

    if on_midline(i1) {
        if dxc >= dxo {
            i2 = xt.next_dxc_table[(i1 as usize) - 1];
            i3 = xt.next_dxo_table[(i1 as usize) - 1];
        } else {
            i2 = xt.next_dxo_table[(i1 as usize) - 1];
            i3 = xt.next_dxc_table[(i1 as usize) - 1];
        }
        return Ok((i1, i2, i3, i4));
    }

    let num_co_i32 = xt.num_co_tables as i32;
    let num_gt_i32 = xt.num_dxc_gt_dxo as i32;

    if dxc >= dxo {
        if i3 <= 0 {
            if on_diagonal(i1) {
                i3 = num_co_i32;
                if i2 > 0 {
                    return Ok((i1, -1, -1, -1));
                }
            } else {
                i3 = num_gt_i32;
            }
        }
        if !on_diagonal(i3) {
            i4 = xt.next_dxc_table[(i3 as usize) - 1];
            if !on_diagonal(i4) {
                return Ok((i1, -1, -1, -1));
            }
        }
        if i2 <= 0 {
            i2 = xt.next_dxc_table[(i1 as usize) - 1];
        }
        if on_diagonal(i2) {
            return Ok((i1, i2, i3, i4));
        }
        let s4 = &xt.co_tables[(i4 as usize) - 1];
        if dxc_lookup <= s4.dxc_lookup {
            return Ok((i1, i2, i3, i4));
        }
        let s2 = &xt.co_tables[(i2 as usize) - 1];
        if dxc_lookup
            <= s4.dxc_lookup
                + (s2.dxc_lookup - s4.dxc_lookup) * (dxo_lookup - s4.dxo_lookup)
                    / (s2.dxo_lookup - s4.dxo_lookup)
        {
            return Ok((i1, i2, i3, i4));
        }
        i1 = i2;
        i2 = xt.next_dxc_table[(i1 as usize) - 1];
        if !on_diagonal(i2) {
            return Ok((i1, -1, -1, -1));
        }
        i3 = i4;
        i4 = -1;
    } else {
        if i3 <= 0 {
            if on_diagonal(i1) {
                i3 = num_gt_i32;
            } else {
                i3 = num_co_i32;
                if i2 > 0 {
                    return Ok((i1, -1, -1, -1));
                }
            }
        }
        if !on_diagonal(i3) {
            i4 = xt.next_dxo_table[(i3 as usize) - 1];
            if !on_diagonal(i4) {
                return Ok((i1, -1, -1, -1));
            }
        }
        if i2 <= 0 {
            i2 = xt.next_dxo_table[(i1 as usize) - 1];
        }
        if on_diagonal(i2) {
            return Ok((i1, i2, i3, i4));
        }
        let s4 = &xt.co_tables[(i4 as usize) - 1];
        if dxo_lookup <= s4.dxo_lookup {
            return Ok((i1, i2, i3, i4));
        }
        let s2 = &xt.co_tables[(i2 as usize) - 1];
        if dxo_lookup
            <= s4.dxo_lookup
                + (s2.dxo_lookup - s4.dxo_lookup) * (dxc_lookup - s4.dxc_lookup)
                    / (s2.dxc_lookup - s4.dxc_lookup)
        {
            return Ok((i1, i2, i3, i4));
        }
        i1 = i2;
        i2 = xt.next_dxo_table[(i1 as usize) - 1];
        if !on_diagonal(i2) {
            return Ok((i1, -1, -1, -1));
        }
        i3 = num_co_i32;
        i4 = -1;
    }

    Ok((i1, i2, i3, i4))
}

/// Evaluate a single 1-based $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ sub-table at
/// $(\log_{10} \rho, \log_{10} T)$ (`Get_CO_Kap_for_logRho_logT` in `kap/private/kap_eval_co.f90`).
pub(crate) fn get_co_kap_for_log_rho_log_t(
    xt: &KapCoXTable,
    ico_1based: i32,
    log_rho: f64,
    log_t_in: f64,
) -> Result<(f64, f64, f64), KapError> {
    let log_r_in = log_rho - 3.0 * log_t_in + 18.0;
    if log_t_in > xt.log_t_max
        || log_t_in < xt.log_t_min
        || log_r_in < xt.log_r_min
    {
        return Err(KapError::OutOfBounds(
            "coordinate outside CO table boundaries",
        ));
    }
    let (mut log_r, clipped_log_r) = if log_r_in > xt.log_r_max {
        (xt.log_r_max, true)
    } else {
        (log_r_in, false)
    };
    let mut log_t = log_t_in;

    let mut target_idx = (ico_1based as usize) - 1;
    if xt.co_tables[target_idx].raw_i16.is_none() {
        let mut best = 0_usize;
        let mut best_dist = f64::INFINITY;
        for (idx, sub) in xt.co_tables.iter().enumerate() {
            if sub.raw_i16.is_some() {
                let d = (sub.dxc - xt.co_tables[target_idx].dxc)
                    .hypot(sub.dxo - xt.co_tables[target_idx].dxo);
                if d < best_dist {
                    best_dist = d;
                    best = idx;
                }
            }
        }
        target_idx = best;
    }

    let (kap1, ili_log_rs, ili_log_ts) =
        xt.co_tables[target_idx].spline_data(&xt.log_rs, &xt.log_ts)?;
    let (i_r, lr_loc, log_r0, log_r1) = locate_log(
        xt.log_r_min,
        xt.log_r_max,
        *ili_log_rs,
        &xt.log_rs,
        log_r,
        true,
    )?;
    log_r = lr_loc;
    let (j_temp, lt_loc, log_t0, log_t1) = locate_log(
        xt.log_t_min,
        xt.log_t_max,
        *ili_log_ts,
        &xt.log_ts,
        log_t,
        true,
    )?;
    log_t = lt_loc;

    let (log_k, mut df_dx, df_dy) = do_kap_interpolations(
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
    let dlog_k_dlog_rho = df_dx;
    let dlog_k_dlog_t = df_dy - 3.0 * df_dx;
    Ok((log_k, dlog_k_dlog_rho, dlog_k_dlog_t))
}
