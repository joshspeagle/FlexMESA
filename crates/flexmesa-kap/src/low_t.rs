//! Low-temperature molecular/dust opacity evaluators (`kap_aesopus.f90` and `kapcn.f90`).
//!
//! Implements Marigo & Aringer (2009) `AESOPUS` (`AESOPUS_get`) and
//! Lederer & Aringer (2009) `kapCN` (`kapCN_get`).

use crate::loader::{kap_tables, AesopusData, AesopusZSet, KapCnData};
use crate::types::KapError;
use flexmesa_interp1d::interpolate_vector_pm;
use flexmesa_interp2d::interp_evbicub_prevalidated_db;
use flexmesa_math::{exp10, safe_log10};
use flexmesa_num::binary_search;

/// Evaluate `AESOPUS` low-$T$ opacity at `(Zref, X, XC, XN, XO, logRho, logT)`
/// (`AESOPUS_get` in `kap/private/kap_aesopus.f90`).
///
/// Returns `(kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if `(logR, logT)` lies outside the AESOPUS table bounds.
pub fn aesopus_get(
    zref: f64,
    x: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let ka = &kap_tables().aesopus;
    let log_r = log_rho - 3.0 * log_t + 18.0;
    let res = aesopus_interp(ka, zref, x, xc, xn, xo, log_r, log_t)?;
    let kap = exp10(res[0]);
    let dlnkap_dln_rho = res[1];
    let dlnkap_dln_t = res[2] - 3.0 * res[1];
    Ok((kap, dlnkap_dln_rho, dlnkap_dln_t))
}

#[allow(clippy::too_many_arguments)]
fn aesopus_interp(
    ka: &AesopusData,
    zref: f64,
    x: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    log_r: f64,
    log_t: f64,
) -> Result<[f64; 3], KapError> {
    let min_lr = ka.log_rs[0];
    let max_lr = ka.log_rs[ka.log_rs.len() - 1];
    let min_lt = ka.log_ts[0];
    let max_lt = ka.log_ts[ka.log_ts.len() - 1];
    if log_r < min_lr || log_r > max_lr || log_t < min_lt || log_t > max_lt {
        return Err(KapError::OutOfBounds(
            "AESOPUS_interp: logR, logT outside table bounds",
        ));
    }

    let num_zs = ka.zs.len();
    if zref <= ka.zs[0] {
        return aesopus_interp_fixed_zref(ka, 0, x, xc, xn, xo, log_r, log_t);
    }
    if zref >= ka.zs[num_zs - 1] {
        return aesopus_interp_fixed_zref(
            ka,
            num_zs - 1,
            x,
            xc,
            xn,
            xo,
            log_r,
            log_t,
        );
    }

    // 1-based binary_search in Fortran returns iZ in 1..=num_Zs-1; in 0-based: 0..=num_zs-2
    let iz = binary_search(&ka.zs, zref, None)
        .unwrap_or(0)
        .clamp(1, num_zs - 2);

    let mut raw_res = [[0.0_f64; 3]; 3];
    for (k, offset) in [iz - 1, iz, iz + 1].iter().enumerate() {
        raw_res[k] = aesopus_interp_fixed_zref(
            ka, *offset, x, xc, xn, xo, log_r, log_t,
        )?;
    }

    let x_old = &ka.log_zs[(iz - 1)..=(iz + 1)];
    let x_new = [safe_log10(zref)];
    let mut res = [0.0_f64; 3];
    for c in 0..3 {
        let v_old = [raw_res[0][c], raw_res[1][c], raw_res[2][c]];
        let v_new =
            interpolate_vector_pm(x_old, &v_old, &x_new).map_err(|_| {
                KapError::NumericalFailure("AESOPUS Z interp_pm failed")
            })?;
        res[c] = v_new[0];
    }
    Ok(res)
}

#[allow(clippy::too_many_arguments)]
fn aesopus_interp_fixed_zref(
    ka: &AesopusData,
    mut iz: usize,
    x: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    log_r: f64,
    log_t: f64,
) -> Result<[f64; 3], KapError> {
    if ka.ts[iz].slices.is_empty() {
        let mut best = iz;
        let mut best_d = f64::INFINITY;
        for (idx, ts_i) in ka.ts.iter().enumerate() {
            if !ts_i.slices.is_empty() {
                let d = (ka.zs[idx] - ka.zs[iz]).abs();
                if d < best_d {
                    best_d = d;
                    best = idx;
                }
            }
        }
        iz = best;
    }

    let zref = ka.zs[iz];
    let fco = safe_log10(xc / xo) - ka.fco_ref;
    let fc = safe_log10(xc / zref) - ka.fc_ref;
    let fn_val = safe_log10(xn / zref) - ka.fn_ref;

    let ts = &ka.ts[iz];
    let (ix, wx) = clipped_linear_weights(x, &ts.xs);
    let (ifco, wfco) = clipped_linear_weights(fco, &ts.fcos);
    let (ifc, wfc) = clipped_linear_weights(fc, &ts.fcs);
    let (ifn, wfn) = clipped_linear_weights(fn_val, &ts.fns);

    let mut res = [0.0_f64; 3];
    for corner in 0..16 {
        let (i, j, k, l) = (
            (corner >> 3) & 1,
            (corner >> 2) & 1,
            (corner >> 1) & 1,
            corner & 1,
        );
        let w = wx[i] * wfco[j] * wfc[k] * wfn[l];
        if w == 0.0 {
            continue;
        }
        let raw = aesopus_interp_rt(
            ka,
            ts,
            ix + i,
            ifco + j,
            ifc + k,
            ifn + l,
            log_r,
            log_t,
        )?;
        res[0] += w * raw[0];
        res[1] += w * raw[1];
        res[2] += w * raw[2];
    }
    Ok(res)
}

fn clipped_linear_weights(val: f64, vec: &[f64]) -> (usize, [f64; 2]) {
    let len = vec.len();
    if val <= vec[0] {
        return (0, [1.0, 0.0]);
    }
    if val >= vec[len - 1] {
        return (len - 1, [1.0, 0.0]);
    }
    let loc = binary_search(vec, val, Some(len / 2))
        .unwrap_or(0)
        .min(len - 2);
    let w2 = (val - vec[loc]) / (vec[loc + 1] - vec[loc]);
    let w1 = 1.0 - w2;
    (loc, [w1, w2])
}

#[allow(clippy::too_many_arguments)]
fn aesopus_interp_rt(
    ka: &AesopusData,
    ts: &AesopusZSet,
    ix: usize,
    ico: usize,
    ic: usize,
    in_: usize,
    log_r: f64,
    log_t: f64,
) -> Result<[f64; 3], KapError> {
    let n_fco = ts.fcos.len();
    let n_fc = ts.fcs.len();
    let n_fn = ts.fns.len();
    let flat_idx = ((ix * n_fco + ico) * n_fc + ic) * n_fn + in_;
    let slice = if ts.slices[flat_idx].raw_f32.is_some() {
        &ts.slices[flat_idx]
    } else {
        ts.slices
            .iter()
            .find(|s| s.raw_f32.is_some())
            .ok_or(KapError::MissingTable("no AESOPUS slice available"))?
    };
    let (kap1, ilinx, iliny) = slice.spline_data(&ka.log_rs, &ka.log_ts)?;
    let ev = interp_evbicub_prevalidated_db(
        log_r, log_t, &ka.log_rs, &ka.log_ts, *ilinx, *iliny, kap1,
    )
    .map_err(|_| {
        KapError::NumericalFailure("interp_evbicub_db failed in AESOPUS")
    })?;
    Ok([ev.f, ev.df_dx, ev.df_dy])
}

/// Evaluate `kapCN` low-$T$ opacity at `(Z, X, fC, fN, logRho, logT)`
/// (`kapCN_get` in `kap/private/kapcn.f90`).
///
/// Returns `(kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if `(logR, logT)` lies outside the `kapCN` table bounds.
pub fn kapcn_get(
    z: f64,
    x: f64,
    fc: f64,
    fn_val: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    let kcn = &kap_tables().kap_cn;
    let log_r = log_rho - 3.0 * log_t + 18.0;
    let res = kapcn_interp(kcn, z, x, fc, fn_val, log_r, log_t)?;
    let kap = exp10(res[0]);
    let dlnkap_dln_rho = res[1];
    let dlnkap_dln_t = res[2] - 3.0 * res[1];
    Ok((kap, dlnkap_dln_rho, dlnkap_dln_t))
}

fn kapcn_interp(
    kcn: &KapCnData,
    z: f64,
    x: f64,
    fc: f64,
    fn_val: f64,
    log_r: f64,
    log_t: f64,
) -> Result<[f64; 3], KapError> {
    if log_r < kcn.log_rs[0]
        || log_r > kcn.log_rs[kcn.log_rs.len() - 1]
        || log_t < kcn.log_ts[0]
        || log_t > kcn.log_ts[kcn.log_ts.len() - 1]
    {
        return Err(KapError::OutOfBounds(
            "kapCN_interp: logR, logT outside table bounds",
        ));
    }
    let num_zs = kcn.zs.len();
    let my_z = z.clamp(kcn.zs[0], kcn.zs[num_zs - 1]);
    // In 1-based Fortran: iZ = max(2, min(iZ, num_kapCN_Zs - 2)); in 0-based: 1..=num_zs-3
    let iz = binary_search(&kcn.zs, z, None)
        .unwrap_or(0)
        .clamp(1, num_zs - 3);

    if (z - kcn.zs[iz]).abs() == 0.0 {
        return kapcn_interp_fixed_z(kcn, iz, x, fc, fn_val, log_r, log_t);
    }

    let mut res_z = [[0.0_f64; 3]; 4];
    let mut log_z_ary = [0.0_f64; 4];
    for (k, &izz) in [iz - 1, iz, iz + 1, iz + 2].iter().enumerate() {
        res_z[k] = kapcn_interp_fixed_z(kcn, izz, x, fc, fn_val, log_r, log_t)?;
        log_z_ary[k] = kcn.zs[izz].log10();
    }
    let x_new = [my_z.log10()];
    let mut result = [0.0_f64; 3];
    for c in 0..3 {
        let v_old = [res_z[0][c], res_z[1][c], res_z[2][c], res_z[3][c]];
        let v_new = interpolate_vector_pm(&log_z_ary, &v_old, &x_new).map_err(
            |_| KapError::NumericalFailure("kapCN Z interp_pm failed"),
        )?;
        result[c] = v_new[0];
    }
    Ok(result)
}

fn lagrange_3pt_weights(a: &[f64; 3], x: f64) -> [f64; 3] {
    let mut b = [1.0_f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            if j != i {
                b[i] *= (x - a[j]) / (a[i] - a[j]);
            }
        }
    }
    b
}

fn kapcn_interp_fixed_z(
    kcn: &KapCnData,
    iz: usize,
    x: f64,
    fc: f64,
    fn_val: f64,
    log_r: f64,
    log_t: f64,
) -> Result<[f64; 3], KapError> {
    let zset = &kcn.z_sets[iz];
    let my_x = x.clamp(kcn.xs[0], kcn.xs[2]);
    let my_fc = fc.clamp(zset.fcs[0], zset.fcs[6]);
    let my_fn = fn_val.clamp(zset.fns[0], zset.fns[2]);

    // In 1-based Fortran: ifC = max(2, min(ifC, 6)); in 0-based: 1..=5
    let ifc = binary_search(&zset.fcs, my_fc, None)
        .unwrap_or(0)
        .clamp(1, 5);

    let wx = lagrange_3pt_weights(&kcn.xs, my_x);
    let wfn = lagrange_3pt_weights(&zset.fns, my_fn);
    let fc_sub = [zset.fcs[ifc - 1], zset.fcs[ifc], zset.fcs[ifc + 1]];
    let wfc = lagrange_3pt_weights(&fc_sub, my_fc);

    let mut result = [0.0_f64; 3];
    for (i, &w_fn) in wfn.iter().enumerate() {
        for (j, &w_x) in wx.iter().enumerate() {
            for (n, &w_fc) in wfc.iter().enumerate() {
                // 0-based tbl index corresponding to Fortran 1-based: 21*(i-1) + 7*(j-1) + (ifC+n-2)
                let tbl_idx = 21 * i + 7 * j + (ifc + n - 1);
                let (kap1, ilinx, iliny) =
                    zset.spline_data(tbl_idx, &kcn.log_rs, &kcn.log_ts);
                let res = interp_evbicub_prevalidated_db(
                    log_r,
                    log_t,
                    &kcn.log_rs,
                    &kcn.log_ts,
                    *ilinx,
                    *iliny,
                    kap1,
                )
                .map_err(|_| {
                    KapError::NumericalFailure(
                        "interp_evbicub_db failed in kapCN",
                    )
                })?;
                let w = w_fn * w_x * w_fc;
                result[0] += w * res.f;
                result[1] += w * res.df_dx;
                result[2] += w * res.df_dy;
            }
        }
    }
    Ok(result)
}
