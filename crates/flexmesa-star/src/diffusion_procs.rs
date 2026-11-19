// SPDX-License-Identifier: LGPL-3.0-or-later
//! Mass-fraction fixup, conservative linear mass redistribution, limit-ramp
//! coefficients, structure face setup, and isotope mapping ported from MESA
//! `26.4.1` `star/private/diffusion_procs.f90`.

use flexmesa_const::{LN10, PI4};
use flexmesa_ionization::get_typical_charge;

use crate::diffusion_support::TINY_MASS;
use crate::job_controls::StarError;

/// Diagnostic details returned when [`fixup`] rejects a diffusion substep
/// (`diffusion_procs.f90:58-59`).
#[derive(Debug, Clone, PartialEq)]
pub struct FixupDiagnostics {
    /// Zero-based offending class index (if applicable).
    pub bad_j: Option<usize>,
    /// Zero-based offending zone index (if applicable).
    pub bad_k: Option<usize>,
    /// Offending mass fraction when `min_X_hard_limit` is breached.
    pub bad_x: f64,
    /// Offending normalized species-conservation error.
    pub bad_xsum: f64,
    /// Human-readable retry reason.
    pub reason: String,
}

/// Applies post-step mass-fraction floor validation, negative-mass neighbor
/// borrowing, global species conservation rescaling, single-point extremum
/// removal, slope-limited `dX/dm` reconstruction, and conservative mass
/// redistribution (`diffusion_procs.f90:42-145`).
///
/// Array layouts:
/// - `x`: flat `(nz, m)` (`x[k * m + j]`), modified in-place on `0..nc`.
/// - `mass`, `dx_dm`: flat `(nz, nc)` (`arr[k * nc + j]`).
/// - `sum_mass`, `cell_dm`, `ln_t`: length `nz`.
/// - `xtotal_init`: length `nc`.
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if any class breaches `min_x_hard_limit`,
/// fails global species conservation tolerance, or produces invalid cell masses
/// during redistribution (signalling the caller to shrink the diffusion
/// substep).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn fixup(
    nz: usize,
    nc: usize,
    m: usize,
    nzlo: usize,
    nzhi: usize,
    min_x_hard_limit: f64,
    x_total_atol: f64,
    x_total_rtol: f64,
    cell_dm: &[f64],
    mtotal: f64,
    xtotal_init: &[f64],
    x: &mut [f64],
    ln_t: &[f64],
    sum_mass: &mut [f64],
    mass: &mut [f64],
    dx_dm: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    let nsmooth_x_in_fixup = 0_usize;
    let max_lnt_for_smooth = LN10 * 99.0;

    for k in nzlo..=nzhi {
        for j in 0..nc {
            let xjk = x[k * m + j];
            if xjk < min_x_hard_limit || !xjk.is_finite() {
                return Err(FixupDiagnostics {
                    bad_j: Some(j),
                    bad_k: Some(k),
                    bad_x: xjk,
                    bad_xsum: 0.0,
                    reason: format!(
                        "X({j},{k})={xjk:e} below min_X_hard_limit={min_x_hard_limit:e}"
                    ),
                });
            }
            mass[k * nc + j] = cell_dm[k] * xjk;
        }
    }

    fix_negative_masses(nc, nzlo, nzhi, cell_dm, mass)?;

    fix_species_conservation(
        nc,
        nzlo,
        nzhi,
        mtotal,
        xtotal_init,
        x_total_atol,
        x_total_rtol,
        mass,
        sum_mass,
    )?;

    fix_single_point_extremes(
        nc,
        nzlo,
        nzhi,
        max_lnt_for_smooth,
        ln_t,
        sum_mass,
        mass,
    )?;

    if nsmooth_x_in_fixup > 0 {
        smooth_x(
            nc,
            nzlo,
            nzhi,
            nsmooth_x_in_fixup,
            max_lnt_for_smooth,
            ln_t,
            sum_mass,
            mass,
        )?;
    }

    for k in nzlo..=nzhi {
        get1_dx_dm(k, nc, nzlo, nzhi, mass, sum_mass, dx_dm)?;
    }

    redistribute_mass(
        nc,
        m,
        nzlo,
        nzhi,
        nz,
        sum_mass,
        mass,
        dx_dm,
        xtotal_init,
        cell_dm,
        x,
    )?;

    Ok(())
}

/// Removes single-point spatial extrema in class mass fractions by exchanging
/// mass conservatively with the closer neighbor (`diffusion_procs.f90:148-227`).
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if any cell sum becomes negative or smaller
/// than a component mass.
pub fn fix_single_point_extremes(
    nc: usize,
    nzlo: usize,
    nzhi: usize,
    max_lnt_for_smooth: f64,
    ln_t: &[f64],
    sum_mass: &mut [f64],
    mass: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    if nzhi >= nzlo + 2 {
        for k in (nzlo + 1)..nzhi {
            if ln_t[k] > max_lnt_for_smooth {
                break;
            }
            for j in 0..nc {
                let xm1 = mass[(k - 1) * nc + j] / sum_mass[k - 1];
                let x00 = mass[k * nc + j] / sum_mass[k];
                let xp1 = mass[(k + 1) * nc + j] / sum_mass[k + 1];
                if (xm1 - x00) * (x00 - xp1) < -1.0e-24 {
                    let k1 = if (xm1 - x00).abs() < (x00 - xp1).abs() {
                        k - 1
                    } else {
                        k + 1
                    };
                    let m0 = mass[k * nc + j];
                    let sum0 = sum_mass[k];
                    let m1 = mass[k1 * nc + j];
                    let sum1 = sum_mass[k1];
                    let mut dm =
                        (m0 * sum1 - m1 * sum0) / (m0 + m1 - sum0 - sum1);
                    if dm > 0.0 {
                        dm = dm.min(0.99999 * sum1);
                    } else {
                        dm = -(-dm).min(0.99999 * sum0);
                    }
                    mass[k * nc + j] = m0 + dm;
                    let mut sk = 0.0;
                    for jj in 0..nc {
                        sk += mass[k * nc + jj];
                    }
                    sum_mass[k] = sk;
                    if sum_mass[k] < 0.0 || !sum_mass[k].is_finite() {
                        return Err(FixupDiagnostics {
                            bad_j: Some(j),
                            bad_k: Some(k),
                            bad_x: 0.0,
                            bad_xsum: 0.0,
                            reason: format!(
                                "negative sum_mass({k})={} in fix_single_point_extremes",
                                sum_mass[k]
                            ),
                        });
                    }
                    mass[k1 * nc + j] = m1 - dm;
                    let mut sk1 = 0.0;
                    for jj in 0..nc {
                        sk1 += mass[k1 * nc + jj];
                    }
                    sum_mass[k1] = sk1;
                    if sum_mass[k1] < 0.0 || !sum_mass[k1].is_finite() {
                        return Err(FixupDiagnostics {
                            bad_j: Some(j),
                            bad_k: Some(k1),
                            bad_x: 0.0,
                            bad_xsum: 0.0,
                            reason: format!(
                                "negative sum_mass({k1})={} in fix_single_point_extremes",
                                sum_mass[k1]
                            ),
                        });
                    }
                }
            }
        }
    }

    for k in nzlo..=nzhi {
        if sum_mass[k] < 0.0 || !sum_mass[k].is_finite() {
            return Err(FixupDiagnostics {
                bad_j: None,
                bad_k: Some(k),
                bad_x: 0.0,
                bad_xsum: 0.0,
                reason: format!("negative sum_mass({k}) in fixup 3a"),
            });
        }
        for j in 0..nc {
            if mass[k * nc + j] > sum_mass[k] {
                return Err(FixupDiagnostics {
                    bad_j: Some(j),
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: 0.0,
                    reason: format!(
                        "mass({j},{k})={} > sum_mass({k})={} in fixup 3b",
                        mass[k * nc + j],
                        sum_mass[k]
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Repairs negative component masses by clamping tiny excursions to zero or
/// conservatively borrowing from immediate neighboring cells
/// (`diffusion_procs.f90:230-313`).
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if conservation or positivity fails after
/// neighbor borrowing.
pub fn fix_negative_masses(
    nc: usize,
    nzlo: usize,
    nzhi: usize,
    cell_dm: &[f64],
    mass: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    for k in nzlo..=nzhi {
        'fix1: for j in 0..nc {
            let mjk = mass[k * nc + j];
            if mjk >= 0.0 {
                continue 'fix1;
            }
            if mjk >= -1.0e-13 * cell_dm[k] {
                mass[k * nc + j] = 0.0;
                continue 'fix1;
            }

            let mut k_hi = (k + 1).min(nzhi);
            let mut k_lo = k.saturating_sub(1).max(nzlo);
            let maxcnt = 2_usize;
            let mut sum_m = 0.0;
            for cnt in 1..=maxcnt {
                sum_m = 0.0;
                for kk in k_lo..=k_hi {
                    sum_m += mass[kk * nc + j];
                }
                if sum_m >= TINY_MASS {
                    break;
                }
                if cnt == maxcnt
                    || mass[k_lo * nc + j] < 0.0
                    || mass[k_hi * nc + j] < 0.0
                {
                    mass[k * nc + j] = 0.0;
                    continue 'fix1;
                }
                k_hi = (k_hi + 1).min(nzhi);
                k_lo = k_lo.saturating_sub(1).max(nzlo);
            }

            let dm = -mass[k * nc + j];
            mass[k * nc + j] = 0.0;
            let source_mass = sum_m + dm;
            let frac = sum_m / source_mass;
            for kk in k_lo..=k_hi {
                mass[kk * nc + j] *= frac;
            }
            let mut new_sum_m = 0.0;
            for kk in k_lo..=k_hi {
                new_sum_m += mass[kk * nc + j];
            }
            if (sum_m - new_sum_m).abs() > 1.0e-12 * sum_m {
                return Err(FixupDiagnostics {
                    bad_j: Some(j),
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: (sum_m - new_sum_m) / sum_m,
                    reason: format!(
                        "mass conservation mismatch in fix_negative_masses at ({j},{k})"
                    ),
                });
            }
        }
    }

    for k in nzlo..=nzhi {
        for j in 0..nc {
            if mass[k * nc + j] < 0.0 {
                return Err(FixupDiagnostics {
                    bad_j: Some(j),
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: 0.0,
                    reason: format!(
                        "remaining negative mass({j},{k})={} after fix_negative_masses",
                        mass[k * nc + j]
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Checks global species conservation against `(x_total_atol, x_total_rtol)`
/// and rescales each class column to preserve `xtotal_init`, then populates
/// `sum_mass` (`diffusion_procs.f90:316-375`).
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if any class drifts beyond tolerance or if
/// any cell total mass is negative.
#[allow(clippy::too_many_arguments)]
pub fn fix_species_conservation(
    nc: usize,
    nzlo: usize,
    nzhi: usize,
    mtotal: f64,
    xtotal_init: &[f64],
    x_total_atol: f64,
    x_total_rtol: f64,
    mass: &mut [f64],
    sum_mass: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    for j in 0..nc {
        if xtotal_init[j] < 1.0e-50 {
            continue;
        }
        let mut mass_sum = 0.0;
        for k in nzlo..=nzhi {
            mass_sum += mass[k * nc + j];
        }
        let xtotal_new = mass_sum / mtotal;
        let frac = xtotal_new / xtotal_init[j];
        let err = (xtotal_new - xtotal_init[j]).abs()
            / (x_total_atol + x_total_rtol * xtotal_new.max(xtotal_init[j]));
        if err > 1.0 || !err.is_finite() {
            return Err(FixupDiagnostics {
                bad_j: Some(j),
                bad_k: None,
                bad_x: 0.0,
                bad_xsum: err,
                reason: format!(
                    "species conservation error err={err:e} for class {j}"
                ),
            });
        }
        if frac <= 0.0 {
            continue;
        }
        for k in nzlo..=nzhi {
            mass[k * nc + j] /= frac;
        }
    }

    for k in nzlo..=nzhi {
        let mut s = 0.0;
        for j in 0..nc {
            s += mass[k * nc + j];
        }
        sum_mass[k] = s;
        if sum_mass[k] < 0.0 || !sum_mass[k].is_finite() {
            return Err(FixupDiagnostics {
                bad_j: None,
                bad_k: Some(k),
                bad_x: 0.0,
                bad_xsum: 0.0,
                reason: format!("negative sum_mass({k}) in fixup 2"),
            });
        }
    }
    Ok(())
}

/// Optional conservative 3-point mass-fraction smoother ported verbatim from
/// `diffusion_procs.f90:378-524`.
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if any cell mass becomes negative or
/// non-finite.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn smooth_x(
    nc: usize,
    nzlo: usize,
    nzhi: usize,
    nsmooth_x_in_fixup: usize,
    max_lnt_for_smooth: f64,
    ln_t: &[f64],
    sum_mass: &mut [f64],
    mass: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    let mut max_lnt = max_lnt_for_smooth;

    for _rep in 0..nsmooth_x_in_fixup {
        if nzhi >= nzlo + 2 {
            for offset in 0..=2 {
                let mut k = nzlo + 1 + offset;
                while k < nzhi {
                    if ln_t[k] > max_lnt {
                        break;
                    }
                    let mut sum_m1 = sum_mass[k - 1];
                    let mut sum_00 = sum_mass[k];
                    let mut sum_p1 = sum_mass[k + 1];

                    for j in 0..nc {
                        let m_m1 = mass[(k - 1) * nc + j];
                        let m_00 = mass[k * nc + j];
                        let m_p1 = mass[(k + 1) * nc + j];
                        if m_m1 + m_p1 < TINY_MASS {
                            continue;
                        }
                        let xavg =
                            (m_m1 + m_00 + m_p1) / (sum_m1 + sum_00 + sum_p1);
                        if (xavg - 1.0).abs() < 1.0e-10 {
                            continue;
                        }
                        let mut dm = (sum_00 * xavg - m_00) / (1.0 - xavg);
                        if dm >= 0.0 {
                            dm = dm.min(0.999 * (m_m1 + m_p1));
                        } else {
                            dm = -(-dm).min(0.999 * m_00);
                        }
                        mass[k * nc + j] = m_00 + dm;
                        if mass[k * nc + j] < 0.0 {
                            return Err(FixupDiagnostics {
                                bad_j: Some(j),
                                bad_k: Some(k),
                                bad_x: 0.0,
                                bad_xsum: 0.0,
                                reason: "negative mass00 in smooth_x".into(),
                            });
                        }
                        sum_00 = (0..nc).map(|jj| mass[k * nc + jj]).sum();
                        if sum_00 < 0.0 || !sum_00.is_finite() {
                            return Err(FixupDiagnostics {
                                bad_j: Some(j),
                                bad_k: Some(k),
                                bad_x: 0.0,
                                bad_xsum: 0.0,
                                reason: "bad sum_00 in smooth_x".into(),
                            });
                        }
                        sum_mass[k] = sum_00;

                        let dm_m1 = dm * m_m1 / (m_m1 + m_p1);
                        mass[(k - 1) * nc + j] = m_m1 - dm_m1;
                        if mass[(k - 1) * nc + j] < 0.0 {
                            return Err(FixupDiagnostics {
                                bad_j: Some(j),
                                bad_k: Some(k - 1),
                                bad_x: 0.0,
                                bad_xsum: 0.0,
                                reason: "negative massm1 in smooth_x".into(),
                            });
                        }
                        sum_m1 =
                            (0..nc).map(|jj| mass[(k - 1) * nc + jj]).sum();
                        sum_mass[k - 1] = sum_m1;
                        if sum_m1 < 0.0 || !sum_m1.is_finite() {
                            return Err(FixupDiagnostics {
                                bad_j: Some(j),
                                bad_k: Some(k - 1),
                                bad_x: 0.0,
                                bad_xsum: 0.0,
                                reason: "bad sum_m1 in smooth_x".into(),
                            });
                        }

                        let dm_p1 = dm * m_p1 / (m_m1 + m_p1);
                        mass[(k + 1) * nc + j] = m_p1 - dm_p1;
                        if mass[(k + 1) * nc + j] < 0.0 {
                            return Err(FixupDiagnostics {
                                bad_j: Some(j),
                                bad_k: Some(k + 1),
                                bad_x: 0.0,
                                bad_xsum: 0.0,
                                reason: "negative massp1 in smooth_x".into(),
                            });
                        }
                        sum_p1 =
                            (0..nc).map(|jj| mass[(k + 1) * nc + jj]).sum();
                        sum_mass[k + 1] = sum_p1;
                        if sum_p1 < 0.0 || !sum_p1.is_finite() {
                            return Err(FixupDiagnostics {
                                bad_j: Some(j),
                                bad_k: Some(k + 1),
                                bad_x: 0.0,
                                bad_xsum: 0.0,
                                reason: "bad sum_p1 in smooth_x".into(),
                            });
                        }
                    }
                    k += 3;
                }
            }
        }
        max_lnt -= 0.1;
    }

    for (k, &sm) in sum_mass.iter().enumerate().take(nzhi + 1).skip(nzlo) {
        if sm < 0.0 {
            return Err(FixupDiagnostics {
                bad_j: None,
                bad_k: Some(k),
                bad_x: 0.0,
                bad_xsum: 0.0,
                reason: "negative sum_mass in smooth_x".into(),
            });
        }
    }
    Ok(())
}

/// Computes monotonic, zero-sum mass-fraction slopes `dX/dm` inside zone `k`
/// (`diffusion_procs.f90:527-691`).
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if `k` lies outside `nzlo..=nzhi`.
#[allow(clippy::similar_names)]
pub fn get1_dx_dm(
    k: usize,
    nc: usize,
    nzlo: usize,
    nzhi: usize,
    mass: &[f64],
    sum_mass: &[f64],
    dx_dm: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    const TINY_SLOPE: f64 = 1.0e-10;

    let row = k * nc;
    let dm_00 = sum_mass[k];
    if dm_00 < TINY_MASS {
        dx_dm[row..(row + nc)].fill(0.0);
        return Ok(());
    }
    let dm_half = 0.5 * dm_00;

    if nzlo < k && k < nzhi {
        let dm_m1 = sum_mass[k - 1];
        let dm_p1 = sum_mass[k + 1];
        if dm_m1 < TINY_MASS || dm_p1 < TINY_MASS {
            dx_dm[row..(row + nc)].fill(0.0);
            return Ok(());
        }
        let dmbar_00 = 0.5 * (dm_00 + dm_m1);
        let dmbar_p1 = 0.5 * (dm_00 + dm_p1);

        for j in 0..nc {
            let xm1 = mass[(k - 1) * nc + j] / dm_m1;
            let x00 = mass[row + j] / dm_00;
            let xp1 = mass[(k + 1) * nc + j] / dm_p1;
            let sm1 = (xm1 - x00) / dmbar_00;
            let s00 = (x00 - xp1) / dmbar_p1;
            let slope = 0.5 * (sm1 + s00);
            if sm1 * s00 <= 0.0 || slope.abs() < TINY_SLOPE {
                dx_dm[row + j] = 0.0;
            } else {
                dx_dm[row + j] = slope;
                let xface_00 = x00 + slope * dm_half;
                let xface_p1 = x00 - slope * dm_half;
                if xface_00 > 1.0
                    || xface_00 < 0.0
                    || (xm1 - xface_00) * (xface_00 - x00) < 0.0
                    || xface_p1 > 1.0
                    || xface_p1 < 0.0
                    || (x00 - xface_p1) * (xface_p1 - xp1) < 0.0
                {
                    if sm1.abs() <= s00.abs() {
                        dx_dm[row + j] = sm1;
                    } else {
                        dx_dm[row + j] = s00;
                    }
                }
            }
        }
    } else if k == nzlo {
        let dm_p1 = sum_mass[k + 1];
        if dm_p1 < TINY_MASS {
            dx_dm[row..(row + nc)].fill(0.0);
            return Ok(());
        }
        let dmbar_p1 = 0.5 * (dm_00 + dm_p1);

        for j in 0..nc {
            let x00 = mass[row + j] / dm_00;
            let xp1 = mass[(k + 1) * nc + j] / dm_p1;
            let slope = (x00 - xp1) / dmbar_p1;
            if slope.abs() < TINY_SLOPE {
                dx_dm[row + j] = 0.0;
            } else {
                dx_dm[row + j] = slope;
                let xface_00 = x00 + slope * dm_half;
                let xface_p1 = x00 - slope * dm_half;
                if xface_00 > 1.0
                    || xface_00 < 0.0
                    || (x00 - xface_p1) * (xface_p1 - xp1) < 0.0
                {
                    dx_dm[row + j] = 0.0;
                }
            }
        }
    } else if k == nzhi {
        let dm_m1 = sum_mass[k - 1];
        if dm_m1 < TINY_MASS {
            dx_dm[row..(row + nc)].fill(0.0);
            return Ok(());
        }
        let dmbar_00 = 0.5 * (dm_00 + dm_m1);

        for j in 0..nc {
            let xm1 = mass[(k - 1) * nc + j] / dm_m1;
            let x00 = mass[row + j] / dm_00;
            let slope = (xm1 - x00) / dmbar_00;
            if slope.abs() < TINY_SLOPE {
                dx_dm[row + j] = 0.0;
            } else {
                dx_dm[row + j] = slope;
                let xface_00 = x00 + slope * dm_half;
                let xface_p1 = x00 - slope * dm_half;
                if xface_p1 > 1.0
                    || xface_p1 < 0.0
                    || (xm1 - xface_00) * (xface_00 - x00) < 0.0
                {
                    dx_dm[row + j] = 0.0;
                }
            }
        }
    } else {
        return Err(FixupDiagnostics {
            bad_j: None,
            bad_k: Some(k),
            bad_x: 0.0,
            bad_xsum: 0.0,
            reason: format!("k={k} outside nzlo..=nzhi ({nzlo}..={nzhi})"),
        });
    }

    // Adjust largest-magnitude opposing slope so sum(dX_dm) == 0.
    let mut s_sum = 0.0;
    for j in 0..nc {
        s_sum += dx_dm[row + j];
    }
    let j_adj = if s_sum > 0.0 {
        let mut best_j = 0_usize;
        let mut best_v = dx_dm[row];
        for j in 1..nc {
            if dx_dm[row + j] > best_v {
                best_v = dx_dm[row + j];
                best_j = j;
            }
        }
        best_j
    } else {
        let mut best_j = 0_usize;
        let mut best_v = dx_dm[row];
        for j in 1..nc {
            if dx_dm[row + j] < best_v {
                best_v = dx_dm[row + j];
                best_j = j;
            }
        }
        best_j
    };
    dx_dm[row + j_adj] = 0.0;
    let mut rem_sum = 0.0;
    for j in 0..nc {
        rem_sum += dx_dm[row + j];
    }
    dx_dm[row + j_adj] = -rem_sum;

    // Recheck face validity; zero out all slopes in zone k if violated.
    for j in 0..nc {
        let x00 = mass[row + j] / dm_00;
        let slope = dx_dm[row + j];
        let xface_00 = x00 + slope * dm_half;
        let xface_p1 = x00 - slope * dm_half;
        if xface_00 > 1.0 || xface_00 < 0.0 || xface_p1 > 1.0 || xface_p1 < 0.0
        {
            dx_dm[row..(row + nc)].fill(0.0);
            break;
        }
    }

    Ok(())
}

/// Serially remaps per-class masses from cells of mass `sum_mass(k)` back onto
/// fixed Lagrangian cells `cell_dm(k)` so `sum(X_new(:, k)) == 1` everywhere
/// (`diffusion_procs.f90:694-982`).
///
/// # Errors
///
/// Returns `Err(FixupDiagnostics)` if mass redistribution fails to fill a cell
/// or loses conservation.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn redistribute_mass(
    nc: usize,
    m: usize,
    nzlo: usize,
    nzhi: usize,
    _nz: usize,
    sum_mass: &mut [f64],
    mass: &[f64],
    dx_dm: &[f64],
    xtotal_init: &[f64],
    cell_dm: &[f64],
    x_new: &mut [f64],
) -> Result<(), FixupDiagnostics> {
    let integrate_mass = |k_source: usize,
                          j: usize,
                          dm0: f64,
                          dm1: f64,
                          diff: f64,
                          sum_mass: &[f64]|
     -> Result<f64, FixupDiagnostics> {
        let (k_use, slope) = if k_source > nzhi {
            (nzhi, 0.0)
        } else {
            (k_source, dx_dm[k_source * nc + j])
        };
        let dm = sum_mass[k_use];
        if dm < TINY_MASS {
            return Ok(0.0);
        }
        let x = mass[k_use * nc + j] / dm;
        let half_dm = 0.5 * dm;
        let x0 = x + slope * (dm0 - half_dm);
        let x1 = x + slope * (dm1 - half_dm);
        if dm0 > dm1 {
            return Err(FixupDiagnostics {
                bad_j: Some(j),
                bad_k: Some(k_use),
                bad_x: 0.0,
                bad_xsum: 0.0,
                reason: format!("dm0={dm0:e} > dm1={dm1:e} in integrate_mass"),
            });
        }
        let xavg = (0.5 * (x0 + x1)).clamp(0.0, 1.0);
        Ok(xavg * diff)
    };

    let mut total_moved = 0.0;
    let mut k_source = nzlo;
    let mut source_cell_mass = sum_mass[k_source];
    let mut remaining_source_mass = source_cell_mass;
    let max_iters = 100_usize;

    for k in nzlo..=nzhi {
        let cell_dm_k = cell_dm[k];
        let mut remaining_needed_mass = cell_dm_k;
        x_new[k * m..(k * m + nc)].fill(0.0);

        for _iter in 0..max_iters {
            if !remaining_needed_mass.is_finite()
                || !remaining_source_mass.is_finite()
            {
                return Err(FixupDiagnostics {
                    bad_j: None,
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: 0.0,
                    reason: "non-finite mass in redistribute_mass".into(),
                });
            }

            if remaining_needed_mass <= remaining_source_mass {
                let diff_dm = remaining_needed_mass;
                let dm0 = remaining_source_mass - diff_dm;
                let dm1 = remaining_source_mass;
                if dm0 < 0.0 || dm1 < 0.0 {
                    return Err(FixupDiagnostics {
                        bad_j: None,
                        bad_k: Some(k),
                        bad_x: 0.0,
                        bad_xsum: 0.0,
                        reason: "negative dm0/dm1 in redistribute_mass".into(),
                    });
                }
                total_moved += remaining_needed_mass;
                for j in 0..nc {
                    let dm =
                        integrate_mass(k_source, j, dm0, dm1, diff_dm, sum_mass)?;
                    if dm < 0.0 {
                        return Err(FixupDiagnostics {
                            bad_j: Some(j),
                            bad_k: Some(k),
                            bad_x: 0.0,
                            bad_xsum: 0.0,
                            reason: "negative integrated dm in redistribute_mass"
                                .into(),
                        });
                    }
                    x_new[k * m + j] += dm / cell_dm_k;
                }
                remaining_source_mass = dm0;
                remaining_needed_mass = 0.0;
                break;
            }

            let diff_dm = remaining_source_mass;
            let dm0 = 0.0;
            let dm1 = diff_dm;
            for j in 0..nc {
                let dm =
                    integrate_mass(k_source, j, dm0, dm1, diff_dm, sum_mass)?;
                x_new[k * m + j] += dm / cell_dm_k;
            }
            if !remaining_needed_mass.is_finite() {
                return Err(FixupDiagnostics {
                    bad_j: None,
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: 0.0,
                    reason: "non-finite remaining_needed_mass".into(),
                });
            }
            total_moved += remaining_source_mass;
            remaining_needed_mass -= remaining_source_mass;
            if remaining_needed_mass > cell_dm_k {
                return Err(FixupDiagnostics {
                    bad_j: None,
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: 0.0,
                    reason: "remaining_needed_mass > cell_dm_k".into(),
                });
            }

            k_source += 1;
            source_cell_mass = sum_mass[nzhi.min(k_source)];
            remaining_source_mass = source_cell_mass;
            if source_cell_mass < 0.0 || !source_cell_mass.is_finite() {
                return Err(FixupDiagnostics {
                    bad_j: None,
                    bad_k: Some(k),
                    bad_x: 0.0,
                    bad_xsum: 0.0,
                    reason: "bad source cell mass in element diffusion".into(),
                });
            }
        }

        if remaining_needed_mass > 0.0 {
            return Err(FixupDiagnostics {
                bad_j: None,
                bad_k: Some(k),
                bad_x: 0.0,
                bad_xsum: 0.0,
                reason: "bad remaining mass in element diffusion".into(),
            });
        }

        let mut sum_x = 0.0;
        for j in 0..nc {
            sum_x += x_new[k * m + j];
        }
        if (sum_x - 1.0).abs() > 1.0e-10 || !sum_x.is_finite() {
            return Err(FixupDiagnostics {
                bad_j: None,
                bad_k: Some(k),
                bad_x: sum_x,
                bad_xsum: sum_x - 1.0,
                reason: format!(
                    "sum(X_new({k})) - 1 = {:e} exceeded 1e-10 in redistribute_mass",
                    sum_x - 1.0
                ),
            });
        }
        for j in 0..nc {
            x_new[k * m + j] /= sum_x;
        }
        sum_mass[k] /= sum_x;
    }

    let mut total_source = 0.0;
    for &sm in &sum_mass[nzlo..=nzhi] {
        total_source += sm;
    }
    if (total_moved / total_source - 1.0).abs() > 1.0e-12 {
        return Err(FixupDiagnostics {
            bad_j: None,
            bad_k: None,
            bad_x: 0.0,
            bad_xsum: total_moved / total_source - 1.0,
            reason: "total_moved/total_source mismatch in redistribute_mass"
                .into(),
        });
    }

    for k in nzlo..=nzhi {
        let mut new_sum = 0.0;
        for j in 0..nc {
            new_sum += x_new[k * m + j];
        }
        if (new_sum - 1.0).abs() > 1.0e-14 || !new_sum.is_finite() {
            return Err(FixupDiagnostics {
                bad_j: None,
                bad_k: Some(k),
                bad_x: new_sum,
                bad_xsum: new_sum - 1.0,
                reason: format!(
                    "redistribute_mass new_sum - 1 = {:e} at zone {k}",
                    new_sum - 1.0
                ),
            });
        }
    }

    let _ = xtotal_init; // Upstream checks conservation and logs without aborting.
    Ok(())
}

/// Computes smooth cosine-tapered `(Gamma, T, phase)` diffusion limit
/// coefficients across faces `nzlo + 1 ..= nzhi` and returns `k_max`
/// (`diffusion_procs.f90:985-1039`).
#[allow(clippy::too_many_arguments)]
pub fn get_limit_coeffs(
    nzlo: usize,
    nzhi: usize,
    gamma_full_on: f64,
    gamma_full_off: f64,
    t_full_on: f64,
    t_full_off: f64,
    gamma: &[f64],
    t: &[f64],
    phase: &[f64],
    limit_coeffs_face: &mut [f64],
) -> usize {
    let mut k_max = nzhi;

    for k in ((nzlo + 1)..=nzhi).rev() {
        let gamma_max = gamma[k].max(gamma[k - 1]);
        let gamma_term = if gamma_max >= gamma_full_off {
            0.0
        } else if gamma_max <= gamma_full_on {
            1.0
        } else {
            (gamma_full_off - gamma_max) / (gamma_full_off - gamma_full_on)
        };

        let t_min = t[k].min(t[k - 1]);
        let t_term = if t_min >= t_full_on {
            1.0
        } else if t_min <= t_full_off {
            0.0
        } else {
            (t_full_off - t_min) / (t_full_off - t_full_on)
        };

        let lim = gamma_term * t_term * (1.0 - phase[k]);
        if lim <= 0.0 {
            limit_coeffs_face[k] = 0.0;
            k_max = k - 1;
        } else if lim >= 1.0 {
            limit_coeffs_face[k] = 1.0;
        } else {
            limit_coeffs_face[k] = 0.5 * (1.0 - flexmesa_math::cospi(lim));
        }
    }

    k_max
}

/// Populates face structure quantities (`T_face`, `rho_face`,
/// `four_pi_r2_rho_face`, `dlnP_dr_face`, `dlnT_dr_face`, `dlnRho_dr_face`,
/// `xm_face`), ionization charges `typical_charge` and `Z`, and validates that
/// radiative levitation (`op_mono`) is not requested
/// (`diffusion_procs.f90:1042-1211`).
///
/// Returns `Ok(kmax_rad_accel)` (always `0` while OP monochromatic opacities
/// are unported).
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if `T_face[nzlo + 1] <= max_t_for_radaccel`,
/// because radiative levitation via OP monochromatic opacities (`op_mono`,
/// `#156` Task 4 / `#161` Slice 3) is not yet implemented.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn setup_struct_info(
    nzlo: usize,
    nzhi: usize,
    nc: usize,
    m: usize,
    dln_p_dm_face: &[f64],
    dln_t_dm_face: &[f64],
    dln_rho_dm_face: &[f64],
    cell_dm: &[f64],
    abar: &[f64],
    free_e: &[f64],
    t: &[f64],
    ln_t: &[f64],
    rho: &[f64],
    lnd: &[f64],
    r_face: &[f64],
    alfa_face: &[f64],
    class_z: &[usize],
    calculate_ionization: bool,
    nsmooth_typical_charge: usize,
    max_t_for_radaccel: f64,
    rho_face: &mut [f64],
    t_face: &mut [f64],
    four_pi_r2_rho_face: &mut [f64],
    dln_p_dr_face: &mut [f64],
    dln_t_dr_face: &mut [f64],
    dln_rho_dr_face: &mut [f64],
    z: &mut [f64],
    typical_charge: &mut [f64],
    xm_face: &mut [f64],
    rad_accel_face: &mut [f64],
    g_rad: &mut [f64],
) -> Result<usize, StarError> {
    for k in nzlo..=nzhi {
        if k > nzlo {
            let alfa = alfa_face[k];
            let beta = 1.0 - alfa;
            t_face[k] = alfa * t[k] + beta * t[k - 1];
            rho_face[k] = alfa * rho[k] + beta * rho[k - 1];
            four_pi_r2_rho_face[k] = PI4 * r_face[k] * r_face[k] * rho_face[k];
            dln_p_dr_face[k] = four_pi_r2_rho_face[k] * dln_p_dm_face[k];
            dln_t_dr_face[k] = four_pi_r2_rho_face[k] * dln_t_dm_face[k];
            dln_rho_dr_face[k] = four_pi_r2_rho_face[k] * dln_rho_dm_face[k];
        }

        if calculate_ionization {
            let log10_t = ln_t[k] / LN10;
            let log10_rho = lnd[k] / LN10;
            let z1 = abar[k] * free_e[k];
            for i in 0..nc {
                typical_charge[k * nc + i] = get_typical_charge(
                    class_z[i],
                    abar[k],
                    z1,
                    t[k],
                    log10_t,
                    rho[k],
                    log10_rho,
                );
            }
        }

        for i in 0..nc {
            rad_accel_face[k * m + i] = 0.0;
            g_rad[k * nc + i] = 0.0;
        }
        rad_accel_face[k * m + nc] = 0.0;
    }

    if nsmooth_typical_charge > 0 && nzhi >= nzlo + 2 {
        for _pass in 0..nsmooth_typical_charge {
            for i in 0..nc {
                typical_charge[nzlo * nc + i] =
                    (2.0 * typical_charge[nzlo * nc + i]
                        + typical_charge[(nzlo + 1) * nc + i])
                        / 3.0;
                for k in (nzlo + 1)..=(nzhi - 2) {
                    typical_charge[k * nc + i] =
                        (typical_charge[(k - 1) * nc + i]
                            + typical_charge[k * nc + i]
                            + typical_charge[(k + 1) * nc + i])
                            / 3.0;
                }
                typical_charge[(nzhi - 1) * nc + i] =
                    (2.0 * typical_charge[(nzhi - 1) * nc + i]
                        + typical_charge[(nzhi - 2) * nc + i])
                        / 3.0;
                for k in ((nzlo + 1)..=(nzhi - 2)).rev() {
                    typical_charge[k * nc + i] =
                        (typical_charge[(k - 1) * nc + i]
                            + typical_charge[k * nc + i]
                            + typical_charge[(k + 1) * nc + i])
                            / 3.0;
                }
                typical_charge[nzhi * nc + i] =
                    typical_charge[(nzhi - 1) * nc + i];
            }
        }
    }

    xm_face[nzlo] = 0.0;
    for k in nzlo..=nzhi {
        for i in 0..nc {
            z[k * m + i] = typical_charge[k * nc + i];
        }
        z[k * m + nc] = -1.0;
        if k < nzhi {
            xm_face[k + 1] = xm_face[k] + cell_dm[k];
        }
    }

    if t_face[nzlo + 1] > max_t_for_radaccel {
        return Ok(0);
    }

    Err(StarError::InvalidInput(format!(
        "radiative levitation via OP monochromatic opacities (op_mono) is not \
         implemented yet (T_face(nzlo+1)={:.6e} K <= diffusion_max_T_for_radaccel={:.6e} K; \
         tracked under #156/#161). Set diffusion_max_T_for_radaccel = 0 to disable \
         radiative acceleration.",
        t_face[nzlo + 1],
        max_t_for_radaccel
    )))
}

/// Maps diffused class mass fractions `X` back onto individual network
/// isotopes `xa` proportionally to `X / X_init` and renormalizes each zone
/// (`diffusion_procs.f90:1716-1744`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if any `X_init` entry or
/// post-update zone mass-fraction sum is non-positive or non-finite.
#[allow(clippy::too_many_arguments)]
pub fn set_new_xa(
    nzlo: usize,
    nzhi: usize,
    species: usize,
    nc: usize,
    m: usize,
    class: &[usize],
    x_init: &[f64],
    x: &[f64],
    xa: &mut [f64],
) -> Result<(), StarError> {
    for k in nzlo..=nzhi {
        for (j, &cls) in class.iter().enumerate().take(species) {
            let xi0 = x_init[k * nc + cls];
            if xi0 <= 0.0 || !xi0.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "non-positive or non-finite X_init({cls},{k})={xi0:e} in set_new_xa"
                )));
            }
            xa[k * species + j] =
                xa[k * species + j] * x[k * m + cls] / xi0;
        }
        let mut tmp = 0.0;
        for j in 0..species {
            tmp += xa[k * species + j];
        }
        if tmp <= 0.0 || !tmp.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "invalid mass-fraction sum {tmp:e} at zone {k} in set_new_xa"
            )));
        }
        for j in 0..species {
            xa[k * species + j] /= tmp;
        }
    }
    Ok(())
}
