// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream MESA `26.4.1` conservative Lagrangian profile remapping
//! (`do_mesh_adjust`, `get1_lpp`, `get_xq_integral`, `do_lnR_and_lnd`, `do_xa`,
//! `do1_lnT`, `set_lnT_for_energy`, `adjust_omega`, `do_v`, `do_u`, `do_etrb`,
//! and `do_prune_mesh_surface` in `star/private/mesh_adjust.f90:1-2889` and
//! `star/private/adjust_mesh.f90:36-580`).

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{FOUR_THIRDS_PI, LN10, ONE_THIRD, STANDARD_CGRAV};
use flexmesa_eos::{eos_dt_get_t, I_LN_E};
use flexmesa_interp1d::interpolate_vector_pm;

use crate::adjust_mesh_support::{
    fortran_maxloc_0based, fortran_minloc_0based,
};
use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_adjust::PlannedMeshOutcome;
use crate::model::StarModel;

/// Electron degeneracy threshold above which `do1_lnT` skips EOS temperature
/// inversion (`eta_limit = -1d-6` in `star/private/mesh_adjust.f90:38, 1403`).
pub const ETA_LIMIT_FOR_MESH_ADJUST: f64 = -1.0e-6;

/// Classification of each remeshed cell relative to the parent grid
/// (`unchanged_type`, `split_type`, `merged_type`, `revised_type` in
/// `star/private/adjust_mesh.f90:519-579`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    /// Cell boundaries `[xq_new(k), xq_new(k+1)]` match parent cell `comes_from[k]`.
    Unchanged,
    /// Cell lies strictly within or subdivides a parent cell.
    Split,
    /// Cell spans two or more merged parent cells.
    Merged,
    /// Cell retained boundaries while thermodynamic state was revised.
    Revised,
}

/// Cached plan produced by `adjust_mesh` (`do_mesh_plan` + `set_types_of_new_cells`
/// in `star/private/adjust_mesh.f90:244-579`).
#[derive(Debug, Clone)]
pub struct StandardMeshPlan {
    /// Parent grid zone count (`nz_old`).
    pub nz_old: usize,
    /// Parent grid `dq` after `dq(nz) = 1 - sum(dq(1:nz-1))` (`adjust_mesh.f90:245-261`).
    pub dq_old: Vec<f64>,
    /// Original `model.dq` bits at `plan_mesh` entry for cache matching.
    pub raw_dq_old: Vec<f64>,
    /// Parent grid exterior mass fractions `xq_old`.
    pub xq_old: Vec<f64>,
    /// Planned new cell mass fractions `dq_new`.
    pub dq_new: Vec<f64>,
    /// Planned new exterior mass fractions `xq_new`.
    pub xq_new: Vec<f64>,
    /// 0-based parent zone index `comes_from[k]` in `0..nz_old`.
    pub comes_from: Vec<usize>,
    /// 1-based index of governing mesh function (`which_gval`).
    pub which_gval: Vec<usize>,
    /// Per-zone [`CellType`] classification (`len = nz_new`).
    pub cell_type: Vec<CellType>,
    /// Count of `Split` cells in `cell_type`.
    pub num_splits: usize,
    /// Count of `Merged` cells in `cell_type`.
    pub num_merges: usize,
    /// Whether `model.xa` was already cleaned in place prior to planning.
    pub(crate) xa_already_cleaned: bool,
}

#[inline]
pub(crate) fn f64_slices_bit_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Floors and renormalizes zone mass fractions matching `clean_up_fractions` /
/// `do_clean1` (`net/private/net_eval.f90:822-850`, `tiny_x = 1d-99`).
pub(crate) fn clean_up_fractions_slice(
    xa: &mut [f64],
    nz: usize,
    species: usize,
    max_sum_abs: f64,
    xsum_tol: f64,
) -> Result<(), StarError> {
    const TINY_X: f64 = 1.0e-99;
    for k in 0..nz {
        let row = &mut xa[k * species..(k + 1) * species];
        if max_sum_abs > 1.0 {
            let sum_abs: f64 = row.iter().map(|x| x.abs()).sum();
            if !sum_abs.is_finite() || sum_abs > max_sum_abs {
                return Err(StarError::InvalidInput(format!(
                    "clean_up_fractions failed during adjust_mesh at zone {}: sum_abs={sum_abs}",
                    k + 1
                )));
            }
        }
        for x in row.iter_mut() {
            *x = (*x).clamp(TINY_X, 1.0);
        }
        let xsum: f64 = row.iter().sum();
        if !xsum.is_finite() || (xsum - 1.0).abs() > xsum_tol {
            return Err(StarError::InvalidInput(format!(
                "clean_up_fractions failed during adjust_mesh at zone {}: sum={xsum}",
                k + 1
            )));
        }
        for x in row.iter_mut() {
            *x /= xsum;
        }
    }
    Ok(())
}

/// Maps `clean_up_fractions_slice` failures during remeshing to retriable
/// `StarError::MicrophysicsError` (`remesh = retry` in
/// `star/private/adjust_mesh.f90:109-116`).
pub(crate) fn clean_up_fractions_remesh(
    xa: &mut [f64],
    nz: usize,
    species: usize,
) -> Result<(), StarError> {
    clean_up_fractions_slice(xa, nz, species, 10.0, 1.0e-2).map_err(|e| {
        StarError::MicrophysicsError(format!(
            "clean_up_fractions failed during adjust mesh: {e}"
        ))
    })
}

/// Computes `dq(nz) = 1 - sum(dq(1:nz-1))` with fallback normalization matching
/// `star/private/adjust_mesh.f90:244-261, 354-369`.
pub(crate) fn fix_tail_dq(dq: &mut [f64]) -> Result<(), StarError> {
    let nz = dq.len();
    if nz == 0 {
        return Ok(());
    }
    if nz == 1 {
        dq[0] = 1.0;
        return Ok(());
    }
    let tmp = dq[nz - 1];
    let sum_m1: f64 = dq[..nz - 1].iter().sum();
    dq[nz - 1] = 1.0 - sum_m1;
    if dq[nz - 1] <= 0.0 {
        dq[nz - 1] = tmp;
        let sum_all: f64 = dq.iter().sum();
        if sum_all <= 0.0 || !sum_all.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "non-positive sum(dq) = {sum_all} in adjust_mesh"
            )));
        }
        for val in dq.iter_mut() {
            *val /= sum_all;
        }
        let sum_m1_retry: f64 = dq[..nz - 1].iter().sum();
        dq[nz - 1] = 1.0 - sum_m1_retry;
        if dq[nz - 1] <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "dq(nz) = {} <= 0 in adjust_mesh",
                dq[nz - 1]
            )));
        }
    }
    Ok(())
}

/// Computes exterior mass coordinates `xq` from `dq` (`set_xqs` in
/// `star/private/star_utils.f90:1557`).
#[must_use]
pub(crate) fn compute_xqs(dq: &[f64]) -> Vec<f64> {
    let nz = dq.len();
    let mut xq = vec![0.0_f64; nz];
    for k in 1..nz {
        xq[k] = xq[k - 1] + dq[k - 1];
    }
    xq
}

/// Classifies new zones against parent zones (`set_types_of_new_cells` in
/// `star/private/adjust_mesh.f90:519-579`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `xq_new[k] < xq_old[comes_from[k]]`.
pub fn classify_new_cells(
    xq_old: &[f64],
    xq_new: &[f64],
    comes_from: &[usize],
) -> Result<(Vec<CellType>, usize, usize, usize), StarError> {
    let nz_old = xq_old.len();
    let nz_new = xq_new.len();
    if comes_from.len() != nz_new {
        return Err(StarError::InvalidInput(String::from(
            "classify_new_cells: comes_from length mismatch",
        )));
    }
    let mut cell_type = Vec::with_capacity(nz_new);
    let mut unchanged = 0_usize;
    let mut split = 0_usize;
    let mut merged = 0_usize;

    for k in 0..nz_new {
        let k_old = comes_from[k];
        if k_old >= nz_old {
            return Err(StarError::InvalidInput(format!(
                "classify_new_cells: comes_from[{k}]={k_old} >= nz_old={nz_old}"
            )));
        }
        let ct = if xq_new[k] < xq_old[k_old] {
            return Err(StarError::InvalidInput(format!(
                "set_types_of_new_cells: xq_new({})={} < xq_old({})={}",
                k + 1,
                xq_new[k],
                k_old + 1,
                xq_old[k_old]
            )));
        } else if xq_new[k] > xq_old[k_old] {
            CellType::Split
        } else if k_old + 1 == nz_old {
            if k + 1 == nz_new {
                CellType::Unchanged
            } else {
                CellType::Split
            }
        } else if k + 1 == nz_new {
            CellType::Split
        } else if xq_new[k + 1] == xq_old[k_old + 1] {
            CellType::Unchanged
        } else if xq_new[k + 1] > xq_old[k_old + 1] {
            CellType::Merged
        } else {
            CellType::Split
        };
        match ct {
            CellType::Unchanged | CellType::Revised => unchanged += 1,
            CellType::Split => split += 1,
            CellType::Merged => merged += 1,
        }
        cell_type.push(ct);
    }

    if unchanged + split + merged != nz_new {
        return Err(StarError::InvalidInput(String::from(
            "set_types_of_new_cells count mismatch",
        )));
    }
    Ok((cell_type, unchanged, split, merged))
}

/// Counts net split and merge operations across minimal aligned mass intervals
/// so `splits + merges == 0` iff every cell is [`CellType::Unchanged`] and
/// `nz_old + splits - merges == nz_new` holds identically.
#[must_use]
pub(crate) fn count_plan_splits_and_merges(
    xq_old: &[f64],
    xq_new: &[f64],
    comes_from: &[usize],
    cell_type: &[CellType],
) -> (usize, usize) {
    let nz_old = xq_old.len();
    let nz_new = xq_new.len();
    if nz_old == 0 || nz_new == 0 {
        return (0, 0);
    }
    let mut splits = 0_usize;
    let mut merges = 0_usize;
    let mut k0 = 0_usize;
    let mut ko0 = 0_usize;
    for k in 1..=nz_new {
        let aligned_ko = if k == nz_new {
            Some(nz_old)
        } else {
            let ko = comes_from[k];
            (xq_new[k] == xq_old[ko]).then_some(ko)
        };
        let Some(ko1) = aligned_ko else {
            continue;
        };
        let new_cnt = k - k0;
        let old_cnt = ko1.saturating_sub(ko0);
        if new_cnt > old_cnt {
            splits += new_cnt - old_cnt;
        } else if old_cnt > new_cnt {
            merges += old_cnt - new_cnt;
        } else if new_cnt > 1
            || (new_cnt == 1 && cell_type[k0] != CellType::Unchanged)
        {
            splits += 1;
            merges += 1;
        }
        k0 = k;
        ko0 = ko1;
    }
    (splits, merges)
}

fn build_plan_for_given_dq(
    model: &StarModel,
    dq_new_in: &[f64],
) -> Result<StandardMeshPlan, StarError> {
    let nz_old = model.nz;
    let nz_new = dq_new_in.len();
    if nz_old < 2 || nz_new < 2 {
        return Err(StarError::InvalidInput(String::from(
            "dq_new and model must each have at least 2 zones",
        )));
    }
    let raw_dq_old = model.dq.clone();
    let mut dq_old = raw_dq_old.clone();
    fix_tail_dq(&mut dq_old)?;
    let xq_old = compute_xqs(&dq_old);

    let mut dq_new = dq_new_in.to_vec();
    fix_tail_dq(&mut dq_new)?;
    let xq_new = compute_xqs(&dq_new);

    let mut comes_from = vec![0_usize; nz_new];
    let mut ko = 0_usize;
    for k in 0..nz_new {
        let mut xk = xq_new[k];
        while ko + 1 < nz_old && xq_old[ko + 1] <= xk {
            ko += 1;
        }
        if ko + 1 < nz_old
            && (xk - xq_old[ko + 1]).abs() <= 1.0e-15 * dq_old[ko]
        {
            ko += 1;
            xk = xq_old[ko];
        }
        if (xk - xq_old[ko]).abs() <= 1.0e-15 * dq_old[ko] {
            let _ = xk;
        }
        comes_from[k] = ko;
    }

    let mut xq_aligned = xq_new;
    for k in 0..nz_new {
        let ko = comes_from[k];
        if (xq_aligned[k] - xq_old[ko]).abs()
            <= 1.0e-15 * dq_old[ko].max(1.0e-30)
        {
            xq_aligned[k] = xq_old[ko];
        }
    }
    let (cell_type, _unch, _split_cells, _merged_cells) =
        classify_new_cells(&xq_old, &xq_aligned, &comes_from)?;
    let (num_splits, num_merges) = count_plan_splits_and_merges(
        &xq_old,
        &xq_aligned,
        &comes_from,
        &cell_type,
    );

    Ok(StandardMeshPlan {
        nz_old,
        dq_old,
        raw_dq_old,
        xq_old,
        dq_new,
        xq_new: xq_aligned,
        comes_from,
        which_gval: vec![0; nz_new],
        cell_type,
        num_splits,
        num_merges,
        xa_already_cleaned: false,
    })
}

pub use crate::cell_energy::{cell_specific_ke, cell_specific_pe};

/// Evaluates `total_angular_momentum` (`star/private/star_utils.f90:1835-1855`).
#[must_use]
pub fn total_angular_momentum(model: &StarModel) -> f64 {
    if !model.controls.rotation_flag {
        return 0.0;
    }
    model
        .dm_bar
        .iter()
        .zip(&model.j_rot)
        .map(|(dm_b, jr)| dm_b * jr)
        .sum()
}

/// Single-zone LPP-HARMOD reconstruction (`get1_lpp` in
/// `star/private/mesh_adjust.f90:1720-1807`, Stiriba 2003).
///
/// Reconstructs $f(\delta q) = c_0 + c_1\,\delta q + \tfrac{1}{2} c_2\,\delta q^2$
/// with $\delta q = q - q_{\mathrm{mid}} = x_{q,\mathrm{mid}} - x_q$.
#[must_use]
pub fn get1_lpp_zone(
    k: usize,
    nz: usize,
    species: usize,
    j: usize,
    dq: &[f64],
    v_zone_major: &[f64],
    quad: bool,
) -> (f64, f64, f64) {
    let v_at = |z: usize| v_zone_major[z * species + j];
    if k == 0 || k + 1 >= nz {
        return (v_at(k), 0.0, 0.0);
    }
    let sm1 = (v_at(k - 1) - v_at(k)) / ((dq[k - 1] + dq[k]) * 0.5);
    let s00 = (v_at(k) - v_at(k + 1)) / ((dq[k] + dq[k + 1]) * 0.5);
    let sprod = sm1 * s00;
    if sprod <= 0.0 {
        return (v_at(k), 0.0, 0.0);
    }

    let (c0, mut c1, mut c2) = if quad {
        let c1_h = (sprod * 2.0) / (s00 + sm1);
        let c2_h = if sm1.abs() <= s00.abs() {
            (sm1 - c1_h) / (2.0 * dq[k])
        } else {
            (c1_h - s00) / (2.0 * dq[k])
        };
        let c0_h = v_at(k) - (c2_h * dq[k] * dq[k]) / 24.0;
        (c0_h, c1_h, c2_h)
    } else {
        (v_at(k), (sm1 + s00) * 0.5, 0.0)
    };

    let dqhalf = dq[k] * 0.5;
    let vbdy1 = c0 + c1 * dqhalf + 0.5 * c2 * dqhalf * dqhalf;
    let vbdy2 = c0 - c1 * dqhalf + 0.5 * c2 * dqhalf * dqhalf;
    if (v_at(k - 1) - vbdy1) * (vbdy1 - v_at(k)) < 0.0
        || (v_at(k) - vbdy2) * (vbdy2 - v_at(k + 1)) < 0.0
    {
        c2 = 0.0;
        c1 = if sm1.abs() <= s00.abs() { sm1 } else { s00 };
    }
    (c0, c1, c2)
}

/// Computes monotone, sum-preserving linear slopes `(xa_c0, xa_c1)` (`(nz_old, species)`)
/// in `q - q_mid = xq_mid - xq` (`get1_lpp` + sum-zero normalization in
/// `star/private/mesh_adjust.f90:255-288, 1720-1807`).
///
/// Note: `compute_monotone_xa_slopes` returns `xa_c1_xq = -xa_c1_q` in exterior
/// mass coordinate `xq - xq_mid` for backwards compatibility with callers using
/// `xa_c0 + xa_c1_xq * (xq - xq_mid)`.
#[must_use]
pub fn compute_monotone_xa_slopes(
    dq_old: &[f64],
    xa_old: &[f64],
    nz_old: usize,
    species: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (c0, c1_q) =
        compute_upstream_xa_lpp(dq_old, xa_old, nz_old, species, 0, nz_old);
    let c1_xq = c1_q.into_iter().map(|s| -s).collect();
    (c0, c1_xq)
}

fn compute_upstream_xa_lpp(
    dq_old: &[f64],
    xa_old: &[f64],
    nz_old: usize,
    species: usize,
    k_start: usize,
    k_end_excl: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut xa_c0 = vec![0.0; nz_old * species];
    let mut xa_c1 = vec![0.0; nz_old * species];
    if nz_old == 0 || species == 0 {
        return (xa_c0, xa_c1);
    }

    for k in k_start..k_end_excl.min(nz_old) {
        let base = k * species;
        for j in 0..species {
            let (c0, c1, _c2) =
                get1_lpp_zone(k, nz_old, species, j, dq_old, xa_old, false);
            xa_c0[base + j] = c0;
            xa_c1[base + j] = c1;
        }
        let sumx: f64 = xa_old[base..base + species].iter().sum();
        if sumx > 0.0 {
            for j in 0..species {
                xa_c0[base + j] = xa_old[base + j] / sumx;
            }
        }

        let slope_row = &mut xa_c1[base..base + species];
        let slope_sum: f64 = slope_row.iter().sum();
        let j_piv = if slope_sum > 0.0 {
            fortran_maxloc_0based(slope_row)
        } else {
            fortran_minloc_0based(slope_row)
        };
        slope_row[j_piv] = 0.0;
        let rem_sum: f64 = slope_row.iter().sum();
        slope_row[j_piv] = -rem_sum;

        let half_dq = dq_old[k] * 0.5;
        let mut bad = false;
        for j in 0..species {
            let dxa = slope_row[j].abs() * half_dq;
            if xa_c0[base + j] + dxa > 1.0 || xa_c0[base + j] - dxa < 0.0 {
                bad = true;
                break;
            }
        }
        if bad {
            slope_row.fill(0.0);
        }
    }
    (xa_c0, xa_c1)
}

/// Integrates reconstructed polynomial `(c0, c1, c2)` over `[xq_outer, xq_outer + dq]`
/// (`get_xq_integral` in `star/private/mesh_adjust.f90:1810-1924`).
#[must_use]
pub fn get_xq_integral_col(
    k_old_in: usize,
    nz_old: usize,
    xq_old: &[f64],
    xq_outer: f64,
    dq: f64,
    order: usize,
    species: usize,
    j: usize,
    c0: &[f64],
    c1: &[f64],
    c2: Option<&[f64]>,
) -> f64 {
    let mut k_old = k_old_in.min(nz_old.saturating_sub(1));
    while k_old > 0 && xq_old[k_old] > xq_outer {
        k_old -= 1;
    }
    let mut xq_inner = xq_outer + dq;
    let mut old_xq_inner = xq_old[k_old];
    let mut sum_dqs = 0.0_f64;
    let mut integral = 0.0_f64;

    for k in k_old..nz_old {
        if dq <= sum_dqs {
            break;
        }
        let old_xq_outer = old_xq_inner;
        old_xq_inner = if k + 1 == nz_old { 1.0 } else { xq_old[k + 1] };
        let xq_overlap_outer = xq_outer.max(old_xq_outer);
        let mut xq_overlap_inner = xq_inner.min(old_xq_inner);

        let dq1 = if sum_dqs == 0.0
            && xq_overlap_outer == xq_outer
            && xq_overlap_inner == xq_inner
        {
            xq_inner = xq_outer + dq;
            xq_overlap_inner = xq_inner;
            dq
        } else if old_xq_inner >= xq_inner {
            dq - sum_dqs
        } else {
            (xq_overlap_inner - xq_overlap_outer).max(0.0)
        };
        sum_dqs += dq1;

        let idx = k * species + j;
        let a = c0[idx];
        let b = c1[idx];
        let c = if order == 2 {
            c2.map_or(0.0, |arr| arr[idx])
        } else {
            0.0
        };

        let avg_value = if dq1 == 0.0 || (b == 0.0 && c == 0.0) || order == 0 {
            a
        } else {
            let old_xq_mid = (old_xq_outer + old_xq_inner) * 0.5;
            let dq_outer = old_xq_mid - xq_overlap_outer;
            let dq_inner = old_xq_mid - xq_overlap_inner;
            if order == 1 || c == 0.0 {
                let v_outer = a + dq_outer * b;
                let v_inner = a + dq_inner * b;
                (v_outer + v_inner) * 0.5
            } else {
                a + b * (dq_inner + dq_outer) * 0.5
                    + c * (dq_inner * dq_inner
                        + dq_inner * dq_outer
                        + dq_outer * dq_outer)
                        / 6.0
            }
        };
        integral += dq1 * avg_value;
        if old_xq_inner >= xq_inner {
            break;
        }
    }
    integral
}

/// Integrates piecewise-constant cell profile `value_old` over
/// `[xq_outer, xq_outer + dq_range]` (`get_old_integral` in
/// `star/private/mesh_adjust.f90:1530-1624`).
#[must_use]
pub fn get_old_value_integral(
    k_old_in: usize,
    nz_old: usize,
    xq_old: &[f64],
    dq_old: &[f64],
    xq_outer: f64,
    dq_range: f64,
    value_old: &[f64],
) -> f64 {
    let mut k_old = k_old_in.min(nz_old.saturating_sub(1));
    while k_old > 0 && xq_old[k_old] > xq_outer {
        k_old -= 1;
    }
    let xq_inner = xq_outer + dq_range;
    let mut old_xq_inner = xq_old[k_old];
    let mut sum_dqs = 0.0_f64;
    let mut integral = 0.0_f64;

    for k in k_old..nz_old {
        if dq_range <= sum_dqs {
            break;
        }
        let old_xq_outer = old_xq_inner;
        old_xq_inner = if k + 1 == nz_old { 1.0 } else { xq_old[k + 1] };
        let val = value_old[k];

        if old_xq_inner <= xq_inner && old_xq_outer >= xq_outer {
            sum_dqs += dq_old[k];
            integral += val * dq_old[k];
        } else if old_xq_inner >= xq_inner && old_xq_outer <= xq_outer {
            sum_dqs = dq_range;
            integral = val * dq_range;
        } else if xq_inner <= old_xq_inner {
            integral += val * (dq_range - sum_dqs);
            sum_dqs = dq_range;
        } else {
            let dq_overlap = (old_xq_inner - xq_outer).max(0.0);
            sum_dqs += dq_overlap;
            integral += val * dq_overlap;
        }
    }
    integral
}

/// Solves EOS $\ln T(\rho, e, \mathbf{X})$ at a single zone (`set_lnT_for_energy_with_tol`
/// in `star/private/mesh_adjust.f90:1627-1702` and `solve_eos_given_DE` in
/// `star/private/eos_support.f90:106-143`).
///
/// # Errors
/// Returns [`StarError`] if EOS inversion fails or produces non-finite $\ln T$.
pub fn solve_cell_lnt_for_energy(
    model: &StarModel,
    xa_k: &[f64],
    log_rho: f64,
    energy: f64,
    lnt_guess: f64,
    tol: f64,
) -> Result<(f64, f64), StarError> {
    if energy <= 0.0 || !energy.is_finite() || !log_rho.is_finite() {
        return Err(StarError::MicrophysicsError(format!(
            "solve_cell_lnt_for_energy invalid inputs: logRho={log_rho}, energy={energy}"
        )));
    }
    let ht = crate::micro::builtin_helm_table()?;
    let log_e = flexmesa_math::log10(energy);
    let inv = eos_dt_get_t(
        ht,
        &model.chem_id,
        xa_k,
        log_rho,
        I_LN_E as isize,
        log_e * LN10,
        lnt_guess / LN10,
        tol,
        tol * LN10,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("solve_eos_given_DE failed: {e}"))
    })?;
    let lnt = inv.log10_result * LN10;
    let res_energy = flexmesa_math::exp(inv.eos.res[I_LN_E]);
    if !lnt.is_finite() || !res_energy.is_finite() {
        return Err(StarError::MicrophysicsError(format!(
            "solve_cell_lnt_for_energy non-finite output: lnT={lnt}"
        )));
    }
    Ok((lnt, res_energy))
}

pub(crate) fn dqbar_profile(dq: &[f64]) -> Vec<f64> {
    let n = dq.len();
    let mut dqbar = vec![0.0; n];
    if n == 0 {
        return dqbar;
    }
    if n == 1 {
        dqbar[0] = dq[0];
        return dqbar;
    }
    dqbar[0] = 0.5 * dq[0];
    for k in 1..(n - 1) {
        dqbar[k] = 0.5 * (dq[k - 1] + dq[k]);
    }
    dqbar[n - 1] = 0.5 * dq[n - 2] + dq[n - 1];
    dqbar
}

fn do_interp_pt_val(
    nz: usize,
    nz_old: usize,
    nzlo1: usize,
    nzhi1: usize,
    val_out: &mut [f64],
    val_old: &[f64],
    center_val: f64,
    xq_old_plus1: &[f64],
    xq_new_slice: &[f64],
    force_non_negative: bool,
) -> Result<(), StarError> {
    let mut val_old_plus1 = Vec::with_capacity(nz_old + 1);
    val_old_plus1.extend_from_slice(&val_old[..nz_old]);
    val_old_plus1.push(center_val);

    let val_new =
        interpolate_vector_pm(xq_old_plus1, &val_old_plus1, xq_new_slice)
            .map_err(|e| {
                StarError::InvalidInput(format!(
                    "mesh_adjust do_interp_pt_val failed: {e}"
                ))
            })?;

    for (idx, k1) in (nzlo1..=nzhi1).enumerate() {
        let mut v = val_new[idx];
        if force_non_negative && v < 0.0 {
            v = 0.0;
        }
        val_out[k1 - 1] = v;
    }

    if nzlo1 > 1 {
        val_out[..(nzlo1 - 1)].copy_from_slice(&val_old[..(nzlo1 - 1)]);
    }
    if nzhi1 < nz {
        let n = nz - nzhi1 - 1;
        for k in 0..=n {
            val_out[nz - 1 - k] = val_old[nz_old - 1 - k];
        }
    }
    Ok(())
}

/// Cell-midpoint `interp_pm` interpolation (`do_interp_cell_val` in
/// `star/private/mesh_adjust.f90:874-961`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if spline construction fails.
pub fn do_interp_cell_val(
    nz: usize,
    nz_old: usize,
    nzlo1: usize,
    nzhi1: usize,
    val_out: &mut [f64],
    val_old: &[f64],
    xq: &[f64],
    xq_old_plus1: &[f64],
    dq: &[f64],
    dq_old: &[f64],
) -> Result<(), StarError> {
    let n = nzhi1 - nzlo1 + 1;
    let mut val_old_plus1 = vec![0.0; nz_old + 1];
    let mut mid_xq_old_plus1 = vec![0.0; nz_old + 1];
    for k in 0..nz_old {
        val_old_plus1[k] = val_old[k];
        mid_xq_old_plus1[k] = xq_old_plus1[k] + 0.5 * dq_old[k];
    }
    val_old_plus1[nz_old] = val_old_plus1[nz_old - 1];
    mid_xq_old_plus1[nz_old] = 1.0;

    let mut mid_xq_new = vec![0.0; n];
    for i in 0..n {
        let idx = nzlo1 - 1 + i;
        mid_xq_new[i] = xq[idx] + 0.5 * dq[idx];
    }

    let val_new =
        interpolate_vector_pm(&mid_xq_old_plus1, &val_old_plus1, &mid_xq_new)
            .map_err(|e| {
            StarError::InvalidInput(format!(
                "mesh_adjust do_interp_cell_val failed: {e}"
            ))
        })?;

    for i in 0..n {
        val_out[nzlo1 - 1 + i] = val_new[i];
    }
    if nzlo1 > 1 {
        val_out[..(nzlo1 - 1)].copy_from_slice(&val_old[..(nzlo1 - 1)]);
    }
    if nzhi1 < nz {
        let tail = nz - nzhi1 - 1;
        for i in 0..=tail {
            val_out[nz - 1 - i] = val_old[nz_old - 1 - i];
        }
    }
    Ok(())
}

fn do_lnr_and_lnd(
    nz: usize,
    nz_old: usize,
    nzlo1: usize,
    nzhi1: usize,
    cell_type: &[CellType],
    comes_from: &[usize],
    xmstar: f64,
    r_center: f64,
    lnr_old: &[f64],
    lnd_old: &[f64],
    old_r: &[f64],
    old_rho: &[f64],
    dq: &[f64],
    xq: &[f64],
    xq_old_plus1: &[f64],
    lnr_out: &mut [f64],
    lnd_out: &mut [f64],
    density_new_out: &mut [f64],
) -> Result<(), StarError> {
    let interp_lo1 = 1_usize.max(nzlo1.saturating_sub(1));
    let interp_hi1 = nz.min(nzhi1 + 1);

    let mut vol_old_plus1 = vec![0.0; nz_old + 1];
    for k in 0..nz_old {
        let rk = old_r[k];
        vol_old_plus1[k] = FOUR_THIRDS_PI * rk * rk * rk;
    }
    let vol_center = FOUR_THIRDS_PI * r_center * r_center * r_center;
    vol_old_plus1[nz_old] = vol_center;

    for k in 1..=nz_old {
        if vol_old_plus1[k] >= vol_old_plus1[k - 1] {
            return Err(StarError::InvalidInput(format!(
                "do_lnR_and_lnd: non-decreasing old volume at k={}",
                k + 1
            )));
        }
    }
    for k in 1..nz {
        if xq[k] <= xq[k - 1] {
            return Err(StarError::InvalidInput(format!(
                "do_lnR_and_lnd: non-increasing xq at k={}",
                k + 1
            )));
        }
    }

    let interp_xq = &xq[(interp_lo1 - 1)..interp_hi1];
    let interp_vol_new =
        interpolate_vector_pm(xq_old_plus1, &vol_old_plus1, interp_xq)
            .map_err(|e| {
                StarError::InvalidInput(format!(
                    "mesh_adjust do_lnR_and_lnd interp failed: {e}"
                ))
            })?;

    let sz = nz.max(nz_old) + 2;
    let mut vol_new = vec![0.0; sz];
    vol_new[nz + 1] = vol_center;
    for (idx, k1) in (interp_lo1..=interp_hi1).enumerate() {
        vol_new[k1] = interp_vol_new[idx];
    }

    if interp_lo1 < interp_hi1 && vol_new[interp_lo1 + 1] >= vol_new[interp_lo1]
    {
        let v_hi2 = if interp_lo1 + 2 <= interp_hi1 {
            vol_new[interp_lo1 + 2]
        } else {
            vol_center
        };
        vol_new[interp_lo1 + 1] = (vol_new[interp_lo1] + v_hi2) * 0.5;
        if vol_new[interp_lo1 + 1] >= vol_new[interp_lo1] {
            return Err(StarError::InvalidInput(String::from(
                "do_lnR_and_lnd: Vol_new(interp_lo+1) >= Vol_new(interp_lo)",
            )));
        }
    }

    if interp_hi1 >= interp_lo1 + 2 {
        for k1 in (interp_lo1 + 1)..interp_hi1 {
            if vol_new[k1 + 1] >= vol_new[k1] || vol_new[k1] >= vol_new[k1 - 1]
            {
                let vol_min =
                    vol_new[k1 - 1].min(vol_new[k1]).min(vol_new[k1 + 1]);
                let vol_max =
                    vol_new[k1 - 1].max(vol_new[k1]).max(vol_new[k1 + 1]);
                if vol_min == vol_max
                    || !vol_min.is_finite()
                    || !vol_max.is_finite()
                {
                    return Err(StarError::InvalidInput(String::from(
                        "do_lnR_and_lnd: degenerate non-monotonic Vol_new",
                    )));
                }
                vol_new[k1 - 1] = vol_max;
                vol_new[k1] = (vol_max + vol_min) * 0.5;
                vol_new[k1 + 1] = vol_min;
                if vol_new[k1 + 1] >= vol_new[k1]
                    || vol_new[k1] >= vol_new[k1 - 1]
                {
                    return Err(StarError::InvalidInput(String::from(
                        "do_lnR_and_lnd: interpolation gave non-positive volume",
                    )));
                }
            }
        }
    }

    let mut new_r = vec![0.0; nz + 1];
    let mut density_new = vec![0.0; nz + 1];
    let set1_new_r = |k1: usize, vol_arr: &[f64]| -> f64 {
        if cell_type[k1 - 1] == CellType::Unchanged {
            old_r[comes_from[k1 - 1]]
        } else {
            flexmesa_math::pow(vol_arr[k1] / FOUR_THIRDS_PI, ONE_THIRD)
        }
    };

    new_r[nzlo1] = set1_new_r(nzlo1, &vol_new);
    let k_loop_end = nzhi1.min(nz.saturating_sub(1));
    for k1 in nzlo1..=k_loop_end {
        new_r[k1 + 1] = set1_new_r(k1 + 1, &vol_new);
        if cell_type[k1 - 1] == CellType::Unchanged {
            density_new[k1] = old_rho[comes_from[k1 - 1]];
            continue;
        }
        if new_r[k1] <= new_r[k1 + 1] {
            return Err(StarError::InvalidInput(format!(
                "do_lnR_and_lnd: new_r({k1}) <= new_r({})",
                k1 + 1
            )));
        }
        let cell_vol = vol_new[k1] - vol_new[k1 + 1];
        if cell_vol <= 0.0 || dq[k1 - 1] <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "do_lnR_and_lnd: non-positive cell_Vol={cell_vol} or dq={} at k={k1}",
                dq[k1 - 1]
            )));
        }
        density_new[k1] = (xmstar * dq[k1 - 1]) / cell_vol;
    }

    if nzlo1 > 1 {
        for k1 in 1..nzlo1 {
            new_r[k1] = old_r[k1 - 1];
            density_new[k1] = old_rho[k1 - 1];
            vol_new[k1] = FOUR_THIRDS_PI * new_r[k1] * new_r[k1] * new_r[k1];
        }
    }

    if nzhi1 < nz {
        let n = nz - nzhi1 - 1;
        for k in 0..=n {
            new_r[nz - k] = old_r[nz_old - 1 - k];
            density_new[nz - k] = old_rho[nz_old - 1 - k];
            let rk = new_r[nz - k];
            vol_new[nz - k] = FOUR_THIRDS_PI * rk * rk * rk;
        }
    } else {
        let denom = vol_new[nz] - vol_center;
        if denom <= 0.0 {
            return Err(StarError::InvalidInput(String::from(
                "do_lnR_and_lnd: Vol_new(nz) <= Vol_center",
            )));
        }
        density_new[nz] = (xmstar * dq[nz - 1]) / denom;
        new_r[nz] = flexmesa_math::pow(vol_new[nz] / FOUR_THIRDS_PI, ONE_THIRD);
    }

    for k1 in 1..=nz {
        let from_k = comes_from[k1 - 1];
        lnr_out[k1 - 1] = if new_r[k1] == old_r[from_k] {
            lnr_old[from_k]
        } else {
            flexmesa_math::log(new_r[k1])
        };
        lnd_out[k1 - 1] = if density_new[k1] == old_rho[from_k] {
            lnd_old[from_k]
        } else {
            flexmesa_math::log(density_new[k1])
        };
        density_new_out[k1 - 1] = density_new[k1];
    }
    Ok(())
}

/// Conservative overlap integration on `xout_old` / `xout_new` matching
/// `adjust1_omega` and `adjust1_v` (`include_upstream_gap_fill = true`) and
/// `adjust1_u` and `adjust1_etrb` (`include_upstream_gap_fill = false`) in
/// `star/private/mesh_adjust.f90:1955-2887`.
fn adjust1_control_volume(
    k1: usize,
    nz: usize,
    nz_old: usize,
    comes_from: &[usize],
    xout_old: &[f64],
    xout_new: &[f64],
    old_mass_wt: &[f64],
    new_point_wt: f64,
    old_val: &[f64],
    include_upstream_gap_fill: bool,
    divide_by_old_wt: bool,
) -> Result<(f64, f64), StarError> {
    let xq_outer = xout_new[k1 - 1];
    let xq_inner = if k1 < nz {
        xq_outer + new_point_wt
    } else {
        1.0
    };
    let contrib = |idx: usize, dq: f64| -> f64 {
        if divide_by_old_wt {
            (old_val[idx] * dq) / old_mass_wt[idx]
        } else {
            old_val[idx] * dq
        }
    };

    let mut dq_sum = 0.0_f64;
    let mut accum = 0.0_f64;
    let k_outer1 = if xq_outer >= xout_old[nz_old - 1] {
        nz_old
    } else if k1 == 1 {
        1
    } else {
        comes_from[k1 - 2] + 1
    };

    for kk1 in k_outer1..=nz_old {
        let xq1 = if kk1 == nz_old { 1.0 } else { xout_old[kk1] };
        if xq1 <= xq_outer {
            continue;
        }
        let xq0 = xout_old[kk1 - 1];
        if include_upstream_gap_fill && xq0 >= xq_inner {
            if dq_sum < new_point_wt && kk1 > 1 {
                let dq = new_point_wt - dq_sum;
                dq_sum = new_point_wt;
                accum += contrib(kk1 - 2, dq);
            }
            break;
        }
        if xq1 < xq_outer {
            return Err(StarError::InvalidInput(String::from(
                "mesh_adjust control-volume overlap xq1 < xq_outer",
            )));
        }

        if xq0 >= xq_outer && xq1 <= xq_inner {
            let mut dq = old_mass_wt[kk1 - 1];
            dq_sum += dq;
            if dq_sum > new_point_wt {
                // Upstream `star/private/mesh_adjust.f90:2035, 2284, 2518, 2811`
                // computes `dq = dq - (target - dq_sum)` when `dq_sum > target`
                // (deferred bug #244); reproduced verbatim for MESA 26.4.1 parity.
                dq -= new_point_wt - dq_sum;
                dq_sum = new_point_wt;
            }
            accum += contrib(kk1 - 1, dq);
        } else if xq0 <= xq_outer && xq1 >= xq_inner {
            let dq = new_point_wt;
            dq_sum += dq;
            accum += contrib(kk1 - 1, dq);
        } else {
            let dq = if xq_inner <= xq1 {
                let rem = new_point_wt - dq_sum;
                dq_sum = new_point_wt;
                rem
            } else {
                let mut ov = (xq1 - xq_outer).max(0.0);
                if dq_sum + ov > new_point_wt {
                    ov = new_point_wt - dq_sum;
                }
                dq_sum += ov;
                ov
            };
            accum += contrib(kk1 - 1, dq);
            if dq <= 0.0 {
                return Err(StarError::InvalidInput(format!(
                    "mesh_adjust control-volume non-positive partial overlap dq={dq} at k={k1}"
                )));
            }
        }

        if dq_sum >= new_point_wt {
            break;
        }
    }
    Ok((accum, dq_sum))
}

#[inline]
fn unchanged_face_for_vel(k: usize, cell_type: &[CellType]) -> bool {
    matches!(cell_type[k], CellType::Unchanged | CellType::Revised)
        && (k == 0 || cell_type[k - 1] == CellType::Unchanged)
}

pub(crate) fn execute_adjust_mesh_from_plan(
    model: &mut StarModel,
    plan: StandardMeshPlan,
) -> Result<(), StarError> {
    execute_adjust_mesh_from_plan_with(model, plan, true)
}

/// Executes the remap. With `finish = true` it ends by setting the variables
/// and storing the start and old values, so the model is complete for
/// stand-alone callers. With `finish = false`, as in the step loop, it ends
/// like upstream's `remesh` (`star/private/adjust_mesh.f90:36-512`): only
/// `need_to_setvars = .true.` is set, and `prepare_for_new_step` sets the
/// variables afterwards (`set_vars_if_needed`, `evolve.f90:1891`), before
/// `new_generation` stores the old values; `save_start_values` runs in
/// `do_struct_burn_mix` (`struct_burn_mix.f90:163`).
pub(crate) fn execute_adjust_mesh_from_plan_with(
    model: &mut StarModel,
    plan: StandardMeshPlan,
    finish: bool,
) -> Result<(), StarError> {
    let nz_old = plan.nz_old;
    let nz = plan.dq_new.len();
    let species = model.species;
    let mw = model.controls.mix_wind.clone();

    model.aux.mesh_adjust_ke_conservation = 0.0;
    model.aux.mesh_adjust_ie_conservation = 0.0;
    model.aux.mesh_adjust_pe_conservation = 0.0;
    model.aux.mesh_adjust_eturb_conservation = 0.0;

    if model.controls.rotation_flag
        && model.j_rot.iter().all(|&j| j == 0.0)
        && model.omega.iter().any(|&w| w != 0.0)
    {
        crate::rotation::update_rotation_distortion_factors(model);
    }
    let j_tot1 = total_angular_momentum(model);

    // `clean_up_fractions(1, nz, species, nz, s% xa, 10d0, 1d-2, ierr)`
    // (`adjust_mesh.f90:109-116`, `net/private/net_eval.f90:822-850`).
    if !plan.xa_already_cleaned {
        clean_up_fractions_remesh(&mut model.xa, nz_old, species)?;
    }

    let specific_pe_old: Vec<f64> =
        (0..nz_old).map(|k| cell_specific_pe(model, k)).collect();
    let specific_ke_old: Vec<f64> =
        (0..nz_old).map(|k| cell_specific_ke(model, k)).collect();
    model.aux.mesh_call_number += 1;

    let dq_old = plan.dq_old;
    let xq_old = plan.xq_old;
    let comes_from = plan.comes_from;
    let cell_type = plan.cell_type;

    let xa_old = model.xa.clone();
    let energy_old = model.energy.clone();
    let eta_old = model.eta.clone();
    let lnd_old = model.lnd.clone();
    let lnt_old = model.ln_t.clone();
    let lnr_old = model.ln_r.clone();
    let lum_old = model.lum.clone();
    let v_old = model.v.clone();
    let u_old = model.u.clone();
    let nvar = model.layout.nvar_hydro;
    let w_old: Vec<f64> = if let Some(iw) =
        model.layout.i_w.filter(|_| model.xh.len() >= nz_old * nvar)
    {
        (0..nz_old).map(|k| model.xh[k * nvar + iw]).collect()
    } else {
        (0..nz_old)
            .map(|k| {
                model
                    .aux
                    .rsp
                    .et_rsp
                    .get(k)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
                    .sqrt()
            })
            .collect()
    };
    let hp_face_old = model.aux.rsp.hp_face.clone();
    let alpha_rti_old = model.aux.alpha_rti.clone();
    let dpdr_drhodr_old = model.aux.dpdr_drhodr_info.clone();
    let j_rot_old = model.j_rot.clone();
    let d_omega_old = model.aux.rotmix.d_omega.clone();
    let mlt_vc_old = model.mlt_vc.clone();
    let d_mix_old = model.d_mix.clone();
    let old_r = model.r.clone();
    let old_rho = model.rho.clone();
    let (age_old, dt_old, mstar_old) =
        (model.star_age_old, model.dt_old, model.mstar_old);
    let prev_okay_mlt_vc = model.aux.okay_to_set_mlt_vc;
    let prev_have_new_generation = model.aux.mixing.have_new_generation;

    model.resize_zones(nz);
    // `resize_star_info_arrays` (`star/private/adjust_mesh.f90:319`) runs
    // `star_info_arrays` with `do_copy_pointers_and_resize`, which sets
    // `s% cgrav(1:nz) = standard_cgrav` (`star/private/alloc.f90:965-969`).
    // The remap below (`cell_specific_PE` in `do1_lnT`,
    // `star/private/mesh_adjust.f90:1465`; `adjust1_omega`, `:2100`)
    // therefore sees `standard_cgrav` in every zone, also when
    // `use_other_cgrav` is set; the hook's values return at the next
    // `set_hydro_vars`.
    model.cgrav.fill(STANDARD_CGRAV);
    // The same resize sets `s% alpha_mlt(1:nz) = s% mixing_length_alpha`
    // (`star/private/alloc.f90:980-984`); `other_alpha_mlt` values return at
    // the next `set_hydro_vars` (`star/private/hydro_vars.f90:571-579`).
    model
        .aux
        .mixing
        .alpha_mlt
        .fill(model.controls.mixing_length_alpha);
    model.dq.copy_from_slice(&plan.dq_new);
    fix_tail_dq(&mut model.dq)?;
    model.set_qs();
    model.set_xqs();
    for k in 1..nz {
        if model.xq[k] <= model.xq[k - 1] {
            return Err(StarError::InvalidInput(format!(
                "bad xq_new before do_mesh_adjust at k={}: {} <= {}",
                k + 1,
                model.xq[k],
                model.xq[k - 1]
            )));
        }
    }
    model.set_m_and_dm();

    for (k, &xk) in model.xq.iter().enumerate() {
        if !(0.0..=1.0).contains(&xk) {
            return Err(StarError::InvalidInput(format!(
                "do_mesh_adjust out-of-range xq[{}]={xk}",
                k + 1
            )));
        }
    }

    let mut nzlo1 = 0_usize;
    for (k, &ct) in cell_type.iter().enumerate() {
        if ct != CellType::Unchanged {
            nzlo1 = k + 1;
            break;
        }
    }
    if nzlo1 == 0 {
        nzlo1 = nz;
    }
    let mut nzhi1 = nzlo1;
    for k1 in (nzlo1..=nz).rev() {
        if cell_type[k1 - 1] != CellType::Unchanged {
            nzhi1 = k1;
            break;
        }
    }
    if nzhi1 < nz {
        nzhi1 += 1;
    }
    if nzlo1 > 1 {
        nzlo1 -= 1;
    }

    let nzlo_old0 = comes_from[nzlo1 - 1];
    let nzhi_old0 = if nzhi1 == nz {
        nz_old - 1
    } else {
        comes_from[nzhi1]
    };

    let dqbar = dqbar_profile(&model.dq);
    let dqbar_old = dqbar_profile(&dq_old);

    let mut xq_old_plus1 = Vec::with_capacity(nz_old + 1);
    xq_old_plus1.extend_from_slice(&xq_old);
    xq_old_plus1.push(1.0);

    let xq_new_slice = model.xq[(nzlo1 - 1)..nzhi1].to_vec();

    let mut xout_old = vec![0.0; nz_old];
    xout_old[0] = xq_old[0];
    for k in 1..nz_old {
        xout_old[k] = xout_old[k - 1] + dqbar_old[k - 1];
    }
    let mut xout_new = vec![0.0; nz];
    xout_new[0] = model.xq[0];
    for k in 1..nz {
        xout_new[k] = xout_new[k - 1] + dqbar[k - 1];
    }

    do_interp_pt_val(
        nz,
        nz_old,
        nzlo1,
        nzhi1,
        &mut model.lum,
        &lum_old,
        model.l_center,
        &xq_old_plus1,
        &xq_new_slice,
        false,
    )?;

    if model.controls.rti_flag {
        do_interp_pt_val(
            nz,
            nz_old,
            nzlo1,
            nzhi1,
            &mut model.aux.alpha_rti,
            &alpha_rti_old,
            0.0,
            &xq_old_plus1,
            &xq_new_slice,
            true,
        )?;
        do_interp_pt_val(
            nz,
            nz_old,
            nzlo1,
            nzhi1,
            &mut model.aux.dpdr_drhodr_info,
            &dpdr_drhodr_old,
            0.0,
            &xq_old_plus1,
            &xq_new_slice,
            true,
        )?;
    }

    let mut density_new = vec![0.0; nz];
    let dq_snap = model.dq.clone();
    let xq_snap = model.xq.clone();
    do_lnr_and_lnd(
        nz,
        nz_old,
        nzlo1,
        nzhi1,
        &cell_type,
        &comes_from,
        model.xmstar,
        model.r_center,
        &lnr_old,
        &lnd_old,
        &old_r,
        &old_rho,
        &dq_snap,
        &xq_snap,
        &xq_old_plus1,
        &mut model.ln_r,
        &mut model.lnd,
        &mut density_new,
    )?;

    if model.controls.v_flag {
        let mut old_ke = vec![0.0_f64; nz_old];
        let mut old_ke_tot = 0.0_f64;
        for k in 0..nz_old {
            old_ke[k] = dqbar_old[k] * v_old[k] * v_old[k];
            old_ke_tot += old_ke[k];
        }
        for k in 0..nz {
            if unchanged_face_for_vel(k, &cell_type) {
                model.v[k] = v_old[comes_from[k]];
                continue;
            }
            let (ke_sum, _dq_s) = adjust1_control_volume(
                k + 1,
                nz,
                nz_old,
                &comes_from,
                &xout_old,
                &xout_new,
                &dqbar_old,
                dqbar[k],
                &old_ke,
                true,
                true,
            )?;
            let mut v_new = (ke_sum / dqbar[k]).max(0.0).sqrt();
            if v_old[comes_from[k]] < 0.0 {
                v_new = -v_new;
            }
            model.v[k] = v_new;
        }
        let new_ke_tot: f64 = dqbar
            .iter()
            .zip(&model.v)
            .map(|(dqb, &v)| dqb * v * v)
            .sum();
        if (old_ke_tot - new_ke_tot).abs() > 1.0e-7 * new_ke_tot {
            return Err(StarError::InvalidInput(String::from(
                "failed kinetic-energy conservation check in mesh_adjust do_v",
            )));
        }
        let err = (old_ke_tot - new_ke_tot).abs()
            / new_ke_tot.max(old_ke_tot).max(1.0);
        model.aux.mesh_adjust_ke_conservation = err;
        if mw.trace_mesh_adjust_error_in_conservation && err > 1.0e-10 {
            return Err(StarError::InvalidInput(format!(
                "mesh_adjust KE conservation error {err} > 1e-10"
            )));
        }
    }

    if model.controls.u_flag {
        let mut old_ke = vec![0.0_f64; nz_old];
        let mut old_ke_tot = 0.0_f64;
        for k in 0..nz_old {
            old_ke[k] = dq_old[k] * u_old[k] * u_old[k];
            old_ke_tot += old_ke[k];
        }
        for k in 0..nz {
            if unchanged_face_for_vel(k, &cell_type) {
                model.u[k] = u_old[comes_from[k]];
                continue;
            }
            let (ke_sum, _dq_s) = adjust1_control_volume(
                k + 1,
                nz,
                nz_old,
                &comes_from,
                &xout_old,
                &xout_new,
                &dq_old,
                model.dq[k],
                &old_ke,
                false,
                true,
            )?;
            let mut u_new = (ke_sum / model.dq[k]).max(0.0).sqrt();
            if u_old[comes_from[k]] < 0.0 {
                u_new = -u_new;
            }
            model.u[k] = u_new;
        }
        let new_ke_tot: f64 =
            model.dq.iter().zip(&model.u).map(|(d, &u)| d * u * u).sum();
        let err = (old_ke_tot - new_ke_tot).abs()
            / new_ke_tot.max(old_ke_tot).max(1.0);
        model.aux.mesh_adjust_ke_conservation = err;
        if mw.trace_mesh_adjust_error_in_conservation && err > 1.0e-10 {
            return Err(StarError::InvalidInput(format!(
                "mesh_adjust KE (u_flag) conservation error {err} > 1e-10"
            )));
        }
    }

    if model.controls.rsp2_flag {
        let mut old_et = vec![0.0_f64; nz_old];
        let mut old_et_tot = 0.0_f64;
        for k in 0..nz_old {
            old_et[k] = dq_old[k] * flexmesa_math::pow2(w_old[k]);
            old_et_tot += old_et[k];
        }
        for k in 0..nz {
            if unchanged_face_for_vel(k, &cell_type) {
                let wk = w_old[comes_from[k]];
                let et = flexmesa_math::pow2(wk);
                model.aux.rsp.et_rsp[k] = et;
                model.aux.etrb[k] = et;
                continue;
            }
            let (et_sum, _dq_s) = adjust1_control_volume(
                k + 1,
                nz,
                nz_old,
                &comes_from,
                &xout_old,
                &xout_new,
                &dq_old,
                model.dq[k],
                &old_et,
                false,
                true,
            )?;
            let wk = (et_sum / model.dq[k]).max(0.0).sqrt();
            let et = flexmesa_math::pow2(wk);
            model.aux.rsp.et_rsp[k] = et;
            model.aux.etrb[k] = et;
        }
        let new_et_tot: f64 = model
            .dq
            .iter()
            .zip(&model.aux.rsp.et_rsp)
            .map(|(d, &et)| d * et)
            .sum();
        let err = (old_et_tot - new_et_tot).abs()
            / new_et_tot.max(old_et_tot).max(1.0);
        model.aux.mesh_adjust_eturb_conservation = err;
        if mw.trace_mesh_adjust_error_in_conservation && err > 1.0e-10 {
            return Err(StarError::InvalidInput(format!(
                "mesh_adjust turbulent-energy conservation error {err} > 1e-10"
            )));
        }

        let hp_center = hp_face_old.get(nz_old - 1).copied().unwrap_or(1.0);
        do_interp_pt_val(
            nz,
            nz_old,
            nzlo1,
            nzhi1,
            &mut model.aux.rsp.hp_face,
            &hp_face_old,
            hp_center,
            &xq_old_plus1,
            &xq_new_slice,
            false,
        )?;
    }

    do_interp_pt_val(
        nz,
        nz_old,
        nzlo1,
        nzhi1,
        &mut model.mlt_vc,
        &mlt_vc_old,
        0.0,
        &xq_old_plus1,
        &xq_new_slice,
        true,
    )?;
    do_interp_pt_val(
        nz,
        nz_old,
        nzlo1,
        nzhi1,
        &mut model.d_mix,
        &d_mix_old,
        0.0,
        &xq_old_plus1,
        &xq_new_slice,
        true,
    )?;

    let (xa_c0, xa_c1) = compute_upstream_xa_lpp(
        &dq_old,
        &xa_old,
        nz_old,
        species,
        nzlo_old0,
        nzhi_old0 + 1,
    );

    let total_internal_energy1: f64 =
        dq_old.iter().zip(&energy_old).map(|(d, e)| d * e).sum();

    for k in 0..nz {
        if matches!(cell_type[k], CellType::Unchanged | CellType::Revised) {
            let ko = comes_from[k];
            model.xa[k * species..(k + 1) * species]
                .copy_from_slice(&xa_old[ko * species..(ko + 1) * species]);
            continue;
        }
        let xq_outer = model.xq[k];
        let cell_dq = if k + 1 == nz {
            1.0 - xq_outer
        } else {
            model.dq[k]
        };
        let k_old_start = comes_from[k].saturating_sub(1);
        let row = &mut model.xa[k * species..(k + 1) * species];
        for (j, slot) in row.iter_mut().enumerate() {
            let tot_j = get_xq_integral_col(
                k_old_start,
                nz_old,
                &xq_old,
                xq_outer,
                cell_dq,
                2,
                species,
                j,
                &xa_c0,
                &xa_c1,
                None,
            );
            *slot = tot_j / cell_dq;
            if *slot > 1.0 + 1.0e-8 || *slot < -1.0e-8 || !slot.is_finite() {
                return Err(StarError::InvalidInput(format!(
                    "do_xa out-of-range abundance {} at zone {}",
                    *slot,
                    k + 1
                )));
            }
        }
        let xa_sum: f64 = row.iter().sum();
        if !xa_sum.is_finite() || (1.0 - xa_sum).abs() > 1.0e-3 {
            return Err(StarError::InvalidInput(format!(
                "do_xa sum deviation xa_sum={xa_sum} at zone {}",
                k + 1
            )));
        }
        for slot in row.iter_mut() {
            *slot /= xa_sum;
        }
    }

    let catalog = NuclideCatalog::builtin_ref();
    for k in 0..nz {
        let xa_k = &model.xa[k * species..(k + 1) * species];
        let comp = get_composition_info(
            catalog,
            &model.chem_id,
            xa_k,
            model.controls.use_mass_corrections,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "do_mesh_adjust composition error at zone {}: {e}",
                k + 1
            ))
        })?;
        model.xh_mass[k] = comp.xh.clamp(0.0, 1.0);
        model.yhe_mass[k] = comp.xhe.clamp(0.0, 1.0);
        model.z_mass[k] = comp.xz.clamp(0.0, 1.0);
        model.abar[k] = comp.abar;
        model.zbar[k] = comp.zbar;
        model.z2bar[k] = comp.z2bar;
        model.z53bar[k] = comp.z53bar;
        model.ye[k] = comp.ye;
        model.aux.mass_correction[k] = comp.mass_correction;
        model.r[k] = flexmesa_math::exp(model.ln_r[k]);
    }
    model.set_m_grav_and_grav();

    // Upstream runs `adjust1_omega` unconditionally for every face `k = 1..nz`
    // without a `cell_type` shortcut (`star/private/mesh_adjust.f90:232-236,
    // 1943-2112`), so `xout_new`/`dqbar` rounding enters `j_rot` even on
    // `CellType::Unchanged` faces.
    if model.controls.rotation_flag {
        let max_w = model.controls.mix_wind.w_div_wcrit_max;
        let max_w2 = model.controls.mix_wind.w_div_wcrit_max2;
        for k in 0..nz {
            let (j_tot, dq_s) = adjust1_control_volume(
                k + 1,
                nz,
                nz_old,
                &comes_from,
                &xout_old,
                &xout_new,
                &dqbar_old,
                dqbar[k],
                &j_rot_old,
                true,
                false,
            )?;
            let jr = if dq_s > 0.0 { j_tot / dq_s } else { 0.0 };
            // Verbatim `adjust1_omega` (`star/private/mesh_adjust.f90:2097-2103`):
            // compute `w_div_w_crit_roche` from `j_rot`, update `i_rot` from
            // `xh` (`ln_r`), and set `omega = j_rot / i_rot`.
            model.j_rot[k] = jr;
            let r00 = flexmesa_math::exp(model.ln_r[k]);
            model.aux.w_div_wc[k] = crate::hydro_rotation::w_div_w_roche_jrot(
                r00,
                model.m[k],
                model.j_rot[k],
                model.cgrav[k],
                max_w,
                max_w2,
                // Deferred #505: `w_div_wc_flag` not yet on `StarControls`.
                false,
            );
            crate::hydro_rotation::update1_i_rot_from_xh(model, k);
            model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
        }
        if model.controls.d_omega_flag && d_omega_old.len() >= nz_old {
            model.aux.rotmix.d_omega.resize(nz, 0.0);
            do_interp_pt_val(
                nz,
                nz_old,
                nzlo1,
                nzhi1,
                &mut model.aux.rotmix.d_omega,
                &d_omega_old,
                0.0,
                &xq_old_plus1,
                &xq_new_slice,
                true,
            )?;
        }
    }

    let mut energy_new = vec![0.0; nz];
    for k in 0..nz {
        let k_old = comes_from[k];
        if cell_type[k] == CellType::Unchanged {
            model.ln_t[k] = lnt_old[k_old];
            energy_new[k] = energy_old[k_old];
            model.energy[k] = energy_new[k];
            continue;
        }
        let xq_outer = model.xq[k];
        let cell_dq = model.dq[k];
        let avg_lnt = if cell_type[k] == CellType::Revised {
            lnt_old[k]
        } else {
            get_old_value_integral(
                k_old, nz_old, &xq_old, &dq_old, xq_outer, cell_dq, &lnt_old,
            ) / cell_dq
        };
        if !avg_lnt.is_finite() || !(0.0..=100.0).contains(&avg_lnt) {
            return Err(StarError::InvalidInput(format!(
                "do1_lnT out-of-range avg_lnT={avg_lnt} at zone {}",
                k + 1
            )));
        }

        if !mw.mesh_adjust_get_t_from_e
            || eta_old[k_old] >= ETA_LIMIT_FOR_MESH_ADJUST
        {
            model.ln_t[k] = avg_lnt;
            energy_new[k] = energy_old[k_old];
            model.energy[k] = energy_new[k];
            continue;
        }

        let avg_energy = if cell_type[k] == CellType::Revised {
            energy_old[k_old]
        } else {
            get_old_value_integral(
                k_old,
                nz_old,
                &xq_old,
                &dq_old,
                xq_outer,
                cell_dq,
                &energy_old,
            ) / cell_dq
        };

        let target_e =
            if mw.max_rel_delta_ie_for_mesh_total_energy_balance == 0.0 {
                avg_energy
            } else {
                let avg_pe = if cell_type[k] == CellType::Revised {
                    specific_pe_old[k_old]
                } else {
                    get_old_value_integral(
                        k_old,
                        nz_old,
                        &xq_old,
                        &dq_old,
                        xq_outer,
                        cell_dq,
                        &specific_pe_old,
                    ) / cell_dq
                };
                let avg_ke = if cell_type[k] == CellType::Revised {
                    specific_ke_old[k_old]
                } else {
                    get_old_value_integral(
                        k_old,
                        nz_old,
                        &xq_old,
                        &dq_old,
                        xq_outer,
                        cell_dq,
                        &specific_ke_old,
                    ) / cell_dq
                };
                let new_pe = cell_specific_pe(model, k);
                let new_ke = cell_specific_ke(model, k);
                let max_delta = avg_energy
                    * mw.max_rel_delta_ie_for_mesh_total_energy_balance;
                let mut delta_e = avg_pe + avg_ke - (new_pe + new_ke);
                if delta_e.abs() > max_delta {
                    delta_e = if delta_e < 0.0 {
                        -max_delta.abs()
                    } else {
                        max_delta.abs()
                    };
                }
                let cand = avg_energy + delta_e;
                if cand <= 0.0 {
                    avg_energy
                } else {
                    cand
                }
            };
        energy_new[k] = target_e;

        let rho_k = density_new[k];
        let log_rho = flexmesa_math::log10(rho_k);
        let xa_k = model.xa[k * species..(k + 1) * species].to_vec();
        let new_lnt = if let Ok((lnt_sol, _rev_e)) = solve_cell_lnt_for_energy(
            model, &xa_k, log_rho, target_e, avg_lnt, 1.0e-11,
        ) {
            lnt_sol
        } else {
            energy_new[k] = energy_old[k_old];
            avg_lnt
        };
        model.ln_t[k] = new_lnt;
        model.energy[k] = energy_new[k];
    }

    let total_internal_energy2: f64 =
        model.dq.iter().zip(&energy_new).map(|(d, e)| d * e).sum();
    let ie_err = (total_internal_energy1 - total_internal_energy2).abs()
        / total_internal_energy1
            .abs()
            .max(total_internal_energy2.abs())
            .max(1.0);
    model.aux.mesh_adjust_ie_conservation = ie_err;
    if mw.trace_mesh_adjust_error_in_conservation && ie_err > 1.0e-8 {
        return Err(StarError::InvalidInput(format!(
            "mesh_adjust IE conservation error {ie_err} > 1e-8"
        )));
    }

    // `check_species_conservation` (`mesh_adjust.f90:469-498`).
    for j in 0..species {
        let old_total: f64 = (0..nz_old)
            .map(|k| xa_old[k * species + j] * dq_old[k])
            .sum();
        if old_total < 1.0e-9 {
            continue;
        }
        let new_total: f64 = (0..nz)
            .map(|k| model.xa[k * species + j] * model.dq[k])
            .sum();
        if (new_total - old_total).abs() > 1.0e-4 {
            return Err(StarError::InvalidInput(format!(
                "check_species_conservation failed for species index {j}: old={old_total}, new={new_total}"
            )));
        }
    }

    if model.controls.rotation_flag {
        let j_tot2 = total_angular_momentum(model);
        if (j_tot2 - j_tot1).abs() > 1.0e-2 * j_tot1.abs().max(j_tot2.abs()) {
            return Err(StarError::InvalidInput(format!(
                "adjust_mesh failed angular momentum conservation: J_tot1={j_tot1}, J_tot2={j_tot2}"
            )));
        }
    }

    model.pack_xh();
    if finish {
        model.aux.okay_to_set_mlt_vc = false;
        // `skip_set_cz_bdy_mass = .true.` as in the `set_vars` of
        // `prepare_for_new_step` (`evolve.f90:377, 1891`, `hydro_vars.f90:81`).
        model.aux.mixing.have_new_generation = false;
        let hydro_res = set_hydro_vars(model, false, false);
        model.aux.okay_to_set_mlt_vc = prev_okay_mlt_vc;
        model.aux.mixing.have_new_generation = prev_have_new_generation;
        hydro_res?;
    } else {
        // `remesh` sets `need_to_setvars` (`adjust_mesh_split_merge.f90:505`,
        // `evolve.f90:1973`) and leaves the variables to `prepare_for_new_step`.
        model.aux.need_to_setvars = true;
    }

    if finish {
        model.save_start_values()?;
        model.save_old_values();
        model.star_age_old = age_old;
        model.dt_old = dt_old;
        model.mstar_old = mstar_old;
    }
    Ok(())
}

/// Conservatively projects `model` onto `dq_new` (`remesh` / `do_mesh_adjust`
/// in `star/private/adjust_mesh.f90` and `star/private/mesh_adjust.f90`, or
/// `remesh_split_merge` when `use_split_merge_amr` is active).
///
/// # Errors
/// Returns [`StarError`] if `dq_new` has fewer than 2 zones, conservation checks
/// fail, or EOS inversion encounters an invalid state.
pub fn remap_onto_dq(
    model: &mut StarModel,
    dq_new: &[f64],
) -> Result<(), StarError> {
    if let Some(outcome) = model.aux.planned_mesh.take() {
        match outcome {
            PlannedMeshOutcome::DeferredError(err) => return Err(err),
            PlannedMeshOutcome::SplitMergeAmr { raw_dq_old }
                if f64_slices_bit_eq(&raw_dq_old, &model.dq) =>
            {
                let _ =
                    crate::adjust_mesh_split_merge::remesh_split_merge(model)?;
                return Ok(());
            }
            PlannedMeshOutcome::Standard(plan)
                if plan.nz_old == model.nz
                    && f64_slices_bit_eq(&plan.raw_dq_old, &model.dq)
                    && (f64_slices_bit_eq(&plan.dq_new, dq_new)
                        || (plan.num_splits == 0
                            && plan.num_merges == 0
                            && f64_slices_bit_eq(
                                &plan.raw_dq_old,
                                dq_new,
                            ))) =>
            {
                // Upstream `star/private/adjust_mesh.f90:155-159, 280` copies
                // `mesh_max_allowed_ratio = s% mesh_max_allowed_ratio` before
                // clamping `s% mesh_max_allowed_ratio = 2.5d0`, so the first
                // remesh uses the unclamped value and subsequent calls see `2.5`
                // (`#250` under `#169`).
                if model.controls.mix_wind.mesh_max_allowed_ratio < 2.5 {
                    model.controls.mix_wind.mesh_max_allowed_ratio = 2.5;
                }
                return execute_adjust_mesh_from_plan(model, *plan);
            }
            _ => {}
        }
    }

    let plan = build_plan_for_given_dq(model, dq_new)?;
    execute_adjust_mesh_from_plan(model, plan)
}

/// Strips the outermost `nz_old - nz_new` zones from `model` without altering
/// interior zone values (`do_prune_mesh_surface` in
/// `star/private/mesh_adjust.f90:553-640`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `nz_new == 0` or `nz_new > model.nz`.
pub fn do_prune_mesh_surface(
    model: &mut StarModel,
    nz_new: usize,
) -> Result<(), StarError> {
    let nz_old = model.nz;
    if nz_new == 0 || nz_new > nz_old {
        return Err(StarError::InvalidInput(format!(
            "do_prune_mesh_surface invalid nz_new={nz_new} for nz_old={nz_old}"
        )));
    }
    let skip = nz_old - nz_new;
    if skip == 0 {
        return Ok(());
    }
    let species = model.species;
    let nvar = model.layout.nvar_hydro;
    let new_mstar = model.m[skip];
    let new_xmstar = new_mstar - model.m_center;
    if new_xmstar <= 0.0 || !new_xmstar.is_finite() {
        return Err(StarError::InvalidInput(format!(
            "do_prune_mesh_surface non-positive xmstar={new_xmstar} at skip={skip}"
        )));
    }
    let dm_kept: Vec<f64> = model.dm[skip..nz_old].to_vec();

    model
        .xa
        .copy_within((skip * species)..(nz_old * species), 0);
    if model.xh.len() >= nz_old * nvar {
        model.xh.copy_within((skip * nvar)..(nz_old * nvar), 0);
    }
    for vec in [
        &mut model.lnd,
        &mut model.ln_t,
        &mut model.ln_r,
        &mut model.lum,
        &mut model.v,
        &mut model.u,
        &mut model.m_grav,
        &mut model.r,
        &mut model.rmid,
        &mut model.rho,
        &mut model.t,
        &mut model.energy,
        &mut model.ln_e,
        &mut model.d_mix,
        &mut model.mlt_vc,
        &mut model.j_rot,
        &mut model.omega,
        &mut model.cgrav,
        &mut model.aux.mass_correction,
        &mut model.aux.i_rot,
        &mut model.aux.fp_rot,
        &mut model.aux.ft_rot,
        &mut model.aux.alpha_rti,
        &mut model.aux.dpdr_drhodr_info,
    ] {
        if vec.len() >= nz_old {
            vec.copy_within(skip..nz_old, 0);
        }
    }
    for vec in [
        &mut model.aux.rotmix.nu_st,
        &mut model.aux.rotmix.d_st,
        &mut model.aux.rotmix.d_dsi,
        &mut model.aux.rotmix.d_sh,
        &mut model.aux.rotmix.d_ssi,
        &mut model.aux.rotmix.d_es,
        &mut model.aux.rotmix.d_gsf,
        &mut model.aux.rotmix.d_omega,
    ] {
        if vec.len() >= nz_old {
            vec.copy_within(skip..nz_old, 0);
            vec.truncate(nz_new);
        }
    }
    model.resize_zones(nz_new);
    model.mstar = new_mstar;
    model.xmstar = new_xmstar;
    model.q[0] = 1.0;
    for k in 0..nz_new {
        model.dq[k] = dm_kept[k] / new_xmstar;
        if k > 0 {
            model.q[k] = model.q[k - 1] - model.dq[k - 1];
        }
    }
    model.dq[nz_new - 1] = model.q[nz_new - 1];
    model.xq[0] = 0.0;
    for k in 0..nz_new {
        model.dm[k] = model.dq[k] * new_xmstar;
        model.m[k] = model.q[k] * new_xmstar + model.m_center;
        if k + 1 < nz_new {
            model.xq[k + 1] = model.xq[k] + model.dq[k];
        }
    }
    model.set_dm_bar();
    model.set_m_grav_and_grav();
    model.unpack_xh();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_adjust::{try_plan_standard_mesh, MeshFunctionConfig};
    use crate::{load_zams_model, StarControls, StarJobControls};

    #[test]
    fn test_execute_adjust_mesh_for_step_interpolates_d_omega() {
        let mut model = load_zams_model(
            StarControls {
                initial_mass: 1.0,
                initial_z: 0.02,
                zbase: 0.02,
                okay_to_remesh: true,
                rotation_flag: true,
                d_omega_flag: true,
                mesh_delta_coeff: 0.5,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("load 1 Msun ZAMS");
        set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

        let nz_old = model.nz;
        let d_omega_orig: Vec<f64> = (0..nz_old)
            .map(|k| 1.0e6 + 500.0 * f64::from(u32::try_from(k).expect("k")))
            .collect();
        model.aux.rotmix.d_omega = d_omega_orig.clone();
        let d_surf_bits = d_omega_orig[0].to_bits();
        let d_core_bits = d_omega_orig[nz_old - 1].to_bits();

        let cfg = MeshFunctionConfig::from_controls(&model.controls);
        let plan = try_plan_standard_mesh(&model, &cfg)
            .expect("try_plan_standard_mesh")
            .expect("some mesh plan");
        assert!(plan.num_splits + plan.num_merges > 0);

        let nz = plan.dq_new.len();
        let mut nzlo1 = 0_usize;
        for (k, &ct) in plan.cell_type.iter().enumerate() {
            if ct != CellType::Unchanged {
                nzlo1 = k + 1;
                break;
            }
        }
        if nzlo1 == 0 {
            nzlo1 = nz;
        }
        let mut nzhi1 = nzlo1;
        for k1 in (nzlo1..=nz).rev() {
            if plan.cell_type[k1 - 1] != CellType::Unchanged {
                nzhi1 = k1;
                break;
            }
        }
        if nzhi1 < nz {
            nzhi1 += 1;
        }
        if nzlo1 > 1 {
            nzlo1 -= 1;
        }
        let mut xq_old_plus1 = Vec::with_capacity(nz_old + 1);
        xq_old_plus1.extend_from_slice(&plan.xq_old);
        xq_old_plus1.push(1.0);

        execute_adjust_mesh_from_plan_with(&mut model, plan, false)
            .expect("execute_adjust_mesh_from_plan_with step");
        assert_eq!(model.aux.rotmix.d_omega.len(), model.nz);
        assert!(model
            .aux
            .rotmix
            .d_omega
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0));
        assert_eq!(model.aux.rotmix.d_omega[0].to_bits(), d_surf_bits);
        assert_eq!(
            model.aux.rotmix.d_omega[model.nz - 1].to_bits(),
            d_core_bits
        );

        let xq_new_slice = model.xq[(nzlo1 - 1)..nzhi1].to_vec();
        let mut expected = vec![0.0_f64; model.nz];
        do_interp_pt_val(
            model.nz,
            nz_old,
            nzlo1,
            nzhi1,
            &mut expected,
            &d_omega_orig,
            0.0,
            &xq_old_plus1,
            &xq_new_slice,
            true,
        )
        .expect("reference do_interp_pt_val");
        for (k, (&got, &want)) in
            model.aux.rotmix.d_omega.iter().zip(&expected).enumerate()
        {
            assert_eq!(got.to_bits(), want.to_bits(), "d_omega[{k}] mismatch");
        }
    }
}
