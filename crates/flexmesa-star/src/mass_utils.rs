// SPDX-License-Identifier: GPL-3.0-or-later
//! Mass-coordinate bookkeeping in `real(qp)` (`star/private/mass_utils.f90`).
//!
//! When a star gains or loses mass its cells are redefined, and quantities
//! have to be moved between the old cells and the new ones. Working with
//! mass coordinates directly would subtract numbers as large as the stellar
//! mass to get cell-sized differences, so upstream works from the cell
//! masses `dm` in `real(qp)` and uses compensated sums
//! ([`flexmesa_num::AccurateReal`]). This module translates those routines
//! statement by statement with [`flexmesa_math::Quad`] (`DECISIONS #46`).
//!
//! # Indexing
//!
//! Upstream's arrays are one-based and some run from zero, with `0` meaning
//! "outside the star" and `-1` meaning "unset". To keep the translation
//! checkable against the Fortran, every slice here uses the **same indices
//! as upstream**: `dm[1..=nz]`, `mesh_intersects[1..=2*nz]`,
//! `ranges[1..=2*nz]`, `i_min[0..=nz]`, and so on, with element `0` unused
//! where upstream's array starts at one. The helpers [`one_based`] and
//! [`zero_slot`] build such arrays.

use flexmesa_math::Quad;
use flexmesa_num::{neumaier_sum_qp, AccurateReal};

/// Returns `values` shifted to one-based indexing, with an unused zero in
/// element `0`.
#[must_use]
pub fn one_based(values: &[f64]) -> Vec<Quad> {
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(Quad::ZERO);
    out.extend(values.iter().map(|&v| Quad::from(v)));
    out
}

/// Returns a one-based array of `n` zeros (length `n + 1`).
#[must_use]
pub fn zero_slot(n: usize) -> Vec<Quad> {
    vec![Quad::ZERO; n + 1]
}

/// `reconstruct_m` (`mass_utils.f90:48-64`): `m(j) = Sum_{l >= j} dm(l)`.
#[must_use]
pub fn reconstruct_m(dm: &[Quad], nz: usize, j: usize) -> Quad {
    let (mut sum, mut compensator) = (Quad::ZERO, Quad::ZERO);
    for l in (j..=nz).rev() {
        neumaier_sum_qp(&mut sum, &mut compensator, dm[l]);
    }
    sum + compensator
}

/// `reconstruct_xm` (`mass_utils.f90:69-85`): `xm(j) = Sum_{l <= j} dm(l)`.
#[must_use]
pub fn reconstruct_xm(dm: &[Quad], j: usize) -> Quad {
    let (mut sum, mut compensator) = (Quad::ZERO, Quad::ZERO);
    for &d in &dm[1..=j] {
        neumaier_sum_qp(&mut sum, &mut compensator, d);
    }
    sum + compensator
}

/// `accurate_mass_difference` (`mass_utils.f90:120-154`): `m1(j) - m2(k)`
/// with `m_i(j) = Sum_{l >= j} dm_i(l)`, for `j` and `k` in `1..=nz + 1`.
#[must_use]
pub fn accurate_mass_difference(
    dm1: &[Quad],
    dm2: &[Quad],
    j: usize,
    k: usize,
    nz: usize,
) -> Quad {
    let mut sum = AccurateReal::ZERO;
    let hi = j.max(k);
    if hi <= nz {
        for l in (hi..=nz).rev() {
            sum = sum + dm1[l];
            sum = sum - dm2[l];
        }
    }
    if j != k {
        for l in (j.min(k)..hi).rev() {
            let summand = if j < k { dm1[l] } else { -dm2[l] };
            sum = sum + summand;
        }
    }
    sum.value()
}

/// Overlaps between two meshes, from `make_compressed_intersect`.
#[derive(Clone, Debug)]
pub struct CompressedIntersect {
    /// `mesh_intersects(1:2*nz)`: overlap masses, element `0` unused.
    pub mesh_intersects: Vec<Quad>,
    /// `ranges(1:2*nz, 1:2)` as `[j, k]` pairs, element `0` unused. `j`
    /// indexes `dm1` and `k` indexes `dm2`; `0` means the mass lies above
    /// every cell of the other mesh, and `-1` marks an entry that was never
    /// written.
    pub ranges: Vec<[i32; 2]>,
}

/// `make_compressed_intersect` (`mass_utils.f90:195-349`): the size of the
/// overlap between cell `j` of `dm1` and cell `k` of `dm2` for the at most
/// `2*nz` pairs where it is not zero.
///
/// Upstream also maintains four running differences (`dTop`, `dBottom`,
/// `dTopBottom`, `dBottomTop`) that it never reads; they are not carried
/// here.
///
/// # Panics
///
/// Panics if `dm1` or `dm2` is shorter than `nz + 1` or `nz` is zero.
#[must_use]
pub fn make_compressed_intersect(
    dm1: &[Quad],
    dm2: &[Quad],
    nz: usize,
) -> CompressedIntersect {
    assert!(nz >= 1 && dm1.len() > nz && dm2.len() > nz);
    let mut mesh_intersects = zero_slot(2 * nz);
    let mut ranges = vec![[-1_i32, -1_i32]; 2 * nz + 1];
    // `remainders1(j)` is the mass in `dm1(j)` above all cells of `dm2`;
    // `remainders2(k)` likewise.
    let mut remainders1 = dm1[..=nz].to_vec();
    let mut remainders2 = dm2[..=nz].to_vec();

    let (mut j, mut k) = (nz, nz);
    let mut counter = 2 * nz;
    let idx = |v: usize| i32::try_from(v).expect("zone index fits in i32");

    let mut side;
    let mut diff;
    if dm1[j] > dm2[j] {
        side = 1;
        ranges[counter] = [idx(j), idx(k)];
        mesh_intersects[counter] = dm2[k];
        counter -= 1;
        remainders1[j] -= dm2[k];
        remainders2[j] -= dm2[k];
        diff = AccurateReal::from_quad(dm1[j] - dm2[k]);
    } else {
        side = 2;
        ranges[counter] = [idx(j), idx(k)];
        mesh_intersects[counter] = dm1[j];
        counter -= 1;
        remainders1[j] -= dm1[j];
        remainders2[j] -= dm1[j];
        diff = AccurateReal::from_quad(dm2[k] - dm1[j]);
    }

    // `do while (k > 0 .and. j > 0 .and. ((side == 1 .and. k > 1) .or.
    // (side == 2 .and. j > 1)))`
    while k > 0 && j > 0 && ((side == 1 && k > 1) || (side == 2 && j > 1)) {
        if side == 1 {
            k -= 1;
            ranges[counter] = [idx(j), idx(k)];
            // `dm2(k) < diff` compares with `diff%value()`.
            if dm2[k] < diff.value() {
                mesh_intersects[counter] = dm2[k];
                counter -= 1;
                remainders1[j] -= dm2[k];
                remainders2[k] -= dm2[k];
                diff = diff - dm2[k];
            } else {
                mesh_intersects[counter] = diff.value();
                counter -= 1;
                // `real(qp) - accurate_real`, assigned to `real(qp)`.
                remainders1[j] = diff.rsub(remainders1[j]).value();
                remainders2[k] = diff.rsub(remainders2[k]).value();
                diff = diff.rsub(dm2[k]);
                side = 2;
            }
        } else {
            j -= 1;
            ranges[counter] = [idx(j), idx(k)];
            if dm1[j] < diff.value() {
                mesh_intersects[counter] = dm1[j];
                counter -= 1;
                remainders1[j] -= dm1[j];
                remainders2[k] -= dm1[j];
                diff = diff - dm1[j];
            } else {
                mesh_intersects[counter] = diff.value();
                counter -= 1;
                remainders1[j] = diff.rsub(remainders1[j]).value();
                remainders2[k] = diff.rsub(remainders2[k]).value();
                diff = diff.rsub(dm1[j]);
                side = 1;
            }
        }
    }

    if j > 1 {
        for i in (1..=j).rev() {
            ranges[counter] = [idx(i), 0];
            mesh_intersects[counter] = remainders1[i];
            counter -= 1;
        }
    } else {
        for i in (1..=k).rev() {
            ranges[counter] = [0, idx(i)];
            mesh_intersects[counter] = remainders2[i];
            counter -= 1;
        }
    }
    CompressedIntersect {
        mesh_intersects,
        ranges,
    }
}

/// `find_i_ranges` (`mass_utils.f90:353-377`): for each `j` in `0..=nz`, the
/// smallest and largest `i = ranges(:, 2)` paired with it. A `j` with no
/// entry keeps `i_min = nz + 1` and `i_max = -1`.
///
/// # Panics
///
/// Panics if a `j` in `ranges` is negative, which upstream would read out of
/// bounds.
#[must_use]
pub fn find_i_ranges(nz: usize, ranges: &[[i32; 2]]) -> (Vec<i32>, Vec<i32>) {
    let unset = i32::try_from(nz + 1).expect("nz fits in i32");
    let mut i_min = vec![unset; nz + 1];
    let mut i_max = vec![-1_i32; nz + 1];
    for &[j, i] in &ranges[1..=2 * nz] {
        let j = usize::try_from(j).expect("ranges(:,1) is set and >= 0");
        if i < i_min[j] {
            i_min[j] = i;
        }
        if i > i_max[j] {
            i_max[j] = i;
        }
    }
    (i_min, i_max)
}

/// `type non_rect_array` rows: `arr(lo..)` for one `j`.
#[derive(Clone, Debug, Default)]
pub struct PassFractionRow {
    /// Lower bound `i_min(j)` of the row.
    pub lo: i32,
    /// `arr(i_min(j):i_max(j))`; empty when the row has no entries.
    pub arr: Vec<Quad>,
}

impl PassFractionRow {
    fn at(&self, i: i32) -> Quad {
        self.arr[usize::try_from(i - self.lo).expect("index in row")]
    }

    fn at_mut(&mut self, i: i32) -> &mut Quad {
        &mut self.arr[usize::try_from(i - self.lo).expect("index in row")]
    }
}

/// Output of [`prepare_pass_fraction`].
#[derive(Clone, Debug)]
pub struct PassFraction {
    /// `i_min(0:nz)`.
    pub i_min: Vec<i32>,
    /// `i_max(0:nz)`.
    pub i_max: Vec<i32>,
    /// `pf(0:nz)`.
    pub pf: Vec<PassFractionRow>,
}

/// `prepare_pass_fraction` (`mass_utils.f90:389-443`): the fraction of the
/// mass that ends in cell `j` (or, for `j = 0`, of the mass `delta_m` that
/// leaves or enters the star) which passes through cell `i` on the way.
#[must_use]
pub fn prepare_pass_fraction(
    nz: usize,
    delta_m: Quad,
    dm: &[Quad],
    mesh_intersects: &[Quad],
    ranges: &[[i32; 2]],
) -> PassFraction {
    let (i_min, i_max) = find_i_ranges(nz, ranges);
    let mut pf: Vec<PassFractionRow> = (0..=nz)
        .map(|j| {
            let len = usize::try_from(i_max[j] - i_min[j] + 1).unwrap_or(0);
            PassFractionRow {
                lo: i_min[j],
                arr: vec![Quad::ZERO; len],
            }
        })
        .collect();

    let mut counter = 1;
    for j in 0..=nz {
        for i in i_min[j]..=i_max[j] {
            *pf[j].at_mut(i) = mesh_intersects[counter];
            counter += 1;
        }
    }

    for j in 0..=nz {
        let jj = i32::try_from(j).expect("nz fits in i32");
        for i in (i_min[j] + 1)..=jj.min(i_max[j]) {
            let below = pf[j].at(i - 1);
            *pf[j].at_mut(i) = pf[j].at(i) + below;
        }
        for i in (jj.max(i_min[j])..i_max[j]).rev() {
            let above = pf[j].at(i + 1);
            *pf[j].at_mut(i) = pf[j].at(i) + above;
        }
    }

    for j in 0..=nz {
        for l in i_min[j]..=i_max[j] {
            let value = pf[j].at(l);
            *pf[j].at_mut(l) = if j > 0 {
                value / dm[j.max(1)]
            } else {
                -value / delta_m
            };
        }
    }
    PassFraction { i_min, i_max, pf }
}

/// `compute_pass_fraction` (`mass_utils.f90:448-462`).
#[must_use]
pub fn compute_pass_fraction(i: i32, j: usize, p: &PassFraction) -> Quad {
    let jj = i32::try_from(j).expect("nz fits in i32");
    if i >= p.i_min[j] && i <= p.i_max[j] {
        p.pf[j].at(i)
    } else if i > jj.max(p.i_max[j]) || i < jj.min(p.i_min[j]) {
        Quad::ZERO
    } else {
        Quad::ONE
    }
}

/// `find_j_ranges` (`mass_utils.f90:467-542`): for each cell `i` in
/// `1..=nz`, the range of destination cells `j` whose mass passes through
/// it. `mass_flux` is indexed `1..=nz + 1`. The returned arrays are indexed
/// `1..=nz` with element `0` unused.
#[must_use]
pub fn find_j_ranges(
    nz: usize,
    ranges: &[[i32; 2]],
    mass_flux: &[Quad],
) -> (Vec<i32>, Vec<i32>) {
    let zero = Quad::ZERO;
    let unset = i32::try_from(nz + 1).expect("nz fits in i32");
    let mut j_min = vec![unset; nz + 1];
    let mut j_max = vec![-1_i32; nz + 1];
    for i in 1..=nz {
        let ii = i32::try_from(i).expect("nz fits in i32");
        if mass_flux[i] >= zero && mass_flux[i + 1] >= zero {
            j_min[i] = ii;
        }
        if mass_flux[i] <= zero && mass_flux[i + 1] <= zero {
            j_max[i] = ii;
        }
        if mass_flux[i] >= zero && mass_flux[i + 1] <= zero {
            j_min[i] = ii;
            j_max[i] = ii;
        }
    }
    for &[j, i] in &ranges[1..=2 * nz] {
        if i > 0 {
            let i = usize::try_from(i).expect("positive index");
            if mass_flux[i] >= zero && mass_flux[i + 1] >= zero && j > j_max[i]
            {
                j_max[i] = j;
            }
            if mass_flux[i] <= zero && mass_flux[i + 1] <= zero && j < j_min[i]
            {
                j_min[i] = j;
            }
            if mass_flux[i] <= zero && mass_flux[i + 1] >= zero {
                if j < j_min[i] {
                    j_min[i] = j;
                }
                if j > j_max[i] {
                    j_max[i] = j;
                }
            }
        }
    }
    (j_min, j_max)
}

/// `integrate_conserved` (`mass_utils.f90:547-592`): moves cell averages
/// `vals_old` on mesh `dm_old` to mesh `dm_new`, taking `vals_outside` for
/// mass that was not in the star before.
///
/// `vals_old[k][l]` and the result `vals_new[j][l]` are indexed by cell
/// (`1..=nz`, row `0` unused) and variable (`0..nvars`). The values are
/// `real(dp)`: each product `mesh_intersects * vals_old` is formed in
/// `real(qp)` and each running total is converted back to `f64` on
/// assignment, as upstream's `real(dp)` array does.
#[must_use]
pub fn integrate_conserved(
    vals_old: &[Vec<f64>],
    vals_outside: &[f64],
    dm_new: &[Quad],
    dm_old: &[Quad],
    nz: usize,
    nvars: usize,
) -> Vec<Vec<f64>> {
    let CompressedIntersect {
        mesh_intersects,
        ranges,
    } = make_compressed_intersect(dm_new, dm_old, nz);
    let mut vals_new = vec![vec![0.0_f64; nvars]; nz + 1];
    for i in 1..=2 * nz {
        let [j, k] = ranges[i];
        if j == 0 {
            continue;
        }
        let j = usize::try_from(j).expect("ranges(:,1) is set and >= 0");
        for l in 0..nvars {
            let source = if k > 0 {
                vals_old[usize::try_from(k).expect("positive index")][l]
            } else {
                vals_outside[l]
            };
            vals_new[j][l] = (Quad::from(vals_new[j][l])
                + mesh_intersects[i] * Quad::from(source))
            .to_f64();
        }
    }
    for (row, &dm) in vals_new.iter_mut().zip(dm_new.iter()).skip(1) {
        for v in row.iter_mut() {
            *v = (Quad::from(*v) / dm).to_f64();
        }
    }
    vals_new
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sum of the overlaps involving cell `j` of mesh 1 (or cell `k` of mesh
    /// 2 when `col == 1`).
    fn overlap_sum(ci: &CompressedIntersect, nz: usize, col: usize, cell: i32) -> Quad {
        let mut acc = AccurateReal::ZERO;
        for i in 1..=2 * nz {
            if ci.ranges[i][col] == cell {
                acc = acc + ci.mesh_intersects[i];
            }
        }
        acc.value()
    }

    /// The overlap recorded for the pair `(j, k)`, or zero.
    fn pair(ci: &CompressedIntersect, nz: usize, j: i32, k: i32) -> Quad {
        (1..=2 * nz)
            .filter(|&i| ci.ranges[i] == [j, k])
            .map(|i| ci.mesh_intersects[i])
            .fold(Quad::ZERO, |a, b| a + b)
    }

    /// The part of old cell `k` recorded as leaving the star (`j == 0`).
    fn lost_from(ci: &CompressedIntersect, nz: usize, k: i32) -> Quad {
        pair(ci, nz, 0, k)
    }

    #[test]
    fn test_reconstruct_and_difference_agree() {
        let dm1 = one_based(&[3.0, 2.0, 1.5, 0.5]);
        let dm2 = one_based(&[2.5, 2.5, 1.0, 1.0]);
        let nz = 4;
        assert_eq!(reconstruct_m(&dm1, nz, 1), Quad::from(7.0));
        assert_eq!(reconstruct_m(&dm1, nz, 3), Quad::from(2.0));
        assert_eq!(reconstruct_xm(&dm1, 2), Quad::from(5.0));
        for j in 1..=nz + 1 {
            for k in 1..=nz + 1 {
                let want = reconstruct_m(&dm1, nz, j) - reconstruct_m(&dm2, nz, k);
                assert_eq!(accurate_mass_difference(&dm1, &dm2, j, k, nz), want);
            }
        }
    }

    #[test]
    fn test_identical_meshes_overlap_cell_by_cell() {
        let dm = one_based(&[4.0, 3.0, 2.0, 1.0]);
        let nz = 4;
        let ci = make_compressed_intersect(&dm, &dm, nz);
        for cell in 1..=4_i32 {
            let c = usize::try_from(cell).unwrap();
            assert_eq!(overlap_sum(&ci, nz, 0, cell), dm[c]);
            assert_eq!(overlap_sum(&ci, nz, 1, cell), dm[c]);
        }
        // Cell j of one mesh only overlaps cell j of the other.
        for i in 1..=2 * nz {
            let [j, k] = ci.ranges[i];
            if j > 0 && k > 0 && j != k {
                assert_eq!(ci.mesh_intersects[i], Quad::ZERO);
            }
        }
    }

    #[test]
    fn test_overlaps_for_mass_loss() {
        // The new mesh (1) has lost 1.25 from the old (2), all of it from
        // old cell 1.
        let dm1 = one_based(&[1.0, 1.5, 1.5, 1.0]);
        let dm2 = one_based(&[2.0, 1.75, 1.5, 1.0]);
        let nz = 4;
        let ci = make_compressed_intersect(&dm1, &dm2, nz);
        assert!(ci.ranges[1..].iter().all(|r| r[0] >= 0 && r[1] >= 0));
        for cell in 1..=4_i32 {
            let c = usize::try_from(cell).unwrap();
            // Every new cell is made of old material; every old cell is
            // either in a new cell or lost (j = 0).
            assert_eq!(overlap_sum(&ci, nz, 0, cell), dm1[c]);
            let kept = overlap_sum(&ci, nz, 1, cell);
            let want = if cell == 1 { Quad::from(0.75) } else { dm2[c] };
            assert_eq!(kept - lost_from(&ci, nz, cell), want);
        }
        assert_eq!(overlap_sum(&ci, nz, 0, 0), Quad::from(1.25));
        // New cell 1 holds 0.75 of old cell 1 and 0.25 of old cell 2.
        assert_eq!(pair(&ci, nz, 1, 1), Quad::from(0.75));
        assert_eq!(pair(&ci, nz, 1, 2), Quad::from(0.25));
    }

    #[test]
    fn test_overlaps_for_accretion_spanning_more_than_one_new_cell() {
        // The new mesh (1) has 1.25 more than the old (2): new cell 1 and a
        // quarter of new cell 2 are new material (k = 0).
        let dm1 = one_based(&[1.0, 1.0, 1.5, 1.0]);
        let dm2 = one_based(&[0.25, 0.5, 1.5, 1.0]);
        let nz = 4;
        let ci = make_compressed_intersect(&dm1, &dm2, nz);
        assert!(ci.ranges[1..].iter().all(|r| r[0] >= 0 && r[1] >= 0));
        for cell in 1..=4_i32 {
            let c = usize::try_from(cell).unwrap();
            assert_eq!(overlap_sum(&ci, nz, 0, cell), dm1[c]);
            assert_eq!(overlap_sum(&ci, nz, 1, cell), dm2[c]);
        }
        assert_eq!(pair(&ci, nz, 1, 0), Quad::from(1.0));
        assert_eq!(pair(&ci, nz, 2, 0), Quad::from(0.25));
        assert_eq!(overlap_sum(&ci, nz, 1, 0), Quad::from(1.25));
    }

    #[test]
    fn test_accretion_inside_the_outermost_new_cell_is_not_recorded() {
        // When the walk ends with `j == 1`, upstream writes the `(0, k)`
        // entries (`mass_utils.f90:335-348`), so the 1.25 of new material in
        // new cell 1 gets no `(1, 0)` entry. The pass fractions do not need
        // it; `integrate_conserved`, which upstream never calls, would.
        let dm1 = one_based(&[2.0, 1.75, 1.5, 1.0]);
        let dm2 = one_based(&[1.0, 1.5, 1.5, 1.0]);
        let nz = 4;
        let ci = make_compressed_intersect(&dm1, &dm2, nz);
        assert_eq!(overlap_sum(&ci, nz, 0, 1), Quad::from(0.75));
        assert_eq!(overlap_sum(&ci, nz, 1, 0), Quad::ZERO);
        for cell in 2..=4_i32 {
            let c = usize::try_from(cell).unwrap();
            assert_eq!(overlap_sum(&ci, nz, 0, cell), dm1[c]);
        }
    }

    #[test]
    fn test_integrate_conserved_on_equal_meshes_mass_loss_and_accretion() {
        let nz = 4;
        let vals_old: Vec<Vec<f64>> =
            vec![vec![0.0], vec![8.0], vec![4.0], vec![2.0], vec![1.0]];
        let outside = [16.0];

        let dm = one_based(&[1.0, 1.5, 1.5, 1.0]);
        let same = integrate_conserved(&vals_old, &outside, &dm, &dm, nz, 1);
        for k in 1..=nz {
            assert_eq!(same[k][0].to_bits(), vals_old[k][0].to_bits());
        }

        // Mass loss: new cell 1 is 0.75 of old cell 1 and 0.25 of old cell 2.
        let dm_new = one_based(&[1.0, 1.5, 1.5, 1.0]);
        let dm_old = one_based(&[2.0, 1.75, 1.5, 1.0]);
        let lost = integrate_conserved(&vals_old, &outside, &dm_new, &dm_old, nz, 1);
        assert_eq!(lost[1][0], 0.75 * 8.0 + 0.25 * 4.0);
        assert_eq!(lost[2][0], 4.0);
        assert_eq!(lost[4][0], 1.0);

        // Accretion over more than one new cell: new cell 1 is all outside
        // material; new cell 2 is 0.25 outside, 0.25 old cell 1, 0.5 old 2.
        let dm_new = one_based(&[1.0, 1.0, 1.5, 1.0]);
        let dm_old = one_based(&[0.25, 0.5, 1.5, 1.0]);
        let gained = integrate_conserved(&vals_old, &outside, &dm_new, &dm_old, nz, 1);
        assert_eq!(gained[1][0], 16.0);
        assert_eq!(gained[2][0], 0.25 * 16.0 + 0.25 * 8.0 + 0.5 * 4.0);
        assert_eq!(gained[3][0], 2.0);
    }

    #[test]
    fn test_pass_fractions_for_mass_loss() {
        let nz = 4;
        let dm_new = one_based(&[1.0, 1.5, 1.5, 1.0]);
        let dm_old = one_based(&[2.0, 1.75, 1.5, 1.0]);
        let ci = make_compressed_intersect(&dm_new, &dm_old, nz);
        // delta_m is negative for mass loss.
        let p = prepare_pass_fraction(
            nz,
            Quad::from(-1.25),
            &dm_new,
            &ci.mesh_intersects,
            &ci.ranges,
        );
        let f = |i: i32, j: usize| compute_pass_fraction(i, j, &p).to_f64();
        // All of the lost mass (j = 0) came through old cell 1 and none
        // through cell 2.
        assert_eq!((f(1, 0), f(2, 0)), (1.0, 0.0));
        // New cell 1: all of it passes through cell 1, and the quarter that
        // came from old cell 2 passes through cell 2.
        assert_eq!((f(1, 1), f(2, 1), f(3, 1)), (1.0, 0.25, 0.0));
        // Deeper cells did not move.
        for j in 2..=nz {
            for i in 1..=4_i32 {
                let want = if usize::try_from(i).unwrap() == j { 1.0 } else { 0.0 };
                assert_eq!(f(i, j), want, "pf({i},{j})");
            }
        }
        let (j_min, j_max) = find_j_ranges(
            nz,
            &ci.ranges,
            &one_based(&[-1.25, -0.25, 0.0, 0.0, 0.0]),
        );
        // Cell 1 feeds the lost mass (j = 0) and new cell 1; cell 2 feeds
        // new cells 1 and 2.
        assert_eq!((j_min[1], j_max[1]), (0, 1));
        assert_eq!((j_min[2], j_max[2]), (1, 2));
    }
}
