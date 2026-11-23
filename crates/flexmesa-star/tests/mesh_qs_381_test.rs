// SPDX-License-Identifier: LGPL-3.0-or-later
//! `set_qs` and `normalize_dqs` follow upstream `star/private/star_utils.f90`
//! (tag `26.4.1`, lines 669-752), and the mesh sync leaves a model's `dq`
//! untouched, as upstream's read path does (`#381`).
//!
//! Expected values are the upstream formulas evaluated by hand (exact
//! fractions) or, for `normalize_dqs`, in Python in upstream's summation
//! order; they are not `FlexMESA` output.

use flexmesa_star::{StarControls, StarJobControls, StarModel};

fn model_with_dq(dq: &[f64]) -> StarModel {
    let mut m = StarModel::new(
        dq.len(),
        vec![1],
        StarJobControls::default(),
        StarControls::default(),
    );
    m.dq.copy_from_slice(dq);
    m.mstar = 2.0e33;
    m.m_center = 0.0;
    m
}

#[test]
fn test_sync_mesh_and_pack_keeps_dq_bit_for_bit() {
    // dq that does not sum to one exactly: a renormalization would move it.
    let dq = [0.1, 0.3, 0.2, 0.15];
    let mut m = model_with_dq(&dq);
    m.sync_mesh_and_pack().expect("sync");
    for (k, (&got, &want)) in m.dq.iter().zip(dq.iter()).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "dq[{k}] changed");
    }
}

#[test]
fn test_set_qs_is_cumulative_from_the_surface_when_q_stays_in_unit_interval() {
    let mut m = model_with_dq(&[0.25, 0.5, 0.25]);
    m.set_qs();
    assert_eq!(m.q, vec![1.0, 0.75, 0.25]);
}

#[test]
fn test_set_qs_fallback_meets_at_first_maximum_like_fortran_maxloc() {
    // q(3) = 1 - 0.75 - 0.75 < 0, so upstream redoes q from both ends,
    // meeting at maxloc(dq) = the first of the two equal maxima (k = 1).
    // Surface inward: q(1) = 1. Center outward: q(3) = 0.125,
    // q(2) = 0.125 + 0.75 = 0.875.
    let mut m = model_with_dq(&[0.75, 0.75, 0.125]);
    m.set_qs();
    assert_eq!(m.midq_index(), 0);
    assert_eq!(m.q, vec![1.0, 0.875, 0.125]);
}

#[test]
fn test_normalize_dqs_divides_by_two_sided_sum_without_adjusting_largest_cell()
{
    // midq = 1 (0.3); dqsum1 = 0.1 + 0.3, dqsum2 = 0.15 + 0.2, sum = 0.75.
    let mut m = model_with_dq(&[0.1, 0.3, 0.2, 0.15]);
    m.normalize_dqs().expect("normalize");
    let want: [u64; 4] = [
        0x3FC1_1111_1111_1111,
        0x3FD9_9999_9999_9999,
        0x3FD1_1111_1111_1111,
        0x3FC9_9999_9999_9999,
    ];
    for (k, (&got, &w)) in m.dq.iter().zip(want.iter()).enumerate() {
        assert_eq!(got.to_bits(), w, "dq[{k}]");
    }
    // The pre-#381 largest-cell fix-up would have set dq[1] to
    // 1 - (dq[0] + (dq[2] + dq[3])) = 0x3FD999999999999A.
    assert_ne!(m.dq[1].to_bits(), 0x3FD9_9999_9999_999A);
}

#[test]
fn test_normalize_dqs_rejects_a_non_positive_cell() {
    let mut m = model_with_dq(&[0.5, 0.0, 0.5]);
    assert!(m.normalize_dqs().is_err());
}
