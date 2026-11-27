// SPDX-License-Identifier: GPL-3.0-or-later
//! Tests of the call-by-call capture comparison (`#552`) on synthetic
//! captures. These check the matching and reporting logic only; no value
//! here comes from MESA.

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::unreadable_literal)]

use std::io::Cursor;

use flexmesa_replay::argdiff::compare_captures;
use flexmesa_replay::record::{encode_ad, encode_real, MesaAd, STAR_AD_LANES};

const SCHEMA: &str =
    "SCHEMA GETR site:i id:i model:i solver_iter:i k:i x:r y:a out_z:a\n";

fn ad(val: f64, lane: usize, d: f64) -> String {
    let mut d1 = [0.0; STAR_AD_LANES];
    d1[lane - 1] = d;
    encode_ad(&MesaAd::new(val, d1))
}

/// One record at `(model, iter, k)` with field values `x`, `y`, `z`.
fn rec(model: i64, iter: i64, k: i64, x: f64, y: &str, z: f64) -> String {
    format!(
        "GETR 5 1 {model} {iter} {k} {} {y} {}\n",
        encode_real(x),
        encode_real(z)
    )
}

/// A sweep over faces `1..=n` at `(model, iter)` with `x = k`.
fn sweep(model: i64, iter: i64, n: i64, scale: f64) -> String {
    (1..=n)
        .map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let x = k as f64 * scale;
            rec(model, iter, k, x, &ad(2.0, 5, 0.5), x + 1.0)
        })
        .collect()
}

fn capture(body: &str) -> String {
    format!("RUN\n{SCHEMA}{body}")
}

#[test]
fn identical_captures_match_every_call() {
    let body = sweep(1, 0, 4, 1.0) + &sweep(1, 1, 4, 1.0);
    let s = compare_captures(
        Cursor::new(capture(&body)),
        Cursor::new(capture(&body)),
    )
    .unwrap();
    let g = &s.families["GETR"];
    assert_eq!(
        (g.matched, g.identical, g.mesa_only.0, g.flexmesa_only.0),
        (8, 8, 0, 0)
    );
    assert!(s.all_identical());
}

#[test]
fn earliest_differing_argument_is_reported_by_sweep_then_face() {
    let mesa = sweep(1, 0, 4, 1.0) + &sweep(1, 1, 4, 1.0);
    // FlexMESA differs at k=3 of iteration 0 (argument x) and at k=1 of
    // iteration 1 (lane 5 of y); the earlier sweep wins.
    let flex = sweep(1, 0, 4, 1.0).replace(
        &rec(1, 0, 3, 3.0, &ad(2.0, 5, 0.5), 4.0),
        &rec(1, 0, 3, 3.0000000000000004, &ad(2.0, 5, 0.5), 4.0),
    ) + &sweep(1, 1, 4, 1.0).replacen(
        &ad(2.0, 5, 0.5),
        &ad(2.0, 5, 0.5000000000000001),
        1,
    );
    let s = compare_captures(
        Cursor::new(capture(&mesa)),
        Cursor::new(capture(&flex)),
    )
    .unwrap();
    let (_, ctx, rd) = s.first.as_ref().unwrap();
    assert!(ctx.contains("solver_iter=0 k=3"), "{ctx}");
    assert_eq!(rd.diff.mismatches[0].field, "x");
    assert_eq!(rd.diff.mismatches[0].lane, None);
    let y = &s.fields["GETR.y"];
    assert_eq!((y.val_calls, y.d1_calls), (0, 1));
    assert!(!s.all_identical());
}

#[test]
fn extra_sweep_on_one_side_is_reported_and_does_not_shift_matching() {
    // FlexMESA makes one extra sweep (different values) before the sweep
    // that matches MESA's only sweep at (model 0, iter 0).
    let mesa = sweep(0, 0, 3, 1.0);
    let flex = sweep(0, 0, 3, 7.0) + &sweep(0, 0, 3, 1.0);
    let s = compare_captures(
        Cursor::new(capture(&mesa)),
        Cursor::new(capture(&flex)),
    )
    .unwrap();
    let g = &s.families["GETR"];
    assert_eq!((g.matched, g.identical, g.flexmesa_only.0), (3, 3, 3));
    assert_eq!(s.sweep_notes.len(), 1);
    assert!(
        s.sweep_notes[0].contains("3 FlexMESA calls"),
        "{}",
        s.sweep_notes[0]
    );
    assert!(s.first.is_none());
}

#[test]
fn signed_zero_partials_count_as_identical_and_are_tallied() {
    let mesa = rec(1, 0, 1, 1.0, &ad(2.0, 30, -0.0), 3.0);
    let flex = rec(1, 0, 1, 1.0, &encode_real(2.0), 3.0);
    let s = compare_captures(
        Cursor::new(capture(&mesa)),
        Cursor::new(capture(&flex)),
    )
    .unwrap();
    assert_eq!(s.families["GETR"].identical, 1);
    assert_eq!(s.signed_zero_calls, 1);
    assert!(s.all_identical());
}

#[test]
fn calls_outside_the_other_window_are_not_differences() {
    let mesa = sweep(1, 0, 2, 1.0);
    let flex = sweep(1, 0, 2, 1.0) + &sweep(2, 0, 2, 1.0);
    let s = compare_captures(
        Cursor::new(capture(&mesa)),
        Cursor::new(capture(&flex)),
    )
    .unwrap();
    let g = &s.families["GETR"];
    assert_eq!(g.beyond_window, (0, 2));
    assert_eq!(g.flexmesa_only.0, 0);
    assert!(s.all_identical());
}

#[test]
fn field_written_as_not_available_is_listed_not_compared() {
    let mesa = rec(1, 0, 1, 1.0, &encode_real(2.0), 3.0);
    let flex = format!(
        "GETR 5 1 1 0 1 NA {} {}\n",
        encode_real(2.0),
        encode_real(3.0)
    );
    let s = compare_captures(
        Cursor::new(capture(&mesa)),
        Cursor::new(capture(&flex)),
    )
    .unwrap();
    assert_eq!(s.unavailable["GETR.x (FlexMESA)"], 1);
    assert_eq!(s.families["GETR"].identical, 1);
}
