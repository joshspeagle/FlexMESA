// SPDX-License-Identifier: GPL-3.0-or-later
//! `remove_fallback_at_each_step`, `fallback_check_total_energy`, and
//! `remove_fallback_speed_limit` (`star/defaults/star_job.defaults:2142-2165`)
//! and `do_remove_fallback` (`star/private/remove_shells.f90:268-367`), called
//! from `before_step_loop` (`star/job/run_star_support.f90:596-599`).
//!
//! Indices in comments are 0-based: cell `nz - 1` is the innermost cell
//! (upstream `k = nz`). `do_remove_fallback` removes cells `k0..nz` through
//! `do_remove_inner_fraction_q(q(k0))` (`remove_shells.f90:362`), which keeps
//! cells `0..k0`, so the expected new `nz` is `k0` and the new `M_center` is
//! the old `m[k0]` (`remove_shells.f90:653-654`).

#![forbid(unsafe_code)]
// `0.5 * (a + b)` transcribes upstream `0.5d0*(rR + rL)` (`remove_shells.f90:308`).
#![allow(clippy::manual_midpoint)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_inventory::{STAR_JOB_SUPPORTED, STAR_JOB_UNUSED};
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::{
    do_remove_fallback, load_zams_model, StarControls, StarError,
    StarJobControls, StarModel,
};

fn default_zams() -> StarModel {
    load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS model")
}

/// A ZAMS model with `u_flag` on, `u = 0`, and `csound = 1e7` cm/s, so the
/// speed criterion depends only on the `u` profile a test sets.
fn u_flag_model() -> StarModel {
    let mut m = default_zams();
    m.controls.u_flag = true;
    m.u = vec![0.0; m.nz];
    m.csound = vec![1.0e7; m.nz];
    m
}

/// Upstream cell potential energy
/// `pe = -cgrav(k) * (m(k) - dm(k)/2) * dm(k) / ((r(k) + rL)/2)` with
/// `rL = R_center` for the innermost cell and `r(k+1)` otherwise
/// (`remove_shells.f90:304-311`), in erg.
fn cell_pe(m: &StarModel, k: usize) -> f64 {
    let r_l = if k == m.nz - 1 {
        m.r_center
    } else {
        m.r[k + 1]
    };
    let r_c = 0.5 * (m.r[k] + r_l);
    -m.cgrav[k] * (m.m[k] - 0.5 * m.dm[k]) * m.dm[k] / r_c
}

/// Sets `energy` so cell `k`'s total `ie + ke + pe` is `f[k] * scale` erg,
/// with `scale = max_k |pe_k|` (`u = 0`, so `ke = 0`). Cells not listed get
/// `f = +1` (unbound). Scaling by `max |pe|` keeps the rounding of
/// `(E - pe)/dm * dm - |pe|` (about `1e-16 * scale`) far below the margins
/// of the sign tests.
fn set_cell_energies(m: &mut StarModel, inner: &[(usize, f64)]) {
    m.controls.u_flag = true;
    m.u = vec![0.0; m.nz];
    let scale = (0..m.nz).map(|k| cell_pe(m, k).abs()).fold(0.0, f64::max);
    for k in 0..m.nz {
        let f = inner.iter().find(|&&(kk, _)| kk == k).map_or(1.0, |e| e.1);
        m.energy[k] = (f * scale - cell_pe(m, k)) / m.dm[k];
    }
}

#[test]
fn fallback_controls_have_upstream_defaults_and_parse_strictly() {
    for key in [
        "fallback_check_total_energy",
        "remove_fallback_at_each_step",
        "remove_fallback_speed_limit",
    ] {
        assert!(STAR_JOB_SUPPORTED.contains(&key), "`{key}` not supported");
        assert!(!STAR_JOB_UNUSED.contains(&key), "`{key}` listed unused");
    }
    assert!(STAR_JOB_SUPPORTED.windows(2).all(|w| w[0] < w[1]));

    // `star/defaults/star_job.defaults:2163-2165`.
    let d = StarJobControls::default();
    assert!(!d.remove_fallback_at_each_step);
    assert!(!d.fallback_check_total_energy);
    assert_eq!(d.remove_fallback_speed_limit.to_bits(), 0.1_f64.to_bits());

    let root = std::env::temp_dir().join(format!(
        "flexmesa_star_remove_fallback_parse_{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create temp dir");
    let inlist = root.join("inlist");
    fs::write(
        &inlist,
        "&star_job\n\
           remove_fallback_at_each_step = .true.\n\
           fallback_check_total_energy = .true.\n\
           remove_fallback_speed_limit = 0.25d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
         /\n",
    )
    .expect("write inlist");
    let parsed = InlistBundle::from_file(&inlist);
    let _ = fs::remove_file(&inlist);
    let _ = fs::remove_dir(&root);
    let bundle = parsed.expect("strict parse of the fallback controls");
    assert!(
        bundle.validation_report.unsupported_controls.is_empty(),
        "unexpected unsupported controls: {:?}",
        bundle.validation_report.unsupported_controls
    );
    assert!(bundle.job.remove_fallback_at_each_step);
    assert!(bundle.job.fallback_check_total_energy);
    assert_eq!(
        bundle.job.remove_fallback_speed_limit.to_bits(),
        0.25_f64.to_bits()
    );
}

#[test]
fn ccsn_iip_shock_part1_reads_shock_common_fallback_controls() {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/flexmesa/mesa_test_inputs")
        },
        PathBuf::from,
    );
    let path = base.join("star/test_suite/ccsn_IIp/inlist_shock_part1_header");
    assert!(
        path.is_file(),
        "missing MESA test input {}; run `flexmesa data fetch --include-test-inputs` or set MESA_DIR",
        path.display()
    );
    let bundle = InlistBundle::from_file(&path)
        .unwrap_or_else(|e| panic!("strict parse of {}: {e}", path.display()));
    assert!(
        bundle.validation_report.unsupported_controls.is_empty(),
        "ccsn_IIp shock part1 unsupported controls: {:?}",
        bundle.validation_report.unsupported_controls
    );
    // `ccsn_IIp/inlist_shock_common:11-14`.
    assert!(bundle.job.remove_fallback_at_each_step);
    assert!(bundle.job.fallback_check_total_energy);
    assert_eq!(
        bundle.job.remove_fallback_speed_limit.to_bits(),
        0.1_f64.to_bits()
    );
    assert!(bundle.job.remove_center_set_zero_v_center);
}

#[test]
fn speed_criterion_removes_inner_infall_run_and_respects_limit() {
    let mut extras = DefaultStarExtras;
    let base = u_flag_model();
    let nz = base.nz;
    assert!(nz > 20);

    // Inner 5 cells (nz-5..nz) fall in at 0.2 csound; cell nz-6 at exactly
    // 0.1 csound; the rest are at rest. With limit 0.1:
    // `-u(nz-1) = 0.2 cs > 0.1 cs`, so continue (`remove_shells.f90:351`);
    // the loop sets `k0 = k+1` and exits at the first `-u(k) < 0.1 cs`
    // (strict, `:354`). Cell nz-6 has `-u = 0.1 cs`, not `<`, so it is kept
    // in the run; cell nz-7 has `-u = 0 < 0.1 cs`, so `k0 = nz-6`.
    let mut infall = base.clone();
    for k in nz - 5..nz {
        infall.u[k] = -0.2 * infall.csound[k];
    }
    infall.u[nz - 6] = -(0.1 * infall.csound[nz - 6]);
    let want_k0 = nz - 6;
    let mass_at_cut = infall.m[want_k0];
    let radius_at_cut = infall.r[want_k0];
    let velocity_at_cut = infall.u[want_k0];
    let mut m = infall.clone();
    m.job.remove_fallback_speed_limit = 0.1;
    do_remove_fallback(&mut m, &mut extras).expect("speed criterion");
    assert_eq!(m.nz, want_k0);
    assert_eq!(m.m_center.to_bits(), mass_at_cut.to_bits());
    assert_eq!(m.r_center.to_bits(), radius_at_cut.to_bits());
    // `remove_center_set_zero_v_center = .false.`: `v_center = u(k0)`
    // (`remove_shells.f90:661-662`).
    assert_eq!(m.v_center.to_bits(), velocity_at_cut.to_bits());

    // `remove_center_set_zero_v_center` zeroes it (`remove_shells.f90:364`).
    let mut m_zero = infall.clone();
    m_zero.job.remove_center_set_zero_v_center = true;
    do_remove_fallback(&mut m_zero, &mut extras).expect("zero v_center");
    assert_eq!(m_zero.nz, want_k0);
    assert_eq!(m_zero.v_center.to_bits(), 0.0_f64.to_bits());

    // A larger limit: `-u(nz-1) = 0.2 cs <= 0.3 cs`, so nothing is removed
    // (`remove_shells.f90:351`).
    let mut m_hi = infall.clone();
    m_hi.job.remove_fallback_speed_limit = 0.3;
    do_remove_fallback(&mut m_hi, &mut extras).expect("limit 0.3");
    assert_eq!(m_hi.nz, nz);
    assert_eq!(m_hi.m_center.to_bits(), infall.m_center.to_bits());

    // Innermost infall exactly at the limit is not fallback (`<=`).
    let mut m_eq = base.clone();
    m_eq.u[nz - 1] = -(0.1 * m_eq.csound[nz - 1]);
    do_remove_fallback(&mut m_eq, &mut extras).expect("equal limit");
    assert_eq!(m_eq.nz, nz);
}

#[test]
fn velocity_source_follows_u_then_v_flag_and_noops_without_either() {
    let mut extras = DefaultStarExtras;
    let mut base = default_zams();
    let nz = base.nz;
    base.csound = vec![1.0e7; nz];
    base.u = vec![0.0; nz];
    base.v = vec![0.0; nz];
    base.v[nz - 3..].fill(-2.0e6);

    // Neither `u_flag` nor `v_flag`: return (`remove_shells.f90:290`).
    let mut m_none = base.clone();
    do_remove_fallback(&mut m_none, &mut extras).expect("no velocity");
    assert_eq!(m_none.nz, nz);

    // `v_flag`: `-v = 2e6 > 0.1 * 1e7` for the inner 3 cells, so `k0 = nz-3`.
    let mut m_v = base.clone();
    m_v.controls.v_flag = true;
    do_remove_fallback(&mut m_v, &mut extras).expect("v_flag");
    assert_eq!(m_v.nz, nz - 3);

    // `u_flag` takes precedence over `v_flag` (`remove_shells.f90:285-288`):
    // `u = 0`, so nothing is removed.
    let mut m_u = base;
    m_u.controls.v_flag = true;
    m_u.controls.u_flag = true;
    do_remove_fallback(&mut m_u, &mut extras).expect("u_flag");
    assert_eq!(m_u.nz, nz);
}

#[test]
fn total_energy_criterion_cuts_at_first_unbound_cell_above_bound_core() {
    let mut extras = DefaultStarExtras;
    let mut base = default_zams();
    base.job.fallback_check_total_energy = true;
    let nz = base.nz;

    // Cell totals in units of `scale` (innermost first):
    //   nz-1: +0.1, nz-2: +0.1, nz-3: -1.0, nz-4: -0.5, nz-5: +0.3, rest +1.
    // First loop (`remove_shells.f90:300-329`): running sum 0.1, 0.2, then
    // -0.8 at nz-3, so it exits with `k0 = nz-2` (last cell with sum >= 0).
    // Second loop from `k0-1 = nz-3` outward (`:334-346`): nz-3 (-1.0) and
    // nz-4 (-0.5) are bound; nz-5 (+0.3) is unbound, so `k0 = nz-4`.
    let mut m = base.clone();
    set_cell_energies(
        &mut m,
        &[
            (nz - 1, 0.1),
            (nz - 2, 0.1),
            (nz - 3, -1.0),
            (nz - 4, -0.5),
            (nz - 5, 0.3),
        ],
    );
    let mass_at_cut = m.m[nz - 4];
    do_remove_fallback(&mut m, &mut extras).expect("bound inner region");
    assert_eq!(m.nz, nz - 4);
    assert_eq!(m.m_center.to_bits(), mass_at_cut.to_bits());

    // Exit at the innermost cell keeps `k0 = nz-1` (`:296`); the second loop
    // starts at nz-2 (+0.5, unbound) and sets `k0 = nz-1`, so only the
    // innermost cell is removed.
    let mut m_inner = base.clone();
    set_cell_energies(&mut m_inner, &[(nz - 1, -1.0), (nz - 2, 0.5)]);
    do_remove_fallback(&mut m_inner, &mut extras).expect("innermost bound");
    assert_eq!(m_inner.nz, nz - 1);

    // Every cell unbound: the sum never goes negative, so there is no bound
    // inner region and nothing is removed (`:330-333`).
    let mut m_unbound = base.clone();
    set_cell_energies(&mut m_unbound, &[]);
    do_remove_fallback(&mut m_unbound, &mut extras).expect("all unbound");
    assert_eq!(m_unbound.nz, nz);

    // `remove_fallback_speed_limit` is read only on the speed path (`:350`):
    // a limit that would make that path return early (and `u = 0`) leaves
    // the energy path's cut unchanged.
    let mut m_limit = base.clone();
    m_limit.job.remove_fallback_speed_limit = 1.0e99;
    set_cell_energies(&mut m_limit, &[(nz - 1, -1.0), (nz - 2, 0.5)]);
    do_remove_fallback(&mut m_limit, &mut extras).expect("limit ignored");
    assert_eq!(m_limit.nz, nz - 1);

    // A non-finite cell energy is an error (`mesa_error`, `:313-318`).
    let mut m_bad = base;
    set_cell_energies(&mut m_bad, &[]);
    m_bad.energy[nz - 1] = f64::NAN;
    let err = do_remove_fallback(&mut m_bad, &mut extras)
        .expect_err("NaN energy must fail");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert_eq!(m_bad.nz, nz);
}
