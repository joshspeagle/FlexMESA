// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` invariant tests for the `#160`
//! `binary_core` modules (`mdot_implicit`, `edd_wind`, `orbit`,
//! `roche_deformation`, `jdot_tides`, `disk`, and `timestep`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::fmt::Write as _;

use flexmesa_autodiff::DualOrder1;
use flexmesa_binary::{
    bh_initial_equatorial_mass, binary_pick_next_timestep, build_anomaly_grid,
    check_binary_hard_limits, eval_accreted_material_j, eval_bh_spin,
    eval_bondi_hoyle_wind_fraction, eval_edot_tidal_hut, eval_j_orb,
    eval_jdot_gr, eval_jdot_ls, eval_jdot_ls_with_am_removed,
    eval_jdot_mb_with_scale_mass, eval_jdot_missing_wind, eval_jdot_ml,
    eval_k_div_t_convective, eval_k_div_t_radiative,
    eval_l2_mass_loss_fraction, eval_l2_overflow_limit, eval_mdot_edd,
    eval_mdot_edd_with_overrides, eval_point_mass_edd_and_retention,
    eval_rl_relative_gap, eval_rlobe, eval_rlobe_ad, eval_synchronicity_switch,
    eval_tout_wind_enhancement, eval_tout_wind_enhancement_orbit_averaged,
    eval_tsync, hut_f2, hut_f3, hut_f4, hut_f5, period_from_separation,
    separation_from_j_orb, separation_from_period,
    update_change_factor_after_step, BinaryHardLimitViolation,
    BinaryInlistControls, BinaryTimestepControls, ImplicitMdotSolver,
    ImplicitRlofSolver, ImplicitStepAction, RocheDeformationTables,
    RocheTable2D, B_TLIM_ENV, B_TLIM_JORB, B_TLIM_ROCHE, ROCHE_NUDGE,
};
use flexmesa_const::{
    CLIGHT, MSUN, ONE_THIRD, PI, RSUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{atan, cos, exp, log, log1p, pow, pow2, sin, tan};

fn ref_dsign1(v: f64) -> f64 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Independent full-array 1D Steffen cubic constructor mirroring
/// `interp_1d/private/interp_1d_pm.f90` (`mk_pmcub` / `mk_pmquad` /
/// `mk_pmlinear`).
fn ref_mk_pm(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    let mut c = vec![[0.0_f64; 4]; n];
    for i in 0..n {
        c[i][0] = y[i];
    }
    if n == 2 {
        let h = x[1] - x[0];
        c[0][1] = (y[1] - y[0]) / h;
        return c;
    }
    let mut h = vec![0.0_f64; n - 1];
    let mut s = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        h[i] = x[i + 1] - x[i];
        s[i] = (y[i + 1] - y[i]) / h[i];
    }
    for i in 1..(n - 1) {
        let p = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
        c[i][1] = (ref_dsign1(s[i - 1]) + ref_dsign1(s[i]))
            * s[i - 1].abs().min(s[i].abs()).min(0.5 * p.abs());
    }
    let p0 = s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1]);
    c[0][1] = if p0 * s[0] <= 0.0 {
        0.0
    } else if p0.abs() > 2.0 * s[0].abs() {
        2.0 * s[0]
    } else {
        p0
    };
    let m = n - 1;
    let pn = s[m - 1] * (1.0 + h[m - 1] / (h[m - 1] + h[m - 2]))
        - s[m - 2] * h[m - 1] / (h[m - 1] + h[m - 2]);
    c[m][1] = if pn * s[m - 1] <= 0.0 {
        0.0
    } else if pn.abs() > 2.0 * s[m - 1].abs() {
        2.0 * s[m - 1]
    } else {
        pn
    };
    for i in 0..(n - 1) {
        c[i][2] = (3.0 * s[i] - 2.0 * c[i][1] - c[i + 1][1]) / h[i];
        c[i][3] = (c[i][1] + c[i + 1][1] - 2.0 * s[i]) / (h[i] * h[i]);
    }
    c
}

/// Independent full-table analytical evaluator matching MESA `do_mkbipm_db` +
/// `do_evbipm_db` (`interp_2d/private/bipm_db.f90:28-118`).
fn ref_bipm_eval(table: &RocheTable2D, xget: f64, yget: f64) -> f64 {
    let nx = table.xs.len();
    let ny = table.ys.len();
    let x = if xget >= table.xs[nx - 1] {
        table.xs[nx - 1] - 1.0e-12
    } else if xget < table.xs[0] {
        table.xs[0]
    } else {
        xget
    };
    let y = if yget >= table.ys[ny - 1] {
        table.ys[ny - 1] - 2.0 * ROCHE_NUDGE
    } else if yget < table.ys[0] {
        table.ys[0]
    } else {
        yget
    };
    let ix = table
        .xs
        .partition_point(|&xv| xv <= x)
        .saturating_sub(1)
        .min(nx - 2);
    let iy = table
        .ys
        .partition_point(|&yv| yv <= y)
        .saturating_sub(1)
        .min(ny - 2);
    let (jlo, jhi) = if ny >= 4 {
        let lo = if iy == 0 {
            0
        } else if iy >= ny - 2 {
            ny - 4
        } else {
            iy - 1
        };
        (lo, lo + 3)
    } else {
        (0, ny - 1)
    };
    let dx = x - table.xs[ix];
    let mut y_sub = Vec::with_capacity(jhi - jlo + 1);
    let mut ff_sub = Vec::with_capacity(jhi - jlo + 1);
    for jy in jlo..=jhi {
        let mut row = Vec::with_capacity(nx);
        for jx in 0..nx {
            row.push(table.values[jx * ny + jy]);
        }
        let x_coeffs = ref_mk_pm(&table.xs, &row);
        let c = x_coeffs[ix];
        let val_x = c[0] + dx * (c[1] + dx * (c[2] + dx * c[3]));
        y_sub.push(table.ys[jy]);
        ff_sub.push(val_x);
    }
    let y_coeffs = ref_mk_pm(&y_sub, &ff_sub);
    let c = y_coeffs[iy - jlo];
    let dy = y - table.ys[iy];
    c[0] + dy * (c[1] + dy * (c[2] + dy * c[3]))
}

#[test]
fn test_mdot_implicit_unit_conversion_order_and_arras() {
    let mut ctrl = BinaryInlistControls {
        mdot_scheme: String::from("Arras"),
        solver_type: String::from("bisect"),
        implicit_scheme_tolerance: 1.0e-4,
        implicit_scheme_tiny_factor: 1.0e-6,
        starting_mdot: 1.3e-10,
        roche_min_mdot: 2.7e-14,
        min_mdot_for_implicit: 1.7e-15,
        max_implicit_abs_mdot: 3.1e-2,
        ..BinaryInlistControls::default()
    };
    let solver = ImplicitMdotSolver::from_controls(&ctrl, 1.25).unwrap();
    assert_eq!(
        solver.starting_mdot_cgs.to_bits(),
        (ctrl.starting_mdot * MSUN / SECYER).to_bits()
    );
    assert_eq!(
        solver.roche_min_mdot_cgs.to_bits(),
        (ctrl.roche_min_mdot * MSUN / SECYER).to_bits()
    );
    assert_eq!(
        solver.min_mdot_for_implicit_cgs.to_bits(),
        (ctrl.min_mdot_for_implicit * MSUN / SECYER).to_bits()
    );
    assert_eq!(
        solver.max_abs_mdot_cgs.to_bits(),
        (ctrl.max_implicit_abs_mdot * MSUN / SECYER).to_bits()
    );

    // Implicit "Arras" (`binary_mdot.f90:165-193`) must accept `"Arras"`:
    // tiny rates accept `explicit_mdot`, larger active rates iterate via
    // `(explicit_mdot - mdot_current) / explicit_mdot`.
    let mut arras_tiny = solver.clone();
    let tiny_exp = -1.0e-16 * MSUN / SECYER;
    assert_eq!(
        arras_tiny
            .evaluate_trial("Arras", 1, 20, 0.0, tiny_exp, -0.1, -0.5, 0.8)
            .unwrap(),
        ImplicitStepAction::Converged(tiny_exp)
    );

    let mut arras_active = solver.clone();
    let active_exp = -2.0e-9 * MSUN / SECYER;
    let try1 = arras_active
        .evaluate_trial("Arras", 1, 20, 0.0, active_exp, 0.01, -0.5, 0.8)
        .unwrap();
    assert_eq!(
        try1,
        ImplicitStepAction::TryNextMdot(-ctrl.starting_mdot * MSUN / SECYER)
    );
    assert_eq!(
        arras_active
            .evaluate_trial(
                "Arras",
                2,
                20,
                active_exp * (1.0 - 5.0e-5),
                active_exp,
                0.01,
                -0.5,
                0.8,
            )
            .unwrap(),
        ImplicitStepAction::Converged(active_exp * (1.0 - 5.0e-5))
    );

    // Bisection midpoint must use `(mdot_hi + mdot_lo) / 2.0`.
    let mut bisect_solver =
        ImplicitMdotSolver::from_controls(&ctrl, 1.5).unwrap();
    let m_lo = -1.3e-8 * MSUN / SECYER;
    let m_hi = -3.7e-8 * MSUN / SECYER;
    let _ = bisect_solver
        .evaluate_trial("roche_lobe", 1, 20, m_lo, 0.0, 0.02, -0.5, 0.8)
        .unwrap();
    let mid_action = bisect_solver
        .evaluate_trial("roche_lobe", 2, 20, m_hi, 0.0, -0.03, -0.5, 0.8)
        .unwrap();
    let ImplicitStepAction::TryNextMdot(next_mid) = mid_action else {
        panic!("expected TryNextMdot bisection, got {mid_action:?}");
    };
    assert_eq!(next_mid.to_bits(), ((m_hi + m_lo) / 2.0).to_bits());

    // `ImplicitRlofSolver::solve_roche_lobe_mdot` and `solve_contact_mdot`.
    let rlof = ImplicitRlofSolver {
        tolerance: 1.0e-5,
        change_factor: 1.5,
        use_quadratic: true,
        ..ImplicitRlofSolver::default()
    };
    let target = -4.2e-8 * MSUN / SECYER;
    let (solved_mdot, tries, final_f) =
        rlof.solve_roche_lobe_mdot(-1.0e-10 * MSUN / SECYER, 40, |m| {
            2.0e-2 * (m - target) / target.abs()
        });
    assert!(tries <= 41);
    assert!(final_f.abs() <= rlof.tolerance);
    assert!((solved_mdot - target).abs() / target.abs() < 1.0e-3);

    // Contact solver with companion filling its Roche lobe (`q = 0.8 < 1`).
    let q = 0.8_f64;
    let sc = pow(q, -0.52);
    let (contact_mdot, _c_tries, contact_f) =
        rlof.solve_contact_mdot(-2.0e-8 * MSUN / SECYER, q, 40, |m| {
            let gd = 0.02 * (m - target) / target.abs() + 0.005 / sc;
            let ga = 0.005 + 0.002 * (m - target) / target.abs();
            (gd, ga)
        });
    assert!(contact_f.abs() <= rlof.tolerance);
    let gd_at = 0.02 * (contact_mdot - target) / target.abs() + 0.005 / sc;
    let ga_at = 0.005 + 0.002 * (contact_mdot - target) / target.abs();
    assert!((gd_at * sc - ga_at).abs() <= rlof.tolerance);

    // `flip_for_contact_donor_switch` + `update_change_factor_after_step`.
    ctrl.mdot_scheme = String::from("contact");
    let mut flip_solver =
        ImplicitMdotSolver::from_controls(&ctrl, 1.3).unwrap();
    let _ = flip_solver
        .evaluate_trial("contact", 1, 20, m_lo, 0.0, 0.02, 0.01, 0.9)
        .unwrap();
    flip_solver.flip_for_contact_donor_switch();
    let cf_clamped = update_change_factor_after_step(
        false,
        1.5,
        -1.0e-8 * MSUN / SECYER,
        -2.0e-8 * MSUN / SECYER,
        &ctrl,
    );
    assert!(cf_clamped >= ctrl.min_change_factor);
    assert!(cf_clamped <= ctrl.max_change_factor);
}

#[test]
fn test_edd_wind_bondi_hoyle_and_accreted_j_mesa_order() {
    let m_d = 12.3 * MSUN;
    let m_a = 1.47 * MSUN;
    let r_d = 9.8 * RSUN;
    let sep = 42.5 * RSUN;
    let ecc = 0.37;
    let alpha_bh = 1.5;
    let beta_bh = 0.7;
    let max_frac = 0.8;

    let actual_bh = eval_bondi_hoyle_wind_fraction(
        m_d, m_a, r_d, sep, ecc, alpha_bh, beta_bh, max_frac,
    );
    let v_orb = (STANDARD_CGRAV * (m_d + m_a) / sep).sqrt();
    let v_wind = (2.0 * beta_bh * STANDARD_CGRAV * m_d / r_d).sqrt();
    let expected_bh = (alpha_bh / pow2(sep) / (2.0 * (1.0 - pow2(ecc)).sqrt())
        * pow2(STANDARD_CGRAV * m_a / pow2(v_wind))
        * pow(1.0 + pow2(v_orb / v_wind), -1.5))
    .clamp(0.0, max_frac);
    assert_eq!(actual_bh.to_bits(), expected_bh.to_bits());
    assert_eq!(
        eval_bondi_hoyle_wind_fraction(
            m_d, m_a, 0.0, sep, ecc, alpha_bh, beta_bh, max_frac,
        ),
        0.0
    );

    // Keplerian disk (`r_accretor < r_min`, type 2) vs ballistic impact
    // (`r_accretor >= r_min`, type 1: `sqrt(G * M_a * 1.7 * r_min)`).
    let q_acc = (m_a / m_d).clamp(0.0667, 15.0);
    let r_min = 0.0425 * sep * pow(q_acc + q_acc * q_acc, 0.25);
    let r_acc_disk = 0.5 * r_min;
    let (j_disk, j_type_disk) =
        eval_accreted_material_j(m_d, m_a, r_acc_disk, sep);
    assert_eq!(j_type_disk, 2);
    assert_eq!(
        j_disk.to_bits(),
        (STANDARD_CGRAV * m_a * r_acc_disk).sqrt().to_bits()
    );

    let (j_ballistic, j_type_ballistic) =
        eval_accreted_material_j(m_d, m_a, 1.5 * r_min, sep);
    let expected_j_ballistic = (STANDARD_CGRAV * m_a * 1.7 * r_min).sqrt();
    assert_eq!(j_type_ballistic, 1);
    assert_eq!(j_ballistic.to_bits(), expected_j_ballistic.to_bits());

    // Kerr BH ISCO & Eddington accretion retention invariants.
    let m0 = 5.0 * MSUN;
    assert!((bh_initial_equatorial_mass(m0, 0.0) - m0).abs() / m0 < 1.0e-15);
    let m_eq_max = bh_initial_equatorial_mass(m0, 1.0);
    assert!((m_eq_max - m0 / 6.0_f64.sqrt()).abs() / m0 < 1.0e-7);
    assert!(eval_bh_spin(m0, m0).abs() < 1.0e-14);
    assert!(eval_bh_spin(6.0_f64.sqrt() * m0, m0) > 0.999);

    let (mdot_edd_default, eta_default) = eval_mdot_edd(m0, m0, 0.34);
    assert!((eta_default - (1.0 - (8.0 / 9.0_f64).sqrt())).abs() < 1.0e-15);
    assert!(mdot_edd_default > 0.0);

    let (mdot_ovr, eta_ovr) =
        eval_mdot_edd_with_overrides(m0, m0, 0.34, 0.1, 2.5e-8);
    assert!((eta_ovr - 0.1).abs() < 1.0e-15);
    assert!((mdot_ovr - 2.5e-8 * (MSUN / SECYER)).abs() < 1.0e-15);

    let ret = eval_point_mass_edd_and_retention(
        m0,
        m0,
        0.70,
        0.34,
        false,
        0.1,
        1.0e-8,
        true,
        true,
        -1.0e-6 * MSUN / SECYER,
        -1.0e-9 * MSUN / SECYER,
        0.05,
        0.10,
        0.02,
    );
    assert!((ret.mdot_acc_raw - ret.mdot_edd).abs() < 1.0e-15);
    assert!((ret.component_mdot_accretor - 0.9 * ret.mdot_edd).abs() < 1.0e-15);
    assert!(
        (ret.accretion_luminosity - 0.1 * ret.mdot_edd * CLIGHT * CLIGHT).abs()
            / ret.accretion_luminosity
            < 1.0e-15
    );

    // Circular (`e = 0`) orbit-averaged Tout wind over `time_co[0..n]`
    // integrates `(n - 1) / n` of the period (`binary_wind.f90:162-164`).
    let mwind = -1.0e-7 * MSUN / SECYER;
    let circ_rate = mwind * eval_tout_wind_enhancement(r_d, 20.0 * RSUN, 100.0);
    let orb_avg = eval_tout_wind_enhancement_orbit_averaged(
        mwind,
        r_d,
        20.0 * RSUN,
        100.0,
        0.0,
        64,
    );
    let expected_orb_avg = (63.0 / 64.0) * circ_rate;
    assert!(
        (orb_avg - expected_orb_avg).abs() / expected_orb_avg.abs() < 1.0e-12
    );
}

#[test]
fn test_orbit_kepler_roche_and_anomaly_grid_mesa_order() {
    let m1 = 7.4 * MSUN;
    let m2 = 3.1 * MSUN;
    let a = 18.7 * RSUN;

    // Bitwise `eval_rlobe` and `separation_from_period` operation order.
    let q = pow(m1 / m2, ONE_THIRD);
    let expected_rl = a * 0.49 * q * q / (0.6 * q * q + log1p(q));
    assert_eq!(eval_rlobe(m1, m2, a).to_bits(), expected_rl.to_bits());

    let p = period_from_separation(m1, m2, a);
    let p_over_2pi = p / (2.0 * PI);
    let gm = STANDARD_CGRAV * (m1 + m2);
    let expected_a = pow(gm * pow2(p_over_2pi), ONE_THIRD);
    assert_eq!(
        separation_from_period(m1, m2, p).to_bits(),
        expected_a.to_bits()
    );
    assert!((expected_a - a).abs() / a < 1.0e-14);

    // `eval_j_orb` <-> `separation_from_j_orb` round trip at nonzero `e`.
    let e = 0.43;
    let j_orb = eval_j_orb(m1, m2, a, e);
    let a_back = separation_from_j_orb(m1, m2, j_orb, e);
    assert!((a_back - a).abs() / a < 1.0e-14);

    // Double division in `eval_rl_relative_gap` + `eval_rlobe_ad` check.
    let r_star = 0.85 * expected_rl * (1.0 - e);
    let expected_gap =
        (r_star - expected_rl * (1.0 - e)) / expected_rl / (1.0 - e);
    assert_eq!(
        eval_rl_relative_gap(r_star, expected_rl, e).to_bits(),
        expected_gap.to_bits()
    );
    let ad = eval_rlobe_ad::<3>(
        DualOrder1::independent(m1, 0),
        DualOrder1::independent(m2, 1),
        DualOrder1::independent(a, 2),
    );
    assert_eq!(ad.val.to_bits(), expected_rl.to_bits());
    assert!((ad.d1[2] - expected_rl / a).abs() / (expected_rl / a) < 1.0e-14);

    // L2 overflow formula & anomaly grid `/ 2.0 / PI` for even and odd N.
    let l2_eq = eval_l2_overflow_limit(m1, m1);
    assert!((l2_eq - 0.29858997 * atan(1.83530121)).abs() < 1.0e-15);

    for &n in &[9_usize, 10_usize] {
        let (theta, t_co) = build_anomaly_grid(n, e);
        assert_eq!(theta.len(), n);
        assert_eq!(t_co.len(), n);
        assert_eq!(theta[0], 0.0);
        assert_eq!(t_co[0], 0.0);
        let sqrt_f = ((1.0 - e) / (1.0 + e)).sqrt();
        let e_sq = e * (1.0 - e * e).sqrt();
        for i in 1..n {
            assert!(theta[i] > theta[i - 1] && theta[i] < 2.0 * PI);
            assert!(t_co[i] > t_co[i - 1] && t_co[i] < 1.0);
            let th = theta[i];
            let mut raw = (2.0 * atan(sqrt_f * tan(0.5 * th))
                - e_sq * sin(th) / (1.0 + e * cos(th)))
                / 2.0
                / PI;
            if raw < -1.0e-15 {
                raw += 1.0;
            } else if raw < 0.0 {
                raw = 0.0;
            }
            assert_eq!(t_co[i].to_bits(), raw.to_bits());
        }
    }
}

#[test]
fn test_roche_deformation_2d_steffen_parabolic_boundaries_and_clamp() {
    // 5x6 non-uniform smooth monotonic grid across corners, edges, interior.
    let xs = vec![-2.5, -1.0, 0.0, 1.5, 2.8];
    let ys = vec![0.0, 0.25, 0.6, 1.0, 1.4, 1.61];
    let (nx, ny) = (xs.len(), ys.len());
    let mut vals_smooth = Vec::with_capacity(nx * ny);
    for &x in &xs {
        for &y in &ys {
            vals_smooth.push(
                1.0 + 0.15 * (x + 3.0) * (x + 3.0)
                    + 1.2 * y
                    + 0.35 * y * y
                    + 0.04 * x * y,
            );
        }
    }
    let smooth = RocheTable2D {
        xs: xs.clone(),
        ys: ys.clone(),
        values: vals_smooth,
    };

    let sample_x = [-2.9, -2.5, -2.1, -0.35, 0.75, 2.45, 2.8, 3.4];
    let sample_y = [
        -0.05,
        0.0,
        0.11,
        0.42,
        0.83,
        1.25,
        1.56,
        1.61 - 1.5 * ROCHE_NUDGE,
        1.61 - 0.5 * ROCHE_NUDGE,
        1.61,
        1.75,
    ];
    for &lq in &sample_x {
        for &ar in &sample_y {
            let got = smooth.eval(lq, ar);
            let want = ref_bipm_eval(&smooth, lq, ar);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "smooth mismatch at lq={lq}, ar={ar}: \
                 got={got:.17e}, want={want:.17e}"
            );
        }
    }

    // Non-monotonic surface triggering overshoot clamping (`|p| > 2*|s|`),
    // sign reversals (`d = 0`), and parabolic boundary clamping.
    let mut vals_nonmono = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            let phase = ((i * 7 + j * 3) as f64) * 0.9;
            let bump = if j == 2 { 2.5 } else { 0.0 };
            vals_nonmono
                .push(sin(phase) + 0.3 * cos(phase) + bump - 0.2 * (j as f64));
        }
    }
    let nonmono = RocheTable2D {
        xs,
        ys,
        values: vals_nonmono,
    };
    for &lq in &[-2.4, -0.5, 0.6, 2.3] {
        for &ar in &[0.08, 0.35, 0.72, 1.18, 1.55] {
            let got = nonmono.eval(lq, ar);
            let want = ref_bipm_eval(&nonmono, lq, ar);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "nonmono mismatch at lq={lq}, ar={ar}: \
                 got={got:.17e}, want={want:.17e}"
            );
        }
    }

    // `ar` in `[ys[ny-1] - 2*ROCHE_NUDGE, ys[ny-1])` must NOT clamp early,
    // whereas `ar >= ys[ny-1]` clamps to `ys[ny-1] - 2*ROCHE_NUDGE`.
    let ar_below_top = 1.61 - 0.5 * ROCHE_NUDGE;
    let v_below = smooth.eval(0.5, ar_below_top);
    let v_at = smooth.eval(0.5, 1.61);
    let v_above = smooth.eval(0.5, 1.85);
    let v_nudge = smooth.eval(0.5, 1.61 - 2.0 * ROCHE_NUDGE);
    assert_eq!(v_at.to_bits(), v_nudge.to_bits());
    assert_eq!(v_above.to_bits(), v_nudge.to_bits());
    assert_ne!(v_below.to_bits(), v_nudge.to_bits());

    // `roche_fp_ft`, `roche_irot`, and `eval_psi` (`-inf` -> `-9e99` vs
    // finite stencils skipping `jy = 0`).
    let mut psi_text = String::from("3 -1.0 0.0 1.0 5 0.0 0.25 0.5 0.75 1.0 ");
    for i in 0..3 {
        let shift = 0.05 * f64::from(i);
        psi_text.push_str("-inf ");
        let _ = write!(
            psi_text,
            "{} {} {} {} ",
            -2.2 + shift,
            -1.7 + shift,
            -1.3 + shift,
            -1.0 + shift
        );
    }
    let psi_tbl = RocheTable2D::parse_ascii(&psi_text).unwrap();
    let tables = RocheDeformationTables {
        fp: smooth.clone(),
        ft: smooth.clone(),
        irot: smooth,
        psi: psi_tbl,
    };
    assert_eq!(tables.eval_psi(0.2, 0.15), -9.0e99);
    assert_eq!(tables.eval_psi(0.2, 0.35), -9.0e99);
    let finite_psi = tables.eval_psi(0.2, 0.60);
    assert!(finite_psi.is_finite() && finite_psi > -10.0 && finite_psi < 0.0);

    // Derivative just below top must remain nonzero (`nudge`).
    let (_fp, dfp_dlnr, _ft, dft_dlnr) =
        tables.roche_fp_ft(0.5, 1.61 - 1.5 * ROCHE_NUDGE);
    assert!(dfp_dlnr > 0.0 && dft_dlnr > 0.0);

    let r_cm = 2.0 * RSUN;
    let (i_rot, d_irot_dlnr) = tables.roche_irot(0.5, 0.7, r_cm);
    assert!(i_rot > 0.0 && d_irot_dlnr > 0.0);

    // Synchronicity sigmoid midpoint (`0.5`), symmetry, and hard cutoffs.
    let p_sec = 5.0 * 86400.0;
    let w_sync = 2.0 * PI / p_sec;
    assert!(
        (eval_synchronicity_switch(0.8 * w_sync, p_sec, 0.8, 10.0, 0.01) - 0.5)
            .abs()
            < 1.0e-15
    );
    assert_eq!(
        eval_synchronicity_switch(w_sync / 0.8, p_sec, 0.8, 10.0, 0.01)
            .to_bits(),
        eval_synchronicity_switch(0.8 * w_sync, p_sec, 0.8, 10.0, 0.01)
            .to_bits()
    );
    assert_eq!(eval_synchronicity_switch(0.0, p_sec, 0.8, 50.0, 0.01), 0.0);
    assert_eq!(
        eval_synchronicity_switch(w_sync, p_sec, 0.8, 50.0, 0.01),
        1.0
    );
}

#[test]
fn test_jdot_tides_disk_and_timestep_contracts() {
    // Hut polynomials vanish correction at e = 0 and remain finite at e = 0.5.
    assert_eq!(hut_f2(0.0), 1.0);
    assert_eq!(hut_f3(0.0), 1.0);
    assert_eq!(hut_f4(0.0), 1.0);
    assert_eq!(hut_f5(0.0), 1.0);
    let f2_half_exact = 1.0 + 7.5 * 0.25 + 5.625 * 0.0625 + 0.3125 / 64.0;
    assert!((hut_f2(0.5) - f2_half_exact).abs() < 1.0e-15);

    let m_s = 1.2 * MSUN;
    let m_c = 1.4 * MSUN;
    let r_s = 1.5 * RSUN;
    let a = 12.0 * RSUN;
    let period = period_from_separation(m_s, m_c, a);
    let j_orb = eval_j_orb(m_s, m_c, a, 0.2);

    assert!(eval_jdot_gr(m_s, m_c, a, j_orb) < 0.0);
    assert!(
        eval_jdot_ml(
            m_s,
            m_c,
            a,
            period,
            0.2,
            -1.0e-7 * MSUN / SECYER,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ) < 0.0
    );

    let kt_conv_sync = eval_k_div_t_convective(
        1.2,
        0.5,
        1.5,
        0.8,
        2.0,
        period,
        2.0 * PI / period,
        1.0,
    );
    let kt_conv_async = eval_k_div_t_convective(
        1.2,
        0.5,
        1.5,
        0.8,
        2.0,
        period,
        20.0 * PI / period,
        2.0,
    );
    assert!(kt_conv_sync > 0.0 && kt_conv_async <= kt_conv_sync);
    assert!(
        eval_k_div_t_radiative(5.0 * MSUN, 3.0 * MSUN, 3.0 * RSUN, 20.0 * RSUN,)
            > 0.0
    );
    assert!(eval_tsync(kt_conv_sync, m_c / m_s, 0.1, r_s, a, 1.0) > 0.0);
    assert_eq!(
        eval_edot_tidal_hut(
            kt_conv_sync,
            m_s,
            m_c,
            r_s,
            a,
            0.0,
            0.0,
            period,
            1.0,
        ),
        0.0
    );
    assert!(
        eval_edot_tidal_hut(
            kt_conv_sync,
            m_s,
            m_c,
            r_s,
            a,
            0.3,
            0.5 * (2.0 * PI / period),
            period,
            1.0,
        ) < 0.0
    );

    // MB scaling suppression below `mass_frac_for_scale` vs unity above.
    let mb_full = eval_jdot_mb_with_scale_mass(
        m_s,
        r_s,
        3.0 * RSUN,
        period,
        4.0,
        0.05,
        true,
        0.02,
    );
    let mb_damped = eval_jdot_mb_with_scale_mass(
        m_s,
        r_s,
        3.0 * RSUN,
        period,
        4.0,
        0.01,
        true,
        0.02,
    );
    assert!((mb_damped / mb_full - exp(-1.0)).abs() < 1.0e-14);

    // `jdot_ls` and `jdot_missing_wind`.
    assert_eq!(
        eval_jdot_ls_with_am_removed(
            true, 0, false, 1.0e50, 9.0e49, -1.0e-8, 1.0e45, -1.0e-8, 0.0, 0.0,
            0.0, 0.0, 1.0e7,
        ),
        0.0
    );
    let jls = eval_jdot_ls(1.0e50, 9.0e49, 2.0e50, 1.9e50, 1.0e8);
    assert!((jls - 2.0e41).abs() / 2.0e41 < 1.0e-14);
    assert_eq!(
        eval_jdot_missing_wind(true, -1.0e-8, -1.0e-6, 0.5, -1.0e-9, 1.0e17),
        0.0
    );

    // Disk L2 mass-loss fraction bounded on `[0, 1]`.
    let fl2 =
        eval_l2_mass_loss_fraction(20.0, 10.0, 1.0e-3, 2.0, 0.1, 1.3 / 2.4);
    assert!(fl2 > 0.0 && fl2 < 1.0);

    // Binary timestep `fj` limit and hard-limit violation detection.
    let mut ts_ctrl = BinaryTimestepControls {
        donor_time_step_yr: 100.0,
        max_timestep_sec: 1.0e12,
        dt_softening_factor: 0.0,
        angular_momentum_j: 0.99 * 1.0e52,
        angular_momentum_j_old: 1.0e52,
        fj: 1.0e-3,
        fm: -1.0,
        fr: -1.0,
        fa: -1.0,
        fe: -1.0,
        fdm: -1.0,
        ..BinaryTimestepControls::default()
    };
    let (next_dt, reason) = binary_pick_next_timestep(&ts_ctrl).unwrap();
    assert_eq!(reason, B_TLIM_JORB);
    assert!(next_dt > 0.0 && next_dt < 10.0 * SECYER);
    assert_eq!(
        check_binary_hard_limits(&ts_ctrl, -1.0, -1.0, 0.005, -1.0, -1.0, -1.0),
        Some(BinaryHardLimitViolation::FjHard)
    );
    ts_ctrl.ignore_hard_limits_this_step = true;
    assert_eq!(
        check_binary_hard_limits(&ts_ctrl, -1.0, -1.0, 0.005, -1.0, -1.0, -1.0),
        None
    );

    // `doing_first_model_of_run` suppresses `step_changes` on step 1.
    ts_ctrl.ignore_hard_limits_this_step = false;
    ts_ctrl.doing_first_model_of_run = true;
    assert_eq!(
        check_binary_hard_limits(&ts_ctrl, -1.0, -1.0, 0.005, -1.0, -1.0, -1.0),
        None
    );
    let (first_dt, first_reason) =
        binary_pick_next_timestep(&BinaryTimestepControls {
            doing_first_model_of_run: true,
            max_timestep_sec: -1.0,
            donor_dt_sec: 5.0 * SECYER,
            dt_softening_factor: 0.0,
            rl_relative_gap: [-0.2, 0.0],
            rl_relative_gap_old: [-0.1, 0.0],
            fm: -1.0,
            fr: 0.1,
            fj: -1.0,
            fa: -1.0,
            fe: -1.0,
            fdm: -1.0,
            ..BinaryTimestepControls::default()
        })
        .unwrap();
    assert_eq!(first_reason, B_TLIM_ENV);
    assert!(first_dt / SECYER > 1.0e50);
    let (active_dt, active_reason) =
        binary_pick_next_timestep(&BinaryTimestepControls {
            doing_first_model_of_run: false,
            max_timestep_sec: -1.0,
            donor_dt_sec: 5.0 * SECYER,
            dt_softening_factor: 0.0,
            rl_relative_gap: [-0.2, 0.0],
            rl_relative_gap_old: [-0.1, 0.0],
            fm: -1.0,
            fr: 0.1,
            fj: -1.0,
            fa: -1.0,
            fe: -1.0,
            fdm: -1.0,
            ..BinaryTimestepControls::default()
        })
        .unwrap();
    assert_eq!(active_reason, B_TLIM_ROCHE);
    assert!((active_dt / SECYER - 10.0).abs() < 1.0e-12);
}

#[test]
fn test_eval_rlobe_and_ad_left_to_right_1ulp_witness() {
    // Deterministic witnesses where left-to-right `a * 0.49 * q * q /
    // (0.6 * q * q + log1p(q))` (`binary_utils.f90:211`) differs from `q2 = q * q`
    // regroupings by >= 1 ULP while `eval_rlobe` and `eval_rlobe_ad(...).val`
    // match upstream bit-for-bit.
    let witnesses = [
        (0.3 * MSUN, 1.0 * MSUN, 10.0 * RSUN),
        (0.4 * MSUN, 1.4 * MSUN, 2.0 * RSUN),
        (4.0 * MSUN, 2.0 * MSUN, 15.0 * RSUN),
    ];
    for &(m1, m2, a) in &witnesses {
        let q = pow(m1 / m2, ONE_THIRD);
        let upstream_l2r = a * 0.49 * q * q / (0.6 * q * q + log1p(q));
        let q2 = q * q;
        let old_scalar = a * 0.49 * q2 / (0.6 * q2 + log1p(q));
        let old_ad_regroup = a * (q2 * 0.49) / (q2 * 0.6 + log1p(q));
        let old_ad_ln = a * (q2 * 0.49) / (q2 * 0.6 + log(q + 1.0));

        assert_ne!(old_scalar.to_bits(), upstream_l2r.to_bits());
        assert_ne!(old_ad_regroup.to_bits(), upstream_l2r.to_bits());
        assert_ne!(old_ad_ln.to_bits(), upstream_l2r.to_bits());

        let scalar = eval_rlobe(m1, m2, a);
        let ad = eval_rlobe_ad::<3>(
            DualOrder1::independent(m1, 0),
            DualOrder1::independent(m2, 1),
            DualOrder1::independent(a, 2),
        );
        assert_eq!(scalar.to_bits(), upstream_l2r.to_bits());
        assert_eq!(ad.val.to_bits(), upstream_l2r.to_bits());

        let denom = 0.6 * q * q + log1p(q);
        let drl_dq = a
            * (0.98 * q * denom - 0.49 * q * q * (1.2 * q + 1.0 / (1.0 + q)))
            / (denom * denom);
        let drl_dm1 = drl_dq * q / (3.0 * m1);
        let drl_dm2 = -drl_dq * q / (3.0 * m2);
        assert!((ad.d1[0] - drl_dm1).abs() / drl_dm1.abs() < 1.0e-14);
        assert!((ad.d1[1] - drl_dm2).abs() / drl_dm2.abs() < 1.0e-14);
        assert!(
            (ad.d1[2] - upstream_l2r / a).abs() / (upstream_l2r / a) < 1.0e-14
        );
    }
}
