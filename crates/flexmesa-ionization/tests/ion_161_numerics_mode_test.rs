// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]
//! Parity, continuity, AD, and validation tests for `flexmesa-ionization`
//! (`#161`).
//!
//! Runs in a dedicated `*numerics_mode_test.rs` binary per issue #210
//! routing governance.
//!
//! Covers:
//! 1. Exact `z36th = 1d0/36d0` scalar contract (`ion_tables_eval.f90:460`) and
//!    domain validation in `IonTableGrid::eval` (`ion_tables_eval.f90:92-130`).
//! 2. All 5 `Z` regimes (`Z < 1e-6`, 3-point Lagrange `[1e-6, 0.04]`, cosine
//!    blends `(0.04, 0.20]` and `(0.20, 1 - 1e-6]`, pure metal `Z > 1 - 1e-6`),
//!    boundary continuity, and all 4 `X` regimes (`Get_ion_for_X`).
//! 3. `DualOrder1<4>` analytical Jacobian vs centered finite differences.
//! 4. `Fe56He4Table` `Strict` vs `Fast` modes, `eval_val_grad_prevalidated`,
//!    `eval_dual`, `eval_zones` determinism, and domain clamping/error bounds.
//! 5. `chi_info`, `chi_effective`, `get_typical_charge`, `eval_typical_charge`,
//!    and `ionization_potential` against `mod_typical_charge.f90`.

#![allow(
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_autodiff::DualOrder1;
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{ONE_THIRD, TWO_THIRDS};
use flexmesa_ionization::{
    chi_effective, chi_info, eval_charge_of_fe56_in_he4, eval_ionization,
    eval_typical_charge, get_typical_charge, ionization_potential,
    Fe56He4Table, IonError, IonTableGrid, ION_XS, ION_ZS, LOG_NE_FE56_HE4,
    LOG_T_FE56_HE4, MAX_TYPICAL_CHARGE_Z, NUM_ION_XS_FOR_Z,
    PARALLEL_ZONE_THRESHOLD,
};
use flexmesa_math::{exp10, log10, pow, with_numerics_mode, NumericsMode};

fn assert_rel_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    if expected.abs() <= 1.0e-30 && actual.abs() <= 1.0e-30 {
        return;
    }
    let scale = expected.abs().max(actual.abs());
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= rtol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    );
}

#[test]
fn test_ion_z36th_constant_and_eval_input_validation() {
    // MESA `ion_tables_eval.f90:460, 514` evaluates `z36th * hx2 * hy2`
    // left-to-right with `z36th = 1d0/36d0`, which differs by 1 ULP from
    // `hx2 * hy2 * ((1/6) * (1/6))` on `(del_log_Q, del_log_T) = (0.08, 0.05)`.
    let z36th = 1.0_f64 / 36.0_f64;
    let sixth = 1.0_f64 / 6.0_f64;
    let hx2 = 0.08_f64 * 0.08_f64;
    let hy2 = 0.05_f64 * 0.05_f64;
    let mesa_order = z36th * hx2 * hy2;
    let right_assoc = hx2 * hy2 * (sixth * sixth);
    assert_eq!(z36th.to_bits(), 0x3f9c_71c7_1c71_c71c);
    assert_eq!(mesa_order.to_bits(), 0x3e9d_d37f_5698_c2c4);
    assert_eq!(right_assoc.to_bits(), 0x3e9d_d37f_5698_c2c3);
    assert_eq!(mesa_order.to_bits().abs_diff(right_assoc.to_bits()), 1);

    // Non-finite Z / X, missing density/T, non-positive rho/T, and non-finite
    // log10_Rho / log10_T must all return `IonError::InvalidInput`.
    assert!(matches!(
        eval_ionization(f64::NAN, 0.7, None, Some(-5.0), None, Some(4.5)),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(0.02, f64::INFINITY, None, Some(-5.0), None, Some(4.5)),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(0.02, 0.7, None, None, Some(1.0e4), Some(4.0)),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(0.02, 0.7, Some(1.0e-5), Some(-5.0), None, None),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(0.02, 0.7, Some(0.0), None, Some(1.0e4), None),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(0.02, 0.7, Some(1.0e-5), None, Some(-10.0), None),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(
            0.02,
            0.7,
            Some(1.0e-5),
            Some(f64::NAN),
            Some(1.0e4),
            Some(4.0)
        ),
        Err(IonError::InvalidInput { .. })
    ));
    assert!(matches!(
        eval_ionization(
            0.02,
            0.7,
            Some(1.0e-5),
            Some(-5.0),
            Some(1.0e4),
            Some(f64::INFINITY)
        ),
        Err(IonError::InvalidInput { .. })
    ));
}

#[test]
fn test_ion_table_z_x_regimes_continuity_and_zones_determinism() {
    with_numerics_mode(NumericsMode::Strict, || {
        let lrho = -5.2;
        let lt = 4.6;
        let rho = exp10(lrho);
        let t = exp10(lt);

        // Linear-only vs log-only input reconstruction must agree closely.
        let r_log =
            eval_ionization(0.02, 0.70, None, Some(lrho), None, Some(lt))
                .expect("log eval");
        let r_lin = eval_ionization(0.02, 0.70, Some(rho), None, Some(t), None)
            .expect("lin eval");
        for (&lin_v, &log_v) in r_lin.iter().zip(&r_log) {
            assert_rel_close(lin_v, log_v, 1.0e-12, "lin vs log");
        }

        // `Z < 1e-6` (and negative `Z` snapped to `0`) match `Z = 0.0`.
        let r_z0 = eval_ionization(0.0, 0.70, None, Some(lrho), None, Some(lt))
            .expect("z=0");
        let r_z_tiny =
            eval_ionization(5.0e-7, 0.70, None, Some(lrho), None, Some(lt))
                .expect("z=5e-7");
        let r_z_neg =
            eval_ionization(-0.01, 0.70, None, Some(lrho), None, Some(lt))
                .expect("z=-0.01");
        assert_eq!(r_z0, r_z_tiny);
        assert_eq!(r_z0, r_z_neg);

        // `Z > 1 - 1e-6` uses pure-metal `iz = 4`, ignoring `X > 0`.
        assert_eq!(NUM_ION_XS_FOR_Z[4], 1);
        let r_z1_x0 =
            eval_ionization(1.0, 0.0, None, Some(lrho), None, Some(lt))
                .expect("z=1, x=0");
        let r_z1_x7 =
            eval_ionization(1.0, 0.7, None, Some(lrho), None, Some(lt))
                .expect("z=1, x=0.7");
        assert_eq!(r_z1_x0, r_z1_x7);

        // Continuity across `Z = 0.04` (quadratic -> cospi blend) and `Z = 0.20`.
        // Note: `logpp` slopes near `Z = 0.04` reach ~1e3, so probe `dz = 1e-12`.
        for z_b in [ION_ZS[2], ION_ZS[3]] {
            let r_lo = eval_ionization(
                z_b - 1.0e-12,
                0.50,
                None,
                Some(lrho),
                None,
                Some(lt),
            )
            .expect("z_b-");
            let r_at =
                eval_ionization(z_b, 0.50, None, Some(lrho), None, Some(lt))
                    .expect("z_b");
            let r_hi = eval_ionization(
                z_b + 1.0e-12,
                0.50,
                None,
                Some(lrho),
                None,
                Some(lt),
            )
            .expect("z_b+");
            for ((&v_lo, &v_at), &v_hi) in r_lo.iter().zip(&r_at).zip(&r_hi) {
                assert_rel_close(v_lo, v_at, 1.0e-9, "z boundary lo");
                assert_rel_close(v_hi, v_at, 1.0e-9, "z boundary hi");
            }
        }

        // Continuity across `X = 0.2` (3-pt -> 4-pt) and high-`X` (`X = 0.9` at
        // `Z = 0.02` where `num_xs = 5`).
        let r_x2_lo = eval_ionization(
            0.02,
            ION_XS[1] - 1.0e-11,
            None,
            Some(lrho),
            None,
            Some(lt),
        )
        .expect("x=0.2-");
        let r_x2_hi = eval_ionization(
            0.02,
            ION_XS[1] + 1.0e-11,
            None,
            Some(lrho),
            None,
            Some(lt),
        )
        .expect("x=0.2+");
        for (&v_lo, &v_hi) in r_x2_lo.iter().zip(&r_x2_hi) {
            assert_rel_close(v_lo, v_hi, 1.0e-9, "x=0.2 cont");
        }
        let r_x9 =
            eval_ionization(0.02, 0.90, None, Some(lrho), None, Some(lt))
                .expect("x=0.9 at z=0.02");
        assert!(r_x9.iter().all(|v| v.is_finite()));

        // `eval_zones` sequential vs parallel (`rayon`) bit-exactness.
        let grid = IonTableGrid::builtin();
        let n_u32 = u32::try_from(PARALLEL_ZONE_THRESHOLD + 8)
            .expect("zone count fits in u32");
        let n_f = f64::from(n_u32);
        let zs: Vec<f64> =
            (0..n_u32).map(|k| f64::from(k) / n_f * 0.95).collect();
        let xs: Vec<f64> = (0..n_u32)
            .map(|k| f64::from(n_u32 - 1 - k) / n_f * 0.8)
            .collect();
        let lrs: Vec<f64> =
            (0..n_u32).map(|k| -7.0 + 0.05 * f64::from(k)).collect();
        let lts: Vec<f64> =
            (0..n_u32).map(|k| 3.5 + 0.04 * f64::from(k)).collect();

        let par = grid.eval_zones(&zs, &xs, &lrs, &lts).expect("par zones");
        assert_eq!(par.len(), zs.len());
        for (k, &par_row) in par.iter().enumerate() {
            let seq = grid
                .eval(zs[k], xs[k], None, Some(lrs[k]), None, Some(lts[k]))
                .expect("seq eval");
            assert_eq!(par_row, seq, "zone {k} parallel/scalar mismatch");
        }
        assert!(matches!(
            grid.eval_zones(&zs[..zs.len() - 1], &xs, &lrs, &lts),
            Err(IonError::InvalidInput { .. })
        ));
    });
}

#[test]
fn test_ion_table_dual_order1_jacobian_matches_finite_differences() {
    let grid = IonTableGrid::builtin();
    let step = 1.0e-6;
    with_numerics_mode(NumericsMode::Strict, || {
        // Exercises 3-pt Z + 4-pt X, mid-Z cospi + 3-pt X, and high-Z cospi.
        for (z, x, lrho, lt) in [
            (0.02, 0.50, -5.0, 4.5),
            (0.12, 0.725, -4.5, 4.8),
            (0.50, 0.30, -3.0, 5.5),
        ] {
            let zd = DualOrder1::<4>::independent(z, 0);
            let xd = DualOrder1::<4>::independent(x, 1);
            let lrd = DualOrder1::<4>::independent(lrho, 2);
            let ltd = DualOrder1::<4>::independent(lt, 3);
            let dual_out = grid.eval_generic(zd, xd, lrd, ltd);

            let base = grid
                .eval(z, x, None, Some(lrho), None, Some(lt))
                .expect("base");
            let zp = grid
                .eval(z + step, x, None, Some(lrho), None, Some(lt))
                .expect("zp");
            let zm = grid
                .eval(z - step, x, None, Some(lrho), None, Some(lt))
                .expect("zm");
            let xp = grid
                .eval(z, x + step, None, Some(lrho), None, Some(lt))
                .expect("xp");
            let xm = grid
                .eval(z, x - step, None, Some(lrho), None, Some(lt))
                .expect("xm");
            let rp = grid
                .eval(z, x, None, Some(lrho + step), None, Some(lt))
                .expect("rp");
            let rm = grid
                .eval(z, x, None, Some(lrho - step), None, Some(lt))
                .expect("rm");
            let tp = grid
                .eval(z, x, None, Some(lrho), None, Some(lt + step))
                .expect("tp");
            let tm = grid
                .eval(z, x, None, Some(lrho), None, Some(lt - step))
                .expect("tm");

            for (k, (&d_ch, &b_val)) in dual_out.iter().zip(&base).enumerate() {
                assert_rel_close(d_ch.val, b_val, 1.0e-14, "dual primal");
                let fds = [
                    (zp[k] - zm[k]) / (2.0 * step),
                    (xp[k] - xm[k]) / (2.0 * step),
                    (rp[k] - rm[k]) / (2.0 * step),
                    (tp[k] - tm[k]) / (2.0 * step),
                ];
                for (j, &fd) in fds.iter().enumerate() {
                    let actual = d_ch.d1[j];
                    let err = (actual - fd).abs();
                    let tol = 2.0e-8 + 1.0e-6 * fd.abs();
                    assert!(
                        err <= tol,
                        "dual deriv k={k} j={j}: {actual:.16e} vs {fd:.16e}"
                    );
                }
            }
        }
    });
}

#[test]
fn test_fe56_he4_methods_modes_dual_and_domain_bounds() {
    let tbl = Fe56He4Table::builtin();
    let (lne, lt) = (26.25, 7.15);

    // Consistency across `eval`, `eval_full`, `eval_val_grad_prevalidated`,
    // `eval_dual`, and `eval_charge_of_fe56_in_he4` in both `Strict` and `Fast`.
    for mode in [NumericsMode::Strict, NumericsMode::Fast] {
        with_numerics_mode(mode, || {
            let z = eval_charge_of_fe56_in_he4(lne, lt).expect("fe56 eval");
            let ev = tbl.eval_full(lne, lt).expect("eval_full");
            let vg = tbl.eval_val_grad_prevalidated(lne, lt);
            assert_eq!(z, ev.f);
            assert_eq!(vg, [ev.f, ev.df_dx, ev.df_dy]);

            let d_lne = DualOrder1::<2>::independent(lne, 0);
            let d_lt = DualOrder1::<2>::independent(lt, 1);
            let d_out = tbl.eval_dual(d_lne, d_lt).expect("eval_dual");
            assert_eq!(d_out.val, ev.f);
            assert_eq!(d_out.d1, [ev.df_dx, ev.df_dy]);

            // `Strict` quantizes local basis scalars to `f32`, adding ~5e-7
            // value steps, so use `h = 1e-3` in `Strict` and `h = 1e-6` in
            // pure-`f64` `Fast` mode.
            let (h, rtol) = if mode == NumericsMode::Fast {
                (1.0e-6, 1.0e-8)
            } else {
                (1.0e-3, 1.0e-3)
            };
            let fd_x = (tbl.eval(lne + h, lt).expect("xp")
                - tbl.eval(lne - h, lt).expect("xm"))
                / (2.0 * h);
            let fd_y = (tbl.eval(lne, lt + h).expect("yp")
                - tbl.eval(lne, lt - h).expect("ym"))
                / (2.0 * h);
            assert_rel_close(ev.df_dx, fd_x, rtol, "fe56 df_dx");
            assert_rel_close(ev.df_dy, fd_y, rtol, "fe56 df_dy");
        });
    }

    // Parallel `eval_zones` matches scalar `eval_full` bit-for-bit in `Strict`.
    with_numerics_mode(NumericsMode::Strict, || {
        let n_u32 = u32::try_from(PARALLEL_ZONE_THRESHOLD + 10)
            .expect("zone count fits in u32");
        let lnes: Vec<f64> =
            (0..n_u32).map(|k| 24.2 + 0.08 * f64::from(k)).collect();
        let lts: Vec<f64> =
            (0..n_u32).map(|k| 6.1 + 0.02 * f64::from(k)).collect();
        let zones = tbl.eval_zones(&lnes, &lts).expect("fe56 zones");
        for (k, &zone) in zones.iter().enumerate() {
            let ev = tbl.eval_full(lnes[k], lts[k]).expect("scalar");
            assert_eq!(zone, [ev.f, ev.df_dx, ev.df_dy]);
        }
        assert!(matches!(
            tbl.eval_zones(&lnes[..lnes.len() - 1], &lts),
            Err(IonError::InvalidInput { .. })
        ));
    });

    // Boundary clamping within `4e-7` relative tolerance (`herm2xy_db`) and
    // explicit `IonError::Interp` outside tolerance or on `NaN`.
    let ne_min = LOG_NE_FE56_HE4[0];
    let ne_max = LOG_NE_FE56_HE4[LOG_NE_FE56_HE4.len() - 1];
    let t_min = LOG_T_FE56_HE4[0];
    let t_max = LOG_T_FE56_HE4[LOG_T_FE56_HE4.len() - 1];
    assert_eq!(
        tbl.eval(ne_min - 1.0e-7, t_min).expect("clamped min"),
        tbl.eval(ne_min, t_min).expect("exact min")
    );
    assert_eq!(
        tbl.eval(ne_max, t_max + 1.0e-7).expect("clamped max"),
        tbl.eval(ne_max, t_max).expect("exact max")
    );
    for (bad_ne, bad_t) in [
        (ne_min - 0.1, t_min),
        (ne_max + 0.1, t_max),
        (ne_min, t_min - 0.1),
        (ne_max, t_max + 0.1),
        (f64::NAN, 7.0),
        (26.0, f64::INFINITY),
    ] {
        assert!(matches!(tbl.eval(bad_ne, bad_t), Err(IonError::Interp(_))));
    }
}

#[test]
fn test_typical_charge_chi_and_potentials_parity() {
    let cat = NuclideCatalog::builtin();

    with_numerics_mode(NumericsMode::Strict, || {
        // Boundary `Z` cases in `mod_typical_charge.f90:97-109`:
        // `Z = 0` -> 0, `Z = 1` -> 1, `Z > 30` -> `Z`, and `1 <= Z <= 30`
        // never drops below `1.0`.
        assert_eq!(
            get_typical_charge(0, 1.0, 0.5, 1.0e5, 5.0, 1.0e-2, -2.0),
            0.0
        );
        assert_eq!(
            get_typical_charge(1, 1.3, 0.85, 1.0e3, 3.0, 1.0e-8, -8.0),
            1.0
        );
        assert_eq!(
            get_typical_charge(
                MAX_TYPICAL_CHARGE_Z + 1,
                1.3,
                0.85,
                1.0e3,
                3.0,
                1.0e-8,
                -8.0
            ),
            31.0
        );
        assert_eq!(
            get_typical_charge(26, 1.3, 0.5, 1.0e3, 3.0, 1.0e-10, -10.0),
            1.0
        );

        // `chi_info` exact analytical check (`mod_typical_charge.f90:68-77`).
        let (a1, z1, t, rho) = (4.0, 2.0, 1.0e6, 10.0);
        let (lt, lrho) = (log10(t), log10(rho));
        let (chi, c0, c1, c2) = chi_info(a1, z1, t, lt, rho, lrho);
        let exp_chi =
            1.987e-4 * t * (-8.392 - lrho + 1.5 * lt - log10(z1 / a1));
        let exp_c0 = 1.085e-4 * rho * t / a1;
        let exp_c1 = 1.617e4 * (rho * (z1 * z1 + z1) / (t * a1)).sqrt();
        let exp_c2 = 29.38 * z1 * pow(rho / a1, ONE_THIRD);
        assert_eq!(chi.to_bits(), exp_chi.to_bits());
        assert_eq!(c0.to_bits(), exp_c0.to_bits());
        assert_eq!(c1.to_bits(), exp_c1.to_bits());
        assert_eq!(c2.to_bits(), exp_c2.to_bits());

        // `chi_effective` checks both `min(c1 * z2, c2 * (...))` branches and
        // `DualOrder1<2>` derivatives vs centered FD.
        let z2 = 4.0;
        let ceff_a = chi_effective(chi, c0, 1.0e-4, 1.0e2, z1, z2);
        let ceff_b = chi_effective(chi, c0, 1.0e2, 1.0e-2, z1, z2);
        let exp_b = chi
            + c0 / (z2 * z2 * z2)
            + 1.0e-2 * (pow(z2 / z1, TWO_THIRDS) + 0.6);
        assert_eq!(ceff_b.to_bits(), exp_b.to_bits());
        assert!(ceff_a < ceff_b);

        let h = 1.0e-6;
        let t_d = DualOrder1::<2>::independent(t, 0);
        let rho_d = DualOrder1::<2>::independent(rho, 1);
        let (chi_d, c0_d, c1_d, c2_d) = chi_info(
            DualOrder1::<2>::constant(a1),
            DualOrder1::<2>::constant(z1),
            t_d,
            t_d.log10(),
            rho_d,
            rho_d.log10(),
        );
        let ce_d = chi_effective(
            chi_d,
            c0_d,
            c1_d,
            c2_d,
            DualOrder1::<2>::constant(z1),
            DualOrder1::<2>::constant(z2),
        );
        let eval_ce = |tv: f64, rv: f64| {
            let (ch, d0, d1, d2) =
                chi_info(a1, z1, tv, log10(tv), rv, log10(rv));
            chi_effective(ch, d0, d1, d2, z1, z2)
        };
        let fd_t = (eval_ce(t * (1.0 + h), rho) - eval_ce(t * (1.0 - h), rho))
            / (2.0 * t * h);
        let fd_r = (eval_ce(t, rho * (1.0 + h)) - eval_ce(t, rho * (1.0 - h)))
            / (2.0 * rho * h);
        assert_rel_close(ce_d.d1[0], fd_t, 1.0e-7, "chi_effective d_t");
        assert_rel_close(ce_d.d1[1], fd_r, 1.0e-7, "chi_effective d_rho");

        // `eval_typical_charge` and `ionization_potential` bounds.
        assert_eq!(ionization_potential(0, 1), None);
        assert_eq!(ionization_potential(31, 1), None);
        assert_eq!(ionization_potential(6, 7), None);
        assert_eq!(ionization_potential(6, 6), Some(489.98));

        let ih1 = cat.get_nuclide_index("h1").expect("h1");
        let z_h1 = eval_typical_charge(&cat, ih1, 1.3, 0.85, t, lt, rho, lrho)
            .expect("h1 typical charge");
        assert_eq!(z_h1, 1.0);
        assert!(matches!(
            eval_typical_charge(&cat, 0, 1.3, 0.85, t, lt, rho, lrho),
            Err(IonError::InvalidInput { .. })
        ));
    });
}
