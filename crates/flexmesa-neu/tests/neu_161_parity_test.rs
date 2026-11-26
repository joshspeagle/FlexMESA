// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]
//! Parity and analytical invariant tests for `flexmesa-neu` (`#161`).
//!
//! Covers edge cases in `neu_get` (`mod_neu.f90:223-419`), `brem_neu`
//! weak-degeneracy / liquid-metal blending (`mod_neu.f90:1208-1280`),
//! `phot_neu` three-band Fourier evaluation (`mod_neu.f90:553-831`),
//! `pair_neu` $T_9 = 10$ split (`mod_neu.f90:1620-1641`), `plas_neu`
//! $f_{xy}$ branches (`mod_neu.f90:1507-1551`), and `reco_neu` / `ifermi12` /
//! `zfermim12` inverse-derivative and $\nu \in [-20, 10]$ window semantics
//! (`mod_neu.f90:66-180, 1310-1406`).

#![allow(
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_const::{LN10, PI, TWO_THIRDS};
use flexmesa_math::{
    cospi, exp10, log10, pow, sinpi, with_numerics_mode, NumericsMode,
};
use flexmesa_neu::{
    brem_neu, ifermi12, neu_get, pair_neu, phot_neu, plas_neu, reco_neu,
    zfermim12, NeuChannel, NeuError, NeuFlags, NeuInputs, NeuRate, NeuResult,
    LOG10_TMIN_NEU, TMIN_NEU,
};

fn assert_rel_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    if expected.abs() <= 1.0e-200 && actual.abs() <= 1.0e-200 {
        return;
    }
    let scale = expected.abs().max(actual.abs());
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= rtol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    );
}

fn check_rate_fd<F>(
    t: f64,
    rho: f64,
    abar: f64,
    zbar: f64,
    rtol: f64,
    label: &str,
    eval: F,
) where
    F: Fn(&NeuInputs) -> NeuRate,
{
    let step = 1.0e-6;
    let mk = |tv: f64, rv: f64, av: f64, zv: f64| {
        let inp = NeuInputs::new(tv, log10(tv), rv, log10(rv), av, zv);
        eval(&inp)
    };
    let base = mk(t, rho, abar, zbar);
    assert!(
        base.val > 0.0 && base.val.is_finite(),
        "{label} base={base:?}"
    );

    let cases = [
        (
            "d_t",
            base.d_t,
            mk(t * (1.0 + step), rho, abar, zbar).val,
            mk(t * (1.0 - step), rho, abar, zbar).val,
            t * step,
        ),
        (
            "d_rho",
            base.d_rho,
            mk(t, rho * (1.0 + step), abar, zbar).val,
            mk(t, rho * (1.0 - step), abar, zbar).val,
            rho * step,
        ),
        (
            "d_abar",
            base.d_abar,
            mk(t, rho, abar * (1.0 + step), zbar).val,
            mk(t, rho, abar * (1.0 - step), zbar).val,
            abar * step,
        ),
        (
            "d_zbar",
            base.d_zbar,
            mk(t, rho, abar, zbar * (1.0 + step)).val,
            mk(t, rho, abar, zbar * (1.0 - step)).val,
            zbar * step,
        ),
    ];

    for (name, analytic, vp, vm, h) in cases {
        let fd = (vp - vm) / (2.0 * h);
        assert_rel_close(analytic, fd, rtol, &format!("{label}.{name}"));
    }
}

#[test]
fn test_neu_get_extreme_negative_log10_t_and_domain_guards() {
    let flags = NeuFlags::all();
    with_numerics_mode(NumericsMode::Strict, || {
        // When only `log10_T <= log10Tmin_neu` is passed below `f64` subnormal
        // underflow (`exp10(-400) == 0`), `mod_neu.f90:223-228` returns zero.
        for lt in [-400.0, -350.0, 6.5, LOG10_TMIN_NEU] {
            let res =
                neu_get(None, Some(lt), None, Some(5.0), 12.0, 6.0, 7.5, flags)
                    .expect("valid sub-threshold log10_T must return zero");
            assert_eq!(res, NeuResult::default());
        }

        // Exact `T == TMIN_NEU` cutoff returns zero.
        let at_tmin = neu_get(
            Some(TMIN_NEU),
            None,
            Some(1.0e5),
            None,
            12.0,
            6.0,
            7.5,
            flags,
        )
        .expect("T == TMIN_NEU returns zero");
        assert_eq!(at_tmin, NeuResult::default());

        // Invalid bounds matching `mod_neu.f90:240-280, 308-317`.
        assert!(matches!(
            neu_get(
                None,
                Some(f64::NEG_INFINITY),
                Some(1.0e5),
                Some(5.0),
                12.0,
                6.0,
                7.5,
                flags,
            ),
            Err(NeuError::InvalidInput(_))
        ));
        assert!(matches!(
            neu_get(
                None,
                Some(20.01),
                Some(1.0e5),
                Some(5.0),
                12.0,
                6.0,
                7.5,
                flags,
            ),
            Err(NeuError::InvalidInput(_))
        ));
        assert!(matches!(
            neu_get(
                Some(1.0e8),
                Some(8.0),
                None,
                Some(20.01),
                12.0,
                6.0,
                7.5,
                flags,
            ),
            Err(NeuError::InvalidInput(_))
        ));
        assert!(matches!(
            neu_get(
                Some(1.0e8),
                Some(8.0),
                Some(1.0e5),
                Some(5.0),
                12.0,
                0.0,
                7.5,
                flags,
            ),
            Err(NeuError::InvalidInput(_))
        ));
        assert!(matches!(
            neu_get(
                Some(1.0e8),
                Some(8.0),
                Some(1.0e5),
                Some(5.0),
                12.0,
                6.0,
                f64::NAN,
                flags,
            ),
            Err(NeuError::InvalidInput(_))
        ));
    });
}

#[test]
fn test_neu_get_cutoff_ramp_and_log10_tlim_edge_cases() {
    with_numerics_mode(NumericsMode::Strict, || {
        let log10_t = 7.25;
        let t = exp10(log10_t);
        let log10_rho = 5.5;
        let rho = exp10(log10_rho);
        let abar = 12.0;
        let zbar = 6.0;
        let flags = NeuFlags::all();

        // When `log10_Tlim <= log10Tmin_neu`, no cutoff ramp is applied.
        let no_ramp_68 = neu_get(
            Some(t),
            Some(log10_t),
            Some(rho),
            Some(log10_rho),
            abar,
            zbar,
            6.8,
            flags,
        )
        .expect("log10_tlim = 6.8");
        let no_ramp_70 = neu_get(
            Some(t),
            Some(log10_t),
            Some(rho),
            Some(log10_rho),
            abar,
            zbar,
            7.0,
            flags,
        )
        .expect("log10_tlim = 7.0");
        assert_eq!(no_ramp_68, no_ramp_70);

        // At midpoint `log10_T = 7.25` with `log10_Tlim = 7.5`, `phase = 0.5`.
        let ramped = neu_get(
            Some(t),
            Some(log10_t),
            Some(rho),
            Some(log10_rho),
            abar,
            zbar,
            7.5,
            flags,
        )
        .expect("ramped");
        let dtlim = 7.5 - LOG10_TMIN_NEU;
        let phase = (log10_t - LOG10_TMIN_NEU) / dtlim;
        let f_cut = 0.5 * (1.0 - cospi(phase));
        let df_dt = (0.5 * PI * sinpi(phase)) / (dtlim * t * LN10);
        assert_rel_close(f_cut, 0.5, 1.0e-15, "midpoint f_cut");
        assert_rel_close(
            ramped.total.val,
            f_cut * no_ramp_70.total.val,
            1.0e-15,
            "midpoint val",
        );
        assert_rel_close(
            ramped.total.d_t,
            f_cut * no_ramp_70.total.d_t + df_dt * no_ramp_70.total.val,
            1.0e-14,
            "midpoint d_t",
        );
        for ch in NeuChannel::ALL {
            let r_ch = ramped.channel(ch);
            let raw_ch = no_ramp_70.channel(ch);
            assert_rel_close(
                r_ch.val,
                f_cut * raw_ch.val,
                1.0e-15,
                "channel val",
            );
            assert_rel_close(
                r_ch.d_rho,
                f_cut * raw_ch.d_rho,
                1.0e-15,
                "channel d_rho",
            );
        }

        // At `log10_T == log10_Tlim = 7.5`, `phase = 1`, `f_cut = 1`, `df/dT = 0`.
        let t_edge = exp10(7.5);
        let at_edge = neu_get(
            Some(t_edge),
            Some(7.5),
            Some(rho),
            Some(log10_rho),
            abar,
            zbar,
            7.5,
            flags,
        )
        .expect("at edge");
        let un_ramped_edge = neu_get(
            Some(t_edge),
            Some(7.5),
            Some(rho),
            Some(log10_rho),
            abar,
            zbar,
            7.0,
            flags,
        )
        .expect("unramped edge");
        assert_rel_close(
            at_edge.total.val,
            un_ramped_edge.total.val,
            1.0e-15,
            "edge val",
        );
        assert_rel_close(
            at_edge.total.d_t,
            un_ramped_edge.total.d_t,
            1.0e-15,
            "edge d_t",
        );
    });
}

#[test]
fn test_brem_weak_liquid_blend_boundaries_and_derivatives() {
    with_numerics_mode(NumericsMode::Strict, || {
        let log10_rho = 8.0;
        let rho = exp10(log10_rho);
        let abar = 12.0;
        let zbar = 6.0;
        let den6_ye = (rho * 1.0e-6) * (zbar / abar);
        let u = pow(den6_ye, TWO_THIRDS);
        let tfermi = 5.9302e9 * (u.sqrt() - 1.0);
        assert!(tfermi > 1.0e9);

        // Continuity across `0.05 * T_F` and `0.30 * T_F`.
        let eps = 1.0e-11;
        for frac in [0.05, 0.30] {
            let t_lo = tfermi * frac * (1.0 - eps);
            let t_hi = tfermi * frac * (1.0 + eps);
            let r_lo = brem_neu(&NeuInputs::new(
                t_lo,
                log10(t_lo),
                rho,
                log10_rho,
                abar,
                zbar,
            ));
            let r_hi = brem_neu(&NeuInputs::new(
                t_hi,
                log10(t_hi),
                rho,
                log10_rho,
                abar,
                zbar,
            ));
            assert_rel_close(
                r_lo.val,
                r_hi.val,
                1.0e-9,
                "brem blend boundary val",
            );
            assert_rel_close(
                r_lo.d_t,
                r_hi.d_t,
                1.0e-7,
                "brem blend boundary d_t",
            );
        }

        // Centered FD verification inside liquid, blended, and weak regimes.
        for frac in [0.02, 0.175, 0.45] {
            check_rate_fd(
                frac * tfermi,
                rho,
                abar,
                zbar,
                1.0e-6,
                &format!("brem_frac_{frac}"),
                brem_neu,
            );
        }
    });
}

#[test]
fn test_phot_bands_and_reco_fermi_dirac_windows() {
    with_numerics_mode(NumericsMode::Strict, || {
        // `phot_neu` returns exact zero at `T <= 1e7`.
        let inp_cut = NeuInputs::new(1.0e7, 7.0, 1.0e4, 4.0, 4.0, 2.0);
        assert_eq!(phot_neu(&inp_cut), NeuRate::ZERO);

        // All three Fourier bands: [1e7, 1e8), [1e8, 1e9), [1e9, inf).
        for (lt, lrho, lbl) in [
            (7.6, 3.0, "phot_band_7_8"),
            (8.5, 4.0, "phot_band_8_9"),
            (9.4, 5.0, "phot_band_9_up"),
        ] {
            check_rate_fd(
                exp10(lt),
                exp10(lrho),
                12.0,
                6.0,
                1.0e-6,
                lbl,
                phot_neu,
            );
        }

        // Inverse derivative rule `d(ifermi12(f))/df == 2 / zfermim12(nu)`
        // across both `f < 4` and `f >= 4`.
        for f in [0.5, 10.0] {
            let h = f * 1.0e-6;
            let fd = (ifermi12(f + h) - ifermi12(f - h)) / (2.0 * h);
            let nu = ifermi12(f);
            let analytic = 1.0 / (0.5 * zfermim12(nu));
            assert_rel_close(analytic, fd, 1.0e-7, "ifermi12_deriv");
        }

        // `reco_neu` Table 12 branches (`nu < 0` and `nu >= 0`) + FD check.
        check_rate_fd(
            1.0e8,
            1.0e3,
            56.0,
            26.0,
            1.0e-6,
            "reco_nu_neg",
            reco_neu,
        );
        check_rate_fd(
            1.0e8,
            1.0e5,
            56.0,
            26.0,
            1.0e-6,
            "reco_nu_pos",
            reco_neu,
        );

        // Outside `nu in [-20, 10]`, `reco_neu` returns exact zero.
        let inp_nondeg = NeuInputs::new(1.0e10, 10.0, 1.0e-4, -4.0, 12.0, 6.0);
        assert_eq!(reco_neu(&inp_nondeg), NeuRate::ZERO);
        let inp_deg =
            NeuInputs::new(2.0e7, log10(2.0e7), 1.0e7, 7.0, 12.0, 6.0);
        assert_eq!(reco_neu(&inp_deg), NeuRate::ZERO);
    });
}

#[test]
fn test_pair_t9_boundary_and_plas_fxy_branches() {
    with_numerics_mode(NumericsMode::Strict, || {
        // `pair_neu` on both sides of `T_9 = 10` (`1e10 K`).
        for (t, lbl) in [(9.9e9, "pair_t9_lt_10"), (1.01e10, "pair_t9_ge_10")] {
            check_rate_fd(t, 1.0e6, 16.0, 8.0, 1.0e-6, lbl, pair_neu);
        }

        // `plas_neu` across all three `f_xy` branches:
        // 1) `f_xy == 1.0` (`xden < 0`)
        // 2) active `f_xy` with `c_val < 0`
        // 3) active `f_xy` with `c_val == 0`
        let cases = [
            (7.5, 1.0, "plas_fxy_unity"),
            (7.5, 4.0, "plas_fxy_c_neg"),
            (9.5, 9.5, "plas_fxy_c_zero"),
        ];
        for (lt, lrho, lbl) in cases {
            check_rate_fd(
                exp10(lt),
                exp10(lrho),
                4.0,
                2.0,
                1.0e-6,
                lbl,
                plas_neu,
            );
        }
    });
}
