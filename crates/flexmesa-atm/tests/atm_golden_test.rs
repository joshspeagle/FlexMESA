// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity and partial-derivative tests for `flexmesa-atm` against
//! upstream MESA `atm/test/test_output` (`tests/fixtures/golden/atm/test_output.txt`).

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unreadable_literal
)]

use flexmesa_atm::{
    atm_l, atm_teff, build_t_tau_uniform, build_t_tau_varying, eval_e2,
    eval_irradiated, eval_t_tau, eval_t_tau_dq_dtau, eval_t_tau_uniform,
    eval_t_tau_varying, eval_table, get_t_tau_base, get_table_alfa_beta,
    get_table_base, radiation_pressure, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, ATM_DELTA_R, ATM_LNP, ATM_LNT,
    ATM_TABLE_DB_WD_TAU_25, ATM_TABLE_PHOTOSPHERE, ATM_TABLE_TAU_1,
    ATM_TABLE_TAU_10, ATM_TABLE_TAU_100, ATM_TABLE_TAU_1M1,
    ATM_TABLE_WD_TAU_25, ATM_TAU, ATM_T_TAU_EDDINGTON, ATM_T_TAU_KRISHNA_SWAMY,
    ATM_T_TAU_SOLAR_HOPF, ATM_T_TAU_TRAMPEDACH_SOLAR,
};
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{
    LN10, LSUN, MSUN, M_JUPITER, RSUN, R_JUPITER, STANDARD_CGRAV,
};
use flexmesa_eos::{eos_pt_get, EosConfig, HelmTable, I_ETA, I_LN_FREE_E};
use flexmesa_kap::{kap_get, KapConfig, KapLowTOption};
use flexmesa_math::exp10;

fn solar_species_and_xa(x: f64, z: f64) -> (Vec<usize>, Vec<f64>) {
    let cat = NuclideCatalog::builtin_ref();
    let names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| cat.get_by_name(n).unwrap().chem_id)
        .collect();
    let y = 1.0 - (x + z);
    let xc = 3.272_459_210_526_323_5e-3;
    let xn = 9.502_384_210_526_329_2e-4;
    let xo = 8.821_800_000_000_060_1e-3;
    let sum6 = x + y + xc + xn + xo;
    let xa = vec![x, y, xc, xn, xo, 0.0, 1.0 - sum6];
    (chem_id, xa)
}

fn make_eos_proc(
    chem_id: Vec<usize>,
    xa: Vec<f64>,
) -> impl FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError> {
    let ht = HelmTable::builtin();
    let cfg = EosConfig::default();
    move |ln_p: f64, ln_t: f64| {
        let t = ln_t.exp();
        let p = ln_p.exp();
        let prad = radiation_pressure(t);
        let pgas = (p - prad).max(1.0e-99);
        let log_pgas = pgas.log10();
        let log_t = ln_t / LN10;
        let pt = eos_pt_get(ht, &chem_id, &xa, log_pgas, log_t, &cfg).map_err(
            |e| AtmError::CallbackError {
                context: "eos_proc",
                detail: e.to_string(),
            },
        )?;
        Ok(AtmEosCallbackResult {
            ln_rho: pt.log10_rho * LN10,
            res: pt.eos.res,
            dres_dln_rho: pt.eos.d_dlnd,
            dres_dln_t: pt.eos.d_dln_t,
        })
    }
}

fn make_kap_proc(
    chem_id: Vec<usize>,
    xa: Vec<f64>,
    zbase: f64,
    low_t_option: KapLowTOption,
) -> impl FnMut(
    f64,
    f64,
    &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    let cfg = KapConfig {
        zbase,
        kap_low_t_option: low_t_option,
        ..KapConfig::default()
    };
    move |ln_rho: f64, ln_t: f64, eos: &AtmEosCallbackResult| {
        let out = kap_get(
            &cfg,
            &chem_id,
            &xa,
            ln_rho / LN10,
            ln_t / LN10,
            eos.res[I_LN_FREE_E],
            eos.dres_dln_rho[I_LN_FREE_E],
            eos.dres_dln_t[I_LN_FREE_E],
            eos.res[I_ETA],
            eos.dres_dln_rho[I_ETA],
            eos.dres_dln_t[I_ETA],
        )
        .map_err(|e| AtmError::CallbackError {
            context: "kap_proc",
            detail: e.to_string(),
        })?;
        Ok(AtmKapCallbackResult {
            kap: out.kap,
            dlnkap_dln_rho: out.dlnkap_dln_rho,
            dlnkap_dln_t: out.dlnkap_dln_t,
        })
    }
}

#[test]
fn test_section1_atm_tables_match_golden_oracle() {
    let cgrav = STANDARD_CGRAV;
    let z = 0.02;

    // 5 solar cases + 2 WD cases from test_atm_support.f90::test_table
    #[rustfmt::skip]
    let solar_cases = [
        (ATM_TABLE_TAU_1M1, 5090_i64, 3.706_693_87, 4.579_606_50, 4.579_587_15),
        (ATM_TABLE_TAU_1, 6238_i64, 3.795_056_97, 5.053_514_84, 5.053_500_18),
        (ATM_TABLE_TAU_10, 8591_i64, 3.934_038_27, 5.203_967_80, 5.203_930_50),
        (ATM_TABLE_TAU_100, 9977_i64, 3.998_990_69, 5.319_829_69, 5.319_777_73),
        (ATM_TABLE_PHOTOSPHERE, 5772_i64, 3.761_326_58, 4.916_999_29, 4.916_984_57),
    ];

    for (table_id, exp_t_int, exp_log_t, exp_log_p, exp_log_pgas) in solar_cases
    {
        let m = MSUN;
        let r = RSUN;
        let l = LSUN;
        let teff = atm_teff(l, r);
        let out = eval_table(l, r, m, cgrav, table_id, z, true, teff).unwrap();
        let t = out.ln_t.exp();
        let p = out.ln_p.exp();
        let prad = radiation_pressure(t);
        let t_int = (0.5 + t).floor() as i64;
        let log_t = out.ln_t / LN10;
        let log_p = p.log10();
        let log_pgas = (p - prad).log10();

        assert_eq!(t_int, exp_t_int, "table {table_id} integer T");
        assert!(
            (log_t - exp_log_t).abs() < 1.0e-8,
            "table {table_id} log_T got {log_t:.10} expected {exp_log_t:.8}"
        );
        assert!(
            (log_p - exp_log_p).abs() < 1.0e-8,
            "table {table_id} log_P got {log_p:.10} expected {exp_log_p:.8}"
        );
        assert!(
            (log_pgas - exp_log_pgas).abs() < 1.0e-8,
            "table {table_id} logPgas got {log_pgas:.10} expected {exp_log_pgas:.8}"
        );
    }

    // WD tau=25
    {
        let logg = 7.1_f64;
        let teff = 5000.0_f64;
        let m = 0.8 * MSUN;
        let r = (cgrav * m / exp10(logg)).sqrt();
        let l = atm_l(teff, r);
        let out =
            eval_table(l, r, m, cgrav, ATM_TABLE_WD_TAU_25, z, true, teff)
                .unwrap();
        let t = out.ln_t.exp();
        let p = out.ln_p.exp();
        let prad = radiation_pressure(t);
        assert_eq!((0.5 + t).floor() as i64, 6066);
        assert!((out.ln_t / LN10 - 3.782_918_44).abs() < 1.0e-8);
        assert!((p.log10() - 7.894_946_62).abs() < 1.0e-8);
        assert!(((p - prad).log10() - 7.894_946_61).abs() < 1.0e-8);
    }

    // DB WD tau=25
    {
        let logg = 8.0_f64;
        let teff = 25000.0_f64;
        let m = 0.8 * MSUN;
        let r = (cgrav * m / exp10(logg)).sqrt();
        let l = atm_l(teff, r);
        let out =
            eval_table(l, r, m, cgrav, ATM_TABLE_DB_WD_TAU_25, z, true, teff)
                .unwrap();
        let t = out.ln_t.exp();
        let p = out.ln_p.exp();
        let prad = radiation_pressure(t);
        assert_eq!((0.5 + t).floor() as i64, 51950);
        assert!((out.ln_t / LN10 - 4.715_587_22).abs() < 1.0e-8);
        assert!((p.log10() - 7.384_729_07).abs() < 1.0e-8);
        assert!(((p - prad).log10() - 7.384_400_00).abs() < 1.0e-8);
    }
}

#[test]
fn test_section2_t_tau_varying_matches_golden_oracle() {
    let cgrav = STANDARD_CGRAV;
    let log_teff = 5776.0_f64.log10();
    let log_rsurf = RSUN.log10();
    let log_m = MSUN.log10();
    let r = exp10(log_rsurf);
    let m = exp10(log_m);
    let teff = exp10(log_teff);
    let l = atm_l(teff, r);
    let x = 0.70;
    let z = 0.02;
    let (chem_id, xa) = solar_species_and_xa(x, z);

    #[rustfmt::skip]
    let cases = [
        (ATM_T_TAU_EDDINGTON, -1.0, 5776_i64, 3.761_627_18, 4.954_983_35),
        (ATM_T_TAU_KRISHNA_SWAMY, -1.0, 5776_i64, 3.761_627_18, 4.774_027_06),
        (ATM_T_TAU_SOLAR_HOPF, -1.0, 5776_i64, 3.761_627_18, 4.836_533_40),
        (ATM_T_TAU_TRAMPEDACH_SOLAR, -1.0, 5776_i64, 3.761_627_62, 4.893_468_12),
        (ATM_T_TAU_EDDINGTON, 100.0, 17026_i64, 4.231_113_92, 5.208_723_30),
    ];

    for (t_tau_id, tau_in, exp_t_int, exp_log_t, exp_log_p) in cases {
        let tau_base = if tau_in < 0.0 {
            get_t_tau_base(t_tau_id).unwrap()
        } else {
            tau_in
        };
        let eos_proc = make_eos_proc(chem_id.clone(), xa.clone());
        let kap_proc = make_kap_proc(
            chem_id.clone(),
            xa.clone(),
            z,
            KapLowTOption::LowTFa05Gs98,
        );
        let out = eval_t_tau_varying(
            tau_base, l, r, m, cgrav, t_tau_id, eos_proc, kap_proc, 1.0e-9,
            500, true, teff,
        )
        .unwrap();
        let t = out.ln_t.exp();
        let p = out.ln_p.exp();
        assert_eq!((0.5 + t).floor() as i64, exp_t_int);
        assert!(
            (t.log10() - exp_log_t).abs() < 1.0e-8,
            "varying id={t_tau_id} logT got {:.10} expected {exp_log_t:.8}",
            t.log10()
        );
        assert!(
            (p.log10() - exp_log_p).abs() < 2.0e-7,
            "varying id={t_tau_id} logP got {:.10} expected {exp_log_p:.8}",
            p.log10()
        );
    }
}

#[test]
fn test_section3_t_tau_uniform_and_irradiated_match_golden_oracle() {
    let cgrav = STANDARD_CGRAV;
    let x = 0.70;
    let z = 0.02;
    let (chem_id, xa) = solar_species_and_xa(x, z);

    // 1. Uniform fixed (tau = 100, max_iters = 0)
    {
        let m = 1.989_200_000_000_000_2e32;
        let r = 6.355_623_157_754_558_6e10;
        let l = 2.401_539_919_019_911_8e32;
        let teff = atm_teff(l, r);
        let kap_guess = 5.885_080_248_117_446_9e-2;
        let eos_proc = make_eos_proc(chem_id.clone(), xa.clone());
        let kap_proc = make_kap_proc(
            chem_id.clone(),
            xa.clone(),
            z,
            KapLowTOption::LowTFa05Gs98,
        );
        let out = eval_t_tau_uniform(
            100.0,
            l,
            r,
            m,
            cgrav,
            kap_guess,
            1.0,
            ATM_T_TAU_EDDINGTON,
            eos_proc,
            kap_proc,
            1.0e-6,
            0,
            true,
            teff,
        )
        .unwrap();
        assert!((out.ln_p / LN10 - 6.747_015_515_614_720_9).abs() < 1.0e-14);
        assert!((out.ln_t / LN10 - 3.949_824_720_843_101_2).abs() < 1.0e-14);
    }

    // 2. Uniform iterated (tau = 150, max_iters = 30)
    {
        let log_teff = 5776.0_f64.log10();
        let r = exp10(RSUN.log10());
        let m = exp10(MSUN.log10());
        let teff = exp10(log_teff);
        let l = atm_l(teff, r);
        let eos_proc = make_eos_proc(chem_id.clone(), xa.clone());
        let kap_proc = make_kap_proc(
            chem_id.clone(),
            xa.clone(),
            z,
            KapLowTOption::LowTFa05Gs98,
        );
        let out = eval_t_tau_uniform(
            150.0,
            l,
            r,
            m,
            cgrav,
            0.5,
            1.0,
            ATM_T_TAU_EDDINGTON,
            eos_proc,
            kap_proc,
            1.0e-6,
            30,
            true,
            teff,
        )
        .unwrap();
        assert!(
            (out.ln_p / LN10 - 4.748_369_074_037_571_9).abs() < 1.0e-12,
            "uniform iterated logP got {:.16e}",
            out.ln_p / LN10
        );
        assert!(
            (out.ln_t / LN10 - 4.274_896_795_432_437_3).abs() < 1.0e-14,
            "uniform iterated logT got {:.16e}",
            out.ln_t / LN10
        );
    }

    // 3. Irradiated grey atmosphere (Freedman11 low-T opacity)
    {
        let z_irr = 1.0e-2;
        let (chem_id_irr, xa_irr) = solar_species_and_xa(0.70, z_irr);
        let eos_proc = make_eos_proc(chem_id_irr.clone(), xa_irr.clone());
        let kap_proc = make_kap_proc(
            chem_id_irr,
            xa_irr,
            z_irr,
            KapLowTOption::LowTFreedman11,
        );

        let t_eq = 1000.0;
        let kap_v = 4.0e-3;
        let kap_guess = 1.5e-2;
        let gamma = 0.0;
        let p = 1.0e6;
        let m = 1.5 * M_JUPITER;
        let tau_guess = 10.0;
        let r = (cgrav * m * tau_guess / (p * kap_guess)).sqrt();
        let t_int = 900.0;
        let l = atm_l(t_int, r);

        assert!((m / M_JUPITER - 1.5).abs() < 1.0e-14);
        assert!((r / R_JUPITER - 1.574_373_509_563_38).abs() < 1.0e-13);
        assert!((r / RSUN - 1.617_868_491_385_728_9e-1).abs() < 1.0e-14);

        let out = eval_irradiated(
            l, r, m, cgrav, t_eq, p, kap_guess, kap_v, gamma, eos_proc,
            kap_proc, 1.0e-6, 30, true,
        )
        .unwrap();

        assert!(
            (out.kap - 4.749_836_820_148_591_7e-3).abs() < 1.0e-14,
            "irradiated kap got {:.16e}",
            out.kap
        );
        assert!(
            (out.tau - 3.166_976_303_697_707_9).abs() < 1.0e-12,
            "irradiated tau got {:.16e}",
            out.tau
        );
        assert!(
            (out.ln_t.exp() - 1.312_422_280_827_965_6e3).abs() < 1.0e-10,
            "irradiated T got {:.16e}",
            out.ln_t.exp()
        );
        assert!(
            (out.ln_t / LN10 - 3.118_073_594_739_383_4).abs() < 1.0e-13,
            "irradiated logT got {:.16e}",
            out.ln_t / LN10
        );
    }
}

#[test]
fn test_section4_partials_e2_and_build_structure() {
    // Check E2(x) against analytical finite difference
    let (e2, de2_dx) = eval_e2(1.25).unwrap();
    let (e2_p, _) = eval_e2(1.25 + 1.0e-6).unwrap();
    let (e2_m, _) = eval_e2(1.25 - 1.0e-6).unwrap();
    let fd_e2 = (e2_p - e2_m) / 2.0e-6;
    assert!(e2 > 0.0 && e2 < 1.0);
    assert!(((de2_dx - fd_e2) / fd_e2).abs() < 1.0e-7);

    // Check dq_dtau against eval_t_tau finite difference for all 4 T(tau) relations
    for id in [
        ATM_T_TAU_EDDINGTON,
        ATM_T_TAU_SOLAR_HOPF,
        ATM_T_TAU_KRISHNA_SWAMY,
        ATM_T_TAU_TRAMPEDACH_SOLAR,
    ] {
        let tau = 0.5;
        let teff = 5777.0;
        let dq = eval_t_tau_dq_dtau(id, tau).unwrap();
        let eps = 1.0e-6;
        let lnt_p = eval_t_tau(id, tau + eps, teff).unwrap();
        let lnt_m = eval_t_tau(id, tau - eps, teff).unwrap();
        let q_of = |lnt: f64, t_val: f64| -> f64 {
            (4.0 / 3.0) * (lnt.exp() / teff).powi(4) - t_val
        };
        let dq_fd =
            (q_of(lnt_p, tau + eps) - q_of(lnt_m, tau - eps)) / (2.0 * eps);
        // Note: Trampedach uses Ball (2021) smooth transition formula for dq_dtau
        let tol = if id == ATM_T_TAU_TRAMPEDACH_SOLAR {
            1.0e-3
        } else {
            1.0e-7
        };
        assert!((dq - dq_fd).abs() <= tol, "id={id} dq={dq} dq_fd={dq_fd}");
    }

    // Check table partials (skip_partials = false) against finite differences
    let cgrav = STANDARD_CGRAV;
    let m = MSUN;
    let r = RSUN;
    let l = LSUN;
    let teff = atm_teff(l, r);
    let z_sol = flexmesa_atm::table::GN93_ZSOL;
    let out =
        eval_table(l, r, m, cgrav, ATM_TABLE_PHOTOSPHERE, z_sol, false, teff)
            .unwrap();
    let eps = 1.0e-5;
    let r_p = r * (1.0 + eps);
    let r_m = r * (1.0 - eps);
    let out_rp = eval_table(
        l,
        r_p,
        m,
        cgrav,
        ATM_TABLE_PHOTOSPHERE,
        z_sol,
        true,
        atm_teff(l, r_p),
    )
    .unwrap();
    let out_rm = eval_table(
        l,
        r_m,
        m,
        cgrav,
        ATM_TABLE_PHOTOSPHERE,
        z_sol,
        true,
        atm_teff(l, r_m),
    )
    .unwrap();
    let fd_lnp_dlnr = (out_rp.ln_p - out_rm.ln_p) / (r_p.ln() - r_m.ln());
    let fd_lnt_dlnr = (out_rp.ln_t - out_rm.ln_t) / (r_p.ln() - r_m.ln());
    assert!(((out.dln_p_dln_r - fd_lnp_dlnr) / fd_lnp_dlnr).abs() < 1.0e-6);
    assert!(((out.dln_t_dln_r - fd_lnt_dlnr) / fd_lnt_dlnr).abs() < 1.0e-6);

    // Check get_table_alfa_beta and get_table_base
    let (alfa, beta) =
        get_table_alfa_beta(l, teff, r, m, cgrav, ATM_TABLE_PHOTOSPHERE)
            .unwrap();
    assert_eq!(alfa, 0.0);
    assert_eq!(beta, 1.0);
    assert!(
        (get_table_base(ATM_TABLE_PHOTOSPHERE).unwrap() - 2.0 / 3.0).abs()
            < 1.0e-15
    );

    // Check build_t_tau_uniform and build_t_tau_varying outward-in ordering
    let (chem_id, xa) = solar_species_and_xa(0.70, 0.02);
    let prof_u = build_t_tau_uniform(
        2.0 / 3.0,
        l,
        r,
        teff,
        m,
        cgrav,
        0.4,
        1.0,
        1.0e-3,
        ATM_T_TAU_EDDINGTON,
        make_eos_proc(chem_id.clone(), xa.clone()),
        1.0e-5,
        0.1,
    )
    .unwrap();
    assert!(prof_u.points.len() >= 5);
    assert!(prof_u.points[0][ATM_TAU] < prof_u.points.last().unwrap()[ATM_TAU]);
    assert!(prof_u.points[0][ATM_DELTA_R] > 0.0);
    assert_eq!(prof_u.points.last().unwrap()[ATM_DELTA_R], 0.0);

    let surf_v = eval_t_tau_varying(
        2.0 / 3.0,
        l,
        r,
        m,
        cgrav,
        ATM_T_TAU_EDDINGTON,
        make_eos_proc(chem_id.clone(), xa.clone()),
        make_kap_proc(
            chem_id.clone(),
            xa.clone(),
            0.02,
            KapLowTOption::LowTFa05Gs98,
        ),
        1.0e-9,
        500,
        true,
        teff,
    )
    .unwrap();
    let prof_v = build_t_tau_varying(
        2.0 / 3.0,
        l,
        r,
        teff,
        m,
        cgrav,
        surf_v.ln_p,
        1.0e-3,
        ATM_T_TAU_EDDINGTON,
        make_eos_proc(chem_id.clone(), xa.clone()),
        make_kap_proc(chem_id, xa, 0.02, KapLowTOption::LowTFa05Gs98),
        1.0e-5,
        0.1,
    )
    .unwrap();
    assert!(prof_v.points.len() >= 5);
    assert!(prof_v.points[0][ATM_LNP] < prof_v.points.last().unwrap()[ATM_LNP]);
    assert!(prof_v.points[0][ATM_LNT] < prof_v.points.last().unwrap()[ATM_LNT]);
}
