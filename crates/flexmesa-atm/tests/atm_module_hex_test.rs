// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-atm` against MESA
//! `26.4.1` `atm/test/test_output` (`exact_tested_values.tsv`, 39 outputs).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_atm::{
    atm_teff, eval_irradiated, eval_t_tau_uniform, eval_t_tau_varying,
    eval_table, get_t_tau_base, get_t_tau_id, get_table_id, radiation_pressure,
    AtmEosCallbackResult, AtmError, AtmKapCallbackResult,
};
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{
    BOLTZ_SIGMA, CLIGHT, CRAD, LN10, LSUN, MSUN, M_JUPITER, PI, RSUN,
    STANDARD_CGRAV,
};
use flexmesa_eos::{
    eos_pt_get, eos_pt_get_with_linear, EosConfig, HelmTable, I_ETA,
    I_LN_FREE_E,
};
use flexmesa_kap::{kap_get, KapConfig, KapLowTOption};
use flexmesa_math::{
    exp, exp10, log10, pow, safe_sqrt, with_numerics_mode, NumericsMode,
};

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let ua = a.to_bits();
    let ub = b.to_bits();
    if (ua >> 63) != (ub >> 63) {
        return u64::MAX;
    }
    ua.abs_diff(ub)
}

fn assert_zero_ulp(label: &str, got: f64, expected_bits: u64) {
    let ulp = ulp_distance(got, f64::from_bits(expected_bits));
    assert_eq!(
        ulp,
        0,
        "{label}: expected {expected_bits:#018x}, got {:#018x} ({ulp} ULP)",
        got.to_bits()
    );
}

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
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cfg = EosConfig::default();
    move |ln_p: f64, ln_t: f64| {
        let t = exp(ln_t);
        let p = exp(ln_p);
        let prad = radiation_pressure(t);
        let pgas = (p - prad).max(1.0e-99);
        let log_pgas = log10(pgas);
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

/// Mirrors `eosPT_get` in `eos/private/eospt_eval.f90:148-206` by passing the
/// caller's linear `(Pgas, T)` directly into `eos_pt_get_with_linear`.
fn make_eos_proc_linear_guess(
    chem_id: Vec<usize>,
    xa: Vec<f64>,
) -> impl FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError> {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cfg = EosConfig::default();
    move |ln_p: f64, ln_t: f64| {
        let t = exp(ln_t);
        let p = exp(ln_p);
        let prad = radiation_pressure(t);
        let pgas = (p - prad).max(1.0e-99);
        let log_pgas = log10(pgas);
        let log_t = ln_t / LN10;
        let pt = eos_pt_get_with_linear(
            ht,
            &chem_id,
            &xa,
            log_pgas,
            log_t,
            Some(pgas),
            Some(t),
            &cfg,
        )
        .map_err(|e| AtmError::CallbackError {
            context: "eos_proc_linear_guess",
            detail: e.to_string(),
        })?;
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
fn test_atm_table_module_hex_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cgrav = STANDARD_CGRAV;
        let table_refs: [(&str, u64, u64, u64); 7] = [
            (
                "tau_1m1",
                0x400da74f1d9acb1d,
                0x401251845dc59a41,
                0x4012517f4b2e2069,
            ),
            (
                "tau_1",
                0x400e5c46d3afb65c,
                0x401436cc9848a563,
                0x401436c8c03d7e7d,
            ),
            (
                "tau_10",
                0x400f78e90e4dbec3,
                0x4014d0dcef5e5ed3,
                0x4014d0d32827b63a,
            ),
            (
                "tau_100",
                0x400ffdeed4696b14,
                0x401547816f1facc5,
                0x40154773d01d7d33,
            ),
            (
                "WD_tau_25",
                0x400e436abe9e89a4,
                0x401f946ce352e7a7,
                0x401f946ce20e5ec7,
            ),
            (
                "DB_WD_tau_25",
                0x4012dcc2e5afac9f,
                0x401d89f66b0fb64d,
                0x401d89a027082649,
            ),
            (
                "photosphere",
                0x400e173263e44194,
                0x4013ab01dc5fdad1,
                0x4013aafe00b65f3b,
            ),
        ];

        for (tname, exp_logt, exp_logp, exp_logpgas) in table_refs {
            let tid = get_table_id(tname).unwrap();
            let (l_val, r_val, m_val, teff) = if tname == "WD_tau_25"
                || tname == "DB_WD_tau_25"
            {
                let logg = if tname == "WD_tau_25" { 7.1 } else { 8.0 };
                let teff = if tname == "WD_tau_25" {
                    5000.0
                } else {
                    25000.0
                };
                let m = 0.8 * MSUN;
                let r = safe_sqrt(cgrav * m / exp10(logg));
                let l = PI * CRAD * CLIGHT * r * r * teff * teff * teff * teff;
                (l, r, m, teff)
            } else {
                let teff =
                    pow(LSUN / (4.0 * PI * RSUN * RSUN * BOLTZ_SIGMA), 0.25);
                (LSUN, RSUN, MSUN, teff)
            };
            let out =
                eval_table(l_val, r_val, m_val, cgrav, tid, 0.02, true, teff)
                    .unwrap();
            let t_val = exp(out.ln_t);
            let p_val = exp(out.ln_p);
            let prad = (CRAD * (t_val * t_val) * (t_val * t_val)) / 3.0;
            assert_zero_ulp(
                &format!("{tname} logT"),
                out.ln_t / LN10,
                exp_logt,
            );
            assert_zero_ulp(&format!("{tname} logP"), log10(p_val), exp_logp);
            assert_zero_ulp(
                &format!("{tname} logPgas"),
                log10(p_val - prad),
                exp_logpgas,
            );
        }
    });
}

#[test]
fn test_atm_t_tau_uniform_fixed_and_analytic_dopri5_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cgrav = STANDARD_CGRAV;
        let (chem_id, xa_sol) = solar_species_and_xa(0.70, 0.02);
        let tid_edd = get_t_tau_id("Eddington").unwrap();

        let m_fix = 1.9892000000000002e32;
        let r_fix = 6.3556231577545586e10;
        let l_fix = 2.4015399190199118e32;
        let teff_fix = atm_teff(l_fix, r_fix);
        let out_fix = eval_t_tau_uniform(
            100.0,
            l_fix,
            r_fix,
            m_fix,
            cgrav,
            5.8850802481174469e-02,
            1.0,
            tid_edd,
            make_eos_proc(chem_id.clone(), xa_sol.clone()),
            make_kap_proc(chem_id, xa_sol, 0.02, KapLowTOption::LowTFa05Gs98),
            1.0e-6,
            0,
            true,
            teff_fix,
        )
        .unwrap();
        assert_zero_ulp(
            "uniform fixed logP",
            out_fix.ln_p / LN10,
            0x401afcf1a2a4ade1,
        );
        assert_zero_ulp(
            "uniform fixed logT",
            out_fix.ln_t / LN10,
            0x400f993db407a0aa,
        );

        let r_s = exp10(log10(RSUN));
        let m_s = exp10(log10(MSUN));
        let teff_s = exp10(log10(5776.0));
        let l_s =
            PI * CRAD * CLIGHT * r_s * r_s * teff_s * teff_s * teff_s * teff_s;

        let mut eos_analytic = |ln_p: f64, ln_t: f64| {
            Ok(AtmEosCallbackResult {
                ln_rho: ln_p - 18.0 - ln_t,
                res: [0.0; 26],
                dres_dln_rho: [0.0; 26],
                dres_dln_t: [0.0; 26],
            })
        };
        let mut kap_analytic =
            |ln_rho: f64, ln_t: f64, _: &AtmEosCallbackResult| {
                Ok(AtmKapCallbackResult {
                    kap: 1.0e-3
                        * exp(0.25 * (ln_rho + 18.0) + 0.5 * (ln_t - 8.5)),
                    dlnkap_dln_rho: 0.25,
                    dlnkap_dln_t: 0.5,
                })
            };

        let tid_tr = get_t_tau_id("Trampedach_solar").unwrap();
        let analytic_refs = [
            (
                "Eddington tau_base",
                tid_edd,
                get_t_tau_base(tid_edd).unwrap(),
                0x402e7a3343ce66fd_u64,
                0x401a78e7b04695cd_u64,
            ),
            (
                "Eddington tau=100",
                tid_edd,
                100.0,
                0x403310b30658b280_u64,
                0x40208f5251d55b75_u64,
            ),
            (
                "Trampedach_solar",
                tid_tr,
                get_t_tau_base(tid_tr).unwrap(),
                0x402e10baa7e78d0c_u64,
                0x401a1d4b41f17edb_u64,
            ),
        ];
        for (name, tid, tau_b, exp_lnp, exp_logp) in analytic_refs {
            let out = eval_t_tau_varying(
                tau_b,
                l_s,
                r_s,
                m_s,
                cgrav,
                tid,
                &mut eos_analytic,
                &mut kap_analytic,
                1.0e-9,
                500,
                true,
                teff_s,
            )
            .unwrap();
            assert_zero_ulp(&format!("{name} lnP"), out.ln_p, exp_lnp);
            assert_zero_ulp(
                &format!("{name} logP"),
                log10(exp(out.ln_p)),
                exp_logp,
            );
        }
    });
}

#[test]
fn test_atm_kap_standalone_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let (cid_sol, xa_sol) = solar_species_and_xa(0.70, 0.02);
        let cfg_fa05 = KapConfig {
            zbase: 0.02,
            kap_low_t_option: KapLowTOption::LowTFa05Gs98,
            ..KapConfig::default()
        };
        let (cid_irr, xa_irr) = solar_species_and_xa(0.70, 0.01);
        let cfg_fr = KapConfig {
            zbase: 0.01,
            kap_low_t_option: KapLowTOption::LowTFreedman11,
            ..KapConfig::default()
        };

        let cases = [
            (
                "uniform_iter #2",
                &cfg_fa05,
                &cid_sol,
                &xa_sol,
                0xc035a14c01d17323_u64,
                0x4023afc6ce07587c_u64,
                0xbfd138c908a8c40a_u64,
                0xc03205d168d36c9b_u64,
                0x40006fdbbc22eae3_u64,
            ),
            (
                "varying_edd #2",
                &cfg_fa05,
                &cid_sol,
                &xa_sol,
                0xc036c22cfa0ed295_u64,
                0x4020f9f30aac056c_u64,
                0xc02028097dbdb25c_u64,
                0xc038ed77ceb48709_u64,
                0x3f533f5a131349e3_u64,
            ),
            (
                "irradiated #2",
                &cfg_fr,
                &cid_irr,
                &xa_irr,
                0xc025c2a3cdcc1d0c_u64,
                0x401d4933d157cc36_u64,
                0xc0226bb1bbb55516_u64,
                0xc034000000000000_u64,
                0x3f6b6656d95bed00_u64,
            ),
            (
                "irradiated #8",
                &cfg_fr,
                &cid_irr,
                &xa_irr,
                0xc0257259559edeed_u64,
                0x401ca89f45c00cb0_u64,
                0xc0226bb1bbb55516_u64,
                0xc034000000000000_u64,
                0x3f71e10204a2be02_u64,
            ),
        ];
        for (name, cfg, cid, xa, lnr, lnt, lnfe, eta, exp_kap) in cases {
            let out = kap_get(
                cfg,
                cid,
                xa,
                f64::from_bits(lnr) / LN10,
                f64::from_bits(lnt) / LN10,
                f64::from_bits(lnfe),
                0.0,
                0.0,
                f64::from_bits(eta),
                0.0,
                0.0,
            )
            .unwrap();
            assert_zero_ulp(name, out.kap, exp_kap);
        }
    });
}

#[test]
fn test_atm_linear_pt_eos_module_hex_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cgrav = STANDARD_CGRAV;
        let (cid_sol, xa_sol) = solar_species_and_xa(0.70, 0.02);
        let r_s = exp10(log10(RSUN));
        let m_s = exp10(log10(MSUN));
        let teff_s = exp10(log10(5776.0));
        let l_s =
            PI * CRAD * CLIGHT * r_s * r_s * teff_s * teff_s * teff_s * teff_s;

        let varying_refs: [(&str, f64, u64, u64); 5] = [
            ("Eddington", -1.0, 0x400e17cffe4b7e0c, 0x4013d1e7278351eb),
            (
                "Krishna_Swamy",
                -1.0,
                0x400e17cffdaa30c0,
                0x4013189a8c6d9828,
            ),
            ("solar_Hopf", -1.0, 0x400e17cffe4b7859, 0x4013589c35f91e04),
            (
                "Trampedach_solar",
                -1.0,
                0x400e17d0382f5fb9,
                0x401392e94e4b8a8a,
            ),
            ("Eddington", 100.0, 0x4010eca920ca9dcf, 0x4014d5bb8fa7f1b2),
        ];
        for (rel_name, tau_in, exp_t, exp_p) in varying_refs {
            let tid = get_t_tau_id(rel_name).unwrap();
            let tau_b = if tau_in < 0.0 {
                get_t_tau_base(tid).unwrap()
            } else {
                tau_in
            };
            let out_v = eval_t_tau_varying(
                tau_b,
                l_s,
                r_s,
                m_s,
                cgrav,
                tid,
                make_eos_proc_linear_guess(cid_sol.clone(), xa_sol.clone()),
                make_kap_proc(
                    cid_sol.clone(),
                    xa_sol.clone(),
                    0.02,
                    KapLowTOption::LowTFa05Gs98,
                ),
                1.0e-9,
                500,
                true,
                teff_s,
            )
            .unwrap();
            assert_zero_ulp(
                &format!("varying {rel_name} ({tau_b}) logT"),
                log10(exp(out_v.ln_t)),
                exp_t,
            );
            let got_lp = log10(exp(out_v.ln_p));
            assert_zero_ulp(
                &format!("varying {rel_name} ({tau_b}) logP"),
                got_lp,
                exp_p,
            );
        }

        let tid_edd = get_t_tau_id("Eddington").unwrap();
        let out_it = eval_t_tau_uniform(
            150.0,
            l_s,
            r_s,
            m_s,
            cgrav,
            0.5,
            1.0,
            tid_edd,
            make_eos_proc_linear_guess(cid_sol.clone(), xa_sol.clone()),
            make_kap_proc(cid_sol, xa_sol, 0.02, KapLowTOption::LowTFa05Gs98),
            1.0e-6,
            30,
            true,
            teff_s,
        )
        .unwrap();
        assert_zero_ulp(
            "uniform iter logP",
            out_it.ln_p / LN10,
            0x4012fe5476695113,
        );
        assert_zero_ulp(
            "uniform iter logT",
            out_it.ln_t / LN10,
            0x4011197e8ba8a14d,
        );

        let (cid_irr, xa_irr) = solar_species_and_xa(0.70, 0.01);
        let m_irr = 1.5 * M_JUPITER;
        let r_irr = safe_sqrt(cgrav * m_irr * 10.0 / (1.0e6 * 1.5e-2));
        let l_irr = PI * CRAD * CLIGHT * r_irr * r_irr * pow(900.0, 4.0);
        let out_irr = eval_irradiated(
            l_irr,
            r_irr,
            m_irr,
            cgrav,
            1000.0,
            1.0e6,
            1.5e-2,
            4.0e-3,
            0.0,
            make_eos_proc_linear_guess(cid_irr.clone(), xa_irr.clone()),
            make_kap_proc(cid_irr, xa_irr, 0.01, KapLowTOption::LowTFreedman11),
            1.0e-6,
            30,
            true,
        )
        .unwrap();
        assert_zero_ulp("irrad kap", out_irr.kap, 0x3f71e10204a2be02);
        assert_zero_ulp("irrad tau", out_irr.tau, 0x400747e92a1d7a46);
        assert_zero_ulp("irrad T", exp(out_irr.ln_t), 0x409433bd66451f06);
        assert_zero_ulp("irrad logT", out_irr.ln_t / LN10, 0x4008e48250523706);
    });
}

#[test]
fn test_atm_eos_pt_get_module_hex_subset_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cgrav = STANDARD_CGRAV;
        let (cid_sol, xa_sol) = solar_species_and_xa(0.70, 0.02);
        let r_s = exp10(log10(RSUN));
        let m_s = exp10(log10(MSUN));
        let teff_s = exp10(log10(5776.0));
        let l_s =
            PI * CRAD * CLIGHT * r_s * r_s * teff_s * teff_s * teff_s * teff_s;

        for (rel_name, tau_in, exp_t) in [
            ("Eddington", -1.0, 0x400e17cffe4b7e0c_u64),
            ("Krishna_Swamy", -1.0, 0x400e17cffdaa30c0_u64),
            ("solar_Hopf", -1.0, 0x400e17cffe4b7859_u64),
            ("Trampedach_solar", -1.0, 0x400e17d0382f5fb9_u64),
            ("Eddington", 100.0, 0x4010eca920ca9dcf_u64),
        ] {
            let tid = get_t_tau_id(rel_name).unwrap();
            let tau_b = if tau_in < 0.0 {
                get_t_tau_base(tid).unwrap()
            } else {
                tau_in
            };
            let out_v = eval_t_tau_varying(
                tau_b,
                l_s,
                r_s,
                m_s,
                cgrav,
                tid,
                make_eos_proc(cid_sol.clone(), xa_sol.clone()),
                make_kap_proc(
                    cid_sol.clone(),
                    xa_sol.clone(),
                    0.02,
                    KapLowTOption::LowTFa05Gs98,
                ),
                1.0e-9,
                500,
                true,
                teff_s,
            )
            .unwrap();
            assert_zero_ulp(
                &format!("varying {rel_name} ({tau_b}) logT"),
                log10(exp(out_v.ln_t)),
                exp_t,
            );
        }

        let tid_edd = get_t_tau_id("Eddington").unwrap();
        let out_it = eval_t_tau_uniform(
            150.0,
            l_s,
            r_s,
            m_s,
            cgrav,
            0.5,
            1.0,
            tid_edd,
            make_eos_proc(cid_sol.clone(), xa_sol.clone()),
            make_kap_proc(cid_sol, xa_sol, 0.02, KapLowTOption::LowTFa05Gs98),
            1.0e-6,
            30,
            true,
            teff_s,
        )
        .unwrap();
        assert_zero_ulp(
            "uniform iter logT",
            out_it.ln_t / LN10,
            0x4011197e8ba8a14d,
        );

        let (cid_irr, xa_irr) = solar_species_and_xa(0.70, 0.01);
        let m_irr = 1.5 * M_JUPITER;
        let r_irr = safe_sqrt(cgrav * m_irr * 10.0 / (1.0e6 * 1.5e-2));
        let l_irr = PI * CRAD * CLIGHT * r_irr * r_irr * pow(900.0, 4.0);
        let out_irr = eval_irradiated(
            l_irr,
            r_irr,
            m_irr,
            cgrav,
            1000.0,
            1.0e6,
            1.5e-2,
            4.0e-3,
            0.0,
            make_eos_proc(cid_irr.clone(), xa_irr.clone()),
            make_kap_proc(cid_irr, xa_irr, 0.01, KapLowTOption::LowTFreedman11),
            1.0e-6,
            30,
            true,
        )
        .unwrap();
        assert_zero_ulp("irrad T", exp(out_irr.ln_t), 0x409433bd66451f06);
        assert_zero_ulp("irrad logT", out_irr.ln_t / LN10, 0x4008e48250523706);
    });
}
