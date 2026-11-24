// SPDX-License-Identifier: GPL-3.0-or-later
//! Wave 1 integration and parity tests for Issue `#376`:
//! `atm_option = 'irradiated_grey'` and `&controls` irradiation inputs
//! (`star/defaults/controls.defaults:4292-4297, 4524-4613`,
//! `star/private/atm_support.f90:93-100, 201-204, 736-888`,
//! `atm/private/atm_irradiated.f90:35-332`, and
//! `star/private/hydro_eqns.f90:922-966`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_atm::{
    eval_irradiated, AtmEosCallbackResult, AtmError, AtmIrradiatedOutput,
    AtmKapCallbackResult,
};
use flexmesa_const::{
    BOLTZ_SIGMA, CRAD, LN10, M_JUPITER, PI, R_JUPITER, STANDARD_CGRAV,
};
use flexmesa_eos::{
    eos_dt_get_rho, EosConfig, HelmTable, I_ETA, I_LN_FREE_E, I_LN_PGAS,
};
use flexmesa_kap::{get_kap_results, KapConfig, KapLowTOption};
use flexmesa_star::controls::{
    AtmIrradiatedOpacity, AtmOption, StarControls, StarError, StarJobControls,
};
use flexmesa_star::evolve::star_evolve_step;
use flexmesa_star::hydro_vars::{eval_surf_bc, set_hydro_vars};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::star_ad::{
    I_LND_00, I_LNR_00, I_LNT_00, I_L_00, NUM_STAR_AD_VARS,
};
use flexmesa_star::zams::load_zams_model;

struct TempDirGuard {
    root: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_atm_irrad_376_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp test dir");
        Self { root }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn solar_zams_controls() -> StarControls {
    StarControls {
        zbase: 0.02,
        ..StarControls::default()
    }
}

fn eval_gamma_rho(pressure: f64, temp: f64) -> f64 {
    let mu = 0.6;
    let m_u = 1.660_539_066_60e-24;
    let k_b = 1.380_649e-16;
    (pressure * mu * m_u / (k_b * temp)).max(1.0e-40)
}

fn planet_eos_proc(
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    chem_id: &[usize],
    xa: &[f64],
    ln_p: f64,
    ln_t: f64,
) -> Result<AtmEosCallbackResult, AtmError> {
    let temp = flexmesa_math::exp(ln_t);
    let pressure = flexmesa_math::exp(ln_p);
    let prad = CRAD * flexmesa_math::pow4(temp) / 3.0;
    let pgas = (pressure - prad).max(1.0e-99);
    let rho_guess = eval_gamma_rho(pressure, temp);
    let inv = eos_dt_get_rho(
        ht,
        chem_id,
        xa,
        ln_t / LN10,
        I_LN_PGAS.cast_signed(),
        flexmesa_math::log(pgas),
        flexmesa_math::log10(rho_guess),
        1.0e-11,
        1.0e-11 * LN10,
        100,
        eos_cfg,
    )
    .map_err(|e| AtmError::CallbackError {
        context: "test_eos_proc",
        detail: e.to_string(),
    })?;
    Ok(AtmEosCallbackResult {
        ln_rho: inv.log10_result * LN10,
        res: inv.eos.res,
        dres_dln_rho: inv.eos.d_dlnd,
        dres_dln_t: inv.eos.d_dln_t,
    })
}

fn planet_kap_proc(
    kap_cfg: &KapConfig,
    x_h: f64,
    z_metal: f64,
    zbar: f64,
    ln_rho: f64,
    ln_t: f64,
    eos_res: &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    let ko = get_kap_results(
        kap_cfg,
        zbar,
        x_h,
        z_metal,
        0.0020,
        0.0010,
        0.0040,
        0.0010,
        ln_rho / LN10,
        ln_t / LN10,
        eos_res.res[I_LN_FREE_E],
        eos_res.dres_dln_rho[I_LN_FREE_E],
        eos_res.dres_dln_t[I_LN_FREE_E],
        eos_res.res[I_ETA],
        eos_res.dres_dln_rho[I_ETA],
        eos_res.dres_dln_t[I_ETA],
    )
    .map_err(|e| AtmError::CallbackError {
        context: "test_kap_proc",
        detail: e.to_string(),
    })?;
    Ok(AtmKapCallbackResult {
        kap: ko.kap,
        dlnkap_dln_rho: ko.dlnkap_dln_rho,
        dlnkap_dln_t: ko.dlnkap_dln_t,
    })
}

fn unused_eos(
    _ln_p: f64,
    _ln_t: f64,
) -> Result<AtmEosCallbackResult, AtmError> {
    Err(AtmError::InvalidArgument {
        context: "test",
        detail: String::from("unused"),
    })
}

fn unused_kap(
    _ln_rho: f64,
    _ln_t: f64,
    _eos: &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    Err(AtmError::InvalidArgument {
        context: "test",
        detail: String::from("unused"),
    })
}

#[derive(Clone, Copy)]
struct FixedIrrParams {
    lum: f64,
    rad: f64,
    mass: f64,
    t_eq: f64,
    p_surf: f64,
    kap: f64,
    kap_v: f64,
    gamma: f64,
}

fn eval_fixed_planet_irr(
    p: FixedIrrParams,
    skip_partials: bool,
) -> Result<AtmIrradiatedOutput, AtmError> {
    eval_irradiated(
        p.lum,
        p.rad,
        p.mass,
        STANDARD_CGRAV,
        p.t_eq,
        p.p_surf,
        p.kap,
        p.kap_v,
        p.gamma,
        unused_eos,
        unused_kap,
        1.0e-7,
        0,
        skip_partials,
    )
}

#[test]
fn test_irradiated_controls_defaults_and_inlist_parsing() {
    let defaults = StarControls::default();
    assert_eq!(defaults.atm_irradiated_opacity, AtmIrradiatedOpacity::Fixed);
    assert!((defaults.atm_irradiated_errtol - 1.0e-7).abs() <= 1.0e-20);
    assert_eq!(defaults.atm_irradiated_max_iters, 50);
    assert!((defaults.atm_irradiated_t_eq - 100.0).abs() <= 1.0e-15);
    assert_eq!(defaults.atm_irradiated_kap_v_div_kap_th, 0.0);
    assert!((defaults.atm_irradiated_kap_v - 4.0e-3).abs() <= 1.0e-18);
    assert!((defaults.atm_irradiated_p_surf - 1.0e6).abs() <= 1.0e-10);

    let tmp = TempDirGuard::new("inlist");
    let fixed_path = tmp.root.join("inlist_fixed");
    fs::write(
        &fixed_path,
        "&controls\n\
           atm_option = 'irradiated_grey'\n\
           atm_irradiated_opacity = 'fixed'\n\
           atm_irradiated_T_eq = 1000\n\
           atm_irradiated_kap_v = 4d-3\n\
           atm_irradiated_P_surf = 1d6\n\
         /\n",
    )
    .expect("write fixed inlist");
    let bundle_fixed = InlistBundle::from_file(&fixed_path)
        .expect("irradiated_grey fixed inlist must pass strict validation");
    assert_eq!(
        bundle_fixed.validation_report.unsupported_controls.len(),
        0,
        "unexpected unsupported controls: {:?}",
        bundle_fixed.validation_report.unsupported_controls
    );
    assert_eq!(bundle_fixed.controls.atm_option, AtmOption::IrradiatedGrey);
    assert_eq!(
        bundle_fixed.controls.atm_irradiated_opacity,
        AtmIrradiatedOpacity::Fixed
    );
    assert!(
        (bundle_fixed.controls.atm_irradiated_t_eq - 1000.0).abs() <= 1.0e-12
    );
    assert!(
        (bundle_fixed.controls.atm_irradiated_kap_v - 4.0e-3).abs() <= 1.0e-15
    );
    assert!(
        (bundle_fixed.controls.atm_irradiated_p_surf - 1.0e6).abs() <= 1.0e-9
    );

    let iter_path = tmp.root.join("inlist_iter");
    fs::write(
        &iter_path,
        "&controls\n\
           atm_option = 'irradiated_grey'\n\
           atm_irradiated_opacity = 'iterated'\n\
           atm_irradiated_errtol = 1d-8\n\
           atm_irradiated_max_iters = 75\n\
           atm_irradiated_T_eq = 1200.0d0\n\
           atm_irradiated_kap_v_div_kap_th = 0.4d0\n\
           atm_irradiated_kap_v = 0.0d0\n\
           atm_irradiated_P_surf = 2.0d6\n\
         /\n",
    )
    .expect("write iter inlist");
    let bundle_iter = InlistBundle::from_file(&iter_path)
        .expect("irradiated_grey iterated inlist must pass strict validation");
    assert_eq!(
        bundle_iter.controls.atm_irradiated_opacity,
        AtmIrradiatedOpacity::Iterated
    );
    assert!(
        (bundle_iter.controls.atm_irradiated_errtol - 1.0e-8).abs() <= 1.0e-20
    );
    assert_eq!(bundle_iter.controls.atm_irradiated_max_iters, 75);
    assert!(
        (bundle_iter.controls.atm_irradiated_t_eq - 1200.0).abs() <= 1.0e-12
    );
    assert!(
        (bundle_iter.controls.atm_irradiated_kap_v_div_kap_th - 0.4).abs()
            <= 1.0e-15
    );
    assert_eq!(bundle_iter.controls.atm_irradiated_kap_v, 0.0);
    assert!(
        (bundle_iter.controls.atm_irradiated_p_surf - 2.0e6).abs() <= 1.0e-9
    );
}

#[test]
fn test_irradiated_controls_invalid_inlists_fail_loudly() {
    let tmp = TempDirGuard::new("bad_inlist");
    for (idx, body) in [
        "&controls\n  atm_option = 'grey_irradiated'\n/\n",
        "&controls\n  atm_irradiated_opacity = 'varying'\n/\n",
        "&controls\n  atm_irradiated_opacity = 'bogus'\n/\n",
        "&controls\n  atm_irradiated_errtol = 0.0\n/\n",
        "&controls\n  atm_irradiated_T_eq = -1.0\n/\n",
        "&controls\n  atm_irradiated_kap_v_div_kap_th = -0.1\n/\n",
        "&controls\n  atm_irradiated_kap_v = -1.0e-3\n/\n",
        "&controls\n  atm_irradiated_P_surf = 0.0\n/\n",
        "&controls\n  atm_option = 'irradiated_grey'\n  atm_irradiated_kap_v = 0.0\n/\n",
        "&controls\n  atm_option = 'irradiated_grey'\n  atm_irradiated_opacity = 'iterated'\n  atm_irradiated_max_iters = 0\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let path = tmp.root.join(format!("inlist_bad_{idx}"));
        fs::write(&path, body).expect("write bad inlist");
        let err = InlistBundle::from_file(&path)
            .expect_err("invalid irradiated control must fail loudly");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {body}: got {err}"
        );
    }
}

fn assert_fixed_partials_match_fd(p: FixedIrrParams) {
    let eps = 1.0e-6;
    let base = eval_fixed_planet_irr(p, false).expect("fixed eval_irradiated");
    let dl = p.lum * eps;
    let lum_p = eval_fixed_planet_irr(
        FixedIrrParams {
            lum: p.lum + dl,
            ..p
        },
        true,
    )
    .expect("+L");
    let lum_m = eval_fixed_planet_irr(
        FixedIrrParams {
            lum: p.lum - dl,
            ..p
        },
        true,
    )
    .expect("-L");
    let fd_lum = (lum_p.ln_t - lum_m.ln_t) / (2.0 * dl);
    assert!(
        (base.dln_t_dl - fd_lum).abs() <= 1.0e-7 * fd_lum.abs().max(1.0e-35),
        "fixed gamma={} dlnT/dL={} vs fd={fd_lum}",
        p.gamma,
        base.dln_t_dl
    );

    let rad_p = eval_fixed_planet_irr(
        FixedIrrParams {
            rad: p.rad * flexmesa_math::exp(eps),
            ..p
        },
        true,
    )
    .expect("+lnR");
    let rad_m = eval_fixed_planet_irr(
        FixedIrrParams {
            rad: p.rad * flexmesa_math::exp(-eps),
            ..p
        },
        true,
    )
    .expect("-lnR");
    let fd_rad = (rad_p.ln_t - rad_m.ln_t) / (2.0 * eps);
    assert!(
        (base.dln_t_dln_r - fd_rad).abs() <= 1.0e-7 * fd_rad.abs().max(1.0),
        "fixed gamma={} dlnT/dlnR={} vs fd={fd_rad}",
        p.gamma,
        base.dln_t_dln_r
    );

    let mass_p = eval_fixed_planet_irr(
        FixedIrrParams {
            mass: p.mass * flexmesa_math::exp(eps),
            ..p
        },
        true,
    )
    .expect("+lnM");
    let mass_m = eval_fixed_planet_irr(
        FixedIrrParams {
            mass: p.mass * flexmesa_math::exp(-eps),
            ..p
        },
        true,
    )
    .expect("-lnM");
    let fd_mass = (mass_p.ln_t - mass_m.ln_t) / (2.0 * eps);
    assert!(
        (base.dln_t_dln_m - fd_mass).abs() <= 1.0e-7 * fd_mass.abs().max(1.0),
        "fixed gamma={} dlnT/dlnM={} vs fd={fd_mass}",
        p.gamma,
        base.dln_t_dln_m
    );

    let kap_p = eval_fixed_planet_irr(
        FixedIrrParams {
            kap: p.kap * flexmesa_math::exp(eps),
            ..p
        },
        true,
    )
    .expect("+lnkap");
    let kap_m = eval_fixed_planet_irr(
        FixedIrrParams {
            kap: p.kap * flexmesa_math::exp(-eps),
            ..p
        },
        true,
    )
    .expect("-lnkap");
    let fd_kap = (kap_p.ln_t - kap_m.ln_t) / (2.0 * eps);
    assert!(
        (base.dln_t_dln_kap - fd_kap).abs() <= 1.0e-7 * fd_kap.abs().max(1.0),
        "fixed gamma={} dlnT/dlnkap={} vs fd={fd_kap}",
        p.gamma,
        base.dln_t_dln_kap
    );
}

#[test]
fn test_eval_irradiated_closed_form_zero_teq_and_fixed_partials() {
    let mass = 1.5 * M_JUPITER;
    let rad = 1.5 * R_JUPITER;
    let t_int = 900.0;
    let lum =
        4.0 * PI * rad * rad * BOLTZ_SIGMA * t_int * t_int * t_int * t_int;
    let p_surf = 1.0e6;
    let kap = 1.5e-2;
    let kap_v = 4.0e-3;
    let grav = STANDARD_CGRAV * mass / (rad * rad);
    let tau_exact = p_surf * kap / grav;
    let base_params = FixedIrrParams {
        lum,
        rad,
        mass,
        t_eq: 0.0,
        p_surf,
        kap,
        kap_v,
        gamma: 0.0,
    };

    let zero_teq = eval_fixed_planet_irr(base_params, false)
        .expect("eval_irradiated at T_eq = 0");
    assert!((zero_teq.tau - tau_exact).abs() <= 1.0e-15 * tau_exact);
    let expected_t =
        t_int * flexmesa_math::pow(0.75 * (tau_exact + 2.0 / 3.0), 0.25);
    assert!(
        (flexmesa_math::exp(zero_teq.ln_t) - expected_t).abs()
            <= 1.0e-12 * expected_t
    );

    for (gamma, kv) in [(0.0, kap_v), (0.45, 0.0)] {
        assert_fixed_partials_match_fd(FixedIrrParams {
            t_eq: 1000.0,
            kap_v: kv,
            gamma,
            ..base_params
        });
    }
}

#[test]
fn test_eval_irradiated_invalid_arguments_fail_loudly() {
    let base = FixedIrrParams {
        lum: 1.0e28,
        rad: 1.5 * R_JUPITER,
        mass: 1.5 * M_JUPITER,
        t_eq: 1000.0,
        p_surf: 1.0e6,
        kap: 1.5e-2,
        kap_v: 4.0e-3,
        gamma: 0.0,
    };

    let bad_cases = [
        FixedIrrParams { lum: 0.0, ..base },
        FixedIrrParams { rad: 0.0, ..base },
        FixedIrrParams { mass: 0.0, ..base },
        FixedIrrParams { t_eq: -1.0, ..base },
        FixedIrrParams {
            p_surf: 0.0,
            ..base
        },
        FixedIrrParams { kap: 0.0, ..base },
        FixedIrrParams { kap_v: 0.0, ..base },
        FixedIrrParams {
            gamma: -0.1,
            ..base
        },
    ];
    for case in bad_cases {
        assert!(eval_fixed_planet_irr(case, false).is_err());
    }
    assert!(eval_irradiated(
        base.lum,
        base.rad,
        base.mass,
        STANDARD_CGRAV,
        base.t_eq,
        base.p_surf,
        base.kap,
        base.kap_v,
        0.0,
        unused_eos,
        unused_kap,
        0.0,
        10,
        false
    )
    .is_err());
}

#[test]
fn test_eval_irradiated_iterated_implicit_partials_match_fd() {
    let ht = HelmTable::builtin().expect("helm table");
    let eos_cfg = EosConfig::default();
    let kap_cfg = KapConfig {
        kap_low_t_option: KapLowTOption::LowTFreedman11,
        ..KapConfig::default()
    };
    let chem_id = [1usize, 3, 4];
    let xa = [0.70, 0.29, 0.01];
    let x_h = 0.70;
    let z_metal = 0.01;
    let zbar = 1.1;

    let mass_planet = 1.5 * M_JUPITER;
    let rad_planet = 1.57 * R_JUPITER;
    let t_int = 900.0;
    let lum_planet = 4.0
        * PI
        * rad_planet
        * rad_planet
        * BOLTZ_SIGMA
        * t_int
        * t_int
        * t_int
        * t_int;

    let run = |lum_in: f64, rad_in: f64, mass_in: f64, skip: bool| {
        eval_irradiated(
            lum_in,
            rad_in,
            mass_in,
            STANDARD_CGRAV,
            1000.0,
            1.0e6,
            1.5e-2,
            4.0e-3,
            0.0,
            |ln_p, ln_t| {
                planet_eos_proc(ht, &eos_cfg, &chem_id, &xa, ln_p, ln_t)
            },
            |ln_rho, ln_t, eos_res| {
                planet_kap_proc(
                    &kap_cfg, x_h, z_metal, zbar, ln_rho, ln_t, eos_res,
                )
            },
            1.0e-11,
            60,
            skip,
        )
        .expect("iterated eval_irradiated")
    };

    let base = run(lum_planet, rad_planet, mass_planet, false);
    assert_eq!(base.dln_t_dln_kap, 0.0);

    let eps = 1.0e-5;
    let dl = lum_planet * eps;
    let fd_lum = (run(lum_planet + dl, rad_planet, mass_planet, true).ln_t
        - run(lum_planet - dl, rad_planet, mass_planet, true).ln_t)
        / (2.0 * dl);
    assert!(
        (base.dln_t_dl - fd_lum).abs() <= 1.0e-4 * fd_lum.abs().max(1.0e-35),
        "iterated dlnT/dL analytic={} vs fd={fd_lum}",
        base.dln_t_dl
    );

    let fd_rad = (run(
        lum_planet,
        rad_planet * flexmesa_math::exp(eps),
        mass_planet,
        true,
    )
    .ln_t
        - run(
            lum_planet,
            rad_planet * flexmesa_math::exp(-eps),
            mass_planet,
            true,
        )
        .ln_t)
        / (2.0 * eps);
    assert!(
        (base.dln_t_dln_r - fd_rad).abs() <= 1.0e-4 * fd_rad.abs().max(1.0),
        "iterated dlnT/dlnR analytic={} vs fd={fd_rad}",
        base.dln_t_dln_r
    );

    let fd_mass = (run(
        lum_planet,
        rad_planet,
        mass_planet * flexmesa_math::exp(eps),
        true,
    )
    .ln_t
        - run(
            lum_planet,
            rad_planet,
            mass_planet * flexmesa_math::exp(-eps),
            true,
        )
        .ln_t)
        / (2.0 * eps);
    assert!(
        (base.dln_t_dln_m - fd_mass).abs() <= 1.0e-4 * fd_mass.abs().max(1.0),
        "iterated dlnT/dlnM analytic={} vs fd={fd_mass}",
        base.dln_t_dln_m
    );
}

#[test]
fn test_eval_surf_bc_irradiated_grey_fixed_mode_and_tau_factor() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load ZAMS model");
    model.controls.atm_option = AtmOption::IrradiatedGrey;
    model.controls.atm_irradiated_opacity = AtmIrradiatedOpacity::Fixed;
    model.controls.atm_irradiated_t_eq = 1000.0;
    model.controls.atm_irradiated_kap_v = 4.0e-3;
    model.controls.atm_irradiated_kap_v_div_kap_th = 0.0;
    model.controls.atm_irradiated_p_surf = 1.0e6;
    model.controls.atm_irradiated_errtol = 1.0e-9;
    model.controls.atm_irradiated_max_iters = 50;
    model.tau_base = 10.0;

    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    assert!((model.tau_base - 2.0 / 3.0).abs() <= 1.0e-15);
    let expected_tau_surf = model.controls.atm_irradiated_p_surf
        * model.opacity[0]
        / (model.cgrav[0] * model.m_grav[0] / (model.r[0] * model.r[0]));
    assert!(
        (model.controls.tau_factor - expected_tau_surf / (2.0 / 3.0)).abs()
            <= 1.0e-13 * (expected_tau_surf / (2.0 / 3.0))
    );
    assert!(
        (model.ln_peos_surf_ad.val
            - flexmesa_math::log(model.controls.atm_irradiated_p_surf))
        .abs()
            <= 1.0e-15
    );
    assert_eq!(model.ln_peos_surf_ad.d1, [0.0; NUM_STAR_AD_VARS]);

    let lnt_fixed = model.ln_t_surf_ad;
    let direct_fixed = eval_irradiated(
        model.lum[0],
        model.r[0],
        model.m_grav[0],
        model.cgrav[0],
        model.controls.atm_irradiated_t_eq,
        model.controls.atm_irradiated_p_surf,
        model.opacity[0],
        model.controls.atm_irradiated_kap_v,
        model.controls.atm_irradiated_kap_v_div_kap_th,
        |_ln_p, _ln_t| unreachable!("fixed mode must not invoke eos_proc"),
        |_ln_rho, _ln_t, _eos| {
            unreachable!("fixed mode must not invoke kap_proc")
        },
        model.controls.atm_irradiated_errtol,
        0,
        false,
    )
    .expect("direct eval_irradiated fixed");
    assert!((lnt_fixed.val - direct_fixed.ln_t).abs() <= 1.0e-15);
    assert!((lnt_fixed.d1[I_L_00] - direct_fixed.dln_t_dl).abs() <= 1.0e-15);
    assert!(
        (lnt_fixed.d1[I_LNR_00] - direct_fixed.dln_t_dln_r).abs() <= 1.0e-15
    );
    let k0 = model.opacity[0];
    let partial_by_density =
        direct_fixed.dln_t_dln_kap * (model.d_opacity_dlnd[0] / k0);
    let partial_by_temp =
        direct_fixed.dln_t_dln_kap * (model.d_opacity_dlnt[0] / k0);
    assert!((lnt_fixed.d1[I_LND_00] - partial_by_density).abs() <= 1.0e-15);
    assert!((lnt_fixed.d1[I_LNT_00] - partial_by_temp).abs() <= 1.0e-15);

    // Solver-iteration `opacity_start_surf` behavior (`atm_support.f90:769-773`).
    model.aux.opacity_start_surf = k0;
    model.opacity[0] = k0 * 2.5;
    model.aux.doing_struct_burn_mix = true;
    model.aux.solver_mlt_eval_count = 1;
    eval_surf_bc(&mut model).expect("eval_surf_bc fixed inside solver iter");
    assert!(
        (model.ln_t_surf_ad.val - lnt_fixed.val).abs() <= 1.0e-14,
        "during solver_iter > 0, get_irradiated must use opacity_start_surf"
    );
}

#[test]
fn test_eval_surf_bc_irradiated_grey_iterated_mode_and_loud_errors() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load ZAMS model");
    model.controls.atm_option = AtmOption::IrradiatedGrey;
    model.controls.atm_irradiated_opacity = AtmIrradiatedOpacity::Iterated;
    model.controls.atm_irradiated_t_eq = 1000.0;
    model.controls.atm_irradiated_kap_v = 4.0e-3;
    model.controls.atm_irradiated_p_surf = 1.0e6;
    model.controls.atm_irradiated_errtol = 1.0e-9;
    model.controls.atm_irradiated_max_iters = 50;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let k0 = model.opacity[0];
    model.aux.opacity_start_surf = k0 * 2.5;
    model.aux.doing_struct_burn_mix = true;
    model.aux.solver_mlt_eval_count = 1;
    eval_surf_bc(&mut model).expect("eval_surf_bc iterated");
    let lnt_iter = model.ln_t_surf_ad;
    assert_eq!(model.ln_peos_surf_ad.d1, [0.0; NUM_STAR_AD_VARS]);
    assert_eq!(lnt_iter.d1[I_LND_00], 0.0);
    assert_eq!(lnt_iter.d1[I_LNT_00], 0.0);

    model.aux.doing_struct_burn_mix = false;
    model.aux.solver_mlt_eval_count = 0;
    model.opacity[0] = k0;
    eval_surf_bc(&mut model).expect("eval_surf_bc iterated unperturbed");
    assert!((model.ln_t_surf_ad.val - lnt_iter.val).abs() <= 1.0e-10);

    let eps = 1.0e-5;
    let l0 = model.lum[0];
    let dl = l0 * eps;
    model.lum[0] = l0 + dl;
    eval_surf_bc(&mut model).expect("iter +L");
    let lnt_lum_p = model.ln_t_surf_ad.val;
    model.lum[0] = l0 - dl;
    eval_surf_bc(&mut model).expect("iter -L");
    let lnt_lum_m = model.ln_t_surf_ad.val;
    model.lum[0] = l0;
    let fd_lum = (lnt_lum_p - lnt_lum_m) / (2.0 * dl);
    assert!(
        (model.ln_t_surf_ad.d1[I_L_00] - fd_lum).abs()
            <= 1.0e-4 * fd_lum.abs().max(1.0e-35),
        "iterated eval_surf_bc dlnT/dL={} vs fd={fd_lum}",
        model.ln_t_surf_ad.d1[I_L_00]
    );

    let lnr0 = model.ln_r[0];
    model.ln_r[0] = lnr0 + eps;
    model.r[0] = flexmesa_math::exp(model.ln_r[0]);
    eval_surf_bc(&mut model).expect("iter +lnR");
    let lnt_rad_p = model.ln_t_surf_ad.val;
    model.ln_r[0] = lnr0 - eps;
    model.r[0] = flexmesa_math::exp(model.ln_r[0]);
    eval_surf_bc(&mut model).expect("iter -lnR");
    let lnt_rad_m = model.ln_t_surf_ad.val;
    model.ln_r[0] = lnr0;
    model.r[0] = flexmesa_math::exp(lnr0);
    let fd_rad = (lnt_rad_p - lnt_rad_m) / (2.0 * eps);
    assert!(
        (model.ln_t_surf_ad.d1[I_LNR_00] - fd_rad).abs()
            <= 1.0e-4 * fd_rad.abs().max(1.0),
        "iterated eval_surf_bc dlnT/dlnR={} vs fd={fd_rad}",
        model.ln_t_surf_ad.d1[I_LNR_00]
    );

    let mut bad_p = model.clone();
    bad_p.controls.atm_irradiated_p_surf = -1.0;
    assert!(matches!(
        eval_surf_bc(&mut bad_p),
        Err(StarError::InvalidInput(_))
    ));

    let mut bad_iters = model.clone();
    bad_iters.controls.atm_irradiated_max_iters = 0;
    assert!(matches!(
        eval_surf_bc(&mut bad_iters),
        Err(StarError::InvalidInput(_))
    ));

    let mut bad_lum = model;
    bad_lum.lum[0] = -1.0;
    assert!(matches!(
        eval_surf_bc(&mut bad_lum),
        Err(StarError::MicrophysicsError(_))
    ));
}

#[test]
fn test_eval_surf_bc_irradiated_grey_takes_newton_step_on_zams_model() {
    for opacity_mode in
        [AtmIrradiatedOpacity::Fixed, AtmIrradiatedOpacity::Iterated]
    {
        let mut model =
            load_zams_model(solar_zams_controls(), StarJobControls::default())
                .expect("load ZAMS model");
        set_hydro_vars(&mut model, false, false).expect("init default hydro");
        let p_phot = model.peos[0];
        model.controls.atm_option = AtmOption::IrradiatedGrey;
        model.controls.atm_irradiated_opacity = opacity_mode;
        model.controls.atm_irradiated_t_eq = 1000.0;
        model.controls.atm_irradiated_kap_v = 4.0e-3;
        model.controls.atm_irradiated_p_surf = p_phot;
        set_hydro_vars(&mut model, false, false).expect("init irrad hydro");
        model.save_start_values().expect("save_start_values");
        model.save_old_values();
        let rep = star_evolve_step(&mut model, false, false, true)
            .expect("irradiated_grey Newton step must converge");
        assert!(rep.iterations >= 1);
        assert!(
            model.controls.tau_factor > 0.0
                && model.controls.tau_factor.is_finite()
        );
    }
}
