// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#161` (`F2-2`, `F2-3`, `F2-14`, `F2-15`, `F2-24`) star microphysics
//! call-site alignment tests (`do_eos`, `do_neu`, `do_kap`, `do_net`, and
//! profile diagnostic columns).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_const::{AVO, CRAD, FOUR_THIRDS_PI, KERG, LN10, ONE_THIRD, QE};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::mesa_profile::{
    format_star_profile_text, parse_star_profile_columns,
};
use flexmesa_star::micro::{
    change_model_network, do_eos, do_kap, do_net, do_neu, resolve_model_net,
    set_basic_vars,
};
use flexmesa_star::{
    load_zams_model, op_split_burn_zones, StarControls, StarError,
    StarJobControls,
};

fn zams_controls() -> StarControls {
    StarControls {
        zbase: 0.02,
        ..StarControls::default()
    }
}

#[test]
fn test_do_eos_matches_upstream_prad_gam_and_rejects_extreme_states() {
    let mut model =
        load_zams_model(zams_controls(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");

    let k = model.nz - 1;
    let t_k = model.t[k];
    let rho_k = model.rho[k];
    let exp_prad = ((((CRAD * t_k) * t_k) * t_k) * t_k) / 3.0;
    let exp_gam = model.z53bar[k]
        * QE
        * QE
        * flexmesa_math::pow(
            FOUR_THIRDS_PI * AVO * rho_k * model.zbar[k] / model.abar[k],
            ONE_THIRD,
        )
        / (KERG * t_k);
    assert!((model.prad[k] - exp_prad).abs() <= 1.0e-15 * exp_prad);
    assert!((model.gam[k] - exp_gam).abs() <= 1.0e-15 * exp_gam);

    // `star/private/eos_support.f90:70`: reject `logRho < -25`.
    let mut low_rho = model.clone();
    low_rho.lnd[0] = -25.1 * LN10;
    low_rho.rho[0] = flexmesa_math::exp(low_rho.lnd[0]);
    let err = do_eos(&mut low_rho).expect_err("logRho < -25 must fail");
    assert!(err.to_string().contains("logRho"));
}

#[test]
fn test_do_neu_includes_reco_channel_stores_unscaled_sources_and_rejects_hook()
{
    let mut model =
        load_zams_model(zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    let k = model.nz - 1;
    // Set non-degenerate warm core state (`nu in [-20, 10]`) where recombination
    // neutrinos (`reco_neu`) are active (`T = 5e7 K`, `rho = 1e3 g/cm^3`).
    model.ln_t[k] = flexmesa_math::log(5.0e7);
    model.lnd[k] = flexmesa_math::log(1.0e3);
    set_basic_vars(&mut model).expect("set_basic_vars");

    model.controls.non_nuc_neu_factor = 2.5;
    do_neu(&mut model).expect("do_neu");
    let ch = model.aux.nonnucneu_channels[k];
    // Channels `[plas, brem, phot, pair, reco]` are stored unscaled (`star/private/neu.f90:106-115`),
    // while `non_nuc_neu` equals `2.5 * sum(channels)`.
    assert!(
        ch[4] > 0.0,
        "recombination neutrino channel must be active: {ch:?}"
    );
    let sum_unscaled: f64 = ch.iter().sum();
    assert!(
        (model.non_nuc_neu[k] - 2.5 * sum_unscaled).abs()
            <= 1.0e-12 * model.non_nuc_neu[k]
    );

    model.controls.non_nuc_neu_factor = 0.0;
    do_neu(&mut model).expect("do_neu factor=0");
    assert_eq!(model.non_nuc_neu[k], 0.0);
    assert_eq!(model.aux.nonnucneu_channels[k], [0.0; 5]);

    model.controls.use_other_neu = true;
    let err = do_neu(&mut model).expect_err("use_other_neu must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
}

#[test]
fn test_do_kap_simple_es_starting_comp_logt_ramp_and_clamps() {
    let mut model =
        load_zams_model(zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");
    do_kap(&mut model).expect("baseline do_kap");
    let base_kap: Vec<f64> = model.opacity.clone();

    // 1. `use_simple_es_for_kap` (`star/private/kap_support.f90:302-307`)
    model.controls.use_simple_es_for_kap = true;
    do_kap(&mut model).expect("simple es do_kap");
    for k in 0..model.nz {
        let exp = 0.2 * (1.0 + model.xh_mass[k]);
        assert!((model.opacity[k] - exp).abs() <= 1.0e-15);
        assert_eq!(model.d_opacity_dlnd[k], 0.0);
        assert_eq!(model.d_opacity_dlnt[k], 0.0);
        assert_eq!(model.aux.kap_frac[k], [0.0; 4]);
    }

    // 2. `use_starting_composition_for_kap` (`micro.f90:589-595`)
    model.xa_start.clone_from(&model.xa);
    // Perturb current `xa` in zone 0 (`H1 -> He4`) while keeping `xa_start` intact.
    let x_h_orig = model.xh_mass[0];
    model.xa[0] = 0.0;
    model.xa[1] = 1.0 - model.z_mass[0];
    set_basic_vars(&mut model).expect("set_basic_vars perturbed");
    model.controls.use_starting_composition_for_kap = true;
    do_kap(&mut model).expect("starting comp simple es");
    assert!((model.opacity[0] - 0.2 * (1.0 + x_h_orig)).abs() <= 1.0e-14);
    model.controls.use_starting_composition_for_kap = false;
    model.controls.use_simple_es_for_kap = false;
    model.xa.clone_from(&model.xa_start);
    set_basic_vars(&mut model).expect("restore basic_vars");
    do_eos(&mut model).expect("restore do_eos");

    // 3. Temperature-ramped `opacity_factor` (`micro.f90:622-640`)
    model.controls.opacity_factor = 2.0;
    model.controls.rsp.min_logt_for_opacity_factor_off = 5.0;
    model.controls.rsp.min_logt_for_opacity_factor_on = 5.5;
    model.controls.rsp.max_logt_for_opacity_factor_on = 6.5;
    model.controls.rsp.max_logt_for_opacity_factor_off = 7.0;
    do_kap(&mut model).expect("ramped do_kap");
    for k in 0..model.nz {
        let log_t = model.ln_t[k] / LN10;
        let ratio = model.opacity[k] / base_kap[k];
        if log_t <= 5.0 || log_t >= 7.0 {
            assert!((ratio - 1.0).abs() <= 1.0e-14, "zone {k} logT={log_t}");
        } else if (5.5..=6.5).contains(&log_t) {
            assert!((ratio - 2.0).abs() <= 1.0e-14, "zone {k} logT={log_t}");
        } else {
            assert!(ratio > 1.0 && ratio < 2.0, "zone {k} logT={log_t}");
        }
    }

    // 4. `opacity_max` and `opacity_min` clamps (`micro.f90:642-652`)
    model.controls.opacity_factor = 1.0;
    model.controls.rsp.min_logt_for_opacity_factor_off = -1.0;
    model.controls.opacity_max = 1.0;
    model.controls.opacity_min = 0.5;
    do_kap(&mut model).expect("clamped do_kap");
    for k in 0..model.nz {
        assert!((0.5..=1.0).contains(&model.opacity[k]));
        if base_kap[k] > 1.0 || base_kap[k] < 0.5 {
            assert_eq!(model.d_opacity_dlnd[k], 0.0);
            assert_eq!(model.d_opacity_dlnt[k], 0.0);
        }
    }

    model.controls.use_other_kap = true;
    let err = do_kap(&mut model).expect_err("use_other_kap must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
}

#[test]
fn test_do_net_special_rates_neuqs_logtcut_clamp_and_category_renorm() {
    let mut model =
        load_zams_model(zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");

    let net_base = resolve_model_net(&model).expect("resolve basic.net");
    do_net(&mut model, &net_base, false).expect("baseline do_net");
    let kc = model.nz - 1;
    let base_eps = model.eps_nuc[kc];
    let base_neu = model.eps_nuc_neu_total[kc];
    assert!(base_eps > 0.0 && base_neu > 0.0);
    let cat_sum_base: f64 = model.aux.eps_nuc_categories[kc].iter().sum();
    assert!((cat_sum_base - base_eps).abs() <= 1.0e-11 * base_eps);

    // 1. `reaction_neuQs_factor` scales nuclear neutrino losses (`star/private/net.f90:202-210`)
    // and `eps_nuc_factor` does NOT scale `eps_nuc_neu_total` (`star/private/net.f90:365-372`).
    model.controls.reaction_neuqs_factor = 0.5;
    model.controls.eps_nuc_factor = 3.0;
    let net_half_neu = resolve_model_net(&model).expect("resolve half neu");
    do_net(&mut model, &net_half_neu, false).expect("do_net half neu");
    assert!(
        (model.eps_nuc_neu_total[kc] - 0.5 * base_neu).abs()
            <= 1.0e-12 * base_neu
    );
    // Reducing Q_neu increases net thermal `eps_nuc` before the 3x `eps_nuc_factor`.
    let expected_eps = 3.0 * (base_eps + 0.5 * base_neu);
    assert!(
        (model.eps_nuc[kc] - expected_eps).abs() <= 1.0e-11 * expected_eps,
        "got {}, expected {expected_eps}",
        model.eps_nuc[kc]
    );
    let cat_sum_scaled: f64 = model.aux.eps_nuc_categories[kc].iter().sum();
    assert!(
        (cat_sum_scaled - model.eps_nuc[kc]).abs() <= 1.0e-11 * expected_eps
    );

    // Reset factors
    model.controls.reaction_neuqs_factor = 1.0;
    model.controls.eps_nuc_factor = 1.0;

    // 2. `max_abs_eps_nuc` clamps `eps_nuc`, zeroes partials, and renormalizes categories (`star/private/net.f90:311-331`).
    model.controls.max_abs_eps_nuc = 0.25 * base_eps;
    let net_std = resolve_model_net(&model).expect("resolve std");
    do_net(&mut model, &net_std, false).expect("do_net clamped");
    assert!((model.eps_nuc[kc] - 0.25 * base_eps).abs() <= 1.0e-14 * base_eps);
    assert_eq!(model.d_epsnuc_dlnd[kc], 0.0);
    assert_eq!(model.d_epsnuc_dlnt[kc], 0.0);
    let sp = model.species;
    assert!(model.d_epsnuc_dx[kc * sp..(kc + 1) * sp]
        .iter()
        .all(|&v| v == 0.0));
    let cat_sum_clamped: f64 = model.aux.eps_nuc_categories[kc].iter().sum();
    assert!(
        (cat_sum_clamped - model.eps_nuc[kc]).abs()
            <= 1.0e-12 * model.eps_nuc[kc]
    );
    model.controls.max_abs_eps_nuc = 1.0e99;

    // 3. `num_special_rate_factors` scales `rpp_to_he3` linearly and fails on unknown reaction.
    model.job.num_special_rate_factors = 1;
    model.job.reaction_for_special_factor = vec![String::from("rpp_to_he3")];
    model.job.special_rate_factor = vec![0.0];
    do_net(&mut model, &net_std, false).expect("do_net rpp=0");
    let eps_rpp_0 = model.eps_nuc[kc];
    let rpp_contrib = base_eps - eps_rpp_0;
    assert!(
        rpp_contrib > 0.25 * base_eps,
        "rpp_to_he3 contribution ({rpp_contrib}) must exceed 25% of ZAMS core eps_nuc ({base_eps})"
    );
    model.job.special_rate_factor = vec![2.0];
    do_net(&mut model, &net_std, false).expect("do_net rpp=2");
    assert!(
        ((model.eps_nuc[kc] - base_eps) - rpp_contrib).abs()
            <= 1.0e-11 * base_eps
    );

    model.job.reaction_for_special_factor =
        vec![String::from("not_a_reaction")];
    let err = do_net(&mut model, &net_std, false)
        .expect_err("unknown reaction_for_special_factor must fail");
    assert!(matches!(err, StarError::InvalidInput(_)));
    model.job.num_special_rate_factors = 0;
    model.job.reaction_for_special_factor.clear();
    model.job.special_rate_factor.clear();

    // 4. `net_logtcut_lo` / `net_logtcut_lim` cutoff (`net_screen.f90:131-152`).
    let log_tc = model.ln_t[kc] / LN10;
    model.controls.net_logtcut_lo = log_tc + 0.1;
    model.controls.net_logtcut_lim = log_tc + 0.2;
    let net_cut = resolve_model_net(&model).expect("resolve cut");
    do_net(&mut model, &net_cut, false).expect("do_net cut");
    assert_eq!(model.eps_nuc[kc], 0.0);

    // 5. `approx21.net` also honours `reaction_neuQs_factor`.
    model.controls.net_logtcut_lo = -1.0;
    model.controls.net_logtcut_lim = -1.0;
    change_model_network(&mut model, "approx21.net", None)
        .expect("switch to approx21.net");
    set_basic_vars(&mut model).expect("basic_vars approx21");
    do_eos(&mut model).expect("do_eos approx21");
    let net_a21_1 = resolve_model_net(&model).expect("approx21 1.0");
    do_net(&mut model, &net_a21_1, false).expect("do_net approx21 1.0");
    let a21_neu_1 = model.eps_nuc_neu_total[kc];
    assert!(a21_neu_1 > 0.0);
    model.controls.reaction_neuqs_factor = 0.25;
    let net_a21_q = resolve_model_net(&model).expect("approx21 0.25");
    do_net(&mut model, &net_a21_q, false).expect("do_net approx21 0.25");
    assert!(
        (model.eps_nuc_neu_total[kc] - 0.25 * a21_neu_1).abs()
            <= 1.0e-12 * a21_neu_1
    );
}

#[test]
fn test_op_split_burn_zones_threads_rate_factors_weak_fl87_and_unscaled_neu() {
    let base_model =
        load_zams_model(zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    let kc = base_model.nz - 1;
    let sp = base_model.species;
    let setup = |eps_fac: f64,
                 weak_fac: f64,
                 fl87: bool,
                 r3a_fac: Option<f64>|
     -> (f64, f64, f64) {
        let mut m = base_model.clone();
        // Hot, dense H+He core zone (`T = 1.2e8 K`, `rho = 1e5 g/cm^3`)
        // activating both pp weak reactions and strongly screened 3-alpha.
        m.ln_t[kc] = flexmesa_math::log(1.2e8);
        m.ln_t_start = m.ln_t.clone();
        m.lnd[kc] = flexmesa_math::log(1.0e5);
        m.xa[kc * sp..kc * sp + sp]
            .copy_from_slice(&[0.20, 0.0, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0]);
        set_basic_vars(&mut m).expect("set_basic_vars");
        do_eos(&mut m).expect("do_eos");
        m.controls.op_split_burn = true;
        m.controls.op_split_burn_min_t = 1.0e8;
        m.controls.eps_nuc_factor = eps_fac;
        m.controls.weak_rate_factor = weak_fac;
        m.job.use_3a_fl87 = fl87;
        if let Some(fac) = r3a_fac {
            m.job.num_special_rate_factors = 1;
            m.job.reaction_for_special_factor =
                vec![String::from("r_he4_he4_he4_to_c12")];
            m.job.special_rate_factor = vec![fac];
        }
        op_split_burn_zones(&mut m, 1.0).expect("op_split_burn_zones");
        (m.eps_nuc[kc], m.eps_nuc_neu_total[kc], m.xa[kc * sp + 3])
    };

    let (eps_base, neu_base, xc12_base) = setup(1.0, 1.0, false, None);
    assert!(eps_base > 0.0 && neu_base > 0.0 && xc12_base > 0.0);

    // 1. `eps_nuc_factor` scales `eps_nuc` while `eps_nuc_neu_total` stays unscaled.
    let (eps_3x, neu_3x, _) = setup(3.0, 1.0, false, None);
    assert!((eps_3x - 3.0 * eps_base).abs() <= 1.0e-12 * eps_3x);
    assert!((neu_3x - neu_base).abs() <= 1.0e-13 * neu_base);

    // 2. `special_rate_factor` suppresses 3-alpha `c12` synthesis when set to `0`.
    let (_, _, xc12_no_3a) = setup(1.0, 1.0, false, Some(0.0));
    assert!(xc12_no_3a < 1.0e-12 * xc12_base);

    // 3. `weak_rate_factor = 0` suppresses weak electron-capture burning (`irn56ec`)
    //    and neutrino emission on `approx21.net`.
    let eval_a21_weak = |weak_fac: f64| -> (f64, f64) {
        let mut m = base_model.clone();
        change_model_network(&mut m, "approx21.net", None)
            .expect("switch to approx21.net");
        let sp_a21 = m.species;
        let i_ni56 = m
            .iso_names
            .iter()
            .position(|s| s == "ni56")
            .expect("ni56 in approx21");
        let i_fe56 = m
            .iso_names
            .iter()
            .position(|s| s == "fe56")
            .expect("fe56 in approx21");
        m.ln_t[kc] = flexmesa_math::log(5.0e8);
        m.ln_t_start = m.ln_t.clone();
        m.lnd[kc] = flexmesa_math::log(1.0e8);
        m.xa[kc * sp_a21..(kc + 1) * sp_a21].fill(0.0);
        m.xa[kc * sp_a21 + i_ni56] = 1.0;
        set_basic_vars(&mut m).expect("set_basic_vars a21");
        do_eos(&mut m).expect("do_eos a21");
        m.controls.op_split_burn = true;
        m.controls.op_split_burn_min_t = 1.0e8;
        m.controls.weak_rate_factor = weak_fac;
        op_split_burn_zones(&mut m, 10.0).expect("op_split_burn_zones a21");
        (m.eps_nuc_neu_total[kc], m.xa[kc * sp_a21 + i_fe56])
    };
    let (neu_wk_1, xfe_wk_1) = eval_a21_weak(1.0);
    let (neu_wk_0, xfe_wk_0) = eval_a21_weak(0.0);
    assert!(neu_wk_1 > 0.0 && xfe_wk_1 > 0.0);
    assert!(neu_wk_0 < 1.0e-12 * neu_wk_1);
    assert!(xfe_wk_0 < 1.0e-12 * xfe_wk_1);

    // 4. `use_3a_fl87 = true` modifies screened 3-alpha `c12` yield.
    let (_, _, xc12_fl87) = setup(1.0, 1.0, true, None);
    assert!((xc12_fl87 - xc12_base).abs() > 1.0e-3 * xc12_base);
}

#[test]
fn test_profile_columns_and_star_micro_inlist_population() {
    let dir: PathBuf = std::env::temp_dir()
        .join(format!("flexmesa_star_161_wiring_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    let inlist_path = dir.join("inlist");
    fs::write(
        &inlist_path,
        "&star_job\n\
           num_special_rate_factors = 1\n\
           reaction_for_special_factor(1) = 'r_c12_ag_o16'\n\
           special_rate_factor(1) = 1.25d0\n\
           use_3a_fl87 = .true.\n\
           use_suzuki_weak_rates = .true.\n\
         /\n\
         &controls\n\
           use_simple_es_for_kap = .true.\n\
           use_starting_composition_for_kap = .true.\n\
           opacity_max = 12.0d0\n\
           opacity_min = 0.15d0\n\
           net_logTcut_lo = 5.4d0\n\
           net_logTcut_lim = 5.6d0\n\
           weak_rate_factor = 0.5d0\n\
           reaction_neuQs_factor = 0.75d0\n\
           max_abs_eps_nuc = 1.0d14\n\
         /\n",
    )
    .expect("write inlist");
    let bundle = InlistBundle::from_file(&inlist_path).expect("parse bundle");
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(bundle.job.num_special_rate_factors, 1);
    assert_eq!(bundle.job.reaction_for_special_factor[0], "r_c12_ag_o16");
    assert_eq!(bundle.job.special_rate_factor[0], 1.25);
    assert!(bundle.job.use_3a_fl87);
    assert!(bundle.job.use_suzuki_weak_rates);
    assert!(bundle.controls.use_simple_es_for_kap);
    assert!(bundle.controls.use_starting_composition_for_kap);
    assert_eq!(bundle.controls.opacity_max, 12.0);
    assert_eq!(bundle.controls.opacity_min, 0.15);
    assert_eq!(bundle.controls.net_logtcut_lo, 5.4);
    assert_eq!(bundle.controls.net_logtcut_lim, 5.6);
    assert_eq!(bundle.controls.weak_rate_factor, 0.5);
    assert_eq!(bundle.controls.reaction_neuqs_factor, 0.75);
    assert_eq!(bundle.controls.max_abs_eps_nuc, 1.0e14);

    let mut model =
        load_zams_model(zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");
    do_kap(&mut model).expect("do_kap");
    do_neu(&mut model).expect("do_neu");
    let specs = parse_star_profile_columns(
        "zone\nnonnucneu_plas\nnonnucneu_brem\nnonnucneu_phot\nnonnucneu_pair\n\
         nonnucneu_reco\nkap_frac_lowT\nkap_frac_highT\nkap_frac_Type2\n\
         kap_frac_Compton\neos_frac_HELM\neos_frac_OPAL_SCVH\neos_frac_FreeEOS\n\
         eos_frac_PC\neos_frac_Skye\neos_frac_CMS\neos_frac_ideal\n",
        None,
    )
    .expect("parse profile columns");
    let txt = format_star_profile_text(&model, &specs);
    assert!(txt.contains("nonnucneu_reco"));
    assert!(txt.contains("kap_frac_Compton"));
    assert!(txt.contains("eos_frac_FreeEOS"));
    assert!(txt.lines().count() > 10);
}
