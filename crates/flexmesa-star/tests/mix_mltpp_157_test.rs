// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone, zero-`$MESA_DIR` invariant tests for MLT++
//! (`eval_lrad_div_ledd`, `set_gradt_excess_alpha`, `adjust_gradt_excess`,
//! `adjust_gradt_fraction`), mixing-state smoothing helpers, and
//! predictive-mixing / overshooting boundary edge cases.

#![forbid(unsafe_code)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CRAD, FOUR_PI, LN10, M_SUN, STANDARD_CGRAV};
use flexmesa_star::controls::{StarControls, StarJobControls};
use flexmesa_star::hydro_vars::set_hydro_vars;
use flexmesa_star::mix_info::{
    adjust_gradt_excess, adjust_gradt_fraction, eval_lrad_div_ledd,
    set_gradt_excess_alpha,
};
use flexmesa_star::mixing_state::{threshold_smoothing, weighted_smoothing};
use flexmesa_star::model::StarModel;
use flexmesa_star::overshoot::add_overshooting;
use flexmesa_star::predictive_mix::add_predictive_mixing;
use flexmesa_star::star_ad::{StarAdExt, StarAdOrder1, I_LND_00, I_LNT_00};
use flexmesa_star::zams::load_zams_model;

fn chem_ids(names: &[&str]) -> Vec<usize> {
    let catalog = NuclideCatalog::builtin_ref();
    names
        .iter()
        .map(|name| {
            catalog
                .get_by_name(name)
                .unwrap_or_else(|| panic!("missing builtin isotope {name}"))
                .chem_id
        })
        .collect()
}

fn make_mltpp_model(nz: usize, iso_names: &[&str]) -> StarModel {
    let mut controls = StarControls::default();
    controls.mix_wind.okay_to_reduce_gradt_excess = true;
    let ids = chem_ids(iso_names);
    let mut model =
        StarModel::new(nz, ids, StarJobControls::default(), controls);
    model.generations = 1;
    let nz_f = f64::from(u32::try_from(nz.max(1)).expect("small nz"));
    for k in 0..nz {
        let k_f = f64::from(u32::try_from(k).expect("small k"));
        let rem_f = f64::from(u32::try_from(nz - k).expect("small nz-k"));
        model.t[k] = 1.0e6 * (1.0 + 0.25 * k_f);
        model.ln_t[k] = flexmesa_math::log(model.t[k]);
        model.r[k] = 1.0e11 * (rem_f / nz_f);
        model.dm_bar[k] = 1.0e28;
        model.m_grav[k] = 10.0 * M_SUN * (rem_f / nz_f);
        model.cgrav[k] = STANDARD_CGRAV;
        model.pgas[k] = 5.0e14;
        model.ln_pgas[k] = flexmesa_math::log(model.pgas[k]);
        model.peos[k] = 1.0e15;
        model.ln_peos[k] = flexmesa_math::log(model.peos[k]);
        model.tau[k] = 1.0e4;
        model.gradt[k] = 0.45;
        model.gradt_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(0.45, -0.02, 0.08);
        model.gradr_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(0.60, 0.05, -0.15);
        model.aux.mixing.grada_face[k] = 0.25;
        model.aux.mixing.grada_face_ad[k] =
            StarAdOrder1::from_dlnd_dlnt_00(0.25, 0.01, -0.04);
    }
    model
}

fn expected_lrad_div_ledd(
    t: &[f64],
    r: &[f64],
    dm_bar: &[f64],
    cgrav: &[f64],
    m_grav: &[f64],
    k: usize,
) -> f64 {
    let nz = t.len();
    if k >= nz || nz < 2 || cgrav[k] <= 0.0 {
        return 0.0;
    }
    let j = if k == 0 { 1 } else { k };
    let del_m = dm_bar[j];
    if del_m == 0.0 || cgrav[j] <= 0.0 || m_grav[j] == 0.0 {
        return 0.0;
    }
    let del_t4 = flexmesa_math::pow4(t[j - 1]) - flexmesa_math::pow4(t[j]);
    let area = FOUR_PI * r[j] * r[j];
    -(area * area * CRAD * (del_t4 / del_m) / 3.0)
        / (FOUR_PI * cgrav[j] * m_grav[j])
}

fn configure_beta_lambda(
    model: &mut StarModel,
    beta_pgas_ratio: f64,
    k_beta: usize,
    target_lambda: f64,
    k_lambda: usize,
) {
    assert!(k_lambda >= 1 && k_lambda <= k_beta);
    let nz = model.nz;
    for k in 0..nz {
        model.peos[k] = 1.0e15;
        model.pgas[k] = 0.99 * model.peos[k];
        model.t[k] = 1.0e7;
    }
    model.pgas[k_beta] = beta_pgas_ratio * model.peos[k_beta];

    // Enforce target Lrad/Ledd at face k_lambda using upstream scaling:
    // -(area^2 * CRAD * (T_{j-1}^4 - T_j^4) / dm_bar_j / 3) / (4*pi*G*m_grav_j) = target_lambda
    let j = k_lambda;
    let area = FOUR_PI * model.r[j] * model.r[j];
    let ledd = FOUR_PI * model.cgrav[j] * model.m_grav[j];
    let t_bot = model.t[j];
    let t_bot4 = flexmesa_math::pow4(t_bot);
    let rhs =
        target_lambda * ledd * 3.0 * model.dm_bar[j] / (area * area * CRAD);
    let t_top4 = (t_bot4 - rhs).max(1.0);
    model.t[j - 1] = flexmesa_math::pow(t_top4, 0.25);
}

#[test]
fn test_eval_lrad_div_ledd_exact_edge_and_formula() {
    let mut model = make_mltpp_model(4, &["h1", "he4"]);
    assert_eq!(eval_lrad_div_ledd(&model, 4).to_bits(), 0.0_f64.to_bits());

    // Face k = 0 uses shifted index j = 1 (`star_utils.f90:2127-2128`).
    let f0 = eval_lrad_div_ledd(&model, 0);
    let f1 = eval_lrad_div_ledd(&model, 1);
    assert_eq!(f0.to_bits(), f1.to_bits());

    for k in 0..model.nz {
        let got = eval_lrad_div_ledd(&model, k);
        let expected = expected_lrad_div_ledd(
            &model.t,
            &model.r,
            &model.dm_bar,
            &model.cgrav,
            &model.m_grav,
            k,
        );
        assert_eq!(got.to_bits(), expected.to_bits(), "face k={k}");
    }

    // cgrav <= 0 returns exact 0.0 (`star_utils.f90:2125`).
    model.cgrav[0] = 0.0;
    assert_eq!(eval_lrad_div_ledd(&model, 0).to_bits(), 0.0_f64.to_bits());
    model.cgrav[2] = -1.0;
    assert_eq!(eval_lrad_div_ledd(&model, 2).to_bits(), 0.0_f64.to_bits());

    // nz < 2 returns 0.0 cleanly without out-of-bounds indexing.
    let single = make_mltpp_model(1, &["h1"]);
    assert_eq!(eval_lrad_div_ledd(&single, 0).to_bits(), 0.0_f64.to_bits());
}

#[test]
fn test_set_gradt_excess_alpha_gates_and_net_iso_lookup() {
    // 1. Disabled control flag forces alpha = 0.0.
    let mut m = make_mltpp_model(4, &["h1", "he4"]);
    m.controls.mix_wind.okay_to_reduce_gradt_excess = false;
    m.aux.mixing.gradt_excess_alpha = 0.77;
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 0.0_f64.to_bits());

    // 2. Hydrogen gate inspects net_iso(ih1), not combined xh_mass (h1 + h2).
    let mut hydrogen_model = make_mltpp_model(4, &["h1", "h2", "he4"]);
    hydrogen_model.controls.mix_wind.gradt_excess_lambda1 = -1.0;
    hydrogen_model.controls.mix_wind.gradt_excess_max_center_h1 = 0.40;
    let nz = hydrogen_model.nz;
    let sp = hydrogen_model.species;
    hydrogen_model.xa[(nz - 1) * sp] = 0.35; // h1 <= 0.40
    hydrogen_model.xa[(nz - 1) * sp + 1] = 0.30; // h2
    hydrogen_model.xa[(nz - 1) * sp + 2] = 0.35; // he4
    hydrogen_model.xh_mass[nz - 1] = 0.65; // combined H mass exceeds 0.40, must not gate!
    set_gradt_excess_alpha(&mut hydrogen_model);
    assert_eq!(
        hydrogen_model.aux.mixing.gradt_excess_alpha.to_bits(),
        1.0_f64.to_bits()
    );

    // Setting actual h1 > max_center_h1 zeros alpha.
    hydrogen_model.xa[(nz - 1) * sp] = 0.41;
    set_gradt_excess_alpha(&mut hydrogen_model);
    assert_eq!(
        hydrogen_model.aux.mixing.gradt_excess_alpha.to_bits(),
        0.0_f64.to_bits()
    );

    // 3. Helium gate inspects net_iso(ihe4), not combined yhe_mass (he3 + he4).
    let mut helium_model = make_mltpp_model(4, &["he3", "he4", "c12"]);
    helium_model.controls.mix_wind.gradt_excess_lambda1 = -1.0;
    helium_model.controls.mix_wind.gradt_excess_min_center_he4 = 0.50;
    let sp_he = helium_model.species;
    helium_model.xa[(nz - 1) * sp_he] = 0.40; // he3
    helium_model.xa[(nz - 1) * sp_he + 1] = 0.20; // he4 < 0.50
    helium_model.xa[(nz - 1) * sp_he + 2] = 0.40; // c12
    helium_model.yhe_mass[nz - 1] = 0.60; // combined He exceeds 0.50, must still gate on he4!
    set_gradt_excess_alpha(&mut helium_model);
    assert_eq!(
        helium_model.aux.mixing.gradt_excess_alpha.to_bits(),
        0.0_f64.to_bits()
    );

    // 4. Non-H / non-He net (e.g. CO/ONe mixture) skips both missing-species gates.
    let mut wd_net = make_mltpp_model(4, &["c12", "o16", "ne20"]);
    wd_net.controls.mix_wind.gradt_excess_lambda1 = -1.0;
    wd_net.controls.mix_wind.gradt_excess_max_center_h1 = 0.10;
    wd_net.controls.mix_wind.gradt_excess_min_center_he4 = 0.50;
    let sp_wd = wd_net.species;
    wd_net.xa[(nz - 1) * sp_wd] = 0.50;
    wd_net.xa[(nz - 1) * sp_wd + 1] = 0.50;
    // Deliberately poison legacy mass totals to ensure they are ignored.
    wd_net.xh_mass[nz - 1] = 0.99;
    wd_net.yhe_mass[nz - 1] = 0.0;
    set_gradt_excess_alpha(&mut wd_net);
    assert_eq!(
        wd_net.aux.mixing.gradt_excess_alpha.to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn test_set_gradt_excess_alpha_beta_lambda_plane_all_branches() {
    // Verify `pgas / peos` (not `exp(ln_pgas)`) and `(1 + xa(1, nz))` factor,
    // plus `k_beta == 0` -> `max_lambda == 0.0`.
    let mut m0 = make_mltpp_model(5, &["h1", "he4"]);
    let nz = m0.nz;
    let sp = m0.species;
    m0.xa[(nz - 1) * sp] = 0.25; // center h1
    m0.pgas[0] = 0.20 * m0.peos[0];
    // Corrupt ln_pgas to prove `pgas / peos` is used directly (`turb_info.f90:444`).
    m0.ln_pgas[0] = 999.0;
    set_gradt_excess_alpha(&mut m0);
    let expected_beta = (0.20_f64 * 1.0e15 / 1.0e15) * (1.0 + 0.25);
    assert_eq!(
        m0.aux.mixing.gradt_excess_min_beta.to_bits(),
        expected_beta.to_bits()
    );
    assert_eq!(
        m0.aux.mixing.gradt_excess_max_lambda.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        m0.aux.mixing.gradt_excess_alpha.to_bits(),
        0.0_f64.to_bits()
    );

    // Now place k_beta at interior index 4 and evaluate all regions in (beta, lambda):
    // Defaults: lambda1=1.0, beta1=0.35, lambda2=0.5, beta2=0.25, dlambda=0.1, dbeta=0.1
    let mut base = make_mltpp_model(6, &["h1", "he4"]);
    base.xa[(base.nz - 1) * base.species] = 0.0; // so min_beta == beta_pgas_ratio

    // Branch A: lambda >= lambda1 (= 1.0, via min(1.0, raw_lambda))
    // Subcase A1: beta <= beta1 -> alpha = 1.0
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.30, 5, 1.35, 3);
    set_gradt_excess_alpha(&mut m);
    assert_eq!(
        m.aux.mixing.gradt_excess_max_lambda.to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 1.0_f64.to_bits());

    // Subcase A2: beta1 < beta < beta1 + dbeta -> (beta1 + dbeta - beta) / dbeta
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.41, 5, 1.20, 3);
    set_gradt_excess_alpha(&mut m);
    let beta_expected = 0.41 * 1.0e15 / 1.0e15;
    let expected_ramp_above_l1: f64 = (0.35 + 0.1 - beta_expected) / 0.1;
    assert_eq!(
        m.aux.mixing.gradt_excess_alpha.to_bits(),
        expected_ramp_above_l1.to_bits()
    );

    // Subcase A3: beta >= beta1 + dbeta -> alpha = 0.0
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.46, 5, 1.20, 3);
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 0.0_f64.to_bits());

    // Branch B: lambda2 <= lambda < lambda1 (e.g. lambda ~ 0.75 -> beta_limit = 0.30)
    // Subcase B1: beta <= beta_limit -> alpha = 1.0
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.28, 5, 0.75, 2);
    let lam_b1 =
        expected_lrad_div_ledd(&m.t, &m.r, &m.dm_bar, &m.cgrav, &m.m_grav, 2);
    assert!((0.5..1.0).contains(&lam_b1));
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 1.0_f64.to_bits());

    // Subcase B2: beta_limit < beta < beta_limit + dbeta -> ramp to 0 ULP
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.34, 5, 0.75, 2);
    set_gradt_excess_alpha(&mut m);
    let lam_b2 = m.aux.mixing.gradt_excess_max_lambda;
    let beta_b2 = m.aux.mixing.gradt_excess_min_beta;
    let beta_limit = 0.25 + (lam_b2 - 0.5) * (0.35 - 0.25) / (1.0 - 0.5);
    let expected_ramp_mid = (beta_limit + 0.1 - beta_b2) / 0.1;
    assert_eq!(
        m.aux.mixing.gradt_excess_alpha.to_bits(),
        expected_ramp_mid.to_bits()
    );

    // Subcase B3: beta >= beta_limit + dbeta -> alpha = 0.0
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.42, 5, 0.75, 2);
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn test_set_gradt_excess_alpha_lower_lambda_transition_and_floor() {
    let mut base = make_mltpp_model(6, &["h1", "he4"]);
    base.xa[(base.nz - 1) * base.species] = 0.0;

    // Branch C: lambda2 - dlambda < lambda < lambda2 (0.40 < lambda < 0.50)
    // Upstream MESA `turb_info.f90:490-497`:
    //   beta <= beta2 -> alpha = 1.0
    //   beta < beta2 + dbeta -> (lambda - (lambda2 - dlambda)) / dlambda
    //   else -> 0.0
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.20, 5, 0.44, 2);
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 1.0_f64.to_bits());

    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.30, 5, 0.44, 2);
    set_gradt_excess_alpha(&mut m);
    let lam_c2 = m.aux.mixing.gradt_excess_max_lambda;
    let expected_c2 = (lam_c2 - (0.5 - 0.1)) / 0.1;
    assert_eq!(
        m.aux.mixing.gradt_excess_alpha.to_bits(),
        expected_c2.to_bits()
    );

    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.36, 5, 0.44, 2);
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 0.0_f64.to_bits());

    // Branch D: lambda <= lambda2 - dlambda -> 0.0
    let mut m = base.clone();
    configure_beta_lambda(&mut m, 0.10, 5, 0.39, 2);
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn test_set_gradt_excess_alpha_smoothing_clamping_and_snap() {
    let mut m = make_mltpp_model(6, &["h1", "he4"]);
    m.xa[(m.nz - 1) * m.species] = 0.0;
    configure_beta_lambda(&mut m, 0.39, 5, 1.20, 3);
    // Raw alpha = (0.35 + 0.1 - 0.39) / 0.1 = 0.6.
    let raw_alpha = (0.35 + 0.1 - (0.39 * 1.0e15 / 1.0e15)) / 0.1;

    // 1. generations == 2 with age_fraction = 0.8 and no max_change cap.
    m.generations = 2;
    m.controls.mix_wind.gradt_excess_age_fraction = 0.8;
    m.controls.mix_wind.gradt_excess_max_change = -1.0;
    m.aux.mixing.gradt_excess_alpha_old = 0.1;
    set_gradt_excess_alpha(&mut m);
    let expected_unclamped: f64 = (1.0 - 0.8) * raw_alpha + 0.8 * 0.1;
    assert_eq!(
        m.aux.mixing.gradt_excess_alpha.to_bits(),
        expected_unclamped.to_bits()
    );

    // 2. Positive max_change caps both upward and downward steps.
    m.controls.mix_wind.gradt_excess_age_fraction = 0.0;
    m.controls.mix_wind.gradt_excess_max_change = 0.15;
    m.aux.mixing.gradt_excess_alpha_old = 0.2;
    set_gradt_excess_alpha(&mut m);
    let expected_up = (0.2_f64 + 0.15).min(raw_alpha);
    assert_eq!(
        m.aux.mixing.gradt_excess_alpha.to_bits(),
        expected_up.to_bits()
    );

    m.aux.mixing.gradt_excess_alpha_old = 0.9;
    set_gradt_excess_alpha(&mut m);
    let expected_down = (0.9_f64 - 0.15).max(raw_alpha);
    assert_eq!(
        m.aux.mixing.gradt_excess_alpha.to_bits(),
        expected_down.to_bits()
    );

    // 3. Snap thresholds: < 1e-4 -> 0.0, > 0.9999 -> 1.0 (`turb_info.f90:517-518`).
    m.controls.mix_wind.gradt_excess_max_change = -1.0;
    m.controls.mix_wind.gradt_excess_age_fraction = 0.0;
    m.pgas[5] = 0.449_995 * m.peos[5]; // raw alpha = 5e-5 < 1e-4
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 0.0_f64.to_bits());

    m.pgas[5] = 0.350_005 * m.peos[5]; // raw alpha = 0.99995 > 0.9999
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 1.0_f64.to_bits());

    // 4. Negative lambda1 bypasses time smoothing even when generations > 1 (`turb_info.f90:501`).
    m.controls.mix_wind.gradt_excess_lambda1 = -1.0;
    m.aux.mixing.gradt_excess_alpha_old = 0.0;
    m.controls.mix_wind.gradt_excess_age_fraction = 0.95;
    set_gradt_excess_alpha(&mut m);
    assert_eq!(m.aux.mixing.gradt_excess_alpha.to_bits(), 1.0_f64.to_bits());
}

#[test]
fn test_adjust_gradt_excess_and_fraction_zero_ulp_all_gates() {
    let base = make_mltpp_model(4, &["h1", "he4"]);
    let k = 1;

    // Gate 1: gradt_excess_alpha <= 0.0 leaves gradt_ad untouched and effect = 0.0.
    let mut m = base.clone();
    m.aux.mixing.gradt_excess_alpha = 0.0;
    let orig_ad = m.gradt_ad[k];
    adjust_gradt_excess(&mut m, k);
    assert_eq!(
        m.aux.mixing.gradt_excess_effect[k].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(m.gradt_ad[k].val.to_bits(), orig_ad.val.to_bits());
    assert_eq!(
        m.gradt_ad[k].d1[I_LNT_00].to_bits(),
        orig_ad.d1[I_LNT_00].to_bits()
    );

    // Gate 2: gradT - grada_face <= gradT_excess_f1 leaves gradT untouched.
    let mut m = base.clone();
    m.aux.mixing.gradt_excess_alpha = 1.0;
    m.gradt[k] = 0.25 + 1.0e-4;
    m.gradt_ad[k].val = m.gradt[k];
    adjust_gradt_excess(&mut m, k);
    assert_eq!(
        m.aux.mixing.gradt_excess_effect[k].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(m.gradt[k].to_bits(), (0.25_f64 + 1.0e-4).to_bits());

    // Gate 3: lnT/ln10 > gradT_excess_max_logT leaves gradT untouched.
    let mut m = base.clone();
    m.aux.mixing.gradt_excess_alpha = 1.0;
    m.controls.mix_wind.gradt_excess_max_logt = 6.0;
    m.ln_t[k] = 6.1 * LN10;
    adjust_gradt_excess(&mut m, k);
    assert_eq!(
        m.aux.mixing.gradt_excess_effect[k].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(m.gradt[k].to_bits(), 0.45_f64.to_bits());

    // Gate 4: log10(tau) < gradT_excess_max_log_tau_full_off leaves gradT untouched.
    let mut m = base.clone();
    m.aux.mixing.gradt_excess_alpha = 1.0;
    m.controls.mix_wind.gradt_excess_max_log_tau_full_off = 0.5;
    m.controls.mix_wind.gradt_excess_min_log_tau_full_on = 2.5;
    m.tau[k] = 2.0; // log10(2.0) < 0.5
    adjust_gradt_excess(&mut m, k);
    assert_eq!(
        m.aux.mixing.gradt_excess_effect[k].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(m.gradt[k].to_bits(), 0.45_f64.to_bits());

    // Gate 5: partial optical-depth taper (max_log_tau_full_off <= log_tau < min_log_tau_full_on).
    // Deliberately pick non-power-of-two inputs to check left-to-right
    // operation order:
    //   gradT_excess_alpha = (gradT_excess_alpha * (log_tau - off)) / (on - off)
    let mut m = base.clone();
    m.aux.mixing.gradt_excess_alpha = 0.73;
    m.controls.mix_wind.gradt_excess_max_log_tau_full_off = 0.3;
    m.controls.mix_wind.gradt_excess_min_log_tau_full_on = 2.7;
    m.controls.mix_wind.gradt_excess_f2 = 0.15;
    m.tau[k] = 23.0;
    let init_ad = m.gradt_ad[k];
    let grada_ad = m.aux.mixing.grada_face_ad[k];
    adjust_gradt_excess(&mut m, k);

    let log_tau = flexmesa_math::log10(23.0);
    let eff_alpha = (0.73 * (log_tau - 0.3)) / (2.7 - 0.3);
    let alfa = 0.15 + (1.0 - 0.15) * (1.0 - eff_alpha);
    let expected_ad = alfa * init_ad + (1.0 - alfa) * grada_ad;
    assert_eq!(m.gradt_ad[k].val.to_bits(), expected_ad.val.to_bits());
    assert_eq!(
        m.gradt_ad[k].d1[I_LND_00].to_bits(),
        expected_ad.d1[I_LND_00].to_bits()
    );
    assert_eq!(
        m.gradt_ad[k].d1[I_LNT_00].to_bits(),
        expected_ad.d1[I_LNT_00].to_bits()
    );
    assert_eq!(m.gradt[k].to_bits(), expected_ad.val.to_bits());
    assert_eq!(
        m.aux.mixing.gradt_excess_effect[k].to_bits(),
        (1.0 - alfa).to_bits()
    );

    // Gate 6: adjust_gradt_fraction for f == 0.0, f = 0.35, and out-of-range f.
}

#[test]
fn test_adjust_gradt_fraction_blends_and_updates_sub_grada() {
    let base = make_mltpp_model(4, &["h1", "he4"]);
    let k = 1;
    let mut m0 = base.clone();
    m0.aux.mixing.gradt_excess_alpha = 0.0;
    adjust_gradt_fraction(&mut m0, k, 0.0);
    assert_eq!(m0.gradt_ad[k].val.to_bits(), m0.gradr_ad[k].val.to_bits());
    assert_eq!(
        m0.aux.mixing.gradt_sub_grada[k].to_bits(),
        (m0.gradt[k] - m0.aux.mixing.grada_face[k]).to_bits()
    );

    let mut mf = base.clone();
    mf.aux.mixing.gradt_excess_alpha = 1.0;
    mf.controls.mix_wind.gradt_excess_f2 = 0.05;
    adjust_gradt_fraction(&mut mf, k, 0.35);
    let pre_excess_ad =
        0.35 * mf.aux.mixing.grada_face_ad[k] + (1.0 - 0.35) * mf.gradr_ad[k];
    let post_excess_ad =
        0.05 * pre_excess_ad + (1.0 - 0.05) * mf.aux.mixing.grada_face_ad[k];
    assert_eq!(mf.gradt_ad[k].val.to_bits(), post_excess_ad.val.to_bits());
    assert_eq!(
        mf.aux.mixing.gradt_sub_grada[k].to_bits(),
        (post_excess_ad.val - mf.aux.mixing.grada_face[k]).to_bits()
    );
    assert_eq!(
        mf.aux.mixing.gradt_excess_effect[k].to_bits(),
        (1.0_f64 - 0.05).to_bits()
    );

    let mut mout = base;
    mout.aux.mixing.gradt_excess_alpha = 0.0;
    adjust_gradt_fraction(&mut mout, k, -1.0);
    assert_eq!(mout.gradt[k].to_bits(), 0.45_f64.to_bits());
    assert_eq!(
        mout.aux.mixing.gradt_sub_grada[k].to_bits(),
        (0.45_f64 - 0.25).to_bits()
    );
}

#[test]
fn test_mltpp_on_zams_zero_mesa_dir() {
    let mut controls = StarControls {
        initial_mass: 20.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    controls.mix_wind.okay_to_reduce_gradt_excess = true;
    controls.mix_wind.gradt_excess_max_center_h1 = 1.0;
    controls.mix_wind.gradt_excess_lambda1 = 1.0;
    controls.mix_wind.gradt_excess_beta1 = 0.99;
    controls.mix_wind.gradt_excess_lambda2 = 0.0;
    controls.mix_wind.gradt_excess_beta2 = 0.99;
    controls.mix_wind.gradt_excess_dlambda = 0.1;
    controls.mix_wind.gradt_excess_dbeta = 0.1;

    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load 20M ZAMS");
    set_hydro_vars(&mut model, false, true)
        .expect("set_hydro_vars on 20M ZAMS");

    set_gradt_excess_alpha(&mut model);
    assert!(
        model.aux.mixing.gradt_excess_min_beta > 0.0
            && model.aux.mixing.gradt_excess_min_beta < 1.5,
        "unexpected min_beta={}",
        model.aux.mixing.gradt_excess_min_beta
    );
    assert!(
        (0.0..=1.0).contains(&model.aux.mixing.gradt_excess_max_lambda),
        "unexpected max_lambda={}",
        model.aux.mixing.gradt_excess_max_lambda
    );
    assert!(
        (0.0..=1.0).contains(&model.aux.mixing.gradt_excess_alpha),
        "unexpected alpha={}",
        model.aux.mixing.gradt_excess_alpha
    );

    // Pick a superadiabatic face if present and verify adjust_gradt_excess monotonicity.
    model.aux.mixing.gradt_excess_alpha = 1.0;
    model.controls.mix_wind.gradt_excess_f1 = 1.0e-8;
    model.controls.mix_wind.gradt_excess_f2 = 0.1;
    if let Some(k) = (1..model.nz)
        .find(|&k| model.gradt[k] - model.aux.mixing.grada_face[k] > 1.0e-6)
    {
        let old_gt = model.gradt[k];
        let ga = model.aux.mixing.grada_face[k];
        adjust_gradt_excess(&mut model, k);
        assert!(
            model.gradt[k] >= ga && model.gradt[k] < old_gt,
            "expected MLT++ reduction at face {k}: old={old_gt}, new={}, ga={ga}",
            model.gradt[k]
        );
        assert!(
            model.aux.mixing.gradt_excess_effect[k] > 0.0,
            "expected positive gradt_excess_effect at face {k}"
        );
    }
}

#[test]
fn test_predictive_mix_on_zams_mlt_vc_and_trailing_blanks() {
    // Predictive mixing regression (`predictive_mix.f90:806` and trailing
    // spaces):
    // When `okay_to_set_mlt_vc == false`, `eval_mixing_coeffs` must read `model.mlt_vc[k]`
    // rather than `model.mlt_vc_ad[k].val`, and selectors with trailing blanks must match.
    let controls = StarControls {
        initial_mass: 20.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load 20M ZAMS");
    set_hydro_vars(&mut model, false, true)
        .expect("set_hydro_vars on 20M ZAMS");
    let top_cz = model
        .aux
        .mixing
        .conv_bdy_loc
        .first()
        .copied()
        .expect("20M ZAMS core CZ boundary");
    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_by_name("h1").expect("h1").chem_id;
    let he4_id = catalog.get_by_name("he4").expect("he4").chem_id;
    if let (Some(ih1), Some(ihe4)) = (
        model.net_iso.get(h1_id).copied().flatten(),
        model.net_iso.get(he4_id).copied().flatten(),
    ) {
        let sp = model.species;
        let k_edge = top_cz;
        let dh = (model.xa[k_edge * sp + ih1] - 0.05).max(0.0);
        model.xa[k_edge * sp + ih1] -= dh;
        model.xa[k_edge * sp + ihe4] += dh;
        // Raise local luminosity at `k_edge` so it stays radiative with depleted `h1`
        // (`kap` and `P` lowered), yet exceeds `grada` once mixed to core `h1 ~ 0.70`.
        model.lum[k_edge] *= 1.38;
        model.lum[k_edge + 1] *= 2.0;
        set_hydro_vars(&mut model, false, true)
            .expect("refresh hydro_vars after boundary composition step");
    }

    let nz = model.nz;
    for k in 0..nz {
        model.mlt_vc[k] = 9.876_54e8;
        model.mlt_vc_ad[k].val = 1.25e3;
    }
    model.aux.okay_to_set_mlt_vc = false;
    model.controls.mix_wind.predictive_sets[0].predictive_mix = true;
    model.controls.mix_wind.predictive_sets[0].zone_type =
        String::from("burn_H  ");
    model.controls.mix_wind.predictive_sets[0].zone_loc =
        String::from("core  ");
    model.controls.mix_wind.predictive_sets[0].bdy_loc = String::from("top  ");
    model.controls.mix_wind.predictive_sets[0].avoid_reversal =
        String::from("   ");
    model.controls.mix_wind.predictive_sets[0].limit_ingestion =
        String::from("   ");
    model.controls.mix_wind.predictive_sets[0].superad_thresh = 0.0;

    let updated = add_predictive_mixing(&mut model)
        .expect("predictive mixing with trailing-blank controls");
    assert!(
        updated > 0,
        "expected predictive mixing to touch core CZ faces (top_cz={top_cz}, bdys={:?}, gradr[top]={}/{}, grada[top]={}/{})",
        model.aux.mixing.conv_bdy_loc,
        model.gradr[top_cz],
        model.gradr[top_cz + 1],
        model.aux.mixing.grada_face[top_cz],
        model.aux.mixing.grada_face[top_cz + 1],
    );
    assert!(
        model
            .conv_vel
            .iter()
            .any(|&v| v.to_bits() == 9.876_54e8_f64.to_bits()),
        "expected predictive_mix conv_vel to read model.mlt_vc (9.87654e8), not mlt_vc_ad.val"
    );
}

#[test]
fn test_overshoot_zero_and_smoothing_helpers() {
    let mut controls = StarControls {
        initial_mass: 5.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    controls.mix_wind.overshoot_sets[0].scheme = String::from("exponential");
    controls.mix_wind.overshoot_sets[0].zone_type = String::from("burn_H");
    controls.mix_wind.overshoot_sets[0].zone_loc = String::from("core");
    controls.mix_wind.overshoot_sets[0].bdy_loc = String::from("top");
    controls.mix_wind.overshoot_sets[0].f = 0.0;
    controls.mix_wind.overshoot_sets[0].f0 = 0.0;

    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load 5M ZAMS");
    // `overshoot_exp` / `overshoot_step` require `f > 0` and `f0 > 0` (`overshoot.f90`).
    assert!(
        set_hydro_vars(&mut model, false, true).is_err(),
        "exponential overshoot with f=0, f0=0 must fail validation"
    );

    model.controls.mix_wind.overshoot_sets[0].f = 0.014;
    model.controls.mix_wind.overshoot_sets[0].f0 = 0.004;
    set_hydro_vars(&mut model, false, true)
        .expect("set_hydro_vars on 5M ZAMS with positive overshoot");
    add_overshooting(&mut model).expect("overshoot f=0.014 f0=0.004");
    assert!(model.d_mix.iter().all(|&d| d.is_finite() && d >= 0.0));

    // Exact invariant check on threshold_smoothing & weighted_smoothing bounds.
    let mut arr = vec![1.0, 2.0, 10.0, 2.0, 1.0];
    let mut arr_old = vec![0.0; arr.len()];
    weighted_smoothing(&mut arr, 1, true, &mut arr_old);
    assert!(arr[2] < 10.0 && arr[1] > 2.0 && arr[3] > 2.0);

    let mut t_arr = vec![1.0e-2, 1.0e3, 1.0e8, 1.0e3, 1.0e-2];
    let mut t_old = vec![0.0; t_arr.len()];
    threshold_smoothing(&mut t_arr, 1.0e2, 1, true, &mut t_old);
    assert!(t_arr[2].is_finite() && t_arr[2] < 1.0e8);
}
