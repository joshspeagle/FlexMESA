// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` regression and parity tests for binary mass-transfer,
//! eccentricity-enhancement, and coupled `part1` -> `adjust_mdots` -> `part2`
//! wind/RLOF accounting (`#160`, `#530`).
//!
//! Zero external MESA installation required: all checks verify self-contained
//! closed-form oracles and coupled `BinarySystem` invariants.

#![forbid(unsafe_code)]

use flexmesa_binary::{
    blend_explicit_mdot, eval_arras_mdot_thin, eval_coupled_edot_breakdown,
    eval_edot_enhancement_isotropic, eval_kolb_mdot_eccentric,
    eval_kolb_mdot_eccentric_photosphere, eval_kolb_mdot_thick,
    eval_kolb_mdot_thick_photosphere, eval_ritter_mdot_eccentric,
    eval_ritter_mdot_thin, eval_rlobe, eval_scheme_explicit_mdot,
    BinaryCompanion, BinaryExtras, BinaryExtrasAction, BinaryInlistControls,
    BinarySystem,
};
use flexmesa_const::{MSUN, RSUN, SECYER};
use flexmesa_star::{
    eval_photosphere_info, load_zams_model, set_hydro_vars, StarControls,
    StarError, StarJobControls, StarModel,
};

#[derive(Default)]
struct StepVaryingWindExtras {
    wind_calls_per_star: [usize; 2],
    base_wind_msun_yr: f64,
    step2_wind_msun_yr: Option<f64>,
    checked_mstar_dot_old: [f64; 2],
}

impl BinaryExtras for StepVaryingWindExtras {
    fn other_wind_cgs(
        &mut self,
        star: &StarModel,
        star_idx: usize,
        ignore_rlof_flag: bool,
    ) -> Result<Option<f64>, StarError> {
        let slot = star_idx.saturating_sub(1).min(1);
        self.wind_calls_per_star[slot] += 1;
        if ignore_rlof_flag {
            return Ok(Some(0.0));
        }
        let step_num =
            f64::from(u32::try_from(star.model_number).expect("model u32"));
        let w_msun_yr = match self.step2_wind_msun_yr {
            Some(w2) if star.model_number >= 2 => w2,
            Some(_) => self.base_wind_msun_yr,
            None => self.base_wind_msun_yr * (step_num + 1.0),
        };
        Ok(Some(-w_msun_yr * (MSUN / SECYER)))
    }

    fn check_model(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        self.checked_mstar_dot_old[0] = bin.donor.aux.step_hist.mstar_dot_old;
        self.checked_mstar_dot_old[1] = match &bin.companion {
            BinaryCompanion::Star(acc) => acc.aux.step_hist.mstar_dot_old,
            BinaryCompanion::PointMass { .. } => 0.0,
        };
        BinaryExtrasAction::KeepGoing
    }
}

fn load_subphotospheric_zams(mass_msun: f64) -> (StarModel, f64, f64) {
    let mut donor = load_zams_model(
        StarControls {
            initial_mass: mass_msun,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("ZAMS donor");
    set_hydro_vars(&mut donor, false, false).expect("set_hydro_vars");
    // Lower `tau_factor` puts the photosphere inside zone 1 (`r_phot < r[0]`).
    donor.controls.tau_factor = 1.0e-3;
    let r_phot = RSUN * eval_photosphere_info(&donor).r_rsun;
    assert!(
        r_phot > 0.0 && r_phot < donor.r[0],
        "expected r_phot ({r_phot:.6e}) < donor.r[0] ({:.6e})",
        donor.r[0]
    );
    let rl_mid = f64::midpoint(r_phot, donor.r[0]);
    (donor, r_phot, rl_mid)
}

#[test]
fn kolb_circular_photosphere_gate_zeroes_thick_branch_below_rlobe() {
    let (donor, r_phot, rl_mid) = load_subphotospheric_zams(5.0);
    let m_d = donor.mstar;
    let m_acc = 1.4 * MSUN;
    let separation = 10.0 * RSUN;

    // Circular (`binary_mdot.f90:775`): outer zone boundary exceeds `rl_mid`,
    // yet photosphere underfills it so thick Kolb overflow must be zero.
    let raw_thick = eval_kolb_mdot_thick(
        m_d,
        m_acc,
        rl_mid,
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
    );
    assert!(
        raw_thick < 0.0,
        "expected nonzero thick branch when r[0] > rl"
    );
    let gated_thick = eval_kolb_mdot_thick_photosphere(
        m_d,
        m_acc,
        r_phot,
        rl_mid,
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
    );
    assert_eq!(gated_thick.to_bits(), 0.0_f64.to_bits());

    let kolb_explicit = eval_scheme_explicit_mdot(
        "Kolb", &donor, m_acc, rl_mid, separation, 0.0,
    )
    .expect("explicit Kolb evaluation");
    let expected_thin = eval_ritter_mdot_thin(
        m_d,
        m_acc,
        r_phot,
        rl_mid,
        donor.rho[0],
        donor.peos[0],
        donor.t[0],
        donor.mu[0],
        true,
    );
    assert!(expected_thin < 0.0);
    let rel_err =
        (kolb_explicit - expected_thin).abs() / expected_thin.abs().max(1.0);
    assert!(
        rel_err <= 1.0e-14,
        "Kolb underfilling photosphere must match Ritter thin only: \
         got {kolb_explicit:.15e}, expected {expected_thin:.15e}"
    );

    let r_over = 1.01 * rl_mid;
    let overflow_thick = eval_kolb_mdot_thick_photosphere(
        m_d,
        m_acc,
        r_over,
        rl_mid,
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
    );
    assert_eq!(overflow_thick.to_bits(), raw_thick.to_bits());
}

#[test]
fn kolb_eccentric_photosphere_gate_applies_to_mdot_and_edot() {
    let (donor, r_phot, rl_mid) = load_subphotospheric_zams(5.0);
    let m_d = donor.mstar;
    let m_acc = 1.4 * MSUN;
    let ecc = 0.25;
    let rl_ecc = rl_mid * (1.0 + ecc) / (1.0 - ecc * ecc);

    // Eccentric (`binary_mdot.f90:913`, `binary_edot.f90:216`): periastron RL
    // lies strictly between `r_phot` and `donor.r[0]`.
    let raw_ecc_thick = eval_kolb_mdot_eccentric(
        m_d,
        m_acc,
        rl_ecc,
        ecc,
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
        200,
    );
    assert!(raw_ecc_thick < 0.0);
    let gated_ecc_thick = eval_kolb_mdot_eccentric_photosphere(
        m_d,
        m_acc,
        r_phot,
        rl_ecc,
        ecc,
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
        200,
    );
    assert_eq!(gated_ecc_thick.to_bits(), 0.0_f64.to_bits());

    let edot_kolb = eval_edot_enhancement_isotropic(
        m_d,
        m_acc,
        r_phot,
        rl_ecc,
        ecc,
        donor.rho[0],
        donor.peos[0],
        donor.t[0],
        donor.mu[0],
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
        true,
        200,
    );
    let edot_thin = eval_edot_enhancement_isotropic(
        m_d,
        m_acc,
        r_phot,
        rl_ecc,
        ecc,
        donor.rho[0],
        donor.peos[0],
        donor.t[0],
        donor.mu[0],
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
        false,
        200,
    );
    assert!(edot_thin > 0.0);
    assert_eq!(edot_kolb.to_bits(), edot_thin.to_bits());

    let r_over = 1.01 * rl_mid;
    let overflow_ecc_thick = eval_kolb_mdot_eccentric_photosphere(
        m_d,
        m_acc,
        r_over,
        rl_ecc,
        ecc,
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
        200,
    );
    assert_eq!(overflow_ecc_thick.to_bits(), raw_ecc_thick.to_bits());
}

fn load_unclamped_surface_zams() -> StarModel {
    let mut donor = load_zams_model(
        StarControls {
            initial_mass: 2.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("2 Msun ZAMS donor");
    set_hydro_vars(&mut donor, false, false).expect("set_hydro_vars");
    // Values strictly below legacy floors (`Peos>=1`, `T>=10`, `mu>=0.5`).
    donor.rho[0] = 5.0e-12;
    donor.peos[0] = 0.25;
    donor.t[0] = 4.0;
    donor.mu[0] = 0.25;
    donor
}

#[test]
fn explicit_mdot_preserves_raw_surface_values_below_legacy_floors() {
    let donor = load_unclamped_surface_zams();
    let m_d = donor.mstar;
    let m_acc = 1.4 * MSUN;
    let r_phot = RSUN * eval_photosphere_info(&donor).r_rsun;
    let ecc = 0.2;
    let rl_d = (donor.r[0] + 5.0e3) / (1.0 - ecc);
    let separation = 12.0 * RSUN;

    let ritter_actual = eval_scheme_explicit_mdot(
        "Ritter", &donor, m_acc, rl_d, separation, ecc,
    )
    .expect("explicit Ritter");
    let ritter_expected = eval_ritter_mdot_eccentric(
        m_d,
        m_acc,
        r_phot,
        rl_d,
        ecc,
        donor.rho[0],
        donor.peos[0],
        donor.t[0],
        donor.mu[0],
        false,
        500,
    );
    assert!(ritter_expected < 0.0 && ritter_expected.is_finite());
    assert!(
        (ritter_actual - ritter_expected).abs()
            <= 1.0e-13 * ritter_expected.abs()
    );

    let kolb_actual =
        eval_scheme_explicit_mdot("Kolb", &donor, m_acc, rl_d, separation, ecc)
            .expect("explicit Kolb");
    let kolb_expected = eval_ritter_mdot_eccentric(
        m_d,
        m_acc,
        r_phot,
        rl_d,
        ecc,
        donor.rho[0],
        donor.peos[0],
        donor.t[0],
        donor.mu[0],
        true,
        500,
    );
    assert!(kolb_expected < 0.0 && kolb_expected.is_finite());
    assert!(
        (kolb_actual - kolb_expected).abs() <= 1.0e-13 * kolb_expected.abs()
    );

    let sep_arras = (r_phot + 1.0) / eval_rlobe(m_d, m_acc, 1.0);
    let arras_actual =
        eval_scheme_explicit_mdot("Arras", &donor, m_acc, rl_d, sep_arras, ecc)
            .expect("explicit Arras");
    let arras_expected = eval_arras_mdot_thin(
        m_d,
        m_acc,
        sep_arras,
        r_phot,
        donor.rho[0],
        donor.t[0],
        donor.mu[0],
    );
    assert!(arras_expected < 0.0 && arras_expected.is_finite());
    assert!(
        (arras_actual - arras_expected).abs() <= 1.0e-13 * arras_expected.abs()
    );
}

#[test]
fn coupled_edot_preserves_raw_surface_values_below_legacy_floors() {
    let donor = load_unclamped_surface_zams();
    let m_d = donor.mstar;
    let m_acc = 1.4 * MSUN;
    let r_phot = RSUN * eval_photosphere_info(&donor).r_rsun;
    let ecc = 0.2;
    let separation =
        (donor.r[0] + 5.0e3) / ((1.0 - ecc) * eval_rlobe(m_d, m_acc, 1.0));

    let bctrl = BinaryInlistControls {
        do_tidal_circ: false,
        use_eccentricity_enhancement: true,
        mdot_scheme: String::from("Ritter"),
        anomaly_steps: 200,
        max_abs_edot_enhance: 1.0e30,
        ..BinaryInlistControls::default()
    };
    let eb = eval_coupled_edot_breakdown(
        &bctrl, &donor, None, 1, m_d, m_acc, separation, 1.0e6, ecc,
    )
    .expect("coupled edot");
    let rl_edot = eval_rlobe(m_d, m_acc, separation);
    let expected_enh = eval_edot_enhancement_isotropic(
        m_d,
        m_acc,
        r_phot,
        rl_edot,
        ecc,
        donor.rho[0],
        donor.peos[0],
        donor.t[0],
        donor.mu[0],
        &donor.r,
        &donor.peos,
        &donor.t,
        &donor.mu,
        &donor.gamma1,
        false,
        200,
    );
    assert!(expected_enh > 0.0 && expected_enh.is_finite());
    assert!(
        (eb.edot_enhance - expected_enh).abs() <= 1.0e-13 * expected_enh.abs(),
        "edot_enhance mismatch: actual={:.15e}, expected={expected_enh:.15e}",
        eb.edot_enhance
    );
}

#[test]
fn blend_explicit_mdot_zeroes_positive_rate_before_cur_mdot_frac_blend() {
    let prev = -1.0e-8 * (MSUN / SECYER);
    let blended = blend_explicit_mdot(prev, 2.5e20, 0.75, 1.0);
    let expected = 0.75 * prev;
    assert!(
        (blended - expected).abs() <= 1.0e-15 * expected.abs(),
        "positive explicit rate must zero before cur_mdot_frac blend: \
         actual={blended:.15e}, expected={expected:.15e}"
    );
    assert_eq!(blend_explicit_mdot(0.0, 2.5e20, 0.5, 1.0), 0.0);

    let cap_msun_yr = 1.0e-3;
    let capped =
        blend_explicit_mdot(prev, -10.0 * (MSUN / SECYER), 0.0, cap_msun_yr);
    let expected_cap = -cap_msun_yr * (MSUN / SECYER);
    assert!((capped - expected_cap).abs() <= 1.0e-15 * expected_cap.abs());
}

#[test]
fn coupled_binary_steps_preserve_star_mdot_controls_and_track_mstar_dot_old() {
    let star_ctrl = StarControls {
        initial_mass: 2.0,
        zbase: 0.02,
        use_other_wind: true,
        ..StarControls::default()
    };
    let s1 = load_zams_model(star_ctrl.clone(), StarJobControls::default())
        .expect("2 Msun ZAMS donor");
    let s2 = load_zams_model(
        StarControls {
            initial_mass: 1.5,
            ..star_ctrl
        },
        StarJobControls::default(),
    )
    .expect("1.5 Msun ZAMS accretor");

    let mut bctrl = BinaryInlistControls {
        evolve_both_stars: true,
        m1_msun: 2.0,
        m2_msun: 1.5,
        initial_period_in_days: 0.5045,
        mdot_scheme: String::from("Ritter"),
        max_tries_to_achieve: 20,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    let mut bin = BinarySystem::from_controls_two_stars(s1, s2, bctrl)
        .expect("two-star RLOF binary");
    let mut extras = StepVaryingWindExtras {
        wind_calls_per_star: [0, 0],
        base_wind_msun_yr: 1.0e-9,
        step2_wind_msun_yr: None,
        checked_mstar_dot_old: [0.0, 0.0],
    };

    let mut dt = 1.0e3 * SECYER;
    let mut checked_donor_old = Vec::new();
    for step in 1..=3 {
        let prev_donor_mdot = bin.donor.mstar_dot;
        let prev_accretor_mdot = match &bin.companion {
            BinaryCompanion::Star(acc) => acc.mstar_dot,
            BinaryCompanion::PointMass { .. } => unreachable!(),
        };
        let (next_dt, _) = bin
            .try_step_with_extras(dt, &mut extras)
            .unwrap_or_else(|e| panic!("two-star step {step} failed: {e}"));
        dt = next_dt;

        assert_eq!(bin.donor.controls.mass_change, 0.0);
        assert!(!bin.donor.controls.use_other_adjust_mdot);
        assert_eq!(
            extras.checked_mstar_dot_old[0].to_bits(),
            prev_donor_mdot.to_bits(),
            "step {step}: donor mstar_dot_old at check_model must match prior mstar_dot"
        );
        assert_eq!(
            bin.donor.aux.step_hist.mstar_dot_old.to_bits(),
            bin.donor.mstar_dot.to_bits(),
            "step {step}: accepted donor mstar_dot_old must advance to current mstar_dot"
        );
        assert_eq!(
            bin.last_mdot.component_mdot[0].to_bits(),
            bin.donor.mstar_dot.to_bits()
        );
        checked_donor_old.push(extras.checked_mstar_dot_old[0]);

        let BinaryCompanion::Star(ref acc) = bin.companion else {
            unreachable!();
        };
        assert_eq!(acc.controls.mass_change, 0.0);
        assert!(!acc.controls.use_other_adjust_mdot);
        assert_eq!(
            extras.checked_mstar_dot_old[1].to_bits(),
            prev_accretor_mdot.to_bits(),
            "step {step}: accretor mstar_dot_old at check_model must match prior mstar_dot"
        );
        assert_eq!(
            acc.aux.step_hist.mstar_dot_old.to_bits(),
            acc.mstar_dot.to_bits(),
            "step {step}: accepted accretor mstar_dot_old must advance to current mstar_dot"
        );
        assert_eq!(
            bin.last_mdot.component_mdot[1].to_bits(),
            acc.mstar_dot.to_bits()
        );
    }

    assert!(extras.wind_calls_per_star[0] >= 3);
    assert!(extras.wind_calls_per_star[1] >= 3);
    assert_ne!(
        checked_donor_old[1].to_bits(),
        checked_donor_old[2].to_bits(),
        "donor mstar_dot_old must update between steps rather than freeze"
    );
}

#[test]
fn post_part2_mdot_redo_syncs_component_mdot_and_mdot_system_wind() {
    let mut s1_ctrl = StarControls {
        initial_mass: 2.0,
        zbase: 0.02,
        use_other_wind: true,
        ..StarControls::default()
    };
    s1_ctrl.mix_wind.max_mdot_redo_cnt = 5;
    s1_ctrl.mix_wind.mdot_revise_factor = 1.5;
    let donor = load_zams_model(s1_ctrl, StarJobControls::default())
        .expect("2 Msun ZAMS donor");

    let bctrl = BinaryInlistControls {
        evolve_both_stars: false,
        m1_msun: 2.0,
        m2_msun: 1.4,
        initial_period_in_days: 5.0,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    let mut bin = BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("detached point-mass binary");
    let mut extras = StepVaryingWindExtras {
        wind_calls_per_star: [0, 0],
        base_wind_msun_yr: 1.0e-8,
        step2_wind_msun_yr: Some(5.0e-8),
        checked_mstar_dot_old: [0.0, 0.0],
    };

    let (next_dt, _) = bin
        .try_step_with_extras(100.0 * SECYER, &mut extras)
        .expect("step 1");
    let expected_step1 = -1.0e-8 * (MSUN / SECYER);
    assert!(
        (bin.donor.mstar_dot - expected_step1).abs()
            <= 1.0e-12 * expected_step1.abs()
    );

    // Arm `was_in_implicit_wind_limit` so step 2's `MdotRedoState::setup`
    // revises the 5x jump back to `mstar_dot_old` inside `part2`.
    bin.donor.aux.step_hist.was_in_implicit_wind_limit = true;
    let _ = bin
        .try_step_with_extras(next_dt.min(100.0 * SECYER), &mut extras)
        .expect("step 2");

    assert!(
        (bin.donor.mstar_dot - expected_step1).abs()
            <= 1.0e-12 * expected_step1.abs(),
        "expected part2 MdotRedoState clamp back to {expected_step1:.6e}, \
         got {:.6e}",
        bin.donor.mstar_dot
    );
    assert_eq!(
        bin.last_mdot.component_mdot[0].to_bits(),
        bin.donor.mstar_dot.to_bits(),
        "post-part2 component_mdot[0] must track upward mstar_dot revisions"
    );
    assert_eq!(
        bin.last_mdot.mdot_system_wind[0].to_bits(),
        bin.donor.mstar_dot.to_bits(),
        "post-part2 mdot_system_wind[0] must reflect revised donor wind"
    );
}
