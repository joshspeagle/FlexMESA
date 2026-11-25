// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` parity and regression tests for `#154`
//! (`create_initial_model`, `pre_ms`, `relax`, `relax_comp`, `relax_mass_comp`,
//! `relax_driver`, and `job_rotation`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::fmt::Write as _;

use flexmesa_const::SECYER;
use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapConfig, KapLowTOption};
use flexmesa_math::exp;
use flexmesa_star::create_initial_model::{
    build_initial_model, build_initial_model_with_configs,
};
use flexmesa_star::evolve::star_evolve_step;
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::model::StarModel;
use flexmesa_star::pre_ms::build_pre_ms_model_with_configs;
use flexmesa_star::relax::{
    apply_uniform_rotation, cell_center_xq, do_relax_angular_momentum,
    parse_angular_momentum_profile_str, parse_entropy_profile_str,
    parse_entropy_profile_with_eos_str, EntropyProfileSource,
    RelaxAngularMomentumConfig, RelaxScalarInterpolant,
};
use flexmesa_star::{MixWindControls, StarControls, StarJobControls};
use flexmesa_turb::MltOption;

fn assert_zone_matches_oracle(
    label: &str,
    zone: usize,
    model: &StarModel,
    expected: [f64; 5],
    log_tol: f64,
) {
    let [exp_lnd, exp_lnt, exp_lnr, exp_lum, exp_dq] = expected;
    let d_lnd = (model.lnd[zone] - exp_lnd).abs();
    let d_lnt = (model.ln_t[zone] - exp_lnt).abs();
    let d_lnr = (model.ln_r[zone] - exp_lnr).abs();
    let d_lum = (model.lum[zone] - exp_lum).abs() / exp_lum.abs().max(1.0);
    let d_dq = (model.dq[zone] - exp_dq).abs() / exp_dq.abs().max(1.0);
    assert!(
        d_lnd <= log_tol
            && d_lnt <= log_tol
            && d_lnr <= log_tol
            && d_lum <= 1.0e-13
            && d_dq <= 1.0e-13,
        "{label} zone {zone} oracle mismatch: \
         d_lnd={d_lnd:.3e} d_lnT={d_lnt:.3e} d_lnR={d_lnr:.3e} \
         rel_lum={d_lum:.3e} rel_dq={d_dq:.3e}"
    );
}

#[test]
fn test_create_initial_model_with_change_net_matches_mesa_oracle() {
    let controls = StarControls {
        initial_z: 1.0e-4,
        ..StarControls::default()
    };

    let mut job = StarJobControls {
        create_initial_model: true,
        change_net: true,
        new_net_name: String::from("pp_and_cno_extras.net"),
        mass_in_gm_for_create_initial_model: 1.9892e31,
        radius_in_cm_for_create_initial_model: 3.5e10,
        initial_model_relax_num_steps: 0,
        ..StarJobControls::default()
    };

    let kap_cfg = KapConfig {
        zbase: 1.0e-4,
        kap_low_t_option: KapLowTOption::LowTFreedman11,
        ..KapConfig::default()
    };

    let model = build_initial_model_with_configs(
        controls.clone(),
        job.clone(),
        EosConfig::default(),
        kap_cfg.clone(),
        None,
    )
    .expect("create_initial_model with pp_and_cno_extras.net");

    assert_eq!(model.job.net_name, "pp_and_cno_extras.net");
    assert_eq!(model.nz, 774);
    assert_eq!(model.species, 25);

    // MESA 26.4.1 surface, mid, and core structure oracle (`create_initial_model`).
    assert_zone_matches_oracle(
        "create_initial_model pp_and_cno_extras",
        0,
        &model,
        [
            -1.502_215_698_358_524_8e1,
            7.939_228_370_935_964e0,
            2.427_862_964_894_775_6e1,
            5.405_347_987_376_163e31,
            3.036_094_708_974_701_5e-10,
        ],
        1.0e-14,
    );
    assert_zone_matches_oracle(
        "create_initial_model pp_and_cno_extras",
        387,
        &model,
        [
            -5.963_245_688_581_785e-1,
            1.199_398_218_312_824e1,
            2.305_940_819_639_525e1,
            1.224_299_202_354_761_4e31,
            6.892_221_727_241_965e-3,
        ],
        1.0e-14,
    );
    assert_zone_matches_oracle(
        "create_initial_model pp_and_cno_extras",
        773,
        &model,
        [
            -3.321_984_323_330_834e-1,
            1.218_337_008_424_682_3e1,
            1.667_771_143_889_374e1,
            7.714_467_377_693_785e22,
            8.095_322_172_780_311e-10,
        ],
        1.0e-14,
    );

    let expected_xa_0: [f64; 25] = [
        7.597_000_000_000_000e-1,
        0.0,
        2.556_211_424_044_105_8e-5,
        2.401_744_378_857_595_7e-1,
        6.213_055_135_916_682e-11,
        0.0,
        0.0,
        1.702_772_177_330_853_5e-5,
        2.066_950_306_203_294e-7,
        0.0,
        5.031_896_909_381_936e-6,
        1.981_679_528_553_610_2e-8,
        0.0,
        0.0,
        4.677_854_838_371_318_7e-5,
        1.853_812_743_945_649_7e-8,
        1.055_517_145_898_504_9e-7,
        0.0,
        0.0,
        2.488_470_214_224_966e-9,
        0.0,
        0.0,
        1.042_512_211_937_402_3e-5,
        0.0,
        2.038_355_854_551_055_4e-5,
    ];
    for (j, &exp_x) in expected_xa_0.iter().enumerate() {
        let diff = (model.xa[j] - exp_x).abs();
        assert!(
            diff <= 1.0e-15,
            "xa[{j}] mismatch: got {}, expected {exp_x}",
            model.xa[j]
        );
    }

    // Setting `job.net_name` with `change_net = false` selects the same net.
    job.change_net = false;
    job.new_net_name.clear();
    job.net_name = String::from("pp_and_cno_extras.net");
    job.mass_in_gm_for_create_initial_model = 1.8986e30;
    job.radius_in_cm_for_create_initial_model = 1.398_22e10;
    let model_direct = build_initial_model_with_configs(
        controls.clone(),
        job.clone(),
        EosConfig::default(),
        kap_cfg.clone(),
        None,
    )
    .expect("create_initial_model with job.net_name");
    assert_eq!(model_direct.job.net_name, "pp_and_cno_extras.net");
    assert_eq!(model_direct.species, 25);
    assert_eq!(model_direct.chem_id, model.chem_id);

    // Non-existent network fails cleanly.
    job.change_net = true;
    job.new_net_name = String::from("nonexistent_154_test.net");
    assert!(build_initial_model_with_configs(
        controls,
        job,
        EosConfig::default(),
        kap_cfg,
        None,
    )
    .is_err());
}

#[test]
fn test_create_initial_model_brown_dwarf_and_prems_15m_match_mesa_oracles() {
    let bd_controls = StarControls {
        initial_z: 1.0e-4,
        ..StarControls::default()
    };
    let bd_job = StarJobControls {
        create_initial_model: true,
        mass_in_gm_for_create_initial_model: 1.9892e31,
        radius_in_cm_for_create_initial_model: 3.5e10,
        initial_model_relax_num_steps: 0,
        ..StarJobControls::default()
    };
    let bd_kap = KapConfig {
        zbase: 1.0e-4,
        kap_low_t_option: KapLowTOption::LowTFreedman11,
        ..KapConfig::default()
    };
    let bd =
        build_initial_model(bd_controls, bd_job, EosConfig::default(), bd_kap)
            .expect("brown dwarf initial model");
    assert_eq!(bd.nz, 774);
    assert_eq!(bd.species, 8);
    assert_zone_matches_oracle(
        "make_brown_dwarf basic.net",
        0,
        &bd,
        [
            -1.502_215_698_358_522_1e1,
            7.939_228_370_935_96e0,
            2.427_862_964_894_774_2e1,
            5.405_347_987_375_947_6e31,
            3.036_094_708_974_701_5e-10,
        ],
        1.0e-14,
    );
    assert_zone_matches_oracle(
        "make_brown_dwarf basic.net",
        387,
        &bd,
        [
            -5.963_245_688_581_578e-1,
            1.199_398_218_312_820_5e1,
            2.305_940_819_639_526e1,
            1.224_299_202_354_768_8e31,
            6.892_221_727_242_298e-3,
        ],
        1.0e-14,
    );
    assert_zone_matches_oracle(
        "make_brown_dwarf basic.net",
        773,
        &bd,
        [
            -3.321_984_323_330_38e-1,
            1.218_337_008_424_678_4e1,
            1.667_771_143_889_374e1,
            7.714_467_377_693_762e22,
            8.095_322_172_780_746e-10,
        ],
        1.0e-14,
    );

    let m15_controls = StarControls {
        initial_mass: 15.0,
        initial_z: 0.02,
        initial_y: -1.0,
        zbase: 0.02,
        mlt_option: MltOption::Cox,
        mixing_length_alpha: 1.6,
        mix_wind: MixWindControls {
            max_dq: 4.0e-3,
            ..MixWindControls::default()
        },
        ..StarControls::default()
    };
    let m15_job = StarJobControls {
        create_pre_ms_model: true,
        pre_ms_relax_num_steps: 0,
        ..StarJobControls::default()
    };
    let m15_kap = KapConfig {
        zbase: 0.02,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let m15 = build_pre_ms_model_with_configs(
        m15_controls,
        m15_job,
        EosConfig::default(),
        m15_kap,
        None,
    )
    .expect("15M pre-MS model");
    assert_eq!(m15.nz, 1033);
    assert_zone_matches_oracle(
        "15M_dynamo pre_ms",
        0,
        &m15,
        [
            -1.718_824_429_865_272_7e1,
            8.550_014_188_513_828e0,
            3.074_563_644_374_423_2e1,
            3.335_672_940_956_81e38,
            5.704_188_204_408_657e-6,
        ],
        1.0e-12,
    );
    assert_zone_matches_oracle(
        "15M_dynamo pre_ms",
        516,
        &m15,
        [
            -1.549_799_169_346_027e1,
            1.088_726_828_045_913_6e1,
            3.059_433_113_892_054_8e1,
            3.167_745_738_574_408_8e38,
            3.856_189_376_803_351_3e-4,
        ],
        1.0e-12,
    );
    assert_zone_matches_oracle(
        "15M_dynamo pre_ms",
        1032,
        &m15,
        [
            -1.248_531_967_587_368e1,
            1.261_127_651_838_039_6e1,
            2.630_689_192_729_168_7e1,
            3.335_800_556_088_442e33,
            1.000_038_257_687_096_5e-5,
        ],
        1.0e-12,
    );
}

#[test]
fn test_scalar_profile_parsers_and_eos_modes_parity() {
    let j_text = "\
        3\n\
        0.0000000000000000D+00  1.0394233174861947D+19\n\
        5.0000000000000000d-01  5.1971165874309734d+18\n\
        9.9999992092967227D-01  5.6182190750389082D+12\n";
    let j_prof = parse_angular_momentum_profile_str(j_text).expect("j parse");
    assert_eq!(j_prof.num_pts, 3);
    assert_eq!(j_prof.xq[2], 9.999_999_209_296_723e-1);
    assert_eq!(j_prof.values[0], 1.039_423_317_486_194_7e19);

    let s_text = "\
        2\n\
        4.0001335577244390D-13  2.3204991006457005D+09\n\
        9.9999996046483952D-01  5.3548470293320835D+08\n";
    let s_prof = parse_entropy_profile_str(s_text).expect("s parse");
    assert_eq!(s_prof.num_pts, 2);
    assert_eq!(s_prof.values[1], 5.354_847_029_332_083_5e8);

    assert!(parse_angular_momentum_profile_str("1\n0.0 1.0\n").is_err());
    assert!(parse_entropy_profile_str("3\n0.0 1.0\n0.5 2.0\n").is_err());
    assert_eq!(
        EntropyProfileSource::from_mesa_str("eosDT").unwrap(),
        EntropyProfileSource::EosDt
    );
    assert_eq!(
        EntropyProfileSource::from_mesa_str("eosPT").unwrap(),
        EntropyProfileSource::EosPt
    );
    assert_eq!(
        EntropyProfileSource::from_mesa_str("eosDE").unwrap(),
        EntropyProfileSource::EosDe
    );
    assert!(EntropyProfileSource::from_mesa_str("bogus").is_err());

    // Build a stepped 1 Msun pre-MS model and verify all EOS entropy modes agree.
    let model = build_stepped_prems_1m_model();
    let eos_cfg = EosConfig::default();
    let sample_zones = [5_usize, model.nz / 2, model.nz - 5];
    let mut direct_rows = String::from("3\n");
    let mut dt_rows = String::from("3\n");
    let mut pt_rows = String::from("3\n");
    let mut de_rows = String::from("3\n");
    let mut expected_s = Vec::with_capacity(3);
    let xq_centers = cell_center_xq(&model).expect("cell_center_xq");

    for &k in &sample_zones {
        let rho = exp(model.lnd[k]);
        let t = exp(model.ln_t[k]);
        let s_k = exp(model.ln_s[k]);
        let pgas_k = model.pgas[k];
        let e_k = exp(model.ln_e[k]);
        expected_s.push(s_k);
        let xq_k = xq_centers[k];
        let _ = writeln!(direct_rows, "{xq_k:.17e} {s_k:.17e}");
        let _ = writeln!(dt_rows, "{xq_k:.17e} {rho:.17e} {t:.17e}");
        let _ = writeln!(pt_rows, "{xq_k:.17e} {pgas_k:.17e} {t:.17e}");
        let _ = writeln!(de_rows, "{xq_k:.17e} {rho:.17e} {e_k:.17e}");
    }

    let prof_direct = parse_entropy_profile_with_eos_str(
        &direct_rows,
        EntropyProfileSource::Direct,
        &model,
        &eos_cfg,
    )
    .expect("Direct");
    let prof_dt = parse_entropy_profile_with_eos_str(
        &dt_rows,
        EntropyProfileSource::EosDt,
        &model,
        &eos_cfg,
    )
    .expect("EosDt");
    let prof_pt = parse_entropy_profile_with_eos_str(
        &pt_rows,
        EntropyProfileSource::EosPt,
        &model,
        &eos_cfg,
    )
    .expect("EosPt");
    let prof_de = parse_entropy_profile_with_eos_str(
        &de_rows,
        EntropyProfileSource::EosDe,
        &model,
        &eos_cfg,
    )
    .expect("EosDe");

    for i in 0..3 {
        let s_ref = expected_s[i];
        assert!(
            (prof_direct.values[i] - s_ref).abs() / s_ref < 1.0e-15,
            "Direct mismatch at {i}"
        );
        assert!(
            (prof_dt.values[i] - s_ref).abs() / s_ref < 1.0e-12,
            "EosDt mismatch at {i}"
        );
        assert!(
            (prof_pt.values[i] - s_ref).abs() / s_ref < 1.0e-9,
            "EosPt mismatch at {i}: got {}, expected {s_ref}",
            prof_pt.values[i]
        );
        assert!(
            (prof_de.values[i] - s_ref).abs() / s_ref < 1.0e-7,
            "EosDe mismatch at {i}: got {}, expected {s_ref}",
            prof_de.values[i]
        );
    }

    let interp = RelaxScalarInterpolant::from_profile(&prof_dt)
        .expect("scalar interpolant");
    assert_eq!(interp.num_pts(), 3);
    let (xq_star, vals) = interp.interpolate_to_model(&model).expect("interp");
    assert_eq!(xq_star, xq_centers);
    for &k in &sample_zones {
        let idx = sample_zones.iter().position(|&z| z == k).unwrap();
        assert!(
            (vals[k] - prof_dt.values[idx]).abs() / prof_dt.values[idx]
                < 1.0e-13
        );
    }
    assert!(RelaxScalarInterpolant::from_slices(
        &[0.1, 0.5, 0.2],
        &[1.0, 2.0, 3.0],
    )
    .is_err());
}

fn build_stepped_prems_1m_model() -> StarModel {
    let controls = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls {
        create_pre_ms_model: true,
        pre_ms_relax_num_steps: 0,
        ..StarJobControls::default()
    };
    let kap_cfg = KapConfig {
        zbase: 0.02,
        ..KapConfig::default()
    };
    let mut model = build_pre_ms_model_with_configs(
        controls,
        job,
        EosConfig::default(),
        kap_cfg,
        None,
    )
    .expect("1M pre-ms model");
    model.dt_next = 1.0e-4 * SECYER;
    star_evolve_step(&mut model, true, false, true).expect("step 1");
    star_evolve_step(&mut model, true, false, true).expect("step 2");
    model
}

#[test]
fn test_relax_angular_momentum_constant_omega_center_and_restore() {
    let mut model = build_stepped_prems_1m_model();
    apply_uniform_rotation(&mut model, 1.0e-9);
    model.controls.rotation_flag = true;
    model.controls.mix_wind.premix_omega = true;
    model.controls.mix_wind.am_d_mix_factor = 0.0333;
    model.controls.dxdt_nuc_factor = 1.0;
    model.controls.mix_factor = 1.0;
    model.controls.do_element_diffusion = true;
    model.controls.ts_stop.max_timestep = 1.0e9;
    model.controls.max_model_number = Some(555);
    model.controls.use_other_torque = false;
    model.doing_first_model_of_run = false;
    model.star_age = 1.0e8;
    model.star_age_old = 1.0e8;
    model.time_sec = 1.0e8 * SECYER;
    model.time_old_sec = 1.0e8 * SECYER;
    model.aux.step_hist.time_sec = 1.0e8 * SECYER;
    model.aux.step_hist.time_old_sec = 1.0e8 * SECYER;
    model.dt_next = 1.0e-2 * SECYER;

    let xq_full = cell_center_xq(&model).expect("xq_full");
    let keep_pairs: Vec<(f64, f64)> = xq_full
        .iter()
        .zip(&model.j_rot)
        .filter(|(&xq, _)| xq <= 0.85)
        .map(|(&xq, &jr)| (xq, 1.5 * jr))
        .collect();
    let xq_grid: Vec<f64> = keep_pairs.iter().map(|&(xq, _)| xq).collect();
    let j_target: Vec<f64> = keep_pairs.iter().map(|&(_, jr)| jr).collect();
    let interp = RelaxScalarInterpolant::from_slices(&xq_grid, &j_target)
        .expect("j interpolant");

    let cfg = RelaxAngularMomentumConfig {
        timescale_for_relax_angular_momentum: 1.0e-8,
        max_dt_for_relax_angular_momentum: 1.0e-7,
        num_timescales_for_relax_angular_momentum: 50.0,
        constant_omega_center: true,
    };
    let mut extras = DefaultStarExtras;
    do_relax_angular_momentum(
        &mut model,
        200,
        &j_target,
        &xq_grid,
        cfg,
        &mut extras,
    )
    .expect("do_relax_angular_momentum");

    let (xq_star, vals) = interp.interpolate_to_model(&model).unwrap();
    let x_last = *xq_grid.last().unwrap();
    let outer_errs: Vec<f64> = xq_star
        .iter()
        .zip(model.j_rot.iter().zip(&vals))
        .filter(|(&xq, _)| xq <= x_last)
        .map(|(_, (&jr, &vt))| ((jr - vt) / vt).abs())
        .collect();
    let mean_outer_err =
        outer_errs.iter().sum::<f64>() / (outer_errs.len() as f64);
    assert!(
        mean_outer_err < 1.0e-2,
        "outer j_rot did not relax to target: mean_outer_err={mean_outer_err:.6e}"
    );

    // Core `constant_omega_center` parity (`relax.f90:1208-1230`):
    // innermost cell matches `omega_target = vals[nz - 2] / i_rot[nz - 2]`.
    let nz = model.nz;
    assert!(nz >= 2 && xq_star[nz - 1] > x_last);
    let omega_inner_ref = vals[nz - 2] / model.aux.i_rot[nz - 2];
    let omega_center = model.j_rot[nz - 1] / model.aux.i_rot[nz - 1];
    let core_omega_rel =
        ((omega_center - omega_inner_ref) / omega_inner_ref).abs();
    assert!(
        core_omega_rel < 5.0e-2,
        "constant_omega_center mismatch: omega_center={omega_center:.6e}, \
         ref={omega_inner_ref:.6e}, rel={core_omega_rel:.6e}"
    );

    assert!(model.aux.need_to_setvars);
    assert!(!model.aux.st_start.have_st_start_info);
    assert_eq!(model.star_age, 1.0e8);
    assert_eq!(model.star_age_old, 1.0e8);
    assert_eq!(model.time_sec, 1.0e8 * SECYER);
    assert_eq!(model.time_old_sec, 1.0e8 * SECYER);
    assert_eq!(model.dt_next, 1.0e-2 * SECYER);
    assert_eq!(model.controls.max_model_number, Some(555));
    assert_eq!(model.controls.mix_wind.am_d_mix_factor, 0.0333);
    assert_eq!(model.controls.dxdt_nuc_factor, 1.0);
    assert_eq!(model.controls.mix_factor, 1.0);
    assert!(model.controls.do_element_diffusion);
    assert_eq!(model.controls.ts_stop.max_timestep, 1.0e9);
    assert!(!model.controls.use_other_torque);
    assert!(model.aux.extra_jdot.iter().all(|&v| v == 0.0));
    assert!(model.aux.extra_omegadot.iter().all(|&v| v == 0.0));
}
