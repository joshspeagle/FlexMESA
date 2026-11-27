// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` initial-model oracle and `#154` relax/rotation regression
//! tests (`create_initial_model`, `pre_ms`, `relax`, `relax_comp`,
//! `relax_mass_comp`, `relax_driver`, and `job_rotation`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

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

struct LoadedMesaSample {
    nz: usize,
    species: usize,
    net_name: String,
    zones: Vec<(usize, [u64; 5])>,
    xa: Vec<(usize, Vec<u64>)>,
}

impl LoadedMesaSample {
    fn zone_bits(&self, k_1based: usize) -> [u64; 5] {
        self.zones.iter().find(|(k, _)| *k == k_1based).map_or_else(
            || panic!("missing ZONE LOADED {k_1based}"),
            |(_, bits)| *bits,
        )
    }

    fn xa_bits(&self, k_1based: usize) -> &[u64] {
        self.xa.iter().find(|(k, _)| *k == k_1based).map_or_else(
            || panic!("missing XA LOADED {k_1based}"),
            |(_, bits)| bits.as_slice(),
        )
    }
}

fn parse_hex_u64(tok: &str) -> u64 {
    u64::from_str_radix(tok, 16)
        .unwrap_or_else(|e| panic!("invalid IEEE-754 hex token {tok:?}: {e}"))
}

fn load_relax154_loaded_sample(filename: &str) -> LoadedMesaSample {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/relax_154_oracles/samples")
        .join(filename);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read MESA oracle {}: {e}", path.display())
    });
    assert!(text.contains("# mesa_version: r26.4.1"));
    assert!(text.contains("# compiler: gfortran 15.2.0"));
    assert!(text.contains("# mesa_sdk_version: x86_64-linux-26.6.1"));

    let mut nz = None;
    let mut species = None;
    let mut net_name = None;
    let mut zones = Vec::new();
    let mut xa = Vec::new();

    for line in text.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.as_slice() {
            ["INT", "LOADED", "nz", v] => {
                nz = Some(v.parse::<usize>().expect("parse nz"));
            }
            ["INT", "LOADED", "species", v] => {
                species = Some(v.parse::<usize>().expect("parse species"));
            }
            ["STR", "LOADED", "net_name", v] => {
                net_name = Some((*v).to_string());
            }
            ["ZONE", "LOADED", k_str, ..] if toks.len() >= 13 => {
                let k1 = k_str.parse::<usize>().expect("parse zone k");
                let bits = [
                    parse_hex_u64(toks[4]),
                    parse_hex_u64(toks[6]),
                    parse_hex_u64(toks[8]),
                    parse_hex_u64(toks[10]),
                    parse_hex_u64(toks[12]),
                ];
                zones.push((k1, bits));
            }
            ["XA", "LOADED", k_str, rest @ ..] => {
                let k1 = k_str.parse::<usize>().expect("parse xa zone k");
                xa.push((k1, rest.iter().map(|t| parse_hex_u64(t)).collect()));
            }
            _ => {}
        }
    }

    LoadedMesaSample {
        nz: nz.expect("sample missing INT LOADED nz"),
        species: species.expect("sample missing INT LOADED species"),
        net_name: net_name.expect("sample missing STR LOADED net_name"),
        zones,
        xa,
    }
}

fn assert_zone_matches_mesa_hex(
    label: &str,
    k_1based: usize,
    model: &StarModel,
    sample: &LoadedMesaSample,
) {
    let zone = k_1based - 1;
    let exp_bits = sample.zone_bits(k_1based);
    let actual = [
        ("lnd", model.lnd[zone], exp_bits[0]),
        ("lnT", model.ln_t[zone], exp_bits[1]),
        ("lnR", model.ln_r[zone], exp_bits[2]),
        ("L", model.lum[zone], exp_bits[3]),
        ("dq", model.dq[zone], exp_bits[4]),
    ];
    for (field, got, want_bits) in actual {
        let got_bits = got.to_bits();
        assert_eq!(
            got_bits,
            want_bits,
            "{label} zone k={k_1based} {field} bit mismatch: \
             got {got_bits:016X} ({got:.17e}), expected {want_bits:016X} \
             ({:.17e})",
            f64::from_bits(want_bits),
        );
    }
}

#[test]
fn test_create_initial_model_with_change_net_matches_mesa_26_4_1() {
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

    let sample = load_relax154_loaded_sample(
        "create_initial_model_pp_and_cno_extras.sample",
    );
    assert_eq!(model.job.net_name, sample.net_name);
    assert_eq!(model.nz, sample.nz);
    assert_eq!(model.species, sample.species);

    // Surface, mid, and core zones compared bit-exact against MESA 26.4.1
    // `LOADED` hex captures.
    for k1 in [1_usize, 388, 774] {
        assert_zone_matches_mesa_hex(
            "create_initial_model pp_and_cno_extras",
            k1,
            &model,
            &sample,
        );
    }

    let expected_xa_0 = sample.xa_bits(1);
    assert_eq!(expected_xa_0.len(), sample.species);
    for (j, &exp_bits) in expected_xa_0.iter().enumerate() {
        let got_bits = model.xa[j].to_bits();
        assert_eq!(
            got_bits,
            exp_bits,
            "xa[{j}] bit mismatch: got {got_bits:016X} ({}), expected \
             {exp_bits:016X} ({})",
            model.xa[j],
            f64::from_bits(exp_bits),
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
fn test_create_initial_model_brown_dwarf_and_prems_15m_match_mesa_26_4_1() {
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
    let bd_sample =
        load_relax154_loaded_sample("make_brown_dwarf_basic_net.sample");
    assert_eq!(bd.nz, bd_sample.nz);
    assert_eq!(bd.species, bd_sample.species);
    assert_eq!(bd.job.net_name, bd_sample.net_name);
    for k1 in [1_usize, 388, 774] {
        assert_zone_matches_mesa_hex(
            "make_brown_dwarf basic.net",
            k1,
            &bd,
            &bd_sample,
        );
    }

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
    let m15_sample = load_relax154_loaded_sample("pre_ms_15m_dynamo.sample");
    assert_eq!(m15.nz, m15_sample.nz);
    assert_eq!(m15.species, m15_sample.species);
    assert_eq!(m15.job.net_name, m15_sample.net_name);
    for k1 in [1_usize, 517, 1033] {
        assert_zone_matches_mesa_hex(
            "15M_dynamo pre_ms",
            k1,
            &m15,
            &m15_sample,
        );
    }
}

#[test]
fn test_scalar_profile_parsers_and_eos_modes() {
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

    // Core `constant_omega_center` invariant (`relax.f90:1208-1230`):
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
