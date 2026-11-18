// SPDX-License-Identifier: LGPL-3.0-or-later
//! Fresh-OS-process `.photo` and `BinarySystem` checkpoint restart parity and
//! validation tests (`#150`).

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use flexmesa_binary::{
    BinaryInlistControls, BinarySystem, DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_math::{numerics_mode, NumericsMode};
use flexmesa_star::{
    load_photo_bytes, load_zams_model, star_evolve_step, write_photo_bytes,
    StarControls, StarJobControls, StarModel,
};

fn unique_temp_path(label: &str, ext: &str) -> PathBuf {
    let pid = std::process::id();
    env::temp_dir().join(format!("flexmesa_150_{label}_{pid}.{ext}"))
}

fn build_mutated_star() -> StarModel {
    let controls = StarControls {
        initial_mass: 1.0,
        mixing_length_alpha: 1.93,
        mesh_delta_coeff: 0.85,
        ..StarControls::default()
    };
    let job = StarJobControls {
        set_initial_age: true,
        initial_age: 125.0,
        ..StarJobControls::default()
    };
    load_zams_model(controls, job).expect("1 Msun ZAMS")
}

fn build_rlof_binary() -> BinarySystem {
    let donor_ctrl = StarControls {
        initial_mass: 1.5,
        ..StarControls::default()
    };
    let mut donor = load_zams_model(donor_ctrl, StarJobControls::default())
        .expect("1.5 Msun donor ZAMS");
    donor.dt = 1.0e3 * SECYER;
    donor.dt_next = 1.0e3 * SECYER;

    let mut bctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 0.45,
        mdot_scheme: String::from("Ritter"),
        max_tries_to_achieve: 20,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("RLOF point-mass binary")
}

fn advance_binary_steps_one_to_three(sys: &mut BinarySystem) {
    let mut extras = DefaultBinaryExtras;
    let (next_dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut extras)
        .expect("binary step 1");
    sys.controls.fr_hard = 1.0e-5;
    let (next_dt2, _) = sys
        .try_step_with_extras(next_dt1.max(1.0e4 * SECYER), &mut extras)
        .expect("binary step 2");
    let _ = sys
        .try_step_with_extras(next_dt2, &mut extras)
        .expect("binary step 3");
    assert_eq!(sys.model_number, 3);
}

fn advance_binary_steps_four_to_six(sys: &mut BinarySystem) {
    let mut extras = DefaultBinaryExtras;
    let mut dt = sys.donor.dt_next;
    assert!(dt > 0.0, "donor.dt_next must be positive at step 3");
    for step in 4..=6 {
        let (next_dt, _) = sys
            .try_step_with_extras(dt, &mut extras)
            .unwrap_or_else(|e| panic!("binary step {step} failed: {e}"));
        dt = next_dt;
    }
    assert_eq!(sys.model_number, 6);
}

fn run_child_star(in_path: &Path, out_path: &Path) {
    let bytes = fs::read(in_path).expect("child read star .photo");
    let mut model = load_photo_bytes(&bytes).expect("child load_photo_bytes");
    assert_eq!(model.model_number, 4);
    assert_eq!(
        model.controls.mixing_length_alpha.to_bits(),
        1.93_f64.to_bits()
    );
    assert_eq!(
        model.controls.mesh_delta_coeff.to_bits(),
        0.85_f64.to_bits()
    );
    assert!(model.job.set_initial_age);
    assert_eq!(model.job.initial_age.to_bits(), 125.0_f64.to_bits());

    for step in 5..=7 {
        let rep = star_evolve_step(&mut model, true, true, true)
            .unwrap_or_else(|e| panic!("child star step {step} failed: {e}"));
        assert!(rep.converged, "child star step {step} did not converge");
    }
    assert_eq!(model.model_number, 7);
    fs::write(out_path, write_photo_bytes(&model))
        .expect("child write star output .photo");
}

fn run_child_binary(in_path: &Path, out_path: &Path) {
    let mut sys =
        BinarySystem::load_checkpoint(in_path).expect("child load_checkpoint");
    assert_eq!(sys.model_number, 3);
    assert!(sys.counters.num_implicit_trials >= 1);
    assert_eq!(sys.controls.fr_hard.to_bits(), 1.0e-5_f64.to_bits());
    advance_binary_steps_four_to_six(&mut sys);
    sys.save_checkpoint(out_path)
        .expect("child save_checkpoint");
}

fn run_child_mismatch(in_path: &Path) {
    assert_eq!(numerics_mode(), NumericsMode::Fast);
    let bytes = fs::read(in_path).expect("child read mismatch checkpoint");
    let err = BinarySystem::from_checkpoint_bytes(&bytes)
        .expect_err("Fast mode must reject Strict BinarySystem checkpoint");
    let msg = err.to_string();
    assert!(
        msg.contains("numerics_mode mismatch"),
        "expected numerics_mode mismatch error, got: {msg}"
    );
}

#[test]
fn child_process_restart_worker() {
    let Ok(mode) = env::var("FLEXMESA_CHILD_RESTART_MODE") else {
        return;
    };
    let in_path = PathBuf::from(
        env::var("FLEXMESA_CHILD_IN").expect("FLEXMESA_CHILD_IN set"),
    );
    match mode.as_str() {
        "star" => {
            let out_path = PathBuf::from(
                env::var("FLEXMESA_CHILD_OUT").expect("FLEXMESA_CHILD_OUT set"),
            );
            run_child_star(&in_path, &out_path);
        }
        "binary" => {
            let out_path = PathBuf::from(
                env::var("FLEXMESA_CHILD_OUT").expect("FLEXMESA_CHILD_OUT set"),
            );
            run_child_binary(&in_path, &out_path);
        }
        "mismatch" => run_child_mismatch(&in_path),
        other => panic!("unexpected FLEXMESA_CHILD_RESTART_MODE: {other}"),
    }
}

fn spawn_restart_child(mode: &str, in_path: &Path, out_path: Option<&Path>) {
    let exe = env::current_exe().expect("current test executable");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", "child_process_restart_worker", "--nocapture"])
        .env("FLEXMESA_CHILD_RESTART_MODE", mode)
        .env("FLEXMESA_CHILD_IN", in_path);
    if mode == "mismatch" {
        cmd.env("FLEXMESA_NUMERICS_MODE", "fast");
    }
    if let Some(out) = out_path {
        cmd.env("FLEXMESA_CHILD_OUT", out);
    }
    let output = cmd.output().expect("spawn child restart test process");
    assert!(
        output.status.success(),
        "child restart mode {mode} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_bytes_bit_exact(actual: &[u8], expected: &[u8], label: &str) {
    if actual == expected {
        return;
    }
    let first_diff =
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b);
    panic!(
        "{label}: bytes differ (actual_len={}, expected_len={}, first_diff={first_diff:?})",
        actual.len(),
        expected.len()
    );
}

#[test]
fn test_star_photo_restart_in_child_process_is_bit_exact() {
    let in_path = unique_temp_path("star_in", "photo");
    let out_path = unique_temp_path("star_out", "photo");

    let mut split_star = build_mutated_star();
    for step in 1..=3 {
        let rep = star_evolve_step(&mut split_star, true, true, true)
            .unwrap_or_else(|e| panic!("split star step {step} failed: {e}"));
        assert!(rep.converged);
    }
    fs::write(&in_path, write_photo_bytes(&split_star))
        .expect("write intermediate star .photo");

    spawn_restart_child("star", &in_path, Some(&out_path));
    let child_bytes = fs::read(&out_path).expect("read child star .photo");
    let _ = fs::remove_file(&in_path);
    let _ = fs::remove_file(&out_path);

    let mut uninterrupted = build_mutated_star();
    for step in 1..=6 {
        let rep = star_evolve_step(&mut uninterrupted, true, true, true)
            .unwrap_or_else(|e| {
                panic!("uninterrupted star step {step} failed: {e}")
            });
        assert!(rep.converged);
    }
    let continuous_bytes = write_photo_bytes(&uninterrupted);
    assert_bytes_bit_exact(
        &child_bytes,
        &continuous_bytes,
        "child-restarted StarModel .photo",
    );
}

#[test]
fn test_binary_checkpoint_restart_in_child_process_is_bit_exact() {
    let in_path = unique_temp_path("bin_in", "chk");
    let out_path = unique_temp_path("bin_out", "chk");

    let mut split_sys = build_rlof_binary();
    advance_binary_steps_one_to_three(&mut split_sys);
    split_sys
        .save_checkpoint(&in_path)
        .expect("save binary checkpoint");

    spawn_restart_child("binary", &in_path, Some(&out_path));
    let child_bytes =
        fs::read(&out_path).expect("read child binary checkpoint");
    let _ = fs::remove_file(&in_path);
    let _ = fs::remove_file(&out_path);

    let mut uninterrupted = build_rlof_binary();
    advance_binary_steps_one_to_three(&mut uninterrupted);
    advance_binary_steps_four_to_six(&mut uninterrupted);
    let continuous_bytes = uninterrupted
        .to_checkpoint_bytes()
        .expect("uninterrupted to_checkpoint_bytes");
    assert_bytes_bit_exact(
        &child_bytes,
        &continuous_bytes,
        "child-restarted BinarySystem checkpoint",
    );
}

#[test]
fn test_binary_checkpoint_rejects_corruption_and_numerics_mismatch() {
    let mut sys = build_rlof_binary();
    advance_binary_steps_one_to_three(&mut sys);
    let bytes = sys.to_checkpoint_bytes().expect("to_checkpoint_bytes");

    let truncated = &bytes[..bytes.len().saturating_sub(1)];
    assert!(
        BinarySystem::from_checkpoint_bytes(truncated).is_err(),
        "truncated checkpoint must fail"
    );

    let mut trailing = bytes.clone();
    trailing.push(0xA5);
    assert!(
        BinarySystem::from_checkpoint_bytes(&trailing).is_err(),
        "trailing garbage byte must fail"
    );

    let chk_path = unique_temp_path("bin_mismatch", "chk");
    fs::write(&chk_path, &bytes).expect("write mismatch checkpoint");
    spawn_restart_child("mismatch", &chk_path, None);
    let _ = fs::remove_file(&chk_path);
}
