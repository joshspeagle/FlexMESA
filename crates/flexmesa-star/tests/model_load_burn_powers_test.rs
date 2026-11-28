// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for `.mod` header burning-power persistence across
//! `read_mod_str`, `write_mod_string`, `finish_load_model`, `.photo` /
//! [`PrimaryStateBuffer`] round-trips, `restore_from_old`, relax steps, and
//! step-1 to step-2 `power_*` / `power_*_old` handoff
//! (`star/private/read_model.f90:285, 827-831`,
//! `star/private/write_model.f90:163-167`, `star/private/evolve.f90:715`,
//! `star/private/evolve_support.f90:94-99, 168-172`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::{
    compute_power_categories, init_or_update_model_from_bundle,
    load_photo_bytes, load_zams_model, read_mod_str, restore_from_old,
    star_evolve_step, star_evolve_step_part1, star_evolve_step_part2,
    star_pick_next_timestep_after_step, write_mod_string, write_photo_bytes,
    DefaultStarExtras, InlistBundle, PrimaryStateBuffer, StarControls,
    StarError, StarJobControls, StarModel, StarPickTimestepOutcome,
    StarStepAttemptOutcome,
};

const POWER_KEYS: [&str; 5] = [
    "power_h_burn",
    "power_he_burn",
    "power_z_burn",
    "power_photo",
    "power_nuc_burn",
];

const SENTINEL_MOD_OVERRIDES: [(&str, Option<&str>); 5] = [
    ("power_h_burn", Some("3.2500000000000000D+00")),
    ("power_he_burn", Some("4.5000000000000000D-02")),
    ("power_z_burn", Some("1.5000000000000000D-04")),
    ("power_photo", Some("-2.5000000000000000D-05")),
    ("power_nuc_burn", Some("3.2951250000000000D+00")),
];

const SENTINEL_POWERS: [f64; 5] = [3.25, 4.5e-2, 1.5e-4, -2.5e-5, 3.295_125];

fn zams_baseline() -> StarModel {
    let controls = StarControls {
        zbase: 0.02,
        ..StarControls::default()
    };
    load_zams_model(controls, StarJobControls::default())
        .expect("builtin 1.0 Msun ZAMS model")
}

fn current_powers(model: &StarModel) -> [f64; 5] {
    let sh = &model.aux.step_hist;
    [
        sh.power_h_burn,
        sh.power_he_burn,
        sh.power_z_burn,
        sh.power_photo,
        sh.power_nuc_burn,
    ]
}

fn old_powers(model: &StarModel) -> [f64; 5] {
    let sh = &model.aux.step_hist;
    [
        sh.power_h_burn_old,
        sh.power_he_burn_old,
        sh.power_z_burn_old,
        sh.power_photo_old,
        sh.power_nuc_burn_old,
    ]
}

fn computed_powers(model: &StarModel) -> [f64; 5] {
    let c = compute_power_categories(model);
    [
        c.power_h_burn,
        c.power_he_burn,
        c.power_z_burn,
        c.power_photo,
        c.power_nuc_burn,
    ]
}

fn assert_power_array_eq(actual: [f64; 5], expected: [f64; 5], ctx: &str) {
    for (i, (got, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "{ctx}: {} mismatch (got {got:.16e}, expected {exp:.16e})",
            POWER_KEYS[i]
        );
    }
}

fn rewrite_mod_power_lines(
    text: &str,
    overrides: &[(&str, Option<&str>)],
) -> String {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        let matched = overrides.iter().find(|(key, _)| {
            trimmed.starts_with(key)
                && trimmed[key.len()..].starts_with(char::is_whitespace)
        });
        match matched {
            Some((key, Some(val_str))) => {
                out.push(format!("{key:>19}      {val_str:>26}"));
            }
            Some((_, None)) => {}
            None => out.push(line.to_owned()),
        }
    }
    let mut joined = out.join("\n");
    if text.ends_with('\n') {
        joined.push('\n');
    }
    joined
}

fn temp_case_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("flexmesa_mod_powers_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temporary case dir");
    dir
}

fn load_patched_mod_via_inlist(
    tag: &str,
    overrides: &[(&str, Option<&str>)],
    extra_star_job: &str,
) -> StarModel {
    let zams = zams_baseline();
    let mod_text = write_mod_string(&zams).expect("serialize .mod");
    let patched = rewrite_mod_power_lines(&mod_text, overrides);

    let dir = temp_case_dir(tag);
    fs::write(dir.join("saved.mod"), &patched).expect("write saved.mod");
    let inlist = format!(
        "\
&star_job
   load_saved_model = .true.
   load_model_filename = 'saved.mod'
{extra_star_job}/
&eos
/
&kap
   Zbase = 0.02d0
   kap_file_prefix = 'gs98'
/
&controls
   initial_mass = 1.0d0
   initial_z = 0.02d0
/
"
    );
    fs::write(dir.join("inlist"), inlist).expect("write inlist");
    let bundle = InlistBundle::from_file_in_case_dir(&dir, "inlist")
        .expect("load inlist bundle");
    let mut extras = DefaultStarExtras;
    let model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("init_or_update_model_from_bundle from .mod");
    let _ = fs::remove_dir_all(&dir);
    model
}

#[test]
fn mod_header_burn_powers_round_trip_preserves_signed_values_and_format() {
    let zams = zams_baseline();
    assert_power_array_eq(
        current_powers(&zams),
        [-1.0; 5],
        "fresh ZAMS power_*",
    );
    assert_power_array_eq(
        old_powers(&zams),
        [-1.0; 5],
        "fresh ZAMS power_*_old",
    );

    let mod_text = write_mod_string(&zams).expect("serialize .mod");

    // Verify upstream `write_model.f90:162-167` order right after `Teff`.
    let lines: Vec<&str> = mod_text.lines().collect();
    let teff_idx = lines
        .iter()
        .position(|ln| ln.trim_start().starts_with("Teff"))
        .expect("Teff header line");
    let expected_order = [
        "power_nuc_burn",
        "power_h_burn",
        "power_he_burn",
        "power_z_burn",
        "power_photo",
    ];
    for (offset, key) in expected_order.iter().enumerate() {
        let line = lines[teff_idx + 1 + offset].trim_start();
        assert!(
            line.starts_with(key),
            "expected '{key}' right after Teff, got '{line}'"
        );
    }

    // Override with sentinel powers including a negative `power_photo`
    // (as in MESA's `split_burn_big_net`).
    let patched = rewrite_mod_power_lines(
        &mod_text,
        &[
            ("power_nuc_burn", Some("1.9134817500000000D+02")),
            ("power_h_burn", Some("1.2345675000000000D+02")),
            ("power_he_burn", Some("6.7890125000000000D+01")),
            ("power_z_burn", Some("1.2500000000000000D-03")),
            ("power_photo", Some("-8.8056750000000000D+16")),
        ],
    );
    let expected = [
        1.234_567_5e2_f64,
        6.789_012_5e1_f64,
        1.25e-3_f64,
        -8.805_675e16_f64,
        1.913_481_75e2_f64,
    ];

    let loaded = read_mod_str(
        &patched,
        zams.controls.clone(),
        StarJobControls::default(),
    )
    .expect("parse patched .mod");
    assert!(
        (computed_powers(&loaded)[0] - expected[0]).abs() > 1.0,
        "sentinel must differ from recomputed ZAMS H power"
    );

    assert_power_array_eq(
        current_powers(&loaded),
        expected,
        "after read_mod_str current power_*",
    );
    assert_power_array_eq(
        old_powers(&loaded),
        expected,
        "after read_mod_str power_*_old",
    );

    let reserialized =
        write_mod_string(&loaded).expect("re-serialize loaded .mod");
    let reloaded =
        read_mod_str(&reserialized, zams.controls, StarJobControls::default())
            .expect("re-parse serialized .mod");
    assert_power_array_eq(
        current_powers(&reloaded),
        expected,
        "after .mod round-trip",
    );
}

#[test]
fn loaded_mod_powers_survive_init_photo_buffer_and_retry_restore() {
    let model = load_patched_mod_via_inlist(
        "init_photo_buf",
        &SENTINEL_MOD_OVERRIDES,
        "",
    );

    assert_power_array_eq(
        current_powers(&model),
        SENTINEL_POWERS,
        "after finish_load_model power_*",
    );
    assert_power_array_eq(
        old_powers(&model),
        SENTINEL_POWERS,
        "after finish_load_model power_*_old",
    );

    // 1. PrimaryStateBuffer round-trip before step 1.
    let buf = PrimaryStateBuffer::from_model(&model);
    let mut restored = model.clone();
    restored.aux.step_hist.power_h_burn = -99.0;
    restored.aux.step_hist.power_h_burn_old = -99.0;
    buf.restore_into(&mut restored)
        .expect("PrimaryStateBuffer restore");
    assert_power_array_eq(
        current_powers(&restored),
        SENTINEL_POWERS,
        "after PrimaryStateBuffer round-trip power_*",
    );
    assert_power_array_eq(
        old_powers(&restored),
        SENTINEL_POWERS,
        "after PrimaryStateBuffer round-trip power_*_old",
    );

    // 2. Binary .photo round-trip before step 1.
    let photo_bytes = write_photo_bytes(&model);
    let from_photo =
        load_photo_bytes(&photo_bytes).expect("reload .photo bytes");
    assert_power_array_eq(
        current_powers(&from_photo),
        SENTINEL_POWERS,
        "after .photo round-trip power_*",
    );
    assert_power_array_eq(
        old_powers(&from_photo),
        SENTINEL_POWERS,
        "after .photo round-trip power_*_old",
    );
}

#[test]
fn loaded_mod_powers_flow_into_step1_and_update_for_step2() {
    let mut model =
        load_patched_mod_via_inlist("step1_step2", &SENTINEL_MOD_OVERRIDES, "");
    let mut extras = DefaultStarExtras;

    let p1 = star_evolve_step_part1(&mut model, true, true, true, &mut extras)
        .expect("step 1 part1");
    assert_power_array_eq(
        current_powers(&model),
        SENTINEL_POWERS,
        "step 1 part1 power_* across remesh",
    );
    assert_power_array_eq(
        old_powers(&model),
        SENTINEL_POWERS,
        "step 1 part1 power_*_old across remesh",
    );

    let outcome =
        star_evolve_step_part2(&mut model, p1, true, false, false, &mut extras)
            .expect("step 1 part2");
    let StarStepAttemptOutcome::Converged(conv) = outcome else {
        panic!("expected step 1 to converge, got {outcome:?}");
    };
    let end_step1_powers = computed_powers(&model);
    assert!(
        (end_step1_powers[0] - SENTINEL_POWERS[0]).abs() > 1.0,
        "step 1 physical H power should differ from .mod sentinel"
    );
    // `set_power_info` updates `power_*` at end of `part2`, while `power_*_old`
    // retains `.mod` sentinels until step 2's `save_old_values`.
    assert_power_array_eq(
        current_powers(&model),
        end_step1_powers,
        "step 1 part2 updated power_*",
    );
    assert_power_array_eq(
        old_powers(&model),
        SENTINEL_POWERS,
        "step 1 part2 preserved power_*_old",
    );

    // Simulate `restore_from_old` (`evolve_support.f90:168-172`) on retry/redo
    // after `part2` ran `set_power_info`.
    let mut rewind = model.clone();
    restore_from_old(&mut rewind).expect("restore_from_old");
    assert_power_array_eq(
        current_powers(&rewind),
        SENTINEL_POWERS,
        "restore_from_old restores power_* from power_*_old",
    );

    let ts =
        star_pick_next_timestep_after_step(&mut model, &conv, 0, &mut extras)
            .expect("step 1 pick_next_timestep");
    assert!(
        matches!(ts, StarPickTimestepOutcome::KeepGoing(_)),
        "expected KeepGoing on step 1, got {ts:?}"
    );

    // Step 2 `part1` carries step 1's end-of-step `power_*` into `power_*_old`.
    let p2 = star_evolve_step_part1(&mut model, true, true, true, &mut extras)
        .expect("step 2 part1");
    assert_power_array_eq(
        old_powers(&model),
        end_step1_powers,
        "step 2 part1 power_*_old",
    );
    let outcome2 =
        star_evolve_step_part2(&mut model, p2, true, false, false, &mut extras)
            .expect("step 2 part2");
    assert!(
        matches!(outcome2, StarStepAttemptOutcome::Converged(_)),
        "expected step 2 convergence, got {outcome2:?}"
    );
    assert_power_array_eq(
        old_powers(&model),
        end_step1_powers,
        "step 2 part2 power_*_old",
    );
}

#[test]
fn single_call_step_persists_end_powers_across_between_step_control_mutation() {
    let mut single_model =
        load_patched_mod_via_inlist("single_call", &SENTINEL_MOD_OVERRIDES, "");
    star_evolve_step(&mut single_model, true, false, false)
        .expect("single-call step 1");
    let step1_end = computed_powers(&single_model);
    assert_power_array_eq(
        current_powers(&single_model),
        step1_end,
        "single-call step 1 end power_*",
    );
    assert_power_array_eq(
        old_powers(&single_model),
        SENTINEL_POWERS,
        "single-call step 1 power_*_old",
    );

    // Mutate `eps_nuc_factor` between step 1 and step 2 (analogous to
    // between-step hooks or post-relax control restores): `new_generation` in
    // step 2 must copy persisted `power_*` from step 1 (`step1_end`) rather
    // than recomputing 0.0 from the new controls.
    single_model.controls.eps_nuc_factor = 0.0;
    star_evolve_step(&mut single_model, true, false, false)
        .expect("single-call step 2");
    assert_power_array_eq(
        old_powers(&single_model),
        step1_end,
        "step 2 power_*_old unaffected by between-step eps_nuc_factor=0",
    );
    assert_power_array_eq(
        current_powers(&single_model),
        [0.0; 5],
        "step 2 end power_* reflects eps_nuc_factor=0",
    );
}

#[test]
fn missing_or_partial_mod_burn_powers_default_to_neg_one_like_mesa() {
    let zams = zams_baseline();
    let mod_text = write_mod_string(&zams).expect("serialize .mod");

    // Strip all five `power_*` header lines (simulating legacy `.mod`s such as
    // `co_wd_0.96M.mod`). Upstream initializes `power_* = -1.0` (`init.f90:375-379`).
    let stripped = rewrite_mod_power_lines(
        &mod_text,
        &[
            ("power_h_burn", None),
            ("power_he_burn", None),
            ("power_z_burn", None),
            ("power_photo", None),
            ("power_nuc_burn", None),
        ],
    );
    let mut legacy = read_mod_str(
        &stripped,
        zams.controls.clone(),
        StarJobControls::default(),
    )
    .expect("read legacy .mod without power_* headers");
    assert_power_array_eq(
        current_powers(&legacy),
        [-1.0; 5],
        "legacy .mod power_* defaults to -1.0",
    );
    assert_power_array_eq(
        old_powers(&legacy),
        [-1.0; 5],
        "legacy .mod power_*_old defaults to -1.0",
    );

    star_evolve_step(&mut legacy, true, false, false)
        .expect("legacy .mod step 1 evolves cleanly");
    let step1_end = computed_powers(&legacy);
    assert_power_array_eq(
        current_powers(&legacy),
        step1_end,
        "legacy .mod step 1 end power_* populated",
    );
    assert_power_array_eq(
        old_powers(&legacy),
        [-1.0; 5],
        "legacy .mod step 1 keeps power_*_old == -1.0",
    );

    star_evolve_step(&mut legacy, true, false, false)
        .expect("legacy .mod step 2 evolves cleanly");
    assert_power_array_eq(
        old_powers(&legacy),
        step1_end,
        "legacy .mod step 2 carries step 1 end power_*",
    );

    // Partial header: only `power_he_burn` present; other 4 stay `-1.0`.
    let partial = rewrite_mod_power_lines(
        &mod_text,
        &[
            ("power_h_burn", None),
            ("power_he_burn", Some("9.8765432500000000D+00")),
            ("power_z_burn", None),
            ("power_photo", None),
            ("power_nuc_burn", None),
        ],
    );
    let loaded_partial =
        read_mod_str(&partial, zams.controls, StarJobControls::default())
            .expect("read partial .mod header");
    let sentinel_he = 9.876_543_25_f64;
    assert_power_array_eq(
        old_powers(&loaded_partial),
        [-1.0, sentinel_he, -1.0, -1.0, -1.0],
        "partial .mod header keeps missing channels at -1.0",
    );
}

#[test]
fn loaded_mod_burn_powers_interact_cleanly_with_relax_initial_tau_factor() {
    let sentinel_h = 4.25_f64;
    let mut model = load_patched_mod_via_inlist(
        "relax_tau",
        &[
            ("power_h_burn", Some("4.2500000000000000D+00")),
            ("power_he_burn", Some("0.0000000000000000D+00")),
            ("power_z_burn", Some("0.0000000000000000D+00")),
            ("power_photo", Some("0.0000000000000000D+00")),
            ("power_nuc_burn", Some("4.2500000000000000D+00")),
        ],
        "   relax_initial_tau_factor = .true.\n   relax_to_this_tau_factor = 1.5d0\n   dlogtau_factor = 0.2d0\n",
    );

    // Internal relax evolution calls `set_power_info` at `evolve.f90:715`, and
    // `SavedStuff::restore` does not restore old `.mod` header powers
    // (`star/private/relax.f90:4234-4284`).
    assert!(
        (model.aux.step_hist.power_h_burn - sentinel_h).abs() > 1.0,
        "post-relax power_h_burn should reflect relax steps, not sentinel"
    );
    let post_relax = current_powers(&model);
    star_evolve_step(&mut model, true, false, false)
        .expect("post-relax step 1 succeeds");
    assert_power_array_eq(
        old_powers(&model),
        post_relax,
        "post-relax evolution step 1 carries relax end power_*",
    );
}

#[test]
fn non_finite_mod_burn_powers_return_io_error() {
    let zams = zams_baseline();
    let mod_text = write_mod_string(&zams).expect("serialize .mod");
    for bad in ["NaN", "Infinity"] {
        let bad_mod =
            rewrite_mod_power_lines(&mod_text, &[("power_photo", Some(bad))]);
        let err = read_mod_str(
            &bad_mod,
            zams.controls.clone(),
            StarJobControls::default(),
        )
        .expect_err("non-finite power_photo must fail");
        assert!(
            matches!(err, StarError::IoError(_)),
            "expected StarError::IoError for '{bad}', got {err:?}"
        );
    }
}
