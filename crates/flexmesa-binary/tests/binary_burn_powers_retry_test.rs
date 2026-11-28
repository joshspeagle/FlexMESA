// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled binary integration tests verifying `power_*` and `power_*_old`
//! persistence across pre-step remesh, explicit/implicit redos, outer retries,
//! `restore_mesh_on_retry`, and `.mod`-loaded binary donor initialization
//! (`#573`).

#![forbid(unsafe_code)]

mod common;

use flexmesa_binary::{
    BinaryExtras, BinaryExtrasAction, BinaryInlistControls, BinarySystem,
    DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_star::{
    compute_power_categories, load_zams_model, read_mod_str, write_mod_string,
    StarControls, StarJobControls, StarModel,
};

fn current_powers(star: &StarModel) -> [f64; 5] {
    let sh = &star.aux.step_hist;
    [
        sh.power_h_burn,
        sh.power_he_burn,
        sh.power_z_burn,
        sh.power_photo,
        sh.power_nuc_burn,
    ]
}

fn old_powers(star: &StarModel) -> [f64; 5] {
    let sh = &star.aux.step_hist;
    [
        sh.power_h_burn_old,
        sh.power_he_burn_old,
        sh.power_z_burn_old,
        sh.power_photo_old,
        sh.power_nuc_burn_old,
    ]
}

fn computed_powers(star: &StarModel) -> [f64; 5] {
    let c = compute_power_categories(star);
    [
        c.power_h_burn,
        c.power_he_burn,
        c.power_z_burn,
        c.power_photo,
        c.power_nuc_burn,
    ]
}

fn assert_powers_bit_eq(actual: [f64; 5], expected: [f64; 5], ctx: &str) {
    for (i, (got, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "{ctx}: channel {i} mismatch (got {got:.16e}, expected {exp:.16e})"
        );
    }
}

struct RedoThenRetryHarness {
    step_target: usize,
    expected_old: [f64; 5],
    calls: usize,
    did_redo: bool,
    did_retry: bool,
}

impl BinaryExtras for RedoThenRetryHarness {
    fn check_model(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        if bin.donor.model_number != self.step_target {
            return BinaryExtrasAction::KeepGoing;
        }
        self.calls += 1;
        // Regardless of pre-step remesh (`remesh_star_before_step`), prior
        // aborted `part2` updates to `power_*`, explicit `Redo`, or outer
        // `Retry` snapshot restoration, `power_*_old` must match step N-1's
        // end-of-step `power_*`.
        assert_powers_bit_eq(
            old_powers(&bin.donor),
            self.expected_old,
            &format!("step {} attempt {}", self.step_target, self.calls),
        );
        if !self.did_redo {
            self.did_redo = true;
            BinaryExtrasAction::Redo
        } else if !self.did_retry {
            self.did_retry = true;
            BinaryExtrasAction::Retry
        } else {
            BinaryExtrasAction::KeepGoing
        }
    }
}

fn run_binary_step2_remesh_redo_and_retry(restore_mesh_on_retry: bool) {
    let mut sys = common::build_rlof_binary();
    // Ensure step 2 runs `remesh_star_before_step` before `part1`, and toggle
    // `restore_mesh_on_retry` across both branches of `system_step.rs:352`.
    sys.donor.controls.mesh_delta_coeff = 0.2;
    sys.donor.controls.mix_wind.restore_mesh_on_retry = restore_mesh_on_retry;

    assert_powers_bit_eq(
        current_powers(&sys.donor),
        [-1.0; 5],
        "initial ZAMS binary donor power_*",
    );
    assert_powers_bit_eq(
        old_powers(&sys.donor),
        [-1.0; 5],
        "initial ZAMS binary donor power_*_old",
    );

    let (dt2, _) = sys
        .try_step_with_extras(5.0e2 * SECYER, &mut DefaultBinaryExtras)
        .expect("binary step 1");
    let step1_end = computed_powers(&sys.donor);
    assert_powers_bit_eq(
        current_powers(&sys.donor),
        step1_end,
        "after binary step 1 power_*",
    );
    assert!(step1_end[0] > 0.0 && step1_end[4] > 0.0);

    let mut extras = RedoThenRetryHarness {
        step_target: 2,
        expected_old: step1_end,
        calls: 0,
        did_redo: false,
        did_retry: false,
    };
    sys.try_step_with_extras(dt2, &mut extras)
        .expect("binary step 2 across redo + outer retry");
    assert!(extras.did_redo && extras.did_retry && extras.calls >= 3);
    assert_powers_bit_eq(
        old_powers(&sys.donor),
        step1_end,
        "step 2 accepted power_*_old matches step 1 end power_*",
    );
    assert_powers_bit_eq(
        current_powers(&sys.donor),
        computed_powers(&sys.donor),
        "step 2 end power_* updated via set_power_info",
    );
}

#[test]
fn binary_step_preserves_end_powers_across_remesh_redo_and_outer_retry() {
    run_binary_step2_remesh_redo_and_retry(false);
    run_binary_step2_remesh_redo_and_retry(true);
}

#[test]
fn binary_loaded_mod_burn_powers_survive_init_and_step1_retry() {
    let donor_base = load_zams_model(
        StarControls {
            initial_mass: 2.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("2.0 Msun donor ZAMS");
    let mod_text = write_mod_string(&donor_base).expect("serialize .mod");
    let sentinel = [4.5, 0.125, 2.5e-3, -1.25e-4, 4.627_375];
    let replacements = [
        ("power_h_burn", "4.5000000000000000D+00"),
        ("power_he_burn", "1.2500000000000000D-01"),
        ("power_z_burn", "2.5000000000000000D-03"),
        ("power_photo", "-1.2500000000000000D-04"),
        ("power_nuc_burn", "4.6273750000000000D+00"),
    ];
    let patched: String = mod_text
        .lines()
        .map(|ln| {
            let trimmed = ln.trim_start();
            if let Some((k, v)) = replacements.iter().find(|(k, _)| {
                trimmed.starts_with(k)
                    && trimmed[k.len()..].starts_with(char::is_whitespace)
            }) {
                format!("{k:>19}      {v:>26}")
            } else {
                ln.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let mut donor =
        read_mod_str(&patched, donor_base.controls, StarJobControls::default())
            .expect("read patched donor .mod");
    donor.dt = 1.0e3 * SECYER;
    donor.dt_next = 1.0e3 * SECYER;

    let mut bctrl = BinaryInlistControls {
        m1_msun: 2.0,
        m2_msun: 1.5,
        initial_period_in_days: 0.55,
        mdot_scheme: String::from("Ritter"),
        max_tries_to_achieve: 20,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;

    let mut sys = BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("binary from .mod donor");
    assert_powers_bit_eq(
        old_powers(&sys.donor),
        sentinel,
        "after prep_and_settle_binary_star power_*_old",
    );

    let mut extras = RedoThenRetryHarness {
        step_target: 1,
        expected_old: sentinel,
        calls: 0,
        did_redo: true,
        did_retry: false,
    };
    let (dt2, _) = sys
        .try_step_with_extras(5.0e2 * SECYER, &mut extras)
        .expect("step 1 with outer retry");
    assert!(extras.did_retry && extras.calls >= 2);
    assert_powers_bit_eq(
        old_powers(&sys.donor),
        sentinel,
        "step 1 power_*_old retains .mod header values across retry",
    );
    let step1_end = current_powers(&sys.donor);
    assert!(
        (step1_end[0] - sentinel[0]).abs() > 1.0,
        "step 1 end power_h_burn updated away from sentinel"
    );

    sys.try_step_with_extras(dt2, &mut DefaultBinaryExtras)
        .expect("step 2");
    assert_powers_bit_eq(
        old_powers(&sys.donor),
        step1_end,
        "step 2 carries step 1 end power_*",
    );
}
