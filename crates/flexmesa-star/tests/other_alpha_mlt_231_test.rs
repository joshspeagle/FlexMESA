// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration invariants for per-zone `alpha_mlt(1:nz)` via `other_alpha_mlt`
//! and per-zone `burn_num_iters(1:nz)` (`#231`).

#![forbid(unsafe_code)]

use flexmesa_math::log;
use flexmesa_star::hydro_state::{StarModelHook, StarModelHookHandle};
use flexmesa_star::micro::{do_eos, set_basic_vars};
use flexmesa_star::mix_info::set_mixing_info;
use flexmesa_star::{
    do_struct_burn_mix, load_zams_model, op_split_burn_zones,
    record_start_of_step_energy_integrals, set_hydro_vars, PrimaryStateBuffer,
    StarControls, StarError, StarJobControls, StarModel,
};

struct ZoneSplitAlphaHook {
    alpha_outer: f64,
    alpha_inner: f64,
    q_split: f64,
}

impl StarModelHook for ZoneSplitAlphaHook {
    fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
        let nz = model.nz;
        for k in 0..nz {
            model.aux.mixing.alpha_mlt[k] = if model.q[k] >= self.q_split {
                self.alpha_outer
            } else {
                self.alpha_inner
            };
        }
        Ok(())
    }
}

#[test]
fn test_set_hydro_vars_invokes_other_alpha_mlt_leaving_do1_mlt_2_scalar() {
    let base_ctrl = StarControls {
        initial_mass: 4.0,
        zbase: 0.02,
        mixing_length_alpha: 2.0,
        ..StarControls::default()
    };
    let mut m_scalar =
        load_zams_model(base_ctrl.clone(), StarJobControls::default())
            .expect("4 Msun ZAMS");
    let mut m_hook = m_scalar.clone();

    set_hydro_vars(&mut m_scalar, false, false)
        .expect("baseline set_hydro_vars");

    m_hook.controls.use_other_alpha_mlt = true;
    m_hook.aux.other_alpha_mlt =
        Some(StarModelHookHandle::new(ZoneSplitAlphaHook {
            alpha_outer: 3.2,
            alpha_inner: 1.4,
            q_split: 0.5,
        }));
    set_hydro_vars(&mut m_hook, false, false).expect("hooked set_hydro_vars");

    let nz = m_hook.nz;
    assert_eq!(m_hook.aux.mixing.alpha_mlt.len(), nz);
    assert_eq!(m_hook.aux.mixing.alpha_mlt[0].to_bits(), 3.2_f64.to_bits());
    assert_eq!(
        m_hook.aux.mixing.alpha_mlt[nz - 1].to_bits(),
        1.4_f64.to_bits()
    );

    // Upstream `turb_info.f90:107-111` (`do1_mlt_2`) reads scalar
    // `mixing_length_alpha` rather than per-zone `alpha_mlt(k)`.
    for k in 0..nz {
        assert_eq!(
            m_hook.gradt[k].to_bits(),
            m_scalar.gradt[k].to_bits(),
            "gradt mismatch at k={k}"
        );
        assert_eq!(
            m_hook.mlt_vc[k].to_bits(),
            m_scalar.mlt_vc[k].to_bits(),
            "mlt_vc mismatch at k={k}"
        );
    }

    // Strict error when `use_other_alpha_mlt = true` without an installed hook.
    let mut m_missing = m_scalar;
    m_missing.controls.use_other_alpha_mlt = true;
    m_missing.aux.other_alpha_mlt = None;
    let err = set_hydro_vars(&mut m_missing, false, false)
        .expect_err("missing other_alpha_mlt hook must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
}

#[test]
fn test_set_mixing_info_propagates_per_zone_alpha_mlt_and_primary_state_round_trip(
) {
    let base_ctrl = StarControls {
        initial_mass: 4.0,
        zbase: 0.02,
        mixing_length_alpha: 2.0,
        ..StarControls::default()
    };
    let mut m_scalar = load_zams_model(base_ctrl, StarJobControls::default())
        .expect("4 Msun ZAMS");
    let mut m_hook = m_scalar.clone();

    set_hydro_vars(&mut m_scalar, false, false).expect("scalar hydro_vars");
    set_mixing_info(&mut m_scalar).expect("scalar set_mixing_info");

    m_hook.controls.use_other_alpha_mlt = true;
    m_hook.aux.other_alpha_mlt =
        Some(StarModelHookHandle::new(ZoneSplitAlphaHook {
            alpha_outer: 3.2,
            alpha_inner: 1.4,
            q_split: 0.5,
        }));
    set_hydro_vars(&mut m_hook, false, false).expect("hooked hydro_vars");
    set_mixing_info(&mut m_hook).expect("hooked set_mixing_info");

    // In `mix_info.rs:258-266` (`mix_info.f90:364-376`), `conv_vel` reads
    // `model.aux.mixing.alpha_mlt[k]`. Pick a convective core face (`q < 0.5`).
    let nz = m_hook.nz;
    let k_core_conv = (1..nz)
        .find(|&k| {
            m_hook.q[k] < 0.5
                && m_hook.d_mix[k] > 0.0
                && m_hook.conv_vel[k] > 0.0
        })
        .expect("4 Msun ZAMS must have convective core face with q < 0.5");
    assert_ne!(
        m_hook.conv_vel[k_core_conv].to_bits(),
        m_scalar.conv_vel[k_core_conv].to_bits(),
        "set_mixing_info must propagate per-zone alpha_mlt into conv_vel"
    );

    // `PrimaryStateBuffer` round-trips per-zone `alpha_mlt`, `burn_num_iters`,
    // and the installed `other_alpha_mlt` handle across retry rollbacks.
    m_hook.aux.burn_num_iters = (0..nz).map(|k| (k % 7) + 1).collect();
    let expected_alpha: Vec<u64> = m_hook
        .aux
        .mixing
        .alpha_mlt
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let expected_iters = m_hook.aux.burn_num_iters.clone();

    let buf = PrimaryStateBuffer::from_model(&m_hook);
    m_hook.aux.mixing.alpha_mlt.fill(9.9);
    m_hook.aux.burn_num_iters.fill(999);
    m_hook.aux.other_alpha_mlt = None;
    buf.restore_into(&mut m_hook).expect("restore_into");

    let actual_alpha: Vec<u64> = m_hook
        .aux
        .mixing
        .alpha_mlt
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(actual_alpha, expected_alpha);
    assert_eq!(m_hook.aux.burn_num_iters, expected_iters);
    assert!(m_hook.aux.other_alpha_mlt.is_some());
}

#[test]
fn test_op_split_burn_zones_and_do_struct_burn_mix_burn_num_iters() {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1 Msun ZAMS");
    let nz = model.nz;
    let sp = model.species;
    let kc = nz - 1;

    model.ln_t[kc] = log(1.8e8);
    model.ln_t_start = model.ln_t.clone();
    model.lnd[kc] = log(1.0e5);
    model.xa[kc * sp..kc * sp + sp]
        .copy_from_slice(&[0.25, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0]);
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");

    model.controls.op_split_burn = true;
    model.controls.op_split_burn_min_t = 1.0e8;
    model.controls.eps_nuc_factor = 1.0;
    model.save_start_values().expect("save_start_values");

    // Pre-existing values in cool or skipped zones must be zeroed.
    model.aux.burn_num_iters.fill(99);
    op_split_burn_zones(&mut model, 10.0).expect("op_split_burn_zones");
    assert_eq!(model.aux.burn_num_iters.len(), nz);
    assert!(model.aux.burn_num_iters[kc] > 0);
    assert_eq!(model.aux.burn_num_iters[0], 0);

    // Zeroing both burn factors skips split burn and resets `burn_num_iters`.
    let mut m_zero_factors = model.clone();
    m_zero_factors.controls.eps_nuc_factor = 0.0;
    m_zero_factors.controls.dxdt_nuc_factor = 0.0;
    op_split_burn_zones(&mut m_zero_factors, 10.0)
        .expect("op_split_burn_zones zero factors");
    assert!(m_zero_factors.aux.burn_num_iters.iter().all(|&n| n == 0));

    // `do_struct_burn_mix` preserves `burn_num_iters` when split burn is
    // disabled (`struct_burn_mix.f90:82-84`) and clears cool/skipped zones when
    // split burn runs.
    let mut clean = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1 Msun ZAMS for do_struct_burn_mix");
    clean.aux.need_to_setvars = true;
    set_hydro_vars(&mut clean, false, false).expect("clean set_hydro_vars");
    clean.aux.need_to_setvars = false;
    clean.save_start_values().expect("clean save_start_values");
    record_start_of_step_energy_integrals(&mut clean);
    clean.save_old_values();
    clean.aux.burn_num_iters.fill(77);
    let rep_coupled = do_struct_burn_mix(&mut clean, 10.0, false, true, &[])
        .expect("do_struct_burn_mix coupled");
    assert!(rep_coupled.converged, "coupled sbm = {rep_coupled:?}");
    assert!(clean.aux.burn_num_iters.iter().all(|&n| n == 77));

    clean
        .save_start_values()
        .expect("clean save_start_values 2");
    record_start_of_step_energy_integrals(&mut clean);
    clean.save_old_values();
    clean.aux.burn_num_iters.fill(77);
    let rep_split = do_struct_burn_mix(&mut clean, 10.0, true, true, &[])
        .expect("do_struct_burn_mix split");
    assert!(rep_split.converged, "split sbm = {rep_split:?}");
    assert_eq!(clean.aux.burn_num_iters[0], 0);
}
