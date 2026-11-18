// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for `PrimaryStateBuffer` rollback preservation and
//! `MicrophysicsCache` 0-ULP hit/miss behavior (`#150`).

#![forbid(unsafe_code)]

use flexmesa_star::{
    load_zams_model, set_hydro_vars, star_evolve_step, PrimaryStateBuffer,
    StarControls, StarJobControls,
};

fn assert_bits_eq(label: &str, actual: &[f64], expected_bits: &[u64]) {
    assert_eq!(actual.len(), expected_bits.len(), "{label} length mismatch");
    for (idx, (&val, &want)) in actual.iter().zip(expected_bits).enumerate() {
        assert_eq!(val.to_bits(), want, "{label} mismatch at idx {idx}");
    }
}

#[test]
fn test_primary_state_buffer_zero_ulp_round_trip_memory_and_counter_survival() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        rotation_flag: true,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save_start_values");
    model.save_old_values();

    let buf = PrimaryStateBuffer::from_model(&model);
    let model_bytes = model.heap_bytes();
    let buf_bytes = buf.heap_bytes();
    assert!(
        buf_bytes * 4 < model_bytes,
        "PrimaryStateBuffer ({buf_bytes} B) should be >4x smaller than StarModel ({model_bytes} B)"
    );

    let ref_lnd: Vec<u64> = model.lnd.iter().map(|v| v.to_bits()).collect();
    let ref_peos: Vec<u64> = model.peos.iter().map(|v| v.to_bits()).collect();
    let ref_kap: Vec<u64> = model.opacity.iter().map(|v| v.to_bits()).collect();
    let ref_eps: Vec<u64> = model.eps_nuc.iter().map(|v| v.to_bits()).collect();
    let ref_jac: Vec<u64> =
        model.d_dxdt_nuc_dx.iter().map(|v| v.to_bits()).collect();
    let ref_teff = model.teff.to_bits();

    // Mutate state and advance all 6 diagnostic counters as a failed attempt would.
    model.model_number += 1;
    model.star_age += 123.0;
    model.lnd[0] += 0.25;
    model.ln_t[0] -= 0.15;
    model.d_dxdt_nuc_dx[0] += 999.0;
    model.num_retries += 3;
    model.num_redos += 2;
    model.total_num_solver_iterations += 17;
    model.total_num_solver_calls += 5;
    model.micro_cache_hits += 11;
    model.micro_cache_misses += 4;
    buf.restore_into(&mut model).expect("restore_into");

    assert_eq!(
        (
            model.num_retries,
            model.num_redos,
            model.total_num_solver_iterations,
            model.total_num_solver_calls,
        ),
        (3, 2, 17, 5),
        "rollback must preserve solver/retry/redo counters"
    );
    assert_eq!(
        (model.micro_cache_hits, model.micro_cache_misses),
        (12, 5),
        "restore_into preserves cache and reuses warm microphysics entry"
    );
    assert_eq!(model.teff.to_bits(), ref_teff);
    assert_bits_eq("lnd", &model.lnd, &ref_lnd);
    assert_bits_eq("peos", &model.peos, &ref_peos);
    assert_bits_eq("opacity", &model.opacity, &ref_kap);
    assert_bits_eq("eps_nuc", &model.eps_nuc, &ref_eps);
    assert_bits_eq("d_dxdt_nuc_dx", &model.d_dxdt_nuc_dx, &ref_jac);
}

#[test]
fn test_microphysics_cache_hit_exactness_and_one_ulp_invalidation() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    model.aux.micro_cache.clear();
    model.micro_cache_hits = 0;
    model.micro_cache_misses = 0;

    set_hydro_vars(&mut model, false, false).expect("cold eval");
    assert_eq!(model.micro_cache_misses, 1);
    assert_eq!(model.micro_cache_hits, 0);

    let peos0: Vec<u64> = model.peos.iter().map(|v| v.to_bits()).collect();
    let kap0: Vec<u64> = model.opacity.iter().map(|v| v.to_bits()).collect();
    let eps0: Vec<u64> = model.eps_nuc.iter().map(|v| v.to_bits()).collect();

    // Exact unchanged repeat hits cache with 0 ULP diff.
    set_hydro_vars(&mut model, false, false).expect("cached eval");
    assert_eq!(model.micro_cache_misses, 1);
    assert_eq!(model.micro_cache_hits, 1);
    assert_eq!(
        model.peos.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        peos0
    );
    assert_eq!(
        model
            .opacity
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        kap0
    );
    assert_eq!(
        model
            .eps_nuc
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        eps0
    );

    // 1-ULP perturbation of lnd invalidates cache.
    let orig_dens0 = model.lnd[0];
    model.lnd[0] = f64::from_bits(orig_dens0.to_bits() + 1);
    set_hydro_vars(&mut model, false, false).expect("1-ULP lnd");
    assert_eq!(model.micro_cache_misses, 2);

    // Restoring orig_dens0 hits slot 0 (2-slot LRU), then 1-ULP ln_t misses.
    model.lnd[0] = orig_dens0;
    set_hydro_vars(&mut model, false, false).expect("back to baseline");
    assert_eq!(model.micro_cache_hits, 2);

    let orig_temp0 = model.ln_t[0];
    model.ln_t[0] = f64::from_bits(orig_temp0.to_bits() + 1);
    set_hydro_vars(&mut model, false, false).expect("1-ULP ln_t");
    assert_eq!(model.micro_cache_misses, 3);

    model.ln_t[0] = orig_temp0;
    model.xa[0] = f64::from_bits(model.xa[0].to_bits() + 1);
    set_hydro_vars(&mut model, false, false).expect("1-ULP xa");
    assert_eq!(model.micro_cache_misses, 4);
}

#[test]
fn test_star_evolve_step_populates_solver_calls_and_cache_hits() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    model.dt = 1.0e11;
    model.dt_next = 1.0e11;
    let rep = star_evolve_step(&mut model, true, false, true).expect("step 1");
    assert!(rep.converged);
    assert!(model.total_num_solver_calls >= 1);
    assert!(model.total_num_solver_iterations >= rep.iterations);
    assert!(model.micro_cache_hits > 0);
}
