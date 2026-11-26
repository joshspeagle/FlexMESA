// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` unit and MESA `26.4.1` parity tests for
//! `#153` adaptive mesh planning, mesh function weighting, gravitational
//! mass and the `cgrav` reset on remesh, surface pruning, and remapping
//! conservation.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_chem::{category_id, NUM_CATEGORIES};
use flexmesa_const::{MixingType, CLIGHT, LN10, M_SUN, R_SUN, STANDARD_CGRAV};
use flexmesa_math::{exp, log, log10};
use flexmesa_star::adjust_mesh_support::set_delta_gval_max_with_controls;
use flexmesa_star::controls_mix_wind::MixWindControls;
use flexmesa_star::{
    classify_new_cells, do_adjust_mesh, do_mesh_plan, do_prune_mesh_surface,
    get_gval_info_with_controls, load_zams_model, num_mesh_functions,
    set_hydro_vars, set_mesh_function_data, try_plan_standard_mesh, CellType,
    MeshFunctionConfig, MeshFunctionData, StarControls, StarJobControls,
    StarModel,
};

fn solar_zams_model(initial_mass: f64) -> StarModel {
    let controls = StarControls {
        initial_mass,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load builtin ZAMS model");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars on ZAMS");
    model
}

fn iso_idx(model: &StarModel, name: &str) -> usize {
    model
        .iso_names
        .iter()
        .position(|iso| iso == name)
        .expect("isotope in net")
}

fn total_species_mass(model: &StarModel, species_idx: usize) -> f64 {
    let sp = model.species;
    (0..model.nz)
        .map(|k| model.xa[k * sp + species_idx] * model.dq[k])
        .sum()
}

/// Verify `StarModel::set_m_grav_and_grav` reproduces upstream
/// `star/private/star_utils.f90:387-410` both with and without
/// `use_mass_corrections`.
#[test]
fn test_set_m_grav_and_grav_respects_use_mass_corrections() {
    let mut empty = StarModel::new(
        0,
        Vec::new(),
        StarJobControls::default(),
        StarControls::default(),
    );
    empty.set_m_grav_and_grav();

    let nz = 6_usize;
    let mut model = StarModel::new(
        nz,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    model.m_center = 1.4 * M_SUN;
    model.xmstar = 1.0e-4 * M_SUN;
    model.r_center = 1.0e6;
    model.dq = vec![1.0 / (nz as f64); nz];
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    for k in 0..nz {
        model.ln_r[k] = log(1.0e7 - (k as f64) * 5.0e5);
        model.cgrav[k] = STANDARD_CGRAV * (1.1 + 0.02 * (k as f64));
        model.aux.mass_correction[k] = 0.998 + 0.001 * (k as f64);
    }

    model.controls.use_mass_corrections = false;
    model.set_m_grav_and_grav();
    for k in 0..nz {
        assert_eq!(model.m_grav[k].to_bits(), model.m[k].to_bits());
        let expected_grav =
            model.cgrav[k] * model.m[k] / (model.r[k] * model.r[k]);
        assert!(
            (model.grav[k] - expected_grav).abs() <= 1.0e-15 * expected_grav
        );
    }

    model.controls.use_mass_corrections = true;
    model.set_m_grav_and_grav();
    let mut expected_mg = vec![0.0_f64; nz];
    expected_mg[nz - 1] =
        model.m_center + model.dm[nz - 1] * model.aux.mass_correction[nz - 1];
    for k in (0..(nz - 1)).rev() {
        expected_mg[k] =
            expected_mg[k + 1] + model.dm[k] * model.aux.mass_correction[k];
    }
    for k in 0..nz {
        assert_eq!(model.m_grav[k].to_bits(), expected_mg[k].to_bits());
        let expected_grav =
            model.cgrav[k] * expected_mg[k] / (model.r[k] * model.r[k]);
        assert!(
            (model.grav[k] - expected_grav).abs() <= 1.0e-15 * expected_grav
        );
    }
}

/// Verify `do_adjust_mesh` leaves `s% cgrav(1:nz) = standard_cgrav` on both
/// net merges (`nz_new < nz_old`) and net splits (`nz_new > nz_old`), as
/// upstream's `resize_star_info_arrays` (`star/private/adjust_mesh.f90:319`)
/// does through `do_copy_pointers_and_resize`
/// (`star/private/alloc.f90:965-969`), even when the old mesh carried a
/// zone-dependent `other_cgrav`-style profile.
#[test]
fn test_do_adjust_mesh_resets_cgrav_to_standard_cgrav_on_merge_and_split() {
    let mut model = solar_zams_model(1.0);
    let c2 = CLIGHT * CLIGHT;
    let assign_profile = |m: &mut StarModel| {
        for k in 0..m.nz {
            let f3 = 1.0 - (2.0 * STANDARD_CGRAV * m.m[k]) / (m.r[k] * c2);
            m.cgrav[k] = STANDARD_CGRAV / f3 + (k as f64) * 1.0e-11;
        }
    };
    let check_standard = |m: &StarModel, what: &str| {
        assert_eq!(m.cgrav.len(), m.nz);
        for k in 0..m.nz {
            assert_eq!(
                m.cgrav[k].to_bits(),
                STANDARD_CGRAV.to_bits(),
                "cgrav at {what} zone {k} is not standard_cgrav"
            );
        }
    };

    // Pass 1: coarsen ZAMS grid (`nz_new < nz_old`).
    let nz_old = model.nz;
    assign_profile(&mut model);
    model.controls.mesh_delta_coeff = 1.8;
    let cfg_merge = MeshFunctionConfig::from_controls(&model.controls);
    let plan_merge = try_plan_standard_mesh(&model, &cfg_merge)
        .expect("try_plan_standard_mesh merge")
        .expect("some merge plan");
    assert!(plan_merge.dq_new.len() < nz_old, "expected net merge");
    let _ =
        do_adjust_mesh(&mut model, &cfg_merge).expect("do_adjust_mesh merge");
    assert_eq!(model.nz, plan_merge.dq_new.len());
    check_standard(&model, "merged");

    // Pass 2: refine from the coarsened grid (`nz_new > nz_coarse`).
    let nz_coarse = model.nz;
    assign_profile(&mut model);
    model.controls.mesh_delta_coeff = 0.35;
    let cfg_split = MeshFunctionConfig::from_controls(&model.controls);
    let plan_split = try_plan_standard_mesh(&model, &cfg_split)
        .expect("try_plan_standard_mesh split")
        .expect("some split plan");
    assert!(plan_split.dq_new.len() > nz_coarse, "expected net split");
    let _ =
        do_adjust_mesh(&mut model, &cfg_split).expect("do_adjust_mesh split");
    assert_eq!(model.nz, plan_split.dq_new.len());
    check_standard(&model, "split");
    let sum_dq: f64 = model.dq.iter().sum();
    assert!((sum_dq - 1.0).abs() < 1.0e-14);
}

/// Verify `do_prune_mesh_surface` shifts `cgrav` and `aux.mass_correction`
/// for kept zones and updates `m_grav` using `use_mass_corrections`.
#[test]
fn test_do_prune_mesh_surface_shifts_cgrav_and_mass_correction() {
    let mut model = solar_zams_model(1.0);
    let nz_old = model.nz;
    let skip = 5_usize;
    let nz_new = nz_old - skip;
    model.controls.use_mass_corrections = true;
    for k in 0..nz_old {
        model.cgrav[k] = STANDARD_CGRAV * (1.0 + 1.0e-3 * (k as f64));
        model.aux.mass_correction[k] = 1.0 + 1.0e-4 * (k as f64);
    }
    let old_cgrav = model.cgrav.clone();
    let old_mc = model.aux.mass_correction.clone();

    do_prune_mesh_surface(&mut model, nz_new).expect("prune surface");
    assert_eq!(model.nz, nz_new);
    assert_eq!(model.cgrav.len(), nz_new);
    assert_eq!(model.aux.mass_correction.len(), nz_new);

    for k in 0..nz_new {
        assert_eq!(model.cgrav[k].to_bits(), old_cgrav[k + skip].to_bits());
        assert_eq!(
            model.aux.mass_correction[k].to_bits(),
            old_mc[k + skip].to_bits()
        );
    }
    let exp_mg_inner = model.m_center
        + model.dm[nz_new - 1] * model.aux.mass_correction[nz_new - 1];
    assert!(
        (model.m_grav[nz_new - 1] - exp_mg_inner).abs()
            <= 1.0e-15 * exp_mg_inner
    );
    let exp_mg_outer =
        model.m_grav[1] + model.dm[0] * model.aux.mass_correction[0];
    assert!((model.m_grav[0] - exp_mg_outer).abs() <= 1.0e-15 * exp_mg_outer);
    assert_eq!(
        model.dq[nz_new - 1].to_bits(),
        model.q[nz_new - 1].to_bits()
    );
    let sum_dq: f64 = model.dq.iter().sum();
    assert!((sum_dq - 1.0).abs() < 1.0e-12);
}

/// Verify exact MESA `26.4.1` formulas (`mesh_functions.f90`) for `P`, `T1`,
/// `T2`, `R`, `M`, and `xa` gradient functions across a ZAMS star.
#[test]
fn test_mesh_functions_exact_weights_p_t_r_m_xa() {
    let mut model = solar_zams_model(1.0);
    let nz = model.nz;
    let sp = model.species;
    let ih1 = iso_idx(&model, "h1");

    let mut mw = MixWindControls {
        p_function_weight: 25.0,
        t_function1_weight: 80.0,
        t_function2_weight: 20.0,
        t_function2_param: 1.0e4,
        r_function_weight: 10.0,
        r_function_param: 0.5,
        m_function_weight: 15.0,
        m_function_param: 0.25,
        gradt_function_weight: 0.0,
        omega_function_weight: 0.0,
        gam_function_weight: 0.0,
        ..MixWindControls::default()
    };
    mw.xa_function_species[0] = String::from("h1");
    mw.xa_function_weight[0] = 40.0;
    mw.xa_function_param[0] = 1.0e-3;
    for j in 1..mw.xa_function_weight.len() {
        mw.xa_function_weight[j] = 0.0;
    }
    model.controls.mix_wind = mw.clone();

    let nf = num_mesh_functions(&model).expect("num_mesh_functions");
    assert_eq!(nf, 6);
    let data =
        set_mesh_function_data(&model, nf).expect("set_mesh_function_data");
    let ginfo =
        get_gval_info_with_controls(&model, &mw, nf).expect("get_gval_info");
    assert_eq!(ginfo.fcns.names, data.names);

    for k in 0..nz {
        let exp_p = mw.p_function_weight * (model.ln_peos[k] / LN10);
        let t_k = model.t[k];
        let exp_t1 = mw.t_function1_weight * (model.ln_t[k] / LN10);
        let exp_t2 =
            mw.t_function2_weight * log10(t_k / (t_k + mw.t_function2_param));
        let exp_r = mw.r_function_weight
            * log10(1.0 + (model.r[k] / R_SUN) / mw.r_function_param);
        let exp_m = mw.m_function_weight
            * log10(
                1.0 + ((model.xmstar * model.q[k]) / M_SUN)
                    / mw.m_function_param,
            );
        let exp_x = mw.xa_function_weight[0]
            * log10(model.xa[k * sp + ih1] + mw.xa_function_param[0]);

        assert!(
            (data.column(0, nz)[k] - exp_p).abs() < 1.0e-12,
            "P gval mismatch at k={k}"
        );
        assert!(
            (data.column(1, nz)[k] - exp_t1).abs() < 1.0e-12,
            "T1 gval mismatch at k={k}"
        );
        assert!(
            (data.column(2, nz)[k] - exp_t2).abs() < 1.0e-12,
            "T2 gval mismatch at k={k}"
        );
        assert!(
            (data.column(3, nz)[k] - exp_r).abs() < 1.0e-12,
            "R gval mismatch at k={k}"
        );
        assert!(
            (data.column(4, nz)[k] - exp_m).abs() < 1.0e-12,
            "M gval mismatch at k={k}"
        );
        assert!(
            (data.column(5, nz)[k] - exp_x).abs() < 1.0e-12,
            "xa gval mismatch at k={k}"
        );
    }
}

/// Verify `merge_if_dlnR_too_small`, the `do1_dlog_eps_dlogp_coef` negative
/// `dlnP` sign-flip guard, and the `round_off_xq` / `mesh_max_allowed_ratio`
/// 1-ULP merge boundary match upstream MESA `26.4.1`.
#[test]
fn test_mesh_plan_dlnr_forced_merge_dlog_eps_sign_guard_and_roundoff() {
    // 1. `merge_if_dlnR_too_small` force-merge vs resolved grid.
    let nz = 12_usize;
    let mut tiny_model = StarModel::new(
        nz,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    tiny_model.m_center = 1.4 * M_SUN;
    tiny_model.xmstar = 1.0e-11 * M_SUN;
    tiny_model.r_center = 1.0e6;
    let dqs = vec![
        0.02, 0.03, 0.04, 0.03, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.15, 0.10,
    ];
    tiny_model.dq = dqs.clone();
    tiny_model.set_qs();
    tiny_model.set_xqs();
    tiny_model.set_m_and_dm();
    // Choose gval gradient ~0.65/cell so adjacent cells do not split (<1)
    // and do not merge normally (>1), isolating the `dlnR < mesh_min_dlnr`
    // forced-merge trigger across zones k=3,4.
    for k in 0..nz {
        tiny_model.ln_peos[k] = 20.0 + 0.01 * (k as f64);
        tiny_model.peos[k] = exp(tiny_model.ln_peos[k]);
        tiny_model.ln_t[k] = 16.0 + 0.005 * (k as f64);
        tiny_model.t[k] = exp(tiny_model.ln_t[k]);
        tiny_model.csound[k] = 1.0e7;
        tiny_model.mixing_type[k] = MixingType::NoMixing;
        tiny_model.xa[k * 2] = 0.7;
        tiny_model.xa[k * 2 + 1] = 0.3;
    }
    let ln_r_vals = [
        15.0,
        14.99,
        14.98,
        14.97,
        14.969_999_999_9,
        14.90,
        14.80,
        14.70,
        14.60,
        14.50,
        14.40,
        14.30,
    ];
    for k in 0..nz {
        tiny_model.ln_r[k] = ln_r_vals[k];
        tiny_model.r[k] = exp(ln_r_vals[k]);
    }
    tiny_model.controls.mix_wind.p_function_weight = 150.0;
    tiny_model.controls.mix_wind.t_function1_weight = 0.0;
    tiny_model.controls.mix_wind.xa_function_weight.fill(0.0);
    tiny_model.controls.mix_wind.max_surface_cell_dq = 0.05;
    tiny_model.controls.mix_wind.max_center_cell_dq = 0.25;
    tiny_model.controls.mix_wind.max_dq = 0.25;
    tiny_model.controls.mix_wind.mesh_max_allowed_ratio = 2.5;
    tiny_model.controls.mix_wind.mesh_min_dlnr = 1.0e-7;
    tiny_model.controls.mix_wind.mesh_dlog_pp_dlogp_extra = 1.0;

    tiny_model.controls.mix_wind.merge_if_dlnr_too_small = false;
    let cfg_off = MeshFunctionConfig::from_controls(&tiny_model.controls);
    let plan_off = try_plan_standard_mesh(&tiny_model, &cfg_off)
        .expect("plan with merge_if_dlnr_too_small = false")
        .expect("some plan");
    assert_eq!(plan_off.num_merges, 0);

    tiny_model.controls.mix_wind.merge_if_dlnr_too_small = true;
    let cfg = MeshFunctionConfig::from_controls(&tiny_model.controls);
    let plan_merged = try_plan_standard_mesh(&tiny_model, &cfg)
        .expect("plan with tiny dlnR")
        .expect("some plan");
    assert!(
        plan_merged.num_merges >= 1 && plan_merged.dq_new.len() < nz,
        "expected force-merge for dlnR < mesh_min_dlnr: merges={}, nz_new={}, cell_type={:?}",
        plan_merged.num_merges,
        plan_merged.dq_new.len(),
        plan_merged.cell_type
    );

    // 2. Negative dlnP (`1`-ULP density/radius inversion) suppresses extra
    // burning-gradient refinement (`adjust_mesh_support.f90:211-241`).
    let mut burn_model = StarModel::new(
        4,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    for k in 0..4 {
        burn_model.xa[k * 2] = 0.7;
        burn_model.xa[k * 2 + 1] = 0.3;
    }
    burn_model.aux.eps_nuc_categories = vec![[0.0; NUM_CATEGORIES]; 4];
    let i_pp = category_id("pp") - 1;
    burn_model.aux.eps_nuc_categories[0][i_pp] = 1.0e8;
    burn_model.aux.eps_nuc_categories[1][i_pp] = 1.0e2;
    burn_model.aux.eps_nuc_categories[2][i_pp] = 1.0e2;
    burn_model.aux.eps_nuc_categories[3][i_pp] = 1.0e2;
    let mw_burn = MixWindControls {
        mesh_dlog_eps_min_for_extra: 2.0,
        mesh_dlog_eps_dlogp_full_off: 0.5,
        mesh_dlog_eps_dlogp_full_on: 1.0,
        mesh_delta_coeff_factor_smooth_iters: 0,
        mesh_dlog_pp_dlogp_extra: 0.25,
        ..MixWindControls::default()
    };
    burn_model.ln_peos = vec![30.0, 30.0 + 6.0e-8, 30.5, 31.0];
    let dg_pos =
        set_delta_gval_max_with_controls(&burn_model, &mw_burn, &burn_model.xa)
            .expect("positive dlnP");
    assert!(
        dg_pos[1] < 0.30,
        "positive dlnP must activate pp gradient refinement: {dg_pos:?}"
    );
    burn_model.ln_peos = vec![30.0, 30.0 - 1.0e-8, 30.5, 31.0];
    let dg_neg =
        set_delta_gval_max_with_controls(&burn_model, &mw_burn, &burn_model.xa)
            .expect("negative dlnP");
    assert!(
        (dg_neg[0] - 1.0).abs() < 1.0e-12 && (dg_neg[1] - 1.0).abs() < 1.0e-12,
        "negative dlnP must return 1.0 without refinement: {dg_neg:?}"
    );

    // 3. `round_off_xq` 1-ULP boundary around `xq_cand == xq_old(5)`.
    let mut rk_model = StarModel::new(
        10,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    rk_model.m_center = 0.0;
    rk_model.xmstar = M_SUN;
    rk_model.r_center = 1.0;
    for k in 0..10 {
        rk_model.ln_peos[k] = 20.0 + 0.1 * (k as f64);
        rk_model.peos[k] = exp(rk_model.ln_peos[k]);
        rk_model.ln_t[k] = 16.0 + 1.0e-6 * (k as f64);
        rk_model.t[k] = exp(rk_model.ln_t[k]);
        rk_model.ln_r[k] = 24.0 - 0.1 * (k as f64);
        rk_model.r[k] = exp(rk_model.ln_r[k]);
        rk_model.csound[k] = 1.0e7;
        if k < 2 {
            rk_model.xa[k * 2] = 0.7;
            rk_model.xa[k * 2 + 1] = 0.3;
        } else {
            rk_model.xa[k * 2] = 0.3;
            rk_model.xa[k * 2 + 1] = 0.7;
        }
    }
    rk_model.controls.mix_wind.mesh_max_allowed_ratio = 2.5;
    rk_model.controls.mix_wind.max_num_merge_cells = 2;
    rk_model.controls.mix_wind.max_num_merge_surface_cells = 2;
    rk_model.controls.mix_wind.max_surface_cell_dq = 1.0e-12;
    rk_model.controls.mix_wind.max_center_cell_dq = 0.99;
    rk_model.controls.mix_wind.max_dq = 1.0;
    rk_model.controls.mix_wind.mesh_min_dlnr = 0.0;
    rk_model.controls.mix_wind.merge_if_dlnr_too_small = false;

    let fcns_rk = MeshFunctionData {
        names: vec![String::from("P_function")],
        is_xa_function: vec![false],
        is_logt_function: vec![false],
        vals: vec![0.0, 0.75, 1.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    };
    let delta_gval_max_rk = vec![1.0_f64; 10];
    let do_not_split_rk = vec![false; 10];

    let base_bits = 1.0e-12_f64.to_bits();
    let dq0_merge = f64::from_bits(base_bits - 7);
    let dq0_nomerge = f64::from_bits(base_bits - 8);
    let dq1 = 2.13e-12_f64;
    let dq2 = 3.07e-12_f64;
    let dq3 = 1.23e-12_f64;
    let dq4 = 2.00e-12_f64;
    for (dq0, expect_merged) in [(dq0_merge, true), (dq0_nomerge, false)] {
        let mut dq_vec = vec![
            dq0, dq1, dq2, dq3, dq4, 4.0e-12, 8.0e-12, 1.6e-11, 3.2e-11, 0.0,
        ];
        dq_vec[9] = 1.0 - dq_vec[..9].iter().sum::<f64>();
        rk_model.dq = dq_vec;
        rk_model.set_qs();
        rk_model.set_xqs();
        rk_model.set_m_and_dm();
        let plan_rk = do_mesh_plan(
            &rk_model,
            &rk_model.controls.mix_wind,
            &rk_model.xq,
            &rk_model.dq,
            1.0e-14,
            1.0,
            1.0e-14,
            2.5,
            &do_not_split_rk,
            &fcns_rk,
            &delta_gval_max_rk,
            0.99,
            1.0e-12,
            1.0e-99,
        )
        .expect("round_off_xq do_mesh_plan");
        let (cell_type_rk, _, _, _) = classify_new_cells(
            &rk_model.xq,
            &plan_rk.xq_new,
            &plan_rk.comes_from,
        )
        .expect("classify_new_cells");
        assert_eq!(cell_type_rk[0], CellType::Unchanged);
        assert_eq!(cell_type_rk[1], CellType::Unchanged);
        assert_eq!(
            cell_type_rk[2] == CellType::Merged,
            expect_merged,
            "expected merged={expect_merged} for dq0={dq0:.17e}, cell_type={cell_type_rk:?}, dq_new={:?}",
            plan_rk.dq_new
        );
        let sum_rk: f64 = plan_rk.dq_new.iter().sum();
        assert!((sum_rk - 1.0).abs() < 1.0e-14);
    }
}

/// Verify species mass conservation, internal energy balance, and strict
/// volume monotonicity across chained split and merge `do_adjust_mesh` calls.
#[test]
fn test_remap_conservation_species_energy_and_volume() {
    let mut model = solar_zams_model(1.3);
    let ih1 = iso_idx(&model, "h1");
    let ihe4 = iso_idx(&model, "he4");
    let h1_before = total_species_mass(&model, ih1);
    let he4_before = total_species_mass(&model, ihe4);
    let nz_zams = model.nz;

    model
        .controls
        .mix_wind
        .max_rel_delta_ie_for_mesh_total_energy_balance = 0.0;
    model.controls.mesh_delta_coeff = 1.6;
    let cfg_merge = MeshFunctionConfig::from_controls(&model.controls);
    let (_, nm1) =
        do_adjust_mesh(&mut model, &cfg_merge).expect("merge remesh");
    let nz_coarse = model.nz;
    assert!(nm1 > 0 && nz_coarse < nz_zams);
    assert!(
        model.aux.mesh_adjust_ie_conservation < 1.0e-12,
        "merge ie_err={}",
        model.aux.mesh_adjust_ie_conservation
    );

    model.controls.mesh_delta_coeff = 0.35;
    let cfg_split = MeshFunctionConfig::from_controls(&model.controls);
    let (ns2, _) =
        do_adjust_mesh(&mut model, &cfg_split).expect("split remesh");
    assert!(ns2 > 0 && model.nz > nz_coarse);
    assert!(
        model.aux.mesh_adjust_ie_conservation < 1.0e-12,
        "split ie_err={}",
        model.aux.mesh_adjust_ie_conservation
    );

    model
        .controls
        .mix_wind
        .max_rel_delta_ie_for_mesh_total_energy_balance = 0.05;
    model.controls.mesh_delta_coeff = 1.2;
    let cfg_bal = MeshFunctionConfig::from_controls(&model.controls);
    let _ = do_adjust_mesh(&mut model, &cfg_bal).expect("balanced remesh");
    assert!(
        model.aux.mesh_adjust_ie_conservation < 1.0e-4,
        "balanced ie_err={}",
        model.aux.mesh_adjust_ie_conservation
    );

    let h1_after = total_species_mass(&model, ih1);
    let he4_after = total_species_mass(&model, ihe4);
    assert!((h1_after - h1_before).abs() <= 1.0e-12);
    assert!((he4_after - he4_before).abs() <= 1.0e-12);

    let sum_dq: f64 = model.dq.iter().sum();
    assert!((sum_dq - 1.0).abs() < 1.0e-14);
    for k in 1..model.nz {
        assert!(model.xq[k] > model.xq[k - 1]);
        assert!(model.ln_r[k - 1] > model.ln_r[k]);
    }
}
