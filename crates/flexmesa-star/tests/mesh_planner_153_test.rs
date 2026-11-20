// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit and integration tests for the MESA `26.4.1` adaptive mesh planner,
//! mesh functions, multipliers, conservative remapping, and `split_merge_amr`
//! (`#153`, `#241`, `#242`, `#243`, `#244`, `#250`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{LN10, M_SUN, MixingType, SECYER};
use flexmesa_star::{
    InlistBundle, InlistLoadOptions, MeshFunctionConfig,
    OtherMeshDeltaCoeffFactorHandle, OtherMeshDeltaCoeffFactorHook,
    OtherMeshFunctionColumn, OtherMeshFunctionsHandle, OtherMeshFunctionsHook,
    StarControls, StarError, StarJobControls, StarModel, do_adjust_mesh,
    do_prune_mesh_surface, eval_effective_mesh_delta_coeff, get_gval_info,
    load_zams_model, num_mesh_functions, read_mod_str, remesh_split_merge,
    set_hydro_vars, set_mesh_function_data, star_evolve_step,
};

fn star_suite_dir() -> PathBuf {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home =
                std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    base.join("star").join("test_suite")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn read_reference_num_zones(history_path: &Path) -> Vec<(usize, usize)> {
    let text = std::fs::read_to_string(history_path)
        .expect("read reference history.data");
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.len() >= 7, "expected MESA history header and rows");
    let col_names: Vec<&str> = lines[5].split_whitespace().collect();
    let idx_mod = col_names
        .iter()
        .position(|&c| c == "model_number")
        .expect("model_number column");
    let idx_nz = col_names
        .iter()
        .position(|&c| c == "num_zones")
        .expect("num_zones column");
    let mut out = Vec::new();
    for line in &lines[6..] {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() > idx_mod.max(idx_nz) {
            let m: usize = toks[idx_mod].parse().expect("parse model_number");
            let nz: usize = toks[idx_nz].parse().expect("parse num_zones");
            out.push((m, nz));
        }
    }
    out
}

fn species_masses(model: &StarModel) -> Vec<f64> {
    let ns = model.species;
    let mut out = vec![0.0_f64; ns];
    for k in 0..model.nz {
        for j in 0..ns {
            out[j] += model.dm[k] * model.xa[k * ns + j];
        }
    }
    out
}

fn total_angular_momentum_cgs(model: &StarModel) -> f64 {
    let nz = model.nz;
    let mut sum = 0.0_f64;
    for k in 0..nz {
        let dmbar = if k == 0 {
            0.5 * model.dm[0]
        } else if k + 1 < nz {
            f64::midpoint(model.dm[k - 1], model.dm[k])
        } else {
            model.dm[nz - 1] + 0.5 * model.dm[nz - 2]
        };
        sum += dmbar * model.j_rot[k];
    }
    sum
}

#[test]
fn test_species_energy_and_untouched_zone_angular_momentum_are_conserved_f4_26()
{
    let mut ctrl = StarControls {
        initial_mass: 2.0,
        initial_z: 0.02,
        zbase: 0.02,
        rotation_flag: true,
        ..StarControls::default()
    };
    // Use `mesh_ok_to_merge = false` with low `P`/`T` weights and a localized
    // interior `he4` gradient (`k >= 1200`) at high `he4` weight so outer zones
    // `0..800` are genuinely untouched (`CellType::Unchanged`) while deep core
    // zones split (`F4-26`, `mesh_adjust.f90:1927-2112`).
    ctrl.mix_wind.mesh_ok_to_merge = false;
    ctrl.mix_wind.max_surface_cell_dq = 1.0;
    ctrl.mix_wind.p_function_weight = 1.0;
    ctrl.mix_wind.t_function1_weight = 1.0;
    ctrl.mix_wind.xa_function_weight[0] = 2000.0;
    ctrl.mix_wind.max_dq = 0.2;

    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 2 Msun ZAMS");
    let cat = NuclideCatalog::builtin_ref();
    let cid_h1 = cat.get_by_name("h1").expect("h1").chem_id;
    let cid_he4 = cat.get_by_name("he4").expect("he4").chem_id;
    let j_h1 = model
        .chem_id
        .iter()
        .position(|&c| c == cid_h1)
        .expect("h1 idx");
    let j_he4 = model
        .chem_id
        .iter()
        .position(|&c| c == cid_he4)
        .expect("he4 idx");
    let ns = model.species;
    for k in 1200..model.nz {
        let frac = ((k - 1200) as f64) / ((model.nz - 1200) as f64);
        let d_he = 0.25 * frac;
        model.xa[k * ns + j_h1] -= d_he;
        model.xa[k * ns + j_he4] += d_he;
    }
    for k in 0..model.nz {
        let q_k = model.q[k];
        let omega_k = 1.0e-5 * (1.0 + 2.5 * (1.0 - q_k));
        let i_k = ((2.0 / 3.0) * model.r[k] * model.r[k]).max(1.0e-99);
        model.omega[k] = omega_k;
        model.aux.i_rot[k] = i_k;
        model.j_rot[k] = omega_k * i_k;
    }
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let omega_before = model.omega.clone();
    let j_rot_before = model.j_rot.clone();
    let sp_before = species_masses(&model);
    let ie_before: f64 = model
        .dm
        .iter()
        .zip(&model.energy)
        .map(|(dm, e)| dm * e)
        .sum();
    let j_before = total_angular_momentum_cgs(&model);

    let cfg = MeshFunctionConfig::from_controls(&model.controls);
    let (splits, merges) =
        do_adjust_mesh(&mut model, &cfg).expect("interior band remesh");
    assert!(
        splits + merges > 0,
        "expected interior band splits or merges"
    );

    // 1. Species mass conservation across remesh.
    let sp_after = species_masses(&model);
    for (a, b) in sp_after.iter().zip(&sp_before) {
        if *b > 1.0e-12 * M_SUN {
            assert!(
                ((a - b) / b).abs() < 1.0e-13,
                "species mass drifted: {a} vs {b}"
            );
        }
    }

    // 2. Total internal energy balance bounded by `max_rel_delta_IE_for_mesh_total_energy_balance`.
    let ie_after: f64 = model
        .dm
        .iter()
        .zip(&model.energy)
        .map(|(dm, e)| dm * e)
        .sum();
    let rel_ie = (ie_after - ie_before).abs() / ie_before.abs();
    assert!(
        rel_ie
            <= model
                .controls
                .mix_wind
                .max_rel_delta_ie_for_mesh_total_energy_balance,
        "relative IE change {rel_ie:.3e} exceeded upstream cap"
    );

    // 3. Global angular momentum conservation AND exact bit-identity (`0 ULP`)
    //    on untouched outer zones (`0..150`) (`F4-26`, `mesh_adjust.f90:1994-1998`).
    let j_after = total_angular_momentum_cgs(&model);
    let rel_j = (j_after - j_before).abs() / j_before.abs();
    assert!(
        rel_j < 1.0e-13,
        "total angular momentum drifted: rel_j={rel_j:.3e}"
    );
    for k in 30..150 {
        assert_eq!(
            model.omega[k].to_bits(),
            omega_before[k].to_bits(),
            "untouched zone {k} omega was rescaled instead of preserved bit-for-bit"
        );
        assert_eq!(
            model.j_rot[k].to_bits(),
            j_rot_before[k].to_bits(),
            "untouched zone {k} j_rot was rescaled instead of preserved bit-for-bit"
        );
    }

    // 4. Degenerate `eta >= -1e-6` branch skips `solve_eos_given_DE` (`mesh_adjust.f90:1767`).
    let mut m_deg = model.clone();
    m_deg.eta.fill(5.0);
    m_deg.controls.mix_wind.p_function_weight = 40.0;
    m_deg.controls.mix_wind.t_function1_weight = 110.0;
    m_deg.controls.mix_wind.max_dq = 0.01;
    let cfg_deg = MeshFunctionConfig::from_controls(&m_deg.controls);
    let _ =
        do_adjust_mesh(&mut m_deg, &cfg_deg).expect("degenerate eta remesh");
    assert!(m_deg.ln_t.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hydro_velocity_and_rsp2_turbulent_remaps_conserve_control_volumes() {
    let mut ctrl_v = StarControls {
        initial_mass: 3.0,
        initial_z: 0.02,
        zbase: 0.02,
        v_flag: true,
        ..StarControls::default()
    };
    ctrl_v.mix_wind.p_function_weight = 300.0;
    let mut mv = load_zams_model(ctrl_v, StarJobControls::default())
        .expect("load 3 Msun ZAMS v_flag");
    set_hydro_vars(&mut mv, false, false).expect("set_hydro_vars v_flag");
    for k in 0..mv.nz {
        mv.v[k] = 1.0e5 * (1.0 - mv.q[k]);
    }
    let cfg_v = MeshFunctionConfig::from_controls(&mv.controls);
    let (sv, mgv) = do_adjust_mesh(&mut mv, &cfg_v).expect("remesh v_flag");
    assert!(sv + mgv > 0);
    assert!(mv.v.iter().all(|v| v.is_finite()));
    assert!(
        mv.aux.mesh_adjust_ke_conservation < 1.0e-12,
        "do_v KE conservation error={}",
        mv.aux.mesh_adjust_ke_conservation
    );

    let mut ctrl_u = StarControls {
        initial_mass: 3.0,
        initial_z: 0.02,
        zbase: 0.02,
        u_flag: true,
        rsp2_flag: true,
        ..StarControls::default()
    };
    ctrl_u.mix_wind.p_function_weight = 300.0;
    let mut mu = load_zams_model(ctrl_u, StarJobControls::default())
        .expect("load 3 Msun ZAMS u_flag + rsp2_flag");
    set_hydro_vars(&mut mu, false, false).expect("set_hydro_vars u_flag");
    mu.aux.rsp.et_rsp = vec![1.0e10; mu.nz];
    mu.aux.rsp.hp_face = vec![1.0e9; mu.nz];
    for k in 0..mu.nz {
        mu.u[k] = -2.0e5 * mv.q[k.min(mv.nz - 1)];
    }
    let cfg_u = MeshFunctionConfig::from_controls(&mu.controls);
    let (su, mgu) =
        do_adjust_mesh(&mut mu, &cfg_u).expect("remesh u_flag + rsp2");
    assert!(su + mgu > 0);
    assert!(mu.u.iter().all(|u| u.is_finite()));
    assert_eq!(mu.aux.rsp.et_rsp.len(), mu.nz);
    assert_eq!(mu.aux.rsp.hp_face.len(), mu.nz);
}

struct MockOtherMeshFcns;
impl OtherMeshFunctionsHook for MockOtherMeshFcns {
    fn eval_other_mesh_functions(
        &self,
        model: &StarModel,
    ) -> Result<Vec<OtherMeshFunctionColumn>, StarError> {
        Ok(vec![OtherMeshFunctionColumn {
            name: String::from("custom_q_fcn"),
            is_xa_function: false,
            vals: model.q.iter().map(|&q| 25.0 * q).collect(),
        }])
    }
}

struct MockOtherDeltaFactor;
impl OtherMeshDeltaCoeffFactorHook for MockOtherDeltaFactor {
    fn eval_mesh_delta_coeff_factor(
        &self,
        model: &StarModel,
        mesh_delta_coeff_factor: &mut [f64],
    ) -> Result<(), StarError> {
        assert_eq!(model.nz, mesh_delta_coeff_factor.len());
        for f in mesh_delta_coeff_factor.iter_mut() {
            *f *= 0.5;
        }
        Ok(())
    }
}

#[test]
fn test_all_mesh_functions_multipliers_hooks_and_deferred_bugs_241_243_250() {
    let cat = NuclideCatalog::builtin_ref();
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            zbase: 0.02,
            rotation_flag: true,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.dt = 1.0e4 * SECYER;

    // 1. Default active mesh functions: `P_function`, `T_function1`, and `xa_function_1` (`he4` at weight 30).
    let n_def = num_mesh_functions(&model).expect("default num_mesh_functions");
    assert_eq!(n_def, 3, "upstream defaults activate P, T1, and he4 only");
    let data_def =
        set_mesh_function_data(&model, n_def).expect("default mesh data");
    assert_eq!(
        data_def.names,
        vec![
            String::from("P_function"),
            String::from("T_function1"),
            String::from("he4"),
        ]
    );
    assert_eq!(data_def.is_xa_function, vec![false, false, true]);
    assert_eq!(data_def.is_logt_function, vec![false, true, false]);

    // Perturbing `h1` and `o16` (keeping `he4` fixed) leaves `xa_function_1` unchanged.
    let mut m_pert = model.clone();
    let cid_h1 = cat.get_by_name("h1").expect("h1").chem_id;
    let cid_o16 = cat.get_by_name("o16").expect("o16").chem_id;
    let idx_h1 = m_pert
        .chem_id
        .iter()
        .position(|&id| id == cid_h1)
        .expect("h1 in basic.net");
    let idx_o16 = m_pert
        .chem_id
        .iter()
        .position(|&id| id == cid_o16)
        .expect("o16 in basic.net");
    let ns = m_pert.species;
    let delta = 1.0e-4;
    m_pert.xa[10 * ns + idx_h1] -= delta;
    m_pert.xa[10 * ns + idx_o16] += delta;
    let data_pert =
        set_mesh_function_data(&m_pert, n_def).expect("perturbed mesh data");
    assert_eq!(
        data_pert.column(2, m_pert.nz)[10].to_bits(),
        data_def.column(2, model.nz)[10].to_bits(),
        "default he4-only xa_function must not depend on h1/o16"
    );

    // 2. Enable all 13 non-`convective_bdy` standard functions simultaneously,
    //    then test `convective_bdy` at `i <= 10` and `i = 15 > 10` (`#241`).
    {
        let mw = &mut model.controls.mix_wind;
        mw.e_function_weight = 10.0;
        mw.t_function2_weight = 10.0;
        mw.r_function_weight = 10.0;
        mw.r_function2_weight = 10.0;
        mw.r_function2_param1 = 0.8;
        mw.r_function2_param2 = 0.2;
        mw.r_function3_weight = 10.0;
        mw.m_function_weight = 15.0;
        mw.gradt_function_weight = 10.0;
        mw.log_tau_function_weight = 10.0;
        mw.log_kap_function_weight = 10.0;
        mw.omega_function_weight = 10.0;
        mw.gam_function_weight = 10.0;
    }
    for k in 0..model.nz {
        model.omega[k] = 1.0e-5 * (1.0 + model.q[k]);
    }
    let n_all = num_mesh_functions(&model).expect("all mesh functions count");
    assert_eq!(n_all, 14);
    let data_all =
        set_mesh_function_data(&model, n_all).expect("all mesh functions");
    assert!(data_all.vals.iter().all(|v| v.is_finite()));
    let m_idx = data_all
        .names
        .iter()
        .position(|s| s == "M_function")
        .expect("M_function present");
    let m_col = data_all.column(m_idx, model.nz);
    let m_param = model.controls.mix_wind.m_function_param;
    let expected_m0 = 15.0
        * flexmesa_math::log10(
            1.0 + ((model.xmstar * model.q[0]) / M_SUN) / m_param,
        );
    assert!((m_col[0] - expected_m0).abs() < 1.0e-12);

    // 3a. `convective_bdy` at column `i = 4 <= 10` succeeds and populates finite step values.
    let mut m_conv_ok = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 4 Msun ZAMS for conv_bdy");
    set_hydro_vars(&mut m_conv_ok, false, false).expect("set_hydro_vars");
    m_conv_ok.dt = 1.0e4 * SECYER;
    m_conv_ok.controls.mix_wind.convective_bdy_weight = 5.0;
    let n_conv_ok = num_mesh_functions(&m_conv_ok).expect("conv_bdy count");
    assert_eq!(n_conv_ok, 4);
    let data_conv_ok =
        set_mesh_function_data(&m_conv_ok, n_conv_ok).expect("conv_bdy <= 10");
    assert!(data_conv_ok.vals.iter().all(|v| v.is_finite()));

    // 3b. Verify `#241`: when `convective_bdy` lands at column `i = 14 > max(10, num_conv_boundaries)`,
    //     `set_mesh_function_data` raises loud `InvalidInput` citing `#241`.
    let mut m_many_fcns = model.clone();
    let _ = MixingType::NoMixing;
    m_many_fcns.controls.mix_wind.convective_bdy_weight = 5.0;
    let n_with_conv14 =
        num_mesh_functions(&m_many_fcns).expect("15 functions count");
    let err_241 = set_mesh_function_data(&m_many_fcns, n_with_conv14)
        .expect_err("expected #241 uninitialized top_conv_bdy error");
    assert!(
        matches!(err_241, StarError::InvalidInput(_))
            && err_241.to_string().contains("#241"),
        "expected #241 in error: {err_241}"
    );

    // 4. Verify `#243` (`do_mesh_dlogX_dlogP_coef` `* ln10`), `mesh_Pgas_div_P_exponent`,
    //    `dlog_eps_dlogP`, `highT` blending, and `xa_mesh_delta_coeff`.
    model.controls.mix_wind.convective_bdy_weight = 0.0;
    model.controls.mix_wind.mesh_pgas_div_p_exponent = 0.5;
    model.controls.mix_wind.mesh_logx_species[0] = String::from("h1");
    model.controls.mix_wind.mesh_dlogx_dlogp_extra[0] = 0.25;
    model.controls.mix_wind.mesh_dlogx_dlogp_full_on[0] = 0.01;
    model.controls.mix_wind.mesh_dlogx_dlogp_full_off[0] = 0.001;
    let n_sub = num_mesh_functions(&model).expect("n_sub");
    let ginfo = get_gval_info(&model, n_sub).expect("get_gval_info");
    assert!(ginfo.delta_gval_max.iter().all(|&d| d > 0.0 && d <= 1.0));

    // High-T blending and heaviest-isotope `xa_mesh_delta_coeff`.
    let mut m_ht = model.clone();
    let nz_ht = m_ht.nz;
    m_ht.ln_t[nz_ht - 1] = 9.25 * LN10;
    let heaviest_iso = m_ht
        .chem_id
        .iter()
        .filter_map(|&cid| cat.get_by_id(cid))
        .max_by(|a, b| a.w.total_cmp(&b.w))
        .expect("heaviest iso");
    m_ht.controls
        .mix_wind
        .xa_function_species
        .push(heaviest_iso.name.clone());
    m_ht.controls.mix_wind.xa_mesh_delta_coeff.push(0.6);
    let eff_coeff =
        eval_effective_mesh_delta_coeff(&m_ht, &m_ht.controls.mix_wind);
    // Midpoint between standard `1.0` (`logT <= 9.0`) and highT `3.0` (`logT >= 9.5`) is `2.0`, times `0.6` = `1.2`.
    assert!((eff_coeff - 1.2).abs() < 1.0e-12);

    // 5. Custom hooks (`use_other_mesh_functions` & `use_other_mesh_delta_coeff_factor`):
    //    fail loudly when enabled without a hook, succeed when attached.
    let mut m_hook = model.clone();
    m_hook.controls.mix_wind.use_other_mesh_functions = true;
    assert!(num_mesh_functions(&m_hook).is_err());
    m_hook.aux.other_mesh_functions =
        Some(OtherMeshFunctionsHandle(Arc::new(MockOtherMeshFcns)));
    let n_with_hook = num_mesh_functions(&m_hook).expect("hook count");
    assert_eq!(n_with_hook, n_sub + 1);

    m_hook.controls.mix_wind.use_other_mesh_delta_coeff_factor = true;
    assert!(get_gval_info(&m_hook, n_with_hook).is_err());
    m_hook.aux.other_mesh_delta_coeff_factor = Some(
        OtherMeshDeltaCoeffFactorHandle(Arc::new(MockOtherDeltaFactor)),
    );
    let ginfo_hook =
        get_gval_info(&m_hook, n_with_hook).expect("hook get_gval_info");
    assert!(ginfo_hook.fcns.names.iter().any(|s| s == "custom_q_fcn"));

    // 6. Verify `#250`: `mesh_max_allowed_ratio < 2.5` mutates control to `2.5` after first `do_adjust_mesh`.
    let mut m_250 = load_zams_model(
        StarControls {
            initial_mass: 2.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 2 Msun ZAMS for #250");
    set_hydro_vars(&mut m_250, false, false).expect("set_hydro_vars #250");
    m_250.controls.mix_wind.mesh_max_allowed_ratio = 2.0;
    let cfg_250 = MeshFunctionConfig::from_controls(&m_250.controls);
    let _ = do_adjust_mesh(&mut m_250, &cfg_250).expect("first remesh #250");
    assert_eq!(m_250.controls.mix_wind.mesh_max_allowed_ratio, 2.5);

    // 7. Verify `do_prune_mesh_surface` strips outer cells while preserving interior bit-for-bit.
    let nz_before_prune = m_250.nz;
    let center_lnt_bits = m_250.ln_t[nz_before_prune - 1].to_bits();
    let keep = nz_before_prune - 5;
    do_prune_mesh_surface(&mut m_250, keep).expect("prune 5 surface cells");
    assert_eq!(m_250.nz, keep);
    assert_eq!(m_250.ln_t[keep - 1].to_bits(), center_lnt_bits);
    assert!((m_250.dq.iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
}

#[test]
fn test_split_merge_amr_zoning_modes_conserve_invariants_and_reject_242() {
    let base = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            zbase: 0.02,
            rotation_flag: true,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 4 Msun ZAMS for split_merge_amr");

    // 1. `#242`: `split_merge_amr_flipped_hybrid_zoning = true` returns explicit `InvalidInput` citing `#242`.
    let mut m_flip = base.clone();
    set_hydro_vars(&mut m_flip, false, false).expect("set_hydro_vars");
    m_flip.controls.mix_wind.use_split_merge_amr = true;
    m_flip
        .controls
        .mix_wind
        .split_merge_amr_flipped_hybrid_zoning = true;
    let err_242 = remesh_split_merge(&mut m_flip)
        .expect_err("expected #242 error on flipped_hybrid_zoning");
    assert!(
        matches!(err_242, StarError::InvalidInput(_))
            && err_242.to_string().contains("#242"),
        "expected #242 in error: {err_242}"
    );

    // 2. Exercise `log_zoning`, `logtau_zoning`, `hybrid_zoning`, and linear `r` zoning.
    for (log_z, logtau_z, hybrid_z) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (false, false, false),
    ] {
        let mut m = base.clone();
        set_hydro_vars(&mut m, false, false).expect("set_hydro_vars");
        for k in 0..m.nz {
            let i_k = ((2.0 / 3.0) * m.r[k] * m.r[k]).max(1.0e-99);
            m.omega[k] = 2.0e-5;
            m.aux.i_rot[k] = i_k;
            m.j_rot[k] = m.omega[k] * i_k;
        }
        m.controls.mix_wind.use_split_merge_amr = true;
        m.controls.mix_wind.split_merge_amr_nz_baseline = 600;
        if hybrid_z {
            m.controls.mix_wind.split_merge_amr_nz_r_core = 150;
            m.controls.mix_wind.split_merge_amr_r_core_cm = 0.25 * m.r[0];
        }
        m.controls.mix_wind.split_merge_amr_max_iters = 40;
        m.controls.mix_wind.split_merge_amr_log_zoning = log_z;
        m.controls.mix_wind.split_merge_amr_logtau_zoning = logtau_z;
        m.controls.mix_wind.split_merge_amr_hybrid_zoning = hybrid_z;

        let sp0 = species_masses(&m);
        let j0 = total_angular_momentum_cgs(&m);
        let cfg = MeshFunctionConfig::from_controls(&m.controls);
        let (splits, merges) =
            do_adjust_mesh(&mut m, &cfg).expect("split_merge_amr remesh");
        assert!(splits + merges > 0);
        assert!((m.dq.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);

        let sp1 = species_masses(&m);
        // Upstream `split1_non_negative` (`adjust_mesh_split_merge.f90:1203-1220`)
        // weights subcell abundances by volume `(dVL, dVR)` while `dm` uses linear
        // `grad_rho`, giving second-order `O((grad_rho*dr/rho)*(grad_xa*dr)) ~ 1e-11`.
        for (a, b) in sp1.iter().zip(&sp0) {
            if *b > 1.0e-12 * M_SUN {
                assert!(
                    ((a - b) / b).abs() < 5.0e-11,
                    "split_merge_amr species drift: {a} vs {b}"
                );
            }
        }
        let j1 = total_angular_momentum_cgs(&m);
        assert!(
            ((j1 - j0) / j0).abs() < 1.0e-11,
            "split_merge_amr J_tot drift: {j1} vs {j0}"
        );
    }
}

#[test]
fn test_num_zones_tracks_mesa_26_4_1_across_star_test_hb_2m_and_1p3m_high_z() {
    let root = repo_root();

    // 1. `star/test` (`4 Msun, Z = 0.02` ZAMS, `mesh_delta_coeff = 1.0`):
    //    MESA `26.4.1` `star/test/LOGS/history.data` transitions `2464 -> 992` zones by step 5.
    let star_test_ref = read_reference_num_zones(
        &root
            .join("tests")
            .join("fixtures")
            .join("golden")
            .join("star_test")
            .join("history.data"),
    );
    assert_eq!(star_test_ref[0], (1, 2464));
    assert_eq!(star_test_ref[1], (5, 992));

    let mut m_st = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut m_st, false, false).expect("set_hydro_vars");
    assert_eq!(m_st.nz, 2464);
    let cfg_st = MeshFunctionConfig::from_controls(&m_st.controls);
    let mut st_nz_seq = Vec::new();
    for _ in 0..4 {
        let _ = do_adjust_mesh(&mut m_st, &cfg_st).expect("remesh star/test");
        st_nz_seq.push(m_st.nz);
    }
    println!(
        "star/test 4-pass remesh num_zones: {st_nz_seq:?} (MESA 26.4.1: [1556, 1124, 999, 992])"
    );
    assert_eq!(st_nz_seq[2], 999);
    assert_eq!(m_st.nz, 992);

    // 2. `hb_2M` (`standard_2.0_ZACHeB.mod`, `mesh_delta_coeff = 0.5`):
    //    MESA `26.4.1` starts at `1346` zones on step 1 and refines to `2629` -> `2569` -> `2566`.
    let hb_ref = read_reference_num_zones(
        &root
            .join("benchmarks")
            .join("history_harness")
            .join("reference_histories")
            .join("star")
            .join("hb_2M.data"),
    );
    assert_eq!(hb_ref[0], (1, 1346));
    assert_eq!(hb_ref[1], (2, 2629));

    let suite = star_suite_dir();
    let hb_dir = suite.join("hb_2M");
    assert!(
        hb_dir.is_dir(),
        "missing hb_2M test_suite inputs at {}",
        hb_dir.display()
    );
    let hb_bundle = InlistBundle::from_file_with_options(
        &hb_dir.join("inlist_hb_2M_header"),
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("parse hb_2M inlist");
    let hb_mod_text =
        std::fs::read_to_string(hb_dir.join("standard_2.0_ZACHeB.mod"))
            .expect("read standard_2.0_ZACHeB.mod");
    let mut m_hb = read_mod_str(
        &hb_mod_text,
        hb_bundle.controls.clone(),
        hb_bundle.job.clone(),
    )
    .expect("parse hb_2M starting mod");
    m_hb.eos_config = hb_bundle.eos_config;
    m_hb.kap_config = hb_bundle.kap_config;
    set_hydro_vars(&mut m_hb, false, false).expect("set_hydro_vars hb_2M");
    assert_eq!(m_hb.nz, 1346);

    let cfg_hb = MeshFunctionConfig::from_controls(&m_hb.controls);
    let (hb_splits, hb_merges) =
        do_adjust_mesh(&mut m_hb, &cfg_hb).expect("remesh hb_2M step 2");
    println!(
        "hb_2M step-2 remesh: 1346 -> {} (splits={hb_splits}, merges={hb_merges}; MESA 26.4.1: 2629)",
        m_hb.nz
    );
    assert!(hb_splits > 1000, "hb_2M must refine from 1346 toward ~2629");
    let rel_hb = (m_hb.nz as f64 - 2629.0).abs() / 2629.0;
    assert!(
        rel_hb < 0.01,
        "hb_2M step-2 num_zones={} differs from MESA 26.4.1 2629 (rel={rel_hb:.4})",
        m_hb.nz
    );

    // 3. `1.3M_ms_high_Z` (`standard_zams_0.040_1.30.mod`, evolved 4 steps with remeshing):
    //    MESA `26.4.1` `num_zones` along steps 1..5 is `[807, 811, 809, 810, 805]`.
    let highz_ref = read_reference_num_zones(
        &root
            .join("benchmarks")
            .join("history_harness")
            .join("reference_histories")
            .join("star")
            .join("1.3M_ms_high_Z.data"),
    );
    assert_eq!(highz_ref[0], (1, 807));
    assert_eq!(highz_ref[1], (2, 811));

    let highz_dir = suite.join("1.3M_ms_high_Z");
    assert!(
        highz_dir.is_dir(),
        "missing 1.3M_ms_high_Z test_suite inputs at {}",
        highz_dir.display()
    );
    let highz_bundle = InlistBundle::from_file_with_options(
        &highz_dir.join("inlist_1.3M_ms_high_Z_header"),
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("parse 1.3M_ms_high_Z inlist");
    let highz_mod_text =
        std::fs::read_to_string(highz_dir.join("standard_zams_0.040_1.30.mod"))
            .expect("read standard_zams_0.040_1.30.mod");
    let mut m_highz = read_mod_str(
        &highz_mod_text,
        highz_bundle.controls.clone(),
        highz_bundle.job.clone(),
    )
    .expect("parse 1.3M_ms_high_Z starting mod");
    m_highz.eos_config = highz_bundle.eos_config;
    m_highz.kap_config = highz_bundle.kap_config;
    set_hydro_vars(&mut m_highz, false, false)
        .expect("set_hydro_vars 1.3M_ms_high_Z");
    assert_eq!(m_highz.nz, 807);

    let mut highz_seq = vec![m_highz.nz];
    for (step_idx, &(_, expected_zones)) in
        highz_ref.iter().enumerate().take(5).skip(1)
    {
        let rep = star_evolve_step(&mut m_highz, true, false, true)
            .expect("evolve 1.3M_ms_high_Z step with remesh");
        assert!(rep.converged);
        highz_seq.push(m_highz.nz);
        let rel_diff = (m_highz.nz as f64 - expected_zones as f64).abs()
            / (expected_zones as f64);
        assert!(
            rel_diff < 0.01,
            "1.3M_ms_high_Z step {} num_zones={} vs MESA 26.4.1 {expected_zones} (rel={rel_diff:.4})",
            step_idx + 1,
            m_highz.nz
        );
    }
    println!(
        "1.3M_ms_high_Z steps 1..5 num_zones: {highz_seq:?} (MESA 26.4.1: [807, 811, 809, 810, 805])"
    );
}

#[test]
fn test_zero_change_remesh_250_clamp_and_tiny_x_preplan_clean() {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    // Relax at `mesh_max_allowed_ratio = 2.0` with merging disabled until 0
    // splits remain, guaranteeing 0 splits and 0 merges at `ratio = 2.2 < 2.5`.
    model.controls.mesh_delta_coeff = 5.0;
    model.controls.mix_wind.mesh_ok_to_merge = false;
    model.controls.mix_wind.merge_if_dlnr_too_small = false;
    model.controls.mix_wind.merge_if_dr_div_cs_too_small = false;
    model.controls.mix_wind.merge_if_dr_div_drstar_too_small = false;
    model.controls.mix_wind.min_dq = 1.0e-30;
    model.controls.mix_wind.max_dq = 1.0;
    model.controls.mix_wind.max_surface_cell_dq = 1.0;
    model.controls.mix_wind.max_center_cell_dq = 1.0e-20;
    model.controls.mix_wind.p_function_weight = 1.0e-6;
    model.controls.mix_wind.t_function1_weight = 0.0;
    model.controls.mix_wind.xa_function_weight.fill(0.0);
    for _ in 0..32 {
        model.controls.mix_wind.mesh_max_allowed_ratio = 2.0;
        let init_cfg = MeshFunctionConfig::from_controls(&model.controls);
        let (s, m) =
            do_adjust_mesh(&mut model, &init_cfg).expect("initial ratio relax");
        if s == 0 && m == 0 {
            break;
        }
    }

    // Zero an unused abundance slot to verify `clean_up_fractions` `tiny_x = 1e-99` floor.
    let species = model.species;
    let last_sp = species - 1;
    let orig = model.xa[last_sp];
    model.xa[0] += orig;
    model.xa[last_sp] = 0.0;

    model.controls.mix_wind.mesh_max_allowed_ratio = 2.2;
    let calls_before = model.aux.mesh_call_number;
    let cfg = MeshFunctionConfig::from_controls(&model.controls);
    let (ns, nm) =
        do_adjust_mesh(&mut model, &cfg).expect("0-change do_adjust_mesh");
    assert_eq!((ns, nm), (0, 0));
    assert_eq!(model.aux.mesh_call_number, calls_before + 1);
    assert_eq!(model.controls.mix_wind.mesh_max_allowed_ratio, 2.5);
    assert!(
        model.xa[last_sp] >= 0.99e-99,
        "expected tiny_x=1e-99 floor on zero abundance, got {}",
        model.xa[last_sp]
    );

    // Verify `plan_mesh` -> `remap_onto_dq` on a 0-change step also applies #250 clamp.
    model.controls.mix_wind.mesh_max_allowed_ratio = 2.2;
    let cfg2 = MeshFunctionConfig::from_controls(&model.controls);
    let (dq_new, ns2, nm2) = flexmesa_star::plan_mesh(&model, &cfg2);
    assert_eq!((ns2, nm2), (0, 0));
    flexmesa_star::remap_onto_dq(&mut model, &dq_new)
        .expect("0-change remap_onto_dq");
    assert_eq!(model.controls.mix_wind.mesh_max_allowed_ratio, 2.5);
    assert_eq!(model.aux.mesh_call_number, calls_before + 2);
}
