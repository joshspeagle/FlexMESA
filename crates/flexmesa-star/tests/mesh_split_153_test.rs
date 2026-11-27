// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone unit and invariant tests for adaptive
//! mesh planning, spacing functions, `delta_gval_max` limits, `remesh_dt_limit`,
//! `#250` `mesh_max_allowed_ratio` clamping, `#241` / `#243`, and
//! `split_merge_amr` `alpha_mlt` preservation (`#153`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::sync::Arc;

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{MixingType, LN10, M_SUN, R_SUN, SECYER};
use flexmesa_star::adjust_mesh_support::{
    blend_coef, check_validity, decreasing_outward,
    get_gval_info_with_controls, increasing_inward, increasing_outward,
    set_delta_gval_max_with_controls, single_peak, smooth_gvals,
};
use flexmesa_star::mesh_functions::{
    num_mesh_functions_with_controls, set_mesh_function_data_with_controls,
    MAX_ALLOWED_GVALS,
};
use flexmesa_star::{
    do_adjust_mesh, do_mesh_plan, do_prune_mesh_surface,
    eval_effective_mesh_delta_coeff, eval_mesh_gval, get1_grad,
    load_zams_model, minmod3, num_mesh_functions, plan_mesh,
    preview_split_merge_amr, remap_onto_dq, remesh_split_merge, set_hydro_vars,
    set_mesh_function_data, try_plan_standard_mesh, MeshFunctionConfig,
    OtherMeshDeltaCoeffFactorHandle, OtherMeshDeltaCoeffFactorHook,
    OtherMeshFunctionColumn, OtherMeshFunctionsHandle, OtherMeshFunctionsHook,
    StarControls, StarError, StarJobControls, StarModel,
};

fn four_species_ids() -> Vec<usize> {
    let cat = NuclideCatalog::builtin_ref();
    vec![
        cat.get_by_name("h1").expect("h1").chem_id,
        cat.get_by_name("he4").expect("he4").chem_id,
        cat.get_by_name("o16").expect("o16").chem_id,
        cat.get_by_name("mg24").expect("mg24").chem_id,
    ]
}

fn synthetic_hydro_model(nz: usize) -> StarModel {
    let chem_id = four_species_ids();
    let sp = chem_id.len();
    let mut controls = StarControls {
        rotation_flag: true,
        okay_to_remesh: true,
        mesh_delta_coeff: 1.0,
        ..StarControls::default()
    };
    controls.mix_wind.xa_function_species = vec![String::new(); 9];
    controls.mix_wind.xa_function_weight = vec![0.0; 9];
    controls.mix_wind.xa_function_param = vec![1.0e-2; 9];
    controls.mix_wind.xa_mesh_delta_coeff = vec![1.0; 9];
    controls.mix_wind.p_function_weight = 0.0;
    controls.mix_wind.t_function1_weight = 0.0;

    let mut model =
        StarModel::new(nz, chem_id, StarJobControls::default(), controls);
    model.xmstar = 2.0 * M_SUN;
    model.dt = 10.0 * SECYER;
    model.r_center = 1.0e6;
    model.aux.eps_nuc_categories = vec![[0.0; NUM_CATEGORIES]; nz];

    let dq_val = 1.0 / (nz as f64);
    let mut q_acc = 1.0_f64;
    let mut xq_acc = 0.0_f64;
    for k in 0..nz {
        model.dq[k] = dq_val;
        model.xq[k] = xq_acc;
        model.q[k] = q_acc;
        model.dm[k] = dq_val * model.xmstar;
        model.m[k] = q_acc * model.xmstar;
        xq_acc += dq_val;
        q_acc -= dq_val;

        let frac = (k as f64) / ((nz - 1).max(1) as f64);
        let rad = (1.0 - 0.85 * frac) * R_SUN;
        model.r[k] = rad;
        model.ln_r[k] = flexmesa_math::log(rad);
        let temp = 1.0e5 * (1.0 + 99.0 * frac);
        model.t[k] = temp;
        model.ln_t[k] = flexmesa_math::log(temp);
        let peos = 1.0e10 * flexmesa_math::exp(4.0 * frac);
        model.peos[k] = peos;
        model.ln_peos[k] = flexmesa_math::log(peos);
        model.pgas[k] = 0.9 * peos;
        let eng = 1.0e14 * (1.0 + 3.0 * frac);
        model.energy[k] = eng;
        model.ln_e[k] = flexmesa_math::log(eng);
        let rho = 1.0e-2 * flexmesa_math::exp(3.0 * frac);
        model.rho[k] = rho;
        model.lnd[k] = flexmesa_math::log(rho);
        model.gradt[k] = 0.25 + 0.1 * frac;
        model.tau[k] = flexmesa_math::exp10(-1.0 + 6.0 * frac);
        model.opacity[k] = 0.4 + 0.5 * frac;
        model.gam[k] = 150.0 + 10.0 * frac;
        model.omega[k] = 1.0e-5 * (1.0 + 2.0 * frac);
        model.csound[k] = 1.0e7 * (1.0 + frac);

        let x_h = 0.7 - 0.4 * frac;
        let x_he = 0.28 + 0.35 * frac;
        let x_o = 0.015 + 0.04 * frac;
        let x_mg = (1.0 - x_h - x_he - x_o).max(1.0e-6);
        model.xa[k * sp] = x_h;
        model.xa[k * sp + 1] = x_he;
        model.xa[k * sp + 2] = x_o;
        model.xa[k * sp + 3] = x_mg;
    }
    model.dq[nz - 1] = 1.0 - model.dq[..(nz - 1)].iter().sum::<f64>();
    model
}

fn load_clean_zams_1m() -> StarModel {
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
    model.dt = 1.0e4 * SECYER;
    model
}

struct FixedColsHook(Vec<OtherMeshFunctionColumn>);

impl OtherMeshFunctionsHook for FixedColsHook {
    fn eval_other_mesh_functions(
        &self,
        _model: &StarModel,
    ) -> Result<Vec<OtherMeshFunctionColumn>, StarError> {
        Ok(self.0.clone())
    }
}

struct CustomDeltaFactorHook(Vec<f64>);

impl OtherMeshDeltaCoeffFactorHook for CustomDeltaFactorHook {
    fn eval_mesh_delta_coeff_factor(
        &self,
        _model: &StarModel,
        mesh_delta_coeff_factor: &mut [f64],
    ) -> Result<(), StarError> {
        mesh_delta_coeff_factor.copy_from_slice(&self.0);
        Ok(())
    }
}

#[test]
fn remesh_dt_limit_and_logt_suppress_before_cleanup_or_clamp() {
    let mut model = load_clean_zams_1m();
    let species = model.species;
    let last_sp = species - 1;
    let orig_last = model.xa[last_sp];
    model.xa[0] += orig_last;
    model.xa[last_sp] = 0.0;

    model.controls.mesh_delta_coeff = 0.5;
    model.controls.mix_wind.mesh_max_allowed_ratio = 2.0;
    model.controls.mix_wind.remesh_dt_limit = 1.0e5;
    model.dt = 1.0e4;

    let nz_before = model.nz;
    let dq_before = model.dq.clone();
    let calls_before = model.aux.mesh_call_number;
    let cfg = MeshFunctionConfig::from_controls(&model.controls);

    assert!(
        try_plan_standard_mesh(&model, &cfg)
            .expect("try_plan_standard_mesh under remesh_dt_limit")
            .is_none(),
        "dt < remesh_dt_limit must suppress try_plan_standard_mesh"
    );
    let (dq_plan, ns_plan, nm_plan) = plan_mesh(&model, &cfg);
    assert_eq!((ns_plan, nm_plan), (0, 0));
    assert_eq!(dq_plan, dq_before);

    let (ns, nm) = do_adjust_mesh(&mut model, &cfg)
        .expect("do_adjust_mesh under remesh_dt_limit");
    assert_eq!((ns, nm), (0, 0));
    assert_eq!(model.nz, nz_before);
    assert_eq!(model.aux.mesh_call_number, calls_before);
    assert_eq!(
        model.controls.mix_wind.mesh_max_allowed_ratio, 2.0,
        "remesh_dt_limit early return must precede #250 clamp"
    );
    assert_eq!(
        model.xa[last_sp], 0.0,
        "remesh_dt_limit early return must precede clean_up_fractions"
    );

    // At dt == remesh_dt_limit remeshing proceeds normally.
    model.dt = 1.0e5;
    assert!(
        try_plan_standard_mesh(&model, &cfg)
            .expect("try_plan_standard_mesh at boundary")
            .is_some(),
        "dt == remesh_dt_limit must allow planning"
    );
    let (ns_go, nm_go) =
        do_adjust_mesh(&mut model, &cfg).expect("do_adjust_mesh at boundary");
    assert!(ns_go + nm_go > 0);
    assert_eq!(model.aux.mesh_call_number, calls_before + 1);
    assert_eq!(model.controls.mix_wind.mesh_max_allowed_ratio, 2.5);
    assert!(model.xa[last_sp] >= 0.99e-99);

    // remesh_max_allowed_logT suppression also leaves clamp untouched.
    let mut m_hot = load_clean_zams_1m();
    m_hot.controls.mix_wind.mesh_max_allowed_ratio = 2.0;
    let max_logt =
        m_hot.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max) / LN10;
    m_hot.controls.mix_wind.remesh_max_allowed_logt = max_logt - 0.1;
    let hot_cfg = MeshFunctionConfig::from_controls(&m_hot.controls);
    assert!(try_plan_standard_mesh(&m_hot, &hot_cfg)
        .expect("try_plan_standard_mesh hot")
        .is_none());
    let (ns_hot, nm_hot) =
        do_adjust_mesh(&mut m_hot, &hot_cfg).expect("do_adjust_mesh hot");
    assert_eq!((ns_hot, nm_hot), (0, 0));
    assert_eq!(m_hot.controls.mix_wind.mesh_max_allowed_ratio, 2.0);
}

#[test]
fn mesh_max_allowed_ratio_250_clamps_even_when_planning_errors() {
    let mut model = load_clean_zams_1m();
    model.controls.mix_wind.mesh_max_allowed_ratio = 2.1;
    model.controls.mix_wind.p_function_weight = 0.0;
    model.controls.mix_wind.t_function1_weight = 0.0;
    model.controls.mix_wind.xa_function_weight.fill(0.0);

    let cfg = MeshFunctionConfig::from_controls(&model.controls);
    let err = do_adjust_mesh(&mut model, &cfg)
        .expect_err("zero active mesh functions should error");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert_eq!(
        model.controls.mix_wind.mesh_max_allowed_ratio, 2.5,
        "upstream adjust_mesh.f90:155-159 clamps before get_num_gvals"
    );
}

#[test]
fn effective_mesh_delta_coeff_hight_blend_and_heaviest_isotope() {
    let mut model = synthetic_hydro_model(5);
    model.controls.mesh_delta_coeff = 0.8;
    model.controls.mix_wind.mesh_delta_coeff_for_hight = 2.4;
    model
        .controls
        .mix_wind
        .logt_max_for_standard_mesh_delta_coeff = 9.0;
    model.controls.mix_wind.logt_min_for_hight_mesh_delta_coeff = 9.5;

    // Below standard ceiling: uses mesh_delta_coeff directly.
    model.ln_t[4] = 8.5 * LN10;
    assert!(
        (eval_effective_mesh_delta_coeff(&model, &model.controls.mix_wind)
            - 0.8)
            .abs()
            < 1.0e-15
    );

    // Midpoint blend: alfa = 0.5 => 0.5 * 2.4 + 0.5 * 0.8 = 1.6.
    model.ln_t[4] = 9.25 * LN10;
    assert!(
        (eval_effective_mesh_delta_coeff(&model, &model.controls.mix_wind)
            - 1.6)
            .abs()
            < 1.0e-14
    );

    // Above highT floor: uses mesh_delta_coeff_for_highT.
    model.ln_t[4] = 9.8 * LN10;
    assert!(
        (eval_effective_mesh_delta_coeff(&model, &model.controls.mix_wind)
            - 2.4)
            .abs()
            < 1.0e-14
    );

    // Non-heaviest species (`o16`) must not scale delta_coeff, while heaviest
    // species in net (`mg24`) multiplies delta_coeff (`adjust_mesh.f90:161-179`).
    model.controls.mix_wind.xa_function_species[0] = String::from("o16");
    model.controls.mix_wind.xa_mesh_delta_coeff[0] = 0.5;
    model.controls.mix_wind.xa_function_species[1] = String::from("mg24");
    model.controls.mix_wind.xa_mesh_delta_coeff[1] = 0.25;
    assert!(
        (eval_effective_mesh_delta_coeff(&model, &model.controls.mix_wind)
            - 0.6)
            .abs()
            < 1.0e-14
    );
}

#[test]
fn builtin_gval_formulas_rotation_gate_and_duplicate_xa_overwrite() {
    let nz = 5_usize;
    let mut model = synthetic_hydro_model(nz);
    {
        let mw = &mut model.controls.mix_wind;
        mw.e_function_weight = 10.0;
        mw.e_function_param = 14.5;
        mw.p_function_weight = 40.0;
        mw.t_function1_weight = 110.0;
        mw.t_function2_weight = 12.0;
        mw.t_function2_param = 2.0e5;
        mw.r_function_weight = 5.0;
        mw.r_function_param = 1.0e-3;
        mw.r_function2_weight = 6.0;
        mw.r_function2_param1 = 0.7;
        mw.r_function2_param2 = 0.25;
        mw.r_function3_weight = 7.0;
        mw.m_function_weight = 8.0;
        mw.m_function_param = 1.0e-2;
        mw.gradt_function_weight = 9.0;
        mw.log_tau_function_weight = 4.0;
        mw.log_kap_function_weight = 3.0;
        mw.gam_function_weight = 11.0;
        mw.gam_function_param1 = 155.0;
        mw.gam_function_param2 = 10.0;
        mw.omega_function_weight = 13.0;
        mw.xa_function_species[0] = String::from("h1");
        mw.xa_function_weight[0] = 20.0;
        mw.xa_function_param[0] = 1.0e-2;
        mw.xa_function_species[1] = String::from("he4");
        mw.xa_function_weight[1] = 30.0;
        mw.xa_function_param[1] = 2.0e-2;
    }

    let nfcns = num_mesh_functions(&model).expect("15 active functions");
    assert_eq!(nfcns, 15);
    let data = set_mesh_function_data(&model, nfcns).expect("mesh data");
    assert_eq!(
        data.names,
        vec![
            "E_function",
            "P_function",
            "T_function1",
            "T_function2",
            "R_function",
            "R_function2",
            "R_function3",
            "M_function",
            "gradT_function",
            "log_tau_function",
            "log_kap_function",
            "gam_function",
            "omega_function",
            "h1",
            "he4",
        ]
    );
    assert!(data.is_logt_function[2]);
    assert!(data.is_logt_function[3]);
    assert!(data.is_xa_function[13]);
    assert!(data.is_xa_function[14]);

    let r_surf = model.r[0];
    let sp = model.species;
    for k in 0..nz {
        let e_exp = 10.0 * 14.5_f64.max(model.ln_e[k] / LN10);
        let p_exp = 40.0 * model.ln_peos[k] / LN10;
        let t1_exp = 110.0 * model.ln_t[k] / LN10;
        let t2_exp =
            12.0 * flexmesa_math::log10(model.t[k] / (model.t[k] + 2.0e5));
        let r_exp =
            5.0 * flexmesa_math::log10(1.0 + (model.r[k] / R_SUN) / 1.0e-3);
        let r2_exp = 6.0 * 0.7_f64.min(0.25_f64.max(model.r[k] / r_surf));
        let r3_exp = 7.0 * model.r[k] / r_surf;
        let m_exp = 8.0
            * flexmesa_math::log10(
                1.0 + ((model.xmstar * model.q[k]) / M_SUN) / 1.0e-2,
            );
        let gt_exp = 9.0 * model.gradt[k];
        let tau_exp = 4.0 * flexmesa_math::log10(model.tau[k]);
        let kap_exp = 3.0 * flexmesa_math::log10(model.opacity[k]);
        let gam_exp = 11.0 * flexmesa_math::tanh((model.gam[k] - 155.0) / 10.0);
        let om_exp =
            13.0 * flexmesa_math::log10(1.0e-99_f64.max(model.omega[k].abs()));
        let h1_exp = 20.0 * flexmesa_math::log10(model.xa[k * sp] + 1.0e-2);
        let he4_exp =
            30.0 * flexmesa_math::log10(model.xa[k * sp + 1] + 2.0e-2);

        let expected = [
            e_exp, p_exp, t1_exp, t2_exp, r_exp, r2_exp, r3_exp, m_exp, gt_exp,
            tau_exp, kap_exp, gam_exp, om_exp, h1_exp, he4_exp,
        ];
        for (j, &exp_val) in expected.iter().enumerate() {
            assert!(
                (data.column(j, nz)[k] - exp_val).abs() < 1.0e-12,
                "col {} ({}) zone {}: got {}, expected {}",
                j,
                data.names[j],
                k,
                data.column(j, nz)[k],
                exp_val
            );
        }
    }

    // Disabling rotation_flag drops omega_function even if weight > 0.
    model.controls.rotation_flag = false;
    assert_eq!(num_mesh_functions(&model).expect("non-rot count"), 14);

    // Duplicate xa_function_species entry overwrites earlier matching name slot.
    model.controls.mix_wind.xa_function_species[2] = String::from("h1");
    model.controls.mix_wind.xa_function_weight[2] = 50.0;
    model.controls.mix_wind.xa_function_param[2] = 5.0e-2;
    let n_dup = num_mesh_functions(&model).expect("dup count");
    assert_eq!(n_dup, 15);
    let data_dup = set_mesh_function_data(&model, n_dup).expect("dup data");
    let h1_first = 50.0 * flexmesa_math::log10(model.xa[0] + 5.0e-2);
    assert!((data_dup.column(12, nz)[0] - h1_first).abs() < 1.0e-12);
    assert!((data_dup.column(14, nz)[0] - h1_first).abs() < 1.0e-12);
}

#[test]
fn convective_bdy_dt_gate_dq_limit_step_and_241_limit() {
    let mut model = synthetic_hydro_model(7);
    model.controls.mix_wind.convective_bdy_weight = 10.0;
    model.controls.mix_wind.convective_bdy_dq_limit = 0.05;
    model.controls.mix_wind.convective_bdy_min_dt_yrs = 1.0e-2;
    // Center=convective(k1=7,6), radiative(k1=5,4), convective(k1=3), rad(2,1)
    model.mixing_type = vec![
        MixingType::NoMixing,
        MixingType::NoMixing,
        MixingType::ConvectiveMixing,
        MixingType::NoMixing,
        MixingType::NoMixing,
        MixingType::ConvectiveMixing,
        MixingType::ConvectiveMixing,
    ];

    // dt < convective_bdy_min_dt_yrs * SECYER => zero column.
    model.dt = 1.0e-3 * SECYER;
    let nfcns = num_mesh_functions(&model).expect("nfcns");
    assert_eq!(nfcns, 1);
    let data_small_dt =
        set_mesh_function_data(&model, nfcns).expect("small dt data");
    assert!(data_small_dt.column(0, 7).iter().all(|&v| v == 0.0));

    // With dt >= threshold and dq[1] below dq_limit, only boundaries k1=5,4 emit.
    model.dt = 1.0 * SECYER;
    model.dq[1] = 0.01;
    let data_dq_lim =
        set_mesh_function_data(&model, nfcns).expect("dq limit data");
    let col = data_dq_lim.column(0, 7);
    // Boundaries (1-based) in locate_fortran_conv_boundaries order:
    // j=1: (k1=5, top=true), j=2: (k1=4, top=false), j=3: (k1=2, top=true).
    // j=3 is skipped by dq[1] < dq_limit, while #241 reads top_conv_bdy(i=1)
    // = true for j=2 as well.
    assert_eq!(col[0], 0.0);
    assert_eq!(col[2], 0.0);
    assert_eq!(col[3], 10.0);
    assert_eq!(col[4], 20.0);
    assert_eq!(col[5], 30.0);
    assert_eq!(col[6], 30.0);

    // Exceeding MAX_ALLOWED_GVALS triggers explicit validation error.
    let err_max = set_mesh_function_data(&model, MAX_ALLOWED_GVALS + 1)
        .expect_err("exceed MAX_ALLOWED_GVALS");
    assert!(matches!(err_max, StarError::InvalidInput(_)));
}

#[test]
fn other_mesh_functions_zero_cols_mismatch_and_min_dq_for_xa() {
    let mut model = synthetic_hydro_model(4);
    model.controls.mix_wind.use_other_mesh_functions = true;

    // 1. Hook returning 0 columns is valid when all built-ins are disabled.
    model.aux.other_mesh_functions = Some(OtherMeshFunctionsHandle(Arc::new(
        FixedColsHook(Vec::new()),
    )));
    assert_eq!(num_mesh_functions(&model).expect("0 custom cols"), 0);
    let empty_data =
        set_mesh_function_data(&model, 0).expect("0-col mesh data");
    assert_eq!(empty_data.names, Vec::<String>::new());
    assert_eq!(empty_data.vals, Vec::<f64>::new());

    // 2. Column length mismatch produces clear InvalidInput error.
    model.aux.other_mesh_functions =
        Some(OtherMeshFunctionsHandle(Arc::new(FixedColsHook(vec![
            OtherMeshFunctionColumn {
                name: String::from("bad_len"),
                is_xa_function: false,
                vals: vec![1.0; 3],
            },
        ]))));
    let len_err =
        set_mesh_function_data(&model, 1).expect_err("length mismatch error");
    assert!(matches!(len_err, StarError::InvalidInput(_)));

    // 3. Compare `is_xa_function = true` vs `false` on `min_dq_for_xa` cutoff
    //    via `do_mesh_plan` on a steep step in zone `k1 = 3` (0-based `k0 = 2`).
    let mut m_xa = synthetic_hydro_model(5);
    m_xa.controls.mix_wind.use_other_mesh_functions = true;
    m_xa.controls.mix_wind.min_dq = 0.04;
    m_xa.controls.mix_wind.min_dq_for_split = 0.01;
    m_xa.controls.mix_wind.min_dq_for_xa = 0.25;
    m_xa.controls.mix_wind.min_dq_for_xa_convective = 0.25;
    m_xa.controls.mix_wind.max_dq = 0.5;
    m_xa.controls.mix_wind.max_surface_cell_dq = 0.5;
    m_xa.controls.mix_wind.max_center_cell_dq = 0.25;
    m_xa.controls.mix_wind.mesh_ok_to_merge = false;

    let step_vals = vec![0.0, 0.0, 100.0, 0.0, 0.0];
    let xq_old = vec![0.0, 0.2, 0.4, 0.6, 0.8];
    let dq_old = m_xa.dq.clone();
    let do_not_split = vec![false; 5];
    let delta_gval_max = vec![1.0; 5];

    m_xa.aux.other_mesh_functions =
        Some(OtherMeshFunctionsHandle(Arc::new(FixedColsHook(vec![
            OtherMeshFunctionColumn {
                name: String::from("custom_xa"),
                is_xa_function: true,
                vals: step_vals.clone(),
            },
        ]))));
    let fcns_xa =
        set_mesh_function_data(&m_xa, 1).expect("custom_xa mesh data");
    let plan_xa = do_mesh_plan(
        &m_xa,
        &m_xa.controls.mix_wind,
        &xq_old,
        &dq_old,
        m_xa.controls.mix_wind.min_dq,
        m_xa.controls.mix_wind.max_dq,
        m_xa.controls.mix_wind.min_dq_for_split,
        2.5,
        &do_not_split,
        &fcns_xa,
        &delta_gval_max,
        0.25,
        0.5,
        1.0e-30,
    )
    .expect("do_mesh_plan is_xa=true");

    let mut m_nonxa = m_xa.clone();
    m_nonxa.aux.other_mesh_functions =
        Some(OtherMeshFunctionsHandle(Arc::new(FixedColsHook(vec![
            OtherMeshFunctionColumn {
                name: String::from("custom_nonxa"),
                is_xa_function: false,
                vals: step_vals,
            },
        ]))));
    let fcns_nonxa =
        set_mesh_function_data(&m_nonxa, 1).expect("custom_nonxa mesh data");
    let plan_nonxa = do_mesh_plan(
        &m_nonxa,
        &m_nonxa.controls.mix_wind,
        &xq_old,
        &dq_old,
        m_nonxa.controls.mix_wind.min_dq,
        m_nonxa.controls.mix_wind.max_dq,
        m_nonxa.controls.mix_wind.min_dq_for_split,
        2.5,
        &do_not_split,
        &fcns_nonxa,
        &delta_gval_max,
        0.25,
        0.5,
        1.0e-30,
    )
    .expect("do_mesh_plan is_xa=false");

    assert_eq!(
        plan_xa.dq_new.len(),
        5,
        "min_dq_for_xa=0.25 suppresses splitting dq=0.2 zones when is_xa=true"
    );
    assert!(
        plan_nonxa.dq_new.len() > 5,
        "non-xa custom column splits steep jump down toward min_dq_for_split"
    );
}

#[test]
fn delta_gval_max_pgas_dlogx_243_categories_smoothing_and_monotone_helpers() {
    assert_eq!(blend_coef(0.4, 0.2), 0.4);
    assert!((blend_coef(0.4, 0.75) - 0.7).abs() < 1.0e-15);

    let mut v_peak = vec![1.0, 5.0, 3.0, 4.0, 2.0];
    single_peak(&mut v_peak);
    assert_eq!(v_peak, vec![1.0, 5.0, 3.0, 3.0, 2.0]);

    let mut v_inc_in = vec![2.0, 5.0, 3.0, 6.0];
    increasing_inward(&mut v_inc_in);
    assert_eq!(v_inc_in, vec![2.0, 5.0, 5.0, 6.0]);

    let mut v_dec_out = vec![6.0, 3.0, 5.0, 2.0];
    decreasing_outward(&mut v_dec_out);
    assert_eq!(v_dec_out, vec![2.0, 2.0, 2.0, 2.0]);

    let mut v_inc_out = vec![2.0, 4.0, 3.0, 5.0];
    increasing_outward(&mut v_inc_out);
    assert_eq!(v_inc_out, vec![5.0, 5.0, 5.0, 5.0]);

    let mut gvals = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    smooth_gvals(5, 1, &mut gvals);
    assert!(gvals[2] < 1.0 && gvals[1] > 0.0 && gvals[0] > 0.0);

    // Test Pgas/P exponent, #243 dlogX/dlogP, i_other=23, smoothing cap=3, hook.
    let mut model = synthetic_hydro_model(5);
    let sp = model.species;
    model.pgas = vec![0.25 * model.peos[0]; 5];
    model.pgas[1] = 0.04 * model.peos[1];
    model.pgas[4] = 0.0; // pow(0.0, 0.5) = 0.0 in MESA without guard
    model.controls.mix_wind.p_function_weight = 40.0;
    model.controls.mix_wind.mesh_pgas_div_p_exponent = 0.5;
    model.controls.mix_wind.mesh_delta_coeff_factor_smooth_iters = 0;

    let dg_pgas = set_delta_gval_max_with_controls(
        &model,
        &model.controls.mix_wind,
        &model.xa,
    )
    .expect("pgas/P delta_gval_max");
    assert!((dg_pgas[0] - 0.5).abs() < 1.0e-15);
    assert!((dg_pgas[1] - 0.2).abs() < 1.0e-15);
    assert!((dg_pgas[4] - 0.0).abs() < 1.0e-15);

    // #243 dlogX/dlogP uses abs(lnPeos[k] - lnPeos[k-1]) * LN10.
    model.controls.mix_wind.mesh_pgas_div_p_exponent = 0.0;
    model.ln_peos = vec![20.0 * LN10; 5];
    model.ln_peos[2] = 20.3 * LN10;
    model.ln_peos[3] = 20.3 * LN10;
    model.ln_peos[4] = 20.3 * LN10;
    for k in 0..5 {
        model.xa[k * sp] = if k < 2 { 0.5 } else { 0.05 };
    }
    model.controls.mix_wind.mesh_logx_species = vec![String::from("h1")];
    model.controls.mix_wind.mesh_logx_min_for_extra = vec![-2.0];
    model.controls.mix_wind.mesh_dlogx_dlogp_extra = vec![0.2];
    model.controls.mix_wind.mesh_dlogx_dlogp_full_off = vec![1.0];
    model.controls.mix_wind.mesh_dlogx_dlogp_full_on = vec![2.0];
    // Upstream `1.0 / (0.3 * LN10^2) ~= 0.629 < 1.0` => inactive under #243,
    // whereas `1.0 / 0.3 = 3.333` without bug would saturate to 0.2.
    let dg_243_sub = set_delta_gval_max_with_controls(
        &model,
        &model.controls.mix_wind,
        &model.xa,
    )
    .expect("#243 below full_off");
    assert!(
        dg_243_sub.iter().all(|&v| (v - 1.0).abs() < 1.0e-15),
        "expected #243 LN10 factor to keep dlogX/dlogP below full_off=1.0"
    );

    // Steep nuclear category `other` (index 23) + smooth_iters 5 == 3 cap.
    model.controls.mix_wind.mesh_logx_species.clear();
    model.controls.mix_wind.mesh_dlog_other_dlogp_extra = 0.1;
    model.aux.eps_nuc_categories[2][23] = 1.0e8;
    model.aux.eps_nuc_categories[1][23] = 1.0e1;
    model.aux.eps_nuc_categories[3][23] = 1.0e1;

    let mut mw3 = model.controls.mix_wind.clone();
    mw3.mesh_delta_coeff_factor_smooth_iters = 3;
    let mut mw5 = model.controls.mix_wind.clone();
    mw5.mesh_delta_coeff_factor_smooth_iters = 5;
    let dg_s3 = set_delta_gval_max_with_controls(&model, &mw3, &model.xa)
        .expect("smooth 3");
    let dg_s5 = set_delta_gval_max_with_controls(&model, &mw5, &model.xa)
        .expect("smooth 5");
    assert_eq!(dg_s3, dg_s5, "smoothing stops after 3 passes");
    assert!(dg_s3[2] < 1.0);

    // Custom `other_mesh_delta_coeff_factor` multiplies ginfo + eval_mesh_gval.
    model.controls.mix_wind.mesh_delta_coeff_factor_smooth_iters = 0;
    model.controls.mix_wind.use_other_mesh_delta_coeff_factor = true;
    model.aux.other_mesh_delta_coeff_factor =
        Some(OtherMeshDeltaCoeffFactorHandle(Arc::new(
            CustomDeltaFactorHook(vec![0.5; 5]),
        )));
    let n_base = num_mesh_functions(&model).expect("nfcns");
    let info =
        get_gval_info_with_controls(&model, &model.controls.mix_wind, n_base)
            .expect("gval info with factor hook");
    assert!(info.delta_gval_max.iter().all(|&d| d <= 0.5 + 1.0e-15));
    let g_vec = eval_mesh_gval(
        &model,
        &MeshFunctionConfig::from_controls(&model.controls),
    );
    assert_eq!(g_vec.len(), 5);
    assert!(g_vec.iter().all(|v| v.is_finite()));
}

#[test]
fn mesh_plan_dq_bounds_ratio_dlnr_and_check_validity() {
    // 1. check_validity catches non-monotonic ln_r and non-positive dq.
    let mut bad = synthetic_hydro_model(4);
    bad.ln_r[1] = bad.ln_r[2];
    assert!(check_validity(&bad).is_err());

    let mut bad_dq = synthetic_hydro_model(4);
    bad_dq.dq[2] = 0.0;
    assert!(check_validity(&bad_dq).is_err());

    // 2. Adjacent ratio limit + max_surface_cell_dq + max_center_cell_dq +
    //    do_prune_mesh_surface + eval_effective_mesh_delta_coeff scaling.
    let mut model = load_clean_zams_1m();
    model.controls.mesh_delta_coeff = 2.0;
    let mut mw = model.controls.mix_wind.clone();
    mw.p_function_weight = 1.0e-12;
    mw.t_function1_weight = 0.0;
    mw.xa_function_weight.fill(0.0);
    mw.mesh_max_allowed_ratio = 2.5;
    mw.max_surface_cell_dq = 1.0e-6;
    mw.max_center_cell_dq = 1.0e-4;
    mw.max_dq = 0.1;
    mw.min_dq = 1.0e-14;

    let nfcns = num_mesh_functions_with_controls(&model, &mw).expect("1 fcn");
    let raw_fcns = set_mesh_function_data_with_controls(&model, &mw, nfcns)
        .expect("raw fcns");
    let ginfo =
        get_gval_info_with_controls(&model, &mw, nfcns).expect("gval info");
    assert_eq!(raw_fcns.names, ginfo.fcns.names);
    let dq_old = model.dq.clone();
    let mut xq_old = vec![0.0_f64; model.nz];
    for k in 1..model.nz {
        xq_old[k] = xq_old[k - 1] + dq_old[k - 1];
    }
    let dns = vec![false; model.nz];
    let plan = do_mesh_plan(
        &model,
        &mw,
        &xq_old,
        &dq_old,
        mw.min_dq,
        mw.max_dq * model.controls.mesh_delta_coeff,
        mw.min_dq_for_split,
        mw.mesh_max_allowed_ratio,
        &dns,
        &ginfo.fcns,
        &ginfo.delta_gval_max,
        mw.max_center_cell_dq * model.controls.mesh_delta_coeff,
        mw.max_surface_cell_dq * model.controls.mesh_delta_coeff,
        mw.min_surface_cell_dq * model.controls.mesh_delta_coeff,
    )
    .expect("bounded mesh plan");

    let n_new = plan.dq_new.len();
    assert!(plan.dq_new[0] <= 2.0 * 1.0e-6 * 1.0001);
    assert!(plan.dq_new[n_new - 1] <= 2.0 * 1.0e-4 * 1.0001);
    for k in 0..n_new {
        assert!(plan.dq_new[k] >= mw.min_dq * 0.99);
        assert!(plan.dq_new[k] <= 0.2 * 1.0001);
        if k > 0 {
            let ratio = (plan.dq_new[k] / plan.dq_new[k - 1])
                .max(plan.dq_new[k - 1] / plan.dq_new[k]);
            assert!(
                ratio <= 2.5001,
                "adjacent cell ratio {ratio} exceeded 2.5 at k={k}"
            );
        }
    }

    // 3. mesh_min_dlnr sets do_not_split in try_plan_standard_mesh.
    let mut m_dlnr = synthetic_hydro_model(5);
    m_dlnr.ln_peos[4] = m_dlnr.ln_peos[3];
    m_dlnr.peos[4] = m_dlnr.peos[3];
    m_dlnr.controls.mix_wind.p_function_weight = 500.0;
    m_dlnr.controls.mix_wind.max_surface_cell_dq = 1.0;
    m_dlnr.controls.mix_wind.max_center_cell_dq = 0.5;
    m_dlnr.controls.mix_wind.max_dq = 1.0;
    m_dlnr.controls.mix_wind.mesh_ok_to_merge = false;
    m_dlnr.controls.mix_wind.mesh_min_dlnr = 10.0;
    let dlnr_cfg = MeshFunctionConfig::from_controls(&m_dlnr.controls);
    let plan_dlnr = try_plan_standard_mesh(&m_dlnr, &dlnr_cfg)
        .expect("dlnr try_plan")
        .expect("plan present");
    assert_eq!(plan_dlnr.num_splits, 0);

    // 4. do_prune_mesh_surface reduces nz cleanly by removing outer cells.
    let mut m_prune = load_clean_zams_1m();
    let nz_orig = m_prune.nz;
    do_prune_mesh_surface(&mut m_prune, nz_orig - 3)
        .expect("prune 3 surface cells");
    assert_eq!(m_prune.nz, nz_orig - 3);
    let (dq_p, ns_p, nm_p) = plan_mesh(
        &m_prune,
        &MeshFunctionConfig::from_controls(&m_prune.controls),
    );
    assert_eq!(dq_p.len(), m_prune.nz + ns_p - nm_p);
    remap_onto_dq(&mut m_prune, &dq_p).expect("remap after prune");
    assert_eq!(m_prune.nz, dq_p.len());
}

#[test]
fn split_merge_amr_alpha_mlt_preservation_minmod3_and_guards() {
    assert_eq!(minmod3(2.0, -1.0, 3.0), 0.0);
    assert_eq!(minmod3(2.0, 4.0, 1.5), 1.5);
    assert_eq!(minmod3(-2.0, -4.0, -1.5), -1.5);
    assert_eq!(get1_grad(1.0, 2.0, 3.0, 0.0, 1.0, 1.0), 0.0);
    let g_lin = get1_grad(0.0, 1.0, 2.0, 1.0, 1.0, 1.0);
    assert!((g_lin - 1.0).abs() < 1.0e-14);

    // Split path (including center cell i1 == nz_old) preserves/copies alpha_mlt.
    let mut m_split = load_clean_zams_1m();
    m_split.aux.mixing.okay_to_set_mixing_info = false;
    for k in 0..m_split.nz {
        m_split.aux.mixing.alpha_mlt[k] = 3.5 + 0.002 * (k as f64);
    }
    let alpha_center_before = m_split.aux.mixing.alpha_mlt[m_split.nz - 1];
    m_split.controls.mix_wind.use_split_merge_amr = true;
    m_split.controls.mix_wind.split_merge_amr_nz_baseline = 2000;
    m_split.controls.mix_wind.split_merge_amr_maxlong = 1.2;
    m_split.controls.mix_wind.split_merge_amr_maxshort = 1.0e10;
    m_split.controls.mix_wind.split_merge_amr_max_iters = 3;
    m_split.controls.mix_wind.split_merge_amr_okay_to_split_nz = true;

    let (prev_ns, prev_nm) =
        preview_split_merge_amr(&m_split).expect("preview split");
    assert!(prev_ns > 0 && prev_nm == 0);

    let nz_before_split = m_split.nz;
    let (ns, nm) = remesh_split_merge(&mut m_split).expect("remesh split");
    assert!(ns > 0 && nm == 0);
    assert_eq!(m_split.nz, nz_before_split + ns);
    assert_eq!(m_split.aux.mixing.alpha_mlt.len(), m_split.nz);
    assert!(
        m_split.aux.mixing.alpha_mlt.iter().all(|&a| a >= 3.5),
        "all split/shifted zones must preserve alpha_mlt >= 3.5"
    );
    assert_eq!(
        m_split.aux.mixing.alpha_mlt[m_split.nz - 1],
        alpha_center_before,
        "splitting centermost zone must copy alpha_mlt to new inner child"
    );

    // Merge path shifts alpha_mlt downward along with zone arrays.
    let mut m_merge = load_clean_zams_1m();
    m_merge.aux.mixing.okay_to_set_mixing_info = false;
    let nz_before_merge = m_merge.nz;
    for k in 0..nz_before_merge {
        m_merge.aux.mixing.alpha_mlt[k] = 3.6 + 0.003 * (k as f64);
    }
    {
        let cfg = &mut m_merge.controls.mix_wind;
        cfg.use_split_merge_amr = true;
        cfg.split_merge_amr_nz_baseline = 40;
        cfg.split_merge_amr_maxshort = 1.2;
        cfg.split_merge_amr_maxlong = 1.0e10;
        cfg.split_merge_amr_max_iters = 3;
        cfg.merge_amr_ignore_surface_cells = false;
        cfg.split_merge_amr_avoid_repeated_remesh = false;
    }

    let mut m_merge_guard = m_merge.clone();
    m_merge_guard
        .controls
        .mix_wind
        .split_merge_amr_avoid_repeated_remesh = true;

    let (ns_m, nm_m) = remesh_split_merge(&mut m_merge).expect("remesh merge");
    assert!(nm_m > 0 && ns_m == 0);
    assert_eq!(m_merge.nz, nz_before_merge - nm_m);
    assert_eq!(m_merge.aux.mixing.alpha_mlt.len(), m_merge.nz);
    assert!(
        m_merge.aux.mixing.alpha_mlt.iter().all(|&a| a >= 3.6),
        "all remaining/merged zones must retain alpha_mlt >= 3.6"
    );

    // With avoid_repeated_remesh enabled, neighboring merges can return early
    // cleanly while preserving alpha_mlt across all remaining zones.
    let (ns_g, nm_g) =
        remesh_split_merge(&mut m_merge_guard).expect("remesh merge guard");
    assert!(nm_g > 0 && ns_g == 0);
    assert!(m_merge_guard.nz <= nz_before_merge);
    assert!(m_merge_guard.nz >= nz_before_merge - nm_g);
    assert!(m_merge_guard.aux.mixing.alpha_mlt.iter().all(|&a| a >= 3.6));
}
