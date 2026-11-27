// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-MESA_DIR parity tests for Wave 11 (`#162`/`#170`/`#171`)
//! fixes across pulsation exporters (`pulse`, `pulse_utils`, `pulse_exporters`)
//! and coupled `RSP2` (`rsp2_ad`, `rsp2_eqns`, `rsp2_remesh`).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::similar_names)]

use flexmesa_const::{CGAS, CLIGHT, CRAD, PI4, R_SUN, SQRT_2_DIV_3};
use flexmesa_star::hydro_energy::set_energy_eqn_scal;
use flexmesa_star::pulse_exporters::{
    export_cafein, export_gr1d, export_gr1d_strict, export_saio,
    export_saio_strict, fortran_e_no_1p, get_cafein_data, get_gr1d_data_strict,
    get_saio_data_strict, try_export_gsm_data,
};
use flexmesa_star::pulse_gyre_data::{get_gyre_data, GyreDataInput};
use flexmesa_star::rsp2::{rsp2_flux_limiter, X_ALFAS};
use flexmesa_star::rsp2_ad::{
    eval_rsp2_hp_face_ad, eval_rsp2_ptrb_cell_ad, update_rsp2_vars_ad,
};
use flexmesa_star::rsp2_eqns::{
    eval_rsp2_hp_face_zone, eval_rsp2_profile_equations,
};
use flexmesa_star::rsp2_remesh::remesh_for_rsp2;
use flexmesa_star::zams::load_zams_model;
use flexmesa_star::{
    eval_center_rho, eval_face, eval_henyey_equations, eval_photosphere_info,
    export_gyre, export_pulse_by_format, export_pulse_data_text,
    record_start_of_step_energy_integrals, set_hydro_vars, StarControls,
    StarJobControls, StarModel, VarLayout,
};

fn hydrated_solar_zams() -> StarModel {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            calculate_brunt_n2: true,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    model
}

fn remeshed_rsp2_zams(customize: impl FnOnce(&mut StarControls)) -> StarModel {
    let mut controls = StarControls {
        v_flag: true,
        rsp2_flag: true,
        mixing_length_alpha: 1.5,
        zbase: 0.02,
        ..StarControls::default()
    };
    controls.rsp2.nz = 80;
    controls.rsp2.nz_outer = 24;
    controls.rsp2.t_anchor = 2.5e4;
    controls.rsp2.alfap = 1.0;
    controls.rsp2.alfad = 1.0;
    controls.rsp2.alfam = 0.25;
    controls.rsp2.alfat = 0.01;
    controls.rsp2.alfar = 1.0;
    controls.rsp2.num_outermost_cells_forced_nonturbulent = 2;
    controls.rsp2.nz_div_ibotom = 6.0;
    customize(&mut controls);

    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    remesh_for_rsp2(&mut model).expect("remesh_for_rsp2");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    model.dt = 25.0;
    model.save_start_values().expect("save_start_values");
    record_start_of_step_energy_integrals(&mut model);
    model
}

#[test]
fn test_pulse_exporters_gsm_matches_get_gyre_data_and_case_insensitive_text() {
    let model = hydrated_solar_zams();
    let gsm =
        try_export_gsm_data(&model, true, false, false, 110).expect("gsm v110");
    let mut gyre_in = GyreDataInput::from_model(&model).expect("gyre input");
    gyre_in.gyre_data_schema = 110;
    let (global_ref, rows_ref) =
        get_gyre_data(&gyre_in, true, false, false).expect("get_gyre_data");

    assert_eq!(gsm.n, rows_ref.len());
    assert_eq!(gsm.datasets.len(), 18);
    assert_eq!(gsm.m_star.to_bits(), global_ref[0].to_bits());
    assert_eq!(gsm.r_star.to_bits(), global_ref[1].to_bits());
    assert_eq!(gsm.l_star.to_bits(), global_ref[2].to_bits());
    let expected_r_star = R_SUN * eval_photosphere_info(&model).r_rsun;
    assert_eq!(gsm.r_star.to_bits(), expected_r_star.to_bits());

    for (col_idx, (_, col)) in gsm.datasets.iter().enumerate() {
        for (j, (&got, ref_row)) in col.iter().zip(rows_ref.iter()).enumerate()
        {
            assert_eq!(
                got.to_bits(),
                ref_row[col_idx].to_bits(),
                "gsm col {col_idx} row {j} differs from get_gyre_data"
            );
        }
    }

    let gsm_120 =
        try_export_gsm_data(&model, true, false, false, 120).expect("gsm v120");
    assert_eq!(gsm_120.datasets.len(), 19);
    assert_eq!(gsm_120.datasets[17].0, "eps_grav");
    assert_eq!(gsm_120.datasets[18].0, "Omega_rot");

    assert!(try_export_gsm_data(&model, true, false, false, 101).is_err());
    assert!(try_export_gsm_data(&model, true, false, true, 110).is_err());

    let mut atm_model = model.clone();
    atm_model.controls.add_atmosphere_to_pulse_data = true;
    assert!(export_pulse_by_format(&atm_model, "gsm", true).is_err());

    let gyre_upper =
        export_gyre(&model, true, false, false).expect("export_gyre");
    let gyre_lower = export_pulse_data_text(&model, "gyre", true, false, false)
        .expect("lowercase pulse.f90 match");
    assert_eq!(gyre_upper, gyre_lower);
}

#[test]
fn test_pulse_cafein_matches_mesa_formulas_log_deriv_and_formatting() {
    let model = hydrated_solar_zams();
    let (g, rows) =
        get_cafein_data(&model, true, true, false).expect("get_cafein_data");
    let nz = model.nz;
    assert_eq!(rows.len(), nz + 1);

    let phot = eval_photosphere_info(&model);
    let r_star = R_SUN * phot.r_rsun;
    let m_star = model.m_grav[0];
    let cgrav1 = model.cgrav[0];
    let s_units = (flexmesa_math::pow3(r_star) / (cgrav1 * m_star)).sqrt();
    let erg_units = cgrav1 * flexmesa_math::pow2(m_star) / r_star;
    let k_units = cgrav1 * flexmesa_math::pow2(m_star) / (r_star * CGAS);
    assert_eq!(g[1].to_bits(), (r_star / R_SUN).to_bits());
    assert_eq!(g[11].to_bits(), k_units.to_bits());

    // Center row invariants (`store_point_data_ctr` + `log_deriv` defaults).
    assert_eq!(rows[0][4], 0.0);
    assert_eq!(rows[0][7], 3.0);
    assert_eq!(rows[0][10], f64::MAX * s_units * s_units);
    assert_eq!(rows[0][24], 3.0);
    assert_eq!(rows[0][21], rows[0][34]);

    // Interior `dlnLr/dlnr` is centered logarithmic derivative, not constant 3.
    assert!(rows
        .iter()
        .skip(1)
        .any(|row| (row[24] - 3.0).abs() > 1.0e-3));
    assert!(rows.iter().all(
        |row| (row[21] - row[34]).abs() <= 1.0e-15 * row[34].abs().max(1.0)
    ));

    // Entropy uses `exp(lnS)` scaled by `cp_units`.
    let cp_units = erg_units / (k_units * m_star);
    let k_mid = nz / 2;
    let lns_face =
        eval_face(&model.dq, &model.ln_s, k_mid, 0, nz - 1, None, None);
    let expected_s = flexmesa_math::exp(lns_face) / cp_units;
    let row_idx = nz - k_mid;
    assert!(
        (rows[row_idx][29] - expected_s).abs() / expected_s.abs() <= 1.0e-12
    );

    // Check `E24.14E3` formatter and exact upstream CAFEIN header strings.
    assert_eq!(fortran_e_no_1p(1.0, 24, 14, 3), "   0.10000000000000E+001");
    assert_eq!(
        fortran_e_no_1p(-0.999_999_999_999_999, 24, 14, 3),
        "  -0.10000000000000E+001"
    );
    let text = export_cafein(&model, true, true, false);
    assert!(text.contains("..............Pc(erg/cm3)"));
    assert!(text.contains("...................chiRho"));
    assert!(text.contains("  0."));
    assert!(get_cafein_data(&model, true, true, true).is_err());
}

#[test]
fn test_pulse_saio_gr1d_decisions43_errors_and_pulse_utils_eval_face() {
    let model = hydrated_solar_zams();
    let saio_err = export_saio_strict(&model, true, false)
        .expect_err("SAIO reads uninitialized kap per DECISIONS #43");
    assert!(saio_err.to_string().contains("#169"));
    assert!(get_saio_data_strict(&model, false, false).is_err());

    let gr1d_err = export_gr1d_strict(&model)
        .expect_err("GR1D missing j increment per DECISIONS #43");
    assert!(gr1d_err.to_string().contains("#169"));
    assert!(get_gr1d_data_strict(&model).is_err());

    let legacy_saio = export_saio(&model, true, false);
    assert_eq!(legacy_saio.lines().count(), 1 + 4 * model.nz);
    let legacy_gr1d = export_gr1d(&model);
    assert_eq!(legacy_gr1d.lines().count(), 1 + model.nz);

    // Cubic `eval_face` matches `interp_4_to_1` (`star_utils.f90:792`) exactly.
    let k = model.nz / 2;
    let mut gyre_in = GyreDataInput::from_model(&model).expect("gyre input");
    gyre_in.gyre_data_schema = 110;
    let (_, gyre_rows) =
        get_gyre_data(&gyre_in, false, true, false).expect("gyre_data");
    let got_t = eval_face(&model.dq, &model.t, k, 0, model.nz - 1, None, None);
    let ref_t = gyre_rows[model.nz - 1 - k][4];
    assert_eq!(got_t.to_bits(), ref_t.to_bits());

    // `eval_center_rho` evaluates MESA formula directly without clamp fallbacks.
    let mut synth = StarModel::new(
        2,
        vec![1],
        StarJobControls::default(),
        StarControls::default(),
    );
    synth.rmid = vec![2.0, 1.0];
    synth.r = vec![3.0, 1.5];
    synth.m = vec![2.0, 1.0];
    synth.dm = vec![1.0, 1.0];
    synth.rho = vec![0.1, 1.0];
    let rho_c = eval_center_rho(&synth, 1);
    let expected =
        3.0 * (5.0 * 0.5 / (flexmesa_const::PI * 1.0) - 4.0 * 1.0) / 8.0;
    assert_eq!(rho_c.to_bits(), expected.to_bits());
}

#[test]
#[allow(clippy::manual_midpoint)]
fn test_rsp2_remesh_ledoux_pii_and_lr_parity() {
    let mut model = remeshed_rsp2_zams(|c| {
        c.use_ledoux_criterion = true;
        c.calculate_brunt_b = true;
        c.rsp2.use_rsp_eqn_for_y_face = false;
        c.rsp2.use_tdc_enthalpy_flux_limiter = true;
        c.using_velocity_time_centering = true;
        c.include_p_in_velocity_time_centering = true;
        c.include_l_in_velocity_time_centering = true;
        c.p_theta_for_velocity_time_centering = 0.6;
        c.l_theta_for_velocity_time_centering = 0.7;
        c.max_logt_for_include_p_and_l_in_velocity_time_centering = 0.0;
    });
    let nz = model.nz;

    // 0. `remesh_for_rsp2` preserves MESA `set_xm_new` mass coordinates
    //    (within 4 eps after trailing `set_hydro_vars` -> `set_m_and_dm`).
    for k in 1..nz {
        assert!(
            (model.m[k] - (model.m[k - 1] - model.dm[k - 1])).abs()
                <= 4.0 * f64::EPSILON * model.mstar
        );
        assert!(
            (model.dq[k] - model.dm[k] / model.xmstar).abs()
                <= 4.0 * f64::EPSILON
        );
        assert_eq!(
            model.q[k].to_bits(),
            (model.q[k - 1] - model.dq[k - 1]).to_bits()
        );
        assert_eq!(
            model.dm_bar[k].to_bits(),
            (0.5 * (model.dm[k - 1] + model.dm[k])).to_bits()
        );
    }

    // 1. `compute_Y_face` includes `gradL_composition_term` when Ledoux is active.
    let k_ledoux = nz / 3;
    update_rsp2_vars_ad(&mut model);
    let y_base = model.aux.rsp.y_face_rsp[k_ledoux];
    model.aux.mixing.gradl_composition_term[k_ledoux] = 0.25;
    update_rsp2_vars_ad(&mut model);
    let y_shifted = model.aux.rsp.y_face_rsp[k_ledoux];
    assert!(((y_base - y_shifted) - 0.25).abs() <= 1.0e-13);
    model.aux.mixing.gradl_composition_term[k_ledoux] = 0.0;

    // 2. `compute_PII_face` reproduces upstream `T_face = alfa*Cp_00 + beta*Cp_m1`
    // (`hydro_rsp2.f90:567`, #178) inside the enthalpy flux limiter branch.
    let k_lim = 12;
    let dln_peos = model.ln_peos[k_lim - 1] - model.ln_peos[k_lim];
    model.ln_t[k_lim - 1] =
        model.ln_t[k_lim] + (model.grada[k_lim] + 4.0) * dln_peos;
    set_hydro_vars(&mut model, false, false).expect("hydro_vars steep");
    update_rsp2_vars_ad(&mut model);
    let y_face = model.aux.rsp.y_face_rsp[k_lim];
    assert!(y_face > 0.0);
    let (alfa, beta) = if model.controls.rsp2.use_mass_interp_face_values {
        let a = model.dq[k_lim - 1] / (model.dq[k_lim - 1] + model.dq[k_lim]);
        (a, 1.0 - a)
    } else {
        (0.5, 0.5)
    };
    let cp_face = alfa * model.cp[k_lim] + beta * model.cp[k_lim - 1];
    let rho_face = alfa * model.rho[k_lim] + beta * model.rho[k_lim - 1];
    let peos_face = alfa * model.peos[k_lim] + beta * model.peos[k_lim - 1];
    let e_face = alfa * flexmesa_math::exp(model.ln_e[k_lim])
        + beta * flexmesa_math::exp(model.ln_e[k_lim - 1]);
    let h_face = e_face + peos_face / rho_face;
    let alfas_alfa = X_ALFAS * model.controls.mixing_length_alpha;
    let t_face_bug = cp_face;
    let x_val = (((cp_face * t_face_bug / h_face) * alfas_alfa) * y_face)
        / SQRT_2_DIV_3;
    assert!(x_val >= 0.95);
    let (fl_v, _) = rsp2_flux_limiter(x_val);
    let expected_pii = ((cp_face * alfas_alfa) * y_face) * (fl_v / x_val);
    assert!(
        (model.aux.rsp.pii_rsp[k_lim] - expected_pii).abs()
            / expected_pii.abs()
            <= 1.0e-12
    );

    // 3. `compute_Lr(0)` matches upstream evaluation order `(area * factor) * c * (a * T^4)`.
    let area0 = PI4 * model.r[0] * model.r[0];
    let t4_0 = flexmesa_math::pow4(model.t[0]);
    let expected_lr0 =
        ((area0 * model.controls.rsp2.lsurf_factor) * CLIGHT * (CRAD * t4_0))
            / area0;
    assert_eq!(model.aux.rsp.fr_rsp[0].to_bits(), expected_lr0.to_bits());
}

#[test]
fn test_rsp2_time_centering_scal_and_unclamped_hp() {
    let mut model = remeshed_rsp2_zams(|c| {
        c.using_velocity_time_centering = true;
        c.include_p_in_velocity_time_centering = true;
        c.include_l_in_velocity_time_centering = true;
        c.p_theta_for_velocity_time_centering = 0.6;
        c.l_theta_for_velocity_time_centering = 0.7;
        c.max_logt_for_include_p_and_l_in_velocity_time_centering = 0.0;
    });
    let nz = model.nz;

    // 4. `eval_rsp2_ptrb_cell_ad` and `L_theta` omit `max_logt_for_include_p_and_l`.
    let k_turb = nz / 2;
    let ptrb_tw = eval_rsp2_ptrb_cell_ad(&model, k_turb, false).val;
    let pt_s = model.aux.rsp.ptrb_start[k_turb];
    let p_now_expected = (2.0 / 3.0)
        * model.controls.rsp2.alfap
        * model.rho[k_turb]
        * model.aux.rsp.et_rsp[k_turb];
    assert!(
        (ptrb_tw - (0.6 * p_now_expected + 0.4 * pt_s)).abs()
            <= 1.0e-12 * ptrb_tw.abs().max(1.0)
    );

    model.aux.rsp.lt_start[k_turb] += 1.0e34;
    let sys_a = eval_henyey_equations(&mut model, false).expect("henyey a");
    let i_etrb = model.layout.i_turbulent_energy.expect("i_etrb");
    let r_a = sys_a.equ[k_turb * sys_a.nvar + i_etrb];
    model.aux.rsp.lt_start[k_turb] -= 1.0e34;
    let sys_b = eval_henyey_equations(&mut model, false).expect("henyey b");
    let r_b = sys_b.equ[k_turb * sys_b.nvar + i_etrb];
    assert!((r_a - r_b).abs() > 1.0e-10);

    // 5. `set_energy_eqn_scal` applies `dedt_eqn_r_scale` to `do1_turbulent_energy_eqn`.
    model.aux.rsp.et_rsp[k_turb] *= 1.1;
    model.aux.rsp.w[k_turb] = model.aux.rsp.et_rsp[k_turb].sqrt();
    model.pack_xh();
    model.controls.dedt_eqn_r_scale = 1.0e8;
    let res_hi = eval_rsp2_profile_equations(&mut model).equ_etrb[k_turb];
    let scal_hi = set_energy_eqn_scal(&model, k_turb).expect("scal hi");
    model.controls.dedt_eqn_r_scale = 1.0;
    let res_lo = eval_rsp2_profile_equations(&mut model).equ_etrb[k_turb];
    let scal_lo = set_energy_eqn_scal(&model, k_turb).expect("scal lo");
    assert!(scal_lo < scal_hi);
    assert!(((res_lo / res_hi) - (scal_lo / scal_hi)).abs() <= 1.0e-12);

    // 6. `!rsp2_flag` sets `w - w_start`.
    model.controls.rsp2_flag = false;
    let res_off = eval_rsp2_profile_equations(&mut model).equ_etrb[k_turb];
    let expected_off =
        model.aux.rsp.et_rsp[k_turb].sqrt() - model.aux.rsp.w_start[k_turb];
    assert!((res_off - expected_off).abs() <= 1.0e-13);
}

#[test]
fn test_rsp2_hp_face_unclamped_and_get_scale_height_init() {
    // `Hp_face` is not clamped to 1.0 cm, and `hp_face <= 0` uses `get_scale_height_face_val`.
    let mut tiny = StarModel::new(
        2,
        vec![1],
        StarJobControls::default(),
        StarControls::default(),
    );
    tiny.layout = VarLayout::with_rsp2(1, true, false, true);
    tiny.controls.rsp2_flag = true;
    tiny.controls.mixing_length_alpha = 0.0;
    tiny.controls.rsp2.assume_hse = false;
    tiny.controls.mix_wind.alt_scale_height_flag = false;
    tiny.r = vec![1.0, 0.5];
    tiny.ln_r = vec![0.0, flexmesa_math::log(0.5)];
    tiny.m = vec![1.0, 0.5];
    tiny.m_grav = vec![1.0, 0.5];
    tiny.dm = vec![0.5, 0.5];
    tiny.dm_bar = vec![0.25, 0.5];
    tiny.dq = vec![0.5, 0.5];
    tiny.cgrav = vec![1.0, 1.0];
    tiny.rho = vec![1.0, 1.0];
    tiny.lnd = vec![0.0, 0.0];
    tiny.t = vec![1.0, 1.0];
    tiny.ln_t = vec![0.0, 0.0];
    tiny.peos = vec![0.1, 0.2];
    tiny.ln_peos = vec![flexmesa_math::log(0.1), flexmesa_math::log(0.2)];
    tiny.ln_peos_ad = vec![
        flexmesa_star::StarAdOrder1::constant(tiny.ln_peos[0]),
        flexmesa_star::StarAdOrder1::constant(tiny.ln_peos[1]),
    ];
    tiny.cp = vec![1.0, 1.0];
    tiny.chi_t = vec![1.0, 1.0];
    tiny.chi_rho = vec![1.0, 1.0];
    tiny.ln_e = vec![0.0, 0.0];
    tiny.opacity = vec![1.0, 1.0];
    tiny.d_opacity_dlnd = vec![0.0, 0.0];
    tiny.d_opacity_dlnt = vec![0.0, 0.0];
    tiny.grada = vec![0.4, 0.4];
    tiny.eos_d_dlnd = vec![[0.0; 26]; 2];
    tiny.eos_d_dlnt = vec![[0.0; 26]; 2];
    tiny.aux.rsp.resize(2);
    tiny.xh = vec![0.0; 2 * tiny.layout.nvar_hydro];
    assert!((eval_rsp2_hp_face_zone(&tiny, 0) - 0.1).abs() <= 1.0e-14);
    assert!((eval_rsp2_hp_face_ad(&tiny, 0).val - 0.1).abs() <= 1.0e-14);
    assert!(eval_rsp2_hp_face_zone(&tiny, 1) < 1.0);
    assert!(eval_rsp2_hp_face_ad(&tiny, 1).val < 1.0);
    update_rsp2_vars_ad(&mut tiny);
    assert!((tiny.aux.rsp.hp_face[0] - 0.1).abs() <= 1.0e-14);
    assert!((tiny.aux.rsp.hp_face[1] - 0.075).abs() <= 1.0e-14);
    let ihp = tiny.layout.i_hp.expect("i_hp");
    assert!((tiny.xh[ihp] - 0.1).abs() <= 1.0e-14);
    assert!((tiny.xh[tiny.layout.nvar_hydro + ihp] - 0.075).abs() <= 1.0e-14);
}
