// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#374` Wave 1 star history columns and sibling family tests:
//! `gradT_excess_*`, `shock_*`, `grav_dark_*`, `cz_*`, `v_wind_Km_per_s`,
//! `*_below_const_q`, `*_const_mass`, and `logxq_below_just_added`.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::fs;

use flexmesa_const::{
    FOUR_PI, L_SUN, LN10, M_SUN, MixingType, PI, R_SUN, SECYER,
};
use flexmesa_star::report_utils::{
    eval_shock_info, eval_surface_w_div_w_kep, gravity_darkening_l_coeff,
    gravity_darkening_table, gravity_darkening_teff_coeff,
};
use flexmesa_star::{
    HistoryColumnSpec, MesaCellValue, PrimaryStateBuffer, StarControls,
    StarJobControls, StarModel, StepReportContext, do_adjust_mass_step,
    evaluate_star_history_row, get_shock_mass_msun, load_zams_model,
    parse_star_history_columns, safe_log10, set_gradt_excess_alpha,
    set_hydro_vars,
};
use flexmesa_utils::require_mesa_test_input_path;

fn zams_model() -> StarModel {
    let ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_old_values();
    model
}

fn eval_one_col(model: &StarModel, col_name: &str) -> MesaCellValue {
    let specs =
        parse_star_history_columns(col_name, None, model).expect("parse col");
    assert_eq!(specs.len(), 1);
    let row = evaluate_star_history_row(model, &specs);
    assert_eq!(row.len(), 1);
    row[0].1
}

fn as_float(cell: &MesaCellValue) -> f64 {
    match cell {
        MesaCellValue::Float(v) => *v,
        other @ MesaCellValue::Int(_) => {
            panic!("expected Float cell, got {other:?}")
        }
    }
}

fn as_int(cell: &MesaCellValue) -> i64 {
    match cell {
        MesaCellValue::Int(v) => *v,
        other @ MesaCellValue::Float(_) => {
            panic!("expected Int cell, got {other:?}")
        }
    }
}

#[test]
fn test_gravity_darkening_table_matches_upstream_data_bit_for_bit() {
    let Some(path) = require_mesa_test_input_path(
        "data/star_data/gravity_darkening_coefficients.data",
    ) else {
        return;
    };
    let text = fs::read_to_string(&path).expect("read gravity darkening table");
    let mut rows = text.lines().skip(1).filter(|l| !l.trim().is_empty());

    let (omega_grid, incl_grid, c_t_vals, c_l_vals) = gravity_darkening_table();
    for (i, &expected_omega) in omega_grid.iter().enumerate() {
        for (j, &expected_incl) in incl_grid.iter().enumerate() {
            let line = rows.next().expect("441 table rows");
            let cols: Vec<f64> = line
                .split_whitespace()
                .map(|tok| tok.parse::<f64>().expect("parse float"))
                .collect();
            assert_eq!(cols.len(), 4);
            assert_eq!(
                expected_omega.to_bits(),
                cols[0].to_bits(),
                "omega mismatch at i={i}, j={j}"
            );
            assert_eq!(
                expected_incl.to_bits(),
                cols[1].to_bits(),
                "inclination mismatch at i={i}, j={j}"
            );
            let idx = i * 21 + j;
            assert_eq!(
                c_t_vals[idx].to_bits(),
                cols[2].to_bits(),
                "C_T mismatch at i={i}, j={j}"
            );
            assert_eq!(
                c_l_vals[idx].to_bits(),
                cols[3].to_bits(),
                "C_L mismatch at i={i}, j={j}"
            );

            let spline_ct = gravity_darkening_teff_coeff(cols[0], cols[1]);
            let spline_cl = gravity_darkening_l_coeff(cols[0], cols[1]);
            assert!(
                (spline_ct - cols[2]).abs() <= 1.0e-11,
                "C_T spline knot ({i}, {j}) mismatch: {spline_ct} vs {}",
                cols[2]
            );
            assert!(
                (spline_cl - cols[3]).abs() <= 1.0e-11,
                "C_L spline knot ({i}, {j}) mismatch: {spline_cl} vs {}",
                cols[3]
            );
        }
    }
    assert!(rows.next().is_none(), "unexpected trailing rows in table");

    // Verify domain clamping and NaN guards matching `gravity_darkening.f90:124-132`.
    assert!(
        (gravity_darkening_l_coeff(-0.5, -0.3) - 1.0).abs() <= 1.0e-15,
        "negative omega must clamp to 0 without abs()"
    );
    assert!(
        (gravity_darkening_teff_coeff(-0.5, 0.5) - 1.0).abs() <= 1.0e-15,
        "negative omega must clamp to 0 for C_T"
    );
    assert!(
        (gravity_darkening_l_coeff(1.5, 2.5)
            - gravity_darkening_l_coeff(1.0, 0.5 * PI))
        .abs()
            <= 1.0e-14
    );
    assert!(
        (gravity_darkening_teff_coeff(1.5, 2.5)
            - gravity_darkening_teff_coeff(1.0, 0.5 * PI))
        .abs()
            <= 1.0e-14
    );
    assert_eq!(gravity_darkening_l_coeff(f64::NAN, 0.0), 1.0);
    assert_eq!(gravity_darkening_teff_coeff(0.5, f64::INFINITY), 1.0);
}

#[test]
fn test_wave1_and_sibling_columns_parse_case_insensitively() {
    let model = zams_model();
    let canonical = [
        "gradT_excess_alpha",
        "gradT_excess_min_beta",
        "gradT_excess_max_lambda",
        "shock_mass",
        "shock_q",
        "shock_radius",
        "shock_velocity",
        "shock_csound",
        "shock_lgT",
        "shock_lgRho",
        "shock_lgP",
        "shock_gamma1",
        "shock_entropy",
        "shock_tau",
        "shock_k",
        "shock_pre_lgRho",
        "grav_dark_L_polar",
        "grav_dark_Teff_polar",
        "grav_dark_L_equatorial",
        "grav_dark_Teff_equatorial",
        "cz_bot_radius",
        "cz_log_xmsun",
        "cz_scale_height",
        "cz_top_radius",
        "cz_top_scale_height",
        "v_wind_Km_per_s",
        "k_below_const_q",
        "q_below_const_q",
        "logxq_below_const_q",
        "k_const_mass",
        "q_const_mass",
        "logxq_const_mass",
        "logxq_below_just_added",
    ];
    let mixed_case = canonical
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i % 2 == 0 {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let specs =
        parse_star_history_columns(&mixed_case, None, &model).expect("parse");
    let got_names: Vec<String> =
        specs.iter().map(HistoryColumnSpec::column_name).collect();
    assert_eq!(got_names, canonical);
}

#[test]
fn test_gradt_excess_columns_track_mltpp_activation() {
    let mut model = zams_model();
    assert_eq!(as_float(&eval_one_col(&model, "gradT_excess_alpha")), 0.0);
    assert_eq!(
        as_float(&eval_one_col(&model, "gradT_excess_min_beta")),
        0.0
    );
    assert_eq!(
        as_float(&eval_one_col(&model, "gradT_excess_max_lambda")),
        0.0
    );

    // Trigger MLT++ (`turb_info.f90:411-519`): low gas-pressure fraction at `k_beta = 10`
    // (`k >= 1` so `set_gradt_excess_alpha`'s `1..=k_beta` scan runs) and steep T^4 gradient.
    model.controls.mix_wind.okay_to_reduce_gradt_excess = true;
    model.controls.mix_wind.gradt_excess_lambda1 = 0.2;
    model.controls.mix_wind.gradt_excess_dlambda = 0.05;
    model.controls.mix_wind.gradt_excess_beta1 = 0.9;
    model.controls.mix_wind.gradt_excess_dbeta = 0.05;
    model.controls.mix_wind.gradt_excess_max_center_h1 = 1.0;
    model.controls.mix_wind.gradt_excess_min_center_he4 = 0.0;
    model.aux.mixing.gradt_excess_alpha_old = 0.5;

    let nz = model.nz;
    model.ln_pgas[10] = flexmesa_math::log(0.1 * model.peos[10]);
    model.t[10] *= 5.0;
    for k in 0..nz {
        model.xh_mass[k] = 0.0;
        model.yhe_mass[k] = 1.0;
    }
    set_gradt_excess_alpha(&mut model);

    assert!(model.aux.mixing.gradt_excess_alpha > 0.0);
    assert_eq!(
        as_float(&eval_one_col(&model, "gradT_excess_alpha")),
        model.aux.mixing.gradt_excess_alpha
    );
    assert_eq!(
        as_float(&eval_one_col(&model, "gradT_excess_min_beta")),
        model.aux.mixing.gradt_excess_min_beta
    );
    assert_eq!(
        as_float(&eval_one_col(&model, "gradT_excess_max_lambda")),
        model.aux.mixing.gradt_excess_max_lambda
    );
}

#[test]
fn test_shock_columns_outward_inward_glitch_and_absent() {
    let mut model = zams_model();
    let shock_cols = "\
        shock_mass\nshock_q\nshock_radius\nshock_velocity\nshock_csound\n\
        shock_lgT\nshock_lgRho\nshock_lgP\nshock_gamma1\nshock_entropy\n\
        shock_tau\nshock_k\nshock_pre_lgRho\n";
    let specs =
        parse_star_history_columns(shock_cols, None, &model).expect("parse");

    // 1. Neither `u_flag` nor `v_flag`: all shock columns are 0.
    model.controls.u_flag = false;
    model.controls.v_flag = false;
    let row = evaluate_star_history_row(&model, &specs);
    for (i, (_, cell)) in row.iter().enumerate() {
        if specs[i].column_name() == "shock_k" {
            assert_eq!(as_int(cell), 0);
        } else {
            assert_eq!(as_float(cell), 0.0);
        }
    }

    // 2. Single-point glitch where `u[k+1] <= csound[k+1]` is ignored (`star_utils.f90:1366`).
    model.controls.u_flag = true;
    model.u.fill(0.0);
    model.csound.fill(1.0e7);
    model.u[10] = 1.5e7;
    let glitch_info = eval_shock_info(&model);
    assert_eq!(glitch_info.k_1based, 0);
    assert_eq!(glitch_info.mass_msun, 0.0);

    // 3. Outward shock: construct exact Mach 1 crossing in the interior (`k_shock = nz / 2`),
    //    with `k + 2` strictly supersonic (`> csound`).
    let nz = model.nz;
    let k_s = nz / 2;
    for k in 0..nz {
        model.csound[k] = 2.0e7;
        model.u[k] = if k <= k_s {
            0.5 * model.csound[k]
        } else {
            1.5 * model.csound[k]
        };
    }
    // `find0` crosses linearly between `k_s + 1` (`v/cs - 1 = 0.5`) and `k_s` (`-0.5`),
    // so `shock_radius` is exact midpoint `(r[k_s] + r[k_s + 1]) / 2` and `shock_k = k_s + 1`.
    let r_mid = 0.5 * (model.r[k_s] + model.r[k_s + 1]);
    let expected_m_msun = 0.5 * (model.m[k_s] + model.m[k_s + 1]) / M_SUN;
    let expected_q = 0.5 * (model.q[k_s] + model.q[k_s + 1]);
    let expected_u = 0.5 * (model.u[k_s] + model.u[k_s + 1]);

    let row_out = evaluate_star_history_row(&model, &specs);
    assert!((as_float(&row_out[0].1) - expected_m_msun).abs() <= 1.0e-13);
    assert!(
        (as_float(&row_out[0].1) - get_shock_mass_msun(&model)).abs()
            <= 1.0e-15
    );
    assert!((as_float(&row_out[1].1) - expected_q).abs() <= 1.0e-13);
    assert!((as_float(&row_out[2].1) - r_mid / R_SUN).abs() <= 1.0e-13);
    assert!(
        (as_float(&row_out[3].1) - expected_u).abs() / expected_u <= 1.0e-12
    );
    assert_eq!(as_float(&row_out[4].1), model.csound[k_s]);
    assert_eq!(as_float(&row_out[5].1), model.ln_t[k_s] / LN10);
    assert_eq!(as_float(&row_out[6].1), model.lnd[k_s] / LN10);
    assert_eq!(as_float(&row_out[7].1), model.ln_peos[k_s] / LN10);
    assert_eq!(as_float(&row_out[8].1), model.gamma1[k_s]);
    assert_eq!(as_float(&row_out[9].1), model.entropy[k_s]);
    assert_eq!(as_float(&row_out[10].1), model.tau[k_s]);
    assert_eq!(as_int(&row_out[11].1), (k_s + 1) as i64);
    assert_eq!(as_float(&row_out[12].1), 0.0);

    // 4. Inward shock via `v_flag=true` (`v / cs <= -1`, `v[k+1] < -csound[k+1]`).
    model.controls.u_flag = false;
    model.controls.v_flag = true;
    for k in 0..nz {
        model.v[k] = if k <= 34 {
            -0.5 * model.csound[k]
        } else {
            -1.5 * model.csound[k]
        };
    }
    let in_info = eval_shock_info(&model);
    assert_eq!(in_info.k_1based, 35);
    assert!(in_info.velocity_cms < 0.0);
    assert!(in_info.mass_msun > 0.0);
}

#[test]
fn test_gravity_darkening_columns_rotating_retrograde_and_nonrotating() {
    let mut model = zams_model();
    let gd_cols = "\
        grav_dark_L_polar\ngrav_dark_Teff_polar\n\
        grav_dark_L_equatorial\ngrav_dark_Teff_equatorial\n";
    let specs =
        parse_star_history_columns(gd_cols, None, &model).expect("parse");

    // Non-rotating (`rotation_flag = false`) yields 0 even with nonzero `omega[0]`.
    model.controls.rotation_flag = false;
    model.omega[0] = 1.5e-4;
    let row0 = evaluate_star_history_row(&model, &specs);
    for (_, cell) in &row0 {
        assert_eq!(as_float(cell), 0.0);
    }

    // Upstream MESA `26.4.1` surface state matching high-rotation darkening
    // step 1 (`w_div_wc = 0.8552251338958738` -> `r_eq / r = 1.1448360363224068`,
    // `w_div_w_Kep = 0.8536257228696654`): verify exact formula and reference agreement.
    model.controls.rotation_flag = true;
    model.m[0] = 2.999_999_980_280_599 * M_SUN;
    let r_surf = flexmesa_math::exp(0.388_389_276_171_118_6 * LN10) * R_SUN;
    model.r[0] = r_surf;
    model.ln_r[0] = flexmesa_math::log(r_surf);
    model.aux.w_div_wc[0] = 0.855_225_133_895_873_8;
    model.omega[0] = 1.981_308_396_795_946e-4;
    model.lum[0] = 77.233_741_732_674_45 * L_SUN;
    model.teff = flexmesa_math::exp(4.039_083_711_235_257 * LN10);

    let w_kep = eval_surface_w_div_w_kep(&model);
    assert!(w_kep > 0.8 && w_kep < 0.9);
    let l_surf = model.surface_l_lsun();
    let row_rot = evaluate_star_history_row(&model, &specs);

    let l_pol = as_float(&row_rot[0].1);
    let t_pol = as_float(&row_rot[1].1);
    let l_eq = as_float(&row_rot[2].1);
    let t_eq = as_float(&row_rot[3].1);

    assert!(
        (l_pol - gravity_darkening_l_coeff(w_kep, 0.0) * l_surf).abs()
            <= 1.0e-13
    );
    assert!(
        (t_pol - gravity_darkening_teff_coeff(w_kep, 0.0) * model.teff).abs()
            <= 1.0e-11
    );
    assert!(
        (l_eq - gravity_darkening_l_coeff(w_kep, 0.5 * PI) * l_surf).abs()
            <= 1.0e-13
    );
    assert!(
        (t_eq - gravity_darkening_teff_coeff(w_kep, 0.5 * PI) * model.teff)
            .abs()
            <= 1.0e-11
    );
    // Pole is brighter and hotter than equator at rapid rotation.
    assert!(l_pol > l_surf && l_eq < l_surf);
    assert!(t_pol > model.teff && t_eq < model.teff);
    // Verify tight agreement with upstream `26.4.1` log reference (`110.5765`, `11260.24`, `61.7077`, `10662.98`).
    assert!((l_pol - 110.576_538_822_917_26).abs() / 110.5765 < 1.0e-4);
    assert!((t_pol - 11_260.236_310_822_038).abs() / 11_260.24 < 1.0e-4);
    assert!((l_eq - 61.707_677_561_177_206).abs() / 61.7077 < 1.0e-4);
    assert!((t_eq - 10_662.983_849_810_229).abs() / 10_662.98 < 1.0e-4);

    // Negative retrograde `omega[0]` clamps `w_div_w_Kep` to `0.0` (`C = 1.0`).
    model.omega[0] = -2.0e-4;
    let row_neg = evaluate_star_history_row(&model, &specs);
    assert!((as_float(&row_neg[0].1) - l_surf).abs() <= 1.0e-13);
    assert!((as_float(&row_neg[1].1) - model.teff).abs() <= 1.0e-11);
}

#[test]
fn test_convective_zone_and_wind_velocity_columns() {
    let mut model = zams_model();
    let nz = model.nz;
    let cz_cols = "\
        cz_bot_radius\ncz_log_xmsun\ncz_scale_height\n\
        cz_top_radius\ncz_top_scale_height\nv_wind_Km_per_s\n";
    let specs =
        parse_star_history_columns(cz_cols, None, &model).expect("parse");

    // 1. Fully radiative model: `largest_conv_mixing_region == 0`.
    model.mixing_type.fill(MixingType::NoMixing);
    model.mstar_dot = 1.0e-10 * M_SUN / SECYER; // accreting -> wind speed 0
    let row_rad = evaluate_star_history_row(&model, &specs);
    assert_eq!(as_float(&row_rad[0].1), 0.0);
    assert_eq!(as_float(&row_rad[1].1), -99.0);
    assert_eq!(as_float(&row_rad[2].1), 0.0);
    assert_eq!(as_float(&row_rad[3].1), 0.0);
    assert_eq!(as_float(&row_rad[4].1), 0.0);
    assert_eq!(as_float(&row_rad[5].1), 0.0);

    // 2. Convective core reaching innermost zone (`bot + 1 == nz`) with nonzero `R_center`.
    model.r_center = 0.05 * R_SUN;
    let top_core = nz - 15;
    model.mixing_type[top_core..nz].fill(MixingType::ConvectiveMixing);
    model.mstar_dot = -2.5e-8 * M_SUN / SECYER;
    let ctx_core = StepReportContext::compute(&model);
    let row_core = evaluate_star_history_row(&model, &specs);
    let bot_core = nz - 1;
    let expected_sum_dq: f64 = model.dq[..bot_core].iter().sum();
    let expected_v_wind = 1.0e-5 * model.opacity[0] * (-model.mstar_dot)
        / (FOUR_PI * ctx_core.phot.r_rsun * R_SUN * model.tau_base);
    assert!(
        (as_float(&row_core[0].1) - model.r_center / R_SUN).abs() <= 1.0e-15
    );
    assert!(
        (as_float(&row_core[1].1)
            - safe_log10(model.xmstar * expected_sum_dq / M_SUN))
        .abs()
            <= 1.0e-14
    );
    assert_eq!(as_float(&row_core[2].1), model.scale_height[bot_core]);
    assert!(
        (as_float(&row_core[3].1) - model.r[top_core] / R_SUN).abs() <= 1.0e-15
    );
    assert_eq!(as_float(&row_core[4].1), model.scale_height[top_core]);
    assert!((as_float(&row_core[5].1) - expected_v_wind).abs() <= 1.0e-14);

    // 3. Interior convective shell (`bot + 1 < nz`): `cz_bot_radius` uses `r[bot] / R_SUN`.
    model.mixing_type.fill(MixingType::NoMixing);
    let shell_top = 10;
    let shell_bot = 60;
    model.mixing_type[shell_top..=shell_bot].fill(MixingType::ConvectiveMixing);
    let row_shell = evaluate_star_history_row(&model, &specs);
    assert!(
        (as_float(&row_shell[0].1) - model.r[shell_bot] / R_SUN).abs()
            <= 1.0e-15
    );
    assert_eq!(as_float(&row_shell[2].1), model.scale_height[shell_bot]);
    assert!(
        (as_float(&row_shell[3].1) - model.r[shell_top] / R_SUN).abs()
            <= 1.0e-15
    );
    assert_eq!(as_float(&row_shell[4].1), model.scale_height[shell_top]);
}

#[test]
fn test_mass_adjust_boundary_columns_and_state_restore() {
    let mut model = zams_model();

    let mass_cols = "\
        k_below_const_q\nq_below_const_q\nlogxq_below_const_q\n\
        k_const_mass\nq_const_mass\nlogxq_const_mass\nlogxq_below_just_added\n";
    let specs =
        parse_star_history_columns(mass_cols, None, &model).expect("parse");

    // 1. Initial defaults (`k_below_const_q = 1`, `k_const_mass = 1` 1-based, `k_below_just_added = 1`).
    let neg99 = safe_log10(0.0);
    let row0 = evaluate_star_history_row(&model, &specs);
    assert_eq!(as_int(&row0[0].1), 1);
    assert_eq!(as_float(&row0[1].1), model.q[0]);
    assert_eq!(as_float(&row0[2].1), neg99);
    assert_eq!(as_int(&row0[3].1), 1);
    assert_eq!(as_float(&row0[4].1), model.q[0]);
    assert_eq!(as_float(&row0[5].1), neg99);
    assert_eq!(as_float(&row0[6].1), neg99);

    // 2. Mass accretion step: `k_below_const_q` and `k_below_just_added` move inward.
    let dt = 10.0 * SECYER;
    model.mstar_dot = 1.0e-4 * M_SUN / SECYER;
    model.dt = dt;
    do_adjust_mass_step(&mut model, dt).expect("accrete step");

    let k_a_1 = model.aux.k_below_const_q;
    let k_b_0 = model.aux.k_const_mass;
    let k_b_1 = (k_b_0 + 1).min(model.nz);
    let k_ja_1 = model.k_below_just_added;
    assert!(k_a_1 > 1);
    assert!(k_b_1 >= k_a_1);
    assert!(k_ja_1 > 1);

    let row_acc = evaluate_star_history_row(&model, &specs);
    assert_eq!(as_int(&row_acc[0].1), k_a_1 as i64);
    assert_eq!(as_float(&row_acc[1].1), model.q[k_a_1 - 1]);
    let expected_logxq_a =
        safe_log10(model.dq[..k_a_1 - 1].iter().sum::<f64>());
    assert!((as_float(&row_acc[2].1) - expected_logxq_a).abs() <= 1.0e-14);
    assert_eq!(as_int(&row_acc[3].1), k_b_1 as i64);
    assert_eq!(as_float(&row_acc[4].1), model.q[k_b_1 - 1]);
    let expected_logxq_b =
        safe_log10(model.dq[..k_b_1 - 1].iter().sum::<f64>());
    assert!((as_float(&row_acc[5].1) - expected_logxq_b).abs() <= 1.0e-14);
    let expected_logxq_ja =
        safe_log10(model.dq[..k_ja_1 - 1].iter().sum::<f64>());
    assert!((as_float(&row_acc[6].1) - expected_logxq_ja).abs() <= 1.0e-14);

    // 3. `PrimaryStateBuffer` snapshot preserves `k_below_const_q` across mutations/restore.
    let snap = PrimaryStateBuffer::from_model(&model);
    model.aux.k_below_const_q = 7;
    snap.restore_into(&mut model).expect("restore state");
    assert_eq!(model.aux.k_below_const_q, k_a_1);

    // 4. Zero mass-change step resets boundaries cleanly to surface `1` / `-99.0`.
    model.mstar_dot = 0.0;
    do_adjust_mass_step(&mut model, dt).expect("zero mdot step");
    assert_eq!(model.aux.k_below_const_q, 1);
    assert_eq!(model.aux.k_const_mass, 0);
    assert_eq!(as_int(&eval_one_col(&model, "k_below_const_q")), 1);
    assert_eq!(as_int(&eval_one_col(&model, "k_const_mass")), 1);
    assert_eq!(
        as_float(&eval_one_col(&model, "logxq_below_const_q")),
        neg99
    );
    assert_eq!(as_float(&eval_one_col(&model, "logxq_const_mass")), neg99);
}
