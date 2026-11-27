// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#152` invariant tests for `flexmesa-star` core
//! boundaries, radial profiles, step histories, and surface-averaged report
//! utilities (`star/private/{report,profile_getval,history}.f90`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names
)]

use flexmesa_const::{MixingType, L_SUN, M_SUN, R_SUN};
use flexmesa_star::hydro_rotation::use_xh_to_update_i_rot_and_j_rot;
use flexmesa_star::mesa_history::{
    format_star_history_data_line_with_extra, format_star_history_file_header,
};
use flexmesa_star::report_utils::{equatorial_and_polar_r_cm, eval_ledd_erg_s};
use flexmesa_star::{
    change_model_network, compute_core_masses, eval_photosphere_info,
    eval_surf_avg_rotation_info, eval_upstream_omega_crit,
    evaluate_star_history_row, extract_history_row, format_star_profile_text,
    load_zams_model, parse_star_history_columns, parse_star_profile_columns,
    read_mesa_log_str, set_hydro_vars, set_rotation_info, MesaCellValue,
    MesaHeaderValue, MesaLogTable, OutputControls, StarControls,
    StarJobControls, StarModel,
};

fn load_zams_hydro(mass_msun: f64) -> StarModel {
    let ctrl = StarControls {
        initial_mass: mass_msun,
        initial_z: 0.02,
        zbase: 0.02,
        output: OutputControls {
            max_num_profile_zones: -1,
            ..OutputControls::default()
        },
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_old_values();
    model
}

fn rel_err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1.0e-300)
}

fn cell_float(row: &[(String, MesaCellValue)], col: &str) -> f64 {
    row.iter()
        .find_map(|(name, val)| {
            (name == col).then_some(match *val {
                MesaCellValue::Float(v) => v,
                MesaCellValue::Int(i) => i as f64,
            })
        })
        .unwrap_or_else(|| panic!("missing float column {col}"))
}

fn cell_int(row: &[(String, MesaCellValue)], col: &str) -> i64 {
    row.iter()
        .find_map(|(name, val)| match (name == col, *val) {
            (true, MesaCellValue::Int(i)) => Some(i),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing int column {col}"))
}

fn hdr_float(tbl: &MesaLogTable, key: &str) -> f64 {
    match tbl.header_value(key) {
        Some(&MesaHeaderValue::Float(v)) => v,
        other => panic!("expected Float header for {key}, got {other:?}"),
    }
}

fn build_nested_core_model(with_fe_core: bool) -> (StarModel, [usize; 4]) {
    let mut model = load_zams_hydro(1.0);
    change_model_network(&mut model, "approx21.net", None)
        .expect("switch to embedded approx21.net");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let nz = model.nz;
    assert!(nz >= 100);
    let iso_idx = |iso: &str| -> usize {
        model
            .iso_names
            .iter()
            .position(|n| n == iso)
            .unwrap_or_else(|| panic!("missing isotope {iso}"))
    };
    let (i_h1, i_he4, i_c12) = (iso_idx("h1"), iso_idx("he4"), iso_idx("c12"));
    let (i_o16, i_ne20, i_mg24, i_fe56) = (
        iso_idx("o16"),
        iso_idx("ne20"),
        iso_idx("mg24"),
        iso_idx("fe56"),
    );

    let bounds = [nz / 5, (2 * nz) / 5, (3 * nz) / 5, (4 * nz) / 5];
    for k in 0..nz {
        let cell = model.cell_xa_mut(k);
        cell.fill(0.0);
        if k < bounds[0] {
            cell[i_h1] = 0.70;
            cell[i_he4] = 0.28;
            cell[i_c12] = 0.02;
            model.ye[k] = 0.85;
        } else if k < bounds[1] {
            cell[i_he4] = 0.98;
            cell[i_c12] = 0.02;
            model.ye[k] = 0.50;
        } else if k < bounds[2] {
            cell[i_c12] = 0.50;
            cell[i_o16] = 0.50;
            model.ye[k] = 0.50;
        } else if k < bounds[3] || !with_fe_core {
            cell[i_c12] = 0.02;
            cell[i_o16] = 0.48;
            cell[i_ne20] = 0.30;
            cell[i_mg24] = 0.20;
            model.ye[k] = if k >= bounds[3] { 0.46 } else { 0.50 };
        } else {
            cell[i_fe56] = 1.0;
            model.ye[k] = 0.49;
        }
    }
    (model, bounds)
}

#[test]
fn test_core_boundaries_detects_nested_he_co_one_fe_and_neutron_rich_cores() {
    let (model, [k_he, k_co, k_one, k_fe]) = build_nested_core_model(true);

    // MESA `report.f90:1093-1188` keeps walking past `hit_co` and `hit_one`
    // so deeper `one_core` and `fe_core` zones remain reachable.
    let cores = compute_core_masses(&model);
    let hk = cores.he_core_k.expect("he_core_k must be present");
    let ck = cores.co_core_k.expect("co_core_k must be present");
    let ok = cores.one_core_k.expect("one_core_k must be present");
    let fk = cores.fe_core_k.expect("fe_core_k must be present");
    assert!((hk as isize - k_he as isize).abs() <= 1);
    assert!((ck as isize - k_co as isize).abs() <= 1);
    assert!((ok as isize - k_one as isize).abs() <= 1);
    assert!((fk as isize - k_fe as isize).abs() <= 1);
    assert!(hk < ck && ck < ok && ok < fk);
    assert!(cores.he_core_mass > cores.co_core_mass);
    assert!(cores.co_core_mass > cores.one_core_mass);
    assert!(cores.one_core_mass > cores.fe_core_mass);
    assert!(cores.fe_core_mass > 0.0);

    let specs = parse_star_history_columns(
        "he_core_mass\nco_core_mass\none_core_mass\nfe_core_mass\n\
         neutron_rich_core_mass\n",
        None,
        &model,
    )
    .expect("parse core mass history columns");
    let row = evaluate_star_history_row(&model, &specs)
        .expect("evaluate core mass history row");
    assert_eq!(cell_float(&row, "he_core_mass"), cores.he_core_mass);
    assert_eq!(cell_float(&row, "co_core_mass"), cores.co_core_mass);
    assert_eq!(cell_float(&row, "one_core_mass"), cores.one_core_mass);
    assert_eq!(cell_float(&row, "fe_core_mass"), cores.fe_core_mass);

    // Neutron-rich inner ONeMg core (`ye <= 0.48`) is also detected cleanly.
    let (m_nr, _) = build_nested_core_model(false);
    let cores_nr = compute_core_masses(&m_nr);
    assert!(cores_nr.he_core_mass > cores_nr.co_core_mass);
    assert!(cores_nr.co_core_mass > cores_nr.one_core_mass);
    assert!(cores_nr.one_core_mass >= cores_nr.neutron_rich_core_mass);
    assert!(cores_nr.neutron_rich_core_mass > 0.0);
    assert_eq!(cores_nr.fe_core_mass, 0.0);
    let row_nr = evaluate_star_history_row(&m_nr, &specs)
        .expect("evaluate neutron-rich core history row");
    assert_eq!(
        cell_float(&row_nr, "neutron_rich_core_mass"),
        cores_nr.neutron_rich_core_mass
    );
}

#[test]
fn test_profile_dm_eps_grav_and_mlt_arrays_match_profile_getval() {
    let mut model = load_zams_hydro(1.0);
    let nz = model.nz;
    let k = nz / 2;

    model.eps_grav_ad[k].val = 2.5e3;
    model.mixing_type[k] = MixingType::OvershootMixing;
    model.aux.mixing.mlt_mixing_type[k] = MixingType::SemiconvectiveMixing;
    model.aux.mixing.mlt_mixing_length[k] = 7.654_321e9;
    let alpha_hp = model.controls.mixing_length_alpha * model.scale_height[k];
    assert!(
        rel_err(model.aux.mixing.mlt_mixing_length[k], alpha_hp) > 1.0e-3,
        "test fixture requires mlt_mixing_length != alpha * scale_height"
    );

    let specs = parse_star_profile_columns(
        "zone\ndm_eps_grav\nmixing_type\nmlt_mixing_type\nmlt_mixing_length\n",
        None,
    )
    .expect("parse profile columns");
    let tbl = read_mesa_log_str(&format_star_profile_text(&model, &specs))
        .expect("parse profile text");
    assert_eq!(tbl.num_rows, nz);
    assert_eq!(tbl.column("zone").expect("zone")[k], (k + 1) as f64);

    let got_dm_eps = tbl.column("dm_eps_grav").expect("dm_eps_grav")[k];
    let want_dm_eps = model.eps_grav_ad[k].val * model.dm[k];
    assert!(
        rel_err(got_dm_eps, want_dm_eps) <= 1.0e-13,
        "dm_eps_grav mismatch: got {got_dm_eps:.16e}, want {want_dm_eps:.16e}"
    );
    assert_eq!(
        tbl.column("mixing_type").expect("mixing_type")[k],
        f64::from(MixingType::OvershootMixing.code())
    );
    assert_eq!(
        tbl.column("mlt_mixing_type").expect("mlt_mixing_type")[k],
        f64::from(MixingType::SemiconvectiveMixing.code())
    );
    let got_mlt_len =
        tbl.column("mlt_mixing_length").expect("mlt_mixing_length")[k];
    assert!(
        rel_err(got_mlt_len, 7.654_321e9) <= 1.0e-13,
        "mlt_mixing_length mismatch: got {got_mlt_len:.16e}"
    );
}

#[test]
fn test_profile_rotation_gating_fp_ft_and_r_e_div_r_p_match_profile_getval() {
    let mut model = load_zams_hydro(1.0);
    let nz = model.nz;
    let k = nz / 2;

    // Non-rotating with zeroed fp_rot/ft_rot (`relax.rs`): emits 1, 1, 0.
    model.controls.rotation_flag = false;
    model.aux.fp_rot.fill(0.0);
    model.aux.ft_rot.fill(0.0);

    let specs = parse_star_profile_columns(
        "zone\nfp_rot\nft_rot\nr_equatorial\nr_polar\nr_e_div_r_p\n",
        None,
    )
    .expect("parse rotation profile columns");
    let tbl_nonrot =
        read_mesa_log_str(&format_star_profile_text(&model, &specs))
            .expect("parse nonrot profile text");
    assert_eq!(tbl_nonrot.column("fp_rot").expect("fp_rot")[k], 1.0);
    assert_eq!(tbl_nonrot.column("ft_rot").expect("ft_rot")[k], 1.0);
    assert_eq!(
        tbl_nonrot.column("r_e_div_r_p").expect("r_e_div_r_p")[k],
        0.0
    );

    // Enable rotation with subcritical omega across all zones.
    model.controls.rotation_flag = true;
    for i in 0..nz {
        model.omega[i] = 0.5 * eval_upstream_omega_crit(&model, i);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut model);
    set_rotation_info(&mut model, false);
    assert!(model.aux.fp_rot[k] < 1.0 && model.aux.ft_rot[k] < 1.0);

    let (re_cm, rp_cm) = equatorial_and_polar_r_cm(&model, k);
    assert!(rp_cm > 1.0 && re_cm > rp_cm);

    let tbl_rot = read_mesa_log_str(&format_star_profile_text(&model, &specs))
        .expect("parse rot profile");
    assert!(
        rel_err(
            tbl_rot.column("fp_rot").expect("fp_rot")[k],
            model.aux.fp_rot[k],
        ) <= 1.0e-13
    );
    assert!(
        rel_err(
            tbl_rot.column("ft_rot").expect("ft_rot")[k],
            model.aux.ft_rot[k],
        ) <= 1.0e-13
    );
    assert!(
        rel_err(
            tbl_rot.column("r_equatorial").expect("r_equatorial")[k],
            re_cm / R_SUN,
        ) <= 1.0e-13
    );
    assert!(
        rel_err(
            tbl_rot.column("r_polar").expect("r_polar")[k],
            rp_cm / R_SUN,
        ) <= 1.0e-13
    );
    assert!(
        rel_err(
            tbl_rot.column("r_e_div_r_p").expect("r_e_div_r_p")[k],
            re_cm / rp_cm,
        ) <= 1.0e-13
    );

    // `profile_getval.f90:1512`: `r_polar <= 1.0` yields `0.0` while rotating.
    model.ln_r[nz - 1] = -2.0;
    model.r[nz - 1] = flexmesa_math::exp(-2.0);
    let (_, rp_tiny) = equatorial_and_polar_r_cm(&model, nz - 1);
    assert!(rp_tiny <= 1.0);
    let tbl_tiny = read_mesa_log_str(&format_star_profile_text(&model, &specs))
        .expect("parse tiny-r profile");
    assert_eq!(
        tbl_tiny.column("r_e_div_r_p").expect("r_e_div_r_p")[nz - 1],
        0.0
    );
}

#[test]
fn test_history_rotation_solver_steps_and_photosphere_roundtrip() {
    let mut model = load_zams_hydro(1.0);
    model.model_number = 5;
    model.aux.step_hist.num_rotation_solver_steps = 17;

    let specs = parse_star_history_columns(
        "model_number\nnum_zones\nstar_mass\nrotation_solver_steps\n",
        None,
        &model,
    )
    .expect("parse rotation_solver_steps history columns");
    let row = evaluate_star_history_row(&model, &specs)
        .expect("evaluate history row");
    assert_eq!(cell_int(&row, "rotation_solver_steps"), 17);

    let mut text = format_star_history_file_header(&model, &specs);
    text.push_str(&format_star_history_data_line_with_extra(&row, &[]));
    let tbl = read_mesa_log_str(&text).expect("parse history text");
    assert_eq!(tbl.num_rows, 1);
    assert_eq!(
        tbl.column("rotation_solver_steps")
            .expect("rotation_solver_steps")[0],
        17.0
    );

    let hrow = extract_history_row(&model);
    assert_eq!(hrow.num_zones, model.nz);
    assert!(rel_err(hrow.star_mass, model.mstar / M_SUN) <= 1.0e-15);

    let prof_hdr_tbl = read_mesa_log_str(&format_star_profile_text(
        &model,
        &parse_star_profile_columns("zone\n", None).expect("zone spec"),
    ))
    .expect("parse minimal profile for header");
    let phot = eval_photosphere_info(&model);
    assert!(phot.r_rsun > 0.0 && phot.l_lsun > 0.0 && phot.logg.is_finite());
    assert!(rel_err(phot.l_lsun, model.lum[0] / L_SUN) <= 1.0e-13);
    assert!(
        rel_err(hdr_float(&prof_hdr_tbl, "photosphere_L"), phot.l_lsun)
            <= 1.0e-13
    );
    assert!(
        rel_err(hdr_float(&prof_hdr_tbl, "photosphere_r"), phot.r_rsun)
            <= 1.0e-13
    );
}

#[test]
fn test_report_utils_surf_avg_tau_reads_canonical_output_controls() {
    let mut model = load_zams_hydro(1.0);
    model.controls.rotation_flag = true;
    model.controls.output.surf_avg_tau_min = 0.0;
    model.controls.output.surf_avg_tau = 500.0;
    model.controls.mix_wind.surf_avg_tau_min = 0.0;
    model.controls.mix_wind.surf_avg_tau = 500.0;

    let nz = model.nz;
    assert!(model.tau[0] < 5.0);
    assert!(
        (0..nz).any(|i| model.tau[i] > 5.0 && model.tau[i] < 500.0),
        "expected zones spanning 5 < tau < 500"
    );
    for i in 0..nz {
        let wc = eval_upstream_omega_crit(&model, i);
        if model.tau[i] <= 5.0 {
            model.opacity[i] = 0.5;
            model.omega[i] = 0.15 * wc;
        } else {
            model.opacity[i] = 5.0;
            model.omega[i] = 0.65 * wc;
        }
    }
    use_xh_to_update_i_rot_and_j_rot(&mut model);
    set_rotation_info(&mut model, false);

    let ledd_wide = eval_ledd_erg_s(&model);
    let rot_wide = eval_surf_avg_rotation_info(&model);

    // Changing only duplicate `controls.mix_wind.surf_avg_tau` leaves
    // `report_utils` unchanged.
    model.controls.mix_wind.surf_avg_tau = 5.0;
    assert_eq!(eval_ledd_erg_s(&model).to_bits(), ledd_wide.to_bits());
    assert_eq!(
        eval_surf_avg_rotation_info(&model).v_div_v_crit.to_bits(),
        rot_wide.v_div_v_crit.to_bits()
    );

    // Changing canonical `controls.output.surf_avg_tau` shifts both averages.
    model.controls.output.surf_avg_tau = 5.0;
    let ledd_thin = eval_ledd_erg_s(&model);
    let rot_thin = eval_surf_avg_rotation_info(&model);
    assert!(ledd_thin > ledd_wide);
    assert!(rot_thin.v_div_v_crit < rot_wide.v_div_v_crit);
}
