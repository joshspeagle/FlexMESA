// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for [`flexmesa_star::pulse_fgong_data`] (`#171` Package
//! A3): in-memory `star_get_pulse_data_fgong` / `get_fgong_data`, faithful
//! `build_atm` (`T_tau`), double points, error checks, and optional
//! `FLEXMESA_ADIPLS_CAPTURES` parity against MESA `26.4.1` `store_*.fgong.bin`
//! captures (`astero_glue_tap.inc` binary layout).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp)]

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use flexmesa_const::{MixingType, R_SUN, STANDARD_CGRAV};
use flexmesa_math::log;
use flexmesa_star::job_controls::{AtmOption, AtmTTauOpacity, AtmTTauRelation};
use flexmesa_star::model_io::read_mod_str_with_dir;
use flexmesa_star::pulse_fgong_data::{
    get_fgong_data, star_get_pulse_data_fgong, FgongDataError, FgongDataInput,
    FgongPulseData, FGONG_ICONST, FGONG_IVAR,
};
use flexmesa_star::{
    eval_photosphere_info, load_zams_model, set_hydro_vars, InlistBundle,
    InlistLoadOptions, StarControls, StarJobControls, StarModel,
};

fn build_hydrated_solar_zams() -> StarModel {
    let mut controls = StarControls {
        zbase: 0.02,
        calculate_brunt_n2: true,
        atm_option: AtmOption::TTau,
        atm_t_tau_relation: AtmTTauRelation::Eddington,
        atm_t_tau_opacity: AtmTTauOpacity::Fixed,
        ..StarControls::default()
    };
    controls.calculate_brunt_n2 = true;
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
    model
}

#[test]
fn test_fgong_data_structure_variants_and_flag_combinations() {
    let model = build_hydrated_solar_zams();
    let nz = model.nz;
    let phot = eval_photosphere_info(&model);
    let r_outer = R_SUN * phot.r_rsun;

    // 1. Default asteroseismic store flags: add_center=true, keep_surf=false, add_atm=false
    let fg_def = star_get_pulse_data_fgong(&model, true, false, false)
        .expect("default fgong");
    assert_eq!(fg_def.nn, nz);
    assert_eq!(fg_def.point_data.len(), nz);
    assert_eq!(fg_def.point_var_flat().len(), FGONG_IVAR * nz);
    assert_eq!(fg_def.global_data_vec().len(), FGONG_ICONST);

    // Check global_data(1:15)
    assert_eq!(fg_def.global_data[0], model.m_grav[0]);
    assert!((fg_def.global_data[1] - r_outer).abs() <= 1.0e-12 * r_outer);
    assert_eq!(fg_def.global_data[2], model.lum[0]);
    assert_eq!(fg_def.global_data[3], model.controls.initial_z);
    let expected_y = if model.controls.initial_y < 0.0 {
        (0.24 + 2.0 * model.controls.initial_z).clamp(0.0, 1.0)
    } else {
        model.controls.initial_y
    };
    assert_eq!(
        fg_def.global_data[4],
        1.0 - (expected_y + model.controls.initial_z)
    );
    assert_eq!(fg_def.global_data[5], model.controls.mixing_length_alpha);
    for &val in &fg_def.global_data[6..=9] {
        assert_eq!(val, 0.0);
    }
    assert!(fg_def.global_data[10] < 0.0 && fg_def.global_data[10].is_finite());
    assert!(fg_def.global_data[11] < 0.0 && fg_def.global_data[11].is_finite());
    assert_eq!(fg_def.global_data[12], model.star_age);
    assert_eq!(fg_def.global_data[13], model.teff);
    assert_eq!(fg_def.global_data[14], STANDARD_CGRAV);

    // Verify monotonic radii, center point properties, and unused rows
    assert!(fg_def.point_data[0][0] < model.r[0]);
    for j in 1..fg_def.nn {
        assert!(
            fg_def.point_data[j][0] < fg_def.point_data[j - 1][0],
            "radius not strictly decreasing at j={j}"
        );
    }
    let ctr = fg_def.point_data.last().expect("center row");
    assert_eq!(ctr[0], 0.0);
    assert_eq!(ctr[1], log(f64::MIN_POSITIVE));
    assert_eq!(ctr[6], 0.0);
    assert_eq!(ctr[14], 0.0);
    assert_eq!(ctr[17], fg_def.global_data[1]);

    for (j, col) in fg_def.point_data.iter().enumerate() {
        assert_eq!(col[19], 0.0, "unused FGONG row 20 non-zero at j={j}");
        for (offset, &val) in col[36..FGONG_IVAR].iter().enumerate() {
            assert_eq!(
                val,
                0.0,
                "unused FGONG row {} non-zero at j={j}",
                37 + offset
            );
        }
        assert!(col[2] > 0.0 && col[3] > 0.0 && col[4] > 0.0);
        assert!(col[9] > 1.0 && col[9] < 2.0);
        assert!(
            col[25].is_finite() && col[26].is_finite() && col[27].is_finite()
        );
    }

    // 2. keep_surface_point=true, add_center_point=false
    let fg_surf = star_get_pulse_data_fgong(&model, false, true, false)
        .expect("keep_surf fgong");
    assert_eq!(fg_surf.nn, nz);
    assert_eq!(fg_surf.point_data[0][0], model.r[0]);
    assert!(fg_surf.point_data.last().expect("last")[0] > 0.0);

    // 3. Both keep_surface_point=true and add_center_point=true
    let fg_both = star_get_pulse_data_fgong(&model, true, true, false)
        .expect("both endpoints fgong");
    assert_eq!(fg_both.nn, nz + 1);
}

#[test]
fn test_fgong_data_atmosphere_t_tau_modes_and_table_loud_error() {
    let base_model = build_hydrated_solar_zams();
    let base_no_atm =
        star_get_pulse_data_fgong(&base_model, true, false, false)
            .expect("no-atm");

    for opacity_mode in [
        AtmTTauOpacity::Fixed,
        AtmTTauOpacity::Iterated,
        AtmTTauOpacity::Varying,
    ] {
        let mut model = base_model.clone();
        model.controls.atm_option = AtmOption::TTau;
        model.controls.atm_t_tau_relation = AtmTTauRelation::Eddington;
        model.controls.atm_t_tau_opacity = opacity_mode;
        let fg = star_get_pulse_data_fgong(&model, true, false, true)
            .expect("T_tau atmosphere fgong");
        assert!(
            fg.nn > base_no_atm.nn,
            "add_atmosphere=true ({opacity_mode:?}) must prepend atm points"
        );
        let n_atm_added = fg.nn - base_no_atm.nn;
        // Outermost atmospheric point sits above r(1) with m_grav(1)/m_outer == 1.
        assert!(fg.point_data[0][0] > model.r[0]);
        assert_eq!(fg.point_data[0][1], 0.0);
        assert_eq!(fg.point_data[0][6], model.lum[0]);
        assert_eq!(fg.point_data[0][8], 0.0);
        assert_eq!(fg.point_data[0][15], 0.0);
        assert_eq!(fg.point_data[0][18], 0.0);
        // Atmosphere-to-envelope transition at k=1.. preserves monotonic r.
        for j in 1..=n_atm_added {
            assert!(
                fg.point_data[j][0] < fg.point_data[j - 1][0],
                "non-monotonic r across atm/env transition at j={j}"
            );
        }
        // Envelope interior points remain identical when atm is added.
        assert_eq!(fg.point_data[n_atm_added], base_no_atm.point_data[0]);
    }

    // Unsupported atm_option (`Table`, `SimplePhotosphere`) must fail loudly.
    for bad_opt in [AtmOption::Table, AtmOption::SimplePhotosphere] {
        let mut model = base_model.clone();
        model.controls.atm_option = bad_opt;
        let err = star_get_pulse_data_fgong(&model, true, false, true)
            .expect_err("non-T_tau build_atm must fail loudly");
        assert!(
            matches!(err, FgongDataError::UnsupportedOption(_)),
            "unexpected error variant for {bad_opt:?}: {err}"
        );
    }
}

#[test]
fn test_fgong_data_double_points_rho_mode_zero_a_and_error_checks() {
    let mut model = build_hydrated_solar_zams();
    let nz = model.nz;

    // 1. interpolate_rho_for_pulse_data = false switches to eval_face_rho
    let fg_interp = star_get_pulse_data_fgong(&model, true, false, false)
        .expect("interp rho");
    let mut input_shell_rho = FgongDataInput::from_model(&model);
    input_shell_rho.interpolate_rho_for_pulse_data = false;
    let fg_shell = get_fgong_data(&input_shell_rho, true, false, false)
        .expect("shell-mass rho");
    assert_eq!(fg_shell.nn, fg_interp.nn);
    assert!(
        (fg_shell.point_data[10][4] - fg_interp.point_data[10][4]).abs() > 0.0
    );
    assert!(
        fg_shell.point_data[10][4].is_finite()
            && fg_shell.point_data[10][4] > 0.0
    );

    // 2. Sharp mu jump + double-point placement (unlimited and capped to 1)
    let k_split = nz / 2;
    for mu_k in &mut model.mu[k_split..nz] {
        *mu_k *= 1.4;
    }
    let mut input_dp = FgongDataInput::from_model(&model);
    input_dp.add_double_points_to_pulse_data = true;
    input_dp.threshold_grad_mu_for_double_point = 0.05;
    input_dp.max_number_of_double_points = 0;
    let fg_dp_all =
        get_fgong_data(&input_dp, true, false, false).expect("double pts all");
    assert!(fg_dp_all.nn > nz);

    input_dp.max_number_of_double_points = 1;
    let fg_dp_one =
        get_fgong_data(&input_dp, true, false, false).expect("double pt cap=1");
    assert_eq!(fg_dp_one.nn, nz + 1);
    let r_dup = model.r[k_split];
    let dup_count = fg_dp_one
        .point_data
        .iter()
        .filter(|col| (col[0] - r_dup).abs() <= 1.0e-12 * r_dup.max(1.0))
        .count();
    assert_eq!(dup_count, 2, "expected double point pair at k_split");

    // 3. fgong_zero_A_inside_r zeroes A* inside convective regions only
    let mut input_zero_a = FgongDataInput::from_model(&model);
    let k_conv = (1..(nz - 1))
        .find(|&k| model.mixing_type[k] != MixingType::NoMixing)
        .expect("solar ZAMS has convective zone");
    input_zero_a.fgong_zero_a_inside_r = (model.r[k_conv] / R_SUN) + 1.0e-4;
    let fg_za = get_fgong_data(&input_zero_a, true, false, false)
        .expect("zero_A fgong");
    let row_conv = &fg_za.point_data[k_conv - 1];
    assert_eq!(row_conv[14], 0.0);

    // 4. Loud errors on disabled calculate_brunt_n2, nz < 2, and bad slices
    let mut bad_brunt = model.clone();
    bad_brunt.controls.calculate_brunt_n2 = false;
    assert!(matches!(
        star_get_pulse_data_fgong(&bad_brunt, true, false, false),
        Err(FgongDataError::InvalidInput(_))
    ));

    let mut bad_nz = FgongDataInput::from_model(&model);
    bad_nz.nz = 1;
    assert!(matches!(
        get_fgong_data(&bad_nz, true, false, false),
        Err(FgongDataError::InvalidInput(_))
    ));

    let mut bad_slice = FgongDataInput::from_model(&model);
    bad_slice.dq = &model.dq[..nz - 1];
    assert!(matches!(
        get_fgong_data(&bad_slice, true, false, false),
        Err(FgongDataError::InvalidInput(_))
    ));
}

#[derive(Debug)]
struct CapturedFgongBin {
    iconst: usize,
    ivar: usize,
    nn: usize,
    global_data: Vec<f64>,
    point_data: Vec<f64>,
}

fn read_captured_fgong_bin(path: &Path) -> CapturedFgongBin {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!("failed to open {}: {e}", path.display());
    });
    let mut reader = BufReader::new(file);
    let mut raw = Vec::new();
    reader
        .read_to_end(&mut raw)
        .unwrap_or_else(|e| panic!("read failed {}: {e}", path.display()));
    assert!(raw.len() >= 12, "truncated header in {}", path.display());
    let iconst =
        usize::try_from(i32::from_le_bytes(raw[0..4].try_into().unwrap()))
            .expect("non-negative iconst");
    let ivar =
        usize::try_from(i32::from_le_bytes(raw[4..8].try_into().unwrap()))
            .expect("non-negative ivar");
    let nn =
        usize::try_from(i32::from_le_bytes(raw[8..12].try_into().unwrap()))
            .expect("non-negative nn");
    let expected_bytes = 12 + 8 * iconst + 8 * ivar * nn;
    assert_eq!(
        raw.len(),
        expected_bytes,
        "byte size mismatch in {}",
        path.display()
    );
    let mut global_data = Vec::with_capacity(iconst);
    for i in 0..iconst {
        let off = 12 + 8 * i;
        global_data
            .push(f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
    }
    let mut point_data = Vec::with_capacity(ivar * nn);
    let base = 12 + 8 * iconst;
    for i in 0..(ivar * nn) {
        let off = base + 8 * i;
        point_data
            .push(f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
    }
    CapturedFgongBin {
        iconst,
        ivar,
        nn,
        global_data,
        point_data,
    }
}

fn load_and_hydrate_captured_mod(
    case_dir: &Path,
    mod_path: &Path,
    alpha_override: Option<f64>,
) -> StarModel {
    let bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        "inlist",
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .unwrap_or_else(|e| {
        panic!("inlist load failed {}: {e}", case_dir.display())
    });
    let mod_text = std::fs::read_to_string(mod_path).unwrap_or_else(|e| {
        panic!("read mod failed {}: {e}", mod_path.display())
    });
    let mut controls = bundle.controls;
    controls.calculate_brunt_n2 = true;
    if let Some(alpha) = alpha_override {
        controls.mixing_length_alpha = alpha;
    }
    let mut model =
        read_mod_str_with_dir(&mod_text, controls, bundle.job, Some(case_dir))
            .unwrap_or_else(|e| {
                panic!("read_mod failed {}: {e}", mod_path.display())
            });
    model.eos_config = bundle.eos_config;
    model.kap_config = bundle.kap_config;
    set_hydro_vars(&mut model, false, true).unwrap_or_else(|e| {
        panic!("set_hydro_vars failed {}: {e}", mod_path.display())
    });
    model
}

fn assert_fgong_matches_capture(
    label: &str,
    got: &FgongPulseData,
    cap: &CapturedFgongBin,
    allow_atm_step_diff: bool,
) {
    assert_eq!(cap.iconst, FGONG_ICONST, "{label}: iconst");
    assert_eq!(cap.ivar, FGONG_IVAR, "{label}: ivar");

    for (idx, (&g, &c)) in got
        .global_data
        .iter()
        .zip(cap.global_data.iter())
        .enumerate()
    {
        if idx == 4 {
            // Skip global_data(5) if case ran without explicit initial_y match
            let denom = c.abs().max(1.0);
            assert!(
                (g - c).abs() / denom <= 1.0e-3,
                "{label}: global[{idx}] got={g} cap={c}"
            );
            continue;
        }
        let tol = if idx == 10 || idx == 11 {
            1.0e-2
        } else {
            5.0e-4
        };
        let denom = c.abs().max(1.0e-30);
        assert!(
            (g - c).abs() / denom <= tol,
            "{label}: global[{idx}] got={g:.12e} cap={c:.12e}"
        );
    }

    if allow_atm_step_diff {
        // When `add_atmosphere = true`, `nn_atm` comes from `build_atm`.
        assert!(
            got.nn.abs_diff(cap.nn) <= 4,
            "{label}: atm nn got={} cap={}",
            got.nn,
            cap.nn
        );
        // Check atmosphere outermost point (k=0) and center point (last)
        for row in [0usize, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13] {
            let g = got.point_data[0][row];
            let c = cap.point_data[row];
            let rel = (g - c).abs() / c.abs().max(1.0e-30);
            assert!(
                rel <= 2.0e-2,
                "{label} atm top row={row}: got={g:.10e} cap={c:.10e} rel={rel:.3e}"
            );
        }
        let g_ctr = got.point_data.last().expect("ctr");
        let c_ctr_base = (cap.nn - 1) * FGONG_IVAR;
        assert_eq!(g_ctr[0], 0.0);
        assert_eq!(g_ctr[1], cap.point_data[c_ctr_base + 1]);
        for row in [2usize, 3, 4, 5, 9, 10, 11, 12, 13, 17, 25, 26, 27] {
            let g = g_ctr[row];
            let c = cap.point_data[c_ctr_base + row];
            let rel = (g - c).abs() / c.abs().max(1.0e-12);
            assert!(
                rel <= 5.0e-3,
                "{label} ctr row={row}: got={g:.10e} cap={c:.10e} rel={rel:.3e}"
            );
        }
    } else {
        assert_eq!(got.nn, cap.nn, "{label}: nn");
        // Verify envelope-plus-center FGONG columns match closely.
        // Rows 8/15/18 (eps, r_X, eps_grav) depend on timestep `eps_grav_ad`
        // not preserved in `.mod` headers, so compare structure/EOS rows here.
        let check_rows = [
            0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 25, 26,
            27,
        ];
        for j in 0..got.nn {
            for &row in &check_rows {
                let g = got.point_data[j][row];
                let c = cap.point_data[j * FGONG_IVAR + row];
                if c == 0.0 && g == 0.0 {
                    continue;
                }
                let scale = c.abs().max(1.0e-12);
                let rel = (g - c).abs() / scale;
                // Surface/boundary interpolation or Gamma1 finite diffs may show
                // tiny hydrated-model deltas from `.mod` serialization.
                assert!(
                    rel <= 2.0e-2 || (g - c).abs() <= 1.0e-4,
                    "{label} j={j} row={row}: got={g:.10e} cap={c:.10e} rel={rel:.3e}"
                );
            }
        }
    }
}

#[test]
fn test_fgong_data_parity_against_a0_captures() {
    let raw = std::env::var("FLEXMESA_ADIPLS_CAPTURES").ok();
    let path_str = raw.as_deref().map_or("", str::trim);
    if path_str.is_empty() {
        eprintln!(
            "INFO: FLEXMESA_ADIPLS_CAPTURES unset; skipping external A3 FGONG captures parity test"
        );
        return;
    }
    let root = PathBuf::from(path_str);
    assert!(
        root.is_dir(),
        "FLEXMESA_ADIPLS_CAPTURES path `{}` is not a directory",
        root.display()
    );

    // 1. astero_adipls (`add_atmosphere = true`, `T_tau` Eddington)
    {
        let case_dir = root.join("astero_adipls");
        let model = load_and_hydrate_captured_mod(
            &case_dir,
            &case_dir
                .join("pulse_tap/models/adipls_eval_000001_step_0000050.mod"),
            None,
        );
        let got = star_get_pulse_data_fgong(&model, true, false, true)
            .expect("astero_adipls fgong");
        let cap = read_captured_fgong_bin(
            &case_dir.join("a0tap/store_000001.fgong.bin"),
        );
        assert_fgong_matches_capture("astero_adipls#1", &got, &cap, true);
    }

    // 2. surface_effects (`add_atmosphere = false`, `atm_option = 'table'`)
    {
        let case_dir = root.join("surface_effects");
        let model = load_and_hydrate_captured_mod(
            &case_dir,
            &case_dir
                .join("pulse_tap/models/adipls_eval_000001_step_0000954.mod"),
            None,
        );
        let got = star_get_pulse_data_fgong(&model, true, false, false)
            .expect("surface_effects fgong");
        let cap = read_captured_fgong_bin(
            &case_dir.join("a0tap/store_000001.fgong.bin"),
        );
        assert_fgong_matches_capture("surface_effects#1", &got, &cap, false);
    }

    // 3. example_astero (`add_atmosphere = false`, search sample alpha)
    {
        let case_dir = root.join("example_astero");
        let model = load_and_hydrate_captured_mod(
            &case_dir,
            &case_dir
                .join("pulse_tap/models/adipls_eval_000001_step_0000700.mod"),
            Some(1.692_825),
        );
        let got = star_get_pulse_data_fgong(&model, true, false, false)
            .expect("example_astero fgong");
        let cap = read_captured_fgong_bin(
            &case_dir.join("a0tap/store_000001.fgong.bin"),
        );
        assert_fgong_matches_capture("example_astero#1", &got, &cap, false);
    }

    // 4. fast_scan_grid (`add_atmosphere = false`, grid sample alpha)
    {
        let case_dir = root.join("fast_scan_grid");
        let model = load_and_hydrate_captured_mod(
            &case_dir,
            &case_dir
                .join("pulse_tap/models/adipls_eval_000001_step_0000194.mod"),
            Some(1.7),
        );
        let got = star_get_pulse_data_fgong(&model, true, false, false)
            .expect("fast_scan_grid fgong");
        let cap = read_captured_fgong_bin(
            &case_dir.join("a0tap/store_000001.fgong.bin"),
        );
        assert_fgong_matches_capture("fast_scan_grid#1", &got, &cap, false);
    }

    // 5. fast_simplex (`add_atmosphere = false`, simplex sample alpha)
    {
        let case_dir = root.join("fast_simplex");
        let model = load_and_hydrate_captured_mod(
            &case_dir,
            &case_dir
                .join("pulse_tap/models/adipls_eval_000001_step_0000179.mod"),
            Some(1.7),
        );
        let got = star_get_pulse_data_fgong(&model, true, false, false)
            .expect("fast_simplex fgong");
        let cap = read_captured_fgong_bin(
            &case_dir.join("a0tap/store_000001.fgong.bin"),
        );
        assert_fgong_matches_capture("fast_simplex#1", &got, &cap, false);
    }
}
