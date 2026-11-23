// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for MESA `26.4.1` `&colors` namelist defaults, recursive
//! includes, strict validation, and history photometry evaluation (`#454`,
//! `#161`, `#162`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_colors::{
    ColorsContext, ColorsLoadConfig, SedCube, SedInterpMethod, D_10PC_CM,
    Z_OVER_X_REF_GS98,
};
use flexmesa_const::{L_SUN, R_SUN, STANDARD_CGRAV};
use flexmesa_schema::is_known_upstream_only_section;
use flexmesa_star::controls::ColorsControls;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_inventory::supported_star_control_inventory;
use flexmesa_star::mesa_history::{
    evaluate_star_history_row, parse_star_history_columns,
    resolve_star_history_columns, HistoryColumnSpec, IncrementalHistoryWriter,
};
use flexmesa_star::report_utils::StepReportContext;
use flexmesa_star::{StarControls, StarError, StarJobControls, StarModel};

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_colors_test_{tag}_{}_{id}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn solar_surface_model(colors: ColorsControls) -> StarModel {
    let ctrl = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        colors,
        ..StarControls::default()
    };
    let cat = flexmesa_chem::NuclideCatalog::builtin_ref();
    let h1 = cat.get_nuclide_index("h1").expect("h1 id");
    let he4 = cat.get_nuclide_index("he4").expect("he4 id");
    let mut model =
        StarModel::new(2, vec![h1, he4], StarJobControls::default(), ctrl);
    model.model_number = 7;
    let r0 = R_SUN;
    let g0 = flexmesa_math::pow(10.0, 4.44);
    let m0 = g0 * r0 * r0 / STANDARD_CGRAV;
    model.mstar = m0;
    model.t = vec![5778.0, 6200.0];
    model.r = vec![r0, 0.99 * r0];
    model.m = vec![m0, 0.99 * m0];
    model.cgrav = vec![STANDARD_CGRAV, STANDARD_CGRAV];
    model.grav = vec![g0, g0];
    model.dq = vec![0.01, 0.99];
    model.opacity = vec![1.0, 1.0];
    model.rho = vec![1.0e-7, 1.0e-6];
    model.lum = vec![L_SUN, L_SUN];
    // Choose X and Z so (Z / X) == Z_OVER_X_REF_GS98 -> [M/H] == 0.0 dex.
    let x = 0.70;
    let z = x * Z_OVER_X_REF_GS98;
    let y = 1.0 - x - z;
    model.xa = vec![x, y, x, y];
    model
}

fn write_synthetic_colors_fixture(dir: &Path) {
    let inst_dir = dir.join("my_inst");
    let atm_dir = dir.join("my_atm");
    fs::create_dir_all(&inst_dir).expect("inst dir");
    fs::create_dir_all(&atm_dir).expect("atm dir");

    fs::write(inst_dir.join("my_inst"), "Band1.dat\nBand2.dat\n")
        .expect("write index");
    fs::write(
        inst_dir.join("Band1.dat"),
        "Wavelength,Transmission\n3000.0,0.0\n4000.0,1.0\n5000.0,0.0\n",
    )
    .expect("write Band1");
    fs::write(
        inst_dir.join("Band2.dat"),
        "Wavelength,Transmission\n5000.0,0.0\n6000.0,1.0\n7000.0,0.0\n",
    )
    .expect("write Band2");

    fs::write(
        dir.join("vega.csv"),
        "WAVELENGTH,FLUX\n2000.0,1.0e-8\n3000.0,1.0e-8\n4000.0,1.0e-8\n\
         5000.0,1.0e-8\n6000.0,1.0e-8\n7000.0,1.0e-8\n8000.0,1.0e-8\n",
    )
    .expect("write vega");

    fs::write(
        atm_dir.join("lookup_table.csv"),
        "#file_name, teff\r\nsed_lo.txt,4000.0\r\nsed_hi.txt,8000.0\r\n",
    )
    .expect("write lookup_table");
    let sed_lo =
        "# synthetic 4000K\n2000.0 1.0e5\n3000.0 1.5e5\n4000.0 2.0e5\n\
                  5000.0 2.0e5\n6000.0 2.0e5\n7000.0 1.5e5\n8000.0 1.0e5\n";
    let sed_hi =
        "# synthetic 8000K\n2000.0 4.0e5\n3000.0 6.0e5\n4000.0 8.0e5\n\
                  5000.0 8.0e5\n6000.0 8.0e5\n7000.0 6.0e5\n8000.0 4.0e5\n";
    fs::write(atm_dir.join("sed_lo.txt"), sed_lo).expect("write sed_lo");
    fs::write(atm_dir.join("sed_hi.txt"), sed_hi).expect("write sed_hi");
}

#[test]
fn test_colors_defaults_and_schema_inventory_parity() {
    assert!(!is_known_upstream_only_section("colors"));
    let d = ColorsControls::default();
    assert!(!d.use_colors);
    assert_eq!(d.instrument, "data/colors_data/filters/Generic/Johnson");
    assert_eq!(
        d.stellar_atm,
        "data/colors_data/stellar_models/Kurucz2003all/"
    );
    assert_eq!(d.vega_sed, "data/colors_data/stellar_models/vega_flam.csv");
    assert!((d.distance - D_10PC_CM).abs() <= 1.0e-15);
    assert!(!d.make_csv);
    assert_eq!(d.colors_results_directory, "SED");
    assert_eq!(d.mag_system, "Vega");
    assert!(!d.sed_per_model);
    assert!((d.z_over_x_ref - Z_OVER_X_REF_GS98).abs() <= 1.0e-15);

    let inv = supported_star_control_inventory();
    assert_eq!(inv.colors_supported.len(), 10);
    assert_eq!(inv.colors_unused.len(), 0);
    assert_eq!(inv.colors_terminal_output.len(), 0);
}

#[test]
fn test_colors_namelist_recursive_includes_and_loud_validation() {
    let dir = TempDirGuard::new("includes");
    fs::write(
        dir.path().join("inlist"),
        "&star_job\n/\n&colors\n  use_colors = .true.\n\
         read_extra_colors_inlist(1) = .true.\n\
         extra_colors_inlist_name(1) = 'inlist_c1'\n/\n&controls\n/\n",
    )
    .expect("write entry inlist");
    fs::write(
        dir.path().join("inlist_c1"),
        "&colors\n  mag_system = 'AB'\n  make_csv = .true.\n\
         read_extra_colors_inlist(2) = .true.\n\
         extra_colors_inlist_name(2) = 'inlist_c2'\n/\n",
    )
    .expect("write inlist_c1");
    fs::write(
        dir.path().join("inlist_c2"),
        "&colors\n  sed_per_model = .true.\n\
         colors_results_directory = 'custom_sed'\n  distance = 6.1714d19\n/\n",
    )
    .expect("write inlist_c2");

    let bundle = InlistBundle::from_file_in_case_dir(dir.path(), "inlist")
        .expect("recursive &colors includes must parse cleanly");
    assert_eq!(bundle.validation_report.unsupported_sections.len(), 0);
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
    let cc = &bundle.controls.colors;
    assert!(cc.use_colors);
    assert_eq!(cc.mag_system, "AB");
    assert!(cc.make_csv);
    assert!(cc.sed_per_model);
    assert_eq!(cc.colors_results_directory, "custom_sed");
    assert!((cc.distance - 6.1714e19).abs() <= 1.0e5);

    let bad_dir = TempDirGuard::new("bad_ctrl");
    fs::write(
        bad_dir.path().join("inlist_typo"),
        "&colors\n  bogus_colors_key = 1\n/\n",
    )
    .expect("write typo");
    let err = InlistBundle::from_file_with_options(
        &bad_dir.path().join("inlist_typo"),
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect_err("unknown &colors key must always fail");
    assert!(matches!(err, StarError::InvalidInput(_)));

    for (tag, body) in [
        (
            "bad_mag",
            "&colors\n  use_colors = .true.\n  mag_system = 'Bogus'\n/\n",
        ),
        (
            "bad_dist",
            "&colors\n  use_colors = .true.\n  distance = 0.0\n/\n",
        ),
        (
            "bad_zref",
            "&colors\n  use_colors = .true.\n  z_over_x_ref = -0.02\n/\n",
        ),
        (
            "empty_inst",
            "&colors\n  use_colors = .true.\n  instrument = ''\n/\n",
        ),
    ] {
        let p = bad_dir.path().join(format!("inlist_{tag}"));
        fs::write(&p, body).expect("write bad inlist");
        let err = InlistBundle::from_file(&p)
            .expect_err("invalid &colors control must fail loudly");
        assert!(matches!(err, StarError::InvalidInput(_)), "got {err}");
    }

    let miss_inc = bad_dir.path().join("inlist_miss_inc");
    fs::write(
        &miss_inc,
        "&colors\n  read_extra_colors_inlist(1) = .true.\n\
         extra_colors_inlist_name(1) = 'no_such_file'\n/\n",
    )
    .expect("write missing include");
    let err = InlistBundle::from_file(&miss_inc)
        .expect_err("missing extra_colors_inlist must fail");
    assert!(matches!(err, StarError::IoError(_)), "got {err}");

    let bad_cfg_ctrl = ColorsControls {
        use_colors: true,
        instrument: String::from("data/colors_data/filters/NO_SUCH_INST"),
        ..ColorsControls::default()
    };
    let bad_model = solar_surface_model(bad_cfg_ctrl);
    let load_err =
        resolve_star_history_columns(&bad_model, Some(bad_dir.path()))
            .expect_err("missing custom instrument must fail loudly");
    assert!(matches!(load_err, StarError::IoError(_)), "got {load_err}");
}

#[test]
fn test_colors_custom_synthetic_atm_csv_export_and_extra_col_order() {
    let dir = TempDirGuard::new("synthetic");
    write_synthetic_colors_fixture(dir.path());

    // Guard check: 1D cube evaluates cleanly without panicking on dummy axes.
    let cube_1d = SedCube::from_dense(
        vec![4000.0, 8000.0],
        vec![-999.0],
        vec![0.0],
        vec![2000.0, 4000.0, 6000.0, 8000.0],
        vec![1.0e5, 2.0e5, 2.0e5, 1.0e5, 4.0e5, 8.0e5, 8.0e5, 4.0e5],
    )
    .expect("1D cube");
    let sed_1d = cube_1d.eval_sed(
        6000.0,
        4.4,
        0.0,
        R_SUN,
        D_10PC_CM,
        SedInterpMethod::Hermite,
    );
    assert_eq!(sed_1d.len(), 4);
    assert!(sed_1d.iter().all(|v| v.is_finite() && *v > 0.0));

    let colors = ColorsControls {
        use_colors: true,
        instrument: String::from("./my_inst"),
        stellar_atm: String::from("./my_atm"),
        vega_sed: String::from("./vega.csv"),
        distance: D_10PC_CM,
        make_csv: true,
        colors_results_directory: String::from("out_sed"),
        mag_system: String::from("Vega"),
        sed_per_model: true,
        z_over_x_ref: Z_OVER_X_REF_GS98,
    };
    let model = solar_surface_model(colors);

    // `#435` regression: legacy add_* history directives remain 0-col no-ops.
    let base_parsed =
        parse_star_history_columns("model_number\n", None, &model)
            .expect("parse base");
    let noop_parsed = parse_star_history_columns(
        "model_number\nadd_bc\nadd_abs_mag\nadd_lum_band\nadd_log_lum_band\n",
        None,
        &model,
    )
    .expect("parse add_*");
    assert_eq!(base_parsed, noop_parsed);

    fs::write(
        dir.path().join("history_columns.list"),
        "model_number\nstar_mass\n",
    )
    .expect("write local history_columns.list");
    let specs = resolve_star_history_columns(&model, Some(dir.path()))
        .expect("resolve colors columns");
    let spec_names: Vec<String> =
        specs.iter().map(HistoryColumnSpec::column_name).collect();
    assert_eq!(
        spec_names,
        vec![
            "model_number",
            "star_mass",
            "Mag_bol",
            "Flux_bol",
            "Interp_rad",
            "Band1",
            "Band2",
        ]
    );

    let hist_out = dir.path().join("LOGS").join("history.data");
    let mut hw = IncrementalHistoryWriter::new(hist_out.clone(), specs);
    let ctx = StepReportContext::compute(&model);
    let extra = vec![(String::from("rsp_extra_col"), 42.5)];
    hw.write_step_with_ctx_and_extras(&model, &ctx, &[], &extra)
        .expect("write history row + CSVs");
    drop(hw);

    let text = fs::read_to_string(&hist_out).expect("read history.data");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 7);
    let header_cols: Vec<&str> = lines[5].split_whitespace().collect();
    assert_eq!(
        header_cols,
        vec![
            "model_number",
            "star_mass",
            "rsp_extra_col",
            "Mag_bol",
            "Flux_bol",
            "Interp_rad",
            "Band1",
            "Band2",
        ]
    );

    let csv1 = dir.path().join("out_sed").join("Band1_SED_00000007.csv");
    assert!(csv1.is_file(), "expected {}", csv1.display());
    assert!(dir
        .path()
        .join("out_sed")
        .join("Band2_SED_00000007.csv")
        .is_file());
    let csv1_text = fs::read_to_string(&csv1).expect("read csv1");
    assert!(csv1_text.starts_with(
        "wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans\n"
    ));
}

#[test]
fn test_colors_ab_st_and_flat_csv_export() {
    let dir = TempDirGuard::new("ab_st");
    write_synthetic_colors_fixture(dir.path());
    for sys in ["AB", "ST"] {
        let cfg = ColorsLoadConfig {
            instrument: "./my_inst",
            stellar_atm: "./my_atm",
            vega_sed: "./vega.csv",
            mag_system: sys,
            distance: D_10PC_CM,
            z_over_x_ref: Z_OVER_X_REF_GS98,
            make_csv: sys == "ST",
            sed_per_model: false,
            colors_results_directory: "out_sed_flat",
        };
        let c_ctx =
            ColorsContext::load(&cfg, Some(dir.path())).expect("load ctx");
        let out = c_ctx
            .eval_and_maybe_write_csv(5778.0, 4.44, 0.7, 0.016, R_SUN, 1, true)
            .expect("eval ctx");
        assert!(out.filter_mags.iter().all(|v| v.is_finite()));
    }
    assert!(dir
        .path()
        .join("out_sed_flat")
        .join("Band1_SED.csv")
        .is_file());
}

#[test]
fn test_builtin_solar_johnson_parity() {
    let cc = ColorsControls {
        use_colors: true,
        ..ColorsControls::default()
    };
    let model = solar_surface_model(cc);
    let dir = TempDirGuard::new("builtin_solar");
    fs::write(dir.path().join("history_columns.list"), "model_number\n")
        .expect("write list");
    let specs = resolve_star_history_columns(&model, Some(dir.path()))
        .expect("resolve default Johnson");
    let row = evaluate_star_history_row(&model, &specs);
    let names: Vec<&str> = row.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "model_number",
            "Mag_bol",
            "Flux_bol",
            "Interp_rad",
            "B",
            "I",
            "J",
            "M",
            "R",
            "U",
            "V",
        ]
    );
    let expected = [
        ("Mag_bol", 16.233_783_742_415_f64),
        ("Flux_bol", 3.209_863_046_816_5e-07),
        ("Interp_rad", 1.201_509_822_140_3e-02),
        ("B", 5.432_831_508_849_5),
        ("I", 4.024_363_240_540_5),
        ("J", 3.654_466_716_232_9),
        ("M", 3.279_404_031_070_8),
        ("R", 4.332_242_343_580_0),
        ("U", 5.534_292_813_775_7),
        ("V", 4.776_981_134_694_0),
    ];
    for (col, gold) in expected {
        let val = row
            .iter()
            .find(|(k, _)| k == col)
            .expect("missing col")
            .1
            .as_f64();
        let rel = (val - gold).abs() / gold.abs().max(1.0);
        assert!(
            rel < 1.0e-10,
            "{col}: got {val}, expected {gold} (rel {rel})"
        );
    }
}

fn assert_upstream_colors_case(
    case_dir: &Path,
    entry_inlist: &str,
    strict_controls: bool,
    expect_cols: &[&str],
) {
    let opts = InlistLoadOptions {
        allow_unsupported_controls: !strict_controls,
    };
    let bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        entry_inlist,
        opts,
    )
    .expect("upstream &colors case must load cleanly");
    assert_eq!(bundle.validation_report.unsupported_sections.len(), 0);
    let colors_unsupported: Vec<_> = bundle
        .validation_report
        .unsupported_controls
        .iter()
        .filter(|(sec, _)| sec == "colors")
        .collect();
    assert_eq!(colors_unsupported.len(), 0);
    if strict_controls {
        assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
    }
    assert!(bundle.controls.colors.use_colors);
    let mut m = solar_surface_model(bundle.controls.colors.clone());
    m.job = bundle.job.clone();
    let resolved = resolve_star_history_columns(&m, Some(case_dir))
        .expect("resolve upstream colors columns");
    let resolved_names: Vec<String> = resolved
        .iter()
        .map(HistoryColumnSpec::column_name)
        .collect();
    let tail = &resolved_names[resolved_names.len() - expect_cols.len()..];
    let tail_refs: Vec<&str> = tail.iter().map(String::as_str).collect();
    assert_eq!(tail_refs, expect_cols);
}

#[test]
fn test_upstream_colors_cases_load_and_columns() {
    let Some(mesa_dir) = std::env::var_os("MESA_DIR") else {
        return;
    };
    let mesa_root = PathBuf::from(mesa_dir);
    let gaia_cols = [
        "Mag_bol",
        "Flux_bol",
        "Interp_rad",
        "G",
        "Gbp",
        "Grp",
        "Grvs",
    ];
    let johnson_cols = [
        "Mag_bol",
        "Flux_bol",
        "Interp_rad",
        "B",
        "I",
        "J",
        "M",
        "R",
        "U",
        "V",
    ];
    let cases: [(&str, &str, bool, &[&str]); 7] = [
        (
            "star/test_suite/custom_colors",
            "inlist",
            true,
            &johnson_cols,
        ),
        (
            "star/test_suite/rsp_BEP",
            "inlist_rsp_BEP_header",
            true,
            &gaia_cols,
        ),
        (
            "star/test_suite/rsp_RR_Lyrae",
            "inlist_rsp_RR_Lyrae_header",
            true,
            &gaia_cols,
        ),
        (
            "star/test_suite/rsp_BLAP",
            "inlist_rsp_BLAP_header",
            true,
            &gaia_cols,
        ),
        (
            "star/test_suite/rsp_Cepheid",
            "inlist_rsp_Cepheid_header",
            true,
            &gaia_cols,
        ),
        (
            "star/test_suite/rsp_save_and_load_file",
            "inlist_rsp_save_file_header",
            false,
            &gaia_cols,
        ),
        (
            "star/test_suite/rsp_save_and_load_file",
            "inlist_rsp_load_file_header",
            false,
            &gaia_cols,
        ),
    ];
    for (case_rel, entry_inlist, strict_controls, expect_cols) in cases {
        let case_dir = mesa_root.join(case_rel);
        if case_dir.is_dir() {
            assert_upstream_colors_case(
                &case_dir,
                entry_inlist,
                strict_controls,
                expect_cols,
            );
        }
    }
}
