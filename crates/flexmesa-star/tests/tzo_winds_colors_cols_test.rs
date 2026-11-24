// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#435` regression tests unblocking `star/test_suite/tzo` (`#161`):
//! 1. Lazy `Dutch_wind_lowT_scheme` validation in namelist loading and
//!    `do_set_mdot` (`star/private/winds.f90:576-593`).
//! 2. MESA 26.4.1 `colors` history directives (`add_bc`, `add_abs_mag`,
//!    `add_lum_band`, `add_log_lum_band` as 0-column no-ops; single-column
//!    `bc`, `abs_mag`, `lum_band`, `log_lum_band` failing loudly per
//!    `star/private/history_specs.f90:237-271, 456-469, 675-690`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::{
    do_set_mdot, evaluate_star_history_row, load_zams_model,
    parse_star_history_columns, set_hydro_vars, set_mdot, DutchLowTScheme,
    StarControls, StarError, StarJobControls, StarModel, WindScheme,
};
use flexmesa_utils::require_mesa_test_input_path;

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_435_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp case dir");
        Self { root }
    }
}

impl Drop for TempCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn zams_model(mass_msun: f64) -> StarModel {
    let ctrl = StarControls {
        initial_mass: mass_msun,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

#[test]
fn test_tzo_inlist_common_and_non_dutch_winds_accept_other_lowt_scheme() {
    let tmp = TempCaseGuard::new("tzo_non_dutch");
    let p = tmp.root.join("inlist_tzo_winds");
    fs::write(
        &p,
        "&controls\n  cool_wind_RGB_scheme = 'van Loon'\n  \
         cool_wind_AGB_scheme = 'van Loon'\n  \
         Dutch_wind_lowT_scheme = 'other'\n  \
         hot_wind_scheme = 'Vink'\n/\n",
    )
    .expect("write inlist");
    let bundle = InlistBundle::from_file(&p)
        .expect("non-Dutch schemes must allow Dutch_wind_lowT_scheme='other'");
    assert_eq!(bundle.controls.cool_wind_rgb_scheme, "van Loon");
    assert_eq!(bundle.controls.cool_wind_agb_scheme, "van Loon");
    assert_eq!(bundle.controls.mix_wind.hot_wind_scheme, "Vink");
    assert_eq!(bundle.controls.mix_wind.dutch_wind_lowt_scheme, "other");

    if let Some(tzo_common) =
        require_mesa_test_input_path("star/test_suite/tzo/inlist_common")
    {
        let tzo_bundle = InlistBundle::from_file_with_options(
            &tzo_common,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .expect(
            "star/test_suite/tzo/inlist_common must parse without \
             dutch_wind_lowt_scheme='other' error",
        );
        assert_eq!(tzo_bundle.controls.cool_wind_rgb_scheme, "van Loon");
        assert_eq!(tzo_bundle.controls.cool_wind_agb_scheme, "van Loon");
        assert_eq!(tzo_bundle.controls.mix_wind.hot_wind_scheme, "Vink");
        assert_eq!(
            tzo_bundle.controls.mix_wind.dutch_wind_lowt_scheme,
            "other"
        );
    }
}

#[test]
fn test_inlist_load_rejects_invalid_dutch_lowt_only_when_dutch_wind_active() {
    let tmp = TempCaseGuard::new("dutch_active_inlist");

    for (idx, body) in [
        "&controls\n  hot_wind_scheme = 'Dutch'\n  \
         Dutch_wind_lowT_scheme = 'other'\n/\n",
        "&controls\n  cool_wind_RGB_scheme = 'Dutch'\n  \
         Dutch_wind_lowT_scheme = 'other'\n/\n",
        "&controls\n  cool_wind_AGB_scheme = 'dutch'\n  \
         Dutch_wind_lowT_scheme = 'other'\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_bad_{idx}"));
        fs::write(&p, body).expect("write inlist");
        let err = InlistBundle::from_file(&p)
            .expect_err("Dutch wind with 'other' lowT scheme must fail");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {body}: got {err}"
        );
    }

    for (idx, body) in [
        "&controls\n  Dutch_wind_lowT_scheme = 'other'\n/\n",
        "&controls\n  cool_wind_RGB_scheme = 'Reimers'\n  \
         hot_wind_scheme = 'Vink'\n  Dutch_wind_lowT_scheme = 'other'\n/\n",
        "&controls\n  hot_wind_scheme = 'Dutch'\n  \
         Dutch_wind_lowT_scheme = 'van Loon'\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_ok_{idx}"));
        fs::write(&p, body).expect("write inlist");
        InlistBundle::from_file(&p).expect("valid configuration must parse");
    }
}

#[test]
fn test_winds_driver_lazy_dutch_lowt_validation_tracks_active_branch() {
    let mut model = zams_model(5.0);
    model.controls.hot_wind_full_on_t = 12_000.0;
    model.controls.cool_wind_full_on_t = 8_000.0;
    model.controls.dutch_scaling_factor = 1.0;
    model.controls.vink_scaling_factor = 1.0;
    model.controls.mix_wind.van_loon_scaling_factor = 1.0;
    model.controls.mix_wind.dutch_wind_lowt_scheme = String::from("other");

    // 1. TZO-like non-Dutch schemes succeed regardless of Teff branch.
    model.controls.mix_wind.hot_wind_scheme = String::from("Vink");
    model.controls.cool_wind_rgb_scheme = String::from("van Loon");
    model.controls.cool_wind_agb_scheme = String::from("van Loon");
    for teff in [4_000.0, 10_000.0, 20_000.0] {
        model.teff = teff;
        do_set_mdot(&mut model).expect(
            "non-Dutch schemes must ignore unused dutch_wind_lowt_scheme",
        );
    }

    // 2. Hot scheme = 'Dutch' rejects 'other' only when hot branch is active.
    model.controls.mix_wind.hot_wind_scheme = String::from("Dutch");
    model.controls.cool_wind_rgb_scheme = String::from("van Loon");
    model.teff = 4_000.0;
    assert!(
        do_set_mdot(&mut model).is_ok(),
        "cool-only Teff must not evaluate hot Dutch branch"
    );
    model.teff = 20_000.0;
    assert!(matches!(
        do_set_mdot(&mut model),
        Err(StarError::InvalidInput(_))
    ));

    // 3. Cool RGB scheme = 'Dutch' rejects 'other' only when cool branch runs.
    model.controls.mix_wind.hot_wind_scheme = String::from("Vink");
    model.controls.cool_wind_rgb_scheme = String::from("Dutch");
    model.teff = 20_000.0;
    assert!(
        do_set_mdot(&mut model).is_ok(),
        "hot-only Teff must not evaluate cool Dutch branch"
    );
    model.teff = 6_000.0;
    assert!(matches!(
        do_set_mdot(&mut model),
        Err(StarError::InvalidInput(_))
    ));

    // 4. Valid `DutchLowTScheme` names succeed and preserve helper signatures.
    for (name, expected_variant) in [
        ("de Jager", DutchLowTScheme::DeJager),
        ("van Loon", DutchLowTScheme::VanLoon),
        ("Nieuwenhuijzen", DutchLowTScheme::Nieuwenhuijzen),
    ] {
        assert_eq!(DutchLowTScheme::parse_name(name), Some(expected_variant));
        model.controls.mix_wind.dutch_wind_lowt_scheme = String::from(name);
        let mdot = do_set_mdot(&mut model)
            .expect("valid Dutch_wind_lowT_scheme must evaluate");
        assert!(mdot.is_finite() && mdot <= 0.0);
        let direct =
            set_mdot(&mut model, WindScheme::Dutch, expected_variant, 1.0);
        assert!(direct.is_finite() && direct <= 0.0);
    }
}

#[test]
fn test_colors_history_directives_match_mesa_26_4_1_semantics_and_tzo_list() {
    let model = zams_model(1.0);

    // 1. All four `add_*` directives are 0-column no-ops in MESA 26.4.1.
    let baseline = parse_star_history_columns("model_number\n", None, &model)
        .expect("baseline column");
    let with_directives = parse_star_history_columns(
        "model_number\nadd_bc\nadd_abs_mag\nadd_lum_band\n\
         add_log_lum_band\nADD_ABS_MAG\n",
        None,
        &model,
    )
    .expect("colors add_* directives must parse cleanly as no-ops");
    assert_eq!(with_directives, baseline);

    // 2. Single-column legacy color lookups fail loudly (`get_bc_id_by_name = -1`).
    for bad in [
        "bc V\n",
        "abs_mag B\n",
        "lum_band J\n",
        "log_lum_band U\n",
        "bc_V\n",
        "abs_mag_B\n",
        "lum_band_J\n",
        "log_lum_band_U\n",
        "bc\n",
        "abs_mag\n",
        "lum_band\n",
        "log_lum_band\n",
    ] {
        let err = parse_star_history_columns(bad, None, &model)
            .expect_err("single-column legacy colors spec must fail loudly");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {bad:?}: got {err}"
        );
    }

    // 3. Upstream `star/test_suite/tzo/history_columns.list` parses completely
    //    and matches the exact column list with `add_abs_mag` removed.
    if let Some(tzo_hist) =
        require_mesa_test_input_path("star/test_suite/tzo/history_columns.list")
    {
        let text =
            fs::read_to_string(&tzo_hist).expect("read tzo history columns");
        let specs =
            parse_star_history_columns(&text, tzo_hist.parent(), &model)
                .expect("tzo/history_columns.list must parse cleanly");
        let stripped: String = text
            .lines()
            .filter(|line| line.trim() != "add_abs_mag")
            .collect::<Vec<_>>()
            .join("\n");
        let specs_without =
            parse_star_history_columns(&stripped, tzo_hist.parent(), &model)
                .expect("stripped tzo history columns");
        assert_eq!(specs, specs_without);

        let row =
            evaluate_star_history_row(&model, &specs).expect("history row");
        assert_eq!(row.len(), specs.len());
        for (col_name, _) in &row {
            assert!(
                !col_name.starts_with("abs_mag_")
                    && !col_name.starts_with("bc_")
                    && !col_name.starts_with("lum_band_")
                    && !col_name.starts_with("log_lum_band_"),
                "unexpected legacy color column emitted for tzo: {col_name}"
            );
        }
    }
}
