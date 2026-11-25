// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#147` fail-loudly validation tests for `flexmesa-astero`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_astero::inlist_astero::{
    parse_astero_search_controls, parse_astero_search_controls_with_options,
};
use flexmesa_astero::sample_eval::parse_gyre_outer_bound;
use flexmesa_star::{InlistLoadOptions, StarError};

struct TempAsteroGuard {
    root: PathBuf,
}

impl TempAsteroGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_ast_147_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp astero dir");
        Self { root }
    }
}

impl Drop for TempAsteroGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn test_astero_unsupported_unused_aliases_and_pgstar_validation() {
    let tmp = TempAsteroGuard::new("schema");
    let unimpl = tmp.root.join("unimpl");
    fs::create_dir_all(&unimpl).expect("mkdir");
    fs::write(
        unimpl.join("inlist_astero_search_controls"),
        "&astero_search_controls\n  unknown_non_mesa_key = .true.\n/\n",
    )
    .expect("write");
    let err = parse_astero_search_controls(&unimpl)
        .expect_err("unknown astero control must fail by default");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert!(err.to_string().contains("unknown_non_mesa_key"));
    assert!(
        parse_astero_search_controls_with_options(
            &unimpl,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .is_err(),
        "non-upstream key must fail even with allow_unsupported_controls"
    );

    let impl_dir = tmp.root.join("implemented_controls");
    fs::create_dir_all(&impl_dir).expect("mkdir");
    fs::write(
        impl_dir.join("inlist_astero_search_controls"),
        "&astero_search_controls
  write_fgong_for_each_model = .true.
  trace_limits = .true.
  gyre_non_ad = .true.
  y_frac_he3 = 2.0d-4
/\n",
    )
    .expect("write");
    let parsed = parse_astero_search_controls(&impl_dir)
        .expect("implemented astero controls must parse by default");
    assert!(parsed.write_fgong_for_each_model);
    assert!(parsed.trace_limits);
    assert!(parsed.gyre_non_ad);
    assert!((parsed.y_frac_he3 - 2.0e-4).abs() <= 1.0e-18);

    for (idx, body) in [
        "&astero_search_controls\n  a_div_r = 0.0\n/\n",
        "&astero_search_controls\n  nl0 = 10\n/\n",
        "&astero_search_controls\n  bobyqa_rho_beg = 0.1d0\n/\n",
        "&astero_search_controls\n  newuoa_max_fcn_calls = 50\n/\n",
        "&astero_search_controls\n  chi2_seismo_freq_fraction = 0.5d0\n/\n",
        "&astero_search_controls\n  solar_nu_max = 3100.0d0\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let d = tmp.root.join(format!("alias_{idx}"));
        fs::create_dir_all(&d).expect("mkdir");
        fs::write(d.join("inlist_astero_search_controls"), body).expect("w");
        assert!(
            parse_astero_search_controls_with_options(
                &d,
                InlistLoadOptions {
                    allow_unsupported_controls: true,
                },
            )
            .is_err(),
            "non-upstream astero alias must fail: {body}"
        );
    }
}

#[test]
fn test_astero_pgstar_controls_are_validated_only_when_pgstar_is_active() {
    let tmp = TempAsteroGuard::new("pgstar");
    let pg_ok = tmp.root.join("pg_ok");
    fs::create_dir_all(&pg_ok).expect("mkdir");
    fs::write(
        pg_ok.join("inlist_astero_search_controls"),
        "&astero_pgstar_controls\n  echelle_delta_nu = 135.0d0\n/\n",
    )
    .expect("w");
    assert!(parse_astero_search_controls(&pg_ok).is_ok());

    let pg_bad = tmp.root.join("pg_bad");
    fs::create_dir_all(&pg_bad).expect("mkdir");
    fs::write(
        pg_bad.join("inlist"),
        "&star_job\n  pgstar_flag = .true.\n/\n",
    )
    .expect("w");
    fs::write(
        pg_bad.join("inlist_astero_search_controls"),
        "&astero_pgstar_controls\n  bogus_astero_plot_key = 1\n/\n",
    )
    .expect("w");
    assert!(parse_astero_search_controls(&pg_bad).is_err());

    let pg_inactive = tmp.root.join("pg_inactive");
    fs::create_dir_all(&pg_inactive).expect("mkdir");
    fs::write(
        pg_inactive.join("inlist"),
        "&star_job\n  pgstar_flag = .false.\n/\n",
    )
    .expect("w");
    fs::write(
        pg_inactive.join("inlist_astero_search_controls"),
        "&astero_pgstar_controls\n  bogus_astero_plot_key = 1\n  echelle_delta_nu = 135.0d0\n/\n",
    )
    .expect("w");
    assert!(parse_astero_search_controls(&pg_inactive).is_ok());
}

#[test]
fn test_astero_option_strings_and_missing_files_fail_loudly() {
    let tmp = TempAsteroGuard::new("opts");
    for (idx, body) in [
        "&astero_search_controls\n  correction_scheme = 'bg14_1'\n/\n",
        "&astero_search_controls\n  correction_scheme = 'bg14_2'\n/\n",
        "&astero_search_controls\n  correction_scheme = 'bogus'\n/\n",
        "&astero_search_controls\n  search_type = 'bogus'\n/\n",
        "&astero_search_controls\n  oscillation_code = 'bogus'\n/\n",
        "&astero_search_controls\n  param_name(1) = 'bogus_p'\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let d = tmp.root.join(format!("opt_{idx}"));
        fs::create_dir_all(&d).expect("mkdir");
        fs::write(d.join("inlist_astero_search_controls"), body).expect("w");
        assert!(parse_astero_search_controls(&d).is_err());
    }

    assert!(parse_gyre_outer_bound(&tmp.root, "missing_gyre.in").is_err());
    fs::write(
        tmp.root.join("gyre_bad.in"),
        "&osc\n  outer_bound = 'BOGUS'\n/\n",
    )
    .expect("write");
    assert!(parse_gyre_outer_bound(&tmp.root, "gyre_bad.in").is_err());

    let from_file_dir = tmp.root.join("from_file_bad");
    fs::create_dir_all(&from_file_dir).expect("mkdir");
    fs::write(
        from_file_dir.join("inlist_astero_search_controls"),
        "&astero_search_controls\n  search_type = 'from_file'\n  filename_for_parameters = 'missing_params.txt'\n  param_name(1) = 'mass'\n  vary_param(1) = .true.\n  file_column_for_param(1) = 2\n/\n",
    )
    .expect("w");
    let ctrl = parse_astero_search_controls(&from_file_dir).expect("parse ok");
    assert!(ctrl.read_parameters_from_file(&from_file_dir).is_err());

    fs::write(
        from_file_dir.join("missing_params.txt"),
        "col1 col2\n1.0 not_a_float\n",
    )
    .expect("w");
    assert!(ctrl.read_parameters_from_file(&from_file_dir).is_err());

    fs::write(from_file_dir.join("missing_params.txt"), "col1 col2\n1.0\n")
        .expect("w");
    assert!(ctrl.read_parameters_from_file(&from_file_dir).is_err());
}
