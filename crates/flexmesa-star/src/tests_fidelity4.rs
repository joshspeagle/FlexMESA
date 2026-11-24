// SPDX-License-Identifier: GPL-3.0-or-later
//! Unit and integration tests for Stage `Fidelity-4` ([#115](https://github.com/joshspeagle/FlexMESA/issues/115)):
//! `.mod` checkpoint round-trip with extended header scalars (`M_center`,
//! `cumulative_energy_error`) and optional zone columns (`w`, `et_RSP`,
//! `w_div_wc`, `alpha_RTI`), uniform rotation (`relax.rs`), and physical E2E
//! execution parity.

use flexmesa_const::M_SUN;

use crate::controls::{StarControls, StarJobControls};
use crate::model_io::{read_mod_str, write_mod_string};
use crate::relax::apply_uniform_rotation;
use crate::zams::load_zams_model;

#[test]
fn test_mod_io_extended_headers_and_columns_roundtrip() {
    let controls = StarControls {
        initial_mass: 1.0,
        rti_flag: true,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls.clone(), job.clone())
        .expect("1 Msun ZAMS should load");

    model.m_center = 0.05 * M_SUN;
    model.xmstar = model.mstar - model.m_center;
    model.r_center = 1.0e9;
    model.l_center = 1.0e32;
    model.aux.cumulative_energy_error = -1.25e41;
    for k in 0..model.nz {
        model.aux.alpha_rti[k] = 1.0e-3 * ((k + 1) as f64);
    }

    let mod_text = write_mod_string(&model).expect("write_mod_string");
    assert!(mod_text.contains("M_center"));
    assert!(mod_text.contains("cumulative_energy_error"));
    assert!(mod_text.contains("alpha_RTI"));

    let restored = read_mod_str(&mod_text, controls, job)
        .expect("read_mod_str should parse extended headers and alpha_RTI");
    assert!(
        (restored.m_center - model.m_center).abs() < 1.0e-8 * model.m_center
    );
    assert!(
        (restored.aux.cumulative_energy_error
            - model.aux.cumulative_energy_error)
            .abs()
            < 1.0e-8 * model.aux.cumulative_energy_error.abs()
    );
    for k in 0..model.nz {
        assert!(
            (restored.aux.alpha_rti[k] - model.aux.alpha_rti[k]).abs()
                < 1.0e-10
        );
    }
}

#[test]
fn test_read_mod_str_parses_w_and_w_div_wc_columns() {
    let mod_sample = "\
! sample mod with w and w_div_wc
           1 -- model

             M/Msun        1.0000000000000000E+00
       model_number                             5
           star_age        1.0000000000000000E+06
          initial_z        2.0000000000000000E-02
           n_shells                             2
           net_name   'basic.net'
            species                             8
               Teff        5.7770000000000000E+03
   year_for_time_step      1.0000000000000000E+03

        lnd                       lnT                       lnR                       L                         dq                        w                         w_div_wc                  h1                        he3                       he4                       c12                       n14                       o16                       ne20                      mg24
    1   -1.0000000000000000E+01    1.0000000000000000E+01    2.4966216312141033E+01    3.8280000000000000E+33    5.0000000000000000E-01    2.5000000000000000E+05    3.5000000000000000E-01    7.0000000000000000E-01    0.0000000000000000E+00    3.0000000000000000E-01    0.0000000000000000E+00    0.0000000000000000E+00    0.0000000000000000E+00    0.0000000000000000E+00    0.0000000000000000E+00
    2    2.0000000000000000E+00    1.6000000000000000E+01    2.4000000000000000E+01    1.9140000000000000E+33    5.0000000000000000E-01    1.2000000000000000E+05    1.5000000000000000E-01    7.0000000000000000E-01    0.0000000000000000E+00    3.0000000000000000E-01    0.0000000000000000E+00    0.0000000000000000E+00    0.0000000000000000E+00    0.0000000000000000E+00    0.0000000000000000E+00
";
    let parsed = read_mod_str(
        mod_sample,
        StarControls::default(),
        StarJobControls::default(),
    )
    .expect("read_mod_str should parse w and w_div_wc");
    assert!((parsed.aux.etrb[0] - 6.25e10).abs() < 1.0e-2);
    assert!((parsed.aux.etrb[1] - 1.44e10).abs() < 1.0e-2);
    assert!((parsed.aux.w_div_wc[0] - 0.35).abs() < 1.0e-12);
    assert!((parsed.aux.w_div_wc[1] - 0.15).abs() < 1.0e-12);
}

#[test]
fn test_uniform_rotation_sets_rotation_flag_and_w_div_wc() {
    let controls = StarControls::default();
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls, job.clone())
        .expect("1 Msun ZAMS should load");

    apply_uniform_rotation(&mut model, 1.0e-5);
    assert!(model.controls.rotation_flag);
    assert!(model.aux.w_div_wc[0] > 0.0);
}

#[test]
fn test_mod_load_resets_num_retries_by_default_and_initializes_dt_limit_ratio_old_zero()
{
    use crate::inlist::InlistBundle;
    use crate::inlist_runner::DefaultStarExtras;
    use crate::star_job_init::{
        init_or_update_model_from_bundle_outcome, InitModelOutcome,
    };
    use crate::timestep::eval_h211_factor;

    let controls = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut src_model =
        load_zams_model(controls.clone(), StarJobControls::default())
            .expect("1 Msun ZAMS should load");
    src_model.num_retries = 7;
    let mod_text = write_mod_string(&src_model).expect("write_mod_string");

    // 1. Default `StarJobControls` (`set_initial_number_retries = true`,
    //    `initial_number_retries = 0`) resets `.mod` `num_retries` to 0 and
    //    sets `dt_limit_ratio_old = 0.0` (`init.f90:509,696`,
    //    `run_star_support.f90:2009-2012`).
    let loaded_default =
        read_mod_str(&mod_text, controls.clone(), StarJobControls::default())
            .expect("read_mod_str default");
    assert_eq!(loaded_default.num_retries, 0);
    assert_eq!(loaded_default.dt_limit_ratio_old, 0.0);

    // 2. Custom `initial_number_retries` is applied when enabled.
    let job_custom = StarJobControls {
        set_initial_number_retries: true,
        initial_number_retries: 3,
        ..StarJobControls::default()
    };
    let loaded_custom = read_mod_str(&mod_text, controls.clone(), job_custom)
        .expect("read_mod_str custom");
    assert_eq!(loaded_custom.num_retries, 3);

    // 3. `set_initial_number_retries = false` preserves `.mod` header count.
    let job_preserve = StarJobControls {
        set_initial_number_retries: false,
        initial_number_retries: 0,
        ..StarJobControls::default()
    };
    let loaded_preserved =
        read_mod_str(&mod_text, controls.clone(), job_preserve)
            .expect("read_mod_str preserve");
    assert_eq!(loaded_preserved.num_retries, 7);

    // 4. First step after `.mod` load (`dt_limit_ratio_old == 0.0`) uses
    //    the 1st-order rule `1 / dt_limit_ratio` rather than H211b with a
    //    fabricated `1.0` prior ratio (`timestep.f90:261-278`).
    let ratio = 1.25;
    let (fac_zero, _, _) = eval_h211_factor(
        ratio,
        loaded_default.dt_limit_ratio_old,
        loaded_default.dt,
        loaded_default.dt_old,
        1.2,
        0.5,
    );
    let (fac_one, _, _) = eval_h211_factor(
        ratio,
        1.0,
        loaded_default.dt,
        loaded_default.dt_old,
        1.2,
        0.5,
    );
    assert!(
        (fac_zero - 1.0 / ratio).abs() <= 1.0e-15,
        "first step with dt_limit_ratio_old=0.0 must use 1/ratio = {}, got {}",
        1.0 / ratio,
        fac_zero
    );
    assert!(
        (fac_zero - fac_one).abs() > 0.1,
        "dt_limit_ratio_old=0.0 must differ sharply from H211b with prior ratio 1.0"
    );

    // 5. Inlist parsing + `star_job_init` respects `set_initial_number_retries`
    //    on a `.mod` load and on a model carried over from an earlier stage,
    //    which upstream loads from the saved `.mod` file: only a photo
    //    restart is a `restart` (`star/job/run_star_support.f90:433-442`).
    let tmp_dir = std::env::temp_dir().join(format!(
        "flexmesa_retry_reset_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let mod_path = tmp_dir.join("seed.mod");
    std::fs::write(&mod_path, &mod_text).expect("write seed.mod");

    let inlist_preserve = tmp_dir.join("inlist_preserve");
    std::fs::write(
        &inlist_preserve,
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'seed.mod'\n\
           set_initial_number_retries = .false.\n\
           initial_number_retries = 4\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n",
    )
    .expect("write inlist_preserve");
    let bundle_preserve = InlistBundle::from_file(&inlist_preserve)
        .expect("inlist_preserve strict parse");
    assert_eq!(
        bundle_preserve
            .validation_report
            .unsupported_controls
            .len(),
        0
    );
    assert!(!bundle_preserve.job.set_initial_number_retries);
    assert_eq!(bundle_preserve.job.initial_number_retries, 4);

    let mut extras = DefaultStarExtras;
    let out_preserve = init_or_update_model_from_bundle_outcome(
        &bundle_preserve,
        None,
        &mut extras,
    )
    .expect("init preserve");
    let InitModelOutcome::Model(m_preserve) = out_preserve else {
        panic!("expected Model outcome");
    };
    assert_eq!(m_preserve.num_retries, 7);

    // A model carried over from an earlier stage (`existing = Some(...)`) is
    // not a photo restart, so `set_initial_number_retries = .true.` sets
    // `num_retries` to `initial_number_retries`
    // (`star/job/run_star_support.f90:2009-2012`).
    let inlist_restart = tmp_dir.join("inlist_restart");
    std::fs::write(
        &inlist_restart,
        "&star_job\n\
           set_initial_number_retries = .true.\n\
           initial_number_retries = 0\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n",
    )
    .expect("write inlist_restart");
    let bundle_restart = InlistBundle::from_file(&inlist_restart)
        .expect("inlist_restart strict parse");
    let out_restart = init_or_update_model_from_bundle_outcome(
        &bundle_restart,
        Some(*m_preserve),
        &mut extras,
    )
    .expect("init restart");
    let InitModelOutcome::Model(m_restart) = out_restart else {
        panic!("expected Model outcome");
    };
    assert_eq!(m_restart.num_retries, 0);
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
