// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit and integration tests for Stage `Fidelity-4` ([#115](https://github.com/joshspeagle/FlexMESA/issues/115)):
//! `.mod` checkpoint round-trip with extended header scalars (`M_center`,
//! `cumulative_energy_error`) and optional zone columns (`w`, `et_RSP`,
//! `w_div_wc`, `alpha_RTI`), multi-step physical relaxation loops (`relax.rs`),
//! and physical E2E execution parity.

use flexmesa_const::M_SUN;

use crate::controls::{StarControls, StarJobControls};
use crate::model_io::{read_mod_str, write_mod_string};
use crate::relax::{
    apply_composition_relaxation, apply_uniform_rotation, relax_core_boundary,
    relax_from_profile_files, relax_irradiation_to_target,
    relax_mass_to_target, relax_tau_factor_to_target, relax_to_remove_h_env,
};
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
            species                             2
               Teff        5.7770000000000000E+03
   year_for_time_step      1.0000000000000000E+03

        lnd                       lnT                       lnR                       L                         dq                        w                         w_div_wc                  h1                        he4
    1   -1.0000000000000000E+01    1.0000000000000000E+01    2.4966216312141033E+01    3.8280000000000000E+33    5.0000000000000000E-01    2.5000000000000000E+05    3.5000000000000000E-01    7.0000000000000000E-01    3.0000000000000000E-01
    2    2.0000000000000000E+00    1.6000000000000000E+01    2.4000000000000000E+01    1.9140000000000000E+33    5.0000000000000000E-01    1.2000000000000000E+05    1.5000000000000000E-01    7.0000000000000000E-01    3.0000000000000000E-01
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
fn test_multi_step_relaxation_routines_preserve_physical_invariants() {
    let controls = StarControls::default();
    let mut job = StarJobControls::default();
    let mut model = load_zams_model(controls, job.clone())
        .expect("1 Msun ZAMS should load");

    // 1. Mass relaxation (`relax_mass_to_target`)
    relax_mass_to_target(&mut model, 1.5, 8, false);
    assert!(!model.aux.doing_relax);
    assert!((model.mstar - 1.5 * M_SUN).abs() < 1.0e-9 * M_SUN);
    assert!((model.controls.initial_mass - 1.5).abs() < 1.0e-12);

    // 2. H-envelope stripping (`relax_to_remove_h_env`)
    let species = model.species;
    let mid = model.nz / 2;
    for k in mid..model.nz {
        model.xa[k * species] = 0.0;
        model.xa[k * species + 1] = 0.98;
    }
    let m_before_strip = model.mstar;
    relax_to_remove_h_env(&mut model, 1.0e-4, 8, false);
    assert!(model.mstar < m_before_strip);
    assert!(model.mstar > 0.1 * M_SUN);

    // 3. Core boundary relaxation (`relax_core_boundary`)
    relax_core_boundary(&mut model, 0.02, 10.0, 1.0e-4, 8, true);
    assert!((model.m_center - 0.02 * M_SUN).abs() < 1.0e-8 * M_SUN);
    assert!(model.r_center > 0.0);

    // 4. Optical depth relaxation (`relax_tau_factor_to_target`)
    relax_tau_factor_to_target(&mut model, 100.0, 8);
    assert!((model.controls.tau_factor - 100.0).abs() < 1.0e-9);

    // 5. Irradiation relaxation (`relax_irradiation_to_target`)
    relax_irradiation_to_target(&mut model, 1.0e9, 300.0, 8);
    assert!((model.controls.irradiation_flux - 1.0e9).abs() < 1.0e-6);
    assert!(
        (model.controls.column_depth_for_irradiation - 300.0).abs() < 1.0e-9
    );
    assert!(model.aux.irradiation_heat[0] > 0.0);

    // 6. Uniform rotation & composition relaxation
    apply_uniform_rotation(&mut model, 1.0e-5);
    assert!(model.controls.rotation_flag);
    assert!(model.aux.w_div_wc[0] > 0.0);
    job.relax_y = true;
    job.new_y = 0.28;
    apply_composition_relaxation(&mut model, &job);
    assert!(!model.aux.doing_relax);
}

#[test]
fn test_relax_from_profile_files_interpolates_j_and_entropy() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("1 Msun ZAMS should load");

    let tmp_dir = std::env::temp_dir().join("flexmesa_fidelity4_relax_test");
    let _ = std::fs::create_dir_all(&tmp_dir);
    crate::tier5_extras::write_relax_profile_files(&model, &tmp_dir)
        .expect("write_relax_profile_files");

    relax_from_profile_files(&mut model, &tmp_dir);
    assert!(model.controls.rotation_flag);
    assert_eq!(model.j_rot.len(), model.nz);
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
