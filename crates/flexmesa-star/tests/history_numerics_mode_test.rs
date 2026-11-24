// SPDX-License-Identifier: GPL-3.0-or-later
//! The numerics mode is recorded in history, profile, `.mod` and photo output.
//!
//! This test switches the process-wide numerics mode, so it lives in its own
//! test binary. In a shared binary the switch races with other tests that
//! read the mode (issue #210).

use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_star::{
    default_star_history_specs, default_star_profile_specs,
    format_star_history_file_header, format_star_profile_text, load_zams_model,
    photo_numerics_mode, read_mesa_log_str, read_photo_bytes, set_hydro_vars,
    write_mod_string, write_photo_bytes, MesaHeaderValue, StarControls,
    StarJobControls,
};

#[test]
fn test_numerics_mode_recorded_in_history_profile_mod_and_photo() {
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

    let hist_specs = default_star_history_specs();
    let prof_specs = default_star_profile_specs();

    for (mode, expected) in [
        (NumericsMode::Strict, "strict"),
        (NumericsMode::Fast, "fast"),
    ] {
        with_numerics_mode(mode, || {
            let h_hdr = format_star_history_file_header(&model, &hist_specs);
            let h_tbl = read_mesa_log_str(&format!(
                "{h_hdr}{}\n",
                " 1".repeat(hist_specs.len())
            ))
            .expect("parse history header");
            assert_eq!(
                h_tbl.header_value("math_backend"),
                Some(&MesaHeaderValue::Text(String::from(expected)))
            );

            let p_txt = format_star_profile_text(&model, &prof_specs);
            let p_tbl = read_mesa_log_str(&p_txt).expect("parse profile");
            assert_eq!(
                p_tbl.header_value("math_backend"),
                Some(&MesaHeaderValue::Text(String::from(expected)))
            );

            let mod_txt = write_mod_string(&model).expect("write_mod_string");
            assert!(
                mod_txt
                    .contains(&format!("! flexmesa math_backend: {expected}")),
                "missing math_backend comment in .mod for {expected}"
            );

            let photo_bytes = write_photo_bytes(&model);
            assert_eq!(
                photo_numerics_mode(&photo_bytes).expect("photo mode"),
                mode
            );
            let other_mode = if mode == NumericsMode::Strict {
                NumericsMode::Fast
            } else {
                NumericsMode::Strict
            };
            with_numerics_mode(other_mode, || {
                let err = read_photo_bytes(
                    &photo_bytes,
                    model.controls.clone(),
                    model.job.clone(),
                )
                .expect_err("cross-mode .photo load must be rejected");
                assert!(
                    err.to_string().contains("photo numerics_mode mismatch"),
                    "unexpected error: {err}"
                );
            });
        });
    }
}
