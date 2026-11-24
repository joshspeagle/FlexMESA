// SPDX-License-Identifier: GPL-3.0-or-later
//! The numerics mode is recorded in the `binary_history.data` header.
//!
//! This test switches the process-wide numerics mode, so it lives in its own
//! test binary. In a shared binary the switch races with other tests that
//! read the mode (issue #210).

use flexmesa_binary::{parse_binary_history_columns, BinarySystem};
use flexmesa_star::{
    load_zams_model, read_mesa_log_str, MesaHeaderValue, StarControls,
    StarJobControls,
};

#[test]
fn test_binary_history_header_records_numerics_mode() {
    let c1 = StarControls {
        initial_mass: 2.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut c2 = c1.clone();
    c2.initial_mass = 1.2;
    let s1 = load_zams_model(c1, StarJobControls::default()).expect("s1");
    let s2 = load_zams_model(c2, StarJobControls::default()).expect("s2");
    let sys = BinarySystem::new_two_stars(s1, s2, 2.5, 0.3);
    let specs =
        parse_binary_history_columns("edot\nedot_tidal\nedot_enhance\n", None)
            .expect("edot columns");

    for (mode, expected) in [
        (flexmesa_math::NumericsMode::Strict, "strict"),
        (flexmesa_math::NumericsMode::Fast, "fast"),
    ] {
        flexmesa_math::with_numerics_mode(mode, || {
            let hdr = flexmesa_binary::format_binary_history_file_header(
                &sys, &specs,
            );
            let tbl = read_mesa_log_str(&format!(
                "{hdr}{}\n",
                " 1".repeat(specs.len())
            ))
            .expect("parse binary_history header");
            assert_eq!(
                tbl.header_value("math_backend"),
                Some(&MesaHeaderValue::Text(String::from(expected)))
            );
        });
    }
}
