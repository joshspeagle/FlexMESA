// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact IEEE-754 `pow3` multiplication-grouping unit tests for
//! `eval_surf_coefs` (`#581` part c).

#![forbid(unsafe_code)]

use flexmesa_astero::eval_surf_coefs;
use flexmesa_math::pow3;

#[test]
fn eval_surf_coefs_cubic_and_combined_match_primary_times_pow3_ratio() {
    let primary = -1.234_567_e-12_f64;
    let secondary = 2.5e-4_f64;
    let model_nu_max = 2_101.37_f64;
    let nu_max_sun = 3_090.0_f64;
    let ratio = 5_000.0_f64 * model_nu_max / nu_max_sun;

    let expected_c1 = primary * pow3(ratio);
    let naive_c1 = ((primary * ratio) * ratio) * ratio;
    assert_ne!(
        expected_c1.to_bits(),
        naive_c1.to_bits(),
        "witness inputs must separate primary * pow3(ratio) from left-to-right",
    );

    let (cubic_c1, cubic_c2) =
        eval_surf_coefs("cubic", primary, secondary, model_nu_max, nu_max_sun);
    assert_eq!(cubic_c1.to_bits(), expected_c1.to_bits());
    assert_eq!(cubic_c2.to_bits(), 0.0_f64.to_bits());

    let (comb_c1, comb_c2) = eval_surf_coefs(
        "combined",
        primary,
        secondary,
        model_nu_max,
        nu_max_sun,
    );
    assert_eq!(comb_c1.to_bits(), expected_c1.to_bits());
    assert_eq!(comb_c2.to_bits(), (secondary / ratio).to_bits());
}
