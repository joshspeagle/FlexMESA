// SPDX-License-Identifier: GPL-3.0-or-later
//! `chi2_delta_nu` depends only on `delta_nu_sigma > 0` (`#511`,
//! `astero/private/extras_support.f90:236-241` at MESA 26.4.1).

#![forbid(unsafe_code)]

use flexmesa_astero::inlist_astero::{AsteroSearchControls, MAX_CONSTRAINTS};
use flexmesa_astero::sample_eval::eval_spectro_and_delta_nu_chi2;
use flexmesa_star::{load_zams_model, StarControls, StarJobControls};

#[test]
fn chi2_delta_nu_is_computed_without_a_positive_delta_nu_target() {
    let model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.0 Msun ZAMS should load");
    let values = [f64::NAN; MAX_CONSTRAINTS];

    // Upstream computes the term whenever delta_nu_sigma > 0, also for the
    // default target delta_nu = -1.
    let mut ctrl = AsteroSearchControls::default();
    ctrl.delta_nu = -1.0;
    ctrl.delta_nu_sigma = 2.0;
    let (_, chi2_dnu, dnu_model) =
        eval_spectro_and_delta_nu_chi2(&model, &ctrl, &values);
    assert!(dnu_model > 0.0);
    let d = (ctrl.delta_nu - dnu_model) / ctrl.delta_nu_sigma;
    assert_eq!(chi2_dnu, d * d);

    // delta_nu_sigma <= 0 gives 0 whatever the target.
    let mut ctrl_off = AsteroSearchControls::default();
    ctrl_off.delta_nu = 135.0;
    ctrl_off.delta_nu_sigma = -1.0;
    let (_, chi2_off, _) =
        eval_spectro_and_delta_nu_chi2(&model, &ctrl_off, &values);
    assert_eq!(chi2_off, 0.0);
}
