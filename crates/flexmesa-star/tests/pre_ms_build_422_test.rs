// SPDX-License-Identifier: LGPL-3.0-or-later
//! Regression test for GitHub issue `#422`: the pre-main-sequence builder
//! (`build_pre_ms_model`, `star/private/pre_ms_model.f90:34-288`) reproduces
//! MESA `26.4.1` bit for bit for `star/test_suite/make_zams`.
//!
//! Upstream `eval_gradT` (`star/private/pre_ms_model.f90:673-736`) evaluates
//! `get_gradT` on `auto_diff_real_star_order1` values, whose division is
//! `x*powm1(y)`; the builder's mixing-length arithmetic follows it.
//!
//! Expected values: MESA `26.4.1` (release tag `26.4.1`, MESA SDK
//! `x86_64-linux-26.6.1`, `gfortran 15.2.0`), run unmodified on
//! `make_zams/inlist_zams_header` with one diagnostic `write` statement added
//! after the final `set_qs` of `build1_pre_ms_model`
//! (`star/private/pre_ms_model.f90:651-656`), printing `xh(1:4, k)` of the
//! last `pre_ms_f` evaluation as IEEE-754 `Z16.16` hexadecimal. These are
//! upstream reference values, not `FlexMESA` output.

#![forbid(unsafe_code)]

use flexmesa_star::{
    build_pre_ms_model_with_configs, InlistBundle, InlistLoadOptions,
};
use flexmesa_utils::require_mesa_test_input_path;

/// MESA `26.4.1` built-model zone count for `make_zams`.
const MESA_NZ: usize = 1419;

/// `(k, lnd, lnT, lnR, L)` from the MESA `26.4.1` record, `k` 1-based.
const MESA_ZONES: [(usize, u64, u64, u64, u64); 3] = [
    (
        1,
        0xC032_8939_41B2_DE8E,
        0x4021_3021_772A_7C24,
        0x403D_9F4A_F4AB_0B2C,
        0x4797_5014_E0AA_3140,
    ),
    (
        2,
        0xC032_87B6_41F6_1B27,
        0x4021_31D1_4101_F0FA,
        0x403D_9F47_FE7D_30D0,
        0x4797_5014_7E99_C52D,
    ),
    (
        MESA_NZ,
        0xC024_73C5_2FEC_5E25,
        0x4029_38F3_A31C_6C03,
        0x4039_1CFD_2AB8_847F,
        0x468E_8ECD_2A5A_86F5,
    ),
];

#[test]
fn make_zams_pre_ms_build_matches_mesa_26_4_1_bit_for_bit() {
    let Some(inlist) = require_mesa_test_input_path(
        "star/test_suite/make_zams/inlist_zams_header",
    ) else {
        return;
    };
    let bundle = InlistBundle::from_file_with_options(
        &inlist,
        InlistLoadOptions::default(),
    )
    .expect("load make_zams/inlist_zams_header");
    assert!(bundle.job.create_pre_ms_model, "make_zams builds a pre-MS model");

    let model = build_pre_ms_model_with_configs(
        bundle.controls.clone(),
        bundle.job.clone(),
        bundle.eos_config,
        bundle.kap_config.clone(),
        Some(&bundle.case_dir),
    )
    .expect("build the make_zams pre-MS model");

    assert_eq!(model.nz, MESA_NZ, "built-model zone count");
    for (k, lnd, ln_t, ln_r, lum) in MESA_ZONES {
        let i = k - 1;
        let got = [model.lnd[i], model.ln_t[i], model.ln_r[i], model.lum[i]];
        let want = [lnd, ln_t, ln_r, lum];
        for (name, (g, w)) in
            ["lnd", "lnT", "lnR", "L"].iter().zip(got.iter().zip(want))
        {
            assert_eq!(
                g.to_bits(),
                w,
                "{name} at k = {k}: FlexMESA {:016X} ({g:e}), MESA {w:016X} ({:e})",
                g.to_bits(),
                f64::from_bits(w)
            );
        }
    }
}
