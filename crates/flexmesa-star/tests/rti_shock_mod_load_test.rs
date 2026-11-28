//! Loading `ccsn_IIp`'s RTI shock models (`standard_shock_part2.mod`,
//! `standard_shock_part3.mod`) with and without the control upstream uses for
//! them.
//!
//! Both files carry `alpha_RTI`, so `RTI_flag` is set at load
//! (`star/private/read_model.f90:351`) and `finish_load_model`'s `set_vars`
//! (`read_model.f90:162`) reaches `set_RTI_mixing_info`
//! (`star/private/mix_info.f90:83`). Upstream loads them only from stages that
//! read `inlist_shock_common`, which sets
//! `RTI_min_dm_behind_shock_for_full_on = 0.01` (line 51); with that the load
//! succeeds. With the default `0` the shock taper
//! `C*(shock_mass_start - m(k))/min_dm` (`mix_info.f90:2022-2023`) is infinite
//! outside the shock, `eta_RTI` is NaN where `alpha_RTI = 0`, and upstream
//! returns `ierr = -1` (`mix_info.f90:2033-2035`).

use std::path::PathBuf;

use flexmesa_star::{read_mod_str_with_dir, StarControls, StarJobControls};

fn require_zero_skips_enabled() -> bool {
    !matches!(
        std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

/// `$MESA_DIR/star/test_suite/ccsn_IIp`, else the cached test-input bundle.
fn ccsn_iip_dir() -> Option<PathBuf> {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    let dir = base.join("star").join("test_suite").join("ccsn_IIp");
    if dir.join("standard_shock_part2.mod").is_file() {
        return Some(dir);
    }
    assert!(
        !require_zero_skips_enabled(),
        "missing ccsn_IIp test inputs in `{}` (run `flexmesa data fetch \
         --include-test-inputs` or set FLEXMESA_REQUIRE_ZERO_SKIPS=0)",
        dir.display()
    );
    None
}

const SHOCK_MODS: [&str; 2] =
    ["standard_shock_part2.mod", "standard_shock_part3.mod"];

#[test]
fn shock_mods_load_with_shock_common_min_dm_and_fail_with_default() {
    let Some(dir) = ccsn_iip_dir() else {
        return;
    };
    for name in SHOCK_MODS {
        let text = std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|e| panic!("read {name}: {e}"));

        // The value upstream loads these models with
        // (`inlist_shock_common:51`).
        let controls = StarControls {
            rti_min_dm_behind_shock_for_full_on: 0.01,
            ..StarControls::default()
        };
        let model = read_mod_str_with_dir(
            &text,
            controls,
            StarJobControls::default(),
            Some(&dir),
        )
        .unwrap_or_else(|e| panic!("{name} with min_dm = 0.01: {e}"));
        assert!(model.nz >= 2, "{name}: nz = {}", model.nz);
        assert!(model.controls.rti_flag, "{name}: RTI_flag not set at load");

        // The default `0` divides by zero in the shock taper, as upstream.
        let err = read_mod_str_with_dir(
            &text,
            StarControls::default(),
            StarJobControls::default(),
            Some(&dir),
        )
        .expect_err("default RTI_min_dm_behind_shock_for_full_on must fail");
        assert!(
            err.to_string().contains("non-finite eta_RTI"),
            "{name}: unexpected error {err}"
        );
    }
}
