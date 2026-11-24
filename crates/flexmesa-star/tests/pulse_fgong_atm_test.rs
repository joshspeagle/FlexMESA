// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for [`flexmesa_star::pulse_fgong`] atmospheric attachment
//! (`add_atmosphere = true` via `build_atm` and `store_point_data_atm`), loud
//! unsupported/invalid error checks, `#746` `eos_proc` `log10(pgas) * LN10`
//! consistency, and optional `FLEXMESA_ADIPLS_CAPTURES` parity on
//! `astero_adipls` (`a0tap/store_000001.fgong.bin`).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp)]

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use flexmesa_atm::radiation_pressure;
use flexmesa_const::{LN10, RSUN, STANDARD_CGRAV};
use flexmesa_eos::{eos_dt_get, HelmTable, I_LN_PGAS};
use flexmesa_math::log10;
use flexmesa_star::job_controls::{AtmOption, AtmTTauOpacity, AtmTTauRelation};
use flexmesa_star::model_io::read_mod_str_with_dir;
use flexmesa_star::pulse_fgong::{get_fgong_data, PulseDataInput};
use flexmesa_star::pulse_gyre_data::GyreDataError;
use flexmesa_star::pulse_write::{FGONG_ICONST, FGONG_IVAR};
use flexmesa_star::{
    eval_photosphere_info, load_zams_model, set_hydro_vars, InlistBundle,
    InlistLoadOptions, StarControls, StarJobControls, StarModel,
};

fn build_hydrated_solar_zams() -> StarModel {
    let controls = StarControls {
        zbase: 0.02,
        calculate_brunt_n2: true,
        atm_option: AtmOption::TTau,
        atm_t_tau_relation: AtmTTauRelation::Eddington,
        atm_t_tau_opacity: AtmTTauOpacity::Fixed,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
    model
}

#[test]
fn test_fgong_atm_t_tau_modes_monotonicity_interior_identity_and_eos_consistency(
) {
    let base_model = build_hydrated_solar_zams();
    let base_input = PulseDataInput::from_model(&base_model).expect("input");
    assert_eq!(
        base_input
            .atm
            .expect("atm input")
            .atm_build_dlogtau
            .to_bits(),
        0x3F84_7AE1_4000_0000,
        "atm_build_dlogtau must match MESA controls.defaults real literal"
    );
    let (base_g, base_pd) =
        get_fgong_data(&base_input, true, false, false).expect("no-atm fgong");
    assert_eq!(base_g.len(), FGONG_ICONST);
    assert_eq!(base_pd.len(), base_model.nz);
    assert_eq!(base_g[14], STANDARD_CGRAV);
    let phot = eval_photosphere_info(&base_model);
    let r_outer = RSUN * phot.r_rsun;
    assert!((base_g[1] - r_outer).abs() <= 1.0e-12 * r_outer);

    let ht = HelmTable::builtin().expect("helm table");
    let xa_surf = &base_model.xa[0..base_model.species];

    for opacity_mode in [
        AtmTTauOpacity::Fixed,
        AtmTTauOpacity::Iterated,
        AtmTTauOpacity::Varying,
    ] {
        let mut model = base_model.clone();
        model.controls.atm_option = AtmOption::TTau;
        model.controls.atm_t_tau_relation = AtmTTauRelation::Eddington;
        model.controls.atm_t_tau_opacity = opacity_mode;

        let input = PulseDataInput::from_model(&model).expect("atm input");
        let (g, pd) = get_fgong_data(&input, true, false, true)
            .expect("T_tau atmosphere fgong");
        assert_eq!(g, base_g, "global_data changed with add_atmosphere=true");
        assert!(
            pd.len() > base_pd.len(),
            "add_atmosphere=true ({opacity_mode:?}) must prepend atm points"
        );
        let n_atm_added = pd.len() - base_pd.len();

        // Outermost atmospheric point sits above r[0] with q=1 (lnq=0).
        assert!(pd[0][0] > model.r[0]);
        assert_eq!(pd[0][1], 0.0);
        assert_eq!(pd[0][6], model.lum[0]);
        assert_eq!(pd[0][8], 0.0);
        assert_eq!(pd[0][15], 0.0);
        assert_eq!(pd[0][18], 0.0);

        // Radius must stay strictly decreasing across atm and atm->env boundary.
        for j in 1..pd.len() {
            assert!(
                pd[j][0] < pd[j - 1][0],
                "non-monotonic r at j={j} ({opacity_mode:?}): prev={} cur={}",
                pd[j - 1][0],
                pd[j][0]
            );
        }
        assert_eq!(pd.last().expect("center")[0], 0.0);

        // Interior envelope and center rows stay bit-identical to add_atmosphere=false.
        assert_eq!(&pd[n_atm_added..], base_pd.as_slice());

        // Guard #746: atmosphere EOS pressure uses log10(pgas) * LN10.
        for (j, row) in pd[..n_atm_added].iter().enumerate() {
            let t = row[2];
            let p_tot = row[3];
            let rho = row[4];
            let p_gas = p_tot - radiation_pressure(t);
            assert!(
                p_gas > 0.0 && t > 0.0 && rho > 0.0,
                "invalid atm thermodynamics at j={j} ({opacity_mode:?})"
            );
            let eos = eos_dt_get(
                ht,
                rho,
                t,
                &model.chem_id,
                xa_surf,
                &model.eos_config,
            )
            .expect("eos_dt_get");
            let expected_ln_pgas = log10(p_gas) * LN10;
            assert!(
                (eos.res[I_LN_PGAS] - expected_ln_pgas).abs() <= 1.0e-8,
                "EOS lnPgas mismatch at j={j} ({opacity_mode:?}): eos={} expected={}",
                eos.res[I_LN_PGAS],
                expected_ln_pgas
            );
        }
    }
}

#[test]
fn test_fgong_atm_unsupported_options_and_missing_inputs_fail_loudly() {
    let base_model = build_hydrated_solar_zams();

    // Non-T_tau atmosphere options must return GyreDataError::Unported.
    for bad_opt in [AtmOption::Table, AtmOption::IrradiatedGrey] {
        let mut model = base_model.clone();
        model.controls.atm_option = bad_opt;
        let input = PulseDataInput::from_model(&model).expect("input");
        let err = get_fgong_data(&input, true, false, true)
            .expect_err("non-T_tau build_atm must fail loudly");
        assert!(
            matches!(err, GyreDataError::Unported { .. }),
            "unexpected error for {bad_opt:?}: {err}"
        );
    }

    // use_other_kap = true must fail loudly with Unported.
    let mut other_kap = base_model.clone();
    other_kap.controls.use_other_kap = true;
    let input_other_kap =
        PulseDataInput::from_model(&other_kap).expect("input");
    assert!(matches!(
        get_fgong_data(&input_other_kap, true, false, true),
        Err(GyreDataError::Unported { .. })
    ));

    // atm = None must fail loudly with Upstream when add_atmosphere = true.
    let mut no_atm_input = PulseDataInput::from_model(&base_model).expect("in");
    no_atm_input.atm = None;
    assert!(matches!(
        get_fgong_data(&no_atm_input, true, false, true),
        Err(GyreDataError::Upstream(_))
    ));

    // Missing h1 or he4 must fail loudly even when add_atmosphere = true.
    let mut no_h1 = PulseDataInput::from_model(&base_model).expect("in");
    no_h1.iso.h1 = None;
    no_h1.dxdt_nuc_h1 = None;
    assert!(matches!(
        get_fgong_data(&no_h1, true, false, true),
        Err(GyreDataError::Upstream(_))
    ));

    let mut no_he4 = PulseDataInput::from_model(&base_model).expect("in");
    no_he4.iso.he4 = None;
    assert!(matches!(
        get_fgong_data(&no_he4, true, false, true),
        Err(GyreDataError::Upstream(_))
    ));
}

#[derive(Debug)]
struct CapturedFgongBin {
    iconst: usize,
    ivar: usize,
    nn: usize,
    global_data: Vec<f64>,
    point_data: Vec<f64>,
}

fn read_captured_fgong_bin(path: &Path) -> CapturedFgongBin {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!("failed to open {}: {e}", path.display());
    });
    let mut reader = BufReader::new(file);
    let mut raw = Vec::new();
    reader
        .read_to_end(&mut raw)
        .unwrap_or_else(|e| panic!("read failed {}: {e}", path.display()));
    assert!(raw.len() >= 12, "truncated header in {}", path.display());
    let iconst =
        usize::try_from(i32::from_le_bytes(raw[0..4].try_into().unwrap()))
            .expect("non-negative iconst");
    let ivar =
        usize::try_from(i32::from_le_bytes(raw[4..8].try_into().unwrap()))
            .expect("non-negative ivar");
    let nn =
        usize::try_from(i32::from_le_bytes(raw[8..12].try_into().unwrap()))
            .expect("non-negative nn");
    let expected_bytes = 12 + 8 * iconst + 8 * ivar * nn;
    assert_eq!(
        raw.len(),
        expected_bytes,
        "byte size mismatch in {}",
        path.display()
    );
    let mut global_data = Vec::with_capacity(iconst);
    for i in 0..iconst {
        let off = 12 + 8 * i;
        global_data
            .push(f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
    }
    let mut point_data = Vec::with_capacity(ivar * nn);
    let base = 12 + 8 * iconst;
    for i in 0..(ivar * nn) {
        let off = base + 8 * i;
        point_data
            .push(f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
    }
    CapturedFgongBin {
        iconst,
        ivar,
        nn,
        global_data,
        point_data,
    }
}

fn load_and_hydrate_captured_mod(
    case_dir: &Path,
    mod_path: &Path,
) -> StarModel {
    let bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        "inlist",
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .unwrap_or_else(|e| {
        panic!("inlist load failed {}: {e}", case_dir.display())
    });
    let mod_text = std::fs::read_to_string(mod_path).unwrap_or_else(|e| {
        panic!("read mod failed {}: {e}", mod_path.display())
    });
    let mut controls = bundle.controls;
    controls.calculate_brunt_n2 = true;
    // `.mod` headers omit `initial_y`; MESA's `model_builder` resolves `< 0` to `0.24 + 2*initial_z`.
    if controls.initial_y < 0.0 {
        controls.initial_y = (0.24 + 2.0 * controls.initial_z).clamp(0.0, 1.0);
    }
    let mut model =
        read_mod_str_with_dir(&mod_text, controls, bundle.job, Some(case_dir))
            .unwrap_or_else(|e| {
                panic!("read_mod failed {}: {e}", mod_path.display())
            });
    model.eos_config = bundle.eos_config;
    model.kap_config = bundle.kap_config;
    set_hydro_vars(&mut model, false, true).unwrap_or_else(|e| {
        panic!("set_hydro_vars failed {}: {e}", mod_path.display())
    });
    model
}

#[test]
fn test_fgong_atm_parity_against_astero_adipls_capture() {
    let raw = std::env::var("FLEXMESA_ADIPLS_CAPTURES").ok();
    let path_str = raw.as_deref().map_or("", str::trim);
    if path_str.is_empty() {
        eprintln!(
            "INFO: FLEXMESA_ADIPLS_CAPTURES unset; external capture parity not run"
        );
        return;
    }
    let root = PathBuf::from(path_str);
    assert!(
        root.is_dir(),
        "FLEXMESA_ADIPLS_CAPTURES path `{}` is not a directory",
        root.display()
    );

    let case_dir = root.join("astero_adipls");
    let model = load_and_hydrate_captured_mod(
        &case_dir,
        &case_dir.join("pulse_tap/models/adipls_eval_000001_step_0000050.mod"),
    );
    let input = PulseDataInput::from_model(&model).expect("pulse input");
    let (g, pd) =
        get_fgong_data(&input, true, false, true).expect("astero_adipls fgong");
    let cap =
        read_captured_fgong_bin(&case_dir.join("a0tap/store_000001.fgong.bin"));

    assert_eq!(cap.iconst, FGONG_ICONST);
    assert_eq!(cap.ivar, FGONG_IVAR);
    for (idx, (&got_v, &cap_v)) in
        g.iter().zip(cap.global_data.iter()).enumerate()
    {
        if idx == 4 {
            let denom = cap_v.abs().max(1.0);
            assert!(
                (got_v - cap_v).abs() / denom <= 1.0e-3,
                "astero_adipls global[{idx}] got={got_v} cap={cap_v}"
            );
            continue;
        }
        let tol = if idx == 10 || idx == 11 {
            1.0e-2
        } else {
            5.0e-4
        };
        let denom = cap_v.abs().max(1.0e-30);
        assert!(
            (got_v - cap_v).abs() / denom <= tol,
            "astero_adipls global[{idx}] got={got_v:.12e} cap={cap_v:.12e}"
        );
    }

    assert!(
        pd.len().abs_diff(cap.nn) <= 4,
        "astero_adipls atm nn got={} cap={}",
        pd.len(),
        cap.nn
    );
    for row in [0usize, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13] {
        let got_v = pd[0][row];
        let cap_v = cap.point_data[row];
        let rel = (got_v - cap_v).abs() / cap_v.abs().max(1.0e-30);
        assert!(
            rel <= 2.0e-2,
            "astero_adipls atm top row={row}: got={got_v:.10e} cap={cap_v:.10e} rel={rel:.3e}"
        );
    }
    let got_ctr = pd.last().expect("ctr");
    let cap_ctr_base = (cap.nn - 1) * FGONG_IVAR;
    assert_eq!(got_ctr[0], 0.0);
    assert_eq!(got_ctr[1], cap.point_data[cap_ctr_base + 1]);
    for row in [2usize, 3, 4, 5, 9, 10, 11, 12, 13, 17, 25, 26, 27] {
        let got_v = got_ctr[row];
        let cap_v = cap.point_data[cap_ctr_base + row];
        let rel = (got_v - cap_v).abs() / cap_v.abs().max(1.0e-12);
        assert!(
            rel <= 5.0e-3,
            "astero_adipls ctr row={row}: got={got_v:.10e} cap={cap_v:.10e} rel={rel:.3e}"
        );
    }
}
