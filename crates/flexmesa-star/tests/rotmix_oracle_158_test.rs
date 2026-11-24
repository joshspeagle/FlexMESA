// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA 26.4.1 oracle for `set_rotation_mixing_info` and
//! `update_rotation_mixing_info` (`#158`).
//!
//! The fixture `benchmarks/fortran_mesa/rotmix_oracles/` holds the inputs
//! that real MESA 26.4.1 had at entry to `update_rotation_mixing_info`
//! (`star/private/mix_info.f90:1561-1757`) on model 2 of
//! `20M_z2m2_high_rotation`, and the values it produced. These tests load
//! the same state into a `StarModel` and require every face value to be
//! bit-identical: the rotational diffusivities of `set_rotation_mixing_info`
//! (`star/private/rotation_mix_info.f90:47-1123`, with the Spruit-Tayler
//! time smoothing of `:191-215` active), and `D_mix`, `D_mix_rotation` and
//! the `am_nu_*` viscosities of `update_rotation_mixing_info`, which this
//! case runs with `D_omega_flag` (`rotation_mix_info.f90:255-428`).

#![forbid(unsafe_code)]

use std::path::PathBuf;

use flexmesa_const::MixingType;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::model::StarModel;

fn star_suite_dir() -> PathBuf {
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
    base.join("star").join("test_suite")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/rotmix_oracles")
        .join("mesa_26_4_1_rotmix_20M_model2.txt")
}

fn hex(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex value"))
}

struct Fixture {
    hdr: Vec<String>,
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f64>>,
}

fn read_fixture() -> Fixture {
    let path = fixture_path();
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("cannot read rotmix oracle {}: {e}", path.display())
    });
    assert!(
        txt.contains("# mesa_version: r26.4.1"),
        "fixture header must record MESA r26.4.1"
    );
    let mut fx = Fixture {
        hdr: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    for line in txt.lines().filter(|l| !l.starts_with('#')) {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t[0] {
            "HDR" => fx.hdr = t.iter().map(|s| (*s).to_string()).collect(),
            "IN" => fx.inputs.push(t[3..].iter().map(|s| hex(s)).collect()),
            "RMX" => fx.outputs.push(t[4..].iter().map(|s| hex(s)).collect()),
            other => panic!("unexpected fixture row tag {other}"),
        }
    }
    fx
}

#[allow(clippy::too_many_lines)]
fn build_model(fx: &Fixture) -> StarModel {
    let nz: usize = fx.hdr[3].parse().expect("nz");
    assert_eq!(fx.inputs.len(), nz);
    assert_eq!(fx.outputs.len(), nz);

    let case_dir = star_suite_dir().join("20M_z2m2_high_rotation");
    assert!(
        case_dir.is_dir(),
        "missing 20M_z2m2_high_rotation test_suite inputs at {}",
        case_dir.display()
    );
    // The inlist still sets four unported controls (`max_safe_logT_for_rates`,
    // `warn_rates_for_high_temp`, `near_zams_relax_omega_div_omega_crit`,
    // `num_steps_to_relax_rotation`); none enters these routines.
    let bundle = InlistBundle::from_file_in_case_dir_with_options(
        &case_dir,
        "inlist_to_end_core_he_burn_header",
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("load 20M_z2m2_high_rotation inlist");

    let mut controls = bundle.controls.clone();
    controls.rotation_flag = true;
    assert!(
        controls.d_omega_flag,
        "the inlist sets change_D_omega_flag and new_D_omega_flag"
    );
    let mut m = StarModel::new(nz, vec![2, 7], bundle.job.clone(), controls);
    m.kap_config = bundle.kap_config.clone();
    m.aux.doing_relax = fx.hdr[4] == "T";
    m.doing_first_model_of_run = fx.hdr[5] == "T";
    m.aux.st_start.prev_mesh_have_st_start_info = fx.hdr[6] == "T";
    m.r_center = hex(&fx.hdr[7]);
    m.mstar_old = hex(&fx.hdr[8]);
    m.dt = hex(&fx.hdr[9]);
    assert!(
        !m.aux.doing_relax
            && !m.doing_first_model_of_run
            && m.aux.st_start.prev_mesh_have_st_start_info,
        "fixture call must exercise the ST time smoothing"
    );

    let mut grad_density = vec![0.0; nz];
    let mut grad_temperature = vec![0.0; nz];
    let mut prev_dq = vec![0.0; nz];
    let mut prev_d_st = vec![0.0; nz];
    let mut prev_nu_st = vec![0.0; nz];
    let mut prev_nz = 0_usize;
    m.aux.mixing.d_mix_non_rotation = vec![0.0; nz];
    m.aux.smoothed_brunt_b = vec![0.0; nz];
    m.aux.mixing.grada_face.resize(nz, 0.0);
    for (k, c) in fx.inputs.iter().enumerate() {
        assert_eq!(c.len(), 41, "IN row {} must have 41 values", k + 1);
        m.r[k] = c[0];
        m.m[k] = c[1];
        m.lum[k] = c[2];
        m.j_rot[k] = c[3];
        m.omega[k] = c[4];
        m.gradt[k] = c[5];
        m.aux.mixing.grada_face[k] = c[6];
        m.grav[k] = c[7];
        m.cgrav[k] = c[8];
        m.eps_nuc[k] = c[9];
        m.eps_nuc_neu_total[k] = c[10];
        m.non_nuc_neu[k] = c[11];
        m.csound[k] = c[12];
        m.rho[k] = c[13];
        m.t[k] = c[14];
        m.peos[k] = c[15];
        m.cp[k] = c[16];
        m.cv[k] = c[17];
        m.chi_rho[k] = c[18];
        m.chi_t[k] = c[19];
        m.abar[k] = c[20];
        m.zbar[k] = c[21];
        m.opacity[k] = c[22];
        m.gamma1[k] = c[23];
        m.dm[k] = c[24];
        m.dq[k] = c[25];
        m.rmid[k] = c[26];
        grad_density[k] = c[27];
        grad_temperature[k] = c[28];
        m.aux.smoothed_brunt_b[k] = c[29];
        #[allow(clippy::cast_possible_truncation)]
        let code = c[30] as i32;
        m.mixing_type[k] = MixingType::from_code(code).expect("mixing_type");
        m.tau[k] = c[31];
        m.ln_t[k] = c[32];
        m.ye[k] = c[33];
        m.aux.mixing.d_mix_non_rotation[k] = c[34];
        m.d_mix[k] = c[34];
        // c[35] is MESA's face `kap_cond`; the routine computes its own.
        m.q[k] = c[36];
        prev_dq[k] = c[37];
        prev_d_st[k] = c[38];
        prev_nu_st[k] = c[39];
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            prev_nz = c[40] as usize;
        }
    }
    m.aux.st_start.prev_mesh_dq = prev_dq;
    m.aux.st_start.prev_mesh_d_st_start = prev_d_st;
    m.aux.st_start.prev_mesh_nu_st_start = prev_nu_st;
    m.aux.st_start.prev_mesh_nz = prev_nz;
    m.aux.rotmix.grad_density = grad_density;
    m.aux.rotmix.grad_temperature = grad_temperature;

    m
}

/// Lists, per array, the faces from `k_min` on that differ from MESA.
fn compare(
    fx: &Fixture,
    checks: &[(&str, usize, &[f64])],
    k_min: usize,
) -> Vec<String> {
    let nz = fx.outputs.len();
    let mut report = Vec::new();
    for &(name, col, flex) in checks {
        let mut bad = 0_usize;
        let mut first = None;
        for k in k_min..nz {
            let mesa = fx.outputs[k][col];
            if mesa.to_bits() != flex[k].to_bits() {
                bad += 1;
                first.get_or_insert((k + 1, mesa, flex[k]));
            }
        }
        if bad > 0 {
            report.push(format!("{name}: {bad}/{nz} differ, first {first:?}"));
        }
    }
    report
}

#[test]
fn test_set_rotation_mixing_info_20m_model2_matches_mesa_bitwise() {
    let fx = read_fixture();
    let mut m = build_model(&fx);
    flexmesa_star::set_rotation_mixing_info(&mut m)
        .expect("set_rotation_mixing_info");
    let rm = &m.aux.rotmix;
    // (name, RMX column, FlexMESA array)
    let checks: [(&str, usize, &[f64]); 11] = [
        ("D_visc", 0, &rm.d_visc),
        ("D_DSI", 1, &rm.d_dsi),
        ("D_SH", 2, &rm.d_sh),
        ("D_SSI", 3, &rm.d_ssi),
        ("D_ES", 4, &rm.d_es),
        ("D_GSF", 5, &rm.d_gsf),
        ("D_ST", 6, &rm.d_st),
        ("nu_ST", 7, &rm.nu_st),
        ("richardson_number", 15, &rm.richardson_number),
        ("omega_shear", 16, &rm.omega_shear),
        ("domega_dlnR", 17, &rm.domega_dlnr),
    ];
    let report = compare(&fx, &checks, 0);
    // The fixture must exercise ST: some face has nonzero D_ST.
    assert!(rm.d_st.iter().any(|&v| v > 0.0), "D_ST is zero everywhere");
    assert!(
        report.is_empty(),
        "not bit-identical to MESA:\n{}",
        report.join("\n")
    );
}

#[test]
fn test_update_rotation_mixing_info_d_omega_20m_model2_matches_mesa_bitwise() {
    let fx = read_fixture();
    let mut m = build_model(&fx);
    flexmesa_star::update_rotation_mixing_info(&mut m)
        .expect("update_rotation_mixing_info");
    let rm = &m.aux.rotmix;
    // `D_omega` is not in the fixture; it enters `D_mix_rotation` and
    // `am_nu_rot` (`star/private/mix_info.f90:1622-1623, 1701-1702`).
    assert!(
        rm.d_omega.iter().any(|&v| v > 0.0),
        "D_omega is zero everywhere"
    );
    // Face 1 of `D_mix` and `D_mix_rotation` is not set by this routine
    // (its loop starts at `k = 2`, `mix_info.f90:1600`).
    let from_face_2: [(&str, usize, &[f64]); 2] = [
        ("D_mix", 8, &m.d_mix),
        ("D_mix_rotation", 10, &rm.d_mix_rotation),
    ];
    let all_faces: [(&str, usize, &[f64]); 4] = [
        ("am_nu_rot", 11, &m.aux.am_nu_rot),
        ("am_nu_non_rot", 12, &m.aux.am_nu_non_rot),
        ("am_nu_omega", 13, &m.aux.am_nu_omega),
        ("am_nu_j", 14, &m.aux.am_nu_j),
    ];
    let mut report = compare(&fx, &from_face_2, 1);
    report.extend(compare(&fx, &all_faces, 0));
    assert!(
        report.is_empty(),
        "not bit-identical to MESA:\n{}",
        report.join("\n")
    );
}
