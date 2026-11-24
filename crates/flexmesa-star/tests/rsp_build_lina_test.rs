// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA (release `26.4.1`, commit
//! `c8c202662922a22b3efefa824473ad615c623afd`) parity tests for static `RSP`
//! envelope integration (`STAH`), Kuhfuss/Smolec envelope relaxation
//! (`RELAX_ENV`), linear non-adiabatic radial pulsation stability (`do_LINA`),
//! and Cepheid 2nd-crossing instability-strip boundary interpolation
//! (`rsp_check_2nd_crossing`, Issue `#5`, re-baselined in `#141`).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_star::{
    do_rsp_build_model, read_mod_str, StarControls, StarJobControls,
};

fn golden_rsp_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("tests")
        .join("fixtures")
        .join("golden")
        .join("rsp")
}

fn a09_kap_config(zbase: f64) -> KapConfig {
    KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase,
        ..Default::default()
    }
}

fn base_rsp_job() -> StarJobControls {
    StarJobControls {
        create_rsp_model: true,
        initial_zfracs: 6,
        ..Default::default()
    }
}

fn load_lina_26_4_1(path: &Path) -> HashMap<String, ([f64; 3], [f64; 3])> {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        raw.contains("# mesa_version_number=r26.4.1"),
        "missing r26.4.1 header in {}",
        path.display()
    );
    let mut out = HashMap::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 7, "unexpected LINA row: {trimmed}");
        let periods = [
            cols[1].parse::<f64>().expect("p0"),
            cols[3].parse::<f64>().expect("p1"),
            cols[5].parse::<f64>().expect("p2"),
        ];
        let growths = [
            cols[2].parse::<f64>().expect("g0"),
            cols[4].parse::<f64>().expect("g1"),
            cols[6].parse::<f64>().expect("g2"),
        ];
        out.insert(cols[0].to_string(), (periods, growths));
    }
    out
}

struct RspReferenceCase {
    slug: &'static str,
    fname: &'static str,
    configure: fn(&mut StarControls),
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_rsp_build_and_lina_match_all_7_mesa_26_4_1_reference_models() {
    let gdir = golden_rsp_dir();
    let lina_refs = load_lina_26_4_1(&gdir.join("lina_26_4_1.tsv"));
    let cases = [
        RspReferenceCase {
            slug: "rsp_cepheid",
            fname: "rsp_cepheid_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 4.165;
                c.rsp_teff = 6050.0;
                c.rsp_l = 1438.8;
                c.rsp_x = 0.73;
                c.rsp_z = 0.007;
            },
        },
        RspReferenceCase {
            slug: "rsp_cepheid_6M",
            fname: "rsp_cepheid_6M_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 6.0;
                c.rsp_teff = 4892.0;
                c.rsp_l = 4660.0;
                c.rsp_x = 0.73;
                c.rsp_z = 0.003;
            },
        },
        RspReferenceCase {
            slug: "rsp_RR_Lyrae",
            fname: "rsp_RR_Lyrae_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 0.65;
                c.rsp_teff = 6500.0;
                c.rsp_l = 60.0;
                c.rsp_x = 0.75;
                c.rsp_z = 0.0004;
            },
        },
        RspReferenceCase {
            slug: "rsp_BEP",
            fname: "rsp_BEP_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 0.26;
                c.rsp_teff = 6968.0;
                c.rsp_l = 33.0;
                c.rsp_x = 0.70;
                c.rsp_z = 0.01;
            },
        },
        RspReferenceCase {
            slug: "rsp_BLAP",
            fname: "rsp_BLAP_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 0.36;
                c.rsp_teff = 26_000.0;
                c.rsp_l = 320.0;
                c.rsp_x = 0.70;
                c.rsp_z = 0.05;
                c.rsp.t_anchor = 2.0e5;
                c.rsp.alfam = 0.15;
                c.rsp.alfat = 0.01;
                c.rsp.use_prad_for_psurf = true;
                c.rsp.kick_vsurf_km_per_sec = 0.5;
            },
        },
        RspReferenceCase {
            slug: "rsp_Delta_Scuti",
            fname: "rsp_Delta_Scuti_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 2.0;
                c.rsp_teff = 6900.0;
                c.rsp_l = 30.0;
                c.rsp_x = 0.70;
                c.rsp_z = 0.02;
                c.rsp.alfam = 0.25;
                c.rsp.alfa = 1.2;
                c.rsp.kick_vsurf_km_per_sec = 3.0;
                c.rsp.fraction_1st_overtone = 0.5;
            },
        },
        RspReferenceCase {
            slug: "rsp_Type_II_Cepheid",
            fname: "rsp_Type_II_Cepheid_init_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 0.55;
                c.rsp_teff = 6410.0;
                c.rsp_l = 136.0;
                c.rsp_x = 0.76;
                c.rsp_z = 0.0001;
                c.rsp.do_check_omega = false;
                c.rsp.alfam = 0.05;
                c.rsp_target_steps_per_cycle = 1200;
                c.rsp.kick_vsurf_km_per_sec = 4.5;
                c.rsp.nz = 200;
                c.rsp.nz_outer = 60;
                c.rsp.t_anchor = 11.0e3;
            },
        },
    ];

    for case in &cases {
        let fname = case.fname;
        let &(ref_periods_sec, ref_growth_rates) = lina_refs
            .get(case.slug)
            .unwrap_or_else(|| panic!("missing LINA row for {}", case.slug));
        let raw = fs::read_to_string(gdir.join(fname))
            .unwrap_or_else(|e| panic!("read {fname}: {e}"));
        assert!(
            raw.contains("'r26.4.1'"),
            "{fname}: expected MESA version_number 'r26.4.1' in header"
        );
        let golden = read_mod_str(
            &raw,
            StarControls::default(),
            StarJobControls::default(),
        )
        .unwrap_or_else(|e| panic!("parse {fname}: {e}"));

        let mut ctrl = StarControls::default();
        (case.configure)(&mut ctrl);
        ctrl.zbase = ctrl.rsp_z;
        let kap = a09_kap_config(ctrl.rsp_z);
        let (built, lina_opt) = do_rsp_build_model(
            ctrl,
            base_rsp_job(),
            EosConfig::default(),
            kap,
            None,
        )
        .unwrap_or_else(|e| panic!("do_rsp_build_model {fname}: {e}"));
        let lina = lina_opt.expect("LINA output");

        assert_eq!(built.nz, golden.nz, "{fname}: nz mismatch");
        let rel_xm = (built.xmstar - golden.xmstar).abs() / golden.xmstar;
        let rel_core_mass =
            (built.m_center - golden.m_center).abs() / golden.m_center;
        let rel_core_radius =
            (built.r_center - golden.r_center).abs() / golden.r_center;
        let rel_core_lum =
            (built.l_center - golden.l_center).abs() / golden.l_center;
        let rel_tf = (built.controls.tau_factor - golden.controls.tau_factor)
            .abs()
            / golden.controls.tau_factor;

        let mut max_rel_dq = 0.0_f64;
        let mut max_abs_ln_rho = 0.0_f64;
        let mut max_abs_ln_temp = 0.0_f64;
        let mut max_abs_ln_rad = 0.0_f64;
        let mut max_rel_erad = 0.0_f64;
        let mut max_rel_fr = 0.0_f64;
        let mut max_rel_v = 0.0_f64;
        let v_scale = built.v[0].abs().max(1.0);
        for k in 0..built.nz {
            max_rel_dq = max_rel_dq
                .max((built.dq[k] - golden.dq[k]).abs() / golden.dq[k]);
            max_abs_ln_rho =
                max_abs_ln_rho.max((built.lnd[k] - golden.lnd[k]).abs());
            max_abs_ln_temp =
                max_abs_ln_temp.max((built.ln_t[k] - golden.ln_t[k]).abs());
            max_abs_ln_rad =
                max_abs_ln_rad.max((built.ln_r[k] - golden.ln_r[k]).abs());
            max_rel_erad = max_rel_erad.max(
                (built.aux.rsp.erad_rsp[k] - golden.aux.rsp.erad_rsp[k]).abs()
                    / golden.aux.rsp.erad_rsp[k].abs().max(1.0),
            );
            max_rel_fr = max_rel_fr.max(
                (built.aux.rsp.fr_rsp[k] - golden.aux.rsp.fr_rsp[k]).abs()
                    / golden.aux.rsp.fr_rsp[k].abs().max(1.0),
            );
            max_rel_v =
                max_rel_v.max((built.v[k] - golden.v[k]).abs() / v_scale);
        }

        let mut max_rel_p = 0.0_f64;
        let mut max_abs_g = 0.0_f64;
        for m in 0..3 {
            max_rel_p = max_rel_p.max(
                (lina.periods_sec[m] - ref_periods_sec[m]).abs()
                    / ref_periods_sec[m],
            );
            max_abs_g = max_abs_g
                .max((lina.growth_rates[m] - ref_growth_rates[m]).abs());
        }

        eprintln!(
            "{fname}: rel_xm={rel_xm:.3e} rel_rc={rel_core_radius:.3e} dq={max_rel_dq:.3e} lnd={max_abs_ln_rho:.3e} lnt={max_abs_ln_temp:.3e} lnr={max_abs_ln_rad:.3e} erad={max_rel_erad:.3e} fr={max_rel_fr:.3e} v={max_rel_v:.3e} rel_P={max_rel_p:.3e} abs_eta={max_abs_g:.3e}"
        );

        assert!(rel_xm < 1.0e-13, "{fname}: rel_xm={rel_xm:.3e}");
        assert!(
            rel_core_mass < 1.0e-13,
            "{fname}: rel_core_mass={rel_core_mass:.3e}"
        );
        assert!(
            rel_core_radius < 2.0e-9,
            "{fname}: rel_core_radius={rel_core_radius:.3e}"
        );
        assert!(
            rel_core_lum < 1.0e-13,
            "{fname}: rel_core_lum={rel_core_lum:.3e}"
        );
        assert!(rel_tf < 1.0e-13, "{fname}: rel_tf={rel_tf:.3e}");
        assert!(max_rel_dq < 2.0e-8, "{fname}: max_rel_dq={max_rel_dq:.3e}");
        assert!(
            max_abs_ln_rho < 2.0e-8,
            "{fname}: max_abs_ln_rho={max_abs_ln_rho:.3e}"
        );
        assert!(
            max_abs_ln_temp < 1.0e-8,
            "{fname}: max_abs_ln_temp={max_abs_ln_temp:.3e}"
        );
        assert!(
            max_abs_ln_rad < 5.0e-9,
            "{fname}: max_abs_ln_rad={max_abs_ln_rad:.3e}"
        );
        assert!(
            max_rel_erad < 5.0e-8,
            "{fname}: max_rel_erad={max_rel_erad:.3e}"
        );
        assert!(max_rel_fr < 1.0e-8, "{fname}: max_rel_fr={max_rel_fr:.3e}");
        assert!(max_rel_v < 1.0e-8, "{fname}: max_rel_v={max_rel_v:.3e}");
        assert!(max_rel_p < 1.0e-7, "{fname}: max_rel_p={max_rel_p:.3e}");
        assert!(max_abs_g < 1.0e-6, "{fname}: max_abs_g={max_abs_g:.3e}");
    }
}
