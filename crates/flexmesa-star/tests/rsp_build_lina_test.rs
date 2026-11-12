// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA (`r24.08.1`, commit `0b2e6bb`) golden parity tests for
//! static `RSP` envelope integration (`STAH`), Kuhfuss/Smolec envelope
//! relaxation (`RELAX_ENV`), linear non-adiabatic radial pulsation stability
//! (`do_LINA`), and Cepheid 2nd-crossing instability-strip boundary
//! interpolation (`rsp_check_2nd_crossing`, Issue `#5`).

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

struct GoldenRspCase {
    fname: &'static str,
    periods_sec: [f64; 3],
    growth_rates: [f64; 3],
    configure: fn(&mut StarControls),
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_rsp_build_and_lina_match_all_7_r24_08_1_golden_models() {
    let gdir = golden_rsp_dir();
    let cases = [
        GoldenRspCase {
            fname: "rsp_cepheid_init_r24.mod",
            periods_sec: [
                3.437_594_666_629_719e5,
                2.445_728_144_258_661_2e5,
                1.869_175_519_541_832_6e5,
            ],
            growth_rates: [
                1.448_218_706_060_893_9e-2,
                2.832_279_134_588_332_2e-2,
                -8.729_792_420_348_392e-2,
            ],
            configure: |c| {
                c.rsp_mass = 4.165;
                c.rsp_teff = 6050.0;
                c.rsp_l = 1438.8;
                c.rsp_x = 0.73;
                c.rsp_z = 0.007;
            },
        },
        GoldenRspCase {
            fname: "rsp_cepheid_6M_init_r24.mod",
            periods_sec: [
                1.685_039_581_255_813_4e6,
                1.024_492_940_651_358_6e6,
                7.184_241_879_354_038e5,
            ],
            growth_rates: [
                -3.240_294_802_432_712e-2,
                -1.937_301_362_027_289_4e-1,
                -2.921_241_693_949_774e-1,
            ],
            configure: |c| {
                c.rsp_mass = 6.0;
                c.rsp_teff = 4892.0;
                c.rsp_l = 4660.0;
                c.rsp_x = 0.73;
                c.rsp_z = 0.003;
            },
        },
        GoldenRspCase {
            fname: "rsp_RR_Lyrae_init_r24.mod",
            periods_sec: [
                6.069_707_926_490_174_6e4,
                4.477_071_797_338_348_6e4,
                3.447_269_805_105_282e4,
            ],
            growth_rates: [
                3.033_911_230_917_817_8e-2,
                1.894_849_716_330_139_5e-2,
                -1.036_883_853_911_131_9e-1,
            ],
            configure: |c| {
                c.rsp_mass = 0.65;
                c.rsp_teff = 6500.0;
                c.rsp_l = 60.0;
                c.rsp_x = 0.75;
                c.rsp_z = 0.0004;
            },
        },
        GoldenRspCase {
            fname: "rsp_BEP_init_r24.mod",
            periods_sec: [
                5.482_092_622_149_857e4,
                3.927_329_732_257_694e4,
                2.899_262_276_863_794e4,
            ],
            growth_rates: [
                1.597_408_974_806_609e-2,
                -5.471_814_147_342_12e-2,
                -4.134_710_726_758_465e-1,
            ],
            configure: |c| {
                c.rsp_mass = 0.26;
                c.rsp_teff = 6968.0;
                c.rsp_l = 33.0;
                c.rsp_x = 0.70;
                c.rsp_z = 0.01;
            },
        },
        GoldenRspCase {
            fname: "rsp_BLAP_init_r24.mod",
            periods_sec: [
                4.781_288_732_677_035e3,
                3.381_132_151_645_098_5e3,
                2.641_235_651_243_388_5e3,
            ],
            growth_rates: [
                2.968_583_434_358_451e-2,
                -1.864_785_307_640_226_5e-1,
                -3.645_866_163_423_907e-1,
            ],
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
        GoldenRspCase {
            fname: "rsp_Delta_Scuti_init_r24.mod",
            periods_sec: [
                1.477_096_710_202_621_2e4,
                1.093_582_896_802_943e4,
                8.509_078_847_937_462e3,
            ],
            growth_rates: [
                1.031_356_637_333_276_4e-5,
                5.899_497_559_683_98e-5,
                -4.115_919_947_307_024_3e-4,
            ],
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
        GoldenRspCase {
            fname: "rsp_Type_II_Cepheid_init_r24.mod",
            periods_sec: [
                1.421_140_390_920_954e5,
                9.921_261_881_928_619e4,
                7.185_248_489_518_021e4,
            ],
            growth_rates: [
                1.648_116_805_650_379_2e-1,
                4.438_102_393_856_967_4e-5,
                -5.367_501_851_688_521e-1,
            ],
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
        let raw = fs::read_to_string(gdir.join(fname))
            .unwrap_or_else(|e| panic!("read {fname}: {e}"));
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
                (lina.periods_sec[m] - case.periods_sec[m]).abs()
                    / case.periods_sec[m],
            );
            max_abs_g = max_abs_g
                .max((lina.growth_rates[m] - case.growth_rates[m]).abs());
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
