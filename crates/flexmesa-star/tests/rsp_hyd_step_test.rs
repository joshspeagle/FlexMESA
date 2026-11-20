// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA (release `26.4.1`, commit
//! `c8c202662922a22b3efefa824473ad615c623afd`) parity tests for
//! nonlinear `RSP` `HYD` time integration (`star/private/rsp_step.f90` &
//! `star/private/rsp.f90`), multi-cycle period/amplitude detection, `GYRE`
//! `l = 0` BVP velocity kick (`rsp_gyre`), lossless `.mod` save/load
//! continuation, and bit-identical `.photo` restart parity (`#5`, `#141`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_star::inlist_runner::StarExtras;
use flexmesa_star::rsp_eos_kap::RspChemCache;
use flexmesa_star::{
    do_brunt_b, do_brunt_n2, do_rsp_build_model, export_gyre_with_schema,
    read_mod_str, rsp_setup_model, solve_gyre_mesa_modes, star_evolve_step,
    star_evolve_step_with_extras, verify_photo_restart_parity, write_mod_string,
    StarControls, StarError, StarJobControls, StarModel,
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
        set_initial_age: true,
        initial_age: 0.0,
        set_initial_model_number: true,
        initial_model_number: 0,
        set_initial_cumulative_energy_error: true,
        new_cumulative_energy_error: 0.0,
        ..Default::default()
    }
}

struct GoldenStep1Case {
    fname: &'static str,
    configure: fn(&mut StarControls),
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_rsp_hyd_step1_matches_all_7_mesa_26_4_1_reference_models() {
    let gdir = golden_rsp_dir();
    let cases = [
        GoldenStep1Case {
            fname: "rsp_cepheid_step1_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 4.165;
                c.rsp_teff = 6050.0;
                c.rsp_l = 1438.8;
                c.rsp_x = 0.73;
                c.rsp_z = 0.007;
            },
        },
        GoldenStep1Case {
            fname: "rsp_cepheid_6M_step1_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 6.0;
                c.rsp_teff = 4892.0;
                c.rsp_l = 4660.0;
                c.rsp_x = 0.73;
                c.rsp_z = 0.003;
            },
        },
        GoldenStep1Case {
            fname: "rsp_RR_Lyrae_step1_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 0.65;
                c.rsp_teff = 6500.0;
                c.rsp_l = 60.0;
                c.rsp_x = 0.75;
                c.rsp_z = 0.0004;
            },
        },
        GoldenStep1Case {
            fname: "rsp_BEP_step1_26_4_1.mod",
            configure: |c| {
                c.rsp_mass = 0.26;
                c.rsp_teff = 6968.0;
                c.rsp_l = 33.0;
                c.rsp_x = 0.70;
                c.rsp_z = 0.01;
            },
        },
        GoldenStep1Case {
            fname: "rsp_BLAP_step1_26_4_1.mod",
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
        GoldenStep1Case {
            fname: "rsp_Delta_Scuti_step1_26_4_1.mod",
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
        GoldenStep1Case {
            fname: "rsp_Type_II_Cepheid_step1_26_4_1.mod",
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
        let (mut model, _) = do_rsp_build_model(
            ctrl,
            base_rsp_job(),
            EosConfig::default(),
            kap,
            None,
        )
        .unwrap_or_else(|e| panic!("build {fname}: {e}"));

        let rep = star_evolve_step(&mut model, false, true, true)
            .unwrap_or_else(|e| panic!("step 1 {fname}: {e}"));
        assert!(rep.converged && rep.model_number == 1);

        let rel_age =
            (model.star_age - golden.star_age).abs() / golden.star_age;
        let rel_dt = (model.dt - golden.dt).abs() / golden.dt;
        let rel_teff = (model.teff - golden.teff).abs() / golden.teff;
        let rel_etot = (model.aux.total_energy - golden.aux.total_energy).abs()
            / golden.aux.total_energy.abs();
        let rel_run_e =
            (model.aux.cumulative_energy_error / model.aux.total_energy).abs();

        let mut err_rho = 0.0_f64;
        let mut err_temp = 0.0_f64;
        let mut err_rad = 0.0_f64;
        let mut max_rel_erad = 0.0_f64;
        let mut max_rel_fr = 0.0_f64;
        let mut max_rel_l = 0.0_f64;
        let mut max_rel_v = 0.0_f64;
        let v_scale = golden.v[0].abs().max(1.0);
        let l_scale = golden.l_center.abs().max(1.0);
        for k in 0..model.nz {
            err_rho = err_rho.max((model.lnd[k] - golden.lnd[k]).abs());
            err_temp = err_temp.max((model.ln_t[k] - golden.ln_t[k]).abs());
            err_rad = err_rad.max((model.ln_r[k] - golden.ln_r[k]).abs());
            max_rel_erad = max_rel_erad.max(
                (model.aux.rsp.erad_rsp[k] - golden.aux.rsp.erad_rsp[k]).abs()
                    / golden.aux.rsp.erad_rsp[k].abs().max(1.0),
            );
            max_rel_fr = max_rel_fr.max(
                (model.aux.rsp.fr_rsp[k] - golden.aux.rsp.fr_rsp[k]).abs()
                    / golden.aux.rsp.fr_rsp[k].abs().max(1.0),
            );
            max_rel_l =
                max_rel_l.max((model.lum[k] - golden.lum[k]).abs() / l_scale);
            max_rel_v =
                max_rel_v.max((model.v[k] - golden.v[k]).abs() / v_scale);
        }

        eprintln!(
            "{fname}: iters={} age={rel_age:.3e} dt={rel_dt:.3e} teff={rel_teff:.3e} etot={rel_etot:.3e} relE={rel_run_e:.3e} lnd={err_rho:.3e} lnt={err_temp:.3e} lnr={err_rad:.3e} erad={max_rel_erad:.3e} fr={max_rel_fr:.3e} L={max_rel_l:.3e} v={max_rel_v:.3e}",
            rep.iterations
        );

        assert!(rel_age < 1.0e-7, "{fname}: rel_age={rel_age:.3e}");
        assert!(rel_dt < 1.0e-7, "{fname}: rel_dt={rel_dt:.3e}");
        assert!(rel_teff < 1.0e-6, "{fname}: rel_teff={rel_teff:.3e}");
        assert!(rel_etot < 1.0e-7, "{fname}: rel_etot={rel_etot:.3e}");
        assert!(rel_run_e < 1.0e-10, "{fname}: rel_run_e={rel_run_e:.3e}");
        assert!(err_rho < 1.0e-7, "{fname}: err_rho={err_rho:.3e}");
        assert!(err_temp < 1.0e-7, "{fname}: err_temp={err_temp:.3e}");
        assert!(err_rad < 1.0e-8, "{fname}: err_rad={err_rad:.3e}");
        assert!(
            max_rel_erad < 1.0e-7,
            "{fname}: max_rel_erad={max_rel_erad:.3e}"
        );
        assert!(max_rel_fr < 1.0e-6, "{fname}: max_rel_fr={max_rel_fr:.3e}");
        assert!(max_rel_l < 1.0e-6, "{fname}: max_rel_l={max_rel_l:.3e}");
        assert!(max_rel_v < 1.0e-7, "{fname}: max_rel_v={max_rel_v:.3e}");
    }
}

#[test]
fn test_rsp_hyd_multicycle_period_save_load_and_photo_restart_parity() {
    let mut ctrl = StarControls {
        rsp_mass: 0.36,
        rsp_teff: 26_000.0,
        rsp_l: 320.0,
        rsp_x: 0.70,
        rsp_z: 0.05,
        zbase: 0.05,
        rsp_target_steps_per_cycle: 150,
        ..StarControls::default()
    };
    ctrl.rsp.t_anchor = 2.0e5;
    ctrl.rsp.alfam = 0.15;
    ctrl.rsp.alfat = 0.01;
    ctrl.rsp.use_prad_for_psurf = true;
    ctrl.rsp.kick_vsurf_km_per_sec = 5.0;
    ctrl.rsp.nz = 80;
    ctrl.rsp.nz_outer = 24;

    let kap = a09_kap_config(ctrl.rsp_z);
    let (mut model, lina_opt) = do_rsp_build_model(
        ctrl.clone(),
        base_rsp_job(),
        EosConfig::default(),
        kap.clone(),
        None,
    )
    .expect("build BLAP");
    let f0_sec = lina_opt.expect("LINA").periods_sec[0];

    for _ in 0..650 {
        star_evolve_step(&mut model, false, true, true).expect("HYD step");
        if model.aux.rsp.num_periods >= 1 {
            break;
        }
    }
    assert!(
        model.aux.rsp.num_periods >= 1,
        "expected >= 1 nonlinear cycle, got {}",
        model.aux.rsp.num_periods
    );
    let rel_p = (model.aux.rsp.period - f0_sec).abs() / f0_sec;
    assert!(rel_p < 0.05, "nonlinear vs LINA period rel_p={rel_p:.4e}");
    assert!(model.aux.rsp.delta_r > 0.0);
    assert!(model.aux.rsp.delta_mag > 0.0);
    let rel_e =
        (model.aux.cumulative_energy_error / model.aux.total_energy).abs();
    assert!(rel_e < 1.0e-4, "rel_e={rel_e:.3e}");

    let (c1, c2) =
        verify_photo_restart_parity(&model, 3).expect("photo restart parity");
    assert_eq!(c1, c2);

    let mod_txt = write_mod_string(&model).expect("write .mod");
    let mut loaded_ctrl = ctrl;
    loaded_ctrl.rsp.initial_dt_factor = 1.0;
    let mut loaded =
        read_mod_str(&mod_txt, loaded_ctrl, StarJobControls::default())
            .expect("read .mod");
    loaded.eos_config = EosConfig::default();
    loaded.kap_config = kap;
    rsp_setup_model(&mut loaded, false, false).expect("rsp_setup_model loaded");
    let rep_loaded = star_evolve_step(&mut loaded, false, true, true)
        .expect("step after .mod load");
    assert!(rep_loaded.converged);
}

struct TestGyreRspExtras;

impl StarExtras for TestGyreRspExtras {
    fn other_rsp_linear_analysis(
        &mut self,
        model: &mut StarModel,
        chem: &RspChemCache,
        log_dir: Option<&Path>,
    ) -> Result<bool, StarError> {
        let nzn = model.nz;
        if model.aux.mixing.gradt_sub_grada.len() != nzn {
            model.aux.mixing.resize(nzn);
        }
        for k0 in 0..nzn {
            model.aux.mixing.gradt_sub_grada[k0] = model.aux.rsp.y_face_rsp[k0];
            if k0 > 0 {
                model.gradt[k0] = model.aux.rsp.y_face_rsp[k0]
                    + f64::midpoint(model.grada[k0 - 1], model.grada[k0]);
            }
            model.abar[k0] = chem.abar;
            model.zbar[k0] = chem.zbar;
            for j in 0..model.species.min(chem.xa.len()) {
                model.xa[k0 * model.species + j] = chem.xa[j];
            }
        }
        if nzn >= 2 {
            model.gradt[0] = model.gradt[1];
        }
        model.controls.calculate_brunt_n2 = true;
        do_brunt_b(model)?;
        do_brunt_n2(model);

        let gyre_txt = export_gyre_with_schema(model, false, 120);
        if let Some(dir) = log_dir {
            fs::write(dir.join("gyre.data"), &gyre_txt).map_err(|e| {
                StarError::IoError(format!(
                    "failed writing {}: {e}",
                    dir.join("gyre.data").display()
                ))
            })?;
        }
        let gyre_modes = solve_gyre_mesa_modes(&gyre_txt, 0, 0.5, 25.0, 400);
        if gyre_modes.is_empty() {
            return Err(StarError::ConvergenceFailure(String::from(
                "rsp_gyre BVP scan returned zero radial eigenmodes",
            )));
        }
        for (m_idx, mode) in gyre_modes.iter().take(3).enumerate() {
            let p_sec = 1.0 / (mode.freq_uhz * 1.0e-6).max(1.0e-30);
            model.aux.rsp.lina_periods[m_idx] = p_sec;
            model.aux.rsp.lina_growth_rates[m_idx] = mode.stellingwerf_eta;
            if let Some(dir) = log_dir {
                let mut ef = String::new();
                for (j, yj) in mode.y.iter().enumerate() {
                    let _ = writeln!(ef, "{:6} {:24.16e}", j + 1, yj[0]);
                }
                let ef_path = dir.join(format!("eigfunc.{}.dat", m_idx + 1));
                fs::write(&ef_path, ef).map_err(|e| {
                    StarError::IoError(format!(
                        "failed writing {}: {e}",
                        ef_path.display()
                    ))
                })?;
            }
        }
        let mode_idx = model
            .controls
            .rsp
            .mode_for_setting_periodlin
            .min(gyre_modes.len() - 1);
        model.aux.rsp.period =
            1.0 / (gyre_modes[mode_idx].freq_uhz * 1.0e-6).max(1.0e-30);

        let amix1 = model.controls.x_ctrl.get(3).copied().unwrap_or(0.0);
        let amix2 = model.controls.x_ctrl.get(4).copied().unwrap_or(0.0);
        let velkm = model
            .controls
            .x_ctrl
            .get(5)
            .copied()
            .filter(|&v| v != 0.0)
            .unwrap_or(model.controls.rsp.kick_vsurf_km_per_sec);
        let amix_f = 1.0 - (amix1 + amix2);

        let n_pts = gyre_modes[0].y.len();
        let xi_at = |m: usize, j: usize| -> f64 {
            gyre_modes.get(m).map_or(0.0, |md| {
                let k0 = nzn.saturating_sub(1 + j);
                let x_j = model.r[k0] / model.r[0].max(1.0e-30);
                md.y[j][0] / x_j.max(1.0e-30)
            })
        };
        let j_surf = n_pts - 1;
        let v_surf = amix_f * xi_at(0, j_surf)
            + amix1 * xi_at(1, j_surf)
            + amix2 * xi_at(2, j_surf);
        if v_surf.abs() > 0.0 && velkm != 0.0 {
            let scale = 1.0e5 * velkm / v_surf;
            for j in 0..n_pts.min(nzn) {
                let k0 = nzn - 1 - j;
                model.v[k0] = scale
                    * (amix_f * xi_at(0, j)
                        + amix1 * xi_at(1, j)
                        + amix2 * xi_at(2, j));
            }
            if n_pts < nzn && nzn >= 2 {
                model.v[0] = model.v[1];
            }
        }
        model.v_center = 0.0;
        model.aux.rsp.have_set_velocities = true;
        Ok(true)
    }
}

#[test]
fn test_rsp_gyre_bvp_linear_kick_and_nonlinear_hyd_steps() {
    let mut ctrl = StarControls {
        rsp_mass: 4.165,
        rsp_teff: 6050.0,
        rsp_l: 1438.8,
        rsp_x: 0.73,
        rsp_z: 0.007,
        zbase: 0.007,
        ..StarControls::default()
    };
    ctrl.rsp.use_other_rsp_linear_analysis = true;
    ctrl.x_ctrl[5] = 0.1;

    let kap = a09_kap_config(ctrl.rsp_z);
    let (mut model, lina_opt) = do_rsp_build_model(
        ctrl,
        base_rsp_job(),
        EosConfig::default(),
        kap,
        None,
    )
    .expect("build rsp_gyre");
    assert!(lina_opt.is_none());
    assert!(!model.aux.rsp.have_set_velocities);

    let mut extras = TestGyreRspExtras;
    for _ in 0..3 {
        let rep = star_evolve_step_with_extras(
            &mut model,
            false,
            true,
            true,
            &mut extras,
        )
        .expect("rsp_gyre step");
        assert!(rep.converged);
    }
    assert!(model.aux.rsp.have_set_velocities);
    assert!(model.v[0].abs() > 1.0e3, "v[0]={}", model.v[0]);
    let p_f_days = model.aux.rsp.lina_periods[0] / 86_400.0;
    assert!(
        (p_f_days - 3.9437).abs() / 3.9437 < 0.05,
        "GYRE radial F-mode period={p_f_days:.4} d"
    );
    let rel_e =
        (model.aux.cumulative_energy_error / model.aux.total_energy).abs();
    assert!(rel_e < 1.0e-6, "rsp_gyre rel_e={rel_e:.3e}");
}
