// SPDX-License-Identifier: LGPL-3.0-or-later
//! 1D stellar structure, ZAMS/pre-MS initialization, `.mod`/`.photo` model I/O,
//! and zone/face microphysics evaluation (`flexmesa-star`, Wave 4).
//!
//! Ports upstream Fortran MESA's `star_data` and `star` modules into Safe Rust
//! (`#![forbid(unsafe_code)]`) with `rayon` zone parallelism and 3-cell stencil
//! forward-mode automatic differentiation ([`StarAdOrder1`]).

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::fn_params_excessive_bools,
    clippy::inline_always,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod controls;
pub mod eps_grav;
pub mod hydro_eqns;
pub mod hydro_vars;
pub mod mesh;
pub mod micro;
pub mod model;
pub mod model_io;
pub mod pre_ms;
pub mod solver;
pub mod star_ad;
pub mod zams;

pub use controls::{
    AtmOption, AtmTTauRelation, EnergyEqnOption, StarControls, StarError,
    StarJobControls, VarLayout,
};
pub use eps_grav::{
    eval_eps_grav, eval_eps_grav_lns_cell, eval_eps_grav_std_cell,
};
pub use hydro_eqns::{
    compute_x_scale, do1_chem_eqns, do1_density_eqn, do1_energy_eqn,
    do1_momentum_eqn, do1_radius_eqn, do1_temperature_eqn, do_surf_bc_eqns,
    eval_henyey_equations, HenyeySystem,
};
pub use hydro_vars::{
    check_for_redo_mlt, do_brunt_b, do_brunt_n2, eval_surf_bc, set_hydro_vars,
    set_mlt_vars, update_sig_from_d_mix,
};
pub use micro::{
    do_eos, do_kap, do_net, do_neu, get_tau, global_basic_net, set_basic_vars,
};
pub use model::StarModel;
pub use model_io::{
    read_mod_str, read_photo_bytes, write_mod_string, write_photo_bytes,
    STAR_DEF_VERSION,
};
pub use pre_ms::{
    build1_pre_ms_model, build_pre_ms_model, default_basic_net_composition,
    PreMsShell,
};
pub use solver::{
    compute_corr_coeff, compute_correction_norms, compute_vector_norms,
    solve_henyey_step, HenyeySolverReport,
};
pub use star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LND_P1, I_LNR_00, I_LNR_M1,
    I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1, I_L_00, I_L_M1, I_L_P1, I_OMEGA_00,
    I_OMEGA_M1, I_OMEGA_P1, I_V_00, I_V_M1, I_V_P1, NUM_STAR_AD_VARS,
};
pub use zams::{
    interp_q, load_zams_model, load_zams_model_from_catalog,
    yrs_for_init_timestep, ZamsCatalog, ZamsEntry,
};

#[cfg(test)]
mod tests {
    use super::*;
    use flexmesa_const::{MixingType, M_SUN, SECYER};

    #[test]
    fn test_star_ad_order1_shifts_and_chain_rule_match_analytic_derivatives() {
        let lnd_00 = StarAdOrder1::variable(2.0, I_LND_00);
        let lnt_00 = StarAdOrder1::variable(16.0, I_LNT_00);
        let f_cell = (lnd_00 * 1.5 + lnt_00 * 0.5).exp();
        let expected_val = (2.0_f64 * 1.5 + 16.0 * 0.5).exp();
        assert!((f_cell.val - expected_val).abs() <= 1.0e-12 * expected_val);
        assert!(
            (f_cell.d1[I_LND_00] - 1.5 * expected_val).abs()
                <= 1.0e-12 * expected_val
        );
        assert!(
            (f_cell.d1[I_LNT_00] - 0.5 * expected_val).abs()
                <= 1.0e-12 * expected_val
        );

        let shifted_m1 = f_cell.shift_m1();
        assert_eq!(shifted_m1.d1[I_LND_M1], f_cell.d1[I_LND_00]);
        assert_eq!(shifted_m1.d1[I_LNT_M1], f_cell.d1[I_LNT_00]);
        assert_eq!(shifted_m1.d1[I_LND_00], 0.0);

        let shifted_p1 = f_cell.shift_p1();
        assert_eq!(shifted_p1.d1[I_LND_P1], f_cell.d1[I_LND_00]);
        assert_eq!(shifted_p1.d1[I_LNT_P1], f_cell.d1[I_LNT_00]);
        assert_eq!(shifted_p1.d1[I_LND_00], 0.0);
    }

    #[test]
    fn test_zams_4msun_star_test_interpolation_and_initial_timestep() {
        let controls = StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            ..StarControls::default()
        };
        let job = StarJobControls::default();

        let yrs = yrs_for_init_timestep(4.0);
        assert!((yrs - 3125.0).abs() <= 1.0e-12);

        let model =
            load_zams_model(controls, job).expect("load 4 Msun ZAMS model");
        assert_eq!(model.nz, 2464);
        assert_eq!(model.species, 8);
        assert!((model.mstar - 4.0 * M_SUN).abs() <= 1.0e-12 * model.mstar);
        assert!((model.dt - 3125.0 * SECYER).abs() <= 1.0e-12 * model.dt);
        assert!((model.q[0] - 1.0).abs() <= 1.0e-15);

        let dq_sum: f64 = model.dq.iter().sum();
        assert!((dq_sum - 1.0).abs() <= 1.0e-14);
        for k in [0, model.nz / 2, model.nz - 1] {
            let xa_sum: f64 = model.cell_xa(k).iter().sum();
            assert!((xa_sum - 1.0).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn test_mod_and_photo_roundtrip_preserve_state() {
        let controls = StarControls {
            initial_mass: 1.0,
            ..StarControls::default()
        };
        let job = StarJobControls::default();
        let model = load_zams_model(controls.clone(), job.clone())
            .expect("load 1 Msun ZAMS");

        let mod_text = write_mod_string(&model).expect("write .mod");
        let model_from_mod =
            read_mod_str(&mod_text, controls.clone(), job.clone())
                .expect("read .mod");
        assert_eq!(model_from_mod.nz, model.nz);
        assert_eq!(model_from_mod.species, model.species);
        for k in [0, model.nz / 2, model.nz - 1] {
            assert!((model_from_mod.lnd[k] - model.lnd[k]).abs() <= 1.0e-14);
            assert!((model_from_mod.ln_t[k] - model.ln_t[k]).abs() <= 1.0e-14);
            assert!((model_from_mod.ln_r[k] - model.ln_r[k]).abs() <= 1.0e-14);
            assert!(
                (model_from_mod.lum[k] - model.lum[k]).abs()
                    <= 1.0e-14 * model.lum[k].abs().max(1.0)
            );
        }

        let photo_bytes = write_photo_bytes(&model);
        let model_from_photo =
            read_photo_bytes(&photo_bytes, controls, job).expect("read .photo");
        assert_eq!(model_from_photo.lnd, model.lnd);
        assert_eq!(model_from_photo.ln_t, model.ln_t);
        assert_eq!(model_from_photo.ln_r, model.ln_r);
        assert_eq!(model_from_photo.lum, model.lum);
        assert_eq!(model_from_photo.dq, model.dq);
        assert_eq!(model_from_photo.xa, model.xa);
    }

    #[test]
    fn test_set_hydro_vars_on_4msun_zams_model_populates_convective_core() {
        let controls = StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            ..StarControls::default()
        };
        let job = StarJobControls::default();
        let mut model = load_zams_model(controls, job).expect("load 4 Msun");

        set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

        let k_center = model.nz - 1;
        assert!(model.peos[k_center] > 1.0e16);
        assert!(model.t[k_center] > 2.0e7);
        assert!(model.eps_nuc[k_center] > 100.0);
        assert!(model.teff > 1.0e4 && model.teff < 2.5e4);
        assert!(model.ln_peos_surf_ad.val.is_finite());
        assert!(model.ln_t_surf_ad.val.is_finite());

        // 4 Msun ZAMS star has a convective core (`MixingType::ConvectiveMixing`)
        let n_conv_core = (model.nz - 200..model.nz)
            .filter(|&k| model.mixing_type[k] == MixingType::ConvectiveMixing)
            .count();
        assert!(n_conv_core > 50);
    }

    #[test]
    fn test_build_pre_ms_model_1msun_converges_to_convective_polytrope() {
        let controls = StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            ..StarControls::default()
        };
        let job = StarJobControls {
            create_pre_ms_model: true,
            pre_ms_d_log10_p: 0.04,
            ..StarJobControls::default()
        };

        let mut model =
            build_pre_ms_model(controls, job).expect("build 1 Msun pre-MS");
        assert!(model.nz > 50);
        set_hydro_vars(&mut model, true, true)
            .expect("set_hydro_vars on pre-MS");
        assert!(model.t[model.nz - 1] > 5.0e5);
        assert!(model.peos[model.nz - 1] > model.peos[0]);
    }

    #[test]
    fn test_henyey_jacobian_blocks_match_central_finite_differences() {
        let controls = StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            ..StarControls::default()
        };
        let job = StarJobControls::default();
        let mut model = load_zams_model(controls, job).expect("load 4 Msun");
        model.dt = 1.0;
        set_hydro_vars(&mut model, false, false).expect("init hydro vars");
        model.save_start_values();

        let eps = 1.0e-7;
        for include_chem in [false, true] {
            let sys = eval_henyey_equations(&mut model, include_chem)
                .expect("eval equations");
            let nvar = sys.nvar;
            let blk_sz = nvar * nvar;
            let check_blk = |blk: &[f64],
                             sys_p: &HenyeySystem,
                             sys_m: &HenyeySystem,
                             k_eq: usize,
                             i_equ: usize,
                             j_var: usize,
                             name: &str| {
                let idx_eq = k_eq * nvar + i_equ;
                let fd = (sys_p.equ[idx_eq] - sys_m.equ[idx_eq]) / (2.0 * eps);
                let ad = blk[k_eq * blk_sz + i_equ * nvar + j_var];
                let tol = 5.0e-5 * ad.abs().max(fd.abs()).max(1.0);
                assert!(
                    (ad - fd).abs() <= tol,
                    "{name} mismatch (chem={include_chem}) at k={k_eq}, \
                     equ={i_equ}, var={j_var}: ad={ad}, fd={fd}"
                );
            };

            let nvar_hydro = model.layout.nvar_hydro;
            let k_list = if include_chem {
                vec![model.nz - 2]
            } else {
                vec![0, model.nz / 2, model.nz - 1]
            };
            for k_var in k_list {
                for j_var in 0..nvar {
                    let scale = sys.x_scale[k_var * nvar + j_var];
                    let delta = eps * scale;

                    let perturb = |sign: f64| -> HenyeySystem {
                        let mut m = model.clone();
                        if j_var < nvar_hydro {
                            m.xh[k_var * nvar_hydro + j_var] += sign * delta;
                            m.unpack_xh();
                            m.set_m_grav_and_grav();
                        } else {
                            let j_iso = j_var - nvar_hydro;
                            m.xa[k_var * m.species + j_iso] += sign * delta;
                        }
                        eval_henyey_equations(&mut m, include_chem)
                            .expect("perturbed eval")
                    };
                    let sys_p = perturb(1.0);
                    let sys_m = perturb(-1.0);

                    let i_start = if j_var < nvar_hydro {
                        0
                    } else {
                        nvar_hydro
                    };
                    for i_equ in i_start..nvar {
                        check_blk(
                            &sys.dblk, &sys_p, &sys_m, k_var, i_equ, j_var,
                            "dblk",
                        );
                        if k_var + 1 < model.nz {
                            check_blk(
                                &sys.lblk,
                                &sys_p,
                                &sys_m,
                                k_var + 1,
                                i_equ,
                                j_var,
                                "lblk",
                            );
                        }
                        if k_var > 0 {
                            check_blk(
                                &sys.ublk,
                                &sys_p,
                                &sys_m,
                                k_var - 1,
                                i_equ,
                                j_var,
                                "ublk",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_solve_henyey_step_converges_on_perturbed_model() {
        let controls = StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            max_newton_iters: 25,
            ..StarControls::default()
        };
        let job = StarJobControls {
            create_pre_ms_model: true,
            pre_ms_d_log10_p: 0.15,
            ..StarJobControls::default()
        };
        let mut model =
            build_pre_ms_model(controls, job).expect("build pre-MS");
        model.dt = 10.0 * SECYER;
        set_hydro_vars(&mut model, false, false).expect("init hydro");
        model.save_start_values();

        let mut model_thomas = model.clone();
        let report_bcyclic =
            solve_henyey_step(&mut model, false, true).expect("solve bcyclic");
        assert!(
            report_bcyclic.converged,
            "expected bcyclic convergence, got {report_bcyclic:?}"
        );
        assert!(
            report_bcyclic.residual_norm <= model.controls.tol_residual_norm
        );

        let report_thomas = solve_henyey_step(&mut model_thomas, false, false)
            .expect("solve thomas");
        assert!(
            report_thomas.converged,
            "expected thomas convergence, got {report_thomas:?}"
        );
        for k in 0..model.nz {
            let diff_d = (model.lnd[k] - model_thomas.lnd[k]).abs();
            assert!(
                diff_d <= 1.0e-6,
                "bcyclic vs thomas lnd mismatch at zone {k}: diff={diff_d}"
            );
            let diff_t = (model.ln_t[k] - model_thomas.ln_t[k]).abs();
            assert!(
                diff_t <= 1.0e-6,
                "bcyclic vs thomas ln_t mismatch at zone {k}: diff={diff_t}"
            );
        }

        // Now test coupled hydro + chem solve (`include_chem = true`)
        model.save_start_values();
        let report_chem =
            solve_henyey_step(&mut model, true, true).expect("solve coupled");
        assert!(
            report_chem.converged,
            "expected coupled hydro+chem convergence, got {report_chem:?}"
        );
    }
}
