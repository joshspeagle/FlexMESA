// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone, zero-`$MESA_DIR` MESA `26.4.1` parity tests for nonlinear `RSP`
//! hydrodynamics/turbulence equations, banded Jacobian assembly/solve,
//! turbulent partials, and driver setup/step logic (`#162`).
//!
//! All tests inspect only in-memory arithmetic or built-in envelope physics
//! without consulting `$MESA_DIR` or external `.mod` test fixtures.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::collections::HashMap;
use std::path::PathBuf;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CRAD, L_SUN, M_SUN, PI, R_SUN, SECYER};
use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_math::{
    intrinsic_log, intrinsic_pow, log, pow2, pow3, powi_f90,
    with_numerics_mode, NumericsMode,
};
use flexmesa_schema::ControlValidationReport;
use flexmesa_star::controls::{StarControls, StarError, StarJobControls};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::{DefaultStarExtras, StarExtras};
use flexmesa_star::model::StarModel;
use flexmesa_star::rsp_build::do_rsp_build_model;
use flexmesa_star::rsp_config::RspConfig;
use flexmesa_star::rsp_def::{RspDerivedConsts, F_EDD_ISOTROPIC};
use flexmesa_star::rsp_driver::{
    rsp_one_step_with_extras, rsp_setup_model, rsp_total_energy_integrals,
};
use flexmesa_star::rsp_eos_kap::RspChemCache;
use flexmesa_star::rsp_eqns::{
    eval_eqns, solve_for_corrections, RspHydBand, HD_DIAG, LD_HD,
};
use flexmesa_star::rsp_grid::RspEnvGrid;
use flexmesa_star::rsp_hyd_turb::{calc_chi, calc_eq, calc_source_sink};
use flexmesa_star::rsp_hyd_vars::{do1_specific_volume, zero_boundaries};
use flexmesa_star::rsp_hyd_work::RspHydWork;
use flexmesa_star::rsp_turb_partials::RspTurbPartials;
use flexmesa_star::rsp_zone_micro::{
    eval_hp_face_partials, HpFacePartials, LcFilterMode, RspZoneMicro,
};
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;

fn build_synthetic_rsp_model(nz: usize, hydro_only: bool) -> StarModel {
    let cat = NuclideCatalog::builtin_ref();
    let h1 = cat.get_nuclide_index("h1").expect("h1");
    let he4 = cat.get_nuclide_index("he4").expect("he4");

    let ctrl = StarControls {
        rsp_flag: true,
        v_flag: true,
        rsp_mass: 2.0,
        rsp_teff: 6900.0,
        rsp_l: 30.0,
        rsp_x: 0.70,
        rsp_z: 0.02,
        zbase: 0.02,
        rsp: RspConfig {
            nz,
            hydro_only,
            alfa: 1.5,
            alfac: 1.0,
            alfas: 1.0,
            alfad: 1.0,
            alfap: 0.2,
            alfat: 0.05,
            alfam: 0.25,
            gammar: 0.0,
            ..RspConfig::default()
        },
        ..StarControls::default()
    };

    let mut model =
        StarModel::new(nz, vec![h1, he4], StarJobControls::default(), ctrl);
    model.m_center = 1.9 * M_SUN;
    model.xmstar = 0.1 * M_SUN;
    model.mstar = model.m_center + model.xmstar;
    model.r_center = 1.1 * R_SUN;
    model.l_center = 30.0 * L_SUN;
    model.v_center = 0.0;
    model.dt = 25.0;
    model.dt_next = 25.0;

    let dm_cell = model.xmstar / (nz as f64);
    for k0 in 0..nz {
        // Cell `k0 = 0` is surface, `k0 = nz - 1` is bottom.
        let i_smolec = nz - k0;
        let frac = (i_smolec as f64) / (nz as f64);
        let r_in =
            model.r_center + ((i_smolec - 1) as f64) / (nz as f64) * R_SUN;
        let r_out = model.r_center + frac * R_SUN;
        model.dm[k0] = dm_cell;
        model.dm_bar[k0] = dm_cell;
        model.m[k0] = model.m_center + frac * model.xmstar;
        model.m_grav[k0] = model.m[k0];
        model.r[k0] = r_out;
        model.ln_r[k0] = log(r_out);
        model.aux.rsp.r_start_rsp[k0] = r_out - 2.0e3 * frac;
        model.v[k0] = 1.0e5 * frac;
        model.aux.rsp.v_start_rsp[k0] = 0.9e5 * frac;

        let t_k = 5.0e3 + 3.0e5 * (1.0 - frac + 1.0 / (nz as f64));
        model.t[k0] = t_k;
        model.ln_t[k0] = log(t_k);
        model.aux.rsp.t_start_rsp[k0] = t_k;

        let vol_k = (4.0 * PI / 3.0) * (pow3(r_out) - pow3(r_in)) / dm_cell;
        model.aux.rsp.vol_rsp[k0] = vol_k;
        model.aux.rsp.vol_start[k0] = vol_k * 0.999;
        model.rho[k0] = 1.0 / vol_k;
        model.lnd[k0] = log(model.rho[k0]);

        model.aux.rsp.f_edd_rsp[k0] = F_EDD_ISOTROPIC;
        model.aux.rsp.erad_rsp[k0] = CRAD * powi_f90(t_k, 4) * vol_k;
        model.aux.rsp.erad_start[k0] = model.aux.rsp.erad_rsp[k0] * 0.9995;
        model.prad[k0] = F_EDD_ISOTROPIC * model.aux.rsp.erad_rsp[k0] / vol_k;
        model.aux.rsp.prad_start[k0] = model.prad[k0];

        model.pgas[k0] = 1.0e8 * (1.01 - 0.8 * frac);
        model.aux.rsp.pgas_start[k0] = model.pgas[k0];
        model.peos[k0] = model.pgas[k0] + model.prad[k0];
        model.aux.rsp.egas_rsp[k0] = 1.5 * model.pgas[k0] * vol_k;
        model.aux.rsp.egas_start[k0] = model.aux.rsp.egas_rsp[k0] * 0.999;

        model.cp[k0] = 2.5e8;
        model.aux.rsp.qq_rsp[k0] = 1.0 / t_k;
        model.opacity[k0] = 0.5 + 4.0 * (1.0 - frac);
        model.csound[k0] = 3.0e6 * (1.05 - 0.4 * frac);

        let w_k = if i_smolec > 1 && i_smolec < nz {
            1.2e4 * (1.0 + 0.1 * frac)
        } else {
            0.0
        };
        model.aux.rsp.et_rsp[k0] = w_k * w_k;
        model.aux.rsp.w_start[k0] = w_k * 0.98;
        model.aux.rsp.hp_face[k0] = 5.0e8;
        model.aux.rsp.pii_rsp[k0] = 1.0e7 * (1.0 + 0.2 * frac);
        model.aux.rsp.fr_rsp[k0] = model.l_center / (4.0 * PI * pow2(r_out));
        model.aux.rsp.fr_start[k0] = model.aux.rsp.fr_rsp[k0];
        model.lum[k0] = model.l_center;
        model.lum_start[k0] = model.l_center;
    }
    model
}

fn rel_err(got: f64, expected: f64) -> f64 {
    let denom = expected.abs().max(1.0e-300);
    (got - expected).abs() / denom
}

fn band_col_value(band: &RspHydBand, r_row: usize, c_col: usize) -> f64 {
    let diff = c_col as isize - r_row as isize + HD_DIAG as isize;
    if (1..=(LD_HD as isize)).contains(&diff) {
        band.hd[r_row][diff as usize]
    } else {
        0.0
    }
}

#[test]
fn test_rsp_eqns_cell_and_face_residuals_jacobian_fd_and_band_solve() {
    with_numerics_mode(NumericsMode::Strict, || {
        let nz = 32;
        let mut model = build_synthetic_rsp_model(nz, false);
        let dc = RspDerivedConsts::from_config(&model.controls.rsp, nz);
        let mut wk = RspHydWork::from_model(&model);
        for i in 1..=nz {
            do1_specific_volume(&mut model, &dc, &mut wk, i)
                .expect("specific volume");
            calc_chi(&mut model, &dc, &mut wk, i);
            calc_eq(&mut model, &dc, &mut wk, i);
            calc_source_sink(&mut model, &dc, &mut wk, i);
        }
        zero_boundaries(&mut model, &dc, &mut wk);

        let mut band = RspHydBand::new(nz);
        eval_eqns(&mut model, &dc, &wk, &mut band, 0.0);

        // 1. Inner boundary equation: row 1 enforces delta R_0 = 0.
        assert_eq!(band.hr[1], 0.0);
        for (j, &val) in band.hd[1].iter().enumerate().take(LD_HD + 1).skip(1) {
            let expected = if j == HD_DIAG { 1.0 } else { 0.0 };
            assert_eq!(val, expected, "inner BC col {j}");
        }

        // 2. Surface radiative flux boundary (i == nzn, k0 == 0): Fr = 2 sigma T^4.
        let (_, _, ie_surf, il_surf, _) = RspHydBand::row_indices(nz);
        let t_surf = model.t[0];
        let fr_surf_expected = 2.0 * dc.sig * powi_f90(t_surf, 4);
        assert!(
            rel_err(
                -band.hr[il_surf],
                fr_surf_expected - model.aux.rsp.fr_rsp[0]
            ) < 1.0e-14
        );
        assert_eq!(band_col_value(&band, il_surf, il_surf), -1.0);
        let (it_surf, _, _, _, _) = RspHydBand::row_indices(nz);
        assert!(
            rel_err(
                band_col_value(&band, il_surf, it_surf),
                4.0 * fr_surf_expected / t_surf
            ) < 1.0e-14
        );

        // 3. T_form_of_erad_eqn + total_energy_eqn ergs_error parity (`rsp_step.f90:3256, 3668-3676`).
        let erad_exp_surf =
            CRAD * powi_f90(t_surf, 4) * model.aux.rsp.vol_rsp[0];
        assert!(
            rel_err(
                -band.hr[ie_surf],
                erad_exp_surf - model.aux.rsp.erad_rsp[0]
            ) < 1.0e-14
        );
        assert_eq!(band_col_value(&band, ie_surf, ie_surf), -1.0);
        assert!(
            rel_err(
                band_col_value(&band, ie_surf, it_surf),
                4.0 * CRAD * pow3(t_surf) * model.aux.rsp.vol_rsp[0]
            ) < 1.0e-14
        );

        for i in 1..=nz {
            let k0 = nz - i;
            let (it, _, _, _, _) = RspHydBand::row_indices(i);
            let expected_ergs = model.dm[k0] * (-band.hr[it]);
            assert_eq!(
                model.aux.ergs_error[k0].to_bits(),
                expected_ergs.to_bits(),
                "ergs_error exact bit parity at zone i={i}"
            );
        }

        // 4. Boundary turbulent equation identity for i <= ibotom and i == nzn.
        for i in [1_usize, dc.ibotom, nz] {
            let (_, iw, _, _, _) = RspHydBand::row_indices(i);
            assert_eq!(band.hr[iw], 0.0);
            for c in 1..=band.n {
                let expected = if c == iw { 1.0 } else { 0.0 };
                assert_eq!(band_col_value(&band, iw, c), expected);
            }
        }

        // 5. Dense verification of banded LU assembly and solve (`solve_for_corrections`).
        let n = band.n;
        let mut a_dense = vec![0.0_f64; n * n];
        for r in 1..=n {
            for (j_hd, &val) in
                band.hd[r].iter().enumerate().take(LD_HD + 1).skip(1)
            {
                let c = r as isize + j_hd as isize - HD_DIAG as isize;
                if c >= 1 && c <= n as isize {
                    a_dense[(r - 1) * n + (c as usize - 1)] = val;
                }
            }
        }
        let b_orig = band.hr[1..=n].to_vec();
        solve_for_corrections(&mut band).expect("band solve succeeds");
        let dx = &band.hr[1..=n];

        let mut max_ax_b = 0.0_f64;
        let mut max_a_inf = 0.0_f64;
        for r in 0..n {
            let mut row_dot = 0.0_f64;
            let mut row_abs = 0.0_f64;
            for c in 0..n {
                let a = a_dense[r * n + c];
                row_dot += a * dx[c];
                row_abs += a.abs();
            }
            max_ax_b = max_ax_b.max((row_dot - b_orig[r]).abs());
            max_a_inf = max_a_inf.max(row_abs);
        }
        let dx_inf = dx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let b_inf = b_orig.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let rel_solve_res = max_ax_b / (max_a_inf * dx_inf + b_inf);
        assert!(
            rel_solve_res < 1.0e-12,
            "banded solver dense residual {rel_solve_res:.3e}"
        );

        // 6. Finite-difference verification of Fr and w analytical Jacobian columns.
        let mut fd_model = build_synthetic_rsp_model(nz, false);
        // Zero eddy viscosity and coupling to isolate live eval_eqns w-terms (Lt, Lc, w^2),
        // and use a larger `dt` with matched `w_start` to keep FD subtraction well-conditioned.
        fd_model.controls.rsp.alfam = 0.0;
        fd_model.controls.rsp.alfap = 0.0;
        fd_model.dt = 2.5e4;
        let dc_fd = RspDerivedConsts::from_config(&fd_model.controls.rsp, nz);
        let mut wk_fd = RspHydWork::from_model(&fd_model);
        for i in 1..=nz {
            do1_specific_volume(&mut fd_model, &dc_fd, &mut wk_fd, i)
                .expect("fd specific volume");
        }
        fd_model.aux.rsp.w_start.copy_from_slice(&wk_fd.w);
        fd_model
            .aux
            .rsp
            .egas_start
            .copy_from_slice(&fd_model.aux.rsp.egas_rsp);
        fd_model
            .aux
            .rsp
            .erad_start
            .copy_from_slice(&fd_model.aux.rsp.erad_rsp);
        let mut band_ref = RspHydBand::new(nz);
        eval_eqns(&mut fd_model, &dc_fd, &wk_fd, &mut band_ref, 0.0);

        let i_probe = 12_usize;
        let k_probe = nz - i_probe;
        let (it_p, iw_p, _, il_p, _) = RspHydBand::row_indices(i_probe);
        let (it_o, iw_o, _, _, _) = RspHydBand::row_indices(i_probe + 1);
        let (it_m, iw_m, _, _, _) = RspHydBand::row_indices(i_probe - 1);

        // FD wrt w_k (`wk.w[k_probe]`):
        let dw = 1.0e-4 * wk_fd.w[k_probe];
        let mut wk_plus = wk_fd.clone();
        wk_plus.w[k_probe] += dw;
        let mut wk_minus = wk_fd.clone();
        wk_minus.w[k_probe] -= dw;
        let mut b_plus = RspHydBand::new(nz);
        let mut b_minus = RspHydBand::new(nz);
        eval_eqns(&mut fd_model, &dc_fd, &wk_plus, &mut b_plus, 0.0);
        eval_eqns(&mut fd_model, &dc_fd, &wk_minus, &mut b_minus, 0.0);
        for &(row, label) in &[
            (it_p, "dEtot_00/dw"),
            (it_o, "dEtot_out/dw"),
            (it_m, "dEtot_in/dw"),
            (iw_p, "dEturb_00/dw"),
            (iw_o, "dEturb_out/dw"),
            (iw_m, "dEturb_in/dw"),
        ] {
            let fd_deriv =
                ((-b_plus.hr[row]) - (-b_minus.hr[row])) / (2.0 * dw);
            let ana_deriv = band_col_value(&band_ref, row, iw_p);
            assert!(
                rel_err(ana_deriv, fd_deriv) < 1.0e-8,
                "{label}: ana={ana_deriv:.8e}, fd={fd_deriv:.8e}"
            );
        }

        // FD wrt Fr_k (`fr_rsp[k_probe]`; equations are linear in Fr so larger step avoids cancellation):
        let dfr = 1.0e-3 * fd_model.aux.rsp.fr_rsp[k_probe].abs();
        let fr_orig = fd_model.aux.rsp.fr_rsp[k_probe];
        fd_model.aux.rsp.fr_rsp[k_probe] = fr_orig + dfr;
        eval_eqns(&mut fd_model, &dc_fd, &wk_fd, &mut b_plus, 0.0);
        fd_model.aux.rsp.fr_rsp[k_probe] = fr_orig - dfr;
        eval_eqns(&mut fd_model, &dc_fd, &wk_fd, &mut b_minus, 0.0);
        fd_model.aux.rsp.fr_rsp[k_probe] = fr_orig;
        for &(row, label) in &[
            (il_p, "dFr/dFr_00"),
            (it_p, "dEtot_00/dFr"),
            (it_o, "dEtot_out/dFr"),
        ] {
            let fd_deriv =
                ((-b_plus.hr[row]) - (-b_minus.hr[row])) / (2.0 * dfr);
            let ana_deriv = band_col_value(&band_ref, row, il_p);
            assert!(
                rel_err(ana_deriv, fd_deriv) < 1.0e-9,
                "{label}: ana={ana_deriv:.8e}, fd={fd_deriv:.8e}"
            );
        }

        // 7. `hydro_only = true` sets `Fr_eqn` to `-Fr` and zeroes core `L_in`.
        let mut hydro_m = build_synthetic_rsp_model(nz, true);
        let dc_h = RspDerivedConsts::from_config(&hydro_m.controls.rsp, nz);
        let wk_h = RspHydWork::from_model(&hydro_m);
        let mut band_h = RspHydBand::new(nz);
        eval_eqns(&mut hydro_m, &dc_h, &wk_h, &mut band_h, 0.0);
        let (it1, _, _, il1, _) = RspHydBand::row_indices(1);
        assert!(
            rel_err(band_h.hr[il1], hydro_m.aux.rsp.fr_rsp[nz - 1]) < 1.0e-15
        );
        assert_eq!(band_col_value(&band_h, il1, il1), -1.0);
        assert_eq!(band_col_value(&band_h, il1, it1), 0.0);
        assert_eq!(
            band_col_value(&band_h, it1, il1),
            hydro_m.dt / hydro_m.dm[nz - 1]
                * dc_h.wtr
                * 4.0
                * PI
                * pow2(hydro_m.r[nz - 1])
        );
    });
}

#[test]
fn test_rsp_hyd_flux_ibotom_zero_matches_mesa_calc_lt_and_lc() {
    // `nz = 10` with `nz_div_ibotom = 30.0` gives `ibotom = 0` (`rsp_step.f90:2158, 2237`).
    let mut model = build_synthetic_rsp_model(10, false);
    model.aux.rsp.min_k_for_turbulent_flux = 0;
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, 10);
    assert_eq!(dc.ibotom, 0);

    let mut wk = RspHydWork::from_model(&model);
    let nzn = model.nz;
    let k0 = nzn - 1;
    let ko = k0 - 1;
    wk.w[k0] = 8.0e3;
    wk.w[ko] = 1.4e4;
    model.aux.rsp.et_rsp[k0] = wk.w[k0] * wk.w[k0];
    model.aux.rsp.et_rsp[ko] = wk.w[ko] * wk.w[ko];

    let mut band = RspHydBand::new(nzn);
    eval_eqns(&mut model, &dc, &wk, &mut band, 0.0);

    let pom3 = (pow3(wk.w[ko]) - pow3(wk.w[k0])) / model.dm_bar[k0];
    let pom = -2.0 / 3.0 * dc.alfa * dc.alfat * pow2(dc.p4 * pow2(model.r[k0]));
    let v0 = model.aux.rsp.vol_rsp[k0];
    let vo = model.aux.rsp.vol_rsp[ko];
    let expected_lt = pom
        * model.aux.rsp.hp_face[k0]
        * f64::midpoint(1.0 / (v0 * v0), 1.0 / (vo * vo))
        * pom3;
    assert!(
        expected_lt.abs() > 0.0
            && rel_err(model.aux.rsp.lt_rsp[k0], expected_lt) < 1.0e-14,
        "ibotom=0 calc_Lt got {}, expected {expected_lt}",
        model.aux.rsp.lt_rsp[k0]
    );

    // Gating by `min_k_for_turbulent_flux` (`k_f = nzn + 1 - 1 = nzn < min_k`).
    model.aux.rsp.min_k_for_turbulent_flux = nzn + 1;
    eval_eqns(&mut model, &dc, &wk, &mut band, 0.0);
    assert_eq!(model.aux.rsp.lt_rsp[k0], 0.0);
}

#[test]
fn test_rsp_hyd_turb_and_zero_boundaries_parity() {
    let nz = 32;
    let mut model = build_synthetic_rsp_model(nz, false);
    model.controls.rsp.gammar = 0.5;
    let dc_active = RspDerivedConsts::from_config(&model.controls.rsp, nz);
    assert!(dc_active.ibotom >= 1);

    let mut wk = RspHydWork::from_model(&model);
    for i in 1..=nz {
        do1_specific_volume(&mut model, &dc_active, &mut wk, i)
            .expect("specific volume");
        calc_chi(&mut model, &dc_active, &mut wk, i);
        calc_eq(&mut model, &dc_active, &mut wk, i);
        calc_source_sink(&mut model, &dc_active, &mut wk, i);
    }

    let i_test = dc_active.ibotom + 2;
    let k0 = nz - i_test;
    let ki = k0 + 1;

    // Exact closed-form checks for Chi, Eq, SOURCE, DAMP, DAMPR, COUPL (`rsp_step.f90:1639-2033`).
    let vol = model.aux.rsp.vol_rsp[k0];
    let chi_expected = (16.0 / 3.0) * PI * dc_active.alfa * dc_active.alfam
        / model.dm[k0]
        * 0.5
        * (pow3(model.r[k0]) * pow3(model.r[k0])
            + pow3(model.r[ki]) * pow3(model.r[ki]))
        * (model.v[k0] / model.r[k0] - model.v[ki] / model.r[ki])
        * 0.5
        * (model.aux.rsp.hp_face[k0] + model.aux.rsp.hp_face[ki])
        * (wk.w[k0] / (vol * vol));
    assert!(rel_err(model.aux.rsp.chi_rsp[k0], chi_expected) < 1.0e-13);

    let rri = f64::midpoint(model.r[k0], model.aux.rsp.r_start_rsp[k0]);
    let rrm = f64::midpoint(model.r[ki], model.aux.rsp.r_start_rsp[ki]);
    let uui = f64::midpoint(model.v[k0], model.aux.rsp.v_start_rsp[k0]);
    let uum = f64::midpoint(model.v[ki], model.aux.rsp.v_start_rsp[ki]);
    let eq_expected = dc_active.p4 / model.dm[k0]
        * (uui / rri - uum / rrm)
        * (dc_active.thetau * model.aux.rsp.chi_rsp[k0]
            + dc_active.thetau1 * model.aux.rsp.chi_start[k0]);
    assert!(rel_err(model.aux.rsp.eq_rsp[k0], eq_expected) < 1.0e-14);

    let p_k = model.pgas[k0] + model.prad[k0];
    let hp0 = model.aux.rsp.hp_face[k0];
    let hpi = model.aux.rsp.hp_face[ki];
    let dampr_expected = ((dc_active.gammar * dc_active.gammar)
        / (dc_active.alfa * dc_active.alfa)
        * 4.0
        * dc_active.sig)
        * (pow3(model.t[k0]) * vol * vol / (model.cp[k0] * model.opacity[k0]))
        * (wk.w[k0] * wk.w[k0])
        / f64::midpoint(hp0 * hp0, hpi * hpi);
    assert!(
        dampr_expected > 0.0
            && rel_err(model.aux.rsp.dampr_rsp[k0], dampr_expected) < 1.0e-14
    );
    assert!(
        rel_err(
            model.aux.rsp.coupl_rsp[k0],
            model.aux.rsp.source_rsp[k0]
                - model.aux.rsp.damp_rsp[k0]
                - model.aux.rsp.dampr_rsp[k0]
        ) < 1.0e-14
    );
    let _ = p_k;

    // Verify `calc_source_sink` zeroes `wk.dc_*` partials in its guard branch (`alfa == 0`).
    let mut band_active = RspHydBand::new(nz);
    eval_eqns(&mut model, &dc_active, &wk, &mut band_active, 0.0);
    let (_, iw_test, _, _, _) = RspHydBand::row_indices(i_test);
    let diag_with_dc = band_col_value(&band_active, iw_test, iw_test);

    let mut cfg_zero = model.controls.rsp.clone();
    cfg_zero.alfa = 0.0;
    let dc_zero = RspDerivedConsts::from_config(&cfg_zero, nz);
    calc_source_sink(&mut model, &dc_zero, &mut wk, i_test);
    assert_eq!(model.aux.rsp.source_rsp[k0], 0.0);
    assert_eq!(model.aux.rsp.damp_rsp[k0], 0.0);
    assert_eq!(model.aux.rsp.dampr_rsp[k0], 0.0);
    assert_eq!(model.aux.rsp.coupl_rsp[k0], 0.0);

    let mut band_cleared = RspHydBand::new(nz);
    eval_eqns(&mut model, &dc_active, &wk, &mut band_cleared, 0.0);
    let diag_without_dc = band_col_value(&band_cleared, iw_test, iw_test);
    assert_ne!(
        diag_with_dc.to_bits(),
        diag_without_dc.to_bits(),
        "stale dc_dw_00 would preserve nonzero coupling derivative if not zeroed"
    );

    // Zero boundaries clears inner 1..=ibotom and surface nzn, and handles nzn == 0 safely.
    model.aux.rsp.eq_rsp[0] = 1.0;
    model.aux.rsp.chi_rsp[0] = 2.0;
    model.aux.rsp.coupl_rsp[0] = 3.0;
    model.aux.rsp.ptrb_rsp[0] = 4.0;
    zero_boundaries(&mut model, &dc_active, &mut wk);
    assert_eq!(model.aux.rsp.eq_rsp[0], 0.0);
    assert_eq!(model.aux.rsp.chi_rsp[0], 0.0);
    assert_eq!(model.aux.rsp.coupl_rsp[0], 0.0);
    assert_eq!(model.aux.rsp.ptrb_rsp[0], 0.0);
    for i in 1..=dc_active.ibotom {
        assert_eq!(model.aux.rsp.eq_rsp[nz - i], 0.0);
        assert_eq!(model.aux.rsp.chi_rsp[nz - i], 0.0);
        assert_eq!(model.aux.rsp.coupl_rsp[nz - i], 0.0);
        assert_eq!(model.aux.rsp.ptrb_rsp[nz - i], 0.0);
    }

    let mut empty = build_synthetic_rsp_model(0, false);
    let mut wk_empty = RspHydWork::from_model(&empty);
    let dc_empty = RspDerivedConsts::from_config(&empty.controls.rsp, 0);
    zero_boundaries(&mut empty, &dc_empty, &mut wk_empty);
}

#[test]
fn test_rsp_zone_micro_and_turb_partials_fd_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cfg = RspConfig {
            alfa: 1.5,
            alfap: 0.3,
            alfam: 0.25,
            alfat: 0.05,
            gammar: 0.6,
            ..RspConfig::default()
        };

        let nzn = 34;
        let dc = RspDerivedConsts::from_config(&cfg, nzn);
        let efl02 = cfg.efl0 * cfg.efl0;

        let build_grid_micro = || -> (RspEnvGrid, RspZoneMicro) {
            let mut grid = RspEnvGrid::new(nzn);
            let mut mc = RspZoneMicro::new(nzn);
            grid.r[0] = 1.1 * R_SUN;
            grid.m[0] = 1.9 * M_SUN;
            let dm = 0.05 * M_SUN / (nzn as f64);
            for i in 1..=nzn {
                let frac = (i as f64) / (nzn as f64);
                grid.m[i] = grid.m[0] + frac * 0.05 * M_SUN;
                grid.dm[i] = dm;
                grid.dm_bar[i] = dm;
                grid.r[i] = grid.r[0] + frac * 0.8 * R_SUN;
                let t1 = dc.p43 / dm;
                grid.vol[i] = t1 * (pow3(grid.r[i]) - pow3(grid.r[i - 1]));
                mc.dvr[i] = 3.0 * t1 * pow2(grid.r[i]);
                mc.dvrm[i] = -3.0 * t1 * pow2(grid.r[i - 1]);
                // Keep outer temperatures cooler than inner so dlog(T)/dP > 0 (convective PII > 0).
                grid.t[i] = 3.0e5 * (1.0 - 0.85 * frac);
                grid.et[i] = if i > dc.ibotom && i < nzn {
                    1.0e8 * (1.0 + 0.3 * frac)
                } else {
                    0.0
                };
                mc.p[i] = 1.0e9 * (1.0 - 0.85 * frac);
                mc.dpv[i] = -0.25 * mc.p[i] / grid.vol[i];
                mc.dp_dt_00[i] = 0.8 * mc.p[i] / grid.t[i];
                mc.cps[i] = 2.0e8 * (1.0 + 0.1 * frac);
                mc.cpv[i] = 0.05 * mc.cps[i] / grid.vol[i];
                mc.dcp_dt_00[i] = 0.12 * mc.cps[i] / grid.t[i];
                mc.qqs[i] = (1.1 / grid.t[i]) * (1.0 + 0.05 * frac);
                mc.qqv[i] = 0.04 * mc.qqs[i] / grid.vol[i];
                mc.dqq_dt_00[i] = -0.9 * mc.qqs[i] / grid.t[i];
                mc.k[i] = 1.5 * (1.0 + 2.0 * frac);
                mc.dk_dv_00[i] = -0.3 * mc.k[i] / grid.vol[i];
                mc.dk_dt_00[i] = -1.2 * mc.k[i] / grid.t[i];
                mc.update_radial_chain(i);
            }
            (grid, mc)
        };

        let (mut grid0, mc0) = build_grid_micro();
        let hp0 = eval_hp_face_partials(&dc, &mut grid0, &mc0);
        let mut tp0 = RspTurbPartials::new(nzn);
        tp0.evaluate(
            &dc,
            cfg.alfa,
            cfg.alfap,
            cfg.alfat,
            efl02,
            &mut grid0,
            &mc0,
            LcFilterMode::Lina { efl02 },
        );

        // Verify `DLTZP(NZN-1) = 0` (`rsp_lina.f90:673`) and `DLCZP(NZN-1) = 0`.
        assert_eq!(tp0.dltzp[nzn - 1], 0.0);
        assert_eq!(tp0.dlczp[nzn - 1], 0.0);

        // Check `intrinsic_pow` and `intrinsic_log` parity on interior face/zone.
        let i_chk = dc.ibotom + 3;
        let expected_lt = -(2.0 / 3.0)
            * cfg.alfa
            * cfg.alfat
            * pow2(4.0 * PI * pow2(grid0.r[i_chk]))
            * grid0.hp_face[i_chk]
            * 0.5
            * (1.0 / pow2(grid0.vol[i_chk]) + 1.0 / pow2(grid0.vol[i_chk + 1]))
            * (intrinsic_pow(grid0.et[i_chk + 1], 1.5)
                - intrinsic_pow(grid0.et[i_chk], 1.5))
            / grid0.dm_bar[i_chk];
        assert!(rel_err(tp0.lt[i_chk], expected_lt) < 1.0e-14);
        let _check_log =
            intrinsic_log(grid0.t[i_chk + 1]) - intrinsic_log(grid0.t[i_chk]);

        // Finite-difference check wrt R_{i_chk} (using affine microphysics in V).
        let eps_r = 1.0e-5 * grid0.r[i_chk];
        let eval_at_r =
            |dr: f64| -> (RspEnvGrid, HpFacePartials, RspTurbPartials) {
                let (mut g, mut m) = build_grid_micro();
                g.r[i_chk] += dr;
                for j in [i_chk, i_chk + 1] {
                    let t1 = dc.p43 / g.dm[j];
                    let vol_new = t1 * (pow3(g.r[j]) - pow3(g.r[j - 1]));
                    let dv = vol_new - g.vol[j];
                    g.vol[j] = vol_new;
                    m.p[j] += m.dpv[j] * dv;
                    m.cps[j] += m.cpv[j] * dv;
                    m.qqs[j] += m.qqv[j] * dv;
                    m.k[j] += m.dk_dv_00[j] * dv;
                    m.dvr[j] = 3.0 * t1 * pow2(g.r[j]);
                    m.dvrm[j] = -3.0 * t1 * pow2(g.r[j - 1]);
                    m.update_radial_chain(j);
                }
                let hp = eval_hp_face_partials(&dc, &mut g, &m);
                let mut tp = RspTurbPartials::new(nzn);
                tp.evaluate(
                    &dc,
                    cfg.alfa,
                    cfg.alfap,
                    cfg.alfat,
                    efl02,
                    &mut g,
                    &m,
                    LcFilterMode::Lina { efl02 },
                );
                (g, hp, tp)
            };
        let (gp, hpp, tpp) = eval_at_r(eps_r);
        let (gm, hpm, tpm) = eval_at_r(-eps_r);
        let fd_hp = (gp.hp_face[i_chk] - gm.hp_face[i_chk]) / (2.0 * eps_r);
        assert!(rel_err(hp0.dr_00[i_chk], fd_hp) < 1.0e-6);
        let _ = (hpp, hpm);

        let c_p = tpp.source[i_chk] - tpp.damp[i_chk] - tpp.dampr[i_chk];
        let c_m = tpm.source[i_chk] - tpm.damp[i_chk] - tpm.dampr[i_chk];
        let fd_dc_dr0 = (c_p - c_m) / (2.0 * eps_r);
        assert!(
            rel_err(tp0.dc_dr_00[i_chk], fd_dc_dr0) < 1.0e-5,
            "dc_dr_00 ana={}, fd={fd_dc_dr0}",
            tp0.dc_dr_00[i_chk]
        );
        let fd_dlc_dr0 = (tpp.lc[i_chk] - tpm.lc[i_chk]) / (2.0 * eps_r);
        assert!(
            rel_err(tp0.dlcx0[i_chk], fd_dlc_dr0) < 1.0e-5,
            "dlcx0 ana={}, fd={fd_dlc_dr0}",
            tp0.dlcx0[i_chk]
        );
        let fd_dlt_dr0 = (tpp.lt[i_chk] - tpm.lt[i_chk]) / (2.0 * eps_r);
        assert!(
            rel_err(tp0.dltx0[i_chk], fd_dlt_dr0) < 1.0e-5,
            "dltx0 ana={}, fd={fd_dlt_dr0}",
            tp0.dltx0[i_chk]
        );

        // Finite-difference check wrt T_{i_chk} (checks 1/cp*dcp_dt + 1/k*dk_dt in GAMMAR!).
        let eps_t = 1.0e-5 * grid0.t[i_chk];
        let eval_at_t = |dt: f64| -> RspTurbPartials {
            let (mut g, mut m) = build_grid_micro();
            g.t[i_chk] += dt;
            m.p[i_chk] += m.dp_dt_00[i_chk] * dt;
            m.cps[i_chk] += m.dcp_dt_00[i_chk] * dt;
            m.qqs[i_chk] += m.dqq_dt_00[i_chk] * dt;
            m.k[i_chk] += m.dk_dt_00[i_chk] * dt;
            m.update_radial_chain(i_chk);
            let mut tp = RspTurbPartials::new(nzn);
            tp.evaluate(
                &dc,
                cfg.alfa,
                cfg.alfap,
                cfg.alfat,
                efl02,
                &mut g,
                &m,
                LcFilterMode::Lina { efl02 },
            );
            tp
        };
        let tpt = eval_at_t(eps_t);
        let tmt = eval_at_t(-eps_t);
        let fd_dc_dt0 =
            ((tpt.source[i_chk] - tpt.damp[i_chk] - tpt.dampr[i_chk])
                - (tmt.source[i_chk] - tmt.damp[i_chk] - tmt.dampr[i_chk]))
                / (2.0 * eps_t);
        assert!(
            rel_err(tp0.dc_dt_00[i_chk], fd_dc_dt0) < 1.0e-5,
            "dc_dt_00 ana={}, fd={fd_dc_dt0}",
            tp0.dc_dt_00[i_chk]
        );

        // Finite-difference check wrt e_{t,i_chk}.
        let eps_et = 1.0e-5 * grid0.et[i_chk];
        let eval_at_et = |det: f64| -> RspTurbPartials {
            let (mut g, m) = build_grid_micro();
            g.et[i_chk] += det;
            let mut tp = RspTurbPartials::new(nzn);
            tp.evaluate(
                &dc,
                cfg.alfa,
                cfg.alfap,
                cfg.alfat,
                efl02,
                &mut g,
                &m,
                LcFilterMode::Lina { efl02 },
            );
            tp
        };
        let tpw = eval_at_et(eps_et);
        let tmw = eval_at_et(-eps_et);
        let fd_dc_dw0 =
            ((tpw.source[i_chk] - tpw.damp[i_chk] - tpw.dampr[i_chk])
                - (tmw.source[i_chk] - tmw.damp[i_chk] - tmw.dampr[i_chk]))
                / (2.0 * eps_et);
        assert!(rel_err(tp0.dc_dw_00[i_chk], fd_dc_dw0) < 1.0e-7);

        // `LcFilterMode::Relax` negative-PII filtering (`negflu = 0` vs `negflu = 1, II > 300`).
        let (mut g_neg, m_neg) = build_grid_micro();
        // Flip temperature profile sign so `PII < 0`.
        for (i, t_cell) in g_neg.t.iter_mut().enumerate().take(nzn + 1).skip(1)
        {
            *t_cell = 1.0e4 * (1.0 + 0.5 * (i as f64) / (nzn as f64));
        }
        let mut tp_rel0 = RspTurbPartials::new(nzn);
        tp_rel0.evaluate(
            &dc,
            cfg.alfa,
            cfg.alfap,
            cfg.alfat,
            efl02,
            &mut g_neg,
            &m_neg,
            LcFilterMode::Relax {
                negflu: 0,
                iter_ii: 10,
            },
        );
        assert_eq!(tp_rel0.lc[i_chk], 0.0);

        tp_rel0.evaluate(
            &dc,
            cfg.alfa,
            cfg.alfap,
            cfg.alfat,
            efl02,
            &mut g_neg,
            &m_neg,
            LcFilterMode::Relax {
                negflu: 1,
                iter_ii: 301,
            },
        );
        assert_eq!(tp_rel0.dlcz0[i_chk], 0.0);
        assert_eq!(tp_rel0.dlczp[i_chk], 0.0);

        // Small `nzn = 6` (`ibotom = 0`) & empty `nzn = 0` must remain finite (no 0/0 NaN at i = 1).
        let mut g_small = RspEnvGrid::new(6);
        let mut m_small = RspZoneMicro::new(6);
        g_small.r[0] = R_SUN;
        g_small.m[0] = M_SUN;
        for i in 1..=6 {
            g_small.m[i] = g_small.m[0] + (i as f64) * 1.0e30;
            g_small.dm[i] = 1.0e30;
            g_small.dm_bar[i] = 1.0e30;
            g_small.r[i] = g_small.r[0] + (i as f64) * 1.0e9;
            g_small.vol[i] = 1.0e4;
            g_small.t[i] = 1.0e5 / (1.0 + 0.1 * (i as f64));
            g_small.et[i] = 1.0e8;
            m_small.p[i] = 1.0e9 / (1.0 + 0.1 * (i as f64));
            m_small.cps[i] = 1.0e8;
            m_small.qqs[i] = 1.0e-5;
            m_small.k[i] = 1.0;
        }
        let dc_small = RspDerivedConsts::from_config(&cfg, 6);
        let mut tp_small = RspTurbPartials::new(6);
        tp_small.evaluate(
            &dc_small,
            cfg.alfa,
            cfg.alfap,
            cfg.alfat,
            efl02,
            &mut g_small,
            &m_small,
            LcFilterMode::Lina { efl02 },
        );
        assert!(tp_small.source.iter().all(|v| v.is_finite()));
        assert!(tp_small.damp.iter().all(|v| v.is_finite()));
        assert!(tp_small.dampr.iter().all(|v| v.is_finite()));

        let mut g_zero = RspEnvGrid::new(0);
        let m_zero = RspZoneMicro::new(0);
        let mut tp_zero = RspTurbPartials::new(0);
        tp_zero.evaluate(
            &dc_small,
            cfg.alfa,
            cfg.alfap,
            cfg.alfat,
            efl02,
            &mut g_zero,
            &m_zero,
            LcFilterMode::Lina { efl02 },
        );
    });
}

#[test]
fn test_rsp_driver_energy_integrals_and_bundle_overrides_zero_mesa_dir() {
    let mut model = build_synthetic_rsp_model(12, false);
    model.dt = 10.0;
    let (etot_rad, rad_loss) = rsp_total_energy_integrals(&mut model);
    assert!(etot_rad.is_finite());
    assert!(rad_loss.is_finite());

    model.controls.rsp.hydro_only = true;
    let (etot_hydro, rad_zero) = rsp_total_energy_integrals(&mut model);
    assert_eq!(rad_zero, 0.0, "hydro_only disables total_radiation");
    assert_eq!(etot_rad.to_bits(), etot_hydro.to_bits());

    // Continuation stage bundle (`keep_rsp`): `set_tau_factor`, `set_initial_tau_factor`,
    // and `limit_initial_dt` (`run_star_support.f90:1949-1986`).
    let mut prev = build_synthetic_rsp_model(12, false);
    prev.dt = 50.0;
    prev.dt_next = 50.0;
    prev.aux.rsp.rsp_dt = 50.0;
    prev.controls.tau_factor = 1.0;

    let job = StarJobControls {
        set_tau_factor: true,
        set_to_this_tau_factor: 0.25,
        limit_initial_dt: true,
        seconds_for_initial_dt: 12.5,
        ..StarJobControls::default()
    };
    let mut ctrl = prev.controls.clone();
    ctrl.rsp_flag = true;

    let bundle = InlistBundle {
        case_dir: PathBuf::from("."),
        entry_inlist: String::from("inlist"),
        job,
        controls: ctrl,
        eos_config: EosConfig::default(),
        kap_config: KapConfig::default(),
        validation_report: ControlValidationReport::default(),
        sections: HashMap::new(),
        allow_unsupported_controls: false,
    };
    let updated = init_or_update_model_from_bundle(
        &bundle,
        Some(prev),
        &mut DefaultStarExtras,
    )
    .expect("keep_rsp update");
    assert_eq!(updated.controls.tau_factor, 0.25);
    assert_eq!(updated.aux.rsp.rsp_tau_factor, 0.25);
    assert_eq!(updated.dt_next, 12.5);
    assert_eq!(updated.dt, 12.5);
    assert_eq!(updated.aux.rsp.rsp_dt, 12.5);

    // `limit_initial_dt` must never enlarge a smaller existing `dt_next`.
    let mut bundle_bigger = bundle.clone();
    bundle_bigger.job.set_tau_factor = false;
    bundle_bigger.job.set_initial_tau_factor = true;
    bundle_bigger.job.set_to_this_tau_factor = 0.5;
    bundle_bigger.job.seconds_for_initial_dt = 200.0;
    let updated2 = init_or_update_model_from_bundle(
        &bundle_bigger,
        Some(updated),
        &mut DefaultStarExtras,
    )
    .expect("keep_rsp limit_initial_dt no enlarge");
    assert_eq!(updated2.controls.tau_factor, 0.5);
    assert_eq!(updated2.aux.rsp.rsp_tau_factor, 0.5);
    assert_eq!(updated2.dt_next, 12.5);

    // Zero period (`target_dt == 0`) fails loudly in `do1_step`.
    let mut zero_dt = build_synthetic_rsp_model(12, false);
    zero_dt.aux.rsp.have_set_velocities = true;
    zero_dt.aux.rsp.period = 0.0;
    zero_dt.dt = 10.0;
    let err =
        rsp_one_step_with_extras(&mut zero_dt, None, &mut DefaultStarExtras)
            .expect_err("zero period must error on non-positive dt");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert!(zero_dt.controls.output.do_history_file);
}

struct MockOtherLinearAnalysis {
    calls: usize,
    saw_expected_chem: bool,
    saw_finite_brunt: bool,
}

impl StarExtras for MockOtherLinearAnalysis {
    fn other_rsp_linear_analysis(
        &mut self,
        model: &mut StarModel,
        chem: &RspChemCache,
        _log_dir: Option<&std::path::Path>,
    ) -> Result<bool, StarError> {
        self.calls += 1;
        let nz = model.nz;
        self.saw_expected_chem = (0..nz).all(|k| {
            model.abar[k] == chem.abar
                && model.zbar[k] == chem.zbar
                && model.xh_mass[k] == chem.x
                && model.z_mass[k] == chem.z
                && model.cell_xa(k) == chem.xa.as_slice()
        });
        self.saw_finite_brunt = model.brunt_n2.iter().all(|v| v.is_finite());
        model.aux.rsp.period = 100_000.0;
        model.v.fill(0.0);
        // Deliberately do NOT set `have_set_velocities`; `rsp_one_step_with_extras`
        // must set it (`star/private/rsp.f90:466`).
        Ok(true)
    }
}

#[test]
fn test_rsp_driver_other_linear_analysis_and_timestep_controls() {
    let ctrl = StarControls {
        rsp_mass: 4.165,
        rsp_teff: 6050.0,
        rsp_l: 1438.8,
        rsp_x: 0.73,
        rsp_z: 0.007,
        zbase: 0.007,
        rsp: RspConfig {
            use_other_rsp_linear_analysis: true,
            do_check_omega: false,
            ..RspConfig::default()
        },
        ..StarControls::default()
    };

    let kap = KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase: ctrl.rsp_z,
        ..KapConfig::default()
    };
    let job = StarJobControls {
        create_rsp_model: true,
        initial_zfracs: 6,
        set_initial_age: true,
        initial_age: 0.0,
        set_initial_model_number: true,
        initial_model_number: 0,
        set_initial_cumulative_energy_error: true,
        new_cumulative_energy_error: 0.0,
        ..StarJobControls::default()
    };

    let (mut model, _) =
        do_rsp_build_model(ctrl, job, EosConfig::default(), kap, None)
            .expect("build rsp model for linear hook parity");
    assert!(!model.aux.rsp.have_set_velocities);

    let mut extras = MockOtherLinearAnalysis {
        calls: 0,
        saw_expected_chem: false,
        saw_finite_brunt: false,
    };

    rsp_one_step_with_extras(&mut model, None, &mut extras)
        .expect("step 1 invokes other_rsp_linear_analysis");
    assert_eq!(extras.calls, 1);
    assert!(extras.saw_expected_chem);
    assert!(extras.saw_finite_brunt);
    assert!(
        model.aux.rsp.have_set_velocities,
        "rsp_driver must set have_set_velocities = true after hook"
    );
    assert!(model.controls.output.do_history_file);
    assert_eq!(model.aux.step_hist.cumulative_energy_error_old, 0.0);

    // Step 2 must not re-invoke linear analysis.
    rsp_one_step_with_extras(&mut model, None, &mut extras)
        .expect("step 2 skips other_rsp_linear_analysis");
    assert_eq!(extras.calls, 1);

    // `ts_stop.max_timestep` fallback when `rsp.max_timestep == 0.0`.
    model.controls.ts_stop.max_timestep = 10.0;
    rsp_one_step_with_extras(&mut model, None, &mut extras)
        .expect("step 3 ts_stop.max_timestep");
    assert!((model.dt - 10.0).abs() < 1.0e-12);

    // `ts_stop.force_timestep` fallback when `rsp.force_timestep == 0.0`.
    model.controls.ts_stop.force_timestep = 5.0;
    rsp_one_step_with_extras(&mut model, None, &mut extras)
        .expect("step 4 ts_stop.force_timestep");
    assert!((model.dt - 5.0).abs() < 1.0e-12);

    // `rsp_setup_model` enforces `max_years_for_timestep` and `ts_stop.max_timestep`.
    model.controls.ts_stop.force_timestep = 0.0;
    model.controls.ts_stop.max_timestep = 0.0;
    model.controls.rsp.initial_dt_factor = 1.0;
    model.dt_next = 100.0;
    model.controls.max_years_for_timestep = 1.0e-8;
    rsp_setup_model(&mut model, false, false)
        .expect("setup caps dt_next via max_years_for_timestep");
    let expected_cap_yr = 1.0e-8 * SECYER;
    assert!((model.dt_next - expected_cap_yr).abs() < 1.0e-12);
    assert!((model.dt - expected_cap_yr).abs() < 1.0e-12);

    model.dt_next = 100.0;
    model.controls.max_years_for_timestep = 0.0;
    model.controls.ts_stop.max_timestep = 0.1;
    rsp_setup_model(&mut model, false, false)
        .expect("setup caps dt_next via ts_stop.max_timestep");
    assert!((model.dt_next - 0.1).abs() < 1.0e-12);
    assert!((model.dt - 0.1).abs() < 1.0e-12);
}
