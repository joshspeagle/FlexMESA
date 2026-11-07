// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical Jacobian finite-difference and Rosenbrock variant tests for `flexmesa-net`.

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
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unreadable_literal
)]

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_eos::{EosConfig, HelmTable};
use flexmesa_math::exp10;
use flexmesa_net::{
    burn_1_zone_const_p, net_get, net_get_symbolic_d_dxdt_dx, BurnConstPInput,
    NetEvalResult, NetGeneralInfo,
};
use flexmesa_num::RosenbrockVariant;
use flexmesa_rates::ScreeningMode;

fn set_test_composition(net_file: &str, g: &NetGeneralInfo) -> Vec<f64> {
    let cat = NuclideCatalog::builtin_ref();
    let species = g.num_isos;
    let mut xin = vec![0.0; species];
    let idx = |name: &str| -> usize {
        let cid = cat.get_nuclide_index(name).unwrap();
        g.net_iso[cid] - 1
    };

    match net_file {
        "basic.net" => {
            xin[idx("h1")] = f64::from(0.655_186_f32);
            xin[idx("he4")] = 0.310_021_64;
            xin[idx("c12")] = 0.002_725e-1;
            xin[idx("n14")] =
                0.203_101e-1 + 0.612_124e-6 + 0.109_305e-2 + 0.356_004e-4;
            xin[idx("o16")] = 0.094_000e-1;
            xin[idx("ne20")] = 0.162_163e-2;
            xin[idx("mg24")] = 0.658_226e-25;
            xin[idx("he3")] = 0.201_852e-2;
        }
        "approx21.net" => {
            xin[idx("fe56")] = 8.038_702_148_431_816_6e-1;
            xin[idx("fe54")] = 1.609_664_873_676_083_2e-1;
            xin[idx("cr56")] = 2.948_094_553_592_052_5e-2;
            xin[idx("he4")] = 4.862_463_716_132_056_5e-3;
            xin[idx("ni56")] = 2.837_627_073_189_036_0e-4;
            xin[idx("si28")] = 1.501_862_890_613_573_9e-4;
            xin[idx("s32")] = 1.161_327_163_557_345_7e-4;
            xin[idx("prot")] = 1.113_943_163_367_365_3e-4;
            xin[idx("ca40")] = 5.368_837_747_349_418_5e-5;
            xin[idx("ar36")] = 5.270_283_182_256_706_2e-5;
            xin[idx("fe52")] = 3.786_650_413_193_518_5e-5;
            xin[idx("cr48")] = 5.840_112_303_766_797_4e-6;
            xin[idx("neut")] = 4.914_122_770_311_839_7e-6;
            xin[idx("ti44")] = 1.508_503_856_115_474_6e-6;
            xin[idx("o16")] = 9.538_401_960_904_925_5e-7;
            xin[idx("mg24")] = 6.380_820_771_772_558_0e-7;
            xin[idx("c12")] = 2.904_865_667_399_186_8e-7;
            xin[idx("ne20")] = 9.646_886_502_360_942_7e-9;
            xin[idx("he3")] = 4.620_360_386_226_309_6e-80;
            xin[idx("n14")] = 7.586_747_222_584_123_5e-99;
        }
        _ => panic!("unknown net {net_file}"),
    }
    let s: f64 = xin.iter().sum();
    for x in &mut xin {
        *x /= s;
    }
    xin
}

fn assert_rel_close(label: &str, actual: f64, expected: f64, rtol: f64) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(1.0e-30);
    assert!(
        diff / scale <= rtol,
        "{label}: actual={actual:.12e}, expected={expected:.12e}, rel_err={:.3e} > {rtol:.3e}",
        diff / scale
    );
}

fn eval_at(
    g: &NetGeneralInfo,
    skip_jac: bool,
    x: &[f64],
    log_t: f64,
    log_rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    ye: f64,
    eta: f64,
    rf: &[f64],
) -> NetEvalResult {
    net_get(
        g,
        skip_jac,
        x,
        exp10(log_t),
        log_t,
        exp10(log_rho),
        log_rho,
        abar,
        zbar,
        z2bar,
        ye,
        eta,
        0.0,
        0.0,
        rf,
        1.0,
        ScreeningMode::Extended,
    )
    .unwrap()
}

#[test]
fn test_net_analytical_jacobians_match_central_finite_differences() {
    let g = NetGeneralInfo::from_name("basic.net").unwrap();
    let xin = set_test_composition("basic.net", &g);
    let cat = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(cat, &g.chem_id, &xin, false).unwrap();
    let log_t = 7.833;
    let log_rho = 2.0;
    let rf = vec![1.0; g.num_reactions];

    let out0 = eval_at(
        &g, false, &xin, log_t, log_rho, comp.abar, comp.zbar, comp.z2bar,
        comp.ye, 0.0, &rf,
    );

    // Central FD wrt T
    let dlnt = 1.0e-6;
    let lt_p = log_t + dlnt / std::f64::consts::LN_10;
    let lt_m = log_t - dlnt / std::f64::consts::LN_10;
    let out_tp = eval_at(
        &g, true, &xin, lt_p, log_rho, comp.abar, comp.zbar, comp.z2bar,
        comp.ye, 0.0, &rf,
    );
    let out_tm = eval_at(
        &g, true, &xin, lt_m, log_rho, comp.abar, comp.zbar, comp.z2bar,
        comp.ye, 0.0, &rf,
    );
    let fd_deps_dt =
        (out_tp.eps_nuc - out_tm.eps_nuc) / (exp10(lt_p) - exp10(lt_m));
    assert_rel_close(
        "d_eps_nuc_dT vs FD",
        out0.d_eps_nuc_dt,
        fd_deps_dt,
        1.0e-5,
    );

    // Central FD wrt mass fractions X_j at fixed (T, rho, abar, zbar, z2bar, ye)
    let s = g.num_isos;
    for j in 0..s {
        let h = 1.0e-7 * xin[j].max(1.0e-4);
        let mut xp = xin.clone();
        let mut xm = xin.clone();
        xp[j] += h;
        xm[j] -= h;
        let out_xp = eval_at(
            &g, true, &xp, log_t, log_rho, comp.abar, comp.zbar, comp.z2bar,
            comp.ye, 0.0, &rf,
        );
        let out_xm = eval_at(
            &g, true, &xm, log_t, log_rho, comp.abar, comp.zbar, comp.z2bar,
            comp.ye, 0.0, &rf,
        );
        let fd_deps_dxj = (out_xp.eps_nuc - out_xm.eps_nuc) / (2.0 * h);
        if out0.d_eps_nuc_dx[j].abs() > 1.0e-6 {
            assert_rel_close(
                "basic.net d_eps_nuc_dx vs FD",
                out0.d_eps_nuc_dx[j],
                fd_deps_dxj,
                1.0e-4,
            );
        }
        for i in 0..s {
            let fd_dxdt_ij = (out_xp.dxdt[i] - out_xm.dxdt[i]) / (2.0 * h);
            let ana_ij = out0.d_dxdt_dx[i * s + j];
            if ana_ij.abs() > 1.0e-12 {
                assert_rel_close(
                    "basic.net d_dxdt_dx vs FD",
                    ana_ij,
                    fd_dxdt_ij,
                    1.0e-5,
                );
            }
        }
    }

    // Symbolic Jacobian sparsity check
    let sym = net_get_symbolic_d_dxdt_dx(
        &g,
        &xin,
        exp10(log_t),
        log_t,
        exp10(log_rho),
        log_rho,
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        0.0,
        0.0,
        0.0,
        &rf,
        1.0,
        ScreeningMode::Extended,
    )
    .unwrap();
    assert_eq!(sym.len(), s * s);
    for (idx, &val) in out0.d_dxdt_dx.iter().enumerate() {
        if val.abs() > 0.0 {
            assert!(sym[idx] != 0.0, "missing symbolic entry at index {idx}");
        }
    }

    // Check approx21.net analytical derivatives vs central finite differences
    let g21 = NetGeneralInfo::from_name("approx21.net").unwrap();
    let xin21 = set_test_composition("approx21.net", &g21);
    let comp21 =
        get_composition_info(cat, &g21.chem_id, &xin21, false).unwrap();
    let log_t21 = 9.48;
    let log_rho21 = 6.23;
    let rf21 = vec![1.0; g21.num_reactions];
    let out21 = eval_at(
        &g21,
        false,
        &xin21,
        log_t21,
        log_rho21,
        comp21.abar,
        comp21.zbar,
        comp21.z2bar,
        comp21.ye,
        2.5,
        &rf21,
    );

    let dt_step = 1.0e-6;
    let lt_p = log_t21 + dt_step / std::f64::consts::LN_10;
    let lt_m = log_t21 - dt_step / std::f64::consts::LN_10;
    let out21_tp = eval_at(
        &g21,
        true,
        &xin21,
        lt_p,
        log_rho21,
        comp21.abar,
        comp21.zbar,
        comp21.z2bar,
        comp21.ye,
        2.5,
        &rf21,
    );
    let out21_tm = eval_at(
        &g21,
        true,
        &xin21,
        lt_m,
        log_rho21,
        comp21.abar,
        comp21.zbar,
        comp21.z2bar,
        comp21.ye,
        2.5,
        &rf21,
    );
    let dt21 = exp10(lt_p) - exp10(lt_m);
    assert_rel_close(
        "approx21 d_eps_nuc_dT vs FD",
        out21.d_eps_nuc_dt,
        (out21_tp.eps_nuc - out21_tm.eps_nuc) / dt21,
        1.0e-5,
    );
    for i in 0..g21.num_isos {
        let fd_dxdt_dt = (out21_tp.dxdt[i] - out21_tm.dxdt[i]) / dt21;
        if fd_dxdt_dt.abs() > 1.0e-6 {
            assert_rel_close(
                "approx21 d_dxdt_dT vs FD",
                out21.d_dxdt_dt[i],
                fd_dxdt_dt,
                1.0e-4,
            );
        }
    }

    let lr_p = log_rho21 + dt_step / std::f64::consts::LN_10;
    let lr_m = log_rho21 - dt_step / std::f64::consts::LN_10;
    let out21_rp = eval_at(
        &g21,
        true,
        &xin21,
        log_t21,
        lr_p,
        comp21.abar,
        comp21.zbar,
        comp21.z2bar,
        comp21.ye,
        2.5,
        &rf21,
    );
    let out21_rm = eval_at(
        &g21,
        true,
        &xin21,
        log_t21,
        lr_m,
        comp21.abar,
        comp21.zbar,
        comp21.z2bar,
        comp21.ye,
        2.5,
        &rf21,
    );
    let dr21 = exp10(lr_p) - exp10(lr_m);
    assert_rel_close(
        "approx21 d_eps_nuc_dRho vs FD",
        out21.d_eps_nuc_drho,
        (out21_rp.eps_nuc - out21_rm.eps_nuc) / dr21,
        1.0e-5,
    );
    for i in 0..g21.num_isos {
        let fd_dxdt_drho = (out21_rp.dxdt[i] - out21_rm.dxdt[i]) / dr21;
        if fd_dxdt_drho.abs() > 1.0e-6 {
            assert_rel_close(
                "approx21 d_dxdt_dRho vs FD",
                out21.d_dxdt_drho[i],
                fd_dxdt_drho,
                1.0e-4,
            );
        }
    }

    // Central FD wrt mass fractions X_j for approx21 d_dxdt_dx
    let s21 = g21.num_isos;
    for j in 0..s21 {
        if xin21[j] < 1.0e-12 {
            continue;
        }
        let h = 1.0e-7 * xin21[j];
        let mut xp = xin21.clone();
        let mut xm = xin21.clone();
        xp[j] += h;
        xm[j] -= h;
        let out_xp = eval_at(
            &g21,
            true,
            &xp,
            log_t21,
            log_rho21,
            comp21.abar,
            comp21.zbar,
            comp21.z2bar,
            comp21.ye,
            2.5,
            &rf21,
        );
        let out_xm = eval_at(
            &g21,
            true,
            &xm,
            log_t21,
            log_rho21,
            comp21.abar,
            comp21.zbar,
            comp21.z2bar,
            comp21.ye,
            2.5,
            &rf21,
        );
        for i in 0..s21 {
            let fd_ij = (out_xp.dxdt[i] - out_xm.dxdt[i]) / (2.0 * h);
            let ana_ij = out21.d_dxdt_dx[i * s21 + j];
            if (ana_ij * xin21[j]).abs() > 1.0e-4 * out21.dxdt[i].abs().max(1.0)
            {
                assert_rel_close(
                    "approx21 d_dxdt_dx vs FD",
                    ana_ij,
                    fd_ij,
                    1.0e-4,
                );
            }
        }
    }
}

#[test]
fn test_burn_const_p_rosenbrock_variants_ros3p_and_ros3pl_converge() {
    let g = NetGeneralInfo::from_name("approx21.net").unwrap();
    let cat = NuclideCatalog::builtin_ref();
    let mut xin = vec![0.0; g.num_isos];
    let ic12 = g.net_iso[cat.get_nuclide_index("c12").unwrap()] - 1;
    let io16 = g.net_iso[cat.get_nuclide_index("o16").unwrap()] - 1;
    xin[ic12] = 0.37;
    xin[io16] = 0.63;

    let rate_factors = vec![1.0; g.num_reactions];
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();

    for solver in [
        RosenbrockVariant::Rodas4,
        RosenbrockVariant::Rodasp,
        RosenbrockVariant::Ros3p,
        RosenbrockVariant::Ros3pl,
    ] {
        let input = BurnConstPInput {
            solver,
            starting_temp: 3.0e9,
            starting_x: &xin,
            clip: false,
            t_end: 1.0e-4,
            pressure: exp10(24.0),
            rate_factors: &rate_factors,
            weak_rate_factor: 1.0,
            reaction_qs: &[],
            reaction_neu_qs: &[],
            screening_mode: ScreeningMode::Extended,
            h0: 1.0e-6,
            max_step_size: 1.0,
            max_steps: 1000,
            rtol: 1.0e-7,
            atol: 1.0e-8,
            x_min: -1.0e-10,
        };
        let out = burn_1_zone_const_p(&g, ht, &eos_cfg, &input).unwrap();
        let xsum: f64 = out.ending_x.iter().sum();
        assert_rel_close("burn_const_P xsum", xsum, 1.0, 1.0e-10);
        assert_rel_close(
            "burn_const_P o16",
            out.ending_x[io16],
            6.955_981_012_989_815_9e-1,
            1.0e-3,
        );
    }
}
