// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` Fortran oracle parity and physics invariant tests for element
//! diffusion and phase separation (`#156`, `Refs #271, #272, #273`).

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

use std::collections::HashMap;
use std::path::PathBuf;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, AMU, L_SUN, ME, M_SUN, PI4, STANDARD_CGRAV};
use flexmesa_eos::{SkyeSolidMixingRule, I_PHASE};
use flexmesa_kap::KapConfig;
use flexmesa_star::diffusion::{
    do_solve_diffusion, eval_caplan_ocp_dstar, SolveDiffusionInput,
    SolveDiffusionOutput,
};
use flexmesa_star::diffusion_procs::setup_struct_info;
use flexmesa_star::diffusion_support::{
    get_matrix_coeffs, MatrixCoeffsInput, MatrixCoeffsOutput,
};
use flexmesa_star::element_diffusion::{
    do_element_diffusion_step, finish_element_diffusion,
};
use flexmesa_star::hydro_energy::setup_sources_and_others;
use flexmesa_star::hydro_vars::{set_hydro_vars, set_mlt_vars};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::model_io::{
    load_photo_bytes, read_mod_str, write_mod_string, write_photo_bytes,
    PHOTO_FORMAT_VERSION,
};
use flexmesa_star::paquette_coeffs::paquette_coefficients;
use flexmesa_star::phase_separation::{
    blouin_delta_xne, blouin_delta_xo, do_phase_separation_step,
};
use flexmesa_star::primary_state::PrimaryStateBuffer;
use flexmesa_star::{
    load_zams_model, StarControls, StarError, StarJobControls, StarModel,
};

type OracleMap = HashMap<(String, String, String, usize, usize), f64>;

fn load_oracles() -> OracleMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/diffusion_oracles/mesa_26_4_1_diffusion_oracles.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", path.display());
    });
    let mut has_mesa_ver = false;
    let mut has_sdk_ver = false;
    let mut has_compiler = false;
    let mut map = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# mesa_version:") {
            has_mesa_ver = true;
            continue;
        }
        if trimmed.starts_with("# mesasdk_version:") {
            has_sdk_ver = true;
            continue;
        }
        if trimmed.starts_with("# compiler:") {
            assert!(
                trimmed.contains("GCC version"),
                "unexpected compiler header: {trimmed}"
            );
            has_compiler = true;
            continue;
        }
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("section\t")
        {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 7, "unexpected oracle line: {trimmed}");
        let sec = cols[0].to_string();
        let cid = cols[1].to_string();
        let fld = cols[2].to_string();
        let ii: usize = cols[3].parse().expect("parse i");
        let jj: usize = cols[4].parse().expect("parse j");
        let hex = cols[5]
            .strip_prefix("0x")
            .unwrap_or_else(|| panic!("missing 0x prefix: {}", cols[5]));
        let bits = u64::from_str_radix(hex, 16)
            .unwrap_or_else(|e| panic!("invalid hex {}: {e}", cols[5]));
        let val = f64::from_bits(bits);
        map.insert((sec, cid, fld, ii, jj), val);
    }
    assert!(has_mesa_ver, "oracle TSV missing # mesa_version header");
    assert!(has_sdk_ver, "oracle TSV missing # mesasdk_version header");
    assert!(has_compiler, "oracle TSV missing # compiler header");
    assert!(
        map.len() >= 2400,
        "expected >= 2400 oracle entries, got {}",
        map.len()
    );
    map
}

fn oracle_val(
    map: &OracleMap,
    sec: &str,
    cid: &str,
    fld: &str,
    i: usize,
    j: usize,
) -> f64 {
    *map.get(&(sec.to_string(), cid.to_string(), fld.to_string(), i, j))
        .unwrap_or_else(|| {
            panic!("missing oracle key ({sec}, {cid}, {fld}, {i}, {j})")
        })
}

fn assert_close(label: &str, got: f64, want: f64, rtol: f64, atol: f64) {
    let diff = (got - want).abs();
    let scale = atol.max(rtol * want.abs().max(got.abs()));
    assert!(
        diff <= scale,
        "{label}: got={got:.17e} (0x{:016X}), want={want:.17e} (0x{:016X}), \
         abs_diff={diff:.3e}, rel_diff={:.3e} > tol={scale:.3e}",
        got.to_bits(),
        want.to_bits(),
        if want == 0.0 { diff } else { diff / want.abs() }
    );
}

#[test]
fn test_paquette_coefficients_match_mesa_26_4_1_oracles() {
    let oracles = load_oracles();
    let nn = 5_usize;
    let can1 = [1.00794_f64, 4.002602, 12.011, 15.9994, ME / AMU];
    let kzn1 = [1.0_f64, 2.0, 6.0, 8.0, -1.0];

    let cases: [(&str, f64, f64, [f64; 4]); 4] = [
        ("solar_core_low_psi", 1.5e2, 1.5e7, [0.40, 0.58, 0.01, 0.01]),
        ("solar_env_neg_psi", 2.0e-1, 2.0e6, [0.70, 0.28, 0.01, 0.01]),
        ("he_shell_mod_psi", 1.0e3, 1.0e7, [0.01, 0.79, 0.15, 0.05]),
        ("co_wd_high_psi", 2.0e5, 1.5e7, [1.0e-4, 0.0999, 0.45, 0.45]),
    ];

    for (cid, rho, temp, xs) in cases {
        let mut na1 = [0.0_f64; 5];
        let mut ne_sum = 0.0_f64;
        for i in 0..4 {
            na1[i] = rho * xs[i] / (can1[i] * AMU);
            ne_sum += kzn1[i] * na1[i];
        }
        na1[4] = ne_sum;

        let mut ddiff = vec![0.0_f64; nn * nn];
        let mut kdiff = vec![0.0_f64; nn * nn];
        let mut zdiff = vec![0.0_f64; nn * nn];
        let mut zdiff1 = vec![0.0_f64; nn * nn];
        let mut zdiff2 = vec![0.0_f64; nn * nn];
        let mut ath = vec![0.0_f64; nn * nn];

        paquette_coefficients(
            rho,
            temp,
            nn,
            &can1,
            &kzn1,
            &na1,
            &mut ddiff,
            &mut kdiff,
            &mut zdiff,
            &mut zdiff1,
            &mut zdiff2,
            &mut ath,
        );

        for i in 0..nn {
            for j in 0..nn {
                let idx = i * nn + j;
                for (fld, arr) in [
                    ("Ddiff", &ddiff),
                    ("Kdiff", &kdiff),
                    ("Zdiff", &zdiff),
                    ("Zdiff1", &zdiff1),
                    ("Zdiff2", &zdiff2),
                    ("Ath", &ath),
                ] {
                    let want = oracle_val(
                        &oracles,
                        "paquette",
                        cid,
                        fld,
                        i + 1,
                        j + 1,
                    );
                    assert_close(
                        &format!("paquette/{cid}/{fld}[{},{}]", i + 1, j + 1),
                        arr[idx],
                        want,
                        1.0e-13,
                        1.0e-30,
                    );
                }
            }
        }
    }
}

#[test]
fn test_get_matrix_coeffs_all_8_regimes_match_mesa_26_4_1_oracles() {
    let oracles = load_oracles();
    let nz = 5_usize;
    let nc = 4_usize;
    let m = 5_usize;

    let dt = 1.0e11_f64;
    let diff_fac = [1.0_f64; 4];
    let a_vec = [1.00794_f64, 4.002602, 12.011, 15.9994, ME / AMU];
    let mut r_face = vec![0.0_f64; nz];
    let mut r_mid = vec![0.0_f64; nz];
    let mut cell_dm = vec![1.0e30_f64; nz];
    let dm_bar = vec![1.0e30_f64; nz];
    let mut xm_face = vec![0.0_f64; nz];
    let mut grav = vec![0.0_f64; nz];
    let mut dlnp_dr = vec![0.0_f64; nz];
    let mut dlnt_dr = vec![0.0_f64; nz];
    let mut dlnrho_dr = vec![0.0_f64; nz];
    let gam_t_lim = vec![1.0_f64; nz];
    let alfa_face = vec![0.5_f64; nz];
    let rad_acc = vec![0.0_f64; nz * m];
    let mut z_mat = vec![0.0_f64; nz * m];
    let mut x_init = vec![0.0_f64; nz * m];

    for k in 0..nz {
        let kf = k as f64;
        r_face[k] = 5.0e10 * (1.0 - 0.08 * kf);
        r_mid[k] = 5.0e10 * (1.0 - 0.08 * (kf + 0.5));
        cell_dm[k] = 1.0e30;
        xm_face[k] = kf * 1.0e30;
        grav[k] = 2.7e4 * (1.0 + 0.1 * kf);
        dlnp_dr[k] = -2.0e-10 * (1.0 + 0.05 * kf);
        dlnt_dr[k] = -6.0e-11 * (1.0 + 0.04 * kf);
        dlnrho_dr[k] = -1.4e-10 * (1.0 + 0.05 * kf);
        z_mat[k * m] = 1.0;
        z_mat[k * m + 1] = 2.0;
        z_mat[k * m + 2] = 6.0;
        z_mat[k * m + 3] = 8.0;
        z_mat[k * m + 4] = -1.0;
        x_init[k * m] = 0.70 - 0.03 * kf;
        x_init[k * m + 1] = 0.26 + 0.025 * kf;
        x_init[k * m + 2] = 0.02 + 0.003 * kf;
        x_init[k * m + 3] = 0.02 + 0.002 * kf;
        x_init[k * m + 4] = 0.0;
    }

    // (cid, pure_coulomb, use_paq, use_cgs, use_caplan, rho, T, eta, gam)
    let regimes: [(&str, bool, bool, bool, bool, f64, f64, f64, f64); 8] = [
        (
            "pure_coulomb_thoul",
            true,
            false,
            false,
            false,
            10.0,
            5.0e6,
            -1.0,
            0.05,
        ),
        (
            "paquette_thoul",
            false,
            true,
            false,
            false,
            10.0,
            5.0e6,
            -1.0,
            0.05,
        ),
        (
            "sm_thoul", false, false, false, false, 10.0, 5.0e6, -1.0, 0.05,
        ),
        (
            "sm_cgs_low_eta",
            false,
            false,
            true,
            false,
            10.0,
            5.0e6,
            2.0,
            0.05,
        ),
        (
            "sm_cgs_blend_eta",
            false,
            false,
            true,
            false,
            10.0,
            5.0e6,
            25.0,
            0.05,
        ),
        (
            "sm_cgs_high_eta",
            false,
            false,
            true,
            false,
            1.0e5,
            1.0e7,
            60.0,
            1.5,
        ),
        (
            "caplan_blend_cgs",
            false,
            false,
            true,
            true,
            5.0e5,
            8.0e6,
            60.0,
            6.0,
        ),
        (
            "caplan_pure_cgs",
            false,
            false,
            true,
            true,
            2.0e6,
            6.0e6,
            80.0,
            50.0,
        ),
    ];

    for (
        cid,
        pure_coulomb,
        use_paq,
        use_cgs,
        use_caplan,
        rho,
        temp,
        eta_v,
        gam_v,
    ) in regimes
    {
        let rho_face = vec![rho; nz];
        let t_face = vec![temp; nz];
        let eta = vec![eta_v; nz];
        let gam = vec![gam_v; nz];
        let fpr2r: Vec<f64> =
            r_face.iter().map(|&r| PI4 * r * r * rho).collect();

        let mut x_work = x_init.clone();
        let mut x_face = vec![0.0_f64; nz * m];
        let mut c_mat = vec![0.0_f64; nz * m];
        let mut c_div_x = vec![0.0_f64; nz * m];
        let mut c_div_x_f = vec![0.0_f64; nz * m];
        let mut e_ap = vec![0.0_f64; nz];
        let mut e_at = vec![0.0_f64; nz];
        let mut e_ar = vec![0.0_f64; nz];
        let mut e_ax = vec![0.0_f64; nz * m];
        let mut e_fld = vec![0.0_f64; nz];
        let mut g_ap = vec![0.0_f64; nz];
        let mut g_at = vec![0.0_f64; nz];
        let mut g_ar = vec![0.0_f64; nz];
        let mut g_ax = vec![0.0_f64; nz * m];
        let mut g_fld = vec![0.0_f64; nz];
        let mut v_adv = vec![0.0_f64; nz * nc];
        let mut v_tot = vec![0.0_f64; nz * nc];
        let mut vlnp = vec![0.0_f64; nz * nc];
        let mut vlnt = vec![0.0_f64; nz * nc];
        let mut vrad = vec![0.0_f64; nz * nc];
        let mut gt_face = vec![0.0_f64; nz * nc];
        let mut d_self = vec![0.0_f64; nz * nc];
        let mut ad_face = vec![0.0_f64; nz];
        let mut sig_face = vec![0.0_f64; nz * nc * nc];
        let mut sigma_lnc = vec![0.0_f64; nz * nc * nc];

        let inp = MatrixCoeffsInput {
            nz,
            nc,
            m,
            nzlo: 0,
            nzhi: nz - 1,
            pure_coulomb,
            diffusion_use_paquette: use_paq,
            diffusion_use_caplan: use_caplan,
            diffusion_use_cgs_solver: use_cgs,
            cgs_thermal_diffusion_eta_full_on: 10.0,
            cgs_thermal_diffusion_eta_full_off: 40.0,
            use_other_diffusion_factor: false,
            star_dt: dt,
            v_advection_max: 1.0e99,
            diffusion_vlimit: 0.0,
            diffusion_vlimit_dm_full_on: -1.0,
            diffusion_vlimit_dm_full_off: -1.0,
            diffusion_ad_dm_full_on: -1.0,
            diffusion_ad_dm_full_off: -1.0,
            diffusion_ad_boost_factor: 1.0,
            diffusion_sig_factor: 1.0,
            diffusion_gt_factor: 1.0,
            min_t_for_radaccel: 1.0e99,
            max_t_for_radaccel: -1.0,
            diffusion_factor: &diff_fac,
            extra_diffusion_factor: None,
            a: &a_vec,
            z: &z_mat,
            rho_face: &rho_face,
            t_face: &t_face,
            four_pi_r2_rho_face: &fpr2r,
            xm_face: &xm_face,
            cell_dm: &cell_dm,
            dm_bar: &dm_bar,
            dln_p_dr_face: &dlnp_dr,
            dln_t_dr_face: &dlnt_dr,
            dln_rho_dr_face: &dlnrho_dr,
            r_mid: &r_mid,
            grav: &grav,
            eta: &eta,
            gam: &gam,
            gamma_t_limit_coeffs_face: &gam_t_lim,
            alfa_face: &alfa_face,
            rad_accel_face: &rad_acc,
        };
        let mut out = MatrixCoeffsOutput {
            x: &mut x_work,
            x_face: &mut x_face,
            c: &mut c_mat,
            c_div_x: &mut c_div_x,
            c_div_x_face: &mut c_div_x_f,
            e_ap: &mut e_ap,
            e_at: &mut e_at,
            e_ar: &mut e_ar,
            e_ax: &mut e_ax,
            e_field_face: &mut e_fld,
            g_ap: &mut g_ap,
            g_at: &mut g_at,
            g_ar: &mut g_ar,
            g_ax: &mut g_ax,
            g_field_face: &mut g_fld,
            v_advection_face: &mut v_adv,
            v_total_face: &mut v_tot,
            vln_p_face: &mut vlnp,
            vln_t_face: &mut vlnt,
            v_rad_face: &mut vrad,
            gt_face: &mut gt_face,
            d_self_face: &mut d_self,
            ad_face: &mut ad_face,
            sig_face: &mut sig_face,
            sigma_lnc: &mut sigma_lnc,
        };
        get_matrix_coeffs(&inp, &mut out)
            .unwrap_or_else(|e| panic!("get_matrix_coeffs({cid}) failed: {e}"));

        // Interior face k = 2 (1-based k = 3 in Fortran oracle)
        let k = 2_usize;
        let k1 = k + 1;
        assert_close(
            &format!("mc/{cid}/AD_face"),
            ad_face[k],
            oracle_val(&oracles, "matrix_coeffs", cid, "AD_face", k1, 0),
            1.0e-12,
            1.0e-30,
        );
        assert_close(
            &format!("mc/{cid}/E_field_face"),
            e_fld[k],
            oracle_val(&oracles, "matrix_coeffs", cid, "E_field_face", k1, 0),
            1.0e-11,
            1.0e-30,
        );
        assert_close(
            &format!("mc/{cid}/g_field_face"),
            g_fld[k],
            oracle_val(&oracles, "matrix_coeffs", cid, "g_field_face", k1, 0),
            1.0e-11,
            1.0e-30,
        );
        for i in 0..nc {
            for (fld, arr) in [
                ("v_advection", &v_adv),
                ("v_total", &v_tot),
                ("vlnP", &vlnp),
                ("vlnT", &vlnt),
                ("GT_face", &gt_face),
                ("D_self_face", &d_self),
            ] {
                let want =
                    oracle_val(&oracles, "matrix_coeffs", cid, fld, i + 1, k1);
                assert_close(
                    &format!("mc/{cid}/{fld}[{}, {k1}]", i + 1),
                    arr[k * nc + i],
                    want,
                    1.0e-11,
                    1.0e-25,
                );
            }
            for j in 0..nc {
                let idx = (k * nc + i) * nc + j;
                assert_close(
                    &format!("mc/{cid}/SIG_face[{},{}]", i + 1, j + 1),
                    sig_face[idx],
                    oracle_val(
                        &oracles,
                        "matrix_coeffs",
                        cid,
                        "SIG_face",
                        i + 1,
                        j + 1,
                    ),
                    1.0e-11,
                    1.0e-25,
                );
                assert_close(
                    &format!("mc/{cid}/sigma_lnC[{},{}]", i + 1, j + 1),
                    sigma_lnc[idx],
                    oracle_val(
                        &oracles,
                        "matrix_coeffs",
                        cid,
                        "sigma_lnC",
                        i + 1,
                        j + 1,
                    ),
                    1.0e-11,
                    1.0e-25,
                );
            }
        }
    }
}

#[test]
fn test_do_solve_diffusion_be_and_isolve_match_mesa_26_4_1_oracles() {
    let oracles = load_oracles();
    let nz = 12_usize;
    let species = 4_usize;
    let nc = 4_usize;
    let cls = [0_usize, 1, 2, 3];
    let class_a = [1.0_f64, 4.0, 12.0, 56.0];
    let class_z = [1_usize, 2, 6, 26];
    let diff_fac = [1.0_f64; 4];

    // (cid, use_isolve, solver_name, use_cgs, use_caplan, total_time, eta_v, gam_v, is_wd)
    let cases: [(&str, bool, &str, bool, bool, f64, f64, f64, bool); 9] = [
        (
            "be_thoul_solar",
            false,
            "ros2_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_ros2_solar",
            true,
            "ros2_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_rose2_solar",
            true,
            "rose2_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_ros3p_solar",
            true,
            "ros3p_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_ros3pl_solar",
            true,
            "ros3pl_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_rodas3_solar",
            true,
            "rodas3_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_rodas4_solar",
            true,
            "rodas4_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "isolve_rodasp_solar",
            true,
            "rodasp_solver",
            false,
            false,
            1.0e14,
            -1.0,
            0.05,
            false,
        ),
        (
            "be_cgs_caplan_wd",
            false,
            "ros2_solver",
            true,
            true,
            1.0e12,
            50.0,
            20.0,
            true,
        ),
    ];

    for (
        cid,
        use_isolve,
        solver_name,
        use_cgs,
        use_caplan,
        total_time,
        eta_v,
        gam_v,
        is_wd,
    ) in cases
    {
        let m_tot = if is_wd { 0.6 * M_SUN } else { M_SUN };
        let r_out = if is_wd { 8.0e8 } else { 6.0e10 };
        let dr = 0.5 * r_out / (nz as f64);

        let mut r_face = vec![0.0_f64; nz];
        let mut r_mid = vec![0.0_f64; nz];
        let mut dm_in = vec![0.0_f64; nz];
        let mut dm_bar = vec![0.0_f64; nz];
        let mut rho_arr = vec![0.0_f64; nz];
        let mut t_arr = vec![0.0_f64; nz];
        let mut lnd = vec![0.0_f64; nz];
        let mut lnt = vec![0.0_f64; nz];
        let mut grav = vec![0.0_f64; nz];
        let mut abar = vec![0.0_f64; nz];
        let mut free_e = vec![0.0_f64; nz];
        let mut dlnp_dm_f = vec![0.0_f64; nz];
        let mut dlnt_dm_f = vec![0.0_f64; nz];
        let mut dlnrho_dm_f = vec![0.0_f64; nz];
        let mut typ_q = vec![0.0_f64; nz * nc];
        let mut xa = vec![0.0_f64; nz * species];
        let eta = vec![eta_v; nz];
        let gam = vec![gam_v; nz];
        let phase = vec![0.0_f64; nz];

        for k in 0..nz {
            let kf = k as f64;
            r_face[k] = r_out - kf * dr;
            r_mid[k] = r_out - (kf + 0.5) * dr;
            dm_in[k] = (0.05 * m_tot) / (nz as f64);
            dm_bar[k] = dm_in[k];
            if is_wd {
                rho_arr[k] = 1.0e5 * (1.0 + 0.15 * kf);
                t_arr[k] = 1.0e7 * (1.0 + 0.03 * kf);
                grav[k] = 1.0e8;
                xa[k * species] = 1.0e-5;
                xa[k * species + 1] = 0.20 - 0.015 * kf;
                xa[k * species + 2] = 0.78 + 0.015 * kf;
                xa[k * species + 3] = 0.02 - 1.0e-5;
                abar[k] = 10.0;
                free_e[k] = 0.5;
            } else {
                rho_arr[k] = 1.0 * (1.0 + 0.35 * kf);
                t_arr[k] = 3.0e6 * (1.0 + 0.12 * kf);
                grav[k] = 2.7e4;
                xa[k * species] = 0.72 - 0.012 * kf;
                xa[k * species + 1] = 0.26 + 0.010 * kf;
                xa[k * species + 2] = 0.015 + 0.0015 * kf;
                xa[k * species + 3] = 0.005 + 0.0005 * kf;
                abar[k] = 1.30;
                free_e[k] = 0.85;
            }
            lnd[k] = flexmesa_math::log(rho_arr[k]);
            lnt[k] = flexmesa_math::log(t_arr[k]);
            let denom = PI4 * r_mid[k] * r_mid[k] * rho_arr[k];
            if is_wd {
                dlnp_dm_f[k] = -1.5e-8 / denom;
                dlnt_dm_f[k] = -1.0e-9 / denom;
                dlnrho_dm_f[k] = -1.2e-8 / denom;
            } else {
                dlnp_dm_f[k] = -2.0e-10 / denom;
                dlnt_dm_f[k] = -6.0e-11 / denom;
                dlnrho_dm_f[k] = -1.4e-10 / denom;
            }
            typ_q[k * nc] = 1.0;
            typ_q[k * nc + 1] = 2.0;
            typ_q[k * nc + 2] = 6.0;
            typ_q[k * nc + 3] = 26.0;
        }

        let mut x_init = vec![0.0_f64; nz * nc];
        let mut x_final = vec![0.0_f64; nz * nc];
        let mut d_self = vec![0.0_f64; nz * nc];
        let mut v_adv = vec![0.0_f64; nz * nc];
        let mut v_tot = vec![0.0_f64; nz * nc];
        let mut vlnp = vec![0.0_f64; nz * nc];
        let mut vlnt = vec![0.0_f64; nz * nc];
        let mut vrad = vec![0.0_f64; nz * nc];
        let mut g_rad = vec![0.0_f64; nz * nc];
        let mut e_fld = vec![0.0_f64; nz];
        let mut g_fld = vec![0.0_f64; nz];

        let inp = SolveDiffusionInput {
            nz,
            species,
            nc,
            class: &cls,
            class_a: &class_a,
            class_z: &class_z,
            has_h1: true,
            has_he4: true,
            abar: &abar,
            free_e: &free_e,
            dm_bar_in: &dm_bar,
            dm_in: &dm_in,
            t: &t_arr,
            ln_t: &lnt,
            rho: &rho_arr,
            lnd: &lnd,
            r_mid: &r_mid,
            r_face_in: &r_face,
            dln_p_dm_face: &dlnp_dm_f,
            dln_t_dm_face: &dlnt_dm_f,
            dln_rho_dm_face: &dlnrho_dm_f,
            grav: &grav,
            eta: &eta,
            gamma: &gam,
            phase: &phase,
            r_center: 0.0,
            star_dt: total_time,
            total_time,
            pure_coulomb: false,
            diffusion_use_paquette: false,
            diffusion_use_caplan: use_caplan,
            diffusion_use_cgs_solver: use_cgs,
            cgs_thermal_diffusion_eta_full_on: 10.0,
            cgs_thermal_diffusion_eta_full_off: 40.0,
            dt_div_timescale: 1.0,
            min_num_substeps: 1,
            max_steps: 500,
            max_iters_total: 1000,
            max_iters_per_substep: 20,
            max_retries_per_substep: 20,
            calculate_ionization: false,
            nsmooth_typical_charge: 0,
            min_t_for_radaccel: 1.0e99,
            max_t_for_radaccel: -1.0,
            v_advection_max: 1.0e99,
            diffusion_vlimit: 0.0,
            diffusion_vlimit_dm_full_on: -1.0,
            diffusion_vlimit_dm_full_off: -1.0,
            diffusion_ad_dm_full_on: -1.0,
            diffusion_ad_dm_full_off: -1.0,
            diffusion_ad_boost_factor: 1.0,
            diffusion_sig_factor: 1.0,
            diffusion_gt_factor: 1.0,
            upwind_limit: 1.0e99,
            gamma_full_on: 40.0,
            gamma_full_off: 50.0,
            t_full_on: -1.0,
            t_full_off: -1.0,
            min_x_hard_limit: -1.0e-3,
            x_total_atol: 1.0e-6,
            x_total_rtol: 1.0e-6,
            tol_correction_max: 3.0e-1,
            tol_correction_norm: 3.0e-3,
            diffusion_factor: &diff_fac,
            use_other_diffusion_factor: false,
            extra_diffusion_factor: None,
            use_isolve,
            rtol_for_isolve: 1.0e-4,
            atol_for_isolve: 1.0e-5,
            maxsteps_for_isolve: 1000,
            isolve_solver_name: solver_name,
            nzlo_in: 0,
            nzhi_in: nz - 1,
        };
        let mut out = SolveDiffusionOutput {
            typical_charge: &mut typ_q,
            xa: &mut xa,
            x_init: &mut x_init,
            x_final: &mut x_final,
            d_self_face: &mut d_self,
            v_advection_face: &mut v_adv,
            v_total_face: &mut v_tot,
            vln_p_face: &mut vlnp,
            vln_t_face: &mut vlnt,
            v_rad_face: &mut vrad,
            g_rad_face: &mut g_rad,
            e_field_face: &mut e_fld,
            g_field_face: &mut g_fld,
        };

        let stats = do_solve_diffusion(&inp, &mut out).unwrap_or_else(|e| {
            panic!("do_solve_diffusion({cid}) failed: {e}")
        });

        let want_steps =
            oracle_val(&oracles, "solve_diff", cid, "steps_used", 0, 0);
        assert_eq!(
            stats.steps_used as f64, want_steps,
            "solve_diff/{cid}/steps_used mismatch"
        );

        let x_tol = 1.0e-12;
        let v_tol = 1.0e-11;

        let mut moved = false;
        for k in 0..nz {
            let k1 = k + 1;
            let we = oracle_val(&oracles, "solve_diff", cid, "E_field", k1, 0);
            let wg = oracle_val(&oracles, "solve_diff", cid, "g_field", k1, 0);
            assert_close(
                &format!("solve_diff/{cid}/E_field[{k1}]"),
                e_fld[k],
                we,
                v_tol,
                1.0e-25,
            );
            assert_close(
                &format!("solve_diff/{cid}/g_field[{k1}]"),
                g_fld[k],
                wg,
                v_tol,
                1.0e-25,
            );
            for j in 0..nc {
                let j1 = j + 1;
                let wx =
                    oracle_val(&oracles, "solve_diff", cid, "X_final", j1, k1);
                let wv =
                    oracle_val(&oracles, "solve_diff", cid, "v_total", j1, k1);
                if (x_final[k * nc + j] - x_init[k * nc + j]).abs() > 1.0e-9 {
                    moved = true;
                }
                assert_close(
                    &format!("solve_diff/{cid}/X_final[{j1},{k1}]"),
                    x_final[k * nc + j],
                    wx,
                    x_tol,
                    1.0e-14,
                );
                assert_close(
                    &format!("solve_diff/{cid}/v_total[{j1},{k1}]"),
                    v_tot[k * nc + j],
                    wv,
                    v_tol,
                    1.0e-25,
                );
            }
        }
        assert!(moved, "solve_diff/{cid} did not evolve X_final from X_init");

        // Verify species conservation across the domain
        for j in 0..nc {
            let m0: f64 = (0..nz).map(|k| dm_in[k] * x_init[k * nc + j]).sum();
            let m1: f64 = (0..nz).map(|k| dm_in[k] * x_final[k * nc + j]).sum();
            assert_close(
                &format!("solve_diff/{cid}/mass_conserve[{j}]"),
                m1,
                m0,
                1.0e-11,
                1.0e-20,
            );
        }
    }
}

fn build_wd_phase_sep_model(is_one: bool) -> StarModel {
    let mut controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        calculate_brunt_b: true,
        use_ledoux_criterion: true,
        do_phase_separation: true,
        do_phase_separation_heating: true,
        phase_separation_option: if is_one {
            String::from("ONe")
        } else {
            String::from("CO")
        },
        ..StarControls::default()
    };
    controls.diffusion.phase_separation_mixing_use_brunt = false;
    controls.mix_wind.min_dq = 1.0e-14;

    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load zams base");
    let nz = 8_usize;
    model.resize_zones(nz);
    model.eos_config.use_skye = true;
    model.eos_config.use_pc = false;
    model.eos_config.skye.skye_solid_mixing_rule = SkyeSolidMixingRule::Ogata;
    model.kap_config = KapConfig {
        use_type2_opacities: true,
        zbase: 0.02,
        ..model.kap_config
    };
    model.mstar = 0.6 * M_SUN;
    model.xmstar = model.mstar;
    model.m_center = 0.0;
    model.r_center = 0.0;
    model.l_center = 0.0;
    model.aux.crystal_core_boundary_mass = 0.0;

    let catalog = NuclideCatalog::builtin_ref();
    let iso_idx = |name: &str| -> usize {
        let cid = catalog.get_by_name(name).expect(name).chem_id;
        model.net_iso[cid].expect(name)
    };
    let ic12_i = iso_idx("c12");
    let io16_i = iso_idx("o16");
    let ine20_i = iso_idx("ne20");

    let sp = model.species;
    model.xa.fill(0.0);
    for k in 0..nz {
        model.dq[k] = 1.0 / (nz as f64);
    }
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();

    for k in 0..nz {
        let kf = k as f64;
        if is_one {
            model.lnd[k] = flexmesa_math::log(3.0e6 * (1.0 + 1.2 * kf));
            model.ln_t[k] = flexmesa_math::log(9.5e6 - 6.0e5 * kf);
            if k < 2 {
                model.xa[k * sp + io16_i] = 0.90;
                model.xa[k * sp + ine20_i] = 0.10;
            } else {
                model.xa[k * sp + io16_i] = 0.55;
                model.xa[k * sp + ine20_i] = 0.45;
            }
        } else {
            model.lnd[k] = flexmesa_math::log(1.5e6 * (1.0 + 1.2 * kf));
            model.ln_t[k] = flexmesa_math::log(5.5e6 - 4.0e5 * kf);
            if k < 2 {
                model.xa[k * sp + ic12_i] = 0.85;
                model.xa[k * sp + io16_i] = 0.15;
            } else {
                model.xa[k * sp + ic12_i] = 0.40;
                model.xa[k * sp + io16_i] = 0.60;
            }
        }
    }

    let mut r3_run = 0.0_f64;
    for k in (0..nz).rev() {
        let dm_k = model.dm[k];
        let rho_k = flexmesa_math::exp(model.lnd[k]);
        r3_run += (3.0 * dm_k) / (PI4 * rho_k);
        model.ln_r[k] =
            flexmesa_math::log(flexmesa_math::pow(r3_run, 1.0 / 3.0));
        model.lum[k] = 1.0e-3 * L_SUN * model.q[k];
    }
    model.cgrav.fill(STANDARD_CGRAV);
    model.xa_start.clone_from(&model.xa);
    model.dt = 1.0e13;
    model.pack_xh();
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars wd");
    model
}

#[test]
fn test_phase_separation_co_and_one_match_mesa_26_4_1_oracles() {
    let oracles = load_oracles();

    // Direct checks against Section 5 (`kernels`) compiled MESA 26.4.1 hex oracles
    for idx in 1..=3 {
        let gam = oracle_val(&oracles, "kernels", "dstar_ocp", "Gamma", idx, 0);
        let kap = oracle_val(&oracles, "kernels", "dstar_ocp", "kappa", idx, 0);
        let want =
            oracle_val(&oracles, "kernels", "dstar_ocp", "Dstar", idx, 0);
        assert_close(
            &format!("kernels/dstar_ocp[{idx}]"),
            eval_caplan_ocp_dstar(gam, kap),
            want,
            1.0e-14,
            0.0,
        );
    }
    for idx in 1..=5 {
        let x_in = oracle_val(&oracles, "kernels", "blouin_xo", "X_in", idx, 0);
        let want_o = oracle_val(&oracles, "kernels", "blouin_xo", "dX", idx, 0);
        assert_close(
            &format!("kernels/blouin_xo[{idx}]"),
            blouin_delta_xo(x_in),
            want_o,
            1.0e-14,
            1.0e-16,
        );
        let x_ne =
            oracle_val(&oracles, "kernels", "blouin_xne", "X_in", idx, 0);
        let want_ne =
            oracle_val(&oracles, "kernels", "blouin_xne", "dX", idx, 0);
        assert_close(
            &format!("kernels/blouin_xne[{idx}]"),
            blouin_delta_xne(x_ne),
            want_ne,
            1.0e-14,
            1.0e-16,
        );
    }
    assert_eq!(blouin_delta_xo(0.0), 0.0);
    assert_eq!(blouin_delta_xo(1.0), 0.0);
    assert_eq!(blouin_delta_xne(0.0), 0.0);
    assert_eq!(blouin_delta_xne(1.0), 0.0);

    for (is_one, use_brunt, cid) in [
        (false, false, "blouin_co"),
        (true, false, "blouin_one"),
        (false, true, "blouin_co_brunt"),
    ] {
        let mut model = build_wd_phase_sep_model(is_one);
        model.controls.diffusion.phase_separation_mixing_use_brunt = use_brunt;
        let nz = model.nz;
        let sp = model.species;
        let catalog = NuclideCatalog::builtin_ref();
        let ic12_i = model.net_iso[catalog.get_by_name("c12").unwrap().chem_id]
            .expect("c12");
        let io16_i = model.net_iso[catalog.get_by_name("o16").unwrap().chem_id]
            .expect("o16");
        let ine20_i = model.net_iso
            [catalog.get_by_name("ne20").unwrap().chem_id]
            .expect("ne20");

        // Verify pre-step Skye EOS phase matches MESA 26.4.1
        for k in 0..nz {
            let k1 = k + 1;
            assert_close(
                &format!("ps/{cid}/phase_init[{k1}]"),
                model.eos_res[k][I_PHASE],
                oracle_val(&oracles, "phase_sep", cid, "phase_init", k1, 0),
                1.0e-5,
                1.0e-6,
            );
        }

        do_phase_separation_step(&mut model, 1.0e13).unwrap_or_else(|e| {
            panic!("do_phase_separation_step({cid}) failed: {e}")
        });

        assert_close(
            &format!("ps/{cid}/crystal_core_boundary_mass"),
            model.aux.crystal_core_boundary_mass,
            oracle_val(
                &oracles,
                "phase_sep",
                cid,
                "crystal_core_boundary_mass",
                0,
                0,
            ),
            1.0e-12,
            0.0,
        );
        assert_close(
            &format!("ps/{cid}/phase_sep_mixing_mass"),
            model.aux.phase_sep_mixing_mass,
            oracle_val(
                &oracles,
                "phase_sep",
                cid,
                "phase_sep_mixing_mass",
                0,
                0,
            ),
            1.0e-12,
            0.0,
        );

        for k in 0..nz {
            let k1 = k + 1;
            assert_close(
                &format!("ps/{cid}/X_c12[{k1}]"),
                model.xa[k * sp + ic12_i],
                oracle_val(&oracles, "phase_sep", cid, "X_c12", k1, 0),
                1.0e-12,
                1.0e-14,
            );
            assert_close(
                &format!("ps/{cid}/X_o16[{k1}]"),
                model.xa[k * sp + io16_i],
                oracle_val(&oracles, "phase_sep", cid, "X_o16", k1, 0),
                1.0e-12,
                1.0e-14,
            );
            assert_close(
                &format!("ps/{cid}/X_ne20[{k1}]"),
                model.xa[k * sp + ine20_i],
                oracle_val(&oracles, "phase_sep", cid, "X_ne20", k1, 0),
                1.0e-12,
                1.0e-14,
            );
            assert_close(
                &format!("ps/{cid}/eps_phase_sep[{k1}]"),
                model.aux.eps_phase_separation[k],
                oracle_val(&oracles, "phase_sep", cid, "eps_phase_sep", k1, 0),
                1.0e-8,
                1.0e-10,
            );
            assert_close(
                &format!("ps/{cid}/lnd_final[{k1}]"),
                model.lnd[k],
                oracle_val(&oracles, "phase_sep", cid, "lnd_final", k1, 0),
                1.0e-12,
                1.0e-12,
            );
            assert_close(
                &format!("ps/{cid}/lnPgas_final[{k1}]"),
                model.ln_pgas[k],
                oracle_val(&oracles, "phase_sep", cid, "lnPgas_final", k1, 0),
                1.0e-12,
                1.0e-12,
            );
            if k > 0 {
                assert_close(
                    &format!("ps/{cid}/gradr_final[{k1}]"),
                    model.gradr[k],
                    oracle_val(
                        &oracles,
                        "phase_sep",
                        cid,
                        "gradr_final",
                        k1,
                        0,
                    ),
                    1.0e-11,
                    1.0e-15,
                );
                assert_close(
                    &format!("ps/{cid}/grada_face_final[{k1}]"),
                    model.aux.mixing.grada_face[k],
                    oracle_val(
                        &oracles,
                        "phase_sep",
                        cid,
                        "grada_face_final",
                        k1,
                        0,
                    ),
                    1.0e-7,
                    1.0e-12,
                );
            }
        }

        // Verify turb_info.f90:143-159 mixing types persist through set_hydro_vars
        // (even when stale non-zero `sig` is present before the call).
        model.sig.fill(1.0e25);
        set_hydro_vars(&mut model, false, false)
            .expect("set_hydro_vars after phase_separation");
        assert_eq!(model.mixing_type[nz - 1], MixingType::Crystallized);
        assert_eq!(model.d_mix[nz - 1], 0.0);
        assert_eq!(model.mlt_d[nz - 1], 0.0);
        assert_eq!(model.conv_vel[nz - 1], 0.0);
        // `set_no_mixing` does not write `sig` (`turb_info.f90:291-332`).
        assert_eq!(model.sig[nz - 1], 1.0e25);
        assert_eq!(model.gradt[nz - 1], model.gradr[nz - 1]);
        assert!(
            model
                .mixing_type
                .contains(&MixingType::PhaseSeparationMixing),
            "{cid}: expected at least one PhaseSeparationMixing zone after set_hydro_vars"
        );
        for k in 0..nz {
            if matches!(
                model.mixing_type[k],
                MixingType::Crystallized | MixingType::PhaseSeparationMixing
            ) {
                assert_eq!(model.d_mix[k], 0.0, "{cid}: d_mix[{k}] must be 0");
                assert_eq!(model.mlt_d[k], 0.0, "{cid}: mlt_d[{k}] must be 0");
                assert_eq!(
                    model.conv_vel[k], 0.0,
                    "{cid}: conv_vel[{k}] must be 0"
                );
                assert_eq!(
                    model.sig[k], 1.0e25,
                    "{cid}: sig[{k}] must keep its prior value"
                );
                assert_eq!(
                    model.gradt[k], model.gradr[k],
                    "{cid}: gradt[{k}] must equal gradr[{k}]"
                );
            }
        }
    }
}

#[test]
fn test_phase_sep_brunt_b_mode_rejection_and_unconditional_solid_shutoff() {
    // 1. Loud rejection when calculate_brunt_b == false (`phase_separation.f90:375-377`),
    //    both when phase_separation_mixing_use_brunt is true and false.
    for use_brunt in [true, false] {
        let mut m_no_brunt = build_wd_phase_sep_model(false);
        m_no_brunt.controls.calculate_brunt_b = false;
        m_no_brunt
            .controls
            .diffusion
            .phase_separation_mixing_use_brunt = use_brunt;
        let err = do_phase_separation_step(&mut m_no_brunt, 1.0e13)
            .expect_err("calculate_brunt_b = false must fail loudly");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "use_brunt={use_brunt}: expected InvalidInput, got {err:?}"
        );
    }

    // 2. Default `phase_separation_mixing_use_brunt = true` (Ledoux/Brunt-B
    //    stability stopping) conserves species mass, updates shared `micro.rs`
    //    EOS/kap aux buffers, and stops at `brunt_B + grada_face - gradr > 0`.
    let mut m_brunt = build_wd_phase_sep_model(false);
    m_brunt.controls.diffusion.phase_separation_mixing_use_brunt = true;
    let nz = m_brunt.nz;
    let sp = m_brunt.species;
    let catalog = NuclideCatalog::builtin_ref();
    let ic12 = m_brunt.net_iso[catalog.get_by_name("c12").unwrap().chem_id]
        .expect("c12");
    let io16 = m_brunt.net_iso[catalog.get_by_name("o16").unwrap().chem_id]
        .expect("o16");
    let m_c12_before: f64 = (0..nz)
        .map(|k| m_brunt.dm[k] * m_brunt.xa[k * sp + ic12])
        .sum();
    let m_o16_before: f64 = (0..nz)
        .map(|k| m_brunt.dm[k] * m_brunt.xa[k * sp + io16])
        .sum();
    m_brunt.aux.kap_frac.fill([0.0; 4]);
    m_brunt.aux.dln_e_dxa_for_partials.fill(0.0);

    do_phase_separation_step(&mut m_brunt, 1.0e13)
        .expect("phase_separation_mixing_use_brunt = true must succeed");

    let m_c12_after: f64 = (0..nz)
        .map(|k| m_brunt.dm[k] * m_brunt.xa[k * sp + ic12])
        .sum();
    let m_o16_after: f64 = (0..nz)
        .map(|k| m_brunt.dm[k] * m_brunt.xa[k * sp + io16])
        .sum();
    assert_close(
        "brunt_mode/c12_mass",
        m_c12_after,
        m_c12_before,
        1.0e-12,
        0.0,
    );
    assert_close(
        "brunt_mode/o16_mass",
        m_o16_after,
        m_o16_before,
        1.0e-12,
        0.0,
    );
    assert!(m_brunt.aux.crystal_core_boundary_mass > 0.0);
    assert!(
        m_brunt.aux.phase_sep_mixing_mass
            > m_brunt.aux.crystal_core_boundary_mass
    );
    assert!(
        m_brunt
            .aux
            .eps_phase_separation
            .iter()
            .all(|v| v.is_finite()),
        "eps_phase_separation must be finite"
    );
    assert!(
        m_brunt
            .aux
            .kap_frac
            .iter()
            .any(|f| f.iter().any(|&v| v != 0.0)),
        "shared do_kap_range must populate aux.kap_frac"
    );
    // At the top of the mixed layer (`ktop`), Brunt-B stability must hold.
    let ktop = (1..nz)
        .find(|&k| m_brunt.m[k] <= m_brunt.aux.phase_sep_mixing_mass)
        .expect("locate mixed layer top");
    let alfa = m_brunt.dq[ktop - 1] / (m_brunt.dq[ktop - 1] + m_brunt.dq[ktop]);
    let grada_f =
        alfa * m_brunt.grada[ktop] + (1.0 - alfa) * m_brunt.grada[ktop - 1];
    assert!(
        m_brunt.brunt_b[ktop] + grada_f - m_brunt.gradr[ktop] > 0.0,
        "Brunt-B stability must hold at ktop={ktop}: B={}, grada={}, gradr={}",
        m_brunt.brunt_b[ktop],
        grada_f,
        m_brunt.gradr[ktop]
    );

    // 3. Unconditional solid-core mixing shutoff when `do_phase_separation = false`
    //    (`star/private/turb_info.f90:144-152`): `set_hydro_vars` must still mark
    //    crystallized zones (`phase > 0.5 && mu > 1.7`) as `Crystallized` with
    //    `d_mix == 0`, `sig == 0`, `gradt == gradr` even with sentinel boundary masses.
    let mut m_off = build_wd_phase_sep_model(false);
    m_off.controls.do_phase_separation = false;
    m_off.aux.crystal_core_boundary_mass = -1.0;
    m_off.aux.phase_sep_mixing_mass = -1.0;
    m_off.sig.fill(1.0e25);
    set_hydro_vars(&mut m_off, false, false)
        .expect("set_hydro_vars with do_phase_separation = false");
    assert_eq!(
        m_off.mixing_type[nz - 1],
        MixingType::Crystallized,
        "solid CO core must stay Crystallized even with do_phase_separation = false"
    );
    assert_eq!(m_off.d_mix[nz - 1], 0.0);
    assert_eq!(m_off.mlt_d[nz - 1], 0.0);
    assert_eq!(m_off.conv_vel[nz - 1], 0.0);
    assert_eq!(m_off.sig[nz - 1], 1.0e25);
    assert_eq!(m_off.gradt[nz - 1], m_off.gradr[nz - 1]);
}

#[test]
fn test_diffusion_phase_sep_guards_energy_gating_and_photo_v26_restart() {
    // 1. Parse #156 controls strictly and reject unsupported op_mono / other hooks
    let inlist_ok = "
&controls
  do_element_diffusion = .true.
  diffusion_use_cgs_solver = .true.
  diffusion_use_caplan = .true.
  diffusion_use_full_net = .true.
  do_WD_sedimentation_heating = .true.
  do_diffusion_heating = .false.
  diffusion_use_isolve = .true.
  diffusion_isolve_solver = 'ros2_solver'
  diffusion_rtol_for_isolve = 1d-4
  diffusion_atol_for_isolve = 1d-5
  diffusion_maxsteps_for_isolve = 1000
  do_phase_separation = .true.
  do_phase_separation_heating = .true.
  phase_separation_option = 'CO'
  phase_separation_mixing_use_brunt = .true.
  phase_separation_no_diffusion = .true.
/
";
    let tmp_dir = std::env::temp_dir()
        .join(format!("flexmesa_156_inlist_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("mkdir tmp");
    let ok_path = tmp_dir.join("inlist_ok");
    std::fs::write(&ok_path, inlist_ok).expect("write ok inlist");
    let bundle = InlistBundle::from_file(&ok_path)
        .expect("parse and validate valid #156 inlist");
    let ctrl = &bundle.controls;
    assert!(ctrl.do_element_diffusion);
    assert!(ctrl.diffusion.diffusion_use_cgs_solver);
    assert!(ctrl.diffusion.diffusion_use_caplan);
    assert!(ctrl.diffusion.diffusion_use_full_net);
    assert!(ctrl.diffusion.do_wd_sedimentation_heating);
    assert!(ctrl.diffusion.diffusion_use_isolve);
    assert_eq!(ctrl.diffusion.diffusion_isolve_solver, "ros2_solver");
    assert!(ctrl.diffusion.phase_separation_no_diffusion);

    let rad_path = tmp_dir.join("inlist_rad");
    std::fs::write(&rad_path, "&controls\n op_mono_method = 'hu'\n/\n")
        .expect("write rad inlist");
    assert!(
        InlistBundle::from_file(&rad_path).is_err(),
        "op_mono_method must fail loudly until C10 is ported"
    );

    let hook_path = tmp_dir.join("inlist_hook");
    std::fs::write(&hook_path, "&controls\n use_other_diffusion = .true.\n/\n")
        .expect("write hook inlist");
    assert!(
        InlistBundle::from_file(&hook_path).is_err(),
        "use_other_diffusion must fail loudly"
    );
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // 2. Mutual exclusion of do_diffusion_heating && do_wd_sedimentation_heating
    let mut base =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    base.controls.do_element_diffusion = true;
    base.controls.diffusion.do_diffusion_heating = true;
    base.controls.diffusion.do_wd_sedimentation_heating = true;
    let err_both = do_element_diffusion_step(&mut base, 1.0e11)
        .expect_err("both heating flags must fail loudly");
    assert!(matches!(err_both, StarError::InvalidInput(_)));

    // 3. Loud radiative levitation rejection when T_face <= max_T_for_radaccel
    let mut rad_m =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    rad_m.controls.do_element_diffusion = true;
    rad_m.controls.diffusion.diffusion_max_t_for_radaccel = 1.0e9;
    rad_m.controls.diffusion.diffusion_min_t_for_radaccel = 1.0e3;
    rad_m.d_mix.fill(0.0);
    let err_rad = do_element_diffusion_step(&mut rad_m, 1.0e11)
        .expect_err("radlev op_mono request must fail loudly");
    assert!(matches!(err_rad, StarError::InvalidInput(_)));

    // 4. Deferred bug #272 loud guard: nzlo == 0 with D_mix[0] > min_D_mix
    let mut m272 =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    m272.controls.do_element_diffusion = true;
    m272.controls.diffusion.diffusion_min_dq_at_surface = 0.0;
    m272.controls.diffusion.diffusion_min_dq_ratio_at_surface = 0.0;
    m272.controls.diffusion.diffusion_min_t_at_surface = 0.0;
    m272.controls.diffusion.d_mix_ignore_diffusion = 1.0;
    m272.d_mix.fill(0.0);
    m272.d_mix[0] = 1.0e8;
    let err_272 = do_element_diffusion_step(&mut m272, 1.0e11)
        .expect_err("#272 nzlo==0 mixed surface must fail loudly");
    assert!(matches!(err_272, StarError::InvalidInput(_)));

    // 5. Deferred bug #273 verbatim behavior: invalid class representative -> Ok(())
    let mut m273 =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    m273.controls.do_element_diffusion = true;
    m273.controls.diffusion.diffusion_num_classes = 1;
    m273.controls.diffusion.diffusion_class_representative[0] =
        String::from("not_an_isotope");
    m273.d_mix.fill(0.0);
    let xa_before = m273.xa.clone();
    do_element_diffusion_step(&mut m273, 1.0e11)
        .expect("#273 invalid class rep returns Ok(()) verbatim");
    assert_eq!(m273.xa, xa_before);

    // 6. Deferred bug #271 loud guard: uniform CO WD where mix_outward reaches k=0
    let mut m271 = build_wd_phase_sep_model(false);
    let sp = m271.species;
    let catalog = NuclideCatalog::builtin_ref();
    let ic12_i =
        m271.net_iso[catalog.get_by_name("c12").unwrap().chem_id].expect("c12");
    let io16_i =
        m271.net_iso[catalog.get_by_name("o16").unwrap().chem_id].expect("o16");
    for k in 0..2 {
        m271.xa[k * sp + ic12_i] = 0.40;
        m271.xa[k * sp + io16_i] = 0.60;
    }
    set_hydro_vars(&mut m271, false, false).expect("set_hydro_vars #271");
    let err_271 = do_phase_separation_step(&mut m271, 1.0e13)
        .expect_err("#271 mix_outward hitting k=0 must fail loudly");
    assert!(matches!(err_271, StarError::InvalidInput(_)));

    // 7. Hydro energy 3-way gating (hydro_energy.f90:252-258) & finish_element_diffusion;
    // these heating terms are `others_ad`, the second return value.
    let mut em =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    let k = em.nz / 2;
    em.controls.do_element_diffusion = true;
    em.controls.diffusion.do_wd_sedimentation_heating = false;
    em.controls.diffusion.do_diffusion_heating = false;
    em.aux.eps_wd_sedimentation[k] = 1.0e3;
    em.aux.eps_diffusion[k] = 2.0e3;
    let eq_off = setup_sources_and_others(&em, k).1.val;
    em.controls.diffusion.do_wd_sedimentation_heating = true;
    let eq_wd = setup_sources_and_others(&em, k).1.val;
    em.controls.diffusion.do_wd_sedimentation_heating = false;
    em.controls.diffusion.do_diffusion_heating = true;
    let eq_diff = setup_sources_and_others(&em, k).1.val;
    assert_ne!(
        eq_wd, eq_off,
        "WD sedimentation heating must affect energy eqn"
    );
    assert_ne!(eq_diff, eq_off, "diffusion heating must affect energy eqn");
    assert_ne!(eq_wd, eq_diff, "the two heating channels must differ");

    // Verify finish_element_diffusion computes eps_diffusion when enabled
    em.controls.diffusion.do_diffusion_heating = true;
    em.aux.eps_diffusion.fill(0.0);
    em.energy[k] = em.energy_start[k] - 5.0e12;
    finish_element_diffusion(&mut em, 1.0e10);
    assert!(
        (em.aux.eps_diffusion[k] - 5.0e2).abs() < 1.0e-10,
        "expected eps_diffusion = 500, got {}",
        em.aux.eps_diffusion[k]
    );

    // 8. PrimaryStateBuffer rollback, .mod header, and FMPHOTO2 v27 round-trip
    let mut wd = build_wd_phase_sep_model(false);
    do_phase_separation_step(&mut wd, 1.0e13).expect("wd phase sep");
    let saved_cryst = wd.aux.crystal_core_boundary_mass;
    let saved_mix = wd.aux.phase_sep_mixing_mass;
    assert!(saved_cryst > 0.0);
    assert!(saved_mix > saved_cryst);
    wd.aux.diffusion_call_number = 7;

    let snap = PrimaryStateBuffer::from_model(&wd);
    wd.aux.crystal_core_boundary_mass = 0.0;
    wd.aux.phase_sep_mixing_mass = -1.0;
    wd.aux.diffusion_call_number = 99;
    snap.restore_into(&mut wd).expect("restore primary state");
    assert_eq!(wd.aux.crystal_core_boundary_mass, saved_cryst);
    assert_eq!(wd.aux.phase_sep_mixing_mass, saved_mix);
    assert_eq!(wd.aux.diffusion_call_number, 7);

    let mod_txt = write_mod_string(&wd).expect("write_mod_string");
    assert!(
        mod_txt.contains("crystal_core_boundary_mass"),
        ".mod must serialize positive crystal_core_boundary_mass"
    );
    let wd_mod = read_mod_str(&mod_txt, wd.controls.clone(), wd.job.clone())
        .expect("read_mod_str");
    assert_close(
        "mod roundtrip crystal_core_boundary_mass",
        wd_mod.aux.crystal_core_boundary_mass,
        saved_cryst,
        1.0e-15,
        0.0,
    );

    assert_eq!(PHOTO_FORMAT_VERSION, 30);
    let photo = write_photo_bytes(&wd);
    let wd_photo = load_photo_bytes(&photo).expect("load_photo_bytes v30");
    assert_eq!(
        wd_photo.aux.crystal_core_boundary_mass.to_bits(),
        saved_cryst.to_bits()
    );
    assert_eq!(
        wd_photo.aux.phase_sep_mixing_mass.to_bits(),
        saved_mix.to_bits()
    );
    assert_eq!(wd_photo.aux.diffusion_call_number, 7);
    assert_eq!(
        wd_photo.controls.diffusion.diffusion_use_cgs_solver,
        wd.controls.diffusion.diffusion_use_cgs_solver
    );
    assert_eq!(
        wd_photo.aux.eps_phase_separation,
        wd.aux.eps_phase_separation
    );
}

#[test]
fn test_phase_sep_update_model_range_matches_set_mlt_vars_and_rotation_gating()
{
    // Verify `phase_separation::update_model_range` calls `set_mlt_vars_range`
    // (`phase_separation.f90:387-390`, `turb_info.f90:125-134`), applying both
    // `gradr_factor` and `rotation_flag && mlt_use_rotation_correction` (and
    // NOT keying on `use_gravity_rotation_correction`).
    for (rot_on, mlt_rot, grav_rot, gradr_fac) in [
        (false, false, false, 1.35_f64),
        (true, true, false, 1.25_f64),
        (true, false, true, 1.15_f64),
    ] {
        let mut model = build_wd_phase_sep_model(false);
        let nz = model.nz;
        model.controls.rotation_flag = rot_on;
        model.controls.mix_wind.mlt_use_rotation_correction = mlt_rot;
        model.controls.use_gravity_rotation_correction = grav_rot;
        model.aux.mixing.gradr_factor.fill(gradr_fac);
        model.aux.fp_rot = vec![0.85; nz];
        model.aux.ft_rot = vec![0.92; nz];

        do_phase_separation_step(&mut model, 1.0e13)
            .expect("phase separation step with MLT gradr_factor/rotation");

        let gradr_after_ps = model.gradr.clone();
        let grada_after_ps = model.aux.mixing.grada_face.clone();
        let gradt_after_ps = model.gradt.clone();
        let mlt_d_after_ps = model.mlt_d.clone();

        // Compare directly against `set_mlt_vars` on the exact same post-step
        // thermodynamic state (disable `rotation_flag` side-effect recalc by
        // calling `set_mlt_vars` after restoring fixed fp_rot/ft_rot).
        let saved_fp = model.aux.fp_rot.clone();
        let saved_ft = model.aux.ft_rot.clone();
        model.aux.doing_struct_burn_mix = true;
        set_mlt_vars(&mut model).expect("set_mlt_vars reference");
        model.aux.doing_struct_burn_mix = false;
        assert_eq!(model.aux.fp_rot, saved_fp);
        assert_eq!(model.aux.ft_rot, saved_ft);

        for k in 1..nz {
            assert_eq!(
                gradr_after_ps[k].to_bits(),
                model.gradr[k].to_bits(),
                "gradr[{k}] mismatch (rot={rot_on}, mlt_rot={mlt_rot}, grav_rot={grav_rot})"
            );
            assert_eq!(
                grada_after_ps[k].to_bits(),
                model.aux.mixing.grada_face[k].to_bits(),
                "grada_face[{k}] mismatch"
            );
            assert_eq!(
                gradt_after_ps[k].to_bits(),
                model.gradt[k].to_bits(),
                "gradt[{k}] mismatch"
            );
            assert_eq!(
                mlt_d_after_ps[k].to_bits(),
                model.mlt_d[k].to_bits(),
                "mlt_d[{k}] mismatch"
            );
        }
    }

    // Confirm `gradr_factor` and `mlt_use_rotation_correction` measurably shift
    // `gradr` relative to baseline (so the comparison above is non-trivial).
    let mut m_base = build_wd_phase_sep_model(false);
    do_phase_separation_step(&mut m_base, 1.0e13).expect("baseline ps");

    let mut m_fac = build_wd_phase_sep_model(false);
    m_fac.aux.mixing.gradr_factor.fill(1.5);
    do_phase_separation_step(&mut m_fac, 1.0e13).expect("gradr_factor ps");
    assert_close(
        "gradr_factor scaling on interior face",
        m_fac.gradr[1] / m_base.gradr[1],
        1.5,
        1.0e-14,
        0.0,
    );

    let mut m_rot_off = build_wd_phase_sep_model(false);
    let nz = m_rot_off.nz;
    m_rot_off.controls.rotation_flag = true;
    m_rot_off.controls.mix_wind.mlt_use_rotation_correction = false;
    m_rot_off.controls.use_gravity_rotation_correction = true;
    m_rot_off.aux.fp_rot = vec![0.8; nz];
    m_rot_off.aux.ft_rot = vec![0.9; nz];
    do_phase_separation_step(&mut m_rot_off, 1.0e13).expect("rot_off ps");
    assert_eq!(
        m_rot_off.gradr[1].to_bits(),
        m_base.gradr[1].to_bits(),
        "use_gravity_rotation_correction must NOT alter gradr when mlt_use_rotation_correction is false"
    );

    let mut m_rot_on = build_wd_phase_sep_model(false);
    m_rot_on.controls.rotation_flag = true;
    m_rot_on.controls.mix_wind.mlt_use_rotation_correction = true;
    m_rot_on.controls.use_gravity_rotation_correction = false;
    m_rot_on.aux.fp_rot = vec![0.8; nz];
    m_rot_on.aux.ft_rot = vec![0.9; nz];
    do_phase_separation_step(&mut m_rot_on, 1.0e13).expect("rot_on ps");
    assert_close(
        "mlt_use_rotation_correction ft/fp ratio",
        m_rot_on.gradr[1] / m_base.gradr[1],
        0.9 / 0.8,
        1.0e-14,
        0.0,
    );
}

#[test]
fn test_diffusion_dump_call_number_and_deferred_286_287_288_guards() {
    // 1. Strict inlist loading rejects `diffusion_dump_call_number`
    let tmp_dir = std::env::temp_dir()
        .join(format!("flexmesa_156_dump_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("mkdir tmp");
    let dump_path = tmp_dir.join("inlist_dump");
    std::fs::write(
        &dump_path,
        "&controls\n  diffusion_dump_call_number = 1\n/\n",
    )
    .expect("write dump inlist");
    assert!(
        InlistBundle::from_file(&dump_path).is_err(),
        "diffusion_dump_call_number must fail strict inlist validation"
    );
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // 2. Runtime `diffusion_dump_call_number`: call 1 runs when set to 2, call 2
    //    fails loudly matching `element_diffusion.f90:242, 293, 361`.
    let mut m_dump =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    m_dump.controls.do_element_diffusion = true;
    m_dump.controls.diffusion.diffusion_dump_call_number = 2;
    m_dump.d_mix.fill(0.0);
    do_element_diffusion_step(&mut m_dump, 1.0e11)
        .expect("call 1 before diffusion_dump_call_number = 2 must succeed");
    assert_eq!(m_dump.aux.diffusion_call_number, 1);
    let err_dump = do_element_diffusion_step(&mut m_dump, 1.0e11).expect_err(
        "call 2 matching diffusion_dump_call_number = 2 must fail loudly",
    );
    let msg_dump = err_dump.to_string();
    assert!(
        matches!(err_dump, StarError::InvalidInput(_))
            && msg_dump.contains("dump_diffusion_info"),
        "unexpected error for diffusion_dump_call_number: {msg_dump}"
    );

    // 3. Radiative levitation (`op_mono`) loud error cites #286
    let mut m_rad =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    m_rad.controls.do_element_diffusion = true;
    m_rad.controls.diffusion.diffusion_max_t_for_radaccel = 1.0e9;
    m_rad.d_mix.fill(0.0);
    let err_rad = do_element_diffusion_step(&mut m_rad, 1.0e11)
        .expect_err("op_mono must fail loudly");
    assert!(
        err_rad.to_string().contains("#286"),
        "op_mono error must cite #286, got: {err_rad}"
    );

    // 4. #287 loud OOB guards: diffusion_num_classes == 0 and dq(nz+1) when
    //    diffusion_min_T_at_surface pushes nzlo to the innermost zone.
    let mut m_nc0 =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    m_nc0.controls.do_element_diffusion = true;
    m_nc0.controls.diffusion.diffusion_use_full_net = false;
    m_nc0.controls.diffusion.diffusion_num_classes = 0;
    let err_nc0 = do_element_diffusion_step(&mut m_nc0, 1.0e11)
        .expect_err("diffusion_num_classes == 0 must fail loudly (#287)");
    assert!(
        matches!(err_nc0, StarError::InvalidInput(_))
            && err_nc0.to_string().contains("#287"),
        "unexpected error for nc=0: {err_nc0}"
    );

    let mut m_dq_oob =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    m_dq_oob.controls.do_element_diffusion = true;
    m_dq_oob.controls.diffusion.diffusion_min_t_at_surface = 1.0e20;
    m_dq_oob
        .controls
        .diffusion
        .diffusion_min_dq_ratio_at_surface = 10.0;
    let err_dq = do_element_diffusion_step(&mut m_dq_oob, 1.0e11).expect_err(
        "nzlo == nz-1 with min_dq_ratio > 0 must fail loudly (#287)",
    );
    assert!(
        matches!(err_dq, StarError::InvalidInput(_))
            && err_dq.to_string().contains("#287"),
        "unexpected error for dq(nz+1) OOB: {err_dq}"
    );

    // 5. #288 `setup_struct_info` typical_charge smoothing for 2 active zones:
    //    fails loudly for nzlo == 0 (OOB) and calculate_ionization == true
    //    (uninitialized), and matches upstream stencil when nzlo > 0 and
    //    !calculate_ionization.
    let nz = 4_usize;
    let nc = 1_usize;
    let m = 2_usize;
    let zeros = vec![0.0_f64; nz];
    let ones = vec![1.0_f64; nz];
    let temps = vec![1.0e7_f64; nz];
    let lnts: Vec<f64> = temps.iter().map(|&t| flexmesa_math::log(t)).collect();
    let rhos = vec![1.0_f64; nz];
    let lnds = vec![0.0_f64; nz];
    let class_z = [2_usize];

    let run_smooth = |nzlo: usize, nzhi: usize, calc_ion: bool| {
        let mut rho_f = vec![0.0; nz];
        let mut t_f = vec![0.0; nz];
        let mut fpr = vec![0.0; nz];
        let mut dlnp = vec![0.0; nz];
        let mut dlnt = vec![0.0; nz];
        let mut dlnr = vec![0.0; nz];
        let mut z = vec![0.0; nz * m];
        let mut typ_q = vec![1.0, 2.0, 5.0, 8.0];
        let mut xm_f = vec![0.0; nz];
        let mut rad_f = vec![0.0; nz * m];
        let mut g_rad = vec![0.0; nz * nc];
        let res = setup_struct_info(
            nzlo, nzhi, nc, m, &zeros, &zeros, &zeros, &ones, &ones, &ones,
            &temps, &lnts, &rhos, &lnds, &ones, &ones, &class_z, calc_ion, 1,
            -1.0, &mut rho_f, &mut t_f, &mut fpr, &mut dlnp, &mut dlnt,
            &mut dlnr, &mut z, &mut typ_q, &mut xm_f, &mut rad_f, &mut g_rad,
        );
        (res, typ_q)
    };

    let (r_oob, _) = run_smooth(0, 1, false);
    let err_oob =
        r_oob.expect_err("2-zone smoothing at nzlo=0 must fail (#288)");
    assert!(err_oob.to_string().contains("#288"));

    let (r_uninit, _) = run_smooth(1, 2, true);
    let err_uninit = r_uninit.expect_err(
        "2-zone smoothing with calculate_ionization must fail (#288)",
    );
    assert!(err_uninit.to_string().contains("#288"));

    let (r_defined, q_defined) = run_smooth(1, 2, false);
    r_defined
        .expect("2-zone smoothing with nzlo=1 and fixed charges is defined");
    // Upstream pass: q[1] = (2*2 + 5)/3 = 3.0; then q[1] = (2*3.0 + q[0]=1.0)/3 = 7/3;
    // then q[2] = q[1] = 7/3.
    assert_close(
        "2-zone upstream smooth q[1]",
        q_defined[1],
        7.0 / 3.0,
        1.0e-15,
        0.0,
    );
    assert_close(
        "2-zone upstream smooth q[2]",
        q_defined[2],
        7.0 / 3.0,
        1.0e-15,
        0.0,
    );
}
