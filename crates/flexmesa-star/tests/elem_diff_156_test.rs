// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone self-contained unit and invariant tests for element diffusion
//! (`#156`): Rosenbrock `isolve` tableau/solver coverage, `use_full_net`
//! ionization flag persistence, `nzlo`/`nzhi` boundary controls, sedimentation
//! heating accounting, `diffusion_procs` fixup/taper/remap invariants, and
//! CGS velocity limiting.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, L_SUN, M_SUN, PI, PI4, STANDARD_CGRAV};
use flexmesa_eos::EosConfig;
use flexmesa_kap::KapConfig;
use flexmesa_math::{atan, exp, log, pow, with_numerics_mode, NumericsMode};
use flexmesa_star::diffusion::{
    adjust_timestep, do_solve_diffusion, positivity, set_diffusion_classes,
    IsolveSolver, SolveDiffusionInput, SolveDiffusionOutput,
    SolveDiffusionStats,
};
use flexmesa_star::diffusion_procs::{
    fix_negative_masses, fix_single_point_extremes, fix_species_conservation,
    fixup, get1_dx_dm, get_limit_coeffs, redistribute_mass,
};
use flexmesa_star::diffusion_support::{
    get1_diffusion_velocities, get1_flow_coeffs,
};
use flexmesa_star::element_diffusion::{
    do_element_diffusion_step, finish_element_diffusion,
};
use flexmesa_star::{
    load_zams_model, set_hydro_vars, StarControls, StarError, StarJobControls,
    StarModel,
};

fn assert_close(label: &str, got: f64, want: f64, rtol: f64, atol: f64) {
    let diff = (got - want).abs();
    let scale = atol.max(rtol * want.abs().max(got.abs()));
    assert!(
        diff <= scale,
        "{label}: got={got:.17e}, want={want:.17e}, diff={diff:.3e} > tol={scale:.3e}"
    );
}

struct SynthDiffusionGrid {
    nz: usize,
    species: usize,
    nc: usize,
    cls: [usize; 4],
    class_a: [f64; 4],
    class_z: [usize; 4],
    diff_fac: [f64; 4],
    abar: Vec<f64>,
    free_e: Vec<f64>,
    dm_in: Vec<f64>,
    dm_bar: Vec<f64>,
    t: Vec<f64>,
    ln_t: Vec<f64>,
    rho: Vec<f64>,
    lnd: Vec<f64>,
    r_mid: Vec<f64>,
    r_face: Vec<f64>,
    dlnp_dm_f: Vec<f64>,
    dlnt_dm_f: Vec<f64>,
    dlnrho_dm_f: Vec<f64>,
    grav: Vec<f64>,
    eta: Vec<f64>,
    gam: Vec<f64>,
    phase: Vec<f64>,
    xa_init: Vec<f64>,
}

fn build_solar_diffusion_grid(nz: usize) -> SynthDiffusionGrid {
    let species = 4_usize;
    let nc = 4_usize;
    let m_tot = M_SUN;
    let r_out = 6.0e10_f64;
    let dr = 0.5 * r_out / (nz as f64);

    let mut r_face = vec![0.0_f64; nz];
    let mut r_mid = vec![0.0_f64; nz];
    let mut dm_in = vec![0.0_f64; nz];
    let mut dm_bar = vec![0.0_f64; nz];
    let mut rho = vec![0.0_f64; nz];
    let mut t = vec![0.0_f64; nz];
    let mut lnd = vec![0.0_f64; nz];
    let mut ln_t = vec![0.0_f64; nz];
    let mut grav = vec![0.0_f64; nz];
    let mut abar = vec![0.0_f64; nz];
    let mut free_e = vec![0.0_f64; nz];
    let mut dlnp_dm_f = vec![0.0_f64; nz];
    let mut dlnt_dm_f = vec![0.0_f64; nz];
    let mut dlnrho_dm_f = vec![0.0_f64; nz];
    let mut xa_init = vec![0.0_f64; nz * species];

    for k in 0..nz {
        let kf = k as f64;
        r_face[k] = r_out - kf * dr;
        r_mid[k] = r_out - (kf + 0.5) * dr;
        dm_in[k] = (0.05 * m_tot) / (nz as f64);
        dm_bar[k] = dm_in[k];
        rho[k] = 1.0 * (1.0 + 0.35 * kf);
        t[k] = 3.0e6 * (1.0 + 0.12 * kf);
        grav[k] = 2.7e4;
        abar[k] = 1.30;
        free_e[k] = 0.85;
        xa_init[k * species] = 0.72 - 0.012 * kf;
        xa_init[k * species + 1] = 0.26 + 0.010 * kf;
        xa_init[k * species + 2] = 0.015 + 0.0015 * kf;
        xa_init[k * species + 3] = 0.005 + 0.0005 * kf;
        lnd[k] = log(rho[k]);
        ln_t[k] = log(t[k]);
        let denom = PI4 * r_mid[k] * r_mid[k] * rho[k];
        dlnp_dm_f[k] = -2.0e-10 / denom;
        dlnt_dm_f[k] = -6.0e-11 / denom;
        dlnrho_dm_f[k] = -1.4e-10 / denom;
    }

    SynthDiffusionGrid {
        nz,
        species,
        nc,
        cls: [0, 1, 2, 3],
        class_a: [1.0, 4.0, 12.0, 56.0],
        class_z: [1, 2, 6, 26],
        diff_fac: [1.0; 4],
        abar,
        free_e,
        dm_in,
        dm_bar,
        t,
        ln_t,
        rho,
        lnd,
        r_mid,
        r_face,
        dlnp_dm_f,
        dlnt_dm_f,
        dlnrho_dm_f,
        grav,
        eta: vec![-1.0; nz],
        gam: vec![0.05; nz],
        phase: vec![0.0; nz],
        xa_init,
    }
}

struct IsolveRun {
    stats: SolveDiffusionStats,
    x_init: Vec<f64>,
    x_final: Vec<f64>,
    xa: Vec<f64>,
}

fn run_isolve_case(
    g: &SynthDiffusionGrid,
    solver_name: &str,
    rtol: f64,
    atol: f64,
    maxsteps: usize,
) -> Result<IsolveRun, StarError> {
    let nz = g.nz;
    let nc = g.nc;
    let mut xa = g.xa_init.clone();
    let mut x_init = vec![0.0_f64; nz * nc];
    let mut x_final = vec![0.0_f64; nz * nc];
    let mut typ_q = vec![0.0_f64; nz * nc];
    for k in 0..nz {
        typ_q[k * nc] = 1.0;
        typ_q[k * nc + 1] = 2.0;
        typ_q[k * nc + 2] = 6.0;
        typ_q[k * nc + 3] = 26.0;
    }
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
        species: g.species,
        nc,
        class: &g.cls,
        class_a: &g.class_a,
        class_z: &g.class_z,
        has_h1: true,
        has_he4: true,
        abar: &g.abar,
        free_e: &g.free_e,
        dm_bar_in: &g.dm_bar,
        dm_in: &g.dm_in,
        t: &g.t,
        ln_t: &g.ln_t,
        rho: &g.rho,
        lnd: &g.lnd,
        r_mid: &g.r_mid,
        r_face_in: &g.r_face,
        dln_p_dm_face: &g.dlnp_dm_f,
        dln_t_dm_face: &g.dlnt_dm_f,
        dln_rho_dm_face: &g.dlnrho_dm_f,
        grav: &g.grav,
        eta: &g.eta,
        gamma: &g.gam,
        phase: &g.phase,
        r_center: 0.0,
        star_dt: 1.0e14,
        total_time: 1.0e14,
        pure_coulomb: false,
        diffusion_use_paquette: false,
        diffusion_use_caplan: false,
        diffusion_use_cgs_solver: false,
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
        diffusion_factor: &g.diff_fac,
        use_other_diffusion_factor: false,
        extra_diffusion_factor: None,
        use_isolve: true,
        rtol_for_isolve: rtol,
        atol_for_isolve: atol,
        maxsteps_for_isolve: maxsteps,
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
    let stats = do_solve_diffusion(&inp, &mut out)?;
    Ok(IsolveRun {
        stats,
        x_init,
        x_final,
        xa,
    })
}

#[test]
fn test_diffusion_use_full_net_persists_calculates_ionization() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    model.controls.do_element_diffusion = true;
    model.controls.diffusion.diffusion_use_full_net = true;
    model.controls.diffusion.diffusion_calculates_ionization = false;
    model.controls.diffusion.diffusion_dt_limit = 0.0;
    model.d_mix.fill(0.0);

    do_element_diffusion_step(&mut model, 1.0e11)
        .expect("full-net element diffusion step");

    // Upstream `element_diffusion.f90:244-246` persists
    // `diffusion_calculates_ionization = .true.` when `diffusion_use_full_net`
    // is enabled.
    assert!(
        model.controls.diffusion.diffusion_calculates_ionization,
        "diffusion_use_full_net must persist diffusion_calculates_ionization"
    );

    // Subsequent lumped-class call with `diffusion_use_full_net = false` must
    // continue calculating ionization rather than reading the dummy table.
    model.controls.diffusion.diffusion_use_full_net = false;
    model.controls.diffusion.diffusion_num_classes = 4;
    for (idx, name) in ["h1", "he4", "c12", "o16"].iter().enumerate() {
        model.controls.diffusion.diffusion_class_representative[idx] =
            String::from(*name);
        model.controls.diffusion.diffusion_class_typical_charge[idx] = 99.0;
    }
    model.controls.diffusion.diffusion_class_a_max[..4]
        .copy_from_slice(&[2.0, 4.0, 12.0, 10000.0]);

    do_element_diffusion_step(&mut model, 1.0e11)
        .expect("subsequent lumped-class diffusion step");
    for &q in &model.aux.typical_charge {
        assert!(
            q <= 30.0 && (q - 99.0).abs() > 1.0,
            "persisted ionization must compute charges rather than 99.0, got {q}"
        );
    }
}

#[test]
fn test_isolve_ros3p_exact_tableau_and_all_solvers_converge() {
    with_numerics_mode(NumericsMode::Strict, || {
        assert_eq!(
            IsolveSolver::from_name("ros3p_solver").expect("ros3p"),
            IsolveSolver::Ros3p
        );
        assert!(matches!(
            IsolveSolver::from_name("not_a_solver"),
            Err(StarError::InvalidInput(_))
        ));

        let g = build_solar_diffusion_grid(12);
        let ref_run =
            run_isolve_case(&g, "rodas4_solver", 1.0e-7, 1.0e-8, 2000)
                .expect("rodas4 fine reference");

        let solvers = [
            "ros2_solver",
            "rose2_solver",
            "ros3p_solver",
            "ros3pl_solver",
            "rodas3_solver",
            "rodas4_solver",
            "rodasp_solver",
        ];
        for name in solvers {
            let run = run_isolve_case(&g, name, 1.0e-5, 1.0e-6, 1000)
                .unwrap_or_else(|e| panic!("{name} failed: {e}"));
            assert!(run.stats.steps_used > 0, "{name} must accept >= 1 step");
            assert_eq!(
                run.stats.total_num_iters, 0,
                "{name} reports BE iters = 0"
            );

            for k in 0..g.nz {
                // `solve_with_isolve` does not re-normalize `X_final`, leaving
                // row sums within solver tolerances, while `set_new_xa`
                // normalizes `xa` to machine precision.
                let row_sum_x: f64 =
                    (0..g.nc).map(|j| run.x_final[k * g.nc + j]).sum();
                assert_close(
                    &format!("{name} x_final row_sum[{k}]"),
                    row_sum_x,
                    1.0,
                    1.0e-5,
                    0.0,
                );
                let row_sum_xa: f64 =
                    (0..g.species).map(|j| run.xa[k * g.species + j]).sum();
                assert_close(
                    &format!("{name} xa row_sum[{k}]"),
                    row_sum_xa,
                    1.0,
                    1.0e-13,
                    0.0,
                );
            }
            for j in 0..g.nc {
                let m0: f64 = (0..g.nz)
                    .map(|k| g.dm_in[k] * run.x_init[k * g.nc + j])
                    .sum();
                let m1: f64 = (0..g.nz)
                    .map(|k| g.dm_in[k] * run.x_final[k * g.nc + j])
                    .sum();
                assert_close(
                    &format!("{name} mass[{j}]"),
                    m1,
                    m0,
                    1.0e-11,
                    1.0e-20,
                );
            }
            let max_err = run
                .x_final
                .iter()
                .zip(ref_run.x_final.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_err < 1.0e-4,
                "{name} drifted {max_err:.3e} from fine rodas4 reference"
            );
        }

        // Verify Ros3p is bit-identical across runs and matches fine ref.
        let run_a = run_isolve_case(&g, "ros3p_solver", 1.0e-5, 1.0e-6, 1000)
            .expect("ros3p run a");
        let run_b = run_isolve_case(&g, "ros3p_solver", 1.0e-5, 1.0e-6, 1000)
            .expect("ros3p run b");
        assert_eq!(run_a.stats, run_b.stats);
        assert_eq!(run_a.x_final, run_b.x_final);
        assert_eq!(run_a.xa, run_b.xa);

        // Guard checks on invalid tolerances / unknown solver.
        assert!(matches!(
            run_isolve_case(&g, "bad_solver", 1.0e-4, 1.0e-5, 1000),
            Err(StarError::InvalidInput(_))
        ));
        assert!(matches!(
            run_isolve_case(&g, "ros2_solver", 1.0e-4, 0.0, 1000),
            Err(StarError::InvalidInput(_))
        ));
        assert!(matches!(
            run_isolve_case(&g, "ros2_solver", 1.0e-16, 1.0e-5, 1000),
            Err(StarError::InvalidInput(_))
        ));
    });
}

#[test]
fn test_element_diffusion_skip_sedimentation_heating_and_finish_accounting() {
    let base =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    let nz = base.nz;
    let sp = base.species;

    // 1. Skip when `do_element_diffusion == false`.
    let mut m_off = base.clone();
    m_off.controls.do_element_diffusion = false;
    m_off.aux.step_hist.num_diffusion_solver_steps = 7;
    m_off.aux.step_hist.num_diffusion_solver_iters = 11;
    m_off.aux.edv.resize(nz * sp, 1.0);
    m_off.aux.edv.fill(1.0);
    m_off.aux.eps_wd_sedimentation.resize(nz, 2.0);
    m_off.aux.eps_wd_sedimentation.fill(2.0);
    m_off.aux.eps_diffusion.resize(nz, 3.0);
    m_off.aux.eps_diffusion.fill(3.0);
    let xa_before = m_off.xa.clone();
    do_element_diffusion_step(&mut m_off, 1.0e11).expect("skip when disabled");
    assert_eq!(m_off.xa, xa_before);
    assert_eq!(m_off.aux.step_hist.num_diffusion_solver_steps, 0);
    assert_eq!(m_off.aux.step_hist.num_diffusion_solver_iters, 0);
    assert!(m_off.aux.edv.iter().all(|&v| v == 0.0));
    assert!(m_off.aux.eps_wd_sedimentation.iter().all(|&v| v == 0.0));
    assert!(m_off.aux.eps_diffusion.iter().all(|&v| v == 0.0));

    // 2. Skip when `dt < diffusion_dt_limit`.
    let mut m_dt = base.clone();
    m_dt.controls.do_element_diffusion = true;
    m_dt.controls.diffusion.diffusion_dt_limit = 1.0e12;
    m_dt.aux.step_hist.num_diffusion_solver_iters = 9;
    m_dt.aux.edv.resize(nz * sp, 1.0);
    m_dt.aux.edv.fill(1.0);
    m_dt.aux.eps_wd_sedimentation.resize(nz, 2.0);
    m_dt.aux.eps_wd_sedimentation.fill(2.0);
    do_element_diffusion_step(&mut m_dt, 1.0e11).expect("skip when dt < limit");
    assert_eq!(m_dt.xa, xa_before);
    assert_eq!(m_dt.aux.step_hist.num_diffusion_solver_steps, 0);
    assert_eq!(m_dt.aux.step_hist.num_diffusion_solver_iters, 0);
    assert!(m_dt.aux.edv.iter().all(|&v| v == 0.0));
    assert!(m_dt.aux.eps_wd_sedimentation.iter().all(|&v| v == 0.0));

    // 3. Active WD sedimentation heating and surface/center boundary fills,
    //    followed by `finish_element_diffusion`.
    let mut m_heat = base;
    m_heat.controls.do_element_diffusion = true;
    m_heat.controls.diffusion.diffusion_dt_limit = 0.0;
    m_heat.controls.diffusion.diffusion_use_full_net = true;
    m_heat.controls.diffusion.diffusion_use_cgs_solver = true;
    m_heat.controls.diffusion.do_diffusion_heating = false;
    m_heat.controls.diffusion.do_wd_sedimentation_heating = true;
    m_heat
        .controls
        .diffusion
        .min_xa_for_wd_sedimentation_heating = 1.0e-10;
    m_heat.d_mix.fill(0.0);
    m_heat.d_mix[(nz - 4)..nz].fill(1.0e8);

    do_element_diffusion_step(&mut m_heat, 1.0e11)
        .expect("sedimentation heating step");
    assert!(m_heat.aux.step_hist.num_diffusion_solver_steps > 0);
    assert_eq!(m_heat.aux.e_field[0], 0.0);
    assert_eq!(m_heat.aux.g_field_element_diffusion[0], 0.0);
    assert_eq!(m_heat.aux.eps_wd_sedimentation[0], 0.0);
    assert_eq!(m_heat.aux.e_field[nz - 1], 0.0);
    assert_eq!(m_heat.aux.g_field_element_diffusion[nz - 1], 0.0);
    assert_eq!(m_heat.aux.eps_wd_sedimentation[nz - 1], 0.0);
    assert!(m_heat.aux.eps_diffusion.iter().all(|&v| v == 0.0));

    let max_wd_sed = m_heat
        .aux
        .eps_wd_sedimentation
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_wd_sed > 0.0,
        "interior WD sedimentation heating expected"
    );

    let k_mid = nz / 2;
    m_heat.energy[k_mid] = m_heat.energy_start[k_mid] - 2.5e12;
    finish_element_diffusion(&mut m_heat, 1.0e10);
    assert_close(
        "finish_element_diffusion",
        m_heat.aux.eps_diffusion[k_mid],
        2.5e2,
        1.0e-12,
        0.0,
    );
}

fn build_16_zone_test_model() -> StarModel {
    let mut controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        do_element_diffusion: true,
        ..StarControls::default()
    };
    controls.diffusion.diffusion_dt_limit = 0.0;
    controls.diffusion.diffusion_use_full_net = true;
    controls.diffusion.diffusion_use_cgs_solver = true;
    controls.diffusion.diffusion_calculates_ionization = true;
    controls.diffusion.diffusion_min_dq_ratio_at_surface = 0.0;
    controls.diffusion.diffusion_min_dq_at_surface = 0.0;
    controls.diffusion.diffusion_min_t_at_surface = 0.0;

    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load zams base");
    let nz = 16_usize;
    model.resize_zones(nz);
    model.eos_config = EosConfig::default();
    model.kap_config = KapConfig {
        use_type2_opacities: true,
        zbase: 0.02,
        ..model.kap_config
    };
    model.mstar = 1.0 * M_SUN;
    model.m_center = 0.90 * M_SUN;
    model.xmstar = model.mstar - model.m_center;
    model.r_center = 3.5e10;
    model.dq.fill(1.0 / (nz as f64));
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();

    let cat = NuclideCatalog::builtin_ref();
    let idx = |name: &str| -> usize {
        let cid = cat.get_by_name(name).expect(name).chem_id;
        model.net_iso[cid].expect(name)
    };
    let ih1 = idx("h1");
    let ihe4 = idx("he4");
    let ic12 = idx("c12");
    let io16 = idx("o16");
    let sp = model.species;
    model.xa.fill(0.0);
    for k in 0..nz {
        let frac = (k as f64) / ((nz - 1) as f64);
        model.lnd[k] = log(1.0e-2 + 2.0 * frac);
        model.ln_t[k] = log(2.0e5 + 4.0e6 * frac);
        if k < 4 {
            model.xa[k * sp + ih1] = 0.72;
            model.xa[k * sp + ihe4] = 0.26;
        } else {
            model.xa[k * sp + ih1] = 0.02;
            model.xa[k * sp + ihe4] = 0.96;
        }
        model.xa[k * sp + ic12] = 0.01;
        model.xa[k * sp + io16] = 0.01;
    }

    let mut r3_run = pow(model.r_center, 3.0);
    for k in (0..nz).rev() {
        r3_run += (3.0 * model.dm[k]) / (PI4 * exp(model.lnd[k]));
        model.ln_r[k] = log(pow(r3_run, 1.0 / 3.0));
        model.lum[k] = L_SUN * model.q[k];
    }
    model.cgrav.fill(STANDARD_CGRAV);
    model.dt = 1.0e12;
    model.pack_xh();
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.d_mix.fill(0.0);
    model.mixing_type.fill(MixingType::NoMixing);
    model
}

#[test]
fn test_nzlo_nzhi_boundary_controls_and_phase_separation_cutoff() {
    with_numerics_mode(NumericsMode::Strict, || {
        let base = build_16_zone_test_model();
        let nz = base.nz;
        let sp = base.species;

        // 1. `diffusion_min_dq_at_surface` stops cleanly at the h1->he4 jump
        //    (`k = 4`), so zones `0..=3` merge identically and have zero edv.
        let mut m_dq = base.clone();
        m_dq.controls.diffusion.diffusion_min_dq_at_surface = 0.5;
        do_element_diffusion_step(&mut m_dq, 1.0e12)
            .expect("min_dq stops at comp jump");
        for k in 0..=3 {
            assert_eq!(m_dq.aux.e_field[k], 0.0);
            for j in 0..sp {
                assert_eq!(m_dq.aux.edv[k * sp + j], 0.0);
                assert_eq!(m_dq.xa[k * sp + j], m_dq.xa[3 * sp + j]);
            }
        }
        assert!(
            m_dq.aux.e_field[5].abs() > 0.0 || m_dq.aux.edv[5 * sp].abs() > 0.0,
            "interior below composition jump must remain active"
        );

        // 2. `phase_separation_no_diffusion` zeroes zones beyond the phase-
        //    separation boundary (`k >= 10`) without altering their `xa`.
        let mut m_ps = base.clone();
        m_ps.controls.do_phase_separation = true;
        m_ps.controls.diffusion.phase_separation_no_diffusion = true;
        m_ps.controls.diffusion.diffusion_min_t_at_surface =
            f64::midpoint(m_ps.t[1], m_ps.t[2]);
        m_ps.mixing_type[10] = MixingType::PhaseSeparationMixing;
        let xa_pre_ps = m_ps.xa.clone();
        do_element_diffusion_step(&mut m_ps, 1.0e12)
            .expect("phase_separation_no_diffusion step");
        for k in 11..nz {
            assert_eq!(m_ps.aux.e_field[k], 0.0);
            for j in 0..sp {
                assert_eq!(m_ps.aux.edv[k * sp + j], 0.0);
                assert_eq!(m_ps.xa[k * sp + j], xa_pre_ps[k * sp + j]);
            }
        }
        assert!(m_ps.aux.step_hist.num_diffusion_solver_steps > 0);

        // 3. When `nzhi_eff <= nzlo_eff`, `do_solve_diffusion` returns cleanly
        //    with zero steps and leaves `xa` completely untouched.
        let mut m_empty = base;
        m_empty.controls.do_phase_separation = true;
        m_empty.controls.diffusion.phase_separation_no_diffusion = true;
        m_empty.controls.diffusion.diffusion_min_t_at_surface =
            f64::midpoint(m_empty.t[3], m_empty.t[4]);
        m_empty.mixing_type[2] = MixingType::PhaseSeparationMixing;
        let xa_pre_empty = m_empty.xa.clone();
        do_element_diffusion_step(&mut m_empty, 1.0e12)
            .expect("nzhi <= nzlo clean zero-step return");
        assert_eq!(m_empty.xa, xa_pre_empty);
        assert_eq!(m_empty.aux.step_hist.num_diffusion_solver_steps, 0);
        assert_eq!(m_empty.aux.step_hist.num_diffusion_solver_iters, 0);
        assert!(m_empty.aux.e_field.iter().all(|&v| v == 0.0));
        assert!(m_empty.aux.edv.iter().all(|&v| v == 0.0));
    });
}

#[test]
fn test_diffusion_procs_fixup_extremes_negative_mass_taper_and_remap() {
    with_numerics_mode(NumericsMode::Strict, || {
        // 1. Cosine-tapered `get_limit_coeffs` across Gamma + truncation `k_max`.
        let nz = 6_usize;
        let gamma = [1.0, 2.0, 10.0, 15.0, 20.0, 25.0];
        let t = [1.0e6; 6];
        let phase = [0.0; 6];
        let mut lim_face = [0.0; 6];
        let k_max = get_limit_coeffs(
            0,
            nz - 1,
            5.0,
            20.0,
            1.0e4,
            1.0e3,
            &gamma,
            &t,
            &phase,
            &mut lim_face,
        );
        assert_eq!(k_max, 3);
        assert_close("lim_face[1]", lim_face[1], 1.0, 1.0e-15, 0.0);
        assert_close("lim_face[2]", lim_face[2], 0.75, 1.0e-15, 0.0);
        assert_close("lim_face[3]", lim_face[3], 0.25, 1.0e-15, 0.0);
        assert_eq!(lim_face[4], 0.0);
        assert_eq!(lim_face[5], 0.0);

        // 2. `fix_negative_masses`: tiny negative clip and neighbor borrowing.
        let nc = 2_usize;
        let cell_dm = [1.0e28; 4];
        let mut mass_neg = vec![
            0.70e28,
            0.30e28,
            -1.0e-14 * 1.0e28,
            0.30e28,
            0.70e28,
            0.30e28,
            0.70e28,
            0.30e28,
        ];
        fix_negative_masses(nc, 0, 3, &cell_dm, &mut mass_neg)
            .expect("tiny clip");
        assert_eq!(mass_neg[2], 0.0);

        let mut mass_borrow = vec![
            0.70e28, 0.30e28, -0.01e28, 0.30e28, 0.70e28, 0.30e28, 0.70e28,
            0.30e28,
        ];
        let col0_before: f64 = (0..4).map(|k| mass_borrow[k * nc]).sum();
        fix_negative_masses(nc, 0, 3, &cell_dm, &mut mass_borrow)
            .expect("neighbor borrow");
        assert_eq!(mass_borrow[2], 0.0);
        assert!(mass_borrow.iter().all(|&m| m >= 0.0));
        let col0_after: f64 = (0..4).map(|k| mass_borrow[k * nc]).sum();
        assert_close("borrow conserves", col0_after, col0_before, 1e-13, 0.0);

        // 3. `fix_species_conservation` rescale vs tolerance rejection.
        let mtotal: f64 = cell_dm.iter().sum();
        let xtotal_init = [0.70_f64, 0.30];
        let mut mass_cons = vec![
            0.70001e28, 0.30e28, 0.70001e28, 0.30e28, 0.70001e28, 0.30e28,
            0.70001e28, 0.30e28,
        ];
        let mut sum_mass = vec![0.0; 4];
        fix_species_conservation(
            nc,
            0,
            3,
            mtotal,
            &xtotal_init,
            1.0e-6,
            1.0e-3,
            &mut mass_cons,
            &mut sum_mass,
        )
        .expect("rescale within rtol");
        let new_tot0: f64 = (0..4).map(|k| mass_cons[k * nc]).sum();
        assert_close("rescaled mass", new_tot0, mtotal * 0.70, 1.0e-14, 0.0);

        let mut mass_drift = [
            0.72e28, 0.30e28, 0.72e28, 0.30e28, 0.72e28, 0.30e28, 0.72e28,
            0.30e28,
        ];
        let err_cons = fix_species_conservation(
            nc,
            0,
            3,
            mtotal,
            &xtotal_init,
            1.0e-9,
            1.0e-6,
            &mut mass_drift,
            &mut sum_mass,
        )
        .expect_err("large drift must reject");
        assert_eq!(err_cons.bad_j, Some(0));

        // 4. `fix_single_point_extremes`, `get1_dx_dm`, `redistribute_mass`,
        //    and full `fixup`.
        let ln_t = [15.0; 4];
        let mut mass_saw = vec![
            0.80e28, 0.20e28, 0.40e28, 0.60e28, 0.80e28, 0.20e28, 0.75e28,
            0.25e28,
        ];
        let mut sum_saw = vec![1.0e28; 4];
        let saw_c0_pre: f64 = (0..4).map(|k| mass_saw[k * nc]).sum();
        let saw_c1_pre: f64 = (0..4).map(|k| mass_saw[k * nc + 1]).sum();
        fix_single_point_extremes(
            nc,
            0,
            3,
            100.0,
            &ln_t,
            &mut sum_saw,
            &mut mass_saw,
        )
        .expect("remove sawtooth");
        assert_close(
            "extremes conserve c0",
            (0..4).map(|k| mass_saw[k * nc]).sum(),
            saw_c0_pre,
            1.0e-13,
            0.0,
        );
        assert_close(
            "extremes conserve c1",
            (0..4).map(|k| mass_saw[k * nc + 1]).sum(),
            saw_c1_pre,
            1.0e-13,
            0.0,
        );

        let mut dx_dm = vec![0.0; 4 * nc];
        for k in 0..4 {
            get1_dx_dm(k, nc, 0, 3, &mass_saw, &sum_saw, &mut dx_dm)
                .expect("get1_dx_dm");
            let s: f64 = (0..nc).map(|j| dx_dm[k * nc + j]).sum();
            assert_close("sum(dX_dm) == 0", s, 0.0, 0.0, 1.0e-15);
        }

        let m = nc + 1;
        let mut x = vec![
            0.72, 0.28, 0.0, 0.64, 0.36, 0.0, 0.74, 0.26, 0.0, 0.68, 0.32, 0.0,
        ];
        let xtotal_fix = [
            0.25 * (0.72 + 0.64 + 0.74 + 0.68),
            0.25 * (0.28 + 0.36 + 0.26 + 0.32),
        ];
        let mut mass_buf = vec![0.0; 4 * nc];
        let mut sum_buf = vec![0.0; 4];
        let mut slope_buf = vec![0.0; 4 * nc];
        fixup(
            4,
            nc,
            m,
            0,
            3,
            -1.0e-4,
            1.0e-9,
            1.0e-6,
            &cell_dm,
            mtotal,
            &xtotal_fix,
            &mut x,
            &ln_t,
            &mut sum_buf,
            &mut mass_buf,
            &mut slope_buf,
        )
        .expect("fixup succeeds");
        for k in 0..4 {
            assert_close(
                &format!("fixup sum_x[{k}]"),
                x[k * m] + x[k * m + 1],
                1.0,
                1.0e-14,
                0.0,
            );
        }

        let mut x_remap = vec![0.0; 4 * m];
        let xtotal_saw = [saw_c0_pre / mtotal, saw_c1_pre / mtotal];
        redistribute_mass(
            nc,
            m,
            0,
            3,
            4,
            &mut sum_saw,
            &mass_saw,
            &dx_dm,
            &xtotal_saw,
            &cell_dm,
            &mut x_remap,
        )
        .expect("redistribute_mass");
        for k in 0..4 {
            assert_close(
                &format!("remap sum[{k}]"),
                x_remap[k * m] + x_remap[k * m + 1],
                1.0,
                1.0e-14,
                0.0,
            );
        }
        let remap_c0: f64 = (0..4).map(|k| cell_dm[k] * x_remap[k * m]).sum();
        assert_close("remap conserves c0", remap_c0, saw_c0_pre, 1.0e-12, 0.0);
    });
}

#[test]
fn test_diffusion_utilities_positivity_classes_vlimit_and_h211b_controller() {
    with_numerics_mode(NumericsMode::Strict, || {
        // 1. `set_diffusion_classes` boundaries and full-net mapping.
        let a_vals = [1.0, 2.0, 3.0, 4.0, 14.0, 20.0];
        let cls_full = set_diffusion_classes(6, &a_vals, &[], true);
        assert_eq!(cls_full, vec![0, 1, 2, 3, 4, 5]);
        let cls_grp =
            set_diffusion_classes(4, &a_vals, &[2.0, 4.0, 16.0], false);
        assert_eq!(cls_grp, vec![0, 0, 1, 1, 2, 3]);

        // 2. `positivity` exact damping factor `0.8 * (-(old_x + eps) / dx)`.
        let x_pos = [0.5, 0.5, 0.2, 0.8];
        let del_ok = [0.1, -0.1, 0.05, -0.05];
        assert_eq!(positivity(2, 0, 1, &x_pos, 1.0e-3, &del_ok), 1.0);
        let del_neg = [0.0, 0.0, -0.5, 0.5];
        let want_lam = 0.8 * (-(0.2 + 1.0e-3) / (-0.5));
        assert_close(
            "positivity lambda",
            positivity(2, 0, 1, &x_pos, 1.0e-3, &del_neg),
            want_lam,
            1.0e-15,
            0.0,
        );

        // 3. `adjust_timestep` H211b low-pass controller & clamps.
        let dt_cold = adjust_timestep(0.1, 0.2, 0.0, 100.0, 0.0, 2.0, 0.75);
        assert_close("h211b cold-start clamp", dt_cold, 75.0, 1.0e-15, 0.0);

        let lim = |val: f64| 1.0 + 2.0 * atan((val - 1.0) / 2.0);
        let r1 = lim(0.1 / 0.12);
        let r2 = lim(0.1 / 0.15);
        let fac = lim(pow(r1, 0.25) * pow(r2, 0.25) * pow(100.0 / 80.0, -0.25));
        let dt_warm = adjust_timestep(0.1, 0.12, 0.15, 100.0, 80.0, 2.0, 0.75);
        assert_close("h211b warm", dt_warm, 100.0 * fac, 1.0e-15, 0.0);

        // 4. `get1_diffusion_velocities` `Vlimit` cap scales `sigma_lnc` and
        //    preserves baryon conservation; `get1_flow_coeffs` scales fluxes.
        let nc = 2_usize;
        let m = 3_usize;
        let ap = [-1.0e15, -2.0e15, 0.0];
        let at = [0.0; 3];
        let ar = [0.0; 3];
        let ax = [-1.0e10, 0.0, 0.0, 0.0, -1.0e10, 0.0, 0.0, 0.0, 0.0];
        let x_face = [0.7, 0.3, 0.0];
        let r_mid = [1.0e10, 0.9e10];
        let diff_fac = [1.0, 1.0];

        let mut vgt_uncap = [0.0; 2];
        let mut vgt_cap = [0.0; 2];
        let mut vlnp = [0.0; 2];
        let mut vlnt = [0.0; 2];
        let mut vrad = [0.0; 2];
        let mut sig_uncap = [0.0; 4];
        let mut sig_cap = [0.0; 4];

        get1_diffusion_velocities(
            1,
            nc,
            m,
            &ap,
            &at,
            &ar,
            &ax,
            1.0,
            1.0e7,
            -1.0e-10,
            0.0,
            1.0e5,
            0.0,
            &x_face,
            1.0e-2 * M_SUN,
            2.0e-2 * M_SUN,
            0.0,
            0.5e-2 * M_SUN,
            &r_mid,
            1.0e15,
            1.0,
            1.0e99,
            &diff_fac,
            true,
            &mut vgt_uncap,
            &mut vlnp,
            &mut vlnt,
            &mut vrad,
            &mut sig_uncap,
        );
        // Set `vlimit = 0.01` over `dt = 1e12` so `vmax_limit = 1e-5 cm/s`
        // actively caps `vgt` and rescales `sigma_lnc`.
        get1_diffusion_velocities(
            1,
            nc,
            m,
            &ap,
            &at,
            &ar,
            &ax,
            1.0,
            1.0e7,
            -1.0e-10,
            0.0,
            1.0e5,
            0.0,
            &x_face,
            1.0e-2 * M_SUN,
            2.0e-2 * M_SUN,
            0.01,
            0.5e-2 * M_SUN,
            &r_mid,
            1.0e12,
            1.0,
            1.0e99,
            &diff_fac,
            true,
            &mut vgt_cap,
            &mut vlnp,
            &mut vlnt,
            &mut vrad,
            &mut sig_cap,
        );

        let vmax_uncap = vgt_uncap[0].abs().max(vgt_uncap[1].abs());
        let vmax_cap = vgt_cap[0].abs().max(vgt_cap[1].abs());
        let vmax_limit = 0.01 * (r_mid[0] - r_mid[1]) / 1.0e12;
        assert!(vmax_uncap > vmax_limit);
        assert_close("vlimit cap", vmax_cap, vmax_limit, 1.0e-14, 0.0);
        let frac = vmax_limit / vmax_uncap;
        assert_close(
            "sigma_lnc scaled",
            sig_cap[0],
            sig_uncap[0] * frac,
            1.0e-14,
            0.0,
        );
        assert_close(
            "baryon conservation",
            x_face[0] * vgt_cap[0] + x_face[1] * vgt_cap[1],
            0.0,
            0.0,
            1.0e-14,
        );

        let c_div_x_f = [1.0, 0.25, 1.0];
        let mut gt_face = [0.0; 2];
        let mut d_self_face = [0.0; 2];
        let mut sig_face = [0.0; 4];
        get1_flow_coeffs(
            nc,
            &vgt_cap,
            1.5,
            2.0,
            &sig_cap,
            4.0 * PI * 1.0e20,
            1.0e30,
            &c_div_x_f,
            &mut gt_face,
            &mut d_self_face,
            &mut sig_face,
        );
        assert_close(
            "gt_face scaling",
            gt_face[1],
            2.0 * 4.0 * PI * 1.0e20 * vgt_cap[1],
            1.0e-14,
            0.0,
        );
        assert_eq!(d_self_face[0], sig_cap[0]);
    });
}
