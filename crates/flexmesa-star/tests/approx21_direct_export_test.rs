// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant tests verifying direct CGS / mass-fraction export for
//! `doing_approx21` networks (`net/private/net_eval.f90:232-237,308-338,431`,
//! `net/private/net_burn_const_density.f90:482`) without `QCONV` or `aion`
//! round-trips.

#![forbid(unsafe_code)]

use flexmesa_chem::{get_composition_info, NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{LN10, QCONV};
use flexmesa_eos::{eos_dt_get, EosConfig, HelmTable, I_CV, I_ETA};
use flexmesa_math::{exp10, Quad};
use flexmesa_net::{
    burn_1_zone, burn_1_zone_const_density, eval_net, eval_net_with_work,
    Burn1ZoneInput, BurnConstDensityInput, NetEvalResult, NetGeneralInfo,
    NetWorkOutputs, NetWorkState, NUM_RVS,
};
use flexmesa_rates::ScreeningMode;
use flexmesa_star::micro::{do_eos, do_net, resolve_model_net, set_basic_vars};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

fn build_single_zone_model(
    net_name: &str,
    log_t: f64,
    log_rho: f64,
    x_by_iso: &[(&str, f64)],
) -> (StarModel, std::sync::Arc<NetGeneralInfo>) {
    let g = NetGeneralInfo::from_name(net_name).expect("load net");
    let catalog = NuclideCatalog::builtin_ref();
    let job = StarJobControls {
        net_name: String::from(net_name),
        ..Default::default()
    };
    let controls = StarControls {
        initial_z: 0.02,
        zbase: 0.02,
        ..Default::default()
    };
    let mut model = StarModel::new(1, g.chem_id.clone(), job, controls);
    model.eos_config = EosConfig::default();
    model.lnd[0] = log_rho * LN10;
    model.ln_t[0] = log_t * LN10;
    model.ln_r[0] = 20.0;
    model.dq[0] = 1.0;
    model.mstar = 1.988_409_870_698_051e33;
    model.xmstar = model.mstar;
    model.dm[0] = model.mstar;
    model.xa.fill(1.0e-20);
    let mut sum_x = 0.0_f64;
    for &(name, val) in x_by_iso {
        let pos = g
            .chem_id
            .iter()
            .position(|&cid| {
                catalog.get_by_id(cid).is_some_and(|n| n.name == name)
            })
            .expect("isotope in network");
        model.xa[pos] = val;
        sum_x += val;
    }
    let scale = 1.0 / (sum_x + model.xa.iter().sum::<f64>() - sum_x);
    for v in &mut model.xa {
        *v *= scale;
    }
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");
    let net_info = resolve_model_net(&model).expect("resolve_model_net");
    (model, net_info)
}

fn eval_net_for_model(g: &NetGeneralInfo, model: &StarModel) -> NetEvalResult {
    let rf = vec![1.0_f64; g.num_reactions];
    eval_net(
        g,
        false,
        false,
        &model.xa,
        model.t[0],
        model.ln_t[0] / LN10,
        model.rho[0],
        model.lnd[0] / LN10,
        model.abar[0],
        model.zbar[0],
        model.z2bar[0],
        model.ye[0],
        model.eta[0],
        model.eos_d_dlnt[0][I_ETA],
        model.eos_d_dlnd[0][I_ETA],
        &rf,
        model.controls.weak_rate_factor,
        model.controls.screening_mode,
        model.job.use_3a_fl87,
        model.job.use_suzuki_weak_rates,
        false,
        false,
    )
    .expect("eval_net")
}

struct WorkEvalOut {
    work: NetWorkState,
    dydt: Vec<[Quad; NUM_RVS]>,
    scalars: (f64, f64, f64, f64),
}

fn eval_net_with_work_for_model(
    g: &NetGeneralInfo,
    model: &StarModel,
) -> WorkEvalOut {
    let rf = vec![1.0_f64; g.num_reactions];
    let ni = g.num_isos;
    let mut w = NetWorkState::new(g);
    let mut dydt = vec![[Quad::ONE; NUM_RVS]; ni];
    let out = eval_net_with_work(
        g,
        &mut w,
        &mut dydt,
        false,
        &model.xa,
        model.t[0],
        model.ln_t[0] / LN10,
        model.rho[0],
        model.lnd[0] / LN10,
        model.abar[0],
        model.zbar[0],
        model.z2bar[0],
        model.ye[0],
        model.eta[0],
        model.eos_d_dlnt[0][I_ETA],
        model.eos_d_dlnd[0][I_ETA],
        &rf,
        model.controls.weak_rate_factor,
        model.controls.screening_mode,
        model.job.use_3a_fl87,
        model.job.use_suzuki_weak_rates,
        false,
    )
    .expect("eval_net_with_work");
    WorkEvalOut {
        work: w,
        dydt,
        scalars: out,
    }
}

fn assert_direct_cgs_outputs_match(
    net_name: &str,
    g: &NetGeneralInfo,
    w: &NetWorkState,
    res: &NetEvalResult,
) {
    let ni = g.num_isos;
    match w.outputs() {
        NetWorkOutputs::DirectCgs {
            eps_nuc_categories,
            d_eps_nuc_dx,
            dxdt,
            d_dxdt_dt,
            d_dxdt_drho,
            d_dxdt_dx,
        } => {
            for c in 0..NUM_CATEGORIES {
                assert_eq!(
                    eps_nuc_categories[c].to_bits(),
                    res.eps_nuc_categories[c].to_bits(),
                    "{net_name} DirectCgs cat[{c}]"
                );
            }
            for i in 0..ni {
                assert_eq!(dxdt[i].to_bits(), res.dxdt[i].to_bits());
                assert_eq!(
                    d_eps_nuc_dx[i].to_bits(),
                    res.d_eps_nuc_dx[i].to_bits()
                );
                assert_eq!(d_dxdt_dt[i].to_bits(), res.d_dxdt_dt[i].to_bits());
                assert_eq!(
                    d_dxdt_drho[i].to_bits(),
                    res.d_dxdt_drho[i].to_bits()
                );
                for j in 0..ni {
                    let k = i * ni + j;
                    assert_eq!(
                        d_dxdt_dx[k].to_bits(),
                        res.d_dxdt_dx[k].to_bits()
                    );
                }
            }
        }
        NetWorkOutputs::Molar { .. } => {
            panic!("{net_name} must populate NetWorkOutputs::DirectCgs");
        }
    }
}

fn verify_approx21_net_direct_export(net_name: &str) {
    let mix = [
        ("he4", 0.08),
        ("c12", 0.42),
        ("o16", 0.45),
        ("ne20", 0.03),
        ("mg24", 0.01),
        ("si28", 0.005),
        ("ni56", 0.005),
    ];
    let (mut model, g) = build_single_zone_model(net_name, 8.95, 7.60, &mix);
    assert!(g.doing_approx21);
    do_net(&mut model, &g, false).expect("do_net");

    let res = eval_net_for_model(&g, &model);
    let eval_work = eval_net_with_work_for_model(&g, &model);
    let w = eval_work.work;
    let work_scalars = eval_work.scalars;
    assert_eq!(work_scalars.0.to_bits(), res.eps_nuc.to_bits());
    assert_eq!(work_scalars.1.to_bits(), res.d_eps_nuc_dt.to_bits());
    assert_eq!(work_scalars.2.to_bits(), res.d_eps_nuc_drho.to_bits());
    assert_eq!(work_scalars.3.to_bits(), res.eps_neu_total.to_bits());
    assert!(w.eps_neu_total.is_nan());
    assert!(eval_work.dydt.iter().flatten().all(|q| q.is_nan()));

    assert_eq!(model.eps_nuc[0].to_bits(), res.eps_nuc.to_bits());
    assert_eq!(
        model.eps_nuc_neu_total[0].to_bits(),
        res.eps_neu_total.to_bits()
    );
    assert_eq!(
        model.d_epsnuc_dlnd[0].to_bits(),
        (res.d_eps_nuc_drho * model.rho[0]).to_bits()
    );
    assert_eq!(
        model.d_epsnuc_dlnt[0].to_bits(),
        (res.d_eps_nuc_dt * model.t[0]).to_bits()
    );

    let (mut m_nojac, _) = build_single_zone_model(net_name, 8.95, 7.60, &mix);
    do_net(&mut m_nojac, &g, true).expect("do_net skip_jacobian");
    let cat_sum: f64 = res.eps_nuc_categories.iter().sum();
    let alfa = if cat_sum.abs() < 1.0e-10 {
        1.0
    } else {
        res.eps_nuc / cat_sum
    };
    for c in 0..NUM_CATEGORIES {
        let expected_cat = res.eps_nuc_categories[c] * alfa;
        assert_eq!(
            model.aux.eps_nuc_categories[0][c].to_bits(),
            expected_cat.to_bits(),
            "{net_name} cat {c} mismatch vs direct eval_net"
        );
        assert_eq!(
            m_nojac.aux.eps_nuc_categories[0][c].to_bits(),
            model.aux.eps_nuc_categories[0][c].to_bits(),
            "{net_name} cat {c} skip_jacobian mismatch"
        );
    }
    assert!(m_nojac.d_epsnuc_dx.iter().all(|v| v.to_bits() == 0));

    let ni = g.num_isos;
    for i in 0..ni {
        assert_eq!(
            model.dxdt_nuc[i].to_bits(),
            res.dxdt[i].to_bits(),
            "{net_name} dxdt[{i}]"
        );
        assert_eq!(
            model.d_epsnuc_dx[i].to_bits(),
            res.d_eps_nuc_dx[i].to_bits(),
            "{net_name} d_epsnuc_dx[{i}]"
        );
        assert_eq!(
            model.d_dxdt_nuc_dlnd[i].to_bits(),
            (res.d_dxdt_drho[i] * model.rho[0]).to_bits(),
            "{net_name} d_dxdt_nuc_dlnd[{i}]"
        );
        assert_eq!(
            model.d_dxdt_nuc_dlnt[i].to_bits(),
            (res.d_dxdt_dt[i] * model.t[0]).to_bits(),
            "{net_name} d_dxdt_nuc_dlnt[{i}]"
        );
        for j in 0..ni {
            assert_eq!(
                model.d_dxdt_nuc_dx[i * ni + j].to_bits(),
                res.d_dxdt_dx[i * ni + j].to_bits(),
                "{net_name} d_dxdt_nuc_dx[{i},{j}]"
            );
        }
    }
    assert_direct_cgs_outputs_match(net_name, &g, &w, &res);
}

/// Verifies `do_net` (`skip_jacobian` false and true) and `eval_net_with_work`
/// on `doing_approx21` networks match direct `eval_net` exports bit-for-bit,
/// select `NetWorkOutputs::DirectCgs`, and NaN-poison `dydt` and molar fields.
#[test]
fn test_approx21_do_net_matches_eval_net_direct_export_bitwise() {
    for net_name in ["approx21.net", "approx21_plus_co56.net"] {
        verify_approx21_net_direct_export(net_name);
    }
}

/// Verifies non-`approx21` `eval_net_with_work` selects `NetWorkOutputs::Molar`
/// with finite `dydt` and bitwise parity against `do_net` / `eval_net`.
#[test]
fn test_basic_net_eval_uses_molar_outputs_and_matches_eval_net() {
    let (mut basic_model, basic_net) = build_single_zone_model(
        "basic.net",
        8.4,
        4.2,
        &[("he4", 0.8), ("c12", 0.15), ("o16", 0.05)],
    );
    assert!(!basic_net.doing_approx21);
    do_net(&mut basic_model, &basic_net, false).expect("basic do_net");
    let res_basic = eval_net_for_model(&basic_net, &basic_model);
    for i in 0..basic_net.num_isos {
        assert_eq!(
            basic_model.d_epsnuc_dx[i].to_bits(),
            res_basic.d_eps_nuc_dx[i].to_bits()
        );
    }
    let eval_work = eval_net_with_work_for_model(&basic_net, &basic_model);
    assert!(eval_work.dydt.iter().flatten().all(|q| q.is_finite()));
    match eval_work.work.outputs() {
        NetWorkOutputs::Molar {
            eps_nuc_categories_mev,
            d_eps_nuc_dy,
            ..
        } => {
            for c in 0..NUM_CATEGORIES {
                assert_eq!(
                    (QCONV * eps_nuc_categories_mev[c]).to_bits(),
                    res_basic.eps_nuc_categories[c].to_bits()
                );
            }
            assert!(d_eps_nuc_dy.iter().all(|v| v.is_finite()));
        }
        NetWorkOutputs::DirectCgs { .. } => {
            panic!("basic.net must populate NetWorkOutputs::Molar");
        }
    }
}

#[derive(Clone, Copy)]
struct ZoneEvalTarget<'a> {
    lg_t: f64,
    lg_rho: f64,
    ending_x: &'a [f64],
}

fn assert_cat_sum_matches_eval_net(
    label: &str,
    cats: &[f64; NUM_CATEGORIES],
    g: &NetGeneralInfo,
    model: &mut StarModel,
    target: ZoneEvalTarget<'_>,
) {
    let ht = HelmTable::builtin().expect("builtin HELM");
    let eos_cfg = EosConfig::default();
    let rho = exp10(target.lg_rho);
    let temp = exp10(target.lg_t);
    let cat = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(cat, &g.chem_id, target.ending_x, false)
        .expect("composition");
    let eos = eos_dt_get(ht, rho, temp, &g.chem_id, target.ending_x, &eos_cfg)
        .expect("eos");
    model.xa.copy_from_slice(target.ending_x);
    model.rho[0] = rho;
    model.lnd[0] = target.lg_rho * LN10;
    model.t[0] = temp;
    model.ln_t[0] = target.lg_t * LN10;
    model.abar[0] = comp.abar;
    model.zbar[0] = comp.zbar;
    model.z2bar[0] = comp.z2bar;
    model.ye[0] = comp.ye;
    model.eta[0] = eos.res[I_ETA];
    let res = eval_net_for_model(g, model);
    let burn_cat_sum: f64 = cats.iter().sum();
    let ref_cat_sum: f64 = res.eps_nuc_categories.iter().sum();
    assert!(
        (burn_cat_sum - ref_cat_sum).abs() / ref_cat_sum.abs().max(1.0e-30)
            < 5.0e-2,
        "{label} cats sum {burn_cat_sum:.6e} vs CGS ref {ref_cat_sum:.6e}"
    );
}

/// Verifies `burn_1_zone` runs `approx21` with direct CGS
/// `eps_nuc_categories` (not unscaled `MeV`).
#[test]
fn test_approx21_burn_1_zone_direct_cgs() {
    let (mut model, g) = build_single_zone_model(
        "approx21.net",
        8.95,
        7.60,
        &[("he4", 0.08), ("c12", 0.42), ("o16", 0.45), ("ni56", 0.05)],
    );
    let ht = HelmTable::builtin().expect("builtin HELM");
    let eos_cfg = EosConfig::default();
    let rf = vec![1.0_f64; g.num_reactions];
    let lg_t = 8.95;
    let lg_rho = 7.60;
    let t_end = 1.0;

    let b1 = burn_1_zone(
        &g,
        ht,
        &eos_cfg,
        &Burn1ZoneInput {
            t_start: 0.0,
            t_end,
            starting_x: &model.xa,
            times: &[t_end],
            log10_ts: &[lg_t],
            log10_rhos: &[lg_rho],
            etas: &[model.eta[0]],
            dxdt_source_term: None,
            rate_factors: &rf,
            weak_rate_factor: 1.0,
            reaction_qs: &[],
            reaction_neu_qs: &[],
            screening_mode: model.controls.screening_mode,
            stptry_in: 0.0,
            max_steps: 500,
            eps: 1.0e-8,
            odescal: 1.0e-12,
        },
    )
    .expect("approx21 burn_1_zone");
    let b1_xsum: f64 = b1.ending_x.iter().sum();
    assert!(
        (b1_xsum - 1.0).abs() < 1.0e-9,
        "b1 sum={b1_xsum:.17e}, nstep={}, naccpt={}",
        b1.nstep,
        b1.naccpt,
    );
    assert!(b1.naccpt > 0 && b1.njac > 0);
    assert!(b1.avg_eps_nuc.is_finite() && b1.avg_eps_nuc > 0.0);
    assert!(b1.eps_nuc_categories.iter().all(|v| v.is_finite()));
    assert_cat_sum_matches_eval_net(
        "burn_1_zone",
        &b1.eps_nuc_categories,
        &g,
        &mut model,
        ZoneEvalTarget {
            lg_t,
            lg_rho,
            ending_x: &b1.ending_x,
        },
    );
}

/// Verifies `burn_1_zone_const_density` runs `approx21` with direct CGS
/// `eps_nuc_categories` (not unscaled `MeV`).
#[test]
fn test_approx21_const_density_direct_cgs() {
    let (mut model, g) = build_single_zone_model(
        "approx21.net",
        8.50,
        5.0,
        &[("he4", 0.08), ("c12", 0.42), ("o16", 0.45), ("ni56", 0.05)],
    );
    let ht = HelmTable::builtin().expect("builtin HELM");
    let eos_cfg = EosConfig::default();
    let rf = vec![1.0_f64; g.num_reactions];
    let lg_t_cd = 8.50;
    let lg_rho_cd = 5.0;
    let out = burn_1_zone_const_density(
        &g,
        ht,
        &eos_cfg,
        &BurnConstDensityInput {
            t_start: 0.0,
            t_end: 1.0,
            starting_x: &model.xa,
            starting_log10_t: lg_t_cd,
            log10_rho: lg_rho_cd,
            rate_factors: &rf,
            weak_rate_factor: 1.0,
            screening_mode: model.controls.screening_mode,
            stptry_in: 0.0,
            max_steps: 500,
            eps: 1.0e-8,
            odescal: 1.0e-12,
        },
    )
    .expect("approx21 burn_1_zone_const_density");
    let sum_end: f64 = out.ending_x.iter().sum();
    assert!((sum_end - 1.0).abs() < 1.0e-11);
    assert!(out.naccpt > 0 && out.njac > 0);
    assert!(out.avg_eps_nuc.is_finite() && out.avg_eps_nuc > 0.0);
    assert!(out.eps_nuc_categories.iter().all(|v| v.is_finite()));
    assert!(exp10(out.ending_log10_t).is_finite());
    assert_cat_sum_matches_eval_net(
        "const-density",
        &out.eps_nuc_categories,
        &g,
        &mut model,
        ZoneEvalTarget {
            lg_t: out.ending_log10_t,
            lg_rho: lg_rho_cd,
            ending_x: &out.ending_x,
        },
    );
}

#[derive(Clone, Copy)]
struct FrozenMoments {
    abar: f64,
    zbar: f64,
    z_sq_bar: f64,
    ye: f64,
    cv: f64,
    eta: f64,
    d_eta_dlnt: f64,
}

#[allow(clippy::too_many_arguments)]
fn eval_const_density_rhs(
    g: &NetGeneralInfo,
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    aion: &[f64],
    rf: &[f64],
    lg_rho: f64,
    y: &[f64],
    freeze_comp: Option<FrozenMoments>,
) -> (Vec<f64>, Vec<f64>, FrozenMoments) {
    let cat = NuclideCatalog::builtin_ref();
    let species = g.num_isos;
    let nvar = species + 1;
    let rho = exp10(lg_rho);
    let x_cur: Vec<f64> = (0..species).map(|i| y[i] * aion[i]).collect();
    let lg_t = y[species] / LN10;
    let temp = exp10(lg_t);
    let comp = get_composition_info(cat, &g.chem_id, &x_cur, false).unwrap();
    let eos = eos_dt_get(ht, rho, temp, &g.chem_id, &x_cur, eos_cfg).unwrap();
    let moments = freeze_comp.unwrap_or(FrozenMoments {
        abar: comp.abar,
        zbar: comp.zbar,
        z_sq_bar: comp.z2bar,
        ye: comp.ye,
        cv: eos.res[I_CV],
        eta: eos.res[I_ETA],
        d_eta_dlnt: eos.d_dln_t[I_ETA],
    });
    let d_cv_dlnt = eos.d_dln_t[I_CV];
    let mut dydt_buf = vec![[Quad::ZERO; NUM_RVS]; species];
    let mut w = NetWorkState::new(g);
    let (eps_nuc, d_eps_dt, _, _) = eval_net_with_work(
        g,
        &mut w,
        &mut dydt_buf,
        false,
        &x_cur,
        temp,
        lg_t,
        rho,
        lg_rho,
        moments.abar,
        moments.zbar,
        moments.z_sq_bar,
        moments.ye,
        moments.eta,
        moments.d_eta_dlnt,
        0.0,
        rf,
        1.0,
        ScreeningMode::Extended,
        false,
        false,
        false,
    )
    .unwrap();
    let mut f = vec![0.0; nvar];
    let mut dfdy = vec![0.0; nvar * nvar];
    let cv_temp = moments.cv * temp;
    match w.outputs() {
        NetWorkOutputs::DirectCgs {
            d_eps_nuc_dx,
            dxdt,
            d_dxdt_dx,
            ..
        } => {
            for j in 0..species {
                f[j] = dxdt[j] / aion[j];
            }
            f[species] = eps_nuc / cv_temp;
            for i in 0..species {
                let row_n = i * nvar;
                let row_s = i * species;
                let a_i = aion[i];
                for j in 0..species {
                    dfdy[row_n + j] = d_dxdt_dx[row_s + j] * aion[j] / a_i;
                }
                dfdy[row_n + species] = 0.0;
            }
            for j in 0..species {
                dfdy[species * nvar + j] = d_eps_nuc_dx[j] / cv_temp;
            }
            dfdy[species * nvar + species] = d_eps_dt / moments.cv
                - (1.0 + d_cv_dlnt / moments.cv) * eps_nuc / cv_temp;
        }
        NetWorkOutputs::Molar { .. } => {
            panic!("approx21.net must populate DirectCgs");
        }
    }
    (f, dfdy, moments)
}

fn assert_fd_entry_close(
    label: &str,
    analytic: f64,
    fd: f64,
    f_val: f64,
    step: f64,
) {
    let scale = analytic.abs().max(fd.abs());
    let floor = 32.0 * f64::EPSILON * f_val.abs() / step;
    assert!(
        (analytic - fd).abs() <= floor.max(1.0e-18)
            || (analytic - fd).abs() <= 1.0e-6 * scale,
        "{label}: analytic={analytic:.8e}, fd={fd:.8e}"
    );
}

struct CdFdContext<'a> {
    g: &'a NetGeneralInfo,
    ht: &'a HelmTable,
    eos_cfg: &'a EosConfig,
    aion: &'a [f64],
    rf: &'a [f64],
    lg_rho: f64,
    y_base: &'a [f64],
    f_base: &'a [f64],
    dfdy: &'a [f64],
    frozen: FrozenMoments,
}

fn check_cd_species_column(ctx: &CdFdContext<'_>, col: usize) {
    let species = ctx.g.num_isos;
    let nvar = species + 1;
    let h_y = 1.0e-6 * ctx.y_base[col].max(1.0e-3);
    let mut y_plus = ctx.y_base.to_vec();
    let mut y_minus = ctx.y_base.to_vec();
    y_plus[col] += h_y;
    y_minus[col] -= h_y;
    let (rhs_plus, _, _) = eval_const_density_rhs(
        ctx.g,
        ctx.ht,
        ctx.eos_cfg,
        ctx.aion,
        ctx.rf,
        ctx.lg_rho,
        &y_plus,
        Some(ctx.frozen),
    );
    let (rhs_minus, _, _) = eval_const_density_rhs(
        ctx.g,
        ctx.ht,
        ctx.eos_cfg,
        ctx.aion,
        ctx.rf,
        ctx.lg_rho,
        &y_minus,
        Some(ctx.frozen),
    );
    for i in 0..species {
        let fd_ij = (rhs_plus[i] - rhs_minus[i]) / (2.0 * h_y);
        assert_fd_entry_close(
            &format!("approx21 df[{i}]/dy[{col}]"),
            ctx.dfdy[i * nvar + col],
            fd_ij,
            ctx.f_base[i],
            h_y,
        );
    }
    // `net_burn_const_density.f90:482` leaves `dfdy(nvar, j)` wrt `X_j`.
    let h_x = h_y * ctx.aion[col];
    let fd_temp_x = (rhs_plus[species] - rhs_minus[species]) / (2.0 * h_x);
    assert_fd_entry_close(
        &format!("approx21 df_lnT/dX[{col}]"),
        ctx.dfdy[species * nvar + col],
        fd_temp_x,
        ctx.f_base[species],
        h_x,
    );
}

/// Verifies `approx21` const-density ODE RHS and Jacobian from
/// `NetWorkOutputs::DirectCgs` match frozen-moment central finite differences.
#[test]
fn test_approx21_const_density_jacobian_matches_central_fd() {
    let g = NetGeneralInfo::from_name("approx21.net").expect("approx21.net");
    let ht = HelmTable::builtin().expect("builtin HELM");
    let eos_cfg = EosConfig::default();
    let cat = NuclideCatalog::builtin_ref();
    let species = g.num_isos;
    let nvar = species + 1;

    let aion: Vec<f64> = g
        .chem_id
        .iter()
        .map(|&cid| {
            f64::from(
                u32::try_from(cat.get_by_id(cid).unwrap().z_plus_n)
                    .expect("z_plus_n"),
            )
        })
        .collect();
    let idx_of = |name: &str| {
        g.chem_id
            .iter()
            .position(|&c| cat.get_by_id(c).is_some_and(|n| n.name == name))
            .expect("isotope in approx21")
    };
    let idx_helium = idx_of("he4");
    let idx_carbon = idx_of("c12");
    let idx_oxygen = idx_of("o16");
    let mut x0 = vec![1.0e-20_f64; species];
    x0[idx_helium] = 0.08;
    x0[idx_carbon] = 0.42;
    x0[idx_oxygen] = 0.45;
    x0[idx_of("ni56")] = 0.05;
    let sx: f64 = x0.iter().sum();
    for v in &mut x0 {
        *v /= sx;
    }

    let lg_rho = 7.60;
    let lg_t0 = 8.95;
    let rf = vec![1.0_f64; g.num_reactions];
    let mut y_base: Vec<f64> = (0..species).map(|i| x0[i] / aion[i]).collect();
    y_base.push(lg_t0 * LN10);

    let (f_base, dfdy, frozen) = eval_const_density_rhs(
        &g, ht, &eos_cfg, &aion, &rf, lg_rho, &y_base, None,
    );
    let h_lnt = 1.0e-6;
    let mut y_plus_t = y_base.clone();
    let mut y_minus_t = y_base.clone();
    y_plus_t[species] += h_lnt;
    y_minus_t[species] -= h_lnt;
    let (rhs_plus_t, _, _) = eval_const_density_rhs(
        &g, ht, &eos_cfg, &aion, &rf, lg_rho, &y_plus_t, None,
    );
    let (rhs_minus_t, _, _) = eval_const_density_rhs(
        &g, ht, &eos_cfg, &aion, &rf, lg_rho, &y_minus_t, None,
    );
    let fd_temp_diag =
        (rhs_plus_t[species] - rhs_minus_t[species]) / (2.0 * h_lnt);
    let an_temp_diag = dfdy[species * nvar + species];
    assert!(
        (an_temp_diag - fd_temp_diag).abs()
            / an_temp_diag.abs().max(fd_temp_diag.abs())
            <= 1.0e-5,
        "approx21 df_lnT/dlnT: \
         analytic={an_temp_diag:.8e}, fd={fd_temp_diag:.8e}"
    );

    let ctx = CdFdContext {
        g: &g,
        ht,
        eos_cfg: &eos_cfg,
        aion: &aion,
        rf: &rf,
        lg_rho,
        y_base: &y_base,
        f_base: &f_base,
        dfdy: &dfdy,
        frozen,
    };
    for &col in &[idx_helium, idx_carbon, idx_oxygen] {
        check_cd_species_column(&ctx, col);
    }
}
