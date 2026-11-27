// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant tests verifying direct CGS / mass-fraction export for
//! `doing_approx21` networks (`net/private/net_eval.f90:232-237,308-338,431`,
//! `net/private/net_burn_const_density.f90:482`) without `QCONV` or `aion`
//! round-trips.

#![forbid(unsafe_code)]

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{LN10, QCONV, SECYER};
use flexmesa_eos::EosConfig;
use flexmesa_math::exp10;
use flexmesa_net::{
    burn_1_zone_const_density, eval_net, BurnConstDensityInput, NetEvalResult,
    NetGeneralInfo,
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
        model.eos_d_dlnt[0][flexmesa_eos::I_ETA],
        model.eos_d_dlnd[0][flexmesa_eos::I_ETA],
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

fn verify_approx21_net_direct_export(net_name: &str) -> bool {
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

    let mut saw_roundtrip_ulp_diff = false;
    let catalog = NuclideCatalog::builtin_ref();
    let ni = g.num_isos;
    for i in 0..ni {
        let zn =
            u32::try_from(catalog.get_by_id(g.chem_id[i]).unwrap().z_plus_n)
                .expect("z_plus_n fits u32");
        let a_i = f64::from(zn);
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
        let round_tripped = QCONV * (res.d_eps_nuc_dx[i] * a_i / QCONV) / a_i;
        if round_tripped.to_bits() != res.d_eps_nuc_dx[i].to_bits() {
            saw_roundtrip_ulp_diff = true;
        }
        for j in 0..ni {
            assert_eq!(
                model.d_dxdt_nuc_dx[i * ni + j].to_bits(),
                res.d_dxdt_dx[i * ni + j].to_bits(),
                "{net_name} d_dxdt_nuc_dx[{i},{j}]"
            );
        }
    }
    saw_roundtrip_ulp_diff
}

/// Verifies `do_net` (`skip_jacobian` false and true) on `doing_approx21`
/// networks matches direct `eval_net` exports bit-for-bit (0 ULP), avoiding
/// the `a_i / QCONV` and `QCONV * / a_i` round-trip (`>= 1` ULP shift).
#[test]
fn test_approx21_do_net_matches_eval_net_direct_export_bitwise() {
    let mut saw_roundtrip_ulp_diff = false;
    for net_name in ["approx21.net", "approx21_plus_co56.net"] {
        if verify_approx21_net_direct_export(net_name) {
            saw_roundtrip_ulp_diff = true;
        }
    }
    assert!(
        saw_roundtrip_ulp_diff,
        "expected QCONV/a_i round-trip to introduce >= 1 ULP drift"
    );
}

/// Verifies `burn_1_zone_const_density` executes cleanly on `approx21.net`
/// while non-approx21 `do_net` continues to match `eval_net` bit-for-bit.
#[test]
fn test_approx21_const_density_burn_and_basic_net_invariance() {
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

    let (model, g) = build_single_zone_model(
        "approx21.net",
        8.95,
        7.60,
        &[("he4", 0.08), ("c12", 0.42), ("o16", 0.45), ("ni56", 0.05)],
    );
    let ht = flexmesa_eos::HelmTable::builtin().expect("builtin HELM");
    let eos_cfg = EosConfig::default();
    let rf = vec![1.0_f64; g.num_reactions];
    let lg_t = 8.95;
    let lg_rho = 7.60;
    let out = burn_1_zone_const_density(
        &g,
        ht,
        &eos_cfg,
        &BurnConstDensityInput {
            t_start: 0.0,
            t_end: 1.0e-3 * SECYER,
            starting_x: &model.xa,
            starting_log10_t: lg_t,
            log10_rho: lg_rho,
            rate_factors: &rf,
            weak_rate_factor: 1.0,
            screening_mode: ScreeningMode::Extended,
            stptry_in: 1.0e-4,
            max_steps: 500,
            eps: 1.0e-7,
            odescal: 1.0e-8,
        },
    )
    .expect("approx21 burn_1_zone_const_density");
    let sum_end: f64 = out.ending_x.iter().sum();
    assert!((sum_end - 1.0).abs() < 1.0e-11);
    assert!(out.naccpt > 0);
    assert!(out.avg_eps_nuc.is_finite() && out.avg_eps_nuc > 0.0);
    assert!(exp10(out.ending_log10_t).is_finite());
}
