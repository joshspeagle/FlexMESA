// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` integration tests for `#161` Group 7a
//! `burn_1_zone` and `simpr` parity (`net/private/net_burn.f90:418-421,449,461`,
//! `net/private/net_burn_support.f90:515,529,549`,
//! `net/private/net_eval.f90:274,566,570-571`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_eos::{EosConfig, HelmTable};
use flexmesa_net::{burn_1_zone, Burn1ZoneInput, NetGeneralInfo};
use flexmesa_rates::ScreeningMode;

fn initial_comp(g: &NetGeneralInfo, weights: &[(&str, f64)]) -> Vec<f64> {
    let cat = NuclideCatalog::builtin_ref();
    let mut x = vec![0.0_f64; g.num_isos];
    for &(name, val) in weights {
        let cid = cat
            .get_nuclide_index(name)
            .unwrap_or_else(|| panic!("unknown nuclide {name}"));
        let slot = g.net_iso[cid];
        assert!(slot > 0, "nuclide {name} absent from {}", g.net_filename);
        x[slot - 1] = val;
    }
    x
}

fn run_constant_burn(
    net_name: &str,
    weights: &[(&str, f64)],
    lt: f64,
    lr: f64,
    dt: f64,
) -> (Vec<f64>, f64, usize) {
    let g = NetGeneralInfo::from_name(net_name).expect("load net");
    let ht = HelmTable::builtin().expect("builtin helm");
    let eos_cfg = EosConfig::default();
    let x0 = initial_comp(&g, weights);
    let rf = vec![1.0_f64; g.num_reactions];
    let times = [0.0_f64];
    let lts = [lt];
    let lrs = [lr];
    let etas = [0.0_f64];
    let out = burn_1_zone(
        &g,
        ht,
        &eos_cfg,
        &Burn1ZoneInput {
            t_start: 0.0,
            t_end: dt,
            starting_x: &x0,
            times: &times,
            log10_ts: &lts,
            log10_rhos: &lrs,
            etas: &etas,
            dxdt_source_term: None,
            rate_factors: &rf,
            weak_rate_factor: 1.0,
            reaction_qs: &[],
            reaction_neu_qs: &[],
            screening_mode: ScreeningMode::Chugunov,
            stptry_in: 0.0,
            max_steps: 50_000,
            eps: 1.0e-8,
            odescal: 1.0e-12,
        },
    )
    .expect("burn_1_zone succeeds");
    (out.ending_x, out.avg_eps_nuc, out.nstep)
}

#[test]
fn test_burn_1_zone_general_net_conserves_mass_and_is_deterministic() {
    let weights = [
        ("he4", 0.02),
        ("c12", 0.45),
        ("o16", 0.48),
        ("ne20", 0.03),
        ("mg24", 0.02),
    ];
    let (x1, eps1, nstep1) =
        run_constant_burn("co_burn.net", &weights, 9.05, 6.50, 1.0);
    let (x2, eps2, nstep2) =
        run_constant_burn("co_burn.net", &weights, 9.05, 6.50, 1.0);
    assert_eq!(nstep1, nstep2);
    assert!(nstep1 > 0);
    assert_eq!(eps1.to_bits(), eps2.to_bits());
    for (a, b) in x1.iter().zip(&x2) {
        assert_eq!(a.to_bits(), b.to_bits());
        assert!(a.is_finite());
    }
    let sum_x: f64 = x1.iter().sum();
    assert!(
        (sum_x - 1.0).abs() < 1.0e-8,
        "mass fraction conservation failed: sum_x={sum_x}"
    );
}

#[test]
fn test_burn_1_zone_mesa_80_trace_species_aion_roundtrip_stable() {
    let weights = [
        ("he4", 0.05),
        ("o16", 0.35),
        ("ne20", 0.10),
        ("mg24", 0.15),
        ("si28", 0.30),
        ("s32", 0.05),
    ];
    let (x1, eps1, nstep1) =
        run_constant_burn("mesa_80.net", &weights, 9.15, 6.20, 1.0e-3);
    let (x2, eps2, _) =
        run_constant_burn("mesa_80.net", &weights, 9.15, 6.20, 1.0e-3);
    assert!(nstep1 > 0);
    assert!(eps1.is_finite());
    assert_eq!(eps1.to_bits(), eps2.to_bits());
    assert_eq!(x1.len(), 80);
    let sum_x: f64 = x1.iter().sum();
    assert!((sum_x - 1.0).abs() < 1.0e-8, "sum_x={sum_x}");
    for (a, b) in x1.iter().zip(&x2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn test_burn_1_zone_approx21_high_t_burn_preserves_mass_and_determinism() {
    let weights = [
        ("o16", 0.55),
        ("ne20", 0.05),
        ("mg24", 0.10),
        ("si28", 0.25),
        ("s32", 0.05),
    ];
    let (x1, eps1, nstep1) = run_constant_burn(
        "approx21_cr60_plus_co56.net",
        &weights,
        9.38,
        7.10,
        1.0e-2,
    );
    let (x2, eps2, nstep2) = run_constant_burn(
        "approx21_cr60_plus_co56.net",
        &weights,
        9.38,
        7.10,
        1.0e-2,
    );
    assert_eq!(nstep1, nstep2);
    assert!(nstep1 > 0);
    assert!(eps1.is_finite());
    assert_eq!(eps1.to_bits(), eps2.to_bits());
    let sum_x: f64 = x1.iter().sum();
    assert!((sum_x - 1.0).abs() < 1.0e-8, "sum_x={sum_x}");
    assert_eq!(x1.len(), x2.len());
    for (a, b) in x1.iter().zip(&x2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
