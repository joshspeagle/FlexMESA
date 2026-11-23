// SPDX-License-Identifier: LGPL-3.0-or-later
//! MESA `26.4.1` Model-1 Zone-7 `equ_he3` bit-parity tests (`#450`).
//!
//! Reproduces `star/1.3M_ms_high_Z` Model 1 iteration 1 radiative zones (`z=7`
//! and `z=700`, where `dxdt_mix == 0`, `dxdt_actual == 0`, and
//! `eqn_scale == min_chem_eqn_scale == 1.0`, so `equ_chem == dxdt_nuc`)
//! against MESA `26.4.1` IEEE-754 hex stepdumps.
//!
//! Verifies that `do_net` (`micro_net.rs`) multiplies `Z_plus_N * dydt` in
//! `Quad` (`real(qp)`) before rounding to `f64` and computes
//! `d_dydt_dy * a_i / a_j` left-to-right, matching `net_eval.f90:274,565-573`
//! (`store_partials`) and `eval_net`, rather than rounding `dydt.to_f64()`
//! before multiplying by `Z_plus_N`.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::LN10;
use flexmesa_eos::I_ETA;
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_net::net_get;
use flexmesa_star::micro::{do_eos, do_net, resolve_model_net, set_basic_vars};
use flexmesa_star::{
    compute_x_scale, do1_chem_eqns, HenyeySystem, StarControls,
    StarJobControls, StarModel,
};

fn hex(bits: u64) -> f64 {
    f64::from_bits(bits)
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -((bits ^ (1_u64 << 63)) as i128)
    } else {
        bits as i128
    }
}

fn ulp_distance(actual: f64, expected: f64) -> u64 {
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected.to_bits());
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

const BASIC_SPECIES: &[&str] =
    &["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];

struct ZoneOracle<'a> {
    label: &'a str,
    lnd: u64,
    lnt: u64,
    lnr: u64,
    dt: u64,
    dm: u64,
    xa_hex: &'a [u64; 8],
    want_peos_hex: u64,
    want_cp_hex: u64,
    want_eps_nuc_hex: u64,
    want_equ_chem_hex: &'a [u64; 8],
}

fn build_zone_model(zone: &ZoneOracle<'_>) -> StarModel {
    let catalog = NuclideCatalog::builtin_ref();
    let chem_id: Vec<usize> = BASIC_SPECIES
        .iter()
        .map(|name| catalog.get_nuclide_index(name).expect(name))
        .collect();
    let job = StarJobControls {
        net_name: "basic.net".to_string(),
        ..Default::default()
    };
    let ctrl = StarControls::default();

    let mut model = StarModel::new(1, chem_id, job, ctrl);
    model.lnd[0] = hex(zone.lnd);
    model.ln_t[0] = hex(zone.lnt);
    model.ln_r[0] = hex(zone.lnr);
    model.dt = hex(zone.dt);
    model.dq[0] = 1.0;
    model.mstar = 2.0e33;
    model.xmstar = 2.0e33;
    model.dm[0] = hex(zone.dm);
    model.xa.copy_from_slice(&zone.xa_hex.map(hex));
    model.save_start_values().expect("save_start_values");
    model
}

fn assert_zone_dxdt_and_equ_zero_ulp(zone: &ZoneOracle<'_>) {
    let mut model = build_zone_model(zone);
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");

    assert_eq!(
        ulp_distance(model.peos[0], hex(zone.want_peos_hex)),
        0,
        "{} peos mismatch: got=0x{:016X} want=0x{:016X}",
        zone.label,
        model.peos[0].to_bits(),
        zone.want_peos_hex,
    );
    assert_eq!(
        ulp_distance(model.cp[0], hex(zone.want_cp_hex)),
        0,
        "{} cp mismatch: got=0x{:016X} want=0x{:016X}",
        zone.label,
        model.cp[0].to_bits(),
        zone.want_cp_hex,
    );

    let net_info = resolve_model_net(&model).expect("resolve_model_net");
    do_net(&mut model, &net_info, false).expect("do_net");

    assert_eq!(
        ulp_distance(model.eps_nuc[0], hex(zone.want_eps_nuc_hex)),
        0,
        "{} eps_nuc mismatch: got=0x{:016X} want=0x{:016X}",
        zone.label,
        model.eps_nuc[0].to_bits(),
        zone.want_eps_nuc_hex,
    );

    let x_scale = compute_x_scale(&model, true);
    let mut sys = HenyeySystem::new(1, model.layout.nvar(true), x_scale);
    do1_chem_eqns(&model, 0, &mut sys);

    for (j, (&name, &want_bits)) in BASIC_SPECIES
        .iter()
        .zip(zone.want_equ_chem_hex.iter())
        .enumerate()
    {
        let want = hex(want_bits);
        let got_dxdt = model.dxdt_nuc[j];
        let ulp_dxdt = ulp_distance(got_dxdt, want);
        assert_eq!(
            ulp_dxdt,
            0,
            "{} dxdt_nuc[{}] mismatch: got=0x{:016X} ({:.17e}) want=0x{:016X} ({:.17e}) ulp={}",
            zone.label,
            name,
            got_dxdt.to_bits(),
            got_dxdt,
            want_bits,
            want,
            ulp_dxdt,
        );

        let i_equ = model.layout.chem_var_index(j);
        let got_equ = sys.equ[i_equ];
        let ulp_equ = ulp_distance(got_equ, want);
        assert_eq!(
            ulp_equ,
            0,
            "{} equ_{} mismatch: got=0x{:016X} ({:.17e}) want=0x{:016X} ({:.17e}) ulp={}",
            zone.label,
            name,
            got_equ.to_bits(),
            got_equ,
            want_bits,
            want,
            ulp_equ,
        );
    }

    // Verify `do_net` matches `net_get` (`eval_net`) at 0 ULP across all
    // abundance derivatives and Jacobian entries.
    let rate_factors = vec![1.0_f64; net_info.num_reactions];
    let res = net_get(
        &net_info,
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
        &rate_factors,
        model.controls.weak_rate_factor,
        model.controls.screening_mode,
    )
    .expect("net_get");

    let sp = model.species;
    for (i, &name) in BASIC_SPECIES.iter().enumerate() {
        assert_eq!(
            model.dxdt_nuc[i].to_bits(),
            res.dxdt[i].to_bits(),
            "{} do_net vs net_get dxdt[{}]",
            zone.label,
            name,
        );
        assert_eq!(
            model.d_epsnuc_dx[i].to_bits(),
            res.d_eps_nuc_dx[i].to_bits(),
            "{} do_net vs net_get d_epsnuc_dx[{}]",
            zone.label,
            name,
        );
        assert_eq!(
            model.d_dxdt_nuc_dlnd[i].to_bits(),
            (res.d_dxdt_drho[i] * model.rho[0]).to_bits(),
            "{} do_net vs net_get d_dxdt_nuc_dlnd[{}]",
            zone.label,
            name,
        );
        assert_eq!(
            model.d_dxdt_nuc_dlnt[i].to_bits(),
            (res.d_dxdt_dt[i] * model.t[0]).to_bits(),
            "{} do_net vs net_get d_dxdt_nuc_dlnt[{}]",
            zone.label,
            name,
        );
        for (j, &col_name) in BASIC_SPECIES.iter().enumerate() {
            assert_eq!(
                model.d_dxdt_nuc_dx[i * sp + j].to_bits(),
                res.d_dxdt_dx[i * sp + j].to_bits(),
                "{} do_net vs net_get d_dxdt_nuc_dx[{}, {}]",
                zone.label,
                name,
                col_name,
            );
        }
    }
}

#[test]
fn test_1p3m_ms_high_z_model1_zone7_equ_he3_zero_ulp() {
    let zone7 = ZoneOracle {
        label: "star/1.3M_ms_high_Z m=1 z=7",
        lnd: 0xC02FD575C7046FB6,
        lnt: 0x40217CF5B658ACFA,
        lnr: 0x40393A91352A93B1,
        dt: 0x42C5EF3ED89CD52F,
        dm: 0x448B5E797788EFC4,
        xa_hex: &[
            0x3FE47AE147AE147C,
            0x3F01DAB5A8957502,
            0x3FD47A527200CFD0,
            0x3F7C3CA369D8B430,
            0x3F608DAF41EC844F,
            0x3F9336191BA4210F,
            0x3F71149CF6C7EBFF,
            0x3F80B346E6A49EB9,
        ],
        want_peos_hex: 0x40E65AF87DE141DE,
        want_cp_hex: 0x41A32AE08E5D1DC8,
        want_eps_nuc_hex: 0x3A92913F9208ACA4,
        want_equ_chem_hex: &[
            0xB6C3F60FF9F3C74C,
            0x36C3F60FF9F3C74C,
            0x321876493C598441,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        assert_zone_dxdt_and_equ_zero_ulp(&zone7);
    });
}

#[test]
fn test_1p3m_ms_high_z_model1_zone700_n14_ne20_zero_ulp() {
    let zone700 = ZoneOracle {
        label: "star/1.3M_ms_high_Z m=1 z=700",
        lnd: 0x400AD41E6CF446FD,
        lnt: 0x4030277D245770ED,
        lnr: 0x4037826342922CBF,
        dt: 0x42C5EF3ED89CD52F,
        dm: 0x466A859DD37E7B20,
        xa_hex: &[
            0x3FE47A771DBCEC81,
            0x3F1588CAE2BECA3C,
            0x3FD47A5A142725C9,
            0x3F7C382B512FB01B,
            0x3F60981CD0EA6285,
            0x3F9336191B9FD1E6,
            0x3F71149CF6C7EBFE,
            0x3F80B346E6A49EB8,
        ],
        want_peos_hex: 0x4360C1D4A56B14D5,
        want_cp_hex: 0x41B331998B7E496D,
        want_eps_nuc_hex: 0x3FE7A0D68B0BE903,
        want_equ_chem_hex: &[
            0xBC15AB1464DFE24A,
            0x3C13A20587CFBA3C,
            0x3BD91745F2C8DC54,
            0xBBE66D0DE7AF7D0B,
            0x3BEA29FF92CFB91D,
            0xBAEDBC0369C95591,
            0x32C121A0DD3A003F,
            0x2F11ED1E1961A38E,
        ],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        assert_zone_dxdt_and_equ_zero_ulp(&zone700);
    });
}

#[test]
fn test_do_net_dxdt_nuc_factor_order_matches_mesa_net() {
    let zone700 = ZoneOracle {
        label: "star/1.3M_ms_high_Z m=1 z=700 scaled",
        lnd: 0x400AD41E6CF446FD,
        lnt: 0x4030277D245770ED,
        lnr: 0x4037826342922CBF,
        dt: 0x42C5EF3ED89CD52F,
        dm: 0x466A859DD37E7B20,
        xa_hex: &[
            0x3FE47A771DBCEC81,
            0x3F1588CAE2BECA3C,
            0x3FD47A5A142725C9,
            0x3F7C382B512FB01B,
            0x3F60981CD0EA6285,
            0x3F9336191B9FD1E6,
            0x3F71149CF6C7EBFE,
            0x3F80B346E6A49EB8,
        ],
        want_peos_hex: 0x4360C1D4A56B14D5,
        want_cp_hex: 0x41B331998B7E496D,
        want_eps_nuc_hex: 0x3FE7A0D68B0BE903,
        want_equ_chem_hex: &[0; 8],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        let mut model = build_zone_model(&zone700);
        let dxdt_fac = 0.37_f64;
        model.controls.dxdt_nuc_factor = dxdt_fac;
        set_basic_vars(&mut model).expect("set_basic_vars");
        do_eos(&mut model).expect("do_eos");
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        do_net(&mut model, &net_info, false).expect("do_net");

        let rate_factors = vec![1.0_f64; net_info.num_reactions];
        let res = net_get(
            &net_info,
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
            &rate_factors,
            model.controls.weak_rate_factor,
            model.controls.screening_mode,
        )
        .expect("net_get");

        let sp = model.species;
        for (i, &name) in BASIC_SPECIES.iter().enumerate() {
            assert_eq!(
                model.dxdt_nuc[i].to_bits(),
                (res.dxdt[i] * dxdt_fac).to_bits(),
                "scaled dxdt_nuc[{name}]",
            );
            assert_eq!(
                model.d_dxdt_nuc_dlnd[i].to_bits(),
                ((res.d_dxdt_drho[i] * dxdt_fac) * model.rho[0]).to_bits(),
                "scaled d_dxdt_nuc_dlnd[{name}]",
            );
            assert_eq!(
                model.d_dxdt_nuc_dlnt[i].to_bits(),
                ((res.d_dxdt_dt[i] * dxdt_fac) * model.t[0]).to_bits(),
                "scaled d_dxdt_nuc_dlnt[{name}]",
            );
            for (j, &col_name) in BASIC_SPECIES.iter().enumerate() {
                assert_eq!(
                    model.d_dxdt_nuc_dx[i * sp + j].to_bits(),
                    (res.d_dxdt_dx[i * sp + j] * dxdt_fac).to_bits(),
                    "scaled d_dxdt_nuc_dx[{name}, {col_name}]",
                );
            }
        }
    });
}
