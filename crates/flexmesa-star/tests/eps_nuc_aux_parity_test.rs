// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` AUX step-dump parity tests for nuclear `eps_nuc` (`#394`).
//!
//! Verifies exact IEEE-754 bit-level (`0 ULP`) parity in `NumericsMode::Strict`
//! against MESA `26.4.1` `AUX` step dumps for:
//! - `astero/surface_effects` zone 1 (`pp_and_cno_extras.net`, `8.29e15 ULP` -> `0 ULP`):
//!   `r_be7_wk_li7` (`id=251`, predefined in `reactions.list:334`) resolves `weak_lowT_rate`
//!   during `read_reaction_parameters` (`rates/private/rates_initialize.f90:807-810`) before
//!   `finish_rates_def_init` (`rates_initialize.f90:1256-1262`) extracts `reaclib_rates`, so
//!   `do_reaclib_indices_for_reaction` fails (`reaclib_eval.f90:249-252`) and falls back to
//!   `weak_info.list:17` (`ln2 / 4.604256d6 = 0x3E8434AECB5CCB90`).
//! - `star/5M_cepheid_blue_loop` zone 1 (`pp_cno_extras_o18_ne22.net`, `3.56e7 ULP` -> `0 ULP`):
//!   same `r_be7_wk_li7` `weak_lowT_rate` lookup path.
//! - `star/1M_thermohaline` zone 82 (`basic.net`, `2 ULP` -> `0 ULP`) and
//!   `star/irradiated_planet` zone 3 (`basic.net`, `2 ULP` -> `0 ULP`):
//!   low-T clipping (`net/private/net_eval.f90:146-149`) uses `rattab_temp_lo = exp10(rattab_tlo)`
//!   (`rates/public/rates_def.f90:1141,1166`, `0x410869FFFFFFFFE9`) rather than exact `2.0e5`
//!   (`0x41086A0000000000`), shifting Chugunov screening on `rpp_to_he3` by 1-2 ULP.

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
use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};
use flexmesa_net::{
    eval_net_with_work, with_thread_local_work, Quad, NUM_RVS, RATTAB_TEMP_LO,
    RATTAB_TLO,
};
use flexmesa_star::micro::{do_eos, do_net, resolve_model_net, set_basic_vars};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

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

struct CaseInput<'a> {
    label: &'a str,
    net_name: &'a str,
    lnd: u64,
    lnt: u64,
    lnr: u64,
    species_names: &'a [&'a str],
    xa_hex: &'a [u64],
    want_eps_nuc_hex: u64,
    want_peos_hex: u64,
    want_cp_hex: u64,
    reaction_assertions: &'a [(&'a str, u64)],
}

fn assert_aux_case(case: &CaseInput<'_>) {
    let catalog = NuclideCatalog::builtin_ref();
    let chem_id: Vec<usize> = case
        .species_names
        .iter()
        .map(|name| catalog.get_nuclide_index(name).expect(name))
        .collect();
    let job = StarJobControls {
        net_name: case.net_name.to_string(),
        ..Default::default()
    };
    let ctrl = StarControls::default();

    let mut model = StarModel::new(1, chem_id, job, ctrl);
    model.lnd[0] = hex(case.lnd);
    model.ln_t[0] = hex(case.lnt);
    model.ln_r[0] = hex(case.lnr);
    model.dq[0] = 1.0;
    model.mstar = 2.0e33;
    model.xmstar = 2.0e33;
    model.dm[0] = 2.0e33;
    for (i, &h) in case.xa_hex.iter().enumerate() {
        model.xa[i] = hex(h);
    }

    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");

    assert_eq!(
        ulp_distance(model.peos[0], hex(case.want_peos_hex)),
        0,
        "{} peos mismatch: got=0x{:016X} want=0x{:016X}",
        case.label,
        model.peos[0].to_bits(),
        case.want_peos_hex,
    );
    assert_eq!(
        ulp_distance(model.cp[0], hex(case.want_cp_hex)),
        0,
        "{} cp mismatch: got=0x{:016X} want=0x{:016X}",
        case.label,
        model.cp[0].to_bits(),
        case.want_cp_hex,
    );

    let net_info = resolve_model_net(&model).expect("resolve_model_net");
    do_net(&mut model, &net_info, false).expect("do_net");

    let got = model.eps_nuc[0];
    let want = hex(case.want_eps_nuc_hex);
    let ulp = ulp_distance(got, want);
    assert_eq!(
        ulp,
        0,
        "{} eps_nuc mismatch: got=0x{:016X} ({:.17e}) want=0x{:016X} ({:.17e}) ulp={}",
        case.label,
        got.to_bits(),
        got,
        want.to_bits(),
        want,
        ulp,
    );

    let rf = vec![1.0_f64; net_info.num_reactions];
    let mut dydt = vec![[Quad::ZERO; NUM_RVS]; net_info.num_isos];
    with_thread_local_work(&net_info, |w| {
        let log_t = model.ln_t[0] / LN10;
        let log_rho = model.lnd[0] / LN10;
        let _ = eval_net_with_work(
            &net_info,
            w,
            &mut dydt,
            false,
            &model.xa,
            model.t[0],
            log_t,
            model.rho[0],
            log_rho,
            model.abar[0],
            model.zbar[0],
            model.z2bar[0],
            model.ye[0],
            model.eta[0],
            model.eos_d_dlnt[0][I_ETA],
            model.eos_d_dlnd[0][I_ETA],
            &rf,
            1.0,
            model.controls.screening_mode,
            false,
            false,
            false,
        )
        .expect("eval_net_with_work");
        for &(rxn_name, want_scr_hex) in case.reaction_assertions {
            let idx = net_info
                .reactions
                .iter()
                .position(|r| r.name == rxn_name)
                .unwrap_or_else(|| {
                    panic!("{} missing reaction {}", case.label, rxn_name)
                });
            let got_scr = w.rate_screened[idx];
            assert_eq!(
                got_scr.to_bits(),
                want_scr_hex,
                "{} rate_screened[{}] mismatch: got=0x{:016X} ({:.17e}) want=0x{:016X}",
                case.label,
                rxn_name,
                got_scr.to_bits(),
                got_scr,
                want_scr_hex,
            );
        }
    });
}

#[test]
fn test_rattab_temp_lo_matches_mesa_exp10() {
    with_numerics_mode(NumericsMode::Strict, || {
        assert_eq!(RATTAB_TEMP_LO.to_bits(), 0x4108_69FF_FFFF_FFE9);
        assert_eq!(RATTAB_TEMP_LO.to_bits(), exp10(RATTAB_TLO).to_bits());
    });
}

#[test]
fn test_aux_eps_nuc_surface_effects_zone1_zero_ulp() {
    let case = CaseInput {
        label: "astero/surface_effects z=1",
        net_name: "pp_and_cno_extras.net",
        lnd: 0xC02E9C2F2D63BB4E,
        lnt: 0x4021523660DECEDE,
        lnr: 0x4038F7320AEFEAAB,
        species_names: &[
            "h1", "h2", "he3", "he4", "li7", "be7", "b8", "c12", "c13", "n13",
            "n14", "n15", "o14", "o15", "o16", "o17", "o18", "f17", "f18",
            "f19", "ne18", "ne19", "ne20", "mg22", "mg24",
        ],
        xa_hex: &[
            0x3FE74865FA4C7B85,
            0x3C7A1CAFBBBD71FA,
            0x3EFAA3D8BF640F99,
            0x3FD03A1808076B14,
            0x3CE3EFDF7FB810FE,
            0x38B9D2E7DBB2E682,
            0x2CE5400000000000,
            0x3F6A747EC6DF308D,
            0x3F048D6909F90521,
            0x3135C1B000000000,
            0x3F4F456A6AE17E55,
            0x3ECF86F768C2D669,
            0x2B879281C4A594B0,
            0x2D8275A000000000,
            0x3F822B594D568881,
            0x3ECCDE4772890D75,
            0x3EF48BD00EFF0A23,
            0x2B61E90000000000,
            0x2B6D7D9B0951CADF,
            0x3E9F0049AF1773E1,
            0x2B618233F8B38ECF,
            0x2B619118171AC220,
            0x3F5FB532D0C09E7F,
            0x39D68FC710B6C169,
            0x3F6EFF8368B33077,
        ],
        want_eps_nuc_hex: 0x3AA09AB29A3A76C9,
        want_peos_hex: 0x40F4C48EC5705E4B,
        want_cp_hex: 0x41A421ADF98DA30B,
        reaction_assertions: &[
            ("r_be7_wk_li7", 0x3E8434AECB5CCB90),
            ("r_n13_wk_c13", 0x3F52FDAC61EF833C),
            ("r_o15_wk_n15", 0x3F77458B4A112F36),
        ],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        assert_aux_case(&case);
    });
}

#[test]
fn test_aux_eps_nuc_cepheid_blue_loop_zone1_zero_ulp() {
    let case = CaseInput {
        label: "star/5M_cepheid_blue_loop z=1",
        net_name: "pp_cno_extras_o18_ne22.net",
        lnd: 0xC032858E29BEA6D8,
        lnt: 0x4020C1239B3F2CC5,
        lnr: 0x403D2F6F5C1859E3,
        species_names: &[
            "h1", "h2", "he3", "he4", "li7", "be7", "b8", "c12", "c13", "n13",
            "n14", "n15", "o14", "o15", "o16", "o17", "o18", "f17", "f18",
            "f19", "ne18", "ne19", "ne20", "ne22", "mg22", "mg24",
        ],
        xa_hex: &[
            0x3FE78D4FDF3B645A,
            0x3C4080794463C676,
            0x3EFC9122A7558802,
            0x3FD061DB8E670CA6,
            0x3E3810919D2901EF,
            0x36A28AB77F6485C5,
            0x2B617F7D4ED8C33E,
            0x3F56F1FA10A4B480,
            0x3EF1D35EBEA159E1,
            0x2B618A4CD915F112,
            0x3F3B1F52DA2FD92D,
            0x3EBB582D7B3F4283,
            0x2B6181BD071A7563,
            0x2B6183D45ACE8D0D,
            0x3F6C170D4E693CB6,
            0x3EB6CC5A33570444,
            0x3EE039D9EE54DC89,
            0x2B6181747C1871C0,
            0x2B6DC00AC6ED65A2,
            0x3E9450C66FEF10A3,
            0x2B617F7D4ED8C341,
            0x2B617F8E03504800,
            0x3F46A3BC7B75B1C7,
            0x3F0D4C50DA1D969C,
            0x331B1C85F15CD5DC,
            0x3F603C5786D815A4,
        ],
        want_eps_nuc_hex: 0x3A34C611A5E6C185,
        want_peos_hex: 0x40A4493E51B3BB96,
        want_cp_hex: 0x41A4299C87647F8B,
        reaction_assertions: &[("r_be7_wk_li7", 0x3E8434AECB5CCB90)],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        assert_aux_case(&case);
    });
}

#[test]
fn test_aux_eps_nuc_thermohaline_zone82_zero_ulp() {
    let case = CaseInput {
        label: "star/1M_thermohaline z=82",
        net_name: "basic.net",
        lnd: 0xC02AE86F391FEFCA,
        lnt: 0x4022F89CC8AB820B,
        lnr: 0x4038E1B34FC07131,
        species_names: &[
            "h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24",
        ],
        xa_hex: &[
            0x3FE6666663AC7E28,
            0x3EFF401ADB241158,
            0x3FD1EB0823C0B5D7,
            0x3F6C3192DD1B49B9,
            0x3F508485F7FDD8A6,
            0x3F832B923C428DCC,
            0x3F6132E342FF8102,
            0x3F70C10418265ABB,
        ],
        want_eps_nuc_hex: 0x3AD052DC0505B6B0,
        want_peos_hex: 0x4135567157A5B9AC,
        want_cp_hex: 0x41D28FD38EAE2BC4,
        reaction_assertions: &[
            ("rpp_to_he3", 0x3707E0521917E2D2),
            ("rpep_to_he3", 0x350DB538AFC66540),
        ],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        assert_aux_case(&case);
    });
}

#[test]
fn test_aux_eps_nuc_irradiated_planet_zone3_zero_ulp() {
    let case = CaseInput {
        label: "star/irradiated_planet z=3",
        net_name: "basic.net",
        lnd: 0xC028F4B1BDD0E7B3,
        lnt: 0x401CD7E93DD0ACCA,
        lnr: 0x4037564A9053DA59,
        species_names: &[
            "h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24",
        ],
        xa_hex: &[
            0x3FE7AE147AE147B0,
            0x3EFAC8107CE02F80,
            0x3FCEB77BAB0137B5,
            0x3F6C3192DD1B4A1B,
            0x3F508485F7FDD8A3,
            0x3F832B923C428DC9,
            0x3F6132E342FF80FF,
            0x3F70C10418265AB8,
        ],
        want_eps_nuc_hex: 0x3AE83E42E0BC2494,
        want_peos_hex: 0x41069E95D2B0A8B5,
        want_cp_hex: 0x419F272844B29BD0,
        reaction_assertions: &[
            ("rpp_to_he3", 0x371FBAE6260A539E),
            ("rpep_to_he3", 0x353ACF99C9DCAC22),
        ],
    };
    with_numerics_mode(NumericsMode::Strict, || {
        assert_aux_case(&case);
    });
}
