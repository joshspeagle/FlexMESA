// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#161` MESA `26.4.1` `ecapture` derivative wiring, guard parity, and
//! zero-`$MESA_DIR` target-suite network tests for `flexmesa-net`.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_math::{exp, exp10, log10, with_numerics_mode, NumericsMode};
use flexmesa_net::derivs_support::{eval_single_weak_blended, NetWorkState};
use flexmesa_net::{
    eval_net, eval_net_with_work, NetGeneralInfo, Quad, NUM_RVS,
};
use flexmesa_rates::{
    eval_ecapture_pair_with_derivs, CoulombMuiOption, CoulombVsOption,
    ScreeningMode, WeakRateEval,
};

const WD_AIC_NET_FIXTURE: &str = "\
      include 'co_burn.net'

      add_isos(
         o20
         f20
         ne24
         na24
         )

      add_reactions(
          r_ne20_wk_f20
          r_f20_wk-minus_ne20,
          r_f20_wk_o20,
          r_o20_wk-minus_f20,
          r_mg24_wk_na24
          r_na24_wk-minus_mg24
          r_na24_wk_ne24,
          r_ne24_wk-minus_na24
         )
";

struct TempNetDir {
    root: PathBuf,
}

impl TempNetDir {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_net_161_ecap_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp net dir");
        Self { root }
    }
}

impl Drop for TempNetDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn weak_pair_ids(cat: &NuclideCatalog, lhs: &str, rhs: &str) -> [u16; 2] {
    let cid_l = cat
        .get_by_name(lhs)
        .unwrap_or_else(|| panic!("unknown nuclide {lhs}"))
        .chem_id;
    let cid_r = cat
        .get_by_name(rhs)
        .unwrap_or_else(|| panic!("unknown nuclide {rhs}"))
        .chem_id;
    [
        u16::try_from(cid_l).expect("lhs chem_id fits u16"),
        u16::try_from(cid_r).expect("rhs chem_id fits u16"),
    ]
}

fn assert_weak_eval_exact_bits(
    label: &str,
    got: &WeakRateEval,
    exp: &[u64; 9],
) {
    let fields = [
        ("lambda", got.lambda, exp[0]),
        ("dlambda_dln_t", got.dlambda_dln_t, exp[1]),
        ("dlambda_dln_rho", got.dlambda_dln_rho, exp[2]),
        ("q", got.q, exp[3]),
        ("dq_dln_t", got.dq_dln_t, exp[4]),
        ("dq_dln_rho", got.dq_dln_rho, exp[5]),
        ("qneu", got.qneu, exp[6]),
        ("dqneu_dln_t", got.dqneu_dln_t, exp[7]),
        ("dqneu_dln_rho", got.dqneu_dln_rho, exp[8]),
    ];
    for (name, val, want_bits) in fields {
        let got_bits = val.to_bits();
        assert_eq!(
            got_bits,
            want_bits,
            "{label}.{name}: got {val:.17e} ({got_bits:#018x}) != \
             want {:.17e} ({want_bits:#018x})",
            f64::from_bits(want_bits)
        );
    }
}

fn assert_weak_eval_identical(
    label: &str,
    lhs: &WeakRateEval,
    rhs: &WeakRateEval,
) {
    assert_weak_eval_exact_bits(
        label,
        lhs,
        &[
            rhs.lambda.to_bits(),
            rhs.dlambda_dln_t.to_bits(),
            rhs.dlambda_dln_rho.to_bits(),
            rhs.q.to_bits(),
            rhs.dq_dln_t.to_bits(),
            rhs.dq_dln_rho.to_bits(),
            rhs.qneu.to_bits(),
            rhs.dqneu_dln_t.to_bits(),
            rhs.dqneu_dln_rho.to_bits(),
        ],
    );
}

fn assert_rel_close(label: &str, actual: f64, expected: f64, rtol: f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{label}: non-finite values actual={actual}, expected={expected}"
    );
    let scale = expected.abs().max(1.0e-30);
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= rtol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel_err={rel:.3e} > {rtol:.3e}"
    );
}

fn load_wd_aic_fixture() -> (TempNetDir, NetGeneralInfo) {
    let dir = TempNetDir::new("wd_aic");
    fs::write(dir.root.join("wd_aic.net"), WD_AIC_NET_FIXTURE)
        .expect("write wd_aic.net fixture");
    let g = NetGeneralInfo::from_name_or_dir("wd_aic.net", Some(&dir.root))
        .expect("load wd_aic.net fixture");
    (dir, g)
}

type EcapHexOracle<'a> = (&'a str, &'a str, &'a str, [u64; 7], [u64; 9]);

#[test]
fn test_eval_single_weak_blended_ecapture_matches_mesa_26_4_1_hex_with_derivs()
{
    // MESA 26.4.1 `rates/private/eval_ecapture.f90` `cmode = 1`
    // (`Pcr2009 + Itoh2002`) hex oracles routed through
    // `eval_single_weak_blended` with `(d_eta_dln_t, d_eta_dln_rho)` active.
    let oracles: &[EcapHexOracle<'_>] = &[
        (
            "r_na24_wk_ne24",
            "na24",
            "ne24",
            [
                0x3FF3_3333_3333_3333,
                0x41FA_D274_8000_0000,
                0x403C_0000_0000_0000,
                0xC03B_8000_0000_0000,
                0x4022_CCCC_CCCC_CCCD,
                0x4028_0000_0000_0000,
                0x4039_0000_0000_0000,
            ],
            [
                0x3EBA_B3D8_7887_F164,
                0x3EE1_ED0A_01CF_2694,
                0x3EDA_B2B5_EB30_4BF6,
                0xC003_BB95_1C5C_5718,
                0,
                0,
                0x3FE3_5D43_F478_9108,
                0x3FD1_D1D1_E20B_5D02,
                0x3FE1_88D7_D3CE_B4AB,
            ],
        ),
        (
            "r_na24_wk_ne24",
            "na24",
            "ne24",
            [
                0x400C_0000_0000_0000,
                0x4195_752A_0000_0000,
                0x4004_0000_0000_0000,
                0xC000_CCCC_CCCC_CCCD,
                0x3FEC_CCCC_CCCC_CCCD,
                0x4022_0000_0000_0000,
                0x4034_0000_0000_0000,
            ],
            [
                0x3E8D_7654_56D5_0594,
                0x3EC1_8C37_BCE7_B0A9,
                0x3E88_D5B0_8683_A0FF,
                0xC003_BB95_1C5C_5718,
                0,
                0,
                0x3FF1_2CA6_2990_1910,
                0x3FF3_499D_F10B_DEAC,
                0xBF34_217E_EB25_5AFA,
            ],
        ),
        (
            "r_na24_wk-minus_mg24",
            "na24",
            "mg24",
            [
                0x3FF3_3333_3333_3333,
                0x41FA_D274_8000_0000,
                0x403C_0000_0000_0000,
                0xC03B_8000_0000_0000,
                0x4022_CCCC_CCCC_CCCD,
                0x4028_0000_0000_0000,
                0x4039_0000_0000_0000,
            ],
            [
                0x3F1C_6B85_16A6_F61B,
                0x3F40_2104_782F_3643,
                0xBF08_996B_3964_AED3,
                0x4016_0FD2_E946_8018,
                0,
                0,
                0x4002_BE68_3A75_052A,
                0x3F78_C76D_C5C5_EC3C,
                0xBFDF_BD3D_E052_0AE9,
            ],
        ),
        (
            "r_na24_wk-minus_mg24",
            "na24",
            "mg24",
            [
                0x400C_0000_0000_0000,
                0x4195_752A_0000_0000,
                0x4004_0000_0000_0000,
                0xC000_CCCC_CCCC_CCCD,
                0x3FEC_CCCC_CCCC_CCCD,
                0x4022_0000_0000_0000,
                0x4034_0000_0000_0000,
            ],
            [
                0x3F79_F7AB_4DCF_D6B2,
                0x3F92_10C6_A6A4_0D7A,
                0xBF15_FE9D_FF7A_E30D,
                0x4016_0FD2_E946_8018,
                0,
                0,
                0x400A_F5AA_1942_76F8,
                0x3FD0_AB21_F41F_456B,
                0xBFA7_70E8_E943_8DE2,
            ],
        ),
        (
            "r_ne24_wk-minus_na24",
            "ne24",
            "na24",
            [
                0x3FF3_3333_3333_3333,
                0x41FA_D274_8000_0000,
                0x403C_0000_0000_0000,
                0xC03B_8000_0000_0000,
                0x4022_CCCC_CCCC_CCCD,
                0x4028_0000_0000_0000,
                0x4039_0000_0000_0000,
            ],
            [
                0x3E61_B2DF_16BD_F829,
                0x3E92_0C33_7E9A_89BE,
                0xBE92_95CE_44F4_E756,
                0x4003_BB95_1C5C_5718,
                0,
                0,
                0x3FD2_42DD_724B_6FDF,
                0x3FD0_91EA_1000_17AF,
                0x3F3E_76F0_83E9_D6EB,
            ],
        ),
        (
            "r_ne24_wk-minus_na24",
            "ne24",
            "na24",
            [
                0x400C_0000_0000_0000,
                0x4195_752A_0000_0000,
                0x4004_0000_0000_0000,
                0xC000_CCCC_CCCC_CCCD,
                0x3FEC_CCCC_CCCC_CCCD,
                0x4022_0000_0000_0000,
                0x4034_0000_0000_0000,
            ],
            [
                0x3F5E_597A_E569_21CA,
                0xBF39_A270_9D80_7E16,
                0xBF3D_68F9_9573_79D7,
                0x4003_BB95_1C5C_5718,
                0,
                0,
                0x3FF0_0423_984F_B1C8,
                0x3FBA_FEEC_A468_7364,
                0xBFB1_9966_4D98_75AE,
            ],
        ),
        (
            "r_o20_wk-minus_f20",
            "o20",
            "f20",
            [
                0x3FF3_3333_3333_3333,
                0x41FA_D274_8000_0000,
                0x403C_0000_0000_0000,
                0xC03B_8000_0000_0000,
                0x4022_CCCC_CCCC_CCCD,
                0x4028_0000_0000_0000,
                0x4039_0000_0000_0000,
            ],
            [
                0x3F2B_0762_D03B_F60A,
                0x3F34_261A_4D12_59D6,
                0xBF57_D776_3C2E_067E,
                0x400E_84D8_7724_FA8A,
                0,
                0,
                0x3FD4_A5BA_59D1_0DCB,
                0x3FD1_189A_61F3_246D,
                0xBFC8_662E_9A9A_9F3C,
            ],
        ),
        (
            "r_o20_wk-minus_f20",
            "o20",
            "f20",
            [
                0x400C_0000_0000_0000,
                0x4195_752A_0000_0000,
                0x4004_0000_0000_0000,
                0xC000_CCCC_CCCC_CCCD,
                0x3FEC_CCCC_CCCC_CCCD,
                0x4022_0000_0000_0000,
                0x4034_0000_0000_0000,
            ],
            [
                0x3FA4_9EE0_7724_8E6C,
                0xBF82_6B01_2874_828D,
                0xBF77_1735_DD44_A9C0,
                0x400E_84D8_7724_FA8A,
                0,
                0,
                0x3FF6_77CE_0537_3FD3,
                0x3FB8_4830_5270_73C9,
                0xBFB3_B203_3C9B_213B,
            ],
        ),
    ];

    let cat = NuclideCatalog::builtin_ref();
    with_numerics_mode(NumericsMode::Strict, || {
        for (idx, &(rxn_name, lhs, rhs, inp, exp)) in oracles.iter().enumerate()
        {
            let ids = weak_pair_ids(cat, lhs, rhs);
            let t9 = f64::from_bits(inp[0]);
            let ye_rho = f64::from_bits(inp[1]);
            let eta_k = f64::from_bits(inp[2]);
            let d_eta_dln_t = f64::from_bits(inp[3]);
            let d_eta_dln_rho = f64::from_bits(inp[4]);
            let zbar = f64::from_bits(inp[5]);
            let abar = f64::from_bits(inp[6]);
            let temp = t9 * 1.0e9;
            let rho = ye_rho / (zbar / abar);

            let got = eval_single_weak_blended(
                rxn_name,
                ids,
                true,
                t9,
                ye_rho,
                temp,
                rho,
                abar,
                zbar,
                zbar * zbar,
                eta_k,
                d_eta_dln_t,
                d_eta_dln_rho,
                false,
                true,
            );
            assert_weak_eval_exact_bits(
                &format!("{rxn_name}#{idx}"),
                &got,
                &exp,
            );

            // Dropping `(d_eta_dln_t, d_eta_dln_rho)` must change derivatives.
            let zero_deta = eval_single_weak_blended(
                rxn_name,
                ids,
                true,
                t9,
                ye_rho,
                temp,
                rho,
                abar,
                zbar,
                zbar * zbar,
                eta_k,
                0.0,
                0.0,
                false,
                true,
            );
            assert_ne!(
                got.dlambda_dln_t.to_bits(),
                zero_deta.dlambda_dln_t.to_bits(),
                "{rxn_name}#{idx} dlambda_dln_t should depend on d_eta_dln_t"
            );
            assert_ne!(
                got.dlambda_dln_rho.to_bits(),
                zero_deta.dlambda_dln_rho.to_bits(),
                "{rxn_name}#{idx} dlambda_dln_rho should depend on d_eta_dln_rho"
            );
        }
    });
}

#[test]
fn test_eval_single_weak_blended_ecapture_guards_preserve_weaklib_rate() {
    let cat = NuclideCatalog::builtin_ref();
    let na24_ne24 = weak_pair_ids(cat, "na24", "ne24");
    let na24_mg24 = weak_pair_ids(cat, "na24", "mg24");
    let ne24_na24 = weak_pair_ids(cat, "ne24", "na24");

    with_numerics_mode(NumericsMode::Strict, || {
        // 1. Degeneracy guard (`rates/private/eval_ecapture.f90:99-114`):
        //    when `eta + beta < 2 * beta`, `do_ecapture = true` must keep the
        //    weaklib rate instead of overwriting `ev` with `0.0`.
        for (rxn_name, ids) in [
            ("r_na24_wk_ne24", na24_ne24),
            ("r_na24_wk-minus_mg24", na24_mg24),
        ] {
            let t9 = 1.5;
            let ye_rho = 1.0e5;
            let abar = 24.0;
            let zbar = 12.0;
            let off = eval_single_weak_blended(
                rxn_name,
                ids,
                true,
                t9,
                ye_rho,
                t9 * 1.0e9,
                ye_rho / (zbar / abar),
                abar,
                zbar,
                zbar * zbar,
                -5.0,
                -3.0,
                1.0,
                false,
                false,
            );
            let on = eval_single_weak_blended(
                rxn_name,
                ids,
                true,
                t9,
                ye_rho,
                t9 * 1.0e9,
                ye_rho / (zbar / abar),
                abar,
                zbar,
                zbar * zbar,
                -5.0,
                -3.0,
                1.0,
                false,
                true,
            );
            assert!(
                off.lambda > 0.0,
                "{rxn_name} weaklib baseline must be nonzero"
            );
            assert_weak_eval_identical(
                &format!("{rxn_name}_nondeg"),
                &on,
                &off,
            );
        }

        // 2. `has_weaklib == false` guard (`ids(i) <= 0` in
        //    `rates/private/eval_ecapture.f90:122`): even in the degenerate
        //    regime (`eta = 28.0`), pairs without a weaklib slot preserve the
        //    `weak_low_t_rate` decay rather than overriding with ecapture.
        for (rxn_name, ids) in [
            ("r_na24_wk-minus_mg24", na24_mg24),
            ("r_ne24_wk-minus_na24", ne24_na24),
        ] {
            let t9 = 1.2;
            let ye_rho = 1.0e9;
            let abar = 24.0;
            let zbar = 12.0;
            let off = eval_single_weak_blended(
                rxn_name,
                ids,
                false,
                t9,
                ye_rho,
                t9 * 1.0e9,
                ye_rho / (zbar / abar),
                abar,
                zbar,
                zbar * zbar,
                28.0,
                -27.5,
                9.2,
                false,
                false,
            );
            let on = eval_single_weak_blended(
                rxn_name,
                ids,
                false,
                t9,
                ye_rho,
                t9 * 1.0e9,
                ye_rho / (zbar / abar),
                abar,
                zbar,
                zbar * zbar,
                28.0,
                -27.5,
                9.2,
                false,
                true,
            );
            assert!(
                off.lambda > 0.0,
                "{rxn_name} low-T decay rate must be positive"
            );
            assert_weak_eval_identical(
                &format!("{rxn_name}_no_weaklib"),
                &on,
                &off,
            );
        }

        // 3. Positivity guards (`t9`, `ye_rho`, `zbar`, `abar`): zero inputs must
        //    neither divide by zero (`1/kbt`, `ye_rho / (zbar / abar)`) nor NaN.
        for (label, t9, ye_rho, abar, zbar) in [
            ("zero_t9", 0.0, 1.0e9, 24.0, 12.0),
            ("zero_ye_rho", 1.2, 0.0, 24.0, 12.0),
            ("zero_zbar", 1.2, 1.0e9, 24.0, 0.0),
            ("zero_abar", 1.2, 1.0e9, 0.0, 12.0),
        ] {
            let off = eval_single_weak_blended(
                "r_na24_wk_ne24",
                na24_ne24,
                true,
                t9,
                ye_rho,
                t9 * 1.0e9,
                1.0e9,
                abar,
                zbar,
                zbar * zbar,
                28.0,
                -27.5,
                9.2,
                false,
                false,
            );
            let on = eval_single_weak_blended(
                "r_na24_wk_ne24",
                na24_ne24,
                true,
                t9,
                ye_rho,
                t9 * 1.0e9,
                1.0e9,
                abar,
                zbar,
                zbar * zbar,
                28.0,
                -27.5,
                9.2,
                false,
                true,
            );
            assert!(on.lambda.is_finite(), "{label} lambda must be finite");
            assert!(
                on.dlambda_dln_t.is_finite(),
                "{label} dlambda_dln_t must be finite"
            );
            assert!(on.qneu.is_finite(), "{label} qneu must be finite");
            assert_weak_eval_identical(label, &on, &off);
        }
    });
}

#[test]
fn test_wd_aic_eval_net_wires_ecapture_eta_derivs_and_fd_consistency() {
    let (_dir, g) = load_wd_aic_fixture();
    let cat = NuclideCatalog::builtin_ref();

    let mut x = vec![0.0_f64; g.num_isos];
    let set_x = |x: &mut [f64], iso: &str, val: f64| {
        let cid = cat.get_by_name(iso).unwrap().chem_id;
        let idx = g.net_iso[cid] - 1;
        x[idx] = val;
    };
    set_x(&mut x, "ne20", 0.25);
    set_x(&mut x, "f20", 0.10);
    set_x(&mut x, "o20", 0.10);
    set_x(&mut x, "mg24", 0.15);
    set_x(&mut x, "na24", 0.25);
    set_x(&mut x, "ne24", 0.15);

    let comp = get_composition_info(cat, &g.chem_id, &x, false)
        .expect("wd_aic composition info");
    let rf = vec![1.0_f64; g.num_reactions];

    with_numerics_mode(NumericsMode::Strict, || {
        let t9 = 1.2;
        let ye_rho = 1.0e9;
        let temp = t9 * 1.0e9;
        let log_t = log10(temp);
        let rho = ye_rho / comp.ye;
        let log_rho = log10(rho);
        let eta = 28.0;
        let d_eta_dln_t = -27.5;
        let d_eta_dln_rho = 9.2;

        // 1. Reaction-level `dlambda_dln_t` and `dlambda_dln_rho` inside
        //    `eval_net_with_work` match `eval_ecapture_pair_with_derivs`.
        let mut work = NetWorkState::new(&g);
        let mut dydt = vec![[Quad::ZERO; NUM_RVS]; g.num_isos];
        let _ = eval_net_with_work(
            &g,
            &mut work,
            &mut dydt,
            false,
            &x,
            temp,
            log_t,
            rho,
            log_rho,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta,
            d_eta_dln_t,
            d_eta_dln_rho,
            &rf,
            1.0,
            ScreeningMode::Chugunov,
            false,
            false,
            true,
        )
        .expect("wd_aic eval_net_with_work");

        for (rxn_name, lhs, rhs) in [
            ("r_na24_wk_ne24", "na24", "ne24"),
            ("r_na24_wk-minus_mg24", "na24", "mg24"),
            ("r_ne24_wk-minus_na24", "ne24", "na24"),
            ("r_f20_wk_o20", "f20", "o20"),
            ("r_o20_wk-minus_f20", "o20", "f20"),
            ("r_ne20_wk_f20", "ne20", "f20"),
            ("r_f20_wk-minus_ne20", "f20", "ne20"),
            ("r_mg24_wk_na24", "mg24", "na24"),
        ] {
            let ir = g
                .reaction_index_by_name(rxn_name)
                .unwrap_or_else(|| panic!("missing {rxn_name}"))
                - 1;
            let wk = g.weak_reaction_index[ir] - 1;
            assert!(g.weaklib_ids[wk] > 0, "{rxn_name} must use weaklib slot");
            let ec = eval_ecapture_pair_with_derivs(
                lhs,
                rhs,
                temp * 1.0e-9,
                comp.ye * rho,
                eta,
                d_eta_dln_t,
                d_eta_dln_rho,
                CoulombMuiOption::Pcr2009,
                CoulombVsOption::Itoh2002,
                comp.zbar,
                comp.abar,
            )
            .expect("eval_ecapture_pair_with_derivs");
            assert_eq!(work.wk_lambda[wk].to_bits(), ec.lambda.to_bits());
            assert_eq!(
                work.wk_dlambda_dln_t[wk].to_bits(),
                ec.dlambda_dln_t.to_bits()
            );
            assert_eq!(
                work.wk_dlambda_dln_rho[wk].to_bits(),
                ec.dlambda_dln_rho.to_bits()
            );
            assert_eq!(work.wk_q[wk].to_bits(), ec.q.to_bits());
            assert_eq!(work.wk_qneu[wk].to_bits(), ec.qneu.to_bits());
            assert_eq!(
                work.wk_dqneu_dln_t[wk].to_bits(),
                ec.dqneu_dln_t.to_bits()
            );
            assert_eq!(
                work.wk_dqneu_dln_rho[wk].to_bits(),
                ec.dqneu_dln_rho.to_bits()
            );
            assert_rel_close(
                &format!("{rxn_name}.rate_screened"),
                work.rate_screened[ir],
                ec.lambda,
                1.0e-14,
            );
            assert_rel_close(
                &format!("{rxn_name}.rate_screened_dt*T"),
                work.rate_screened_dt[ir] * temp,
                ec.dlambda_dln_t,
                1.0e-13,
            );
            assert_rel_close(
                &format!("{rxn_name}.rate_screened_drho*Rho"),
                work.rate_screened_drho[ir] * rho,
                ec.dlambda_dln_rho,
                1.0e-13,
            );
        }

        // 2. Network `d_eps_nuc_{dt,drho}` and `d_dxdt_{dt,drho}` respond to
        //    `(d_eta_dln_t, d_eta_dln_rho)` and agree with centered FD.
        let res = eval_net(
            &g,
            false,
            false,
            &x,
            temp,
            log_t,
            rho,
            log_rho,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta,
            d_eta_dln_t,
            d_eta_dln_rho,
            &rf,
            1.0,
            ScreeningMode::Chugunov,
            false,
            false,
            true,
            false,
        )
        .expect("wd_aic eval_net");

        let res_no_deta = eval_net(
            &g,
            false,
            false,
            &x,
            temp,
            log_t,
            rho,
            log_rho,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta,
            0.0,
            0.0,
            &rf,
            1.0,
            ScreeningMode::Chugunov,
            false,
            false,
            true,
            false,
        )
        .expect("wd_aic eval_net zero d_eta");

        let ina24 = g.net_iso[cat.get_by_name("na24").unwrap().chem_id] - 1;
        assert!(
            (res.d_eps_nuc_dt - res_no_deta.d_eps_nuc_dt).abs()
                > 1.0e-3 * res.d_eps_nuc_dt.abs(),
            "d_eps_nuc_dt must reflect d_eta_dln_t"
        );
        assert!(
            (res.d_eps_nuc_drho - res_no_deta.d_eps_nuc_drho).abs()
                > 1.0e-3 * res.d_eps_nuc_drho.abs(),
            "d_eps_nuc_drho must reflect d_eta_dln_rho"
        );
        assert!(
            (res.d_dxdt_dt[ina24] - res_no_deta.d_dxdt_dt[ina24]).abs()
                > 1.0e-3 * res.d_dxdt_dt[ina24].abs(),
            "d_dxdt_dt[na24] must reflect d_eta_dln_t"
        );
        assert!(
            (res.d_dxdt_drho[ina24] - res_no_deta.d_dxdt_drho[ina24]).abs()
                > 1.0e-3 * res.d_dxdt_drho[ina24].abs(),
            "d_dxdt_drho[na24] must reflect d_eta_dln_rho"
        );

        let h = 1.0e-6;
        let eval_at = |t_val: f64, rho_val: f64, eta_val: f64| {
            eval_net(
                &g,
                false,
                false,
                &x,
                t_val,
                log10(t_val),
                rho_val,
                log10(rho_val),
                comp.abar,
                comp.zbar,
                comp.z2bar,
                comp.ye,
                eta_val,
                d_eta_dln_t,
                d_eta_dln_rho,
                &rf,
                1.0,
                ScreeningMode::Chugunov,
                false,
                false,
                true,
                false,
            )
            .expect("wd_aic eval_net FD point")
        };

        let tp = eval_at(temp * exp(h), rho, eta + h * d_eta_dln_t);
        let tm = eval_at(temp * exp(-h), rho, eta - h * d_eta_dln_t);
        let rp = eval_at(temp, rho * exp(h), eta + h * d_eta_dln_rho);
        let rm = eval_at(temp, rho * exp(-h), eta - h * d_eta_dln_rho);

        assert_rel_close(
            "fd.d_eps_nuc_dlnt",
            res.d_eps_nuc_dt * temp,
            (tp.eps_nuc - tm.eps_nuc) / (2.0 * h),
            1.0e-6,
        );
        assert_rel_close(
            "fd.d_eps_nuc_dlnrho",
            res.d_eps_nuc_drho * rho,
            (rp.eps_nuc - rm.eps_nuc) / (2.0 * h),
            1.0e-6,
        );
        for iso in ["o20", "f20", "ne20", "ne24", "na24", "mg24"] {
            let idx = g.net_iso[cat.get_by_name(iso).unwrap().chem_id] - 1;
            assert_rel_close(
                &format!("fd.d_dxdt_dlnt[{iso}]"),
                res.d_dxdt_dt[idx] * temp,
                (tp.dxdt[idx] - tm.dxdt[idx]) / (2.0 * h),
                1.0e-6,
            );
            assert_rel_close(
                &format!("fd.d_dxdt_dlnrho[{iso}]"),
                res.d_dxdt_drho[idx] * rho,
                (rp.dxdt[idx] - rm.dxdt[idx]) / (2.0 * h),
                1.0e-6,
            );
        }

        // 3. Non-degenerate `wd_aic` full-net guard: `do_ecapture = true` vs
        //    `false` must agree bit-for-bit when `eta + beta < 2 * beta`.
        let nd_temp = 1.5e9;
        let nd_rho = 1.0e5 / comp.ye;
        let nd_on = eval_net(
            &g,
            false,
            false,
            &x,
            nd_temp,
            log10(nd_temp),
            nd_rho,
            log10(nd_rho),
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            -5.0,
            -3.0,
            1.0,
            &rf,
            1.0,
            ScreeningMode::Chugunov,
            false,
            false,
            true,
            false,
        )
        .expect("wd_aic nondeg on");
        let nd_off = eval_net(
            &g,
            false,
            false,
            &x,
            nd_temp,
            log10(nd_temp),
            nd_rho,
            log10(nd_rho),
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            -5.0,
            -3.0,
            1.0,
            &rf,
            1.0,
            ScreeningMode::Chugunov,
            false,
            false,
            false,
            false,
        )
        .expect("wd_aic nondeg off");
        assert_eq!(nd_on.eps_nuc.to_bits(), nd_off.eps_nuc.to_bits());
        assert_eq!(nd_on.d_eps_nuc_dt.to_bits(), nd_off.d_eps_nuc_dt.to_bits());
        assert_eq!(
            nd_on.d_eps_nuc_drho.to_bits(),
            nd_off.d_eps_nuc_drho.to_bits()
        );
        assert_eq!(
            nd_on.eps_neu_total.to_bits(),
            nd_off.eps_neu_total.to_bits()
        );
        for (i, (&got_d, &want_d)) in
            nd_on.dxdt.iter().zip(&nd_off.dxdt).enumerate()
        {
            assert_eq!(
                got_d.to_bits(),
                want_d.to_bits(),
                "nondeg dxdt[{i}] bits mismatch"
            );
        }
    });
}

#[test]
fn test_target_suite_nets_zero_mesa_dir_load_and_evaluate() {
    let cat = NuclideCatalog::builtin_ref();
    let (_dir, wd_aic) = load_wd_aic_fixture();

    let mut nets: Vec<(&str, NetGeneralInfo)> = [
        "o18_and_ne22.net",
        "cno_extras_plus_fe56.net",
        "cno_extras_o18_to_mg26_plus_fe56.net",
        "approx21.net",
        "approx21_cr60_plus_co56.net",
        "rp_153.net",
        "co_burn.net",
    ]
    .into_iter()
    .map(|name| {
        let g = NetGeneralInfo::from_name(name)
            .unwrap_or_else(|e| panic!("failed to load builtin {name}: {e}"));
        (name, g)
    })
    .collect();
    nets.push(("wd_aic.net", wd_aic));

    with_numerics_mode(NumericsMode::Strict, || {
        for (name, g) in &nets {
            let n = u32::try_from(g.num_isos).expect("num_isos fits u32");
            let x = vec![1.0 / f64::from(n); g.num_isos];
            let comp = get_composition_info(cat, &g.chem_id, &x, false)
                .unwrap_or_else(|e| panic!("{name} composition: {e}"));
            let rf = vec![1.0_f64; g.num_reactions];
            let log_t = 8.6_f64;
            let log_rho = 3.0_f64;
            let temp = exp10(log_t);
            let rho = exp10(log_rho);

            let res_off = eval_net(
                g,
                false,
                false,
                &x,
                temp,
                log_t,
                rho,
                log_rho,
                comp.abar,
                comp.zbar,
                comp.z2bar,
                comp.ye,
                -2.0,
                0.0,
                0.0,
                &rf,
                1.0,
                ScreeningMode::Chugunov,
                false,
                false,
                false,
                false,
            )
            .unwrap_or_else(|e| panic!("{name} eval_net off: {e}"));

            let res_on = eval_net(
                g,
                false,
                false,
                &x,
                temp,
                log_t,
                rho,
                log_rho,
                comp.abar,
                comp.zbar,
                comp.z2bar,
                comp.ye,
                -2.0,
                0.0,
                0.0,
                &rf,
                1.0,
                ScreeningMode::Chugunov,
                false,
                false,
                true,
                false,
            )
            .unwrap_or_else(|e| panic!("{name} eval_net on: {e}"));

            assert!(res_on.eps_nuc.is_finite(), "{name} eps_nuc finite");
            assert!(
                res_on.d_eps_nuc_dt.is_finite(),
                "{name} d_eps_nuc_dt finite"
            );
            assert!(
                res_on.d_eps_nuc_drho.is_finite(),
                "{name} d_eps_nuc_drho finite"
            );
            assert!(
                res_on.eps_neu_total.is_finite(),
                "{name} eps_neu_total finite"
            );
            assert!(
                res_on.dxdt.iter().all(|v| v.is_finite()),
                "{name} dxdt finite"
            );
            assert!(
                res_on.d_dxdt_dt.iter().all(|v| v.is_finite()),
                "{name} d_dxdt_dt finite"
            );
            assert!(
                res_on.d_dxdt_drho.iter().all(|v| v.is_finite()),
                "{name} d_dxdt_drho finite"
            );
            assert!(
                res_on.d_dxdt_dx.iter().all(|v| v.is_finite()),
                "{name} d_dxdt_dx finite"
            );

            // At non-degenerate conditions (`eta = -2.0`), `do_ecapture = true`
            // must preserve exact `0-ULP` parity with `do_ecapture = false`.
            assert_eq!(
                res_on.eps_nuc.to_bits(),
                res_off.eps_nuc.to_bits(),
                "{name} nondeg eps_nuc must match"
            );
            assert_eq!(
                res_on.d_eps_nuc_dt.to_bits(),
                res_off.d_eps_nuc_dt.to_bits(),
                "{name} nondeg d_eps_nuc_dt must match"
            );
            assert_eq!(
                res_on.d_eps_nuc_drho.to_bits(),
                res_off.d_eps_nuc_drho.to_bits(),
                "{name} nondeg d_eps_nuc_drho must match"
            );
        }
    });
}
