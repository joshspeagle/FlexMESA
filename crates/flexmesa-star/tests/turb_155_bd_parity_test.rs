// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#155` / `#153` parity regression tests for MESA `26.4.1` turbulent
//! mixing fallback (`turb_support.f90:398-430`), `semiconvection.rs`, and
//! `thermohaline.rs`, motivated by `star/make_brown_dwarf` `RELAX m=1 ITER 2`.

#![forbid(unsafe_code)]
#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{MixingType, CRAD, STANDARD_CGRAV};
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_math::{exp, with_numerics_mode, NumericsMode};
use flexmesa_star::hydro_vars::set_mlt_vars;
use flexmesa_star::micro::{do_eos, do_kap, global_basic_net, set_basic_vars};
use flexmesa_star::star_ad::StarAdOrder1;
use flexmesa_star::{StarControls, StarJobControls, StarModel};
use flexmesa_turb::{
    set_mlt, set_semiconvection, set_thermohaline, DrivingSpecies, MltInputs,
    MltOption, MltParams, SemiconvectionInputs, SemiconvectionOption,
    SemiconvectionParams, ThermohalineInputs, ThermohalineOption,
    ThermohalineParams,
};

fn ad_seed(val: f64, d0: f64) -> DualOrder1<1> {
    DualOrder1::new(val, [d0])
}

fn star_ad_hex(bits: u64, d0: f64) -> StarAdOrder1 {
    let mut d1 = [0.0; 24];
    d1[0] = d0;
    StarAdOrder1::new(f64::from_bits(bits), d1)
}

fn assert_dual1_bits(
    name: &str,
    actual: DualOrder1<1>,
    exp_val: u64,
    exp_d0: u64,
) {
    assert_eq!(
        actual.val.to_bits(),
        exp_val,
        "{name}.val mismatch: got {:016X} ({}), expected {:016X} ({})",
        actual.val.to_bits(),
        actual.val,
        exp_val,
        f64::from_bits(exp_val),
    );
    assert_eq!(
        actual.d1[0].to_bits(),
        exp_d0,
        "{name}.d1[0] mismatch: got {:016X} ({}), expected {:016X} ({})",
        actual.d1[0].to_bits(),
        actual.d1[0],
        exp_d0,
        f64::from_bits(exp_d0),
    );
}

/// MESA `26.4.1` outputs of `turb::set_semiconvection` and
/// `turb::set_thermohaline` at this file's operating points, written by
/// `benchmarks/fortran_mesa/turb_mix_oracles/gen_turb_mix_oracles.f90`.
const TURB_MIX_ORACLES: &str = include_str!(
    "../../../benchmarks/fortran_mesa/turb_mix_oracles/mesa_26_4_1_turb_mix_oracles.tsv"
);

/// Return the MESA value of `field` (`part` = `val` or `d1`) in `case`.
fn mesa(case: &str, field: &str, part: &str) -> f64 {
    for line in TURB_MIX_ORACLES.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 5
            && cols[0] == case
            && cols[2] == field
            && cols[3] == part
        {
            let hex = cols[4].trim_start_matches("0x");
            return f64::from_bits(
                u64::from_str_radix(hex, 16).expect("hex column"),
            );
        }
    }
    panic!("no MESA oracle row for {case} {field} {part}");
}

/// Assert that `actual` (value and slot-1 partial) equals MESA bit for bit.
fn assert_dual1_mesa(case: &str, field: &str, actual: DualOrder1<1>) {
    assert_dual1_bits(
        &format!("{case}.{field}"),
        actual,
        mesa(case, field, "val").to_bits(),
        mesa(case, field, "d1").to_bits(),
    );
}

fn assert_star_ad_bits_prefix5(
    name: &str,
    actual: StarAdOrder1,
    expected: [u64; 5],
) {
    assert_eq!(
        actual.val.to_bits(),
        expected[0],
        "{name}.val mismatch: got {:016X}, expected {:016X}",
        actual.val.to_bits(),
        expected[0],
    );
    for (i, &exp_bits) in expected[1..].iter().enumerate() {
        assert_eq!(
            actual.d1[i].to_bits(),
            exp_bits,
            "{name}.d1[{i}] mismatch: got {:016X}, expected {:016X}",
            actual.d1[i].to_bits(),
            exp_bits,
        );
    }
}

fn build_bd_k375_k376_model() -> StarModel {
    let net = global_basic_net();
    let controls = StarControls {
        initial_z: 0.0001,
        zbase: 0.0001,
        use_ledoux_criterion: false,
        mixing_length_alpha: 2.0,
        mlt_option: MltOption::Cox,
        ..StarControls::default()
    };
    let mut model = StarModel::new(
        2,
        net.chem_id.clone(),
        StarJobControls::default(),
        controls,
    );
    model.kap_config = KapConfig {
        zbase: 0.0001,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        kap_low_t_option: KapLowTOption::LowTFreedman11,
        ..KapConfig::default()
    };
    model.tau_base = 2.0 / 3.0;
    model.mstar = 1.9892e31;
    model.xmstar = model.mstar;
    model.dt = 3.15576e2;
    model.dt_next = model.dt;
    model.aux.have_mlt_vc = true;
    model.aux.okay_to_set_mlt_vc = true;
    model.aux.mixing.gradr_factor.fill(1.0);
    model.aux.mixing.alpha_mlt.fill(2.0);

    let xa_bits: [u64; 8] = [
        0x3FE84F765FD8ADAD,
        0x3EFACDC71D680D6B,
        0x3FCEBE093606FA8A,
        0x3EF212541AC81F32,
        0x3ED5303C7D57B102,
        0x3F08972A60D212D5,
        0x3EE5DCF1DFB835A5,
        0x3EF5605AC0D2B419,
    ];
    // Cells k=375 (idx 0) and k=376 (idx 1) from brown-dwarf RELAX m=1 ITER 2.
    let cells = [
        (
            0x3F8AEAB7B6268FA0_u64,
            0x460A665E73AD84C1_u64,
            0x465114479A92AA36_u64,
            0xBFE8044DE9D25020_u64,
            0x4027C0F22BCC3AA9_u64,
            0x40374D620C6BC0BF_u64,
            0x477D90B286EF6A7C_u64,
        ),
        (
            0x3F899BD7A3ABF210_u64,
            0x46091DECF12AC6A7_u64,
            0x46504114A6F53E10_u64,
            0xBFE77371FBD0C86F_u64,
            0x4027C794232E853F_u64,
            0x4037484FFA3E2A14_u64,
            0x477DEB3809BC024C_u64,
        ),
    ];
    for (i, &(dq, dm, m, lnd, lnt, lnr, lum)) in cells.iter().enumerate() {
        model.dq[i] = f64::from_bits(dq);
        model.dm[i] = f64::from_bits(dm);
        model.m[i] = f64::from_bits(m);
        model.m_grav[i] = f64::from_bits(m);
        model.cgrav[i] = STANDARD_CGRAV;
        model.lnd[i] = f64::from_bits(lnd);
        model.rho[i] = exp(model.lnd[i]);
        model.ln_t[i] = f64::from_bits(lnt);
        model.t[i] = exp(model.ln_t[i]);
        model.ln_r[i] = f64::from_bits(lnr);
        model.r[i] = exp(model.ln_r[i]);
        model.lum[i] = f64::from_bits(lum);
        let base = i * 8;
        for j in 0..8 {
            model.xa[base + j] = f64::from_bits(xa_bits[j]);
        }
    }
    model.q[0] = model.m[0] / model.xmstar;
    model.q[1] = model.m[1] / model.xmstar;
    model.dm_bar[0] = 0.5 * model.dm[0];
    model.dm_bar[1] = f64::midpoint(model.dm[0], model.dm[1]);
    model.lnd_start.clone_from(&model.lnd);
    model.ln_t_start.clone_from(&model.ln_t);
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("eos");
    do_kap(&mut model).expect("kap");
    // Pin exact MESA 26.4.1 kap_get 0-ULP parity on k=375 and k=376
    // (face opacity = 0x40B35E41F74F29B7; Blouin conduction fixed in #464).
    assert_eq!(model.opacity[0].to_bits(), 0x40B38299E83645D1);
    assert_eq!(model.opacity[1].to_bits(), 0x40B33BAE2D9EC068);
    model
}

#[test]
fn test_k376_mlt_zeta_saturation_0_ulp_and_partials() {
    with_numerics_mode(NumericsMode::Strict, || {
        // Part 1: Pin exact MESA 26.4.1 set_MLT ('Cox') outputs & partials
        // when Zeta rounds >= 1.0 (turb/private/mlt.f90:170-179).
        let chi_t = star_ad_hex(0x3FF308A12B9A5F54, 0.125);
        let chi_rho = star_ad_hex(0x3FF01F0B253C30A0, -0.0625);
        let cp = star_ad_hex(0x41B4D40475F3ECBE, 1.0e7);
        let grav = star_ad_hex(0x40A013B580B1213E, 2.5e2);
        let mut lambda = star_ad_hex(0x420C98C32EAD5B0E, 1.0e8);
        lambda.d1[2] = -2.0e8;
        let rho = star_ad_hex(0x3FDE7DE7CD0AF62A, 0.25);
        let mut p = star_ad_hex(0x429B614BA63113D3, 1.0e12);
        p.d1[1] = 5.0e11;
        let mut t = star_ad_hex(0x4101AECD8D1746F0, 5.0e4);
        t.d1[1] = 1.0e4;
        let opacity = star_ad_hex(0x40B35E41F74F29B4, 1.0e2);
        let mut gradr = star_ad_hex(0x4242C4D856BA7448, 1.0e10);
        gradr.d1[1] = -2.0e10;
        gradr.d1[2] = 3.5e9;
        gradr.d1[3] = 1.612238e11;
        let grada = star_ad_hex(0x3FD7954A2BC074B6, 0.05);
        let gradl = grada;

        let params = MltParams {
            option: MltOption::Cox,
            mixing_length_alpha: 2.0,
            henyey_nu: 0.0,
            henyey_y: 0.0,
            max_conv_vel: 1.0e99,
        };
        let inputs = MltInputs {
            chi_t,
            chi_rho,
            cp,
            grav,
            lambda,
            rho,
            p,
            t,
            opacity,
            gradr,
            grada,
            gradl,
        };
        let out = set_mlt(&params, &inputs).expect("set_mlt");
        assert_eq!(out.mixing_type, MixingType::NoMixing);
        assert_star_ad_bits_prefix5(
            "k376_mlt.grad_t",
            out.grad_t,
            [
                0x3FD7954A2BC074B6,
                0x3FAAA05D6E57C962,
                0xBF535AFF19701746,
                0xBF0E7FDF8CEEB050,
                0x3F5F6119B0FF6B8D,
            ],
        );
        assert_star_ad_bits_prefix5(
            "k376_mlt.y_face",
            out.y_face,
            [
                0x0000000000000000,
                0x3F606C3D4BE2FC80,
                0xBF535AFF19701746,
                0xBF0E7FDF8CEEB050,
                0x3F5F6119B0FF6B8D,
            ],
        );
        assert_star_ad_bits_prefix5(
            "k376_mlt.conv_vel",
            out.conv_vel,
            [
                0x40F2D0E98FF42B13,
                0xC0D2DCCDF38F9012,
                0x40B5AAFC0DB30A7D,
                0x40932C279253F4ED,
                0x40D9168C447E06E3,
            ],
        );
        assert_star_ad_bits_prefix5(
            "k376_mlt.d",
            out.d,
            [
                0x42F66B7F4A83EC91,
                0xC2D5E42164A1C376,
                0x42B9D16683B679F5,
                0x42709C2BA2F9F38D,
                0x42DDE4A92598EBEE,
            ],
        );
        assert_star_ad_bits_prefix5(
            "k376_mlt.gamma",
            out.gamma,
            [
                0x42D9941743BC2F66,
                0x42C1512082E6DC7B,
                0xC2B49F619C59EB64,
                0xC27096831C30AC68,
                0x42C10D6483EAD766,
            ],
        );

        // Part 2: Through set_mlt_vars, unconditional semiconvection call
        // sets D = 0.0 (even with alpha_semiconvection == 0), then
        // remove_small_d_limit resets gradT = gradr, vc = 0, D = 0, Gamma = 0.
        let mut model = build_bd_k375_k376_model();
        // At Gamma ~ 1e14, Zeta = exp(3*log(Gamma) - log(B^3)) sits within
        // 1 ULP of 1.0; pin Part 1's Zeta >= 1.0 face opacity (0x40B35E41F74F29B4)
        // so set_mlt_vars deterministically exercises the turb_support.f90:398-430
        // NoMixing -> unconditional set_semiconvection -> remove_small_d_limit path.
        model.opacity.fill(f64::from_bits(0x40B35E41F74F29B4));
        set_mlt_vars(&mut model).expect("set_mlt_vars");
        assert_eq!(model.aux.mixing.mlt_mixing_type[1], MixingType::NoMixing);
        assert_eq!(model.gradt[1].to_bits(), 0x4242C4D856BA7448);
        assert_eq!(model.gradr[1].to_bits(), 0x4242C4D856BA7448);
        assert_eq!(
            (model.gradt[1] - model.gradl[1]).to_bits(),
            0x4242C4D856BA451D
        );
        assert_eq!(model.mlt_vc[1].to_bits(), 0);
        assert_eq!(model.mlt_d[1].to_bits(), 0);
        assert_eq!(model.mlt_gamma[1].to_bits(), 0);

        // Part 3: Disabling remove_small_d_limit still yields mlt_d == 0.0
        // because set_semiconvection runs unconditionally and writes D = 0.
        let mut model_no_limit = build_bd_k375_k376_model();
        model_no_limit
            .opacity
            .fill(f64::from_bits(0x40B35E41F74F29B4));
        model_no_limit.controls.mix_wind.remove_small_d_limit = -1.0;
        set_mlt_vars(&mut model_no_limit).expect("set_mlt_vars no limit");
        assert_eq!(
            model_no_limit.aux.mixing.mlt_mixing_type[1],
            MixingType::NoMixing
        );
        assert_eq!(model_no_limit.mlt_d[1], 0.0);
        assert_eq!(model_no_limit.gradt[1].to_bits(), 0x3FD7954A2BC074B6);
    });
}

#[test]
fn test_turb_support_unconditional_sc_and_thermohaline_fallbacks() {
    with_numerics_mode(NumericsMode::Strict, || {
        // Case A: alpha_semiconvection = 0 and gradr == gradl > grada where
        // Langer_85 formula yields 0/0 = NaN in D. MESA sets D = NaN without
        // returning an error and catches is_bad(D) via remove_small_D_limit.
        let mut model_sc_nan = build_bd_k375_k376_model();
        let face_lum_to_gradr = {
            set_mlt_vars(&mut model_sc_nan).expect("setup");
            let target_gradr = model_sc_nan.grada[1]
                * model_sc_nan.get_face_weights(1).0
                + model_sc_nan.grada[0] * model_sc_nan.get_face_weights(1).1;
            target_gradr / model_sc_nan.gradr[1]
        };
        model_sc_nan.controls.alpha_semiconvection = 0.0;
        model_sc_nan.lum[1] *= face_lum_to_gradr;
        set_mlt_vars(&mut model_sc_nan).expect("semiconvection NaN caught");
        assert_eq!(
            model_sc_nan.aux.mixing.mlt_mixing_type[1],
            MixingType::NoMixing
        );
        assert_eq!(
            model_sc_nan.gradt[1].to_bits(),
            model_sc_nan.gradr[1].to_bits()
        );
        assert_eq!(model_sc_nan.mlt_vc[1], 0.0);
        assert_eq!(model_sc_nan.mlt_d[1], 0.0);
        assert_eq!(model_sc_nan.mlt_gamma[1], 0.0);

        // Case B: thermohaline_coeff = 0 with gradL_composition_term < 0 in a
        // radiatively stable region still unconditionally invokes
        // set_thermohaline (turb_support.f90:399-407) and resets via
        // remove_small_d_limit = 5.0 (gradT = gradr, vc = 0, D = 0, Gamma = 0).
        let mut model_thrm = build_bd_k375_k376_model();
        model_thrm.controls.use_ledoux_criterion = true;
        model_thrm.controls.thermohaline_coeff = 0.0;
        model_thrm.controls.mix_wind.remove_small_d_limit = 5.0;
        model_thrm.aux.mixing.gradl_composition_term.fill(-0.05);
        model_thrm.lum.fill(1.0e20);
        set_mlt_vars(&mut model_thrm).expect("thermohaline with limit");
        assert_eq!(
            model_thrm.aux.mixing.mlt_mixing_type[1],
            MixingType::NoMixing
        );
        assert_eq!(
            model_thrm.gradt[1].to_bits(),
            model_thrm.gradr[1].to_bits()
        );
        assert_eq!(model_thrm.mlt_vc[1], 0.0);
        assert_eq!(model_thrm.mlt_d[1], 0.0);
        assert_eq!(model_thrm.mlt_gamma[1], 0.0);

        // With remove_small_d_limit < 0, unconditional call leaves
        // ThermohalineMixing even at thermohaline_coeff = 0.0.
        let mut model_thrm_nolim = build_bd_k375_k376_model();
        model_thrm_nolim.controls.use_ledoux_criterion = true;
        model_thrm_nolim.controls.thermohaline_coeff = 0.0;
        model_thrm_nolim.controls.mix_wind.remove_small_d_limit = -1.0;
        model_thrm_nolim
            .aux
            .mixing
            .gradl_composition_term
            .fill(-0.05);
        model_thrm_nolim.lum.fill(1.0e20);
        set_mlt_vars(&mut model_thrm_nolim).expect("thermohaline no limit");
        assert_eq!(
            model_thrm_nolim.aux.mixing.mlt_mixing_type[1],
            MixingType::ThermohalineMixing
        );
        assert_eq!(model_thrm_nolim.mlt_d[1], 0.0);
        assert_eq!(model_thrm_nolim.mlt_vc[1], 0.0);
    });
}

#[test]
fn test_semiconvection_langer85_0_ulp_scalar_and_ad() {
    with_numerics_mode(NumericsMode::Strict, || {
        let l = ad_seed(4.0e37, 4.0e37);
        let lambda = ad_seed(3.0e9, -1.5e9);
        let t = ad_seed(1.8e7, 1.8e7);
        let p = ad_seed(1.2e16, 2.4e16);
        let pr = DualOrder1::from(CRAD / 3.0) * (t * t * t * t);
        let beta = (p - pr) / p;
        let opacity = ad_seed(0.38, -0.76);
        let rho = ad_seed(5.5, 5.5);
        let cp = ad_seed(4.2e8, -1.0e8);
        let grada = ad_seed(0.28, 0.02);
        let gradl_comp = 0.22;
        let gradl = grada + DualOrder1::from(gradl_comp);
        let gradr = ad_seed(0.41, 0.15);

        let inputs_ad = SemiconvectionInputs {
            l,
            lambda,
            t,
            p,
            pr,
            beta,
            opacity,
            rho,
            cp,
            gradr,
            grada,
            gradl,
        };
        let params_gradr = SemiconvectionParams {
            option: SemiconvectionOption::Langer85GradTIsGradr,
            alpha_semiconvection: 0.1,
            cgrav: STANDARD_CGRAV,
            gradl_composition_term: gradl_comp,
            m: 2.0e33,
        };
        let out_lg = set_semiconvection(&params_gradr, &inputs_ad)
            .expect("sc Langer85GradTIsGradr");
        assert_eq!(out_lg.mixing_type, MixingType::SemiconvectiveMixing);
        assert_dual1_mesa("sc_langer85_gradt_is_gradr", "gradT", out_lg.grad_t);
        assert_dual1_mesa(
            "sc_langer85_gradt_is_gradr",
            "Y_face",
            out_lg.y_face,
        );
        assert_dual1_mesa(
            "sc_langer85_gradt_is_gradr",
            "conv_vel",
            out_lg.conv_vel,
        );
        assert_dual1_mesa("sc_langer85_gradt_is_gradr", "D", out_lg.d);

        let params_l85 = SemiconvectionParams {
            option: SemiconvectionOption::Langer85,
            ..params_gradr
        };
        let out_l85 =
            set_semiconvection(&params_l85, &inputs_ad).expect("sc Langer85");
        assert_eq!(out_l85.mixing_type, MixingType::SemiconvectiveMixing);
        assert_dual1_mesa("sc_langer85", "gradT", out_l85.grad_t);
        assert_dual1_mesa("sc_langer85", "Y_face", out_l85.y_face);
        assert_dual1_mesa("sc_langer85", "conv_vel", out_l85.conv_vel);
        assert_dual1_mesa("sc_langer85", "D", out_l85.d);

        // Edge case: alpha_sc = 0 and gradr == gradL (0/0 => NaN diffusivity)
        // must return Ok with NaN D so the caller's !is_finite() guard resets.
        // Departure: MESA 26.4.1 does not take the `D%val <= 0` early return
        // for NaN and reports `mixing_type = 3` (semiconvective) with NaN
        // `conv_vel` (case `sc_langer85_alpha0_gradr_eq_gradl` of the TSV);
        // FlexMESA returns `NoMixing` here.
        let inputs_eq_sc = SemiconvectionInputs {
            gradr: gradl,
            ..inputs_ad
        };
        let params_alpha0 = SemiconvectionParams {
            alpha_semiconvection: 0.0,
            ..params_l85
        };
        let out_alpha0 = set_semiconvection(&params_alpha0, &inputs_eq_sc)
            .expect("sc alpha_sc=0 eq must not Err");
        assert_eq!(out_alpha0.mixing_type, MixingType::NoMixing);
        assert!(out_alpha0.d.val.is_nan());
        assert!(mesa("sc_langer85_alpha0_gradr_eq_gradl", "D", "val").is_nan());
    });
}

#[test]
fn test_thermohaline_prescriptions_0_ulp_scalar_and_ad() {
    with_numerics_mode(NumericsMode::Strict, || {
        let inputs = ThermohalineInputs {
            lambda: ad_seed(4.5e9, -1.25e9),
            grada: ad_seed(0.39, 0.03),
            gradr: ad_seed(0.24, 0.11),
            t: ad_seed(2.5e7, 2.5e7),
            opacity: ad_seed(0.42, -0.2),
            rho: ad_seed(15.0, 15.0),
            cp: ad_seed(3.8e8, -0.5e8),
        };
        let base_params = ThermohalineParams {
            option: ThermohalineOption::Kippenhahn,
            gradl_composition_term: -0.05,
            xh1: 0.68,
            thermohaline_coeff: 2.0,
            species: DrivingSpecies::new(2.0, 3.0),
        };

        let out_kipp =
            set_thermohaline(&base_params, &inputs).expect("Kippenhahn");
        assert_eq!(out_kipp.mixing_type, MixingType::ThermohalineMixing);
        assert_dual1_mesa("thrm_kippenhahn", "D", out_kipp.d);
        assert_dual1_mesa("thrm_kippenhahn", "gradT", out_kipp.grad_t);
        assert_dual1_mesa("thrm_kippenhahn", "Y_face", out_kipp.y_face);
        assert_dual1_mesa("thrm_kippenhahn", "conv_vel", out_kipp.conv_vel);

        let params_tgs11 = ThermohalineParams {
            option: ThermohalineOption::TraxlerGaraudStellmach11,
            ..base_params
        };
        let out_tgs11 =
            set_thermohaline(&params_tgs11, &inputs).expect("TGS11");
        assert_eq!(out_tgs11.mixing_type, MixingType::ThermohalineMixing);
        assert_dual1_mesa("thrm_tgs11", "D", out_tgs11.d);
        assert_dual1_mesa("thrm_tgs11", "conv_vel", out_tgs11.conv_vel);

        let params_bgs13 = ThermohalineParams {
            option: ThermohalineOption::BrownGaraudStellmach13,
            ..base_params
        };
        let out_bgs13 =
            set_thermohaline(&params_bgs13, &inputs).expect("BGS13");
        assert_eq!(out_bgs13.mixing_type, MixingType::ThermohalineMixing);
        // MESA NR stops at Euclidean step <= 1e-13 using LAPACK dgesvx; Cramer
        // 2x2 matches within ~1.5e-14 (< ACY = 1e-13), while conv_vel = 3*D/L
        // preserves exact div_ad AD propagation.
        let exp_bgs13_d = mesa("thrm_bgs13", "D", "val");
        let rel_d = (out_bgs13.d.val - exp_bgs13_d).abs() / exp_bgs13_d;
        assert!(rel_d < 1.0e-13, "BGS13 D rel diff too large: {rel_d}");
        assert_eq!(
            out_bgs13.d.d1[0].to_bits(),
            mesa("thrm_bgs13", "D", "d1").to_bits()
        );
        let exp_bgs13_vc =
            (DualOrder1::from(3.0) * out_bgs13.d) / inputs.lambda;
        assert_eq!(
            out_bgs13.conv_vel.val.to_bits(),
            exp_bgs13_vc.val.to_bits()
        );
        assert_eq!(
            out_bgs13.conv_vel.d1[0].to_bits(),
            exp_bgs13_vc.d1[0].to_bits()
        );
        let oracle_bgs13_vc = mesa("thrm_bgs13", "conv_vel", "val");
        let rel_vc =
            (out_bgs13.conv_vel.val - oracle_bgs13_vc).abs() / oracle_bgs13_vc;
        assert!(rel_vc < 1.0e-13, "BGS13 vc rel diff too large: {rel_vc}");
        let oracle_bgs13_dvc = mesa("thrm_bgs13", "conv_vel", "d1");
        let rel_dvc = (out_bgs13.conv_vel.d1[0] - oracle_bgs13_dvc).abs()
            / oracle_bgs13_dvc;
        assert!(
            rel_dvc < 1.0e-13,
            "BGS13 vc.d1 rel diff too large: {rel_dvc}"
        );
    });
}
