// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` `#156` parity and regression tests for `wd_cool_0.6M`
//! (`phase_separation` + `unpack_xh` at `m=165..169`) and `wd_diffusion`
//! (`m=1408..1409` CGS diffusion and thermodynamic sensitivity).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CRAD, LN10, L_SUN, M_SUN, PI4, STANDARD_CGRAV};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, EosConfig, HelmTable, SkyeSolidMixingRule,
    I_LN_PGAS, I_PHASE,
};
use flexmesa_kap::KapConfig;
use flexmesa_math::{exp, pow, with_numerics_mode, NumericsMode};
use flexmesa_star::micro::{do_eos, set_basic_vars};
use flexmesa_star::{
    apply_step_pre_physics, do_element_diffusion_step,
    do_phase_separation_step, finish_element_diffusion, load_zams_model,
    set_hydro_vars, threshold_smoothing, StarAdExt, StarAdOrder1, StarControls,
    StarJobControls, StarModel,
};

const WD_SPECIES: [&str; 11] = [
    "h1", "he3", "he4", "c12", "n14", "o16", "o18", "ne20", "ne22", "mg24",
    "si28",
];

struct PhaseSepZoneOracle {
    label: &'static str,
    t_bits: u64,
    rho_bits: u64,
    lnd_bits: u64,
    xa_pre_bits: [u64; 11],
    peos_pre_bits: u64,
    ln_pgas_pre_bits: u64,
    phase_pre_bits: u64,
    xa_post_bits: [u64; 11],
    peos_post_bits: u64,
    ln_pgas_post_bits: u64,
    phase_post_bits: u64,
}

// Exact MESA `26.4.1` snapshots from `star/test_suite/wd_cool_0.6M` before
// (`do_phase_separation` input) and after `do_phase_separation` +
// `set_vars_if_needed` -> `unpack_xh` (`evolve.f90:622-627`), where `lnd` and
// `rho` stay at `xh` while `Peos`, `lnPgas`, and `phase` update at fixed
// `(lnd, T, xa_post)`.
const WD_COOL_PHASE_SEP_ORACLES: [PhaseSepZoneOracle; 5] = [
    PhaseSepZoneOracle {
        label: "wd_cool_0.6M/m165_z1250_outside_core",
        t_bits: 0x414C66D5E6A08862,
        rho_bits: 0x414E76722A08F7D0,
        lnd_bits: 0x402E6666F9688481,
        xa_pre_bits: [
            0x37AD6B3A2FEDFDB4,
            0x3B68D3D448626535,
            0x3C804EFB6550C201,
            0x3FD3419208CAAE31,
            0x3CDB15E58F8CA14F,
            0x3FE583C1EF28E699,
            0x3D8E325AC2BB4F81,
            0x3F5FA937A6509778,
            0x3F954817D7CA13A1,
            0x3F49E64A7508F1FC,
            0x3F6AE61F428E528D,
        ],
        peos_pre_bits: 0x44C67B1A3B18D030,
        ln_pgas_pre_bits: 0x404ADB3036AD5C2E,
        phase_pre_bits: 0x3FEC9C4B83525623,
        xa_post_bits: [
            0x37B5A68DA9011190,
            0x3B69B0940920C45A,
            0x3C806E7653CC8945,
            0x3FD3576334CD02F7,
            0x3CDB44400EB39B2A,
            0x3FE5792F54D510B7,
            0x3D8E52F8F01D7028,
            0x3F5FA99850CC797C,
            0x3F953D43392B8CA7,
            0x3F49E6A39EF01321,
            0x3F6AE681EA75CA18,
        ],
        peos_post_bits: 0x44C67B2C70854D73,
        ln_pgas_post_bits: 0x404ADB309E5A15BB,
        phase_post_bits: 0x3FEC5BC4703A3B1C,
    },
    PhaseSepZoneOracle {
        label: "wd_cool_0.6M/m165_z1251_crystal_boundary",
        t_bits: 0x414C682242B2B656,
        rho_bits: 0x414EA051BB2FA598,
        lnd_bits: 0x402E6924E0079EA2,
        xa_pre_bits: [
            0x37AE812125837CAA,
            0x3B694F62A3391AFB,
            0x3C804D764629973F,
            0x3FD3419BCCDAF5C4,
            0x3CDB0A1EE5871DDD,
            0x3FE583B8569FDAC4,
            0x3D8E30FF24BB5677,
            0x3F5FA932B3C87BBB,
            0x3F9548AFAE607925,
            0x3F49E64621F3798C,
            0x3F6AE61A9CCCA37B,
        ],
        peos_pre_bits: 0x44C6A985B332BF04,
        ln_pgas_pre_bits: 0x404ADC3774874C97,
        phase_pre_bits: 0x3FED0C55612E1ADF,
        xa_post_bits: [
            0x37B2C84F45926D1F,
            0x3B6962D82761F3FD,
            0x3C8063C51BB306F6,
            0x3FC8352C0AC702AA,
            0x3CDB33FDFEA68FD0,
            0x3FE91777DB029B4A,
            0x3D8E470CB281ECF7,
            0x3F5FA975DBC0C603,
            0x3F95410D08398F5B,
            0x3F49E68387FE0EC8,
            0x3F6AE65E397E80E1,
        ],
        peos_post_bits: 0x44C6A6582AD480EE,
        ln_pgas_post_bits: 0x404ADC25804BE17B,
        phase_post_bits: 0x3FEFFFDB6DF3637C,
    },
    PhaseSepZoneOracle {
        label: "wd_cool_0.6M/m165_z1267_center",
        t_bits: 0x414C6A48D724066F,
        rho_bits: 0x414EEAD5995E39B1,
        lnd_bits: 0x402E6DFCB67F02C4,
        xa_pre_bits: [
            0x37AAFB7EAB39B96B,
            0x3B68593210E0C33A,
            0x3C804AC4A580A690,
            0x3FD341AE900002AE,
            0x3CDB0D80DD9CE6D1,
            0x3FE583A8A90032C9,
            0x3D8E2E40A482A423,
            0x3F5FA92B4D1F4958,
            0x3F95497AD99DC941,
            0x3F49E63F2557E50C,
            0x3F6AE612C27F3166,
        ],
        peos_pre_bits: 0x44C6FC69C8CF24F0,
        ln_pgas_pre_bits: 0x404ADE0853122584,
        phase_pre_bits: 0x3FEDB45B13E53CD3,
        xa_post_bits: [
            0x37AAFB7EAB39B96B,
            0x3B68593210E0C33A,
            0x3C804AC4A580A690,
            0x3FC80ED9FA23A2FB,
            0x3CDB0D80DD9CE6D1,
            0x3FE920C972774B61,
            0x3D8E2E40A482A423,
            0x3F5FA92B4D1F4958,
            0x3F95497AD99DC941,
            0x3F49E63F2557E50C,
            0x3F6AE612C27F3166,
        ],
        peos_post_bits: 0x44C6F922F92A1345,
        ln_pgas_post_bits: 0x404ADDF612C22059,
        phase_post_bits: 0x3FEFFFE7DC6A8E11,
    },
    PhaseSepZoneOracle {
        label: "wd_cool_0.6M/m168_z1246_crystal_core",
        t_bits: 0x414ADCBD980F045B,
        rho_bits: 0x414C038F345B70C0,
        lnd_bits: 0x402E3B7EF673458F,
        xa_pre_bits: [
            0x37BBBAFC8CF16553,
            0x3B70763EAD7FC8CB,
            0x3C80B7E2CE3E3837,
            0x3FD38F1765042B84,
            0x3CDBDE6FC908CDD1,
            0x3FE55E49FE1E6F9C,
            0x3D8EB5D952C4A35E,
            0x3F5FAAB328099A8F,
            0x3F951E6C02D35751,
            0x3F49E7AD3269A921,
            0x3F6AE7AA67C5CB1B,
        ],
        peos_pre_bits: 0x44C3D243DA3958CD,
        ln_pgas_pre_bits: 0x404ACB128DBA9747,
        phase_pre_bits: 0x3FEFABD52F240761,
        xa_post_bits: [
            0x37BBBAFC8CF16553,
            0x3B70763EAD7FC8CB,
            0x3C80B7E2CE3E3837,
            0x3FC89B9930CB092D,
            0x3CDBDE6FC908CDD1,
            0x3FE8FEEF646DC312,
            0x3D8EB5D952C4A35E,
            0x3F5FAAB328099A8F,
            0x3F951E6C02D35751,
            0x3F49E7AD3269A921,
            0x3F6AE7AA67C5CB1B,
        ],
        peos_post_bits: 0x44C3CF61E15821E6,
        ln_pgas_post_bits: 0x404ACAFFEEC895C7,
        phase_post_bits: 0x3FEFFFFD24D327B1,
    },
    PhaseSepZoneOracle {
        label: "wd_cool_0.6M/m169_z1240_crystal_boundary",
        t_bits: 0x414AC5AD12007ECD,
        rho_bits: 0x414A21F5F42FAF2D,
        lnd_bits: 0x402E17E77A608E69,
        xa_pre_bits: [
            0x2B617F7D4ED8C33D,
            0x3BA3376EB9CA2461,
            0x3C809936871E0DB2,
            0x3FD3E8FB966C1155,
            0x3CDBE8DD22504283,
            0x3FE53171319AA7FD,
            0x3D8E40742C34267D,
            0x3F5FAA987CFF3522,
            0x3F951B48DB6E1BC3,
            0x3F49E78DE8C4BE4A,
            0x3F6AE78C9AB49EF2,
        ],
        peos_pre_bits: 0x44C1D9C450EED186,
        ln_pgas_pre_bits: 0x404ABDAB1AD3341A,
        phase_pre_bits: 0x3FED259D9026169B,
        xa_post_bits: [
            0x2B617F7D4ED8C33D,
            0x3BA3376EB9CA2461,
            0x3C809936871E0DB2,
            0x3FC941D4A927B8B3,
            0x3CDBE8DD22504283,
            0x3FE8D579D286C27B,
            0x3D8E40742C34267D,
            0x3F5FAA987CFF3522,
            0x3F951B48DB6E1BC3,
            0x3F49E78DE8C4BE4A,
            0x3F6AE78C9AB49EF2,
        ],
        peos_post_bits: 0x44C1D721BB1983B1,
        ln_pgas_post_bits: 0x404ABD98343E82B6,
        phase_post_bits: 0x3FEFFFD924C8B521,
    },
];

fn ulp_distance(a: f64, b: f64) -> u64 {
    let to_ordered = |x: f64| -> u64 {
        let bits = x.to_bits();
        if (bits >> 63) != 0 {
            (!bits).wrapping_add(1)
        } else {
            bits | (1_u64 << 63)
        }
    };
    to_ordered(a).abs_diff(to_ordered(b))
}

fn wd_chem_id() -> Vec<usize> {
    let cat = NuclideCatalog::builtin_ref();
    WD_SPECIES
        .iter()
        .map(|name| cat.get_by_name(name).expect("known nuclide").chem_id)
        .collect()
}

fn assert_within_ulp(label: &str, got: f64, want_bits: u64, max_ulp: u64) {
    let want = f64::from_bits(want_bits);
    let u = ulp_distance(got, want);
    assert!(
        u <= max_ulp,
        "{label}: got={got:.17e} (0x{:016X}) vs want={want:.17e} (0x{want_bits:016X}), ulp={u} > {max_ulp}",
        got.to_bits(),
    );
}

#[test]
fn test_wd_cool_0_6m_m165_169_skye_fixed_lnd_vs_fix_pgas_matches_mesa_26_4_1() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().expect("builtin helm");
        let chem_id = wd_chem_id();
        let mut cfg = EosConfig {
            use_skye: true,
            use_pc: false,
            ..EosConfig::default()
        };
        cfg.skye.skye_solid_mixing_rule = SkyeSolidMixingRule::Ogata;
        cfg.skye.skye_use_ion_offsets = true;

        for o in &WD_COOL_PHASE_SEP_ORACLES {
            let t = f64::from_bits(o.t_bits);
            let rho = f64::from_bits(o.rho_bits);
            let lnd = f64::from_bits(o.lnd_bits);
            let xa_pre: Vec<f64> =
                o.xa_pre_bits.iter().map(|&b| f64::from_bits(b)).collect();
            let xa_post: Vec<f64> =
                o.xa_post_bits.iter().map(|&b| f64::from_bits(b)).collect();

            let prad = CRAD * t * t * t * t / 3.0;

            let out_pre = eos_dt_get(ht, rho, t, &chem_id, &xa_pre, &cfg)
                .expect("eos pre phase sep");
            let peos_pre = exp(out_pre.res[I_LN_PGAS]) + prad;
            assert_within_ulp(
                &format!("{}/pre_lnPgas", o.label),
                out_pre.res[I_LN_PGAS],
                o.ln_pgas_pre_bits,
                8,
            );
            assert_within_ulp(
                &format!("{}/pre_Peos", o.label),
                peos_pre,
                o.peos_pre_bits,
                8,
            );
            assert_within_ulp(
                &format!("{}/pre_phase", o.label),
                out_pre.res[I_PHASE],
                o.phase_pre_bits,
                8,
            );

            // After `do_phase_separation` + `set_vars_if_needed` (`unpack_xh`),
            // MESA `26.4.1` keeps `(lnd, T)` at `xh` and re-evaluates `Skye` at
            // `xa_post`.
            let out_post = eos_dt_get(ht, rho, t, &chem_id, &xa_post, &cfg)
                .expect("eos post phase sep after unpack_xh");
            let peos_post = exp(out_post.res[I_LN_PGAS]) + prad;
            assert_within_ulp(
                &format!("{}/post_lnPgas", o.label),
                out_post.res[I_LN_PGAS],
                o.ln_pgas_post_bits,
                8,
            );
            assert_within_ulp(
                &format!("{}/post_Peos", o.label),
                peos_post,
                o.peos_post_bits,
                8,
            );
            assert_within_ulp(
                &format!("{}/post_phase", o.label),
                out_post.res[I_PHASE],
                o.phase_post_bits,
                8,
            );

            // Without `unpack_xh`, `fix_Pgas = true` inside `phase_separation`
            // holds `lnPgas` fixed at the pre-PS value and shifts `lnd` by
            // `+3.7e-4` inside crystallized zones (`m=165..169`).
            if o.label.contains("crystal_") {
                let log_t = flexmesa_math::log10(t);
                let inv = eos_dt_get_rho(
                    ht,
                    &chem_id,
                    &xa_post,
                    log_t,
                    I_LN_PGAS.cast_signed(),
                    f64::from_bits(o.ln_pgas_pre_bits),
                    lnd / LN10,
                    1.0e-8,
                    1.0e-8 * LN10,
                    100,
                    &cfg,
                )
                .expect("fix_Pgas density solve");
                let fix_pgas_lnd = inv.log10_result * LN10;
                assert!(
                    fix_pgas_lnd - lnd > 3.5e-4,
                    "{}: expected fix_Pgas dlnd > 3.5e-4 without unpack_xh, got {}",
                    o.label,
                    fix_pgas_lnd - lnd
                );
            }
        }
    });
}

fn build_crystallizing_co_wd_with_diffusion() -> StarModel {
    let mut controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        calculate_brunt_b: true,
        use_ledoux_criterion: true,
        do_phase_separation: true,
        do_phase_separation_heating: true,
        phase_separation_option: String::from("CO"),
        do_element_diffusion: true,
        ..StarControls::default()
    };
    controls.diffusion.phase_separation_mixing_use_brunt = true;
    controls.diffusion.diffusion_use_cgs_solver = true;
    controls.diffusion.diffusion_use_caplan = true;
    controls.diffusion.diffusion_use_full_net = true;
    controls.diffusion.diffusion_calculates_ionization = true;
    controls.mix_wind.min_dq = 1.0e-14;

    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load zams base");
    let nz = 16_usize;
    model.resize_zones(nz);
    model.eos_config = EosConfig::default();
    model.eos_config.use_skye = true;
    model.eos_config.use_pc = false;
    model.eos_config.skye.skye_solid_mixing_rule = SkyeSolidMixingRule::Ogata;
    model.eos_config.skye.skye_use_ion_offsets = true;
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

    let cat = NuclideCatalog::builtin_ref();
    let iso = |name: &str| -> usize {
        let cid = cat.get_by_name(name).expect(name).chem_id;
        model.net_iso[cid].expect(name)
    };
    let he4_idx = iso("he4");
    let carbon_idx = iso("c12");
    let oxygen_idx = iso("o16");
    let neon_idx = iso("ne20");

    let sp = model.species;
    model.xa.fill(0.0);
    model.dq.fill(1.0 / (nz as f64));
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();

    // Outer He-rich envelope (`k < 4`) supplies a stabilizing `brunt_B` composition
    // jump that arrests `mix_outward` above surface `k = 0`; inner core (`k >= 10`)
    // crosses `phase > 0.9` with `c12 + o16 > 0.9` (`wd_cool_0.6M` `m=165..169`
    // regime) so `do_phase_separation_step` and `do_element_diffusion_step` run.
    for k in 0..nz {
        let kf = k as f64;
        model.lnd[k] = flexmesa_math::log(1.8e6 * (1.0 + 0.14 * kf));
        model.ln_t[k] = flexmesa_math::log(4.4e6 - 5.5e4 * kf);
        if k < 4 {
            model.xa[k * sp + he4_idx] = 0.85;
            model.xa[k * sp + carbon_idx] = 0.10;
            model.xa[k * sp + oxygen_idx] = 0.03;
            model.xa[k * sp + neon_idx] = 0.02;
        } else if k < 10 {
            model.xa[k * sp + carbon_idx] = 0.42;
            model.xa[k * sp + oxygen_idx] = 0.55;
            model.xa[k * sp + neon_idx] = 0.03;
        } else {
            model.xa[k * sp + carbon_idx] = 0.30;
            model.xa[k * sp + oxygen_idx] = 0.67;
            model.xa[k * sp + neon_idx] = 0.03;
        }
    }

    let mut r3_run = 0.0_f64;
    for k in (0..nz).rev() {
        let rho_k = exp(model.lnd[k]);
        r3_run += (3.0 * model.dm[k]) / (PI4 * rho_k);
        model.ln_r[k] = flexmesa_math::log(pow(r3_run, 1.0 / 3.0));
        model.lum[k] = 1.0e-3 * L_SUN * model.q[k];
    }
    model.cgrav.fill(STANDARD_CGRAV);
    model.xa_start.clone_from(&model.xa);
    model.dt = 5.0e14;
    model.pack_xh();
    set_hydro_vars(&mut model, false, false).expect("initial set_hydro_vars");
    model.aux.need_to_setvars = false;
    model
}

#[test]
fn test_apply_step_pre_physics_phase_separation_unpacks_xh_before_diffusion() {
    with_numerics_mode(NumericsMode::Strict, || {
        let base = build_crystallizing_co_wd_with_diffusion();
        let dt = base.dt;
        let nz = base.nz;
        let xh_lnd: Vec<f64> = (0..nz)
            .map(|k| base.xh[k * base.layout.nvar_hydro + base.layout.i_lnd])
            .collect();

        // 1. Unit contract (`phase_separation.f90:355-361`):
        // `do_phase_separation_step` updates `lnd` under `fix_Pgas = true`.
        let mut ps_direct = base.clone();
        do_phase_separation_step(&mut ps_direct, dt)
            .expect("do_phase_separation_step");
        let max_dlnd_direct = (0..nz)
            .map(|k| (ps_direct.lnd[k] - xh_lnd[k]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dlnd_direct > 1.0e-4,
            "expected fix_Pgas density shift > 1e-4 in core, got {max_dlnd_direct}"
        );

        // 2. Full `apply_step_pre_physics` (`evolve.f90:622-627` ->
        // `hydro_vars.f90:226`): unpacks `xh` back into `lnd` after phase
        // separation and before `do_element_diffusion_step`.
        let mut stepped = base.clone();
        apply_step_pre_physics(&mut stepped, dt)
            .expect("apply_step_pre_physics");
        for k in 0..nz {
            assert_eq!(
                stepped.lnd[k].to_bits(),
                xh_lnd[k].to_bits(),
                "zone {k}: lnd must match xh after phase separation + unpack_xh"
            );
        }
        assert!(
            stepped.aux.crystal_core_boundary_mass > 0.0,
            "expected crystallization boundary mass > 0"
        );
        assert!(
            stepped.aux.eps_phase_separation.iter().any(|&e| e != 0.0),
            "expected nonzero eps_phase_separation"
        );

        // 3. Verify `apply_step_pre_physics` equals explicit `unpack_xh` seq
        // bit-for-bit across `lnd`, `peos`, `ln_pgas`, and all `xa`.
        let mut ref_seq = base.clone();
        ref_seq.aux.phase_sep_mixing_mass = -1.0;
        do_phase_separation_step(&mut ref_seq, dt).expect("ref ps");
        ref_seq.unpack_xh();
        set_hydro_vars(&mut ref_seq, false, false).expect("ref hydro post ps");
        ref_seq.aux.mixing.okay_to_set_mixing_info = false;
        do_element_diffusion_step(&mut ref_seq, dt).expect("ref diff");
        set_hydro_vars(&mut ref_seq, false, false)
            .expect("ref hydro post diff");
        finish_element_diffusion(&mut ref_seq, dt);

        assert_eq!(
            stepped.aux.crystal_core_boundary_mass.to_bits(),
            ref_seq.aux.crystal_core_boundary_mass.to_bits()
        );
        for k in 0..nz {
            assert_eq!(stepped.peos[k].to_bits(), ref_seq.peos[k].to_bits());
            assert_eq!(
                stepped.ln_pgas[k].to_bits(),
                ref_seq.ln_pgas[k].to_bits()
            );
        }
        for i in 0..stepped.xa.len() {
            assert_eq!(stepped.xa[i].to_bits(), ref_seq.xa[i].to_bits());
        }

        // 4. Contrast against omitting `unpack_xh()`: leaves `fix_Pgas` `lnd`
        // and shifts both `Peos` and post-diffusion `xa`.
        let mut no_unpack = base;
        no_unpack.aux.phase_sep_mixing_mass = -1.0;
        do_phase_separation_step(&mut no_unpack, dt).expect("no_unpack ps");
        set_hydro_vars(&mut no_unpack, false, false)
            .expect("no_unpack hydro post ps");
        no_unpack.aux.mixing.okay_to_set_mixing_info = false;
        do_element_diffusion_step(&mut no_unpack, dt).expect("no_unpack diff");
        set_hydro_vars(&mut no_unpack, false, false)
            .expect("no_unpack hydro post diff");
        finish_element_diffusion(&mut no_unpack, dt);

        let max_dp_rel_err = (0..nz)
            .map(|k| {
                ((no_unpack.peos[k] - stepped.peos[k]) / stepped.peos[k]).abs()
            })
            .fold(0.0_f64, f64::max);
        let max_dxa_err = no_unpack
            .xa
            .iter()
            .zip(stepped.xa.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dp_rel_err > 1.0e-4 && max_dxa_err > 1.0e-11,
            "expected missing unpack_xh to perturb Peos ({max_dp_rel_err:.3e}) and post-diff xa ({max_dxa_err:.3e})"
        );
    });
}

struct WdDiffZoneOracle {
    label: &'static str,
    lnd_bits: u64,
    ln_t_bits: u64,
    peos_bits: u64,
    gam_bits: u64,
    eta_bits: u64,
    xa_bits: [u64; 11],
}

// Exact MESA `26.4.1` transition-zone states from `star/test_suite/wd_diffusion`
// (`m=1408 z=725` H/He transition, `m=1409 z=887` He/CO boundary).
const WD_DIFFUSION_ORACLES: [WdDiffZoneOracle; 2] = [
    WdDiffZoneOracle {
        label: "wd_diffusion/m1408_z725",
        lnd_bits: 0x401A1BB70501E739,
        ln_t_bits: 0x4030AEA87F39C421,
        peos_bits: 0x43AF3B4A7A6A2D78,
        gam_bits: 0x3FC904F213395FF9,
        eta_bits: 0xBFB1E7C76FD159F4,
        xa_bits: [
            0x3FD0A23D7BA8D623,
            0x3EBFAD2F054F6300,
            0x3FE6E2360347CACD,
            0x3F07E4BD05A4BB6A,
            0x3F8318442E3EC5C9,
            0x3F811E8EFB74E75F,
            0x3EFACAF4F68B5B8D,
            0x3F630A81D75A067F,
            0x3F2818A2BE30F1D8,
            0x3F4E3F10E370E09D,
            0x3F6F1B037CFDCB58,
        ],
    },
    WdDiffZoneOracle {
        label: "wd_diffusion/m1409_z887",
        lnd_bits: 0x402098932DA7BEEB,
        ln_t_bits: 0x40310FF8E5B9B3A3,
        peos_bits: 0x43DA4ED62997891B,
        gam_bits: 0x3FD7BC43476FC80E,
        eta_bits: 0x3FF4BFB56F417B90,
        xa_bits: [
            0x2B617F7D4ED8C33D,
            0x2B617F7D4ED8C33D,
            0x3FEF62BBB930F113,
            0x3F2912C8E1A7ED44,
            0x3F8A28C7B254F116,
            0x3F202931ED964944,
            0x3EF5D5AF24944390,
            0x3F5F53C1F7B35BC6,
            0x3F2430075073A2D5,
            0x3F49BFFE9FE47699,
            0x3F6A84DBE1A415D1,
        ],
    },
];

#[test]
fn test_wd_diffusion_m1408_1409_states_match_mesa_26_4_1_and_cgs_sensitivity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let chem_id = wd_chem_id();
        let sp = chem_id.len();
        let mut eos_model = StarModel::new(
            WD_DIFFUSION_ORACLES.len(),
            chem_id,
            StarJobControls::default(),
            StarControls::default(),
        );
        eos_model.eos_config = EosConfig::default();
        eos_model.dq.fill(0.5);
        eos_model.set_qs();
        for (k, o) in WD_DIFFUSION_ORACLES.iter().enumerate() {
            eos_model.lnd[k] = f64::from_bits(o.lnd_bits);
            eos_model.ln_t[k] = f64::from_bits(o.ln_t_bits);
            for (j, &bits) in o.xa_bits.iter().enumerate() {
                eos_model.xa[k * sp + j] = f64::from_bits(bits);
            }
        }
        set_basic_vars(&mut eos_model).expect("set_basic_vars wd_diff");
        do_eos(&mut eos_model).expect("do_eos wd_diff");
        for (k, o) in WD_DIFFUSION_ORACLES.iter().enumerate() {
            assert_within_ulp(
                &format!("{}/peos", o.label),
                eos_model.peos[k],
                o.peos_bits,
                8,
            );
            assert_within_ulp(
                &format!("{}/gam", o.label),
                eos_model.gam[k],
                o.gam_bits,
                8,
            );
            assert_within_ulp(
                &format!("{}/eta", o.label),
                eos_model.eta[k],
                o.eta_bits,
                16,
            );
        }

        // Verify strict determinism and density vs `gradt` sensitivity across
        // the `eta`-crossing CGS diffusion regime (`m=1408..1409` envelope).
        let mut controls = StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            do_element_diffusion: true,
            ..StarControls::default()
        };
        controls.diffusion.diffusion_use_cgs_solver = true;
        controls.diffusion.diffusion_use_full_net = true;
        controls.diffusion.diffusion_calculates_ionization = true;
        controls.diffusion.diffusion_min_dq_ratio_at_surface = 0.0;
        let mut diff_base =
            load_zams_model(controls, StarJobControls::default())
                .expect("load zams base");
        let nz = 12_usize;
        diff_base.resize_zones(nz);
        diff_base.eos_config = EosConfig::default();
        diff_base.kap_config = KapConfig {
            use_type2_opacities: true,
            zbase: 0.02,
            ..diff_base.kap_config
        };
        diff_base.mstar = 0.61 * M_SUN;
        diff_base.m_center = 0.609 * M_SUN;
        diff_base.xmstar = diff_base.mstar - diff_base.m_center;
        diff_base.r_center = 7.5e8;
        diff_base.dq.fill(1.0 / (nz as f64));
        diff_base.set_qs();
        diff_base.set_xqs();
        diff_base.set_m_and_dm();

        let cat = NuclideCatalog::builtin_ref();
        let iso = |name: &str| -> usize {
            let cid = cat.get_by_name(name).expect(name).chem_id;
            diff_base.net_iso[cid].expect(name)
        };
        let h1_idx = iso("h1");
        let he4_idx = iso("he4");
        let carbon_idx = iso("c12");
        let oxygen_idx = iso("o16");
        let d_sp = diff_base.species;
        diff_base.xa.fill(0.0);
        for k in 0..nz {
            let frac = (k as f64) / ((nz - 1) as f64);
            diff_base.lnd[k] = flexmesa_math::log(5.0e2 + 3.0e4 * frac);
            diff_base.ln_t[k] = flexmesa_math::log(1.6e7 + 6.0e6 * frac);
            if k < 5 {
                diff_base.xa[k * d_sp + h1_idx] = 0.30;
                diff_base.xa[k * d_sp + he4_idx] = 0.68;
                diff_base.xa[k * d_sp + carbon_idx] = 0.01;
                diff_base.xa[k * d_sp + oxygen_idx] = 0.01;
            } else {
                diff_base.xa[k * d_sp + h1_idx] = 1.0e-7;
                diff_base.xa[k * d_sp + he4_idx] = 0.98 - 1.0e-7;
                diff_base.xa[k * d_sp + carbon_idx] = 0.01;
                diff_base.xa[k * d_sp + oxygen_idx] = 0.01;
            }
        }
        let mut r3_run = pow(diff_base.r_center, 3.0);
        for k in (0..nz).rev() {
            r3_run += (3.0 * diff_base.dm[k]) / (PI4 * exp(diff_base.lnd[k]));
            diff_base.ln_r[k] = flexmesa_math::log(pow(r3_run, 1.0 / 3.0));
            diff_base.lum[k] = 1.0e-6 * L_SUN * diff_base.q[k];
        }
        diff_base.cgrav.fill(STANDARD_CGRAV);
        diff_base.dt = 2.0e12;
        diff_base.pack_xh();
        set_hydro_vars(&mut diff_base, false, false).expect("diff hydro");
        assert!(
            diff_base.d_mix.iter().all(|&d| d == 0.0),
            "expected radiative WD envelope with zero convective mixing"
        );

        let dt = diff_base.dt;
        let mut run_a = diff_base.clone();
        let mut run_b = diff_base.clone();
        do_element_diffusion_step(&mut run_a, dt).expect("diff a");
        do_element_diffusion_step(&mut run_b, dt).expect("diff b");
        assert_eq!(run_a.xa, run_b.xa, "CGS diffusion must be bit-identical");
        let diffused = run_a
            .xa
            .iter()
            .zip(diff_base.xa.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            diffused > 1.0e-4,
            "expected active diffusion, got {diffused}"
        );

        // Verify strict determinism and `2e-10` `rho`/`lnd` sensitivity across
        // the `eta`-crossing CGS diffusion regime (`m=1408..1409` envelope).
        let mut pert_rho = diff_base;
        for k in 0..nz {
            pert_rho.rho[k] *= 1.0 + 2.0e-10;
            pert_rho.lnd[k] = flexmesa_math::log(pert_rho.rho[k]);
        }
        do_element_diffusion_step(&mut pert_rho, dt).expect("diff rho");
        let max_shift = run_a
            .xa
            .iter()
            .zip(pert_rho.xa.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_shift > 0.0,
            "rho/lnd perturbation must propagate through CGS diffusion"
        );
    });
}

#[test]
fn test_phase_separation_persists_unsmoothed_brunt_b_and_smooths_brunt_b() {
    with_numerics_mode(NumericsMode::Strict, || {
        let mut no_smooth = build_crystallizing_co_wd_with_diffusion();
        let nz = no_smooth.nz;
        let dt = no_smooth.dt;
        no_smooth.controls.mix_wind.num_cells_for_smooth_brunt_b = 0;

        let mut with_smooth = no_smooth.clone();
        with_smooth.controls.mix_wind.num_cells_for_smooth_brunt_b = 2;
        with_smooth.controls.mix_wind.threshold_for_smooth_brunt_b = 0.0;

        // Seed stale sentinels so we verify `unsmoothed_brunt_b` is updated
        // across every zone re-evaluated by `do_brunt_B`.
        no_smooth.aux.mixing.unsmoothed_brunt_b.fill(-999.0);
        with_smooth.aux.mixing.unsmoothed_brunt_b.fill(-999.0);

        do_phase_separation_step(&mut no_smooth, dt)
            .expect("phase separation without brunt smoothing");
        do_phase_separation_step(&mut with_smooth, dt)
            .expect("phase separation with brunt smoothing");

        // `mix_outward` must use `unsmoothed_brunt_b`, keeping final
        // composition, boundaries, and unsmoothed Brunt-B bit-identical.
        assert_eq!(
            with_smooth.aux.crystal_core_boundary_mass.to_bits(),
            no_smooth.aux.crystal_core_boundary_mass.to_bits()
        );
        assert_eq!(
            with_smooth.aux.phase_sep_mixing_mass.to_bits(),
            no_smooth.aux.phase_sep_mixing_mass.to_bits()
        );
        for (i, (&a, &b)) in
            with_smooth.xa.iter().zip(no_smooth.xa.iter()).enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "xa[{i}] mismatch");
        }
        for k in 0..nz {
            assert_eq!(
                with_smooth.aux.mixing.unsmoothed_brunt_b[k].to_bits(),
                no_smooth.aux.mixing.unsmoothed_brunt_b[k].to_bits(),
                "unsmoothed_brunt_b[{k}] must match regardless of smoothing"
            );
        }

        // At least the crystallization/mixing front must overwrite the `-999`
        // sentinel, and `no_smooth` must leave `brunt_b == unsmoothed_brunt_b`
        // on those refreshed zones while `with_smooth` smooths `brunt_b[..nz]`.
        let mut refreshed = 0_usize;
        let mut max_smooth_diff = 0.0_f64;
        for k in 0..nz {
            let ub = no_smooth.aux.mixing.unsmoothed_brunt_b[k];
            if ub.to_bits() != (-999.0_f64).to_bits() {
                refreshed += 1;
                assert_eq!(no_smooth.brunt_b[k].to_bits(), ub.to_bits());
            }
            max_smooth_diff = max_smooth_diff
                .max((with_smooth.brunt_b[k] - no_smooth.brunt_b[k]).abs());
        }
        assert!(refreshed > 0, "expected refreshed unsmoothed_brunt_b zones");
        assert!(
            max_smooth_diff > 1.0e-12,
            "expected threshold_smoothing to smooth model.brunt_b, diff={max_smooth_diff}"
        );

        // Verify exact `threshold_smoothing` semantics on single-pass input
        // matching `brunt.f90:100-103`.
        let mut expected = no_smooth.brunt_b.clone();
        let mut work = vec![0.0_f64; nz];
        threshold_smoothing(&mut expected, 0.0, 2, false, &mut work);
        assert_eq!(expected[0].to_bits(), no_smooth.brunt_b[0].to_bits());
        assert_eq!(
            expected[nz - 1].to_bits(),
            no_smooth.brunt_b[nz - 1].to_bits()
        );
    });
}

#[test]
fn test_phase_separation_set_micro_vars_updates_csound_start_and_p_face_ad() {
    with_numerics_mode(NumericsMode::Strict, || {
        let mut model = build_crystallizing_co_wd_with_diffusion();
        let nz = model.nz;
        let dt = model.dt;

        // Zone 0 kept positive to check `< 0` guard (`micro.f90:85`); zones
        // `1..nz` negative to verify `csound_start(k) = csound(k)`.
        model.aux.csound_start.fill(-1.0);
        model.aux.csound_start[0] = 1.2345e8;
        model.aux.p_face_ad.fill(StarAdOrder1::ZERO);
        model.controls.u_flag = false;

        do_phase_separation_step(&mut model, dt)
            .expect("phase separation with u_flag = false");

        assert_eq!(model.aux.csound_start[0].to_bits(), 1.2345e8_f64.to_bits());
        // Unmixed outer He envelope zones (`k = 1..3`) are only touched by the
        // final full-star `update_model_range`, so `csound_start` equals `csound`;
        // deeper zones (`4..nz`) latch on their first `update_model_range` pass.
        for k in 1..3 {
            assert_eq!(
                model.aux.csound_start[k].to_bits(),
                model.csound[k].to_bits(),
                "zone {k}: csound_start must update to post-fix_Pgas csound"
            );
        }
        for (k, &cs) in
            model.aux.csound_start.iter().enumerate().take(nz).skip(1)
        {
            assert!(
                cs > 0.0 && cs.is_finite(),
                "zone {k}: negative csound_start must latch to finite csound"
            );
        }
        assert_eq!(
            model.aux.p_face_ad[0].val.to_bits(),
            model.peos[0].to_bits(),
            "surface p_face_ad[0] must equal peos[0]"
        );
        for k in 1..nz {
            let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
            let beta = 1.0 - alfa;
            let want = alfa * model.peos[k] + beta * model.peos[k - 1];
            assert_eq!(
                model.aux.p_face_ad[k].val.to_bits(),
                want.to_bits(),
                "face {k}: p_face_ad.val must match micro.f90:96-99"
            );
        }

        // Verify once `csound_start >= 0`, a subsequent `do_phase_separation_step`
        // call leaves `csound_start` bit-identical (`micro.f90:85`).
        let latched_csound_start = model.aux.csound_start.clone();
        model.aux.crystal_core_boundary_mass = 0.0;
        do_phase_separation_step(&mut model, dt).expect(
            "second phase separation must preserve latched csound_start",
        );
        for (k, (&after, &before)) in model
            .aux
            .csound_start
            .iter()
            .zip(latched_csound_start.iter())
            .enumerate()
        {
            assert_eq!(
                after.to_bits(),
                before.to_bits(),
                "zone {k}: csound_start >= 0 must not overwrite"
            );
        }

        // Verify `!u_flag` guard leaves `p_face_ad` untouched when `u_flag = true`
        // (`micro.f90:93, 99`).
        let mut m_uflag = build_crystallizing_co_wd_with_diffusion();
        m_uflag.controls.u_flag = true;
        for p in &mut m_uflag.aux.p_face_ad {
            p.val = -42.0;
        }
        do_phase_separation_step(&mut m_uflag, dt)
            .expect("phase separation with u_flag = true");
        for (k, p) in m_uflag.aux.p_face_ad.iter().enumerate() {
            assert_eq!(
                p.val.to_bits(),
                (-42.0_f64).to_bits(),
                "face {k}: p_face_ad must stay untouched when u_flag = true"
            );
        }
    });
}
