// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA (release `26.4.1`) parity and unit regression tests
//! for `#463` covering:
//! - Standard metal mixture normalization using complement `ztotal`
//!   (`star/private/adjust_xyz.f90:1082,1147`),
//! - `LINA` turbulent luminosity preservation at `i <= ibotom`
//!   (`star/private/rsp_lina.f90:497-570`) and IEEE-754 evaluation order in
//!   `rsp_turb_partials` (`star/private/rsp_relax_env.f90:705,762,824-834`),
//! - `set_star_vars_rsp`, `finish_after_build_model`, and `copy_results`
//!   (`star/private/rsp_def.f90:440-449,710-740`) `vol_rsp`/`rmid`/`erad`/`prad`
//!   restoration,
//! - GCC `-O2` integer-power addition-chain parity (`T**4` via `powi_f90`,
//!   `r**6` via `(r^3)^2`, `r**5` via `(r*r^2)*r^2`) in
//!   `star/private/rsp_eval_eos_and_kap.f90:245,281` and
//!   `star/private/rsp_step.f90:1642,1694,1699`,
//! - 10-period `star/rsp_Type_II_Cepheid` nonlinear pulsation parity from a
//!   freshly built model (`do_rsp_build_model`) initialized with the upstream
//!   post-build `RANMAR` state `(10, 43)`.

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_const::{CRAD, PI};
use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_star::rsp_def::RspDerivedConsts;
use flexmesa_star::rsp_eos_kap::{
    compute_standard_metals_xa, eval1_mesa_eos_and_kap, RspChemCache,
};
use flexmesa_star::rsp_grid::{
    cleanup_for_lina, finish_after_build_model, set_star_vars_rsp, RspEnvGrid,
};
use flexmesa_star::rsp_hyd_turb::calc_chi;
use flexmesa_star::rsp_hyd_work::RspHydWork;
use flexmesa_star::rsp_lina::do_lina;
use flexmesa_star::rsp_relax_env::relax_env;
use flexmesa_star::rsp_stah::{stah, znvar};
use flexmesa_star::rsp_turb_partials::RspTurbPartials;
use flexmesa_star::rsp_zone_micro::{LcFilterMode, RspZoneMicro};
use flexmesa_star::{
    do_rsp_build_model, get_or_load_net, read_mod_str, rsp_setup_model,
    star_evolve_step, StarControls, StarJobControls, StarModel,
};

fn golden_rsp_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("tests")
        .join("fixtures")
        .join("golden")
        .join("rsp")
}

fn a09_kap(zbase: f64) -> KapConfig {
    KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase,
        ..Default::default()
    }
}

fn base_job(create: bool) -> StarJobControls {
    StarJobControls {
        create_rsp_model: create,
        initial_zfracs: 6,
        set_initial_age: true,
        initial_age: 0.0,
        set_initial_model_number: true,
        initial_model_number: 0,
        set_initial_cumulative_energy_error: true,
        new_cumulative_energy_error: 0.0,
        ..Default::default()
    }
}

fn t2c_controls() -> StarControls {
    StarControls {
        rsp_mass: 0.55,
        rsp_teff: 6410.0,
        rsp_l: 136.0,
        rsp_x: 0.76,
        rsp_z: 0.0001,
        zbase: 0.0001,
        rsp_target_steps_per_cycle: 1200,
        rsp: flexmesa_star::RspConfig {
            do_check_omega: false,
            alfam: 0.05,
            kick_vsurf_km_per_sec: 4.5,
            nz: 200,
            nz_outer: 60,
            t_anchor: 11.0e3,
            min_deltar_for_periods: 0.2,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn delta_scuti_controls() -> StarControls {
    StarControls {
        rsp_mass: 2.0,
        rsp_teff: 6900.0,
        rsp_l: 30.0,
        rsp_x: 0.70,
        rsp_z: 0.02,
        zbase: 0.02,
        rsp: flexmesa_star::RspConfig {
            alfam: 0.25,
            alfa: 1.2,
            kick_vsurf_km_per_sec: 3.0,
            fraction_1st_overtone: 0.5,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn history_metrics(m: &StarModel) -> (f64, f64, f64) {
    let log_l = flexmesa_math::log10(m.surface_l_lsun().max(1.0e-99));
    let log_teff = flexmesa_math::log10(m.teff.max(1.0e-99));
    let log_r = flexmesa_math::log10(m.surface_r_rsun().max(1.0e-99));
    (log_l, log_teff, log_r)
}

#[test]
fn test_rsp_compute_standard_metals_xa_matches_mesa_complement_ztotal() {
    let gdir = golden_rsp_dir();
    let t2c_mod =
        fs::read_to_string(gdir.join("rsp_Type_II_Cepheid_init_26_4_1.mod"))
            .expect("read t2c init mod");
    let ds_mod =
        fs::read_to_string(gdir.join("rsp_Delta_Scuti_init_26_4_1.mod"))
            .expect("read delta scuti init mod");

    let t2c_ctrl = t2c_controls();
    let ds_ctrl = delta_scuti_controls();
    let m_t2c = read_mod_str(&t2c_mod, t2c_ctrl.clone(), base_job(false))
        .expect("parse t2c mod");
    let m_ds = read_mod_str(&ds_mod, ds_ctrl.clone(), base_job(false))
        .expect("parse ds mod");

    let xa_t2c = compute_standard_metals_xa(
        &m_t2c.chem_id,
        t2c_ctrl.rsp_x,
        t2c_ctrl.rsp_z,
        -1.0,
        6,
        t2c_ctrl.rsp.initial_dump_missing_heaviest,
    )
    .expect("t2c metals");
    let xa_ds = compute_standard_metals_xa(
        &m_ds.chem_id,
        ds_ctrl.rsp_x,
        ds_ctrl.rsp_z,
        -1.0,
        6,
        ds_ctrl.rsp.initial_dump_missing_heaviest,
    )
    .expect("ds metals");

    for (j, (&got, &want)) in
        xa_t2c.iter().zip(m_t2c.cell_xa(0).iter()).enumerate()
    {
        let diff = (got - want).abs();
        assert!(
            diff <= 1.0e-15,
            "t2c species {j} xa mismatch: got={got:.17e} want={want:.17e} diff={diff:.3e}"
        );
    }
    for (j, (&got, &want)) in
        xa_ds.iter().zip(m_ds.cell_xa(0).iter()).enumerate()
    {
        let diff = (got - want).abs();
        assert!(
            diff <= 1.0e-15,
            "delta_scuti species {j} xa mismatch: got={got:.17e} want={want:.17e} diff={diff:.3e}"
        );
    }
}

fn build_lina_turb_fixture(
    dc: &RspDerivedConsts,
    nzn: usize,
) -> (RspEnvGrid, RspZoneMicro) {
    let mut grid = RspEnvGrid::new(nzn);
    let mut mc = RspZoneMicro::new(nzn);
    for i in 1..=nzn {
        let fi = f64::from(u32::try_from(i).expect("zone index fits u32"));
        grid.r[i] = 1.0e11 + fi * 3.0e9;
        grid.vol[i] = 8.0e2 / (1.0 + 0.05 * fi);
        grid.t[i] = 6.0e4 - 2.0e3 * fi;
        grid.dm[i] = 2.0e26;
        grid.dm_bar[i] = 2.0e26;
        grid.m[i] = 1.0e33;
        grid.et[i] = if i > dc.ibotom && i < nzn {
            1.0e10 + fi * 5.0e8
        } else {
            0.0
        };
        mc.p[i] = 1.0e8 - fi * 1.0e6;
        mc.cps[i] = 2.5e8;
        mc.qqs[i] = 1.1e-4;
        mc.k[i] = 0.45;
        mc.dvr[i] = 1.0;
        mc.dvrm[i] = -0.8;
        mc.dcp_dr_00[i] = 1.0e-4;
        mc.dcp_dr_in[i] = -0.5e-4;
        mc.dcp_dt_00[i] = 1.0e3;
        mc.dk_dv_00[i] = 2.0e-3;
        mc.dk_dt_00[i] = -1.0e-5;
    }
    (grid, mc)
}

#[test]
fn test_rsp_turb_partials_lina_preserves_ibotom_lt_and_eval_order() {
    let nzn = 60_usize;
    let cfg = flexmesa_star::RspConfig {
        alfa: 1.5,
        alfap: 0.0,
        alfat: 0.01,
        alfac: 1.0,
        alfas: 1.0,
        alfam: 0.15,
        gammar: 0.25,
        ..Default::default()
    };
    let dc = RspDerivedConsts::from_config(&cfg, nzn);
    assert!(dc.ibotom >= 1 && dc.ibotom + 2 < nzn);

    let (mut grid, mc) = build_lina_turb_fixture(&dc, nzn);

    let efl02 = dc.efl0 * dc.efl0;
    let mut tb = RspTurbPartials::new(nzn);
    tb.evaluate(
        &dc,
        cfg.alfa,
        cfg.alfap,
        cfg.alfat,
        efl02,
        &mut grid,
        &mc,
        LcFilterMode::Lina { efl02 },
    );

    // `star/private/rsp_lina.f90:547-568` computes `Lt(IBOTOM)` using `Et(IBOTOM)=0`
    // and `Et(IBOTOM+1)>0` without clamping `Lt(I)` at `I == IBOTOM`.
    let ib = dc.ibotom;
    assert!(
        tb.lt[ib].abs() > 0.0 && tb.lt[ib].is_finite(),
        "expected non-zero finite Lt[ibotom={ib}] in LINA mode, got {}",
        tb.lt[ib]
    );
    assert_eq!(tb.dltzp[nzn - 1], 0.0);

    // Verify `star/private/rsp_relax_env.f90:824-827` exact `Lc` evaluation order.
    let i_chk = ib + 2;
    let vsum = grid.vol[i_chk] + grid.vol[i_chk + 1];
    let pom_v = 2.0 / vsum;
    let qc_i = mc.qqs[i_chk] / mc.cps[i_chk];
    let qc_p = mc.qqs[i_chk + 1] / mc.cps[i_chk + 1];
    let igr1 = f64::midpoint(qc_i, qc_p) * (mc.p[i_chk + 1] - mc.p[i_chk])
        - (flexmesa_math::intrinsic_log(grid.t[i_chk + 1])
            - flexmesa_math::intrinsic_log(grid.t[i_chk]));
    let igr2 = 4.0
        * PI
        * (grid.r[i_chk] * grid.r[i_chk])
        * grid.hp_face[i_chk]
        * pom_v
        / grid.dm_bar[i_chk];
    let pii_exp = (dc.alfas * cfg.alfa)
        * f64::midpoint(mc.cps[i_chk], mc.cps[i_chk + 1])
        * (igr1 * igr2);
    let lc_prefactor = 4.0
        * PI
        * (grid.r[i_chk] * grid.r[i_chk])
        * (grid.t[i_chk] / grid.vol[i_chk]
            + grid.t[i_chk + 1] / grid.vol[i_chk + 1])
        * 0.5
        * (dc.alfac / dc.alfas);
    let sqrt_et_avg =
        f64::midpoint(grid.et[i_chk].sqrt(), grid.et[i_chk + 1].sqrt());
    assert_eq!(
        tb.lc[i_chk].to_bits(),
        (lc_prefactor * pii_exp * sqrt_et_avg).to_bits(),
        "Lc[i] must match upstream star/private/rsp_relax_env.f90:824-827 bit-for-bit"
    );

    // Verify `star/private/rsp_lina.f90:594` `(R**6)` uses GCC `(r^3)*(r^3)` addition chain.
    let r3_i = flexmesa_math::pow3(grid.r[i_chk]);
    let r3_m = flexmesa_math::pow3(grid.r[i_chk - 1]);
    let ev_prefactor =
        (16.0 / 3.0) * PI * cfg.alfa * cfg.alfam * grid.et[i_chk].sqrt();
    let inv_v2_dm = 1.0 / (grid.vol[i_chk] * grid.vol[i_chk] * grid.dm[i_chk]);
    let r6_hp_avg = (r3_i * r3_i + r3_m * r3_m)
        * (grid.hp_face[i_chk] + grid.hp_face[i_chk - 1])
        * 0.25;
    assert_eq!(
        tb.evuu0[i_chk].to_bits(),
        (ev_prefactor * inv_v2_dm * r6_hp_avg / grid.r[i_chk]).to_bits(),
        "EVUU0[i] must use (r^3)^2 for R**6"
    );
}

#[test]
fn test_rsp_grid_invariants_eos_prad_and_calc_chi_match_mesa_26_4_1() {
    let gdir = golden_rsp_dir();
    let raw = fs::read_to_string(gdir.join("rsp_Delta_Scuti_init_26_4_1.mod"))
        .expect("read delta scuti init mod");
    let ctrl = delta_scuti_controls();
    let mut m = read_mod_str(&raw, ctrl.clone(), base_job(false))
        .expect("parse delta scuti mod");
    m.eos_config = EosConfig::default();
    m.kap_config = a09_kap(ctrl.rsp_z);
    rsp_setup_model(&mut m, false, false).expect("rsp setup");

    let nz = m.nz;
    let dc = RspDerivedConsts::from_config(&ctrl.rsp, nz);
    let r_pre = m.r.clone();
    m.aux.rsp.vol_rsp.fill(0.0);
    m.aux.rsp.erad_rsp.fill(0.0);
    set_star_vars_rsp(&mut m, &dc).expect("set_star_vars_rsp");

    // Verify `set_star_vars_rsp` and `finish_after_build_model` invariants
    // (`star/private/rsp_def.f90:440-449,710-740`).
    for k in 0..nz {
        assert_eq!(m.aux.rsp.vol_rsp[k].to_bits(), (1.0 / m.rho[k]).to_bits());
        let r_in = if k + 1 == nz {
            m.r_center
        } else {
            r_pre[k + 1]
        };
        assert_eq!(
            m.rmid[k].to_bits(),
            f64::midpoint(r_pre[k], r_in).to_bits()
        );
        let erad_exp =
            CRAD * flexmesa_math::powi_f90(m.t[k], 4) * m.aux.rsp.vol_rsp[k];
        assert_eq!(m.aux.rsp.erad_rsp[k].to_bits(), erad_exp.to_bits());
    }
    finish_after_build_model(&mut m);
    for k in 0..nz {
        let vol = m.aux.rsp.vol_rsp[k];
        let erad = CRAD * flexmesa_math::powi_f90(m.t[k], 4) * vol;
        assert_eq!(m.aux.rsp.erad_rsp[k].to_bits(), erad.to_bits());
        assert_eq!(
            m.prad[k].to_bits(),
            (m.aux.rsp.f_edd_rsp[k] * erad / vol).to_bits()
        );
    }

    // Verify `eval1_mesa_eos_and_kap` (`star/private/rsp_eval_eos_and_kap.f90:245,281`)
    // computes `Prad = crad * powi_f90(T, 4) / 3` -> `erad = 3 * Prad / rho`.
    let chem = RspChemCache::from_model_zone0(&m).expect("chem cache");
    let t_test = 1.25e5_f64;
    let vol_test = 1.0e4_f64;
    let out =
        eval1_mesa_eos_and_kap(&mut m, &chem, None, true, t_test, vol_test)
            .expect("eval1_mesa_eos_and_kap");
    let prad_exp = CRAD * flexmesa_math::powi_f90(t_test, 4) / 3.0;
    assert_eq!(
        out.d_erad_dv.to_bits(),
        (3.0 * prad_exp).to_bits(),
        "d_erad_dV must use powi_f90(T, 4)"
    );

    // Verify `calc_chi` (`star/private/rsp_step.f90:1642`) uses GCC `(r^3)*(r^3)`.
    m.dt = 10.0;
    let mut wk = RspHydWork::from_model(&m);
    let i_test = dc.ibotom + 5;
    let k0 = nz - i_test;
    wk.w[k0] = 1.0e5;
    m.v[k0] = 2.0e5;
    m.v[k0 + 1] = 1.0e5;
    m.aux.rsp.hp_face[k0] = 1.0e9;
    m.aux.rsp.hp_face[k0 + 1] = 0.9e9;
    calc_chi(&mut m, &dc, &mut wk, i_test);
    let vol = m.aux.rsp.vol_rsp[k0];
    let pom = (16.0 / 3.0) * PI * dc.alfa * dc.alfam / m.dm[k0];
    let pom1 = wk.w[k0] / (vol * vol);
    let r_out_3 = flexmesa_math::pow3(m.r[k0]);
    let r_in_3 = flexmesa_math::pow3(m.r[k0 + 1]);
    let pom2 = f64::midpoint(r_out_3 * r_out_3, r_in_3 * r_in_3);
    let pom4 = f64::midpoint(m.aux.rsp.hp_face[k0], m.aux.rsp.hp_face[k0 + 1]);
    let pom3 = m.v[k0] / m.r[k0] - m.v[k0 + 1] / m.r[k0 + 1];
    let chi_exp = (pom * pom2 * pom3 * pom4) * pom1;
    assert_eq!(
        m.aux.rsp.chi_rsp[k0].to_bits(),
        chi_exp.to_bits(),
        "Chi(k) must match star/private/rsp_step.f90:1642 GCC addition chain"
    );
}

// MESA 26.4.1 post-`RELAX_ENV` `RANMAR` state `(10, 43)` (7071 draws) stored with
// full 17-digit IEEE-754 decimal round-trip precision for exact replay.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
#[rustfmt::skip]
const MESA_T2C_RNG_STATE: [f64; 103] = [
    0.0,
    4.7497272491455078e-2, 5.4045271873474121e-1, 7.3200285434722900e-1, 4.1435885429382324e-1,
    6.7773550748825073e-1, 8.3523869514465332e-2, 4.9337804317474365e-2, 4.6320331096649170e-1,
    8.0310058593750000e-1, 5.3261458873748779e-1, 8.0216389894485474e-1, 8.9462459087371826e-2,
    7.0968627929687500e-1, 8.6966681480407715e-1, 6.3076299428939819e-1, 7.1785861253738403e-1,
    9.4501072168350220e-1, 3.8700777292251587e-1, 4.6900320053100586e-1, 7.5763064622879028e-1,
    9.5749384164810181e-1, 4.2993831634521484e-1, 5.7583642005920410e-1, 1.8884319067001343e-1,
    2.1580457687377930e-2, 4.7084850072860718e-1, 9.3671971559524536e-1, 8.2320749759674072e-1,
    4.9433797597885132e-1, 3.5344064235687256e-1, 7.0401221513748169e-1, 1.9727712869644165e-1,
    6.5202903747558594e-1, 7.0327609777450562e-1, 2.7660846710205078e-1, 7.3337078094482422e-1,
    9.9276632070541382e-1, 3.8407582044601440e-1, 5.1563727855682373e-1, 4.9874275922775269e-1,
    6.8588614463806152e-2, 4.1133624315261841e-1, 6.6186249256134033e-2, 9.6953481435775757e-1,
    9.0078157186508179e-1, 9.6055686473846436e-1, 9.0585631132125854e-1, 9.6893775463104248e-1,
    1.5103781223297119e-1, 2.3048931360244751e-1, 6.9573450088500977e-1, 8.8094532489776611e-1,
    6.6375631093978882e-1, 5.6535482406616211e-1, 3.4632694721221924e-1, 9.7119218111038208e-1,
    5.0120568275451660e-1, 5.8320367336273193e-1, 6.0023248195648193e-2, 3.4505540132522583e-1,
    9.0215027332305908e-2, 1.8698966503143311e-1, 6.6455841064453125e-2, 9.9032557010650635e-1,
    6.6107827425003052e-1, 5.8693367242813110e-1, 9.3074029684066772e-1, 3.7010568380355835e-1,
    6.2468796968460083e-1, 7.3904669284820557e-1, 8.4791696071624756e-1, 3.2912850379943848e-1,
    7.1115493774414062e-3, 5.9478437900543213e-1, 6.7199707031250000e-1, 8.9839857816696167e-1,
    2.6267814636230469e-1, 1.0587638616561890e-1, 9.3495494127273560e-1, 9.9107789993286133e-1,
    5.0981825590133667e-1, 9.6975696086883545e-1, 6.1916261911392212e-1, 1.1414635181427002e-1,
    5.3248554468154907e-1, 1.1208379268646240e-1, 5.5403345823287964e-1, 3.5712575912475586e-1,
    9.5578378438949585e-1, 8.6278992891311646e-1, 9.2059129476547241e-1, 9.2922645807266235e-1,
    5.4787606000900269e-1, 3.1624758243560791e-1, 3.4879761934280396e-1, 1.4383316040039062e-2,
    8.9705604314804077e-1, 9.9688744544982910e-1, 4.5623308420181274e-1, 9.9999982118606567e-1,
    10.0, 43.0,
];

// Upstream MESA 26.4.1 `star/rsp_Type_II_Cepheid` checkpoints:
// `(step, log_L, log_Teff, log_R, rsp_num_periods, rsp_period_in_days, rsp_DeltaMag)`.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
const T2C_REF_ROWS: &[(usize, f64, f64, f64, usize, f64, f64)] = &[
    (
        100,
        2.1349223969055586,
        3.8085092176447022,
        9.807802546149554e-1,
        0,
        1.1483299305212364,
        0.0,
    ),
    (
        1000,
        2.110493508211977,
        3.8054288366113864,
        9.736486988019374e-1,
        0,
        1.1483299305212364,
        0.0,
    ),
    (
        3620,
        2.1530796822801124,
        3.811252950023334,
        9.850065514286126e-1,
        1,
        1.686913674831677,
        2.083355521929617e-1,
    ),
    (
        4490,
        2.1593283032673152,
        3.816898938521944,
        9.742822490131794e-1,
        1,
        1.6869136748316769,
        2.083355521929617e-1,
    ),
    (
        4760,
        2.1616073532443654,
        3.8119311222554786,
        9.92501952954413e-1,
        2,
        1.607876646170786,
        2.3831070237787416e-1,
    ),
    (
        5930,
        2.169822315231969,
        3.814556879541559,
        9.922028342480703e-1,
        3,
        1.5686207150569567,
        2.4269950603189425e-1,
    ),
    (
        7270,
        2.1621792456788187,
        3.8116455799164233,
        9.936791120775436e-1,
        4,
        1.7425780485895626,
        2.7031111140674646e-1,
    ),
    (
        8350,
        2.168886684589078,
        3.8130664096777713,
        9.952927551875723e-1,
        5,
        1.573407171746252,
        2.969700758060576e-1,
    ),
    (
        9280,
        2.1471931045277,
        3.815479650187092,
        9.692982628125483e-1,
        5,
        1.5734071717462521,
        2.969700758060576e-1,
    ),
    (
        9570,
        2.1847698862900176,
        3.8179651743337333,
        9.953588613551477e-1,
        6,
        1.5957784609687937,
        3.218501864916139e-1,
    ),
    (
        10450,
        2.120139998041459,
        3.81156815054221,
        9.628254438999801e-1,
        6,
        1.5957784609687937,
        3.218501864916139e-1,
    ),
    (
        10920,
        2.152649842818962,
        3.807255738943663,
        9.983073184358436e-1,
        7,
        1.797887417791047,
        3.9351893067685056e-1,
    ),
    (
        11900,
        2.186944711405566,
        3.8184211984778567,
        9.950044419967669e-1,
        8,
        1.433385651122266,
        3.580961276584791e-1,
    ),
    (
        12930,
        2.111911722408387,
        3.8106712357179844,
        9.603580315700044e-1,
        8,
        1.433385651122266,
        3.580961276584791e-1,
    ),
    (
        13540,
        2.133826815933002,
        3.801235957563389,
        1.000431937989085,
        9,
        1.951511716913778,
        5.118759466155135e-1,
    ),
    (
        13860,
        2.0532635285124434,
        3.7902119822476665,
        9.769828673264457e-1,
        9,
        1.9515117169137779,
        5.118759466155135e-1,
    ),
    (
        14683,
        2.1228455764839693,
        3.7993716694540143,
        9.962823030061446e-1,
        10,
        1.7272428644329085,
        5.054232385777698e-1,
    ),
];

#[test]
fn test_rsp_t2c_built_model_with_mesa_rng_state_matches_10_periods_mesa_26_4_1()
{
    let ctrl = t2c_controls();
    let (mut m, lina) = do_rsp_build_model(
        ctrl.clone(),
        base_job(true),
        EosConfig::default(),
        a09_kap(ctrl.rsp_z),
        None,
    )
    .expect("build t2c model");
    let lina = lina.expect("lina output");
    let lina_p0_days = lina.periods_sec[0] / 86_400.0;
    let lina_p1_days = lina.periods_sec[1] / 86_400.0;
    assert!(
        (lina_p0_days - 1.421_191_475_369_742_5e5 / 86_400.0).abs() < 1.0e-7,
        "LINA P0 mismatch: {lina_p0_days}"
    );
    assert!(
        (lina_p1_days - 9.921_570_599_703_483e4 / 86_400.0).abs() < 1.0e-7,
        "LINA P1 mismatch: {lina_p1_days}"
    );

    // Natural `RELAX_ENV` stops at 7057 draws (vs 7071 in MESA) because
    // `flexmesa_math::intrinsic_{log,log10,pow}` (`crlibm` `*_rn`, `#165`/`#198`)
    // differs by 1 ULP on isolated calls from MESA's `glibc libm` intrinsics
    // (`dlog`, `dlog10`, `10.d0**`, `**1.5d0` in `star/private/rsp_relax_env.f90:599,737,1019-1020`).
    // Pinning the post-relax `RANMAR` state isolates nonlinear evolution parity
    // over all 14,683 steps (10 periods) including peak-luminosity extrema.
    m.aux.rsp.rng.unpack(&MESA_T2C_RNG_STATE);

    let mut checked = 0_usize;
    let mut step = 0_usize;
    while m.aux.rsp.num_periods < 10 && step < 15_000 {
        step += 1;
        star_evolve_step(&mut m, false, true, true).expect("t2c evolve step");
        if let Some(&(_, ml_ref, mt_ref, mr_ref, mnp_ref, mp_ref, mdm_ref)) =
            T2C_REF_ROWS.iter().find(|r| r.0 == step)
        {
            checked += 1;
            let (ml, mt, mr) = history_metrics(&m);
            let hyd_period_days = m.aux.rsp.period / 86_400.0;
            let dmag = m.aux.rsp.delta_mag;
            assert_eq!(
                m.aux.rsp.num_periods, mnp_ref,
                "step {step} num_periods"
            );
            assert!(
                (ml - ml_ref).abs() < 1.0e-4,
                "step {step} log_L diff {}",
                ml - ml_ref
            );
            assert!(
                (mt - mt_ref).abs() < 5.0e-5,
                "step {step} log_Teff diff {}",
                mt - mt_ref
            );
            assert!(
                (mr - mr_ref).abs() < 5.0e-5,
                "step {step} log_R diff {}",
                mr - mr_ref
            );
            assert!(
                (hyd_period_days - mp_ref).abs() < 1.0e-4,
                "step {step} period diff {}",
                hyd_period_days - mp_ref
            );
            assert!(
                (dmag - mdm_ref).abs() < 1.0e-4,
                "step {step} DeltaMag diff {}",
                dmag - mdm_ref
            );
        }
    }

    assert_eq!(step, 14_683, "exact MESA 26.4.1 step count for 10 periods");
    assert_eq!(m.aux.rsp.num_periods, 10);
    assert_eq!(checked, T2C_REF_ROWS.len());
}

struct RspTempDir {
    root: PathBuf,
}

impl RspTempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_rsp_parity_{tag}_{}_{}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp test dir");
        Self { root }
    }
}

impl Drop for RspTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn test_rsp_13_controls_defaults_and_inventory() {
    use flexmesa_star::inlist_inventory::{
        CONTROLS_SUPPORTED, CONTROLS_TERMINAL_OUTPUT, CONTROLS_UNUSED,
    };
    use flexmesa_star::RspConfig;

    let def = RspConfig::default();
    assert!(!def.write_map);
    assert_eq!(def.map_columns_filename, "map_columns.list");
    assert_eq!(def.map_filename, "map.data");
    assert_eq!(def.map_first_period, -1);
    assert_eq!(def.map_last_period, -1);
    assert_eq!(def.map_zone_interval, 2);
    assert_eq!(def.map_history_filename, "map_history.data");
    assert_eq!(def.qvisc_linear, 0.0);
    assert_eq!(def.qvisc_quadratic, 0.0);
    assert!((def.tol_max_resid - 1.0e-6).abs() < 1.0e-18);
    assert!(!def.report_limit_dt);
    assert!(!def.report_undercorrections);
    assert!(!def.trace_rsp_build_model);

    for k in [
        "rsp_write_map",
        "rsp_map_columns_filename",
        "rsp_map_filename",
        "rsp_map_first_period",
        "rsp_map_last_period",
        "rsp_map_zone_interval",
        "rsp_map_history_filename",
        "rsp_qvisc_linear",
        "rsp_qvisc_quadratic",
        "rsp_tol_max_resid",
    ] {
        assert!(CONTROLS_SUPPORTED.contains(&k), "{k} in CONTROLS_SUPPORTED");
        assert!(!CONTROLS_UNUSED.contains(&k), "{k} not in CONTROLS_UNUSED");
    }
    for k in [
        "rsp_report_limit_dt",
        "rsp_report_undercorrections",
        "rsp_trace_rsp_build_model",
    ] {
        assert!(
            CONTROLS_TERMINAL_OUTPUT.contains(&k),
            "{k} in CONTROLS_TERMINAL_OUTPUT"
        );
        assert!(!CONTROLS_UNUSED.contains(&k), "{k} not in CONTROLS_UNUSED");
    }
}

#[test]
fn test_rsp_13_controls_strict_parsing_and_validation() {
    use flexmesa_star::InlistBundle;

    let tmp = RspTempDir::new("strict_parse");
    let valid_path = tmp.root.join("inlist_valid");
    fs::write(
        &valid_path,
        "&controls\n\
         rsp_write_map = .true.\n\
         rsp_map_columns_filename = 'custom_cols.list'\n\
         rsp_map_filename = 'custom_map.data'\n\
         rsp_map_first_period = 3\n\
         rsp_map_last_period = 5\n\
         rsp_map_history_filename = 'custom_hist.data'\n\
         rsp_map_zone_interval = 4\n\
         rsp_qvisc_linear = 0.25d0\n\
         rsp_qvisc_quadratic = 1.5d0\n\
         rsp_tol_max_resid = 5.0d-7\n\
         rsp_report_limit_dt = .true.\n\
         rsp_report_undercorrections = .true.\n\
         rsp_trace_rsp_build_model = .true.\n\
         /\n",
    )
    .expect("write valid inlist");
    let bundle = InlistBundle::from_file(&valid_path).expect("strict parse");
    assert_eq!(bundle.validation_report.unsupported_controls, Vec::new());
    let rsp = &bundle.controls.rsp;
    assert!(rsp.write_map);
    assert_eq!(rsp.map_columns_filename, "custom_cols.list");
    assert_eq!(rsp.map_filename, "custom_map.data");
    assert_eq!(rsp.map_first_period, 3);
    assert_eq!(rsp.map_last_period, 5);
    assert_eq!(rsp.map_history_filename, "custom_hist.data");
    assert_eq!(rsp.map_zone_interval, 4);
    assert!((rsp.qvisc_linear - 0.25).abs() < 1.0e-15);
    assert!((rsp.qvisc_quadratic - 1.5).abs() < 1.0e-15);
    assert!((rsp.tol_max_resid - 5.0e-7).abs() < 1.0e-18);
    assert!(rsp.report_limit_dt);
    assert!(rsp.report_undercorrections);
    assert!(rsp.trace_rsp_build_model);
    for k in [
        "rsp_report_limit_dt",
        "rsp_report_undercorrections",
        "rsp_trace_rsp_build_model",
    ] {
        assert!(
            bundle
                .validation_report
                .terminal_output_keys
                .iter()
                .any(|(_, name)| name == k),
            "terminal_output records {k}"
        );
    }

    for (tag, bad) in [
        ("zero_stride", "rsp_map_zone_interval = 0"),
        ("bad_write_map", "rsp_write_map = 123"),
        ("bad_qvisc", "rsp_qvisc_linear = 'bad'"),
        ("bad_tol", "rsp_tol_max_resid = .true."),
        ("bad_report", "rsp_report_limit_dt = 1"),
    ] {
        let p = tmp.root.join(format!("inlist_{tag}"));
        fs::write(&p, format!("&controls\n  {bad}\n/\n")).expect("write");
        assert!(
            InlistBundle::from_file(&p).is_err(),
            "{tag} should fail strict validation"
        );
    }
}

#[test]
fn test_rsp_profile_scalars_uq_photo_and_qvisc_invariance() {
    use flexmesa_star::mesa_profile::{
        format_star_profile_text, parse_star_profile_columns,
    };
    use flexmesa_star::photo_io::{load_photo_bytes, write_photo_bytes};
    use flexmesa_star::rsp_driver::rsp_total_energy_integrals;

    let mod_text =
        fs::read_to_string(golden_rsp_dir().join("rsp_Delta_Scuti_cycle0.mod"))
            .expect("read delta scuti cycle0.mod");
    let mut c_default = delta_scuti_controls();
    c_default.rsp_target_steps_per_cycle = 120;
    let mut m_def = read_mod_str(&mod_text, c_default, base_job(false))
        .expect("read_mod_str default");
    rsp_setup_model(&mut m_def, false, false).expect("setup default");

    let mut c_qvisc = delta_scuti_controls();
    c_qvisc.rsp_target_steps_per_cycle = 120;
    c_qvisc.rsp.qvisc_linear = 0.5;
    c_qvisc.rsp.qvisc_quadratic = 2.0;
    c_qvisc.rsp.tol_max_resid = 1.0e-3;
    let mut m_qv = read_mod_str(&mod_text, c_qvisc, base_job(false))
        .expect("read_mod_str qvisc");
    rsp_setup_model(&mut m_qv, false, false).expect("setup qvisc");

    for _ in 0..3 {
        let r_def =
            star_evolve_step(&mut m_def, false, true, true).expect("step def");
        let r_qv =
            star_evolve_step(&mut m_qv, false, true, true).expect("step qv");
        assert_eq!(r_def.iterations, r_qv.iterations);
        assert_eq!(r_def.retries, r_qv.retries);
        assert_eq!(m_def.teff.to_bits(), m_qv.teff.to_bits());
        assert_eq!(m_def.aux.rsp.uq_rsp, m_qv.aux.rsp.uq_rsp);
    }
    let _ = rsp_total_energy_integrals(&mut m_def);

    assert!(
        m_def.aux.rsp.uq_rsp.iter().any(|&v| v.abs() > 0.0),
        "expected non-zero Uq from momentum equation"
    );

    let nz = m_def.nz;
    let specs = parse_star_profile_columns(
        "zone\nrsp_Hp_face\nrsp_Y_face\nrsp_Lc_div_L\nrsp_Lr_div_L\n\
         rsp_Lt_div_L\nrsp_Et\nrsp_Pt\nrsp_Eq\nrsp_src_snk\nrsp_Uq\nrsp_Pvsc\n",
        None,
    )
    .expect("parse all 11 rsp_* profile columns");
    let text = format_star_profile_text(&m_def, &specs);
    let data_lines: Vec<&str> = text.lines().skip(6).collect();
    assert_eq!(data_lines.len(), nz);
    let row0: Vec<f64> = data_lines[0]
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("float col"))
        .collect();
    let row1: Vec<f64> = data_lines[1]
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("float col"))
        .collect();
    // `profile_getval.f90:1960-2020`: k=1 surface cell uses k=2 values for
    // `rsp_Y_face`, `rsp_Lc_div_L`, `rsp_Lt_div_L`, `rsp_Uq`.
    assert_eq!(row0[2], row1[2], "rsp_Y_face k=1 uses k=2");
    assert_eq!(row0[3], row1[3], "rsp_Lc_div_L k=1 uses k=2");
    assert_eq!(row0[5], row1[5], "rsp_Lt_div_L k=1 uses k=2");
    assert_eq!(row0[10], row1[10], "rsp_Uq k=1 uses k=2");
    assert!(
        (row0[1] - m_def.aux.rsp.hp_face[0]).abs()
            <= 1.0e-12 * row0[1].abs() + 1.0e-30
    );
    let lr_div_l_0 =
        m_def.aux.rsp.fr_rsp[0] * 4.0 * PI * m_def.r[0] * m_def.r[0]
            / m_def.lum[0];
    assert!((row0[4] - lr_div_l_0).abs() < 1.0e-12);

    let mut non_rsp = m_def.clone();
    non_rsp.controls.rsp_flag = false;
    let non_rsp_text = format_star_profile_text(&non_rsp, &specs);
    let non_rsp_row0: Vec<f64> = non_rsp_text
        .lines()
        .nth(6)
        .expect("row 0")
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("float"))
        .collect();
    for val in &non_rsp_row0[1..] {
        assert_eq!(*val, 0.0);
    }

    m_def.aux.rsp.writing_map = true;
    m_def.aux.rsp.fase0 = 12.5;
    m_def.aux.rsp.map_col_names = vec![String::from("rsp_Uq")];
    let bytes = write_photo_bytes(&m_def);
    let mut restored = load_photo_bytes(&bytes).expect("photo roundtrip");
    assert!(restored.aux.rsp.writing_map);
    assert_eq!(restored.aux.rsp.fase0, 12.5);
    assert_eq!(restored.aux.rsp.map_col_names, vec![String::from("rsp_Uq")]);
    assert_eq!(restored.aux.rsp.uq_rsp, m_def.aux.rsp.uq_rsp);
    rsp_setup_model(&mut restored, false, true).expect("photo setup");
    assert!(!restored.aux.rsp.writing_map);
}

fn verify_map_data_output(map_text: &str) {
    let mut map_lines = map_text.lines();
    let idx_hdr: Vec<usize> = map_lines
        .next()
        .expect("line 1")
        .split_whitespace()
        .map(|s| s.parse::<usize>().expect("col idx"))
        .collect();
    assert_eq!(idx_hdr, (1..=20).collect::<Vec<_>>());
    let name_hdr: Vec<&str> = map_lines
        .next()
        .expect("line 2")
        .split_whitespace()
        .collect();
    assert_eq!(name_hdr[0], "phase");
    assert_eq!(name_hdr[1], "zone");
    assert_eq!(name_hdr[2], "temperature");
    assert_eq!(name_hdr[15], "rsp_Uq");
    assert_eq!(name_hdr.len(), 20);

    let first_row: Vec<f64> = map_lines
        .next()
        .expect("row 1")
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("float"))
        .collect();
    assert_eq!(first_row.len(), 20);
    assert_eq!(first_row[0], 0.0, "first map phase must be 0.0");
    assert!(
        (first_row[1] - 3.0).abs() < 1.0e-12,
        "k=1 skipped by I<NZN, k=3 first with zone_interval=2"
    );
    let mut saw_nonzero_uq = false;
    for row_line in map_lines {
        let vals: Vec<f64> = row_line
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("float"))
            .collect();
        assert_eq!(vals.len(), 20);
        assert!(vals.iter().all(|v| v.is_finite()));
        if vals[15].abs() > 0.0 {
            saw_nonzero_uq = true;
        }
    }
    assert!(saw_nonzero_uq, "expected non-zero rsp_Uq in map.data");
}

#[test]
fn test_rsp_write_map_emits_pm3d_map_and_rotates_history_file() {
    use flexmesa_star::mesa_log_reader::read_mesa_log_file;
    use flexmesa_star::{run_test_suite_case_with_logs, StarLogsOptions};

    let tmp = RspTempDir::new("map_rotate");
    let case_dir = tmp.root.join("case");
    let out_dir = tmp.root.join("out");
    fs::create_dir_all(&case_dir).expect("mkdir case");
    fs::create_dir_all(&out_dir).expect("mkdir out");

    fs::copy(
        golden_rsp_dir().join("rsp_Delta_Scuti_cycle0.mod"),
        case_dir.join("cycle0.mod"),
    )
    .expect("copy cycle0.mod");
    fs::write(
        case_dir.join("history_columns.list"),
        "model_number\nstar_age\nrsp_num_periods\nrsp_phase\n",
    )
    .expect("write history_columns.list");
    fs::write(
        case_dir.join("map_columns.list"),
        "temperature\nrsp_Hp_face\nrsp_Y_face\nrsp_Lc_div_L\nrsp_Lr_div_L\n\
         rsp_Lt_div_L\nrsp_Et\nenergy\ncsound\nrsp_Pt\npressure\nrsp_Eq\n\
         rsp_src_snk\nrsp_Uq\nrsp_Pvsc\nvel_km_per_s\nv_div_csound\nxq\n",
    )
    .expect("write map_columns.list");
    fs::write(
        case_dir.join("inlist"),
        "&star_job\n\
         load_saved_model = .true.\n\
         load_model_filename = 'cycle0.mod'\n\
         set_initial_age = .true.\n\
         initial_age = 0\n\
         set_initial_model_number = .true.\n\
         initial_model_number = 0\n\
         set_initial_dt = .true.\n\
         seconds_for_initial_dt = 160.0d0\n\
         /\n\
         &eos\n/\n\
         &kap\n\
         Zbase = 0.02d0\n\
         kap_file_prefix = 'a09'\n\
         kap_lowT_prefix = 'lowT_fa05_a09p'\n\
         kap_CO_prefix = 'a09_co'\n\
         /\n\
         &controls\n\
         RSP_mass = 2.0\n\
         RSP_Teff = 6900\n\
         RSP_L = 30\n\
         RSP_X = 0.70\n\
         RSP_Z = 0.02\n\
         RSP_alfa = 1.2d0\n\
         RSP_alfam = 0.25d0\n\
         RSP_target_steps_per_cycle = 80\n\
         RSP_max_num_periods = 2\n\
         RSP_write_map = .true.\n\
         RSP_map_first_period = 2\n\
         RSP_map_last_period = 2\n\
         RSP_map_zone_interval = 2\n\
         RSP_map_history_filename = 'map_history.data'\n\
         history_interval = 5\n\
         /\n",
    )
    .expect("write inlist");

    let logs_opts = StarLogsOptions::from_target(&out_dir, true, true, false);
    let rep = run_test_suite_case_with_logs(
        &case_dir,
        false,
        Some(500),
        false,
        logs_opts,
    )
    .expect("run map case");
    assert_eq!(rep.termination_code, "reached max number of periods");

    let hist_tbl = read_mesa_log_file(&out_dir.join("LOGS/history.data"))
        .expect("read LOGS/history.data");
    let map_hist_tbl =
        read_mesa_log_file(&out_dir.join("LOGS/map_history.data"))
            .expect("read LOGS/map_history.data");
    let hist_models = hist_tbl.column("model_number").expect("hist models");
    let hist_periods =
        hist_tbl.column("rsp_num_periods").expect("hist periods");
    let map_hist_models = map_hist_tbl
        .column("model_number")
        .expect("map hist models");
    assert_ne!(hist_models.len(), 0);
    assert_ne!(map_hist_models.len(), 0);

    let hist_last_model = *hist_models.last().expect("hist last");
    let map_hist_first_model = *map_hist_models.first().expect("map first");
    assert!(
        hist_last_model < map_hist_first_model,
        "history.data ({hist_last_model}) must end before map_history.data ({map_hist_first_model})"
    );
    assert!((hist_last_model % 5.0).abs() < 1.0e-12);
    for &np in hist_periods {
        assert!(np < 1.0, "history.data should stop before period 2 opens");
    }

    let map_text = fs::read_to_string(out_dir.join("LOGS/map.data"))
        .expect("read LOGS/map.data");
    verify_map_data_output(&map_text);
}

#[allow(clippy::unreadable_literal)]
fn setup_stah_case_model(
    mut ctrl: StarControls,
) -> (StarModel, RspChemCache, RspDerivedConsts) {
    ctrl.rsp_flag = true;
    ctrl.v_flag = true;
    let job = base_job(true);
    let net = get_or_load_net("o18_and_ne22.net", None).expect("load net");
    let nzn = ctrl.rsp.nz;
    let mut model = StarModel::new(nzn, net.chem_id.clone(), job, ctrl);
    model.eos_config = EosConfig::default();
    model.kap_config = a09_kap(model.controls.rsp_z);
    model.controls.tau_factor =
        if model.controls.rsp.use_atm_grey_with_kap_for_psurf {
            model.controls.rsp.tau_surf_for_atm_grey_with_kap / model.tau_base
        } else {
            model.controls.rsp.surface_tau / model.tau_base
        };
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, nzn);
    let chem = RspChemCache::from_controls(
        &model.chem_id,
        &model.job,
        &model.controls,
    )
    .expect("chem cache");
    (model, chem, dc)
}

#[allow(clippy::unreadable_literal)]
struct StahOracle {
    name: &'static str,
    dm_n_bits: u64,
    h_bits: u64,
    t1_bits: u64,
    r1_bits: u64,
    xm_bits: u64,
    t100_bits: u64,
    r100_bits: u64,
    vol100_bits: u64,
    rc_bits: u64,
    mc_bits: u64,
}

fn assert_stah_oracle(ctrl: StarControls, want: &StahOracle) {
    let (mut model, chem, dc) = setup_stah_case_model(ctrl);
    let nzn_init = model.nz;
    let mx = model.controls.rsp_mass * dc.sunm;
    let l_surf = model.controls.rsp_l * dc.sunl;
    let te = model.controls.rsp_teff;
    let th0 = model.controls.rsp.t_anchor;
    let tin = model.controls.rsp.t_inner;
    let nzt = model.controls.rsp.nz_outer;

    let (h_init, dmn_init) =
        znvar(&mut model, &chem, &dc, nzn_init, l_surf, te, mx).expect("znvar");
    assert!(h_init > 1.0 && dmn_init > 0.0, "{} znvar finite", want.name);

    let grid = stah(
        &mut model, &chem, &dc, mx, l_surf, te, th0, tin, nzt, nzn_init,
    )
    .expect("stah");
    let n = grid.nzn;

    assert_eq!(
        grid.dm[n - 1].to_bits(),
        want.dm_n_bits,
        "{} STAH dmN bits",
        want.name
    );
    assert_eq!(
        grid.t[1].to_bits(),
        want.t1_bits,
        "{} STAH T(1) bits",
        want.name
    );
    assert_eq!(
        grid.r[1].to_bits(),
        want.r1_bits,
        "{} STAH R(1) bits",
        want.name
    );
    assert_eq!(
        model.xmstar.to_bits(),
        want.xm_bits,
        "{} STAH xmstar bits",
        want.name
    );
    assert_eq!(
        grid.t[100].to_bits(),
        want.t100_bits,
        "{} STAH T(100) bits",
        want.name
    );
    assert_eq!(
        grid.r[100].to_bits(),
        want.r100_bits,
        "{} STAH R(100) bits",
        want.name
    );
    assert_eq!(
        grid.vol[100].to_bits(),
        want.vol100_bits,
        "{} STAH Vol(100) bits",
        want.name
    );
    assert_eq!(
        model.r_center.to_bits(),
        want.rc_bits,
        "{} STAH R_center bits",
        want.name
    );
    assert_eq!(
        model.m_center.to_bits(),
        want.mc_bits,
        "{} STAH M_center bits",
        want.name
    );
    let h_ratio = grid.dm[1] / grid.dm[2];
    let h_want = f64::from_bits(want.h_bits);
    let h_rel = (h_ratio - h_want).abs() / h_want;
    assert!(
        h_rel <= 1.0e-15,
        "{} STAH inner H ratio mismatch: got={h_ratio:.17e} want={h_want:.17e}",
        want.name
    );
}

#[test]
#[allow(clippy::unreadable_literal)]
fn test_rsp_stah_0ulp_parity_t2c_and_delta_scuti_mesa_26_4_1() {
    // MESA 26.4.1 `star/private/rsp_build.f90` STAH outputs for
    // `star/rsp_Type_II_Cepheid` (`nzn=200`, `nzt=60`) and `star/rsp_Delta_Scuti`.
    let t2c_want = StahOracle {
        name: "rsp_Type_II_Cepheid",
        dm_n_bits: 0x4501_3B8A_6E73_3802,
        h_bits: 0x3FF1_9795_26A1_7248,
        t1_bits: 0x413E_847F_FDD5_6B03,
        r1_bits: 0x4229_4AF1_4E22_87FE,
        xm_bits: 0x466A_980B_321C_A800,
        t100_bits: 0x40E2_7DB0_4BBA_9E15,
        r100_bits: 0x4261_6AF0_039F_A3C4,
        vol100_bits: 0x4197_1D17_6BA2_B3EE,
        rc_bits: 0x4227_792B_ECD7_56DE,
        mc_bits: 0x46CA_8B5E_FF4D_0707,
    };
    let ds_want = StahOracle {
        name: "rsp_Delta_Scuti",
        dm_n_bits: 0x44BA_E048_08E1_A8EA,
        h_bits: 0x3FF2_FC2D_7330_D504,
        t1_bits: 0x413E_8480_0286_85F9,
        r1_bits: 0x423C_7F08_31F0_DC6B,
        xm_bits: 0x4693_F132_6732_8B20,
        t100_bits: 0x40D0_A476_6530_68E4,
        r100_bits: 0x424E_F58F_978D_AB70,
        vol100_bits: 0x418B_41BF_3E85_8E15,
        rc_bits: 0x423B_7C61_C779_2E3A,
        mc_bits: 0x46E7_E2C7_207C_DA3E,
    };

    assert_stah_oracle(t2c_controls(), &t2c_want);
    assert_stah_oracle(delta_scuti_controls(), &ds_want);
}

#[allow(clippy::unreadable_literal)]
struct RelaxLinaOracle {
    name: &'static str,
    dm_n_bits: u64,
    t1_bits: u64,
    r1_bits: u64,
    vol1_bits: u64,
    lina_pers_bits: [u64; 3],
    max_state_rel: f64,
    max_period_rel: f64,
}

fn assert_relax_lina_oracle(ctrl: StarControls, want: &RelaxLinaOracle) {
    let (mut model, chem, dc_init) = setup_stah_case_model(ctrl);
    let nzn_init = model.nz;
    let mut rng = model.aux.rsp.rng.clone();
    let mx = model.controls.rsp_mass * dc_init.sunm;
    let l_surf = model.controls.rsp_l * dc_init.sunl;
    let te = model.controls.rsp_teff;
    let th0 = model.controls.rsp.t_anchor;
    let tin = model.controls.rsp.t_inner;
    let nzt = model.controls.rsp.nz_outer;

    let mut grid = stah(
        &mut model, &chem, &dc_init, mx, l_surf, te, th0, tin, nzt, nzn_init,
    )
    .expect("stah");
    let nzn = grid.nzn;
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, nzn);
    relax_env(
        &mut model, &chem, &dc, &mut rng, &mut grid, l_surf, th0, te, nzt,
    )
    .expect("relax_env");

    let checks = [
        (
            "dm(NZN-1)",
            grid.dm[nzn - 1],
            f64::from_bits(want.dm_n_bits),
        ),
        ("T(1)", grid.t[1], f64::from_bits(want.t1_bits)),
        ("R(1)", grid.r[1], f64::from_bits(want.r1_bits)),
        ("Vol(1)", grid.vol[1], f64::from_bits(want.vol1_bits)),
    ];
    for (label, got, ref_val) in checks {
        let rel = (got - ref_val).abs() / ref_val.abs();
        assert!(
            rel <= want.max_state_rel,
            "{} POST_RELAX {label} drift: got={got:.17e} (0x{:016X}) \
             want={ref_val:.17e} (0x{:016X}) rel={rel:.3e}",
            want.name,
            got.to_bits(),
            ref_val.to_bits(),
        );
    }

    cleanup_for_lina(&mut model, &dc, &mut grid).expect("cleanup_for_lina");
    let nmodes = model.controls.rsp.nmodes;
    let lina = do_lina(&mut model, &chem, &dc, &grid, l_surf, nmodes, None)
        .expect("do_lina");

    for (mode, &want_bits) in want.lina_pers_bits.iter().enumerate() {
        let got = lina.periods_sec[mode];
        let ref_val = f64::from_bits(want_bits);
        let rel = (got - ref_val).abs() / ref_val.abs();
        assert!(
            rel <= want.max_period_rel,
            "{} LINA mode {mode} period drift: got={got:.17e} (0x{:016X}) \
             want={ref_val:.17e} (0x{:016X}) rel={rel:.3e}",
            want.name,
            got.to_bits(),
            want_bits,
        );
    }
}

#[test]
#[allow(clippy::unreadable_literal)]
fn test_rsp_relax_env_and_lina_periods_parity_t2c_and_delta_scuti_mesa_26_4_1()
{
    let t2c_want = RelaxLinaOracle {
        name: "rsp_Type_II_Cepheid",
        dm_n_bits: 0x4501_617A_1276_8D1D,
        t1_bits: 0x413E_864D_5CE5_A8C0,
        r1_bits: 0x4229_4BA9_BBC2_5438,
        vol1_bits: 0x4056_1F4C_7E03_EB64,
        lina_pers_bits: [
            0x4101_5939_2E27_DD7B,
            0x40F8_38FB_4BC3_8BFB,
            0x40F1_8AE5_98EA_6C34,
        ],
        max_state_rel: 1.0e-10,
        max_period_rel: 5.0e-8,
    };
    let ds_want = RelaxLinaOracle {
        name: "rsp_Delta_Scuti",
        dm_n_bits: 0x44BA_6224_6AD3_0846,
        t1_bits: 0x413E_8A4C_9B74_CACF,
        r1_bits: 0x423C_7D73_3F5A_8EC2,
        vol1_bits: 0x4048_CB52_E783_6BEE,
        lina_pers_bits: [
            0x40CC_D97E_52A4_B40E,
            0x40C5_5BEB_AA1D_814F,
            0x40C0_9E8B_3254_6536,
        ],
        max_state_rel: 1.0e-9,
        max_period_rel: 5.0e-8,
    };

    assert_relax_lina_oracle(t2c_controls(), &t2c_want);
    assert_relax_lina_oracle(delta_scuti_controls(), &ds_want);
}
