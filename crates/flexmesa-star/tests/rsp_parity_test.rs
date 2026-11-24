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
    finish_after_build_model, set_star_vars_rsp, RspEnvGrid,
};
use flexmesa_star::rsp_hyd_turb::calc_chi;
use flexmesa_star::rsp_hyd_work::RspHydWork;
use flexmesa_star::rsp_turb_partials::RspTurbPartials;
use flexmesa_star::rsp_zone_micro::{LcFilterMode, RspZoneMicro};
use flexmesa_star::{
    do_rsp_build_model, read_mod_str, rsp_setup_model, star_evolve_step,
    StarControls, StarJobControls, StarModel,
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
        9570,
        2.1847698862900176,
        3.8179651743337333,
        9.953588613551477e-1,
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
        13540,
        2.133826815933002,
        3.801235957563389,
        1.000431937989085,
        9,
        1.951511716913778,
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

    // Natural `RELAX_ENV` stops at 5243 draws because `flexmesa_math::intrinsic_log`
    // (`crlibm::log_rn`, `#165`) differs by 1 ULP from MESA's `glibc libm dlog`
    // (`star/private/rsp_relax_env.f90:1019-1020`) under Jacobian `cond ~ 1.93e19`.
    // Pinning the RNG seed isolates nonlinear evolution parity on a freshly built
    // envelope over all 14,683 steps (10 periods) of `star/rsp_Type_II_Cepheid`.
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
