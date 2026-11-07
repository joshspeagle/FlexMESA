// SPDX-License-Identifier: LGPL-3.0-or-later
//! Hardwired `approx21` and `approx21_plus_co56` molar abundance derivatives
//! (`approx21_dydt` with `F64x2` double-double accumulation in
//! `net/private/net_approx21.f90` lines 926–1544).

use crate::approx21::Approx21Indices;
use crate::approx21_dfdy_heavy::approx21_dydt_iron_group;
use crate::f64x2::F64x2;

#[inline]
fn eval_fe56ec_fake_factor(
    fe56ec_fake_factor_in: f64,
    min_t_for_fe56ec_fake_factor: f64,
    temp: f64,
) -> f64 {
    if temp >= min_t_for_fe56ec_fake_factor {
        fe56ec_fake_factor_in
    } else {
        0.0
    }
}

/// Evaluates `approx21_dydt` (`net/private/net_approx21.f90` lines 926–1544).
///
/// Here `rat0` is the un-differentiated screened rate array (`ratdum` in MESA),
/// and `ratd` is either `rat0` (`deriva == false`) or `dratdumdt` / `dratdumdd`
/// (`deriva == true`, `rate` in MESA).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn approx21_dydt(
    idx: &Approx21Indices,
    y: &[f64],
    rat0: &[f64],
    ratd: &[f64],
    deriva: bool,
    fe56ec_fake_factor_in: f64,
    min_t_for_fe56ec_fake_factor: f64,
    fe56ec_n_neut: i32,
    temp: f64,
    dydt: &mut [f64],
) {
    let fe56ec_fake_factor = if !deriva {
        eval_fe56ec_fake_factor(
            fe56ec_fake_factor_in,
            min_t_for_fe56ec_fake_factor,
            temp,
        )
    } else {
        0.0
    };
    let nf = f64::from(fe56ec_n_neut);
    dydt.fill(0.0);

    // Helper closures using F64x2 to match Fortran real(qp)
    let m2 = |a: f64, b: f64| F64x2::from_f64(a * b);
    let m3 = |a: f64, b: f64, c: f64| F64x2::from_f64(a * b * c);
    let m4 = |a: f64, b: f64, c: f64, d: f64| F64x2::from_f64(a * b * c * d);
    let m5 = |a: f64, b: f64, c: f64, d: f64, e: f64| {
        F64x2::from_f64(a * b * c * d * e)
    };
    let m6 = |a: f64, b: f64, c: f64, d: f64, e: f64, f: f64| {
        F64x2::from_f64(a * b * c * d * e * f)
    };

    // h1 (lines 960–969)
    let q_h1 = m4(-1.5, y[idx.ih1], y[idx.ih1], ratd[idx.irpp])
        + m3(y[idx.ihe3], y[idx.ihe3], ratd[idx.ir33])
        + m3(-y[idx.ihe3], y[idx.ihe4], ratd[idx.irhe3ag])
        + m4(-2.0, y[idx.ic12], y[idx.ih1], ratd[idx.ircpg])
        + m4(-2.0, y[idx.in14], y[idx.ih1], ratd[idx.irnpg])
        + m4(-2.0, y[idx.io16], y[idx.ih1], ratd[idx.iropg])
        + m3(-3.0, y[idx.ih1], ratd[idx.irpen]);
    dydt[idx.ih1] = q_h1.to_f64();

    // he3 (lines 973–978)
    let q_he3 = m4(0.5, y[idx.ih1], y[idx.ih1], ratd[idx.irpp])
        + m3(-y[idx.ihe3], y[idx.ihe3], ratd[idx.ir33])
        + m3(-y[idx.ihe3], y[idx.ihe4], ratd[idx.irhe3ag])
        + m2(y[idx.ih1], ratd[idx.irpen]);
    dydt[idx.ihe3] = q_he3.to_f64();

    // he4 (lines 983–1122)
    let mut q_he4 = m4(0.5, y[idx.ic12], y[idx.ic12], ratd[idx.ir1212])
        + m4(0.5, y[idx.ic12], y[idx.io16], ratd[idx.ir1216])
        + m5(0.56, 0.5, y[idx.io16], y[idx.io16], ratd[idx.ir1616])
        + m4(-y[idx.ihe4], y[idx.in14], ratd[idx.irnag], 1.5);

    q_he4 += m5(-0.5, y[idx.ihe4], y[idx.ihe4], y[idx.ihe4], ratd[idx.ir3a])
        + m3(3.0, y[idx.ic12], ratd[idx.irg3a])
        + m3(-y[idx.ihe4], y[idx.ic12], ratd[idx.ircag])
        + m2(y[idx.io16], ratd[idx.iroga])
        + m3(-y[idx.ihe4], y[idx.io16], ratd[idx.iroag])
        + m2(y[idx.ine20], ratd[idx.irnega])
        + m3(-y[idx.ihe4], y[idx.ine20], ratd[idx.irneag])
        + m2(y[idx.img24], ratd[idx.irmgga])
        + m3(-y[idx.ihe4], y[idx.img24], ratd[idx.irmgag])
        + m2(y[idx.isi28], ratd[idx.irsiga])
        + m3(-y[idx.ihe4], y[idx.isi28], ratd[idx.irsiag])
        + m2(y[idx.is32], ratd[idx.irsga]);

    q_he4 += m3(-y[idx.ihe4], y[idx.is32], ratd[idx.irsag])
        + m2(y[idx.iar36], ratd[idx.irarga])
        + m3(-y[idx.ihe4], y[idx.iar36], ratd[idx.irarag])
        + m2(y[idx.ica40], ratd[idx.ircaga])
        + m3(-y[idx.ihe4], y[idx.ica40], ratd[idx.ircaag])
        + m2(y[idx.iti44], ratd[idx.irtiga])
        + m3(-y[idx.ihe4], y[idx.iti44], ratd[idx.irtiag])
        + m2(y[idx.icr48], ratd[idx.ircrga])
        + m3(-y[idx.ihe4], y[idx.icr48], ratd[idx.ircrag])
        + m2(y[idx.ife52], ratd[idx.irfega])
        + m3(-y[idx.ihe4], y[idx.ife52], ratd[idx.irfeag])
        + m2(y[idx.ini56], ratd[idx.irniga]);

    if !deriva {
        q_he4 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.irs1],
            ratd[idx.ir1616],
        );
        for (y0, yp1, rap, r1, rgp) in [
            (y[idx.img24], y[idx.isi28], idx.irmgap, idx.irr1, idx.irsigp),
            (y[idx.isi28], y[idx.is32], idx.irsiap, idx.irs1, idx.irsgp),
            (y[idx.is32], y[idx.iar36], idx.irsap, idx.irt1, idx.irargp),
            (y[idx.iar36], y[idx.ica40], idx.irarap, idx.iru1, idx.ircagp),
            (y[idx.ica40], y[idx.iti44], idx.ircaap, idx.irv1, idx.irtigp),
            (y[idx.iti44], y[idx.icr48], idx.irtiap, idx.irw1, idx.ircrgp),
            (y[idx.icr48], y[idx.ife52], idx.ircrap, idx.irx1, idx.irfegp),
        ] {
            q_he4 += m4(-y[idx.ihe4], y0, ratd[rap], 1.0 - ratd[r1])
                + m3(yp1, ratd[rgp], ratd[r1]);
        }
    } else {
        q_he4 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            rat0[idx.irs1],
            ratd[idx.ir1616],
        ) + m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.irs1],
            rat0[idx.ir1616],
        );
        for (y0, yp1, rap, r1, rgp) in [
            (y[idx.img24], y[idx.isi28], idx.irmgap, idx.irr1, idx.irsigp),
            (y[idx.isi28], y[idx.is32], idx.irsiap, idx.irs1, idx.irsgp),
            (y[idx.is32], y[idx.iar36], idx.irsap, idx.irt1, idx.irargp),
            (y[idx.iar36], y[idx.ica40], idx.irarap, idx.iru1, idx.ircagp),
            (y[idx.ica40], y[idx.iti44], idx.ircaap, idx.irv1, idx.irtigp),
            (y[idx.iti44], y[idx.icr48], idx.irtiap, idx.irw1, idx.ircrgp),
            (y[idx.icr48], y[idx.ife52], idx.ircrap, idx.irx1, idx.irfegp),
        ] {
            q_he4 += m4(-y[idx.ihe4], y0, ratd[rap], 1.0 - rat0[r1])
                + m4(y[idx.ihe4], y0, rat0[rap], ratd[r1])
                + m3(yp1, rat0[rgp], ratd[r1])
                + m3(yp1, ratd[rgp], rat0[r1]);
        }
    }

    q_he4 += m4(y[idx.ife54], y[idx.iprot], y[idx.iprot], ratd[idx.ir5f54])
        + m3(-y[idx.ife52], y[idx.ihe4], ratd[idx.ir6f54])
        + m4(-y[idx.ife52], y[idx.ihe4], y[idx.iprot], ratd[idx.ir7f54])
        + m3(y[idx.ini56], y[idx.iprot], ratd[idx.ir8f54])
        + m2(-y[idx.ihe4], ratd[idx.iralf1])
        + m5(
            y[idx.ineut],
            y[idx.ineut],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.iralf2],
        )
        + m4(
            y[idx.ife56],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.irfe56_aux3],
        )
        + m3(-y[idx.ife54], y[idx.ihe4], ratd[idx.irfe56_aux4]);

    q_he4 += m4(0.5, y[idx.ihe3], y[idx.ihe3], ratd[idx.ir33])
        + m3(y[idx.ihe3], y[idx.ihe4], ratd[idx.irhe3ag]);
    q_he4 += m3(y[idx.io16], y[idx.ih1], ratd[idx.iropg]);
    if !deriva {
        q_he4 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], ratd[idx.irnpg]);
    } else {
        q_he4 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], rat0[idx.irnpg])
            + m4(y[idx.in14], y[idx.ih1], rat0[idx.ifa], ratd[idx.irnpg]);
    }
    dydt[idx.ihe4] = q_he4.to_f64();

    // c12 (lines 1126–1143)
    let mut q_c12 = m3(-y[idx.ic12], y[idx.ic12], ratd[idx.ir1212])
        + m3(-y[idx.ic12], y[idx.io16], ratd[idx.ir1216])
        + m5(
            1.0 / 6.0,
            y[idx.ihe4],
            y[idx.ihe4],
            y[idx.ihe4],
            ratd[idx.ir3a],
        )
        + m2(-y[idx.ic12], ratd[idx.irg3a])
        + m3(-y[idx.ic12], y[idx.ihe4], ratd[idx.ircag])
        + m2(y[idx.io16], ratd[idx.iroga])
        + m3(-y[idx.ic12], y[idx.ih1], ratd[idx.ircpg]);
    if !deriva {
        q_c12 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], ratd[idx.irnpg]);
    } else {
        q_c12 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], rat0[idx.irnpg])
            + m4(y[idx.in14], y[idx.ih1], rat0[idx.ifa], ratd[idx.irnpg]);
    }
    dydt[idx.ic12] = q_c12.to_f64();

    // n14 (lines 1147–1152)
    let q_n14 = m3(y[idx.ic12], y[idx.ih1], ratd[idx.ircpg])
        + m3(-y[idx.in14], y[idx.ih1], ratd[idx.irnpg])
        + m3(y[idx.io16], y[idx.ih1], ratd[idx.iropg])
        + m3(-y[idx.ihe4], y[idx.in14], ratd[idx.irnag]);
    dydt[idx.in14] = q_n14.to_f64();

    // o16 (lines 1156–1173)
    let mut q_o16 = m3(-y[idx.ic12], y[idx.io16], ratd[idx.ir1216])
        + m3(-y[idx.io16], y[idx.io16], ratd[idx.ir1616])
        + m3(y[idx.ic12], y[idx.ihe4], ratd[idx.ircag])
        + m3(-y[idx.io16], y[idx.ihe4], ratd[idx.iroag])
        + m2(-y[idx.io16], ratd[idx.iroga])
        + m2(y[idx.ine20], ratd[idx.irnega])
        + m3(-y[idx.io16], y[idx.ih1], ratd[idx.iropg]);
    if !deriva {
        q_o16 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifg], ratd[idx.irnpg]);
    } else {
        q_o16 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifg], rat0[idx.irnpg])
            + m4(y[idx.in14], y[idx.ih1], rat0[idx.ifg], ratd[idx.irnpg]);
    }
    dydt[idx.io16] = q_o16.to_f64();

    // ne20 (lines 1177–1184)
    let q_ne20 = m4(0.5, y[idx.ic12], y[idx.ic12], ratd[idx.ir1212])
        + m3(y[idx.io16], y[idx.ihe4], ratd[idx.iroag])
        + m3(-y[idx.ine20], y[idx.ihe4], ratd[idx.irneag])
        + m2(-y[idx.ine20], ratd[idx.irnega])
        + m2(y[idx.img24], ratd[idx.irmgga])
        + m3(y[idx.in14], y[idx.ihe4], ratd[idx.irnag]);
    dydt[idx.ine20] = q_ne20.to_f64();

    // mg24 (lines 1188–1209)
    let mut q_mg24 = m4(0.5, y[idx.ic12], y[idx.io16], ratd[idx.ir1216])
        + m3(y[idx.ine20], y[idx.ihe4], ratd[idx.irneag])
        + m3(-y[idx.img24], y[idx.ihe4], ratd[idx.irmgag])
        + m2(-y[idx.img24], ratd[idx.irmgga])
        + m2(y[idx.isi28], ratd[idx.irsiga]);
    if !deriva {
        q_mg24 += m4(
            -y[idx.img24],
            y[idx.ihe4],
            ratd[idx.irmgap],
            1.0 - ratd[idx.irr1],
        ) + m3(y[idx.isi28], ratd[idx.irr1], ratd[idx.irsigp]);
    } else {
        q_mg24 +=
            m4(
                -y[idx.img24],
                y[idx.ihe4],
                ratd[idx.irmgap],
                1.0 - rat0[idx.irr1],
            ) + m4(y[idx.img24], y[idx.ihe4], rat0[idx.irmgap], ratd[idx.irr1])
                + m3(y[idx.isi28], rat0[idx.irr1], ratd[idx.irsigp])
                + m3(y[idx.isi28], ratd[idx.irr1], rat0[idx.irsigp]);
    }
    dydt[idx.img24] = q_mg24.to_f64();

    // si28 (lines 1213–1246)
    let mut q_si28 = m4(0.5, y[idx.ic12], y[idx.io16], ratd[idx.ir1216])
        + m5(0.56, 0.5, y[idx.io16], y[idx.io16], ratd[idx.ir1616])
        + m3(y[idx.img24], y[idx.ihe4], ratd[idx.irmgag])
        + m3(-y[idx.isi28], y[idx.ihe4], ratd[idx.irsiag])
        + m2(-y[idx.isi28], ratd[idx.irsiga])
        + m2(y[idx.is32], ratd[idx.irsga]);
    if !deriva {
        q_si28 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.irs1],
            ratd[idx.ir1616],
        ) + m4(
            y[idx.img24],
            y[idx.ihe4],
            ratd[idx.irmgap],
            1.0 - ratd[idx.irr1],
        ) + m3(-y[idx.isi28], ratd[idx.irr1], ratd[idx.irsigp])
            + m4(
                -y[idx.isi28],
                y[idx.ihe4],
                ratd[idx.irsiap],
                1.0 - ratd[idx.irs1],
            )
            + m3(y[idx.is32], ratd[idx.irs1], ratd[idx.irsgp]);
    } else {
        q_si28 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            rat0[idx.irs1],
            ratd[idx.ir1616],
        ) + m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.irs1],
            rat0[idx.ir1616],
        ) + m4(
            y[idx.img24],
            y[idx.ihe4],
            ratd[idx.irmgap],
            1.0 - rat0[idx.irr1],
        ) + m4(
            -y[idx.img24],
            y[idx.ihe4],
            rat0[idx.irmgap],
            ratd[idx.irr1],
        ) + m3(-y[idx.isi28], rat0[idx.irr1], ratd[idx.irsigp])
            + m3(-y[idx.isi28], ratd[idx.irr1], rat0[idx.irsigp])
            + m4(
                -y[idx.isi28],
                y[idx.ihe4],
                ratd[idx.irsiap],
                1.0 - rat0[idx.irs1],
            )
            + m4(y[idx.isi28], y[idx.ihe4], rat0[idx.irsiap], ratd[idx.irs1])
            + m3(y[idx.is32], rat0[idx.irs1], ratd[idx.irsgp])
            + m3(y[idx.is32], ratd[idx.irs1], rat0[idx.irsgp]);
    }
    dydt[idx.isi28] = q_si28.to_f64();

    // s32 (lines 1250–1282)
    let mut q_s32 = m5(0.1, 0.5, y[idx.io16], y[idx.io16], ratd[idx.ir1616])
        + m3(y[idx.isi28], y[idx.ihe4], ratd[idx.irsiag])
        + m3(-y[idx.is32], y[idx.ihe4], ratd[idx.irsag])
        + m2(-y[idx.is32], ratd[idx.irsga])
        + m2(y[idx.iar36], ratd[idx.irarga]);
    if !deriva {
        q_s32 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.ir1616],
            1.0 - ratd[idx.irs1],
        ) + m4(
            y[idx.isi28],
            y[idx.ihe4],
            ratd[idx.irsiap],
            1.0 - ratd[idx.irs1],
        ) + m3(-y[idx.is32], ratd[idx.irs1], ratd[idx.irsgp])
            + m4(
                -y[idx.is32],
                y[idx.ihe4],
                ratd[idx.irsap],
                1.0 - ratd[idx.irt1],
            )
            + m3(y[idx.iar36], ratd[idx.irt1], ratd[idx.irargp]);
    } else {
        q_s32 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.ir1616],
            1.0 - rat0[idx.irs1],
        ) + m6(
            -0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            rat0[idx.ir1616],
            ratd[idx.irs1],
        ) + m4(
            y[idx.isi28],
            y[idx.ihe4],
            ratd[idx.irsiap],
            1.0 - rat0[idx.irs1],
        ) + m4(
            -y[idx.isi28],
            y[idx.ihe4],
            rat0[idx.irsiap],
            ratd[idx.irs1],
        ) + m3(-y[idx.is32], rat0[idx.irs1], ratd[idx.irsgp])
            + m3(-y[idx.is32], ratd[idx.irs1], rat0[idx.irsgp])
            + m4(
                -y[idx.is32],
                y[idx.ihe4],
                ratd[idx.irsap],
                1.0 - rat0[idx.irt1],
            )
            + m4(y[idx.is32], y[idx.ihe4], rat0[idx.irsap], ratd[idx.irt1])
            + m3(y[idx.iar36], rat0[idx.irt1], ratd[idx.irargp])
            + m3(y[idx.iar36], ratd[idx.irt1], rat0[idx.irargp]);
    }
    dydt[idx.is32] = q_s32.to_f64();

    // ar36, ca40, ti44, cr48 (lines 1286–1403)
    for (
        iso,
        ym1,
        y0,
        yp1,
        rag0,
        rag1,
        rga0,
        rga1,
        rap0,
        r1_0,
        rgp0,
        rap1,
        r1_1,
        rgp1,
    ) in [
        (
            idx.iar36,
            y[idx.is32],
            y[idx.iar36],
            y[idx.ica40],
            idx.irsag,
            idx.irarag,
            idx.irarga,
            idx.ircaga,
            idx.irsap,
            idx.irt1,
            idx.irargp,
            idx.irarap,
            idx.iru1,
            idx.ircagp,
        ),
        (
            idx.ica40,
            y[idx.iar36],
            y[idx.ica40],
            y[idx.iti44],
            idx.irarag,
            idx.ircaag,
            idx.ircaga,
            idx.irtiga,
            idx.irarap,
            idx.iru1,
            idx.ircagp,
            idx.ircaap,
            idx.irv1,
            idx.irtigp,
        ),
        (
            idx.iti44,
            y[idx.ica40],
            y[idx.iti44],
            y[idx.icr48],
            idx.ircaag,
            idx.irtiag,
            idx.irtiga,
            idx.ircrga,
            idx.ircaap,
            idx.irv1,
            idx.irtigp,
            idx.irtiap,
            idx.irw1,
            idx.ircrgp,
        ),
        (
            idx.icr48,
            y[idx.iti44],
            y[idx.icr48],
            y[idx.ife52],
            idx.irtiag,
            idx.ircrag,
            idx.ircrga,
            idx.irfega,
            idx.irtiap,
            idx.irw1,
            idx.ircrgp,
            idx.ircrap,
            idx.irx1,
            idx.irfegp,
        ),
    ] {
        let mut q = m3(ym1, y[idx.ihe4], ratd[rag0])
            + m3(-y0, y[idx.ihe4], ratd[rag1])
            + m2(-y0, ratd[rga0])
            + m2(yp1, ratd[rga1]);
        if !deriva {
            q += m4(ym1, y[idx.ihe4], ratd[rap0], 1.0 - ratd[r1_0])
                + m3(-y0, ratd[r1_0], ratd[rgp0])
                + m4(-y0, y[idx.ihe4], ratd[rap1], 1.0 - ratd[r1_1])
                + m3(yp1, ratd[r1_1], ratd[rgp1]);
        } else {
            q += m4(ym1, y[idx.ihe4], ratd[rap0], 1.0 - rat0[r1_0])
                + m4(-ym1, y[idx.ihe4], rat0[rap0], ratd[r1_0])
                + m3(-y0, rat0[r1_0], ratd[rgp0])
                + m3(-y0, ratd[r1_0], rat0[rgp0])
                + m4(-y0, y[idx.ihe4], ratd[rap1], 1.0 - rat0[r1_1])
                + m4(y0, y[idx.ihe4], rat0[rap1], ratd[r1_1])
                + m3(yp1, rat0[r1_1], ratd[rgp1])
                + m3(yp1, ratd[r1_1], rat0[rgp1]);
        }
        dydt[iso] = q.to_f64();
    }

    approx21_dydt_iron_group(
        idx,
        y,
        rat0,
        ratd,
        deriva,
        fe56ec_fake_factor,
        nf,
        dydt,
    );
}
