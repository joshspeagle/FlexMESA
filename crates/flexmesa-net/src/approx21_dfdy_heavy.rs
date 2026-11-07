// SPDX-License-Identifier: LGPL-3.0-or-later
//! Iron-group rows (`ar36`..`prot`) of `approx21_dydt` and `approx21_dfdy`
//! (`net/private/net_approx21.f90` lines 1407–1544 and 2374–2652).

use crate::approx21::Approx21Indices;
use crate::f64x2::F64x2;

/// Populates the `crx`, `fe52`, `fe54`, `fe56`, `co56`, `ni56`, `neut`, and
/// `prot` entries of `dydt` for `approx21_dydt`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(crate) fn approx21_dydt_iron_group(
    idx: &Approx21Indices,
    y: &[f64],
    rat0: &[f64],
    ratd: &[f64],
    deriva: bool,
    fe56ec_fake_factor: f64,
    nf: f64,
    dydt: &mut [f64],
) {
    let m2 = |a: f64, b: f64| F64x2::from_f64(a * b);
    let m3 = |a: f64, b: f64, c: f64| F64x2::from_f64(a * b * c);
    let m4 = |a: f64, b: f64, c: f64, d: f64| F64x2::from_f64(a * b * c * d);
    let m5 = |a: f64, b: f64, c: f64, d: f64, e: f64| {
        F64x2::from_f64(a * b * c * d * e)
    };
    let m6 = |a: f64, b: f64, c: f64, d: f64, e: f64, f: f64| {
        F64x2::from_f64(a * b * c * d * e * f)
    };

    let q_crx = m3(y[idx.ife56], fe56ec_fake_factor, ratd[idx.irn56ec]);
    dydt[idx.icrx] = q_crx.to_f64();

    let mut q_fe52 = m3(y[idx.icr48], y[idx.ihe4], ratd[idx.ircrag])
        + m3(-y[idx.ife52], y[idx.ihe4], ratd[idx.irfeag])
        + m2(-y[idx.ife52], ratd[idx.irfega])
        + m2(y[idx.ini56], ratd[idx.irniga]);
    if !deriva {
        q_fe52 += m4(
            y[idx.icr48],
            y[idx.ihe4],
            ratd[idx.ircrap],
            1.0 - ratd[idx.irx1],
        ) + m3(-y[idx.ife52], ratd[idx.irx1], ratd[idx.irfegp]);
    } else {
        q_fe52 += m4(
            y[idx.icr48],
            y[idx.ihe4],
            ratd[idx.ircrap],
            1.0 - rat0[idx.irx1],
        ) + m4(
            -y[idx.icr48],
            y[idx.ihe4],
            rat0[idx.ircrap],
            ratd[idx.irx1],
        ) + m3(-y[idx.ife52], rat0[idx.irx1], ratd[idx.irfegp])
            + m3(-y[idx.ife52], ratd[idx.irx1], rat0[idx.irfegp]);
    }
    q_fe52 += m2(y[idx.ife54], ratd[idx.ir1f54])
        + m4(-y[idx.ife52], y[idx.ineut], y[idx.ineut], ratd[idx.ir2f54])
        + m4(y[idx.ife54], y[idx.iprot], y[idx.iprot], ratd[idx.ir5f54])
        + m3(-y[idx.ife52], y[idx.ihe4], ratd[idx.ir6f54])
        + m4(-y[idx.ife52], y[idx.ihe4], y[idx.iprot], ratd[idx.ir7f54])
        + m3(y[idx.ini56], y[idx.iprot], ratd[idx.ir8f54]);
    dydt[idx.ife52] = q_fe52.to_f64();

    let q_fe54 = m2(-y[idx.ife54], ratd[idx.ir1f54])
        + m4(y[idx.ife52], y[idx.ineut], y[idx.ineut], ratd[idx.ir2f54])
        + m4(-y[idx.ife54], y[idx.iprot], y[idx.iprot], ratd[idx.ir3f54])
        + m2(y[idx.ini56], ratd[idx.ir4f54])
        + m4(-y[idx.ife54], y[idx.iprot], y[idx.iprot], ratd[idx.ir5f54])
        + m3(y[idx.ife52], y[idx.ihe4], ratd[idx.ir6f54])
        + m2(y[idx.ife56], ratd[idx.irfe56_aux1])
        + m4(
            -y[idx.ife54],
            y[idx.ineut],
            y[idx.ineut],
            ratd[idx.irfe56_aux2],
        )
        + m4(
            y[idx.ife56],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.irfe56_aux3],
        )
        + m3(-y[idx.ife54], y[idx.ihe4], ratd[idx.irfe56_aux4]);
    dydt[idx.ife54] = q_fe54.to_f64();

    let a1_fe56 = if idx.plus_co56 {
        m2(y[idx.ico56], ratd[idx.irco56ec])
    } else {
        m2(y[idx.ini56], ratd[idx.irn56ec])
    };
    let q_fe56 = a1_fe56
        + m3(-y[idx.ife56], fe56ec_fake_factor, ratd[idx.irn56ec])
        + m2(-y[idx.ife56], ratd[idx.irfe56_aux1])
        + m4(
            y[idx.ife54],
            y[idx.ineut],
            y[idx.ineut],
            ratd[idx.irfe56_aux2],
        )
        + m4(
            -y[idx.ife56],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.irfe56_aux3],
        )
        + m3(y[idx.ife54], y[idx.ihe4], ratd[idx.irfe56_aux4]);
    dydt[idx.ife56] = q_fe56.to_f64();

    if idx.plus_co56 {
        let q_co56 = m2(y[idx.ini56], ratd[idx.irn56ec])
            + m2(-y[idx.ico56], ratd[idx.irco56ec]);
        dydt[idx.ico56] = q_co56.to_f64();
    }

    let mut q_ni56 = m3(y[idx.ife52], y[idx.ihe4], ratd[idx.irfeag])
        + m2(-y[idx.ini56], ratd[idx.irniga])
        + m2(-y[idx.ini56], ratd[idx.irn56ec]);
    q_ni56 += m4(y[idx.ife54], y[idx.iprot], y[idx.iprot], ratd[idx.ir3f54])
        + m2(-y[idx.ini56], ratd[idx.ir4f54])
        + m4(y[idx.ife52], y[idx.ihe4], y[idx.iprot], ratd[idx.ir7f54])
        + m3(-y[idx.ini56], y[idx.iprot], ratd[idx.ir8f54]);
    dydt[idx.ini56] = q_ni56.to_f64();

    let q_neut = m3(2.0, y[idx.ife54], ratd[idx.ir1f54])
        + m5(
            -2.0,
            y[idx.ife52],
            y[idx.ineut],
            y[idx.ineut],
            ratd[idx.ir2f54],
        )
        + m3(2.0, y[idx.ihe4], ratd[idx.iralf1])
        + m6(
            -2.0,
            y[idx.ineut],
            y[idx.ineut],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.iralf2],
        )
        + m2(y[idx.iprot], ratd[idx.irpen])
        + m2(-y[idx.ineut], ratd[idx.irnep])
        + m3(2.0, y[idx.ife56], ratd[idx.irfe56_aux1])
        + m5(
            -2.0,
            y[idx.ife54],
            y[idx.ineut],
            y[idx.ineut],
            ratd[idx.irfe56_aux2],
        )
        + m4(-nf, y[idx.ife56], fe56ec_fake_factor, ratd[idx.irn56ec]);
    dydt[idx.ineut] = q_neut.to_f64();

    let q_prot = m5(
        -2.0,
        y[idx.ife54],
        y[idx.iprot],
        y[idx.iprot],
        ratd[idx.ir3f54],
    ) + m3(2.0, y[idx.ini56], ratd[idx.ir4f54])
        + m5(
            -2.0,
            y[idx.ife54],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.ir5f54],
        )
        + m4(2.0, y[idx.ife52], y[idx.ihe4], ratd[idx.ir6f54])
        + m3(2.0, y[idx.ihe4], ratd[idx.iralf1])
        + m6(
            -2.0,
            y[idx.ineut],
            y[idx.ineut],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.iralf2],
        )
        + m2(-y[idx.iprot], ratd[idx.irpen])
        + m2(y[idx.ineut], ratd[idx.irnep])
        + m5(
            -2.0,
            y[idx.ife56],
            y[idx.iprot],
            y[idx.iprot],
            ratd[idx.irfe56_aux3],
        )
        + m4(2.0, y[idx.ife54], y[idx.ihe4], ratd[idx.irfe56_aux4]);
    dydt[idx.iprot] = q_prot.to_f64();
}

/// Populates the `ar36` through `prot` rows of `approx21_dfdy`
/// (`net/private/net_approx21.f90` lines 2374–2652).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(crate) fn approx21_dfdy_heavy(
    idx: &Approx21Indices,
    species: usize,
    y: &[f64],
    ratdum: &[f64],
    dratdumdy1: &[f64],
    dratdumdy2: &[f64],
    fe56ec_fake_factor: f64,
    nf: f64,
    dfdy: &mut [f64],
) {
    for (
        ym1,
        y0,
        yp1,
        rag0,
        rga0,
        rag1,
        rga1,
        rap0,
        r1_0,
        rgp0,
        rap1,
        r1_1,
        rgp1,
    ) in [
        (
            idx.is32, idx.iar36, idx.ica40, idx.irsag, idx.irarga, idx.irarag,
            idx.ircaga, idx.irsap, idx.irt1, idx.irargp, idx.irarap, idx.iru1,
            idx.ircagp,
        ),
        (
            idx.iar36, idx.ica40, idx.iti44, idx.irarag, idx.ircaga,
            idx.ircaag, idx.irtiga, idx.irarap, idx.iru1, idx.ircagp,
            idx.ircaap, idx.irv1, idx.irtigp,
        ),
        (
            idx.ica40, idx.iti44, idx.icr48, idx.ircaag, idx.irtiga,
            idx.irtiag, idx.ircrga, idx.ircaap, idx.irv1, idx.irtigp,
            idx.irtiap, idx.irw1, idx.ircrgp,
        ),
        (
            idx.iti44, idx.icr48, idx.ife52, idx.irtiag, idx.ircrga,
            idx.ircrag, idx.irfega, idx.irtiap, idx.irw1, idx.ircrgp,
            idx.ircrap, idx.irx1, idx.irfegp,
        ),
    ] {
        dfdy[y0 * species + idx.ihe4] = y[ym1] * ratdum[rag0]
            - y[y0] * ratdum[rag1]
            + y[ym1] * ratdum[rap0] * (1.0 - ratdum[r1_0])
            - y[y0] * ratdum[rap1] * (1.0 - ratdum[r1_1]);
        dfdy[y0 * species + ym1] = y[idx.ihe4] * ratdum[rag0]
            + y[idx.ihe4] * ratdum[rap0] * (1.0 - ratdum[r1_0]);
        dfdy[y0 * species + y0] = -y[idx.ihe4] * ratdum[rag1]
            - ratdum[rga0]
            - ratdum[r1_0] * ratdum[rgp0]
            - y[idx.ihe4] * ratdum[rap1] * (1.0 - ratdum[r1_1]);
        dfdy[y0 * species + yp1] = ratdum[rga1] + ratdum[r1_1] * ratdum[rgp1];
    }

    let mut set = |i: usize, j: usize, v: f64| {
        dfdy[i * species + j] = v;
    };

    set(
        idx.icrx,
        idx.ife56,
        fe56ec_fake_factor * ratdum[idx.irn56ec],
    );

    set(
        idx.ife52,
        idx.ihe4,
        y[idx.icr48] * ratdum[idx.ircrag] - y[idx.ife52] * ratdum[idx.irfeag]
            + y[idx.icr48] * ratdum[idx.ircrap] * (1.0 - ratdum[idx.irx1])
            - y[idx.ife52] * ratdum[idx.ir6f54]
            - y[idx.ife52] * y[idx.iprot] * ratdum[idx.ir7f54],
    );
    set(
        idx.ife52,
        idx.icr48,
        y[idx.ihe4] * ratdum[idx.ircrag]
            + y[idx.ihe4] * ratdum[idx.ircrap] * (1.0 - ratdum[idx.irx1]),
    );
    set(
        idx.ife52,
        idx.ife52,
        -y[idx.ihe4] * ratdum[idx.irfeag]
            - ratdum[idx.irfega]
            - ratdum[idx.irx1] * ratdum[idx.irfegp]
            - y[idx.ineut] * y[idx.ineut] * ratdum[idx.ir2f54]
            - y[idx.ihe4] * ratdum[idx.ir6f54]
            - y[idx.ihe4] * y[idx.iprot] * ratdum[idx.ir7f54],
    );
    set(
        idx.ife52,
        idx.ife54,
        ratdum[idx.ir1f54] + y[idx.iprot] * y[idx.iprot] * ratdum[idx.ir5f54],
    );
    set(
        idx.ife52,
        idx.ini56,
        ratdum[idx.irniga] + y[idx.iprot] * ratdum[idx.ir8f54],
    );
    set(
        idx.ife52,
        idx.ineut,
        y[idx.ife54] * dratdumdy1[idx.ir1f54]
            - 2.0 * y[idx.ife52] * y[idx.ineut] * ratdum[idx.ir2f54]
            - y[idx.ife52]
                * y[idx.ineut]
                * y[idx.ineut]
                * dratdumdy1[idx.ir2f54],
    );
    set(
        idx.ife52,
        idx.iprot,
        2.0 * y[idx.ife54] * y[idx.iprot] * ratdum[idx.ir5f54]
            + y[idx.ife54]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.ir5f54]
            - y[idx.ihe4] * y[idx.ife52] * dratdumdy1[idx.ir6f54]
            - y[idx.ife52] * y[idx.ihe4] * ratdum[idx.ir7f54]
            - y[idx.ife52]
                * y[idx.ihe4]
                * y[idx.iprot]
                * dratdumdy1[idx.ir7f54]
            + y[idx.ini56] * ratdum[idx.ir8f54]
            + y[idx.ini56] * y[idx.iprot] * dratdumdy1[idx.ir8f54],
    );

    set(
        idx.ife54,
        idx.ihe4,
        y[idx.ife52] * ratdum[idx.ir6f54]
            - y[idx.ife54] * ratdum[idx.irfe56_aux4],
    );
    set(
        idx.ife54,
        idx.ife52,
        y[idx.ineut] * y[idx.ineut] * ratdum[idx.ir2f54]
            + y[idx.ihe4] * ratdum[idx.ir6f54],
    );
    set(
        idx.ife54,
        idx.ife54,
        -ratdum[idx.ir1f54]
            - y[idx.ineut] * y[idx.ineut] * ratdum[idx.irfe56_aux2]
            - y[idx.iprot] * y[idx.iprot] * ratdum[idx.ir3f54]
            - y[idx.iprot] * y[idx.iprot] * ratdum[idx.ir5f54]
            - y[idx.ihe4] * ratdum[idx.irfe56_aux4],
    );
    set(
        idx.ife54,
        idx.ife56,
        ratdum[idx.irfe56_aux1]
            + y[idx.iprot] * y[idx.iprot] * ratdum[idx.irfe56_aux3],
    );
    set(idx.ife54, idx.ini56, ratdum[idx.ir4f54]);
    set(
        idx.ife54,
        idx.ineut,
        -y[idx.ife54] * dratdumdy1[idx.ir1f54]
            + 2.0 * y[idx.ife52] * y[idx.ineut] * ratdum[idx.ir2f54]
            + y[idx.ife52]
                * y[idx.ineut]
                * y[idx.ineut]
                * dratdumdy1[idx.ir2f54]
            + y[idx.ife56] * dratdumdy1[idx.irfe56_aux1]
            - 2.0 * y[idx.ife54] * y[idx.ineut] * ratdum[idx.irfe56_aux2]
            - y[idx.ife54]
                * y[idx.ineut]
                * y[idx.ineut]
                * dratdumdy1[idx.irfe56_aux2],
    );
    set(
        idx.ife54,
        idx.iprot,
        -2.0 * y[idx.ife54] * y[idx.iprot] * ratdum[idx.ir3f54]
            - y[idx.ife54]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.ir3f54]
            + y[idx.ini56] * dratdumdy1[idx.ir4f54]
            - 2.0 * y[idx.ife54] * y[idx.iprot] * ratdum[idx.ir5f54]
            - y[idx.ife54]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.ir5f54]
            + y[idx.ihe4] * y[idx.ife52] * dratdumdy1[idx.ir6f54]
            + 2.0 * y[idx.ife56] * y[idx.iprot] * ratdum[idx.irfe56_aux3]
            + y[idx.ife56]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.irfe56_aux3]
            - y[idx.ihe4] * y[idx.ife54] * dratdumdy1[idx.irfe56_aux4],
    );

    set(idx.ife56, idx.ihe4, y[idx.ife54] * ratdum[idx.irfe56_aux4]);
    set(
        idx.ife56,
        idx.ife54,
        y[idx.ineut] * y[idx.ineut] * ratdum[idx.irfe56_aux2]
            + y[idx.ihe4] * ratdum[idx.irfe56_aux4],
    );
    set(
        idx.ife56,
        idx.ife56,
        -fe56ec_fake_factor * ratdum[idx.irn56ec]
            - ratdum[idx.irfe56_aux1]
            - y[idx.iprot] * y[idx.iprot] * ratdum[idx.irfe56_aux3],
    );
    if idx.plus_co56 {
        set(idx.ife56, idx.ico56, ratdum[idx.irco56ec]);
        set(idx.ico56, idx.ini56, ratdum[idx.irn56ec]);
        set(idx.ico56, idx.ico56, -ratdum[idx.irco56ec]);
    } else {
        set(idx.ife56, idx.ini56, ratdum[idx.irn56ec]);
    }
    set(
        idx.ife56,
        idx.ineut,
        -y[idx.ife56] * dratdumdy1[idx.irfe56_aux1]
            + 2.0 * y[idx.ife54] * y[idx.ineut] * ratdum[idx.irfe56_aux2]
            + y[idx.ife54]
                * y[idx.ineut]
                * y[idx.ineut]
                * dratdumdy1[idx.irfe56_aux2],
    );
    set(
        idx.ife56,
        idx.iprot,
        -2.0 * y[idx.ife56] * y[idx.iprot] * ratdum[idx.irfe56_aux3]
            - y[idx.ife56]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.irfe56_aux3]
            + y[idx.ihe4] * y[idx.ife54] * dratdumdy1[idx.irfe56_aux4],
    );

    set(
        idx.ini56,
        idx.ihe4,
        y[idx.ife52] * ratdum[idx.irfeag]
            + y[idx.ife52] * y[idx.iprot] * ratdum[idx.ir7f54],
    );
    set(
        idx.ini56,
        idx.ife52,
        y[idx.ihe4] * ratdum[idx.irfeag]
            + y[idx.ihe4] * y[idx.iprot] * ratdum[idx.ir7f54],
    );
    set(
        idx.ini56,
        idx.ife54,
        y[idx.iprot] * y[idx.iprot] * ratdum[idx.ir3f54],
    );
    set(
        idx.ini56,
        idx.ini56,
        -ratdum[idx.irniga]
            - ratdum[idx.ir4f54]
            - y[idx.iprot] * ratdum[idx.ir8f54]
            - ratdum[idx.irn56ec],
    );
    set(
        idx.ini56,
        idx.iprot,
        2.0 * y[idx.ife54] * y[idx.iprot] * ratdum[idx.ir3f54]
            + y[idx.ife54]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.ir3f54]
            - y[idx.ini56] * dratdumdy1[idx.ir4f54]
            + y[idx.ife52] * y[idx.ihe4] * ratdum[idx.ir7f54]
            + y[idx.ife52]
                * y[idx.ihe4]
                * y[idx.iprot]
                * dratdumdy1[idx.ir7f54]
            - y[idx.ini56] * ratdum[idx.ir8f54]
            - y[idx.ini56] * y[idx.iprot] * dratdumdy1[idx.ir8f54],
    );

    set(idx.ineut, idx.ihe4, 2.0 * ratdum[idx.iralf1]);
    set(
        idx.ineut,
        idx.ife52,
        -2.0 * y[idx.ineut] * y[idx.ineut] * ratdum[idx.ir2f54],
    );
    set(
        idx.ineut,
        idx.ife54,
        2.0 * ratdum[idx.ir1f54]
            - 2.0 * y[idx.ineut] * y[idx.ineut] * ratdum[idx.irfe56_aux2],
    );
    set(
        idx.ineut,
        idx.ife56,
        2.0 * ratdum[idx.irfe56_aux1]
            - nf * fe56ec_fake_factor * ratdum[idx.irn56ec],
    );
    let yn2 = y[idx.ineut] * y[idx.ineut];
    let yp2 = y[idx.iprot] * y[idx.iprot];
    let d_alf_dn = 2.0 * y[idx.ihe4] * dratdumdy1[idx.iralf1]
        - 4.0 * y[idx.ineut] * yp2 * ratdum[idx.iralf2]
        - 2.0 * yn2 * yp2 * dratdumdy1[idx.iralf2];
    let d_alf_dp = 2.0 * y[idx.ihe4] * dratdumdy2[idx.iralf1]
        - 4.0 * yn2 * y[idx.iprot] * ratdum[idx.iralf2]
        - 2.0 * yn2 * yp2 * dratdumdy2[idx.iralf2];

    set(
        idx.ineut,
        idx.ineut,
        2.0 * y[idx.ife54] * dratdumdy1[idx.ir1f54]
            - 4.0 * y[idx.ife52] * y[idx.ineut] * ratdum[idx.ir2f54]
            - 2.0 * y[idx.ife52] * yn2 * dratdumdy1[idx.ir2f54]
            + d_alf_dn
            - ratdum[idx.irnep]
            + 2.0 * y[idx.ife56] * dratdumdy1[idx.irfe56_aux1]
            - 4.0 * y[idx.ife54] * y[idx.ineut] * ratdum[idx.irfe56_aux2]
            - 2.0 * y[idx.ife54] * yn2 * dratdumdy1[idx.irfe56_aux2],
    );
    set(idx.ineut, idx.iprot, d_alf_dp + ratdum[idx.irpen]);

    set(
        idx.iprot,
        idx.ihe4,
        2.0 * y[idx.ife52] * ratdum[idx.ir6f54]
            + 2.0 * ratdum[idx.iralf1]
            + 2.0 * y[idx.ife54] * ratdum[idx.irfe56_aux4],
    );
    set(idx.iprot, idx.ife52, 2.0 * y[idx.ihe4] * ratdum[idx.ir6f54]);
    set(
        idx.iprot,
        idx.ife54,
        -2.0 * yp2 * ratdum[idx.ir3f54] - 2.0 * yp2 * ratdum[idx.ir5f54]
            + 2.0 * y[idx.ihe4] * ratdum[idx.irfe56_aux4],
    );
    set(idx.iprot, idx.ife56, -2.0 * yp2 * ratdum[idx.irfe56_aux3]);
    set(idx.iprot, idx.ini56, 2.0 * ratdum[idx.ir4f54]);
    set(idx.iprot, idx.ineut, d_alf_dn + ratdum[idx.irnep]);
    set(
        idx.iprot,
        idx.iprot,
        -4.0 * y[idx.ife54] * y[idx.iprot] * ratdum[idx.ir3f54]
            - 2.0 * y[idx.ife54] * yp2 * dratdumdy1[idx.ir3f54]
            + 2.0 * y[idx.ini56] * dratdumdy1[idx.ir4f54]
            - 4.0 * y[idx.ife54] * y[idx.iprot] * ratdum[idx.ir5f54]
            - 2.0 * y[idx.ife54] * yp2 * dratdumdy1[idx.ir5f54]
            + 2.0 * y[idx.ihe4] * y[idx.ife52] * dratdumdy1[idx.ir6f54]
            + d_alf_dp
            - ratdum[idx.irpen]
            - 4.0 * y[idx.ife56] * y[idx.iprot] * ratdum[idx.irfe56_aux3]
            - 2.0 * y[idx.ife56] * yp2 * dratdumdy1[idx.irfe56_aux3]
            + 2.0 * y[idx.ihe4] * y[idx.ife54] * dratdumdy1[idx.irfe56_aux4],
    );
}
