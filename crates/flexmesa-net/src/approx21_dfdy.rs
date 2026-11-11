// SPDX-License-Identifier: LGPL-3.0-or-later
//! Hardwired `approx21` and `approx21_plus_co56` analytical Jacobian
//! (`approx21_dfdy`, light-through-`s32` rows; delegates iron-group rows to
//! `approx21_dfdy_heavy`).

use crate::approx21::Approx21Indices;
use crate::approx21_dfdy_heavy::approx21_dfdy_heavy;

/// Evaluates `approx21_dfdy` (`net/private/net_approx21.f90` lines 2033–2652)
/// into row-major `dfdy[i * species + j]`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn approx21_dfdy(
    idx: &Approx21Indices,
    species: usize,
    y: &[f64],
    ratdum: &[f64],
    dratdumdy1: &[f64],
    dratdumdy2: &[f64],
    fe56ec_fake_factor_in: f64,
    min_t_for_fe56ec_fake_factor: f64,
    fe56ec_n_neut: i32,
    temp: f64,
    dfdy: &mut [f64],
) {
    dfdy.fill(0.0);
    let fe56ec_fake_factor = if temp >= min_t_for_fe56ec_fake_factor {
        fe56ec_fake_factor_in
    } else {
        0.0
    };
    let nf = f64::from(fe56ec_n_neut);
    let mut set = |i: usize, j: usize, v: f64| {
        dfdy[i * species + j] = v;
    };

    // h1 row (lines 2059–2078)
    set(
        idx.ih1,
        idx.ih1,
        -3.0 * y[idx.ih1] * ratdum[idx.irpp]
            - 2.0 * y[idx.ic12] * ratdum[idx.ircpg]
            - 2.0 * y[idx.in14] * ratdum[idx.irnpg]
            - 2.0 * y[idx.in14] * y[idx.ih1] * dratdumdy1[idx.irnpg]
            - 2.0 * y[idx.io16] * ratdum[idx.iropg]
            - 2.0 * y[idx.io16] * y[idx.ih1] * dratdumdy1[idx.iropg]
            - 3.0 * ratdum[idx.irpen],
    );
    set(
        idx.ih1,
        idx.ihe3,
        2.0 * y[idx.ihe3] * ratdum[idx.ir33]
            - y[idx.ihe4] * ratdum[idx.irhe3ag],
    );
    set(
        idx.ih1,
        idx.ihe4,
        -y[idx.ihe3] * ratdum[idx.irhe3ag]
            - y[idx.ihe3] * y[idx.ihe4] * dratdumdy1[idx.irhe3ag],
    );
    set(idx.ih1, idx.ic12, -2.0 * y[idx.ih1] * ratdum[idx.ircpg]);
    set(idx.ih1, idx.in14, -2.0 * y[idx.ih1] * ratdum[idx.irnpg]);
    set(idx.ih1, idx.io16, -2.0 * y[idx.ih1] * ratdum[idx.iropg]);

    // he3 row (lines 2081–2088)
    set(
        idx.ihe3,
        idx.ih1,
        y[idx.ih1] * ratdum[idx.irpp] + ratdum[idx.irpen],
    );
    set(
        idx.ihe3,
        idx.ihe3,
        -2.0 * y[idx.ihe3] * ratdum[idx.ir33]
            - y[idx.ihe4] * ratdum[idx.irhe3ag],
    );
    set(
        idx.ihe3,
        idx.ihe4,
        -y[idx.ihe3] * ratdum[idx.irhe3ag]
            - y[idx.ihe3] * y[idx.ihe4] * dratdumdy1[idx.irhe3ag],
    );

    // he4 row (lines 2092–2223)
    set(
        idx.ihe4,
        idx.ih1,
        y[idx.in14] * ratdum[idx.ifa] * ratdum[idx.irnpg]
            + y[idx.in14]
                * y[idx.ih1]
                * ratdum[idx.ifa]
                * dratdumdy1[idx.irnpg]
            + y[idx.io16] * ratdum[idx.iropg]
            + y[idx.io16] * y[idx.ih1] * dratdumdy1[idx.iropg],
    );
    set(
        idx.ihe4,
        idx.ihe3,
        y[idx.ihe3] * ratdum[idx.ir33] + y[idx.ihe4] * ratdum[idx.irhe3ag],
    );
    set(
        idx.ihe4,
        idx.ihe4,
        -1.5 * y[idx.ihe4] * y[idx.ihe4] * ratdum[idx.ir3a]
            - y[idx.ic12] * ratdum[idx.ircag]
            - y[idx.io16] * ratdum[idx.iroag]
            - y[idx.ine20] * ratdum[idx.irneag]
            - y[idx.img24] * ratdum[idx.irmgag]
            - y[idx.isi28] * ratdum[idx.irsiag]
            - y[idx.is32] * ratdum[idx.irsag]
            - y[idx.iar36] * ratdum[idx.irarag]
            - y[idx.ica40] * ratdum[idx.ircaag]
            - y[idx.iti44] * ratdum[idx.irtiag]
            - y[idx.icr48] * ratdum[idx.ircrag]
            - y[idx.ife52] * ratdum[idx.irfeag]
            - y[idx.img24] * ratdum[idx.irmgap] * (1.0 - ratdum[idx.irr1])
            - y[idx.isi28] * ratdum[idx.irsiap] * (1.0 - ratdum[idx.irs1])
            - y[idx.is32] * ratdum[idx.irsap] * (1.0 - ratdum[idx.irt1])
            - y[idx.iar36] * ratdum[idx.irarap] * (1.0 - ratdum[idx.iru1])
            - y[idx.ica40] * ratdum[idx.ircaap] * (1.0 - ratdum[idx.irv1])
            - y[idx.iti44] * ratdum[idx.irtiap] * (1.0 - ratdum[idx.irw1])
            - y[idx.icr48] * ratdum[idx.ircrap] * (1.0 - ratdum[idx.irx1])
            - y[idx.ife52] * ratdum[idx.ir6f54]
            - y[idx.ife52] * y[idx.iprot] * ratdum[idx.ir7f54]
            - ratdum[idx.iralf1]
            - y[idx.ife54] * ratdum[idx.irfe56_aux4]
            + y[idx.ihe3] * ratdum[idx.irhe3ag]
            + y[idx.ihe3] * y[idx.ihe4] * dratdumdy1[idx.irhe3ag]
            - y[idx.in14] * ratdum[idx.irnag] * 1.5,
    );
    set(
        idx.ihe4,
        idx.ic12,
        y[idx.ic12] * ratdum[idx.ir1212]
            + 0.5 * y[idx.io16] * ratdum[idx.ir1216]
            + 3.0 * ratdum[idx.irg3a]
            - y[idx.ihe4] * ratdum[idx.ircag],
    );
    set(
        idx.ihe4,
        idx.in14,
        y[idx.ih1] * ratdum[idx.ifa] * ratdum[idx.irnpg]
            - y[idx.ihe4] * ratdum[idx.irnag] * 1.5,
    );
    set(
        idx.ihe4,
        idx.io16,
        0.5 * y[idx.ic12] * ratdum[idx.ir1216]
            + 1.12 * 0.5 * y[idx.io16] * ratdum[idx.ir1616]
            + 0.68 * ratdum[idx.irs1] * 0.5 * y[idx.io16] * ratdum[idx.ir1616]
            + ratdum[idx.iroga]
            - y[idx.ihe4] * ratdum[idx.iroag]
            + y[idx.ih1] * ratdum[idx.iropg],
    );
    set(
        idx.ihe4,
        idx.ine20,
        ratdum[idx.irnega] - y[idx.ihe4] * ratdum[idx.irneag],
    );
    set(
        idx.ihe4,
        idx.img24,
        ratdum[idx.irmgga]
            - y[idx.ihe4] * ratdum[idx.irmgag]
            - y[idx.ihe4] * ratdum[idx.irmgap] * (1.0 - ratdum[idx.irr1]),
    );
    for (col, rga, rag, rap, r1_next, r1_prev, rgp) in [
        (
            idx.isi28, idx.irsiga, idx.irsiag, idx.irsiap, idx.irs1, idx.irr1,
            idx.irsigp,
        ),
        (
            idx.is32, idx.irsga, idx.irsag, idx.irsap, idx.irt1, idx.irs1,
            idx.irsgp,
        ),
        (
            idx.iar36, idx.irarga, idx.irarag, idx.irarap, idx.iru1, idx.irt1,
            idx.irargp,
        ),
        (
            idx.ica40, idx.ircaga, idx.ircaag, idx.ircaap, idx.irv1, idx.iru1,
            idx.ircagp,
        ),
        (
            idx.iti44, idx.irtiga, idx.irtiag, idx.irtiap, idx.irw1, idx.irv1,
            idx.irtigp,
        ),
        (
            idx.icr48, idx.ircrga, idx.ircrag, idx.ircrap, idx.irx1, idx.irw1,
            idx.ircrgp,
        ),
    ] {
        set(
            idx.ihe4,
            col,
            ratdum[rga]
                - y[idx.ihe4] * ratdum[rag]
                - y[idx.ihe4] * ratdum[rap] * (1.0 - ratdum[r1_next])
                + ratdum[r1_prev] * ratdum[rgp],
        );
    }
    set(
        idx.ihe4,
        idx.ife52,
        ratdum[idx.irfega] - y[idx.ihe4] * ratdum[idx.irfeag]
            + ratdum[idx.irx1] * ratdum[idx.irfegp]
            - y[idx.ihe4] * ratdum[idx.ir6f54]
            - y[idx.ihe4] * y[idx.iprot] * ratdum[idx.ir7f54],
    );
    set(
        idx.ihe4,
        idx.ife54,
        y[idx.iprot] * y[idx.iprot] * ratdum[idx.ir5f54]
            - y[idx.ihe4] * ratdum[idx.irfe56_aux4],
    );
    set(
        idx.ihe4,
        idx.ife56,
        y[idx.iprot] * y[idx.iprot] * ratdum[idx.irfe56_aux3],
    );
    set(
        idx.ihe4,
        idx.ini56,
        ratdum[idx.irniga] + y[idx.iprot] * ratdum[idx.ir8f54],
    );
    set(
        idx.ihe4,
        idx.ineut,
        -y[idx.ihe4] * dratdumdy1[idx.iralf1]
            + 2.0
                * y[idx.ineut]
                * y[idx.iprot]
                * y[idx.iprot]
                * ratdum[idx.iralf2]
            + y[idx.ineut]
                * y[idx.ineut]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.iralf2],
    );
    set(
        idx.ihe4,
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
            + y[idx.ini56] * y[idx.iprot] * dratdumdy1[idx.ir8f54]
            - y[idx.ihe4] * dratdumdy2[idx.iralf1]
            + 2.0
                * y[idx.ineut]
                * y[idx.ineut]
                * y[idx.iprot]
                * ratdum[idx.iralf2]
            + y[idx.ineut]
                * y[idx.ineut]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy2[idx.iralf2]
            + 2.0 * y[idx.ife56] * y[idx.iprot] * ratdum[idx.irfe56_aux3]
            + y[idx.ife56]
                * y[idx.iprot]
                * y[idx.iprot]
                * dratdumdy1[idx.irfe56_aux3]
            - y[idx.ihe4] * y[idx.ife54] * dratdumdy1[idx.irfe56_aux4],
    );

    // c12 row (lines 2228–2245)
    set(
        idx.ic12,
        idx.ih1,
        -y[idx.ic12] * ratdum[idx.ircpg]
            + y[idx.in14] * ratdum[idx.ifa] * ratdum[idx.irnpg]
            + y[idx.in14]
                * y[idx.ih1]
                * ratdum[idx.ifa]
                * dratdumdy1[idx.irnpg],
    );
    set(
        idx.ic12,
        idx.ihe4,
        0.5 * y[idx.ihe4] * y[idx.ihe4] * ratdum[idx.ir3a]
            - y[idx.ic12] * ratdum[idx.ircag],
    );
    set(
        idx.ic12,
        idx.ic12,
        -2.0 * y[idx.ic12] * ratdum[idx.ir1212]
            - y[idx.io16] * ratdum[idx.ir1216]
            - ratdum[idx.irg3a]
            - y[idx.ihe4] * ratdum[idx.ircag]
            - y[idx.ih1] * ratdum[idx.ircpg],
    );
    set(
        idx.ic12,
        idx.in14,
        y[idx.ih1] * ratdum[idx.ifa] * ratdum[idx.irnpg],
    );
    set(
        idx.ic12,
        idx.io16,
        -y[idx.ic12] * ratdum[idx.ir1216] + ratdum[idx.iroga],
    );

    // n14 row (lines 2248–2261)
    set(
        idx.in14,
        idx.ih1,
        y[idx.ic12] * ratdum[idx.ircpg]
            - y[idx.in14] * ratdum[idx.irnpg]
            - y[idx.in14] * y[idx.ih1] * dratdumdy1[idx.irnpg]
            + y[idx.io16] * ratdum[idx.iropg]
            + y[idx.io16] * y[idx.ih1] * dratdumdy1[idx.iropg],
    );
    set(idx.in14, idx.ihe4, -y[idx.in14] * ratdum[idx.irnag]);
    set(idx.in14, idx.ic12, y[idx.ih1] * ratdum[idx.ircpg]);
    set(
        idx.in14,
        idx.in14,
        -y[idx.ih1] * ratdum[idx.irnpg] - y[idx.ihe4] * ratdum[idx.irnag],
    );
    set(idx.in14, idx.io16, y[idx.ih1] * ratdum[idx.iropg]);

    // o16 row (lines 2266–2285)
    set(
        idx.io16,
        idx.ih1,
        y[idx.in14] * ratdum[idx.ifg] * ratdum[idx.irnpg]
            + y[idx.in14]
                * y[idx.ih1]
                * ratdum[idx.ifg]
                * dratdumdy1[idx.irnpg]
            - y[idx.io16] * ratdum[idx.iropg]
            - y[idx.io16] * y[idx.ih1] * dratdumdy1[idx.iropg],
    );
    set(
        idx.io16,
        idx.ihe4,
        y[idx.ic12] * ratdum[idx.ircag] - y[idx.io16] * ratdum[idx.iroag],
    );
    set(
        idx.io16,
        idx.ic12,
        -y[idx.io16] * ratdum[idx.ir1216] + y[idx.ihe4] * ratdum[idx.ircag],
    );
    set(
        idx.io16,
        idx.in14,
        y[idx.ih1] * ratdum[idx.ifg] * ratdum[idx.irnpg],
    );
    set(
        idx.io16,
        idx.io16,
        -y[idx.ic12] * ratdum[idx.ir1216]
            - 2.0 * y[idx.io16] * ratdum[idx.ir1616]
            - y[idx.ihe4] * ratdum[idx.iroag]
            - ratdum[idx.iroga]
            - y[idx.ih1] * ratdum[idx.iropg],
    );
    set(idx.io16, idx.ine20, ratdum[idx.irnega]);

    // ne20 row (lines 2289–2302)
    set(
        idx.ine20,
        idx.ihe4,
        y[idx.io16] * ratdum[idx.iroag] - y[idx.ine20] * ratdum[idx.irneag]
            + y[idx.in14] * ratdum[idx.irnag],
    );
    set(idx.ine20, idx.ic12, y[idx.ic12] * ratdum[idx.ir1212]);
    set(idx.ine20, idx.in14, y[idx.ihe4] * ratdum[idx.irnag]);
    set(idx.ine20, idx.io16, y[idx.ihe4] * ratdum[idx.iroag]);
    set(
        idx.ine20,
        idx.ine20,
        -y[idx.ihe4] * ratdum[idx.irneag] - ratdum[idx.irnega],
    );
    set(idx.ine20, idx.img24, ratdum[idx.irmgga]);

    // mg24 row (lines 2307–2322)
    set(
        idx.img24,
        idx.ihe4,
        y[idx.ine20] * ratdum[idx.irneag]
            - y[idx.img24] * ratdum[idx.irmgag]
            - y[idx.img24] * ratdum[idx.irmgap] * (1.0 - ratdum[idx.irr1]),
    );
    set(idx.img24, idx.ic12, 0.5 * y[idx.io16] * ratdum[idx.ir1216]);
    set(idx.img24, idx.io16, 0.5 * y[idx.ic12] * ratdum[idx.ir1216]);
    set(idx.img24, idx.ine20, y[idx.ihe4] * ratdum[idx.irneag]);
    set(
        idx.img24,
        idx.img24,
        -y[idx.ihe4] * ratdum[idx.irmgag]
            - ratdum[idx.irmgga]
            - y[idx.ihe4] * ratdum[idx.irmgap] * (1.0 - ratdum[idx.irr1]),
    );
    set(
        idx.img24,
        idx.isi28,
        ratdum[idx.irsiga] + ratdum[idx.irr1] * ratdum[idx.irsigp],
    );

    // si28 row (lines 2327–2347)
    set(
        idx.isi28,
        idx.ihe4,
        y[idx.img24] * ratdum[idx.irmgag] - y[idx.isi28] * ratdum[idx.irsiag]
            + y[idx.img24] * ratdum[idx.irmgap] * (1.0 - ratdum[idx.irr1])
            - y[idx.isi28] * ratdum[idx.irsiap] * (1.0 - ratdum[idx.irs1]),
    );
    set(idx.isi28, idx.ic12, 0.5 * y[idx.io16] * ratdum[idx.ir1216]);
    set(
        idx.isi28,
        idx.io16,
        0.5 * y[idx.ic12] * ratdum[idx.ir1216]
            + 1.12 * 0.5 * y[idx.io16] * ratdum[idx.ir1616]
            + 0.68 * 0.5 * y[idx.io16] * ratdum[idx.irs1] * ratdum[idx.ir1616],
    );
    set(
        idx.isi28,
        idx.img24,
        y[idx.ihe4] * ratdum[idx.irmgag]
            + y[idx.ihe4] * ratdum[idx.irmgap] * (1.0 - ratdum[idx.irr1]),
    );
    set(
        idx.isi28,
        idx.isi28,
        -y[idx.ihe4] * ratdum[idx.irsiag]
            - ratdum[idx.irsiga]
            - ratdum[idx.irr1] * ratdum[idx.irsigp]
            - y[idx.ihe4] * ratdum[idx.irsiap] * (1.0 - ratdum[idx.irs1]),
    );
    set(
        idx.isi28,
        idx.is32,
        ratdum[idx.irsga] + ratdum[idx.irs1] * ratdum[idx.irsgp],
    );

    // s32 row (lines 2352–2370)
    set(
        idx.is32,
        idx.ihe4,
        y[idx.isi28] * ratdum[idx.irsiag] - y[idx.is32] * ratdum[idx.irsag]
            + y[idx.isi28] * ratdum[idx.irsiap] * (1.0 - ratdum[idx.irs1])
            - y[idx.is32] * ratdum[idx.irsap] * (1.0 - ratdum[idx.irt1]),
    );
    set(
        idx.is32,
        idx.io16,
        0.68 * 0.5
            * y[idx.io16]
            * ratdum[idx.ir1616]
            * (1.0 - ratdum[idx.irs1])
            + 0.2 * 0.5 * y[idx.io16] * ratdum[idx.ir1616],
    );
    set(
        idx.is32,
        idx.isi28,
        y[idx.ihe4] * ratdum[idx.irsiag]
            + y[idx.ihe4] * ratdum[idx.irsiap] * (1.0 - ratdum[idx.irs1]),
    );
    set(
        idx.is32,
        idx.is32,
        -y[idx.ihe4] * ratdum[idx.irsag]
            - ratdum[idx.irsga]
            - ratdum[idx.irs1] * ratdum[idx.irsgp]
            - y[idx.ihe4] * ratdum[idx.irsap] * (1.0 - ratdum[idx.irt1]),
    );
    set(
        idx.is32,
        idx.iar36,
        ratdum[idx.irarga] + ratdum[idx.irt1] * ratdum[idx.irargp],
    );

    approx21_dfdy_heavy(
        idx,
        species,
        y,
        ratdum,
        dratdumdy1,
        dratdumdy2,
        fe56ec_fake_factor,
        nf,
        dfdy,
    );
}
