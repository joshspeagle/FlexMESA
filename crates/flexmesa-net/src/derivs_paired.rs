// SPDX-License-Identifier: LGPL-3.0-or-later
//! Paired forward/reverse REACLIB derivative kernels (`get_general_*` and
//! `get_basic_*` in `net/private/net_derivs.f90` lines 166–775).

use flexmesa_const::{AVO, QCONV};

use crate::derivs_support::NetWorkState;
use crate::f64x2::F64x2;
use crate::types::{NetGeneralInfo, NUM_RVS};

fn poly_y_term(c: u8, y: f64) -> (f64, f64) {
    match c {
        1 => (y, 1.0),
        2 => (0.5 * y * y, y),
        3 => (y * y * y / 6.0, 0.5 * y * y),
        _ => (0.0, 0.0),
    }
}

/// Evaluates a paired $1 \leftrightarrow 1$ stoichiometric reaction
/// (`get_general_1_to_1_derivs` in `net/private/net_derivs.f90`).
pub(crate) fn get_general_1_to_1_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
    i: usize,
    symbolic: bool,
    just_dydt: bool,
) {
    let ni = g.num_isos;
    let r_i = i + 1;
    let rxn = &g.reactions[i];
    let c1 = f64::from(rxn.in_num[0]);
    let i1 = g.net_iso[rxn.in_iso[0] as usize] - 1;
    let c2 = f64::from(rxn.out_num[0]);
    let i2 = g.net_iso[rxn.out_iso[0] as usize] - 1;

    if symbolic {
        for &r in &[i1, i2] {
            for &c in &[i1, i2] {
                w.d_dydt_dy[r * ni + c] = 1.0;
            }
        }
        return;
    }

    let (d_f, d_ysf_dy1) = poly_y_term(rxn.in_num[0], w.y[i1]);
    let (d_r, d_ysr_dy2) = poly_y_term(rxn.out_num[0], w.y[i2]);

    let b_f = F64x2::from_f64(w.rate_screened[i]) * d_f;
    let b_r = F64x2::from_f64(w.rate_screened[r_i]) * d_r;
    let b = b_f - b_r;
    dydt[i1][0] -= b * c1;
    dydt[i2][0] += b * c2;

    w.raw_rate[i] = d_f * w.rate_raw[i] * AVO;
    w.raw_rate[r_i] = d_r * w.rate_raw[r_i] * AVO;
    w.screened_rate[i] = d_f * w.rate_screened[i] * AVO;
    w.screened_rate[r_i] = d_r * w.rate_screened[r_i] * AVO;

    if just_dydt {
        return;
    }

    let q = rxn.std_q;
    let icat_f = (rxn.category as usize).saturating_sub(1);
    let icat_r = (g.reactions[r_i].category as usize).saturating_sub(1);
    let bf_q = b_f * q;
    let br_q = b_r * (-q);
    eps_nuc_mev[0] += bf_q + br_q;
    w.eps_nuc_categories[icat_f] += bf_q.to_f64();
    w.eps_nuc_categories[icat_r] += br_q.to_f64();
    w.eps_nuc_rate[i] = bf_q.to_f64() * QCONV;
    w.eps_nuc_rate[r_i] = br_q.to_f64() * QCONV;

    for (rv_idx, (rf, rr)) in [
        (1, (w.rate_screened_dt[i], w.rate_screened_dt[r_i])),
        (2, (w.rate_screened_drho[i], w.rate_screened_drho[r_i])),
    ] {
        let bf_d = F64x2::from_f64(rf) * d_f;
        let br_d = F64x2::from_f64(rr) * d_r;
        let b_d = bf_d - br_d;
        dydt[i1][rv_idx] -= b_d * c1;
        dydt[i2][rv_idx] += b_d * c2;
        eps_nuc_mev[rv_idx] += bf_d * q + br_d * (-q);
    }

    let d1 = d_ysf_dy1 * w.rate_screened[i];
    let d2 = d_ysr_dy2 * w.rate_screened[r_i];
    w.d_eps_nuc_dy[i1] += q * d1;
    w.d_eps_nuc_dy[i2] -= q * d2;

    w.d_dydt_dy[i1 * ni + i1] -= c1 * d1;
    w.d_dydt_dy[i2 * ni + i1] += c2 * d1;
    w.d_dydt_dy[i1 * ni + i2] += c1 * d2;
    w.d_dydt_dy[i2 * ni + i2] -= c2 * d2;
}

/// Evaluates a paired $2 \leftrightarrow 1$ stoichiometric reaction
/// (`get_general_2_to_1_derivs` in `net/private/net_derivs.f90`).
pub(crate) fn get_general_2_to_1_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
    i: usize,
    symbolic: bool,
    just_dydt: bool,
) {
    let ni = g.num_isos;
    let r_i = i + 1;
    let rxn = &g.reactions[i];
    let c1 = f64::from(rxn.in_num[0]);
    let i1 = g.net_iso[rxn.in_iso[0] as usize] - 1;
    let c2 = f64::from(rxn.in_num[1]);
    let i2 = g.net_iso[rxn.in_iso[1] as usize] - 1;
    let c3 = f64::from(rxn.out_num[0]);
    let i3 = g.net_iso[rxn.out_iso[0] as usize] - 1;

    if symbolic {
        for &r in &[i1, i2, i3] {
            for &c in &[i1, i2, i3] {
                w.d_dydt_dy[r * ni + c] = 1.0;
            }
        }
        return;
    }

    let (ys1, dy1) = poly_y_term(rxn.in_num[0], w.y[i1]);
    let (ys2, dy2) = poly_y_term(rxn.in_num[1], w.y[i2]);
    let d_f = ys1 * ys2;
    let d_ysf_dy1 = dy1 * ys2;
    let d_ysf_dy2 = ys1 * dy2;
    let (d_r, d_ysr_dy3) = poly_y_term(rxn.out_num[0], w.y[i3]);

    let b_f = F64x2::from_f64(w.rate_screened[i]) * d_f;
    let b_r = F64x2::from_f64(w.rate_screened[r_i]) * d_r;
    let b = b_f - b_r;
    dydt[i1][0] -= b * c1;
    dydt[i2][0] -= b * c2;
    dydt[i3][0] += b * c3;

    w.raw_rate[i] = d_f * w.rate_raw[i] * AVO;
    w.raw_rate[r_i] = d_r * w.rate_raw[r_i] * AVO;
    w.screened_rate[i] = d_f * w.rate_screened[i] * AVO;
    w.screened_rate[r_i] = d_r * w.rate_screened[r_i] * AVO;

    if just_dydt {
        return;
    }

    let q = rxn.std_q;
    let icat_f = (rxn.category as usize).saturating_sub(1);
    let icat_r = (g.reactions[r_i].category as usize).saturating_sub(1);
    let bf_q = b_f * q;
    let br_q = b_r * (-q);
    eps_nuc_mev[0] += bf_q + br_q;
    w.eps_nuc_categories[icat_f] += bf_q.to_f64();
    w.eps_nuc_categories[icat_r] += br_q.to_f64();
    w.eps_nuc_rate[i] = bf_q.to_f64() * QCONV;
    w.eps_nuc_rate[r_i] = br_q.to_f64() * QCONV;

    for (rv_idx, (rf, rr)) in [
        (1, (w.rate_screened_dt[i], w.rate_screened_dt[r_i])),
        (2, (w.rate_screened_drho[i], w.rate_screened_drho[r_i])),
    ] {
        let bf_d = F64x2::from_f64(rf) * d_f;
        let br_d = F64x2::from_f64(rr) * d_r;
        let b_d = bf_d - br_d;
        dydt[i1][rv_idx] -= b_d * c1;
        dydt[i2][rv_idx] -= b_d * c2;
        dydt[i3][rv_idx] += b_d * c3;
        eps_nuc_mev[rv_idx] += bf_d * q + br_d * (-q);
    }

    let d1 = d_ysf_dy1 * w.rate_screened[i];
    let d2 = d_ysf_dy2 * w.rate_screened[i];
    let d3 = d_ysr_dy3 * w.rate_screened[r_i];
    w.d_eps_nuc_dy[i1] += q * d1;
    w.d_eps_nuc_dy[i2] += q * d2;
    w.d_eps_nuc_dy[i3] -= q * d3;

    for (row, coeff) in [(i1, -c1), (i2, -c2), (i3, c3)] {
        w.d_dydt_dy[row * ni + i1] += coeff * d1;
        w.d_dydt_dy[row * ni + i2] += coeff * d2;
        w.d_dydt_dy[row * ni + i3] -= coeff * d3;
    }
}

/// Evaluates a paired $2 \leftrightarrow 2$ stoichiometric reaction
/// (`get_general_2_to_2_derivs` in `net/private/net_derivs.f90`).
pub(crate) fn get_general_2_to_2_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
    i: usize,
    symbolic: bool,
    just_dydt: bool,
) {
    let ni = g.num_isos;
    let r_i = i + 1;
    let rxn = &g.reactions[i];
    let c1 = f64::from(rxn.in_num[0]);
    let i1 = g.net_iso[rxn.in_iso[0] as usize] - 1;
    let c2 = f64::from(rxn.in_num[1]);
    let i2 = g.net_iso[rxn.in_iso[1] as usize] - 1;
    let c3 = f64::from(rxn.out_num[0]);
    let i3 = g.net_iso[rxn.out_iso[0] as usize] - 1;
    let c4 = f64::from(rxn.out_num[1]);
    let i4 = g.net_iso[rxn.out_iso[1] as usize] - 1;

    if symbolic {
        for &r in &[i1, i2, i3, i4] {
            for &c in &[i1, i2, i3, i4] {
                w.d_dydt_dy[r * ni + c] = 1.0;
            }
        }
        return;
    }

    let (ys1, dy1) = poly_y_term(rxn.in_num[0], w.y[i1]);
    let (ys2, dy2) = poly_y_term(rxn.in_num[1], w.y[i2]);
    let d_f = ys1 * ys2;
    let d_ysf_dy1 = dy1 * ys2;
    let d_ysf_dy2 = ys1 * dy2;

    let (ys3, dy3) = poly_y_term(rxn.out_num[0], w.y[i3]);
    let (ys4, dy4) = poly_y_term(rxn.out_num[1], w.y[i4]);
    let d_r = ys3 * ys4;
    let d_ysr_dy3 = dy3 * ys4;
    let d_ysr_dy4 = ys3 * dy4;

    let b_f = F64x2::from_f64(w.rate_screened[i]) * d_f;
    let b_r = F64x2::from_f64(w.rate_screened[r_i]) * d_r;
    let b = b_f - b_r;
    dydt[i1][0] -= b * c1;
    dydt[i2][0] -= b * c2;
    dydt[i3][0] += b * c3;
    dydt[i4][0] += b * c4;

    w.raw_rate[i] = d_f * w.rate_raw[i] * AVO;
    w.raw_rate[r_i] = d_r * w.rate_raw[r_i] * AVO;
    w.screened_rate[i] = d_f * w.rate_screened[i] * AVO;
    w.screened_rate[r_i] = d_r * w.rate_screened[r_i] * AVO;

    if just_dydt {
        return;
    }

    let q = rxn.std_q;
    let icat_f = (rxn.category as usize).saturating_sub(1);
    let icat_r = (g.reactions[r_i].category as usize).saturating_sub(1);
    let bf_q = b_f * q;
    let br_q = b_r * (-q);
    eps_nuc_mev[0] += bf_q + br_q;
    w.eps_nuc_categories[icat_f] += bf_q.to_f64();
    w.eps_nuc_categories[icat_r] += br_q.to_f64();
    w.eps_nuc_rate[i] = bf_q.to_f64() * QCONV;
    w.eps_nuc_rate[r_i] = br_q.to_f64() * QCONV;

    for (rv_idx, (rf, rr)) in [
        (1, (w.rate_screened_dt[i], w.rate_screened_dt[r_i])),
        (2, (w.rate_screened_drho[i], w.rate_screened_drho[r_i])),
    ] {
        let bf_d = F64x2::from_f64(rf) * d_f;
        let br_d = F64x2::from_f64(rr) * d_r;
        let b_d = bf_d - br_d;
        dydt[i1][rv_idx] -= b_d * c1;
        dydt[i2][rv_idx] -= b_d * c2;
        dydt[i3][rv_idx] += b_d * c3;
        dydt[i4][rv_idx] += b_d * c4;
        eps_nuc_mev[rv_idx] += bf_d * q + br_d * (-q);
    }

    let d1 = d_ysf_dy1 * w.rate_screened[i];
    let d2 = d_ysf_dy2 * w.rate_screened[i];
    let d3 = d_ysr_dy3 * w.rate_screened[r_i];
    let d4 = d_ysr_dy4 * w.rate_screened[r_i];
    w.d_eps_nuc_dy[i1] += q * d1;
    w.d_eps_nuc_dy[i2] += q * d2;
    w.d_eps_nuc_dy[i3] -= q * d3;
    w.d_eps_nuc_dy[i4] -= q * d4;

    for (row, coeff) in [(i1, -c1), (i2, -c2), (i3, c3), (i4, c4)] {
        w.d_dydt_dy[row * ni + i1] += coeff * d1;
        w.d_dydt_dy[row * ni + i2] += coeff * d2;
        w.d_dydt_dy[row * ni + i3] -= coeff * d3;
        w.d_dydt_dy[row * ni + i4] -= coeff * d4;
    }
}

/// Evaluates a basic $1 + 1 \leftrightarrow 1 + 1$ reaction (`(p, n)`,
/// `(\alpha, p)`, `(\alpha, n)` in `net/private/net_derivs.f90`).
pub(crate) fn get_basic_2_to_2_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
    i: usize,
    symbolic: bool,
    just_dydt: bool,
) {
    let ni = g.num_isos;
    let r_i = i + 1;
    let rxn = &g.reactions[i];
    let i1 = g.net_iso[rxn.in_iso[0] as usize] - 1;
    let i2 = g.net_iso[rxn.in_iso[1] as usize] - 1;
    let i3 = g.net_iso[rxn.out_iso[0] as usize] - 1;
    let i4 = g.net_iso[rxn.out_iso[1] as usize] - 1;

    if symbolic {
        for &r in &[i1, i2, i3, i4] {
            for &c in &[i1, i2, i3, i4] {
                w.d_dydt_dy[r * ni + c] = 1.0;
            }
        }
        return;
    }

    let y1 = w.y[i1];
    let y2 = w.y[i2];
    let y3 = w.y[i3];
    let y4 = w.y[i4];
    let d_f = y1 * y2;
    let d_r = y3 * y4;

    let b_f = F64x2::from_f64(w.rate_screened[i]) * d_f;
    let b_r = F64x2::from_f64(w.rate_screened[r_i]) * d_r;
    let b = b_f - b_r;
    dydt[i1][0] -= b;
    dydt[i2][0] -= b;
    dydt[i3][0] += b;
    dydt[i4][0] += b;

    w.raw_rate[i] = d_f * w.rate_raw[i] * AVO;
    w.raw_rate[r_i] = d_r * w.rate_raw[r_i] * AVO;
    w.screened_rate[i] = d_f * w.rate_screened[i] * AVO;
    w.screened_rate[r_i] = d_r * w.rate_screened[r_i] * AVO;

    if just_dydt {
        return;
    }

    let q = rxn.std_q;
    let icat_f = (rxn.category as usize).saturating_sub(1);
    let icat_r = (g.reactions[r_i].category as usize).saturating_sub(1);
    eps_nuc_mev[0] += b * q;
    w.eps_nuc_categories[icat_f] += (b_f * q).to_f64();
    w.eps_nuc_categories[icat_r] -= (b_r * q).to_f64();
    w.eps_nuc_rate[i] = (b_f * q).to_f64() * QCONV;
    w.eps_nuc_rate[r_i] = (-b_r * q).to_f64() * QCONV;

    for (rv_idx, (rf, rr)) in [
        (1, (w.rate_screened_dt[i], w.rate_screened_dt[r_i])),
        (2, (w.rate_screened_drho[i], w.rate_screened_drho[r_i])),
    ] {
        let bf_d = F64x2::from_f64(rf) * d_f;
        let br_d = F64x2::from_f64(rr) * d_r;
        let b_d = bf_d - br_d;
        dydt[i1][rv_idx] -= b_d;
        dydt[i2][rv_idx] -= b_d;
        dydt[i3][rv_idx] += b_d;
        dydt[i4][rv_idx] += b_d;
        eps_nuc_mev[rv_idx] += b_d * q;
    }

    let e_f = w.rate_screened[i];
    let e_r = w.rate_screened[r_i];
    let d1 = y2 * e_f;
    let d2 = y1 * e_f;
    let d3 = y4 * e_r;
    let d4 = y3 * e_r;

    w.d_eps_nuc_dy[i1] += q * d1;
    w.d_eps_nuc_dy[i2] += q * d2;
    w.d_eps_nuc_dy[i3] -= q * d3;
    w.d_eps_nuc_dy[i4] -= q * d4;

    for (col, dval) in [(i1, d1), (i2, d2), (i3, -d3), (i4, -d4)] {
        w.d_dydt_dy[i1 * ni + col] -= dval;
        w.d_dydt_dy[i2 * ni + col] -= dval;
        w.d_dydt_dy[i3 * ni + col] += dval;
        w.d_dydt_dy[i4 * ni + col] += dval;
    }
}

/// Evaluates a basic $1 + 1 \leftrightarrow 1$ reaction (`(n, \gamma)`,
/// `(p, \gamma)`, `(\alpha, \gamma)` in `net/private/net_derivs.f90`).
pub(crate) fn get_basic_2_to_1_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
    i: usize,
    symbolic: bool,
    just_dydt: bool,
) {
    let ni = g.num_isos;
    let r_i = i + 1;
    let rxn = &g.reactions[i];
    let i1 = g.net_iso[rxn.in_iso[0] as usize] - 1;
    let i2 = g.net_iso[rxn.in_iso[1] as usize] - 1;
    let i3 = g.net_iso[rxn.out_iso[0] as usize] - 1;

    if symbolic {
        for &r in &[i1, i2, i3] {
            for &c in &[i1, i2, i3] {
                w.d_dydt_dy[r * ni + c] = 1.0;
            }
        }
        return;
    }

    let y1 = w.y[i1];
    let y2 = w.y[i2];
    let y3 = w.y[i3];
    let d_f = y1 * y2;
    let d_r = y3;

    let b_f = F64x2::from_f64(w.rate_screened[i]) * d_f;
    let b_r = F64x2::from_f64(w.rate_screened[r_i]) * d_r;
    let b = b_f - b_r;
    dydt[i1][0] -= b;
    dydt[i2][0] -= b;
    dydt[i3][0] += b;

    w.raw_rate[i] = d_f * w.rate_raw[i] * AVO;
    w.raw_rate[r_i] = d_r * w.rate_raw[r_i] * AVO;
    w.screened_rate[i] = d_f * w.rate_screened[i] * AVO;
    w.screened_rate[r_i] = d_r * w.rate_screened[r_i] * AVO;

    if just_dydt {
        return;
    }

    let q = rxn.std_q;
    let icat_f = (rxn.category as usize).saturating_sub(1);
    let icat_r = (g.reactions[r_i].category as usize).saturating_sub(1);
    eps_nuc_mev[0] += b * q;
    w.eps_nuc_categories[icat_f] += (b_f * q).to_f64();
    w.eps_nuc_categories[icat_r] -= (b_r * q).to_f64();
    w.eps_nuc_rate[i] = (b_f * q).to_f64() * QCONV;
    w.eps_nuc_rate[r_i] = (-b_r * q).to_f64() * QCONV;

    for (rv_idx, (rf, rr)) in [
        (1, (w.rate_screened_dt[i], w.rate_screened_dt[r_i])),
        (2, (w.rate_screened_drho[i], w.rate_screened_drho[r_i])),
    ] {
        let bf_d = F64x2::from_f64(rf) * d_f;
        let br_d = F64x2::from_f64(rr) * d_r;
        let b_d = bf_d - br_d;
        dydt[i1][rv_idx] -= b_d;
        dydt[i2][rv_idx] -= b_d;
        dydt[i3][rv_idx] += b_d;
        eps_nuc_mev[rv_idx] += b_d * q;
    }

    let e_f = w.rate_screened[i];
    let e_r = w.rate_screened[r_i];
    let d1 = y2 * e_f;
    let d2 = y1 * e_f;
    let d3 = e_r;

    w.d_eps_nuc_dy[i1] += q * d1;
    w.d_eps_nuc_dy[i2] += q * d2;
    w.d_eps_nuc_dy[i3] -= q * d3;

    for (col, dval) in [(i1, d1), (i2, d2), (i3, -d3)] {
        w.d_dydt_dy[i1 * ni + col] -= dval;
        w.d_dydt_dy[i2 * ni + col] -= dval;
        w.d_dydt_dy[i3 * ni + col] += dval;
    }
}
