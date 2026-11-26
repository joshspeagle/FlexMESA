// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact MESA `26.4.1` hexadecimal parity for the convective-boundary radius
//! and pressure scale height used by exponential overshoot (`#550`).
//!
//! The inputs and expected values are IEEE-754 bit patterns written by an
//! instrumented MESA `26.4.1` build (MESA SDK `x86_64-linux-26.6.1`,
//! `gfortran 15.2.0`) from `overshoot_exp.f90`, immediately after
//! `eval_over_bdy_params`, in the upstream test cases named on each vector.
//! Before `#550`, `two_piece_linear_coeffs` computed the near-node weight
//! first and missed these `Hp_cb` values by 7 to 20 ULP.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]

use flexmesa_star::overshoot::{eval_conv_bdy_hp, eval_conv_bdy_r};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

/// One boundary record: `(case, top, cz_bdy_dq, conv_bdy_q, dq[3], lnPeos[3],
/// lnd[3], cgrav, M_center, xmstar, r_k, r_kp1, want r_cb, want Hp_cb)`,
/// where the three-element arrays are cells `k-1, k, k+1` around the
/// boundary cell `k`.
struct Bdy {
    case: &'static str,
    top: bool,
    cz_bdy_dq: u64,
    conv_bdy_q: u64,
    dq: [u64; 3],
    ln_peos: [u64; 3],
    lnd: [u64; 3],
    cgrav: u64,
    m_center: u64,
    xmstar: u64,
    r_k: u64,
    r_kp1: u64,
    want_r_cb: u64,
    want_hp_cb: u64,
}

const VECTORS: [Bdy; 3] = [
    // `star/5M_cepheid_blue_loop`, model 2112, boundary 2 (bottom), k = 467.
    Bdy {
        case: "5M_cepheid_blue_loop model 2112 bdy 2",
        top: false,
        cz_bdy_dq: 0x3F4E5DBF584F5D84,
        conv_bdy_q: 0x3FDA49D7F81A64CD,
        dq: [0x3F785B3F7F9ECD80, 0x3F787F86C53A5E80, 0x3F78A348EA18E600],
        ln_peos: [0x403753E0BAE85760, 0x403760D0C7DDD404, 0x40376E2EC9FA3E39],
        lnd: [0xC021E7D753E6FC93, 0xC021D7A69A375330, 0xC021C688E323C791],
        cgrav: 0x3E71EA8B45EDF98C,
        m_center: 0x0000000000000000,
        xmstar: 0x46FEA2E4E0A40A3D,
        r_k: 0x4270F201E6EE4522,
        r_kp1: 0x42708E61700ECEC8,
        want_r_cb: 0x4270E2DEC2CC4396,
        want_hp_cb: 0x425E6EB3A1B32B14,
    },
    // `star/high_z`, model 1, boundary 1 (top), k = 749.
    Bdy {
        case: "high_z model 1 bdy 1",
        top: true,
        cz_bdy_dq: 0x3F41DF208EA52C9C,
        conv_bdy_q: 0x3FD0878AEA0B2A03,
        dq: [0x3F5AD4AF36CC1FFC, 0x3F6AD4AF36CC1EFD, 0x3F6AE00C40DA78FD],
        ln_peos: [0x404238C8CB4BAEEA, 0x404239D83BD3A28E, 0x40423B424F116DEA],
        lnd: [0x3FF325DB81F33ABF, 0x3FF33ABFC757BAB5, 0x3FF356A0FC40024E],
        cgrav: 0x3E71EA8B45EDF98C,
        m_center: 0x0000000000000000,
        xmstar: 0x470572069D3FA0C4,
        r_k: 0x422B3A4360C8EBED,
        r_kp1: 0x422B1406540622F9,
        want_r_cb: 0x422B33ECBE997FB3,
        want_hp_cb: 0x421AF099CA0229C7,
    },
    // `star/high_z`, model 2, boundary 3 (top), k = 165.
    Bdy {
        case: "high_z model 2 bdy 3",
        top: true,
        cz_bdy_dq: 0x3E11F4391BDD032E,
        conv_bdy_q: 0x3FEFFFFFDD1E5B66,
        dq: [0x3E26DF0910F54AE0, 0x3E26DF0910F54AE0, 0x3E212746CCB7F800],
        ln_peos: [0x402E450C3B51FAE8, 0x402E5ADD1AB05D93, 0x402E6D3DA2D8B6D6],
        lnd: [0xC02ED868E0909AB8, 0xC02EC9DE2A190AA3, 0xC02EBDBA4729BAD3],
        cgrav: 0x3E71EA8B45EDF98C,
        m_center: 0x0000000000000000,
        xmstar: 0x470572063D08C39D,
        r_k: 0x4254F6A360DD07FC,
        r_kp1: 0x4254F4FEAE3C97A8,
        want_r_cb: 0x4254F5FE48D49031,
        want_hp_cb: 0x41E3A82D152A476B,
    },
];

/// Builds a five-cell model whose boundary cell is index 2 (0-based), with
/// cells 1, 2, 3 holding the recorded stencil.
fn model_for(v: &Bdy) -> StarModel {
    let nz = 5_usize;
    let mut ctrl = StarControls::default();
    // The recorded `Hp_cb` values are the unlimited scale heights; keep the
    // size-of-convection-zone cap out of this formula check.
    ctrl.mix_wind
        .limit_overshoot_hp_using_size_of_convection_zone = false;
    let mut m =
        StarModel::new(nz, vec![2, 7], StarJobControls::default(), ctrl);
    let f = f64::from_bits;
    let k = 2_usize;
    for (j, kk) in (k - 1..=k + 1).enumerate() {
        m.dq[kk] = f(v.dq[j]);
        m.ln_peos[kk] = f(v.ln_peos[j]);
        m.lnd[kk] = f(v.lnd[j]);
        m.cgrav[kk] = f(v.cgrav);
    }
    m.r = vec![
        2.0 * f(v.r_k),
        1.5 * f(v.r_k),
        f(v.r_k),
        f(v.r_kp1),
        0.5 * f(v.r_kp1),
    ];
    m.m_center = f(v.m_center);
    m.xmstar = f(v.xmstar);
    m.aux.mixing.cz_bdy_dq = vec![0.0; nz];
    m.aux.mixing.cz_bdy_dq[k] = f(v.cz_bdy_dq);
    m.aux.mixing.conv_bdy_q = vec![f(v.conv_bdy_q)];
    m.aux.mixing.top_conv_bdy = vec![v.top];
    // `eval_conv_bdy_k`: a top boundary sits in cell `loc`, a bottom one in
    // cell `loc - 1`.
    m.aux.mixing.conv_bdy_loc = vec![if v.top { k } else { k + 1 }];
    m
}

#[test]
fn eval_conv_bdy_r_and_hp_match_mesa_26_4_1_hex_at_recorded_boundaries() {
    for v in &VECTORS {
        let m = model_for(v);
        let r_cb = eval_conv_bdy_r(&m, 0).expect("r_cb");
        let hp_cb = eval_conv_bdy_hp(&m, 0).expect("hp_cb");
        assert_eq!(
            r_cb.to_bits(),
            v.want_r_cb,
            "{}: r_cb {r_cb:e} vs MESA {:e}",
            v.case,
            f64::from_bits(v.want_r_cb)
        );
        assert_eq!(
            hp_cb.to_bits(),
            v.want_hp_cb,
            "{}: Hp_cb {hp_cb:e} vs MESA {:e}",
            v.case,
            f64::from_bits(v.want_hp_cb)
        );
    }
}
