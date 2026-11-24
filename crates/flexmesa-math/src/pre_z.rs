// SPDX-License-Identifier: GPL-3.0-or-later
//! Precomputed integer powers and fractional powers `pre_z(1:1000)` porting
//! `math/public/math_def.f90` and `math/public/precompute_zs.inc`.

use std::sync::OnceLock;

use crate::{pow2, pow_strict};

/// Upper bound of `pre_z(1:1000)` (`math/public/math_def.f90:26`).
pub const MAX_PRECOMP_INTS: usize = 1000;

/// Precomputed integer and fractional powers for a positive integer $Z \in [1, 1000]$
/// (`math/public/math_def.f90:28-45`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecompInt {
    /// $Z^2$.
    pub z2: f64,
    /// $Z^3$.
    pub z3: f64,
    /// $Z^4$.
    pub z4: f64,
    /// $Z^5$.
    pub z5: f64,
    /// $Z^6$.
    pub z6: f64,
    /// $Z^7$.
    pub z7: f64,
    /// $Z^8$.
    pub z8: f64,
    /// $Z^{1/3}$.
    pub z1_3: f64,
    /// $Z^{2/3}$.
    pub z2_3: f64,
    /// $Z^{4/3}$.
    pub z4_3: f64,
    /// $Z^{5/3}$.
    pub z5_3: f64,
    /// $Z^{7/3}$.
    pub z7_3: f64,
    /// $Z^{-1/3}$.
    pub zm1_3: f64,
    /// $Z^{-2/3}$.
    pub zm2_3: f64,
    /// $Z^{-4/3}$.
    pub zm4_3: f64,
    /// $Z^{-5/3}$.
    pub zm5_3: f64,
    /// $Z^{-7/3}$.
    pub zm7_3: f64,
    /// $Z^{7/6}$.
    pub z7_6: f64,
    /// $Z^{1/5}$.
    pub z1_5: f64,
    /// $Z^{2/5}$.
    pub z2_5: f64,
    /// $Z^{3/5}$.
    pub z3_5: f64,
    /// $Z^{4/5}$.
    pub z4_5: f64,
    /// $Z^{1/2}$.
    pub z1_2: f64,
    /// $Z^{3/2}$.
    pub z3_2: f64,
    /// $Z^{-1/2}$.
    pub zm1_2: f64,
    /// $Z^{-1/4}$.
    pub zm1_4: f64,
    /// $Z^{-3/2}$.
    pub zm3_2: f64,
    /// $\ln(Z)$.
    pub logz: f64,
    /// $(\ln Z)^2$.
    pub sqlogz: f64,
    /// $(\ln Z)^{3/2}$.
    pub logz_3_2: f64,
    /// $Z^{0.475}$.
    pub z0p475: f64,
    /// $(1 + Z)^{3/2}$.
    pub zp1_3_2: f64,
    /// $Z^{-0.267}$.
    pub zm0p267: f64,
}

static PRE_Z_TABLE: OnceLock<Box<[PrecompInt; MAX_PRECOMP_INTS]>> =
    OnceLock::new();

fn build_pre_z_table() -> Box<[PrecompInt; MAX_PRECOMP_INTS]> {
    let mut vec = Vec::with_capacity(MAX_PRECOMP_INTS);
    for int_i in 1..=MAX_PRECOMP_INTS {
        #[allow(clippy::cast_precision_loss)]
        let z = int_i as f64;
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z3 * z;
        let z5 = z4 * z;
        let z6 = z5 * z;
        let z7 = z6 * z;
        let z8 = z7 * z;
        let logz = flexmesa_crlibm_sys::log_rz(z);
        let sqlogz = pow2(logz);
        let logz_3_2 = pow_strict(logz, 3.0 / 2.0);

        vec.push(PrecompInt {
            z2,
            z3,
            z4,
            z5,
            z6,
            z7,
            z8,
            z1_3: pow_strict(z, 1.0 / 3.0),
            z2_3: pow_strict(z, 2.0 / 3.0),
            z4_3: pow_strict(z, 4.0 / 3.0),
            z5_3: pow_strict(z, 5.0 / 3.0),
            z7_3: pow_strict(z, 7.0 / 3.0),
            zm1_3: pow_strict(z, -1.0 / 3.0),
            zm2_3: pow_strict(z, -2.0 / 3.0),
            zm4_3: pow_strict(z, -4.0 / 3.0),
            zm5_3: pow_strict(z, -5.0 / 3.0),
            zm7_3: pow_strict(z, -7.0 / 3.0),
            z7_6: pow_strict(z, 7.0 / 6.0),
            z1_5: pow_strict(z, 1.0 / 5.0),
            z2_5: pow_strict(z, 2.0 / 5.0),
            z3_5: pow_strict(z, 3.0 / 5.0),
            z4_5: pow_strict(z, 4.0 / 5.0),
            z1_2: pow_strict(z, 0.5),
            z3_2: pow_strict(z, 1.5),
            zm1_2: pow_strict(z, -0.5),
            zm1_4: pow_strict(z, -0.25),
            zm3_2: pow_strict(z, -3.0 / 2.0),
            logz,
            sqlogz,
            logz_3_2,
            z0p475: pow_strict(z, 0.475),
            zp1_3_2: pow_strict(1.0 + z, 3.0 / 2.0),
            zm0p267: pow_strict(z, -0.267),
        });
    }
    vec.into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!())
}

/// Return the precomputed powers for 1-based integer $z \in [1, 1000]$
/// (`math/public/math_def.f90:47`, `math/public/precompute_zs.inc:1-52`).
///
/// # Panics
/// Panics if `z == 0` or `z > MAX_PRECOMP_INTS`.
#[must_use]
#[inline]
pub fn pre_z(z: usize) -> &'static PrecompInt {
    assert!(
        (1..=MAX_PRECOMP_INTS).contains(&z),
        "pre_z index {z} out of 1..={MAX_PRECOMP_INTS} bounds"
    );
    let table = PRE_Z_TABLE.get_or_init(build_pre_z_table);
    &table[z - 1]
}
