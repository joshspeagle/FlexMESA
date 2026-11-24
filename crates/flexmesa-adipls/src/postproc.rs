// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust ports of MESA 26.4.1 ADIPLS mode post-processing routines
//! (`#171`, Package A2b):
//!
//! - [`kiner`] / [`kiner_with_observer`]
//!   (`adipls/adipack.c/adipls/kiner.n.d.f:1-85`)
//! - [`varfrq`] / [`varfrq_with_observer`]
//!   (`adipls/adipack.c/adipls/varfrq.n.d.f:1-427`)
//! - [`setobs_st`] (`adipls/adipack.c/adipls/setobs_st.c.d.f:1-150`)
//! - [`setssm`] (`adipls/adipack.c/adipls/setssm.n.d.f:1-59`)
//! - [`sigout`] and [`DefaultSigoutHandler`]
//!   (`adipls/adipack.c/adipls/sigout.c.d.f:1-418`,
//!   `adipls/adipack.c/gensr/ofiles.c.f:205-232, 317-353`)

use std::borrow::Cow;

use flexmesa_const::PI4;
use flexmesa_math::intrinsic_pow;

use crate::error::AdiplsError;
use crate::gensr::{
    aramax, derive, derivk, lir, notwin, notwni, vinta, vintk, DerivkCache,
    LirState, VintkCache,
};
use crate::model::AmdlAa;
use crate::output::{AdiModeStore, Csummm, ObsSt};
use crate::parser::{
    AdiFileUnit, AdiplsDsnControls, AdiplsInput, AdiplsOutControls,
};
use crate::rhs::{sp_radial_eps, Cprcns, RhsDat};
use crate::solve::{
    AdiplsTapEvent, AdiplsTapObserver, NoopTapObserver, SigoutArgs,
    SigoutHandler,
};

/// Variational integration controls matching `common/varcon/`
/// (`adipls/adipack.c/adipls/adipls.c.d.f:62`,
/// `adipls/adipack.c/adipls/varfrq.n.d.f:50`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Varcon {
    /// Variational formulation (`1` p-mode, `2` g-mode, `3` nonradial for
    /// radial modes; `ivarf`).
    pub ivarf: i32,
    /// Diagnostic print frequency (`npvarf`).
    pub npvarf: i32,
    /// Active differentiation/integration order parameter (`kvarfc`).
    pub kvarfc: i32,
}

impl Default for Varcon {
    fn default() -> Self {
        Self {
            ivarf: 1,
            npvarf: 0,
            kvarfc: 2,
        }
    }
}

/// Persistent mesh/coefficient cache for [`varfrq`] (`save` variables and
/// `cderst`/`cintst` caches in `adipls/adipack.c/adipls/varfrq.n.d.f:56-84`).
#[derive(Debug, Clone)]
pub struct VarfrqCache {
    xpfrst: f64,
    xphalf: f64,
    xplast: f64,
    nnp: usize,
    kvarfp: i32,
    kvarfi: i32,
    derivk_cache: DerivkCache,
    vintk_cache: VintkCache,
}

impl Default for VarfrqCache {
    fn default() -> Self {
        Self {
            xpfrst: -1.0_f64,
            xphalf: -1.0_f64,
            xplast: -1.0_f64,
            nnp: 0,
            kvarfp: 0,
            kvarfi: 0,
            derivk_cache: DerivkCache::default(),
            vintk_cache: VintkCache::default(),
        }
    }
}

/// Result returned by [`kiner`] and [`kiner_with_observer`]
/// (`adipls/adipack.c/adipls/kiner.n.d.f:83`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinerOutput {
    /// Normalization squared displacement `ynorm`.
    pub ynorm: f64,
    /// Normalized dimensionless kinetic energy `ekin`.
    pub ekin: f64,
}

/// Compute dimensionless mode kinetic energy integral `ekin` and normalization
/// `ynorm`, matching `kiner` (`adipls/adipack.c/adipls/kiner.n.d.f:1-85`).
///
/// # Errors
/// Returns [`AdiplsError::InvalidModel`] when `nn == 0` or input slices are
/// too short, or propagates [`AdiplsError::UnsetAmdlRow`] /
/// [`AdiplsError::InvalidAmdlIndex`] from `aa`.
#[allow(clippy::too_many_arguments)]
pub fn kiner(
    x: &[f64],
    y: &[f64],
    aa: &AmdlAa,
    el: f64,
    iy: usize,
    nn: usize,
    iplneq: i32,
    iekinr: i32,
    eprufl: f64,
) -> Result<KinerOutput, AdiplsError> {
    let mut obs = NoopTapObserver;
    kiner_with_observer(x, y, aa, el, iy, nn, iplneq, iekinr, eprufl, &mut obs)
}

/// Compute mode kinetic energy integral and emit [`AdiplsTapEvent::Kiner`],
/// matching `kiner` (`adipls/adipack.c/adipls/kiner.n.d.f:1-85`).
///
/// # Errors
/// Returns [`AdiplsError::InvalidModel`] when `nn == 0` or input slices are
/// too short, or propagates [`AdiplsError`] from `aa`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn kiner_with_observer(
    x: &[f64],
    y: &[f64],
    aa: &AmdlAa,
    el: f64,
    iy: usize,
    nn: usize,
    iplneq: i32,
    iekinr: i32,
    eprufl: f64,
    observer: &mut (impl AdiplsTapObserver + ?Sized),
) -> Result<KinerOutput, AdiplsError> {
    if nn == 0
        || iy < 2
        || x.len() < nn
        || aa.nn() < nn
        || y.len() < (nn - 1) * iy + 2
    {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "kiner dimension mismatch: nn={nn}, x.len()={}, aa.nn()={}, y.len()={}",
                x.len(),
                aa.nn(),
                y.len()
            ),
        });
    }

    let mut wrk1 = vec![0.0_f64; nn];
    let mut wrk2 = vec![0.0_f64; nn];
    let mut y1 = 0.0_f64;
    let mut elli = 0.0_f64;

    if iplneq == 1 {
        elli = 1.0_f64 / (el * el);
        let a5 = aa.row(5)?;
        for n in 0..nn {
            y1 = y[n * iy];
            let mut y2 = y[n * iy + 1];
            if y1.abs() < eprufl {
                y1 = 0.0_f64;
            }
            if y2.abs() < eprufl {
                y2 = 0.0_f64;
            }
            wrk1[n] = (y1 * y1 + (elli * y2) * y2) * a5[n];
        }
    } else if el > sp_radial_eps() {
        elli = 1.0_f64 / (el * (el + 1.0_f64));
        let a1 = aa.row(1)?;
        let a5 = aa.row(5)?;
        let a10 = aa.row(10)?;
        for n in 0..nn {
            y1 = y[n * iy];
            let mut y2 = y[n * iy + 1];
            if y1.abs() < eprufl {
                y1 = 0.0_f64;
            }
            if y2.abs() < eprufl {
                y2 = 0.0_f64;
            }
            let xx = x[n];
            wrk1[n] = (((((y1 * y1 + (y2 * y2) * elli) * a1[n]) * a5[n])
                * a10[n])
                * xx)
                * xx;
        }
    } else {
        let a1 = aa.row(1)?;
        let a5 = aa.row(5)?;
        let a10 = aa.row(10)?;
        for n in 0..nn {
            let xx = x[n];
            y1 = y[n * iy];
            wrk1[n] = (((((y1 * y1) * a1[n]) * a5[n]) * a10[n]) * xx) * xx;
        }
    }

    vinta(x, &wrk1, &mut wrk2, nn, 1, 1);

    let ynorm = if iekinr == 1 {
        let mut yi = [0.0_f64; 2];
        let mut lir_state = LirState::default();
        let _ = lir(1.0_f64, x, &mut yi, y, 2, iy, nn, 1, &mut lir_state);
        if el <= sp_radial_eps() {
            yi[0] * yi[0]
        } else {
            yi[0] * yi[0] + (elli * yi[1]) * yi[1]
        }
    } else {
        if y1 == 0.0_f64 {
            y1 = 1.0_f64;
        }
        y1 * y1
    };

    let ekin = wrk2[nn - 1] / (PI4 * ynorm);
    observer.on_tap(AdiplsTapEvent::Kiner { el, ynorm, ekin });
    Ok(KinerOutput { ynorm, ekin })
}

/// Compute variational squared dimensionless frequency `sigv`, matching
/// `varfrq` (`adipls/adipack.c/adipls/varfrq.n.d.f:1-427`).
///
/// Routes non-integer powers through [`flexmesa_math::intrinsic_pow`] per
/// `DECISIONS #60` and preserves all single-precision literal thresholds in
/// `adipls_sp_literals_reached.tsv`.
///
/// # Errors
/// Returns [`AdiplsError::InvalidModel`] when `nn < 2` or array lengths are
/// inconsistent, [`AdiplsError::UnsupportedControl`] if `ivarf` is not in
/// `1..=3` in the nonradial branch, or propagates [`AdiplsError`] from `aa`.
#[allow(clippy::too_many_arguments)]
pub fn varfrq(
    x: &[f64],
    y: &[f64],
    data: &[f64; 8],
    aa: &AmdlAa,
    sig: f64,
    iy: usize,
    nn: usize,
    istsbc: i32,
    rhs: &RhsDat,
    cprcns: &Cprcns,
    varcon: &mut Varcon,
    cache: &mut VarfrqCache,
) -> Result<f64, AdiplsError> {
    let mut obs = NoopTapObserver;
    varfrq_with_observer(
        x, y, data, aa, sig, iy, nn, istsbc, rhs, cprcns, varcon, cache,
        &mut obs,
    )
}

/// Compute variational squared dimensionless frequency `sigv` and emit
/// [`AdiplsTapEvent::Varfrq`], matching `varfrq`
/// (`adipls/adipack.c/adipls/varfrq.n.d.f:1-427`).
///
/// # Errors
/// Returns [`AdiplsError::InvalidModel`] when `nn < 2` or array lengths are
/// inconsistent, [`AdiplsError::UnsupportedControl`] if `ivarf` is not in
/// `1..=3` in the nonradial branch, or propagates [`AdiplsError`] from `aa`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp
)]
pub fn varfrq_with_observer(
    x: &[f64],
    y: &[f64],
    data: &[f64; 8],
    aa: &AmdlAa,
    sig: f64,
    iy: usize,
    nn: usize,
    istsbc: i32,
    rhs: &RhsDat,
    cprcns: &Cprcns,
    varcon: &mut Varcon,
    cache: &mut VarfrqCache,
    observer: &mut (impl AdiplsTapObserver + ?Sized),
) -> Result<f64, AdiplsError> {
    if nn < 2
        || iy < 2
        || x.len() < nn
        || aa.nn() < nn
        || y.len() < (nn - 1) * iy + 2
    {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "varfrq dimension mismatch: nn={nn}, x.len()={}, aa.nn()={}, y.len()={}",
                x.len(),
                aa.nn(),
                y.len()
            ),
        });
    }

    let el = rhs.el;
    let ell = rhs.ell;
    let els = rhs.els;

    if (el >= sp_radial_eps() || varcon.ivarf == 2)
        && (iy < 3 || y.len() < (nn - 1) * iy + 3)
    {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "varfrq nonradial/ivarf=2 requires iy >= 3 and y.len() >= {}, got iy={iy}, y.len()={}",
                (nn - 1) * iy + 3,
                y.len()
            ),
        });
    }

    if varcon.kvarfc == 1 || varcon.kvarfc > 3 {
        varcon.kvarfc = 2;
    }

    let x_half = x[nn / 2 - 1];
    if cache.xpfrst != x[0]
        || cache.xphalf != x_half
        || cache.xplast != x[nn - 1]
        || cache.nnp != nn
        || cache.kvarfp != varcon.kvarfc
    {
        cache.kvarfi = varcon.kvarfc;
        cache.kvarfp = varcon.kvarfc;
        cache.xpfrst = x[0];
        cache.xphalf = x_half;
        cache.xplast = x[nn - 1];
        cache.nnp = nn;
    }

    let mut wrk: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0_f64; nn]);

    let kvarfi_isize = isize::try_from(cache.kvarfi).unwrap_or(0);
    let _ = derivk(
        x,
        y,
        &mut wrk[0],
        1,
        nn,
        iy,
        1,
        kvarfi_isize,
        &mut cache.derivk_cache,
    );

    if el > sp_radial_eps() && varcon.ivarf == 3 {
        varcon.ivarf = 1;
    }

    let a1 = aa.row(1)?;
    let a2 = aa.row(2)?;
    let a3 = aa.row(3)?;
    let a4 = aa.row(4)?;
    let a5 = aa.row(5)?;
    let a10 = aa.row(10)?;

    let mut sigv;

    if el <= sp_radial_eps() && varcon.ivarf != 3 {
        // Radial variational integral (`varfrq.n.d.f:281-366`).
        let usegam = a3[0] > f64::from(0.1_f32);
        if usegam {
            derive(x, a3, &mut wrk[2], nn, 1, 1, 1, 1);
        }
        let sp_1e10 = f64::from(1.0e-10_f32);
        let mut vg = 0.0_f64;
        for n in 1..nn {
            let xx = x[n];
            let qtx = (a1[n] * xx) * xx;
            let mut z1 = y[n * iy];
            let mut qxu = (a10[n] * qtx) * a5[n];
            wrk[1][n] = (z1 * z1) * qxu;
            qxu *= qtx;
            vg = a2[n];
            if vg == 0.0_f64 {
                vg = cprcns.epsofl;
            }
            if usegam {
                z1 /= xx;
                let d1 = wrk[0][n] - z1;
                let gm1 = a3[n];
                let mut dgm1 = wrk[2][n];
                if dgm1.abs() < sp_1e10 {
                    dgm1 = 0.0_f64;
                }
                wrk[2][n] = ((d1 * d1) / vg
                    + ((3.0_f64 * gm1 - 4.0_f64) * z1) * z1)
                    * qxu;
                wrk[3][n] =
                    (((((-3.0_f64 * qxu) * xx) * dgm1) * z1) * z1) / (vg * gm1);
                wrk[7][n] = wrk[0][n];
            } else {
                z1 = (2.0_f64 * z1) / xx;
                let d1 = wrk[0][n] + z1;
                wrk[2][n] = ((d1 * d1) * qxu) / vg;
                wrk[3][n] = (z1 * z1) * qxu;
            }
        }
        for slot in wrk.iter_mut().take(7).skip(1) {
            slot[0] = 0.0_f64;
        }

        let (in_rows, out_rows) = wrk.split_at_mut(4);
        let (out45, rest) = out_rows.split_at_mut(2);
        let (out4, out5) = out45.split_at_mut(1);
        vintk_rows_3(
            x,
            [&in_rows[1], &in_rows[2], &in_rows[3]],
            &mut [&mut out4[0], &mut out5[0], &mut rest[0]],
            nn,
            kvarfi_isize,
            &mut cache.vintk_cache,
        );
        cache.kvarfi = 0;

        let nn0 = nn - 1;
        if usegam {
            let z1 = y[nn0 * iy];
            let ajs = (z1
                * ((3.0_f64 / vg - 1.0_f64) * z1 + sig * y[nn0 * iy + 1]))
                * a5[nn0];
            let dj = (ajs + wrk[5][nn0]) + wrk[6][nn0];
            sigv = dj / wrk[4][nn0];
        } else {
            let mut dj = wrk[5][nn0] - wrk[6][nn0];
            if istsbc != 0 {
                let z1 = y[nn0 * iy];
                let ajs = (z1 * (sig * y[nn0 * iy + 1] - z1)) * a5[nn0];
                dj += ajs;
            }
            sigv = dj / wrk[4][nn0];
        }
    } else {
        // Nonradial variational integral (`varfrq.n.d.f:101-255`).
        let xlim = intrinsic_pow(cprcns.epsufl, 1.0_f64 / (1.0_f64 + el));

        for n in 1..nn {
            let xx = x[n];
            let y1n = y[n * iy];
            if (el > 1.0_f64 && xx < xlim) || y1n.abs() <= cprcns.eprufl {
                for row in wrk.iter_mut().take(7).skip(2) {
                    row[n] = 0.0_f64;
                }
            } else {
                let y2n = y[n * iy + 1];
                let (d1, z2) = match varcon.ivarf {
                    1 => (wrk[0][n] + (2.0_f64 * y1n - y2n) / xx, y2n),
                    2 => {
                        let d1 = (a2[n]
                            * ((y1n - y2n / (els * a1[n])) - y[n * iy + 2]))
                            / xx;
                        let z2 = 2.0_f64 * y1n + xx * (wrk[0][n] - d1);
                        (d1, z2)
                    }
                    3 => (wrk[0][n] + (2.0_f64 * y1n) / xx, y2n),
                    other => {
                        return Err(AdiplsError::UnsupportedControl {
                            control: "ivarf",
                            message: format!(
                                "varfrq ivarf={other} not allowed"
                            ),
                        });
                    }
                };
                let d2 = d1 - ((a2[n] + a4[n]) * y1n) / xx;
                let qxu = ((a1[n] * a5[n]) * xx) * xx;
                wrk[2][n] = (d2 * intrinsic_pow(xx, el)) * qxu;
                wrk[3][n] = d1;
                wrk[4][n] = d2;
                wrk[5][n] = qxu;
                wrk[6][n] = z2;
            }
        }

        for row in wrk.iter_mut().take(6).skip(1) {
            row[0] = 0.0_f64;
        }
        wrk[6][0] = (el + 1.0_f64) * y[0];

        let an = data[6];
        if an > 0.0_f64 && an <= 1.0_f64 {
            wrk[2][nn - 1] = wrk[2][nn - 2];
        }

        let (wrk2_in, wrk1_out) = {
            let (left, right) = wrk.split_at_mut(2);
            (&right[0], &mut left[1])
        };
        let _ = vintk(
            x,
            wrk2_in,
            wrk1_out,
            1,
            nn,
            1,
            1,
            kvarfi_isize,
            &mut cache.vintk_cache,
        );

        if an > 0.0_f64 {
            wrk[1][nn - 1] =
                wrk[1][nn - 2] + ((1.0_f64 - x[nn - 2]) * wrk[2][nn - 2]) / an;
        }

        cache.kvarfi = 0;

        let elli = if el > sp_radial_eps() {
            1.0_f64 / ell
        } else {
            0.0_f64
        };
        wrk[7][0] = 0.0_f64;
        let mut xl1 = 0.0_f64;
        let mut z1 = 0.0_f64;
        let mut z2 = 0.0_f64;

        for n in 0..nn {
            z1 = y[n * iy];
            z2 = wrk[6][n];
            let xx = x[n];

            if el > 1.0_f64 && (xx < xlim || z1.abs() < cprcns.eprufl) {
                wrk[5][n] = 0.0_f64;
                wrk[6][n] = 0.0_f64;
                wrk[7][n] = 0.0_f64;
            } else {
                let xd1 = wrk[3][n] * xx;
                let qxu = wrk[5][n];
                let mut vg = a2[n];
                if vg == 0.0_f64 {
                    vg = cprcns.epsofl;
                }
                wrk[5][n] = qxu * (z1 * z1 + (z2 * z2) * elli);
                if n > 0 {
                    xl1 = intrinsic_pow(xx, -1.0_f64 - el);
                }
                wrk[6][n] = ((wrk[1][n] * wrk[4][n]) * xl1) * qxu;
                if n > 0 {
                    wrk[7][n] = ((((xd1 * xd1) / vg - (2.0_f64 * xd1) * z1)
                        + ((a2[n] + a4[n]) * z1) * z1)
                        * qxu)
                        * a1[n];
                }
            }
        }

        if an > 0.0_f64 && an <= 1.0_f64 {
            wrk[7][nn - 1] = wrk[7][nn - 2];
        }

        let (left, right) = wrk.split_at_mut(5);
        let (out2_3, out4) = left.split_at_mut(4);
        let (out2_slice, out3_slice) = out2_3.split_at_mut(3);
        vintk_rows_3(
            x,
            [&right[0], &right[1], &right[2]],
            &mut [&mut out2_slice[2], &mut out3_slice[0], &mut out4[0]],
            nn,
            0,
            &mut cache.vintk_cache,
        );

        if an > 0.0_f64 {
            wrk[4][nn - 1] =
                wrk[4][nn - 2] + ((1.0_f64 - x[nn - 2]) * wrk[7][nn - 2]) / an;
        }

        let nn0 = nn - 1;
        let el21 = 2.0_f64 * el + 1.0_f64;
        let akv = wrk[4][nn0] - (2.0_f64 * wrk[3][nn0]) / el21;
        let uz = a5[nn0] * z1;
        let pprs = if el < sp_radial_eps() {
            sig * z2
        } else {
            z2 / els + y[nn0 * iy + 2]
        };
        let aks = uz * (pprs - (uz - 2.0_f64 * wrk[1][nn0]) / el21);
        sigv = (akv + aks) / wrk[2][nn0];
    }

    sigv = sigv.abs();
    observer.on_tap(AdiplsTapEvent::Varfrq { sig, sigv });
    Ok(sigv)
}

fn vintk_rows_3(
    x: &[f64],
    src: [&[f64]; 3],
    dst: &mut [&mut [f64]; 3],
    nn: usize,
    karg: isize,
    cache: &mut VintkCache,
) {
    let mut a_buf = vec![0.0_f64; 3 * nn];
    let mut y_buf = vec![0.0_f64; 3 * nn];
    for n in 0..nn {
        let base = n * 3;
        a_buf[base] = src[0][n];
        a_buf[base + 1] = src[1][n];
        a_buf[base + 2] = src[2][n];
    }
    y_buf[0] = dst[0][0];
    y_buf[1] = dst[1][0];
    y_buf[2] = dst[2][0];
    let _ = vintk(x, &a_buf, &mut y_buf, 3, nn, 3, 3, karg, cache);
    for n in 0..nn {
        let base = n * 3;
        dst[0][n] = y_buf[base];
        dst[1][n] = y_buf[base + 1];
        dst[2][n] = y_buf[base + 2];
    }
}

/// Rotational splitting quantities matching `common/crot_split/`
/// (`adipls/adipack.c/adipls/setobs_st.c.d.f:54`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrotSplit {
    /// Ledoux factor $\beta$.
    pub beta: f64,
    /// Splitting coefficients `split(1:4)`.
    pub split: [f64; 4],
}

/// Accumulation state for `common/cobs_param/` across modes of one ADIPLS
/// call (`adipls/adipack.c/adipls/setobs_st.c.d.f:64-105`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ObsStState {
    /// Mode observation flag `icobs_st = icobs_st0 + 10 * icobs_st1`.
    pub icobs_st: i32,
    /// Frequency source selector `icobs_st0` (`1`, `4`, `5`, `6`, ...).
    pub icobs_st0: i32,
    /// Rotational splitting selector `icobs_st1` (`0`, `1`, `11`).
    pub icobs_st1: i32,
    /// Accumulated `obs_st(1:10, 1:nobs_st)` records.
    pub records: Vec<ObsSt>,
}

impl ObsStState {
    /// Clear accumulated observations and flag (`nobs_st = 0`).
    pub fn clear(&mut self) {
        self.icobs_st = 0;
        self.icobs_st0 = 0;
        self.icobs_st1 = 0;
        self.records.clear();
    }
}

/// Append one [`ObsSt`] record from grand summary `csummm(1:50)` into
/// `state`, matching `setobs_st` (`adipls/adipack.c/adipls/setobs_st.c.d.f:1-150`).
///
/// # Errors
/// Returns [`AdiplsError::InvalidCsummmIndex`] on summary access failure or
/// [`AdiplsError::Solver`] if `state.records.len() >= nobs_stmx`.
#[allow(clippy::too_many_arguments)]
pub fn setobs_st(
    csummm: &Csummm,
    iriche: i32,
    iper: i32,
    ispcpr: i32,
    irotkr: i32,
    nobs_stmx: usize,
    perfac: f64,
    crot_split: &CrotSplit,
    state: &mut ObsStState,
) -> Result<ObsSt, AdiplsError> {
    if state.records.is_empty() {
        let ic0 = if ispcpr == 0 || ispcpr == 5 {
            if iriche == 1 {
                5
            } else if iper == 1 {
                1
            } else {
                6
            }
        } else if ispcpr == 1 {
            if iper == 1 {
                1
            } else {
                6
            }
        } else {
            ispcpr
        };

        let ic1 = if irotkr == 11 {
            1
        } else if irotkr == 21 {
            11
        } else {
            0
        };
        state.icobs_st0 = ic0;
        state.icobs_st1 = ic1;
        state.icobs_st = ic0 + 10 * ic1;
    }

    if state.records.len() >= nobs_stmx {
        return Err(AdiplsError::Solver {
            routine: "setobs_st",
            message: format!(
                "number of modes {} reaches maximum nobs_stmx={nobs_stmx}",
                state.records.len()
            ),
        });
    }

    let mut obs = [0.0_f64; 10];
    obs[0] = csummm.get_f64(18)?;
    obs[1] = csummm.get_f64(19)?;
    obs[3] = csummm.get_f64(24)?;
    obs[2] = if state.icobs_st0 == 5 {
        1000.0_f64 * csummm.get_f64(37)?
    } else if state.icobs_st0 == 1 {
        1000.0_f64 * csummm.get_f64(27)?
    } else if state.icobs_st0 == 6 {
        csummm.get_f64(21)?.sqrt() / (60.0e-6_f64 * perfac)
    } else {
        csummm.get_f64(20)?.sqrt() / (60.0e-6_f64 * perfac)
    };

    if irotkr == 11 {
        obs[4] = crot_split.beta;
        obs[5] = crot_split.split[0];
    }
    if irotkr == 21 {
        obs[4] = crot_split.beta;
        obs[5] = crot_split.split[0];
        obs[6] = crot_split.split[1];
        obs[7] = crot_split.split[2];
    }

    let rec = ObsSt::from_array(obs);
    state.records.push(rec);
    Ok(rec)
}

/// Seven-word short summary record `ssummm(1:7)` matching `setssm`
/// (`adipls/adipack.c/adipls/setssm.n.d.f:1-59`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ssummm {
    /// Real short-summary fields `ssummm(1:5)`: `[el, iord, sig, ekin, frq_mHz]`.
    pub ss: [f64; 5],
    /// Equivalenced integer fields `issumm(1:2) = [icase, iorign]`.
    pub iss: [i32; 2],
}

/// Short-summary model header record `ssmod(1:6)` emitted when `irmod == 1`
/// (`adipls/adipack.c/adipls/setssm.n.d.f:50-56`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsmModelRecord {
    /// Sentinel `-1.0` (`ssmod(1)`).
    pub header_minus_one: f64,
    /// Model index `xmod` (`ssmod(2) = cs(1)`).
    pub xmod: f64,
    /// First four AMDL header scalars `data(1:4)` (`ssmod(3:6) = cs(2:5)`).
    pub data_1_to_4: [f64; 4],
}

/// Combined output of one [`setssm`] call
/// (`adipls/adipack.c/adipls/setssm.n.d.f:1-59`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetssmOutput {
    /// Short summary record `ssummm`.
    pub ssummm: Ssummm,
    /// Model change flag `irmod`.
    pub irmod: i32,
    /// Model header record when `irmod == 1`.
    pub model_record: Option<SsmModelRecord>,
}

/// Persistent model-change tracker for [`setssm`]
/// (`csmodp(2:5)` in `adipls/adipack.c/adipls/setssm.n.d.f:21-24`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetssmState {
    csmodp: [f64; 4],
}

impl Default for SetssmState {
    fn default() -> Self {
        Self {
            csmodp: [-1.0_f64; 4],
        }
    }
}

/// Convert grand summary `csummm(1:50)` into short summary `ssummm` and emit
/// an optional model record when model header scalars `cs(2:5)` change,
/// matching `setssm` (`adipls/adipack.c/adipls/setssm.n.d.f:1-59`).
///
/// # Errors
/// Returns [`AdiplsError::InvalidCsummmIndex`] on summary access failure.
pub fn setssm(
    csummm: &Csummm,
    state: &mut SetssmState,
) -> Result<SetssmOutput, AdiplsError> {
    let mut ss = [0.0_f64; 5];
    ss[0] = csummm.get_f64(18)?;
    ss[1] = csummm.get_f64(19)?;
    ss[2] = csummm.get_f64(21)?;
    if ss[2] <= 0.0_f64 {
        ss[2] = csummm.get_f64(20)?;
    }
    ss[3] = csummm.get_f64(24)?;
    ss[4] = csummm.get_f64(27)?;
    if ss[4] <= 0.0_f64 {
        ss[4] = 16.666_666_667_f64 / csummm.get_f64(25)?;
    }
    let ics = csummm.icsumm();
    let iss = [ics[4], ics[5]];

    let sp_1e5 = f64::from(1.0e-5_f32);
    let mut irmod = 0_i32;
    for (i, &cs_prev) in state.csmodp.iter().enumerate() {
        let csi = csummm.get_f64(i + 2)?;
        if csi != 0.0_f64 && (cs_prev / csi - 1.0_f64).abs() > sp_1e5 {
            irmod = 1;
        }
    }

    let model_record = if irmod == 1 {
        let mut data_1_to_4 = [0.0_f64; 4];
        for i in 0..4 {
            let csi = csummm.get_f64(i + 2)?;
            state.csmodp[i] = csi;
            data_1_to_4[i] = csi;
        }
        Some(SsmModelRecord {
            header_minus_one: -1.0_f64,
            xmod: csummm.get_f64(1)?,
            data_1_to_4,
        })
    } else {
        None
    };

    Ok(SetssmOutput {
        ssummm: Ssummm { ss, iss },
        irmod,
        model_record,
    })
}

/// Mode post-processing state carried across modes within one
/// [`AdiplsSolver`](crate::driver::AdiplsSolver) call (`sigout` + `varfrq` +
/// `setssm` + `setobs_st` + `spcout_adi`).
#[derive(Debug, Clone)]
pub struct SigoutState {
    /// Output controls (`common/coutpt/` + `common/cmdtst/`).
    pub out: AdiplsOutControls,
    /// Dataset unit numbers (`common/cdadsg/`).
    pub dsn: AdiplsDsnControls,
    /// File units configured in `adipls.c.pruned.in` (`common/cofile/`).
    pub files: Vec<AdiFileUnit>,
    /// Maximum stored modes (`nobs_stmx = 20000`).
    pub nobs_stmx: usize,
    /// Scan eigenfunction output flag (`nfmscn`).
    pub nfmscn: i32,
    /// NRKM verification flag (`itssol`).
    pub itssol: i32,
    /// Variational frequency controls (`common/varcon/`).
    pub varcon: Varcon,
    /// Cached mesh/coefficient state for [`varfrq`].
    pub varfrq_cache: VarfrqCache,
    /// Rotational splitting buffer (`common/crot_split/`).
    pub crot_split: CrotSplit,
    /// Mode observation buffer (`common/cobs_param/`).
    pub obs_state: ObsStState,
    /// Short-summary model-change state (`setssm`).
    pub ssm_state: SetssmState,
    /// Short-summary records emitted via [`setssm`].
    pub ssm_records: Vec<SetssmOutput>,
    /// Final mode records populated via [`spcout_adi`](crate::output::spcout_adi).
    pub mode_store: AdiModeStore,
}

impl SigoutState {
    /// Construct initial post-processing state from parsed [`AdiplsInput`] and
    /// effective per-call kernel flags (`coutpt`).
    #[must_use]
    pub fn new(
        input: &AdiplsInput,
        irotkr: i32,
        nprtkr: i32,
        igm1kr: i32,
        npgmkr: i32,
        nfmscn: i32,
        nobs_stmx: usize,
    ) -> Self {
        let mut out = input.out;
        out.nfmode %= 10;
        out.irotkr = irotkr;
        out.nprtkr = nprtkr;
        out.igm1kr = igm1kr;
        out.npgmkr = npgmkr;

        Self {
            out,
            dsn: input.dsn,
            files: input.files.clone(),
            nobs_stmx,
            nfmscn,
            itssol: input.dgn.itssol,
            varcon: Varcon {
                ivarf: out.ivarf,
                npvarf: out.npvarf,
                kvarfc: out.kvarf,
            },
            varfrq_cache: VarfrqCache::default(),
            crot_split: CrotSplit::default(),
            obs_state: ObsStState::default(),
            ssm_state: SetssmState::default(),
            ssm_records: Vec::new(),
            mode_store: AdiModeStore::new(),
        }
    }

    /// Replicate ADIPLS `nscfil(unit)` (`adipls/adipack.c/gensr/ofiles.c.f:205-232,
    /// 317-353`): returns `true` if `unit > 0`, mapped in `files`, and not
    /// opened as a scratch file (2nd+ `/dev/null` mapping).
    #[must_use]
    pub fn nscfil(&self, unit: i32) -> bool {
        if unit <= 0 {
            return false;
        }
        let mut null_count = 0_usize;
        for entry in &self.files {
            if entry.path == "/dev/null" {
                null_count += 1;
            }
            if entry.unit == unit {
                return entry.path != "/dev/null" || null_count <= 1;
            }
        }
        false
    }
}

/// Default Package A2b [`SigoutHandler`] wrapping [`sigout`] and
/// [`SigoutState`].
#[derive(Debug, Clone)]
pub struct DefaultSigoutHandler {
    /// Accumulated post-processing outputs and state.
    pub state: SigoutState,
}

impl DefaultSigoutHandler {
    /// Create a handler from parsed [`AdiplsInput`] and effective per-call
    /// `coutpt` kernel controls.
    #[must_use]
    pub fn new(
        input: &AdiplsInput,
        irotkr: i32,
        nprtkr: i32,
        igm1kr: i32,
        npgmkr: i32,
        nfmscn: i32,
        nobs_stmx: usize,
    ) -> Self {
        Self {
            state: SigoutState::new(
                input, irotkr, nprtkr, igm1kr, npgmkr, nfmscn, nobs_stmx,
            ),
        }
    }
}

impl SigoutHandler for DefaultSigoutHandler {
    fn on_sigout(
        &mut self,
        args: &mut SigoutArgs<'_>,
    ) -> Result<(), AdiplsError> {
        sigout(args, &mut self.state)
    }
}

/// Post-process one solved (or scanned) adiabatic oscillation mode:
/// normalize eigenfunction, compute energy-normalized peak, compute kinetic
/// energy via [`kiner_with_observer`], compute variational frequency via
/// [`varfrq_with_observer`] when `iper == 1`, apply mode window filters,
/// update `csummm`, emit short summary via [`setssm`], store observation via
/// [`setobs_st`], and record [`ModeRecord`](crate::output::ModeRecord) via
/// [`spcout_adi`](crate::output::spcout_adi), matching `sigout`
/// (`adipls/adipack.c/adipls/sigout.c.d.f:1-418`).
///
/// # Errors
/// Returns:
/// - [`AdiplsError::UnsupportedControl`] if unported paths `itssol == 1`,
///   `irotkr > 0`, `igm1kr == 1`, or disk eigenfunction write (`nfmode`/`nfmscn`)
///   are requested.
/// - [`AdiplsError`] from [`kiner_with_observer`], [`varfrq_with_observer`],
///   [`setssm`], [`setobs_st`], or `spcout_adi`.
#[allow(clippy::too_many_lines, clippy::float_cmp, clippy::similar_names)]
pub fn sigout(
    args: &mut SigoutArgs<'_>,
    state: &mut SigoutState,
) -> Result<(), AdiplsError> {
    let radial = args.rhs.el <= sp_radial_eps();
    let per = args.rhs.perfac / args.sig.abs().max(args.cprcns.epsufl).sqrt();
    args.csummm.set_f64(25, per)?;

    let nn_col = (args.nn - 1) * args.iy;
    let mut fcy = args.y[nn_col];
    if !radial
        && args.cincnt.icow == 0
        && fcy.abs() < f64::from(1.0e-5_f32) * args.y[nn_col + 2].abs()
    {
        fcy = args.y[nn_col + 2];
    }
    let fcy1 = fcy;
    fcy = 1.0_f64;
    if fcy1 != 0.0_f64 {
        fcy = f64::from(1.0_f32) / fcy1;
    }
    if (fcy - 1.0_f64).abs() >= f64::from(1.0e-8_f32) {
        for n in args.nw1..=args.nn {
            let base = (n - 1) * args.iy;
            for i in 0..args.ii {
                args.y[base + i] *= fcy;
            }
        }
    }

    if state.itssol == 1 {
        return Err(AdiplsError::UnsupportedControl {
            control: "itssol",
            message: String::from(
                "sigout with itssol=1 (nrtssl NRKM check) is unported in Package A2b",
            ),
        });
    }

    if args.cincnt.iplneq == 1 {
        let a5 = args.aa.row(5)?;
        for n in args.nw1..=args.nn {
            let n0 = n - 1;
            args.y[n0 * args.iy + 5] = a5[n0].sqrt();
        }
    } else {
        let a1 = args.aa.row(1)?;
        let a5 = args.aa.row(5)?;
        let a10 = args.aa.row(10)?;
        for n in args.nw1..=args.nn {
            let n0 = n - 1;
            let xn = args.x[n0];
            let efac = ((((xn * xn) * xn) * a1[n0]) * a5[n0]) * a10[n0];
            args.y[n0 * args.iy + 5] =
                if efac < 0.0_f64 { 0.0_f64 } else { efac.sqrt() };
        }
    }

    let iwn = if radial { 1_usize } else { 2_usize };
    let mut fct = 1.0_f64;
    for i in 1..=iwn {
        let kwn0 = 5 + i;
        if i == 2 {
            fct = 1.0_f64 / args.rhs.ell.sqrt();
        }
        for n in args.nw1..=args.nn {
            let base = (n - 1) * args.iy;
            args.y[base + kwn0] =
                (fct * args.y[base + (i - 1)]) * args.y[base + 5];
        }
    }

    let nw1_off = (args.nw1 - 1) * args.iy;
    let (xrmx, mut drmx, _d2drmx) = aramax(
        &args.x[args.nw1 - 1..],
        &args.y[nw1_off + 6..],
        args.iy,
        args.nnw,
    );
    args.csummm.set_f64(34, drmx)?;
    args.csummm.set_f64(35, xrmx)?;
    if drmx == 0.0_f64 {
        drmx = 1.0_f64;
    }
    let fct_norm = 1.0_f64 / drmx;
    for k0 in 6..(6 + iwn) {
        for n in args.nw1..=args.nn {
            let base = (n - 1) * args.iy;
            args.y[base + k0] *= fct_norm;
        }
    }

    let (xmx, dmx, _d2dmx) = aramax(
        &args.x[args.nw1 - 1..],
        &args.y[nw1_off..],
        args.iy,
        args.nnw,
    );
    args.csummm.set_f64(22, dmx)?;
    args.csummm.set_f64(23, xmx)?;

    let aa_nw1: Cow<'_, AmdlAa> = if args.nw1 == 1 {
        Cow::Borrowed(args.aa)
    } else {
        Cow::Owned(args.aa.slice_from(args.nw1 - 1)?)
    };
    let x_nw1 = &args.x[args.nw1 - 1..args.nw1 - 1 + args.nnw];
    let y_nw1 = &args.y[nw1_off..];

    let kiner_out = kiner_with_observer(
        x_nw1,
        y_nw1,
        aa_nw1.as_ref(),
        args.rhs.el,
        args.iy,
        args.nnw,
        args.cincnt.iplneq,
        args.cincnt.iekinr,
        args.cprcns.eprufl,
        args.observer,
    )?;
    args.csummm.set_f64(24, kiner_out.ekin)?;

    if state.out.iper == 1 {
        let sigv = varfrq_with_observer(
            x_nw1,
            y_nw1,
            &args.rhs.data,
            aa_nw1.as_ref(),
            args.sig,
            args.iy,
            args.nnw,
            args.istsb1,
            args.rhs,
            args.cprcns,
            &mut state.varcon,
            &mut state.varfrq_cache,
            args.observer,
        )?;
        let perv = args.rhs.perfac / sigv.abs().max(args.cprcns.epsufl).sqrt();
        let frqv = 16.666_666_666_667_f64 / perv;
        args.csummm.set_f64(26, perv)?;
        args.csummm.set_f64(27, frqv)?;
    }

    if args.isigcv <= 0 {
        return Ok(());
    }

    let mut icase = 10 * state.out.iper
        + 100 * args.rot.irotsl
        + 10_000 * args.istsb1
        + 100_000 * args.cincnt.iplneq
        + 1_000_000 * args.cincnt.iturpr;
    if args.cincnt.icow == 2 {
        icase += 1;
    } else if args.cincnt.icow == 3 {
        icase += 2;
    }
    if args.rhs.alb != 1.0_f64 || args.bcs.fctsbc != 0.0_f64 {
        icase += 1000;
    }
    let mut ics = args.csummm.icsumm();
    ics[4] = icase;
    args.csummm.set_icsumm(ics);

    if args.rot.irotsl == 1 {
        args.csummm.set_f64(38, args.rot.em)?;
    } else {
        args.csummm.set_f64(38, 0.0_f64)?;
    }

    let frmuhz = 16_666.666_667_f64 / per;
    let flmuhz = frmuhz / (args.rhs.el + 0.5_f64);
    if (state.out.icaswn >= 0 && icase != state.out.icaswn)
        || notwin(state.out.sigwn1, state.out.sigwn2, args.sig)
        || notwin(state.out.frqwn1, state.out.frqwn2, frmuhz)
        || notwni(state.out.iorwn1, state.out.iorwn2, args.iord)
        || notwin(state.out.frlwn1, state.out.frlwn2, flmuhz)
    {
        return Ok(());
    }

    for (idx, &dval) in args.rhs.data.iter().enumerate() {
        args.csummm.set_f64(2 + idx, dval)?;
    }
    args.csummm.set_f64(15, args.bcs.fctsbc)?;
    args.csummm.set_f64(16, args.bcs.fcttbc)?;
    args.csummm.set_f64(17, args.rhs.alb)?;
    args.csummm.set_f64(18, args.rhs.el)?;
    args.csummm.set_f64(20, args.sig)?;
    args.csummm.set_f64(19, f64::from(args.iord))?;

    if args.rot.irotsl == 1 {
        let nn0 = args.nn - 1;
        let om2 = 1.5_f64
            * (args.aa.get(10, nn0)? - 1.0_f64)
            * args.aa.get(1, nn0)?;
        args.csummm.set_f64(13, om2)?;
    }

    for i in 0..4 {
        args.csummm.set_f64(30 + i, args.y[nn_col + i])?;
    }

    let mut ics = args.csummm.icsumm();
    ics[2] = if args.cincnt.iriche == 1 {
        args.mdintg + 10
    } else {
        args.mdintg
    };
    ics[3] = state.varcon.ivarf;
    ics[5] = 3;
    ics[6] = args.cincnt.iekinr;
    args.csummm.set_icsumm(ics);

    if state.out.irotkr > 0 {
        return Err(AdiplsError::UnsupportedControl {
            control: "irotkr",
            message: format!(
                "sigout with irotkr={} (rotker) is unported in Package A2b",
                state.out.irotkr
            ),
        });
    }
    if state.out.igm1kr == 1 {
        return Err(AdiplsError::UnsupportedControl {
            control: "igm1kr",
            message: String::from(
                "sigout with igm1kr=1 (gm1ker) is unported in Package A2b",
            ),
        });
    }
    if ((state.out.nfmode >= 1 && args.isolcv != -1) || state.nfmscn == 1)
        && state.nscfil(state.dsn.idsefn)
    {
        return Err(AdiplsError::UnsupportedControl {
            control: "nfmode",
            message: format!(
                "sigout disk eigenfunction output (nfmode={}, nfmscn={}) is unported in Package A2b",
                state.out.nfmode, state.nfmscn
            ),
        });
    }

    if state.dsn.idsssm > 0 && state.nscfil(state.dsn.idsssm) {
        let ssm = setssm(args.csummm, &mut state.ssm_state)?;
        state.ssm_records.push(ssm);
    }

    let obs = setobs_st(
        args.csummm,
        args.cincnt.iriche,
        state.out.iper,
        state.out.ispcpr,
        state.out.irotkr,
        state.nobs_stmx,
        args.rhs.perfac,
        &state.crot_split,
        &mut state.obs_state,
    )?;

    if state.out.ispcpr != 0 {
        state.mode_store.push_from_spcout(&obs, args.csummm)?;
    }

    Ok(())
}
