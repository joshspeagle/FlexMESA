// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust ports of MESA 26.4.1 ADIPLS model setup, boundary conditions,
//! ODE right-hand sides, and eigenfunction scaling (`#171`, Package A2a):
//!
//! - [`setups_adi`] (`adipls/adipack.c/adipls/setups_adi.c.d.f`)
//! - [`res_adimod`] (`adipls/adipack.c/adipls/res_adimod.c.d.f`, honoring `#479`)
//! - [`setbcs`] (`adipls/adipack.c/adipls/setbcs.n.d.f`)
//! - [`adirhs`] (`adipls/adipack.c/adipls/adirhs.c.d.f`, honoring `#479`)
//! - [`geninh`] (`adipls/adipack.c/adipls/geninh.d.f`)
//! - [`danorm`] (`adipls/adipack.c/adipls/danorm.d.f`)
//! - [`xlmult`] (`adipls/adipack.c/adipls/xlmult.d.f`) and [`sclasl`]
//!   (`adipls/adipack.c/adipls/sclasl.c.d.f`)

use flexmesa_math::{intrinsic_exp, intrinsic_log, intrinsic_pow};

use crate::error::AdiplsError;
use crate::gensr::lininh4;
use crate::model::{AmdlAa, IAA_ARG};
use crate::parser::{
    AdiplsInput, ADIPLS_DEFAULT_EPSPRC, ADIPLS_DEFAULT_EPSUFL,
};

/// Maximum number of mesh points configured by [`setups_adi`] (`nnmax` in
/// `adipls.c.d.incl:4`).
pub const ADIPLS_NNMAX: usize = 20_000;

/// Maximum number of internally stored modes configured by [`setups_adi`]
/// (`nobs_stmax` in `setups_adi.c.d.f:25`).
pub const ADIPLS_NOBS_STMAX: usize = 20_000;

/// Solar mass constant hardcoded in `res_adimod.c.d.f:25` ($\mathrm{g}$).
pub const ADIPLS_AMSUN: f64 = 1.989e33_f64;

/// Default single-precision threshold used for radial degree checks (`el <= 1.e-6`).
#[inline]
#[must_use]
pub const fn sp_radial_eps() -> f64 {
    1.0e-6_f32 as f64
}

/// Workspace configuration initialized by [`setups_adi`]
/// (`adipls/adipack.c/adipls/setups_adi.c.d.f:1-77`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdiplsSetupState {
    /// Maximum mesh point capacity (`nnmax = 20000`).
    pub nnmax: usize,
    /// Maximum internal `obs_st` mode count (`nobs_stmx = 20000`).
    pub nobs_stmx: usize,
    /// NRKM workspace flag (`nnww = 0`).
    pub nnww: usize,
    /// Legacy standard print unit override (`istdpr_in = 6`).
    pub istdpr_in: i32,
}

/// Initialize ADIPLS workspace dimensions matching `setups_adi`
/// (`adipls/adipack.c/adipls/setups_adi.c.d.f:1-77`).
#[must_use]
pub const fn setups_adi() -> AdiplsSetupState {
    AdiplsSetupState {
        nnmax: ADIPLS_NNMAX,
        nobs_stmx: ADIPLS_NOBS_STMAX,
        nnww: 0,
        istdpr_in: 6,
    }
}

/// Numerical precision thresholds matching `common/cprcns/`
/// (`adipls/adipack.c/adipls/adipls.c.d.f:66, 137-140`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cprcns {
    /// Machine accuracy threshold (`epsprc = 1e-13`).
    pub epsprc: f64,
    /// Smallest positive scale (`epsufl = 1e-37`).
    pub epsufl: f64,
    /// Largest representable scale (`epsofl = 1 / epsufl`).
    pub epsofl: f64,
    /// Square root of `epsufl` (`eprufl = sqrt(epsufl)`).
    pub eprufl: f64,
}

impl Default for Cprcns {
    fn default() -> Self {
        let epsprc = ADIPLS_DEFAULT_EPSPRC;
        let epsufl = ADIPLS_DEFAULT_EPSUFL;
        Self {
            epsprc,
            epsufl,
            epsofl: 1.0_f64 / epsufl,
            eprufl: epsufl.sqrt(),
        }
    }
}

/// Equation parameters matching `common/rhsdat/` (`adipls/adipack.c/adipls/adipls.c.d.f:45-46`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhsDat {
    /// Spherical harmonic degree $l$ (`el`).
    pub el: f64,
    /// $l(l + 1)$ in spherical geometry or $l^2$ in plane-parallel (`ell`).
    pub ell: f64,
    /// Poisson fudge factor $\lambda$ (`alb`).
    pub alb: f64,
    /// $l(l + 1) / \sigma^2$ (`els`).
    pub els: f64,
    /// Exponent shift near the center (`el1 = el - 1` inside `nev1`, else `0`).
    pub el1: f64,
    /// Current dimensionless squared frequency $\sigma^2$ (`sig`).
    pub sig: f64,
    /// Residual exponent `anres` near a polytropic surface ($0 < n < 1$).
    pub anres: f64,
    /// Period conversion factor in minutes (`perfac`).
    pub perfac: f64,
    /// AMDL header array `data(1:8)`.
    pub data: [f64; 8],
}

impl Default for RhsDat {
    fn default() -> Self {
        Self {
            el: 1.0_f64,
            ell: 2.0_f64,
            alb: 1.0_f64,
            els: 0.0_f64,
            el1: 0.0_f64,
            sig: 0.0_f64,
            anres: 0.0_f64,
            perfac: 0.0_f64,
            data: [0.0_f64; 8],
        }
    }
}

/// Rotational parameters matching `common/rotdat/` and `common/comgrp/`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RotDat {
    /// Azimuthal order $m$ (`em`).
    pub em: f64,
    /// First-order rotational flag (`irotsl`).
    pub irotsl: i32,
    /// Profile rotation flag (`isprtp`).
    pub isprtp: i32,
    /// Rotation frequency profile `omgrtp`.
    pub omgrtp: Vec<f64>,
}

/// Boundary-condition controls matching `common/bcsdat/`
/// (`adipls/adipack.c/adipls/adipls.c.d.f:48`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BcsDat {
    /// Surface boundary interpolation factor (`fctsbc`).
    pub fctsbc: f64,
    /// Bottom boundary interpolation factor (`fcttbc`).
    pub fcttbc: f64,
    /// Surface mechanical boundary condition (`istsbc`).
    pub istsbc: i32,
    /// Bottom boundary condition for truncated models (`ibotbc`).
    pub ibotbc: i32,
}

impl Default for BcsDat {
    fn default() -> Self {
        Self {
            fctsbc: 0.0_f64,
            fcttbc: 0.0_f64,
            istsbc: 1,
            ibotbc: 0,
        }
    }
}

/// Integration controls matching `common/cincnt/` and `common/cincnt_new/`
/// (`adipls/adipack.c/adipls/adipls.c.d.f:50-53`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cincnt {
    /// Inner minimum radius for evanescent transition (`xmnevn`).
    pub xmnevn: f64,
    /// Fitting point radius (`xfit`).
    pub xfit: f64,
    /// NRKM normalization radius (`fcnorm`).
    pub fcnorm: f64,
    /// Eigenfrequency convergence tolerance (`eps`).
    pub eps: f64,
    /// Eigenfunction matching tolerance (`epssol`).
    pub epssol: f64,
    /// Maximum relative frequency step per iteration (`dsigmx`).
    pub dsigmx: f64,
    /// Secant initial relative step (`fsig`).
    pub fsig: f64,
    /// Original `fsig` before scan/Richardson adjustments (`fsig0`).
    pub fsig0: f64,
    /// Richardson extrapolation frequency offset (`dsigre`).
    pub dsigre: f64,
    /// Cowling approximation flag (`icow`).
    pub icow: i32,
    /// Turbulent pressure flag (`iturpr`).
    pub iturpr: i32,
    /// Plane-parallel geometry flag (`iplneq`).
    pub iplneq: i32,
    /// Richardson extrapolation flag (`iriche`).
    pub iriche: i32,
    /// NRKM input workspace size (`nnwwin`).
    pub nnwwin: i32,
    /// Determinant modification flag (`moddet`).
    pub moddet: i32,
    /// Maximum secant iterations (`itmax`).
    pub itmax: i32,
    /// Evanescent transition reset mode (`irsevn`).
    pub irsevn: i32,
    /// Fitting-point retry control (`nftmax`).
    pub nftmax: i32,
    /// Mode order classification scheme (`irsord`).
    pub irsord: i32,
    /// Kinetic energy normalization flag (`iekinr`).
    pub iekinr: i32,
    /// Trial-order check flag (`itsord`).
    pub itsord: i32,
    /// Ignored matching equation (`imissl`).
    pub imissl: i32,
    /// Fixed matching coefficient (`imjssl`).
    pub imjssl: i32,
    /// Matching coefficient selection mode (`imstsl`).
    pub imstsl: i32,
    /// Initial mode / acoustic cutoff stop control (`inomd1`).
    pub inomd1: i32,
}

impl Cincnt {
    /// Build from parsed `adipls.c.pruned.in` settings.
    #[must_use]
    pub const fn from_input(input: &AdiplsInput) -> Self {
        let ic = input.int_ctrl;
        let out = input.out;
        let dgn = input.dgn;
        Self {
            xmnevn: ic.xmnevn,
            xfit: ic.xfit,
            fcnorm: ic.fcnorm,
            eps: ic.eps,
            epssol: ic.epssol,
            dsigmx: ic.dsigmx,
            fsig: ic.fsig,
            fsig0: ic.fsig,
            dsigre: ic.dsigre,
            icow: ic.icow,
            iturpr: ic.iturpr,
            iplneq: ic.iplneq,
            iriche: ic.iriche,
            nnwwin: 0,
            moddet: dgn.moddet,
            itmax: ic.itmax,
            irsevn: ic.irsevn,
            nftmax: ic.nftmax,
            irsord: out.irsord,
            iekinr: out.iekinr,
            itsord: ic.itsord,
            imissl: dgn.imissl,
            imjssl: dgn.imjssl,
            imstsl: dgn.imstsl,
            inomd1: input.osc.inomd1,
        }
    }
}

/// Singular-surface expansion coefficients matching `common/csexpc/`
/// (`adipls/adipack.c/adipls/setbcs.n.d.f:82-84`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Csexpc {
    /// Linear term `y11`.
    pub y11: f64,
    /// Linear term `y13`.
    pub y13: f64,
    /// Linear term `y21`.
    pub y21: f64,
    /// Linear term `y23`.
    pub y23: f64,
    /// Linear term `y31`.
    pub y31: f64,
    /// Linear term `y33`.
    pub y33: f64,
    /// Linear term `y41`.
    pub y41: f64,
    /// Linear term `y43`.
    pub y43: f64,
    /// Fractional power term `yt11`.
    pub yt11: f64,
    /// Fractional power term `yt13`.
    pub yt13: f64,
    /// Fractional power term `yt21`.
    pub yt21: f64,
    /// Fractional power term `yt23`.
    pub yt23: f64,
    /// Fractional power term `yt31`.
    pub yt31: f64,
    /// Fractional power term `yt33`.
    pub yt33: f64,
    /// Fractional power term `yt41`.
    pub yt41: f64,
    /// Fractional power term `yt43`.
    pub yt43: f64,
    /// Second-order surface coefficient `yt241`.
    pub yt241: f64,
    /// Second-order surface coefficient `yt243`.
    pub yt243: f64,
    /// Nonlinear surface coefficient `ytt41`.
    pub ytt41: f64,
    /// Nonlinear surface coefficient `ytt43`.
    pub ytt43: f64,
}

/// NRKM boundary coefficients matching `common/cnrkbc/`
/// (`adipls/adipack.c/adipls/setbcs.n.d.f:80-81`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cnrkbc {
    /// Boundary condition matrix `cbc(4, 4)`.
    pub cbc: [[f64; 4]; 4],
    /// Central expansion recovery matrix `cavc(4, 4)`.
    pub cavc: [[f64; 4]; 4],
    /// Surface expansion recovery matrix `cavs(4, 4)`.
    pub cavs: [[f64; 4]; 4],
    /// Singular center flag (`isngbt`).
    pub isngbt: i32,
    /// Singular surface flag (`isngsf`).
    pub isngsf: i32,
    /// Bottom boundary flag (`ibotb1`).
    pub ibotb1: i32,
}

/// Cached exponentiation state for [`xlmult`] (`adipls/adipack.c/adipls/xlmult.d.f:29`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XlmultCache {
    xp: f64,
    elp: f64,
    axl: f64,
    icxp: i32,
}

impl Default for XlmultCache {
    fn default() -> Self {
        Self {
            xp: 0.0_f64,
            elp: 0.0_f64,
            axl: 0.0_f64,
            icxp: -1,
        }
    }
}

/// Resampled equilibrium model produced by [`res_adimod`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResAdimodOutput {
    /// Number of mesh points `nn` in the solver model.
    pub nn: usize,
    /// Potentially updated `iturpr` (`8` when `iggt == 1`).
    pub iturpr: i32,
    /// Model header scalars `data(1:8)`.
    pub data: [f64; 8],
    /// Fractional radius grid `x(1:nn)`.
    pub x: Vec<f64>,
    /// Ten-row `aa` matrix with rows `1..=5` and `10` set (unset input rows
    /// `6..=9` remain explicitly unset per `#479`).
    pub aa: AmdlAa,
}

/// Reset an input AMDL model into ADIPLS's internal mesh and `aa(10, :)`
/// layout, matching `res_adimod` (`adipls/adipack.c/adipls/res_adimod.c.d.f:1-195`).
///
/// Honors `#479` (`DECISIONS #43`): unset input rows `6..=9` are never read
/// when copying `aa_in` rows `1..=ivar_in`; if `iggt == 1` (`mod(idata8/10, 10)
/// == 2`) requests `aa(6, :)`, returns [`AdiplsError::UnsetAmdlRow`].
///
/// # Errors
/// Returns:
/// - [`AdiplsError::InvalidModel`] if array lengths are inconsistent or
///   subsampling with stride `in_step > 1` fails the `n1 == nn_in` check.
/// - [`AdiplsError::ConversionOverflow`] if `data_in[7] + 0.1_f32` cannot be
///   converted to `i32`.
/// - [`AdiplsError::UnsetAmdlRow`] if required rows `1..=5` are unset, or if
///   `iggt == 1` attempts to read unset row `6`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn res_adimod(
    x_in: &[f64],
    aa_in: &AmdlAa,
    data_in: [f64; 8],
    nn_in: usize,
    ivar_in: usize,
    in_step: i32,
    iriche: i32,
    iturpr: &mut i32,
    isprtp: i32,
    omgrtp: &mut [f64],
) -> Result<ResAdimodOutput, AdiplsError> {
    if nn_in == 0 || x_in.len() < nn_in || aa_in.nn() < nn_in {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "res_adimod dimension mismatch: nn_in={nn_in}, x_in={}, aa_in.nn()={}",
                x_in.len(),
                aa_in.nn()
            ),
        });
    }
    if ivar_in > IAA_ARG {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "res_adimod ivar_in={ivar_in} exceeds iaa={IAA_ARG}"
            ),
        });
    }

    let data = data_in;
    let d8_shifted = data_in[7] + f64::from(0.1_f32);
    if !d8_shifted.is_finite()
        || d8_shifted < f64::from(i32::MIN)
        || d8_shifted > f64::from(i32::MAX)
    {
        return Err(AdiplsError::ConversionOverflow {
            field: "data_in(8)",
            value: data_in[7],
        });
    }
    #[allow(clippy::cast_possible_truncation)]
    let idata8 = d8_shifted as i32;
    let iggt = if (idata8 / 10) % 10 == 2 {
        *iturpr = 8;
        // Explicit read of `aa_in(6, nn_in)` at `res_adimod.c.d.f:46` (#479).
        let _ = aa_in.get(6, nn_in - 1)?;
        true
    } else {
        false
    };

    let sincen = x_in[0] == 0.0_f64;
    let sinsur = data_in[6] >= 0.0_f64;
    let nsin = usize::from(sincen) + usize::from(sinsur);

    // Build index map `src_indices` (0-based into `x_in`/`aa_in`) matching
    // `res_adimod.c.d.f:65-158`.
    let (nn, src_indices): (usize, Vec<usize>) = if in_step <= 1 {
        let nshift = usize::from(!(iriche != 1 || (nn_in - nsin) % 2 == 1));
        let nn = nn_in.checked_sub(nshift).ok_or_else(|| {
            AdiplsError::InvalidModel {
                message: String::from("res_adimod nshift underflow"),
            }
        })?;
        let mut idx = Vec::with_capacity(nn);
        idx.push(0);
        for n in 1..nn {
            idx.push(n + nshift);
        }
        if isprtp != 0 {
            for (dst, &src) in idx.iter().enumerate().skip(1) {
                omgrtp[dst] = omgrtp[src];
            }
        }
        (nn, idx)
    } else {
        let step = usize::try_from(in_step).map_err(|_| {
            AdiplsError::InvalidModel {
                message: format!("invalid in_step={in_step}"),
            }
        })?;
        let mut nn = if sincen {
            (nn_in - 2) / step + 2
        } else {
            (nn_in - 1) / step + 1
        };
        if iriche == 1 && (nn - nsin).is_multiple_of(2) {
            nn -= 1;
        }
        let mut idx = Vec::with_capacity(nn);
        let mut n1_1based: isize;
        let nn_in_i = isize::try_from(nn_in).unwrap_or(isize::MAX);
        let step_i = isize::try_from(step).unwrap_or(1);
        let nn_i = isize::try_from(nn).unwrap_or(1);
        let nstart_1based = if sincen {
            idx.push(0);
            n1_1based = nn_in_i - step_i * (nn_i - 1);
            2_usize
        } else {
            n1_1based = nn_in_i - step_i * (nn_i + 1);
            1_usize
        };
        for n in nstart_1based..=nn {
            n1_1based += step_i;
            let src = usize::try_from(n1_1based - 1).map_err(|_| {
                AdiplsError::InvalidModel {
                    message: format!(
                        "res_adimod invalid source index {n1_1based}"
                    ),
                }
            })?;
            idx.push(src);
            if isprtp != 0 {
                omgrtp[n - 1] = omgrtp[src];
            }
        }
        if n1_1based != nn_in_i {
            return Err(AdiplsError::Solver {
                routine: "res_adimod",
                message: format!(
                    "subsampling mismatch: n1={n1_1based} != nn_in={nn_in} for in={in_step}"
                ),
            });
        }
        (nn, idx)
    };

    let mut x_out = Vec::with_capacity(nn);
    for &src in &src_indices {
        x_out.push(x_in[src]);
    }

    let mut aa_out = AmdlAa::new_unset(nn);
    for row in 1..=ivar_in {
        if row <= 5 || aa_in.is_row_set(row)? {
            let in_row = aa_in.row(row)?;
            let mut vals = Vec::with_capacity(nn);
            for &src in &src_indices {
                vals.push(in_row[src]);
            }
            aa_out.set_row(row, vals)?;
        }
    }

    // Set `aa(10, :)` (`res_adimod.c.d.f:162-179`).
    let mut row10 = Vec::with_capacity(nn);
    if *iturpr == 1 {
        let r1 = aa_out.row(1)?;
        for n in 0..nn {
            let xn = x_out[n];
            let ggt = if xn < f64::from(0.999_f32) {
                1.0_f64
            } else {
                1.0_f64 / (((xn * xn) * xn) * r1[n])
            };
            row10.push(ggt);
        }
    } else if iggt {
        let r6 = aa_out.row(6)?;
        row10.extend_from_slice(r6);
    } else {
        row10.resize(nn, 1.0_f64);
    }
    aa_out.set_row(10, row10)?;

    Ok(ResAdimodOutput {
        nn,
        iturpr: *iturpr,
        data,
        x: x_out,
        aa: aa_out,
    })
}

/// Set central and surface boundary conditions in `y` (or NRK boundary
/// matrices when `istnrk == 1`), matching `setbcs` (`adipls/adipack.c/adipls/setbcs.n.d.f:1-774`).
///
/// Updates `(nd, ny)`, `kdgbcs`, `*istsb1`, `bcs.ibotbc`, `csexpc`, and
/// `cnrkbc` identically to upstream. All indices `nn`, `nibc`, `nd`, `ny` are
/// 1-based.
///
/// # Errors
/// Returns:
/// - [`AdiplsError::Solver`] if `ig` is not `1` or `2`.
/// - [`AdiplsError::UnsetAmdlRow`] or [`AdiplsError::InvalidAmdlIndex`] on invalid `aa` access.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::manual_midpoint
)]
pub fn setbcs(
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    nn: usize,
    nibc: usize,
    icncbc: i32,
    istsb1: &mut i32,
    ii: usize,
    ig: usize,
    nd: &mut usize,
    ny: &mut usize,
    istnrk: i32,
    aa: &AmdlAa,
    rhs: &RhsDat,
    bcs: &mut BcsDat,
    cincnt: &Cincnt,
    csexpc: &mut Csexpc,
    cnrkbc: &mut Cnrkbc,
    kdgbcs: &mut i32,
) -> Result<(), AdiplsError> {
    const AC11: f64 = 1.0_f64;
    const AC31: f64 = 0.0_f64;
    const AC12: f64 = 1.0_f64;
    const AC32: f64 = 1.0_f64;
    const AS11: f64 = 1.0_f64;
    const AS31: f64 = 0.0_f64;
    const AS12: f64 = 0.0_f64;
    const AS32: f64 = 1.0_f64;

    *kdgbcs = 0;

    let el = rhs.el;
    let ell = rhs.ell;
    let alb = rhs.alb;
    let els = rhs.els;
    let sig = rhs.sig;
    let anres = rhs.anres;
    let data = &rhs.data;

    let radial = el <= sp_radial_eps();
    let fulmod = nibc == 2;
    let singsf = data[6] >= 0.0_f64;

    let kex = match ig {
        1 => 1_usize,
        2 => 5_usize,
        _ => {
            return Err(AdiplsError::Solver {
                routine: "setbcs",
                message: format!("ig = {ig} not allowed (expected 1 or 2)"),
            });
        }
    };

    let mut gm1 = 1.666_666_666_666_7_f64;

    if istnrk == 1 {
        cnrkbc.isngbt = 0;
        cnrkbc.isngsf = 0;
        cnrkbc.cbc = [[0.0_f64; 4]; 4];
        cnrkbc.cavc = [[0.0_f64; 4]; 4];
        cnrkbc.cavs = [[0.0_f64; 4]; 4];
    }

    // Center / inner boundary conditions (`setbcs.n.d.f:146-398`).
    if fulmod {
        cnrkbc.ibotb1 = -1;
        let p2 = data[4];
        let rh2 = data[5];
        let a2 = rh2 - p2;
        let bc = aa.get(1, 0)?;
        let x1 = x[1];

        if radial {
            // Radial center expansion (`setbcs.n.d.f:359-397`).
            let x2 = x1 * x1;
            if aa.get(3, 0)? > f64::from(0.1_f32) {
                gm1 = aa.get(3, 0)?;
            }
            let f2 = p2 * sig / bc;
            let c1 = if cincnt.icow == 0 {
                p2 * (sig / bc + 3.0_f64 * alb)
            } else {
                f2
            };
            let c12 = x1
                * f2
                * (-1.0_f64
                    + 0.1_f64 * (p2 * (1.0_f64 - 3.0_f64 * gm1) + c1) * x2)
                / 3.0_f64;
            let c22 = 1.0_f64 + x2 * (0.5_f64 * a2 - c1 / 6.0_f64);

            if istnrk == 1 {
                cnrkbc.cbc[0][0] = c22;
                cnrkbc.cbc[0][1] = -c12;
                cnrkbc.cavc[1][1] = 1.0_f64 / c22;
                cnrkbc.isngbt = 1;
            } else {
                y[0] = 0.0_f64;
                y[1] = 1.0_f64;
                y[iy] = c12;
                y[iy + 1] = c22;
            }
        } else {
            let ec = els * bc;
            let x2 = 0.5_f64 * x1 * x1 / (2.0_f64 * el + 3.0_f64);
            let f2 = 1.0_f64 - sig / (bc * el);
            let c1 = (p2 * (el + 2.0_f64) - ec * a2) * x2;
            let c2 = ell * (p2 - (el + 3.0_f64) * bc * a2 / sig) * x2;

            let c11 = 1.0_f64 + c1 * f2;
            let c13 = -c1;
            let c21 = el + 1.0_f64 + c2 * f2;
            let c23 = -c2;
            let c31 = -3.0_f64 * alb * (a2 + (el + 1.0_f64) * p2 / ec) * x2;
            let c3 = 3.0_f64
                * (0.4_f64 * (el - 1.0_f64) * rh2 + a2 + (1.0_f64 - alb) * p2)
                * x2;
            let c33 = 1.0_f64 + c3;
            let c41 = el * c31;
            let c43 = el - 2.0_f64 + el * c3;

            if istnrk == 1 {
                cnrkbc.isngbt = 1;
                if ii == 2 {
                    cnrkbc.cbc[0][0] = c21;
                    cnrkbc.cbc[0][1] = -c11;
                    cnrkbc.cavc[0][0] = 1.0_f64 / c11;
                    cnrkbc.cavc[1][0] = (el + 1.0_f64) / c11;
                } else {
                    let delta = c11 * c33 - c31 * c13;
                    cnrkbc.cbc[0][1] = -delta;
                    cnrkbc.cbc[0][0] = c21 * c33 - c23 * c31;
                    cnrkbc.cbc[0][2] = c23 * c11 - c21 * c13;
                    cnrkbc.cbc[1][3] = -delta;
                    cnrkbc.cbc[1][0] = c41 * c33 - c43 * c31;
                    cnrkbc.cbc[1][2] = c43 * c11 - c41 * c13;

                    cnrkbc.cavc[0][0] = c33 / delta;
                    cnrkbc.cavc[0][2] = -c13 / delta;
                    cnrkbc.cavc[1][0] = (el + 1.0_f64) * c33 / delta;
                    cnrkbc.cavc[1][2] = -(el + 1.0_f64) * c13 / delta;
                    cnrkbc.cavc[2][0] = -c31 / delta;
                    cnrkbc.cavc[2][2] = c11 / delta;
                    cnrkbc.cavc[3][0] = -(el - 2.0_f64) * c31 / delta;
                    cnrkbc.cavc[3][2] = (el - 2.0_f64) * c11 / delta;
                }
            } else {
                let mut a10 = AC11;
                let mut a30 = AC31;
                let mut k = 1_usize;
                while k <= kex {
                    let k0 = k - 1;
                    y[iy + k0] = c11 * a10 + c13 * a30;
                    y[iy + k0 + 1] = c21 * a10 + c23 * a30;
                    if ii != 2 {
                        y[iy + k0 + 2] = c31 * a10 + c33 * a30;
                        y[iy + k0 + 3] = c41 * a10 + c43 * a30;
                    }
                    y[k0] = a10;
                    y[k0 + 1] = (el + 1.0_f64) * a10;
                    if ii != 2 {
                        y[k0 + 2] = a30;
                        y[k0 + 3] = (el - 2.0_f64) * a30;
                    }
                    a10 = AC12;
                    a30 = AC32;
                    k += 4;
                }

                if icncbc == 2 && el != 1.0_f64 {
                    let xl1 = intrinsic_pow(x1, el - 1.0_f64);
                    let mut k = 1_usize;
                    while k <= kex {
                        for i1 in (k - 1..).take(ii) {
                            y[iy + i1] *= xl1;
                            if el > 1.0_f64 {
                                y[i1] = 0.0_f64;
                            }
                        }
                        k += 4;
                    }
                }
            }
        }
    } else {
        cnrkbc.ibotb1 = bcs.ibotbc;
        if radial && bcs.ibotbc == 0 {
            bcs.ibotbc = 1;
            cnrkbc.ibotb1 = 1;
            *kdgbcs -= 1;
        }
        let nibc0 = nibc - 1;
        let (c11, c21) = if bcs.ibotbc == 1 {
            (1.0_f64 - bcs.fcttbc, bcs.fcttbc)
        } else if bcs.ibotbc == 0 {
            let qxs = aa.get(1, nibc0)? / sig;
            let ak1 = 1.0_f64 - aa.get(2, nibc0)? / (qxs * ell);
            let ak2 = 1.0_f64 - qxs * aa.get(4, nibc0)?;
            let akr = ak1 * ak2;
            if akr > 0.0_f64 {
                (1.0_f64, (ell * akr).sqrt() / ak1)
            } else {
                cnrkbc.ibotb1 = 1;
                *kdgbcs -= 2;
                (1.0_f64 - bcs.fcttbc, bcs.fcttbc)
            }
        } else if bcs.ibotbc == 2 {
            let c11 = if radial {
                -sig * aa.get(2, nibc0)? / (x[nibc0] * aa.get(1, nibc0)?)
            } else {
                1.0_f64 - sig * aa.get(2, nibc0)? / (ell * aa.get(1, nibc0)?)
            };
            (c11, 2.0_f64 - aa.get(2, nibc0)?)
        } else {
            let c11 = if radial {
                -sig * aa.get(2, nibc0)? / (x[nibc0] * aa.get(1, nibc0)?)
            } else {
                1.0_f64 - sig * aa.get(2, nibc0)? / (ell * aa.get(1, nibc0)?)
            };
            (c11, 3.0_f64 - aa.get(2, nibc0)?)
        };

        if istnrk == 1 {
            cnrkbc.cbc[0][0] = c21;
            cnrkbc.cbc[0][1] = -c11;
        } else {
            y[nibc0 * iy] = c11;
            y[nibc0 * iy + 1] = c21;
        }
    }

    // Surface boundary conditions (`setbcs.n.d.f:404-763`).
    let an = data[6];
    let mut ny1 = 0_usize;
    let mut us = 0.0_f64;

    let (c11, c13, c21, c23, c31, c33, c41, c43): (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    );

    if singsf {
        // Singular surface (`setbcs.n.d.f:508-657`).
        *nd = nn - 1;
        *ny = *nd + 1;
        ny1 = *nd + 2;
        let nd0 = *nd - 1;
        let ang = (an + 1.0_f64) / gm1;
        let as_val = an - ang;
        let dt = 1.0_f64 - x[nd0];

        let u01 = aa.get(5, nd0)? / intrinsic_pow(dt, an);
        let u02 = u01 / (1.0_f64 - (3.0_f64 - an) * dt);
        let corr = (8.0_f64 / ((an + 1.0_f64) * (an + 2.0_f64)))
            * u02
            * intrinsic_pow(dt, an + 1.0_f64)
            / (1.0_f64 - (3.0_f64 - an) * dt);
        let u0 = 2.0_f64 * u02 / (1.0_f64 + (1.0_f64 + corr).sqrt());

        if radial {
            // Radial singular surface (`setbcs.n.d.f:631-657`).
            csexpc.y11 = (2.0_f64 * gm1 - 4.0_f64 - sig) / gm1;
            csexpc.yt11 = 0.0_f64;
            c11 = 1.0_f64 + csexpc.y11 * dt;
            csexpc.y13 = 0.0_f64;
            csexpc.yt13 = 0.0_f64;
            c13 = 0.0_f64;

            if an <= f64::from(1.0e-10_f32) {
                csexpc.y21 = -(3.0_f64
                    + 4.0_f64 / gm1
                    + sig * (1.0_f64 + 1.0_f64 / gm1))
                    / sig;
                c21 = 1.0_f64 / sig + csexpc.y21 * dt;
                csexpc.yt21 = 0.0_f64;
            } else {
                csexpc.y21 = (4.0_f64 * as_val - sig * (ang + 1.0_f64))
                    / ((an + 1.0_f64) * sig);
                csexpc.yt21 = u0 / (sig * (an + 1.0_f64));
                c21 = 1.0_f64 / sig
                    + csexpc.y21 * dt
                    + csexpc.yt21 * intrinsic_pow(dt, an + 1.0_f64);
            }
            csexpc.y23 = 0.0_f64;
            csexpc.yt23 = 0.0_f64;
            c23 = 0.0_f64;
            c31 = 0.0_f64;
            c33 = 0.0_f64;
            c41 = 0.0_f64;
            c43 = 0.0_f64;
        } else if an > f64::from(1.0e-10_f32) {
            let dt_an = intrinsic_pow(dt, an);
            let dt_an1 = intrinsic_pow(dt, an + 1.0_f64);
            let dt_2an1 = intrinsic_pow(dt, 2.0_f64 * an + 1.0_f64);

            csexpc.y11 = -1.0_f64
                - ((els - 3.0_f64) * (1.0_f64 + as_val)
                    + ang * (1.0_f64 + sig))
                    / (an + 1.0_f64);
            csexpc.yt11 =
                (1.0_f64 - alb) * u0 / (gm1 * (2.0_f64 * an + 1.0_f64));
            c11 = 1.0_f64 + csexpc.y11 * dt + csexpc.yt11 * dt_an1;

            csexpc.y13 = ((1.0_f64 + as_val) * els + (el + 1.0_f64) * ang)
                / (an + 1.0_f64);
            csexpc.yt13 = 0.0_f64;
            c13 = csexpc.y13 * dt;

            csexpc.y21 = -els
                + (els * as_val * (6.0_f64 - els)
                    + (1.0_f64 + ang) * (2.0_f64 * els - ell))
                    / (an + 1.0_f64);
            csexpc.yt21 = (1.0_f64 - alb) * els * u0 * as_val
                / ((an + 1.0_f64) * (2.0_f64 * an + 1.0_f64));
            c21 = els + csexpc.y21 * dt + csexpc.yt21 * dt_an1;

            csexpc.y23 = els
                * (1.0_f64
                    + (as_val * (els - 3.0_f64 - el)
                        - 2.0_f64 * (1.0_f64 + ang))
                        / (an + 1.0_f64));
            c23 = -els + csexpc.y23 * dt;

            csexpc.y31 = 0.0_f64;
            csexpc.yt31 = -alb * u0 / (an + 1.0_f64);
            c31 = csexpc.yt31 * dt_an1;

            csexpc.y33 = el - 1.0_f64;
            csexpc.yt33 = u0 / (an + 1.0_f64);
            c33 = 1.0_f64 + csexpc.y33 * dt + csexpc.yt33 * dt_an1;

            csexpc.y41 = 0.0_f64;
            csexpc.yt41 = u0 * alb;
            csexpc.yt241 = u0
                * (alb * as_val * csexpc.y11 + (alb * ang / els) * csexpc.y21
                    - (as_val + (1.0_f64 - alb) * ang) * csexpc.y31
                    + alb * (an * (an - 2.0_f64) - 2.0_f64 - 3.0_f64 * ang))
                / (1.0_f64 + an);
            let denom_sq = intrinsic_pow(2.0_f64 * an + 1.0_f64, 2.0_f64);
            csexpc.ytt41 = u0
                * u0
                * alb
                * (((1.0_f64 - alb) / denom_sq)
                    * (4.0_f64 * an + 1.0_f64
                        - 2.0_f64 * (an + 1.0_f64) / gm1)
                    / gm1
                    + (an + 4.0_f64) / ((an + 1.0_f64) * (an + 2.0_f64)));
            c41 = if anres == 0.0_f64 {
                csexpc.yt41 * dt_an
                    + csexpc.yt241 * dt_an1
                    + csexpc.ytt41 * dt_2an1
            } else {
                csexpc.yt241 * dt_an1 + csexpc.ytt41 * dt_2an1
            };

            csexpc.y43 = -el * (el - 1.0_f64);
            csexpc.yt43 = -u0;
            csexpc.yt243 = u0
                * (alb * as_val * csexpc.y13 + (alb * ang / els) * csexpc.y23
                    - (as_val + (1.0_f64 - alb) * ang) * csexpc.y33
                    + (3.0_f64 * alb * ang - an * (an - 2.0_f64) + 4.0_f64
                        - 2.0_f64 * el))
                / (1.0_f64 + an);
            csexpc.ytt43 =
                -u0 * u0 * (an + 4.0_f64) / ((an + 1.0_f64) * (an + 2.0_f64));
            c43 = if anres == 0.0_f64 {
                -el + csexpc.y43 * dt
                    + csexpc.yt43 * dt_an
                    + csexpc.yt243 * dt_an1
                    + csexpc.ytt43 * dt_2an1
            } else {
                -el + csexpc.y43 * dt
                    + csexpc.yt243 * dt_an1
                    + csexpc.ytt43 * dt_2an1
            };
            us = 0.0_f64;
        } else {
            csexpc.y11 =
                els * (ang - 1.0_f64) - ang * (4.0_f64 + sig) + 2.0_f64;
            csexpc.y13 = (1.0_f64 + el) * ang - els * (ang - 1.0_f64);
            csexpc.y21 = els * (els * ang + 1.0_f64 - 4.0_f64 * ang)
                - ell * (1.0_f64 + ang);
            csexpc.y23 = els * ((1.0_f64 + el - els) * ang - 1.0_f64);
            csexpc.y31 = -3.0_f64;
            csexpc.y33 = 2.0_f64 + el;
            csexpc.y41 =
                3.0_f64 * ang * (-csexpc.y11 + csexpc.y21 / els + csexpc.y31)
                    + 12.0_f64;
            csexpc.y43 = -(6.0_f64 + el * (el + 5.0_f64))
                + 3.0_f64 * ang * (-csexpc.y13 + csexpc.y23 / els + csexpc.y33);

            c11 = 1.0_f64 + csexpc.y11 * dt;
            c13 = csexpc.y13 * dt;
            c21 = els + csexpc.y21 * dt;
            c23 = -els + csexpc.y23 * dt;
            c31 = csexpc.y31 * dt;
            c33 = 1.0_f64 + csexpc.y33 * dt;
            c41 = 3.0_f64 + csexpc.y41 * dt;
            c43 = -(3.0_f64 + el) + csexpc.y43 * dt;
            us = 3.0_f64;
        }
    } else {
        *nd = nn;
        *ny = *nd + 1;
        let nn0 = nn - 1;
        let fctsb1 = 1.0_f64 - bcs.fctsbc;

        let mut c21_iso = 0.0_f64;
        let mut vg_iso = 0.0_f64;
        let mut a_iso = 0.0_f64;
        let mut sigx_iso = 0.0_f64;
        let mut elsx_iso = 0.0_f64;

        if *istsb1 == 1 {
            vg_iso = aa.get(2, nn0)?;
            if aa.get(3, 0)? > f64::from(0.1_f32) {
                gm1 = aa.get(3, nn0)?;
            }
            sigx_iso = sig / aa.get(1, nn0)?;
            elsx_iso = ell / sigx_iso;
            a_iso = vg_iso * (gm1 - 1.0_f64);
            let g0 = a_iso - vg_iso + 4.0_f64;
            let gamma =
                g0 * g0 + 4.0_f64 * (elsx_iso - vg_iso) * (sigx_iso - a_iso);
            if gamma >= 0.0_f64 {
                c21_iso = ((vg_iso - a_iso + gamma.sqrt()) / 2.0_f64 - 2.0_f64)
                    / (vg_iso - elsx_iso);
            } else {
                *istsb1 = 0;
                *kdgbcs -= 20;
            }
        }

        if *istsb1 == 9 {
            c11 = 0.0_f64;
            c13 = 0.0_f64;
            c21 = 1.0_f64;
            c23 = 0.0_f64;
            c31 = 0.0_f64;
            c33 = 1.0_f64;
            c41 = 0.0_f64;
            c43 = -el - aa.get(5, nn0)?;
        } else if radial {
            c11 = 1.0_f64;
            c13 = 0.0_f64;
            c23 = 0.0_f64;
            c31 = 0.0_f64;
            c33 = 0.0_f64;
            c41 = 0.0_f64;
            c43 = 0.0_f64;
            c21 = if *istsb1 == 1 {
                x[nn0] * c21_iso / sigx_iso
            } else {
                fctsb1 * aa.get(1, nn0)? * x[nn0] / sig
            };
        } else {
            c11 = 1.0_f64;
            c13 = 0.0_f64;
            c31 = 0.0_f64;
            c33 = 1.0_f64;
            c41 = aa.get(5, nn0)?;
            c43 = -el - c41;
            if *istsb1 == 1 {
                c21 = elsx_iso * c21_iso;
                c23 = -elsx_iso
                    * (1.0_f64 - (el + 1.0_f64 - elsx_iso) / (vg_iso + a_iso));
            } else {
                let etat = els * aa.get(1, nn0)?;
                c21 = fctsb1 * etat;
                c23 = -aa.get(10, nn0)? * etat;
            }
        }
    }

    if istnrk == 1 {
        if an >= 0.0_f64 {
            cnrkbc.isngsf = 1;
        }
        if ii == 2 {
            cnrkbc.cbc[2][0] = c21;
            cnrkbc.cbc[2][1] = -c11;
            if cnrkbc.isngsf == 0 {
                return Ok(());
            }
            cnrkbc.cavs[0][0] = 1.0_f64 / c11;
            cnrkbc.cavs[1][0] = if radial {
                1.0_f64 / (sig * c11)
            } else {
                els / c11
            };
        } else {
            let delta = c11 * c33 - c31 * c13;
            cnrkbc.cbc[2][1] = -delta;
            cnrkbc.cbc[2][0] = c21 * c33 - c23 * c31;
            cnrkbc.cbc[2][2] = c23 * c11 - c21 * c13;
            cnrkbc.cbc[3][3] = -delta;
            cnrkbc.cbc[3][0] = c41 * c33 - c43 * c31;
            cnrkbc.cbc[3][2] = c43 * c11 - c41 * c13;
            if cnrkbc.isngsf == 0 {
                return Ok(());
            }
            cnrkbc.cavs[0][0] = c33 / delta;
            cnrkbc.cavs[0][2] = -c13 / delta;
            cnrkbc.cavs[1][0] = els * (c31 + c33) / delta;
            cnrkbc.cavs[1][2] = -els * (c11 + c13) / delta;
            cnrkbc.cavs[2][0] = -c31 / delta;
            cnrkbc.cavs[2][2] = c11 / delta;
            cnrkbc.cavs[3][0] = (us * c33 + (el + us) * c31) / delta;
            cnrkbc.cavs[3][2] = -(us * c13 + (el + us) * c11) / delta;
        }
        return Ok(());
    }

    // Populate surface values `y(:, ny)` (and `y(:, ny1)` when `an >= 0`).
    let ny0 = *ny - 1;
    let ny1_0 = ny1.saturating_sub(1);
    let mut a10 = AS11;
    let mut a30 = AS31;
    let mut k = 1_usize;
    while k <= kex {
        let k0 = k - 1;
        y[ny0 * iy + k0] = a10 * c11 + a30 * c13;
        y[ny0 * iy + k0 + 1] = a10 * c21 + a30 * c23;
        if ii != 2 {
            y[ny0 * iy + k0 + 2] = a10 * c31 + a30 * c33;
            y[ny0 * iy + k0 + 3] = a10 * c41 + a30 * c43;
        }
        if an >= 0.0_f64 {
            y[ny1_0 * iy + k0] = a10;
            y[ny1_0 * iy + k0 + 1] =
                if radial { a10 / sig } else { els * (a10 - a30) };
            if ii != 2 {
                y[ny1_0 * iy + k0 + 2] = a30;
                y[ny1_0 * iy + k0 + 3] = us * a10 - (el + us) * a30;
            }
        }
        a30 = AS32;
        a10 = AS12;
        k += 4;
    }

    Ok(())
}

/// Evaluate the adiabatic oscillation ODE right-hand side matrix `fd(1:ii, 1:ii)`
/// and inhomogeneous forcing `finh`, matching `adirhs`
/// (`adipls/adipack.c/adipls/adirhs.c.d.f:1-252`).
///
/// Honors `#479` (`DECISIONS #43`): if `iturpr == 2` attempts to read `aa(6, n)`
/// when row `6` is unset, returns [`AdiplsError::UnsetAmdlRow`].
///
/// # Errors
/// Returns [`AdiplsError::UnsetAmdlRow`] or [`AdiplsError::InvalidAmdlIndex`]
/// when reading `aa`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
pub fn adirhs(
    x: f64,
    fd: &mut [[f64; 4]; 4],
    finh: &mut [f64; 8],
    n_1based: usize,
    aa: &AmdlAa,
    rhs: &RhsDat,
    rot: &RotDat,
    cincnt: &Cincnt,
    csexpc: &Csexpc,
) -> Result<(), AdiplsError> {
    let n = n_1based - 1;
    let el = rhs.el;
    let ell = rhs.ell;
    let alb = rhs.alb;
    let els = rhs.els;
    let el1 = rhs.el1;
    let sig = rhs.sig;
    let anres = rhs.anres;
    let iturpr = cincnt.iturpr;
    let icow = cincnt.icow;

    if cincnt.iplneq == 1 {
        let a2 = aa.get(2, n)?;
        let a4 = aa.get(4, n)?;
        fd[0][0] = a2 - el1;
        fd[0][1] = 1.0_f64 - a2 / els;
        fd[1][0] = ell - a4 * els;
        fd[1][1] = a4 - el1;
        finh[0] = 0.0_f64;
        finh[1] = 0.0_f64;
        return Ok(());
    }

    let xi = 1.0_f64 / x;
    let vgt = aa.get(2, n)?;
    let at = aa.get(4, n)?;
    let u = aa.get(5, n)?;

    let (ggt, vg, a) = if iturpr == 1 || iturpr == 8 {
        let ggt = aa.get(10, n)?;
        let vg = ggt * vgt;
        let a = at - (vg - vgt);
        (ggt, vg, a)
    } else {
        (1.0_f64, vgt, at)
    };

    let (vg1, vg0, a1, a0) = if iturpr == 2 {
        let vg1 = aa.get(2, n)?;
        let r6 = aa.get(6, n)?;
        let vg0 = vg1 + r6;
        let a1 = aa.get(4, n)?;
        let a0 = a1 - r6;
        (vg1, vg0, a1, a0)
    } else {
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64)
    };

    if el <= sp_radial_eps() {
        // Radial case (`adirhs.c.d.f:223-251`).
        let qxt = aa.get(1, n)?;
        let qx = ggt * qxt;
        if iturpr == 2 {
            fd[0][0] = (vg0 - 2.0_f64) * xi;
            fd[0][1] = (-vg1 * sig) / ((qx * x) * x);
            fd[1][0] = if icow == 0 {
                1.0_f64 - (qxt * a0 - alb * qx * u) / sig
            } else {
                1.0_f64 - qxt * a0 / sig
            };
            fd[1][1] = a1 * xi;
        } else {
            fd[0][0] = (vgt - 2.0_f64) * xi;
            fd[0][1] = (-vg * sig) / ((qx * x) * x);
            fd[1][0] = if icow == 0 {
                1.0_f64 - (qxt * at - alb * qx * u) / sig
            } else {
                1.0_f64 - qxt * at / sig
            };
            fd[1][1] = at * xi;
        }
        finh[0] = 0.0_f64;
        finh[1] = 0.0_f64;
        return Ok(());
    }

    // Nonradial case (`adirhs.c.d.f:123-219`).
    let etat = els * aa.get(1, n)?;
    let eta = ggt * etat;
    let vge = vg / eta;
    let aet = at * etat;

    if iturpr == 2 {
        fd[0][0] = (vg0 - el1 - 2.0_f64) * xi;
        fd[0][1] = (1.0_f64 - vg1 / eta) * xi;
        fd[1][0] = (ell - eta * a0) * xi;
        fd[1][1] = (a1 - el1 - 1.0_f64) * xi;
    } else {
        fd[0][0] = (vgt - el1 - 2.0_f64) * xi;
        fd[0][1] = (1.0_f64 - vge) * xi;
        fd[1][0] = (ell - aet) * xi;
        fd[1][1] = (at - el1 - 1.0_f64) * xi;
    }

    finh[0] = 0.0_f64;
    finh[1] = 0.0_f64;

    if icow != 0 {
        return Ok(());
    }

    if iturpr == 2 {
        fd[0][2] = -vg1 * xi;
        fd[0][3] = 0.0_f64;
        fd[1][2] = eta * a1 * xi;
        fd[1][3] = 0.0_f64;
        fd[2][0] = 0.0_f64;
        fd[2][1] = 0.0_f64;
        fd[2][2] = (1.0_f64 - el1) * xi;
        fd[2][3] = xi;
        fd[3][0] = -alb * u * a0 * xi;
        fd[3][1] = -alb * u * (vg1 / eta) * xi;
        fd[3][2] = (ell + u * (a1 - 2.0_f64) + (1.0_f64 - alb) * u * vg1) * xi;
        fd[3][3] = (2.0_f64 - el1 - 2.0_f64 * u) * xi;
    } else {
        fd[0][2] = -vg * xi;
        fd[0][3] = 0.0_f64;
        fd[1][2] = eta * at * xi;
        fd[1][3] = 0.0_f64;
        fd[2][0] = 0.0_f64;
        fd[2][1] = 0.0_f64;
        fd[2][2] = (1.0_f64 - el1) * xi;
        fd[2][3] = xi;
        fd[3][0] = -alb * u * at * xi;
        fd[3][1] = -alb * u * vge * xi;
        fd[3][2] = (ell + u * (a - 2.0_f64) + (1.0_f64 - alb) * u * vg) * xi;
        fd[3][3] = (2.0_f64 - el1 - 2.0_f64 * u) * xi;
    }

    finh[4] = 0.0_f64;
    finh[5] = 0.0_f64;

    if anres == 0.0_f64 {
        finh[2] = 0.0_f64;
        finh[3] = 0.0_f64;
        finh[6] = 0.0_f64;
        finh[7] = 0.0_f64;
    } else {
        let t = 1.0_f64 - x;
        let tn = intrinsic_pow(t, anres);
        finh[2] = xi * csexpc.yt41 * tn;
        finh[3] = (fd[3][3] + anres / t) * csexpc.yt41 * tn;
        finh[6] = xi * csexpc.yt43 * tn;
        finh[7] = (fd[3][3] + anres / t) * csexpc.yt43 * tn;
    }

    if rot.irotsl == 1 && rot.em != 0.0_f64 {
        let sigom2 = 1.5_f64 * (aa.get(10, n)? - 1.0_f64) * aa.get(1, n)?;
        let sigom = sigom2.sqrt();
        let sighat = sig.sqrt() + rot.em * sigom;
        let sigfct = sighat * sighat / sig;
        let alpha = 2.0_f64 * rot.em * sigom / sighat;
        let ellfct = ell / (ell - alpha);
        let h1 = ellfct * alpha;
        let zeta = ellfct * etat / sigfct;

        fd[0][0] += h1 * xi;
        fd[0][1] = (zeta - vgt) * xi / etat;
        fd[1][0] = (ell * sigfct - etat * (at + h1 * h1 / zeta)) * xi;
        fd[1][1] -= h1 * xi;
        fd[3][2] =
            (ell + u * (a - 2.0_f64) + (1.0_f64 / ggt - alb) * u * vg) * xi;
    }

    Ok(())
}

/// Dispatch initial-value ODE integration according to `mdintg`, matching
/// `geninh` (`adipls/adipack.c/adipls/geninh.d.f:1-60`).
///
/// Only `mdintg == 5` (4th-order Cash-Moore via [`lininh4`]) is reached in
/// MESA 26.4.1; other `mdintg` values fail loudly with
/// [`AdiplsError::UnsupportedControl`].
///
/// # Errors
/// Returns [`AdiplsError::UnsupportedControl`] for `mdintg != 5` or propagates
/// [`AdiplsError`] from [`lininh4`] / [`adirhs`].
#[allow(clippy::too_many_arguments)]
pub fn geninh(
    x: &[f64],
    y: &mut [f64],
    ii: usize,
    iy: usize,
    ig: usize,
    isn: isize,
    nd1: usize,
    nn: usize,
    x_start: usize,
    y_start: usize,
    mdintg: i32,
    aa: &AmdlAa,
    rhs: &RhsDat,
    rot: &RotDat,
    cincnt: &Cincnt,
    csexpc: &Csexpc,
) -> Result<i32, AdiplsError> {
    if mdintg != 5 {
        return Err(AdiplsError::UnsupportedControl {
            control: "mdintg",
            message: format!(
                "geninh only supports mdintg=5 in Package A2a (got mdintg={mdintg})"
            ),
        });
    }
    lininh4(
        x,
        y,
        iy,
        ii,
        ig,
        isn,
        nd1,
        nn,
        x_start,
        y_start,
        |xv, fd, finh, nd| {
            adirhs(xv, fd, finh, nd, aa, rhs, rot, cincnt, csexpc)
        },
    )?;
    Ok(0)
}

/// Overflow-safe Euclidean norm of `a[..n]` using underflow threshold `eprufl`,
/// matching `danorm` (`adipls/adipack.c/adipls/danorm.d.f:1-34`).
#[must_use]
pub fn danorm(a: &[f64], n: usize, eprufl: f64) -> f64 {
    let mut amx = 0.0_f64;
    for &val in a.iter().take(n) {
        amx = amx.max(val.abs());
    }
    if amx <= 0.0_f64 {
        return 0.0_f64;
    }
    let mut sum = 0.0_f64;
    for &val in a.iter().take(n) {
        let mut b = (val / amx).abs();
        if b < eprufl {
            b = 0.0_f64;
        }
        sum += b * b;
    }
    sum.sqrt() * amx
}

/// Evaluate `y * x**el` with overflow/sign guards and single-call caching,
/// matching `xlmult` (`adipls/adipack.c/adipls/xlmult.d.f:1-124`) and routing `pow`/`log`/`exp`
/// through `flexmesa_math::intrinsic_*` (`DECISIONS #60`).
///
/// Returns `(value, ierr)`.
pub fn xlmult(
    y: f64,
    x: f64,
    el: f64,
    icxm: i32,
    cache: &mut XlmultCache,
) -> (f64, i32) {
    const ALIMLN: f64 = 85.0_f64;

    if y == 0.0_f64 {
        return (0.0_f64, 0);
    }

    #[allow(clippy::float_cmp)]
    let samex = x == cache.xp && el == cache.elp && icxm == cache.icxp;
    cache.xp = x;
    cache.elp = el;
    cache.icxp = icxm;

    let mut ierr = 0_i32;
    let (mut isx, ax) = if x < 0.0_f64 {
        let ax = -x;
        #[allow(clippy::cast_possible_truncation)]
        let iel = el as i32;
        #[allow(clippy::float_cmp)]
        let s = if f64::from(iel) == el {
            if iel % 2 != 0 {
                -1_i32
            } else {
                1_i32
            }
        } else {
            ierr = -1;
            1_i32
        };
        (s, ax)
    } else if x == 0.0_f64 {
        if el < 0.0_f64 {
            return (0.0_f64, -2);
        }
        if el == 0.0_f64 {
            return (y, 0);
        }
        return (0.0_f64, 0);
    } else {
        (1_i32, x)
    };

    if icxm == 0 {
        if !samex {
            cache.axl = intrinsic_pow(ax, el);
        }
        let xlm = f64::from(isx) * y * cache.axl;
        return (xlm, ierr);
    }

    let ay = if y < 0.0_f64 {
        isx = -isx;
        -y
    } else {
        y
    };
    if !samex {
        cache.axl = el * intrinsic_log(ax);
    }
    let xml = intrinsic_log(ay) + cache.axl;
    if xml.abs() > ALIMLN {
        return (0.0_f64, 1);
    }
    let xlm = f64::from(isx) * intrinsic_exp(xml);
    (xlm, ierr)
}

/// Rescale interior solution by `(x / x(nev1))**(el - 1)` (or exponential in
/// plane-parallel geometry), matching `sclasl` (`adipls/adipack.c/adipls/sclasl.c.d.f:1-45`).
///
/// Indices `nev1` and `nw1` are 1-based.
#[allow(clippy::too_many_arguments)]
pub fn sclasl(
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    _nn: usize,
    ii: usize,
    nev1: usize,
    nw1: usize,
    el: f64,
    iplneq: i32,
    cache: &mut XlmultCache,
) {
    let xev1 = x[nev1 - 1];
    if iplneq == 1 {
        for n in nw1..=nev1 {
            let yy = intrinsic_exp(el * (x[n - 1] - xev1));
            let base = (n - 1) * iy;
            for i in 0..ii {
                y[base + i] *= yy;
            }
        }
    } else {
        let icxm = i32::from(el > 15.0_f64);
        for n in nw1..=nev1 {
            let xx = x[n - 1] / xev1;
            let base = (n - 1) * iy;
            for i in 0..ii {
                let (yy, _) =
                    xlmult(y[base + i], xx, el - 1.0_f64, icxm, cache);
                y[base + i] = yy;
            }
        }
    }
}
