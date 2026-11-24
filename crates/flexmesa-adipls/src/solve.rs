// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust ports of MESA 26.4.1 ADIPLS shooting, iteration, matching,
//! mode-order classification, and frequency-scanning routines (`#171`,
//! Package A2a):
//!
//! - [`shtint`] (`adipls/adipack.c/adipls/shtint.c.d.f`)
//! - [`sigint`] (`adipls/adipack.c/adipls/sigint.n.d.f`)
//! - [`mchcff`], [`mchcf1`] (`adipls/adipack.c/adipls/mchcff.d.f`)
//! - [`mchsol`] (`adipls/adipack.c/adipls/mchsol.n.d.f`)
//! - [`stevft`] (`adipls/adipack.c/adipls/stevft.n.d.f`)
//! - [`rsnfit`] (`adipls/adipack.c/adipls/rsnfit.n.d.f`)
//! - [`order`] (`adipls/adipack.c/adipls/order.d.f`) and [`takata`]
//!   (`adipls/adipack.c/adipls/takata.d.f`)
//! - [`signew`] (`adipls/adipack.c/adipls/signew.n.d.f`) and [`signw_pg`]
//!   (`adipls/adipack.c/adipls/sigscn.c.d.f`)
//! - [`sigitr`] (`adipls/adipack.c/adipls/sigitr.n.d.f`)
//! - [`sigitc`] (`adipls/adipack.c/adipls/sigitc.c.d.f`)
//! - [`sigsol`] (`adipls/adipack.c/adipls/sigsol.c.d.f`)
//! - [`sigscn`] (`adipls/adipack.c/adipls/sigscn.c.d.f`)
//! - [`setlog`] (`adipls/adipack.c/adipls/adipls.c.d.f:2095-2205`)

use flexmesa_math::{intrinsic_pow, powi_f90};

use crate::error::AdiplsError;
use crate::gensr::{leq, notwin};
use crate::model::AmdlAa;
use crate::output::Csummm;
use crate::rhs::{
    danorm, geninh, sclasl, setbcs, sp_radial_eps, BcsDat, Cincnt, Cnrkbc,
    Cprcns, Csexpc, RhsDat, RotDat, XlmultCache,
};

/// Intermediate oracle event matching `apply_adipls_taps.py` (`#171`).
#[derive(Debug, Clone, PartialEq)]
pub enum AdiplsTapEvent {
    /// Start of a per-model `adipls` call (`adipls.c.d.f`, label `10000`).
    Enter,
    /// Exit of `sigint` (`sigint.n.d.f:92`).
    Det {
        /// Squared dimensionless frequency $\sigma^2$.
        sig: f64,
        /// Matching determinant.
        det: f64,
        /// Scaled determinant residual `|det| / detnrm`.
        ddsol: f64,
        /// Radial order `iord`.
        iord: i32,
        /// Integration return code `icry`.
        icry: i32,
        /// 1-based fitting point index `nfit`.
        nfit: usize,
    },
    /// Normal exit of `mchsol` (`mchsol.n.d.f:146-148`).
    Match {
        /// Matching determinant / residual returned by `mchcff` or shooting.
        det: f64,
        /// Matched solution vector `y(1:ii, nfit)` at the fitting point.
        y_nfit: Vec<f64>,
    },
    /// Exit of `sigitc` (`adipls/adipack.c/adipls/sigitc.c.d.f:108`).
    Itc {
        /// Converged (or final) $\sigma^2$.
        sig: f64,
        /// Last relative step `|dsig / sig|`.
        ddsig: f64,
        /// Last normalized continuity residual `ddsol`.
        ddsol: f64,
        /// Fractional fitting radius `x(nfit)`.
        xfit1: f64,
        /// 1-based fitting point index `nfit`.
        nfit: usize,
        /// Mode radial order `iord`.
        iord: i32,
        /// Status code `icry`.
        icry: i32,
        /// Solution convergence flag `isolcv` (`1` converged, `-1` failed).
        isolcv: i32,
        /// Frequency convergence flag `isigcv` (`1` converged, `< 0` failed).
        isigcv: i32,
        /// Log status indicator `istatus`.
        istatus: i32,
    },
    /// Exit of `kiner` (`adipls/adipack.c/adipls/kiner.n.d.f:84`).
    Kiner {
        /// Spherical harmonic degree `el`.
        el: f64,
        /// Normalization squared displacement `ynorm`.
        ynorm: f64,
        /// Dimensionless kinetic energy integral `ekin`.
        ekin: f64,
    },
    /// Exit of `varfrq` (`adipls/adipack.c/adipls/varfrq.n.d.f:401`).
    Varfrq {
        /// Eigenvalue $\sigma^2$ passed into `varfrq`.
        sig: f64,
        /// Variational squared dimensionless frequency `sigv`.
        sigv: f64,
    },
}

impl AdiplsTapEvent {
    /// Return the upstream tag name (`ENTER`, `DET`, `MATCH`, `ITC`, `KINER`,
    /// `VARFRQ`).
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::Enter => "ENTER",
            Self::Det { .. } => "DET",
            Self::Match { .. } => "MATCH",
            Self::Itc { .. } => "ITC",
            Self::Kiner { .. } => "KINER",
            Self::Varfrq { .. } => "VARFRQ",
        }
    }

    /// Convert payload to the exact `f64` sequence written by `a0trec`.
    #[must_use]
    pub fn to_values(&self) -> Vec<f64> {
        match self {
            Self::Enter => Vec::new(),
            Self::Det {
                sig,
                det,
                ddsol,
                iord,
                icry,
                nfit,
            } => vec![
                *sig,
                *det,
                *ddsol,
                f64::from(*iord),
                f64::from(*icry),
                f64::from(i32::try_from(*nfit).unwrap_or(i32::MAX)),
            ],
            Self::Match { det, y_nfit } => {
                let mut out = Vec::with_capacity(1 + y_nfit.len());
                out.push(*det);
                out.extend_from_slice(y_nfit);
                out
            }
            Self::Itc {
                sig,
                ddsig,
                ddsol,
                xfit1,
                nfit,
                iord,
                icry,
                isolcv,
                isigcv,
                istatus,
            } => vec![
                *sig,
                *ddsig,
                *ddsol,
                *xfit1,
                f64::from(i32::try_from(*nfit).unwrap_or(i32::MAX)),
                f64::from(*iord),
                f64::from(*icry),
                f64::from(*isolcv),
                f64::from(*isigcv),
                f64::from(*istatus),
            ],
            Self::Kiner { el, ynorm, ekin } => vec![*el, *ynorm, *ekin],
            Self::Varfrq { sig, sigv } => vec![*sig, *sigv],
        }
    }
}

/// Sink notified at intermediate ADIPLS tap points (`ENTER`, `DET`, `MATCH`,
/// `ITC`).
pub trait AdiplsTapObserver {
    /// Record one intermediate tap event.
    fn on_tap(&mut self, event: AdiplsTapEvent);
}

/// Zero-overhead no-op observer for production runs.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTapObserver;

impl AdiplsTapObserver for NoopTapObserver {
    #[inline]
    fn on_tap(&mut self, _event: AdiplsTapEvent) {}
}

/// In-memory tap collector for parity and regression testing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TapRecorder {
    /// Captured events in exact execution order.
    pub events: Vec<AdiplsTapEvent>,
}

impl TapRecorder {
    /// Create an empty tap recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self { events: Vec::new() }
    }
}

impl AdiplsTapObserver for TapRecorder {
    #[inline]
    fn on_tap(&mut self, event: AdiplsTapEvent) {
        self.events.push(event);
    }
}

/// Diagnostic log entry emitted by [`setlog`] (`adipls/adipack.c/adipls/adipls.c.d.f:2095-2205`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiStatusLogEntry {
    /// Convergence/diagnostic status code (`1..=9`).
    pub istatus: i32,
    /// Spherical harmonic degree `el`.
    pub el: f64,
    /// Radial order `iord`.
    pub iord: i32,
    /// Period in minutes (`per`).
    pub per_min: f64,
    /// Cyclic frequency in $\mu\mathrm{Hz}$ (`anu`).
    pub nu_uhz: f64,
    /// Dimensionless squared frequency `sig`.
    pub sig: f64,
    /// Relative change in `sig` (`ddsig`).
    pub ddsig: f64,
    /// Normalized eigenfunction discontinuity (`ddsol`).
    pub ddsol: f64,
    /// Active fitting radius `xfit1`.
    pub xfit1: f64,
    /// Original fitting radius `xfitst`.
    pub xfitst: f64,
}

/// Persistent state for [`signw_pg`] (`nsig == 4` mixed p/g-mode scan in
/// `adipls/adipack.c/adipls/sigscn.c.d.f:443-480`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SignwPgState {
    beta: f64,
    gamma: f64,
    bsigc: f64,
}

/// Mode post-processing arguments passed across the Package A2a -> A2b
/// `sigout` boundary (`adipls/adipack.c/adipls/sigout.c.d.f:1-2`).
pub struct SigoutArgs<'a> {
    /// Dimensionless squared frequency $\sigma^2$.
    pub sig: f64,
    /// Fractional radius grid `x(1:nn)`.
    pub x: &'a [f64],
    /// Solution buffer `y(1:iy, 1:nn+1)`.
    pub y: &'a mut [f64],
    /// First dimension of `y` (`8`).
    pub iy: usize,
    /// Active ODE system order `ii` (`2` or `4`, `common/sysord/`).
    pub ii: usize,
    /// 1-based inner solution start index `nw1`.
    pub nw1: usize,
    /// 1-based inner boundary index `nibc`.
    pub nibc: usize,
    /// Number of mesh points `nn`.
    pub nn: usize,
    /// Number of solution points `nnw = nn - nw1 + 1`.
    pub nnw: usize,
    /// Integration method `mdintg`.
    pub mdintg: i32,
    /// 1-based evanescent transition index `nev1`.
    pub nev1: usize,
    /// 1-based fitting point index `nfit`.
    pub nfit: usize,
    /// Active surface boundary condition flag `istsb1`.
    pub istsb1: i32,
    /// Radial order `iord`.
    pub iord: i32,
    /// Eigenfunction convergence flag `isolcv`.
    pub isolcv: i32,
    /// Eigenfrequency convergence flag `isigcv`.
    pub isigcv: i32,
    /// Model structure matrix `aa`.
    pub aa: &'a AmdlAa,
    /// Equation parameters `rhs`.
    pub rhs: &'a RhsDat,
    /// Rotational state `rot`.
    pub rot: &'a RotDat,
    /// Boundary-condition parameters `bcs`.
    pub bcs: &'a BcsDat,
    /// Integration controls `cincnt`.
    pub cincnt: &'a Cincnt,
    /// Precision parameters `cprcns`.
    pub cprcns: &'a Cprcns,
    /// Grand summary buffer `csummm`.
    pub csummm: &'a mut Csummm,
    /// Tap observer for `KINER` / `VARFRQ` instrumentation.
    pub observer: &'a mut dyn AdiplsTapObserver,
}

/// Callback boundary invoked whenever [`sigscn`] or `adipls` finishes a mode
/// and upstream calls `sigout` (Package A2b).
pub trait SigoutHandler {
    /// Process one solved (or scanned) mode.
    ///
    /// # Errors
    /// Propagates any post-processing error.
    fn on_sigout(
        &mut self,
        args: &mut SigoutArgs<'_>,
    ) -> Result<(), AdiplsError>;
}

/// Default no-op `sigout` handler used by Package A2a scan/solve tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopSigoutHandler;

impl SigoutHandler for NoopSigoutHandler {
    #[inline]
    fn on_sigout(
        &mut self,
        _args: &mut SigoutArgs<'_>,
    ) -> Result<(), AdiplsError> {
        Ok(())
    }
}

/// Mutable per-call ADIPLS solver context bundled for passing across solve
/// routines without global common blocks.
pub struct SolverContext<'a> {
    /// Equation parameters (`common/rhsdat/`).
    pub rhs: &'a mut RhsDat,
    /// Rotational parameters (`common/rotdat/`).
    pub rot: &'a mut RotDat,
    /// Boundary condition parameters (`common/bcsdat/`).
    pub bcs: &'a mut BcsDat,
    /// Integration controls (`common/cincnt/`).
    pub cincnt: &'a mut Cincnt,
    /// Singular-surface expansion terms (`common/csexpc/`).
    pub csexpc: &'a mut Csexpc,
    /// NRKM boundary coefficients (`common/cnrkbc/`).
    pub cnrkbc: &'a mut Cnrkbc,
    /// Precision thresholds (`common/cprcns/`).
    pub cprcns: &'a Cprcns,
    /// Diagnostic boundary flag (`common/cdgsbr/`).
    pub kdgbcs: &'a mut i32,
    /// Gravitational constant `cgrav` (`common/ccgrav/`).
    pub cgrav: f64,
    /// Cached `xlmult` state.
    pub xlmult_cache: &'a mut XlmultCache,
    /// `signw_pg` state (`nsig == 4`).
    pub signw_pg_state: &'a mut SignwPgState,
    /// Grand summary buffer (`common/csumma/`).
    pub csummm: &'a mut Csummm,
    /// Target trial radial order `iordtr` (`common/cmdtst/`).
    pub iordtr: i32,
    /// Status log entries collected via [`setlog`].
    pub status_log: &'a mut Vec<AdiStatusLogEntry>,
}

/// Shooting integration from center and surface and matching determinant
/// computation, matching `shtint` (`adipls/adipack.c/adipls/shtint.c.d.f:1-235`).
///
/// All indices (`nw1`, `nibc`, `nn`, `nev1`, `nd`, `ny`, `nfit`) are 1-based.
///
/// # Errors
/// Propagates errors from [`geninh`] / [`adirhs`](crate::rhs::adirhs).
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
pub fn shtint(
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    _nw1: usize,
    nibc: usize,
    _nn: usize,
    nev1: usize,
    nd: usize,
    ny: usize,
    iasn: usize,
    ii: usize,
    ig: usize,
    nfit: usize,
    mdintg: i32,
    det: &mut f64,
    ddsol: &mut f64,
    ds1: &mut [f64; 16],
    iord: &mut i32,
    icry: &mut i32,
    aa: &AmdlAa,
    ctx: &mut SolverContext<'_>,
) -> Result<(), AdiplsError> {
    *icry = 0;

    if nibc == 0 || nev1 < nibc || nfit < nev1 || nd < nfit {
        return Err(AdiplsError::Solver {
            routine: "shtint",
            message: format!(
                "invalid shooting bounds nibc={nibc}, nev1={nev1}, nfit={nfit}, nd={nd}"
            ),
        });
    }

    if iasn > 1
        && (!(nev1 - nibc).is_multiple_of(iasn)
            || !(nfit - nev1).is_multiple_of(iasn)
            || !(nd - nfit).is_multiple_of(iasn))
    {
        *icry = -1;
        return Ok(());
    }

    let nd1 = nibc;
    let nni1 = (nev1 - nd1) / iasn + 1;
    let nd2 = nev1;
    let nni2 = (nfit - nd2) / iasn + 1;
    let nnis = (nd - nfit) / iasn + 1;

    *iord = 0;
    let iasn_s = isize::try_from(iasn).unwrap_or(1);

    // Integrate from center (`shtint.c.d.f:92-110`).
    let mut isn = iasn_s;
    ctx.rhs.anres = 0.0_f64;

    if nni1 > 1 {
        ctx.rhs.el1 = if ctx.cincnt.iplneq == 1 {
            ctx.rhs.el
        } else {
            ctx.rhs.el - 1.0_f64
        };
        let iclcn = geninh(
            x,
            y,
            ii,
            iy,
            ig,
            isn,
            nd1,
            nni1,
            nibc - 1,
            nibc - 1,
            mdintg,
            aa,
            ctx.rhs,
            ctx.rot,
            ctx.cincnt,
            ctx.csexpc,
        )?;
        *iord = iclcn;
    }

    // Integrate in intermediate region (`shtint.c.d.f:113-121`).
    if nni2 > 1 {
        ctx.rhs.el1 = 0.0_f64;
        let iclcn = geninh(
            x,
            y,
            ii,
            iy,
            ig,
            isn,
            nd2,
            nni2,
            nev1 - 1,
            nev1 - 1,
            mdintg,
            aa,
            ctx.rhs,
            ctx.rot,
            ctx.cincnt,
            ctx.csexpc,
        )?;
        *iord += iclcn;
    }

    // Integrate from surface (`shtint.c.d.f:125-174`).
    isn = -iasn_s;
    ctx.rhs.el1 = 0.0_f64;
    ctx.rhs.anres = if ctx.rhs.data[6] < 1.0_f64
        && ctx.rhs.data[6] > 0.0_f64
        && ii == 4
        && mdintg == 1
    {
        ctx.rhs.data[6]
    } else {
        0.0_f64
    };

    let iclcn = geninh(
        x,
        y,
        ii,
        iy,
        ig,
        isn,
        nd,
        nnis,
        nd - 1,
        ny - 1,
        mdintg,
        aa,
        ctx.rhs,
        ctx.rot,
        ctx.cincnt,
        ctx.csexpc,
    )?;

    if ctx.rhs.anres != 0.0_f64 {
        for n in 1..=nnis {
            let n1 = nd + 1 - n;
            let n2 = ny + 1 - n;
            let t = 1.0_f64 - x[n1 - 1];
            let tn = intrinsic_pow(t, ctx.rhs.anres);
            let y_col = (n2 - 1) * iy;
            y[y_col + 3] += ctx.csexpc.yt41 * tn;
            y[y_col + 7] += ctx.csexpc.yt43 * tn;
        }
    }

    *iord += iclcn;

    if mdintg == 2 && y[(nibc - 1) * iy] * y[(nibc - 1) * iy + 1] < 0.0_f64 {
        *iord += 1;
    }

    // Calculate matching determinant (`shtint.c.d.f:179-223`).
    let detnrm = if ig == 1 {
        let n0 = nfit - 1;
        let n1 = nfit;
        *det = y[n0 * iy] * y[n1 * iy + 1] - y[n0 * iy + 1] * y[n1 * iy];
        danorm(&y[n0 * iy..], 2, ctx.cprcns.eprufl)
            * danorm(&y[n1 * iy..], 2, ctx.cprcns.eprufl)
    } else {
        let mut ds = [0.0_f64; 16];
        let mut j = 0_usize;
        for n in nfit..=nfit + 1 {
            let col_base = (n - 1) * iy;
            let mut l = 0_usize;
            for _k in 0..2 {
                for i in 0..4 {
                    let val = y[col_base + l];
                    ds[j * 4 + i] = val;
                    ds1[j * 4 + i] = val;
                    l += 1;
                }
                j += 1;
            }
        }
        let mut nrm = 1.0_f64;
        for j in 0..4 {
            nrm *= danorm(&ds[j * 4..], 4, ctx.cprcns.eprufl);
        }
        let mut dummy_b = [0.0_f64; 4];
        *det = leq(&mut ds, &mut dummy_b, 4, 0, 4, 4);
        nrm
    };

    if detnrm == 0.0_f64 {
        *icry = -1;
    } else {
        *ddsol = det.abs() / detnrm;
    }

    Ok(())
}

/// Integrate the oscillation equations for fixed `(sig, el)` by setting
/// boundary conditions via [`setbcs`] and integrating via [`shtint`], matching
/// `sigint` (`adipls/adipack.c/adipls/sigint.n.d.f:1-93`).
///
/// Emits [`AdiplsTapEvent::Det`] on exit.
///
/// # Errors
/// Returns [`AdiplsError::UnsupportedControl`] if `mdintg == 3` (`nrkint` is
/// unported/unreached) or propagates errors from [`setbcs`] / [`shtint`].
#[allow(clippy::too_many_arguments)]
pub fn sigint(
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    nw1: usize,
    nibc: usize,
    nn: usize,
    mdintg: i32,
    nev1: usize,
    iasn: usize,
    ii: usize,
    ig: usize,
    _initnr: &mut i32,
    nfit: usize,
    istsb1: &mut i32,
    det: &mut f64,
    ddsol: &mut f64,
    ds1: &mut [f64; 16],
    iord: &mut i32,
    icry: &mut i32,
    aa: &AmdlAa,
    ctx: &mut SolverContext<'_>,
    observer: &mut impl AdiplsTapObserver,
) -> Result<(), AdiplsError> {
    if mdintg == 3 {
        return Err(AdiplsError::UnsupportedControl {
            control: "mdintg",
            message: String::from(
                "sigint with mdintg=3 (nrkint) is unported in Package A2a",
            ),
        });
    }

    let istnrk = 0_i32;
    let icncbc = if nibc == nev1 && ctx.cincnt.irsevn < 3 {
        2
    } else {
        1
    };
    ctx.rhs.anres = if ctx.rhs.data[6] > 0.0_f64
        && ctx.rhs.data[6] < 1.0_f64
        && ii == 4
        && mdintg == 1
    {
        ctx.rhs.data[6]
    } else {
        0.0_f64
    };

    let mut nd = 0_usize;
    let mut ny = 0_usize;
    setbcs(
        x, y, iy, nn, nibc, icncbc, istsb1, ii, ig, &mut nd, &mut ny, istnrk,
        aa, ctx.rhs, ctx.bcs, ctx.cincnt, ctx.csexpc, ctx.cnrkbc, ctx.kdgbcs,
    )?;
    ctx.rhs.anres = 0.0_f64;

    shtint(
        x, y, iy, nw1, nibc, nn, nev1, nd, ny, iasn, ii, ig, nfit, mdintg, det,
        ddsol, ds1, iord, icry, aa, ctx,
    )?;

    observer.on_tap(AdiplsTapEvent::Det {
        sig: ctx.rhs.sig,
        det: *det,
        ddsol: *ddsol,
        iord: *iord,
        icry: *icry,
        nfit,
    });

    Ok(())
}

/// Compute matching coefficients `c(1:4)` for one ignored equation `is`
/// (1-based `1..=4`) and fixed coefficient `js` (1-based `1..=4`), matching
/// `mchcf1` (`adipls/adipack.c/adipls/mchcff.d.f:72-149`).
///
/// `ds` is column-major `4x4`. Returns the residual norm `err` (`-1.0` when
/// singular).
#[must_use]
#[allow(clippy::similar_names)]
pub fn mchcf1(ds: &[f64; 16], c: &mut [f64; 4], is: usize, js: usize) -> f64 {
    let mut ds1 = *ds;
    let is0 = is - 1;
    let js0 = js - 1;

    for i in 0..4 {
        ds1[i * 4 + is0] = 0.0_f64;
        ds1[js0 * 4 + i] = 0.0_f64;
        c[i] = -ds[js0 * 4 + i];
    }
    ds1[js0 * 4 + is0] = 1.0_f64;
    c[js0] = 0.0_f64;

    let det = leq(&mut ds1, c, 4, 1, 4, 4);
    if det == 0.0_f64 {
        return -1.0_f64;
    }

    c[js0] = 1.0_f64;

    let mut cmax = 0.0_f64;
    let mut imax = 0_usize;
    for (i, &ci) in c.iter().enumerate() {
        let ca = ci.abs();
        if ca >= cmax {
            imax = i;
            cmax = ca;
        }
    }

    let cimax = f64::from(1.0_f32) / c[imax];
    for ci in c.iter_mut() {
        *ci *= cimax;
    }

    let mut err = 0.0_f64;
    for i in 0..4 {
        let mut ssi = 0.0_f64;
        for j in 0..4 {
            ssi += ds[j * 4 + i] * c[j];
        }
        err += ssi * ssi;
    }
    err.sqrt()
}

/// Select matching coefficients `c(1:4)` for the almost-singular `4x4` matrix
/// `ds`, matching `mchcff` (`adipls/adipack.c/adipls/mchcff.d.f:1-71`).
///
/// When `isetsl == 1`, searches all 16 `(is, js)` pairs and updates `*is` and
/// `*js` to the minimum-residual pair.
pub fn mchcff(
    ds: &[f64; 16],
    c: &mut [f64; 4],
    is: &mut i32,
    js: &mut i32,
    isetsl: i32,
    err: &mut f64,
) {
    if isetsl != 1 {
        let isu = usize::try_from(*is).unwrap_or(1).clamp(1, 4);
        let jsu = usize::try_from(*js).unwrap_or(1).clamp(1, 4);
        *err = mchcf1(ds, c, isu, jsu);
        return;
    }

    let mut ismin = 0_i32;
    let mut jsmin = 0_i32;
    let mut errmin = 0.0_f64;
    let mut c1 = [0.0_f64; 4];

    for is1 in 1..=4 {
        for js1 in 1..=4 {
            let e = mchcf1(ds, &mut c1, is1, js1);
            *err = e;
            if e < 0.0_f64 {
                continue;
            }
            if ismin > 0 && e > errmin {
                continue;
            }
            ismin = i32::try_from(is1).unwrap_or(1);
            jsmin = i32::try_from(js1).unwrap_or(1);
            errmin = e;
            *c = c1;
        }
    }

    if ismin == 0 {
        return;
    }

    *is = ismin;
    *js = jsmin;
    *err = errmin;
}

/// Combine center- and surface-integrated shooting solutions at fitting point
/// `nfit`, matching `mchsol` (`adipls/adipack.c/adipls/mchsol.n.d.f:1-166`).
///
/// Emits [`AdiplsTapEvent::Match`] on normal completion.
///
/// # Errors
/// Returns [`AdiplsError::UnsupportedControl`] if `moddet == 1` (`anldet` is
/// unported in Package A2a).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn mchsol(
    y: &mut [f64],
    iy: usize,
    nw1: usize,
    nn: usize,
    ii: usize,
    ig: usize,
    nfit: usize,
    imissl: &mut i32,
    imjssl: &mut i32,
    imstsl: i32,
    moddet: i32,
    det: &mut f64,
    ds1: &[f64; 16],
    icry: &mut i32,
    observer: &mut impl AdiplsTapObserver,
) -> Result<(), AdiplsError> {
    if ig == 1 {
        let col_fit = (nfit - 1) * iy;
        let if_idx = usize::from(y[col_fit] == 0.0_f64);
        if y[col_fit + if_idx] == 0.0_f64 {
            *icry = -1;
            return Ok(());
        }
        let c1 = y[nfit * iy + if_idx] / y[col_fit + if_idx];
        for n in nw1..=nfit {
            let base = (n - 1) * iy;
            y[base] *= c1;
            y[base + 1] *= c1;
        }
        for n in (nfit + 1)..=nn {
            let dst = (n - 1) * iy;
            let src = n * iy;
            y[dst] = y[src];
            y[dst + 1] = y[src + 1];
        }
    } else {
        let mut c = [0.0_f64; 4];
        mchcff(ds1, &mut c, imissl, imjssl, imstsl, det);
        if *det < 0.0_f64 {
            *icry = -1;
            return Ok(());
        }
        if moddet == 1 {
            return Err(AdiplsError::UnsupportedControl {
                control: "moddet",
                message: String::from(
                    "mchsol with moddet=1 (anldet) is unported in Package A2a",
                ),
            });
        }
        for n in nw1..=nfit {
            let base = (n - 1) * iy;
            for i in 0..4 {
                y[base + i] = c[0] * y[base + i] + c[1] * y[base + i + 4];
            }
        }
        for n in (nfit + 1)..=nn {
            let dst = (n - 1) * iy;
            let src = n * iy;
            for i in 0..4 {
                y[dst + i] = -c[2] * y[src + i] - c[3] * y[src + i + 4];
            }
        }
    }

    *icry = 0;
    let fit_base = (nfit - 1) * iy;
    let y_nfit = y[fit_base..fit_base + ii].to_vec();
    observer.on_tap(AdiplsTapEvent::Match { det: *det, y_nfit });
    Ok(())
}

/// Locate the inner evanescent transition index `nev1` (and optionally reset
/// `nfit` / `xfit1`), matching `stevft` (`adipls/adipack.c/adipls/stevft.n.d.f:1-152`).
///
/// All mesh indices (`nibc`, `nfit`, `nev1`) are 1-based.
///
/// # Errors
/// Propagates [`AdiplsError`] from `aa.get(...)`.
#[allow(clippy::too_many_arguments, clippy::similar_names, clippy::float_cmp)]
pub fn stevft(
    x: &[f64],
    nn: usize,
    nibc: usize,
    nfit: &mut usize,
    iasn: usize,
    radial: bool,
    fulmod: bool,
    nev1: &mut usize,
    xfit1: &mut f64,
    aa: &AmdlAa,
    rhs: &RhsDat,
    cincnt: &Cincnt,
) -> Result<(), AdiplsError> {
    if radial || cincnt.irsevn == -1 {
        *nev1 = nibc;
    } else {
        *nev1 = *nfit;
        if *nfit > 5 || cincnt.xfit != -1.0_f64 {
            let nsev = if fulmod { 5_usize } else { nibc };
            let nlev = if cincnt.xfit == -1.0_f64 || cincnt.irsevn >= 2 {
                nn
            } else {
                *nfit
            };
            for n in nsev..=nlev {
                *nev1 = n;
                let a1 = aa.get(1, n - 1)?;
                let a2 = aa.get(2, n - 1)?;
                let a4 = aa.get(4, n - 1)?;
                let akr = (1.0_f64 - rhs.sig * a2 / (a1 * rhs.ell))
                    * (1.0_f64 - a1 * a4 / rhs.sig);
                if akr < f64::from(0.25_f32) && x[n - 1] >= cincnt.xmnevn {
                    break;
                }
            }
        }
    }

    if *nev1 <= nibc + 5 {
        *nev1 = nibc;
    }
    if nn > 30 && *nev1 >= nn - 30 {
        *nev1 = nn - 30;
    }
    if !(*nev1 - nibc).is_multiple_of(iasn) {
        let nstp = (*nev1 - nibc) / iasn + 1;
        *nev1 = nibc + nstp * iasn;
    }
    if cincnt.xfit == -1.0_f64 {
        *nfit = *nev1;
        *xfit1 = x[*nfit - 1];
    }
    if *nev1 > nibc && cincnt.irsevn >= 2 && *nev1 > *nfit {
        *nfit = *nev1;
        *xfit1 = x[*nfit - 1];
    }

    Ok(())
}

/// Shift the fitting point index `nfit` on convergence retry, matching
/// `rsnfit` (`adipls/adipack.c/adipls/rsnfit.n.d.f:1-68`) including `f32` `0.05` arithmetic.
pub fn rsnfit(
    x: &[f64],
    nibc: usize,
    nn: usize,
    nfit: &mut usize,
    xfit1: &mut f64,
    nftcnt: &mut i32,
    nftrcs: i32,
) {
    *nftcnt += 1;
    if nn == 0 || x.len() < nn {
        return;
    }
    let lo = 10_usize.min(nn).max(nibc.min(nn)).max(1);
    let hi = nn.saturating_sub(10).max(lo).min(nn);
    let nn_i32 = i32::try_from(nn).unwrap_or(i32::MAX);

    if nftrcs == 0 {
        let lo_i32 = i32::try_from(lo).unwrap_or(1);
        let hi_i32 = i32::try_from(hi).unwrap_or(nn_i32);
        #[allow(clippy::cast_precision_loss)]
        let nfit_f = (*nfit) as f32;
        #[allow(clippy::cast_precision_loss)]
        let step_f = (0.05_f32 * (*nftcnt as f32)) * (nn_i32 as f32);
        let new_nfit = if *nftcnt % 2 == 1 {
            #[allow(clippy::cast_possible_truncation)]
            let cand = (nfit_f - step_f).round() as i32;
            cand.max(lo_i32)
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let cand = (nfit_f + step_f).round() as i32;
            cand.min(hi_i32)
        };
        *nfit = usize::try_from(new_nfit).unwrap_or(lo).clamp(lo, hi);
    } else {
        let dx = 0.05_f64 * (*xfit1).min(1.0_f64 - *xfit1);
        if *nftcnt % 2 == 1 {
            *xfit1 -= f64::from(*nftcnt) * dx;
        } else {
            *xfit1 += f64::from(*nftcnt) * dx;
        }
        let mut nf = 1_usize;
        for (n0, &xn) in x.iter().take(nn).enumerate() {
            nf = n0 + 1;
            if xn > *xfit1 {
                break;
            }
        }
        *nfit = nf.clamp(lo, hi);
    }

    if ((*nfit).cast_signed() - nibc.cast_signed()).rem_euclid(2) != 0 {
        if *nfit < nn {
            *nfit += 1;
        } else if *nfit > 1 {
            *nfit -= 1;
        }
    }
    *xfit1 = x[*nfit - 1];
}

/// Takata (2006) dipolar ($l = 1$) radial order classifier matching `takata`
/// (`adipls/adipack.c/adipls/takata.d.f:1-96`).
///
/// Preserves upstream's literal `pi = 3.141593_f64` (`takata.d.f:37`),
/// `powi_f90(data[1], 3)`, and `data[1].powi(2) -> data[1] * data[1]`.
///
/// # Errors
/// Propagates [`AdiplsError`] from `aa.get(...)`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn takata(
    x: &[f64],
    y: &[f64],
    nn: usize,
    iasn: usize,
    data: &[f64; 8],
    aa: &AmdlAa,
    iy: usize,
    sig: f64,
    cgrav: f64,
) -> Result<i32, AdiplsError> {
    #[allow(clippy::approx_constant)]
    const PI_TAKATA: f64 = 3.141_593_f64;
    let g = cgrav;
    let r3 = powi_f90(data[1], 3);
    let r2 = data[1] * data[1];

    let mut yy1 = vec![0.0_f64; nn];
    let mut yy2 = vec![0.0_f64; nn];

    for n in 1..=nn {
        if n == 1 || (n - 2) % iasn == 0 {
            let n0 = n - 1;
            let a1 = aa.get(1, n0)?;
            let a5 = aa.get(5, n0)?;
            let rho_n = data[0] * a1 * a5 / (4.0_f64 * PI_TAKATA * r3);
            let j_n = 1.0_f64 - a5 / 3.0_f64;
            let gr_n = g * a1 * x[n0] * data[0] / r2;
            if x[n0] > 0.0_f64 {
                let y_base = n0 * iy;
                let dphipdr_n = -(gr_n / x[n0])
                    * (y[y_base + 3] - y[y_base + 2])
                    - (y[y_base + 2]
                        * g
                        * data[1]
                        * 4.0_f64
                        * PI_TAKATA
                        * rho_n);
                let t2_n = -y[y_base + 2] / (3.0_f64 * x[n0])
                    - dphipdr_n / (3.0_f64 * gr_n);
                yy1[n0] = (j_n * y[y_base] / x[n0]) + t2_n;
                yy2[n0] = j_n * y[y_base + 1] * g * data[0] * sig
                    / (2.0_f64 * gr_n * r2)
                    + j_n * y[y_base + 2] / x[n0]
                    + t2_n;
            }
        }
    }

    if x[0] == 0.0_f64 && nn >= 2 {
        yy1[0] = yy1[1];
        yy2[0] = yy2[1];
    }

    let mut iordp = 0_i32;
    let mut iordg = 0_i32;
    for k in 1..nn {
        if k == 1 || (k - 2) % iasn == 0 {
            let k0 = k - 1;
            let k1 = k0 + iasn;
            let phx1 = yy1[k0];
            let phy1 = yy2[k0];
            let phx2 = yy1[k1];
            let phy2 = yy2[k1];
            let dfyy = phx2 * phy1 - phx1 * phy2;
            if phx1 * phx2 <= 0.0_f64 && dfyy < 0.0_f64 {
                iordp += 1;
            } else if phx1 * phx2 <= 0.0_f64 && dfyy > 0.0_f64 {
                iordg += 1;
            }
        }
    }

    Ok(if iordp >= iordg {
        iordp - iordg + 1
    } else {
        iordp - iordg
    })
}

/// Compute mode radial order `iord` using Scuflaire, Lee, or Takata
/// classification, matching `order` (`adipls/adipack.c/adipls/order.d.f:1-293`).
///
/// # Errors
/// Propagates [`AdiplsError`] from [`takata`].
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp
)]
pub fn order(
    x: &[f64],
    y: &[f64],
    data: &[f64; 8],
    aa: &AmdlAa,
    el: f64,
    sig: f64,
    icow: i32,
    irsord: i32,
    iy: usize,
    iord: &mut i32,
    nn: usize,
    iasn: usize,
    mdintg: i32,
    eprufl: f64,
    cgrav: f64,
) -> Result<(), AdiplsError> {
    if mdintg == 2 {
        if (1..=10).contains(&irsord)
            && el == 1.0_f64
            && icow <= 1
            && (0..=irsord).contains(iord)
        {
            *iord += 1;
        }
        return Ok(());
    }

    if irsord == 20 && (el - 1.0_f64).abs() <= 1.0e-10_f64 {
        *iord = takata(x, y, nn, iasn, data, aa, iy, sig, cgrav)?;
        return Ok(());
    }

    let irsor1 = if ((el - 1.0_f64).abs() > 1.0e-10_f64
        && (!(-11..=11).contains(&irsord)))
        || icow != 0
    {
        0
    } else {
        irsord
    };

    let nn0 = nn - 1;
    let phxmin = eprufl * y[nn0 * iy].abs().max(y[nn0 * iy + 2].abs());

    let mut ns = if x[0] == 0.0_f64 { 2_usize } else { 1_usize };
    let nsr = if data[6] < 0.0_f64 { nn } else { nn - 1 };

    if (nsr - ns) % iasn != 0 {
        return Ok(());
    }

    while ns <= nsr && y[(ns - 1) * iy].abs() <= phxmin {
        ns += iasn;
    }
    if ns > nsr {
        *iord = 0;
        return Ok(());
    }

    let (ncase, mut iphx) = if irsor1 < -10 {
        (2_usize, 2_i32)
    } else if irsor1 > 10 {
        (1_usize, 2_i32)
    } else {
        (1_usize, 1_i32)
    };

    let mut iordst = [0_i32; 2];
    for slot in iordst.iter_mut().take(ncase) {
        let ns_start = ns;
        let base_s = (ns_start - 1) * iy;
        let mut phx2 = if iphx == 1 {
            y[base_s]
        } else {
            y[base_s] - y[base_s + 2]
        };
        // Upstream `order.d.f:159-164, 193-198` assigns `y(2, n)` in both
        // `iphy == 1` and `iphy != 1` branches.
        let mut phy2 = y[base_s + 1];

        let mut iordp = i32::from(phx2 * phy2 < 0.0_f64 && irsor1 <= 10);
        let mut iordg = 0_i32;

        let mut n = ns_start + iasn;
        while n <= nsr {
            let phx1 = phx2;
            let phy1 = phy2;
            let base = (n - 1) * iy;
            phx2 = if iphx == 1 {
                y[base]
            } else {
                y[base] - y[base + 2]
            };
            phy2 = y[base + 1];

            if phx2.abs() >= phxmin && phx1 * phx2 <= 0.0_f64 {
                let phy2m = (phx1 * phy2 - phx2 * phy1) / (phx1 - phx2);
                let dfyy = phy2m * (phx2 - phx1);
                if dfyy < 0.0_f64 {
                    iordp += 1;
                } else if dfyy > 0.0_f64 {
                    iordg += 1;
                }
            }
            n += iasn;
        }

        *slot = if iphx == 2 && iordp >= iordg {
            iordp - iordg + 1
        } else {
            iordp - iordg
        };
        iphx = 1;
    }

    *iord = iordst[0];
    if (1..=10).contains(&irsor1)
        && el == 1.0_f64
        && icow <= 1
        && (0..=irsor1).contains(iord)
    {
        *iord += 1;
    }

    Ok(())
}

/// Step trial squared frequency $\sigma^2$, matching `signew`
/// (`adipls/adipack.c/adipls/signew.n.d.f:1-371`) for `itrsig == 1` (`|itrsig| >= 2` external
/// summary files fail loudly).
///
/// # Errors
/// Returns [`AdiplsError::UnsupportedControl`] if `|itrsig| >= 2`.
#[allow(clippy::similar_names)]
pub fn signew(
    itrsig: i32,
    nsig: i32,
    sigp: f64,
    dfsig: f64,
    sgtrw1: f64,
    sgtrw2: f64,
    icry: &mut i32,
) -> Result<f64, AdiplsError> {
    let itrsga = itrsig.abs();
    *icry = 1;

    if itrsga != 1 {
        return Err(AdiplsError::UnsupportedControl {
            control: "itrsig",
            message: format!(
                "signew external dataset trial mode itrsig={itrsig} is unported in Package A2a"
            ),
        });
    }

    let sign = if nsig == 1 {
        sigp + dfsig
    } else if nsig == 2 || nsig == 3 {
        let isig = if sigp < 0.0_f64 { -1.0_f64 } else { 1.0_f64 };
        let mut s = (isig * sigp).sqrt();
        if nsig == 2 {
            s += dfsig;
        } else {
            s /= 1.0_f64 + dfsig * s;
        }
        isig * s * s
    } else {
        *icry = -1;
        return Ok(sigp);
    };

    if notwin(sgtrw1, sgtrw2, sign) {
        *icry = -1;
        Ok(sigp)
    } else {
        Ok(sign)
    }
}

/// Mixed p/g-mode frequency scan stepping function (`nsig == 4`), matching
/// `signw_pg` (`adipls/adipack.c/adipls/sigscn.c.d.f:443-480`).
#[must_use]
#[allow(clippy::similar_names)]
pub fn signw_pg(
    sig1: f64,
    sig2: f64,
    iscan: i32,
    isig: i32,
    icry: &mut i32,
    state: &mut SignwPgState,
) -> f64 {
    const SIGC: f64 = 4.0_f64;

    if isig <= 2 {
        let iscan1 = f64::from(iscan - 1);
        state.beta = (sig2.sqrt() - sig1.sqrt()
            + SIGC * (1.0_f64 / sig1.sqrt() - 1.0_f64 / sig2.sqrt()))
            / iscan1;
        state.gamma = (SIGC / sig1.sqrt() - sig1.sqrt()) / state.beta;
        state.bsigc = 4.0_f64 * SIGC / (state.beta * state.beta);
    }

    let disig = f64::from(isig - 1) - state.gamma;
    let sign =
        0.5_f64 * state.beta * (disig + (disig * disig + state.bsigc).sqrt());
    *icry = 0;
    sign * sign
}

/// Secant iteration for dimensionless squared frequency $\sigma^2$, matching
/// `sigitr` (`adipls/adipack.c/adipls/sigitr.n.d.f:1-243`).
///
/// # Errors
/// Propagates errors from [`stevft`] and [`sigint`].
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
pub fn sigitr(
    sig: &mut f64,
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    nw1: usize,
    nibc: usize,
    nn: usize,
    mdintg: i32,
    nev1: &mut usize,
    iasn: usize,
    ii: usize,
    ig: usize,
    ifrich: i32,
    nfit: &mut usize,
    istsb1: &mut i32,
    iscan: i32,
    xfit1: &mut f64,
    det: &mut f64,
    ddsig: &mut f64,
    ddsol: &mut f64,
    ds1: &mut [f64; 16],
    iord: &mut i32,
    icry: &mut i32,
    isolcv: &mut i32,
    isigcv: &mut i32,
    istatus: &mut i32,
    aa: &AmdlAa,
    ctx: &mut SolverContext<'_>,
    observer: &mut impl AdiplsTapObserver,
) -> Result<(), AdiplsError> {
    let radial = ctx.rhs.el <= sp_radial_eps();
    let fulmod = nibc == 2;

    *isolcv = 1;
    *isigcv = 1;
    *det = 0.0_f64;
    let mut detp;

    if ifrich != 2 && (iscan <= 1 || ctx.cincnt.irsevn >= 1) {
        ctx.rhs.els = ctx.rhs.ell / *sig;
        ctx.rhs.sig = *sig;
        stevft(
            x, nn, nibc, nfit, iasn, radial, fulmod, nev1, xfit1, aa, ctx.rhs,
            ctx.cincnt,
        )?;
    }

    let mut initnr = 1_i32;
    let mut isig = -1_i32;
    let mut dsig = 0.0_f64;

    loop {
        isig += 1;
        detp = *det;
        ctx.rhs.els = ctx.rhs.ell / *sig;
        ctx.rhs.sig = *sig;

        sigint(
            x,
            y,
            iy,
            nw1,
            nibc,
            nn,
            mdintg,
            *nev1,
            iasn,
            ii,
            ig,
            &mut initnr,
            *nfit,
            istsb1,
            det,
            ddsol,
            ds1,
            iord,
            icry,
            aa,
            ctx,
            observer,
        )?;

        if *icry < 0 {
            return Ok(());
        }

        if ctx.cincnt.itmax == 0 {
            if *ddsol >= ctx.cincnt.epssol {
                *isolcv = -1;
                *isigcv = -1;
            }
            break;
        }

        if isig == 0 {
            let sigp = *sig;
            dsig = ctx.cincnt.fsig * sigp;
            *sig = sigp + dsig;
            continue;
        }

        if (detp - *det).abs() <= ctx.cprcns.epsprc * detp.abs() {
            dsig = 0.0_f64;
            *isolcv = -1;
            *isigcv = -1;
        } else {
            dsig = dsig * *det / (detp - *det);
        }

        *ddsig = (dsig / *sig).abs();
        if *ddsig >= ctx.cincnt.dsigmx {
            dsig = ctx.cincnt.dsigmx.abs().copysign(dsig) * sig.abs();
            *ddsig = ctx.cincnt.dsigmx;
        }
        *sig += dsig;

        if *ddsig > ctx.cincnt.eps {
            if isig < ctx.cincnt.itmax {
                continue;
            }
            *isigcv = -1;
            *isolcv = -1;
            *istatus = 1;
            break;
        }

        if *ddsol < ctx.cincnt.epssol {
            break;
        }

        if *ddsig > ctx.cprcns.epsprc && isig < ctx.cincnt.itmax {
            continue;
        }

        *isolcv = -1;
        *istatus = 2;
        break;
    }

    Ok(())
}

/// Iteration driver with fitting-point retries (`rsnfit`), solution matching
/// (`mchsol`), and evanescent interior rescaling (`sclasl`), matching `sigitc`
/// (`adipls/adipack.c/adipls/sigitc.c.d.f:1-114`).
///
/// Emits [`AdiplsTapEvent::Itc`] on exit.
///
/// # Errors
/// Propagates errors from [`sigitr`] and [`mchsol`].
#[allow(clippy::too_many_arguments)]
pub fn sigitc(
    sig: &mut f64,
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    nw1: usize,
    nibc: usize,
    nn: usize,
    mdintg: i32,
    nev1: &mut usize,
    iasn: usize,
    ii: usize,
    ig: usize,
    ifrich: i32,
    nfit: &mut usize,
    istsb1: &mut i32,
    iscan: i32,
    xfit1: &mut f64,
    ddsig: &mut f64,
    ddsol: &mut f64,
    iord: &mut i32,
    icry: &mut i32,
    isolcv: &mut i32,
    isigcv: &mut i32,
    istatus: &mut i32,
    aa: &AmdlAa,
    ctx: &mut SolverContext<'_>,
    observer: &mut impl AdiplsTapObserver,
) -> Result<(), AdiplsError> {
    let radial = ctx.rhs.el <= sp_radial_eps();
    let nftrcs = ctx.cincnt.nftmax / 10;
    let nftmx1 = ctx.cincnt.nftmax % 10;

    let mut nftcnt = 0_i32;
    let nfitst = *nfit;
    let sigst = *sig;
    *xfit1 = x[nfitst - 1];
    let mut ds1 = [0.0_f64; 16];
    let mut det = 0.0_f64;

    loop {
        sigitr(
            sig, x, y, iy, nw1, nibc, nn, mdintg, nev1, iasn, ii, ig, ifrich,
            nfit, istsb1, iscan, xfit1, &mut det, ddsig, ddsol, &mut ds1, iord,
            icry, isolcv, isigcv, istatus, aa, ctx, observer,
        )?;

        if *isigcv == 1 && *isolcv == 1 {
            break;
        }
        if nftcnt <= nftmx1 {
            rsnfit(x, nibc, nn, nfit, xfit1, &mut nftcnt, nftrcs);
            *sig = sigst;
            *istatus = 3;
            continue;
        }
        *istatus = if *isigcv == -1 { 1 } else { 2 };
        break;
    }

    if mdintg != 3 {
        let moddet = ctx.cincnt.moddet;
        let imstsl = ctx.cincnt.imstsl;
        mchsol(
            y,
            iy,
            nw1,
            nn,
            ii,
            ig,
            *nfit,
            &mut ctx.cincnt.imissl,
            &mut ctx.cincnt.imjssl,
            imstsl,
            moddet,
            &mut det,
            &ds1,
            icry,
            observer,
        )?;
    }

    if *nev1 > nibc && !radial {
        sclasl(
            x,
            y,
            iy,
            nn,
            ii,
            *nev1,
            nw1,
            ctx.rhs.el,
            ctx.cincnt.iplneq,
            ctx.xlmult_cache,
        );
    }

    observer.on_tap(AdiplsTapEvent::Itc {
        sig: *sig,
        ddsig: *ddsig,
        ddsol: *ddsol,
        xfit1: *xfit1,
        nfit: *nfit,
        iord: *iord,
        icry: *icry,
        isolcv: *isolcv,
        isigcv: *isigcv,
        istatus: *istatus,
    });

    Ok(())
}

/// Record a convergence/diagnostic entry matching `setlog`
/// (`adipls/adipack.c/adipls/adipls.c.d.f:2095-2205`).
#[allow(clippy::too_many_arguments)]
pub fn setlog(
    istatus: i32,
    el: f64,
    iord: i32,
    sig: f64,
    ddsig: f64,
    ddsol: f64,
    xfit1: f64,
    xfitst: f64,
    perfac: f64,
    epsufl: f64,
    log: &mut Vec<AdiStatusLogEntry>,
) {
    let per = perfac / sig.abs().max(epsufl).sqrt();
    let anu = 16666.667_f64 / per;
    log.push(AdiStatusLogEntry {
        istatus,
        el,
        iord,
        per_min: per,
        nu_uhz: anu,
        sig,
        ddsig,
        ddsol,
        xfit1,
        xfitst,
    });
}

/// Solve for an eigenfrequency and eigenfunction from a trial $\sigma^2$,
/// matching `sigsol` (`adipls/adipack.c/adipls/sigsol.c.d.f:1-470`).
///
/// # Errors
/// Returns [`AdiplsError::UnsupportedControl`] when `iriche == 1` (`testri` /
/// Richardson extrapolation is unreached in MESA 26.4.1) or when `icow >= 1`
/// non-radially (`gravpo` is unreached in MESA 26.4.1), or propagates errors
/// from [`sigitc`] and [`order`].
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
pub fn sigsol(
    sig: &mut f64,
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    nw1: usize,
    nibc: usize,
    nn: usize,
    nnw: usize,
    mdintg: i32,
    nev1: &mut usize,
    nfit: &mut usize,
    istsb1: &mut i32,
    iscan: i32,
    iord: &mut i32,
    icry: &mut i32,
    isolcv: &mut i32,
    isigcv: &mut i32,
    sigtst: &mut f64,
    aa: &AmdlAa,
    ctx: &mut SolverContext<'_>,
    observer: &mut impl AdiplsTapObserver,
) -> Result<(), AdiplsError> {
    let dsigre0 = ctx.cincnt.dsigre;
    let mut istatus = 0_i32;
    let nftrcs = ctx.cincnt.nftmax / 10;
    let nftmx1 = ctx.cincnt.nftmax % 10;
    let radial = ctx.rhs.el <= sp_radial_eps();
    let mut nftcnt = 0_i32;

    let nfitst = *nfit;
    let sigst = *sig;
    let mut xfit1 = x[nfitst - 1];
    let xfitst = xfit1;
    let icowc = ctx.cincnt.icow;

    if ctx.cincnt.iriche == 1 {
        return Err(AdiplsError::UnsupportedControl {
            control: "iriche",
            message: String::from(
                "sigsol with iriche=1 (Richardson extrapolation / testri) is unported in Package A2a",
            ),
        });
    }

    let mut ddsig = 0.0_f64;
    let mut ddsol = 0.0_f64;
    let mut sigc;

    loop {
        let ifrich = 0_i32;
        let iasn = 1_usize;

        let (ii, ig) = if radial {
            if ctx.cincnt.icow == 1 {
                ctx.cincnt.icow = 0;
            }
            (2_usize, 1_usize)
        } else if ctx.cincnt.icow == 0 {
            (4_usize, 2_usize)
        } else {
            return Err(AdiplsError::UnsupportedControl {
                control: "icow",
                message: format!(
                    "sigsol with nonradial icow={} requires gravpo (Package A2b)",
                    ctx.cincnt.icow
                ),
            });
        };

        sigitc(
            sig,
            x,
            y,
            iy,
            nw1,
            nibc,
            nn,
            mdintg,
            nev1,
            iasn,
            ii,
            ig,
            ifrich,
            nfit,
            istsb1,
            iscan,
            &mut xfit1,
            &mut ddsig,
            &mut ddsol,
            iord,
            icry,
            isolcv,
            isigcv,
            &mut istatus,
            aa,
            ctx,
            observer,
        )?;

        sigc = *sig;
        let nwor1 = if ctx.rhs.el < 50.0_f64 { nw1 } else { *nev1 };
        let nnwor = nn - nwor1 + 1;
        let y_order_off = (nwor1 - 1) * iy;
        // `order` only accesses `aa` via `takata` when `mdintg != 2 && irsord == 20 && el ~= 1`.
        let aa_ord: std::borrow::Cow<'_, AmdlAa> = if nwor1 == 1
            || mdintg == 2
            || ctx.cincnt.irsord != 20
            || (ctx.rhs.el - 1.0_f64).abs() > 1.0e-10_f64
        {
            std::borrow::Cow::Borrowed(aa)
        } else {
            std::borrow::Cow::Owned(aa.slice_from(nwor1 - 1)?)
        };
        order(
            &x[nwor1 - 1..],
            &y[y_order_off..],
            &ctx.rhs.data,
            aa_ord.as_ref(),
            ctx.rhs.el,
            *sig,
            ctx.cincnt.icow,
            ctx.cincnt.irsord,
            iy,
            iord,
            nnwor,
            iasn,
            mdintg,
            ctx.cprcns.eprufl,
            ctx.cgrav,
        )?;

        *sigtst = *sig;

        if ctx.cincnt.itsord == 1 && *iord != ctx.iordtr {
            if nftcnt <= nftmx1 {
                rsnfit(x, nibc, nn, nfit, &mut xfit1, &mut nftcnt, nftrcs);
                *sig = sigst;
                istatus = 8;
                continue;
            }
            istatus = 7;
        }

        if radial {
            for n in nw1..=nn {
                let base = (n - 1) * iy;
                y[base + 2] = 0.0_f64;
                y[base + 3] = 0.0_f64;
            }
        }
        break;
    }

    // Store summary scalars `xfit1`, `sigc`, `ddsig`, `ddsol` into `csummm`
    // matching `common/csumma/` equivalence (`adipls.c.d.f:40-44`).
    ctx.csummm.set_f64(14, xfit1)?;
    ctx.csummm.set_f64(21, sigc)?;
    ctx.csummm.set_f64(28, ddsig)?;
    ctx.csummm.set_f64(29, ddsol)?;
    let _ = nnw;

    if *nfit != nfitst {
        *nfit = nfitst;
    }
    if ctx.cincnt.icow != icowc {
        ctx.cincnt.icow = icowc;
    }

    if istatus > 0 {
        setlog(
            istatus,
            ctx.rhs.el,
            *iord,
            *sig,
            ddsig,
            ddsol,
            xfit1,
            xfitst,
            ctx.rhs.perfac,
            ctx.cprcns.epsufl,
            ctx.status_log,
        );
    }

    ctx.cincnt.dsigre = dsigre0;
    Ok(())
}

/// Scan in squared dimensionless frequency $\sigma^2$ between `sig1` and `sig2`
/// and iterate on zero crossings of the matching determinant when `itrsig == 1`,
/// matching `sigscn` (`adipls/adipack.c/adipls/sigscn.c.d.f:1-442`).
///
/// # Errors
/// Returns [`AdiplsError::UnsupportedControl`] if `nsig` is outside `1..=4` or
/// propagates errors from [`stevft`], [`sigint`], [`signew`], [`sigsol`], and
/// `sigout_handler`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp
)]
pub fn sigscn(
    sig1: f64,
    sig2: f64,
    iscan: &mut i32,
    nsig: &mut i32,
    itrsig: i32,
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    nw1: usize,
    nibc: usize,
    nn: usize,
    nnw: usize,
    mdintg: i32,
    nev1: &mut usize,
    nfit: &mut usize,
    istsb1: &mut i32,
    icry: &mut i32,
    nfmscn: i32,
    sgtrw1: f64,
    sgtrw2: f64,
    aa: &AmdlAa,
    ctx: &mut SolverContext<'_>,
    observer: &mut impl AdiplsTapObserver,
    sigout: &mut impl SigoutHandler,
) -> Result<(), AdiplsError> {
    let iscan_orig = *iscan;
    let nsig_orig = *nsig;
    let mut ifine = 0_i32;

    *istsb1 = ctx.bcs.istsbc;
    let radial = ctx.rhs.el <= sp_radial_eps();
    let fulmod = nibc == 2;
    let iasn = if ctx.cincnt.iriche == 1 {
        2_usize
    } else {
        1_usize
    };

    let (ii, ig) = if radial || ctx.cincnt.icow > 0 {
        (2_usize, 1_usize)
    } else {
        (4_usize, 2_usize)
    };

    ctx.rhs.els = ctx.rhs.ell / sig1;
    ctx.rhs.sig = sig1;
    let mut xfit1 = x[*nfit - 1];
    stevft(
        x, nn, nibc, nfit, iasn, radial, fulmod, nev1, &mut xfit1, aa, ctx.rhs,
        ctx.cincnt,
    )?;

    let mut isig = 0_i32;
    let mut isigfn = 1_i32;

    let iscan1 = f64::from(*iscan - 1);
    let mut dsigs = match *nsig {
        1 => (sig2 - sig1) / iscan1,
        2 => (sig2.sqrt() - sig1.sqrt()) / iscan1,
        3 => (1.0_f64 / sig2.sqrt() - 1.0_f64 / sig1.sqrt()) / iscan1,
        4 => 0.0_f64,
        other => {
            return Err(AdiplsError::UnsupportedControl {
                control: "nsig",
                message: format!("sigscn nsig={other} not allowed"),
            });
        }
    };

    let mut det = 0.0_f64;
    let mut detp = 0.0_f64;
    let mut detpp;
    let mut sigp = 0.0_f64;
    let mut sigpp = 0.0_f64;
    let mut istsbp = ctx.bcs.istsbc;
    let mut istspp;

    let mut istop = 0_i32;
    let mut initnr = 1_i32;
    let mut sig = sig1;

    // Saved coarse-scan state during fine scan (`ifine == 1`).
    let mut isigst = 0_i32;
    let mut sigfst = 0.0_f64;
    let mut sigpst = 0.0_f64;
    let mut detst = 0.0_f64;
    let mut detpst = 0.0_f64;
    let mut iscans = 0_i32;
    let mut nsigst = 0_i32;
    let mut dsigst = 0.0_f64;

    let mut ddsol = 0.0_f64;
    let mut ds1 = [0.0_f64; 16];
    let mut iord = 0_i32;

    loop {
        if istop > 0 {
            *iscan = iscan_orig;
            *nsig = nsig_orig;
            return Ok(());
        }

        isig += 1;
        if isig >= 2 {
            sigpp = sigp;
            sigp = sig;
            sig = if *nsig == 4 {
                signw_pg(sig1, sig2, *iscan, isig, icry, ctx.signw_pg_state)
            } else {
                signew(1, *nsig, sigp, dsigs, sgtrw1, sgtrw2, icry)?
            };
        }

        ctx.rhs.els = ctx.rhs.ell / sig;
        ctx.rhs.sig = sig;
        detpp = detp;
        detp = det;
        istspp = istsbp;
        istsbp = *istsb1;

        sigint(
            x,
            y,
            iy,
            nw1,
            nibc,
            nn,
            mdintg,
            *nev1,
            iasn,
            ii,
            ig,
            &mut initnr,
            *nfit,
            istsb1,
            &mut det,
            &mut ddsol,
            &mut ds1,
            &mut iord,
            icry,
            aa,
            ctx,
            observer,
        )?;

        if *icry < 0 {
            return Ok(());
        }

        if ctx.cincnt.inomd1 >= 2 && *ctx.kdgbcs < 0 {
            istop = 1;
        }

        if itrsig == 1 && isig >= 2 {
            if isig > isigfn + 1
                && detpp * det > 0.0_f64
                && ifine == 0
                && (2.0_f64 * detp) / (detpp + det) <= f64::from(0.6_f32)
            {
                isigst = isig;
                sigfst = sig;
                sigpst = sigp;
                detst = det;
                detpst = detp;
                iscans = *iscan;
                nsigst = *nsig;
                dsigst = dsigs;
                *nsig = 1;
                *iscan = 50;
                dsigs = (sig - sigpp) / f64::from(*iscan - 1);
                isig = 1;
                isigfn = 1;
                ifine = 1;
                *istsb1 = istspp;
                sig = sigpp;
                continue;
            } else if isig == isigfn + 1 && det * detp < 0.0_f64 && ifine == 0 {
                isigst = isig;
                sigfst = sig;
                sigpst = sigp;
                detst = det;
                detpst = detp;
                iscans = *iscan;
                nsigst = *nsig;
                dsigst = dsigs;
                *nsig = 1;
                *iscan = 25;
                dsigs = (sig - sigp) / f64::from(*iscan - 1);
                isig = 1;
                isigfn = 1;
                ifine = 1;
                *istsb1 = istsbp;
                sig = sigp;
                det = detp;
                continue;
            }

            let mut ismode = 0_i32;
            let mut sigi1 = 0.0_f64;
            let mut sigi2 = 0.0_f64;
            let mut deti1 = 0.0_f64;
            let mut deti2 = 0.0_f64;
            let mut deti3 = 0.0_f64;

            if isig > isigfn + 1 && (detpp * detp < 0.0_f64 || detp == 0.0_f64)
            {
                ismode = 1;
                sigi1 = sigpp;
                sigi2 = sigp;
                deti1 = detpp;
                deti2 = detp;
                deti3 = det;
                if *istsb1 != istsbp {
                    *istsb1 = istsbp;
                }
            } else if isig == *iscan && (detp * det < 0.0_f64 || det == 0.0_f64)
            {
                ismode = 1;
                sigi1 = sig;
                sigi2 = sigp;
                deti1 = det;
                deti2 = detp;
                deti3 = detpp;
            }

            if mdintg == 3 && isig >= 3 && deti2.abs() > deti3.abs() {
                ismode = 0;
            }

            if ismode == 1 {
                let sigst = sig;
                let fsigst = ctx.cincnt.fsig;
                let mut iretry = 0_i32;
                let mut isolcv = 0_i32;
                let mut isigcv = 0_i32;
                let mut sigtst = 0.0_f64;

                loop {
                    ctx.cincnt.fsig = 0.1_f64 * (sigi2 - sigi1).abs() / sigi1;
                    if deti2 != deti1 {
                        let dsig = (sigi1 - sigi2) * deti1 / (deti2 - deti1);
                        sig = sigi1 + dsig;
                    }
                    let sig0 = sig;
                    ctx.rhs.els = ctx.rhs.ell / sig;
                    ctx.rhs.sig = sig;

                    sigsol(
                        &mut sig,
                        x,
                        y,
                        iy,
                        nw1,
                        nibc,
                        nn,
                        nnw,
                        mdintg,
                        nev1,
                        nfit,
                        istsb1,
                        *iscan,
                        &mut iord,
                        icry,
                        &mut isolcv,
                        &mut isigcv,
                        &mut sigtst,
                        aa,
                        ctx,
                        observer,
                    )?;

                    if (sigtst < sigi1.min(sigi2)
                        || sigtst > sigi1.max(sigi2)
                        || isolcv < 0)
                        && iretry <= 3
                    {
                        iretry += 1;
                        sig = sig0;
                        // Upstream `sigscn.c.d.f:350-352` updates `sig` in
                        // `common/rhsdat/` without recomputing `els = ell/sig`
                        // before the retry `sigint` call.
                        ctx.rhs.sig = sig;
                        let mut det0 = 0.0_f64;
                        sigint(
                            x,
                            y,
                            iy,
                            nw1,
                            nibc,
                            nn,
                            mdintg,
                            *nev1,
                            iasn,
                            ii,
                            ig,
                            &mut initnr,
                            *nfit,
                            istsb1,
                            &mut det0,
                            &mut ddsol,
                            &mut ds1,
                            &mut iord,
                            icry,
                            aa,
                            ctx,
                            observer,
                        )?;
                        if det0 * deti1 < 0.0_f64 {
                            deti2 = det0;
                            sigi2 = sig0;
                        } else {
                            deti1 = det0;
                            sigi1 = sig0;
                        }
                        continue;
                    }
                    break;
                }

                ctx.cincnt.fsig = fsigst;
                if *icry < 0 {
                    return Ok(());
                }

                let ii_out = if radial || ctx.cincnt.icow >= 2 { 2 } else { 4 };
                let mut args = SigoutArgs {
                    sig,
                    x,
                    y,
                    iy,
                    ii: ii_out,
                    nw1,
                    nibc,
                    nn,
                    nnw,
                    mdintg,
                    nev1: *nev1,
                    nfit: *nfit,
                    istsb1: *istsb1,
                    iord,
                    isolcv,
                    isigcv,
                    aa,
                    rhs: ctx.rhs,
                    rot: ctx.rot,
                    bcs: ctx.bcs,
                    cincnt: ctx.cincnt,
                    cprcns: ctx.cprcns,
                    csummm: ctx.csummm,
                    observer,
                };
                sigout.on_sigout(&mut args)?;

                sig = sigst;
                *istsb1 = ctx.bcs.istsbc;
                initnr = 1;
            }
        } else if nfmscn > 0 {
            let mut args = SigoutArgs {
                sig,
                x,
                y,
                iy,
                ii,
                nw1,
                nibc,
                nn,
                nnw,
                mdintg,
                nev1: *nev1,
                nfit: *nfit,
                istsb1: *istsb1,
                iord,
                isolcv: 1,
                isigcv: 1,
                aa,
                rhs: ctx.rhs,
                rot: ctx.rot,
                bcs: ctx.bcs,
                cincnt: ctx.cincnt,
                cprcns: ctx.cprcns,
                csummm: ctx.csummm,
                observer,
            };
            sigout.on_sigout(&mut args)?;
            initnr = 1;
        }

        if isig < *iscan {
            continue;
        }
        if ifine == 1 {
            ifine = 0;
            *iscan = iscans;
            isig = isigst;
            isigfn = isig;
            sig = sigfst;
            sigp = sigpst;
            det = detst;
            detp = detpst;
            dsigs = dsigst;
            *nsig = nsigst;
            continue;
        }
        return Ok(());
    }
}
