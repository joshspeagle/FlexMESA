// SPDX-License-Identifier: LGPL-3.0-or-later
//! `HelmTable` data structure, grid coordinate arrays (`d`, `t`, differences,
//! and reciprocals), full `2701 x 1001` table loader, and 4-corner cell
//! stencil extraction (`eos/private/helm_alloc.f90`).

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::OnceLock;

use crate::helm::table_io::{
    load_or_build_full_helm_vertices, parse_helm_ascii_reader,
};

#[allow(clippy::float_cmp, clippy::cast_possible_truncation)]
fn helm_exp10(x: f64) -> f64 {
    if x.is_finite() && x == x.floor() && (-308.0..=308.0).contains(&x) {
        let ix = x as i32;
        if ix < 0 {
            1.0_f64 / 10.0_f64.powi(-ix)
        } else {
            10.0_f64.powi(ix)
        }
    } else {
        (x * std::f64::consts::LN_10).exp()
    }
}

/// Number of electron density grid points (`log10(rho * Y_e)` from `-12` to `15`).
pub const HELM_IMAX: usize = 2701;
/// Number of temperature grid points (`log10(T)` from `3` to `13`).
pub const HELM_JMAX: usize = 1001;
/// Minimum `log10(T)` of the `HELM` table (`3.0`).
pub const HELM_LOGTLO: f64 = 3.0;
/// Maximum `log10(T)` of the `HELM` table (`13.0`).
pub const HELM_LOGTHI: f64 = 13.0;
/// Minimum `log10(rho * Y_e)` of the `HELM` table (`-12.0`).
pub const HELM_LOGDLO: f64 = -12.0;
/// Maximum `log10(rho * Y_e)` of the `HELM` table (`15.0`).
pub const HELM_LOGDHI: f64 = 15.0;
/// Grid spacing in `log10(T)` (`0.01`).
pub const HELM_LOGTSTP: f64 =
    (HELM_LOGTHI - HELM_LOGTLO) / ((HELM_JMAX - 1) as f64);
/// Reciprocal grid spacing in `log10(T)` (`100.0`).
pub const HELM_LOGTSTPI: f64 = 1.0 / HELM_LOGTSTP;
/// Grid spacing in `log10(rho * Y_e)` (`0.01`).
pub const HELM_LOGDSTP: f64 =
    (HELM_LOGDHI - HELM_LOGDLO) / ((HELM_IMAX - 1) as f64);
/// Reciprocal grid spacing in `log10(rho * Y_e)` (`100.0`).
pub const HELM_LOGDSTPI: f64 = 1.0 / HELM_LOGDSTP;

/// The 21 tabulated values at a single `(i, j)` grid vertex of the `HELM` table.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HelmVertex {
    /// Free energy and its 8 partial derivatives `[f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt]`.
    pub f: [f64; 9],
    /// Pressure derivative table `[dpdf, dpdfd, dpdft, dpdfdt]`.
    pub dpdf: [f64; 4],
    /// Electron chemical potential table `[ef, efd, eft, efdt]`.
    pub ef: [f64; 4],
    /// Electron number density table `[xf, xfd, xft, xfdt]`.
    pub xf: [f64; 4],
}

impl HelmVertex {
    /// Analytical non-degenerate ideal/Fermi-Dirac electron [`HelmVertex`] at
    /// electron density `d` ($\rho Y_e$, $\mathrm{g\,cm^{-3}}$) and temperature
    /// `t` ($\mathrm{K}$), matching Timmes & Swesty (2000) `helm_table.dat` to
    /// 16 significant digits in the non-degenerate regime.
    #[inline]
    #[must_use]
    pub fn analytic(d: f64, t: f64) -> Self {
        const R_HELM: f64 = 8.314_472_616_995_627e7;
        const NA_HELM: f64 = 6.022_141_793e23;
        const ETA0_HELM: f64 = 18.641_406_341_775_237;

        let d_s = d.max(1.0e-30);
        let t_s = t.max(1.0);
        let eta_id = ETA0_HELM + d_s.ln() - 1.5 * t_s.ln();
        let rt = R_HELM * t_s;
        let inv_d = 1.0 / d_s;
        let inv_d2 = inv_d * inv_d;
        let inv_t = 1.0 / t_s;

        let f0 = rt * (eta_id - 1.0);
        let fd = rt * inv_d;
        let ft = R_HELM * (eta_id - 2.5);
        let fdd = -rt * inv_d2;
        let ftt = -1.5 * R_HELM * inv_t;
        let fdt = R_HELM * inv_d;
        let fddt = -R_HELM * inv_d2;

        Self {
            f: [f0, fd, ft, fdd, ftt, fdt, fddt, 0.0, 0.0],
            dpdf: [rt, 0.0, R_HELM, 0.0],
            ef: [eta_id, inv_d, -1.5 * inv_t, 0.0],
            xf: [NA_HELM * d_s, NA_HELM, 0.0, 0.0],
        }
    }
}

/// Extracted 4-corner stencil (`fi` in `helm.f90`) and cell geometry for one `(iat, jat)` cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmCell {
    /// 1-based electron density cell index (`1..HELM_IMAX`).
    pub iat: usize,
    /// 1-based temperature cell index (`1..HELM_JMAX`).
    pub jat: usize,
    /// Lower density bound `d[iat]`, upper bound `d[iat+1]`, and step widths.
    pub d_lo: f64,
    /// Upper density bound `d[iat+1]`.
    pub d_hi: f64,
    /// Density cell width `dd_sav[iat]`.
    pub dd: f64,
    /// Squared density cell width `dd2_sav[iat]`.
    pub dd2: f64,
    /// Reciprocal density cell width `ddi_sav[iat]`.
    pub ddi: f64,
    /// Reciprocal squared density cell width `dd2i_sav[iat]`.
    pub dd2i: f64,
    /// Reciprocal cubed density cell width `dd3i_sav[iat]`.
    pub dd3i: f64,
    /// Lower temperature bound `t[jat]`.
    pub t_lo: f64,
    /// Upper temperature bound `t[jat+1]`.
    pub t_hi: f64,
    /// Temperature cell width `dt_sav[jat]`.
    pub dth: f64,
    /// Squared temperature cell width `dt2_sav[jat]`.
    pub dt2: f64,
    /// Reciprocal temperature cell width `dti_sav[jat]`.
    pub dti: f64,
    /// Reciprocal squared temperature cell width `dt2i_sav[jat]`.
    pub dt2i: f64,
    /// Reciprocal cubed temperature cell width `dt3i_sav[jat]`.
    pub dt3i: f64,
    /// Packed 36-element biquintic free-energy stencil (`fi[0..36]` in `helm.f90`).
    pub fi_f: [f64; 36],
    /// Packed 16-element bicubic `dpdf` stencil (`fi[0..16]`).
    pub fi_dpdf: [f64; 16],
    /// Packed 16-element bicubic `ef` stencil (`fi[0..16]`).
    pub fi_ef: [f64; 16],
    /// Packed 16-element bicubic `xf` stencil (`fi[0..16]`).
    pub fi_xf: [f64; 16],
}

/// Complete Helmholtz (`HELM`) table state corresponding to `Helm_Table` in `eos_def.f90`.
#[derive(Debug, Clone)]
pub struct HelmTable {
    /// Number of density points (`2701`).
    pub imax: usize,
    /// Number of temperature points (`1001`).
    pub jmax: usize,
    /// Minimum `log10(T)` (`3.0`).
    pub logtlo: f64,
    /// Maximum `log10(T)` (`13.0`).
    pub logthi: f64,
    /// Minimum `log10(rho * Y_e)` (`-12.0`).
    pub logdlo: f64,
    /// Maximum `log10(rho * Y_e)` (`15.0`).
    pub logdhi: f64,
    /// Minimum temperature `10^logtlo`.
    pub templo: f64,
    /// Maximum temperature `10^logthi`.
    pub temphi: f64,
    /// Minimum electron density `10^logdlo`.
    pub denlo: f64,
    /// Maximum electron density `10^logdhi`.
    pub denhi: f64,
    /// Step in `log10(T)` (`0.01`).
    pub logtstp: f64,
    /// Reciprocal step in `log10(T)` (`100.0`).
    pub logtstpi: f64,
    /// Step in `log10(rho * Y_e)` (`0.01`).
    pub logdstp: f64,
    /// Reciprocal step in `log10(rho * Y_e)` (`100.0`).
    pub logdstpi: f64,
    /// Electron density grid `d[0..imax]`.
    pub d: Vec<f64>,
    /// Temperature grid `t[0..jmax]`.
    pub t: Vec<f64>,
    /// Temperature step widths `dt_sav[0..jmax-1]`.
    pub dt_sav: Vec<f64>,
    /// Squared temperature step widths `dt2_sav[0..jmax-1]`.
    pub dt2_sav: Vec<f64>,
    /// Reciprocal temperature step widths `dti_sav[0..jmax-1]`.
    pub dti_sav: Vec<f64>,
    /// Reciprocal squared temperature step widths `dt2i_sav[0..jmax-1]`.
    pub dt2i_sav: Vec<f64>,
    /// Reciprocal cubed temperature step widths `dt3i_sav[0..jmax-1]`.
    pub dt3i_sav: Vec<f64>,
    /// Density step widths `dd_sav[0..imax-1]`.
    pub dd_sav: Vec<f64>,
    /// Squared density step widths `dd2_sav[0..imax-1]`.
    pub dd2_sav: Vec<f64>,
    /// Reciprocal density step widths `ddi_sav[0..imax-1]`.
    pub ddi_sav: Vec<f64>,
    /// Reciprocal squared density step widths `dd2i_sav[0..imax-1]`.
    pub dd2i_sav: Vec<f64>,
    /// Reciprocal cubed density step widths `dd3i_sav[0..imax-1]`.
    pub dd3i_sav: Vec<f64>,
    /// Full dense vertex grid (`jmax * imax`) when loaded.
    dense: Option<Vec<HelmVertex>>,
    /// Sparse patch lookup table `(i_1based, j_1based) -> HelmVertex`.
    patches: HashMap<(u16, u16), HelmVertex>,
}

const BUILTIN_PATCHES_BIN: &[u8] =
    include_bytes!("../../data/helm_patches.bin");
static BUILTIN_TABLE: OnceLock<HelmTable> = OnceLock::new();
static SMOOTH_BUILTIN_TABLE: OnceLock<HelmTable> = OnceLock::new();

impl HelmTable {
    /// Returns a reference to the lazily initialized built-in [`HelmTable`].
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_TABLE.get_or_init(Self::load_builtin)
    }

    /// Returns a reference to the full dense [`HelmTable`] when available, or a
    /// globally $C^2$-smooth analytical [`HelmTable`] without sparse patch
    /// islands when the external table cache is absent.
    #[must_use]
    pub fn smooth_builtin() -> &'static Self {
        let b = Self::builtin();
        if b.is_full_table() {
            b
        } else {
            SMOOTH_BUILTIN_TABLE.get_or_init(Self::new_empty_grid)
        }
    }

    /// Returns `true` when all `2701 x 1001` vertices are populated in dense storage.
    #[inline]
    #[must_use]
    pub fn is_full_table(&self) -> bool {
        self.dense.is_some()
    }

    /// Returns the dense `2701 x 1001` vertex slice when the full table is loaded.
    #[inline]
    #[must_use]
    pub fn dense_vertices(&self) -> Option<&[HelmVertex]> {
        self.dense.as_deref()
    }

    /// Constructs the coordinate grids matching `alloc_helm_table` (`helm_alloc.f90` lines 171-323).
    #[must_use]
    pub fn new_empty_grid() -> Self {
        let imax = HELM_IMAX;
        let jmax = HELM_JMAX;
        let logtlo = HELM_LOGTLO;
        let logthi = HELM_LOGTHI;
        let logdlo = HELM_LOGDLO;
        let logdhi = HELM_LOGDHI;
        let templo = helm_exp10(logtlo);
        let temphi = helm_exp10(logthi);
        let denlo = helm_exp10(logdlo);
        let denhi = helm_exp10(logdhi);
        let logtstp = (logthi - logtlo) / ((jmax - 1) as f64);
        let logtstpi = 1.0 / logtstp;
        let logdstp = (logdhi - logdlo) / ((imax - 1) as f64);
        let logdstpi = 1.0 / logdstp;

        let mut d = vec![0.0; imax];
        let mut t = vec![0.0; jmax];
        for (j, tj) in t.iter_mut().enumerate() {
            *tj = helm_exp10(logtlo + (j as f64) * logtstp);
        }
        for (i, di) in d.iter_mut().enumerate() {
            *di = helm_exp10(logdlo + (i as f64) * logdstp);
        }

        let mut dt_sav = vec![0.0; jmax - 1];
        let mut dt2_sav = vec![0.0; jmax - 1];
        let mut dti_sav = vec![0.0; jmax - 1];
        let mut dt2i_sav = vec![0.0; jmax - 1];
        let mut dt3i_sav = vec![0.0; jmax - 1];
        for j in 0..(jmax - 1) {
            let dth = t[j + 1] - t[j];
            let dt2 = dth * dth;
            dt_sav[j] = dth;
            dt2_sav[j] = dt2;
            dti_sav[j] = 1.0 / dth;
            dt2i_sav[j] = 1.0 / dt2;
            dt3i_sav[j] = dti_sav[j] * dt2i_sav[j];
        }

        let mut dd_sav = vec![0.0; imax - 1];
        let mut dd2_sav = vec![0.0; imax - 1];
        let mut ddi_sav = vec![0.0; imax - 1];
        let mut dd2i_sav = vec![0.0; imax - 1];
        let mut dd3i_sav = vec![0.0; imax - 1];
        for i in 0..(imax - 1) {
            let dd = d[i + 1] - d[i];
            let dd2 = dd * dd;
            dd_sav[i] = dd;
            dd2_sav[i] = dd2;
            ddi_sav[i] = 1.0 / dd;
            dd2i_sav[i] = 1.0 / dd2;
            dd3i_sav[i] = ddi_sav[i] * dd2i_sav[i];
        }

        Self {
            imax,
            jmax,
            logtlo,
            logthi,
            logdlo,
            logdhi,
            templo,
            temphi,
            denlo,
            denhi,
            logtstp,
            logtstpi,
            logdstp,
            logdstpi,
            d,
            t,
            dt_sav,
            dt2_sav,
            dti_sav,
            dt2i_sav,
            dt3i_sav,
            dd_sav,
            dd2_sav,
            ddi_sav,
            dd2i_sav,
            dd3i_sav,
            dense: None,
            patches: HashMap::new(),
        }
    }

    fn load_builtin() -> Self {
        let mut tbl = Self::new_empty_grid();
        if let Some(dense) = load_or_build_full_helm_vertices() {
            tbl.dense = Some(dense);
            return tbl;
        }
        let raw = BUILTIN_PATCHES_BIN;
        if raw.len() >= 4 {
            let n =
                u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            tbl.patches.reserve(n);
            let stride = 4 + 21 * 8;
            for k in 0..n {
                let base = 4 + k * stride;
                if base + stride > raw.len() {
                    break;
                }
                let i = u16::from_le_bytes([raw[base], raw[base + 1]]);
                let j = u16::from_le_bytes([raw[base + 2], raw[base + 3]]);
                let read_f64 = |idx: usize| -> f64 {
                    let off = base + 4 + idx * 8;
                    f64::from_le_bytes(
                        raw[off..off + 8].try_into().unwrap_or([0; 8]),
                    )
                };
                let mut v = HelmVertex::default();
                for m in 0..9 {
                    v.f[m] = read_f64(m);
                }
                for m in 0..4 {
                    v.dpdf[m] = read_f64(9 + m);
                    v.ef[m] = read_f64(13 + m);
                    v.xf[m] = read_f64(17 + m);
                }
                tbl.patches.insert((i, j), v);
            }
        }
        tbl
    }

    /// Loads a full `helm_table.dat` ASCII file from disk.
    pub fn from_ascii_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let file =
            std::fs::File::open(path.as_ref()).map_err(|e| e.to_string())?;
        Self::from_ascii_reader(BufReader::with_capacity(1024 * 1024, file))
    }

    /// Parses the full `helm_table.dat` stream (`helm_alloc.f90` lines 250-325).
    pub fn from_ascii_reader<R: BufRead>(reader: R) -> Result<Self, String> {
        let mut tbl = Self::new_empty_grid();
        tbl.dense = Some(parse_helm_ascii_reader(reader)?);
        Ok(tbl)
    }

    /// Looks up a 1-based vertex `(i, j)` (`1 <= i <= imax`, `1 <= j <= jmax`).
    #[inline]
    #[must_use]
    pub fn vertex(&self, i: usize, j: usize) -> Option<HelmVertex> {
        if i == 0 || i > self.imax || j == 0 || j > self.jmax {
            return None;
        }
        if let Some(dense) = &self.dense {
            return Some(dense[(j - 1) * self.imax + (i - 1)]);
        }
        self.patches.get(&(i as u16, j as u16)).copied()
    }

    /// Extracts the 4-corner stencil [`HelmCell`] for `(log_temp, din)` (`helm.f90` lines 387-542).
    #[must_use]
    pub fn cell(&self, log_temp: f64, din: f64) -> HelmCell {
        let ldin = din.log10();
        let iat_raw = ((ldin - self.logdlo) * self.logdstpi) as isize + 1;
        let iat = iat_raw.clamp(1, (self.imax - 1) as isize) as usize;
        let jat_raw = ((log_temp - self.logtlo) * self.logtstpi) as isize + 1;
        let jat = jat_raw.clamp(1, (self.jmax - 1) as isize) as usize;

        let (v00, v10, v01, v11) = if let Some(dense) = &self.dense {
            let row0 = (jat - 1) * self.imax + (iat - 1);
            let row1 = row0 + self.imax;
            (dense[row0], dense[row0 + 1], dense[row1], dense[row1 + 1])
        } else if let (Some(a), Some(b), Some(c), Some(d)) = (
            self.vertex(iat, jat),
            self.vertex(iat + 1, jat),
            self.vertex(iat, jat + 1),
            self.vertex(iat + 1, jat + 1),
        ) {
            (a, b, c, d)
        } else {
            let d0 = self.d[iat - 1];
            let d1 = self.d[iat];
            let t0 = self.t[jat - 1];
            let t1 = self.t[jat];
            (
                HelmVertex::analytic(d0, t0),
                HelmVertex::analytic(d1, t0),
                HelmVertex::analytic(d0, t1),
                HelmVertex::analytic(d1, t1),
            )
        };

        let fi_f = [
            v00.f[0], v10.f[0], v01.f[0], v11.f[0], v00.f[2], v10.f[2],
            v01.f[2], v11.f[2], v00.f[4], v10.f[4], v01.f[4], v11.f[4],
            v00.f[1], v10.f[1], v01.f[1], v11.f[1], v00.f[3], v10.f[3],
            v01.f[3], v11.f[3], v00.f[5], v10.f[5], v01.f[5], v11.f[5],
            v00.f[6], v10.f[6], v01.f[6], v11.f[6], v00.f[7], v10.f[7],
            v01.f[7], v11.f[7], v00.f[8], v10.f[8], v01.f[8], v11.f[8],
        ];
        let pack16 =
            |a: &[f64; 4], b: &[f64; 4], c: &[f64; 4], d: &[f64; 4]| {
                [
                    a[0], b[0], c[0], d[0], a[2], b[2], c[2], d[2], a[1], b[1],
                    c[1], d[1], a[3], b[3], c[3], d[3],
                ]
            };

        HelmCell {
            iat,
            jat,
            d_lo: self.d[iat - 1],
            d_hi: self.d[iat],
            dd: self.dd_sav[iat - 1],
            dd2: self.dd2_sav[iat - 1],
            ddi: self.ddi_sav[iat - 1],
            dd2i: self.dd2i_sav[iat - 1],
            dd3i: self.dd3i_sav[iat - 1],
            t_lo: self.t[jat - 1],
            t_hi: self.t[jat],
            dth: self.dt_sav[jat - 1],
            dt2: self.dt2_sav[jat - 1],
            dti: self.dti_sav[jat - 1],
            dt2i: self.dt2i_sav[jat - 1],
            dt3i: self.dt3i_sav[jat - 1],
            fi_f,
            fi_dpdf: pack16(&v00.dpdf, &v10.dpdf, &v01.dpdf, &v11.dpdf),
            fi_ef: pack16(&v00.ef, &v10.ef, &v01.ef, &v11.ef),
            fi_xf: pack16(&v00.xf, &v10.xf, &v01.xf, &v11.xf),
        }
    }
}
