//! `HELM` electron-positron free-energy table (`HelmTable`) and cell lookup.
//!
//! This module ports `eos/public/helm_def.dek`, `eos/private/helm_alloc.f90`,
//! and `eos/private/skye_ideal.f90` (`compute_ideal_ele`).
//!
//! The `HELM` table spans $\log_{10}(\rho Y_e / \mathrm{g\,cm^{-3}}) \in
//! [-12, 15]$ across `IMAX = 2701` points ($\Delta \log_{10} d = 0.01$) and
//! $\log_{10}(T / \mathrm{K}) \in [3, 13]$ across `JMAX = 1001` points
//! ($\Delta \log_{10} T = 0.01$). Each vertex stores:
//! - 9 Helmholtz free-energy values (`f, fd, ft, fdd, ftt, fdt, fddt, fdtt,
//!   fddtt`) for biquintic Hermite interpolation (`h5`),
//! - 4 pressure-derivative values (`dpdf, dpdfd, dpdft, dpdfdt`) for bicubic
//!   Hermite interpolation (`h3`),
//! - 4 electron chemical-potential values (`ef, efd, eft, efdt`) for `h3`,
//! - 4 electron-positron number-density values (`xf, xfd, xft, xfdt`) for `h3`.

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;
use std::sync::OnceLock;

use flexmesa_const::{AVO, CLIGHT, KERG, ME, PI, PLANCK_H};

#[allow(clippy::float_cmp)]
fn helm_exp10(x: f64) -> f64 {
    if x.is_finite() && x == x.floor() && (-308.0..=308.0).contains(&x) {
        #[allow(clippy::cast_possible_truncation)]
        let ix = x as i32;
        let mag = 10.0_f64.powi(ix.abs());
        if ix < 0 { 1.0_f64 / mag } else { mag }
    } else {
        (x * std::f64::consts::LN_10).exp()
    }
}

/// Number of density grid points ($\rho Y_e$) in the `HELM` table.
pub const HELM_IMAX: usize = 2701;

/// Number of temperature grid points ($T$) in the `HELM` table.
pub const HELM_JMAX: usize = 1001;

/// Lower $\log_{10}(T/\mathrm{K})$ bound of the `HELM` table.
pub const HELM_LOGTLO: f64 = 3.0;

/// Upper $\log_{10}(T/\mathrm{K})$ bound of the `HELM` table.
pub const HELM_LOGTHI: f64 = 13.0;

/// Step size $\Delta \log_{10} T$ of the `HELM` table.
pub const HELM_LOGTSTP: f64 = 0.01;

/// Inverse step size $1 / \Delta \log_{10} T$ of the `HELM` table.
pub const HELM_LOGTSTPI: f64 = 100.0;

/// Lower $\log_{10}(\rho Y_e / \mathrm{g\,cm^{-3}})$ bound of the `HELM` table.
pub const HELM_LOGDLO: f64 = -12.0;

/// Upper $\log_{10}(\rho Y_e / \mathrm{g\,cm^{-3}})$ bound of the `HELM` table.
pub const HELM_LOGDHI: f64 = 15.0;

/// Step size $\Delta \log_{10}(\rho Y_e)$ of the `HELM` table.
pub const HELM_LOGDSTP: f64 = 0.01;

/// Inverse step size $1 / \Delta \log_{10}(\rho Y_e)$ of the `HELM` table.
pub const HELM_LOGDSTPI: f64 = 100.0;

const BUILTIN_PATCHES: &[u8] = include_bytes!("../../data/helm_patches.bin");

/// The 21 table values stored at a single `(i, j)` vertex of the `HELM` grid.
#[derive(Debug, Clone, Copy)]
pub struct HelmVertex {
    /// Free energy and derivatives
    /// `[f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt]`.
    pub f: [f64; 9],
    /// Pressure derivative table `[dpdf, dpdfd, dpdft, dpdfdt]`.
    pub dpdf: [f64; 4],
    /// Electron degeneracy table `[ef, efd, eft, efdt]`.
    pub ef: [f64; 4],
    /// Electron-positron number density table `[xf, xfd, xft, xfdt]`.
    pub xf: [f64; 4],
}

/// Pre-packed cell geometry and corner arrays for a single `(iat, jat)` cell.
#[derive(Debug, Clone, Copy)]
pub struct HelmCell {
    /// 1-based density cell index in `1..=HELM_IMAX - 1`.
    pub iat: usize,
    /// 1-based temperature cell index in `1..=HELM_JMAX - 1`.
    pub jat: usize,
    /// Lower cell temperature $T_{\mathrm{jat}}$ ($\mathrm{K}$).
    pub t_lo: f64,
    /// Temperature cell width $\Delta T$ ($\mathrm{K}$).
    pub dth: f64,
    /// Squared temperature cell width $(\Delta T)^2$.
    pub dt2: f64,
    /// Inverse temperature cell width $1 / \Delta T$.
    pub dti: f64,
    /// Inverse squared temperature cell width $1 / (\Delta T)^2$.
    pub dt2i: f64,
    /// Inverse cubed temperature cell width $1 / (\Delta T)^3$.
    pub dt3i: f64,
    /// Lower cell electron density $d_{\mathrm{iat}}$ ($\mathrm{g\,cm^{-3}}$).
    pub d_lo: f64,
    /// Density cell width $\Delta d$ ($\mathrm{g\,cm^{-3}}$).
    pub dd: f64,
    /// Squared density cell width $(\Delta d)^2$.
    pub dd2: f64,
    /// Inverse density cell width $1 / \Delta d$.
    pub ddi: f64,
    /// Inverse squared density cell width $1 / (\Delta d)^2$.
    pub dd2i: f64,
    /// Inverse cubed density cell width $1 / (\Delta d)^3$.
    pub dd3i: f64,
    /// 36 biquintic free-energy corner coefficients for `h5`.
    pub fi_f: [f64; 36],
    /// 16 bicubic pressure-derivative corner coefficients for `h3`.
    pub fi_dpdf: [f64; 16],
    /// 16 bicubic chemical-potential corner coefficients for `h3`.
    pub fi_ef: [f64; 16],
    /// 16 bicubic number-density corner coefficients for `h3`.
    pub fi_xf: [f64; 16],
}

/// The `HELM` electron-positron free-energy table (`Helm_Table` in MESA).
#[derive(Debug, Clone)]
pub struct HelmTable {
    /// Precomputed temperature grid $T_j = 10^{3 + 0.01(j-1)}$ (`len = 1001`).
    pub t: Vec<f64>,
    /// Precomputed density grid $d_i = 10^{-12 + 0.01(i-1)}$ (`len = 2701`).
    pub d: Vec<f64>,
    /// Embedded sparse vertex map `(i, j) -> HelmVertex` (1-based indices).
    patches: HashMap<(u16, u16), HelmVertex>,
    /// Optional full dense grid of `2701 * 1001` vertices
    /// (`(j - 1) * IMAX + (i - 1)`).
    dense: Option<Vec<HelmVertex>>,
}

static BUILTIN_TABLE: OnceLock<HelmTable> = OnceLock::new();

impl HelmTable {
    /// Returns a reference to the lazily initialized built-in `HelmTable`.
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_TABLE.get_or_init(Self::load_builtin)
    }

    fn build_axes() -> (Vec<f64>, Vec<f64>) {
        let mut t = Vec::with_capacity(HELM_JMAX);
        for j in 1..=HELM_JMAX {
            t.push(helm_exp10(HELM_LOGTLO + ((j - 1) as f64) * HELM_LOGTSTP));
        }
        let mut d = Vec::with_capacity(HELM_IMAX);
        for i in 1..=HELM_IMAX {
            d.push(helm_exp10(HELM_LOGDLO + ((i - 1) as f64) * HELM_LOGDSTP));
        }
        (t, d)
    }

    fn load_builtin() -> Self {
        let (t, d) = Self::build_axes();
        let count = u32::from_le_bytes(
            BUILTIN_PATCHES[0..4].try_into().unwrap_or([0; 4]),
        ) as usize;
        let mut patches = HashMap::with_capacity(count);
        let mut pos = 4;
        for _ in 0..count {
            let i = u16::from_le_bytes([
                BUILTIN_PATCHES[pos],
                BUILTIN_PATCHES[pos + 1],
            ]);
            let j = u16::from_le_bytes([
                BUILTIN_PATCHES[pos + 2],
                BUILTIN_PATCHES[pos + 3],
            ]);
            pos += 4;
            let mut vals = [0.0_f64; 21];
            for v in &mut vals {
                let bytes: [u8; 8] = BUILTIN_PATCHES[pos..pos + 8]
                    .try_into()
                    .unwrap_or([0; 8]);
                *v = f64::from_le_bytes(bytes);
                pos += 8;
            }
            let vertex = HelmVertex {
                f: vals[0..9].try_into().unwrap_or([0.0; 9]),
                dpdf: vals[9..13].try_into().unwrap_or([0.0; 4]),
                ef: vals[13..17].try_into().unwrap_or([0.0; 4]),
                xf: vals[17..21].try_into().unwrap_or([0.0; 4]),
            };
            patches.insert((i, j), vertex);
        }
        Self {
            t,
            d,
            patches,
            dense: None,
        }
    }

    /// Loads the full $2701 \times 1001$ `HELM` table from a MESA unformatted
    /// Fortran cache file (`helm_table.bin`).
    ///
    /// # Errors
    /// Returns `io::Error` if the file cannot be read or has invalid
    /// dimensions.
    #[allow(
        clippy::cast_sign_loss,
        clippy::chunks_exact_to_as_chunks,
        clippy::missing_panics_doc,
        clippy::needless_range_loop
    )]
    pub fn from_mesa_bin(path: &Path) -> io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let read_i32_rec = |f: &mut std::fs::File| -> io::Result<i32> {
            let mut buf = [0u8; 12];
            f.read_exact(&mut buf)?;
            Ok(i32::from_le_bytes(buf[4..8].try_into().unwrap()))
        };
        let imax = read_i32_rec(&mut file)? as usize;
        let jmax = read_i32_rec(&mut file)? as usize;
        if imax != HELM_IMAX || jmax != HELM_JMAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid helm_table.bin dimensions",
            ));
        }
        let n = imax * jmax;
        let mut planes = vec![vec![0.0_f64; n]; 21];
        let mut marker = [0u8; 4];
        let mut raw = vec![0u8; n * 8];
        for plane in &mut planes {
            file.read_exact(&mut marker)?;
            file.read_exact(&mut raw)?;
            file.read_exact(&mut marker)?;
            for (idx, chunk) in raw.chunks_exact(8).enumerate() {
                plane[idx] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        let mut dense = Vec::with_capacity(n);
        for idx in 0..n {
            dense.push(HelmVertex {
                f: std::array::from_fn(|k| planes[k][idx]),
                dpdf: std::array::from_fn(|k| planes[9 + k][idx]),
                ef: std::array::from_fn(|k| planes[13 + k][idx]),
                xf: std::array::from_fn(|k| planes[17 + k][idx]),
            });
        }
        let (t, d) = Self::build_axes();
        Ok(Self {
            t,
            d,
            patches: HashMap::new(),
            dense: Some(dense),
        })
    }

    /// Analytical ideal-Fermi fallback vertex when querying outside the
    /// embedded golden patches without a dense table file loaded.
    #[inline]
    fn analytical_vertex(&self, i: usize, j: usize) -> HelmVertex {
        let den = self.d[i - 1];
        let temp = self.t[j - 1];
        let kt = KERG * temp;
        let n_e = AVO * den;
        let lambda3 = (PLANCK_H * PLANCK_H / (2.0 * PI * ME * kt)).powf(1.5);
        let z_deg = (0.5 * n_e * lambda3).max(1.0e-99);
        let eta = z_deg.ln();
        let f_val = AVO * kt * (eta - 1.0);
        let fd = AVO * kt / den;
        let ft = AVO * KERG * (eta - 2.5);
        let fdd = -AVO * kt / (den * den);
        let ftt = -1.5 * AVO * KERG / temp;
        let fdt = AVO * KERG / den;
        let fddt = -AVO * KERG / (den * den);
        let fdtt = 0.0;
        let fddtt = 0.0;
        let p_f = (PLANCK_H / (2.0 * PI)) * (3.0 * PI * PI * n_e).cbrt();
        let e_f = p_f * p_f / (2.0 * ME)
            * (1.0 - 0.25 * (p_f / (ME * CLIGHT)).powi(2));
        let dpdf = AVO * kt + (2.0 / 3.0) * AVO * e_f.max(0.0);
        HelmVertex {
            f: [f_val, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt],
            dpdf: [dpdf, 0.0, AVO * KERG, 0.0],
            ef: [eta, 1.0 / den, -1.5 / temp, 0.0],
            xf: [n_e, AVO, 0.0, 0.0],
        }
    }

    /// Returns the `HelmVertex` at 1-based grid indices `(i, j)`.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn vertex(&self, i: usize, j: usize) -> HelmVertex {
        let i = i.clamp(1, HELM_IMAX);
        let j = j.clamp(1, HELM_JMAX);
        if let Some(ref dense) = self.dense {
            return dense[(j - 1) * HELM_IMAX + (i - 1)];
        }
        if let Some(&v) = self.patches.get(&(i as u16, j as u16)) {
            return v;
        }
        self.analytical_vertex(i, j)
    }

    /// Hash-locates the cell containing `(log_temp, din)` and packs its 4
    /// corner vertices in MESA's `helm_electron_positron.dek` order.
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub fn cell(&self, log_temp: f64, din: f64) -> HelmCell {
        let jat = (((log_temp - HELM_LOGTLO) * HELM_LOGTSTPI) as isize + 1)
            .clamp(1, (HELM_JMAX - 1) as isize) as usize;
        let iat = (((din.log10() - HELM_LOGDLO) * HELM_LOGDSTPI) as isize + 1)
            .clamp(1, (HELM_IMAX - 1) as isize) as usize;

        let t_lo = self.t[jat - 1];
        let t_hi = self.t[jat];
        let dth = t_hi - t_lo;
        let dt2 = dth * dth;
        let dti = 1.0 / dth;
        let dt2i = 1.0 / dt2;
        let dt3i = dt2i * dti;

        let d_lo = self.d[iat - 1];
        let d_hi = self.d[iat];
        let dd = d_hi - d_lo;
        let dd2 = dd * dd;
        let ddi = 1.0 / dd;
        let dd2i = 1.0 / dd2;
        let dd3i = dd2i * ddi;

        let v00 = self.vertex(iat, jat);
        let v10 = self.vertex(iat + 1, jat);
        let v01 = self.vertex(iat, jat + 1);
        let v11 = self.vertex(iat + 1, jat + 1);

        let f_cols = [0, 2, 4, 1, 3, 5, 6, 7, 8];
        let mut fi_f = [0.0; 36];
        for (blk, &col) in f_cols.iter().enumerate() {
            fi_f[4 * blk] = v00.f[col];
            fi_f[4 * blk + 1] = v10.f[col];
            fi_f[4 * blk + 2] = v01.f[col];
            fi_f[4 * blk + 3] = v11.f[col];
        }

        let c_cols = [0, 2, 1, 3];
        let mut fi_dpdf = [0.0; 16];
        let mut fi_ef = [0.0; 16];
        let mut fi_xf = [0.0; 16];
        for (blk, &col) in c_cols.iter().enumerate() {
            fi_dpdf[4 * blk] = v00.dpdf[col];
            fi_dpdf[4 * blk + 1] = v10.dpdf[col];
            fi_dpdf[4 * blk + 2] = v01.dpdf[col];
            fi_dpdf[4 * blk + 3] = v11.dpdf[col];

            fi_ef[4 * blk] = v00.ef[col];
            fi_ef[4 * blk + 1] = v10.ef[col];
            fi_ef[4 * blk + 2] = v01.ef[col];
            fi_ef[4 * blk + 3] = v11.ef[col];

            fi_xf[4 * blk] = v00.xf[col];
            fi_xf[4 * blk + 1] = v10.xf[col];
            fi_xf[4 * blk + 2] = v01.xf[col];
            fi_xf[4 * blk + 3] = v11.xf[col];
        }

        HelmCell {
            iat,
            jat,
            t_lo,
            dth,
            dt2,
            dti,
            dt2i,
            dt3i,
            d_lo,
            dd,
            dd2,
            ddi,
            dd2i,
            dd3i,
            fi_f,
            fi_dpdf,
            fi_ef,
            fi_xf,
        }
    }
}
