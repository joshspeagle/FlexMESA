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
        let t = (1..=HELM_JMAX)
            .map(|j| helm_exp10(HELM_LOGTLO + ((j - 1) as f64) * HELM_LOGTSTP))
            .collect();
        let d = (1..=HELM_IMAX)
            .map(|i| helm_exp10(HELM_LOGDLO + ((i - 1) as f64) * HELM_LOGDSTP))
            .collect();
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
            let vals: [f64; 21] = std::array::from_fn(|k| {
                let s = pos + 8 * k;
                f64::from_le_bytes(
                    BUILTIN_PATCHES[s..s + 8].try_into().unwrap_or([0; 8]),
                )
            });
            pos += 21 * 8;
            patches.insert(
                (i, j),
                HelmVertex {
                    f: vals[0..9].try_into().unwrap_or([0.0; 9]),
                    dpdf: vals[9..13].try_into().unwrap_or([0.0; 4]),
                    ef: vals[13..17].try_into().unwrap_or([0.0; 4]),
                    xf: vals[17..21].try_into().unwrap_or([0.0; 4]),
                },
            );
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
        let (imax, jmax) = (
            read_i32_rec(&mut file)? as usize,
            read_i32_rec(&mut file)? as usize,
        );
        if imax != HELM_IMAX || jmax != HELM_JMAX {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }
        let n = imax * jmax;
        let mut planes = vec![vec![0.0_f64; n]; 21];
        let (mut marker, mut raw) = ([0u8; 4], vec![0u8; n * 8]);
        for plane in &mut planes {
            file.read_exact(&mut marker)?;
            file.read_exact(&mut raw)?;
            file.read_exact(&mut marker)?;
            for (idx, chunk) in raw.chunks_exact(8).enumerate() {
                plane[idx] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        let dense = (0..n)
            .map(|idx| HelmVertex {
                f: std::array::from_fn(|k| planes[k][idx]),
                dpdf: std::array::from_fn(|k| planes[9 + k][idx]),
                ef: std::array::from_fn(|k| planes[13 + k][idx]),
                xf: std::array::from_fn(|k| planes[17 + k][idx]),
            })
            .collect();
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
        let (den, temp) = (self.d[i - 1], self.t[j - 1]);
        let (kt, k, n_e) = (KERG * temp, AVO * KERG, AVO * den);
        let lambda3 = (PLANCK_H * PLANCK_H / (2.0 * PI * ME * kt)).powf(1.5);
        let z_deg = (0.5 * n_e * lambda3).max(1.0e-99);
        let eta = z_deg.ln();
        let p_f = (PLANCK_H / (2.0 * PI)) * (3.0 * PI * PI * n_e).cbrt();
        let w = std::f64::consts::E / z_deg;
        let (p, q, ln_1pw) = if z_deg < 1.0e-8 {
            let q_small = z_deg / std::f64::consts::E;
            (1.0 - q_small, q_small, 1.0 - eta + q_small)
        } else {
            (w / (1.0 + w), 1.0 / (1.0 + w), w.ln_1p())
        };
        let x_f = p_f / (ME * CLIGHT);
        let (x2, gam, mc2) = (
            x_f * x_f,
            (1.0 + x_f * x_f).sqrt(),
            AVO * ME * CLIGHT * CLIGHT,
        );
        let (f_cold, fd_cold, fdd_cold) = if x_f < 1.0e-2 {
            let x4 = x2 * x2;
            let x6 = x4 * x2;
            let f_c = mc2 * (0.3 * x2 - (3.0 / 56.0) * x4 + (1.0 / 48.0) * x6);
            let fd_c = (mc2 / den)
                * (0.2 * x2 - (1.0 / 14.0) * x4 + (1.0 / 24.0) * x6);
            let fdd_c =
                (mc2 / (den * den)) * (-(1.0 / 15.0) * x2 + (1.0 / 42.0) * x4);
            (f_c, fd_c, fdd_c)
        } else {
            let f_c = mc2
                * (0.375 * (x_f * (2.0 * x2 + 1.0) * gam - x_f.asinh())
                    / (x2 * x_f)
                    - 1.0);
            let mu_c = mc2 * x2 / (gam + 1.0);
            let fd_c = (mu_c - f_c) / den;
            let fdd_c = (mc2 * x2 / (3.0 * gam * den) - 2.0 * fd_c) / den;
            (f_c, fd_c, fdd_c)
        };
        let f_val = f_cold - k * temp * ln_1pw;
        let fd = fd_cold + k * p * temp / den;
        let ft = -k * (1.5 * p + ln_1pw);
        let fdd = fdd_cold + k * p * temp * (p - 2.0) / (den * den);
        let ftt = 0.75 * k * p * (3.0 * p - 5.0) / temp;
        let fdt = -0.5 * k * p * (3.0 * p - 5.0) / den;
        let fddt = -k * p * (3.0 * p * p - 7.0 * p + 5.0) / (den * den);
        let fdtt = -0.75 * k * p * q * (6.0 * p - 5.0) / (den * temp);
        let fddtt = 1.5 * k * p * q * q * (9.0 * p - 5.0) / (den * den * temp);
        let dpdf_c = mc2 * x2 / (3.0 * gam);
        let dpdfd_c = dpdf_c * (2.0 + x2) / (3.0 * den * (1.0 + x2));
        let p2 = p * p;
        HelmVertex {
            f: [f_val, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt],
            dpdf: [
                dpdf_c + k * p2 * temp,
                dpdfd_c - 2.0 * k * p2 * q * temp / den,
                k * p2 * (4.0 - 3.0 * p),
                k * p2 * q * (9.0 * p - 8.0) / den,
            ],
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

        let (t_lo, dth) = (self.t[jat - 1], self.t[jat] - self.t[jat - 1]);
        let (dt2, dti) = (dth * dth, 1.0 / dth);
        let (dt2i, dt3i) = (1.0 / dt2, (1.0 / dt2) * dti);
        let (d_lo, dd) = (self.d[iat - 1], self.d[iat] - self.d[iat - 1]);
        let (dd2, ddi) = (dd * dd, 1.0 / dd);
        let (dd2i, dd3i) = (1.0 / dd2, (1.0 / dd2) * ddi);
        let c = [
            (iat, jat),
            (iat + 1, jat),
            (iat, jat + 1),
            (iat + 1, jat + 1),
        ];
        let [v00, v10, v01, v11] = if self.dense.is_some() {
            c.map(|(i, j)| self.vertex(i, j))
        } else if c
            .iter()
            .all(|&(i, j)| self.patches.contains_key(&(i as u16, j as u16)))
        {
            c.map(|(i, j)| self.patches[&(i as u16, j as u16)])
        } else {
            c.map(|(i, j)| self.analytical_vertex(i, j))
        };

        let (f_c, c_c, v) = (
            [0, 2, 4, 1, 3, 5, 6, 7, 8],
            [0, 2, 1, 3],
            [v00, v10, v01, v11],
        );
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
            fi_f: std::array::from_fn(|k| v[k % 4].f[f_c[k / 4]]),
            fi_dpdf: std::array::from_fn(|k| v[k % 4].dpdf[c_c[k / 4]]),
            fi_ef: std::array::from_fn(|k| v[k % 4].ef[c_c[k / 4]]),
            fi_xf: std::array::from_fn(|k| v[k % 4].xf[c_c[k / 4]]),
        }
    }
}
