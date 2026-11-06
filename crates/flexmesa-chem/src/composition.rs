//! Composition moments ($\bar{A}, \bar{Z}, \overline{Z^2}, \overline{Z^{5/3}},
//! Y_e$), atomic mass-excess correction factor, and analytical partial
//! derivatives (`get_composition_info`, `basic_composition_info`,
//! `composition_info`).
//!
//! Supports scalar/analytic-partial evaluation (`f64`), generic forward-mode
//! dual evaluation ([`ChemScalar`] / [`flexmesa_autodiff::DualOrder1`]), and
//! multi-zone parallel evaluation via `rayon`.

use rayon::prelude::*;

use crate::nuclides::NuclideCatalog;
use crate::{ChemError, ChemScalar};

/// Minimum zone count for switching from sequential to `rayon` parallel loop.
pub const PARALLEL_ZONE_THRESHOLD: usize = 64;

/// Output of `get_composition_info` / `composition_info` for a single zone.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionResult {
    /// Hydrogen mass fraction $X_\mathrm{H} = \sum_{Z_i = 1} X_i$.
    pub xh: f64,
    /// Helium mass fraction $X_\mathrm{He} = \sum_{Z_i = 2} X_i$.
    pub xhe: f64,
    /// Metal mass fraction $Z = \max(0, \min(1, 1 - (X_\mathrm{H} +
    /// X_\mathrm{He})))$.
    pub xz: f64,
    /// Mean nucleon number $\bar{A} = (\sum_i X_i) / (\sum_i Y_i)$.
    pub abar: f64,
    /// Mean proton number $\bar{Z} = \bar{A} \sum_i Y_i Z_i$.
    pub zbar: f64,
    /// Mean squared charge $\overline{Z^2} = \bar{A} \sum_i Y_i Z_i^2$.
    pub z2bar: f64,
    /// Mean $5/3$-power charge $\overline{Z^{5/3}} = \bar{A} \sum_i Y_i
    /// Z_i^{5/3}$.
    pub z53bar: f64,
    /// Electron-to-baryon ratio $Y_e = \bar{Z} / \bar{A}$.
    pub ye: f64,
    /// Atomic mass correction $\sum_i Y_i W_i$.
    pub mass_correction: f64,
    /// Sum of baryon mass fractions $\sum_i X_i$.
    pub sumx: f64,
    /// Partial derivatives $\partial \bar{A} / \partial X_i$ (empty if
    /// `skip_partials`).
    pub dabar_dx: Vec<f64>,
    /// Partial derivatives $\partial \bar{Z} / \partial X_i$ (empty if
    /// `skip_partials`).
    pub dzbar_dx: Vec<f64>,
    /// Partial derivatives $\partial (\text{mass\_correction}) / \partial X_i$
    /// (empty if `skip_partials`).
    pub dmc_dx: Vec<f64>,
}

/// Generic composition moments over any [`ChemScalar`] (`f64` or
/// `DualOrder1<N>`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositionMoments<T: ChemScalar> {
    /// Hydrogen mass fraction $X_\mathrm{H}$.
    pub xh: T,
    /// Helium mass fraction $X_\mathrm{He}$.
    pub xhe: T,
    /// Metal mass fraction $Z$.
    pub xz: T,
    /// Mean nucleon number $\bar{A}$.
    pub abar: T,
    /// Mean charge number $\bar{Z}$.
    pub zbar: T,
    /// Mean squared charge $\overline{Z^2}$.
    pub z2bar: T,
    /// Mean $5/3$-power charge $\overline{Z^{5/3}}$.
    pub z53bar: T,
    /// Electron-to-baryon ratio $Y_e$.
    pub ye: T,
    /// Atomic mass correction $\sum_i Y_i W_i$.
    pub mass_correction: T,
    /// Sum of baryon mass fractions $\sum_i X_i$.
    pub sumx: T,
}

impl<T: ChemScalar> CompositionMoments<T> {
    /// Pack the 10 scalar moments into a fixed-length `[T; 10]` array.
    #[must_use]
    pub const fn to_array(self) -> [T; 10] {
        [
            self.xh,
            self.xhe,
            self.xz,
            self.abar,
            self.zbar,
            self.z2bar,
            self.z53bar,
            self.ye,
            self.mass_correction,
            self.sumx,
        ]
    }
}

/// Pre-extracted per-species nuclear properties $(Z_i, A_i, W_i, Z_i^{5/3})$
/// for fast multi-zone or generic [`ChemScalar`] composition moment evaluation.
#[derive(Debug, Clone)]
pub struct SpeciesProps {
    /// Proton numbers $Z_i$ (`usize`).
    pub z_int: Vec<usize>,
    /// Proton numbers $Z_i$ (`f64`).
    pub z: Vec<f64>,
    /// Baryon numbers $A_i = Z_i + N_i$ (`f64`).
    pub a: Vec<f64>,
    /// Atomic weights $W_i$ (`f64`).
    pub w: Vec<f64>,
    /// Coulomb powers $Z_i^{5/3}$ (`f64`).
    pub z53: Vec<f64>,
}

impl SpeciesProps {
    /// Extract [`SpeciesProps`] for a slice of 1-based `chem_ids`.
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `chem_ids` is empty or any ID
    /// is missing from `catalog`.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_catalog(
        catalog: &NuclideCatalog,
        chem_ids: &[usize],
    ) -> Result<Self, ChemError> {
        if chem_ids.is_empty() {
            return Err(ChemError::InvalidInput {
                context: "SpeciesProps::from_catalog",
                detail: "chem_ids must not be empty".to_string(),
            });
        }
        let n = chem_ids.len();
        let mut z_int = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        let mut z53 = Vec::with_capacity(n);
        for &cid in chem_ids {
            let iso = catalog.get_by_id(cid).ok_or_else(|| {
                ChemError::InvalidInput {
                    context: "SpeciesProps::from_catalog",
                    detail: format!("unknown chem_id {cid}"),
                }
            })?;
            z_int.push(iso.z);
            z.push(iso.z as f64);
            a.push(iso.z_plus_n as f64);
            w.push(iso.w);
            z53.push(iso.z53);
        }
        Ok(Self {
            z_int,
            z,
            a,
            w,
            z53,
        })
    }
}

/// Evaluate composition moments generically over `T: ChemScalar` (`f64` or
/// `DualOrder1<N>`) from pre-extracted [`SpeciesProps`].
///
/// # Panics
/// Panics if `x.len() != props.z.len()`.
#[must_use]
pub fn eval_composition_moments_generic<T: ChemScalar>(
    props: &SpeciesProps,
    x: &[T],
) -> CompositionMoments<T> {
    let n = props.z.len();
    assert_eq!(x.len(), n, "x length must equal props.z.len()");
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let mut xh = zero;
    let mut xhe = zero;
    let mut sumx = zero;
    let mut sumy = zero;
    let mut sum_yz = zero;
    let mut sum_yz2 = zero;
    let mut sum_yz53 = zero;
    let mut mass_correction = zero;

    for (i, &xi) in x.iter().enumerate().take(n) {
        let ai = T::from_f64(props.a[i]);
        let zi = T::from_f64(props.z[i]);
        let z2i = T::from_f64(props.z[i] * props.z[i]);
        let wi = T::from_f64(props.w[i]);
        let z53i = T::from_f64(props.z53[i]);
        let yi = xi / ai;
        if props.z_int[i] == 1 {
            xh = xh + xi;
        } else if props.z_int[i] == 2 {
            xhe = xhe + xi;
        }
        sumx = sumx + xi;
        sumy = sumy + yi;
        sum_yz = sum_yz + yi * zi;
        sum_yz2 = sum_yz2 + yi * z2i;
        sum_yz53 = sum_yz53 + yi * z53i;
        mass_correction = mass_correction + yi * wi;
    }

    let raw_xz = one - (xh + xhe);
    let clamped_hi = if raw_xz.val() < 1.0 { raw_xz } else { one };
    let xz = if clamped_hi.val() > 0.0 {
        clamped_hi
    } else {
        zero
    };
    let abar = sumx / sumy;
    let zbar = sum_yz * abar;
    let z2bar = sum_yz2 * abar;
    let z53bar = sum_yz53 * abar;
    let ye = zbar / abar;

    CompositionMoments {
        xh,
        xhe,
        xz,
        abar,
        zbar,
        z2bar,
        z53bar,
        ye,
        mass_correction,
        sumx,
    }
}

/// Core single-zone composition info kernel operating on pre-extracted
/// [`SpeciesProps`].
#[must_use]
pub fn get_composition_info_from_props(
    props: &SpeciesProps,
    x: &[f64],
    skip_partials: bool,
) -> CompositionResult {
    let m = eval_composition_moments_generic::<f64>(props, x);
    let n = props.z.len();
    let (dabar_dx, dzbar_dx, dmc_dx) = if skip_partials {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        let mut da = Vec::with_capacity(n);
        let mut dz = Vec::with_capacity(n);
        let mut dmc = Vec::with_capacity(n);
        for i in 0..n {
            let ai = props.a[i];
            let zi = props.z[i];
            let wi = props.w[i];
            da.push(m.abar * (ai - m.abar) / ai / m.sumx);
            dz.push(m.abar * (zi - m.zbar) / ai / m.sumx);
            dmc.push(wi / ai - m.mass_correction);
        }
        (da, dz, dmc)
    };
    CompositionResult {
        xh: m.xh,
        xhe: m.xhe,
        xz: m.xz,
        abar: m.abar,
        zbar: m.zbar,
        z2bar: m.z2bar,
        z53bar: m.z53bar,
        ye: m.ye,
        mass_correction: m.mass_correction,
        sumx: m.sumx,
        dabar_dx,
        dzbar_dx,
        dmc_dx,
    }
}

/// Compute composition moments and optional analytical partial derivatives
/// (`get_composition_info` in `chem_lib.f90`).
///
/// # Errors
/// Returns [`ChemError::InvalidInput`] if `chem_ids` is empty, `x.len() !=
/// chem_ids.len()`, or any `chem_id` is unknown.
pub fn get_composition_info(
    catalog: &NuclideCatalog,
    chem_ids: &[usize],
    x: &[f64],
    skip_partials: bool,
) -> Result<CompositionResult, ChemError> {
    if x.len() != chem_ids.len() {
        return Err(ChemError::InvalidInput {
            context: "get_composition_info",
            detail: format!(
                "x length ({}) != chem_ids length ({})",
                x.len(),
                chem_ids.len()
            ),
        });
    }
    let props = SpeciesProps::from_catalog(catalog, chem_ids)?;
    Ok(get_composition_info_from_props(&props, x, skip_partials))
}

/// Evaluate `get_composition_info` across `n_zones` stellar zones (`x_zones`
/// flattened row-major of shape `(n_zones, n_species)`), parallelizing over
/// `rayon` when `n_zones >= 64`.
///
/// # Errors
/// Returns [`ChemError::InvalidInput`] if `x_zones.len() != n_zones *
/// chem_ids.len()` or any `chem_id` is unknown.
pub fn get_composition_info_zones(
    catalog: &NuclideCatalog,
    chem_ids: &[usize],
    x_zones: &[f64],
    n_zones: usize,
    skip_partials: bool,
) -> Result<Vec<CompositionResult>, ChemError> {
    let n_species = chem_ids.len();
    if x_zones.len() != n_zones * n_species {
        return Err(ChemError::InvalidInput {
            context: "get_composition_info_zones",
            detail: format!(
                "x_zones length ({}) != n_zones ({n_zones}) * n_species \
                 ({n_species})",
                x_zones.len()
            ),
        });
    }
    let props = SpeciesProps::from_catalog(catalog, chem_ids)?;
    if n_zones >= PARALLEL_ZONE_THRESHOLD {
        Ok(x_zones
            .par_chunks_exact(n_species)
            .map(|row| {
                get_composition_info_from_props(&props, row, skip_partials)
            })
            .collect())
    } else {
        Ok(x_zones
            .chunks_exact(n_species)
            .map(|row| {
                get_composition_info_from_props(&props, row, skip_partials)
            })
            .collect())
    }
}
