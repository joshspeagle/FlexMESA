//! Solar elemental abundance scales (`AG89` through `AAG21`), dominant-isotope
//! atomic weights, CNO metal fractions, and `[M/H]` (`chem_M_div_h`).
//!
//! Matches `init_element_atomic_weights`, `init_*_data` (`chem_def.f90`), and
//! `chem_M_div_h` (`chem_lib.f90`).

use flexmesa_math::{exp10, log10};

use crate::elements::{MAIN_ISO_NAME, NUM_CHEM_ELEMENTS};
use crate::nuclides::NuclideCatalog;
use crate::solar_scales_data::{
    A09_PRZ_EPS, AAG21_EPS, AG89_RAW, AGS05_EPS, AGSS09_EPS, GN93_EPS,
    GS98_EPS, L03_EPS, L09_RAW, MB22_EPS,
};
use crate::ChemError;

/// Enumeration of the 10 solar elemental abundance scales in `chem_def.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SolarScale {
    /// Anders & Grevesse (1989) meteoritic + photospheric scale (`1`).
    Ag89 = 1,
    /// Grevesse & Noels (1993) scale (`2`).
    Gn93 = 2,
    /// Grevesse & Sauval (1998, Table 1) scale (`3`).
    Gs98 = 3,
    /// Lodders (2003, ApJ 591, 1220, Table 1) recommended scale (`4`).
    L03 = 4,
    /// Asplund, Grevesse & Sauval (2005) scale (`5`).
    Ags05 = 5,
    /// Asplund, Grevesse, Sauval & Scott (2009) scale (`6`).
    Agss09 = 6,
    /// Lodders (2009) scale (`7`).
    L09 = 7,
    /// Asplund et al. (2009) + Przybilla (2008) cosmic B-star scale (`8`).
    A09Prz = 8,
    /// Magg et al. (2022) photospheric scale (`9`).
    Mb22Photospheric = 9,
    /// Asplund, Amarsi & Grevesse (2021) photospheric scale (`10`).
    Aag21Photospheric = 10,
}

impl SolarScale {
    /// Parse a 1-based integer choice (`1..=10`) matching `*_zfracs`.
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `id` is outside `1..=10`.
    pub fn from_id(id: usize) -> Result<Self, ChemError> {
        match id {
            1 => Ok(Self::Ag89),
            2 => Ok(Self::Gn93),
            3 => Ok(Self::Gs98),
            4 => Ok(Self::L03),
            5 => Ok(Self::Ags05),
            6 => Ok(Self::Agss09),
            7 => Ok(Self::L09),
            8 => Ok(Self::A09Prz),
            9 => Ok(Self::Mb22Photospheric),
            10 => Ok(Self::Aag21Photospheric),
            _ => Err(ChemError::InvalidInput {
                context: "SolarScale::from_id",
                detail: format!(
                    "unknown zfrac_choice id {id} (expected 1..=10)"
                ),
            }),
        }
    }

    /// Parse a case-insensitive scale name (`"AG89"`, `"GS98"`, `"AGSS09"`).
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `name` is unrecognized.
    pub fn from_name(name: &str) -> Result<Self, ChemError> {
        match name.trim().to_ascii_uppercase().as_str() {
            "AG89" => Ok(Self::Ag89),
            "GN93" => Ok(Self::Gn93),
            "GS98" => Ok(Self::Gs98),
            "L03" => Ok(Self::L03),
            "AGS05" => Ok(Self::Ags05),
            "AGSS09" => Ok(Self::Agss09),
            "L09" => Ok(Self::L09),
            "A09_PRZ" | "A09PRZ" => Ok(Self::A09Prz),
            "MB22" | "MB22_PHOTOSPHERIC" => Ok(Self::Mb22Photospheric),
            "AAG21" | "AAG21_PHOTOSPHERIC" => Ok(Self::Aag21Photospheric),
            other => Err(ChemError::InvalidInput {
                context: "SolarScale::from_name",
                detail: format!("unknown solar scale '{other}'"),
            }),
        }
    }

    /// Return the bulk solar `(Y_sol, Z_sol)` mass fractions (`chem_def.f90`).
    #[must_use]
    pub const fn bulk_yz(self) -> (f64, f64) {
        match self {
            Self::Ag89 => (0.2485, 0.0201),
            Self::Gn93 => (0.2485, 0.0179),
            Self::Gs98 | Self::Mb22Photospheric => (0.2485, 0.0169),
            Self::L03 => (0.2377, 0.0133),
            Self::Ags05 => (0.2485, 0.0122),
            Self::Agss09 => (0.2485, 0.0134),
            Self::L09 => (0.2751, 0.0154),
            Self::A09Prz => (0.2760, 0.0140),
            Self::Aag21Photospheric => (0.2485, 0.0139),
        }
    }
}

/// Compute `element_atomic_weight(1..=112)` as the double-precision atomic
/// weight $W_i$ of each element's dominant isotope (`MAIN_ISO_NAME[z]`).
#[must_use]
pub fn element_atomic_weights(
    catalog: &NuclideCatalog,
) -> [f64; NUM_CHEM_ELEMENTS] {
    let mut weights = [0.0_f64; NUM_CHEM_ELEMENTS];
    for z in 1..=NUM_CHEM_ELEMENTS {
        if let Some(iso) = catalog.get_by_name(MAIN_ISO_NAME[z]) {
            weights[z - 1] = iso.w;
        }
    }
    weights
}

/// Compute the 112-element solar metal mass-fraction array `*_element_zfrac`
/// for `scale`, matching `init_*_data` in `chem_def.f90`.
///
/// Note that for spectroscopic/meteoritic $\log_{10}\epsilon$ tables (`GN93`
/// through `AAG21`), elements $Z \in \{1, 2\}$ and $Z \in [93, 112]$ remain at
/// the sentinel `-20.0`, while elements $Z \in [3, 92]$ are normalized to sum
/// to `1.0`.
#[must_use]
pub fn solar_element_zfrac(
    scale: SolarScale,
    catalog: &NuclideCatalog,
) -> [f64; NUM_CHEM_ELEMENTS] {
    match scale {
        SolarScale::Ag89 => {
            let mut out = AG89_RAW;
            let z_sum: f64 = out.iter().sum();
            for v in &mut out {
                *v /= z_sum;
            }
            out
        }
        SolarScale::L09 => {
            let mut out = L09_RAW;
            let z_sum: f64 = out[2..92].iter().sum();
            for v in &mut out[2..92] {
                *v /= z_sum;
            }
            out
        }
        other => {
            let eps = match other {
                SolarScale::Gn93 => &GN93_EPS,
                SolarScale::Gs98 => &GS98_EPS,
                SolarScale::L03 => &L03_EPS,
                SolarScale::Ags05 => &AGS05_EPS,
                SolarScale::Agss09 => &AGSS09_EPS,
                SolarScale::A09Prz => &A09_PRZ_EPS,
                SolarScale::Mb22Photospheric => &MB22_EPS,
                SolarScale::Aag21Photospheric => &AAG21_EPS,
                SolarScale::Ag89 | SolarScale::L09 => unreachable!(),
            };
            let weights = element_atomic_weights(catalog);
            let mut out = *eps;
            let mut z_sum = 0.0_f64;
            for idx in 2..92 {
                out[idx] = exp10(out[idx]) * weights[idx];
                z_sum += out[idx];
            }
            for item in &mut out[2..92] {
                *item /= z_sum;
            }
            out
        }
    }
}

/// Return the CNO mass fraction of total metallicity
/// $(Z_\mathrm{C} + Z_\mathrm{N} + Z_\mathrm{O}) / Z$ for `scale`.
#[must_use]
pub fn solar_cno_fraction(scale: SolarScale, catalog: &NuclideCatalog) -> f64 {
    let zfrac = solar_element_zfrac(scale, catalog);
    zfrac[5] + zfrac[6] + zfrac[7]
}

/// Compute the logarithmic metallicity `[M/H]` (`chem_M_div_h`):
///
/// $$[\mathrm{M/H}] = \log_{10}(Z / X)
///   - \log_{10}\!\left(\frac{Z_\odot}{1 - Z_\odot - Y_\odot}\right).$$
///
/// # Errors
/// Returns [`ChemError::InvalidInput`] if $X \le 0$ or $Z \le 0$.
pub fn chem_m_div_h(
    x: f64,
    z: f64,
    scale: SolarScale,
) -> Result<f64, ChemError> {
    if x <= 0.0 || z <= 0.0 {
        return Err(ChemError::InvalidInput {
            context: "chem_m_div_h",
            detail: format!("x ({x}) and z ({z}) must be strictly positive"),
        });
    }
    let (ysol, zsol) = scale.bulk_yz();
    let xsol = 1.0 - zsol - ysol;
    Ok(log10(z / x) - log10(zsol / xsol))
}
