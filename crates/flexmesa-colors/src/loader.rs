// SPDX-License-Identifier: GPL-3.0-or-later
//! Runtime path resolution, table/filter/Vega loaders, CSV SED writer, and
//! dynamic multi-filter photometry context (`colors_lib.f90`,
//! `colors_utils.f90`, `colors_history.f90`, `synthetic.f90`).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::filters_data::JOHNSON_FILTERS;
use crate::photometry::{
    compute_ab_zero_point, compute_st_zero_point, compute_vega_zero_point,
    flux_to_magnitude,
};
use crate::sed_cube::{builtin_sed_cube_arc, SedCube};
use crate::types::{ColorsError, MagSystem};
use crate::utils::{
    compute_interp_radius, interpolate_array, remove_dat,
    simpson_integration_fn,
};

const CANON_KURUCZ_ATM: &str = "data/colors_data/stellar_models/kurucz2003all";
const CANON_JOHNSON_INST: &str = "data/colors_data/filters/generic/johnson";
const CANON_GAIA_INST: &str = "data/colors_data/filters/gaia/gaia";
const CANON_VEGA_SED: &str = "data/colors_data/stellar_models/vega_flam.csv";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CubeCacheKey {
    path: PathBuf,
    byte_len: u64,
}

static FLUX_CUBE_CACHE: OnceLock<Mutex<HashMap<CubeCacheKey, Arc<SedCube>>>> =
    OnceLock::new();

/// Single photometric filter loaded from disk or built-in Johnson tables.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedFilter {
    /// Original filename from the instrument index (e.g., `"G.dat"`).
    pub raw_filename: String,
    /// Stripped filter identifier used as the history column header (e.g., `"G"`).
    pub name: String,
    /// Strictly increasing filter wavelength grid in Angstroms ($\mathrm{\AA}$).
    pub wavelengths: Vec<f64>,
    /// Dimensionless filter transmission response $S_\lambda$.
    pub transmission: Vec<f64>,
    /// Precomputed Vega zero-point flux ($\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$).
    pub vega_zero_point: f64,
    /// Precomputed AB zero-point flux ($\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$).
    pub ab_zero_point: f64,
    /// Precomputed ST zero-point flux ($\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$).
    pub st_zero_point: f64,
}

impl LoadedFilter {
    /// Returns the precomputed zero-point flux for magnitude system `sys`.
    #[inline]
    #[must_use]
    pub const fn zero_point(&self, sys: MagSystem) -> f64 {
        match sys {
            MagSystem::Vega => self.vega_zero_point,
            MagSystem::Ab => self.ab_zero_point,
            MagSystem::St => self.st_zero_point,
        }
    }
}

/// Output produced by evaluating `colors` history columns for one stellar state.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorsEvalOutput {
    /// Bolometric magnitude $m_{\mathrm{bol}}$ (`Mag_bol`).
    pub mag_bol: f64,
    /// Bolometric flux $F_{\mathrm{bol}}$ in $\mathrm{erg\,s^{-1}\,cm^{-2}}$ (`Flux_bol`).
    pub flux_bol: f64,
    /// Normalized nearest-grid distance (`Interp_rad`).
    pub interp_rad: f64,
    /// Synthetic passband magnitudes ordered identically to `filters`.
    pub filter_mags: Vec<f64>,
}

/// Loaded stellar atmosphere cube, filter passbands, zero-points, and runtime
/// output settings for `&colors` history evaluation.
#[derive(Debug, Clone)]
pub struct ColorsContext {
    /// Loaded atmosphere SED cube.
    pub cube: Arc<SedCube>,
    /// Minimum metallicity `[M/H]` from `lookup_table.csv` (`minval(lu_meta)`).
    pub lu_meta_min: f64,
    /// Maximum metallicity `[M/H]` from `lookup_table.csv` (`maxval(lu_meta)`).
    pub lu_meta_max: f64,
    /// Loaded photometric filters in canonical instrument order.
    pub filters: Vec<LoadedFilter>,
    /// Filter transmissions resampled onto `cube.wavelengths`.
    filters_on_sed_grid: Vec<Vec<f64>>,
    /// Denominator integrals $\int S_\lambda \lambda\,d\lambda$ on `cube.wavelengths`.
    filter_norm_integrals: Vec<f64>,
    /// Active magnitude zero-point system (`Vega`, `Ab`, `St`).
    pub mag_system: MagSystem,
    /// Observer distance $d$ in $\mathrm{cm}$.
    pub distance: f64,
    /// Reference solar $(Z/X)_\odot$ ratio for $[\mathrm{M/H}]$ mapping.
    pub z_over_x_ref: f64,
    /// Whether to write per-filter SED CSV files (`make_csv`).
    pub make_csv: bool,
    /// Whether `make_csv` includes `_<model_number:08>` per-step suffixes.
    pub sed_per_model: bool,
    /// Directory where `make_csv` writes CSV files.
    pub results_dir: PathBuf,
    /// Resolved/normalized cache key fields for `PartialEq`/`Eq` comparison.
    pub instrument_key: String,
    /// Normalized `stellar_atm` identifier.
    pub stellar_atm_key: String,
    /// Normalized `vega_sed` identifier.
    pub vega_sed_key: String,
}

impl PartialEq for ColorsContext {
    fn eq(&self, other: &Self) -> bool {
        self.instrument_key == other.instrument_key
            && self.stellar_atm_key == other.stellar_atm_key
            && self.vega_sed_key == other.vega_sed_key
            && self.mag_system == other.mag_system
            && self.distance.to_bits() == other.distance.to_bits()
            && self.z_over_x_ref.to_bits() == other.z_over_x_ref.to_bits()
            && self.make_csv == other.make_csv
            && self.sed_per_model == other.sed_per_model
            && self.results_dir == other.results_dir
            && self.filters.len() == other.filters.len()
            && self
                .filters
                .iter()
                .zip(&other.filters)
                .all(|(a, b)| a.name == b.name)
    }
}

impl Eq for ColorsContext {}

/// Options specifying `&colors` controls needed to load a [`ColorsContext`].
#[derive(Debug, Clone, PartialEq)]
pub struct ColorsLoadConfig<'a> {
    /// Filter directory / instrument index (`instrument`).
    pub instrument: &'a str,
    /// Stellar atmosphere SED grid directory (`stellar_atm`).
    pub stellar_atm: &'a str,
    /// Vega reference spectrum path (`vega_sed`).
    pub vega_sed: &'a str,
    /// Magnitude system identifier (`mag_system`: `"Vega"`, `"AB"`, `"ST"`).
    pub mag_system: &'a str,
    /// Observer distance in $\mathrm{cm}$ (`distance`).
    pub distance: f64,
    /// Solar metal-to-hydrogen reference ratio (`z_over_x_ref`).
    pub z_over_x_ref: f64,
    /// Whether to write SED CSV files (`make_csv`).
    pub make_csv: bool,
    /// Whether CSV filenames include model numbers (`sed_per_model`).
    pub sed_per_model: bool,
    /// Output directory for SED CSV files (`colors_results_directory`).
    pub colors_results_directory: &'a str,
}

impl ColorsContext {
    /// Loads and validates a [`ColorsContext`] from `&colors` configuration
    /// parameters, resolving paths against `case_dir`, `MESA_DIR`, and
    /// `FLEXMESA_DATA_DIR` without silent fallbacks.
    ///
    /// Built-in embedded assets are used only when the exact default paths
    /// (`Kurucz2003all`, `Generic/Johnson`, `GAIA/GAIA`, `vega_flam.csv`) are
    /// configured and not present on disk under `case_dir` / `MESA_DIR` /
    /// `FLEXMESA_DATA_DIR`.
    ///
    /// # Errors
    /// Returns [`ColorsError`] if any control is invalid (`mag_system`,
    /// `distance <= 0`, `z_over_x_ref <= 0`, empty path) or if required filter,
    /// Vega SED, or atmosphere files are missing or malformed.
    pub fn load(
        cfg: &ColorsLoadConfig<'_>,
        case_dir: Option<&Path>,
    ) -> Result<Self, ColorsError> {
        let mag_system = MagSystem::parse(cfg.mag_system)?;
        if !cfg.distance.is_finite() || cfg.distance <= 0.0 {
            return Err(ColorsError::InvalidControl {
                name: "distance",
                detail: format!("must be finite and > 0, got {}", cfg.distance),
            });
        }
        if !cfg.z_over_x_ref.is_finite() || cfg.z_over_x_ref <= 0.0 {
            return Err(ColorsError::InvalidControl {
                name: "z_over_x_ref",
                detail: format!(
                    "must be finite and > 0, got {}",
                    cfg.z_over_x_ref
                ),
            });
        }

        let (cube, lu_meta_min, lu_meta_max) =
            load_stellar_atmosphere(cfg.stellar_atm, case_dir)?;

        let vega_opt = load_optional_vega(cfg.vega_sed, case_dir)?;
        let filters = load_instrument_filters(
            cfg.instrument,
            vega_opt.as_ref(),
            case_dir,
        )?;
        if filters.is_empty() {
            return Err(ColorsError::InvalidData {
                context: "load_instrument_filters",
                detail: format!(
                    "instrument '{}' listed zero filters",
                    cfg.instrument.trim()
                ),
            });
        }

        let mut filters_on_sed_grid = Vec::with_capacity(filters.len());
        let mut filter_norm_integrals = Vec::with_capacity(filters.len());
        for filt in &filters {
            let f_interp = interpolate_array(
                &filt.wavelengths,
                &filt.transmission,
                &cube.wavelengths,
            )?;
            let norm = simpson_integration_fn(
                &cube.wavelengths,
                f_interp.len(),
                |i| f_interp[i] * cube.wavelengths[i],
            )?;
            if !norm.is_finite() || norm <= 0.0 {
                return Err(ColorsError::NonPositiveFlux {
                    context: "ColorsContext::load (filter integral)",
                    flux: norm,
                });
            }
            filters_on_sed_grid.push(f_interp);
            filter_norm_integrals.push(norm);
        }

        let raw_dir = cfg.colors_results_directory.trim();
        let rel_dir = if raw_dir.is_empty() { "SED" } else { raw_dir };
        let results_dir = match case_dir {
            Some(base) if Path::new(rel_dir).is_relative() => {
                base.join(rel_dir)
            }
            _ => PathBuf::from(rel_dir),
        };

        Ok(Self {
            cube,
            lu_meta_min,
            lu_meta_max,
            filters,
            filters_on_sed_grid,
            filter_norm_integrals,
            mag_system,
            distance: cfg.distance,
            z_over_x_ref: cfg.z_over_x_ref,
            make_csv: cfg.make_csv,
            sed_per_model: cfg.sed_per_model,
            results_dir,
            instrument_key: normalize_rel_key(cfg.instrument),
            stellar_atm_key: normalize_rel_key(cfg.stellar_atm),
            vega_sed_key: normalize_rel_key(cfg.vega_sed),
        })
    }

    /// Returns the ordered output history column names:
    /// `["Mag_bol", "Flux_bol", "Interp_rad", <filter_1>, ...]`
    /// (`colors_history.f90:102-113`).
    #[must_use]
    pub fn column_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(3 + self.filters.len());
        names.push(String::from("Mag_bol"));
        names.push(String::from("Flux_bol"));
        names.push(String::from("Interp_rad"));
        for f in &self.filters {
            names.push(f.name.clone());
        }
        names
    }

    /// Evaluates bolometric photometry, `Interp_rad`, and all configured
    /// synthetic passbands at surface state `(t_eff, log_g, r_cm)` and
    /// photospheric composition `(phot_x, phot_z)`, optionally writing per-filter
    /// SED CSV files (`colors_history.f90`, `history.f90:272-293`,
    /// `synthetic.f90:68-122`).
    ///
    /// # Errors
    /// Returns [`ColorsError`] if Simpson quadrature or CSV writing fails.
    pub fn eval_and_maybe_write_csv(
        &self,
        t_eff: f64,
        log_g: f64,
        phot_x: f64,
        phot_z: f64,
        r_cm: f64,
        model_number: usize,
        write_csv: bool,
    ) -> Result<ColorsEvalOutput, ColorsError> {
        let m_div_h = if phot_x > 0.0 && phot_z > 0.0 {
            let raw =
                flexmesa_math::log10((phot_z / phot_x) / self.z_over_x_ref);
            raw.clamp(self.lu_meta_min, self.lu_meta_max)
        } else {
            self.lu_meta_min
        };

        let interp_rad = compute_interp_radius(
            t_eff,
            log_g,
            m_div_h,
            &self.cube.teff_grid,
            &self.cube.logg_grid,
            &self.cube.meta_grid,
        );
        let fluxes = self.cube.eval_sed_hermite_generic(
            t_eff,
            log_g,
            m_div_h,
            r_cm,
            self.distance,
        );
        let w = &self.cube.wavelengths;
        let n = w.len();
        if fluxes.len() != n || n < 2 {
            return Err(ColorsError::InvalidArrayLength {
                context: "ColorsContext::eval_and_maybe_write_csv",
                got: fluxes.len(),
                expected: n.max(2),
            });
        }

        // Match `bolometric.f90:calculate_bolometric_phot`: clamp negative or
        // non-positive-wavelength entries on `0..n-1` before bolometric and
        // synthetic filter integration.
        let mut cleaned = fluxes;
        for i in 0..(n - 1) {
            if w[i] <= 0.0 || cleaned[i] < 0.0 {
                cleaned[i] = 0.0;
            }
        }

        let bol_flux = simpson_integration_fn(w, n, |i| cleaned[i])?;
        let mag_bol = flux_to_magnitude(bol_flux);

        let mut filter_mags = Vec::with_capacity(self.filters.len());
        if write_csv && self.make_csv {
            fs::create_dir_all(&self.results_dir).map_err(|e| {
                ColorsError::IoError {
                    path: self.results_dir.display().to_string(),
                    detail: e.to_string(),
                }
            })?;
        }

        for (idx, filt) in self.filters.iter().enumerate() {
            if t_eff < 0.0 {
                filter_mags.push(-1.0);
                continue;
            }
            let f_on_sed = &self.filters_on_sed_grid[idx];
            if write_csv && self.make_csv {
                write_sed_csv(
                    &self.results_dir,
                    filt,
                    w,
                    &cleaned,
                    f_on_sed,
                    self.sed_per_model,
                    model_number,
                )?;
            }
            let int_flux = simpson_integration_fn(w, n, |i| {
                (cleaned[i] * f_on_sed[i]) * w[i]
            })?;
            let syn_flux = int_flux / self.filter_norm_integrals[idx];
            let zp = filt.zero_point(self.mag_system);
            if zp > 0.0 && syn_flux > 0.0 {
                filter_mags
                    .push(-2.5 * flexmesa_math::intrinsic_log10(syn_flux / zp));
            } else {
                filter_mags.push(f64::MAX);
            }
        }

        Ok(ColorsEvalOutput {
            mag_bol,
            flux_bol: bol_flux,
            interp_rad,
            filter_mags,
        })
    }
}

fn normalize_rel_key(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_ascii_lowercase()
}

fn is_explicit_relative(raw: &str) -> bool {
    let t = raw.trim();
    t.starts_with("./") || t.starts_with("../")
}

fn candidate_roots() -> Vec<(PathBuf, bool)> {
    let mut roots = Vec::new();
    for var in ["MESA_DIR", "FLEXMESA_DATA_DIR"] {
        if let Some(val) = std::env::var_os(var) {
            if !val.is_empty() {
                roots.push((PathBuf::from(val), var == "MESA_DIR"));
            }
        }
    }
    roots
}

fn resolve_colors_path(
    raw: &str,
    case_dir: Option<&Path>,
    expect_dir: bool,
) -> Result<PathBuf, ColorsError> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(ColorsError::InvalidControl {
            name: if expect_dir {
                "instrument_or_stellar_atm"
            } else {
                "vega_sed"
            },
            detail: String::from(
                "path must be non-empty when use_colors = true",
            ),
        });
    }
    let matches_kind = |p: &Path| {
        if expect_dir {
            p.is_dir()
        } else {
            p.is_file()
        }
    };

    if is_explicit_relative(trimmed) {
        let p = match case_dir {
            Some(base) => base.join(trimmed),
            None => PathBuf::from(trimmed),
        };
        if matches_kind(&p) {
            return Ok(p);
        }
        return Err(ColorsError::IoError {
            path: p.display().to_string(),
            detail: String::from("not found"),
        });
    }

    let as_path = Path::new(trimmed);
    if as_path.is_absolute() {
        if matches_kind(as_path) {
            return Ok(as_path.to_path_buf());
        }
        let stripped = trimmed.trim_start_matches('/');
        for (root, _) in candidate_roots() {
            let cand = root.join(stripped);
            if matches_kind(&cand) {
                return Ok(cand);
            }
        }
        return Err(ColorsError::IoError {
            path: trimmed.to_string(),
            detail: String::from("absolute colors path not found"),
        });
    }

    let mut tried = Vec::new();
    if let Some(base) = case_dir {
        let cand = base.join(trimmed);
        if matches_kind(&cand) {
            return Ok(cand);
        }
        tried.push(cand.display().to_string());
    }
    for (root, is_mesa) in candidate_roots() {
        let cand = root.join(trimmed);
        if matches_kind(&cand) {
            return Ok(cand);
        }
        tried.push(cand.display().to_string());
        if is_mesa && !trimmed.starts_with("data/") {
            let cand2 = root.join("data").join(trimmed);
            if matches_kind(&cand2) {
                return Ok(cand2);
            }
            tried.push(cand2.display().to_string());
        } else if !is_mesa {
            if let Some(without_data) = trimmed.strip_prefix("data/") {
                let cand2 = root.join(without_data);
                if matches_kind(&cand2) {
                    return Ok(cand2);
                }
                tried.push(cand2.display().to_string());
            }
        }
    }

    Err(ColorsError::IoError {
        path: trimmed.to_string(),
        detail: format!("not found (searched: {})", tried.join(", ")),
    })
}

fn load_stellar_atmosphere(
    stellar_atm: &str,
    case_dir: Option<&Path>,
) -> Result<(Arc<SedCube>, f64, f64), ColorsError> {
    let key = normalize_rel_key(stellar_atm);
    match resolve_colors_path(stellar_atm, case_dir, true) {
        Ok(dir) => {
            let lookup_path = dir.join("lookup_table.csv");
            let entries = load_lookup_table_entries(&lookup_path)?;
            let (meta_min, meta_max) =
                iter_min_max(entries.iter().map(|r| r.meta));
            let cube_path = dir.join("flux_cube.bin");
            let cube = if cube_path.is_file() {
                load_or_cached_flux_cube(&cube_path)?
            } else {
                Arc::new(load_atmosphere_from_sed_files(&dir, &entries)?)
            };
            Ok((cube, meta_min, meta_max))
        }
        Err(err) => {
            if !is_explicit_relative(stellar_atm) && key == CANON_KURUCZ_ATM {
                let cube = builtin_sed_cube_arc();
                let (meta_min, meta_max) =
                    iter_min_max(cube.meta_grid.iter().copied());
                Ok((cube, meta_min, meta_max))
            } else {
                Err(err)
            }
        }
    }
}

fn load_or_cached_flux_cube(path: &Path) -> Result<Arc<SedCube>, ColorsError> {
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let byte_len = fs::metadata(&canon)
        .map_err(|e| ColorsError::IoError {
            path: canon.display().to_string(),
            detail: e.to_string(),
        })?
        .len();
    let cache_key = CubeCacheKey {
        path: canon.clone(),
        byte_len,
    };
    let cache = FLUX_CUBE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(existing) = guard.get(&cache_key) {
            return Ok(Arc::clone(existing));
        }
    }
    let loaded = Arc::new(SedCube::from_mesa_flux_cube(&canon)?);
    if let Ok(mut guard) = cache.lock() {
        guard.insert(cache_key, Arc::clone(&loaded));
    }
    Ok(loaded)
}

#[derive(Debug, Clone)]
struct LookupRow {
    file_name: String,
    teff: f64,
    logg: f64,
    meta: f64,
}

fn load_lookup_table_entries(
    lookup_path: &Path,
) -> Result<Vec<LookupRow>, ColorsError> {
    let text =
        fs::read_to_string(lookup_path).map_err(|e| ColorsError::IoError {
            path: lookup_path.display().to_string(),
            detail: e.to_string(),
        })?;
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let Some(raw_header) = lines.next() else {
        return Err(ColorsError::InvalidData {
            context: "load_lookup_table_entries",
            detail: format!("empty lookup table '{}'", lookup_path.display()),
        });
    };
    let header_line = raw_header.trim_start_matches('#');
    let headers: Vec<String> = header_line
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .collect();
    let find_col = |aliases: &[&str]| -> Option<usize> {
        aliases
            .iter()
            .find_map(|alias| headers.iter().position(|h| h == alias))
    };
    let logg_col = find_col(&["logg", "log_g", "log(g)", "log10g", "log10_g"]);
    let teff_col =
        find_col(&["teff", "t_eff", "t(eff)", "temperature", "temp"]);
    let meta_col = find_col(&[
        "meta",
        "feh",
        "fe_h",
        "[fe/h]",
        "mh",
        "[m/h]",
        "m_h",
        "z",
        "logz",
        "metallicity",
        "metals",
    ]);

    let mut rows = Vec::new();
    for (row_idx, line) in lines.enumerate() {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.is_empty() || cols[0].is_empty() {
            continue;
        }
        let parse_col = |col_opt: Option<usize>,
                         col_name: &str,
                         default: f64|
         -> Result<f64, ColorsError> {
            let Some(idx) = col_opt else {
                return Ok(default);
            };
            let raw =
                cols.get(idx).ok_or_else(|| ColorsError::InvalidData {
                    context: "load_lookup_table_entries",
                    detail: format!(
                        "missing '{col_name}' column (index {idx}) on row {} \
                     in '{}'",
                        row_idx + 1,
                        lookup_path.display()
                    ),
                })?;
            let tok = raw.trim_matches('"').trim();
            if tok.is_empty() {
                return Err(ColorsError::InvalidData {
                    context: "load_lookup_table_entries",
                    detail: format!(
                        "empty '{col_name}' value on row {} in '{}'",
                        row_idx + 1,
                        lookup_path.display()
                    ),
                });
            }
            let val =
                tok.parse::<f64>().map_err(|e| ColorsError::InvalidData {
                    context: "load_lookup_table_entries",
                    detail: format!(
                        "invalid '{col_name}' value '{tok}' on row {} in \
                         '{}': {e}",
                        row_idx + 1,
                        lookup_path.display()
                    ),
                })?;
            if !val.is_finite() {
                return Err(ColorsError::InvalidData {
                    context: "load_lookup_table_entries",
                    detail: format!(
                        "non-finite '{col_name}' value '{tok}' on row {} in \
                         '{}'",
                        row_idx + 1,
                        lookup_path.display()
                    ),
                });
            }
            Ok(val)
        };
        rows.push(LookupRow {
            file_name: cols[0].trim_matches('"').to_string(),
            teff: parse_col(teff_col, "teff", 0.0)?,
            logg: parse_col(logg_col, "logg", -999.0)?,
            meta: parse_col(meta_col, "meta", 0.0)?,
        });
    }
    if rows.is_empty() {
        return Err(ColorsError::InvalidData {
            context: "load_lookup_table_entries",
            detail: format!("no rows in '{}'", lookup_path.display()),
        });
    }
    Ok(rows)
}

fn iter_min_max(vals: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for v in vals {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    (min_v, max_v)
}

fn unique_sorted_axis(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut out: Vec<f64> = values.collect();
    out.sort_by(f64::total_cmp);
    out.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);
    out
}

fn find_axis_slot(axis: &[f64], val: f64) -> Option<usize> {
    axis.iter().position(|&v| (v - val).abs() <= 1.0e-9)
}

fn load_sed_file(path: &Path) -> Result<(Vec<f64>, Vec<f64>), ColorsError> {
    let text = fs::read_to_string(path).map_err(|e| ColorsError::IoError {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let mut waves = Vec::new();
    let mut fluxes = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut toks = line
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|t| !t.is_empty());
        let (Some(w_str), Some(f_str)) = (toks.next(), toks.next()) else {
            continue;
        };
        if let (Ok(w), Ok(fl)) = (w_str.parse::<f64>(), f_str.parse::<f64>()) {
            waves.push(w);
            fluxes.push(fl);
        }
    }
    if waves.len() < 2 {
        return Err(ColorsError::InvalidData {
            context: "load_sed_file",
            detail: format!(
                "SED '{}' has fewer than 2 data rows",
                path.display()
            ),
        });
    }
    Ok((waves, fluxes))
}

fn load_atmosphere_from_sed_files(
    dir: &Path,
    rows: &[LookupRow],
) -> Result<SedCube, ColorsError> {
    let teff_grid = unique_sorted_axis(rows.iter().map(|r| r.teff));
    let logg_grid = unique_sorted_axis(rows.iter().map(|r| r.logg));
    let meta_grid = unique_sorted_axis(rows.iter().map(|r| r.meta));
    let (nt, ng, nm) = (teff_grid.len(), logg_grid.len(), meta_grid.len());
    let (first_wave, _) = load_sed_file(&dir.join(&rows[0].file_name))?;
    let nlam = first_wave.len();
    let mut flux = vec![0.0_f64; nt * ng * nm * nlam];
    let mut populated = vec![false; nt * ng * nm];

    for row in rows {
        let (Some(it), Some(ig), Some(im)) = (
            find_axis_slot(&teff_grid, row.teff),
            find_axis_slot(&logg_grid, row.logg),
            find_axis_slot(&meta_grid, row.meta),
        ) else {
            continue;
        };
        let (w_row, f_row) = load_sed_file(&dir.join(&row.file_name))?;
        let resampled = if w_row == first_wave {
            f_row
        } else {
            interpolate_array(&w_row, &f_row, &first_wave)?
        };
        let cell = (it * ng + ig) * nm + im;
        let off = cell * nlam;
        flux[off..off + nlam].copy_from_slice(&resampled);
        populated[cell] = true;
    }

    if let Some(missing) = populated.iter().position(|&p| !p) {
        return Err(ColorsError::InvalidData {
            context: "load_atmosphere_from_sed_files",
            detail: format!(
                "incomplete atmosphere grid in '{}': missing cell index {missing}",
                dir.display()
            ),
        });
    }

    SedCube::from_dense(teff_grid, logg_grid, meta_grid, first_wave, flux)
}

struct VegaSpectrum {
    wavelengths: Vec<f64>,
    fluxes: Vec<f64>,
}

fn load_optional_vega(
    vega_sed: &str,
    case_dir: Option<&Path>,
) -> Result<Option<VegaSpectrum>, ColorsError> {
    let key = normalize_rel_key(vega_sed);
    match resolve_colors_path(vega_sed, case_dir, false) {
        Ok(path) => Ok(Some(load_vega_file(&path)?)),
        Err(err) => {
            if !is_explicit_relative(vega_sed) && key == CANON_VEGA_SED {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

fn load_vega_file(path: &Path) -> Result<VegaSpectrum, ColorsError> {
    let text = fs::read_to_string(path).map_err(|e| ColorsError::IoError {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let Some(_) = lines.next() else {
        return Err(ColorsError::InvalidData {
            context: "load_vega_file",
            detail: format!("empty Vega SED file '{}'", path.display()),
        });
    };
    let mut wavelengths = Vec::new();
    let mut fluxes = Vec::new();
    for line in lines {
        if line.starts_with('#') {
            continue;
        }
        let mut toks = line
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|t| !t.is_empty());
        let (Some(w_s), Some(f_s)) = (toks.next(), toks.next()) else {
            continue;
        };
        let w = w_s.parse::<f64>().map_err(|e| ColorsError::InvalidData {
            context: "load_vega_file",
            detail: format!(
                "invalid wavelength '{w_s}' in '{}': {e}",
                path.display()
            ),
        })?;
        let fl = f_s.parse::<f64>().map_err(|e| ColorsError::InvalidData {
            context: "load_vega_file",
            detail: format!(
                "invalid flux '{f_s}' in '{}': {e}",
                path.display()
            ),
        })?;
        wavelengths.push(w);
        fluxes.push(fl);
    }
    if wavelengths.len() < 2 {
        return Err(ColorsError::InvalidData {
            context: "load_vega_file",
            detail: format!(
                "Vega SED '{}' contains fewer than 2 rows",
                path.display()
            ),
        });
    }
    Ok(VegaSpectrum {
        wavelengths,
        fluxes,
    })
}

fn load_filter_file(path: &Path) -> Result<(Vec<f64>, Vec<f64>), ColorsError> {
    let text = fs::read_to_string(path).map_err(|e| ColorsError::IoError {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    parse_filter_text(&text, &path.display().to_string())
}

fn parse_filter_text(
    text: &str,
    label: &str,
) -> Result<(Vec<f64>, Vec<f64>), ColorsError> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .peekable();
    // Skip column header (`Wavelength,Transmission`) if first non-comment row
    // is non-numeric, matching `colors_utils.f90:load_filter`.
    if let Some(&first) = lines.peek() {
        let first_tok = first
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .find(|t| !t.is_empty())
            .unwrap_or("");
        if first_tok.parse::<f64>().is_err() {
            lines.next();
        }
    }

    let mut wavelengths = Vec::new();
    let mut transmission = Vec::new();
    for line in lines {
        let mut toks = line
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|t| !t.is_empty());
        let (Some(w_s), Some(t_s)) = (toks.next(), toks.next()) else {
            continue;
        };
        let w = w_s.parse::<f64>().map_err(|e| ColorsError::InvalidData {
            context: "load_filter_file",
            detail: format!("invalid wavelength '{w_s}' in '{label}': {e}"),
        })?;
        let tr = t_s.parse::<f64>().map_err(|e| ColorsError::InvalidData {
            context: "load_filter_file",
            detail: format!("invalid transmission '{t_s}' in '{label}': {e}"),
        })?;
        wavelengths.push(w);
        transmission.push(tr);
    }
    if wavelengths.len() < 2 {
        return Err(ColorsError::InvalidData {
            context: "load_filter_file",
            detail: format!("filter '{label}' contains fewer than 2 rows"),
        });
    }
    Ok((wavelengths, transmission))
}

fn load_instrument_filters(
    instrument: &str,
    vega_opt: Option<&VegaSpectrum>,
    case_dir: Option<&Path>,
) -> Result<Vec<LoadedFilter>, ColorsError> {
    let key = normalize_rel_key(instrument);
    match resolve_colors_path(instrument, case_dir, true) {
        Ok(dir) => {
            let Some(vega) = vega_opt else {
                return Err(ColorsError::IoError {
                    path: String::from(CANON_VEGA_SED),
                    detail: String::from(
                        "Vega SED file is required on disk for on-disk \
                         filter sets",
                    ),
                });
            };
            let basename = dir
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| ColorsError::InvalidControl {
                    name: "instrument",
                    detail: format!(
                        "invalid UTF-8 directory name '{}'",
                        dir.display()
                    ),
                })?;
            let index_path = dir.join(basename);
            let index_text = fs::read_to_string(&index_path).map_err(|e| {
                ColorsError::IoError {
                    path: index_path.display().to_string(),
                    detail: e.to_string(),
                }
            })?;
            let mut out = Vec::new();
            for raw in index_text.lines() {
                let fname = raw.trim();
                if fname.is_empty() || fname.starts_with('#') {
                    continue;
                }
                let fpath = dir.join(fname);
                let (fw, ft) = load_filter_file(&fpath)?;
                let vega_zp = compute_vega_zero_point(
                    &vega.wavelengths,
                    &vega.fluxes,
                    &fw,
                    &ft,
                )?;
                let ab_zp = compute_ab_zero_point(&fw, &ft)?;
                let st_zp = compute_st_zero_point(&fw, &ft)?;
                out.push(LoadedFilter {
                    raw_filename: fname.to_string(),
                    name: remove_dat(fname).to_string(),
                    wavelengths: fw,
                    transmission: ft,
                    vega_zero_point: vega_zp,
                    ab_zero_point: ab_zp,
                    st_zero_point: st_zp,
                });
            }
            Ok(out)
        }
        Err(err) => {
            if !is_explicit_relative(instrument) && key == CANON_JOHNSON_INST {
                builtin_johnson_loaded_filters(vega_opt)
            } else if !is_explicit_relative(instrument)
                && key == CANON_GAIA_INST
            {
                builtin_gaia_loaded_filters(vega_opt)
            } else {
                Err(err)
            }
        }
    }
}

fn builtin_johnson_loaded_filters(
    vega_opt: Option<&VegaSpectrum>,
) -> Result<Vec<LoadedFilter>, ColorsError> {
    JOHNSON_FILTERS
        .iter()
        .map(|f| {
            let vega_zp = match vega_opt {
                Some(v) => compute_vega_zero_point(
                    &v.wavelengths,
                    &v.fluxes,
                    f.wavelengths,
                    f.transmission,
                )?,
                None => f.vega_zero_point,
            };
            Ok(LoadedFilter {
                raw_filename: format!("{}.dat", f.name),
                name: f.name.to_string(),
                wavelengths: f.wavelengths.to_vec(),
                transmission: f.transmission.to_vec(),
                vega_zero_point: vega_zp,
                ab_zero_point: f.ab_zero_point,
                st_zero_point: f.st_zero_point,
            })
        })
        .collect()
}

fn builtin_gaia_loaded_filters(
    vega_opt: Option<&VegaSpectrum>,
) -> Result<Vec<LoadedFilter>, ColorsError> {
    // Verbatim MESA 26.4.1 `data/colors_data/filters/GAIA/GAIA` curves with
    // exact Vega zero-point fluxes computed against `vega_flam.csv`.
    const GAIA_BANDS: [(&str, &str, &str, f64); 4] = [
        (
            "G",
            "G.dat",
            include_str!("../data/gaia/G.dat"),
            4.69633530055375771e-12,
        ),
        (
            "Gbp",
            "Gbp.dat",
            include_str!("../data/gaia/Gbp.dat"),
            4.03333366396942215e-09,
        ),
        (
            "Grp",
            "Grp.dat",
            include_str!("../data/gaia/Grp.dat"),
            1.54248206144501547e-11,
        ),
        (
            "Grvs",
            "Grvs.dat",
            include_str!("../data/gaia/Grvs.dat"),
            8.86733979418697526e-10,
        ),
    ];

    GAIA_BANDS
        .iter()
        .map(|&(name, raw_filename, body, precomputed_vega_zp)| {
            let (fw, ft) = parse_filter_text(body, raw_filename)?;
            let vega_zp = match vega_opt {
                Some(v) => compute_vega_zero_point(
                    &v.wavelengths,
                    &v.fluxes,
                    &fw,
                    &ft,
                )?,
                None => precomputed_vega_zp,
            };
            let ab_zp = compute_ab_zero_point(&fw, &ft)?;
            let st_zp = compute_st_zero_point(&fw, &ft)?;
            Ok(LoadedFilter {
                raw_filename: raw_filename.to_string(),
                name: name.to_string(),
                wavelengths: fw,
                transmission: ft,
                vega_zero_point: vega_zp,
                ab_zero_point: ab_zp,
                st_zero_point: st_zp,
            })
        })
        .collect()
}

fn write_sed_csv(
    results_dir: &Path,
    filt: &LoadedFilter,
    wavelengths: &[f64],
    fluxes: &[f64],
    filter_on_sed: &[f64],
    sed_per_model: bool,
    model_number: usize,
) -> Result<(), ColorsError> {
    let fname = if sed_per_model {
        format!("{}_SED_{model_number:08}.csv", filt.name)
    } else {
        format!("{}_SED.csv", filt.name)
    };
    let path = results_dir.join(fname);
    let max_size = wavelengths.len().max(filt.wavelengths.len());
    let mut out = String::with_capacity(64 + max_size * 80);
    out.push_str(
        "wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans\n",
    );
    for i in 0..max_size {
        let wv = wavelengths.get(i).copied().unwrap_or(0.0);
        let fl = fluxes.get(i).copied().unwrap_or(0.0);
        let cf = fl * filter_on_sed.get(i).copied().unwrap_or(0.0);
        let fwv = filt.wavelengths.get(i).copied().unwrap_or(0.0);
        let ftr = filt.transmission.get(i).copied().unwrap_or(0.0);
        push_fortran_es14_6(&mut out, wv);
        out.push(',');
        push_fortran_es14_6(&mut out, fl);
        out.push(',');
        push_fortran_es14_6(&mut out, cf);
        out.push(',');
        push_fortran_es14_6(&mut out, fwv);
        out.push(',');
        push_fortran_es14_6(&mut out, ftr);
        out.push('\n');
    }
    fs::write(&path, out).map_err(|e| ColorsError::IoError {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

fn push_fortran_es14_6(out: &mut String, val: f64) {
    if !val.is_finite() {
        let _ = write!(out, "{val:>14}");
        return;
    }
    let raw = format!("{val:.6E}");
    if let Some((mant, exp_str)) = raw.split_once('E') {
        let exp: i32 = exp_str.parse().unwrap_or(0);
        let sign = if exp < 0 { '-' } else { '+' };
        let abs_exp = exp.unsigned_abs();
        let formatted = if abs_exp < 100 {
            format!("{mant}E{sign}{abs_exp:02}")
        } else {
            format!("{mant}E{sign}{abs_exp}")
        };
        let _ = write!(out, "{formatted:>14}");
    } else {
        let _ = write!(out, "{raw:>14}");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::sed_cube::builtin_sed_cube;

    static TEST_TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(tag: &str) -> Self {
            let id = TEST_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "flexmesa_colors_loader_{tag}_{}_{id}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).expect("create temp test dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn builtin_sed_cube_arc_shares_singleton_allocation() {
        let static_ref = builtin_sed_cube();
        let arc_ref = builtin_sed_cube_arc();
        assert!(std::ptr::eq(static_ref, arc_ref.as_ref()));
    }

    #[test]
    fn builtin_johnson_propagates_custom_vega_zero_point_error() {
        let narrow_vega = VegaSpectrum {
            wavelengths: vec![4000.0, 5000.0],
            fluxes: vec![1.0e-8, 1.0e-8],
        };
        let err = builtin_johnson_loaded_filters(Some(&narrow_vega))
            .expect_err("narrow Vega SED must error instead of falling back");
        assert!(matches!(err, ColorsError::NonPositiveFlux { .. }));
    }

    #[test]
    fn load_instrument_filters_on_disk_johnson_requires_vega_sed() {
        let temp = TempDirGuard::new("disk_johnson");
        let inst_dir =
            temp.path().join("data/colors_data/filters/Generic/Johnson");
        fs::create_dir_all(&inst_dir).expect("create disk Johnson dir");
        let err = load_instrument_filters(
            "data/colors_data/filters/Generic/Johnson",
            None,
            Some(temp.path()),
        )
        .expect_err("on-disk Johnson dir without disk Vega SED must error");
        assert!(matches!(err, ColorsError::IoError { .. }));
    }

    #[test]
    fn load_lookup_table_rejects_malformed_present_columns_and_allows_absent_columns(
    ) {
        let temp = TempDirGuard::new("lookup_validation");
        let absent_path = temp.path().join("absent_cols.csv");
        fs::write(&absent_path, "#file_name, teff\na.txt, 5000.0\n")
            .expect("write absent-cols lookup table");
        let rows =
            load_lookup_table_entries(&absent_path).expect("parse 2-col table");
        assert_eq!(rows.len(), 1);
        assert!((rows[0].teff - 5000.0).abs() < 1.0e-12);
        assert!((rows[0].logg + 999.0).abs() < 1.0e-12);
        assert!(rows[0].meta.abs() < 1.0e-12);

        let invalid_cases = [
            (
                "non_numeric",
                "file_name, teff, logg, meta\na.txt, abc, 4.0, 0.0\n",
            ),
            (
                "blank_cell",
                "file_name, teff, logg, meta\na.txt, 5000.0, , 0.0\n",
            ),
            ("truncated", "file_name, teff, logg, meta\na.txt, 5000.0\n"),
            (
                "nan_cell",
                "file_name, teff, logg, meta\na.txt, NaN, 4.0, 0.0\n",
            ),
            (
                "inf_cell",
                "file_name, teff, logg, meta\na.txt, 5000.0, inf, 0.0\n",
            ),
        ];
        for (tag, contents) in invalid_cases {
            let p = temp.path().join(format!("{tag}.csv"));
            fs::write(&p, contents).expect("write invalid lookup table");
            let err = load_lookup_table_entries(&p)
                .expect_err("malformed present lookup cell must fail");
            assert!(
                matches!(err, ColorsError::InvalidData { .. }),
                "expected InvalidData for case {tag}, got {err:?}"
            );
        }
    }

    #[test]
    fn builtin_gaia_filters_parse_propagate_error_and_match_disk() {
        let filters = builtin_gaia_loaded_filters(None)
            .expect("built-in Gaia filters must load");
        let names: Vec<&str> =
            filters.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["G", "Gbp", "Grp", "Grvs"]);
        let counts: Vec<usize> =
            filters.iter().map(|f| f.wavelengths.len()).collect();
        assert_eq!(counts, [731, 426, 471, 963]);
        for f in &filters {
            assert!(f.vega_zero_point.is_finite() && f.vega_zero_point > 0.0);
            assert!(f.ab_zero_point.is_finite() && f.ab_zero_point > 0.0);
            assert!(f.st_zero_point.is_finite() && f.st_zero_point > 0.0);
        }

        let off_band_vega = VegaSpectrum {
            wavelengths: vec![20000.0, 30000.0],
            fluxes: vec![1.0e-8, 1.0e-8],
        };
        let err = builtin_gaia_loaded_filters(Some(&off_band_vega))
            .expect_err("off-band Vega SED must error for built-in Gaia");
        assert!(matches!(err, ColorsError::NonPositiveFlux { .. }));

        if let Some(mesa_dir) = std::env::var_os("MESA_DIR") {
            let mesa_root = PathBuf::from(mesa_dir);
            let gaia_index =
                mesa_root.join("data/colors_data/filters/GAIA/GAIA/GAIA");
            let vega_path =
                mesa_root.join("data/colors_data/stellar_models/vega_flam.csv");
            if gaia_index.is_file() && vega_path.is_file() {
                let vega = load_vega_file(&vega_path).expect("load MESA Vega");
                let disk_filters = load_instrument_filters(
                    "data/colors_data/filters/GAIA/GAIA",
                    Some(&vega),
                    None,
                )
                .expect("load disk Gaia filters");
                assert_eq!(disk_filters.len(), filters.len());
                for (builtin, disk) in filters.iter().zip(disk_filters.iter()) {
                    assert_eq!(builtin.name, disk.name);
                    assert_eq!(
                        builtin.vega_zero_point.to_bits(),
                        disk.vega_zero_point.to_bits()
                    );
                    assert_eq!(
                        builtin.ab_zero_point.to_bits(),
                        disk.ab_zero_point.to_bits()
                    );
                    assert_eq!(
                        builtin.st_zero_point.to_bits(),
                        disk.st_zero_point.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn load_instrument_filters_on_disk_gaia_requires_vega_sed() {
        let temp = TempDirGuard::new("disk_gaia");
        let inst_dir = temp.path().join("data/colors_data/filters/GAIA/GAIA");
        fs::create_dir_all(&inst_dir).expect("create disk Gaia dir");
        let err = load_instrument_filters(
            "data/colors_data/filters/GAIA/GAIA",
            None,
            Some(temp.path()),
        )
        .expect_err("on-disk Gaia dir without disk Vega SED must error");
        assert!(matches!(err, ColorsError::IoError { .. }));
    }
}
