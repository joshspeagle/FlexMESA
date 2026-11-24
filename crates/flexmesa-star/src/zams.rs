// SPDX-License-Identifier: GPL-3.0-or-later
//! Zero-Age Main Sequence (ZAMS) model catalog, `interp_q` Lagrangian
//! interpolation, and `load_zams_model` (`star/private/init_model.f90`).

use std::sync::OnceLock;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{M_SUN, SECYER};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;

const BUILTIN_ZAMS_BYTES: &[u8] =
    include_bytes!("../data/zams_z2m2_y28_subset.bin");

static BUILTIN_ZAMS_CATALOG: OnceLock<ZamsCatalog> = OnceLock::new();

/// Rounds a ZAMS model header mass to the 8-decimal catalog index-table value
/// (`star/private/init_model.f90:210, 254, 444-446`).
fn round_zams_index_mass(m_msun: f64) -> f64 {
    format!("{m_msun:.8}").parse::<f64>().unwrap_or(m_msun)
}

/// Single pre-computed ZAMS model entry from `zams_z2m2_y28.data`.
#[derive(Debug, Clone)]
pub struct ZamsEntry {
    /// Model header stellar mass in solar masses ($M_\odot$).
    pub m_msun: f64,
    /// 8-decimal index-table stellar mass in solar masses ($M_\odot$) used by
    /// upstream `get1_zams_model` / `get1_mass` (`star/private/init_model.f90:210, 254, 444-446`).
    pub index_m_msun: f64,
    /// Number of Lagrangian zones (`nz`).
    pub nz: usize,
    /// Number of nuclear species (`species = 8` for `basic.net`).
    pub species: usize,
    /// 1-based `chem_id` indices of species (`len = species`).
    pub chem_id: Vec<usize>,
    /// Cell-centered $\ln(\rho/\mathrm{g\,cm^{-3}})$ (`len = nz`).
    pub lnd: Vec<f64>,
    /// Cell-centered $\ln(T/\mathrm{K})$ (`len = nz`).
    pub ln_t: Vec<f64>,
    /// Face-centered $\ln(R/\mathrm{cm})$ (`len = nz`).
    pub ln_r: Vec<f64>,
    /// Face-centered luminosity $L$ in $\mathrm{erg\,s^{-1}}$ (`len = nz`).
    pub lum: Vec<f64>,
    /// Cell relative mass fraction $\Delta q_k$ (`len = nz`).
    pub dq: Vec<f64>,
    /// Cell species mass fractions `xa[k * species + j]` (`len = nz * species`).
    pub xa: Vec<f64>,
}

/// Catalog of pre-computed ZAMS models sorted by increasing `m_msun`.
#[derive(Debug, Clone)]
pub struct ZamsCatalog {
    /// Initial metal mass fraction $Z$ (`0.02`).
    pub initial_z: f64,
    /// Initial helium mass fraction $Y$ (`0.28`).
    pub initial_y: f64,
    /// Ordered ZAMS model entries.
    pub models: Vec<ZamsEntry>,
}

impl ZamsCatalog {
    /// Returns the built-in $Z = 0.02, Y = 0.28$ ZAMS catalog (`basic.net`,
    /// 8 species: `h1, he3, he4, c12, n14, o16, ne20, mg24`).
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_ZAMS_CATALOG.get_or_init(|| {
            Self::from_binary_bytes(BUILTIN_ZAMS_BYTES)
                .expect("valid embedded zams_z2m2_y28_subset.bin")
        })
    }

    /// Parses a binary `MZAM` catalog blob.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if the binary header or payload is malformed.
    pub fn from_binary_bytes(bytes: &[u8]) -> Result<Self, StarError> {
        if bytes.len() < 16 || &bytes[0..4] != b"MZAM" {
            return Err(StarError::IoError(String::from(
                "invalid ZAMS binary magic header",
            )));
        }
        let mut pos = 4;
        let read_u32 = |pos: &mut usize| -> Result<usize, StarError> {
            let s = bytes.get(*pos..*pos + 4).ok_or_else(|| {
                StarError::IoError(String::from("EOF reading u32 in ZAMS"))
            })?;
            *pos += 4;
            Ok(u32::from_le_bytes(s.try_into().unwrap()) as usize)
        };
        let read_f64 = |pos: &mut usize| -> Result<f64, StarError> {
            let s = bytes.get(*pos..*pos + 8).ok_or_else(|| {
                StarError::IoError(String::from("EOF reading f64 in ZAMS"))
            })?;
            *pos += 8;
            Ok(f64::from_le_bytes(s.try_into().unwrap()))
        };
        let read_f64_vec =
            |pos: &mut usize, n: usize| -> Result<Vec<f64>, StarError> {
                let s = bytes.get(*pos..*pos + 8 * n).ok_or_else(|| {
                    StarError::IoError(String::from("EOF reading f64 slice"))
                })?;
                *pos += 8 * n;
                Ok(s.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&c| f64::from_le_bytes(c))
                    .collect())
            };

        let _version = read_u32(&mut pos)?;
        let n_models = read_u32(&mut pos)?;
        let initial_z = read_f64(&mut pos)?;
        let initial_y = read_f64(&mut pos)?;
        let species = 8_usize;
        let catalog = NuclideCatalog::builtin_ref();
        let iso_names =
            ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];
        let chem_id: Vec<usize> = iso_names
            .iter()
            .map(|name| {
                catalog.get_nuclide_index(name).ok_or_else(|| {
                    StarError::IoError(format!("unknown ZAMS nuclide '{name}'"))
                })
            })
            .collect::<Result<Vec<usize>, StarError>>()?;

        let mut models = Vec::with_capacity(n_models);
        for _ in 0..n_models {
            let m_msun = read_f64(&mut pos)?;
            let nz = read_u32(&mut pos)?;
            let row_vals = read_f64_vec(&mut pos, nz * (5 + species))?;
            let mut lnd = Vec::with_capacity(nz);
            let mut ln_t = Vec::with_capacity(nz);
            let mut ln_r = Vec::with_capacity(nz);
            let mut lum = Vec::with_capacity(nz);
            let mut dq = Vec::with_capacity(nz);
            let mut xa = Vec::with_capacity(nz * species);
            for k in 0..nz {
                let b = k * (5 + species);
                lnd.push(row_vals[b]);
                ln_t.push(row_vals[b + 1]);
                ln_r.push(row_vals[b + 2]);
                lum.push(row_vals[b + 3]);
                dq.push(row_vals[b + 4]);
                // Preserve catalog `xa` values without renormalization,
                // matching upstream `read1_model` (`read_model.f90:648-650`).
                xa.extend_from_slice(&row_vals[b + 5..b + 5 + species]);
            }
            models.push(ZamsEntry {
                m_msun,
                index_m_msun: round_zams_index_mass(m_msun),
                nz,
                species,
                chem_id: chem_id.clone(),
                lnd,
                ln_t,
                ln_r,
                lum,
                dq,
                xa,
            });
        }
        Ok(Self {
            initial_z,
            initial_y,
            models,
        })
    }

    /// Parses an upstream ASCII `zams_*.data` file (`init_model.f90:155-394`).
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if parsing fails.
    pub fn parse_ascii(text: &str) -> Result<Self, StarError> {
        let mut lines = text.lines();
        let mut initial_z = 0.02;
        let mut initial_y = 0.28;
        for line in lines.by_ref() {
            let trimmed = line.trim();
            if trimmed.starts_with("initial_Z")
                || trimmed.starts_with("initial_z")
            {
                if let Some(tok) = trimmed.split_whitespace().nth(1) {
                    initial_z = parse_fortran_f64(tok)?;
                }
            } else if trimmed.starts_with("initial_Y")
                || trimmed.starts_with("initial_y")
            {
                if let Some(tok) = trimmed.split_whitespace().nth(1) {
                    initial_y = parse_fortran_f64(tok)?;
                }
            } else if trimmed.starts_with("M/Msun") {
                break;
            }
        }
        let mut index_masses: Vec<f64> = Vec::new();
        for line in lines.by_ref() {
            let t = line.trim();
            if t == "1"
                || t.starts_with("-1.")
                || t.starts_with("-1 ")
                || t == "-1"
            {
                break;
            }
            if t.is_empty() {
                continue;
            }
            if let Some(tok) = t.split_whitespace().next() {
                if let Ok(m_idx) = parse_fortran_f64(tok) {
                    if m_idx > 0.0 {
                        index_masses.push(m_idx);
                    }
                }
            }
        }

        let catalog = NuclideCatalog::builtin_ref();
        let mut models = Vec::new();
        while let Some(line) = lines.next() {
            let trimmed = line.trim();
            if trimmed.starts_with("M/Msun") {
                let m_tok =
                    trimmed.split_whitespace().nth(1).ok_or_else(|| {
                        StarError::IoError(String::from("missing M/Msun value"))
                    })?;
                let m_msun = parse_fortran_f64(m_tok)?;
                let mut nz = 0_usize;
                for hdr in lines.by_ref() {
                    let h = hdr.trim();
                    if h.starts_with("n_shells") {
                        let n_tok =
                            h.split_whitespace().nth(1).ok_or_else(|| {
                                StarError::IoError(String::from(
                                    "missing n_shells value",
                                ))
                            })?;
                        nz = n_tok.parse::<usize>().map_err(|e| {
                            StarError::IoError(format!("invalid n_shells: {e}"))
                        })?;
                    } else if h.starts_with("lnd ") || h.contains("lnT") {
                        let cols: Vec<&str> = h.split_whitespace().collect();
                        // Columns: [k?, lnd, lnT, lnR, L, dq, iso1, iso2, ...]
                        let lnd_pos = cols
                            .iter()
                            .position(|&c| c == "lnd")
                            .ok_or_else(|| {
                                StarError::IoError(format!(
                                    "missing 'lnd' column in ZAMS header: {h}"
                                ))
                            })?;
                        if lnd_pos + 5 > cols.len() {
                            return Err(StarError::IoError(format!(
                                "truncated ZAMS column header: {h}"
                            )));
                        }
                        let iso_cols = &cols[lnd_pos + 5..];
                        let species = iso_cols.len();
                        let chem_id: Vec<usize> = iso_cols
                            .iter()
                            .map(|name| {
                                catalog.get_nuclide_index(name).ok_or_else(
                                    || {
                                        StarError::IoError(format!(
                                            "unknown ZAMS nuclide '{name}'"
                                        ))
                                    },
                                )
                            })
                            .collect::<Result<Vec<usize>, StarError>>()?;
                        let mut lnd = Vec::with_capacity(nz);
                        let mut ln_t = Vec::with_capacity(nz);
                        let mut ln_r = Vec::with_capacity(nz);
                        let mut lum = Vec::with_capacity(nz);
                        let mut dq = Vec::with_capacity(nz);
                        let mut xa = Vec::with_capacity(nz * species);
                        for _ in 0..nz {
                            let row_line = lines.next().ok_or_else(|| {
                                StarError::IoError(String::from(
                                    "unexpected EOF in ZAMS zone table",
                                ))
                            })?;
                            let toks: Vec<&str> =
                                row_line.split_whitespace().collect();
                            if toks.len() < 6 + species {
                                return Err(StarError::IoError(format!(
                                    "short ZAMS zone line: {row_line}"
                                )));
                            }
                            lnd.push(parse_fortran_f64(toks[1])?);
                            ln_t.push(parse_fortran_f64(toks[2])?);
                            ln_r.push(parse_fortran_f64(toks[3])?);
                            lum.push(parse_fortran_f64(toks[4])?);
                            dq.push(parse_fortran_f64(toks[5])?);
                            // Preserve ASCII catalog `xa` values without
                            // renormalization (`read_model.f90:648-650`).
                            for j in 0..species {
                                xa.push(parse_fortran_f64(toks[6 + j])?);
                            }
                        }
                        let index_m_msun = index_masses
                            .get(models.len())
                            .copied()
                            .filter(|&m_idx| (m_idx - m_msun).abs() <= 1.0e-4)
                            .or_else(|| {
                                index_masses.iter().copied().find(|&m_idx| {
                                    (m_idx - m_msun).abs() <= 1.0e-4
                                })
                            })
                            .unwrap_or_else(|| round_zams_index_mass(m_msun));
                        models.push(ZamsEntry {
                            m_msun,
                            index_m_msun,
                            nz,
                            species,
                            chem_id,
                            lnd,
                            ln_t,
                            ln_r,
                            lum,
                            dq,
                            xa,
                        });
                        break;
                    }
                }
            }
        }
        Ok(Self {
            initial_z,
            initial_y,
            models,
        })
    }
}

/// Parses a Fortran floating-point string (supporting `D`/`d` exponents and
/// 3-digit exponent overflows without `E`/`D`, e.g. `9.9999999999999982-100`).
pub(crate) fn parse_fortran_f64(s: &str) -> Result<f64, StarError> {
    let trimmed = s.trim();
    let cleaned = if trimmed.contains(['D', 'd']) {
        trimmed.replace(['D', 'd'], "e")
    } else {
        trimmed.to_string()
    };
    if let Ok(v) = cleaned.parse::<f64>() {
        return Ok(v);
    }
    // Handle Fortran 3-digit exponents that omit 'E'/'D' (e.g. `1.2345-100`)
    let bytes = cleaned.as_bytes();
    for i in (1..bytes.len()).rev() {
        if (bytes[i] == b'-' || bytes[i] == b'+')
            && bytes[i - 1] != b'e'
            && bytes[i - 1] != b'E'
        {
            let mut fixed = String::with_capacity(cleaned.len() + 1);
            fixed.push_str(&cleaned[..i]);
            fixed.push('e');
            fixed.push_str(&cleaned[i..]);
            if let Ok(v) = fixed.parse::<f64>() {
                return Ok(v);
            }
        }
    }
    Err(StarError::IoError(format!("invalid float '{s}'")))
}

/// Computes the initial ZAMS timestep in years (`yrs_for_init_timestep` in
/// `star/private/star_utils.f90:2900-2907`).
#[must_use]
pub fn yrs_for_init_timestep(m_msun: f64) -> f64 {
    if m_msun < 0.3 {
        1.0e7
    } else if m_msun <= 30.0 {
        1.0e5 / flexmesa_math::pow(m_msun, 2.5)
    } else {
        20.0 * flexmesa_math::pow(30.0 / m_msun, 1.5)
    }
}

/// Computes top-down outer-face enclosed mass fractions $q_k$ matching
/// `set_qs` in `star/private/star_utils.f90:729-752`.
fn face_q_from_dq(dq: &[f64]) -> Vec<f64> {
    let nz = dq.len();
    if nz == 0 {
        return Vec::new();
    }
    let mut q = vec![0.0; nz];
    q[0] = 1.0;
    let mut okay = true;
    for k in 1..nz {
        q[k] = q[k - 1] - dq[k - 1];
        if q[k] < 0.0 || q[k] > 1.0 {
            okay = false;
            break;
        }
    }
    if okay {
        return q;
    }
    let mut midq = 0_usize;
    let mut max_dq = dq[0];
    for (k, &val) in dq.iter().enumerate().skip(1) {
        if val > max_dq {
            max_dq = val;
            midq = k;
        }
    }
    let mut dqsum1 = 0.0;
    for k in 0..=midq {
        q[k] = 1.0 - dqsum1;
        dqsum1 += dq[k];
    }
    let mut dqsum2 = 0.0;
    for k in (midq + 1..nz).rev() {
        dqsum2 += dq[k];
        q[k] = dqsum2;
    }
    q
}

/// Binary search on a decreasing array `vec` returning `k` in `0..vec.len()-1`
/// with `vec[k] >= val > vec[k + 1]` (`binary_search` in
/// `num/public/num_binary_search.dek:23-53`).
fn binary_search_decreasing(vec: &[f64], val: f64) -> Option<usize> {
    let n = vec.len();
    if n <= 1 || val > vec[0] {
        return None;
    }
    if val == vec[0] {
        return Some(0);
    }
    if val <= vec[n - 1] {
        return Some(n - 1);
    }
    let mut first = 0_usize;
    let mut last = n - 2;
    while first <= last {
        let mid = (first + last) / 2;
        if vec[mid] >= val {
            if val > vec[mid + 1] {
                return Some(mid);
            }
            first = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            last = mid - 1;
        }
    }
    None
}

/// Interpolates profile `v1` from grid `dq1` onto grid `dq2` using upstream
/// piecewise-linear interpolation on top-down outer-face enclosed mass
/// fraction $q_k$ (`set_qs` and `interp_q` in
/// `star/private/star_utils.f90:729-752, 1232-1265`).
///
/// # Errors
/// Returns [`StarError`] if `dq1`/`v1` lengths are invalid or bracketing fails.
pub fn interp_q(
    dq1: &[f64],
    v1: &[f64],
    dq2: &[f64],
) -> Result<Vec<f64>, StarError> {
    let nz1 = dq1.len();
    let nz2 = dq2.len();
    if nz1 < 2 || v1.len() != nz1 {
        return Err(StarError::InvalidInput(String::from(
            "invalid dq1/v1 length in interp_q",
        )));
    }
    if nz2 == 0 {
        return Ok(Vec::new());
    }
    let q1 = face_q_from_dq(dq1);
    let q2 = face_q_from_dq(dq2);

    let mut out = Vec::with_capacity(nz2);
    for &qval in &q2 {
        if qval <= q1[nz1 - 1] {
            out.push(v1[nz1 - 1]);
            continue;
        }
        let Some(k) = binary_search_decreasing(&q1, qval) else {
            return Err(StarError::InvalidInput(format!(
                "interp_q binary_search failed for qval={qval}"
            )));
        };
        if k >= nz1 - 1 || !(qval <= q1[k] && qval > q1[k + 1]) {
            return Err(StarError::InvalidInput(format!(
                "interp_q bracket invalid at k={k} for qval={qval}"
            )));
        }
        let alfa = (qval - q1[k + 1]) / dq1[k];
        out.push(alfa * v1[k] + (1.0 - alfa) * v1[k + 1]);
    }
    Ok(out)
}

/// Constructs an initial [`StarModel`] by logarithmic mass interpolation in a
/// [`ZamsCatalog`] (`get_zams_model` / `get1_zams_model` / `get1_mass` in
/// `star/private/init_model.f90:65-466`).
///
/// # Errors
/// Returns [`StarError`] if the catalog is empty, `controls.initial_mass` is
/// non-finite or non-positive, or interpolation / model initialization fails.
pub fn load_zams_model_from_catalog(
    catalog: &ZamsCatalog,
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let m_target = controls.initial_mass;
    if catalog.models.is_empty() {
        return Err(StarError::InvalidInput(String::from(
            "empty ZAMS catalog",
        )));
    }
    if !m_target.is_finite() || m_target <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "invalid initial_mass={m_target} for ZAMS catalog"
        )));
    }

    // Locate bracketing models matching upstream `get1_zams_model`
    // (`star/private/init_model.f90:209-243`):
    // - Scan in ascending mass for the first catalog entry with
    //   `m2 >= initial_mass` (`init_model.f90:221`).
    // - If the very first entry (`i == 0`, where `m1 == 0`) matches
    //   (including `initial_mass <= m_min`), clamp `m1 = m2`, `in_range = false`,
    //   and revise `initial_mass` to `m_min` (`init_model.f90:222-228`).
    // - If the end of the catalog is reached (`initial_mass > m_max`), clamp
    //   `m2 = m1`, `in_range = false`, and revise `initial_mass` to `m_max`
    //   (`init_model.f90:212-220`).
    // - Otherwise (`m_min < initial_mass <= m_max`), bracket `(i - 1, i)` so
    //   exact interior catalog masses (`initial_mass == m2`, e.g. `1.0` or
    //   `10.0`) interpolate onto `m1`'s grid with `alfa = 0` rather than
    //   copying `m2` directly (`init_model.f90:221, 240-241, 443-458`).
    let (idx1, idx2, in_range, m_effective) = match catalog
        .models
        .iter()
        .position(|entry| entry.index_m_msun >= m_target)
    {
        Some(0) => (0, 0, false, catalog.models[0].index_m_msun),
        Some(i2) => (i2 - 1, i2, true, m_target),
        None => {
            let last = catalog.models.len() - 1;
            (last, last, false, catalog.models[last].index_m_msun)
        }
    };
    let e1 = &catalog.models[idx1];
    let e2 = &catalog.models[idx2];

    let mut model_controls = controls;
    if !in_range {
        // Match `get1_zams_model` (`star/private/init_model.f90:217-218, 226-227`):
        // revise `initial_mass` to the clamped edge catalog mass.
        // Upstream `get_zams_model` (`init_model.f90:118-127`) subsequently runs
        // `do_relax_mass` when `.not. in_range` (tracked under #154).
        model_controls.initial_mass = m_effective;
    }

    let nz = e1.nz;
    let species = e1.species;
    let mut model = StarModel::new(nz, e1.chem_id.clone(), job, model_controls);
    model.dq.clone_from(&e1.dq);

    if idx1 == idx2 || e1.index_m_msun == e2.index_m_msun {
        model.lnd.clone_from(&e1.lnd);
        model.ln_t.clone_from(&e1.ln_t);
        model.ln_r.clone_from(&e1.ln_r);
        model.lum.clone_from(&e1.lum);
        model.xa.clone_from(&e1.xa);
    } else {
        // Upstream MESA `get1_mass` (`star/private/init_model.f90:210, 254, 443-458`):
        // uses 8-decimal index-table masses m1, m2 for log-mass interpolation
        // onto `m1`'s `q` grid (`alfa = 0` when `initial_mass == m2`).
        let lnm1 = flexmesa_math::log(e1.index_m_msun);
        let lnm2 = flexmesa_math::log(e2.index_m_msun);
        let alfa = (flexmesa_math::log(m_effective) - lnm2) / (lnm1 - lnm2);
        let beta = 1.0 - alfa;

        let lnd2 = interp_q(&e2.dq, &e2.lnd, &e1.dq)?;
        let lnt2 = interp_q(&e2.dq, &e2.ln_t, &e1.dq)?;
        let lnr2 = interp_q(&e2.dq, &e2.ln_r, &e1.dq)?;
        let lum2 = interp_q(&e2.dq, &e2.lum, &e1.dq)?;

        for k in 0..nz {
            model.lnd[k] = alfa * e1.lnd[k] + beta * lnd2[k];
            model.ln_t[k] = alfa * e1.ln_t[k] + beta * lnt2[k];
            model.ln_r[k] = alfa * e1.ln_r[k] + beta * lnr2[k];
            model.lum[k] = alfa * e1.lum[k] + beta * lum2[k];
        }

        for j in 0..species {
            let x2_col: Vec<f64> =
                (0..e2.nz).map(|k| e2.xa[k * species + j]).collect();
            let x2_interp = interp_q(&e2.dq, &x2_col, &e1.dq)?;
            for k in 0..nz {
                let x1_kj = e1.xa[k * species + j];
                // Upstream `get1_mass` (`star/private/init_model.f90:455`)
                // blends `xa` linearly in log-mass without clamping or
                // renormalizing cell totals (`read_model.f90:649`).
                model.xa[k * species + j] = alfa * x1_kj + beta * x2_interp[k];
            }
        }
    }

    model.mstar = m_effective * M_SUN;
    (model.xmstar, model.mstar_start, model.mstar_old) =
        (model.mstar, model.mstar, model.mstar);
    // Upstream `alloc_star_data` initializes `model_number = 0` (`star/private/init.f90:367`).
    (model.model_number, model.star_age) = (0, 0.0);
    let dt_sec = yrs_for_init_timestep(m_effective) * SECYER;
    (model.dt, model.dt_next, model.dt_old) = (dt_sec, dt_sec, 0.0);
    model.finish_init_model()?;
    Ok(model)
}

/// Loads a ZAMS [`StarModel`] from the built-in `zams_z2m2_y28` catalog.
///
/// # Errors
/// Returns [`StarError`] if `controls.initial_mass` is non-finite or
/// non-positive, or if model initialization fails.
pub fn load_zams_model(
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    load_zams_model_from_catalog(ZamsCatalog::builtin(), controls, job)
}

#[cfg(test)]
mod zams_catalog_tests {
    use super::*;

    #[test]
    fn test_builtin_zams_catalog_loads_full_mesa_grid_and_interpolates_0p08_to_100_msun(
    ) {
        let cat = ZamsCatalog::builtin();
        assert_eq!(cat.models.len(), 35);
        assert!((cat.initial_z - 0.02).abs() <= 1.0e-15);
        assert!((cat.initial_y - 0.28).abs() <= 1.0e-15);
        assert!((cat.models[0].m_msun - 0.12589254117941673).abs() <= 1.0e-12);
        assert!(
            (cat.models.last().unwrap().m_msun - 316.22776601683796).abs()
                <= 1.0e-10
        );

        if let Some(ascii_path) = flexmesa_utils::require_mesa_test_input_path(
            "data/star_data/zams_models/zams_z2m2_y28.data",
        ) {
            let text = std::fs::read_to_string(&ascii_path)
                .expect("read zams_z2m2_y28.data");
            let parsed =
                ZamsCatalog::parse_ascii(&text).expect("parse_ascii zams");
            assert_eq!(parsed.models.len(), cat.models.len());
            for (a, b) in cat.models.iter().zip(parsed.models.iter()) {
                assert!((a.m_msun - b.m_msun).abs() <= 1.0e-14 * a.m_msun);
                assert_eq!(a.index_m_msun.to_bits(), b.index_m_msun.to_bits());
                assert_eq!(a.nz, b.nz);
                assert_eq!(a.lnd, b.lnd);
                assert_eq!(a.ln_t, b.ln_t);
            }
        }

        // Upstream `get1_zams_model` (`star/private/init_model.f90:221-228`)
        // clamps masses below the catalog minimum (`m_min = 0.12589254`) to
        // entry 0 (`nz = 3777`, `initial_mass = m_min`) instead of returning an
        // out-of-range error (replaces old non-upstream `InvalidInput` check).
        for &m_msun in &[0.08, 0.10] {
            let controls = StarControls {
                initial_mass: m_msun,
                initial_z: 0.02,
                ..StarControls::default()
            };
            let model = load_zams_model(controls, StarJobControls::default())
                .expect("sub-grid ZAMS mass clamps to minimum entry");
            assert_eq!(model.model_number, 0);
            assert_eq!(model.nz, cat.models[0].nz);
            assert_eq!(model.controls.initial_mass, cat.models[0].index_m_msun);
            assert_eq!(model.mstar, cat.models[0].index_m_msun * M_SUN);
        }
        let (mut prev_l, mut prev_r) = (0.0_f64, 0.0_f64);
        for &m_msun in
            &[0.13, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0, 50.0, 100.0, 300.0]
        {
            let controls = StarControls {
                initial_mass: m_msun,
                initial_z: 0.02,
                ..StarControls::default()
            };
            let model = load_zams_model(controls, StarJobControls::default())
                .expect("load_zams_model across 0.13..=300 Msun");
            assert_eq!(model.model_number, 0);
            assert!(model.nz >= 1980);
            assert!(
                (model.mstar - m_msun * M_SUN).abs() <= 1.0e-12 * model.mstar
            );
            assert!((model.dq.iter().sum::<f64>() - 1.0).abs() <= 1.0e-13);
            let (l_lsun, r_rsun) =
                (model.surface_l_lsun(), model.surface_r_rsun());
            assert!(l_lsun.is_finite() && l_lsun > prev_l);
            assert!(r_rsun.is_finite() && r_rsun > prev_r);
            (prev_l, prev_r) = (l_lsun, r_rsun);
        }
    }

    #[test]
    fn test_interp_q_uses_piecewise_linear_top_down_face_q_without_outer_clamp()
    {
        // Outer zones with tiny dq ~ 1e-12 must match upstream linear face-q
        // interpolation (`star/private/star_utils.f90:729-733, 1232-1265`)
        // without artificial abscissa shifting.
        let dq1 = [1.0e-12, 2.0e-12, 0.5 - 3.0e-12, 0.5];
        let v1 = [10.0, 20.0, 40.0, 80.0];
        let dq2 = [0.5e-12, 1.5e-12, 0.25, 0.75 - 2.0e-12];
        let out = interp_q(&dq1, &v1, &dq2).expect("interp_q");
        assert_eq!(out.len(), dq2.len());
        // k=0: q2[0] = 1.0 falls in [q1[1], q1[0]] where q1[1] = 1.0 - dq1[0],
        // so upstream's `(1.0 - q1[1]) / dq1[0]` reflects IEEE-754 rounding
        // when dq1[0] ~ 1e-12 (`star_utils.f90:729-733, 1256`).
        let q1_1 = 1.0 - dq1[0];
        let alfa0 = (1.0 - q1_1) / dq1[0];
        let expected0 = alfa0 * v1[0] + (1.0 - alfa0) * v1[1];
        assert_eq!(out[0].to_bits(), expected0.to_bits());
        // Exact binary-representable dq1[0] gives exact v1[0] at k=0.
        let out_exact = interp_q(&[0.25, 0.75], &[10.0, 20.0], &[0.25, 0.75])
            .expect("interp_q exact");
        assert_eq!(out_exact[0], 10.0);
        // k=1: q2[1] = 1 - 0.5e-12 lies halfway in [q1[1], q1[0]]
        let alfa1 = ((1.0 - dq2[0]) - q1_1) / dq1[0];
        let expected1 = alfa1 * v1[0] + (1.0 - alfa1) * v1[1];
        assert!((out[1] - expected1).abs() <= 1.0e-14);
        // k=2: q2[2] = 1 - 2e-12 lies halfway in [q1[2], q1[1]]
        let q1_2 = q1_1 - dq1[1];
        let alfa2 = ((1.0 - dq2[0] - dq2[1]) - q1_2) / dq1[1];
        let expected2 = alfa2 * v1[1] + (1.0 - alfa2) * v1[2];
        assert!((out[2] - expected2).abs() <= 1.0e-14);
        // k=3: q2[3] = 0.75 - 2e-12 > q1[3] = 0.5, lies inside [q1[3], q1[2]]
        let q2_3 = (1.0 - dq2[0] - dq2[1]) - dq2[2];
        let q1_3 = q1_2 - dq1[2];
        let alfa3 = (q2_3 - q1_3) / dq1[2];
        let expected3 = alfa3 * v1[2] + (1.0 - alfa3) * v1[3];
        assert!((out[3] - expected3).abs() <= 1.0e-14);

        // Center clamp (`qval <= q1[nz1 - 1]` -> `v1[nz1 - 1]`)
        let dq_center = [0.6, 0.4];
        let out_c = interp_q(&dq1, &v1, &dq_center).expect("interp_q center");
        assert_eq!(out_c[1], v1[3]);
    }

    #[test]
    fn test_zams_1p5_msun_uses_8_decimal_index_masses_and_model_number_zero() {
        let cat = ZamsCatalog::builtin();
        let e1 = &cat.models[10];
        let e2 = &cat.models[11];
        assert_eq!(e1.index_m_msun, 1.25892541);
        assert_eq!(e2.index_m_msun, 1.58489319);
        assert_eq!(e1.nz, 2402);
        assert_eq!(e2.nz, 2490);

        let model = load_zams_model(
            StarControls {
                initial_mass: 1.5,
                initial_z: 0.02,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1.5 Msun ZAMS");
        assert_eq!(model.model_number, 0);
        assert_eq!(model.nz, 2402);

        // Verify exact bit-for-bit match against upstream formula with
        // 8-decimal index-table masses (`init_model.f90:210, 254, 444-446`)
        // and `interp_q` (`star_utils.f90:729-733, 1256`), matching MESA
        // 26.4.1 Model 0 zone 1 (`lnd = -17.37923774243046`).
        let lnm1 = flexmesa_math::log(1.25892541);
        let lnm2 = flexmesa_math::log(1.58489319);
        let alfa = (flexmesa_math::log(1.5) - lnm2) / (lnm1 - lnm2);
        let beta = 1.0 - alfa;
        let alfa_q0 = (1.0 - (1.0 - e2.dq[0])) / e2.dq[0];
        let e2_lnd0 = alfa_q0 * e2.lnd[0] + (1.0 - alfa_q0) * e2.lnd[1];
        let expected_lnd0 = alfa * e1.lnd[0] + beta * e2_lnd0;
        assert_eq!(model.lnd[0].to_bits(), expected_lnd0.to_bits());
        assert!((model.lnd[0] - (-17.379237742430462)).abs() <= 1.0e-15);
    }
}
