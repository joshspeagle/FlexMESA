// SPDX-License-Identifier: LGPL-3.0-or-later
//! Zero-Age Main Sequence (ZAMS) model catalog, `interp_q` Lagrangian
//! interpolation, and `load_zams_model` (`star/private/init_model.f90`).

use std::sync::OnceLock;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{M_SUN, SECYER};
use flexmesa_interp1d::{interp_pm, interp_value};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;

const BUILTIN_ZAMS_BYTES: &[u8] =
    include_bytes!("../data/zams_z2m2_y28_subset.bin");

static BUILTIN_ZAMS_CATALOG: OnceLock<ZamsCatalog> = OnceLock::new();

/// Single pre-computed ZAMS model entry from `zams_z2m2_y28.data`.
#[derive(Debug, Clone)]
pub struct ZamsEntry {
    /// Stellar mass in solar masses ($M_\odot$).
    pub m_msun: f64,
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
            if *pos + 4 > bytes.len() {
                return Err(StarError::IoError(String::from(
                    "unexpected EOF reading u32 in ZAMS binary",
                )));
            }
            let val =
                u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap())
                    as usize;
            *pos += 4;
            Ok(val)
        };
        let read_f64 = |pos: &mut usize| -> Result<f64, StarError> {
            if *pos + 8 > bytes.len() {
                return Err(StarError::IoError(String::from(
                    "unexpected EOF reading f64 in ZAMS binary",
                )));
            }
            let val =
                f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(val)
        };
        let read_f64_vec =
            |pos: &mut usize, n: usize| -> Result<Vec<f64>, StarError> {
                if *pos + 8 * n > bytes.len() {
                    return Err(StarError::IoError(String::from(
                        "unexpected EOF reading f64 slice in ZAMS binary",
                    )));
                }
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let val = f64::from_le_bytes(
                        bytes[*pos..*pos + 8].try_into().unwrap(),
                    );
                    *pos += 8;
                    v.push(val);
                }
                Ok(v)
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
            .map(|name| catalog.get_nuclide_index(name).unwrap_or(1))
            .collect();

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
                xa.extend_from_slice(&row_vals[b + 5..b + 5 + species]);
            }
            models.push(ZamsEntry {
                m_msun,
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
            if trimmed.starts_with("initial_Z") {
                if let Some(tok) = trimmed.split_whitespace().nth(1) {
                    initial_z = parse_fortran_f64(tok)?;
                }
            } else if trimmed.starts_with("initial_Y") {
                if let Some(tok) = trimmed.split_whitespace().nth(1) {
                    initial_y = parse_fortran_f64(tok)?;
                }
            } else if trimmed.starts_with("M/Msun") {
                break;
            }
        }
        for line in lines.by_ref() {
            if line.trim() == "1" {
                break;
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
                        let lnd_pos =
                            cols.iter().position(|&c| c == "lnd").unwrap_or(0);
                        let iso_cols = &cols[lnd_pos + 5..];
                        let species = iso_cols.len();
                        let chem_id: Vec<usize> = iso_cols
                            .iter()
                            .map(|name| {
                                catalog.get_nuclide_index(name).unwrap_or(1)
                            })
                            .collect();
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
                            for j in 0..species {
                                xa.push(parse_fortran_f64(toks[6 + j])?);
                            }
                        }
                        models.push(ZamsEntry {
                            m_msun,
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

/// Parses a Fortran floating-point string (supporting `D`/`d` exponents).
pub(crate) fn parse_fortran_f64(s: &str) -> Result<f64, StarError> {
    let cleaned = if s.contains(['D', 'd']) {
        s.replace(['D', 'd'], "e")
    } else {
        s.to_string()
    };
    cleaned
        .parse::<f64>()
        .map_err(|e| StarError::IoError(format!("invalid float '{s}': {e}")))
}

/// Computes the initial ZAMS timestep in years (`yrs_for_init_timestep` in
/// `star/private/init_model.f90:472-498`).
#[must_use]
pub fn yrs_for_init_timestep(m_msun: f64) -> f64 {
    if m_msun < 0.3 {
        1.0e7
    } else if m_msun <= 30.0 {
        1.0e5 / m_msun.powf(2.5)
    } else {
        20.0 * (30.0 / m_msun).powf(1.5)
    }
}

fn mid_q_from_dq(dq: &[f64]) -> Vec<f64> {
    let nz = dq.len();
    if nz == 0 {
        return Vec::new();
    }
    let mut q = vec![0.0; nz];
    q[0] = 1.0;
    for k in 1..nz.saturating_sub(1) {
        q[k] = q[k - 1] - dq[k - 1];
    }
    if nz >= 2 {
        q[nz - 1] = dq[nz - 1];
    }
    for k in 0..nz {
        q[k] -= 0.5 * dq[k];
    }
    q
}

/// Interpolates cell-centered or face-centered profile `v1` from grid `dq1`
/// onto grid `dq2` using monotone piecewise cubic interpolation (`interp_pm`)
/// in mid-zone enclosed mass fraction $q_{\mathrm{mid},k} = q_k - \tfrac{1}{2}\Delta q_k$
/// (`interp_q` in `star/private/init_model.f90:397-468`).
///
/// # Errors
/// Returns [`StarError`] if spline construction fails.
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
    let q1 = mid_q_from_dq(dq1);
    let q2 = mid_q_from_dq(dq2);

    // Reverse to strictly increasing x for interp_pm
    let mut x_inc: Vec<f64> = q1.into_iter().rev().collect();
    let y_inc: Vec<f64> = v1.iter().copied().rev().collect();
    // Ensure strictly increasing abscissae in case of tiny duplicate dq
    for i in 1..x_inc.len() {
        if x_inc[i] <= x_inc[i - 1] {
            x_inc[i] = x_inc[i - 1] + 1.0e-15;
        }
    }

    let coeffs = interp_pm(&x_inc, &y_inc).map_err(|e| {
        StarError::InvalidInput(format!("interp_pm error: {e}"))
    })?;

    let mut out = Vec::with_capacity(nz2);
    for &xq in &q2 {
        let x_clamped = xq.clamp(x_inc[0], x_inc[nz1 - 1]);
        let val = interp_value(&x_inc, &coeffs, x_clamped).map_err(|e| {
            StarError::InvalidInput(format!("interp_value error: {e}"))
        })?;
        out.push(val);
    }
    Ok(out)
}

/// Constructs an initial [`StarModel`] by logarithmic mass interpolation in a
/// [`ZamsCatalog`] (`get_zams_model` / `get1_mass` in
/// `star/private/init_model.f90:65-394`).
///
/// # Errors
/// Returns [`StarError`] if `controls.initial_mass` is outside the catalog
/// mass range or interpolation fails.
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
    let m_min = catalog.models.first().unwrap().m_msun;
    let m_max = catalog.models.last().unwrap().m_msun;
    if m_target < m_min * (1.0 - 1.0e-7) || m_target > m_max * (1.0 + 1.0e-7) {
        return Err(StarError::InvalidInput(format!(
            "initial_mass={m_target} outside ZAMS catalog range [{m_min}, {m_max}]"
        )));
    }

    // Locate bracketing models m1 <= m_target <= m2
    let mut idx1 = 0;
    for (i, entry) in catalog.models.iter().enumerate() {
        if entry.m_msun <= m_target * (1.0 + 1.0e-9) {
            idx1 = i;
        } else {
            break;
        }
    }
    let idx2 = (idx1 + 1).min(catalog.models.len() - 1);
    let e1 = &catalog.models[idx1];
    let e2 = &catalog.models[idx2];

    let nz = e1.nz;
    let species = e1.species;
    let mut model = StarModel::new(nz, e1.chem_id.clone(), job, controls);
    model.dq.clone_from(&e1.dq);

    if idx1 == idx2 || (m_target - e1.m_msun).abs() <= 1.0e-7 * e1.m_msun {
        model.lnd.clone_from(&e1.lnd);
        model.ln_t.clone_from(&e1.ln_t);
        model.ln_r.clone_from(&e1.ln_r);
        model.lum.clone_from(&e1.lum);
        model.xa.clone_from(&e1.xa);
    } else {
        // Upstream MESA `get1_mass`:
        // alfa = (log(m) - log(m2)) / (log(m1) - log(m2)), beta = 1 - alfa
        let alfa = (m_target.ln() - e2.m_msun.ln())
            / (e1.m_msun.ln() - e2.m_msun.ln());
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
                model.xa[k * species + j] =
                    (alfa * x1_kj + beta * x2_interp[k]).max(0.0);
            }
        }
    }

    // Normalize cell mass fractions sum(xa[k, :]) == 1
    for k in 0..nz {
        let row = model.cell_xa_mut(k);
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            for x in row.iter_mut() {
                *x /= sum;
            }
        }
    }

    model.mstar = m_target * M_SUN;
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    model.model_number = 1;
    model.star_age = 0.0;
    let dt_sec = yrs_for_init_timestep(m_target) * SECYER;
    model.dt = dt_sec;
    model.dt_next = dt_sec;
    model.dt_old = dt_sec;
    model.finish_init_model()?;

    Ok(model)
}

/// Loads a ZAMS [`StarModel`] from the built-in `zams_z2m2_y28` catalog.
///
/// # Errors
/// Returns [`StarError`] if `controls.initial_mass` is out of bounds.
pub fn load_zams_model(
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    load_zams_model_from_catalog(ZamsCatalog::builtin(), controls, job)
}
