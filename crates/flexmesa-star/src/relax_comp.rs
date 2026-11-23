// SPDX-License-Identifier: LGPL-3.0-or-later
//! Composition arithmetic for the composition relaxes: `set_y`, `set_z`,
//! `eval_current_y`/`eval_current_z` and the `do_relax_composition` target
//! profile (`star/private/adjust_xyz.f90`, `star/private/star_utils.f90`,
//! `star/private/relax.f90:316-548`). The evolve loops are in
//! `crate::relax_mass_comp`.

use crate::controls::StarError;
use crate::model::StarModel;

/// Validates a 0-based inclusive zone range `[nzlo, nzhi]` and backing arrays
/// (`dm` in g, `xa` row-major `[nz * species]`).
fn validate_zone_range(
    model: &StarModel,
    nzlo: usize,
    nzhi: usize,
) -> Result<(), StarError> {
    if model.nz == 0 || model.species == 0 {
        return Err(StarError::InvalidInput(
            "composition helper requires nonzero nz and species".to_string(),
        ));
    }
    if nzlo > nzhi || nzhi >= model.nz {
        return Err(StarError::InvalidInput(format!(
            "invalid zone range [{nzlo}, {nzhi}] for nz={}",
            model.nz
        )));
    }
    if model.dm.len() < model.nz {
        return Err(StarError::InvalidInput(format!(
            "dm length {} < nz {}",
            model.dm.len(),
            model.nz
        )));
    }
    let expected_xa = model.nz * model.species;
    if model.xa.len() != expected_xa {
        return Err(StarError::InvalidInput(format!(
            "xa length {} != nz * species ({expected_xa})",
            model.xa.len()
        )));
    }
    Ok(())
}

/// Resolves the 0-based species index for `iso_name` via `NuclideCatalog` +
/// `model.net_iso`, falling back to `chem_id` and `iso_names`.
fn resolve_species_idx(model: &StarModel, iso_name: &str) -> Option<usize> {
    if let Some(cid) =
        flexmesa_chem::NuclideCatalog::builtin_ref().get_nuclide_index(iso_name)
    {
        if let Some(idx) = model.net_iso.get(cid).copied().flatten() {
            if idx < model.species {
                return Some(idx);
            }
        }
        if let Some(idx) = model.chem_id.iter().position(|&c| c == cid) {
            return Some(idx);
        }
    }
    model
        .iso_names
        .iter()
        .position(|name| name.eq_ignore_ascii_case(iso_name))
}

/// Computes the mass-weighted average helium (`Y`) and metal (`Z`) mass
/// fractions (dimensionless) over the 0-based inclusive zone range
/// `nzlo..=nzhi`.
///
/// Matches `get_average_Y_and_Z` in upstream MESA `26.4.1`
/// `star/private/star_utils.f90:215-242`, including `h1 + h2` hydrogen summation
/// and `he4`-then-`he3` helium summation order with `total_mass -
/// (total_mass_h + total_mass_he)`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if `[nzlo, nzhi]` is invalid, array
/// lengths are inconsistent, or the zone-range mass sum is non-positive or
/// non-finite.
pub fn eval_current_y_and_z(
    model: &StarModel,
    nzlo: usize,
    nzhi: usize,
) -> Result<(f64, f64), StarError> {
    validate_zone_range(model, nzlo, nzhi)?;

    let h1_idx = resolve_species_idx(model, "h1");
    let h2_idx = resolve_species_idx(model, "h2");
    let he3_idx = resolve_species_idx(model, "he3");
    let he4_idx = resolve_species_idx(model, "he4");

    let mut total_mass = 0.0_f64;
    let mut total_mass_h = 0.0_f64;
    let mut total_mass_he = 0.0_f64;
    for k in nzlo..=nzhi {
        let cell_mass = model.dm[k];
        total_mass += cell_mass;
        let xa_k = model.cell_xa(k);
        if let Some(j_h1) = h1_idx {
            total_mass_h += cell_mass * xa_k[j_h1];
        }
        if let Some(j_h2) = h2_idx {
            total_mass_h += cell_mass * xa_k[j_h2];
        }
        if let Some(j_he4) = he4_idx {
            total_mass_he += cell_mass * xa_k[j_he4];
        }
        if let Some(j_he3) = he3_idx {
            total_mass_he += cell_mass * xa_k[j_he3];
        }
    }
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "eval_current_y_and_z requires positive finite mass, got {total_mass}"
        )));
    }
    let total_mass_z = total_mass - (total_mass_h + total_mass_he);
    let y_avg = total_mass_he / total_mass;
    let z_avg = total_mass_z / total_mass;
    Ok((y_avg, z_avg))
}

/// Computes the mass-weighted average helium mass fraction (`Y`, dimensionless)
/// over the 0-based inclusive zone range `nzlo..=nzhi`.
///
/// Matches `eval_current_y` in `star/private/star_utils.f90:244-253`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] under the same conditions as
/// [`eval_current_y_and_z`].
pub fn eval_current_y(
    model: &StarModel,
    nzlo: usize,
    nzhi: usize,
) -> Result<f64, StarError> {
    Ok(eval_current_y_and_z(model, nzlo, nzhi)?.0)
}

/// Computes the mass-weighted average metal mass fraction (`Z`, dimensionless)
/// over the 0-based inclusive zone range `nzlo..=nzhi`.
///
/// Matches `eval_current_z` in `star/private/star_utils.f90:254-263`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] under the same conditions as
/// [`eval_current_y_and_z`].
pub fn eval_current_z(
    model: &StarModel,
    nzlo: usize,
    nzhi: usize,
) -> Result<f64, StarError> {
    Ok(eval_current_y_and_z(model, nzlo, nzhi)?.1)
}

/// Computes the mass-weighted average mass fraction (dimensionless) of a
/// single species `iso_idx` over the 0-based inclusive zone range
/// `nzlo..=nzhi`.
///
/// Returns `0.0` when `iso_idx` is `None` (absent species), matching
/// `eval_current_abundance` in `star/private/star_utils.f90:264-287`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if `[nzlo, nzhi]` is invalid, `iso_idx`
/// is out of bounds (`>= model.species`), or total mass is non-positive.
pub fn eval_current_abundance(
    model: &StarModel,
    iso_idx: Option<usize>,
    nzlo: usize,
    nzhi: usize,
) -> Result<f64, StarError> {
    validate_zone_range(model, nzlo, nzhi)?;
    let Some(j) = iso_idx else {
        return Ok(0.0);
    };
    if j >= model.species {
        return Err(StarError::InvalidInput(format!(
            "species index {j} out of bounds for species={}",
            model.species
        )));
    }
    let mut total_mass = 0.0_f64;
    let mut jmass = 0.0_f64;
    for k in nzlo..=nzhi {
        let cell_mass = model.dm[k];
        total_mass += cell_mass;
        jmass += cell_mass * model.cell_xa(k)[j];
    }
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "eval_current_abundance requires positive finite mass, got {total_mass}"
        )));
    }
    Ok(jmass / total_mass)
}

/// Adjusts two species' mass fractions across `nzlo..=nzhi` (0-based inclusive)
/// so their ratio equals `ratio` while holding their zone sum fixed.
///
/// Matches `set_abundance_ratio` in `star/private/adjust_xyz.f90:684-713`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if `[nzlo, nzhi]` is invalid, either
/// species is `None`, indices coincide or exceed `model.species`, or `ratio` is
/// negative or non-finite.
pub fn set_abundance_ratio(
    model: &mut StarModel,
    iso_idx_1: Option<usize>,
    iso_idx_2: Option<usize>,
    ratio: f64,
    nzlo: usize,
    nzhi: usize,
) -> Result<(), StarError> {
    validate_zone_range(model, nzlo, nzhi)?;
    if !ratio.is_finite() || ratio < 0.0 {
        return Err(StarError::InvalidInput(format!(
            "set_abundance_ratio requires finite ratio >= 0, got {ratio}"
        )));
    }
    let j1 = iso_idx_1.ok_or_else(|| {
        StarError::InvalidInput(
            "set_abundance_ratio requires first species to exist in net"
                .to_string(),
        )
    })?;
    let j2 = iso_idx_2.ok_or_else(|| {
        StarError::InvalidInput(
            "set_abundance_ratio requires second species to exist in net"
                .to_string(),
        )
    })?;
    if j1 >= model.species || j2 >= model.species || j1 == j2 {
        return Err(StarError::InvalidInput(format!(
            "invalid species indices ({j1}, {j2}) for species={}",
            model.species
        )));
    }
    for k in nzlo..=nzhi {
        let xa_k = model.cell_xa_mut(k);
        let xsum = xa_k[j1] + xa_k[j2];
        xa_k[j2] = xsum / (1.0 + ratio);
        xa_k[j1] = xsum - xa_k[j2];
    }
    Ok(())
}

/// Sets the helium mass fraction `y` (dimensionless) over the 0-based
/// inclusive zone range `nzlo..=nzhi` while preserving `he3` and `Z`.
///
/// Matches `set_y` in upstream MESA `26.4.1` `star/private/adjust_xyz.f90:647-681`,
/// including the exact upstream control flow when `y == 0.0` or
/// `desired_xhe4 <= 0.0`. Callers that require updated thermodynamic or
/// hydrostatic derived variables should follow with `set_hydro_vars`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if `[nzlo, nzhi]` is invalid, `y` is
/// outside `[0, 1]` or non-finite, or `h1`/`he4` are absent from the network.
pub fn set_y(
    model: &mut StarModel,
    y: f64,
    nzlo: usize,
    nzhi: usize,
) -> Result<(), StarError> {
    validate_zone_range(model, nzlo, nzhi)?;
    if !y.is_finite() || !(0.0..=1.0).contains(&y) {
        return Err(StarError::InvalidInput(format!(
            "set_y requires finite y in [0, 1], got {y}"
        )));
    }
    let h1_idx = resolve_species_idx(model, "h1");
    let he3_idx = resolve_species_idx(model, "he3");
    let he4_idx = resolve_species_idx(model, "he4");

    // Verbatim upstream flow (`adjust_xyz.f90:659-680`): `y == 0` adjusts he4/h1
    // to 0 then proceeds through the average-abundance computation below.
    if y == 0.0 {
        set_abundance_ratio(model, he4_idx, h1_idx, 0.0, nzlo, nzhi)?;
    }

    let xh1 = eval_current_abundance(model, h1_idx, nzlo, nzhi)?;
    let xhe3 = eval_current_abundance(model, he3_idx, nzlo, nzhi)?;
    let xhe4 = eval_current_abundance(model, he4_idx, nzlo, nzhi)?;
    let z = 1.0 - (xh1 + xhe3 + xhe4);
    let desired_xhe4 = y - xhe3;
    let desired_xh1 = 1.0 - z - y;

    let ratio = if desired_xh1 <= 0.0 {
        0.0
    } else if desired_xhe4 <= 0.0 {
        1.0
    } else {
        desired_xh1 / desired_xhe4
    };
    set_abundance_ratio(model, h1_idx, he4_idx, ratio, nzlo, nzhi)
}

/// Rescales metal mass fraction across the 0-based inclusive zone range
/// `nzlo..=nzhi` so the mass-weighted average equals `new_z` (dimensionless).
///
/// Exact translation of `set_z` in upstream MESA `26.4.1`
/// `star/private/adjust_xyz.f90:1223-1353`:
/// 1. Cleans all zones via `clean_up_fractions_slice` (`max_sum_abs = 10`,
///    `xsum_tol = 1e-2`), clamping species to `[1e-99, 1]`.
/// 2. Evaluates mass-weighted `current_z` over `nzlo..=nzhi` and sets
///    `frac_z = new_z / current_z`.
/// 3. In `convert`, accumulates `old_xy` and `sumfracs` in reverse species order
///    `(0..species).rev()` across `h1`, `he3`, `he4` (`h2` scales with metals,
///    matching `adjust_xyz.f90:1308`), handling `old_z >= 1.0` via `(h1, he4)`.
/// 4. Re-checks post-conversion `eval_current_z` within `1e-8` and updates
///    `model.controls.initial_z` when `model.doing_first_model_of_run` is set.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if `[nzlo, nzhi]` is invalid, `new_z` is
/// outside `[0, 1]` or non-finite, `clean_up_fractions_slice` rejects a bad
/// row sum, `current_z <= 0` with `new_z > 0`, or the post-conversion `Z`
/// verification fails.
pub fn set_z(
    model: &mut StarModel,
    new_z: f64,
    nzlo: usize,
    nzhi: usize,
) -> Result<(), StarError> {
    validate_zone_range(model, nzlo, nzhi)?;
    if !new_z.is_finite() || !(0.0..=1.0).contains(&new_z) {
        return Err(StarError::InvalidInput(format!(
            "set_z requires finite new_z in [0, 1], got {new_z}"
        )));
    }

    crate::mesh_remap::clean_up_fractions_slice(
        &mut model.xa,
        model.nz,
        model.species,
        10.0,
        1.0e-2,
    )?;

    let current_z = eval_current_z(model, nzlo, nzhi)?;
    if current_z <= 0.0 {
        if new_z > 0.0 {
            return Err(StarError::InvalidInput(format!(
                "set_z cannot adjust current_z = {current_z} to new_z = {new_z}"
            )));
        }
        return Ok(());
    }

    let frac_z = new_z / current_z;
    if (1.0 - frac_z).abs() < 1.0e-20 {
        return Ok(());
    }

    let h1_idx = resolve_species_idx(model, "h1");
    let he3_idx = resolve_species_idx(model, "he3");
    let he4_idx = resolve_species_idx(model, "he4");
    let species = model.species;

    for k in nzlo..=nzhi {
        let xa_k = model.cell_xa_mut(k);
        let mut old_xy = 0.0_f64;
        for i in (0..species).rev() {
            if Some(i) == h1_idx || Some(i) == he3_idx || Some(i) == he4_idx {
                old_xy += xa_k[i];
            }
        }
        let old_z = 1.0 - old_xy;

        let mut sum_z = 0.0_f64;
        for i in (0..species).rev() {
            if Some(i) == h1_idx || Some(i) == he3_idx || Some(i) == he4_idx {
                continue;
            }
            xa_k[i] *= frac_z;
            sum_z += xa_k[i];
        }

        if old_z < 1.0 {
            let frac_xy = (1.0 - old_z * frac_z) / old_xy;
            if let Some(j) = h1_idx {
                xa_k[j] *= frac_xy;
            }
            if let Some(j) = he3_idx {
                xa_k[j] *= frac_xy;
            }
            if let Some(j) = he4_idx {
                xa_k[j] *= frac_xy;
            }
        } else {
            let half_rem = (1.0 - sum_z) / 2.0;
            if let Some(j) = h1_idx {
                xa_k[j] = half_rem;
            }
            if let Some(j) = he3_idx {
                xa_k[j] = 0.0;
            }
            if let Some(j) = he4_idx {
                xa_k[j] = half_rem;
            }
        }

        let mut sumfracs = 0.0_f64;
        for i in (0..species).rev() {
            sumfracs += xa_k[i];
        }
        for val in xa_k.iter_mut() {
            *val /= sumfracs;
        }
    }

    let post_z = eval_current_z(model, nzlo, nzhi)?;
    if !post_z.is_finite() || (post_z - new_z).abs() > 1.0e-8 {
        return Err(StarError::InvalidInput(format!(
            "set_z failed verification: target={new_z}, current_z={post_z}"
        )));
    }
    if model.doing_first_model_of_run {
        model.controls.initial_z = new_z;
    }
    Ok(())
}

/// Tabulated target composition profile for `do_relax_composition`
/// (`run_star_support.f90:2793-2845` and `relax.f90:316-548`).
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionProfile {
    /// Number of radial target points (`num_pts >= 2`).
    pub num_pts: usize,
    /// Number of species per point (`species >= 1`).
    pub species: usize,
    /// Exterior mass coordinate $x_q = 1 - q$ (dimensionless, monotonic, `len = num_pts`).
    pub xq: Vec<f64>,
    /// Row-major target mass fractions (`len = num_pts * species`, dimensionless).
    pub xa: Vec<f64>,
}

/// Parses an ASCII composition-relaxation profile (`relax_composition_filename`)
/// matching `do_relax_initial_composition` in MESA `26.4.1`
/// `star/job/run_star_support.f90:2810-2839`.
///
/// The first non-empty record gives `num_pts num_species`, followed by
/// `num_pts` lines each containing `xq` (exterior mass coordinate $1 - q$) and
/// `num_species` mass fractions (`D`/`d` exponents supported).
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] if the header is malformed,
/// `num_pts < 2`, `num_species == 0` or mismatches `expected_species`, any
/// numeric token fails to parse, or row counts do not match `num_pts`.
pub fn parse_composition_profile_str(
    content: &str,
    expected_species: Option<usize>,
) -> Result<CompositionProfile, StarError> {
    let mut lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());

    let header = lines.next().ok_or_else(|| {
        StarError::InvalidInput(
            "composition profile is empty; expected num_pts num_species"
                .to_string(),
        )
    })?;
    let mut hdr_tokens = header.split_whitespace();
    let num_pts: usize = hdr_tokens
        .next()
        .ok_or_else(|| {
            StarError::InvalidInput(
                "missing num_pts in composition profile header".to_string(),
            )
        })?
        .parse()
        .map_err(|e| {
            StarError::InvalidInput(format!(
                "invalid num_pts in composition profile header: {e}"
            ))
        })?;
    let species: usize = hdr_tokens
        .next()
        .ok_or_else(|| {
            StarError::InvalidInput(
                "missing num_species in composition profile header".to_string(),
            )
        })?
        .parse()
        .map_err(|e| {
            StarError::InvalidInput(format!(
                "invalid num_species in composition profile header: {e}"
            ))
        })?;

    if num_pts < 2 || species == 0 {
        return Err(StarError::InvalidInput(format!(
            "composition profile requires num_pts >= 2 and species >= 1, \
             got num_pts={num_pts}, species={species}"
        )));
    }
    if let Some(exp) = expected_species {
        if species != exp {
            return Err(StarError::InvalidInput(format!(
                "composition profile has {species} species, expected {exp}"
            )));
        }
    }

    let mut xq = Vec::with_capacity(num_pts);
    let mut xa = Vec::with_capacity(num_pts * species);
    for pt_idx in 0..num_pts {
        let row = lines.next().ok_or_else(|| {
            StarError::InvalidInput(format!(
                "composition profile truncated at row {} of {num_pts}",
                pt_idx + 1
            ))
        })?;
        let mut tokens = row.split_whitespace();
        let xq_tok = tokens.next().ok_or_else(|| {
            StarError::InvalidInput(format!(
                "missing xq at composition profile row {}",
                pt_idx + 1
            ))
        })?;
        let xq_val = crate::zams::parse_fortran_f64(xq_tok)?;
        if !xq_val.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "non-finite xq at composition profile row {}",
                pt_idx + 1
            )));
        }
        xq.push(xq_val);

        for j in 0..species {
            let tok = tokens.next().ok_or_else(|| {
                StarError::InvalidInput(format!(
                    "missing species {j} at composition profile row {}",
                    pt_idx + 1
                ))
            })?;
            let val = crate::zams::parse_fortran_f64(tok)?;
            if !val.is_finite() {
                return Err(StarError::InvalidInput(format!(
                    "non-finite species {j} at composition profile row {}",
                    pt_idx + 1
                )));
            }
            xa.push(val);
        }
    }

    Ok(CompositionProfile {
        num_pts,
        species,
        xq,
        xa,
    })
}

/// Precomputed Steffen (1990) piecewise-monotonic cubic interpolants (`interp_pm`)
/// for per-step composition profile relaxation (`store_rpar` and `adjust_xa`
/// in upstream MESA `26.4.1` `star/private/relax.f90:398-509`).
#[derive(Debug, Clone, PartialEq)]
pub struct RelaxCompositionInterpolants {
    xq: Vec<f64>,
    species: usize,
    coeffs: Vec<Vec<flexmesa_interp1d::CubicCoeffs>>,
}

impl RelaxCompositionInterpolants {
    /// Constructs per-species Steffen piecewise-monotonic cubic spline tables
    /// matching `store_rpar` in `star/private/relax.f90:398-419`.
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidInput`] if array dimensions are
    /// inconsistent or if `flexmesa_interp1d::interp_pm` rejects the knot grid
    /// (e.g., fewer than 2 points or non-monotonic `xq`).
    pub fn from_profile(
        profile: &CompositionProfile,
    ) -> Result<Self, StarError> {
        let n = profile.num_pts;
        let species = profile.species;
        if profile.xq.len() != n || profile.xa.len() != n * species || n < 2 {
            return Err(StarError::InvalidInput(
                "invalid CompositionProfile dimensions for interp_pm"
                    .to_string(),
            ));
        }

        let mut coeffs = Vec::with_capacity(species);
        let mut y_col = vec![0.0_f64; n];
        for j in 0..species {
            for k in 0..n {
                y_col[k] = profile.xa[k * species + j];
            }
            let c = flexmesa_interp1d::interp_pm(&profile.xq, &y_col).map_err(
                |e| {
                    StarError::InvalidInput(format!(
                        "composition interp_pm failed for species {j}: {e}"
                    ))
                },
            )?;
            coeffs.push(c);
        }

        Ok(Self {
            xq: profile.xq.clone(),
            species,
            coeffs,
        })
    }

    /// Number of target profile knots.
    #[must_use]
    pub fn num_pts(&self) -> usize {
        self.xq.len()
    }

    /// Number of species represented by the interpolant tables.
    #[must_use]
    pub fn species(&self) -> usize {
        self.species
    }

    /// Target exterior mass coordinates `xq = 1 - q` (dimensionless).
    #[must_use]
    pub fn xq(&self) -> &[f64] {
        &self.xq
    }

    /// Computes the per-step blend factor `lambda` (dimensionless in `[0, 1]`)
    /// matching `relax_composition_adjust_model` in
    /// `star/private/relax.f90:459-464`.
    ///
    /// When `timescale_for_relax_composition < 0.0`, linearly ramps `lambda`
    /// over `num_steps_to_use` steps starting at `starting_model_number`;
    /// otherwise uses `min(1.0, dt / timescale_for_relax_composition)`.
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidInput`] if `dt` or `timescale` is
    /// non-finite, or `dt < 0.0` (or `dt == 0.0 && timescale == 0.0`) in the
    /// positive-timescale branch.
    pub fn eval_lambda(
        dt: f64,
        timescale_for_relax_composition: f64,
        model_number: usize,
        starting_model_number: usize,
        num_steps_to_use: usize,
    ) -> Result<f64, StarError> {
        if !timescale_for_relax_composition.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "non-finite timescale_for_relax_composition: \
                 {timescale_for_relax_composition}"
            )));
        }
        if timescale_for_relax_composition < 0.0 {
            let numer =
                (model_number as f64) - (starting_model_number as f64) - 1.0;
            let denom = ((num_steps_to_use as f64) - 1.0).max(1.0);
            return Ok((numer / denom).clamp(0.0, 1.0));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(StarError::InvalidInput(format!(
                "eval_lambda requires finite dt >= 0, got {dt}"
            )));
        }
        if timescale_for_relax_composition == 0.0 {
            if dt == 0.0 {
                return Err(StarError::InvalidInput(
                    "eval_lambda encountered dt == 0 and timescale == 0"
                        .to_string(),
                ));
            }
            return Ok(1.0);
        }
        Ok((dt / timescale_for_relax_composition).min(1.0))
    }

    /// Interpolates the target composition profile onto cell centers `xq_star`
    /// of `model` (`relax.f90:489-502`), clamping negative spline undershoots
    /// to `0.0`.
    ///
    /// Returns row-major target abundances (`len = model.nz * model.species`,
    /// dimensionless).
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidInput`] if `model.species != self.species`,
    /// `model.nz == 0`, `model.dq.len() < model.nz`, or `interp_values` fails.
    pub fn interpolate_target_to_model(
        &self,
        model: &StarModel,
    ) -> Result<Vec<f64>, StarError> {
        if model.nz == 0 || model.species != self.species {
            return Err(StarError::InvalidInput(format!(
                "model (nz={}, species={}) incompatible with interpolant \
                 species={}",
                model.nz, model.species, self.species
            )));
        }
        if model.dq.len() < model.nz {
            return Err(StarError::InvalidInput(format!(
                "dq length {} < nz {}",
                model.dq.len(),
                model.nz
            )));
        }

        let nz = model.nz;
        let species = model.species;
        let mut xq_star = vec![0.0_f64; nz];
        xq_star[0] = 0.5 * model.dq[0];
        for k in 1..nz {
            xq_star[k] = xq_star[k - 1] + 0.5 * (model.dq[k] + model.dq[k - 1]);
        }

        let mut vals = vec![0.0_f64; nz * species];
        for j in 0..species {
            let interp_j = flexmesa_interp1d::interp_values(
                &self.xq,
                &self.coeffs[j],
                &xq_star,
            )
            .map_err(|e| {
                StarError::InvalidInput(format!(
                    "composition interp_values failed for species {j}: {e}"
                ))
            })?;
            for k in 0..nz {
                let v = interp_j[k];
                vals[k * species + j] = if v < 0.0 { 0.0 } else { v };
            }
        }
        Ok(vals)
    }

    /// Applies one relaxation blend step `xa = (1 - lambda) * xa + lambda * vals`
    /// across `model.xa` and returns the mean absolute residual `avg_err`,
    /// matching `adjust_xa` in `star/private/relax.f90:478-509`.
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidInput`] if `lambda` is outside `[0, 1]` or
    /// non-finite, `model.xa.len() != model.nz * model.species`, or target
    /// interpolation fails.
    pub fn apply_step(
        &self,
        model: &mut StarModel,
        lambda: f64,
    ) -> Result<f64, StarError> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(StarError::InvalidInput(format!(
                "apply_step requires finite lambda in [0, 1], got {lambda}"
            )));
        }
        let vals = self.interpolate_target_to_model(model)?;
        let nz = model.nz;
        let species = model.species;
        if model.xa.len() != nz * species {
            return Err(StarError::InvalidInput(format!(
                "xa length {} != nz * species ({})",
                model.xa.len(),
                nz * species
            )));
        }

        let one_minus = 1.0 - lambda;
        for (xa_val, &tgt_val) in model.xa.iter_mut().zip(vals.iter()) {
            *xa_val = one_minus * (*xa_val) + lambda * tgt_val;
        }

        // Accumulate cell-major matching Fortran `sum(abs(s% xa(:,1:nz) - vals(:,1:nz)))`.
        let mut abs_sum = 0.0_f64;
        for (xa_val, &tgt_val) in model.xa.iter().zip(vals.iter()) {
            abs_sum += (*xa_val - tgt_val).abs();
        }
        Ok(abs_sum / ((nz * species) as f64))
    }
}
