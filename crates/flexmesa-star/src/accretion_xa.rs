// SPDX-License-Identifier: LGPL-3.0-or-later
//! Accreted material composition construction (`get_xa_for_accretion`,
//! `get_xa_for_standard_metals`, `get_xa`, and `adjust_z_fractions`) matching
//! `star/private/adjust_xyz.f90:910-1220`.

use flexmesa_chem::{
    lodders03_element_atom_percent, solar_element_zfrac, NuclideCatalog,
    SolarScale, NUM_CHEM_ELEMENTS,
};

use crate::controls::StarError;
use crate::controls_mix_wind::MixWindControls;
use crate::model::StarModel;

/// Builds the accreted mass-fraction vector (`len = model.species`) matching
/// `get_xa_for_accretion` in `star/private/adjust_xyz.f90:910-947` and
/// `adjust_mass.f90:382-395`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `accrete_given_mass_fractions` sums
/// outside `1 +/- 1e-2`, if `accretion_zfracs` is invalid for nonzero metals,
/// if custom `z_fraction_*` entries are negative or do not sum to `1`, or if
/// `h1` is absent from the network when using standard-metal accretion.
pub fn get_xa_for_accretion(model: &StarModel) -> Result<Vec<f64>, StarError> {
    let species = model.species;
    if species == 0 {
        return Ok(Vec::new());
    }
    if model.controls.accrete_same_as_surface {
        return Ok(model.cell_xa(0).to_vec());
    }

    if model.controls.accrete_given_mass_fractions {
        let mut xa = vec![0.0_f64; species];
        let n_spec = model
            .controls
            .mix_wind
            .num_accretion_species
            .min(model.controls.accretion_species_id.len())
            .min(model.controls.accretion_species_xa.len());
        for j in 0..n_spec {
            let iso = model.controls.accretion_species_id[j].trim();
            if iso.is_empty() {
                continue;
            }
            if let Some(i) = model
                .iso_names
                .iter()
                .position(|n| n.eq_ignore_ascii_case(iso))
            {
                xa[i] = model.controls.accretion_species_xa[j];
            }
        }
        let sum_xa: f64 = xa.iter().sum();
        if !sum_xa.is_finite() || (1.0 - sum_xa).abs() > 1.0e-2 {
            return Err(StarError::InvalidInput(format!(
                "get_xa_for_accretion: accretion species mass fractions do not add to 1 (sum={sum_xa})"
            )));
        }
        for x in &mut xa {
            *x /= sum_xa;
        }
        return Ok(xa);
    }

    get_xa_for_standard_metals(model)
}

/// Legacy infallible wrapper around [`get_xa_for_accretion`], falling back to
/// surface composition if `model.controls` is incomplete.
///
/// Prefer [`get_xa_for_accretion`] in evolution code so invalid accretion
/// specifications propagate an error matching `adjust_xyz.f90:929-938`.
#[must_use]
pub fn build_accreted_xa(model: &StarModel) -> Vec<f64> {
    if model.species == 0 {
        return Vec::new();
    }
    get_xa_for_accretion(model).unwrap_or_else(|_| model.cell_xa(0).to_vec())
}

fn custom_zfrac_array(
    mw: &MixWindControls,
) -> Result<[f64; NUM_CHEM_ELEMENTS], StarError> {
    let mut zfrac = [0.0_f64; NUM_CHEM_ELEMENTS];
    // 0-based index = atomic number Z - 1 (`e_li = 3` -> 2, ..., `e_zn = 30` -> 29).
    let custom: [(usize, f64); 28] = [
        (2, mw.z_fraction_li),
        (3, mw.z_fraction_be),
        (4, mw.z_fraction_b),
        (5, mw.z_fraction_c),
        (6, mw.z_fraction_n),
        (7, mw.z_fraction_o),
        (8, mw.z_fraction_f),
        (9, mw.z_fraction_ne),
        (10, mw.z_fraction_na),
        (11, mw.z_fraction_mg),
        (12, mw.z_fraction_al),
        (13, mw.z_fraction_si),
        (14, mw.z_fraction_p),
        (15, mw.z_fraction_s),
        (16, mw.z_fraction_cl),
        (17, mw.z_fraction_ar),
        (18, mw.z_fraction_k),
        (19, mw.z_fraction_ca),
        (20, mw.z_fraction_sc),
        (21, mw.z_fraction_ti),
        (22, mw.z_fraction_v),
        (23, mw.z_fraction_cr),
        (24, mw.z_fraction_mn),
        (25, mw.z_fraction_fe),
        (26, mw.z_fraction_co),
        (27, mw.z_fraction_ni),
        (28, mw.z_fraction_cu),
        (29, mw.z_fraction_zn),
    ];
    for (idx, val) in custom {
        if val < 0.0 || !val.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "get_xa_for_standard_metals: invalid negative/non-finite z_fraction at Z={}: {val}",
                idx + 1
            )));
        }
        zfrac[idx] = val;
    }
    let sum_z: f64 = zfrac.iter().sum();
    if (sum_z - 1.0).abs() > 1.0e-6 {
        return Err(StarError::InvalidInput(format!(
            "get_xa_for_standard_metals: bad sum(zfrac) for specified z fractions: {sum_z}"
        )));
    }
    for v in &mut zfrac {
        *v /= sum_z;
    }
    Ok(zfrac)
}

fn get_xa_for_standard_metals(model: &StarModel) -> Result<Vec<f64>, StarError> {
    let mw = &model.controls.mix_wind;
    standard_metals_xa(
        &model.chem_id,
        mw,
        model.controls.accretion_h1,
        mw.accretion_h2,
        mw.accretion_he3,
        model.controls.accretion_he4,
        mw.accretion_zfracs,
        mw.accretion_dump_missing_metals_into_heaviest,
    )
}

/// Builds the homogeneous mass fractions `xa` (`len = chem_id.len()`) for
/// given `h1`, `h2`, `he3`, `he4` and the solar metal mixture `which_zfracs`
/// (`get_xa_for_standard_metals` in `star/private/adjust_xyz.f90:991-1093`).
///
/// `which_zfracs = 0` (`Custom_zfracs`) takes the `z_fraction_*` controls in
/// `mw`; `1..=10` select the tabulated solar scales; any other value is
/// accepted only for metal-free mixtures (`adjust_xyz.f90:1065-1075`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for an invalid `which_zfracs` with
/// non-zero metals, invalid custom `z_fraction_*` values, `h1` missing from
/// the network, or a negative or non-finite result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn standard_metals_xa(
    chem_id: &[usize],
    mw: &MixWindControls,
    h1_in: f64,
    h2_in: f64,
    he3_in: f64,
    he4_in: f64,
    which_zfracs: i32,
    dump_missing_into_heaviest: bool,
) -> Result<Vec<f64>, StarError> {
    let catalog = NuclideCatalog::builtin_ref();
    let mut h1 = h1_in;
    let h2 = h2_in;
    let he3 = he3_in;
    let mut he4 = he4_in;

    let zfrac = match which_zfracs {
        1..=10 => {
            let scale = SolarScale::from_id(which_zfracs as usize)
                .map_err(|e| StarError::InvalidInput(e.to_string()))?;
            solar_element_zfrac(scale, catalog)
        }
        0 => custom_zfrac_array(mw)?,
        other => {
            if (1.0 - (h1 + h2 + he3 + he4)).abs() < 1.0e-8 {
                if h1 > he4 {
                    h1 = (1.0 - (h2 + he3 + he4)).clamp(0.0, 1.0);
                } else {
                    he4 = (1.0 - (h1 + h2 + he3)).clamp(0.0, 1.0);
                }
                [0.0_f64; NUM_CHEM_ELEMENTS]
            } else {
                return Err(StarError::InvalidInput(format!(
                    "get_xa_for_standard_metals: invalid zfracs={other} with non-zero metals (1-(h1+h2+he3+he4)={})",
                    1.0 - (h1 + h2 + he3 + he4)
                )));
            }
        }
    };

    let z_init = (1.0 - (h1 + h2 + he3 + he4)).clamp(0.0, 1.0);
    let xa = get_xa(
        chem_id,
        catalog,
        h1,
        h2,
        he3,
        he4,
        &zfrac,
        z_init,
        dump_missing_into_heaviest,
    )?;
    for (i, &val) in xa.iter().enumerate() {
        if val < 0.0 || !val.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "get_xa_for_standard_metals: bad xa[{i}]={val}"
            )));
        }
    }
    Ok(xa)
}

#[allow(clippy::too_many_arguments)]
fn get_xa(
    chem_id: &[usize],
    catalog: &NuclideCatalog,
    h1: f64,
    h2: f64,
    he3: f64,
    he4: f64,
    zfrac: &[f64; NUM_CHEM_ELEMENTS],
    z_init: f64,
    dump_missing_into_heaviest: bool,
) -> Result<Vec<f64>, StarError> {
    const TINY: f64 = 1.0e-15;
    if (h1 < 0.0 && h1.abs() > TINY)
        || (h2 < 0.0 && h2.abs() > TINY)
        || (he3 < 0.0 && he3.abs() > TINY)
        || (he4 < 0.0 && he4.abs() > TINY)
        || (z_init < 0.0 && z_init.abs() > TINY)
    {
        return Err(StarError::InvalidInput(format!(
            "get_xa: negative composition component (h1={h1}, h2={h2}, he3={he3}, he4={he4}, Z={z_init})"
        )));
    }
    if (1.0 - (h1 + h2 + he3 + he4 + z_init)).abs() > TINY {
        return Err(StarError::InvalidInput(format!(
            "get_xa: sum of (H1+H2+He3+He4+Z) does not sum to 1 (1-sum={})",
            1.0 - (h1 + h2 + he3 + he4 + z_init)
        )));
    }

    let species = chem_id.len();
    let mut xa = vec![0.0_f64; species];
    // `net_iso(ih1)`, `net_iso(ih2)`, ... (`adjust_xyz.f90:1129-1141`).
    let idx_of = |name: &str| {
        chem_id.iter().position(|&cid| {
            catalog.get_by_id(cid).is_some_and(|iso| iso.name == name)
        })
    };
    let idx_h1 = idx_of("h1");
    let idx_h2 = idx_of("h2");
    let idx_he3 = idx_of("he3");
    let idx_he4 = idx_of("he4");

    if let Some(i2) = idx_h2 {
        xa[i2] = h2;
        if let Some(i1) = idx_h1 {
            xa[i1] = h1;
        }
    } else if let Some(i1) = idx_h1 {
        xa[i1] = h1 + h2;
    } else {
        return Err(StarError::InvalidInput(String::from(
            "get_xa: require h1 to be in net",
        )));
    }
    if let Some(i3) = idx_he3 {
        xa[i3] = he3;
    }
    if let Some(i4) = idx_he4 {
        xa[i4] = he4;
    }

    let ztotal = 1.0 - (h1 + h2 + he3 + he4);
    adjust_z_fractions(
        chem_id,
        catalog,
        ztotal,
        zfrac,
        dump_missing_into_heaviest,
        &mut xa,
    )?;

    let sum_xa: f64 = xa.iter().sum();
    if sum_xa <= 0.0 || !sum_xa.is_finite() {
        return Err(StarError::InvalidInput(format!(
            "get_xa: invalid total abundance sum {sum_xa}"
        )));
    }
    for x in &mut xa {
        *x /= sum_xa;
    }
    Ok(xa)
}

fn adjust_z_fractions(
    chem_id: &[usize],
    catalog: &NuclideCatalog,
    ztotal: f64,
    zfrac: &[f64; NUM_CHEM_ELEMENTS],
    dump_missing_into_heaviest: bool,
    xa: &mut [f64],
) -> Result<(), StarError> {
    let species = chem_id.len();
    let mut z_num = vec![0_usize; species];
    let mut weights = vec![0.0_f64; species];
    let mut names: Vec<&str> = Vec::with_capacity(species);

    for (j, &cid) in chem_id.iter().enumerate() {
        let iso = catalog.get_by_id(cid).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "adjust_z_fractions: unknown chem_id {cid}"
            ))
        })?;
        z_num[j] = iso.z;
        weights[j] = iso.w;
        names.push(&iso.name);
    }

    let jskip = if dump_missing_into_heaviest {
        let mut best_j = 0_usize;
        let mut best_w = f64::NEG_INFINITY;
        for (j, &w) in weights.iter().enumerate() {
            if w > best_w {
                best_w = w;
                best_j = j;
            }
        }
        Some(best_j)
    } else {
        None
    };

    let mut new_xa = vec![0.0_f64; species];
    for j in 0..species {
        if jskip == Some(j) || z_num[j] <= 2 {
            continue;
        }
        let frac = 1.0e-2 * lodders03_element_atom_percent(names[j]) * weights[j];
        if frac == 0.0 {
            continue;
        }
        let mut frac_sum = 0.0_f64;
        for i in 0..species {
            if z_num[i] != z_num[j] {
                continue;
            }
            frac_sum +=
                1.0e-2 * lodders03_element_atom_percent(names[i]) * weights[i];
        }
        if frac_sum > 0.0 && z_num[j] <= NUM_CHEM_ELEMENTS {
            let iso_frac = frac / frac_sum;
            let elem_zfrac = zfrac[z_num[j] - 1];
            new_xa[j] = ztotal * elem_zfrac * iso_frac;
        }
    }

    if let Some(js) = jskip {
        let sum_new: f64 = new_xa.iter().sum();
        new_xa[js] = (ztotal - sum_new).max(0.0);
        for j in 0..species {
            if j == js || z_num[j] <= 2 {
                continue;
            }
            xa[j] = new_xa[j];
        }
        xa[js] = new_xa[js];
    } else {
        let zsum: f64 = new_xa.iter().sum();
        if zsum > 0.0 {
            for j in 0..species {
                if z_num[j] <= 2 {
                    continue;
                }
                xa[j] = new_xa[j] * ztotal / zsum;
            }
        }
    }
    Ok(())
}
