// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream MESA `26.4.1` mesh spacing function evaluation
//! (`star/private/mesh_functions.f90:1-327`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, LN10, M_SUN, R_SUN, SECYER};

use crate::controls::StarError;
use crate::controls_mix_wind::{MixWindControls, NUM_XA_FUNCTION};
use crate::model::StarModel;

/// Maximum number of simultaneous mesh spacing functions (`max_allowed_gvals`
/// in `star/private/mesh_functions.f90:33`).
pub const MAX_ALLOWED_GVALS: usize = 50;

/// One caller-supplied mesh spacing column returned by [`OtherMeshFunctionsHook`]
/// (`use_other_mesh_functions` in `star/other/other_mesh_functions.f90`).
#[derive(Debug, Clone)]
pub struct OtherMeshFunctionColumn {
    /// Mesh function name (`names(i)`, up to 32 characters in upstream MESA).
    pub name: String,
    /// Whether this column represents an abundance profile (`gval_is_xa_function(i)`).
    pub is_xa_function: bool,
    /// Zone profile values of length `nz` (`vals(1:nz, i)`).
    pub vals: Vec<f64>,
}

/// Callback trait for `use_other_mesh_functions` (`how_many_other_mesh_fcns` and
/// `other_mesh_fcn_data` in `star/private/mesh_functions.f90:69-70, 116-127`).
pub trait OtherMeshFunctionsHook: Send + Sync {
    /// Returns custom mesh spacing function columns for `model`.
    fn eval_other_mesh_functions(
        &self,
        model: &StarModel,
    ) -> Result<Vec<OtherMeshFunctionColumn>, StarError>;
}

/// Cloneable reference-counted handle wrapping an [`OtherMeshFunctionsHook`].
#[derive(Clone)]
pub struct OtherMeshFunctionsHandle(
    pub std::sync::Arc<dyn OtherMeshFunctionsHook>,
);

impl std::fmt::Debug for OtherMeshFunctionsHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OtherMeshFunctionsHandle(..)")
    }
}

/// Evaluated mesh function table produced by [`set_mesh_function_data`].
#[derive(Debug, Clone)]
pub struct MeshFunctionData {
    /// Function names (`names(1..=nfcns)`).
    pub names: Vec<String>,
    /// Abundance-function flags (`gval_is_xa_function(1..=nfcns)`).
    pub is_xa_function: Vec<bool>,
    /// Temperature-function flags (`gval_is_logT_function(1..=nfcns)`).
    pub is_logt_function: Vec<bool>,
    /// Column-major mesh function profiles `vals[j * nz + k]` (`vals(1:nz, 1:nfcns)`).
    pub vals: Vec<f64>,
}

impl MeshFunctionData {
    /// Returns the `nz`-element slice for 0-based function index `j`.
    #[must_use]
    #[inline]
    pub fn column(&self, j: usize, nz: usize) -> &[f64] {
        &self.vals[j * nz..(j + 1) * nz]
    }

    /// Returns the mutable `nz`-element slice for 0-based function index `j`.
    #[inline]
    pub fn column_mut(&mut self, j: usize, nz: usize) -> &mut [f64] {
        &mut self.vals[j * nz..(j + 1) * nz]
    }
}

/// Resolves `species` to its 0-based net species index in `model` (`get_net_iso`
/// in `star/private/mesh_functions.f90:37-51`).
#[must_use]
pub fn get_net_iso_0based(model: &StarModel, species: &str) -> Option<usize> {
    let trimmed = species.trim();
    if trimmed.is_empty() {
        return None;
    }
    let cid = NuclideCatalog::builtin_ref()
        .get_by_name(trimmed)
        .map(|iso| iso.chem_id)?;
    if let Some(Some(j0)) = model.net_iso.get(cid) {
        if *j0 < model.species {
            return Some(*j0);
        }
    }
    model.chem_id.iter().position(|&c| c == cid)
}

#[inline]
fn do_mass_function(model: &StarModel, species: &str, weight: f64) -> bool {
    weight > 0.0 && get_net_iso_0based(model, species).is_some()
}

/// Counts active mesh spacing functions (`num_mesh_functions` in
/// `star/private/mesh_functions.f90:65-91`) using `mw` mesh weights.
///
/// # Errors
/// Returns [`StarError`] if `use_other_mesh_functions` is enabled without a
/// registered [`OtherMeshFunctionsHook`] or if the hook returns an error.
pub fn num_mesh_functions_with_controls(
    model: &StarModel,
    mw: &MixWindControls,
) -> Result<usize, StarError> {
    let mut count = 0_usize;
    if mw.use_other_mesh_functions {
        let Some(ref hook) = model.aux.other_mesh_functions else {
            return Err(StarError::InvalidInput(String::from(
                "use_other_mesh_functions = .true. requires an other_mesh_functions hook",
            )));
        };
        count += hook.0.eval_other_mesh_functions(model)?.len();
    }
    if mw.convective_bdy_weight > 0.0 {
        count += 1;
    }
    if mw.e_function_weight > 0.0 {
        count += 1;
    }
    if mw.p_function_weight > 0.0 {
        count += 1;
    }
    if mw.t_function1_weight > 0.0 {
        count += 1;
    }
    if mw.t_function2_weight > 0.0 {
        count += 1;
    }
    if mw.r_function_weight > 0.0 {
        count += 1;
    }
    if mw.r_function2_weight > 0.0 {
        count += 1;
    }
    if mw.r_function3_weight > 0.0 {
        count += 1;
    }
    if mw.m_function_weight > 0.0 {
        count += 1;
    }
    if mw.gradt_function_weight > 0.0 {
        count += 1;
    }
    if mw.log_tau_function_weight > 0.0 {
        count += 1;
    }
    if mw.log_kap_function_weight > 0.0 {
        count += 1;
    }
    if mw.gam_function_weight > 0.0 {
        count += 1;
    }
    if mw.omega_function_weight > 0.0 && model.controls.rotation_flag {
        count += 1;
    }
    for k in 0..NUM_XA_FUNCTION {
        let sp = mw.xa_function_species.get(k).map_or("", String::as_str);
        let wt = mw.xa_function_weight.get(k).copied().unwrap_or(0.0);
        if do_mass_function(model, sp, wt) {
            count += 1;
        }
    }
    Ok(count)
}

/// Counts active mesh spacing functions on `model` (`num_mesh_functions` in
/// `star/private/mesh_functions.f90:65-91`).
///
/// # Errors
/// Returns [`StarError`] if `use_other_mesh_functions` fails.
pub fn num_mesh_functions(model: &StarModel) -> Result<usize, StarError> {
    num_mesh_functions_with_controls(model, &model.controls.mix_wind)
}

/// Locates convective boundaries in exact upstream `locate_convection_boundaries`
/// order (`star/private/mix_info.f90:580-623, 738-746`), returning 1-based
/// `(conv_bdy_loc, top_conv_bdy)` pairs on the radiative side of each interface.
fn locate_fortran_conv_boundaries(model: &StarModel) -> Vec<(usize, bool)> {
    let nz = model.nz;
    if nz == 0 {
        return Vec::new();
    }
    let mut bdys = Vec::new();
    let mut in_conv = model.mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    if nz >= 3 {
        for k1 in (2..nz).rev() {
            let k0 = k1 - 1;
            let is_conv = model.mixing_type[k0] == MixingType::ConvectiveMixing;
            if in_conv {
                if !is_conv {
                    in_conv = false;
                    bdys.push((k1, true));
                }
            } else if is_conv {
                in_conv = true;
                bdys.push((k1 + 1, false));
            }
        }
    }
    if in_conv {
        bdys.push((1, true));
    }
    bdys
}

fn eval_conv_bdy_column(
    model: &StarModel,
    mw: &MixWindControls,
    i_1based: usize,
    col: &mut [f64],
) -> Result<(), StarError> {
    let nz = model.nz;
    col.fill(0.0);
    if nz == 0 || model.dt < mw.convective_bdy_min_dt_yrs * SECYER {
        return Ok(());
    }
    let bdys = locate_fortran_conv_boundaries(model);
    let num_conv = bdys.len();
    let init_len = num_conv.max(10);

    for &(k1, _) in &bdys {
        let k0 = k1 - 1;
        if model.dq[k0] < mw.convective_bdy_dq_limit {
            continue;
        }
        col[k0] = mw.convective_bdy_weight;
        // Upstream `star/private/mesh_functions.f90:294, 296` subscripts
        // `s% top_conv_bdy(i)` (1-based mesh-function column `i`) instead of
        // `s% top_conv_bdy(j)` (deferred bug #241). `mix_info.f90:89, 95`
        // initializes `top_conv_bdy(1..=max(10, num_conv_boundaries))` with
        // unset trailing slots `.false.`; `i > max(10, num_conv_boundaries)`
        // would read out-of-bounds heap memory and is rejected explicitly.
        if i_1based > init_len {
            return Err(StarError::InvalidInput(format!(
                "do_conv_bdy mesh-function column i={i_1based} exceeds initialized top_conv_bdy capacity {init_len} (upstream bug #241)"
            )));
        }
        let top_i = bdys.get(i_1based - 1).is_some_and(|&(_, is_top)| is_top);
        if k1 < nz && top_i {
            col[k0 + 1] = mw.convective_bdy_weight;
        } else if k1 > 1 && !top_i {
            col[k0 - 1] = mw.convective_bdy_weight;
        }
    }
    for k0 in 1..nz {
        col[k0] += col[k0 - 1];
    }
    Ok(())
}

/// Evaluates all active mesh spacing functions (`set_mesh_function_data` in
/// `star/private/mesh_functions.f90:94-325`).
///
/// # Errors
/// Returns [`StarError`] if `nfcns` mismatches active function count, exceeds
/// [`MAX_ALLOWED_GVALS`], or encounters an invalid custom hook or `#241` OOB read.
pub fn set_mesh_function_data_with_controls(
    model: &StarModel,
    mw: &MixWindControls,
    nfcns: usize,
) -> Result<MeshFunctionData, StarError> {
    set_mesh_function_data_with_controls_and_xa(model, mw, nfcns, &model.xa)
}

pub(crate) fn set_mesh_function_data_with_controls_and_xa(
    model: &StarModel,
    mw: &MixWindControls,
    nfcns: usize,
    xa_slice: &[f64],
) -> Result<MeshFunctionData, StarError> {
    if nfcns > MAX_ALLOWED_GVALS {
        return Err(StarError::InvalidInput(format!(
            "num_mesh_functions={nfcns} exceeds max_allowed_gvals={MAX_ALLOWED_GVALS}"
        )));
    }
    let nz = model.nz;
    let mut names = Vec::with_capacity(nfcns);
    let mut is_xa = vec![false; nfcns];
    let mut is_logt = vec![false; nfcns];
    let mut vals = vec![0.0_f64; nz * nfcns];

    let mut i_other = 0_usize;
    if mw.use_other_mesh_functions {
        let Some(ref hook) = model.aux.other_mesh_functions else {
            return Err(StarError::InvalidInput(String::from(
                "use_other_mesh_functions = .true. requires an other_mesh_functions hook",
            )));
        };
        let other_cols = hook.0.eval_other_mesh_functions(model)?;
        i_other = other_cols.len();
        if i_other > nfcns {
            return Err(StarError::InvalidInput(format!(
                "other_mesh_functions returned {i_other} columns, exceeding nfcns={nfcns}"
            )));
        }
        for (idx, col) in other_cols.into_iter().enumerate() {
            if col.vals.len() != nz {
                return Err(StarError::InvalidInput(format!(
                    "other_mesh_functions column '{}' has len {}, expected nz={nz}",
                    col.name,
                    col.vals.len()
                )));
            }
            names.push(col.name);
            is_xa[idx] = col.is_xa_function;
            vals[idx * nz..(idx + 1) * nz].copy_from_slice(&col.vals);
        }
    }

    let mut push_builtin = |name: &str, xa_flag: bool, logt_flag: bool| {
        let idx = names.len();
        names.push(String::from(name));
        if let Some(slot) = is_xa.get_mut(idx) {
            *slot = xa_flag;
        }
        if let Some(slot) = is_logt.get_mut(idx) {
            *slot = logt_flag;
        }
    };

    if mw.e_function_weight > 0.0 {
        push_builtin("E_function", false, false);
    }
    if mw.p_function_weight > 0.0 {
        push_builtin("P_function", false, false);
    }
    if mw.t_function1_weight > 0.0 {
        push_builtin("T_function1", false, true);
    }
    if mw.t_function2_weight > 0.0 {
        push_builtin("T_function2", false, true);
    }
    if mw.r_function_weight > 0.0 {
        push_builtin("R_function", false, false);
    }
    if mw.r_function2_weight > 0.0 {
        push_builtin("R_function2", false, false);
    }
    if mw.r_function3_weight > 0.0 {
        push_builtin("R_function3", false, false);
    }
    if mw.m_function_weight > 0.0 {
        push_builtin("M_function", false, false);
    }
    if mw.gradt_function_weight > 0.0 {
        push_builtin("gradT_function", false, false);
    }
    if mw.log_tau_function_weight > 0.0 {
        push_builtin("log_tau_function", false, false);
    }
    if mw.log_kap_function_weight > 0.0 {
        push_builtin("log_kap_function", false, false);
    }
    if mw.gam_function_weight > 0.0 {
        push_builtin("gam_function", false, false);
    }
    if mw.omega_function_weight > 0.0 && model.controls.rotation_flag {
        push_builtin("omega_function", false, false);
    }
    if mw.convective_bdy_weight > 0.0 {
        push_builtin("convective_bdy", false, false);
    }
    for k in 0..NUM_XA_FUNCTION {
        let sp = mw.xa_function_species.get(k).map_or("", |s| s.trim());
        let wt = mw.xa_function_weight.get(k).copied().unwrap_or(0.0);
        if do_mass_function(model, sp, wt) {
            push_builtin(sp, true, false);
        }
    }

    if names.len() != nfcns {
        return Err(StarError::InvalidInput(format!(
            "set_mesh_function_data: constructed {} names, expected nfcns={nfcns}",
            names.len()
        )));
    }

    let r_surf = model.r.first().copied().unwrap_or(1.0);
    for idx in i_other..nfcns {
        let i_1based = idx + 1;
        let col = &mut vals[idx * nz..(idx + 1) * nz];
        match names[idx].as_str() {
            "E_function" => {
                for k in 0..nz {
                    let lne = if model.ln_e.is_empty() {
                        flexmesa_math::log(model.energy[k])
                    } else {
                        model.ln_e[k]
                    };
                    col[k] = mw.e_function_weight
                        * mw.e_function_param.max(lne / LN10);
                }
            }
            "P_function" => {
                for k in 0..nz {
                    col[k] = (mw.p_function_weight * model.ln_peos[k]) / LN10;
                }
            }
            "T_function1" => {
                for k in 0..nz {
                    col[k] = (mw.t_function1_weight * model.ln_t[k]) / LN10;
                }
            }
            "T_function2" => {
                for k in 0..nz {
                    let tk = model.t[k];
                    col[k] = mw.t_function2_weight
                        * flexmesa_math::log10(
                            tk / (tk + mw.t_function2_param),
                        );
                }
            }
            "R_function" => {
                for k in 0..nz {
                    col[k] = mw.r_function_weight
                        * flexmesa_math::log10(
                            1.0 + (model.r[k] / R_SUN) / mw.r_function_param,
                        );
                }
            }
            "R_function2" => {
                for k in 0..nz {
                    col[k] = mw.r_function2_weight
                        * mw.r_function2_param1.min(
                            mw.r_function2_param2.max(model.r[k] / r_surf),
                        );
                }
            }
            "R_function3" => {
                for k in 0..nz {
                    col[k] = (mw.r_function3_weight * model.r[k]) / r_surf;
                }
            }
            "M_function" => {
                for k in 0..nz {
                    col[k] = mw.m_function_weight
                        * flexmesa_math::log10(
                            1.0 + ((model.xmstar * model.q[k]) / M_SUN)
                                / mw.m_function_param,
                        );
                }
            }
            "gradT_function" => {
                for k in 0..nz {
                    col[k] = mw.gradt_function_weight * model.gradt[k];
                }
            }
            "log_tau_function" => {
                for k in 0..nz {
                    col[k] = mw.log_tau_function_weight
                        * flexmesa_math::log10(model.tau[k]);
                }
            }
            "log_kap_function" => {
                for k in 0..nz {
                    col[k] = mw.log_kap_function_weight
                        * flexmesa_math::log10(model.opacity[k]);
                }
            }
            "gam_function" => {
                for k in 0..nz {
                    col[k] = mw.gam_function_weight
                        * flexmesa_math::tanh(
                            (model.gam[k] - mw.gam_function_param1)
                                / mw.gam_function_param2,
                        );
                }
            }
            "omega_function" => {
                for k in 0..nz {
                    col[k] = mw.omega_function_weight
                        * flexmesa_math::log10(
                            (1.0e-99_f64).max(model.omega[k].abs()),
                        );
                }
            }
            "convective_bdy" => {
                eval_conv_bdy_column(model, mw, i_1based, col)?;
            }
            other_name => {
                for slot in 0..NUM_XA_FUNCTION {
                    let sp = mw
                        .xa_function_species
                        .get(slot)
                        .map_or("", |s| s.trim());
                    if sp.is_empty() || sp != other_name {
                        continue;
                    }
                    let Some(j0) = get_net_iso_0based(model, sp) else {
                        return Err(StarError::InvalidInput(format!(
                            "unknown or inactive xa_function_species '{sp}'"
                        )));
                    };
                    let weight =
                        mw.xa_function_weight.get(slot).copied().unwrap_or(0.0);
                    let param =
                        mw.xa_function_param.get(slot).copied().unwrap_or(0.0);
                    for m in 0..nz {
                        let x = xa_slice[m * model.species + j0];
                        col[m] = weight * flexmesa_math::log10(x + param);
                    }
                }
            }
        }
    }

    Ok(MeshFunctionData {
        names,
        is_xa_function: is_xa,
        is_logt_function: is_logt,
        vals,
    })
}

/// Evaluates all active mesh spacing functions on `model` (`set_mesh_function_data`
/// in `star/private/mesh_functions.f90:94-325`).
///
/// # Errors
/// Returns [`StarError`] if mesh function evaluation fails.
pub fn set_mesh_function_data(
    model: &StarModel,
    nfcns: usize,
) -> Result<MeshFunctionData, StarError> {
    set_mesh_function_data_with_controls(model, &model.controls.mix_wind, nfcns)
}
