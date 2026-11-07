// SPDX-License-Identifier: LGPL-3.0-or-later
//! 4D Tabulated Equation of State (`OPAL/SCVH` and `FreeEOS`) evaluation (`tables_4d`).
//!
//! Ports the 2D bicubic $(\log_{10} Q, \log_{10} T)$ spline tables and $(X, Z)$
//! composition interpolation from upstream MESA `eosdt_load_tables.f90` and
//! `eosdt_eval.f90` (`Get1_eosdt_Results`, `Get1_eosdt_for_X`, `Eval_Xtable`,
//! `Do_EoS_Interpolations`, `Get_FreeEOS_alfa`, and `get_opal_scvh_alfa_and_partials`).

pub mod blend_alfa;
pub mod spline;
pub mod table_set;

pub use blend_alfa::{
    get_free_eos_alfa, get_opal_scvh_alfa_and_partials, FreeEosConfig,
    OpalScvhConfig,
};
pub use spline::{
    do_eos_interpolations, eval_xtable, locate_log_q, locate_log_t,
    EosDtTableKind, EosDtVertex, EosDtXzTable, NUM_EOS_TABLE_VALS,
};
pub use table_set::{
    eosdt_relative_data_path, format_xz_str, DtXzInfo, EosDtTableSet,
};

use crate::types::{
    EosError, EosEvalOutput, I_FRAC_FREE_EOS, I_FRAC_OPAL_SCVH, I_LN_E,
    I_LN_PGAS, I_LN_S, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_utils::is_bad;

/// Evaluates `Get1_eosdt_for_X` at fixed metallicity index `iz_1based` and hydrogen fraction `x`.
///
/// Mirrors `Get1_eosdt_for_X` in `eos/private/eosdt_eval.f90` (lines 1751..2020),
/// returning `(EosEvalOutput, d_dX)`.
pub fn get1_eosdt_for_x(
    tables: &EosDtTableSet,
    kind: EosDtTableKind,
    iz_1based: usize,
    x: f64,
    log_rho: f64,
    log_t: f64,
    use_linear_interp_for_x: bool,
    tiny: f64,
) -> Result<(EosEvalOutput, [f64; NUM_EOS_BASIC_RESULTS]), EosError> {
    let xz = match kind {
        EosDtTableKind::OpalScvh => &tables.opal_xz,
        EosDtTableKind::FreeEos => &tables.free_xz,
    };
    let xs = &xz.xs_for_z[iz_1based - 1];
    let num_xs = xs.len();

    if x < tiny || num_xs == 1 {
        let tbl = tables.get_table(kind, 1, iz_1based)?;
        let out = eval_xtable(&tbl, log_rho, log_t)?;
        return Ok((out, [0.0; NUM_EOS_BASIC_RESULTS]));
    }

    if x >= xs[num_xs - 1] {
        let tbl = tables.get_table(kind, num_xs, iz_1based)?;
        let out = eval_xtable(&tbl, log_rho, log_t)?;
        if is_bad(out.res[I_LN_S]) {
            return Err(EosError::InvalidState {
                context: "get1_eosdt_for_x",
                detail: "bad lnS at upper X boundary".to_string(),
            });
        }
        return Ok((out, [0.0; NUM_EOS_BASIC_RESULTS]));
    }

    let do_linear = |tables: &EosDtTableSet| -> Result<
        (EosEvalOutput, [f64; NUM_EOS_BASIC_RESULTS]),
        EosError,
    > {
        let mut ix_1based = 2;
        for candidate in 2..=num_xs {
            if xs[candidate - 1] >= x {
                ix_1based = candidate;
                break;
            }
        }
        let tbl1 = tables.get_table(kind, ix_1based - 1, iz_1based)?;
        let tbl2 = tables.get_table(kind, ix_1based, iz_1based)?;
        let out1 = eval_xtable(&tbl1, log_rho, log_t)?;
        let out2 = eval_xtable(&tbl2, log_rho, log_t)?;

        let x1 = xs[ix_1based - 2];
        let x2 = xs[ix_1based - 1];
        let alfa = (x - x2) / (x1 - x2);
        let beta = 1.0 - alfa;
        let dalfa_dx = 1.0 / (x1 - x2);
        let dbeta_dx = -dalfa_dx;

        let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
        let mut d_dlnd = [0.0_f64; NUM_EOS_BASIC_RESULTS];
        let mut d_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
        let mut d_dx = [0.0_f64; NUM_EOS_BASIC_RESULTS];

        for j in 0..NUM_EOS_TABLE_VALS {
            res[j] = alfa * out1.res[j] + beta * out2.res[j];
            d_dlnd[j] = alfa * out1.d_dlnd[j] + beta * out2.d_dlnd[j];
            d_dln_t[j] = alfa * out1.d_dln_t[j] + beta * out2.d_dln_t[j];
            d_dx[j] = dalfa_dx * out1.res[j] + dbeta_dx * out2.res[j];
        }

        Ok((
            EosEvalOutput {
                res,
                d_dlnd,
                d_dln_t,
                d_dxa: Vec::new(),
            },
            d_dx,
        ))
    };

    if use_linear_interp_for_x || num_xs == 2 {
        return do_linear(tables);
    }

    let (ix_lo, ix_hi) = if x <= xs[1] {
        (1, 3)
    } else if x >= xs[num_xs - 2] {
        (num_xs - 2, num_xs)
    } else {
        let mut found = (1, 3);
        for ix in 3..num_xs {
            if x <= xs[ix - 1] {
                found = (ix - 2, ix + 1);
                break;
            }
        }
        found
    };

    let dx1 = xs[ix_lo] - xs[ix_lo - 1];
    let dx2 = xs[ix_lo + 1] - xs[ix_lo];
    let equal_spaced = if ix_hi - ix_lo == 2 {
        (dx1 - dx2).abs() <= tiny
    } else {
        let dx3 = xs[ix_hi - 1] - xs[ix_lo + 1];
        (dx1 - dx2).abs() <= tiny && (dx2 - dx3).abs() <= tiny
    };

    if !equal_spaced {
        return do_linear(tables);
    }

    let mut outs = Vec::with_capacity(ix_hi - ix_lo + 1);
    for ix in ix_lo..=ix_hi {
        let tbl = tables.get_table(kind, ix, iz_1based)?;
        outs.push(eval_xtable(&tbl, log_rho, log_t)?);
    }

    let del_x = x - xs[ix_lo - 1];
    let dx = dx1;
    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dlnd = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dx = [0.0_f64; NUM_EOS_BASIC_RESULTS];

    if ix_hi - ix_lo == 2 {
        let denom = 2.0 * dx * dx;
        let c = [
            (2.0 * dx * dx - 3.0 * dx * del_x + del_x * del_x) / denom,
            2.0 * (2.0 * dx - del_x) * del_x / denom,
            del_x * (del_x - dx) / denom,
        ];
        let dcdx = [
            (-3.0 * dx + 2.0 * del_x) / denom,
            2.0 * (2.0 * dx - 2.0 * del_x) / denom,
            (2.0 * del_x - dx) / denom,
        ];
        for j in 0..NUM_EOS_TABLE_VALS {
            res[j] = c[0] * outs[0].res[j]
                + c[1] * outs[1].res[j]
                + c[2] * outs[2].res[j];
            d_dlnd[j] = c[0] * outs[0].d_dlnd[j]
                + c[1] * outs[1].d_dlnd[j]
                + c[2] * outs[2].d_dlnd[j];
            d_dln_t[j] = c[0] * outs[0].d_dln_t[j]
                + c[1] * outs[1].d_dln_t[j]
                + c[2] * outs[2].d_dln_t[j];
            d_dx[j] = dcdx[0] * outs[0].res[j]
                + dcdx[1] * outs[1].res[j]
                + dcdx[2] * outs[2].res[j];
        }
    } else {
        let coef = ((x - xs[ix_lo]) / dx).clamp(0.0, 1.0);
        let c = [
            -coef * (coef - 1.0) * (coef - 1.0) / 2.0,
            (2.0 - coef * coef * (5.0 - 3.0 * coef)) / 2.0,
            coef * (1.0 + coef * (4.0 - 3.0 * coef)) / 2.0,
            coef * coef * (coef - 1.0) / 2.0,
        ];
        let dcoef_dx = 1.0 / dx;
        let dcdx = [
            -(3.0 * coef * coef - 4.0 * coef + 1.0) / 2.0 * dcoef_dx,
            (9.0 * coef * coef - 10.0 * coef) / 2.0 * dcoef_dx,
            -(9.0 * coef * coef - 8.0 * coef - 1.0) / 2.0 * dcoef_dx,
            coef * (3.0 * coef - 2.0) / 2.0 * dcoef_dx,
        ];
        for j in 0..NUM_EOS_TABLE_VALS {
            res[j] = c[0] * outs[0].res[j]
                + (c[1] * outs[1].res[j]
                    + (c[2] * outs[2].res[j] + c[3] * outs[3].res[j]));
            d_dlnd[j] = c[0] * outs[0].d_dlnd[j]
                + (c[1] * outs[1].d_dlnd[j]
                    + (c[2] * outs[2].d_dlnd[j] + c[3] * outs[3].d_dlnd[j]));
            d_dln_t[j] = c[0] * outs[0].d_dln_t[j]
                + (c[1] * outs[1].d_dln_t[j]
                    + (c[2] * outs[2].d_dln_t[j] + c[3] * outs[3].d_dln_t[j]));
            d_dx[j] = dcdx[0] * outs[0].res[j]
                + dcdx[1] * outs[1].res[j]
                + dcdx[2] * outs[2].res[j]
                + dcdx[3] * outs[3].res[j];
        }
    }

    Ok((
        EosEvalOutput {
            res,
            d_dlnd,
            d_dln_t,
            d_dxa: Vec::new(),
        },
        d_dx,
    ))
}

/// Evaluates `Get1_eosdt_Results` across $(X, Z, \log_{10} \rho, \log_{10} T)$.
///
/// Mirrors `Get1_eosdt_Results` in `eos/private/eosdt_eval.f90` (lines 1632..1748),
/// returning `(EosEvalOutput, d_dX, d_dZ)`.
pub fn get1_eosdt_results(
    tables: &EosDtTableSet,
    kind: EosDtTableKind,
    z: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
    use_linear_interp_for_x: bool,
    tiny: f64,
) -> Result<
    (
        EosEvalOutput,
        [f64; NUM_EOS_BASIC_RESULTS],
        [f64; NUM_EOS_BASIC_RESULTS],
    ),
    EosError,
> {
    let xz = match kind {
        EosDtTableKind::OpalScvh => &tables.opal_xz,
        EosDtTableKind::FreeEos => &tables.free_xz,
    };
    let n_zs = xz.zs.len();

    if z <= xz.zs[0].max(1.0e-20) {
        let (out, d_dx) = get1_eosdt_for_x(
            tables,
            kind,
            1,
            x,
            log_rho,
            log_t,
            use_linear_interp_for_x,
            tiny,
        )?;
        return Ok((out, d_dx, [0.0; NUM_EOS_BASIC_RESULTS]));
    }

    if z >= xz.zs[n_zs - 1] {
        let (out, d_dx) = get1_eosdt_for_x(
            tables,
            kind,
            n_zs,
            x,
            log_rho,
            log_t,
            use_linear_interp_for_x,
            tiny,
        )?;
        return Ok((out, d_dx, [0.0; NUM_EOS_BASIC_RESULTS]));
    }

    let mut iz_hi = 2;
    for iz in 2..=n_zs {
        if z < xz.zs[iz - 1] {
            iz_hi = iz;
            break;
        }
    }
    let iz1 = iz_hi - 1;
    let iz2 = iz_hi;
    let z1 = xz.zs[iz1 - 1];
    let z2 = xz.zs[iz2 - 1];
    let c2 = (z - z1) / (z2 - z1);
    let c1 = 1.0 - c2;
    let dcdz2 = 1.0 / (z2 - z1);
    let dcdz1 = -dcdz2;

    let (out1, dx1) = get1_eosdt_for_x(
        tables,
        kind,
        iz1,
        x,
        log_rho,
        log_t,
        use_linear_interp_for_x,
        tiny,
    )?;
    let (out2, dx2) = get1_eosdt_for_x(
        tables,
        kind,
        iz2,
        x,
        log_rho,
        log_t,
        use_linear_interp_for_x,
        tiny,
    )?;

    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dlnd = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dx = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dz = [0.0_f64; NUM_EOS_BASIC_RESULTS];

    for j in 0..NUM_EOS_TABLE_VALS {
        res[j] = c1 * out1.res[j] + c2 * out2.res[j];
        d_dlnd[j] = c1 * out1.d_dlnd[j] + c2 * out2.d_dlnd[j];
        d_dln_t[j] = c1 * out1.d_dln_t[j] + c2 * out2.d_dln_t[j];
        d_dx[j] = c1 * dx1[j] + c2 * dx2[j];
        d_dz[j] = dcdz1 * out1.res[j] + dcdz2 * out2.res[j];
    }

    Ok((
        EosEvalOutput {
            res,
            d_dlnd,
            d_dln_t,
            d_dxa: Vec::new(),
        },
        d_dx,
        d_dz,
    ))
}

/// Evaluates `get_opal_scvh_for_eosdt` or `get_FreeEOS_for_eosdt` from precomputed nuclear charges `z_int`.
///
/// Mirrors `get1_for_eosdt` in `eos/private/eosdt_eval.f90` (lines 1584..1629).
#[allow(clippy::too_many_arguments)]
pub fn eval_eosdt_table_from_weights(
    tables: &EosDtTableSet,
    kind: EosDtTableKind,
    z: f64,
    x: f64,
    log_rho: f64,
    log_t: f64,
    z_int: &[usize],
    use_linear_interp_for_x: bool,
    tiny: f64,
) -> Result<EosEvalOutput, EosError> {
    let (mut out, d_dx, d_dz) = get1_eosdt_results(
        tables,
        kind,
        z,
        x,
        log_rho,
        log_t,
        use_linear_interp_for_x,
        tiny,
    )?;

    let mut d_dxa = Vec::with_capacity(z_int.len());
    for &zi in z_int {
        match zi {
            1 => d_dxa.push([d_dx[I_LN_PGAS], d_dx[I_LN_E]]),
            2 => d_dxa.push([0.0, 0.0]),
            _ => d_dxa.push([d_dz[I_LN_PGAS], d_dz[I_LN_E]]),
        }
    }
    out.d_dxa = d_dxa;
    let frac_idx = match kind {
        EosDtTableKind::OpalScvh => I_FRAC_OPAL_SCVH,
        EosDtTableKind::FreeEos => I_FRAC_FREE_EOS,
    };
    out.set_single_component_fraction(frac_idx);
    Ok(out)
}

/// Evaluates the standalone `OPAL/SCVH` tabulated EOS for `(rho, T, chem_id, xa)`.
pub fn eval_opal_scvh_eos(
    tables: &EosDtTableSet,
    rho: f64,
    t: f64,
    chem_id: &[usize],
    xa: &[f64],
    cfg: &OpalScvhConfig,
) -> Result<EosEvalOutput, EosError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|e| {
            EosError::InvalidState {
                context: "eval_opal_scvh_eos",
                detail: e.to_string(),
            }
        })?;
    let mut z_int = Vec::with_capacity(chem_id.len());
    for &cid in chem_id {
        let nuc =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context: "eval_opal_scvh_eos",
                    detail: format!("unknown chem_id {cid}"),
                })?;
        z_int.push(nuc.z as usize);
    }
    eval_eosdt_table_from_weights(
        tables,
        EosDtTableKind::OpalScvh,
        comp.xz,
        comp.xh,
        rho.log10(),
        t.log10(),
        &z_int,
        cfg.use_linear_interp_for_x,
        cfg.tiny_fuzz,
    )
}

/// Evaluates the standalone `FreeEOS` tabulated EOS for `(rho, T, chem_id, xa)`.
pub fn eval_free_eos(
    tables: &EosDtTableSet,
    rho: f64,
    t: f64,
    chem_id: &[usize],
    xa: &[f64],
    cfg: &FreeEosConfig,
) -> Result<EosEvalOutput, EosError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|e| {
            EosError::InvalidState {
                context: "eval_free_eos",
                detail: e.to_string(),
            }
        })?;
    let mut z_int = Vec::with_capacity(chem_id.len());
    for &cid in chem_id {
        let nuc =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context: "eval_free_eos",
                    detail: format!("unknown chem_id {cid}"),
                })?;
        z_int.push(nuc.z as usize);
    }
    eval_eosdt_table_from_weights(
        tables,
        EosDtTableKind::FreeEos,
        comp.xz,
        comp.xh,
        rho.log10(),
        t.log10(),
        &z_int,
        cfg.use_linear_interp_for_x,
        cfg.tiny_fuzz,
    )
}
