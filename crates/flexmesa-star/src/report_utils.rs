// SPDX-License-Identifier: LGPL-3.0-or-later
//! Exact upstream MESA `26.4.1` report scalar evaluators, sub-cell boundary
//! and burning-zone diagnostics, and fixed-width ASCII log formatters.
//!
//! Corresponds to:
//! - `math/public/math_lib_crmath.f90:146-161` (`safe_log10`, via `flexmesa_math::safe_log10`)
//! - `star/private/star_utils.f90:959-1229, 1328-1485, 1771-1796, 2063-2137, 2752-2774, 2890-2897, 3611-3655, 3700-3751`
//! - `star/private/mix_info.f90:428-529` (`set_cz_boundary_info`, `set_cz_bdy_mass`)
//! - `star/private/report.f90:170-375, 456-501, 573-851, 1389-1417`
//! - `star/private/hydro_rotation.f90:588-739, 789-795, 882-888`
//! - `star/private/gravity_darkening.f90:43-149`
//!   (`data/star_data/gravity_darkening_coefficients.data`, SHA-256
//!   `6c051e3f1eec1c871680a9df53b16ab7b5c40f1158432ae26b89bada255b08d5`)
//! - `star/private/history.f90:453-466, 614-905, 1155-1204, 3209-3217`

use std::fmt::Write as _;
use std::sync::OnceLock;

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{
    MixingType, AMU, BOLTZ_SIGMA, CLIGHT, CRAD, FOUR_PI, KERG, LN10, L_SUN,
    M_SUN, PI, R_SUN, SECYER,
};
use flexmesa_interp2d::{
    BicubicBc, interp_evbicub_prevalidated_db, interp_mkbicub_db,
};
pub use flexmesa_math::safe_log10;

use crate::core_boundaries::{
    compute_core_masses, compute_power_categories, CoreMasses, PowerCategories,
};
use crate::evolve_support::eval_total_energy_integrals;
use crate::model::StarModel;
use crate::rotation_mix::eval_d_mix_non_rotation_vec;

/// Default column width for `star` `history.data` and `profileN.data`
/// (`(1pes40.16e3, 1x)`, `(i40, 1x)`, `(a40, 1x)` in `star/defaults/controls.defaults:275-295, 378-398`).
pub const STAR_LOG_COL_WIDTH: usize = 40;

/// Upstream MESA `26.4.1` version string written in `history.data`, `profileN.data`,
/// and `binary_history.data` headers (`data/version_number`).
pub const MESA_REF_VERSION_NUMBER: &str = "r26.4.1";

/// Appends an IEEE-754 `f64` formatted with Fortran `(1pes<width>.16e3, 1x)`
/// scientific notation (`controls.defaults:275, 378`; non-finite values map to
/// `-1e99` per `star/private/history.f90:1158` and
/// `binary/private/binary_history.f90:434`) directly into `out`.
pub fn write_mesa_dbl_width(out: &mut String, x: f64, width: usize) {
    let v = if !x.is_finite() {
        -1.0e99
    } else if x == 0.0 {
        0.0
    } else {
        x
    };
    let mut raw = String::with_capacity(28);
    let _ = write!(raw, "{v:.16E}");
    if let Some((mant, exp_s)) = raw.split_once('E') {
        let exp: i32 = exp_s.parse().unwrap_or(0);
        let sign = if exp < 0 { '-' } else { '+' };
        // `mant` + `E` + sign + 3-digit exponent => exact token length `mant.len() + 5`.
        let tok_len = mant.len() + 5;
        for _ in 0..width.saturating_sub(tok_len) {
            out.push(' ');
        }
        out.push_str(mant);
        out.push('E');
        out.push(sign);
        let _ = write!(out, "{:03} ", exp.unsigned_abs());
    } else {
        let _ = write!(out, "{raw:>width$} ");
    }
}

/// Formats an IEEE-754 `f64` using Fortran `(1pes<width>.16e3, 1x)` scientific
/// notation (`star/private/history.f90:1158` and `binary/private/binary_history.f90:434`).
#[must_use]
pub fn format_mesa_dbl_width(x: f64, width: usize) -> String {
    let mut out = String::with_capacity(width + 1);
    write_mesa_dbl_width(&mut out, x, width);
    out
}

/// Formats an `f64` in 40-character `star` log width (`STAR_LOG_COL_WIDTH`).
#[must_use]
#[inline]
pub fn format_mesa_dbl(x: f64) -> String {
    format_mesa_dbl_width(x, STAR_LOG_COL_WIDTH)
}

/// Appends a 64-bit integer using Fortran `(i<width>, 1x)` directly into `out`.
#[inline]
pub fn write_mesa_int_width(out: &mut String, v: i64, width: usize) {
    let _ = write!(out, "{v:>width$} ");
}

/// Formats a 64-bit integer using Fortran `(i<width>, 1x)`.
#[must_use]
#[inline]
pub fn format_mesa_int_width(v: i64, width: usize) -> String {
    format!("{v:>width$} ")
}

/// Formats a 64-bit integer in 40-character `star` log width (`STAR_LOG_COL_WIDTH`).
#[must_use]
#[inline]
pub fn format_mesa_int(v: i64) -> String {
    format_mesa_int_width(v, STAR_LOG_COL_WIDTH)
}

/// Appends an unquoted identifier/label using Fortran `(a<width>, 1x)` into `out`.
#[inline]
pub fn write_mesa_txt_width(out: &mut String, s: &str, width: usize) {
    let _ = write!(out, "{:>width$} ", s.trim());
}

/// Formats an unquoted identifier/label using Fortran `(a<width>, 1x)`.
#[must_use]
#[inline]
pub fn format_mesa_txt_width(s: &str, width: usize) -> String {
    format!("{:>width$} ", s.trim())
}

/// Formats an unquoted identifier/label in 40-character `star` log width.
#[must_use]
#[inline]
pub fn format_mesa_txt(s: &str) -> String {
    format_mesa_txt_width(s, STAR_LOG_COL_WIDTH)
}

/// Formats a double-quoted header string value using Fortran `(a<width>, 1x)`
/// (`write_string` in `star/private/history.f90:1189` and `binary_history.f90:492`).
#[must_use]
#[inline]
pub fn format_mesa_header_str_width(s: &str, width: usize) -> String {
    let quoted = format!("\"{}\"", s.trim());
    format!("{quoted:>width$} ")
}

/// Formats a double-quoted header string in 40-character `star` log width.
#[must_use]
#[inline]
pub fn format_mesa_header_str(s: &str) -> String {
    format_mesa_header_str_width(s, STAR_LOG_COL_WIDTH)
}

/// Emits the standard 6-line MESA ASCII header (`history.data`, `profileN.data`,
/// or `binary_history.data`) at column width `width`.
pub fn push_mesa_6line_header(
    out: &mut String,
    hdr_items: &[(&str, String)],
    col_names: &[&str],
    width: usize,
) {
    for idx in 1..=hdr_items.len() {
        write_mesa_int_width(out, idx as i64, width);
    }
    out.push('\n');
    for (name, _) in hdr_items {
        write_mesa_txt_width(out, name, width);
    }
    out.push('\n');
    for (_, formatted_val) in hdr_items {
        out.push_str(formatted_val);
    }
    out.push('\n');
    out.push('\n');
    for idx in 1..=col_names.len() {
        write_mesa_int_width(out, idx as i64, width);
    }
    out.push('\n');
    for name in col_names {
        write_mesa_txt_width(out, name, width);
    }
    out.push('\n');
}

/// Parses a single single- or double-quoted argument (e.g. after `include`),
/// returning `None` if quotes are missing or any non-whitespace trailing tokens
/// follow the closing quote.
#[must_use]
pub fn parse_quoted_include_arg(rest: &str) -> Option<&str> {
    let t = rest.trim();
    let bytes = t.as_bytes();
    if bytes.len() >= 2
        && bytes[0] == bytes[bytes.len() - 1]
        && (bytes[0] == b'\'' || bytes[0] == b'"')
    {
        let inner = &t[1..t.len() - 1];
        if !inner.bytes().any(|b| b == bytes[0]) {
            return Some(inner);
        }
    }
    None
}

/// Evaluates the central `dq`-weighted average of a zone closure `f(k)` over
/// `0..n` within `center_avg_value_dq` (`center_value` in
/// `star/private/star_utils.f90:1210-1229`).
#[must_use]
pub fn center_value_by(
    model: &StarModel,
    n: usize,
    mut f: impl FnMut(usize) -> f64,
) -> f64 {
    let n = n.min(model.nz);
    if n == 0 {
        return 0.0;
    }
    let target = model.controls.output.center_avg_value_dq.max(0.0);
    if target <= 0.0 {
        return f(n - 1);
    }
    let mut sum_x = 0.0;
    let mut sum_dq = 0.0;
    for k in (0..n).rev() {
        let dq = model.dq[k];
        let dx = f(k) * dq;
        if sum_dq + dq >= target {
            if dq > 0.0 {
                sum_x += dx * (target - sum_dq) / dq;
            }
            sum_dq = target;
            break;
        }
        sum_x += dx;
        sum_dq += dq;
    }
    if sum_dq > 0.0 {
        sum_x / sum_dq
    } else {
        f(n - 1)
    }
}

/// Evaluates the central `dq`-weighted average of a zone array `p` over
/// `center_avg_value_dq` (`center_value` in `star/private/star_utils.f90:1210-1229`).
#[must_use]
pub fn center_value(model: &StarModel, p: &[f64]) -> f64 {
    center_value_by(model, p.len(), |k| p[k])
}

/// Resolves a catalog `chem_id` (`cid`) to its 0-based active network species slot.
#[must_use]
#[inline]
pub fn iso_slot_for_chem_id(model: &StarModel, cid: usize) -> Option<usize> {
    model
        .net_iso
        .get(cid)
        .copied()
        .flatten()
        .or_else(|| model.chem_id.iter().position(|&id| id == cid))
}

/// Resolves a nuclide name (`"h1"`, `"he4"`, ...) to its 0-based active network
/// species slot (`None` if not in the active network, `Err` if not a valid nuclide).
pub fn resolve_iso_slot(
    model: &StarModel,
    iso_name: &str,
) -> Result<(usize, Option<usize>), String> {
    let cat = NuclideCatalog::builtin_ref();
    let trimmed = iso_name.trim();
    let Some(cid) = cat.get_nuclide_index(trimmed) else {
        return Err(format!("unknown nuclide name {trimmed:?}"));
    };
    Ok((cid, iso_slot_for_chem_id(model, cid)))
}

/// Central `dq`-weighted mass fraction of active species slot `j` over
/// `center_avg_value_dq` (`center_avg_x` in `star/private/star_utils.f90:3632-3655`).
#[must_use]
pub fn center_avg_x_slot(model: &StarModel, slot: Option<usize>) -> f64 {
    let Some(j) = slot.filter(|&j| j < model.species) else {
        return 0.0;
    };
    center_value_by(model, model.nz, |k| model.cell_xa(k)[j])
}

/// Surface `dq`-weighted mass fraction of active species slot `j` over
/// `surface_avg_abundance_dq` (`surface_avg_x` in `star/private/star_utils.f90:3611-3629`).
#[must_use]
pub fn surface_avg_x_slot(model: &StarModel, slot: Option<usize>) -> f64 {
    let Some(j) = slot.filter(|&j| j < model.species) else {
        return 0.0;
    };
    if model.nz == 0 {
        return 0.0;
    }
    let target = model.controls.output.surface_avg_abundance_dq;
    let mut sum_x = 0.0;
    let mut sum_dq = 0.0;
    for k in 0..model.nz {
        let dq = model.dq[k];
        sum_x += model.cell_xa(k)[j] * dq;
        sum_dq += dq;
        if sum_dq >= target {
            break;
        }
    }
    if sum_dq > 0.0 {
        sum_x / sum_dq
    } else {
        model.cell_xa(0)[j]
    }
}

/// Envelope-averaged mass fraction of active species slot `j`
/// (`star_avg_x` in `star/private/history.f90:3209-3217`).
#[must_use]
pub fn star_avg_x_slot(model: &StarModel, slot: Option<usize>) -> f64 {
    let Some(j) = slot.filter(|&j| j < model.species) else {
        return 0.0;
    };
    if model.nz == 0 || model.xmstar <= 0.0 {
        return 0.0;
    }
    let mut sum_dq_x = 0.0;
    let mut sum_dq = 0.0;
    for k in 0..model.nz {
        sum_dq_x += model.dq[k] * model.cell_xa(k)[j];
        sum_dq += model.dq[k];
    }
    if sum_dq > 0.0 {
        sum_dq_x / sum_dq
    } else {
        0.0
    }
}

/// Central `dq`-weighted mass fraction by nuclide name (`"h1"`, `"he4"`, `"c12"`, ...).
#[must_use]
pub fn center_avg_x_by_name(model: &StarModel, iso_name: &str) -> f64 {
    let slot = resolve_iso_slot(model, iso_name).ok().and_then(|(_, s)| s);
    center_avg_x_slot(model, slot)
}

/// Surface `dq`-weighted mass fraction by nuclide name.
#[must_use]
pub fn surface_avg_x_by_name(model: &StarModel, iso_name: &str) -> f64 {
    let slot = resolve_iso_slot(model, iso_name).ok().and_then(|(_, s)| s);
    surface_avg_x_slot(model, slot)
}

/// Face-interpolated adiabatic sound speed `csound_face(k)` (`eval_csound` / face
/// interpolation in `star/private/star_utils.f90`).
#[must_use]
#[inline]
pub fn csound_face_at(model: &StarModel, k: usize) -> f64 {
    if model.nz == 0 {
        0.0
    } else if k == 0 || k >= model.nz {
        model.csound[0]
    } else {
        model.interp_val_to_pt(&model.csound, k)
    }
}

/// Shellular equatorial and polar radii `(r_equatorial, r_polar)` in cm at face `k`
/// (`eval_fp_ft` / `re_from_rpsi_factor`, Fabry+2022 Eq. A.3 in
/// `star/private/hydro_rotation.f90:789-795, 882-888`).
#[must_use]
#[inline]
pub fn equatorial_and_polar_r_cm(model: &StarModel, k: usize) -> (f64, f64) {
    let r = model.r.get(k).copied().unwrap_or(0.0);
    if !model.controls.rotation_flag {
        return (r, r);
    }
    let w = model.aux.w_div_wc.get(k).copied().unwrap_or(0.0).abs();
    let w2 = w * w;
    let w4 = w2 * w2;
    let w6 = w2 * w4;
    let re_fac = 1.0 + (1.0 / 6.0) * w2 - 0.005124 * w4 + 0.06562 * w6;
    let r_eq = (r * re_fac).max(r);
    let r_pol = (r_eq / (1.0 + 0.5 * w2)).min(r);
    (r_eq, r_pol)
}

/// Local face Eddington luminosity `get_Ledd(s, k)` in $\mathrm{erg\,s^{-1}}$
/// (`star/private/star_utils.f90:2063-2075`, `0`-based `k`).
#[must_use]
pub fn get_ledd(model: &StarModel, k: usize) -> f64 {
    if model.nz < 2 || k >= model.nz {
        return 0.0;
    }
    let j = if k == 0 { 1 } else { k };
    let kap_face = model.interp_val_to_pt(&model.opacity, j);
    if kap_face <= 0.0 {
        return 0.0;
    }
    FOUR_PI * CLIGHT * model.cgrav[j] * model.m_grav[j] / kap_face
}

/// Local radiative-to-Eddington luminosity ratio `get_Lrad_div_Ledd(s, k)`
/// (`star/private/star_utils.f90:2080-2114`, `0`-based `k`).
#[must_use]
pub fn get_lrad_div_ledd(model: &StarModel, k: usize) -> f64 {
    if model.nz < 2 || k >= model.nz || model.cgrav[k] <= 0.0 {
        return 0.0;
    }
    let j = if k == 0 { 1 } else { k };
    let del_m = model.dm_bar[j];
    let denom = FOUR_PI * model.cgrav[j] * model.m_grav[j];
    if del_m <= 0.0 || denom <= 0.0 {
        return 0.0;
    }
    let del_t4 =
        flexmesa_math::pow4(model.t[j - 1]) - flexmesa_math::pow4(model.t[j]);
    let area = FOUR_PI * model.r[j] * model.r[j];
    -(area * area * CRAD * (del_t4 / del_m) / 3.0) / denom
}

/// Local critical angular velocity `omega_crit(s, k)` in $\mathrm{rad\,s^{-1}}$
/// (`star/private/star_utils.f90:2752-2774`, `0`-based `k`).
#[must_use]
pub fn eval_upstream_omega_crit(model: &StarModel, k: usize) -> f64 {
    let nz = model.nz;
    if k >= nz {
        return 0.0;
    }
    let rmid = if model.controls.rotation_flag {
        let r_eq0 = equatorial_and_polar_r_cm(model, k).0;
        let r003 = r_eq0 * r_eq0 * r_eq0;
        let rp13 = if k + 1 < nz {
            let r_eq1 = equatorial_and_polar_r_cm(model, k + 1).0;
            r_eq1 * r_eq1 * r_eq1
        } else {
            model.r_center * model.r_center * model.r_center
        };
        flexmesa_math::pow(0.5 * (r003 + rp13), 1.0 / 3.0)
    } else {
        model.rmid[k]
    };
    if rmid <= 0.0 {
        return 0.0;
    }
    let lrad_div_ledd = get_lrad_div_ledd(model, k);
    let gamma_factor = 1.0 - lrad_div_ledd.min(0.9999);
    (gamma_factor * model.cgrav[k] * model.m_grav[k] / (rmid * rmid * rmid))
        .max(0.0)
        .sqrt()
}

/// Local Brunt-based convective timescale `conv_time_scale(s, k_in)` in seconds
/// (`star/private/star_utils.f90:3700-3728`, `0`-based `k_in`).
#[must_use]
pub fn eval_conv_time_scale_zone(model: &StarModel, k_in: usize) -> f64 {
    let nz = model.nz;
    if !model.controls.calculate_brunt_b || nz < 2 || k_in >= nz {
        return 0.0;
    }
    let k = k_in.max(1);
    let brunt_b = model.brunt_b[k];
    if brunt_b.abs() <= 0.0 {
        return 0.0;
    }
    let denom_dq = model.dq[k - 1] + model.dq[k];
    if denom_dq <= 0.0 {
        return 0.0;
    }
    let alfa = model.dq[k - 1] / denom_dq;
    let beta = 1.0 - alfa;
    let rho_face = alfa * model.rho[k] + beta * model.rho[k - 1];
    let peos_face = alfa * model.peos[k] + beta * model.peos[k - 1];
    let chit_face = alfa * model.chi_t[k] + beta * model.chi_t[k - 1];
    let chirho_face = alfa * model.chi_rho[k] + beta * model.chi_rho[k - 1];
    if peos_face <= 0.0 || chirho_face == 0.0 {
        return 0.0;
    }
    let f = (model.grav[k] * model.grav[k])
        * (rho_face / peos_face)
        * (chit_face / chirho_face);
    let dlnp = model.ln_peos[k - 1] - model.ln_peos[k];
    let dlnt = model.ln_t[k - 1] - model.ln_t[k];
    let grada_face = alfa * model.grada[k] + beta * model.grada[k - 1];
    let gradt_actual = if dlnp.abs() < 1.0e-20 {
        0.0
    } else {
        dlnt / dlnp
    };
    let brunt_n2 = f * (brunt_b - (gradt_actual - grada_face));
    if brunt_n2 == 0.0 {
        0.0
    } else {
        1.0 / brunt_n2.abs().sqrt()
    }
}

/// Domain-filtered `(min_conv_time_scale, max_conv_time_scale)` in seconds
/// (`set_conv_time_scales` in `star/private/star_utils.f90:3731-3751`).
#[must_use]
pub fn eval_conv_time_scales(model: &StarModel) -> (f64, f64) {
    let nz = model.nz;
    let ts = &model.controls.output;
    let mut min_tau = 1.0e99_f64;
    let mut max_tau = 0.0_f64;
    for k in 0..nz {
        let x_k = model.xh_mass.get(k).copied().unwrap_or(0.0);
        if x_k > ts.max_x_for_conv_timescale
            || x_k < ts.min_x_for_conv_timescale
        {
            continue;
        }
        if model.q[k] > ts.max_q_for_conv_timescale {
            continue;
        }
        if model.q[k] < ts.min_q_for_conv_timescale {
            break;
        }
        let tau = eval_conv_time_scale_zone(model, k);
        if tau < min_tau {
            min_tau = tau;
        }
        if tau > max_tau {
            max_tau = tau;
        }
    }
    if max_tau == 0.0 {
        max_tau = 1.0e99;
    }
    if min_tau == 1.0e99 {
        min_tau = 0.0;
    }
    (min_tau, max_tau)
}

/// Optical-depth-weighted Eddington luminosity `eval_Ledd` in $\mathrm{erg\,s^{-1}}$
/// (`star/private/star_utils.f90:1771-1796`).
#[must_use]
pub fn eval_ledd_erg_s(model: &StarModel) -> f64 {
    let nz = model.nz;
    if nz == 0 || model.cgrav[0] <= 0.0 || model.opacity[0] <= 0.0 {
        return 0.0;
    }
    let surf_avg_tau = model.controls.output.surf_avg_tau;
    let mut dtau = model.dm[0] * model.opacity[0]
        / (FOUR_PI * model.rmid[0] * model.rmid[0]).max(1.0e-99);
    let mut tau = model.controls.tau_factor * model.tau_base;
    let mut dqsum = model.dq[0];
    let num0 = FOUR_PI * CLIGHT * model.cgrav[0] * model.m_grav[0];
    let mut ledd_sum = model.dq[0] * num0 / model.opacity[0];
    for k in 1..nz {
        tau += dtau;
        if tau > surf_avg_tau {
            break;
        }
        dtau = model.dm[k] * model.opacity[k]
            / (FOUR_PI * model.rmid[k] * model.rmid[k]).max(1.0e-99);
        if model.opacity[k] > 0.0 {
            dqsum += model.dq[k];
            ledd_sum += model.dq[k] * num0 / model.opacity[k];
        }
    }
    if dqsum > 0.0 {
        ledd_sum / dqsum
    } else {
        0.0
    }
}

/// Photosphere summary matching `get_phot_info` + `set_phot_info`
/// (`star/private/star_utils.f90:1094-1207`).
#[derive(Debug, Clone, Copy)]
pub struct PhotosphereInfo {
    /// Photospheric radius in $R_\odot$ (`photosphere_r`).
    pub r_rsun: f64,
    /// Photospheric enclosed mass in $M_\odot$ (`photosphere_m`).
    pub m_msun: f64,
    /// Photospheric radial velocity in $\mathrm{cm\,s^{-1}}$ (`photosphere_v`).
    pub v_cm_s: f64,
    /// Photospheric luminosity in $L_\odot$ (`photosphere_L`).
    pub l_lsun: f64,
    /// Photospheric gas temperature in $\mathrm{K}$ (`photosphere_T`).
    pub t_phot: f64,
    /// Photospheric black-body temperature in $\mathrm{K}$ (`photosphere_black_body_T`).
    pub black_body_t: f64,
    /// Photospheric sound speed in $\mathrm{cm\,s^{-1}}$ (`photosphere_csound`).
    pub csound: f64,
    /// Photospheric opacity in $\mathrm{cm^2\,g^{-1}}$ (`photosphere_opacity`).
    pub opacity: f64,
    /// Photospheric $\log_{10} g$ (`photosphere_logg`).
    pub logg: f64,
    /// Photospheric column density in $\mathrm{g\,cm^{-2}}$ (`photosphere_column_density`).
    pub column_density: f64,
    /// 1-based cell index of photosphere (`photosphere_cell_k`).
    pub cell_k_1based: usize,
}

/// Evaluates [`PhotosphereInfo`] reproducing `get_phot_info` and `set_phot_info`
/// (`star/private/star_utils.f90:1094-1207`, including upstream's exact `tau_factor < 1`
/// branch at lines `1186-1189`).
#[must_use]
pub fn eval_photosphere_info(model: &StarModel) -> PhotosphereInfo {
    let nz = model.nz;
    if nz == 0 {
        return PhotosphereInfo {
            r_rsun: 0.0,
            m_msun: 0.0,
            v_cm_s: 0.0,
            l_lsun: 0.0,
            t_phot: model.teff,
            black_body_t: model.teff,
            csound: 0.0,
            opacity: 0.0,
            logg: -99.0,
            column_density: 0.0,
            cell_k_1based: 1,
        };
    }
    let mut r = model.r[0];
    let mut m = model.m[0];
    let mut v = if model.controls.u_flag {
        model.u[0]
    } else if model.controls.v_flag {
        model.v[0]
    } else {
        0.0
    };
    let mut l_erg = model.lum[0].max(1.0);
    let mut t_phot = model.t[0];
    let mut cs = model.csound[0];
    let mut kap = model.opacity[0];
    let mut logg = safe_log10(model.cgrav[0] * m / (r * r).max(1.0e-99));
    let mut k_phot_0 = 0_usize;
    let mut ysum = 0.0;

    if model.controls.tau_factor < 1.0 && nz >= 2 {
        let tau_phot = model.tau_base;
        let mut tau00 = model.controls.tau_factor * model.tau_base;
        let mut found = false;
        for k in 0..(nz - 1) {
            let dtau = model.dm[k] * model.opacity[k]
                / (FOUR_PI * model.rmid[k] * model.rmid[k]).max(1.0e-99);
            let taup1 = tau00 + dtau;
            ysum += model.rho[k] * (model.r[k] - model.r[k + 1]);
            if taup1 >= tau_phot && dtau > 0.0 {
                let frac = (tau_phot - tau00) / dtau;
                let tface_0 = if k == 0 {
                    flexmesa_math::exp(model.ln_t_surf_ad.val)
                } else {
                    0.5 * (model.t[k] + model.t[k - 1])
                };
                let tface_1 = 0.5 * (model.t[k] + model.t[k + 1]);
                t_phot = tface_0 + (tface_1 - tface_0) * frac;
                let r003 = flexmesa_math::pow3(model.r[k]);
                let rp13 = flexmesa_math::pow3(model.r[k + 1]);
                r = flexmesa_math::pow(
                    (r003 + (rp13 - r003) * frac).max(0.0),
                    1.0 / 3.0,
                );
                m = model.m[k] - model.dm[k] * frac;
                if model.controls.u_flag {
                    v = model.v_center;
                } else if model.controls.v_flag {
                    v = model.v[k] + (model.v[k + 1] - model.v[k]) * frac;
                }
                // Reproduces `star/private/star_utils.f90:1186-1189` verbatim
                // (surface `L` and surface `cgrav`; `#182` under `#169`).
                l_erg = model.lum[0].max(1.0);
                logg = safe_log10(model.cgrav[0] * m / (r * r).max(1.0e-99));
                k_phot_0 = k;
                cs = model.csound[k_phot_0];
                kap = model.opacity[k_phot_0];
                found = true;
                break;
            }
            tau00 = taup1;
        }
        if !found {
            k_phot_0 = nz - 1;
            r = model.r_center;
            m = model.m_center;
            v = model.v_center;
            t_phot = model.t[k_phot_0];
            l_erg = model.l_center.max(1.0);
            cs = model.csound[k_phot_0];
            kap = model.opacity[k_phot_0];
            logg = safe_log10(model.cgrav[k_phot_0] * m / (r * r).max(1.0e-99));
        }
    }

    let bb_t = if r > 0.0 && l_erg > 0.0 {
        flexmesa_math::pow(l_erg / (FOUR_PI * r * r * BOLTZ_SIGMA), 0.25)
    } else {
        model.teff
    };
    PhotosphereInfo {
        r_rsun: r / R_SUN,
        m_msun: m / M_SUN,
        v_cm_s: v,
        l_lsun: l_erg / L_SUN,
        t_phot,
        black_body_t: bb_t,
        csound: cs,
        opacity: kap,
        logg,
        column_density: ysum,
        cell_k_1based: k_phot_0 + 1,
    }
}

/// Surface optical-depth-averaged rotation diagnostics matching
/// `set_surf_avg_rotation_info` (`star/private/hydro_rotation.f90:588-739`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SurfAvgRotationInfo {
    /// Optical-depth-averaged $\Omega_{\mathrm{surf}}$ in $\mathrm{rad\,s^{-1}}$ (`omega_avg_surf`).
    pub omega: f64,
    /// Optical-depth-averaged $\Omega_{\mathrm{crit,surf}}$ in $\mathrm{rad\,s^{-1}}$ (`omega_crit_avg_surf`).
    pub omega_crit: f64,
    /// Optical-depth-averaged $|\Omega / \Omega_{\mathrm{crit}}|$ (`w_div_w_crit_avg_surf`).
    pub w_div_w_crit: f64,
    /// Optical-depth-averaged $j_{\mathrm{rot,surf}}$ in $\mathrm{cm^2\,s^{-1}}$ (`j_rot_avg_surf`).
    pub j_rot: f64,
    /// Optical-depth-averaged $v_{\mathrm{rot,surf}}$ in $\mathrm{cm\,s^{-1}}$ (`v_rot_avg_surf`).
    pub v_rot_cm_s: f64,
    /// Optical-depth-averaged $v_{\mathrm{crit,surf}}$ in $\mathrm{cm\,s^{-1}}$ (`v_crit_avg_surf`).
    pub v_crit_cm_s: f64,
    /// Optical-depth-averaged $|v_{\mathrm{rot}} / v_{\mathrm{crit}}|$ (`v_div_v_crit_avg_surf`).
    pub v_div_v_crit: f64,
    /// Optical-depth-averaged $L_{\mathrm{rad}} / L_{\mathrm{Edd}}$ (`Lrad_div_Ledd_avg_surf`).
    pub lrad_div_ledd: f64,
    /// Optical-depth-averaged opacity (`opacity_avg_surf`; reproduces upstream double `kap_sum` at `hydro_rotation.f90:707,722`).
    pub opacity: f64,
    /// Optical-depth-averaged $\log_{10} T$ (`logT_avg_surf`).
    pub log_t: f64,
    /// Optical-depth-averaged $\log_{10} \rho$ (`logRho_avg_surf`).
    pub log_rho: f64,
}

/// Evaluates [`SurfAvgRotationInfo`] matching `set_surf_avg_rotation_info`
/// (`star/private/hydro_rotation.f90:588-739`).
#[must_use]
pub fn eval_surf_avg_rotation_info(model: &StarModel) -> SurfAvgRotationInfo {
    if !model.controls.rotation_flag || model.nz < 2 {
        return SurfAvgRotationInfo::default();
    }
    let tau_min = model.controls.output.surf_avg_tau_min;
    let tau_max = model.controls.output.surf_avg_tau;

    let mut tau = model.controls.tau_factor * model.tau_base;
    let mut dmsum = 0.0;
    let mut lrad_div_ledd_sum = 0.0;
    for k in 0..(model.nz - 1) {
        let kap = model.opacity[k];
        let rmid = model.rmid[k];
        let mut dm = model.dm[k];
        let dtau = dm * kap / (FOUR_PI * rmid * rmid).max(1.0e-99);
        if tau + dtau <= tau_min {
            tau += dtau;
            continue;
        }
        if tau < tau_max && dtau > 0.0 {
            if tau < tau_min {
                dm *= (tau + dtau - tau_min) / dtau;
            } else if tau + dtau > tau_max {
                dm *= (tau_max - tau) / dtau;
            }
        }
        dmsum += dm;
        lrad_div_ledd_sum += dm * get_lrad_div_ledd(model, k);
        tau += dtau;
        if tau >= tau_max {
            break;
        }
    }
    let lrad_div_ledd = if dmsum > 0.0 {
        lrad_div_ledd_sum / dmsum
    } else {
        0.0
    };
    let gamma_factor = 1.0 - lrad_div_ledd.min(0.9999);

    tau = model.controls.tau_factor * model.tau_base;
    dmsum = 0.0;
    let (
        mut j_rot_sum,
        mut omega_sum,
        mut omega_crit_sum,
        mut w_div_wc_sum,
        mut v_rot_sum,
        mut v_crit_sum,
        mut v_div_vc_sum,
        mut kap_sum,
        mut log_t_sum,
        mut log_rho_sum,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for k in 0..(model.nz - 1) {
        let kap = model.opacity[k];
        let (re_k, _) = equatorial_and_polar_r_cm(model, k);
        let (re_kp1, _) = equatorial_and_polar_r_cm(model, k + 1);
        let rmid = 0.5 * (re_k + re_kp1);
        let mut dm = model.dm[k];
        let dtau = dm * kap / (FOUR_PI * rmid * rmid).max(1.0e-99);
        if tau + dtau <= tau_min {
            tau += dtau;
            continue;
        }
        if tau < tau_max && dtau > 0.0 {
            if tau < tau_min {
                dm *= (tau + dtau - tau_min) / dtau;
            } else if tau + dtau > tau_max {
                dm *= (tau_max - tau) / dtau;
            }
        }
        dmsum += dm;
        let cgrav = 0.5 * (model.cgrav[k] + model.cgrav[k + 1]);
        let mmid = 0.5 * (model.m_grav[k] + model.m_grav[k + 1]);
        let omega = 0.5 * (model.omega[k] + model.omega[k + 1]);
        let j_rot = 0.5 * (model.j_rot[k] + model.j_rot[k + 1]);

        // Reproduces `hydro_rotation.f90:707` AND `:722` verbatim (`kap_sum`
        // accumulated twice; `#181` under `#169`).
        kap_sum += dm * kap;
        j_rot_sum += dm * j_rot;

        let omega_crit = (gamma_factor * cgrav * mmid
            / flexmesa_math::pow3(rmid).max(1.0e-99))
        .max(0.0)
        .sqrt();
        let w_ratio = if omega_crit > 0.0 {
            (omega / omega_crit).abs()
        } else {
            0.0
        };
        w_div_wc_sum += dm * w_ratio;
        let v_rot = omega * rmid;
        let v_crit = omega_crit * rmid;
        omega_sum += dm * omega;
        omega_crit_sum += dm * omega_crit;
        v_rot_sum += dm * v_rot;
        v_crit_sum += dm * v_crit;
        if v_crit > 0.0 {
            v_div_vc_sum += dm * (v_rot / v_crit).abs();
        }
        log_t_sum += dm * model.ln_t[k] / LN10;
        log_rho_sum += dm * model.lnd[k] / LN10;
        kap_sum += dm * kap;
        tau += dtau;
        if tau >= tau_max {
            break;
        }
    }
    if dmsum <= 0.0 {
        return SurfAvgRotationInfo {
            lrad_div_ledd,
            ..SurfAvgRotationInfo::default()
        };
    }
    SurfAvgRotationInfo {
        omega: omega_sum / dmsum,
        omega_crit: omega_crit_sum / dmsum,
        w_div_w_crit: w_div_wc_sum / dmsum,
        j_rot: j_rot_sum / dmsum,
        v_rot_cm_s: v_rot_sum / dmsum,
        v_crit_cm_s: v_crit_sum / dmsum,
        v_div_v_crit: v_div_vc_sum / dmsum,
        lrad_div_ledd,
        opacity: kap_sum / dmsum,
        log_t: log_t_sum / dmsum,
        log_rho: log_rho_sum / dmsum,
    }
}

/// Sub-cell convective/semiconvective boundary offset `cz_bdy_dq` matching
/// `set_cz_boundary_info` (`star/private/mix_info.f90:428-469`).
#[must_use]
pub fn eval_cz_bdy_dq(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    let mut cz_bdy_dq = vec![0.0; nz];
    if nz < 2 {
        return cz_bdy_dq;
    }
    for k in 0..(nz - 1) {
        let mt1 = model.mixing_type[k];
        let mt2 = model.mixing_type[k + 1];
        if mt1 == mt2 {
            continue;
        }
        let (dg0, dg1) = if mt1 == MixingType::ConvectiveMixing
            || mt2 == MixingType::ConvectiveMixing
        {
            (
                model.gradl[k] - model.gradr[k],
                model.gradl[k + 1] - model.gradr[k + 1],
            )
        } else if mt1 == MixingType::SemiconvectiveMixing
            || mt2 == MixingType::SemiconvectiveMixing
        {
            let ga0 = if k == 0 {
                model.grada[0]
            } else {
                model.interp_val_to_pt(&model.grada, k)
            };
            let ga1 = model.interp_val_to_pt(&model.grada, k + 1);
            (ga0 - model.gradr[k], ga1 - model.gradr[k + 1])
        } else {
            continue;
        };
        if dg0 * dg1 >= 0.0 {
            continue;
        }
        if let Ok(x0) = flexmesa_num::find0(0.0, dg0, model.dq[k], dg1) {
            cz_bdy_dq[k] = x0.clamp(0.0, model.dq[k]);
        }
    }
    cz_bdy_dq
}

/// Convective-core mass and top-2 convective / total mixing region bounds
/// (`star/private/mix_info.f90:472-529` and `star/private/report.f90:456-501, 595-730, 1389-1417`).
#[derive(Debug, Clone, Default)]
pub struct ConvAndMixSummary {
    /// Sub-cell boundary offsets `cz_bdy_dq[0..nz]`.
    pub cz_bdy_dq: Vec<f64>,
    /// Convective core mass in $M_\odot$ (`mass_conv_core`).
    pub mass_conv_core: f64,
    /// Semiconvective + convective core mass in $M_\odot$ (`mass_semiconv_core`).
    pub mass_semiconv_core: f64,
    /// 0-based `(top, bot)` indices of largest convective mixing region (`None` if none).
    pub largest_conv_top_bot: Option<(usize, usize)>,
    /// `[conv_mx1_top, conv_mx1_bot, conv_mx2_top, conv_mx2_bot]` in $q$.
    pub conv_mx_q: [f64; 4],
    /// `[conv_mx1_top_r, conv_mx1_bot_r, conv_mx2_top_r, conv_mx2_bot_r]` in $R_\odot$.
    pub conv_mx_r: [f64; 4],
    /// `[mx1_top, mx1_bot, mx2_top, mx2_bot]` in $q$.
    pub mx_q: [f64; 4],
    /// `[mx1_top_r, mx1_bot_r, mx2_top_r, mx2_bot_r]` in $R_\odot$.
    pub mx_r: [f64; 4],
}

/// Evaluates [`ConvAndMixSummary`] matching `mix_info.f90:472-529` and
/// `report.f90:456-501, 595-730, 1389-1417`.
#[must_use]
pub fn eval_conv_and_mx_summary(model: &StarModel) -> ConvAndMixSummary {
    let nz = model.nz;
    if nz == 0 || model.controls.rsp_flag {
        return ConvAndMixSummary {
            cz_bdy_dq: vec![0.0; nz],
            ..ConvAndMixSummary::default()
        };
    }
    let cz_bdy_dq = if model.aux.mixing.have_mixing_info
        && model.aux.mixing.cz_bdy_dq.len() == nz
    {
        model.aux.mixing.cz_bdy_dq.clone()
    } else {
        eval_cz_bdy_dq(model)
    };

    // 1. `set_cz_bdy_mass` (`mix_info.f90:472-517`, ordered center -> surface).
    let mut cz_regions: Vec<(f64, f64)> = Vec::new();
    let mut in_conv = model.mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    let mut cur_bot_mass = model.m_center;
    if nz >= 3 {
        for k in (1..(nz - 1)).rev() {
            let is_conv = model.mixing_type[k] == MixingType::ConvectiveMixing;
            if in_conv {
                if !is_conv {
                    let top_m = model.m_center
                        + (model.q[k] - cz_bdy_dq[k]) * model.xmstar;
                    cz_regions.push((cur_bot_mass, top_m));
                    in_conv = false;
                }
            } else if is_conv {
                cur_bot_mass =
                    model.m_center + (model.q[k] - cz_bdy_dq[k]) * model.xmstar;
                in_conv = true;
            }
        }
    }
    if in_conv {
        cz_regions.push((cur_bot_mass, model.mstar));
    }

    // 2. `set_mass_conv_core` (`report.f90:456-475`).
    let gap_dq = model.controls.conv_core_gap_dq_limit.max(0.0);
    let dm_limit = gap_dq * model.xmstar;
    let mut mass_conv_core = 0.0;
    for (j, &(bot_m, top_m)) in cz_regions.iter().enumerate() {
        if bot_m <= model.m[nz - 1] + dm_limit {
            mass_conv_core = top_m / M_SUN;
            let mut prev_top = top_m;
            for &(next_bot, next_top) in &cz_regions[(j + 1)..] {
                if next_bot - prev_top >= dm_limit {
                    break;
                }
                mass_conv_core = next_top / M_SUN;
                prev_top = next_top;
            }
            break;
        }
    }

    // 3. `set_mass_semiconv_core` (`star/private/report.f90:478-501`).
    // Verbatim reproduction of upstream MESA `26.4.1` (`#184` under `#169`):
    // `report.f90:486-492` exits the core scan on `mixing_type(k) == convective_mixing`
    // and then immediately returns because `mixing_type(ktop) /= semiconvective_mixing`,
    // leaving `mass_semiconv_core == mass_conv_core` unconditionally.
    let mass_semiconv_core = mass_conv_core;

    // 4. `get_mixing_regions` (`report.f90:1389-1417`, surface -> center).
    let mut regions: Vec<(MixingType, usize, usize)> = Vec::new();
    let mut cur_type = model.mixing_type[0];
    let mut cur_top = 0_usize;
    for k in 1..nz {
        let prev_type = cur_type;
        cur_type = model.mixing_type[k];
        if cur_type == prev_type && k + 1 < nz {
            continue;
        }
        if prev_type != MixingType::NoMixing {
            let bot = if k + 1 == nz { k } else { k - 1 };
            regions.push((prev_type, cur_top, bot));
            if regions.len() == 1000 {
                break;
            }
        }
        cur_top = k;
    }

    // 5. `find_conv_mx_regions` (`report.f90:595-652`).
    let mut conv1: Option<(f64, usize, usize)> = None;
    let mut conv2: Option<(f64, usize, usize)> = None;
    for &(mt, top, bot) in &regions {
        if mt != MixingType::ConvectiveMixing {
            continue;
        }
        let mx_dq: f64 = model.dq[top..=bot].iter().sum();
        if conv1.is_none_or(|(dq1, _, _)| mx_dq > dq1) {
            conv2 = conv1;
            conv1 = Some((mx_dq, top, bot));
        } else if conv2.is_none_or(|(dq2, _, _)| mx_dq > dq2) {
            conv2 = Some((mx_dq, top, bot));
        }
    }
    let mut conv_mx_q = [0.0; 4];
    let mut conv_mx_r = [0.0; 4];
    let largest_conv_top_bot = conv1.map(|(_, top, bot)| {
        conv_mx_q[0] = model.q[top] - cz_bdy_dq[top];
        conv_mx_q[1] = model.q[bot] - cz_bdy_dq[bot];
        conv_mx_r[0] = model.r[top] / R_SUN;
        conv_mx_r[1] = model.r[bot] / R_SUN;
        (top, bot)
    });
    if let Some((_, top, bot)) = conv2 {
        conv_mx_q[2] = model.q[top] - cz_bdy_dq[top];
        conv_mx_q[3] = model.q[bot] - cz_bdy_dq[bot];
        conv_mx_r[2] = model.r[top] / R_SUN;
        conv_mx_r[3] = model.r[bot] / R_SUN;
    }

    // 6. `find_mx_regions` (`report.f90:655-730`).
    let mut mx1: Option<(f64, usize, usize)> = None;
    let mut mx2: Option<(f64, usize, usize)> = None;
    let mut idx = 0_usize;
    while idx < regions.len() {
        let top_pt = regions[idx].1;
        let mut bot_pt = regions[idx].2;
        idx += 1;
        while idx < regions.len() && regions[idx].1 == bot_pt + 1 {
            bot_pt = regions[idx].2;
            idx += 1;
        }
        let mx_dq: f64 = model.dq[top_pt..=bot_pt].iter().sum();
        if mx1.is_none_or(|(dq1, _, _)| mx_dq > dq1) {
            mx2 = mx1;
            mx1 = Some((mx_dq, top_pt, bot_pt));
        } else if mx2.is_none_or(|(dq2, _, _)| mx_dq > dq2) {
            mx2 = Some((mx_dq, top_pt, bot_pt));
        }
    }
    let mut mx_q = [0.0; 4];
    let mut mx_r = [0.0; 4];
    if let Some((_, top, bot)) = mx1 {
        mx_q[0] = model.q[top] - cz_bdy_dq[top];
        mx_q[1] = model.q[bot] - cz_bdy_dq[bot];
        mx_r[0] = model.r[top] / R_SUN;
        mx_r[1] = model.r[bot] / R_SUN;
    }
    if let Some((_, top, bot)) = mx2 {
        mx_q[2] = model.q[top] - cz_bdy_dq[top];
        mx_q[3] = model.q[bot] - cz_bdy_dq[bot];
        mx_r[2] = model.r[top] / R_SUN;
        mx_r[3] = model.r[bot] / R_SUN;
    }

    ConvAndMixSummary {
        cz_bdy_dq,
        mass_conv_core,
        mass_semiconv_core,
        largest_conv_top_bot,
        conv_mx_q,
        conv_mx_r,
        mx_q,
        mx_r,
    }
}

/// Evaluates up to 3 burning-zone enclosed-mass contours `epsnuc_out[0..12]` in grams
/// (`get_burn_zone_info` + `find_epsnuc_zone` in `star/private/report.f90:735-851`,
/// `-20.0` sentinel when absent).
#[must_use]
pub fn eval_epsnuc_zones(model: &StarModel) -> [f64; 12] {
    const NULL_ZONE: f64 = -20.0;
    let mut out = [NULL_ZONE; 12];
    let nz = model.nz;
    if nz == 0 || model.controls.rsp_flag {
        return out;
    }
    let burn_min1 = model.controls.output.burn_min1;
    let burn_min2 = model.controls.output.burn_min2;
    let find0_m = |x1: f64, y1: f64, x2: f64, y2: f64| -> f64 {
        flexmesa_num::find0(x1, y1, x2, y2).unwrap_or(0.5 * (x1 + x2))
    };

    let mut i_start: isize = (nz as isize) - 1;
    for zone_slot in 0..3 {
        let base = 4 * zone_slot;
        if i_start < 0 {
            break;
        }
        let (mut b1, mut b2, mut b3, mut b4) =
            (NULL_ZONE, NULL_ZONE, NULL_ZONE, NULL_ZONE);
        let mut burn_zone = 0_u8;
        let (mut prev_m, mut prev_x) = if (i_start as usize) + 1 < nz {
            let ip = (i_start as usize) + 1;
            (model.m[ip], model.eps_nuc[ip])
        } else {
            (0.0, 0.0)
        };
        let mut cur_m = 0.0;
        let mut finished_early = false;
        let mut idx = i_start;
        while idx >= 0 {
            let k = idx as usize;
            cur_m = model.m[k];
            let cur_x = model.eps_nuc[k];
            match burn_zone {
                0 => {
                    if cur_x > burn_min2 {
                        b2 = if k + 1 == nz {
                            0.0
                        } else {
                            find0_m(
                                prev_m,
                                prev_x - burn_min2,
                                cur_m,
                                cur_x - burn_min2,
                            )
                        };
                        b1 = b2;
                        burn_zone = 2;
                    } else if cur_x > burn_min1 {
                        b1 = if k + 1 == nz {
                            0.0
                        } else {
                            find0_m(
                                prev_m,
                                prev_x - burn_min1,
                                cur_m,
                                cur_x - burn_min1,
                            )
                        };
                        burn_zone = 1;
                    }
                }
                1 => {
                    if cur_x > burn_min2 {
                        b2 = find0_m(
                            prev_m,
                            prev_x - burn_min2,
                            cur_m,
                            cur_x - burn_min2,
                        );
                        burn_zone = 2;
                    } else if cur_x < burn_min1 {
                        b4 = find0_m(
                            prev_m,
                            prev_x - burn_min1,
                            cur_m,
                            cur_x - burn_min1,
                        );
                        i_start = idx;
                        finished_early = true;
                        break;
                    }
                }
                2 => {
                    if cur_x < burn_min1 {
                        b4 = find0_m(
                            prev_m,
                            prev_x - burn_min1,
                            cur_m,
                            cur_x - burn_min1,
                        );
                        b3 = b4;
                        i_start = idx;
                        finished_early = true;
                        break;
                    }
                    if cur_x < burn_min2 {
                        b3 = find0_m(
                            prev_m,
                            prev_x - burn_min2,
                            cur_m,
                            cur_x - burn_min2,
                        );
                        burn_zone = 3;
                    }
                }
                _ => {
                    if cur_x < burn_min1 {
                        b4 = find0_m(
                            prev_m,
                            prev_x - burn_min1,
                            cur_m,
                            cur_x - burn_min1,
                        );
                        i_start = idx;
                        finished_early = true;
                        break;
                    }
                }
            }
            prev_m = cur_m;
            prev_x = cur_x;
            idx -= 1;
        }
        if !finished_early {
            i_start = -1;
            match burn_zone {
                0 => {}
                1 | 3 => b4 = cur_m,
                _ => {
                    b3 = cur_m;
                    b4 = cur_m;
                }
            }
        }
        out[base] = b1;
        out[base + 1] = b2;
        out[base + 2] = b3;
        out[base + 3] = b4;
    }
    out
}

fn count_regions(q: &[f64], types: &[i32]) -> (usize, usize, usize) {
    let nz = types.len();
    let mut cnt = 1_usize;
    let mut min_ktop = 0_usize;
    let mut min_kbot = nz - 1;
    let mut prev_qtop = 0.0;
    let mut min_dq = 1.0;
    let mut kbot = nz - 1;
    let mut cur_type = types[nz - 1];
    for k in (0..(nz - 1)).rev() {
        if cur_type == types[k] {
            continue;
        }
        let qtop = q[k + 1];
        let dq = qtop - prev_qtop;
        if dq < min_dq {
            min_dq = dq;
            min_ktop = k + 1;
            min_kbot = kbot;
        }
        kbot = k;
        cnt += 1;
        cur_type = types[k];
        prev_qtop = qtop;
    }
    let dq = 1.0 - prev_qtop;
    if dq < min_dq {
        min_ktop = 0;
        min_kbot = kbot;
    }
    (cnt, min_ktop, min_kbot)
}

fn remove_region(types: &mut [i32], min_ktop: usize, min_kbot: usize) {
    let nz = types.len();
    let new_type = if min_ktop > 0 {
        types[min_ktop - 1]
    } else if min_kbot + 1 < nz {
        types[min_kbot + 1]
    } else {
        return;
    };
    for v in &mut types[min_ktop..=min_kbot] {
        *v = new_type;
    }
}

fn region_top_0based(types: &[i32], j_1based: usize) -> Option<usize> {
    let nz = types.len();
    if nz == 0 || j_1based == 0 {
        return None;
    }
    let mut cnt = 1_usize;
    let mut cur_type = types[nz - 1];
    for k in (0..(nz - 1)).rev() {
        if cur_type == types[k] {
            continue;
        }
        if cnt == j_1based {
            return Some(k + 1);
        }
        cnt += 1;
        cur_type = types[k];
    }
    (cnt == j_1based).then_some(0)
}

/// Evaluates pruned mixing-region history entries `(mix_type, mix_qtop, mix_relr_top)`
/// for slots `1..=max_regions` (`set_mix_types` + `prune_weak_mixing_regions` in
/// `star/private/history.f90:642-853, 1109-1147`).
#[must_use]
pub fn eval_history_mixing_regions(
    model: &StarModel,
    cz_bdy_dq: &[f64],
    max_regions: usize,
) -> Vec<(i64, f64, f64)> {
    let nz = model.nz;
    if nz == 0 || max_regions == 0 {
        return vec![(-1, 1.0, 1.0); max_regions];
    }
    let mut mx_type: Vec<i32> =
        model.mixing_type[..nz].iter().map(|m| m.code()).collect();
    let mut prev_cnt = usize::MAX;
    for _ in 0..nz {
        let (cnt, min_ktop, min_kbot) = count_regions(&model.q[..nz], &mx_type);
        if cnt <= max_regions || cnt >= prev_cnt {
            break;
        }
        prev_cnt = cnt;
        remove_region(&mut mx_type, min_ktop, min_kbot);
    }

    let d_cutoff = model.controls.output.mixing_d_limit_for_log;
    let d_non_rot = eval_d_mix_non_rotation_vec(model);
    let no_mix = MixingType::NoMixing.code();
    let semi_mix = MixingType::SemiconvectiveMixing.code();
    let thermo_mix = MixingType::ThermohalineMixing.code();
    for _ in 0..1000 {
        let mut min_ktop = 0_usize;
        let mut min_kbot = nz - 1;
        let mut d_max_in_region = 1.0e99;
        let mut kbot = nz - 1;
        let mut cur_type = mx_type[nz - 1];
        for k in (0..(nz - 1)).rev() {
            if cur_type == mx_type[k] {
                continue;
            }
            if cur_type != no_mix
                && cur_type != thermo_mix
                && cur_type != semi_mix
            {
                let d_max = d_non_rot[(k + 1)..=kbot]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if d_max < d_max_in_region {
                    d_max_in_region = d_max;
                    min_ktop = k + 1;
                    min_kbot = kbot;
                }
            }
            kbot = k;
            cur_type = mx_type[k];
        }
        if cur_type == no_mix {
            let d_max = d_non_rot[0..=kbot]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if d_max < d_max_in_region {
                d_max_in_region = d_max;
                min_ktop = 0;
                min_kbot = kbot;
            }
        }
        if d_max_in_region > d_cutoff || mx_type[min_ktop] == no_mix {
            break;
        }
        for v in &mut mx_type[min_ktop..=min_kbot] {
            *v = no_mix;
        }
    }

    let r_surf = model.r[0].max(1.0e-99);
    let mut out = Vec::with_capacity(max_regions);
    for ii in 1..=max_regions {
        match region_top_0based(&mx_type, ii) {
            None => out.push((-1, 1.0, 1.0)),
            Some(0) => out.push((i64::from(mx_type[0]), 1.0, 1.0)),
            Some(k) => {
                let bdy_dq = cz_bdy_dq.get(k - 1).copied().unwrap_or(0.0);
                let qtop = model.q[k - 1] - bdy_dq;
                let frac = bdy_dq / model.dq[k - 1].max(1.0e-99);
                let r3 = (1.0 - frac) * flexmesa_math::pow3(model.r[k - 1])
                    + frac * flexmesa_math::pow3(model.r[k]);
                let relr = flexmesa_math::pow(r3.max(0.0), 1.0 / 3.0) / r_surf;
                out.push((i64::from(mx_type[k]), qtop, relr));
            }
        }
    }
    out
}

/// Evaluates burning-region history entries `(burn_type, burn_qtop, burn_relr_top)`
/// for slots `1..=max_regions` (`set_burn_types`, `interpolate_burn_bdy_q`,
/// `interpolate_burn_bdy_r` in `star/private/history.f90:457, 464, 614-639, 856-905, 1081-1108`).
///
/// Reproduces MESA `26.4.1` verbatim (`#185` under `#169`): `history.f90:457, 464`
/// passes `burning_offset` (`240000`) and `burn_relr_offset` (`250000`) as the
/// second argument `b_regions` to `set_burn_types`, so `cnt <= b_regions` holds
/// on the first check and small burning regions are not merged.
#[must_use]
pub fn eval_history_burning_regions(
    model: &StarModel,
    max_regions: usize,
) -> Vec<(i64, f64, f64)> {
    let nz = model.nz;
    if nz == 0 || max_regions == 0 {
        return vec![(-9999, -1.0, -1.0); max_regions];
    }
    let b_type: Vec<i32> = (0..nz)
        .map(|k| {
            let val = model.eps_nuc[k] - model.non_nuc_neu[k];
            let sgn = if val < 0.0 { -1.0 } else { 1.0 };
            (sgn * (1.0 + safe_log10(val.abs())).max(0.0)) as i32
        })
        .collect();

    let r_surf = model.r[0].max(1.0e-99);
    let mut out = Vec::with_capacity(max_regions);
    for ii in 1..=max_regions {
        let Some(k) = region_top_0based(&b_type, ii) else {
            out.push((-9999, -1.0, -1.0));
            continue;
        };
        let bt = i64::from(b_type[k]);
        if k == 0 {
            out.push((bt, model.q[0], model.r[0] / r_surf));
            continue;
        }
        let eps1 = model.eps_nuc[k] - model.non_nuc_neu[k];
        let eps0 = model.eps_nuc[k - 1] - model.non_nuc_neu[k - 1];
        let bv1 = (if eps1 < 0.0 { -1.0 } else { 1.0 }
            * flexmesa_math::log10(eps1.abs().max(1.0)))
            as i32;
        let bv0 = (if eps0 < 0.0 { -1.0 } else { 1.0 }
            * flexmesa_math::log10(eps0.abs().max(1.0)))
            as i32;
        let eps = flexmesa_math::powi(10.0_f64, bv0.max(bv1));
        let d0 = eps0 - eps;
        let d1 = eps1 - eps;
        if d0 * d1 >= 0.0 {
            out.push((bt, model.q[k], model.r[k] / r_surf));
        } else {
            let q1 = model.q[k] - 0.5 * model.dq[k];
            let q0 = model.q[k - 1] - 0.5 * model.dq[k - 1];
            let bq =
                flexmesa_num::find0(q1, d1, q0, d0).unwrap_or(0.5 * (q0 + q1));
            let br =
                flexmesa_num::find0(model.rmid[k], d1, model.rmid[k - 1], d0)
                    .unwrap_or(0.5 * (model.rmid[k] + model.rmid[k - 1]))
                    / r_surf;
            out.push((bt, bq, br));
        }
    }
    out
}

/// Evaluates asteroseismic summary scalars `(acoustic_radius_s, delta_nu_uhz, nu_max_uhz, delta_pg_s)`
/// matching `report.f90:329-364` and `get_delta_Pg_bildsten2012` (`star_utils.f90:959-1021`).
#[must_use]
pub fn eval_astero_report_scalars(
    model: &StarModel,
    phot: &PhotosphereInfo,
) -> (f64, f64, f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let dr_div_cs: Vec<f64> = (0..nz)
        .map(|k| {
            let dr = if k + 1 == nz {
                model.r[k] - model.r_center
            } else {
                model.r[k] - model.r[k + 1]
            };
            dr / model.csound[k].max(1.0e-99)
        })
        .collect();
    let r_phot_cm = phot.r_rsun * R_SUN;
    let mut acoustic_r = 0.0;
    if r_phot_cm >= model.r[0] {
        acoustic_r = dr_div_cs.iter().sum::<f64>()
            + (r_phot_cm - model.r[0]) / model.csound[0].max(1.0e-99);
    } else {
        for k in 1..nz {
            if r_phot_cm > model.r[k] {
                acoustic_r = dr_div_cs[k..nz].iter().sum::<f64>()
                    + (r_phot_cm - model.r[k])
                        / model.csound[k - 1].max(1.0e-99);
                break;
            }
        }
    }
    let star_mass = model.mstar / M_SUN;
    let o = &model.controls.output;
    let delta_nu = if !o.get_delta_nu_from_scaled_solar {
        if acoustic_r > 0.0 {
            1.0e6 / (2.0 * acoustic_r)
        } else {
            0.0
        }
    } else if phot.l_lsun > 0.0 && o.astero_teff_sun > 0.0 {
        o.delta_nu_sun
            * star_mass.max(0.0).sqrt()
            * flexmesa_math::pow3(model.teff / o.astero_teff_sun)
            / flexmesa_math::pow(phot.l_lsun, 0.75)
    } else {
        0.0
    };
    let nu_max = if phot.r_rsun > 0.0 && o.astero_teff_sun > 0.0 {
        o.nu_max_sun * star_mass
            / (phot.r_rsun
                * phot.r_rsun
                * (model.teff.max(0.0) / o.astero_teff_sun)
                    .sqrt()
                    .max(1.0e-99))
    } else {
        0.0
    };

    let mut delta_pg = 0.0;
    if model.controls.calculate_brunt_n2 && nu_max > 0.0 && nz >= 2 {
        let omega2 = flexmesa_math::pow2(2.0 * PI * nu_max / 1.0e6);
        let l2 = 2.0_f64;
        let mut integral = 0.0;
        let mut i_integral = 0.0;
        for k in 1..nz {
            let n2 = model.brunt_n2[k];
            let r = model.r[k];
            let cs_f = csound_face_at(model, k);
            let cs2 = cs_f * cs_f;
            let sl2 = l2 * cs2 / (r * r).max(1.0e-99);
            let dr = model.rmid[k - 1] - model.rmid[k];
            if omega2 >= sl2 {
                continue;
            }
            if n2 > omega2 {
                integral += n2.sqrt() * dr / r.max(1.0e-99);
            } else if integral > 0.0 {
                let kr2 = (1.0 - n2 / omega2) * (1.0 - sl2 / omega2) * omega2
                    / cs2.max(1.0e-99);
                if kr2 < 0.0 {
                    i_integral += (-kr2).sqrt() * dr;
                }
                if i_integral > 0.5 {
                    break;
                }
            }
        }
        if integral > 0.0 {
            let dp = 2.0_f64.sqrt() * PI * PI / integral;
            if dp.is_finite() {
                delta_pg = dp;
            }
        }
    }
    (acoustic_r, delta_nu, nu_max, delta_pg)
}

/// Cached per-step report state shared across all columns of a `history.data`
/// row or `profileN.data` header (`star/private/report.f90:139-376`).
#[derive(Debug, Clone)]
pub struct StepReportContext {
    /// Core boundary masses and indices (`get_core_info`).
    pub cores: CoreMasses,
    /// Integrated nuclear and neutrino luminosities (`set_power_info`).
    pub powers: PowerCategories,
    /// Integrated category luminosities $L_c / L_\odot$ (`L_by_category`).
    pub cat_lum_lsun: [f64; NUM_CATEGORIES],
    /// Central `dq`-averaged category energy rates in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`center_eps_burn`).
    pub center_eps_burn: [f64; NUM_CATEGORIES],
    /// Photospheric scalars (`get_phot_info` / `set_phot_info`).
    pub phot: PhotosphereInfo,
    /// Surface-averaged rotational quantities (`set_surf_avg_rotation_info`).
    pub surf_rot: SurfAvgRotationInfo,
    /// Convective core and mixing-region summary (`ConvAndMixSummary`).
    pub mix: ConvAndMixSummary,
    /// Up to 3 burning-zone contours `epsnuc_out[0..12]` in grams (`get_burn_zone_info`).
    pub epsnuc_out: [f64; 12],
    /// Global energy components `(ie, pe, ke_rad, ke_rot, et_turb, e_tot)` in erg (`eval_total_energy_integrals`).
    pub energies_erg: (f64, f64, f64, f64, f64, f64),
    /// Total spin angular momentum $\sum_k \overline{\Delta m}_k j_{\mathrm{rot},k}$ in $\mathrm{g\,cm^2\,s^{-1}}$.
    pub total_angular_momentum: f64,
    /// Kelvin-Helmholtz timescale in years (`kh_timescale` in `report.f90:233`).
    pub kh_timescale_yr: f64,
    /// Nuclear burning timescale in years (`nuc_timescale` in `report.f90:245-249`).
    pub nuc_timescale_yr: f64,
    /// Dynamical timescale in seconds (`dynamic_timescale` in `report.f90:251-255`).
    pub dynamic_timescale_s: f64,
    /// Non-rotational Eulerian mixing diffusivity $D_{\mathrm{mix,non\text{-}rot},k}$ (`D_mix_non_rotation`).
    pub d_mix_non_rotation: Vec<f64>,
}

impl StepReportContext {
    /// Computes all per-step report diagnostics from `model`.
    #[must_use]
    pub fn compute(model: &StarModel) -> Self {
        let nz = model.nz;
        let cores = compute_core_masses(model);
        let powers = compute_power_categories(model);

        let cat_lum_lsun = powers.l_by_category_lsun;
        let mut center_eps_burn = [0.0; NUM_CATEGORIES];
        for (c, dst) in center_eps_burn.iter_mut().enumerate() {
            *dst = center_value_by(model, nz, |k| {
                model.aux.eps_nuc_categories.get(k).map_or(0.0, |v| v[c])
            });
        }

        let phot = eval_photosphere_info(model);
        let surf_rot = eval_surf_avg_rotation_info(model);
        let mix = eval_conv_and_mx_summary(model);
        let epsnuc_out = eval_epsnuc_zones(model);
        let sh = &model.aux.step_hist;
        let energies_erg = if sh.total_energy_end != 0.0
            && (sh.total_internal_energy_end != 0.0
                || sh.total_gravitational_energy_end != 0.0)
        {
            (
                sh.total_internal_energy_end,
                sh.total_gravitational_energy_end,
                sh.total_radial_kinetic_energy_end,
                sh.total_rotational_kinetic_energy_end,
                sh.total_turbulent_energy_end,
                sh.total_energy_end,
            )
        } else {
            let mut e = eval_total_energy_integrals(model);
            if sh.total_energy_end != 0.0 {
                e.5 = sh.total_energy_end;
            }
            e
        };
        let total_angular_momentum = if model.controls.rotation_flag {
            model
                .dm_bar
                .iter()
                .take(nz)
                .zip(model.j_rot.iter().take(nz))
                .map(|(&dm, &j)| dm * j)
                .sum()
        } else {
            0.0
        };

        // Timescales (`report.f90:233-255`, `eval_kh_timescale = 0.75 * G * M^2 / (R * L)`
        // in `star/private/star_utils.f90:2890-2897`).
        let r_surf = model.r.first().copied().unwrap_or(0.0);
        let l_surf_erg = model.lum.first().copied().unwrap_or(0.0);
        let cgrav0 = model.cgrav.first().copied().unwrap_or(0.0);
        let kh_timescale_yr = if r_surf > 0.0 && l_surf_erg > 0.0 {
            (0.75 * cgrav0 * model.mstar * model.mstar / (r_surf * l_surf_erg))
                / SECYER
        } else {
            1.0e99
        };
        let star_mass_msun = model.mstar / M_SUN;
        let nuc_timescale_yr = if l_surf_erg > 0.0 {
            1.0e10 * star_mass_msun / (l_surf_erg / L_SUN)
        } else {
            1.0e99
        };
        let dynamic_timescale_s = if cgrav0 <= 0.0 || model.mstar <= 0.0 {
            1.0e99
        } else {
            2.0 * PI
                * (flexmesa_math::pow3(r_surf) / (cgrav0 * model.mstar))
                    .max(0.0)
                    .sqrt()
        };
        let d_mix_non_rotation = eval_d_mix_non_rotation_vec(model);

        Self {
            cores,
            powers,
            cat_lum_lsun,
            center_eps_burn,
            phot,
            surf_rot,
            mix,
            epsnuc_out,
            energies_erg,
            total_angular_momentum,
            kh_timescale_yr,
            nuc_timescale_yr,
            dynamic_timescale_s,
            d_mix_non_rotation,
        }
    }

    /// Central specific entropy in $k_{\mathrm{B}}$ per baryon (`center_entropy`
    /// in `star/private/report.f90:189`).
    #[must_use]
    pub fn center_entropy(&self, model: &StarModel) -> f64 {
        if model.nz == 0 {
            0.0
        } else {
            flexmesa_math::exp(center_value(model, &model.ln_s)) * AMU / KERG
        }
    }
}

/// Shock-front diagnostic scalars matching `get_shock_info` and
/// `get_shock_location_info` (`star/private/star_utils.f90:1328-1485`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShockInfo {
    /// Enclosed baryonic shock mass coordinate in $M_\odot$ (`shock_mass`).
    pub mass_msun: f64,
    /// Relative interior mass coordinate $q$ at the shock (`shock_q`).
    pub q: f64,
    /// Interpolated shock radius in $R_\odot$ (`shock_radius`).
    pub radius_rsun: f64,
    /// Interpolated shock velocity in $\mathrm{cm\,s^{-1}}$ (`shock_velocity`).
    pub velocity_cms: f64,
    /// Pre-shock adiabatic sound speed at `shock_k` in $\mathrm{cm\,s^{-1}}$ (`shock_csound`).
    pub csound_cms: f64,
    /// $\log_{10}(T/\mathrm{K})$ at `shock_k` (`shock_lgT`).
    pub lg_t: f64,
    /// $\log_{10}(\rho/\mathrm{g\,cm^{-3}})$ at `shock_k` (`shock_lgRho`).
    pub lg_rho: f64,
    /// $\log_{10}(P_{\mathrm{eos}}/\mathrm{dyn\,cm^{-2}})$ at `shock_k` (`shock_lgP`).
    pub lg_p: f64,
    /// First adiabatic exponent $\Gamma_1$ at `shock_k` (`shock_gamma1`).
    pub gamma1: f64,
    /// Specific entropy at `shock_k` (`shock_entropy`).
    pub entropy: f64,
    /// Optical depth at `shock_k` (`shock_tau`).
    pub tau: f64,
    /// 1-based shock cell index (`shock_k`, `0` when no shock is detected).
    pub k_1based: usize,
}

/// Evaluates [`ShockInfo`] matching `get_shock_info` and
/// `get_shock_location_info` in `star/private/star_utils.f90:1328-1485`.
#[must_use]
pub fn eval_shock_info(model: &StarModel) -> ShockInfo {
    let nz = model.nz;
    if nz < 3 || !(model.controls.u_flag || model.controls.v_flag) {
        return ShockInfo::default();
    }
    let v = if model.controls.u_flag {
        &model.u
    } else {
        &model.v
    };
    if v.len() < nz || model.csound.len() < nz {
        return ShockInfo::default();
    }

    let mut shock_radius = -1.0_f64;
    let mut k_shock = 0_usize;
    let mut v_div_cs_00 = if model.csound[0] == 0.0 {
        0.0
    } else {
        v[0] / model.csound[0]
    };
    for k in 1..(nz - 1) {
        let v_div_cs_m1 = v_div_cs_00;
        v_div_cs_00 = if model.csound[k] == 0.0 {
            0.0
        } else {
            v[k] / model.csound[k]
        };
        let v_div_cs_max = v_div_cs_00.max(v_div_cs_m1);
        let v_div_cs_min = v_div_cs_00.min(v_div_cs_m1);
        if v_div_cs_max >= 1.0
            && v_div_cs_min < 1.0
            && v[k + 1] > model.csound[k + 1]
        {
            shock_radius = flexmesa_num::find0(
                model.r[k],
                v_div_cs_00 - 1.0,
                model.r[k - 1],
                v_div_cs_m1 - 1.0,
            )
            .unwrap_or_else(|_| f64::midpoint(model.r[k], model.r[k - 1]));
            k_shock = k - 1;
            break;
        }
        if v_div_cs_min <= -1.0
            && v_div_cs_max > -1.0
            && v[k + 1] < -model.csound[k + 1]
        {
            shock_radius = flexmesa_num::find0(
                model.r[k],
                v_div_cs_00 + 1.0,
                model.r[k - 1],
                v_div_cs_m1 + 1.0,
            )
            .unwrap_or_else(|_| f64::midpoint(model.r[k], model.r[k - 1]));
            k_shock = k - 1;
            break;
        }
    }

    if shock_radius <= 0.0
        || shock_radius < model.r_center
        || shock_radius > model.r[0]
    {
        return ShockInfo::default();
    }

    let (mass_msun, q, velocity_cms) = if k_shock + 1 < nz {
        let denom = model.r[k_shock + 1] - model.r[k_shock];
        if denom == 0.0 {
            (
                model.m[k_shock] / M_SUN,
                model.q[k_shock],
                v[k_shock],
            )
        } else {
            let alfa = (shock_radius - model.r[k_shock]) / denom;
            let beta = 1.0 - alfa;
            (
                (alfa * model.m[k_shock + 1] + beta * model.m[k_shock]) / M_SUN,
                alfa * model.q[k_shock + 1] + beta * model.q[k_shock],
                alfa * v[k_shock + 1] + beta * v[k_shock],
            )
        }
    } else {
        (model.m[k_shock] / M_SUN, model.q[k_shock], v[k_shock])
    };

    ShockInfo {
        mass_msun,
        q,
        radius_rsun: shock_radius / R_SUN,
        velocity_cms,
        csound_cms: model.csound[k_shock],
        lg_t: model.ln_t[k_shock] / LN10,
        lg_rho: model.lnd[k_shock] / LN10,
        lg_p: model.ln_peos[k_shock] / LN10,
        gamma1: model.gamma1[k_shock],
        entropy: model.entropy[k_shock],
        tau: model.tau[k_shock],
        k_1based: k_shock + 1,
    }
}

/// Tabulated `omega / omega_Kep` grid (`star/private/gravity_darkening.f90:43-149`,
/// `data/star_data/gravity_darkening_coefficients.data`, SHA-256
/// `6c051e3f1eec1c871680a9df53b16ab7b5c40f1158432ae26b89bada255b08d5`).
const GD_OMEGA_GRID: [f64; 21] = [
    0.00000000, 0.05000000, 0.10000000, 0.15000000, 0.20000000,
    0.25000000, 0.30000000, 0.35000000, 0.40000000, 0.45000000,
    0.50000000, 0.55000000, 0.60000000, 0.65000000, 0.70000000,
    0.75000000, 0.80000000, 0.85000000, 0.90000000, 0.95000000,
    1.00000000,
];

/// Tabulated inclination grid in radians (`star/private/gravity_darkening.f90:43-149`,
/// `data/star_data/gravity_darkening_coefficients.data`, SHA-256
/// `6c051e3f1eec1c871680a9df53b16ab7b5c40f1158432ae26b89bada255b08d5`).
const GD_INCLINATION_GRID: [f64; 21] = [
    0.00000000, 0.07853982, 0.15707963, 0.23561945, 0.31415927,
    0.39269908, 0.47123890, 0.54977871, 0.62831853, 0.70685835,
    0.78539816, 0.86393798, 0.94247780, 1.02101761, 1.09955743,
    1.17809725, 1.25663706, 1.33517688, 1.41371669, 1.49225651,
    1.57079633,
];

/// Row-major `C_T(omega, inclination)` table (`21 x 21`,
/// `star/private/gravity_darkening.f90:43-149`,
/// `data/star_data/gravity_darkening_coefficients.data`, SHA-256
/// `6c051e3f1eec1c871680a9df53b16ab7b5c40f1158432ae26b89bada255b08d5`).
const GD_C_T_VALUES: [f64; 441] = [
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00020914, 1.00020722, 1.00020150, 1.00019213,
    1.00017933, 1.00016341, 1.00014477, 1.00012386, 1.00010119,
    1.00007733, 1.00005284, 1.00002834, 1.00000443, 0.99998170,
    0.99996070, 0.99994196, 0.99992593, 0.99991302, 0.99990354,
    0.99989772, 0.99989571, 1.00083385, 1.00082623, 1.00080356,
    1.00076637, 1.00071556, 1.00065234, 1.00057823, 1.00049501,
    1.00040470, 1.00030950, 1.00021172, 1.00011376, 1.00001804,
    0.99992692, 0.99984268, 0.99976741, 0.99970299, 0.99965105,
    0.99961289, 0.99958948, 0.99958139, 1.00186576, 1.00184887,
    1.00179859, 1.00171607, 1.00160321, 1.00146262, 1.00129757,
    1.00111194, 1.00091012, 1.00069695, 1.00047757, 1.00025736,
    1.00004174, 0.99983611, 0.99964564, 0.99947519, 0.99932911,
    0.99921119, 0.99912448, 0.99907124, 0.99905284, 1.00329018,
    1.00326077, 1.00317320, 1.00302937, 1.00283238, 1.00258658,
    1.00229744, 1.00197148, 1.00161620, 1.00123992, 1.00085162,
    1.00046074, 1.00007696, 0.99970999, 0.99936925, 0.99906364,
    0.99880124, 0.99858907, 0.99843287, 0.99833687, 0.99830371,
    1.00508493, 1.00504023, 1.00490706, 1.00468806, 1.00438764,
    1.00401195, 1.00356884, 1.00306784, 1.00252000, 1.00193776,
    1.00133478, 1.00072563, 1.00012548, 0.99954969, 0.99901342,
    0.99853108, 0.99811595, 0.99777962, 0.99753164, 0.99737910,
    0.99732638, 1.00721889, 1.00715672, 1.00697138, 1.00666613,
    1.00624652, 1.00572036, 1.00509779, 1.00439130, 1.00361568,
    1.00278790, 1.00192694, 1.00105342, 1.00018920, 0.99935677,
    0.99857861, 0.99787643, 0.99727037, 0.99677822, 0.99641471,
    0.99619084, 0.99611344, 1.00965018, 1.00956909, 1.00932709,
    1.00892786, 1.00837767, 1.00768557, 1.00686351, 1.00592658,
    1.00489311, 1.00378468, 1.00262599, 1.00144449, 1.00026990,
    0.99913334, 0.99806643, 0.99710011, 0.99626339, 0.99558218,
    0.99507802, 0.99476713, 0.99465960, 1.01232416, 1.01222350,
    1.01192282, 1.01142579, 1.01073882, 1.00987139, 1.00883646,
    1.00765095, 1.00633613, 1.00491786, 1.00342668, 1.00189748,
    1.00036888, 0.99888222, 0.99748018, 0.99620513, 0.99509725,
    0.99419273, 0.99352188, 0.99310764, 0.99296428, 1.01517127,
    1.01505135, 1.01469273, 1.01409853, 1.01327449, 1.01222945,
    1.01097618, 1.00953220, 1.00792068, 1.00617105, 1.00431942,
    1.00240844, 1.00048664, 0.99860713, 0.99682571, 0.99519856,
    0.99377956, 0.99261759, 0.99175390, 0.99121980, 0.99103487,
    1.01810491, 1.01796706, 1.01755423, 1.01686844, 1.01591369,
    1.01469687, 1.01322896, 1.01152646, 1.00961286, 1.00751998,
    1.00528886, 1.00296997, 1.00062255, 0.99831302, 0.99611243,
    0.99409325, 0.99232572, 0.99087401, 0.98979254, 0.98912278,
    0.98889074, 1.02101938, 1.02086599, 1.02040596, 1.01963947,
    1.01856768, 1.01719397, 1.01552564, 1.01357605, 1.01136696,
    1.00893086, 1.00631263, 1.00357032, 1.00077455, 0.99800649,
    0.99535454, 0.99290997, 0.99076197, 0.98899251, 0.98767143,
    0.98685210, 0.98656805, 1.02378793, 1.02362253, 1.02312567,
    1.02229502, 1.02112776, 1.01962207, 1.01777938, 1.01560737,
    1.01312352, 1.01035868, 1.00736004, 1.00419266, 1.00093904,
    0.99769648, 0.99457268, 0.99167997, 0.98912882, 0.98702124,
    0.98544440, 0.98446508, 0.98412530, 1.02626085, 1.02608811,
    1.02556825, 1.02469586, 1.02346308, 1.02186122, 1.01988350,
    1.01752898, 1.01480765, 1.01174583, 1.00839100, 1.00481448,
    1.00111100, 0.99739525, 0.99379588, 0.99044804, 0.98748519,
    0.98503092, 0.98319107, 0.98204689, 0.98164957, 1.02826333,
    1.02808905, 1.02756346, 1.02667776, 1.02541824, 1.02376787,
    1.02170926, 1.01922956, 1.01632734, 1.01302079, 1.00935523,
    1.00540732, 1.00128450, 0.99711989, 0.99306419, 0.98927622,
    0.98591309, 0.98312041, 0.98102309, 0.97971715, 0.97926323,
    1.02959218, 1.02942323, 1.02891260, 1.02804807, 1.02680983,
    1.02517150, 1.02310289, 1.02057549, 1.01757173, 1.01409705,
    1.01019169, 1.00593703, 1.00145389, 0.99689462, 0.99243214,
    0.98824841, 0.98452316, 0.98142298, 0.97909093, 0.97763713,
    0.97713122, 1.03000953, 1.02985383, 1.02938212, 1.02857937,
    1.02742017, 1.02586874, 1.02388039, 1.02140697, 1.01840850,
    1.01487160, 1.01082821, 1.00636452, 1.00161603, 0.99675457,
    0.99197385, 0.98747642, 0.98346158, 0.98011388, 0.97759192,
    0.97601796, 0.97546943, 1.02922841, 1.02909487, 1.02868929,
    1.02799514, 1.02698334, 1.02561018, 1.02381625, 1.02152940,
    1.01867846, 1.01522206, 1.01118064, 1.00664736, 1.00177443,
    0.99675209, 0.99179115, 0.98710952, 0.98292055, 0.97942145,
    0.97678182, 0.97513256, 0.97455669, 1.02687509, 1.02677360,
    1.02646470, 1.02593281, 1.02514898, 1.02406631, 1.02261370,
    1.02069126, 1.01818140, 1.01499834, 1.01114961, 1.00674212,
    1.00194702, 0.99696925, 0.99202969, 0.98735347, 0.98315961,
    0.97965024, 0.97699917, 0.97534069, 0.97476005, 1.02236518,
    1.02230694, 1.02212957, 1.02182227, 1.02136343, 1.02071413,
    1.01980521, 1.01851101, 1.01662985, 1.01399216, 1.01060538,
    1.00660609, 1.00218383, 0.99754962, 0.99292340, 0.98852593,
    0.98457052, 0.98125345, 0.97874334, 0.97717045, 0.97661749,
    1.01397932, 1.01397930, 1.01397929, 1.01397927, 1.01397926,
    1.01397924, 1.01397923, 1.01394991, 1.01327918, 1.01166668,
    1.00923544, 1.00616622, 1.00265406, 0.99889900, 0.99510226,
    0.99146221, 0.98816850, 0.98539452, 0.98328850, 0.98196465,
    0.98149538,
];

/// Row-major `C_L(omega, inclination)` table (`21 x 21`,
/// `star/private/gravity_darkening.f90:43-149`,
/// `data/star_data/gravity_darkening_coefficients.data`, SHA-256
/// `6c051e3f1eec1c871680a9df53b16ab7b5c40f1158432ae26b89bada255b08d5`).
const GD_C_L_VALUES: [f64; 441] = [
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
    1.00000000, 1.00166901, 1.00165436, 1.00160856, 1.00153348,
    1.00143099, 1.00130355, 1.00115439, 1.00098708, 1.00080581,
    1.00061500, 1.00041938, 1.00022369, 1.00003283, 0.99985145,
    0.99968403, 0.99953465, 0.99940703, 0.99930427, 0.99922891,
    0.99918280, 0.99916725, 1.00668483, 1.00662393, 1.00644061,
    1.00614003, 1.00572963, 1.00521952, 1.00462220, 1.00395240,
    1.00322658, 1.00246258, 1.00167924, 1.00089578, 1.00013152,
    0.99940521, 0.99873478, 0.99813667, 0.99762562, 0.99721419,
    0.99691249, 0.99672792, 0.99666519, 1.01505267, 1.01491465,
    1.01450191, 1.01382528, 1.01290138, 1.01175292, 1.01040812,
    1.00890003, 1.00726575, 1.00554551, 1.00378158, 1.00201741,
    1.00029636, 0.99866085, 0.99715108, 0.99580421, 0.99465338,
    0.99372688, 0.99304749, 0.99263187, 0.99249041, 1.02678098,
    1.02653499, 1.02580086, 1.02459732, 1.02295391, 1.02091095,
    1.01851860, 1.01583563, 1.01292802, 1.00986729, 1.00672873,
    1.00358969, 1.00052738, 0.99761724, 0.99493093, 0.99253453,
    0.99048698, 0.98883862, 0.98762992, 0.98689054, 0.98663875,
    1.04187405, 1.04148914, 1.04034172, 1.03846048, 1.03589151,
    1.03269765, 1.02895727, 1.02476215, 1.02021544, 1.01542903,
    1.01052087, 1.00561190, 1.00082314, 0.99627263, 0.99207241,
    0.98832584, 0.98512500, 0.98254841, 0.98065924, 0.97950365,
    0.97911004, 1.06032426, 1.05976972, 1.05811740, 1.05540823,
    1.05170819, 1.04710758, 1.04171896, 1.03567454, 1.02912294,
    1.02222562, 1.01515283, 1.00807925, 1.00117963, 0.99462424,
    0.98857460, 0.98317946, 0.97857112, 0.97486226, 0.97214331,
    0.97048035, 0.96991386, 1.08210183, 1.08134720, 1.07909952,
    1.07541370, 1.07037893, 1.06411757, 1.05678237, 1.04855314,
    1.03963238, 1.03024051, 1.02061006, 1.01097982, 1.00158842,
    0.99266812, 0.98443885, 0.97710258, 0.97083847, 0.96579871,
    0.96210509, 0.95984645, 0.95907702, 1.10714220, 1.10615786,
    1.10322624, 1.09841806, 1.09184868, 1.08367665, 1.07410063,
    1.06335511, 1.05170506, 1.03943943, 1.02686356, 1.01429098,
    1.00203474, 0.99039895, 0.97967050, 0.97011190, 0.96195500,
    0.95539584, 0.95059073, 0.94765329, 0.94665267, 1.13533093,
    1.13408832, 1.13038756, 1.12431653, 1.11601907, 1.10569366,
    1.09359010, 1.08000450, 1.06527290, 1.04976292, 1.03386371,
    1.01797498, 1.00249516, 0.98780980, 0.97428101, 0.96223807,
    0.95196994, 0.94371945, 0.93767907, 0.93398808, 0.93273090,
    1.16648638, 1.16495873, 1.16040883, 1.15294242, 1.14273354,
    1.13002346, 1.11511768, 1.09838063, 1.08022823, 1.06111747,
    1.04153341, 1.02197418, 1.00293492, 0.98489230, 0.96829076,
    0.95353099, 0.94096147, 0.93087248, 0.92349240, 0.91898538,
    0.91745054, 1.20034074, 1.19850383, 1.19303220, 1.18404950,
    1.17176059, 1.15645136, 1.13848663, 1.11830531, 1.09641230,
    1.07336580, 1.04976015, 1.02620538, 1.00330541, 0.98163707,
    0.96173254, 0.94406610, 0.92904520, 0.91700536, 0.90820792,
    0.90283935, 0.90101155, 1.23651993, 1.23435305, 1.22789727,
    1.21729337, 1.20277635, 1.18467693, 1.16342150, 1.13952907,
    1.11360301, 1.08631597, 1.05838716, 1.03055402, 1.00354157,
    0.97803466, 0.95465604, 0.93395179, 0.91638379, 0.90232692,
    0.89206969, 0.88581600, 0.88368746, 1.27452333, 1.27201044,
    1.26452150, 1.25221273, 1.23534681, 1.21429739, 1.18955289,
    1.16171687, 1.13150148, 1.09970957, 1.06720415, 1.03486784,
    1.00355968, 0.97407776, 0.94713348, 0.92333775, 0.90319707,
    0.88711561, 0.87540002, 0.86826482, 0.86583705, 1.31370368,
    1.31083451, 1.30228065, 1.28821060, 1.26891011, 1.24479116,
    1.21640180, 1.18443329, 1.14971780, 1.11320837, 1.07593622,
    1.03895054, 1.00325621, 0.96976484, 0.93926727, 0.91242480,
    0.88977286, 0.87173034, 0.85861029, 0.85062926, 0.84791462,
    1.35324650, 1.35001810, 1.34038896, 1.32453523, 1.30275856,
    1.27550091, 1.24336344, 1.20712577, 1.16775367, 1.12637782,
    1.08423111, 1.04255510, 1.00250844, 0.96510683, 0.93120026,
    0.90147681, 0.87647876, 0.85662162, 0.84221102, 0.83345618,
    0.83047929, 1.39214431, 1.38856261, 1.37787396, 1.36025593,
    1.33601586, 1.30561198, 1.26968578, 1.22910180, 1.18497717,
    1.13866233, 1.09164136, 1.04537724, 1.00117899, 0.96013904,
    0.92313101, 0.89083631, 0.86377693, 0.84234475, 0.82682384,
    0.81740685, 0.81420568, 1.42914908, 1.42523074, 1.41353023,
    1.39421881, 1.36759507, 1.33411264, 1.29442989, 1.24948519,
    1.20057364, 1.14933962, 1.09759577, 1.04704988, 0.99912794,
    0.95494298, 0.91533906, 0.88094986, 0.85224823, 0.82958280,
    0.81320348, 0.80327851, 0.79990523, 1.46264409, 1.45841937,
    1.44579500, 1.42492643, 1.39608445, 1.35968639, 1.31636486,
    1.26710235, 1.21342180, 1.15741691, 1.10134722, 1.04714036,
    0.99624102, 0.94968964, 0.90823529, 0.87242241, 0.84264926,
    0.81920590, 0.80229882, 0.79206643, 0.78858875, 1.49016476,
    1.48568343, 1.47228141, 1.45008692, 1.41931816, 1.38030909,
    1.33358385, 1.28008026, 1.22168197, 1.16134388, 1.10185570,
    1.04515005, 0.99249604, 0.94474900, 0.90250435, 0.86618730,
    0.83610589, 0.81248356, 0.79547899, 0.78519857, 0.78170389,
    1.50404837, 1.49941196, 1.48553117, 1.46249159, 1.43043526,
    1.38955982, 1.34011728, 1.28249582, 1.22006730, 1.15743945,
    1.09705843, 1.04036511, 0.98827414, 0.94139579, 0.90015278,
    0.86484606, 0.83569343, 0.81285286, 0.79643664, 0.78652019,
    0.78314733,
];

struct GravityDarkeningSplines {
    c_t: Vec<[f64; 4]>,
    c_l: Vec<[f64; 4]>,
    ilinx: bool,
    iliny: bool,
}

static GD_SPLINES: OnceLock<GravityDarkeningSplines> = OnceLock::new();

fn gravity_darkening_splines() -> &'static GravityDarkeningSplines {
    GD_SPLINES.get_or_init(|| {
        let (c_t, ilinx, iliny) = interp_mkbicub_db(
            &GD_OMEGA_GRID,
            &GD_INCLINATION_GRID,
            &GD_C_T_VALUES,
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
        )
        .expect("gravity darkening C_T bicubic spline construction must succeed");
        let (c_l, _, _) = interp_mkbicub_db(
            &GD_OMEGA_GRID,
            &GD_INCLINATION_GRID,
            &GD_C_L_VALUES,
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
        )
        .expect("gravity darkening C_L bicubic spline construction must succeed");
        GravityDarkeningSplines {
            c_t,
            c_l,
            ilinx,
            iliny,
        }
    })
}

fn eval_gd_coeff(omega: f64, inclination: f64, coeffs: &[[f64; 4]]) -> f64 {
    if !omega.is_finite() || !inclination.is_finite() {
        return 1.0;
    }
    let splines = gravity_darkening_splines();
    let safe_omega = omega.clamp(0.0, 1.0);
    let safe_incl = inclination.clamp(0.0, 0.5 * PI);
    interp_evbicub_prevalidated_db(
        safe_omega,
        safe_incl,
        &GD_OMEGA_GRID,
        &GD_INCLINATION_GRID,
        splines.ilinx,
        splines.iliny,
        coeffs,
    )
    .map_or(1.0, |ev| ev.f)
}

/// Returns the embedded upstream `gravity_darkening_coefficients.data` grids and
/// `(C_T, C_L)` tables (`star/private/gravity_darkening.f90:43-149`).
#[must_use]
pub const fn gravity_darkening_table() -> (
    &'static [f64; 21],
    &'static [f64; 21],
    &'static [f64; 441],
    &'static [f64; 441],
) {
    (
        &GD_OMEGA_GRID,
        &GD_INCLINATION_GRID,
        &GD_C_T_VALUES,
        &GD_C_L_VALUES,
    )
}

/// Evaluates the gravity-darkening effective temperature scaling coefficient
/// `gravity_darkening_Teff_coeff(omega, inclination)`
/// (`star/private/gravity_darkening.f90:136-140`).
#[must_use]
pub fn gravity_darkening_teff_coeff(omega: f64, inclination: f64) -> f64 {
    eval_gd_coeff(omega, inclination, &gravity_darkening_splines().c_t)
}

/// Evaluates the gravity-darkening luminosity scaling coefficient
/// `gravity_darkening_L_coeff(omega, inclination)`
/// (`star/private/gravity_darkening.f90:143-147`).
#[must_use]
pub fn gravity_darkening_l_coeff(omega: f64, inclination: f64) -> f64 {
    eval_gd_coeff(omega, inclination, &gravity_darkening_splines().c_l)
}

/// Evaluates surface Keplerian rotation ratio
/// $\Omega_0 \sqrt{r_{\mathrm{eq},0}^3 / (G_0 M_0)}$ used by gravity-darkening
/// history columns (`star/private/history.f90:2903`).
#[must_use]
pub fn eval_surface_w_div_w_kep(model: &StarModel) -> f64 {
    if !model.controls.rotation_flag
        || model.nz == 0
        || model.cgrav[0] <= 0.0
        || model.m[0] <= 0.0
    {
        return 0.0;
    }
    let (r_eq, _) = equatorial_and_polar_r_cm(model, 0);
    let r_eq3 = flexmesa_math::pow3(r_eq);
    model.omega[0] * (r_eq3 / (model.cgrav[0] * model.m[0])).max(0.0).sqrt()
}
