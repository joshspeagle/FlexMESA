// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream MESA `26.4.1` radial `LOGS/profileN.data` and `LOGS/profiles.index`
//! column parser, evaluator, priority-evicting index manager, and stage log
//! coordinator (`star/private/profile.f90`, `star/private/profile_getval.f90`,
//! and `star/defaults/profile_columns.list`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_chem::{
    category_id, category_name, NuclideCatalog, NUM_CATEGORIES,
};
use flexmesa_const::{
    MixingType, BOLTZ_SIGMA, FOUR_PI, LN10, L_SUN, M_SUN, R_SUN, SECYER,
};
use flexmesa_eos::{
    I_FRAC_CMS, I_FRAC_FREE_EOS, I_FRAC_HELM, I_FRAC_IDEAL, I_FRAC_OPAL_SCVH,
    I_FRAC_PC, I_FRAC_SKYE,
};
use flexmesa_schema::strip_fortran_comment;

use crate::controls::StarError;
use crate::eps_mdot::eval_cell_specific_total_energy;
use crate::mesa_history::{
    validate_extra_header_items, IncrementalHistoryWriter, MesaCellValue,
};
use crate::model::StarModel;
use crate::report_utils::{
    center_avg_x_by_name, center_value, csound_face_at,
    equatorial_and_polar_r_cm, eval_upstream_omega_crit, format_mesa_dbl,
    format_mesa_header_str, format_mesa_int, get_ledd, get_lrad_div_ledd,
    iso_slot_for_chem_id, parse_quoted_include_arg, push_mesa_6line_header,
    resolve_iso_slot, safe_log10, write_mesa_dbl_width, write_mesa_int_width,
    StepReportContext, MESA_REF_VERSION_NUMBER, STAR_LOG_COL_WIDTH,
};

/// Active column declarations from upstream MESA `26.4.1`
/// `star/defaults/profile_columns.list` (`12` columns).
pub const DEFAULT_STAR_PROFILE_COLUMNS_TEXT: &str = "\
zone
mass
logR
logT
logRho
logP
x_mass_fraction_H
y_mass_fraction_He
z_mass_fraction_metals
pp
cno
tri_alpha
";

/// Resolved column specification from `profile_columns.list`
/// (`star/private/profile.f90:43-258` and `star/private/profile_getval.f90:39-216`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileColumnSpec {
    /// Built-in scalar profile column with canonical output header name.
    Scalar(&'static str),
    /// `add_abundances` directive expanding to all active `model.chem_id` nuclides (`profile.f90:160-165`).
    AddAbundances,
    /// `add_log_abundances` directive expanding to `log_<iso>` for all active nuclides (`profile.f90:167-172`).
    AddLogAbundances,
    /// Specific nuclide mass fraction `<iso>` (`abundance_offset`, `profile_getval.f90:161-165, 378-381`).
    Nuclide(usize, String),
    /// Base-10 logarithm of specific nuclide mass fraction `log <iso>` (`log_abundance_offset`, `profile_getval.f90:131-132, 370-377`).
    LogNuclide(usize, String),
    /// Linear nuclear reaction category energy rate in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`category_offset`, `profile_getval.f90:166-170, 382-385`).
    CategoryEps(usize),
}

const MAX_INCLUDE_LEVEL: usize = 20;

const SUPPORTED_PROFILE_SCALARS: &[&str] = &[
    "zone",
    "k",
    "mass",
    "log_mass",
    "logM",
    "mass_grams",
    "mmid",
    "dm",
    "dm_bar",
    "m_grav",
    "m_grav_div_m_baryonic",
    "mass_correction_factor",
    "q",
    "log_q",
    "dq",
    "log_dq",
    "logdq",
    "dq_ratio",
    "xm",
    "logxm",
    "xq",
    "logxq",
    "radius",
    "radius_cm",
    "radius_km",
    "rmid",
    "logR_cm",
    "logR",
    "r_div_R",
    "dr",
    "log_dr",
    "dlogR",
    "xr",
    "xr_cm",
    "xr_div_R",
    "temperature",
    "logT",
    "logT_face",
    "logT_bb",
    "density",
    "rho",
    "logRho",
    "pgas",
    "logPgas",
    "prad",
    "pressure",
    "logP",
    "pgas_div_ptotal",
    "pgas_div_p",
    "prad_div_pgas",
    "energy",
    "logE",
    "entropy",
    "logS",
    "grada",
    "cv",
    "cp",
    "chiRho",
    "chiT",
    "gamma1",
    "gamma3",
    "eta",
    "gam",
    "mu",
    "abar",
    "zbar",
    "z2bar",
    "ye",
    "free_e",
    "logfree_e",
    "csound",
    "csound_face",
    "cs_at_cell_bdy",
    "log_csound",
    "dr_div_cs",
    "log_dr_div_cs",
    "dt_cs_div_dr",
    "log_dt_cs_div_dr",
    "x",
    "log_x",
    "x_mass_fraction_H",
    "y",
    "log_y",
    "y_mass_fraction_He",
    "z",
    "log_z",
    "z_mass_fraction_metals",
    "luminosity",
    "logL",
    "lum_erg_s",
    "log_abs_lum_erg_s",
    "eps_nuc",
    "log_abs_eps_nuc",
    "eps_nuc_neu_total",
    "non_nuc_neu",
    "nonnucneu_plas",
    "nonnucneu_brem",
    "nonnucneu_phot",
    "nonnucneu_pair",
    "nonnucneu_reco",
    "eps_grav",
    "signed_log_eps_grav",
    "eps_mdot",
    "eps_grav_plus_eps_mdot",
    "extra_heat",
    "net_nuclear_energy",
    "net_energy",
    "signed_log_power",
    "opacity",
    "log_opacity",
    "log_kap",
    "kap_frac_lowT",
    "kap_frac_highT",
    "kap_frac_Type2",
    "kap_frac_Compton",
    "eos_frac_OPAL_SCVH",
    "eos_frac_HELM",
    "eos_frac_Skye",
    "eos_frac_PC",
    "eos_frac_FreeEOS",
    "eos_frac_CMS",
    "eos_frac_ideal",
    "tau",
    "logtau",
    "xtau",
    "xlogtau",
    "logtau_sub_xlogtau",
    "gradT",
    "gradr",
    "gradL",
    "diff_grads",
    "log_diff_grads",
    "gradT_sub_grada",
    "grada_sub_gradT",
    "gradr_sub_grada",
    "gradT_div_grada",
    "scale_height",
    "pressure_scale_height",
    "pressure_scale_height_cm",
    "mlt_mixing_length",
    "mlt_mixing_type",
    "mixing_type",
    "mix_type",
    "conv_vel",
    "log_conv_vel",
    "conv_vel_div_csound",
    "log_D_mix",
    "log_mlt_D_mix",
    "log_mlt_Gamma",
    "log_sig_mix",
    "log_D_conv",
    "log_D_semi",
    "log_D_ovr",
    "log_D_thrm",
    "log_D_minimum",
    "brunt_B",
    "brunt_N2",
    "brunt_N2_composition_term",
    "brunt_N2_structure_term",
    "brunt_N",
    "sign_brunt_N2",
    "lamb_S",
    "lamb_S2",
    "grav",
    "log_g",
    "v",
    "velocity",
    "v_kms",
    "vel_km_per_s",
    "u",
    "u_face",
    "v_div_csound",
    "v_div_cs",
    "v_div_r",
    "v_div_vesc",
    "v_div_v_escape",
    "omega",
    "log_omega",
    "omega_crit",
    "omega_div_omega_crit",
    "j_rot",
    "log_j_rot",
    "i_rot",
    "v_rot",
    "fp_rot",
    "ft_rot",
    "r_equatorial",
    "r_polar",
    "r_e_div_r_p",
    "log_r_equatorial",
    "log_r_polar",
    "alpha_RTI",
    "log_alpha_RTI",
    "eta_RTI",
    "log_eta_RTI",
    "log_sig_RTI",
    "dPdr_dRhodr_info",
    "log_source_plus_alpha_RTI",
    "log_source_minus_alpha_RTI",
    "log_L_div_Ledd",
    "log_Lrad_div_Ledd",
    "logQ",
    "logS_per_baryon",
    "total_energy",
    "acoustic_depth",
    "log_acoustic_depth",
    "acoustic_radius",
    "log_acoustic_radius",
    "eps_nuc_plus_nuc_neu",
    "cgrav_factor",
];

fn match_profile_scalar(tok: &str) -> Option<&'static str> {
    SUPPORTED_PROFILE_SCALARS
        .iter()
        .copied()
        .find(|c| c.eq_ignore_ascii_case(tok))
}

fn push_unique_prof(specs: &mut Vec<ProfileColumnSpec>, sp: ProfileColumnSpec) {
    if !specs.contains(&sp) {
        specs.push(sp);
    }
}

/// Parses `profile_columns.list` text recursively into deduplicated
/// [`ProfileColumnSpec`] entries (`add_profile_columns` in `star/private/profile.f90:43-258`).
pub fn parse_star_profile_columns(
    text: &str,
    base_dir: Option<&Path>,
) -> Result<Vec<ProfileColumnSpec>, StarError> {
    let mut specs = Vec::new();
    parse_profile_columns_inner(text, base_dir, 1, &mut specs)?;
    if specs.is_empty() {
        return Err(StarError::InvalidInput(String::from(
            "profile_columns.list produced zero active columns",
        )));
    }
    Ok(specs)
}

fn parse_profile_columns_inner(
    text: &str,
    base_dir: Option<&Path>,
    level: usize,
    specs: &mut Vec<ProfileColumnSpec>,
) -> Result<(), StarError> {
    if level > MAX_INCLUDE_LEVEL {
        return Err(StarError::InvalidInput(format!(
            "profile_columns.list exceeded max include depth ({MAX_INCLUDE_LEVEL})"
        )));
    }
    let cat = NuclideCatalog::builtin_ref();
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let (code, _) = strip_fortran_comment(raw_line);
        if code.is_empty() {
            continue;
        }
        if let Some(rest) = code.strip_prefix("include") {
            if rest.is_empty()
                || rest.starts_with(char::is_whitespace)
                || rest.starts_with(['\'', '"'])
            {
                let Some(inc_target) = parse_quoted_include_arg(rest) else {
                    return Err(StarError::InvalidInput(format!(
                        "profile_columns.list line {line_no}: malformed include directive {code:?}"
                    )));
                };
                if inc_target.trim().is_empty() {
                    parse_profile_columns_inner(
                        DEFAULT_STAR_PROFILE_COLUMNS_TEXT,
                        base_dir,
                        level + 1,
                        specs,
                    )?;
                } else {
                    let inc_path = match base_dir {
                        Some(d) => d.join(inc_target.trim()),
                        None => PathBuf::from(inc_target.trim()),
                    };
                    let inc_text =
                        fs::read_to_string(&inc_path).map_err(|e| {
                            StarError::IoError(format!(
                            "failed to read included profile columns {}: {e}",
                            inc_path.display()
                        ))
                        })?;
                    parse_profile_columns_inner(
                        &inc_text,
                        inc_path.parent().or(base_dir),
                        level + 1,
                        specs,
                    )?;
                }
                continue;
            }
        }

        let parts: Vec<&str> = code.split_whitespace().collect();
        if parts.len() == 1 {
            let tok = parts[0];
            let low = tok.to_ascii_lowercase();
            match low.as_str() {
                "add_abundances" => {
                    push_unique_prof(specs, ProfileColumnSpec::AddAbundances);
                    continue;
                }
                "add_log_abundances" => {
                    push_unique_prof(
                        specs,
                        ProfileColumnSpec::AddLogAbundances,
                    );
                    continue;
                }
                "add_reaction_categories" => {
                    for cid in 1..=NUM_CATEGORIES {
                        push_unique_prof(
                            specs,
                            ProfileColumnSpec::CategoryEps(cid),
                        );
                    }
                    continue;
                }
                _ => {}
            }
            if let Some(canon) = match_profile_scalar(tok) {
                push_unique_prof(specs, ProfileColumnSpec::Scalar(canon));
                continue;
            }
            if let Some(cid) = cat.get_nuclide_index(&low) {
                let canon = cat
                    .get_by_id(cid)
                    .map_or_else(|| low.clone(), |n| n.name.clone());
                push_unique_prof(specs, ProfileColumnSpec::Nuclide(cid, canon));
                continue;
            }
            let rc = category_id(&low);
            if rc > 0 {
                push_unique_prof(specs, ProfileColumnSpec::CategoryEps(rc));
                continue;
            }
            return Err(StarError::InvalidInput(format!(
                "profile_columns.list line {line_no}: unsupported or unknown profile column {tok:?}"
            )));
        }

        if parts.len() == 2 && parts[0].eq_ignore_ascii_case("log") {
            let iso_str = parts[1].trim();
            let Some(cid) = cat.get_nuclide_index(iso_str) else {
                return Err(StarError::InvalidInput(format!(
                    "profile_columns.list line {line_no}: unknown nuclide {iso_str:?} after 'log'"
                )));
            };
            let canon = cat.get_by_id(cid).map_or_else(
                || iso_str.to_ascii_lowercase(),
                |n| n.name.clone(),
            );
            push_unique_prof(specs, ProfileColumnSpec::LogNuclide(cid, canon));
            continue;
        }

        return Err(StarError::InvalidInput(format!(
            "profile_columns.list line {line_no}: unsupported profile directive {code:?}"
        )));
    }
    Ok(())
}

/// Resolves `profile_columns_file` for `model` relative to `case_dir`
/// (`set_profile_columns` in `star/private/profile.f90:261-294`).
pub fn resolve_star_profile_columns(
    model: &StarModel,
    case_dir: Option<&Path>,
) -> Result<Vec<ProfileColumnSpec>, StarError> {
    let requested = model.job.profile_columns_file.trim();
    if !requested.is_empty() {
        let p = match case_dir {
            Some(d) => d.join(requested),
            None => PathBuf::from(requested),
        };
        let text = fs::read_to_string(&p).map_err(|e| {
            StarError::IoError(format!(
                "failed to open profile_columns_file {}: {e}",
                p.display()
            ))
        })?;
        return parse_star_profile_columns(&text, p.parent().or(case_dir));
    }
    if let Some(d) = case_dir {
        let local = d.join("profile_columns.list");
        if local.is_file() {
            let text = fs::read_to_string(&local).map_err(|e| {
                StarError::IoError(format!(
                    "failed to open {}: {e}",
                    local.display()
                ))
            })?;
            return parse_star_profile_columns(&text, Some(d));
        }
    }
    parse_star_profile_columns(DEFAULT_STAR_PROFILE_COLUMNS_TEXT, case_dir)
}

/// Returns the default 12 [`ProfileColumnSpec`] items of MESA `26.4.1`.
#[must_use]
pub fn default_star_profile_specs() -> Vec<ProfileColumnSpec> {
    parse_star_profile_columns(DEFAULT_STAR_PROFILE_COLUMNS_TEXT, None)
        .expect("embedded default star profile_columns.list is valid")
}

/// Signed base-10 logarithm `sign(1d0, val) * log10(max(1d0, abs(val)))`
/// (`star/private/profile_getval.f90:468, 476, 480, 483`).
#[inline]
fn signed_log(val: f64) -> f64 {
    let sgn = if val.is_sign_negative() { -1.0 } else { 1.0 };
    sgn * flexmesa_math::log10(val.abs().max(1.0))
}

struct ProfilePrefixCache {
    dm_prefix: Vec<f64>,
    dq_prefix: Vec<f64>,
    acoustic_depth: Vec<f64>,
    acoustic_radius: Vec<f64>,
}

impl ProfilePrefixCache {
    fn build(model: &StarModel, specs: &[ProfileColumnSpec]) -> Self {
        let nz = model.nz;
        let mut need_dm = false;
        let mut need_dq = false;
        let mut need_ac = false;
        for sp in specs {
            if let ProfileColumnSpec::Scalar(n) = sp {
                match *n {
                    "xm" | "logxm" => need_dm = true,
                    "xq" | "logxq" => need_dq = true,
                    "acoustic_depth"
                    | "log_acoustic_depth"
                    | "acoustic_radius"
                    | "log_acoustic_radius" => need_ac = true,
                    _ => {}
                }
            }
        }
        let dm_prefix = if need_dm {
            let mut v = Vec::with_capacity(nz);
            let mut acc = 0.0;
            for k in 0..nz {
                v.push(acc);
                acc += model.dm[k];
            }
            v
        } else {
            Vec::new()
        };
        let dq_prefix = if need_dq {
            let mut v = Vec::with_capacity(nz);
            let mut acc = 0.0;
            for k in 0..nz {
                v.push(acc);
                acc += model.dq[k];
            }
            v
        } else {
            Vec::new()
        };
        let (acoustic_depth, acoustic_radius) = if need_ac {
            let dr_cs: Vec<f64> = (0..nz)
                .map(|j| {
                    let dr = if j + 1 == nz {
                        model.r[j] - model.r_center
                    } else {
                        model.r[j] - model.r[j + 1]
                    };
                    dr / model.csound[j].max(1.0e-99)
                })
                .collect();
            let mut ad = Vec::with_capacity(nz);
            let mut acc_d = 0.0;
            for &x in &dr_cs {
                ad.push(acc_d);
                acc_d += x;
            }
            let mut ar = vec![0.0; nz];
            let mut acc_r = 0.0;
            for j in (0..nz).rev() {
                acc_r += dr_cs[j];
                ar[j] = acc_r;
            }
            (ad, ar)
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            dm_prefix,
            dq_prefix,
            acoustic_depth,
            acoustic_radius,
        }
    }
}

#[allow(clippy::too_many_lines)]
fn eval_profile_scalar(
    name: &'static str,
    model: &StarModel,
    ctx: &StepReportContext,
    prefixes: &ProfilePrefixCache,
    k: usize,
) -> MesaCellValue {
    let nz = model.nz;
    let r_surf = model.r[0].max(1.0e-99);
    let dr = if k + 1 == nz {
        model.r[k] - model.r_center
    } else {
        model.r[k] - model.r[k + 1]
    };
    let v_k = if model.controls.u_flag {
        model.u[k]
    } else if model.controls.v_flag {
        model.v[k]
    } else {
        0.0
    };
    let cs_f = csound_face_at(model, k);
    let mix_code = model.mixing_type[k].code();
    let d_non_rot = ctx.d_mix_non_rotation.get(k).copied().unwrap_or(0.0);

    match name {
        "zone" | "k" => MesaCellValue::Int((k + 1) as i64),
        "mixing_type" | "mix_type" | "mlt_mixing_type" => {
            MesaCellValue::Int(i64::from(mix_code))
        }

        "mass" => MesaCellValue::Float(model.m[k] / M_SUN),
        "log_mass" | "logM" => {
            MesaCellValue::Float(safe_log10(model.m[k] / M_SUN))
        }
        "mass_grams" => MesaCellValue::Float(model.m[k]),
        "mmid" => MesaCellValue::Float(
            (model.m_center + model.xmstar * (model.q[k] - 0.5 * model.dq[k]))
                / M_SUN,
        ),
        "dm" => MesaCellValue::Float(model.dm[k]),
        "dm_bar" => MesaCellValue::Float(model.dm_bar[k]),
        "m_grav" => MesaCellValue::Float(model.m_grav[k] / M_SUN),
        "m_grav_div_m_baryonic" => {
            MesaCellValue::Float(model.m_grav[k] / model.m[k].max(1.0e-99))
        }
        "mass_correction_factor" => {
            MesaCellValue::Float(model.aux.mass_correction[k])
        }

        "q" => MesaCellValue::Float(model.q[k]),
        "log_q" => MesaCellValue::Float(safe_log10(model.q[k])),
        "dq" => MesaCellValue::Float(model.dq[k]),
        "log_dq" | "logdq" => MesaCellValue::Float(safe_log10(model.dq[k])),
        "dq_ratio" => MesaCellValue::Float(if k == 0 || k + 1 == nz {
            1.0
        } else {
            model.dq[k - 1] / model.dq[k].max(1.0e-99)
        }),
        "xm" => MesaCellValue::Float(prefixes.dm_prefix[k] / M_SUN),
        "logxm" => {
            MesaCellValue::Float(safe_log10(prefixes.dm_prefix[k] / M_SUN))
        }
        "xq" => MesaCellValue::Float(prefixes.dq_prefix[k]),
        "logxq" => MesaCellValue::Float(safe_log10(prefixes.dq_prefix[k])),

        "radius" => MesaCellValue::Float(model.r[k] / R_SUN),
        "radius_cm" => MesaCellValue::Float(model.r[k]),
        "radius_km" => MesaCellValue::Float(model.r[k] * 1.0e-5),
        "rmid" => MesaCellValue::Float(model.rmid[k] / R_SUN),
        "logR_cm" => MesaCellValue::Float(safe_log10(model.r[k])),
        "logR" => MesaCellValue::Float(safe_log10(model.r[k] / R_SUN)),
        "r_div_R" => MesaCellValue::Float(model.r[k] / r_surf),
        "dr" => MesaCellValue::Float(dr),
        "log_dr" => MesaCellValue::Float(safe_log10(dr)),
        "dlogR" => MesaCellValue::Float(
            (if k + 1 == nz {
                model.ln_r[k] - flexmesa_math::log(model.r_center.max(1.0))
            } else {
                model.ln_r[k] - model.ln_r[k + 1]
            }) / LN10,
        ),
        "xr" => MesaCellValue::Float((model.r[0] - model.r[k]) / R_SUN),
        "xr_cm" => MesaCellValue::Float(model.r[0] - model.r[k]),
        "xr_div_R" => MesaCellValue::Float((model.r[0] - model.r[k]) / r_surf),

        "temperature" => MesaCellValue::Float(model.t[k]),
        "logT" => MesaCellValue::Float(model.ln_t[k] / LN10),
        "logT_face" => MesaCellValue::Float(if k == 0 {
            safe_log10(flexmesa_math::exp(model.ln_t_surf_ad.val))
        } else {
            (model.dq[k - 1] * model.ln_t[k] + model.dq[k] * model.ln_t[k - 1])
                / (model.dq[k - 1] + model.dq[k]).max(1.0e-99)
                / LN10
        }),
        "logT_bb" => MesaCellValue::Float(safe_log10(flexmesa_math::pow(
            model.lum[k].max(0.0)
                / (FOUR_PI * model.r[k] * model.r[k] * BOLTZ_SIGMA)
                    .max(1.0e-99),
            0.25,
        ))),
        "density" | "rho" => MesaCellValue::Float(model.rho[k]),
        "logRho" => MesaCellValue::Float(model.lnd[k] / LN10),
        "pgas" => MesaCellValue::Float(model.pgas[k]),
        "logPgas" => MesaCellValue::Float(model.ln_pgas[k] / LN10),
        "prad" => MesaCellValue::Float(model.prad[k]),
        "pressure" => MesaCellValue::Float(model.peos[k]),
        "logP" => MesaCellValue::Float(model.ln_peos[k] / LN10),
        "pgas_div_ptotal" | "pgas_div_p" => {
            MesaCellValue::Float(model.pgas[k] / model.peos[k].max(1.0e-99))
        }
        "prad_div_pgas" => {
            MesaCellValue::Float(model.prad[k] / model.pgas[k].max(1.0e-99))
        }

        "energy" => MesaCellValue::Float(model.energy[k]),
        "logE" => MesaCellValue::Float(model.ln_e[k] / LN10),
        "entropy" => MesaCellValue::Float(model.entropy[k]),
        "logS" => MesaCellValue::Float(model.ln_s[k] / LN10),
        "grada" => MesaCellValue::Float(model.grada[k]),
        "cv" => MesaCellValue::Float(model.cv[k]),
        "cp" => MesaCellValue::Float(model.cp[k]),
        "chiRho" => MesaCellValue::Float(model.chi_rho[k]),
        "chiT" => MesaCellValue::Float(model.chi_t[k]),
        "gamma1" => MesaCellValue::Float(model.gamma1[k]),
        "gamma3" => MesaCellValue::Float(model.gamma3[k]),
        "eta" => MesaCellValue::Float(model.eta[k]),
        "gam" => MesaCellValue::Float(model.gam[k]),
        "mu" => MesaCellValue::Float(model.mu[k]),
        "abar" => MesaCellValue::Float(model.abar[k]),
        "zbar" => MesaCellValue::Float(model.zbar[k]),
        "z2bar" => MesaCellValue::Float(model.z2bar[k]),
        "ye" => MesaCellValue::Float(model.ye[k]),
        "free_e" => MesaCellValue::Float(flexmesa_math::exp(model.lnfree_e[k])),
        "logfree_e" => MesaCellValue::Float(model.lnfree_e[k] / LN10),
        "csound" => MesaCellValue::Float(model.csound[k]),
        "csound_face" | "cs_at_cell_bdy" => MesaCellValue::Float(cs_f),
        "log_csound" => MesaCellValue::Float(safe_log10(model.csound[k])),
        "dr_div_cs" => MesaCellValue::Float(dr / model.csound[k].max(1.0e-99)),
        "log_dr_div_cs" => {
            MesaCellValue::Float(safe_log10(dr / model.csound[k].max(1.0e-99)))
        }
        "dt_cs_div_dr" => {
            MesaCellValue::Float(model.dt * model.csound[k] / dr.max(1.0e-99))
        }
        "log_dt_cs_div_dr" => MesaCellValue::Float(safe_log10(
            model.dt * model.csound[k] / dr.max(1.0e-99),
        )),

        "x" | "x_mass_fraction_H" => MesaCellValue::Float(model.xh_mass[k]),
        "log_x" => MesaCellValue::Float(safe_log10(model.xh_mass[k])),
        "y" | "y_mass_fraction_He" => MesaCellValue::Float(model.yhe_mass[k]),
        "log_y" => MesaCellValue::Float(safe_log10(model.yhe_mass[k])),
        "z" | "z_mass_fraction_metals" => MesaCellValue::Float(model.z_mass[k]),
        "log_z" => MesaCellValue::Float(safe_log10(model.z_mass[k])),

        "luminosity" => MesaCellValue::Float(model.lum[k] / L_SUN),
        // `p_logL` in `star/private/profile_getval.f90:485-486`: `safe_log10(max(1d-12, s% L(k)/Lsun))`.
        "logL" => MesaCellValue::Float(safe_log10(
            (model.lum[k] / L_SUN).max(1.0e-12),
        )),
        "lum_erg_s" => MesaCellValue::Float(model.lum[k]),
        "log_abs_lum_erg_s" => {
            MesaCellValue::Float(safe_log10(model.lum[k].abs()))
        }
        "eps_nuc" => MesaCellValue::Float(model.eps_nuc[k]),
        "log_abs_eps_nuc" => {
            MesaCellValue::Float(safe_log10(model.eps_nuc[k].abs()))
        }
        "eps_nuc_neu_total" => MesaCellValue::Float(model.eps_nuc_neu_total[k]),
        "non_nuc_neu" => MesaCellValue::Float(model.non_nuc_neu[k]),
        "nonnucneu_plas" => MesaCellValue::Float(
            model.aux.nonnucneu_channels.get(k).map_or(0.0, |c| c[0]),
        ),
        "nonnucneu_brem" => MesaCellValue::Float(
            model.aux.nonnucneu_channels.get(k).map_or(0.0, |c| c[1]),
        ),
        "nonnucneu_phot" => MesaCellValue::Float(
            model.aux.nonnucneu_channels.get(k).map_or(0.0, |c| c[2]),
        ),
        "nonnucneu_pair" => MesaCellValue::Float(
            model.aux.nonnucneu_channels.get(k).map_or(0.0, |c| c[3]),
        ),
        "nonnucneu_reco" => MesaCellValue::Float(
            model.aux.nonnucneu_channels.get(k).map_or(0.0, |c| c[4]),
        ),
        "eps_grav" => MesaCellValue::Float(model.eps_grav_ad[k].val),
        "signed_log_eps_grav" => {
            MesaCellValue::Float(signed_log(model.eps_grav_ad[k].val))
        }
        "eps_mdot" => MesaCellValue::Float(model.eps_mdot[k]),
        "eps_grav_plus_eps_mdot" => {
            MesaCellValue::Float(model.eps_grav_ad[k].val + model.eps_mdot[k])
        }
        "extra_heat" => MesaCellValue::Float(model.aux.extra_heat[k]),
        "net_nuclear_energy" => {
            let en = model.eps_nuc[k] - model.non_nuc_neu[k];
            MesaCellValue::Float(signed_log(en))
        }
        // `p_net_energy` in `star/private/profile_getval.f90:479-481`.
        "net_energy" => {
            let en = model.eps_nuc[k] - model.non_nuc_neu[k]
                + model.eps_grav_ad[k].val;
            MesaCellValue::Float(signed_log(en))
        }
        // `p_signed_log_power` in `star/private/profile_getval.f90:482-484`.
        "signed_log_power" => MesaCellValue::Float(signed_log(model.lum[k])),

        "opacity" => MesaCellValue::Float(model.opacity[k]),
        "log_opacity" | "log_kap" => {
            MesaCellValue::Float(safe_log10(model.opacity[k]))
        }
        "kap_frac_lowT" => MesaCellValue::Float(
            model.aux.kap_frac.get(k).map_or(0.0, |f| f[0]),
        ),
        "kap_frac_highT" => MesaCellValue::Float(
            model.aux.kap_frac.get(k).map_or(0.0, |f| f[1]),
        ),
        "kap_frac_Type2" => MesaCellValue::Float(
            model.aux.kap_frac.get(k).map_or(0.0, |f| f[2]),
        ),
        "kap_frac_Compton" => MesaCellValue::Float(
            model.aux.kap_frac.get(k).map_or(0.0, |f| f[3]),
        ),
        "eos_frac_OPAL_SCVH" => {
            MesaCellValue::Float(model.eos_res[k][I_FRAC_OPAL_SCVH])
        }
        "eos_frac_HELM" => MesaCellValue::Float(model.eos_res[k][I_FRAC_HELM]),
        "eos_frac_Skye" => MesaCellValue::Float(model.eos_res[k][I_FRAC_SKYE]),
        "eos_frac_PC" => MesaCellValue::Float(model.eos_res[k][I_FRAC_PC]),
        "eos_frac_FreeEOS" => {
            MesaCellValue::Float(model.eos_res[k][I_FRAC_FREE_EOS])
        }
        "eos_frac_CMS" => MesaCellValue::Float(model.eos_res[k][I_FRAC_CMS]),
        "eos_frac_ideal" => {
            MesaCellValue::Float(model.eos_res[k][I_FRAC_IDEAL])
        }
        "tau" => MesaCellValue::Float(model.tau[k]),
        "logtau" => MesaCellValue::Float(safe_log10(model.tau[k])),
        "xtau" => MesaCellValue::Float(model.tau[nz - 1] - model.tau[k]),
        "xlogtau" => {
            MesaCellValue::Float(safe_log10(model.tau[nz - 1] - model.tau[k]))
        }
        "logtau_sub_xlogtau" => MesaCellValue::Float(
            safe_log10(model.tau[k])
                - safe_log10(model.tau[nz - 1] - model.tau[k]),
        ),

        "gradT" => MesaCellValue::Float(model.gradt[k]),
        "gradr" => MesaCellValue::Float(model.gradr[k]),
        "gradL" => MesaCellValue::Float(model.gradl[k]),
        "diff_grads" => MesaCellValue::Float(model.gradr[k] - model.gradl[k]),
        "log_diff_grads" => MesaCellValue::Float(safe_log10(
            (model.gradr[k] - model.gradl[k]).abs(),
        )),
        "gradT_sub_grada" => {
            let ga = if k == 0 {
                model.grada[0]
            } else {
                model.interp_val_to_pt(&model.grada, k)
            };
            MesaCellValue::Float(model.gradt[k] - ga)
        }
        "grada_sub_gradT" => {
            let ga = if k == 0 {
                model.grada[0]
            } else {
                model.interp_val_to_pt(&model.grada, k)
            };
            MesaCellValue::Float(ga - model.gradt[k])
        }
        "gradr_sub_grada" => {
            let ga = if k == 0 {
                model.grada[0]
            } else {
                model.interp_val_to_pt(&model.grada, k)
            };
            MesaCellValue::Float(model.gradr[k] - ga)
        }
        "gradT_div_grada" => {
            let ga = if k == 0 {
                model.grada[0]
            } else {
                model.interp_val_to_pt(&model.grada, k)
            };
            MesaCellValue::Float(model.gradt[k] / ga.max(1.0e-99))
        }
        // `p_scale_height` vs `p_pressure_scale_height{_cm}` (`profile_getval.f90:1419-1426`).
        "scale_height" => MesaCellValue::Float(model.scale_height[k] / R_SUN),
        "pressure_scale_height" => {
            let denom = model.rho[k] * model.grav[k];
            let hp_cm = if denom == 0.0 {
                0.0
            } else {
                model.peos[k] / denom
            };
            MesaCellValue::Float(hp_cm / R_SUN)
        }
        "pressure_scale_height_cm" => {
            let denom = model.rho[k] * model.grav[k];
            MesaCellValue::Float(if denom == 0.0 {
                0.0
            } else {
                model.peos[k] / denom
            })
        }
        "mlt_mixing_length" => MesaCellValue::Float(
            model.controls.mixing_length_alpha * model.scale_height[k],
        ),
        "conv_vel" => MesaCellValue::Float(model.conv_vel[k]),
        "log_conv_vel" => MesaCellValue::Float(safe_log10(model.conv_vel[k])),
        "conv_vel_div_csound" => {
            MesaCellValue::Float(model.conv_vel[k] / cs_f.max(1.0e-99))
        }
        "log_D_mix" => MesaCellValue::Float(safe_log10(model.d_mix[k])),
        "log_mlt_D_mix" => MesaCellValue::Float(safe_log10(model.mlt_d[k])),
        "log_mlt_Gamma" => MesaCellValue::Float(safe_log10(model.mlt_gamma[k])),
        "log_sig_mix" => MesaCellValue::Float(safe_log10(model.sig[k])),
        // `p_log_D_{conv,semi,ovr,thrm}` use `D_mix_non_rotation(k)` (`profile_getval.f90:1263-1299`).
        "log_D_conv" => MesaCellValue::Float(
            if model.mixing_type[k] == MixingType::ConvectiveMixing {
                safe_log10(d_non_rot)
            } else {
                -99.0
            },
        ),
        "log_D_semi" => MesaCellValue::Float(
            if model.mixing_type[k] == MixingType::SemiconvectiveMixing {
                safe_log10(d_non_rot)
            } else {
                -99.0
            },
        ),
        "log_D_ovr" => MesaCellValue::Float(
            if model.mixing_type[k] == MixingType::OvershootMixing {
                safe_log10(d_non_rot)
            } else {
                -99.0
            },
        ),
        "log_D_thrm" => MesaCellValue::Float(
            if model.mixing_type[k] == MixingType::ThermohalineMixing {
                safe_log10(d_non_rot)
            } else {
                -99.0
            },
        ),
        "log_D_minimum" => MesaCellValue::Float(
            if model.mixing_type[k] == MixingType::MinimumMixing {
                safe_log10(model.d_mix[k])
            } else {
                -99.0
            },
        ),

        // `p_brunt_*` gated on `calculate_Brunt_N2` (`profile_getval.f90:2077-2156`).
        "brunt_B" => {
            MesaCellValue::Float(if model.controls.calculate_brunt_n2 {
                model.brunt_b[k]
            } else {
                0.0
            })
        }
        "brunt_N2" => {
            MesaCellValue::Float(if model.controls.calculate_brunt_n2 {
                model.brunt_n2[k]
            } else {
                0.0
            })
        }
        "brunt_N2_composition_term" => {
            MesaCellValue::Float(if model.controls.calculate_brunt_n2 {
                model.aux.brunt_n2_composition_term[k]
            } else {
                0.0
            })
        }
        "brunt_N2_structure_term" => {
            MesaCellValue::Float(if model.controls.calculate_brunt_n2 {
                model.brunt_n2[k] - model.aux.brunt_n2_composition_term[k]
            } else {
                0.0
            })
        }
        "brunt_N" => {
            MesaCellValue::Float(if model.controls.calculate_brunt_n2 {
                model.brunt_n2[k].max(0.0).sqrt()
            } else {
                0.0
            })
        }
        "sign_brunt_N2" => {
            MesaCellValue::Float(if !model.controls.calculate_brunt_n2 {
                0.0
            } else if model.brunt_n2[k] < 0.0 {
                -1.0
            } else {
                1.0
            })
        }
        "lamb_S" => MesaCellValue::Float(
            2.0_f64.sqrt() * cs_f / model.r[k].max(1.0e-99),
        ),
        "lamb_S2" => {
            let s = 2.0_f64.sqrt() * cs_f / model.r[k].max(1.0e-99);
            MesaCellValue::Float(s * s)
        }
        "grav" => MesaCellValue::Float(model.grav[k]),
        "log_g" => MesaCellValue::Float(safe_log10(model.grav[k])),

        "v" | "velocity" => MesaCellValue::Float(v_k),
        "v_kms" | "vel_km_per_s" => MesaCellValue::Float(v_k * 1.0e-5),
        "u" => MesaCellValue::Float(if model.controls.u_flag {
            model.u[k]
        } else {
            0.0
        }),
        "u_face" => MesaCellValue::Float(if model.controls.u_flag {
            model.aux.u_face_ad[k].val
        } else {
            0.0
        }),
        "v_div_csound" | "v_div_cs" => {
            MesaCellValue::Float(v_k / cs_f.max(1.0e-99))
        }
        "v_div_r" => {
            let vf = if model.controls.u_flag {
                model.aux.u_face_ad[k].val
            } else if model.controls.v_flag {
                model.v[k]
            } else {
                0.0
            };
            MesaCellValue::Float(vf / model.r[k].max(1.0e-99))
        }
        // `p_v_div_vesc` (`u(k)`) vs `p_v_div_v_escape` (`u_face_ad(k)%val`) (`profile_getval.f90:967-980`).
        "v_div_vesc" => {
            let vesc = (2.0 * model.cgrav[k] * model.m[k]
                / model.r[k].max(1.0e-99))
            .max(0.0)
            .sqrt();
            MesaCellValue::Float(if vesc > 0.0 { v_k / vesc } else { 0.0 })
        }
        "v_div_v_escape" => {
            let vf = if model.controls.u_flag {
                model.aux.u_face_ad[k].val
            } else if model.controls.v_flag {
                model.v[k]
            } else {
                0.0
            };
            let vesc = (2.0 * model.cgrav[k] * model.m[k]
                / model.r[k].max(1.0e-99))
            .max(0.0)
            .sqrt();
            MesaCellValue::Float(if vesc > 0.0 { vf / vesc } else { 0.0 })
        }

        "omega" => MesaCellValue::Float(if model.controls.rotation_flag {
            model.omega[k]
        } else {
            0.0
        }),
        "log_omega" => {
            MesaCellValue::Float(safe_log10(if model.controls.rotation_flag {
                model.omega[k]
            } else {
                0.0
            }))
        }
        // `p_omega_crit` & `p_omega_div_omega_crit` (`profile_getval.f90:1515-1525`).
        "omega_crit" => {
            MesaCellValue::Float(eval_upstream_omega_crit(model, k))
        }
        "omega_div_omega_crit" => {
            MesaCellValue::Float(if model.controls.rotation_flag {
                let wc = eval_upstream_omega_crit(model, k);
                if wc > 0.0 {
                    model.omega[k] / wc
                } else {
                    0.0
                }
            } else {
                0.0
            })
        }
        "j_rot" => MesaCellValue::Float(if model.controls.rotation_flag {
            model.j_rot[k]
        } else {
            0.0
        }),
        "log_j_rot" => {
            MesaCellValue::Float(safe_log10(if model.controls.rotation_flag {
                model.j_rot[k].abs()
            } else {
                0.0
            }))
        }
        "i_rot" => MesaCellValue::Float(if model.controls.rotation_flag {
            model.aux.i_rot[k]
        } else {
            0.0
        }),
        "v_rot" => MesaCellValue::Float(if model.controls.rotation_flag {
            let (re, _) = equatorial_and_polar_r_cm(model, k);
            model.omega[k] * re * 1.0e-5
        } else {
            0.0
        }),
        "fp_rot" => MesaCellValue::Float(model.aux.fp_rot[k]),
        "ft_rot" => MesaCellValue::Float(model.aux.ft_rot[k]),
        "r_equatorial" => {
            MesaCellValue::Float(equatorial_and_polar_r_cm(model, k).0 / R_SUN)
        }
        "r_polar" => {
            MesaCellValue::Float(equatorial_and_polar_r_cm(model, k).1 / R_SUN)
        }
        "r_e_div_r_p" => {
            let (re, rp) = equatorial_and_polar_r_cm(model, k);
            MesaCellValue::Float(re / rp.max(1.0e-99))
        }
        "log_r_equatorial" => MesaCellValue::Float(safe_log10(
            equatorial_and_polar_r_cm(model, k).0 / R_SUN,
        )),
        "log_r_polar" => MesaCellValue::Float(safe_log10(
            equatorial_and_polar_r_cm(model, k).1 / R_SUN,
        )),

        // `p_*_RTI` (`profile_getval.f90:1321-1356`).
        "alpha_RTI" => MesaCellValue::Float(if model.controls.rti_flag {
            model.aux.alpha_rti[k]
        } else {
            0.0
        }),
        "log_alpha_RTI" => MesaCellValue::Float(if model.controls.rti_flag {
            safe_log10(model.aux.alpha_rti[k])
        } else {
            0.0
        }),
        "eta_RTI" => MesaCellValue::Float(if model.controls.rti_flag {
            model.aux.eta_rti[k]
        } else {
            0.0
        }),
        "log_eta_RTI" => MesaCellValue::Float(if model.controls.rti_flag {
            safe_log10(model.aux.eta_rti[k].abs())
        } else {
            0.0
        }),
        "log_sig_RTI" => MesaCellValue::Float(if model.controls.rti_flag {
            safe_log10(model.aux.sig_rti[k])
        } else {
            0.0
        }),
        "dPdr_dRhodr_info" => {
            MesaCellValue::Float(if model.controls.rti_flag {
                model.aux.dpdr_drhodr_info[k]
            } else {
                0.0
            })
        }
        "log_source_plus_alpha_RTI" => {
            MesaCellValue::Float(if model.controls.rti_flag {
                safe_log10(model.aux.source_plus_alpha_rti[k])
            } else {
                0.0
            })
        }
        "log_source_minus_alpha_RTI" => {
            MesaCellValue::Float(if model.controls.rti_flag {
                safe_log10(model.aux.source_minus_alpha_rti[k].abs())
            } else {
                0.0
            })
        }

        // `p_log_L_div_Ledd` vs `p_log_Lrad_div_Ledd` (`profile_getval.f90:456-457, 491-492`).
        "log_L_div_Ledd" => {
            let ledd = get_ledd(model, k);
            let ratio = if ledd > 0.0 { model.lum[k] / ledd } else { 0.0 };
            MesaCellValue::Float(safe_log10(ratio.max(1.0e-12)))
        }
        "log_Lrad_div_Ledd" => {
            MesaCellValue::Float(safe_log10(get_lrad_div_ledd(model, k)))
        }

        "eps_nuc_plus_nuc_neu" => {
            MesaCellValue::Float(model.eps_nuc[k] + model.eps_nuc_neu_total[k])
        }
        "logS_per_baryon" => MesaCellValue::Float(
            model.ln_s[k] / LN10 + flexmesa_math::log10(flexmesa_const::AMU),
        ),
        "cgrav_factor" => MesaCellValue::Float(
            model.cgrav[k] / flexmesa_const::STANDARD_CGRAV,
        ),
        "logQ" => MesaCellValue::Float(
            model.lnd[k] / LN10 - 2.0 * model.ln_t[k] / LN10 + 12.0,
        ),
        "total_energy" => {
            MesaCellValue::Float(eval_cell_specific_total_energy(model, k))
        }
        "acoustic_radius" => MesaCellValue::Float(prefixes.acoustic_radius[k]),
        "log_acoustic_radius" => {
            MesaCellValue::Float(safe_log10(prefixes.acoustic_radius[k]))
        }
        "acoustic_depth" => MesaCellValue::Float(prefixes.acoustic_depth[k]),
        "log_acoustic_depth" => {
            MesaCellValue::Float(safe_log10(prefixes.acoustic_depth[k]))
        }
        other => unreachable!(
            "validated profile scalar column {other:?} missing evaluator arm"
        ),
    }
}

fn species_mass_msun(model: &StarModel, iso_name: &str) -> f64 {
    let Some(j) = resolve_iso_slot(model, iso_name).ok().and_then(|(_, s)| s)
    else {
        return 0.0;
    };
    let mut sum = 0.0;
    for k in 0..model.nz {
        sum += model.dm[k] * model.cell_xa(k)[j];
    }
    sum / M_SUN
}

fn validate_extra_profile_inputs(
    nz: usize,
    extra_header_items: &[(String, f64)],
    extra_cols: &[(String, Vec<f64>)],
) -> Result<(), StarError> {
    validate_extra_header_items("profile", extra_header_items)?;
    for (idx, (name, vals)) in extra_cols.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(StarError::InvalidInput(format!(
                "extra profile column #{idx} has an empty name"
            )));
        }
        if vals.len() != nz {
            return Err(StarError::InvalidInput(format!(
                "extra profile column '{}' has length {}, expected nz={nz}",
                name.trim(),
                vals.len()
            )));
        }
    }
    Ok(())
}

/// Formats a complete MESA `26.4.1` `profileN.data` snapshot from `model` and
/// `specs` (`do_profile_info` in `star/private/profile.f90:519-660`).
#[must_use]
pub fn format_star_profile_text(
    model: &StarModel,
    specs: &[ProfileColumnSpec],
) -> String {
    let ctx = StepReportContext::compute(model);
    let (hdr, body) = format_star_profile_parts_with_extras_impl(
        model,
        specs,
        &ctx,
        &[],
        &[],
    );
    let mut out = String::with_capacity(hdr.len() + body.len());
    out.push_str(&hdr);
    out.push_str(&body);
    out
}

/// Formats a complete MESA `26.4.1` `profileN.data` snapshot including
/// `extra_profile_header_items` (`star/private/profile.f90:590-592`) and
/// `extra_profile_columns` (`profile.f90:653-656`).
pub fn format_star_profile_text_with_extras(
    model: &StarModel,
    specs: &[ProfileColumnSpec],
    extra_header_items: &[(String, f64)],
    extra_cols: &[(String, Vec<f64>)],
) -> Result<String, StarError> {
    validate_extra_profile_inputs(model.nz, extra_header_items, extra_cols)?;
    let ctx = StepReportContext::compute(model);
    let (hdr, body) = format_star_profile_parts_with_extras_impl(
        model,
        specs,
        &ctx,
        extra_header_items,
        extra_cols,
    );
    let mut out = String::with_capacity(hdr.len() + body.len());
    out.push_str(&hdr);
    out.push_str(&body);
    Ok(out)
}

fn format_star_profile_parts_with_extras_impl(
    model: &StarModel,
    specs: &[ProfileColumnSpec],
    ctx: &StepReportContext,
    extra_header_items: &[(String, f64)],
    extra_cols: &[(String, Vec<f64>)],
) -> (String, String) {
    let nz = model.nz;
    let o = &model.controls.output;
    let star_mass = model.mstar / M_SUN;
    let star_mdot = model.mstar_dot / (M_SUN / SECYER);
    let star_age = if model.model_number == 0 {
        0.0
    } else {
        model.star_age
    };

    // 1. Header items (`profile.f90:520-589`).
    let mut hdr: Vec<(&str, String)> = Vec::with_capacity(56);
    hdr.push(("model_number", format_mesa_int(model.model_number as i64)));
    hdr.push(("num_zones", format_mesa_int(nz as i64)));
    hdr.push(("initial_mass", format_mesa_dbl(model.controls.initial_mass)));
    hdr.push(("initial_z", format_mesa_dbl(model.controls.initial_z)));
    hdr.push(("star_age", format_mesa_dbl(star_age)));
    hdr.push(("time_step", format_mesa_dbl(model.dt / SECYER)));
    if model.m_center != 0.0 {
        hdr.push(("M_center", format_mesa_dbl(model.m_center)));
    }
    if model.v_center != 0.0 {
        hdr.push(("v_center", format_mesa_dbl(model.v_center)));
    }
    if model.r_center != 0.0 {
        hdr.push(("R_center", format_mesa_dbl(model.r_center)));
    }
    if model.l_center != 0.0 {
        hdr.push(("L_center", format_mesa_dbl(model.l_center)));
    }
    hdr.push(("Teff", format_mesa_dbl(model.teff)));
    hdr.push(("photosphere_L", format_mesa_dbl(ctx.phot.l_lsun)));
    hdr.push(("photosphere_r", format_mesa_dbl(ctx.phot.r_rsun)));
    hdr.push((
        "center_eta",
        format_mesa_dbl(center_value(model, &model.eta)),
    ));
    for (label, iso) in [
        ("center_h1", "h1"),
        ("center_he3", "he3"),
        ("center_he4", "he4"),
        ("center_c12", "c12"),
        ("center_n14", "n14"),
        ("center_o16", "o16"),
        ("center_ne20", "ne20"),
    ] {
        hdr.push((label, format_mesa_dbl(center_avg_x_by_name(model, iso))));
    }
    hdr.push(("star_mass", format_mesa_dbl(star_mass)));
    hdr.push(("star_mdot", format_mesa_dbl(star_mdot)));
    for (label, iso) in [
        ("star_mass_h1", "h1"),
        ("star_mass_he3", "he3"),
        ("star_mass_he4", "he4"),
        ("star_mass_c12", "c12"),
        ("star_mass_n14", "n14"),
        ("star_mass_o16", "o16"),
        ("star_mass_ne20", "ne20"),
    ] {
        hdr.push((label, format_mesa_dbl(species_mass_msun(model, iso))));
    }
    hdr.push(("he_core_mass", format_mesa_dbl(ctx.cores.he_core_mass)));
    hdr.push(("co_core_mass", format_mesa_dbl(ctx.cores.co_core_mass)));
    hdr.push(("fe_core_mass", format_mesa_dbl(ctx.cores.fe_core_mass)));
    hdr.push((
        "neutron_rich_core_mass",
        format_mesa_dbl(ctx.cores.neutron_rich_core_mass),
    ));
    hdr.push(("dynamic_time", format_mesa_dbl(ctx.dynamic_timescale_s)));
    hdr.push(("kh_timescale", format_mesa_dbl(ctx.kh_timescale_yr)));
    hdr.push(("nuc_timescale", format_mesa_dbl(ctx.nuc_timescale_yr)));
    hdr.push(("power_nuc_burn", format_mesa_dbl(ctx.powers.power_nuc_burn)));
    hdr.push(("power_h_burn", format_mesa_dbl(ctx.powers.power_h_burn)));
    hdr.push(("power_he_burn", format_mesa_dbl(ctx.powers.power_he_burn)));
    hdr.push(("power_neu", format_mesa_dbl(ctx.powers.power_neutrinos)));
    hdr.push(("burn_min1", format_mesa_dbl(o.burn_min1)));
    hdr.push(("burn_min2", format_mesa_dbl(o.burn_min2)));
    hdr.push(("time_seconds", format_mesa_dbl(star_age * SECYER)));
    hdr.push((
        "version_number",
        format_mesa_header_str(MESA_REF_VERSION_NUMBER),
    ));
    if o.profile_header_include_sys_details {
        hdr.push(("compiler", format_mesa_header_str("rustc")));
        hdr.push(("build", format_mesa_header_str("flexmesa")));
        hdr.push(("MESA_SDK_version", format_mesa_header_str("flexmesa-0.1")));
        hdr.push((
            "math_backend",
            format_mesa_header_str(flexmesa_math::numerics_mode().as_str()),
        ));
        hdr.push(("date", format_mesa_header_str("20260401")));
    }
    hdr.push(("msun", format_mesa_dbl(M_SUN)));
    hdr.push(("rsun", format_mesa_dbl(R_SUN)));
    hdr.push(("lsun", format_mesa_dbl(L_SUN)));
    for (name, val) in extra_header_items {
        hdr.push((name.trim(), format_mesa_dbl(*val)));
    }

    // 2. Expand column names (`profile.f90:633-656`) and pre-resolve nuclide slots.
    let mut col_names_owned: Vec<String> = Vec::new();
    let mut spec_slots: Vec<Option<usize>> = Vec::with_capacity(specs.len());
    for sp in specs {
        match sp {
            ProfileColumnSpec::Scalar(name) => {
                col_names_owned.push((*name).to_owned());
                spec_slots.push(None);
            }
            ProfileColumnSpec::AddAbundances => {
                for iso in &model.iso_names {
                    col_names_owned.push(iso.clone());
                }
                spec_slots.push(None);
            }
            ProfileColumnSpec::AddLogAbundances => {
                for iso in &model.iso_names {
                    col_names_owned.push(format!("log_{iso}"));
                }
                spec_slots.push(None);
            }
            ProfileColumnSpec::Nuclide(cid, iso) => {
                col_names_owned.push(iso.clone());
                spec_slots.push(iso_slot_for_chem_id(model, *cid));
            }
            ProfileColumnSpec::LogNuclide(cid, iso) => {
                col_names_owned.push(format!("log_{iso}"));
                spec_slots.push(iso_slot_for_chem_id(model, *cid));
            }
            ProfileColumnSpec::CategoryEps(cid) => {
                col_names_owned
                    .push(category_name(*cid).unwrap_or("unknown").to_owned());
                spec_slots.push(None);
            }
        }
    }
    for (name, _) in extra_cols {
        col_names_owned.push(name.trim().to_owned());
    }
    let col_refs: Vec<&str> =
        col_names_owned.iter().map(String::as_str).collect();
    let mut hdr_text =
        String::with_capacity(hdr.len() * 128 + col_refs.len() * 84);
    push_mesa_6line_header(&mut hdr_text, &hdr, &col_refs, STAR_LOG_COL_WIDTH);

    let n_rows = if o.max_num_profile_zones > 1 {
        nz.min(o.max_num_profile_zones as usize)
    } else {
        nz
    };
    let prefixes = ProfilePrefixCache::build(model, specs);
    let mut body = String::with_capacity(n_rows * col_refs.len() * 41);

    for row_idx in 0..n_rows {
        let kk = if n_rows > 1 && n_rows < nz {
            ((0.5
                + ((nz - 1) as f64) * (row_idx as f64) / ((n_rows - 1) as f64))
                .floor() as usize)
                .min(nz - 1)
        } else {
            row_idx
        };
        for (sp, &slot) in specs.iter().zip(&spec_slots) {
            match sp {
                ProfileColumnSpec::Scalar(name) => {
                    match eval_profile_scalar(name, model, ctx, &prefixes, kk) {
                        MesaCellValue::Int(v) => {
                            write_mesa_int_width(
                                &mut body,
                                v,
                                STAR_LOG_COL_WIDTH,
                            );
                        }
                        MesaCellValue::Float(v) => {
                            write_mesa_dbl_width(
                                &mut body,
                                v,
                                STAR_LOG_COL_WIDTH,
                            );
                        }
                    }
                }
                ProfileColumnSpec::AddAbundances => {
                    for &x in model.cell_xa(kk) {
                        write_mesa_dbl_width(&mut body, x, STAR_LOG_COL_WIDTH);
                    }
                }
                ProfileColumnSpec::AddLogAbundances => {
                    for &x in model.cell_xa(kk) {
                        write_mesa_dbl_width(
                            &mut body,
                            safe_log10(x),
                            STAR_LOG_COL_WIDTH,
                        );
                    }
                }
                ProfileColumnSpec::Nuclide(_, _) => {
                    let v = slot.map_or(0.0, |j| model.cell_xa(kk)[j]);
                    write_mesa_dbl_width(&mut body, v, STAR_LOG_COL_WIDTH);
                }
                ProfileColumnSpec::LogNuclide(_, _) => {
                    let v = slot
                        .map_or(-99.0, |j| safe_log10(model.cell_xa(kk)[j]));
                    write_mesa_dbl_width(&mut body, v, STAR_LOG_COL_WIDTH);
                }
                ProfileColumnSpec::CategoryEps(cid) => {
                    let v = model
                        .aux
                        .eps_nuc_categories
                        .get(kk)
                        .map_or(0.0, |c| c[cid - 1]);
                    write_mesa_dbl_width(&mut body, v, STAR_LOG_COL_WIDTH);
                }
            }
        }
        for (_, vals) in extra_cols {
            let v = vals.get(kk).copied().unwrap_or(f64::NAN);
            write_mesa_dbl_width(&mut body, v, STAR_LOG_COL_WIDTH);
        }
        let _ = writeln!(body);
    }
    (hdr_text, body)
}

/// Single entry in `LOGS/profiles.index` (`star/private/profile.f90:1024-1160`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfilesIndexEntry {
    /// Stellar `model_number` of the saved profile.
    pub model_number: usize,
    /// Retention priority (`1` = `delta_priority`, `2` = `phase_priority`, `3` = termination/explicit).
    pub priority: i32,
    /// 1-based profile slot number (`profile<profile_number>.data`).
    pub profile_number: usize,
}

/// Priority-evicting `LOGS/profiles.index` manager (`do_save_profiles` in
/// `star/private/profile.f90:944-1184`).
#[derive(Debug, Clone, Default)]
pub struct ProfilesIndex {
    /// Ordered entries currently recorded in `profiles.index`.
    pub entries: Vec<ProfilesIndexEntry>,
}

impl ProfilesIndex {
    /// Formats `self` in exact MESA `profiles.index` list-directed layout
    /// (`write_profiles_list` in `star/private/profile.f90:1024-1049`).
    #[must_use]
    pub fn format_text(&self) -> String {
        let n = self.entries.len();
        let noun = if n == 1 { "model." } else { "models." };
        let mut out = format!(
            "{n:12} {noun}    lines hold model number, priority, and profile number.\n"
        );
        for e in &self.entries {
            let _ = writeln!(
                out,
                "{:12}{:12}{:12}",
                e.model_number, e.priority, e.profile_number
            );
        }
        out
    }

    /// Saves a radial profile for `model` into `log_dir` with `priority`,
    /// writing `profileN.data` **before** updating `profiles.index` so an
    /// interrupted write never corrupts the index (`profile.f90:999-1010`).
    pub fn save_profile(
        &mut self,
        model: &StarModel,
        specs: &[ProfileColumnSpec],
        log_dir: &Path,
        priority: i32,
    ) -> Result<PathBuf, StarError> {
        let ctx = StepReportContext::compute(model);
        self.save_profile_with_ctx(model, specs, &ctx, log_dir, priority)
    }

    /// Saves a radial profile reusing `ctx` (also honoring `profile_data_header_suffix`
    /// per `star/private/profile.f90:484-488, 613-620`).
    pub fn save_profile_with_ctx(
        &mut self,
        model: &StarModel,
        specs: &[ProfileColumnSpec],
        ctx: &StepReportContext,
        log_dir: &Path,
        priority: i32,
    ) -> Result<PathBuf, StarError> {
        self.save_profile_with_ctx_and_extras(
            model,
            specs,
            ctx,
            log_dir,
            priority,
            &[],
            &[],
        )
    }

    /// Saves a radial profile reusing `ctx` together with `extra_profile_header_items`
    /// and `extra_profile_columns` (`star/private/profile.f90:590-592, 653-656`).
    pub fn save_profile_with_ctx_and_extras(
        &mut self,
        model: &StarModel,
        specs: &[ProfileColumnSpec],
        ctx: &StepReportContext,
        log_dir: &Path,
        priority: i32,
        extra_header_items: &[(String, f64)],
        extra_cols: &[(String, Vec<f64>)],
    ) -> Result<PathBuf, StarError> {
        validate_extra_profile_inputs(
            model.nz,
            extra_header_items,
            extra_cols,
        )?;
        fs::create_dir_all(log_dir).map_err(|e| {
            StarError::IoError(format!(
                "failed to create log directory {}: {e}",
                log_dir.display()
            ))
        })?;
        let o = &model.controls.output;
        let max_mods = if o.max_num_profile_models < 0 {
            model.model_number.max(self.entries.len() + 1).max(1)
        } else {
            o.max_num_profile_models.max(1) as usize
        };

        // `make_room_for_profile_info` (`profile.f90:1080-1135`).
        self.entries.retain(|e| {
            e.model_number < model.model_number && e.profile_number <= max_mods
        });
        if self.entries.len() >= max_mods {
            let mut drop_idx = 0_usize;
            for i in 1..self.entries.len() {
                let cur = self.entries[i];
                let best = self.entries[drop_idx];
                if cur.priority < best.priority
                    || (cur.priority == best.priority
                        && cur.model_number < best.model_number)
                {
                    drop_idx = i;
                }
            }
            self.entries.remove(drop_idx);
        }

        // `pick_model_profile_number` (`profile.f90:1052-1077`).
        let mut in_use = vec![false; max_mods + 1];
        for e in &self.entries {
            if e.profile_number <= max_mods {
                in_use[e.profile_number] = true;
            }
        }
        let prof_num = (1..=max_mods)
            .find(|&n| !in_use[n])
            .unwrap_or(self.entries.len() + 1);

        let prof_filename = format!(
            "{}{}{}",
            o.profile_data_prefix.trim(),
            prof_num,
            o.profile_data_suffix.trim()
        );
        let prof_path = log_dir.join(&prof_filename);
        let (hdr_text, body_text) = format_star_profile_parts_with_extras_impl(
            model,
            specs,
            ctx,
            extra_header_items,
            extra_cols,
        );
        let hdr_suffix = o.profile_data_header_suffix.trim();
        if hdr_suffix.is_empty() {
            let mut full =
                String::with_capacity(hdr_text.len() + body_text.len());
            full.push_str(&hdr_text);
            full.push_str(&body_text);
            fs::write(&prof_path, full).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write profile {}: {e}",
                    prof_path.display()
                ))
            })?;
        } else {
            let hdr_path = log_dir.join(format!("{prof_filename}{hdr_suffix}"));
            fs::write(&hdr_path, hdr_text).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write profile header {}: {e}",
                    hdr_path.display()
                ))
            })?;
            fs::write(&prof_path, body_text).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write profile {}: {e}",
                    prof_path.display()
                ))
            })?;
        }

        self.entries.push(ProfilesIndexEntry {
            model_number: model.model_number,
            priority,
            profile_number: prof_num,
        });
        let index_path = log_dir.join(o.profiles_index_name.trim());
        fs::write(&index_path, self.format_text()).map_err(|e| {
            StarError::IoError(format!(
                "failed to write {}: {e}",
                index_path.display()
            ))
        })?;
        Ok(prof_path)
    }
}

/// Caller-supplied extra log headers and columns for one converged step.
#[derive(Debug, Default, Clone, Copy)]
pub struct StarStepLogExtras<'a> {
    /// Extra header scalars emitted on `history.data` header creation
    /// (`data_for_extra_history_header_items`).
    pub history_header_items: &'a [(String, f64)],
    /// Extra per-step scalars appended to `history.data` rows
    /// (`data_for_extra_history_columns`).
    pub history_columns: &'a [(String, f64)],
    /// Extra header scalars emitted on each `profileN.data` header
    /// (`data_for_extra_profile_header_items`).
    pub profile_header_items: &'a [(String, f64)],
    /// Extra per-zone arrays appended to `profileN.data` rows
    /// (`data_for_extra_profile_columns`).
    pub profile_columns: &'a [(String, Vec<f64>)],
}

/// Stage-level `LOGS/` coordinator writing `history.data`, `profileN.data`, and
/// `profiles.index` incrementally according to `do_one_check_model`
/// (`star/private/do_one_utils.f90:507-536, 1520-1540`) and
/// `terminate_normal_evolve_loop` (`star/job/run_star_support.f90:783-806`).
pub struct StarLogsWriter {
    log_dir: PathBuf,
    output_root: PathBuf,
    history_writer: Option<IncrementalHistoryWriter>,
    profile_specs: Vec<ProfileColumnSpec>,
    profiles_index: ProfilesIndex,
    history_interval_override: Option<usize>,
    write_profiles_override: Option<bool>,
}

impl StarLogsWriter {
    /// Initializes a [`StarLogsWriter`] for `model` rooted at `output_root`
    /// (writing to `<output_root>/<log_directory>/`), resolving column list
    /// files from `case_dir`.
    pub fn for_stage(
        model: &StarModel,
        case_dir: &Path,
        output_root: &Path,
        history_interval_override: Option<usize>,
        write_profiles_override: Option<bool>,
    ) -> Result<Self, StarError> {
        Self::for_stage_with_logs_dir(
            model,
            case_dir,
            output_root,
            None,
            history_interval_override,
            write_profiles_override,
        )
    }

    /// Initializes a [`StarLogsWriter`] honoring an optional `do_one` 3rd-argument
    /// `stage_logs_dir` (`star/test_suite/test_suite_helpers:45-49`) when
    /// `model.controls.output.log_directory` has its default `"LOGS"` value.
    pub fn for_stage_with_logs_dir(
        model: &StarModel,
        case_dir: &Path,
        output_root: &Path,
        stage_logs_dir: Option<&str>,
        history_interval_override: Option<usize>,
        write_profiles_override: Option<bool>,
    ) -> Result<Self, StarError> {
        Self::for_stage_with_logs_dir_and_profile_extras(
            model,
            case_dir,
            output_root,
            stage_logs_dir,
            history_interval_override,
            write_profiles_override,
            |_| Ok((Vec::new(), Vec::new())),
        )
    }

    /// Initializes a [`StarLogsWriter`] honoring `stage_logs_dir` and fetching
    /// `extra_profile_header_items` / `extra_profile_columns` only when
    /// `profile_starting_model` triggers an initial profile snapshot.
    pub fn for_stage_with_logs_dir_and_profile_extras<F>(
        model: &StarModel,
        case_dir: &Path,
        output_root: &Path,
        stage_logs_dir: Option<&str>,
        history_interval_override: Option<usize>,
        write_profiles_override: Option<bool>,
        fetch_starting_profile_extras: F,
    ) -> Result<Self, StarError>
    where
        F: FnOnce(
            &StarModel,
        ) -> Result<(Vec<(String, f64)>, Vec<(String, Vec<f64>)>), StarError>,
    {
        let o = &model.controls.output;
        let ctrl_log_dir = o.log_directory.trim();
        let eff_log_dir = match stage_logs_dir.map(str::trim) {
            Some(s) if !s.is_empty() && ctrl_log_dir == "LOGS" => s,
            _ => ctrl_log_dir,
        };
        let log_dir = output_root.join(eff_log_dir);
        let hist_specs = crate::mesa_history::resolve_star_history_columns(
            model,
            Some(case_dir),
        )?;
        let write_profs =
            write_profiles_override.unwrap_or(o.write_profiles_flag);
        let allow_term_prof = write_profiles_override != Some(false)
            && model.job.write_profile_when_terminate;
        let profile_specs = if write_profs || allow_term_prof {
            resolve_star_profile_columns(model, Some(case_dir))?
        } else {
            default_star_profile_specs()
        };
        let history_writer = if o.do_history_file {
            let hdr_name = o.star_history_header_name.trim();
            let hdr_path =
                (!hdr_name.is_empty()).then(|| log_dir.join(hdr_name));
            let hw = IncrementalHistoryWriter::with_header_path(
                log_dir.join(o.star_history_name.trim()),
                hdr_path,
                hist_specs,
            );
            hw.clear_stale_file()?;
            Some(hw)
        } else {
            None
        };
        let existing_index_path = log_dir.join(o.profiles_index_name.trim());
        let profiles_index = if existing_index_path.is_file() {
            if model.model_number <= 1 || stage_logs_dir.is_some() {
                fs::remove_file(&existing_index_path).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to clear stale {}: {e}",
                        existing_index_path.display()
                    ))
                })?;
                ProfilesIndex::default()
            } else {
                crate::mesa_log_reader::read_profiles_index_file(
                    &existing_index_path,
                )?
            }
        } else {
            ProfilesIndex::default()
        };

        let mut out = Self {
            log_dir,
            output_root: output_root.to_path_buf(),
            history_writer,
            profile_specs,
            profiles_index,
            history_interval_override,
            write_profiles_override,
        };
        if model.job.profile_starting_model && write_profs {
            let (extra_hdr, extra_cols) =
                fetch_starting_profile_extras(model)?;
            let ctx = StepReportContext::compute(model);
            out.profiles_index.save_profile_with_ctx_and_extras(
                model,
                &out.profile_specs,
                &ctx,
                &out.log_dir,
                3,
                &extra_hdr,
                &extra_cols,
            )?;
        }
        Ok(out)
    }

    /// Creates a [`StarLogsWriter`] writing `history.data` directly to
    /// `explicit_history_path` (without profiles unless requested).
    pub fn for_explicit_history_file(
        model: &StarModel,
        case_dir: &Path,
        explicit_history_path: PathBuf,
        history_interval_override: Option<usize>,
    ) -> Result<Self, StarError> {
        let o = &model.controls.output;
        let hist_specs = crate::mesa_history::resolve_star_history_columns(
            model,
            Some(case_dir),
        )?;
        let log_dir = explicit_history_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let hdr_name = o.star_history_header_name.trim();
        let hdr_path = (!hdr_name.is_empty()).then(|| log_dir.join(hdr_name));
        let hw = IncrementalHistoryWriter::with_header_path(
            explicit_history_path,
            hdr_path,
            hist_specs,
        );
        hw.clear_stale_file()?;
        Ok(Self {
            output_root: log_dir.clone(),
            log_dir,
            history_writer: Some(hw),
            profile_specs: default_star_profile_specs(),
            profiles_index: ProfilesIndex::default(),
            history_interval_override,
            write_profiles_override: Some(false),
        })
    }

    /// Path to `LOGS/` directory.
    #[must_use]
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }

    fn due_flags(
        &self,
        model: &StarModel,
        is_first_step: bool,
        is_terminating: bool,
    ) -> (bool, bool, bool) {
        let o = &model.controls.output;
        let hist_int =
            self.history_interval_override.unwrap_or(o.history_interval);
        let write_profs = self
            .write_profiles_override
            .unwrap_or(o.write_profiles_flag);
        let mnum = model.model_number;
        let match_prof_model =
            o.profile_model >= 0 && (mnum as isize) == o.profile_model;
        let prof_due = write_profs
            && (is_terminating
                || match_prof_model
                || (o.profile_interval > 0
                    && (is_first_step
                        || mnum.is_multiple_of(o.profile_interval))));
        let hist_due = self.history_writer.is_some()
            && (is_first_step
                || is_terminating
                || prof_due
                || (hist_int > 0 && mnum.is_multiple_of(hist_int)));
        (prof_due, hist_due, match_prof_model)
    }

    /// Returns whether `record_step*` will write a `history.data` row on this step.
    #[must_use]
    pub fn is_history_due(
        &self,
        model: &StarModel,
        is_first_step: bool,
        is_terminating: bool,
    ) -> bool {
        self.due_flags(model, is_first_step, is_terminating).1
    }

    /// Returns whether `record_step*` will emit a `profileN.data` or termination
    /// profile on this step.
    #[must_use]
    pub fn is_profile_due(
        &self,
        model: &StarModel,
        is_first_step: bool,
        is_terminating: bool,
    ) -> bool {
        let (prof_due, _, _) =
            self.due_flags(model, is_first_step, is_terminating);
        let term_prof_due = is_terminating
            && self.write_profiles_override != Some(false)
            && model.job.write_profile_when_terminate;
        prof_due || term_prof_due
    }

    /// Returns whether the stage `history.data` writer still needs its header row.
    #[must_use]
    pub fn needs_history_header(&self) -> bool {
        self.history_writer
            .as_ref()
            .is_some_and(|hw| !hw.header_written())
    }

    /// Records a converged step (`is_first_step` corresponds to
    /// `doing_first_model_of_run`, `is_terminating` corresponds to stage termination).
    pub fn record_step(
        &mut self,
        model: &StarModel,
        is_first_step: bool,
        is_terminating: bool,
    ) -> Result<(), StarError> {
        self.record_step_with_extra(model, is_first_step, is_terminating, &[])
    }

    /// Records a converged step with optional appended history columns
    /// (`append_to_star_history` in `star/private/history.f90:475-477`).
    pub fn record_step_with_extra(
        &mut self,
        model: &StarModel,
        is_first_step: bool,
        is_terminating: bool,
        extra_history_cols: &[(String, f64)],
    ) -> Result<(), StarError> {
        self.record_step_with_extras(
            model,
            is_first_step,
            is_terminating,
            StarStepLogExtras {
                history_columns: extra_history_cols,
                ..StarStepLogExtras::default()
            },
        )
    }

    /// Records a converged step with caller-supplied history and profile extra
    /// headers and columns (`history.f90:353-398, 475-477` and
    /// `profile.f90:590-592, 653-656`).
    pub fn record_step_with_extras(
        &mut self,
        model: &StarModel,
        is_first_step: bool,
        is_terminating: bool,
        extras: StarStepLogExtras<'_>,
    ) -> Result<(), StarError> {
        let o = &model.controls.output;
        let mnum = model.model_number;
        let (prof_due, hist_due, match_prof_model) =
            self.due_flags(model, is_first_step, is_terminating);
        let term_prof_due = is_terminating
            && self.write_profiles_override != Some(false)
            && model.job.write_profile_when_terminate;

        if !prof_due && !hist_due && !term_prof_due {
            return Ok(());
        }

        let ctx = StepReportContext::compute(model);

        // Upstream `finish_step` (`evolve.f90:2318-2336`) saves profiles first,
        // then appends `history.data`.
        if prof_due {
            let priority = if is_terminating {
                3
            } else if match_prof_model
                || is_first_step
                || (o.priority_profile_interval > 0
                    && mnum.is_multiple_of(o.priority_profile_interval))
            {
                2
            } else {
                1
            };
            self.profiles_index.save_profile_with_ctx_and_extras(
                model,
                &self.profile_specs,
                &ctx,
                &self.log_dir,
                priority,
                extras.profile_header_items,
                extras.profile_columns,
            )?;
        }

        if hist_due {
            if let Some(ref mut hw) = self.history_writer {
                hw.write_step_with_ctx_and_extras(
                    model,
                    &ctx,
                    extras.history_header_items,
                    extras.history_columns,
                )?;
            }
        }

        if term_prof_due {
            let fname = model.job.filename_for_profile_when_terminate.trim();
            if fname.is_empty() {
                return Err(StarError::InvalidInput(String::from(
                    "filename_for_profile_when_terminate must be non-empty when write_profile_when_terminate is true",
                )));
            }
            validate_extra_profile_inputs(
                model.nz,
                extras.profile_header_items,
                extras.profile_columns,
            )?;
            let out_path = self.output_root.join(fname);
            if let Some(parent) =
                out_path.parent().filter(|p| !p.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to create {}: {e}",
                        parent.display()
                    ))
                })?;
            }
            let (hdr_t, body_t) = format_star_profile_parts_with_extras_impl(
                model,
                &self.profile_specs,
                &ctx,
                extras.profile_header_items,
                extras.profile_columns,
            );
            let mut full = String::with_capacity(hdr_t.len() + body_t.len());
            full.push_str(&hdr_t);
            full.push_str(&body_t);
            fs::write(&out_path, full).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write {}: {e}",
                    out_path.display()
                ))
            })?;
        }
        Ok(())
    }
}
