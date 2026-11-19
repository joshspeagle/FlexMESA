// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream MESA `26.4.1` `binary_history.data` column parser, evaluator, and
//! incremental flushing file writer (`binary/private/binary_private_def.f90`,
//! `binary/private/binary_history_specs.f90`, `binary/private/binary_history.f90`,
//! `binary/private/run_binary_support.f90`, and
//! `binary/defaults/binary_history_columns.list`).

use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flexmesa_const::{
    MixingType, LSUN, MSUN, PI, RSUN, SECDAY, SECYER, STANDARD_CGRAV,
};
use flexmesa_schema::strip_fortran_comment;
use flexmesa_star::{
    eval_photosphere_info, eval_surf_avg_rotation_info, format_mesa_dbl_width,
    format_mesa_header_str_width, parse_quoted_include_arg,
    push_mesa_6line_header, safe_log10, write_mesa_dbl_width,
    write_mesa_int_width, MesaCellValue, StarError, StarLogsWriter, StarModel,
    MESA_REF_VERSION_NUMBER,
};

use crate::edd_wind::eval_accreted_material_j;
use crate::jdot_tides::{
    eval_k_div_t_convective, eval_k_div_t_radiative, eval_tsync,
};
use crate::system::{BinaryCompanion, BinarySystem};
use crate::system_jdot::eval_star_spin_j;

/// Default column width for `binary_history.data`
/// (`(1pes32.16e3, 1x)`, `(i32, 1x)`, `(a32, 1x)` in `binary/defaults/binary_controls.defaults:119-121`).
pub const BINARY_LOG_COL_WIDTH: usize = 32;

/// Default `binary/defaults/binary_history_columns.list` in MESA `26.4.1`
/// (31 active columns, lines `37-131`).
pub const DEFAULT_BINARY_HISTORY_COLUMNS_TEXT: &str = "\
model_number
age
period_days
binary_separation
v_orb_1
v_orb_2
rl_1
rl_2
rl_relative_overflow_1
rl_relative_overflow_2
star_1_mass
star_2_mass
lg_mtransfer_rate
lg_mstar_dot_1
lg_mstar_dot_2
lg_system_mdot_1
lg_system_mdot_2
lg_wind_mdot_1
lg_wind_mdot_2
fixed_xfer_fraction
eff_xfer_fraction
J_orb
Jdot
jdot_mb
jdot_gr
jdot_ml
jdot_ls
jdot_missing_wind
extra_jdot
donor_index
point_mass_index
";

/// All 74 canonical column names defined in `binary_history_column_names_init`
/// (`binary/private/binary_private_def.f90:118-191`).
const CANONICAL_BINARY_HISTORY_COLUMNS: &[&str] = &[
    "model_number",
    "age",
    "donor_index",
    "period_days",
    "period_hr",
    "period_minutes",
    "lg_separation",
    "binary_separation",
    "eccentricity",
    "star_1_radius",
    "star_2_radius",
    "rl_1",
    "rl_2",
    "rl_overflow_1",
    "rl_overflow_2",
    "rl_relative_overflow_1",
    "rl_relative_overflow_2",
    "P_rot_div_P_orb_1",
    "P_rot_div_P_orb_2",
    "lg_t_sync_1",
    "lg_t_sync_2",
    "star_1_mass",
    "lg_star_1_mass",
    "star_2_mass",
    "lg_star_2_mass",
    "sum_of_masses",
    "mass_ratio",
    "obs_mass_ratio",
    "lg_mtransfer_rate",
    "lg_mstar_dot_1",
    "lg_mstar_dot_2",
    "lg_system_mdot_1",
    "lg_system_mdot_2",
    "lg_wind_mdot_1",
    "lg_wind_mdot_2",
    "star_1_div_star_2_mass",
    "delta_star_1_mass",
    "delta_star_2_mass",
    "fixed_xfer_fraction",
    "eff_xfer_fraction",
    "lg_mdot_edd",
    "mdot_edd_eta",
    "lg_accretion_luminosity",
    "bh_spin",
    "v_orb_1",
    "v_orb_2",
    "lg_F_irr",
    "J_orb",
    "J_spin_1",
    "J_spin_2",
    "J_total",
    "Jdot",
    "jdot_mb",
    "jdot_gr",
    "jdot_ml",
    "jdot_ls",
    "jdot_missing_wind",
    "extra_jdot",
    "accretion_mode",
    "acc_am_div_kep_am",
    "edot",
    "edot_tidal",
    "edot_enhance",
    "point_mass_index",
    "ignore_rlof_flag",
];

/// Validated `binary_history_columns.list` column specification (`#163`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryHistoryColumnSpec {
    /// Canonical upstream column name (`binary_private_def.f90:118-191`).
    pub name: &'static str,
}

impl BinaryHistoryColumnSpec {
    /// Canonical header column name.
    #[must_use]
    pub const fn column_name(self) -> &'static str {
        self.name
    }
}

/// Returns the default 31 `binary_history.data` column specs.
#[must_use]
pub fn default_binary_history_specs() -> Vec<BinaryHistoryColumnSpec> {
    parse_binary_history_columns(DEFAULT_BINARY_HISTORY_COLUMNS_TEXT, None)
        .expect("embedded DEFAULT_BINARY_HISTORY_COLUMNS_TEXT must be valid")
}

/// Parses a `binary_history_columns.list` text buffer (`add_binary_history_columns`
/// in `binary/private/binary_history_specs.f90:37-211`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] on unknown column names or malformed
/// `include` directives, and [`StarError::IoError`] if an included file cannot
/// be read.
pub fn parse_binary_history_columns(
    text: &str,
    base_dir: Option<&Path>,
) -> Result<Vec<BinaryHistoryColumnSpec>, StarError> {
    let mut out = Vec::new();
    parse_binary_history_columns_rec(text, base_dir, 1, &mut out)?;
    if out.is_empty() {
        return Err(StarError::InvalidInput(String::from(
            "binary_history_columns.list produced zero active columns",
        )));
    }
    Ok(out)
}

/// Resolves `sys.controls.binary_history_columns_file` (falling back to
/// `<case_dir>/binary_history_columns.list` or [`DEFAULT_BINARY_HISTORY_COLUMNS_TEXT`]
/// per `binary_history_specs.f90:66-78`).
pub fn resolve_binary_history_columns(
    sys: &BinarySystem,
    case_dir: Option<&Path>,
) -> Result<Vec<BinaryHistoryColumnSpec>, StarError> {
    let raw = sys.controls.binary_history_columns_file.trim();
    if raw.is_empty() {
        if let Some(dir) = case_dir {
            let local = dir.join("binary_history_columns.list");
            if local.is_file() {
                let text = fs::read_to_string(&local).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to read {}: {e}",
                        local.display()
                    ))
                })?;
                return parse_binary_history_columns(&text, Some(dir));
            }
        }
        return Ok(default_binary_history_specs());
    }
    if let Some(p) = resolve_binary_col_file_path(raw, case_dir) {
        let text = fs::read_to_string(&p).map_err(|e| {
            StarError::IoError(format!("failed to read {}: {e}", p.display()))
        })?;
        return parse_binary_history_columns(&text, p.parent().or(case_dir));
    }
    if raw == "binary_history_columns.list" {
        return Ok(default_binary_history_specs());
    }
    Err(StarError::IoError(format!(
        "failed to locate binary_history_columns_file {raw:?}"
    )))
}

fn resolve_binary_col_file_path(
    name: &str,
    base_dir: Option<&Path>,
) -> Option<PathBuf> {
    let p = Path::new(name);
    if p.is_file() {
        return Some(p.to_path_buf());
    }
    if let Some(dir) = base_dir {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
        for anc in dir.ancestors() {
            let d = anc.join("binary").join("defaults").join(name);
            if d.is_file() {
                return Some(d);
            }
        }
    }
    None
}

fn parse_quoted_arg(rest: &str) -> Option<&str> {
    parse_quoted_include_arg(rest)
}

fn push_unique_binary_spec(
    out: &mut Vec<BinaryHistoryColumnSpec>,
    spec: BinaryHistoryColumnSpec,
) {
    if !out.contains(&spec) {
        out.push(spec);
    }
}

fn parse_binary_history_columns_rec(
    text: &str,
    base_dir: Option<&Path>,
    level: usize,
    out: &mut Vec<BinaryHistoryColumnSpec>,
) -> Result<(), StarError> {
    if level > 20 {
        return Err(StarError::InvalidInput(String::from(
            "too many levels of include nesting in binary_history_columns.list",
        )));
    }
    for (line_idx, raw_line) in text.lines().enumerate() {
        let stripped = strip_fortran_comment(raw_line).0.trim();
        if stripped.is_empty() {
            continue;
        }
        if let Some(rest) = stripped.strip_prefix("include") {
            if rest.is_empty()
                || rest.starts_with(char::is_whitespace)
                || rest.starts_with(['\'', '"'])
            {
                let inc = parse_quoted_arg(rest).ok_or_else(|| {
                    StarError::InvalidInput(format!(
                        "binary_history_columns.list line {}: expected quoted string after include",
                        line_idx + 1
                    ))
                })?;
                let inc_trim = inc.trim();
                if inc_trim.is_empty()
                    || inc_trim == "binary_history_columns.list"
                {
                    parse_binary_history_columns_rec(
                        DEFAULT_BINARY_HISTORY_COLUMNS_TEXT,
                        base_dir,
                        level + 1,
                        out,
                    )?;
                } else if let Some(inc_path) =
                    resolve_binary_col_file_path(inc_trim, base_dir)
                {
                    let inc_text =
                        fs::read_to_string(&inc_path).map_err(|e| {
                            StarError::IoError(format!(
                                "failed to read included {}: {e}",
                                inc_path.display()
                            ))
                        })?;
                    parse_binary_history_columns_rec(
                        &inc_text,
                        inc_path.parent().or(base_dir),
                        level + 1,
                        out,
                    )?;
                } else {
                    return Err(StarError::IoError(format!(
                        "failed to open included binary_history_columns file {inc_trim:?}"
                    )));
                }
                continue;
            }
        }
        let toks: Vec<&str> = stripped.split_whitespace().collect();
        if toks.len() != 1 {
            return Err(StarError::InvalidInput(format!(
                "binary_history_columns.list line {}: unsupported directive {stripped:?}",
                line_idx + 1
            )));
        }
        let Some(&canon) = CANONICAL_BINARY_HISTORY_COLUMNS
            .iter()
            .find(|&&c| c.eq_ignore_ascii_case(toks[0]))
        else {
            return Err(StarError::InvalidInput(format!(
                "binary_history_columns.list line {}: unknown column {:?}",
                line_idx + 1,
                toks[0]
            )));
        };
        push_unique_binary_spec(out, BinaryHistoryColumnSpec { name: canon });
    }
    Ok(())
}

#[inline]
fn map_donor_acc_to_stars<T: Copy>(donor_i: usize, arr: [T; 2]) -> [T; 2] {
    if donor_i == 1 {
        arr
    } else {
        [arr[1], arr[0]]
    }
}

fn eval_upstream_k_div_t(
    star: &StarModel,
    presc: &str,
    r_phot_cm: f64,
    l_phot_lsun: f64,
    omega_surf: f64,
    m_comp_g: f64,
    separation_cm: f64,
    period_sec: f64,
    tidal_reduction: f64,
) -> f64 {
    if presc.eq_ignore_ascii_case("Hut_rad") {
        eval_k_div_t_radiative(star.mstar, m_comp_g, r_phot_cm, separation_cm)
    } else if presc.eq_ignore_ascii_case("Hut_conv") && star.nz > 0 {
        // `k_div_T` in `binary/private/binary_tides.f90:417-438`.
        let mut m_env = star.m[0] / MSUN;
        let mut r_env = star.r[0] / RSUN;
        let rho_surf = star.rho[0];
        for k in 0..star.nz {
            if star.mixing_type[k] != MixingType::ConvectiveMixing
                && star.rho[k] > 1.0e5 * rho_surf
            {
                r_env = (r_phot_cm - star.r[k]) / RSUN;
                m_env = (star.m[0] - star.m[k]) / MSUN;
                break;
            }
        }
        if m_env > 0.0 && r_env > 0.0 && l_phot_lsun > 0.0 {
            eval_k_div_t_convective(
                star.mstar / MSUN,
                m_env,
                r_phot_cm / RSUN,
                r_env,
                l_phot_lsun,
                period_sec,
                omega_surf,
                tidal_reduction,
            )
        } else {
            0.0
        }
    } else {
        0.0
    }
}

fn eval_star_tsync_sec(
    sys: &BinarySystem,
    star_idx: usize,
    m_by_star: [f64; 2],
) -> f64 {
    if !sys.controls.do_tidal_sync || sys.point_mass_i == star_idx {
        return 0.0;
    }
    let Some(star) = sys.star_ref(star_idx) else {
        return 0.0;
    };
    if !star.controls.rotation_flag
        || star.nz == 0
        || sys.separation <= 0.0
        || sys.period <= 0.0
    {
        return 0.0;
    }
    let (sync_type, f_tid, m_comp) = if star_idx == 1 {
        (
            sys.controls.sync_type_1.as_str(),
            sys.controls.ftid_1,
            m_by_star[1],
        )
    } else {
        (
            sys.controls.sync_type_2.as_str(),
            sys.controls.ftid_2,
            m_by_star[0],
        )
    };
    // `binary_tides.f90:133-134, 142, 159`: `None` and `Instantaneous` leave `t_sync = 0`.
    if f_tid <= 0.0
        || sync_type.eq_ignore_ascii_case("None")
        || sync_type.eq_ignore_ascii_case("Instantaneous")
    {
        return 0.0;
    }
    if sync_type.eq_ignore_ascii_case("Orb_period") {
        return sys.period / f_tid;
    }
    let phot = eval_photosphere_info(star);
    let r_phot_cm = phot.r_rsun * RSUN;
    if r_phot_cm <= 0.0 || star.mstar <= 0.0 {
        return 0.0;
    }
    let omega_surf = eval_surf_avg_rotation_info(star).omega;
    let k_div_t = eval_upstream_k_div_t(
        star,
        sync_type,
        r_phot_cm,
        phot.l_lsun,
        omega_surf,
        m_comp,
        sys.separation,
        sys.period,
        sys.controls.tidal_reduction,
    );
    if k_div_t <= 0.0 {
        return 0.0;
    }
    // `binary_tides.f90:355-359`: `rGyr_squared = moment_of_inertia / (m * r_phot^2)`.
    let moi: f64 = star
        .aux
        .i_rot
        .iter()
        .take(star.nz)
        .zip(star.dm_bar.iter().take(star.nz))
        .map(|(&ir, &dm)| ir * dm)
        .sum();
    let r_gyr_sq = moi / (star.mstar * r_phot_cm * r_phot_cm);
    if r_gyr_sq <= 0.0 {
        return 0.0;
    }
    let ts = eval_tsync(
        k_div_t,
        m_comp / star.mstar,
        r_gyr_sq,
        r_phot_cm,
        sys.separation,
        f_tid,
    );
    if ts.is_finite() && ts > 0.0 {
        ts
    } else {
        0.0
    }
}

/// Evaluates all requested [`BinaryHistoryColumnSpec`] columns on `sys`
/// matching `binary_history_getval` (`binary/private/binary_history.f90:507-749`).
#[must_use]
pub fn evaluate_binary_history_row(
    sys: &BinarySystem,
    specs: &[BinaryHistoryColumnSpec],
) -> Vec<(String, MesaCellValue)> {
    let m_by_star = sys.star_masses_g();
    let r_d_cm = eval_photosphere_info(&sys.donor).r_rsun * RSUN;
    let r_a_cm = match &sys.companion {
        BinaryCompanion::PointMass { .. } => 0.0,
        BinaryCompanion::Star(s2) => eval_photosphere_info(s2).r_rsun * RSUN,
    };
    let r_by_star = map_donor_acc_to_stars(sys.donor_i, [r_d_cm, r_a_cm]);
    let rl_by_star = map_donor_acc_to_stars(sys.donor_i, sys.rl);
    // `binary_evolve.f90:367-370`: periastron relative overflow from photospheric `b% r(i)`.
    let gap_by_star = [0_usize, 1_usize].map(|i| {
        let denom = rl_by_star[i] * (1.0 - sys.eccentricity);
        if denom > 0.0 {
            (r_by_star[i] - denom) / denom
        } else {
            0.0
        }
    });
    let comp_mdot_by_star =
        map_donor_acc_to_stars(sys.donor_i, sys.last_mdot.component_mdot);
    let sys_tr_by_star =
        map_donor_acc_to_stars(sys.donor_i, sys.last_mdot.mdot_system_transfer);
    let sys_wind_by_star =
        map_donor_acc_to_stars(sys.donor_i, sys.last_mdot.mdot_system_wind);

    let j_spin_1 = sys.star_ref(1).map_or(0.0, eval_star_spin_j);
    let j_spin_2 = sys.star_ref(2).map_or(0.0, eval_star_spin_j);
    let m_tot = (m_by_star[0] + m_by_star[1]).max(1.0e-99);
    let fixed_xfer = 1.0
        - sys.mass_transfer_alpha
        - sys.mass_transfer_beta
        - sys.mass_transfer_delta;
    // `binary_history.f90` writes the stored `b% edot`, `b% edot_tidal` and
    // `b% edot_enhance`, so report the terms the orbital update applied.
    // While the engine still carries the gravitational-wave term (`#175`)
    // and per-star `Ftid` (`#186`), both owed under `#160`, these columns
    // show that departure instead of hiding it.
    let (edot_tot, edot_tidal, edot_enhance) = (
        sys.last_edot.edot_total,
        sys.last_edot.edot_tidal,
        sys.last_edot.edot_enhance,
    );

    // `binary_mdot.f90:562-565, 964-982` and `binary_history.f90:671-675`.
    let (accretion_mode, acc_am_div_kep_am) = if sys.controls.do_j_accretion
        && sys.point_mass_i == 0
        && sys.model_number > 0
        && sys.separation > 0.0
        && r_a_cm > 0.0
    {
        let m_acc = sys.accretor_mass_g();
        let (j_acc, mode) = eval_accreted_material_j(
            sys.donor.mstar,
            m_acc,
            r_a_cm,
            sys.separation,
        );
        let j_kep = (STANDARD_CGRAV * m_acc * r_a_cm).max(0.0).sqrt();
        let ratio = if j_kep > 0.0 { j_acc / j_kep } else { 0.0 };
        (i64::from(mode), ratio)
    } else {
        (0_i64, 0.0)
    };

    specs
        .iter()
        .map(|sp| {
            let val = match sp.name {
                "model_number" => MesaCellValue::Int(
                    i64::try_from(sys.model_number).unwrap_or(i64::MAX),
                ),
                "age" => MesaCellValue::Float(sys.binary_age_yr),
                "donor_index" => MesaCellValue::Int(
                    i64::try_from(sys.donor_i).unwrap_or(1),
                ),
                "period_days" => MesaCellValue::Float(sys.period / SECDAY),
                "period_hr" => MesaCellValue::Float(sys.period / 3600.0),
                "period_minutes" => MesaCellValue::Float(sys.period / 60.0),
                // Reproduces `binary_history.f90:538` verbatim (`safe_log10(b% separation)` in cm).
                // Reproduces `binary_history.f90:538` verbatim (no division by `Rsun`; `#183` under `#169`).
                "lg_separation" => MesaCellValue::Float(safe_log10(sys.separation)),
                "binary_separation" => MesaCellValue::Float(sys.separation / RSUN),
                "eccentricity" => MesaCellValue::Float(sys.eccentricity),
                "star_1_radius" => MesaCellValue::Float(r_by_star[0] / RSUN),
                "star_2_radius" => MesaCellValue::Float(r_by_star[1] / RSUN),
                "rl_1" => MesaCellValue::Float(rl_by_star[0] / RSUN),
                "rl_2" => MesaCellValue::Float(rl_by_star[1] / RSUN),
                "rl_overflow_1" => {
                    MesaCellValue::Float((r_by_star[0] - rl_by_star[0]) / RSUN)
                }
                "rl_overflow_2" => {
                    MesaCellValue::Float((r_by_star[1] - rl_by_star[1]) / RSUN)
                }
                "rl_relative_overflow_1" => MesaCellValue::Float(gap_by_star[0]),
                "rl_relative_overflow_2" => MesaCellValue::Float(gap_by_star[1]),
                "P_rot_div_P_orb_1" => {
                    let v = sys.star_ref(1).map_or(0.0, |s| {
                        let om = eval_surf_avg_rotation_info(s).omega;
                        if om != 0.0 && sys.period > 0.0 {
                            2.0 * PI / om / sys.period
                        } else {
                            0.0
                        }
                    });
                    MesaCellValue::Float(v)
                }
                "P_rot_div_P_orb_2" => {
                    let v = sys.star_ref(2).map_or(0.0, |s| {
                        let om = eval_surf_avg_rotation_info(s).omega;
                        if om != 0.0 && sys.period > 0.0 {
                            2.0 * PI / om / sys.period
                        } else {
                            0.0
                        }
                    });
                    MesaCellValue::Float(v)
                }
                "lg_t_sync_1" => MesaCellValue::Float(safe_log10(
                    eval_star_tsync_sec(sys, 1, m_by_star).abs() / SECYER,
                )),
                "lg_t_sync_2" => MesaCellValue::Float(safe_log10(
                    eval_star_tsync_sec(sys, 2, m_by_star).abs() / SECYER,
                )),
                "star_1_mass" => MesaCellValue::Float(m_by_star[0] / MSUN),
                "lg_star_1_mass" => {
                    MesaCellValue::Float(safe_log10(m_by_star[0] / MSUN))
                }
                "star_2_mass" => MesaCellValue::Float(m_by_star[1] / MSUN),
                "lg_star_2_mass" => {
                    MesaCellValue::Float(safe_log10(m_by_star[1] / MSUN))
                }
                "sum_of_masses" => MesaCellValue::Float(m_tot / MSUN),
                "mass_ratio" => {
                    MesaCellValue::Float(m_by_star[1] / m_by_star[0].max(1.0e-99))
                }
                "obs_mass_ratio" => MesaCellValue::Float(
                    (m_by_star[1] / m_by_star[0].max(1.0e-99))
                        .min(m_by_star[0] / m_by_star[1].max(1.0e-99)),
                ),
                "lg_mtransfer_rate" => MesaCellValue::Float(safe_log10(
                    sys.step_mtransfer_rate.abs() / MSUN * SECYER,
                )),
                "lg_mstar_dot_1" => MesaCellValue::Float(safe_log10(
                    comp_mdot_by_star[0].abs() / MSUN * SECYER,
                )),
                "lg_mstar_dot_2" => MesaCellValue::Float(safe_log10(
                    comp_mdot_by_star[1].abs() / MSUN * SECYER,
                )),
                "lg_system_mdot_1" => MesaCellValue::Float(safe_log10(
                    sys_tr_by_star[0].abs() / MSUN * SECYER,
                )),
                "lg_system_mdot_2" => MesaCellValue::Float(safe_log10(
                    sys_tr_by_star[1].abs() / MSUN * SECYER,
                )),
                "lg_wind_mdot_1" => MesaCellValue::Float(safe_log10(
                    sys_wind_by_star[0].abs() / MSUN * SECYER,
                )),
                "lg_wind_mdot_2" => MesaCellValue::Float(safe_log10(
                    sys_wind_by_star[1].abs() / MSUN * SECYER,
                )),
                "star_1_div_star_2_mass" => {
                    MesaCellValue::Float(m_by_star[0] / m_by_star[1].max(1.0e-99))
                }
                // Reproduces `binary_history.f90:610-612` verbatim (`b% m(i) - initial_mass(i)`,
                // grams minus solar masses; `#183` under `#169`).
                "delta_star_1_mass" => {
                    MesaCellValue::Float(m_by_star[0] - sys.initial_mass_msun[0])
                }
                "delta_star_2_mass" => {
                    MesaCellValue::Float(m_by_star[1] - sys.initial_mass_msun[1])
                }
                "lg_F_irr" => MesaCellValue::Float(safe_log10(
                    sys.donor.controls.irradiation_flux,
                )),
                "fixed_xfer_fraction" => MesaCellValue::Float(fixed_xfer),
                "eff_xfer_fraction" => {
                    let dm_d = sys.last_mdot.component_mdot[0];
                    let dm_a = sys.last_mdot.component_mdot[1];
                    MesaCellValue::Float(if dm_d == 0.0 {
                        1.0
                    } else {
                        (-dm_a) / dm_d
                    })
                }
                "lg_mdot_edd" => MesaCellValue::Float(
                    if sys.controls.limit_retention_by_mdot_edd {
                        safe_log10(sys.last_mdot.mdot_edd / MSUN * SECYER)
                    } else {
                        safe_log10(0.0)
                    },
                ),
                "mdot_edd_eta" => MesaCellValue::Float(
                    if sys.controls.limit_retention_by_mdot_edd {
                        sys.last_mdot.mdot_edd_eta
                    } else {
                        0.0
                    },
                ),
                "lg_accretion_luminosity" => MesaCellValue::Float(safe_log10(
                    sys.last_mdot.accretion_luminosity / LSUN,
                )),
                "bh_spin" => {
                    let spin = match &sys.companion {
                        BinaryCompanion::PointMass {
                            mass_g,
                            m_bh_zero_eq_g,
                            ..
                        } if sys.point_mass_i != 0 && *m_bh_zero_eq_g > 0.0 => {
                            let denom =
                                (*mass_g).min(6.0_f64.sqrt() * *m_bh_zero_eq_g);
                            let ratio = *m_bh_zero_eq_g / denom.max(1.0e-99);
                            (2.0_f64 / 3.0).sqrt()
                                * ratio
                                * (4.0 - (18.0 * ratio * ratio - 2.0).max(0.0).sqrt())
                        }
                        _ => 0.0,
                    };
                    MesaCellValue::Float(spin)
                }
                "v_orb_1" => MesaCellValue::Float(if sys.period > 0.0 {
                    2.0 * PI * m_by_star[1] / m_tot * sys.separation
                        / sys.period
                        / 1.0e5
                } else {
                    0.0
                }),
                "v_orb_2" => MesaCellValue::Float(if sys.period > 0.0 {
                    2.0 * PI * m_by_star[0] / m_tot * sys.separation
                        / sys.period
                        / 1.0e5
                } else {
                    0.0
                }),
                "J_orb" => MesaCellValue::Float(sys.angular_momentum_j),
                "J_spin_1" => MesaCellValue::Float(j_spin_1),
                "J_spin_2" => MesaCellValue::Float(j_spin_2),
                "J_total" => {
                    MesaCellValue::Float(sys.angular_momentum_j + j_spin_1 + j_spin_2)
                }
                "Jdot" => MesaCellValue::Float(sys.jdot),
                "jdot_mb" => MesaCellValue::Float(sys.last_jdot.jdot_mb),
                "jdot_gr" => MesaCellValue::Float(sys.last_jdot.jdot_gr),
                "jdot_ml" => MesaCellValue::Float(sys.last_jdot.jdot_ml),
                "jdot_ls" => MesaCellValue::Float(sys.last_jdot.jdot_ls),
                "jdot_missing_wind" => {
                    MesaCellValue::Float(sys.last_jdot.jdot_missing_wind)
                }
                "extra_jdot" => MesaCellValue::Float(sys.last_jdot.jdot_extra),
                "accretion_mode" => MesaCellValue::Int(accretion_mode),
                "acc_am_div_kep_am" => MesaCellValue::Float(acc_am_div_kep_am),
                "edot" => MesaCellValue::Float(edot_tot),
                "edot_tidal" => MesaCellValue::Float(edot_tidal),
                "edot_enhance" => MesaCellValue::Float(edot_enhance),
                "point_mass_index" => MesaCellValue::Int(
                    i64::try_from(sys.point_mass_i).unwrap_or(0),
                ),
                "ignore_rlof_flag" => MesaCellValue::Int(i64::from(sys.ignore_rlof_flag)),
                other => unreachable!("validated binary history column {other:?} missing evaluator arm"),
            };
            (sp.name.to_owned(), val)
        })
        .collect()
}

/// Converts evaluated binary history columns to `(name, f64)` pairs matching
/// `data_for_binary_history_columns` (`binary/private/binary_history.f90:56-90`)
/// for `append_to_star_history = .true.`.
#[must_use]
pub fn binary_history_row_as_f64_pairs(
    row: &[(String, MesaCellValue)],
) -> Vec<(String, f64)> {
    row.iter()
        .map(|(name, v)| {
            let fv = match *v {
                MesaCellValue::Int(i) => i as f64,
                MesaCellValue::Float(x) => x,
            };
            (name.clone(), fv)
        })
        .collect()
}

/// Formats the 6-line MESA `26.4.1` `binary_history.data` header at width 32
/// (`do_binary_history_info` in `binary/private/binary_history.f90:291-325` and
/// `binary/defaults/binary_controls.defaults:119-121`).
#[must_use]
pub fn format_binary_history_file_header(
    sys: &BinarySystem,
    specs: &[BinaryHistoryColumnSpec],
) -> String {
    format_binary_history_file_header_with_extra(sys, specs, &[])
}

fn format_binary_history_file_header_with_extra(
    sys: &BinarySystem,
    specs: &[BinaryHistoryColumnSpec],
    extra_names: &[&str],
) -> String {
    let w = BINARY_LOG_COL_WIDTH;
    let hdr_items: [(&str, String); 9] = [
        (
            "version_number",
            format_mesa_header_str_width(MESA_REF_VERSION_NUMBER, w),
        ),
        (
            "initial_don_mass",
            format_mesa_dbl_width(sys.initial_mass_msun[0], w),
        ),
        (
            "initial_acc_mass",
            format_mesa_dbl_width(sys.initial_mass_msun[1], w),
        ),
        (
            "initial_period_days",
            format_mesa_dbl_width(sys.initial_binary_period_sec / SECDAY, w),
        ),
        ("compiler", format_mesa_header_str_width("rustc", w)),
        ("build", format_mesa_header_str_width("flexmesa", w)),
        (
            "MESA_SDK_version",
            format_mesa_header_str_width("flexmesa-0.1", w),
        ),
        (
            "math_backend",
            format_mesa_header_str_width(
                flexmesa_math::numerics_mode().as_str(),
                w,
            ),
        ),
        ("date", format_mesa_header_str_width("20260401", w)),
    ];
    let mut col_names: Vec<&str> =
        specs.iter().map(|sp| sp.column_name()).collect();
    col_names.extend_from_slice(extra_names);
    let mut out = String::with_capacity(512 + col_names.len() * 68);
    push_mesa_6line_header(&mut out, &hdr_items, &col_names, w);
    out
}

/// Formats one evaluated `binary_history.data` row line at width 32
/// (`binary/private/binary_history.f90:409-451`).
#[must_use]
pub fn format_binary_history_data_line(
    row: &[(String, MesaCellValue)],
) -> String {
    let w = BINARY_LOG_COL_WIDTH;
    let mut line = String::with_capacity(row.len() * (w + 1) + 1);
    for (_, val) in row {
        match *val {
            MesaCellValue::Int(i) => write_mesa_int_width(&mut line, i, w),
            MesaCellValue::Float(x) => write_mesa_dbl_width(&mut line, x, w),
        }
    }
    let _ = writeln!(line);
    line
}

/// Incremental `binary_history.data` writer that writes the 6-line MESA header
/// on the first recorded step and flushes every row immediately to disk (`#163`).
pub struct IncrementalBinaryHistoryWriter {
    path: PathBuf,
    specs: Vec<BinaryHistoryColumnSpec>,
    writer: Option<BufWriter<File>>,
    rows_written: usize,
    last_model_number: Option<usize>,
}

impl IncrementalBinaryHistoryWriter {
    /// Creates an incremental binary history writer targeting `path`.
    #[must_use]
    pub fn new(path: PathBuf, specs: Vec<BinaryHistoryColumnSpec>) -> Self {
        Self {
            path,
            specs,
            writer: None,
            rows_written: 0,
            last_model_number: None,
        }
    }

    /// Removes any pre-existing file at `self.path` before step 1.
    pub fn clear_stale_file(&self) -> Result<(), StarError> {
        if self.path.is_file() {
            fs::remove_file(&self.path).map_err(|e| {
                StarError::IoError(format!(
                    "failed to clear stale {}: {e}",
                    self.path.display()
                ))
            })?;
        }
        Ok(())
    }

    /// Output file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of rows flushed to disk so far.
    #[must_use]
    pub const fn rows_written(&self) -> usize {
        self.rows_written
    }

    /// Evaluates `sys`, writes the header on the first row, appends the row, and
    /// flushes immediately (`write_binary_history_info` in `binary_history.f90:92-98`).
    pub fn write_step(&mut self, sys: &BinarySystem) -> Result<(), StarError> {
        if self.last_model_number == Some(sys.model_number) {
            return Ok(());
        }
        let row = evaluate_binary_history_row(sys, &self.specs);
        self.write_evaluated_row_with_extra(sys, row, &[])
    }

    fn write_evaluated_row_with_extra(
        &mut self,
        sys: &BinarySystem,
        mut row: Vec<(String, MesaCellValue)>,
        extra_cols: &[(String, f64)],
    ) -> Result<(), StarError> {
        if self.last_model_number == Some(sys.model_number) {
            return Ok(());
        }
        if self.writer.is_none() {
            if let Some(parent) =
                self.path.parent().filter(|p| !p.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to create binary log directory {}: {e}",
                        parent.display()
                    ))
                })?;
            }
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.path)
                .map_err(|e| {
                    StarError::IoError(format!(
                        "failed to open {}: {e}",
                        self.path.display()
                    ))
                })?;
            let mut bw = BufWriter::new(f);
            let extra_names: Vec<&str> =
                extra_cols.iter().map(|(k, _)| k.as_str()).collect();
            let hdr = format_binary_history_file_header_with_extra(
                sys,
                &self.specs,
                &extra_names,
            );
            bw.write_all(hdr.as_bytes()).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write header {}: {e}",
                    self.path.display()
                ))
            })?;
            self.writer = Some(bw);
        }

        for (name, val) in extra_cols {
            row.push((name.clone(), MesaCellValue::Float(*val)));
        }
        let line = format_binary_history_data_line(&row);
        if let Some(ref mut bw) = self.writer {
            bw.write_all(line.as_bytes()).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write row {}: {e}",
                    self.path.display()
                ))
            })?;
            bw.flush().map_err(|e| {
                StarError::IoError(format!(
                    "failed to flush {}: {e}",
                    self.path.display()
                ))
            })?;
        }
        self.rows_written += 1;
        self.last_model_number = Some(sys.model_number);
        Ok(())
    }
}

/// Coupled binary + component-star log coordinator (`run_binary_support.f90:662-752`).
pub struct BinaryLogsWriter {
    binary_specs: Vec<BinaryHistoryColumnSpec>,
    binary_writer: IncrementalBinaryHistoryWriter,
    star_writers: [Option<StarLogsWriter>; 2],
    history_interval_override: Option<usize>,
    append_to_star_history: bool,
}

impl BinaryLogsWriter {
    /// Initializes a [`BinaryLogsWriter`] rooted at `output_root` (writing
    /// `<output_root>/<log_directory>/<history_name>` plus component `LOGS1/` /
    /// `LOGS2/` files when `write_star_logs` is `true`).
    pub fn for_run(
        sys: &BinarySystem,
        case_dir: &Path,
        output_root: &Path,
        explicit_binary_history_file: Option<&Path>,
        history_interval_override: Option<usize>,
        write_star_logs: bool,
    ) -> Result<Self, StarError> {
        let binary_specs = resolve_binary_history_columns(sys, Some(case_dir))?;
        let bin_path = if let Some(exp) = explicit_binary_history_file {
            exp.to_path_buf()
        } else {
            output_root
                .join(sys.controls.log_directory.trim())
                .join(sys.controls.history_name.trim())
        };
        let binary_writer =
            IncrementalBinaryHistoryWriter::new(bin_path, binary_specs.clone());
        binary_writer.clear_stale_file()?;

        let mut star_writers = [None, None];
        if write_star_logs {
            for idx in 1..=2_usize {
                if let Some(star) = sys.star_ref(idx) {
                    star_writers[idx - 1] = Some(StarLogsWriter::for_stage(
                        star,
                        case_dir,
                        output_root,
                        history_interval_override,
                        None,
                    )?);
                }
            }
            if let (Some(w0), Some(w1)) = (&star_writers[0], &star_writers[1]) {
                if w0.log_dir() == w1.log_dir() {
                    return Err(StarError::InvalidInput(format!(
                        "two-star binary component log_directory paths collide at {}; set distinct log_directory values (e.g. LOGS1 and LOGS2)",
                        w0.log_dir().display()
                    )));
                }
            }
        }
        Ok(Self {
            binary_specs,
            binary_writer,
            star_writers,
            history_interval_override,
            append_to_star_history: sys.controls.append_to_star_history,
        })
    }

    /// Path to `binary_history.data`.
    #[must_use]
    pub fn binary_history_path(&self) -> &Path {
        self.binary_writer.path()
    }

    /// Records a converged coupled binary step (`run_binary_support.f90:662-752`).
    pub fn record_step(
        &mut self,
        sys: &BinarySystem,
        is_first_step: bool,
        is_terminating: bool,
    ) -> Result<(), StarError> {
        self.record_step_with_extra(
            sys,
            is_first_step,
            is_terminating,
            &[],
            &[Vec::new(), Vec::new()],
        )
    }

    /// Returns whether `binary_history.data` or component `LOGS*` will consume
    /// extra history columns on the current step (`is_binary_due`, `is_star_due`).
    #[must_use]
    pub fn extra_history_due_flags(
        &self,
        sys: &BinarySystem,
        is_first_step: bool,
        is_terminating: bool,
    ) -> (bool, [bool; 2]) {
        let hist_int = self
            .history_interval_override
            .unwrap_or(sys.controls.history_interval);
        let bin_due = is_terminating
            || (hist_int > 0 && sys.model_number.is_multiple_of(hist_int));
        let star_due = [
            match (self.star_writers[0].as_ref(), sys.star_ref(1)) {
                (Some(sw), Some(star)) => {
                    sw.is_history_due(star, is_first_step, is_terminating)
                }
                _ => false,
            },
            match (self.star_writers[1].as_ref(), sys.star_ref(2)) {
                (Some(sw), Some(star)) => {
                    sw.is_history_due(star, is_first_step, is_terminating)
                }
                _ => false,
            },
        ];
        (bin_due, star_due)
    }

    /// Evaluates and writes binary and component logs for the current step with
    /// caller-supplied extra binary and component history columns (`#148`).
    pub fn record_step_with_extra(
        &mut self,
        sys: &BinarySystem,
        is_first_step: bool,
        is_terminating: bool,
        extra_bin_cols: &[(String, f64)],
        extra_star_cols: &[Vec<(String, f64)>; 2],
    ) -> Result<(), StarError> {
        let (bin_due, star_due) =
            self.extra_history_due_flags(sys, is_first_step, is_terminating);
        let any_star_hist_due = star_due[0] || star_due[1];

        let need_extra = self.append_to_star_history && any_star_hist_due;
        let evaluated = (bin_due || need_extra)
            .then(|| evaluate_binary_history_row(sys, &self.binary_specs));

        let appended_bin_cols = if need_extra {
            evaluated
                .as_deref()
                .map_or_else(Vec::new, binary_history_row_as_f64_pairs)
        } else {
            Vec::new()
        };

        if bin_due {
            if let Some(row) = evaluated {
                self.binary_writer.write_evaluated_row_with_extra(
                    sys,
                    row,
                    extra_bin_cols,
                )?;
            }
        }

        for idx in 1..=2_usize {
            if !star_due[idx - 1] {
                continue;
            }
            if let (Some(sw), Some(star)) =
                (self.star_writers[idx - 1].as_mut(), sys.star_ref(idx))
            {
                let mut combined_star_cols = Vec::with_capacity(
                    appended_bin_cols.len() + extra_star_cols[idx - 1].len(),
                );
                combined_star_cols.extend_from_slice(&appended_bin_cols);
                combined_star_cols.extend_from_slice(&extra_star_cols[idx - 1]);
                sw.record_step_with_extra(
                    star,
                    is_first_step,
                    is_terminating,
                    &combined_star_cols,
                )?;
            }
        }
        Ok(())
    }
}
