// SPDX-License-Identifier: GPL-3.0-or-later
//! Replay of the cell EOS call (`EOS` records, `#552`).
//!
//! The overlay records, after the `get_eos` call of `do_eos_for_cell`
//! (`star/private/micro.f90:357-361`), that call's inputs (`chem_id`, `xa`,
//! `Rho`, `logRho`, `T`, `logT`) and results (`res`, `d_dlnd`, `d_dlnT`,
//! `d_dxa`, `ierr`). Before the first record of a run it writes the EOS
//! handle's controls (`EoS_General_Info`, `eos/public/eos_def.f90`) as one
//! `CONFIG EOS` line, from which [`eos_config_from_config`] builds the
//! [`EosConfig`] of the replay. [`replay_eos`] then calls the EOS entry
//! point `flexmesa-star`'s `do_eos` uses
//! ([`flexmesa_eos::eos_dt_get_zones_with_logs`]) on the captured inputs and
//! compares every result bit for bit.

use std::collections::{BTreeMap, BTreeSet};

use flexmesa_eos::{
    eos_dt_get_zones_with_logs, EosConfig, HelmTable, SkyeSolidMixingRule,
    NUM_EOS_BASIC_RESULTS, NUM_EOS_D_DXA_RESULTS,
};

use crate::compare::CallDiff;
use crate::record::{FormatError, Record, Value};
use crate::turb::Replayed;

/// `SCHEMA EOS` field list of `replay_patch.py` (`EOS_SCHEMA`).
pub const EOS_SCHEMA: &str = "site:i id:i model:i solver_iter:i k:i \
    chem_id:s xa:v Rho:r logRho:r T:r logT:r out_res:v out_d_dlnd:v \
    out_d_dlnT:v out_d_dxa:v out_ierr:i";

/// Names of `res(1:26)` (`eos/public/eos_def.f90`, `i_lnPgas` to
/// `i_frac_ideal`), for reports.
pub const EOS_RESULT_NAMES: [&str; NUM_EOS_BASIC_RESULTS] = [
    "lnPgas",
    "lnE",
    "lnS",
    "mu",
    "lnfree_e",
    "eta",
    "grad_ad",
    "chiRho",
    "chiT",
    "Cp",
    "Cv",
    "dE_dRho",
    "dS_dT",
    "dS_dRho",
    "gamma1",
    "gamma3",
    "phase",
    "latent_ddlnT",
    "latent_ddlnRho",
    "frac_HELM",
    "frac_OPAL_SCVH",
    "frac_FreeEOS",
    "frac_PC",
    "frac_Skye",
    "frac_CMS",
    "frac_ideal",
];

/// Lower `logRho` limit below which upstream `get_eos` returns `ierr = -1`
/// without evaluating the EOS (`star/private/eos_support.f90:70-76`).
const LOG_RHO_FLOOR: f64 = -25.0;

/// Scalar controls of the `CONFIG EOS` line (`EOS_CONFIG_FIELDS` of
/// `replay_patch.py`), in its order.
const SCALAR_CONTROLS: &[&str] = &[
    "Z_all_HELM",
    "logT_all_HELM",
    "logT_low_all_HELM",
    "coulomb_temp_cut_HELM",
    "coulomb_den_cut_HELM",
    "use_OPAL_SCVH",
    "logT_low_all_SCVH",
    "logT_all_OPAL",
    "logRho1_OPAL_SCVH_limit",
    "logRho2_OPAL_SCVH_limit",
    "logRho_min_OPAL_SCVH_limit",
    "logQ_max_OPAL_SCVH",
    "logQ_min_OPAL_SCVH",
    "Z_all_OPAL",
    "use_FreeEOS",
    "logQ_max_FreeEOS_hi",
    "logQ_max_FreeEOS_lo",
    "logQ_min_FreeEOS_hi",
    "logQ_min_FreeEOS_lo",
    "logRho_min_FreeEOS_hi",
    "logRho_min_FreeEOS_lo",
    "logRho_max_FreeEOS_hi",
    "logRho_max_FreeEOS_lo",
    "logT_min_FreeEOS_hi",
    "logT_min_FreeEOS_lo",
    "logT_max_FreeEOS_hi",
    "logT_max_FreeEOS_lo",
    "logQ_cut_FreeEOS_lo_Z_max",
    "logQ_cut_lo_Z_FreeEOS_hi",
    "logQ_cut_lo_Z_FreeEOS_lo",
    "logQ_cut_hi_Z_FreeEOS_hi",
    "logQ_cut_hi_Z_FreeEOS_lo",
    "logRho_cut_FreeEOS_hi",
    "logRho_cut_FreeEOS_lo",
    "logT_cut_FreeEOS_hi",
    "logT_cut_FreeEOS_lo",
    "use_CMS",
    "CMS_use_fixed_composition",
    "CMS_fixed_composition_index",
    "max_Z_for_any_CMS",
    "max_Z_for_all_CMS",
    "logQ_max_for_any_CMS",
    "logQ_max_for_all_CMS",
    "logQ_min_for_any_CMS",
    "logQ_min_for_all_CMS",
    "logRho_max_for_all_CMS",
    "logRho_max_for_any_CMS",
    "logRho_min_for_all_CMS",
    "logRho_min_for_any_CMS",
    "logT_max_for_all_CMS",
    "logT_max_for_any_CMS",
    "logT_min_for_all_CMS",
    "logT_min_for_any_CMS",
    "logT_max_for_all_CMS_pure_He",
    "logT_max_for_any_CMS_pure_He",
    "use_PC",
    "mass_fraction_limit_for_PC",
    "logRho1_PC_limit",
    "logRho2_PC_limit",
    "PC_use_Gamma_limit_instead_of_T",
    "logT1_PC_limit",
    "logT2_PC_limit",
    "log_Gamma_e_all_HELM",
    "Gamma_e_all_HELM",
    "log_Gamma_e_all_PC",
    "tiny_fuzz",
    "PC_Gamma_start_crystal",
    "PC_Gamma_full_crystal",
    "use_Skye",
    "Skye_use_ion_offsets",
    "mass_fraction_limit_for_Skye",
    "Skye_min_gamma_for_solid",
    "Skye_max_gamma_for_liquid",
    "Skye_solid_mixing_rule",
    "use_simple_Skye_blends",
    "logRho_min_for_any_Skye",
    "logRho_min_for_all_Skye",
    "logT_min_for_any_Skye",
    "logT_min_for_all_Skye",
    "include_radiation",
    "include_elec_pos",
    "eosDT_use_linear_interp_for_X",
    "eosDT_use_linear_interp_to_HELM",
    "eosDT_file_prefix",
    "okay_to_convert_ierr_to_skip",
    "use_other_eos_component",
    "use_other_eos_results",
    "dbg",
];

/// Array controls of the `CONFIG EOS` line, written as `name(i)`.
const ARRAY_CONTROLS: &[(&str, usize)] = &[
    ("suffix_for_FreeEOS_Z", 15),
    ("eos_ctrl", 10),
    ("eos_integer_ctrl", 10),
    ("eos_logical_ctrl", 10),
    ("eos_character_ctrl", 10),
];

/// Every control name the `CONFIG EOS` line must carry.
fn expected_controls() -> BTreeSet<String> {
    let mut names: BTreeSet<String> =
        SCALAR_CONTROLS.iter().map(ToString::to_string).collect();
    for (name, count) in ARRAY_CONTROLS {
        for i in 1..=*count {
            names.insert(format!("{name}({i})"));
        }
    }
    names
}

/// Raw `CONFIG EOS` entries by control name.
struct Controls(BTreeMap<String, String>);

impl Controls {
    fn raw(&self, name: &str) -> Result<&str, FormatError> {
        self.0.get(name).map(String::as_str).ok_or_else(|| {
            FormatError(format!("CONFIG EOS: no control `{name}`"))
        })
    }

    fn real(&self, name: &str) -> Result<f64, FormatError> {
        let tok = self.raw(name)?;
        u64::from_str_radix(tok, 16)
            .map(f64::from_bits)
            .map_err(|e| {
                FormatError(format!("CONFIG EOS: {name}=`{tok}`: {e}"))
            })
    }

    fn logical(&self, name: &str) -> Result<bool, FormatError> {
        match self.raw(name)? {
            "T" => Ok(true),
            "F" => Ok(false),
            tok => Err(FormatError(format!(
                "CONFIG EOS: {name}=`{tok}` is not T or F"
            ))),
        }
    }

    fn int(&self, name: &str) -> Result<i64, FormatError> {
        let tok = self.raw(name)?;
        tok.parse().map_err(|e| {
            FormatError(format!("CONFIG EOS: {name}=`{tok}`: {e}"))
        })
    }

    /// A string control: `'...'` with blanks written as `~`, trimmed.
    fn string(&self, name: &str) -> Result<String, FormatError> {
        let tok = self.raw(name)?;
        let inner = tok
            .strip_prefix('\'')
            .and_then(|t| t.strip_suffix('\''))
            .ok_or_else(|| {
            FormatError(format!("CONFIG EOS: {name}=`{tok}` not quoted"))
        })?;
        Ok(inner.replace('~', " ").trim().to_string())
    }
}

fn unsupported(name: &str, value: &str, why: &str) -> FormatError {
    FormatError(format!(
        "CONFIG EOS: {name} = {value} is not replayable: {why}"
    ))
}

/// Builds the [`EosConfig`] of a `CONFIG EOS` line (the tokens after
/// `CONFIG EOS`, each `name=value`).
///
/// Controls are mapped onto [`EosConfig`] as `flexmesa-star` maps the
/// `&eos` namelist (`inlist_populate_micro.rs`, after
/// `eos/private/eos_ctrls_io.f90`). Fields of [`EosConfig`] that are not
/// namelist controls keep their defaults. `Gamma_e_all_HELM` is derived by
/// upstream from `log_Gamma_e_all_HELM` and is read only for presence;
/// `logT_lo` to `Z_hi` take effect only with `dbg`, which must be false.
///
/// # Errors
/// Returns [`FormatError`] for a malformed entry, a duplicated, unknown or
/// missing control, or a control value `FlexMESA` does not support
/// (`use_other_eos_*`, `dbg`, non-bundled table files).
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub fn eos_config_from_config<'a>(
    tokens: impl Iterator<Item = &'a str>,
) -> Result<EosConfig, FormatError> {
    let mut map = BTreeMap::new();
    for tok in tokens {
        let (name, value) = tok.split_once('=').ok_or_else(|| {
            FormatError(format!("CONFIG EOS: entry `{tok}` lacks `=`"))
        })?;
        if map.insert(name.to_string(), value.to_string()).is_some() {
            return Err(FormatError(format!(
                "CONFIG EOS: control `{name}` given twice"
            )));
        }
    }
    let expected = expected_controls();
    if let Some(extra) = map.keys().find(|k| !expected.contains(*k)) {
        return Err(FormatError(format!(
            "CONFIG EOS: unknown control `{extra}`"
        )));
    }
    if let Some(missing) = expected.iter().find(|k| !map.contains_key(*k)) {
        return Err(FormatError(format!("CONFIG EOS: no control `{missing}`")));
    }
    let c = Controls(map);

    for name in ["use_other_eos_component", "use_other_eos_results", "dbg"] {
        if c.logical(name)? {
            return Err(unsupported(name, "T", "hook or debug path"));
        }
    }
    let prefix = c.string("eosDT_file_prefix")?;
    if prefix != "mesa" {
        return Err(unsupported(
            "eosDT_file_prefix",
            &prefix,
            "only the bundled 'mesa' tables exist",
        ));
    }
    for i in 1..=15 {
        let name = format!("suffix_for_FreeEOS_Z({i})");
        let s = c.string(&name)?;
        let expected = if i >= 13 { "_CO" } else { "" };
        if s != expected {
            return Err(unsupported(
                &name,
                &s,
                "only the bundled FreeEOS tables exist",
            ));
        }
    }
    let _ = c.real("Gamma_e_all_HELM")?;

    let mut cfg = EosConfig {
        use_free_eos: c.logical("use_FreeEOS")?,
        use_opal_scvh: c.logical("use_OPAL_SCVH")?,
        use_pc: c.logical("use_PC")?,
        use_skye: c.logical("use_Skye")?,
        use_cms: c.logical("use_CMS")?,
        okay_to_convert_ierr_to_skip: c
            .logical("okay_to_convert_ierr_to_skip")?,
        eosdt_use_linear_interp_to_helm: c
            .logical("eosDT_use_linear_interp_to_HELM")?,
        ..EosConfig::default()
    };

    let include_radiation = c.logical("include_radiation")?;
    cfg.helm.include_radiation = include_radiation;
    cfg.pc.include_radiation = include_radiation;
    cfg.cms.include_radiation = include_radiation;
    cfg.helm.include_elec_pos = c.logical("include_elec_pos")?;
    cfg.helm.coulomb_temp_cut = c.real("coulomb_temp_cut_HELM")?;
    cfg.helm.coulomb_den_cut = c.real("coulomb_den_cut_HELM")?;

    let lin_x = c.logical("eosDT_use_linear_interp_for_X")?;
    cfg.free_eos.use_linear_interp_for_x = lin_x;
    cfg.opal_scvh.use_linear_interp_for_x = lin_x;
    let fuzz = c.real("tiny_fuzz")?;
    cfg.free_eos.tiny_fuzz = fuzz;
    cfg.opal_scvh.tiny_fuzz = fuzz;

    let o = &mut cfg.opal_scvh;
    for (dst, key) in [
        (&mut o.log_t_low_all_helm, "logT_low_all_HELM"),
        (&mut o.log_t_low_all_scvh, "logT_low_all_SCVH"),
        (&mut o.log_t_all_opal, "logT_all_OPAL"),
        (&mut o.log_t_all_helm, "logT_all_HELM"),
        (&mut o.log_rho1_opal_scvh_limit, "logRho1_OPAL_SCVH_limit"),
        (&mut o.log_rho2_opal_scvh_limit, "logRho2_OPAL_SCVH_limit"),
        (
            &mut o.log_rho_min_opal_scvh_limit,
            "logRho_min_OPAL_SCVH_limit",
        ),
        (&mut o.log_q_max_opal_scvh, "logQ_max_OPAL_SCVH"),
        (&mut o.log_q_min_opal_scvh, "logQ_min_OPAL_SCVH"),
        (&mut o.z_all_opal, "Z_all_OPAL"),
        (&mut o.z_all_helm, "Z_all_HELM"),
    ] {
        *dst = c.real(key)?;
    }

    let f = &mut cfg.free_eos;
    for (dst, key) in [
        (&mut f.log_q_max_hi, "logQ_max_FreeEOS_hi"),
        (&mut f.log_q_max_lo, "logQ_max_FreeEOS_lo"),
        (&mut f.log_q_min_hi, "logQ_min_FreeEOS_hi"),
        (&mut f.log_q_min_lo, "logQ_min_FreeEOS_lo"),
        (&mut f.log_rho_min_hi, "logRho_min_FreeEOS_hi"),
        (&mut f.log_rho_min_lo, "logRho_min_FreeEOS_lo"),
        (&mut f.log_rho_max_hi, "logRho_max_FreeEOS_hi"),
        (&mut f.log_rho_max_lo, "logRho_max_FreeEOS_lo"),
        (&mut f.log_t_min_hi, "logT_min_FreeEOS_hi"),
        (&mut f.log_t_min_lo, "logT_min_FreeEOS_lo"),
        (&mut f.log_t_max_hi, "logT_max_FreeEOS_hi"),
        (&mut f.log_t_max_lo, "logT_max_FreeEOS_lo"),
        (&mut f.log_q_cut_lo_z_max, "logQ_cut_FreeEOS_lo_Z_max"),
        (&mut f.log_q_cut_lo_z_hi, "logQ_cut_lo_Z_FreeEOS_hi"),
        (&mut f.log_q_cut_lo_z_lo, "logQ_cut_lo_Z_FreeEOS_lo"),
        (&mut f.log_q_cut_hi_z_hi, "logQ_cut_hi_Z_FreeEOS_hi"),
        (&mut f.log_q_cut_hi_z_lo, "logQ_cut_hi_Z_FreeEOS_lo"),
        (&mut f.log_rho_cut_hi, "logRho_cut_FreeEOS_hi"),
        (&mut f.log_rho_cut_lo, "logRho_cut_FreeEOS_lo"),
        (&mut f.log_t_cut_hi, "logT_cut_FreeEOS_hi"),
        (&mut f.log_t_cut_lo, "logT_cut_FreeEOS_lo"),
    ] {
        *dst = c.real(key)?;
    }

    cfg.cms.use_fixed_composition = c.logical("CMS_use_fixed_composition")?;
    cfg.cms.fixed_composition_index =
        isize::try_from(c.int("CMS_fixed_composition_index")?)
            .map_err(|e| FormatError(format!("CONFIG EOS: CMS index: {e}")))?;
    let m = &mut cfg.cms;
    for (dst, key) in [
        (&mut m.max_z_for_any_cms, "max_Z_for_any_CMS"),
        (&mut m.max_z_for_all_cms, "max_Z_for_all_CMS"),
        (&mut m.logq_max_for_any_cms, "logQ_max_for_any_CMS"),
        (&mut m.logq_max_for_all_cms, "logQ_max_for_all_CMS"),
        (&mut m.logq_min_for_any_cms, "logQ_min_for_any_CMS"),
        (&mut m.logq_min_for_all_cms, "logQ_min_for_all_CMS"),
        (&mut m.logrho_max_for_all_cms, "logRho_max_for_all_CMS"),
        (&mut m.logrho_max_for_any_cms, "logRho_max_for_any_CMS"),
        (&mut m.logrho_min_for_all_cms, "logRho_min_for_all_CMS"),
        (&mut m.logrho_min_for_any_cms, "logRho_min_for_any_CMS"),
        (&mut m.logt_max_for_all_cms, "logT_max_for_all_CMS"),
        (&mut m.logt_max_for_any_cms, "logT_max_for_any_CMS"),
        (&mut m.logt_min_for_all_cms, "logT_min_for_all_CMS"),
        (&mut m.logt_min_for_any_cms, "logT_min_for_any_CMS"),
        (
            &mut m.logt_max_for_all_cms_pure_he,
            "logT_max_for_all_CMS_pure_He",
        ),
        (
            &mut m.logt_max_for_any_cms_pure_he,
            "logT_max_for_any_CMS_pure_He",
        ),
    ] {
        *dst = c.real(key)?;
    }

    cfg.pc.pc_use_gamma_limit_instead_of_t =
        c.logical("PC_use_Gamma_limit_instead_of_T")?;
    let p = &mut cfg.pc;
    for (dst, key) in [
        (
            &mut p.mass_fraction_limit_for_pc,
            "mass_fraction_limit_for_PC",
        ),
        (&mut p.log_rho1_pc_limit, "logRho1_PC_limit"),
        (&mut p.log_rho2_pc_limit, "logRho2_PC_limit"),
        (&mut p.log_t1_pc_limit, "logT1_PC_limit"),
        (&mut p.log_t2_pc_limit, "logT2_PC_limit"),
        (&mut p.log_gamma_e_all_helm, "log_Gamma_e_all_HELM"),
        (&mut p.log_gamma_e_all_pc, "log_Gamma_e_all_PC"),
        (&mut p.pc_gamma_start_crystal, "PC_Gamma_start_crystal"),
        (&mut p.pc_gamma_full_crystal, "PC_Gamma_full_crystal"),
    ] {
        *dst = c.real(key)?;
    }

    cfg.skye.skye_use_ion_offsets = c.logical("Skye_use_ion_offsets")?;
    cfg.skye.use_simple_skye_blends = c.logical("use_simple_Skye_blends")?;
    let rule = c.string("Skye_solid_mixing_rule")?;
    cfg.skye.skye_solid_mixing_rule = match rule.as_str() {
        "PC" => SkyeSolidMixingRule::Pc,
        "Ogata" => SkyeSolidMixingRule::Ogata,
        other => {
            return Err(unsupported(
                "Skye_solid_mixing_rule",
                other,
                "expected 'PC' or 'Ogata'",
            ))
        }
    };
    let s = &mut cfg.skye;
    for (dst, key) in [
        (
            &mut s.mass_fraction_limit_for_skye,
            "mass_fraction_limit_for_Skye",
        ),
        (&mut s.skye_min_gamma_for_solid, "Skye_min_gamma_for_solid"),
        (
            &mut s.skye_max_gamma_for_liquid,
            "Skye_max_gamma_for_liquid",
        ),
        (&mut s.log_rho_min_for_any_skye, "logRho_min_for_any_Skye"),
        (&mut s.log_rho_min_for_all_skye, "logRho_min_for_all_Skye"),
        (&mut s.log_t_min_for_any_skye, "logT_min_for_any_Skye"),
        (&mut s.log_t_min_for_all_skye, "logT_min_for_all_Skye"),
    ] {
        *dst = c.real(key)?;
    }

    for i in 0..10 {
        cfg.eos_ctrl[i] = c.real(&format!("eos_ctrl({})", i + 1))?;
        cfg.eos_integer_ctrl[i] =
            c.int(&format!("eos_integer_ctrl({})", i + 1))?;
        cfg.eos_logical_ctrl[i] =
            c.logical(&format!("eos_logical_ctrl({})", i + 1))?;
        // `eos_character_ctrl` is read for presence only: no `FlexMESA`
        // EOS path reads it (`inlist_populate_micro.rs`).
        let _ = c.string(&format!("eos_character_ctrl({})", i + 1))?;
    }
    Ok(cfg)
}

fn vec_field<'r>(
    rec: &'r Record,
    name: &str,
) -> Result<&'r [f64], FormatError> {
    match rec.value(name) {
        Some(Value::Vec(v)) => Ok(v),
        Some(_) => Err(FormatError(format!(
            "line {}: field `{name}` is not a vector",
            rec.line
        ))),
        None => {
            Err(FormatError(format!("line {}: no field `{name}`", rec.line)))
        }
    }
}

/// Report name of element `i` (0-based) of an EOS result vector: the
/// `res` name for `res`, `d_dlnd` and `d_dlnT`, and
/// `(lnPgas|lnE, species j)` for `d_dxa` (column-major, `d_dxa(1:2, j)`).
#[must_use]
pub fn element_name(field: &str, i: usize) -> String {
    let base = field
        .strip_prefix("out_")
        .or_else(|| field.strip_prefix("out "))
        .unwrap_or(field);
    match base {
        "res" | "d_dlnd" | "d_dlnT" => EOS_RESULT_NAMES.get(i).map_or_else(
            || format!("{field}({})", i + 1),
            |n| format!("{field}({n})"),
        ),
        "d_dxa" => format!(
            "{field}({}, species {})",
            if i.is_multiple_of(NUM_EOS_D_DXA_RESULTS) {
                "lnPgas"
            } else {
                "lnE"
            },
            i / NUM_EOS_D_DXA_RESULTS + 1
        ),
        _ => format!("{field}({})", i + 1),
    }
}

fn compare_vec(diff: &mut CallDiff, field: &str, mesa: &[f64], flex: &[f64]) {
    if mesa.len() != flex.len() {
        diff.discrete(&format!("{field} length"), &mesa.len(), &flex.len());
        return;
    }
    for (i, (a, b)) in mesa.iter().zip(flex).enumerate() {
        diff.real(&element_name(field, i), *a, *b);
    }
}

/// Replays one `EOS` record with the configuration of the run's
/// `CONFIG EOS` line.
///
/// Upstream `get_eos` returns `ierr = -1` without evaluating the EOS for
/// `logRho < -25` (`star/private/eos_support.f90:70-76`); the replay does
/// the same. On a failed call (either side) only `ierr` is compared:
/// upstream leaves the results undefined.
///
/// # Errors
/// Returns [`FormatError`] for a record missing a field or with a bad
/// `chem_id` list.
pub fn replay_eos(
    rec: &Record,
    cfg: &EosConfig,
    ht: &HelmTable,
) -> Result<Replayed, FormatError> {
    let chem_id: Vec<usize> = rec
        .str("chem_id")?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|e| FormatError(format!("line {}: chem_id: {e}", rec.line)))?;
    let xa = vec_field(rec, "xa")?;
    if xa.len() != chem_id.len() {
        return Err(FormatError(format!(
            "line {}: {} mass fractions for {} species",
            rec.line,
            xa.len(),
            chem_id.len()
        )));
    }
    let rho = rec.real("Rho")?;
    let log_rho = rec.real("logRho")?;
    let t = rec.real("T")?;
    let log_t = rec.real("logT")?;
    let mesa_ierr = rec.int("out_ierr")?;
    let mut diff = CallDiff::default();

    let result = if log_rho < LOG_RHO_FLOOR {
        Err("logRho < -25 (get_eos ierr = -1)".to_string())
    } else {
        eos_dt_get_zones_with_logs(
            ht,
            &[rho],
            &[log_rho],
            &[t],
            &[log_t],
            &chem_id,
            xa,
            cfg,
        )
        .map_err(|e| e.to_string())
        .and_then(|mut v| v.pop().ok_or_else(|| "no zone result".to_string()))
    };
    match result {
        Ok(out) => {
            diff.discrete("ierr", &mesa_ierr, &0);
            if mesa_ierr != 0 {
                return Ok(Replayed::Compared(diff));
            }
            compare_vec(&mut diff, "res", vec_field(rec, "out_res")?, &out.res);
            compare_vec(
                &mut diff,
                "d_dlnd",
                vec_field(rec, "out_d_dlnd")?,
                &out.d_dlnd,
            );
            compare_vec(
                &mut diff,
                "d_dlnT",
                vec_field(rec, "out_d_dlnT")?,
                &out.d_dln_t,
            );
            let flex_dxa: Vec<f64> =
                out.d_dxa.iter().flatten().copied().collect();
            compare_vec(
                &mut diff,
                "d_dxa",
                vec_field(rec, "out_d_dxa")?,
                &flex_dxa,
            );
        }
        Err(e) => {
            if mesa_ierr == 0 {
                diff.discrete("ierr", "0", format!("error: {e}").as_str());
            }
        }
    }
    Ok(Replayed::Compared(diff))
}
