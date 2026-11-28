// SPDX-License-Identifier: GPL-3.0-or-later
//! Recursive multi-section Fortran `inlist` parser (`InlistBundle`).
//!
//! Resolves `read_extra_<section>_inlist(1..5)` and
//! `extra_<section>_inlist_name(1..5)` (along with legacy un-subscripted
//! `read_extra_<section>_inlist1..5`) across `&star_job`, `&eos`, `&kap`,
//! `&colors`, `&controls`, `&pgstar`, `&binary_job`, `&binary_controls`, `&pgbinary`,
//! `&astero_search_controls`, and `&astero_pgstar_controls`, and maps parsed
//! values into [`StarJobControls`], [`StarControls`], [`EosConfig`], and
//! [`KapConfig`].

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use flexmesa_eos::EosConfig;
use flexmesa_kap::KapConfig;
use flexmesa_schema::{
    is_headless_plotting_section, parse_namelist_file, section_include_prefix,
    ControlValidationReport, ControlValue, NamelistAssignment,
};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::inlist_inventory::supported_star_control_inventory;

/// `max_extra_inlists` (`const/public/const_def.f90:39`).
const MAX_EXTRA_INLISTS: usize = 5;
/// Nesting limit of the ctrls readers (`level >= 10` is an error,
/// `star/private/star_job_ctrls_io.f90:557-561`).
const MAX_INCLUDE_LEVEL: usize = 10;

/// The module-level `read_extra_<section>_inlist(1:5)` and
/// `extra_<section>_inlist_name(1:5)` arrays of one namelist reader.
///
/// MESA's readers keep these in place across the recursion: a file's
/// namelist read sets some slots, and the include loop copies and resets each
/// slot before reading the file it names (`star_job_ctrls_io.f90:597-607`;
/// the same pattern is in `read_controls_file`, `ctrls_io.f90:700-709`, and
/// the `eos` and `kap` readers). A child inlist that sets slot `j` thus
/// overrides the parent's slot `j`, and the parent's later slots are already
/// reset when the child returns (Deferred bug #705).
#[derive(Debug, Clone, Default)]
struct ExtraSlots {
    read: [bool; MAX_EXTRA_INLISTS],
    name: [Option<String>; MAX_EXTRA_INLISTS],
}

/// Number of abundance-limit slots per block in MESA
/// (`star_data/public/star_data_def.inc:374-376`).
pub(crate) const NUM_XA_LIMITS: usize = 9;

/// Flattened key-value and indexed-element store for a single namelist section.
#[derive(Debug, Clone, Default)]
pub struct SectionControls {
    /// Scalar `name -> ControlValue` assignments (lowercased keys).
    pub scalars: HashMap<String, ControlValue>,
    /// Indexed `name -> [(subscript, ControlValue)]` assignments (lowercased keys).
    pub indexed: HashMap<String, Vec<(String, ControlValue)>>,
    /// Chronological assignments per lowercased key (`None` = unsubscripted scalar).
    timeline: HashMap<String, Vec<(Option<String>, ControlValue)>>,
}

impl SectionControls {
    /// Records a [`NamelistAssignment`] in `scalars` or `indexed`, preserving
    /// chronological order across single-element and unsubscripted assignments.
    pub fn insert(&mut self, assign: &NamelistAssignment) {
        let key = assign.name.to_ascii_lowercase();
        if let Some(sub) = &assign.subscript {
            let entry = self.indexed.entry(key.clone()).or_default();
            if let Some(existing) = entry.iter_mut().find(|(s, _)| s == sub) {
                existing.1 = assign.value.clone();
            } else {
                entry.push((sub.clone(), assign.value.clone()));
            }
            self.timeline
                .entry(key)
                .or_default()
                .push((Some(sub.clone()), assign.value.clone()));
        } else {
            self.scalars.insert(key.clone(), assign.value.clone());
            self.timeline
                .entry(key)
                .or_default()
                .push((None, assign.value.clone()));
        }
    }

    fn ordered_assignments(
        &self,
        key: &str,
    ) -> Vec<(Option<&str>, &ControlValue)> {
        let low = key.to_ascii_lowercase();
        if let Some(entries) = self.timeline.get(&low) {
            return entries
                .iter()
                .map(|(sub, val)| (sub.as_deref(), val))
                .collect();
        }
        let mut out = Vec::new();
        if let Some(val) = self.scalars.get(&low) {
            out.push((None, val));
        }
        if let Some(list) = self.indexed.get(&low) {
            for (sub, val) in list {
                out.push((Some(sub.as_str()), val));
            }
        }
        out
    }

    /// Returns a deduplicated sorted list of lowercase control keys in `self`.
    #[must_use]
    pub fn all_keys(&self) -> Vec<String> {
        let mut set: HashSet<String> = self.scalars.keys().cloned().collect();
        set.extend(self.indexed.keys().cloned());
        let mut out: Vec<String> = set.into_iter().collect();
        out.sort();
        out
    }

    /// Strict boolean scalar getter (`Err` on wrong value type or unexpected subscript).
    pub fn try_get_bool(
        &self,
        section: &str,
        name: &str,
    ) -> Result<Option<bool>, StarError> {
        let low = name.to_ascii_lowercase();
        if self.indexed.contains_key(&low) {
            return Err(StarError::InvalidInput(format!(
                "unexpected array subscript on scalar control '&{section}::{low}'"
            )));
        }
        match self.scalars.get(&low) {
            None => Ok(None),
            Some(ControlValue::Logical(b)) => Ok(Some(*b)),
            Some(other) => Err(StarError::InvalidInput(format!(
                "expected boolean for '&{section}::{low}', found {other:?}"
            ))),
        }
    }

    /// Strict integer scalar getter (`Err` on non-integer or unexpected subscript).
    pub fn try_get_i64(
        &self,
        section: &str,
        name: &str,
    ) -> Result<Option<i64>, StarError> {
        let low = name.to_ascii_lowercase();
        if self.indexed.contains_key(&low) {
            return Err(StarError::InvalidInput(format!(
                "unexpected array subscript on scalar control '&{section}::{low}'"
            )));
        }
        match self.scalars.get(&low) {
            None => Ok(None),
            Some(ControlValue::Integer(i)) => Ok(Some(*i)),
            Some(other) => Err(StarError::InvalidInput(format!(
                "expected integer for '&{section}::{low}', found {other:?}"
            ))),
        }
    }

    /// Strict finite `f64` scalar getter (`Err` on wrong type, non-finite value, or subscript).
    pub fn try_get_f64(
        &self,
        section: &str,
        name: &str,
    ) -> Result<Option<f64>, StarError> {
        let low = name.to_ascii_lowercase();
        if self.indexed.contains_key(&low) {
            return Err(StarError::InvalidInput(format!(
                "unexpected array subscript on scalar control '&{section}::{low}'"
            )));
        }
        let val = match self.scalars.get(&low) {
            None => return Ok(None),
            Some(ControlValue::Float(f)) => *f,
            Some(ControlValue::Integer(i)) => *i as f64,
            Some(other) => {
                return Err(StarError::InvalidInput(format!(
                    "expected real number for '&{section}::{low}', found {other:?}"
                )));
            }
        };
        if !val.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "non-finite value for '&{section}::{low}': {val}"
            )));
        }
        Ok(Some(val))
    }

    /// Strict string scalar getter (`Err` on non-string or unexpected subscript).
    pub fn try_get_str(
        &self,
        section: &str,
        name: &str,
    ) -> Result<Option<&str>, StarError> {
        let low = name.to_ascii_lowercase();
        if self.indexed.contains_key(&low) {
            return Err(StarError::InvalidInput(format!(
                "unexpected array subscript on scalar control '&{section}::{low}'"
            )));
        }
        match self.scalars.get(&low) {
            None => Ok(None),
            Some(ControlValue::String(s)) => Ok(Some(s.as_str())),
            Some(other) => Err(StarError::InvalidInput(format!(
                "expected string for '&{section}::{low}', found {other:?}"
            ))),
        }
    }

    /// Strict indexed boolean getter for `base_name(idx)` or legacy `base_name<idx>`.
    pub fn try_get_indexed_bool(
        &self,
        section: &str,
        base_name: &str,
        idx: usize,
    ) -> Result<Option<bool>, StarError> {
        let key = base_name.to_ascii_lowercase();
        let idx_str = idx.to_string();
        if let Some(list) = self.indexed.get(&key) {
            if let Some((_, val)) =
                list.iter().find(|(s, _)| s.trim() == idx_str)
            {
                return match val {
                    ControlValue::Logical(b) => Ok(Some(*b)),
                    other => Err(StarError::InvalidInput(format!(
                        "expected boolean for '&{section}::{key}({idx})', found {other:?}"
                    ))),
                };
            }
        }
        self.try_get_bool(section, &format!("{key}{idx}"))
    }

    /// Strict indexed string getter for `extra_<sec>_inlist_name(idx)` or legacy aliases.
    pub fn try_get_indexed_inlist_name(
        &self,
        section: &str,
        extra_prefix: &str,
        idx: usize,
    ) -> Result<Option<&str>, StarError> {
        let sec = extra_prefix.to_ascii_lowercase();
        let key = format!("extra_{sec}_inlist_name");
        let idx_str = idx.to_string();
        if let Some(list) = self.indexed.get(&key) {
            if let Some((_, val)) =
                list.iter().find(|(s, _)| s.trim() == idx_str)
            {
                return match val {
                    ControlValue::String(s) => Ok(Some(s.as_str())),
                    other => Err(StarError::InvalidInput(format!(
                        "expected string for '&{section}::{key}({idx})', found {other:?}"
                    ))),
                };
            }
        }
        if let Some(s) =
            self.try_get_str(section, &format!("extra_{sec}_inlist{idx}_name"))?
        {
            return Ok(Some(s));
        }
        self.try_get_str(section, &key)
    }

    fn apply_positional_1based<T: Clone>(
        &self,
        section: &str,
        key: &str,
        max_len: Option<usize>,
        default_fill: &T,
        extract: impl Fn(&ControlValue) -> Result<T, String>,
    ) -> Result<Vec<T>, StarError> {
        let low = key.to_ascii_lowercase();
        let mut out = Vec::new();
        for (sub_opt, raw_val) in self.ordered_assignments(&low) {
            let idx = if let Some(sub) = sub_opt {
                let parsed = sub.trim().parse::<usize>().map_err(|_| {
                    StarError::InvalidInput(format!(
                        "invalid subscript '({sub})' for '&{section}::{low}'"
                    ))
                })?;
                if parsed == 0 {
                    return Err(StarError::InvalidInput(format!(
                        "non-positive 1-based subscript '(0)' for '&{section}::{low}'"
                    )));
                }
                parsed
            } else {
                1
            };
            if let Some(limit) = max_len {
                if idx > limit {
                    return Err(StarError::InvalidInput(format!(
                        "subscript '({idx})' out of 1-based bounds (1..={limit}) for '&{section}::{low}'"
                    )));
                }
            }
            let ctx = sub_opt.map_or_else(
                || format!("'&{section}::{low}'"),
                |_| format!("'&{section}::{low}({idx})'"),
            );
            let v = extract(raw_val).map_err(|msg| {
                StarError::InvalidInput(format!(
                    "invalid value for {ctx}: {msg}"
                ))
            })?;
            if out.len() < idx {
                out.resize(idx, default_fill.clone());
            }
            out[idx - 1] = v;
        }
        Ok(out)
    }

    pub(crate) fn try_get_indexed_strings_bounded(
        &self,
        section: &str,
        key: &str,
        max_len: Option<usize>,
    ) -> Result<Vec<String>, StarError> {
        self.apply_positional_1based(
            section,
            key,
            max_len,
            &String::new(),
            |val| match val {
                ControlValue::String(s) => Ok(s.clone()),
                other => Err(format!("expected string, found {other:?}")),
            },
        )
    }

    pub(crate) fn try_get_indexed_f64s_bounded(
        &self,
        section: &str,
        key: &str,
        max_len: Option<usize>,
    ) -> Result<Vec<f64>, StarError> {
        self.apply_positional_1based(section, key, max_len, &0.0, |val| {
            match val {
                ControlValue::Float(f) if f.is_finite() => Ok(*f),
                ControlValue::Float(f) => Err(format!("non-finite float {f}")),
                ControlValue::Integer(i) => Ok(*i as f64),
                other => Err(format!("expected real number, found {other:?}")),
            }
        })
    }

    /// Strict indexed string array getter preserving 1-based positions and chronological overrides.
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] on non-positive or non-integer
    /// subscripts or non-string values.
    pub fn try_get_indexed_strings(
        &self,
        section: &str,
        key: &str,
    ) -> Result<Vec<String>, StarError> {
        self.try_get_indexed_strings_bounded(section, key, None)
    }

    /// Strict indexed finite `f64` array getter preserving 1-based positions and chronological overrides.
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] on non-positive or non-integer
    /// subscripts or non-numeric / non-finite values.
    pub fn try_get_indexed_f64s(
        &self,
        section: &str,
        key: &str,
    ) -> Result<Vec<f64>, StarError> {
        self.try_get_indexed_f64s_bounded(section, key, None)
    }
}

/// Configuration controlling how unsupported upstream controls and sections
/// are handled when loading an inlist bundle (`#147`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InlistLoadOptions {
    /// When `true`, allow known upstream MESA controls and sections that
    /// FlexMESA has not yet implemented. Unknown/misspelled keys, unknown
    /// sections, wrong value types, and unrecognized option strings always fail.
    pub allow_unsupported_controls: bool,
}

/// Complete recursively resolved MESA `inlist` bundle for a single run stage.
#[derive(Debug, Clone)]
pub struct InlistBundle {
    /// Root case directory containing the entry `inlist`.
    pub case_dir: PathBuf,
    /// Entry `inlist` filename relative to `case_dir`.
    pub entry_inlist: String,
    /// Parsed `&star_job` controls.
    pub job: StarJobControls,
    /// Parsed `&controls` evolutionary and physics controls.
    pub controls: StarControls,
    /// Parsed `&eos` configuration.
    pub eos_config: EosConfig,
    /// Parsed `&kap` configuration.
    pub kap_config: KapConfig,
    /// Control validation summary across parsed namelist sections (`#147`).
    pub validation_report: ControlValidationReport,
    /// Raw flattened assignments for every namelist section (`star_job`, `eos`,
    /// `kap`, `controls`, `pgstar`, `binary_job`, `binary_controls`, `pgbinary`,
    /// `astero_search_controls`, `simplex_search_controls`, etc.).
    pub sections: HashMap<String, SectionControls>,
    /// Whether the run continues past unsupported upstream controls
    /// (`--allow-unsupported-controls`, `#147`, `DECISIONS #26`). Checks made
    /// after the inlists are read (for example an `other_*` hook flag whose
    /// case handler is not ported) honour it the same way.
    pub allow_unsupported_controls: bool,
}

impl InlistBundle {
    /// Resolves `name` against `case_dir` or `case_dir/../shared/`.
    #[must_use]
    pub fn resolve_inlist_path(case_dir: &Path, name: &str) -> Option<PathBuf> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }
        let direct = case_dir.join(trimmed);
        if direct.exists() {
            return Some(direct);
        }
        if let Some(parent) = case_dir.parent() {
            let shared = parent.join("shared").join(trimmed);
            if shared.exists() {
                return Some(shared);
            }
        }
        None
    }

    /// Recursively parses `inlist_name` inside `case_dir` into raw flattened
    /// sections without validating against single-star control inventories.
    ///
    /// Used internally by single-star loading and externally by `flexmesa-binary`
    /// when reading `&binary_job`/`&binary_controls`/`&pgbinary` entry inlists.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] or [`StarError::InvalidInput`] if a
    /// required `inlist` file cannot be found, read, or parsed.
    pub fn load_sections_in_case_dir(
        case_dir: &Path,
        inlist_name: &str,
    ) -> Result<HashMap<String, SectionControls>, StarError> {
        let Some(entry_path) = Self::resolve_inlist_path(case_dir, inlist_name)
        else {
            return Err(StarError::IoError(format!(
                "missing entry inlist '{inlist_name}' in '{}'",
                case_dir.display()
            )));
        };
        let mut sections: HashMap<String, SectionControls> = HashMap::new();
        let mut slots: HashMap<String, ExtraSlots> = HashMap::new();
        let mut pending_gui_includes: Vec<(String, PathBuf, SectionControls)> =
            Vec::new();

        Self::load_file_recursive(
            case_dir,
            &entry_path,
            None,
            &mut sections,
            &mut slots,
            1,
            Some(&mut pending_gui_includes),
        )?;

        let pgstar_active = sections
            .get("star_job")
            .and_then(|s| {
                s.try_get_bool("star_job", "pgstar_flag").ok().flatten()
            })
            .unwrap_or(false);
        let pgbinary_active = sections
            .get("binary_job")
            .and_then(|s| {
                s.try_get_bool("binary_job", "pgbinary_flag").ok().flatten()
            })
            .unwrap_or(false);

        for (sec_name, parent_file, local_sec) in pending_gui_includes {
            let active = match sec_name.as_str() {
                "pgbinary" => pgbinary_active,
                _ => pgstar_active,
            };
            if !active {
                continue;
            }
            // Deferred headless sections start from fresh slots: only the
            // block that named them is replayed.
            let mut fresh: HashMap<String, ExtraSlots> = HashMap::new();
            Self::resolve_section_includes(
                case_dir,
                &parent_file,
                &sec_name,
                &local_sec,
                &mut fresh,
                &mut sections,
                1,
                None,
            )?;
        }
        Ok(sections)
    }

    /// Parses `inlist_name` inside `case_dir` with default strict options
    /// (`allow_unsupported_controls = false`).
    ///
    /// # Errors
    /// Returns [`StarError`] if a required `inlist` file cannot be read/parsed
    /// or if unsupported/unknown controls or option values are present.
    pub fn from_file_in_case_dir(
        case_dir: &Path,
        inlist_name: &str,
    ) -> Result<Self, StarError> {
        Self::from_file_in_case_dir_with_options(
            case_dir,
            inlist_name,
            InlistLoadOptions::default(),
        )
    }

    /// Parses `inlist_name` inside `case_dir` (checking `../shared/` if needed),
    /// recursively resolves `read_extra_*_inlist` includes, validates sections and
    /// keys according to `options`, and populates typed control structs.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if a required `inlist` file cannot be
    /// read or parsed, or [`StarError::InvalidInput`] if validation fails.
    pub fn from_file_in_case_dir_with_options(
        case_dir: &Path,
        inlist_name: &str,
        options: InlistLoadOptions,
    ) -> Result<Self, StarError> {
        let sections = Self::load_sections_in_case_dir(case_dir, inlist_name)?;

        let mut bundle = Self {
            case_dir: case_dir.to_path_buf(),
            entry_inlist: inlist_name.to_string(),
            job: StarJobControls::default(),
            controls: StarControls::default(),
            eos_config: EosConfig::default(),
            kap_config: KapConfig::default(),
            validation_report: ControlValidationReport::default(),
            sections,
            allow_unsupported_controls: options.allow_unsupported_controls,
        };
        let report = bundle.validate_star_sections();
        report
            .into_result(options.allow_unsupported_controls)
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        bundle.validation_report = report;
        bundle.populate_typed_structs_with_options(
            options.allow_unsupported_controls,
        )?;
        Ok(bundle)
    }

    /// Validates `self.sections` against FlexMESA's single-star inventory and
    /// upstream MESA `26.4.1` defaults (`DECISIONS #25`).
    #[must_use]
    pub fn validate_star_sections(&self) -> ControlValidationReport {
        let inv = supported_star_control_inventory();
        let empty: HashSet<&'static str> = HashSet::new();
        let mut report = ControlValidationReport::default();

        let pgstar_active = self
            .sections
            .get("star_job")
            .and_then(|s| {
                s.try_get_bool("star_job", "pgstar_flag").ok().flatten()
            })
            .unwrap_or(false);

        for (sec, ctrls) in &self.sections {
            let keys = ctrls.all_keys();
            match sec.as_str() {
                "pgstar" => {
                    report.validate_headless_plotting_section_with_active(
                        sec,
                        &keys,
                        pgstar_active,
                    );
                }
                "star_job" => {
                    report.validate_implemented_section_with_terminal_output(
                        sec,
                        &keys,
                        &inv.star_job_supported,
                        &inv.star_job_unused,
                        &inv.star_job_headless_gui_flags,
                        &inv.star_job_terminal_output,
                    );
                }
                "eos" => {
                    report.validate_implemented_section_with_terminal_output(
                        sec,
                        &keys,
                        &inv.eos_supported,
                        &empty,
                        &empty,
                        &inv.eos_terminal_output,
                    );
                }
                "kap" => {
                    report.validate_implemented_section_with_terminal_output(
                        sec,
                        &keys,
                        &inv.kap_supported,
                        &empty,
                        &empty,
                        &inv.kap_terminal_output,
                    );
                }
                "colors" => {
                    report.validate_implemented_section_with_terminal_output(
                        sec,
                        &keys,
                        &inv.colors_supported,
                        &inv.colors_unused,
                        &empty,
                        &inv.colors_terminal_output,
                    );
                }
                "controls" => {
                    report.validate_implemented_section_with_terminal_output(
                        sec,
                        &keys,
                        &inv.controls_supported,
                        &inv.controls_unused,
                        &empty,
                        &inv.controls_terminal_output,
                    );
                }
                other => {
                    report.validate_other_section(other);
                }
            }
        }
        report
    }

    /// Parses `inlist_path` with default strict options
    /// (`allow_unsupported_controls = false`).
    ///
    /// # Errors
    /// Returns [`StarError`] if a required `inlist` file cannot be read/parsed
    /// or if unsupported/unknown controls or option values are present.
    pub fn from_file(inlist_path: &Path) -> Result<Self, StarError> {
        Self::from_file_with_options(inlist_path, InlistLoadOptions::default())
    }

    /// Parses `inlist_path` and recursively resolves all `read_extra_*_inlist`
    /// includes relative to `inlist_path.parent()` according to `options`.
    ///
    /// # Errors
    /// Returns [`StarError`] if a required `inlist` file cannot be read/parsed
    /// or if validation fails.
    pub fn from_file_with_options(
        inlist_path: &Path,
        options: InlistLoadOptions,
    ) -> Result<Self, StarError> {
        let case_dir = inlist_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let entry_inlist = inlist_path.file_name().map_or_else(
            || String::from("inlist"),
            |s| s.to_string_lossy().into_owned(),
        );
        Self::from_file_in_case_dir_with_options(
            &case_dir,
            &entry_inlist,
            options,
        )
    }

    /// Merges the include slots set by `local_sec` into `slots`, then runs
    /// MESA's include loop (`star_job_ctrls_io.f90:597-607`): for each slot,
    /// copy it, reset it, and read the named file at once if it was set.
    /// The reset is in place, so this is not a tree walk (#705).
    fn resolve_section_includes(
        case_dir: &Path,
        parent_file: &Path,
        section_name: &str,
        local_sec: &SectionControls,
        slots: &mut HashMap<String, ExtraSlots>,
        sections: &mut HashMap<String, SectionControls>,
        level: usize,
        mut pending_gui_includes: Option<
            &mut Vec<(String, PathBuf, SectionControls)>,
        >,
    ) -> Result<(), StarError> {
        let extra_prefix = section_include_prefix(section_name);
        let state = slots.entry(section_name.to_ascii_lowercase()).or_default();
        for idx in 1..=MAX_EXTRA_INLISTS {
            let flag = local_sec.try_get_indexed_bool(
                section_name,
                &format!("read_extra_{extra_prefix}_inlist"),
                idx,
            )?;
            if let Some(flag) = flag {
                state.read[idx - 1] = flag;
            }
            // A name is read when this block sets the slot's name or its
            // flag (the legacy fallbacks of `try_get_indexed_inlist_name`
            // apply only with the flag, as before).
            let named_here = local_sec
                .indexed
                .get(&format!("extra_{extra_prefix}_inlist_name"))
                .is_some_and(|l| {
                    l.iter().any(|(s, _)| s.trim() == idx.to_string())
                });
            if named_here || flag == Some(true) {
                if let Some(name) = local_sec.try_get_indexed_inlist_name(
                    section_name,
                    extra_prefix,
                    idx,
                )? {
                    state.name[idx - 1] = Some(name.to_string());
                }
            }
        }
        for idx in 1..=MAX_EXTRA_INLISTS {
            let state =
                slots.entry(section_name.to_ascii_lowercase()).or_default();
            let read_flag = state.read[idx - 1];
            let extra_name = state.name[idx - 1].take();
            state.read[idx - 1] = false;
            if !read_flag {
                continue;
            }
            if let Some(extra_name) = extra_name {
                let trimmed = extra_name.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let Some(next_path) =
                    Self::resolve_inlist_path(case_dir, trimmed)
                else {
                    return Err(StarError::IoError(format!(
                        "missing extra_{extra_prefix}_inlist({idx}) = '{trimmed}' \
                         referenced from '{}'",
                        parent_file.display()
                    )));
                };
                Self::load_file_recursive(
                    case_dir,
                    &next_path,
                    Some(section_name),
                    sections,
                    slots,
                    level + 1,
                    pending_gui_includes.as_deref_mut(),
                )?;
            }
        }
        Ok(())
    }

    fn load_file_recursive(
        case_dir: &Path,
        file_path: &Path,
        target_section: Option<&str>,
        sections: &mut HashMap<String, SectionControls>,
        slots: &mut HashMap<String, ExtraSlots>,
        level: usize,
        mut pending_gui_includes: Option<
            &mut Vec<(String, PathBuf, SectionControls)>,
        >,
    ) -> Result<(), StarError> {
        if level >= MAX_INCLUDE_LEVEL {
            return Err(StarError::InvalidInput(format!(
                "too many levels of nested extra inlist files at '{}'",
                file_path.display()
            )));
        }

        let blocks = parse_namelist_file(file_path).map_err(|e| {
            StarError::IoError(format!(
                "failed to parse inlist '{}': {e}",
                file_path.display()
            ))
        })?;

        for block in &blocks {
            let bname = block.name.to_ascii_lowercase();
            if let Some(only_sec) = target_section {
                if bname != only_sec.to_ascii_lowercase() {
                    continue;
                }
            }
            let mut local_sec = SectionControls::default();
            let dest = sections.entry(bname.clone()).or_default();
            for assign in &block.assignments {
                local_sec.insert(assign);
                dest.insert(assign);
            }

            if is_headless_plotting_section(&bname) {
                if let Some(ref mut pending) = pending_gui_includes {
                    pending.push((bname, file_path.to_path_buf(), local_sec));
                    continue;
                }
            }
            Self::resolve_section_includes(
                case_dir,
                file_path,
                &bname,
                &local_sec,
                slots,
                sections,
                level,
                pending_gui_includes.as_deref_mut(),
            )?;
        }
        Ok(())
    }
}
