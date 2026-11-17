// SPDX-License-Identifier: LGPL-3.0-or-later
//! Authoritative MESA `26.4.1` control inventory and shared section-key
//! validation for single-star, binary, asteroseismology, and headless
//! `&pgstar` / `&pgbinary` namelists (`DECISIONS #25`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use crate::{parse_namelist_file, ControlValue, SchemaError};

const UPSTREAM_KEYS_TEXT: &str =
    include_str!("../data/mesa_26_4_1_namelist_keys.txt");

static UPSTREAM_SECTION_KEYS: OnceLock<
    HashMap<&'static str, HashSet<&'static str>>,
> = OnceLock::new();

fn load_upstream_section_keys(
) -> &'static HashMap<&'static str, HashSet<&'static str>> {
    UPSTREAM_SECTION_KEYS.get_or_init(|| {
        let mut map: HashMap<&'static str, HashSet<&'static str>> =
            HashMap::new();
        let mut current: Option<&'static str> = None;
        for raw_line in UPSTREAM_KEYS_TEXT.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(sec) =
                line.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
            {
                current = Some(sec);
                map.entry(sec).or_default();
            } else if let Some(sec) = current {
                map.entry(sec).or_default().insert(line);
            }
        }
        map
    })
}

/// Returns the authoritative MESA `26.4.1` lowercase bare control key set for
/// `section`, if known.
#[must_use]
pub fn upstream_section_keys(
    section: &str,
) -> Option<&'static HashSet<&'static str>> {
    let low = section.to_ascii_lowercase();
    load_upstream_section_keys().get(low.as_str())
}

/// Returns `true` if `section` is a headless plotting namelist validated against
/// upstream defaults and ignored at runtime (`DECISIONS #25`).
#[must_use]
pub fn is_headless_plotting_section(section: &str) -> bool {
    matches!(
        section.to_ascii_lowercase().as_str(),
        "pgstar" | "pgbinary" | "astero_pgstar_controls"
    )
}

/// Returns `true` if `section` is a recognized upstream MESA namelist outside
/// the currently implemented runner sections.
#[must_use]
pub fn is_known_upstream_only_section(section: &str) -> bool {
    matches!(
        section.to_ascii_lowercase().as_str(),
        "colors" | "multi_stars_job" | "simplex_search_controls" | "stella"
    )
}

/// Canonical include prefix for `read_extra_<prefix>_inlist` directives.
///
/// Expects a lowercase section name (as produced by `to_ascii_lowercase`).
#[must_use]
pub fn section_include_prefix(section: &str) -> &str {
    if section.eq_ignore_ascii_case("astero_pgstar_controls") {
        "astero_pgstar"
    } else if section.eq_ignore_ascii_case("astero_search_controls") {
        "astero_search"
    } else if section.eq_ignore_ascii_case("simplex_search_controls") {
        "simplex_search"
    } else {
        section
    }
}

/// Returns `true` if `key` is an upstream `read_extra_<prefix>_inlist` or
/// `extra_<prefix>_inlist_name` include directive (including legacy `1..=5`
/// unsubscripted variants).
#[must_use]
pub fn is_include_key(section: &str, key: &str) -> bool {
    let low = key.to_ascii_lowercase();
    if !low.starts_with("read_extra_") && !low.starts_with("extra_") {
        return false;
    }
    let sec_low = section.to_ascii_lowercase();
    let prefix = section_include_prefix(&sec_low);
    let read_base = format!("read_extra_{prefix}_inlist");
    let name_base = format!("extra_{prefix}_inlist_name");
    if low == read_base || low == name_base {
        return true;
    }
    for idx in 1..=5 {
        if low == format!("{read_base}{idx}")
            || low == format!("extra_{prefix}_inlist{idx}_name")
        {
            return true;
        }
    }
    false
}

/// Aggregated result of validating parsed inlist sections against `FlexMESA`'s
/// supported controls and upstream MESA `26.4.1` defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ControlValidationReport {
    /// Known upstream sections not yet implemented (e.g., `colors`, `stella`).
    pub unsupported_sections: Vec<String>,
    /// Unrecognized namelist section names not found in MESA `26.4.1` defaults.
    pub unknown_sections: Vec<String>,
    /// Unsupported upstream or parsed-but-unused `(section, key)` pairs.
    pub unsupported_controls: Vec<(String, String)>,
    /// Unrecognized `(section, key)` pairs absent from MESA `26.4.1` defaults.
    pub unknown_controls: Vec<(String, String)>,
    /// Valid plotting `(section, key)` pairs recorded for headless `#173` audit.
    pub pgstar_plotting_keys: Vec<(String, String)>,
    /// `(section, key)` pairs whose only upstream effect is text written to
    /// the terminal (`DECISIONS #26`). Accepted and recorded; they never
    /// change the model, `LOGS/` output or termination.
    pub terminal_output_keys: Vec<(String, String)>,
}

impl ControlValidationReport {
    /// Merges all findings from `other` into `self`.
    pub fn merge(&mut self, other: Self) {
        self.unsupported_sections.extend(other.unsupported_sections);
        self.unknown_sections.extend(other.unknown_sections);
        self.unsupported_controls.extend(other.unsupported_controls);
        self.unknown_controls.extend(other.unknown_controls);
        self.pgstar_plotting_keys.extend(other.pgstar_plotting_keys);
        self.terminal_output_keys.extend(other.terminal_output_keys);
    }

    /// Validates a headless plotting section (`&pgstar`, `&pgbinary`, or
    /// `&astero_pgstar_controls`) against upstream `26.4.1` defaults.
    ///
    /// When `active` is `true` (`pgstar_flag` / `pgbinary_flag`), unknown or
    /// stale keys are recorded in `unknown_controls` and fail loudly. When
    /// `active` is `false`, valid keys are still recorded in
    /// `pgstar_plotting_keys` for `#173` auditing while inactive stale/typo
    /// keys are ignored cleanly (mirroring upstream `if (pgstar_flag)` /
    /// `if (.not. pgbinary_flag) return` gating).
    pub fn validate_headless_plotting_section_with_active<S: AsRef<str>>(
        &mut self,
        section: &str,
        keys: &[S],
        active: bool,
    ) {
        let sec = section.to_ascii_lowercase();
        let valid_keys = upstream_section_keys(&sec);
        for raw_key in keys {
            let k = raw_key.as_ref().to_ascii_lowercase();
            if is_include_key(&sec, &k) {
                continue;
            }
            if valid_keys.is_some_and(|set| set.contains(k.as_str())) {
                self.pgstar_plotting_keys.push((sec.clone(), k));
            } else if active {
                self.unknown_controls.push((sec.clone(), k));
            }
        }
    }

    /// Validates an active headless plotting section (`&pgstar`, `&pgbinary`,
    /// or `&astero_pgstar_controls`) against upstream `26.4.1` defaults.
    pub fn validate_headless_plotting_section<S: AsRef<str>>(
        &mut self,
        section: &str,
        keys: &[S],
    ) {
        self.validate_headless_plotting_section_with_active(
            section, keys, true,
        );
    }

    /// Classifies an unimplemented or unrecognized section name (`other =>`).
    pub fn validate_other_section(&mut self, section: &str) {
        let sec = section.to_ascii_lowercase();
        if is_known_upstream_only_section(&sec)
            || upstream_section_keys(&sec).is_some()
        {
            self.unsupported_sections.push(sec);
        } else {
            self.unknown_sections.push(sec);
        }
    }

    /// Validates keys inside an implemented section (`&star_job`, `&controls`,
    /// `&eos`, `&kap`, `&binary_job`, `&binary_controls`, `&astero_search_controls`).
    pub fn validate_implemented_section<S: AsRef<str>>(
        &mut self,
        section: &str,
        keys: &[S],
        supported: &HashSet<&str>,
        unused_fields: &HashSet<&str>,
        headless_gui_flags: &HashSet<&str>,
    ) {
        self.validate_implemented_section_with_terminal_output(
            section,
            keys,
            supported,
            unused_fields,
            headless_gui_flags,
            &HashSet::new(),
        );
    }

    /// Same as [`Self::validate_implemented_section`], with a set of
    /// `terminal_output` keys that are accepted and recorded in
    /// `terminal_output_keys` instead of being rejected (`DECISIONS #26`).
    ///
    /// A key belongs in `terminal_output` only if every upstream use of it at
    /// MESA `26.4.1` gates or formats a `write` to the terminal.
    pub fn validate_implemented_section_with_terminal_output<S: AsRef<str>>(
        &mut self,
        section: &str,
        keys: &[S],
        supported: &HashSet<&str>,
        unused_fields: &HashSet<&str>,
        headless_gui_flags: &HashSet<&str>,
        terminal_output: &HashSet<&str>,
    ) {
        let sec = section.to_ascii_lowercase();
        let upstream = upstream_section_keys(&sec);
        for raw_key in keys {
            let k = raw_key.as_ref().to_ascii_lowercase();
            if is_include_key(&sec, &k) {
                continue;
            }
            if headless_gui_flags.contains(k.as_str()) {
                self.pgstar_plotting_keys.push((sec.clone(), k));
                continue;
            }
            if terminal_output.contains(k.as_str()) {
                self.terminal_output_keys.push((sec.clone(), k));
                continue;
            }
            if unused_fields.contains(k.as_str()) {
                self.unsupported_controls.push((sec.clone(), k));
                continue;
            }
            if supported.contains(k.as_str()) {
                continue;
            }
            if upstream.is_some_and(|set| set.contains(k.as_str())) {
                self.unsupported_controls.push((sec.clone(), k));
            } else {
                self.unknown_controls.push((sec.clone(), k));
            }
        }
    }

    /// Formats a diagnostic error message when validation fails.
    ///
    /// Unknown/misspelled controls and sections always produce an error.
    /// Unsupported upstream/unwired controls and sections produce an error when
    /// `allow_unsupported_controls` is `false`.
    #[must_use]
    pub fn format_error(
        &self,
        allow_unsupported_controls: bool,
    ) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();

        let unknown_secs: BTreeSet<&str> =
            self.unknown_sections.iter().map(String::as_str).collect();
        if !unknown_secs.is_empty() {
            let list: Vec<String> =
                unknown_secs.iter().map(|s| format!("&{s}")).collect();
            parts.push(format!(
                "unknown namelist section(s): {}",
                list.join(", ")
            ));
        }

        let unknown_ctrls: BTreeSet<String> = self
            .unknown_controls
            .iter()
            .map(|(s, k)| format!("&{s}::{k}"))
            .collect();
        if !unknown_ctrls.is_empty() {
            parts.push(format_key_sample(
                "unknown/misspelled control(s)",
                &unknown_ctrls,
            ));
        }

        if !allow_unsupported_controls {
            let unsup_secs: BTreeSet<&str> = self
                .unsupported_sections
                .iter()
                .map(String::as_str)
                .collect();
            if !unsup_secs.is_empty() {
                let list: Vec<String> =
                    unsup_secs.iter().map(|s| format!("&{s}")).collect();
                parts.push(format!(
                    "unsupported upstream namelist section(s): {}",
                    list.join(", ")
                ));
            }

            let unsup_ctrls: BTreeSet<String> = self
                .unsupported_controls
                .iter()
                .map(|(s, k)| format!("&{s}::{k}"))
                .collect();
            if !unsup_ctrls.is_empty() {
                parts.push(format_key_sample(
                    "unsupported control(s)",
                    &unsup_ctrls,
                ));
            }
        }

        if parts.is_empty() {
            None
        } else if !allow_unsupported_controls
            && (!self.unsupported_controls.is_empty()
                || !self.unsupported_sections.is_empty())
        {
            Some(format!(
                "{} (pass --allow-unsupported-controls / allow_unsupported_controls=true to opt out)",
                parts.join("; ")
            ))
        } else {
            Some(parts.join("; "))
        }
    }

    /// Converts the report into `Ok(())` or [`SchemaError::ValidationError`].
    ///
    /// # Errors
    /// Returns [`SchemaError::ValidationError`] when unknown keys/sections are
    /// present or when unsupported keys/sections are present with
    /// `allow_unsupported_controls == false`.
    pub fn into_result(
        &self,
        allow_unsupported_controls: bool,
    ) -> Result<(), SchemaError> {
        if let Some(message) = self.format_error(allow_unsupported_controls) {
            return Err(SchemaError::ValidationError {
                control: String::from("<inlist>"),
                line_number: 1,
                message,
            });
        }
        if allow_unsupported_controls {
            if let Some(message) = self.format_error(false) {
                eprintln!(
                    "warning: --allow-unsupported-controls active; \
                     continuing past: {message}"
                );
            }
        }
        Ok(())
    }
}

fn format_key_sample(label: &str, items: &BTreeSet<String>) -> String {
    const MAX_SHOWN: usize = 20;
    let shown: Vec<&str> =
        items.iter().take(MAX_SHOWN).map(String::as_str).collect();
    if items.len() > MAX_SHOWN {
        format!(
            "{label} ({} total, showing first {MAX_SHOWN}): {}, ...",
            items.len(),
            shown.join(", ")
        )
    } else {
        format!("{label} ({}): {}", items.len(), shown.join(", "))
    }
}

/// Formats a [`ControlValue`] as a Fortran namelist literal.
#[must_use]
pub fn format_fortran_value(v: &ControlValue) -> String {
    match v {
        ControlValue::Logical(b) => {
            String::from(if *b { ".true." } else { ".false." })
        }
        ControlValue::Integer(i) => i.to_string(),
        ControlValue::Float(x) => format!("{x:?}"),
        ControlValue::String(s) => format!("'{}'", s.replace('\'', "''")),
    }
}

/// Copies an upstream MESA test case directory from `src_case` to `dst_case`
/// (skipping root `LOGS`/`photos`) and rewrites `inlist*` files keeping only
/// sections accepted by `keep_section` and assignments accepted by `keep_assign`.
///
/// # Errors
/// Returns [`SchemaError`] if reading, parsing, or writing files fails.
pub fn copy_cleaned_case_dir<FS, FK>(
    src_case: &Path,
    dst_case: &Path,
    keep_section: &FS,
    keep_assign: &FK,
) -> Result<(), SchemaError>
where
    FS: Fn(&str) -> bool,
    FK: Fn(&str, &str) -> bool,
{
    fn copy_rec<FS, FK>(
        src: &Path,
        dst: &Path,
        is_root: bool,
        keep_section: &FS,
        keep_assign: &FK,
    ) -> Result<(), SchemaError>
    where
        FS: Fn(&str) -> bool,
        FK: Fn(&str, &str) -> bool,
    {
        fs::create_dir_all(dst).map_err(|e| SchemaError::IoError {
            message: format!("failed to create '{}': {e}", dst.display()),
        })?;
        let entries = fs::read_dir(src).map_err(|e| SchemaError::IoError {
            message: format!("failed to read '{}': {e}", src.display()),
        })?;
        for ent in entries {
            let ent = ent.map_err(|e| SchemaError::IoError {
                message: format!("dir entry error in '{}': {e}", src.display()),
            })?;
            let name = ent.file_name();
            let name_str = name.to_string_lossy();
            if is_root && (name_str == "LOGS" || name_str == "photos") {
                continue;
            }
            let sp = ent.path();
            let dp = dst.join(&name);
            let ftype = ent.file_type().map_err(|e| SchemaError::IoError {
                message: format!(
                    "file_type failed for '{}': {e}",
                    sp.display()
                ),
            })?;
            if ftype.is_dir() {
                copy_rec(&sp, &dp, false, keep_section, keep_assign)?;
            } else if is_root
                && name_str.starts_with("inlist")
                && name_str != "inlist_multi_stars_job"
            {
                let blocks = parse_namelist_file(&sp)?;
                let mut out = String::new();
                for blk in &blocks {
                    let sec = blk.name.to_ascii_lowercase();
                    if !keep_section(&sec) {
                        continue;
                    }
                    let _ = writeln!(out, "&{}", blk.name);
                    for a in &blk.assignments {
                        let key = a.name.to_ascii_lowercase();
                        if !keep_assign(&sec, &key) {
                            continue;
                        }
                        let _ = write!(out, "  {}", a.name);
                        if let Some(ref sub) = a.subscript {
                            let _ = write!(out, "({sub})");
                        }
                        let _ = writeln!(
                            out,
                            " = {}",
                            format_fortran_value(&a.value)
                        );
                    }
                    out.push_str("/\n");
                }
                fs::write(&dp, out).map_err(|e| SchemaError::IoError {
                    message: format!("failed to write '{}': {e}", dp.display()),
                })?;
            } else {
                fs::copy(&sp, &dp).map_err(|e| SchemaError::IoError {
                    message: format!(
                        "failed to copy '{}' -> '{}': {e}",
                        sp.display(),
                        dp.display()
                    ),
                })?;
            }
        }
        Ok(())
    }

    copy_rec(src_case, dst_case, true, keep_section, keep_assign)
}
