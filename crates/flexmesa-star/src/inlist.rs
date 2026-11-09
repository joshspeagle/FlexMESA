// SPDX-License-Identifier: LGPL-3.0-or-later
//! Recursive multi-section Fortran `inlist` parser (`InlistBundle`) and
//! upstream MESA `./rn` shell script stage parser (`RnStage`, `parse_rn_stages`).
//!
//! Resolves `read_extra_<section>_inlist(1..5)` and
//! `extra_<section>_inlist_name(1..5)` (along with legacy un-subscripted
//! `read_extra_<section>_inlist1..5`) across `&star_job`, `&eos`, `&kap`,
//! `&controls`, `&pgstar`, `&binary_job`, `&binary_controls`, `&pgbinary`,
//! `&astero_search_controls`, and `&astero_pgstar_controls`, and maps parsed
//! values into [`StarJobControls`], [`StarControls`], [`EosConfig`], and
//! [`KapConfig`].

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_eos::EosConfig;
use flexmesa_kap::KapConfig;
use flexmesa_schema::{parse_namelist_file, ControlValue, NamelistAssignment};

use crate::controls::{StarControls, StarError, StarJobControls};

/// Flattened key-value and indexed-element store for a single namelist section.
#[derive(Debug, Clone, Default)]
pub struct SectionControls {
    /// Scalar `name -> ControlValue` assignments (lowercased keys).
    pub scalars: HashMap<String, ControlValue>,
    /// Indexed `name -> [(subscript, ControlValue)]` assignments (lowercased keys).
    pub indexed: HashMap<String, Vec<(String, ControlValue)>>,
}

impl SectionControls {
    /// Records a [`NamelistAssignment`] in `scalars` or `indexed`.
    pub fn insert(&mut self, assign: &NamelistAssignment) {
        let key = assign.name.to_ascii_lowercase();
        if let Some(sub) = &assign.subscript {
            let entry = self.indexed.entry(key).or_default();
            if let Some(existing) = entry.iter_mut().find(|(s, _)| s == sub) {
                existing.1 = assign.value.clone();
            } else {
                entry.push((sub.clone(), assign.value.clone()));
            }
        } else {
            self.scalars.insert(key, assign.value.clone());
        }
    }

    /// Returns a boolean scalar if present.
    #[must_use]
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.scalars.get(&name.to_ascii_lowercase())? {
            ControlValue::Logical(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns an `i64` scalar if present.
    #[must_use]
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        match self.scalars.get(&name.to_ascii_lowercase())? {
            ControlValue::Integer(i) => Some(*i),
            ControlValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Returns an `f64` scalar if present.
    #[must_use]
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        match self.scalars.get(&name.to_ascii_lowercase())? {
            ControlValue::Float(f) => Some(*f),
            ControlValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Returns a string scalar if present.
    #[must_use]
    pub fn get_str(&self, name: &str) -> Option<&str> {
        match self.scalars.get(&name.to_ascii_lowercase())? {
            ControlValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns an indexed boolean `name(idx)` or legacy `name<idx>` if present.
    #[must_use]
    pub fn get_indexed_bool(
        &self,
        base_name: &str,
        idx: usize,
    ) -> Option<bool> {
        let key = base_name.to_ascii_lowercase();
        let idx_str = idx.to_string();
        if let Some(list) = self.indexed.get(&key) {
            if let Some((_, ControlValue::Logical(b))) =
                list.iter().find(|(s, _)| s.trim() == idx_str)
            {
                return Some(*b);
            }
        }
        self.get_bool(&format!("{key}{idx}"))
    }

    /// Returns an indexed string `name_name(idx)` or legacy `name<idx>_name`.
    #[must_use]
    pub fn get_indexed_inlist_name(
        &self,
        section: &str,
        idx: usize,
    ) -> Option<&str> {
        let sec = section.to_ascii_lowercase();
        let key = format!("extra_{sec}_inlist_name");
        let idx_str = idx.to_string();
        if let Some(list) = self.indexed.get(&key) {
            if let Some((_, ControlValue::String(s))) =
                list.iter().find(|(s, _)| s.trim() == idx_str)
            {
                return Some(s.as_str());
            }
        }
        self.get_str(&format!("extra_{sec}_inlist{idx}_name"))
            .or_else(|| self.get_str(&key))
    }

    /// Returns indexed string assignments sorted by integer subscript.
    #[must_use]
    pub fn get_indexed_strings(&self, key: &str) -> Vec<String> {
        let Some(list) = self.indexed.get(&key.to_ascii_lowercase()) else {
            return Vec::new();
        };
        let mut pairs: Vec<(usize, String)> = list
            .iter()
            .enumerate()
            .filter_map(|(pos, (sub, val))| {
                let idx = sub.trim().parse::<usize>().unwrap_or(pos + 1);
                if let ControlValue::String(s) = val {
                    Some((idx, s.clone()))
                } else {
                    None
                }
            })
            .collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        pairs.into_iter().map(|(_, s)| s).collect()
    }

    /// Returns indexed `f64` assignments sorted by integer subscript.
    #[must_use]
    pub fn get_indexed_f64s(&self, key: &str) -> Vec<f64> {
        let Some(list) = self.indexed.get(&key.to_ascii_lowercase()) else {
            return Vec::new();
        };
        let mut pairs: Vec<(usize, f64)> = list
            .iter()
            .enumerate()
            .filter_map(|(pos, (sub, val))| {
                let idx = sub.trim().parse::<usize>().unwrap_or(pos + 1);
                match val {
                    ControlValue::Float(f) => Some((idx, *f)),
                    ControlValue::Integer(i) => Some((idx, *i as f64)),
                    _ => None,
                }
            })
            .collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        pairs.into_iter().map(|(_, v)| v).collect()
    }
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
    /// Raw flattened assignments for every namelist section (`star_job`, `eos`,
    /// `kap`, `controls`, `pgstar`, `binary_job`, `binary_controls`, `pgbinary`,
    /// `astero_search_controls`, `simplex_search_controls`, etc.).
    pub sections: HashMap<String, SectionControls>,
}

impl InlistBundle {
    /// Parses `inlist_path` and recursively resolves all `read_extra_*_inlist`
    /// includes relative to `inlist_path.parent()`.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if a required `inlist` file cannot be
    /// read or parsed.
    pub fn from_file(inlist_path: &Path) -> Result<Self, StarError> {
        let case_dir = inlist_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let entry_inlist = inlist_path.file_name().map_or_else(
            || String::from("inlist"),
            |s| s.to_string_lossy().into_owned(),
        );
        let mut sections: HashMap<String, SectionControls> = HashMap::new();
        let mut visited: HashSet<(String, PathBuf)> = HashSet::new();

        Self::load_file_recursive(
            &case_dir,
            inlist_path,
            None,
            &mut sections,
            &mut visited,
        )?;

        let mut bundle = Self {
            case_dir,
            entry_inlist,
            job: StarJobControls::default(),
            controls: StarControls::default(),
            eos_config: EosConfig::default(),
            kap_config: KapConfig::default(),
            sections,
        };
        bundle.populate_typed_structs();
        Ok(bundle)
    }

    fn load_file_recursive(
        case_dir: &Path,
        file_path: &Path,
        target_section: Option<&str>,
        sections: &mut HashMap<String, SectionControls>,
        visited: &mut HashSet<(String, PathBuf)>,
    ) -> Result<(), StarError> {
        let canon = file_path
            .canonicalize()
            .unwrap_or_else(|_| file_path.to_path_buf());
        let sec_tag = target_section.unwrap_or("*").to_ascii_lowercase();
        if !visited.insert((sec_tag, canon)) {
            return Ok(());
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

            let extra_prefix = match bname.as_str() {
                "simplex_search_controls" => "simplex_search",
                other => other,
            };
            let is_optional_gui = matches!(
                bname.as_str(),
                "pgstar" | "pgbinary" | "astero_pgstar_controls"
            );
            for idx in 1..=5 {
                let read_flag = local_sec
                    .get_indexed_bool(
                        &format!("read_extra_{extra_prefix}_inlist"),
                        idx,
                    )
                    .unwrap_or(false);
                if !read_flag {
                    continue;
                }
                if let Some(extra_name) =
                    local_sec.get_indexed_inlist_name(extra_prefix, idx)
                {
                    let trimmed = extra_name.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let next_path = case_dir.join(trimmed);
                    if !next_path.exists() {
                        if is_optional_gui || trimmed == "inlist_nzhi" {
                            continue;
                        }
                        return Err(StarError::IoError(format!(
                            "missing extra_{extra_prefix}_inlist({idx}) = '{trimmed}' \
                             referenced from '{}'",
                            file_path.display()
                        )));
                    }
                    Self::load_file_recursive(
                        case_dir,
                        &next_path,
                        Some(&bname),
                        sections,
                        visited,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Single execution stage parsed from an upstream MESA `./rn` bash script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RnStage {
    /// Execute `do_one <inlist> [output_mod] [logs_dir]`.
    DoOne {
        /// Entry `inlist` filename relative to the case directory.
        inlist: String,
        /// Optional expected output `.mod` filename.
        output_mod: Option<String>,
        /// Optional `LOGS` destination directory name.
        logs_dir: Option<String>,
    },
    /// Copy a pre-computed checkpoint (`cp standard_*.mod *.mod`).
    CopyCheckpoint {
        /// Source `.mod` file in the case directory.
        src_mod: String,
        /// Destination `.mod` file in the case directory.
        dst_mod: String,
    },
}

/// Parses an upstream MESA `./rn` bash script in `case_dir` into an ordered
/// sequence of [`RnStage`] steps, honoring `skip_optional`
/// (`MESA_SKIP_OPTIONAL`).
///
/// # Errors
/// Returns [`StarError::IoError`] if `case_dir/rn` cannot be read.
pub fn parse_rn_stages(
    case_dir: &Path,
    skip_optional: bool,
) -> Result<Vec<RnStage>, StarError> {
    let rn_path = case_dir.join("rn");
    if !rn_path.exists() {
        let default_inlist = if case_dir.join("inlist_project").exists() {
            String::from("inlist_project")
        } else {
            String::from("inlist")
        };
        return Ok(vec![RnStage::DoOne {
            inlist: default_inlist,
            output_mod: None,
            logs_dir: None,
        }]);
    }
    let text = fs::read_to_string(&rn_path).map_err(|e| {
        StarError::IoError(format!(
            "failed to read '{}': {e}",
            rn_path.display()
        ))
    })?;

    let mut stages = Vec::new();
    let mut in_skip_if = false;
    let mut in_else = false;

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("if ") && line.contains("MESA_SKIP_OPTIONAL") {
            in_skip_if = true;
            in_else = false;
            continue;
        }
        if in_skip_if && (line == "else" || line.starts_with("else ")) {
            in_else = true;
            continue;
        }
        if in_skip_if && (line == "fi" || line.starts_with("fi ")) {
            in_skip_if = false;
            in_else = false;
            continue;
        }
        if in_skip_if {
            let active_branch =
                (skip_optional && !in_else) || (!skip_optional && in_else);
            if !active_branch {
                continue;
            }
        }

        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        if toks[0] == "do_one" || toks[0] == "do_one_binary" {
            let inlist = toks
                .get(1)
                .map_or_else(|| String::from("inlist"), |s| (*s).to_string());
            let output_mod = toks.get(2).map(|s| (*s).to_string());
            let logs_dir = toks.get(3).map(|s| (*s).to_string());
            stages.push(RnStage::DoOne {
                inlist,
                output_mod,
                logs_dir,
            });
        } else if toks[0] == "cp" && toks.len() >= 3 {
            stages.push(RnStage::CopyCheckpoint {
                src_mod: toks[1].to_string(),
                dst_mod: toks[2].to_string(),
            });
        } else if toks[0] == "./rn1"
            || toks[0] == "./star"
            || toks[0] == "./binary"
        {
            let inlist = if case_dir.join("inlist_project").exists()
                && !case_dir.join("inlist").exists()
            {
                String::from("inlist_project")
            } else {
                String::from("inlist")
            };
            stages.push(RnStage::DoOne {
                inlist,
                output_mod: None,
                logs_dir: None,
            });
        }
    }

    if stages.is_empty() {
        stages.push(RnStage::DoOne {
            inlist: String::from("inlist"),
            output_mod: None,
            logs_dir: None,
        });
    }
    Ok(stages)
}
