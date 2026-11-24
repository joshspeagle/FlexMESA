// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream MESA `./rn` shell script stage parser (`RnStage`,
//! `parse_rn_stages`).
//!
//! Resolves `do_one` / `do_one_binary`, `cp`, `MESA_SKIP_OPTIONAL` /
//! `MESA_RUN_OPTIONAL` conditionals, `./rn1` / `./star` / `./binary`, and
//! `inlist_multi_stars_job` expansion cleanly for single-star, binary, and
//! astero suites.

use std::fs;
use std::path::Path;

use flexmesa_schema::parse_namelist_file;

use crate::controls::StarError;
use crate::inlist::SectionControls;

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
/// Returns [`StarError::IoError`] if `case_dir/rn` cannot be read, or
/// [`StarError::InvalidInput`] if no runnable stages exist or
/// `inlist_multi_stars_job` fails validation.
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
    let mut skip_if_is_nonempty = true;

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("if ")
            && (line.contains("MESA_SKIP_OPTIONAL")
                || line.contains("MESA_RUN_OPTIONAL"))
        {
            in_skip_if = true;
            in_else = false;
            let has_z = line.contains("-z");
            skip_if_is_nonempty = if line.contains("MESA_RUN_OPTIONAL") {
                has_z
            } else {
                !has_z
            };
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
            let cond_true = if skip_if_is_nonempty {
                skip_optional
            } else {
                !skip_optional
            };
            let active_branch =
                (cond_true && !in_else) || (!cond_true && in_else);
            if !active_branch {
                continue;
            }
        }

        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        if toks[0] == "return" || toks[0] == "exit" {
            break;
        }
        if toks[0] == "do_one" || toks[0] == "do_one_binary" {
            let inlist = toks
                .get(1)
                .map_or_else(|| String::from("inlist"), |s| (*s).to_string());
            let output_mod = toks.get(2).map(|s| (*s).to_string());
            let logs_dir = toks.get(3).map(|s| (*s).to_string());
            if inlist == "inlist"
                && !case_dir.join("inlist").exists()
                && case_dir.join("inlist_multi_stars_job").exists()
            {
                let ms_path = case_dir.join("inlist_multi_stars_job");
                let blocks = parse_namelist_file(&ms_path).map_err(|e| {
                    StarError::IoError(format!(
                        "failed parsing '{}': {e}",
                        ms_path.display()
                    ))
                })?;
                let mut sec = SectionControls::default();
                for b in &blocks {
                    for a in &b.assignments {
                        sec.insert(a);
                    }
                }
                let names = sec.try_get_indexed_strings(
                    "multi_stars_job",
                    "inlist_names",
                )?;
                if names.is_empty() {
                    return Err(StarError::InvalidInput(format!(
                        "no 'inlist_names' entries found in '{}'",
                        ms_path.display()
                    )));
                }
                for n in names {
                    stages.push(RnStage::DoOne {
                        inlist: n,
                        output_mod: None,
                        logs_dir: None,
                    });
                }
            } else {
                stages.push(RnStage::DoOne {
                    inlist,
                    output_mod,
                    logs_dir,
                });
            }
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
        return Err(StarError::InvalidInput(format!(
            "no runnable stages parsed from '{}'",
            rn_path.display()
        )));
    }
    Ok(stages)
}
