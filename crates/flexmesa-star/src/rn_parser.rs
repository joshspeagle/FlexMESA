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

#[derive(Clone, Copy)]
enum ShellQuote {
    None,
    Single,
    Double,
}

#[derive(Clone, Copy)]
struct IfFrame {
    parent_active: bool,
    any_taken: bool,
    current_active: Option<bool>,
}

impl IfFrame {
    fn is_active(self) -> bool {
        self.parent_active && self.current_active.unwrap_or(true)
    }
}

fn join_continued_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for raw in text.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if let Some(prefix) = line.strip_suffix('\\') {
            cur.push_str(prefix);
        } else {
            cur.push_str(line);
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn split_shell_statements(line: &str) -> Vec<Vec<String>> {
    let mut stmts: Vec<Vec<String>> = Vec::new();
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut has_tok = false;
    let mut quote = ShellQuote::None;
    let mut escaped = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if escaped {
            cur.push(ch);
            has_tok = true;
            escaped = false;
            continue;
        }
        match quote {
            ShellQuote::Single => {
                if ch == '\'' {
                    quote = ShellQuote::None;
                } else {
                    cur.push(ch);
                }
                has_tok = true;
            }
            ShellQuote::Double => {
                if ch == '\\' {
                    if matches!(chars.peek(), Some('"' | '\\' | '$')) {
                        escaped = true;
                    } else {
                        cur.push(ch);
                    }
                } else if ch == '"' {
                    quote = ShellQuote::None;
                } else {
                    cur.push(ch);
                }
                has_tok = true;
            }
            ShellQuote::None => match ch {
                '#' if !has_tok => break,
                ';' => {
                    if has_tok {
                        toks.push(std::mem::take(&mut cur));
                        has_tok = false;
                    }
                    if !toks.is_empty() {
                        stmts.push(std::mem::take(&mut toks));
                    }
                }
                '\\' => {
                    escaped = true;
                    has_tok = true;
                }
                '\'' => {
                    quote = ShellQuote::Single;
                    has_tok = true;
                }
                '"' => {
                    quote = ShellQuote::Double;
                    has_tok = true;
                }
                c if c.is_whitespace() => {
                    if has_tok {
                        toks.push(std::mem::take(&mut cur));
                        has_tok = false;
                    }
                }
                _ => {
                    cur.push(ch);
                    has_tok = true;
                }
            },
        }
    }
    if has_tok {
        toks.push(cur);
    }
    if !toks.is_empty() {
        stmts.push(toks);
    }
    stmts
}

fn eval_optional_condition(
    toks: &[String],
    skip_optional: bool,
) -> Option<bool> {
    let has_skip = toks.iter().any(|t| t.contains("MESA_SKIP_OPTIONAL"));
    let has_run = toks.iter().any(|t| t.contains("MESA_RUN_OPTIONAL"));
    if !has_skip && !has_run {
        return None;
    }
    let has_z = toks.iter().any(|t| t == "-z");
    let negated = toks.iter().filter(|t| t.as_str() == "!").count() % 2 == 1;
    let var_nonempty = if has_run {
        !skip_optional
    } else {
        skip_optional
    };
    let base = if has_z { !var_nonempty } else { var_nonempty };
    Some(if negated { !base } else { base })
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
    let mut if_stack: Vec<IfFrame> = Vec::new();
    let logical_lines = join_continued_lines(&text);

    'lines: for line in logical_lines {
        for toks in split_shell_statements(&line) {
            if toks.is_empty() {
                continue;
            }
            let cur_active =
                if_stack.last().copied().is_none_or(IfFrame::is_active);
            match toks[0].as_str() {
                "if" => {
                    let branch = eval_optional_condition(&toks, skip_optional);
                    if_stack.push(IfFrame {
                        parent_active: cur_active,
                        any_taken: branch == Some(true),
                        current_active: branch,
                    });
                    continue;
                }
                "elif" => {
                    if let Some(top) = if_stack.last_mut() {
                        if top.any_taken {
                            top.current_active = Some(false);
                        } else if top.current_active.is_some() {
                            match eval_optional_condition(&toks, skip_optional)
                            {
                                Some(true) => {
                                    top.any_taken = true;
                                    top.current_active = Some(true);
                                }
                                Some(false) => {
                                    top.current_active = Some(false);
                                }
                                None => {
                                    top.any_taken = true;
                                    top.current_active = None;
                                }
                            }
                        }
                    }
                    continue;
                }
                "else" => {
                    if let Some(top) = if_stack.last_mut() {
                        if top.any_taken {
                            top.current_active = Some(false);
                        } else if let Some(cur) = top.current_active {
                            let next = !cur;
                            top.any_taken |= next;
                            top.current_active = Some(next);
                        }
                    }
                    continue;
                }
                "fi" => {
                    let _ = if_stack.pop();
                    continue;
                }
                "then" => continue,
                _ => {}
            }
            if !cur_active {
                continue;
            }
            if toks[0] == "return" || toks[0] == "exit" {
                break 'lines;
            }
            if toks[0] == "do_one" || toks[0] == "do_one_binary" {
                let inlist = toks
                    .get(1)
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .unwrap_or_else(|| String::from("inlist"));
                let output_mod = toks.get(2).filter(|s| !s.is_empty()).cloned();
                let logs_dir = toks.get(3).filter(|s| !s.is_empty()).cloned();
                if inlist == "inlist"
                    && !case_dir.join("inlist").exists()
                    && case_dir.join("inlist_multi_stars_job").exists()
                {
                    let ms_path = case_dir.join("inlist_multi_stars_job");
                    let blocks =
                        parse_namelist_file(&ms_path).map_err(|e| {
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
            } else if toks[0] == "cp" {
                let mut operands = Vec::new();
                let mut after_sep = false;
                for arg in &toks[1..] {
                    if !after_sep && arg == "--" {
                        after_sep = true;
                        continue;
                    }
                    if !after_sep && arg.starts_with('-') && arg.len() > 1 {
                        continue;
                    }
                    operands.push(arg);
                }
                if operands.len() >= 2 {
                    stages.push(RnStage::CopyCheckpoint {
                        src_mod: operands[0].clone(),
                        dst_mod: operands[1].clone(),
                    });
                }
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
    }

    if stages.is_empty() {
        return Err(StarError::InvalidInput(format!(
            "no runnable stages parsed from '{}'",
            rn_path.display()
        )));
    }
    Ok(stages)
}
