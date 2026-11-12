// SPDX-License-Identifier: LGPL-3.0-or-later
//! Read-only RAII temporary case directory staging (`TempCaseStaging`) so
//! upstream `$MESA_DIR` `test_suite` and `dev_cases_*` directories are never
//! mutated by `.mod`, `lina.data`, or relaxation profile outputs during parallel
//! test execution.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::controls::StarError;

static STAGING_SEQ: AtomicU64 = AtomicU64::new(0);

/// RAII temporary workspace staging a single `test_suite` case directory under
/// `std::env::temp_dir()` while preserving relative `../` and `../../` sibling
/// lookups (`../shared`, `../../rsp2_utils`, `../../test_suite`).
#[derive(Debug)]
pub(crate) struct TempCaseStaging {
    tmp_root: Option<PathBuf>,
    /// Staged writable case directory whose basename equals `orig_case_dir`.
    pub staged_dir: PathBuf,
}

fn is_readonly_mesa_case_dir(orig_case_dir: &Path) -> bool {
    if orig_case_dir
        .components()
        .any(|c| c.as_os_str() == "mesa_test_inputs")
    {
        return true;
    }
    if let Ok(mesa_env) = std::env::var("MESA_DIR") {
        let trimmed = mesa_env.trim();
        if !trimmed.is_empty() {
            let mesa_path = Path::new(trimmed);
            if orig_case_dir.starts_with(mesa_path) {
                return true;
            }
            if let (Ok(canon_case), Ok(canon_mesa)) =
                (fs::canonicalize(orig_case_dir), fs::canonicalize(mesa_path))
            {
                if canon_case.starts_with(&canon_mesa) {
                    return true;
                }
            }
        }
    }
    false
}

impl TempCaseStaging {
    /// Copies `orig_case_dir` into a fresh process- and thread-unique temporary
    /// directory when `orig_case_dir` resides inside read-only `$MESA_DIR` /
    /// `mesa_test_inputs`, or executes in place for user-owned directories.
    pub fn new(orig_case_dir: &Path) -> Result<Self, StarError> {
        if !is_readonly_mesa_case_dir(orig_case_dir) {
            return Ok(Self {
                tmp_root: None,
                staged_dir: orig_case_dir.to_path_buf(),
            });
        }
        let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let case_name = orig_case_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("case");
        let parent_name = orig_case_dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("suite");

        let tmp_root = std::env::temp_dir()
            .join(format!("flexmesa_case_{pid}_{seq}_{case_name}"));
        let staged_gparent = tmp_root.join("g");
        let staged_parent = staged_gparent.join(parent_name);
        let staged_dir = staged_parent.join(case_name);

        fs::create_dir_all(&staged_dir).map_err(|e| {
            StarError::IoError(format!(
                "failed to create staged case dir {}: {e}",
                staged_dir.display()
            ))
        })?;

        if let Some(orig_parent) = orig_case_dir.parent() {
            if let Some(orig_gparent) = orig_parent.parent() {
                link_sibling_entries(
                    orig_gparent,
                    &staged_gparent,
                    parent_name,
                )?;
            }
            link_sibling_entries(orig_parent, &staged_parent, case_name)?;
        }

        copy_dir_tree(orig_case_dir, &staged_dir, 0)?;

        Ok(Self {
            tmp_root: Some(tmp_root),
            staged_dir,
        })
    }
}

impl Drop for TempCaseStaging {
    fn drop(&mut self) {
        if let Some(root) = &self.tmp_root {
            let _ = fs::remove_dir_all(root);
        }
    }
}

fn copy_dir_tree(
    src: &Path,
    dst: &Path,
    depth: usize,
) -> Result<(), StarError> {
    if depth > 16 {
        return Err(StarError::IoError(format!(
            "exceeded symlink/directory nesting limit while staging {}",
            src.display()
        )));
    }
    fs::create_dir_all(dst).map_err(|e| {
        StarError::IoError(format!("mkdir {}: {e}", dst.display()))
    })?;
    let entries = fs::read_dir(src).map_err(|e| {
        StarError::IoError(format!("read_dir {}: {e}", src.display()))
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| StarError::IoError(e.to_string()))?;
        let name = entry.file_name();
        if name == "docs" || name == "LOGS" || name == "photos" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        let ft = entry
            .file_type()
            .map_err(|e| StarError::IoError(e.to_string()))?;
        if ft.is_dir() {
            copy_dir_tree(&src_path, &dst_path, depth + 1)?;
        } else if ft.is_file() {
            fs::copy(&src_path, &dst_path).map_err(|e| {
                StarError::IoError(format!(
                    "copy {} -> {}: {e}",
                    src_path.display(),
                    dst_path.display()
                ))
            })?;
        } else if ft.is_symlink() {
            let target_meta = fs::metadata(&src_path).map_err(|e| {
                StarError::IoError(format!(
                    "broken symlink {}: {e}",
                    src_path.display()
                ))
            })?;
            if target_meta.is_dir() {
                copy_dir_tree(&src_path, &dst_path, depth + 1)?;
            } else if target_meta.is_file() {
                fs::copy(&src_path, &dst_path).map_err(|e| {
                    StarError::IoError(format!(
                        "copy symlink target {} -> {}: {e}",
                        src_path.display(),
                        dst_path.display()
                    ))
                })?;
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn link_sibling_entries(
    orig_dir: &Path,
    staged_dir: &Path,
    skip_name: &str,
) -> Result<(), StarError> {
    let Ok(entries) = fs::read_dir(orig_dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_str() == Some(skip_name) {
            continue;
        }
        let dst = staged_dir.join(&name);
        if !dst.exists() {
            let src =
                fs::canonicalize(entry.path()).unwrap_or_else(|_| entry.path());
            std::os::unix::fs::symlink(&src, &dst).map_err(|e| {
                StarError::IoError(format!(
                    "symlink {} -> {}: {e}",
                    dst.display(),
                    src.display()
                ))
            })?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn link_sibling_entries(
    orig_dir: &Path,
    staged_dir: &Path,
    skip_name: &str,
) -> Result<(), StarError> {
    for rel in ["shared", "rsp2_utils"] {
        if rel == skip_name {
            continue;
        }
        let src = orig_dir.join(rel);
        let dst = staged_dir.join(rel);
        if src.is_dir() && !dst.exists() {
            copy_dir_tree(&src, &dst, 0)?;
        }
    }
    Ok(())
}
