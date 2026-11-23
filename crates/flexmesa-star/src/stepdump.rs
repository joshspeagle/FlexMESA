// SPDX-License-Identifier: LGPL-3.0-or-later
//! Opt-in solver step dumps in the record layout of the instrumented MESA
//! 26.4.1 overlay (`benchmarks/fortran_mesa/stepdumps/stepdump_patch.py`,
//! `#372`), so that a FlexMESA run can be compared record by record with
//! MESA's.
//!
//! Disabled unless the environment variable `FLEXMESA_STEPDUMP` names an
//! output file. Records are written for the first `FLEXMESA_STEPDUMP_STEPS`
//! (default 3) distinct model numbers of each star, from model number
//! `FLEXMESA_STEPDUMP_FIRST` on when that is set (a window later in the run). The dump only reads
//! solver state: it changes no value, branch or order of arithmetic.
//!
//! Every floating-point value is written as the upper-case hexadecimal of its
//! IEEE-754 binary64 bits (Fortran `Z16.16`); zone, iteration and equation
//! indices are 1-based, as upstream prints them. The star id follows MESA's
//! `s% id` (1 for a single star or a binary's donor, 2 for its accretor) and
//! is set by the caller with [`with_star_id`].

use std::cell::Cell;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

const MAX_IDS: usize = 16;

thread_local! {
    static CURRENT_ID: Cell<usize> = const { Cell::new(1) };
}

struct DumpState {
    file: File,
    nmax: usize,
    first_model: usize,
    nseen: [usize; MAX_IDS],
    last_model: [Option<usize>; MAX_IDS],
    named: [bool; MAX_IDS],
    /// Star has had its `nmax` models; nothing more is written for it, as in
    /// the MESA overlay. A relax restarts at model 0, so a later step can
    /// repeat the last dumped model number.
    finished: [bool; MAX_IDS],
    done: bool,
}

static DUMP: OnceLock<Option<Mutex<DumpState>>> = OnceLock::new();

fn dump() -> Option<&'static Mutex<DumpState>> {
    DUMP.get_or_init(|| {
        let path = std::env::var("FLEXMESA_STEPDUMP").ok()?;
        if path.trim().is_empty() {
            return None;
        }
        let nmax = std::env::var("FLEXMESA_STEPDUMP_STEPS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(3);
        let first_model = std::env::var("FLEXMESA_STEPDUMP_FIRST")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.trim())
            .ok()?;
        let _ = writeln!(file, "RUN");
        Some(Mutex::new(DumpState {
            file,
            nmax,
            first_model,
            nseen: [0; MAX_IDS],
            last_model: [None; MAX_IDS],
            named: [false; MAX_IDS],
            finished: [false; MAX_IDS],
            done: false,
        }))
    })
    .as_ref()
}

/// Returns `true` when `FLEXMESA_STEPDUMP` is set, so callers can skip
/// assembling record arguments otherwise.
#[must_use]
pub fn active() -> bool {
    dump().is_some()
}

/// Runs `f` with the dump's star id set to `id` (MESA `s% id`).
pub fn with_star_id<R>(id: usize, f: impl FnOnce() -> R) -> R {
    let prev = CURRENT_ID.with(|c| c.replace(id));
    let out = f();
    CURRENT_ID.with(|c| c.set(prev));
    out
}

/// Decides whether a record of `model` is written and passes the open file.
///
/// `f` receives the file, the star id and whether this is the first record
/// of this model for this star. Returns without calling `f` when the dump is
/// off or this star has already had its `FLEXMESA_STEPDUMP_STEPS` models.
pub fn record(
    model: usize,
    names: impl FnOnce() -> (Vec<String>, Vec<String>, usize, usize, usize),
    f: impl FnOnce(&mut String, usize, bool),
) {
    let Some(mtx) = dump() else { return };
    let id = CURRENT_ID.with(Cell::get);
    let Ok(mut st) = mtx.lock() else { return };
    if id == 0 || id > MAX_IDS {
        return;
    }
    if model < st.first_model {
        return;
    }
    let slot = id - 1;
    if st.finished[slot] {
        return;
    }
    let mut new_model = false;
    if st.last_model[slot] != Some(model) {
        if st.nseen[slot] >= st.nmax {
            st.finished[slot] = true;
            if !st.done {
                st.done = true;
                let _ = writeln!(st.file, "DONE");
                let _ = st.file.flush();
            }
            return;
        }
        st.nseen[slot] += 1;
        st.last_model[slot] = Some(model);
        new_model = true;
    }
    let mut buf = String::new();
    if !st.named[slot] {
        st.named[slot] = true;
        let (vars, equs, nvar_hydro, species, nz) = names();
        let _ = write!(
            buf,
            "NAMES {id} {} {nvar_hydro} {species} {nz}",
            vars.len()
        );
        for v in &vars {
            let _ = write!(buf, " {v}");
        }
        let _ = write!(buf, "\nEQNAMES {id} {}", equs.len());
        for e in &equs {
            let _ = write!(buf, " {e}");
        }
        buf.push('\n');
    }
    f(&mut buf, id, new_model);
    let _ = st.file.write_all(buf.as_bytes());
}

/// Appends ` <hex>` for the IEEE-754 bits of `x` (Fortran `Z16.16`).
pub fn push_hex(buf: &mut String, x: f64) {
    let _ = write!(buf, " {:016X}", x.to_bits());
}

/// Appends one record line: `tag id model`, the integers in `ints`, the
/// values in `vals` as hex, then the integers in `tail`.
pub fn push_line(
    buf: &mut String,
    tag: &str,
    id: usize,
    model: usize,
    ints: &[i64],
    vals: &[f64],
    tail: &[i64],
) {
    let _ = write!(buf, "{tag} {id} {model}");
    for i in ints {
        let _ = write!(buf, " {i}");
    }
    for &v in vals {
        push_hex(buf, v);
    }
    for i in tail {
        let _ = write!(buf, " {i}");
    }
    buf.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_hex_matches_fortran_z16_16_upper_case_with_leading_zeros() {
        let mut s = String::new();
        push_hex(&mut s, 1.0);
        push_hex(&mut s, -0.0);
        push_hex(&mut s, 0.0);
        assert_eq!(s, " 3FF0000000000000 8000000000000000 0000000000000000");
    }

    #[test]
    fn test_push_line_orders_ints_hex_values_then_tail_ints() {
        let mut s = String::new();
        push_line(&mut s, "CORR", 1, 7, &[2], &[1.0, -2.0], &[734, -1]);
        assert_eq!(
            s,
            "CORR 1 7 2 3FF0000000000000 C000000000000000 734 -1\n"
        );
    }
}
