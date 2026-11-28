// SPDX-License-Identifier: GPL-3.0-or-later
//! Opt-in solver step dumps in the record layout of the instrumented MESA
//! 26.4.1 overlay (`benchmarks/fortran_mesa/stepdumps/stepdump_patch.py`,
//! `#372`), so that a FlexMESA run can be compared record by record with
//! MESA's.
//!
//! Disabled unless the environment variable `FLEXMESA_STEPDUMP` names an
//! output file. Records are written for the first `FLEXMESA_STEPDUMP_STEPS`
//! (default 3) distinct model numbers of each star, from model number
//! `FLEXMESA_STEPDUMP_FIRST` on when that is set (a window later in the run).
//!
//! Two optional selectors place the window as the MESA overlay does.
//! `FLEXMESA_STEPDUMP_OCCURRENCE` (default 1, used with `..._FIRST`) opens
//! the window the K-th time the model number rises to the first model, since
//! a relax phase restarts the model number. `FLEXMESA_STEPDUMP_STAGE` (1-based,
//! default every stage) dumps only the S-th stage that reaches the solver:
//! each `./rn` stage is a separate MESA process, and the overlay counts the
//! processes that write a `RUN` record. Callers mark a stage boundary with
//! [`begin_stage`]; [`stop_after_window`] and [`selected_stage_ended`] tell
//! the driver nothing later is dumped. Without a stage selector the counters
//! run over the whole trace, as before.
//!
//! The dump only reads solver state: it changes no value, branch or order of
//! arithmetic.
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

/// Which models of which stage a window covers (pure bookkeeping, no I/O).
struct Selector {
    first_model: usize,
    /// `FLEXMESA_STEPDUMP_FIRST` was set (the window is gated on it).
    first_set: bool,
    /// Window opens at this rise of the model number to `first_model`.
    occurrence: usize,
    /// 1-based stage to dump, or 0 for every stage.
    stage_sel: usize,
    /// Stages that have reached the solver so far.
    run_count: usize,
    /// The current stage has been counted in `run_count`.
    stage_counted: bool,
    prev_model: [Option<usize>; MAX_IDS],
    crossings: [usize; MAX_IDS],
}

impl Selector {
    /// Returns `true` when this stage was just counted as a new process.
    fn count_stage(&mut self) -> bool {
        if self.stage_counted {
            return false;
        }
        self.stage_counted = true;
        self.run_count += 1;
        true
    }

    /// Returns whether a record of `model` for star `slot` lies in the
    /// selected stage and past the opening of the window.
    fn admit(&mut self, slot: usize, model: usize) -> bool {
        if self.stage_sel > 0 && self.run_count != self.stage_sel {
            return false;
        }
        if self.first_set && self.prev_model[slot] != Some(model) {
            if self.prev_model[slot].is_none_or(|p| p < self.first_model)
                && model >= self.first_model
            {
                self.crossings[slot] += 1;
            }
            self.prev_model[slot] = Some(model);
        }
        if self.first_set && self.crossings[slot] < self.occurrence {
            return false;
        }
        model >= self.first_model
    }

    /// Forgets the model history of every star (a new process).
    fn reset_models(&mut self) {
        self.prev_model = [None; MAX_IDS];
        self.crossings = [0; MAX_IDS];
    }
}

struct DumpState {
    file: File,
    nmax: usize,
    sel: Selector,
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
        let first_env = std::env::var("FLEXMESA_STEPDUMP_FIRST")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        let first_model = first_env.unwrap_or(0);
        let occurrence = env_count("FLEXMESA_STEPDUMP_OCCURRENCE", 1).max(1);
        let stage_sel = env_count("FLEXMESA_STEPDUMP_STAGE", 0);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.trim())
            .ok()?;
        let first_text = first_env.map_or_else(
            || String::from("first of the run"),
            |m| m.to_string(),
        );
        let stage_text = if stage_sel == 0 {
            String::from("all")
        } else {
            stage_sel.to_string()
        };
        let _ = writeln!(file, "# first_model: {first_text}");
        let _ = writeln!(file, "# steps_per_star: {nmax}");
        let _ = writeln!(file, "# occurrence: {occurrence}");
        let _ = writeln!(file, "# stage: {stage_text}");
        let _ = writeln!(file, "RUN");
        Some(Mutex::new(DumpState {
            file,
            nmax,
            sel: Selector {
                first_model,
                first_set: first_env.is_some(),
                occurrence,
                stage_sel,
                run_count: 0,
                stage_counted: false,
                prev_model: [None; MAX_IDS],
                crossings: [0; MAX_IDS],
            },
            nseen: [0; MAX_IDS],
            last_model: [None; MAX_IDS],
            named: [false; MAX_IDS],
            finished: [false; MAX_IDS],
            done: false,
        }))
    })
    .as_ref()
}

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Marks the start of a `./rn` stage (one MESA process). The stage counts
/// toward `FLEXMESA_STEPDUMP_STAGE` once it reaches the solver, as a MESA
/// process counts once it writes its `RUN` record.
pub fn begin_stage() {
    let Some(mtx) = dump() else { return };
    if let Ok(mut st) = mtx.lock() {
        st.sel.stage_counted = false;
    }
}

/// Returns `true` once the selected stage's window is complete, so the step
/// loop can stop: nothing later is dumped. Always `false` without
/// `FLEXMESA_STEPDUMP_STAGE`.
#[must_use]
pub fn stop_after_window() -> bool {
    let Some(mtx) = dump() else { return false };
    let Ok(st) = mtx.lock() else { return false };
    st.sel.stage_sel > 0 && st.sel.run_count >= st.sel.stage_sel && st.done
}

/// Returns `true` when the stage that just ended was the selected one,
/// whether or not its window was complete: later stages are not dumped.
#[must_use]
pub fn selected_stage_ended() -> bool {
    let Some(mtx) = dump() else { return false };
    let Ok(st) = mtx.lock() else { return false };
    st.sel.stage_sel > 0 && st.sel.run_count >= st.sel.stage_sel
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
    let slot = id - 1;
    if st.sel.count_stage() && st.sel.stage_sel > 0 {
        // A new process: its models are counted afresh.
        st.sel.reset_models();
        st.nseen = [0; MAX_IDS];
        st.last_model = [None; MAX_IDS];
        st.finished = [false; MAX_IDS];
    }
    if !st.sel.admit(slot, model) {
        return;
    }
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

    fn selector(
        first: Option<usize>,
        occurrence: usize,
        stage: usize,
    ) -> Selector {
        Selector {
            first_model: first.unwrap_or(0),
            first_set: first.is_some(),
            occurrence,
            stage_sel: stage,
            run_count: 0,
            stage_counted: false,
            prev_model: [None; MAX_IDS],
            crossings: [0; MAX_IDS],
        }
    }

    #[test]
    fn test_admit_opens_window_at_second_rise_of_model_number() {
        let mut sel = selector(Some(70), 2, 0);
        // First phase climbs through 70; the second restarts and rises again.
        let first_phase = [68, 69, 70, 71].map(|m| sel.admit(0, m));
        assert_eq!(first_phase, [false; 4]);
        let second_phase = [1, 2, 69, 70, 71].map(|m| sel.admit(0, m));
        assert_eq!(second_phase, [false, false, false, true, true]);
    }

    #[test]
    fn test_admit_without_first_model_admits_every_model() {
        let mut sel = selector(None, 3, 0);
        assert!([0, 1, 2].iter().all(|&m| sel.admit(0, m)));
    }

    #[test]
    fn test_admit_dumps_only_the_selected_stage() {
        let mut sel = selector(None, 1, 2);
        assert!(sel.count_stage());
        assert!(!sel.admit(0, 1));
        sel.stage_counted = false;
        assert!(sel.count_stage());
        assert!(sel.admit(0, 1));
        assert!(!sel.count_stage());
    }

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
        assert_eq!(s, "CORR 1 7 2 3FF0000000000000 C000000000000000 734 -1\n");
    }
}
