// SPDX-License-Identifier: GPL-3.0-or-later
//! Run-level [`NumericsMode`] state, environment parsing, and scoped overrides
//! (`#165`, `DECISIONS #23`).

use core::cell::Cell;
use core::fmt;
use core::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard};

const MODE_UNINIT: u8 = 0;
const MODE_STRICT: u8 = 1;
const MODE_FAST: u8 = 2;

static ACTIVE_MODE: AtomicU8 = AtomicU8::new(MODE_UNINIT);
static MODE_SCOPE_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Run-level floating-point arithmetic mode (`#165`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NumericsMode {
    /// Hardware-independent `CRMATH` / `crlibm` arithmetic with no FMA
    /// contraction (default; sole mode compared against MESA).
    #[default]
    Strict = 0,
    /// Platform `std` `f64` intrinsics with [`crate::fma`] (`mul_add`) enabled.
    Fast = 1,
}

impl NumericsMode {
    /// Canonical lowercase identifier (`"strict"` or `"fast"`).
    #[must_use]
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Fast => "fast",
        }
    }

    /// Encode this mode as its 1-byte binary wire tag (`0` for `Strict`, `1`
    /// for `Fast`, used in `.photo` headers).
    #[must_use]
    #[inline]
    pub const fn to_wire_tag(self) -> u8 {
        match self {
            Self::Strict => 0,
            Self::Fast => 1,
        }
    }

    /// Decode a 1-byte binary wire tag (`0` -> `Strict`, `1` -> `Fast`).
    #[must_use]
    #[inline]
    pub const fn from_wire_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Strict),
            1 => Some(Self::Fast),
            _ => None,
        }
    }

    #[inline]
    const fn to_atomic_tag(self) -> u8 {
        match self {
            Self::Strict => MODE_STRICT,
            Self::Fast => MODE_FAST,
        }
    }
}

impl fmt::Display for NumericsMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when parsing an invalid [`NumericsMode`] string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseNumericsModeError {
    invalid: String,
}

impl fmt::Display for ParseNumericsModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid numerics mode {:?}: expected \"strict\" or \"fast\"",
            self.invalid
        )
    }
}

impl std::error::Error for ParseNumericsModeError {}

impl FromStr for NumericsMode {
    type Err = ParseNumericsModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("strict") {
            Ok(Self::Strict)
        } else if trimmed.eq_ignore_ascii_case("fast") {
            Ok(Self::Fast)
        } else {
            Err(ParseNumericsModeError {
                invalid: s.to_string(),
            })
        }
    }
}

#[cold]
fn init_mode_from_env() -> NumericsMode {
    let parsed = match std::env::var("FLEXMESA_NUMERICS_MODE") {
        Ok(raw) if !raw.trim().is_empty() => raw
            .parse::<NumericsMode>()
            .unwrap_or_else(|err| panic!("{err}")),
        _ => NumericsMode::Strict,
    };
    let _ = ACTIVE_MODE.compare_exchange(
        MODE_UNINIT,
        parsed.to_atomic_tag(),
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
    match ACTIVE_MODE.load(Ordering::Relaxed) {
        MODE_FAST => NumericsMode::Fast,
        _ => NumericsMode::Strict,
    }
}

/// Return the process-wide active [`NumericsMode`].
///
/// Defaults to [`NumericsMode::Strict`] unless overridden via
/// [`set_numerics_mode`], [`with_numerics_mode`], or `FLEXMESA_NUMERICS_MODE`.
#[must_use]
#[inline]
pub fn numerics_mode() -> NumericsMode {
    match ACTIVE_MODE.load(Ordering::Relaxed) {
        MODE_STRICT => NumericsMode::Strict,
        MODE_FAST => NumericsMode::Fast,
        _ => init_mode_from_env(),
    }
}

/// Return `true` when [`NumericsMode::Strict`] is active.
#[must_use]
#[inline]
pub(crate) fn is_strict() -> bool {
    numerics_mode() == NumericsMode::Strict
}

/// Set the process-wide [`NumericsMode`] at run start.
#[inline]
pub fn set_numerics_mode(mode: NumericsMode) {
    ACTIVE_MODE.store(mode.to_atomic_tag(), Ordering::Relaxed);
}

struct ModeResetGuard<'a> {
    prev: NumericsMode,
    _lock_guard: Option<MutexGuard<'a, ()>>,
}

impl Drop for ModeResetGuard<'_> {
    fn drop(&mut self) {
        set_numerics_mode(self.prev);
        SCOPE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Execute `f` with `mode` active, serializing concurrent mode-switching callers
/// across threads (while allowing nested reentrant calls on the same thread)
/// and restoring the previous mode when `f` returns or unwinds.
pub fn with_numerics_mode<T, F>(mode: NumericsMode, f: F) -> T
where
    F: FnOnce() -> T,
{
    let lock_guard = SCOPE_DEPTH.with(|depth| {
        let cur = depth.get();
        let guard = if cur == 0 {
            Some(
                MODE_SCOPE_LOCK
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        } else {
            None
        };
        depth.set(cur + 1);
        guard
    });
    let reset = ModeResetGuard {
        prev: numerics_mode(),
        _lock_guard: lock_guard,
    };
    set_numerics_mode(mode);
    let out = f();
    drop(reset);
    out
}
