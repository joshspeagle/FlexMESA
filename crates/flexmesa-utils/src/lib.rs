// SPDX-License-Identifier: GPL-3.0-or-later
//! Core numerical guards, dictionaries, and stream tokenizers for `FlexMESA`.
//!
//! This crate ports upstream MESA's `utils` module, providing:
//! - IEEE-754 finiteness checks ([`is_bad`], [`is_bad_slice`], and
//!   [`replace_bad_nums`]) that avoid `0.0 * NaN` propagation.
//! - String-keyed ([`IntegerDict`]) and integer-pair-keyed ([`IntegerIdict`])
//!   dictionaries with duplicate-key reporting.
//! - The stateful stream tokenizer ([`Token`], [`Tokenizer`], [`tokenize`])
//!   used to parse nuclear reaction network (`.net`) definition files.

pub mod bin_codec;
pub mod cache_envelope;
pub mod sha256;

use std::collections::BTreeMap;
use std::path::PathBuf;

/// On-disk envelope schema version for runtime derived caches (`FMCENV01`).
pub const RUNTIME_CACHE_SCHEMA_VERSION: u32 = 1;

/// Resolve the canonical immutable `FlexMESA` physics data directory (`DECISIONS #3` & `#13`).
///
/// Precedence:
/// 1. `FLEXMESA_DATA_DIR` when set and non-empty.
/// 2. `XDG_CACHE_HOME/flexmesa/data` when `XDG_CACHE_HOME` is set and non-empty.
/// 3. `HOME/.cache/flexmesa/data` (or `LOCALAPPDATA/flexmesa/data` on Windows).
///
/// Runtime derived caches are stored separately via [`resolve_flexmesa_cache_dir`]
/// and never written inside this directory.
#[must_use]
pub fn resolve_flexmesa_data_dir() -> PathBuf {
    if let Some(val) = std::env::var_os("FLEXMESA_DATA_DIR") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("flexmesa").join("data");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("data");
        }
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        if !local.is_empty() {
            return PathBuf::from(local).join("flexmesa").join("data");
        }
    }
    PathBuf::from(".flexmesa_data")
}

/// Return `true` when `FLEXMESA_DISABLE_RUNTIME_CACHE` disables derived runtime caches.
#[must_use]
pub fn is_runtime_cache_disabled() -> bool {
    std::env::var("FLEXMESA_DISABLE_RUNTIME_CACHE").is_ok_and(|val| {
        matches!(
            val.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn normalize_lexical_path(path: &std::path::Path) -> PathBuf {
    use std::path::Component;
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn paths_overlap(a: &std::path::Path, b: &std::path::Path) -> bool {
    let na = normalize_lexical_path(a);
    let nb = normalize_lexical_path(b);
    na.starts_with(&nb) || nb.starts_with(&na)
}

/// Resolve the versioned runtime derived-cache directory, or `None` when disabled.
///
/// Returns `None` when [`is_runtime_cache_disabled`] is `true` or when the
/// configured `FLEXMESA_CACHE_DIR` overlaps [`resolve_flexmesa_data_dir`].
/// Otherwise resolves `FLEXMESA_CACHE_DIR/v1`,
/// `XDG_CACHE_HOME/flexmesa/derived/v1`, `HOME/.cache/flexmesa/derived/v1`,
/// `LOCALAPPDATA/flexmesa/derived/v1`, or `.flexmesa_cache/v1`.
#[must_use]
pub fn resolve_flexmesa_cache_dir() -> Option<PathBuf> {
    if is_runtime_cache_disabled() {
        return None;
    }
    let ver_dir = format!("v{RUNTIME_CACHE_SCHEMA_VERSION}");
    let candidate = if let Some(val) =
        std::env::var_os("FLEXMESA_CACHE_DIR").filter(|v| !v.is_empty())
    {
        let base = PathBuf::from(val);
        if base.file_name().and_then(|s| s.to_str()) == Some(ver_dir.as_str()) {
            base
        } else {
            base.join(&ver_dir)
        }
    } else if let Some(xdg) =
        std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty())
    {
        PathBuf::from(xdg)
            .join("flexmesa")
            .join("derived")
            .join(&ver_dir)
    } else if let Some(home) =
        std::env::var_os("HOME").filter(|v| !v.is_empty())
    {
        PathBuf::from(home)
            .join(".cache")
            .join("flexmesa")
            .join("derived")
            .join(&ver_dir)
    } else if let Some(local) =
        std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty())
    {
        PathBuf::from(local)
            .join("flexmesa")
            .join("derived")
            .join(&ver_dir)
    } else {
        PathBuf::from(".flexmesa_cache").join(&ver_dir)
    };
    let data_dir = resolve_flexmesa_data_dir();
    if paths_overlap(&candidate, &data_dir) {
        return None;
    }
    Some(candidate)
}

/// Resolve the upstream MESA release `26.4.1` verification directory (`MESA_DIR` or `mesa_test_inputs`).
#[must_use]
pub fn resolve_mesa_test_inputs_dir() -> PathBuf {
    if let Some(val) = std::env::var_os("MESA_DIR") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg)
                .join("flexmesa")
                .join("mesa_test_inputs");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs");
        }
    }
    PathBuf::from(".flexmesa_test_inputs")
}

/// Return `true` when an environment override explicitly disables zero-skip
/// enforcement (`"0"`, `"false"`, `"no"`, or `"off"`).
#[must_use]
pub fn is_zero_skips_opt_out(raw: Option<&str>) -> bool {
    raw.is_some_and(|val| {
        matches!(
            val.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Return `true` unless `FLEXMESA_REQUIRE_ZERO_SKIPS` is explicitly opted out.
#[must_use]
pub fn require_zero_skips_enabled() -> bool {
    let val = std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").ok();
    !is_zero_skips_opt_out(val.as_deref())
}

/// Resolve an upstream MESA test input (`rel_path`) under
/// [`resolve_mesa_test_inputs_dir`].
///
/// # Panics
/// Panics if the target file or directory is absent while `MESA_DIR` is set or
/// zero-skip enforcement is active (default).
#[must_use]
pub fn require_mesa_test_input_path(rel_path: &str) -> Option<PathBuf> {
    let path = resolve_mesa_test_inputs_dir().join(rel_path);
    if path.exists() {
        return Some(path);
    }
    let has_mesa_dir =
        std::env::var_os("MESA_DIR").is_some_and(|v| !v.is_empty());
    assert!(
        !has_mesa_dir && !require_zero_skips_enabled(),
        "missing required upstream MESA test input `{}` (set MESA_DIR, run \
         `flexmesa data fetch --include-test-inputs`, or set \
         FLEXMESA_REQUIRE_ZERO_SKIPS=0 to opt out locally)",
        path.display()
    );
    // Reachable only with the opt-out set. The calling test then checks
    // nothing and is still reported as passed, so say so.
    eprintln!(
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; the \
         calling test checks nothing",
        path.display()
    );
    None
}

/// Return `true` when `x` is `NaN` or $\pm\infty$ (`!x.is_finite()`).
///
/// Matches upstream MESA `utils_lib::is_bad`.
///
/// # Examples
/// ```
/// use flexmesa_utils::is_bad;
///
/// assert!(!is_bad(1.0));
/// assert!(is_bad(f64::NAN));
/// assert!(is_bad(f64::INFINITY));
/// ```
#[must_use]
pub const fn is_bad(x: f64) -> bool {
    !x.is_finite()
}

/// Return `true` if any element in `xs` is `NaN` or $\pm\infty$.
#[must_use]
pub fn is_bad_slice(xs: &[f64]) -> bool {
    xs.iter().copied().any(is_bad)
}

/// Replace any `NaN` or $\pm\infty$ element in `xs` with `replacement`.
///
/// Uses explicit conditional selection (`if x.is_finite()`) rather than
/// multiplicative masking so `NaN` values never propagate through `0.0 * NaN`.
///
/// # Examples
/// ```
/// use flexmesa_utils::replace_bad_nums;
///
/// let cleaned = replace_bad_nums(&[1.0, f64::NAN, -2.0], 0.0);
/// assert_eq!(cleaned, vec![1.0, 0.0, -2.0]);
/// ```
#[must_use]
pub fn replace_bad_nums(xs: &[f64], replacement: f64) -> Vec<f64> {
    xs.iter()
        .map(|&x| if x.is_finite() { x } else { replacement })
        .collect()
}

/// Replace underscore characters (`'_'`) with spaces after trimming trailing
/// whitespace, matching upstream MESA `utils_lib::remove_underbars`.
#[must_use]
pub fn remove_underbars(s: &str) -> String {
    s.trim_end().replace('_', " ")
}

/// `split_line` (`utils/public/utils_lib.f90:1104-1138`) with
/// `size(out) == num`: the `num` entries of `out`, each with its Fortran
/// blank padding removed.
///
/// The loop is upstream's, on bytes: a run of blanks is skipped one at a
/// time without moving the token start, so an entry can keep blanks from
/// such a run (they read as padding and are removed here). A leading blank
/// gives an empty first entry, and a final token with no blank after it
/// loses its last character (`line(kstart:k-1)` at `k >= len(line)`).
///
/// # Errors
///
/// Upstream reads `line(k:k)` with `k > len(line)` when `line` holds fewer
/// than `num` tokens (or is empty), outside the string, which has no defined
/// value (DECISIONS #43, deferred #681). That returns an error.
pub fn split_line(line: &str, num: usize) -> Result<Vec<String>, String> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let at = |k: usize| -> Result<u8, String> {
        bytes.get(k - 1).copied().ok_or_else(|| {
            format!(
                "split_line reads line({k}:{k}) past len(line) = {len} \
                 (utils/public/utils_lib.f90:1120-1128, #681)"
            )
        })
    };
    let mut out = Vec::with_capacity(num);
    let mut k = 1_usize;
    let mut kstart = 1_usize;
    for _ in 0..num {
        loop {
            if k < len && at(k)? == b' ' && at(k + 1)? == b' ' {
                k += 1;
                continue;
            }
            if at(k)? == b' ' || k >= len {
                let piece = &bytes[kstart - 1..k - 1];
                let s = String::from_utf8_lossy(piece);
                out.push(s.trim_end_matches(' ').to_owned());
                k += 1;
                kstart = k;
                break;
            }
            k += 1;
        }
    }
    Ok(out)
}

/// Stored key length, `maxlen_key_string` (`utils/public/utils_def.f90:33`).
pub const MAXLEN_KEY_STRING: usize = 50;

/// String-keyed integer dictionary matching upstream MESA `integer_dict`.
///
/// `node% key = key` stores the key in `character(len=50)`, truncating a
/// longer key, and `node% key == key` compares with blank padding
/// (`utils/private/utils_dict.f90:101-141, 273-310`). So a key is stored as
/// its first 50 bytes without trailing blanks, a lookup key matches when it
/// equals that after removing trailing blanks (tabs are significant), and a
/// lookup key longer than 50 bytes after that never matches (#665).
/// Upstream's comment `error if len_trim(key) > maxlen_key_string`
/// (`utils/public/utils_lib.f90:677, 687`) is not implemented upstream;
/// the truncation is reproduced (deferred #677).
///
/// # Examples
/// ```
/// use flexmesa_utils::IntegerDict;
///
/// let mut dict = IntegerDict::new();
/// assert!(!dict.define_and_report_duplicates("he4", 1));
/// assert!(dict.define_and_report_duplicates("he4", 2));
/// assert_eq!(dict.lookup("he4"), Some(2));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntegerDict {
    map: BTreeMap<String, i64>,
}

impl IntegerDict {
    /// Construct an empty [`IntegerDict`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Define `key -> value` and return `true` if `key` was already present.
    ///
    /// When `key` already exists, its previous value is overwritten by `value`
    /// without allocating a new key string.
    pub fn define_and_report_duplicates(
        &mut self,
        key: &str,
        value: i64,
    ) -> bool {
        let stored = stored_key(key);
        if let Some(slot) = self.map.get_mut(&stored) {
            *slot = value;
            return true;
        }
        self.map.insert(stored, value);
        false
    }

    /// Define `key -> value`, overwriting any existing value.
    pub fn define(&mut self, key: &str, value: i64) {
        let _ = self.define_and_report_duplicates(key, value);
    }

    /// Compatibility hook for upstream MESA `integer_dict_create_hash`.
    pub const fn create_hash(&mut self) {}

    /// Look up the integer value associated with `key`.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<i64> {
        let probe = key.trim_end_matches(' ');
        if probe.len() > MAXLEN_KEY_STRING {
            return None;
        }
        self.map.get(probe).copied()
    }

    /// Return the number of entries in the dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the dictionary contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return all `(key, value)` pairs in lexicographical key order.
    #[must_use]
    pub fn entries(&self) -> Vec<(String, i64)> {
        self.map.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }
}

/// `node% key = key` into `character(len=maxlen_key_string)`: the first 50
/// bytes, without the blank padding.
fn stored_key(key: &str) -> String {
    let bytes = key.as_bytes();
    let cut = &bytes[..bytes.len().min(MAXLEN_KEY_STRING)];
    String::from_utf8_lossy(cut)
        .trim_end_matches(' ')
        .to_owned()
}

/// Ordered integer-pair `(k1, k2)` dictionary matching upstream MESA
/// `integer_idict`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntegerIdict {
    map: BTreeMap<(i64, i64), i64>,
}

impl IntegerIdict {
    /// Construct an empty [`IntegerIdict`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Define `(k1, k2) -> value` and return `true` if `(k1, k2)` was already
    /// present.
    pub fn define_and_report_duplicates(
        &mut self,
        k1: i64,
        k2: i64,
        value: i64,
    ) -> bool {
        self.map.insert((k1, k2), value).is_some()
    }

    /// Define `(k1, k2) -> value`, overwriting any existing value.
    pub fn define(&mut self, k1: i64, k2: i64, value: i64) {
        let _ = self.define_and_report_duplicates(k1, k2, value);
    }

    /// Compatibility hook for upstream MESA `integer_idict_create_hash`.
    pub const fn create_hash(&mut self) {}

    /// Look up the integer value associated with `(k1, k2)`.
    #[must_use]
    pub fn lookup(&self, k1: i64, k2: i64) -> Option<i64> {
        self.map.get(&(k1, k2)).copied()
    }

    /// Return the number of entries in the dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the dictionary contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return all `(k1, k2, value)` entries in lexicographical key order.
    #[must_use]
    pub fn entries(&self) -> Vec<(i64, i64, i64)> {
        self.map.iter().map(|(&(k1, k2), &v)| (k1, k2, v)).collect()
    }
}

/// Lexical token produced by [`Tokenizer`] when parsing MESA `.net` and list
/// files, matching upstream MESA `utils_def` token codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// Bare identifier or numeric literal token (`name_token`).
    Name(String),
    /// Single- or double-quoted string literal token (`string_token`).
    String(String),
    /// Opening parenthesis `'('` (`left_paren_token`).
    LeftParen,
    /// Closing parenthesis `')'` (`right_paren_token`).
    RightParen,
    /// Comma separator `','` (`comma_token`).
    Comma,
    /// End-of-stream marker (`eof_token`).
    Eof,
}

/// Formatted records of a text file as the gfortran runtime reads them:
/// a record ends at `\n`, at `\r\n` or at a lone `\r`, and the terminator
/// is not part of the record. A final record without a terminator is still a
/// record; a terminator at the very end does not start an empty one.
#[derive(Debug, Clone)]
pub struct Records<'a> {
    rest: &'a str,
}

impl<'a> Records<'a> {
    /// Records of `input`.
    #[must_use]
    pub const fn new(input: &'a str) -> Self {
        Self { rest: input }
    }
}

impl<'a> Iterator for Records<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        let bytes = self.rest.as_bytes();
        let Some(end) = bytes.iter().position(|&c| c == b'\n' || c == b'\r')
        else {
            let record = self.rest;
            self.rest = "";
            return Some(record);
        };
        let record = &self.rest[..end];
        let skip = if bytes[end] == b'\r' && bytes.get(end + 1) == Some(&b'\n')
        {
            2
        } else {
            1
        };
        self.rest = &self.rest[end + skip..];
        Some(record)
    }
}

/// Line buffer and string length the `.net` reader passes to `token`
/// (`character (len=256)`, `net/private/net_initialize.f90:276`).
pub const TOKEN_BUFFER_LEN: usize = 256;

/// Stateful stream tokenizer, `token` (`utils/public/utils_lib.f90:567-667`)
/// transcribed on bytes (#665).
///
/// Each record is read into a buffer of `buffer_len` bytes (a longer record
/// is cut), and `n = len_trim(buffer)` drops trailing blanks only. Records
/// end where the gfortran runtime ends a formatted record: at `\n`, at
/// `\r\n` and at a lone `\r` (compiler runtime behaviour of the MESA SDK's
/// gfortran 15.2.0, not MESA source), so a CR is never in the buffer. At the start
/// of a token, tabs, blanks and `&` are skipped and `!` ends the line;
/// `(`, `)` and `,` are single tokens; a quote starts a string that runs to
/// the matching quote or the end of the line, cut to `string_len`; anything
/// else starts a name that ends only before a blank, `(`, `)` or `,` (a tab,
/// `!`, `&` or quote inside a name stays in it), cut to `string_len`.
/// Returned names and strings omit the Fortran blank padding.
#[derive(Debug, Clone)]
pub struct Tokenizer<'a> {
    lines: Records<'a>,
    buffer: &'a [u8],
    n: usize,
    i: usize,
    buffer_len: usize,
    string_len: usize,
}

impl<'a> Tokenizer<'a> {
    /// Create a [`Tokenizer`] over `input` with the `.net` reader's lengths
    /// ([`TOKEN_BUFFER_LEN`] for both the line buffer and the string).
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Self::with_lengths(input, TOKEN_BUFFER_LEN, TOKEN_BUFFER_LEN)
    }

    /// Create a [`Tokenizer`] with the caller's `len(buffer)` and
    /// `len(string)`.
    #[must_use]
    pub fn with_lengths(
        input: &'a str,
        buffer_len: usize,
        string_len: usize,
    ) -> Self {
        Self {
            lines: Records::new(input),
            buffer: &[],
            n: 0,
            i: 0,
            buffer_len,
            string_len,
        }
    }

    /// Advance and return the next [`Token`] in the stream, returning
    /// [`Token::Eof`] once the input is exhausted.
    pub fn next_token(&mut self) -> Token {
        loop {
            // `do while (i >= n)`: read lines until one has text (`:579-588`).
            while self.i >= self.n {
                let Some(line) = self.lines.next() else {
                    return Token::Eof;
                };
                let bytes = line.as_bytes();
                self.buffer = &bytes[..bytes.len().min(self.buffer_len)];
                self.n = self
                    .buffer
                    .iter()
                    .rposition(|&c| c != b' ')
                    .map_or(0, |p| p + 1);
                self.i = 0;
            }
            // `token_loop` (`:589-665`), with `i` the 1-based last read byte.
            while self.i < self.n {
                self.i += 1;
                let c = self.buffer[self.i - 1];
                match c {
                    b'\t' | b' ' | b'&' => {}
                    b'!' => {
                        self.i = self.n;
                    }
                    b'(' => return Token::LeftParen,
                    b')' => return Token::RightParen,
                    b',' => return Token::Comma,
                    b'"' | b'\'' => {
                        let mut out = Vec::new();
                        loop {
                            self.i += 1;
                            if self.i > self.n
                                || self.buffer[self.i - 1] == c
                                || out.len() >= self.string_len
                            {
                                break;
                            }
                            out.push(self.buffer[self.i - 1]);
                        }
                        let text = String::from_utf8_lossy(&out);
                        return Token::String(
                            text.trim_end_matches(' ').to_owned(),
                        );
                    }
                    _ => {
                        let j1 = self.i;
                        let mut j2 = self.i;
                        while self.i < self.n {
                            let next = self.buffer[self.i];
                            if matches!(next, b' ' | b'(' | b')' | b',') {
                                break;
                            }
                            self.i += 1;
                            j2 = self.i;
                        }
                        let l = (j2 - j1 + 1).min(self.string_len);
                        let name = &self.buffer[j1 - 1..j1 - 1 + l];
                        return Token::Name(
                            String::from_utf8_lossy(name).into_owned(),
                        );
                    }
                }
            }
        }
    }
}

/// Tokenize an entire string into a sequence of [`Token`] values terminated by
/// [`Token::Eof`].
///
/// # Examples
/// ```
/// use flexmesa_utils::{tokenize, Token};
///
/// let tokens = tokenize("add_isos(ih1, ihe4) ! comment");
/// assert_eq!(
///     tokens,
///     vec![
///         Token::Name("add_isos".to_owned()),
///         Token::LeftParen,
///         Token::Name("ih1".to_owned()),
///         Token::Comma,
///         Token::Name("ihe4".to_owned()),
///         Token::RightParen,
///         Token::Eof,
///     ]
/// );
/// ```
#[must_use]
pub fn tokenize(input: &str) -> Vec<Token> {
    let mut lexer = Tokenizer::new(input);
    let mut tokens = Vec::new();
    loop {
        let tok = lexer.next_token();
        if tok == Token::Eof {
            tokens.push(tok);
            break;
        }
        tokens.push(tok);
    }
    tokens
}
