// SPDX-License-Identifier: LGPL-3.0-or-later
//! Core numerical guards, dictionaries, and stream tokenizers for `FlexMESA`.
//!
//! This crate ports upstream MESA's `utils` module, providing:
//! - IEEE-754 finiteness checks ([`is_bad`], [`is_bad_slice`], and
//!   [`replace_bad_nums`]) that avoid `0.0 * NaN` propagation.
//! - String-keyed ([`IntegerDict`]) and integer-pair-keyed ([`IntegerIdict`])
//!   dictionaries with duplicate-key reporting.
//! - The stateful stream tokenizer ([`Token`], [`Tokenizer`], [`tokenize`]) used
//!   to parse nuclear reaction network (`.net`) definition files.

use std::collections::BTreeMap;

/// Return `true` when `x` is `NaN` or $\pm\infty$ (`!x.is_finite()`).
///
/// Matches upstream MESA `utils_lib::is_bad`.
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
    s.trim_end()
        .chars()
        .map(|ch| if ch == '_' { ' ' } else { ch })
        .collect()
}

/// Split `line` on runs of spaces into at most `num` tokens, matching upstream
/// MESA `utils_lib::split_line`.
#[must_use]
pub fn split_line(line: &str, num: usize) -> Vec<String> {
    line.split_whitespace()
        .take(num)
        .map(ToOwned::to_owned)
        .collect()
}

/// String-keyed integer dictionary matching upstream MESA `integer_dict`.
///
/// Keys are trimmed of trailing whitespace on insertion and lookup to match
/// Fortran `len_trim` semantics.
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
    /// When `key` already exists, its previous value is overwritten by `value`.
    pub fn define_and_report_duplicates(&mut self, key: &str, value: i64) -> bool {
        let normalized = key.trim_end().to_owned();
        self.map.insert(normalized, value).is_some()
    }

    /// Define `key -> value`, overwriting any existing value.
    pub fn define(&mut self, key: &str, value: i64) {
        let _ = self.define_and_report_duplicates(key, value);
    }

    /// Compatibility hook for upstream MESA `integer_dict_create_hash`.
    ///
    /// Because [`IntegerDict`] maintains a balanced [`BTreeMap`] dynamically,
    /// this method is a zero-cost no-op.
    pub const fn create_hash(&mut self) {}

    /// Look up the integer value associated with `key`.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<i64> {
        self.map.get(key.trim_end()).copied()
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
    pub fn define_and_report_duplicates(&mut self, k1: i64, k2: i64, value: i64) -> bool {
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
        self.map
            .iter()
            .map(|(&(k1, k2), &v)| (k1, k2, v))
            .collect()
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

/// Stateful stream tokenizer matching upstream MESA `utils_lib::token`.
///
/// The lexer processes input line by line, skipping spaces, tabs, and `'&'`
/// continuation markers, discarding `'!'` line comments to end-of-line, and
/// emitting [`Token`] items.
#[derive(Debug, Clone)]
pub struct Tokenizer<'a> {
    lines: std::str::Lines<'a>,
    current: Vec<char>,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    /// Create a new [`Tokenizer`] over `input`.
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Self {
            lines: input.lines(),
            current: Vec::new(),
            pos: 0,
        }
    }

    /// Advance and return the next [`Token`] in the stream, returning
    /// [`Token::Eof`] once the input is exhausted.
    pub fn next_token(&mut self) -> Token {
        loop {
            while self.pos >= self.current.len() {
                let Some(line) = self.lines.next() else {
                    return Token::Eof;
                };
                self.current = line.trim_end().chars().collect();
                self.pos = 0;
            }

            while self.pos < self.current.len() {
                let ch = self.current[self.pos];
                self.pos += 1;
                match ch {
                    '\t' | ' ' | '&' => {}
                    '!' => {
                        self.pos = self.current.len();
                        break;
                    }
                    '(' => return Token::LeftParen,
                    ')' => return Token::RightParen,
                    ',' => return Token::Comma,
                    '"' | '\'' => {
                        let quote = ch;
                        let mut content = String::new();
                        while self.pos < self.current.len() {
                            let c = self.current[self.pos];
                            self.pos += 1;
                            if c == quote {
                                break;
                            }
                            content.push(c);
                        }
                        return Token::String(content);
                    }
                    _ => {
                        let mut name = String::from(ch);
                        while self.pos < self.current.len() {
                            let next_ch = self.current[self.pos];
                            if matches!(next_ch, ' ' | '\t' | '(' | ')' | ',') {
                                break;
                            }
                            name.push(next_ch);
                            self.pos += 1;
                        }
                        return Token::Name(name);
                    }
                }
            }
        }
    }
}

/// Tokenize an entire string into a sequence of [`Token`] values terminated by
/// [`Token::Eof`].
#[must_use]
pub fn tokenize(input: &str) -> Vec<Token> {
    let mut lexer = Tokenizer::new(input);
    let mut tokens = Vec::new();
    loop {
        let tok = lexer.next_token();
        let is_eof = tok == Token::Eof;
        tokens.push(tok);
        if is_eof {
            break;
        }
    }
    tokens
}
