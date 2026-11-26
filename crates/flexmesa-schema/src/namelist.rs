// SPDX-License-Identifier: GPL-3.0-or-later
//! Parser and schema validator for legacy MESA Fortran `&namelist` (`inlist`)
//! files.
//!
//! Extracts [`NamelistBlock`] and [`NamelistAssignment`] entries from Fortran
//! `&namelist ... /` text and validates every assigned parameter against a
//! [`ControlDefault`] schema.

use std::collections::HashMap;
use std::path::Path;

use crate::{
    is_valid_identifier, parse_control_value, read_file_text,
    strip_fortran_comment, ControlDefault, ControlType, ControlValue,
    SchemaError,
};

const MAX_UNBOUNDED_EXPAND: usize = 1_000_000;

/// Single `name[(subscript)] = value` assignment inside a Fortran namelist.
#[derive(Clone, Debug, PartialEq)]
pub struct NamelistAssignment {
    /// Control parameter name as written in the `inlist`.
    pub name: String,
    /// Optional subscript expression inside `(...)`.
    pub subscript: Option<String>,
    /// Parsed [`ControlValue`] on the right-hand side.
    pub value: ControlValue,
    /// 1-based line number in the `inlist` source.
    pub line_number: usize,
}

/// Parsed Fortran `&namelist ... /` block from a MESA `inlist` file.
#[derive(Clone, Debug, PartialEq)]
pub struct NamelistBlock {
    /// Namelist group name without leading `&` (for example, `"eos"`).
    pub name: String,
    /// Ordered list of assignments inside the block.
    pub assignments: Vec<NamelistAssignment>,
}

#[derive(Clone, Debug)]
enum TokenKind {
    Equals,
    Comma,
    LParen,
    RParen,
    Colon,
    QuotedString(String),
    Word(String),
    RepeatPrefix { count: usize, adjacent: bool },
}

#[derive(Clone, Debug)]
struct SpannedToken {
    kind: TokenKind,
    line_number: usize,
}

#[derive(Clone, Copy, Debug)]
enum SubscriptDim {
    Scalar(i64),
    Slice {
        lo: Option<i64>,
        hi: Option<i64>,
        step: i64,
        has_explicit_step: bool,
    },
}

#[derive(Clone, Debug)]
enum RhsItem {
    Single {
        value: ControlValue,
        line_number: usize,
    },
    Repeated {
        count: usize,
        value: ControlValue,
        line_number: usize,
    },
    Null(usize),
}

#[derive(Clone, Copy, Debug)]
enum ResolvedDim {
    Bounded { start: i64, step: i64, len: usize },
    Unbounded { start: i64, step: i64 },
}

#[derive(Clone, Debug)]
struct SliceCoordIter {
    dims: Vec<ResolvedDim>,
    pos: Vec<usize>,
    exhausted: bool,
}

fn resolve_slice_dim(
    dim: SubscriptDim,
    has_later_slice: bool,
    name: &str,
    line_number: usize,
) -> Result<ResolvedDim, SchemaError> {
    match dim {
        SubscriptDim::Scalar(v) => Ok(ResolvedDim::Bounded {
            start: v,
            step: 1,
            len: 1,
        }),
        SubscriptDim::Slice { lo, hi, step, .. } => {
            if lo.is_none() && step < 0 {
                return Err(SchemaError::ParseError {
                    line_number,
                    message: format!(
                        "negative-stride slice in '{name}' requires an \
                         explicit start subscript"
                    ),
                });
            }
            let start = lo.unwrap_or(1);
            if let Some(end) = hi {
                let diff = if step > 0 {
                    i128::from(end) - i128::from(start)
                } else {
                    i128::from(start) - i128::from(end)
                };
                let stride = i128::from(step).abs();
                let len = if diff < 0 {
                    0
                } else {
                    usize::try_from(diff / stride + 1).map_err(|_| {
                        SchemaError::ParseError {
                            line_number,
                            message: format!(
                                "subscript slice extent overflow in '{name}'"
                            ),
                        }
                    })?
                };
                Ok(ResolvedDim::Bounded { start, step, len })
            } else if has_later_slice {
                Err(SchemaError::ParseError {
                    line_number,
                    message: format!(
                        "unbounded slice in non-trailing subscript dimension \
                         for '{name}'"
                    ),
                })
            } else {
                Ok(ResolvedDim::Unbounded { start, step })
            }
        }
    }
}

impl SliceCoordIter {
    fn from_dims(
        dims: &[SubscriptDim],
        name: &str,
        line_number: usize,
    ) -> Result<Self, SchemaError> {
        let mut resolved = Vec::with_capacity(dims.len());
        for (idx, &dim) in dims.iter().enumerate() {
            let later_has_slice = dims[idx + 1..]
                .iter()
                .any(|d| matches!(d, SubscriptDim::Slice { .. }));
            resolved.push(resolve_slice_dim(
                dim,
                later_has_slice,
                name,
                line_number,
            )?);
        }
        let exhausted = resolved
            .iter()
            .any(|d| matches!(d, ResolvedDim::Bounded { len: 0, .. }));
        let pos = vec![0; resolved.len()];
        Ok(Self {
            dims: resolved,
            pos,
            exhausted,
        })
    }

    fn is_fully_bounded(&self) -> bool {
        self.dims
            .iter()
            .all(|d| matches!(d, ResolvedDim::Bounded { .. }))
    }

    fn format_current(
        &self,
        name: &str,
        line_number: usize,
    ) -> Result<String, SchemaError> {
        let mut parts = Vec::with_capacity(self.dims.len());
        for (dim, &p) in self.dims.iter().zip(&self.pos) {
            let (start, step) = match *dim {
                ResolvedDim::Bounded { start, step, .. }
                | ResolvedDim::Unbounded { start, step } => (start, step),
            };
            let offset =
                i128::try_from(p).map_err(|_| SchemaError::ParseError {
                    line_number,
                    message: format!("subscript index overflow in '{name}'"),
                })?;
            let coord = i128::from(start) + offset * i128::from(step);
            let coord_i64 =
                i64::try_from(coord).map_err(|_| SchemaError::ParseError {
                    line_number,
                    message: format!("subscript index overflow in '{name}'"),
                })?;
            parts.push(coord_i64.to_string());
        }
        Ok(parts.join(","))
    }

    fn advance(&mut self, mut steps: usize) -> bool {
        if self.exhausted || steps == 0 {
            return !self.exhausted;
        }
        for d in 0..self.dims.len() {
            let Some(next_val) = self.pos[d].checked_add(steps) else {
                self.exhausted = true;
                return false;
            };
            match self.dims[d] {
                ResolvedDim::Bounded { len: 0, .. } => {
                    self.exhausted = true;
                    return false;
                }
                ResolvedDim::Bounded { len, .. } => {
                    self.pos[d] = next_val % len;
                    steps = next_val / len;
                    if steps == 0 {
                        return true;
                    }
                }
                ResolvedDim::Unbounded { start, step } => {
                    let Ok(offset) = i128::try_from(next_val) else {
                        self.exhausted = true;
                        return false;
                    };
                    let coord = i128::from(start) + offset * i128::from(step);
                    if i64::try_from(coord).is_err() {
                        self.exhausted = true;
                        return false;
                    }
                    self.pos[d] = next_val;
                    return true;
                }
            }
        }
        self.exhausted = true;
        false
    }
}

fn starts_with_end_keyword(rest: &str) -> bool {
    let bytes = rest.as_bytes();
    bytes.len() >= 4
        && bytes[..4].eq_ignore_ascii_case(b"&end")
        && !bytes
            .get(4)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

fn is_eol_continuation(rest: &str) -> bool {
    let Some(after) = rest.strip_prefix('&') else {
        return false;
    };
    let trimmed = after.trim_start_matches([' ', '\t', '\r']);
    trimmed.is_empty() || trimmed.starts_with(['\n', '!'])
}

fn is_atom_stop_char(ch: char) -> bool {
    matches!(
        ch,
        ' ' | '\t'
            | '\r'
            | '\n'
            | ','
            | '='
            | '('
            | ')'
            | ':'
            | '/'
            | '!'
            | '\''
            | '"'
    )
}

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            pos: 0,
            line: 1,
        }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
        }
        Some(ch)
    }

    fn skip_horiz_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\r')) {
            self.bump();
        }
    }

    fn skip_to_eol(&mut self) {
        while let Some(ch) = self.bump() {
            if ch == '\n' {
                break;
            }
        }
    }

    fn skip_outer_ws_and_comments(&mut self) {
        loop {
            while matches!(self.peek(), Some(' ' | '\t' | '\r' | '\n')) {
                self.bump();
            }
            if self.peek() == Some('!') {
                self.skip_to_eol();
            } else {
                break;
            }
        }
    }

    fn read_group_header(
        &mut self,
    ) -> Result<Option<(String, usize)>, SchemaError> {
        self.skip_outer_ws_and_comments();
        let Some(first) = self.peek() else {
            return Ok(None);
        };
        let line_number = self.line;
        if first != '&' {
            let line_rest = self.rest().split('\n').next().unwrap_or("");
            let (code, _) = strip_fortran_comment(line_rest);
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("expected '&namelist' header: {code:?}"),
            });
        }
        if starts_with_end_keyword(self.rest()) {
            return Err(SchemaError::ParseError {
                line_number,
                message: "unexpected '&end' outside namelist block".to_owned(),
            });
        }
        self.bump();
        let mut group = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                group.push(ch);
                self.bump();
            } else {
                break;
            }
        }
        if !is_valid_identifier(&group) {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("invalid namelist name: {group:?}"),
            });
        }
        let valid_delim = match self.peek() {
            None | Some(' ' | '\t' | '\r' | '\n' | '/' | '!') => true,
            Some('&') => is_eol_continuation(self.rest()),
            Some(_) => false,
        };
        if !valid_delim {
            let bad = self
                .rest()
                .split(|c: char| c.is_whitespace() || c == '/' || c == '!')
                .next()
                .unwrap_or("");
            let full = format!("{group}{bad}");
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("invalid namelist name: {full:?}"),
            });
        }
        Ok(Some((group, line_number)))
    }

    fn skip_body_gaps(&mut self, pending_cont: &mut bool) {
        loop {
            self.skip_horiz_ws();
            match self.peek() {
                Some('\n') => {
                    self.bump();
                }
                Some('!') => self.skip_to_eol(),
                Some('&')
                    if *pending_cont
                        && !starts_with_end_keyword(self.rest()) =>
                {
                    self.bump();
                    *pending_cont = false;
                }
                _ => {
                    *pending_cont = false;
                    break;
                }
            }
        }
    }

    fn lex_quoted_string(
        &mut self,
        quote: char,
    ) -> Result<String, SchemaError> {
        let start_line = self.line;
        self.bump();
        let mut out = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(SchemaError::ParseError {
                        line_number: start_line,
                        message: "unterminated string literal".to_owned(),
                    });
                }
                Some(q) if q == quote => {
                    self.bump();
                    if self.peek() == Some(quote) {
                        self.bump();
                        out.push(quote);
                    } else {
                        return Ok(out);
                    }
                }
                Some('&')
                    if self.rest()[1..].starts_with('\n')
                        || self.rest()[1..].starts_with("\r\n") =>
                {
                    self.bump();
                    if self.peek() == Some('\r') {
                        self.bump();
                    }
                    self.bump();
                    let saved_pos = self.pos;
                    let saved_line = self.line;
                    while matches!(self.peek(), Some(' ' | '\t')) {
                        self.bump();
                    }
                    if self.peek() == Some('&') {
                        self.bump();
                    } else {
                        self.pos = saved_pos;
                        self.line = saved_line;
                    }
                }
                Some('\r') if self.rest().starts_with("\r\n") => {
                    self.bump();
                }
                Some('\n') => {
                    self.bump();
                }
                Some(ch) => {
                    self.bump();
                    out.push(ch);
                }
            }
        }
    }

    fn advance_token_continuation(&mut self, pending_cont: &mut bool) -> bool {
        self.bump();
        self.skip_to_eol();
        loop {
            self.skip_horiz_ws();
            match self.peek() {
                Some('\n') => {
                    self.bump();
                }
                Some('!') => self.skip_to_eol(),
                Some('&') if !starts_with_end_keyword(self.rest()) => {
                    self.bump();
                    *pending_cont = false;
                    return !matches!(self.peek(), None | Some('&'))
                        && !self.peek().is_some_and(is_atom_stop_char);
                }
                _ => {
                    *pending_cont = false;
                    return false;
                }
            }
        }
    }

    fn lex_word_or_repeat(
        &mut self,
        pending_cont: &mut bool,
        out: &mut Vec<SpannedToken>,
    ) -> Result<(), SchemaError> {
        let start_line = self.line;
        let mut buf = String::new();
        while let Some(ch) = self.peek() {
            if is_atom_stop_char(ch) {
                break;
            }
            if ch == '&' {
                if !starts_with_end_keyword(self.rest())
                    && is_eol_continuation(self.rest())
                    && self.advance_token_continuation(pending_cont)
                {
                    continue;
                }
                break;
            }
            self.bump();
            buf.push(ch);
        }
        if let Some((count, suffix)) = parse_repeat_split(&buf, start_line)? {
            if suffix.is_empty() {
                let adjacent = matches!(
                    self.peek(),
                    Some('\'' | '"' | '=' | '(' | ')' | ':')
                );
                out.push(SpannedToken {
                    kind: TokenKind::RepeatPrefix { count, adjacent },
                    line_number: start_line,
                });
            } else {
                out.push(SpannedToken {
                    kind: TokenKind::RepeatPrefix {
                        count,
                        adjacent: true,
                    },
                    line_number: start_line,
                });
                out.push(SpannedToken {
                    kind: TokenKind::Word(suffix.to_owned()),
                    line_number: start_line,
                });
            }
            return Ok(());
        }
        out.push(SpannedToken {
            kind: TokenKind::Word(buf),
            line_number: start_line,
        });
        Ok(())
    }

    fn check_trailing_after_terminator(
        &mut self,
        term_line: usize,
        term: &str,
    ) -> Result<(), SchemaError> {
        self.skip_horiz_ws();
        match self.peek() {
            None => Ok(()),
            Some('\n') => {
                self.bump();
                Ok(())
            }
            Some('!') => {
                self.skip_to_eol();
                Ok(())
            }
            Some('&') if !starts_with_end_keyword(self.rest()) => Ok(()),
            Some(_) => {
                let line_rest = self.rest().split('\n').next().unwrap_or("");
                let (trailing, _) = strip_fortran_comment(line_rest);
                Err(SchemaError::ParseError {
                    line_number: term_line,
                    message: format!(
                        "unexpected trailing tokens after '{term}': \
                         {trailing:?}"
                    ),
                })
            }
        }
    }

    fn lex_block_body(
        &mut self,
        group_name: &str,
    ) -> Result<Vec<SpannedToken>, SchemaError> {
        let mut tokens = Vec::new();
        let mut pending_cont = false;
        let total_lines = self.src.lines().count().max(1);
        loop {
            self.skip_body_gaps(&mut pending_cont);
            let Some(ch) = self.peek() else {
                return Err(SchemaError::ParseError {
                    line_number: total_lines,
                    message: format!("unclosed namelist '&{group_name}'"),
                });
            };
            let line_number = self.line;
            let simple = match ch {
                '/' => {
                    self.bump();
                    self.check_trailing_after_terminator(line_number, "/")?;
                    return Ok(tokens);
                }
                '&' => {
                    if starts_with_end_keyword(self.rest()) {
                        for _ in 0..4 {
                            self.bump();
                        }
                        self.check_trailing_after_terminator(
                            line_number,
                            "&end",
                        )?;
                        return Ok(tokens);
                    }
                    if is_eol_continuation(self.rest()) {
                        self.bump();
                        self.skip_to_eol();
                        pending_cont = true;
                        continue;
                    }
                    return Err(SchemaError::ParseError {
                        line_number,
                        message: format!(
                            "unexpected '&' inside '&{group_name}' namelist"
                        ),
                    });
                }
                '=' => Some(TokenKind::Equals),
                ',' => Some(TokenKind::Comma),
                '(' => Some(TokenKind::LParen),
                ')' => Some(TokenKind::RParen),
                ':' => Some(TokenKind::Colon),
                '\'' | '"' => {
                    let s = self.lex_quoted_string(ch)?;
                    tokens.push(SpannedToken {
                        kind: TokenKind::QuotedString(s),
                        line_number,
                    });
                    None
                }
                _ => {
                    self.lex_word_or_repeat(&mut pending_cont, &mut tokens)?;
                    None
                }
            };
            if let Some(kind) = simple {
                self.bump();
                tokens.push(SpannedToken { kind, line_number });
            }
        }
    }
}

fn parse_repeat_split(
    buf: &str,
    line_number: usize,
) -> Result<Option<(usize, &str)>, SchemaError> {
    let Some((prefix, suffix)) = buf.split_once('*') else {
        return Ok(None);
    };
    if prefix.is_empty() || !prefix.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(None);
    }
    let count =
        prefix
            .parse::<usize>()
            .map_err(|_| SchemaError::ParseError {
                line_number,
                message: format!("invalid repeat count in {buf:?}"),
            })?;
    if count == 0 {
        return Err(SchemaError::ParseError {
            line_number,
            message: "repeat count must be positive, got 0".to_owned(),
        });
    }
    if suffix.contains('*') {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!("invalid nested repeat specifier in {buf:?}"),
        });
    }
    Ok(Some((count, suffix)))
}

fn parse_opt_i64(
    part: &[SpannedToken],
    name: &str,
    line_number: usize,
) -> Result<Option<i64>, SchemaError> {
    if part.is_empty() {
        return Ok(None);
    }
    if let [tok] = part {
        if let TokenKind::Word(ref w) = tok.kind {
            if let Ok(idx) = w.parse::<i64>() {
                return Ok(Some(idx));
            }
        }
    }
    Err(SchemaError::ParseError {
        line_number,
        message: format!("invalid subscript index in '{name}(...)'"),
    })
}

fn parse_subscript_dims(
    slice: &[SpannedToken],
    name: &str,
    line_number: usize,
) -> Result<Vec<SubscriptDim>, SchemaError> {
    if slice.is_empty() {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!("empty subscript in '{name}()'"),
        });
    }
    let mut dims = Vec::new();
    for dim_toks in slice.split(|t| matches!(t.kind, TokenKind::Comma)) {
        if dim_toks.is_empty() {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("empty subscript dimension in '{name}(...)'"),
            });
        }
        let parts: Vec<&[SpannedToken]> = dim_toks
            .split(|t| matches!(t.kind, TokenKind::Colon))
            .collect();
        match parts.len() {
            1 => {
                let Some(idx) = parse_opt_i64(parts[0], name, line_number)?
                else {
                    return Err(SchemaError::ParseError {
                        line_number,
                        message: format!(
                            "empty subscript index in '{name}(...)'"
                        ),
                    });
                };
                dims.push(SubscriptDim::Scalar(idx));
            }
            2 => {
                let lo = parse_opt_i64(parts[0], name, line_number)?;
                let hi = parse_opt_i64(parts[1], name, line_number)?;
                dims.push(SubscriptDim::Slice {
                    lo,
                    hi,
                    step: 1,
                    has_explicit_step: false,
                });
            }
            3 => {
                let lo = parse_opt_i64(parts[0], name, line_number)?;
                let hi = parse_opt_i64(parts[1], name, line_number)?;
                let Some(step) = parse_opt_i64(parts[2], name, line_number)?
                else {
                    return Err(SchemaError::ParseError {
                        line_number,
                        message: format!(
                            "missing stride after second ':' in '{name}(...)'"
                        ),
                    });
                };
                if step == 0 {
                    return Err(SchemaError::ParseError {
                        line_number,
                        message: format!(
                            "zero stride in subscript slice for '{name}'"
                        ),
                    });
                }
                dims.push(SubscriptDim::Slice {
                    lo,
                    hi,
                    step,
                    has_explicit_step: true,
                });
            }
            _ => {
                return Err(SchemaError::ParseError {
                    line_number,
                    message: format!("too many ':' in subscript for '{name}'"),
                });
            }
        }
    }
    Ok(dims)
}

fn format_broadcast_subscript(dims: &[SubscriptDim]) -> String {
    let mut parts = Vec::with_capacity(dims.len());
    for dim in dims {
        match *dim {
            SubscriptDim::Scalar(v) => parts.push(v.to_string()),
            SubscriptDim::Slice { lo, hi, .. } => match (lo, hi) {
                (Some(a), Some(b)) if a == b => parts.push(a.to_string()),
                _ => {
                    let lo_s = lo.map_or_else(String::new, |v| v.to_string());
                    let hi_s = hi.map_or_else(String::new, |v| v.to_string());
                    parts.push(format!("{lo_s}:{hi_s}"));
                }
            },
        }
    }
    parts.join(",")
}

type LhsMatch<'a> = (&'a str, Vec<SubscriptDim>, usize, usize);

fn detect_lhs(
    tokens: &[SpannedToken],
    idx: usize,
) -> Option<Result<LhsMatch<'_>, SchemaError>> {
    let first = tokens.get(idx)?;
    let TokenKind::Word(ref name) = first.kind else {
        return None;
    };
    if !is_valid_identifier(name) {
        return None;
    }
    match tokens.get(idx + 1).map(|t| &t.kind) {
        Some(TokenKind::Equals) => {
            Some(Ok((name.as_str(), Vec::new(), idx + 2, first.line_number)))
        }
        Some(TokenKind::LParen) => {
            let mut rp = idx + 2;
            while rp < tokens.len() {
                match tokens[rp].kind {
                    TokenKind::RParen => break,
                    TokenKind::LParen | TokenKind::Equals => return None,
                    _ => rp += 1,
                }
            }
            if rp >= tokens.len() {
                return None;
            }
            if !matches!(
                tokens.get(rp + 1).map(|t| &t.kind),
                Some(TokenKind::Equals)
            ) {
                return None;
            }
            Some(
                parse_subscript_dims(
                    &tokens[idx + 2..rp],
                    name,
                    first.line_number,
                )
                .map(|dims| (name.as_str(), dims, rp + 2, first.line_number)),
            )
        }
        _ => None,
    }
}

fn parse_repeated_literal(
    tokens: &[SpannedToken],
    idx: usize,
    count: usize,
    name: &str,
    tok_line: usize,
) -> Result<RhsItem, SchemaError> {
    let Some(next) = tokens.get(idx + 1) else {
        return Err(SchemaError::ParseError {
            line_number: tok_line,
            message: format!(
                "expected literal after repeat specifier '{count}*' for \
                 '{name}'"
            ),
        });
    };
    let value = match &next.kind {
        TokenKind::QuotedString(s) => ControlValue::String(s.clone()),
        TokenKind::Word(w) => parse_control_value(w, next.line_number)?,
        _ => {
            return Err(SchemaError::ParseError {
                line_number: next.line_number,
                message: format!(
                    "invalid literal after repeat specifier '{count}*' for \
                     '{name}'"
                ),
            });
        }
    };
    Ok(RhsItem::Repeated {
        count,
        value,
        line_number: next.line_number,
    })
}

fn parse_rhs_items(
    tokens: &[SpannedToken],
    mut idx: usize,
    name: &str,
    lhs_line: usize,
) -> Result<(Vec<RhsItem>, usize), SchemaError> {
    let mut items = Vec::new();
    let mut at_rhs_start = true;
    let mut after_comma = false;
    while idx < tokens.len() {
        if detect_lhs(tokens, idx).is_some() {
            break;
        }
        let tok = &tokens[idx];
        match &tok.kind {
            TokenKind::Comma => {
                if at_rhs_start || after_comma {
                    items.push(RhsItem::Null(1));
                }
                at_rhs_start = false;
                after_comma = true;
                idx += 1;
            }
            TokenKind::RepeatPrefix { count, adjacent } => {
                at_rhs_start = false;
                after_comma = false;
                if *adjacent {
                    items.push(parse_repeated_literal(
                        tokens,
                        idx,
                        *count,
                        name,
                        tok.line_number,
                    )?);
                    idx += 2;
                } else {
                    items.push(RhsItem::Null(*count));
                    idx += 1;
                }
            }
            TokenKind::QuotedString(s) => {
                at_rhs_start = false;
                after_comma = false;
                items.push(RhsItem::Single {
                    value: ControlValue::String(s.clone()),
                    line_number: tok.line_number,
                });
                idx += 1;
            }
            TokenKind::Word(w) => {
                at_rhs_start = false;
                after_comma = false;
                items.push(RhsItem::Single {
                    value: parse_control_value(w, tok.line_number)?,
                    line_number: tok.line_number,
                });
                idx += 1;
            }
            TokenKind::Equals
            | TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::Colon => {
                return Err(SchemaError::ParseError {
                    line_number: tok.line_number,
                    message: format!(
                        "unexpected delimiter in value list for '{name}'"
                    ),
                });
            }
        }
    }
    if items.is_empty() {
        return Err(SchemaError::ParseError {
            line_number: lhs_line,
            message: format!("missing value for namelist control '{name}'"),
        });
    }
    Ok((items, idx))
}

fn check_unbounded_cap(
    total: &mut usize,
    add: usize,
    name: &str,
    line_number: usize,
) -> Result<(), SchemaError> {
    *total = total
        .checked_add(add)
        .filter(|&n| n <= MAX_UNBOUNDED_EXPAND)
        .ok_or_else(|| SchemaError::ParseError {
            line_number,
            message: format!(
                "namelist value expansion exceeds limit for '{name}'"
            ),
        })?;
    Ok(())
}

fn expand_unbounded_1d(
    name: &str,
    start: i64,
    lhs_line: usize,
    items: &[RhsItem],
    out: &mut Vec<NamelistAssignment>,
) -> Result<(), SchemaError> {
    let mut cursor = start;
    let mut total = 0_usize;
    for item in items {
        match item {
            RhsItem::Single { value, line_number } => {
                check_unbounded_cap(&mut total, 1, name, *line_number)?;
                out.push(NamelistAssignment {
                    name: name.to_owned(),
                    subscript: Some(cursor.to_string()),
                    value: value.clone(),
                    line_number: *line_number,
                });
                cursor = cursor.checked_add(1).ok_or_else(|| {
                    SchemaError::ParseError {
                        line_number: *line_number,
                        message: format!("subscript overflow in '{name}'"),
                    }
                })?;
            }
            RhsItem::Repeated {
                count,
                value,
                line_number,
            } => {
                check_unbounded_cap(&mut total, *count, name, *line_number)?;
                for _ in 0..*count {
                    out.push(NamelistAssignment {
                        name: name.to_owned(),
                        subscript: Some(cursor.to_string()),
                        value: value.clone(),
                        line_number: *line_number,
                    });
                    cursor = cursor.checked_add(1).ok_or_else(|| {
                        SchemaError::ParseError {
                            line_number: *line_number,
                            message: format!("subscript overflow in '{name}'"),
                        }
                    })?;
                }
            }
            RhsItem::Null(n) => {
                check_unbounded_cap(&mut total, *n, name, lhs_line)?;
                let n_i64 =
                    i64::try_from(*n).map_err(|_| SchemaError::ParseError {
                        line_number: lhs_line,
                        message: format!("subscript overflow in '{name}'"),
                    })?;
                cursor = cursor.checked_add(n_i64).ok_or_else(|| {
                    SchemaError::ParseError {
                        line_number: lhs_line,
                        message: format!("subscript overflow in '{name}'"),
                    }
                })?;
            }
        }
    }
    Ok(())
}

fn expand_unsubscripted(
    name: &str,
    lhs_line: usize,
    items: &[RhsItem],
    out: &mut Vec<NamelistAssignment>,
) -> Result<(), SchemaError> {
    if let [item] = items {
        match item {
            RhsItem::Single { value, line_number }
            | RhsItem::Repeated {
                count: 1,
                value,
                line_number,
            } => {
                out.push(NamelistAssignment {
                    name: name.to_owned(),
                    subscript: None,
                    value: value.clone(),
                    line_number: *line_number,
                });
                return Ok(());
            }
            RhsItem::Null(_) => return Ok(()),
            RhsItem::Repeated { .. } => {}
        }
    }
    expand_unbounded_1d(name, 1, lhs_line, items, out)
}

#[allow(clippy::too_many_arguments)]
fn emit_slice_repeats(
    iter: &mut SliceCoordIter,
    bounded: bool,
    unbounded_count: &mut usize,
    name: &str,
    count: usize,
    value: &ControlValue,
    line_number: usize,
    out: &mut Vec<NamelistAssignment>,
) -> Result<(), SchemaError> {
    if count == 0 {
        return Ok(());
    }
    if iter.exhausted || !iter.clone().advance(count - 1) {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!(
                "too many values ({count}) for subscript slice '{name}'"
            ),
        });
    }
    if !bounded {
        check_unbounded_cap(unbounded_count, count, name, line_number)?;
    }
    for _ in 0..count {
        let sub = iter.format_current(name, line_number)?;
        out.push(NamelistAssignment {
            name: name.to_owned(),
            subscript: Some(sub),
            value: value.clone(),
            line_number,
        });
        iter.advance(1);
    }
    Ok(())
}

fn expand_slice_dims(
    name: &str,
    lhs_line: usize,
    dims: &[SubscriptDim],
    items: &[RhsItem],
    out: &mut Vec<NamelistAssignment>,
) -> Result<(), SchemaError> {
    let mut iter = SliceCoordIter::from_dims(dims, name, lhs_line)?;
    let bounded = iter.is_fully_bounded();
    let mut unbounded_count = 0_usize;
    for item in items {
        match item {
            RhsItem::Single { value, line_number } => {
                emit_slice_repeats(
                    &mut iter,
                    bounded,
                    &mut unbounded_count,
                    name,
                    1,
                    value,
                    *line_number,
                    out,
                )?;
            }
            RhsItem::Repeated {
                count,
                value,
                line_number,
            } => {
                emit_slice_repeats(
                    &mut iter,
                    bounded,
                    &mut unbounded_count,
                    name,
                    *count,
                    value,
                    *line_number,
                    out,
                )?;
            }
            RhsItem::Null(n) => {
                if *n == 0 {
                    continue;
                }
                if iter.exhausted || !iter.advance(*n - 1) {
                    return Err(SchemaError::ParseError {
                        line_number: lhs_line,
                        message: format!(
                            "null item exceeds subscript slice extent for \
                             '{name}'"
                        ),
                    });
                }
                if !bounded {
                    check_unbounded_cap(
                        &mut unbounded_count,
                        *n,
                        name,
                        lhs_line,
                    )?;
                }
                iter.advance(1);
            }
        }
    }
    Ok(())
}

fn try_broadcast_slice(
    name: &str,
    dims: &[SubscriptDim],
    items: &[RhsItem],
    out: &mut Vec<NamelistAssignment>,
) -> bool {
    let [RhsItem::Single { value, line_number }] = items else {
        return false;
    };
    let all_unit = dims.iter().all(|d| match *d {
        SubscriptDim::Scalar(_) => true,
        SubscriptDim::Slice {
            lo,
            hi,
            step,
            has_explicit_step,
        } => {
            step == 1
                && !has_explicit_step
                && !matches!((lo, hi), (Some(a), Some(b)) if a > b)
        }
    });
    if !all_unit {
        return false;
    }
    out.push(NamelistAssignment {
        name: name.to_owned(),
        subscript: Some(format_broadcast_subscript(dims)),
        value: value.clone(),
        line_number: *line_number,
    });
    true
}

fn expand_assignment(
    name: &str,
    lhs_line: usize,
    dims: &[SubscriptDim],
    items: &[RhsItem],
    out: &mut Vec<NamelistAssignment>,
) -> Result<(), SchemaError> {
    if dims.is_empty() {
        return expand_unsubscripted(name, lhs_line, items, out);
    }
    let has_slice =
        dims.iter().any(|d| matches!(d, SubscriptDim::Slice { .. }));
    if !has_slice {
        if let [SubscriptDim::Scalar(start)] = dims {
            return expand_unbounded_1d(name, *start, lhs_line, items, out);
        }
        if let [item] = items {
            match item {
                RhsItem::Single { value, line_number }
                | RhsItem::Repeated {
                    count: 1,
                    value,
                    line_number,
                } => {
                    out.push(NamelistAssignment {
                        name: name.to_owned(),
                        subscript: Some(format_broadcast_subscript(dims)),
                        value: value.clone(),
                        line_number: *line_number,
                    });
                    return Ok(());
                }
                RhsItem::Null(1) => return Ok(()),
                _ => {}
            }
        }
        return Err(SchemaError::ParseError {
            line_number: lhs_line,
            message: format!(
                "cannot expand value list across multi-dimensional scalar \
                 subscript for '{name}' without slice bounds"
            ),
        });
    }
    if try_broadcast_slice(name, dims, items, out) {
        return Ok(());
    }
    expand_slice_dims(name, lhs_line, dims, items, out)
}

fn parse_block_tokens(
    tokens: &[SpannedToken],
) -> Result<Vec<NamelistAssignment>, SchemaError> {
    let mut assigns = Vec::new();
    let mut idx = 0_usize;
    while idx < tokens.len() {
        let Some(lhs_res) = detect_lhs(tokens, idx) else {
            let tok = &tokens[idx];
            let msg = match &tok.kind {
                TokenKind::Word(w) => {
                    format!("expected '=' in assignment: {w:?}")
                }
                TokenKind::QuotedString(s) => {
                    format!(
                        "unexpected string literal without assignment: {s:?}"
                    )
                }
                TokenKind::RepeatPrefix { count, .. } => {
                    format!(
                        "unexpected repeat count '{count}*' outside value list"
                    )
                }
                _ => "unexpected token outside assignment".to_owned(),
            };
            return Err(SchemaError::ParseError {
                line_number: tok.line_number,
                message: msg,
            });
        };
        let (name, dims, rhs_start, lhs_line) = lhs_res?;
        let (items, next_idx) =
            parse_rhs_items(tokens, rhs_start, name, lhs_line)?;
        expand_assignment(name, lhs_line, &dims, &items, &mut assigns)?;
        idx = next_idx;
    }
    Ok(assigns)
}

/// Parse one or more Fortran `&namelist ... /` blocks from `inlist` text.
///
/// Supports `/` and `&end` terminators, multi-line value lists, Fortran
/// `&` continuations, signed subscripts, comma-separated array-slice value
/// lists (`key(1:6) = v1, v2, ...` and strided slices `lo:hi:step`), repeat
/// counts (`n*val`), null items (`, ,`, `n*`), quoted string literals with
/// doubled-quote escapes, and broadcast slices (`key(1:6) = val`).
///
/// # Errors
/// Returns [`SchemaError::ParseError`] on malformed namelist syntax.
///
/// # Examples
/// ```
/// use flexmesa_schema::parse_namelist_text;
///
/// let blocks = parse_namelist_text("&eos\n  use_PC = .true.\n/\n").unwrap();
/// assert_eq!(blocks.len(), 1);
/// assert_eq!(blocks[0].name, "eos");
/// ```
pub fn parse_namelist_text(
    text: &str,
) -> Result<Vec<NamelistBlock>, SchemaError> {
    let mut cursor = Cursor::new(text);
    let mut blocks = Vec::new();
    while let Some((name, _line)) = cursor.read_group_header()? {
        let tokens = cursor.lex_block_body(&name)?;
        let assignments = parse_block_tokens(&tokens)?;
        blocks.push(NamelistBlock { name, assignments });
    }
    Ok(blocks)
}

/// Read and parse a MESA Fortran `inlist` file from disk.
///
/// # Errors
/// Returns [`SchemaError`] on I/O failure or malformed namelist syntax.
pub fn parse_namelist_file(
    path: &Path,
) -> Result<Vec<NamelistBlock>, SchemaError> {
    parse_namelist_text(&read_file_text(path)?)
}

/// Validate a [`NamelistBlock`] against a [`ControlDefault`] slice.
///
/// # Errors
/// Returns [`SchemaError::UnknownControl`] or [`SchemaError::ValidationError`]
/// if an assignment references an unknown control, indexes a scalar, or
/// supplies an incompatible value type.
///
/// # Examples
/// ```
/// use flexmesa_schema::{
///     parse_defaults_text, parse_namelist_text, validate_namelist_block,
/// };
///
/// let schema = parse_defaults_text("  use_PC = .false.\n").unwrap();
/// let blocks = parse_namelist_text("&eos\n  use_PC = .true.\n/\n").unwrap();
/// assert!(validate_namelist_block(&blocks[0], &schema, false).is_ok());
/// ```
pub fn validate_namelist_block(
    block: &NamelistBlock,
    schema: &[ControlDefault],
    strict_numeric_types: bool,
) -> Result<(), SchemaError> {
    let by_name: HashMap<String, &ControlDefault> = schema
        .iter()
        .map(|c| (c.name.to_ascii_lowercase(), c))
        .collect();
    for assign in &block.assignments {
        let Some(&spec) = by_name.get(&assign.name.to_ascii_lowercase()) else {
            return Err(SchemaError::UnknownControl {
                namelist: block.name.clone(),
                control: assign.name.clone(),
                line_number: assign.line_number,
            });
        };
        if assign.subscript.is_some() && !spec.is_array {
            return Err(SchemaError::ValidationError {
                control: spec.name.clone(),
                line_number: assign.line_number,
                message: "scalar control cannot be indexed".to_owned(),
            });
        }
        let actual_t = assign.value.control_type();
        let expected_t = spec.control_type;
        let ok = actual_t == expected_t
            || (expected_t == ControlType::Float
                && actual_t == ControlType::Integer)
            || (!strict_numeric_types
                && expected_t == ControlType::Integer
                && actual_t == ControlType::Float);
        if !ok {
            return Err(SchemaError::ValidationError {
                control: spec.name.clone(),
                line_number: assign.line_number,
                message: format!("expected {expected_t:?}, got {actual_t:?}"),
            });
        }
    }
    Ok(())
}
