// SPDX-License-Identifier: GPL-3.0-only
//! Fortran namelist input as GYRE's `read_*_par` routines consume it
//! (`read(unit, NML=...)`, gfortran 15.2.0 list-directed namelist rules),
//! restricted to the scalar forms GYRE's parameter groups declare.
//!
//! A file is split into its groups in file order; each `read_*_par`
//! routine then reads the occurrences of its own group in sequence
//! (gfortran skips groups with other names). Supported value forms:
//! integers, reals (integer literals, decimal and `E`/`D` exponents),
//! logicals (`.TRUE.`, `T`, `.false.`, ...) and quoted character strings
//! (`'...'` or `"..."`, doubled delimiters escaping). Array elements,
//! repeat counts and substrings are rejected with an explicit error.

use std::fmt;

/// A value as written in the namelist text, typed later by the reader.
#[derive(Debug, Clone, PartialEq)]
pub enum NmlValue {
    /// An unquoted token (number or logical).
    Token(String),
    /// A quoted character string (delimiters removed).
    Str(String),
}

/// One `&name ... /` group: its name and assignments in order.
#[derive(Debug, Clone, PartialEq)]
pub struct NmlGroup {
    /// Group name, lower case.
    pub name: String,
    /// `(object name in lower case, value)` in the order written.
    pub items: Vec<(String, NmlValue)>,
}

/// Namelist parse or conversion error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NmlError(pub String);

impl fmt::Display for NmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "namelist: {}", self.0)
    }
}

impl std::error::Error for NmlError {}

fn err<T>(msg: impl Into<String>) -> Result<T, NmlError> {
    Err(NmlError(msg.into()))
}

/// Split namelist text into groups.
///
/// # Errors
///
/// [`NmlError`] for unterminated groups or strings, or for forms outside
/// the supported subset.
pub fn parse_groups(text: &str) -> Result<Vec<NmlGroup>, NmlError> {
    // Tokenise: strings, '&name', '/', '=', ',', bare tokens; '!' comments.
    #[derive(Debug)]
    enum Tok {
        Amp(String),
        Slash,
        Eq,
        Comma,
        Word(String),
        Str(String),
    }
    let mut toks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '!' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '\'' || c == '"' {
            let q = c;
            let mut s = String::new();
            i += 1;
            loop {
                if i >= chars.len() {
                    return err("unterminated character string");
                }
                if chars[i] == q {
                    if i + 1 < chars.len() && chars[i + 1] == q {
                        s.push(q);
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                s.push(chars[i]);
                i += 1;
            }
            toks.push(Tok::Str(s));
        } else if c == '&' || c == '$' {
            let mut s = String::new();
            i += 1;
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '_')
            {
                s.push(chars[i].to_ascii_lowercase());
                i += 1;
            }
            toks.push(Tok::Amp(s));
        } else if c == '/' {
            toks.push(Tok::Slash);
            i += 1;
        } else if c == '=' {
            toks.push(Tok::Eq);
            i += 1;
        } else if c == ',' || c == ';' {
            toks.push(Tok::Comma);
            i += 1;
        } else {
            let mut s = String::new();
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !matches!(chars[i], '=' | ',' | ';' | '/' | '!' | '&')
            {
                s.push(chars[i]);
                i += 1;
            }
            toks.push(Tok::Word(s));
        }
    }

    let mut groups = Vec::new();
    let mut k = 0;
    while k < toks.len() {
        let Tok::Amp(name) = &toks[k] else {
            return err(format!(
                "text outside a namelist group: {:?}",
                toks[k]
            ));
        };
        if name == "end" {
            return err("unexpected &end");
        }
        let mut items = Vec::new();
        k += 1;
        loop {
            match toks.get(k) {
                None => return err(format!("group &{name} is not terminated")),
                Some(Tok::Slash) => {
                    k += 1;
                    break;
                }
                Some(Tok::Amp(e)) if e == "end" => {
                    k += 1;
                    break;
                }
                Some(Tok::Comma) => k += 1,
                Some(Tok::Word(obj)) => {
                    let obj_l = obj.to_ascii_lowercase();
                    if obj_l.contains('(') || obj_l.contains('%') {
                        return err(format!(
                            "array element or component '{obj}' in &{name} is not supported"
                        ));
                    }
                    if !matches!(toks.get(k + 1), Some(Tok::Eq)) {
                        return err(format!(
                            "expected '=' after '{obj}' in &{name}"
                        ));
                    }
                    let val = match toks.get(k + 2) {
                        Some(Tok::Word(w)) => {
                            if w.contains('*') {
                                return err(format!(
                                    "repeat count in '{w}' (&{name}) is not supported"
                                ));
                            }
                            NmlValue::Token(w.clone())
                        }
                        Some(Tok::Str(s)) => NmlValue::Str(s.clone()),
                        other => {
                            return err(format!(
                            "missing value for '{obj}' in &{name}: {other:?}"
                        ))
                        }
                    };
                    // A second value before the next object would be an
                    // array assignment.
                    if let Some(Tok::Word(_) | Tok::Str(_)) = toks.get(k + 3) {
                        if !matches!(toks.get(k + 4), Some(Tok::Eq)) {
                            return err(format!(
                                "multiple values for '{obj}' in &{name} are not supported"
                            ));
                        }
                    }
                    if let Some(Tok::Comma) = toks.get(k + 3) {
                        if let Some(Tok::Word(_) | Tok::Str(_)) =
                            toks.get(k + 4)
                        {
                            if !matches!(toks.get(k + 5), Some(Tok::Eq)) {
                                return err(format!(
                                    "multiple values for '{obj}' in &{name} are not supported"
                                ));
                            }
                        }
                    }
                    items.push((obj_l, val));
                    k += 3;
                }
                Some(t) => return err(format!("unexpected {t:?} in &{name}")),
            }
        }
        groups.push(NmlGroup {
            name: name.clone(),
            items,
        });
    }
    Ok(groups)
}

/// Typed conversion of namelist values (list-directed input rules).
impl NmlValue {
    /// `REAL(RD)` input: integer or real literal, `D` or `E` exponent.
    ///
    /// # Errors
    ///
    /// [`NmlError`] if the value is not a numeric literal.
    pub fn to_real(&self, obj: &str) -> Result<f64, NmlError> {
        match self {
            Self::Token(t) => {
                let s = t.replace(['d', 'D'], "e");
                let s = s.strip_prefix('+').unwrap_or(&s);
                if s.is_empty()
                    || !s.chars().all(|c| {
                        c.is_ascii_digit()
                            || matches!(c, '.' | 'e' | 'E' | '+' | '-')
                    })
                {
                    return err(format!("bad real value '{t}' for '{obj}'"));
                }
                s.parse::<f64>().or_else(|_| {
                    err(format!("bad real value '{t}' for '{obj}'"))
                })
            }
            Self::Str(s) => {
                err(format!("character value '{s}' for real '{obj}'"))
            }
        }
    }

    /// `INTEGER` input.
    ///
    /// # Errors
    ///
    /// [`NmlError`] if the value is not an integer literal.
    pub fn to_int(&self, obj: &str) -> Result<i32, NmlError> {
        match self {
            Self::Token(t) => {
                let s = t.strip_prefix('+').unwrap_or(t);
                s.parse::<i32>().or_else(|_| {
                    err(format!("bad integer value '{t}' for '{obj}'"))
                })
            }
            Self::Str(s) => {
                err(format!("character value '{s}' for integer '{obj}'"))
            }
        }
    }

    /// `LOGICAL` input: an optional leading `.`, then `T` or `F`.
    ///
    /// # Errors
    ///
    /// [`NmlError`] for anything else.
    pub fn to_logical(&self, obj: &str) -> Result<bool, NmlError> {
        match self {
            Self::Token(t) => {
                let u = t.to_ascii_uppercase();
                let u = u.strip_prefix('.').unwrap_or(&u);
                match u.chars().next() {
                    Some('T') => Ok(true),
                    Some('F') => Ok(false),
                    _ => err(format!("bad logical value '{t}' for '{obj}'")),
                }
            }
            Self::Str(s) => {
                err(format!("character value '{s}' for logical '{obj}'"))
            }
        }
    }

    /// `CHARACTER(len)` input: truncated to `len` characters, trailing
    /// blanks removed (Fortran comparisons ignore them).
    ///
    /// # Errors
    ///
    /// [`NmlError`] for an unquoted value (gfortran accepts some unquoted
    /// strings; GYRE's inputs quote them, so only that form is supported).
    pub fn to_string_len(
        &self,
        obj: &str,
        len: usize,
    ) -> Result<String, NmlError> {
        match self {
            Self::Str(s) => {
                let t: String = s.chars().take(len).collect();
                Ok(t.trim_end_matches(' ').to_string())
            }
            Self::Token(t) => err(format!(
                "unquoted character value '{t}' for '{obj}' is not supported"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_values_and_comments_parse_in_file_order() {
        let g = parse_groups(
            "&model /\n&mode\n  l = 0 ! degree\n  tag = 'l=0'\n/\n&num diff_scheme = \"COLLOC_GL4\", n_iter_max=5 /\n&osc adiabatic = .TRUE. x_ref = 1d0 /",
        )
        .expect("parse");
        assert_eq!(g.len(), 4);
        assert_eq!(g[0].name, "model");
        assert_eq!(g[0].items.len(), 0);
        assert_eq!(g[1].items[0], ("l".into(), NmlValue::Token("0".into())));
        assert_eq!(g[1].items[1], ("tag".into(), NmlValue::Str("l=0".into())));
        assert_eq!(g[2].items[1].1.to_int("n_iter_max").unwrap(), 5);
        assert!(g[3].items[0].1.to_logical("adiabatic").unwrap());
        assert_eq!(g[3].items[1].1.to_real("x_ref").unwrap(), 1.0);
    }

    #[test]
    fn unsupported_forms_are_rejected_with_errors() {
        assert!(parse_groups("&grid x(1) = 2 /").is_err());
        assert!(parse_groups("&grid w_osc = 2 3 /").is_err());
        assert!(parse_groups("&grid w_osc = 3*2 /").is_err());
        assert!(parse_groups("&grid w_osc = 2").is_err());
        assert!(parse_groups("&grid tag = 'abc /").is_err());
    }
}
