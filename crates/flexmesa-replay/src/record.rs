// SPDX-License-Identifier: GPL-3.0-or-later
//! Capture-file format of the MESA record/replay overlay (`#552`).
//!
//! A capture file is plain text, one record per line, written by the overlay
//! of `benchmarks/fortran_mesa/replay/replay_patch.py`:
//!
//! ```text
//! # <provenance header lines>
//! RUN
//! SCHEMA <family> <name>:<type> ...
//! <family> <field> <field> ...
//! ```
//!
//! Field types are `i` (integer), `l` (logical `T`/`F`), `s` (string without
//! blanks), `r` (`real(dp)` as its IEEE-754 binary64 bit pattern in
//! hexadecimal) and `a` (`auto_diff_real_star_order1`: the value's bit
//! pattern followed by `/<lane>:<bits>` for every nonzero-bit `d1Array`
//! lane, 1-based, of [`STAR_AD_LANES`]).

use std::collections::HashMap;
use std::fmt;

use flexmesa_autodiff::DualOrder1;

/// Number of `d1Array` lanes of MESA's `auto_diff_real_star_order1`
/// (`auto_diff/private/auto_diff_real_star_order1_module.f90:87`, the
/// `auto_diff_star_num_vars = i_xtra2_p1 = 33` slots of
/// `star_data/public/star_data_def.inc:255-288`).
pub const STAR_AD_LANES: usize = 33;

/// MESA's `auto_diff_real_star_order1` in MESA's own lane layout.
///
/// The `turb` kernels are generic over the lane count and treat every lane
/// alike, so replaying in MESA's 33-lane layout needs no mapping onto
/// `flexmesa-star`'s 24-lane `StarAdOrder1`.
pub type MesaAd = DualOrder1<STAR_AD_LANES>;

/// Error in a capture file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError(pub String);

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capture format error: {}", self.0)
    }
}

impl std::error::Error for FormatError {}

/// Field type of a schema entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// Integer (`i`).
    Int,
    /// Logical (`l`).
    Logical,
    /// String without blanks (`s`).
    Str,
    /// `real(dp)` bit pattern (`r`).
    Real,
    /// `auto_diff_real_star_order1` value and nonzero lanes (`a`).
    Ad,
}

impl FieldType {
    fn parse(code: &str) -> Result<Self, FormatError> {
        match code {
            "i" => Ok(Self::Int),
            "l" => Ok(Self::Logical),
            "s" => Ok(Self::Str),
            "r" => Ok(Self::Real),
            "a" => Ok(Self::Ad),
            _ => Err(FormatError(format!("unknown field type `{code}`"))),
        }
    }
}

/// One decoded field value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Integer.
    Int(i64),
    /// Logical.
    Logical(bool),
    /// String.
    Str(String),
    /// Real, exact bits.
    Real(f64),
    /// Auto-diff value with all [`STAR_AD_LANES`] lanes, exact bits.
    Ad(Box<MesaAd>),
    /// `NA`: the writer had no value for this field (any type).
    Missing,
}

/// Ordered field list of one record family, from its `SCHEMA` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// Field names and types in record order.
    pub fields: Vec<(String, FieldType)>,
}

impl Schema {
    /// Parses the `<name>:<type>` tokens after `SCHEMA <family>`.
    ///
    /// # Errors
    /// Returns [`FormatError`] for a malformed token or unknown type.
    pub fn parse<'a>(
        tokens: impl Iterator<Item = &'a str>,
    ) -> Result<Self, FormatError> {
        let mut fields = Vec::new();
        for tok in tokens {
            let (name, code) = tok.split_once(':').ok_or_else(|| {
                FormatError(format!("schema token `{tok}` lacks a type"))
            })?;
            fields.push((name.to_string(), FieldType::parse(code)?));
        }
        Ok(Self { fields })
    }

    /// Checks that this schema lists exactly `expected` (as written in the
    /// overlay patch), in order.
    ///
    /// # Errors
    /// Returns [`FormatError`] naming the first difference.
    pub fn check(
        &self,
        family: &str,
        expected: &str,
    ) -> Result<(), FormatError> {
        let want = Self::parse(expected.split_whitespace())?;
        if *self != want {
            return Err(FormatError(format!(
                "{family} schema in the capture differs from the one this \
                 replay tool implements:\n  capture: {:?}\n  expected: {:?}",
                self.fields, want.fields
            )));
        }
        Ok(())
    }
}

/// One record: its family and fields by name.
#[derive(Debug, Clone)]
pub struct Record {
    /// Record family (`TDC`, `MLT`, ...).
    pub family: String,
    /// 1-based line number in the capture file.
    pub line: usize,
    /// Index of the `RUN` block (process) the record belongs to, from 1.
    pub run: usize,
    values: HashMap<String, Value>,
}

/// Token written for a field the writer has no value for.
pub const NOT_AVAILABLE: &str = "NA";

/// Encodes a real as the overlay does (`Z16.16`: 16 upper-case hex digits).
#[must_use]
pub fn encode_real(x: f64) -> String {
    format!("{:016X}", x.to_bits())
}

/// Encodes an auto-diff value as the overlay does: the value's bits, then
/// `/<lane>:<bits>` for every lane (1-based) whose bits are not `+0.0`.
#[must_use]
pub fn encode_ad(x: &MesaAd) -> String {
    use std::fmt::Write as _;
    let mut s = encode_real(x.val);
    for (i, d) in x.d1.iter().enumerate() {
        if d.to_bits() != 0 {
            let _ = write!(s, "/{}:{:016X}", i + 1, d.to_bits());
        }
    }
    s
}

fn bits(tok: &str) -> Result<f64, FormatError> {
    u64::from_str_radix(tok, 16)
        .map(f64::from_bits)
        .map_err(|e| FormatError(format!("bad hex `{tok}`: {e}")))
}

fn parse_ad(tok: &str) -> Result<MesaAd, FormatError> {
    let mut parts = tok.split('/');
    let val = bits(parts.next().unwrap_or(""))?;
    let mut d1 = [0.0_f64; STAR_AD_LANES];
    for lane in parts {
        let (idx, hex) = lane
            .split_once(':')
            .ok_or_else(|| FormatError(format!("bad lane `{lane}`")))?;
        let idx: usize = idx
            .parse()
            .map_err(|e| FormatError(format!("bad lane index `{idx}`: {e}")))?;
        if idx == 0 || idx > STAR_AD_LANES {
            return Err(FormatError(format!("lane {idx} out of range")));
        }
        d1[idx - 1] = bits(hex)?;
    }
    Ok(MesaAd::new(val, d1))
}

impl Record {
    /// Decodes the tokens after the family tag against `schema`.
    ///
    /// # Errors
    /// Returns [`FormatError`] on a field-count mismatch or a bad token.
    pub fn parse<'a>(
        family: &str,
        line: usize,
        run: usize,
        schema: &Schema,
        tokens: impl Iterator<Item = &'a str>,
    ) -> Result<Self, FormatError> {
        let toks: Vec<&str> = tokens.collect();
        if toks.len() != schema.fields.len() {
            return Err(FormatError(format!(
                "line {line}: {family} record has {} fields, schema {}",
                toks.len(),
                schema.fields.len()
            )));
        }
        let mut values = HashMap::with_capacity(toks.len());
        for (tok, (name, ty)) in toks.iter().zip(&schema.fields) {
            if *tok == NOT_AVAILABLE {
                values.insert(name.clone(), Value::Missing);
                continue;
            }
            let v = match ty {
                FieldType::Int => Value::Int(tok.parse().map_err(|e| {
                    FormatError(format!("line {line}: {name}=`{tok}`: {e}"))
                })?),
                FieldType::Logical => match *tok {
                    "T" => Value::Logical(true),
                    "F" => Value::Logical(false),
                    _ => {
                        return Err(FormatError(format!(
                            "line {line}: {name}=`{tok}` is not T or F"
                        )))
                    }
                },
                FieldType::Str => Value::Str((*tok).to_string()),
                FieldType::Real => Value::Real(bits(tok)?),
                FieldType::Ad => Value::Ad(Box::new(parse_ad(tok)?)),
            };
            values.insert(name.clone(), v);
        }
        Ok(Self {
            family: family.to_string(),
            line,
            run,
            values,
        })
    }

    /// Field by name, if the schema has it.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    fn get(&self, name: &str) -> Result<&Value, FormatError> {
        self.values.get(name).ok_or_else(|| {
            FormatError(format!("line {}: no field `{name}`", self.line))
        })
    }

    fn wrong(&self, name: &str, want: &str) -> FormatError {
        FormatError(format!("line {}: field `{name}` is not {want}", self.line))
    }

    /// Integer field.
    ///
    /// # Errors
    /// Returns [`FormatError`] if absent or of another type.
    pub fn int(&self, name: &str) -> Result<i64, FormatError> {
        match self.get(name)? {
            Value::Int(v) => Ok(*v),
            _ => Err(self.wrong(name, "an integer")),
        }
    }

    /// Logical field.
    ///
    /// # Errors
    /// Returns [`FormatError`] if absent or of another type.
    pub fn logical(&self, name: &str) -> Result<bool, FormatError> {
        match self.get(name)? {
            Value::Logical(v) => Ok(*v),
            _ => Err(self.wrong(name, "a logical")),
        }
    }

    /// String field.
    ///
    /// # Errors
    /// Returns [`FormatError`] if absent or of another type.
    pub fn str(&self, name: &str) -> Result<&str, FormatError> {
        match self.get(name)? {
            Value::Str(v) => Ok(v),
            _ => Err(self.wrong(name, "a string")),
        }
    }

    /// Real field.
    ///
    /// # Errors
    /// Returns [`FormatError`] if absent or of another type.
    pub fn real(&self, name: &str) -> Result<f64, FormatError> {
        match self.get(name)? {
            Value::Real(v) => Ok(*v),
            _ => Err(self.wrong(name, "a real")),
        }
    }

    /// Auto-diff field.
    ///
    /// # Errors
    /// Returns [`FormatError`] if absent or of another type.
    pub fn ad(&self, name: &str) -> Result<MesaAd, FormatError> {
        match self.get(name)? {
            Value::Ad(v) => Ok(**v),
            _ => Err(self.wrong(name, "an auto_diff value")),
        }
    }

    /// Call context `site model solver_iter k` for reports.
    #[must_use]
    pub fn context(&self) -> String {
        let f =
            |n: &str| self.int(n).map_or("?".to_string(), |v| v.to_string());
        format!(
            "{} site={} run={} model={} solver_iter={} k={} (line {})",
            self.family,
            f("site"),
            self.run,
            f("model"),
            f("solver_iter"),
            f("k"),
            self.line
        )
    }
}

/// Streaming reader of a capture file.
///
/// Header (`#`) lines are kept in [`CaptureReader::header`]; `RUN` lines
/// advance the run index; `SCHEMA` lines define (or must repeat) a family's
/// field list; every other line is a record of a family with a schema.
pub struct CaptureReader<R> {
    lines: std::io::Lines<R>,
    line_no: usize,
    run: usize,
    schemas: HashMap<String, Schema>,
    /// Provenance header lines, without the leading `# `.
    pub header: Vec<String>,
}

impl<R: std::io::BufRead> CaptureReader<R> {
    /// Wraps a buffered reader.
    pub fn new(reader: R) -> Self {
        Self {
            lines: reader.lines(),
            line_no: 0,
            run: 0,
            schemas: HashMap::new(),
            header: Vec::new(),
        }
    }

    /// Schema of `family`, once its `SCHEMA` line has been read.
    #[must_use]
    pub fn schema(&self, family: &str) -> Option<&Schema> {
        self.schemas.get(family)
    }

    /// Returns the next record, or `None` at the end of the file.
    ///
    /// # Errors
    /// Returns [`FormatError`] on an I/O error, a record before its schema,
    /// a schema that changes between runs, or a malformed line.
    pub fn next_record(&mut self) -> Result<Option<Record>, FormatError> {
        for line in self.lines.by_ref() {
            self.line_no += 1;
            let line = line.map_err(|e| FormatError(format!("read: {e}")))?;
            if let Some(h) = line.strip_prefix('#') {
                self.header.push(h.trim().to_string());
                continue;
            }
            let mut toks = line.split_whitespace();
            let Some(tag) = toks.next() else { continue };
            match tag {
                "RUN" => self.run += 1,
                "SCHEMA" => {
                    let family = toks.next().ok_or_else(|| {
                        FormatError(format!(
                            "line {}: bare SCHEMA",
                            self.line_no
                        ))
                    })?;
                    let schema = Schema::parse(toks)?;
                    if let Some(old) = self.schemas.get(family) {
                        if *old != schema {
                            return Err(FormatError(format!(
                                "line {}: {family} schema changed",
                                self.line_no
                            )));
                        }
                    } else {
                        self.schemas.insert(family.to_string(), schema);
                    }
                }
                family => {
                    let schema = self.schemas.get(family).ok_or_else(|| {
                        FormatError(format!(
                            "line {}: record `{family}` before its SCHEMA",
                            self.line_no
                        ))
                    })?;
                    return Record::parse(
                        family,
                        self.line_no,
                        self.run,
                        schema,
                        toks,
                    )
                    .map(Some);
                }
            }
        }
        Ok(None)
    }
}
