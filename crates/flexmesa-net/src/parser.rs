// SPDX-License-Identifier: GPL-3.0-or-later
//! Embedded `.net` bundle loader and recursive `.net` tokenizer/parser
//! (`net/private/net_initialize.f90` and `net/private/net_approx21.f90`).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use flexmesa_chem::{chem_get_element_id, NuclideCatalog};
use flexmesa_rates::ReactionInfo;

use crate::reaction_synth::synthesize_reaction_info;
use crate::types::NetError;

const RAW_NETS_BUNDLE: &[u8] = include_bytes!("../data/nets.bin");

static NET_FILES: OnceLock<HashMap<String, String>> = OnceLock::new();

/// Returns the embedded map of all 83 MESA `.net` and `add_*` files.
#[must_use]
pub fn builtin_net_files() -> &'static HashMap<String, String> {
    NET_FILES.get_or_init(|| {
        let mut pos = 8_usize; // skip b"FNET0001"
        let read_u32 = |p: &mut usize| -> usize {
            let v = u32::from_le_bytes(
                RAW_NETS_BUNDLE[*p..*p + 4].try_into().unwrap_or([0; 4]),
            ) as usize;
            *p += 4;
            v
        };
        let count = read_u32(&mut pos);
        let mut map = HashMap::with_capacity(count);
        for _ in 0..count {
            let nlen = read_u32(&mut pos);
            let name =
                String::from_utf8_lossy(&RAW_NETS_BUNDLE[pos..pos + nlen])
                    .into_owned();
            pos += nlen;
            let clen = read_u32(&mut pos);
            let content =
                String::from_utf8_lossy(&RAW_NETS_BUNDLE[pos..pos + clen])
                    .into_owned();
            pos += clen;
            map.insert(name, content);
        }
        map
    })
}

/// Intermediate builder state populated while parsing `.net` files.
#[derive(Debug, Clone)]
pub struct ParsedNetDef {
    /// Primary `.net` filename.
    pub net_filename: String,
    /// Set of active 1-based `chem_id` values.
    pub active_isos: HashSet<usize>,
    /// Active reactions keyed by reaction handle name.
    pub active_reactions: HashMap<String, ReactionInfo>,
    /// Dynamic reverse reaction pairs `(r1, r2)` recorded during parsing.
    pub reverse_pairs: Vec<(String, String)>,
    /// Whether `approx21` or `approx21_plus_co56` was requested.
    pub doing_approx21: bool,
    /// Whether `co56` is included in `approx21`.
    pub add_co56_to_approx21: bool,
    /// 1-based `chem_id` of the `approx21` neutron-rich sink isotope.
    pub approx21_ye_iso: usize,
    /// Neutron count offset `fe56ec_n_neut` for `approx21`.
    pub fe56ec_n_neut: i32,
    next_dynamic_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NetToken {
    Name(String),
    Quoted(String),
    LParen,
    RParen,
    Comma,
}

/// `character (len=256) :: buffer, string` in `do_read_net_file`
/// (`net/private/net_initialize.f90:276`).
const NET_TOKEN_LEN: usize = 256;

/// Formatted records of a `.net` file as `read(iounit,fmt='(a)')` returns
/// them under the MESA SDK's gfortran 15.2.0 runtime: a record ends at `\n`,
/// at `\r\n` or at a lone `\r`, and the terminator is not part of the record
/// (compiler runtime behaviour, not MESA source). A terminator at the very
/// end does not start an empty record.
fn gfortran_records(src: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = src;
    while !rest.is_empty() {
        let bytes = rest.as_bytes();
        let Some(end) = bytes.iter().position(|&c| c == b'\n' || c == b'\r')
        else {
            out.push(rest);
            break;
        };
        out.push(&rest[..end]);
        let skip = if bytes[end] == b'\r' && bytes.get(end + 1) == Some(&b'\n')
        {
            2
        } else {
            1
        };
        rest = &rest[end + skip..];
    }
    out
}

/// The `.net` token stream as upstream `token` produces it
/// (`utils/public/utils_lib.f90:567-667`), transcribed on bytes (#666).
///
/// Each record is read into a 256-byte buffer (a longer record is cut) and
/// `n = len_trim(buffer)` drops trailing blanks only. Records end where the
/// gfortran runtime ends a formatted record ([`gfortran_records`]). At a
/// token start,
/// tabs, blanks and `&` are skipped and `!` ends the line; `(`, `)` and `,`
/// are single tokens; a quote starts a string that runs to the matching
/// quote or the end of the line; anything else starts a name that ends only
/// before a blank, `(`, `)` or `,`, so a tab, `!`, `&` or quote after
/// its first byte stays in it. Names and strings are cut to 256 bytes and
/// returned without the Fortran blank padding.
fn tokenize_net(src: &str) -> Vec<NetToken> {
    let mut tokens = Vec::new();
    for raw_line in gfortran_records(src) {
        let bytes = raw_line.as_bytes();
        let buffer = &bytes[..bytes.len().min(NET_TOKEN_LEN)];
        let n = buffer.iter().rposition(|&c| c != b' ').map_or(0, |p| p + 1);
        // `i` is the 1-based index of the last byte read (`:589-665`).
        let mut i = 0_usize;
        while i < n {
            i += 1;
            let c = buffer[i - 1];
            match c {
                b'\t' | b' ' | b'&' => {}
                b'!' => i = n,
                b'(' => tokens.push(NetToken::LParen),
                b')' => tokens.push(NetToken::RParen),
                b',' => tokens.push(NetToken::Comma),
                b'"' | b'\'' => {
                    let mut out = Vec::new();
                    loop {
                        i += 1;
                        if i > n
                            || buffer[i - 1] == c
                            || out.len() >= NET_TOKEN_LEN
                        {
                            break;
                        }
                        out.push(buffer[i - 1]);
                    }
                    let text = String::from_utf8_lossy(&out);
                    tokens.push(NetToken::Quoted(
                        text.trim_end_matches(' ').to_owned(),
                    ));
                }
                _ => {
                    let j1 = i;
                    let mut j2 = i;
                    while i < n
                        && !matches!(buffer[i], b' ' | b'(' | b')' | b',')
                    {
                        i += 1;
                        j2 = i;
                    }
                    let l = (j2 - j1 + 1).min(NET_TOKEN_LEN);
                    let name = &buffer[j1 - 1..j1 - 1 + l];
                    tokens.push(NetToken::Name(
                        String::from_utf8_lossy(name).into_owned(),
                    ));
                }
            }
        }
    }
    tokens
}

/// `read(string,fmt=*,iostat=ierr) A` for one integer
/// (`net/private/net_initialize.f90:509, 521, 532`): a list-directed read
/// takes the first field, which ends at a blank, tab, comma or slash, so
/// `"2\t"` (a name with a tab kept in it) reads as 2.
fn list_directed_usize(s: &str) -> Option<usize> {
    let t = s.trim_start_matches([' ', '\t']);
    let end = t.find([' ', '\t', ',', '/']).unwrap_or(t.len());
    t[..end].parse().ok()
}

impl ParsedNetDef {
    /// Parses a named `.net` file from the embedded bundle.
    pub fn parse_from_name(net_filename: &str) -> Result<Self, NetError> {
        Self::parse_from_name_or_dir(net_filename, None)
    }

    /// Parses a named `.net` file from the embedded bundle or an optional
    /// case-local directory `search_dir`.
    pub fn parse_from_name_or_dir(
        net_filename: &str,
        search_dir: Option<&Path>,
    ) -> Result<Self, NetError> {
        let mut def = Self {
            net_filename: net_filename.trim().to_string(),
            active_isos: HashSet::new(),
            active_reactions: HashMap::new(),
            reverse_pairs: Vec::new(),
            doing_approx21: false,
            add_co56_to_approx21: false,
            approx21_ye_iso: 0,
            fe56ec_n_neut: 0,
            next_dynamic_id: 317,
        };
        def.read_net_file(net_filename.trim(), search_dir)?;
        if def.doing_approx21 {
            def.mark_approx21()?;
        }
        Ok(def)
    }

    fn read_net_file(
        &mut self,
        filename: &str,
        search_dir: Option<&Path>,
    ) -> Result<(), NetError> {
        self.read_net_file_with_depth(filename, search_dir, 0)?;
        self.validate_al26_isomers()
    }

    fn validate_al26_isomers(&self) -> Result<(), NetError> {
        let nuclides = NuclideCatalog::builtin_ref();
        let req_iso = |name: &str| {
            nuclides
                .get_nuclide_index(name)
                .ok_or_else(|| NetError::InvalidReactionOrIso(name.to_string()))
        };
        let has_al26 = self.active_isos.contains(&req_iso("al26")?);
        let has_al26_1 = self.active_isos.contains(&req_iso("al26-1")?);
        let has_al26_2 = self.active_isos.contains(&req_iso("al26-2")?);
        if has_al26 && (has_al26_1 || has_al26_2) {
            return Err(NetError::ParseError(
                "cannot specify al26 and al26-1 or al26-2".to_string(),
            ));
        }
        if has_al26_1 && !has_al26_2 {
            return Err(NetError::ParseError(
                "must specify al26-2 if al26-1 is set".to_string(),
            ));
        }
        if has_al26_2 && !has_al26_1 {
            return Err(NetError::ParseError(
                "must specify al26-1 if al26-2 is set".to_string(),
            ));
        }
        Ok(())
    }

    fn read_net_file_with_depth(
        &mut self,
        filename: &str,
        search_dir: Option<&Path>,
        depth: usize,
    ) -> Result<(), NetError> {
        const MAX_NET_INCLUDE_DEPTH: usize = 32;
        let trimmed = filename.trim();
        if trimmed.is_empty() {
            return Err(NetError::UnknownNetFile(filename.to_string()));
        }
        if depth > MAX_NET_INCLUDE_DEPTH {
            return Err(NetError::ParseError(format!(
                "net include depth exceeded {MAX_NET_INCLUDE_DEPTH} while reading '{trimmed}'"
            )));
        }
        // Upstream MESA `net/private/net_initialize.f90:286-302`
        // (`do_read_net_file`): resolve `<dir>/<file>`, then
        // `<dir>/nets/<file>`, then built-in `data/net_data/nets/<file>`.
        let (local_path, nets_sub_path) = match search_dir {
            Some(dir) => (dir.join(trimmed), dir.join("nets").join(trimmed)),
            None => (
                Path::new(trimmed).to_path_buf(),
                Path::new("nets").join(trimmed),
            ),
        };
        let files = builtin_net_files();
        let mut owned_src = None;
        for candidate in [&local_path, &nets_sub_path] {
            if candidate.is_file() {
                owned_src =
                    Some(fs::read_to_string(candidate).map_err(|e| {
                        NetError::ParseError(format!(
                            "failed to read net file '{trimmed}': {e}"
                        ))
                    })?);
                break;
            }
        }
        let src: &str = if let Some(ref s) = owned_src {
            s.as_str()
        } else if let Some(s) = files.get(trimmed) {
            s.as_str()
        } else {
            return Err(NetError::UnknownNetFile(trimmed.to_string()));
        };
        let tokens = tokenize_net(src);
        let mut idx = 0_usize;
        while idx < tokens.len() {
            let NetToken::Name(cmd) = &tokens[idx] else {
                return Err(NetError::ParseError(format!(
                    "unexpected token in {filename} at index {idx}"
                )));
            };
            idx += 1;
            match cmd.as_str() {
                "add_iso" | "add_isos" => {
                    self.do_isos(&tokens, &mut idx, true, false)?;
                }
                "remove_iso" | "remove_isos" => {
                    self.do_isos(&tokens, &mut idx, false, false)?;
                }
                "add_reaction" | "add_reactions" => {
                    self.do_reactions(&tokens, &mut idx, true)?;
                }
                "remove_reaction" | "remove_reactions" => {
                    self.do_reactions(&tokens, &mut idx, false)?;
                }
                "add_iso_and_reactions" | "add_isos_and_reactions" => {
                    self.do_isos(&tokens, &mut idx, true, true)?;
                }
                "approx21" => self.do_approx21(&tokens, &mut idx, false)?,
                "approx21_plus_co56" => {
                    self.do_approx21(&tokens, &mut idx, true)?;
                }
                "include" => {
                    let Some(NetToken::Quoted(inc)) = tokens.get(idx) else {
                        return Err(NetError::ParseError(
                            "expected quoted filename after include"
                                .to_string(),
                        ));
                    };
                    let inc_name = inc.trim().to_string();
                    idx += 1;
                    self.read_net_file_with_depth(
                        &inc_name,
                        search_dir,
                        depth + 1,
                    )?;
                }
                other => {
                    return Err(NetError::ParseError(format!(
                        "unknown net command '{other}' in {filename}"
                    )))
                }
            }
        }
        Ok(())
    }

    fn do_approx21(
        &mut self,
        tokens: &[NetToken],
        idx: &mut usize,
        add_co56: bool,
    ) -> Result<(), NetError> {
        if tokens.get(*idx) != Some(&NetToken::LParen) {
            return Err(NetError::ParseError("expected '('".to_string()));
        }
        *idx += 1;
        let Some(NetToken::Name(iso_name)) = tokens.get(*idx) else {
            return Err(NetError::ParseError("expected iso name".to_string()));
        };
        *idx += 1;
        if tokens.get(*idx) != Some(&NetToken::RParen) {
            return Err(NetError::ParseError("expected ')'".to_string()));
        }
        *idx += 1;
        let nuclides = NuclideCatalog::builtin_ref();
        let lookup_nuclide = |name: &str| {
            nuclides
                .get_nuclide_index(name)
                .ok_or_else(|| NetError::InvalidReactionOrIso(name.to_string()))
        };
        let ye_id = lookup_nuclide(iso_name)?;
        let ife56 = lookup_nuclide("fe56")?;
        let e_ye = nuclides.get_by_id(ye_id).ok_or_else(|| {
            NetError::InvalidReactionOrIso(format!("id={ye_id}"))
        })?;
        let e_fe = nuclides.get_by_id(ife56).ok_or_else(|| {
            NetError::InvalidReactionOrIso("fe56".to_string())
        })?;
        self.approx21_ye_iso = ye_id;
        self.fe56ec_n_neut =
            (e_ye.n as i32 - e_fe.n as i32) - (e_fe.z as i32 - e_ye.z as i32);
        self.doing_approx21 = true;
        self.add_co56_to_approx21 = add_co56;
        Ok(())
    }

    fn do_isos(
        &mut self,
        tokens: &[NetToken],
        idx: &mut usize,
        add_flag: bool,
        add_rxns: bool,
    ) -> Result<(), NetError> {
        if tokens.get(*idx) != Some(&NetToken::LParen) {
            return Err(NetError::ParseError("expected '('".to_string()));
        }
        *idx += 1;
        let nuclides = NuclideCatalog::builtin_ref();
        while *idx < tokens.len() {
            if tokens[*idx] == NetToken::RParen {
                *idx += 1;
                return Ok(());
            }
            if tokens[*idx] == NetToken::Comma {
                *idx += 1;
                continue;
            }
            let NetToken::Name(name) = &tokens[*idx] else {
                return Err(NetError::ParseError(
                    "expected isotope or element name".to_string(),
                ));
            };
            *idx += 1;
            if let Some(cid) = nuclides.get_nuclide_index(name) {
                self.apply_iso(cid, add_flag);
                if add_rxns {
                    self.do_basic_reactions_for_iso(cid)?;
                }
            } else {
                let z = chem_get_element_id(name)
                    .or_else(|| list_directed_usize(name))
                    .ok_or_else(|| {
                        NetError::InvalidReactionOrIso(name.clone())
                    })?;
                let Some(NetToken::Name(a1_s)) = tokens.get(*idx) else {
                    return Err(NetError::ParseError(
                        "expected A1".to_string(),
                    ));
                };
                *idx += 1;
                let Some(NetToken::Name(a2_s)) = tokens.get(*idx) else {
                    return Err(NetError::ParseError(
                        "expected A2".to_string(),
                    ));
                };
                *idx += 1;
                let a1: usize = list_directed_usize(a1_s)
                    .ok_or_else(|| NetError::ParseError(a1_s.clone()))?;
                let a2: usize = list_directed_usize(a2_s)
                    .ok_or_else(|| NetError::ParseError(a2_s.clone()))?;
                for a in a1..=a2 {
                    let cid = nuclides
                        .lookup_zn_isomeric_state(z, a - z, 0)
                        .ok_or_else(|| {
                            NetError::InvalidReactionOrIso(format!(
                                "Z={z}, A={a}"
                            ))
                        })?;
                    self.apply_iso(cid, add_flag);
                    if add_rxns {
                        self.do_basic_reactions_for_iso(cid)?;
                    }
                }
            }
        }
        Err(NetError::ParseError("unclosed '(' in do_isos".to_string()))
    }

    fn try_pair(
        &mut self,
        in_id: usize,
        out_id: usize,
        op1: &str,
        op2: &str,
    ) -> Result<(), NetError> {
        let nuclides = NuclideCatalog::builtin_ref();
        let n1 = &nuclides
            .get_by_id(in_id)
            .ok_or_else(|| {
                NetError::InvalidReactionOrIso(format!("id={in_id}"))
            })?
            .name;
        let n2 = &nuclides
            .get_by_id(out_id)
            .ok_or_else(|| {
                NetError::InvalidReactionOrIso(format!("id={out_id}"))
            })?
            .name;
        let r1 = format!("r_{n1}_{op1}_{n2}");
        let ok1 = self.add_reaction_by_name(&r1).is_ok();
        if !op2.is_empty() {
            let r2 = format!("r_{n2}_{op2}_{n1}");
            let ok2 = self.add_reaction_by_name(&r2).is_ok();
            if ok1 && ok2 {
                self.reverse_pairs.push((r1, r2));
            }
        }
        Ok(())
    }

    fn do_basic_reactions_for_iso(
        &mut self,
        id: usize,
    ) -> Result<(), NetError> {
        let nuclides = NuclideCatalog::builtin_ref();
        let e = nuclides.get_by_id(id).ok_or_else(|| {
            NetError::InvalidReactionOrIso(format!("id={id}"))
        })?;
        let z = e.z as isize;
        let n = e.n as isize;
        let req_iso = |name: &str| {
            nuclides
                .get_nuclide_index(name)
                .ok_or_else(|| NetError::InvalidReactionOrIso(name.to_string()))
        };
        let ineut = req_iso("neut")?;
        let ih1 = req_iso("h1")?;
        let ih2 = req_iso("h2")?;
        let ihe4 = req_iso("he4")?;
        let ib8 = req_iso("b8")?;
        let ial26_1 = req_iso("al26-1")?;
        let ial26_2 = req_iso("al26-2")?;

        // Special-case multi-body and non-pattern reactions from upstream
        // `net/private/net_initialize.f90:756-799` (`special_loop`).
        for spec in SPECIAL_CASE_REACTIONS {
            if !spec.reactants.contains(&e.name.as_str()) {
                continue;
            }
            let mut all_active = true;
            for &rname in spec.reactants {
                let rid = req_iso(rname)?;
                if !self.active_isos.contains(&rid) {
                    all_active = false;
                    break;
                }
            }
            if !all_active {
                continue;
            }
            self.add_reaction_by_name(spec.r1)?;
            if !spec.r2.is_empty() {
                self.add_reaction_by_name(spec.r2)?;
                self.reverse_pairs
                    .push((spec.r1.to_string(), spec.r2.to_string()));
            }
        }

        let get_active = |z2: isize, n2: isize, active: &HashSet<usize>| {
            if z2 < 0 || n2 < 0 {
                return None;
            }
            let cid = nuclides.lookup_zn_isomeric_state(
                z2 as usize,
                n2 as usize,
                0,
            )?;
            active.contains(&cid).then_some(cid)
        };

        if self.active_isos.contains(&ih1) {
            if let Some(other) = get_active(z - 1, n, &self.active_isos) {
                self.try_pair(other, id, "pg", "gp")?;
            }
            if let Some(other) = get_active(z + 1, n, &self.active_isos) {
                self.try_pair(id, other, "pg", "gp")?;
            }
        }
        if self.active_isos.contains(&ihe4) {
            if id == ib8 {
                // Upstream net_initialize.f90:821-825 treats failure as fatal.
                self.add_reaction_by_name("r_b8_wk_he4_he4")?;
            }
            if let Some(other) = get_active(z - 2, n - 2, &self.active_isos) {
                self.try_pair(other, id, "ag", "ga")?;
            }
            if let Some(other) = get_active(z + 2, n + 2, &self.active_isos) {
                self.try_pair(id, other, "ag", "ga")?;
            }
        }
        if self.active_isos.contains(&ih1)
            && self.active_isos.contains(&ihe4)
            && id != ihe4
        {
            if let Some(other) = get_active(z - 1, n - 2, &self.active_isos) {
                self.try_pair(other, id, "ap", "pa")?;
            }
            if let Some(other) = get_active(z + 1, n + 2, &self.active_isos) {
                self.try_pair(id, other, "ap", "pa")?;
            }
        }
        if id != ih1
            && self.active_isos.contains(&ih1)
            && self.active_isos.contains(&ineut)
        {
            if let Some(other) = get_active(z - 1, n + 1, &self.active_isos) {
                self.try_pair(id, other, "np", "pn")?;
            }
            if id != ineut {
                if let Some(other) = get_active(z + 1, n - 1, &self.active_isos)
                {
                    self.try_pair(other, id, "np", "pn")?;
                }
            }
        }
        if id != ihe4
            && self.active_isos.contains(&ihe4)
            && self.active_isos.contains(&ineut)
        {
            if let Some(other) = get_active(z - 2, n - 1, &self.active_isos) {
                if other != ihe4 {
                    self.try_pair(id, other, "na", "an")?;
                }
            }
            if let Some(other) = get_active(z + 2, n + 1, &self.active_isos) {
                if other != ihe4 {
                    self.try_pair(other, id, "na", "an")?;
                }
            }
        }
        if self.active_isos.contains(&ineut) && id != ih2 {
            if let Some(other) = get_active(z, n - 1, &self.active_isos) {
                self.try_pair(other, id, "ng", "gn")?;
            }
        }
        if let Some(other) = get_active(z - 1, n + 1, &self.active_isos) {
            self.try_pair(id, other, "wk", "wk-minus")?;
        }
        if let Some(other) = get_active(z + 1, n - 1, &self.active_isos) {
            self.try_pair(other, id, "wk", "wk-minus")?;
        }
        // Upstream `net/private/net_initialize.f90:933-951` literally passes
        // `'wk-h1'` and `'wk-he4'` (hyphenated), which fail lookup against
        // `_wk_h1_` / `_wk_he4_` reaclib handles and silently cycle; keep
        // verbatim for strict MESA 26.4.1 parity.
        if self.active_isos.contains(&ih1) {
            if let Some(other) = get_active(z - 2, n + 1, &self.active_isos) {
                self.try_pair(id, other, "wk-h1", "")?;
            }
            if let Some(other) = get_active(z + 2, n - 1, &self.active_isos) {
                self.try_pair(other, id, "wk-h1", "")?;
            }
        }
        if self.active_isos.contains(&ihe4) {
            if let Some(other) = get_active(z - 3, n - 1, &self.active_isos) {
                self.try_pair(id, other, "wk-he4", "")?;
            }
            if let Some(other) = get_active(z + 3, n + 1, &self.active_isos) {
                self.try_pair(other, id, "wk-he4", "")?;
            }
        }
        // Al-26 ground/isomer internal transitions (`net_initialize.f90:954-960`).
        if id == ial26_1 {
            self.add_reaction_by_name("r_al26-1_to_al26-2")?;
        } else if id == ial26_2 {
            self.add_reaction_by_name("r_al26-2_to_al26-1")?;
        }
        Ok(())
    }

    fn apply_iso(&mut self, cid: usize, add_flag: bool) {
        if add_flag {
            self.active_isos.insert(cid);
        } else {
            self.active_isos.remove(&cid);
            self.active_reactions.retain(|_, info| {
                let in_match = (0..info.num_in)
                    .any(|k| usize::from(info.in_iso[k]) == cid);
                let out_match = (0..info.num_out)
                    .any(|k| usize::from(info.out_iso[k]) == cid);
                !(in_match || out_match)
            });
        }
    }

    fn do_reactions(
        &mut self,
        tokens: &[NetToken],
        idx: &mut usize,
        add_flag: bool,
    ) -> Result<(), NetError> {
        if tokens.get(*idx) != Some(&NetToken::LParen) {
            return Err(NetError::ParseError("expected '('".to_string()));
        }
        *idx += 1;
        while *idx < tokens.len() {
            if tokens[*idx] == NetToken::RParen {
                *idx += 1;
                return Ok(());
            }
            if tokens[*idx] == NetToken::Comma {
                *idx += 1;
                continue;
            }
            let NetToken::Name(rname) = &tokens[*idx] else {
                return Err(NetError::ParseError(
                    "expected reaction name".to_string(),
                ));
            };
            *idx += 1;
            if add_flag {
                self.add_reaction_by_name(rname)?;
            } else {
                self.active_reactions.remove(rname.as_str());
            }
        }
        Err(NetError::ParseError(
            "unclosed '(' in do_reactions".to_string(),
        ))
    }

    pub(crate) fn add_reaction_by_name(
        &mut self,
        rname: &str,
    ) -> Result<(), NetError> {
        if self.active_reactions.contains_key(rname) {
            return Ok(());
        }
        let id = self.next_dynamic_id;
        let info = synthesize_reaction_info(rname, id)?;
        if info.id == id {
            self.next_dynamic_id += 1;
        }
        self.active_reactions.insert(rname.to_string(), info);
        Ok(())
    }

    fn mark_approx21(&mut self) -> Result<(), NetError> {
        let nuclides = NuclideCatalog::builtin_ref();
        let ye_name = &nuclides
            .get_by_id(self.approx21_ye_iso)
            .ok_or_else(|| {
                NetError::InvalidReactionOrIso(format!(
                    "approx21 ye iso id={}",
                    self.approx21_ye_iso
                ))
            })?
            .name;
        #[rustfmt::skip]
        let mut isos = vec![
            "neut", "h1", "prot", "he3", "he4", "c12", "n14", "o16", "ne20",
            "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48",
            ye_name.as_str(), "fe52", "fe54", "ni56",
        ];
        if self.add_co56_to_approx21 {
            isos.push("co56");
        }
        isos.push("fe56");
        for name in isos {
            let cid = nuclides.get_nuclide_index(name).ok_or_else(|| {
                NetError::InvalidReactionOrIso(name.to_string())
            })?;
            self.active_isos.insert(cid);
        }

        #[rustfmt::skip]
        let mut rxns = vec![
            "r_he4_he4_he4_to_c12", "r_c12_to_he4_he4_he4", "r_c12_ag_o16",
            "r1212", "r1216", "r1616", "r_o16_ga_c12", "r_o16_ag_ne20",
            "r_ne20_ga_o16", "r_ne20_ag_mg24", "r_mg24_ga_ne20",
            "r_mg24_ag_si28", "r_si28_ga_mg24", "r_mg24_ap_al27",
            "r_al27_pa_mg24", "r_al27_pg_si28", "r_si28_gp_al27",
            "r_si28_ag_s32", "r_s32_ga_si28", "r_si28_ap_p31",
            "r_p31_pa_si28", "r_p31_pg_s32", "r_s32_gp_p31",
            "r_s32_ag_ar36", "r_ar36_ga_s32", "r_s32_ap_cl35",
            "r_cl35_pa_s32", "r_cl35_pg_ar36", "r_ar36_gp_cl35",
            "r_ar36_ag_ca40", "r_ca40_ga_ar36", "r_ar36_ap_k39",
            "r_k39_pa_ar36", "r_k39_pg_ca40", "r_ca40_gp_k39",
            "r_ca40_ag_ti44", "r_ti44_ga_ca40", "r_ca40_ap_sc43",
            "r_sc43_pa_ca40", "r_sc43_pg_ti44", "r_ti44_gp_sc43",
            "r_ti44_ag_cr48", "r_cr48_ga_ti44", "r_ti44_ap_v47",
            "r_v47_pa_ti44", "r_v47_pg_cr48", "r_cr48_gp_v47",
            "r_cr48_ag_fe52", "r_fe52_ga_cr48", "r_cr48_ap_mn51",
            "r_mn51_pa_cr48", "r_mn51_pg_fe52", "r_fe52_gp_mn51",
            "r_fe52_ag_ni56", "r_ni56_ga_fe52", "r_fe52_ap_co55",
            "r_co55_pa_fe52", "r_co55_pg_ni56", "r_ni56_gp_co55",
            "r_fe52_ng_fe53", "r_fe53_gn_fe52", "r_fe53_ng_fe54",
            "r_fe54_gn_fe53", "r_fe54_pg_co55", "r_co55_gp_fe54",
            "r_he3_ng_he4", "r_he4_gn_he3", "r_h1_ng_h2", "r_h2_gn_h1",
            "r_h2_pg_he3", "r_he3_gp_h2", "rprot_to_neut", "rneut_to_prot",
            "rpp_to_he3", "r_he3_he3_to_h1_h1_he4", "r_he3_ag_be7",
            "r_be7_wk_li7", "r_be7_pg_b8", "r_c12_pg_n13", "r_n14_pg_o15",
            "r_o16_pg_f17", "r_n15_pg_o16", "r_n15_pa_c12", "r_n14_ag_f18",
            "r_fe54_ng_fe55", "r_fe55_gn_fe54", "r_fe55_ng_fe56",
            "r_fe56_gn_fe55", "r_fe54_ap_co57", "r_co57_pa_fe54",
            "r_fe56_pg_co57", "r_co57_gp_fe56",
        ];
        if self.add_co56_to_approx21 {
            rxns.push("rni56ec_to_co56");
            rxns.push("rco56ec_to_fe56");
        } else {
            rxns.push("rni56ec_to_fe56");
        }
        for rname in rxns {
            self.add_reaction_by_name(rname)?;
        }
        Ok(())
    }
}

struct SpecialCaseReaction {
    reactants: &'static [&'static str],
    r1: &'static str,
    r2: &'static str,
}

const fn special(
    reactants: &'static [&'static str],
    r1: &'static str,
    r2: &'static str,
) -> SpecialCaseReaction {
    SpecialCaseReaction { reactants, r1, r2 }
}

/// Upstream `net/private/net_initialize.f90:2155-2295`
/// (`init_special_case_reaction_info`, all 51 multi-body / non-pattern rows).
const SPECIAL_CASE_REACTIONS: &[SpecialCaseReaction] = &[
    special(&["h1", "h2"], "r_h1_h1_wk_h2", "r_h1_h1_ec_h2"),
    special(
        &["neut", "h1", "h2"],
        "r_neut_h1_h1_to_h1_h2",
        "r_h1_h2_to_neut_h1_h1",
    ),
    special(&["h1", "h2", "h3"], "r_h2_h2_to_h1_h3", "r_h1_h3_to_h2_h2"),
    special(&["h3", "he3"], "r_he3_ec_h3", ""),
    special(
        &["neut", "h2", "he3"],
        "r_h2_h2_to_neut_he3",
        "r_neut_he3_to_h2_h2",
    ),
    special(&["h1", "he3", "he4"], "r_h1_he3_wk_he4", ""),
    special(&["h2", "he4"], "r_h2_h2_to_he4", "r_he4_to_h2_h2"),
    special(
        &["neut", "h2", "h3", "he4"],
        "r_h2_h3_to_neut_he4",
        "r_neut_he4_to_h2_h3",
    ),
    special(
        &["h1", "h2", "he3", "he4"],
        "r_h2_he3_to_h1_he4",
        "r_h1_he4_to_h2_he3",
    ),
    special(
        &["h2", "h3", "he3", "he4"],
        "r_h3_he3_to_h2_he4",
        "r_h2_he4_to_h3_he3",
    ),
    special(
        &["neut", "h3", "he4"],
        "r_h3_h3_to_neut_neut_he4",
        "r_neut_neut_he4_to_h3_h3",
    ),
    special(
        &["h1", "he3", "he4"],
        "r_he3_he3_to_h1_h1_he4",
        "r_h1_h1_he4_to_he3_he3",
    ),
    special(
        &["neut", "h1", "he4", "li6"],
        "r_li6_to_neut_h1_he4",
        "r_neut_h1_he4_to_li6",
    ),
    special(
        &["neut", "h2", "he4", "li7"],
        "r_h2_li7_to_neut_he4_he4",
        "r_neut_he4_he4_to_h2_li7",
    ),
    special(
        &["neut", "h3", "he4", "li7"],
        "r_h3_li7_to_neut_neut_he4_he4",
        "r_neut_neut_he4_he4_to_h3_li7",
    ),
    special(
        &["h1", "h2", "li6", "li7"],
        "r_h1_li7_to_h2_li6",
        "r_h2_li6_to_h1_li7",
    ),
    special(&["be7", "li7"], "r_be7_wk_li7", ""),
    special(
        &["h1", "h2", "he4", "be7"],
        "r_h2_be7_to_h1_he4_he4",
        "r_h1_he4_he4_to_h2_be7",
    ),
    special(
        &["neut", "he4", "be7"],
        "r_neut_be7_to_he4_he4",
        "r_he4_he4_to_neut_be7",
    ),
    special(
        &["h1", "he3", "he4", "be7"],
        "r_he3_be7_to_h1_h1_he4_he4",
        "r_h1_h1_he4_he4_to_he3_be7",
    ),
    special(
        &["neut", "h2", "li6", "be7"],
        "r_neut_be7_to_h2_li6",
        "r_h2_li6_to_neut_be7",
    ),
    special(
        &["neut", "h3", "li7", "be9"],
        "r_neut_be9_to_h3_li7",
        "r_h3_li7_to_neut_be9",
    ),
    special(
        &["neut", "he4", "be9"],
        "r_be9_to_neut_he4_he4",
        "r_neut_he4_he4_to_be9",
    ),
    special(
        &["h1", "h2", "he4", "be9"],
        "r_h1_be9_to_h2_he4_he4",
        "r_h2_he4_he4_to_h1_be9",
    ),
    special(
        &["neut", "h1", "he4", "b8"],
        "r_neut_b8_to_h1_he4_he4",
        "r_h1_he4_he4_to_neut_b8",
    ),
    special(
        &["h1", "he4", "b11"],
        "r_h1_b11_to_he4_he4_he4",
        "r_he4_he4_he4_to_h1_b11",
    ),
    special(
        &["neut", "h3", "be9", "b11"],
        "r_neut_b11_to_h3_be9",
        "r_h3_be9_to_neut_b11",
    ),
    special(&["h1", "he4", "c9"], "r_c9_wk_h1_he4_he4", ""),
    special(
        &["neut", "he4", "c11"],
        "r_neut_c11_to_he4_he4_he4",
        "r_he4_he4_he4_to_neut_c11",
    ),
    special(
        &["he4", "c12"],
        "r_he4_he4_he4_to_c12",
        "r_c12_to_he4_he4_he4",
    ),
    special(
        &["neut", "h2", "c13", "n14"],
        "r_neut_n14_to_h2_c13",
        "r_h2_c13_to_neut_n14",
    ),
    special(
        &["neut", "h2", "c14", "n15"],
        "r_neut_n15_to_h2_c14",
        "r_h2_c14_to_neut_n15",
    ),
    special(
        &["h1", "he4", "o13", "o15"],
        "r_he4_o13_to_h1_h1_o15",
        "r_h1_h1_o15_to_he4_o13",
    ),
    special(
        &["he4", "c12", "ne20"],
        "r_c12_c12_to_he4_ne20",
        "r_he4_ne20_to_c12_c12",
    ),
    special(
        &["h1", "c12", "na23"],
        "r_c12_c12_to_h1_na23",
        "r_h1_na23_to_c12_c12",
    ),
    special(
        &["neut", "c12", "mg23"],
        "r_neut_mg23_to_c12_c12",
        "r_c12_c12_to_neut_mg23",
    ),
    special(
        &["he4", "c12", "o16", "mg24"],
        "r_c12_o16_to_he4_mg24",
        "r_he4_mg24_to_c12_o16",
    ),
    special(
        &["h1", "c12", "o16", "al27"],
        "r_c12_o16_to_h1_al27",
        "r_h1_al27_to_c12_o16",
    ),
    special(
        &["neut", "c12", "o16", "si27"],
        "r_neut_si27_to_c12_o16",
        "r_c12_o16_to_neut_si27",
    ),
    special(
        &["he4", "o16", "si28"],
        "r_o16_o16_to_he4_si28",
        "r_he4_si28_to_o16_o16",
    ),
    special(
        &["he4", "c12", "ne20", "si28"],
        "r_c12_ne20_to_he4_si28",
        "r_he4_si28_to_c12_ne20",
    ),
    special(
        &["h1", "o16", "p31"],
        "r_o16_o16_to_h1_p31",
        "r_h1_p31_to_o16_o16",
    ),
    special(
        &["h1", "c12", "ne20", "p31"],
        "r_c12_ne20_to_h1_p31",
        "r_h1_p31_to_c12_ne20",
    ),
    special(&["h1", "al25", "s27"], "r_s27_wk_h1_h1_al25", ""),
    special(
        &["neut", "o16", "s31"],
        "r_o16_o16_to_neut_s31",
        "r_neut_s31_to_o16_o16",
    ),
    special(
        &["neut", "c12", "ne20", "s31"],
        "r_c12_ne20_to_neut_s31",
        "r_neut_s31_to_c12_ne20",
    ),
    special(&["h1", "p29", "ar31"], "r_ar31_wk_h1_h1_p29", ""),
    special(
        &["h1", "he4", "ca36", "ca38"],
        "r_he4_ca36_to_h1_h1_ca38",
        "r_h1_h1_ca38_to_he4_ca36",
    ),
    special(
        &["neut", "h1", "h3", "he3", "he4"],
        "r_h3_he3_to_neut_h1_he4",
        "",
    ),
    special(
        &["neut", "h1", "he3", "he4", "li7"],
        "r_he3_li7_to_neut_h1_he4_he4",
        "",
    ),
    special(
        &["neut", "h1", "h3", "he4", "be7"],
        "r_h3_be7_to_neut_h1_he4_he4",
        "",
    ),
];

#[cfg(test)]
mod tokenizer_666_tests {
    use super::*;

    /// The `.net` tokenizer before #666, kept only to check that the
    /// bundled nets tokenize the same way under `token`'s rules.
    fn tokenize_net_before_666(src: &str) -> Vec<NetToken> {
        let mut tokens = Vec::new();
        for raw_line in src.lines() {
            let line = raw_line.split('!').next().unwrap_or("");
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let b = bytes[i];
                if b.is_ascii_whitespace() {
                    i += 1;
                } else if b == b'(' {
                    tokens.push(NetToken::LParen);
                    i += 1;
                } else if b == b')' {
                    tokens.push(NetToken::RParen);
                    i += 1;
                } else if b == b',' {
                    tokens.push(NetToken::Comma);
                    i += 1;
                } else if b == b'\'' || b == b'"' {
                    let quote = b;
                    i += 1;
                    let start = i;
                    while i < bytes.len() && bytes[i] != quote {
                        i += 1;
                    }
                    let s =
                        String::from_utf8_lossy(&bytes[start..i]).into_owned();
                    tokens.push(NetToken::Quoted(s));
                    if i < bytes.len() {
                        i += 1;
                    }
                } else {
                    let start = i;
                    while i < bytes.len()
                        && !bytes[i].is_ascii_whitespace()
                        && !matches!(
                            bytes[i],
                            b'(' | b')' | b',' | b'\'' | b'"'
                        )
                    {
                        i += 1;
                    }
                    let s =
                        String::from_utf8_lossy(&bytes[start..i]).into_owned();
                    tokens.push(NetToken::Name(s));
                }
            }
        }
        tokens
    }

    /// Every bundled `.net` file gives the same token stream as before,
    /// except `mesa_206.net`, whose `2<TAB>` names now keep the tab as
    /// upstream `token` does; the list-directed read still gives 2 (#666).
    #[test]
    fn bundled_nets_tokenize_as_before_except_mesa_206_tabs() {
        let mut differing = Vec::new();
        for (name, src) in builtin_net_files() {
            // The bundle also carries the nets directory's README.rst,
            // which is documentation, not a net.
            if name == "README.rst" {
                continue;
            }
            if tokenize_net(src) != tokenize_net_before_666(src) {
                differing.push(name.clone());
            }
        }
        assert_eq!(differing, vec![String::from("mesa_206.net")]);
        assert_eq!(list_directed_usize("2\t"), Some(2));
        assert!(ParsedNetDef::parse_from_name("mesa_206.net").is_ok());
    }

    /// `token` keeps `!`, `&`, quotes and tabs after a name's first byte,
    /// skips `&` and tabs at a token start, and starts a comment with `!`
    /// only at a token start (`utils/public/utils_lib.f90:589-665`).
    #[test]
    fn net_tokens_follow_upstream_token() {
        assert_eq!(
            tokenize_net("add_isos(h1!x, &c12, he4'q 'a!b' ) o16\r! end\n"),
            vec![
                NetToken::Name(String::from("add_isos")),
                NetToken::LParen,
                NetToken::Name(String::from("h1!x")),
                NetToken::Comma,
                NetToken::Name(String::from("c12")),
                NetToken::Comma,
                NetToken::Name(String::from("he4'q")),
                NetToken::Quoted(String::from("a!b")),
                NetToken::RParen,
                // A lone CR ends the record, so `! end` starts a new one.
                NetToken::Name(String::from("o16")),
            ]
        );
        // MESA SDK gfortran 15.2.0 reads `ab\r\n`, `cd \r\n`, `xy\rzz\n` as
        // the records `ab`, `cd`, `xy`, `zz`.
        assert_eq!(
            gfortran_records("ab\r\ncd \r\nxy\rzz\n"),
            vec!["ab", "cd ", "xy", "zz"]
        );
        let long = "x".repeat(300);
        assert_eq!(
            tokenize_net(&long),
            vec![NetToken::Name("x".repeat(NET_TOKEN_LEN))]
        );
    }
}
