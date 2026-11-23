// SPDX-License-Identifier: LGPL-3.0-or-later
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

fn tokenize_net(src: &str) -> Vec<NetToken> {
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
                let s = String::from_utf8_lossy(&bytes[start..i]).into_owned();
                tokens.push(NetToken::Quoted(s));
                if i < bytes.len() {
                    i += 1;
                }
            } else {
                let start = i;
                while i < bytes.len()
                    && !bytes[i].is_ascii_whitespace()
                    && !matches!(bytes[i], b'(' | b')' | b',' | b'\'' | b'"')
                {
                    i += 1;
                }
                let s = String::from_utf8_lossy(&bytes[start..i]).into_owned();
                tokens.push(NetToken::Name(s));
            }
        }
    }
    tokens
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
        self.read_net_file_with_depth(filename, search_dir, 0)
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
                owned_src = Some(fs::read_to_string(candidate).map_err(|e| {
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
                    .or_else(|| name.parse::<usize>().ok())
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
                let a1: usize = a1_s
                    .parse()
                    .map_err(|_| NetError::ParseError(a1_s.clone()))?;
                let a2: usize = a2_s
                    .parse()
                    .map_err(|_| NetError::ParseError(a2_s.clone()))?;
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
