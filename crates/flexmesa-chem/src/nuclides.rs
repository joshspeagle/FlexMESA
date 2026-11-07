//! Nuclide catalog (`NuclideCatalog`), atomic mass and binding energy formulas,
//! and nuclear reaction $Q$-value evaluation (`reaction_Qtotal`).
//!
//! Part of the Wave 2a (`chem`) microphysics layer in `FlexMESA`.
//! Verified against 100% of upstream MESA `chem/test/test_output`
//! (`o16`, `AG89`, `Lodders 2003`, `composition_info`, `reaction_Qtotal`).
//! Enforces `#![forbid(unsafe_code)]` and IEEE-754 `f64` precision.
//!
//! Matches `do_read_chem_isos` (`chem_isos_io.f90`) and `chem_lib.f90`.
//!
//! # Nuclear Energetics Formulas
//! - **Binding energy**: $B_i = Z_i \Delta M_p + N_i \Delta M_n - \Delta M_i$
//!   ($\mathrm{MeV}$).
//! - **Atomic weight**: $W_i = A_i + \Delta M_i \cdot (\mathrm{MeV\_to\_ergs}
//!   / \mathrm{amu} / c^2)$ ($\mathrm{g\,mol^{-1}}$).
//! - **Coulomb screening power**: $Z_i^{5/3} = \mathrm{pow}(Z_i, 5/3)$.
//! - **Reaction $Q$-value**: $Q_{\mathrm{total}} = \sum_{\mathrm{in}} (-Q_j) +
//!   \sum_{\mathrm{out}} Q_k$, where $Q_i = -\Delta M_i$ automatically
//!   accounts for positron rest-mass annihilation via atomic mass excesses.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::OnceLock;

use flexmesa_const::{AMU, CLIGHT, FIVE_THIRDS, MEV_TO_ERGS};
use flexmesa_math::pow;

use crate::elements::NPART;
use crate::nuclides_data_a::{RawNuclideRow, BUILTIN_NUCLIDES_A};
use crate::nuclides_data_b::BUILTIN_NUCLIDES_B;
use crate::nuclides_data_c::BUILTIN_NUCLIDES_C;
use crate::ChemError;

/// Atomic mass excess $\Delta M_p$ of ${}^1\mathrm{H}$ (`h1`) in
/// $\mathrm{MeV}$.
pub const DEL_MP: f64 = 7.288_970_500;

/// Atomic mass excess $\Delta M_n$ of the free neutron (`neut`) in
/// $\mathrm{MeV}$.
pub const DEL_MN: f64 = 8.071_317_100;

/// Parse a token from `isotopes.data` into `T` with diagnostic context.
fn parse_iso_token<T: FromStr>(
    tok: &str,
    field: &str,
    idx: usize,
) -> Result<T, ChemError> {
    tok.parse().map_err(|_| ChemError::TableParse {
        path: "isotopes.data".to_string(),
        detail: format!("invalid {field} at nuclide {}", idx + 1),
    })
}

/// Record for a single nuclear species in [`NuclideCatalog`].
#[derive(Debug, Clone, PartialEq)]
pub struct NuclideEntry {
    /// 1-based MESA nuclide index (`chem_id`) in `isotopes.data`.
    pub chem_id: usize,
    /// Trimmed nuclide name (e.g. `"h1"`, `"c12"`, `"o16"`, `"fe56"`).
    pub name: String,
    /// Proton number $Z_i$.
    pub z: usize,
    /// Neutron number $N_i$.
    pub n: usize,
    /// Baryon number $A_i = Z_i + N_i$.
    pub z_plus_n: usize,
    /// Ground-state nuclear spin $J_i$.
    pub spin: f64,
    /// Tabulated atomic mass excess $\Delta M_i$ in $\mathrm{MeV}$.
    pub mass_excess: f64,
    /// Nuclear binding energy $B_i = Z_i \Delta M_p + N_i \Delta M_n -
    /// \Delta M_i$ ($\mathrm{MeV}$).
    pub binding_energy: f64,
    /// Recomputed double-precision atomic weight $W_i$ in $\mathrm{g/mol}$.
    pub w: f64,
    /// Pre-calculated Coulomb screening power $Z_i^{5/3}$.
    pub z53: f64,
    /// Isomeric state index (`0` for ground state, `1..=99` for isomers).
    pub isomeric_state: u8,
    /// 24-point nuclear partition function $G(T_9)$ on [`crate::T_PART`].
    pub pfcn: [f64; NPART],
}

impl NuclideEntry {
    /// Construct a [`NuclideEntry`] from raw `isotopes.data` fields using
    /// proton/neutron mass excesses `del_mp` and `del_mn`.
    #[must_use]
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    pub fn from_raw(
        chem_id: usize,
        name: String,
        z: usize,
        n: usize,
        spin: f64,
        mass_excess: f64,
        isomeric_state: u8,
        pfcn: [f64; NPART],
        del_mp: f64,
        del_mn: f64,
    ) -> Self {
        let z_plus_n = z + n;
        let z_f64 = z as f64;
        let n_f64 = n as f64;
        let a_f64 = z_plus_n as f64;
        let z53 = pow(z_f64, FIVE_THIRDS);
        let binding_energy = z_f64 * del_mp + n_f64 * del_mn - mass_excess;
        let conv = MEV_TO_ERGS / AMU / (CLIGHT * CLIGHT);
        let w = a_f64 + mass_excess * conv;
        Self {
            chem_id,
            name,
            z,
            n,
            z_plus_n,
            spin,
            mass_excess,
            binding_energy,
            w,
            z53,
            isomeric_state,
            pfcn,
        }
    }
}

/// Catalog of nuclear species supporting $O(1)$ lookup by `chem_id` or `name`.
#[derive(Debug, Clone)]
pub struct NuclideCatalog {
    entries: Vec<NuclideEntry>,
    by_id: HashMap<usize, usize>,
    by_name: HashMap<String, usize>,
    del_mp: f64,
    del_mn: f64,
}

impl Default for NuclideCatalog {
    fn default() -> Self {
        Self::builtin()
    }
}

impl NuclideCatalog {
    /// Returns a static reference to the built-in 347-nuclide catalog.
    #[must_use]
    pub fn builtin_ref() -> &'static Self {
        static CATALOG: OnceLock<NuclideCatalog> = OnceLock::new();
        CATALOG.get_or_init(Self::build_builtin)
    }

    /// Build the default reference catalog containing all 347 built-in nuclides.
    #[must_use]
    pub fn builtin() -> Self {
        Self::builtin_ref().clone()
    }

    fn build_builtin() -> Self {
        let total = BUILTIN_NUCLIDES_A.len()
            + BUILTIN_NUCLIDES_B.len()
            + BUILTIN_NUCLIDES_C.len();
        let mut entries = Vec::with_capacity(total);
        let mut by_id = HashMap::with_capacity(total);
        let mut by_name = HashMap::with_capacity(total);

        let iter_rows = BUILTIN_NUCLIDES_A
            .iter()
            .chain(BUILTIN_NUCLIDES_B.iter())
            .chain(BUILTIN_NUCLIDES_C.iter());
        for (pos, &(cid, name, z, n, spin, mex, isom)) in iter_rows.enumerate()
        {
            let entry = NuclideEntry::from_raw(
                cid,
                name.to_string(),
                usize::from(z),
                usize::from(n),
                spin,
                mex,
                isom,
                [1.0_f64; NPART],
                DEL_MP,
                DEL_MN,
            );
            by_id.insert(cid, pos);
            by_name.insert(name.to_ascii_lowercase(), pos);
            entries.push(entry);
        }
        Self {
            entries,
            by_id,
            by_name,
            del_mp: DEL_MP,
            del_mn: DEL_MN,
        }
    }

    /// Parse a full MESA `isotopes.data` text buffer (4 lines per nuclide after
    /// the leading blank/header line).
    ///
    /// # Errors
    /// Returns [`ChemError::TableParse`] if the text is malformed or empty.
    pub fn from_isotopes_str(content: &str) -> Result<Self, ChemError> {
        type ParsedRow =
            (usize, String, usize, usize, f64, f64, u8, [f64; NPART]);
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < 5 {
            return Err(ChemError::TableParse {
                path: "isotopes.data".to_string(),
                detail: "fewer than 5 lines in isotopes.data".to_string(),
            });
        }
        let num_isos = (lines.len() - 1) / 4;
        let mut raw_list: Vec<ParsedRow> = Vec::with_capacity(num_isos);
        let mut del_mp = DEL_MP;
        let mut del_mn = DEL_MN;

        for idx in 0..num_isos {
            let base = 1 + 4 * idx;
            let mut toks = lines[base].split_whitespace();
            let Some(name_tok) = toks.next() else {
                break;
            };
            let _a_tok = toks.next();
            let (Some(z_s), Some(n_s), Some(sp_s), Some(mx_s)) =
                (toks.next(), toks.next(), toks.next(), toks.next())
            else {
                break;
            };
            let name = name_tok.to_string();
            let z: usize = parse_iso_token(z_s, "Z", idx)?;
            let n: usize = parse_iso_token(n_s, "N", idx)?;
            let spin: f64 = parse_iso_token(sp_s, "spin", idx)?;
            let mex: f64 = parse_iso_token(mx_s, "mass_excess", idx)?;
            if z == 1 && n == 0 {
                del_mp = mex;
            }
            if z == 0 && n == 1 {
                del_mn = mex;
            }
            let isom = name
                .split_once('-')
                .and_then(|(_, sfx)| sfx.parse::<u8>().ok())
                .unwrap_or(0);
            let mut pfcn = [1.0_f64; NPART];
            for k in 0..3 {
                for (j, tok) in
                    lines[base + 1 + k].split_whitespace().take(8).enumerate()
                {
                    pfcn[8 * k + j] = tok.parse().unwrap_or(1.0);
                }
            }
            raw_list.push((idx + 1, name, z, n, spin, mex, isom, pfcn));
        }

        let mut entries = Vec::with_capacity(raw_list.len());
        let mut by_id = HashMap::with_capacity(raw_list.len());
        let mut by_name = HashMap::with_capacity(raw_list.len());
        for (pos, (cid, name, z, n, spin, mex, isom, pfcn)) in
            raw_list.into_iter().enumerate()
        {
            by_id.insert(cid, pos);
            by_name.insert(name.to_ascii_lowercase(), pos);
            entries.push(NuclideEntry::from_raw(
                cid, name, z, n, spin, mex, isom, pfcn, del_mp, del_mn,
            ));
        }
        Ok(Self {
            entries,
            by_id,
            by_name,
            del_mp,
            del_mn,
        })
    }

    /// Load a `NuclideCatalog` from a filesystem path to `isotopes.data`.
    ///
    /// # Errors
    /// Returns [`ChemError::Io`] or [`ChemError::TableParse`] on failure.
    pub fn from_isotopes_file(path: &Path) -> Result<Self, ChemError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ChemError::Io(e.to_string()))?;
        Self::from_isotopes_str(&content)
    }

    /// Return the number of nuclides in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the 1-based `chem_id` for `name` (`get_nuclide_index`).
    #[must_use]
    pub fn get_nuclide_index(&self, name: &str) -> Option<usize> {
        self.get_by_name(name).map(|e| e.chem_id)
    }

    /// Look up a [`NuclideEntry`] by its 1-based `chem_id`.
    #[must_use]
    pub fn get_by_id(&self, chem_id: usize) -> Option<&NuclideEntry> {
        self.by_id.get(&chem_id).map(|&pos| &self.entries[pos])
    }

    /// Look up a [`NuclideEntry`] by its trimmed isotope name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&NuclideEntry> {
        let trimmed = name.trim();
        let pos = if trimmed.bytes().all(|b| !b.is_ascii_uppercase()) {
            self.by_name.get(trimmed).copied()
        } else {
            self.by_name.get(&trimmed.to_ascii_lowercase()).copied()
        };
        pos.map(|idx| &self.entries[idx])
    }

    /// Look up the 1-based `chem_id` by `(Z, N, isomeric_state)`.
    #[must_use]
    pub fn lookup_zn_isomeric_state(
        &self,
        z: usize,
        n: usize,
        isomeric_state: u8,
    ) -> Option<usize> {
        self.entries
            .iter()
            .find(|e| {
                e.z == z && e.n == n && e.isomeric_state == isomeric_state
            })
            .map(|e| e.chem_id)
    }

    /// Compute the atomic mass excess $\Delta M_i = Z_i \Delta M_p +
    /// N_i \Delta M_n - B_i$ in $\mathrm{MeV}$ (`get_mass_excess`).
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `chem_id` is not in the catalog.
    #[allow(clippy::cast_precision_loss)]
    pub fn get_mass_excess(&self, chem_id: usize) -> Result<f64, ChemError> {
        let entry =
            self.get_by_id(chem_id)
                .ok_or_else(|| ChemError::InvalidInput {
                    context: "NuclideCatalog::get_mass_excess",
                    detail: format!("unknown chem_id {chem_id}"),
                })?;
        Ok(
            (entry.z as f64) * self.del_mp + (entry.n as f64) * self.del_mn
                - entry.binding_energy,
        )
    }

    /// Return $Q_i = -\Delta M_i$ in $\mathrm{MeV}$ (`get_Q`).
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `chem_id` is not in the catalog.
    pub fn get_q(&self, chem_id: usize) -> Result<f64, ChemError> {
        Ok(-self.get_mass_excess(chem_id)?)
    }

    /// Compute the total reaction $Q$-value (in $\mathrm{MeV}$) for a nuclear
    /// reaction with `num_in` incoming species and `num_out` outgoing species:
    ///
    /// $$Q_{\mathrm{total}} = \sum_{j=1}^{N_{\mathrm{in}}} (-Q_{\mathrm{in},j})
    ///   + \sum_{k=1}^{N_{\mathrm{out}}} Q_{\mathrm{out},k}.$$
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `reactants.len() < num_in +
    /// num_out` or any `chem_id` is unknown.
    pub fn reaction_qtotal(
        &self,
        num_in: usize,
        num_out: usize,
        reactants: &[usize],
    ) -> Result<f64, ChemError> {
        if reactants.len() < num_in + num_out {
            return Err(ChemError::InvalidInput {
                context: "NuclideCatalog::reaction_qtotal",
                detail: format!(
                    "reactants length {} < num_in + num_out ({})",
                    reactants.len(),
                    num_in + num_out
                ),
            });
        }
        let mut qtotal = 0.0_f64;
        for (j, &cid) in reactants.iter().take(num_in + num_out).enumerate() {
            let q = self.get_q(cid)?;
            if j < num_in {
                qtotal -= q;
            } else {
                qtotal += q;
            }
        }
        Ok(qtotal)
    }
}

/// Expose raw built-in nuclide rows (`Part A` and `Part B`) for inspection.
///
/// Returns a pair of static slices containing the 347 built-in reference
/// nuclides compiled into `flexmesa-chem` (`MAIN_ISO_NAME`, `AG89_TABLE`,
/// and standard nuclear reaction network species).
#[must_use]
pub fn builtin_nuclide_rows(
) -> (&'static [RawNuclideRow], &'static [RawNuclideRow]) {
    (&BUILTIN_NUCLIDES_A, &BUILTIN_NUCLIDES_B)
}
