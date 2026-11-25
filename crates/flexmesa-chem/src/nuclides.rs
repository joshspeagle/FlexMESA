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
use flexmesa_math::{pow_strict, str_to_double, str_to_vector};

use crate::elements::{MAX_EL_Z, NPART};
use crate::ChemError;

const NUCLIDES_FULL_BIN: &[u8] = include_bytes!("../data/nuclides_full.bin");

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

/// Parse a finite Fortran/IEEE-754 float from `isotopes.data` (`str_to_double`).
fn parse_iso_double(
    tok: &str,
    field: &str,
    idx: usize,
) -> Result<f64, ChemError> {
    str_to_double(tok)
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| ChemError::TableParse {
            path: "isotopes.data".to_string(),
            detail: format!("invalid {field} at nuclide {}", idx + 1),
        })
}

/// Parse an optional `-N` isomeric state (`0..=99`), failing loudly on invalid
/// suffixes (`chem/private/chem_isos_io.f90:152-169`).
fn parse_isomeric_state(name: &str, idx: usize) -> Result<u8, ChemError> {
    let Some((_, suffix)) = name.split_once('-') else {
        return Ok(0);
    };
    suffix
        .parse::<i32>()
        .ok()
        .and_then(|v| u8::try_from(v).ok())
        .filter(|&v| v <= 99)
        .ok_or_else(|| ChemError::TableParse {
            path: "isotopes.data".to_string(),
            detail: format!(
                "invalid isomeric state in '{name}' at nuclide {}",
                idx + 1
            ),
        })
}

/// Compute per-element `(min_N, max_N)` bounds for $Z \in [0, 112]$
/// (`chem/private/chem_isos_io.f90:141-148`).
fn compute_element_n_bounds(
    entries: &[NuclideEntry],
) -> [Option<(usize, usize)>; MAX_EL_Z + 1] {
    let mut bounds: [Option<(usize, usize)>; MAX_EL_Z + 1] =
        [None; MAX_EL_Z + 1];
    for entry in entries {
        if entry.z <= MAX_EL_Z {
            bounds[entry.z] = Some(match bounds[entry.z] {
                None => (entry.n, entry.n),
                Some((mn, mx)) => (mn.min(entry.n), mx.max(entry.n)),
            });
        }
    }
    bounds
}

struct RawNuclide {
    cid: usize,
    name: String,
    z: usize,
    n: usize,
    spin: f64,
    mex: f64,
    isom: u8,
    pfcn: [f64; NPART],
}

/// Parse one 4-line record from `isotopes.data` (`parse_line`,
/// `get_isomeric_state`, and 3×8 `pfcn` grid in `chem_isos_io.f90:82-209`).
fn parse_isotopes_record(
    chunk: &[&str; 4],
    idx: usize,
) -> Result<RawNuclide, ChemError> {
    let hdr: Vec<&str> = chunk[0].split_whitespace().collect();
    let [name_tok, w_tok, z_s, n_s, sp_s, mx_s] = hdr.as_slice() else {
        return Err(ChemError::TableParse {
            path: "isotopes.data".to_string(),
            detail: format!(
                "expected 6 columns at nuclide {}, found {}",
                idx + 1,
                hdr.len()
            ),
        });
    };
    let name = (*name_tok).to_string();
    let _w_tab = parse_iso_double(w_tok, "W", idx)?;
    let z: usize = parse_iso_token(z_s, "Z", idx)?;
    let n: usize = parse_iso_token(n_s, "N", idx)?;
    let spin = parse_iso_double(sp_s, "spin", idx)?;
    let mex = parse_iso_double(mx_s, "mass_excess", idx)?;
    let isom = parse_isomeric_state(&name, idx)?;
    let mut pfcn = [1.0_f64; NPART];
    for k in 0..3 {
        let vec =
            str_to_vector(chunk[1 + k]).map_err(|_| ChemError::TableParse {
                path: "isotopes.data".to_string(),
                detail: format!(
                    "invalid pfcn float at nuclide {} row {}",
                    idx + 1,
                    k + 1
                ),
            })?;
        if vec.len() != 8 || !vec.iter().all(|v| v.is_finite()) {
            return Err(ChemError::TableParse {
                path: "isotopes.data".to_string(),
                detail: format!(
                    "expected 8 finite pfcn values at nuclide {} row {}, \
                     found {}",
                    idx + 1,
                    k + 1,
                    vec.len()
                ),
            });
        }
        pfcn[8 * k..8 * (k + 1)].copy_from_slice(&vec);
    }
    Ok(RawNuclide {
        cid: idx + 1,
        name,
        z,
        n,
        spin,
        mex,
        isom,
        pfcn,
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
        // Load-time MESA `chem/private/chem_isos_io.f90:133` uses strict
        // IEEE-754 `pow`, invariant across `NumericsMode::Strict` and `Fast`.
        let z53 = pow_strict(z_f64, FIVE_THIRDS);
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
    element_n_bounds: [Option<(usize, usize)>; MAX_EL_Z + 1],
}

impl Default for NuclideCatalog {
    fn default() -> Self {
        Self::builtin()
    }
}

impl NuclideCatalog {
    /// Returns a static reference to the built-in 7,856-nuclide catalog.
    #[must_use]
    pub fn builtin_ref() -> &'static Self {
        static CATALOG: OnceLock<NuclideCatalog> = OnceLock::new();
        CATALOG.get_or_init(Self::build_builtin)
    }

    /// Build the default reference catalog containing all 7,856 built-in nuclides
    /// from MESA `isotopes.data`, including 24-point partition functions `pfcn`.
    #[must_use]
    pub fn builtin() -> Self {
        Self::builtin_ref().clone()
    }

    /// Return a slice of all [`NuclideEntry`] records in 1-based `chem_id` order.
    #[must_use]
    pub fn entries(&self) -> &[NuclideEntry] {
        &self.entries
    }

    #[allow(clippy::cast_sign_loss)]
    fn build_builtin() -> Self {
        const REC_SIZE: usize = 228;
        assert!(
            &NUCLIDES_FULL_BIN[0..8] == b"FMCHM001",
            "invalid nuclides_full.bin magic header"
        );
        let total =
            u32::from_le_bytes(NUCLIDES_FULL_BIN[8..12].try_into().unwrap())
                as usize;

        // First pass: resolve del_mp / del_mn across all records before
        // computing binding energies (chem/private/chem_isos_io.f90:122-136).
        let mut del_mp = DEL_MP;
        let mut del_mn = DEL_MN;
        for pos in 0..total {
            let off = 12 + pos * REC_SIZE;
            let rec = &NUCLIDES_FULL_BIN[off..off + REC_SIZE];
            let z =
                i16::from_le_bytes(rec[12..14].try_into().unwrap()) as usize;
            let n =
                i16::from_le_bytes(rec[14..16].try_into().unwrap()) as usize;
            if (z == 1 && n == 0) || (z == 0 && n == 1) {
                let mex = f64::from_le_bytes(rec[28..36].try_into().unwrap());
                if z == 1 && n == 0 {
                    del_mp = mex;
                }
                if z == 0 && n == 1 {
                    del_mn = mex;
                }
            }
        }

        let mut entries = Vec::with_capacity(total);
        let mut by_id = HashMap::with_capacity(total);
        let mut by_name = HashMap::with_capacity(total);
        for pos in 0..total {
            let off = 12 + pos * REC_SIZE;
            let rec = &NUCLIDES_FULL_BIN[off..off + REC_SIZE];
            let cid =
                u32::from_le_bytes(rec[0..4].try_into().unwrap()) as usize;
            let name_bytes = &rec[4..12];
            let name_end = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            let name = std::str::from_utf8(&name_bytes[..name_end])
                .expect("valid ascii nuclide name")
                .to_string();
            let z =
                i16::from_le_bytes(rec[12..14].try_into().unwrap()) as usize;
            let n =
                i16::from_le_bytes(rec[14..16].try_into().unwrap()) as usize;
            let isom = rec[16];
            let spin = f64::from_le_bytes(rec[20..28].try_into().unwrap());
            let mex = f64::from_le_bytes(rec[28..36].try_into().unwrap());
            let mut pfcn = [1.0_f64; NPART];
            for (k, p_val) in pfcn.iter_mut().enumerate().take(NPART) {
                let p_off = 36 + k * 8;
                *p_val = f64::from_le_bytes(
                    rec[p_off..p_off + 8].try_into().unwrap(),
                );
            }
            let entry = NuclideEntry::from_raw(
                cid,
                name.clone(),
                z,
                n,
                spin,
                mex,
                isom,
                pfcn,
                del_mp,
                del_mn,
            );
            by_id.insert(cid, pos);
            by_name.insert(name.to_ascii_lowercase(), pos);
            entries.push(entry);
        }
        let element_n_bounds = compute_element_n_bounds(&entries);
        Self {
            entries,
            by_id,
            by_name,
            del_mp,
            del_mn,
            element_n_bounds,
        }
    }

    /// Parse a full MESA `isotopes.data` text buffer (4 lines per nuclide after
    /// the leading blank/header line), matching `do_read_chem_isos` in
    /// `chem/private/chem_isos_io.f90:53-209`.
    ///
    /// # Errors
    /// Returns [`ChemError::TableParse`] if the text is empty, truncated, or
    /// contains malformed nuclide headers, isomeric tags, or partition grids.
    pub fn from_isotopes_str(content: &str) -> Result<Self, ChemError> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(ChemError::TableParse {
                path: "isotopes.data".to_string(),
                detail: "empty isotopes.data".to_string(),
            });
        }
        let mut body_end = lines.len();
        while body_end > 1 && lines[body_end - 1].trim().is_empty() {
            body_end -= 1;
        }
        let body = &lines[1..body_end];
        if body.is_empty() {
            return Err(ChemError::TableParse {
                path: "isotopes.data".to_string(),
                detail: "no isotopes found in isotopes.data".to_string(),
            });
        }
        if !body.len().is_multiple_of(4) {
            return Err(ChemError::TableParse {
                path: "isotopes.data".to_string(),
                detail: format!(
                    "incomplete nuclide record: {} body lines not divisible \
                     by 4",
                    body.len()
                ),
            });
        }
        let num_isos = body.len() / 4;
        let mut raw_list: Vec<RawNuclide> = Vec::with_capacity(num_isos);
        let mut del_mp = DEL_MP;
        let mut del_mn = DEL_MN;

        for (idx, chunk) in body.as_chunks::<4>().0.iter().enumerate() {
            let raw = parse_isotopes_record(chunk, idx)?;
            if raw.z == 1 && raw.n == 0 {
                del_mp = raw.mex;
            }
            if raw.z == 0 && raw.n == 1 {
                del_mn = raw.mex;
            }
            raw_list.push(raw);
        }

        let mut entries = Vec::with_capacity(raw_list.len());
        let mut by_id = HashMap::with_capacity(raw_list.len());
        let mut by_name = HashMap::with_capacity(raw_list.len());
        for (pos, raw) in raw_list.into_iter().enumerate() {
            by_id.insert(raw.cid, pos);
            by_name.insert(raw.name.to_ascii_lowercase(), pos);
            entries.push(NuclideEntry::from_raw(
                raw.cid, raw.name, raw.z, raw.n, raw.spin, raw.mex, raw.isom,
                raw.pfcn, del_mp, del_mn,
            ));
        }
        let element_n_bounds = compute_element_n_bounds(&entries);
        Ok(Self {
            entries,
            by_id,
            by_name,
            del_mp,
            del_mn,
            element_n_bounds,
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

    /// Return the resolved proton (`Z = 1, N = 0`) atomic mass excess
    /// $\Delta M_p$ in $\mathrm{MeV}$ for this catalog.
    #[must_use]
    pub fn del_mp(&self) -> f64 {
        self.del_mp
    }

    /// Return the resolved neutron (`Z = 0, N = 1`) atomic mass excess
    /// $\Delta M_n$ in $\mathrm{MeV}$ for this catalog.
    #[must_use]
    pub fn del_mn(&self) -> f64 {
        self.del_mn
    }

    /// Look up the ground-state (`isomeric_state = 0`) 1-based `chem_id` by
    /// `(Z, N)` (`lookup_ZN` in `chem/public/chem_lib.f90:474-477`).
    #[must_use]
    pub fn lookup_zn(&self, z: usize, n: usize) -> Option<usize> {
        self.lookup_zn_isomeric_state(z, n, 0)
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

    /// Return the minimum neutron number $N$ in the catalog for `z`, or
    /// `None` if no isotopes exist for `z` (`element_min_N` in
    /// `chem/public/chem_def.f90:73` and `chem/private/chem_isos_io.f90:141-148`).
    #[must_use]
    pub fn element_min_n(&self, z: usize) -> Option<usize> {
        if z <= MAX_EL_Z {
            self.element_n_bounds[z].map(|(mn, _)| mn)
        } else {
            self.entries.iter().filter(|e| e.z == z).map(|e| e.n).min()
        }
    }

    /// Return the maximum neutron number $N$ in the catalog for `z`, or
    /// `None` if no isotopes exist for `z` (`element_max_N` in
    /// `chem/public/chem_def.f90:73` and `chem/private/chem_isos_io.f90:141-148`).
    #[must_use]
    pub fn element_max_n(&self, z: usize) -> Option<usize> {
        if z <= MAX_EL_Z {
            self.element_n_bounds[z].map(|(_, mx)| mx)
        } else {
            self.entries.iter().filter(|e| e.z == z).map(|e| e.n).max()
        }
    }

    /// Compute total molar nuclear binding energy $\sum_i B_i Y_i$ (in
    /// $\mathrm{MeV}$) for molar abundances `y` aligned with [`Self::entries`]
    /// (`binding_energy` in `chem/public/chem_lib.f90:516-522`).
    ///
    /// # Errors
    /// Returns [`ChemError::InvalidInput`] if `y.len() != self.len()`.
    pub fn binding_energy(&self, y: &[f64]) -> Result<f64, ChemError> {
        if y.len() != self.entries.len() {
            return Err(ChemError::InvalidInput {
                context: "NuclideCatalog::binding_energy",
                detail: format!(
                    "y length ({}) != catalog size ({})",
                    y.len(),
                    self.entries.len()
                ),
            });
        }
        let mut total = 0.0_f64;
        for (entry, &yi) in self.entries.iter().zip(y.iter()) {
            total += entry.binding_energy * yi;
        }
        Ok(total)
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

/// Entry in a sorted nuclide subset lookup table (`type nuclide_set` in
/// `chem/public/chem_def.f90:406-409`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NuclideSetEntry {
    /// Trimmed nuclide name (`adjustl` ordering semantics).
    pub nuclide: String,
    /// 1-based original rank in the input slice.
    pub rank: usize,
}

/// Construct and sort a nuclide subset lookup table preserving 1-based input
/// ranks (`generate_nuclide_set` in `chem/public/chem_lib.f90:120-128` and
/// `sort_nuclide_set` in `chem/private/nuclide_set_mod.f90:63-110`).
#[must_use]
pub fn generate_nuclide_set(names: &[&str]) -> Vec<NuclideSetEntry> {
    let mut set: Vec<NuclideSetEntry> = names
        .iter()
        .enumerate()
        .map(|(i, &name)| NuclideSetEntry {
            nuclide: name.trim().to_string(),
            rank: i + 1,
        })
        .collect();
    set.sort_by(|a, b| a.nuclide.cmp(&b.nuclide));
    set
}

/// Binary-search a sorted [`NuclideSetEntry`] slice for `nuclei` and return its
/// 1-based rank, or `None` if absent (`get_nuclide_index_in_set` in
/// `chem/public/chem_lib.f90:108-117` and `nuclide_set_mod.f90:29-60`).
#[must_use]
pub fn get_nuclide_index_in_set(
    nuclei: &str,
    set: &[NuclideSetEntry],
) -> Option<usize> {
    if set.is_empty() {
        return None;
    }
    let key = nuclei.trim();
    if key < set[0].nuclide.as_str()
        || key > set[set.len() - 1].nuclide.as_str()
    {
        return None;
    }
    set.binary_search_by(|entry| entry.nuclide.as_str().cmp(key))
        .ok()
        .map(|idx| set[idx].rank)
}

/// Expose the built-in [`NuclideEntry`] rows from [`NuclideCatalog::builtin_ref`].
///
/// Returns a static slice of all 7,856 built-in nuclides compiled into
/// `flexmesa-chem` from MESA `isotopes.data`.
#[must_use]
pub fn builtin_nuclide_rows() -> &'static [NuclideEntry] {
    NuclideCatalog::builtin_ref().entries()
}
