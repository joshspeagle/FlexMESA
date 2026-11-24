// SPDX-License-Identifier: GPL-3.0-or-later
//! ADIPLS mode output structs (`ObsSt`, `Csummm`, `ModeRecord`) and
//! `spcout_adi` (`astero/private/adipls_support_procs.f90:23-65`,
//! `adipls/adipack.c/adipls/setobs_st.c.d.f:124-148`,
//! `adipls/adipack.c/adipls/adipls.c.d.f:40-44, 101-102`).

use flexmesa_const::PI4;

use crate::error::AdiplsError;

/// Ten-element mode observation vector `obs_st(1:10)` populated by `setobs_st`
/// (`adipls/adipack.c/adipls/setobs_st.c.d.f:124-148`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ObsSt(pub [f64; 10]);

impl ObsSt {
    /// Construct from raw `obs_st(1:10)` values.
    #[must_use]
    pub const fn from_array(raw: [f64; 10]) -> Self {
        Self(raw)
    }

    /// Borrow the underlying `[f64; 10]` array.
    #[must_use]
    pub const fn as_array(&self) -> &[f64; 10] {
        &self.0
    }

    /// Raw degree `obs_st(1)` (`csummm(18)`).
    #[must_use]
    pub const fn degree_raw(&self) -> f64 {
        self.0[0]
    }

    /// Raw radial order `obs_st(2)` (`csummm(19)`).
    #[must_use]
    pub const fn order_raw(&self) -> f64 {
        self.0[1]
    }

    /// Cyclic mode frequency in $\mu\mathrm{Hz}$ (`obs_st(3)`).
    #[must_use]
    pub const fn cyclic_freq_uhz(&self) -> f64 {
        self.0[2]
    }

    /// Dimensionless kinetic energy / raw inertia $E_{\mathrm{kin}}$ (`obs_st(4)`),
    /// prior to multiplication by $4\pi$ in [`spcout_adi`].
    #[must_use]
    pub const fn ekin(&self) -> f64 {
        self.0[3]
    }

    /// First-order rotational splitting Ledoux parameter $\beta$ (`obs_st(5)`).
    #[must_use]
    pub const fn rot_beta(&self) -> f64 {
        self.0[4]
    }

    /// First-order rotational splitting coefficient `split(1)` (`obs_st(6)`).
    #[must_use]
    pub const fn rot_split1(&self) -> f64 {
        self.0[5]
    }

    /// Second-order rotational splitting $m^0$ coefficient `split(2)`
    /// (`obs_st(7)`).
    #[must_use]
    pub const fn rot_split2(&self) -> f64 {
        self.0[6]
    }

    /// Second-order rotational splitting $m^2$ coefficient `split(3)`
    /// (`obs_st(8)`).
    #[must_use]
    pub const fn rot_split3(&self) -> f64 {
        self.0[7]
    }
}

/// Fifty-word ADIPLS grand summary buffer `csummm(1:50)`
/// (`adipls/adipack.c/adipls/adipls.c.d.f:40-44, 101-102`).
///
/// Stored as raw IEEE-754 64-bit words (`[u64; 50]`) so `EQUIVALENCE`d integer
/// words `39..=42` (`icsumm(1:8)`) and model name words `43..=44` (`mlname(1:4)`)
/// retain exact bit patterns without floating-point canonicalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Csummm {
    words: [u64; 50],
}

impl Default for Csummm {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Csummm {
    /// Return a zero-initialized `csummm(1:50)` buffer (`call zero(csummm, 50)`).
    #[must_use]
    pub const fn zeroed() -> Self {
        Self { words: [0_u64; 50] }
    }

    /// Construct from 50 raw IEEE-754 bit patterns.
    #[must_use]
    pub const fn from_bits(words: [u64; 50]) -> Self {
        Self { words }
    }

    /// Construct from 50 `f64` values preserving exact bit representations.
    #[must_use]
    pub fn from_f64_array(vals: [f64; 50]) -> Self {
        let mut words = [0_u64; 50];
        for (dst, src) in words.iter_mut().zip(vals) {
            *dst = src.to_bits();
        }
        Self { words }
    }

    /// Borrow the 50 raw 64-bit words.
    #[must_use]
    pub const fn as_bits(&self) -> &[u64; 50] {
        &self.words
    }

    /// Read a 1-based scalar `csummm(index)` (`1..=50`) as `f64`.
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidCsummmIndex`] when `index` is outside
    /// `1..=50`.
    pub fn get_f64(&self, index_1based: usize) -> Result<f64, AdiplsError> {
        let idx = checked_csum_index(index_1based)?;
        Ok(f64::from_bits(self.words[idx]))
    }

    /// Write a 1-based scalar `csummm(index)` (`1..=50`) from `f64`.
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidCsummmIndex`] when `index` is outside
    /// `1..=50`.
    pub fn set_f64(
        &mut self,
        index_1based: usize,
        value: f64,
    ) -> Result<(), AdiplsError> {
        let idx = checked_csum_index(index_1based)?;
        self.words[idx] = value.to_bits();
        Ok(())
    }

    /// Azimuthal order `csummm(38)` stored in `dosdum(5)` (`sigout.c.d.f:245`).
    #[must_use]
    pub const fn em_raw(&self) -> f64 {
        f64::from_bits(self.words[37])
    }

    /// Unpack the 8 equivalenced `i32` entries `icsumm(1:8)` from words
    /// `39..=42` (`[in, nnw, mdints, ivarfs, icase, iorign, iekins, idum8]`)
    /// in little-endian layout.
    #[must_use]
    pub const fn icsumm(&self) -> [i32; 8] {
        let mut out = [0_i32; 8];
        let mut k = 0;
        while k < 4 {
            let w = self.words[38 + k];
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap
            )]
            {
                out[2 * k] = (w & 0xffff_ffff) as u32 as i32;
                out[2 * k + 1] = ((w >> 32) & 0xffff_ffff) as u32 as i32;
            }
            k += 1;
        }
        out
    }

    /// Pack 8 `i32` entries `icsumm(1:8)` into words `39..=42` in little-endian
    /// layout (`[in, nnw, mdints, ivarfs, icase, iorign, iekins, idum8]`).
    pub fn set_icsumm(&mut self, ics: [i32; 8]) {
        let mut k = 0;
        while k < 4 {
            #[allow(clippy::cast_sign_loss)]
            let lo = u64::from(ics[2 * k] as u32);
            #[allow(clippy::cast_sign_loss)]
            let hi = u64::from(ics[2 * k + 1] as u32);
            self.words[38 + k] = lo | (hi << 32);
            k += 1;
        }
    }
}

const fn checked_csum_index(index_1based: usize) -> Result<usize, AdiplsError> {
    if index_1based >= 1 && index_1based <= 50 {
        Ok(index_1based - 1)
    } else {
        Err(AdiplsError::InvalidCsummmIndex {
            index: index_1based,
        })
    }
}

/// Mode record stored by [`spcout_adi`] (`adipls_support_procs.f90:50-58`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeRecord {
    /// Spherical harmonic degree $l$ (`int(obs_st(1) + 0.5_dp)`).
    pub el: i32,
    /// Radial order $n_{pg}$ (`int(obs_st(2) + 0.5_dp)`).
    pub order: i32,
    /// Azimuthal order $m$ (`int(csummm(38))`).
    pub em: i32,
    /// Cyclic frequency $\nu$ in $\mu\mathrm{Hz}$ (`obs_st(3)`).
    pub cyclic_freq_uhz: f64,
    /// Mode inertia $E = 4\pi E_{\mathrm{kin}}$ (`obs_st(4) * pi4`).
    pub inertia: f64,
    /// Growth rate (`0.0` for adiabatic ADIPLS, `adipls_support_procs.f90:57`).
    pub growth_rate: f64,
    /// Full 10-element `obs_st` vector.
    pub obs_st: ObsSt,
    /// Full 50-word `csummm` grand summary.
    pub csummm: Csummm,
}

/// Collection of ADIPLS mode outputs mirroring `astero_def` oscillation
/// storage (`el`, `order`, `em`, `cyclic_freq`, `inertia`, `growth_rate`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdiModeStore {
    /// Stored modes in emission order.
    pub records: Vec<ModeRecord>,
}

impl AdiModeStore {
    /// Create an empty mode store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Number of stored modes (`num_results`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return whether no modes have been stored yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Clear stored results before a new ADIPLS call (`num_results = 0`).
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Append one mode converted from `(obs_st, csummm)` via [`spcout_adi`].
    ///
    /// # Errors
    /// Propagates [`AdiplsError::ConversionOverflow`] if integer conversion
    /// fails.
    pub fn push_from_spcout(
        &mut self,
        obs_st: &ObsSt,
        csummm: &Csummm,
    ) -> Result<&ModeRecord, AdiplsError> {
        let rec = spcout_adi(obs_st, csummm)?;
        self.records.push(rec);
        Ok(&self.records[self.records.len() - 1])
    }

    /// Sort stored modes by ascending `cyclic_freq_uhz` (`qsort` in
    /// `do_adipls_get_one_el_info:124` and `get_one_el_info:153`).
    pub fn sort_by_frequency(&mut self) {
        self.records
            .sort_by(|a, b| a.cyclic_freq_uhz.total_cmp(&b.cyclic_freq_uhz));
    }
}

/// Convert ADIPLS common-block mode quantities (`obs_st(1:10)` and
/// `csummm(1:50)`) into MESA's [`ModeRecord`] matching `spcout_adi`
/// (`astero/private/adipls_support_procs.f90:50-58`).
///
/// # Errors
/// Returns [`AdiplsError::ConversionOverflow`] if `obs_st(1) + 0.5`,
/// `obs_st(2) + 0.5`, or `csummm(38)` is non-finite or outside `i32` bounds.
pub fn spcout_adi(
    obs_st: &ObsSt,
    csummm: &Csummm,
) -> Result<ModeRecord, AdiplsError> {
    let el = f64_to_fortran_int(obs_st.0[0] + 0.5_f64, "obs_st(1) + 0.5")?;
    let order = f64_to_fortran_int(obs_st.0[1] + 0.5_f64, "obs_st(2) + 0.5")?;
    let em = f64_to_fortran_int(csummm.em_raw(), "csummm(38)")?;
    let inertia = obs_st.0[3] * PI4;
    let cyclic_freq_uhz = obs_st.0[2];

    Ok(ModeRecord {
        el,
        order,
        em,
        cyclic_freq_uhz,
        inertia,
        growth_rate: 0.0_f64,
        obs_st: *obs_st,
        csummm: *csummm,
    })
}

/// Convert `obs_st` + `csummm` and append directly to `store`, matching
/// `spcout_adi` + `store_new_oscillation_results`
/// (`adipls_support_procs.f90:50-58`).
///
/// # Errors
/// Propagates [`AdiplsError::ConversionOverflow`].
pub fn spcout_adi_store(
    obs_st: &ObsSt,
    csummm: &Csummm,
    store: &mut AdiModeStore,
) -> Result<ModeRecord, AdiplsError> {
    let rec = spcout_adi(obs_st, csummm)?;
    store.records.push(rec);
    Ok(rec)
}

fn f64_to_fortran_int(
    val: f64,
    field: &'static str,
) -> Result<i32, AdiplsError> {
    if !val.is_finite()
        || val <= f64::from(i32::MIN) - 1.0_f64
        || val >= f64::from(i32::MAX) + 1.0_f64
    {
        return Err(AdiplsError::ConversionOverflow { field, value: val });
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(val as i32)
}
