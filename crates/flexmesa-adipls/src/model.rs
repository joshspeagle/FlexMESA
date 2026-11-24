// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust ports of MESA 26.4.1 `fgong_amdl` and `store_amdl`
//! (`astero/private/adipls_support.f90:688-908`), with `#479` explicit unset
//! tracking for uninitialized `aa` rows (`DECISIONS #43`).

use flexmesa_const::{FOUR_THIRDS_PI, PI4};
use flexmesa_math::{exp, pow2, pow3};

use crate::error::AdiplsError;

/// Number of model variables passed to ADIPLS (`ivarmd` in
/// `adipls_support.f90:36`).
pub const IVARMD: usize = 6;

/// First dimension of ADIPLS `aa(10, nn)` (`iaa_arg` in
/// `adipls_support.f90:36`).
pub const IAA_ARG: usize = 10;

/// Minimum `glob` length required by `fgong_amdl` (`glob(1..=12)`).
const MIN_FGONG_GLOB: usize = 12;

/// Minimum `var` variable count required by `fgong_amdl` (`var(1..=15, :)`).
const MIN_FGONG_IVAR: usize = 15;

/// Minimum `var` variable count required when `fgong_amdl` inserts a center
/// point (`ireset` includes index 20 in `adipls_support.f90:815-816`).
const PREPEND_FGONG_IVAR: usize = 20;

/// 1-based `var` variable indices copied from zone 2 to the inserted center
/// point when `var(1, 1) > 1.d6` (`adipls_support.f90:815-816`).
const IRESET_1BASED: [usize; 16] =
    [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20];

/// FGONG pulse-data model produced by `star_get_pulse_data` and consumed by
/// [`fgong_amdl`].
///
/// The `var` array mirrors Fortran column-major storage `(ivar, nn)`: zone
/// `zone` (0-based, `0..nn`) and 1-based variable `var_1based` (`1..=ivar`) sits
/// at `var[zone * ivar + (var_1based - 1)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FgongModel {
    /// Global stellar scalars (`glob(1..=iconst)`, CGS units).
    ///
    /// `glob[0]` is total mass $M$ ($\mathrm{g}$), `glob[1]` is photospheric
    /// radius $R$ ($\mathrm{cm}$), and `glob[10..=11]` hold central second
    /// derivatives when negative and above `-10000`.
    pub glob: Vec<f64>,
    /// Number of variables per mesh point (`ivar`).
    pub ivar: usize,
    /// Number of mesh points (`nn`).
    pub nn: usize,
    /// Column-major flattened zone variables of length `ivar * nn`.
    pub var: Vec<f64>,
}

impl FgongModel {
    /// Construct an [`FgongModel`] and validate its dimensions.
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidModel`] when `nn == 0`, `ivar < 15`,
    /// `glob.len() < 12`, or `var.len() != ivar * nn`.
    pub fn new(
        glob: Vec<f64>,
        ivar: usize,
        nn: usize,
        var: Vec<f64>,
    ) -> Result<Self, AdiplsError> {
        if nn == 0 {
            return Err(AdiplsError::InvalidModel {
                message: String::from("FGONG model must have nn >= 1"),
            });
        }
        if glob.len() < MIN_FGONG_GLOB {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "FGONG glob length {} is smaller than required {MIN_FGONG_GLOB}",
                    glob.len()
                ),
            });
        }
        if ivar < MIN_FGONG_IVAR {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "FGONG ivar {ivar} is smaller than required {MIN_FGONG_IVAR}"
                ),
            });
        }
        let expected =
            ivar.checked_mul(nn)
                .ok_or_else(|| AdiplsError::InvalidModel {
                    message: String::from("FGONG var size overflow"),
                })?;
        if var.len() != expected {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "FGONG var length {} does not match ivar * nn = {expected}",
                    var.len()
                ),
            });
        }
        Ok(Self {
            glob,
            ivar,
            nn,
            var,
        })
    }

    /// Return the 1-based variable `row_1based` at 0-based zone `zone`.
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidModel`] if either index is out of range.
    pub fn var_at(
        &self,
        row_1based: usize,
        zone: usize,
    ) -> Result<f64, AdiplsError> {
        if row_1based == 0 || row_1based > self.ivar || zone >= self.nn {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "FGONG var index ({row_1based}, {zone}) out of bounds (ivar={}, nn={})",
                    self.ivar, self.nn
                ),
            });
        }
        Ok(self.var[zone * self.ivar + (row_1based - 1)])
    }
}

/// Ten-row AMDL structure matrix `aa(1:10, 1:nn)` with explicit tracking of
/// populated vs unset rows (`#479`, `DECISIONS #43`).
///
/// Each 1-based row `1..=10` is either initialized over `nn` zones or left
/// explicitly unset (`None`). Reading any unset row returns
/// [`AdiplsError::UnsetAmdlRow`] instead of copying uninitialized memory.
#[derive(Debug, Clone, PartialEq)]
pub struct AmdlAa {
    nn: usize,
    rows: [Option<Vec<f64>>; IAA_ARG],
}

impl AmdlAa {
    /// Create an `AmdlAa` with `nn` zones and all 10 rows initially unset.
    #[must_use]
    pub fn new_unset(nn: usize) -> Self {
        Self {
            nn,
            rows: std::array::from_fn(|_| None),
        }
    }

    /// Number of mesh points `nn` per row.
    #[must_use]
    pub const fn nn(&self) -> usize {
        self.nn
    }

    /// Return whether 1-based row `row_1based` (`1..=10`) is populated.
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidAmdlIndex`] if `row_1based` is outside
    /// `1..=10`.
    pub fn is_row_set(&self, row_1based: usize) -> Result<bool, AdiplsError> {
        let idx = checked_row_index(row_1based, self.nn)?;
        Ok(self.rows[idx].is_some())
    }

    /// Return an immutable slice for 1-based row `row_1based` (`1..=10`).
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidAmdlIndex`] if `row_1based` is outside
    /// `1..=10`, or [`AdiplsError::UnsetAmdlRow`] if that row is unset
    /// (`#479`).
    pub fn row(&self, row_1based: usize) -> Result<&[f64], AdiplsError> {
        let idx = checked_row_index(row_1based, self.nn)?;
        self.rows[idx]
            .as_deref()
            .ok_or(AdiplsError::UnsetAmdlRow { row: row_1based })
    }

    /// Read a single value at 1-based `row_1based` (`1..=10`) and 0-based
    /// `zone` (`0..nn`).
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidAmdlIndex`] on out-of-range coordinates or
    /// [`AdiplsError::UnsetAmdlRow`] when reading an unset row (`#479`).
    pub fn get(
        &self,
        row_1based: usize,
        zone: usize,
    ) -> Result<f64, AdiplsError> {
        if zone >= self.nn {
            return Err(AdiplsError::InvalidAmdlIndex {
                row: row_1based,
                zone,
                nn: self.nn,
            });
        }
        let slice = self.row(row_1based)?;
        Ok(slice[zone])
    }

    /// Populate 1-based row `row_1based` (`1..=10`) with a vector of length
    /// `self.nn()`.
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidAmdlIndex`] if `row_1based` is outside
    /// `1..=10` or [`AdiplsError::InvalidModel`] if `values.len() != self.nn()`.
    pub fn set_row(
        &mut self,
        row_1based: usize,
        values: Vec<f64>,
    ) -> Result<(), AdiplsError> {
        let idx = checked_row_index(row_1based, self.nn)?;
        if values.len() != self.nn {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "AMDL row {row_1based} length {} does not match nn = {}",
                    values.len(),
                    self.nn
                ),
            });
        }
        self.rows[idx] = Some(values);
        Ok(())
    }

    /// Truncate every populated row to `new_nn` points (`new_nn <= self.nn()`).
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidModel`] if `new_nn > self.nn()`.
    pub fn truncate(&mut self, new_nn: usize) -> Result<(), AdiplsError> {
        if new_nn > self.nn {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "cannot truncate AMDL aa from nn={} to larger new_nn={new_nn}",
                    self.nn
                ),
            });
        }
        for row in self.rows.iter_mut().flatten() {
            row.truncate(new_nn);
        }
        self.nn = new_nn;
        Ok(())
    }

    /// Return a cloned sub-matrix starting at 0-based zone `start_zone`.
    ///
    /// Preserves unset rows identically (`#479`).
    ///
    /// # Errors
    /// Returns [`AdiplsError::InvalidModel`] if `start_zone >= self.nn`.
    pub fn slice_from(&self, start_zone: usize) -> Result<Self, AdiplsError> {
        if start_zone >= self.nn {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "slice_from start_zone={start_zone} >= nn={}",
                    self.nn
                ),
            });
        }
        let new_nn = self.nn - start_zone;
        let mut out = Self::new_unset(new_nn);
        for (idx, row_opt) in self.rows.iter().enumerate() {
            if let Some(vals) = row_opt {
                out.rows[idx] = Some(vals[start_zone..].to_vec());
            }
        }
        Ok(out)
    }
}

fn checked_row_index(
    row_1based: usize,
    nn: usize,
) -> Result<usize, AdiplsError> {
    if (1..=IAA_ARG).contains(&row_1based) {
        Ok(row_1based - 1)
    } else {
        Err(AdiplsError::InvalidAmdlIndex {
            row: row_1based,
            zone: 0,
            nn,
        })
    }
}

/// Output of [`fgong_amdl`] (`astero/private/adipls_support.f90:805-908`).
///
/// Populates `data(1:8)` and `aa(1:6, 1:nn)`; `aa` rows `7..=10` (1-based) are
/// left explicitly unset.
#[derive(Debug, Clone, PartialEq)]
pub struct FgongAmdlOutput {
    /// Number of AMDL mesh points (`nn`), ordered center-to-surface.
    pub nn: usize,
    /// AMDL global header scalars `data(1:8)` (`[M, R, P_c, rho_c, ... , -1, 0]`).
    pub data: [f64; 8],
    /// Ten-row `aa` matrix with 1-based rows `1..=6` populated (`rows 7..=10` unset).
    pub aa: AmdlAa,
}

/// Output of [`store_amdl`] (`astero/private/adipls_support.f90:688-802`).
///
/// Populates `x(1:nn)`, `data(1:8)`, and `aa` rows `1..=5` and `10` (1-based).
/// Per `#479` (`DECISIONS #43`), rows `6..=9` (1-based) are left explicitly
/// unset because upstream `store_amdl` copied uninitialized `aa1(7, :)` into
/// `aa(6, :)` and never initialized rows `7..=9`.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreAmdlOutput {
    /// Number of stored AMDL mesh points (`nn`).
    pub nn: usize,
    /// Potentially updated turbulent-pressure flag `iturpr` (`8` when `iggt == 1`).
    pub iturpr: i32,
    /// AMDL global header scalars `data(1:8)`.
    pub data: [f64; 8],
    /// Dimensionless fractional radius grid `x(1:nn) = r / R` (`aa1(1, :)`).
    pub x: Vec<f64>,
    /// Ten-row `aa` matrix with 1-based rows `1..=5` and `10` populated
    /// (`rows 6..=9` explicitly unset per `#479`).
    pub aa: AmdlAa,
}

/// Convert FGONG pulse data into intermediate AMDL arrays matching MESA 26.4.1
/// `fgong_amdl` (`astero/private/adipls_support.f90:805-908`).
///
/// Preserves upstream operation order and uses [`flexmesa_math::exp`]
/// (`crlibm::exp_rd`) matching `math_lib::exp` imported via `astero_def` in
/// `adipls_support.f90:23`.
///
/// # Errors
/// Returns [`AdiplsError::InvalidModel`] when `model` fails dimension checks,
/// when prepending a center point requires `ivar < 20` (`ireset` reads up to
/// index 20), or when trimming the outer point empties the model.
pub fn fgong_amdl(
    cgrav: f64,
    model: &FgongModel,
) -> Result<FgongAmdlOutput, AdiplsError> {
    let ivar = model.ivar;
    let nn = model.nn;
    let glob = &model.glob;
    if nn == 0 || glob.len() < MIN_FGONG_GLOB || ivar < MIN_FGONG_IVAR {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "invalid FGONG dimensions for fgong_amdl: nn={nn}, iconst={}, ivar={ivar}",
                glob.len()
            ),
        });
    }
    if model.var.len() != ivar * nn {
        return Err(AdiplsError::InvalidModel {
            message: String::from("FGONG var length mismatch"),
        });
    }

    let (wvar, mut nn) = prepare_fgong_work_var(model)?;

    // `adipls_support.f90:859-865`: q = exp(var(2,:)), x = var(1,:)/glob(2),
    // then force x(1) = 0, q(1) = 0.
    let mut q = Vec::with_capacity(nn);
    let mut x = Vec::with_capacity(nn);
    for n in 0..nn {
        let base = n * ivar;
        q.push(exp(wvar[base + 1]));
        x.push(wvar[base] / glob[1]);
    }
    x[0] = 0.0_f64;
    q[0] = 0.0_f64;

    let mut rows = populate_fgong_aa_rows(cgrav, glob, &wvar, ivar, &x, &q, nn);

    // `adipls_support.f90:884-887`: chop outermost point when U = aa(5, nn) <= 10.
    if rows[4][nn - 1] <= 10.0_f64 {
        nn -= 1;
        if nn == 0 {
            return Err(AdiplsError::InvalidModel {
                message: String::from(
                    "fgong_amdl outer-point trim left zero mesh points",
                ),
            });
        }
        for row in &mut rows {
            row.truncate(nn);
        }
    }

    let data = populate_fgong_data(cgrav, glob, &wvar, &x, &rows[4], nn);
    let mut aa = AmdlAa::new_unset(nn);
    for (idx, row_vals) in rows.into_iter().enumerate() {
        aa.set_row(idx + 1, row_vals)?;
    }

    Ok(FgongAmdlOutput { nn, data, aa })
}

fn prepare_fgong_work_var(
    model: &FgongModel,
) -> Result<(Vec<f64>, usize), AdiplsError> {
    let ivar = model.ivar;
    let mut nn = model.nn;

    // `adipls_support.f90:823-833`: reverse surface-to-center into center-to-surface.
    let mut wvar = vec![0.0_f64; ivar * nn];
    if model.var[0] > model.var[(nn - 1) * ivar] {
        for n in 0..nn {
            let src = (nn - 1 - n) * ivar;
            let dst = n * ivar;
            wvar[dst..dst + ivar].copy_from_slice(&model.var[src..src + ivar]);
        }
    } else {
        wvar.copy_from_slice(&model.var);
    }

    // `adipls_support.f90:835-857`: prepend center point if innermost r > 1e6 cm.
    if wvar[0] > 1.0e6_f64 {
        if ivar < PREPEND_FGONG_IVAR {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "fgong_amdl center-point insertion requires ivar >= {PREPEND_FGONG_IVAR}, got {ivar}"
                ),
            });
        }
        let mut shifted = vec![0.0_f64; ivar * (nn + 1)];
        shifted[ivar..].copy_from_slice(&wvar);
        for &r1 in &IRESET_1BASED {
            let idx = r1 - 1;
            shifted[idx] = shifted[ivar + idx];
        }
        wvar = shifted;
        nn += 1;
    }

    Ok((wvar, nn))
}

fn populate_fgong_aa_rows(
    cgrav: f64,
    glob: &[f64],
    wvar: &[f64],
    ivar: usize,
    x: &[f64],
    q: &[f64],
    nn: usize,
) -> [Vec<f64>; IVARMD] {
    // `adipls_support.f90:867-883`: populate aa(1:6, 1:nn); rows 7..=10 remain unset.
    let mut row1 = vec![0.0_f64; nn];
    let mut row2 = vec![0.0_f64; nn];
    let mut row3 = vec![0.0_f64; nn];
    let mut row4 = vec![0.0_f64; nn];
    let mut row5 = vec![0.0_f64; nn];
    let mut row6 = vec![0.0_f64; nn];

    for n in 1..nn {
        let base = n * ivar;
        let v1 = wvar[base];
        let v4 = wvar[base + 3];
        let v5 = wvar[base + 4];
        let v10 = wvar[base + 9];
        let v15 = wvar[base + 14];
        row1[n] = x[n];
        row2[n] = q[n] / pow3(x[n]);
        // Left-to-right numerator `cgrav * glob(1) * q(n) * var(5,n)` divided by
        // parenthesized `(var(10,n) * var(4,n) * var(1,n))`.
        row3[n] = (((cgrav * glob[0]) * q[n]) * v5) / ((v10 * v4) * v1);
        row4[n] = v10;
        row5[n] = v15;
        row6[n] = ((PI4 * v5) * pow3(v1)) / (glob[0] * q[n]);
    }

    row1[0] = 0.0_f64;
    row2[0] = ((FOUR_THIRDS_PI * wvar[4]) * pow3(glob[1])) / glob[0];
    row3[0] = 0.0_f64;
    row4[0] = wvar[9];
    row5[0] = 0.0_f64;
    row6[0] = 3.0_f64;

    [row1, row2, row3, row4, row5, row6]
}

fn populate_fgong_data(
    cgrav: f64,
    glob: &[f64],
    wvar: &[f64],
    x: &[f64],
    row5: &[f64],
    nn: usize,
) -> [f64; 8] {
    // `adipls_support.f90:888-906`: populate data(1:8).
    let mut data = [0.0_f64; 8];
    data[0] = glob[0];
    data[1] = glob[1];
    data[2] = wvar[3];
    data[3] = wvar[4];
    if glob[10] < 0.0_f64 && glob[10] > -10000.0_f64 {
        data[4] = -glob[10] / wvar[9];
        data[5] = -glob[11];
    } else {
        data[4] = ((FOUR_THIRDS_PI * cgrav) * pow2(wvar[4] * glob[1]))
            / (wvar[3] * wvar[9]);
        let mut d2amax = 0.0_f64;
        for n in 1..nn {
            d2amax = d2amax.max(row5[n] / pow2(x[n]));
            if x[n] >= 0.05_f64 {
                break;
            }
        }
        data[5] = d2amax + data[4];
    }
    data[6] = -1.0_f64;
    data[7] = 0.0_f64;
    data
}

/// Shift intermediate AMDL arrays into ADIPLS layout matching MESA 26.4.1
/// `store_amdl` (`astero/private/adipls_support.f90:688-802`), honoring `#479`
/// (`DECISIONS #43`).
///
/// Copies only initialized 1-based rows `1..=6` from `aa_in` (`ivarmd = 6`),
/// sets `x` from `aa1(1, :)`, sets `aa` rows `1..=5` from `aa1(2..=6, :)`,
/// leaves `aa` rows `6..=9` explicitly unset (never copying uninitialized
/// `aa1(7, :)` into `aa(6, :)`), and sets row `10` (`g / g_tilde`).
///
/// # Errors
/// Returns:
/// - [`AdiplsError::InvalidModel`] when `nn_in == 0`, when `aa_in.nn() != nn_in`,
///   or when `iriche == 1` and `(nn_in - nsin)` is even (where upstream's loop
///   bound `nnr = nn_in` reads `aa1(:, nn_in + 1)` out of bounds,
///   `adipls_support.f90:736-771`, `DECISIONS #43`).
/// - [`AdiplsError::UnsetAmdlRow`] when any required input row `1..=6` is unset,
///   or when `iggt == 1` (`mod(int(data(8) + 0.1) / 10, 10) == 2`) attempts to
///   read unset row `6` (`#479`).
pub fn store_amdl(
    nn_in: usize,
    iriche: i32,
    iturpr: i32,
    data: [f64; 8],
    aa_in: &AmdlAa,
) -> Result<StoreAmdlOutput, AdiplsError> {
    if nn_in == 0 || aa_in.nn() != nn_in {
        return Err(AdiplsError::InvalidModel {
            message: format!(
                "store_amdl nn_in={nn_in} mismatch with aa.nn()={}",
                aa_in.nn()
            ),
        });
    }

    // Read only rows 1..=6 (`ivarmd = 6`), honoring `#479`.
    let r1_in = aa_in.row(1)?;
    let r2_in = aa_in.row(2)?;
    let r3_in = aa_in.row(3)?;
    let r4_in = aa_in.row(4)?;
    let r5_in = aa_in.row(5)?;
    let r6_in = aa_in.row(6)?;
    let aa1: [&[f64]; IVARMD] = [r1_in, r2_in, r3_in, r4_in, r5_in, r6_in];

    // `adipls_support.f90:716-720`
    let sincen = aa1[0][0] == 0.0_f64;
    let sinsur = data[6] >= 0.0_f64;
    let nsin = usize::from(sincen) + usize::from(sinsur);

    // `adipls_support.f90:724-730` (`0.1` is default-kind single precision).
    let d8_shifted = data[7] + f64::from(0.1_f32);
    if !d8_shifted.is_finite()
        || d8_shifted < f64::from(i32::MIN)
        || d8_shifted > f64::from(i32::MAX)
    {
        return Err(AdiplsError::ConversionOverflow {
            field: "data(8)",
            value: data[7],
        });
    }
    #[allow(clippy::cast_possible_truncation)]
    let idata8 = d8_shifted as i32;
    let (iggt, iturpr_out) = if (idata8 / 10) % 10 == 2 {
        (true, 8_i32)
    } else {
        (false, iturpr)
    };

    // `adipls_support.f90:736-771`: if nshift == 1, upstream loops `n = 2..nnr`
    // with `n1 = n + 1` up to `nn_in + 1` past allocated `aa1` and `x(nn)`.
    if nn_in < nsin {
        return Err(AdiplsError::InvalidModel {
            message: format!("store_amdl nn_in={nn_in} < nsin={nsin}"),
        });
    }
    let nshift = usize::from(iriche == 1 && (nn_in - nsin).is_multiple_of(2));
    if nshift != 0 {
        return Err(AdiplsError::InvalidModel {
            message: String::from(
                "store_amdl with iriche=1 and even (nn - nsin) reads out of bounds at aa1(:, nn_in + 1) (DECISIONS #43)",
            ),
        });
    }

    let nn = nn_in;
    let x = aa1[0].to_vec();

    // `adipls_support.f90:750-771`: shift `aa(i, n) = aa1(i + 1, n)` for
    // `i = 1..=5` (`rows 6..=9` remain explicitly unset per `#479`).
    let mut aa_out = AmdlAa::new_unset(nn);
    aa_out.set_row(1, aa1[1].to_vec())?;
    aa_out.set_row(2, aa1[2].to_vec())?;
    aa_out.set_row(3, aa1[3].to_vec())?;
    aa_out.set_row(4, aa1[4].to_vec())?;
    aa_out.set_row(5, aa1[5].to_vec())?;

    // `adipls_support.f90:778-795`: set row 10 (`g / g_tilde`).
    let mut row10 = Vec::with_capacity(nn);
    if iturpr_out == 1 {
        let out_r1 = aa_out.row(1)?;
        for n in 0..nn {
            let ggt = if x[n] < 0.999_f64 {
                1.0_f64
            } else {
                1.0_f64 / (x[n] * x[n] * x[n] * out_r1[n])
            };
            row10.push(ggt);
        }
    } else if iggt {
        // Explicit read of row 6 triggers `AdiplsError::UnsetAmdlRow { row: 6 }` (#479).
        let out_r6 = aa_out.row(6)?;
        row10.extend_from_slice(out_r6);
    } else {
        row10.resize(nn, 1.0_f64);
    }
    aa_out.set_row(10, row10)?;

    Ok(StoreAmdlOutput {
        nn,
        iturpr: iturpr_out,
        data,
        x,
        aa: aa_out,
    })
}
