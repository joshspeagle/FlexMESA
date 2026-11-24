// SPDX-License-Identifier: GPL-3.0-or-later
//! FGONG pulsation data: `get_fgong_data` (`star/private/pulse_fgong.f90:46-630`)
//! and the `pulse_utils` routines only FGONG and OSC use (`eval_face_X`,
//! `eval_center_X`, `eval_center_d2`, `star/private/pulse_utils.f90`),
//! transcribed in upstream order of operations (#491).
//!
//! The producer reads a [`PulseDataInput`] view of the star (MESA cell or
//! face `k = 1..nz`, surface first, stored 0-based) and returns
//! `global_data(15)` and `point_data(40, nn)` from the surface inwards, as
//! [`crate::pulse_write::write_fgong_data`] writes them.

use flexmesa_const::{MixingType, FOUR_THIRDS, PI, RSUN, STANDARD_CGRAV};
use flexmesa_eos::{
    eos_dt_get, EosConfig, I_CHI_RHO, I_CHI_T, I_GAMMA1, I_LN_PGAS,
    NUM_EOS_BASIC_RESULTS,
};

use crate::model::StarModel;
use crate::pulse_gyre_data::{
    eval_center, eval_center_rho, eval_face, eval_face_a_ast, eval_face_rho,
    set_segment_indices, GyreDataError, GyreDataInput,
};
use crate::pulse_write::{FGONG_ICONST, FGONG_IVAR};

/// The star state the FGONG and OSC producers read beyond
/// [`GyreDataInput`], as MESA's `star_info` fields. Cell and face arrays
/// have length `nz`; `xa` and `dxdt_nuc` are zone-major
/// (`[(k-1)*species + i]`).
#[derive(Debug, Clone)]
pub struct PulseDataInput<'a> {
    /// The fields `get_gyre_data` also reads.
    pub base: GyreDataInput<'a>,
    /// `s%species`.
    pub species: usize,
    /// `s%xa` (zone-major).
    pub xa: &'a [f64],
    /// 0-based species index of each `net_iso` entry FGONG and OSC use,
    /// `None` where upstream's `net_iso` is 0.
    pub iso: PulseIsoIndices,
    /// `s%X` (cells).
    pub x: Vec<f64>,
    /// `s%Y` (cells).
    pub y: Vec<f64>,
    /// `s%cp` (cells).
    pub cp: &'a [f64],
    /// `s%lnfree_e` (cells).
    pub lnfree_e: &'a [f64],
    /// `s%eps_grav_ad%val` (cells).
    pub eps_grav: Vec<f64>,
    /// `s%dxdt_nuc(h1,:)` (cells), `None` when the net has no `h1`.
    pub dxdt_nuc_h1: Option<Vec<f64>>,
    /// `s%mixing_type` (faces).
    pub mixing_type: &'a [MixingType],
    /// `s%Pgas` (cells).
    pub pgas: &'a [f64],
    /// `s%gradr` (faces).
    pub gradr: &'a [f64],
    /// `s%chem_id`.
    pub chem_id: &'a [usize],
    /// `s%eos_handle` settings.
    pub eos_config: &'a EosConfig,
    /// `s%Teff` (K).
    pub teff: f64,
    /// `s%star_age` (yr).
    pub star_age: f64,
    /// `s%initial_z`.
    pub initial_z: f64,
    /// `s%initial_y`.
    pub initial_y: f64,
    /// `s%mixing_length_alpha`.
    pub mixing_length_alpha: f64,
    /// `s%fgong_zero_A_inside_r` ($R_\odot$).
    pub fgong_zero_a_inside_r: f64,
    /// `s%mixing_region_bottom(s%largest_conv_mixing_region)` (1-based), or
    /// `None` when `s%largest_conv_mixing_region == 0`.
    pub largest_conv_mixing_region_bottom: Option<usize>,
}

/// 0-based species indices of the `net_iso` entries FGONG and OSC read.
#[derive(Debug, Clone, Copy, Default)]
#[allow(missing_docs)]
pub struct PulseIsoIndices {
    pub h1: Option<usize>,
    pub h2: Option<usize>,
    pub he3: Option<usize>,
    pub he4: Option<usize>,
    pub li7: Option<usize>,
    pub be7: Option<usize>,
    pub be9: Option<usize>,
    pub c12: Option<usize>,
    pub c13: Option<usize>,
    pub n14: Option<usize>,
    pub n15: Option<usize>,
    pub o16: Option<usize>,
    pub o17: Option<usize>,
    pub o18: Option<usize>,
    pub ne20: Option<usize>,
    pub si28: Option<usize>,
}

/// 1-based view of a slice.
fn at(v: &[f64], k: usize) -> f64 {
    v[k - 1]
}

/// `MAX(v, lo)` and `MIN(v, hi)` of the optional limits (`pulse_utils.f90:180-186`).
fn limit(v: f64, lo: f64, hi: f64) -> f64 {
    let v = if v > lo { v } else { lo };
    if v < hi {
        v
    } else {
        hi
    }
}

impl PulseDataInput<'_> {
    fn xa_at(&self, i: usize, k: usize) -> f64 {
        self.xa[(k - 1) * self.species + i]
    }

    /// `eval_face_X(s, i, k, k_a, k_b)` (`pulse_utils.f90:192-245`).
    pub(crate) fn eval_face_x(
        &self,
        i: Option<usize>,
        k: usize,
        k_a: usize,
        k_b: usize,
    ) -> Result<f64, GyreDataError> {
        if k_b < k_a {
            return Err(GyreDataError::Upstream(
                "eval_face_X: invalid segment indices",
            ));
        }
        if k < k_a || k > k_b + 1 {
            return Err(GyreDataError::Upstream(
                "eval_face_X: out-of-bounds interpolation",
            ));
        }
        let Some(i) = i else {
            return Ok(0.0);
        };
        let dq = self.base.dq;
        let x = |kk: usize| self.xa_at(i, kk);
        let x_face = if k_b == k_a {
            x(k_a)
        } else if k == k_a {
            x(k_a)
                - at(dq, k_a) * (x(k_a + 1) - x(k_a))
                    / (at(dq, k_a + 1) + at(dq, k_a))
        } else if k == k_a + 1 {
            x(k_a)
                + at(dq, k_a) * (x(k_a + 1) - x(k_a))
                    / (at(dq, k_a + 1) + at(dq, k_a))
        } else if k == k_b {
            x(k_b)
                - at(dq, k_b) * (x(k_b) - x(k_b - 1))
                    / (at(dq, k_b) + at(dq, k_b - 1))
        } else if k == k_b + 1 {
            x(k_b)
                + at(dq, k_b) * (x(k_b) - x(k_b - 1))
                    / (at(dq, k_b) + at(dq, k_b - 1))
        } else {
            let v: Vec<f64> = (k_a..=k_b).map(x).collect();
            crate::pulse_gyre_data::interp_val_to_pt_slices(
                &v,
                k - k_a + 1,
                k_b - k_a + 1,
                &dq[k_a - 1..k_b],
            )?
        };
        // X_face = MIN(1d0, MAX(0d0, X_face))
        Ok(limit(x_face, 0.0, 1.0))
    }

    /// `eval_center_X(s, i, k_a, k_b)` (`pulse_utils.f90:407-460`).
    pub(crate) fn eval_center_x(
        &self,
        i: Option<usize>,
        k_a: usize,
        k_b: usize,
    ) -> Result<f64, GyreDataError> {
        let Some(i) = i else {
            return Ok(0.0);
        };
        if k_b < k_a {
            return Err(GyreDataError::Upstream(
                "eval_center: invalid segment indices",
            ));
        }
        let x_center = if k_a == k_b {
            self.xa_at(i, k_a)
        } else {
            let r_1 = at(self.base.rmid, k_b);
            let r_2 = at(self.base.rmid, k_b - 1);
            let x_1 = self.xa_at(i, k_b);
            let x_2 = self.xa_at(i, k_b - 1);
            (x_1 * r_2 * r_2 - x_2 * r_1 * r_1) / (r_2 * r_2 - r_1 * r_1)
        };
        Ok(limit(x_center, 0.0, 1.0))
    }
}

/// `eval_center_d2(r, v, k_a, k_b)` (`pulse_utils.f90:485-521`).
pub(crate) fn eval_center_d2(
    r: &[f64],
    v: &[f64],
    k_a: usize,
    k_b: usize,
) -> Result<f64, GyreDataError> {
    if k_b < k_a {
        return Err(GyreDataError::Upstream(
            "eval_center_d2: invalid segment indices",
        ));
    }
    Ok(if k_a == k_b {
        0.0
    } else {
        let r_1 = at(r, k_b);
        let r_2 = at(r, k_b - 1);
        let v_1 = at(v, k_b);
        let v_2 = at(v, k_b - 1);
        2.0 * (v_2 - v_1) / (r_2 * r_2 - r_1 * r_1)
    })
}

/// `eval_face(...)` with the optional `v_lo`, `v_hi` limits
/// (`pulse_utils.f90:136-189`).
pub(crate) fn eval_face_lim(
    dq: &[f64],
    v: &[f64],
    k: usize,
    k_a: usize,
    k_b: usize,
    lo: f64,
    hi: f64,
) -> Result<f64, GyreDataError> {
    Ok(limit(eval_face(dq, v, k, k_a, k_b)?, lo, hi))
}

/// `eval_center(...)` with the optional `v_lo`, `v_hi` limits
/// (`pulse_utils.f90:354-404`).
pub(crate) fn eval_center_lim(
    r: &[f64],
    v: &[f64],
    k_a: usize,
    k_b: usize,
    lo: f64,
    hi: f64,
) -> Result<f64, GyreDataError> {
    Ok(limit(eval_center(r, v, k_a, k_b)?, lo, hi))
}

/// `dxa`, the abundance perturbation for $\partial\ln\Gamma_1/\partial Y$
/// (`pulse_fgong.f90:98`).
const DXA: f64 = 1.0e-3;

/// `[dlnGamma1_dlnrho, dlnGamma1_dlnP, dlnGamma1_dY]` from three
/// `get_eos` calls at `(rho, T)` (`pulse_fgong.f90:298-330`): low X, high X
/// and the cell abundances `xa0` of cell `k_x`.
fn gamma1_derivatives(
    s: &PulseDataInput<'_>,
    k_x: usize,
    rho: f64,
    t: f64,
) -> Result<[f64; 3], GyreDataError> {
    let (Some(h1), Some(he4)) = (s.iso.h1, s.iso.he4) else {
        // `xa(h1)` with `h1 = 0` reads outside the array (DECISIONS #43).
        return Err(GyreDataError::Upstream(
            "get_fgong_data: the net has no h1 or he4 (xa(0) is undefined)",
        ));
    };
    let ht = crate::micro::builtin_helm_table()
        .map_err(|_| GyreDataError::Upstream("helm table unavailable"))?;
    let xa0 = &s.xa[(k_x - 1) * s.species..k_x * s.species];
    // `get_eos` (`star/private/eos_support.f90:44-101`); `ierr` is not
    // checked upstream, so a failed call has no defined result.
    let get_eos = |xa: &[f64]| {
        if flexmesa_math::log10(rho) < -25.0 {
            return Err(GyreDataError::Upstream(
                "eos evaluated at too low a density",
            ));
        }
        eos_dt_get(ht, rho, t, s.chem_id, xa, s.eos_config)
            .map_err(|_| GyreDataError::Upstream("get_eos failed"))
    };
    let mut xa = xa0.to_vec();
    xa[h1] = xa0[h1] - DXA;
    xa[he4] = xa0[he4] + DXA;
    let res_lo_x = get_eos(&xa)?.res;
    xa[h1] = xa0[h1] + DXA;
    xa[he4] = xa0[he4] - DXA;
    let res_hi_x = get_eos(&xa)?.res;
    let out = get_eos(xa0)?;
    let (res, dres_dlnrho, dres_dlnt) = (out.res, out.d_dlnd, out.d_dln_t);
    // dres_dxa(:,1) = (res_hiX-res_loX)/2/dxa
    let mut dres_dxa1 = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    for (d, (hi, lo)) in dres_dxa1
        .iter_mut()
        .zip(res_hi_x.iter().zip(res_lo_x.iter()))
    {
        *d = (hi - lo) / 2.0 / DXA;
    }
    let dlng1_dlnrho = (dres_dlnrho[I_GAMMA1]
        - dres_dlnt[I_GAMMA1] * res[I_CHI_RHO] / res[I_CHI_T])
        / res[I_GAMMA1];
    let dlng1_dlnp = dres_dlnt[I_GAMMA1] / res[I_CHI_T] / res[I_GAMMA1];
    let dlng1_dy = dres_dxa1[I_GAMMA1]
        - dres_dlnt[I_GAMMA1] * dres_dxa1[I_LN_PGAS] / res[I_CHI_T];
    let dlng1_dy = -dlng1_dy / res[I_GAMMA1];
    Ok([dlng1_dlnrho, dlng1_dlnp, dlng1_dy])
}

/// `get_fgong_data(id, add_center_point, keep_surface_point,
/// add_atmosphere, global_data, point_data, ierr)`
/// (`star/private/pulse_fgong.f90:46-630`).
///
/// # Errors
///
/// [`GyreDataError::Unported`] for an attached atmosphere (`build_atm`);
/// [`GyreDataError::Upstream`] where upstream stops or reads an undefined
/// value.
#[allow(clippy::too_many_lines)]
pub fn get_fgong_data(
    s: &PulseDataInput<'_>,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), GyreDataError> {
    let b = &s.base;
    let nz = b.r.len();
    let (k_a, k_b) = set_segment_indices(b, add_center_point)?;
    let n_sg = k_a.len();
    if add_atmosphere {
        return Err(GyreDataError::Unported {
            what: "add_atmosphere (build_atm)",
            at: "star/private/pulse_fgong.f90:136-146",
        });
    }
    let nn_atm = 0usize;
    let n_env = nz;
    let nn_env = if keep_surface_point {
        n_env + n_sg - 1
    } else {
        n_env - 1 + n_sg - 1
    };
    let nn = if add_center_point {
        nn_env + nn_atm + 1
    } else {
        nn_env + nn_atm
    };

    let mut g = vec![0.0_f64; FGONG_ICONST];
    let r_outer = RSUN * b.photosphere_r;
    let m_outer = at(b.m_grav, 1);
    g[0] = m_outer;
    g[1] = r_outer;
    g[2] = at(b.l, 1);
    g[3] = s.initial_z;
    g[4] = 1.0 - (s.initial_y + s.initial_z);
    g[5] = s.mixing_length_alpha;
    let rho_c = if b.interpolate_rho_for_pulse_data {
        eval_center_rho(b, k_b[n_sg - 1])
    } else {
        eval_center(b.rmid, b.rho, k_a[n_sg - 1], k_b[n_sg - 1])?
    };
    // at the centre d2P/dr2 = -4 pi G rho^2/3
    let d2p_dr2_c = -(FOUR_THIRDS * PI * at(b.cgrav, nz) * (rho_c * rho_c));
    let rm = at(b.rmid, nz);
    let p_c = at(b.peos, nz) - 0.5 * d2p_dr2_c * (rm * rm);
    g[10] = r_outer * r_outer * d2p_dr2_c / p_c;
    g[11] = r_outer
        * r_outer
        * eval_center_d2(b.rmid, b.rho, k_a[n_sg - 1], k_b[n_sg - 1])?
        / rho_c;
    g[12] = s.star_age;
    g[13] = s.teff;
    g[14] = STANDARD_CGRAV;

    // `s%dxdt_nuc(h1,:)` with `h1 = 0` reads outside the array
    // (DECISIONS #43).
    let dxdt_h1 = s.dxdt_nuc_h1.as_deref().ok_or(GyreDataError::Upstream(
        "get_fgong_data: the net has no h1 (dxdt_nuc(0,:) is undefined)",
    ))?;
    let mut pd = vec![vec![0.0_f64; FGONG_IVAR]; nn];
    let dq = b.dq;
    let iso = s.iso;

    let abundances =
        |p: &mut [f64],
         face: &dyn Fn(Option<usize>) -> Result<f64, GyreDataError>|
         -> Result<(), GyreDataError> {
            p[20] = face(iso.he3)?;
            p[21] = face(iso.c12)?;
            p[22] = face(iso.c13)?;
            p[23] = face(iso.n14)?;
            p[24] = face(iso.o16)?;
            p[28] = face(iso.h2)?;
            p[29] = face(iso.he4)?;
            p[30] = face(iso.li7)?;
            p[31] = face(iso.be7)?;
            p[32] = face(iso.n15)?;
            p[33] = face(iso.o17)?;
            p[34] = face(iso.o18)?;
            p[35] = face(iso.ne20)?;
            Ok(())
        };

    // store_point_data_env (pulse_fgong.f90:335-454)
    let store_env = |p: &mut [f64],
                     k: usize,
                     k_a: usize,
                     k_b: usize|
     -> Result<(), GyreDataError> {
        let r = at(b.r, k);
        p[0] = r;
        p[1] = flexmesa_math::log(at(b.m_grav, k) / m_outer);
        let t = eval_face(dq, b.t, k, 1, nz)?;
        p[2] = t;
        p[3] = eval_face(dq, b.peos, k, 1, nz)?;
        let rho = if b.interpolate_rho_for_pulse_data {
            eval_face(dq, b.rho, k, k_a, k_b)?
        } else {
            eval_face_rho(b, k, k_a, k_b)?
        };
        p[4] = rho;
        p[5] = eval_face_lim(dq, &s.x, k, k_a, k_b, 0.0, 1.0)?;
        p[6] = at(b.l, k);
        p[7] = eval_face(dq, b.opacity, k, k_a, k_b)?;
        p[8] = eval_face(dq, b.eps_nuc, k, k_a, k_b)?
            + eval_face(dq, &s.eps_grav, k, k_a, k_b)?;
        p[9] = eval_face(dq, b.gamma1, k, k_a, k_b)?;
        p[10] = eval_face(dq, b.grada, k, k_a, k_b)?;
        p[11] = eval_face(dq, b.chi_t, k, k_a, k_b)?
            / eval_face(dq, b.chi_rho, k, k_a, k_b)?;
        p[12] = eval_face(dq, s.cp, k, k_a, k_b)?;
        p[13] = flexmesa_math::exp(eval_face(dq, s.lnfree_e, k, k_a, k_b)?);
        p[14] = if r <= s.fgong_zero_a_inside_r * RSUN
            && s.mixing_type[k - 1] != MixingType::NoMixing
        {
            0.0
        } else {
            eval_face_a_ast(b, k, k_a, k_b)?
        };
        p[15] = eval_face(dq, dxdt_h1, k, k_a, k_b)?;
        p[16] = limit(
            1.0 - eval_face(dq, &s.x, k, k_a, k_b)?
                - eval_face(dq, &s.y, k, k_a, k_b)?,
            0.0,
            1.0,
        );
        p[17] = r_outer - at(b.r, k);
        p[18] = eval_face(dq, &s.eps_grav, k, k_a, k_b)?;
        abundances(p, &|i| s.eval_face_x(i, k, k_a, k_b))?;
        let [d1, d2, d3] = gamma1_derivatives(s, k, rho, t)?;
        p[25] = d1;
        p[26] = d2;
        p[27] = d3;
        Ok(())
    };

    let mut j = 0usize;
    let mut sg = 0usize;
    for k in 1..=n_env {
        if k == 1 && !keep_surface_point {
            continue;
        }
        store_env(&mut pd[j], k, k_a[sg], k_b[sg])?;
        j += 1;
        if k == k_b[sg] + 1 {
            sg += 1;
            store_env(&mut pd[j], k, k_a[sg], k_b[sg])?;
            j += 1;
        }
    }

    // store_point_data_ctr (pulse_fgong.f90:457-575)
    if add_center_point {
        let (k_a, k_b) = (k_a[n_sg - 1], k_b[n_sg - 1]);
        let rmid = b.rmid;
        let p = &mut pd[j];
        p[0] = 0.0;
        p[1] = flexmesa_math::log(f64::MIN_POSITIVE);
        let t = eval_center(rmid, b.t, 1, nz)?;
        p[2] = t;
        p[3] = p_c;
        p[4] = rho_c;
        p[5] = eval_center_lim(rmid, &s.x, k_a, k_b, 0.0, 1.0)?;
        p[6] = 0.0;
        p[7] = eval_center(rmid, b.opacity, k_a, k_b)?;
        p[8] = eval_center(rmid, b.eps_nuc, k_a, k_b)?
            + eval_center(rmid, &s.eps_grav, k_a, k_b)?;
        p[9] = eval_center(rmid, b.gamma1, k_a, k_b)?;
        p[10] = eval_center(rmid, b.grada, k_a, k_b)?;
        p[11] = eval_center(rmid, b.chi_t, k_a, k_b)?
            / eval_center(rmid, b.chi_rho, k_a, k_b)?;
        p[12] = eval_center(rmid, s.cp, k_a, k_b)?;
        p[13] = flexmesa_math::exp(eval_center(rmid, s.lnfree_e, k_a, k_b)?);
        p[14] = 0.0;
        p[15] = eval_center(rmid, dxdt_h1, k_a, k_b)?;
        p[16] = limit(
            1.0 - eval_center(rmid, &s.x, k_a, k_b)?
                - eval_center(rmid, &s.y, k_a, k_b)?,
            0.0,
            1.0,
        );
        p[17] = r_outer;
        p[18] = eval_center(rmid, &s.eps_grav, k_a, k_b)?;
        abundances(p, &|i| s.eval_center_x(i, k_a, k_b))?;
        // "I assume the abundances at r=0 are the same as in the innermost
        // cell" (pulse_fgong.f90:534-536).
        let [d1, d2, d3] = gamma1_derivatives(s, nz, rho_c, t)?;
        p[25] = d1;
        p[26] = d2;
        p[27] = d3;
        j += 1;
    }
    if j != nn {
        return Err(GyreDataError::Upstream(
            "get_fgong_data: point count mismatch",
        ));
    }
    Ok((g, pd))
}

impl<'a> PulseDataInput<'a> {
    /// The view of a [`StarModel`] that `get_fgong_data` and
    /// `get_osc_data` read.
    ///
    /// # Errors
    ///
    /// As [`GyreDataInput::from_model`], and
    /// [`GyreDataError::Upstream`] if a model array is shorter than `nz`
    /// or the composition cannot be evaluated.
    pub fn from_model(model: &'a StarModel) -> Result<Self, GyreDataError> {
        let base = GyreDataInput::from_model(model)?;
        let nz = model.nz;
        let species = model.species;
        let short = || GyreDataError::Upstream("model array shorter than nz");
        let take = |v: &'a [f64]| v.get(..nz).ok_or_else(short);
        let xa = model.xa.get(..nz * species).ok_or_else(short)?;
        let idx = |name: &str| model.iso_names.iter().position(|n| n == name);
        let iso = PulseIsoIndices {
            h1: idx("h1"),
            h2: idx("h2"),
            he3: idx("he3"),
            he4: idx("he4"),
            li7: idx("li7"),
            be7: idx("be7"),
            be9: idx("be9"),
            c12: idx("c12"),
            c13: idx("c13"),
            n14: idx("n14"),
            n15: idx("n15"),
            o16: idx("o16"),
            o17: idx("o17"),
            o18: idx("o18"),
            ne20: idx("ne20"),
            si28: idx("si28"),
        };
        // `basic_composition_info` (`star/private/micro.f90:321-324`).
        let catalog = flexmesa_chem::NuclideCatalog::builtin_ref();
        let mut x = Vec::with_capacity(nz);
        let mut y = Vec::with_capacity(nz);
        for k in 0..nz {
            let c = flexmesa_chem::get_composition_info(
                catalog,
                &model.chem_id,
                &xa[k * species..(k + 1) * species],
                true,
            )
            .map_err(|_| GyreDataError::Upstream("get_composition_info"))?;
            x.push(c.xh);
            y.push(c.xhe);
        }
        let eps_grav: Vec<f64> = model
            .eps_grav_ad
            .get(..nz)
            .ok_or_else(short)?
            .iter()
            .map(|a| a.val)
            .collect();
        let dxdt_nuc_h1 = match iso.h1 {
            Some(h1) => {
                if model.dxdt_nuc.len() < nz * species {
                    return Err(short());
                }
                Some(
                    (0..nz).map(|k| model.dxdt_nuc[k * species + h1]).collect(),
                )
            }
            None => None,
        };
        let summary = crate::report_utils::eval_conv_and_mx_summary(model);
        let c = &model.controls;
        Ok(Self {
            base,
            species,
            xa,
            iso,
            x,
            y,
            cp: take(&model.cp)?,
            lnfree_e: take(&model.lnfree_e)?,
            eps_grav,
            dxdt_nuc_h1,
            mixing_type: model.mixing_type.get(..nz).ok_or_else(short)?,
            pgas: take(&model.pgas)?,
            gradr: take(&model.gradr)?,
            chem_id: &model.chem_id,
            eos_config: &model.eos_config,
            teff: model.teff,
            star_age: model.star_age,
            initial_z: c.initial_z,
            initial_y: c.initial_y,
            mixing_length_alpha: c.mixing_length_alpha,
            fgong_zero_a_inside_r: c.fgong_zero_a_inside_r,
            largest_conv_mixing_region_bottom: summary
                .largest_conv_top_bot
                .map(|(_, bot)| bot + 1),
        })
    }
}
