// SPDX-License-Identifier: GPL-3.0-or-later
//! FGONG pulsation data: `get_fgong_data` (`star/private/pulse_fgong.f90:46-630`),
//! atmospheric attachment via `build_atm` (`star/private/atm_support.f90:1001-1287`),
//! and the `pulse_utils` routines only FGONG and OSC use (`eval_face_X`,
//! `eval_center_X`, `eval_center_d2`, `star/private/pulse_utils.f90`),
//! transcribed in upstream order of operations (#491, #171).
//!
//! The producer reads a [`PulseDataInput`] view of the star (MESA cell or
//! face `k = 1..nz`, surface first, stored 0-based) and returns
//! `global_data(15)` and `point_data(40, nn)` from the surface inwards, as
//! [`crate::pulse_write::write_fgong_data`] writes them.

use flexmesa_atm::{
    build_t_tau_uniform, build_t_tau_varying, eval_t_tau_uniform,
    eval_t_tau_varying, radiation_pressure, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, AtmStructure, ATM_CHIRHO, ATM_CHIT, ATM_CP,
    ATM_DELTA_R, ATM_GAMMA1, ATM_GRADA, ATM_GRADT, ATM_KAP, ATM_LND,
    ATM_LNFREE_E, ATM_LNP, ATM_LNT,
};
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{
    MixingType, AVO, FOUR_THIRDS, KERG, LN10, PI, RSUN, STANDARD_CGRAV,
};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, EosConfig, I_CHI_RHO, I_CHI_T, I_ETA, I_GAMMA1,
    I_LN_FREE_E, I_LN_PGAS, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_kap::{get_kap_results, KapConfig};

use crate::controls::StarControls;
use crate::job_controls::{AtmOption, AtmTTauOpacity, AtmTTauRelation};
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
    /// Optional atmospheric parameters and surface composition metadata read
    /// when `add_atmosphere = true`.
    pub atm: Option<PulseAtmInput<'a>>,
}

/// Atmospheric controls and surface scalar state consumed by `build_atm`
/// (`star/private/atm_support.f90:1001-1287`) when `get_fgong_data` runs with
/// `add_atmosphere = true`.
#[derive(Debug, Clone, Copy)]
pub struct PulseAtmInput<'a> {
    /// `s%atm_option`.
    pub atm_option: AtmOption,
    /// `s%atm_T_tau_relation`.
    pub atm_t_tau_relation: AtmTTauRelation,
    /// `s%atm_T_tau_opacity`.
    pub atm_t_tau_opacity: AtmTTauOpacity,
    /// `s%atm_T_tau_errtol`.
    pub atm_t_tau_errtol: f64,
    /// `s%atm_T_tau_max_iters`.
    pub atm_t_tau_max_iters: usize,
    /// `s%atm_T_tau_max_steps`.
    pub atm_t_tau_max_steps: usize,
    /// `s%atm_build_tau_outer`.
    pub atm_build_tau_outer: f64,
    /// `s%atm_build_dlogtau`.
    pub atm_build_dlogtau: f64,
    /// `s%atm_build_errtol`.
    pub atm_build_errtol: f64,
    /// `s%Pextra_factor`.
    pub pextra_factor: f64,
    /// `s%tau_factor`.
    pub tau_factor: f64,
    /// `s%tau_base`.
    pub tau_base: f64,
    /// Surface cell mean atomic mass `s%abar(1)`.
    pub abar_surf: f64,
    /// Surface cell mean ion charge `s%zbar(1)`.
    pub zbar_surf: f64,
    /// `s%kap_handle` settings.
    pub kap_config: &'a KapConfig,
    /// Controls providing `opacity_factor`, `opacity_min`, and `opacity_max`.
    pub kap_controls: &'a StarControls,
    /// `s%use_other_kap`.
    pub use_other_kap: bool,
    /// `s%use_simple_es_for_kap`.
    pub use_simple_es_for_kap: bool,
}

impl<'a> PulseAtmInput<'a> {
    /// Construct [`PulseAtmInput`] from a [`StarModel`], returning `None` if
    /// `model.abar` or `model.zbar` is empty.
    #[must_use]
    pub fn from_model(model: &'a StarModel) -> Option<Self> {
        let &abar_surf = model.abar.first()?;
        let &zbar_surf = model.zbar.first()?;
        let c = &model.controls;
        Some(Self {
            atm_option: c.atm_option,
            atm_t_tau_relation: c.atm_t_tau_relation,
            atm_t_tau_opacity: c.atm_t_tau_opacity,
            atm_t_tau_errtol: c.atm_t_tau_errtol,
            atm_t_tau_max_iters: c.atm_t_tau_max_iters,
            atm_t_tau_max_steps: c.atm_t_tau_max_steps,
            // MESA 26.4.1 `star/defaults/controls.defaults:4741-4743`:
            // `atm_build_tau_outer = 1d-3`, `atm_build_dlogtau = 0.01` (default-real
            // literal widened to double, `0x3F84_7AE1_4000_0000`), `atm_build_errtol = 1d-8`.
            atm_build_tau_outer: 1.0e-3,
            atm_build_dlogtau: f64::from(0.01_f32),
            atm_build_errtol: 1.0e-8,
            pextra_factor: c.pextra_factor,
            tau_factor: c.tau_factor,
            tau_base: model.tau_base,
            abar_surf,
            zbar_surf,
            kap_config: &model.kap_config,
            kap_controls: c,
            use_other_kap: c.use_other_kap,
            use_simple_es_for_kap: c.use_simple_es_for_kap,
        })
    }
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
        // `xa(h1)` with `h1 = 0` reads outside the array (DECISIONS #43, deferred #699).
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

/// `LOGRHO_TOL` and `LOGPGAS_TOL` in `atm_support::eos_proc`
/// (`star/private/atm_support.f90:1194-1195`).
const ATM_LOG_TOL: f64 = 1.0e-11;

/// Maximum Newton iterations in `solve_eos_given_PgasT`
/// (`star/private/eos_support.f90:345`).
const ATM_EOS_MAX_ITERS: usize = 100;

/// Precomputed surface envelope quantities shared across all `store_point_data_atm`
/// rows (`star/private/pulse_fgong.f90:142-151, 240-330`).
struct SurfaceAtmFgong {
    lnq: f64,
    x: f64,
    l: f64,
    z: f64,
    iso_row: [f64; FGONG_IVAR],
}

impl SurfaceAtmFgong {
    fn new(
        s: &PulseDataInput<'_>,
        k_a0: usize,
        k_b0: usize,
        m_outer: f64,
        abundances: &dyn Fn(
            &mut [f64],
            &dyn Fn(Option<usize>) -> Result<f64, GyreDataError>,
        ) -> Result<(), GyreDataError>,
    ) -> Result<Self, GyreDataError> {
        let b = &s.base;
        let dq = b.dq;
        let lnq = flexmesa_math::log(at(b.m_grav, 1) / m_outer);
        let x = eval_face_lim(dq, &s.x, 1, k_a0, k_b0, 0.0, 1.0)?;
        let z = limit(
            1.0 - eval_face(dq, &s.x, 1, k_a0, k_b0)?
                - eval_face(dq, &s.y, 1, k_a0, k_b0)?,
            0.0,
            1.0,
        );
        let mut iso_row = [0.0_f64; FGONG_IVAR];
        abundances(&mut iso_row, &|i| s.eval_face_x(i, 1, k_a0, k_b0))?;
        Ok(Self {
            lnq,
            x,
            l: at(b.l, 1),
            z,
            iso_row,
        })
    }
}

fn atm_kap_composition(
    s: &PulseDataInput<'_>,
    xa_surf: &[f64],
) -> (f64, f64, f64, f64, f64, f64) {
    let catalog = NuclideCatalog::builtin_ref();
    let (mut xh_s, mut xhe_s, mut xc, mut xn, mut xo, mut xne) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (j, &cid) in s.chem_id.iter().enumerate() {
        if let Some(n) = catalog.get_by_id(cid) {
            match (n.z, n.z_plus_n) {
                (1, 1 | 2) => xh_s += xa_surf[j],
                (2, 3 | 4) => xhe_s += xa_surf[j],
                (6, _) => xc += xa_surf[j],
                (7, _) => xn += xa_surf[j],
                (8, _) => xo += xa_surf[j],
                (10, _) => xne += xa_surf[j],
                _ => {}
            }
        }
    }
    let xh_kap = xh_s.clamp(0.0, 1.0);
    let xhe_kap = xhe_s.clamp(0.0, 1.0);
    let z_kap = (1.0 - (xh_kap + xhe_kap)).clamp(0.0, 1.0);
    (xh_kap, z_kap, xc, xn, xo, xne)
}

/// Builds the atmosphere profile via upstream `build_atm`
/// (`star/private/atm_support.f90:1001-1287`).
#[allow(clippy::too_many_lines)]
fn build_atm_structure(
    s: &PulseDataInput<'_>,
) -> Result<AtmStructure, GyreDataError> {
    let atm = s.atm.as_ref().ok_or(GyreDataError::Upstream(
        "get_fgong_data: add_atmosphere requires PulseAtmInput",
    ))?;
    if atm.atm_option != AtmOption::TTau {
        return Err(GyreDataError::Unported {
            what: "add_atmosphere for non-T_tau atm_option (build_atm)",
            at: "star/private/atm_support.f90:1032-1166",
        });
    }
    if atm.use_other_kap {
        return Err(GyreDataError::Unported {
            what: "build_atm with use_other_kap = .true.",
            at: "star/private/atm_support.f90:1244-1267",
        });
    }
    let sp = s.species;
    if sp == 0 || s.chem_id.len() != sp || s.xa.len() < sp {
        return Err(GyreDataError::Upstream(
            "build_atm: species/chem_id/xa surface slice mismatch",
        ));
    }
    let b = &s.base;
    if b.r[0] <= 0.0
        || !b.r[0].is_finite()
        || b.m_grav[0] <= 0.0
        || !b.m_grav[0].is_finite()
    {
        return Err(GyreDataError::Upstream(
            "build_atm: invalid surface radius or mass",
        ));
    }

    let tau_surf = atm.tau_factor * atm.tau_base;
    let l = at(b.l, 1);
    let r = at(b.r, 1);
    let teff = s.teff;
    let m = at(b.m_grav, 1);
    let cgrav = at(b.cgrav, 1);
    let t_tau_id = atm.atm_t_tau_relation.id();
    let ht = crate::micro::builtin_helm_table()
        .map_err(|_| GyreDataError::Upstream("helm table unavailable"))?;

    let xa_surf = &s.xa[0..sp];
    let abar_surf = atm.abar_surf;
    let zbar_surf = atm.zbar_surf;
    let (xh_kap, z_kap, xc_kap, xn_kap, xo_kap, xne_kap) =
        atm_kap_composition(s, xa_surf);

    let eos_cfg = s.eos_config;
    let chem_id = s.chem_id;
    let kap_cfg = atm.kap_config;
    let ctrl = atm.kap_controls;
    let use_simple_es = atm.use_simple_es_for_kap;

    let eos_proc =
        |ln_p: f64, ln_t: f64| -> Result<AtmEosCallbackResult, AtmError> {
            let t = flexmesa_math::exp(ln_t);
            let p = flexmesa_math::exp(ln_p);
            let prad = radiation_pressure(t);
            let pgas = (p - prad).max(1.0e-99);
            let rho_guess = ((p / t) * abar_surf / (AVO * KERG)).max(1.0e-40);
            let inv = eos_dt_get_rho(
                ht,
                chem_id,
                xa_surf,
                ln_t / LN10,
                I_LN_PGAS as isize,
                flexmesa_math::log10(pgas) * LN10,
                flexmesa_math::log10(rho_guess),
                ATM_LOG_TOL,
                ATM_LOG_TOL * LN10,
                ATM_EOS_MAX_ITERS,
                eos_cfg,
            )
            .map_err(|e| AtmError::CallbackError {
                context: "build_atm::eos_proc",
                detail: e.to_string(),
            })?;
            Ok(AtmEosCallbackResult {
                ln_rho: inv.log10_result * LN10,
                res: inv.eos.res,
                dres_dln_rho: inv.eos.d_dlnd,
                dres_dln_t: inv.eos.d_dln_t,
            })
        };

    let kap_proc = |ln_rho: f64,
                    ln_t: f64,
                    eos: &AtmEosCallbackResult|
     -> Result<AtmKapCallbackResult, AtmError> {
        let log_rho = ln_rho / LN10;
        let log_t = ln_t / LN10;
        let (kap_raw, dlnkap_dlnd_raw, dlnkap_dlnt_raw) = if use_simple_es {
            (0.2 * (1.0 + xh_kap), 0.0, 0.0)
        } else {
            let out = get_kap_results(
                kap_cfg,
                zbar_surf,
                xh_kap,
                z_kap,
                xc_kap,
                xn_kap,
                xo_kap,
                xne_kap,
                log_rho,
                log_t,
                eos.res[I_LN_FREE_E],
                eos.dres_dln_rho[I_LN_FREE_E],
                eos.dres_dln_t[I_LN_FREE_E],
                eos.res[I_ETA],
                eos.dres_dln_rho[I_ETA],
                eos.dres_dln_t[I_ETA],
            )
            .map_err(|e| AtmError::CallbackError {
                context: "build_atm::kap_proc",
                detail: e.to_string(),
            })?;
            (out.kap, out.dlnkap_dln_rho, out.dlnkap_dln_t)
        };

        let opacity_factor =
            crate::rsp_eos_kap::effective_opacity_factor(ctrl, log_t);
        let mut kap = kap_raw * opacity_factor;
        let mut dlnkap_dlnd = dlnkap_dlnd_raw;
        let mut dlnkap_dlnt = dlnkap_dlnt_raw;
        if ctrl.opacity_max > 0.0 && kap > ctrl.opacity_max {
            kap = ctrl.opacity_max;
            dlnkap_dlnd = 0.0;
            dlnkap_dlnt = 0.0;
        }
        if ctrl.opacity_min > 0.0 && kap < ctrl.opacity_min {
            kap = ctrl.opacity_min;
            dlnkap_dlnd = 0.0;
            dlnkap_dlnt = 0.0;
        }
        if kap <= 0.0 || !kap.is_finite() {
            return Err(AtmError::CallbackError {
                context: "build_atm::kap_proc",
                detail: format!("invalid atmosphere opacity kap={kap}"),
            });
        }
        Ok(AtmKapCallbackResult {
            kap,
            dlnkap_dln_rho: dlnkap_dlnd,
            dlnkap_dln_t: dlnkap_dlnt,
        })
    };

    let run_uniform =
        |max_iters: usize| -> Result<AtmStructure, GyreDataError> {
            let surf = eval_t_tau_uniform(
                tau_surf,
                l,
                r,
                m,
                cgrav,
                at(b.opacity, 1),
                atm.pextra_factor,
                t_tau_id,
                eos_proc,
                kap_proc,
                atm.atm_t_tau_errtol,
                max_iters,
                true,
                teff,
            )
            .map_err(|_| {
                GyreDataError::Upstream("build_atm: eval_t_tau_uniform failed")
            })?;

            build_t_tau_uniform(
                tau_surf,
                l,
                r,
                teff,
                m,
                cgrav,
                surf.kap,
                atm.pextra_factor,
                atm.atm_build_tau_outer,
                t_tau_id,
                eos_proc,
                atm.atm_build_errtol,
                atm.atm_build_dlogtau,
            )
            .map_err(|_| {
                GyreDataError::Upstream("build_atm: build_t_tau_uniform failed")
            })
        };

    match atm.atm_t_tau_opacity {
        AtmTTauOpacity::Fixed => run_uniform(0),
        AtmTTauOpacity::Iterated => run_uniform(atm.atm_t_tau_max_iters),
        AtmTTauOpacity::Varying => {
            let surf = eval_t_tau_varying(
                tau_surf,
                l,
                r,
                m,
                cgrav,
                t_tau_id,
                eos_proc,
                kap_proc,
                atm.atm_t_tau_errtol,
                atm.atm_t_tau_max_steps,
                true,
                teff,
            )
            .map_err(|_| {
                GyreDataError::Upstream("build_atm: eval_t_tau_varying failed")
            })?;

            build_t_tau_varying(
                tau_surf,
                l,
                r,
                teff,
                m,
                cgrav,
                surf.ln_p,
                atm.atm_build_tau_outer,
                t_tau_id,
                eos_proc,
                kap_proc,
                atm.atm_build_errtol,
                atm.atm_build_dlogtau,
            )
            .map_err(|_| {
                GyreDataError::Upstream("build_atm: build_t_tau_varying failed")
            })
        }
    }
}

/// Populates one FGONG atmosphere row (`star/private/pulse_fgong.f90:240-330`).
fn store_point_data_atm(
    s: &PulseDataInput<'_>,
    atm_pt: &[f64; 19],
    surf: &SurfaceAtmFgong,
    p: &mut [f64],
    r_outer: f64,
) -> Result<(), GyreDataError> {
    let b = &s.base;
    let r = at(b.r, 1) + atm_pt[ATM_DELTA_R];
    let t = flexmesa_math::exp(atm_pt[ATM_LNT]);
    let p_tot = flexmesa_math::exp(atm_pt[ATM_LNP]);
    let rho = flexmesa_math::exp(atm_pt[ATM_LND]);
    let kap = atm_pt[ATM_KAP];
    let gamma1 = atm_pt[ATM_GAMMA1];
    let nabla_ad = atm_pt[ATM_GRADA];
    let delta = atm_pt[ATM_CHIT] / atm_pt[ATM_CHIRHO];
    let c_p = atm_pt[ATM_CP];
    let rec_mu_e = flexmesa_math::exp(atm_pt[ATM_LNFREE_E]);

    let grav = at(b.cgrav, 1) * at(b.m_grav, 1) / (r * r);
    let n2 =
        grav * grav * (rho / p_tot) * delta * (nabla_ad - atm_pt[ATM_GRADT]);
    let a_ast = n2 * r / grav;

    p[0] = r;
    p[1] = surf.lnq;
    p[2] = t;
    p[3] = p_tot;
    p[4] = rho;
    p[5] = surf.x;
    p[6] = surf.l;
    p[7] = kap;
    p[8] = 0.0;
    p[9] = gamma1;
    p[10] = nabla_ad;
    p[11] = delta;
    p[12] = c_p;
    p[13] = rec_mu_e;
    p[14] = a_ast;
    p[15] = 0.0;
    p[16] = surf.z;
    p[17] = r_outer - r;
    p[18] = 0.0;

    let [d1, d2, d3] = gamma1_derivatives(s, 1, rho, t)?;
    p[25] = d1;
    p[26] = d2;
    p[27] = d3;

    for slot in [20usize, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35] {
        p[slot] = surf.iso_row[slot];
    }
    Ok(())
}

/// `get_fgong_data(id, add_center_point, keep_surface_point,
/// add_atmosphere, global_data, point_data, ierr)`
/// (`star/private/pulse_fgong.f90:46-630`).
///
/// # Errors
///
/// [`GyreDataError::Unported`] when `add_atmosphere` requests an atmosphere
/// option other than `T_tau` or `use_other_kap = .true.`;
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
    let atm_opt: Option<AtmStructure> = if add_atmosphere {
        Some(build_atm_structure(s)?)
    } else {
        None
    };
    let n_atm = atm_opt.as_ref().map_or(0, |a| a.points.len());
    if add_atmosphere && n_atm < 2 {
        return Err(GyreDataError::Upstream(
            "get_fgong_data: build_atm returned fewer than 2 points",
        ));
    }
    let nn_atm = n_atm.saturating_sub(1);
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
    // (DECISIONS #43, deferred #699).
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
    if let Some(ref atm) = atm_opt {
        let surf =
            SurfaceAtmFgong::new(s, k_a[0], k_b[0], m_outer, &abundances)?;
        for atm_pt in &atm.points[..nn_atm] {
            store_point_data_atm(s, atm_pt, &surf, &mut pd[j], r_outer)?;
            j += 1;
        }
    }
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
            atm: PulseAtmInput::from_model(model),
        })
    }
}
