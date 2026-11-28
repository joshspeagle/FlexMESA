// SPDX-License-Identifier: GPL-3.0-or-later
//! Reaction-network derivative accumulation helpers, weaklib low-$T$ blending,
//! and special reaction updates (`net_derivs_support.f90` and `net_eval.f90`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AVO, CLIGHT, KEV, LN2, ME, MEV_TO_ERGS, QCONV};
use flexmesa_rates::{
    eval_ecapture_pair_with_derivs, eval_fl_epsnuc_3alf_with_derivs,
    eval_reaclib_terms, eval_weak_reaction_pair, CoulombMuiOption,
    CoulombVsOption, RatesCatalog, ReaclibTerm, WeakRateEval,
    DEFAULT_REACLIB_MIN_T9,
};

use crate::types::{
    NetGeneralInfo, IC12, IHE4, IN14, INE20, NUM_CATEGORIES, NUM_RVS,
};
use flexmesa_math::Quad;

const ZERO_WEAK: WeakRateEval = WeakRateEval {
    lambda: 0.0,
    dlambda_dln_t: 0.0,
    dlambda_dln_rho: 0.0,
    q: 0.0,
    dq_dln_t: 0.0,
    dq_dln_rho: 0.0,
    qneu: 0.0,
    dqneu_dln_t: 0.0,
    dqneu_dln_rho: 0.0,
};

/// Evaluates `FL_epsnuc_3alf(T, Rho, Y, UE)` and its analytical `(T, Rho)`
/// partials matching `rates/private/pycno.f90:31-248` (`0 ULP` parity).
#[must_use]
pub fn fl_epsnuc_3alf_with_derivs(
    t: f64,
    rho: f64,
    y: f64,
    ue: f64,
) -> (f64, f64, f64) {
    eval_fl_epsnuc_3alf_with_derivs(t, rho, y, ue)
}

const R_MIN: f64 = 1.0e-99;
const T9_WEAKLIB_FULL_OFF: f64 = 0.01;
const T9_WEAKLIB_FULL_ON: f64 = 0.02;
const WEAKLIB_BLEND_HI_Z: usize = 26;
const T9_WEAKLIB_FULL_OFF_HI_Z: f64 = 0.063;
const T9_WEAKLIB_FULL_ON_HI_Z: f64 = 0.073;

/// Branch-checked view of populated output buffers on [`NetWorkState`].
///
/// Network evaluation populates either `DirectCgs` (when `g.doing_approx21`
/// is true) or `Molar` otherwise, while poisoning the inactive branch's
/// buffers with `f64::NAN`.
#[derive(Debug, Clone, Copy)]
pub enum NetWorkOutputs<'a> {
    /// Direct CGS / mass-fraction outputs when `g.doing_approx21` is true.
    DirectCgs {
        /// Category energy rates ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
        eps_nuc_categories: &'a [f64; NUM_CATEGORIES],
        /// $\partial\epsilon_{\mathrm{nuc}}/\partial X_i$ (`num_isos`).
        d_eps_nuc_dx: &'a [f64],
        /// Mass-fraction rates $dX_i/dt$ ($\mathrm{s^{-1}}$, `num_isos`).
        dxdt: &'a [f64],
        /// $\partial(dX_i/dt)/\partial T$ (`num_isos`).
        d_dxdt_dt: &'a [f64],
        /// $\partial(dX_i/dt)/\partial\rho$ (`num_isos`).
        d_dxdt_drho: &'a [f64],
        /// $\partial(dX_i/dt)/\partial X_j$ (`num_isos * num_isos`).
        d_dxdt_dx: &'a [f64],
    },
    /// Molar / MeV outputs populated for non-`approx21` networks.
    Molar {
        /// Category energy rates in $\mathrm{MeV\,g^{-1}\,s^{-1}}$.
        eps_nuc_categories_mev: &'a [f64; NUM_CATEGORIES],
        /// Total neutrino loss rate in $\mathrm{MeV\,g^{-1}\,s^{-1}}$.
        eps_neu_total_mev: f64,
        /// $\partial\epsilon_{\mathrm{nuc,MeV}}/\partial Y_i$ (`num_isos`).
        d_eps_nuc_dy: &'a [f64],
        /// Molar Jacobian $\partial(dY_i/dt)/\partial Y_j$
        /// (`num_isos * num_isos`).
        d_dydt_dy: &'a [f64],
    },
}

/// Per-evaluation working buffers corresponding to `type(Net_Info)` in
/// `net/public/net_def.f90`.
#[derive(Debug, Clone)]
pub struct NetWorkState {
    /// Molar abundances $Y_i = X_i / A_i$ (`num_isos`).
    pub y: Vec<f64>,
    /// Unscreened reaction rates (`num_reactions`).
    pub rate_raw: Vec<f64>,
    /// $\partial r_{\mathrm{raw}} / \partial T$ (`num_reactions`).
    pub rate_raw_dt: Vec<f64>,
    /// $\partial r_{\mathrm{raw}} / \partial \rho$ (`num_reactions`).
    pub rate_raw_drho: Vec<f64>,
    /// Screened reaction rates (`num_reactions`).
    pub rate_screened: Vec<f64>,
    /// $\partial r_{\mathrm{screened}} / \partial T$ (`num_reactions`).
    pub rate_screened_dt: Vec<f64>,
    /// $\partial r_{\mathrm{screened}} / \partial \rho$ (`num_reactions`).
    pub rate_screened_drho: Vec<f64>,
    /// Exported unscreened rate $\times Y \times N_A$ (`num_reactions`).
    pub raw_rate: Vec<f64>,
    /// Exported screened rate $\times Y \times N_A$ (`num_reactions`).
    pub screened_rate: Vec<f64>,
    /// Nuclear energy rate per reaction in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`num_reactions`).
    pub eps_nuc_rate: Vec<f64>,
    /// Neutrino loss rate per reaction in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`num_reactions`).
    pub eps_neu_rate: Vec<f64>,
    /// Nuclear energy generation by category (`NUM_CATEGORIES = 24`).
    pub(crate) eps_nuc_categories: [f64; NUM_CATEGORIES],
    /// Total neutrino loss rate in $\mathrm{MeV\,g^{-1}\,s^{-1}}$ (scaled by `QCONV` on export).
    pub eps_neu_total: f64,
    /// $\partial \epsilon_{\mathrm{nuc,MeV}} / \partial Y_i$ (`num_isos`).
    pub(crate) d_eps_nuc_dy: Vec<f64>,
    /// Molar Jacobian $\partial (dY_i/dt) / \partial Y_j$ (`num_isos * num_isos`, row-major).
    pub(crate) d_dydt_dy: Vec<f64>,
    /// Direct CGS category energy rates ($\mathrm{erg\,g^{-1}\,s^{-1}}$)
    /// populated when `g.doing_approx21` (`net_eval.f90:232-237,308-338`).
    pub(crate) direct_eps_nuc_categories: [f64; NUM_CATEGORIES],
    /// Direct CGS $\partial\epsilon_{\mathrm{nuc}}/\partial X_i$ (`num_isos`)
    /// populated when `g.doing_approx21` (`net_eval.f90:431`).
    pub(crate) direct_d_eps_nuc_dx: Vec<f64>,
    /// Direct mass-fraction rates $dX_i/dt$ ($\mathrm{s^{-1}}$, `num_isos`)
    /// populated when `g.doing_approx21` (`net_eval.f90:381`).
    pub(crate) direct_dxdt: Vec<f64>,
    /// Direct $\partial(dX_i/dt)/\partial T$ (`num_isos`) populated when
    /// `g.doing_approx21` (`net_eval.f90:433`).
    pub(crate) direct_d_dxdt_dt: Vec<f64>,
    /// Direct $\partial(dX_i/dt)/\partial\rho$ (`num_isos`) populated when
    /// `g.doing_approx21` (`net_eval.f90:432`).
    pub(crate) direct_d_dxdt_drho: Vec<f64>,
    /// Direct mass-fraction Jacobian $\partial(dX_i/dt)/\partial X_j$
    /// (`num_isos * num_isos`, row-major) populated when `g.doing_approx21`
    /// (`net_eval.f90:435-436`).
    pub(crate) direct_d_dxdt_dx: Vec<f64>,
    last_eval_direct_cgs: Option<bool>,
    /// Weaklib combined rate $\lambda$ (`num_wk_reactions`).
    pub wk_lambda: Vec<f64>,
    /// $\partial \lambda / \partial \ln T$ (`num_wk_reactions`).
    pub wk_dlambda_dln_t: Vec<f64>,
    /// $\partial \lambda / \partial \ln \rho$ (`num_wk_reactions`).
    pub wk_dlambda_dln_rho: Vec<f64>,
    /// Weaklib total $Q$ in MeV (`num_wk_reactions`).
    pub wk_q: Vec<f64>,
    /// Weaklib average neutrino energy $Q_{\nu}$ in MeV (`num_wk_reactions`).
    pub wk_qneu: Vec<f64>,
    /// $\partial Q_{\nu} / \partial \ln T$ (`num_wk_reactions`).
    pub wk_dqneu_dln_t: Vec<f64>,
    /// $\partial Q_{\nu} / \partial \ln \rho$ (`num_wk_reactions`).
    pub wk_dqneu_dln_rho: Vec<f64>,
}

impl NetWorkState {
    /// Allocates working buffers for `g`, poisoning inactive-branch outputs.
    #[must_use]
    pub fn new(g: &NetGeneralInfo) -> Self {
        let ni = g.num_isos;
        let nd = if g.doing_approx21 { ni } else { 0 };
        let nr = g.num_reactions.max(120);
        let nw = g.num_wk_reactions;
        let molar_init = if g.doing_approx21 { f64::NAN } else { 0.0 };
        let direct_cat_init = if g.doing_approx21 { 0.0 } else { f64::NAN };
        Self {
            y: vec![0.0; ni],
            rate_raw: vec![0.0; nr],
            rate_raw_dt: vec![0.0; nr],
            rate_raw_drho: vec![0.0; nr],
            rate_screened: vec![0.0; nr],
            rate_screened_dt: vec![0.0; nr],
            rate_screened_drho: vec![0.0; nr],
            raw_rate: vec![0.0; nr],
            screened_rate: vec![0.0; nr],
            eps_nuc_rate: vec![0.0; nr],
            eps_neu_rate: vec![0.0; nr],
            eps_nuc_categories: [molar_init; NUM_CATEGORIES],
            eps_neu_total: molar_init,
            d_eps_nuc_dy: vec![molar_init; ni],
            d_dydt_dy: vec![molar_init; ni * ni],
            direct_eps_nuc_categories: [direct_cat_init; NUM_CATEGORIES],
            direct_d_eps_nuc_dx: vec![0.0; nd],
            direct_dxdt: vec![0.0; nd],
            direct_d_dxdt_dt: vec![0.0; nd],
            direct_d_dxdt_drho: vec![0.0; nd],
            direct_d_dxdt_dx: vec![0.0; nd * nd],
            last_eval_direct_cgs: None,
            wk_lambda: vec![0.0; nw],
            wk_dlambda_dln_t: vec![0.0; nw],
            wk_dlambda_dln_rho: vec![0.0; nw],
            wk_q: vec![0.0; nw],
            wk_qneu: vec![0.0; nw],
            wk_dqneu_dln_t: vec![0.0; nw],
            wk_dqneu_dln_rho: vec![0.0; nw],
        }
    }

    /// Resets per-evaluation accumulators in `self` for zero-allocation reuse.
    pub fn reset(&mut self, g: &NetGeneralInfo) {
        let ni = g.num_isos;
        let nr = g.num_reactions;
        self.last_eval_direct_cgs = None;
        if g.doing_approx21 {
            self.poison_molar_outputs(ni);
            self.ensure_direct_export_len(ni);
            self.direct_eps_nuc_categories = [0.0; NUM_CATEGORIES];
            self.direct_d_eps_nuc_dx[..ni].fill(0.0);
            self.direct_dxdt[..ni].fill(0.0);
            self.direct_d_dxdt_dt[..ni].fill(0.0);
            self.direct_d_dxdt_drho[..ni].fill(0.0);
            self.direct_d_dxdt_dx[..ni * ni].fill(0.0);
        } else {
            self.eps_nuc_categories = [0.0; NUM_CATEGORIES];
            self.eps_neu_total = 0.0;
            self.d_eps_nuc_dy[..ni].fill(0.0);
            self.d_dydt_dy[..ni * ni].fill(0.0);
            self.poison_direct_outputs();
        }
        self.raw_rate[..nr].fill(0.0);
        self.screened_rate[..nr].fill(0.0);
        self.eps_nuc_rate[..nr].fill(0.0);
        self.eps_neu_rate[..nr].fill(0.0);
    }

    /// Returns a branch-checked view of the output buffers populated by the
    /// most recent full network evaluation.
    ///
    /// # Panics
    ///
    /// Panics if called before a full evaluation (or after a `rates_only`
    /// evaluation or [`Self::reset`]).
    #[must_use]
    pub fn outputs(&self) -> NetWorkOutputs<'_> {
        let ni = self.y.len();
        let direct = self
            .last_eval_direct_cgs
            .expect("NetWorkState::outputs called before full net evaluation");
        if direct {
            NetWorkOutputs::DirectCgs {
                eps_nuc_categories: &self.direct_eps_nuc_categories,
                d_eps_nuc_dx: &self.direct_d_eps_nuc_dx[..ni],
                dxdt: &self.direct_dxdt[..ni],
                d_dxdt_dt: &self.direct_d_dxdt_dt[..ni],
                d_dxdt_drho: &self.direct_d_dxdt_drho[..ni],
                d_dxdt_dx: &self.direct_d_dxdt_dx[..ni * ni],
            }
        } else {
            NetWorkOutputs::Molar {
                eps_nuc_categories_mev: &self.eps_nuc_categories,
                eps_neu_total_mev: self.eps_neu_total,
                d_eps_nuc_dy: &self.d_eps_nuc_dy[..ni],
                d_dydt_dy: &self.d_dydt_dy[..ni * ni],
            }
        }
    }

    pub(crate) fn poison_molar_outputs(&mut self, ni: usize) {
        self.eps_nuc_categories = [f64::NAN; NUM_CATEGORIES];
        self.eps_neu_total = f64::NAN;
        self.d_eps_nuc_dy[..ni].fill(f64::NAN);
        self.d_dydt_dy[..ni * ni].fill(f64::NAN);
    }

    pub(crate) fn poison_direct_outputs(&mut self) {
        self.direct_eps_nuc_categories = [f64::NAN; NUM_CATEGORIES];
        self.direct_d_eps_nuc_dx.fill(f64::NAN);
        self.direct_dxdt.fill(f64::NAN);
        self.direct_d_dxdt_dt.fill(f64::NAN);
        self.direct_d_dxdt_drho.fill(f64::NAN);
        self.direct_d_dxdt_dx.fill(f64::NAN);
    }

    pub(crate) fn set_eval_branch(&mut self, direct_cgs: bool) {
        self.last_eval_direct_cgs = Some(direct_cgs);
    }

    /// Ensures `direct_*` export buffers match `species` isotopes.
    pub(crate) fn ensure_direct_export_len(&mut self, species: usize) {
        if self.direct_dxdt.len() != species {
            self.direct_d_eps_nuc_dx.resize(species, 0.0);
            self.direct_dxdt.resize(species, 0.0);
            self.direct_d_dxdt_dt.resize(species, 0.0);
            self.direct_d_dxdt_drho.resize(species, 0.0);
            self.direct_d_dxdt_dx.resize(species * species, 0.0);
        }
    }
}

/// Evaluates a single weak reaction with low-$T_9$ Reaclib/halflife blending
/// (`eval_weak_reaction_info` in `rates/private/eval_weak.f90`).
#[must_use]
pub fn eval_single_weak_blended(
    rxn_name: &str,
    weak_info: [u16; 2],
    has_weaklib: bool,
    t9: f64,
    ye_rho: f64,
    temp: f64,
    _rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    eta: f64,
    d_eta_dln_t: f64,
    d_eta_dln_rho: f64,
    use_suzuki: bool,
    do_ecapture: bool,
) -> WeakRateEval {
    let nuclides = NuclideCatalog::builtin_ref();
    let cat = RatesCatalog::builtin();
    let cid_lhs = weak_info[0] as usize;
    let cid_rhs = weak_info[1] as usize;
    if cid_lhs == 0 || cid_rhs == 0 {
        return ZERO_WEAK;
    }
    let e_lhs = nuclides.get_by_id(cid_lhs).unwrap();
    let e_rhs = nuclides.get_by_id(cid_rhs).unwrap();

    let mut ev = if has_weaklib {
        // Optional weaklib pair lookup: fall back to Reaclib / half-life below
        // when the pair has no tabulated weaklib entry (upstream weaklib table guard).
        eval_weak_reaction_pair(
            &e_lhs.name,
            &e_rhs.name,
            t9,
            ye_rho,
            use_suzuki,
        )
        .unwrap_or(ZERO_WEAK)
    } else {
        ZERO_WEAK
    };

    let low_t_rate =
        weak_low_t_rate(cat, rxn_name, weak_info[0], weak_info[1], has_weaklib);
    if low_t_rate > 0.0 {
        if !has_weaklib {
            ev.lambda = low_t_rate;
            ev.dlambda_dln_t = 0.0;
            ev.dlambda_dln_rho = 0.0;
        } else {
            let (t_off, t_on) = if e_lhs.z >= WEAKLIB_BLEND_HI_Z {
                (T9_WEAKLIB_FULL_OFF_HI_Z, T9_WEAKLIB_FULL_ON_HI_Z)
            } else {
                (T9_WEAKLIB_FULL_OFF, T9_WEAKLIB_FULL_ON)
            };
            let (alfa, d_alfa_dln_t) = if t9 >= t_on {
                (1.0, 0.0)
            } else if t9 > t_off {
                ((t9 - t_off) / (t_on - t_off), t9 / (t_on - t_off))
            } else {
                (0.0, 0.0)
            };
            let beta = 1.0 - alfa;
            let raw_lam = ev.lambda;
            ev.lambda = alfa * raw_lam + beta * low_t_rate;
            // Matches `rates/private/eval_weak.f90:110-120`: MESA updates
            // `lambda(i)` first and forms `dlambda_dlnT(i)` with post-blend
            // `lambda(i)`.
            ev.dlambda_dln_t = alfa * ev.dlambda_dln_t
                + d_alfa_dln_t * (ev.lambda - low_t_rate);
            ev.dlambda_dln_rho *= alfa;
        }
    }

    let _ = (temp, z2bar);
    // Optional special electron-capture override matching upstream
    // `rates/private/eval_ecapture.f90:99-126` (`do_eval_ecapture`): requires a
    // valid weaklib entry (`ids(i) > 0`), positive thermodynamic inputs, and
    // sufficient electron degeneracy (`eta + beta >= 2 * beta`); otherwise the
    // weaklib/low-T evaluation in `ev` is preserved untouched.
    if do_ecapture
        && has_weaklib
        && t9 > 0.0
        && ye_rho > 0.0
        && zbar > 0.0
        && abar > 0.0
    {
        let kbt = (1.0e3 * KEV) * t9;
        let mec2 = ME * CLIGHT * CLIGHT / MEV_TO_ERGS;
        let beta = mec2 * (1.0 / kbt);
        if eta + beta >= 2.0 * beta {
            if let Ok(ec) = eval_ecapture_pair_with_derivs(
                &e_lhs.name,
                &e_rhs.name,
                t9,
                ye_rho,
                eta,
                d_eta_dln_t,
                d_eta_dln_rho,
                CoulombMuiOption::Pcr2009,
                CoulombVsOption::Itoh2002,
                zbar,
                abar,
            ) {
                ev = ec;
            }
        }
    }

    ev
}

/// Returns `true` when `term` is MESA's weaklib-only `reaclib_rates`
/// placeholder (`also_in_reaclib = .false.`, `chapter = 1`, `rflag = 'w'`,
/// `coefficients = [-99, 0, 0, 0, 0, 0, 0]` in
/// `rates/private/reaclib_input.f90:279-294`).
#[inline]
pub(crate) fn is_weaklib_placeholder(term: &ReaclibTerm) -> bool {
    term.chapter == 1
        && term.rflag == b'w'
        && term.coeffs == [-99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

fn weak_low_t_rate(
    cat: &RatesCatalog,
    rxn_name: &str,
    lhs_id: u16,
    rhs_id: u16,
    has_weaklib: bool,
) -> f64 {
    // In upstream `init_rates_info` (`rates/private/rates_initialize.f90:1256-1262`),
    // predefined `reactions.list` entries call `set_weak_lowT_rate` (`808`, `403-459`)
    // before `finish_rates_def_init` (`55`) extracts `reaclib_rates`. Because
    // `reaclib_rates% reaction_dict` is still null (`reaclib_eval.f90:249-252`),
    // `do_reaclib_indices_for_reaction` fails and falls back to `weak_info.list`
    // (`LN2 / half_life` when present, e.g. `r_be7_wk_li7`, else `-1.0e-99`).
    //
    // Reactions added dynamically via `add_reaction_from_reaclib` (`142-239`,
    // `325`) run after `finish_rates_def_init`: weaklib transitions always
    // resolve at `lo <= num_from_weaklib` (`reaclib_input.f90:279-294`, either
    // via a `-99` placeholder or directly from weaklib), so
    // `also_in_reaclib == .false.` sets `weak_lowT_rate = -1` without
    // evaluating `exp(-99)` or falling back to `weak_info.list`
    // (`rates_initialize.f90:448-451`).
    let is_predefined = cat.reaction_id_by_name(rxn_name).is_some();
    if !is_predefined {
        if let Ok((is_rev, terms)) = cat.reaclib_terms(rxn_name) {
            if !terms.is_empty() {
                let slice = if has_weaklib {
                    &terms[terms.len() - 1..]
                } else {
                    terms
                };
                if has_weaklib && is_weaklib_placeholder(&slice[0]) {
                    return -1.0;
                }
                let (fw, _, rv, _) = eval_reaclib_terms(
                    slice,
                    0.01,
                    DEFAULT_REACLIB_MIN_T9,
                    false,
                );
                return if is_rev { rv } else { fw };
            }
        }
        if has_weaklib {
            return -1.0;
        }
    }
    if let Some((half_life, _)) = cat.weak_info_by_ids(lhs_id, rhs_id) {
        if half_life > 0.0 {
            return LN2 / half_life;
        }
    }
    -1.0e-99
}

/// Populates `w.wk_*` arrays for all weak reactions in `g` (`get_weaklib_rates`
/// in `net/private/net_eval.f90`).
pub fn get_weaklib_rates(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    temp: f64,
    rho: f64,
    ye: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    eta: f64,
    d_eta_dln_t: f64,
    d_eta_dln_rho: f64,
    weak_rate_factor: f64,
    use_suzuki: bool,
    do_ecapture: bool,
) {
    let t9 = temp * 1.0e-9;
    let ye_rho = ye * rho;
    for k in 0..g.num_wk_reactions {
        let rxn_idx = g.weak_reaction_num[k] - 1;
        let rxn = &g.reactions[rxn_idx];
        let has_weaklib = g.weaklib_ids[k] > 0;
        let ev = eval_single_weak_blended(
            &rxn.name,
            rxn.weak_isos,
            has_weaklib,
            t9,
            ye_rho,
            temp,
            rho,
            abar,
            zbar,
            z2bar,
            eta,
            d_eta_dln_t,
            d_eta_dln_rho,
            use_suzuki,
            do_ecapture,
        );
        let fac = if weak_rate_factor < 1.0 {
            weak_rate_factor
        } else {
            1.0
        };
        w.wk_lambda[k] = fac * ev.lambda;
        w.wk_dlambda_dln_t[k] = fac * ev.dlambda_dln_t;
        w.wk_dlambda_dln_rho[k] = fac * ev.dlambda_dln_rho;
        w.wk_q[k] = ev.q;
        w.wk_qneu[k] = ev.qneu;
        w.wk_dqneu_dln_t[k] = ev.dqneu_dln_t;
        w.wk_dqneu_dln_rho[k] = ev.dqneu_dln_rho;
    }
}

/// Accumulates reaction rate and Jacobian contributions for reaction `i` (0-based)
/// with reaction $Q$ and $Q_{\nu}$ (`do_in_out` in `net_derivs_support.f90`).
pub fn do_in_out(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[Quad; NUM_RVS]],
    eps_nuc_mev: &mut [Quad; NUM_RVS],
    i: usize,
    r_in: f64,
    n_in: usize,
    i_in: [usize; 3],
    c_in: [f64; 3],
    n_out: usize,
    i_out: [usize; 3],
    c_out: [f64; 3],
    idr: [usize; 3],
    dr: [f64; 3],
    symbolic: bool,
    just_dydt: bool,
) {
    let q = g.reactions[i].std_q;
    let qneu = g.reactions[i].std_qneu * g.reaction_neuqs_factor;
    do_in_out_neu(
        g,
        w,
        dydt,
        eps_nuc_mev,
        i,
        r_in,
        n_in,
        i_in,
        c_in,
        n_out,
        i_out,
        c_out,
        idr,
        dr,
        q,
        qneu,
        0.0,
        0.0,
        symbolic,
        just_dydt,
    );
}

/// Accumulates reaction rate, neutrino energy derivative, and Jacobian
/// contributions for reaction `i` (0-based) (`do_in_out_neu` in
/// `net_derivs_support.f90`).
pub fn do_in_out_neu(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[Quad; NUM_RVS]],
    eps_nuc_mev: &mut [Quad; NUM_RVS],
    i: usize,
    r_in: f64,
    n_in: usize,
    i_in: [usize; 3],
    c_in: [f64; 3],
    n_out: usize,
    i_out: [usize; 3],
    c_out: [f64; 3],
    idr: [usize; 3],
    dr: [f64; 3],
    q: f64,
    qneu: f64,
    dqneu_dt: f64,
    dqneu_drho: f64,
    symbolic: bool,
    just_dydt: bool,
) {
    let r = if r_in < R_MIN || w.rate_screened[i] < R_MIN {
        0.0
    } else {
        r_in
    };
    let icat = (g.reactions[i].category as usize)
        .saturating_sub(1)
        .min(NUM_CATEGORIES - 1);
    let d = w.rate_screened[i];
    let d1 = dr[0] * d;
    let d2 = dr[1] * d;
    let d3 = dr[2] * d;
    let rvs = [
        r * w.rate_screened[i],
        r * w.rate_screened_dt[i],
        r * w.rate_screened_drho[i],
    ];

    w.raw_rate[i] = w.rate_raw[i] * r * AVO;
    w.screened_rate[i] = w.rate_screened[i] * r * AVO;
    w.eps_nuc_rate[i] = w.rate_screened[i] * r * (q - qneu) * QCONV;
    w.eps_neu_rate[i] = w.rate_screened[i] * r * qneu * QCONV;

    for j in 0..n_in {
        do_signed_iso(
            g.num_isos,
            w,
            dydt,
            -c_in[j],
            i_in[j],
            &rvs,
            idr,
            [d1, d2, d3],
            symbolic,
            just_dydt,
        );
    }
    for j in 0..n_out {
        do_signed_iso(
            g.num_isos,
            w,
            dydt,
            c_out[j],
            i_out[j],
            &rvs,
            idr,
            [d1, d2, d3],
            symbolic,
            just_dydt,
        );
    }

    let q_net = q - qneu;
    eps_nuc_mev[0] += rvs[0] * q_net;
    // Accumulate left-to-right in `Quad` (`net_derivs_support.f90:174-175`).
    eps_nuc_mev[1] += rvs[1] * q_net;
    eps_nuc_mev[1] -= rvs[0] * dqneu_dt;
    eps_nuc_mev[2] += rvs[2] * q_net;
    eps_nuc_mev[2] -= rvs[0] * dqneu_drho;

    w.eps_nuc_categories[icat] += rvs[0] * q_net;
    w.eps_neu_total += rvs[0] * qneu;

    for (idx_k, dk) in idr.into_iter().zip([d1, d2, d3]) {
        if idx_k > 0 {
            w.d_eps_nuc_dy[idx_k - 1] += dk * q_net;
        }
    }
}

fn do_signed_iso(
    ni: usize,
    w: &mut NetWorkState,
    dydt: &mut [[Quad; NUM_RVS]],
    signed_c: f64,
    i1: usize,
    rvs: &[f64; NUM_RVS],
    idr: [usize; 3],
    dr: [f64; 3],
    symbolic: bool,
    just_dydt: bool,
) {
    let r1 = i1 - 1;
    if symbolic {
        for idx_k in idr {
            if idx_k > 0 {
                w.d_dydt_dy[r1 * ni + (idx_k - 1)] = 1.0;
            }
        }
        return;
    }
    for j in 0..NUM_RVS {
        dydt[r1][j] += signed_c * rvs[j];
    }
    if just_dydt {
        return;
    }
    for (idx_k, dk) in idr.into_iter().zip(dr) {
        if idx_k > 0 {
            w.d_dydt_dy[r1 * ni + (idx_k - 1)] += signed_c * dk;
        }
    }
}

/// Evaluates special composition-dependent reactions (`update_special_rates`
/// in `net/private/net_derivs.f90`).
pub fn update_special_rates(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[Quad; NUM_RVS]],
    eps_nuc_mev: &mut [Quad; NUM_RVS],
    i: usize,
    temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
    rate_factors: &[f64],
    use_3a_fl87: bool,
    symbolic: bool,
    just_dydt: bool,
) {
    let rxn = &g.reactions[i];
    if rxn.out_num[0] == 0 {
        return;
    }
    if rxn.name == "r_he4_he4_he4_to_c12" && use_3a_fl87 {
        let he4 = g.net_iso[IHE4];
        let c12 = g.net_iso[IC12];
        let ue = abar / zbar;
        let y_he4 = w.y[he4 - 1];
        let x_he4 = 4.0 * y_he4;
        let (r, dr1) = if y_he4 < 1.0e-50 {
            w.rate_screened[i] = 0.0;
            w.rate_screened_dt[i] = 0.0;
            w.rate_screened_drho[i] = 0.0;
            (0.0, 0.0)
        } else {
            let (fl_eps, dfl_dt, dfl_drho) =
                fl_epsnuc_3alf_with_derivs(temp, den, x_he4, ue);
            let conv = QCONV * rxn.std_q;
            let r_val = y_he4 * y_he4 * y_he4 / 6.0;
            let dr1_val = 0.5 * y_he4 * y_he4;
            let fac = rate_factors.get(i).copied().unwrap_or(1.0);
            w.rate_screened[i] = fl_eps / r_val * fac / conv;
            w.rate_screened_dt[i] = dfl_dt / r_val * fac / conv;
            w.rate_screened_drho[i] = dfl_drho / r_val * fac / conv;
            (r_val, dr1_val)
        };
        do_in_out(
            g,
            w,
            dydt,
            eps_nuc_mev,
            i,
            r,
            1,
            [he4, 0, 0],
            [3.0, 0.0, 0.0],
            1,
            [c12, 0, 0],
            [1.0, 0.0, 0.0],
            [he4, 0, 0],
            [dr1, 0.0, 0.0],
            symbolic,
            just_dydt,
        );
    } else if rxn.name == "rn14ag_lite" {
        let he4 = g.net_iso[IHE4];
        let n14 = g.net_iso[IN14];
        let ne20 = g.net_iso[INE20];
        let r = w.y[n14 - 1] * w.y[he4 - 1];
        let dr1 = w.y[n14 - 1];
        let dr2 = w.y[he4 - 1];
        do_in_out(
            g,
            w,
            dydt,
            eps_nuc_mev,
            i,
            r,
            2,
            [n14, he4, 0],
            [1.0, 1.5, 0.0],
            1,
            [ne20, 0, 0],
            [1.0, 0.0, 0.0],
            [he4, n14, 0],
            [dr1, dr2, 0.0],
            symbolic,
            just_dydt,
        );
    }
}

#[cfg(test)]
mod tests {
    use flexmesa_chem::NuclideCatalog;
    use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};

    use super::*;
    use crate::types::{RATTAB_TEMP_LO, RATTAB_TLO};

    #[test]
    fn test_weak_low_t_rate_matches_mesa_init_order() {
        let cat = RatesCatalog::builtin();
        let nuclides = NuclideCatalog::builtin_ref();
        let cid = |name: &str| nuclides.get_nuclide_index(name).unwrap() as u16;

        with_numerics_mode(NumericsMode::Strict, || {
            let be7_rate = weak_low_t_rate(
                cat,
                "r_be7_wk_li7",
                cid("be7"),
                cid("li7"),
                true,
            );
            assert_eq!(be7_rate.to_bits(), 0x3E84_34AE_CB5C_CB90);

            let pp_rate = weak_low_t_rate(
                cat,
                "r_h1_h1_wk_h2",
                cid("h1"),
                cid("h2"),
                false,
            );
            assert_eq!(pp_rate.to_bits(), (-1.0e-99_f64).to_bits());

            let b8_rate = weak_low_t_rate(
                cat,
                "r_b8_wk_he4_he4",
                cid("b8"),
                cid("he4"),
                false,
            );
            assert_eq!(b8_rate.to_bits(), (-1.0e-99_f64).to_bits());

            let n13_rate = weak_low_t_rate(
                cat,
                "r_n13_wk_c13",
                cid("n13"),
                cid("c13"),
                false,
            );
            assert_eq!(n13_rate.to_bits(), 0x3F52_FDAC_61EF_833C);

            let o15_rate = weak_low_t_rate(
                cat,
                "r_o15_wk_n15",
                cid("o15"),
                cid("n15"),
                false,
            );
            assert_eq!(o15_rate.to_bits(), 0x3F77_458B_4A11_2F36);

            let h1_neut_rate = weak_low_t_rate(
                cat,
                "r_h1_wk_neut",
                cid("h1"),
                cid("neut"),
                true,
            );
            assert_eq!(h1_neut_rate.to_bits(), (-1.0_f64).to_bits());

            let neut_h1_rate = weak_low_t_rate(
                cat,
                "r_neut_wk-minus_h1",
                cid("neut"),
                cid("h1"),
                true,
            );
            assert_eq!(neut_h1_rate.to_bits(), (-1.0_f64).to_bits());

            let mn54_fe54_rate = weak_low_t_rate(
                cat,
                "r_mn54_wk-minus_fe54",
                cid("mn54"),
                cid("fe54"),
                true,
            );
            assert_eq!(mn54_fe54_rate.to_bits(), (-1.0_f64).to_bits());

            assert_eq!(RATTAB_TEMP_LO.to_bits(), 0x4108_69FF_FFFF_FFE9);
            assert_eq!(RATTAB_TEMP_LO.to_bits(), exp10(RATTAB_TLO).to_bits());
        });
    }
}
