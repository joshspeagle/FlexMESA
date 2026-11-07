// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reaction-network derivative accumulation helpers, weaklib low-$T$ blending,
//! and special reaction updates (`net_derivs_support.f90` and `net_eval.f90`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AVO, LN2, QCONV};
use flexmesa_rates::{
    eval_ecapture_pair, eval_fl_epsnuc_3alf, eval_reaclib_terms,
    eval_weak_reaction_pair, CoulombMuiOption, CoulombVsOption, RatesCatalog,
    WeakRateEval, DEFAULT_REACLIB_MIN_T9,
};

use crate::f64x2::F64x2;
use crate::types::{
    NetGeneralInfo, IC12, IHE4, IN14, INE20, NUM_CATEGORIES, NUM_RVS,
};

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

/// Evaluates `FL_epsnuc_3alf(T, Rho, Y, UE)` and its `(T, Rho)` partials.
#[must_use]
pub fn fl_epsnuc_3alf_with_derivs(
    t: f64,
    rho: f64,
    y: f64,
    ue: f64,
) -> (f64, f64, f64) {
    let eps = eval_fl_epsnuc_3alf(t, rho, y, ue);
    let dt = 1.0e-6 * t.max(1.0);
    let dr = 1.0e-6 * rho.max(1.0e-30);
    let deps_dt = (eval_fl_epsnuc_3alf(t + dt, rho, y, ue)
        - eval_fl_epsnuc_3alf(t - dt, rho, y, ue))
        / (2.0 * dt);
    let deps_drho = (eval_fl_epsnuc_3alf(t, rho + dr, y, ue)
        - eval_fl_epsnuc_3alf(t, rho - dr, y, ue))
        / (2.0 * dr);
    (eps, deps_dt, deps_drho)
}

const R_MIN: f64 = 1.0e-99;
const T9_WEAKLIB_FULL_OFF: f64 = 0.01;
const T9_WEAKLIB_FULL_ON: f64 = 0.02;
const WEAKLIB_BLEND_HI_Z: usize = 26;
const T9_WEAKLIB_FULL_OFF_HI_Z: f64 = 0.063;
const T9_WEAKLIB_FULL_ON_HI_Z: f64 = 0.073;

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
    pub eps_nuc_categories: [f64; NUM_CATEGORIES],
    /// Total neutrino loss rate in $\mathrm{MeV\,g^{-1}\,s^{-1}}$ (scaled by `QCONV` on export).
    pub eps_neu_total: f64,
    /// $\partial \epsilon_{\mathrm{nuc,MeV}} / \partial Y_i$ (`num_isos`).
    pub d_eps_nuc_dy: Vec<f64>,
    /// Molar Jacobian $\partial (dY_i/dt) / \partial Y_j$ (`num_isos * num_isos`, row-major).
    pub d_dydt_dy: Vec<f64>,
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
    /// Allocates zero-initialized buffers for `g`.
    #[must_use]
    pub fn new(g: &NetGeneralInfo) -> Self {
        let ni = g.num_isos;
        let nr = g.num_reactions.max(120);
        let nw = g.num_wk_reactions;
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
            eps_nuc_categories: [0.0; NUM_CATEGORIES],
            eps_neu_total: 0.0,
            d_eps_nuc_dy: vec![0.0; ni],
            d_dydt_dy: vec![0.0; ni * ni],
            wk_lambda: vec![0.0; nw],
            wk_dlambda_dln_t: vec![0.0; nw],
            wk_dlambda_dln_rho: vec![0.0; nw],
            wk_q: vec![0.0; nw],
            wk_qneu: vec![0.0; nw],
            wk_dqneu_dln_t: vec![0.0; nw],
            wk_dqneu_dln_rho: vec![0.0; nw],
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
            ev.dlambda_dln_t =
                alfa * ev.dlambda_dln_t + d_alfa_dln_t * (raw_lam - low_t_rate);
            ev.dlambda_dln_rho *= alfa;
        }
    }

    let _ = (temp, z2bar, d_eta_dln_t, d_eta_dln_rho);
    if do_ecapture
        && cat.special_ecapture_data(&e_lhs.name, &e_rhs.name).is_ok()
    {
        if let Ok(ec) = eval_ecapture_pair(
            &e_lhs.name,
            &e_rhs.name,
            t9,
            ye_rho,
            eta,
            CoulombMuiOption::Pcr2009,
            CoulombVsOption::Itoh2002,
            zbar,
            abar,
        ) {
            ev = ec;
        }
    }

    ev
}

fn weak_low_t_rate(
    cat: &RatesCatalog,
    rxn_name: &str,
    lhs_id: u16,
    rhs_id: u16,
    has_weaklib: bool,
) -> f64 {
    if let Ok((is_rev, terms)) = cat.reaclib_terms(rxn_name) {
        if !terms.is_empty() {
            let slice = if has_weaklib {
                &terms[terms.len() - 1..]
            } else {
                terms
            };
            let (fw, _, rv, _) =
                eval_reaclib_terms(slice, 0.01, DEFAULT_REACLIB_MIN_T9, false);
            return if is_rev { rv } else { fw };
        }
    }
    if let Some((half_life, _)) = cat.weak_info_by_ids(lhs_id, rhs_id) {
        if half_life > 0.0 {
            return LN2 / half_life;
        }
    }
    if has_weaklib {
        return -1.0;
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
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
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
    let qneu = g.reactions[i].std_qneu;
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
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
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
    eps_nuc_mev[0].add_assign_f64(rvs[0] * q_net);
    eps_nuc_mev[1].add_assign_f64(rvs[1] * q_net - rvs[0] * dqneu_dt);
    eps_nuc_mev[2].add_assign_f64(rvs[2] * q_net - rvs[0] * dqneu_drho);

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
    dydt: &mut [[F64x2; NUM_RVS]],
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
        dydt[r1][j].add_assign_f64(signed_c * rvs[j]);
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
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
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
