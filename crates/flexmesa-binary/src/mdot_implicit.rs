// SPDX-License-Identifier: LGPL-3.0-or-later
//! Implicit Roche-lobe overflow (`"roche_lobe"`, `"Ritter"`, `"Kolb"`, and
//! Marchant et al. 2016 `"contact"`) bracketing and quadratic/secant/bisection
//! root-finding solvers (`check_implicit_rlo` and `pick_mdot_for_implicit_rlo`
//! in `binary/private/binary_mdot.f90:36-425`).

use flexmesa_const::{MSUN, SECYER};
use flexmesa_math::pow;

use crate::inlist::BinaryInlistControls;

#[inline]
fn find0_quadratic(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
) -> Option<f64> {
    let d12 = x1 - x2;
    let d23 = x2 - x3;
    let d13 = x1 - x3;
    if d12.abs() < 1.0e-40 || d23.abs() < 1.0e-40 || d13.abs() < 1.0e-40 {
        return None;
    }
    let s12 = (y1 - y2) / d12;
    let s23 = (y2 - y3) / d23;
    let a = (s12 - s23) / d13;
    let b = s12 - a * (x1 + x2);
    let c = y1 - a * x1 * x1 - b * x1;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1.0e-40 {
        return None;
    }
    let root_sqrt = disc.sqrt();
    let r1 = (-b + root_sqrt) / (2.0 * a);
    let r2 = (-b - root_sqrt) / (2.0 * a);
    let lo = x1.min(x2).min(x3);
    let hi = x1.max(x2).max(x3);
    if r1 > lo && r1 < hi {
        Some(r1)
    } else if r2 > lo && r2 < hi {
        Some(r2)
    } else {
        None
    }
}

/// Bracketing and root-finding helper for callable-based implicit RLOF solves.
#[derive(Debug, Clone)]
pub struct ImplicitRlofSolver {
    /// Target tolerance $\xi$ on $|f(\dot{M}_{\mathrm{tr}})|$ (`implicit_scheme_tolerance`).
    pub tolerance: f64,
    /// Step multiplication factor (`change_factor`, in `[1.05, 1.50]`).
    pub change_factor: f64,
    /// Maximum allowed implicit $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub max_implicit_abs_mdot_msun_yr: f64,
    /// Minimum $|\dot{M}_{\mathrm{tr}}|$ before detachment ($M_\odot\,\mathrm{yr}^{-1}$).
    pub roche_min_mdot_msun_yr: f64,
    /// Starting trial $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub starting_mdot_msun_yr: f64,
    /// Whether to use 3-point quadratic interpolation when bracketed (`use_quadratic`).
    pub use_quadratic: bool,
}

impl Default for ImplicitRlofSolver {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-4,
            change_factor: 1.25,
            max_implicit_abs_mdot_msun_yr: 1.0e-1,
            roche_min_mdot_msun_yr: 1.0e-14,
            starting_mdot_msun_yr: 1.0e-12,
            use_quadratic: true,
        }
    }
}

impl ImplicitRlofSolver {
    /// Evaluates the `"roche_lobe"` implicit objective $2(g_d + \tfrac{1}{2}\xi)$
    /// (`binary_mdot.f90:102-112`), which satisfies $|f| \le \xi \iff |(2g_d + \xi)/\xi| \le 1$.
    #[must_use]
    pub fn roche_lobe_objective(&self, rl_relative_gap_donor: f64) -> f64 {
        2.0 * (rl_relative_gap_donor + 0.5 * self.tolerance)
    }

    /// Evaluates the Marchant et al. (2016) `"contact"` equipotential objective
    /// (`check_implicit_rlo` in `binary_mdot.f90:113-147`) with $s_c = q^{-0.52}$,
    /// falling back to `"roche_lobe"` when the companion is detached.
    #[must_use]
    pub fn contact_objective(
        &self,
        rl_relative_gap_donor: f64,
        rl_relative_gap_accretor: f64,
        q_accretor_over_donor: f64,
    ) -> f64 {
        let q = q_accretor_over_donor.max(1.0e-12);
        let slope_contact = pow(q, -0.52);
        if rl_relative_gap_accretor < 0.0
            && (rl_relative_gap_donor * slope_contact - rl_relative_gap_accretor
                > self.tolerance
                || rl_relative_gap_donor < -self.tolerance)
        {
            self.roche_lobe_objective(rl_relative_gap_donor)
        } else if q < 1.0 {
            rl_relative_gap_donor * slope_contact - rl_relative_gap_accretor
        } else {
            rl_relative_gap_donor - rl_relative_gap_accretor / slope_contact
        }
    }

    /// Solves `"roche_lobe"` implicit RLOF for callable `eval_gap(mdot_g_s)`.
    pub fn solve_roche_lobe_mdot<F>(
        &self,
        initial_mdot_g_s: f64,
        max_tries: usize,
        mut eval_gap: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> f64,
    {
        self.solve_with_objective(initial_mdot_g_s, max_tries, |mdot| {
            self.roche_lobe_objective(eval_gap(mdot))
        })
    }

    /// Solves `"contact"` implicit RLOF for callable `eval_gaps(mdot_g_s)`.
    pub fn solve_contact_mdot<F>(
        &self,
        initial_mdot_g_s: f64,
        q_accretor_over_donor: f64,
        max_tries: usize,
        mut eval_gaps: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> (f64, f64),
    {
        self.solve_with_objective(initial_mdot_g_s, max_tries, |mdot| {
            let (gap_d, gap_a) = eval_gaps(mdot);
            self.contact_objective(gap_d, gap_a, q_accretor_over_donor)
        })
    }

    fn solve_with_objective<F>(
        &self,
        initial_mdot_g_s: f64,
        max_tries: usize,
        mut eval_f: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> f64,
    {
        let mut state = ImplicitMdotSolver::new(
            self.tolerance,
            1.0e-6,
            self.change_factor,
            self.max_implicit_abs_mdot_msun_yr,
            self.roche_min_mdot_msun_yr,
            1.0e-16,
            self.starting_mdot_msun_yr,
            self.use_quadratic,
            false,
        );
        let max_abs = self.max_implicit_abs_mdot_msun_yr * (MSUN / SECYER);
        let mut mdot = initial_mdot_g_s.min(0.0).max(-max_abs);
        let limit = max_tries.max(1);
        for iter in 1..=limit {
            let f = eval_f(mdot);
            match state.record_and_propose(iter, limit, mdot, f, false) {
                ImplicitStepAction::Converged(m_ok) => return (m_ok, iter, f),
                ImplicitStepAction::Detached => {
                    let f0 = eval_f(0.0);
                    return (0.0, iter, f0);
                }
                ImplicitStepAction::TryNextMdot(next_m) => {
                    mdot = next_m;
                }
                ImplicitStepAction::ExceededMaxTries => {
                    return (mdot, iter, f);
                }
            }
        }
        let final_f = eval_f(mdot);
        (mdot, limit, final_f)
    }
}

/// Outcome of a single implicit RLOF step evaluation (`check_implicit_rlo` in
/// `binary/private/binary_mdot.f90:90-275`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplicitStepAction {
    /// Post-step state converged at the given `mtransfer_rate` ($\mathrm{g\,s^{-1}}$).
    Converged(f64),
    /// Donor is detached; re-run with `mtransfer_rate = 0` (`try_detach`) and accept.
    Detached,
    /// Restore `.photo` + orbital snapshot and re-run with new trial `mtransfer_rate`
    /// ($\mathrm{g\,s^{-1}}$; may be `> 0` in `"contact"` to trigger a mid-step donor switch).
    TryNextMdot(f64),
    /// Reached `max_tries_to_achieve` without convergence; restore snapshot, cut $\Delta t$, and retry.
    ExceededMaxTries,
}

/// Stateful per-step implicit mass-transfer root finder matching `check_implicit_rlo`
/// and `pick_mdot_for_implicit_rlo` (`binary/private/binary_mdot.f90:55-425`).
#[derive(Debug, Clone)]
pub struct ImplicitMdotSolver {
    /// Objective tolerance $\xi$ (`implicit_scheme_tolerance`).
    pub tolerance: f64,
    /// Relative bracket width exit factor (`implicit_scheme_tiny_factor`).
    pub tiny_factor: f64,
    /// Multiplicative search step (`change_factor`).
    pub change_factor: f64,
    /// Maximum $|\dot{M}_{\mathrm{tr}}|$ in $\mathrm{g\,s^{-1}}$ (`max_implicit_abs_mdot`).
    pub max_abs_mdot_cgs: f64,
    /// Detachment floor in $\mathrm{g\,s^{-1}}$ (`roche_min_mdot`).
    pub roche_min_mdot_cgs: f64,
    /// Explicit-rate floor in $\mathrm{g\,s^{-1}}$ (`min_mdot_for_implicit`).
    pub min_mdot_for_implicit_cgs: f64,
    /// Initial trial $|\dot{M}_{\mathrm{tr}}|$ in $\mathrm{g\,s^{-1}}$ (`starting_mdot`).
    pub starting_mdot_cgs: f64,
    /// Enable 3-point quadratic interpolation (`solver_type != "bisect"`).
    pub use_quadratic: bool,
    /// Enable additive `use_sum` search across $\dot{M} = 0$ (`mdot_scheme == "contact"`).
    pub use_sum: bool,
    have_lo: bool,
    have_hi: bool,
    have_prev: bool,
    try_detach: bool,
    mdot_lo: f64,
    mdot_hi: f64,
    mdot_prev: f64,
    f_lo: f64,
    f_hi: f64,
    f_prev: f64,
}

impl ImplicitMdotSolver {
    /// Constructs a new per-step implicit solver from [`BinaryInlistControls`].
    #[must_use]
    pub fn from_controls(ctrl: &BinaryInlistControls, change_factor: f64) -> Self {
        let use_quad = !ctrl.solver_type.eq_ignore_ascii_case("bisect");
        let is_contact = ctrl.mdot_scheme.eq_ignore_ascii_case("contact");
        Self::new(
            ctrl.implicit_scheme_tolerance,
            ctrl.implicit_scheme_tiny_factor,
            change_factor,
            ctrl.max_implicit_abs_mdot,
            ctrl.roche_min_mdot,
            ctrl.min_mdot_for_implicit,
            ctrl.starting_mdot,
            use_quad,
            is_contact,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        tolerance: f64,
        tiny_factor: f64,
        change_factor: f64,
        max_abs_msun_yr: f64,
        roche_min_msun_yr: f64,
        min_imp_msun_yr: f64,
        starting_msun_yr: f64,
        use_quadratic: bool,
        use_sum: bool,
    ) -> Self {
        let u = MSUN / SECYER;
        Self {
            tolerance: tolerance.max(1.0e-12),
            tiny_factor: tiny_factor.max(1.0e-14),
            change_factor: change_factor.max(1.01),
            max_abs_mdot_cgs: max_abs_msun_yr * u,
            roche_min_mdot_cgs: roche_min_msun_yr * u,
            min_mdot_for_implicit_cgs: min_imp_msun_yr * u,
            starting_mdot_cgs: starting_msun_yr * u,
            use_quadratic,
            use_sum,
            have_lo: false,
            have_hi: false,
            have_prev: false,
            try_detach: false,
            mdot_lo: 0.0,
            mdot_hi: 0.0,
            mdot_prev: 0.0,
            f_lo: 0.0,
            f_hi: 0.0,
            f_prev: 0.0,
        }
    }

    /// Evaluates the implicit objective and checks scheme-specific early-exit criteria
    /// (`binary_mdot.f90:102-215`) for `"roche_lobe"`, `"contact"`, `"Ritter"`, or `"Kolb"`.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_trial(
        &mut self,
        scheme: &str,
        num_tries: usize,
        max_tries: usize,
        mdot_current: f64,
        explicit_mdot: f64,
        gap_donor: f64,
        gap_accretor: f64,
        q_accretor_over_donor: f64,
    ) -> ImplicitStepAction {
        let helper = ImplicitRlofSolver {
            tolerance: self.tolerance,
            ..ImplicitRlofSolver::default()
        };
        let is_roche = scheme.eq_ignore_ascii_case("roche_lobe");
        let is_contact = scheme.eq_ignore_ascii_case("contact");

        if !is_roche && !is_contact {
            if explicit_mdot > 0.0 {
                return ImplicitStepAction::Converged(0.0);
            }
            if explicit_mdot.abs() <= self.min_mdot_for_implicit_cgs
                && mdot_current.abs() <= self.min_mdot_for_implicit_cgs
            {
                return ImplicitStepAction::Converged(mdot_current);
            }
            let f = if explicit_mdot != 0.0 {
                (explicit_mdot - mdot_current) / explicit_mdot
            } else if mdot_current != 0.0 {
                (explicit_mdot - mdot_current) / mdot_current
            } else {
                0.0
            };
            if num_tries == 1
                && mdot_current == 0.0
                && explicit_mdot.abs() > self.min_mdot_for_implicit_cgs
            {
                let seed = explicit_mdot.max(-self.max_abs_mdot_cgs);
                self.have_lo = true;
                self.mdot_lo = 0.0;
                self.f_lo = f;
                return if max_tries <= 1 {
                    ImplicitStepAction::ExceededMaxTries
                } else {
                    ImplicitStepAction::TryNextMdot(seed)
                };
            }
            return self.record_and_propose(
                num_tries,
                max_tries,
                mdot_current,
                f,
                false,
            );
        }

        let (f, detach_allowed) = if is_contact {
            let q = q_accretor_over_donor.max(1.0e-12);
            let sc = pow(q, -0.52);
            let companion_detached = gap_accretor < 0.0
                && (gap_donor * sc - gap_accretor > self.tolerance
                    || gap_donor < -self.tolerance);
            if companion_detached {
                (helper.roche_lobe_objective(gap_donor), true)
            } else {
                (
                    helper.contact_objective(gap_donor, gap_accretor, q),
                    gap_donor < 0.0 && gap_accretor < 0.0,
                )
            }
        } else {
            (helper.roche_lobe_objective(gap_donor), true)
        };
        self.record_and_propose(
            num_tries,
            max_tries,
            mdot_current,
            f,
            detach_allowed,
        )
    }

    /// Flip sign of bracket endpoints after a mid-step `"contact"` donor switch
    /// (`binary_evolve.f90:222-233`).
    pub fn flip_for_contact_donor_switch(&mut self) {
        if self.have_lo && self.have_hi {
            std::mem::swap(&mut self.mdot_lo, &mut self.mdot_hi);
            std::mem::swap(&mut self.f_lo, &mut self.f_hi);
            self.mdot_lo = -self.mdot_lo;
            self.mdot_hi = -self.mdot_hi;
            self.f_lo = -self.f_lo;
            self.f_hi = -self.f_hi;
        } else if self.have_lo {
            self.have_hi = true;
            self.have_lo = false;
            self.mdot_hi = -self.mdot_lo;
            self.f_hi = -self.f_lo;
        } else if self.have_hi {
            self.have_lo = true;
            self.have_hi = false;
            self.mdot_lo = -self.mdot_hi;
            self.f_lo = -self.f_hi;
        }
        self.have_prev = false;
    }

    fn record_and_propose(
        &mut self,
        num_tries: usize,
        max_tries: usize,
        mdot: f64,
        f: f64,
        detach_allowed: bool,
    ) -> ImplicitStepAction {
        if (f < 0.0 && mdot == 0.0 && (!self.use_sum || detach_allowed))
            || f.abs() <= self.tolerance
            || (f > 0.0
                && (-mdot - self.max_abs_mdot_cgs).abs()
                    <= 1.0e-12 * self.max_abs_mdot_cgs)
            || self.try_detach
        {
            return ImplicitStepAction::Converged(mdot);
        }
        if num_tries >= max_tries {
            return ImplicitStepAction::ExceededMaxTries;
        }

        if f < 0.0 {
            if self.have_hi {
                self.mdot_prev = self.mdot_hi;
                self.f_prev = self.f_hi;
                self.have_prev = true;
            }
            self.have_hi = true;
            self.mdot_hi = mdot;
            self.f_hi = f;
        } else {
            if self.have_lo {
                self.mdot_prev = self.mdot_lo;
                self.f_prev = self.f_lo;
                self.have_prev = true;
            }
            self.have_lo = true;
            self.mdot_lo = mdot;
            self.f_lo = f;
        }

        let raw_next = if self.have_lo && self.have_hi {
            if (self.mdot_hi - self.mdot_lo).abs()
                < self.tiny_factor
                    * self.mdot_hi.abs().min(self.mdot_lo.abs()).max(1.0)
            {
                return ImplicitStepAction::Converged(
                    0.5 * (self.mdot_lo + self.mdot_hi),
                );
            }
            let quad = if self.use_quadratic
                && self.have_prev
                && num_tries.is_multiple_of(2)
            {
                find0_quadratic(
                    self.mdot_lo,
                    self.f_lo,
                    self.mdot_hi,
                    self.f_hi,
                    self.mdot_prev,
                    self.f_prev,
                )
            } else {
                None
            };
            if let Some(q_root) = quad {
                q_root
            } else if num_tries.is_multiple_of(2)
                && (self.f_hi - self.f_lo).abs() > 1.0e-30
            {
                let sec = (self.mdot_lo * self.f_hi - self.mdot_hi * self.f_lo)
                    / (self.f_hi - self.f_lo);
                if sec > self.mdot_hi && sec < self.mdot_lo {
                    sec
                } else {
                    0.5 * (self.mdot_lo + self.mdot_hi)
                }
            } else {
                0.5 * (self.mdot_lo + self.mdot_hi)
            }
        } else if !self.have_hi {
            if self.mdot_lo.abs() < self.starting_mdot_cgs {
                -self.starting_mdot_cgs
            } else if self.use_sum {
                self.mdot_lo - 2.0 * (self.change_factor - 1.0) * self.mdot_lo.abs()
            } else {
                self.mdot_lo * self.change_factor
            }
        } else if self.use_sum {
            let step = (self.change_factor - 1.0)
                * self.mdot_hi.abs().max(self.starting_mdot_cgs);
            self.mdot_hi + step
        } else {
            self.mdot_hi / self.change_factor
        };

        let next_mdot = raw_next.max(-self.max_abs_mdot_cgs);
        if detach_allowed && -next_mdot < self.roche_min_mdot_cgs && f < 0.0 {
            self.try_detach = true;
            return ImplicitStepAction::Detached;
        }
        ImplicitStepAction::TryNextMdot(next_mdot)
    }
}

/// Updates `b%change_factor` after an accepted implicit step (`binary_evolve.f90:506-519`).
#[must_use]
pub fn update_change_factor_after_step(
    old_change_factor: f64,
    mtransfer_rate_old: f64,
    mtransfer_rate_new: f64,
    num_tries: usize,
    ctrl: &BinaryInlistControls,
) -> f64 {
    if mtransfer_rate_new == 0.0 || mtransfer_rate_old == 0.0 {
        return old_change_factor;
    }
    let lam = ctrl.implicit_lambda;
    let frac = ctrl.change_factor_fraction;
    let next = if num_tries <= 1 {
        (1.0 - lam) * old_change_factor + lam * (1.0 + (old_change_factor - 1.0) / frac)
    } else {
        let ratio = if mtransfer_rate_new.abs() > mtransfer_rate_old.abs() {
            mtransfer_rate_new / mtransfer_rate_old
        } else {
            mtransfer_rate_old / mtransfer_rate_new
        };
        let target = frac * ratio + (1.0 - frac);
        (1.0 - lam) * old_change_factor + lam * target
    };
    next.clamp(ctrl.min_change_factor, ctrl.max_change_factor)
}
