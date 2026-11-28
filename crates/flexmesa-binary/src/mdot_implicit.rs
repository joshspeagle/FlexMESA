// SPDX-License-Identifier: GPL-3.0-or-later
//! Implicit Roche-lobe overflow (`"roche_lobe"`, `"Ritter"`, `"Kolb"`, and
//! Marchant et al. 2016 `"contact"`) bracketing and quadratic/secant/bisection
//! root-finding solvers (`check_implicit_rlo` and `pick_mdot_for_implicit_rlo`
//! in `binary/private/binary_mdot.f90:36-425`).

use flexmesa_const::{MSUN, SECYER};
use flexmesa_math::{pow, pow2};
use flexmesa_star::StarError;

use crate::inlist::BinaryInlistControls;

/// Linear secant root finder matching `find0` in `num/public/num_lib.f90:517-534`.
#[inline]
#[must_use]
pub(crate) fn mesa_find0(xx1: f64, yy1: f64, xx2: f64, yy2: f64) -> f64 {
    let a = (xx1 * yy2) - (xx2 * yy1);
    let b = yy2 - yy1;
    let mut x = if b == 0.0
        || (a.abs() >= b.abs() * 1.0e99
            && ((yy1 >= 0.0 && yy2 <= 0.0) || (yy1 <= 0.0 && yy2 >= 0.0)))
    {
        0.5 * (xx1 + xx2)
    } else {
        a / b
    };
    if yy1 * yy2 <= 0.0 {
        let hi = xx1.max(xx2);
        let lo = xx1.min(xx2);
        if x > hi {
            x = hi;
        }
        if x < lo {
            x = lo;
        }
    }
    x
}

/// Three-point quadratic root finder matching `find0_quadratic` in
/// `num/public/num_lib.f90:536-556` without dimensional CGS slope cutoffs.
#[inline]
pub(crate) fn mesa_find0_quadratic(
    xx1: f64,
    yy1: f64,
    xx2: f64,
    yy2: f64,
    xx3: f64,
    yy3: f64,
) -> Option<f64> {
    let xx1_2 = pow2(xx1);
    let xx2_2 = pow2(xx2);
    let xx3_2 = pow2(xx3);
    let term1 =
        xx3_2 * (-yy1 + yy2) + xx2_2 * (yy1 - yy3) + xx1_2 * (-yy2 + yy3);
    let lin = xx3 * (-yy1 + yy2) + xx2 * (yy1 - yy3) + xx1 * (-yy2 + yy3);
    let c_term = xx1 * xx3 * (-xx1 + xx3) * yy2
        + xx2_2 * (xx3 * yy1 - xx1 * yy3)
        + xx2 * (-xx3_2 * yy1 + xx1_2 * yy3);
    let s2 = pow2(term1) - 4.0 * lin * c_term;
    if !s2.is_finite() || s2 < 0.0 {
        return None;
    }
    let b = s2.sqrt();
    let a = xx3_2 * (yy1 - yy2) + xx1_2 * (yy2 - yy3) + xx2_2 * (-yy1 + yy3);
    let denom =
        2.0 * (xx3 * (yy1 - yy2) + xx1 * (yy2 - yy3) + xx2 * (-yy1 + yy3));
    if denom == 0.0 {
        return None;
    }
    let hi = xx1.max(xx2).max(xx3);
    let lo = xx1.min(xx2).min(xx3);
    let mut x = (a + b) / denom;
    if x > hi {
        x = (a - b) / denom;
    }
    if !x.is_finite() || x < lo || x > hi {
        None
    } else {
        Some(x)
    }
}

/// Root-finding mode corresponding to `&binary_controls::solver_type`
/// (`binary/private/binary_mdot.f90:342-354`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplicitSolverMode {
    /// Use `find0_quadratic` (with `find0` fallback) on every bracketed try (`"cubic"`).
    Cubic,
    /// Use midpoint bisection on every bracketed try (`"bisect"`).
    Bisect,
    /// Alternate `find0_quadratic` on even tries and bisection on odd tries (`"both"`).
    Both,
}

impl ImplicitSolverMode {
    /// Parses `&binary_controls::solver_type` (`"cubic"`, `"bisect"`, or `"both"`).
    pub fn parse(raw: &str) -> Result<Self, StarError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cubic" => Ok(Self::Cubic),
            "bisect" => Ok(Self::Bisect),
            "both" => Ok(Self::Both),
            _ => Err(StarError::InvalidInput(format!(
                "unrecognized binary solver_type '{raw}' (expected 'cubic', 'bisect', or 'both')"
            ))),
        }
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
    /// (`binary_mdot.f90:102-112`), which satisfies $|f| \le \xi \iff -\xi \le g_d \le 0$.
    #[must_use]
    pub fn roche_lobe_objective(&self, rl_relative_gap_donor: f64) -> f64 {
        (rl_relative_gap_donor + 0.5 * self.tolerance) * 2.0
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
        let q = q_accretor_over_donor;
        let slope_contact = pow(q, -0.52);
        if rl_relative_gap_accretor < 0.0
            && (rl_relative_gap_donor * slope_contact
                - rl_relative_gap_accretor
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
            (self.roche_lobe_objective(eval_gap(mdot)), false, true)
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
        let slope_contact = pow(q_accretor_over_donor, -0.52);
        self.solve_with_objective(initial_mdot_g_s, max_tries, |mdot| {
            let (gap_d, gap_a) = eval_gaps(mdot);
            let companion_detached = gap_a < 0.0
                && (gap_d * slope_contact - gap_a > self.tolerance
                    || gap_d < -self.tolerance);
            if companion_detached {
                (self.roche_lobe_objective(gap_d), false, true)
            } else {
                (
                    self.contact_objective(gap_d, gap_a, q_accretor_over_donor),
                    true,
                    false,
                )
            }
        })
    }

    fn solve_with_objective<F>(
        &self,
        initial_mdot_g_s: f64,
        max_tries: usize,
        mut eval_spec: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> (f64, bool, bool),
    {
        let mode = if self.use_quadratic {
            ImplicitSolverMode::Both
        } else {
            ImplicitSolverMode::Bisect
        };
        let mut state = ImplicitMdotSolver::new(
            self.tolerance,
            1.0e-6,
            self.change_factor,
            1.5,
            20,
            1.1,
            self.max_implicit_abs_mdot_msun_yr,
            self.roche_min_mdot_msun_yr,
            1.0e-16,
            self.starting_mdot_msun_yr,
            mode,
            false,
            false,
        );
        let mut mdot = initial_mdot_g_s;
        let limit = max_tries;
        for iter in 1..=(limit + 1) {
            let (f, trial_use_sum, detach_allowed) = eval_spec(mdot);
            match state.record_and_propose(
                iter,
                limit,
                mdot,
                f,
                trial_use_sum,
                detach_allowed,
            ) {
                ImplicitStepAction::Converged(m_ok) => return (m_ok, iter, f),
                ImplicitStepAction::Detached => return (0.0, iter, f),
                ImplicitStepAction::TryNextMdot(next_m) => {
                    mdot = next_m;
                }
                ImplicitStepAction::ExceededMaxTries => {
                    return (mdot, iter, f);
                }
            }
        }
        let (final_f, _, _) = eval_spec(mdot);
        (mdot, limit + 1, final_f)
    }
}

/// Outcome of a single implicit RLOF step evaluation (`check_implicit_rlo` in
/// `binary/private/binary_mdot.f90:36-322`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplicitStepAction {
    /// Post-step state converged (`keep_going`) with new `mtransfer_rate` ($\mathrm{g\,s^{-1}}$).
    Converged(f64),
    /// Donor detached (`keep_going` with `new_mdot = 0`, `binary_mdot.f90:106-112, 252-262`).
    Detached,
    /// Restore `.photo` + orbital snapshot and re-run (`redo`) with new trial `mtransfer_rate`
    /// ($\mathrm{g\,s^{-1}}$; may be `> 0` in `"contact"` to trigger a mid-step donor switch).
    TryNextMdot(f64),
    /// Exceeded `max_tries_to_achieve` (`num_tries > max_tries_to_achieve`); restore snapshot and retry.
    ExceededMaxTries,
}

/// Diagnostic context passed when `report_rlo_solver_progress = true`
/// (`report_rlo_iter` in `binary/private/binary_mdot.f90:295-320`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RloIterReportContext {
    /// Binary `model_number` (`b% model_number`).
    pub model_number: usize,
    /// Active donor index (`1` or `2`).
    pub donor_i: usize,
    /// Relative Roche-lobe gaps `[rl_relative_gap(1), rl_relative_gap(2)]` ordered by star index.
    pub rl_gaps_by_star: [f64; 2],
    /// Point-mass component index (`0`, `1`, or `2`).
    pub point_mass_i: usize,
}

/// Stateful per-step implicit mass-transfer root finder matching `check_implicit_rlo`
/// and `pick_mdot_for_implicit_rlo` (`binary/private/binary_mdot.f90:36-425`).
#[derive(Debug, Clone)]
pub struct ImplicitMdotSolver {
    /// Objective tolerance $\xi$ (`implicit_scheme_tolerance`).
    pub tolerance: f64,
    /// Relative bracket width exit factor (`implicit_scheme_tiny_factor`).
    pub tiny_factor: f64,
    /// Multiplicative search step (`change_factor`).
    pub change_factor: f64,
    /// Upper clamp on `change_factor` (`max_change_factor`).
    pub max_change_factor: f64,
    /// Unbracketed try interval between `change_factor` boosts (`num_tries_for_increase_change_factor`).
    pub num_tries_for_increase_change_factor: i64,
    /// Multiplicative boost applied to `change_factor` (`change_factor_increase`).
    pub change_factor_increase: f64,
    /// Maximum $|\dot{M}_{\mathrm{tr}}|$ in $\mathrm{g\,s^{-1}}$ (`max_implicit_abs_mdot`).
    pub max_abs_mdot_cgs: f64,
    /// Detachment floor in $\mathrm{g\,s^{-1}}$ (`roche_min_mdot`).
    pub roche_min_mdot_cgs: f64,
    /// Explicit-rate floor in $\mathrm{g\,s^{-1}}$ (`min_mdot_for_implicit`).
    pub min_mdot_for_implicit_cgs: f64,
    /// Initial trial $|\dot{M}_{\mathrm{tr}}|$ in $\mathrm{g\,s^{-1}}$ (`starting_mdot`).
    pub starting_mdot_cgs: f64,
    /// Root-finding mode (`"cubic"`, `"bisect"`, `"both"`).
    pub solver_mode: ImplicitSolverMode,
    /// Enable 3-point quadratic interpolation (`solver_type != "bisect"`).
    pub use_quadratic: bool,
    /// Enable additive `use_sum` search across $\dot{M} = 0$ (`mdot_scheme == "contact"`).
    pub use_sum: bool,
    /// Print iteration progress (`report_rlo_solver_progress`).
    pub report_progress: bool,
    /// Constant additive step size $\Delta\dot{M} = \dot{M}_{\mathrm{tr,0}}(1 - f_c)$ (`fixed_delta_mdot`).
    pub fixed_delta_mdot: f64,
    have_lo: bool,
    have_hi: bool,
    mdot_lo: f64,
    mdot_hi: f64,
    f_lo: f64,
    f_hi: f64,
}

impl ImplicitMdotSolver {
    /// Constructs a new per-step implicit solver from [`BinaryInlistControls`].
    pub fn from_controls(
        ctrl: &BinaryInlistControls,
        change_factor: f64,
    ) -> Result<Self, StarError> {
        let solver_mode = ImplicitSolverMode::parse(&ctrl.solver_type)?;
        let is_contact = ctrl.mdot_scheme.eq_ignore_ascii_case("contact");
        Ok(Self::new(
            ctrl.implicit_scheme_tolerance,
            ctrl.implicit_scheme_tiny_factor,
            change_factor,
            ctrl.max_change_factor,
            ctrl.num_tries_for_increase_change_factor,
            ctrl.change_factor_increase,
            ctrl.max_implicit_abs_mdot,
            ctrl.roche_min_mdot,
            ctrl.min_mdot_for_implicit,
            ctrl.starting_mdot,
            solver_mode,
            is_contact,
            ctrl.report_rlo_solver_progress,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        tolerance: f64,
        tiny_factor: f64,
        change_factor: f64,
        max_change_factor: f64,
        num_tries_for_increase_change_factor: i64,
        change_factor_increase: f64,
        max_abs_msun_yr: f64,
        roche_min_msun_yr: f64,
        min_imp_msun_yr: f64,
        starting_msun_yr: f64,
        solver_mode: ImplicitSolverMode,
        use_sum: bool,
        report_progress: bool,
    ) -> Self {
        Self {
            tolerance,
            tiny_factor,
            change_factor,
            max_change_factor,
            num_tries_for_increase_change_factor,
            change_factor_increase,
            max_abs_mdot_cgs: max_abs_msun_yr * MSUN / SECYER,
            roche_min_mdot_cgs: roche_min_msun_yr * MSUN / SECYER,
            min_mdot_for_implicit_cgs: min_imp_msun_yr * MSUN / SECYER,
            starting_mdot_cgs: starting_msun_yr * MSUN / SECYER,
            solver_mode,
            use_quadratic: solver_mode != ImplicitSolverMode::Bisect,
            use_sum,
            report_progress,
            fixed_delta_mdot: 0.0,
            have_lo: false,
            have_hi: false,
            mdot_lo: 0.0,
            mdot_hi: 0.0,
            f_lo: 0.0,
            f_hi: 0.0,
        }
    }

    fn emit_progress_header_if_needed(&self, num_tries: usize) {
        if self.report_progress && num_tries == 1 {
            println!(
                "  binary_step rlo_iter di ch_fact  curr_mdot  next_mdot    mdot_up   mdot_low    rl_gap1    rl_gap2          f              result"
            );
            println!(
                "  ________________________________________________________________________________________________________________________________"
            );
        }
    }

    fn emit_progress_row(
        &self,
        ctx: RloIterReportContext,
        num_tries: usize,
        curr_mdot: f64,
        next_mdot: f64,
        f: f64,
        result_label: &str,
        blank_line_after: bool,
    ) {
        if !self.report_progress {
            return;
        }
        let mdot_hi = if self.have_hi {
            -self.mdot_hi / MSUN * SECYER
        } else {
            f64::MAX
        };
        let mdot_lo = if self.have_lo {
            -self.mdot_lo / MSUN * SECYER
        } else {
            -f64::MAX
        };
        println!(
            "{:13}{:9}{:3}{:8.4}{:11.3e}{:11.3e}{:11.3e}{:11.3e}{:11.3e}{:11.3e}{:11.3e}{:>20}",
            ctx.model_number,
            num_tries,
            ctx.donor_i,
            self.change_factor,
            -curr_mdot / MSUN * SECYER,
            -next_mdot / MSUN * SECYER,
            mdot_hi,
            mdot_lo,
            ctx.rl_gaps_by_star[0],
            ctx.rl_gaps_by_star[1],
            f,
            result_label
        );
        if blank_line_after {
            println!();
        }
    }

    /// Evaluates the implicit objective and checks scheme-specific early-exit criteria
    /// (`binary_mdot.f90:36-322`) for `"roche_lobe"`, `"contact"`, `"Ritter"`, or `"Kolb"`.
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
    ) -> Result<ImplicitStepAction, StarError> {
        self.evaluate_trial_with_context(
            scheme,
            num_tries,
            max_tries,
            mdot_current,
            explicit_mdot,
            gap_donor,
            gap_accretor,
            q_accretor_over_donor,
            RloIterReportContext {
                donor_i: 1,
                rl_gaps_by_star: [gap_donor, gap_accretor],
                ..RloIterReportContext::default()
            },
        )
    }

    /// Evaluates `check_implicit_rlo` (`binary/private/binary_mdot.f90:36-322`)
    /// with full diagnostic context (`point_mass_i`, `model_number`, `d_i`).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_trial_with_context(
        &mut self,
        scheme: &str,
        num_tries: usize,
        max_tries: usize,
        mdot_current: f64,
        explicit_mdot: f64,
        gap_donor: f64,
        gap_accretor: f64,
        q_accretor_over_donor: f64,
        ctx: RloIterReportContext,
    ) -> Result<ImplicitStepAction, StarError> {
        self.emit_progress_header_if_needed(num_tries);

        let helper = ImplicitRlofSolver {
            tolerance: self.tolerance,
            ..ImplicitRlofSolver::default()
        };
        let scheme_low = scheme.trim().to_ascii_lowercase();
        let is_roche = scheme_low == "roche_lobe";
        let is_contact = scheme_low == "contact";
        if !is_roche && !is_contact {
            if scheme_low != "ritter"
                && scheme_low != "kolb"
                && scheme_low != "arras"
            {
                return Err(StarError::InvalidInput(format!(
                    concat!(
                        "unsupported implicit mdot_scheme '{}' ",
                        "(expected 'roche_lobe', 'contact', 'Ritter', 'Kolb', ",
                        "or 'Arras')"
                    ),
                    scheme
                )));
            }
            // `binary_mdot.f90:171-176`
            if explicit_mdot > 0.0 {
                return Ok(ImplicitStepAction::Converged(0.0));
            }
            // `binary_mdot.f90:177-182`
            let f = if explicit_mdot != 0.0 {
                (explicit_mdot - mdot_current) / explicit_mdot
            } else if mdot_current != 0.0 {
                (explicit_mdot - mdot_current) / mdot_current
            } else {
                0.0
            };
            // `binary_mdot.f90:183-191`: accept `new_mdot = explicit_mdot` when
            // both `|explicit_mdot|` and `|new_mdot|` lie within `min_mdot_for_implicit`.
            if explicit_mdot.abs() <= self.min_mdot_for_implicit_cgs
                && mdot_current.abs() <= self.min_mdot_for_implicit_cgs
            {
                self.emit_progress_row(
                    ctx,
                    num_tries,
                    mdot_current,
                    explicit_mdot,
                    f,
                    "OK (explicit)",
                    false,
                );
                return Ok(ImplicitStepAction::Converged(explicit_mdot));
            }
            return Ok(self.record_and_propose_with_ctx(
                num_tries,
                max_tries,
                mdot_current,
                f,
                false,
                false,
                ctx,
            ));
        }

        if is_contact && ctx.point_mass_i != 0 {
            // `binary_mdot.f90:114-119`: contact scheme with point-mass companion
            // sets `new_mdot = 0d0` and returns `keep_going`.
            return Ok(ImplicitStepAction::Converged(0.0));
        }

        let (f, trial_use_sum, detach_allowed) = if is_contact {
            let q = q_accretor_over_donor;
            let sc = pow(q, -0.52);
            let companion_detached = gap_accretor < 0.0
                && (gap_donor * sc - gap_accretor > self.tolerance
                    || gap_donor < -self.tolerance);
            if companion_detached {
                (helper.roche_lobe_objective(gap_donor), false, true)
            } else {
                (
                    helper.contact_objective(gap_donor, gap_accretor, q),
                    true,
                    false,
                )
            }
        } else {
            (helper.roche_lobe_objective(gap_donor), false, true)
        };
        Ok(self.record_and_propose_with_ctx(
            num_tries,
            max_tries,
            mdot_current,
            f,
            trial_use_sum,
            detach_allowed,
            ctx,
        ))
    }

    /// Flip sign of bracket endpoints after a mid-step `"contact"` donor switch
    /// (`binary_evolve.f90:222-233`).
    pub fn flip_for_contact_donor_switch(&mut self) {
        if self.report_progress {
            println!(" switching donor");
        }
        // `binary/private/binary_evolve.f90:224-227` swaps and negates only
        // `mdot_lo`/`mdot_hi`. `implicit_function_lo`/`implicit_function_hi`
        // are left as they were, so after a switch each stored function
        // value belongs to the other bound (deferred upstream bug `#274`).
        let mdot_hi_temp = self.mdot_hi;
        self.mdot_hi = -self.mdot_lo;
        self.mdot_lo = -mdot_hi_temp;
        if !self.have_lo {
            self.have_hi = false;
        }
        self.have_lo = true;
        self.fixed_delta_mdot /= 2.0;
    }

    /// Records trial `(mdot, f)` and proposes the next implicit RLOF action
    /// (`check_implicit_rlo` / `pick_mdot_for_implicit_rlo` in `binary_mdot.f90:194-425`).
    pub fn record_and_propose(
        &mut self,
        num_tries: usize,
        max_tries: usize,
        mdot: f64,
        f: f64,
        trial_use_sum: bool,
        detach_allowed: bool,
    ) -> ImplicitStepAction {
        self.record_and_propose_with_ctx(
            num_tries,
            max_tries,
            mdot,
            f,
            trial_use_sum,
            detach_allowed,
            RloIterReportContext::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_and_propose_with_ctx(
        &mut self,
        num_tries: usize,
        max_tries: usize,
        mdot: f64,
        f: f64,
        trial_use_sum: bool,
        detach_allowed: bool,
        ctx: RloIterReportContext,
    ) -> ImplicitStepAction {
        // `binary_mdot.f90:106-112, 133-139, 194`: detached at `mtransfer_rate == 0`.
        if detach_allowed && f < 0.0 && mdot.abs() == 0.0 {
            self.emit_progress_row(
                ctx,
                num_tries,
                mdot,
                mdot,
                f,
                "OK (detached)",
                false,
            );
            return ImplicitStepAction::Detached;
        }

        // `binary_mdot.f90:196-203`: within tolerance.
        if f.abs() <= self.tolerance {
            self.emit_progress_row(
                ctx,
                num_tries,
                mdot,
                mdot,
                f,
                "OK (in tol)",
                true,
            );
            return ImplicitStepAction::Converged(mdot);
        }

        // `binary_mdot.f90:205-212`: retry only when `num_tries > max_tries_to_achieve`.
        if num_tries > max_tries {
            self.emit_progress_row(
                ctx,
                num_tries,
                mdot,
                mdot,
                f,
                "retry (>max tries)",
                false,
            );
            return ImplicitStepAction::ExceededMaxTries;
        }

        // `binary_mdot.f90:214-220`: reset brackets and `fixed_delta_mdot` on try 1.
        if num_tries == 1 {
            self.have_lo = false;
            self.have_hi = false;
            self.mdot_lo = 0.0;
            self.mdot_hi = 0.0;
            self.fixed_delta_mdot = mdot * (1.0 - self.change_factor);
        }

        // `pick_mdot_for_implicit_rlo` (`binary_mdot.f90:324-425`).
        let starting_mdot = -self.starting_mdot_cgs;
        let raw_next = if self.have_lo && self.have_hi {
            let do_cubic = (num_tries.is_multiple_of(2)
                && self.solver_mode == ImplicitSolverMode::Both)
                || self.solver_mode == ImplicitSolverMode::Cubic;
            let cubic_mdot = if do_cubic {
                Some(
                    mesa_find0_quadratic(
                        self.mdot_lo,
                        self.f_lo,
                        mdot,
                        f,
                        self.mdot_hi,
                        self.f_hi,
                    )
                    .unwrap_or_else(|| {
                        mesa_find0(
                            self.mdot_lo,
                            self.f_lo,
                            self.mdot_hi,
                            self.f_hi,
                        )
                    }),
                )
            } else {
                None
            };
            if f >= 0.0 {
                self.mdot_lo = mdot;
                self.f_lo = f;
            } else {
                self.mdot_hi = mdot;
                self.f_hi = f;
            }
            cubic_mdot.unwrap_or((self.mdot_hi + self.mdot_lo) / 2.0)
        } else if self.have_lo {
            if f < 0.0 {
                self.mdot_hi = mdot;
                self.f_hi = f;
                self.have_hi = true;
                (self.mdot_hi + self.mdot_lo) / 2.0
            } else {
                self.mdot_lo = mdot;
                self.f_lo = f;
                self.mdot_lo * self.change_factor
            }
        } else if self.have_hi {
            if f >= 0.0 {
                self.mdot_lo = mdot;
                self.f_lo = f;
                self.have_lo = true;
                (self.mdot_hi + self.mdot_lo) / 2.0
            } else {
                self.mdot_hi = mdot;
                self.f_hi = f;
                if trial_use_sum && num_tries.is_multiple_of(2) {
                    self.mdot_hi + self.fixed_delta_mdot
                } else {
                    self.mdot_hi / self.change_factor
                }
            }
        } else if mdot > starting_mdot && mdot == 0.0 {
            starting_mdot
        } else if f >= 0.0 {
            self.mdot_lo = mdot;
            self.f_lo = f;
            self.have_lo = true;
            self.mdot_lo * self.change_factor
        } else {
            self.mdot_hi = mdot;
            self.f_hi = f;
            self.have_hi = true;
            if trial_use_sum && num_tries.is_multiple_of(2) {
                self.mdot_hi + self.fixed_delta_mdot
            } else {
                self.mdot_hi / self.change_factor
            }
        };

        // `binary_mdot.f90:228-236`: accept at `-max_implicit_abs_mdot` when
        // already at ceiling and next iteration demands even stronger mass loss.
        if mdot == -self.max_abs_mdot_cgs && mdot > raw_next {
            let capped = -self.max_abs_mdot_cgs;
            self.emit_progress_row(
                ctx,
                num_tries,
                mdot,
                capped,
                f,
                "OK (max mdot)",
                false,
            );
            return ImplicitStepAction::Converged(capped);
        }
        let next_mdot = if -raw_next > self.max_abs_mdot_cgs {
            -self.max_abs_mdot_cgs
        } else {
            raw_next
        };

        // `binary_mdot.f90:252-262`: accept detachment immediately with `new_mdot = 0d0`.
        if detach_allowed && -next_mdot < self.roche_min_mdot_cgs && f < 0.0 {
            self.emit_progress_row(
                ctx,
                num_tries,
                mdot,
                0.0,
                f,
                "OK (detachment)",
                false,
            );
            return ImplicitStepAction::Detached;
        }

        // `binary_mdot.f90:264-275`: accept when bracket width is below `tiny_factor`.
        if self.have_hi
            && self.have_lo
            && (self.mdot_hi - self.mdot_lo).abs()
                < self.tiny_factor * self.mdot_hi.abs().min(self.mdot_lo.abs())
        {
            self.emit_progress_row(
                ctx,
                num_tries,
                mdot,
                mdot,
                f,
                "OK (tiny change)",
                false,
            );
            return ImplicitStepAction::Converged(mdot);
        }

        // `binary_mdot.f90:279-285`: boost `change_factor` every
        // `num_tries_for_increase_change_factor` unbracketed tries.
        if self.num_tries_for_increase_change_factor > 0
            && self.change_factor_increase > 1.0
            && !(self.have_lo && self.have_hi)
            && i64::try_from(num_tries).is_ok_and(|n| {
                n % self.num_tries_for_increase_change_factor == 0
            })
        {
            self.change_factor = self
                .max_change_factor
                .min(self.change_factor_increase * self.change_factor);
        }

        self.emit_progress_row(
            ctx, num_tries, mdot, next_mdot, f, "redo", false,
        );
        ImplicitStepAction::TryNextMdot(next_mdot)
    }
}

/// Updates `b% change_factor` after an accepted step (`binary_finish_step` in
/// `binary/private/binary_evolve.f90:506-519`).
#[must_use]
pub fn update_change_factor_after_step(
    doing_first_model_of_run: bool,
    old_change_factor: f64,
    mtransfer_rate_old: f64,
    mtransfer_rate_new: f64,
    ctrl: &BinaryInlistControls,
) -> f64 {
    if doing_first_model_of_run
        || mtransfer_rate_old == mtransfer_rate_new
        || mtransfer_rate_new == 0.0
        || mtransfer_rate_old == 0.0
    {
        return old_change_factor;
    }
    let lam = ctrl.implicit_lambda;
    let frac = ctrl.change_factor_fraction;
    let mut next = if mtransfer_rate_new < mtransfer_rate_old {
        old_change_factor * (1.0 - lam)
            + lam
                * (1.0 + frac * (mtransfer_rate_new / mtransfer_rate_old - 1.0))
    } else {
        old_change_factor * (1.0 - lam)
            + lam
                * (1.0 + frac * (mtransfer_rate_old / mtransfer_rate_new - 1.0))
    };
    if next > ctrl.max_change_factor {
        next = ctrl.max_change_factor;
    }
    if next < ctrl.min_change_factor {
        next = ctrl.min_change_factor;
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f3_4_mesa_find0_quadratic_handles_cgs_mdot_scale_and_cubic_modes() {
        let u = MSUN / SECYER;
        let x_lo = -1.0e-5 * u;
        let x_mid = -2.0e-5 * u;
        let x_hi = -3.0e-5 * u;
        let root_q = mesa_find0_quadratic(x_lo, 0.08, x_mid, 0.01, x_hi, -0.09)
            .expect("CGS quadratic root must succeed without 1e-40 cutoff");
        assert!(root_q < x_lo && root_q > x_hi);

        // Compare `solver_type = 'cubic'` vs `'both'` vs `'bisect'` on odd try 3.
        let make_solver = |st: &str| {
            let mut c = BinaryInlistControls {
                mdot_scheme: String::from("roche_lobe"),
                solver_type: String::from(st),
                implicit_scheme_tolerance: 1.0e-4,
                ..BinaryInlistControls::default()
            };
            c.roche_min_mdot = 1.0e-20;
            let mut s = ImplicitMdotSolver::from_controls(&c, 1.5).unwrap();
            let _ = s
                .evaluate_trial("roche_lobe", 1, 20, x_lo, 0.0, 0.04, -0.5, 0.8)
                .unwrap();
            let _ = s
                .evaluate_trial(
                    "roche_lobe",
                    2,
                    20,
                    x_hi,
                    0.0,
                    -0.05,
                    -0.5,
                    0.8,
                )
                .unwrap();
            s
        };
        let mut s_cubic = make_solver("cubic");
        let mut s_both = make_solver("both");
        let mut s_bisect = make_solver("bisect");

        let p_cubic = s_cubic
            .evaluate_trial("roche_lobe", 3, 20, x_mid, 0.0, 0.005, -0.5, 0.8)
            .unwrap();
        let p_both = s_both
            .evaluate_trial("roche_lobe", 3, 20, x_mid, 0.0, 0.005, -0.5, 0.8)
            .unwrap();
        let p_bisect = s_bisect
            .evaluate_trial("roche_lobe", 3, 20, x_mid, 0.0, 0.005, -0.5, 0.8)
            .unwrap();
        // On odd try 3, `both` and `bisect` bisect `[x_mid, x_hi]`, while `cubic` runs `find0_quadratic`.
        assert_eq!(p_both, p_bisect);
        assert_ne!(p_cubic, p_bisect);
    }

    #[test]
    fn test_f3_4_max_tries_plus_one_detachment_and_change_factor_update() {
        let u = MSUN / SECYER;
        let ctrl = BinaryInlistControls {
            mdot_scheme: String::from("roche_lobe"),
            max_tries_to_achieve: 3,
            implicit_scheme_tolerance: 1.0e-3,
            num_tries_for_increase_change_factor: 2,
            change_factor_increase: 1.2,
            max_change_factor: 2.0,
            ..BinaryInlistControls::default()
        };
        let mut solver = ImplicitMdotSolver::from_controls(&ctrl, 1.2).unwrap();

        // Try 1 (`mdot = 0`, overflowing): proposes `-starting_mdot`.
        let a1 = solver
            .evaluate_trial("roche_lobe", 1, 3, 0.0, 0.0, 0.01, -0.5, 0.8)
            .unwrap();
        assert_eq!(
            a1,
            ImplicitStepAction::TryNextMdot(
                -ctrl.starting_mdot * MSUN / SECYER
            )
        );
        // Try 2 (unbracketed, `mod(2, 2) == 0`): boosts `change_factor` `1.2 -> 1.44`.
        let a2 = solver
            .evaluate_trial(
                "roche_lobe",
                2,
                3,
                -ctrl.starting_mdot * MSUN / SECYER,
                0.0,
                0.01,
                -0.5,
                0.8,
            )
            .unwrap();
        assert!(matches!(a2, ImplicitStepAction::TryNextMdot(_)));
        assert!((solver.change_factor - 1.44).abs() < 1.0e-12);

        // Try 3 (`num_tries == max_tries_to_achieve`): still proposes a 4th try (`num_tries <= max_tries`).
        let a3 = solver
            .evaluate_trial(
                "roche_lobe",
                3,
                3,
                -ctrl.starting_mdot * MSUN / SECYER * 1.44,
                0.0,
                0.01,
                -0.5,
                0.8,
            )
            .unwrap();
        assert!(matches!(a3, ImplicitStepAction::TryNextMdot(_)));

        // Try 4 (`num_tries == max_tries + 1`): if within tolerance, accepts; if outside, retries.
        let mut solver_fail = solver.clone();
        let a4_ok = solver
            .evaluate_trial(
                "roche_lobe",
                4,
                3,
                -2.0 * ctrl.starting_mdot * u,
                0.0,
                -0.0004,
                -0.5,
                0.8,
            )
            .unwrap();
        assert_eq!(
            a4_ok,
            ImplicitStepAction::Converged(-2.0 * ctrl.starting_mdot * u)
        );
        let a4_fail = solver_fail
            .evaluate_trial(
                "roche_lobe",
                4,
                3,
                -2.0 * ctrl.starting_mdot * u,
                0.0,
                0.01,
                -0.5,
                0.8,
            )
            .unwrap();
        assert_eq!(a4_fail, ImplicitStepAction::ExceededMaxTries);

        // Detachment when `-next_mdot < roche_min_mdot` and `f < 0` returns `Detached` immediately,
        // and try 3 with `-next_mdot >= roche_min_mdot` uses `/ change_factor` (not invented `0.0`).
        let mut s_hi = ImplicitMdotSolver::from_controls(&ctrl, 1.5).unwrap();
        for t in 1..=3_i32 {
            let m_in = -1.0e-10 * u / pow(1.5, f64::from(t - 1));
            let act = s_hi
                .evaluate_trial(
                    "roche_lobe",
                    t as usize,
                    20,
                    m_in,
                    0.0,
                    -0.02,
                    -0.5,
                    0.8,
                )
                .unwrap();
            let ImplicitStepAction::TryNextMdot(m_out) = act else {
                panic!("expected TryNextMdot on try {t}, got {act:?}");
            };
            assert!(
                m_out < 0.0,
                "try 3 must divide by change_factor rather than jumping to 0"
            );
        }

        // Ritter `min_mdot_for_implicit` returns `Converged(explicit_mdot)` and does not seed `explicit_mdot` on try 1.
        let r_ctrl = BinaryInlistControls::default();
        let mut s_rit =
            ImplicitMdotSolver::from_controls(&r_ctrl, 1.5).unwrap();
        let exp_tiny = -1.0e-18 * u;
        assert_eq!(
            s_rit
                .evaluate_trial("Ritter", 1, 20, 0.0, exp_tiny, -0.2, -0.5, 0.8)
                .unwrap(),
            ImplicitStepAction::Converged(exp_tiny)
        );
        let exp_active = -1.0e-9 * u;
        assert_eq!(
            s_rit
                .evaluate_trial(
                    "Ritter", 1, 20, 0.0, exp_active, 0.0, -0.5, 0.8
                )
                .unwrap(),
            ImplicitStepAction::TryNextMdot(
                -r_ctrl.starting_mdot * MSUN / SECYER,
            )
        );

        // Post-step `change_factor`: unchanged on step 1 or when `mtransfer_rate_old == mtransfer_rate`.
        assert_eq!(
            update_change_factor_after_step(
                true,
                1.5,
                -1.0e-8 * u,
                -2.0e-8 * u,
                &r_ctrl
            ),
            1.5
        );
        assert_eq!(
            update_change_factor_after_step(
                false,
                1.5,
                -1.0e-8 * u,
                -1.0e-8 * u,
                &r_ctrl
            ),
            1.5
        );
        let cf_small_change = update_change_factor_after_step(
            false,
            1.5,
            -1.0e-8 * u,
            -1.01e-8 * u,
            &r_ctrl,
        );
        assert!(
            cf_small_change < 1.5
                && cf_small_change >= r_ctrl.min_change_factor
        );
    }
}
