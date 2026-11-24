// SPDX-License-Identifier: GPL-3.0-or-later
//! ADIPLS control common blocks (`cadr_param`, `cadi_param`, `coutpt`),
//! `set_adipls_controls` (`astero/private/adipls_support.f90:539-587`), caller
//! `sig_fac` / `sig1` / `sig2` / `dfsig` helpers (`adipls_support.f90:81-94`,
//! `astero_support.f90:109-128`), and documentation of `#480` argument
//! shadowing (`adipls_support.f90:217-243, 330-389`).

use flexmesa_const::TWO_PI;
use flexmesa_math::{pow2, pow3};

use crate::parser::AdiplsInput;

/// Real parameter common block `common/cadr_param/` (12 `f64` values)
/// (`adipls/adipack.c/adipls/res_adipar.c.d.f:20-23`,
/// `astero/private/adipls_support.f90:547-554`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadrParam {
    /// Degree $l$ (`para_el`).
    pub el: f64,
    /// Starting degree for degree loops (`para_els1`).
    pub els1: f64,
    /// Degree step (`para_dels`).
    pub dels: f64,
    /// Lower frequency bound increment (`para_dfsig1`).
    pub dfsig1: f64,
    /// Upper frequency bound increment (`para_dfsig2`).
    pub dfsig2: f64,
    /// Lower dimensionless squared frequency bound `sigma_1^2` (`para_sig1`).
    pub sig1: f64,
    /// Upper dimensionless squared frequency bound `sigma_2^2` (`para_sig2`).
    pub sig2: f64,
    /// Frequency step or squared asymptotic spacing (`para_dfsig`).
    pub dfsig: f64,
    /// Lower trial degree window (`para_eltrw1`).
    pub eltrw1: f64,
    /// Upper trial degree window (`para_eltrw2`).
    pub eltrw2: f64,
    /// Lower trial $\sigma^2$ window (`para_sgtrw1`).
    pub sgtrw1: f64,
    /// Upper trial $\sigma^2$ window (`para_sgtrw2`).
    pub sgtrw2: f64,
}

impl CadrParam {
    /// Construct [`CadrParam`] from parsed `adipls.c.pruned.in` `osc` settings,
    /// matching the initial `res_adipar(1, ...)` call at
    /// `adipls.c.d.f:1223-1227`.
    #[must_use]
    pub const fn from_input(input: &AdiplsInput) -> Self {
        let osc = input.osc;
        Self {
            el: osc.el,
            els1: osc.els1,
            dels: osc.dels,
            dfsig1: osc.dfsig1,
            dfsig2: osc.dfsig2,
            sig1: osc.sig1,
            sig2: osc.sig2,
            dfsig: osc.dfsig,
            eltrw1: osc.eltrw1,
            eltrw2: osc.eltrw2,
            sgtrw1: osc.sgtrw1,
            sgtrw2: osc.sgtrw2,
        }
    }

    /// Return all 12 entries in `common/cadr_param/` order (`calls.tsv`).
    #[must_use]
    pub const fn to_array(self) -> [f64; 12] {
        [
            self.el,
            self.els1,
            self.dels,
            self.dfsig1,
            self.dfsig2,
            self.sig1,
            self.sig2,
            self.dfsig,
            self.eltrw1,
            self.eltrw2,
            self.sgtrw1,
            self.sgtrw2,
        ]
    }
}

/// Integer parameter common block `common/cadi_param/` (8 `i32` values) as
/// declared in `astero/private/adipls_support.f90:556-561`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CadiParam {
    /// Degree loop count (`ipara_nsel`).
    pub nsel: i32,
    /// `dfsig1` step type (`ipara_nsig1`).
    pub nsig1: i32,
    /// `dfsig2` step type (`ipara_nsig2`).
    pub nsig2: i32,
    /// Trial frequency mode (`ipara_itrsig`).
    pub itrsig: i32,
    /// Scan/step spacing type (`ipara_nsig`).
    pub nsig: i32,
    /// Frequency limit interpretation (`ipara_istsig`).
    pub istsig: i32,
    /// Initial mode index (`ipara_inomd1`).
    pub inomd1: i32,
    /// Scan grid count (`ipara_iscan`).
    pub iscan: i32,
}

impl CadiParam {
    /// Construct [`CadiParam`] from parsed `adipls.c.pruned.in` `osc` settings,
    /// matching the initial `res_adipar(1, ...)` call at
    /// `adipls.c.d.f:1223-1227`.
    #[must_use]
    pub const fn from_input(input: &AdiplsInput) -> Self {
        let osc = input.osc;
        Self {
            nsel: osc.nsel,
            nsig1: osc.nsig1,
            nsig2: osc.nsig2,
            itrsig: osc.itrsig,
            nsig: osc.nsig,
            istsig: osc.istsig,
            inomd1: osc.inomd1,
            iscan: osc.iscan,
        }
    }

    /// Return all 8 entries in `common/cadi_param/` order (`calls.tsv`).
    #[must_use]
    pub const fn to_array(self) -> [i32; 8] {
        [
            self.nsel,
            self.nsig1,
            self.nsig2,
            self.itrsig,
            self.nsig,
            self.istsig,
            self.inomd1,
            self.iscan,
        ]
    }
}

/// Four kernel output entries of `common/coutpt/` overwritten by
/// `set_adipls_controls` (`adipls_support.f90:563-570, 582-585`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoutptKernelControls {
    /// Rotational kernel control (`irotkr`).
    pub irotkr: i32,
    /// Rotational kernel print points (`nprtkr`).
    pub nprtkr: i32,
    /// Gamma1 kernel control (`igm1kr`).
    pub igm1kr: i32,
    /// Gamma1 kernel print points (`npgmkr`).
    pub npgmkr: i32,
}

impl CoutptKernelControls {
    /// Construct from parsed `adipls.c.pruned.in` `out` controls.
    #[must_use]
    pub const fn from_input(input: &AdiplsInput) -> Self {
        Self {
            irotkr: input.out.irotkr,
            nprtkr: input.out.nprtkr,
            igm1kr: input.out.igm1kr,
            npgmkr: input.out.npgmkr,
        }
    }

    /// Return all 4 entries `[irotkr, nprtkr, igm1kr, npgmkr]` (`calls.tsv`).
    #[must_use]
    pub const fn to_array(self) -> [i32; 4] {
        [self.irotkr, self.nprtkr, self.igm1kr, self.npgmkr]
    }
}

/// Combined ADIPLS parameter transmission state updated by
/// [`set_adipls_controls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsControlState {
    /// 12 real parameters in `common/cadr_param/`.
    pub cadr_param: CadrParam,
    /// 8 integer parameters in `common/cadi_param/`.
    pub cadi_param: CadiParam,
    /// 4 kernel output entries in `common/coutpt/`.
    pub coutpt: CoutptKernelControls,
}

impl AdiplsControlState {
    /// Initialize from parsed `adipls.c.pruned.in` defaults (`setup_adipls`).
    #[must_use]
    pub const fn from_input(input: &AdiplsInput) -> Self {
        Self {
            cadr_param: CadrParam::from_input(input),
            cadi_param: CadiParam::from_input(input),
            coutpt: CoutptKernelControls::from_input(input),
        }
    }

    /// Update the 14 control fields written by upstream `set_adipls_controls`
    /// (`astero/private/adipls_support.f90:539-587`), leaving the remaining
    /// `cadr_param` and `cadi_param` fields from `adipls.c.pruned.in` intact.
    pub fn apply_controls(&mut self, args: &SetAdiplsControlsArgs) {
        self.cadr_param.el = f64::from(args.el);
        self.cadi_param.nsel = args.nsel;
        self.cadr_param.els1 = args.els1;
        self.cadr_param.dels = args.dels;
        self.cadi_param.itrsig = args.itrsig;
        self.cadi_param.iscan = args.iscan;
        self.cadr_param.sig1 = args.sig1;
        self.cadr_param.sig2 = args.sig2;
        self.cadr_param.dfsig = args.dfsig;
        self.cadi_param.nsig = args.nsig;
        self.coutpt.irotkr = args.irotkr;
        self.coutpt.nprtkr = args.nprtkr;
        self.coutpt.igm1kr = args.igm1kr;
        self.coutpt.npgmkr = args.npgmkr;
    }
}

/// Argument bundle for [`AdiplsControlState::apply_controls`] mirroring the
/// 14 parameters of `set_adipls_controls` (`adipls_support.f90:539-544`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetAdiplsControlsArgs {
    /// Mode spherical harmonic degree $l$.
    pub el: i32,
    /// Degree loop count (`nsel`, `0` for single-$l$ astero calls).
    pub nsel: i32,
    /// Starting degree (`els1 = dble(el)`).
    pub els1: f64,
    /// Degree increment (`dels = 1.0`).
    pub dels: f64,
    /// Trial frequency mode (`itrsig = 1`).
    pub itrsig: i32,
    /// Scan grid point count (`iscan`).
    pub iscan: i32,
    /// Lower dimensionless squared frequency bound `sigma_1^2`.
    pub sig1: f64,
    /// Upper dimensionless squared frequency bound `sigma_2^2`.
    pub sig2: f64,
    /// Squared spacing parameter `dfsig`.
    pub dfsig: f64,
    /// Scan spacing type (`nsig = 2` uniform in frequency).
    pub nsig: i32,
    /// Rotational kernel control (`irotkr`).
    pub irotkr: i32,
    /// Rotational kernel print points (`nprtkr`).
    pub nprtkr: i32,
    /// Gamma1 kernel control (`igm1kr`).
    pub igm1kr: i32,
    /// Gamma1 kernel print points (`npgmkr`).
    pub npgmkr: i32,
}

/// Direct positional wrapper matching MESA 26.4.1 `set_adipls_controls`
/// (`astero/private/adipls_support.f90:539-587`).
#[allow(clippy::too_many_arguments)]
pub fn set_adipls_controls(
    state: &mut AdiplsControlState,
    el: i32,
    nsel: i32,
    els1: f64,
    dels: f64,
    itrsig: i32,
    iscan: i32,
    sig1: f64,
    sig2: f64,
    dfsig: f64,
    nsig: i32,
    irotkr: i32,
    nprtkr: i32,
    igm1kr: i32,
    npgmkr: i32,
) {
    state.apply_controls(&SetAdiplsControlsArgs {
        el,
        nsel,
        els1,
        dels,
        itrsig,
        iscan,
        sig1,
        sig2,
        dfsig,
        nsig,
        irotkr,
        nprtkr,
        igm1kr,
        npgmkr,
    });
}

/// Compute `sig_fac = (2 * pi)**2 * pow3(R) / (G * M)` as evaluated by
/// `do_adipls_get_one_el_info` (`astero/private/adipls_support.f90:81`).
///
/// Preserves the parenthesized `pow3(r_cm)` grouping, which can differ by
/// 1 ULP from left-to-right `r_cm * r_cm * r_cm` in
/// [`sig_fac_astero_support`].
#[must_use]
#[inline]
pub fn sig_fac_adipls_support(r_cm: f64, g_cgs: f64, m_g: f64) -> f64 {
    (pow2(TWO_PI) * pow3(r_cm)) / (g_cgs * m_g)
}

/// Compute `sig_fac = (2 * pi) * (2 * pi) * R * R * R / (G * M)` as evaluated
/// by `get_one_el_info` (`astero/private/astero_support.f90:112`).
///
/// Preserves exact left-to-right multiplication order
/// `((((2*pi)*(2*pi))*R)*R)*R / (G*M)` without parenthesizing `R*R*R`.
#[must_use]
#[inline]
pub fn sig_fac_astero_support(r_cm: f64, g_cgs: f64, m_g: f64) -> f64 {
    ((((TWO_PI * TWO_PI) * r_cm) * r_cm) * r_cm) / (g_cgs * m_g)
}

/// Convert cyclic frequency $\nu$ ($\mu\mathrm{Hz}$) to dimensionless squared
/// frequency $\sigma^2$ (`adipls_support.f90:86-87`,
/// `astero_support.f90:120-121`).
#[must_use]
#[inline]
pub fn sig_from_nu_uhz(sig_fac: f64, nu_uhz: f64) -> f64 {
    (sig_fac * (nu_uhz * 1.0e-6_f64)) * (nu_uhz * 1.0e-6_f64)
}

/// Compute `dfsig = sig_fac * delta_nu_model * delta_nu_model`
/// (`adipls_support.f90:89`, `astero_support.f90:122`).
///
/// Preserves upstream's missing `1e-6` scaling when `delta_nu_model` is in
/// $\mu\mathrm{Hz}$; unused by `sigscn` when `iscan > 1` and `nsel == 0`.
#[must_use]
#[inline]
pub fn dfsig_from_delta_nu(sig_fac: f64, delta_nu_model: f64) -> f64 {
    (sig_fac * delta_nu_model) * delta_nu_model
}

/// Caller-supplied pulse export flags passed to `do_adipls_get_one_el_info` /
/// `run_adipls` (`astero/private/adipls_support.f90:61-62, 336-338`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdiplsCallerFlags {
    /// Caller `add_center_point` argument (shadowed by `astero_def`, `#480`).
    pub caller_add_center_point: bool,
    /// Caller `keep_surface_point` argument (shadowed by `astero_def`, `#480`).
    pub caller_keep_surface_point: bool,
    /// Caller `add_atmosphere` argument (forwarded to `store_model_for_adipls`).
    pub add_atmosphere: bool,
    /// Caller `do_redistribute_mesh` argument (forwarded to `store_model_for_adipls`).
    pub do_redistribute_mesh: bool,
}

/// Module-scope FGONG export flags in `astero_def` (`astero_def.f90:260-262`,
/// defaults in `astero_search.defaults:1285, 1293`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsteroDefPulseFlags {
    /// `astero_def::add_center_point` (default `true`).
    pub add_center_point: bool,
    /// `astero_def::keep_surface_point` (default `false`).
    pub keep_surface_point: bool,
}

impl Default for AsteroDefPulseFlags {
    fn default() -> Self {
        Self {
            add_center_point: true,
            keep_surface_point: false,
        }
    }
}

/// Effective flags passed by `store_model_for_adipls` into
/// `star_get_pulse_data(..., 'FGONG', ...)` (`adipls_support.f90:217-243`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedPulseExportFlags {
    /// Effective `add_center_point` read from `astero_def` (`#480`).
    pub add_center_point: bool,
    /// Effective `keep_surface_point` read from `astero_def` (`#480`).
    pub keep_surface_point: bool,
    /// Effective `add_atmosphere` forwarded from the caller.
    pub add_atmosphere: bool,
    /// Effective `do_redistribute_mesh` forwarded from the caller.
    pub do_redistribute_mesh: bool,
}

/// Resolve the FGONG export flags actually passed to `star_get_pulse_data` by
/// `store_model_for_adipls` (`astero/private/adipls_support.f90:217-243,
/// 384-386`), reproducing deferred issue `#480` without altering upstream
/// semantics (`CLAUDE.md` parity-first policy).
///
/// Upstream `run_adipls` accepts `add_center_point` and `keep_surface_point`
/// parameters but invokes `store_model_for_adipls(s, add_atmosphere,
/// do_redistribute_mesh, ierr)` without forwarding them. Consequently,
/// `store_model_for_adipls` reads module-level `astero_def::add_center_point`
/// and `astero_def::keep_surface_point`, shadowing the caller arguments.
#[must_use]
pub const fn resolve_store_model_flags(
    caller: AdiplsCallerFlags,
    astero_def: AsteroDefPulseFlags,
) -> ResolvedPulseExportFlags {
    // Reproduce `#480` shadowing verbatim from `adipls_support.f90:242-243, 386`:
    // caller_add_center_point and caller_keep_surface_point are ignored.
    ResolvedPulseExportFlags {
        add_center_point: astero_def.add_center_point,
        keep_surface_point: astero_def.keep_surface_point,
        add_atmosphere: caller.add_atmosphere,
        do_redistribute_mesh: caller.do_redistribute_mesh,
    }
}
