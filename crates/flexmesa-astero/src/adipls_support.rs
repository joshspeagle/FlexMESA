// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream `adipls_support` bridge connecting `flexmesa-star` models and
//! [`AsteroSearchControls`] to the pure-Rust `flexmesa-adipls` oscillation
//! solver (`astero/private/adipls_support.f90` and
//! `astero/private/astero_support.f90:108-156`).

use std::path::Path;

use flexmesa_adipls::{
    dfsig_from_delta_nu, fgong_amdl, parse_adipls_pruned_file,
    parse_redistrb_pruned_file, resolve_store_model_flags, set_adipls_controls,
    sig_fac_adipls_support, sig_fac_astero_support, sig_from_nu_uhz,
    store_amdl, AdiplsCallerFlags, AdiplsControlState, AdiplsError,
    AdiplsInput, AdiplsSolver, AsteroDefPulseFlags, FgongModel, RedistrbInput,
    StoreAmdlOutput,
};
use flexmesa_const::{R_SUN, STANDARD_CGRAV};
use flexmesa_star::controls::StarError;
use flexmesa_star::eval_photosphere_info;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_fgong_data::{star_get_pulse_data_fgong, FGONG_IVAR};

use crate::inlist_astero::AsteroSearchControls;

/// Single pulsation mode record matching MESA `astero_def` oscillation arrays
/// (`el`, `order`, `em`, `inertia`, `cyclic_freq`, `growth_rate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscResult {
    /// Spherical harmonic degree $\ell$ (`el`).
    pub el: i32,
    /// Radial mode order $n_{\mathrm{pg}}$ (`order`).
    pub order: i32,
    /// Azimuthal order $m$ (`em`).
    pub em: i32,
    /// Normalized mode inertia $E = 4\pi E_{\mathrm{kin}}$ (`inertia`).
    pub inertia: f64,
    /// Cyclic oscillation frequency in $\mu\mathrm{Hz}$ (`cyclic_freq`).
    pub cyclic_freq: f64,
    /// Growth rate (`0.0` for adiabatic ADIPLS).
    pub growth_rate: f64,
}

/// Stateful ADIPLS asteroseismic solver driver matching MESA's
/// `adipls_support` lifecycle (`init`, `store_model_for_adipls`,
/// `do_adipls_get_one_el_info`, `get_one_el_info`).
#[derive(Debug, Clone)]
pub struct AsteroAdipls {
    input: AdiplsInput,
    solver: AdiplsSolver,
    redistrb_input: Option<RedistrbInput>,
    astero_def_flags: AsteroDefPulseFlags,
    default_add_atmosphere: bool,
    stored_model: Option<StoreAmdlOutput>,
}

impl AsteroAdipls {
    /// Initialize [`AsteroAdipls`] from `case_dir/adipls.c.pruned.in` and
    /// `ctrl` (`do_redistribute_mesh = false`).
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] if `ctrl.oscillation_code` is not
    /// `"adipls"` or if `adipls.c.pruned.in` fails parsing, or
    /// [`StarError::IoError`] if `adipls.c.pruned.in` cannot be read.
    pub fn init(
        case_dir: &Path,
        ctrl: &AsteroSearchControls,
    ) -> Result<Self, StarError> {
        Self::init_with_redistribute(case_dir, ctrl, false)
    }

    /// Initialize [`AsteroAdipls`] from `case_dir/adipls.c.pruned.in`, also
    /// parsing `case_dir/redistrb.c.pruned.in` when `do_redistribute_mesh` is
    /// `true` (matching MESA startup parsing in `adipls_support_procs`).
    ///
    /// Runtime mesh redistribution (`redist_mesh = true` during
    /// [`Self::store_model`]) remains explicitly rejected until `srdist` is
    /// ported.
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] if `ctrl.oscillation_code !=
    /// "adipls"` or either pruned input file fails validation, or
    /// [`StarError::IoError`] if reading a file fails.
    pub fn init_with_redistribute(
        case_dir: &Path,
        ctrl: &AsteroSearchControls,
        do_redistribute_mesh: bool,
    ) -> Result<Self, StarError> {
        let osc_low = ctrl.oscillation_code.trim().to_ascii_lowercase();
        if osc_low != "adipls" {
            return Err(StarError::InvalidInput(format!(
                "AsteroAdipls::init requires oscillation_code == 'adipls', got '{}'",
                ctrl.oscillation_code
            )));
        }

        let adipls_path = case_dir.join("adipls.c.pruned.in");
        let input =
            parse_adipls_pruned_file(&adipls_path).map_err(map_adipls_err)?;

        let redistrb_input = if do_redistribute_mesh {
            let redistrb_path = case_dir.join("redistrb.c.pruned.in");
            Some(
                parse_redistrb_pruned_file(&redistrb_path)
                    .map_err(map_adipls_err)?,
            )
        } else {
            None
        };

        let solver = AdiplsSolver::new(input.clone());
        Ok(Self {
            input,
            solver,
            redistrb_input,
            astero_def_flags: AsteroDefPulseFlags {
                add_center_point: ctrl.add_center_point,
                keep_surface_point: ctrl.keep_surface_point,
            },
            default_add_atmosphere: ctrl.add_atmosphere,
            stored_model: None,
        })
    }

    /// Borrow the underlying [`AdiplsInput`] parsed from `adipls.c.pruned.in`.
    #[must_use]
    pub const fn input(&self) -> &AdiplsInput {
        &self.input
    }

    /// Borrow the underlying [`AdiplsSolver`].
    #[must_use]
    pub const fn solver(&self) -> &AdiplsSolver {
        &self.solver
    }

    /// Borrow the parsed [`RedistrbInput`] when `init_with_redistribute` was
    /// called with `do_redistribute_mesh = true`.
    #[must_use]
    pub const fn redistrb_input(&self) -> Option<&RedistrbInput> {
        self.redistrb_input.as_ref()
    }

    /// Borrow the currently cached [`StoreAmdlOutput`] (`x`, `aa`, `data`), if
    /// [`Self::store_model`] has been run.
    #[must_use]
    pub const fn stored_model(&self) -> Option<&StoreAmdlOutput> {
        self.stored_model.as_ref()
    }

    /// Convert `model` into an ADIPLS AMDL buffer via
    /// `star_get_pulse_data_fgong` -> `fgong_amdl` -> `store_amdl` and cache it
    /// (`store_model_for_adipls` in `astero/private/adipls_support.f90:217-327`).
    ///
    /// Uses `ctrl.add_center_point` and `ctrl.keep_surface_point` from this
    /// instance's [`AsteroDefPulseFlags`] (passing those into
    /// [`resolve_store_model_flags`] to honor `#480` shadowing), and leaves AMDL
    /// rows `6..=9` explicitly uninitialized per `#479`.
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] if `redist_mesh` is `true`
    /// (`srdist` mesh redistribution is not ported), or propagates FGONG/AMDL
    /// errors.
    pub fn store_model(
        &mut self,
        model: &StarModel,
        add_atmosphere: bool,
        redist_mesh: bool,
    ) -> Result<&StoreAmdlOutput, StarError> {
        self.store_model_with_caller_flags(
            model,
            self.astero_def_flags.add_center_point,
            self.astero_def_flags.keep_surface_point,
            add_atmosphere,
            redist_mesh,
        )
    }

    /// Variant of [`Self::store_model`] taking explicit caller endpoint flags
    /// (`caller_add_center_point`, `caller_keep_surface_point`) resolved
    /// through [`resolve_store_model_flags`] so `#480` shadowing against
    /// `astero_def` controls can be exercised and verified directly.
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] if `redist_mesh` is `true`, or
    /// propagates FGONG/AMDL errors.
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn store_model_with_caller_flags(
        &mut self,
        model: &StarModel,
        caller_add_center_point: bool,
        caller_keep_surface_point: bool,
        add_atmosphere: bool,
        redist_mesh: bool,
    ) -> Result<&StoreAmdlOutput, StarError> {
        let resolved = resolve_store_model_flags(
            AdiplsCallerFlags {
                caller_add_center_point,
                caller_keep_surface_point,
                add_atmosphere,
                do_redistribute_mesh: redist_mesh,
            },
            self.astero_def_flags,
        );
        if resolved.do_redistribute_mesh {
            return Err(StarError::InvalidInput(String::from(
                "ADIPLS runtime mesh redistribution (srdist / do_redistribute_mesh = .true.) is not ported",
            )));
        }

        let fg = star_get_pulse_data_fgong(
            model,
            resolved.add_center_point,
            resolved.keep_surface_point,
            resolved.add_atmosphere,
        )
        .map_err(StarError::from)?;

        let fgong_model = FgongModel::new(
            fg.global_data_vec(),
            FGONG_IVAR,
            fg.nn,
            fg.point_var_flat(),
        )
        .map_err(map_adipls_err)?;

        let fg_amdl = fgong_amdl(self.input.cst.cgrav, &fgong_model)
            .map_err(map_adipls_err)?;

        let st = store_amdl(
            fg_amdl.nn,
            self.input.int_ctrl.iriche,
            self.input.int_ctrl.iturpr,
            fg_amdl.data,
            &fg_amdl.aa,
        )
        .map_err(map_adipls_err)?;

        self.stored_model = Some(st);
        self.stored_model.as_ref().ok_or_else(|| {
            StarError::InvalidInput(String::from(
                "internal error storing ADIPLS model",
            ))
        })
    }

    /// Evaluate all ADIPLS modes for degree `l` in `[nu1, nu2]` $\mu\mathrm{Hz}$
    /// following `astero/private/astero_support.f90:108-156`, returning results
    /// sorted by ascending `cyclic_freq`.
    ///
    /// Uses `R = R_SUN * eval_photosphere_info(model).r_rsun`,
    /// `G = STANDARD_CGRAV`, `M = model.m_grav[0]`, and
    /// `sig_fac_astero_support(R, G, M)`.
    ///
    /// # Errors
    /// Returns [`StarError`] if geometry is invalid, storing fails,
    /// `store_for_adipls = false` before any model was stored, or ADIPLS fails.
    #[allow(clippy::too_many_arguments)]
    pub fn get_one_el_info(
        &mut self,
        model: &StarModel,
        l: usize,
        nu1: f64,
        nu2: f64,
        iscan: i32,
        delta_nu_model: f64,
        store_for_adipls: bool,
    ) -> Result<Vec<OscResult>, StarError> {
        if model.m_grav.is_empty() {
            return Err(StarError::InvalidInput(String::from(
                "AsteroAdipls::get_one_el_info requires non-empty model.m_grav",
            )));
        }
        let el = i32::try_from(l).map_err(|_| {
            StarError::InvalidInput(format!("degree l={l} overflows i32"))
        })?;
        let r_cm = R_SUN * eval_photosphere_info(model).r_rsun;
        let g_cgs = STANDARD_CGRAV;
        let m_g = model.m_grav[0];
        let sig_fac = sig_fac_astero_support(r_cm, g_cgs, m_g);
        self.run_adipls_one_el(
            model,
            el,
            nu1,
            nu2,
            iscan,
            sig_fac,
            delta_nu_model,
            self.default_add_atmosphere,
            false,
            store_for_adipls,
            0,
            1.0,
        )
    }

    /// Direct mirror of `do_adipls_get_one_el_info`
    /// (`astero/private/adipls_support.f90:50-130`) taking explicit `(r_cm,
    /// g_cgs, m_g, add_atmosphere, do_redistribute_mesh, store_for_adipls,
    /// nsel, dels)` and computing `sig_fac_adipls_support(r_cm, g_cgs, m_g)`.
    ///
    /// # Errors
    /// Returns [`StarError`] if arguments are invalid, runtime
    /// `do_redistribute_mesh = true`, storing fails, or ADIPLS fails.
    #[allow(clippy::too_many_arguments)]
    pub fn do_adipls_get_one_el_info(
        &mut self,
        model: &StarModel,
        l: i32,
        nu1: f64,
        nu2: f64,
        iscan: i32,
        r_cm: f64,
        g_cgs: f64,
        m_g: f64,
        delta_nu_model: f64,
        add_atmosphere: bool,
        do_redistribute_mesh: bool,
        store_for_adipls: bool,
        nsel: i32,
        dels: f64,
    ) -> Result<Vec<OscResult>, StarError> {
        let sig_fac = sig_fac_adipls_support(r_cm, g_cgs, m_g);
        self.run_adipls_one_el(
            model,
            l,
            nu1,
            nu2,
            iscan,
            sig_fac,
            delta_nu_model,
            add_atmosphere,
            do_redistribute_mesh,
            store_for_adipls,
            nsel,
            dels,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_adipls_one_el(
        &mut self,
        model: &StarModel,
        l: i32,
        nu1: f64,
        nu2: f64,
        iscan: i32,
        sig_fac: f64,
        delta_nu_model: f64,
        add_atmosphere: bool,
        do_redistribute_mesh: bool,
        store_for_adipls: bool,
        nsel: i32,
        dels: f64,
    ) -> Result<Vec<OscResult>, StarError> {
        if l < 0 {
            return Err(StarError::InvalidInput(format!(
                "ADIPLS requires degree l >= 0, got {l}"
            )));
        }
        if iscan < 1 {
            return Err(StarError::InvalidInput(format!(
                "ADIPLS requires iscan >= 1, got {iscan}"
            )));
        }
        if !nu1.is_finite() || !nu2.is_finite() || nu2 <= nu1 {
            return Err(StarError::InvalidInput(format!(
                "ADIPLS requires finite frequency bounds with nu2 > nu1, got [{nu1}, {nu2}]"
            )));
        }
        if !sig_fac.is_finite() || sig_fac <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "ADIPLS invalid sig_fac={sig_fac}"
            )));
        }
        if !delta_nu_model.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "ADIPLS invalid delta_nu_model={delta_nu_model}"
            )));
        }
        if do_redistribute_mesh {
            return Err(StarError::InvalidInput(String::from(
                "ADIPLS runtime mesh redistribution (srdist / do_redistribute_mesh = .true.) is not ported",
            )));
        }

        if store_for_adipls {
            self.store_model(model, add_atmosphere, do_redistribute_mesh)?;
        }
        let stored = self.stored_model.as_ref().ok_or_else(|| {
            StarError::InvalidInput(String::from(
                "ADIPLS store_for_adipls = false called before store_model",
            ))
        })?;

        let sig1 = sig_from_nu_uhz(sig_fac, nu1);
        let sig2 = sig_from_nu_uhz(sig_fac, nu2);
        let dfsig = dfsig_from_delta_nu(sig_fac, delta_nu_model);

        let mut controls = AdiplsControlState::from_input(&self.input);
        set_adipls_controls(
            &mut controls,
            l,
            nsel,
            f64::from(l),
            dels,
            1,
            iscan,
            sig1,
            sig2,
            dfsig,
            2,
            0,
            0,
            0,
            0,
        );

        let call_out = self
            .solver
            .run_call(&controls, &stored.x, &stored.aa, stored.data)
            .map_err(map_adipls_err)?;
        if call_out.ierr_param != 0 {
            return Err(StarError::MicrophysicsError(format!(
                "ADIPLS returned non-zero ierr_param={}",
                call_out.ierr_param
            )));
        }

        let mut modes: Vec<OscResult> = call_out
            .modes
            .records
            .into_iter()
            .map(|m| OscResult {
                el: m.el,
                order: m.order,
                em: m.em,
                inertia: m.inertia,
                cyclic_freq: m.cyclic_freq_uhz,
                growth_rate: m.growth_rate,
            })
            .collect();
        modes.sort_by(|a, b| a.cyclic_freq.total_cmp(&b.cyclic_freq));
        Ok(modes)
    }
}

fn map_adipls_err(err: AdiplsError) -> StarError {
    match err {
        AdiplsError::Io { .. } => StarError::IoError(err.to_string()),
        other => StarError::InvalidInput(other.to_string()),
    }
}
