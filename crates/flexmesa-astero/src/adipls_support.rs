// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream `adipls_support` bridge connecting `flexmesa-star` models and
//! [`AsteroSearchControls`] to the pure-Rust `flexmesa-adipls` oscillation
//! solver (`astero/private/adipls_support.f90` and
//! `astero/private/astero_support.f90:108-156`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flexmesa_adipls::{
    dfsig_from_delta_nu, fgong_amdl, parse_adipls_pruned_file,
    parse_redistrb_pruned_file, resolve_store_model_flags,
    sig_fac_adipls_support, sig_fac_astero_support, sig_from_nu_uhz,
    store_amdl, AdiplsCallerFlags, AdiplsControlState, AdiplsError,
    AdiplsInput, AdiplsSolver, AsteroDefPulseFlags, DefaultSigoutHandler,
    FgongModel, NoopTapObserver, RedistrbInput, SetAdiplsControlsArgs,
    SigoutArgs, SigoutHandler, StoreAmdlOutput,
};
use flexmesa_const::{RSUN, STANDARD_CGRAV};
use flexmesa_star::controls::StarError;
use flexmesa_star::eval_photosphere_info;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse::default_fgong_header;
use flexmesa_star::pulse_fgong::{get_fgong_data, PulseDataInput};
use flexmesa_star::pulse_gyre_data::GyreDataError;
use flexmesa_star::pulse_write::{fortran_1pe, write_fgong_data, FGONG_IVAR};

use crate::gyre_support::OscResult;
use crate::inlist_astero::AsteroSearchControls;

/// Stateful ADIPLS asteroseismic solver driver matching MESA's
/// `adipls_support` lifecycle (`init`, `store_model_for_adipls`,
/// `do_adipls_get_one_el_info`, `get_one_el_info`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct AsteroAdipls {
    solver: AdiplsSolver,
    redistrb_input: Option<RedistrbInput>,
    astero_def_flags: AsteroDefPulseFlags,
    default_add_atmosphere: bool,
    default_do_redistribute_mesh: bool,
    stored_model: Option<StoreAmdlOutput>,
    adipls_irotkr: i32,
    adipls_nprtkr: i32,
    adipls_igm1kr: i32,
    adipls_npgmkr: i32,
    write_fgong_for_each_model: bool,
    fgong_prefix: String,
    fgong_postfix: String,
    model_num_digits: usize,
    astero_results_dir: PathBuf,
    mode_output_dir: PathBuf,
    save_mode_model_number: isize,
    save_mode_filename: String,
    el_to_save: i32,
    order_to_save: i32,
    trace_time_in_oscillation_code: bool,
    last_oscillation_elapsed_sec: Option<f64>,
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
        if !ctrl.oscillation_code.trim().eq_ignore_ascii_case("adipls") {
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

        let solver = AdiplsSolver::new(input);
        Ok(Self {
            solver,
            redistrb_input,
            astero_def_flags: AsteroDefPulseFlags {
                add_center_point: ctrl.add_center_point,
                keep_surface_point: ctrl.keep_surface_point,
            },
            default_add_atmosphere: ctrl.add_atmosphere,
            default_do_redistribute_mesh: do_redistribute_mesh,
            stored_model: None,
            adipls_irotkr: ctrl.adipls_irotkr,
            adipls_nprtkr: ctrl.adipls_nprtkr,
            adipls_igm1kr: ctrl.adipls_igm1kr,
            adipls_npgmkr: ctrl.adipls_npgmkr,
            write_fgong_for_each_model: ctrl.write_fgong_for_each_model,
            fgong_prefix: ctrl.fgong_prefix.clone(),
            fgong_postfix: ctrl.fgong_postfix.clone(),
            model_num_digits: ctrl.model_num_digits,
            astero_results_dir: case_dir.join(&ctrl.astero_results_directory),
            mode_output_dir: case_dir.to_path_buf(),
            save_mode_model_number: ctrl.save_mode_model_number,
            save_mode_filename: ctrl.save_mode_filename.clone(),
            el_to_save: ctrl.el_to_save,
            order_to_save: ctrl.order_to_save,
            trace_time_in_oscillation_code: ctrl.trace_time_in_oscillation_code,
            last_oscillation_elapsed_sec: None,
        })
    }

    /// Redirects per-model FGONG and eigenfunction output roots (e.g. when
    /// running under an explicit `output_root`).
    pub fn set_output_root(
        &mut self,
        output_root: &Path,
        results_subdir: &str,
    ) {
        self.mode_output_dir = output_root.to_path_buf();
        self.astero_results_dir = output_root.join(results_subdir);
    }

    /// Wall-clock duration in seconds of the most recent ADIPLS solver call,
    /// populated on every [`Self::get_one_el_info`] / [`Self::do_adipls_get_one_el_info`].
    #[must_use]
    pub const fn last_oscillation_elapsed_sec(&self) -> Option<f64> {
        self.last_oscillation_elapsed_sec
    }

    /// Borrow the underlying [`AdiplsInput`] parsed from `adipls.c.pruned.in`.
    #[must_use]
    pub const fn input(&self) -> &AdiplsInput {
        self.solver.input()
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

    /// Convert `model` into an ADIPLS AMDL buffer via `pulse_fgong` ->
    /// `fgong_amdl` -> `store_amdl` and cache it (`store_model_for_adipls` in
    /// `astero/private/adipls_support.f90:217-327`).
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

        let pulse_in =
            PulseDataInput::from_model(model).map_err(|e| map_pulse_err(&e))?;
        let (global_data, point_data) = get_fgong_data(
            &pulse_in,
            resolved.add_center_point,
            resolved.keep_surface_point,
            resolved.add_atmosphere,
        )
        .map_err(|e| map_pulse_err(&e))?;

        if self.write_fgong_for_each_model {
            fs::create_dir_all(&self.astero_results_dir).map_err(|e| {
                StarError::IoError(format!(
                    "failed creating astero_results_directory '{}': {e}",
                    self.astero_results_dir.display()
                ))
            })?;
            let num_str = format!(
                "{:0width$}",
                model.model_number,
                width = self.model_num_digits
            );
            let fgong_path = self.astero_results_dir.join(format!(
                "{}{}{}",
                self.fgong_prefix.trim(),
                num_str,
                self.fgong_postfix.trim()
            ));
            let text = write_fgong_data(
                &global_data,
                &point_data,
                &default_fgong_header(),
                model.controls.fgong_ivers,
            )
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
            fs::write(&fgong_path, text).map_err(|e| {
                StarError::IoError(format!(
                    "failed writing FGONG file '{}': {e}",
                    fgong_path.display()
                ))
            })?;
        }

        let nn = point_data.len();
        let mut flat = Vec::with_capacity(nn * FGONG_IVAR);
        for col in &point_data {
            flat.extend_from_slice(col);
        }

        let fgong_model = FgongModel::new(global_data, FGONG_IVAR, nn, flat)
            .map_err(map_adipls_err)?;

        let input = self.solver.input();
        let fg_amdl = fgong_amdl(input.cst.cgrav, &fgong_model)
            .map_err(map_adipls_err)?;

        let st = store_amdl(
            fg_amdl.nn,
            input.int_ctrl.iriche,
            input.int_ctrl.iturpr,
            fg_amdl.data,
            &fg_amdl.aa,
        )
        .map_err(map_adipls_err)?;

        Ok(self.stored_model.insert(st))
    }

    /// Evaluate all ADIPLS modes for degree `l` in `[nu1, nu2]` $\mu\mathrm{Hz}$
    /// following `astero/private/astero_support.f90:108-156`, returning results
    /// sorted by ascending `cyclic_freq`.
    ///
    /// Uses `R = RSUN * eval_photosphere_info(model).r_rsun`,
    /// `G = STANDARD_CGRAV`, `M = model.m_grav[0]`, and
    /// `sig_fac_astero_support(R, G, M)`.
    ///
    /// # Errors
    /// Returns [`StarError`] if geometry is invalid,
    /// `do_redistribute_mesh = true` was configured at startup (`srdist` not
    /// ported), storing fails, `store_for_adipls = false` before any model was
    /// stored, or ADIPLS fails.
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
        let r_cm = RSUN * eval_photosphere_info(model).r_rsun;
        let g_cgs = STANDARD_CGRAV;
        let m_g = model.m_grav[0];
        let sig_fac = sig_fac_astero_support(r_cm, g_cgs, m_g);
        // `astero/private/astero_support.f90:134` passes `do_redistribute_mesh`
        // (default `.true.` in `astero_search.defaults:1363`), not `.false.`.
        self.run_adipls_one_el(
            model,
            el,
            nu1,
            nu2,
            iscan,
            sig_fac,
            delta_nu_model,
            self.default_add_atmosphere,
            self.default_do_redistribute_mesh,
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
        let sig1 = sig_from_nu_uhz(sig_fac, nu1);
        let sig2 = sig_from_nu_uhz(sig_fac, nu2);
        let dfsig = dfsig_from_delta_nu(sig_fac, delta_nu_model);

        let mut controls = AdiplsControlState::from_input(self.solver.input());
        controls.apply_controls(&SetAdiplsControlsArgs {
            el: l,
            nsel,
            els1: f64::from(l),
            dels,
            itrsig: 1,
            iscan,
            sig1,
            sig2,
            dfsig,
            nsig: 2,
            irotkr: self.adipls_irotkr,
            nprtkr: self.adipls_nprtkr,
            igm1kr: self.adipls_igm1kr,
            npgmkr: self.adipls_npgmkr,
        });

        let stored = self.stored_model.as_ref().ok_or_else(|| {
            StarError::InvalidInput(String::from(
                "ADIPLS store_for_adipls = false called before store_model",
            ))
        })?;
        let save_active = self.save_mode_model_number > 0
            && model.model_number.cast_signed() == self.save_mode_model_number
            && l == self.el_to_save;
        let save_path = resolve_mode_save_path(
            &self.mode_output_dir,
            &self.save_mode_filename,
        );
        let mut itrsig = controls.cadi_param.itrsig;
        let iscan_eff = controls.cadi_param.iscan;
        if iscan_eff > 1 && itrsig > 1 {
            itrsig = 1;
        }
        let inp = self.solver.input();
        let nfmod1 = inp.out.nfmode / 10;
        let nfmode_lo = inp.out.nfmode % 10;
        let nfmscn = i32::from(
            (iscan_eff > 1 && itrsig != 1 && nfmode_lo > 0)
                || (inp.int_ctrl.itmax == 0 && nfmod1 == 1),
        );
        let mut handler = AdiplsModeSaveSigoutHandler {
            inner: DefaultSigoutHandler::new(
                inp,
                controls.coutpt.irotkr,
                controls.coutpt.nprtkr,
                controls.coutpt.igm1kr,
                controls.coutpt.npgmkr,
                nfmscn,
                self.solver.setup().nobs_stmx,
            ),
            save_active,
            target_el: self.el_to_save,
            target_order: self.order_to_save,
            save_path,
            save_err: None,
        };
        let t0 = Instant::now();
        let mut obs = NoopTapObserver;
        let call_out = self
            .solver
            .run_call_with_hooks(
                &controls,
                &stored.x,
                &stored.aa,
                stored.data,
                &mut obs,
                &mut handler,
            )
            .map_err(map_adipls_err)?;
        let elapsed = t0.elapsed().as_secs_f64();
        self.last_oscillation_elapsed_sec = Some(elapsed);
        if self.trace_time_in_oscillation_code {
            eprintln!("time in adipls: {elapsed:.6} s");
        }
        if let Some(err) = handler.save_err.take() {
            return Err(err);
        }
        if call_out.ierr_param != 0 {
            return Err(StarError::MicrophysicsError(format!(
                "ADIPLS returned non-zero ierr_param={}",
                call_out.ierr_param
            )));
        }

        let mut modes: Vec<OscResult> = handler
            .inner
            .state
            .mode_store
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

fn resolve_mode_save_path(base_dir: &Path, raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    let rel = if trimmed.is_empty() {
        "save_mode.data"
    } else {
        trimmed
    };
    let p = Path::new(rel);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

struct AdiplsModeSaveSigoutHandler {
    inner: DefaultSigoutHandler,
    save_active: bool,
    target_el: i32,
    target_order: i32,
    save_path: PathBuf,
    save_err: Option<StarError>,
}

impl SigoutHandler for AdiplsModeSaveSigoutHandler {
    fn on_sigout(
        &mut self,
        args: &mut SigoutArgs<'_>,
    ) -> Result<(), AdiplsError> {
        let before = self.inner.state.mode_store.records.len();
        self.inner.on_sigout(args)?;
        if !self.save_active || self.save_err.is_some() {
            return Ok(());
        }
        let after = self.inner.state.mode_store.records.len();
        if after <= before {
            return Ok(());
        }
        let rec = &self.inner.state.mode_store.records[after - 1];
        if rec.el != self.target_el || rec.order != self.target_order {
            return Ok(());
        }
        let skip =
            usize::from(args.x.first().is_some_and(|v| v.abs() < 1.0e-20));
        let nn = args.nn.min(args.x.len());
        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "{:>8}{:>8}{:>8}{:>20}{:>20}{:>20}",
            "el", "order", "em", "freq (microHz)", "inertia", "nn"
        );
        let _ = writeln!(
            buf,
            "{:8}{:8}{:8}{:20.10}{}{:20}",
            rec.el,
            rec.order,
            rec.em,
            rec.cyclic_freq_uhz,
            fortran_1pe(rec.inertia, 20, 10, None),
            nn.saturating_sub(skip)
        );
        let _ = writeln!(
            buf,
            "x = r/R;  y_r = xi_r/R;  y_h = xi_h*l*(l+1)/R;  displacement xi normalized to xi_r = R at surface."
        );
        let _ =
            writeln!(buf, "{:>6}{:>26}{:>26}{:>26}", "i", "x", "y_r", "y_h");
        for idx in skip..nn {
            let base = idx * args.iy;
            let y_r = args.y.get(base).copied().unwrap_or(0.0);
            let y_h = if rec.el > 0 {
                args.y.get(base + 1).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            let _ = writeln!(
                buf,
                "{:6}{}{}{}",
                idx - skip + 1,
                fortran_1pe(args.x[idx], 26, 16, None),
                fortran_1pe(y_r, 26, 16, None),
                fortran_1pe(y_h, 26, 16, None),
            );
        }
        if let Some(parent) = self.save_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                self.save_err = Some(StarError::IoError(format!(
                    "failed creating save_mode_filename parent '{}': {e}",
                    parent.display()
                )));
                return Ok(());
            }
        }
        if let Err(e) = fs::write(&self.save_path, buf) {
            self.save_err = Some(StarError::IoError(format!(
                "failed writing save_mode_filename '{}': {e}",
                self.save_path.display()
            )));
        }
        Ok(())
    }
}

fn map_adipls_err(err: AdiplsError) -> StarError {
    match err {
        AdiplsError::Io { .. } => StarError::IoError(err.to_string()),
        other => StarError::InvalidInput(other.to_string()),
    }
}

fn map_pulse_err(err: &GyreDataError) -> StarError {
    match err {
        GyreDataError::Unported { .. } => {
            StarError::InvalidInput(err.to_string())
        }
        GyreDataError::Upstream(msg) => {
            StarError::MicrophysicsError(String::from(*msg))
        }
    }
}
