// SPDX-License-Identifier: GPL-3.0-only
//! The narrow MESA-facing interface, equivalent to GYRE's `gyre_mesa_m`
//! (`gyre/gyre/src/lib/gyre_mesa_m.fypp`): `init`, `set_constant`,
//! `set_model` and `get_modes`.
//!
//! Upstream keeps the parameters and the model in module `SAVE`
//! variables; here they live in one [`GyreMesa`] value. The bracket-search
//! mode counter `id_m` stays process-wide (`search.rs`).

use crate::ad::AdBvp;
use crate::constants::GyreConstants;
use crate::context::Context;
use crate::ext_real::ExtR;
use crate::fcomplex::Cplx;
use crate::grid::Grid;
use crate::grid_refine::refine_grid;
use crate::mode::Mode;
use crate::model::EvolModel;
use crate::nad::NadBvp;
use crate::par::{select_par_1, select_par_v, GyrePars, ModePar, NumPar};
use crate::prox::prox_search;
use crate::rad::{GyreTap, NoTap, RadBvp};
use crate::search::{bracket_search, build_scan, check_scan, RealBvp};
use crate::GyreError;

/// `[l, n_p, n_g, n_pg, n]` of a mode (the `MODE_AD`/`MODE_NAD` taps).
fn mode_ints(md: &Mode<'_>) -> [i64; 5] {
    [
        i64::from(md.wave.l),
        i64::from(md.n_p),
        i64::from(md.n_g),
        i64::from(md.n_pg),
        i64::try_from(md.wave.n).unwrap_or(i64::MAX),
    ]
}

/// GYRE's MESA library state.
#[derive(Debug, Clone)]
pub struct GyreMesa {
    /// Constants (`set_constant`).
    pub constants: GyreConstants,
    /// Parameters read by `init`.
    pub pars: GyrePars,
    model: Option<EvolModel>,
}

impl GyreMesa {
    /// `init(file)` (`gyre_mesa_m.fypp:82-112`) from the input file's text:
    /// `init_math`, then the seven `read_*_par` calls.
    ///
    /// # Errors
    ///
    /// Namelist errors.
    pub fn init(gyre_in: &str) -> Result<Self, GyreError> {
        Ok(Self {
            constants: GyreConstants::new(),
            pars: GyrePars::parse(gyre_in)?,
            model: None,
        })
    }

    /// `set_constant(name, value)` for a real constant.
    ///
    /// # Errors
    ///
    /// Unknown constant names.
    pub fn set_constant(
        &mut self,
        name: &str,
        value: f64,
    ) -> Result<(), GyreError> {
        self.constants.set_real(name, value)
    }

    /// `set_constant(name, value)` for a character constant.
    ///
    /// # Errors
    ///
    /// Unknown constant names.
    pub fn set_constant_str(
        &mut self,
        name: &str,
        value: &str,
    ) -> Result<(), GyreError> {
        self.constants.set_char(name, value)
    }

    /// `set_model(global_data, point_data, version)`
    /// (`gyre_mesa_m.fypp:224-240`). `point_data[k]` holds the columns of
    /// point `k` (`point_data(:, k+1)` upstream).
    ///
    /// # Errors
    ///
    /// Model construction errors.
    pub fn set_model(
        &mut self,
        global_data: &[f64],
        point_data: &[Vec<f64>],
        version: i32,
    ) -> Result<(), GyreError> {
        self.model = None;
        self.model = Some(EvolModel::from_mesa(
            &self.pars.model,
            global_data,
            point_data,
            version,
            &self.constants,
        )?);
        Ok(())
    }

    /// The current model.
    #[must_use]
    pub fn model(&self) -> Option<&EvolModel> {
        self.model.as_ref()
    }

    /// `get_modes(l, user_sub, ipar, rpar)` (`gyre_mesa_m.fypp:244-482`).
    /// `user_sub` returns the upstream `retcode`, which `gyre_mesa_m` ignores.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for nonadiabatic searches other than
    /// `nad_search = 'AD'` and unported model/boundary/solver options;
    /// errors of every stage.
    pub fn get_modes(
        &mut self,
        l: i32,
        user_sub: &mut dyn FnMut(&Mode<'_>) -> Result<i32, GyreError>,
    ) -> Result<(), GyreError> {
        self.get_modes_tapped(l, user_sub, &mut NoTap)
    }

    /// [`Self::get_modes`] with an observer of intermediate values.
    ///
    /// # Errors
    ///
    /// As [`Self::get_modes`].
    #[allow(clippy::too_many_lines)]
    pub fn get_modes_tapped(
        &mut self,
        l: i32,
        user_sub: &mut dyn FnMut(&Mode<'_>) -> Result<i32, GyreError>,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError> {
        if self.constants.gyre_dir.is_empty() {
            return Err(GyreError::Assert(
                "The GYRE_DIR environment variable is not set",
            ));
        }
        let ml = self
            .model
            .as_ref()
            .ok_or(GyreError::Assert("no model has been set"))?;
        let k = &self.constants;
        let pars = &self.pars;
        for (i, md_p) in pars.mode.iter().enumerate() {
            if md_p.l != l {
                continue;
            }
            tap.mode_par(md_p.l, i + 1);
            let os_p = select_par_1(&pars.osc, &md_p.tag)?;
            let rt_p = select_par_1(&pars.rot, &md_p.tag)?;
            let nm_p = select_par_1(&pars.num, &md_p.tag)?;
            let gr_p = select_par_1(&pars.grid, &md_p.tag)?;
            let sc_p = select_par_v(&pars.scan, &md_p.tag)?;

            // gr = grid_t(ml_m, gr_p_sel) (grid_scaffold_m.fypp:54-82)
            let mut gr = match gr_p.scaffold_src.as_str() {
                "MODEL" => Grid::nest(&ml.gr, gr_p.x_i, gr_p.x_o),
                "FILE" => {
                    return Err(GyreError::Unported {
                        what: "scaffold_src = 'FILE'",
                        at: "grid/grid_scaffold_m.fypp:68-71",
                    })
                }
                _ => return Err(GyreError::Abort("invalid scaffold_src")),
            };
            let cx = Context::new(ml, k, &gr, md_p, &os_p, &rt_p)?;
            let omega_re = build_scan(&cx, &sc_p, "REAL")?;
            let omega_im = build_scan(&cx, &sc_p, "IMAG")?;
            let _ = omega_im;
            refine_grid(&mut gr, &cx, &omega_re, &gr_p, &os_p)?;
            tap.grid_scan(&gr, &omega_re);
            let (omega_min, omega_max) = if nm_p.restrict_roots {
                (
                    omega_re.iter().copied().fold(f64::MAX, |a, b| {
                        if b < a {
                            b
                        } else {
                            a
                        }
                    }),
                    omega_re.iter().copied().fold(-f64::MAX, |a, b| {
                        if b > a {
                            b
                        } else {
                            a
                        }
                    }),
                )
            } else {
                (-f64::MAX, f64::MAX)
            };
            check_scan(&cx, &gr, &omega_re, md_p)?;
            let nonad = os_p.nonadiabatic;
            // Adiabatic modes stored for the nonadiabatic search
            // (omega_ad(:n_ad), id_ad(:n_ad); gyre_mesa_m.fypp:336-340).
            let mut omega_ad: Vec<Cplx> = Vec::new();
            let mut id_ad: Vec<i32> = Vec::new();
            if os_p.adiabatic {
                if md_p.l == 0 && os_p.reduce_order {
                    let mut bp = RadBvp::new(&cx, &gr, md_p, &nm_p, &os_p)?;
                    run_ad_bracket(
                        &mut bp,
                        md_p,
                        &nm_p,
                        &omega_re,
                        omega_min,
                        omega_max,
                        nonad,
                        &mut omega_ad,
                        &mut id_ad,
                        user_sub,
                        tap,
                    )?;
                } else {
                    let mut bp = AdBvp::new(&cx, &gr, md_p, &nm_p, &os_p)?;
                    run_ad_bracket(
                        &mut bp,
                        md_p,
                        &nm_p,
                        &omega_re,
                        omega_min,
                        omega_max,
                        nonad,
                        &mut omega_ad,
                        &mut id_ad,
                        user_sub,
                        tap,
                    )?;
                }
            }
            if nonad {
                let mut bp = NadBvp::new(&cx, &gr, md_p, &nm_p, &os_p)?;
                match nm_p.nad_search.as_str() {
                    "AD" => {
                        if !os_p.adiabatic {
                            return Err(GyreError::Assert(
                                "No adiabatic modes to start from",
                            ));
                        }
                        let mut mode_taps: Vec<(Cplx, [i64; 5])> = Vec::new();
                        // process_mode_nad (gyre_mesa_m.fypp:450-470)
                        let mut process = |md: &Mode<'_>, _n_iter: i32, _chi: ExtR| -> Result<(), GyreError> {
                            mode_taps.push((md.wave.omega, mode_ints(md)));
                            let _retcode = user_sub(md)?;
                            Ok(())
                        };
                        prox_search(
                            &mut bp,
                            &omega_ad,
                            &id_ad,
                            omega_min,
                            omega_max,
                            &nm_p,
                            tap,
                            &mut process,
                        )?;
                        drop(process);
                        for (w, ints) in mode_taps {
                            tap.mode_nad(w, ints);
                        }
                    }
                    "MINMOD" | "CONTOUR" => {
                        return Err(GyreError::Unported {
                            what: "nad_search = 'MINMOD' or 'CONTOUR'",
                            at: "lib/gyre_mesa_m.fypp:381-392",
                        })
                    }
                    _ => return Err(GyreError::Abort("Invalid nad_start")),
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ad_bracket<'a, B: RealBvp<'a>>(
    bp: &mut B,
    md_p: &ModePar,
    nm_p: &NumPar,
    omega_re: &[f64],
    omega_min: f64,
    omega_max: f64,
    nonad: bool,
    omega_ad: &mut Vec<Cplx>,
    id_ad: &mut Vec<i32>,
    user_sub: &mut dyn FnMut(&Mode<'_>) -> Result<i32, GyreError>,
    tap: &mut dyn GyreTap,
) -> Result<(), GyreError> {
    match nm_p.ad_search.as_str() {
        "BRACKET" => {
            if omega_re.len() > 2 {
                let n_pg_min = md_p.n_pg_min;
                let n_pg_max = md_p.n_pg_max;
                let mut mode_taps: Vec<(Cplx, [i64; 5])> = Vec::new();
                // process_mode_ad (gyre_mesa_m.fypp:408-446)
                let mut process = |md: &Mode<'_>,
                                   _n_iter: i32,
                                   _chi: ExtR|
                 -> Result<(), GyreError> {
                    if md.n_pg < n_pg_min || md.n_pg > n_pg_max {
                        return Ok(());
                    }
                    mode_taps.push((md.wave.omega, mode_ints(md)));
                    if nonad {
                        omega_ad.push(md.wave.omega);
                        id_ad.push(md.wave.id);
                    } else {
                        let _retcode = user_sub(md)?;
                    }
                    Ok(())
                };
                bracket_search(
                    bp,
                    omega_re,
                    omega_min,
                    omega_max,
                    nm_p,
                    tap,
                    &mut process,
                )?;
                drop(process);
                for (w, ints) in mode_taps {
                    tap.mode_ad(w, ints);
                }
            }
            Ok(())
        }
        _ => Err(GyreError::Abort("Invalid ad_search")),
    }
}
