// SPDX-License-Identifier: GPL-3.0-only
//! The narrow MESA-facing interface, equivalent to GYRE's `gyre_mesa_m`
//! (`gyre/gyre/src/lib/gyre_mesa_m.fypp`): `init`, `set_constant`,
//! `set_model` and `get_modes`.
//!
//! Upstream keeps the parameters and the model in module `SAVE`
//! variables; here they live in one [`GyreMesa`] value. The bracket-search
//! mode counter `id_m` stays process-wide (`search.rs`).

use crate::constants::GyreConstants;
use crate::context::Context;
use crate::ext_real::ExtR;
use crate::grid::Grid;
use crate::grid_refine::refine_grid;
use crate::mode::Mode;
use crate::model::EvolModel;
use crate::par::{select_par_1, select_par_v, GyrePars};
use crate::rad::{GyreTap, NoTap, RadBvp};
use crate::search::{bracket_search, build_scan, check_scan};
use crate::GyreError;

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

    /// `get_modes(l, user_sub, ipar, rpar)` (`gyre_mesa_m.fypp:244-482`)
    /// for adiabatic radial modes. `user_sub` returns the upstream
    /// `retcode`, which `gyre_mesa_m` ignores.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for non-radial or non-adiabatic searches;
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
            if os_p.adiabatic {
                if !(md_p.l == 0 && os_p.reduce_order) {
                    return Err(GyreError::Unported {
                        what: "adiabatic non-radial (ad_bvp_t) searches",
                        at: "lib/gyre_mesa_m.fypp:345-347",
                    });
                }
                let mut bp = RadBvp::new(&cx, &gr, md_p, &nm_p, &os_p)?;
                match nm_p.ad_search.as_str() {
                    "BRACKET" => {
                        if omega_re.len() > 2 {
                            let n_pg_min = md_p.n_pg_min;
                            let n_pg_max = md_p.n_pg_max;
                            let nonad = os_p.nonadiabatic;
                            let mut process = |md: &Mode<'_>, _n_iter: i32, _chi: ExtR| -> Result<(), GyreError> {
                                if md.n_pg < n_pg_min || md.n_pg > n_pg_max {
                                    return Ok(());
                                }
                                if nonad {
                                    return Err(GyreError::Unported {
                                        what: "nonadiabatic mode processing",
                                        at: "lib/gyre_mesa_m.fypp:433-446",
                                    });
                                }
                                let _retcode = user_sub(md)?;
                                Ok(())
                            };
                            bracket_search(
                                &mut bp,
                                &omega_re,
                                omega_min,
                                omega_max,
                                &nm_p,
                                tap,
                                &mut process,
                            )?;
                        }
                    }
                    _ => return Err(GyreError::Abort("Invalid ad_search")),
                }
            }
            if os_p.nonadiabatic {
                return Err(GyreError::Unported {
                    what: "nonadiabatic searches",
                    at: "lib/gyre_mesa_m.fypp:369-400",
                });
            }
        }
        Ok(())
    }
}
