// SPDX-License-Identifier: GPL-3.0-or-later
//! Python access to GYRE's MESA interface (`gyre/gyre/src/lib/gyre_mesa_m.fypp`,
//! ported in `flexmesa_gyre::mesa`; #170) for test-side star extras that
//! call `init`, `set_constant`, `set_model` and `get_modes` themselves, as
//! the `gyre_in_mesa_*` cases' `run_star_extras.f90` does.

use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::Mode;
use flexmesa_gyre::GyreError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::astero_gyre_py::PyStarModelHandle;

fn gyre_err(e: &GyreError) -> PyErr {
    PyRuntimeError::new_err(format!("GYRE: {e}"))
}

/// One mode passed to `user_sub`: `(n_pg, freq('HZ'))`, the frequency as
/// `(re, im)`.
type ModeInfo = (i32, (f64, f64));

/// `gyre_mesa_m`'s module state for one run.
#[pyclass(name = "GyreMesa", module = "flexmesa._rust", unsendable)]
pub struct PyGyreMesa {
    inner: GyreMesa,
}

#[pymethods]
impl PyGyreMesa {
    /// `init(file)` (`gyre_mesa_m.fypp:82-112`): read the GYRE namelist
    /// file.
    #[new]
    fn new(file: &str) -> PyResult<Self> {
        let text = std::fs::read_to_string(file).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to read GYRE input file '{file}': {e}"
            ))
        })?;
        let inner = GyreMesa::init(&text).map_err(|e| gyre_err(&e))?;
        Ok(Self { inner })
    }

    /// `set_constant(name, value)` for a real constant.
    fn set_constant(&mut self, name: &str, value: f64) -> PyResult<()> {
        self.inner
            .set_constant(name, value)
            .map_err(|e| gyre_err(&e))
    }

    /// `set_constant(name, value)` for a character constant.
    fn set_constant_str(&mut self, name: &str, value: &str) -> PyResult<()> {
        self.inner
            .set_constant_str(name, value)
            .map_err(|e| gyre_err(&e))
    }

    /// The `set_constant` calls of the `gyre_in_mesa_*` cases'
    /// `extras_startup` (`star/test_suite/gyre_in_mesa_ms/src/run_star_extras.f90:86-94`):
    /// `standard_cgrav`, `clight`, `crad`, `Msun`, `Rsun`, `Lsun` from
    /// `const_def`, and `GYRE_DIR = mesa_dir//'/gyre/gyre'`.
    fn set_mesa_constants(&mut self, mesa_dir: &str) -> PyResult<()> {
        use flexmesa_const::{CLIGHT, CRAD, LSUN, MSUN, RSUN, STANDARD_CGRAV};
        for (name, value) in [
            ("G_GRAVITY", STANDARD_CGRAV),
            ("C_LIGHT", CLIGHT),
            ("A_RADIATION", CRAD),
            ("M_SUN", MSUN),
            ("R_SUN", RSUN),
            ("L_SUN", LSUN),
        ] {
            self.set_constant(name, value)?;
        }
        self.set_constant_str(
            "GYRE_DIR",
            &format!("{}/gyre/gyre", mesa_dir.trim_end()),
        )
    }

    /// `star_get_pulse_data(id, 'GYRE', add_center_point,
    /// keep_surface_point, add_atmosphere, global_data, point_data, ierr)`
    /// followed by `set_model(global_data, point_data, s%gyre_data_schema)`
    /// (`star/job/gyre_in_mesa_extras_finish_step.inc:51-57`).
    fn set_model(
        &mut self,
        model: &PyStarModelHandle,
        add_center_point: bool,
        keep_surface_point: bool,
        add_atmosphere: bool,
    ) -> PyResult<()> {
        let (global_data, point_data) =
            flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre(
                &model.model,
                add_center_point,
                keep_surface_point,
                add_atmosphere,
            )
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "star_get_pulse_data('GYRE'): {e}"
                ))
            })?;
        self.inner
            .set_model(
                &global_data,
                &point_data,
                model.model.controls.gyre_data_schema,
            )
            .map_err(|e| gyre_err(&e))
    }

    /// `get_modes(l, user_sub, ipar, rpar)` (`gyre_mesa_m.fypp:244-482`):
    /// the `(n_pg, freq('HZ'))` of every mode passed to `user_sub`, in call
    /// order.
    fn get_modes(&mut self, l: i32) -> PyResult<Vec<ModeInfo>> {
        let mut out: Vec<ModeInfo> = Vec::new();
        let mut cb = |md: &Mode<'_>| -> Result<i32, GyreError> {
            let f = md.freq("HZ")?;
            out.push((md.n_pg, (f.re, f.im)));
            Ok(0)
        };
        self.inner.get_modes(l, &mut cb).map_err(|e| gyre_err(&e))?;
        Ok(out)
    }
}

/// Registers the class.
///
/// # Errors
/// Returns [`PyErr`] if registration fails.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGyreMesa>()?;
    Ok(())
}
