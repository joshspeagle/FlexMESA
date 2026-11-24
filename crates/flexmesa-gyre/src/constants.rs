// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's run-time constants (`common/constants_m.fypp`) and the math
//! constants `init_math` sets (`math/crmath/math_m.fypp`).

use crate::math_m::MathConstants;
use crate::GyreError;

/// The `SAVE`d constants of `constants_m`, which MESA overrides through
/// `set_constant`, together with `math_m`'s constants.
#[derive(Debug, Clone, PartialEq)]
pub struct GyreConstants {
    /// `G_GRAVITY` (cgs).
    pub g_gravity: f64,
    /// `C_LIGHT` (cm/s).
    pub c_light: f64,
    /// `A_RADIATION` (cgs).
    pub a_radiation: f64,
    /// `M_SUN` (g).
    pub m_sun: f64,
    /// `R_SUN` (cm).
    pub r_sun: f64,
    /// `L_SUN` (erg/s).
    pub l_sun: f64,
    /// `GYRE_DIR`.
    pub gyre_dir: String,
    /// `PI`, `TWOPI`, ... from `init_math`.
    pub math: MathConstants,
}

impl GyreConstants {
    /// Initial values: ForUM's defaults (`forum/src/lib/const_m.fypp:35-58`,
    /// folded at compile time) and `init_math`.
    #[must_use]
    pub fn new() -> Self {
        let g_gravity = 6.674_30e-8;
        let c_light = 2.997_924_58e10;
        let sigma_stefan = 5.670_374_419e-5;
        Self {
            g_gravity,
            c_light,
            a_radiation: 4.0 * sigma_stefan / c_light,
            m_sun: 1.327_124_4e26 / g_gravity,
            r_sun: 6.957e10,
            l_sun: 3.828e33,
            gyre_dir: String::new(),
            math: MathConstants::init(),
        }
    }

    /// `set_constant(name, value)` for a real constant
    /// (`constants_m.fypp:136-164`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Abort`] for an unknown name.
    pub fn set_real(
        &mut self,
        name: &str,
        value: f64,
    ) -> Result<(), GyreError> {
        match name {
            "G_GRAVITY" => self.g_gravity = value,
            "C_LIGHT" => self.c_light = value,
            "A_RADIATION" => self.a_radiation = value,
            "M_SUN" => self.m_sun = value,
            "R_SUN" => self.r_sun = value,
            "L_SUN" => self.l_sun = value,
            _ => return Err(GyreError::Abort("invalid name in set_constant")),
        }
        Ok(())
    }

    /// `set_constant(name, value)` for a character constant
    /// (`constants_m.fypp:168-186`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Abort`] for an unknown name.
    pub fn set_char(
        &mut self,
        name: &str,
        value: &str,
    ) -> Result<(), GyreError> {
        match name {
            "GYRE_DIR" => self.gyre_dir = value.chars().take(256).collect(),
            _ => return Err(GyreError::Abort("invalid name in set_constant")),
        }
        Ok(())
    }
}

impl Default for GyreConstants {
    fn default() -> Self {
        Self::new()
    }
}
