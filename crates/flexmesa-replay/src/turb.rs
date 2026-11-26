// SPDX-License-Identifier: GPL-3.0-or-later
//! Replay of captured `set_TDC` and `set_MLT` calls (`turb/public/turb.f90`)
//! through [`flexmesa_turb::set_tdc`] and [`flexmesa_turb::set_mlt`].
//!
//! The capture sits at the kernel boundary inside `Get_results`
//! (`star/private/turb_support.f90:328-334, 349-355, 366-369, 384-387`), so a
//! record holds the complete argument list of a pure function and replay
//! needs no star state. A mismatch therefore localizes a difference to the
//! `turb` kernel; when every call replays identically, a trajectory
//! difference lies upstream of the kernel, in the arguments `FlexMESA`'s
//! `do1_mlt_2` assembles.

use std::str::FromStr;

use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltOption, MltParams, TdcInputs, TdcParams,
};

use crate::compare::CallDiff;
use crate::record::{FormatError, Record};

/// `SCHEMA TDC` field list of `replay_patch.py` (`TDC_SCHEMA`).
pub const TDC_SCHEMA: &str = "site:i id:i model:i solver_iter:i k:i \
    tdc_num_iters_in:i conv_vel_start:r mixing_length_alpha:r TDC_alpha_D:r \
    TDC_alpha_R:r TDC_alpha_Pt:r dt:r cgrav:r m:r scale:r chiT:a chiRho:a \
    gradr:a r:a P:a T:a rho:a dV:a Cp:a opacity:a scale_height:a gradL:a \
    grada:a max_conv_vel:r Eq_div_w:a grav:a include_mlt_corr_to_TDC:l \
    TDC_alpha_C:r TDC_alpha_S:r use_TDC_enthalpy_flux_limiter:l energy:a \
    out_mixing_type:i out_conv_vel:a out_D:a out_Y_face:a out_gradT:a \
    out_tdc_num_iters:i out_ierr:i";

/// `SCHEMA MLT` field list of `replay_patch.py` (`MLT_SCHEMA`).
pub const MLT_SCHEMA: &str = "site:i id:i model:i solver_iter:i k:i \
    MLT_option:s mixing_length_alpha:r Henyey_MLT_nu_param:r \
    Henyey_MLT_y_param:r chiT:a chiRho:a Cp:a grav:a Lambda:a rho:a P:a T:a \
    opacity:a gradr:a grada:a gradL:a max_conv_vel:r out_Gamma:a out_gradT:a \
    out_Y_face:a out_conv_vel:a out_D:a out_mixing_type:i out_ierr:i";

/// Outcome of replaying one record.
pub enum Replayed {
    /// The kernel ran (or failed where MESA also failed); differences found.
    Compared(CallDiff),
    /// The record cannot be replayed through the port as it stands.
    NotReplayed(String),
}

/// Replays one `TDC` record.
///
/// `tdc_num_iters` is compared as captured. Upstream's `get_TDC_solution`
/// leaves its `intent(out)` count unassigned on its radiative exits
/// (`turb/private/tdc.f90:55,113`), so MESA then returns the caller's value
/// (`tdc_num_iters_in`), while [`set_tdc`] returns 0; a difference there is
/// reported, not masked.
///
/// # Errors
/// Returns [`FormatError`] for a record missing a field.
pub fn replay_tdc(rec: &Record) -> Result<Replayed, FormatError> {
    let params = TdcParams {
        conv_vel_start: rec.real("conv_vel_start")?,
        mixing_length_alpha: rec.real("mixing_length_alpha")?,
        tdc_alpha_d: rec.real("TDC_alpha_D")?,
        tdc_alpha_r: rec.real("TDC_alpha_R")?,
        tdc_alpha_pt: rec.real("TDC_alpha_Pt")?,
        tdc_alpha_c: rec.real("TDC_alpha_C")?,
        tdc_alpha_s: rec.real("TDC_alpha_S")?,
        dt: rec.real("dt")?,
        cgrav: rec.real("cgrav")?,
        m: rec.real("m")?,
        scale: rec.real("scale")?,
        max_conv_vel: rec.real("max_conv_vel")?,
        include_mlt_corr_to_tdc: rec.logical("include_mlt_corr_to_TDC")?,
        use_tdc_enthalpy_flux_limiter: rec
            .logical("use_TDC_enthalpy_flux_limiter")?,
    };
    let inp = TdcInputs {
        chi_t: rec.ad("chiT")?,
        chi_rho: rec.ad("chiRho")?,
        gradr: rec.ad("gradr")?,
        r: rec.ad("r")?,
        p: rec.ad("P")?,
        t: rec.ad("T")?,
        rho: rec.ad("rho")?,
        dv: rec.ad("dV")?,
        cp: rec.ad("Cp")?,
        opacity: rec.ad("opacity")?,
        scale_height: rec.ad("scale_height")?,
        gradl: rec.ad("gradL")?,
        grada: rec.ad("grada")?,
        eq_div_w: rec.ad("Eq_div_w")?,
        grav: rec.ad("grav")?,
        energy: rec.ad("energy")?,
    };
    let mesa_ierr = rec.int("out_ierr")?;
    let mut diff = CallDiff::default();
    match set_tdc(&params, &inp) {
        Ok(out) => {
            diff.discrete("ierr", &mesa_ierr, &0);
            if mesa_ierr != 0 {
                return Ok(Replayed::Compared(diff));
            }
            let o = out.output;
            diff.discrete(
                "mixing_type",
                &rec.int("out_mixing_type")?,
                &(o.mixing_type as i64),
            );
            diff.ad("conv_vel", &rec.ad("out_conv_vel")?, &o.conv_vel);
            diff.ad("D", &rec.ad("out_D")?, &o.d);
            diff.ad("Y_face", &rec.ad("out_Y_face")?, &o.y_face);
            diff.ad("gradT", &rec.ad("out_gradT")?, &o.grad_t);
            diff.discrete(
                "tdc_num_iters",
                &rec.int("out_tdc_num_iters")?,
                &i64::try_from(out.num_iters).unwrap_or(i64::MAX),
            );
        }
        Err(e) => {
            // Both failing is agreement on the error path; the outputs of a
            // failed call are not defined upstream.
            if mesa_ierr == 0 {
                diff.discrete("ierr", "0", format!("error: {e}").as_str());
            }
        }
    }
    Ok(Replayed::Compared(diff))
}

/// Replays one `MLT` record.
///
/// # Errors
/// Returns [`FormatError`] for a record missing a field.
pub fn replay_mlt(rec: &Record) -> Result<Replayed, FormatError> {
    let option_name = rec.str("MLT_option")?;
    let Ok(option) = MltOption::from_str(option_name) else {
        return Ok(Replayed::NotReplayed(format!(
            "MLT_option `{option_name}` is not supported by set_mlt"
        )));
    };
    let params = MltParams {
        option,
        mixing_length_alpha: rec.real("mixing_length_alpha")?,
        henyey_nu: rec.real("Henyey_MLT_nu_param")?,
        henyey_y: rec.real("Henyey_MLT_y_param")?,
        max_conv_vel: rec.real("max_conv_vel")?,
    };
    let inp = MltInputs {
        chi_t: rec.ad("chiT")?,
        chi_rho: rec.ad("chiRho")?,
        cp: rec.ad("Cp")?,
        grav: rec.ad("grav")?,
        lambda: rec.ad("Lambda")?,
        rho: rec.ad("rho")?,
        p: rec.ad("P")?,
        t: rec.ad("T")?,
        opacity: rec.ad("opacity")?,
        gradr: rec.ad("gradr")?,
        grada: rec.ad("grada")?,
        gradl: rec.ad("gradL")?,
    };
    let mesa_ierr = rec.int("out_ierr")?;
    let mut diff = CallDiff::default();
    match set_mlt(&params, &inp) {
        Ok(o) => {
            diff.discrete("ierr", &mesa_ierr, &0);
            if mesa_ierr != 0 {
                return Ok(Replayed::Compared(diff));
            }
            diff.discrete(
                "mixing_type",
                &rec.int("out_mixing_type")?,
                &(o.mixing_type as i64),
            );
            diff.ad("Gamma", &rec.ad("out_Gamma")?, &o.gamma);
            diff.ad("gradT", &rec.ad("out_gradT")?, &o.grad_t);
            diff.ad("Y_face", &rec.ad("out_Y_face")?, &o.y_face);
            diff.ad("conv_vel", &rec.ad("out_conv_vel")?, &o.conv_vel);
            diff.ad("D", &rec.ad("out_D")?, &o.d);
        }
        Err(e) => {
            if mesa_ierr == 0 {
                diff.discrete("ierr", "0", format!("error: {e}").as_str());
            }
        }
    }
    Ok(Replayed::Compared(diff))
}
