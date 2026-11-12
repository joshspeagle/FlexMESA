// SPDX-License-Identifier: LGPL-3.0-or-later
//! Complex arithmetic helper (`Cpx`) and ASCII log writers (`LINA_period_growth.data`,
//! `LINA_work*.data`, `LINA_eigen*.data`) for `do_LINA` (`rsp_lina.f90:1010-1166`).

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::controls::StarError;
use crate::rsp_grid::RspEnvGrid;

/// Minimal double-precision complex scalar for `LINA` eigenfunction post-processing.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Cpx {
    pub re: f64,
    pub im: f64,
}

impl Cpx {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    pub fn div(self, rhs: Self) -> Self {
        let d = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }

    pub fn conj_mul_imag(self, rhs: Self) -> f64 {
        self.re * rhs.im - self.im * rhs.re
    }
}

/// Writes `LINA_period_growth.data` (`rsp_build.f90:234-245`).
pub(crate) fn write_lina_summary(
    dir: &Path,
    periods_sec: &[f64],
    growth_rates: &[f64],
) -> Result<(), StarError> {
    let _ = fs::create_dir_all(dir);
    let mut s = String::new();
    for (idx, (&p_s, &g)) in
        periods_sec.iter().zip(growth_rates.iter()).enumerate()
    {
        let _ = writeln!(s, "{idx:3}  {:16.5E}{:16.5E}", p_s / 86_400.0, g);
    }
    fs::write(dir.join("LINA_period_growth.data"), s).map_err(|e| {
        StarError::IoError(format!("write LINA_period_growth.data: {e}"))
    })
}

/// Writes `LINA_work<mode>.data` and `LINA_eigen<mode>.data` (`rsp_lina.f90:1010-1166`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_lina_mode_logs(
    dir: &Path,
    mode_1b: usize,
    nzn: usize,
    l0: f64,
    ek: f64,
    normc: f64,
    scale: Cpx,
    grid: &RspEnvGrid,
    elr: &[f64],
    elrm: &[f64],
    elrp: &[f64],
    elt: &[f64],
    eltp: &[f64],
    elz0: &[f64],
    elzp: &[f64],
    vrr: &[Cpx],
    vrt: &[Cpx],
    vrc: &[Cpx],
    qwk: &[f64],
    qwkev: &[f64],
    qwkpt: &[f64],
) -> Result<(), StarError> {
    let _ = fs::create_dir_all(dir);
    let mut w_buf = String::from(
        "#ZONE  log(T)         X              WORK(P)         WORK(P_NU)      WORK(P_T)        CWORK(P)        CWORK(P_NU)        CWORK(P_T)\n",
    );
    let (mut cw_p, mut cw_ev, mut cw_pt) = (0.0, 0.0, 0.0);
    for i in 1..=nzn {
        cw_p += qwk[i];
        cw_ev += qwkev[i];
        cw_pt += qwkpt[i];
        let _ = writeln!(
            w_buf,
            "{:3} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E}",
            nzn + 1 - i,
            grid.t[i].log10(),
            grid.r[i] / grid.r[nzn],
            qwk[i] / ek,
            qwkev[i] / ek,
            qwkpt[i] / ek,
            cw_p / ek,
            cw_ev / ek,
            cw_pt / ek,
        );
    }
    let _ = writeln!(w_buf, "#KINETIC ENERGY:{ek:15.8E}");
    fs::write(dir.join(format!("LINA_work{mode_1b}.data")), w_buf)
        .map_err(|e| StarError::IoError(format!("write LINA_work: {e}")))?;

    let mut e_buf = String::from(
        "#ZONE  TEMP.           FRAC. RADIUS  ABS(dR/R)       PH(dR/R)        ABS(dT/T)       PH(dT/T)        ABS(dL/L)       PH(dL/L)        ABS(dE_T)       PH(dE_T)\n",
    );
    for i in 1..=nzn {
        let mut vrl = vrt[i]
            .scale(elt[i])
            .add(vrr[i].scale(elr[i]))
            .add(vrc[i].scale(elz0[i]));
        if i > 1 {
            vrl = vrl.add(vrr[i - 1].scale(elrm[i]));
        }
        if i < nzn {
            vrl = vrl
                .add(vrt[i + 1].scale(eltp[i]))
                .add(vrr[i + 1].scale(elrp[i]))
                .add(vrc[i + 1].scale(elzp[i]));
        }
        let dvrr = vrr[i].mul(scale).scale(1.0 / grid.r[i]);
        let dvrt = vrt[i].mul(scale).scale(1.0 / grid.t[i]);
        let dvrl = vrl.mul(scale).scale(1.0 / l0);
        let _ = writeln!(
            e_buf,
            "{:3} {:15.6E} {:15.6E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E} {:15.8E}",
            nzn + 1 - i,
            grid.t[i],
            grid.r[i] / grid.r[nzn],
            dvrr.abs(),
            dvrr.arg(),
            dvrt.abs(),
            dvrt.arg(),
            dvrl.abs(),
            dvrl.arg(),
            vrc[i].abs() / normc,
            vrc[i].arg(),
        );
    }
    fs::write(dir.join(format!("LINA_eigen{mode_1b}.data")), e_buf)
        .map_err(|e| StarError::IoError(format!("write LINA_eigen: {e}")))?;
    Ok(())
}
