// SPDX-License-Identifier: GPL-3.0-or-later
//! Cell-specific kinetic and gravitational potential energy in `real(qp)`
//! (`cell_specific_KE_qp`, `cell_start_specific_KE_qp`, `cell_specific_PE_qp`,
//! `cell_start_specific_PE_qp`, and `get_dke_dt_dpe_dt` in
//! `star/private/star_utils.f90:2147-2370`).
//!
//! Upstream evaluates these in `real(qp)` and converts to `real(dp)` at the
//! points noted on each function. [`flexmesa_math::Quad`] reproduces that
//! arithmetic (`DECISIONS #46`). Every caller in this crate goes through the
//! kernels here, so the energy equation, the remesh energy bookkeeping and the
//! total-energy diagnostics all see the same values.

use flexmesa_math::Quad;

use crate::model::StarModel;

/// `qhalf = 0.5d0` assigned to a `real(qp)` variable.
#[inline]
fn qhalf() -> Quad {
    Quad::from(0.5_f64)
}

/// `Mbar` (`star_utils.f90:2192-2196`): `mass_correction(k)` converted to
/// `real(qp)` when `use_mass_corrections`, otherwise `1.0_qp`.
#[inline]
fn mbar_qp(model: &StarModel, k: usize) -> Quad {
    if model.controls.use_mass_corrections {
        Quad::from(model.aux.mass_correction.get(k).copied().unwrap_or(1.0))
    } else {
        Quad::ONE
    }
}

/// Result of [`specific_pe_qp`].
pub(crate) struct CellPe {
    /// `cell_specific_PE_qp` before any conversion to `real(dp)`.
    pub pe: Quad,
    /// `d_dlnR00` (`real(dp)`).
    pub d_dlnr00: f64,
    /// `d_dlnRp1` (`real(dp)`).
    pub d_dlnrp1: f64,
}

/// Arithmetic of `cell_specific_PE_qp` and `cell_start_specific_PE_qp`
/// (`star/private/star_utils.f90:2233-2277,2288-2325`).
///
/// The seven `real(dp)` inputs are converted to `real(qp)` on assignment, as
/// upstream's local variables are. Upstream has no guard on `r00`; a
/// non-positive `r00` gives a non-finite result here, where upstream stops
/// with `mesa_error`.
pub(crate) fn specific_pe_qp(
    mbar: Quad,
    (gp1, mp1, rp1): (f64, f64, f64),
    (g00, m00, r00): (f64, f64, f64),
) -> CellPe {
    // `gravp1 = -Gp1*mp1/rp1`, `d_gravp1_dlnRp1 = -gravp1` (`real(dp)`).
    let (gravp1, d_gravp1_dlnrp1) = if rp1 <= 0.0 {
        (Quad::ZERO, 0.0)
    } else {
        let g = -(Quad::from(gp1) * Quad::from(mp1) / Quad::from(rp1));
        (g, (-g).to_f64())
    };
    // `grav00 = -G00*m00/r00`, `d_grav00_dlnR00 = -grav00` (`real(dp)`).
    let grav00 = -(Quad::from(g00) * Quad::from(m00) / Quad::from(r00));
    let d_grav00_dlnr00 = (-grav00).to_f64();
    // `qhalf*Mbar*(...)` associates as `(qhalf*Mbar)*(...)`.
    let half_mbar = qhalf() * mbar;
    CellPe {
        pe: half_mbar * (gravp1 + grav00),
        d_dlnr00: (half_mbar * Quad::from(d_grav00_dlnr00)).to_f64(),
        d_dlnrp1: (half_mbar * Quad::from(d_gravp1_dlnrp1)).to_f64(),
    }
}

/// Result of [`specific_ke_qp`].
pub(crate) struct CellKe {
    /// `cell_specific_KE_qp` before any conversion to `real(dp)`.
    pub ke: Quad,
    /// `d_dv00` (`real(dp)`).
    pub d_dv00: f64,
    /// `d_dvp1` (`real(dp)`).
    pub d_dvp1: f64,
}

/// Which velocity variable carries the kinetic energy.
#[derive(Clone, Copy)]
pub(crate) enum KeVelocity {
    /// `u_flag`: cell-centred velocity `u(k)`.
    Cell(f64),
    /// `v_flag`: face velocities `v(k)` and `v(k+1)` (or `v_center`), with
    /// `dv2_dvp1`, which is `v(k+1)` for an interior cell and `0` for the
    /// innermost cell.
    Faces { v0: f64, v1: f64, dv2_dvp1: f64 },
    /// Neither flag: no kinetic energy.
    None,
}

/// Arithmetic of `cell_specific_KE_qp` and `cell_start_specific_KE_qp`
/// (`star/private/star_utils.f90:2147-2173,2184-2221`).
pub(crate) fn specific_ke_qp(mbar: Quad, vel: KeVelocity) -> CellKe {
    let half_mbar = qhalf() * mbar;
    match vel {
        KeVelocity::Cell(u) => {
            // `qhalf*Mbar*v0**2`; `d_dv00 = Mbar*s% u(k)`.
            let v0 = Quad::from(u);
            CellKe {
                ke: half_mbar * (v0 * v0),
                d_dv00: (mbar * v0).to_f64(),
                d_dvp1: 0.0,
            }
        }
        KeVelocity::Faces { v0, v1, dv2_dvp1 } => {
            // `v2 = qhalf*(v0**2 + v1**2)`; `qhalf*Mbar*v2`;
            // `d_dv00 = qhalf*Mbar*dv2_dv00`, `d_dvp1 = qhalf*Mbar*dv2_dvp1`.
            let (q0, q1) = (Quad::from(v0), Quad::from(v1));
            let v2 = qhalf() * (q0 * q0 + q1 * q1);
            CellKe {
                ke: half_mbar * v2,
                d_dv00: (half_mbar * q0).to_f64(),
                d_dvp1: (half_mbar * Quad::from(dv2_dvp1)).to_f64(),
            }
        }
        KeVelocity::None => CellKe {
            ke: Quad::ZERO,
            d_dv00: 0.0,
            d_dvp1: 0.0,
        },
    }
}

/// Inner-face `(G, m_grav, r)` for cell `k` at the current radii
/// (`star_utils.f90:2244-2252`).
fn inner_face_now(model: &StarModel, k: usize, r_p1: f64) -> (f64, f64, f64) {
    let nz = model.nz;
    if k + 1 == nz {
        (model.cgrav[nz - 1], model.m_center, model.r_center)
    } else {
        (model.cgrav[k + 1], model.m_grav[k + 1], r_p1)
    }
}

/// `cell_specific_PE_qp` for cell `k` from the model's current `r`.
pub(crate) fn cell_specific_pe_qp(model: &StarModel, k: usize) -> CellPe {
    let r_p1 = if k + 1 < model.nz {
        model.r[k + 1]
    } else {
        0.0
    };
    specific_pe_qp(
        mbar_qp(model, k),
        inner_face_now(model, k, r_p1),
        (model.cgrav[k], model.m_grav[k], model.r[k]),
    )
}

/// `cell_specific_KE_qp` for cell `k` from the model's current velocities.
pub(crate) fn cell_specific_ke_qp(model: &StarModel, k: usize) -> CellKe {
    let vel = if model.controls.u_flag {
        KeVelocity::Cell(model.u[k])
    } else if model.controls.v_flag {
        if k + 1 < model.nz {
            KeVelocity::Faces {
                v0: model.v[k],
                v1: model.v[k + 1],
                dv2_dvp1: model.v[k + 1],
            }
        } else {
            KeVelocity::Faces {
                v0: model.v[k],
                v1: model.v_center,
                dv2_dvp1: 0.0,
            }
        }
    } else {
        KeVelocity::None
    };
    specific_ke_qp(mbar_qp(model, k), vel)
}

/// `cell_specific_PE` (`star/private/star_utils.f90:2224-2230`): the
/// `real(qp)` value converted to `real(dp)`.
#[must_use]
pub fn cell_specific_pe(model: &StarModel, k: usize) -> f64 {
    if k >= model.nz {
        return 0.0;
    }
    cell_specific_pe_qp(model, k).pe.to_f64()
}

/// `cell_specific_KE` (`star/private/star_utils.f90:2176-2181`): the
/// `real(qp)` value converted to `real(dp)`.
#[must_use]
pub fn cell_specific_ke(model: &StarModel, k: usize) -> f64 {
    if k >= model.nz {
        return 0.0;
    }
    cell_specific_ke_qp(model, k).ke.to_f64()
}

/// `cell_specific_rotational_energy` (`star/private/star_utils.f90:2328-2339`):
/// face average `0.5 * (e_p1 + e_00)` with `e = i_rot * omega * omega` in
/// $\mathrm{erg\,g^{-1}}$.
///
/// Reproduces MESA `26.4.1` verbatim, including the absence of an extra `0.5`
/// factor inside `e_00` and `e_p1`.
#[must_use]
pub fn cell_specific_rotational_energy(model: &StarModel, k: usize) -> f64 {
    let nz = model.nz;
    if k >= nz {
        return 0.0;
    }
    let i_00 = model.aux.i_rot.get(k).copied().unwrap_or(0.0);
    let w_00 = model.omega.get(k).copied().unwrap_or(0.0);
    let e_00 = i_00 * w_00 * w_00;
    let e_p1 = if k + 1 < nz {
        let i_p1 = model.aux.i_rot.get(k + 1).copied().unwrap_or(0.0);
        let w_p1 = model.omega.get(k + 1).copied().unwrap_or(0.0);
        i_p1 * w_p1 * w_p1
    } else {
        0.0
    };
    0.5 * (e_p1 + e_00)
}

/// Returns the `RSP2` cell specific turbulent energy $w_k^2$ in
/// $\mathrm{erg\,g^{-1}}$ (`pow2(s% w(k))` in `star/private/star_utils.f90:2408, 2491`).
#[must_use]
pub(crate) fn cell_rsp2_eturb(model: &StarModel, k: usize) -> f64 {
    if model.layout.i_w.is_some() {
        model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0)
    } else {
        model.aux.etrb.get(k).copied().unwrap_or(0.0)
    }
}

/// `cell_specific_total_energy` (`star/private/star_utils.f90:2397-2410`):
/// specific total energy of cell `k` in $\mathrm{erg\,g^{-1}}$, summing
/// internal, radial kinetic (when `v_flag || u_flag`), gravitational potential,
/// rotational (when `rotation_flag && include_rotation_in_total_energy`),
/// `RSP2` (`w^2`), and `RSP` (`RSP_Et`) contributions in upstream order.
#[must_use]
pub fn cell_specific_total_energy(model: &StarModel, k: usize) -> f64 {
    if k >= model.nz {
        return 0.0;
    }
    let mut cell_total = model.energy[k];
    if model.controls.v_flag || model.controls.u_flag {
        cell_total += cell_specific_ke(model, k);
    }
    cell_total += cell_specific_pe(model, k);
    if model.controls.rotation_flag
        && model.controls.include_rotation_in_total_energy
    {
        cell_total += cell_specific_rotational_energy(model, k);
    }
    if model.controls.rsp2_flag {
        cell_total += cell_rsp2_eturb(model, k);
    }
    if model.controls.rsp_flag {
        cell_total += model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0);
    }
    cell_total
}

/// Output of [`get_dke_dt_dpe_dt`], all `real(dp)`.
#[derive(Clone, Copy, Debug)]
pub struct DkeDpeDt {
    /// Rate of change of specific kinetic energy
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub dke_dt: f64,
    /// Partial of `dke_dt` with respect to `v(k)` or `u(k)`.
    pub d_dkedt_dv00: f64,
    /// Partial of `dke_dt` with respect to `v(k+1)`.
    pub d_dkedt_dvp1: f64,
    /// Rate of change of specific potential energy
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub dpe_dt: f64,
    /// Partial of `dpe_dt` with respect to `lnR(k)`.
    pub d_dpedt_dlnr00: f64,
    /// Partial of `dpe_dt` with respect to `lnR(k+1)`.
    pub d_dpedt_dlnrp1: f64,
}

/// `get_dke_dt_dpe_dt` (`star/private/star_utils.f90:2341-2370`).
///
/// `PE_start`, `PE_new`, `KE_start` and `KE_new` are `real(dp)` upstream, so
/// each `real(qp)` energy is converted to `f64` before the difference is
/// taken, and the difference and the division by `dt` are in `f64`.
///
/// Radii are taken as `exp(lnR)` and `exp(lnR_start)`, which is how
/// `s% r(k)` and `s% r_start(k)` are defined. The start-of-step potential
/// energy uses the current `m_grav` and `mass_correction`, because the model
/// does not store `m_grav_start` or `mass_correction_start`; these differ
/// from upstream only when `use_mass_corrections` is set. Upstream's
/// `cell_start_specific_KE_qp` itself uses `mass_correction(k)`, not the
/// start value (`star_utils.f90:2154`).
#[must_use]
pub fn get_dke_dt_dpe_dt(model: &StarModel, k: usize, dt: f64) -> DkeDpeDt {
    let nz = model.nz;
    let mbar = mbar_qp(model, k);
    let has_p1 = k + 1 < nz;

    // Potential energy.
    let ln_r_start = |j: usize| model.ln_r_start[j];
    let r_start_p1 = if has_p1 {
        flexmesa_math::exp(ln_r_start(k + 1))
    } else {
        0.0
    };
    let pe_start = specific_pe_qp(
        mbar,
        inner_face_now(model, k, r_start_p1),
        (
            model.cgrav[k],
            model.m_grav[k],
            flexmesa_math::exp(ln_r_start(k)),
        ),
    )
    .pe
    .to_f64();
    let r_p1 = if has_p1 {
        flexmesa_math::exp(model.ln_r[k + 1])
    } else {
        0.0
    };
    let pe_new = specific_pe_qp(
        mbar,
        inner_face_now(model, k, r_p1),
        (
            model.cgrav[k],
            model.m_grav[k],
            flexmesa_math::exp(model.ln_r[k]),
        ),
    );
    let q1 = pe_new.pe.to_f64() - pe_start;
    let dpe_dt = q1 / dt;

    // Kinetic energy.
    let v_start =
        |j: usize| model.v_start.get(j).copied().unwrap_or(model.v[j]);
    let ke_start_vel = if model.controls.u_flag {
        KeVelocity::Cell(model.u_start.get(k).copied().unwrap_or(model.u[k]))
    } else if model.controls.v_flag {
        KeVelocity::Faces {
            v0: v_start(k),
            v1: if has_p1 {
                v_start(k + 1)
            } else {
                model.v_center
            },
            dv2_dvp1: 0.0,
        }
    } else {
        KeVelocity::None
    };
    let ke_start = specific_ke_qp(mbar, ke_start_vel).ke.to_f64();
    let ke_new = cell_specific_ke_qp(model, k);
    let q1 = ke_new.ke.to_f64() - ke_start;
    let dke_dt = q1 / dt;

    DkeDpeDt {
        dke_dt,
        d_dkedt_dv00: ke_new.d_dv00 / dt,
        d_dkedt_dvp1: ke_new.d_dvp1 / dt,
        dpe_dt,
        d_dpedt_dlnr00: pe_new.d_dlnr00 / dt,
        d_dpedt_dlnrp1: pe_new.d_dlnrp1 / dt,
    }
}
