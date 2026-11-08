// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-binary` (`RustBinarySystem`, Eggleton Roche
//! lobe, Kepler orbital relations, Ritter/Kolb/Arras RLOF, Lu et al. 2023 thin
//! disk $L_2$ mass loss, Kerr BH ISCO/Eddington limit, Bondi-Hoyle wind
//! accretion, $\dot{J}_{\mathrm{gr}}/\dot{J}_{\mathrm{ml}}/\dot{J}_{\mathrm{mb}}$,
//! Hut/Zahn tides, and `binary_pick_next_timestep`).

use flexmesa_binary::{
    bh_initial_equatorial_mass, binary_pick_next_timestep, build_anomaly_grid,
    eval_accreted_material_j, eval_arras_mdot_thin, eval_bh_spin,
    eval_bondi_hoyle_wind_fraction, eval_edot_tidal_hut, eval_j_orb,
    eval_jdot_gr, eval_jdot_ls, eval_jdot_mb, eval_jdot_ml,
    eval_k_div_t_convective, eval_k_div_t_radiative, eval_kolb_mdot_thick,
    eval_l2_mass_loss_fraction, eval_l2_overflow_limit, eval_mdot_edd,
    eval_ritter_mdot_thin, eval_rl_relative_gap, eval_rlobe,
    eval_tout_wind_enhancement, eval_tsync, hut_f2, hut_f3, hut_f4, hut_f5,
    period_from_separation, separation_from_j_orb, separation_from_period,
    BinaryCompanion, BinarySystem, BinaryTimestepControls,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::star_py::PyStarModel;

/// Python-exposed wrapper around [`flexmesa_binary::BinarySystem`].
#[pyclass(name = "RustBinarySystem")]
#[derive(Clone)]
pub struct PyBinarySystem {
    /// Inner canonical Rust binary system state.
    pub inner: BinarySystem,
}

#[pymethods]
impl PyBinarySystem {
    /// Creates a coupled binary system with a donor `RustStarModel` and a
    /// point-mass companion.
    #[staticmethod]
    #[pyo3(signature = (
        donor,
        companion_mass_msun,
        initial_period_days,
        eccentricity = 0.0,
        is_black_hole = false,
        initial_bh_spin = 0.0
    ))]
    pub fn new_point_mass(
        donor: &PyStarModel,
        companion_mass_msun: f64,
        initial_period_days: f64,
        eccentricity: f64,
        is_black_hole: bool,
        initial_bh_spin: f64,
    ) -> Self {
        Self {
            inner: BinarySystem::new_point_mass(
                donor.inner.clone(),
                companion_mass_msun,
                initial_period_days,
                eccentricity,
                is_black_hole,
                initial_bh_spin,
            ),
        }
    }

    /// Creates a coupled binary system with two evolved `RustStarModel` stars.
    #[staticmethod]
    #[pyo3(signature = (donor, accretor, initial_period_days, eccentricity = 0.0))]
    pub fn new_two_stars(
        donor: &PyStarModel,
        accretor: &PyStarModel,
        initial_period_days: f64,
        eccentricity: f64,
    ) -> Self {
        Self {
            inner: BinarySystem::new_two_stars(
                donor.inner.clone(),
                accretor.inner.clone(),
                initial_period_days,
                eccentricity,
            ),
        }
    }

    /// Advances the coupled binary system by `dt` (s).
    pub fn step(&mut self, dt: f64) -> (f64, i32) {
        self.inner.step(dt)
    }

    /// Returns the donor `RustStarModel`.
    #[getter]
    pub fn donor(&self) -> PyStarModel {
        PyStarModel {
            inner: self.inner.donor.clone(),
        }
    }

    /// Returns the secondary `RustStarModel` if the companion is an evolved star.
    #[getter]
    pub fn accretor(&self) -> Option<PyStarModel> {
        match &self.inner.companion {
            BinaryCompanion::PointMass { .. } => None,
            BinaryCompanion::Star(s2) => Some(PyStarModel {
                inner: (**s2).clone(),
            }),
        }
    }

    /// Current donor mass $M_1$ (g).
    #[getter]
    pub fn donor_mass_g(&self) -> f64 {
        self.inner.donor.mstar
    }
    /// Current accretor mass $M_2$ (g).
    #[getter]
    pub fn accretor_mass_g(&self) -> f64 {
        self.inner.accretor_mass_g()
    }
    /// Orbital separation $a$ (cm).
    #[getter]
    pub fn separation(&self) -> f64 {
        self.inner.separation
    }
    /// Orbital period $P$ (s).
    #[getter]
    pub fn period(&self) -> f64 {
        self.inner.period
    }
    /// Orbital eccentricity $e$.
    #[getter]
    pub fn eccentricity(&self) -> f64 {
        self.inner.eccentricity
    }
    /// Orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
    #[getter]
    pub fn angular_momentum_j(&self) -> f64 {
        self.inner.angular_momentum_j
    }
    /// Roche-lobe radii `(rl_donor, rl_accretor)` (cm).
    #[getter]
    pub fn rl(&self) -> (f64, f64) {
        (self.inner.rl[0], self.inner.rl[1])
    }
    /// Periastron relative Roche-lobe gaps `(gap_donor, gap_accretor)`.
    #[getter]
    pub fn rl_relative_gap(&self) -> (f64, f64) {
        (self.inner.rl_relative_gap[0], self.inner.rl_relative_gap[1])
    }
    /// Current mass transfer rate $\dot{M}_{\mathrm{tr}}$ ($\mathrm{g\,s^{-1}}$).
    #[getter]
    pub fn mtransfer_rate(&self) -> f64 {
        self.inner.mtransfer_rate
    }
    /// Current $\dot{J}_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
    #[getter]
    pub fn jdot(&self) -> f64 {
        self.inner.jdot
    }
    /// Binary age (yr).
    #[getter]
    pub fn binary_age_yr(&self) -> f64 {
        self.inner.binary_age_yr
    }
    /// Completed step count.
    #[getter]
    pub fn model_number(&self) -> usize {
        self.inner.model_number
    }
    /// Next maximum binary timestep (s).
    #[getter]
    pub fn max_timestep_sec(&self) -> f64 {
        self.inner.max_timestep_sec
    }
}

/// Evaluates Eggleton (1983) Roche-lobe radius $R_{L,1}(m_1, m_2, a)$ (cm).
#[pyfunction]
pub fn binary_eval_rlobe(m1: f64, m2: f64, a: f64) -> f64 {
    eval_rlobe(m1, m2, a)
}

/// Evaluates periastron relative Roche-lobe gap $(R - R_L(1-e)) / (R_L(1-e))$.
#[pyfunction]
pub fn binary_eval_rl_relative_gap(r: f64, rl: f64, eccentricity: f64) -> f64 {
    eval_rl_relative_gap(r, rl, eccentricity)
}

/// Computes orbital separation $a$ (cm) from period $P$ (s) and masses $m_1, m_2$ (g).
#[pyfunction]
pub fn binary_separation_from_period(m1: f64, m2: f64, period: f64) -> f64 {
    separation_from_period(m1, m2, period)
}

/// Computes orbital period $P$ (s) from separation $a$ (cm) and masses $m_1, m_2$ (g).
#[pyfunction]
pub fn binary_period_from_separation(m1: f64, m2: f64, separation: f64) -> f64 {
    period_from_separation(m1, m2, separation)
}

/// Computes orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
#[pyfunction]
pub fn binary_eval_j_orb(
    m1: f64,
    m2: f64,
    separation: f64,
    eccentricity: f64,
) -> f64 {
    eval_j_orb(m1, m2, separation, eccentricity)
}

/// Computes orbital separation $a$ (cm) from $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
#[pyfunction]
pub fn binary_separation_from_j_orb(
    m1: f64,
    m2: f64,
    j_orb: f64,
    eccentricity: f64,
) -> f64 {
    separation_from_j_orb(m1, m2, j_orb, eccentricity)
}

/// Evaluates the Marchant et al. (2016) contact binary $L_2$ overflow limit.
#[pyfunction]
pub fn binary_eval_l2_overflow_limit(m1: f64, m2: f64) -> f64 {
    eval_l2_overflow_limit(m1, m2)
}

/// Builds the eccentric true-anomaly and normalized time grid `(theta_co, time_co)`.
#[pyfunction]
pub fn binary_build_anomaly_grid<'py>(
    py: Python<'py>,
    anomaly_steps: usize,
    eccentricity: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (theta, time) = build_anomaly_grid(anomaly_steps, eccentricity);
    (PyArray1::from_vec(py, theta), PyArray1::from_vec(py, time))
}

/// Evaluates the Ritter (1988) optically thin RLOF mass transfer rate ($\mathrm{g\,s^{-1}}$).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn binary_eval_ritter_mdot_thin(
    m_donor: f64,
    m_accretor: f64,
    r_donor: f64,
    rl_donor: f64,
    rho_surf: f64,
    p_surf: f64,
    t_surf: f64,
    mu_surf: f64,
    is_kolb_scheme: bool,
) -> f64 {
    eval_ritter_mdot_thin(
        m_donor,
        m_accretor,
        r_donor,
        rl_donor,
        rho_surf,
        p_surf,
        t_surf,
        mu_surf,
        is_kolb_scheme,
    )
}

/// Evaluates the Kolb & Ritter (1990) optically thick RLOF mass transfer rate ($\mathrm{g\,s^{-1}}$).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn binary_eval_kolb_mdot_thick(
    m_donor: f64,
    m_accretor: f64,
    rl_donor: f64,
    r_profile: PyReadonlyArray1<'_, f64>,
    p_profile: PyReadonlyArray1<'_, f64>,
    t_profile: PyReadonlyArray1<'_, f64>,
    mu_profile: PyReadonlyArray1<'_, f64>,
    gamma1_profile: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    Ok(eval_kolb_mdot_thick(
        m_donor,
        m_accretor,
        rl_donor,
        r_profile.as_slice()?,
        p_profile.as_slice()?,
        t_profile.as_slice()?,
        mu_profile.as_slice()?,
        gamma1_profile.as_slice()?,
    ))
}

/// Evaluates the Arras optically thin RLOF mass transfer rate ($\mathrm{g\,s^{-1}}$).
#[pyfunction]
pub fn binary_eval_arras_mdot_thin(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    r_donor: f64,
    rho_surf: f64,
    t_surf: f64,
    mu_surf: f64,
) -> f64 {
    eval_arras_mdot_thin(
        m_donor, m_accretor, separation, r_donor, rho_surf, t_surf, mu_surf,
    )
}

/// Evaluates the Lu et al. (2023) thin-disk outer $L_2$ mass-loss fraction $f_{L2}$.
#[pyfunction]
pub fn binary_eval_l2_mass_loss_fraction(
    donor_mass_msun: f64,
    accretor_mass_msun: f64,
    mass_transfer_rate_msun_yr: f64,
    orbital_separation_rsun: f64,
    disk_alpha: f64,
    disk_mu: f64,
) -> f64 {
    eval_l2_mass_loss_fraction(
        donor_mass_msun,
        accretor_mass_msun,
        mass_transfer_rate_msun_yr,
        orbital_separation_rsun,
        disk_alpha,
        disk_mu,
    )
}

/// Computes Kerr BH equivalent initial non-rotating mass and current spin `(m_bh_eq, a_spin)`.
#[pyfunction]
pub fn binary_eval_bh_spin_and_edd(
    m_bh_init: f64,
    a_spin_init: f64,
    m_bh_current: f64,
    kappa_surf: f64,
) -> (f64, f64, f64, f64) {
    let m_eq = bh_initial_equatorial_mass(m_bh_init, a_spin_init);
    let a_cur = eval_bh_spin(m_bh_current, m_eq);
    let (mdot_edd, eta) = eval_mdot_edd(m_bh_current, m_eq, kappa_surf);
    (m_eq, a_cur, mdot_edd, eta)
}

/// Evaluates Tout & Eggleton (1988) wind enhancement and Hurley et al. (2002) Bondi-Hoyle fraction.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn binary_eval_wind_transfer(
    m_donor: f64,
    m_accretor: f64,
    r_donor: f64,
    rl_donor: f64,
    separation: f64,
    eccentricity: f64,
    b_wind: f64,
    alpha_bh: f64,
    beta_bh: f64,
    max_frac: f64,
) -> (f64, f64) {
    (
        eval_tout_wind_enhancement(r_donor, rl_donor, b_wind),
        eval_bondi_hoyle_wind_fraction(
            m_donor,
            m_accretor,
            r_donor,
            separation,
            eccentricity,
            alpha_bh,
            beta_bh,
            max_frac,
        ),
    )
}

/// Evaluates de Mink et al. (2013) accreted specific angular momentum `(j_acc, mode)`.
#[pyfunction]
pub fn binary_eval_accreted_material_j(
    m_donor: f64,
    m_accretor: f64,
    r_accretor: f64,
    separation: f64,
) -> (f64, i32) {
    eval_accreted_material_j(m_donor, m_accretor, r_accretor, separation)
}

/// Evaluates gravitational wave $\dot{J}_{\mathrm{gr}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
#[pyfunction]
pub fn binary_eval_jdot_gr(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    j_orb: f64,
) -> f64 {
    eval_jdot_gr(m_donor, m_accretor, separation, j_orb)
}

/// Evaluates system mass-loss $\dot{J}_{\mathrm{ml}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn binary_eval_jdot_ml(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
    mdot_sys_tr_donor: f64,
    mdot_sys_wind_donor: f64,
    mdot_sys_tr_accretor: f64,
    mdot_sys_wind_accretor: f64,
    mdot_sys_cct: f64,
    gamma_mt: f64,
) -> f64 {
    eval_jdot_ml(
        m_donor,
        m_accretor,
        separation,
        period,
        eccentricity,
        mdot_sys_tr_donor,
        mdot_sys_wind_donor,
        mdot_sys_tr_accretor,
        mdot_sys_wind_accretor,
        mdot_sys_cct,
        gamma_mt,
    )
}

/// Evaluates Rappaport-Verbunt-Joss (1983) magnetic braking $\dot{J}_{\mathrm{mb}}$.
#[pyfunction]
pub fn binary_eval_jdot_mb(
    m_star: f64,
    r_star: f64,
    rl_star: f64,
    period: f64,
    gamma_mb: f64,
    q_conv_env: f64,
    scale_for_low_qconv_env: bool,
) -> f64 {
    eval_jdot_mb(
        m_star,
        r_star,
        rl_star,
        period,
        gamma_mb,
        q_conv_env,
        scale_for_low_qconv_env,
    )
}

/// Evaluates spin-orbit tidal torque $\dot{J}_{\mathrm{ls}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
#[pyfunction]
pub fn binary_eval_jdot_ls(
    s_donor_old: f64,
    s_donor: f64,
    s_accretor_old: f64,
    s_accretor: f64,
    dt: f64,
) -> f64 {
    eval_jdot_ls(s_donor_old, s_donor, s_accretor_old, s_accretor, dt)
}

/// Evaluates Hut (1981) eccentricity polynomials and tidal rates.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn binary_eval_tides_hut(
    m_star: f64,
    m_companion: f64,
    r_phot: f64,
    separation: f64,
    eccentricity: f64,
    period: f64,
    omega_star: f64,
    m_env_msun: f64,
    r_env_rsun: f64,
    l_phot_lsun: f64,
    r_gyr_sq: f64,
    f_tid: f64,
    has_convective_envelope: bool,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let k_div_t = if has_convective_envelope {
        eval_k_div_t_convective(
            m_star / flexmesa_const::MSUN,
            m_env_msun,
            r_phot / flexmesa_const::RSUN,
            r_env_rsun,
            l_phot_lsun,
            period,
            omega_star,
            1.0,
        )
    } else {
        eval_k_div_t_radiative(m_star, m_companion, r_phot, separation)
    };
    let tsync = eval_tsync(
        k_div_t,
        m_companion / m_star,
        r_gyr_sq,
        r_phot,
        separation,
        f_tid,
    );
    let edot = eval_edot_tidal_hut(
        k_div_t,
        m_star,
        m_companion,
        r_phot,
        separation,
        eccentricity,
        omega_star,
        period,
        f_tid,
    );
    (
        hut_f2(eccentricity),
        hut_f3(eccentricity),
        hut_f4(eccentricity),
        hut_f5(eccentricity),
        k_div_t,
        tsync,
        edot,
    )
}

/// Evaluates `binary_pick_next_timestep` matching `binary/test/src/test_binary.f90`.
#[pyfunction]
#[pyo3(signature = (
    donor_time_step_yr,
    max_timestep_sec,
    rl_gap_old_donor,
    rl_gap_donor,
    fr = 0.01,
    fr_limit = 0.01,
    fr_dt_limit = 0.0,
    dt_softening_factor = 0.0,
    time_delta_coeff = 1.0
))]
#[allow(clippy::too_many_arguments)]
pub fn binary_pick_next_timestep_rl(
    donor_time_step_yr: f64,
    max_timestep_sec: f64,
    rl_gap_old_donor: f64,
    rl_gap_donor: f64,
    fr: f64,
    fr_limit: f64,
    fr_dt_limit: f64,
    dt_softening_factor: f64,
    time_delta_coeff: f64,
) -> (f64, i32) {
    let ctrl = BinaryTimestepControls {
        donor_time_step_yr,
        max_timestep_sec,
        time_delta_coeff,
        dt_softening_factor,
        rl_relative_gap: [rl_gap_donor, 0.0],
        rl_relative_gap_old: [rl_gap_old_donor, 0.0],
        is_point_mass_companion: true,
        fm: -1.0,
        fr,
        fr_limit,
        fr_dt_limit,
        fj: -1.0,
        fa: -1.0,
        fe: -1.0,
        fdm: -1.0,
        ..BinaryTimestepControls::default()
    };
    binary_pick_next_timestep(&ctrl)
}

/// Registers Wave 5.3 `binary` classes and functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBinarySystem>()?;
    m.add_function(wrap_pyfunction!(binary_eval_rlobe, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_rl_relative_gap, m)?)?;
    m.add_function(wrap_pyfunction!(binary_separation_from_period, m)?)?;
    m.add_function(wrap_pyfunction!(binary_period_from_separation, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_j_orb, m)?)?;
    m.add_function(wrap_pyfunction!(binary_separation_from_j_orb, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_l2_overflow_limit, m)?)?;
    m.add_function(wrap_pyfunction!(binary_build_anomaly_grid, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_ritter_mdot_thin, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_kolb_mdot_thick, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_arras_mdot_thin, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_l2_mass_loss_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_bh_spin_and_edd, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_wind_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_accreted_material_j, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_jdot_gr, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_jdot_ml, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_jdot_mb, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_jdot_ls, m)?)?;
    m.add_function(wrap_pyfunction!(binary_eval_tides_hut, m)?)?;
    m.add_function(wrap_pyfunction!(binary_pick_next_timestep_rl, m)?)?;
    Ok(())
}
