# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end MESA `inlist` parser, `./rn` stage extractor, multi-stage
`test_suite` case execution driver, and Stage `E2E-1` (`#92`) Tier 1 suite
runner (`flexmesa.star.e2e`).
"""

from __future__ import annotations

import pathlib
from typing import Any

from flexmesa import _rust
from flexmesa.star import e2e_tier45
from flexmesa.star import model as star_model

TIER1_CASES: tuple[str, ...] = tuple(_rust.star_tier1_cases())
TIER2_CASES: tuple[str, ...] = tuple(_rust.star_tier2_cases())
TIER3_CASES: tuple[str, ...] = tuple(_rust.star_tier3_cases())
TIER4_STAR_CASES = e2e_tier45.TIER4_STAR_CASES
TIER4_ACTIVE_STAR_CASES = e2e_tier45.TIER4_ACTIVE_STAR_CASES
TIER4_ASTERO_CASES = e2e_tier45.TIER4_ASTERO_CASES
TIER4_DEV_CASES = e2e_tier45.TIER4_DEV_CASES
build_rsp_envelope_model = e2e_tier45.build_rsp_envelope_model
eval_astero_scaling_dnu_numax_uhz = e2e_tier45.eval_astero_scaling_dnu_numax_uhz
eval_linear_oscillation_modes = e2e_tier45.eval_linear_oscillation_modes
eval_rsp_2nd_crossing_edges = e2e_tier45.eval_rsp_2nd_crossing_edges
eval_rsp_lina_modes = e2e_tier45.eval_rsp_lina_modes
eval_target_mode_frequency_and_growth = (
    e2e_tier45.eval_target_mode_frequency_and_growth
)
run_astero_suite = e2e_tier45.run_astero_suite
run_astero_test_suite_case = e2e_tier45.run_astero_test_suite_case
run_dev_cases_suite = e2e_tier45.run_dev_cases_suite
run_tier4_star_suite = e2e_tier45.run_tier4_star_suite
run_tier4_suite = e2e_tier45.run_tier4_suite


def parse_inlist_bundle(inlist_path: str | pathlib.Path) -> dict[str, Any]:
    """Parse a MESA `inlist` file and recursively resolve `read_extra_*_inlist`.

    Args:
        inlist_path: Path to the entry `inlist` file.

    Returns:
        Dictionary of resolved `&star_job` and `&controls` settings.
    """
    return dict(_rust.star_parse_inlist_bundle(str(inlist_path)))


def parse_rn_stages(
    case_dir: str | pathlib.Path,
    skip_optional: bool = True,
) -> list[tuple[str, str, str]]:
    """Parse an upstream MESA `./rn` bash script into execution stages.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.

    Returns:
        List of `(stage_kind, arg1, arg2)` stage tuples.
    """
    return list(_rust.star_parse_rn_stages(str(case_dir), bool(skip_optional)))


def run_test_suite_case(
    case_dir: str | pathlib.Path,
    skip_optional: bool = True,
    max_steps_per_stage: int | None = None,
    verify_restart: bool = False,
) -> dict[str, Any]:
    """Execute a multi-stage upstream MESA `test_suite` case directory.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        Structured dictionary containing per-stage reports, step counts,
        Newton iterations, final physical state, and wall-clock runtime.
    """
    return dict(
        _rust.star_run_test_suite_case(
            str(case_dir),
            bool(skip_optional),
            None if max_steps_per_stage is None else int(max_steps_per_stage),
            bool(verify_restart),
        )
    )


def run_tier1_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 33 Tier 1 `star/test_suite` cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation containing
            `star/test_suite/<case>`.
        max_steps_per_stage: Optional maximum evolutionary steps per `./rn`
            stage (`None` runs until physical termination).
        verify_restart: Whether to verify bit-identical `.photo` restart parity
            (`0 ULP`) on every case.

    Returns:
        List of 33 `CaseRunReport` dictionaries in canonical `TIER1_CASES`
        order.
    """
    raw = _rust.star_run_tier1_suite(
        str(mesa_dir),
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def eval_starspot_luminosity(
    l_surf: float,
    f_spot: float,
    x_spot: float,
) -> tuple[float, float]:
    """Evaluate the YREC starspot effective unspotted luminosity (`starspots`).

    Args:
        l_surf: Bolometric surface luminosity in `erg s^-1` (or `L_sun`).
        f_spot: Fractional surface spot coverage in `[0, 1]`.
        x_spot: Spot-to-ambient temperature ratio `T_spot / T_amb`.

    Returns:
        Tuple `(l_eff, alpha_spot)` where
        `alpha_spot = f_spot * x_spot^4 + 1 - f_spot` and
        `l_eff = l_surf / alpha_spot`.
    """
    l_eff, alpha = _rust.star_eval_starspot_luminosity(
        float(l_surf), float(f_spot), float(x_spot)
    )
    return float(l_eff), float(alpha)


def eval_t_tau_gradr_factor(
    tau: float,
    scheme: str = "Eddington",
) -> tuple[float, float]:
    """Evaluate the $T(\\tau)$ Hopf function $q(\\tau)$ and
    $\\nabla_{\\mathrm{rad}}$ correction factor $1 + \\tau\\,dq/d\\tau$
    (`T_tau_gradr`).

    Args:
        tau: Optical depth $\\tau \\ge 0$ (dimensionless).
        scheme: Atmosphere $T(\\tau)$ relation (`"Eddington"`,
            `"Krishna_Swamy"`, `"solar_Hopf_grey"`, or `"Trampedach_solar"`).

    Returns:
        Tuple `(q_tau, gradr_factor)`.
    """
    q_val, fac = _rust.star_eval_t_tau_gradr_factor(float(tau), str(scheme))
    return float(q_val), float(fac)


def eval_model_colors(model: star_model.StarModel) -> tuple[float, ...]:
    """Evaluate 10-column synthetic Johnson-Cousins photometry
    (`custom_colors`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        10-tuple `(Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V)`.
    """
    cols = _rust.star_eval_model_colors(model._rust_model)
    return tuple(float(v) for v in cols)


def eval_solar_calibration_objective(
    log_l: float,
    log_r: float,
    zx_surf: float,
    target_zx: float = 0.0229,
) -> float:
    """Evaluate the solar calibration $\\chi^2$ objective
    (`simplex_solar_calibration`).

    Args:
        log_l: $\\log_{10}(L / L_\\odot)$.
        log_r: $\\log_{10}(R / R_\\odot)$.
        zx_surf: Surface metal-to-hydrogen ratio $(Z/X)_{\\mathrm{surf}}$.
        target_zx: Target solar $(Z/X)_\\odot$ (`0.0229` for GS98).

    Returns:
        Dimensionless $\\chi^2$ value.
    """
    return float(
        _rust.star_eval_solar_calibration_objective(
            float(log_l),
            float(log_r),
            float(zx_surf),
            float(target_zx),
        )
    )


def nelder_mead_solar_step(
    simplex: list[tuple[tuple[float, float, float], float]],
) -> tuple[float, float, float]:
    """Perform one Nelder-Mead simplex reflection step for solar calibration.

    Args:
        simplex: 4-element list of `((Y_0, Z_0, alpha_mlt), chi2)` vertices.

    Returns:
        Reflected parameter triple `(Y_0, Z_0, alpha_mlt)`.
    """
    raw_in = [
        ([float(p[0]), float(p[1]), float(p[2])], float(c)) for p, c in simplex
    ]
    res = _rust.star_nelder_mead_solar_step(raw_in)
    return float(res[0]), float(res[1]), float(res[2])


def run_tier2_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 17 Tier 2 `star/test_suite` cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation containing
            `star/test_suite/<case>`.
        max_steps_per_stage: Optional maximum evolutionary steps per `./rn`
            stage (`None` runs until physical termination).
        verify_restart: Whether to verify bit-identical `.photo` restart parity
            (`0 ULP`) on every case.

    Returns:
        List of 17 `CaseRunReport` dictionaries in canonical `TIER2_CASES`
        order.
    """
    raw = _rust.star_run_tier2_suite(
        str(mesa_dir),
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def eval_planetary_irradiation_heating(
    model: star_model.StarModel,
    f_irr: float,
    col_depth_irr: float = 250.0,
) -> tuple[list[float], float]:
    """Evaluate planetary surface irradiation heating profile and radius
    (`irradiated_planet`).

    Args:
        model: Active `StarModel` instance.
        f_irr: Incident surface irradiation flux in `erg cm^-2 s^-1`.
        col_depth_irr: Attenuation column depth in `g cm^-2`.

    Returns:
        Tuple `(eps_irr, r_rjup)` where `eps_irr` is the specific heating rate
        profile (`erg g^-1 s^-1`, shape `(nz,)`) and `r_rjup` is `R / R_Jup`.
    """
    eps_irr, r_rjup = _rust.star_eval_planetary_irradiation_heating(
        model._rust_model,
        float(f_irr),
        float(col_depth_irr),
    )
    return [float(v) for v in eps_irr], float(r_rjup)


def eval_wd_phase_separation(
    model: star_model.StarModel,
    option: str = "CO",
) -> tuple[float, float]:
    """Evaluate white dwarf Coulomb crystallization and 2/3-component phase
    separation (`wd_cool_0.6M` and `wd_o_ne_3_phase`).

    Args:
        model: Active `StarModel` instance.
        option: Mixture option (`"CO"` or `"ONe"`).

    Returns:
        Tuple `(crystallized_mass_frac, total_latent_lum_erg_s)`.
    """
    cryst_frac, l_lat = _rust.star_eval_wd_phase_separation(
        model._rust_model, str(option)
    )
    return float(cryst_frac), float(l_lat)


def eval_conductive_flame_state(
    model: star_model.StarModel,
) -> tuple[float, float]:
    """Evaluate conductive flame speed and flame mass coordinate
    (`conductive_flame` and `cburn_inward`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        Tuple `(v_flame_cm_s, q_flame)`.
    """
    v_f, q_f = _rust.star_eval_conductive_flame_state(model._rust_model)
    return float(v_f), float(q_f)


def run_tier3_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 22 Tier 3 `star/test_suite` cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation containing
            `star/test_suite/<case>`.
        max_steps_per_stage: Optional maximum evolutionary steps per `./rn`
            stage (`None` runs until physical termination).
        verify_restart: Whether to verify bit-identical `.photo` restart parity
            (`0 ULP`) on every case.

    Returns:
        List of 22 `CaseRunReport` dictionaries in canonical `TIER3_CASES`
        order.
    """
    raw = _rust.star_run_tier3_suite(
        str(mesa_dir),
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def eval_hot_cool_blended_wind_mdot(model: star_model.StarModel) -> float:
    """Evaluate the temperature-blended Vink + Reimers/Blöcker wind mass-loss
    rate $\\dot{M}$ ($M_\\odot\\,\\mathrm{yr}^{-1}$) (`hot_cool_wind`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        Net wind mass-loss rate $\\dot{M} < 0$ ($M_\\odot\\,\\mathrm{yr}^{-1}$).
    """
    return float(_rust.star_eval_hot_cool_blended_wind_mdot(model._rust_model))


def eval_roche_gravity_darkening(
    omega_div_omega_crit: float,
) -> tuple[float, float, float, float]:
    """Evaluate Roche oblateness $R_{\\mathrm{eq}}/R_{\\mathrm{pol}}$,
    $(f_P, f_T)$, and Espinosa Lara & Rieutord (2011) equatorial-to-polar
    $T_{\\mathrm{eff}}$ ratio (`high_rot_darkening`).

    Args:
        omega_div_omega_crit: Dimensionless rotation ratio
            $\\Omega / \\Omega_{\\mathrm{crit}}$.

    Returns:
        4-tuple `(r_eq_div_r_pol, fp, ft, teff_eq_div_teff_pol)`.
    """
    r_rat, fp, ft, t_rat = _rust.star_eval_roche_gravity_darkening(
        float(omega_div_omega_crit)
    )
    return float(r_rat), float(fp), float(ft), float(t_rat)


def eval_spruit_tayler_dynamo_diagnostics(
    model: star_model.StarModel,
) -> tuple[float, ...]:
    """Evaluate the 13 Spruit-Tayler magnetic dynamo and angular momentum
    transport diagnostics (`15M_dynamo`, `20M_z2m2_high_rotation`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        13-element tuple of diagnostic scalars.
    """
    vals = _rust.star_eval_spruit_tayler_dynamo_diagnostics(model._rust_model)
    return tuple(float(v) for v in vals)


def eval_magnetic_braking_torque(
    model: star_model.StarModel,
    b_dipolar_gauss: float = 5000.0,
) -> tuple[list[float], float]:
    """Evaluate the ud-Doula, Owocki & Townsend (2009) magnetospheric wind
    braking torque profile and braked surface rotation velocity
    (`magnetic_braking`).

    Args:
        model: Active `StarModel` instance.
        b_dipolar_gauss: Surface dipolar magnetic field strength in Gauss.

    Returns:
        Tuple `(extra_jdot, braked_surf_v_rot_kms)`.
    """
    jdot, v_rot = _rust.star_eval_magnetic_braking_torque(
        model._rust_model, float(b_dipolar_gauss)
    )
    return [float(v) for v in jdot], float(v_rot)


def eval_fe_core_infall_state(
    model: star_model.StarModel,
) -> tuple[float, float]:
    """Evaluate iron-group core mass $M_{\\mathrm{Fe,core}}$ ($M_\\odot$) and
    peak core infall velocity $v_{\\mathrm{infall}}$ ($\\mathrm{cm\\,s^{-1}}$).

    Args:
        model: Active `StarModel` instance.

    Returns:
        Tuple `(fe_core_mass_msun, max_infall_speed_cm_s)`.
    """
    m_fe, v_inf = _rust.star_eval_fe_core_infall_state(model._rust_model)
    return float(m_fe), float(v_inf)


def eval_duffell_rti_shock_state(
    model: star_model.StarModel,
) -> tuple[float, float, float]:
    """Evaluate supernova shock mass $M_{\\mathrm{shock}}$ ($M_\\odot$), peak
    Duffell (2016) Rayleigh-Taylor diffusivity $D_{\\mathrm{RTI}}$, and mixed
    $^{1}\\mathrm{H}$ mass fraction (`ccsn_IIp`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        3-tuple `(shock_mass_msun, max_d_rti_cm2_s, mixed_h1_mass_frac)`.
    """
    m_sh, d_rti, h1_m = _rust.star_eval_duffell_rti_shock_state(
        model._rust_model
    )
    return float(m_sh), float(d_rti), float(h1_m)


def export_stella_ascii(model: star_model.StarModel) -> tuple[str, str]:
    """Export `STELLA` radiation-hydrodynamics `.hyd` and 3-group (`h1`, `he4`,
    `z_rem`) `.abn` ASCII tables (`ccsn_IIp`, Issue #4).

    Args:
        model: Active `StarModel` instance.

    Returns:
        Tuple `(hyd_ascii, abn_ascii)`.
    """
    hyd, abn = _rust.star_export_stella_ascii(model._rust_model)
    return str(hyd), str(abn)


def eval_pair_instability_gamma1_and_energy(
    model: star_model.StarModel,
) -> tuple[float, float]:
    """Evaluate pressure-weighted $\\langle\\Gamma_1 - 4/3\\rangle_P$ and total
    binding/explosion energy $E_{\\mathrm{tot}}$ (`pisn`, `ppisn`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        Tuple `(delta_gamma1_p_avg, total_energy_erg)`.
    """
    dg1, etot = _rust.star_eval_pair_instability_gamma1_and_energy(
        model._rust_model
    )
    return float(dg1), float(etot)


def eval_tov_effective_cgrav(model: star_model.StarModel) -> list[float]:
    """Evaluate the post-Newtonian TOV effective gravity profile
    $G_{\\mathrm{eff}}(k)$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$)
    (`ns_h`, `ns_he`, `ns_c`, `tzo`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        List of $G_{\\mathrm{eff}}(k)$ values across all `nz` zones.
    """
    return [
        float(v) for v in _rust.star_eval_tov_effective_cgrav(model._rust_model)
    ]


def eval_tzo_core_accretion_step(
    model: star_model.StarModel,
    eta_ledd: float = 1.0,
    eta_medd: float = 1.0,
    target_mass_msun: float = 1.4,
) -> tuple[float, float]:
    """Evaluate Thorne-Zytkow Eddington-limited neutron core accretion (`tzo`).

    Args:
        model: Active `StarModel` instance.
        eta_ledd: Luminosity efficiency factor relative to $L_{\\mathrm{Edd}}$.
        eta_medd: Mass accretion factor relative to $\\dot{M}_{\\mathrm{Edd}}$.
        target_mass_msun: Target neutron core mass in $M_\\odot$.

    Returns:
        Tuple `(new_core_mass_msun, l_edd_erg_s)`.
    """
    m_c, l_edd = _rust.star_eval_tzo_core_accretion_step(
        model._rust_model,
        float(eta_ledd),
        float(eta_medd),
        float(target_mass_msun),
    )
    return float(m_c), float(l_edd)
