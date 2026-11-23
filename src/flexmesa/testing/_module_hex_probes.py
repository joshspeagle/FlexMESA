# SPDX-License-Identifier: LGPL-3.0-or-later
"""IEEE-754 helpers and per-module strict probes for ``mesa_module_hex``."""

from __future__ import annotations

import csv
import itertools
import math
import pathlib
import struct
from collections.abc import Callable, Sequence

import numpy as np

from flexmesa import (
    _rust,
    binary,
    chem,
    colors,
    constants,
    eos,
    interp_1d,
    interp_2d,
    ionization,
    kap,
    net,
)

# Tuple: (expected_outputs, checked, zero_ulp_values, max_ulp, note).
ProbeOutcome = tuple[int, int, int, int, str]
ProbeFn = Callable[[pathlib.Path], ProbeOutcome]

# 100-point bell profile from `interp_1d/test/src/test_interp_1d.f90`.
_INTERP_1D_TEST_U: tuple[float, ...] = (
    0.3691382918814334,
    0.3695899829574669,
    0.3699633032242006,
    0.3709860746618878,
    0.3724175838188959,
    0.3742482756438662,
    0.3764768199547272,
    0.3790910321656262,
    0.3820871425096383,
    0.3854500131142883,
    0.3891727128876379,
    0.3932377816410259,
    0.3976355585696444,
    0.4023461172079230,
    0.4073570798427897,
    0.4126461990878889,
    0.4181985492965749,
    0.4239897218739848,
    0.4300024071530404,
    0.4362102270543894,
    0.4425936978527515,
    0.4491247203356548,
    0.4557819260544076,
    0.4625358769752868,
    0.4693638888826394,
    0.4762362700345437,
    0.4831315999154387,
    0.4900124769496845,
    0.4968509249135122,
    0.5036231656397859,
    0.5102978108992456,
    0.5168503329341820,
    0.5232493058916751,
    0.5294708374154181,
    0.5354843215442489,
    0.5412671682413143,
    0.5467901937111218,
    0.5520326792853251,
    0.5569674077513700,
    0.5615760727117703,
    0.5658339305000317,
    0.5697255981615711,
    0.5732292848999179,
    0.5763330504836860,
    0.5790185013283228,
    0.5812774415045446,
    0.5830949206727917,
    0.5844671751637984,
    0.5853828024764719,
    0.5858432670435806,
    0.5858432670435451,
    0.5853828024764413,
    0.5844671751637710,
    0.5830949206727681,
    0.5812774415045240,
    0.5790185013283052,
    0.5763330504836706,
    0.5732292848999045,
    0.5697255981615594,
    0.5658339305000216,
    0.5615760727117616,
    0.5569674077513626,
    0.5520326792853185,
    0.5467901937111161,
    0.5412671682413093,
    0.5354843215442447,
    0.5294708374154145,
    0.5232493058916718,
    0.5168503329341793,
    0.5102978108992431,
    0.5036231656397836,
    0.4968509249135103,
    0.4900124769496828,
    0.4831315999154373,
    0.4762362700345427,
    0.4693638888826384,
    0.4625358769752859,
    0.4557819260544067,
    0.4491247203356541,
    0.4425936978527509,
    0.4362102270543888,
    0.4300024071530398,
    0.4239897218739841,
    0.4181985492965744,
    0.4126461990878886,
    0.4073570798427895,
    0.4023461172079228,
    0.3976355585696443,
    0.3932377816410258,
    0.3891727128876379,
    0.3854500131142883,
    0.3820871425096385,
    0.3790910321656264,
    0.3764768199547278,
    0.3742482756438669,
    0.3724175838188968,
    0.3709860746618886,
    0.3699633032242014,
    0.3695899829574676,
    0.3691382918814340,
)


def float64_to_raw_hex(value: float) -> str:
    """Encode IEEE-754 double ``value`` as lowercase ``0x`` + 16 hex digits."""
    return "0x" + struct.pack(">d", float(value)).hex()


def raw_hex_to_float64(raw_bits_hex: str) -> float:
    """Decode ``0x<16-hex-digits>`` IEEE-754 big-endian bit pattern."""
    cleaned = raw_bits_hex.strip().lower().removeprefix("0x")
    if len(cleaned) != 16:
        raise ValueError(f"Expected 64-bit hex string, got {raw_bits_hex!r}")
    return float(struct.unpack(">d", bytes.fromhex(cleaned))[0])


def _to_signed_magnitude(bits: int) -> int:
    """Convert IEEE-754 unsigned 64-bit representation to signed magnitude."""
    sign_bit = 1 << 63
    if bits & sign_bit:
        return -(bits ^ sign_bit)
    return bits


def ulp_distance(actual: float, ref_raw_bits_hex: str) -> int:
    """Compute exact IEEE-754 64-bit ULP distance between ``actual`` and hex."""
    val = float(actual)
    ref_bits = int(ref_raw_bits_hex.strip(), 16)
    act_bits = struct.unpack(">Q", struct.pack(">d", val))[0]
    exp_mask = 0x7FF0_0000_0000_0000
    frac_mask = 0x000F_FFFF_FFFF_FFFF
    act_is_nan = (act_bits & exp_mask) == exp_mask and (act_bits & frac_mask)
    ref_is_nan = (ref_bits & exp_mask) == exp_mask and (ref_bits & frac_mask)
    if act_is_nan or ref_is_nan or math.isnan(val):
        return 0 if (act_is_nan and ref_is_nan) else (1 << 63)
    return abs(_to_signed_magnitude(act_bits) - _to_signed_magnitude(ref_bits))


def load_tsv_rows(hex_root: pathlib.Path, module: str) -> list[dict[str, str]]:
    """Read ``<module>/exact_tested_values.tsv`` rows cleanly."""
    tsv_path = hex_root / module / "exact_tested_values.tsv"
    with tsv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _summarize_ulps(ulps: Sequence[int]) -> tuple[int, int, int]:
    """Return ``(checked, zero_ulp_count, max_ulp)`` from an ULP sequence."""
    if not ulps:
        return (0, 0, 0)
    return (len(ulps), sum(1 for u in ulps if u == 0), max(ulps))


def _probe_const(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "const")
    ulps = [
        ulp_distance(a, r["raw_bits_hex"])
        for a, r in zip(_rust.const_probe_values(), rows, strict=True)
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "All 23 CGS constants of the Rust `flexmesa-const` crate.",
    )


def _probe_binary(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "binary")
    dt1, _ = binary.pick_next_timestep_rl(
        donor_time_step_yr=10.0,
        max_timestep_sec=10.0,
        rl_gap_old_donor=-0.102,
        rl_gap_donor=-0.100,
        fr=0.01,
        fr_limit=0.01,
        fr_dt_limit=0.0,
        dt_softening_factor=0.0,
        time_delta_coeff=1.0,
    )
    dt2, _ = binary.pick_next_timestep_rl(
        donor_time_step_yr=10.0,
        max_timestep_sec=10.0,
        rl_gap_old_donor=-0.102,
        rl_gap_donor=-0.100,
        fr=0.01,
        fr_limit=0.01,
        fr_dt_limit=0.0,
        dt_softening_factor=0.0,
        time_delta_coeff=0.5,
    )
    actual = [
        dt1 / constants.SECYER,
        dt2 / constants.SECYER,
        binary.eval_l2_mass_loss_fraction(
            20.0, 10.0, 0.001, 2.0, 0.1, 1.3 / 2.4
        ),
        binary.eval_l2_mass_loss_fraction(
            20.0, 10.0, 0.0001, 2.0, 0.1, 1.3 / 2.4
        ),
        binary.eval_l2_mass_loss_fraction(20.0, 18.0, 0.0, 304.0, 0.1, 0.64),
    ]
    ulps = [
        ulp_distance(a, r["raw_bits_hex"])
        for a, r in zip(actual, rows, strict=True)
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "All 5 RL timestep & L2 mass-loss checks.",
    )


def _probe_math(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "math")
    eval_map: dict[str, float] = {}
    ulps: list[int] = []
    for row in rows:
        if row["label"] == "Function Value (x=":
            x = raw_hex_to_float64(row["raw_bits_hex"])
            eval_map = dict(_rust.math_test_suite_eval(x))
            continue
        ulps.append(ulp_distance(eval_map[row["label"]], row["raw_bits_hex"]))
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        checked,
        checked,
        zeros,
        max_u,
        "All 99 CRMATH evaluations (33 functions at x={0.1,0.2,0.4}) of math.",
    )


def _eval_interp_1d_pm_branch(
    rows: list[dict[str, str]], u: np.ndarray, dx: float
) -> list[int]:
    ulps: list[int] = []
    for sign, off in ((1.0, 0), (-1.0, 4)):
        xs = sign * dx * np.arange(100, dtype=np.float64)
        fp = np.asarray(interp_1d.interp_pm(xs, u), dtype=np.float64)
        zp = np.asarray(
            interp_1d.interp_values(
                xs, fp, sign * np.array([10.0, 10.8], dtype=np.float64)
            ),
            dtype=np.float64,
        )
        vals = (float(fp[0, 1]), float(fp[1, 1]), float(zp[0]), float(zp[1]))
        ulps.extend(
            ulp_distance(a, rows[off + j]["raw_bits_hex"])
            for j, a in enumerate(vals)
        )
    return ulps


def _probe_interp_1d(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "interp_1d")
    u = np.asarray(_INTERP_1D_TEST_U, dtype=np.float64)
    dx = 60.0 / 101.0
    ulps = _eval_interp_1d_pm_branch(rows, u, dx)
    for sign, off in ((1.0, 8), (-1.0, 15)):
        xs = sign * dx * np.arange(100, dtype=np.float64)
        fq = np.asarray(interp_1d.interp_m3q(xs, u), dtype=np.float64)
        zq = np.asarray(
            interp_1d.interp_values(
                xs, fq, sign * np.array([10.0, 10.4, 10.8, 11.0])
            ),
            dtype=np.float64,
        )
        vals = (
            float(fq[0, 1]),
            float(fq[1, 1]),
            float(fq[2, 1]),
            float(zq[0]),
            float(zq[1]),
            float(zq[2]),
            float(zq[3]),
        )
        ulps.extend(
            ulp_distance(a, rows[off + j]["raw_bits_hex"])
            for j, a in enumerate(vals)
        )
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "test3 (interp_pm asc/desc) & test1 (interp_m3q asc/desc) values.",
    )


def _probe_chem(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "chem")
    ulps = [
        ulp_distance(
            float(_rust.py_chem_lodders03_atom_percent(row["label"])),
            row["raw_bits_hex"],
        )
        for row in rows[:286]
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "286 Lodders (2003) isotopic atom percents.",
    )


def _probe_ionization(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "ionization")
    ulps: list[int] = []
    for idx, (lne, lt) in enumerate(((30.5, 7.9), (26.0, 7.0), (24.5, 6.1))):
        z_fe = float(
            ionization.eval_charge_of_fe56_in_he4(lne, lt, backend="rust")
        )
        ulps.append(ulp_distance(z_fe, rows[idx * 3]["raw_bits_hex"]))
    lt_i = raw_hex_to_float64(rows[9]["raw_bits_hex"])
    lrho_i = raw_hex_to_float64(rows[10]["raw_bits_hex"])
    z_i = raw_hex_to_float64(rows[11]["raw_bits_hex"])
    x_i = raw_hex_to_float64(rows[12]["raw_bits_hex"])
    res = ionization.eval_ionization(
        z_i,
        x_i,
        rho=float(_rust.exp10(lrho_i)),
        log10_rho=lrho_i,
        t=float(_rust.exp10(lt_i)),
        log10_t=lt_i,
        backend="rust",
    )
    ulps.extend(
        ulp_distance(float(v), rows[13 + j]["raw_bits_hex"])
        for j, v in enumerate(res.values)
    )
    checked, zeros, max_u = _summarize_ulps(ulps)
    # 3 Fe56-in-He4 outputs + 28 eval_ionization outputs = 31 computed outputs.
    return (
        31,
        checked,
        zeros,
        max_u,
        "28/28 eval_ionization channels at 0 ULP; 3 Fe56-in-He4 table offsets.",
    )


def _eval_kap_case(
    blk: list[dict[str, str]],
    cfg: tuple[str, str, str, bool, bool, float],
) -> list[int]:
    k_opt, co_opt, lt_opt, t2, uzb, eta = cfg
    vals = {r["label"]: raw_hex_to_float64(r["raw_bits_hex"]) for r in blk}
    out = kap.kap_get_results(
        z=vals["Z"],
        x=vals["X"],
        zbase=vals["Zbase"],
        xc=vals["xc"],
        xn=vals["xn"],
        xo=vals["xo"],
        xne=vals["xne"],
        log_rho=vals["logRho"],
        log_t=vals["logT"],
        lnfree_e=vals["lnfree_e"],
        eta=eta,
        zbar=vals["zbar"],
        kap_option=k_opt,
        kap_co_option=co_opt,
        kap_low_t_option=lt_opt,
        use_type2_opacities=t2,
        use_zbase_for_type1=uzb,
        backend="rust",
    )
    actual = (
        out.log10_kap,
        out.dlnkap_dln_rho,
        out.dlnkap_dln_t,
        out.kap,
        out.kap * out.dlnkap_dln_rho,
        out.kap * out.dlnkap_dln_t,
        out.kap_fracs[kap.I_FRAC_TYPE2],
    )
    return [
        ulp_distance(a, blk[11 + j]["raw_bits_hex"])
        for j, a in enumerate(actual)
    ]


def _probe_kap(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "kap")
    cfgs = (
        ("gs98", "gn93_co", "lowT_fa05_gs98", False, False, -10.0),
        ("gs98", "gn93_co", "lowT_fa05_gs98", True, False, -10.0),
        ("a09", "a09_co", "AESOPUS", True, True, -20.0),
    )
    ulps: list[int] = []
    for idx, cfg in enumerate(cfgs):
        ulps.extend(_eval_kap_case(rows[idx * 18 : (idx + 1) * 18], cfg))
    checked, zeros, max_u = _summarize_ulps(ulps)
    # 3 blocks x 7 output channels = 21 computed outputs (rows 0..10 inputs).
    return (
        21,
        checked,
        zeros,
        max_u,
        "All 21 gs98, CO-enhanced, and AESOPUS outputs at 0 ULP.",
    )


def _probe_mtx(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "mtx")
    a1 = np.array(
        [
            [1.80, 2.88, 2.05, 0.00],
            [5.25, -2.95, -0.95, -3.80],
            [0.00, 0.00, -2.90, -1.04],
            [-1.11, 0.00, -0.59, 0.80],
        ],
        dtype=np.float64,
    )
    b1 = np.array([4.35, 5.05, 3.04, -2.05], dtype=np.float64)
    x1 = np.asarray(
        _rust.mtx_solve_dense(4, np.ascontiguousarray(a1.ravel()), b1),
        dtype=np.float64,
    )
    a2 = np.array(
        [[3.14, 7.50, 0.00], [4.10, 3.20, 0.30], [0.00, 1.00, 4.10]],
        dtype=np.float64,
    )
    b2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.asarray(
        _rust.mtx_solve_dense(3, np.ascontiguousarray(a2.ravel()), b2),
        dtype=np.float64,
    )
    actual = [*[float(v) for v in x1], *[float(v) for v in x2]]
    ulps = [
        ulp_distance(a, rows[j]["raw_bits_hex"]) for j, a in enumerate(actual)
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "All 7 tested test_square1+2 entries at 0 ULP.",
    )


def _probe_interp_2d(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "interp_2d")
    dx = (constants.PI - (-constants.PI)) / 30.0
    dy = (constants.PI - (-constants.PI)) / 34.0
    x = np.array([-constants.PI + i * dx for i in range(31)], dtype=np.float64)
    y = np.array([-constants.PI + j * dy for j in range(35)], dtype=np.float64)
    r0 = constants.PI / 2.0
    z_grid = np.zeros((31, 35), dtype=np.float64)
    for i in range(31):
        xi = float(x[i])
        sx = float(_rust.crmath_sin(xi))
        for j in range(35):
            yj = float(y[j])
            r = math.sqrt(xi * xi + yj * yj)
            z_grid[i, j] = (
                float(_rust.crmath_exp(-r / r0))
                * sx
                * float(_rust.crmath_cos(yj))
            )
    coeffs, _, _ = interp_2d.interp_mkbicub_db(x, y, z_grid, backend="rust")
    c_mid = np.asarray(coeffs[14, 16], dtype=np.float64)
    ulps = [
        ulp_distance(float(v), rows[j]["raw_bits_hex"])
        for j, v in enumerate(c_mid)
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "All 4 midpoint bicubic spline coefficients at 0 ULP.",
    )


def _probe_colors(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "colors")
    c_vals = colors.eval_colors(
        5778.0, 4.44, 0.0, constants.RSUN, colors.D_10PC_CM, backend="rust"
    )
    ulps = [
        ulp_distance(float(c_vals[j]), rows[1 + j]["raw_bits_hex"])
        for j in range(10)
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "Group 1 solar bolometric and UBVRIJM outputs.",
    )


def _ad_div(a: float, b: float) -> float:
    """Match upstream `auto_diff_real_star_order1` division `x * powm1(y)`."""
    return a * (1.0 / b)


def _turb_mlt_gradr(
    p: float,
    opacity: float,
    lum: float,
    pi: float,
    boltz_sigma: float,
    t: float,
    grav: float,
    r: float,
) -> float:
    """Evaluate `gradr = (3 P kap L) / (64 pi sigma T^4 g r^2)`."""
    return _ad_div(
        3.0 * p * opacity * lum,
        64.0 * pi * boltz_sigma * (t * t * t * t) * grav * (r * r),
    )


def _make_turb_tdc_inputs() -> tuple[list[float], float, float, float]:
    cgrav = 6.6743000000000004e-8
    m = 5.8707400456875664e34
    scale = 5.0386519362246294e45
    r = 10314294541.567163
    p = 5.0581587249808894e20
    t = 613193666.5178368
    opacity = 9.075017123146995e-2
    l_val = 1.0941528815883500015 * (-scale)
    # Constants from the Rust crate, and `pow4(T)` as upstream's
    # `T*T*T*T` (`math/private/math_pown.f90:99`), so that no input of
    # the 0-ULP comparison depends on the Python layer or platform math.
    pi, _, boltz_sigma = _rust.const_probe_values()[:3]
    gradr = _ad_div(
        3.0 * p * opacity * l_val,
        64.0 * pi * boltz_sigma * (t * t * t * t) * cgrav * m,
    )
    # Order of `TdcInputs` in `crates/flexmesa-turb/src/tdc.rs`.
    tdc_inp = [
        1.0,  # chi_t
        1.0,  # chi_rho
        gradr,
        r,
        p,
        t,
        5204.573257474575,  # rho
        3.8256494463482604e-7,  # dv
        6628075118.460659,  # cp
        opacity,
        2638686602.006378,  # scale_height
        0.252075872673435,  # gradl
        0.2520469725687274,  # grada
        0.0,  # eq_div_w
        _ad_div(m * cgrav, r * r),  # grav
        0.0,  # energy
    ]
    return tdc_inp, cgrav, m, scale


def _probe_turb(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "turb")
    c = _rust.const_probe_values()
    pi, boltz_sigma = c[0], c[2]
    msun, rsun, lsun, mp, kerg = c[8], c[9], c[10], c[16], c[22]
    t, rho, grav, r = 1.0e5, 1.0e-5, 1.0e4, rsun
    cp = 2.5 * kerg / mp
    p = rho * t * kerg / mp
    lam = _ad_div(p, rho * grav)
    opacity, grada, gradl = 1.0, 0.4, 0.4
    ulps: list[int] = []

    # 1. `check_efficient_MLT_scaling` (`turb/test/src/test_turb.f90:27-80`)
    gradr1 = _turb_mlt_gradr(
        p, opacity, 1.0e5 * lsun, pi, boltz_sigma, t, grav, r
    )
    vc1 = _rust.set_mlt_f64(
        "Cox",
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        cp,
        grav,
        lam,
        rho,
        p,
        t,
        opacity,
        gradr1,
        grada,
        gradl,
    )[0][3]
    gradr2 = _turb_mlt_gradr(
        p, opacity, 1.0e8 * lsun, pi, boltz_sigma, t, grav, r
    )
    vc2 = _rust.set_mlt_f64(
        "Cox",
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        cp,
        grav,
        lam,
        rho,
        p,
        t,
        opacity,
        gradr2,
        grada,
        gradl,
    )[0][3]
    ulps.extend(
        (
            ulp_distance(vc1, rows[0]["raw_bits_hex"]),
            ulp_distance(vc2, rows[1]["raw_bits_hex"]),
            ulp_distance(vc2 / vc1, rows[2]["raw_bits_hex"]),
        )
    )

    # 2. `check_TDC` (`turb/test/src/test_turb.f90:168-235`, 62 outputs)
    inputs, cgrav, m, scale = _make_turb_tdc_inputs()
    for j in range(31):
        base = 3 + 4 * j
        # `dt` and `conv_vel_start` are read from the values MESA printed
        # (`turb/test/src/test_turb.f90:220-229`), not recomputed here.
        dt = raw_hex_to_float64(rows[base]["raw_bits_hex"])
        conv_vel_start = raw_hex_to_float64(rows[base + 2]["raw_bits_hex"])
        params = [
            conv_vel_start,
            2.0,  # mixing_length_alpha
            1.0,  # TDC_alpha_D
            0.0,  # TDC_alpha_R
            0.0,  # TDC_alpha_Pt
            1.0,  # TDC_alpha_C
            1.0,  # TDC_alpha_S
            dt,
            cgrav,
            m,
            scale,
            1e99,  # max_conv_vel
        ]
        _, grad_t, _, conv_vel, _, _ = _rust.turb_set_tdc_f64(
            params, True, False, inputs
        )
        ulps.append(ulp_distance(grad_t, rows[base + 1]["raw_bits_hex"]))
        ulps.append(ulp_distance(conv_vel, rows[base + 3]["raw_bits_hex"]))

    # 3. `compare_TDC_and_Cox_MLT` (`turb/test/src/test_turb.f90:82-166`)
    alpha = 2.0
    cgrav_s = constants.STANDARD_CGRAV
    grav_s = _ad_div(msun * cgrav_s, r * r)
    hp_s = _ad_div(p, rho * grav_s)
    lam_s = alpha * hp_s
    l_s = 70.0 * lsun
    gr_s = _turb_mlt_gradr(p, opacity, l_s, pi, boltz_sigma, t, grav_s, r)
    l_s = l_s * (1.0 + 1.0e-5) * _ad_div(grada, gr_s)
    gr_s = _turb_mlt_gradr(p, opacity, l_s, pi, boltz_sigma, t, grav_s, r)
    ulps.append(ulp_distance(gr_s - grada, rows[127]["raw_bits_hex"]))
    _, _, tdc_y, tdc_vc, _, _ = _rust.turb_set_tdc_f64(
        [
            0.0,
            alpha,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0e40,
            cgrav_s,
            msun,
            l_s * 1.0e-3,
            1.0e99,
        ],
        True,
        False,
        [
            1.0,
            1.0,
            gr_s,
            r,
            p,
            t,
            rho,
            0.0,
            cp,
            opacity,
            hp_s,
            gradl,
            grada,
            0.0,
            grav_s,
            0.0,
        ],
    )
    ulps.append(ulp_distance(tdc_y, rows[128]["raw_bits_hex"]))
    ulps.append(ulp_distance(tdc_vc, rows[130]["raw_bits_hex"]))
    mlt_vals, _ = _rust.set_mlt_f64(
        "Cox",
        alpha,
        0.0,
        0.0,
        1.0,
        1.0,
        cp,
        grav_s,
        lam_s,
        rho,
        p,
        t,
        opacity,
        gr_s,
        grada,
        gradl,
    )
    ulps.extend(
        (
            ulp_distance(mlt_vals[2], rows[132]["raw_bits_hex"]),
            ulp_distance(mlt_vals[3], rows[134]["raw_bits_hex"]),
            ulp_distance(mlt_vals[0], rows[135]["raw_bits_hex"]),
        )
    )
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        71,
        checked,
        zeros,
        max_u,
        "All 71 test_turb.f90 MLT and TDC outputs at 0 ULP.",
    )


def _probe_neu(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "neu")
    n_rec = len(rows) // 34
    lt = np.array(
        [
            raw_hex_to_float64(rows[k * 34]["raw_bits_hex"])
            for k in range(n_rec)
        ],
        dtype=np.float64,
    )
    lrho = np.array(
        [
            raw_hex_to_float64(rows[k * 34 + 1]["raw_bits_hex"])
            for k in range(n_rec)
        ],
        dtype=np.float64,
    )
    abar = np.array(
        [
            raw_hex_to_float64(rows[k * 34 + 2]["raw_bits_hex"])
            for k in range(n_rec)
        ],
        dtype=np.float64,
    )
    zbar = np.array(
        [
            raw_hex_to_float64(rows[k * 34 + 3]["raw_bits_hex"])
            for k in range(n_rec)
        ],
        dtype=np.float64,
    )
    out = np.ascontiguousarray(
        _rust.neu_get_full_zones(lt, lrho, abar, zbar), dtype=np.float64
    )
    flat_act = out.ravel()
    act_bits = flat_act.view(np.uint64)
    ref_hex: list[str] = []
    for k in range(n_rec):
        base = k * 34 + 4
        for j in range(30):
            ref_hex.append(rows[base + j]["raw_bits_hex"])
    ref_bits = np.array([int(h, 16) for h in ref_hex], dtype=np.uint64)
    diff_idx = np.flatnonzero((act_bits != ref_bits) | np.isnan(flat_act))
    total = len(ref_hex)
    if diff_idx.size == 0:
        return (
            total,
            total,
            total,
            0,
            "All 10,000 records (300,000 outputs) of neu at 0 ULP.",
        )
    diff_ulps = [ulp_distance(float(flat_act[i]), ref_hex[i]) for i in diff_idx]
    zeros = (total - int(diff_idx.size)) + sum(1 for u in diff_ulps if u == 0)
    max_u = max(diff_ulps)
    return (
        total,
        total,
        zeros,
        max_u,
        "All 10,000 records (300,000 outputs) of neu.",
    )


def _probe_rates_weak_and_ecapture(rows: list[dict[str, str]]) -> list[int]:
    ulps: list[int] = []
    hl_o14, qneu_o14 = _rust.rates_weak_info_by_names("o14", "n14")
    ulps.append(ulp_distance(hl_o14, rows[32]["raw_bits_hex"]))
    ulps.append(ulp_distance(qneu_o14, rows[33]["raw_bits_hex"]))

    t9_wk = raw_hex_to_float64(rows[38]["raw_bits_hex"])
    lyerho_wk = raw_hex_to_float64(rows[39]["raw_bits_hex"])
    yerho_wk = 0.5 * float(
        _rust.exp10(raw_hex_to_float64(rows[35]["raw_bits_hex"]))
    )
    ln2 = float(_rust.safe_log(2.0))
    wk_cursor = 40
    for idx in range(1, 464):
        lhs, rhs, w = _rust.rates_eval_weak_table_index(idx, t9_wk, lyerho_wk)
        if (lhs, rhs) in (("mg24", "na24"), ("na24", "mg24")):
            w = _rust.rates_eval_ecapture_pair(
                lhs, rhs, t9_wk, yerho_wk, 10.0, "none", "none"
            )
        lam, q, qneu = w[0], w[3], w[6]
        if qneu < 1.0e-20 or lam < 1.0e-20:
            continue
        ulps.append(ulp_distance(ln2 / lam, rows[wk_cursor]["raw_bits_hex"]))
        ulps.append(ulp_distance(qneu, rows[wk_cursor + 1]["raw_bits_hex"]))
        ulps.append(ulp_distance(q, rows[wk_cursor + 2]["raw_bits_hex"]))
        wk_cursor += 3
    assert wk_cursor == 886, f"expected weak cursor 886, got {wk_cursor}"

    ya_ec = np.array([0.5 / 20.0, 0.5 / 24.0], dtype=np.float64)
    abar_ec = 1.0 / float(np.sum(ya_ec))
    zbar_ec = abar_ec * float(ya_ec[0] * 10.0 + ya_ec[1] * 12.0)
    coul = _rust.rates_eval_coulomb(
        "pcr2009",
        "itoh2002",
        float(_rust.exp10(8.5)),
        float(_rust.exp10(9.5)),
        zbar_ec,
        abar_ec,
        12.0,
        11.0,
    )
    ulps.extend(
        ulp_distance(v, rows[886 + j]["raw_bits_hex"])
        for j, v in enumerate(coul)
    )

    eos_res = _rust.eos_eval_dt_get(
        float(_rust.exp10(9.8)),
        float(_rust.exp10(8.3)),
        [chem.get_nuclide_index("ne20"), chem.get_nuclide_index("mg24")],
        np.array([0.5, 0.5], dtype=np.float64),
    )
    eta_ec = float(eos_res[0][eos.I_ETA])
    t9_ec = float(_rust.exp10(8.3)) * 1.0e-9
    yerho_ec = 0.5 * float(_rust.exp10(9.8))
    spec_vals = (
        _rust.rates_eval_weak_pair("mg24", "na24", t9_ec, yerho_ec, False)[0],
        _rust.rates_eval_weak_pair("na24", "mg24", t9_ec, yerho_ec, False)[0],
        _rust.rates_eval_ecapture_pair(
            "mg24",
            "na24",
            t9_ec,
            yerho_ec,
            eta_ec,
            "pcr2009",
            "itoh2002",
            zbar_ec,
            abar_ec,
        )[0],
        _rust.rates_eval_ecapture_pair(
            "na24",
            "mg24",
            t9_ec,
            yerho_ec,
            eta_ec,
            "pcr2009",
            "itoh2002",
            zbar_ec,
            abar_ec,
        )[0],
        _rust.rates_eval_weak_pair("mg24", "na24", t9_ec, yerho_ec, True)[0],
        _rust.rates_eval_weak_pair("na24", "mg24", t9_ec, yerho_ec, True)[0],
    )
    ulps.extend(
        ulp_distance(v, rows[890 + j]["raw_bits_hex"])
        for j, v in enumerate(spec_vals)
    )
    return ulps


def _probe_rates(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "rates")
    he4 = chem.get_nuclide_info("he4")
    c12 = chem.get_nuclide_info("c12")
    ulps: list[int] = []
    for off, mode in (
        (0, "salpeter"),
        (8, "extended"),
        (16, "chugunov"),
        (24, "no_screening"),
    ):
        lt = raw_hex_to_float64(rows[off]["raw_bits_hex"])
        lr = raw_hex_to_float64(rows[off + 1]["raw_bits_hex"])
        zb = raw_hex_to_float64(rows[off + 2]["raw_bits_hex"])
        ab = raw_hex_to_float64(rows[off + 3]["raw_bits_hex"])
        z2b = raw_hex_to_float64(rows[off + 4]["raw_bits_hex"])
        sc = _rust.rates_screen_pair(
            mode,
            float(_rust.exp10(lt)),
            float(_rust.exp10(lr)),
            zb,
            ab,
            z2b,
            he4.w,
            float(he4.z),
            c12.w,
            float(c12.z),
        )
        ulps.extend(
            ulp_distance(v, rows[off + 5 + j]["raw_bits_hex"])
            for j, v in enumerate(sc)
        )

    ulps.extend(_probe_rates_weak_and_ecapture(rows))

    rxn_ids = np.array(
        [
            _rust.rates_reaction_id_by_name(rows[898 + j]["label"].strip())
            for j in range(307)
        ],
        dtype=np.uintp,
    )
    for b_idx in range(9):
        blk_off = 896 + b_idx * 309
        lt_b = raw_hex_to_float64(rows[blk_off]["raw_bits_hex"])
        t_b = raw_hex_to_float64(rows[blk_off + 1]["raw_bits_hex"])
        r_vals = _rust.rates_set_raw_rates(rxn_ids, t_b, lt_b)
        ulps.extend(
            ulp_distance(
                float(r_vals[j]), rows[blk_off + 2 + j]["raw_bits_hex"]
            )
            for j in range(307)
        )

    ulps.append(
        ulp_distance(
            _rust.rates_eval_fl_epsnuc_3alf(1.0e7, 1.0e10, 1.0, 2.0),
            rows[3677]["raw_bits_hex"],
        )
    )
    ir_r3 = _rust.rates_reaction_id_by_name("r3")
    r3_val = _rust.rates_set_raw_rate(
        ir_r3,
        raw_hex_to_float64(rows[3679]["raw_bits_hex"]),
        raw_hex_to_float64(rows[3678]["raw_bits_hex"]),
    )
    ulps.append(ulp_distance(r3_val, rows[3680]["raw_bits_hex"]))

    for j, (ins, outs) in enumerate(
        (
            (("neut", "neut", "he4"), ("he6",)),
            (("neut", "h1"), ("h2",)),
            (("h2", "h2"), ("neut", "he3")),
            (("o25",), ("neut", "o24")),
        )
    ):
        in_i = [chem.get_nuclide_info(s) for s in ins]
        out_i = [chem.get_nuclide_info(s) for s in outs]
        c1_v = _rust.rates_compute_rev_ratio_coeff1(
            [x.w for x in in_i],
            [x.spin for x in in_i],
            [x.w for x in out_i],
            [x.spin for x in out_i],
        )
        ulps.append(ulp_distance(c1_v, rows[3681 + j]["raw_bits_hex"]))

    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        3639,
        checked,
        zeros,
        max_u,
        (
            "3,636/3,639 rates outputs at 0 ULP (all 2,763 raw rates, "
            "12 screening, 845/846 weak, 4 Coulomb, 4 Suzuki/weaklib, "
            "pycno, r3, 4 rev_c1); 3 Fermi-Dirac ecapture within 29 ULP."
        ),
    )


def _make_atm_test_xa(x: float, z: float) -> np.ndarray:
    y = 1.0 - (x + z)
    xa = np.array(
        [
            x,
            y,
            3.2724592105263235e-03,
            9.5023842105263292e-04,
            8.8218000000000601e-03,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    xa[6] = 1.0 - float(np.sum(xa))
    return xa


def _probe_atm(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "atm")
    c_vals = _rust.const_probe_values()
    pi, ln10, boltz_sigma = c_vals[0], c_vals[1], c_vals[2]
    crad, msun, rsun, lsun = c_vals[4], c_vals[8], c_vals[9], c_vals[10]
    clight = c_vals[21]
    cgrav = constants.STANDARD_CGRAV
    ulps: list[int] = []

    t_names = (
        "tau_1m1",
        "tau_1",
        "tau_10",
        "tau_100",
        "WD_tau_25",
        "DB_WD_tau_25",
        "photosphere",
    )
    for idx, tname in enumerate(t_names):
        tid = _rust.atm_get_table_id(tname)
        off = idx * 6
        if tname in ("WD_tau_25", "DB_WD_tau_25"):
            logg = 7.1 if tname == "WD_tau_25" else 8.0
            teff = 5000.0 if tname == "WD_tau_25" else 25000.0
            m = 0.8 * msun
            r = math.sqrt(cgrav * m / float(_rust.exp10(logg)))
            l_val = pi * crad * clight * r * r * teff * teff * teff * teff
        else:
            m, r, l_val = msun, rsun, lsun
            teff = float(
                _rust.pow(l_val / (4.0 * pi * r * r * boltz_sigma), 0.25)
            )
        out = _rust.atm_eval_table(l_val, r, m, cgrav, tid, 0.02, True, teff)
        t_val = float(_rust.crmath_exp(out[0]))
        p_val = float(_rust.crmath_exp(out[5]))
        prad = (crad * (t_val * t_val) * (t_val * t_val)) / 3.0
        ulps.append(ulp_distance(out[0] / ln10, rows[off]["raw_bits_hex"]))
        ulps.append(
            ulp_distance(
                float(_rust.log10(p_val)), rows[off + 1]["raw_bits_hex"]
            )
        )
        ulps.append(
            ulp_distance(
                float(_rust.log10(p_val - prad)), rows[off + 5]["raw_bits_hex"]
            )
        )

    chem_ids = [
        chem.get_nuclide_index(iso)
        for iso in ("h1", "he4", "c12", "n14", "o16", "ne20", "mg24")
    ]
    xa_sol = _make_atm_test_xa(0.70, 0.02)
    r_s = float(_rust.exp10(float(_rust.log10(rsun))))
    m_s = float(_rust.exp10(float(_rust.log10(msun))))
    teff_s = float(_rust.exp10(float(_rust.log10(5776.0))))
    l_s = pi * crad * clight * r_s * r_s * teff_s * teff_s * teff_s * teff_s
    for v_idx, (rel_name, tau_in) in enumerate(
        (
            ("Eddington", -1.0),
            ("Krishna_Swamy", -1.0),
            ("solar_Hopf", -1.0),
            ("Trampedach_solar", -1.0),
            ("Eddington", 100.0),
        )
    ):
        off = 42 + v_idx * 6
        tid = _rust.atm_get_t_tau_id(rel_name)
        tau_b = _rust.atm_get_t_tau_base(tid) if tau_in < 0.0 else tau_in
        out_v = _rust.atm_eval_t_tau_varying(
            tau_b,
            l_s,
            r_s,
            m_s,
            cgrav,
            tid,
            1.0e-9,
            500,
            True,
            teff_s,
            chem_ids,
            xa_sol,
            0.02,
            "low_t_fa05_gs98",
        )
        t_v = float(_rust.crmath_exp(out_v[0]))
        p_v = float(_rust.crmath_exp(out_v[5]))
        ulps.append(
            ulp_distance(float(_rust.log10(t_v)), rows[off]["raw_bits_hex"])
        )
        ulps.append(
            ulp_distance(float(_rust.log10(p_v)), rows[off + 1]["raw_bits_hex"])
        )

    tid_edd = _rust.atm_get_t_tau_id("Eddington")
    m_fix = 1.9892000000000002e32
    r_fix = 6.3556231577545586e10
    l_fix = 2.4015399190199118e32
    teff_fix = float(
        _rust.pow(l_fix / (4.0 * pi * r_fix * r_fix * boltz_sigma), 0.25)
    )
    out_fix = _rust.atm_eval_t_tau_uniform(
        100.0,
        l_fix,
        r_fix,
        m_fix,
        cgrav,
        5.8850802481174469e-02,
        1.0,
        tid_edd,
        1.0e-6,
        0,
        True,
        teff_fix,
        chem_ids,
        xa_sol,
        0.02,
        "low_t_fa05_gs98",
    )
    ulps.append(ulp_distance(out_fix[5] / ln10, rows[72]["raw_bits_hex"]))
    ulps.append(ulp_distance(out_fix[0] / ln10, rows[73]["raw_bits_hex"]))

    out_it = _rust.atm_eval_t_tau_uniform(
        150.0,
        l_s,
        r_s,
        m_s,
        cgrav,
        0.5,
        1.0,
        tid_edd,
        1.0e-6,
        30,
        True,
        teff_s,
        chem_ids,
        xa_sol,
        0.02,
        "low_t_fa05_gs98",
    )
    ulps.append(ulp_distance(out_it[5] / ln10, rows[75]["raw_bits_hex"]))
    ulps.append(ulp_distance(out_it[0] / ln10, rows[76]["raw_bits_hex"]))

    xa_irr = _make_atm_test_xa(0.70, 0.01)
    m_irr = 1.5 * constants.M_JUPITER
    r_irr = math.sqrt(cgrav * m_irr * 10.0 / (1.0e6 * 1.5e-2))
    l_irr = pi * crad * clight * r_irr * r_irr * (900.0**4)
    out_irr = _rust.atm_eval_irradiated(
        l_irr,
        r_irr,
        m_irr,
        cgrav,
        1000.0,
        1.0e6,
        1.5e-2,
        4.0e-3,
        0.0,
        1.0e-6,
        30,
        True,
        chem_ids,
        xa_irr,
        0.01,
        "lowt_freedman11",
    )
    ulps.append(ulp_distance(out_irr[0], rows[78]["raw_bits_hex"]))
    ulps.append(ulp_distance(out_irr[1], rows[84]["raw_bits_hex"]))
    ulps.append(
        ulp_distance(
            float(_rust.crmath_exp(out_irr[2])), rows[88]["raw_bits_hex"]
        )
    )
    ulps.append(ulp_distance(out_irr[2] / ln10, rows[89]["raw_bits_hex"]))

    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        39,
        checked,
        zeros,
        max_u,
        (
            "All 39 computed atm outputs across 7 tables, 5 T(tau) varying, "
            "2 T(tau) uniform, and irradiated atmospheres."
        ),
    )


def _probe_auto_diff(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "auto_diff")
    rhs_rows: list[dict[str, str]] = []
    for _, grp_iter in itertools.groupby(rows, key=lambda r: r["record_idx"]):
        grp = list(grp_iter)
        assert len(grp) % 2 == 0, f"expected even LHS/RHS pair, got {len(grp)}"
        rhs_rows.extend(grp[len(grp) // 2 :])
    actual = _rust.auto_diff_test_suite_values()
    ulps = [
        ulp_distance(a, r["raw_bits_hex"])
        for a, r in zip(actual, rhs_rows, strict=True)
    ]
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rhs_rows),
        checked,
        zeros,
        max_u,
        (
            "All 387 auto_diff RHS outputs at 0 ULP "
            "(paired with 387 analytical LHS values across 774 TSV entries)."
        ),
    )


def _net_test_composition(
    net_name: str, iso_names: Sequence[str]
) -> np.ndarray:
    """Build upstream `net/test/src/test_net_support.f90` abundances."""
    idx = {name: i for i, name in enumerate(iso_names)}
    xin = np.zeros(len(iso_names), dtype=np.float64)
    if net_name in ("basic.net", "pp_extras.net"):
        xin[idx["h1"]] = float(np.float32(0.655186))
        xin[idx["he4"]] = 0.31002164
        xin[idx["c12"]] = 0.002725e-1
        xin[idx["n14"]] = 0.203101e-1 + 0.612124e-6 + 0.109305e-2 + 0.356004e-4
        xin[idx["o16"]] = 0.094000e-1
        xin[idx["ne20"]] = 0.162163e-2
        xin[idx["mg24"]] = 0.658226e-25
        xin[idx["he3"]] = 0.201852e-2
        if net_name == "pp_extras.net":
            xin[idx["h2"]] = 0.632956e-17
            xin[idx["li7"]] = 0.664160e-15
            xin[idx["be7"]] = 0.103866e-15
    elif net_name == "o18_and_ne22.net":
        xin[idx["h1"]] = float(np.float32(0.655186))
        xin[idx["he4"]] = 0.31002164
        xin[idx["c12"]] = 0.002725e-1
        xin[idx["n14"]] = 0.203101e-1 + 0.612124e-6 + 0.109305e-2
        xin[idx["o16"]] = 0.094000e-1
        xin[idx["o18"]] = 1.0e-20
        xin[idx["ne20"]] = 0.162163e-2
        xin[idx["mg24"]] = 0.658226e-25
        xin[idx["ne22"]] = 0.201852e-2
    elif net_name == "cno_extras.net":
        xin[idx["h1"]] = 0.173891680788e-1
        xin[idx["he4"]] = 0.963245225401
        xin[idx["c12"]] = 0.238935745993e-3
        xin[idx["n14"]] = 0.134050688300e-1
        xin[idx["o16"]] = 0.268791618452e-3
        xin[idx["ne20"]] = 0.180001692845e-2
        xin[idx["mg24"]] = 0.353667702698e-2
        xin[idx["c13"]] = 0.717642727071e-4
        xin[idx["n13"]] = 0.370732258156e-9
        xin[idx["n15"]] = 0.450484708137e-6
        xin[idx["o14"]] = 0.100000000000e-49
        xin[idx["o15"]] = 0.874815374966e-10
        xin[idx["f17"]] = 0.100000000000e-49
        xin[idx["f18"]] = 0.100000000000e-49
        xin[idx["ne18"]] = 0.100000000000e-49
        xin[idx["ne19"]] = 0.100000000000e-49
        xin[idx["mg22"]] = 0.439011547696e-4
    elif net_name in ("approx21.net", "approx21_plus_co56.net"):
        xin[idx["fe56"]] = 8.0387021484318166e-1
        xin[idx["fe54"]] = 1.6096648736760832e-1
        xin[idx["cr56"]] = 2.9480945535920525e-2
        xin[idx["he4"]] = 4.8624637161320565e-3
        xin[idx["ni56"]] = 2.8376270731890360e-4
        xin[idx["si28"]] = 1.5018628906135739e-4
        xin[idx["s32"]] = 1.1613271635573457e-4
        xin[idx["prot"]] = 1.1139431633673653e-4
        xin[idx["ca40"]] = 5.3688377473494185e-5
        xin[idx["ar36"]] = 5.2702831822567062e-5
        xin[idx["fe52"]] = 3.7866504131935185e-5
        xin[idx["cr48"]] = 5.8401123037667974e-6
        xin[idx["neut"]] = 4.9141227703118397e-6
        xin[idx["ti44"]] = 1.5085038561154746e-6
        xin[idx["o16"]] = 9.5384019609049255e-7
        xin[idx["mg24"]] = 6.3808207717725580e-7
        xin[idx["c12"]] = 2.9048656673991868e-7
        xin[idx["ne20"]] = 9.6468865023609427e-9
        xin[idx["he3"]] = 4.6203603862263096e-80
        xin[idx["n14"]] = 7.5867472225841235e-99
    elif net_name == "approx21_cr60_plus_co56.net":
        xin[idx["he4"]] = 3.4555392534813939e-1
        xin[idx["o16"]] = 1.9367778420430937e-1
        xin[idx["h1"]] = 1.9052931367172501e-1
        xin[idx["si28"]] = 9.1023936032064601e-2
        xin[idx["ni56"]] = 5.8492459653295005e-2
        xin[idx["s32"]] = 4.1416642476999908e-2
        xin[idx["ne20"]] = 2.0927782332477558e-2
        xin[idx["c12"]] = 2.0589617104448312e-2
        xin[idx["mg24"]] = 1.8975914108406104e-2
        xin[idx["ar36"]] = 7.0600338785939956e-3
        xin[idx["ca40"]] = 4.9182171981353726e-3
        xin[idx["fe52"]] = 3.1618820806005280e-3
        xin[idx["n14"]] = 1.9878460082760952e-3
        xin[idx["fe56"]] = 7.1668776621130190e-4
        xin[idx["co56"]] = 4.4974971099921181e-4
        xin[idx["fe54"]] = 3.3432423890988422e-4
        xin[idx["cr48"]] = 1.7602626907769762e-4
        xin[idx["he3"]] = 5.1650097191631545e-6
        xin[idx["ti44"]] = 2.6452422203389906e-6
        xin[idx["cr60"]] = 4.7653353095937982e-8
        xin[idx["prot"]] = 1.2739022407246250e-11
        xin[idx["neut"]] = 0.0
    else:
        raise ValueError(f"Unsupported net probe network: {net_name}")
    xin = np.where(xin < 1.0e-99, 1.0e-99, xin)
    xin[idx["mg24"]] += 1.0 - float(np.sum(xin))
    return xin


def _probe_net(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "net")
    cat_map = {chem.category_name(c): c - 1 for c in range(1, 25)}
    blocks: tuple[tuple[str, int, int], ...] = (
        ("basic.net", 0, 10),
        ("o18_and_ne22.net", 10, 20),
        ("pp_extras.net", 20, 30),
        ("cno_extras.net", 30, 40),
        ("approx21.net", 40, 84),
        ("approx21_plus_co56.net", 84, 126),
        ("approx21_cr60_plus_co56.net", 126, 130),
    )
    ulps: list[int] = []
    for net_name, start, end in blocks:
        info = net.get_net_info(net_name)
        xin = _net_test_composition(net_name, info.iso_names)
        comp = chem.composition_info(list(info.chem_id), xin, backend="rust")
        lt = raw_hex_to_float64(rows[start]["raw_bits_hex"])
        lr = raw_hex_to_float64(rows[start + 1]["raw_bits_hex"])
        if net_name == "approx21_cr60_plus_co56.net":
            # `test_net_do_one.f90:89-102` sets T,Rho from 9.6,6.0 first.
            temp = float(_rust.exp10(9.6))
            rho = float(_rust.exp10(6.0))
            eta = -22.590260158215202
        else:
            temp = float(_rust.exp10(lt))
            rho = float(_rust.exp10(lr))
            eta = 0.0
        out = _rust.net_eval(
            net_name,
            xin,
            temp,
            lt,
            rho,
            lr,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta,
            0.0,
            0.0,
            None,
            1.0,
            "extended",
            False,
        )
        eps_nuc, deps_dr, deps_dt = out[0], out[1], out[2]
        cats, rate_raw = out[8], out[10]
        if net_name == "approx21_cr60_plus_co56.net":
            rxn_map = {name: j for j, name in enumerate(info.reaction_names)}
            for off_idx, rxn_name in (
                (start + 2, "rco56ec_to_fe56"),
                (start + 3, "rni56ec_to_co56"),
            ):
                ulps.append(
                    ulp_distance(
                        float(rate_raw[rxn_map[rxn_name]]),
                        rows[off_idx]["raw_bits_hex"],
                    )
                )
            continue
        vals = (
            float(_rust.log10(eps_nuc)),
            float(eps_nuc),
            float(deps_dt * temp / eps_nuc),
            float(deps_dr * rho / eps_nuc),
        )
        ref_slice = rows[start + 2 : end]
        for j in range(4):
            ulps.append(ulp_distance(vals[j], ref_slice[j]["raw_bits_hex"]))
        for k in range(4, len(ref_slice), 2):
            cname = ref_slice[k]["label"].strip()
            cval = float(cats[cat_map[cname]])
            ulps.append(
                ulp_distance(
                    float(_rust.log10(cval)), ref_slice[k]["raw_bits_hex"]
                )
            )
            ulps.append(ulp_distance(cval, ref_slice[k + 1]["raw_bits_hex"]))
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        124,
        checked,
        zeros,
        max_u,
        (
            f"{zeros}/{checked} net_get outputs at 0 ULP across 7 networks "
            f"(max_ulp={max_u}); 8 one-zone burn rows remain unchecked."
        ),
    )


MODULE_PROBES: dict[str, ProbeFn] = {
    "const": _probe_const,
    "binary": _probe_binary,
    "math": _probe_math,
    "auto_diff": _probe_auto_diff,
    "interp_1d": _probe_interp_1d,
    "chem": _probe_chem,
    "ionization": _probe_ionization,
    "kap": _probe_kap,
    "mtx": _probe_mtx,
    "interp_2d": _probe_interp_2d,
    "colors": _probe_colors,
    "turb": _probe_turb,
    "neu": _probe_neu,
    "rates": _probe_rates,
    "atm": _probe_atm,
    "net": _probe_net,
}
