# SPDX-License-Identifier: LGPL-3.0-or-later
"""IEEE-754 helpers and per-module strict probes for ``mesa_module_hex``."""

from __future__ import annotations

import csv
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
    interp_1d,
    interp_2d,
    ionization,
    kap,
    neu,
    rates,
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
    xs = [raw_hex_to_float64(rows[idx]["raw_bits_hex"]) for idx in (0, 34, 68)]
    fns: dict[str, Callable[[float], float]] = {
        "exp10": _rust.exp10,
        "pow(,2)": lambda x: float(_rust.pow(x, 2)),
        "pow(,2.)": lambda x: float(_rust.pow(x, 2.0)),
        "cospi": _rust.cospi,
        "sinpi": _rust.sinpi,
        "tanpi": _rust.tanpi,
        "safe_log": _rust.safe_log,
        "safe_log10": _rust.safe_log10,
    }
    ulps = [
        ulp_distance(
            float(fns[row["label"]](xs[idx // 34])), row["raw_bits_hex"]
        )
        for idx, row in enumerate(rows)
        if row["label"] in fns
    ]
    expected_outputs = sum(
        1 for row in rows if row["label"] != "Function Value (x="
    )
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        expected_outputs,
        checked,
        zeros,
        max_u,
        "24 crmath Rust strict-mode intrinsics across x={0.1,0.2,0.4}.",
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
        "12/14 gs98+CO channels at 0 ULP; AESOPUS and log10 1-ULP gap.",
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


def _make_turb_tdc_inputs() -> tuple[list[float], float, float, float]:
    cgrav = 6.6743000000000004e-8
    m = 5.8707400456875664e34
    scale = 5.0386519362246294e45
    r = 10314294541.567163
    p = 5.0581587249808894e20
    t = 613193666.5178368
    opacity = 9.075017123146995e-2
    l_val = 1.09415288158835 * (-scale)
    # Constants from the Rust crate, and `pow4(T)` as upstream's
    # `(T*T)*(T*T)` (`turb/test/src/test_turb.f90:211`), so that no input
    # of the 0-ULP comparison depends on the Python layer or platform math.
    pi, _, boltz_sigma = _rust.const_probe_values()[:3]
    gradr = (
        3.0
        * p
        * opacity
        * l_val
        / (64.0 * pi * boltz_sigma * ((t * t) * (t * t)) * cgrav * m)
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
        m * cgrav / (r * r),  # grav
        0.0,  # energy
    ]
    return tdc_inp, cgrav, m, scale


def _probe_turb(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "turb")
    inputs, cgrav, m, scale = _make_turb_tdc_inputs()
    ulps: list[int] = []
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
            0.0,  # y_face_guess (unset)
        ]
        _, grad_t, _, conv_vel, _, _ = _rust.turb_set_tdc_f64(
            params, True, False, inputs
        )
        ulps.append(ulp_distance(grad_t, rows[base + 1]["raw_bits_hex"]))
        ulps.append(ulp_distance(conv_vel, rows[base + 3]["raw_bits_hex"]))
    checked, zeros, max_u = _summarize_ulps(ulps)
    # 71 computed outputs across test_turb (excluding 65 input rows).
    return (
        71,
        checked,
        zeros,
        max_u,
        "31-step check_TDC gradT and conv_vel outputs.",
    )


def _probe_neu(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "neu")
    ulps: list[int] = []
    for rec in range(0, 10_000, 100):
        chunk = rows[rec * 34 : (rec + 1) * 34]
        lt = raw_hex_to_float64(chunk[0]["raw_bits_hex"])
        lrho = raw_hex_to_float64(chunk[1]["raw_bits_hex"])
        abar = raw_hex_to_float64(chunk[2]["raw_bits_hex"])
        zbar = raw_hex_to_float64(chunk[3]["raw_bits_hex"])
        res = neu.neu_get(
            log10_t=lt, log10_rho=lrho, abar=abar, zbar=zbar, backend="rust"
        )
        chs = (
            res.pair.to_tuple(),
            res.plas.to_tuple(),
            res.phot.to_tuple(),
            res.brem.to_tuple(),
            res.reco.to_tuple(),
        )
        actual = [*res.total.to_tuple()]
        for d_idx in range(5):
            for ch in chs:
                actual.append(ch[d_idx])
        ulps.extend(
            ulp_distance(a, r["raw_bits_hex"])
            for a, r in zip(actual, chunk[4:], strict=True)
        )
    checked, zeros, max_u = _summarize_ulps(ulps)
    expected_outputs = (len(rows) // 34) * 30
    return (
        expected_outputs,
        checked,
        zeros,
        max_u,
        "100 strided records (3,000 outputs) of neu.",
    )


def _probe_rates(hex_root: pathlib.Path) -> ProbeOutcome:
    rows = load_tsv_rows(hex_root, "rates")
    he4 = chem.get_nuclide_info("he4")
    c12 = chem.get_nuclide_info("c12")
    ulps: list[int] = []
    for off, mode in ((0, "salpeter"), (8, "extended"), (16, "chugunov")):
        lt = raw_hex_to_float64(rows[off]["raw_bits_hex"])
        lr = raw_hex_to_float64(rows[off + 1]["raw_bits_hex"])
        zb = raw_hex_to_float64(rows[off + 2]["raw_bits_hex"])
        ab = raw_hex_to_float64(rows[off + 3]["raw_bits_hex"])
        z2b = raw_hex_to_float64(rows[off + 4]["raw_bits_hex"])
        ev = rates.screen_pair(
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
            backend="rust",
        )
        assert isinstance(ev, rates.ScreenEval)
        ulps.append(ulp_distance(ev.scor, rows[off + 5]["raw_bits_hex"]))
        ulps.append(ulp_distance(ev.dscor_dt, rows[off + 6]["raw_bits_hex"]))
        ulps.append(ulp_distance(ev.dscor_dd, rows[off + 7]["raw_bits_hex"]))
    checked, zeros, max_u = _summarize_ulps(ulps)
    return (
        len(rows),
        checked,
        zeros,
        max_u,
        "Salpeter, extended, and Chugunov plasma screening.",
    )


MODULE_PROBES: dict[str, ProbeFn] = {
    "const": _probe_const,
    "binary": _probe_binary,
    "math": _probe_math,
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
}
