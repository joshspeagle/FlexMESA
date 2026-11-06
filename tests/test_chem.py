# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Verification suite for `flexmesa.chem` (`Wave 2a`).

Validates the Three-Tier Hybrid `chem` implementation against:
1. 100% of the golden output in upstream MESA `chem/test/test_output` (`o16`
   properties, `AG89 zsol/yesol`, `CNO/Z` across solar scales, `Lodders 2003`
   atom percentages, `composition_info` on a 50/50 `c12`-`fe56` mixture, and
   `reaction_Qtotal` on $2\,{}^3\mathrm{He} \to {}^4\mathrm{He} +
   2\,{}^1\mathrm{H}$).
2. Two-tier parity (`<= 1e-14` relative error) between Safe Rust (`PyO3` +
   `rayon`) and `@jax.jit` (`get_composition_info_jax`) across 20 deterministic
   random simplex compositions.
3. Automatic differentiation (`jax.test_util.check_grads` and `jax.jacrev`) on
   `get_composition_info_jax`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import test_util as jtu

from flexmesa import _rust, chem


def test_o16_and_element_lookup_match_mesa_golden_output() -> None:
    r"""Verify `o16` nuclear properties and `he` element lookup.

    Checks `get_nuclide_index('o16') == 66`, atomic weight $W(o16)$, nuclear
    binding energy $B(o16)$, and periodic-table lookups (`chem_get_element_id`,
    `element_name`, `element_long_name`, `element_main_iso_name`) against
    upstream MESA `chem/test/test_output`.
    """
    io16 = chem.get_nuclide_index("o16")
    assert io16 == 66
    info = chem.get_nuclide_info(io16)
    assert info.name == "o16"
    assert info.z == 8
    assert info.a == 16
    assert np.isclose(info.w, 1.5994914620075349e1, rtol=1e-15, atol=0.0)
    assert np.isclose(
        info.binding_energy, 1.2761930221000000e2, rtol=1e-15, atol=0.0
    )

    assert chem.chem_get_element_id("he") == 2
    assert chem.element_name(2) == "he"
    assert chem.element_long_name(2) == "helium"
    assert chem.element_main_iso_name(26) == "fe56"


def test_ag89_and_solar_cno_fractions_match_mesa_golden_output() -> None:
    r"""Verify `zsol`, `yesol`, and `CNO/Z` across solar abundance scales.

    Checks Anders & Grevesse (1989) bulk solar metallicity `zsol`, solar
    electron fraction `yesol`, CNO-to-metal mass ratios across `AG89`, `GN93`,
    `GS98`, `L03`, and `AGS05`, normalization $\sum_{Z=3}^{92} z_Z = 1$ across
    all 10 solar scales, and vanishing `[M/H]` at exact solar $(X_\odot,
    Z_\odot)$.
    """
    zsol = chem.ag89_zsol()
    yesol = chem.ag89_yesol()
    assert np.isclose(zsol, 1.8987109631891819e-2, rtol=1e-15, atol=0.0)
    assert np.isclose(yesol, 8.5280190902027331e-1, rtol=1e-15, atol=0.0)

    cno_ag89 = (
        chem.chem_xsol("c12") + chem.chem_xsol("n14") + chem.chem_xsol("o16")
    ) / zsol
    assert np.isclose(cno_ag89, 7.2307477368434370e-1, rtol=1e-15, atol=0.0)

    golden_cno = {
        "GN93": 7.1045735428893797e-1,
        "GS98": 6.9188768734346451e-1,
        "L03": 6.6302570462115973e-1,
        "AGS05": 6.6770954116998993e-1,
    }
    for scale, expected in golden_cno.items():
        val = chem.solar_cno_fraction(scale)
        assert np.isclose(val, expected, rtol=1e-15, atol=0.0)

    for scale in ("AG89", "L09", "AGSS09", "A09_Prz", "MB22", "AAG21"):
        zfrac = chem.solar_element_zfrac(scale)
        assert zfrac.shape == (112,)
        assert np.isclose(np.sum(zfrac[2:92]), 1.0, rtol=1e-14, atol=0.0)

    # At exact solar (X_sol, Z_sol), [M/H] must vanish
    assert np.isclose(
        chem.chem_m_div_h(1.0 - 0.2485 - 0.0134, 0.0134, "AGSS09"),
        0.0,
        atol=1e-15,
    )


def test_lodders03_table_matches_mesa_golden_output() -> None:
    r"""Verify the 286 solar isotopes and Lodders (2003) atom percentages.

    Checks the full 286-entry `namsol` ordering (`h1` through `u238`) and
    tabulated isotopic atom percentages from `lodders03.data` against upstream
    MESA `chem/test/test_output`.
    """
    names = chem.ag89_namsol()
    assert len(names) == 286
    assert names[0] == "h1"
    assert names[-1] == "u238"

    golden_samples = {
        "h1": 99.998060,
        "h2": 0.001940,
        "he3": 0.016597,
        "he4": 99.983403,
        "c12": 98.892200,
        "n14": 99.633700,
        "o16": 99.762800,
        "ne20": 92.943100,
        "mg24": 78.992000,
        "si28": 92.229680,
        "fe56": 91.754000,
        "ni58": 68.076900,
        "ge74": 35.943000,
        "pb208": 58.692900,
        "th232": 100.000000,
        "u235": 0.720000,
        "u238": 99.274500,
    }
    for iso, expected in golden_samples.items():
        pct = chem.lodders03_element_atom_percent(iso)
        assert np.isclose(pct, expected, rtol=0.0, atol=1e-6)

    for iso in names:
        sol = chem.chem_get_solar(iso)
        assert sol is not None
        assert sol[2] > 0.0
        assert chem.lodders03_element_atom_percent(iso) > 0.0


def test_composition_info_c12_fe56_matches_mesa_golden_output() -> None:
    r"""Verify `do_test_composition_info` on a 50/50 `c12`-`fe56` mixture.

    Checks exact rational identities for $\bar{A} = 336/17$, $\bar{Z} = 162/17$,
    $z_2\bar{} = 2532/17$, $Y_e = 27/56$, `mass_correction`, and analytical
    partial derivatives (`dabar_dx`, `dzbar_dx`, `dmc_dx`) across both Safe
    Rust and `@jax.jit` backends.
    """
    ids = ["c12", "fe56"]
    x = np.array([0.5, 0.5], dtype=np.float64)

    res_rust = chem.composition_info(ids, x, backend="rust")
    res_jax = chem.composition_info(ids, x, backend="jax")

    for res in (res_rust, res_jax):
        assert np.isclose(res.abar, 336.0 / 17.0, rtol=1e-15, atol=0.0)
        assert np.isclose(res.zbar, 162.0 / 17.0, rtol=1e-15, atol=0.0)
        assert np.isclose(res.z2bar, 2532.0 / 17.0, rtol=1e-15, atol=0.0)
        assert np.isclose(res.ye, 27.0 / 56.0, rtol=1e-15, atol=0.0)
        assert np.isclose(res.mass_correction, 0.999419, rtol=0.0, atol=1e-6)
        assert np.isclose(res.sumx, 1.0, rtol=1e-15, atol=0.0)

    np.testing.assert_allclose(
        res_rust.dabar_dx, res_jax.dabar_dx, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_allclose(
        res_rust.dzbar_dx, res_jax.dzbar_dx, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_allclose(
        res_rust.dmc_dx, res_jax.dmc_dx, rtol=1e-15, atol=1e-15
    )

    basic = chem.basic_composition_info(ids, x, backend="rust")
    assert basic.dabar_dx.size == 0


def test_reaction_qtotal_two_he3_to_he4_plus_two_h1() -> None:
    r"""Verify `do_test_Qtotal` on `2 he3 -> he4 + 2 h1`.

    Confirms `reaction_qtotal` reproduces `1.2859572850000005e+01` MeV to full
    IEEE-754 double precision.
    """
    qtotal = chem.reaction_qtotal(2, 3, ["he3", "he3", "he4", "h1", "h1"])
    assert np.isclose(qtotal, 1.2859572850000005e1, rtol=1e-15, atol=0.0)


def test_composition_info_zones_rust_vs_jax_across_random_seeds() -> None:
    r"""Verify multi-zone Rust (`rayon`) vs. `@jax.jit` parity across seeds.

    Evaluates 128 zones of 9-isotope Dirichlet mass-fraction mixtures across 20
    deterministic `np.random.default_rng(seed)` seeds and checks conservation
    identities ($X + Y + Z = 1$, $Y_e = \bar{Z}/\bar{A}$) to `1e-14`.
    """
    species = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24", "fe56"]
    n_zones = 128
    for seed in range(20):
        rng = np.random.default_rng(seed)
        raw = rng.dirichlet(np.ones(len(species)), size=n_zones)
        x_zones = np.ascontiguousarray(raw, dtype=np.float64)

        out_rust_par = chem.get_composition_info_zones(
            species, x_zones, parallel=True, backend="rust"
        )
        out_rust_seq = chem.get_composition_info_zones(
            species, x_zones, parallel=False, backend="rust"
        )
        out_jax = np.asarray(
            chem.get_composition_info_zones(
                species, jnp.asarray(x_zones), backend="jax"
            ),
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            out_rust_par, out_rust_seq, rtol=1e-15, atol=1e-15
        )
        np.testing.assert_allclose(
            out_rust_par, out_jax, rtol=1e-14, atol=1e-14
        )
        # Conservation identities: X_H + X_He + Z == 1 and Y_e == Zbar / Abar
        np.testing.assert_allclose(
            out_rust_par[:, 0] + out_rust_par[:, 1] + out_rust_par[:, 2],
            1.0,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            out_rust_par[:, 7],
            out_rust_par[:, 4] / out_rust_par[:, 3],
            rtol=1e-15,
            atol=1e-15,
        )


def test_composition_info_jax_autodiff_and_analytical_partials() -> None:
    r"""Verify `get_composition_info_jax` differentiability via `check_grads`.

    Confirms 2nd-order finite-difference gradient agreement on
    `get_composition_info_jax` and verifies that MESA's fixed-`sumx` analytical
    partials (`dabar_dx`, `dzbar_dx`) match exact forward/reverse-mode AD.
    """
    species = ["h1", "he4", "c12", "o16", "fe56"]
    x0 = jnp.array([0.50, 0.30, 0.05, 0.10, 0.05], dtype=jnp.float64)

    def scalar_loss(x_vec: jax.Array) -> jax.Array:
        moments, _, _, _ = chem.get_composition_info_jax(species, x_vec)
        return jnp.sum(moments)

    jtu.check_grads(scalar_loss, (x0,), order=2, atol=1e-6, rtol=1e-6)

    # Compare MESA's analytical partial d(abar)/dX_i and d(zbar)/dX_i with AD
    _, da_mesa, dz_mesa, _ = chem.get_composition_info_jax(species, x0)
    _, z_vals, a_vals, _, _ = _rust.py_chem_species_props(
        [chem.get_nuclide_index(s) for s in species]
    )
    a_arr = jnp.asarray(a_vals, dtype=jnp.float64)
    z_arr = jnp.asarray(z_vals, dtype=jnp.float64)

    grad_abar_exact = jax.grad(lambda x: jnp.sum(x) / jnp.sum(x / a_arr))(x0)
    grad_zbar_fixed_sumx = jax.grad(
        lambda x: jnp.sum((x / a_arr) * z_arr) / jnp.sum(x / a_arr)
    )(x0)
    np.testing.assert_allclose(
        np.asarray(da_mesa),
        np.asarray(grad_abar_exact),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(dz_mesa),
        np.asarray(grad_zbar_fixed_sumx),
        rtol=1e-14,
        atol=1e-14,
    )


def test_category_partition_and_decay_utilities() -> None:
    r"""Verify reaction category lookups, partition index, and beta decay.

    Tests `category_id`, `category_name`, `get_partition_fcn_indx`,
    `generate_nuclide_names`, `generate_long_nuclide_names`, and
    `get_stable_mass_frac` ($\mathrm{Ni56} \to \mathrm{Fe56}$ isobaric decay).
    """
    assert chem.category_id("pp") == 1
    assert chem.category_id("cno") == 2
    assert chem.category_id("tri_alpha") == 3
    assert chem.category_name(1) == "pp"
    assert chem.get_partition_fcn_indx(0.05) == 0
    assert chem.get_partition_fcn_indx(0.12) == 1

    names = chem.generate_nuclide_names([1, 26], [1, 56])
    long_names = chem.generate_long_nuclide_names([1, 26], [1, 56])
    assert names == ["h1", "fe56"]
    assert long_names == ["hydrogen-1", "iron-56"]

    # Unstable Ni56 (Z=28, A=56) decays completely to stable Fe56 (Z=26, A=56)
    stable = chem.get_stable_mass_frac([28], [56], [1.0])
    fe56_idx = chem.ag89_namsol().index("fe56")
    assert np.isclose(stable[fe56_idx], 1.0, rtol=1e-15, atol=0.0)

    with pytest.raises(ValueError, match="Unknown"):
        chem.get_nuclide_info("not_an_isotope")
    with pytest.raises(ValueError, match="shape"):
        chem.get_composition_info(["c12", "fe56"], [0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="shape"):
        chem.get_composition_info_zones(
            ["c12", "fe56"], np.ones((4, 3), dtype=np.float64)
        )
