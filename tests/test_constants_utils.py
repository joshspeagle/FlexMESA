# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and golden-output parity tests for `flexmesa.constants` and `utils`.

This test module verifies:
- All 23 derived physical and astronomical CGS constants against upstream MESA
  `const/test/test_output` to relative tolerance `rtol <= 1e-15`.
- `MixingType` integer enum round-trips across codes `-1` through `10`.
- `IntegerDict`, `IntegerIdict`, and the `.net` reaction-network stream
  tokenizer against `utils/test/test_output` and `utils/test/token.txt`.
- IEEE-754 finiteness guards (`is_bad`, `is_bad_array`, `replace_bad_nums`) and
  `NaN`-free gradient propagation under `jax.grad`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants, utils
from flexmesa.testing import golden

_EXPECTED_CONST_VALUES: tuple[float, ...] = (
    3.1415926535897931,
    2.3025850929940455,
    5.6703744191844259e-5,
    2.2681497676737704e-4,
    7.5657332502799993e-15,
    1.8914333125699998e-15,
    7.5657332502799993e-15,
    3.15576e7,
    1.9884098706980504e33,
    6.957e10,
    3.8280000000000003e33,
    9.4607304725808e17,
    5.9721678677913790e27,
    1.495978707e13,
    1.6605390671738466e-24,
    1.6749274980400001e-24,
    1.67262192369e-24,
    9.1093837015e-28,
    6.6260701499999999e-27,
    4.8032047125702634e-10,
    6.0221407599999999e23,
    2.99792458e10,
    1.380649e-16,
)


def _squared_norm_after_filtering(arr: jax.Array) -> jax.Array:
    """Compute sum of squares after replacing non-finite entries with zero."""
    safe = utils.replace_bad_nums(arr, replacement=0.0)
    return jnp.sum(safe * safe)


def test_constants_golden_parity_matches_upstream_mesa() -> None:
    """Verify all 23 derived constants against MESA `const/test/test_output`."""
    actual = np.array(
        [
            constants.PI,
            constants.LN10,
            constants.BOLTZ_SIGMA,
            constants.BOLTZ_SIGMA * 4.0,
            constants.BOLTZ_SIGMA * 4.0 / constants.CLIGHT,
            constants.BOLTZ_SIGMA / constants.CLIGHT,
            constants.CRAD,
            constants.SECYER,
            constants.MSUN,
            constants.RSUN,
            constants.LSUN,
            constants.LY,
            constants.M_EARTH,
            constants.AU,
            constants.AMU,
            constants.MN,
            constants.MP,
            constants.ME,
            constants.PLANCK_H,
            constants.QE,
            constants.AVO,
            constants.CLIGHT,
            constants.KERG,
        ],
        dtype=np.float64,
    )
    expected = np.array(_EXPECTED_CONST_VALUES, dtype=np.float64)
    golden.assert_values_match(actual, expected, rtol=1e-15, atol=1e-45)


def test_mixing_type_enum_codes_match_upstream_mesa() -> None:
    """Verify `MixingType` integer codes from `-1` through `10`."""
    for code in range(-1, 11):
        member = constants.MixingType.from_code(code)
        assert member is not None
        assert member.code() == code
    assert constants.MixingType.from_code(-2) is None
    assert constants.MixingType.from_code(11) is None
    assert len(constants.MixingType) == 12
    assert constants.NUMBER_OF_MIXING_TYPES == 11


def test_integer_dict_matches_upstream_mesa() -> None:
    """Verify `IntegerDict` against `utils/test/test_output`."""
    d = utils.IntegerDict()
    assert d.is_empty()
    keys = (
        "he4",
        "c12",
        "n14",
        "o16",
        "ne20",
        "mg24",
        "si28",
        "s32",
        "ar36",
        "ca40",
        "ti44",
        "cr48",
        "fe52",
        "fe54",
        "fe56",
        "ni56",
        "neut",
        "h1",
        "prot",
    )
    for idx, key in enumerate(keys, start=1):
        assert not d.define_and_report_duplicates(key, idx)
    assert d.define_and_report_duplicates("o16", 6)
    assert d.define_and_report_duplicates("ne20", 7)
    d.create_hash()
    assert d.define_and_report_duplicates("o16", 4)
    assert d.define_and_report_duplicates("ne20", 5)
    for idx, key in enumerate(keys, start=1):
        assert d.lookup(key) == idx
    assert d.lookup("missing") is None
    assert len(d) == len(keys)


def test_integer_idict_matches_upstream_mesa() -> None:
    """Verify `IntegerIdict` against `utils/test/test_output`."""
    idict = utils.IntegerIdict()
    assert idict.is_empty()
    k1s = (
        4, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 56, 1, 1, 1,
    )
    k2s = (
        2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 26, 26, 28, 0, 1, 2,
    )
    for idx, (k1, k2) in enumerate(zip(k1s, k2s, strict=True), start=1):
        assert not idict.define_and_report_duplicates(k1, k2, idx)
    assert idict.define_and_report_duplicates(16, 8, 6)
    assert idict.define_and_report_duplicates(20, 10, 7)
    idict.create_hash()
    assert idict.define_and_report_duplicates(16, 8, 4)
    assert idict.define_and_report_duplicates(20, 10, 5)
    for idx, (k1, k2) in enumerate(zip(k1s, k2s, strict=True), start=1):
        assert idict.lookup(k1, k2) == idx
    assert idict.lookup(99, 99) is None


def test_tokenize_matches_upstream_mesa_token_txt() -> None:
    """Verify `.net` stream tokenizer against `utils/test/token.txt`."""
    sample = (
        "!test input for token_read\n\n"
        "add_isos(\n"
        "   ih1, ihe4, & ! ignore & and comment\n"
        "   ic12\n"
        "   )\n\n"
        "'string1'  \"string2\"\n\n"
        "add_reactions(rc12pg, rc12ap)\n"
    )
    tokens = utils.tokenize(sample)
    expected = [
        utils.Token(utils.TokenKind.NAME, "add_isos"),
        utils.Token(utils.TokenKind.LEFT_PAREN, "("),
        utils.Token(utils.TokenKind.NAME, "ih1"),
        utils.Token(utils.TokenKind.COMMA, ","),
        utils.Token(utils.TokenKind.NAME, "ihe4"),
        utils.Token(utils.TokenKind.COMMA, ","),
        utils.Token(utils.TokenKind.NAME, "ic12"),
        utils.Token(utils.TokenKind.RIGHT_PAREN, ")"),
        utils.Token(utils.TokenKind.STRING, "string1"),
        utils.Token(utils.TokenKind.STRING, "string2"),
        utils.Token(utils.TokenKind.NAME, "add_reactions"),
        utils.Token(utils.TokenKind.LEFT_PAREN, "("),
        utils.Token(utils.TokenKind.NAME, "rc12pg"),
        utils.Token(utils.TokenKind.COMMA, ","),
        utils.Token(utils.TokenKind.NAME, "rc12ap"),
        utils.Token(utils.TokenKind.RIGHT_PAREN, ")"),
        utils.Token(utils.TokenKind.EOF, ""),
    ]
    assert tokens == expected


def test_replace_bad_nums_avoids_nan_gradient_contamination() -> None:
    """Verify `replace_bad_nums` filters `NaN`/`Inf` under `jax.grad`."""
    assert not utils.is_bad(0.0)
    assert utils.is_bad(float("nan"))
    assert utils.is_bad(float("inf"))

    xs = jnp.array([2.0, jnp.nan, -3.0, jnp.inf], dtype=jnp.float64)
    assert utils.is_bad_array(xs)
    cleaned = utils.replace_bad_nums(xs, replacement=0.0)
    assert not utils.is_bad_array(cleaned)
    golden.assert_values_match(cleaned, np.array([2.0, 0.0, -3.0, 0.0]))

    grads = jax.grad(_squared_norm_after_filtering)(xs)
    assert not utils.is_bad_array(grads)
    golden.assert_values_match(grads, np.array([4.0, 0.0, -6.0, 0.0]))
    assert utils.remove_underbars("star_mass_msun  ") == "star mass msun"
    assert utils.split_line("a   b c d", 3) == ["a", "b", "c"]
    assert not utils.split_line("a b", 0)
