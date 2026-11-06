# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core numerical guards, dictionaries, and stream tokenizers for FlexMESA.

This module ports upstream MESA's ``utils`` package, routing scalar/NumPy
finiteness checks, string normalizers, and the stateful ``.net`` stream
tokenizer through the compiled Safe Rust ``flexmesa._rust`` extension while
providing JAX ``jnp.where`` finiteness guards for traced arrays.
"""

from __future__ import annotations

import dataclasses
import enum

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust


def is_bad(value: float) -> bool:
    r"""Return ``True`` when ``value`` is ``NaN`` or infinite.

    Delegates to ``flexmesa_utils::is_bad`` via ``flexmesa._rust``.

    Args:
        value: Scalar floating-point number (``float64``).

    Returns:
        ``True`` if ``value`` is ``NaN`` or $\pm\infty$, ``False`` otherwise.

    Examples:
        >>> from flexmesa import utils
        >>> utils.is_bad(1.0)
        False
        >>> utils.is_bad(float("nan"))
        True
    """
    return _rust.is_bad(float(value))


def is_bad_array(values: np.ndarray | jax.Array) -> bool:
    """Return ``True`` if any element in ``values`` is ``NaN`` or infinite.

    Args:
        values: Array of arbitrary shape ``(*batch,)`` in ``float64``.

    Returns:
        ``True`` if at least one element of ``values`` is non-finite.
    """
    if _dispatch.is_jax_input(values):
        return not bool(jnp.all(jnp.isfinite(values)))
    return _rust.is_bad_array(_dispatch.as_c_f64(values))


def replace_bad_nums(
    values: np.ndarray | jax.Array,
    replacement: float = -1.0e99,
) -> jax.Array:
    r"""Replace non-finite (``NaN`` or $\pm\infty$) values with ``replacement``.

    Applies ``jnp.where`` before any arithmetic so that masked ``NaN`` entries
    never propagate through ``0.0 * NaN`` in values or automatic-differentiation
    gradients.

    Args:
        values: Input array of shape ``(*batch,)`` in ``float64``.
        replacement: Finite scalar value substituted for non-finite entries.

    Returns:
        Array of shape ``(*batch,)`` with ``dtype=float64`` and all non-finite
        entries replaced by ``replacement``.
    """
    arr = jnp.asarray(values, dtype=jnp.float64)
    return jnp.where(jnp.isfinite(arr), arr, float(replacement))


def remove_underbars(text: str) -> str:
    """Replace underscores with spaces after trimming trailing whitespace.

    Delegates to ``flexmesa_utils::remove_underbars`` via ``flexmesa._rust``.

    Args:
        text: Input string from a MESA control or column header.

    Returns:
        String with trailing spaces removed and ``'_'`` replaced by ``' '``.
    """
    return _rust.remove_underbars(text)


def split_line(line: str, num: int) -> list[str]:
    """Split ``line`` on whitespace into at most ``num`` tokens.

    Delegates to ``flexmesa_utils::split_line`` via ``flexmesa._rust``.

    Args:
        line: Whitespace-delimited input string.
        num: Maximum number of tokens to extract.

    Returns:
        List of at most ``num`` whitespace-delimited substrings.
    """
    if num <= 0:
        return []
    return _rust.split_line(line, int(num))


class IntegerDict:
    """String-keyed integer dictionary matching upstream MESA ``integer_dict``.

    Keys are stripped of trailing whitespace on insertion and lookup to match
    Fortran ``len_trim`` semantics.
    """

    def __init__(self) -> None:
        """Initialize an empty ``IntegerDict``."""
        self._map: dict[str, int] = {}

    def define_and_report_duplicates(self, key: str, value: int) -> bool:
        """Define ``key -> value`` and return ``True`` if ``key`` existed.

        Args:
            key: String key (trailing whitespace is stripped).
            value: Integer value to associate with ``key``.

        Returns:
            ``True`` if ``key`` already existed prior to this definition.
        """
        normalized = key.rstrip()
        duplicate = normalized in self._map
        self._map[normalized] = int(value)
        return duplicate

    def define(self, key: str, value: int) -> None:
        """Define ``key -> value``, overwriting any existing entry.

        Args:
            key: String key (trailing whitespace is stripped).
            value: Integer value to associate with ``key``.
        """
        self.define_and_report_duplicates(key, value)

    def create_hash(self) -> None:
        """No-op hook for upstream MESA ``integer_dict_create_hash``."""

    def lookup(self, key: str) -> int | None:
        """Look up the integer value associated with ``key``.

        Args:
            key: String key to query.

        Returns:
            Associated integer value, or ``None`` if ``key`` is not present.
        """
        return self._map.get(key.rstrip())

    def is_empty(self) -> bool:
        """Return ``True`` if the dictionary contains no entries."""
        return not self._map

    def __len__(self) -> int:
        """Return the number of entries in the dictionary."""
        return len(self._map)

    def entries(self) -> list[tuple[str, int]]:
        """Return all ``(key, value)`` pairs sorted lexicographically by key.

        Returns:
            Sorted list of ``(key, value)`` pairs.
        """
        return sorted(self._map.items())


class IntegerIdict:
    """Integer-pair ``(k1, k2)`` dictionary matching MESA ``integer_idict``."""

    def __init__(self) -> None:
        """Initialize an empty ``IntegerIdict``."""
        self._map: dict[tuple[int, int], int] = {}

    def define_and_report_duplicates(
        self,
        key1: int,
        key2: int,
        value: int,
    ) -> bool:
        """Define ``(key1, key2) -> value`` and report whether the key existed.

        Args:
            key1: First integer key component.
            key2: Second integer key component.
            value: Integer value to associate with ``(key1, key2)``.

        Returns:
            ``True`` if ``(key1, key2)`` was already defined prior to this call.
        """
        pair = (int(key1), int(key2))
        duplicate = pair in self._map
        self._map[pair] = int(value)
        return duplicate

    def define(self, key1: int, key2: int, value: int) -> None:
        """Define ``(key1, key2) -> value``, overwriting any existing entry.

        Args:
            key1: First integer key component.
            key2: Second integer key component.
            value: Integer value to associate with ``(key1, key2)``.
        """
        self.define_and_report_duplicates(key1, key2, value)

    def create_hash(self) -> None:
        """No-op compatibility hook for MESA ``integer_idict_create_hash``."""

    def lookup(self, key1: int, key2: int) -> int | None:
        """Look up the integer value associated with ``(key1, key2)``.

        Args:
            key1: First integer key component.
            key2: Second integer key component.

        Returns:
            Associated integer value, or ``None`` if ``(key1, key2)`` is absent.
        """
        return self._map.get((int(key1), int(key2)))

    def is_empty(self) -> bool:
        """Return ``True`` if the dictionary contains no entries."""
        return not self._map

    def __len__(self) -> int:
        """Return the number of entries in the dictionary."""
        return len(self._map)

    def entries(self) -> list[tuple[int, int, int]]:
        """Return all ``(key1, key2, value)`` tuples sorted by ``(key1, key2)``.

        Returns:
            Sorted list of ``(key1, key2, value)`` entries.
        """
        return [(k1, k2, val) for (k1, k2), val in sorted(self._map.items())]


class TokenKind(enum.StrEnum):
    """Lexical token classification matching upstream MESA ``utils_def``."""

    NAME = "name"
    STRING = "string"
    LEFT_PAREN = "left_paren"
    RIGHT_PAREN = "right_paren"
    COMMA = "comma"
    EOF = "eof"


@dataclasses.dataclass(frozen=True, slots=True)
class Token:
    """Lexical token emitted by ``Tokenizer`` and ``tokenize``.

    Attributes:
        kind: Token category (``TokenKind``).
        value: Text payload for ``NAME`` and ``STRING`` tokens, or symbol text.
    """

    kind: TokenKind
    value: str = ""


_EOF_TOKEN: Token = Token(TokenKind.EOF, "")


def tokenize(text: str) -> list[Token]:
    """Tokenize ``text`` into a list of ``Token`` objects ending with ``EOF``.

    Delegates directly to ``flexmesa_utils::tokenize`` via ``flexmesa._rust``.

    Args:
        text: Multi-line input string in MESA ``.net`` or list syntax.

    Returns:
        Sequence of ``Token`` objects terminated by
        ``Token(TokenKind.EOF, "")``.

    Examples:
        >>> from flexmesa import utils
        >>> [t.value for t in utils.tokenize("add_isos(ih1, ihe4)")]
        ['add_isos', '(', 'ih1', ',', 'ihe4', ')', '']
    """
    return [Token(TokenKind(kind), val) for kind, val in _rust.tokenize(text)]


class Tokenizer:
    """Stateful stream tokenizer matching upstream MESA ``utils_lib::token``."""

    def __init__(self, text: str) -> None:
        """Initialize the tokenizer over ``text``.

        Args:
            text: Full multi-line input text to tokenize.
        """
        self._tokens: list[Token] = tokenize(text)
        self._idx: int = 0

    def next_token(self) -> Token:
        """Advance and return the next ``Token``, or ``Token(TokenKind.EOF)``.

        Returns:
            Next lexical ``Token`` in the stream.
        """
        if self._idx >= len(self._tokens):
            return _EOF_TOKEN
        tok = self._tokens[self._idx]
        if tok.kind != TokenKind.EOF:
            self._idx += 1
        return tok


__all__ = [
    "IntegerDict",
    "IntegerIdict",
    "Token",
    "TokenKind",
    "Tokenizer",
    "is_bad",
    "is_bad_array",
    "remove_underbars",
    "replace_bad_nums",
    "split_line",
    "tokenize",
]
