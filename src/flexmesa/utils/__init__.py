# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core numerical guards, dictionaries, and stream tokenizers for FlexMESA.

This module ports upstream MESA's `utils` package in Python and JAX (`float64`),
providing IEEE-754 finiteness checks that avoid `0.0 * NaN` propagation,
string- and integer-pair-keyed dictionaries with duplicate reporting, and the
stateful stream tokenizer used to parse nuclear reaction network (`.net`) files.
"""

from __future__ import annotations

import dataclasses
import enum
import math

import jax
import jax.numpy as jnp
import numpy as np


def is_bad(value: float) -> bool:
    r"""Return `True` when `value` is `NaN` or infinite.

    Args:
        value: Scalar floating-point number (`float64`).

    Returns:
        `True` if `value` is `NaN` or $\pm\infty$, `False` otherwise.
    """
    return not math.isfinite(value)


def is_bad_array(values: np.ndarray | jax.Array) -> bool:
    """Return `True` if any element in `values` is `NaN` or infinite.

    Args:
        values: Array of arbitrary shape `(*batch,)` in `float64`.

    Returns:
        `True` if at least one element of `values` is non-finite.
    """
    return bool(np.any(~np.isfinite(np.asarray(values, dtype=np.float64))))


def replace_bad_nums(
    values: np.ndarray | jax.Array,
    replacement: float = -1.0e99,
) -> jax.Array:
    r"""Replace non-finite (`NaN` or $\pm\infty$) elements with `replacement`.

    Applies `jnp.where` before any arithmetic so that masked `NaN` entries never
    propagate through `0.0 * NaN` in values or automatic-differentiation
    gradients.

    Args:
        values: Input array of shape `(*batch,)` in `float64`.
        replacement: Finite scalar value substituted for non-finite entries.

    Returns:
        Array of shape `(*batch,)` with `dtype=float64` and all non-finite
        entries replaced by `replacement`.
    """
    arr = jnp.asarray(values, dtype=jnp.float64)
    repl = jnp.asarray(replacement, dtype=jnp.float64)
    return jnp.where(jnp.isfinite(arr), arr, repl)


def remove_underbars(text: str) -> str:
    """Replace underscores with spaces after trimming trailing whitespace.

    Args:
        text: Input string from a MESA control or column header.

    Returns:
        String with trailing spaces removed and `'_'` replaced by `' '`.
    """
    return text.rstrip().replace("_", " ")


def split_line(line: str, num: int) -> list[str]:
    """Split `line` on whitespace into at most `num` tokens.

    Args:
        line: Whitespace-delimited input string.
        num: Maximum number of tokens to extract.

    Returns:
        List of at most `num` whitespace-delimited substrings.
    """
    return line.split()[:num]


class IntegerDict:
    """String-keyed integer dictionary matching upstream MESA `integer_dict`.

    Keys are stripped of trailing whitespace on insertion and lookup to match
    Fortran `len_trim` semantics.
    """

    def __init__(self) -> None:
        """Initialize an empty `IntegerDict`."""
        self._map: dict[str, int] = {}

    def define_and_report_duplicates(self, key: str, value: int) -> bool:
        """Define `key -> value` and return `True` if `key` was already defined.

        Args:
            key: String key (trailing whitespace is stripped).
            value: Integer value to associate with `key`.

        Returns:
            `True` if `key` already existed prior to this definition.
        """
        normalized = key.rstrip()
        duplicate = normalized in self._map
        self._map[normalized] = int(value)
        return duplicate

    def define(self, key: str, value: int) -> None:
        """Define `key -> value`, overwriting any existing entry.

        Args:
            key: String key (trailing whitespace is stripped).
            value: Integer value to associate with `key`.
        """
        self.define_and_report_duplicates(key, value)

    def create_hash(self) -> None:
        """No-op hook for upstream MESA `integer_dict_create_hash`."""

    def lookup(self, key: str) -> int | None:
        """Look up the integer value associated with `key`.

        Args:
            key: String key to query.

        Returns:
            Associated integer value, or `None` if `key` is not present.
        """
        return self._map.get(key.rstrip())

    def __len__(self) -> int:
        """Return the number of entries in the dictionary."""
        return len(self._map)

    def entries(self) -> list[tuple[str, int]]:
        """Return all `(key, value)` pairs sorted lexicographically by key.

        Returns:
            Sorted list of `(key, value)` pairs.
        """
        return sorted(self._map.items())


class IntegerIdict:
    """Integer-pair `(k1, k2)` dictionary matching MESA `integer_idict`."""

    def __init__(self) -> None:
        """Initialize an empty `IntegerIdict`."""
        self._map: dict[tuple[int, int], int] = {}

    def define_and_report_duplicates(
        self,
        key1: int,
        key2: int,
        value: int,
    ) -> bool:
        """Define `(key1, key2) -> value` and report whether the key existed.

        Args:
            key1: First integer key component.
            key2: Second integer key component.
            value: Integer value to associate with `(key1, key2)`.

        Returns:
            `True` if `(key1, key2)` was already defined prior to this call.
        """
        pair = (int(key1), int(key2))
        duplicate = pair in self._map
        self._map[pair] = int(value)
        return duplicate

    def define(self, key1: int, key2: int, value: int) -> None:
        """Define `(key1, key2) -> value`, overwriting any existing entry.

        Args:
            key1: First integer key component.
            key2: Second integer key component.
            value: Integer value to associate with `(key1, key2)`.
        """
        self.define_and_report_duplicates(key1, key2, value)

    def create_hash(self) -> None:
        """No-op compatibility hook for MESA `integer_idict_create_hash`."""

    def lookup(self, key1: int, key2: int) -> int | None:
        """Look up the integer value associated with `(key1, key2)`.

        Args:
            key1: First integer key component.
            key2: Second integer key component.

        Returns:
            Associated integer value, or `None` if `(key1, key2)` is absent.
        """
        return self._map.get((int(key1), int(key2)))

    def __len__(self) -> int:
        """Return the number of entries in the dictionary."""
        return len(self._map)

    def entries(self) -> list[tuple[int, int, int]]:
        """Return all `(key1, key2, value)` tuples sorted by `(key1, key2)`.

        Returns:
            Sorted list of `(key1, key2, value)` entries.
        """
        return [(k1, k2, val) for (k1, k2), val in sorted(self._map.items())]


class TokenKind(enum.Enum):
    """Lexical token classification matching upstream MESA `utils_def`."""

    NAME = "name"
    STRING = "string"
    LEFT_PAREN = "left_paren"
    RIGHT_PAREN = "right_paren"
    COMMA = "comma"
    EOF = "eof"


@dataclasses.dataclass(frozen=True)
class Token:
    """Lexical token emitted by `Tokenizer` and `tokenize`.

    Attributes:
        kind: Token category (`TokenKind`).
        value: Text payload for `NAME` and `STRING` tokens, or symbol text.
    """

    kind: TokenKind
    value: str = ""


def _scan_quoted_or_name(line: str, pos: int, ch: str) -> tuple[Token, int]:
    """Scan a quoted string or bare identifier starting at `ch` in `line`."""
    if ch in ('"', "'"):
        start = pos
        while pos < len(line) and line[pos] != ch:
            pos += 1
        content = line[start:pos]
        if pos < len(line):
            pos += 1
        return Token(TokenKind.STRING, content), pos

    start = pos - 1
    while pos < len(line) and line[pos] not in (" ", "\t", "(", ")", ","):
        pos += 1
    return Token(TokenKind.NAME, line[start:pos]), pos


class Tokenizer:
    """Stateful stream tokenizer matching upstream MESA `utils_lib::token`."""

    def __init__(self, text: str) -> None:
        """Initialize the tokenizer over `text`.

        Args:
            text: Full multi-line input text to tokenize.
        """
        self._lines: list[str] = [line.rstrip() for line in text.splitlines()]
        self._line_idx: int = 0
        self._current: str = ""
        self._pos: int = 0

    def next_token(self) -> Token:
        """Advance and return the next `Token`, or `Token(TokenKind.EOF)`.

        Returns:
            Next lexical `Token` in the stream.
        """
        while True:
            while self._pos >= len(self._current):
                if self._line_idx >= len(self._lines):
                    return Token(TokenKind.EOF, "")
                self._current = self._lines[self._line_idx]
                self._line_idx += 1
                self._pos = 0

            while self._pos < len(self._current):
                ch = self._current[self._pos]
                self._pos += 1
                if ch in ("\t", " ", "&"):
                    continue
                if ch == "!":
                    self._pos = len(self._current)
                    break
                if ch == "(":
                    return Token(TokenKind.LEFT_PAREN, "(")
                if ch == ")":
                    return Token(TokenKind.RIGHT_PAREN, ")")
                if ch == ",":
                    return Token(TokenKind.COMMA, ",")
                tok, self._pos = _scan_quoted_or_name(
                    self._current, self._pos, ch
                )
                return tok


def tokenize(text: str) -> list[Token]:
    """Tokenize `text` into a list of `Token` objects ending with `EOF`.

    Args:
        text: Multi-line input string in MESA `.net` or list syntax.

    Returns:
        Sequence of `Token` objects terminated by `Token(TokenKind.EOF, "")`.
    """
    lexer = Tokenizer(text)
    tokens: list[Token] = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if tok.kind is TokenKind.EOF:
            break
    return tokens
