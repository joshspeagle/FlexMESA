#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate IEEE-754 binary128 test vectors from exact rational arithmetic.

Each result is computed with `fractions.Fraction` (exact) and then rounded once
to binary128, or to binary64 for conversions, to nearest with ties to even.
Nothing here calls `FlexMESA` or a floating-point library, so the vectors are an
independent check of `flexmesa_math::Quad`.

Usage:
    python gen_binary128_vectors.py > binary128_vectors.tsv

The output is deterministic (fixed seed). `check_binary128_vectors_gcc.c` in
this directory replays the same file against GCC's `__float128`.
"""

from __future__ import annotations

import random
import struct
import sys
from fractions import Fraction

FRAC_BITS = 112
BIAS = 16383
EXP_MAX = 0x7FFF
SIGN = 1 << 127
INF = EXP_MAX << FRAC_BITS
NAN = "nan"

Value = tuple[str, int, Fraction]  # (class, sign, magnitude)


def decode(bits: int) -> Value:
    """Split binary128 bits into (`nan`|`inf`|`zero`|`fin`, sign, |value|)."""
    sign = bits >> 127
    biased = (bits >> FRAC_BITS) & EXP_MAX
    frac = bits & ((1 << FRAC_BITS) - 1)
    if biased == EXP_MAX:
        return ("nan" if frac else "inf", sign, Fraction(0))
    if biased == 0:
        if frac == 0:
            return ("zero", sign, Fraction(0))
        return ("fin", sign, Fraction(frac) * Fraction(2) ** (1 - BIAS - 112))
    mag = Fraction((1 << FRAC_BITS) | frac) * Fraction(2) ** (
        biased - BIAS - 112
    )
    return ("fin", sign, mag)


def floor_log2(q: Fraction) -> int:
    """Largest `e` with `2**e <= q` for `q > 0`."""
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    return e


def round_to(q: Fraction, sign: int, p: int, emin: int, emax: int) -> int:
    """Round `q > 0` to a `p`-bit binary format; return its bit pattern."""
    width = {113: 128, 53: 64}[p]
    ebits = width - p
    bias = emax
    e = max(floor_log2(q), emin)
    scaled = q / Fraction(2) ** (e - (p - 1))
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n == 1 << p:
        n >>= 1
        e += 1
    sign_bits = sign << (width - 1)
    if e > emax:
        return sign_bits | (((1 << ebits) - 1) << (p - 1))
    if n < 1 << (p - 1):
        return sign_bits | n
    return sign_bits | ((e + bias) << (p - 1)) | (n - (1 << (p - 1)))


def round128(q: Fraction, sign: int) -> int:
    return round_to(q, sign, 113, 1 - BIAS, BIAS)


def signed(v: Value) -> Fraction:
    return -v[2] if v[1] else v[2]


def op_add(a: int, b: int) -> int | str:
    va, vb = decode(a), decode(b)
    if "nan" in (va[0], vb[0]):
        return NAN
    if va[0] == "inf" and vb[0] == "inf":
        return a if va[1] == vb[1] else NAN
    if va[0] == "inf":
        return a
    if vb[0] == "inf":
        return b
    s = signed(va) + signed(vb)
    if s == 0:
        both_negative = va[1] == 1 and vb[1] == 1
        return SIGN if both_negative else 0
    return round128(abs(s), 1 if s < 0 else 0)


def op_sub(a: int, b: int) -> int | str:
    if decode(b)[0] == "nan":
        return NAN
    return op_add(a, b ^ SIGN)


def op_mul(a: int, b: int) -> int | str:
    va, vb = decode(a), decode(b)
    sign = va[1] ^ vb[1]
    if "nan" in (va[0], vb[0]):
        return NAN
    if {va[0], vb[0]} == {"inf", "zero"}:
        return NAN
    if "inf" in (va[0], vb[0]):
        return (sign << 127) | INF
    if "zero" in (va[0], vb[0]):
        return sign << 127
    return round128(va[2] * vb[2], sign)


def op_div(a: int, b: int) -> int | str:
    va, vb = decode(a), decode(b)
    sign = va[1] ^ vb[1]
    if "nan" in (va[0], vb[0]):
        return NAN
    if va[0] == vb[0] and va[0] in ("inf", "zero"):
        return NAN
    if va[0] == "inf" or vb[0] == "zero":
        return (sign << 127) | INF
    if va[0] == "zero" or vb[0] == "inf":
        return sign << 127
    return round128(va[2] / vb[2], sign)


def op_powi(a: int, n: int) -> int | str:
    """`libgcc` `__powitf2`: binary exponentiation from the low bit."""
    one = BIAS << FRAC_BITS
    m = abs(n)
    x: int | str = a
    y: int | str = a if m % 2 else one
    while True:
        m >>= 1
        if m == 0:
            break
        x = NAN if x == NAN else op_mul(x, x)  # type: ignore[arg-type]
        if m % 2:
            y = NAN if NAN in (x, y) else op_mul(y, x)  # type: ignore[arg-type]
    if y == NAN:
        return NAN
    assert isinstance(y, int)
    return op_div(one, y) if n < 0 else y


def op_to_f64(a: int) -> int | str:
    va = decode(a)
    if va[0] == "nan":
        return NAN
    if va[0] == "inf":
        return (va[1] << 63) | (0x7FF << 52)
    if va[0] == "zero":
        return va[1] << 63
    return round_to(va[2], va[1], 53, -1022, 1023)


def op_from_f64(d: int) -> int | str:
    x = struct.unpack(">d", d.to_bytes(8, "big"))[0]
    if x != x:
        return NAN
    sign = d >> 63
    if x in (float("inf"), float("-inf")):
        return (sign << 127) | INF
    if x == 0.0:
        return sign << 127
    return round128(abs(Fraction(x)), sign)


def q_from_float(x: float) -> int:
    r = op_from_f64(int.from_bytes(struct.pack(">d", x), "big"))
    assert isinstance(r, int)
    return r


def h128(v: int | str) -> str:
    return v if isinstance(v, str) else f"{v:032x}"


def h64(v: int | str) -> str:
    return v if isinstance(v, str) else f"{v:016x}"


def main() -> None:
    rng = random.Random(26041)
    out: list[str] = []

    def rand_f64(lo: int, hi: int) -> float:
        m = rng.getrandbits(52) | (1 << 52)
        x = float(m) * 2.0 ** (rng.randint(lo, hi) - 52)
        return -x if rng.random() < 0.5 else x

    def rand_quad(lo: int, hi: int) -> int:
        sign = rng.getrandbits(1)
        return (
            (sign << 127)
            | ((rng.randint(lo, hi) + BIAS) << FRAC_BITS)
            | rng.getrandbits(FRAC_BITS)
        )

    def emit2(a: int, b: int) -> None:
        for name, fn in (
            ("add", op_add),
            ("sub", op_sub),
            ("mul", op_mul),
            ("div", op_div),
        ):
            out.append(f"{name}\t{h128(a)}\t{h128(b)}\t{h128(fn(a, b))}")

    # 1. Operands that are exact `real(dp)` values, as in upstream's
    #    `real(x, kind=qp)` call sites; narrow and wide exponent spreads.
    for _ in range(150):
        emit2(q_from_float(rand_f64(-3, 3)), q_from_float(rand_f64(-3, 3)))
    for _ in range(100):
        emit2(
            q_from_float(rand_f64(-300, 300)), q_from_float(rand_f64(-300, 300))
        )
    # 2. Full 113-bit operands with nearby exponents (cancellation, carries).
    for _ in range(200):
        e = rng.randint(-60, 60)
        emit2(rand_quad(e - 2, e + 2), rand_quad(e - 2, e + 2))
    for _ in range(60):
        emit2(rand_quad(-200, 200), rand_quad(-200, 200))
    # 3. Exponent gaps around the significand width (sticky-bit and tie paths).
    for gap in (1, 2, 3, 57, 110, 111, 112, 113, 114, 115, 116, 117, 130, 300):
        for _ in range(4):
            a = rand_quad(0, 0)
            emit2(a, rand_quad(-gap, -gap))
            # An exact power of two `gap` binades below: produces ties.
            tie = (rng.getrandbits(1) << 127) | ((BIAS - gap) << FRAC_BITS)
            emit2(a, tie)
            emit2(a | 1, tie)
    # 4. Results in the subnormal range and at the overflow threshold.
    for _ in range(60):
        emit2(rand_quad(-16382, -16260), rand_quad(-16382, -16260))
        emit2(rand_quad(-8250, -8130), rand_quad(-8250, -8130))
        emit2(rand_quad(16300, 16383), rand_quad(16300, 16383))
        emit2(rand_quad(8130, 8250), rand_quad(-8250, -8130))
    subnormals = [1, 2, 3, (1 << 111) | 1, (1 << 112) - 1, rng.getrandbits(100)]
    for s in subnormals:
        for t in subnormals:
            emit2(s, t | SIGN)
        emit2(s, rand_quad(-16382, -16300))
        emit2(s, rand_quad(-5, 5))
        emit2(rand_quad(100, 16383), s)
    # 5. Special values.
    one = BIAS << FRAC_BITS
    max_finite = INF - 1
    specials = [
        0,
        SIGN,
        INF,
        SIGN | INF,
        INF | (1 << 111),
        one,
        SIGN | one,
        1,
        max_finite,
        SIGN | max_finite,
        one << 0 | 1,
    ]
    for a in specials:
        for b in specials:
            emit2(a, b)
    # 6. Conversions.
    for _ in range(150):
        d = int.from_bytes(struct.pack(">d", rand_f64(-1022, 1023)), "big")
        out.append(f"from_f64\t{h64(d)}\t-\t{h128(op_from_f64(d))}")
    f64_specials = [
        0,
        1 << 63,
        1,
        (1 << 63) | 1,
        (1 << 52) - 1,
        1 << 52,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000000,
        rng.getrandbits(52),
        rng.getrandbits(40),
    ]
    for d in f64_specials:
        out.append(f"from_f64\t{h64(d)}\t-\t{h128(op_from_f64(d))}")
    for _ in range(200):
        q = rand_quad(-1100, 1030)
        out.append(f"to_f64\t{h128(q)}\t-\t{h64(op_to_f64(q))}")
    for _ in range(200):
        # Exactly 53 significant bits followed by a tie or near-tie tail.
        top = rng.getrandbits(52) << 60
        tail = rng.choice(
            [0, 1, (1 << 59) - 1, 1 << 59, (1 << 59) + 1, (1 << 60) - 1]
        )
        e = rng.choice([0, 5, -700, 1023, -1022, -1023, -1040, -1074, -1075])
        q = (rng.getrandbits(1) << 127) | ((e + BIAS) << FRAC_BITS) | top | tail
        out.append(f"to_f64\t{h128(q)}\t-\t{h64(op_to_f64(q))}")
    for q in specials + [(BIAS + 1024) << FRAC_BITS, (BIAS - 1076) << FRAC_BITS]:
        out.append(f"to_f64\t{h128(q)}\t-\t{h64(op_to_f64(q))}")
    # 7. Integer powers.
    for n in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 13):
        for _ in range(8):
            a = q_from_float(rand_f64(-20, 20))
            out.append(f"powi\t{h128(a)}\t{n}\t{h128(op_powi(a, n))}")
        a = rand_quad(-3, 3)
        out.append(f"powi\t{h128(a)}\t{n}\t{h128(op_powi(a, n))}")

    sys.stdout.write(
        "# IEEE-754 binary128 test vectors from exact rational arithmetic.\n"
        "# Generated by gen_binary128_vectors.py (seed 26041); do not edit.\n"
        "# Columns: op, a, b, result. Values are big-endian hexadecimal bit\n"
        "# patterns (32 digits for binary128, 16 for binary64); `nan` means\n"
        "# any NaN; `b` is `-` for conversions and a decimal integer for powi.\n"
    )
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
