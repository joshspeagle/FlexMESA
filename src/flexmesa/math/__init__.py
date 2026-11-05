# SPDX-License-Identifier: LGPL-3.0-or-later
"""Public interface for `flexmesa.math` (`math` module port)."""

from __future__ import annotations

from flexmesa.math import intrinsics, trig

SAFE_LOG_FLOOR = intrinsics.SAFE_LOG_FLOOR
SAFE_LOG_BAD_RETURN = intrinsics.SAFE_LOG_BAD_RETURN
safe_sqrt = intrinsics.safe_sqrt
safe_log = intrinsics.safe_log
safe_log10 = intrinsics.safe_log10
log = intrinsics.log
log10 = intrinsics.log10
log1p = intrinsics.log1p
log2 = intrinsics.log2
exp = intrinsics.exp
expm1 = intrinsics.expm1
exp10 = intrinsics.exp10
pow = intrinsics.pow
powm1 = intrinsics.powm1
pow2 = intrinsics.pow2
pow3 = intrinsics.pow3
pow4 = intrinsics.pow4
pow5 = intrinsics.pow5
pow6 = intrinsics.pow6
pow7 = intrinsics.pow7
pow8 = intrinsics.pow8

cos = trig.cos
sin = trig.sin
tan = trig.tan
cospi = trig.cospi
sinpi = trig.sinpi
tanpi = trig.tanpi
acos = trig.acos
asin = trig.asin
atan = trig.atan
acospi = trig.acospi
asinpi = trig.asinpi
atanpi = trig.atanpi
cosh = trig.cosh
sinh = trig.sinh
tanh = trig.tanh
acosh = trig.acosh
asinh = trig.asinh
atanh = trig.atanh
sgn = trig.sgn
heaviside = trig.heaviside
str_to_double = trig.str_to_double
str_to_vector = trig.str_to_vector

__all__: list[str] = [
    "SAFE_LOG_BAD_RETURN",
    "SAFE_LOG_FLOOR",
    "acos",
    "acosh",
    "acospi",
    "asin",
    "asinh",
    "asinpi",
    "atan",
    "atanh",
    "atanpi",
    "cos",
    "cosh",
    "cospi",
    "exp",
    "exp10",
    "expm1",
    "heaviside",
    "log",
    "log1p",
    "log2",
    "log10",
    "pow",
    "pow2",
    "pow3",
    "pow4",
    "pow5",
    "pow6",
    "pow7",
    "pow8",
    "powm1",
    "safe_log",
    "safe_log10",
    "safe_sqrt",
    "sgn",
    "sin",
    "sinh",
    "sinpi",
    "str_to_double",
    "str_to_vector",
    "tan",
    "tanh",
    "tanpi",
]
