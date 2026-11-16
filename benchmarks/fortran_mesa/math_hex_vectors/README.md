# MESA 26.4.1 Hexadecimal Math Reference Vectors (`#165`)

This directory contains the standalone Fortran generator (`gen_math_vectors.f90`) and build recipe (`Makefile`) that link a built MESA release `26.4.1` (`r26.4.1`) `math` module (`libmath.a` + MESA SDK `CRMATH 1.2` / `crlibm 1.0beta4`) and emit 16-digit IEEE-754 big-endian hexadecimal input/output reference vectors (`Z16.16`) for every numeric function in `flexmesa-math`.

## Regenerating the Reference Vectors

With `MESASDK_ROOT` initialized (`mesasdk_init.sh`) and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/math_hex_vectors generate
```

## Coverage & NaN Rule

- **41 mapped functions**:
  - **36 unary functions** (`[unary:<name>]`): `safe_sqrt`, `log`, `safe_log`, `log10`, `safe_log10`, `log1p`, `log2`, `exp`, `exp10`, `expm1`, `powm1`, `pow2` through `pow8`, `cos`, `sin`, `tan`, `cospi`, `sinpi`, `tanpi`, `acos`, `asin`, `atan`, `acospi`, `asinpi`, `atanpi`, `cosh`, `sinh`, `tanh`, `acosh`, `asinh`, and `atanh`.
  - **5 binary functions**: `pow` (`pow_r_`), `powi` (`pow_i_`), `powi_f90` (Fortran `x ** n` via `__powidf2`), `atan2` (`crmath::atan2`), and `hypot` (`crmath::hypot`).
- **Inputs per function**: `>= 10,058` vectors per function (`5,000` SplitMix64 samples across all biased IEEE-754 exponents `0..=2046`, `5,000` domain-targeted samples, and explicit IEEE-754 edge cases covering `±0.0`, subnormals, `±Inf`, `NaN`, `±HUGE`, `±TINY`, `exp`/`cosh`/`sinh` overflow and underflow boundaries, large `sin`/`cos` arguments up to `10^300`, integer and non-integer `exp10` arguments, and negative/zero/unit bases with integer and non-integer exponents for `pow`, `powi`, and `powi_f90`).
  - *Upstream `atan2` note*: `CRMATH 1.2`'s `atan2_r_r_dp_` (`crmath_r_dp.f90:296-311`) omits an `else` branch when `(y, x) == (0.0, 0.0)` or either argument is `NaN`, leaving the Fortran return register uninitialized; `(0.0, 0.0)` and `NaN` are therefore excluded from the Fortran `atan2` sweep and tested deterministically in Rust unit tests.
- **Bit-comparison rule**:
  - Non-`NaN` outputs must match all 64 IEEE-754 bits (`u64::to_bits()`) identically (`0` bit differences, including `-0.0` vs `+0.0`).
  - **Documented `NaN` rule**: IEEE-754 does not standardize `NaN` sign or payload bits across hardware and compilers (`x86_64` SSE2 vs `aarch64` FPU quiet-`NaN` canonicalization). An expected `NaN` vector matches if and only if the actual output is also an IEEE-754 quiet `NaN` (`actual.is_nan() && (actual.to_bits() & 0x0008_0000_0000_0000) != 0`).
