# MESA 26.4.1 Hexadecimal Math Reference Vectors (`#165`)

This directory contains the standalone Fortran generator (`vec_gen_support.f90` and `gen_math_vectors.f90`) and build recipe (`Makefile`) that link a built MESA release `26.4.1` (`r26.4.1`) `math` module (`libmath.a` + MESA SDK `CRMATH 1.2` / `crlibm 1.0beta4`) and emit 16-digit IEEE-754 big-endian hexadecimal input/output reference vectors (`Z16.16`) for every numeric function in `flexmesa-math`.

## Regenerating the Reference Vectors

With `MESASDK_ROOT` initialized (`mesasdk_init.sh`) and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/math_hex_vectors generate
```

## Coverage & NaN Rule

- **47 reference sections (`474,062` vectors total)**:
  - **36 `math_lib` unary functions** (`[unary:<name>]`): `safe_sqrt`, `log`, `safe_log`, `log10`, `safe_log10`, `log1p`, `log2`, `exp`, `exp10`, `expm1`, `powm1`, `pow2` through `pow8`, `cos`, `sin`, `tan`, `cospi`, `sinpi`, `tanpi`, `acos`, `asin`, `atan`, `acospi`, `asinpi`, `atanpi`, `cosh`, `sinh`, `tanh`, `acosh`, `asinh`, and `atanh`.
  - **4 `math_lib` / `crmath` / integer-power binary functions**: `pow` (`pow_r_`), `powi` (`pow_i_`), `powi_f90` (Fortran `x ** n` via `__powidf2`), and `hypot` (`crmath::hypot`).
  - **7 Fortran platform-intrinsic sections (`#196`, `#198`)**: `dlog`, `dlog10`, `intrinsic_log10`, `intrinsic_exp`, `atan2`, `datan2`, and `intrinsic_pow` (`real(dp) ** real(dp)`), recorded in an isolated Fortran module without `use math_lib` or `use crmath` to measure `crlibm` round-toward-zero vs round-to-nearest match rates against the `26.4.1` reference build (`DECISIONS #23`).
- **Inputs per section**: `>= 10,067` vectors per section (`5,000` SplitMix64 samples across all biased IEEE-754 exponents `0..=2046`, `5,000` domain-targeted samples, and explicit IEEE-754 edge cases covering `±0.0`, subnormals, `±Inf`, `NaN`, `±HUGE`, `±TINY`, `exp`/`cosh`/`sinh` overflow and underflow boundaries, large `sin`/`cos` arguments up to `10^300`, integer and non-integer `exp10` arguments, and negative/zero/unit bases with integer and non-integer exponents for `pow`, `intrinsic_pow`, `powi`, and `powi_f90`).
- **Bit-comparison rule**:
  - For all 40 `math_lib` / `crmath` / `powi_f90` sections (`403,258` vectors), non-`NaN` outputs must match all 64 IEEE-754 bits (`u64::to_bits()`) identically (`0` bit differences, including `-0.0` vs `+0.0`).
  - For the 7 platform-intrinsic sections (`70,804` vectors), `mesa_26_4_1_vectors.rs` verifies `<= 1` ULP across every input and pins the measured `crlibm` round-to-nearest vs round-toward-zero bit-exact match counts (`100.00%` on `dlog` and `intrinsic_exp`, `99.84%`/`99.76%` on `dlog10`/`intrinsic_log10`, `99.97%` on `intrinsic_pow`, and `93.42%`/`93.60%` on `atan2`/`datan2`).
  - **Documented `NaN` rule**: IEEE-754 does not standardize `NaN` sign or payload bits across hardware and compilers (`x86_64` SSE2 vs `aarch64` FPU quiet-`NaN` canonicalization). An expected `NaN` vector matches if and only if the actual output is also an IEEE-754 quiet `NaN` (`actual.is_nan() && (actual.to_bits() & 0x0008_0000_0000_0000) != 0`).
