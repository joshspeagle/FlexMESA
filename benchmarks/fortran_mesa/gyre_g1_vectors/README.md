# GYRE G1 Numerics-Substrate Reference Vectors from MESA 26.4.1 (`#170`, Package G1)

This directory produces the exact IEEE-754 vectors that `crates/flexmesa-gyre/tests/g1_vectors.rs` checks bit for bit. It also checks the GCC expansions that the Rust code transcribes.

- **Reference build:** MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo record `19722306`), with its bundled GYRE 8.1, ForUM 1.0.4 and CRMATH 1.2.
- **Toolchain:** MESA SDK `x86_64-linux-26.6.1`, GCC 15.2.0.

## Contents

| File | Role |
| :--- | :--- |
| `gen_gyre_g1_vectors.f90` | Generator. It calls the pristine MESA GYRE build's `ext_m`, `math_m` and `integ_m`, ForUM's `sort_indices`/`locate`, and CRMATH through GYRE's `math_m`. It also evaluates complex arithmetic, integer powers and `MATMUL` in procedures compiled here with **exactly GYRE's flags**. Inputs come from a fixed xorshift64* sequence. |
| `Makefile`, `stamp_header.py` | Build with GYRE's flags, run single-threaded, and write `crates/flexmesa-gyre/tests/data/mesa_26_4_1_gyre_g1_vectors.tsv` with a provenance header. |
| `probe_expansions.f90`, `check_gimple.py` | Compile probe procedures with GYRE's flags, dump GIMPLE (`-fdump-tree-optimized`) and the front-end tree (`-fdump-tree-original`), assert the expansions below, and write `gimple_evidence.txt`. |
| `gimple_evidence.txt` | Generated excerpts and check results. |

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk        # x86_64-linux-26.6.1
export MESA_DIR=/path/to/mesa-26.04.1       # built with USE_GYRE = YES
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
cd benchmarks/fortran_mesa/gyre_g1_vectors
make OUT_DIR=/path/to/scratch generate      # writes the vector file
python3 check_gimple.py                     # writes gimple_evidence.txt
cargo test --release -p flexmesa-gyre       # bit-for-bit check
```

GYRE's flags, as `gyre/gyre/build/Make.inc` sets them for MESA's build: `-std=f2018 -finit-real=snan -fmax-errors=25 -fPIC -ffpe-trap=invalid,overflow,zero -O2 -fno-backtrace -fopenmp -ffp-contract=off`.

The generator keeps every input inside the range where these flags do not trap.

## Expansions Established (`check_gimple.py`, Then the Vectors)

| Construct | GCC 15.2.0 at `-O2`, `-fcx-fortran-rules` | Rust |
| :--- | :--- | :--- |
| `a*b`, complex | `(ar*br - ai*bi, ar*bi + ai*br)` | `Cplx * Cplx` |
| `a*r`, `r*a` | `r` widened to `(r, 0)` and the full product kept: `(r*ar - ai*0, ar*0 + r*ai)` | `Cplx::mul_real`, `Cplx::real_mul` |
| `a/b` | Smith's method (`expand_complex_div_wide`): branch on `\|br\| < \|bi\|`; `ratio`, `div = b_big + b_small*ratio`, then two quotients | `Cplx / Cplx` |
| `a/r` | Smith's method with `bi = 0`, no branch: `ratio = 0/r`, `div = r + 0*ratio` | `Cplx::div_real` |
| `r/b` | Smith's method with `ai = 0` | `Cplx::real_div` |
| `z**2` | `(zr*zr - zi*zi, (zr*zi)*2)` | `Cplx::powi2` |
| `z**(-1)` | `1/z` (Smith) | `Cplx::powi_m1` |
| `x**n`, `n` constant | `powi_as_mults` chains: `x**5 = (x*x)*(x*(x*x))`, `x**6 = (x*(x*x))**2`, `x**7 = (x*(x*x))*((x*x)*(x*x))`; negative `n` as `1/x**\|n\|` | `powi::powi_const` |
| `x**n`, `n` variable | `__builtin_powi`, i.e. libgcc `__powidf2` | `powi::powi_var` (`flexmesa_math::powi_f90`) |
| `C = MATMUL(A, B)` | Inlined when `(real(m)*real(n))*real(k) <= 2.7e4` (`REAL(4)`). `C = 0`, then `do j; do l; do i: C(i,j) = C(i,j) + A(i,l)*B(l,j)`. Otherwise `_gfortran_matmul_r8`/`_c8` | `matmul::matmul_rr`/`_cc` (an error beyond the limit) |
| `y = MATMUL(A, x)` | Always inlined: `y = 0`, then `do l; do i: y(i) = y(i) + A(i,l)*x(l)` | `matmul::matvec_r`/`_c` |
| complex `z**3`, `z**4` | `_gfortran_pow_c8_i4` (library) | not ported: GYRE's reached code has no such power |

The library-computed sections check the same expansions inside the pristine MESA objects. For example, `ext_complex_m`'s division goes through Smith's method.

## Sections in the Vector File

**Complex arithmetic:** `cmul`, `cdiv`, `cmulr`, `rmulc`, `cdivr`, `rdivc`, `cpow2`, `cpowm1`. These include divisor ties `|br| = |bi|` and signed-zero parts.

**Integer powers:** `rpow_const` (exponents 2 to 8 and -1 to -3) and `rpow_var` (exponents -12 to 12, zero bases, and the radix powers `2**e` for `e` in `[-1021, 1023]` that GYRE's `ext` conversions use).

**CRMATH through GYRE `math_m`:**

- `abs_c`, `sqrt_c` (with every branch, including signed zeros), `exp_c`, `log_c`;
- CRMATH's own `atan2`;
- `hypot`, `exp_r`, `log_r`, `acos_r`;
- `pow_rr`, `pow_rc`, and `math_m_constants`.

**MATMUL:**

- `matmul_rr`, `matmul_cc`, `matvec_r`, `matvec_c` (square 2, 4, 6, 8 and 12, plus random rectangular shapes);
- `matmul_nested_r`, GYRE's `MATMUL(G, MATMUL(xA, H) - dH)`;
- `commutator_c`.

**GYRE `ext_rt`:**

- construction, the four operators, negation, every comparison, `abs`, `max`/`min`, `valid`, `sqrt`;
- mixed ext/real operators;
- `real()`, including the `e < MINEXPONENT` branch;
- `exp`, construction from complex, `scale`, `product`.

**GYRE `ext_ct`:** the same operations, plus:

- the mixed operators with `ext_rt`, complex and real;
- the constructors from reals and `ext_rt` parts;
- `cmplx()`/`real()`, `real_part`/`imag_part`/`ext_rt(cx)`, `conjg`, `exp`, `sqrt`, `scale`, `product`.

**GYRE `integ_m`:** `integrate` (with and without a mask) and `integral`.

**ForUM:** `sort_indices` (ascending and descending, many ties) and `locate` (left and right, every starting-guess regime).

## Negative Controls

`tests/g1_vectors.rs::negative_controls_show_the_vectors_discriminate` shows that the vectors separate the expansions above from the obvious alternatives:

| Alternative | Records it fails to reproduce |
| :--- | :--- |
| Textbook complex division | 134/200 `cdiv` |
| `__powidf2` in place of the constant-exponent chains | 129/200 `rpow_const` |
| Reversed `MATMUL` summation order | 30/36 `matmul_rr` |
| `flexmesa_math::atan2` (the Fortran-intrinsic port) in place of CRMATH's `atan2` | 54/194 `atan2` |
| Componentwise complex-times-real | 2/200 `cmulr` (signed zeros only) |
