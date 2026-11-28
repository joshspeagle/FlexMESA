# Overshoot Convective-Boundary Radius and Scale-Height Oracles from MESA 26.4.1

This directory holds the generator and the recorded output for the exact reference values pinned by `crates/flexmesa-star/tests/overshoot_hp_cb_mesa_hex_test.rs`.

- **Code under test:** `eval_conv_bdy_r` and `eval_conv_bdy_Hp` (with `eval_conv_bdy_k` and `two_piece_linear_coeffs`) in `star/private/overshoot_utils.f90`, MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`).
- **Generator:** `gen_overshoot_cb_oracles.f90` links the compiled MESA libraries, builds a five-cell `star_info` holding only the fields the routines read (`nz`, `dq`, `r`, `lnPeos`, `lnd`, `cgrav`, `cz_bdy_dq`, `M_center`, `xmstar`, `conv_bdy_q`, `top_conv_bdy`, `conv_bdy_loc`, and `limit_overshoot_Hp_using_size_of_convection_zone = .false.`), places the recorded three-cell stencil at cells 2 to 4 (boundary cell `k = 3`), and calls the real routines. Inputs come from the test's own IEEE-754 bit patterns.
- **Output:** `mesa_26_4_1_overshoot_cb_oracles.tsv`, one row per input and output (`case_id`, `role`, `field`, `part`, hex bits, decimal). Header lines record the MESA version, commit, SDK, math backend, compiler and flags.

## Regenerate

```bash
make -C benchmarks/fortran_mesa/overshoot_cb_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

`MESA_DIR` must be a built MESA `26.4.1` tree and the SDK must be initialized (`source $MESASDK_ROOT/bin/mesasdk_init.sh`).
