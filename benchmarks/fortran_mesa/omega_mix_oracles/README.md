# MESA 26.4.1 `hydro_rotation` and `solve_omega_mix` Reference Oracles (`#158`)

This directory contains the standalone Fortran generator (`gen_omega_mix_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_omega_mix_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH` / `crlibm 1.0beta4`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/omega_mix_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **Section 1 (`eval_i_rot_fp_ft`)**: `hydro_rotation::eval_i_rot`, `set1_i_rot`, and `set_rotation_info` (`eval_fp_ft`) across 13 $\omega/\omega_{\mathrm{crit,Roche}}$ values spanning slow, intermediate, near-critical, and retrograde regimes (`0.0, 0.05, 0.15, 0.30, 0.50, 0.60, 0.75, 0.85, 0.89, 0.90, 0.95, 0.99, -0.60`) for both `roche_paxton` (`simple_i_rot_flag = .false.`, `w_div_wc_flag = .true.`) and `roche_simple` (`simple_i_rot_flag = .true.`, `w_div_wc_flag = .false.`), emitting `i_rot`, `d_i_rot_dlnR`, `d_i_rot_dw_div_wc`, `fp_rot`, `ft_rot`, `d_fp_dw_div_wc`, `d_ft_dw_div_wc`, `r_polar`, and `r_equatorial`.
- **Section 2 (`roche_factors`)**: Direct evaluations of public `hydro_rotation::a` and `hydro_rotation::c` (Fabry, Marchant & Sana 2022, Eqs. A.6 and A.12) across 12 non-negative $\omega/\omega_{\mathrm{crit}}$ values in `[0.0, 0.99]`.
- **Section 3 (`w_div_w_roche`)**: `hydro_rotation::w_div_w_roche_omega` and `hydro_rotation::w_div_w_roche_jrot` across 10 $\Omega$ and 10 $j_{\mathrm{rot}}$ inputs spanning zero, sub-critical, transition (`max_w = 0.89` to `max_w2 = 0.90`), supercritical, and retrograde values for both `hard_cap` (`w_div_wc_flag = .false.`) and `sigmoid_cap` (`w_div_wc_flag = .true.`).
- **Section 4 (`rotation_sigmas`)**: `hydro_rotation::get_rotation_sigmas` (`am_sig_j`, `am_sig_omega`) on a 12-zone non-uniform profile in both `sigmas_normal` (`am_sig_term_limit = 1e13`, with zero and negative face viscosity clamps) and `sigmas_capped` (`am_sig_term_limit = 1e3`, saturating the combined diffusivity limit).
- **Section 5 (`solve_omega_mix`)**: Live MESA `26.4.1` `solve_omega_mix::do_solve_omega_mix` (`real(qp)` IEEE-754 `binary128` tridiagonal Newton solver) on a 24-zone rotating $4\,M_\odot$ profile across 12 regimes:
  1. `diffusive_omega_shear`: pure `am_nu_omega` differential shear transport (`11` substeps).
  2. `diffusive_pure_j`: uniform initial $\Omega$ with `am_nu_omega = 0`, `am_nu_j > 0` isolating the `d2irot` curvature and Jacobian terms (`20` substeps).
  3. `diffusive_mixed_omega_j`: coupled `am_nu_omega` and `am_nu_j` transport (`20` substeps).
  4. `diffusive_stiff_1e13`: extreme `am_nu_omega = 1e26` hitting `am_sig_term_limit = 1e13`, conserving total angular momentum to `3.3e-16` (`6` substeps).
  5. `explicit_extra_jdot`: `use_other_torque = .true.` applying spatially varying `extra_jdot` spin-down torque (`11` substeps).
  6. `explicit_extra_omegadot`: `use_other_torque = .true.` applying spatially varying `extra_omegadot` torque alongside `am_nu_omega` and `am_nu_j` (`20` substeps).
  7. `implicit_torque_tridiag`: `use_other_torque_implicit = .true.` with a nonlinear 3-point coupled torque populating all six implicit derivatives `d_extra_jdot_domega_{m1,00,p1}` and `d_extra_omegadot_domega_{m1,00,p1}` (`20` substeps, `23` Newton iterations).
  8. `adaptive_substepping`: localized shear layer with short local mixing timescale driving `steps_used` up to `max_steps = 20`.
  9. `max_iters_retry`: non-converging implicit torque exceeding `max_iters_per_substep = 4` (`outcome = retry`, `steps_used = 1`, `4` iterations).
  10. `conservation_retry_triggered`: zero diffusivity with a surface angular-velocity jump exceeding `angular_momentum_error_retry` (`outcome = retry`, `4` substeps).
  11. `conservation_retry_passed`: identical profile with `angular_momentum_error_retry` above the surface jump (`outcome = keep_going`, `4` substeps).
  12. `simple_i_rot_shear`: diffusive shear transport with `simple_i_rot_flag = .true.` (`11` substeps).
- The TSV header records `mesa_version`, `mesa_git_commit`, `mesasdk_version`, `math_backend`, `compiler` (`iso_fortran_env::compiler_version()`), and `fflags`. Each floating-point value is stored as big-endian 16-hex-digit IEEE-754 raw bits (`0x<16-hex>`) and round-trip decimal (`ES25.17E3`).
