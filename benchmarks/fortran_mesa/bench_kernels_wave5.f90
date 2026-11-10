! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_wave5
!
! Four-stage Fortran MESA Wave 5 benchmarks across:
!   1. w5_diff_rot     (Wave 5.1: Roche rotation distortion, Caplan OCP
!                       diffusion, rotational instabilities & Spruit dynamo)
!   2. w5_riemann_rsp2 (Wave 5.2: Lagrangian HLLC Riemann solver, RSP2
!                       turbulent convection & Dziembowski pulsation variables)
!   3. w5_binary_rlof  (Wave 5.3: Eggleton Roche lobe, Kepler orbit, Ritter
!                       RLOF, Kerr BH ISCO/Eddington, Jdot & Hut tides)
!   4. w5_astero_seism (Wave 5.4: Acoustic transit integrals, scaling
!                       relations, Ball & Gizon surface corrections & ratios)
! -----------------------------------------------------------------------------
module bench_kernels_wave5
   use const_def, only: dp, pi, pi4, ln10, one_third, two_thirds, &
      four_thirds, four_thirds_pi, sqrt_2_div_3, standard_cgrav, &
      clight, crad, kerg, hbar, amu, me, mp, qe, msun, rsun, lsun, secyer
   use hydro_rotation, only: rot_a_poly => a, log_term_power
   use binary_utils, only: eval_rlobe
   use binary_tides, only: hut_f3 => f3, hut_f4 => f4
   use astero_def, only: correction_factor
   use astero_lib, only: astero_get_cubic_all_freq_corr, &
      astero_get_combined_all_freq_corr
   use bench_harness, only: STAGE3_THREADS, BenchRecord, LatencyStats, &
      now_us, compute_latency_stats, max_rel_diff, max_err3
   implicit none
   private

   public :: bench_w5_diff_rot, bench_w5_riemann_rsp2
   public :: bench_w5_binary_rlof, bench_w5_astero_seism

   real(dp), parameter :: RI_CRIT_DSI = 0.25_dp, X_ALFAP = two_thirds
   real(dp), parameter :: X_ALFAS = 0.5_dp * sqrt_2_div_3
   real(dp), parameter :: X_CEDE = (8.0_dp / 3.0_dp) * sqrt_2_div_3
   real(dp), parameter :: X_GAMMAR = 3.4641016151377544_dp
   real(dp), parameter :: NU_MAX_SUN = 3078.0_dp, DELTA_NU_SUN = 134.91_dp
   real(dp), parameter :: TEFF_SUN_ASTERO = 5772.0_dp
contains
   !> Assigns latency statistics for stage 0..3 into a BenchRecord.
   pure subroutine store_stage_stats(stage, stats, rec)
      integer, intent(in) :: stage
      type(LatencyStats), intent(in) :: stats
      type(BenchRecord), intent(inout) :: rec
      select case (stage)
      case (0); rec%stage0 = stats
      case (1); rec%stage1 = stats
      case (2); rec%stage2 = stats
      case default; rec%stage3 = stats
      end select
   end subroutine store_stage_stats

   !> Clamps a scalar x into [lo, hi].
   pure elemental real(dp) function clamp_dp(x, lo, hi) result(val)
      real(dp), intent(in) :: x, lo, hi
      val = min(max(x, lo), hi)
   end function clamp_dp

   !> Evaluates degenerate-corrected electron thermal energy k_B * T_eff (erg).
   pure real(dp) function eval_effective_kt(n_e, kt) result(kt_eff)
      real(dp), intent(in) :: n_e, kt
      real(dp) :: kf, ef
      kf = (3.0_dp * pi * pi * max(n_e, 1.0e-10_dp))**one_third
      ef = (hbar * kf)**2 / (2.0_dp * me)
      kt_eff = hypot(kt, two_thirds * ef)
   end function eval_effective_kt

   !> Evaluates zone k of w5_diff_rot (Roche distortion, Caplan OCP, Spruit ST).
   real(dp) function eval_diff_rot_zone(k, nz) result(obs)
      integer, intent(in) :: k, nz
      real(dp) :: s, r, m_grav, cgrav, lum, opacity, rho, t, cp, hp
      real(dp) :: grada, gradt, brunt_n2_t, brunt_n2_mu, w_div_wc, dln_om
      real(dp) :: abar, zbar, gam_ocp, kap_ocp, l_edd, gam_edd, om_crit, om
      real(dp) :: w, w2, w4, w6, w_log, a_w, fp, ft, i_rot_r2
      real(dp) :: pref, a_k, b_k, c_k, d_star, grav, q_shear, k_therm
      real(dp) :: n2_tot, tau_dyn, shear_rate, ri, d_dsi, kap_epi2, sh_crit
      real(dp) :: d_sh, n2_ssi, ri_sec, d_ssi, grad_sub, f_cen, d_es, d_gsf
      real(dp) :: kt, e2, n_i, n_e, kt_eff, lam_d, x_lam, ln_lam, t_eff, eta_m
      real(dp) :: n_t2, n_mu2, n_eff2, n_eff, q_min, f_damp, n_mu, n_t, eps_k
      real(dp) :: d_st, nu_st, b_phi, b_r

      s = (real(k, dp) - 0.5_dp) / real(nz, dp)
      r = rsun * (0.08_dp + 0.88_dp * s); cgrav = standard_cgrav
      m_grav = msun * (0.05_dp + 0.95_dp * s * s)
      lum = lsun * (0.04_dp + 0.96_dp * s); grada = 0.398_dp
      opacity = 0.34_dp + 1.40_dp * (1.0_dp - s)
      rho = exp(ln10 * (2.0_dp * (1.0_dp - s) - 0.8_dp * s))
      t = exp(ln10 * (7.15_dp - 1.35_dp * s))
      cp = 2.4e8_dp * (1.0_dp + 0.25_dp * s); hp = 0.085_dp * r
      gradt = 0.26_dp + 0.06_dp * sin(5.0_dp * s)
      brunt_n2_t = 1.6e-6_dp * (0.4_dp + s)
      brunt_n2_mu = 4.5e-7_dp * (1.0_dp - 0.4_dp * s)
      w_div_wc = 0.12_dp + 0.58_dp * s; dln_om = -0.40_dp - 0.30_dp * s
      abar = 1.28_dp + 0.42_dp * (1.0_dp - s)
      zbar = 1.08_dp + 0.22_dp * (1.0_dp - s)
      gam_ocp = 4.0_dp + 14.0_dp * (1.0_dp - s); kap_ocp = 0.20_dp + 0.55_dp * s

      l_edd = (pi4 * clight * cgrav * m_grav) / opacity
      gam_edd = clamp_dp(lum / l_edd, 0.0_dp, 0.99_dp)
      om_crit = sqrt((1.0_dp - gam_edd) * cgrav * m_grav / (r**3))
      om = w_div_wc * om_crit; w = min(abs(w_div_wc), 0.9999_dp)
      w2 = w * w; w4 = w2 * w2; w6 = w4 * w2
      w_log = log(max(1.0_dp - (w**log_term_power), 1.0e-12_dp))
      a_w = max(rot_a_poly(w4, w6, w_log), 1.0_dp)
      fp = clamp_dp((1.0_dp - two_thirds * w2 - 0.2133_dp * w4 &
         - 0.1068_dp * w6) / a_w, 0.75_dp, 1.0_dp)
      ft = clamp_dp((1.0_dp - 0.07955_dp * w4 - 0.2322_dp * w6) / a_w, &
         0.95_dp, 1.0_dp)
      i_rot_r2 = two_thirds * (1.0_dp + 0.1325_dp * w2 + 0.0480_dp * w4) / &
         (1.0_dp + 0.0450_dp * w2)

      pref = sqrt(pi / 3.0_dp)
      a_k = pref * (1.22_dp + 0.405_dp * (kap_ocp**1.77_dp))
      b_k = 0.562_dp * exp(-0.185_dp * (kap_ocp**1.31_dp))
      c_k = 1.11_dp + 0.112_dp * (kap_ocp**1.32_dp)
      d_star = max(pref * a_k * (gam_ocp**(-2.5_dp)) + &
         (b_k / gam_ocp) * exp(-c_k * (gam_ocp**one_third)), 1.0e-12_dp)

      grav = fp * (cgrav * m_grav / (r * r)); q_shear = abs(dln_om)
      k_therm = (4.0_dp * crad * clight * (t**3)) / &
         (3.0_dp * opacity * rho * rho * cp)
      n2_tot = brunt_n2_t + brunt_n2_mu; shear_rate = om * q_shear
      tau_dyn = max(sqrt(r / max(grav, 1.0_dp)), 1.0e-6_dp)
      ri = n2_tot / max(shear_rate * shear_rate, 1.0e-30_dp)
      d_dsi = 0.0_dp; d_sh = 0.0_dp; d_ssi = 0.0_dp; d_gsf = 0.0_dp
      if (shear_rate > 0.0_dp .and. n2_tot >= 0.0_dp .and. ri < RI_CRIT_DSI) &
         d_dsi = (hp * hp / tau_dyn) * ((1.0_dp - ri / RI_CRIT_DSI)**2)
      kap_epi2 = 4.0_dp * om * om * (1.0_dp + 0.5_dp * dln_om)
      sh_crit = n2_tot + kap_epi2
      if (sh_crit < 0.0_dp) &
         d_sh = hp * hp * min(sqrt(-sh_crit), 1.0_dp / tau_dyn)
      n2_ssi = max(brunt_n2_t / 6.0_dp, 0.0_dp) + max(brunt_n2_mu, 0.0_dp)
      ri_sec = n2_ssi / max(shear_rate * shear_rate, 1.0e-30_dp)
      if (shear_rate > 0.0_dp .and. n2_ssi > 0.0_dp .and. &
          ri_sec < RI_CRIT_DSI) &
         d_ssi = k_therm * (shear_rate * shear_rate / n2_ssi) * RI_CRIT_DSI
      grad_sub = max(abs(grada - gradt), 1.0e-4_dp)
      f_cen = min(om * om * r / max(grav, 1.0_dp), 0.9_dp)
      d_es = max((k_therm / grad_sub) * f_cen, 0.0_dp)
      if (q_shear > 1.0e-6_dp .and. &
          brunt_n2_mu <= 1.0e-6_dp * abs(brunt_n2_t)) &
         d_gsf = k_therm * (om * om * r / max(grav, 1.0_dp)) * q_shear * &
            min(r / hp, 10.0_dp)

      kt = kerg * max(t, 100.0_dp); e2 = qe * qe
      n_i = max(rho, 1.0e-15_dp) / (abar * amu)
      n_e = max(n_i * zbar, 1.0e-10_dp); kt_eff = eval_effective_kt(n_e, kt)
      lam_d = sqrt(kt_eff / (pi4 * e2 * n_e * (1.0_dp + zbar)))
      x_lam = max(3.0_dp * kt_eff * lam_d / (zbar * e2), 1.5_dp)
      ln_lam = max(log(x_lam), 0.5_dp); t_eff = max(kt_eff / kerg, 100.0_dp)
      eta_m = clamp_dp(7.0e11_dp * zbar * ln_lam * (t_eff**(-1.5_dp)), &
         1.0e-6_dp, 1.0e16_dp)

      n_t2 = max(brunt_n2_t, 0.0_dp); n_mu2 = max(brunt_n2_mu, 0.0_dp)
      n_eff2 = n_mu2 + (eta_m / max(k_therm, eta_m)) * n_t2
      d_st = 0.0_dp; nu_st = 0.0_dp; b_phi = 0.0_dp; b_r = 0.0_dp
      if (q_shear > 1.0e-8_dp .and. n_eff2 > 1.0e-20_dp .and. &
          om > 1.0e-15_dp) then
         n_eff = sqrt(n_eff2)
         q_min = ((n_eff / om)**1.75_dp) * ((eta_m / (r * r * n_eff))**0.25_dp)
         if (q_shear > q_min) then
            f_damp = clamp_dp(1.0_dp - sqrt(q_min / q_shear), 0.0_dp, 1.0_dp)
            if (n_mu2 >= (eta_m / max(k_therm, eta_m)) * n_t2 .and. &
                n_mu2 > 0.0_dp) then
               n_mu = sqrt(n_mu2)
               b_phi = sqrt(pi4 * rho) * r * om * sqrt(q_shear) * (om / n_mu)
               b_r = b_phi * (om / n_mu) * sqrt(q_shear)
               nu_st = (r * r * om / q_shear) * ((om / n_mu)**4) * f_damp
               d_st = (r * r * om) * ((om / n_mu)**4) * &
                  (max(eta_m / (r * r * om), 1.0e-12_dp)**0.25_dp) * f_damp
            else if (n_t2 > 0.0_dp) then
               n_t = sqrt(n_t2)
               eps_k = max(eta_m / (r * r * n_t), 1.0e-20_dp)
               b_phi = sqrt(pi4 * rho) * r * om * sqrt(om / n_t) * &
                  sqrt(q_shear) * &
                  (max(k_therm / (r * r * n_t), 1.0e-20_dp)**0.125_dp)
               b_r = b_phi * sqrt(om / n_t) * (eps_k**0.125_dp)
               nu_st = r * r * om * sqrt(q_shear) * sqrt(om / n_t) * &
                  (max(k_therm / (r * r * n_t), 1.0e-20_dp)**0.25_dp) * f_damp
               d_st = nu_st * clamp_dp(sqrt(eta_m / max(k_therm, eta_m)), &
                  1.0e-4_dp, 1.0_dp)
            end if
         end if
      end if

      obs = fp + ft + i_rot_r2 + d_star + &
         1.0e-12_dp * (clamp_dp(d_dsi, 0.0_dp, 1.0e18_dp) + &
            clamp_dp(d_sh, 0.0_dp, 1.0e18_dp) + &
            clamp_dp(d_ssi, 0.0_dp, 1.0e18_dp) + &
            clamp_dp(d_es, 0.0_dp, 1.0e18_dp) + &
            clamp_dp(d_gsf, 0.0_dp, 1.0e18_dp) + &
            clamp_dp(d_st, 0.0_dp, 1.0e18_dp) + &
            clamp_dp(nu_st, 0.0_dp, 1.0e18_dp)) + &
         1.0e-5_dp * (clamp_dp(b_phi, 0.0_dp, 1.0e12_dp) + &
            clamp_dp(b_r, 0.0_dp, 1.0e12_dp))
   end function eval_diff_rot_zone

   !> Evaluates w5_riemann_rsp2 profile (HLLC + RSP2 + Dziembowski variables).
   subroutine eval_riemann_rsp2_profile(nz, parallel, out_obs)
      integer, intent(in) :: nz
      logical, intent(in) :: parallel
      real(dp), intent(out) :: out_obs(nz)
      real(dp) :: r_out(nz), r_in(nz), m_grav(nz), rho(nz), t(nz), p(nz)
      real(dp) :: cs(nz), u(nz), u_old(nz), dm(nz), cp(nz), grad_t(nz)
      real(dp) :: kappa(nz), hp(nz), dlnd_dt(nz), u_face(nz), p_face(nz)
      real(dp) :: s, p_bc, r2, area, g_area, p_l_s, p_r_s, c_l, c_r, d_l, d_r
      real(dp) :: s_l, s_r, m_l, m_r, denom, imp, dt, grad_ad, delta
      real(dp) :: alfa, alfap, alfas, alfad, alfam, gammar, min_w2
      real(dp) :: p_f_in, g_out, g_in, g_cell, a_out, a_in, dp_force, res_mom
      real(dp) :: y_sup, a_cell, c_s, c_d, c_dr_p, rad_cond, c_dr, disc, w_st
      real(dp) :: w, w2, p_turb, f_eddy, vol_mid, chi_turb, du_r, eq_visc
      real(dp) :: src_s, snk_d, snk_dr, de_t_dt, v_dz, v_g, a_st, u_dz, c_1
      integer :: k

      dt = 100.0_dp; grad_ad = 0.395_dp; delta = 1.0_dp
      alfa = 1.5_dp; alfap = 1.0_dp; alfas = 1.0_dp; alfad = 1.0_dp
      alfam = 0.25_dp; gammar = 1.0_dp; min_w2 = 1.0e-4_dp

      do k = 1, nz
         s = (real(k, dp) - 0.5_dp) / real(nz, dp)
         r_out(k) = rsun * (1.0_dp - 0.85_dp * (real(k - 1, dp) / real(nz, dp)))
         r_in(k) = rsun * (1.0_dp - 0.85_dp * (real(k, dp) / real(nz, dp)))
         m_grav(k) = msun * ((1.0_dp - 0.92_dp * &
            (real(k - 1, dp) / real(nz, dp)))**2)
         rho(k) = exp(ln10 * (-4.5_dp + 5.5_dp * s))
         t(k) = exp(ln10 * (3.85_dp + 3.15_dp * s))
         p(k) = 1.25e8_dp * rho(k) * t(k)
         cs(k) = sqrt((5.0_dp / 3.0_dp) * p(k) / rho(k))
         u(k) = 1.5e5_dp * (1.0_dp - s) * sin(4.0_dp * s)
         u_old(k) = 0.98_dp * u(k)
         dm(k) = (msun * 0.85_dp / real(nz, dp)) * &
            (0.4_dp + 1.2_dp * s * (1.0_dp - s))
         cp(k) = 2.1e8_dp * (1.0_dp + 0.15_dp * s)
         grad_t(k) = 0.445_dp - 0.10_dp * s
         kappa(k) = 0.40_dp + 2.5_dp * ((1.0_dp - s)**2)
         hp(k) = 0.06_dp * r_out(k); dlnd_dt(k) = -1.0e-6_dp * cos(3.0_dp * s)
      end do
      p_bc = 0.65_dp * p(1)

      !$omp parallel do default(private) &
      !$omp shared(nz, r_out, m_grav, rho, p, cs, u, dm, p_bc, u_face, p_face) &
      !$omp num_threads(STAGE3_THREADS) if(parallel)
      do k = 1, nz
         r2 = max(r_out(k) * r_out(k), 1.0e-30_dp)
         area = pi4 * r2
         g_area = standard_cgrav * m_grav(k) / (r2 * area)
         p_l_s = max(p(k) - 0.5_dp * dm(k) * g_area, 1.0e-30_dp * p(k))
         c_l = max(cs(k), 1.0e-10_dp); d_l = max(rho(k), 1.0e-99_dp)
         if (k == 1) then
            p_r_s = max(p_bc, 0.0_dp); imp = max(d_l * c_l, 1.0e-99_dp)
            u_face(k) = u(k) + (p_l_s - p_r_s) / (2.0_dp * imp)
            p_face(k) = 0.5_dp * (p_l_s + p_r_s)
         else
            p_r_s = max(p(k - 1) + 0.5_dp * dm(k - 1) * g_area, &
               1.0e-30_dp * p(k - 1))
            c_r = max(cs(k - 1), 1.0e-10_dp)
            d_r = max(rho(k - 1), 1.0e-99_dp)
            s_l = min(u(k) - c_l, u(k - 1) - c_r)
            s_r = max(u(k - 1) + c_r, u(k) + c_l)
            m_l = d_l * (s_l - u(k)); m_r = d_r * (s_r - u(k - 1))
            denom = max(m_r - m_l, 1.0e-99_dp)
            u_face(k) = (m_r * u(k - 1) - m_l * u(k) + p_l_s - p_r_s) / denom
            p_face(k) = max((m_r * (p_l_s - m_l * u(k)) - &
               m_l * (p_r_s - m_r * u(k - 1))) / denom, 0.0_dp)
         end if
      end do
      !$omp end parallel do

      !$omp parallel do default(private) &
      !$omp shared(nz, r_out, r_in, m_grav, rho, t, p, u, u_old, dm, cp, &
      !$omp        grad_t, kappa, hp, dlnd_dt, u_face, p_face, out_obs, &
      !$omp        dt, grad_ad, delta, alfa, alfap, alfas, alfad, alfam, &
      !$omp        gammar, min_w2) num_threads(STAGE3_THREADS) if(parallel)
      do k = 1, nz
         s = (real(k, dp) - 0.5_dp) / real(nz, dp)
         a_out = pi4 * r_out(k) * r_out(k); a_in = pi4 * r_in(k) * r_in(k)
         g_out = standard_cgrav * m_grav(k) / &
            max(r_out(k) * r_out(k), 1.0e-30_dp)
         if (k < nz) then
            p_f_in = p_face(k + 1)
            g_in = standard_cgrav * m_grav(k + 1) / &
               max(r_in(k) * r_in(k), 1.0e-30_dp)
         else
            p_f_in = p_face(k) + 0.5_dp * dm(k) * g_out / max(a_out, 1.0e-30_dp)
            g_in = 0.0_dp
         end if
         g_cell = 0.5_dp * (g_out + g_in)
         dp_force = a_out * (p_face(k) - p(k)) - a_in * (p_f_in - p(k))
         res_mom = (u(k) - u_old(k)) / dt - dp_force / dm(k) + g_cell

         y_sup = grad_t(k) - grad_ad
         a_cell = t(k) * p(k) * delta / (rho(k) * hp(k))
         c_s = X_ALFAS * alfa * alfas * a_cell * y_sup
         c_d = (X_CEDE * alfad / alfa) / hp(k)
         c_dr_p = (X_GAMMAR * gammar / alfa)**2
         rad_cond = clight * crad * (t(k)**3) * delta / &
            (rho(k) * rho(k) * cp(k) * cp(k) * kappa(k))
         c_dr = c_dr_p * rad_cond / (hp(k) * hp(k))
         if (grad_t(k) > grad_ad) then
            disc = max(c_dr * c_dr + 4.0_dp * c_d * c_s, 0.0_dp)
            w_st = max((-c_dr + sqrt(disc)) / (2.0_dp * c_d), 0.0_dp)
         else
            w_st = 0.0_dp
         end if
         w = w_st + 2.0e4_dp * (1.0_dp - s); w2 = w * w
         p_turb = X_ALFAP * alfap * rho(k) * w2
         f_eddy = merge(clamp_dp(w2 / min_w2, 0.0_dp, 1.0_dp), 1.0_dp, &
            w2 < min_w2)
         vol_mid = four_thirds_pi * 0.5_dp * ((r_out(k)**3) + (r_in(k)**3))
         chi_turb = f_eddy * four_thirds * alfam * hp(k) * rho(k) * rho(k) * &
            w * vol_mid * vol_mid
         du_r = (u(k) / r_out(k)) - (0.95_dp * u(k) / r_in(k))
         eq_visc = max(chi_turb * ((du_r / dm(k))**2), 0.0_dp)
         src_s = c_s * w; snk_d = c_d * w2 * w; snk_dr = c_dr * w2
         de_t_dt = src_s + eq_visc - snk_d - snk_dr + &
            p_turb * dlnd_dt(k) / rho(k)

         v_dz = g_out * r_out(k) * rho(k) / p(k); v_g = v_dz / (5.0_dp / 3.0_dp)
         a_st = r_out(k) * (1.0e-6_dp * s) / g_out
         u_dz = pi4 * (r_out(k)**3) * rho(k) / m_grav(k)
         c_1 = ((r_out(k) / rsun)**3) / (m_grav(k) / msun)

         out_obs(k) = (u_face(k) / 1.0e6_dp) + (p_face(k) / p(k)) + &
            (res_mom / 1.0e4_dp) + (w / 1.0e5_dp) + (p_turb / p(k)) + &
            (de_t_dt / 1.0e8_dp) + &
            1.0e-2_dp * (v_dz + v_g + a_st + u_dz + c_1)
      end do
      !$omp end parallel do
   end subroutine eval_riemann_rsp2_profile

   !> Evaluates configuration k of w5_binary_rlof (orbit, RLOF, BH, Jdot).
   real(dp) function eval_binary_rlof_zone(k, nz) result(obs)
      integer, intent(in) :: k, nz
      real(dp) :: s, md, ma, per, ecc, rho_s, p_s, t_s, mu_s
      real(dp) :: m_bh_i, a_sp_i, m_bh_c, kap_s, k_div_t, om_star
      real(dp) :: sep, rl_d, r_d, rl_gap, j_orb, grav, hp, v_th, q, f1, mdot0
      real(dp) :: q2, log_q2, gam_q, r_exp, mdot_thin
      real(dp) :: one_m_a2, z1, z2, r_isco, m_eq, m_eff, ratio, a_cur, m_rat
      real(dp) :: eta, mdot_edd, jdot_gr, r_eff_mb, om_orb, jdot_mb
      real(dp) :: q_tid, r_over_a, e2, f3, f4, one_m_e2, bracket, edot_tid

      s = (real(k, dp) - 0.5_dp) / real(nz, dp)
      md = msun * (1.20_dp + 6.80_dp * s)
      ma = msun * (0.80_dp + 4.20_dp * (1.0_dp - 0.5_dp * s))
      per = secyer * (1.5e-3_dp + 2.5e-2_dp * s)
      ecc = 0.05_dp + 0.35_dp * s * (1.0_dp - s)
      rho_s = 1.0e-7_dp * (1.0_dp + 2.0_dp * s)
      p_s = 1.2e5_dp * (1.0_dp + 1.5_dp * s)
      t_s = 5500.0_dp + 6500.0_dp * s; mu_s = 0.61_dp + 0.04_dp * (1.0_dp - s)
      m_bh_i = msun * (5.0_dp + 5.0_dp * s); a_sp_i = 0.15_dp + 0.70_dp * s
      m_bh_c = m_bh_i * (1.05_dp + 0.25_dp * s); kap_s = 0.34_dp
      k_div_t = 1.2e-9_dp * (1.0_dp + s)
      om_orb = 2.0_dp * pi / per; om_star = 1.25_dp * om_orb

      sep = (standard_cgrav * (md + ma) * ((per / (2.0_dp * pi))**2))**one_third
      rl_d = eval_rlobe(md, ma, sep); r_d = rl_d * (0.985_dp + 0.030_dp * s)
      rl_gap = (r_d - rl_d * (1.0_dp - ecc)) / (rl_d * (1.0_dp - ecc))
      j_orb = md * ma * sqrt(standard_cgrav * sep * (1.0_dp - ecc * ecc) / &
         (md + ma))

      grav = standard_cgrav * md / (r_d * r_d); hp = p_s / (grav * rho_s)
      v_th = sqrt(kerg * t_s / (mp * mu_s)); q = ma / md
      f1 = 1.23_dp + 0.5_dp * (log(clamp_dp(q, 0.5_dp, 10.0_dp)) / ln10)
      mdot0 = (2.0_dp * pi / exp(0.5_dp)) * (v_th**3) * (rl_d**3) / &
         (standard_cgrav * md) * rho_s * f1
      q2 = clamp_dp(q, 0.04_dp, 20.0_dp); log_q2 = log(q2) / ln10
      gam_q = merge(0.954_dp + 0.025_dp * log_q2 - 0.038_dp * log_q2 * log_q2, &
         0.954_dp + 0.039_dp * log_q2 + 0.114_dp * log_q2 * log_q2, q2 < 1.0_dp)
      r_exp = (r_d - rl_d) / (hp / gam_q)
      mdot_thin = -mdot0 * exp(min(r_exp, 60.0_dp))

      one_m_a2 = max(1.0_dp - a_sp_i * a_sp_i, 1.0e-30_dp)
      z1 = 1.0_dp + (one_m_a2**one_third) * (((1.0_dp + a_sp_i)**one_third) + &
         (max(1.0_dp - a_sp_i, 1.0e-30_dp)**one_third))
      z2 = sqrt(3.0_dp * a_sp_i * a_sp_i + z1 * z1)
      r_isco = 3.0_dp + z2 - sqrt(max((3.0_dp - z1) * &
         (3.0_dp + z1 + 2.0_dp * z2), 1.0e-30_dp))
      m_eq = m_bh_i * sqrt(r_isco / 6.0_dp)
      m_eff = min(m_bh_c, sqrt(6.0_dp) * m_eq); ratio = m_eq / m_eff
      a_cur = sqrt(two_thirds) * ratio * &
         (4.0_dp - sqrt(max(18.0_dp * ratio * ratio - 2.0_dp, 1.0e-30_dp)))
      m_rat = m_eff / (3.0_dp * m_eq)
      eta = max(1.0_dp - sqrt(max(1.0_dp - m_rat * m_rat, 1.0e-30_dp)), &
         1.0e-12_dp)
      mdot_edd = pi4 * standard_cgrav * m_bh_c / (clight * kap_s * eta)

      jdot_gr = -(32.0_dp * (standard_cgrav**3) * ma * md * (ma + md) / &
         (5.0_dp * (clight**5) * (sep**4))) * j_orb
      r_eff_mb = min(r_d, rl_d)
      jdot_mb = -3.8e-30_dp * md * (rsun**4) * ((r_eff_mb / rsun)**3) * &
         (om_orb**3)

      q_tid = ma / md; r_over_a = r_d / sep; e2 = ecc * ecc
      f3 = hut_f3(ecc); f4 = hut_f4(ecc)
      one_m_e2 = max(1.0_dp - e2, 1.0e-12_dp)
      bracket = f3 - (11.0_dp / 18.0_dp) * (om_star / om_orb) * f4 * &
         (one_m_e2**1.5_dp)
      edot_tid = -27.0_dp * k_div_t * q_tid * (1.0_dp + q_tid) * &
         (r_over_a**8) * ecc * (one_m_e2**(-6.5_dp)) * bracket

      obs = (rl_d / sep) + rl_gap + (j_orb / 1.0e53_dp) + &
         (-mdot_thin / msun * secyer) + a_cur + eta + &
         (mdot_edd / msun * secyer) + (-jdot_gr / 1.0e36_dp) + &
         (-jdot_mb / 1.0e36_dp) + (-edot_tid * secyer * 1.0e4_dp)
   end function eval_binary_rlof_zone

   !> Evaluates w5_astero_seism (acoustic profile, scaling, BG14, r01/r10).
   real(dp) function eval_astero_seism_workload(nz) result(checksum)
      integer, intent(in) :: nz
      integer, parameter :: n_modes = 12
      real(dp) :: r(nz), r_mid(nz), csound(nz), brunt_n2(nz), dt_ac(nz)
      real(dp) :: tau_r(nz), tau_d(nz), l1_mod(n_modes), l1_sig(n_modes)
      integer :: nl(0:3), k, m, n_ord
      real(dp) :: obs(0:3, n_modes), sig(0:3, n_modes), freq(0:3, n_modes)
      real(dp) :: ine(0:3, n_modes), corr_cub(0:3, n_modes)
      real(dp) :: corr_comb(0:3, n_modes), r01(n_modes - 2), r10(n_modes - 2)
      real(dp) :: sig_r01(n_modes - 2), sig_r10(n_modes - 2)
      real(dp) :: s, r_center, r_phot, m_star, teff, l_phot, g_surf, gam1_s
      real(dp) :: rho_s, p_s, tau_0, dnu_asymp, dnu_scal, nu_max_v, nu_ac_v
      real(dp) :: int_pg, delta_pg, target_b1, target_b3, a3_c, b3_c, b1_c
      real(dp) :: d01, d10, dnu1, dnu0, sd01_sq, sd10_sq

      r_center = rsun * 0.05_dp; r_phot = rsun * 0.995_dp
      m_star = 1.05_dp * msun; teff = 5820.0_dp; l_phot = 1.12_dp * lsun
      g_surf = standard_cgrav * m_star / (r_phot * r_phot)
      gam1_s = 1.64_dp; rho_s = 2.2e-7_dp; p_s = 1.1e5_dp

      do k = 1, nz
         s = (real(k, dp) - 0.5_dp) / real(nz, dp)
         r(k) = rsun * (1.0_dp - 0.95_dp * (real(k - 1, dp) / real(nz, dp)))
         r_mid(k) = rsun * (1.0_dp - 0.95_dp * s)
         csound(k) = 1.0e6_dp * (0.8_dp + 4.5_dp * s)
         brunt_n2(k) = merge(2.5e-6_dp * (s - 0.35_dp), -1.0e-7_dp, s > 0.35_dp)
      end do
      do k = 1, nz - 1; dt_ac(k) = max(r(k) - r(k + 1), 0.0_dp) / csound(k)
      end do
      dt_ac(nz) = max(r(nz) - r_center, 0.0_dp) / csound(nz)
      tau_r(nz) = dt_ac(nz); tau_d(1) = 0.0_dp
      do k = nz - 1, 1, -1; tau_r(k) = tau_r(k + 1) + dt_ac(k); end do
      do k = 2, nz; tau_d(k) = tau_d(k - 1) + dt_ac(k - 1); end do
      k = 2
      do while (k < nz .and. r_phot <= r(k)); k = k + 1; end do
      tau_0 = tau_r(k) + (r_phot - r(k)) / csound(k - 1)

      dnu_asymp = 1.0e6_dp / (2.0_dp * tau_0)
      dnu_scal = DELTA_NU_SUN * sqrt(m_star / msun) * &
         ((teff / TEFF_SUN_ASTERO)**3) * ((l_phot / lsun)**(-0.75_dp))
      nu_max_v = NU_MAX_SUN * (m_star / msun) / &
         (((r_phot / rsun)**2) * sqrt(teff / TEFF_SUN_ASTERO))
      nu_ac_v = (0.25e6_dp / pi) * g_surf * sqrt(gam1_s * rho_s / p_s)

      int_pg = 0.0_dp
      do k = 2, nz
         if (brunt_n2(k) > 0.0_dp) int_pg = int_pg + sqrt(brunt_n2(k)) * &
            max(r_mid(k - 1) - r_mid(k), 0.0_dp) / r(k)
      end do
      delta_pg = (sqrt(2.0_dp) * pi * pi) / max(int_pg, 1.0e-30_dp)

      nl = [n_modes, 0, 0, 0]; obs = 0.0_dp; sig = 1.0_dp
      freq = 1.0_dp; ine = 1.0_dp; corr_cub = 0.0_dp; corr_comb = 0.0_dp
      target_b1 = -5.1259727935672011e-5_dp
      target_b3 = -1.4463074898793361e-18_dp
      do m = 1, n_modes
         n_ord = 13 + m; freq(0, m) = 135.0_dp * (real(n_ord, dp) + 1.45_dp)
         ine(0, m) = 1.0e-7_dp * exp((3100.0_dp - freq(0, m)) / 1000.0_dp)
         sig(0, m) = 0.15_dp; l1_sig(m) = 0.18_dp
         obs(0, m) = freq(0, m) + (target_b1 / freq(0, m) + &
            target_b3 * (freq(0, m)**3)) / ine(0, m)
         l1_mod(m) = 135.0_dp * (real(n_ord, dp) + 1.93_dp) - &
            1.8_dp * sin(0.3_dp * real(n_ord, dp))
      end do

      correction_factor = 1.0_dp
      call astero_get_cubic_all_freq_corr( &
         a3_c, .true., nl, obs, sig, freq, corr_cub, ine)
      call astero_get_combined_all_freq_corr( &
         b3_c, b1_c, .true., nl, obs, sig, freq, corr_comb, ine)

      do m = 1, n_modes - 2
         d01 = (freq(0, m) - 4.0_dp * l1_mod(m) + 6.0_dp * freq(0, m + 1) - &
            4.0_dp * l1_mod(m + 1) + freq(0, m + 2)) / 8.0_dp
         dnu1 = l1_mod(m + 1) - l1_mod(m); r01(m) = d01 / dnu1
         sd01_sq = (sig(0, m)**2 + 16.0_dp * (l1_sig(m)**2 + &
            l1_sig(m + 1)**2) + 36.0_dp * (sig(0, m + 1)**2) + &
            sig(0, m + 2)**2) / 64.0_dp
         sig_r01(m) = sqrt(sd01_sq / (dnu1 * dnu1) + &
            (l1_sig(m + 1)**2 + l1_sig(m)**2) * (d01 * d01) / (dnu1**4))

         d10 = -(l1_mod(m) - 4.0_dp * freq(0, m + 1) + &
            6.0_dp * l1_mod(m + 1) - 4.0_dp * freq(0, m + 2) + &
            l1_mod(m + 2)) / 8.0_dp
         dnu0 = freq(0, m + 2) - freq(0, m + 1); r10(m) = d10 / dnu0
         sd10_sq = (l1_sig(m)**2 + 16.0_dp * (sig(0, m + 1)**2 + &
            sig(0, m + 2)**2) + 36.0_dp * (l1_sig(m + 1)**2) + &
            l1_sig(m + 2)**2) / 64.0_dp
         sig_r10(m) = sqrt(sd10_sq / (dnu0 * dnu0) + &
            (sig(0, m + 2)**2 + sig(0, m + 1)**2) * (d10 * d10) / (dnu0**4))
      end do

      checksum = (sum(tau_r + tau_d) / real(nz, dp) + tau_0 + dnu_asymp + &
         dnu_scal + nu_max_v + nu_ac_v + delta_pg + &
         sum(corr_cub(0, 1:n_modes) + corr_comb(0, 1:n_modes)) / &
         real(n_modes, dp)) / 1.0e3_dp + &
         sum(r01 + sig_r01 + r10 + sig_r10) / real(n_modes - 2, dp)
   end function eval_astero_seism_workload

   !> Shared 4-stage loop for zone-independent workloads (1: diff_rot, 2: rlof).
   subroutine run_zone_bench(wname, mode, nz, n_warm, n_rep, rec)
      character(len=*), intent(in) :: wname
      integer, intent(in) :: mode, nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: rec
      real(dp) :: out_s(nz, 0:3), samples(n_rep), t0
      type(LatencyStats) :: stats
      integer :: stg, iter, k

      rec%workload = wname; rec%n_zones = nz
      do stg = 0, 3
         do iter = 1, n_warm + n_rep
            t0 = now_us()
            !$omp parallel do default(private) shared(mode, nz, stg, out_s) &
            !$omp num_threads(STAGE3_THREADS) if(stg == 3)
            do k = 1, nz
               out_s(k, stg) = merge(eval_diff_rot_zone(k, nz), &
                  eval_binary_rlof_zone(k, nz), mode == 1)
            end do
            !$omp end parallel do
            if (iter > n_warm) samples(iter - n_warm) = now_us() - t0
         end do
         call compute_latency_stats(samples, n_rep, stats)
         call store_stage_stats(stg, stats, rec)
      end do
      rec%max_rel_err = max_err3(max_rel_diff(nz, out_s(:, 0), out_s(:, 1)), &
         max_rel_diff(nz, out_s(:, 0), out_s(:, 2)), &
         max_rel_diff(nz, out_s(:, 0), out_s(:, 3)))
      rec%checksum = sum(out_s(:, 0)) / real(nz, dp)
   end subroutine run_zone_bench

   !> Runs the four-stage benchmark for w5_diff_rot.
   subroutine bench_w5_diff_rot(nz, n_warm, n_rep, rec)
      integer, intent(in) :: nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: rec
      call run_zone_bench('w5_diff_rot', 1, nz, n_warm, n_rep, rec)
   end subroutine bench_w5_diff_rot

   !> Runs the four-stage benchmark for w5_binary_rlof.
   subroutine bench_w5_binary_rlof(nz, n_warm, n_rep, rec)
      integer, intent(in) :: nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: rec
      call run_zone_bench('w5_binary_rlof', 2, nz, n_warm, n_rep, rec)
   end subroutine bench_w5_binary_rlof

   !> Runs the four-stage benchmark for w5_riemann_rsp2.
   subroutine bench_w5_riemann_rsp2(nz, n_warm, n_rep, rec)
      integer, intent(in) :: nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: rec
      real(dp) :: out_s(nz, 0:3), samples(n_rep), t0
      type(LatencyStats) :: stats
      integer :: stg, iter

      rec%workload = 'w5_riemann_rsp2'; rec%n_zones = nz
      do stg = 0, 3
         do iter = 1, n_warm + n_rep
            t0 = now_us()
            call eval_riemann_rsp2_profile(nz, stg == 3, out_s(:, stg))
            if (iter > n_warm) samples(iter - n_warm) = now_us() - t0
         end do
         call compute_latency_stats(samples, n_rep, stats)
         call store_stage_stats(stg, stats, rec)
      end do
      rec%max_rel_err = max_err3(max_rel_diff(nz, out_s(:, 0), out_s(:, 1)), &
         max_rel_diff(nz, out_s(:, 0), out_s(:, 2)), &
         max_rel_diff(nz, out_s(:, 0), out_s(:, 3)))
      rec%checksum = sum(out_s(:, 0)) / real(nz, dp)
   end subroutine bench_w5_riemann_rsp2

   !> Runs the four-stage benchmark for w5_astero_seism.
   subroutine bench_w5_astero_seism(nz, n_warm, n_rep, rec)
      integer, intent(in) :: nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: rec
      real(dp) :: c(1, 0:3), samples(n_rep), t0
      type(LatencyStats) :: stats
      integer :: stg, iter

      rec%workload = 'w5_astero_seism'; rec%n_zones = nz
      do stg = 0, 3
         do iter = 1, n_warm + n_rep
            t0 = now_us()
            c(1, stg) = eval_astero_seism_workload(nz)
            if (iter > n_warm) samples(iter - n_warm) = now_us() - t0
         end do
         call compute_latency_stats(samples, n_rep, stats)
         call store_stage_stats(stg, stats, rec)
      end do
      rec%max_rel_err = max_err3(max_rel_diff(1, c(:, 0), c(:, 1)), &
         max_rel_diff(1, c(:, 0), c(:, 2)), max_rel_diff(1, c(:, 0), c(:, 3)))
      rec%checksum = c(1, 0)
   end subroutine bench_w5_astero_seism

end module bench_kernels_wave5
