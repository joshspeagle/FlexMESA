! SPDX-License-Identifier: GPL-3.0-or-later
! Generate exact 16-digit IEEE-754 hexadecimal input/output reference vectors
! from real MESA 26.4.1 (`math_lib` + `crmath` + Fortran platform intrinsics)
! for `#165`, `#196`, and `#198`.

program gen_math_vectors
   use, intrinsic :: iso_fortran_env, only: int32, int64, compiler_version
   use const_def, only: dp
   use math_lib, only: &
      MATH_BACKEND, math_init, &
      safe_sqrt, log, safe_log, log10, safe_log10, log1p, log2, &
      exp, exp10, expm1, &
      powm1, pow, pow2, pow3, pow4, pow5, pow6, pow7, pow8, &
      cos, sin, tan, cospi, sinpi, tanpi, &
      acos, asin, atan, acospi, asinpi, atanpi, &
      cosh, sinh, tanh, acosh, asinh, atanh
   use crmath, only: crmath_hypot => hypot
   use mesa_platform_intrinsics, only: &
      f90_dlog, f90_dlog10, f90_log10, f90_exp, f90_atan2, f90_datan2, f90_pow_r
   use vec_gen_support, only: &
      read_mesa_version, current_iso_date, build_unary_edges
   implicit none

   integer, parameter :: N_FULL_EXP = 5000
   integer, parameter :: N_DOMAIN   = 5000

   character(len=512) :: out_path, mesa_dir, sdk_ver, date_iso, mesa_ver
   integer :: unit_out, ios, n_u_edges
   integer(int64) :: rng_state
   integer(int64) :: u_edges(96)

   call get_command_argument(1, out_path)
   if (len_trim(out_path) == 0) out_path = 'mesa_26_4_1_math_vectors.txt'

   call get_environment_variable('MESA_DIR', mesa_dir)
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   call get_environment_variable('VECTOR_GEN_DATE', date_iso)
   if (len_trim(date_iso) == 0) call current_iso_date(date_iso)

   call read_mesa_version(mesa_dir, mesa_ver)

   ! Initialize CRMATH / math_lib exactly as MESA does (`math_lib_crmath.f90:107`).
   call math_init()

   call build_unary_edges(u_edges, n_u_edges)

   open(newunit=unit_out, file=trim(out_path), status='replace', action='write', iostat=ios)
   if (ios /= 0) stop 'Failed to open output file'

   write(unit_out, '(A)') '# MESA 26.4.1 Math Module Exact Hexadecimal Reference Vectors (#165)'
   write(unit_out, '(A)') '# mesa_version: ' // trim(mesa_ver)
   write(unit_out, '(A)') '# math_backend: ' // trim(MATH_BACKEND)
   write(unit_out, '(A)') '# compiler: ' // trim(compiler_version())
   write(unit_out, '(A)') '# mesasdk_version: ' // trim(sdk_ver)
   write(unit_out, '(A)') '# date: ' // trim(date_iso)
   write(unit_out, '(A)') '# seed: 0x2604010000000165'
   write(unit_out, '(A)') '# format_unary: <input_u64_hex16> <output_u64_hex16>'
   write(unit_out, '(A)') '# format_binary_f64: <arg1_u64_hex16> <arg2_u64_hex16> <output_u64_hex16>'
   write(unit_out, '(A)') '# format_binary_i32: <arg1_u64_hex16> <exp_u32_hex8> <output_u64_hex16>'

   ! 36 unary functions
   call emit_unary(unit_out, 'safe_sqrt',  1, u_edges, n_u_edges)
   call emit_unary(unit_out, 'log',        2, u_edges, n_u_edges)
   call emit_unary(unit_out, 'safe_log',   3, u_edges, n_u_edges)
   call emit_unary(unit_out, 'log10',      4, u_edges, n_u_edges)
   call emit_unary(unit_out, 'safe_log10', 5, u_edges, n_u_edges)
   call emit_unary(unit_out, 'log1p',      6, u_edges, n_u_edges)
   call emit_unary(unit_out, 'log2',       7, u_edges, n_u_edges)
   call emit_unary(unit_out, 'exp',        8, u_edges, n_u_edges)
   call emit_unary(unit_out, 'exp10',      9, u_edges, n_u_edges)
   call emit_unary(unit_out, 'expm1',     10, u_edges, n_u_edges)
   call emit_unary(unit_out, 'powm1',     11, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow2',      12, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow3',      13, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow4',      14, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow5',      15, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow6',      16, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow7',      17, u_edges, n_u_edges)
   call emit_unary(unit_out, 'pow8',      18, u_edges, n_u_edges)
   call emit_unary(unit_out, 'cos',       19, u_edges, n_u_edges)
   call emit_unary(unit_out, 'sin',       20, u_edges, n_u_edges)
   call emit_unary(unit_out, 'tan',       21, u_edges, n_u_edges)
   call emit_unary(unit_out, 'cospi',     22, u_edges, n_u_edges)
   call emit_unary(unit_out, 'sinpi',     23, u_edges, n_u_edges)
   call emit_unary(unit_out, 'tanpi',     24, u_edges, n_u_edges)
   call emit_unary(unit_out, 'acos',      25, u_edges, n_u_edges)
   call emit_unary(unit_out, 'asin',      26, u_edges, n_u_edges)
   call emit_unary(unit_out, 'atan',      27, u_edges, n_u_edges)
   call emit_unary(unit_out, 'acospi',    28, u_edges, n_u_edges)
   call emit_unary(unit_out, 'asinpi',    29, u_edges, n_u_edges)
   call emit_unary(unit_out, 'atanpi',    30, u_edges, n_u_edges)
   call emit_unary(unit_out, 'cosh',      31, u_edges, n_u_edges)
   call emit_unary(unit_out, 'sinh',      32, u_edges, n_u_edges)
   call emit_unary(unit_out, 'tanh',      33, u_edges, n_u_edges)
   call emit_unary(unit_out, 'acosh',     34, u_edges, n_u_edges)
   call emit_unary(unit_out, 'asinh',     35, u_edges, n_u_edges)
   call emit_unary(unit_out, 'atanh',     36, u_edges, n_u_edges)

   ! 4 Fortran platform-intrinsic unary sections (#198)
   call emit_unary(unit_out, 'dlog',            37, u_edges, n_u_edges)
   call emit_unary(unit_out, 'dlog10',          38, u_edges, n_u_edges)
   call emit_unary(unit_out, 'intrinsic_log10', 39, u_edges, n_u_edges)
   call emit_unary(unit_out, 'intrinsic_exp',   40, u_edges, n_u_edges)

   ! Binary functions: math_lib + Fortran platform intrinsics (#165, #196, #198)
   rng_state = int(Z'2604010000000201', int64)
   call emit_pow_r(unit_out, 'pow', .true., rng_state)

   rng_state = int(Z'2604010000000202', int64)
   call emit_pow_i(unit_out, 'powi', .true., rng_state)

   rng_state = int(Z'2604010000000203', int64)
   call emit_pow_i(unit_out, 'powi_f90', .false., rng_state)

   rng_state = int(Z'2604010000000204', int64)
   call emit_atan2(unit_out, 'atan2', .false., rng_state)

   rng_state = int(Z'2604010000000205', int64)
   call emit_hypot(unit_out, rng_state)

   rng_state = int(Z'2604010000000206', int64)
   call emit_atan2(unit_out, 'datan2', .true., rng_state)

   rng_state = int(Z'2604010000000207', int64)
   call emit_pow_r(unit_out, 'intrinsic_pow', .false., rng_state)

   close(unit_out)

contains

   integer(int64) function next_u64(state)
      integer(int64), intent(inout) :: state
      integer(int64) :: z
      state = state + int(Z'9E3779B97F4A7C15', int64)
      z = state
      z = ieor(z, ishft(z, -30)) * int(Z'BF58476D1CE4E5B9', int64)
      z = ieor(z, ishft(z, -27)) * int(Z'94D049BB133111EB', int64)
      next_u64 = ieor(z, ishft(z, -31))
   end function next_u64

   real(dp) function bits_to_dp(b)
      integer(int64), intent(in) :: b
      bits_to_dp = transfer(b, 0.0_dp)
   end function bits_to_dp

   integer(int64) function dp_to_bits(x)
      real(dp), intent(in) :: x
      dp_to_bits = transfer(x, 0_int64)
   end function dp_to_bits

   integer(int64) function sample_full_exponent(state, avoid_large_int)
      integer(int64), intent(inout) :: state
      logical, intent(in) :: avoid_large_int
      integer(int64) :: r, sgn, bexp, mant, bits
      real(dp) :: x
      r = next_u64(state)
      sgn = iand(ishft(r, -63), 1_int64)
      bexp = modulo(ishft(r, -52), 2047_int64)
      if (bexp < 0_int64) bexp = bexp + 2047_int64
      mant = iand(r, int(Z'000FFFFFFFFFFFFF', int64))
      bits = ior(ior(ishft(sgn, 63), ishft(bexp, 52)), mant)
      if (avoid_large_int) then
         x = bits_to_dp(bits)
         if (abs(x) >= 1000.0_dp .and. abs(x) <= 2.15e9_dp) then
            if (x == floor(x)) bits = ior(bits, 1_int64)
            x = bits_to_dp(bits)
            if (x == floor(x)) then
               bits = ior(ior(ishft(sgn, 63), ishft(1060_int64, 52)), mant)
            end if
         end if
      end if
      sample_full_exponent = bits
   end function sample_full_exponent

   real(dp) function uniform_01(state)
      integer(int64), intent(inout) :: state
      integer(int64) :: r
      r = ishft(next_u64(state), -11)
      r = iand(r, int(Z'001FFFFFFFFFFFFF', int64))
      uniform_01 = real(r, dp) / real(int(Z'0020000000000000', int64), dp)
   end function uniform_01

   real(dp) function eval_unary_by_id(fid, x) result(y)
      integer, intent(in) :: fid
      real(dp), intent(in) :: x
      select case (fid)
      case (1);  y = safe_sqrt(x)
      case (2);  y = log(x)
      case (3);  y = safe_log(x)
      case (4);  y = log10(x)
      case (5);  y = safe_log10(x)
      case (6);  y = log1p(x)
      case (7);  y = log2(x)
      case (8);  y = exp(x)
      case (9);  y = exp10(x)
      case (10); y = expm1(x)
      case (11); y = powm1(x)
      case (12); y = pow2(x)
      case (13); y = pow3(x)
      case (14); y = pow4(x)
      case (15); y = pow5(x)
      case (16); y = pow6(x)
      case (17); y = pow7(x)
      case (18); y = pow8(x)
      case (19); y = cos(x)
      case (20); y = sin(x)
      case (21); y = tan(x)
      case (22); y = cospi(x)
      case (23); y = sinpi(x)
      case (24); y = tanpi(x)
      case (25); y = acos(x)
      case (26); y = asin(x)
      case (27); y = atan(x)
      case (28); y = acospi(x)
      case (29); y = asinpi(x)
      case (30); y = atanpi(x)
      case (31); y = cosh(x)
      case (32); y = sinh(x)
      case (33); y = tanh(x)
      case (34); y = acosh(x)
      case (35); y = asinh(x)
      case (36); y = atanh(x)
      case (37); y = f90_dlog(x)
      case (38); y = f90_dlog10(x)
      case (39); y = f90_log10(x)
      case (40); y = f90_exp(x)
      case default
         stop 'Unknown unary function id'
      end select
   end function eval_unary_by_id

   integer(int64) function sample_domain_for_unary(fid, idx, state) result(bits)
      integer, intent(in) :: fid, idx
      integer(int64), intent(inout) :: state
      real(dp) :: u, v, x
      integer(int64) :: r, bexp, mant, sgn

      u = uniform_01(state)
      v = uniform_01(state)
      select case (fid)
      case (1, 2, 3, 4, 5, 7, 37, 38, 39)
         r = next_u64(state)
         bexp = modulo(ishft(r, -52), 2047_int64)
         if (bexp < 0_int64) bexp = bexp + 2047_int64
         mant = iand(r, int(Z'000FFFFFFFFFFFFF', int64))
         bits = ior(ishft(bexp, 52), mant)
      case (6)
         if (mod(idx, 2) == 0) then
            r = next_u64(state)
            sgn = iand(ishft(r, -63), 1_int64)
            bexp = modulo(ishft(r, -52), 1023_int64)
            if (bexp < 0_int64) bexp = bexp + 1023_int64
            mant = iand(r, int(Z'000FFFFFFFFFFFFF', int64))
            bits = ior(ior(ishft(sgn, 63), ishft(bexp, 52)), mant)
         else
            x = -0.999999999999_dp + u * 20.0_dp
            bits = dp_to_bits(x)
         end if
      case (8, 10, 31, 32, 33, 40)
         if (mod(idx, 2) == 0) then
            x = -750.0_dp + u * 1465.0_dp
            bits = dp_to_bits(x)
         else
            r = next_u64(state)
            sgn = iand(ishft(r, -63), 1_int64)
            bexp = modulo(ishft(r, -52), 1030_int64)
            if (bexp < 0_int64) bexp = bexp + 1030_int64
            mant = iand(r, int(Z'000FFFFFFFFFFFFF', int64))
            bits = ior(ior(ishft(sgn, 63), ishft(bexp, 52)), mant)
         end if
      case (9)
         if (mod(idx, 4) == 0) then
            x = real(int(-350.0_dp + u * 701.0_dp), dp)
         else
            x = -330.0_dp + u * 645.0_dp + 1.0e-7_dp * (v - 0.5_dp)
            if (x == floor(x)) x = x + 0.125_dp
         end if
         bits = dp_to_bits(x)
      case (25, 26, 28, 29, 36)
         if (mod(idx, 2) == 0) then
            x = -1.0_dp + 2.0_dp * u
            bits = dp_to_bits(x)
         else
            r = next_u64(state)
            sgn = iand(ishft(r, -63), 1_int64)
            bexp = modulo(ishft(r, -52), 1023_int64)
            if (bexp < 0_int64) bexp = bexp + 1023_int64
            mant = iand(r, int(Z'000FFFFFFFFFFFFF', int64))
            bits = ior(ior(ishft(sgn, 63), ishft(bexp, 52)), mant)
         end if
      case (34)
         if (mod(idx, 2) == 0) then
            x = 1.0_dp + u * u * 100.0_dp
            bits = dp_to_bits(x)
         else
            r = next_u64(state)
            bexp = 1023_int64 + modulo(ishft(r, -52), 1024_int64)
            if (bexp < 1023_int64) bexp = bexp + 1024_int64
            mant = iand(r, int(Z'000FFFFFFFFFFFFF', int64))
            bits = ior(ishft(bexp, 52), mant)
         end if
      case default
         if (mod(idx, 2) == 0) then
            x = -100.0_dp + 200.0_dp * u
            bits = dp_to_bits(x)
         else
            bits = sample_full_exponent(state, .false.)
         end if
      end select
   end function sample_domain_for_unary

   subroutine emit_unary(unit_no, fname, fid, edges, n_edges)
      integer, intent(in) :: unit_no, fid, n_edges
      character(len=*), intent(in) :: fname
      integer(int64), intent(in) :: edges(:)
      integer(int64) :: state, xb, yb
      real(dp) :: x, y
      integer :: i
      logical :: avoid_large_int

      avoid_large_int = (fid == 9)
      state = int(Z'2604010000000165', int64) + int(fid, int64) * 1000003_int64

      write(unit_no, '(A)') '[unary:' // trim(fname) // ']'
      do i = 1, n_edges
         xb = edges(i)
         x = bits_to_dp(xb)
         y = eval_unary_by_id(fid, x)
         yb = dp_to_bits(y)
         write(unit_no, '(Z16.16,1X,Z16.16)') xb, yb
      end do
      do i = 1, N_FULL_EXP
         xb = sample_full_exponent(state, avoid_large_int)
         x = bits_to_dp(xb)
         y = eval_unary_by_id(fid, x)
         yb = dp_to_bits(y)
         write(unit_no, '(Z16.16,1X,Z16.16)') xb, yb
      end do
      do i = 1, N_DOMAIN
         xb = sample_domain_for_unary(fid, i, state)
         x = bits_to_dp(xb)
         y = eval_unary_by_id(fid, x)
         yb = dp_to_bits(y)
         write(unit_no, '(Z16.16,1X,Z16.16)') xb, yb
      end do
   end subroutine emit_unary

   subroutine emit_pow_r(unit_no, label, use_math_lib, state)
      integer, intent(in) :: unit_no
      character(len=*), intent(in) :: label
      logical, intent(in) :: use_math_lib
      integer(int64), intent(inout) :: state
      integer(int64) :: bases(14), exps(21), xb, yb, zb
      real(dp) :: x, y, z, u, v
      integer :: i, j

      write(unit_no, '(A)') '[binary_f64:' // trim(label) // ']'

      bases(1)  = int(Z'0000000000000000', int64)
      bases(2)  = int(Z'8000000000000000', int64)
      bases(3)  = dp_to_bits(1.0_dp)
      bases(4)  = dp_to_bits(-1.0_dp)
      bases(5)  = dp_to_bits(2.0_dp)
      bases(6)  = dp_to_bits(-2.5_dp)
      bases(7)  = dp_to_bits(0.5_dp)
      bases(8)  = dp_to_bits(-0.5_dp)
      bases(9)  = int(Z'0000000000000001', int64)
      bases(10) = int(Z'7FEFFFFFFFFFFFFF', int64)
      bases(11) = int(Z'7FF0000000000000', int64)
      bases(12) = int(Z'FFF0000000000000', int64)
      bases(13) = int(Z'7FF8000000000000', int64)
      bases(14) = dp_to_bits(10.0_dp)

      exps(1)  = int(Z'0000000000000000', int64)
      exps(2)  = int(Z'8000000000000000', int64)
      exps(3)  = dp_to_bits(1.0_dp)
      exps(4)  = dp_to_bits(-1.0_dp)
      exps(5)  = dp_to_bits(2.0_dp)
      exps(6)  = dp_to_bits(-2.0_dp)
      exps(7)  = dp_to_bits(3.0_dp)
      exps(8)  = dp_to_bits(-3.0_dp)
      exps(9)  = dp_to_bits(0.5_dp)
      exps(10) = dp_to_bits(-0.5_dp)
      exps(11) = dp_to_bits(1.0_dp / 3.0_dp)
      exps(12) = dp_to_bits(99.0_dp)
      exps(13) = dp_to_bits(-99.0_dp)
      exps(14) = dp_to_bits(100.0_dp)
      exps(15) = dp_to_bits(-100.0_dp)
      exps(16) = dp_to_bits(101.0_dp)
      exps(17) = dp_to_bits(-101.0_dp)
      exps(18) = dp_to_bits(2.5_dp)
      exps(19) = int(Z'7FF0000000000000', int64)
      exps(20) = int(Z'FFF0000000000000', int64)
      exps(21) = int(Z'7FF8000000000000', int64)

      do i = 1, size(bases)
         do j = 1, size(exps)
            xb = bases(i)
            yb = exps(j)
            if (use_math_lib) then
               z = pow(bits_to_dp(xb), bits_to_dp(yb))
            else
               z = f90_pow_r(bits_to_dp(xb), bits_to_dp(yb))
            end if
            zb = dp_to_bits(z)
            write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') xb, yb, zb
         end do
      end do

      do i = 1, N_FULL_EXP
         xb = sample_full_exponent(state, .false.)
         if (mod(i, 3) == 0) then
            u = uniform_01(state)
            y = real(int(-120.0_dp + u * 241.0_dp), dp)
            yb = dp_to_bits(y)
         else
            yb = sample_full_exponent(state, .true.)
         end if
         if (use_math_lib) then
            z = pow(bits_to_dp(xb), bits_to_dp(yb))
         else
            z = f90_pow_r(bits_to_dp(xb), bits_to_dp(yb))
         end if
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') xb, yb, zb
      end do

      do i = 1, N_DOMAIN
         u = uniform_01(state)
         v = uniform_01(state)
         x = exp(-40.0_dp + 80.0_dp * u)
         if (mod(i, 4) == 0) then
            y = real(int(-99.0_dp + v * 199.0_dp), dp)
         else
            y = -20.0_dp + 40.0_dp * v
            if (y == floor(y)) y = y + 0.25_dp
         end if
         xb = dp_to_bits(x)
         yb = dp_to_bits(y)
         if (use_math_lib) then
            z = pow(x, y)
         else
            z = f90_pow_r(x, y)
         end if
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') xb, yb, zb
      end do
   end subroutine emit_pow_r

   subroutine emit_pow_i(unit_no, label, use_math_lib, state)
      integer, intent(in) :: unit_no
      character(len=*), intent(in) :: label
      logical, intent(in) :: use_math_lib
      integer(int64), intent(inout) :: state
      integer(int64) :: bases(12), xb, zb
      integer(int32) :: iexps(17), iy
      real(dp) :: x, z, u
      integer :: i, j

      write(unit_no, '(A)') '[binary_i32:' // trim(label) // ']'

      bases(1)  = int(Z'0000000000000000', int64)
      bases(2)  = int(Z'8000000000000000', int64)
      bases(3)  = dp_to_bits(1.0_dp)
      bases(4)  = dp_to_bits(-1.0_dp)
      bases(5)  = dp_to_bits(2.0_dp)
      bases(6)  = dp_to_bits(-2.5_dp)
      bases(7)  = dp_to_bits(0.5_dp)
      bases(8)  = int(Z'0000000000000001', int64)
      bases(9)  = int(Z'7FEFFFFFFFFFFFFF', int64)
      bases(10) = int(Z'7FF0000000000000', int64)
      bases(11) = int(Z'FFF0000000000000', int64)
      bases(12) = int(Z'7FF8000000000000', int64)

      iexps = [0_int32, 1_int32, -1_int32, 2_int32, -2_int32, 3_int32, -3_int32, &
               4_int32, -4_int32, 7_int32, -7_int32, 8_int32, -8_int32, &
               99_int32, -99_int32, 100_int32, -100_int32]

      do i = 1, size(bases)
         do j = 1, size(iexps)
            xb = bases(i)
            iy = iexps(j)
            x = bits_to_dp(xb)
            if (use_math_lib) then
               z = pow(x, iy)
            else
               z = x ** iy
            end if
            zb = dp_to_bits(z)
            write(unit_no, '(Z16.16,1X,Z8.8,1X,Z16.16)') xb, iy, zb
         end do
      end do

      do i = 1, N_FULL_EXP + N_DOMAIN
         if (i <= N_FULL_EXP) then
            xb = sample_full_exponent(state, .false.)
         else
            u = uniform_01(state)
            xb = dp_to_bits(-50.0_dp + 100.0_dp * u)
         end if
         u = uniform_01(state)
         iy = int(-1100.0_dp + u * 2201.0_dp, int32)
         x = bits_to_dp(xb)
         if (use_math_lib) then
            z = pow(x, iy)
         else
            z = x ** iy
         end if
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z8.8,1X,Z16.16)') xb, iy, zb
      end do
   end subroutine emit_pow_i

   subroutine emit_atan2(unit_no, label, use_datan2, state)
      integer, intent(in) :: unit_no
      character(len=*), intent(in) :: label
      logical, intent(in) :: use_datan2
      integer(int64), intent(inout) :: state
      integer(int64) :: vals(11), yb, xb, zb
      real(dp) :: y, x, z, u, v
      integer :: i, j

      write(unit_no, '(A)') '[binary_f64:' // trim(label) // ']'

      vals(1)  = int(Z'0000000000000000', int64)
      vals(2)  = int(Z'8000000000000000', int64)
      vals(3)  = dp_to_bits(1.0_dp)
      vals(4)  = dp_to_bits(-1.0_dp)
      vals(5)  = int(Z'0000000000000001', int64)
      vals(6)  = int(Z'8000000000000001', int64)
      vals(7)  = int(Z'7FEFFFFFFFFFFFFF', int64)
      vals(8)  = int(Z'FFEFFFFFFFFFFFFF', int64)
      vals(9)  = int(Z'7FF0000000000000', int64)
      vals(10) = int(Z'FFF0000000000000', int64)
      vals(11) = int(Z'7FF8000000000000', int64)

      do i = 1, size(vals)
         do j = 1, size(vals)
            yb = vals(i)
            xb = vals(j)
            if (use_datan2) then
               z = f90_datan2(bits_to_dp(yb), bits_to_dp(xb))
            else
               z = f90_atan2(bits_to_dp(yb), bits_to_dp(xb))
            end if
            zb = dp_to_bits(z)
            write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') yb, xb, zb
         end do
      end do

      do i = 1, N_FULL_EXP
         yb = sample_full_exponent(state, .false.)
         xb = sample_full_exponent(state, .false.)
         if (use_datan2) then
            z = f90_datan2(bits_to_dp(yb), bits_to_dp(xb))
         else
            z = f90_atan2(bits_to_dp(yb), bits_to_dp(xb))
         end if
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') yb, xb, zb
      end do

      do i = 1, N_DOMAIN
         u = uniform_01(state)
         v = uniform_01(state)
         y = -10.0_dp + 20.0_dp * u
         x = -10.0_dp + 20.0_dp * v
         yb = dp_to_bits(y)
         xb = dp_to_bits(x)
         if (use_datan2) then
            z = f90_datan2(y, x)
         else
            z = f90_atan2(y, x)
         end if
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') yb, xb, zb
      end do
   end subroutine emit_atan2

   subroutine emit_hypot(unit_no, state)
      integer, intent(in) :: unit_no
      integer(int64), intent(inout) :: state
      integer(int64) :: vals(12), xb, yb, zb
      real(dp) :: x, y, z, u, v
      integer :: i, j

      write(unit_no, '(A)') '[binary_f64:hypot]'

      vals(1)  = int(Z'0000000000000000', int64)
      vals(2)  = int(Z'8000000000000000', int64)
      vals(3)  = dp_to_bits(3.0_dp)
      vals(4)  = dp_to_bits(-4.0_dp)
      vals(5)  = int(Z'0000000000000001', int64)
      vals(6)  = int(Z'8000000000000001', int64)
      vals(7)  = int(Z'7FEFFFFFFFFFFFFF', int64)
      vals(8)  = int(Z'FFEFFFFFFFFFFFFF', int64)
      vals(9)  = int(Z'7FF0000000000000', int64)
      vals(10) = int(Z'FFF0000000000000', int64)
      vals(11) = int(Z'7FF8000000000000', int64)
      vals(12) = dp_to_bits(1.0_dp)

      do i = 1, size(vals)
         do j = 1, size(vals)
            xb = vals(i)
            yb = vals(j)
            z = crmath_hypot(bits_to_dp(xb), bits_to_dp(yb))
            zb = dp_to_bits(z)
            write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') xb, yb, zb
         end do
      end do

      do i = 1, N_FULL_EXP
         xb = sample_full_exponent(state, .false.)
         yb = sample_full_exponent(state, .false.)
         z = crmath_hypot(bits_to_dp(xb), bits_to_dp(yb))
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') xb, yb, zb
      end do

      do i = 1, N_DOMAIN
         u = uniform_01(state)
         v = uniform_01(state)
         x = -100.0_dp + 200.0_dp * u
         y = -100.0_dp + 200.0_dp * v
         xb = dp_to_bits(x)
         yb = dp_to_bits(y)
         z = crmath_hypot(x, y)
         zb = dp_to_bits(z)
         write(unit_no, '(Z16.16,1X,Z16.16,1X,Z16.16)') xb, yb, zb
      end do
   end subroutine emit_hypot

end program gen_math_vectors
