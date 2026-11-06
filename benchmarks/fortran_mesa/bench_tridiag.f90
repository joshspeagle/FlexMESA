! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_tridiag
!
! Pre-validated 5 x 5 block-tridiagonal fixture and Thomas LU solver
! (make_block_tridiag_fixture and fast_block_thomas_solve) matching
! crates/flexmesa-bench/src/tridiag_fast.rs and src/flexmesa/bench/harness.py.
!
! Provides:
! - make_block_tridiag_fixture: Constructs the diagonally dominant 5 x 5
!   block-tridiagonal Henyey system across nz radial shells.
! - fast_block_thomas_solve: Zero-allocation in-place 5 x 5 block-Thomas LU
!   factorization and forward/backward substitution over pre-allocated arrays.
! -----------------------------------------------------------------------------
module bench_tridiag
   use const_def, only: dp
   implicit none
   private

   public :: make_block_tridiag_fixture
   public :: fast_block_thomas_solve

contains

   !> Constructs the diagonally dominant 5 x 5 block-tridiagonal system across
   !! nz radial shells matching flexmesa-bench and flexmesa.bench:
   !! - lblk(5, 5, nz): Sub-diagonal blocks (lblk(:, :, 1) = 0).
   !! - dblk(5, 5, nz): Diagonal blocks with diagonal 4 + 0.5*(i-1) + 0.2*s.
   !! - ublk(5, 5, nz): Super-diagonal blocks (ublk(:, :, nz) = 0).
   !! - rhs(5, nz): Right-hand side vectors (1 + 0.2*(j-1) - 0.1*s).
   !! where s = k / nz is the normalized shell coordinate in (0, 1].
   subroutine make_block_tridiag_fixture(nz, lblk, dblk, ublk, rhs)
      integer, intent(in) :: nz
      real(dp), intent(out) :: lblk(5, 5, nz), dblk(5, 5, nz), ublk(5, 5, nz)
      real(dp), intent(out) :: rhs(5, nz)
      integer :: k, i, j
      real(dp) :: s, dij

      do k = 1, nz
         s = real(k, dp) / real(nz, dp)
         do j = 1, 5
            do i = 1, 5
               if (i == j) then
                  dblk(i, j, k) = 4.0_dp + 0.5_dp * real(i - 1, dp) &
                     + 0.2_dp * s
               else
                  dij = real(abs(i - j) + 1, dp)
                  dblk(i, j, k) = 0.05_dp / dij
               end if
               lblk(i, j, k) = merge(0.0_dp, &
                  -0.12_dp / real(i + j, dp), k == 1)
               ublk(i, j, k) = merge(0.0_dp, &
                  -0.10_dp / real(i + j, dp), k == nz)
            end do
            rhs(j, k) = (1.0_dp + 0.2_dp * real(j - 1, dp)) - 0.1_dp * s
         end do
      end do
   end subroutine make_block_tridiag_fixture

   !> In-place 5 x 5 LU factorization with partial pivoting (PA = LU).
   !! Overwrites a(1:5, 1:5) with unit-lower L and upper U factors and stores
   !! 1-based row pivot indices in ipiv(1:5).
   pure subroutine lu5_factor(a, ipiv)
      real(dp), intent(inout) :: a(5, 5)
      integer, intent(out) :: ipiv(5)
      integer :: k, i, j, piv
      real(dp) :: max_val, v, tmp, inv_diag, lik

      do k = 1, 5
         piv = k; max_val = abs(a(k, k))
         do i = k + 1, 5
            v = abs(a(i, k))
            if (v > max_val) then
               max_val = v; piv = i
            end if
         end do
         ipiv(k) = piv
         if (piv /= k) then
            do j = 1, 5
               tmp = a(k, j); a(k, j) = a(piv, j); a(piv, j) = tmp
            end do
         end if
         inv_diag = 1.0_dp / a(k, k)
         do i = k + 1, 5
            lik = a(i, k) * inv_diag; a(i, k) = lik
            do j = k + 1, 5
               a(i, j) = a(i, j) - lik * a(k, j)
            end do
         end do
      end do
   end subroutine lu5_factor

   !> Solves a factored 5 x 5 system LU x = P b in-place for a single RHS b(5).
   !! Applies row permutations ipiv(1:5), unit-lower forward substitution, and
   !! upper-triangular back-substitution.
   pure subroutine lu5_solve_vec(lu, ipiv, b)
      real(dp), intent(in) :: lu(5, 5)
      integer, intent(in) :: ipiv(5)
      real(dp), intent(inout) :: b(5)
      integer :: k, i, j, piv
      real(dp) :: tmp, s

      do k = 1, 5
         piv = ipiv(k)
         if (piv /= k) then
            tmp = b(k); b(k) = b(piv); b(piv) = tmp
         end if
      end do
      do i = 2, 5
         s = b(i)
         do j = 1, i - 1
            s = s - lu(i, j) * b(j)
         end do
         b(i) = s
      end do
      do i = 5, 1, -1
         s = b(i)
         do j = i + 1, 5
            s = s - lu(i, j) * b(j)
         end do
         b(i) = s / lu(i, i)
      end do
   end subroutine lu5_solve_vec

   !> Zero-allocation 5 x 5 block-Thomas factorization and solve over
   !! pre-allocated column-major workspace arrays lblk(5,5,nz), dblk(5,5,nz),
   !! ublk(5,5,nz), ipiv(5,nz), and rhs(5,nz).
   !! Overwrites dblk and ublk with block-LU factors and rhs with solution x.
   pure subroutine fast_block_thomas_solve(nz, lblk, dblk, ublk, ipiv, rhs)
      integer, intent(in) :: nz
      real(dp), intent(in) :: lblk(5, 5, nz)
      real(dp), intent(inout) :: dblk(5, 5, nz), ublk(5, 5, nz)
      integer, intent(out) :: ipiv(5, nz)
      real(dp), intent(inout) :: rhs(5, nz)
      integer :: k, i, j, m, col
      real(dp) :: s

      ! Forward block-Thomas elimination pass across shells k = 1..nz.
      do k = 1, nz
         call lu5_factor(dblk(:, :, k), ipiv(:, k))
         if (k < nz) then
            do col = 1, 5
               call lu5_solve_vec(dblk(:, :, k), ipiv(:, k), ublk(:, col, k))
            end do
            do j = 1, 5
               do m = 1, 5
                  do i = 1, 5
                     dblk(i, j, k + 1) = dblk(i, j, k + 1) &
                        - lblk(i, m, k + 1) * ublk(m, j, k)
                  end do
               end do
            end do
         end if
      end do

      ! Forward RHS sweep and backward block substitution.
      do k = 1, nz
         if (k > 1) then
            do i = 1, 5
               s = sum(lblk(i, :, k) * rhs(:, k - 1))
               rhs(i, k) = rhs(i, k) - s
            end do
         end if
         call lu5_solve_vec(dblk(:, :, k), ipiv(:, k), rhs(:, k))
      end do
      do k = nz - 1, 1, -1
         do i = 1, 5
            s = sum(ublk(i, :, k) * rhs(:, k + 1))
            rhs(i, k) = rhs(i, k) - s
         end do
      end do
   end subroutine fast_block_thomas_solve

end module bench_tridiag
