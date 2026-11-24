! Oracle for FlexMESA's pulse-data writers (#491). It runs the write
! statements of MESA 26.4.1's write_gyre_data (star/private/pulse_gyre.f90:
! 419-474), write_fgong_data (pulse_fgong.f90:633-708, both fgong_ivers)
! and write_osc_data (pulse_osc.f90:555-606) on synthetic data, and prints
! edge values with each edit descriptor. Built with the MESA SDK gfortran:
!   gfortran -O0 fmt_oracle.f90 -o fmt_oracle && ./fmt_oracle > fmt_oracle_expected.txt
program fmt_oracle
  implicit none
  integer, parameter :: dp = kind(1d0)
  integer, parameter :: n = 24
  real(dp) :: v(n), g(15), p(40,2)
  character(len=256) :: hdr(4)
  character(len=24) :: version_number
  character(len=256) :: format_for_osc_data
  integer :: i, j, k
  v = [0d0, -0d0, 1d0, -1d0, 1.988409870698050380d33, 6.674300000000000413d-8, &
       -5.522647688173986325d1, 1d99, 1d100, -1d100, 1.5d-100, 1d-99, &
       9.99999999999999999d99, 0.5d0, 2.5d0, 1d-300, 4.9406564584124654d-324, &
       2.2250738585072014d-308, 1.7976931348623157d308, log(tiny(0d0)), &
       123456789.123456789d0, 0.1d0, 9.9999999999999999d-1, 3.14159265358979d250]
  do i = 1, 15
     g(i) = v(mod(i-1, n) + 1)
  end do
  do j = 1, 2
     do i = 1, 40
        p(i,j) = v(mod(i+7*j-1, n) + 1)
     end do
  end do
  version_number = 'r26.4.1'
  hdr(1) = 'FGONG file'
  hdr(2) = 'Created by MESAstar'
  hdr(3) = ''
  hdr(4) = ''
  format_for_osc_data = '(1P5E19.12,x)'

  write(*,'(a)') '=== gyre'
  write(*, 100) 2, g(1:3), 101
100 format(I6, 3(1X,1PE26.16), 1X, I6)
  do j = 1, 2
     write(*, 110) j, p(1:18,j)
110  format(I6, 99(1X,1PE26.16))
  end do

  write(*,'(a)') '=== fgong1300'
  do k = 1, 4
     write(*, *) trim(hdr(k))
  end do
  write(*,'(4I10)') 2, 15, 40, 1300
  write(*,'(1P,5(X,E26.18E3))') (g(i), i=1,15)
  do k = 1, 2
     write(*,'(1P,5(X,E26.18E3))') (p(j,k), j=1,40)
  end do

  write(*,'(a)') '=== fgong300'
  do k = 1, 4
     write(*, *) trim(hdr(k))
  end do
  write(*,'(4I10)') 2, 15, 40, 300
  write(*,'(1P5E16.9,x)') (g(i), i=1,15)
  do k = 1, 2
     write(*,'(1P5E16.9,x)') (p(j,k), j=1,40)
  end do

  write(*,'(a)') '=== osc'
  write(*, *) 'OSC file'
  write(*, *) 'Created by MESAstar version', version_number
  write(*, *)
  write(*, *)
  write(*, '(a)') '14 H1 H2 He3 He4 Li7 Be7 C12 C13 N14 N15 O16 O17 Be9 Si28'
  write(*, '(5I10)') 2, 15, 22, 14, 2000
  write(*, format_for_osc_data) (g(i), i=1,15)
  do k = 1, 2
     write(*, format_for_osc_data) (p(j,k), j=1,36)
  end do

  write(*,'(a)') '=== values'
  do i = 1, n
     write(*,'(4a)') '|', '|'
     write(*,'(1PE26.16)') v(i)
     write(*,'(1PE26.18E3)') v(i)
     write(*,'(1PE16.9)') v(i)
     write(*,'(1PE19.12)') v(i)
  end do
end program fmt_oracle
