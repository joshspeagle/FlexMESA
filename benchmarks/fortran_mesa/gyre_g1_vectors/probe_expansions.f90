! Probe procedures for the GCC 15.2.0 expansions the GYRE G1 port reproduces
! (#170). Compiled by check_gimple.py with exactly the GYRE flags; the dumped
! GIMPLE is compared with the expected operation sequences.
module probe_m
  implicit none
  integer, parameter :: RD = kind(0.d0)
contains
  function cmul(a, b) result(c)
    complex(RD), intent(in) :: a, b
    complex(RD) :: c
    c = a*b
  end function cmul
  function cdiv(a, b) result(c)
    complex(RD), intent(in) :: a, b
    complex(RD) :: c
    c = a/b
  end function cdiv
  function crmul(a, r) result(c)
    complex(RD), intent(in) :: a
    real(RD), intent(in) :: r
    complex(RD) :: c
    c = a*r
  end function crmul
  function cdivr(a, r) result(c)
    complex(RD), intent(in) :: a
    real(RD), intent(in) :: r
    complex(RD) :: c
    c = a/r
  end function cdivr
  function rdivc(r, b) result(c)
    real(RD), intent(in) :: r
    complex(RD), intent(in) :: b
    complex(RD) :: c
    c = r/b
  end function rdivc
  function ipow5(x) result(y)
    real(RD), intent(in) :: x
    real(RD) :: y
    y = x**5
  end function ipow5
  function ipow6(x) result(y)
    real(RD), intent(in) :: x
    real(RD) :: y
    y = x**6
  end function ipow6
  function ipown(x, n) result(y)
    real(RD), intent(in) :: x
    integer, intent(in) :: n
    real(RD) :: y
    y = x**n
  end function ipown
  function cpow2(z) result(y)
    complex(RD), intent(in) :: z
    complex(RD) :: y
    y = z**2
  end function cpow2
  subroutine mm(A, B, C)
    real(RD), intent(in) :: A(:,:), B(:,:)
    real(RD), intent(out) :: C(:,:)
    C = MATMUL(A, B)
  end subroutine mm
  subroutine mv(A, x, y)
    real(RD), intent(in) :: A(:,:), x(:)
    real(RD), intent(out) :: y(:)
    y = MATMUL(A, x)
  end subroutine mv
end module probe_m
module probe_pw_m
implicit none
integer, parameter :: RD = kind(0.d0)
contains
function rp2(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(2); end function
function rp3(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(3); end function
function rp4(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(4); end function
function rp5(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(5); end function
function rp6(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(6); end function
function rp7(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(7); end function
function rp8(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(8); end function
function rpm1(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(-1); end function
function rpm2(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(-2); end function
function rpm3(x) result(y); real(RD), intent(in) :: x; real(RD) :: y; y = x**(-3); end function
function cp2(x) result(y); complex(RD), intent(in) :: x; complex(RD) :: y; y = x**(2); end function
function cp3(x) result(y); complex(RD), intent(in) :: x; complex(RD) :: y; y = x**(3); end function
function cp4(x) result(y); complex(RD), intent(in) :: x; complex(RD) :: y; y = x**(4); end function
function cpm1(x) result(y); complex(RD), intent(in) :: x; complex(RD) :: y; y = x**(-1); end function
end module probe_pw_m
