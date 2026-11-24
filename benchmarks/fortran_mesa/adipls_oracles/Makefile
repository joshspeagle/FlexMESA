# Build adipls_oracle_driver against one ADIPLS build.
#
#   make [MESA_DIR=<built MESA 26.4.1>] [ADIPLS_LIB=<libadipls.a>] \
#        [EXTRA_OBJS=<objects linked before the library>] [LDWRAP=<-Wl,--wrap=... flags>] \
#        [COVERAGE=yes] [OUT=<binary path>]
#
# ADIPLS_LIB defaults to the pristine MESA build ($(MESA_DIR)/lib/libadipls.a),
# which is the reference. Overlays from build_adipls_overlay.sh (taps,
# coverage) supply their own libadipls.a; EXTRA_OBJS adds the tap and libm
# helpers of adipls_tap.c.
#
# The driver itself is compiled like MESA's astero glue that it replaces
# (gfortran, no optimisation-dependent arithmetic of its own: it only copies
# captured values and calls ADIPLS).

MESA_DIR ?= $(HOME)/mesa-26.04.1
ADIPLS_LIB ?= $(MESA_DIR)/lib/libadipls.a
OUT ?= adipls_oracle_driver
COVERAGE ?= no
FC = gfortran

ifeq ($(COVERAGE),yes)
   COV_FLAGS = --coverage
else
   COV_FLAGS =
endif

$(OUT) : adipls_oracle_driver.f90 $(ADIPLS_LIB) $(EXTRA_OBJS)
	$(FC) -O0 -ffree-line-length-none -J$(dir $(OUT)) -o $@ adipls_oracle_driver.f90 \
	   $(EXTRA_OBJS) $(ADIPLS_LIB) $(LDWRAP) -lm $(COV_FLAGS)

.PHONY: clean
clean:
	-@rm -f adipls_oracle_driver a0drv.mod
