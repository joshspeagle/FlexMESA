# Build gyre_oracle_driver against one GYRE build.
#
#   make MESA_DIR=<built MESA 26.4.1> GYRE_BUILD=<dir with libgyre*.a and *.mod> \
#        [COVERAGE=yes] [EXTRA_LIBS=<archives linked before the SDK libraries>]
#        [OUT=<binary path>]
#
# GYRE_BUILD defaults to the pristine MESA build ($(MESA_DIR)/gyre/gyre/build),
# which is the reference for the mode records. An overlay from
# build_overlay.sh (taps or coverage) is used for intermediate records and
# for line coverage.
#
# Compile flags follow MESA's test programs for GYRE
# (gyre/test/make/makefile_base: TEST_COMPILE, linked with
# FCopenmp against LOAD_MATRIX and LOAD_MESA_NUMERICS).

MESA_DIR ?= $(HOME)/mesa-26.04.1
include $(MESA_DIR)/utils/makefile_header

GYRE_BUILD ?= $(MESA_DIR)/gyre/gyre/build
OUT ?= gyre_oracle_driver
COVERAGE ?= no

ifeq ($(COVERAGE),yes)
   COV_FLAGS = --coverage
else
   COV_FLAGS =
endif

$(OUT) : gyre_oracle_driver.f90
	$(FC) $(FCbasic) $(FCopenmp) -O2 -ffp-contract=off \
	   -I$(GYRE_BUILD) -I$(MESA_DIR)/include -J$(dir $(OUT)) \
	   -o $@ $< \
	   $(GYRE_BUILD)/libgyre_mesa.a $(GYRE_BUILD)/libgyre.a $(EXTRA_LIBS) \
	   -L$(MESA_LIB_DIR) $(LOAD_MESA_NUMERICS) $(LD_FLAGS) $(COV_FLAGS)

.PHONY: clean
clean:
	-@rm -f gyre_oracle_driver gyre_oracle_driver_m.mod
