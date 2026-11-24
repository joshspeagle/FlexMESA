# Build and run the GYRE G1 numerics-substrate vector generator (#170).
#
#   make MESA_DIR=<built MESA 26.4.1> OUT_DIR=<scratch dir> generate
#
# The generator is compiled with exactly GYRE's flags as MESA 26.4.1 builds
# it (gyre/gyre/build/Make.inc with DEBUG=no OMP=yes FPE=yes PORTABLE=yes)
# and linked against the pristine MESA GYRE build, ForUM and CRMATH.

MESA_DIR ?= $(HOME)/mesa-26.04.1
include $(MESA_DIR)/utils/makefile_header

GYRE_BUILD ?= $(MESA_DIR)/gyre/gyre/build
OUT_DIR ?= .
GYRE_FFLAGS = -std=f2018 -finit-real=snan -fmax-errors=25 -fPIC \
   -ffpe-trap=invalid,overflow,zero -O2 -fno-backtrace -fopenmp -ffp-contract=off
VECTORS = ../../../crates/flexmesa-gyre/tests/data/mesa_26_4_1_gyre_g1_vectors.tsv

$(OUT_DIR)/gen_gyre_g1_vectors : gen_gyre_g1_vectors.f90
	gfortran $(GYRE_FFLAGS) -I$(GYRE_BUILD) -I$(MESA_DIR)/include -J$(OUT_DIR) \
	   -o $@ $< $(GYRE_BUILD)/libgyre.a \
	   -L$(MESA_LIB_DIR) $(LOAD_MESA_NUMERICS) $(LD_FLAGS)

generate : $(OUT_DIR)/gen_gyre_g1_vectors
	python3 stamp_header.py --flags "$(GYRE_FFLAGS)" --out $(VECTORS) \
	   -- $(OUT_DIR)/gen_gyre_g1_vectors

.PHONY: generate
