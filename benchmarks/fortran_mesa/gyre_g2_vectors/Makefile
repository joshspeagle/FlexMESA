# Build and run the GYRE G2 LAPACK/BLAS vector generator (#170).
#
#   make MESA_DIR=<built MESA 26.4.1> OUT_DIR=<scratch> TAPS_DIR=<G0 oracle data> generate
#
# The generator links the MESA SDK x86_64-linux-26.6.1 LAPACK and BLAS that
# MESA's makefile_header selects (USE_CRMATH = YES exports
# MESASDK_MATH_SLOT = crmath, so mesasdk_lapack_link and mesasdk_blas_link
# resolve to the crmath math slot), exactly as libgyre is linked into MESA.

MESA_DIR ?= $(HOME)/mesa-26.04.1
include $(MESA_DIR)/utils/makefile_header

OUT_DIR ?= .
TAPS_DIR ?= ../gyre_oracles/data
VECTORS = ../../../crates/flexmesa-gyre/tests/data/mesa_26_4_1_gyre_g2_vectors.tsv

$(OUT_DIR)/gen_gyre_g2_vectors : gen_gyre_g2_vectors.f90
	gfortran -O2 -J$(OUT_DIR) -o $@ $< $(LOAD_CRMATH) $(LOAD_LAPACK) $(LOAD_BLAS)

generate : $(OUT_DIR)/gen_gyre_g2_vectors
	python3 extract_tapped_inputs.py $(TAPS_DIR) $(OUT_DIR)/tapped_inputs.txt
	python3 stamp_header.py --taps-dir $(TAPS_DIR) --inputs $(OUT_DIR)/tapped_inputs.txt \
	   --out $(VECTORS) -- $(OUT_DIR)/gen_gyre_g2_vectors

.PHONY: generate
