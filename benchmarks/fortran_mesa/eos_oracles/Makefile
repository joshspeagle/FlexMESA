# SPDX-License-Identifier: LGPL-3.0-or-later
# Build and run the MESA 26.4.1 EOS module oracle generator (#161, T2-15, T1-11).
#
# Prerequisites:
#   - `MESASDK_ROOT` initialized (`source $MESASDK_ROOT/bin/mesasdk_init.sh`)
#   - `MESA_DIR` pointing to a built MESA release 26.4.1 tree

FC = gfortran
FFLAGS ?= -march=x86-64 -mtune=generic -mfpmath=sse -O2 -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check -fopenmp -ffree-form -ffree-line-length-none

ifeq ($(strip $(MESA_DIR)),)
$(error MESA_DIR must be set to a built MESA 26.4.1 checkout)
endif

MESA_INCS := -I$(MESA_DIR)/include -I$(MESA_DIR)/eos/make -I$(MESASDK_ROOT)/include
MESA_LIBS := $(MESA_DIR)/lib/libeos.a $(MESA_DIR)/lib/libinterp_2d.a \
	$(MESA_DIR)/lib/libnum.a $(MESA_DIR)/lib/libmtx.a \
	$(MESA_DIR)/lib/libinterp_1d.a $(MESA_DIR)/lib/libauto_diff.a \
	$(MESA_DIR)/lib/libchem.a $(MESA_DIR)/lib/libmath.a \
	$(MESA_DIR)/lib/libutils.a $(MESA_DIR)/lib/libconst.a \
	$(shell mesasdk_lapack95_link) $(shell mesasdk_lapack_link) \
	$(shell mesasdk_blas_link) $(shell mesasdk_crmath_link) \
	$(shell mesasdk_crlibm_link) $(shell mesasdk_hdf5_link)
SDK_VER := $(shell mesasdk_version 2>/dev/null || echo unknown)

BUILD_DIR ?= build
BIN := $(BUILD_DIR)/gen_eos_oracles
OUT ?= mesa_26_4_1_eos_oracles.tsv

.PHONY: all generate clean

all: $(BIN)

$(BIN): gen_eos_oracles.f90
	@mkdir -p $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INCS) -J$(BUILD_DIR) gen_eos_oracles.f90 $(MESA_LIBS) -o $@

generate: $(BIN)
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MESA_DIR="$(MESA_DIR)" MESASDK_VERSION_STR="$(SDK_VER)" $(BIN) "$(OUT)"

clean:
	rm -f $(BIN) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod
	rmdir $(BUILD_DIR) 2>/dev/null || true
