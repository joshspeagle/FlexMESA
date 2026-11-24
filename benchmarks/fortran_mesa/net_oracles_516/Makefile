# SPDX-License-Identifier: GPL-3.0-or-later
# Build and run the MESA 26.4.1 one-zone `net::net_get` capture for
# approx21_cr60_plus_co56.net at a hydrogen-burning state (#516).
#
# Prerequisites:
#   - `MESASDK_ROOT` initialized (`source $MESASDK_ROOT/bin/mesasdk_init.sh`)
#   - `MESA_DIR` pointing to a built MESA release 26.4.1 tree

MESA_DIR ?= ../../../../mesa
FC ?= gfortran
FFLAGS ?= -march=x86-64 -mtune=generic -mfpmath=sse -O2 -fopenmp -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check -ffree-line-length-none

ifeq ($(strip $(MESA_DIR)),)
$(error MESA_DIR must be set to a built MESA 26.4.1 checkout)
endif

MESA_BASE_MODULES := \
	net rates eos kap ionization colors neu \
	interp_2d num mtx interp_1d auto_diff chem math utils const forum

ifeq ($(wildcard $(MESA_DIR)/build/const/.*),)
MESA_PRIVATE_MOD_DIRS := $(wildcard $(addsuffix /make,$(addprefix $(MESA_DIR)/,$(MESA_BASE_MODULES))))
MESA_INCS := -I$(MESA_DIR)/include $(addprefix -I,$(MESA_PRIVATE_MOD_DIRS))
MESA_LIB_PATH = $(MESA_DIR)/lib/lib$(1).a
else
MESA_INCS := $(foreach m,$(MESA_BASE_MODULES),$(if $(wildcard $(MESA_DIR)/build/$(m)),-I$(MESA_DIR)/build/$(m)/include -I$(MESA_DIR)/build/$(m)/modules))
MESA_LIB_PATH = $(MESA_DIR)/build/$(1)/lib/lib$(1).a
endif

MESA_LIBS := $(foreach m,$(MESA_BASE_MODULES),$(call MESA_LIB_PATH,$(m)))

ifdef MESASDK_ROOT
SDK_ENV := PATH="$(MESASDK_ROOT)/bin:$(PATH)" MESASDK_ROOT="$(MESASDK_ROOT)" MESASDK_MATH_SLOT="$${MESASDK_MATH_SLOT:-crmath}"
MESA_INCS += -I$(MESASDK_ROOT)/include
SDK_CRMATH := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_crmath_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lcrmath -lcrlibm")
SDK_CRLIBM := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_crlibm_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lcrlibm")
SDK_LAPACK95 := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_lapack95_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/crmath/lib -llapack95 -ltmg -llapack -lblas")
SDK_LAPACK := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_lapack_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/crmath/lib -llapack -lblas")
SDK_BLAS := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_blas_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/crmath/lib -lblas")
SDK_HDF5 := $(filter-out -lz,$(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_hdf5_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lhdf5_fortran -lhdf5"))
SDK_LIBS := -Wl,-rpath,$(MESASDK_ROOT)/lib -Wl,-rpath,$(MESASDK_ROOT)/lib64 $(SDK_CRMATH) $(SDK_CRLIBM) $(SDK_LAPACK95) $(SDK_LAPACK) $(SDK_BLAS) $(SDK_HDF5)
SDK_VER := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_version 2>/dev/null || echo unknown)
FC := $(MESASDK_ROOT)/bin/gfortran
else
SDK_LIBS := $(shell mesasdk_crmath_link) $(shell mesasdk_crlibm_link) $(shell mesasdk_lapack95_link) $(shell mesasdk_lapack_link) $(shell mesasdk_blas_link) $(filter-out -lz,$(shell mesasdk_hdf5_link))
SDK_VER := $(shell mesasdk_version 2>/dev/null || echo unknown)
endif

BUILD_DIR ?= build
BIN := $(BUILD_DIR)/gen_net_516
OUT ?= mesa_26_4_1_net_approx21_zone650.tsv

.PHONY: all generate run clean

all: $(BIN)

$(BIN): gen_net_516.f90
	@mkdir -p $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INCS) -J$(BUILD_DIR) gen_net_516.f90 $(MESA_LIBS) $(SDK_LIBS) -o $@

generate: $(BIN)
	MESA_TEMP_CACHES_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MESA_DIR="$(MESA_DIR)" MESASDK_VERSION_STR="$(SDK_VER)" ORACLE_FFLAGS="$(FFLAGS)" $(BIN) "$(OUT)"

run: generate

clean:
	rm -f $(BIN) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod
	rmdir $(BUILD_DIR) 2>/dev/null || true
