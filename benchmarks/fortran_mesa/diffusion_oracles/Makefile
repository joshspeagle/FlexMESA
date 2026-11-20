# SPDX-License-Identifier: LGPL-3.0-or-later
# Build and run the MESA 26.4.1 element diffusion and phase separation oracle
# generator (#156).
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
	atm net rates kap eos colors ionization turb neu \
	interp_2d num mtx interp_1d auto_diff chem math utils const forum

MESA_STAR_MODS := star star_data gyre gyre_mesa $(MESA_BASE_MODULES)

ifeq ($(wildcard $(MESA_DIR)/build/const/.*),)
MESA_PRIVATE_MOD_DIRS := $(wildcard $(addsuffix /make,$(addprefix $(MESA_DIR)/,$(MESA_STAR_MODS))))
MESA_INCS := -I$(MESA_DIR)/include $(addprefix -I,$(MESA_PRIVATE_MOD_DIRS))
MESA_LIB_PATH = $(MESA_DIR)/lib/lib$(1).a
else
MESA_INCS := $(foreach m,$(MESA_STAR_MODS),$(if $(wildcard $(MESA_DIR)/build/$(m)),-I$(MESA_DIR)/build/$(m)/include -I$(MESA_DIR)/build/$(m)/modules))
MESA_LIB_PATH = $(firstword $(wildcard $(MESA_DIR)/build/$(1)/lib/lib$(1).a $(MESA_DIR)/build/gyre/lib/lib$(1).a) $(MESA_DIR)/build/$(1)/lib/lib$(1).a)
endif

MESA_STAR_LIBS := \
	$(call MESA_LIB_PATH,star) \
	$(call MESA_LIB_PATH,star_data) \
	$(call MESA_LIB_PATH,gyre_mesa) \
	$(call MESA_LIB_PATH,gyre) \
	$(foreach m,$(MESA_BASE_MODULES),$(call MESA_LIB_PATH,$(m)))

ifdef MESASDK_ROOT
SDK_ENV := PATH="$(MESASDK_ROOT)/bin:$(PATH)" MESASDK_ROOT="$(MESASDK_ROOT)" MESASDK_MATH_SLOT="$${MESASDK_MATH_SLOT:-crmath}"
MESA_INCS += -I$(MESASDK_ROOT)/include
SDK_CRMATH := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_crmath_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lcrmath -lcrlibm")
SDK_LAPACK95 := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_lapack95_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/crmath/lib -llapack95 -ltmg -llapack -lblas")
SDK_LAPACK := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_lapack_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/crmath/lib -llapack -lblas")
SDK_BLAS := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_blas_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/crmath/lib -lblas")
SDK_HDF5 := $(filter-out -lz,$(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_hdf5_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lhdf5_fortran -lhdf5"))
SDK_PGPLOT := $(filter-out -lz,$(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_pgplot_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lpgplot"))
SDK_LIBS := -Wl,-rpath,$(MESASDK_ROOT)/lib -Wl,-rpath,$(MESASDK_ROOT)/lib64 $(SDK_CRMATH) $(SDK_LAPACK95) $(SDK_LAPACK) $(SDK_BLAS) $(SDK_HDF5) $(SDK_PGPLOT)
SDK_VER := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_version 2>/dev/null || echo unknown)
FC := $(MESASDK_ROOT)/bin/gfortran
else
SDK_LIBS := $(shell mesasdk_crmath_link) $(shell mesasdk_lapack95_link) $(shell mesasdk_lapack_link) $(shell mesasdk_blas_link) $(filter-out -lz,$(shell mesasdk_hdf5_link)) $(filter-out -lz,$(shell mesasdk_pgplot_link))
SDK_VER := $(shell mesasdk_version 2>/dev/null || echo unknown)
endif

BUILD_DIR ?= build
BIN := $(BUILD_DIR)/gen_diffusion_oracles
KERNELS_INC := $(BUILD_DIR)/mesa_private_kernels.inc
OUT ?= mesa_26_4_1_diffusion_oracles.tsv

# Private MESA routines (not exported by libstar.a) whose source text is copied
# verbatim from the MESA tree, each from its opening line to its matching end line.
DIFFUSION_SUPPORT_SRC := $(MESA_DIR)/star/private/diffusion_support.f90
PHASE_SEPARATION_SRC := $(MESA_DIR)/star/private/phase_separation.f90

# $(call extract_routine,<subroutine|function>,<name>,<source file>)
define extract_routine
	awk -v kind='$(1)' -v name='$(2)' ' \
		BEGIN { start = "^[ \t]*([a-z()]+[ \t]+)?" kind "[ \t]+" name "[ \t]*\\("; stop = "^[ \t]*end[ \t]+" kind "[ \t]+" name "[ \t]*$$" } \
		tolower($$0) ~ tolower(start) { p = 1; n++ } \
		p { print } \
		p && tolower($$0) ~ tolower(stop) { p = 0; done++ } \
		END { if (n != 1 || done != 1) { print "extract_routine: " kind " " name " not found exactly once" > "/dev/stderr"; exit 1 } }' \
		'$(3)' >> $@.tmp
endef

.PHONY: all generate run clean

all: $(BIN)

$(KERNELS_INC): $(DIFFUSION_SUPPORT_SRC) $(PHASE_SEPARATION_SRC)
	@mkdir -p $(BUILD_DIR)
	@rm -f $@.tmp
	$(call extract_routine,subroutine,get_Dstar_OCP,$(DIFFUSION_SUPPORT_SRC))
	$(call extract_routine,function,bitsafe_erf_fit,$(DIFFUSION_SUPPORT_SRC))
	$(call extract_routine,function,blouin_delta_xo,$(PHASE_SEPARATION_SRC))
	$(call extract_routine,function,blouin_delta_xne,$(PHASE_SEPARATION_SRC))
	mv $@.tmp $@

$(BIN): gen_diffusion_oracles.f90 $(KERNELS_INC)
	@mkdir -p $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INCS) -I$(BUILD_DIR) -J$(BUILD_DIR) gen_diffusion_oracles.f90 $(MESA_STAR_LIBS) $(SDK_LIBS) -o $@

generate: $(BIN)
	OMP_NUM_THREADS=1 MESA_DIR="$(MESA_DIR)" MESASDK_VERSION_STR="$(SDK_VER)" $(BIN) "$(OUT)"

run: generate

clean:
	rm -f $(BIN) $(KERNELS_INC) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod
	rmdir $(BUILD_DIR) 2>/dev/null || true
