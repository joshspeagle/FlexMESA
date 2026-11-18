# SPDX-License-Identifier: LGPL-3.0-or-later
# Portable Makefile for the Wave 1 to Wave 5 Fortran MESA benchmark suite.
#
# Usage:
#   make MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run

MESA_DIR ?= ../../../mesa
ifeq ($(origin FC), default)
FC := gfortran
endif
FC ?= gfortran
BUILD_DIR ?= build
FFLAGS ?= -O2 -fopenmp -cpp -ffree-form -ffree-line-length-none

MESA_BASE_MODULES := \
	atm net rates kap eos colors ionization turb neu \
	interp_2d num mtx interp_1d auto_diff chem math utils const forum

MESA_STAR_MODS := star star_data gyre gyre_mesa $(MESA_BASE_MODULES)
MESA_WAVE5_MODS := astero adipls binary $(MESA_STAR_MODS)

ifeq ($(wildcard $(MESA_DIR)/build/const/.*),)
# Classic ./install layout: $(MESA_DIR)/lib/lib*.a, $(MESA_DIR)/include/*.mod,
# plus module private $(MESA_DIR)/<module>/make/*.mod directories.
MESA_PRIVATE_MOD_DIRS := $(wildcard $(addsuffix /make,$(addprefix $(MESA_DIR)/,$(MESA_WAVE5_MODS))))
MESA_INC := -I$(MESA_DIR)/include $(addprefix -I,$(MESA_PRIVATE_MOD_DIRS))
MESA_LIB_PATH = $(MESA_DIR)/lib/lib$(1).a
else
# CMake build/<module>/{lib,include,modules} layout.
MESA_INC := $(foreach m,$(MESA_WAVE5_MODS),$(if $(wildcard $(MESA_DIR)/build/$(m)),-I$(MESA_DIR)/build/$(m)/include -I$(MESA_DIR)/build/$(m)/modules))
MESA_LIB_PATH = $(firstword $(wildcard $(MESA_DIR)/build/$(1)/lib/lib$(1).a $(MESA_DIR)/build/gyre/lib/lib$(1).a) $(MESA_DIR)/build/$(1)/lib/lib$(1).a)
endif

MESA_LIBS := $(foreach m,$(MESA_BASE_MODULES),$(call MESA_LIB_PATH,$(m)))

MESA_STAR_LIBS := \
	$(call MESA_LIB_PATH,star) \
	$(call MESA_LIB_PATH,star_data) \
	$(call MESA_LIB_PATH,gyre_mesa) \
	$(call MESA_LIB_PATH,gyre) \
	$(MESA_LIBS)

# Replicate libastero after libstar because libadipls needs astero callbacks.
MESA_WAVE5_LIBS := \
	$(call MESA_LIB_PATH,astero) \
	$(call MESA_LIB_PATH,adipls) \
	$(call MESA_LIB_PATH,binary) \
	$(call MESA_LIB_PATH,star) \
	$(call MESA_LIB_PATH,star_data) \
	$(call MESA_LIB_PATH,astero) \
	$(call MESA_LIB_PATH,gyre_mesa) \
	$(call MESA_LIB_PATH,gyre) \
	$(MESA_LIBS)

ifdef MESASDK_ROOT
SDK_LINK_ENV := PATH="$(MESASDK_ROOT)/bin:$(PATH)" MESASDK_ROOT="$(MESASDK_ROOT)"
SDK_CRMATH_LIBS ?= $(shell $(SDK_LINK_ENV) $(MESASDK_ROOT)/bin/mesasdk_crmath_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lcrmath -lcrlibm")
SDK_LAPACK95_LIBS ?= $(shell $(SDK_LINK_ENV) $(MESASDK_ROOT)/bin/mesasdk_lapack95_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/default/lib -llapack95 -ltmg -llapack -lblas")
SDK_LAPACK_LIBS ?= $(shell $(SDK_LINK_ENV) $(MESASDK_ROOT)/bin/mesasdk_lapack_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/default/lib -llapack -lblas")
SDK_BLAS_LIBS ?= $(shell $(SDK_LINK_ENV) $(MESASDK_ROOT)/bin/mesasdk_blas_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/math-slots/default/lib -lblas")
SDK_HDF5_LIBS ?= $(shell $(SDK_LINK_ENV) $(MESASDK_ROOT)/bin/mesasdk_hdf5_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lhdf5_fortran -lhdf5 -lz")
SDK_PGPLOT_LIBS ?= $(shell $(SDK_LINK_ENV) $(MESASDK_ROOT)/bin/mesasdk_pgplot_link 2>/dev/null || echo "-L$(MESASDK_ROOT)/lib -lpgplot") -lz
BLAS_LAPACK_LIBS ?= $(SDK_CRMATH_LIBS) $(SDK_LAPACK95_LIBS) $(SDK_LAPACK_LIBS) $(SDK_BLAS_LIBS) $(SDK_HDF5_LIBS)
STAR_EXT_LIBS ?= $(SDK_PGPLOT_LIBS)
else
BLAS_LAPACK_LIBS ?= $(shell pkg-config --libs lapack blas 2>/dev/null || echo "-llapack -lblas")
STAR_EXT_LIBS ?=
endif

OBJS := \
	$(BUILD_DIR)/bench_harness.o \
	$(BUILD_DIR)/bench_tridiag.o \
	$(BUILD_DIR)/bench_kernels_micro.o \
	$(BUILD_DIR)/bench_kernels_solver.o \
	$(BUILD_DIR)/run_wave1_bench.o

WAVE2_OBJS := \
	$(BUILD_DIR)/bench_harness.o \
	$(BUILD_DIR)/bench_kernels_wave2.o \
	$(BUILD_DIR)/run_wave2_bench.o

WAVE3A_OBJS := \
	$(BUILD_DIR)/bench_harness.o \
	$(BUILD_DIR)/bench_kernels_wave3a.o \
	$(BUILD_DIR)/run_wave3a_bench.o

WAVE3B_OBJS := \
	$(BUILD_DIR)/bench_harness.o \
	$(BUILD_DIR)/bench_kernels_wave3a.o \
	$(BUILD_DIR)/bench_kernels_wave3b.o \
	$(BUILD_DIR)/run_wave3b_bench.o

WAVE4_OBJS := \
	$(BUILD_DIR)/bench_harness.o \
	$(BUILD_DIR)/bench_kernels_wave4.o \
	$(BUILD_DIR)/run_wave4_bench.o

WAVE5_OBJS := \
	$(BUILD_DIR)/bench_harness.o \
	$(BUILD_DIR)/bench_kernels_wave5.o \
	$(BUILD_DIR)/run_wave5_bench.o

TARGET := $(BUILD_DIR)/run_wave1_bench
TARGET_WAVE2 := $(BUILD_DIR)/run_wave2_bench
TARGET_WAVE3A := $(BUILD_DIR)/run_wave3a_bench
TARGET_WAVE3B := $(BUILD_DIR)/run_wave3b_bench
TARGET_WAVE4 := $(BUILD_DIR)/run_wave4_bench
TARGET_WAVE5 := $(BUILD_DIR)/run_wave5_bench

.PHONY: all run run_wave2 run_wave3a run_wave3b run_wave4 run_wave5 clean

all: $(TARGET) $(TARGET_WAVE2) $(TARGET_WAVE3A) $(TARGET_WAVE3B) $(TARGET_WAVE4) $(TARGET_WAVE5)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/bench_harness.o: bench_harness.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_tridiag.o: bench_tridiag.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_micro.o: bench_kernels_micro.f90 $(BUILD_DIR)/bench_harness.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_solver.o: bench_kernels_solver.f90 $(BUILD_DIR)/bench_harness.o $(BUILD_DIR)/bench_tridiag.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/run_wave1_bench.o: run_wave1_bench.f90 $(BUILD_DIR)/bench_kernels_micro.o $(BUILD_DIR)/bench_kernels_solver.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_wave2.o: bench_kernels_wave2.f90 $(BUILD_DIR)/bench_harness.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/run_wave2_bench.o: run_wave2_bench.f90 $(BUILD_DIR)/bench_kernels_wave2.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_wave3a.o: bench_kernels_wave3a.f90 $(BUILD_DIR)/bench_harness.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/run_wave3a_bench.o: run_wave3a_bench.f90 $(BUILD_DIR)/bench_kernels_wave3a.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_wave3b.o: bench_kernels_wave3b.f90 $(BUILD_DIR)/bench_harness.o $(BUILD_DIR)/bench_kernels_wave3a.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/run_wave3b_bench.o: run_wave3b_bench.f90 $(BUILD_DIR)/bench_kernels_wave3b.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_wave4.o: bench_kernels_wave4.f90 $(BUILD_DIR)/bench_harness.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/run_wave4_bench.o: run_wave4_bench.f90 $(BUILD_DIR)/bench_kernels_wave4.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/bench_kernels_wave5.o: bench_kernels_wave5.f90 $(BUILD_DIR)/bench_harness.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/run_wave5_bench.o: run_wave5_bench.f90 $(BUILD_DIR)/bench_kernels_wave5.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INC) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

$(TARGET_WAVE2): $(WAVE2_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE2_OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

$(TARGET_WAVE3A): $(WAVE3A_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE3A_OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

$(TARGET_WAVE3B): $(WAVE3B_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE3B_OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

$(TARGET_WAVE4): $(WAVE4_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE4_OBJS) $(MESA_STAR_LIBS) $(STAR_EXT_LIBS) $(BLAS_LAPACK_LIBS) -lstdc++

$(TARGET_WAVE5): $(WAVE5_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE5_OBJS) $(MESA_WAVE5_LIBS) $(STAR_EXT_LIBS) $(BLAS_LAPACK_LIBS) -lstdc++

run: $(TARGET)
	MESA_DIR=$(MESA_DIR) $(TARGET)

run_wave2: $(TARGET_WAVE2)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE2)

run_wave3a: $(TARGET_WAVE3A)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE3A)

run_wave3b: $(TARGET_WAVE3B)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE3B)

run_wave4: $(TARGET_WAVE4)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE4)

run_wave5: $(TARGET_WAVE5)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE5)

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod $(TARGET) $(TARGET_WAVE2) $(TARGET_WAVE3A) $(TARGET_WAVE3B) $(TARGET_WAVE4) $(TARGET_WAVE5)
	if [ -d $(BUILD_DIR) ]; then rmdir $(BUILD_DIR) 2>/dev/null || true; fi
