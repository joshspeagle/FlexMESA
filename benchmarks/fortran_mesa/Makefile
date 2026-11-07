# SPDX-License-Identifier: LGPL-3.0-or-later
# Portable Makefile for the Wave 1 four-stage Fortran MESA benchmark suite.
#
# Usage:
#   make MESA_DIR=/path/to/mesa run

MESA_DIR ?= ../../../mesa
ifeq ($(origin FC), default)
FC := gfortran
endif
FC ?= gfortran
BUILD_DIR ?= build
FFLAGS ?= -O2 -fopenmp -cpp -ffree-form -ffree-line-length-none

MESA_INC := -I$(MESA_DIR)/include -I$(MESA_DIR)/build/interp_2d/modules
MESA_LIBS := \
	$(MESA_DIR)/lib/libeos.a \
	$(MESA_DIR)/lib/libcolors.a \
	$(MESA_DIR)/lib/libionization.a \
	$(MESA_DIR)/lib/libturb.a \
	$(MESA_DIR)/lib/libneu.a \
	$(MESA_DIR)/lib/libinterp_2d.a \
	$(MESA_DIR)/lib/libnum.a \
	$(MESA_DIR)/lib/libmtx.a \
	$(MESA_DIR)/lib/libinterp_1d.a \
	$(MESA_DIR)/lib/libauto_diff.a \
	$(MESA_DIR)/lib/libchem.a \
	$(MESA_DIR)/lib/libmath.a \
	$(MESA_DIR)/lib/libutils.a \
	$(MESA_DIR)/lib/libconst.a

BLAS_LAPACK_LIBS ?= $(shell pkg-config --libs lapack blas 2>/dev/null || echo "-llapack -lblas")

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

TARGET := $(BUILD_DIR)/run_wave1_bench
TARGET_WAVE2 := $(BUILD_DIR)/run_wave2_bench
TARGET_WAVE3A := $(BUILD_DIR)/run_wave3a_bench

.PHONY: all run run_wave2 run_wave3a clean

all: $(TARGET) $(TARGET_WAVE2) $(TARGET_WAVE3A)

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

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

$(TARGET_WAVE2): $(WAVE2_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE2_OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

$(TARGET_WAVE3A): $(WAVE3A_OBJS)
	$(FC) $(FFLAGS) -o $@ $(WAVE3A_OBJS) $(MESA_LIBS) $(BLAS_LAPACK_LIBS)

run: $(TARGET)
	MESA_DIR=$(MESA_DIR) $(TARGET)

run_wave2: $(TARGET_WAVE2)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE2)

run_wave3a: $(TARGET_WAVE3A)
	MESA_DIR=$(MESA_DIR) $(TARGET_WAVE3A)

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod $(TARGET) $(TARGET_WAVE2) $(TARGET_WAVE3A)
	if [ -d $(BUILD_DIR) ]; then rmdir $(BUILD_DIR) 2>/dev/null || true; fi
