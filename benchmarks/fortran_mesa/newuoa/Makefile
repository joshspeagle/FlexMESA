# SPDX-License-Identifier: GPL-3.0-or-later
# NEWUOA oracle (package G2): driver linked against MESA 26.4.1's built libnum.
#
#   source $MESASDK_ROOT/bin/mesasdk_init.sh
#   make MESA_DIR=<built MESA 26.4.1> [BUILD=<scratch dir>]       # reference runs
#   make MESA_DIR=<built MESA 26.4.1> BUILD=<scratch dir> coverage # gcov of mod_newuoa.f
#
# The reference build links the pristine $(MESA_DIR)/lib/libnum.a. The coverage
# build compiles a private copy of $(MESA_DIR)/num/private/mod_newuoa.f with the
# flags num/make/makefile_base uses for it (COMPILE_XTRA) plus --coverage, links
# it ahead of libnum.a, and checks that its CALFUN records equal the reference.
# Nothing under $(MESA_DIR) is written.

MESA_DIR ?= $(HOME)/mesa-26.04.1
BUILD ?= build
include $(MESA_DIR)/utils/makefile_header

RUNS = chebyquad_n2 chebyquad_n4 chebyquad_n6 chebyquad_n4_npt6 quad_rhoend1 quad_rhoend1em6
DRIVER_FLAGS = $(FCbasic) $(FCopt) $(FCfree) -ffp-contract=off -I$(MESA_INCLUDE_DIR)
LIBS = -fopenmp -L$(MESA_LIB_DIR) -lnum $(LOAD_MATRIX) -lauto_diff $(LD_FLAGS)

.PHONY: all runs coverage clean
all: runs

$(BUILD)/newuoa_oracle: newuoa_oracle.f90
	mkdir -p $(BUILD)
	$(FC) $(DRIVER_FLAGS) -J$(BUILD) -o $@ newuoa_oracle.f90 $(LIBS)

$(BUILD)/bigden_oracle: bigden_oracle.f90
	mkdir -p $(BUILD)/bigden
	$(FC) $(DRIVER_FLAGS) -I$(MESA_DIR)/num/make -J$(BUILD)/bigden -o $@ bigden_oracle.f90 $(LIBS)

runs: $(BUILD)/newuoa_oracle $(BUILD)/bigden_oracle
	for r in $(RUNS); do \
	   $(BUILD)/newuoa_oracle $$r mesa_26_4_1_newuoa_$$r.hex $(MESA_DIR)/data/version_number || exit 1; \
	done
	$(BUILD)/bigden_oracle mesa_26_4_1_newuoa_bigden.hex $(MESA_DIR)/data/version_number

# Coverage build: same drivers, instrumented copy of mod_newuoa.f. Each run's
# CALFUN records must equal the reference; gcov counts are reported per run.
$(BUILD)/cov/mod_newuoa.o: $(MESA_DIR)/num/private/mod_newuoa.f
	mkdir -p $(BUILD)/cov
	cp $(MESA_DIR)/num/private/mod_newuoa.f $(BUILD)/cov/mod_newuoa.f
	cd $(BUILD)/cov && $(FC) $(COMPILE_BASIC_FLAGS) $(FCopt) $(FCfixed) \
	   -I$(MESA_DIR)/num/public -I$(MESA_INCLUDE_DIR) --coverage -c mod_newuoa.f

$(BUILD)/cov/newuoa_oracle_cov: newuoa_oracle.f90 $(BUILD)/cov/mod_newuoa.o
	$(FC) $(DRIVER_FLAGS) -J$(BUILD)/cov -o $@ newuoa_oracle.f90 -x none $(BUILD)/cov/mod_newuoa.o \
	   $(LIBS) --coverage

$(BUILD)/cov/bigden_oracle_cov: bigden_oracle.f90 $(BUILD)/cov/mod_newuoa.o
	mkdir -p $(BUILD)/cov/bigden
	$(FC) $(DRIVER_FLAGS) -I$(BUILD)/cov -J$(BUILD)/cov/bigden -o $@ bigden_oracle.f90 -x none \
	   $(BUILD)/cov/mod_newuoa.o $(LIBS) --coverage

GCOV_OUT = $(CURDIR)/mesa_26_4_1_newuoa_gcov.txt
coverage: $(BUILD)/cov/newuoa_oracle_cov $(BUILD)/cov/bigden_oracle_cov
	echo "# gcov of MESA r26.4.1 num/private/mod_newuoa.f (copy compiled with num/make COMPILE_XTRA flags plus --coverage)" > $(GCOV_OUT)
	echo "# compiler: `$(FC) --version | head -1`; MESA SDK: $$MESASDK_VERSION" >> $(GCOV_OUT)
	echo "# per run: routine, lines executed (counters reset before each run)" >> $(GCOV_OUT)
	cd $(BUILD)/cov && for r in $(RUNS) bigden; do \
	   python3 -c "import pathlib; [p.unlink() for p in pathlib.Path('.').glob('*.gcda')]"; \
	   if [ $$r = bigden ]; then \
	      ./bigden_oracle_cov cov_$$r.hex $(MESA_DIR)/data/version_number || exit 1; \
	      cmp cov_$$r.hex $(CURDIR)/mesa_26_4_1_newuoa_bigden.hex || exit 1; \
	   else \
	      ./newuoa_oracle_cov $$r cov_$$r.hex $(MESA_DIR)/data/version_number || exit 1; \
	      cmp cov_$$r.hex $(CURDIR)/mesa_26_4_1_newuoa_$$r.hex || exit 1; \
	   fi; \
	   gcov -f -n mod_newuoa.f | grep -A1 --no-group-separator '^Function' | paste - - | \
	      sed -e "s/^Function '__mod_newuoa_MOD_\([a-z_]*\)'[[:space:]]*Lines executed:/$$r \1 /" >> $(GCOV_OUT); \
	done
	cat $(GCOV_OUT)

clean:
	-rm -f $(BUILD)/newuoa_oracle $(BUILD)/*.mod
