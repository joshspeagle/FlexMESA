# SPDX-License-Identifier: GPL-3.0-or-later
# Build and run the MESA 26.4.1 thermal neutrino (`neu`) oracle generator (#161).
#
# Prerequisites:
#   - `MESASDK_ROOT` initialized (`source $MESASDK_ROOT/bin/mesasdk_init.sh`)
#   - `MESA_DIR` pointing to a built MESA release 26.4.1 tree

MESA_DIR ?= ../../../../mesa
FC ?= gfortran
FFLAGS ?= -march=x86-64 -mtune=generic -mfpmath=sse -O2 -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check -ffree-line-length-none

ifeq ($(strip $(MESA_DIR)),)
$(error MESA_DIR must be set to a built MESA 26.4.1 checkout)
endif

MESA_INCS := -I$(MESA_DIR)/include -I$(MESA_DIR)/neu/make
ifdef MESASDK_ROOT
SDK_ENV := PATH="$(MESASDK_ROOT)/bin:$(PATH)" MESASDK_ROOT="$(MESASDK_ROOT)"
MESA_INCS += -I$(MESASDK_ROOT)/include
SDK_LIBS := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_crmath_link) $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_crlibm_link)
SDK_VER := $(shell $(SDK_ENV) $(MESASDK_ROOT)/bin/mesasdk_version 2>/dev/null || echo unknown)
FC := $(MESASDK_ROOT)/bin/gfortran
else
SDK_LIBS := $(shell mesasdk_crmath_link) $(shell mesasdk_crlibm_link)
SDK_VER := $(shell mesasdk_version 2>/dev/null || echo unknown)
endif

MESA_LIBS := -L$(MESA_DIR)/lib -lneu -lmath -lutils -lconst $(SDK_LIBS)

BUILD_DIR ?= build
BIN := $(BUILD_DIR)/gen_neu_oracles
OUT ?= mesa_26_4_1_neu_oracles.tsv

.PHONY: all generate run clean

all: $(BIN)

$(BIN): gen_neu_oracles.f90
	@mkdir -p $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INCS) -J$(BUILD_DIR) gen_neu_oracles.f90 $(MESA_LIBS) -o $@

generate: $(BIN)
	MESA_DIR="$(MESA_DIR)" MESASDK_VERSION_STR="$(SDK_VER)" $(BIN) "$(OUT)"

run: generate

clean:
	rm -f $(BIN) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod
	rmdir $(BUILD_DIR) 2>/dev/null || true
