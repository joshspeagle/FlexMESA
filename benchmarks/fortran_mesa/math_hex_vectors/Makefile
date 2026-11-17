# SPDX-License-Identifier: LGPL-3.0-or-later
# Build and run the MESA 26.4.1 hexadecimal math reference-vector generator (#165).
#
# Prerequisites:
#   - `MESASDK_ROOT` initialized (`source $MESASDK_ROOT/bin/mesasdk_init.sh`)
#   - `MESA_DIR` pointing to a built MESA release 26.4.1 tree

FC = gfortran
FFLAGS ?= -march=x86-64 -mtune=generic -mfpmath=sse -O2 -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check -std=f2008

ifeq ($(strip $(MESA_DIR)),)
$(error MESA_DIR must be set to a built MESA 26.4.1 checkout)
endif

MESA_INCS := -I$(MESA_DIR)/include -I$(MESASDK_ROOT)/include
MESA_LIBS := -L$(MESA_DIR)/lib -lmath -lutils -lconst $(shell mesasdk_crmath_link) $(shell mesasdk_crlibm_link)
SDK_VER := $(shell mesasdk_version 2>/dev/null || echo unknown)

BUILD_DIR ?= build
BIN := $(BUILD_DIR)/gen_math_vectors
OUT ?= mesa_26_4_1_math_vectors.txt

.PHONY: all generate clean

all: $(BIN)

$(BIN): vec_gen_support.f90 gen_math_vectors.f90
	@mkdir -p $(BUILD_DIR)
	$(FC) $(FFLAGS) $(MESA_INCS) -J$(BUILD_DIR) vec_gen_support.f90 gen_math_vectors.f90 $(MESA_LIBS) -o $@

generate: $(BIN)
	MESA_DIR="$(MESA_DIR)" MESASDK_VERSION_STR="$(SDK_VER)" $(BIN) "$(OUT)"

clean:
	rm -f $(BIN) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod
	rmdir $(BUILD_DIR) 2>/dev/null || true
