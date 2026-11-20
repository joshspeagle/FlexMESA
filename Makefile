# SPDX-License-Identifier: LGPL-3.0-or-later
# Reproducible MESA 26.4.1 data bundle and verification targets (#164, N1-10).

PYTHON ?= python3
MESA_DIR ?= $(HOME)/mesa-26.04.1
MESA_ZIP ?= $(HOME)/mesa-26.04.1.zip
OUT_DIR ?= dist/data-v1
GROUND_TRUTH_DIR ?= $(CURDIR)/build/mesa_ground_truth
Z1_SUFFIX ?=

.PHONY: help data-nets data-ionization data-physics data-verify-converters data-test-inputs data-histories data-bundles data-verify verify-data data-clean-cache

help:
	@echo "Reproducible MESA 26.4.1 data pipeline targets:"
	@echo "  make data-nets              MESA_DIR=...   Rebuild crates/flexmesa-net/data/nets.bin"
	@echo "  make data-ionization        MESA_DIR=... [Z1_SUFFIX=| _CO_0 | _CO_1] Rebuild crates/flexmesa-ionization/data/ion_tables_full.{bin,toml}"
	@echo "  make data-physics           MESA_DIR=... OUT_DIR=..."
	@echo "  make data-verify-converters MESA_DIR=...   Verify unpacked physics and ionization converters against pinned SHA-256 manifests"
	@echo "  make data-test-inputs       MESA_DIR=... MESA_ZIP=... OUT_DIR=..."
	@echo "  make data-histories         GROUND_TRUTH_DIR=... OUT_DIR=..."
	@echo "  make data-bundles           Run data-physics + data-test-inputs"
	@echo "  make data-verify            Run flexmesa data verify --include-test-inputs (alias: verify-data)"
	@echo "  make data-clean-cache       Remove runtime derived caches"

data-nets:
	$(PYTHON) scripts/data/build_physics_bundles.py --mesa-dir "$(MESA_DIR)" --nets-only

data-ionization:
	$(PYTHON) scripts/data/build_ion_tables.py --mesa-dir "$(MESA_DIR)" --z1-suffix "$(Z1_SUFFIX)"

data-physics:
	mkdir -p "$(OUT_DIR)"
	$(PYTHON) scripts/data/build_physics_bundles.py --mesa-dir "$(MESA_DIR)" --out-dir "$(OUT_DIR)"

data-verify-converters:
	$(PYTHON) scripts/data/build_ion_tables.py --mesa-dir "$(MESA_DIR)" --z1-suffix "$(Z1_SUFFIX)" --check
	$(PYTHON) scripts/data/build_physics_bundles.py --mesa-dir "$(MESA_DIR)" --verify-unpacked

data-test-inputs:
	mkdir -p "$(OUT_DIR)"
	$(PYTHON) scripts/data/build_mesa_test_inputs.py \
		--mesa-zip "$(MESA_ZIP)" \
		--mesa-dir "$(MESA_DIR)" \
		--out-archive "$(OUT_DIR)/mesa-test-inputs-v2.tar.zst" \
		--manifest-out "$(CURDIR)/src/flexmesa/data/bundle_hashes/mesa_test_inputs.sha256.tsv"

data-histories:
	mkdir -p "$(OUT_DIR)"
	$(PYTHON) scripts/data/build_mesa_histories.py \
		--ground-truth-dir "$(GROUND_TRUTH_DIR)" \
		--staging-dir "$(OUT_DIR)" \
		--repo-root "$(CURDIR)" \
		--data-dir "$$($(PYTHON) -m flexmesa.cli data path)"

data-bundles: data-physics data-test-inputs

data-verify:
	$(PYTHON) -m flexmesa.cli data verify --include-test-inputs

verify-data: data-verify

data-clean-cache:
	$(PYTHON) -m flexmesa.cli data clean-cache --purge-legacy-in-data
