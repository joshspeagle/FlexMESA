#!/usr/bin/env bash
# Shared environment initialization for MESA 26.4.1 + MESA SDK 26.6.1.
export PATH="${MESA_SYSDEPS_DIR:-$HOME/mesasdk-sysdeps}/bin:${MESA_SYSDEPS_DIR:-$HOME/mesasdk-sysdeps}/usr/bin:$PATH"
export LIBRARY_PATH="${MESA_SYSDEPS_DIR:-$HOME/mesasdk-sysdeps}/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="${MESA_SYSDEPS_DIR:-$HOME/mesasdk-sysdeps}/usr/include:${MESA_SYSDEPS_DIR:-$HOME/mesasdk-sysdeps}/usr/include/x86_64-linux-gnu${CPATH:+:$CPATH}"
export MESASDK_ROOT="${MESASDK_ROOT:-$HOME/mesasdk-26.6.1}"
set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u
export MESA_DIR="$HOME/mesa-26.04.1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
unset DISPLAY || true
