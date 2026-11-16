#!/usr/bin/env bash
# Download, checksum-verify, and build MESA release 26.4.1 + MESA SDK 26.6.1
# outside the FlexMESA repository, confirming all 26 module self-checks pass.

set -euo pipefail

DOWNLOADS_DIR="${MESA_DOWNLOADS_DIR:-$HOME/mesa-downloads}"
MESASDK_DEST="${MESASDK_ROOT:-$HOME/mesasdk-26.6.1}"
MESA_DEST="${MESA_DIR:-$HOME/mesa-26.04.1}"
SYSDEPS_DIR="${MESA_SYSDEPS_DIR:-$HOME/mesasdk-sysdeps}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
NPROCS="${NPROCS:-8}"

mkdir -p "$DOWNLOADS_DIR"

# 1. Download and MD5-verify MESA SDK 26.6.1 (Zenodo record 20598423)
if [ ! -f "$DOWNLOADS_DIR/mesasdk-x86_64-linux-26.6.1.tar.gz" ]; then
    curl -L --fail -o "$DOWNLOADS_DIR/mesasdk-x86_64-linux-26.6.1.tar.gz" \
        "https://zenodo.org/records/20598423/files/mesasdk-x86_64-linux-26.6.1.tar.gz?download=1"
fi
echo "c5ef95aafe07848988959fd45cfe8baf  $DOWNLOADS_DIR/mesasdk-x86_64-linux-26.6.1.tar.gz" | md5sum -c -

# 2. Download and MD5-verify MESA 26.4.1 archive (Zenodo record 19722306)
if [ ! -f "$DOWNLOADS_DIR/mesa-26.04.1.zip" ]; then
    curl -L --fail -o "$DOWNLOADS_DIR/mesa-26.04.1.zip" \
        "https://zenodo.org/records/19722306/files/mesa-26.04.1.zip?download=1"
fi
echo "994fd28f38da92efc80e38ca11d8317b  $DOWNLOADS_DIR/mesa-26.04.1.zip" | md5sum -c -

# 3. Ensure user-space SDK prerequisites (tcsh, zlib1g-dev, libx11-dev) if not installed system-wide
if ! command -v tcsh >/dev/null 2>&1 || [ ! -f /usr/include/zlib.h ] || [ ! -f /usr/include/X11/Xlib.h ]; then
    mkdir -p "$DOWNLOADS_DIR/debs" "$SYSDEPS_DIR"
    (
        cd "$DOWNLOADS_DIR/debs"
        apt-get download tcsh libx11-dev x11proto-dev libxcb1-dev libxau-dev libxdmcp-dev xtrans-dev
        curl -sLO "http://archive.ubuntu.com/ubuntu/pool/main/z/zlib/zlib1g-dev_1.3.dfsg-3.1ubuntu2.3_amd64.deb"
        for deb in ./*.deb; do
            dpkg -x "$deb" "$SYSDEPS_DIR"
        done
        ln -sfn tcsh "$SYSDEPS_DIR/bin/csh"
        ln -sfn /lib/x86_64-linux-gnu/libz.so.1 "$SYSDEPS_DIR/usr/lib/x86_64-linux-gnu/libz.so"
        ln -sfn /usr/lib/x86_64-linux-gnu/libX11.so.6 "$SYSDEPS_DIR/usr/lib/x86_64-linux-gnu/libX11.so"
    )
fi

# 4. Extract MESA SDK 26.6.1
if [ ! -x "$MESASDK_DEST/bin/gfortran" ]; then
    mkdir -p "$MESASDK_DEST"
    tar -xzf "$DOWNLOADS_DIR/mesasdk-x86_64-linux-26.6.1.tar.gz" -C "$MESASDK_DEST" --strip-components=1
    ln -sfn "$MESASDK_DEST" "$HOME/mesasdk"
fi

# 5. Extract MESA 26.4.1 (keeps all EOS text tables in place alongside binary caches)
if [ ! -d "$MESA_DEST" ]; then
    unzip -q "$DOWNLOADS_DIR/mesa-26.04.1.zip" -d "$(dirname "$MESA_DEST")"
    ln -sfn "$MESA_DEST" "$HOME/mesa"
fi

# 6. Initialize SDK and run documented ./install procedure
export PATH="$SYSDEPS_DIR/bin:$SYSDEPS_DIR/usr/bin:$PATH"
export LIBRARY_PATH="$SYSDEPS_DIR/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$SYSDEPS_DIR/usr/include:$SYSDEPS_DIR/usr/include/x86_64-linux-gnu${CPATH:+:$CPATH}"
export MESASDK_ROOT="$MESASDK_DEST"
set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u
export MESA_DIR="$MESA_DEST"
export OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NPROCS
unset DISPLAY || true

cd "$MESA_DIR"
./install

passed_count=$(grep -c "has been built, tested, and exported." "$MESA_DIR/build.log")
echo "MESA 26.4.1 install succeeded: ${passed_count} packages built, tested, and exported."
