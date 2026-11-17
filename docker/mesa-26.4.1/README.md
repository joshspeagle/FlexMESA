# MESA 26.4.1 + MESA SDK 26.6.1 Docker Recipe

This container builds a clean, reproducible upstream MESA `26.4.1` installation paired with MESA SDK `26.6.1` (`crmath` backend and `USE_PGSTAR = NO`). It is intended for running MESA Fortran reference benchmarks, module hex comparisons, and MESA-to-MESA compiler spread checks outside a local workstation setup.

## Pinned Inputs

- **Base image**: `ubuntu:24.04` (`linux/amd64`; MESA SDK `26.6.1` targets `x86_64` Linux)
- **MESA SDK**: `26.6.1` (Zenodo record `20598423`, MD5 `c5ef95aafe07848988959fd45cfe8baf`)
- **MESA release**: `26.4.1` (`mesa-26.04.1.zip`, Zenodo record `19722306`, MD5 `994fd28f38da92efc80e38ca11d8317b`)
- **Build configuration**: non-root `mesa` user, `MESASDK_MATH_SLOT=crmath`, and `USE_PGSTAR = NO` in `utils/makefile_header`

## Building the Image

Allocate at least `35 GB` of free Docker storage before building because the compiled MESA installation, module test binaries, and data tables stay unpacked under `/opt/mesa` (`MESA_DIR`).

```bash
docker build \
  --platform=linux/amd64 \
  --build-arg BUILD_NPROCS=4 \
  -t flexmesa/mesa:26.4.1 \
  docker/mesa-26.4.1
```

## Running Verification Inside the Container

The container entrypoint initializes MESA SDK (`mesasdk_init.sh`) and exports `MESA_DIR=/opt/mesa` and `MESASDK_ROOT=/opt/mesasdk`. You can run an unmodified upstream `test_suite` case or measure compiler/flag spread (`-O2` vs `-O0`) across MESA `26.4.1` builds (alignment of `benchmarks/fortran_mesa/Makefile` with MESA `26.4.1` is tracked under `#146`):

```bash
docker run --rm -it \
  --platform=linux/amd64 \
  -v "$(pwd):/work/FlexMESA" \
  -w /work/FlexMESA \
  flexmesa/mesa:26.4.1 \
  python3 scripts/mesa_reference/measure_mesa_spread.py \
    --mesa-dir /opt/mesa \
    --mesasdk-root /opt/mesasdk \
    --alt-mesa-dir /work/mesa-alt
```

You can also run a standalone MESA test-suite case inside an ephemeral copy of `/opt/mesa/star/test_suite/1.3M_ms_high_Z`:

```bash
docker run --rm --platform=linux/amd64 flexmesa/mesa:26.4.1 \
  bash -c 'mkdir -p /work/run && cp -r "$MESA_DIR/star/test_suite/1.3M_ms_high_Z" /work/run/ && cd /work/run/1.3M_ms_high_Z && make MESA_DIR="$MESA_DIR" WITH_PGSTAR=no && ./rn'
```

You can also inspect `build.log` inside the image to verify that all 25 tested MESA modules compiled, tested, and exported cleanly:

```bash
docker run --rm --platform=linux/amd64 flexmesa/mesa:26.4.1 \
  grep -c "has been built, tested, and exported." /opt/mesa/build.log
```
