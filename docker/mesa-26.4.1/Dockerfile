# Reproducible MESA 26.4.1 + MESA SDK 26.6.1 reference container (USE_PGSTAR = NO).
# Build: docker build --platform=linux/amd64 -t flexmesa/mesa:26.4.1 docker/mesa-26.4.1

FROM --platform=linux/amd64 ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MESA_SDK_VERSION=26.6.1
ARG MESA_SDK_MD5=c5ef95aafe07848988959fd45cfe8baf
ARG MESA_VERSION=26.04.1
ARG MESA_MD5=994fd28f38da92efc80e38ca11d8317b
ARG BUILD_NPROCS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        bzip2 \
        ca-certificates \
        curl \
        file \
        make \
        perl \
        procps \
        python3 \
        tcsh \
        unzip \
        xz-utils \
        libx11-dev \
        zlib1g-dev \
        zstd \
    && if [ ! -e /bin/csh ]; then ln -s /usr/bin/tcsh /bin/csh; fi \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

ENV MESASDK_ROOT=/opt/mesasdk \
    MESA_DIR=/opt/mesa \
    MESASDK_MATH_SLOT=crmath \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN useradd --home-dir /opt/mesa-home --create-home --shell /bin/bash mesa \
    && mkdir -p "${MESASDK_ROOT}" "${MESA_DIR}" /opt/mesa-home \
    && chown -R mesa:mesa "${MESASDK_ROOT}" "${MESA_DIR}" /opt/mesa-home

USER mesa
WORKDIR /opt/mesa-home

RUN curl -fsSL -o /opt/mesa-home/mesasdk.tar.gz \
        "https://zenodo.org/records/20598423/files/mesasdk-x86_64-linux-${MESA_SDK_VERSION}.tar.gz?download=1" \
    && echo "${MESA_SDK_MD5}  /opt/mesa-home/mesasdk.tar.gz" | md5sum -c - \
    && tar -xzf /opt/mesa-home/mesasdk.tar.gz -C "${MESASDK_ROOT}" --strip-components=1 \
    && rm -f /opt/mesa-home/mesasdk.tar.gz \
    && curl -fsSL -o /opt/mesa-home/mesa.zip \
        "https://zenodo.org/records/19722306/files/mesa-${MESA_VERSION}.zip?download=1" \
    && echo "${MESA_MD5}  /opt/mesa-home/mesa.zip" | md5sum -c - \
    && unzip -q /opt/mesa-home/mesa.zip -d "${MESA_DIR}" \
    && find "${MESA_DIR}/mesa-${MESA_VERSION}" -mindepth 1 -maxdepth 1 -exec mv -t "${MESA_DIR}" {} + \
    && rmdir "${MESA_DIR}/mesa-${MESA_VERSION}" \
    && rm -f /opt/mesa-home/mesa.zip

RUN bash -ec '\
    set +u; \
    source "${MESASDK_ROOT}/bin/mesasdk_init.sh"; \
    set -u; \
    export NPROCS="${BUILD_NPROCS}"; \
    unset DISPLAY || true; \
    cd "${MESA_DIR}"; \
    sed -i "s/^USE_PGSTAR = YES/USE_PGSTAR = NO/" utils/makefile_header; \
    ./install; \
    built_count="$(grep -c "has been built, tested, and exported." build.log)"; \
    echo "Built and exported ${built_count} MESA packages"; \
    test "${built_count}" -ge 25'

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set +u' \
    'source "${MESASDK_ROOT}/bin/mesasdk_init.sh"' \
    'set -u' \
    'unset DISPLAY || true' \
    'exec "$@"' \
    > /opt/mesa-home/entrypoint.sh \
    && chmod +x /opt/mesa-home/entrypoint.sh

ENTRYPOINT ["/opt/mesa-home/entrypoint.sh"]
CMD ["bash"]
