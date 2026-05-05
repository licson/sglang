ARG CUDA_VERSION=13.0.1

# =============================================================================
# Base Stage: CUDA 13.0 + Ubuntu 24.04 + System Dependencies
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/

# GKE default paths
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install system dependencies, native Python 3.12, and bootstrap pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Python 3.12 (native to Ubuntu 24.04)
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    patchelf \
    git-lfs \
    # MPI & NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # VLM / multimodal
    ffmpeg \
    # InfiniBand / RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    linux-libc-dev \
    # Packaging tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    gnupg2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Locale setup
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Rust toolchain (required by SGLang build)
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# GDRCopy (for GPUDirect RDMA)
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so


# =============================================================================
# Torch Deps Stage: PyTorch cu130 + sgl-kernel + Constraints
# =============================================================================
FROM base AS torch_deps

ARG CUDA_VERSION
ARG BUILD_TYPE=all
ARG SGL_KERNEL_VERSION=0.4.1.post1

WORKDIR /sgl-workspace

# Pre-built sgl-kernel wheel (CUDA 13, abi3 compatible with Python 3.12)
RUN uv pip install --system --python python3.12 --break-system-packages --force-reinstall --no-deps \
    "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl"

# PyTorch ecosystem (cu130)
RUN uv pip install --system --python python3.12 --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio ninja wheel packaging

# Shallow-clone the fork to install its Python dependencies and freeze constraints.
# This layer is invalidated only when pyproject.toml changes upstream.
RUN git clone --depth=1 -b pr-ports-20260428 https://github.com/licson/sglang.git /tmp/sglang \
    && cd /tmp/sglang/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py \
    && echo '__version__ = "0.0.0"' > sglang/version.py \
    && touch README.md \
    && touch LICENSE \
    && sed -i 's/transformers==5\.6\.0/transformers/g' pyproject.toml \
    && uv pip install --system --python python3.12 --break-system-packages ".[${BUILD_TYPE}]" \
    && uv pip freeze --system > /sgl-workspace/constraints.txt \
    && sed -i '/^sglang==/d' /sgl-workspace/constraints.txt \
    && sed -i '/^-e /d' /sgl-workspace/constraints.txt \
    && sed -i '/ @ file:\/\//d' /sgl-workspace/constraints.txt \
    && cd / && rm -rf /tmp/sglang


# =============================================================================
# DeepEP Builder Stage
# =============================================================================
FROM torch_deps AS deepep_builder

ARG CUDA_VERSION
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG MAX_JOBS=8

WORKDIR /build

RUN curl --retry 3 --retry-delay 2 -fsSL -o ${DEEPEP_COMMIT}.zip \
        https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip \
    && unzip -q ${DEEPEP_COMMIT}.zip && rm ${DEEPEP_COMMIT}.zip \
    && mv DeepEP-${DEEPEP_COMMIT} DeepEP && cd DeepEP \
    && sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh \
    && sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh

RUN cd /build/DeepEP \
    && if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi \
    && case "$CUDA_VERSION" in \
        12.6.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' ;; \
        12.8.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0' ;; \
        12.9.1|13.0.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} \
        python3 setup.py bdist_wheel -d /wheels


# =============================================================================
# FlashInfer Builder Stage (with Blackwell PR patches)
# =============================================================================
FROM torch_deps AS flashinfer_builder

ARG MAX_JOBS=8
ARG FI_PR_NUMBERS="3173 3174 3175 3180 3192 3193"

WORKDIR /build

RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /build/flashinfer
WORKDIR /build/flashinfer

RUN git config --global user.email "builder@nube.local" \
    && git config --global user.name "Docker Builder"

RUN set -e; \
    for PR in ${FI_PR_NUMBERS}; do \
        echo "Fetching and merging FlashInfer PR #${PR}..." && \
        git fetch origin pull/${PR}/head:pr-${PR} && \
        git merge --no-edit pr-${PR}; \
    done

ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV MAX_JOBS=${MAX_JOBS}

RUN python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels

# Build flashinfer-cubin wheel (requires main flashinfer package installed)
RUN uv pip install --system --python python3.12 --break-system-packages --no-deps /wheels/flashinfer*.whl \
    && uv pip install --system --python python3.12 --break-system-packages build \
    && cd /build/flashinfer/flashinfer-cubin \
    && python3 -m build --no-isolation --wheel \
    && cp dist/*.whl /wheels/


# =============================================================================
# DeepGEMM Builder Stage
# =============================================================================
FROM torch_deps AS deepgemm_builder

WORKDIR /build

RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /build/DeepGEMM
WORKDIR /build/DeepGEMM

RUN python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels


# =============================================================================
# Framework Stage: Assemble everything + SGLang + dflash + transformers
# =============================================================================
FROM torch_deps AS framework

ARG BUILD_TYPE
ARG CUDA_VERSION
ARG MOONCAKE_VERSION=0.3.10.post2
ARG MAX_JOBS=8

WORKDIR /sgl-workspace

# Copy built wheels from parallel stages
COPY --from=deepep_builder /wheels /tmp/wheels
COPY --from=flashinfer_builder /wheels /tmp/wheels
COPY --from=deepgemm_builder /wheels /tmp/wheels

# Install Mooncake (PD disaggregation support, CUDA 13 variant)
RUN uv pip install --system --python python3.12 --break-system-packages mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}

# Install builder wheels (no constraints needed - we built them ourselves)
RUN uv pip install --system --python python3.12 --break-system-packages \
    /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Fix Triton to use system ptxas for Blackwell (sm_103a) support (CUDA 13+)
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ -d /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin ]; then \
        rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
        ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas; \
    fi

# Clone and install the SGLang fork
RUN git clone --depth=1 -b pr-ports-20260428 https://github.com/licson/sglang.git /sgl-workspace/sglang

RUN cd /sgl-workspace/sglang \
    && sed -i 's/transformers==5\.6\.0/transformers/g' python/pyproject.toml \
    && uv pip install --system --python python3.12 --break-system-packages --no-deps -e "python[${BUILD_TYPE}]" \
    && kernels lock python \
    && ( success=0; for i in 1 2 3; do \
            echo "Attempt $i/3: downloading sgl-kernel cubins..." && \
            kernels download python && success=1 && break; \
            echo "sgl-kernel cubin download failed, retrying in 30s..." && sleep 30; \
        done; [ "$success" = "1" ] || echo "Warning: sgl-kernel cubin download failed, continuing without cubins" ) \
    && mkdir -p /root/.cache/sglang \
    && if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi \
    && find /usr/local/lib/python3.12/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove stale editable / local-path references from constraints before using them
RUN sed -i '/^-e /d' /sgl-workspace/constraints.txt \
    && sed -i '/ @ file:\/\//d' /sgl-workspace/constraints.txt

# Install dflash (without its pinned sglang git dep — our editable fork already satisfies it)
RUN git clone https://github.com/z-lab/dflash.git /workspace/dflash \
    && uv pip install --system --python python3.12 --break-system-packages --no-deps \
        -e "/workspace/dflash"

# Install latest transformers from HuggingFace (unconstrained — user wants latest git)
RUN uv pip install --system --python python3.12 --break-system-packages \
    git+https://github.com/huggingface/transformers.git

WORKDIR /sgl-workspace/sglang


# =============================================================================
# Runtime Stage: Production image (no dev tools, smaller footprint)
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ARG CUDA_VERSION
ARG GDRCOPY_VERSION=2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/

ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install runtime dependencies only (no build tools, no devtools)
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    curl \
    wget \
    git \
    locales \
    # Python runtime
    python3.12-full \
    python3.12-dev \
    # MPI / NUMA runtime
    libopenmpi3 \
    libnuma1 \
    # IB / RDMA runtime
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    ibverbs-providers \
    rdma-core \
    infiniband-diags \
    perftest \
    # Other runtime libraries
    libgoogle-glog0v6t64 \
    libunwind8 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libboost-filesystem1.83.0 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.1.0 \
    libcurl4 \
    libczmq4 \
    libfabric1 \
    libssl3 \
    # JIT compilation support
    ninja-build \
    libnccl2 \
    libnccl-dev \
    linux-libc-dev \
    gnupg2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Copy Python packages from framework
COPY --from=framework /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Copy SGLang workspace and other source checkouts
COPY --from=framework /sgl-workspace /sgl-workspace
COPY --from=framework /workspace /workspace

# Copy caches
COPY --from=framework /root/.cache/sglang /root/.cache/sglang
COPY --from=framework /root/.cache/huggingface /root/.cache/huggingface

# Copy GDRCopy runtime artifacts
COPY --from=framework /usr/lib/libgdrapi.so* /usr/lib/
COPY --from=framework /usr/bin/gdrcopy_* /usr/bin/
COPY --from=framework /usr/src/gdrdrv-${GDRCOPY_VERSION} /usr/src/gdrdrv-${GDRCOPY_VERSION}

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Fix Triton ptxas for Blackwell
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ -d /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin ]; then \
        rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
        ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas; \
    fi

WORKDIR /sgl-workspace/sglang

CMD ["python", "-m", "sglang.launch_server", "--help"]
