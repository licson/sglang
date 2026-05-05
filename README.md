# licson/sglang — CUDA 13 / Blackwell Docker Build

> **Branch:** `forked-sglang-docker-build`  
> **Target Code:** `pr-ports-20260428`

This branch provides a production-grade, multi-stage Dockerfile for building SGLang from the `pr-ports-20260428` branch. It is specifically tuned for **CUDA 13.0.1**, **Python 3.12**, and **Blackwell (sm_103a)** GPU support, with a number of dependency patches that are not yet fully upstream.

---

## What's in `pr-ports-20260428`?

This branch is an integration branch that merges `upstream/main` with a curated set of upstream PRs (and related fixups) that we need before they land in a stable release.

### Ported Upstream PRs

| PR | Title | Why it's ported |
|----|-------|-----------------|
| **#22611** | [Bugfix] Fix swapped marlin/non-marlin branches in AutoRound GPTQ MoE quantization | Fixes a logic swap that broke AutoRound GPTQ MoE loading; also adds missing `desc_act` to non-marlin config |
| **#23000** | [Feature] Spec V2 DFlash Support | Adds speculative decoding v2 for dflash, decouples next-step planning from host metadata lag, fixes SWA/FlashInfer interaction and rope config for transformers v5 |
| **#23664** | Fix: disable CUDA graph when `--cpu-offload-gb` is set | Prevents runtime crashes caused by CUDA graph capture when CPU offloading is active |
| **#23718** | [Bugfix] Re-export `is_port_available` from `sglang.srt.utils` package init | Restores a missing public API that downstream tools depend on |
| **#23721** | [bug fix] flashinfer fusion: preflight workspace allocation to avoid NCCL hang | Eliminates NCCL hangs in FlashInfer fusion by probing and pre-allocating workspace; includes TP=2 regression test |
| **#23897** | account for mamba state pools in unified memory sizing | Fixes OOM/planning errors for Mamba-based models by including state pools in memory calculations |
| **#23903** | [Bug Fix] Reject incompatible combination of `--disable-cuda-graph-padding` and `--enable-torch-compile` | Blocks a configuration that causes O(max_batch_size) torch.compile stalls instead of fixed bucket compilation |
| **#23910** | fix: remove manual rope parameters injection in `PretrainedConfig` | Fixes transformers v5 compatibility by stopping manual RoPE parameter injection that conflicts with HF's native handling |

### Additional fixups on the branch
Beyond the merged PRs, the branch carries a few directly-committed follow-ups:
- FlashInfer fusion preflight probe lifecycle cleanup (handle-type matching, `MAX_COMM_SIZE` caps, documentation)
- dflash spec v2 test updates and benchmark sweep removal
- Mamba memory calculation cleanup

---

## FlashInfer Patches Applied at Build Time

The Dockerfile pulls **FlashInfer `main`** and merges the following PRs at image-build time to unlock **SM120 (Blackwell)** support:

| PR | Title | Impact |
|----|-------|--------|
| **#3173** | fix: add `sm_121` to TMEM column fallback map | Enables TMEM fallback on `sm_121` variants |
| **#3174** | doc: align user-facing SM120 messages with SM12x dispatch | Cleans up architecture-detection messaging |
| **#3175** | fix: align `is_sm120f_supported` with SM12x family semantics | Fixes SM120 feature-gate logic for the whole SM12x family |
| **#3180** | Fix dense blockscaled SM12x | Repairs dense blockscaled kernels on Blackwell |
| **#3192** | fix cudnn SM120 NaN | Eliminates NaN outputs from cuDNN paths on SM120 |
| **#3193** | perf(moe): optimize SM120 b12x MoE short decode | Speeds up short-sequence MoE decode on Blackwell |

These PRs are fetched via `git fetch origin pull/<id>/head` and merged in the `flashinfer_builder` stage before compilation.

---

## Build Requirements

- Docker with BuildKit enabled (`docker buildx` or `DOCKER_BUILDKIT=1`)
- NVIDIA Container Toolkit (for runtime GPU access)
- ~50 GB+ free disk space (multi-stage builds with CUDA toolkits)
- Network access to GitHub, PyPI, and NVIDIA repositories

---

## Quick Start

### 1. Clone this branch

```bash
git clone -b forked-sglang-docker-build https://github.com/licson/sglang.git
cd sglang
```

### 2. Build the image

```bash
docker build \
  --target runtime \
  -t licson/sglang:pr-ports-20260428-cuda13 \
  .
```

### 3. Run the server

```bash
docker run --gpus all -it --rm \
  -p 30000:30000 \
  licson/sglang:pr-ports-20260428-cuda13 \
  python -m sglang.launch_server \
    --model-path <your-model> \
    --tp 8 \
    --port 30000
```

---

## Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `13.0.1` | CUDA base image tag |
| `BUILD_TYPE` | `all` | SGLang extras to install (e.g. `all`, `srt`) |
| `SGL_KERNEL_VERSION` | `0.4.1.post1` | Pre-built `sglang_kernel` wheel version |
| `DEEPEP_COMMIT` | `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` | DeepEP source commit |
| `MOONCAKE_VERSION` | `0.3.10.post2` | Mooncake transfer engine version |
| `MAX_JOBS` | `8` | Parallel compile jobs for CUDA extensions |
| `FI_PR_NUMBERS` | `3173 3174 3175 3180 3192 3193` | FlashInfer PRs to merge |

### Example: custom build

```bash
docker build \
  --target runtime \
  --build-arg CUDA_VERSION=13.0.1 \
  --build-arg MAX_JOBS=16 \
  -t licson/sglang:custom \
  .
```

---

## Dockerfile Stages

```
base
 ├── torch_deps  ──┬── deepep_builder
 │                 ├── flashinfer_builder
 │                 └── deepgemm_builder
 └── framework ────► runtime (final)
```

| Stage | Purpose |
|-------|---------|
| `base` | Ubuntu 24.04 + CUDA 13 + system deps + Python 3.12 + Rust + uv + GDRCopy |
| `torch_deps` | PyTorch cu130, sgl-kernel wheel, and dependency constraints |
| `deepep_builder` | DeepEP wheel with CCCL & timeout patches |
| `flashinfer_builder` | FlashInfer wheel + cubin wheel with Blackwell PRs merged |
| `deepgemm_builder` | DeepGEMM wheel |
| `framework` | Assembles all wheels, installs SGLang fork, dflash, and latest transformers |
| `runtime` | Slim production image with only runtime libraries and Python packages |

---

## Fork-Specific Patches

Beyond the ported PRs, the Dockerfile itself carries patches needed to make the current dependency stack work on CUDA 13:

| Patch | Description |
|-------|-------------|
| **transformers pin removal** | Replaces `transformers==5.6.0` with unconstrained install so latest HuggingFace `main` can be used |
| **dflash dependency fix** | Installs `z-lab/dflash` without its pinned upstream `sglang` git dependency (editable fork satisfies it) |
| **DeepEP CCCL path** | Injects CUDA 13 `cccl` include path into DeepEP's `setup.py` |
| **DeepEP timeouts** | Increases `NUM_CPU_TIMEOUT_SECS` (100 → 1000) and `NUM_TIMEOUT_CYCLES` (10×) |
| **sgl-kernel robustness** | Cubin download is non-fatal with 3× retry; cache directory is pre-created |
| **constraints.txt hygiene** | Strips stale `-e` and `file://` paths from `constraints.txt` before runtime install |
| **Triton ptxas fix** | Symlinks system CUDA `ptxas` into Triton backend for `sm_103a` support |

---

## Troubleshooting

| Issue | Likely Cause / Fix |
|-------|-------------------|
| `flashinfer` build fails | Check `MAX_JOBS` isn't exhausting host RAM; default `TORCH_CUDA_ARCH_LIST="12.1"` is hard-coded for FlashInfer |
| `sgl-kernel` cubin warnings | Download failure is non-fatal; server will fall back to JIT compilation |
| `apt` held packages in runtime | Already fixed by `--allow-change-held-packages` in the runtime stage |
| DeepEP timeout crashes | Timeouts are already 10× upstream defaults; increase further in `Dockerfile` if needed |

---

## Related Branches

- **Code branch:** [`pr-ports-20260428`](https://github.com/licson/sglang/tree/pr-ports-20260428) — the actual SGLang fork with ported PRs
- **Docker branch:** [`forked-sglang-docker-build`](https://github.com/licson/sglang/tree/forked-sglang-docker-build) — this branch (the Dockerfile only)

---

## License

Same as upstream [sgl-project/sglang](https://github.com/sgl-project/sglang).
