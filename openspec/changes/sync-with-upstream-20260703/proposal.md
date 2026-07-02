## Why

The `pr-ports-20260428` branch is now ~2 months behind upstream `sgl-project/sglang` main (2359 new commits, May 9 → July 3, 2026). An in-memory `git merge-tree --write-tree` simulation shows 21 stops (16 content conflicts + 3 modify/delete + 2 add/add). Investigation during explore revealed that 26 of the 32 custom commits on the branch are now superseded upstream — upstream independently landed equivalent (often more-evolved) versions of the dFlash spec v2 port, the mamba pool sizing work, and the flashinfer preflight probe. Only 6 fork-only commits remain genuinely valuable and must be preserved across the sync. This change performs the sync in a way that drops superseded work, re-applies the 6 surviving fixes against upstream's evolved layout, and captures the decision matrix so future syncs are repeatable.

## What Changes

- **Drop** the dFlash spec v2 port (~15 commits). Upstream has its own evolved version (PRs #23000, #27950, #27959, #29228, #29220, #29678, etc.) with a 1817-line diff in `dflash_worker_v2.py` alone. Take upstream's version wholesale for `dflash_info.py`, `dflash_info_v2.py`, `dflash_utils.py`, `dflash_worker_v2.py`, `models/dflash.py`, `spec_info.py`, `overlap_utils.py`, `schedule_batch.py`, `scheduler.py`, `scheduler_output_processor_mixin.py`, `dflash_worker.py`, and the dFlash test.
- **Drop** the dFlash-specific mamba cleanup (`a722fee6a`). Moot without the dFlash port.
- **Drop** `MambaPoolConfigurator` (`d10e980d5` + tests `625770e2d`). Upstream's evolved `handle_max_mamba_cache` method covers the same unified-memory-sizing concern (explicit size, deterministic sizing when `max_running_requests` known, `disable_radix_cache` branch, ratio fallback with joint solve for intermediate memory, spec intermediate accounting in all 3 branches, validation `RuntimeError`). The fork's only unique contribution is a `max_mamba_cache_size < required` warning, which is not worth maintaining a parallel configurator against 40+ commits of upstream churn.
- **Drop** the flashinfer fusion preflight port (6 commits: `3cce2876e` + 5 follow-ups). Upstream PR #24172 ("Fix flashinfer workspace OOM", Khoa Pham, May 4) introduces `_preflight_check_workspace_memory` with byte-for-byte equivalent logic (same signature, same docstring, same `cuMemCreate` probe + `all_reduce(BAND)` vote, same warning text). Five of the fork's six commits are authored by the same person who landed the upstream PR.
- **Re-apply** GPTQ MoE fixes (3 commits: `92aa83b07`, `232ce582f`, `a8e2d16bb`) against the new `gptq/schemes/` layout introduced by upstream PR #26402 (which deleted `gptq.py` and split it into `gptq/__init__.py` + `gptq/gptq.py` + `gptq/schemes/`). **BREAKING** for any downstream consumer relying on `python/sglang/srt/layers/quantization/gptq.py` as a flat module.
- **Re-apply** NVFP4 Cutlass MoE fixes (3 commits: `b945ddd0a`, `cc15dec3c`, `caf47c085`) — these auto-merged into `compressed_tensors/schemes/w4a4_nvfp4_moe.py`, but verify against the new layout (upstream churned this file).
- **Re-apply** CUDA-graph disable on `--cpu-offload-gb` (`115d00a1e`) against upstream's evolved `server_args.py` (~52 lines churned by PR #29678, 131 commits in 6 weeks of churn).
- **Re-apply** `--disable-cuda-graph-padding × --enable-torch-compile` incompatibility assert (`746ff3c87`, 8-line addition to `server_args.py`).
- **Verify** HF rope params fix (`bb6c36b64`) survived auto-merge into `hf_transformers_patches.py`.
- **Verify** `moe_wna16` GELU activation fix (`cb399b1cf`) survived auto-merge into `moe_wna16.py`.
- **Add** `AGENTS.md` entry documenting the sync decision rationale for future maintainers.

## Capabilities

### New Capabilities
- `upstream-sync-workflow`: The repeatable process for syncing the fork with upstream — decision matrix for keep/drop, merge-vs-rebase choice, conflict resolution strategy, and the phases of the sync. Reusable on future syncs.
- `fork-moe-quant-fixes`: The surviving fork-only MoE quantization behavior that must be preserved across upstream syncs — GELU activation in `MoeWNA16Method`, NVFP4 Cutlass MoE crash fixes for EP>1 (global expert ID remap, input global scales scalar, flashinfer_trtllm crash), GPTQ Marlin MoE bf16 scale dtype, AutoRound GPTQ swapped marlin/non-marlin branches, GPTQ/AWQ MoE crashes with TP>1 and non-auto moe-runner backend.
- `fork-server-args-gates`: The surviving fork-only `server_args.py` gates and asserts that must be preserved across upstream syncs — CUDA-graph auto-disable on `--cpu-offload-gb`, `--disable-cuda-graph-padding × --enable-torch-compile` incompatibility assert, HF rope params injection fix.

### Modified Capabilities
<!-- None — this is a fresh OpenSpec installation with no existing specs to modify. -->

## Impact

- **Code**: 35 files in the fork's diff against the merge-base; after the sync, the surviving fix surface shrinks to ~10 files (3 GPTQ scheme files, 1 NVFP4 scheme file, 1 `server_args.py`, 1 `hf_transformers_patches.py`, 1 `moe_wna16.py`, plus verification-only touchpoints).
- **APIs**: `python/sglang/srt/layers/quantization/gptq.py` deleted upstream — callers must update imports to `python/sglang/srt/layers/quantization/gptq/`. `python/sglang/srt/managers/scheduler_output_processor_mixin.py` deleted upstream — batch-result processing moved to `SchedulerBatchResultProcessor`. `python/sglang/srt/speculative/dflash_worker.py` deleted upstream — V1 worker path removed.
- **Dependencies**: Upstream PR #29678 introduces new `unified_memory_pool.py` (1369 lines) and `multi_ended_allocator.py` (2474 lines) gated behind a new `--enable-unified-memory` flag. Fork adopts these as opt-in; the fork's `MambaPoolConfigurator` legacy path is dropped in favor of upstream's evolved `handle_max_mamba_cache`.
- **Systems**: Each release is Docker-pinned per the user's deployment model — adopting upstream's evolved dFlash spec v2 is a feature change on the new feature branch, not a production regression. Branch: `sync-upstream-20260703` (new feature branch from `origin/pr-ports-20260428`).
- **Tests**: `test/registered/spec/dflash/test_dflash.py` exercises the upstream-evolved dFlash path; `test/registered/unit/mem_cache/test_unified_mamba_views.py` is a new upstream test from PR #29678; `test/registered/unit/model_executor/test_pool_configurator.py` auto-merges but needs re-verification given upstream's 11 commits of churn there.
- **Effort**: ~5-7 hours total (Phase 0 done in explore; Phases 1-6 in apply).
