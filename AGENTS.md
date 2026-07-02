# AGENTS.md

Notes for AI agents and maintainers working on this fork of `sgl-project/sglang`.

## Upstream sync 2026-07-03

**Upstream commit synced to:** `cba3801f5214a6423561bd1727a0c65ddcf12437` (`upstream/main` as of 2026-07-02 09:57:38 -0700, by `zijiexia`, "docs: add PD disaggregation to GLM-5.2 cookbook playground (#29544)").

**Sync branch:** `sync-upstream-20260703` (Docker-pinned feature branch from `origin/pr-ports-20260428` HEAD `cb399b1cf`). The pre-sync branch `origin/pr-ports-20260428` is preserved unchanged as a rollback point.

**Sync method:** single merge commit `5dc3b60ad` (`git merge upstream/main --no-ff --no-commit` + bulk `git checkout upstream/main --` resolution for 19 conflicts: 14 content + 3 modify/delete + 2 add/add), followed by 6 re-application commits for surviving fixes.

### Dropped fork-only commits (superseded by upstream)

| Fork commit(s) | Fork intent | Superseded by |
| --- | --- | --- |
| dFlash spec v2 port (~15 commits, `70bdac0da` and follow-ups) | Port upstream's dFlash spec v2 (worker, info, utils) to the fork | Upstream PRs #23000 (dFlash), #27950, #27959, #29228, #29220, #29678 (dFlash evolution + unified-memory pool). Upstream's `dflash_info_v2.py` / `dflash_worker_v2.py` are 1754 / 286 lines vs the fork's 628 / 351. |
| dFlash mamba cleanup (`a722fee6a`) | Small mamba-touchpoint cleanup related to the dFlash port | Moot without the dFlash port. Upstream's evolved mamba paths cover the same concern. |
| `MambaPoolConfigurator._reserve_mamba_bytes` (`d10e980d5` + tests `625770e2d`) | Fork-only configurator for explicit `max_mamba_cache_size` with DP-size adjustment, deterministic sizing when `max_running_requests` known, ratio fallback, `RuntimeError` on insufficient memory | Upstream's evolved `handle_max_mamba_cache` method (in `pool_configurator.py`) covers the same concerns, plus: explicit spec-intermediate accounting in all 3 branches, separate `disable_radix_cache + max_running_requests` branch, joint solve for intermediate memory. Fork's only unique contribution was a `"max_mamba_cache_size < required"` warning — not worth maintaining a parallel configurator against 40+ commits of upstream churn on `memory_pool.py`. |
| flashinfer preflight port (6 commits: `3cce2876e` + follow-ups `cd4206197`, `d1a6b3a60`, `50c777bcf`, `df4eca1e5`, `1bf6f9977`) | `_preflight_check_workspace_memory` — collectively decide whether to enter flashinfer workspace creation, avoiding cross-rank desync inside flashinfer collective | Upstream PR #24172 ("Fix flashinfer workspace OOM", Khoa Pham, merged May 4 2026). Byte-for-byte equivalent: same signature `def _preflight_check_workspace_memory(world_size, max_token_num, hidden_dim, dtype, cpu_group=None) -> bool`, same docstring, same `_make_flashinfer_workspace_allocation_prop` → `_flashinfer_trtllm_workspace_allocation_sizes` → `_probe_cumem_create_sequence` logic, same `dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)` vote, same warning text. Five of the fork's six commits are authored by the same Khoa Pham who landed the upstream PR. |

### Re-applied surviving fixes

The 6 surviving fork-only fixes were re-applied against upstream's evolved layout (or verified preserved through auto-merge). Each was committed with attribution to the original fork commit hash for audit-trail purposes.

| Surviving fix | Original fork commit | New location after sync | Re-application commit |
| --- | --- | --- | --- |
| GPTQ Marlin MoE bf16 scale dtype support | `232ce582f` | `python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py` lines 208 + 216 (`dtype=torch.half` → `dtype=params_dtype`) | `6b6b45600` |
| GPTQ/AWQ MoE TP>1 non-auto moe-runner dispatch + desc_act branch swap | `a8e2d16bb` | `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py::GPTQMarlinMoEKernel.create_moe_runner` (assert → warning + fallback); `python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py` lines 166-169 (desc_act branch swap) | `6842ed50f` |
| AutoRound GPTQ swapped marlin/non-marlin branches | `92aa83b07` | `python/sglang/srt/layers/quantization/auto_round.py` lines 398-411 (Auto-merged cleanly: `use_marlin: True` → `GPTQMarlinMoEMethod`; `False` → `MoeWNA16Config`) | `a15c82fd1` (empty attribution commit) |
| GPTQ MoE non-marlin config explicit `desc_act: False` | `0eff58101` | `python/sglang/srt/layers/quantization/auto_round.py` line 408 (`"desc_act": False` in `MoeWNA16Config.from_config` dict) | `07c39e5b5` (empty attribution commit) |
| CUDA-graph auto-disable on `--cpu-offload-gb` | `115d00a1e` | `python/sglang/srt/server_args.py::ServerArgs.__post_init__` lines 2625-2634 (inserted before `_handle_cuda_graph_config()` so the legacy `disable_cuda_graph` flag propagates to both prefill + decode phases via `_validate_cuda_graph_config`) | `d215a85ed` |
| `--disable-cuda-graph-padding × --enable-torch-compile` incompatibility assert | `746ff3c87` | `python/sglang/srt/server_args.py::ServerArgs.check_server_args` lines 7056-7062 (Auto-merged cleanly: 8-line assert with O(max_batch_size) message after `pp_max_micro_batch_size` validation, before `pp_size > 1` check) | (auto-merge — no separate commit; consistent with Phase 4b's auto-merge handling per task 6.5 wording) |

### Auto-merged surviving fixes (verified present, no separate commit)

The following surviving fixes were verified present in the auto-merged file content. No separate attribution commit was made because (a) the task's wording was conditional on the fix regressing ("If the fix regressed, re-apply...") and (b) the auto-merge result is itself the audit trail via the fork's original commit hashes being in the merge history.

| Fix | Original fork commit | Location in merged file |
| --- | --- | --- |
| NVFP4 Cutlass MoE global-expert-ID-to-local-ID remap when EP > 1 | `b945ddd0a` | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` lines 388-416 (full EP>1 remapping logic, byte-for-byte match) |
| NVFP4 Cutlass MoE EP>1 crash fix (num_local_experts in CutlassMoEParams) | `cc15dec3c` | Same file, line 282 (`num_experts=layer.num_local_experts`) |
| NVFP4 input global scales stays scalar (no `.expand()`) | `caf47c085` | Same file, lines 203-209 and 222-228 (`w13_input_global_scale` and `w2_input_global_scale` use `.min().to(torch.float32)` with no `.expand(layer.num_local_experts)`) |
| HF rope params: removed manual `rope_theta` injection into `PretrainedConfig` | `bb6c36b64` | `python/sglang/srt/utils/hf_transformers_patches.py::_patch_rope_parameters_validation` lines 138-161 (only the `standardize_rope_params` guard; no `PretrainedConfig.from_dict = patched` override) |
| `MoeWNA16Method` accepts both SiLU and GELU activations | `cb399b1cf` | `python/sglang/srt/layers/quantization/moe_wna16.py` lines 389-395 (`assert self.moe_runner_config.activation in ("silu", "gelu")` with debug message `f"got {self.moe_runner_config.activation}."`) |

### Rationale

This sync drops 26 of 32 fork-only commits because upstream has independently landed equivalent (often more-evolved) versions: the dFlash spec v2 port (~15 commits) is superseded by upstream PRs #23000, #27950, #27959, #29228, #29220, #29678 with 1817+ lines of evolution; the `MambaPoolConfigurator` is superseded by upstream's evolved `handle_max_mamba_cache` which covers the same unified-memory-sizing concern with one fewer fork-only warning; and the flashinfer preflight port (6 commits) is byte-for-byte equivalent to upstream PR #24172 by the same author. The 6 surviving fixes were re-applied against upstream's evolved layout because upstream has not addressed them: GPTQ MoE quantization fixes (3 commits), NVFP4 Cutlass MoE fixes (3 commits), CUDA-graph auto-disable on `--cpu-offload-gb`, `--disable-cuda-graph-padding × --enable-torch-compile` incompatibility assert, HF rope params removal, and `MoeWNA16Method` GELU support. Each surviving fix was committed with attribution to the original fork commit hash for audit-trail purposes, and each was placed at the appropriate new location in upstream's refactored layout (e.g., `gptq/schemes/` instead of the deleted flat `gptq.py`). Future syncs should re-run the keep/drop decision matrix from scratch, but the documented supersession chain here should make that analysis much faster.

### Open questions for future changes

- **OQ1**: Future fork-specific work in `scheduler.py` will continue to fight upstream's 131 commits / 6 weeks of churn. Worth a follow-up change to investigate a hook/mixin pattern that moves fork-specific work out of `scheduler.py` entirely.
- **OQ2**: Should the surviving 6 fixes be submitted upstream as separate PRs to reduce future fork-divergence pressure? Likely yes for the GELU fix, NVFP4 fixes, and CUDA-graph-disable work.
- **OQ3**: Upstream PR #29678's `--enable-unified-memory` flag is an 8378-line opt-in feature. Worth evaluating whether any of the fork's production workloads would benefit from migrating to it (vs. staying on the legacy `handle_max_mamba_cache` path).
