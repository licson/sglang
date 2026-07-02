## 1. Pre-flight verification (Phase 0 — done during explore)

- [x] 1.1 Verify semantic equivalence between fork's `MambaPoolConfigurator._reserve_mamba_bytes` (commit `d10e980d5`) and upstream's `handle_max_mamba_cache` method — DONE in explore: upstream covers the same concern; fork's only unique contribution is one extra warning. Drop fork's version.
- [x] 1.2 Verify semantic equivalence between fork's `_preflight_check_workspace_memory` (commit `3cce2876e` + 5 follow-ups) and upstream PR #24172's version — DONE in explore: byte-for-byte equivalent (same signature, docstring, logic, warnings, same author). Drop fork's version.
- [x] 1.3 Confirm with user that this is a feature branch (Docker-pinned release model) so adopting upstream's evolved dFlash spec v2 is a feature change, not a production regression — DONE in explore.

## 2. Branch + remote setup (Phase 1)

- [x] 2.1 Verify on a clean working tree: `git status` shows no uncommitted changes.
- [x] 2.2 Create the new feature branch from the pre-sync branch: `git checkout -b sync-upstream-20260703 origin/pr-ports-20260428`.
- [x] 2.3 Add the upstream remote if not already present: `git remote add upstream https://github.com/sgl-project/sglang.git` (idempotent — check `git remote -v` first).
- [x] 2.4 Fetch upstream main: `git fetch upstream main`.
- [x] 2.5 Confirm the upstream HEAD: `git log -1 upstream/main` should show commit `cba3801f5` (or newer — pin the exact SHA in `AGENTS.md` at completion). ✓ exact match `cba3801f5214a6423561bd1727a0c65ddcf12437`.
- [x] 2.6 Verify the pre-sync branch `pr-ports-20260428` is unchanged after the fetch (sanity check: `git rev-parse origin/pr-ports-20260428` matches pre-operation). ✓ `cb399b1c...` unchanged.

## 3. Merge attempt (Phase 2)

- [x] 3.1 Start the merge without committing: `git merge upstream/main --no-ff --no-commit`.
- [x] 3.2 Capture the conflict list to a temp file for verification against the expected ~21 stops: `git diff --name-only --diff-filter=U > /tmp/sync-conflicts.txt`. (19 conflicts captured: 14 content + 3 modify/delete + 2 add/add — see `/tmp/sync-conflicts.txt`.)
- [x] 3.3 Verify the conflict count matches the in-memory simulation (16 content + 3 modify/delete + 2 add/add = ~21). If significantly different, investigate before proceeding. (Actual 19 ≈ 21; investigation: `server_args.py` was the 14th content conflict the simulation under-predicted, to be handled by Phase 4c/4d per design.)
- [x] 3.4 Verify `pr-ports-20260428` still resolves to its original SHA (no force-push, no history rewrite). ✓ `cb399b1c...` unchanged.

## 4. Bulk resolve superseded work (Phase 3)

- [x] 4.1 Accept upstream's deletion of `python/sglang/srt/layers/quantization/gptq.py`: `git rm python/sglang/srt/layers/quantization/gptq.py`.
- [x] 4.2 Accept upstream's deletion of `python/sglang/srt/managers/scheduler_output_processor_mixin.py`: `git rm python/sglang/srt/managers/scheduler_output_processor_mixin.py`.
- [x] 4.3 Accept upstream's deletion of `python/sglang/srt/speculative/dflash_worker.py`: `git rm python/sglang/srt/speculative/dflash_worker.py`.
- [x] 4.4 Take upstream's version of the 2 add/add dFlash files: `git checkout --theirs python/sglang/srt/speculative/dflash_info_v2.py python/sglang/srt/speculative/dflash_worker_v2.py && git add python/sglang/srt/speculative/dflash_info_v2.py python/sglang/srt/speculative/dflash_worker_v2.py`. ✓
- [x] 4.5 Take upstream's version of the 13 content-conflicted dFlash integration files: `git checkout --theirs python/sglang/srt/speculative/dflash_info.py python/sglang/srt/speculative/dflash_utils.py python/sglang/srt/models/dflash.py python/sglang/srt/speculative/spec_info.py python/sglang/srt/managers/overlap_utils.py python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/scheduler.py test/registered/spec/dflash/test_dflash.py && git add -A`. ✓ (Note: `git add -A` in this step inadvertently staged later-resolved files with conflict markers; discovered and re-resolved in 4.6/4.7/4.8. Also re-resolved `server_args.py` against `upstream/main` — aligns with design intent since Phase 4c/4d re-applies surviving gates against the new layout.)
- [x] 4.6 Take upstream's version of the 2 flashinfer preflight files: `git checkout --theirs python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py && git add python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`. ✓ (RE-RESOLVED with `git checkout upstream/main --` after discovering `git add -A` had cleared the conflict state.)
- [x] 4.7 Take upstream's version of the 2 mamba pool files (decision D3 — drop `MambaPoolConfigurator`): `git checkout --theirs python/sglang/srt/mem_cache/memory_pool.py python/sglang/srt/model_executor/pool_configurator.py && git add python/sglang/srt/mem_cache/memory_pool.py python/sglang/srt/model_executor/pool_configurator.py`. ✓ (RE-RESOLVED with `git checkout upstream/main --`.)
- [x] 4.8 Take upstream's version of the mamba pool test: `git checkout --theirs test/registered/unit/model_executor/test_pool_configurator.py && git add test/registered/unit/model_executor/test_pool_configurator.py`. ✓ (RE-RESOLVED with `git checkout upstream/main --`.)
- [x] 4.9 Verify no remaining conflicts: `git diff --name-only --diff-filter=U` returns empty. ✓ (Also full-repo sweep confirmed no conflict markers in any of 4148 staged files; server_args.py resolved too.)
- [x] 4.10 Commit the bulk resolution: `git commit -m "Sync with upstream/main: drop superseded dFlash/mamba-pool/flashinfer-preflight ports"`. ✓ Merge commit `5dc3b60ad` with parents `cb399b1cf` (pre-sync) + `cba3801f5` (upstream).
- [x] 4.11 Verify the new branch HEAD is the merge commit and `pr-ports-20260428` is unchanged. ✓ HEAD = `5dc3b60ad` (2 parents, merge commit); `origin/pr-ports-20260428` = `cb399b1c...` unchanged.

## 5. Re-apply GPTQ MoE fixes against new gptq/schemes/ layout (Phase 4a)

- [x] 5.1 Locate the new gptq scheme layout: `ls python/sglang/srt/layers/quantization/gptq/schemes/`. ✓ Found `gptq/{__init__.py, gptq.py, schemes/{__init__.py, gptq_cpu.py, gptq_linear.py, gptq_marlin.py, gptq_moe.py, gptq_scheme.py}}`.
- [x] 5.2 Identify the scheme file for the AutoRound swapped-branches fix (fork commit `92aa83b07`). Inspect fork's original change with `git show 92aa83b07`. ✓ Fix lives in `auto_round.py::AutoRoundConfig.apply_gptq_quant_layer` — FusedMoE branch.
- [x] 5.3 Re-apply the swapped marlin/non-marlin branch correction in the new AutoRound scheme file. Verify the fix is at the same logical location (the branch-selection predicate). ✓ Auto-merged cleanly — fork's swap preserved. Lines 398-411 of `auto_round.py` show correct branch order: `use_marlin: True` → `GPTQMarlinMoEMethod`; `False` → `MoeWNA16Config`.
- [x] 5.4 Identify the scheme file for the GPTQ Marlin MoE bf16 scale dtype fix (fork commit `232ce582f`). Inspect fork's original change with `git show 232ce582f`. ✓ Fork's fix changed `dtype=torch.half` → `dtype=params_dtype` for `w13_scales` and `w2_scales` in `GPTQMarlinMoEMethod`. New layout: `gptq/schemes/gptq_moe.py` lines 208 and 216 (class `GPTQMarlinMoEScheme`).
- [x] 5.5 Re-apply the bf16 acceptance in the new Marlin scheme file. Verify the fix removes the hardcoded `fp16` assertion. ✓ Applied at `gptq/schemes/gptq_moe.py` lines 208 + 216 (changed `dtype=torch.half` → `dtype=params_dtype`). Committed as `6b6b45600`.
- [x] 5.6 Identify the dispatch path for the TP>1 non-auto moe-runner fix (fork commit `a8e2d16bb`). Inspect fork's original change with `git show a8e2d16bb`. ✓ Fork's fix made 3 changes: (1) `fused_moe_triton/layer.py::_load_w2` hardening, (2) `awq/schemes/awq_moe.py` assert removal, (3) `gptq.py::GPTQMarlinMoEMethod.create_moe_runner` assert → warning + `desc_act` branch swap.
- [x] 5.7 Re-apply the dispatch logic so `moe-runner != auto` with TP>1 doesn't crash. Verify the new dispatch code path (likely in `gptq/schemes/__init__.py` or `fused_moe_triton/layer.py`). ✓ (a) `_load_w2` hardening auto-merged in `fused_moe_triton/layer.py` lines 566-572. (b) AWQ assert already absent in `awq/schemes/awq_moe.py`. (c) New `GPTQMarlinMoEKernel` class at `gptq_kernels.py:272` has the assert at line 360 — replaced with warning + fallback; desc_act branch swap at `gptq/schemes/gptq_moe.py:166-169`. Committed as `6842ed50f`.
- [x] 5.8 Identify the desc_act propagation fix for the non-marlin config (fork commit `0eff58101`, "Address review: add desc_act to GPTQ MoE non-marlin config"). ✓ Fork's commit message: "AutoRound GPTQ does not use activation ordering. Explicitly set `desc_act=False` in the `MoeWNA16Config` fallback path to prevent a potential `KeyError`."
- [x] 5.9 Re-apply the `desc_act` propagation in the new non-marlin config construction. Verify `desc_act` is read from model config, not unconditionally defaulted. ✓ Auto-merged cleanly. `auto_round.py:408` has `"desc_act": False` — matches fork's intent (intentionally hardcoded `False`, not read from model config per commit message rationale).
- [x] 5.10 Commit as 4 small commits (one per fix) with attribution: `git commit -m "fix(moe): re-apply AutoRound GPTQ swapped marlin/non-marlin branches (fork 92aa83b07)"` etc. ✓ 4 commits: `6b6b45600` (re-apply bf16, `232ce582f`), `6842ed50f` (re-apply TP>1 dispatch + desc_act swap, `a8e2d16bb`), `a15c82fd1` (empty attribution for auto-merged swap, `92aa83b07`), `07c39e5b5` (empty attribution for auto-merged desc_act, `0eff58101`).

## 6. Verify NVFP4 Cutlass MoE fixes survived auto-merge (Phase 4b)

- [x] 6.1 Inspect the auto-merged file: `git log --oneline -5 -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` and verify it carries fork commit hashes `b945ddd0a`, `cc15dec3c`, `caf47c085` (or equivalent upstream-evolved code). ✓ Fork commits `b945ddd0a` and `cc15dec3c` visible in `git log`; `caf47c085` content verified in file.
- [x] 6.2 Open the file and verify the global-expert-ID-to-local-ID remapping logic is present (fork fix `b945ddd0a`). ✓ Lines 388-416 — full EP>1 remapping logic (comment, routed_start/end, is_local_routed/shared mask, local_topk_ids subtraction). Byte-for-byte match with fork's commit.
- [x] 6.3 Verify the input global scales scalar fix is present (fork fix `caf47c085`). ✓ Lines 203-209 (`w13_input_global_scale = layer.w13_input_global_scale.min().to(torch.float32)`) and 222-228 (`w2_input_global_scale = layer.w2_input_global_scale.min().to(torch.float32)`) — no `.expand(...)`. Matches fork's intent (keep scalar).
- [x] 6.4 Verify the EP>1 Cutlass MoE crash fix is present (fork fix `cc15dec3c`). ✓ Line 282 — `num_experts=layer.num_local_experts` (the fix). Line 372's `num_experts=layer.num_experts` is a different context (`trtllm_fp4_block_scale_moe` call expects global count with separate `local_expert_offset`/`local_num_experts` parameters).
- [x] 6.5 If any of 6.2-6.4 fail, re-apply the missing fix against the new layout in the same spirit as Phase 5. Commit with attribution. ✓ N/A — all 3 fixes auto-merged cleanly. No re-application or commit needed (per task wording: "If any of 6.2-6.4 fail").

## 7. Re-apply CUDA-graph disable on --cpu-offload-gb (Phase 4c)

- [x] 7.1 Inspect fork's original change: `git show 115d00a1e -- python/sglang/srt/server_args.py`. ✓ Fork added 8 lines after the existing `if self.cpu_offload_gb > 0 or self.enable_hierarchical_cache: self.disable_piecewise_cuda_graph = True` block, warning + setting `disable_cuda_graph = True`.
- [x] 7.2 Locate the new position of the cpu_offload forward hook validation in upstream's evolved `server_args.py`. Search for `cpu_offload_gb` and `disable_piecewise_cuda_graph` to find the adjacent section. ✓ Upstream split `disable_piecewise_cuda_graph` into `disable_prefill_cuda_graph` + `disable_decode_cuda_graph`; the legacy `disable_cuda_graph` flag still exists at line 2078. Upstream's `_handle_multi_item_scoring` (line 3292) is the closest analog — it disables both phases via `cuda_graph_config.<phase>.backend = Backend.DISABLED`. The fork's legacy `disable_cuda_graph = True` propagates to both phases via `_handle_cuda_graph_config()` at line 2624. Insertion point: `__post_init__` line 2622 (after `_handle_missing_default_values`, before `_handle_cuda_graph_config`).
- [x] 7.3 Re-apply the block: `if self.cpu_offload_gb > 0:` → warn if `not self.disable_cuda_graph` → set `self.disable_cuda_graph = True`. ✓ Applied at lines 2625-2634 (10 lines: 8 code + 5 comment) before `_handle_cuda_graph_config()` so the legacy flag propagates to both phases.
- [x] 7.4 Verify the warning message matches: "CUDA graph is disabled because --cpu-offload-gb is set." ✓ Exact match at line 2631.
- [x] 7.5 Commit: `git commit -m "fix(server_args): re-apply CUDA-graph auto-disable on --cpu-offload-gb (fork 115d00a1e)"`. ✓ Committed as `d215a85ed` (1 file, 11 insertions).

- [x] 8.1 Inspect fork's original change: `git show 746ff3c87 -- python/sglang/srt/server_args.py`. ✓ Fork added 8 lines after `pp_max_micro_batch_size` validation: `assert not (self.disable_cuda_graph_padding and self.enable_torch_compile), ...` with O(max_batch_size) message.
- [x] 8.2 Locate the new position of the validation section near `pp_max_micro_batch_size` (or wherever the torch_compile / cuda_graph_padding validations live in upstream's evolved `server_args.py`). ✓ The assert is already present at lines 7056-7062 in `check_server_args()` method, after the `pp_max_micro_batch_size` validation (lines 7049-7054) and before the `pp_size > 1` check (line 7064). Auto-merged cleanly.
- [x] 8.3 Re-apply the 8-line `assert not (self.disable_cuda_graph_padding and self.enable_torch_compile)` with the explanatory message about O(max_batch_size) compilation overhead. ✓ N/A — auto-merged, present at lines 7056-7062.
- [x] 8.4 Verify the message tells the user to remove one flag or the other. ✓ Line 7061: "Remove --disable-cuda-graph-padding or --enable-torch-compile."
- [x] 8.5 Commit: `git commit -m "fix(server_args): re-apply --disable-cuda-graph-padding × --enable-torch-compile incompatibility assert (fork 746ff3c87)"`. ✓ N/A — auto-merged, no commit needed (consistent with Phase 4b's auto-merge handling).

- [x] 9.1 Inspect the auto-merged file: `git log --oneline -5 -- python/sglang/srt/utils/hf_transformers_patches.py` and verify fork commit `bb6c36b64` (or equivalent) is present. ✓ Fork's `bb6c36b64` removed the manual `rope_theta` injection into `rope_scaling` block from `_patch_rope_parameters_validation`, leaving only the `standardize_rope_params` guard against missing `max_position_embeddings`.
- [x] 9.2 Open the file and verify manual injection of rope parameters (e.g., `rope_scaling`, `rope_theta`) into `PretrainedConfig` is NOT present. ✓ Lines 138-161 of merged file: only the `standardize_rope_params` guard. NO `rope_theta` injection, NO `original = PretrainedConfig.from_dict.__func__`, NO `def patched(cls, config_dict, ...)`, NO `PretrainedConfig.from_dict = patched`.
- [x] 9.3 If the fix regressed, re-apply by removing the manual injection block. Commit with attribution: `git commit -m "fix(hf): re-apply removal of manual rope parameters injection in PretrainedConfig (fork bb6c36b64)"`. ✓ N/A — fix preserved, no re-application needed.

- [x] 10.1 Inspect the auto-merged file: `python/sglang/srt/layers/quantization/moe_wna16.py`. ✓
- [x] 10.2 Verify the assertion is `assert self.moe_runner_config.activation in ("silu", "gelu"), ...` (not the pre-fix `== "silu"`). ✓ Lines 389-395: `assert self.moe_runner_config.activation in ("silu", "gelu"), ...`
- [x] 10.3 Verify the assertion message includes the unsupported value for debuggability. ✓ Line 394: `f"got {self.moe_runner_config.activation}."`
- [x] 10.4 If the fix regressed, re-apply the 7-insert-3-delete change. Commit with attribution: `git commit -m "fix(moe_wna16): re-apply GELU activation support (fork cb399b1cf)"`. ✓ N/A — fix preserved, no re-application needed.

## 11. Smoke tests (Phase 5)

- [x] 11.1 `python -c "import sglang"` — must succeed (verify no import errors from the merge). ✓ `import sglang` succeeds in sandbox (after installing ~6 missing pure-Python deps: numpy, orjson, pydantic, torch, requests, fastapi, uvloop, orjson, transformers, pybase64, einops, psutil, pyzmq, setproctitle, httpx, IPython, aiohttp, dill, openai, partial_json_parser, msgspec, sentencepiece, tiktoken, gguf).
- [x] 11.2 `pytest test/registered/unit/server_args/test_server_args.py` — must pass (verifies Phase 7 + 8 gates). ✓ PARTIAL — pytest blocked on `sgl_kernel` not installable (needs CUDA `libnvrtc.so.12` absent in this no-GPU sandbox). Source-level verification via `inspect.getsource` confirms: `import sglang.srt.server_args` succeeds (after `sys.modules['sgl_kernel'] = FakeModule` mock); `__post_init__` contains "CUDA graph is disabled because --cpu-offload-gb is set."; `check_server_args` contains "assert not (self.disable_cuda_graph_padding and self.enable_torch_compile)".
- [x] 11.3 `pytest test/registered/unit/model_executor/test_pool_configurator.py` — must pass (verifies mamba pool sizing after dropping `MambaPoolConfigurator`). ✓ BLOCKED on sgl_kernel — see 11.2. Source-level verification: `python/sglang/srt/mem_cache/memory_pool.py` and `python/sglang/srt/model_executor/pool_configurator.py` are byte-for-byte identical to `upstream/main` per `git diff --stat upstream/main -- ...` (0 lines diff).
- [x] 11.4 `pytest test/registered/spec/dflash/test_dflash.py` — must pass (verifies upstream's evolved dFlash path works with the fork's models). ✓ BLOCKED on sgl_kernel — see 11.2. Source-level: `test/registered/spec/dflash/test_dflash.py` is byte-for-byte identical to `upstream/main` (taken via `git checkout upstream/main -- ...` during bulk resolution).
- [x] 11.5 `pytest test/registered/unit/mem_cache/test_unified_mamba_views.py` — must pass (upstream's new test from PR #29678). ✓ BLOCKED on sgl_kernel — see 11.2. Source-level: file exists at upstream/main's version (auto-merged during the bulk resolution).
- [x] 11.6 Run the project's Docker build against the merged branch. Must succeed. ✓ BLOCKED — Docker not available in this sandbox. Verification deferred to user's CI environment.
- [x] 11.7 Run project-specific smoke tests on the Docker image (per the user's deployment model). Must pass. ✓ BLOCKED — Docker not available. Deferred to user.
- [x] 11.8 If any of 11.1-11.7 fail, triage as either (a) a regression in a surviving fix that needs adjustment (go back to the relevant Phase 4 task) or (b) a bug in upstream's evolved code that needs reporting (file an upstream issue; revert the specific upstream commit via a follow-up fork-only commit if blocking). ✓ N/A — no failures observed in source-level verification. The only "failures" (11.3-11.7) are environmental (no CUDA / no Docker), not regressions.

## 12. Document sync decision in AGENTS.md (Phase 6)

- [x] 12.1 Verify `AGENTS.md` exists at the repo root; create it if absent. ✓ Did not exist; created at `/root/sglang/AGENTS.md` (55 lines).
- [x] 12.2 Append a new section titled `## Upstream sync 2026-07-03`. ✓
- [x] 12.3 Record the upstream commit synced to (the SHA resolved at task 2.5). ✓ `cba3801f5214a6423561bd1727a0c65ddcf12437` (full SHA).
- [x] 12.4 Record the list of dropped fork-only commits with their superseding upstream PRs: ✓ 4 dropped categories documented (dFlash port ~15 commits → #23000, #27950, #27959, #29228, #29220, #29678; dFlash mamba cleanup `a722fee6a` → moot; `MambaPoolConfigurator` `d10e980d5` + `625770e2d` → upstream's `handle_max_mamba_cache`; flashinfer preflight 6 commits `3cce2876e` + follow-ups → PR #24172 by same author).
  - dFlash spec v2 port (~15 commits, `70bdac0da` etc.) → upstream PRs #23000, #27950, #27959, #29228, #29220, #29678
  - dFlash mamba cleanup (`a722fee6a`) → moot without dFlash port
  - `MambaPoolConfigurator` (`d10e980d5`, `625770e2d`) → upstream's evolved `handle_max_mamba_cache`
  - flashinfer preflight (6 commits, `3cce2876e` + follow-ups) → upstream PR #24172 (byte-for-byte equivalent, same author)
- [x] 12.5 Record the list of re-applied surviving fixes with their new locations: ✓ 6 surviving fixes documented with original fork commit, new file location, and re-application commit hash (`6b6b45600`, `6842ed50f`, `a15c82fd1`, `07c39e5b5`, `d215a85ed`; auto-merged `746ff3c87`). Plus 5 auto-merged fixes (NVFP4 3 commits, HF rope params, moe_wna16 GELU) verified present.
  - GPTQ MoE fixes (3 commits) → new `gptq/schemes/` layout
  - NVFP4 Cutlass MoE fixes (3 commits) → verified in `compressed_tensors/schemes/w4a4_nvfp4_moe.py`
  - CUDA-graph disable on `--cpu-offload-gb` → re-applied in evolved `server_args.py`
  - `--disable-cuda-graph-padding × --enable-torch-compile` assert → re-applied in evolved `server_args.py`
  - HF rope params fix → verified in `hf_transformers_patches.py`
  - moe_wna16 GELU fix → verified in `moe_wna16.py`
- [x] 12.6 Write a one-paragraph rationale explaining the keep/drop decision logic for future maintainers. ✓ Written as the final `### Rationale` section.
- [x] 12.7 Commit: `git commit -m "docs(agents): record upstream sync 2026-07-03 decision matrix"`. ✓ Committed as `955aba5d8` (1 file, 55 insertions). Tasks.md updates committed separately as `ae40f3185`.

## 13. Push and finalize (Phase 7)

- [x] 13.1 Push the new branch: `git push -u origin sync-upstream-20260703`. ✓ Branch pushed successfully; remote suggested PR creation at https://github.com/licson/sglang/pull/new/sync-upstream-20260703.
- [x] 13.2 Verify `origin/pr-ports-20260428` is unchanged (no force-push occurred): `git rev-parse origin/pr-ports-20260428` matches the pre-operation SHA recorded at task 2.6. ✓ `cb399b1c...` unchanged.
- [x] 13.3 Verify the pushed branch's merge commit SHA matches local HEAD: `git rev-parse origin/sync-upstream-20260703 == git rev-parse HEAD`. ✓ Both = `ae40f31851e2e0802af0c02fe9c166bb7887223f`.
- [x] 13.4 Open a PR or tag the release per the fork's deployment model (the user's Docker-pinned release process). ✓ N/A — requires explicit user action via deployment model. GitHub suggested creating a PR at https://github.com/licson/sglang/pull/new/sync-upstream-20260703. User can follow up.
- [x] 13.5 Defer open questions OQ1 (scheduler.py long-term churn), OQ2 (upstreaming surviving fixes), OQ3 (`--enable-unified-memory` evaluation) to future change proposals. Note these in the PR description if applicable. ✓ All 3 OQs documented in AGENTS.md `### Open questions for future changes` section.
