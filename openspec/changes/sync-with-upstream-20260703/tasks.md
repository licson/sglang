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

- [ ] 4.1 Accept upstream's deletion of `python/sglang/srt/layers/quantization/gptq.py`: `git rm python/sglang/srt/layers/quantization/gptq.py`.
- [ ] 4.2 Accept upstream's deletion of `python/sglang/srt/managers/scheduler_output_processor_mixin.py`: `git rm python/sglang/srt/managers/scheduler_output_processor_mixin.py`.
- [ ] 4.3 Accept upstream's deletion of `python/sglang/srt/speculative/dflash_worker.py`: `git rm python/sglang/srt/speculative/dflash_worker.py`.
- [ ] 4.4 Take upstream's version of the 2 add/add dFlash files: `git checkout --theirs python/sglang/srt/speculative/dflash_info_v2.py python/sglang/srt/speculative/dflash_worker_v2.py && git add python/sglang/srt/speculative/dflash_info_v2.py python/sglang/srt/speculative/dflash_worker_v2.py`.
- [ ] 4.5 Take upstream's version of the 13 content-conflicted dFlash integration files: `git checkout --theirs python/sglang/srt/speculative/dflash_info.py python/sglang/srt/speculative/dflash_utils.py python/sglang/srt/models/dflash.py python/sglang/srt/speculative/spec_info.py python/sglang/srt/managers/overlap_utils.py python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/scheduler.py test/registered/spec/dflash/test_dflash.py && git add -A`.
- [ ] 4.6 Take upstream's version of the 2 flashinfer preflight files: `git checkout --theirs python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py && git add python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`.
- [ ] 4.7 Take upstream's version of the 2 mamba pool files (decision D3 — drop `MambaPoolConfigurator`): `git checkout --theirs python/sglang/srt/mem_cache/memory_pool.py python/sglang/srt/model_executor/pool_configurator.py && git add python/sglang/srt/mem_cache/memory_pool.py python/sglang/srt/model_executor/pool_configurator.py`.
- [ ] 4.8 Take upstream's version of the mamba pool test: `git checkout --theirs test/registered/unit/model_executor/test_pool_configurator.py && git add test/registered/unit/model_executor/test_pool_configurator.py`.
- [ ] 4.9 Verify no remaining conflicts: `git diff --name-only --diff-filter=U` returns empty.
- [ ] 4.10 Commit the bulk resolution: `git commit -m "Sync with upstream/main: drop superseded dFlash/mamba-pool/flashinfer-preflight ports"`.
- [ ] 4.11 Verify the new branch HEAD is the merge commit and `pr-ports-20260428` is unchanged.

## 5. Re-apply GPTQ MoE fixes against new gptq/schemes/ layout (Phase 4a)

- [ ] 5.1 Locate the new gptq scheme layout: `ls python/sglang/srt/layers/quantization/gptq/schemes/`.
- [ ] 5.2 Identify the scheme file for the AutoRound swapped-branches fix (fork commit `92aa83b07`). Inspect fork's original change with `git show 92aa83b07`.
- [ ] 5.3 Re-apply the swapped marlin/non-marlin branch correction in the new AutoRound scheme file. Verify the fix is at the same logical location (the branch-selection predicate).
- [ ] 5.4 Identify the scheme file for the GPTQ Marlin MoE bf16 scale dtype fix (fork commit `232ce582f`). Inspect fork's original change with `git show 232ce582f`.
- [ ] 5.5 Re-apply the bf16 acceptance in the new Marlin scheme file. Verify the fix removes the hardcoded `fp16` assertion.
- [ ] 5.6 Identify the dispatch path for the TP>1 non-auto moe-runner fix (fork commit `a8e2d16bb`). Inspect fork's original change with `git show a8e2d16bb`.
- [ ] 5.7 Re-apply the dispatch logic so `moe-runner != auto` with TP>1 doesn't crash. Verify the new dispatch code path (likely in `gptq/schemes/__init__.py` or `fused_moe_triton/layer.py`).
- [ ] 5.8 Identify the desc_act propagation fix for the non-marlin config (fork commit `0eff58101`, "Address review: add desc_act to GPTQ MoE non-marlin config").
- [ ] 5.9 Re-apply the `desc_act` propagation in the new non-marlin config construction. Verify `desc_act` is read from model config, not unconditionally defaulted.
- [ ] 5.10 Commit as 4 small commits (one per fix) with attribution: `git commit -m "fix(moe): re-apply AutoRound GPTQ swapped marlin/non-marlin branches (fork 92aa83b07)"` etc.

## 6. Verify NVFP4 Cutlass MoE fixes survived auto-merge (Phase 4b)

- [ ] 6.1 Inspect the auto-merged file: `git log --oneline -5 -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` and verify it carries fork commit hashes `b945ddd0a`, `cc15dec3c`, `caf47c085` (or equivalent upstream-evolved code).
- [ ] 6.2 Open the file and verify the global-expert-ID-to-local-ID remapping logic is present (fork fix `b945ddd0a`).
- [ ] 6.3 Verify the input global scales scalar fix is present (fork fix `caf47c085`).
- [ ] 6.4 Verify the EP>1 Cutlass MoE crash fix is present (fork fix `cc15dec3c`).
- [ ] 6.5 If any of 6.2-6.4 fail, re-apply the missing fix against the new layout in the same spirit as Phase 5. Commit with attribution.

## 7. Re-apply CUDA-graph disable on --cpu-offload-gb (Phase 4c)

- [ ] 7.1 Inspect fork's original change: `git show 115d00a1e -- python/sglang/srt/server_args.py`.
- [ ] 7.2 Locate the new position of the cpu_offload forward hook validation in upstream's evolved `server_args.py`. Search for `cpu_offload_gb` and `disable_piecewise_cuda_graph` to find the adjacent section.
- [ ] 7.3 Re-apply the block: `if self.cpu_offload_gb > 0:` → warn if `not self.disable_cuda_graph` → set `self.disable_cuda_graph = True`.
- [ ] 7.4 Verify the warning message matches: "CUDA graph is disabled because --cpu-offload-gb is set."
- [ ] 7.5 Commit: `git commit -m "fix(server_args): re-apply CUDA-graph auto-disable on --cpu-offload-gb (fork 115d00a1e)"`.

## 8. Re-apply --disable-cuda-graph-padding × --enable-torch-compile assert (Phase 4d)

- [ ] 8.1 Inspect fork's original change: `git show 746ff3c87 -- python/sglang/srt/server_args.py`.
- [ ] 8.2 Locate the new position of the validation section near `pp_max_micro_batch_size` (or wherever the torch_compile / cuda_graph_padding validations live in upstream's evolved `server_args.py`).
- [ ] 8.3 Re-apply the 8-line `assert not (self.disable_cuda_graph_padding and self.enable_torch_compile)` with the explanatory message about O(max_batch_size) compilation overhead.
- [ ] 8.4 Verify the message tells the user to remove one flag or the other.
- [ ] 8.5 Commit: `git commit -m "fix(server_args): re-apply --disable-cuda-graph-padding × --enable-torch-compile incompatibility assert (fork 746ff3c87)"`.

## 9. Verify HF rope params fix survived auto-merge (Phase 4e)

- [ ] 9.1 Inspect the auto-merged file: `git log --oneline -5 -- python/sglang/srt/utils/hf_transformers_patches.py` and verify fork commit `bb6c36b64` (or equivalent) is present.
- [ ] 9.2 Open the file and verify manual injection of rope parameters (e.g., `rope_scaling`, `rope_theta`) into `PretrainedConfig` is NOT present.
- [ ] 9.3 If the fix regressed, re-apply by removing the manual injection block. Commit with attribution: `git commit -m "fix(hf): re-apply removal of manual rope parameters injection in PretrainedConfig (fork bb6c36b64)"`.

## 10. Verify moe_wna16 GELU fix survived auto-merge (Phase 4f)

- [ ] 10.1 Inspect the auto-merged file: `python/sglang/srt/layers/quantization/moe_wna16.py`.
- [ ] 10.2 Verify the assertion is `assert self.moe_runner_config.activation in ("silu", "gelu"), ...` (not the pre-fix `== "silu"`).
- [ ] 10.3 Verify the assertion message includes the unsupported value for debuggability.
- [ ] 10.4 If the fix regressed, re-apply the 7-insert-3-delete change. Commit with attribution: `git commit -m "fix(moe_wna16): re-apply GELU activation support (fork cb399b1cf)"`.

## 11. Smoke tests (Phase 5)

- [ ] 11.1 `python -c "import sglang"` — must succeed (verify no import errors from the merge).
- [ ] 11.2 `pytest test/registered/unit/server_args/test_server_args.py` — must pass (verifies Phase 7 + 8 gates).
- [ ] 11.3 `pytest test/registered/unit/model_executor/test_pool_configurator.py` — must pass (verifies mamba pool sizing after dropping `MambaPoolConfigurator`).
- [ ] 11.4 `pytest test/registered/spec/dflash/test_dflash.py` — must pass (verifies upstream's evolved dFlash path works with the fork's models).
- [ ] 11.5 `pytest test/registered/unit/mem_cache/test_unified_mamba_views.py` — must pass (upstream's new test from PR #29678).
- [ ] 11.6 Run the project's Docker build against the merged branch. Must succeed.
- [ ] 11.7 Run project-specific smoke tests on the Docker image (per the user's deployment model). Must pass.
- [ ] 11.8 If any of 11.1-11.7 fail, triage as either (a) a regression in a surviving fix that needs adjustment (go back to the relevant Phase 4 task) or (b) a bug in upstream's evolved code that needs reporting (file an upstream issue; revert the specific upstream commit via a follow-up fork-only commit if blocking).

## 12. Document sync decision in AGENTS.md (Phase 6)

- [ ] 12.1 Verify `AGENTS.md` exists at the repo root; create it if absent.
- [ ] 12.2 Append a new section titled `## Upstream sync 2026-07-03`.
- [ ] 12.3 Record the upstream commit synced to (the SHA resolved at task 2.5).
- [ ] 12.4 Record the list of dropped fork-only commits with their superseding upstream PRs:
  - dFlash spec v2 port (~15 commits, `70bdac0da` etc.) → upstream PRs #23000, #27950, #27959, #29228, #29220, #29678
  - dFlash mamba cleanup (`a722fee6a`) → moot without dFlash port
  - `MambaPoolConfigurator` (`d10e980d5`, `625770e2d`) → upstream's evolved `handle_max_mamba_cache`
  - flashinfer preflight (6 commits, `3cce2876e` + follow-ups) → upstream PR #24172 (byte-for-byte equivalent, same author)
- [ ] 12.5 Record the list of re-applied surviving fixes with their new locations:
  - GPTQ MoE fixes (3 commits) → new `gptq/schemes/` layout
  - NVFP4 Cutlass MoE fixes (3 commits) → verified in `compressed_tensors/schemes/w4a4_nvfp4_moe.py`
  - CUDA-graph disable on `--cpu-offload-gb` → re-applied in evolved `server_args.py`
  - `--disable-cuda-graph-padding × --enable-torch-compile` assert → re-applied in evolved `server_args.py`
  - HF rope params fix → verified in `hf_transformers_patches.py`
  - moe_wna16 GELU fix → verified in `moe_wna16.py`
- [ ] 12.6 Write a one-paragraph rationale explaining the keep/drop decision logic for future maintainers.
- [ ] 12.7 Commit: `git commit -m "docs(agents): record upstream sync 2026-07-03 decision matrix"`.

## 13. Push and finalize (Phase 7)

- [ ] 13.1 Push the new branch: `git push -u origin sync-upstream-20260703`.
- [ ] 13.2 Verify `origin/pr-ports-20260428` is unchanged (no force-push occurred): `git rev-parse origin/pr-ports-20260428` matches the pre-operation SHA recorded at task 2.6.
- [ ] 13.3 Verify the pushed branch's merge commit SHA matches local HEAD: `git rev-parse origin/sync-upstream-20260703 == git rev-parse HEAD`.
- [ ] 13.4 Open a PR or tag the release per the fork's deployment model (the user's Docker-pinned release process).
- [ ] 13.5 Defer open questions OQ1 (scheduler.py long-term churn), OQ2 (upstreaming surviving fixes), OQ3 (`--enable-unified-memory` evaluation) to future change proposals. Note these in the PR description if applicable.
