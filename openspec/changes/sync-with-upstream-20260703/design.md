## Context

The SGLang fork at `github.com/licson/sglang` (branch `pr-ports-20260428`) was last synced with upstream `sgl-project/sglang` on May 9, 2026 at commit `73b8eda10`. As of July 3, 2026 (upstream HEAD `cba3801f5`), upstream has advanced by 2359 commits over ~2 months. The fork carries 32 custom commits (plus 18 merge commits) on top of its last sync point, totaling +2841 / −372 lines across 35 files.

Each release is Docker-pinned per the user's deployment model, so a sync landing on a new feature branch (`sync-upstream-20260703`) does not regress production — adopting upstream's evolved dFlash spec v2 in place of the fork's port is a feature change, not a behavioral regression.

In-memory merge simulation (`git merge-tree --write-tree origin/pr-ports-20260428 upstream/main`) produces 21 stops: 16 content conflicts, 3 modify/delete (upstream deleted files the fork modified), and 2 add/add (upstream added the same paths the fork added). Systematic per-commit ancestry checks revealed 26 of the 32 custom commits are now superseded by upstream; only 6 remain genuinely valuable.

High-churn upstream files touched by the fork: `scheduler.py` (131 commits in 6 weeks before the fork's sync), `server_args.py` (69 commits), `schedule_batch.py`, `overlap_utils.py`. These produce recurring conflict pressure on any future fork-specific work touching them.

## Goals / Non-Goals

**Goals:**
- Sync the fork to upstream `cba3801f5` in a single, reversible merge commit on a new Docker-pinned feature branch.
- Drop fork-only work that upstream has independently landed (or evolved beyond), reducing the long-term maintenance surface.
- Preserve the 6 surviving fork-only behavioral fixes by re-applying them against upstream's evolved layout (or verifying their auto-merge didn't regress).
- Capture the keep/drop decision matrix so future syncs are repeatable rather than re-derived.
- Preserve `pr-ports-20260428` unchanged as a rollback point.

**Non-Goals:**
- Refactoring the surviving fixes to be more upstream-friendly (e.g., minimizing `scheduler.py` touchpoints). Future work.
- Submitting the surviving fixes as new upstream PRs. Future work.
- Designing a hook/mixin pattern to move future fork-specific work out of `scheduler.py`. Future work — surfaced as an open question.
- Preserving the 18-way PR-port merge topology of `pr-ports-20260428`. A single merge commit is sufficient.
- Backfilling the surviving fixes with new tests in the upstream test layout. Each surviving fix retains its existing test coverage (where present); no new test scaffolding is added.

## Decisions

### D1: Merge instead of rebase

**Choice:** `git merge upstream/main --no-ff --no-commit` on a new branch, not `git rebase`.

**Rationale:** The fork's `pr-ports-20260428` carries 18 merge commits encoding the PR-port topology. Replaying 32 non-merge commits + redoing 18 merge resolutions through `scheduler.py`'s 131 commits of churn produces the same conflict at 32 different replay points. A single merge surfaces conflicts once.

**Alternatives considered:**
- `git rebase --rebase-merges`: preserves topology but multiplies conflict work.
- Squash-and-rebase: collapses 32 → 5 themed commits, but discards attribution for the 26 superseded commits we're dropping anyway; and the surviving 6 still fight the same conflict surface one at a time.
- `git merge --ff-only`: impossible — histories have diverged; upstream is not an ancestor of the fork.

### D2: Take upstream's version for the 21 stopped files (Phase 3 bulk checkout)

**Choice:** For each of the 21 conflicting files, take upstream's version wholesale via `git checkout --theirs` (or `git rm` for files upstream deleted).

**Rationale:** All 21 stops correspond to superseded fork-only work:
- 13 content conflicts in dFlash files + flashinfer preflight files + mamba pool files (Phase 0 verified supersession).
- 3 modify/delete: upstream deleted `gptq.py`, `scheduler_output_processor_mixin.py`, `dflash_worker.py`; the fork's modifications in those files addressed the dFlash port or pre-existing issues that upstream has since refactored.
- 2 add/add: upstream added `dflash_info_v2.py` and `dflash_worker_v2.py` with its own evolved implementation; the fork's versions are 628/351 lines vs upstream's 1754/286 lines respectively.

**Alternatives considered:**
- Manual 3-way resolution per file: 16 hours of work for no behavioral gain — the fork's versions are strictly less-evolved than upstream's.
- Cherry-pick individual upstream commits: leaves the fork in a half-merged state that's hard to reason about.

### D3: Drop `MambaPoolConfigurator` (T0.1 verified)

**Choice:** Drop the fork's `MambaPoolConfigurator` (`d10e980d5` + tests `625770e2d`); take upstream's evolved `handle_max_mamba_cache` path.

**Rationale:** Phase 0 (T0.1) compared `_reserve_mamba_bytes` (fork) against upstream's `handle_max_mamba_cache` (the very method the fork removed). Upstream has since independently evolved that method to cover the same concern:

| Concern | Fork's `_reserve_mamba_bytes` | Upstream's `handle_max_mamba_cache` |
|---|---|---|
| Explicit `max_mamba_cache_size` with DP-size adjustment | ✅ | ✅ + spec intermediate accounting |
| Deterministic sizing when `max_running_requests` known | ✅ | ✅ |
| `disable_radix_cache` + `max_running_requests` branch | ❌ not separately handled | ✅ separate branch |
| Ratio-based fallback | ✅ | ✅ joint solve for intermediate memory |
| Spec decoding intermediate memory | ✅ in `_reserve_mamba_bytes` | ✅ in all 3 branches |
| Insufficient-memory validation | ✅ `RuntimeError` on `reserved_bytes >= available_bytes` | ✅ `RuntimeError` on `max_mamba_cache_size <= 0` |
| Warning when `max_mamba_cache_size` < required | ✅ only fork has this | ❌ only `_resolve_max_num_reqs` warning |

The fork's only unique contribution is one extra warning. Not worth maintaining a parallel configurator against 40+ commits of upstream churn on `memory_pool.py` alone.

**Alternatives considered:**
- Keep `MambaPoolConfigurator` and re-apply against upstream's evolved `pool_configurator.py`: pays ongoing maintenance cost as upstream continues to churn this area; offers one extra warning.
- Use upstream's new opt-in `--enable-unified-memory` feature (PR #29678) instead of the legacy path: that's a separate, larger architectural change (8378 lines across 28 files); not in scope for this sync.

### D4: Drop the flashinfer fusion preflight port (T0.2 verified)

**Choice:** Drop the fork's 6 preflight commits (`3cce2876e` + 5 follow-ups: `cd4206197`, `d1a6b3a60`, `50c777bcf`, `df4eca1e5`, `1bf6f9977`); take upstream PR #24172's version.

**Rationale:** Phase 0 (T0.2) compared the fork's `_preflight_check_workspace_memory` (in `flashinfer_comm_fusion.py`) against upstream's. They are functionally identical:
- Same function signature: `def _preflight_check_workspace_memory(world_size, max_token_num, hidden_dim, dtype, cpu_group=None) -> bool`.
- Same docstring: "Collectively decide whether to enter FlashInfer workspace creation..." with identical body text about SymmDeviceMemory buffers, `cuMemCreate` failures, CPU-group voting.
- Same logic: `_make_flashinfer_workspace_allocation_prop` → `_flashinfer_trtllm_workspace_allocation_sizes` → `_probe_cumem_create_sequence` → `torch.tensor([1 if local_ok else 0])` + `dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)`.
- Same warning text: "FlashInfer workspace preflight: cuMemCreate probe failed on at least one rank. Skipping allreduce fusion to avoid cross-rank desync inside the flashinfer collective."
- Five of the fork's six commits are by `kpham-sgl` (Khoa Pham); upstream PR #24172 is by Khoa Pham (`khoa.pham@radixark.ai`).

**Alternatives considered:**
- Keep the fork's version and let it conflict on the merged file: pure overhead — same code, same author, no behavioral difference.
- Re-apply the fork's version on top of upstream's: produces a no-op diff, adds merge overhead on every future sync.

### D5: Re-apply the 6 surviving fixes against upstream's evolved layout

**Choice:** Manually re-apply 6 fork-only commits as 6 small commits (one per fix) on top of the merge, rather than replaying their original hashes.

**Rationale:** Each surviving fix touches a file that upstream has refactored (gptq split into `gptq/schemes/`, `server_args.py` churned 69 times). The original hashes can't be cherry-picked cleanly. Re-applying as small, well-described commits gives a clean audit trail on the new branch.

**Alternatives considered:**
- Cherry-pick original hashes: would re-introduce the dFlash integration touches in `scheduler.py`/`overlap_utils.py` that we explicitly dropped in Phase 3.
- Single squashed "re-apply fork fixes" commit: loses per-fix attribution and individual revertability.

### D6: Run upstream's test suite as the gating criterion

**Choice:** The 5 pytest targets listed in tasks T5.1-T5.5 are the gating criterion for merge completion. No new tests are added.

**Rationale:** Upstream's test suite (especially `test_unified_mamba_views.py` from PR #29678 and the evolved `test_dflash.py`) exercises the code paths the fork is now adopting. The surviving 6 fixes already have test coverage in the fork's existing test suite; that coverage carries forward via the auto-merge.

**Alternatives considered:**
- Add a regression test per surviving fix: out of scope for a sync change; would inflate the diff and conflict surface.
- Skip the smoke tests and rely on Docker build success: insufficient — a successful build doesn't catch behavioral regressions in `server_args.py` validation or pool sizing.

### D7: Capture decisions in `AGENTS.md` for future maintainers

**Choice:** Append an entry to `AGENTS.md` describing the sync (date, upstream commit synced to, dropped work, re-applied fixes, rationale).

**Rationale:** The keep/drop decision matrix took ~2 hours of investigation to derive. Without writing it down, the next sync will re-derive it from scratch. `AGENTS.md` is already the conventional location for AI-agent-readable maintainer notes in this repo (the opencode skill system uses it).

**Alternatives considered:**
- Put it in `docs/sync-history.md`: less discoverable for AI agents.
- Put it in the merge commit message: ephemeral; lost when the branch is eventually rebased away.

## Risks / Trade-offs

- **[Risk] Upstream's evolved dFlash spec v2 has behavior differences from the fork's port** → Each release is Docker-pinned, so any behavioral regression is contained to the new feature branch and discovered in smoke tests (T5.4 + T5.6). If a specific dFlash workload fails, the rollback is `git revert <merge-commit>`; `pr-ports-20260428` is preserved unchanged.

- **[Risk] Upstream's `handle_max_mamba_cache` handles the `disable_radix_cache` edge case that the fork's `MambaPoolConfigurator` didn't** → This is a *correction*, not a regression. Any fork-side workload that previously mis-sized under `disable_radix_cache + max_running_requests` will now size correctly. No action needed.

- **[Risk] Upstream's PR #29678 (`--enable-unified-memory`) introduces a 8378-line opt-in feature that the fork won't exercise** → The flag defaults to off; the fork's existing workloads don't set it. Zero behavioral impact on the fork's existing deployments. If future workloads want the new unified-memory path, that's a separate future change.

- **[Risk] The 3 GPTQ MoE fixes won't re-map cleanly to the new `gptq/schemes/` layout** → T4.1 budgeted 2-3 hours for this exact possibility. If a fix doesn't apply (e.g., the scheme it patched no longer exists), document it as dropped in `AGENTS.md` rather than force-applying against the wrong file.

- **[Risk] A surviving fix's auto-merge silently regressed** → T4.5 and T4.6 explicitly verify `hf_transformers_patches.py` and `moe_wna16.py` by inspecting the merged content for the fork's specific change. T5.2-T5.5 pytest targets will catch behavioral regressions.

- **[Risk] Upstream has a bug in its evolved dFlash or mamba pool code that the fork's port didn't have** → Possible but unlikely (upstream has CI coverage we don't). If it happens, file an upstream issue and revert the specific upstream commit via a follow-up fork-only commit.

- **[Trade-off] Dropping `MambaPoolConfigurator` loses the "max_mamba_cache_size too small" warning** → Users who set `--max-mamba-cache-size` below their `max_running_requests * mamba_ratio` will no longer get a warning at config time; they'll only see the `_resolve_max_num_reqs` clamping warning at runtime. Acceptable — the runtime warning is sufficient and upstream-aligned.

- **[Trade-off] A single merge commit discards the 18-way PR-port topology of `pr-ports-20260428`** → The topology is preserved in the original branch (which we keep as a rollback point). The new feature branch has a simpler, single-merge history that's easier to reason about and re-sync in the future.

## Migration Plan

1. **Pre-flight (Phase 0, done in explore)**: Verified T0.1 (mamba pool) and T0.2 (flashinfer preflight) semantic equivalence.
2. **Branch setup (Phase 1)**: `git checkout -b sync-upstream-20260703 origin/pr-ports-20260428`; add `upstream` remote if absent; `git fetch upstream main`.
3. **Merge (Phase 2)**: `git merge upstream/main --no-ff --no-commit`; expect ~21 stops; capture the conflict list for verification.
4. **Bulk resolve (Phase 3)**: For each of the 21 stopped files, `git checkout --theirs <path>` (or `git rm` for upstream-deleted files); `git add -A`; `git commit -m "Sync with upstream/main: drop superseded dFlash/mamba-pool/flashinfer-preflight ports"`.
5. **Re-apply surviving fixes (Phase 4)**: 6 small commits, one per fix, each touching the minimum necessary file(s) in the new layout.
6. **Smoke tests (Phase 5)**: 5 pytest targets + Docker build + project-specific smoke test.
7. **Document (Phase 6)**: Append the sync decision matrix to `AGENTS.md`.

**Rollback strategy:** `git checkout pr-ports-20260428` (original branch preserved). Or `git revert <merge-commit>` to undo just the sync on the new branch.

## Open Questions

- **OQ1**: Future fork-specific work in `scheduler.py` will continue to fight upstream's 131 commits / 6 weeks of churn. Worth a follow-up change to investigate a hook/mixin pattern that moves fork-specific work out of `scheduler.py` entirely. Not in scope for this sync.

- **OQ2**: Should the surviving 6 fixes be submitted upstream as separate PRs to reduce future fork-divergence pressure? Likely yes for the GELU fix, NVFP4 fixes, and CUDA-graph-disable work. Not in scope for this sync; each upstream PR would be its own change proposal.

- **OQ3**: Upstream PR #29678's `--enable-unified-memory` flag is a 8378-line opt-in feature. Worth evaluating whether any of the fork's production workloads would benefit from migrating to it (vs. staying on the legacy `handle_max_mamba_cache` path). Not in scope for this sync; would be a separate change proposal if pursued.
