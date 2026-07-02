## ADDED Requirements

### Requirement: Fork sync SHALL be performed as a single merge commit on a new Docker-pinned feature branch

The sync from upstream `sgl-project/sglang` main onto a fork feature branch SHALL be performed as a single `git merge upstream/main --no-ff --no-commit` followed by bulk conflict resolution and one finalizing commit. The pre-sync branch (`pr-ports-20260428`) SHALL be preserved unchanged as a rollback point. The new feature branch SHALL be Docker-pinned per the fork's release model.

**Rationale:** A single merge surfaces conflicts once; a rebase would replay the same 32 commits through 131 commits of `scheduler.py` churn, multiplying conflict work 32×. Docker-pinned releases mean the sync lands as a feature change on a new branch, not a production regression.

#### Scenario: Successful single-merge sync
- **WHEN** the maintainer runs `git checkout -b sync-upstream-20260703 origin/pr-ports-20260428 && git merge upstream/main --no-ff --no-commit`
- **THEN** git produces a conflict list of approximately 21 stops (16 content conflicts + 3 modify/delete + 2 add/add)
- **AND** the maintainer resolves each stop by taking upstream's version (`git checkout --theirs <path>` or `git rm <path>`)
- **AND** commits once with message "Sync with upstream/main: drop superseded dFlash/mamba-pool/flashinfer-preflight ports"
- **AND** `pr-ports-20260428` remains unchanged as the rollback point

#### Scenario: Rollback after sync failure
- **WHEN** a smoke test fails on the merged feature branch
- **THEN** the maintainer runs `git revert <merge-commit>` or `git checkout pr-ports-20260428`
- **AND** the original branch is intact and deployable as the previous release

### Requirement: Each upstream sync SHALL classify every fork-only commit as Keep or Drop with a documented rationale

For every non-merge commit on the fork branch (those in `git log --no-merges <merge-base>..HEAD`), the sync process SHALL determine whether the commit is superseded by an equivalent or more-evolved upstream commit. The decision (Keep / Drop) and rationale SHALL be recorded in `AGENTS.md` so future syncs don't re-derive the analysis from scratch.

**Rationale:** 26 of 32 custom commits on `pr-ports-20260428` were found to be superseded by upstream during the July 3, 2026 sync. The investigation took ~2 hours of targeted ancestry checks, file diffs, and semantic comparison. Without capturing the decision, the next sync repeats the work.

#### Scenario: Superseded commit is dropped
- **WHEN** a fork-only commit is found to have byte-for-byte equivalent (or strictly more-evolved) upstream counterpart
- **THEN** the sync takes upstream's version wholesale (`git checkout --theirs <path>`)
- **AND** the decision is recorded in `AGENTS.md` with the fork commit hash, the upstream commit/PR that supersedes it, and a one-line rationale

#### Scenario: Surviving commit is re-applied
- **WHEN** a fork-only commit addresses a concern upstream has not addressed
- **THEN** the sync re-applies the fix as a small, well-described commit on top of the merge
- **AND** the re-application is adapted to upstream's evolved layout (e.g., new file paths, refactored class hierarchy)
- **AND** the original fork commit hash is cited in the new commit message for attribution

### Requirement: The sync SHALL verify semantic equivalence before dropping a fork-only port

When the sync drops a fork-only port because upstream has an equivalent, the maintainer SHALL perform a semantic equivalence check (not just a subject-line match) before proceeding. The check SHALL compare function signatures, docstrings, core logic, and warning/error text. The check results SHALL be recorded in `design.md` under Decisions.

**Rationale:** During the July 3, 2026 sync, `FETCH_HEAD` drift caused an early false-positive conclusion that "all fork commits are in upstream." The error was caught only by pinning to a specific upstream commit SHA and re-running ancestry checks. Subject-line matching is insufficient — a `git log --grep` hit doesn't prove semantic equivalence.

#### Scenario: Semantic equivalence confirmed
- **WHEN** the maintainer compares fork's `_preflight_check_workspace_memory` (commit `3cce2876e`) against upstream PR #24172's version
- **THEN** the comparison covers function signature, docstring, `_make_flashinfer_workspace_allocation_prop`, `_flashinfer_trtllm_workspace_allocation_sizes`, `_probe_cumem_create_sequence`, the `dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)` vote, and the warning text
- **AND** all elements match
- **AND** the conclusion ("drop fork's port, take upstream's version") is recorded in `design.md` D4 with the comparison summary

#### Scenario: Semantic equivalence NOT confirmed
- **WHEN** the maintainer compares fork's `MambaPoolConfigurator._reserve_mamba_bytes` (commit `d10e980d5`) against upstream's `handle_max_mamba_cache`
- **THEN** the comparison finds upstream's version covers the same concern with slightly different edge-case handling (upstream has a `disable_radix_cache` branch the fork lacks; the fork has a "max_mamba_cache_size too small" warning upstream lacks)
- **AND** the maintainer judges the difference is not worth maintaining a parallel configurator against 40+ commits of upstream churn
- **AND** the conclusion ("drop, with noted trade-off") is recorded in `design.md` D3 including the trade-off row

### Requirement: Upstream sync SHALL NOT rewrite or delete the pre-sync branch

The sync process SHALL NOT use `git push --force`, `git reset --hard`, or any history-rewriting operation on the pre-sync branch (`pr-ports-20260428`). All sync work SHALL occur on a new branch.

**Rationale:** The pre-sync branch is the rollback point and the historical record of the fork's PR-port topology. Rewriting it would destroy the rollback path and the audit trail of which upstream PRs were ported in what order.

#### Scenario: Pre-sync branch preserved
- **WHEN** the sync completes (successfully or unsuccessfully)
- **THEN** `git log --oneline origin/pr-ports-20260428` shows the same commits as before the sync began
- **AND** no `git push --force` was issued against `origin/pr-ports-20260428`

### Requirement: The sync decision matrix SHALL be captured in AGENTS.md for future maintainers

The sync SHALL append an entry to `AGENTS.md` (or create it if absent) describing: the date of sync, the upstream commit synced to, the list of dropped fork-only commits with their superseding upstream PRs, the list of re-applied surviving fixes with their new locations in upstream's layout, and a one-paragraph rationale.

**Rationale:** Future maintainers (human or AI agent) need to know why specific commits were dropped vs. re-applied, otherwise the next sync re-derives the analysis. `AGENTS.md` is the conventional location for AI-agent-readable maintainer notes in this repo.

#### Scenario: AGENTS.md entry exists after sync
- **WHEN** the sync completes successfully
- **THEN** `AGENTS.md` contains a section titled "Upstream sync 2026-07-03" (or similar)
- **AND** the section lists the upstream commit synced to (`cba3801f5`)
- **AND** the section lists the dropped commits by hash with their superseding upstream PR
- **AND** the section lists the re-applied fixes by new commit hash with their location in upstream's layout
- **AND** the section explains the rationale in one paragraph

### Requirement: The sync SHALL run the gating pytest targets before declaring completion

The sync SHALL run (at minimum) the following pytest targets on the merged branch and all MUST pass before declaring the sync complete: `test/registered/unit/server_args/test_server_args.py`, `test/registered/unit/model_executor/test_pool_configurator.py`, `test/registered/spec/dflash/test_dflash.py`, `test/registered/unit/mem_cache/test_unified_mamba_views.py`, and any project-specific smoke tests. A Docker build success alone is insufficient.

**Rationale:** A successful Docker build doesn't catch behavioral regressions in `server_args.py` validation, memory pool sizing logic, or the dFlash integration. The 5 pytest targets exercise exactly the code paths the fork is now adopting from upstream.

#### Scenario: All gating tests pass
- **WHEN** the maintainer runs the 5 pytest targets on the merged branch
- **THEN** all 5 pass
- **AND** the sync is declared complete
- **AND** the branch is pushed to `origin/sync-upstream-20260703`

#### Scenario: Any gating test fails
- **WHEN** the maintainer runs the 5 pytest targets on the merged branch
- **AND** one or more fail
- **THEN** the maintainer triages each failure as either (a) a regression in a surviving fix that needs adjustment, or (b) a bug in upstream's evolved code that needs reporting
- **AND** does NOT declare the sync complete until all 5 pass
- **AND** does NOT push the branch to origin until completion is declared
