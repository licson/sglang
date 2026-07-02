## ADDED Requirements

### Requirement: CUDA graph SHALL be auto-disabled when --cpu-offload-gb is set

`server_args.py` validation SHALL automatically set `disable_cuda_graph = True` when `--cpu-offload-gb > 0`. The validation SHALL log a warning when this auto-disable fires, so the user knows why CUDA graph is off.

**Rationale:** The CPU offload forward hook swaps `param.data` on every forward call, but `CudaGraphRunner.replay()` bypasses the Python forward entirely and replays kernels at stale captured addresses. Without the auto-disable, running with `--cpu-offload-gb > 0` and CUDA graph enabled produces garbage outputs or crashes due to stale memory addresses in the captured graph.

#### Scenario: CUDA graph auto-disabled with warning
- **WHEN** the user launches SGLang with `--cpu-offload-gb 8` (or any positive value) and does NOT explicitly set `--disable-cuda-graph`
- **THEN** `server_args.py` validation sets `disable_cuda_graph = True`
- **AND** logs a warning: "CUDA graph is disabled because --cpu-offload-gb is set." (or equivalent message)

#### Scenario: User explicitly disabled CUDA graph
- **WHEN** the user launches SGLang with `--cpu-offload-gb 8 --disable-cuda-graph`
- **THEN** `server_args.py` validation does NOT log a warning (the user already knows)
- **AND** `disable_cuda_graph` remains `True`

#### Scenario: CPU offload not used
- **WHEN** the user launches SGLang without `--cpu-offload-gb` (or with `--cpu-offload-gb 0`)
- **THEN** `server_args.py` validation does NOT auto-disable CUDA graph
- **AND** no warning is logged

### Requirement: --disable-cuda-graph-padding SHALL be rejected when --enable-torch-compile is set

`server_args.py` validation SHALL raise an `AssertionError` (or equivalent hard failure) when both `--disable-cuda-graph-padding` and `--enable-torch-compile` are set. The error message SHALL explain why the combination is incompatible and which flag to remove.

**Rationale:** With padding disabled, every distinct batch size gets its own `torch.compile` + Triton autotune cycle (O(max_batch_size) compilations) instead of the small fixed set of padded bucket sizes. This causes engine initialization to stall for many minutes. The user should pick one flag or the other; allowing both produces a broken deployment.

#### Scenario: Both flags set → hard failure
- **WHEN** the user launches SGLang with `--disable-cuda-graph-padding --enable-torch-compile`
- **THEN** `server_args.py` validation raises an `AssertionError`
- **AND** the error message states that `--disable-cuda-graph-padding` is incompatible with `--enable-torch-compile`
- **AND** the error message mentions the O(max_batch_size) compilation overhead
- **AND** the error message tells the user to remove one flag or the other

#### Scenario: Only --disable-cuda-graph-padding set → no failure
- **WHEN** the user launches SGLang with `--disable-cuda-graph-padding` and NOT `--enable-torch-compile`
- **THEN** `server_args.py` validation passes
- **AND** no incompatibility assertion fires

#### Scenario: Only --enable-torch-compile set → no failure
- **WHEN** the user launches SGLang with `--enable-torch-compile` and NOT `--disable-cuda-graph-padding`
- **THEN** `server_args.py` validation passes
- **AND** no incompatibility assertion fires

### Requirement: HF rope parameters SHALL NOT be manually injected into PretrainedConfig

`PretrainedConfig` construction in `python/sglang/srt/utils/hf_transformers_patches.py` (or the equivalent file in upstream's evolved layout) SHALL NOT manually inject rope parameters (e.g., `rope_scaling`, `rope_theta`) into the config dict before passing it to HuggingFace's `PretrainedConfig.from_dict` (or equivalent). Rope parameters SHALL come from the model's `config.json` or `pretrained_config` directly.

**Rationale:** The fork's pre-sync code manually injected rope parameters into `PretrainedConfig` to work around a HuggingFace transformers bug that has since been fixed upstream. The manual injection conflicts with the upstream fix and produces incorrect rope configurations for models that override `rope_scaling` in their `config.json`.

#### Scenario: Model with rope_scaling in config.json
- **WHEN** a model is loaded whose `config.json` contains `rope_scaling` (e.g., a model with sliding window or NTK-aware rope)
- **THEN** the `rope_scaling` value from `config.json` is used
- **AND** no manual injection of `rope_scaling` occurs in `PretrainedConfig` construction
- **AND** the resulting rope configuration matches what the model author intended

#### Scenario: Model without rope_scaling
- **WHEN** a model is loaded whose `config.json` does NOT contain `rope_scaling`
- **THEN** the default rope configuration is used
- **AND** no manual injection of `rope_scaling` occurs
- **AND** no error is raised

### Requirement: The surviving server_args.py gates SHALL be re-applied against upstream's evolved layout on every sync

When syncing with upstream, the fork's surviving `server_args.py` gates (the `--cpu-offload-gb` CUDA-graph auto-disable and the `--disable-cuda-graph-padding × --enable-torch-compile` assert) SHALL be re-applied against upstream's evolved `server_args.py`. The re-application SHALL find the appropriate location in upstream's evolved validation flow (which may have moved due to upstream churn — 69 commits in 6 weeks on this file).

**Rationale:** `server_args.py` is one of the highest-churn files in upstream (69 commits in 6 weeks before the fork's last sync). Fork-specific gates cannot be expected to auto-merge cleanly across long sync gaps. Each sync must re-apply the gates against the new layout.

#### Scenario: Gates re-applied after upstream churn moved the validation section
- **WHEN** the sync merges upstream/main and `server_args.py` has a conflict (or auto-merges but the validation section has moved)
- **THEN** the maintainer locates the new position of the cpu_offload_gb forward hook validation section
- **AND** re-applies the `if self.cpu_offload_gb > 0: self.disable_cuda_graph = True` block with warning
- **AND** locates the new position of the torch_compile validation section
- **AND** re-applies the `assert not (self.disable_cuda_graph_padding and self.enable_torch-compile)` block with explanatory message

#### Scenario: Gates verified after clean auto-merge
- **WHEN** the sync merges upstream/main and `server_args.py` auto-merges cleanly
- **THEN** the maintainer inspects the merged file for both gates' presence
- **AND** if both are present and correct, no re-application is needed
- **AND** if either is missing or regressed, re-applies it as in the conflict scenario
