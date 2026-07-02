## ADDED Requirements

### Requirement: MoeWNA16Method SHALL accept both SiLU and GELU activations

`MoeWNA16Method` (in `python/sglang/srt/layers/quantization/moe_wna16.py`) SHALL accept `"silu"` and `"gelu"` as valid `moe_runner_config.activation` values. The method SHALL raise an `AssertionError` with a message naming the unsupported activation for any other value. The assertion message SHALL include the actual value received for debuggability.

**Rationale:** Gemma-4 uses GELU in its MoE experts, but the fork's pre-sync `MoeWNA16Method` asserted `activation == "silu"`, causing an `AssertionError` when running Gemma-4 with Intel AutoRound INT4 quantization. The underlying Triton MoE kernels already support both `silu` and `gelu`, so the original assertion was overly restrictive.

#### Scenario: SiLU activation accepted
- **WHEN** `MoeWNA16Method` is constructed with `moe_runner_config.activation == "silu"`
- **THEN** `run` proceeds without assertion error
- **AND** the Triton MoE kernel is dispatched in silu mode

#### Scenario: GELU activation accepted
- **WHEN** `MoeWNA16Method` is constructed with `moe_runner_config.activation == "gelu"`
- **THEN** `run` proceeds without assertion error
- **AND** the Triton MoE kernel is dispatched in gelu mode

#### Scenario: Unsupported activation rejected with informative message
- **WHEN** `MoeWNA16Method` is constructed with `moe_runner_config.activation == "relu"` (or any unsupported value)
- **THEN** `run` raises `AssertionError`
- **AND** the error message includes both the unsupported value (`"relu"`) and the list of supported values (`"silu"`, `"gelu"`)

### Requirement: NVFP4 Cutlass MoE fallback SHALL remap global expert IDs to local IDs when EP > 1

The NVFP4 Cutlass MoE fallback path (in `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` or its refactored equivalent) SHALL remap global expert IDs to local IDs before dispatching to the per-rank Cutlass kernel when expert parallelism (EP) size is greater than 1.

**Rationale:** Without remapping, global expert IDs sent to the per-rank Cutlass kernel index out-of-bounds on ranks where the local expert count is less than the global expert count. This was observed as a crash when running NVFP4-quantized MoE models with EP > 1.

#### Scenario: EP=1 (no remapping needed)
- **WHEN** a NVFP4 Cutlass MoE forward is executed with expert parallelism size == 1
- **THEN** the global expert IDs are passed directly to the Cutlass kernel
- **AND** no crash occurs

#### Scenario: EP>1 with remapping
- **WHEN** a NVFP4 Cutlass MoE forward is executed with expert parallelism size > 1
- **THEN** global expert IDs are remapped to local IDs before dispatching to the Cutlass kernel
- **AND** no out-of-bounds crash occurs

### Requirement: NVFP4 Cutlass MoE SHALL keep input global scales scalar

The NVFP4 MoE `flashinfer_trtllm` path SHALL keep the input global scales as a scalar (per-tensor) quantity, not broadcast to a per-row or per-block shape. The path SHALL NOT reshape or expand the global scales tensor before passing it to the flashinfer_trtllm kernel.

**Rationale:** The flashinfer_trtllm kernel expects scalar global scales; reshaping to a per-row shape caused a crash. The fork's pre-sync code accidentally broadcast the scales, which was observed as a runtime crash when running NVFP4-quantized MoE models.

#### Scenario: Scalar global scales passed through unchanged
- **WHEN** the NVFP4 MoE `flashinfer_trtllm` path is executed
- **THEN** the input global scales tensor is passed to the kernel with its original scalar shape
- **AND** no reshape, expand, or broadcast is applied
- **AND** no crash occurs

### Requirement: GPTQ Marlin MoE SHALL support bf16 activation scale dtype

The GPTQ Marlin MoE kernel (in `python/sglang/srt/layers/quantization/gptq/schemes/` after the upstream PR #26402 split, or wherever the Marlin scheme lives in the new layout) SHALL accept `bfloat16` scale dtype without raising an `AssertionError`. The kernel SHALL NOT hardcode `float16` as the only supported scale dtype.

**Rationale:** The fork's pre-sync GPTQ Marlin MoE hardcoded `fp16` scale dtype, causing an `AssertionError` when a model had `bf16` activation scales (a common configuration for modern models). The Marlin kernel itself supports both dtypes; the assertion was overly restrictive.

#### Scenario: fp16 scale dtype accepted
- **WHEN** GPTQ Marlin MoE is invoked with a model using `fp16` activation scales
- **THEN** the kernel dispatches without assertion error

#### Scenario: bf16 scale dtype accepted
- **WHEN** GPTQ Marlin MoE is invoked with a model using `bf16` activation scales
- **THEN** the kernel dispatches without assertion error
- **AND** no hardcoded `fp16` assertion fires

### Requirement: AutoRound GPTQ MoE quantization SHALL dispatch to the correct marlin/non-marlin branch

AutoRound GPTQ MoE quantization (in the new `gptq/schemes/` layout under the AutoRound scheme file) SHALL dispatch to the marlin or non-marlin branch based on the model's actual marlin-eligibility, not a swapped condition. The branch-selection logic SHALL be the inverse of the pre-fix fork's logic (which had marlin and non-marlin branches swapped).

**Rationale:** The fork's pre-sync AutoRound GPTQ MoE had the marlin and non-marlin branches swapped, causing models that should use marlin to use the non-marlin path (and vice versa). This produced incorrect quantization results or crashes depending on the model.

#### Scenario: Marlin-eligible model uses marlin branch
- **WHEN** AutoRound GPTQ MoE is invoked with a marlin-eligible model
- **THEN** the marlin branch is taken
- **AND** the non-marlin branch is NOT taken

#### Scenario: Non-marlin model uses non-marlin branch
- **WHEN** AutoRound GPTQ MoE is invoked with a non-marlin model
- **THEN** the non-marlin branch is taken
- **AND** the marlin branch is NOT taken

### Requirement: GPTQ MoE desc_act SHALL be set in the non-marlin config when required

The GPTQ MoE non-marlin config SHALL set `desc_act` to the value required by the model's quantization config, not unconditionally default it. The config SHALL read `desc_act` from the model config when present.

**Rationale:** The fork's pre-survey of the GPTQ MoE non-marlin config path found `desc_act` was not being propagated from the model config, causing incorrect activation ordering for models that require descending activation quantization.

#### Scenario: Model requires desc_act=True
- **WHEN** GPTQ MoE non-marlin config is constructed for a model with `desc_act=True` in its quantization config
- **THEN** the resulting config has `desc_act=True`

#### Scenario: Model requires desc_act=False
- **WHEN** GPTQ MoE non-marlin config is constructed for a model with `desc_act=False` (or unspecified) in its quantization config
- **THEN** the resulting config has `desc_act=False`

### Requirement: GPTQ/AWQ MoE SHALL not crash with TP>1 and non-auto moe-runner backend

GPTQ and AWQ MoE quantization SHALL work correctly with tensor parallelism size > 1 when the `moe-runner` backend is explicitly set to a non-auto value (e.g., `--moe-runner triton` or `--moe-runner flashinfer_trtllm`). The dispatch logic SHALL NOT assume `moe-runner=auto` when TP>1.

**Rationale:** The fork's pre-sync GPTQ/AWQ MoE dispatch logic had an implicit assumption that `moe-runner=auto` was set when TP>1, causing a crash when a user explicitly set a non-auto runner. This was observed as a runtime error when running GPTQ-quantized MoE models with `--tensor-parallel-size 4 --moe-runner triton`.

#### Scenario: TP=1 with auto moe-runner
- **WHEN** GPTQ MoE is invoked with `--tensor-parallel-size 1` and no explicit `--moe-runner`
- **THEN** the auto-runner dispatch path is taken
- **AND** no crash occurs

#### Scenario: TP>1 with explicit non-auto moe-runner
- **WHEN** GPTQ MoE is invoked with `--tensor-parallel-size 4 --moe-runner triton` (or any other non-auto runner)
- **THEN** the explicitly-selected runner is used
- **AND** no crash occurs
- **AND** the dispatch does not fall back to auto-runner
