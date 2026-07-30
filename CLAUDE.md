# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DiffSynth-Studio is an open-source Diffusion-model engine maintained by the ModelScope community
(`pip` package name `diffsynth`). It provides a unified inference (`Pipeline`) and training
(`DiffusionTrainingModule`) framework for ~20 different image/video/audio diffusion model families
(FLUX, FLUX.2, Qwen-Image, Wan-Video, Z-Image, Stable Diffusion/XL, HiDream-O1, LTX-2, ACE-Step,
Krea-2, JoyAI-Image, ERNIE-Image, Ideogram-4, Boogu-Image, Anima, MOVA, etc.), each with its own
`Pipeline` subclass and model architecture files but sharing common core infrastructure for model
loading, VRAM management, LoRA, and flow-matching/diffusion training.

In this workspace it is vendored as a git submodule under `third_party/DiffSynth-Studio` of the
`ego-moma` repo (own git dir at `../../.git/modules/third_party/DiffSynth-Studio`, remote
`HanzhiC/DiffSynth-Studio`, a fork of `modelscope/DiffSynth-Studio`) — treat it as a semi-independent
project with its own install/versioning, not as part of `ego-moma`'s own `src/`.

There are two sibling ModelScope projects: **DiffSynth-Studio** (this repo — aggressive technical
exploration, academia-facing, cutting-edge model support) and **DiffSynth-Engine** (a different repo
— stable industrial deployment). Don't confuse features/docs between them.

## Install

```bash
pip install -e .                     # editable install, or:
pip install -e ".[audio]"            # + torchaudio/torchcodec/librosa for audio models (ACE-Step, etc.)
pip install -e ".[all]"              # + streamlit (WebUI)
```

Requires Python ≥3.10.1 and a CUDA GPU for anything beyond trivial CPU testing (`accelerator`/device
handling throughout assumes CUDA; some npu-specific extras exist — `.[npu]`/`.[npu_aarch64]` — for
Ascend NPU but are not the default path). Do not add version pins for `torch`/`numpy`/`transformers`/
etc. beyond what `pyproject.toml` already specifies unless a specific model requires it.

Models are auto-downloaded (via `modelscope` by default) into `./models` on first use — set
`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` or `os.environ["MODELSCOPE_ENDPOINT"]` before `import
diffsynth` to change source/region; see `docs/en/Pipeline_Usage/Environment_Variables.md` for the
full list (`DIFFSYNTH_MODEL_BASE_PATH`, `DIFFSYNTH_SKIP_DOWNLOAD`, `DIFFSYNTH_ATTENTION_IMPLEMENTATION`,
`DIFFSYNTH_DISK_MAP_BUFFER_SIZE`). Environment variables must be set **before** `import diffsynth`.

## Common commands

There is no formal pytest-based test suite. `examples/` doubles as both usage documentation and the
project's integration-test corpus, one subdirectory per model family (e.g. `examples/qwen_image/`,
`examples/wanvideo/`, `examples/flux/`, `examples/z_image/`). Each model directory follows the same
layout:

```
examples/<model>/model_inference/<Variant>.py            # single-file inference example
examples/<model>/model_inference_low_vram/<Variant>.py   # same, with VRAM management enabled
examples/<model>/model_training/train.py                 # shared training entry point (argparse)
examples/<model>/model_training/full/<Variant>.sh         # full-finetune launch command
examples/<model>/model_training/lora/<Variant>.sh         # LoRA launch command
examples/<model>/model_training/validate_full/<Variant>.py
examples/<model>/model_training/validate_lora/<Variant>.py
```

Run any example directly, e.g.:
```bash
python examples/qwen_image/model_inference/Qwen-Image.py
accelerate launch examples/qwen_image/model_training/train.py --dataset_base_path ... --lora_rank 32 ...
```
Training scripts are `accelerate launch`-based and expose their config as CLI flags (argparse, not
`options.py`-style dotted overrides) — read the corresponding `.sh` for the flags a given model
expects, and `examples/<model>/model_training/train.py` for what they mean.

`examples/dev_tools/unit_test.py` is the closest thing to a test runner: it batch-runs every
inference/training example for a given model family across multiple GPUs and diffs against expected
outputs manually. Not wired into CI; run/adapt it manually when doing broad regression sweeps
(`test_qwen_image()`, `test_wan()`, `test_flux()`, `test_z_image()` — edit `__main__` to pick one).

`examples/dev_tools/webui.py` / `webui_train.py` launch the (streamlit-based, `.[all]` extra)
inference/training WebUI — see `docs/en/Pipeline_Usage/Inference_WebUI.md`.

Documentation lives in `docs/en/` (authoritative) and `docs/zh/` (Chinese mirror, keep both in sync
when editing docs) and is also published at readthedocs. `README.md` / `README_zh.md` at repo root
are the long-form model catalogue + changelog (very large; grep rather than read in full).

## Architecture

Every model family plugs into the same three-layer structure: **models** (raw `nn.Module`
architectures) → **configs** (registry mapping a state-dict hash to a model class + loader) →
**pipelines** (orchestration: preprocessing units + iterative denoising + `model_fn`). Training reuses
the pipeline almost unmodified via `DiffusionTrainingModule`. Read
`docs/en/Developer_Guide/{Integrating_Your_Model,Building_a_Pipeline,Training_Diffusion_Models,Enabling_VRAM_management}.md`
before adding a new model — they're short, precise, and describe the exact conventions below with
copy-pasteable templates.

- **`diffsynth/models/`** — one `.py` file per model architecture (e.g. `qwen_image_dit.py`,
  `flux_dit.py`, `wan_video_dit.py`-equivalents). Plain `torch.nn.Module` subclasses with no
  framework coupling; HuggingFace-ecosystem models are reimplemented by hand (config copy-pasted
  into code) rather than loaded via `from_pretrained`, because VRAM management requires controlling
  parameter materialization directly. `general_modules.py` holds shared building blocks;
  `model_loader.py` (`ModelPool`) is what actually resolves + instantiates a model from a state dict.
- **`diffsynth/configs/model_configs.py`** — the model registry. Each entry maps a `model_hash`
  (hash of a state dict's keys+shapes, from `hash_model_file`) to `model_name` (the logical role a
  `Pipeline` looks up, e.g. `"qwen_image_dit"`), `model_class`, optional `state_dict_converter`, and
  optional `extra_kwargs`. Multiple configs can share a `model_hash` when one file contains multiple
  logical models. `diffsynth/configs/vram_management_module_maps.py` is the parallel registry of
  default per-layer VRAM-management wrapping (`module_map`) for each model class.
- **`diffsynth/utils/state_dict_converters/`** — per-model key-renaming/reshaping logic, used when a
  community checkpoint's on-disk state dict doesn't already match the framework's expected key
  names (deliberately never re-uploads/repackages third-party checkpoints; converts at load time
  instead, to preserve original model authors' download attribution).
- **`diffsynth/pipelines/`** — one file per model family (`qwen_image.py`, `flux_image.py`,
  `wan_video.py`, `z_image.py`, …), each a `BasePipeline` (`diffsynth/diffusion/base_pipeline.py`)
  subclass with a fixed shape: `__init__` (declares `scheduler`, model attributes as typed `None`
  placeholders, `in_iteration_models`, `units`, `model_fn`), `from_pretrained` (calls
  `download_and_load_models` then `model_pool.fetch_model(name)` per attribute), `__call__` (runs
  `units` to build `inputs_shared`/`inputs_posi`/`inputs_nega`, then the scheduler timestep loop
  calling `model_fn`, then VAE decode), and `units`/`model_fn` as described below.
  - **`PipelineUnit`** (preprocessing steps run before the denoising loop): three modes — *direct*
    (CFG-independent, e.g. noise init), *`seperate_cfg=True`* (same computation run once per
    posi/nega side with different inputs, e.g. prompt encoding), *`take_over=True`* (full access to
    all three input dicts, for cross-cutting concerns like entity/region control). New units should
    prefer direct mode; trigger optional behavior on "was this input param passed" (`is not None`),
    not on "is the optional model loaded" — loading a control model but not passing its input (or
    vice versa) should error, not silently no-op. Always gate VRAM-heavy work behind
    `pipe.load_models_to_device(self.onload_model_names)` and never manually offload afterward.
  - **`model_fn`**: the unified per-timestep forward call signature across all units/loop code;
    trivial for architecturally clean models (`dit(latents, prompt_emb, timestep)`), but is where
    ControlNet branches, gradient checkpointing, and other cross-model glue accumulate for
    ecosystem-heavy models (see `diffsynth/pipelines/qwen_image.py` for the complex end of that
    spectrum).
- **`diffsynth/diffusion/`** — cross-model diffusion/flow-matching machinery: `flow_match.py`
  (`FlowMatchScheduler`), `ddim_scheduler.py`, `dmd2.py`, `loss.py` (`FlowMatchSFTLoss`,
  `DirectDistillLoss`, etc.), `training_module.py` (`DiffusionTrainingModule` — the base class every
  `examples/<model>/model_training/train.py` subclasses; handles LoRA/full-param switching, resuming,
  gradient checkpointing wiring), `template.py` (Diffusion Templates plugin framework — see
  `docs/en/Diffusion_Templates/`), `runner.py`, `logger.py`, `parsers.py`.
- **`diffsynth/core/`** — infrastructure with no model-specific knowledge, re-exported flat via
  `diffsynth/__init__.py` → `diffsynth/core/__init__.py`:
  - `vram/` — the VRAM management engine (`enable_vram_management`, `AutoWrappedLinear`,
    `AutoWrappedModule`, disk-offload support). Wraps parameter-bearing layers so weights can live
    offloaded (CPU RAM or disk) and stream to GPU only around their own forward call; this is what
    lets 20B+ param models run on consumer GPUs and is the single most load-bearing subsystem in the
    codebase — see `docs/en/Pipeline_Usage/VRAM_management.md` before touching model-loading paths.
  - `loader/` — state-dict loading, `hash_model_file`, `skip_model_initialization` (meta-device init
    to avoid allocating real tensors before `load_state_dict(..., assign=True)`), `ModelPool`.
  - `data/` — `UnifiedDataset` and dataset "operators" (`diffsynth/core/data/operators.py`) used by
    every `train.py`.
  - `attention/` — pluggable attention backends (flash-attention 2/3, sage attention, xformers,
    torch SDPA), selected via `DIFFSYNTH_ATTENTION_IMPLEMENTATION`.
  - `gradient/` — `gradient_checkpoint_forward` helper used inside `model_fn`s.
  - `offload_training/` — CPU-offload training support (layer-by-layer CPU↔GPU streaming during
    backprop, distinct from inference-time VRAM management).
  - `device/`, `npu_patch/` — device dispatch and Ascend NPU compatibility shims.
- **`diffsynth/metrics/`** — standalone image-quality metrics (FID, CLIP, Aesthetic, PickScore,
  ImageReward, HPSv2/v3, LPIPS, UnifiedReward) with matching model files in `diffsynth/models/`; not
  wired into the training loop automatically, used for offline evaluation (see
  `examples/image_quality_metric/`).
- **`diffsynth/utils/`** — grab-bag of shared helpers beyond `state_dict_converters/`: `lora/`
  (LoRA application, cold-fuse vs. hot-load — see `docs/en/QA.md`'s LoRA loading Q&A),
  `controlnet/`, `tile/` (tiled inference for large images/video), `dequantizer/`, `demucs/`
  (audio source separation for ACE-Step), `ses/`.

## Working conventions specific to this repo

- Training does not support `batch_size > 1` by design (not a missing feature) — variable-length
  text/variable-resolution images across a batch don't merge cleanly, and multi-GPU + gradient
  accumulation already cover the effective-batch-size use case. See `docs/en/QA.md`.
- Some models retain architecturally unused parameters (upstream bugs preserved for checkpoint
  compatibility); multi-GPU training on those needs `--find_unused_parameters` on the training
  script rather than "fixing" the model to drop the parameter.
- FP8 in this codebase means FP8-precision *storage* with on-the-fly upcast for compute (a VRAM
  optimization), never native FP8 compute/training — don't assume acceleration from enabling it.
- When integrating a new model, follow the 5-step flow in
  `docs/en/Developer_Guide/Integrating_Your_Model.md` in order (architecture → state-dict converter
  → `model_configs.py` entry → load-verification snippet → VRAM management module map) rather than
  wiring a pipeline first — the pipeline layer assumes the model/config layers already resolve.
- `docs/en/` and `docs/zh/` are parallel trees with matching filenames; a doc change affecting
  behavior usually needs both, though this repo's English docs are the ones to trust first when they
  disagree (per the ModelScope team's own stated workflow of English-first authoring for this repo).
