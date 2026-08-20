# Phi-4-mini Reinforcement Learning

This folder contains the GRPO setup for `microsoft/Phi-4-mini-instruct`.

It mirrors the Gemma4 reinforcement-learning folder but adapts the model
loading path to Phi:

- `AutoTokenizer` instead of `AutoProcessor`
- native Transformers Phi loading, no `trust_remote_code` by default
- 4-bit QLoRA with fp16 compute
- requires `models/phi4-mini/supervised-learning/model-finetuned`
- continues from that trainable SFT adapter instead of creating a fresh LoRA
- writes SFT-started runs to a separate output directory

## Files

- `grpo_train.py`: GRPO training script
- `reward_functions.py`: reward functions for VGDL validity and similarity

## Required SFT Stage

Run the supervised stage first:

```powershell
conda activate rlia
python models/phi4-mini/supervised-learning/finetune.py --dry-run
python models/phi4-mini/supervised-learning/finetune.py
```

GRPO now fails with a clear error if the SFT adapter is missing. This prevents
an apparently valid run from silently starting at the base model.

## Dry Run

Run from the repository root:

```powershell
conda activate rlia
$env:GRPO_DRY_RUN="1"
python models/phi4-mini/reinforcement-learning/grpo_train.py
```

The GRPO dry run loads the tokenizer, the Phi base model in 4-bit, the trainable
SFT adapter, maps `dataset_hf`, and initializes `GRPOTrainer`. It therefore has
to be run after the SFT stage and still needs GPU memory.

## py-vgdl Dependency

The reward functions can give full executability reward only if `py-vgdl` can
import `pygame`. Verify it before real training:

```powershell
conda activate rlia
python -c "import pygame; print(pygame.version.ver)"
```

## Training

```powershell
conda activate rlia
Remove-Item Env:GRPO_DRY_RUN -ErrorAction SilentlyContinue
python models/phi4-mini/reinforcement-learning/grpo_train.py
```

Output is saved to:

```text
models/phi4-mini/reinforcement-learning/grpo-output-sft
```

The existing `grpo-output` directory is retained as a legacy experiment: its
metrics report `started_from_sft_adapter=false`, so it must not be resumed for
the new SFT-to-GRPO pipeline. Set `PHI_GRPO_OUTPUT_DIR` only when intentionally
starting or resuming another compatible SFT-based GRPO run.

## Inference

```powershell
python models/phi4-mini/reinforcement-learning/inference/vgdl_gen_phi4mini_grpo.py `
  --description-file dataset/descriptions/14_sokoban.txt
```

The default adapter is `grpo-output-sft/last-model`. The inference prompt is the
same raw prompt used by SFT and GRPO training.

## Hardware Note

The local RTX 2060 has 6 GB VRAM. The script is conservative, but full GRPO can
still exceed memory. If it fails with CUDA OOM, run on a GPU with at least
12-16 GB VRAM or reduce `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, and keep
`NUM_GENERATIONS=2`.
