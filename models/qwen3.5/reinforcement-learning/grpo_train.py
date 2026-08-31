"""
GRPO training for VGDL generation with Qwen3.5-4B.

The run starts from the supervised LoRA adapter and optimizes the same
structural and executability rewards used by the Qwen pipeline. It relies on
the standard Transformers/PEFT stack, which is more stable than the Unsloth
path for the Qwen3.5 multimodal architecture in this environment.

Run from the repository root:
  python models/qwen3.5/reinforcement-learning/grpo_train.py

Dry run:
  $env:GRPO_DRY_RUN="1"
  python models/qwen3.5/reinforcement-learning/grpo_train.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# TRL can discover optional judge packages that are incompatible with the
# installed Transformers version, even though GRPO does not use them.
import transformers.utils.hub as _transformers_hub  # noqa: E402

if not hasattr(_transformers_hub, "TRANSFORMERS_CACHE"):
    _transformers_hub.TRANSFORMERS_CACHE = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
    )

import trl.import_utils as _trl_import_utils  # noqa: E402

for _availability_flag in (
    "_llm_blender_available",
    "_mergekit_available",
    "_weave_available",
):
    if hasattr(_trl_import_utils, _availability_flag):
        setattr(_trl_import_utils, _availability_flag, False)

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "py-vgdl"))

from reward_functions import REWARD_FUNCTIONS  # noqa: E402


BASE_MODEL = "Qwen/Qwen3.5-4B"
SFT_ADAPTER = "models/qwen3.5/supervised-learning/16-32-0.05"
OUTPUT_DIR = os.environ.get(
    "QWEN_GRPO_OUTPUT_DIR",
    "models/qwen3.5/reinforcement-learning/grpo-output",
)
SAVE_LAST_EVERY_STEPS = 5

SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)


def copy_checkpoint(checkpoint: str, destination_name: str) -> None:
    """Copy a complete Trainer checkpoint to a stable, user-facing name."""
    if not checkpoint or not os.path.isdir(checkpoint):
        return
    destination = os.path.join(OUTPUT_DIR, destination_name)
    shutil.copytree(checkpoint, destination, dirs_exist_ok=True)
    print(f"{destination_name} updated from: {checkpoint}")


class LastCheckpointCallback(TrainerCallback):
    """Persist a resumable copy of the most recent checkpoint."""

    def on_step_end(self, args, state, control, **kwargs):
        if (
            state.global_step > 0
            and state.global_step % SAVE_LAST_EVERY_STEPS == 0
        ):
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )
        copy_checkpoint(checkpoint, "last-model")
        return control


def load_qwen_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.5 GRPO training.")

    print(f"Loading processor from {SFT_ADAPTER}...")
    processor = AutoProcessor.from_pretrained(SFT_ADAPTER)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {BASE_MODEL} in 4-bit...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    base_model.config.use_cache = False
    # Do not call prepare_model_for_kbit_training here: it casts the custom
    # Qwen3.5 Conv1d layers to float32, while the model expects bfloat16
    # activations during GRPO generation.

    print(f"Loading trainable SFT adapter from {SFT_ADAPTER}...")
    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER,
        is_trainable=True,
        low_cpu_mem_usage=True,
    )
    model.train()
    model.config.use_cache = False
    model.enable_input_require_grads()

    # GRPOTrainer uses this dictionary only to suppress a token-count warning.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model, processor, tokenizer


def load_dataset(processor):
    print("Loading dataset...")
    dataset = load_from_disk("dataset_hf")

    def format_prompt(example):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["description"].strip()}
                ],
            },
        ]
        return {
            "prompt": processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        }

    dataset = dataset.map(
        format_prompt,
        remove_columns=["description", "vgdl"],
    )
    print(f"Train size: {len(dataset['train'])} | Test size: {len(dataset['test'])}")
    return dataset


def build_grpo_config() -> GRPOConfig:
    return GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        num_generations=2,
        max_prompt_length=384,
        max_completion_length=256,
        temperature=0.8,
        beta=0.04,
        use_vllm=False,
    )


def find_resume_checkpoint() -> str | None:
    last_model_dir = os.path.join(OUTPUT_DIR, "last-model")
    required_files = (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    )
    if os.path.isdir(last_model_dir) and all(
        os.path.isfile(os.path.join(last_model_dir, filename))
        for filename in required_files
    ):
        return last_model_dir
    if os.path.isdir(OUTPUT_DIR):
        return get_last_checkpoint(OUTPUT_DIR)
    return None


def main() -> None:
    model, processor, tokenizer = load_qwen_model()
    dataset = load_dataset(processor)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCTIONS,
        args=build_grpo_config(),
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        callbacks=[LastCheckpointCallback()],
    )

    if os.environ.get("GRPO_DRY_RUN") == "1":
        print("Qwen3.5 GRPO dry run complete. Training source: SFT adapter.")
        return

    print("Starting Qwen3.5 GRPO training...")
    resume_checkpoint = find_resume_checkpoint()
    if resume_checkpoint:
        print(f"Resuming training from: {resume_checkpoint}")
        if os.path.normpath(resume_checkpoint) != os.path.normpath(
            os.path.join(OUTPUT_DIR, "last-model")
        ):
            copy_checkpoint(resume_checkpoint, "last-model")
    else:
        print("No checkpoint found: training starts from the SFT adapter.")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        copy_checkpoint(last_checkpoint, "last-model")

    metrics = {
        "train": train_result.metrics,
        "history": trainer.state.log_history,
        "started_from_sft_adapter": True,
        "sft_adapter": SFT_ADAPTER,
        "last_checkpoint": last_checkpoint,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "grpo_training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"GRPO training complete. Model saved to {OUTPUT_DIR}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
