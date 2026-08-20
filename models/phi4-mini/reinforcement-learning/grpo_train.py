"""
GRPO training for VGDL generation with Phi-4-mini-instruct.

The script mirrors the Gemma4 GRPO pipeline but adapts the model handling to
Phi:
  - AutoTokenizer instead of AutoProcessor
  - native Phi3ForCausalLM loading, no trust_remote_code by default
  - fp16, because RTX 20xx cards do not support bf16 efficiently
  - continue training from the supervised Phi LoRA adapter

Run from the repository root:
  python models/phi4-mini/reinforcement-learning/grpo_train.py

Dry run:
  $env:GRPO_DRY_RUN="1"
  python models/phi4-mini/reinforcement-learning/grpo_train.py
"""

from __future__ import annotations

import json
import importlib.util
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# TRL can import optional judge dependencies that still expect this symbol.
import transformers.utils.hub as _transformers_hub  # noqa: E402

if not hasattr(_transformers_hub, "TRANSFORMERS_CACHE"):
    _transformers_hub.TRANSFORMERS_CACHE = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
    )

# Compatibilita' TRL 0.24 in questo ambiente: `mergekit` viene rilevato come
# installato, ma manca `mergekit.config`. GRPO non usa le funzioni di merge.
import trl.import_utils as _trl_import_utils  # noqa: E402

if hasattr(_trl_import_utils, "_mergekit_available"):
    _trl_import_utils._mergekit_available = False
if hasattr(_trl_import_utils, "_llm_blender_available"):
    _trl_import_utils._llm_blender_available = False
if hasattr(_trl_import_utils, "_weave_available"):
    _trl_import_utils._weave_available = False

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
_PHI_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "py-vgdl"))
sys.path.insert(0, str(_PHI_DIR))

from prompting import build_prompt  # noqa: E402
from reward_functions import REWARD_FUNCTIONS  # noqa: E402


# ========================
# Config
# ========================

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
SFT_ADAPTER = "models/phi4-mini/supervised-learning/model-finetuned"
OUTPUT_DIR = os.environ.get(
    "PHI_GRPO_OUTPUT_DIR",
    "models/phi4-mini/reinforcement-learning/grpo-output-sft",
)
MAX_SEQ_LEN = 1024
SAVE_LAST_EVERY_STEPS = 5
TRUST_REMOTE_CODE = False

# Conservative defaults for consumer GPUs. On 6 GB VRAM this may still be too
# tight for real training; use dry-run first and prefer 12-16 GB VRAM for GRPO.
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_GENERATIONS = 2
MAX_PROMPT_LENGTH = 384
MAX_COMPLETION_LENGTH = 256

def copy_checkpoint(checkpoint: str, destination_name: str) -> None:
    """Copy a full checkpoint into a stable folder name."""
    if not checkpoint or not os.path.isdir(checkpoint):
        return
    destination = os.path.join(OUTPUT_DIR, destination_name)
    shutil.copytree(checkpoint, destination, dirs_exist_ok=True)
    print(f"{destination_name} updated from: {checkpoint}")


class BestLastCheckpointCallback(TrainerCallback):
    """Keep last-model and best-model folders updated."""

    def on_step_end(self, args, state, control, **kwargs):
        if (
            state.global_step > 0
            and state.global_step % SAVE_LAST_EVERY_STEPS == 0
        ):
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        current_checkpoint = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )
        copy_checkpoint(current_checkpoint, "last-model")
        if state.best_model_checkpoint:
            copy_checkpoint(state.best_model_checkpoint, "best-model")
        return control


def has_adapter(adapter_dir: str) -> bool:
    return os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))


def validate_output_provenance() -> None:
    """Refuse to resume the legacy GRPO run that did not start from SFT."""
    metrics_path = os.path.join(OUTPUT_DIR, "grpo_training_metrics.json")
    if not os.path.isfile(metrics_path):
        return
    try:
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot verify GRPO output provenance from '{metrics_path}': {exc}"
        ) from exc

    if metrics.get("started_from_sft_adapter") is False:
        raise RuntimeError(
            f"'{OUTPUT_DIR}' contains a legacy GRPO run that started from the "
            "base model. Choose an empty PHI_GRPO_OUTPUT_DIR for the SFT-based run."
        )


def load_phi_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: Phi GRPO requires a GPU.")

    if not has_adapter(SFT_ADAPTER):
        raise FileNotFoundError(
            f"Required Phi SFT adapter not found in '{SFT_ADAPTER}'. "
            "Run models/phi4-mini/supervised-learning/finetune.py first."
        )

    print(f"Loading tokenizer from {SFT_ADAPTER}...")
    tokenizer = AutoTokenizer.from_pretrained(
        SFT_ADAPTER,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {BASE_MODEL} in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation="eager",
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
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

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


def load_dataset():
    print("Loading dataset...")
    dataset = load_from_disk("dataset_hf")
    dataset = dataset.filter(
        lambda example: bool(example["description"].strip())
        and bool(example["vgdl"].strip()),
        desc="Removing empty prompt/reference pairs",
    )

    def format_prompt(example):
        return {"prompt": build_prompt(example["description"])}

    # Do not remove 'vgdl': reward_similarity receives it as a kwarg.
    dataset = dataset.map(format_prompt)
    print(f"Train size: {len(dataset['train'])} | Test size: {len(dataset['test'])}")
    return dataset


def build_grpo_config():
    return GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=False,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        # TRL 0.24 logs eval_reward, but in this environment Transformers does
        # not receive it in the metrics dict used by load_best_model_at_end.
        # Keep epoch evaluation, but avoid crashing when selecting best model.
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        save_total_limit=2,
        report_to="none",
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.8,
        beta=0.04,
        use_vllm=False,
    )


def find_resume_checkpoint() -> str | None:
    last_model_dir = os.path.join(OUTPUT_DIR, "last-model")
    required_resume_files = (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    )
    last_model_is_complete = os.path.isdir(last_model_dir) and all(
        os.path.isfile(os.path.join(last_model_dir, filename))
        for filename in required_resume_files
    )
    if last_model_is_complete:
        return last_model_dir
    if os.path.isdir(OUTPUT_DIR):
        return get_last_checkpoint(OUTPUT_DIR)
    return None


def main() -> None:
    if importlib.util.find_spec("pygame") is None:
        print(
            "WARN: pygame is not installed. py-vgdl executability reward will "
            "fall back to partial structural credit until pygame is installed."
        )

    validate_output_provenance()
    model, tokenizer = load_phi_model()
    dataset = load_dataset()
    grpo_config = build_grpo_config()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCTIONS,
        args=grpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=[BestLastCheckpointCallback()],
    )

    if os.environ.get("GRPO_DRY_RUN") == "1":
        print("Phi GRPO dry run complete. Training source: SFT adapter.")
        return

    print("Starting Phi-4-mini GRPO training...")
    last_checkpoint = find_resume_checkpoint()
    if last_checkpoint:
        print(f"Resuming training from: {last_checkpoint}")
        last_model_dir = os.path.join(OUTPUT_DIR, "last-model")
        if os.path.normpath(last_checkpoint) != os.path.normpath(last_model_dir):
            copy_checkpoint(last_checkpoint, "last-model")
    else:
        print("No checkpoint found: training starts from current adapter.")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    best_checkpoint = trainer.state.best_model_checkpoint
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if best_checkpoint:
        best_output = os.path.join(OUTPUT_DIR, "best-model")
        shutil.copytree(best_checkpoint, best_output, dirs_exist_ok=True)
        print(f"Best checkpoint (eval_reward): {best_checkpoint}")
        print(f"Best model copied to: {best_output}")
    else:
        print("WARN: no best checkpoint available.")

    if last_checkpoint:
        last_output = os.path.join(OUTPUT_DIR, "last-model")
        shutil.copytree(last_checkpoint, last_output, dirs_exist_ok=True)
        print(f"Last checkpoint: {last_checkpoint}")
        print(f"Last model copied to: {last_output}")

    metrics = {
        "train": train_result.metrics,
        "history": trainer.state.log_history,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "best_metric": trainer.state.best_metric,
        "started_from_sft_adapter": True,
        "sft_adapter": SFT_ADAPTER,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "grpo_training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"GRPO training complete. Model saved to {OUTPUT_DIR}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
