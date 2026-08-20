"""Supervised fine-tuning for VGDL generation with Phi-4-mini-instruct.

The script trains a 4-bit LoRA adapter and saves it where the Phi GRPO stage
expects it:

    models/phi4-mini/supervised-learning/model-finetuned

Run ``python models/phi4-mini/supervised-learning/finetune.py --dry-run``
before starting the full training job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

_THIS_DIR = Path(__file__).resolve().parent
_PHI_DIR = _THIS_DIR.parent
_REPO_ROOT = _PHI_DIR.parents[1]
sys.path.insert(0, str(_PHI_DIR))

from prompting import build_prompt


BASE_MODEL = "microsoft/Phi-4-mini-instruct"
DEFAULT_DATASET_PATH = _REPO_ROOT / "dataset_hf"
DEFAULT_OUTPUT_DIR = _THIS_DIR / "model-finetuned"
DEFAULT_MAX_LENGTH = 2048
LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA supervised fine-tuning of Phi-4-mini for VGDL generation."
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--keep-overlength",
        action="store_true",
        help="Keep sequences over --max-length and let TRL truncate them.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="Use 'auto', 'none', or an explicit Transformers checkpoint path.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow a fresh run to replace an existing final adapter.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="eager",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and tokenize the dataset without loading the model.",
    )
    return parser.parse_args()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_dataset(
    dataset_path: Path,
    tokenizer,
    max_length: int,
    keep_overlength: bool,
) -> tuple[DatasetDict, dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_dataset = load_from_disk(str(dataset_path))
    required_splits = ("train", "test")
    missing_splits = set(required_splits).difference(raw_dataset.keys())
    if missing_splits:
        raise ValueError(f"Missing dataset split(s): {sorted(missing_splits)}")

    original_sizes = {split: len(raw_dataset[split]) for split in required_splits}

    def has_complete_pair(example: dict[str, Any]) -> bool:
        return bool(example.get("description", "").strip()) and bool(
            example.get("vgdl", "").strip()
        )

    dataset = raw_dataset.filter(has_complete_pair, desc="Removing empty pairs")
    eos_token = tokenizer.eos_token or ""

    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        completion = example["vgdl"].strip()
        if eos_token and not completion.endswith(eos_token):
            completion += eos_token
        prompt = build_prompt(example["description"])
        sequence_length = len(
            tokenizer(
                prompt + completion,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        return {
            "prompt": prompt,
            "completion": completion,
            "sequence_length": sequence_length,
        }

    dataset = dataset.map(format_example, desc="Formatting Phi SFT examples")

    dropped_overlength: dict[str, list[str]] = {}
    for split in required_splits:
        dropped_overlength[split] = [
            f"{split}[{index}]"
            for index, length in enumerate(dataset[split]["sequence_length"])
            if length > max_length
        ]

    if not keep_overlength:
        dataset = dataset.filter(
            lambda example: example["sequence_length"] <= max_length,
            desc=f"Dropping examples over {max_length} tokens",
        )

    statistics = {
        "original_sizes": original_sizes,
        "usable_sizes_before_length_filter": {
            split: original_sizes[split]
            - sum(
                1
                for example in raw_dataset[split]
                if not has_complete_pair(example)
            )
            for split in required_splits
        },
        "final_sizes": {split: len(dataset[split]) for split in required_splits},
        "max_length": max_length,
        "overlength_policy": "truncate" if keep_overlength else "drop",
        "overlength_files": dropped_overlength,
    }
    return dataset, statistics


def print_dataset_report(dataset: DatasetDict, statistics: dict[str, Any]) -> None:
    print("\nDataset report")
    print("==============")
    print(f"Maximum sequence length: {statistics['max_length']}")
    print(f"Overlength policy: {statistics['overlength_policy']}")
    for split in ("train", "test"):
        lengths = dataset[split]["sequence_length"]
        print(
            f"{split}: {statistics['original_sizes'][split]} original, "
            f"{statistics['final_sizes'][split]} selected, "
            f"max selected length={max(lengths) if lengths else 0}"
        )
        overlength_files = statistics["overlength_files"][split]
        if overlength_files:
            action = (
                "kept for truncation"
                if statistics["overlength_policy"] == "truncate"
                else "dropped"
            )
            print(f"  {action} ({len(overlength_files)}): {', '.join(overlength_files)}")

    if len(dataset["train"]) > 0:
        sample = dataset["train"][0]
        print("\nFirst training example")
        print("----------------------")
        print("source: train[0]")
        print(f"tokens: {sample['sequence_length']}")
        print(sample["prompt"])
        print(sample["completion"][:500])


def load_model(attn_implementation: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Phi-4-mini QLoRA training requires an NVIDIA GPU."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def resolve_resume_checkpoint(
    output_dir: Path,
    resume_option: str,
    overwrite_output: bool,
) -> str | None:
    normalized = resume_option.strip().lower()
    if normalized == "none":
        checkpoint = None
    elif normalized == "auto":
        checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    else:
        explicit_checkpoint = Path(resume_option).resolve()
        if not explicit_checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {explicit_checkpoint}")
        checkpoint = str(explicit_checkpoint)

    final_adapter_exists = (output_dir / "adapter_config.json").is_file()
    if checkpoint is None and final_adapter_exists and not overwrite_output:
        raise FileExistsError(
            f"A final adapter already exists in {output_dir}. Use a new --output-dir "
            "or pass --overwrite-output to start a fresh run there."
        )
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero.")

    output_dir = Path(args.output_dir).resolve()
    tokenizer = load_tokenizer()
    dataset, dataset_statistics = prepare_dataset(
        dataset_path=Path(args.dataset_path).resolve(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        keep_overlength=args.keep_overlength,
    )
    print_dataset_report(dataset, dataset_statistics)

    if len(dataset["train"]) == 0 or len(dataset["test"]) == 0:
        raise RuntimeError("The length/empty filters produced an empty dataset split.")

    if args.dry_run:
        print("\nDry run completed: dataset and prompt/completion formatting are valid.")
        return

    resume_checkpoint = resolve_resume_checkpoint(
        output_dir,
        args.resume_from_checkpoint,
        args.overwrite_output,
    )
    model = load_model(args.attn_implementation)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        seed=42,
        data_seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
    )

    if resume_checkpoint:
        print(f"Resuming SFT from: {resume_checkpoint}")
    else:
        print("Starting Phi-4-mini supervised fine-tuning from the base model.")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "base_model": BASE_MODEL,
        "adapter_output": str(output_dir),
        "training_source": "base_model",
        "loss_scope": "completion_only",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LORA_TARGET_MODULES,
        },
        "dataset": dataset_statistics,
        "train": train_result.metrics,
        "history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }
    metrics_path = output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"SFT complete. Adapter saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
