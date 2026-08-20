"""Generate VGDL with the supervised Phi-4-mini LoRA adapter."""

from __future__ import annotations

import argparse
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_THIS_DIR = Path(__file__).resolve().parent
_PHI_DIR = _THIS_DIR.parent
_REPO_ROOT = _PHI_DIR.parents[1]
sys.path.insert(0, str(_PHI_DIR))

from prompting import build_prompt, clean_generated_vgdl


BASE_MODEL = "microsoft/Phi-4-mini-instruct"
DEFAULT_ADAPTER_PATH = _THIS_DIR / "model-finetuned"
DEFAULT_DESCRIPTION_DIR = _REPO_ROOT / "dataset" / "descriptions"
DEFAULT_OUTPUT_DIR = _THIS_DIR / "vgdl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VGDL inference with the Phi-4-mini supervised LoRA adapter."
    )
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--description-file")
    parser.add_argument("--description-text")
    parser.add_argument("--description-dir", default=str(DEFAULT_DESCRIPTION_DIR))
    parser.add_argument("--output-path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Batch limit; use 0 to process every description.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-max-memory", default="5GiB")
    parser.add_argument("--cpu-max-memory", default="24GiB")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="eager",
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def numeric_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def selected_description_files(args: argparse.Namespace) -> list[Path]:
    files = sorted(Path(args.description_dir).glob("*.txt"), key=numeric_sort_key)
    if args.limit > 0:
        files = files[: args.limit]
    return files


def validate_adapter(adapter_path: Path) -> None:
    required_files = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required_files if not (adapter_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid SFT adapter directory '{adapter_path}'; missing: {', '.join(missing)}. "
            "Run supervised-learning/finetune.py first."
        )


def load_model(args: argparse.Namespace):
    adapter_path = Path(args.adapter_path).resolve()
    validate_adapter(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "attn_implementation": args.attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs.update(
            {
                "device_map": "auto",
                "max_memory": {0: args.gpu_max_memory, "cpu": args.cpu_max_memory},
                "dtype": torch.float16,
            }
        )
        if not args.no_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
    elif not args.no_4bit:
        raise RuntimeError("4-bit inference requires CUDA; use --no-4bit for CPU inference.")
    else:
        model_kwargs.update({"device_map": "cpu", "dtype": torch.float32})

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    print(f"Loading supervised adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    if args.merge_adapter:
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    description: str,
    max_new_tokens: int,
    adapter_enabled: bool,
) -> str:
    inputs = tokenizer(build_prompt(description), return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    adapter_context = nullcontext()
    if not adapter_enabled and hasattr(model, "disable_adapter"):
        adapter_context = model.disable_adapter()

    started_at = time.time()
    with adapter_context, torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(
        f"Input tokens: {inputs['input_ids'].shape[-1]} | "
        f"generated tokens: {generated_ids.shape[-1]} | "
        f"elapsed: {time.time() - started_at:.2f}s"
    )
    return clean_generated_vgdl(
        tokenizer.decode(generated_ids, skip_special_tokens=True)
    )


def write_output(output_path: Path, vgdl: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(vgdl + "\n", encoding="utf-8")
    print(f"Saved to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")
    if args.merge_adapter and (args.disable_adapter or args.compare_base):
        raise ValueError(
            "--merge-adapter cannot be combined with --disable-adapter or --compare-base."
        )

    jobs: list[tuple[str, str, Path | None]] = []
    if args.description_text:
        jobs.append(
            (
                "inline description",
                args.description_text.strip(),
                Path(args.output_path).resolve() if args.output_path else None,
            )
        )
    elif args.description_file:
        source = Path(args.description_file).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Description file not found: {source}")
        output_path = (
            Path(args.output_path).resolve()
            if args.output_path
            else Path(args.output_dir).resolve()
            / f"{source.stem}_vgdl_phi4mini_sft.txt"
        )
        jobs.append(
            (source.name, source.read_text(encoding="utf-8").strip(), output_path)
        )
    else:
        for source in selected_description_files(args):
            jobs.append(
                (
                    source.name,
                    source.read_text(encoding="utf-8").strip(),
                    Path(args.output_dir).resolve()
                    / f"{source.stem}_vgdl_phi4mini_sft.txt",
                )
            )

    if not jobs:
        raise RuntimeError("No descriptions selected.")

    if args.dry_run:
        print(f"Adapter: {Path(args.adapter_path).resolve()}")
        print(f"Selected descriptions: {len(jobs)}")
        print(f"First item: {jobs[0][0]}\n")
        print(build_prompt(jobs[0][1]))
        return

    model, tokenizer = load_model(args)
    for index, (label, description, output_path) in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {label}")
        if args.compare_base:
            base_vgdl = generate(
                model,
                tokenizer,
                description,
                args.max_new_tokens,
                adapter_enabled=False,
            )
            print("\n=== Base model (adapter disabled) ===")
            print(base_vgdl)

        vgdl = generate(
            model,
            tokenizer,
            description,
            args.max_new_tokens,
            adapter_enabled=not args.disable_adapter,
        )
        print("\n=== Generated VGDL ===")
        print(vgdl)
        if output_path:
            write_output(output_path, vgdl)


if __name__ == "__main__":
    main()
