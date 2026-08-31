"""
Valuta un modello di generazione VGDL sull'intero test set.

Backend supportati:
  - ollama: modello locale Ollama, utile per la baseline zero-shot
  - hf: modello Hugging Face, con adapter LoRA opzionale per SFT/RL

Esempi:
  python evaluation/evaluate_model_testset.py \
      --backend ollama \
      --model gemma4:e4b \
      --run-name gemma4-zero-shot

  python evaluation/evaluate_model_testset.py \
      --backend hf \
      --model google/gemma-4-E4B-it \
      --adapter models/gemma4/supervised-learning/model-finetuned \
      --run-name gemma4-supervised
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import requests
from datasets import load_from_disk
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from check_vgdl_executability import validate_vgdl  # noqa: E402
from eval_similarity import vgdl_similarity  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "models" / "phi4-mini"))
from prompting import build_prompt as build_phi4_prompt  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)

MANDATORY_SECTIONS = (
    "SpriteSet",
    "LevelMapping",
    "InteractionSet",
    "TerminationSet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valuta un modello VGDL sul test set di dataset_hf."
    )
    parser.add_argument("--backend", choices=("ollama", "hf"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--profile",
        choices=("gemma4", "qwen3.5", "phi4-mini"),
        default="gemma4",
        help=(
            "Profilo di architettura e prompt per il backend Hugging Face. "
            "Il valore predefinito conserva il protocollo Gemma4 esistente."
        ),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Adapter LoRA per backend hf. Omettere per valutare il modello base.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset_hf")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "evaluation" / "results",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--limit",
        type=int,
        help="Valuta solo i primi N esempi. Utile per uno smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cancella i risultati esistenti invece di riprendere il run.",
    )
    return parser.parse_args()


def clean_generated_vgdl(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    basic_game_index = text.find("BasicGame")
    if basic_game_index >= 0:
        text = text[basic_game_index:]

    return text.strip()


def validate_vgdl_text(vgdl: str) -> dict:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_file.write(vgdl)
            temp_path = Path(temp_file.name)
        return validate_vgdl(str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class OllamaGenerator:
    def __init__(self, model: str, base_url: str, max_new_tokens: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens

        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        available = {
            item["name"] for item in response.json().get("models", [])
        }
        if model not in available:
            raise RuntimeError(
                f"Modello Ollama '{model}' non installato. "
                f"Disponibili: {sorted(available)}"
            )

    def generate(self, description: str) -> tuple[str, dict]:
        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": description.strip(),
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": self.max_new_tokens,
                    "temperature": 0,
                    "repeat_penalty": 1.1,
                    "seed": 42,
                },
            },
            timeout=1200,
        )
        response.raise_for_status()
        payload = response.json()
        elapsed = time.perf_counter() - start
        metadata = {
            "generation_seconds": elapsed,
            "prompt_tokens": payload.get("prompt_eval_count"),
            "generated_tokens": payload.get("eval_count"),
        }
        return clean_generated_vgdl(payload.get("response", "")), metadata


class HuggingFaceGenerator:
    def __init__(
        self,
        model_name: str,
        adapter: Path | None,
        max_new_tokens: int,
        profile: str,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Il backend Hugging Face richiede CUDA.")

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.profile = profile
        self.processor = None

        input_source = str(adapter) if adapter else model_name
        if profile == "phi4-mini":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    input_source,
                    trust_remote_code=False,
                )
            except OSError:
                if adapter is None:
                    raise
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=False,
                )
            model_class = AutoModelForCausalLM
        else:
            try:
                self.processor = AutoProcessor.from_pretrained(input_source)
            except OSError:
                if adapter is None:
                    raise
                self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = getattr(
                self.processor,
                "tokenizer",
                self.processor,
            )
            model_class = (
                AutoModelForImageTextToText
                if profile == "qwen3.5"
                else AutoModelForCausalLM
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

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
        model = model_class.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            dtype=compute_dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(adapter),
                low_cpu_mem_usage=True,
            )

        self.model = model.eval()

    def generate(self, description: str) -> tuple[str, dict]:
        if self.profile == "phi4-mini":
            inputs = self.tokenizer(
                build_phi4_prompt(description),
                return_tensors="pt",
            )
            inputs = {
                name: tensor.to(self.model.device)
                for name, tensor in inputs.items()
            }
        else:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": description.strip()}
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.model.device)

        start = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        metadata = {
            "generation_seconds": elapsed,
            "prompt_tokens": int(input_length),
            "generated_tokens": int(generated_ids.shape[-1]),
        }
        return clean_generated_vgdl(text), metadata


def create_generator(args: argparse.Namespace):
    if args.backend == "ollama":
        if args.adapter:
            raise ValueError("--adapter può essere usato solo con --backend hf")
        return OllamaGenerator(
            args.model,
            args.ollama_url,
            args.max_new_tokens,
        )

    return HuggingFaceGenerator(
        args.model,
        args.adapter,
        args.max_new_tokens,
        args.profile,
    )


def evaluate_generation(
    index: int,
    description: str,
    reference_vgdl: str,
    generated_vgdl: str,
    metadata: dict,
) -> dict:
    validation = validate_vgdl_text(generated_vgdl)
    similarity = vgdl_similarity(generated_vgdl, reference_vgdl)
    sections_present = {
        section: section in generated_vgdl
        for section in MANDATORY_SECTIONS
    }

    return {
        "index": index,
        "description": description,
        "reference_vgdl": reference_vgdl,
        "generated_vgdl": generated_vgdl,
        "valid": validation["valid"],
        "validation_errors": validation["errors"],
        "starts_with_basic_game": generated_vgdl.startswith("BasicGame"),
        "all_sections_present": all(sections_present.values()),
        "sections_present": sections_present,
        **similarity,
        **metadata,
    }


def load_completed_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def build_summary(records: list[dict], args: argparse.Namespace) -> dict:
    valid_records = [record for record in records if record["valid"]]
    return {
        "run_name": args.run_name,
        "backend": args.backend,
        "model": args.model,
        "profile": args.profile,
        "prompt_protocol": (
            "phi4_raw_shared_prompt"
            if args.profile == "phi4-mini"
            else "system_user_chat"
        ),
        "adapter": str(args.adapter) if args.adapter else None,
        "examples": len(records),
        "executability_rate": mean(
            [float(record["valid"]) for record in records]
        ),
        "basic_game_rate": mean(
            [float(record["starts_with_basic_game"]) for record in records]
        ),
        "complete_structure_rate": mean(
            [float(record["all_sections_present"]) for record in records]
        ),
        "mean_sprite_similarity": mean(
            [record["sprite_similarity"] for record in records]
        ),
        "mean_interaction_similarity": mean(
            [record["interaction_similarity"] for record in records]
        ),
        "mean_termination_similarity": mean(
            [record["termination_similarity"] for record in records]
        ),
        "mean_structural_similarity": mean(
            [record["final_score"] for record in records]
        ),
        "mean_similarity_valid_only": mean(
            [record["final_score"] for record in valid_records]
        ),
        "mean_generation_seconds": mean(
            [record["generation_seconds"] for record in records]
        ),
    }


def write_csv(records: list[dict], path: Path) -> None:
    columns = [
        "index",
        "valid",
        "starts_with_basic_game",
        "all_sections_present",
        "sprite_similarity",
        "interaction_similarity",
        "termination_similarity",
        "final_score",
        "generation_seconds",
        "prompt_tokens",
        "generated_tokens",
        "validation_errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {column: record.get(column) for column in columns}
            row["validation_errors"] = " | ".join(
                record["validation_errors"]
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    details_path = run_dir / "details.jsonl"
    summary_path = run_dir / "summary.json"
    csv_path = run_dir / "metrics.csv"

    if args.overwrite:
        details_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    dataset = load_from_disk(str(args.dataset))["test"]
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    completed_records = load_completed_records(details_path)
    completed_indices = {
        record["index"] for record in completed_records
    }

    generator = create_generator(args)
    with details_path.open("a", encoding="utf-8") as details_file:
        for index, example in enumerate(
            tqdm(dataset, desc=args.run_name, unit="game")
        ):
            if index in completed_indices:
                continue

            try:
                generated_vgdl, metadata = generator.generate(
                    example["description"]
                )
                record = evaluate_generation(
                    index,
                    example["description"],
                    example["vgdl"],
                    generated_vgdl,
                    metadata,
                )
            except Exception as exc:
                record = {
                    "index": index,
                    "description": example["description"],
                    "reference_vgdl": example["vgdl"],
                    "generated_vgdl": "",
                    "valid": False,
                    "validation_errors": [
                        f"Generation error: {type(exc).__name__}: {exc}"
                    ],
                    "starts_with_basic_game": False,
                    "all_sections_present": False,
                    "sections_present": {
                        section: False for section in MANDATORY_SECTIONS
                    },
                    "sprite_similarity": 0.0,
                    "interaction_similarity": 0.0,
                    "termination_similarity": 0.0,
                    "final_score": 0.0,
                    "generation_seconds": 0.0,
                    "prompt_tokens": None,
                    "generated_tokens": None,
                }

            details_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            details_file.flush()
            completed_records.append(record)

    completed_records.sort(key=lambda record: record["index"])
    summary = build_summary(completed_records, args)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(completed_records, csv_path)

    print("\nRisultati")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDettagli: {details_path}")
    print(f"Riepilogo: {summary_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
