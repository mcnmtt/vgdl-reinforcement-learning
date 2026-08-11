"""
GRPO training per generazione VGDL con Gemma4.

Parte dall'adapter SFT in:
  models/gemma4/supervised-learning/model-finetuned

e lo continua con reward automatiche:
  - eseguibilita' VGDL
  - similarita' strutturale rispetto al riferimento
  - struttura VGDL obbligatoria
  - classi sprite, interazioni, terminazioni valide
  - uso corretto di EOS
  - penalita' per markdown/prosa

Eseguire dalla root del progetto:
  python models/gemma4/reinforcement-learning/grpo_train.py
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

# Compatibilita' llm_blender + Transformers dev: TRL importa llm_blender anche
# se GRPO non usa i judge. llm_blender si aspetta ancora TRANSFORMERS_CACHE.
import transformers.utils.hub as _transformers_hub  # noqa: E402

if not hasattr(_transformers_hub, "TRANSFORMERS_CACHE"):
    _transformers_hub.TRANSFORMERS_CACHE = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
    )

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "py-vgdl"))

from reward_functions import REWARD_FUNCTIONS  # noqa: E402


def copy_checkpoint(checkpoint: str, destination_name: str) -> None:
    """Copia un checkpoint completo in una cartella dal nome stabile."""
    if not checkpoint or not os.path.isdir(checkpoint):
        return
    destination = os.path.join(OUTPUT_DIR, destination_name)
    shutil.copytree(checkpoint, destination, dirs_exist_ok=True)
    print(f"{destination_name} aggiornato da: {checkpoint}")


def remove_utf8_bom(file_path: str) -> None:
    """Rimuove l'eventuale BOM che Transformers non accetta nei JSON."""
    path = Path(file_path)
    if not path.is_file():
        return
    content = path.read_bytes()
    utf8_bom = b"\xef\xbb\xbf"
    if content.startswith(utf8_bom):
        path.write_bytes(content[len(utf8_bom) :])
        print(f"BOM UTF-8 rimosso da: {path}")


class BestLastCheckpointCallback(TrainerCallback):
    """Aggiorna last-model e best-model ad ogni checkpoint salvato."""

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


class RewardAwareGRPOTrainer(GRPOTrainer):
    """Espone eval_reward al Trainer per la selezione del best checkpoint."""

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        # GRPOTrainer aggiunge eval_reward solo al log, mentre Transformers
        # sceglie il best usando il dizionario restituito da evaluate().
        for log_entry in reversed(self.state.log_history):
            if "eval_reward" in log_entry:
                metrics["eval_reward"] = log_entry["eval_reward"]
                break

        return metrics


# ========================
# Config
# ========================
BASE_MODEL = "google/gemma-4-E4B-it"
SFT_ADAPTER = "models/gemma4/supervised-learning/model-finetuned"
OUTPUT_DIR = "models/gemma4/reinforcement-learning/grpo-output"
MAX_SEQ_LEN = 1024
SAVE_LAST_EVERY_STEPS = 5

SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)


# ========================
# Model Loading
# ========================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA non disponibile: GRPO su Gemma4 richiede GPU.")

print(f"Loading processor from {SFT_ADAPTER}...")
processor = AutoProcessor.from_pretrained(SFT_ADAPTER)
tokenizer = getattr(processor, "tokenizer", processor)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model {BASE_MODEL} in 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
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

# Compatibilita' TRL 0.24 + Transformers dev/Gemma4. GRPOTrainer usa questo
# dizionario per silenziare il warning sulla stima dei token, ma Gemma4 non lo
# inizializza come fanno le architetture gia' supportate dalla release stabile.
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# ========================
# Dataset
# ========================
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
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {"prompt": prompt}


# NON rimuovere 'vgdl': viene passato come kwarg alle reward functions.
dataset = dataset.map(format_prompt)
print(f"Train size: {len(dataset['train'])} | Test size: {len(dataset['test'])}")


# ========================
# GRPO Config
# ========================
# Config conservativa per RTX 5070 12GB. Se resta memoria libera, puoi alzare
# max_completion_length o num_generations.
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-6,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    per_device_eval_batch_size=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_reward",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    num_generations=2,
    max_prompt_length=512,
    max_completion_length=384,
    temperature=0.8,
    beta=0.04,
    use_vllm=False,
)


# ========================
# GRPO Trainer
# ========================
trainer = RewardAwareGRPOTrainer(
    model=model,
    reward_funcs=REWARD_FUNCTIONS,
    args=grpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    callbacks=[BestLastCheckpointCallback()],
)

if os.environ.get("GRPO_DRY_RUN") == "1":
    print("GRPO dry run completato: modello, dataset e trainer inizializzati.")
    raise SystemExit(0)


# ========================
# Training
# ========================
print("Starting Gemma4 GRPO training...")
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
    last_checkpoint = last_model_dir
elif os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
else:
    last_checkpoint = None

if last_checkpoint:
    print(f"Ripresa training da: {last_checkpoint}")
    if os.path.normpath(last_checkpoint) != os.path.normpath(last_model_dir):
        copy_checkpoint(last_checkpoint, "last-model")
    remove_utf8_bom(os.path.join(last_checkpoint, "trainer_state.json"))
else:
    print("Nessun checkpoint trovato: training avviato da zero.")

train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Con load_best_model_at_end=True, il modello attivo qui e' il best. I due
# checkpoint completi vengono anche copiati in cartelle dal significato stabile,
# cosi' sono immediatamente utilizzabili per inference e confronto.
best_checkpoint = trainer.state.best_model_checkpoint
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if best_checkpoint:
    best_output = os.path.join(OUTPUT_DIR, "best-model")
    shutil.copytree(best_checkpoint, best_output, dirs_exist_ok=True)
    print(f"Best checkpoint (eval_reward): {best_checkpoint}")
    print(f"Best model copiato in: {best_output}")
else:
    print("WARN: nessun best checkpoint disponibile.")

if last_checkpoint:
    last_output = os.path.join(OUTPUT_DIR, "last-model")
    shutil.copytree(last_checkpoint, last_output, dirs_exist_ok=True)
    print(f"Last checkpoint: {last_checkpoint}")
    print(f"Last model copiato in: {last_output}")

metrics = {
    "train": train_result.metrics,
    "history": trainer.state.log_history,
    "best_checkpoint": best_checkpoint,
    "last_checkpoint": last_checkpoint,
    "best_metric": trainer.state.best_metric,
}
metrics_path = os.path.join(OUTPUT_DIR, "grpo_training_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"GRPO training completato! Modello salvato in {OUTPUT_DIR}")
print(f"Metriche salvate in {metrics_path}")
