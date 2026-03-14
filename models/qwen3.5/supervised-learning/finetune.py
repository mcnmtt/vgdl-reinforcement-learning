import json
import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig

# ======================
# Config
# ======================
MODEL_NAME  = "Qwen/Qwen3.5-4B"
OUTPUT_DIR  = "models/qwen3.5/supervised-learning/model-finetuned-unsloth"
MAX_SEQ_LEN = 1024

SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)

# ======================
# Model + LoRA in una sola chiamata
# Unsloth gestisce QLoRA automaticamente
# ======================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,          # QLoRA automatico
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",  # ottimizzazione memoria
    random_state=42,
)
model.print_trainable_parameters()

# ======================
# Dataset
# ======================
dataset = load_from_disk("dataset_hf")

def format_example(example):
    return {
        "text": (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{example['description'].strip()}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{example['vgdl'].strip()}"
            "<|im_end|>"
        )
    }

dataset = dataset.map(format_example)

# ======================
# Trainer
# ======================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=MAX_SEQ_LEN,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=sft_config,
)

# ======================
# Training
# ======================
print("Starting training...")
train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ======================
# Salvataggio metriche
# ======================
metrics = {
    "train": train_result.metrics,
    "history": trainer.state.log_history,
}
metrics_path = f"{OUTPUT_DIR}/training_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Fine-tuning completato! Modello salvato in {OUTPUT_DIR}")
print(f"Metriche salvate in {metrics_path}")