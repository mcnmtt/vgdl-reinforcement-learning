import json

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ======================
# Config
# ======================
MODEL_NAME = "google/gemma-4-E4B-it"
OUTPUT_DIR = "models/gemma4/supervised-learning/model-finetuned"
MAX_SEQ_LEN = 1024

SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)

# ======================
# Model + QLoRA
# ======================
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = getattr(processor, "tokenizer", processor)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Limita LoRA al language model. I vision/audio tower usano wrapper
    # Gemma4ClippableLinear che PEFT non supporta direttamente.
    target_modules=(
        r"model\.language_model\.layers\.\d+\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))"
    ),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# Dataset
# ======================
dataset = load_from_disk("dataset_hf")


def format_example(example):
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
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["vgdl"].strip()}],
        },
    ]

    return {
        "text": processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
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
    per_device_eval_batch_size=1,
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
    prediction_loss_only=True,
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
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
processor.save_pretrained(OUTPUT_DIR)

# ======================
# Salvataggio metriche
# ======================
metrics = {
    "train": train_result.metrics,
    "history": trainer.state.log_history,
}
metrics_path = f"{OUTPUT_DIR}/training_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Fine-tuning completato! Modello salvato in {OUTPUT_DIR}")
print(f"Metriche salvate in {metrics_path}")
