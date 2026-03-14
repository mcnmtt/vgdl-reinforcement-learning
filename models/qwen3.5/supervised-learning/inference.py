import os
import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ======================
# Config
# ======================
BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_DIR   = "models/qwen3.5/supervised-learning/16-32-0.05"

# ======================
# Caricamento (identico al tuo zero-shot + adapter LoRA)
# ======================
print("Loading model...")
start = time.time()

processor = AutoProcessor.from_pretrained(BASE_MODEL)

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,          # ← identico al tuo zero-shot
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True,              # ← identico al tuo zero-shot
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()
print(f"Model loaded in {round(time.time()-start, 2)}s\n")

# ======================
# Funzione di generazione
# Iperparametri identici al tuo zero-shot
# ======================
def generate_vgdl(description: str, output_path: str = None) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": description.strip()}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,           # ← identico al tuo zero-shot
    ).to(model.device)

    print(f"Input tokens: {inputs['input_ids'].shape[-1]}")

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,          # ← identico al tuo zero-shot
            temperature=0,               # ← identico al tuo zero-shot
            do_sample=False,             # ← identico al tuo zero-shot
            repetition_penalty=1.1,      # ← identico al tuo zero-shot
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    print(f"Generated in {round(time.time()-start, 2)}s")

    input_length  = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_length:]
    raw_output    = processor.decode(generated_ids, skip_special_tokens=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(raw_output)
        print(f"Output saved to {output_path}")

    return raw_output


# ======================
# Test
# ======================
description = "Generate a VGDL code of a game with the following description: **Player role:** The player controls a paddle that moves vertically along one side of the screen. The paddle moves at a moderate speed and can be controlled with alternate key bindings. **Entities:** - A green goal area on the player's side that represents the player's scoring zone - A green goal area on the opponent's side that represents the opponent's scoring zone - The player's paddle that moves vertically and can be controlled by the player - The opponent's blue paddle that moves vertically on the opposite side - An orange ball that moves horizontally at high speed without friction, bouncing around the playing field - Walls that form the boundaries of the playing area **Interactions:** - When the ball touches either goal area, the ball disappears - When the ball hits either paddle, it bounces off and changes direction - When the ball hits a wall, it bounces off the wall - When either paddle tries to move into a wall, it is pushed back and cannot pass through **Win condition:** The player wins when the ball has entered the opponent's goal area 6 times. **Lose condition:** The player loses when the ball has entered their own goal area 6 times. **Objective:** Use your paddle to deflect the ball toward the opponent's goal while preventing it from reaching your own goal. Score 6 points before your opponent does to win."

result = generate_vgdl(description, output_path="models/qwen3.5/supervised-learning/16-32-0.05/results/finetuned_output.txt")
print(result)