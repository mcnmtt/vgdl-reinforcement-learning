import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# ======================
# Config
# ======================
BASE_MODEL = "google/gemma-4-E4B-it"
LORA_DIR = "models/gemma4/supervised-learning/model-finetuned"

SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)

# ======================
# Caricamento modello + adapter LoRA
# ======================
print("Loading model...")
start = time.time()

processor = AutoProcessor.from_pretrained(LORA_DIR)
tokenizer = getattr(processor, "tokenizer", processor)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Model loaded in {round(time.time() - start, 2)}s\n")


# ======================
# Funzione di generazione
# ======================
def generate_vgdl(description: str, output_path: str = None) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": description.strip()}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    print(f"Input tokens: {inputs['input_ids'].shape[-1]}")

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"Generated in {round(time.time() - start, 2)}s")

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_length:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        print(f"Output saved to {output_path}")

    return raw_output


# ======================
# Test
# ======================
description = "Player role: This is a two-player game where each player controls a bandit character that can only shoot projectiles. Player A controls a bandit facing right, and Player B controls a bandit facing left. The players can shoot rock, paper, scissors, lizard, or spock projectiles (depending on the game variant) toward their opponent. Entities: - Floor: An invisible grey background surface that covers the playable area - Bandit avatars: Two bandit characters, one facing right (Player A) and one facing left (Player B), that can shoot different types of projectiles - Walls: Grey stone barriers that block projectiles, with connected wall segments forming larger structures - Rock projectiles: Planet-shaped missiles that move horizontally at medium or slow speed and shrink as they travel - Paper projectiles: Scroll-shaped missiles that move horizontally at medium or slow speed and shrink as they travel - Scissors projectiles: Axe-shaped missiles that move horizontally at medium or slow speed and shrink as they travel - Lizard projectiles: Dragon-shaped missiles that move horizontally at medium or slow speed and shrink as they travel - Spock projectiles: Druid-shaped missiles that move horizontally at medium or slow speed and shrink as they travel - Choice indicators: Flickering visual displays that show what each player has selected, with Player A's choices visible to Player A and Player B's choices visible to Player B - Buzzers: Small visual indicators that appear at specific locations and transform to show the current choice when touched by a selection - Timers: Invisible objects that trigger events at regular intervals - some every 50 time units, others every 20 time units - Choose triggers: Invisible objects that activate the shooting of projectiles based on current selections Interactions: - Any projectile hitting a wall is destroyed - When a buzzer touches a rock choice, it transforms into a rock buzzer and the choice disappears - When a buzzer touches a paper choice, it transforms into a paper buzzer and the choice disappears - When a buzzer touches a scissors choice, it transforms into a scissors buzzer and the choice disappears - When a buzzer touches a lizard choice, it transforms into a lizard buzzer and the choice disappears - When a buzzer touches a spock choice, it transforms into a spock buzzer and the choice disappears - When a medium-speed timer trigger touches a rock buzzer, it creates a medium-speed rock projectile - When a medium-speed timer trigger touches a paper buzzer, it creates a medium-speed paper projectile - When a medium-speed timer trigger touches a scissors buzzer, it creates a medium-speed scissors projectile - When a medium-speed timer trigger touches a lizard buzzer, it creates a medium-speed lizard projectile - When a medium-speed timer trigger touches a spock buzzer, it creates a medium-speed spock projectile - When a slow-speed timer trigger touches any buzzer, it creates the corresponding slow-speed projectile - Any timer trigger touching a buzzer destroys the trigger - When identical projectiles from different players collide, both are destroyed - Paper projectiles destroy rock projectiles (paper beats rock) - Paper projectiles destroy spock projectiles (paper beats spock) - Scissors projectiles destroy paper projectiles (scissors beats paper) - Scissors projectiles destroy lizard projectiles (scissors beats lizard) - Rock projectiles destroy scissors projectiles (rock beats scissors) - Rock projectiles destroy lizard projectiles (rock beats lizard) - Spock projectiles destroy rock projectiles (spock beats rock) - Spock projectiles destroy scissors projectiles (spock beats scissors) - Lizard projectiles destroy paper projectiles (lizard beats paper) - Lizard projectiles destroy spock projectiles (lizard beats spock) - When Player B's projectile hits Player A's avatar, Player A is destroyed and Player B gains a point - When Player A's projectile hits Player B's avatar, Player B is destroyed and Player A gains a point Win condition: The player with the higher score when time runs out wins the game. Lose condition: The player with the lower score when the 500 time unit limit expires loses the game. Objective: Engage in rock-paper-scissors-lizard-spock combat by shooting projectiles at your opponent while avoiding their attacks, trying to score more hits than them before time"

result = generate_vgdl(
    description,
    output_path=(
        "models/gemma4/supervised-learning/"
        "model-finetuned/results/finetuned_output.txt"
    ),
)
print(result)
