import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# Model Loading
# ======================

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# ======================
# VGDL Input
# ======================

vgdl_code = """
BasicGame
    SpriteSet 
        pad    > Passive color=BLUE 
        avatar > InertialAvatar physicstype=GravityPhysics
            
    TerminationSet
        SpriteCounter stype=pad limit=4 win=True     
        SpriteCounter stype=avatar      win=False     
           
    InteractionSet
        avatar wall > killSprite 
        avatar EOS > killSprite 
        pad avatar  > killIfSlow    # relative velocity
        
    LevelMapping
        G > pad
"""

# ======================
# Prompt
# ======================

instruction = f"""
You are an expert VGDL analyst.

Your task is to read a VGDL specification and produce a structured description.

STRICT RULES:
- Use ONLY information explicitly present in the VGDL.
- Do NOT invent mechanics.
- Do NOT add explanations outside the required structure.
- If a rule is not explicitly defined, do not mention it.

Each section must contain 1 or 2 sentences.

FORMAT (follow EXACTLY):

Player role:
<text>

Entities:
<text>

Interactions:
<text>

Win condition:
<text>

Lose condition:
<text>

Objective:
<text>

VGDL:
{vgdl_code}

Generate the description following EXACTLY the format above.
"""

# ======================
# Chat template
# ======================

messages = [{"role": "user", "content": instruction}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# ======================
# Tokenization
# ======================

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ======================
# Function to check structure
# ======================

def valid_structure(text):

    required_sections = [
        "Player role:",
        "Entities:",
        "Interactions:",
        "Win condition:",
        "Lose condition:",
        "Objective:"
    ]

    return all(section.lower() in text.lower() for section in required_sections)


# ======================
# Generation with retry
# ======================

max_attempts = 3
description_only = ""

for attempt in range(max_attempts):

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

    description_only = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if valid_structure(description_only):
        break
    else:
        print(f"Structure invalid, regenerating... attempt {attempt+1}")

# ======================
# Save result
# ======================

output_file = "game_description_mistral.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(description_only)

# ======================
# Print
# ======================

print("\nGenerated Description:\n")
print(description_only)

print(f"\nDescrizione salvata in {output_file}")