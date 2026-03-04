import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# Model Loading
# ======================

model_name = "deepseek-ai/deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
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

prompt = f"""You are an expert game designer and technical analyst.

Your task is to read a VGDL (Video Game Description Language) specification and generate an accurate natural language description of the video game.

Strict rules:
- Do NOT invent mechanics that are not explicitly present in the VGDL.
- Do NOT assume hidden rules.
- Only describe what can be inferred from SpriteSet, InteractionSet, TerminationSet and LevelMapping.
- If something is unclear, describe it conservatively.

The description must follow exactly this structure:

Player role:
Describe what the player controls and its abilities.

Entities:
Describe enemies, objects, and other important game entities.

Interactions:
Explain the main gameplay mechanics and interactions between entities.

Win condition:
Explain how the player wins the game.

Lose condition:
Explain how the player loses the game.

Objective:
Summarize the overall goal of the game in one sentence.

Write in clear, concise natural language.
Do not exceed 8 sentences.

VGDL:
{vgdl_code}

Game Description:
"""

# ======================
# Tokenization
# ======================

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ======================
# Generation
# ======================

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# ======================
# Extract ONLY generated tokens
# ======================

generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

description_only = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()

# ======================
# Save to TXT
# ======================

output_file = "game_description_deepseek.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(description_only)

# ======================
# Print result
# ======================

print("\nGenerated Description:\n")
print(description_only)

print(f"\nDescrizione salvata in {output_file}")