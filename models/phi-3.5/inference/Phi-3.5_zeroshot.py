import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# Model Loading
# ======================

model_name = "microsoft/Phi-3.5-mini-instruct"

torch.random.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
    attn_implementation="eager"
)

model.eval()

# ======================
# Natural Language Input
# ======================

game_description = """
Player role:
The player controls an avatar that can aim and shoot a bullet projectile. The bullet is affected by gravity and can exist only once at a time.

Entities:
The game contains pad objects that act as targets, walls that block movement and projectiles, and bullets fired by the avatar.

Interactions:
Bullets are destroyed when they collide with walls, pads, or the edge of the screen. The avatar cannot pass through walls or the edge of the screen and is pushed back when attempting to do so.

Win condition:
The player wins when all pad objects are removed.

Lose condition:
The player loses if the avatar is removed.

Objective:
Shoot bullets to destroy all pads while navigating the environment.
"""

# ======================
# Prompt
# ======================

messages = [
    {
        "role": "system",
        "content": """You are an expert VGDL generator.

Convert a natural language description of a game into one complete VGDL specification.

Rules:
- Output only VGDL code.
- Do not output explanations, comments, markdown, or extra text.
- Preserve only mechanics supported by the description.
- Do not invent unnecessary entities or rules.
- Keep the VGDL simple, coherent, and as correct as possible.
- If details are missing, make only minimal conservative assumptions.
- Include SpriteSet, InteractionSet, TerminationSet, and LevelMapping when appropriate.
"""
    },
    {
        "role": "user",
        "content": f"""Game description:
{game_description}

VGDL:
"""
    }
]

# ======================
# Tokenization
# ======================

encodings = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

input_ids = encodings["input_ids"].to(model.device)
attention_mask = encodings["attention_mask"].to(model.device)

# ======================
# Generation
# ======================

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=400,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# ======================
# Extract ONLY generated tokens
# ======================

generated_tokens = outputs[0][input_ids.shape[-1]:]

vgdl_only = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()

# ======================
# Save to TXT
# ======================

output_file = "generated_vgdl_phi3.5.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(vgdl_only)

# ======================
# Print result
# ======================

print("\nGenerated VGDL:\n")
print(vgdl_only)

print(f"\nVGDL salvato in {output_file}")
# ======================
# Natural Language Input
# ======================

game_description = """
Player role:
The player controls an avatar that can aim and shoot a bullet projectile. The bullet is affected by gravity and can exist only once at a time.

Entities:
The game contains pad objects that act as targets, walls that block movement and projectiles, and bullets fired by the avatar.

Interactions:
Bullets are destroyed when they collide with walls, pads, or the edge of the screen. The avatar cannot pass through walls or the edge of the screen and is pushed back when attempting to do so.

Win condition:
The player wins when all pad objects are removed.

Lose condition:
The player loses if the avatar is removed.

Objective:
Shoot bullets to destroy all pads while navigating the environment.
"""

# ======================
# Prompt
# ======================

messages = [
    {
                "role": "system",
        "content": """You are an expert generator of classical VGDL (Video Game Description Language).

Your task is to convert a natural language game description into one complete VGDL specification in standard classical VGDL format.

Mandatory rules:
- Output only raw VGDL text.
- The first line must be: BasicGame
- Use indentation-based VGDL only.
- Do not use braces, JSON, XML, YAML, markdown, comments, or explanations.
- Do not use fields such as Width, Height, Image, Level names, quoted identifiers, or nested scripting blocks.
- Use classical VGDL sections when appropriate: SpriteSet, InteractionSet, TerminationSet, LevelMapping.
- Write interactions in classical VGDL style, such as:
  avatar wall > stepBack
  bullet wall > killSprite
- Preserve only mechanics explicitly supported by the description.
- Do not invent unnecessary entities or rules.
- If details are missing, make only minimal conservative assumptions.
- Keep the VGDL simple, coherent, and as correct as possible.

Example format:

BasicGame
    SpriteSet
        avatar > MovingAvatar
        wall > Immovable
    InteractionSet
        avatar wall > stepBack
    TerminationSet
        SpriteCounter stype=avatar win=False
    LevelMapping
        w > wall
"""
    },
    {
        "role": "user",
        "content": f"""Generate a VGDL specification for this game.

Game description:
{game_description}

VGDL:
"""
    }
]

# ======================
# Tokenization
# ======================

encodings = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

input_ids = encodings["input_ids"].to(model.device)
attention_mask = encodings["attention_mask"].to(model.device)

# ======================
# Generation
# ======================

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=400,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# ======================
# Extract ONLY generated tokens
# ======================

generated_tokens = outputs[0][input_ids.shape[-1]:]

vgdl_only = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()

# ======================
# Save to TXT
# ======================

output_file = "generated_vgdl_phi3.5.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(vgdl_only)

# ======================
# Print result
# ======================

print("\nGenerated VGDL:\n")
print(vgdl_only)

print(f"\nVGDL salvato in {output_file}")