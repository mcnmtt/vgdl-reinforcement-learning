import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# Model Loading
# ======================

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

# ======================
# Game Description Input
# ======================

vgdl_description = """
Player role:
The player controls an avatar with Mario-like movement under gravity. The avatar can steer while in the air.

Entities:
The game contains enemies called Goombas and Paratroopas, a goal object, and moving elevators. Goombas walk horizontally, while Paratroopas move by jumping. Elevators move upward continuously.

Interactions:
If the avatar jumps on an enemy from above, the enemy is killed and the player gains one point. If the avatar collides with an enemy while the enemy is still alive, the avatar dies. Moving entities stop when they hit walls and are affected by friction. Elevators can carry moving entities along with them as they move. When elevators reach the edge of the level, they wrap around to the opposite side.

Win condition:
The player wins when the avatar reaches the goal object.

Lose condition:
The player loses when the avatar is killed.

Objective:
Navigate the level using movement and elevators, avoid or defeat enemies, and reach the goal.
"""

# ======================
# Prompt / Chat Messages
# ======================

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert in VGDL (Video Game Description Language).\n"
            "Your task is to convert a natural language description of a video game into a valid VGDL specification.\n\n"
            "Instructions:\n"
            "- Output only VGDL code.\n"
            "- The first line must be exactly: BasicGame\n"
            "- Use indentation-based VGDL syntax.\n"
            "- Include the following sections in this order whenever they are needed by the game description:\n"
            "  SpriteSet\n"
            "  TerminationSet\n"
            "  InteractionSet\n"
            "  LevelMapping\n"
            "- Use only mechanics, entities, and win/lose conditions that are explicitly stated or conservatively inferable from the description.\n"
            "- Do not invent hidden rules, extra goals, extra enemies, helper objects, scores, colors, or mechanics unless strongly supported by the description.\n"
            "- Use standard VGDL-style constructs when possible.\n"
            "- Prefer a minimal, clean, and syntactically consistent VGDL specification.\n"
            "- If some implementation detail is missing, choose the simplest VGDL formulation compatible with the description.\n"
            "- Do not output explanations, comments, markdown, headings, or natural language before or after the VGDL.\n"
            "- Do not use JSON, YAML, XML, pseudocode, or parentheses-based syntax.\n"
            "- Keep the output compact and focused on the core playable rules.\n"
        )
    },
    {
        "role": "user",
        "content": (
            "Generate a VGDL specification for the following game description.\n\n"
            "Game description:\n"
            f"{vgdl_description}\n\n"
            "Requirements:\n"
            "- Return only the final VGDL code.\n"
            "- Start directly with BasicGame.\n"
            "- Use conservative assumptions when details are ambiguous.\n"
            "- Preserve the entities, interactions, win condition, and lose condition described in the text as closely as possible.\n"
        )
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# ======================
# Generation
# ======================

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=350,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# ======================
# Extract ONLY generated tokens
# ======================

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, outputs)
]

vgdl_only = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0].strip()

# ======================
# Simple format cleanup
# ======================

lines = [line.rstrip() for line in vgdl_only.splitlines() if line.strip()]

# keep only from BasicGame onward
for i, line in enumerate(lines):
    if line.startswith("BasicGame"):
        lines = lines[i:]
        break

vgdl_only = "\n".join(lines).strip()

# ======================
# Save to TXT
# ======================

output_file = "mario_generated_vgdl_qwen.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(vgdl_only)

# ======================
# Print result
# ======================

print("\nGenerated VGDL:\n")
print(vgdl_only)

print(f"\nVGDL salvato in {output_file}")