import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ======================
# Seed for reproducibility
# ======================

torch.manual_seed(0)

# ======================
# Model Loading
# ======================

model_id = "microsoft/Phi-3.5-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ======================
# VGDL Input
# ======================

vgdl_code = """
BasicGame
    SpriteSet 
        elevator > Missile orientation=UP speed=0.1 color=BLUE
        moving > physicstype=GravityPhysics
            avatar > MarioAvatar airsteering=True
            evil   >  orientation=LEFT
                goomba     > Walker     color=BROWN 
                paratroopa > WalkJumper color=RED
        goal > Immovable color=GREEN
            
    TerminationSet
        SpriteCounter stype=goal      win=True     
        SpriteCounter stype=avatar    win=False     
           
    InteractionSet
        evil avatar > killIfFromAbove scoreChange=1
        avatar evil > killIfAlive
        moving EOS  > killSprite 
        goal avatar > killSprite
        moving wall > wallStop friction=0.1
        moving elevator > pullWithIt        
        elevator EOS    > wrapAround
        
    LevelMapping
        G > goal
        1 > goomba
        2 > paratroopa
        = > elevator
"""

# ======================
# Prompt
# ======================

prompt = f"""
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

DESCRIPTION:
"""

# ======================
# Chat Messages
# ======================

messages = [
    {"role": "system", "content": "You are an expert VGDL analyst."},
    {"role": "user", "content": prompt},
]

# ======================
# Generation settings
# ======================

generation_args = {
    "max_new_tokens": 300,
    "return_full_text": False,
    "do_sample": False,
    "use_cache": False
}

# ======================
# Generate
# ======================

output = pipe(messages, **generation_args)

generated_text = output[0]["generated_text"]

# ======================
# Post-processing
# ======================

# start from Player role
start = generated_text.find("Player role:")

if start != -1:
    generated_text = generated_text[start:]

# stop if VGDL appears again
end = generated_text.find("\nVGDL:")

if end != -1:
    generated_text = generated_text[:end]

description_only = generated_text.strip()

# ======================
# Save to TXT
# ======================

output_file = "artillery_description_phi.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(description_only)

# ======================
# Print result
# ======================

print("\nGenerated Description:\n")
print(description_only)

print(f"\nDescrizione salvata in {output_file}")