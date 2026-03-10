import transformers
import torch

# ======================
# Model Loading
# ======================

model_id = "sapienzanlp/Minerva-7B-instruct-v1.0"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
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
You MUST follow the exact output format below.

The output MUST start with "Player role:" and must contain exactly the following sections:

Player role:
Entities:
Interactions:
Win condition:
Lose condition:
Objective:

Do not write any introductory sentence.
Do not summarize the game.
Do not write anything outside these sections.

VGDL:
{vgdl_code}

Description:
"""

# ======================
# Chat messages
# ======================

messages = [
    {"role": "user", "content": prompt}
]

# ======================
# Generation
# ======================

outputs = pipe(
    messages,
    max_new_tokens=200,
    temperature=0.0,
    do_sample=False,
)

generated_text = outputs[0]["generated_text"][-1]["content"]

# ======================
# Post-processing
# ======================

start = generated_text.find("Player role:")

if start != -1:
    generated_text = generated_text[start:]

end = generated_text.find("\nVGDL:")

if end != -1:
    generated_text = generated_text[:end]

description_only = generated_text.strip()

# ======================
# Save to TXT
# ======================

output_file = "game_description_minerva.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(description_only)

# ======================
# Print result
# ======================

print("\nGenerated Description:\n")
print(description_only)

print(f"\nDescrizione salvata in {output_file}")