import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# Model Loading
# ======================

model_name = "EleutherAI/gpt-neo-2.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-Neo often has no pad token defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

model.to(device)
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
# Prompt Construction
# ======================

prompt = f"""Convert the following video game description into VGDL code.

Rules:
- Output only VGDL code.
- The first line must be exactly: BasicGame
- Use exactly these sections in this order:
  SpriteSet
  LevelMapping
  InteractionSet
  TerminationSet
- Use only entities, mechanics, and win/lose conditions explicitly stated or conservatively inferable.
- Do not add explanations, comments, markdown, or extra text.
- Use EOS for edge-of-screen interactions.
- Keep the VGDL minimal, consistent, and syntactically clean.

Allowed sprite classes:
MovingAvatar, HorizontalAvatar, VerticalAvatar, OrientedAvatar, FlakAvatar, ShootAvatar, AimedFlakAvatar, InertialAvatar, MarioAvatar, Immovable, Passive, ResourcePack, Flicker, OrientedFlicker, Missile, Walker, WalkJumper, RandomNPC, Chaser, Fleeing, Spreader, Conveyor, Portal, SpawnPoint

Allowed interaction effects:
killSprite, cloneSprite, transformTo, stepBack, undoAll, bounceForward, conveySprite, killIfSlow, killIfFromAbove, killIfAlive, wrapAround, pullWithIt, teleportToExit, collectResource, reverseDirection, flipDirection, wallBounce, wallStop

Allowed termination conditions:
SpriteCounter, MultiSpriteCounter, Timeout

Example input:
Player role: The player controls a ShootAvatar that shoots bullets forward.
Entities:
- avatar: ShootAvatar
- bullet: Missile, child of avatar, speed=2
- enemy: Passive
- wall: Immovable
Interactions:
- avatar wall > stepBack
- bullet enemy > killSprite
- bullet wall > killSprite
- bullet EOS > killSprite
Win condition: SpriteCounter stype=enemy limit=0 win=True
Lose condition: SpriteCounter stype=avatar limit=0 win=False

Example output:
BasicGame
    SpriteSet
        avatar > ShootAvatar
            bullet > Missile speed=2
        enemy > Passive
        wall > Immovable
    LevelMapping
        E > enemy
        W > wall
    InteractionSet
        avatar wall > stepBack
        bullet enemy > killSprite
        bullet wall > killSprite
        bullet EOS > killSprite
    TerminationSet
        SpriteCounter stype=enemy limit=0 win=True
        SpriteCounter stype=avatar limit=0 win=False

Now write a NEW VGDL specification for the following game.

Game description:
{vgdl_description.strip()}

Output:
BasicGame
"""

# ======================
# Tokenization
# ======================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048
).to(device)

print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

# ======================
# Generation
# ======================

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# ======================
# Decode full output
# ======================

full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================
# Extract VGDL starting from the LAST BasicGame
# ======================

last_basicgame_idx = full_output.rfind("BasicGame")

if last_basicgame_idx != -1:
    vgdl_candidate = full_output[last_basicgame_idx:].strip()
else:
    vgdl_candidate = full_output.strip()

# ======================
# Cleanup: stop when non-VGDL text starts
# ======================

stop_prefixes = (
    "Description:",
    "Game description:",
    "Example input:",
    "Example output:",
    "Output:",
    "Rules:",
    "Convert the following",
    "Now write",
    "Player role:",
    "Entities:",
    "Interactions:",
    "Win condition:",
    "Lose condition:",
    "Objective:"
)

vgdl_lines = []
for line in vgdl_candidate.splitlines():
    stripped = line.strip()

    if stripped and any(stripped.startswith(prefix) for prefix in stop_prefixes):
        break

    vgdl_lines.append(line.rstrip())

vgdl_only = "\n".join(vgdl_lines).strip()

# ======================
# Final safety cleanup
# ======================

if not vgdl_only.startswith("BasicGame"):
    vgdl_only = "BasicGame\n" + vgdl_only.lstrip()

# remove trailing empty lines
vgdl_only = "\n".join([line.rstrip() for line in vgdl_only.splitlines() if line.strip()])

# ======================
# Save to TXT
# ======================

output_file = "mario_generated_vgdl_gptneo.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(vgdl_only)

# ======================
# Print result
# ======================

print("\nGenerated VGDL:\n")
print(vgdl_only)

print(f"\nVGDL salvato in {output_file}")